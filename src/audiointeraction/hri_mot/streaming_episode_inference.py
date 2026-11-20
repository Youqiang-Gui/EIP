"""Shared whole-episode streaming loop used by both HRI inference scripts."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.audiointeraction.dataset.TOKENS import KEEP_SILENCE, TEXT_BEGIN
from src.audiointeraction.hri_mot.motion_smoothing import stitch_action_chunk_c2
from src.audiointeraction.hri_mot.streaming_language import (
    apply_reply_silence_gate,
)
from src.audiointeraction.hri_mot.streaming_response import StreamingResponseController
from src.audiointeraction.hri_mot.streaming_tts import SynthesizedUtterance, TTSClient


@torch.no_grad()
def infer_episode_streaming_impl(
    *,
    sample: Dict[str, Any],
    wrapper: torch.nn.Module,
    tokenizer: Any,
    normalizer: Any,
    audio_encoder: Optional[torch.nn.Module],
    data_config: Any,
    num_chunks: int,
    max_total_seq_len: int,
    args: Any,
    device: torch.device,
    audio_frontend_modes: Sequence[str],
    language_generator: torch.Generator,
    action_generator: torch.Generator,
    tts_client: TTSClient,
    session_class: type,
    sample_action_fn: Callable[..., torch.Tensor],
    encode_audio_feature_batch_fn: Callable[..., torch.Tensor],
    control_name_fn: Callable[[int], str],
    emotion_token_to_text: Dict[int, str],
) -> Tuple[torch.Tensor, List[Dict[str, Any]], List[SynthesizedUtterance]]:
    """Run causal action inference with one-interval-delayed text commitment."""

    available_chunks = int(sample["target_action"].shape[0])
    num_chunks = min(int(num_chunks), available_chunks)
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if len(audio_frontend_modes) < num_chunks:
        raise ValueError(
            f"audio_frontend_modes has {len(audio_frontend_modes)} entries "
            f"for {num_chunks} chunks"
        )

    dtype = next(wrapper.base_model.parameters()).dtype
    session = session_class(
        wrapper=wrapper,
        tokenizer=tokenizer,
        normalizer=normalizer,
        data_config=data_config,
        device=device,
        max_total_seq_len=max_total_seq_len,
        text_temperature=args.text_temperature,
        text_top_k=args.text_top_k,
        text_top_p=args.text_top_p,
        language_generator=language_generator,
    )
    session.emotion_override = str(
        getattr(args, "emotion_override", "auto")
    )
    response_controller = StreamingResponseController(
        tts_client=tts_client,
        chunk_size_sec=float(data_config.chunk_size_sec),
        language=str(args.tts_language),
        speaker=str(args.tts_speaker),
        instruct=str(args.tts_instruct),
        emotion_token_to_text=emotion_token_to_text,
        # The semantic emotion token is part of AIM's generated response and
        # must remain in the shared causal KV history for checkpoint-invariant
        # language generation. Emotion is additionally supplied to
        # the Action Expert through its direct semantic emotion condition.
        commit_emotion_token_to_action_context=True,
    )
    predicted_chunks: List[torch.Tensor] = []
    rollout_chunks: List[torch.Tensor] = []
    records: List[Dict[str, Any]] = []
    action_horizon = int(sample["target_action"].shape[1])
    action_dim = int(sample["target_action"].shape[2])

    eof_flush_sec = float(getattr(args, "audio_eof_flush_sec", 0.0))
    if eof_flush_sec < 0.0:
        raise ValueError(f"audio_eof_flush_sec must be >= 0, got {eof_flush_sec}")
    max_silence_frames = int(
        math.ceil(eof_flush_sec / float(data_config.chunk_size_sec) - 1e-12)
    )
    min_reply_silence_sec = float(args.min_reply_silence_sec)
    if min_reply_silence_sec < 0.0:
        raise ValueError(
            "min_reply_silence_sec must be >= 0, "
            f"got {min_reply_silence_sec}"
        )
    min_reply_silence_frames = int(
        math.ceil(
            min_reply_silence_sec / float(data_config.chunk_size_sec) - 1e-12
        )
    )
    deferred_reply_prob_threshold = float(
        getattr(args, "deferred_reply_prob_threshold", 0.5)
    )
    response_armed = all(
        mode == "real/unmodified" for mode in audio_frontend_modes[:num_chunks]
    )
    sending = response_armed
    silence_run_frames = 0
    barge_in_waiting_for_silence = False
    reply_pending = False
    samples_per_chunk = int(
        round(data_config.chunk_size_sec * data_config.audio_sample_rate)
    )
    zero_audio_chunk = torch.zeros(samples_per_chunk, dtype=torch.float32)
    input_chunk_count = int(num_chunks)
    episode_audio_end = (
        float(sample["t_start"][input_chunk_count - 1].item())
        + float(data_config.chunk_size_sec)
    )
    chunk_idx = 0

    while (
        chunk_idx < input_chunk_count
        or response_controller.active
        or (
            sending
            and response_armed
            and silence_run_frames < max_silence_frames
        )
    ):
        if chunk_idx < input_chunk_count:
            audio_t_start = float(sample["t_start"][chunk_idx].item())
            audio_frontend_mode = str(audio_frontend_modes[chunk_idx])
            chunk_audio = sample["partner_audio"][chunk_idx]
            audio_kind = "episode"
        else:
            audio_t_start = float(
                episode_audio_end
                + (chunk_idx - input_chunk_count) * data_config.chunk_size_sec
            )
            chunk_audio = zero_audio_chunk
            if response_controller.active:
                audio_frontend_mode = "zero/tts_playback"
                audio_kind = "tts_playback_tail"
            else:
                audio_frontend_mode = "zero/end-of-file"
                audio_kind = "eof_action_flush"
        audio_t_end = audio_t_start + float(data_config.chunk_size_sec)
        target_t_start = audio_t_end
        target_t_end = target_t_start + float(data_config.chunk_size_sec)
        if audio_frontend_mode == "real/activity":
            response_armed = True
            sending = True
            silence_run_frames = 0
        elif audio_frontend_mode == "real/unmodified":
            sending = True
            silence_run_frames = 0
        elif sending:
            silence_run_frames += 1
            if silence_run_frames > max_silence_frames:
                sending = False
        if (
            barge_in_waiting_for_silence
            and silence_run_frames >= min_reply_silence_frames
            and audio_frontend_mode != "real/activity"
        ):
            barge_in_waiting_for_silence = False

        audio_feature = None
        if audio_encoder is not None:
            if args.audio_feature_mode == "streaming_latest":
                audio_feature = session.encode_streaming_audio_feature(
                    audio_encoder=audio_encoder,
                    pcm_chunk=chunk_audio,
                    dtype=dtype,
                )
            elif args.audio_feature_mode == "independent_first":
                audio_feature = encode_audio_feature_batch_fn(
                    chunk_audio.to(device=device).unsqueeze(0),
                    audio_encoder,
                    device,
                    dtype,
                )[0]
            else:
                raise ValueError(f"Unknown audio_feature_mode: {args.audio_feature_mode}")

        response_was_active = response_controller.active
        barge_in_triggered = (
            str(args.barge_in_mode) == "immediate"
            and response_was_active
            and audio_frontend_mode == "real/activity"
        )
        can_sample_response = response_armed and not response_was_active
        sampled_control_token = session.step_audio_chunk(
            chunk_idx=chunk_idx,
            audio_feature=audio_feature,
            allow_text_begin=can_sample_response,
        )
        control_debug = dict(session.last_sample_debug)
        sampled_control = control_name_fn(sampled_control_token)
        reply_allowed = bool(
            can_sample_response
            and (
                audio_frontend_mode == "real/unmodified"
                or (
                    audio_frontend_mode != "real/activity"
                    and silence_run_frames >= min_reply_silence_frames
                )
            )
        )
        reply_deferred = False
        reply_pending_released = False
        reply_probability_deferred = False
        control_token = int(sampled_control_token)
        text_prob_value = control_debug.get("allowed_probs", {}).get(
            str(int(TEXT_BEGIN))
        )
        text_begin_probability = (
            None if text_prob_value is None else float(text_prob_value)
        )
        if can_sample_response:
            reply_gate = apply_reply_silence_gate(
                session,
                sampled_control_token=sampled_control_token,
                reply_pending=reply_pending,
                reply_allowed=reply_allowed,
                text_begin_probability=text_begin_probability,
                deferred_probability_threshold=deferred_reply_prob_threshold,
                allow_probability_defer=barge_in_waiting_for_silence,
            )
            control_token = int(reply_gate.control_token)
            reply_pending = bool(reply_gate.reply_pending)
            reply_deferred = bool(reply_gate.reply_deferred)
            reply_pending_released = bool(reply_gate.reply_pending_released)
            reply_probability_deferred = bool(
                reply_gate.reply_probability_deferred
            )
        response_step = response_controller.advance_before_action(
            session=session,
            chunk_idx=chunk_idx,
            control_token=control_token,
            response_was_active=response_was_active,
            max_response_tokens=args.max_response_tokens,
            interrupt_response=barge_in_triggered,
        )
        committed = response_step.committed_fragment
        planned = response_step.planned_response
        if response_step.interrupted:
            barge_in_waiting_for_silence = True
            reply_pending = False
        if planned is not None:
            reply_pending = False
            response_armed = False
            sending = False
            silence_run_frames = 0
            session.pcm_history.clear()
        planned_next_speak_label = response_controller.action_speak_label()
        planned_response_elapsed_chunk = (
            response_controller.action_response_elapsed_chunk()
        )
        planned_emotion_token_id = (
            response_controller.action_emotion_token_id()
            if bool(
                getattr(wrapper, "use_emotion_conditioning", False)
            )
            else None
        )

        if args.state_source == "rollout" and rollout_chunks:
            robot_state_for_action = rollout_chunks[-1].detach()
            state_source = "rollout_previous_c2_stitched_action"
        elif args.state_source == "dataset":
            if chunk_idx < input_chunk_count:
                robot_state_for_action = sample["robot_state"][chunk_idx]
                state_source = "dataset_ground_truth"
            else:
                robot_state_for_action = sample["robot_state"][
                    input_chunk_count - 1
                ]
                state_source = "dataset_last_state_eof_extension"
        elif args.state_source == "rollout":
            robot_state_for_action = sample["robot_state"][
                min(chunk_idx, input_chunk_count - 1)
            ]
            state_source = "dataset_initial_state"
        else:
            raise ValueError(f"Unknown state_source: {args.state_source}")

        pred_action = sample_action_fn(
            session=session,
            robot_state=robot_state_for_action,
            chunk_idx=chunk_idx,
            action_horizon=action_horizon,
            action_dim=action_dim,
            sampling_steps=args.sampling_steps,
            solver=args.solver,
            initial_noise=args.initial_noise,
            device=device,
            action_generator=action_generator,
            teacher_forced_speak_label=planned_next_speak_label,
            response_elapsed_chunk=planned_response_elapsed_chunk,
            emotion_token_id=planned_emotion_token_id,
        )
        action_context_length = int(session.kv_context_length)
        action_speak_debug = dict(session.last_action_speak_debug)
        predicted_chunks.append(pred_action)
        if rollout_chunks:
            rollout_action = stitch_action_chunk_c2(
                rollout_chunks[-1],
                pred_action,
                bridge_frames=int(args.c2_bridge_frames),
            )
        else:
            rollout_action = pred_action.clone()
        rollout_chunks.append(rollout_action)


        records.append(
            {
                "chunk_idx": int(chunk_idx),
                "decision_chunk_idx": int(chunk_idx),
                "target_chunk_idx": int(chunk_idx + 1),
                "audio_t_start": float(audio_t_start),
                "audio_t_end": float(audio_t_end),
                "audio_kind": audio_kind,
                "audio_frontend_mode": audio_frontend_mode,
                "response_armed": bool(response_armed),
                "barge_in_waiting_for_silence": bool(
                    barge_in_waiting_for_silence
                ),
                "reply_allowed": bool(reply_allowed),
                "reply_pending": bool(reply_pending),
                "reply_deferred": bool(reply_deferred),
                "reply_pending_released": bool(reply_pending_released),
                "reply_probability_deferred": bool(reply_probability_deferred),
                "deferred_reply_prob_threshold": float(
                    deferred_reply_prob_threshold
                ),
                "silence_run_frames": int(silence_run_frames),
                "min_reply_silence_frames": int(min_reply_silence_frames),
                "response_interrupted": bool(response_step.interrupted),
                "response_interrupted_after_onset": bool(
                    response_step.interrupted_after_onset
                ),
                "target_t_start": float(target_t_start),
                "target_t_end": float(target_t_end),
                "control": response_step.control,
                "sampled_control": sampled_control,
                "tokens": (
                    list(committed.token_ids)
                    if committed is not None
                    else [int(KEEP_SILENCE)]
                ),
                "text": committed.text if committed is not None else "",
                "planned_full_text": planned.display_text if planned is not None else "",
                "planned_full_tokens": (
                    list(planned.generated_token_ids) if planned is not None else []
                ),
                "text_completed": bool(committed.end) if committed is not None else False,
                "tts_utterance_index": response_step.utterance_index,
                "playback_chunk_idx": int(chunk_idx) if committed is not None else None,
                "playback_t_start": float(audio_t_start) if committed is not None else None,
                "playback_t_end": float(audio_t_end) if committed is not None else None,
                "planned_response_target_chunk_idx": (
                    int(chunk_idx + 1) if planned is not None else None
                ),
                "playback_remaining_chunks": response_step.playback_remaining_chunks,
                "playback_pcm_start_sample": (
                    committed.pcm_start_sample if committed is not None else None
                ),
                "playback_pcm_end_sample": (
                    committed.pcm_end_sample if committed is not None else None
                ),
                "playback_sample_rate": (
                    committed.sample_rate if committed is not None else None
                ),
                "control_debug": control_debug,
                "state_source": state_source,
                "action_generated": True,
                "language_only_flush": False,
                "action_context_text_cutoff": "through_current_physical_interval_I_c",
                "kv_context_length_at_action": action_context_length,
                "kv_context_length_after_playback_commit": int(
                    session.kv_context_length
                ),
                "speak_conditioning": action_speak_debug,
                "action_conditioning": action_speak_debug,
                "planned_next_speak_label": int(planned_next_speak_label),
                "planned_next_emotion_token_id": planned_emotion_token_id,
                "emotion_token_committed_to_action_kv": bool(
                    committed is not None
                    and any(
                        int(token_id) in emotion_token_to_text
                        for token_id in committed.token_ids
                    )
                ),
                "planned_response_elapsed_chunk": int(
                    planned_response_elapsed_chunk
                ),
            }
        )
        chunk_idx += 1

    try:
        session.wrapper.clear_context_kv_cache()
    except Exception:
        pass
    return (
        torch.stack(predicted_chunks, dim=0),
        records,
        list(response_controller.utterances),
    )
