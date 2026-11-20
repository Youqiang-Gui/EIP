"""Standalone WAV inference loop with synchronized response playback."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.audiointeraction.dataset.TOKENS import KEEP_SILENCE, TEXT_BEGIN
from src.audiointeraction.hri_mot.motion_smoothing import stitch_action_chunk_c2
from src.audiointeraction.hri_mot.realtime_profile import RealtimeProfiler
from src.audiointeraction.hri_mot.streaming_language import (
    apply_reply_silence_gate,
)
from src.audiointeraction.hri_mot.streaming_response import StreamingResponseController
from src.audiointeraction.hri_mot.streaming_tts import SynthesizedUtterance, TTSClient


@torch.no_grad()
def infer_wav_impl(
    *,
    audio_chunks: List[np.ndarray],
    audio_frontend_modes: List[str],
    real_chunk_count: int,
    initial_state: torch.Tensor,
    wrapper: torch.nn.Module,
    tokenizer: Any,
    normalizer: Any,
    audio_encoder: Optional[torch.nn.Module],
    data_config: Any,
    max_total_seq_len: int,
    args: Any,
    device: torch.device,
    language_generator: torch.Generator,
    action_generator: torch.Generator,
    tts_client: TTSClient,
    session_class: type,
    sample_action_fn: Callable[..., torch.Tensor],
    encode_audio_feature_batch_fn: Callable[..., torch.Tensor],
    control_name_fn: Callable[[int], str],
    emotion_token_to_text: Dict[int, str],
    control_probability_fn: Callable[[Dict[str, Any], int], Optional[float]],
    format_probability_fn: Callable[[Optional[float]], str],
    realtime_profile: bool = False,
    onset_probe: Optional[Any] = None,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], List[SynthesizedUtterance]]:
    if not audio_chunks:
        raise ValueError("No audio chunks to process")
    if len(audio_frontend_modes) != len(audio_chunks):
        raise ValueError(
            f"audio_frontend_modes has {len(audio_frontend_modes)} entries "
            f"for {len(audio_chunks)} chunks"
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
        language_sampling_mode=str(
            getattr(args, "language_sampling_mode", "web_legacy")
        ),
        debug_language_sampling=bool(
            getattr(args, "debug_language_sampling", False)
        ),
    )
    session.emotion_override = str(
        getattr(args, "emotion_override", "auto")
    )
    profiler = (
        RealtimeProfiler(
            chunk_size_sec=float(data_config.chunk_size_sec),
            device=device,
        )
        if realtime_profile
        else None
    )
    response_controller = StreamingResponseController(
        tts_client=tts_client,
        chunk_size_sec=float(data_config.chunk_size_sec),
        language=str(args.tts_language),
        speaker=str(args.tts_speaker),
        instruct=str(args.tts_instruct),
        emotion_token_to_text=emotion_token_to_text,
        # Preserve AIM's exact generated-token history in every action mode.
        commit_emotion_token_to_action_context=True,
        measure_stage=profiler.stage if profiler is not None else None,
    )
    predictions: List[torch.Tensor] = []
    rollout_chunks: List[torch.Tensor] = []
    records: List[Dict[str, Any]] = []
    eof_flush_sec = float(args.audio_eof_flush_sec)
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
    response_armed = all(mode == "real/unmodified" for mode in audio_frontend_modes)
    sending = response_armed
    silence_run_frames = 0
    first_response_started = False
    barge_in_waiting_for_silence = False
    reply_pending = False
    input_chunk_count = len(audio_chunks)
    chunk_idx = 0

    # After the WAV ends, preserve the WebUI's VAD clock with exact-zero audio.
    # These are normal physical chunks: they generate actions, can release a
    # deferred response, and continue through synchronized TTS playback.
    while (
        chunk_idx < input_chunk_count
        or response_controller.active
        or (
            sending
            and response_armed
            and silence_run_frames < max_silence_frames
        )
    ):
        if profiler is not None:
            profiler.begin_chunk(chunk_idx)

        if chunk_idx < input_chunk_count:
            pcm_np = audio_chunks[chunk_idx]
            audio_frontend_mode = str(audio_frontend_modes[chunk_idx])
            kind = "wav" if chunk_idx < real_chunk_count else "tail_silence"
        else:
            pcm_np = np.zeros(
                int(round(data_config.chunk_size_sec * data_config.audio_sample_rate)),
                dtype=np.float32,
            )
            if response_controller.active:
                audio_frontend_mode = "zero/tts_playback"
                kind = "tts_playback_tail"
            else:
                audio_frontend_mode = "zero/end-of-file"
                kind = "eof_action_flush"

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

        # Training keeps the listening block that immediately precedes
        # TEXT_BEGIN, but inserts no synthetic partner-audio blocks once the
        # assistant text turn is open.  A planned response is active one chunk
        # before its onset is committed, so ``response_was_active`` alone is
        # not sufficient to decide whether audio still belongs in context.
        response_was_active = response_controller.active
        assistant_text_open = bool(
            response_was_active and response_controller.current_response_started
        )
        barge_in_triggered = (
            str(args.barge_in_mode) == "immediate"
            and response_was_active
            and audio_frontend_mode == "real/activity"
        )
        barge_in_pcm_history_reset = False
        if barge_in_triggered:
            # The interrupt onset starts a new partner utterance.  PCM cached
            # while the assistant was speaking belongs to the old playback
            # interval (and is normally zeroed by the frontend), so carrying
            # it into the streaming audio encoder gives the new utterance a
            # different acoustic prefix from web/infer_wav.py.  Preserve the
            # language KV/actually spoken text, but restart acoustic history
            # before encoding this same physical barge-in chunk.
            session.pcm_history.clear()
            barge_in_pcm_history_reset = True
        # A barge-in chunk belongs to the new partner turn, not to the open
        # assistant turn.  Defer its listening block until the response
        # controller has first committed/cancelled the assistant boundary.
        listening_block_inserted = not assistant_text_open and not barge_in_triggered
        barge_in_audio_block_inserted = False

        pcm_chunk = torch.from_numpy(np.asarray(pcm_np, dtype=np.float32)).float()
        audio_feature = None
        pcm_cached_without_context = False
        audio_feature_stage = (
            profiler.stage("audio_feature")
            if profiler is not None
            else nullcontext()
        )
        with audio_feature_stage:
            if audio_encoder is not None:
                if not listening_block_inserted:
                    if args.audio_feature_mode == "streaming_latest":
                        if barge_in_triggered:
                            # Encode the interrupt onset now, but insert its
                            # listening block only after TEXT_END below.
                            audio_feature = session.encode_streaming_audio_feature(
                                audio_encoder=audio_encoder,
                                pcm_chunk=pcm_chunk,
                                dtype=dtype,
                            )
                        else:
                            # Match the original streaming frontend clock:
                            # retain PCM during uninterrupted robot speech so
                            # resumed listening has continuous acoustic
                            # history, without forwarding a listening block.
                            pcm = (
                                pcm_chunk.detach()
                                .float()
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                                .reshape(-1)
                            )
                            session.pcm_history.append(pcm)
                            pcm_cached_without_context = True
                    elif args.audio_feature_mode == "independent_first":
                        if barge_in_triggered:
                            audio_feature = encode_audio_feature_batch_fn(
                                pcm_chunk.to(device=device).unsqueeze(0),
                                audio_encoder,
                                device,
                                dtype,
                            )[0]
                    else:
                        raise ValueError(
                            f"Unknown audio_feature_mode: {args.audio_feature_mode}"
                        )
                elif args.audio_feature_mode == "streaming_latest":
                    audio_feature = session.encode_streaming_audio_feature(
                        audio_encoder=audio_encoder,
                        pcm_chunk=pcm_chunk,
                        dtype=dtype,
                    )
                elif args.audio_feature_mode == "independent_first":
                    audio_feature = encode_audio_feature_batch_fn(
                        pcm_chunk.to(device=device).unsqueeze(0),
                        audio_encoder,
                        device,
                        dtype,
                    )[0]
                else:
                    raise ValueError(f"Unknown audio_feature_mode: {args.audio_feature_mode}")
        can_sample_response = response_armed and not response_was_active
        control_stage = (
            profiler.stage("control_sampling")
            if profiler is not None
            else nullcontext()
        )
        with control_stage:
            if listening_block_inserted:
                sampled_control_token = session.step_audio_chunk(
                    chunk_idx=chunk_idx,
                    audio_feature=audio_feature,
                    allow_text_begin=can_sample_response,
                )
                control_debug = dict(session.last_sample_debug)
                sampled_control = control_name_fn(sampled_control_token)
            else:
                # The response controller will commit this interval's aligned
                # text fragment directly.  Do not manufacture a listening
                # block or a KEEP_SILENCE token inside TEXT_BEGIN..TEXT_END.
                sampled_control_token = KEEP_SILENCE
                control_debug = {}
                sampled_control = "not_sampled_assistant_text_open"
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
        text_prob = control_probability_fn(control_debug, TEXT_BEGIN)
        if can_sample_response:
            reply_gate = apply_reply_silence_gate(
                session,
                sampled_control_token=sampled_control_token,
                reply_pending=reply_pending,
                reply_allowed=reply_allowed,
                text_begin_probability=text_prob,
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
        response_stage = (
            profiler.stage("response_controller")
            if profiler is not None
            else nullcontext()
        )
        with response_stage:
            response_step = response_controller.advance_before_action(
                session=session,
                chunk_idx=chunk_idx,
                control_token=control_token,
                response_was_active=response_was_active,
                max_response_tokens=args.max_response_tokens,
                interrupt_response=barge_in_triggered,
            )
        realtime_response_ready_t = (
            profiler.mark("response_ready")
            if profiler is not None
            else None
        )
        committed = response_step.committed_fragment
        planned = response_step.planned_response
        if response_step.interrupted:
            # The controller has now closed (or cancelled) the assistant turn.
            # Insert the same physical 400 ms partner-audio chunk immediately,
            # so a short interruption or sharp sound onset is not delayed until
            # the next chunk and reduced to only streaming PCM history.
            sampled_control_token = session.step_audio_chunk(
                chunk_idx=chunk_idx,
                audio_feature=audio_feature,
                allow_text_begin=False,
            )
            if int(sampled_control_token) != int(KEEP_SILENCE):
                raise AssertionError(
                    "Barge-in listening block must schedule KEEP_SILENCE"
                )
            control_debug = dict(session.last_sample_debug)
            sampled_control = control_name_fn(sampled_control_token)
            listening_block_inserted = True
            barge_in_audio_block_inserted = True
            pcm_cached_without_context = False
            barge_in_waiting_for_silence = True
            reply_pending = False
            print(
                f"[barge-in] chunk={chunk_idx} mode={audio_frontend_mode} "
                f"after_onset={response_step.interrupted_after_onset}",
                flush=True,
            )
        if planned is not None:
            first_response_started = True
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
            state_for_action = rollout_chunks[-1].detach()
            state_source = "rollout_previous_c2_stitched_action"
        else:
            state_for_action = initial_state
            state_source = "initial_constant_state"

        action_stage = (
            profiler.stage("action_sampling")
            if profiler is not None
            else nullcontext()
        )
        probe_rng_state = action_generator.get_state().clone() if onset_probe is not None else None
        with action_stage:
            pred_action = sample_action_fn(
                session=session,
                robot_state=state_for_action,
                chunk_idx=chunk_idx,
                action_horizon=data_config.action_horizon,
                action_dim=data_config.action_dim,
                sampling_steps=args.sampling_steps,
                solver="euler",
                initial_noise=args.initial_noise,
                device=device,
                action_generator=action_generator,
                teacher_forced_speak_label=planned_next_speak_label,
                response_elapsed_chunk=planned_response_elapsed_chunk,
                emotion_token_id=planned_emotion_token_id,
            )
        realtime_reply_action_ready_t = None
        if profiler is not None:
            profiler.mark("action_ready")
        action_context_length = int(session.kv_context_length)
        action_speak_debug = dict(session.last_action_speak_debug)
        predictions.append(pred_action)
        motion_stage = (
            profiler.stage("motion_stitch")
            if profiler is not None
            else nullcontext()
        )
        with motion_stage:
            if rollout_chunks:
                rollout_action = stitch_action_chunk_c2(
                    rollout_chunks[-1],
                    pred_action,
                    bridge_frames=int(args.c2_bridge_frames),
                )
            else:
                rollout_action = pred_action.clone()
        if profiler is not None and planned is not None:
            realtime_reply_action_ready_t = profiler.mark("reply_action_ready")
        if onset_probe is not None:
            onset_probe.step(
                session=session, sample_action_fn=sample_action_fn,
                source_state=state_for_action,
                previous_action=rollout_chunks[-1] if rollout_chunks else None,
                action_rng_state=probe_rng_state, chunk_idx=chunk_idx,
                speak_label=planned_next_speak_label,
                elapsed_chunk=planned_response_elapsed_chunk,
                emotion_token_id=planned_emotion_token_id,
                sampling_steps=args.sampling_steps, initial_noise=args.initial_noise,
                main_raw=pred_action, main_c2=rollout_action,
                interrupted=response_step.interrupted,
            )
        rollout_chunks.append(rollout_action)

        realtime_timing = (
            profiler.finish_chunk()
            if profiler is not None
            else None
        )

        audio_t_start = float(chunk_idx * data_config.chunk_size_sec)
        audio_t_end = float(audio_t_start + data_config.chunk_size_sec)
        keep_prob = control_probability_fn(control_debug, KEEP_SILENCE)
        print(
            f"[chunk {chunk_idx:03d}] {kind} {audio_frontend_mode} "
            f"t={audio_t_start:.1f}-{audio_t_end:.1f}s "
            f"listening_block={listening_block_inserted} "
            f"control={response_step.control} "
            f"p_keep={format_probability_fn(keep_prob)} "
            f"p_text_begin={format_probability_fn(text_prob)}",
            flush=True,
        )
        if planned is not None:
            print(f"[response-plan] {planned.display_text}", flush=True)
        if committed is not None:
            print(
                f"[response-fragment] {committed.text!r} "
                f"end={committed.end} remaining={response_step.playback_remaining_chunks}",
                flush=True,
            )

        records.append(
            {
                "chunk_idx": int(chunk_idx),
                "audio_kind": kind,
                "audio_frontend_mode": audio_frontend_mode,
                "assistant_text_open_before_chunk": bool(assistant_text_open),
                "listening_block_inserted": bool(listening_block_inserted),
                "audio_feature_forwarded": bool(audio_feature is not None),
                "pcm_cached_without_context": bool(pcm_cached_without_context),
                "barge_in_audio_block_inserted": bool(
                    barge_in_audio_block_inserted
                ),
                "barge_in_pcm_history_reset": bool(
                    barge_in_pcm_history_reset
                ),
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
                "audio_t_start": audio_t_start,
                "audio_t_end": audio_t_end,
                "target_t_start": audio_t_end,
                "target_t_end": float(audio_t_end + data_config.chunk_size_sec),
                "control": response_step.control,
                "sampled_control": sampled_control,
                "keep_silence_probability": keep_prob,
                "text_begin_probability": text_prob,
                "control_debug": control_debug,
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
                "state_source": state_source,
                "action_solver": "euler",
                "action_sampling_steps": int(args.sampling_steps),
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
                "realtime": realtime_timing,
                "realtime_tts_playback_t_start": (
                    float(realtime_response_ready_t)
                    if planned is not None
                    and realtime_response_ready_t is not None
                    else None
                ),
                "realtime_reply_action_ready_t": (
                    float(realtime_reply_action_ready_t)
                    if realtime_reply_action_ready_t is not None
                    else None
                ),
                "realtime_tts_playback_t_stop": (
                    float(realtime_response_ready_t)
                    if response_step.interrupted
                    and realtime_response_ready_t is not None
                    else None
                ),
            }
        )
        chunk_idx += 1
        if (
            args.stop_after_first_response
            and first_response_started
            and not response_controller.active
        ):
            break

    if onset_probe is not None:
        onset_probe.close()
    try:
        session.wrapper.clear_context_kv_cache()
    except Exception:
        pass
    return (
        torch.stack(predictions, dim=0),
        records,
        list(response_controller.utterances),
    )
