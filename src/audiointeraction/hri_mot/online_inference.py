"""Incremental 400 ms HRI-MoT inference for live microphone clients.

The offline WAV entry point owns the same state machine, but consumes a finite
list of chunks.  This module keeps that state alive between HTTP requests so a
browser can submit one physical audio interval at a time.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from src.audiointeraction.dataset.TOKENS import KEEP_SILENCE, TEXT_BEGIN
from src.audiointeraction.hri_mot.motion_smoothing import stitch_action_chunk_c2
from src.audiointeraction.hri_mot.streaming_language import apply_reply_silence_gate
from src.audiointeraction.hri_mot.streaming_response import StreamingResponseController
from src.audiointeraction.hri_mot.streaming_tts import SynthesizedUtterance, TTSClient


LIVE_AUDIO_MODES = frozenset(
    {
        "real/activity",
        "real/tail",
        "real/unmodified",
        "zero/silence",
        "zero/pre_activity",
        "zero/tts_playback",
    }
)


@dataclass(frozen=True)
class OnlineStepResult:
    """Observable result of one physical 400 ms interaction interval."""

    chunk_idx: int
    action: np.ndarray
    frontend_mode: str
    control: str
    sampled_control: str
    planned_text: str
    committed_text: str
    utterance: Optional[SynthesizedUtterance]
    response_active: bool
    response_ended: bool
    interrupted: bool
    action_speak_label: int
    keep_silence_probability: Optional[float]
    text_begin_probability: Optional[float]
    compute_sec: float


class OnlineHriMotEngine:
    """Stateful online equivalent of ``infer_wav_impl``.

    ``process_chunk`` is deliberately synchronous.  Callers must serialize
    requests: a second physical interval cannot safely mutate the language KV
    cache or closed-loop robot state before the first one finishes.
    """

    def __init__(
        self,
        *,
        wrapper: torch.nn.Module,
        tokenizer: Any,
        normalizer: Any,
        audio_encoder: Optional[torch.nn.Module],
        data_config: Any,
        max_total_seq_len: int,
        initial_state: torch.Tensor,
        device: torch.device,
        language_generator: torch.Generator,
        action_generator: torch.Generator,
        tts_client: TTSClient,
        args: Any,
        session_class: type,
        sample_action_fn: Callable[..., torch.Tensor],
        encode_audio_feature_batch_fn: Callable[..., torch.Tensor],
        control_name_fn: Callable[[int], str],
        emotion_token_to_text: Dict[int, str],
        control_probability_fn: Callable[[Dict[str, Any], int], Optional[float]],
    ) -> None:
        self.wrapper = wrapper
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.audio_encoder = audio_encoder
        self.data_config = data_config
        self.max_total_seq_len = int(max_total_seq_len)
        self.initial_state = initial_state.detach().clone()
        self.device = device
        self.language_generator = language_generator
        self.action_generator = action_generator
        self.tts_client = tts_client
        self.args = args
        self.session_class = session_class
        self.sample_action_fn = sample_action_fn
        self.encode_audio_feature_batch_fn = encode_audio_feature_batch_fn
        self.control_name_fn = control_name_fn
        self.emotion_token_to_text = dict(emotion_token_to_text)
        self.control_probability_fn = control_probability_fn

        if float(args.min_reply_silence_sec) < 0.0:
            raise ValueError("min_reply_silence_sec must be non-negative")
        self.min_reply_silence_frames = int(
            math.ceil(
                float(args.min_reply_silence_sec)
                / float(data_config.chunk_size_sec)
                - 1e-12
            )
        )
        self.dtype = next(wrapper.base_model.parameters()).dtype
        self.chunk_idx = 0
        self.rollout_action: Optional[torch.Tensor] = None
        self._new_context()

    @property
    def initial_action(self) -> np.ndarray:
        return self.initial_state.detach().float().cpu().numpy().copy()

    def _new_context(self) -> None:
        self.session = self.session_class(
            wrapper=self.wrapper,
            tokenizer=self.tokenizer,
            normalizer=self.normalizer,
            data_config=self.data_config,
            device=self.device,
            max_total_seq_len=self.max_total_seq_len,
            text_temperature=float(self.args.text_temperature),
            text_top_k=int(self.args.text_top_k),
            text_top_p=float(self.args.text_top_p),
            language_generator=self.language_generator,
            language_sampling_mode=str(self.args.language_sampling_mode),
            debug_language_sampling=bool(self.args.debug_language_sampling),
        )
        self.response_controller = StreamingResponseController(
            tts_client=self.tts_client,
            chunk_size_sec=float(self.data_config.chunk_size_sec),
            language=str(self.args.tts_language),
            speaker=str(self.args.tts_speaker),
            instruct=str(self.args.tts_instruct),
            emotion_token_to_text=self.emotion_token_to_text,
            # Preserve AIM's exact generated-token history in every action mode.
            commit_emotion_token_to_action_context=True,
        )
        self.response_armed = False
        self.sending = False
        self.silence_run_frames = 0
        self.barge_in_waiting_for_silence = False
        self.reply_pending = False

    def reset(self, *, preserve_rollout: bool = False) -> None:
        """Clear language/audio state and optionally retain the last robot pose."""

        try:
            self.wrapper.clear_context_kv_cache()
        except Exception:
            pass
        if not preserve_rollout:
            self.rollout_action = None
            self.chunk_idx = 0
        self._new_context()

    def _encode_audio(
        self,
        pcm_chunk: torch.Tensor,
        *,
        listening_block_inserted: bool,
        barge_in_triggered: bool,
    ) -> Optional[torch.Tensor]:
        if self.audio_encoder is None:
            return None

        mode = str(self.args.audio_feature_mode)
        if not listening_block_inserted:
            if mode == "streaming_latest":
                if barge_in_triggered:
                    return self.session.encode_streaming_audio_feature(
                        audio_encoder=self.audio_encoder,
                        pcm_chunk=pcm_chunk,
                        dtype=self.dtype,
                    )
                pcm = (
                    pcm_chunk.detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                    .reshape(-1)
                )
                self.session.pcm_history.append(pcm)
                return None
            if mode == "independent_first":
                if not barge_in_triggered:
                    return None
                return self.encode_audio_feature_batch_fn(
                    pcm_chunk.to(device=self.device).unsqueeze(0),
                    self.audio_encoder,
                    self.device,
                    self.dtype,
                )[0]
            raise ValueError(f"Unknown audio_feature_mode: {mode}")

        if mode == "streaming_latest":
            return self.session.encode_streaming_audio_feature(
                audio_encoder=self.audio_encoder,
                pcm_chunk=pcm_chunk,
                dtype=self.dtype,
            )
        if mode == "independent_first":
            return self.encode_audio_feature_batch_fn(
                pcm_chunk.to(device=self.device).unsqueeze(0),
                self.audio_encoder,
                self.device,
                self.dtype,
            )[0]
        raise ValueError(f"Unknown audio_feature_mode: {mode}")

    @torch.no_grad()
    def process_chunk(
        self,
        pcm: np.ndarray,
        *,
        frontend_mode: str,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> OnlineStepResult:
        started = time.perf_counter()

        def report_progress(stage: str) -> None:
            if progress_callback is not None:
                progress_callback(str(stage))

        frontend_mode = str(frontend_mode)
        if frontend_mode not in LIVE_AUDIO_MODES:
            raise ValueError(
                f"Unsupported frontend_mode={frontend_mode!r}; "
                f"expected one of {sorted(LIVE_AUDIO_MODES)}"
            )
        pcm_np = np.asarray(pcm, dtype=np.float32).reshape(-1)
        expected_samples = int(
            round(
                float(self.data_config.chunk_size_sec)
                * int(self.data_config.audio_sample_rate)
            )
        )
        if pcm_np.size != expected_samples:
            raise ValueError(
                f"Audio chunk must contain {expected_samples} samples, got {pcm_np.size}"
            )
        if not np.isfinite(pcm_np).all():
            raise ValueError("Audio chunk contains NaN or infinity")

        if frontend_mode == "real/activity":
            self.response_armed = True
            self.sending = True
            self.silence_run_frames = 0
        elif frontend_mode == "real/unmodified":
            self.sending = True
            self.silence_run_frames = 0
        elif self.sending:
            self.silence_run_frames += 1

        if (
            self.barge_in_waiting_for_silence
            and self.silence_run_frames >= self.min_reply_silence_frames
            and frontend_mode != "real/activity"
        ):
            self.barge_in_waiting_for_silence = False

        response_was_active = self.response_controller.active
        assistant_text_open = bool(
            response_was_active
            and self.response_controller.current_response_started
        )
        barge_in_triggered = bool(
            str(self.args.barge_in_mode) == "immediate"
            and response_was_active
            and frontend_mode == "real/activity"
        )
        if barge_in_triggered:
            self.session.pcm_history.clear()

        listening_block_inserted = not assistant_text_open and not barge_in_triggered
        pcm_chunk = torch.from_numpy(pcm_np).float()
        report_progress("audio_control")
        audio_feature = self._encode_audio(
            pcm_chunk,
            listening_block_inserted=listening_block_inserted,
            barge_in_triggered=barge_in_triggered,
        )

        can_sample_response = self.response_armed and not response_was_active
        if listening_block_inserted:
            sampled_control_token = self.session.step_audio_chunk(
                chunk_idx=self.chunk_idx,
                audio_feature=audio_feature,
                allow_text_begin=can_sample_response,
            )
            control_debug = dict(self.session.last_sample_debug)
            sampled_control = self.control_name_fn(sampled_control_token)
        else:
            sampled_control_token = KEEP_SILENCE
            control_debug = {}
            sampled_control = "not_sampled_assistant_text_open"

        reply_allowed = bool(
            can_sample_response
            and (
                frontend_mode == "real/unmodified"
                or (
                    frontend_mode != "real/activity"
                    and self.silence_run_frames >= self.min_reply_silence_frames
                )
            )
        )
        text_probability = self.control_probability_fn(
            control_debug, TEXT_BEGIN
        )
        control_token = int(sampled_control_token)
        if can_sample_response:
            reply_gate = apply_reply_silence_gate(
                self.session,
                sampled_control_token=sampled_control_token,
                reply_pending=self.reply_pending,
                reply_allowed=reply_allowed,
                text_begin_probability=text_probability,
                deferred_probability_threshold=float(
                    self.args.deferred_reply_prob_threshold
                ),
                allow_probability_defer=self.barge_in_waiting_for_silence,
            )
            control_token = int(reply_gate.control_token)
            self.reply_pending = bool(reply_gate.reply_pending)

        if int(control_token) == int(TEXT_BEGIN):
            report_progress("text_tts")
        response_step = self.response_controller.advance_before_action(
            session=self.session,
            chunk_idx=self.chunk_idx,
            control_token=control_token,
            response_was_active=response_was_active,
            max_response_tokens=int(self.args.max_response_tokens),
            interrupt_response=barge_in_triggered,
        )
        if response_step.interrupted:
            sampled_control_token = self.session.step_audio_chunk(
                chunk_idx=self.chunk_idx,
                audio_feature=audio_feature,
                allow_text_begin=False,
            )
            if int(sampled_control_token) != int(KEEP_SILENCE):
                raise AssertionError(
                    "Barge-in listening block must schedule KEEP_SILENCE"
                )
            sampled_control = self.control_name_fn(sampled_control_token)
            control_debug = dict(self.session.last_sample_debug)
            listening_block_inserted = True
            self.barge_in_waiting_for_silence = True
            self.reply_pending = False

        planned = response_step.planned_response
        committed = response_step.committed_fragment
        utterance: Optional[SynthesizedUtterance] = None
        if planned is not None:
            self.reply_pending = False
            self.response_armed = False
            self.sending = False
            self.silence_run_frames = 0
            self.session.pcm_history.clear()
            if response_step.utterance_index is None:
                raise RuntimeError("Planned response has no synthesized utterance")
            utterance = self.response_controller.utterances[
                int(response_step.utterance_index)
            ]

        speak_label = self.response_controller.action_speak_label()
        response_elapsed_chunk = (
            self.response_controller.action_response_elapsed_chunk()
        )
        emotion_token_id = (
            self.response_controller.action_emotion_token_id()
            if bool(
                getattr(
                    self.wrapper, "use_emotion_conditioning", False
                )
            )
            else None
        )

        if str(self.args.state_source) == "rollout" and self.rollout_action is not None:
            state_for_action = self.rollout_action.detach()
        else:
            state_for_action = self.initial_state
        report_progress("action")
        pred_action = self.sample_action_fn(
            session=self.session,
            robot_state=state_for_action,
            chunk_idx=self.chunk_idx,
            action_horizon=int(self.data_config.action_horizon),
            action_dim=int(self.data_config.action_dim),
            sampling_steps=int(self.args.sampling_steps),
            solver="euler",
            initial_noise=str(self.args.initial_noise),
            device=self.device,
            action_generator=self.action_generator,
            teacher_forced_speak_label=speak_label,
            response_elapsed_chunk=response_elapsed_chunk,
            emotion_token_id=emotion_token_id,
        )
        if self.rollout_action is None:
            rollout_action = pred_action.clone()
        else:
            rollout_action = stitch_action_chunk_c2(
                self.rollout_action,
                pred_action,
                bridge_frames=int(self.args.c2_bridge_frames),
            )
        self.rollout_action = rollout_action.detach()

        keep_probability = self.control_probability_fn(
            control_debug, KEEP_SILENCE
        )
        result = OnlineStepResult(
            chunk_idx=int(self.chunk_idx),
            action=(
                rollout_action.detach().float().cpu().numpy().astype(
                    np.float64, copy=False
                )
            ),
            frontend_mode=frontend_mode,
            control=str(response_step.control),
            sampled_control=str(sampled_control),
            planned_text=(planned.display_text if planned is not None else ""),
            committed_text=(committed.text if committed is not None else ""),
            utterance=utterance,
            response_active=bool(self.response_controller.active),
            response_ended=bool(
                (committed is not None and committed.end)
                or response_step.interrupted
            ),
            interrupted=bool(response_step.interrupted),
            action_speak_label=int(speak_label),
            keep_silence_probability=keep_probability,
            text_begin_probability=text_probability,
            compute_sec=float(time.perf_counter() - started),
        )
        self.chunk_idx += 1
        return result
