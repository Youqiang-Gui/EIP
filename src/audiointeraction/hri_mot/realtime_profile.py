"""Wall-clock profiling for paced 400 ms HRI streaming inference."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, Sequence

import numpy as np
import torch


class RealtimeProfiler:
    """Pace chunks by their arrival time and measure synchronous pipeline lag."""

    def __init__(
        self,
        *,
        chunk_size_sec: float,
        device: Optional[torch.device] = None,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
        pace: bool = True,
    ) -> None:
        self.chunk_size_sec = float(chunk_size_sec)
        if self.chunk_size_sec <= 0.0:
            raise ValueError(
                f"chunk_size_sec must be positive, got {self.chunk_size_sec}"
            )
        self.device = device
        self.clock = clock
        self.sleeper = sleeper
        self.pace = bool(pace)
        self.origin = float(self.clock())
        self.current: Optional[Dict[str, Any]] = None

    def _synchronize(self) -> None:
        if (
            self.device is not None
            and self.device.type == "cuda"
            and torch.cuda.is_available()
        ):
            torch.cuda.synchronize(self.device)

    def elapsed(self) -> float:
        return float(self.clock() - self.origin)

    def begin_chunk(self, chunk_idx: int) -> None:
        if self.current is not None:
            raise RuntimeError("Previous realtime-profile chunk is still active")
        chunk_idx = int(chunk_idx)
        scheduled_input_ready_t = float(
            (chunk_idx + 1) * self.chunk_size_sec
        )
        before_wait_t = self.elapsed()
        requested_wait_sec = max(0.0, scheduled_input_ready_t - before_wait_t)
        if self.pace and requested_wait_sec > 0.0:
            self.sleeper(requested_wait_sec)
        processing_start_t = self.elapsed()
        self.current = {
            "chunk_idx": chunk_idx,
            "scheduled_input_ready_t": scheduled_input_ready_t,
            "scheduled_next_input_ready_t": float(
                scheduled_input_ready_t + self.chunk_size_sec
            ),
            "requested_wait_sec": float(requested_wait_sec if self.pace else 0.0),
            "processing_start_t": float(processing_start_t),
            "input_backlog_sec": float(
                max(0.0, processing_start_t - scheduled_input_ready_t)
            ),
            "stages_sec": {},
            "markers_t": {},
        }

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if self.current is None:
            raise RuntimeError("Realtime profile stage used outside a chunk")
        self._synchronize()
        started = float(self.clock())
        try:
            yield
        finally:
            self._synchronize()
            duration = float(self.clock() - started)
            stages = self.current["stages_sec"]
            stages[str(name)] = float(stages.get(str(name), 0.0) + duration)

    def mark(self, name: str) -> float:
        if self.current is None:
            raise RuntimeError("Realtime profile marker used outside a chunk")
        self._synchronize()
        value = self.elapsed()
        self.current["markers_t"][str(name)] = float(value)
        return float(value)

    def finish_chunk(self) -> Dict[str, Any]:
        if self.current is None:
            raise RuntimeError("No active realtime-profile chunk")
        self._synchronize()
        processing_finish_t = self.elapsed()
        result = self.current
        self.current = None
        result["processing_finish_t"] = float(processing_finish_t)
        result["pipeline_compute_sec"] = float(
            processing_finish_t - result["processing_start_t"]
        )
        result["finish_lag_vs_next_input_sec"] = float(
            max(
                0.0,
                processing_finish_t - result["scheduled_next_input_ready_t"],
            )
        )
        result["deadline_missed"] = bool(
            result["finish_lag_vs_next_input_sec"] > 1e-9
        )
        action_ready_t = result["markers_t"].get("action_ready")
        result["action_ready_latency_sec"] = (
            float(action_ready_t - result["scheduled_input_ready_t"])
            if action_ready_t is not None
            else None
        )
        return result


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean_sec": None,
            "p50_sec": None,
            "p95_sec": None,
            "p99_sec": None,
            "max_sec": None,
        }
    return {
        "count": int(array.size),
        "mean_sec": float(array.mean()),
        "p50_sec": float(np.percentile(array, 50)),
        "p95_sec": float(np.percentile(array, 95)),
        "p99_sec": float(np.percentile(array, 99)),
        "max_sec": float(array.max()),
    }


def build_realtime_profile_report(
    *,
    response_records: Sequence[Dict[str, Any]],
    source_duration_sec: float,
    chunk_size_sec: float,
    motion_fps: float = 30.0,
) -> Dict[str, Any]:
    """Summarize per-chunk wall-clock measurements and response latency."""

    chunk_timings = [
        dict(record["realtime"])
        for record in response_records
        if isinstance(record.get("realtime"), dict)
    ]
    stage_names = sorted(
        {
            name
            for timing in chunk_timings
            for name in timing.get("stages_sec", {})
        }
    )
    stage_distributions = {
        name: _distribution(
            [
                float(timing["stages_sec"][name])
                for timing in chunk_timings
                if name in timing.get("stages_sec", {})
            ]
        )
        for name in stage_names
    }
    deadline_misses = sum(
        bool(timing.get("deadline_missed", False))
        for timing in chunk_timings
    )
    wall_clock_elapsed_sec = max(
        [
            float(timing.get("processing_finish_t", 0.0))
            for timing in chunk_timings
        ]
        or [0.0]
    )
    input_chunk_timings = [
        record["realtime"]
        for record in response_records
        if str(record.get("audio_kind", "")) == "wav"
        and isinstance(record.get("realtime"), dict)
    ]
    generated_input_motion_frames = float(
        len(input_chunk_timings) * float(chunk_size_sec) * float(motion_fps)
    )
    input_action_sampling_sec = float(
        sum(
            float(timing.get("stages_sec", {}).get("action_sampling", 0.0))
            for timing in input_chunk_timings
        )
    )
    input_pipeline_compute_sec = float(
        sum(float(timing.get("pipeline_compute_sec", 0.0)) for timing in input_chunk_timings)
    )
    interaction_timeline_sec = float(
        len(chunk_timings) * float(chunk_size_sec)
    )
    interaction_compute_sec = float(
        sum(float(timing.get("pipeline_compute_sec", 0.0)) for timing in chunk_timings)
    )
    interaction_compute_rtf = (
        interaction_compute_sec / interaction_timeline_sec
        if interaction_timeline_sec > 0.0
        else None
    )

    responses = []
    last_activity_t_end: Optional[float] = None
    for record in response_records:
        if str(record.get("audio_frontend_mode", "")) == "real/activity":
            last_activity_t_end = float(record["audio_t_end"])
        playback_start = record.get("realtime_tts_playback_t_start")
        if playback_start is None:
            continue
        reply_action_ready = record.get("realtime_reply_action_ready_t")
        timing = record.get("realtime", {})
        stages = timing.get("stages_sec", {}) if isinstance(timing, dict) else {}
        responses.append(
            {
                "utterance_index": record.get("tts_utterance_index"),
                "planning_chunk_idx": int(record["chunk_idx"]),
                "last_activity_chunk_t_end": last_activity_t_end,
                "wall_playback_t_start": float(playback_start),
                "wall_reply_action_ready_t": (
                    float(reply_action_ready)
                    if reply_action_ready is not None
                    else None
                ),
                "reply_action_latency_from_input_end_sec": (
                    float(reply_action_ready) - float(source_duration_sec)
                    if reply_action_ready is not None
                    else None
                ),
                "wall_latency_from_last_activity_chunk_end_sec": (
                    float(playback_start) - last_activity_t_end
                    if last_activity_t_end is not None
                    else None
                ),
                "text_decode_sec": float(stages.get("text_decode", 0.0)),
                "tts_synthesis_sec": float(stages.get("tts_synthesis", 0.0)),
                "action_sampling_sec": float(stages.get("action_sampling", 0.0)),
                "pipeline_compute_sec": float(
                    timing.get("pipeline_compute_sec", 0.0)
                ),
            }
        )

    return {
        "mode": "paced_synchronous_400ms_wall_clock_profile",
        "clock": "time.perf_counter",
        "cuda_timing": "torch.cuda.synchronize_before_and_after_each_stage",
        "chunk_size_sec": float(chunk_size_sec),
        "source_duration_sec": float(source_duration_sec),
        "wall_clock_elapsed_sec": float(wall_clock_elapsed_sec),
        "wall_clock_realtime_factor_vs_source": (
            float(wall_clock_elapsed_sec / source_duration_sec)
            if source_duration_sec > 0.0
            else None
        ),
        "interaction_timeline_sec": interaction_timeline_sec,
        "interaction_compute_sec": interaction_compute_sec,
        "interaction_compute_rtf": interaction_compute_rtf,
        "interaction_compute_realtime_speedup": (
            1.0 / interaction_compute_rtf
            if interaction_compute_rtf is not None
            and interaction_compute_rtf > 0.0
            else None
        ),
        "processed_chunks": int(len(chunk_timings)),
        "deadline_budget_sec": float(chunk_size_sec),
        "motion_throughput": {
            "nominal_output_fps": float(motion_fps),
            "input_chunks": int(len(input_chunk_timings)),
            "generated_motion_frames": generated_input_motion_frames,
            "action_sampling_compute_sec": input_action_sampling_sec,
            "action_sampling_fps": (
                generated_input_motion_frames / input_action_sampling_sec
                if input_action_sampling_sec > 0.0
                else None
            ),
            "action_sampling_rtf": (
                input_action_sampling_sec
                / (generated_input_motion_frames / float(motion_fps))
                if input_action_sampling_sec > 0.0
                and generated_input_motion_frames > 0.0
                else None
            ),
            "action_sampling_realtime_speedup": (
                (generated_input_motion_frames / float(motion_fps))
                / input_action_sampling_sec
                if input_action_sampling_sec > 0.0
                else None
            ),
            "audio_to_motion_pipeline_compute_sec": input_pipeline_compute_sec,
            "audio_to_motion_pipeline_fps": (
                generated_input_motion_frames / input_pipeline_compute_sec
                if input_pipeline_compute_sec > 0.0
                else None
            ),
            "audio_to_motion_pipeline_rtf": (
                input_pipeline_compute_sec
                / (generated_input_motion_frames / float(motion_fps))
                if input_pipeline_compute_sec > 0.0
                and generated_input_motion_frames > 0.0
                else None
            ),
            "audio_to_motion_pipeline_realtime_speedup": (
                (generated_input_motion_frames / float(motion_fps))
                / input_pipeline_compute_sec
                if input_pipeline_compute_sec > 0.0
                else None
            ),
            "paced_wait_excluded": True,
        },
        "deadline_miss_count": int(deadline_misses),
        "deadline_miss_rate": (
            float(deadline_misses / len(chunk_timings))
            if chunk_timings
            else 0.0
        ),
        "pipeline_compute": _distribution(
            [
                float(timing["pipeline_compute_sec"])
                for timing in chunk_timings
            ]
        ),
        "input_backlog": _distribution(
            [
                float(timing["input_backlog_sec"])
                for timing in chunk_timings
            ]
        ),
        "finish_lag_vs_next_input": _distribution(
            [
                float(timing["finish_lag_vs_next_input_sec"])
                for timing in chunk_timings
            ]
        ),
        "action_ready_latency": _distribution(
            [
                float(timing["action_ready_latency_sec"])
                for timing in chunk_timings
                if timing.get("action_ready_latency_sec") is not None
            ]
        ),
        "stages": stage_distributions,
        "responses": responses,
        "chunks": chunk_timings,
        "interpretation": {
            "interaction_compute_rtf": (
                "Standard direction: summed live-loop compute time divided by "
                "the generated 30 FPS interaction-motion timeline duration. "
                "Paced waiting, model loading, file output and rendering are "
                "excluded; values below 1 are faster than real time."
            ),
            "measured_scope": (
                "Per-chunk audio feature encoding, control sampling, full "
                "response decode, synchronous TTS request, action sampling, "
                "and motion stitching."
            ),
            "excluded_one_time_work": (
                "Model/checkpoint loading, initial WAV decoding, offline VAD "
                "preclassification, saved-motion filtering, rendering, and "
                "video muxing are outside the live-loop clock."
            ),
            "input_arrival": (
                "Chunk I_c becomes available at wall time (c+1)*chunk_size."
            ),
            "deadline_miss": (
                "The synchronous pipeline did not finish before the next "
                "400 ms input chunk became available."
            ),
            "tts_playback": (
                "Robot playback starts when full-text decode and TTS return; "
                "the waveform is then assumed to play asynchronously."
            ),
            "primary_comparison_metric": (
                "reply_action_latency_from_input_end_sec measures from the "
                "last source-WAV sample arriving to the first reply-conditioned "
                "action chunk being CUDA-complete. Rendering, file output and "
                "robot transport are excluded."
            ),
            "motion_fps": (
                "action_sampling_fps is denoising-only throughput. "
                "audio_to_motion_pipeline_fps includes input-audio features, "
                "control/action inference and stitching for WAV chunks, but "
                "excludes the 400 ms pacing wait. Neither is the robot's "
                "nominal 30 Hz playback rate."
            ),
        },
    }
