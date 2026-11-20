"""Reconstruct a time-aligned partner/robot interaction waveform."""

from __future__ import annotations

from math import gcd
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample_poly

from src.audiointeraction.hri_mot.streaming_tts import SynthesizedUtterance


def _resample_pcm(
    pcm: np.ndarray,
    *,
    source_rate: int,
    target_rate: int,
) -> np.ndarray:
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    source_rate = int(source_rate)
    target_rate = int(target_rate)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError(
            f"Audio sample rates must be positive: {source_rate} -> {target_rate}"
        )
    if pcm.size == 0 or source_rate == target_rate:
        return pcm.copy()
    divisor = gcd(source_rate, target_rate)
    resampled = resample_poly(
        pcm,
        target_rate // divisor,
        source_rate // divisor,
    )
    expected_samples = int(round(pcm.size * target_rate / source_rate))
    if resampled.size < expected_samples:
        resampled = np.pad(resampled, (0, expected_samples - resampled.size))
    return np.asarray(resampled[:expected_samples], dtype=np.float32)


def reconstruct_interaction_audio(
    *,
    partner_pcm: np.ndarray,
    sample_rate: int,
    duration_sec: float,
    response_records: Sequence[Dict[str, Any]],
    tts_utterances: Sequence[SynthesizedUtterance],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Mix original partner audio with only playback-confirmed robot TTS.

    A synthesized response may be longer than what the robot actually played.
    The response records contain the committed half-open TTS PCM intervals, so
    fragments after a barge-in are intentionally absent from the reconstruction.
    """

    sample_rate = int(sample_rate)
    duration_sec = float(duration_sec)
    if sample_rate <= 0 or duration_sec <= 0.0:
        raise ValueError(
            f"Invalid interaction audio timing: rate={sample_rate}, "
            f"duration={duration_sec}"
        )
    total_samples = max(1, int(round(duration_sec * sample_rate)))
    partner_track = np.zeros(total_samples, dtype=np.float32)
    partner = np.asarray(partner_pcm, dtype=np.float32).reshape(-1)
    partner_samples = min(total_samples, int(partner.size))
    partner_track[:partner_samples] = partner[:partner_samples]

    robot_track = np.zeros(total_samples, dtype=np.float32)
    robot_segments = []
    interrupted_utterances = {
        int(record["tts_utterance_index"])
        for record in response_records
        if bool(record.get("response_interrupted", False))
        and record.get("tts_utterance_index") is not None
    }
    for record in response_records:
        utterance_index = record.get("tts_utterance_index")
        playback_t_start = record.get("playback_t_start")
        pcm_start = record.get("playback_pcm_start_sample")
        pcm_end = record.get("playback_pcm_end_sample")
        fragment_rate = record.get("playback_sample_rate")
        if (
            utterance_index is None
            or playback_t_start is None
            or pcm_start is None
            or pcm_end is None
            or fragment_rate is None
        ):
            continue
        utterance_index = int(utterance_index)
        pcm_start = int(pcm_start)
        pcm_end = int(pcm_end)
        fragment_rate = int(fragment_rate)
        if (
            not 0 <= utterance_index < len(tts_utterances)
            or fragment_rate <= 0
            or pcm_end <= pcm_start
        ):
            # A barge-in TEXT_END has a zero-length PCM interval and therefore
            # closes the text/action turn without replaying any robot samples.
            continue
        utterance = tts_utterances[utterance_index]
        source_start = max(0, min(pcm_start, int(utterance.pcm.size)))
        source_end = max(source_start, min(pcm_end, int(utterance.pcm.size)))
        if source_end <= source_start:
            continue
        fragment = _resample_pcm(
            utterance.pcm[source_start:source_end],
            source_rate=int(utterance.sample_rate),
            target_rate=sample_rate,
        )
        destination_start = max(
            0, int(round(float(playback_t_start) * sample_rate))
        )
        destination_end = min(total_samples, destination_start + fragment.size)
        if destination_end <= destination_start:
            continue
        copied = int(destination_end - destination_start)
        robot_track[destination_start:destination_end] += fragment[:copied]
        robot_segments.append(
            {
                "utterance_index": utterance_index,
                "playback_t_start": float(destination_start / sample_rate),
                "playback_t_end": float(destination_end / sample_rate),
                "utterance_pcm_start_sample": source_start,
                "utterance_pcm_end_sample": source_end,
                "interrupted_response": utterance_index in interrupted_utterances,
            }
        )

    mixed = partner_track + robot_track
    peak_before_limit = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    mix_gain = 1.0
    if peak_before_limit > 0.99:
        mix_gain = float(0.99 / peak_before_limit)
        mixed *= mix_gain

    metadata = {
        "sample_rate": sample_rate,
        "duration_sec": float(total_samples / sample_rate),
        "total_samples": int(total_samples),
        "partner_source_samples": int(partner.size),
        "partner_copied_samples": int(partner_samples),
        "robot_segments": robot_segments,
        "robot_segment_count": int(len(robot_segments)),
        "robot_active_samples": int(np.count_nonzero(np.abs(robot_track) > 1e-8)),
        "peak_before_limit": peak_before_limit,
        "mix_gain": mix_gain,
        "barge_in_policy": "omit_uncommitted_tts_pcm_after_interrupt_boundary",
    }
    return np.asarray(mixed, dtype=np.float32), metadata


def reconstruct_realtime_interaction_audio(
    *,
    partner_pcm: np.ndarray,
    sample_rate: int,
    minimum_duration_sec: float,
    response_records: Sequence[Dict[str, Any]],
    tts_utterances: Sequence[SynthesizedUtterance],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Mix partner audio with robot TTS placed on the measured wall clock."""

    sample_rate = int(sample_rate)
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")

    interruptions: Dict[int, float] = {}
    for record in response_records:
        utterance_index = record.get("tts_utterance_index")
        stop_t = record.get("realtime_tts_playback_t_stop")
        if utterance_index is not None and stop_t is not None:
            interruptions[int(utterance_index)] = float(stop_t)

    events = []
    latest_end = float(minimum_duration_sec)
    for record in response_records:
        utterance_index = record.get("tts_utterance_index")
        start_t = record.get("realtime_tts_playback_t_start")
        if utterance_index is None or start_t is None:
            continue
        utterance_index = int(utterance_index)
        if not 0 <= utterance_index < len(tts_utterances):
            continue
        utterance = tts_utterances[utterance_index]
        start_t = float(start_t)
        natural_end_t = float(start_t + utterance.duration_sec)
        stop_t: Optional[float] = interruptions.get(utterance_index)
        end_t = min(natural_end_t, stop_t) if stop_t is not None else natural_end_t
        end_t = max(start_t, float(end_t))
        latest_end = max(latest_end, end_t)
        events.append(
            {
                "utterance_index": utterance_index,
                "wall_playback_t_start": start_t,
                "wall_playback_t_end": end_t,
                "natural_t_end": natural_end_t,
                "interrupted": bool(end_t + 1e-9 < natural_end_t),
            }
        )

    partner = np.asarray(partner_pcm, dtype=np.float32).reshape(-1)
    latest_end = max(latest_end, float(partner.size / sample_rate))
    total_samples = max(1, int(np.ceil(latest_end * sample_rate)))
    partner_track = np.zeros(total_samples, dtype=np.float32)
    partner_track[: partner.size] = partner
    robot_track = np.zeros(total_samples, dtype=np.float32)

    for event in events:
        utterance = tts_utterances[event["utterance_index"]]
        pcm = _resample_pcm(
            utterance.pcm,
            source_rate=int(utterance.sample_rate),
            target_rate=sample_rate,
        )
        destination_start = int(
            round(event["wall_playback_t_start"] * sample_rate)
        )
        requested_samples = int(
            round(
                (
                    event["wall_playback_t_end"]
                    - event["wall_playback_t_start"]
                )
                * sample_rate
            )
        )
        copied = min(
            max(0, requested_samples),
            int(pcm.size),
            total_samples - destination_start,
        )
        if copied > 0:
            robot_track[
                destination_start : destination_start + copied
            ] += pcm[:copied]
        event["copied_samples"] = int(copied)
        audible = np.flatnonzero(np.abs(pcm[:copied]) > 1e-4)
        event["wall_audible_t_start"] = (
            float(
                event["wall_playback_t_start"]
                + int(audible[0]) / sample_rate
            )
            if audible.size
            else None
        )

    mixed = partner_track + robot_track
    peak_before_limit = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    mix_gain = 1.0
    if peak_before_limit > 0.99:
        mix_gain = float(0.99 / peak_before_limit)
        mixed *= mix_gain

    metadata = {
        "sample_rate": sample_rate,
        "duration_sec": float(total_samples / sample_rate),
        "total_samples": int(total_samples),
        "partner_source_samples": int(partner.size),
        "robot_events": events,
        "robot_event_count": int(len(events)),
        "peak_before_limit": peak_before_limit,
        "mix_gain": mix_gain,
        "timeline": "measured_wall_clock_from_partner_playback_t0",
        "playback_model": (
            "full TTS waveform starts when decode+TTS return and plays "
            "asynchronously until its natural end or detected barge-in"
        ),
    }
    return np.asarray(mixed, dtype=np.float32), metadata
