"""Runtime construction of 20-second co-speech training episodes.

The compact JSONL index used here contains one record per source clip.  Audio
padding, transcript shifting and motion composition are performed lazily so the
dataset does not need materialized 20-second WAV/PKL files.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.audiointeraction.hri_mot.data_config import HriMotDataConfig
from src.audiointeraction.hri_mot.emotion_conditioning import (
    emotion_condition_id,
)
from src.audiointeraction.hri_mot.episode_dataset import (
    _read_audio_file,
    _resample_1d,
    partner_state_id_to_text,
    response_state_id_to_text,
)


DEFAULT_IDLE_MOTION_PATH = "data/motion/idle_motion.pkl"


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _resample_matrix(sequence: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[0] == 0:
        raise ValueError(f"Motion must have shape [frames, dims], got {sequence.shape}")
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(f"Motion FPS must be positive, got source={source_fps}, target={target_fps}")
    if abs(float(source_fps) - float(target_fps)) < 1e-8:
        return sequence.astype(np.float32, copy=False)

    target_frames = max(1, int(round(sequence.shape[0] * float(target_fps) / float(source_fps))))
    source_times = np.arange(sequence.shape[0], dtype=np.float64) / float(source_fps)
    target_times = np.arange(target_frames, dtype=np.float64) / float(target_fps)
    result = np.empty((target_frames, sequence.shape[1]), dtype=np.float32)
    for dim in range(sequence.shape[1]):
        result[:, dim] = np.interp(
            target_times,
            source_times,
            sequence[:, dim],
            left=float(sequence[0, dim]),
            right=float(sequence[-1, dim]),
        ).astype(np.float32)
    return result


def _join_tokens(tokens: Sequence[str]) -> str:
    text = " ".join(token.strip() for token in tokens if str(token).strip())
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return text.strip()


def _timed_words_from_transcript(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize BEAT2/Show/GUAVA word times to the cropped clip timeline.

    BEAT2 and Show transcripts retain timestamps from the source recording,
    while their ``audio.wav`` and motion files are cropped to the top-level
    utterance ``[start, end]`` interval.  Their dict word timestamps therefore
    need that top-level ``start`` subtracted.  GUAVA ``text_intervals`` are
    already clip-local and must not be shifted.
    """
    normalized: List[Dict[str, Any]] = []

    def add_dict_words(
        words: Any,
        offset: float = 0.0,
        clip_duration: Optional[float] = None,
    ) -> bool:
        found = False
        if not isinstance(words, list):
            return found
        for word in words:
            if not isinstance(word, dict):
                continue
            start = _finite_float(word.get("start"), float("nan"))
            end = _finite_float(word.get("end"), float("nan"))
            if not (math.isfinite(start) and math.isfinite(end) and end >= start):
                continue
            text = str(word.get("word", word.get("text", ""))).strip()
            if text:
                local_start = max(0.0, start + offset)
                local_end = max(local_start, end + offset)
                if clip_duration is not None:
                    local_start = min(local_start, clip_duration)
                    local_end = min(local_end, clip_duration)
                normalized.append(
                    {"word": text, "start": local_start, "end": local_end}
                )
                found = True
        return found

    direct_words = data.get("words")
    direct_origin = _finite_float(data.get("start"), 0.0)
    direct_end = _finite_float(data.get("end"), float("nan"))
    direct_duration = (
        direct_end - direct_origin
        if math.isfinite(direct_end) and direct_end > direct_origin
        else None
    )
    if add_dict_words(
        direct_words,
        offset=-direct_origin,
        clip_duration=direct_duration,
    ):
        return sorted(normalized, key=lambda item: (item["start"], item["end"]))

    transcript = data.get("metadata:transcript")
    if isinstance(transcript, list):
        for segment in transcript:
            if not isinstance(segment, dict):
                continue
            add_dict_words(segment.get("words"))
        if normalized:
            return sorted(normalized, key=lambda item: (item["start"], item["end"]))

    # GUAVA stores token strings and parallel token start times.  This matches
    # gestalt.data.beta2_motion_val_dataset.normalize_transcript_words:
    # end = next token start, or clip duration for the final token.
    words = data.get("words")
    intervals = data.get("text_intervals")
    if isinstance(words, list) and isinstance(intervals, list) and len(words) == len(intervals):
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        clip_duration = _finite_float(meta.get("duration"), 0.0)
        for index, token in enumerate(words):
            start = _finite_float(intervals[index], 0.0)
            if index + 1 < len(intervals):
                end = max(start, _finite_float(intervals[index + 1], start))
            else:
                end = clip_duration if clip_duration > start else start + 1e-2
            text = str(token).strip()
            if text:
                normalized.append({"word": text, "start": start, "end": end})

    return normalized


def _speech_intervals(words: Sequence[Dict[str, Any]], max_gap: float) -> List[Tuple[float, float]]:
    intervals = sorted(
        (
            (_finite_float(word.get("start")), _finite_float(word.get("end")))
            for word in words
            if _finite_float(word.get("end")) > _finite_float(word.get("start"))
        ),
        key=lambda item: item[0],
    )
    if not intervals:
        return []
    merged: List[Tuple[float, float]] = []
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start - current_end <= float(max_gap):
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


class CospeechEpisodeDataset(Dataset):
    """One compact index row -> one 20-second, 50-chunk training episode."""

    def __init__(
        self,
        index_paths: Sequence[str],
        config: Optional[HriMotDataConfig] = None,
        split: str = "train",
        *,
        idle_motion_path: str = DEFAULT_IDLE_MOTION_PATH,
        idle_duration_sec: float = 9.0,
        transition_sec: float = 0.4,
        vad_merge_gap_sec: float = float("inf"),
        use_emotion_conditioning: bool = False,
    ) -> None:
        self.config = config or HriMotDataConfig()
        if isinstance(index_paths, (str, os.PathLike)):
            index_paths = [str(index_paths)]
        self.index_paths = [str(path) for path in index_paths]
        self.split = str(split)
        self.idle_motion_path = str(idle_motion_path)
        self.idle_duration_sec = float(idle_duration_sec)
        self.transition_sec = float(transition_sec)
        # One source clip is one assistant response.  Keep TEXT_BEGIN open
        # from its first timed word through its last timed word, including
        # natural pauses, matching streaming inference where one generated
        # TTS utterance remains an active response until playback ends.
        self.vad_merge_gap_sec = float(vad_merge_gap_sec)
        self.use_emotion_conditioning = bool(
            use_emotion_conditioning
        )
        self.records: List[Dict[str, Any]] = []
        self.episodes = self.records  # Compatibility with training diagnostics.
        self._transcript_cache: Dict[str, Tuple[List[Dict[str, Any]], List[Tuple[float, float]]]] = {}
        self._motion_cache: Dict[Tuple[str, float], np.ndarray] = {}
        self._cache_max_items = 8

        expected_chunks = int(round(self.config.episode_length_sec / self.config.chunk_size_sec))
        if expected_chunks != self.config.num_chunks_per_episode:
            raise ValueError(
                f"episode/chunk config implies {expected_chunks} chunks, "
                f"configured {self.config.num_chunks_per_episode}"
            )
        expected_steps = int(round(self.config.chunk_size_sec * self.config.robot_fps))
        if expected_steps != self.config.robot_steps_per_chunk:
            raise ValueError(
                f"chunk/FPS config implies {expected_steps} robot steps, "
                f"configured {self.config.robot_steps_per_chunk}"
            )

        self._load_indices()
        self._idle_qpos = self._load_idle_qpos()

    def __len__(self) -> int:
        return len(self.records)

    def emotion_id(self, index: int) -> int:
        record = self.records[index]
        return emotion_condition_id(
            record.get("target_emotion"),
            source=f"{record.get('_index_path')}:{record.get('_line_number')}",
        )

    def _load_indices(self) -> None:
        max_episodes = self.config.max_episodes
        for index_path in self.index_paths:
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Missing co-speech index: {index_path}")
            with open(index_path, "r", encoding="utf-8") as fp:
                for line_number, line in enumerate(fp, start=1):
                    if max_episodes is not None and len(self.records) >= max_episodes:
                        return
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Malformed JSONL at {index_path}:{line_number}: {exc}") from exc
                    if not isinstance(record, dict):
                        raise ValueError(f"Index row must be an object at {index_path}:{line_number}")
                    record.setdefault("_index_path", index_path)
                    record.setdefault("_line_number", line_number)
                    self.records.append(record)

    def _load_idle_qpos(self) -> np.ndarray:
        if not os.path.exists(self.idle_motion_path):
            raise FileNotFoundError(f"Missing idle motion PKL: {self.idle_motion_path}")
        with open(self.idle_motion_path, "rb") as fp:
            data = pickle.load(fp)
        if not isinstance(data, dict):
            raise ValueError(f"Idle motion PKL must contain a dict: {self.idle_motion_path}")
        qpos = np.asarray(data.get(self.config.robot_source_key), dtype=np.float32)
        if qpos.ndim != 2 or qpos.shape[1] <= max(self.config.robot_valid_qpos_indices):
            raise ValueError(
                f"Idle qpos must contain compact S1 indices through "
                f"{max(self.config.robot_valid_qpos_indices)}, got {qpos.shape}"
            )
        source_fps = _finite_float(data.get("fps"), self.config.robot_fps)
        keep_frames = min(qpos.shape[0], max(1, int(round(self.idle_duration_sec * source_fps))))
        selected = qpos[:keep_frames, self.config.robot_valid_qpos_indices]
        selected = _resample_matrix(selected, source_fps, self.config.robot_fps)
        if selected.shape[1] != self.config.robot_state_dim:
            raise ValueError(f"Idle selected qpos has {selected.shape[1]} dims")
        return selected.astype(np.float32, copy=False)

    def _load_motion_csv(self, path: str, source_fps: float) -> np.ndarray:
        cache_key = (str(path), float(source_fps))
        if cache_key in self._motion_cache:
            return self._motion_cache[cache_key]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing target S1 CSV: {path}")
        raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
        raw = np.atleast_2d(raw)
        if raw.shape[1] != 36:
            raise ValueError(
                f"Expected S1 CSV with Frame + 35 qpos columns, got {raw.shape}: {path}"
            )

        # Match MotionDataFactory AstribotS1_CSVConfig.to_anim_frame exactly:
        # qpos[0:2] are chassis XY in metres; qpos[2:] are degrees in CSV.
        qpos35 = raw[:, 1:].copy()
        qpos35[:, 2:] = np.deg2rad(qpos35[:, 2:])
        selected = qpos35[:, self.config.robot_valid_qpos_indices]
        selected = _resample_matrix(selected, source_fps, self.config.robot_fps)
        if selected.shape[1] != self.config.robot_state_dim:
            raise ValueError(f"Target selected qpos has {selected.shape[1]} dims: {path}")

        if len(self._motion_cache) >= self._cache_max_items:
            self._motion_cache.pop(next(iter(self._motion_cache)))
        self._motion_cache[cache_key] = selected.astype(np.float32, copy=False)
        return self._motion_cache[cache_key]

    def _loop_idle(self, frame_count: int) -> np.ndarray:
        idle = self._idle_qpos
        if idle.shape[0] == 0:
            raise ValueError("Idle motion is empty")
        blend_frames = min(
            max(0, int(round(self.transition_sec * self.config.robot_fps))),
            max(0, idle.shape[0] - 1),
        )
        period = idle.shape[0]
        frame_indices = np.arange(frame_count, dtype=np.int64)
        phases = frame_indices % period
        result = idle[phases].astype(np.float32, copy=True)
        if blend_frames > 0:
            blend_mask = (frame_indices >= period) & (phases < blend_frames)
            if blend_mask.any():
                # Compute only the handful of loop-boundary blend poses with
                # the same scalar operation order as the former per-frame
                # implementation, then assign them in one indexed operation.
                # This stays bit-identical while removing the 1000-frame loop.
                blend_values = np.empty(
                    (blend_frames, idle.shape[1]), dtype=np.float32
                )
                for phase in range(blend_frames):
                    weight = float(phase + 1) / float(blend_frames)
                    blend_values[phase] = (
                        idle[-1] * (1.0 - weight) + idle[phase] * weight
                    )
                result[blend_mask] = blend_values[phases[blend_mask]]
        return result

    def _compose_episode_qpos(self, record: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        target = record.get("target") if isinstance(record.get("target"), dict) else {}
        timeline = record.get("timeline") if isinstance(record.get("timeline"), dict) else {}
        csv_path = str(target.get("motion_csv", ""))
        source_fps = _finite_float(target.get("motion_fps"), 50.0)
        target_qpos = self._load_motion_csv(csv_path, source_fps)

        episode_frames = int(round(self.config.episode_length_sec * self.config.robot_fps))
        target_start_sec = _finite_float(timeline.get("target_start_sec"), 0.0)
        target_start_frame = min(
            episode_frames,
            max(0, int(round(target_start_sec * self.config.robot_fps))),
        )
        insert_frames = min(target_qpos.shape[0], max(0, episode_frames - target_start_frame))
        composite = self._loop_idle(episode_frames)
        idle_base = composite.copy()

        exit_frames = 0
        if insert_frames > 0:
            target_part = target_qpos[:insert_frames]
            composite[target_start_frame : target_start_frame + insert_frames] = target_part

            transition_frames = max(0, int(round(self.transition_sec * self.config.robot_fps)))
            enter_frames = min(transition_frames, insert_frames)
            for offset in range(enter_frames):
                weight = float(offset + 1) / float(enter_frames)
                frame = target_start_frame + offset
                composite[frame] = idle_base[frame] * (1.0 - weight) + target_part[offset] * weight

            target_end_frame = target_start_frame + insert_frames
            exit_frames = min(transition_frames, max(0, episode_frames - target_end_frame))
            if exit_frames > 0:
                final_target = target_part[-1]
                for offset in range(exit_frames):
                    weight = float(offset + 1) / float(exit_frames)
                    frame = target_end_frame + offset
                    composite[frame] = final_target * (1.0 - weight) + idle_base[frame] * weight
        else:
            target_end_frame = target_start_frame

        # Frames after this boundary are not observed target motion.  They are
        # only the idle loop used to make every episode exactly 20 seconds and
        # must not dominate action loss or normalization statistics.  Keep the
        # exit blend itself supervised so the model still learns to settle
        # back to idle after speaking.
        action_supervision_end_frame = target_end_frame + exit_frames

        details = {
            "target_start_sec": float(target_start_frame / self.config.robot_fps),
            "target_motion_end_sec": float(target_end_frame / self.config.robot_fps),
            "target_motion_frames_source_30fps": int(target_qpos.shape[0]),
            "target_motion_frames_used": int(insert_frames),
            "target_motion_truncated": bool(insert_frames < target_qpos.shape[0]),
            "target_motion_exit_transition_frames": int(exit_frames),
            "action_supervision_end_frame": int(action_supervision_end_frame),
            "action_supervision_end_sec": float(
                action_supervision_end_frame / self.config.robot_fps
            ),
            "terminal_idle_padding_frames": int(
                max(0, episode_frames - action_supervision_end_frame)
            ),
        }
        return composite, details

    def _slice_qpos(self, qpos: np.ndarray, start: int, length: int) -> np.ndarray:
        indices = np.clip(
            int(start) + np.arange(int(length), dtype=np.int64),
            0,
            qpos.shape[0] - 1,
        )
        return qpos[indices].astype(np.float32, copy=True)

    def _action_state_arrays(
        self, record: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        qpos, details = self._compose_episode_qpos(record)
        steps = self.config.robot_steps_per_chunk
        supervision_end_frame = int(details["action_supervision_end_frame"])
        boundaries = (
            np.arange(
                1,
                self.config.num_chunks_per_episode + 1,
                dtype=np.int64,
            )
            * steps
        )
        step_offsets = np.arange(steps, dtype=np.int64)[None, :]
        state_frames = boundaries[:, None] - steps + step_offsets
        action_frames = boundaries[:, None] + step_offsets
        states = qpos[
            np.clip(state_frames, 0, qpos.shape[0] - 1)
        ].astype(np.float32, copy=True)
        actions = qpos[
            np.clip(action_frames, 0, qpos.shape[0] - 1)
        ].astype(np.float32, copy=True)
        action_valid_masks = action_frames < supervision_end_frame
        return (
            states,
            actions,
            action_valid_masks,
            details,
        )

    def iter_action_state_arrays(
        self,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield arrays plus the mask used by norm and primary action loss."""
        for record in self.records:
            states, actions, action_valid_mask, _ = self._action_state_arrays(record)
            yield states, actions, action_valid_mask

    def _load_transcript(
        self, path: str
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[float, float]]]:
        if path in self._transcript_cache:
            return self._transcript_cache[path]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing target transcript: {path}")
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if not isinstance(data, dict):
            raise ValueError(f"Target transcript must be an object: {path}")
        words = _timed_words_from_transcript(data)
        if not words:
            raise ValueError(f"Target transcript has no valid timed words: {path}")
        intervals = _speech_intervals(words, self.vad_merge_gap_sec)
        if len(self._transcript_cache) >= self._cache_max_items:
            self._transcript_cache.pop(next(iter(self._transcript_cache)))
        self._transcript_cache[path] = (words, intervals)
        return words, intervals

    def _target_chunk_labels(
        self,
        words: Sequence[Dict[str, Any]],
        speech: Sequence[Tuple[float, float]],
        local_start: float,
        local_end: float,
    ) -> Tuple[str, str, int, str]:
        onset = any(local_start <= start < local_end for start, _ in speech)
        speaking = any(start < local_end and end > local_start for start, end in speech)
        if onset:
            state, control, loss_mask = "onset", "response", 1
        elif speaking:
            state, control, loss_mask = "speaking", "silent", 0
        else:
            state, control, loss_mask = "non_speech", "silent", 1

        selected: List[str] = []
        mode = self.config.text_chunk_time_mode
        for word in words:
            start = _finite_float(word.get("start"))
            end = _finite_float(word.get("end"), start)
            timestamp = start if mode == "start" else (start + end) * 0.5
            if local_start <= timestamp < local_end:
                selected.append(str(word.get("word", "")))
        return state, control, loss_mask, _join_tokens(selected)

    def _load_partner_audio(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing partner WAV: {path}")
        if self.config.audio_loader == "whisper":
            import whisper

            audio = whisper.load_audio(path, sr=int(self.config.audio_sample_rate)).astype(np.float32)
        else:
            audio, source_rate = _read_audio_file(path)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            audio = _resample_1d(audio.reshape(-1), source_rate, self.config.audio_sample_rate)

        total_samples = int(round(self.config.episode_length_sec * self.config.audio_sample_rate))
        padded = np.zeros(total_samples, dtype=np.float32)
        keep = min(total_samples, int(audio.shape[0]))
        padded[:keep] = audio[:keep]
        samples_per_chunk = int(round(self.config.chunk_size_sec * self.config.audio_sample_rate))
        return padded.reshape(self.config.num_chunks_per_episode, samples_per_chunk)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        target = record.get("target") if isinstance(record.get("target"), dict) else {}
        partner = record.get("partner") if isinstance(record.get("partner"), dict) else {}
        timeline = record.get("timeline") if isinstance(record.get("timeline"), dict) else {}

        states, actions, action_valid_mask, motion_details = self._action_state_arrays(record)
        partner_audio = self._load_partner_audio(str(partner.get("wav", "")))
        transcript_path = str(target.get("transcript", ""))
        words, speech = self._load_transcript(transcript_path)
        target_start = _finite_float(timeline.get("target_start_sec"), 0.0)
        partner_duration = _finite_float(partner.get("duration_sec"), 0.0)

        partner_state_ids: List[int] = []
        partner_state_text: List[str] = []
        target_text_plain: List[str] = []
        target_text: List[str] = []
        target_control: List[str] = []
        target_loss_mask: List[int] = []
        target_states: List[str] = []
        target_state_ids: List[int] = []
        target_state_text: List[str] = []
        starts: List[float] = []
        ends: List[float] = []

        for chunk_index in range(self.config.num_chunks_per_episode):
            chunk_start = chunk_index * self.config.chunk_size_sec
            chunk_end = chunk_start + self.config.chunk_size_sec
            starts.append(chunk_start)
            ends.append(chunk_end)

            if self.config.include_partner_state:
                if chunk_start <= 0.0 < chunk_end and partner_duration > 0:
                    partner_state = "onset"
                    partner_state_id = 1
                elif chunk_start < partner_duration and chunk_end > 0.0:
                    partner_state = "speaking"
                    partner_state_id = 1
                else:
                    partner_state = "non_speech"
                    partner_state_id = 0
                partner_state_ids.append(partner_state_id)
                partner_state_text.append(partner_state_id_to_text(partner_state_id))

            if self.config.include_text_fields:
                state, control, loss_mask, plain = self._target_chunk_labels(
                    words,
                    speech,
                    round(chunk_start - target_start, 6),
                    round(chunk_end - target_start, 6),
                )
                state_id = 0 if state == "non_speech" else 1
                target_text_plain.append(plain)
                target_text.append(
                    "" if not plain else f"{self.config.text_begin_token} {plain} {self.config.text_end_token}"
                )
                target_control.append(control)
                target_loss_mask.append(loss_mask)
                target_states.append(state)
                target_state_ids.append(state_id)
                target_state_text.append(response_state_id_to_text(state_id))

        episode_id = str(record.get("episode_id", f"cospeech_{index}"))
        metadata = {key: value for key, value in record.items() if not key.startswith("_")}
        metadata.update(motion_details)
        metadata["target_speech_start_sec"] = (
            float(target_start + speech[0][0]) if speech else None
        )
        metadata["target_speech_end_sec"] = (
            float(target_start + speech[-1][1]) if speech else None
        )
        metadata["target_speech_end_clipped_sec"] = (
            float(
                min(
                    self.config.episode_length_sec,
                    target_start + speech[-1][1],
                )
            )
            if speech
            else None
        )
        metadata["target_speech_motion_end_delta_sec"] = (
            float(
                min(
                    self.config.episode_length_sec,
                    target_start + speech[-1][1],
                )
                - motion_details["target_motion_end_sec"]
            )
            if speech
            else None
        )
        sample: Dict[str, Any] = {
            "episode_id": episode_id,
            "metadata": metadata,
            "partner_audio": torch.from_numpy(partner_audio).float(),
            "robot_state": torch.from_numpy(states).float(),
            "target_action": torch.from_numpy(actions).float(),
            "target_action_valid_mask": torch.from_numpy(action_valid_mask).bool(),
            "t_start": torch.tensor(starts, dtype=torch.float32),
            "t_end": torch.tensor(ends, dtype=torch.float32),
        }
        if self.use_emotion_conditioning:
            sample["target_emotion_id"] = torch.tensor(
                self.emotion_id(index),
                dtype=torch.long,
            )
        if self.config.include_partner_state:
            sample.update(
                {
                    "partner_state_ids": torch.tensor(partner_state_ids, dtype=torch.long),
                    "partner_state_text": partner_state_text,
                }
            )
        if self.config.include_text_fields:
            sample.update(
                {
                    "target_text": target_text,
                    "target_text_plain": target_text_plain,
                    "target_control": target_control,
                    "target_response_loss_mask": torch.tensor(target_loss_mask, dtype=torch.long),
                    "target_response_state": target_states,
                    "target_state_ids": torch.tensor(target_state_ids, dtype=torch.long),
                    "target_state_text": target_state_text,
                }
            )
        return sample
