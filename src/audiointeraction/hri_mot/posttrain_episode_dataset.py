"""Episode loader for real interaction post-training data.

The post-training JSONL keeps the original 50-row episode structure, but its
target motion is a 50 Hz Astribot S1 CSV and its transcript uses top-level
absolute word timestamps.  This loader adapts only those two storage formats;
the temporal/audio semantics stay identical to ``HriMotEpisodeDataset``.
"""

from __future__ import annotations

import os
import re
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from src.audiointeraction.hri_mot.data_config import HriMotDataConfig
from src.audiointeraction.hri_mot.episode_dataset import HriMotEpisodeDataset


POSTTRAIN_MOTION_SOURCE_FPS = 50.0


def _resample_matrix(
    sequence: np.ndarray,
    *,
    source_fps: float,
    target_fps: float,
) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[0] == 0:
        raise ValueError(f"Motion must have shape [frames, dims], got {sequence.shape}")
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError(
            f"Motion FPS must be positive, source={source_fps}, target={target_fps}"
        )
    if abs(float(source_fps) - float(target_fps)) < 1e-8:
        return sequence.astype(np.float32, copy=False)

    target_frames = max(
        1,
        int(round(sequence.shape[0] * float(target_fps) / float(source_fps))),
    )
    source_times = np.arange(sequence.shape[0], dtype=np.float64) / float(
        source_fps
    )
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
    text = " ".join(str(token).strip() for token in tokens if str(token).strip())
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([\(\[\{])\s+", r"\1", text)
    return text.strip()


def _finite_word_time(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


class PostTrainEpisodeDataset(HriMotEpisodeDataset):
    """Load 20-second post-training episodes without changing co-speech code."""

    def _load_robot_qpos(self, motion_path: Optional[str]) -> Tuple[np.ndarray, int]:
        if not motion_path or not str(motion_path).lower().endswith(".csv"):
            return super()._load_robot_qpos(motion_path)

        path = str(motion_path)
        cfg = self.config
        if not os.path.exists(path):
            self._handle_missing_file(path, "target motion CSV")
            return (
                np.zeros((0, cfg.robot_state_dim), dtype=np.float32),
                cfg.robot_fps,
            )
        if path in self._robot_cache:
            return self._robot_cache[path]

        raw = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
        raw = np.atleast_2d(raw)
        if raw.shape[1] != 36:
            raise ValueError(
                "Expected S1 CSV with Frame + 35 qpos columns, "
                f"got {raw.shape}: {path}"
            )

        # Match MotionDataFactory AstribotS1_CSVConfig.to_anim_frame: chassis
        # XY stay in metres and every joint from chassis yaw onward is degrees.
        qpos35 = raw[:, 1:].copy()
        qpos35[:, 2:] = np.deg2rad(qpos35[:, 2:])
        max_index = max(cfg.robot_valid_qpos_indices)
        if qpos35.shape[1] <= max_index:
            raise ValueError(
                f"S1 CSV has {qpos35.shape[1]} qpos dims, need index {max_index}: {path}"
            )
        selected = qpos35[:, cfg.robot_valid_qpos_indices]
        selected = _resample_matrix(
            selected,
            source_fps=POSTTRAIN_MOTION_SOURCE_FPS,
            target_fps=float(cfg.robot_fps),
        )
        if selected.shape[1] != cfg.robot_state_dim:
            raise ValueError(
                f"Selected motion has {selected.shape[1]} dims, "
                f"expected {cfg.robot_state_dim}: {path}"
            )

        if len(self._robot_cache) >= self._cache_max_items:
            self._robot_cache.pop(next(iter(self._robot_cache)))
        self._robot_cache[path] = (
            selected.astype(np.float32, copy=False),
            int(cfg.robot_fps),
        )
        return self._robot_cache[path]

    def _load_target_text_plain(
        self,
        json_path: Optional[str],
        t_start: float,
        t_end: float,
    ) -> str:
        data = self._load_json_cached(json_path)
        if data is None:
            return ""
        words = data.get("words")
        if not isinstance(words, list):
            return super()._load_target_text_plain(json_path, t_start, t_end)

        selected: List[str] = []
        for word in words:
            if not isinstance(word, dict):
                continue
            start = _finite_word_time(word.get("start"))
            end = _finite_word_time(word.get("end"))
            if start is None or end is None:
                continue
            timestamp = (
                start
                if self.config.text_chunk_time_mode == "start"
                else (start + end) * 0.5
            )
            if float(t_start) <= timestamp < float(t_end):
                text = str(word.get("word", word.get("text", ""))).strip()
                if text:
                    selected.append(text)
        return _join_tokens(selected)
