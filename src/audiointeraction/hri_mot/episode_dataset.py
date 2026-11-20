import json
import math
import os
import pickle
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.audiointeraction.hri_mot.data_config import (
    DEFAULT_TEXT_CHUNK_TIME_MODE,
    HriMotDataConfig,
    PARTNER_SILENT,
    PARTNER_SPEAKING,
    PARTNER_STATE_ID_TO_TEXT,
    PARTNER_STATE_TO_ID,
    RESPONSE_STATE_ID_TO_TEXT,
    RESPONSE_STATE_TO_ID,
    TEXT_CHUNK_TIME_MODES,
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _as_float(value: Any, default: float = 0.0) -> float:
    if _is_number(value):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _segment_overlaps(start: Any, end: Any, t_start: float, t_end: float) -> bool:
    return _is_number(start) and _is_number(end) and float(start) < t_end and float(end) > t_start


def _word_belongs_to_interval(
    start: Any,
    end: Any,
    t_start: float,
    t_end: float,
    mode: str,
) -> bool:
    if not (_is_number(start) and _is_number(end)):
        return False
    if mode == "midpoint":
        timestamp = (float(start) + float(end)) * 0.5
    elif mode == "start":
        timestamp = float(start)
    else:
        raise ValueError(f"Unknown text_chunk_time_mode {mode!r}; expected one of {TEXT_CHUNK_TIME_MODES}")
    return t_start <= timestamp < t_end


def partner_state_to_id(state: Any, config: Optional[HriMotDataConfig] = None) -> int:
    state_key = "" if state is None else str(state)
    if state_key in PARTNER_STATE_TO_ID:
        return PARTNER_STATE_TO_ID[state_key]
    message = f"Unknown partner response_state {state_key!r}; mapping to PARTNER_SILENT"
    if config is not None and config.strict_unknown_partner_state:
        raise ValueError(message)
    warnings.warn(message)
    return PARTNER_SILENT


def partner_state_id_to_text(state_id: int) -> str:
    return PARTNER_STATE_ID_TO_TEXT.get(int(state_id), PARTNER_STATE_ID_TO_TEXT[PARTNER_SILENT])


def response_state_to_id(state: Any, config: Optional[HriMotDataConfig] = None) -> int:
    state_key = "" if state is None else str(state)
    if state_key in RESPONSE_STATE_TO_ID:
        return RESPONSE_STATE_TO_ID[state_key]
    message = f"Unknown response_state {state_key!r}; mapping to silent"
    if config is not None and config.strict_unknown_partner_state:
        raise ValueError(message)
    warnings.warn(message)
    return PARTNER_SILENT


def response_state_id_to_text(state_id: int) -> str:
    return RESPONSE_STATE_ID_TO_TEXT.get(int(state_id), RESPONSE_STATE_ID_TO_TEXT[PARTNER_SILENT])


def _extract_text_from_transcript(
    data: Dict[str, Any],
    t_start: float,
    t_end: float,
    text_chunk_time_mode: str = DEFAULT_TEXT_CHUNK_TIME_MODE,
) -> str:
    """Extract transcript text assigned to this chunk by timed word ownership."""
    transcript = data.get("metadata:transcript", [])
    if not isinstance(transcript, list):
        return ""

    mode = str(text_chunk_time_mode)
    if mode not in TEXT_CHUNK_TIME_MODES:
        raise ValueError(f"Unknown text_chunk_time_mode {mode!r}; expected one of {TEXT_CHUNK_TIME_MODES}")

    text_parts: List[str] = []
    for segment in transcript:
        if not isinstance(segment, dict):
            continue
        segment_start = segment.get("start")
        segment_end = segment.get("end")
        if not _segment_overlaps(segment_start, segment_end, t_start, t_end):
            continue

        segment_words: List[str] = []
        has_timed_words = False
        words = segment.get("words", [])
        if isinstance(words, list):
            for word in words:
                if not isinstance(word, dict):
                    continue
                word_start = word.get("start")
                word_end = word.get("end")
                if not (_is_number(word_start) and _is_number(word_end)):
                    continue
                has_timed_words = True
                if not _word_belongs_to_interval(word_start, word_end, t_start, t_end, mode):
                    continue
                text = str(word.get("word", "")).strip()
                if text:
                    segment_words.append(text)

        if segment_words:
            text_parts.extend(segment_words)
        elif not has_timed_words and _is_number(segment_start) and t_start <= float(segment_start) < t_end:
            fallback = str(segment.get("transcript", "")).strip()
            if fallback:
                text_parts.append(fallback)

    return " ".join(text_parts).strip()


def extract_text_for_interval(
    json_path: str,
    t_start: float,
    t_end: float,
    config: Optional[HriMotDataConfig] = None,
) -> str:
    config = config or HriMotDataConfig()
    if not json_path or not os.path.exists(json_path):
        message = f"Missing target json file: {json_path}"
        if config.strict_missing_files:
            raise FileNotFoundError(message)
        warnings.warn(message)
        return ""

    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    text = _extract_text_from_transcript(
        data,
        float(t_start),
        float(t_end),
        config.text_chunk_time_mode,
    )
    if not text:
        return ""
    return f"{config.text_begin_token} {text} {config.text_end_token}"


def _read_audio_file(path: str) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)
    except ImportError:
        pass

    try:
        import torchaudio

        audio_tensor, sample_rate = torchaudio.load(path)
        audio = audio_tensor.transpose(0, 1).detach().cpu().numpy().astype(np.float32)
        return audio, int(sample_rate)
    except ImportError:
        pass

    try:
        import librosa

        audio, sample_rate = librosa.load(path, sr=None, mono=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.T
        return audio, int(sample_rate)
    except ImportError as exc:
        raise ImportError(
            "Loading wav files requires soundfile, torchaudio, or librosa to be installed"
        ) from exc


def _resample_1d(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    try:
        import torchaudio.functional as AF

        tensor = torch.from_numpy(audio.astype(np.float32))
        return AF.resample(tensor, source_rate, target_rate).cpu().numpy().astype(np.float32)
    except ImportError:
        pass

    try:
        import librosa

        return librosa.resample(audio.astype(np.float32), orig_sr=source_rate, target_sr=target_rate)
    except ImportError:
        pass

    new_length = max(1, int(round(audio.shape[0] * float(target_rate) / float(source_rate))))
    old_x = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    new_x = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
    return np.interp(new_x, old_x, audio).astype(np.float32)


def _pad_or_trim_1d(audio: np.ndarray, length: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.shape[0] == length:
        return audio
    if audio.shape[0] > length:
        return audio[:length]
    out = np.zeros(length, dtype=np.float32)
    out[: audio.shape[0]] = audio
    return out


class HriMotEpisodeDataset(Dataset):
    def __init__(
        self,
        index_paths: Sequence[str],
        config: Optional[HriMotDataConfig] = None,
        split: str = "train",
        episode_index_cache: Optional[str] = None,
    ) -> None:
        self.config = config or HriMotDataConfig()
        if isinstance(index_paths, (str, os.PathLike)):
            index_paths = [str(index_paths)]
        self.index_paths = [str(path) for path in index_paths]
        self.split = split
        self.episode_index_cache = str(episode_index_cache) if episode_index_cache else None

        self.episodes: List[Dict[str, Any]] = []
        self._audio_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self._json_cache: Dict[str, Dict[str, Any]] = {}
        self._robot_cache: Dict[str, Tuple[np.ndarray, int]] = {}
        self._warned: set = set()
        self._cache_max_items = 8

        if not self._load_episode_index_cache():
            self._load_indices()
            self._write_episode_index_cache()

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode = self.episodes[idx]
        rows = episode.get("chunks")
        if rows is None:
            rows = self._read_episode_rows(episode)
        cfg = self.config
        samples_per_chunk = int(round(cfg.chunk_size_sec * cfg.audio_sample_rate))

        partner_audio: List[torch.Tensor] = []
        partner_state_ids: List[int] = []
        partner_state_text: List[str] = []
        robot_state: List[torch.Tensor] = []
        target_action: List[torch.Tensor] = []
        target_text: List[str] = []
        target_text_plain: List[str] = []
        target_control: List[str] = []
        target_response_loss_mask: List[int] = []
        target_response_state: List[str] = []
        target_state_ids: List[int] = []
        target_state_text: List[str] = []
        t_start_values: List[float] = []
        t_end_values: List[float] = []

        for row in rows:
            chunk = row.get("chunk", {})
            target = chunk.get("target", {}) if isinstance(chunk.get("target"), dict) else {}
            partner = chunk.get("partner", {}) if isinstance(chunk.get("partner"), dict) else {}
            t_start = _as_float(chunk.get("t_start"), 0.0)
            t_end = _as_float(chunk.get("t_end"), t_start + cfg.chunk_size_sec)

            if cfg.include_partner_state:
                state_id = partner_state_to_id(partner.get("response_state"), cfg)
                partner_state_ids.append(state_id)
                partner_state_text.append(partner_state_id_to_text(state_id))

            audio = self._load_partner_audio(partner.get("wav"), t_start, t_end, samples_per_chunk)
            partner_audio.append(torch.from_numpy(audio))

            current_qpos, next_qpos = self._load_robot_segments(target.get("robot_pkl"), t_end)
            robot_state.append(torch.from_numpy(current_qpos))
            target_action.append(torch.from_numpy(next_qpos))

            if cfg.include_text_fields:
                target_state = target.get("response_state")
                target_state_id = response_state_to_id(target_state, cfg)
                plain_text = self._load_target_text_plain(target.get("json"), t_start, t_end)
                target_text_plain.append(plain_text)
                target_text.append(
                    ""
                    if not plain_text
                    else f"{self.config.text_begin_token} {plain_text} {self.config.text_end_token}"
                )
                target_control.append("" if target.get("response_control") is None else str(target.get("response_control")))
                raw_loss_mask = target.get("response_loss_mask", 0)
                target_response_loss_mask.append(0 if raw_loss_mask is None else int(raw_loss_mask))
                target_response_state.append("" if target_state is None else str(target_state))
                target_state_ids.append(target_state_id)
                target_state_text.append(response_state_id_to_text(target_state_id))
            t_start_values.append(t_start)
            t_end_values.append(t_end)

        sample = {
            "episode_id": episode["episode_id"],
            "metadata": dict(episode["metadata"]),
            "partner_audio": torch.stack(partner_audio, dim=0).float(),
            "robot_state": torch.stack(robot_state, dim=0).float(),
            "target_action": torch.stack(target_action, dim=0).float(),
            "t_start": torch.tensor(t_start_values, dtype=torch.float32),
            "t_end": torch.tensor(t_end_values, dtype=torch.float32),
        }
        if cfg.include_partner_state:
            sample.update(
                {
                    "partner_state_ids": torch.tensor(partner_state_ids, dtype=torch.long),
                    "partner_state_text": partner_state_text,
                }
            )
        if cfg.include_text_fields:
            sample.update(
                {
                    "target_text": target_text,
                    "target_text_plain": target_text_plain,
                    "target_control": target_control,
                    "target_response_loss_mask": torch.tensor(target_response_loss_mask, dtype=torch.long),
                    "target_response_state": target_response_state,
                    "target_state_ids": torch.tensor(target_state_ids, dtype=torch.long),
                    "target_state_text": target_state_text,
                }
            )
        return sample

    def _row_episode_id(self, row: Dict[str, Any], index_path: str, line_number: int) -> str:
        episode_id = row.get("episode_id")
        if episode_id not in (None, ""):
            return str(episode_id)

        id_sample = row.get("id_sample")
        if id_sample not in (None, ""):
            split = row.get("split") or self.split or "sample"
            return f"{split}_{id_sample}"

        episode_idx = row.get("episode_idx")
        if episode_idx not in (None, ""):
            split = row.get("split") or self.split or "episode"
            return f"{split}_episode_{episode_idx}"

        raise ValueError(f"Missing episode_id/id_sample at {index_path}:{line_number}")

    def _episode_index_cache_is_fresh(self) -> bool:
        if not self.episode_index_cache or not os.path.exists(self.episode_index_cache):
            return False
        cache_mtime = os.path.getmtime(self.episode_index_cache)
        return all(os.path.exists(path) and os.path.getmtime(path) <= cache_mtime for path in self.index_paths)

    def _load_episode_index_cache(self) -> bool:
        if not self._episode_index_cache_is_fresh():
            return False
        max_episodes = self.config.max_episodes
        with open(self.episode_index_cache, "r", encoding="utf-8") as fp:
            for line in fp:
                if max_episodes is not None and len(self.episodes) >= max_episodes:
                    break
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self.episodes.append(record)
        return True

    def _write_episode_index_cache(self) -> None:
        if not self.episode_index_cache:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.episode_index_cache)) or ".", exist_ok=True)
        tmp_path = f"{self.episode_index_cache}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as fp:
            for episode in self.episodes:
                if "row_offsets" not in episode:
                    continue
                fp.write(json.dumps(episode, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self.episode_index_cache)

    def _load_indices(self) -> None:
        max_episodes = self.config.max_episodes
        for index_path in self.index_paths:
            if max_episodes is not None and len(self.episodes) >= max_episodes:
                break
            if not os.path.exists(index_path):
                self._handle_missing_file(index_path, "index")
                continue

            current_episode_id: Optional[str] = None
            current_refs: List[Tuple[int, int]] = []
            current_first_row: Optional[Dict[str, Any]] = None
            line_number = 0
            with open(index_path, "r", encoding="utf-8") as fp:
                while True:
                    if max_episodes is not None and len(self.episodes) >= max_episodes:
                        break
                    offset = fp.tell()
                    line = fp.readline()
                    if not line:
                        break
                    line_number += 1
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Malformed JSONL at {index_path}:{line_number}: {exc}") from exc

                    episode_id = self._row_episode_id(row, index_path, line_number)
                    if current_episode_id is None:
                        current_episode_id = episode_id
                        current_first_row = row
                    if episode_id != current_episode_id:
                        self._maybe_add_episode(
                            current_episode_id, current_refs, index_path, current_first_row
                        )
                        current_episode_id = episode_id
                        current_refs = []
                        current_first_row = row

                    chunk_idx = int(row.get("episode_chunk_idx", len(current_refs)))
                    current_refs.append((chunk_idx, offset))

            if current_episode_id is not None and (
                max_episodes is None or len(self.episodes) < max_episodes
            ):
                self._maybe_add_episode(current_episode_id, current_refs, index_path, current_first_row)

    def _maybe_add_episode(
        self,
        episode_id: str,
        row_refs: List[Tuple[int, int]],
        index_path: str,
        first_row: Optional[Dict[str, Any]],
    ) -> None:
        if not row_refs:
            return
        row_refs = sorted(row_refs, key=lambda item: item[0])
        expected = self.config.num_chunks_per_episode
        if len(row_refs) != expected:
            message = f"Episode {episode_id!r} from {index_path} has {len(row_refs)} chunks, expected {expected}"
            if self.config.skip_malformed_episodes:
                self._warn_once(("malformed_episode", episode_id), message)
                return
            raise ValueError(message)

        first = first_row or {}
        metadata = {key: value for key, value in first.items() if key != "chunk"}
        metadata["episode_id"] = episode_id
        metadata["index_path"] = index_path
        metadata["row_offsets"] = [offset for _, offset in row_refs]
        self.episodes.append(
            {
                "episode_id": episode_id,
                "index_path": index_path,
                "row_offsets": [offset for _, offset in row_refs],
                "metadata": metadata,
            }
        )

    def _read_episode_rows(self, episode: Dict[str, Any]) -> List[Dict[str, Any]]:
        row_offsets = episode.get("row_offsets")
        index_path = episode.get("index_path") or episode.get("metadata", {}).get("index_path")
        if row_offsets is None or index_path is None:
            raise ValueError(f"Episode {episode.get('episode_id')} is missing row_offsets/index_path")

        rows: List[Dict[str, Any]] = []
        with open(index_path, "r", encoding="utf-8") as fp:
            for offset in row_offsets:
                fp.seek(int(offset))
                line = fp.readline().strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return sorted(rows, key=lambda row: int(row.get("episode_chunk_idx", 0)))

    def _warn_once(self, key: Any, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        warnings.warn(message)

    def _handle_missing_file(self, path: Optional[str], kind: str) -> None:
        message = f"Missing {kind} file: {path}"
        if self.config.strict_missing_files:
            raise FileNotFoundError(message)
        self._warn_once(("missing", kind, path), message)

    def _load_partner_audio(
        self, wav_path: Optional[str], t_start: float, t_end: float, samples_per_chunk: int
    ) -> np.ndarray:
        if not wav_path or not os.path.exists(wav_path):
            self._handle_missing_file(wav_path, "partner wav")
            return np.zeros(samples_per_chunk, dtype=np.float32)

        if wav_path not in self._audio_cache:
            if self.config.audio_loader == "whisper":
                import whisper

                # Match the Web inference decoder: decode/resample the complete
                # source wav to mono float32 at 16 kHz before slicing by time.
                # No gain, normalization, DC removal, or clipping is applied.
                audio = whisper.load_audio(
                    str(wav_path), sr=int(self.config.audio_sample_rate)
                ).astype(np.float32)
                sample_rate = int(self.config.audio_sample_rate)
            else:
                audio, sample_rate = _read_audio_file(wav_path)
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1) if self.config.audio_mono else audio[:, 0]
            elif audio.ndim > 2:
                raise ValueError(f"Expected mono/stereo audio for {wav_path}, got shape {audio.shape}")
            audio = _resample_1d(audio.reshape(-1), sample_rate, self.config.audio_sample_rate)
            if len(self._audio_cache) >= self._cache_max_items:
                self._audio_cache.pop(next(iter(self._audio_cache)))
            self._audio_cache[wav_path] = (audio.astype(np.float32, copy=False), self.config.audio_sample_rate)

        audio, sample_rate = self._audio_cache[wav_path]
        start = max(0, int(round(t_start * sample_rate)))
        end = max(start, int(round(t_end * sample_rate)))
        return _pad_or_trim_1d(audio[start:end], samples_per_chunk)

    def _load_json_cached(self, json_path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not json_path or not os.path.exists(json_path):
            self._handle_missing_file(json_path, "target json")
            return None
        if json_path not in self._json_cache:
            if len(self._json_cache) >= self._cache_max_items:
                self._json_cache.pop(next(iter(self._json_cache)))
            with open(json_path, "r", encoding="utf-8") as fp:
                self._json_cache[json_path] = json.load(fp)
        return self._json_cache[json_path]

    def _load_target_text_plain(self, json_path: Optional[str], t_start: float, t_end: float) -> str:
        data = self._load_json_cached(json_path)
        if data is None:
            return ""
        return _extract_text_from_transcript(data, t_start, t_end, self.config.text_chunk_time_mode)

    def _load_target_text(self, json_path: Optional[str], t_start: float, t_end: float) -> str:
        text = self._load_target_text_plain(json_path, t_start, t_end)
        if not text:
            return ""
        return f"{self.config.text_begin_token} {text} {self.config.text_end_token}"

    def _load_robot_qpos(self, pkl_path: Optional[str]) -> Tuple[np.ndarray, int]:
        cfg = self.config
        if not pkl_path or not os.path.exists(pkl_path):
            self._handle_missing_file(pkl_path, "robot pkl")
            return np.zeros((0, cfg.robot_state_dim), dtype=np.float32), cfg.robot_fps

        if pkl_path in self._robot_cache:
            return self._robot_cache[pkl_path]

        with open(pkl_path, "rb") as fp:
            data = pickle.load(fp)
        if not isinstance(data, dict):
            raise ValueError(f"Robot pkl must contain a dict, got {type(data).__name__}: {pkl_path}")

        fps = int(data.get("fps", cfg.robot_fps))
        qpos = np.asarray(data.get(cfg.robot_source_key), dtype=np.float32)
        if qpos.ndim != 2:
            raise ValueError(f"{cfg.robot_source_key} must be 2D in {pkl_path}, got shape {qpos.shape}")
        max_index = max(cfg.robot_valid_qpos_indices)
        if qpos.shape[1] <= max_index:
            raise ValueError(
                f"{cfg.robot_source_key} in {pkl_path} has {qpos.shape[1]} dims, need index {max_index}"
            )

        qpos = qpos[:, cfg.robot_valid_qpos_indices].astype(np.float32, copy=False)
        if qpos.shape[1] != cfg.robot_state_dim:
            raise ValueError(
                f"Selected robot qpos dim {qpos.shape[1]} does not match robot_state_dim {cfg.robot_state_dim}"
            )
        if len(self._robot_cache) >= self._cache_max_items:
            self._robot_cache.pop(next(iter(self._robot_cache)))
        self._robot_cache[pkl_path] = (qpos, fps)
        return qpos, fps

    def _load_robot_segments(
        self, pkl_path: Optional[str], t_start: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        qpos, fps = self._load_robot_qpos(pkl_path)
        steps = self.config.robot_steps_per_chunk
        start_frame = int(round(t_start * fps))
        previous = self._slice_qpos(qpos, start_frame - steps, steps)
        current = self._slice_qpos(qpos, start_frame, steps)
        return previous, current

    def _slice_qpos(self, qpos: np.ndarray, start: int, length: int) -> np.ndarray:
        dim = self.config.robot_state_dim
        if qpos.shape[0] == 0:
            return np.zeros((length, dim), dtype=np.float32)
        out = np.empty((length, dim), dtype=np.float32)
        last = qpos.shape[0] - 1
        for offset in range(length):
            index = min(max(start + offset, 0), last)
            out[offset] = qpos[index]
        return out


class HriMotEpisodeCollator:
    tensor_keys = (
        "partner_audio",
        "partner_state_ids",
        "robot_state",
        "target_action",
        "target_action_valid_mask",
        "target_response_loss_mask",
        "target_state_ids",
        "target_emotion_id",
        "target_emotion_id",
        "t_start",
        "t_end",
    )
    list_keys = (
        "partner_state_text",
        "target_text",
        "target_text_plain",
        "target_control",
        "target_response_state",
        "target_state_text",
    )

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("Cannot collate an empty batch")
        batch: Dict[str, Any] = {}
        for key in self.tensor_keys:
            if key in samples[0]:
                batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
        for key in self.list_keys:
            if key in samples[0]:
                batch[key] = [sample[key] for sample in samples]
        batch["episode_id"] = [sample["episode_id"] for sample in samples]
        batch["metadata"] = [sample["metadata"] for sample in samples]
        return batch
