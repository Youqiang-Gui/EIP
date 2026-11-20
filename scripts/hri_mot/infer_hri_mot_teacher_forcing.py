#!/usr/bin/env python3
"""Training-set HRI-MoT rollout with teacher-forced semantic context.

For decision chunk ``c`` this script conditions the action expert on:

* partner audio chunks ``0..c``;
* ground-truth target control/text chunks ``0..c-1``;
* either the ground-truth robot state for every decision, or only for chunk
  ``0`` followed by raw predicted-action state feedback.

Actions are sampled sequentially. Ground-truth actions are used only for
metrics and the GT comparison video; offline smoothing runs only after the
complete rollout. For a checkpoint trained with teacher-forced SPEAK
conditioning, decision ``c`` uses the shifted ground-truth label
``target_state_ids[c+1]`` exactly as training does. The default evaluation
draws five expanded-Seamless episodes and five co-speech replay episodes from
the exact post-training indices.
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hri_mot.infer_hri_mot import (  # noqa: E402
    action_boundary_stats,
    build_wrapper,
    episode_rows_from_metadata,
    load_base_model_and_tokenizer,
    load_inference_norm_stats,
    load_train_args,
    emotion_inference_config,
    rank_metrics_path,
    resolve_device,
    resolve_inference_train_args,
    resolve_rank_info,
    write_episode_audio_wav,
    write_metrics,
    write_prediction_pkl,
)
from scripts.hri_mot.train_hri_mot import (  # noqa: E402
    ActionStateNormalizer,
    build_temporal_context_batch,
    encode_audio_feature_batch,
    encode_episode_audio_feature_batch,
    load_trainable_state,
)
from scripts.hri_mot.train_hri_mot_posttrain import (  # noqa: E402
    DEFAULT_COSPEECH_INDEX,
    DEFAULT_POSTTRAIN_INDEX,
    shifted_action_phase_masks,
)
from src.audiointeraction.hri_mot.data_config import (  # noqa: E402
    DEFAULT_TEXT_CHUNK_TIME_MODE,
    HriMotDataConfig,
    TEXT_CHUNK_TIME_MODES,
    VALID_QPOS_DIMS,
)
from src.audiointeraction.hri_mot.cospeech_episode_dataset import (  # noqa: E402
    DEFAULT_IDLE_MOTION_PATH,
    CospeechEpisodeDataset,
)
from src.audiointeraction.hri_mot.emotion_conditioning import (  # noqa: E402
    NO_EMOTION_ID,
    emotion_name,
)
from src.audiointeraction.hri_mot.episode_dataset import (  # noqa: E402
    _read_audio_file,
    _resample_1d,
)
from src.audiointeraction.hri_mot.motion_smoothing import (  # noqa: E402
    action_motion_stats,
    smooth_action_chunks_offline,
)
from src.audiointeraction.hri_mot.response_timing import (  # noqa: E402
    response_elapsed_chunks_from_labels,
)
from src.audiointeraction.hri_mot.video_postprocess import (  # noqa: E402
    add_video_postprocess_args,
    mux_interaction_audio,
    normalize_prediction_video_duration,
    render_prediction_video,
    stack_prediction_left_gt_right,
)
from src.audiointeraction.hri_mot.posttrain_episode_dataset import (  # noqa: E402
    PostTrainEpisodeDataset,
)
from src.audiointeraction.tokenizer import Tokenizer  # noqa: E402
from utils import load_audio_encoder  # noqa: E402


DEFAULT_TEACHER_FORCING_CHECKPOINT = "checkpoints/stage2/final.pt"
DEFAULT_BASE_YAW_DEG = 90.0
ROBOT_STATE_MODES = ("initial_gt_then_rollout", "all_gt_state")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HRI-MoT on training episodes with GT audio/text history. "
            "Robot state can use either only the initial GT state followed by "
            "predicted-state feedback, or the GT state at every decision."
        )
    )
    parser.add_argument("--checkpoint", default=DEFAULT_TEACHER_FORCING_CHECKPOINT)
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Directory containing args.json/norm_stats.json. Defaults to checkpoint parent.",
    )
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--expanded-index", default=DEFAULT_POSTTRAIN_INDEX)
    parser.add_argument("--expanded-index-cache", default=None)
    parser.add_argument("--cospeech-index", default=DEFAULT_COSPEECH_INDEX)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <train-dir>/teacher_forcing_train_inference/<checkpoint-name>.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes-per-source", type=int, default=5)
    parser.add_argument(
        "--episode-selection",
        choices=("random", "first"),
        default="random",
        help="Random selection is reproducible with --seed.",
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=0,
        help=(
            "Number of action decisions. 0 uses every training-valid decision "
            "up to the final valid co-speech action frame."
        ),
    )
    parser.add_argument("--max-total-seq-len", type=int, default=None)
    parser.add_argument(
        "--speak-conditioning",
        choices=("auto", "none", "predicted", "teacher_forced"),
        default="auto",
        help="auto restores the architecture recorded in the checkpoint.",
    )
    parser.add_argument(
        "--text-chunk-time-mode",
        choices=TEXT_CHUNK_TIME_MODES,
        default=None,
        help="Defaults to the value saved in training args.",
    )
    parser.add_argument(
        "--audio-loader",
        choices=("whisper", "native"),
        default=None,
        help="Defaults to the value saved in training args.",
    )
    parser.add_argument(
        "--audio-feature-mode",
        choices=("episode_contiguous", "independent_chunk"),
        default=None,
        help="Defaults to the value saved in training args.",
    )
    parser.add_argument("--norm-stats", default=None)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--solver", choices=("euler", "heun"), default="euler")
    parser.add_argument("--initial-noise", choices=("random", "zeros"), default="random")
    parser.add_argument(
        "--robot-state-mode",
        choices=ROBOT_STATE_MODES,
        default="initial_gt_then_rollout",
        help=(
            "initial_gt_then_rollout uses GT state only for decision 0 and then "
            "feeds back each raw predicted action. all_gt_state matches training "
            "by using sample['robot_state'][c] for every decision."
        ),
    )
    parser.add_argument("--no-audio-features", action="store_true")
    parser.add_argument(
        "--allow-missing-files",
        action="store_true",
        help="Return padded zeros for missing files. Strict checking is enabled by default.",
    )
    parser.add_argument("--c2-bridge-frames", type=int, default=6)
    parser.add_argument("--offline-lowpass-cutoff-hz", type=float, default=4.0)
    parser.add_argument(
        "--default-base-yaw-deg",
        type=float,
        default=DEFAULT_BASE_YAW_DEG,
        help=(
            "Fixed chassis yaw used for both prediction and GT render PKLs. "
            "The default 90 degrees matches standalone WAV inference and the "
            "retarget-data front-facing convention."
        ),
    )
    parser.add_argument(
        "--target-audio-loudness-lufs",
        type=float,
        default=-16.0,
        help=(
            "Normalize the target-only MP4 audio to this integrated loudness. "
            "FFmpeg loudnorm accepts values from -70 to -5 LUFS."
        ),
    )
    parser.add_argument(
        "--target-audio-codec",
        choices=("mp3", "aac"),
        default="aac",
        help=(
            "Audio codec stored in output MP4 files. AAC is the default for "
            "standard MP4 and macOS playback; MP3 remains available as an override."
        ),
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--local-rank", type=int, default=None)
    parser.add_argument("--metrics-wait-timeout", type=int, default=3600)
    add_video_postprocess_args(parser, allow_gt_override=False)
    args = parser.parse_args()
    if args.episodes_per_source <= 0:
        parser.error("--episodes-per-source must be > 0")
    if args.num_chunks < 0:
        parser.error("--num-chunks must be >= 0")
    if args.sampling_steps <= 0:
        parser.error("--sampling-steps must be > 0")
    if not math.isfinite(float(args.default_base_yaw_deg)):
        parser.error("--default-base-yaw-deg must be finite")
    if not -70.0 <= args.target_audio_loudness_lufs <= -5.0:
        parser.error("--target-audio-loudness-lufs must be between -70 and -5")
    return args


def apply_fixed_base_yaw_to_render_pkl(path: Path, *, yaw_deg: float) -> None:
    """Set the unpredicted chassis pose used by the motion renderer."""

    if not math.isfinite(float(yaw_deg)):
        raise ValueError(f"default base yaw must be finite, got {yaw_deg}")

    with path.open("rb") as fp:
        motion = pickle.load(fp)
    if not isinstance(motion, dict):
        raise ValueError(
            f"Prediction pkl must contain a dict, got "
            f"{type(motion).__name__}: {path}"
        )

    raw_qpos = motion.get("qpos_list")
    qpos = np.asarray(raw_qpos)
    if qpos.ndim != 2 or qpos.shape[1] < 3:
        raise ValueError(
            f"qpos_list must have shape [frames, >=3], got {qpos.shape}: {path}"
        )

    yaw_rad = math.radians(float(yaw_deg))
    qpos = qpos.copy()
    qpos[:, 2] = yaw_rad
    motion["qpos_list"] = (
        [frame.copy() for frame in qpos]
        if isinstance(raw_qpos, list)
        else qpos
    )

    raw_root_pos = motion.get("root_pos")
    root_pos = np.asarray(raw_root_pos) if raw_root_pos is not None else None
    if root_pos is None or root_pos.shape != (qpos.shape[0], 3):
        root_pos = np.zeros((qpos.shape[0], 3), dtype=qpos.dtype)
    else:
        root_pos = root_pos.copy()
    root_pos[:, :2] = qpos[:, :2]
    motion["root_pos"] = root_pos

    raw_root_rot = motion.get("root_rot")
    root_rot_dtype = (
        np.asarray(raw_root_rot).dtype
        if raw_root_rot is not None
        else qpos.dtype
    )
    root_rot = np.zeros((qpos.shape[0], 4), dtype=root_rot_dtype)
    root_rot[:, 2] = math.sin(yaw_rad * 0.5)
    root_rot[:, 3] = math.cos(yaw_rad * 0.5)
    motion["root_rot"] = root_rot

    prediction_metadata = dict(motion.get("hri_mot_prediction", {}))
    prediction_metadata["base_pose_source"] = (
        "teacher_forcing_default_fixed_yaw"
    )
    prediction_metadata["default_base_yaw_deg"] = float(yaw_deg)
    motion["hri_mot_prediction"] = prediction_metadata

    with path.open("wb") as fp:
        pickle.dump(motion, fp)


def _load_mono_audio(
    path: str,
    *,
    sample_rate: int,
    audio_loader: str,
    allow_missing: bool,
) -> np.ndarray:
    """Load one source waveform for MP4 playback, without changing model input."""

    audio_path = Path(str(path)).expanduser()
    if not audio_path.is_file():
        if allow_missing:
            return np.zeros(0, dtype=np.float32)
        raise FileNotFoundError(f"Missing target audio: {audio_path}")
    if audio_loader == "whisper":
        import whisper

        audio = whisper.load_audio(
            str(audio_path), sr=int(sample_rate)
        ).astype(np.float32)
    elif audio_loader == "native":
        audio, source_rate = _read_audio_file(str(audio_path))
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = _resample_1d(
            audio.reshape(-1), int(source_rate), int(sample_rate)
        )
    else:
        raise ValueError(f"Unknown audio_loader: {audio_loader}")
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _slice_audio_interval(
    audio: np.ndarray,
    *,
    start_sec: float,
    samples: int,
    sample_rate: int,
) -> np.ndarray:
    """Slice one fixed-size interval, zero-padding outside the source."""

    samples = int(samples)
    result = np.zeros(samples, dtype=np.float32)
    source_start = int(round(float(start_sec) * int(sample_rate)))
    source_end = source_start + samples
    copy_start = max(0, source_start)
    copy_end = min(int(audio.shape[0]), source_end)
    if copy_end > copy_start:
        output_start = copy_start - source_start
        result[output_start : output_start + copy_end - copy_start] = audio[
            copy_start:copy_end
        ]
    return result


def build_video_interaction_audio(
    *,
    sample: Dict[str, Any],
    source: str,
    physical_chunks: int,
    data_config: HriMotDataConfig,
    audio_loader: str,
    allow_missing: bool,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Mix partner and original target speech on the physical episode timeline.

    This waveform is for rendered videos only. The model continues to receive
    the unchanged partner-only ``sample['partner_audio']`` tensor.
    """

    physical_chunks = int(physical_chunks)
    samples_per_chunk = int(
        round(data_config.chunk_size_sec * data_config.audio_sample_rate)
    )
    available_partner_chunks = int(sample["partner_audio"].shape[0])
    if physical_chunks <= 0 or physical_chunks > available_partner_chunks:
        raise ValueError(
            f"physical_chunks must be in [1, {available_partner_chunks}], "
            f"got {physical_chunks}"
        )
    partner_chunks = sample["partner_audio"][:physical_chunks].float().cpu()
    if tuple(partner_chunks.shape) != (physical_chunks, samples_per_chunk):
        raise ValueError(
            "partner_audio chunk shape mismatch: expected "
            f"{(physical_chunks, samples_per_chunk)}, got "
            f"{tuple(partner_chunks.shape)}"
        )

    target_chunks = np.zeros(
        (physical_chunks, samples_per_chunk), dtype=np.float32
    )
    target_paths: List[str] = []
    if source == "expanded_listening":
        rows = episode_rows_from_metadata(sample["metadata"])
        if len(rows) < physical_chunks:
            raise ValueError(
                f"Expanded episode has {len(rows)} rows, needs "
                f"{physical_chunks} physical audio chunks"
            )
        audio_cache: Dict[str, np.ndarray] = {}
        for chunk_idx, row in enumerate(rows[:physical_chunks]):
            chunk = row.get("chunk") if isinstance(row.get("chunk"), dict) else {}
            target = (
                chunk.get("target")
                if isinstance(chunk.get("target"), dict)
                else {}
            )
            target_path = str(target.get("wav", ""))
            if target_path not in audio_cache:
                audio_cache[target_path] = _load_mono_audio(
                    target_path,
                    sample_rate=data_config.audio_sample_rate,
                    audio_loader=audio_loader,
                    allow_missing=allow_missing,
                )
            target_paths.append(target_path)
            chunk_start = float(
                row.get("chunk", {}).get(
                    "t_start", sample["t_start"][chunk_idx].item()
                )
            )
            target_chunks[chunk_idx] = _slice_audio_interval(
                audio_cache[target_path],
                start_sec=chunk_start,
                samples=samples_per_chunk,
                sample_rate=data_config.audio_sample_rate,
            )
        target_alignment = "source_absolute_chunk_timestamps"
    elif source == "cospeech_replay":
        metadata = sample["metadata"]
        target = (
            metadata.get("target")
            if isinstance(metadata.get("target"), dict)
            else {}
        )
        timeline = (
            metadata.get("timeline")
            if isinstance(metadata.get("timeline"), dict)
            else {}
        )
        target_path = str(target.get("audio", ""))
        target_paths.append(target_path)
        target_audio = _load_mono_audio(
            target_path,
            sample_rate=data_config.audio_sample_rate,
            audio_loader=audio_loader,
            allow_missing=allow_missing,
        )
        target_start_sample = int(
            round(
                float(timeline.get("target_start_sec", 0.0))
                * data_config.audio_sample_rate
            )
        )
        flat_target = target_chunks.reshape(-1)
        destination_start = max(0, target_start_sample)
        source_start = max(0, -target_start_sample)
        copy_samples = min(
            int(target_audio.shape[0]) - source_start,
            int(flat_target.shape[0]) - destination_start,
        )
        if copy_samples > 0:
            flat_target[
                destination_start : destination_start + copy_samples
            ] = target_audio[source_start : source_start + copy_samples]
        target_alignment = "clip_local_audio_at_timeline_target_start_sec"
    else:
        raise ValueError(f"Unknown training source: {source}")

    partner = partner_chunks.numpy().astype(np.float32, copy=False)
    mixed = partner + target_chunks
    peak_before_limit = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    mix_gain = 1.0
    if peak_before_limit > 0.99:
        mix_gain = float(0.99 / peak_before_limit)
        mixed *= mix_gain
    unique_target_paths = list(dict.fromkeys(target_paths))
    metadata = {
        "purpose": "video_playback_only_model_input_remains_partner_only",
        "source": source,
        "physical_chunks": physical_chunks,
        "duration_sec": float(
            physical_chunks * data_config.chunk_size_sec
        ),
        "partner_audio": "sample_partner_audio",
        "target_audio_paths": unique_target_paths,
        "target_alignment": target_alignment,
        "peak_before_limit": peak_before_limit,
        "mix_gain": mix_gain,
    }
    return torch.from_numpy(mixed.copy()), metadata


def _single_sample_context_batch(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Build text/audio context metadata without exposing GT action values."""
    return {
        # The temporal context builder uses only the first two shape dimensions.
        "target_action": torch.zeros_like(sample["target_action"]).unsqueeze(0),
        "target_response_state": [list(sample["target_response_state"])],
        "target_text_plain": [list(sample["target_text_plain"])],
    }


def available_rollout_chunks(sample: Dict[str, Any], speak_conditioning: str) -> int:
    """Return the number of action decisions represented in one episode."""
    action_chunks = int(sample["target_action"].shape[0])
    available = action_chunks
    if speak_conditioning == "teacher_forced":
        target_state_ids = sample.get("target_state_ids")
        if target_state_ids is None:
            raise ValueError(
                "teacher_forced SPEAK conditioning requires "
                "sample['target_state_ids']"
            )
        # Training uses target_state_ids[:, 1:prefix_chunks + 1], so the final
        # physical chunk is a label for decision C-2 rather than another decision.
        available = min(
            available,
            max(0, int(target_state_ids.shape[0]) - 1),
        )

    valid_mask = sample.get("target_action_valid_mask")
    if valid_mask is not None and available > 0:
        valid_decisions = valid_mask[:available].bool().any(dim=-1)
        valid_indices = valid_decisions.nonzero(as_tuple=False).flatten()
        if valid_indices.numel() == 0:
            return 0
        available = int(valid_indices[-1].item()) + 1
    return available


def _speak_inputs_for_chunk(
    *,
    sample: Dict[str, Any],
    speak_conditioning: str,
    use_response_time_embedding: bool,
    chunk_idx: int,
    device: torch.device,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if speak_conditioning == "none":
        return None, None, None

    speak_chunk_ids = torch.full(
        (1, 1), int(chunk_idx), device=device, dtype=torch.long
    )
    if speak_conditioning == "predicted":
        return None, speak_chunk_ids, None
    if speak_conditioning != "teacher_forced":
        raise ValueError(f"Unknown speak_conditioning: {speak_conditioning!r}")

    target_state_ids = sample.get("target_state_ids")
    label_idx = int(chunk_idx) + 1
    if target_state_ids is None or label_idx >= int(target_state_ids.shape[0]):
        raise ValueError(
            f"Missing shifted teacher-forced SPEAK label for decision chunk {chunk_idx}"
        )
    speak_labels = target_state_ids[label_idx].to(
        device=device, dtype=torch.long
    ).reshape(1, 1)
    response_elapsed_chunks = None
    if use_response_time_embedding:
        response_elapsed_chunks = response_elapsed_chunks_from_labels(
            target_state_ids.to(device=device, dtype=torch.long).reshape(1, -1)
        )[:, label_idx : label_idx + 1]
    return speak_labels, speak_chunk_ids, response_elapsed_chunks


def _action_condition_inputs_for_chunk(
    *,
    sample: Dict[str, Any],
    wrapper: torch.nn.Module,
    speak_labels: Optional[torch.Tensor],
    speak_chunk_ids: Optional[torch.Tensor],
    chunk_idx: int,
    device: torch.device,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Build the per-decision semantic emotion condition."""

    use_emotion = bool(
        getattr(wrapper, "use_emotion_conditioning", False)
    )
    if not use_emotion:
        return None, None
    if speak_labels is None or speak_chunk_ids is None:
        raise ValueError(
            "conditional action replay requires teacher-forced SPEAK labels"
        )
    speaking = bool(int(speak_labels.item()))

    sample_emotion = sample.get("target_emotion_id")
    if speaking and sample_emotion is None:
        raise ValueError(
            "Emotion teacher forcing cannot replay speaking decision "
            f"{chunk_idx}: sample lacks target_emotion_id. "
            "Use the checkpoint's emotion-tagged CoSpeech index; expanded "
            "listening episodes are not compatible with this checkpoint."
        )
    emotion_id = (
        int(sample_emotion.item())
        if speaking and sample_emotion is not None
        else int(NO_EMOTION_ID)
    )
    emotion_labels = torch.tensor(
        [[emotion_id]], device=device, dtype=torch.long
    )
    return emotion_labels, speak_chunk_ids


@torch.no_grad()
def sample_action_rollout_sequence(
    *,
    sample: Dict[str, Any],
    num_chunks: int,
    wrapper: torch.nn.Module,
    tokenizer: Tokenizer,
    normalizer: ActionStateNormalizer,
    audio_encoder: Optional[torch.nn.Module],
    audio_feature_mode: str,
    max_total_seq_len: int,
    sampling_steps: int,
    solver: str,
    initial_noise: str,
    device: torch.device,
    action_generator: torch.Generator,
    robot_state_mode: str = "initial_gt_then_rollout",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Sample actions with rollout-state or per-decision GT-state conditioning."""
    if robot_state_mode not in ROBOT_STATE_MODES:
        raise ValueError(
            f"Unknown robot_state_mode={robot_state_mode!r}; "
            f"expected one of {ROBOT_STATE_MODES}"
        )
    speak_conditioning = str(getattr(wrapper, "speak_conditioning", "none"))
    use_response_time_embedding = bool(
        getattr(wrapper, "use_response_time_embedding", False)
    )
    use_emotion_conditioning = bool(
        getattr(wrapper, "use_emotion_conditioning", False)
    )
    available_chunks = available_rollout_chunks(sample, speak_conditioning)
    num_chunks = min(int(num_chunks), available_chunks)
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if sampling_steps <= 0:
        raise ValueError(f"sampling_steps must be positive, got {sampling_steps}")

    (
        input_ids,
        context_valid_mask,
        context_chunk_ids,
        context_token_types,
        audio_positions,
    ) = build_temporal_context_batch(
        tokenizer=tokenizer,
        batch=_single_sample_context_batch(sample),
        prefix_chunks=num_chunks,
        max_context_seq_len=max_total_seq_len,
        device=device,
    )
    # -1 means truncation; -2 means audio was intentionally omitted while
    # the target/assistant response was open.
    if audio_positions[:, num_chunks - 1, 0].eq(-1).any():
        raise RuntimeError(
            f"max_total_seq_len={max_total_seq_len} cannot hold all "
            f"{num_chunks} rollout chunks"
        )

    dtype = next(wrapper.base_model.parameters()).dtype
    partner_audio = sample["partner_audio"][:num_chunks].unsqueeze(0)
    audio_features = None
    if audio_encoder is not None:
        if audio_feature_mode == "episode_contiguous":
            audio_features = encode_episode_audio_feature_batch(
                partner_audio, audio_encoder, device, dtype
            )
        elif audio_feature_mode == "independent_chunk":
            flat_audio = partner_audio.reshape(-1, partner_audio.shape[-1])
            encoded = encode_audio_feature_batch(
                flat_audio, audio_encoder, device, dtype
            )
            audio_features = encoded.view(
                1, num_chunks, encoded.shape[1], encoded.shape[2]
            )
        else:
            raise ValueError(f"Unknown audio_feature_mode: {audio_feature_mode}")

    sample_robot_state = sample["robot_state"][:num_chunks].to(
        device=device
    ).float()
    if int(sample_robot_state.shape[0]) != num_chunks:
        raise ValueError(
            f"Need {num_chunks} robot-state chunks, got "
            f"{sample_robot_state.shape[0]}"
        )
    initial_robot_state = sample_robot_state[0]
    action_horizon = int(sample["target_action"].shape[1])
    action_dim = int(sample["target_action"].shape[2])
    robot_steps = int(initial_robot_state.shape[0])
    robot_dim = int(initial_robot_state.shape[1])
    if (robot_steps, robot_dim) != (action_horizon, action_dim):
        raise ValueError(
            "Predicted action cannot be fed back as robot state: "
            f"state shape={(robot_steps, robot_dim)} "
            f"action shape={(action_horizon, action_dim)}"
        )

    context_len = int(input_ids.shape[1])
    model_limit = int(
        getattr(wrapper.base_model, "max_seq_length", max_total_seq_len)
    )
    if context_len > min(model_limit, int(max_total_seq_len)):
        raise RuntimeError(
            f"Rollout context length {context_len} exceeds model/config limit "
            f"{min(model_limit, int(max_total_seq_len))}"
        )

    wrapper.clear_context_kv_cache()
    try:
        wrapper.set_context_kv_cache(
            batch_size=1,
            max_seq_length=context_len,
            device=device,
            dtype=dtype,
        )
        wrapper.append_context_kv_cache(
            input_ids=input_ids,
            input_pos=torch.arange(
                context_len, device=device, dtype=torch.long
            ),
            input_pos_maxp1=context_len,
            audio_features=audio_features,
            audio_positions=audio_positions,
        )
        context = {
            "context_length": context_len,
            "context_valid_mask": context_valid_mask,
            "context_chunk_ids": context_chunk_ids,
            "context_token_types": context_token_types,
        }

        dt = 1.0 / float(sampling_steps)
        raw_actions: List[torch.Tensor] = []
        current_robot_state = initial_robot_state
        shifted_speak_labels: List[int] = []
        shifted_response_elapsed_chunks: List[int] = []
        shifted_emotion_labels: List[int] = []
        for chunk_idx in range(num_chunks):
            if robot_state_mode == "all_gt_state":
                current_robot_state = sample_robot_state[chunk_idx]
            normalized_robot_state = normalizer.normalize_state(
                current_robot_state.unsqueeze(0).unsqueeze(0)
            )
            action_shape = (1, 1, action_horizon, action_dim)
            if initial_noise == "zeros":
                action = torch.zeros(
                    action_shape, device=device, dtype=torch.float32
                )
            elif initial_noise == "random":
                action = torch.randn(
                    action_shape,
                    device=device,
                    dtype=torch.float32,
                    generator=action_generator,
                )
            else:
                raise ValueError(f"Unknown initial_noise: {initial_noise}")

            state_chunk_ids = torch.full(
                (1, robot_steps),
                chunk_idx,
                device=device,
                dtype=torch.long,
            )
            action_chunk_ids = torch.full(
                (1, action_horizon),
                chunk_idx,
                device=device,
                dtype=torch.long,
            )
            (
                speak_labels,
                speak_chunk_ids,
                response_elapsed_chunks,
            ) = _speak_inputs_for_chunk(
                sample=sample,
                speak_conditioning=speak_conditioning,
                use_response_time_embedding=use_response_time_embedding,
                chunk_idx=chunk_idx,
                device=device,
            )
            if speak_labels is not None:
                shifted_speak_labels.append(int(speak_labels.item()))
            if response_elapsed_chunks is not None:
                shifted_response_elapsed_chunks.append(
                    int(response_elapsed_chunks.item())
                )
            (
                emotion_labels,
                emotion_chunk_ids,
            ) = _action_condition_inputs_for_chunk(
                sample=sample,
                wrapper=wrapper,
                speak_labels=speak_labels,
                speak_chunk_ids=speak_chunk_ids,
                chunk_idx=chunk_idx,
                device=device,
            )
            if emotion_labels is not None:
                shifted_emotion_labels.append(
                    int(emotion_labels.item())
                )

            for step_idx in range(sampling_steps):
                tau_value = step_idx / float(sampling_steps)
                tau = torch.full(
                    (1, 1), tau_value, device=device, dtype=torch.float32
                )
                outputs = wrapper.forward_from_context_kv_cache(
                    robot_state=normalized_robot_state,
                    noisy_action=action,
                    tau=tau,
                    state_chunk_ids=state_chunk_ids,
                    speak_labels=speak_labels,
                    speak_chunk_ids=speak_chunk_ids,
                    emotion_labels=emotion_labels,
                    emotion_chunk_ids=emotion_chunk_ids,
                    response_elapsed_chunks=response_elapsed_chunks,
                    action_chunk_ids=action_chunk_ids,
                    **context,
                )
                velocity = outputs["action_velocity"].float().reshape_as(action)
                if solver == "heun":
                    action_euler = action + velocity * dt
                    tau_next = torch.full(
                        (1, 1),
                        min(1.0, tau_value + dt),
                        device=device,
                        dtype=torch.float32,
                    )
                    outputs_next = wrapper.forward_from_context_kv_cache(
                        robot_state=normalized_robot_state,
                        noisy_action=action_euler,
                        tau=tau_next,
                        state_chunk_ids=state_chunk_ids,
                        speak_labels=speak_labels,
                        speak_chunk_ids=speak_chunk_ids,
                        emotion_labels=emotion_labels,
                        emotion_chunk_ids=emotion_chunk_ids,
                        response_elapsed_chunks=response_elapsed_chunks,
                        action_chunk_ids=action_chunk_ids,
                        **context,
                    )
                    velocity_next = (
                        outputs_next["action_velocity"].float().reshape_as(action)
                    )
                    velocity = 0.5 * (velocity + velocity_next)
                action = action + velocity * dt

            raw_action = (
                normalizer.unnormalize_action(action)
                .squeeze(0)
                .squeeze(0)
                .float()
            )
            raw_actions.append(raw_action)
            if robot_state_mode == "initial_gt_then_rollout":
                # Do not feed offline-smoothed or GT motion back into rollout mode.
                current_robot_state = raw_action
    finally:
        wrapper.clear_context_kv_cache()

    action = torch.stack(raw_actions, dim=0)
    context_stats = {
        "context_tokens": context_len,
        "context_valid_tokens": int(context_valid_mask.sum().item()),
        "audio_tokens_visible_through_current_chunk": True,
        "teacher_forced_target_tokens_visible_strictly_before_action_chunk": True,
        "robot_state_mode": robot_state_mode,
        "ground_truth_robot_state_scope": (
            "all_decision_chunks"
            if robot_state_mode == "all_gt_state"
            else "initial_chunk_only"
        ),
        "recurrent_robot_state_source": (
            "ground_truth_robot_state_per_chunk"
            if robot_state_mode == "all_gt_state"
            else "previous_raw_predicted_action"
        ),
        "ground_truth_action_usage": "metrics_only",
        "sequential_action_rollout": (
            robot_state_mode == "initial_gt_then_rollout"
        ),
        "predicted_state_feedback": (
            robot_state_mode == "initial_gt_then_rollout"
        ),
        "offline_smoothing_used_for_state_feedback": False,
        "sequential_chunks": int(num_chunks),
        "predicted_action_physical_offset_chunks": 1,
        "speak_conditioning": speak_conditioning,
        "response_time_embedding": use_response_time_embedding,
        "emotion_conditioning": use_emotion_conditioning,
        "speak_label_timeline": (
            "ground_truth_target_state_c_plus_1"
            if speak_conditioning == "teacher_forced"
            else speak_conditioning
        ),
        "shifted_speak_label_counts": {
            "silent": shifted_speak_labels.count(0),
            "speaking": shifted_speak_labels.count(1),
        },
        "shifted_response_elapsed_chunks": shifted_response_elapsed_chunks,
        "shifted_emotion_labels": shifted_emotion_labels,
        "shifted_emotion_names": [
            emotion_name(condition_id)
            for condition_id in shifted_emotion_labels
        ],
    }
    return action, context_stats


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


def random_episode_indices(dataset: Any, count: int, seed: int) -> List[int]:
    """Select reproducible, non-repeated training episodes."""

    population = len(dataset)
    count = min(int(count), int(population))
    if count <= 0:
        return []
    return random.Random(int(seed)).sample(range(population), k=count)


def action_phase_metrics(
    *,
    raw_action: torch.Tensor,
    smoothed_action: torch.Tensor,
    target_action: torch.Tensor,
    sample: Dict[str, Any],
    source: str,
    num_chunks: int,
) -> Dict[str, Dict[str, Any]]:
    """Return valid-frame MSE split by the three post-training phases."""

    device = raw_action.device
    target_state_ids = sample["target_state_ids"].to(
        device=device, dtype=torch.long
    ).reshape(1, -1)
    partner_state_ids = sample["partner_state_ids"].to(
        device=device, dtype=torch.long
    ).reshape(1, -1)
    speaking, listening, idle = shifted_action_phase_masks(
        target_state_ids,
        partner_state_ids,
        decisions=num_chunks,
    )
    decision_masks = {
        "all_valid": torch.ones_like(speaking[0]),
        "target_speaking": speaking[0],
        "causal_listening": listening[0],
        "causal_idle": idle[0],
    }
    decision_masks["training_supervised"] = (
        listening[0]
        if source == "expanded_listening"
        else speaking[0] | idle[0]
    )

    action_valid_mask = sample.get("target_action_valid_mask")
    if action_valid_mask is None:
        valid = torch.ones(
            target_action.shape[:2], device=device, dtype=torch.bool
        )
    else:
        valid = action_valid_mask[:num_chunks].to(
            device=device, dtype=torch.bool
        )

    raw_sq = (raw_action.float() - target_action.float()).pow(2)
    smooth_sq = (smoothed_action.float() - target_action.float()).pow(2)
    result: Dict[str, Dict[str, Any]] = {}
    for name, decision_mask in decision_masks.items():
        frame_mask = valid & decision_mask[:, None]
        value_count = int(frame_mask.sum().item()) * int(target_action.shape[-1])
        if value_count > 0:
            raw_sse = float(raw_sq[frame_mask].sum().item())
            smooth_sse = float(smooth_sq[frame_mask].sum().item())
            raw_mse = raw_sse / value_count
            smooth_mse = smooth_sse / value_count
        else:
            raw_sse = 0.0
            smooth_sse = 0.0
            raw_mse = None
            smooth_mse = None
        result[name] = {
            "decision_chunks": int(decision_mask.sum().item()),
            "valid_frames": int(frame_mask.sum().item()),
            "values": value_count,
            "raw_sse": raw_sse,
            "smoothed_sse": smooth_sse,
            "raw_mse": raw_mse,
            "smoothed_mse": smooth_mse,
        }
    return result


def aggregate_phase_metrics(
    episodes: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Aggregate phase SSEs without averaging per-episode MSEs."""

    totals: Dict[str, Dict[str, Dict[str, float]]] = {}
    for episode in episodes:
        source = str(episode["source"])
        source_totals = totals.setdefault(source, {})
        for phase, values in episode["phase_metrics"].items():
            phase_totals = source_totals.setdefault(
                phase,
                {"values": 0.0, "raw_sse": 0.0, "smoothed_sse": 0.0},
            )
            phase_totals["values"] += float(values["values"])
            phase_totals["raw_sse"] += float(values["raw_sse"])
            phase_totals["smoothed_sse"] += float(values["smoothed_sse"])

    result: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for source, phases in totals.items():
        result[source] = {}
        for phase, values in phases.items():
            count = int(values["values"])
            result[source][phase] = {
                "values": count,
                "raw_mse": (
                    float(values["raw_sse"] / count) if count > 0 else None
                ),
                "smoothed_mse": (
                    float(values["smoothed_sse"] / count)
                    if count > 0
                    else None
                ),
            }
    return result


def render_teacher_forcing_comparison(
    *,
    prediction_pkl: Path,
    ground_truth_pkl: Path,
    interaction_audio_wav: Path,
    expected_duration_sec: float,
    audio_offset_sec: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Render prediction and generated GT PKLs, then stack them side by side."""

    result: Dict[str, Any] = {
        "status": "skipped" if args.skip_video_postprocess else "pending",
        "layout": "prediction_left_gt_right",
        "prediction_video": None,
        "ground_truth_video": None,
        "comparison_video": None,
    }
    if args.skip_video_postprocess:
        return result
    try:
        prediction_video = render_prediction_video(
            prediction_pkl=prediction_pkl,
            render_script=Path(args.render_script),
            render_conda_env=str(args.render_conda_env),
            render_xml_path=(Path(args.render_xml_path) if args.render_xml_path else None),
        )
        ground_truth_video = render_prediction_video(
            prediction_pkl=ground_truth_pkl,
            render_script=Path(args.render_script),
            render_conda_env=str(args.render_conda_env),
            render_xml_path=(Path(args.render_xml_path) if args.render_xml_path else None),
        )
        prediction_duration = normalize_prediction_video_duration(
            prediction_video=prediction_video,
            expected_duration_sec=expected_duration_sec,
        )
        ground_truth_duration = normalize_prediction_video_duration(
            prediction_video=ground_truth_video,
            expected_duration_sec=expected_duration_sec,
        )
        comparison_video = prediction_pkl.with_name(
            f"{prediction_pkl.stem}__predict-left_gt-right.mp4"
        )
        stack_prediction_left_gt_right(
            prediction_video=prediction_video,
            gt_clip_video=ground_truth_video,
            output_path=comparison_video,
        )
        prediction_audio = mux_interaction_audio(
            video_path=prediction_video,
            interaction_audio=interaction_audio_wav,
            audio_start_sec=audio_offset_sec,
            loudness_lufs=float(args.target_audio_loudness_lufs),
            audio_codec=str(args.target_audio_codec),
        )
        ground_truth_audio = mux_interaction_audio(
            video_path=ground_truth_video,
            interaction_audio=interaction_audio_wav,
            audio_start_sec=audio_offset_sec,
            loudness_lufs=float(args.target_audio_loudness_lufs),
            audio_codec=str(args.target_audio_codec),
        )
        comparison_audio = mux_interaction_audio(
            video_path=comparison_video,
            interaction_audio=interaction_audio_wav,
            audio_start_sec=audio_offset_sec,
            loudness_lufs=float(args.target_audio_loudness_lufs),
            audio_codec=str(args.target_audio_codec),
        )
        result.update(
            {
                "status": "comparison_created",
                "prediction_video": str(prediction_video),
                "ground_truth_video": str(ground_truth_video),
                "comparison_video": str(comparison_video),
                "prediction_duration": prediction_duration,
                "ground_truth_duration": ground_truth_duration,
                "prediction_audio_mux": prediction_audio,
                "ground_truth_audio_mux": ground_truth_audio,
                "comparison_audio_mux": comparison_audio,
            }
        )
    except Exception as exc:
        if args.video_postprocess_strict:
            raise
        result.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        print(
            f"[video][warn] teacher-forcing comparison failed: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
    return result


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = resolve_rank_info(args)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    train_dir = (
        Path(args.train_dir).expanduser().resolve()
        if args.train_dir
        else checkpoint.parent
    )
    file_train_args = load_train_args(train_dir)
    train_args, speak_conditioning_source = resolve_inference_train_args(
        checkpoint=checkpoint,
        file_train_args=file_train_args,
        requested_speak_conditioning=args.speak_conditioning,
    )
    use_emotion_conditioning = bool(
        train_args.get("use_emotion_conditioning", False)
    )
    resolved_cospeech_index = str(args.cospeech_index)
    if (
        use_emotion_conditioning
        and str(args.cospeech_index) == str(DEFAULT_COSPEECH_INDEX)
    ):
        checkpoint_cospeech_index = (
            train_args.get("cospeech_train_index")
            or train_args.get("train_index")
        )
        if checkpoint_cospeech_index:
            resolved_cospeech_index = str(checkpoint_cospeech_index)

    checkpoint_dir = (
        args.checkpoint_dir or train_args.get("checkpoint_dir") or "checkpoints"
    )
    seed = int(
        args.seed if args.seed is not None else train_args.get("seed", 1337)
    )
    rank_seed = seed + rank
    _seed_everything(rank_seed)
    text_mode = str(
        args.text_chunk_time_mode
        or train_args.get("text_chunk_time_mode")
        or DEFAULT_TEXT_CHUNK_TIME_MODE
    )
    audio_loader = str(
        args.audio_loader or train_args.get("audio_loader") or "whisper"
    )
    audio_feature_mode = str(
        args.audio_feature_mode
        or train_args.get("audio_feature_mode")
        or "episode_contiguous"
    )
    max_total_seq_len = int(
        args.max_total_seq_len
        or train_args.get("max_total_seq_len", 4096)
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (
            train_dir / "teacher_forcing_train_inference" / checkpoint.stem
            / str(args.robot_state_mode)
            if args.robot_state_mode == "all_gt_state"
            else train_dir / "teacher_forcing_train_inference" / checkpoint.stem
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device, local_rank=local_rank)
    action_rng_seed = rank_seed + 1
    action_generator = torch.Generator(device=device).manual_seed(
        action_rng_seed
    )

    print(
        f"[rank {rank}/{world_size}] mode: training-set teacher-forced-context "
        f"robot_state={args.robot_state_mode}",
        flush=True,
    )
    print(f"[rank {rank}/{world_size}] checkpoint: {checkpoint}", flush=True)
    print(f"[rank {rank}/{world_size}] output dir: {output_dir}", flush=True)
    print(f"[rank {rank}/{world_size}] device: {device}", flush=True)
    print(
        f"[rank {rank}/{world_size}] selection: mode={args.episode_selection} "
        f"episodes_per_source={args.episodes_per_source} seed={seed}",
        flush=True,
    )
    print(
        f"[rank {rank}/{world_size}] training parity: text_mode={text_mode} "
        f"audio_loader={audio_loader} "
        f"audio_feature_mode={audio_feature_mode} "
        f"speak_conditioning={train_args['speak_conditioning']} "
        f"emotion={use_emotion_conditioning} "
        f"source={speak_conditioning_source}",
        flush=True,
    )
    state_timeline = (
        "GTState[c]"
        if args.robot_state_mode == "all_gt_state"
        else "(GTState[0] if c=0 else RawAction[c-1])"
    )
    print(
        f"[rank {rank}/{world_size}] timeline: Audio[0:c] + GTText[0:c] + "
        f"{state_timeline} + SpeakLabel[c+1] -> Action[c]",
        flush=True,
    )

    expanded_config = HriMotDataConfig(
        train_index_paths=[args.expanded_index],
        text_chunk_time_mode=text_mode,
        audio_loader=audio_loader,
        max_episodes=None,
        strict_missing_files=not bool(args.allow_missing_files),
        include_partner_state=True,
        include_text_fields=True,
    )
    cospeech_config = HriMotDataConfig(
        train_index_paths=[resolved_cospeech_index],
        text_chunk_time_mode=text_mode,
        audio_loader=audio_loader,
        max_episodes=None,
        strict_missing_files=not bool(args.allow_missing_files),
        include_partner_state=True,
        include_text_fields=True,
    )
    expanded_index_cache = (
        args.expanded_index_cache
        or train_args.get("episode_index_cache")
        or f"{args.expanded_index}.hri_mot_posttrain_offsets.jsonl"
    )
    # The emotion-conditioned checkpoint was trained only on emotion-tagged
    # CoSpeech. Expanded-listening samples have no semantic emotion and cannot
    # satisfy the required semantic emotion condition for speaking chunks.
    expanded_dataset = None
    if not use_emotion_conditioning:
        expanded_dataset = PostTrainEpisodeDataset(
            [args.expanded_index],
            config=expanded_config,
            split="train",
            episode_index_cache=str(expanded_index_cache),
        )
    cospeech_dataset = CospeechEpisodeDataset(
        [resolved_cospeech_index],
        config=cospeech_config,
        split="train",
        idle_motion_path=str(
            train_args.get("idle_motion_pkl") or DEFAULT_IDLE_MOTION_PATH
        ),
        idle_duration_sec=float(
            train_args.get("idle_motion_duration_sec", 9.0)
        ),
        transition_sec=float(train_args.get("motion_transition_sec", 0.4)),
        use_emotion_conditioning=(
            use_emotion_conditioning
        ),
    )
    if len(cospeech_dataset) == 0:
        raise RuntimeError("Co-speech training dataset must be non-empty")
    if expanded_dataset is not None and len(expanded_dataset) == 0:
        raise RuntimeError("Expanded training dataset must be non-empty")
    if args.episode_selection == "first":
        expanded_indices = (
            list(range(min(args.episodes_per_source, len(expanded_dataset))))
            if expanded_dataset is not None
            else []
        )
        cospeech_indices = list(
            range(min(args.episodes_per_source, len(cospeech_dataset)))
        )
    else:
        expanded_indices = (
            random_episode_indices(
                expanded_dataset, args.episodes_per_source, seed
            )
            if expanded_dataset is not None
            else []
        )
        cospeech_indices = random_episode_indices(
            cospeech_dataset, args.episodes_per_source, seed + 1
        )
    selected_items = [
        ("expanded_listening", expanded_dataset, index)
        for index in expanded_indices
    ] + [
        ("cospeech_replay", cospeech_dataset, index)
        for index in cospeech_indices
    ]
    shard_items = selected_items[rank::world_size]
    print(
        f"[rank {rank}/{world_size}] expanded="
        f"{len(expanded_dataset) if expanded_dataset is not None else 0} "
        f"selected={len(expanded_indices)}; cospeech={len(cospeech_dataset)} "
        f"selected={len(cospeech_indices)}; shard={len(shard_items)}",
        flush=True,
    )

    norm_stats = load_inference_norm_stats(
        checkpoint=checkpoint,
        train_dir=train_dir,
        norm_stats_path=args.norm_stats,
    )
    normalizer = ActionStateNormalizer(norm_stats, device=device)
    (
        base_model,
        tokenizer,
        qwen_omni_ckpt,
        audio_tower_ckpt,
    ) = load_base_model_and_tokenizer(checkpoint_dir, device, seed)
    use_audio_features = not (
        bool(args.no_audio_features)
        or bool(train_args.get("no_audio_features", False))
    )
    audio_encoder = None
    if use_audio_features:
        audio_encoder = load_audio_encoder(
            qwen_omni_ckpt, audio_tower_ckpt, device
        )
        audio_encoder.eval()
    wrapper = build_wrapper(
        base_model=base_model,
        data_config=expanded_config,
        train_args=train_args,
        device=device,
    )
    checkpoint_step, missing = load_trainable_state(
        wrapper, str(checkpoint), device
    )
    wrapper.eval()
    wrapper.base_model.eval()
    print(
        f"[rank {rank}/{world_size}] loaded step={checkpoint_step} "
        f"missing_trainable_keys={missing} "
        f"audio_features={use_audio_features}",
        flush=True,
    )

    config_payload: Dict[str, Any] = {
        "mode": (
            "training_set_teacher_forced_context_all_gt_state"
            if args.robot_state_mode == "all_gt_state"
            else "training_set_teacher_forced_context_predicted_state_rollout"
        ),
        "robot_state_mode": str(args.robot_state_mode),
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(checkpoint_step),
        "expanded_index": str(args.expanded_index),
        "expanded_index_cache": str(expanded_index_cache),
        "cospeech_index": str(resolved_cospeech_index),
        "cospeech_index_source": (
            "checkpoint_train_args"
            if resolved_cospeech_index != str(args.cospeech_index)
            else "cli_or_default"
        ),
        "expanded_source_compatible": not use_emotion_conditioning,
        "episode_selection": str(args.episode_selection),
        "selection_seed": int(seed),
        "episodes_per_source": int(args.episodes_per_source),
        "selected": {
            "expanded_listening": [
                {
                    "dataset_index": int(index),
                    "episode_id": str(
                        expanded_dataset.episodes[index].get("episode_id", "")
                    ),
                }
                for index in expanded_indices
            ],
            "cospeech_replay": [
                {
                    "dataset_index": int(index),
                    "episode_id": str(
                        cospeech_dataset.records[index].get("episode_id", "")
                    ),
                }
                for index in cospeech_indices
            ],
        },
        "num_chunks_requested": int(args.num_chunks),
        "sampling_steps": int(args.sampling_steps),
        "solver": str(args.solver),
        "initial_noise": str(args.initial_noise),
        "action_rng_seed": int(action_rng_seed),
        "max_total_seq_len": int(max_total_seq_len),
        "text_chunk_time_mode": text_mode,
        "audio_loader": audio_loader,
        "audio_feature_mode": (
            audio_feature_mode if use_audio_features else "none"
        ),
        "audio_preprocess": "none_exact_training_dataset",
        "model_input_audio": "partner_only_exact_training_dataset",
        "video_playback_audio": (
            "partner_plus_original_target_aligned_on_physical_timeline"
        ),
        "speak_conditioning": str(wrapper.speak_conditioning),
        "speak_conditioning_source": speak_conditioning_source,
        **emotion_inference_config(wrapper),
        "conditioning": {
            "partner_audio": "chunks_0_through_c",
            "target_control_text": "ground_truth_chunks_0_through_c",
            "initial_robot_state": "ground_truth_chunk_0",
            "recurrent_robot_state": (
                "ground_truth_robot_state_per_chunk"
                if args.robot_state_mode == "all_gt_state"
                else "previous_raw_predicted_action"
            ),
            "speak_label": (
                "ground_truth_target_state_c_plus_1"
                if wrapper.speak_conditioning == "teacher_forced"
                else str(wrapper.speak_conditioning)
            ),
            "predicted_action": (
                "target_action_array_c_physical_chunk_c_plus_1"
            ),
            "ground_truth_action": "metrics_only",
            "generated_text": False,
            "action_rollout": True,
            "offline_smoothing_used_for_state_feedback": False,
        },
        "teacher_forced_speak_max_decisions_for_50_chunks": 49,
        "c2_bridge_frames": int(args.c2_bridge_frames),
        "default_base_yaw_deg": float(args.default_base_yaw_deg),
        "offline_lowpass_cutoff_hz": float(
            args.offline_lowpass_cutoff_hz
        ),
        "model_input_audio_loudness_lufs": float(
            args.target_audio_loudness_lufs
        ),
        "target_audio_true_peak_limit_dbtp": -1.5,
        "target_audio_codec": str(args.target_audio_codec),
        "video_postprocess": not bool(args.skip_video_postprocess),
        "video_postprocess_strict": bool(
            args.video_postprocess_strict
        ),
        "render_script": str(
            Path(args.render_script).expanduser().resolve()
        ),
        "render_conda_env": str(args.render_conda_env),
        "render_xml_path": (
            str(Path(args.render_xml_path).expanduser().resolve())
            if args.render_xml_path
            else None
        ),
        "ground_truth_video_source": "rendered_from_dataset_target_action",
        "comparison_layout": "prediction_left_gt_right",
        "robot_valid_qpos_indices": list(map(int, VALID_QPOS_DIMS)),
    }
    if rank == 0:
        write_metrics(
            output_dir / "inference_config.json", config_payload
        )

    episode_metrics: List[Dict[str, Any]] = []
    progress = tqdm(
        shard_items,
        desc=f"hri-mot train-fit rollout rank {rank}",
        disable=(
            args.disable_tqdm
            or not sys.stderr.isatty()
            or rank != 0
        ),
    )
    for source, dataset, dataset_idx in progress:
        sample = dataset[dataset_idx]
        data_config = dataset.config
        available_chunks = available_rollout_chunks(
            sample, str(wrapper.speak_conditioning)
        )
        actual_chunks = (
            available_chunks
            if args.num_chunks <= 0
            else min(int(args.num_chunks), available_chunks)
        )
        if actual_chunks <= 0:
            raise RuntimeError(
                f"Episode {sample['episode_id']} has no valid decisions"
            )

        episode_id = str(sample["episode_id"])
        t_start = sample["t_start"][:actual_chunks]
        audio_t_start = float(t_start[0].item())
        audio_t_end = float(
            t_start[-1].item() + data_config.chunk_size_sec
        )
        action_t_start = float(
            audio_t_start + data_config.chunk_size_sec
        )
        action_t_end = float(
            audio_t_end + data_config.chunk_size_sec
        )
        output_duration_sec = float(actual_chunks * data_config.chunk_size_sec)

        raw_action, context_stats = sample_action_rollout_sequence(
            sample=sample,
            num_chunks=actual_chunks,
            wrapper=wrapper,
            tokenizer=tokenizer,
            normalizer=normalizer,
            audio_encoder=audio_encoder,
            audio_feature_mode=audio_feature_mode,
            max_total_seq_len=max_total_seq_len,
            sampling_steps=args.sampling_steps,
            solver=args.solver,
            initial_noise=args.initial_noise,
            device=device,
            action_generator=action_generator,
            robot_state_mode=str(args.robot_state_mode),
        )
        target_action = sample["target_action"][:actual_chunks].to(
            device=device
        ).float()
        raw_boundary = action_boundary_stats(raw_action)
        raw_motion = action_motion_stats(raw_action)
        pred_action = smooth_action_chunks_offline(
            raw_action,
            bridge_frames=int(args.c2_bridge_frames),
            lowpass_cutoff_hz=float(
                args.offline_lowpass_cutoff_hz
            ),
            fps=float(data_config.robot_fps),
        )
        boundary = action_boundary_stats(pred_action)
        motion = action_motion_stats(pred_action)
        phase_metrics = action_phase_metrics(
            raw_action=raw_action,
            smoothed_action=pred_action,
            target_action=target_action,
            sample=sample,
            source=source,
            num_chunks=actual_chunks,
        )
        all_valid = phase_metrics["all_valid"]
        raw_mse = all_valid["raw_mse"]
        mse = all_valid["smoothed_mse"]

        source_output_dir = output_dir / source
        out_pkl = write_prediction_pkl(
            original_pkl=None,
            output_root=source_output_dir / "prediction",
            episode_id=episode_id,
            audio_t_start=audio_t_start,
            audio_t_end=audio_t_end,
            action_t_start=action_t_start,
            action_t_end=action_t_end,
            pred_action=pred_action.detach().cpu().numpy(),
            data_config=data_config,
        )
        ground_truth_pkl = write_prediction_pkl(
            original_pkl=None,
            output_root=source_output_dir / "ground_truth",
            episode_id=f"{episode_id}__ground_truth",
            audio_t_start=audio_t_start,
            audio_t_end=audio_t_end,
            action_t_start=action_t_start,
            action_t_end=action_t_end,
            pred_action=target_action.detach().cpu().numpy(),
            data_config=data_config,
        )
        # Chassis pose is outside the 20-D learned action.  Both generated
        # PKLs have no source template, so explicitly use the same front-facing
        # convention as standalone WAV inference instead of the zero-yaw
        # fallback (which renders the robot facing left).
        apply_fixed_base_yaw_to_render_pkl(
            out_pkl,
            yaw_deg=float(args.default_base_yaw_deg),
        )
        apply_fixed_base_yaw_to_render_pkl(
            ground_truth_pkl,
            yaw_deg=float(args.default_base_yaw_deg),
        )
        input_audio_wav = write_episode_audio_wav(
            audio_chunks=sample["partner_audio"][:actual_chunks],
            output_root=source_output_dir,
            episode_id=episode_id,
            method=(
                f"action_rollout_{audio_loader}_"
                f"{audio_feature_mode}"
            ),
            audio_t_start=audio_t_start,
            audio_t_end=audio_t_end,
            sample_rate=data_config.audio_sample_rate,
        )
        # The video begins at action interval I_1. Build playback audio over
        # physical intervals I_0..I_N, then trim I_0 during muxing so speech
        # remains aligned with the displayed I_1..I_N motion. Model inference
        # still uses only partner chunks I_0..I_(N-1) above.
        video_audio_chunks = min(
            actual_chunks + 1,
            int(sample["partner_audio"].shape[0]),
        )
        interaction_audio, interaction_audio_metadata = (
            build_video_interaction_audio(
                sample=sample,
                source=source,
                physical_chunks=video_audio_chunks,
                data_config=data_config,
                audio_loader=audio_loader,
                allow_missing=bool(args.allow_missing_files),
            )
        )
        interaction_audio_t_end = float(
            audio_t_start
            + video_audio_chunks * data_config.chunk_size_sec
        )
        interaction_audio_wav = write_episode_audio_wav(
            audio_chunks=interaction_audio,
            output_root=source_output_dir,
            episode_id=episode_id,
            method="video_interaction_partner_plus_target",
            audio_t_start=audio_t_start,
            audio_t_end=interaction_audio_t_end,
            sample_rate=data_config.audio_sample_rate,
        )
        interaction_audio_metadata["wav"] = str(interaction_audio_wav)
        interaction_audio_metadata["mux_trim_sec"] = float(
            data_config.chunk_size_sec
        )
        video_outputs = render_teacher_forcing_comparison(
            prediction_pkl=out_pkl,
            ground_truth_pkl=ground_truth_pkl,
            interaction_audio_wav=interaction_audio_wav,
            expected_duration_sec=output_duration_sec,
            audio_offset_sec=float(data_config.chunk_size_sec),
            args=args,
        )

        states = [
            str(value)
            for value in sample["target_response_state"][
                1 : actual_chunks + 1
            ]
        ]
        state_counts = {
            state: states.count(state)
            for state in sorted(set(states))
        }
        episode_result = {
            "episode_id": episode_id,
            "source": source,
            "dataset_index": int(dataset_idx),
            "chunks": int(actual_chunks),
            "frames": int(
                actual_chunks * data_config.action_horizon
            ),
            "output_duration_sec": output_duration_sec,
            "audio_t_start": audio_t_start,
            "audio_t_end": audio_t_end,
            "action_t_start": action_t_start,
            "action_t_end": action_t_end,
            "raw_mse": raw_mse,
            "mse": mse,
            "phase_metrics": phase_metrics,
            "raw_boundary_stats": raw_boundary,
            "smoothed_boundary_stats": boundary,
            "raw_motion_stats": raw_motion,
            "smoothed_motion_stats": motion,
            "ground_truth_target_state_counts": state_counts,
            "context": context_stats,
            "prediction_pkl": str(out_pkl),
            "ground_truth_pkl": str(ground_truth_pkl),
            "model_input_wav": str(input_audio_wav),
            "interaction_audio_wav": str(interaction_audio_wav),
            "interaction_audio": interaction_audio_metadata,
            "video_outputs": video_outputs,
        }
        episode_metrics.append(episode_result)
        progress.set_postfix(
            mse=f"{float(mse):.6f}" if mse is not None else "n/a",
            raw=(f"{float(raw_mse):.6f}" if raw_mse is not None else "n/a"),
        )
        print(
            f"[rank {rank}/{world_size}] source={source} episode={episode_id} "
            f"chunks={actual_chunks} "
            f"duration={output_duration_sec:.3f}s "
            f"raw_mse={float(raw_mse):.8f} "
            f"smoothed_mse={float(mse):.8f} pkl={out_pkl}",
            flush=True,
        )

    metrics = {
        **config_payload,
        "rank": int(rank),
        "world_size": int(world_size),
        "num_episodes": len(episode_metrics),
        "aggregate_phase_metrics": aggregate_phase_metrics(episode_metrics),
        "episodes": episode_metrics,
    }
    metrics_path = rank_metrics_path(
        output_dir, rank, world_size
    )
    write_metrics(metrics_path, metrics)
    print(
        f"[rank {rank}/{world_size}] aggregate phase metrics: "
        f"{metrics['aggregate_phase_metrics']}",
        flush=True,
    )
    print(
        f"[rank {rank}/{world_size}] metrics: {metrics_path}",
        flush=True,
    )
    print(
        f"[rank {rank}/{world_size}] output root: {output_dir}",
        flush=True,
    )

if __name__ == "__main__":
    main()
