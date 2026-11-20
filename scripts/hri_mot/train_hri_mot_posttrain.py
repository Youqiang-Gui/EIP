#!/usr/bin/env python3
"""HRI-MoT baseline post-training and six-class semantic emotion FiLM tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.hri_mot import train_hri_mot as base_train  # noqa: E402
from src.audiointeraction.hri_mot.cospeech_episode_dataset import (  # noqa: E402
    DEFAULT_IDLE_MOTION_PATH,
    CospeechEpisodeDataset,
    _finite_float,
)
from src.audiointeraction.hri_mot.data_config import HriMotDataConfig  # noqa: E402
from src.audiointeraction.hri_mot.episode_dataset import (  # noqa: E402
    HriMotEpisodeCollator,
    partner_state_to_id,
    response_state_to_id,
)
from src.audiointeraction.hri_mot.flow_matching import (  # noqa: E402
    sample_flow_matching_pair,
)
from src.audiointeraction.hri_mot.emotion_conditioning import (  # noqa: E402
    NUM_EMOTION_CONDITIONS,
    EMOTION_MAPPING_VERSION,
    emotion_label,
)
from src.audiointeraction.hri_mot.mot_wrapper import HriMoTWrapper  # noqa: E402
from src.audiointeraction.hri_mot.posttrain_episode_dataset import (  # noqa: E402
    PostTrainEpisodeDataset,
)
from src.audiointeraction.hri_mot.response_timing import (  # noqa: E402
    shifted_response_elapsed_chunks,
)
from utils import load_audio_encoder  # noqa: E402


DEFAULT_INIT_CHECKPOINT = None
DEFAULT_POSTTRAIN_INDEX = None
DEFAULT_COSPEECH_INDEX = None
DEFAULT_COSPEECH_EMOTION_INDEX = None
DEFAULT_EMOTION_INIT_CHECKPOINT = None
BASELINE_COSPEECH_SUBSET_INDICES_SHA256 = "40e768117543c3e6ab1ecd857c52bde6cfe3bbc892036716cf99c38437d05235"
POSTTRAIN_CODE_VERSION = "six_emotions_release_v2"
LOSS_NORMALIZATION = "separate_phase_global_active_frames_ddp"


class RepeatPaddedDataset(Dataset):
    """Pad a dataset with deterministic repeats to make every batch full."""

    def __init__(self, dataset: Dataset, padded_size: int, *, seed: int) -> None:
        source_size = len(dataset)
        padded_size = int(padded_size)
        if source_size <= 0:
            raise ValueError("Cannot pad an empty dataset")
        if padded_size < source_size:
            raise ValueError(
                f"padded_size={padded_size} is smaller than source_size={source_size}"
            )

        repeated_indices = []
        generator = torch.Generator().manual_seed(int(seed))
        while source_size + len(repeated_indices) < padded_size:
            repeated_indices.extend(
                torch.randperm(source_size, generator=generator).tolist()
            )

        self.dataset = dataset
        self.source_size = source_size
        self.indices = list(range(source_size)) + repeated_indices[
            : padded_size - source_size
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[self.indices[idx]]


def full_batch_padded_size(source_size: int, global_batch_size: int) -> int:
    source_size = int(source_size)
    global_batch_size = int(global_batch_size)
    if source_size <= 0:
        raise ValueError(f"source_size must be positive, got {source_size}")
    if global_batch_size <= 0:
        raise ValueError(
            f"global_batch_size must be positive, got {global_batch_size}"
        )
    return (
        (source_size + global_batch_size - 1)
        // global_batch_size
        * global_batch_size
    )


def merge_action_state_norm_stats(
    stats_by_source: Sequence[Dict[str, Any]],
    *,
    std_floor: float,
) -> Dict[str, Any]:
    """Pool per-source population moments without changing source weighting."""

    if not stats_by_source:
        raise ValueError("At least one normalization-stat source is required")
    if std_floor <= 0.0:
        raise ValueError(f"std_floor must be positive, got {std_floor}")

    merged: Dict[str, Any] = {
        "version": max(int(stats.get("version", 1)) for stats in stats_by_source),
        "std_floor": float(std_floor),
    }
    for prefix in ("state", "action"):
        count_key = f"{prefix}_count"
        mean_key = f"{prefix}_mean"
        std_key = f"{prefix}_std"
        counts = [int(stats.get(count_key, 0)) for stats in stats_by_source]
        if any(count <= 0 for count in counts):
            raise ValueError(
                f"All {prefix} normalization sources need positive counts, got {counts}"
            )
        means = [
            torch.as_tensor(stats[mean_key], dtype=torch.float64).cpu()
            for stats in stats_by_source
        ]
        stds = [
            torch.as_tensor(stats[std_key], dtype=torch.float64).cpu()
            for stats in stats_by_source
        ]
        expected_shape = tuple(means[0].shape)
        if any(tuple(value.shape) != expected_shape for value in means + stds):
            raise ValueError(
                f"Incompatible {prefix} normalization shapes: "
                f"{[tuple(value.shape) for value in means + stds]}"
            )

        total_count = int(sum(counts))
        total = sum(
            mean * float(count) for mean, count in zip(means, counts)
        )
        total_sq = sum(
            (std.pow(2) + mean.pow(2)) * float(count)
            for mean, std, count in zip(means, stds, counts)
        )
        mean = total / float(total_count)
        variance = (total_sq / float(total_count)) - mean.pow(2)
        merged[mean_key] = mean.float()
        merged[std_key] = (
            torch.sqrt(variance.clamp_min(0.0))
            .clamp_min(float(std_floor))
            .float()
        )
        merged[count_key] = total_count
    return merged


def deterministic_random_subset(
    dataset: Dataset,
    subset_size: int,
    *,
    seed: int,
) -> Tuple[Dataset, list[int]]:
    """Select one reproducible random subset shared by every DDP rank.

    ``subset_size=0`` disables subsetting.  A requested size larger than the
    available dataset is clipped so ``--max-episodes`` remains useful for
    small smoke tests.
    """

    source_size = len(dataset)
    subset_size = int(subset_size)
    if source_size <= 0:
        raise ValueError("Cannot subset an empty dataset")
    if subset_size < 0:
        raise ValueError(f"subset_size must be non-negative, got {subset_size}")
    if subset_size == 0 or subset_size >= source_size:
        return dataset, list(range(source_size))

    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(source_size, generator=generator)[:subset_size].tolist()
    return Subset(dataset, indices), indices


def indices_sha256(indices: list[int]) -> str:
    payload = ",".join(str(int(index)) for index in indices).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def validate_emotion_index(
    dataset: CospeechEpisodeDataset,
    indices: list[int],
    *,
    baseline_index: str,
    require_locked_subset: bool = True,
) -> str:
    actual = indices_sha256(indices)
    if require_locked_subset and actual != BASELINE_COSPEECH_SUBSET_INDICES_SHA256:
        raise ValueError(f"CoSpeech subset hash mismatch: expected {BASELINE_COSPEECH_SUBSET_INDICES_SHA256}, got {actual}")
    baseline = []
    with open(baseline_index, "r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                baseline.append(json.loads(line))
    if len(baseline) != len(dataset.records):
        raise ValueError(f"Motion/baseline row counts differ: {len(dataset.records)} vs {len(baseline)}")
    digest = hashlib.sha256()
    for index in indices:
        row = dataset.records[index]
        original = {k: v for k, v in row.items() if k != "target_emotion" and not k.startswith("_")}
        if original != baseline[index]:
            raise ValueError(f"Motion index row {index} differs from the baseline row")
        emotion_label(row.get("target_emotion"), source=f"motion row {index}")
        identity = str(original.get("episode_id", index)).encode("utf-8")
        digest.update(len(identity).to_bytes(8, "big")); digest.update(identity)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_fingerprints() -> Dict[str, str]:
    paths = {
        "posttrain_source_sha256": Path(__file__).resolve(),
        "posttrain_dataset_source_sha256": (
            ROOT / "src/audiointeraction/hri_mot/posttrain_episode_dataset.py"
        ),
        "cospeech_dataset_source_sha256": (
            ROOT / "src/audiointeraction/hri_mot/cospeech_episode_dataset.py"
        ),
        "mot_wrapper_source_sha256": ROOT / "src/audiointeraction/hri_mot/mot_wrapper.py",
        "emotion_mapping_source_sha256": (
            ROOT / "src/audiointeraction/hri_mot/emotion_conditioning.py"
        ),
        "shared_train_source_sha256": ROOT / "scripts/hri_mot/train_hri_mot.py",
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the HRI-MoT baseline from Seamless/Cospeech replay, or "
            "fine-tune six semantic emotions with multi-layer FiLM."
        )
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--init-from", default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument(
        "--emotion-init-from",
        default=DEFAULT_EMOTION_INIT_CHECKPOINT,
        help=(
            "Unconditioned baseline action checkpoint used for emotion "
            "fine-tuning. Existing action parameters are restored; the final "
            "Action Expert layers are selectively unfrozen and the emotion "
            "embedding/per-layer FiLM tables start at zero."
        ),
    )
    parser.add_argument("--train-index", default=DEFAULT_POSTTRAIN_INDEX)
    parser.add_argument("--emotion-warmstart-from", default=None,
                        help="Restore a complete six-emotion FiLM checkpoint (including condition weights). Overrides baseline initialization; optimizer and step restart.")
    parser.add_argument("--episode-index-cache", default=None)
    parser.add_argument("--cospeech-train-index", default=DEFAULT_COSPEECH_INDEX)
    parser.add_argument("--cospeech-baseline-index", default=DEFAULT_COSPEECH_INDEX)
    parser.add_argument(
        "--use-emotion-conditioning",
        type=parse_bool,
        default=False,
        help=(
            "Use six independent semantic emotion conditions plus none for silence. "
            "This mode trains from all annotated Cospeech episodes without "
            "motion filtering and does not load Seamless."
        ),
    )
    parser.add_argument(
        "--emotion-film-layers",
        type=int,
        default=4,
        help=(
            "Number of final Action Expert blocks whose action-token outputs "
            "receive independent zero-initialized six-emotion FiLM modulation."
        ),
    )
    parser.add_argument(
        "--motion-unfreeze-action-layers",
        type=int,
        default=8,
        help=(
            "Number of final Action Expert blocks to fine-tune at the scaled "
            "pretrained LR, plus the velocity head. Zero freezes these blocks "
            "and the head; response-time unfreezing is controlled separately."
        ),
    )
    parser.add_argument(
        "--motion-unfreeze-response-time-encoder",
        type=parse_bool,
        default=False,
        help=(
            "Also fine-tune the restored response_time_encoder at the scaled "
            "pretrained LR. Only supported in emotion mode."
        ),
    )
    parser.add_argument(
        "--emotion-samples-per-class",
        type=int,
        default=512,
        help=(
            "Samples per semantic emotion category per epoch: 512 by default, "
            "giving 3072 samples over six categories."
        ),
    )
    parser.add_argument("--idle-motion-pkl", default=DEFAULT_IDLE_MOTION_PATH)
    parser.add_argument("--idle-motion-duration-sec", type=float, default=9.0)
    parser.add_argument("--motion-transition-sec", type=float, default=0.4)
    parser.add_argument(
        "--norm-stats-mode",
        choices=("source", "joint_supervised"),
        default=None,
        help=(
            "source preserves normalization from --init-from; joint_supervised "
            "recomputes population statistics using only action frames selected "
            "by nonzero Seamless/Cospeech loss phases. The default is "
            "joint_supervised for baseline initialization and source for "
            "emotion fine-tuning."
        ),
    )
    parser.add_argument("--norm-std-floor", type=float, default=1e-4)
    parser.add_argument(
        "--norm-stats",
        default=None,
        help=(
            "Precomputed supervised joint norm_stats.json. Distributed training "
            "with --norm-stats-mode joint_supervised requires this file."
        ),
    )
    parser.add_argument(
        "--compute-norm-stats-only",
        action="store_true",
        help="Compute and save supervised joint norm stats, then exit before training.",
    )
    parser.add_argument("--output-dir", default="outputs/hri_mot_posttrain")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--posttrain-episode-batch-size", type=int, default=2)
    parser.add_argument("--cospeech-episode-batch-size", type=int, default=None)
    parser.add_argument(
        "--cospeech-subset-size",
        type=int,
        default=None,
        help=(
            "Fixed random co-speech subset size; 0 uses the full co-speech dataset."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-total-seq-len", type=int, default=None)
    parser.add_argument(
        "--text-chunk-time-mode",
        choices=("start", "midpoint"),
        default=None,
    )

    parser.add_argument("--listen-action-loss-weight", type=float, default=None)
    parser.add_argument("--cospeech-speaking-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--cospeech-listening-loss-weight",
        type=float,
        default=None,
    )
    parser.add_argument("--cospeech-idle-loss-weight", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=None)
    parser.add_argument(
        "--pretrained-lr-scale",
        type=float,
        default=None,
        help=(
            "Learning-rate multiplier for parameters restored from the initialization "
            "checkpoint. Emotion training applies it only to the final "
            "unfrozen Action Expert blocks and action velocity head."
        ),
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=None)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "warmup_cosine"),
        default="warmup_cosine",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--metrics-jsonl", default=None)
    parser.add_argument("--disable-tqdm", action="store_true")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="hri-mot-posttrain")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument(
        "--wandb-mode",
        default=None,
        choices=("online", "offline", "disabled"),
    )
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument("--no-audio-features", action="store_true")
    parser.add_argument(
        "--audio-loader",
        choices=("whisper", "native"),
        default="whisper",
    )
    parser.add_argument(
        "--audio-feature-mode",
        choices=("episode_contiguous", "independent_chunk"),
        default="episode_contiguous",
    )
    args = parser.parse_args()
    conditioning_finetune = bool(args.use_emotion_conditioning)
    if args.emotion_warmstart_from and not conditioning_finetune:
        parser.error("--emotion-warmstart-from requires emotion conditioning")
    if not conditioning_finetune and args.init_from is None:
        parser.error("--init-from is required as the architecture reference checkpoint")
    if args.cospeech_train_index is None:
        parser.error("--cospeech-train-index is required")
    if conditioning_finetune and not (
        args.emotion_init_from or args.emotion_warmstart_from
    ):
        parser.error(
            "Stage II requires --emotion-init-from pointing to the "
            "Stage-I checkpoint"
        )
    if not conditioning_finetune and args.train_index is None:
        parser.error("Stage I requires --train-index")

    args.init_mode = (
        "emotion_finetune"
        if args.use_emotion_conditioning
        else "random"
    )
    if args.norm_stats_mode is None:
        args.norm_stats_mode = (
            "source" if conditioning_finetune else "joint_supervised"
        )
    mode_defaults = {
        "epochs": 100 if args.use_emotion_conditioning else 60,
        "lr": 1e-4,
        "min_lr": 1e-5,
        "pretrained_lr_scale": (
            0.02 if args.use_emotion_conditioning else 1.0
        ),
        "lr_warmup_steps": (
            100 if args.use_emotion_conditioning else 500
        ),
        "save_interval": 500 if conditioning_finetune else 1000,
        "cospeech_episode_batch_size": 2,
        "cospeech_subset_size": 0,
        "cospeech_listening_loss_weight": 0.0,
        "listen_action_loss_weight": (
            0.0
            if args.use_emotion_conditioning
            else 2.0
        ),
        "cospeech_idle_loss_weight": (
            0.0
            if args.use_emotion_conditioning
            else 1.0
        ),
    }
    for name, value in mode_defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    if args.emotion_film_layers <= 0:
        parser.error("--emotion-film-layers must be positive")
    if args.motion_unfreeze_action_layers < 0:
        parser.error("--motion-unfreeze-action-layers must be nonnegative")
    if args.motion_unfreeze_response_time_encoder and not conditioning_finetune:
        parser.error("--motion-unfreeze-response-time-encoder requires emotion conditioning")
    unfreezes_source = (
        args.motion_unfreeze_action_layers > 0
        or args.motion_unfreeze_response_time_encoder
    )
    if conditioning_finetune and not unfreezes_source:
        args.pretrained_lr_scale = 0.0

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.posttrain_episode_batch_size <= 0:
        parser.error("--posttrain-episode-batch-size must be positive")
    if args.cospeech_episode_batch_size <= 0:
        parser.error("--cospeech-episode-batch-size must be positive")
    if args.cospeech_subset_size < 0:
        parser.error("--cospeech-subset-size must be non-negative")
    if args.listen_action_loss_weight < 0.0:
        parser.error("--listen-action-loss-weight must be non-negative")
    if args.cospeech_speaking_loss_weight <= 0.0:
        parser.error("--cospeech-speaking-loss-weight must be positive")
    if args.cospeech_listening_loss_weight < 0.0:
        parser.error("--cospeech-listening-loss-weight must be non-negative")
    if args.cospeech_idle_loss_weight < 0.0:
        parser.error("--cospeech-idle-loss-weight must be non-negative")
    if args.pretrained_lr_scale < 0.0:
        parser.error("--pretrained-lr-scale must be non-negative")
    if args.use_emotion_conditioning:
        if args.cospeech_subset_size != 0:
            parser.error(
                "emotion fine-tuning uses the full emotion index; "
                "--cospeech-subset-size must be 0"
            )
        if args.emotion_samples_per_class <= 0:
            parser.error("--emotion-samples-per-class must be positive")
        if args.listen_action_loss_weight != 0.0:
            parser.error(
                "Cospeech-only emotion fine-tuning does not load "
                "Seamless; --listen-action-loss-weight must be 0"
            )
        if args.cospeech_listening_loss_weight != 0.0:
            parser.error(
                "emotion fine-tuning does not supervise "
                "Cospeech listening; --cospeech-listening-loss-weight must be 0"
            )
        if args.cospeech_idle_loss_weight != 0.0:
            parser.error(
                "emotion fine-tuning does not use idle targets; "
                "--cospeech-idle-loss-weight must be 0"
            )
        if unfreezes_source and args.pretrained_lr_scale <= 0.0:
            parser.error(
                "emotion causal fine-tuning requires "
                "--pretrained-lr-scale > 0"
            )
    elif args.listen_action_loss_weight <= 0.0:
        parser.error(
            "baseline post-training requires --listen-action-loss-weight > 0"
        )
    if args.lr_warmup_steps < 0:
        parser.error("--lr-warmup-steps must be non-negative")
    if args.norm_std_floor <= 0.0:
        parser.error("--norm-std-floor must be positive")
    if conditioning_finetune and args.norm_stats_mode != "source":
        parser.error(
            "conditioning fine-tuning restores existing action weights and therefore "
            "requires --norm-stats-mode source"
        )
    if conditioning_finetune and args.norm_stats:
        parser.error(
            "emotion fine-tuning must reuse normalization embedded in "
            "--emotion-init-from; do not pass --norm-stats"
        )
    if args.compute_norm_stats_only and args.norm_stats_mode != "joint_supervised":
        parser.error(
            "--compute-norm-stats-only requires "
            "--norm-stats-mode joint_supervised"
        )
    if args.compute_norm_stats_only and args.norm_stats:
        parser.error(
            "--compute-norm-stats-only cannot be combined with --norm-stats"
        )

    # Compatibility fields consumed by the shared checkpoint/logger helpers
    # and by inference when it later reads checkpoint['args'].
    args.resume = None
    args.dataset_format = (
        "posttrain_emotion_cospeech_only"
        if args.use_emotion_conditioning
        else "posttrain_balanced_replay"
    )
    args.episode_batch_size = int(
        args.cospeech_episode_batch_size
        if args.use_emotion_conditioning
        else args.posttrain_episode_batch_size + args.cospeech_episode_batch_size
    )
    # Compatibility field for older checkpoint readers. Expanded response
    # frames are always masked; speaking supervision comes only from co-speech.
    args.response_action_loss_weight = 0.0
    args.loss_last_n_chunks = 0
    args.chunks_per_step = 0
    args.temporal_prefix_sampling = "fixed"
    args.posttrain_code_version = POSTTRAIN_CODE_VERSION
    args.loss_normalization = LOSS_NORMALIZATION
    args.initialization_weights_loaded = bool(conditioning_finetune)
    args.training_data_mode = (
        "cospeech_only"
        if args.use_emotion_conditioning
        else "seamless_cospeech"
    )
    args.action_supervision_mode = (
        "cospeech_speaking_flow_matching_only"
        if args.use_emotion_conditioning
        else "weighted_listen_speaking_listening_idle"
    )
    args.emotion_sampling_mode = (
        "uniform_six_emotions"
        if args.use_emotion_conditioning
        else None
    )
    args.emotion_mapping_version = (
        EMOTION_MAPPING_VERSION
        if args.use_emotion_conditioning
        else None
    )
    args.motion_condition_weights_restored = bool(args.emotion_warmstart_from)
    for name, fingerprint in code_fingerprints().items():
        setattr(args, name, fingerprint)
    return args


def load_checkpoint_cpu(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing initialization checkpoint: {path}")
    try:
        return torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_emotion_finetune_weights_strict(
    model: torch.nn.Module,
    checkpoint: Dict[str, Any],
    *,
    device: torch.device,
    restore_conditioning: bool = False,
) -> None:
    """Restore baseline weights, or explicitly restore a complete conditioned model."""

    params = checkpoint.get("trainable_params")
    if not isinstance(params, dict):
        raise ValueError(
            "Emotion initialization checkpoint has no trainable_params"
        )
    source_args = checkpoint.get("args", {})
    if not isinstance(source_args, dict):
        raise ValueError(
            "Emotion initialization checkpoint has no valid args"
        )
    if restore_conditioning:
        if (not source_args.get("use_emotion_conditioning", False)
                or source_args.get("emotion_mapping_version") != EMOTION_MAPPING_VERSION
                or source_args.get("emotion_film_layers") != model.emotion_film_layers
                or source_args.get("speak_conditioning") != model.speak_conditioning
                or bool(source_args.get("response_time_embedding")) != bool(model.use_response_time_embedding)):
            raise ValueError("Conditioned warm-start requires matching FiLM layers, motion mapping, SPEAK and response-time architecture")
    elif bool(source_args.get("use_emotion_conditioning", False)):
        raise ValueError(
            "--emotion-init-from must be an unconditioned baseline "
            "checkpoint"
        )

    named = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    condition_names = {
        name
        for name in named
        if name.startswith("emotion_embedding.")
        or name.startswith("emotion_layer_film.")
    }
    expected_condition_names = set()
    if getattr(model, "emotion_embedding", None) is not None:
        expected_condition_names.add("emotion_embedding.weight")
    expected_condition_names.update(
        name
        for name in named
        if name.startswith("emotion_layer_film.")
    )
    if condition_names != expected_condition_names:
        raise ValueError(
            "Unexpected new conditioning parameters: expected "
            f"{expected_condition_names}, got {condition_names}"
        )
    required_source = set(named) if restore_conditioning else set(named) - condition_names
    available = set(params)
    missing = sorted(required_source - available)
    unexpected = sorted(available - required_source)
    if missing or unexpected:
        raise ValueError(
            "Baseline/model action parameter mismatch: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )

    with torch.no_grad():
        for name in sorted(required_source):
            source = params[name]
            target = named[name]
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: baseline={tuple(source.shape)} "
                    f"model={tuple(target.shape)}"
                )
            target.copy_(source.to(device=device, dtype=target.dtype))


def optimizer_parameter_groups(
    model: torch.nn.Module,
    *,
    pretrained_lr_scale: float,
    use_emotion_conditioning: bool,
    source_optimizer_masters: Optional[Dict[str, torch.nn.Parameter]] = None,
) -> list[Dict[str, Any]]:
    """Separate new condition modules from restored action parameters."""

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Model has no trainable parameters")
    if not use_emotion_conditioning:
        return [
            {
                "params": [parameter for _, parameter in trainable],
                "lr_scale": 1.0,
                "group_name": "all_trainable",
            }
        ]

    condition_prefixes = (
        "emotion_embedding.",
        "emotion_layer_film.",
    )
    condition_parameters = [
        parameter
        for name, parameter in trainable
        if name.startswith(condition_prefixes)
    ]
    pretrained_named_parameters = [
        (name, parameter)
        for name, parameter in trainable
        if not name.startswith(condition_prefixes)
    ]
    if source_optimizer_masters is None:
        pretrained_parameters = [
            parameter for _, parameter in pretrained_named_parameters
        ]
    else:
        expected = {name for name, _ in pretrained_named_parameters}
        actual = set(source_optimizer_masters)
        if expected != actual:
            raise ValueError(
                "FP32 source optimizer master mismatch: "
                f"missing={sorted(expected - actual)[:8]} "
                f"unexpected={sorted(actual - expected)[:8]}"
            )
        pretrained_parameters = [
            source_optimizer_masters[name]
            for name, _ in pretrained_named_parameters
        ]
    if not condition_parameters:
        raise ValueError(
            "Conditioning fine-tuning model has no new embedding parameters"
        )
    if not pretrained_parameters and float(pretrained_lr_scale) > 0.0:
        raise ValueError(
            "Emotion fine-tuning model has no restored action parameters"
        )
    groups: list[Dict[str, Any]] = []
    if pretrained_parameters:
        groups.append({
            "params": pretrained_parameters,
            "lr_scale": float(pretrained_lr_scale),
            "group_name": "unfrozen_source_action",
        })
    groups.append({
            "params": condition_parameters,
            "lr_scale": 1.0,
        "group_name": "emotion_condition",
        })
    return groups


def configure_emotion_trainable_parameters(
    model: torch.nn.Module,
    *,
    restored_parameter_names: Sequence[str],
    unfreeze_action_layers: int,
    unfreeze_response_time_encoder: bool = False,
) -> tuple[str, ...]:
    """Train new conditions and explicitly selected restored action modules."""

    named = dict(model.named_parameters())
    missing = sorted(set(restored_parameter_names) - set(named))
    if missing:
        raise ValueError(
            "Cannot configure restored parameters missing from the model: "
            f"{missing[:8]}"
        )
    action_expert = getattr(model, "action_expert", None)
    action_layers = getattr(action_expert, "layers", None)
    if action_layers is None:
        raise ValueError("Model has no action_expert.layers to selectively unfreeze")
    total_layers = len(action_layers)
    unfreeze_action_layers = int(unfreeze_action_layers)
    if not 0 <= unfreeze_action_layers <= total_layers:
        raise ValueError(
            "unfreeze_action_layers must be between 0 and "
            f"{total_layers}, got {unfreeze_action_layers}"
        )

    for name in restored_parameter_names:
        named[name].requires_grad_(False)
    first_unfrozen_layer = total_layers - unfreeze_action_layers
    selected_source_names = {
        name
        for name in restored_parameter_names
        if (unfreeze_action_layers > 0 and name.startswith("action_velocity_head."))
        or any(
            name.startswith(f"action_expert.layers.{layer_idx}.")
            for layer_idx in range(first_unfrozen_layer, total_layers)
        )
    }
    if unfreeze_action_layers > 0 and not selected_source_names:
        raise ValueError(
            "No restored parameters matched the requested final Action Expert "
            "layers or action_velocity_head"
        )
    if unfreeze_response_time_encoder:
        response_time_names = {
            name for name in restored_parameter_names
            if name.startswith("response_time_encoder.")
        }
        if getattr(model, "response_time_encoder", None) is None or not response_time_names:
            raise ValueError("Response-time unfreezing requires a restored response_time_encoder")
        selected_source_names.update(response_time_names)
    for name in selected_source_names:
        named[name].requires_grad_(True)

    trainable = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    expected_trainable = {"emotion_embedding.weight"}
    expected_trainable.update(
        name
        for name in named
        if name.startswith("emotion_layer_film.")
    )
    expected_trainable.update(selected_source_names)
    if set(trainable) != expected_trainable:
        raise ValueError(
            "Causal emotion mode must leave only its new condition "
            "modules and explicitly selected restored action modules "
            f"trainable, got {trainable}"
        )
    return tuple(sorted(selected_source_names))


def create_fp32_source_optimizer_masters(
    model: torch.nn.Module,
    parameter_names: Sequence[str],
) -> Dict[str, torch.nn.Parameter]:
    """Create persistent FP32 optimizer copies for low-LR BF16 source weights."""

    named = dict(model.named_parameters())
    missing = sorted(set(parameter_names) - set(named))
    if missing:
        raise ValueError(
            f"Cannot create optimizer masters for missing parameters: {missing[:8]}"
        )
    return {
        name: torch.nn.Parameter(
            named[name].detach().float().clone(),
            requires_grad=True,
        )
        for name in sorted(set(parameter_names))
    }


def clear_source_model_gradients(
    model: torch.nn.Module,
    source_optimizer_masters: Dict[str, torch.nn.Parameter],
) -> None:
    """Clear model-side source gradients not owned directly by the optimizer."""

    named = dict(model.named_parameters())
    for name in source_optimizer_masters:
        named[name].grad = None


def copy_source_gradients_to_masters(
    model: torch.nn.Module,
    source_optimizer_masters: Dict[str, torch.nn.Parameter],
) -> None:
    """Move synchronized model gradients into persistent FP32 optimizer copies."""

    named = dict(model.named_parameters())
    for name, master in source_optimizer_masters.items():
        gradient = named[name].grad
        master.grad = (
            None
            if gradient is None
            else gradient.detach().to(device=master.device, dtype=torch.float32)
        )


def sync_source_masters_to_model(
    model: torch.nn.Module,
    source_optimizer_masters: Dict[str, torch.nn.Parameter],
) -> None:
    """Quantize accumulated FP32 master values back into forward parameters."""

    named = dict(model.named_parameters())
    with torch.no_grad():
        for name, master in source_optimizer_masters.items():
            target = named[name]
            target.copy_(master)


def save_complete_conditioning_checkpoint(
    path: Path,
    wrapper: HriMoTWrapper,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    norm_stats: Dict[str, Any],
    *,
    parameter_names: Sequence[str],
) -> None:
    """Save the complete restored action stack plus condition modules."""

    named = dict(wrapper.named_parameters())
    requested = sorted(set(parameter_names))
    missing = sorted(set(requested) - set(named))
    if missing:
        raise ValueError(
            f"Checkpoint parameters are missing from the model: {missing[:8]}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "trainable_params": {
                name: named[name].detach().cpu()
                for name in requested
            },
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "norm_stats": base_train.norm_stats_for_checkpoint(norm_stats),
        },
        path,
    )


def set_scaled_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(base_lr) * float(group.get("lr_scale", 1.0))


def apply_source_configuration(
    args: argparse.Namespace,
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    source_args = checkpoint.get("args", {})
    if not isinstance(source_args, dict):
        raise ValueError("Initialization checkpoint has no valid args dictionary")

    args.action_hidden_size = int(source_args.get("action_hidden_size", 768))
    args.action_intermediate_size = int(
        source_args.get("action_intermediate_size", 3072)
    )
    args.speak_conditioning = str(
        source_args.get("speak_conditioning", "predicted")
    )
    args.speak_loss_weight = float(source_args.get("speak_loss_weight", 1.0))
    args.response_time_embedding = bool(
        source_args.get("response_time_embedding", False)
    )
    args.source_checkpoint_step = int(checkpoint.get("step", 0))
    args.source_dataset_format = str(source_args.get("dataset_format", "unknown"))
    if args.max_total_seq_len is None:
        args.max_total_seq_len = int(source_args.get("max_total_seq_len", 4096))
    if args.text_chunk_time_mode is None:
        args.text_chunk_time_mode = str(
            source_args.get("text_chunk_time_mode", "start")
        )

    if args.speak_conditioning != "teacher_forced":
        raise ValueError(
            "This listen-focused post-training entry requires a teacher_forced "
            f"checkpoint, got {args.speak_conditioning!r}"
        )
    if not args.response_time_embedding:
        raise ValueError(
            "The requested checkpoint must include response_time_embedding; "
            "the post-training run cannot change this architecture flag"
        )
    return source_args


def phase_weighted_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    phase_weight: torch.Tensor,
    *,
    distributed: bool = False,
    world_size: int = 1,
) -> torch.Tensor:
    """Globally normalized per-frame MSE with listen/response phase weights.

    DDP averages gradients from all ranks.  Dividing each rank by its own
    number of active frames would therefore give sparse ranks the same total
    weight as dense ranks.  In distributed mode, use the all-reduced active
    frame count and multiply the local numerator by ``world_size`` so DDP's
    later gradient average equals a true global per-frame mean.
    """
    valid_mask = valid_mask.to(device=prediction.device, dtype=torch.bool)
    phase_weight = phase_weight.to(device=prediction.device, dtype=torch.float32)
    if tuple(valid_mask.shape) != tuple(prediction.shape[:-1]):
        raise ValueError(
            f"valid_mask {tuple(valid_mask.shape)} does not match "
            f"prediction {tuple(prediction.shape)}"
        )
    if tuple(phase_weight.shape) != tuple(valid_mask.shape):
        raise ValueError(
            f"phase_weight {tuple(phase_weight.shape)} does not match "
            f"valid_mask {tuple(valid_mask.shape)}"
        )
    active_mask = valid_mask & phase_weight.gt(0)
    weight = phase_weight * active_mask.to(dtype=phase_weight.dtype)
    active_count = active_mask.sum(dtype=torch.float32)
    ddp_scale = 1.0
    if distributed:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "distributed phase loss requires an initialized process group"
            )
        actual_world_size = int(dist.get_world_size())
        if int(world_size) != actual_world_size:
            raise RuntimeError(
                "phase loss world-size mismatch: "
                f"argument={world_size} process_group={actual_world_size}"
            )
        dist.all_reduce(active_count, op=dist.ReduceOp.SUM)
        ddp_scale = float(actual_world_size)
    elif int(world_size) != 1:
        raise ValueError(
            f"world_size must be 1 when distributed=False, got {world_size}"
        )

    if float(active_count.item()) <= 0.0:
        return prediction.sum() * 0.0
    error = (prediction.float() - target.float()).pow(2)
    numerator = (error * weight.unsqueeze(-1)).sum()
    denominator = active_count * int(prediction.shape[-1])
    return numerator * ddp_scale / denominator.clamp_min(1.0)


def shifted_action_phase_masks(
    target_state_ids: torch.Tensor,
    partner_state_ids: torch.Tensor,
    *,
    decisions: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return response, causal-listening, and causal-idle masks for decision c.

    Decision ``c`` observes partner audio/state from physical interval ``I_c``
    and predicts target motion in ``I_(c+1)``.  Listening supervision therefore
    requires the target to stay silent in ``I_(c+1)`` while the already-observed
    partner is active in ``I_c``.  This matches the causal temporal convention
    used by ``train_hri_mot.py`` and streaming inference.
    """
    decisions = int(decisions)
    if decisions < 0:
        raise ValueError(f"decisions must be non-negative, got {decisions}")
    if target_state_ids.dim() != 2 or partner_state_ids.dim() != 2:
        raise ValueError(
            "target_state_ids and partner_state_ids must both have shape "
            "[B, physical_chunks]"
        )
    if tuple(target_state_ids.shape) != tuple(partner_state_ids.shape):
        raise ValueError(
            "target/partner state shapes must match, got "
            f"{tuple(target_state_ids.shape)} and {tuple(partner_state_ids.shape)}"
        )
    if decisions + 1 > target_state_ids.shape[1]:
        raise ValueError(
            f"Need {decisions + 1} physical labels for {decisions} decisions, "
            f"got {target_state_ids.shape[1]}"
        )
    if (
        ((target_state_ids < 0) | (target_state_ids > 1)).any()
        or ((partner_state_ids < 0) | (partner_state_ids > 1)).any()
    ):
        raise ValueError("target/partner state ids must contain only 0 or 1")

    target_active = target_state_ids[:, 1 : decisions + 1].bool()
    partner_observed_active = partner_state_ids[:, :decisions].bool()
    response = target_active
    causal_listening = ~target_active & partner_observed_active
    causal_idle = ~target_active & ~partner_observed_active
    return response, causal_listening, causal_idle


def select_action_supervision_masks(
    *,
    phase: str,
    response_mask: torch.Tensor,
    causal_listen_mask: torch.Tensor,
    causal_idle_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select speaking, listening, and idle action masks for one source.

    Seamless contributes only curated causal listening. Cospeech exposes its
    speaking, causal-listening, and causal-idle masks independently; the
    corresponding loss weights decide whether each mask affects optimization.
    """

    if phase == "expanded_listen":
        return (
            causal_listen_mask,
            torch.zeros_like(causal_listen_mask),
            torch.zeros_like(causal_listen_mask),
        )
    if phase != "cospeech_replay":
        raise ValueError(f"Unknown replay phase: {phase}")
    return response_mask, causal_listen_mask, causal_idle_mask


def active_normalization_decision_mask(
    target_state_ids: Sequence[int],
    partner_state_ids: Sequence[int],
    *,
    phase: str,
    phase_loss_weight: float,
    listening_loss_weight: float,
    idle_loss_weight: float,
) -> torch.Tensor:
    """Select exactly the decisions whose action frames contribute to loss."""

    target = torch.as_tensor(target_state_ids, dtype=torch.long).view(1, -1)
    partner = torch.as_tensor(partner_state_ids, dtype=torch.long).view(1, -1)
    decisions = int(target.shape[1]) - 1
    response, causal_listen, causal_idle = shifted_action_phase_masks(
        target,
        partner,
        decisions=decisions,
    )
    primary, listening, idle = select_action_supervision_masks(
        phase=phase,
        response_mask=response,
        causal_listen_mask=causal_listen,
        causal_idle_mask=causal_idle,
    )
    active = torch.zeros_like(primary)
    if float(phase_loss_weight) > 0.0:
        active |= primary
    if float(listening_loss_weight) > 0.0:
        active |= listening
    if float(idle_loss_weight) > 0.0:
        active |= idle
    return active[0].cpu()


class ActionStateNormDataset(Dataset):
    """Lightweight dataset view exposing only masked state/action arrays."""

    def __init__(
        self,
        *,
        config: HriMotDataConfig,
        episode_count: int,
        iterator_factory: Callable[
            [], Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        ],
    ) -> None:
        self.config = config
        self.episode_count = int(episode_count)
        self.iterator_factory = iterator_factory

    def __len__(self) -> int:
        return self.episode_count

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raise TypeError(
            f"{type(self).__name__} supports only iter_action_state_arrays(), "
            f"not item access (index={index})"
        )

    def iter_action_state_arrays(
        self,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return self.iterator_factory()


def iter_posttrain_supervised_action_state_arrays(
    dataset: PostTrainEpisodeDataset,
    *,
    listen_loss_weight: float,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield Seamless arrays masked to causal-listening decisions only."""

    cfg = dataset.config
    for episode in dataset.episodes:
        rows = episode.get("chunks")
        if rows is None:
            rows = dataset._read_episode_rows(episode)
        rows = sorted(
            rows,
            key=lambda row: int(row.get("episode_chunk_idx", 0)),
        )
        states = []
        actions = []
        target_state_ids = []
        partner_state_ids = []
        for row in rows:
            chunk = row.get("chunk", {})
            if not isinstance(chunk, dict):
                chunk = {}
            target = (
                chunk.get("target", {})
                if isinstance(chunk.get("target"), dict)
                else {}
            )
            partner = (
                chunk.get("partner", {})
                if isinstance(chunk.get("partner"), dict)
                else {}
            )
            t_start = _finite_float(chunk.get("t_start"), 0.0)
            t_end = _finite_float(
                chunk.get("t_end"),
                t_start + cfg.chunk_size_sec,
            )
            state, action = dataset._load_robot_segments(
                target.get("robot_pkl"),
                t_end,
            )
            states.append(state)
            actions.append(action)
            target_state_ids.append(
                response_state_to_id(target.get("response_state"), cfg)
            )
            partner_state_ids.append(
                partner_state_to_id(partner.get("response_state"), cfg)
            )

        state_array = np.stack(states).astype(np.float32, copy=False)
        action_array = np.stack(actions).astype(np.float32, copy=False)
        decisions = min(state_array.shape[0], action_array.shape[0]) - 1
        active_decisions = active_normalization_decision_mask(
            target_state_ids,
            partner_state_ids,
            phase="expanded_listen",
            phase_loss_weight=listen_loss_weight,
            listening_loss_weight=0.0,
            idle_loss_weight=0.0,
        ).numpy()
        action_valid_mask = np.zeros(action_array.shape[:2], dtype=np.bool_)
        action_valid_mask[:decisions] = active_decisions[:, None]
        if not action_valid_mask.any():
            continue
        yield state_array, action_array, action_valid_mask


def iter_cospeech_supervised_action_state_arrays(
    dataset: CospeechEpisodeDataset,
    *,
    speaking_loss_weight: float,
    listening_loss_weight: float,
    idle_loss_weight: float,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield Cospeech arrays masked to phases with nonzero configured loss."""

    cfg = dataset.config
    for record in dataset.records:
        target = (
            record.get("target", {})
            if isinstance(record.get("target"), dict)
            else {}
        )
        partner = (
            record.get("partner", {})
            if isinstance(record.get("partner"), dict)
            else {}
        )
        timeline = (
            record.get("timeline", {})
            if isinstance(record.get("timeline"), dict)
            else {}
        )
        states, actions, action_valid_mask, _ = dataset._action_state_arrays(
            record
        )
        words, speech = dataset._load_transcript(
            str(target.get("transcript", ""))
        )
        target_start = _finite_float(timeline.get("target_start_sec"), 0.0)
        partner_duration = _finite_float(partner.get("duration_sec"), 0.0)
        target_state_ids = []
        partner_state_ids = []
        for chunk_index in range(cfg.num_chunks_per_episode):
            chunk_start = chunk_index * cfg.chunk_size_sec
            chunk_end = chunk_start + cfg.chunk_size_sec
            state, _, _, _ = dataset._target_chunk_labels(
                words,
                speech,
                round(chunk_start - target_start, 6),
                round(chunk_end - target_start, 6),
            )
            target_state_ids.append(0 if state == "non_speech" else 1)
            partner_state_ids.append(
                int(chunk_start < partner_duration and chunk_end > 0.0)
            )

        decisions = min(states.shape[0], actions.shape[0]) - 1
        active_decisions = active_normalization_decision_mask(
            target_state_ids,
            partner_state_ids,
            phase="cospeech_replay",
            phase_loss_weight=speaking_loss_weight,
            listening_loss_weight=listening_loss_weight,
            idle_loss_weight=idle_loss_weight,
        ).numpy()
        supervised_mask = np.asarray(action_valid_mask, dtype=np.bool_).copy()
        supervised_mask[decisions:] = False
        supervised_mask[:decisions] &= active_decisions[:, None]
        if not supervised_mask.any():
            continue
        yield states, actions, supervised_mask


def shifted_target_speaking_mask(
    target_state_ids: torch.Tensor,
    *,
    decisions: int,
) -> torch.Tensor:
    """Return target-speaking decisions for physical chunks c + 1."""

    decisions = int(decisions)
    if target_state_ids.dim() != 2:
        raise ValueError("target_state_ids must have shape [B, physical_chunks]")
    if decisions < 0 or decisions + 1 > target_state_ids.shape[1]:
        raise ValueError(
            f"Cannot select {decisions} shifted decisions from "
            f"{target_state_ids.shape[1]} physical chunks"
        )
    if ((target_state_ids < 0) | (target_state_ids > 1)).any():
        raise ValueError("target_state_ids must contain only 0 or 1")
    return target_state_ids[:, 1 : decisions + 1].bool()


def create_posttrain_dataset(
    args: argparse.Namespace,
    data_config: HriMotDataConfig,
    episode_index_cache: Optional[str],
) -> PostTrainEpisodeDataset:
    return PostTrainEpisodeDataset(
        data_config.train_index_paths,
        config=data_config,
        split="train",
        episode_index_cache=episode_index_cache,
    )


def create_cospeech_dataset(
    args: argparse.Namespace,
    data_config: HriMotDataConfig,
) -> CospeechEpisodeDataset:
    return CospeechEpisodeDataset(
        data_config.train_index_paths,
        config=data_config,
        split="train",
        idle_motion_path=args.idle_motion_pkl,
        idle_duration_sec=args.idle_motion_duration_sec,
        transition_sec=args.motion_transition_sec,
        use_emotion_conditioning=(
            args.use_emotion_conditioning
        ),
    )


def forward_phase_batch(
    *,
    train_model: torch.nn.Module,
    raw_wrapper: HriMoTWrapper,
    tokenizer: Any,
    audio_encoder: Optional[torch.nn.Module],
    normalizer: Any,
    batch: Dict[str, Any],
    phase: str,
    args: argparse.Namespace,
    device: torch.device,
    distributed: bool,
    world_size: int,
) -> Dict[str, Any]:
    """Forward one source batch and build its disjoint behavioral losses."""

    if phase not in ("expanded_listen", "cospeech_replay"):
        raise ValueError(f"Unknown replay phase: {phase}")

    dtype = next(raw_wrapper.base_model.parameters()).dtype
    batch_size, num_chunks = batch["target_action"].shape[:2]
    prefix_chunks = int(num_chunks) - 1
    if prefix_chunks <= 0:
        raise RuntimeError("Balanced post-training requires at least two chunks")

    robot_steps = int(batch["robot_state"].shape[2])
    action_horizon = int(batch["target_action"].shape[2])
    (
        input_ids,
        context_valid_mask,
        context_chunk_ids,
        context_token_types,
        audio_positions,
    ) = base_train.build_temporal_context_batch(
        tokenizer=tokenizer,
        batch=batch,
        prefix_chunks=prefix_chunks,
        max_context_seq_len=args.max_total_seq_len,
        device=device,
    )
    if audio_positions[:, -1, 0].eq(-1).any():
        raise RuntimeError(
            f"max_total_seq_len={args.max_total_seq_len} is too small for "
            f"{prefix_chunks} {phase} chunks"
        )

    audio_features = None
    if audio_encoder is not None:
        partner_audio = batch["partner_audio"][:, :prefix_chunks]
        if args.audio_feature_mode == "episode_contiguous":
            audio_features = base_train.encode_episode_audio_feature_batch(
                partner_audio,
                audio_encoder,
                device,
                dtype,
            )
        elif args.audio_feature_mode == "independent_chunk":
            flat_audio = partner_audio.reshape(-1, partner_audio.shape[-1])
            audio_features = base_train.encode_audio_feature_batch(
                flat_audio,
                audio_encoder,
                device,
                dtype,
            ).view(
                batch_size,
                prefix_chunks,
                base_train.AUDIO_TOKENS_PER_CHUNK,
                -1,
            )
        else:
            raise ValueError(
                f"Unknown audio_feature_mode={args.audio_feature_mode!r}"
            )

    robot_state = normalizer.normalize_state(
        batch["robot_state"][:, :prefix_chunks].to(device=device)
    )
    raw_target_action = batch["target_action"][:, :prefix_chunks].to(
        device=device
    )
    batch_action_valid_mask = batch.get("target_action_valid_mask")
    if batch_action_valid_mask is None:
        target_action_valid_mask = torch.ones(
            raw_target_action.shape[:3],
            device=device,
            dtype=torch.bool,
        )
    else:
        target_action_valid_mask = batch_action_valid_mask[
            :, :prefix_chunks
        ].to(device=device, dtype=torch.bool)
        if tuple(target_action_valid_mask.shape) != tuple(
            raw_target_action.shape[:3]
        ):
            raise RuntimeError(
                "target_action_valid_mask must match target_action [B,C,H], "
                f"got mask={tuple(target_action_valid_mask.shape)} "
                f"action={tuple(raw_target_action.shape)}"
            )
    target_action = normalizer.normalize_action(raw_target_action)

    target_state_ids = batch["target_state_ids"].to(
        device=device,
        dtype=torch.long,
    )
    speak_labels = target_state_ids[:, 1 : prefix_chunks + 1]
    if tuple(speak_labels.shape) != (batch_size, prefix_chunks):
        raise RuntimeError(
            f"Expected shifted speak labels {(batch_size, prefix_chunks)}, "
            f"got {tuple(speak_labels.shape)}"
        )
    response_elapsed_chunks = shifted_response_elapsed_chunks(
        target_state_ids,
        decisions=prefix_chunks,
    )

    emotion_labels = None
    emotion_chunk_ids = None
    if args.use_emotion_conditioning:
        if phase != "cospeech_replay":
            raise ValueError(
                "emotion fine-tuning accepts Cospeech batches only"
            )
        sample_emotion = batch.get("target_emotion_id")
        if sample_emotion is None:
            raise RuntimeError(
                "emotion Cospeech batch lacks "
                "target_emotion_id"
            )
        sample_emotion = sample_emotion.to(
            device=device,
            dtype=torch.long,
        ).view(batch_size, 1)
        recognized = (sample_emotion > 0) & (sample_emotion < NUM_EMOTION_CONDITIONS)
        if not bool(recognized.all().item()):
            raise RuntimeError(
                "speaking samples must have emotion condition IDs in [1, 6]"
            )
        emotion_labels = torch.where(
            speak_labels.bool(),
            sample_emotion.expand(-1, prefix_chunks),
            torch.zeros_like(speak_labels),
        )

    partner_state_ids = batch.get("partner_state_ids")
    if partner_state_ids is None:
        raise RuntimeError(f"{phase} batches must include partner_state_ids")
    response_mask, causal_listen_mask, causal_idle_mask = (
        shifted_action_phase_masks(
            target_state_ids,
            partner_state_ids.to(device=device, dtype=torch.long),
            decisions=prefix_chunks,
        )
    )
    (
        phase_decision_mask,
        listening_decision_mask,
        idle_decision_mask,
    ) = select_action_supervision_masks(
        phase=phase,
        response_mask=response_mask,
        causal_listen_mask=causal_listen_mask,
        causal_idle_mask=causal_idle_mask,
    )

    action_dim = int(target_action.shape[3])
    x_tau, tau, target_velocity = sample_flow_matching_pair(target_action)
    target_velocity = target_velocity.reshape(
        batch_size,
        prefix_chunks * action_horizon,
        action_dim,
    )
    decision_chunk_ids = torch.arange(
        prefix_chunks,
        device=device,
        dtype=torch.long,
    )
    robot_chunk_ids = decision_chunk_ids.repeat_interleave(robot_steps)
    robot_chunk_ids = robot_chunk_ids.unsqueeze(0).expand(batch_size, -1)
    speak_chunk_ids = decision_chunk_ids.unsqueeze(0).expand(batch_size, -1)
    if args.use_emotion_conditioning:
        emotion_chunk_ids = speak_chunk_ids
    action_chunk_ids = decision_chunk_ids.repeat_interleave(action_horizon)
    action_chunk_ids = action_chunk_ids.unsqueeze(0).expand(batch_size, -1)

    outputs = train_model(
        input_ids=input_ids,
        audio_features=audio_features,
        audio_positions=audio_positions,
        robot_state=robot_state,
        noisy_action=x_tau,
        tau=tau,
        context_valid_mask=context_valid_mask,
        context_chunk_ids=context_chunk_ids,
        context_token_types=context_token_types,
        state_chunk_ids=robot_chunk_ids,
        speak_labels=speak_labels,
        speak_chunk_ids=speak_chunk_ids,
        emotion_labels=emotion_labels,
        emotion_chunk_ids=emotion_chunk_ids,
        response_elapsed_chunks=response_elapsed_chunks,
        action_chunk_ids=action_chunk_ids,
    )
    factual_velocity = outputs["action_velocity"]

    action_valid_mask = target_action_valid_mask.reshape(
        batch_size,
        prefix_chunks * action_horizon,
    )
    phase_frame_mask = phase_decision_mask.repeat_interleave(
        action_horizon,
        dim=1,
    )
    phase_loss = phase_weighted_mse_loss(
        factual_velocity,
        target_velocity,
        action_valid_mask,
        phase_frame_mask.to(dtype=torch.float32),
        distributed=distributed,
        world_size=world_size,
    )
    supervised_frame_mask = action_valid_mask & phase_frame_mask

    listening_frame_mask = listening_decision_mask.repeat_interleave(
        action_horizon,
        dim=1,
    )
    if phase == "cospeech_replay" and not args.use_emotion_conditioning:
        listening_loss = phase_weighted_mse_loss(
            factual_velocity,
            target_velocity,
            action_valid_mask,
            listening_frame_mask.to(dtype=torch.float32),
            distributed=distributed,
            world_size=world_size,
        )
    else:
        listening_loss = factual_velocity.sum() * 0.0
    listening_supervised_frame_mask = action_valid_mask & listening_frame_mask
    if args.use_emotion_conditioning:
        listening_supervised_frame_mask = torch.zeros_like(
            listening_supervised_frame_mask
        )

    idle_frame_mask = idle_decision_mask.repeat_interleave(
        action_horizon,
        dim=1,
    )
    if phase == "cospeech_replay" and args.cospeech_idle_loss_weight > 0.0:
        idle_loss = phase_weighted_mse_loss(
            factual_velocity,
            target_velocity,
            action_valid_mask,
            idle_frame_mask.to(dtype=torch.float32),
            distributed=distributed,
            world_size=world_size,
        )
    else:
        idle_loss = factual_velocity.sum() * 0.0
    idle_supervised_frame_mask = action_valid_mask & idle_frame_mask
    if phase != "cospeech_replay" or args.cospeech_idle_loss_weight <= 0.0:
        idle_supervised_frame_mask = torch.zeros_like(
            idle_supervised_frame_mask
        )

    total_seq_len = (
        int(input_ids.shape[1])
        + prefix_chunks * robot_steps
        + prefix_chunks
        + prefix_chunks * action_horizon
    )
    return {
        "loss": phase_loss,
        "listening_loss": listening_loss,
        "idle_loss": idle_loss,
        "phase_frame_fraction": phase_frame_mask.float().mean(),
        "supervised_frame_fraction": supervised_frame_mask.float().mean(),
        "listening_frame_fraction": listening_frame_mask.float().mean(),
        "listening_supervised_frame_fraction": (
            listening_supervised_frame_mask.float().mean()
        ),
        "idle_frame_fraction": idle_frame_mask.float().mean(),
        "idle_supervised_frame_fraction": idle_supervised_frame_mask.float().mean(),
        "batch_size": int(batch_size),
        "sequence_chunks": int(prefix_chunks),
        "context_seq_len": int(input_ids.shape[1]),
        "total_seq_len": int(total_seq_len),
    }


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size, device = base_train.setup_distributed(
        args
    )
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    base_train.seed_runtime_rngs(args.seed, rank)

    conditioning_finetune = bool(args.use_emotion_conditioning)
    initialization_path = (
        (args.emotion_warmstart_from or args.emotion_init_from)
        if conditioning_finetune
        else args.init_from
    )
    checkpoint = load_checkpoint_cpu(initialization_path)
    apply_source_configuration(args, checkpoint)
    args.initialization_source_checkpoint = str(initialization_path)
    args.initialization_source_step = int(checkpoint.get("step", 0))
    source_norm_stats = base_train._coerce_norm_stats(
        checkpoint.get("norm_stats")
    )
    if args.norm_stats_mode == "source":
        if args.norm_stats:
            source_norm_stats = base_train.load_norm_stats_json(
                args.norm_stats
            )
        if source_norm_stats is None:
            raise ValueError(
                "Neither --norm-stats nor the initialization checkpoint provides "
                "normalization statistics"
            )
        if int(source_norm_stats.get("version", 1)) < 3:
            raise ValueError(
                f"Initialization checkpoint norm stats version is "
                f"{int(source_norm_stats.get('version', 1))}, expected co-speech v3"
            )

    posttrain_config = None
    if not args.use_emotion_conditioning:
        posttrain_config = HriMotDataConfig(
            train_index_paths=[args.train_index],
            max_episodes=args.max_episodes,
            strict_missing_files=True,
            # Observed partner state at decision c distinguishes causal listening
            # from decisions without observed partner speech.
            include_partner_state=True,
            include_text_fields=True,
            text_chunk_time_mode=args.text_chunk_time_mode,
            audio_loader=args.audio_loader,
        )
    cospeech_config = HriMotDataConfig(
        train_index_paths=[args.cospeech_train_index],
        max_episodes=args.max_episodes,
        strict_missing_files=True,
        # Partner state separates causal listening from neutral idle so their
        # independently weighted losses can be enabled or disabled by config.
        include_partner_state=True,
        include_text_fields=True,
        text_chunk_time_mode=args.text_chunk_time_mode,
        audio_loader=args.audio_loader,
    )
    episode_index_cache = None
    posttrain_dataset = None
    if posttrain_config is not None:
        episode_index_cache = args.episode_index_cache
        if episode_index_cache is None and args.max_episodes is None:
            episode_index_cache = (
                f"{args.train_index}.hri_mot_posttrain_offsets.jsonl"
            )
        if distributed and episode_index_cache is not None:
            if rank == 0:
                base_train.rank_zero_print(
                    rank,
                    "building/loading post-train episode cache: "
                    f"{episode_index_cache}",
                )
                create_posttrain_dataset(
                    args,
                    posttrain_config,
                    episode_index_cache,
                )
            dist.barrier(
                device_ids=[local_rank] if torch.cuda.is_available() else None
            )
        posttrain_dataset = create_posttrain_dataset(
            args,
            posttrain_config,
            episode_index_cache,
        )
    cospeech_dataset = create_cospeech_dataset(args, cospeech_config)
    cospeech_full_dataset = cospeech_dataset
    posttrain_source_count = (
        len(posttrain_dataset) if posttrain_dataset is not None else 0
    )
    cospeech_full_source_count = len(cospeech_dataset)
    if posttrain_dataset is not None and posttrain_source_count == 0:
        raise RuntimeError("No post-training episodes loaded")
    if cospeech_full_source_count == 0:
        raise RuntimeError("No co-speech replay episodes loaded")

    if conditioning_finetune:
        cospeech_subset_indices = list(range(cospeech_full_source_count))
    else:
        cospeech_dataset, cospeech_subset_indices = deterministic_random_subset(
            cospeech_dataset,
            args.cospeech_subset_size,
            seed=args.seed,
        )
    selected_full_cospeech = len(cospeech_subset_indices) == cospeech_full_source_count
    locked_subset_hash = indices_sha256(cospeech_subset_indices)
    if not selected_full_cospeech and locked_subset_hash != BASELINE_COSPEECH_SUBSET_INDICES_SHA256:
        raise ValueError(
            f"CoSpeech subset is not the accepted baseline: {locked_subset_hash}"
        )
    if conditioning_finetune:
        args.cospeech_subset_identity_sha256 = validate_emotion_index(
            cospeech_full_dataset,
            cospeech_subset_indices,
            baseline_index=args.cospeech_baseline_index,
            require_locked_subset=not selected_full_cospeech,
        )
    cospeech_source_count = len(cospeech_dataset)

    if args.norm_stats_mode == "joint_supervised":
        if not selected_full_cospeech:
            raise ValueError(
                "--norm-stats-mode joint_supervised requires "
                "--cospeech-subset-size 0 "
                "so normalization covers full Cospeech"
            )
        if args.norm_stats:
            norm_stats = base_train.load_norm_stats_json(args.norm_stats)
            if int(norm_stats.get("version", 1)) < 3:
                raise ValueError(
                    f"Precomputed norm stats version is "
                    f"{int(norm_stats.get('version', 1))}, expected version 3"
                )
            args.normalization_source = (
                f"precomputed_supervised_json:{args.norm_stats}"
            )
        else:
            if distributed:
                raise RuntimeError(
                    "Distributed joint normalization must be precomputed before "
                    "torchrun. Launch through train_hri_mot_posttrain.sh or pass "
                    "--norm-stats /path/to/norm_stats.json."
                )
            base_train.rank_zero_print(
                rank,
                "computing supervised Seamless + Cospeech normalization statistics",
            )
            posttrain_norm_dataset = ActionStateNormDataset(
                config=posttrain_config,
                episode_count=len(posttrain_dataset),
                iterator_factory=lambda: iter_posttrain_supervised_action_state_arrays(
                    posttrain_dataset,
                    listen_loss_weight=args.listen_action_loss_weight,
                ),
            )
            if not isinstance(cospeech_dataset, CospeechEpisodeDataset):
                raise TypeError(
                    "Supervised full Cospeech normalization requires the original dataset"
                )
            cospeech_norm_dataset = ActionStateNormDataset(
                config=cospeech_config,
                episode_count=len(cospeech_dataset),
                iterator_factory=lambda: iter_cospeech_supervised_action_state_arrays(
                    cospeech_dataset,
                    speaking_loss_weight=args.cospeech_speaking_loss_weight,
                    listening_loss_weight=args.cospeech_listening_loss_weight,
                    idle_loss_weight=args.cospeech_idle_loss_weight,
                ),
            )
            posttrain_norm_stats = base_train.compute_action_state_norm_stats(
                posttrain_norm_dataset,
                std_floor=args.norm_std_floor,
            )
            cospeech_norm_stats = base_train.compute_action_state_norm_stats(
                cospeech_norm_dataset,
                std_floor=args.norm_std_floor,
            )
            norm_stats = merge_action_state_norm_stats(
                (posttrain_norm_stats, cospeech_norm_stats),
                std_floor=args.norm_std_floor,
            )
            args.normalization_posttrain_state_count = int(
                posttrain_norm_stats["state_count"]
            )
            args.normalization_posttrain_action_count = int(
                posttrain_norm_stats["action_count"]
            )
            args.normalization_cospeech_state_count = int(
                cospeech_norm_stats["state_count"]
            )
            args.normalization_cospeech_action_count = int(
                cospeech_norm_stats["action_count"]
            )
            args.normalization_source = (
                "computed_full_supervised_seamless_cospeech"
            )
    else:
        if source_norm_stats is None:
            raise RuntimeError("Source normalization unexpectedly missing")
        norm_stats = source_norm_stats
        args.normalization_source = (
            f"json:{args.norm_stats}"
            if args.norm_stats
            else f"init_checkpoint:{initialization_path}"
        )

    if args.compute_norm_stats_only:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        norm_stats_path = out_dir / "norm_stats.json"
        base_train.write_norm_stats_json(norm_stats_path, norm_stats)
        with open(out_dir / "norm_stats_args.json", "w", encoding="utf-8") as fp:
            json.dump(vars(args), fp, indent=2, ensure_ascii=False)
        base_train.rank_zero_print(
            rank,
            f"saved precomputed supervised normalization to {norm_stats_path}",
        )
        return

    global_posttrain_batch_size = (
        0
        if args.use_emotion_conditioning
        else int(world_size * args.posttrain_episode_batch_size)
    )
    global_cospeech_batch_size = int(
        world_size * args.cospeech_episode_batch_size
    )
    posttrain_padded_count = 0
    posttrain_repeated_count = 0
    cospeech_padded_count = 0
    cospeech_repeated_count = 0
    posttrain_sampler = None
    cospeech_sampler = None
    posttrain_selected_count = posttrain_source_count

    if args.use_emotion_conditioning:
        if not isinstance(cospeech_dataset, CospeechEpisodeDataset):
            raise TypeError(
                "Emotion sampling requires the full Cospeech dataset"
            )
        cospeech_sampler = base_train.DistributedBalancedEmotionSampler(
            cospeech_dataset,
            samples_per_class=args.emotion_samples_per_class,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
        )
        if cospeech_sampler.global_num_samples % global_cospeech_batch_size != 0:
            raise ValueError(
                "Strict emotion samples must form complete global "
                f"batches: samples={cospeech_sampler.global_num_samples}, "
                f"global_batch={global_cospeech_batch_size}"
            )
        args.emotion_samples_per_epoch = int(
            cospeech_sampler.global_num_samples
        )
        args.emotion_samples_per_rank = int(len(cospeech_sampler))
        args.emotion_source_class_counts = dict(
            cospeech_sampler.source_class_counts
        )
        args.emotion_eligible_class_counts = dict(
            cospeech_sampler.eligible_class_counts
        )
        args.emotion_label_quotas = dict(cospeech_sampler.label_quotas)
        args.emotion_effective_repetitions_per_epoch = dict(
            cospeech_sampler.effective_repetitions_per_epoch
        )
    else:
        if posttrain_dataset is None:
            raise RuntimeError("Baseline post-training requires Seamless data")
        posttrain_padded_count = full_batch_padded_size(
            posttrain_source_count,
            global_posttrain_batch_size,
        )
        cospeech_padded_count = full_batch_padded_size(
            cospeech_source_count,
            global_cospeech_batch_size,
        )
        posttrain_repeated_count = (
            posttrain_padded_count - posttrain_source_count
        )
        cospeech_repeated_count = (
            cospeech_padded_count - cospeech_source_count
        )
        if posttrain_repeated_count > 0:
            if posttrain_dataset is None:
                raise RuntimeError("Post-training dataset unexpectedly missing")
            posttrain_dataset = RepeatPaddedDataset(
                posttrain_dataset,
                posttrain_padded_count,
                seed=args.seed,
            )
        if cospeech_repeated_count > 0:
            cospeech_dataset = RepeatPaddedDataset(
                cospeech_dataset,
                cospeech_padded_count,
                seed=args.seed + 1,
            )
        posttrain_sampler = (
            DistributedSampler(
                posttrain_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=False,
            )
            if distributed
            else None
        )
        cospeech_sampler = (
            DistributedSampler(
                cospeech_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed + 1,
                drop_last=False,
            )
            if distributed
            else None
        )
    posttrain_loader = (
        DataLoader(
            posttrain_dataset,
            batch_size=args.posttrain_episode_batch_size,
            shuffle=(posttrain_sampler is None),
            sampler=posttrain_sampler,
            num_workers=args.num_workers,
            persistent_workers=(args.num_workers > 0),
            collate_fn=HriMotEpisodeCollator(),
            drop_last=False,
        )
        if posttrain_dataset is not None
        else None
    )
    args.source_episode_count = int(posttrain_source_count)
    args.posttrain_selected_episode_count = int(posttrain_selected_count)
    args.posttrain_subset_seed = int(args.seed + 100_003)
    args.padded_episode_count = int(posttrain_padded_count)
    args.repeated_episode_count = int(posttrain_repeated_count)
    args.cospeech_source_episode_count = int(cospeech_source_count)
    args.cospeech_full_source_episode_count = int(cospeech_full_source_count)
    args.cospeech_selected_episode_count = int(cospeech_source_count)
    args.cospeech_subset_seed = int(args.seed)
    args.cospeech_subset_indices_sha256 = indices_sha256(cospeech_subset_indices)
    args.cospeech_padded_episode_count = int(cospeech_padded_count)
    args.cospeech_repeated_episode_count = int(cospeech_repeated_count)
    args.global_posttrain_batch_size = int(global_posttrain_batch_size)
    args.global_cospeech_batch_size = int(global_cospeech_batch_size)
    args.global_episode_batch_size = int(
        global_cospeech_batch_size
        if args.use_emotion_conditioning
        else global_posttrain_batch_size + global_cospeech_batch_size
    )

    cospeech_loader = DataLoader(
        cospeech_dataset,
        batch_size=args.cospeech_episode_batch_size,
        shuffle=(cospeech_sampler is None),
        sampler=cospeech_sampler,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        collate_fn=HriMotEpisodeCollator(),
        drop_last=False,
    )
    args.steps_per_epoch = int(
        len(cospeech_loader)
        if args.use_emotion_conditioning
        else len(posttrain_loader)
        if posttrain_loader is not None
        else 0
    )
    args.max_steps = int(args.steps_per_epoch * args.epochs)
    if args.max_steps <= 0:
        raise RuntimeError("Post-training DataLoader has no batches")
    base_train.validate_lr_args(args)

    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "args.json", "w", encoding="utf-8") as fp:
            json.dump(vars(args), fp, indent=2, ensure_ascii=False)
    metrics_fp, wandb_run = base_train.create_metric_loggers(
        args,
        out_dir,
        rank=rank,
    )
    if distributed:
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)

    normalizer = base_train.ActionStateNormalizer(norm_stats, device=device)
    if rank == 0:
        base_train.write_norm_stats_json(out_dir / "norm_stats.json", norm_stats)

    base_model, tokenizer, qwen_omni_ckpt, audio_tower_ckpt = (
        base_train.load_base_model_and_tokenizer(
            args.checkpoint_dir,
            device,
            args.seed,
        )
    )
    audio_encoder = None
    if not args.no_audio_features:
        audio_encoder = load_audio_encoder(qwen_omni_ckpt, audio_tower_ckpt, device)
        audio_encoder.eval()

    model_data_config = (
        cospeech_config
        if args.use_emotion_conditioning
        else posttrain_config
    )
    if model_data_config is None:
        raise RuntimeError("Model data configuration is unexpectedly missing")
    wrapper = HriMoTWrapper(
        base_model=base_model,
        decoder_path="_forward_module.transformer.h",
        robot_state_dim=model_data_config.robot_state_dim,
        action_dim=model_data_config.action_dim,
        action_hidden_size=args.action_hidden_size,
        action_intermediate_size=args.action_intermediate_size,
        robot_state_steps=model_data_config.robot_steps_per_chunk,
        speak_conditioning=args.speak_conditioning,
        use_response_time_embedding=args.response_time_embedding,
        use_emotion_conditioning=(
            args.use_emotion_conditioning
        ),
        emotion_film_layers=args.emotion_film_layers,
    ).to(device)
    complete_checkpoint_parameter_names: Optional[list[str]] = None
    source_optimizer_masters: Dict[str, torch.nn.Parameter] = {}
    if conditioning_finetune:
        load_emotion_finetune_weights_strict(
            wrapper,
            checkpoint,
            device=device,
            restore_conditioning=bool(args.emotion_warmstart_from),
        )
    if args.use_emotion_conditioning:
        source_parameters = checkpoint.get("trainable_params")
        if not isinstance(source_parameters, dict):
            raise ValueError(
                "Emotion initialization checkpoint has no "
                "trainable_params"
            )
        new_condition_parameters = {"emotion_embedding.weight"}
        new_condition_parameters.update(
            name
            for name, _ in wrapper.named_parameters()
            if name.startswith("emotion_layer_film.")
        )
        complete_checkpoint_parameter_names = sorted(
            set(source_parameters) | new_condition_parameters
        )
        unfrozen_source_names = configure_emotion_trainable_parameters(
            wrapper,
            restored_parameter_names=tuple(set(source_parameters) - new_condition_parameters),
            unfreeze_action_layers=args.motion_unfreeze_action_layers,
            unfreeze_response_time_encoder=args.motion_unfreeze_response_time_encoder,
        )
        source_optimizer_masters = create_fp32_source_optimizer_masters(
            wrapper,
            unfrozen_source_names,
        )
        total_action_layers = len(wrapper.action_expert.layers)
        args.motion_unfrozen_action_layer_indices = list(
            range(
                total_action_layers - args.motion_unfreeze_action_layers,
                total_action_layers,
            )
        )
        args.motion_unfrozen_source_parameter_names = list(
            unfrozen_source_names
        )
        args.motion_unfrozen_source_parameter_count = int(
            sum(
                dict(wrapper.named_parameters())[name].numel()
                for name in unfrozen_source_names
            )
        )
        args.motion_unfrozen_response_time_parameter_count = int(
            sum(
                dict(wrapper.named_parameters())[name].numel()
                for name in unfrozen_source_names
                if name.startswith("response_time_encoder.")
            )
        )
        args.motion_condition_parameter_count = int(
            sum(
                parameter.numel()
                for name, parameter in wrapper.named_parameters()
                if parameter.requires_grad
                and name.startswith(
                    (
                        "emotion_embedding.",
                        "emotion_layer_film.",
                    )
                )
            )
        )
        args.motion_source_optimizer_precision = "fp32_master_to_model_dtype"
    # The optimizer and global step always restart. In baseline mode all HRI-MoT
    # modules keep constructor initialization; in emotion mode every
    # source action parameter is restored, but only the final selected blocks,
    # velocity head, optional response-time encoder, and conditions are trainable.
    del checkpoint

    if rank == 0:
        # The architecture-dependent trainable parameter audit is only known
        # after model construction, so refresh args.json before training.
        with open(out_dir / "args.json", "w", encoding="utf-8") as fp:
            json.dump(vars(args), fp, indent=2, ensure_ascii=False)

    wrapper.train()
    wrapper.base_model.eval()
    raw_wrapper = wrapper
    train_model = (
        DDP(
            raw_wrapper,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=args.ddp_find_unused_parameters,
        )
        if distributed
        else raw_wrapper
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(
            raw_wrapper,
            use_emotion_conditioning=(
                args.use_emotion_conditioning
            ),
            pretrained_lr_scale=args.pretrained_lr_scale,
            source_optimizer_masters=(
                source_optimizer_masters
                if args.use_emotion_conditioning
                else None
            ),
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    base_train.seed_runtime_rngs(args.seed, rank)
    if args.use_emotion_conditioning:
        base_train.rank_zero_print(
            rank,
            "emotion data: Appendix-B.4 Cospeech only; "
            f"cospeech_source={cospeech_full_source_count}; "
            f"eligible_labels={args.emotion_eligible_class_counts}; "
            f"label_quotas={args.emotion_label_quotas}; "
            f"samples_per_pass={args.emotion_samples_per_epoch}; "
            f"samples_per_rank={args.emotion_samples_per_rank}; "
            f"global_batch={global_cospeech_batch_size}; "
            "sampling=uniform_six_emotions",
        )
    else:
        if posttrain_dataset is None:
            raise RuntimeError("Baseline post-training dataset is missing")
        base_train.rank_zero_print(
            rank,
            f"expanded listening episodes: source={posttrain_source_count} "
            f"padded={len(posttrain_dataset)} "
            f"repeated={posttrain_repeated_count} "
            f"global_batch={global_posttrain_batch_size}",
        )
        base_train.rank_zero_print(
            rank,
            f"co-speech replay episodes: full_source={cospeech_full_source_count} "
            f"selected={cospeech_source_count} subset_seed={args.seed} "
            f"padded={len(cospeech_dataset)} repeated={cospeech_repeated_count} "
            f"global_batch={global_cospeech_batch_size}",
        )
    base_train.rank_zero_print(
        rank,
        f"strict passes: {args.epochs}; steps_per_epoch={args.steps_per_epoch}; "
        f"total_steps={args.max_steps}",
    )
    if args.use_emotion_conditioning:
        initialization_message = (
            "emotion fine-tune: restored action "
            f"parameters from {args.initialization_source_checkpoint} "
            f"(step={args.initialization_source_step}); condition_weights="
            f"{'restored' if args.emotion_warmstart_from else 'zero_initialized'}; "
            "optimizer/step restarted; unfrozen Action Expert "
            f"layers={args.motion_unfrozen_action_layer_indices}; velocity "
            f"head_trainable={args.motion_unfreeze_action_layers > 0}; "
            f"response_time_encoder_trainable={args.motion_unfreeze_response_time_encoder}; "
            f"source_action_lr_scale={args.pretrained_lr_scale:g}"
        )
    else:
        initialization_message = (
            "randomly initialized HRI-MoT action modules; architecture and "
            f"configuration loaded from {args.init_from} "
            f"(reference step={args.source_checkpoint_step})"
        )
    base_train.rank_zero_print(rank, initialization_message)
    loss_message = (
        "emotion objective: Cospeech speaking flow matching only; "
        "all auxiliary objectives are absent"
        if args.use_emotion_conditioning
        else (
            "balanced action losses: "
            f"expanded_listen={args.listen_action_loss_weight:g} "
            f"cospeech_speaking={args.cospeech_speaking_loss_weight:g} "
            f"cospeech_listening={args.cospeech_listening_loss_weight:g} "
            f"cospeech_causal_idle={args.cospeech_idle_loss_weight:g}; "
            "expanded speaking is masked"
        )
    )
    base_train.rank_zero_print(rank, loss_message)
    base_train.rank_zero_print(
        rank,
        f"post-train code: version={args.posttrain_code_version} "
        f"loss_normalization={args.loss_normalization} "
        f"source_sha256={args.posttrain_source_sha256}",
    )
    base_train.rank_zero_print(
        rank,
        f"architecture: hidden={args.action_hidden_size} "
        f"intermediate={args.action_intermediate_size} "
        f"speak={args.speak_conditioning} "
        f"response_time_embedding={args.response_time_embedding} "
        f"emotion_film_layers="
        f"{args.emotion_film_layers}",
    )
    base_train.rank_zero_print(
        rank,
        f"normalization: mode={args.norm_stats_mode} "
        f"source={args.normalization_source} "
        f"version={norm_stats['version']} "
        f"state_count={norm_stats['state_count']} "
        f"action_count={norm_stats['action_count']}",
    )
    base_train.rank_zero_print(
        rank,
        f"lr schedule: {args.lr_schedule} condition_peak={args.lr:g} "
        f"condition_min={args.min_lr:g} "
        f"pretrained_peak={args.lr * args.pretrained_lr_scale:g} "
        f"pretrained_min={args.min_lr * args.pretrained_lr_scale:g} "
        f"warmup={args.lr_warmup_steps}",
    )

    disable_tqdm = rank != 0 or args.disable_tqdm or not sys.stderr.isatty()
    progress = tqdm(
        total=args.max_steps,
        desc="hri-mot post-train",
        disable=disable_tqdm,
    )
    cospeech_iter = (
        None
        if conditioning_finetune
        else base_train.cycle(
            cospeech_loader,
            sampler=cospeech_sampler,
        )
    )
    global_step = 0
    for epoch in range(args.epochs):
        if posttrain_sampler is not None:
            posttrain_sampler.set_epoch(epoch)
        if args.use_emotion_conditioning:
            if cospeech_sampler is None:
                raise RuntimeError(
                    "Emotion sampler is unexpectedly missing"
                )
            cospeech_sampler.set_epoch(epoch)
            epoch_loader = cospeech_loader
        else:
            if posttrain_loader is None:
                raise RuntimeError("Post-training loader unexpectedly missing")
            epoch_loader = posttrain_loader

        for source_batch in epoch_loader:
            global_step += 1
            current_lr = base_train.learning_rate_for_step(args, global_step)
            set_scaled_optimizer_lr(optimizer, current_lr)
            train_model.train()
            raw_wrapper.base_model.eval()
            if audio_encoder is not None:
                audio_encoder.eval()

            optimizer.zero_grad(set_to_none=True)
            if source_optimizer_masters:
                clear_source_model_gradients(
                    raw_wrapper,
                    source_optimizer_masters,
                )
            if args.use_emotion_conditioning:
                cospeech_batch = source_batch
                posttrain_batch = None
            else:
                posttrain_batch = source_batch
                if cospeech_iter is None:
                    raise RuntimeError(
                        "Cospeech replay iterator unexpectedly missing"
                    )
                cospeech_batch = next(cospeech_iter)

            listen_result = None
            if posttrain_batch is not None:
                if distributed:
                    with train_model.no_sync():
                        listen_result = forward_phase_batch(
                            train_model=train_model,
                            raw_wrapper=raw_wrapper,
                            tokenizer=tokenizer,
                            audio_encoder=audio_encoder,
                            normalizer=normalizer,
                            batch=posttrain_batch,
                            phase="expanded_listen",
                            args=args,
                            device=device,
                            distributed=distributed,
                            world_size=world_size,
                        )
                        (
                            args.listen_action_loss_weight
                            * listen_result["loss"]
                        ).backward()
                else:
                    listen_result = forward_phase_batch(
                        train_model=train_model,
                        raw_wrapper=raw_wrapper,
                        tokenizer=tokenizer,
                        audio_encoder=audio_encoder,
                        normalizer=normalizer,
                        batch=posttrain_batch,
                        phase="expanded_listen",
                        args=args,
                        device=device,
                        distributed=distributed,
                        world_size=world_size,
                    )
                    (
                        args.listen_action_loss_weight
                        * listen_result["loss"]
                    ).backward()

            cospeech_result = forward_phase_batch(
                train_model=train_model,
                raw_wrapper=raw_wrapper,
                tokenizer=tokenizer,
                audio_encoder=audio_encoder,
                normalizer=normalizer,
                batch=cospeech_batch,
                phase="cospeech_replay",
                args=args,
                device=device,
                distributed=distributed,
                world_size=world_size,
            )
            cospeech_loss = (
                args.cospeech_speaking_loss_weight * cospeech_result["loss"]
                + args.cospeech_idle_loss_weight
                * cospeech_result["idle_loss"]
            )
            if not conditioning_finetune:
                cospeech_loss = (
                    cospeech_loss
                    + args.cospeech_listening_loss_weight
                    * cospeech_result["listening_loss"]
                )
            cospeech_loss.backward()

            if source_optimizer_masters:
                copy_source_gradients_to_masters(
                    raw_wrapper,
                    source_optimizer_masters,
                )

            if args.grad_clip > 0.0:
                optimized_parameters = [
                    parameter
                    for group in optimizer.param_groups
                    if float(group.get("lr_scale", 1.0)) > 0.0
                    for parameter in group["params"]
                    if parameter.grad is not None
                ]
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    optimized_parameters,
                    args.grad_clip,
                )
            else:
                grad_norm = torch.zeros((), device=device)
            optimizer.step()
            if source_optimizer_masters:
                sync_source_masters_to_model(
                    raw_wrapper,
                    source_optimizer_masters,
                )

            log_speaking_loss = base_train.distributed_mean(
                cospeech_result["loss"].detach(),
                distributed,
                world_size,
            )
            zero_metric = log_speaking_loss.new_zeros(())
            if listen_result is None:
                log_listen_loss = zero_metric
                log_listen_fraction = zero_metric
                log_listen_supervised_fraction = zero_metric
            else:
                log_listen_loss = base_train.distributed_mean(
                    listen_result["loss"].detach(),
                    distributed,
                    world_size,
                )
                log_listen_fraction = base_train.distributed_mean(
                    listen_result["phase_frame_fraction"].detach(),
                    distributed,
                    world_size,
                )
                log_listen_supervised_fraction = base_train.distributed_mean(
                    listen_result["supervised_frame_fraction"].detach(),
                    distributed,
                    world_size,
                )
            log_cospeech_listening_loss = base_train.distributed_mean(
                cospeech_result["listening_loss"].detach(),
                distributed,
                world_size,
            )
            log_idle_loss = base_train.distributed_mean(
                cospeech_result["idle_loss"].detach(),
                distributed,
                world_size,
            )
            log_speaking_fraction = base_train.distributed_mean(
                cospeech_result["phase_frame_fraction"].detach(),
                distributed,
                world_size,
            )
            log_speaking_supervised_fraction = base_train.distributed_mean(
                cospeech_result["supervised_frame_fraction"].detach(),
                distributed,
                world_size,
            )
            log_cospeech_listening_fraction = base_train.distributed_mean(
                cospeech_result["listening_frame_fraction"].detach(),
                distributed,
                world_size,
            )
            log_cospeech_listening_supervised_fraction = (
                base_train.distributed_mean(
                    cospeech_result[
                        "listening_supervised_frame_fraction"
                    ].detach(),
                    distributed,
                    world_size,
                )
            )
            log_idle_fraction = base_train.distributed_mean(
                cospeech_result["idle_frame_fraction"].detach(),
                distributed,
                world_size,
            )
            log_idle_supervised_fraction = base_train.distributed_mean(
                cospeech_result["idle_supervised_frame_fraction"].detach(),
                distributed,
                world_size,
            )
            log_loss = (
                args.listen_action_loss_weight * log_listen_loss
                + args.cospeech_speaking_loss_weight * log_speaking_loss
                + args.cospeech_listening_loss_weight
                * log_cospeech_listening_loss
                + args.cospeech_idle_loss_weight * log_idle_loss
            )
            log_metrics = {
                "loss": float(log_loss.item()),
                "listen_flow_loss": float(log_listen_loss.item()),
                "cospeech_speaking_flow_loss": float(
                    log_speaking_loss.item()
                ),
                "cospeech_listening_flow_loss": float(
                    log_cospeech_listening_loss.item()
                ),
                "cospeech_idle_flow_loss": float(log_idle_loss.item()),
                "listen_frame_fraction": float(log_listen_fraction.item()),
                "listen_supervised_frame_fraction": float(
                    log_listen_supervised_fraction.item()
                ),
                "cospeech_speaking_frame_fraction": float(
                    log_speaking_fraction.item()
                ),
                "cospeech_speaking_supervised_frame_fraction": float(
                    log_speaking_supervised_fraction.item()
                ),
                "cospeech_listening_frame_fraction": float(
                    log_cospeech_listening_fraction.item()
                ),
                "cospeech_listening_supervised_frame_fraction": float(
                    log_cospeech_listening_supervised_fraction.item()
                ),
                "cospeech_idle_frame_fraction": float(log_idle_fraction.item()),
                "cospeech_idle_supervised_frame_fraction": float(
                    log_idle_supervised_fraction.item()
                ),
                "listen_action_loss_weight": float(
                    args.listen_action_loss_weight
                ),
                "cospeech_speaking_loss_weight": float(
                    args.cospeech_speaking_loss_weight
                ),
                "cospeech_listening_loss_weight": float(
                    args.cospeech_listening_loss_weight
                ),
                "cospeech_idle_loss_weight": float(
                    args.cospeech_idle_loss_weight
                ),
                "posttrain_code_version": args.posttrain_code_version,
                "loss_normalization": args.loss_normalization,
                "posttrain_source_sha256": args.posttrain_source_sha256,
                "epoch": int(epoch + 1),
                "training_data_mode": args.training_data_mode,
                "action_supervision_mode": args.action_supervision_mode,
                "posttrain_episode_batch_size": int(
                    0 if listen_result is None else listen_result["batch_size"]
                ),
                "cospeech_episode_batch_size": int(
                    cospeech_result["batch_size"]
                ),
                "episode_batch_size": int(
                    (0 if listen_result is None else listen_result["batch_size"])
                    + cospeech_result["batch_size"]
                ),
                "sequence_chunks": int(
                    cospeech_result["sequence_chunks"]
                    if listen_result is None
                    else listen_result["sequence_chunks"]
                ),
                "posttrain_context_seq_len": int(
                    0
                    if listen_result is None
                    else listen_result["context_seq_len"]
                ),
                "cospeech_context_seq_len": int(
                    cospeech_result["context_seq_len"]
                ),
                "posttrain_total_seq_len": int(
                    0 if listen_result is None else listen_result["total_seq_len"]
                ),
                "cospeech_total_seq_len": int(
                    cospeech_result["total_seq_len"]
                ),
                "grad_norm": float(grad_norm.detach().float().item()),
                "lr": float(current_lr),
                "emotion_condition_lr": float(current_lr),
                "pretrained_action_lr": float(
                    current_lr * args.pretrained_lr_scale
                ),
            }
            if rank == 0:
                progress.update(1)
                progress.set_postfix(
                    loss=f"{log_loss.item():.4f}",
                    listen=f"{log_listen_loss.item():.4f}",
                    speaking=f"{log_speaking_loss.item():.4f}",
                    cospeech_listen=f"{log_cospeech_listening_loss.item():.4f}",
                    idle=f"{log_idle_loss.item():.4f}",
                )
                base_train.write_training_metrics(
                    metrics_fp=metrics_fp,
                    wandb_run=wandb_run,
                    step=global_step,
                    metrics=log_metrics,
                )
            if rank == 0 and (
                global_step % args.log_interval == 0 or global_step == 1
            ):
                print(
                    f"post-step {global_step}/{args.max_steps} "
                    f"loss={log_loss.item():.6f} "
                    f"listen={log_listen_loss.item():.6f} "
                    f"cospeech_speaking={log_speaking_loss.item():.6f} "
                    f"cospeech_listening={log_cospeech_listening_loss.item():.6f} "
                    f"cospeech_idle={log_idle_loss.item():.6f} "
                    f"listen_frames={log_listen_fraction.item():.4f} "
                    f"speaking_frames={log_speaking_fraction.item():.4f} "
                    f"cospeech_listening_frames="
                    f"{log_cospeech_listening_fraction.item():.4f} "
                    f"idle_frames={log_idle_fraction.item():.4f} "
                    f"grad_norm={grad_norm.detach().float().item():.4f} "
                    f"condition_lr={current_lr:.8g} "
                    f"pretrained_lr="
                    f"{current_lr * args.pretrained_lr_scale:.8g}",
                    flush=True,
                )
            if (
                rank == 0
                and args.save_interval > 0
                and global_step % args.save_interval == 0
            ):
                checkpoint_path = out_dir / f"step-{global_step:06d}.pt"
                if complete_checkpoint_parameter_names is not None:
                    save_complete_conditioning_checkpoint(
                        checkpoint_path,
                        raw_wrapper,
                        optimizer,
                        global_step,
                        args,
                        norm_stats,
                        parameter_names=complete_checkpoint_parameter_names,
                    )
                else:
                    base_train.save_checkpoint(
                        checkpoint_path,
                        raw_wrapper,
                        optimizer,
                        global_step,
                        args,
                        norm_stats,
                    )

    if global_step != args.max_steps:
        raise RuntimeError(
            f"Training step mismatch: completed={global_step}, "
            f"expected={args.max_steps}"
        )
    if rank == 0:
        final_path = out_dir / "final.pt"
        if complete_checkpoint_parameter_names is not None:
            save_complete_conditioning_checkpoint(
                final_path,
                raw_wrapper,
                optimizer,
                global_step,
                args,
                norm_stats,
                parameter_names=complete_checkpoint_parameter_names,
            )
        else:
            base_train.save_checkpoint(
                final_path,
                raw_wrapper,
                optimizer,
                global_step,
                args,
                norm_stats,
            )
        if wandb_run is not None:
            wandb_run.finish()
        if metrics_fp is not None:
            metrics_fp.close()
        progress.close()
        print(
            f"saved post-trained checkpoint to {out_dir / 'final.pt'}",
            flush=True,
        )
    if distributed:
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
