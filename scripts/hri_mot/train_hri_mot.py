#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audiointeraction.dataset.TOKENS import (  # noqa: E402
    ASSISTANT,
    AUDIO_BEGIN,
    ENGLISH,
    KEEP_SILENCE,
    ONLINE,
    PAD,
    SYSTEM,
    TEXT_BEGIN,
    TEXT_END,
)
from src.audiointeraction.hri_mot.data_config import (  # noqa: E402
    CONTEXT_TOKEN_PARTNER_AUDIO,
    CONTEXT_TOKEN_SYSTEM,
    CONTEXT_TOKEN_TARGET_CONTROL,
    CONTEXT_TOKEN_TARGET_TEXT,
    DEFAULT_TEXT_CHUNK_TIME_MODE,
    HriMotDataConfig,
    TEXT_CHUNK_TIME_MODES,
    TRAIN_INDEX_PATHS,
)
from src.audiointeraction.hri_mot.episode_dataset import (  # noqa: E402
    HriMotEpisodeCollator,
    HriMotEpisodeDataset,
)
from src.audiointeraction.hri_mot.cospeech_episode_dataset import (  # noqa: E402
    DEFAULT_IDLE_MOTION_PATH,
    CospeechEpisodeDataset,
)
from src.audiointeraction.hri_mot.flow_matching import sample_flow_matching_pair  # noqa: E402
from src.audiointeraction.hri_mot.emotion_conditioning import (  # noqa: E402
    EMOTION_LABEL_TO_CONDITION_ID,
    EMOTION_LABELS,
    EMOTION_MAPPING_VERSION,
    NO_EMOTION_ID,
)
from src.audiointeraction.hri_mot.emotion_sampling import DistributedBalancedEmotionSampler
from src.audiointeraction.hri_mot.mot_wrapper import HriMoTWrapper  # noqa: E402
from src.audiointeraction.hri_mot.response_timing import (  # noqa: E402
    shifted_response_elapsed_chunks,
)
from src.audiointeraction.tokenizer import Tokenizer  # noqa: E402
from src.audiointeraction.utils import get_default_supported_precision  # noqa: E402
from utils import load_audio_encoder, load_model, resolve_checkpoint_paths, set_seed  # noqa: E402


AUDIO_TOKENS_PER_CHUNK = 10
DEFAULT_COSPEECH_INDEX = "data/cospeech/train_index_timed_words.jsonl"
DEFAULT_COSPEECH_EMOTION_INDEX = (
    "data/cospeech/train_index_timed_words_emotion.jsonl"
)
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. When there is no user text, if the audio contains a question, "
    "please answer it. If it is a sound effect, determine based on the sound whether help is needed."
)


def parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal HRI-MoT training loop.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument(
        "--train-index",
        default=None,
        help=(
            "Training index. If omitted, the default episode index is used for "
            "episode_jsonl; Cospeech selects its emotion-tagged index when "
            "emotion conditioning is enabled."
        ),
    )
    parser.add_argument("--episode-index-cache", default=None)
    parser.add_argument(
        "--dataset-format",
        choices=("episode_jsonl", "cospeech"),
        default="episode_jsonl",
        help="episode_jsonl uses the original 50-row episode index; cospeech uses one compact row per clip.",
    )
    parser.add_argument(
        "--idle-motion-pkl",
        default=DEFAULT_IDLE_MOTION_PATH,
        help="35-dim S1 gripper PKL used for co-speech listening/tail idle motion.",
    )
    parser.add_argument("--idle-motion-duration-sec", type=float, default=9.0)
    parser.add_argument("--motion-transition-sec", type=float, default=0.4)
    parser.add_argument("--output-dir", default="outputs/hri_mot_train")
    parser.add_argument("--device", default="auto", help="Single-process device. Ignored under torchrun/DDP.")
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--episode-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--chunks-per-step",
        type=int,
        default=0,
        help="In temporal_prefix, number of episode chunks to use. Use 0 for all available chunks.",
    )
    parser.add_argument(
        "--temporal-prefix-sampling",
        choices=("random", "fixed"),
        default="fixed",
        help="Deprecated compatibility flag. Temporal training now uses a fixed chunk count.",
    )
    parser.add_argument(
        "--loss-last-n-chunks",
        type=int,
        default=0,
        help="If >0, compute action loss only on the last N temporal chunks. 0 means all selected chunks.",
    )
    parser.add_argument(
        "--terminal-idle-action-loss-weight",
        type=float,
        default=0.10,
        help=(
            "Weight for a separately normalized action loss on artificial "
            "terminal idle padding. The padding remains excluded from norm "
            "stats and from the primary valid-action loss."
        ),
    )
    parser.add_argument(
        "--norm-std-floor",
        type=float,
        default=1e-4,
        help="Minimum per-dimension std used for robot state/action normalization.",
    )
    parser.add_argument(
        "--norm-stats",
        default=None,
        help="Optional path to a precomputed robot state/action norm_stats.json. If omitted, stats are computed.",
    )
    parser.add_argument(
        "--max-total-seq-len",
        type=int,
        default=4096,
        help="Maximum frozen audio-interaction context token budget.",
    )
    parser.add_argument(
        "--max-text-seq-len",
        type=int,
        default=None,
        help="Deprecated alias for --max-total-seq-len.",
    )
    parser.add_argument(
        "--text-chunk-time-mode",
        choices=TEXT_CHUNK_TIME_MODES,
        default=DEFAULT_TEXT_CHUNK_TIME_MODE,
        help="Assign timed transcript words to 400ms chunks by word midpoint or word start timestamp.",
    )
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak LR for warmup_cosine, or fixed LR for constant.")
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        help="Final LR for warmup_cosine. Ignored by constant schedule.",
    )
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps before cosine decay.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "warmup_cosine"),
        default="warmup_cosine",
        help="Learning-rate schedule.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init-from",
        default=None,
        help=(
            "Optional unconditioned HRI-MoT checkpoint used to initialize a "
            "six-class emotion run. Mutually exclusive with --resume."
        ),
    )
    parser.add_argument(
        "--metrics-jsonl",
        default=None,
        help="Path for per-step JSONL metrics. Defaults to <output-dir>/metrics.jsonl.",
    )
    parser.add_argument("--use-wandb", action="store_true", help="Log rank-0 metrics to Weights & Biases.")
    parser.add_argument("--wandb-project", default="hri-mot")
    parser.add_argument("--wandb-name", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default=None, choices=("online", "offline", "disabled"))
    parser.add_argument(
        "--tensorboard-dir",
        default=None,
        help="Deprecated compatibility arg. TensorBoard logging is disabled; JSONL and wandb are used instead.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable tqdm progress output. Useful for clean Slurm logs.",
    )

    parser.add_argument("--action-hidden-size", type=int, default=768)
    parser.add_argument("--action-intermediate-size", type=int, default=3072)
    parser.add_argument(
        "--speak-conditioning",
        choices=("predicted", "teacher_forced"),
        default="predicted",
        help=(
            "predicted inserts a learned SpeakQuery and adds a c+1 speech-state BCE loss; "
            "teacher_forced inserts the ground-truth c+1 binary speak embedding."
        ),
    )
    parser.add_argument(
        "--speak-loss-weight",
        type=float,
        default=1.0,
        help="BCE weight for --speak-conditioning predicted; ignored by teacher_forced.",
    )
    parser.add_argument(
        "--use-emotion-conditioning",
        type=parse_bool,
        default=False,
        help=(
            "Add a distinct semantic emotion embedding to the speaking condition. "
            "Six labels plus none are encoded independently, and the "
            "condition is active only while shifted teacher-forced SPEAK is active."
        ),
    )
    parser.add_argument(
        "--emotion-samples-per-class",
        type=int,
        default=0,
        help=(
            "If >0, draw this many episodes from each of the six emotion "
            "categories per epoch, without motion filtering."
        ),
    )
    parser.add_argument(
        "--emotion-film-layers",
        type=int,
        default=4,
        help="Number of final Action Expert blocks with action-token FiLM.",
    )
    parser.add_argument(
        "--response-time-embedding",
        action="store_true",
        help=(
            "Add an action-only embedding of 400ms chunks elapsed since "
            "response onset. Requires teacher_forced SPEAK."
        ),
    )
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")

    parser.add_argument(
        "--no-audio-features",
        action="store_true",
        help="Do not run the frozen audio tower; leave audio PAD positions as token embeddings. Useful for fast smoke tests.",
    )
    parser.add_argument(
        "--audio-loader",
        choices=("whisper", "native"),
        default="whisper",
        help=(
            "whisper decodes/resamples each complete source wav to mono float32 at 16 kHz "
            "before 400 ms slicing, matching Web inference. native keeps the old dataset loader."
        ),
    )
    parser.add_argument(
        "--audio-feature-mode",
        choices=("episode_contiguous", "independent_chunk"),
        default="episode_contiguous",
        help=(
            "episode_contiguous concatenates the selected episode chunks, runs one full-waveform "
            "mel/audio-tower pass, then slices consecutive groups of 10 features. "
            "independent_chunk keeps the old per-400-ms feature extraction for ablations."
        ),
    )
    args = parser.parse_args()
    if args.train_index is None:
        if args.dataset_format == "cospeech":
            args.train_index = (
                DEFAULT_COSPEECH_EMOTION_INDEX
                if args.use_emotion_conditioning
                else DEFAULT_COSPEECH_INDEX
            )
        else:
            args.train_index = TRAIN_INDEX_PATHS[0]
    elif (
        (
            args.use_emotion_conditioning
        )
        and args.dataset_format == "cospeech"
        and args.train_index == DEFAULT_COSPEECH_INDEX
    ):
        args.train_index = DEFAULT_COSPEECH_EMOTION_INDEX
    if args.max_text_seq_len is not None:
        args.max_total_seq_len = args.max_text_seq_len
    if args.speak_loss_weight < 0:
        parser.error("--speak-loss-weight must be non-negative")
    if args.terminal_idle_action_loss_weight < 0:
        parser.error("--terminal-idle-action-loss-weight must be non-negative")
    if args.response_time_embedding and args.speak_conditioning != "teacher_forced":
        parser.error(
            "--response-time-embedding requires "
            "--speak-conditioning teacher_forced"
        )
    if (
        args.use_emotion_conditioning
        and args.dataset_format != "cospeech"
    ):
        parser.error(
            "--use-emotion-conditioning requires --dataset-format cospeech"
        )
    if (
        args.use_emotion_conditioning
        and args.speak_conditioning != "teacher_forced"
    ):
        parser.error(
            "--use-emotion-conditioning requires "
            "--speak-conditioning teacher_forced"
        )
    if args.init_from and args.resume:
        parser.error("--init-from and --resume are mutually exclusive")
    if args.init_from and not args.use_emotion_conditioning:
        parser.error(
            "--init-from requires --use-emotion-conditioning true"
        )
    if args.emotion_film_layers <= 0:
        parser.error("--emotion-film-layers must be positive")
    if args.emotion_samples_per_class < 0:
        parser.error("--emotion-samples-per-class must be non-negative")
    if (
        args.emotion_samples_per_class > 0
        and not args.use_emotion_conditioning
    ):
        parser.error(
            "--emotion-samples-per-class requires "
            "--use-emotion-conditioning true"
        )
    if (
        args.use_emotion_conditioning
        and args.emotion_samples_per_class <= 0
    ):
        parser.error(
            "--use-emotion-conditioning requires a positive "
            "--emotion-samples-per-class for six-class balanced sampling"
        )
    if (
        args.emotion_samples_per_class > 0
        and args.dataset_format != "cospeech"
    ):
        parser.error(
            "--emotion-samples-per-class requires --dataset-format cospeech"
        )
    return args


def setup_distributed(args: argparse.Namespace) -> Tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP training requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
    else:
        if args.device == "auto":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
    return distributed, rank, local_rank, world_size, device


def rank_zero_print(rank: int, *args, **kwargs) -> None:
    if rank == 0:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def distributed_mean(value: torch.Tensor, distributed: bool, world_size: int) -> torch.Tensor:
    value = value.detach().float().clone()
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / world_size
    return value


def seed_runtime_rngs(seed: int, rank: int) -> None:
    rank_seed = int(seed) + int(rank)
    set_seed(rank_seed)
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)


def _row_float(row: Dict[str, Any], key: str, default: float) -> float:
    value = row.get(key)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _stats_from_sums(
    *,
    total: torch.Tensor,
    total_sq: torch.Tensor,
    count: int,
    std_floor: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise RuntimeError("Cannot compute normalization stats from an empty dataset")
    mean = total / float(count)
    var = (total_sq / float(count)) - mean.pow(2)
    std = torch.sqrt(var.clamp_min(0.0)).clamp_min(float(std_floor))
    return mean.float().cpu(), std.float().cpu()


def compute_action_state_norm_stats(
    dataset: Dataset,
    *,
    std_floor: float,
    log_every: int = 1000,
) -> Dict[str, Any]:
    cfg = dataset.config
    state_sum = torch.zeros(cfg.robot_state_dim, dtype=torch.float64)
    state_sum_sq = torch.zeros(cfg.robot_state_dim, dtype=torch.float64)
    action_sum = torch.zeros(cfg.action_dim, dtype=torch.float64)
    action_sum_sq = torch.zeros(cfg.action_dim, dtype=torch.float64)
    state_count = 0
    action_count = 0
    terminal_idle_mask_applied = False

    fast_iterator = getattr(dataset, "iter_action_state_arrays", None)
    if callable(fast_iterator):
        total_episodes = len(dataset)
        for episode_index, arrays in enumerate(fast_iterator(), start=1):
            if not isinstance(arrays, (tuple, list)) or len(arrays) not in (2, 3):
                raise ValueError(
                    "Fast normalization iterator must yield (state, action) or "
                    "(state, action, action_valid_mask)"
                )
            state_np, action_np = arrays[:2]
            action_valid_mask_np = arrays[2] if len(arrays) == 3 else None
            state_np = np.asarray(state_np, dtype=np.float64)
            action_np = np.asarray(action_np, dtype=np.float64)
            if state_np.ndim != 3 or action_np.ndim != 3:
                raise ValueError(
                    "Fast normalization arrays must have shapes "
                    "[chunks, steps, dim], got "
                    f"state={state_np.shape} action={action_np.shape}"
                )
            decision_chunks = min(state_np.shape[0], action_np.shape[0]) - 1
            if decision_chunks <= 0:
                raise ValueError(
                    "Normalization requires at least two physical chunks"
                )
            # The training loop uses C-1 decisions because decision c is
            # paired with the physical c+1 speech label.  Do not let the
            # synthetic terminal action beyond 20 seconds skew the stats.
            state_np = state_np[:decision_chunks]
            action_np = action_np[:decision_chunks]
            if action_valid_mask_np is not None:
                terminal_idle_mask_applied = True
                action_valid_mask_np = np.asarray(
                    action_valid_mask_np[:decision_chunks], dtype=np.bool_
                )
                if action_valid_mask_np.shape != action_np.shape[:2]:
                    raise ValueError(
                        "action_valid_mask must match [chunks, action_steps], got "
                        f"mask={action_valid_mask_np.shape} action={action_np.shape}"
                    )
                # Each action frame has its own loss query.  State tokens are
                # consumed as a whole chunk, so keep all state frames whenever
                # that decision supervises at least one action frame.
                state_valid_mask_np = np.broadcast_to(
                    action_valid_mask_np.any(axis=1, keepdims=True),
                    state_np.shape[:2],
                )
                state_np = state_np[state_valid_mask_np]
                action_np = action_np[action_valid_mask_np]
            else:
                state_np = state_np.reshape(-1, cfg.robot_state_dim)
                action_np = action_np.reshape(-1, cfg.action_dim)
            if state_np.shape[0] <= 0 or action_np.shape[0] <= 0:
                raise ValueError(
                    f"Episode {episode_index} has no valid normalization frames"
                )
            state_sum += torch.from_numpy(state_np.sum(axis=0))
            state_sum_sq += torch.from_numpy((state_np * state_np).sum(axis=0))
            action_sum += torch.from_numpy(action_np.sum(axis=0))
            action_sum_sq += torch.from_numpy((action_np * action_np).sum(axis=0))
            state_count += int(state_np.shape[0])
            action_count += int(action_np.shape[0])
            should_log = log_every > 0 and (
                episode_index % int(log_every) == 0 or episode_index == total_episodes
            )
            if should_log:
                print(
                    f"[norm-stats] {episode_index}/{total_episodes} episodes",
                    flush=True,
                )

        state_mean, state_std = _stats_from_sums(
            total=state_sum,
            total_sq=state_sum_sq,
            count=state_count,
            std_floor=std_floor,
        )
        action_mean, action_std = _stats_from_sums(
            total=action_sum,
            total_sq=action_sum_sq,
            count=action_count,
            std_floor=std_floor,
        )
        return {
            "version": 3 if terminal_idle_mask_applied else 2,
            "state_mean": state_mean,
            "state_std": state_std,
            "action_mean": action_mean,
            "action_std": action_std,
            "state_count": int(state_count),
            "action_count": int(action_count),
            "std_floor": float(std_floor),
            "terminal_idle_mask_applied": bool(terminal_idle_mask_applied),
        }

    for episode in dataset.episodes:
        rows = episode.get("chunks")
        if rows is None:
            rows = dataset._read_episode_rows(episode)
        rows = sorted(
            rows, key=lambda row: int(row.get("episode_chunk_idx", 0))
        )
        for row in rows[:-1]:
            chunk = row.get("chunk", {})
            if not isinstance(chunk, dict):
                continue
            target = chunk.get("target", {})
            if not isinstance(target, dict):
                target = {}
            t_start = _row_float(chunk, "t_start", 0.0)
            t_end = _row_float(chunk, "t_end", t_start + cfg.chunk_size_sec)
            current_qpos, next_qpos = dataset._load_robot_segments(target.get("robot_pkl"), t_end)

            state_np = np.asarray(current_qpos, dtype=np.float64)
            action_np = np.asarray(next_qpos, dtype=np.float64)
            state_sum += torch.from_numpy(state_np.sum(axis=0))
            state_sum_sq += torch.from_numpy((state_np * state_np).sum(axis=0))
            action_sum += torch.from_numpy(action_np.sum(axis=0))
            action_sum_sq += torch.from_numpy((action_np * action_np).sum(axis=0))
            state_count += int(state_np.shape[0])
            action_count += int(action_np.shape[0])

    state_mean, state_std = _stats_from_sums(
        total=state_sum,
        total_sq=state_sum_sq,
        count=state_count,
        std_floor=std_floor,
    )
    action_mean, action_std = _stats_from_sums(
        total=action_sum,
        total_sq=action_sum_sq,
        count=action_count,
        std_floor=std_floor,
    )
    return {
        "version": 2,
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "state_count": int(state_count),
        "action_count": int(action_count),
        "std_floor": float(std_floor),
    }


def _coerce_norm_stats(raw_stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if raw_stats is None:
        return None
    if "state" in raw_stats or "action" in raw_stats:
        state = raw_stats.get("state", {})
        action = raw_stats.get("action", {})
        if not isinstance(state, dict) or not isinstance(action, dict):
            return None
        raw_stats = {
            "version": raw_stats.get("version", 1),
            "state_mean": state.get("mean"),
            "state_std": state.get("std"),
            "action_mean": action.get("mean"),
            "action_std": action.get("std"),
            "state_count": state.get("count", raw_stats.get("state_count", 0)),
            "action_count": action.get("count", raw_stats.get("action_count", 0)),
            "std_floor": raw_stats.get("std_floor", 0.0),
        }
    required = ("state_mean", "state_std", "action_mean", "action_std")
    if not all(key in raw_stats for key in required):
        return None
    stats = {
        "version": int(raw_stats.get("version", 1)),
        "state_mean": torch.as_tensor(raw_stats["state_mean"], dtype=torch.float32).cpu(),
        "state_std": torch.as_tensor(raw_stats["state_std"], dtype=torch.float32).cpu(),
        "action_mean": torch.as_tensor(raw_stats["action_mean"], dtype=torch.float32).cpu(),
        "action_std": torch.as_tensor(raw_stats["action_std"], dtype=torch.float32).cpu(),
        "state_count": int(raw_stats.get("state_count", 0)),
        "action_count": int(raw_stats.get("action_count", 0)),
        "std_floor": float(raw_stats.get("std_floor", 0.0)),
    }
    return stats


def load_norm_stats_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fp:
        raw_stats = json.load(fp)
    stats = _coerce_norm_stats(raw_stats)
    if stats is None:
        raise ValueError(f"Invalid normalization stats file: {path}")
    return stats


def _pack_norm_stats(stats: Dict[str, Any], device: torch.device) -> torch.Tensor:
    return torch.cat(
        [
            stats["state_mean"].to(device=device, dtype=torch.float64),
            stats["state_std"].to(device=device, dtype=torch.float64),
            stats["action_mean"].to(device=device, dtype=torch.float64),
            stats["action_std"].to(device=device, dtype=torch.float64),
            torch.tensor(
                [
                    float(stats.get("version", 1)),
                    float(stats["state_count"]),
                    float(stats["action_count"]),
                    float(stats["std_floor"]),
                ],
                device=device,
                dtype=torch.float64,
            ),
        ]
    )


def _unpack_norm_stats(payload: torch.Tensor, *, state_dim: int, action_dim: int) -> Dict[str, Any]:
    payload = payload.detach().cpu()
    offset = 0
    state_mean = payload[offset: offset + state_dim].float()
    offset += state_dim
    state_std = payload[offset: offset + state_dim].float()
    offset += state_dim
    action_mean = payload[offset: offset + action_dim].float()
    offset += action_dim
    action_std = payload[offset: offset + action_dim].float()
    offset += action_dim
    version = int(round(float(payload[offset].item())))
    state_count = int(round(float(payload[offset + 1].item())))
    action_count = int(round(float(payload[offset + 2].item())))
    std_floor = float(payload[offset + 3].item())
    return {
        "version": version,
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
        "state_count": state_count,
        "action_count": action_count,
        "std_floor": std_floor,
    }


def broadcast_norm_stats(
    stats: Optional[Dict[str, Any]],
    *,
    config: HriMotDataConfig,
    distributed: bool,
    rank: int,
    device: torch.device,
) -> Dict[str, Any]:
    if not distributed:
        if stats is None:
            raise RuntimeError("Normalization stats are missing")
        return stats
    payload_len = 2 * config.robot_state_dim + 2 * config.action_dim + 4
    if rank == 0:
        if stats is None:
            raise RuntimeError("Rank 0 must provide normalization stats for broadcast")
        payload = _pack_norm_stats(stats, device)
    else:
        payload = torch.empty(payload_len, dtype=torch.float64, device=device)
    dist.broadcast(payload, src=0)
    return _unpack_norm_stats(
        payload,
        state_dim=config.robot_state_dim,
        action_dim=config.action_dim,
    )


def norm_stats_for_checkpoint(stats: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "version": int(stats.get("version", 1)),
        "state_mean": stats["state_mean"].detach().cpu(),
        "state_std": stats["state_std"].detach().cpu(),
        "action_mean": stats["action_mean"].detach().cpu(),
        "action_std": stats["action_std"].detach().cpu(),
        "state_count": int(stats.get("state_count", 0)),
        "action_count": int(stats.get("action_count", 0)),
        "std_floor": float(stats.get("std_floor", 0.0)),
    }


def write_norm_stats_json(path: Path, stats: Dict[str, Any]) -> None:
    payload = {
        "version": int(stats.get("version", 1)),
        "state": {
            "mean": stats["state_mean"].detach().cpu().tolist(),
            "std": stats["state_std"].detach().cpu().tolist(),
            "count": int(stats.get("state_count", 0)),
        },
        "action": {
            "mean": stats["action_mean"].detach().cpu().tolist(),
            "std": stats["action_std"].detach().cpu().tolist(),
            "count": int(stats.get("action_count", 0)),
        },
        "std_floor": float(stats.get("std_floor", 0.0)),
    }
    if int(stats.get("version", 1)) >= 3:
        payload["action_supervision_policy"] = (
            "normalization_and_primary_action_loss_keep_pre_listening_and_"
            "source_motion_and_exit_transition;normalization_excludes_"
            "artificial_terminal_idle_per_frame;training_may_add_a_separate_"
            "weighted_terminal_idle_loss"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)


class ActionStateNormalizer:
    def __init__(self, stats: Dict[str, Any], device: torch.device) -> None:
        self.state_mean = stats["state_mean"].to(device=device, dtype=torch.float32)
        self.state_std = stats["state_std"].to(device=device, dtype=torch.float32)
        self.action_mean = stats["action_mean"].to(device=device, dtype=torch.float32)
        self.action_std = stats["action_std"].to(device=device, dtype=torch.float32)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        mean = self.state_mean.to(device=state.device, dtype=state.dtype)
        std = self.state_std.to(device=state.device, dtype=state.dtype)
        return (state - mean) / std

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mean = self.action_mean.to(device=action.device, dtype=action.dtype)
        std = self.action_std.to(device=action.device, dtype=action.dtype)
        return (action - mean) / std

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mean = self.action_mean.to(device=action.device, dtype=action.dtype)
        std = self.action_std.to(device=action.device, dtype=action.dtype)
        return action * std + mean


def validate_lr_args(args: argparse.Namespace) -> None:
    if args.lr <= 0:
        raise ValueError(f"--lr must be positive for {args.lr_schedule}, got {args.lr}")
    if args.min_lr < 0:
        raise ValueError(f"--min-lr must be non-negative, got {args.min_lr}")
    if args.min_lr > args.lr:
        raise ValueError(f"--min-lr ({args.min_lr}) must be <= --lr ({args.lr})")


def learning_rate_for_step(args: argparse.Namespace, step: int) -> float:
    if args.lr_schedule == "constant":
        return float(args.lr)

    warmup_steps = max(0, int(args.lr_warmup_steps))
    step = max(1, int(step))
    if warmup_steps > 0 and step <= warmup_steps:
        return float(args.lr) * float(step) / float(warmup_steps)

    decay_start_step = warmup_steps if warmup_steps > 0 else 1
    decay_steps = max(1, int(args.max_steps) - decay_start_step)
    progress = min(1.0, max(0.0, (step - decay_start_step) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(args.min_lr) + (float(args.lr) - float(args.min_lr)) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def load_base_model_and_tokenizer(checkpoint_dir: str, device: torch.device, seed: int):
    import lightning as L

    model_config_dir, trained_checkpoint, qwen_omni_ckpt, audio_tower_ckpt = resolve_checkpoint_paths(checkpoint_dir)
    set_seed(seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        fabric_devices = [device.index]
        accelerator = "cuda"
    else:
        fabric_devices = 1
        accelerator = "cpu"
    fabric = L.Fabric(
        accelerator=accelerator,
        devices=fabric_devices,
        num_nodes=1,
        strategy="auto",
        precision=get_default_supported_precision(training=False),
        loggers=False,
    )
    model = load_model(fabric, model_config_dir, trained_checkpoint)
    model.to(device)
    model.eval()
    tokenizer = Tokenizer(model_config_dir)
    return model, tokenizer, qwen_omni_ckpt, audio_tower_ckpt


def split_into_chunks(n: int, chunk_size: int) -> List[int]:
    chunks = [chunk_size] * (n // chunk_size)
    if n % chunk_size:
        chunks.append(n % chunk_size)
    return chunks


@torch.no_grad()
def encode_audio_feature_batch(
    audio_batch: torch.Tensor,
    audio_encoder: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    expected_tokens: int = AUDIO_TOKENS_PER_CHUNK,
) -> torch.Tensor:
    import whisper

    feats: List[torch.Tensor] = []
    for audio in audio_batch:
        samples = audio.detach().float().cpu().numpy().astype(np.float32)
        mel = whisper.log_mel_spectrogram(samples, n_mels=128)
        len_feature = mel.shape[1]
        mel_tensor = mel.to(device) if torch.is_tensor(mel) else torch.tensor(mel, device=device)
        feat = audio_encoder(
            mel_tensor,
            torch.tensor(split_into_chunks(len_feature, 40), device=device),
            torch.tensor((len_feature - 1) // 2 + 1, device=device),
        ).last_hidden_state
        if feat.shape[0] < expected_tokens:
            pad = torch.zeros(
                expected_tokens - feat.shape[0], feat.shape[1], device=feat.device, dtype=feat.dtype
            )
            feat = torch.cat([feat, pad], dim=0)
        feat = feat[:expected_tokens].to(device=device, dtype=dtype)
        feats.append(feat)
    return torch.stack(feats, dim=0)


@torch.no_grad()
def encode_episode_audio_feature_batch(
    audio_batch: torch.Tensor,
    audio_encoder: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    expected_tokens_per_chunk: int = AUDIO_TOKENS_PER_CHUNK,
) -> torch.Tensor:
    """Encode each continuous episode once, then split it into 10-token chunks.

    This matches the original audio-interaction online finetune preprocessing:
    the log-mel frontend and audio tower see one continuous waveform, and the
    resulting feature sequence is assigned to language audio spans in order.
    It deliberately does not normalize or otherwise enhance the waveform.
    """
    import whisper

    if audio_batch.dim() != 3:
        raise ValueError(
            f"audio_batch must have shape [B, C, samples], got {tuple(audio_batch.shape)}"
        )

    batch_size, num_chunks, _ = audio_batch.shape
    episode_features: List[torch.Tensor] = []
    for episode_audio in audio_batch:
        samples = (
            episode_audio.detach().float().reshape(-1).cpu().numpy().astype(np.float32)
        )
        mel = whisper.log_mel_spectrogram(samples, n_mels=128)
        len_feature = int(mel.shape[1])
        mel_tensor = mel.to(device) if torch.is_tensor(mel) else torch.tensor(mel, device=device)
        feat = audio_encoder(
            mel_tensor,
            torch.tensor(split_into_chunks(len_feature, 40), device=device),
            torch.tensor((len_feature - 1) // 2 + 1, device=device),
        ).last_hidden_state

        expected_total = int(num_chunks * expected_tokens_per_chunk)
        if int(feat.shape[0]) != expected_total:
            raise RuntimeError(
                "Continuous episode audio feature length mismatch: "
                f"waveform_chunks={num_chunks}, mel_frames={len_feature}, "
                f"audio_tokens={feat.shape[0]}, expected={expected_total}."
            )
        episode_features.append(
            feat.reshape(num_chunks, expected_tokens_per_chunk, feat.shape[-1]).to(
                device=device, dtype=dtype
            )
        )

    if len(episode_features) != batch_size:
        raise RuntimeError(
            f"Encoded {len(episode_features)} episodes for batch_size={batch_size}"
        )
    return torch.stack(episode_features, dim=0)


def listening_block_ids() -> List[int]:
    return [AUDIO_BEGIN] + [PAD] * AUDIO_TOKENS_PER_CHUNK + [ASSISTANT]


def choose_prefix_chunks(num_chunks: int, chunks_per_step: int, sampling: str) -> int:
    fixed_chunks = num_chunks if chunks_per_step <= 0 or chunks_per_step >= num_chunks else chunks_per_step
    if fixed_chunks <= 0:
        raise ValueError(f"Cannot sample from num_chunks={num_chunks}")
    return fixed_chunks


def choose_prefix_chunks_distributed(
    num_chunks: int,
    chunks_per_step: int,
    sampling: str,
    *,
    distributed: bool,
    device: torch.device,
) -> int:
    prefix_chunks = choose_prefix_chunks(num_chunks, chunks_per_step, sampling)
    if distributed:
        value = torch.tensor([prefix_chunks], device=device, dtype=torch.long)
        dist.broadcast(value, src=0)
        prefix_chunks = int(value.item())
    return prefix_chunks


def build_temporal_context_batch(
    tokenizer: Tokenizer,
    batch: Dict[str, Any],
    prefix_chunks: int,
    max_context_seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_chunks = batch["target_action"].shape[:2]
    prefix_chunks = min(prefix_chunks, num_chunks)
    if prefix_chunks <= 0:
        raise ValueError(f"prefix_chunks must be positive, got {prefix_chunks}")

    system_ids = tokenizer.encode(DEFAULT_SYSTEM_PROMPT, bos=False, eos=False).tolist()
    system_prefix = [ONLINE, ENGLISH, SYSTEM, TEXT_BEGIN] + system_ids + [TEXT_END]
    block = listening_block_ids()

    target_response_state = batch.get("target_response_state")
    target_text_plain = batch.get("target_text_plain")
    if target_response_state is None or target_text_plain is None:
        raise ValueError("batch must include target_response_state and target_text_plain for teacher forcing")

    encoded: List[
        Tuple[List[int], List[int], List[int], List[Tuple[int, int, int]]]
    ] = []
    for b in range(batch_size):
        ids = list(system_prefix[:max_context_seq_len])
        chunk_ids = [-1] * len(ids)
        token_types = [CONTEXT_TOKEN_SYSTEM] * len(ids)
        # (physical_chunk_idx, token_start, token_end).  Target speech keeps
        # advancing on the 400 ms physical timeline, but it must not inject
        # synthetic listening blocks into the frozen AudioInteraction text
        # stream between TEXT_BEGIN and TEXT_END.
        audio_spans: List[Tuple[int, int, int]] = []
        in_target_speech = False

        for c in range(prefix_chunks):
            ids_before_chunk = len(ids)
            chunk_ids_before_chunk = len(chunk_ids)
            token_types_before_chunk = len(token_types)
            audio_spans_before_chunk = len(audio_spans)
            in_target_speech_before_chunk = in_target_speech
            state = str(target_response_state[b][c])
            text = str(target_text_plain[b][c]).strip()
            text_ids = tokenizer.encode(text, bos=False, eos=False).tolist() if text else []
            next_state = (
                str(target_response_state[b][c + 1])
                if c + 1 < num_chunks
                else "non_speech"
            )

            # Match the frozen AudioInteraction stream: listening blocks occur
            # before a response starts and resume after TEXT_END, but do not
            # appear inside an open assistant text turn.  The action/speak
            # suffix still carries physical chunk c even when no context token
            # is appended for that interval.
            if not in_target_speech:
                remaining = max_context_seq_len - len(ids)
                if remaining < len(block):
                    break
                start = len(ids)
                ids.extend(block)
                chunk_ids.extend([c] * len(block))
                token_types.extend([CONTEXT_TOKEN_PARTNER_AUDIO] * len(block))
                audio_start = start + 1
                audio_end = start + 1 + AUDIO_TOKENS_PER_CHUNK
                audio_spans.append((c, audio_start, audio_end))
            else:
                # -2 means this physical chunk was intentionally constructed
                # without partner audio because the assistant turn is open.
                # -1 remains reserved for a chunk not built due to truncation.
                audio_spans.append((c, -2, -2))

            target_ids: List[int] = []
            target_types: List[int] = []
            if state == "non_speech":
                if not in_target_speech:
                    target_ids.append(KEEP_SILENCE)
                    target_types.append(CONTEXT_TOKEN_TARGET_CONTROL)
            elif state == "onset":
                target_ids.append(TEXT_BEGIN)
                target_types.append(CONTEXT_TOKEN_TARGET_CONTROL)
                target_ids.extend(text_ids)
                target_types.extend([CONTEXT_TOKEN_TARGET_TEXT] * len(text_ids))
                in_target_speech = True
                if next_state == "non_speech":
                    target_ids.append(TEXT_END)
                    target_types.append(CONTEXT_TOKEN_TARGET_TEXT)
                    in_target_speech = False
            elif state == "speaking":
                target_ids.extend(text_ids)
                target_types.extend([CONTEXT_TOKEN_TARGET_TEXT] * len(text_ids))
                in_target_speech = True
                if next_state == "non_speech":
                    target_ids.append(TEXT_END)
                    target_types.append(CONTEXT_TOKEN_TARGET_TEXT)
                    in_target_speech = False
            else:
                raise ValueError(f"Unknown target response_state {state!r}")

            remaining = max_context_seq_len - len(ids)
            if remaining < len(target_ids):
                # The listening block/audio span is appended before target
                # text is tokenized. Roll the whole physical chunk back so
                # -1 continues to mean "not built/truncated"; otherwise an
                # in-response -2 span could hide a truncated text chunk.
                del ids[ids_before_chunk:]
                del chunk_ids[chunk_ids_before_chunk:]
                del token_types[token_types_before_chunk:]
                del audio_spans[audio_spans_before_chunk:]
                in_target_speech = in_target_speech_before_chunk
                break
            ids.extend(target_ids)
            chunk_ids.extend([c] * len(target_ids))
            token_types.extend(target_types)

        encoded.append((ids, chunk_ids, token_types, audio_spans))

    max_len = max(len(ids) for ids, _, _, _ in encoded)
    input_ids = torch.full((batch_size, max_len), PAD, dtype=torch.long, device=device)
    valid_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    context_chunk_ids = torch.full((batch_size, max_len), -1, dtype=torch.long, device=device)
    context_token_types = torch.full((batch_size, max_len), CONTEXT_TOKEN_SYSTEM, dtype=torch.long, device=device)
    audio_positions = torch.full((batch_size, prefix_chunks, 2), -1, dtype=torch.long, device=device)

    for row, (ids, chunks, token_types, spans) in enumerate(encoded):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids, dtype=torch.long, device=device)
        context_chunk_ids[row, :length] = torch.tensor(chunks, dtype=torch.long, device=device)
        context_token_types[row, :length] = torch.tensor(token_types, dtype=torch.long, device=device)
        valid_mask[row, :length] = True
        for physical_chunk_idx, start, end in spans:
            audio_positions[row, physical_chunk_idx, 0] = start
            audio_positions[row, physical_chunk_idx, 1] = end

    return input_ids, valid_mask, context_chunk_ids, context_token_types, audio_positions


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=pred.device, dtype=torch.bool)
    if not mask.any():
        return pred.sum() * 0.0
    error = (pred.float() - target.float()).pow(2)
    weight = mask.unsqueeze(-1).to(dtype=error.dtype)
    return (error * weight).sum() / (weight.sum() * pred.shape[-1]).clamp_min(1.0)


def terminal_idle_masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    distributed: bool = False,
    world_size: int = 1,
) -> torch.Tensor:
    """Globally normalize sparse terminal-idle frames under DDP.

    Unlike the unchanged primary ``masked_mse_loss``, this helper must handle
    ranks with no local terminal frames. Every rank therefore participates in
    the active-count collective before deciding whether the global mask is
    empty. Scaling the local numerator by world size compensates for DDP's
    later gradient average.
    """

    if tuple(pred.shape) != tuple(target.shape):
        raise ValueError(
            f"pred and target shapes must match, got {tuple(pred.shape)} and "
            f"{tuple(target.shape)}"
        )
    mask = mask.to(device=pred.device, dtype=torch.bool)
    if tuple(mask.shape) != tuple(pred.shape[:-1]):
        raise ValueError(
            f"mask {tuple(mask.shape)} must match prediction prefix "
            f"{tuple(pred.shape[:-1])}"
        )

    active_count = mask.sum(dtype=torch.float32)
    ddp_scale = 1.0
    if distributed:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "distributed terminal-idle MSE requires an initialized process group"
            )
        actual_world_size = int(dist.get_world_size())
        if int(world_size) != actual_world_size:
            raise RuntimeError(
                "terminal-idle MSE world-size mismatch: "
                f"argument={world_size} process_group={actual_world_size}"
            )
        dist.all_reduce(active_count, op=dist.ReduceOp.SUM)
        ddp_scale = float(actual_world_size)
    elif int(world_size) != 1:
        raise ValueError(
            f"world_size must be 1 when distributed=False, got {world_size}"
        )

    if float(active_count.item()) <= 0.0:
        return pred.sum() * 0.0
    error = (pred.float() - target.float()).pow(2)
    weight = mask.unsqueeze(-1).to(dtype=error.dtype)
    numerator = (error * weight).sum()
    denominator = active_count * int(pred.shape[-1])
    return numerator * ddp_scale / denominator.clamp_min(1.0)


def split_valid_and_terminal_action_masks(
    target_action_valid_mask: torch.Tensor,
    decision_selection_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a selected decision window into valid and trailing-idle frames.

    ``target_action_valid_mask`` has shape ``[B, C, H]``. Its false suffix is
    the artificial idle used to pad a co-speech episode to 20 seconds. Only
    that trailing false suffix is eligible for terminal-idle supervision; an
    unexpected false hole before a later valid frame remains unsupervised.
    Both returned masks are flattened to ``[B, C * H]`` for action queries.
    """

    if target_action_valid_mask.dim() != 3:
        raise ValueError(
            "target_action_valid_mask must have shape [B,C,H], got "
            f"{tuple(target_action_valid_mask.shape)}"
        )
    if decision_selection_mask.dim() != 2:
        raise ValueError(
            "decision_selection_mask must have shape [B,C], got "
            f"{tuple(decision_selection_mask.shape)}"
        )
    batch_size, decisions, action_horizon = target_action_valid_mask.shape
    if tuple(decision_selection_mask.shape) != (batch_size, decisions):
        raise ValueError(
            "decision selection must match the first two valid-mask dims, got "
            f"selection={tuple(decision_selection_mask.shape)} "
            f"valid={tuple(target_action_valid_mask.shape)}"
        )

    valid = target_action_valid_mask.to(dtype=torch.bool).reshape(
        batch_size, decisions * action_horizon
    )
    selected = decision_selection_mask.to(
        device=valid.device, dtype=torch.bool
    ).repeat_interleave(action_horizon, dim=1)

    # A terminal frame cannot have any valid frame after it. This preserves
    # the dataset's half-open per-frame boundary while refusing to interpret
    # an internal invalid hole as terminal padding.
    has_valid_at_or_after = valid.flip(dims=(1,)).cummax(dim=1).values.flip(
        dims=(1,)
    )
    terminal_suffix = ~valid & ~has_valid_at_or_after
    return selected & valid, selected & terminal_suffix


def masked_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if not mask.any():
        return logits.sum() * 0.0
    error = F.binary_cross_entropy_with_logits(
        logits.float(),
        target.to(device=logits.device, dtype=torch.float32),
        reduction="none",
    )
    weight = mask.to(dtype=error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1.0)


def trainable_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


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


def validate_emotion_initialization_checkpoint(
    args: argparse.Namespace,
    checkpoint: Dict[str, Any],
) -> Dict[str, Any]:
    params = checkpoint.get("trainable_params")
    if not isinstance(params, dict):
        raise ValueError("Emotion initialization checkpoint has no trainable_params")
    source_args = checkpoint.get("args", {})
    if not isinstance(source_args, dict):
        raise ValueError("Emotion initialization checkpoint has no valid args")
    if bool(
        source_args.get("use_emotion_conditioning", False)
    ):
        raise ValueError(
            "--init-from must be an unconditioned baseline checkpoint"
        )
    if str(source_args.get("dataset_format", "unknown")) != "cospeech":
        raise ValueError(
            "--init-from must use a cospeech baseline checkpoint, got "
            f"dataset_format={source_args.get('dataset_format', 'unknown')!r}"
        )

    architecture = {
        "action_hidden_size": (
            int(source_args.get("action_hidden_size", 768)),
            int(args.action_hidden_size),
        ),
        "action_intermediate_size": (
            int(source_args.get("action_intermediate_size", 3072)),
            int(args.action_intermediate_size),
        ),
        "speak_conditioning": (
            str(source_args.get("speak_conditioning", "predicted")),
            str(args.speak_conditioning),
        ),
        "response_time_embedding": (
            bool(source_args.get("response_time_embedding", False)),
            bool(args.response_time_embedding),
        ),
    }
    mismatches = {
        name: {"checkpoint": source, "requested": requested}
        for name, (source, requested) in architecture.items()
        if source != requested
    }
    if mismatches:
        raise ValueError(
            "Initialization checkpoint architecture mismatch: "
            f"{mismatches}"
        )
    return source_args


def load_emotion_finetune_weights_strict(
    model: torch.nn.Module,
    checkpoint: Dict[str, Any],
    *,
    device: torch.device,
) -> None:
    """Restore every baseline action parameter and leave only new conditions."""

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
    if bool(
        source_args.get("use_emotion_conditioning", False)
    ):
        raise ValueError(
            "--init-from must be an unconditioned baseline checkpoint"
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

    required_source = set(named) - condition_names
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
            target.copy_(
                source.to(device=target.device, dtype=target.dtype)
            )


def load_trainable_state(
    model: torch.nn.Module,
    path: str,
    device: torch.device,
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location=device)
    params = checkpoint.get("trainable_params", checkpoint)
    if not isinstance(params, dict):
        raise ValueError(f"Checkpoint {path} has no trainable parameter mapping")
    named_params = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected = set(named_params)
    available = set(params)
    missing = sorted(expected - available)
    unexpected = sorted(available - expected)
    if missing or unexpected:
        raise ValueError(
            "Checkpoint/model trainable parameter mismatch; refusing to leave "
            "a conditioning branch randomly initialized: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    with torch.no_grad():
        for name in sorted(expected):
            tensor = params[name]
            target = named_params[name]
            if tuple(tensor.shape) != tuple(target.shape):
                raise ValueError(
                    f"Checkpoint shape mismatch for {name}: "
                    f"checkpoint={tuple(tensor.shape)} model={tuple(target.shape)}"
                )
            target.copy_(
                tensor.to(device=target.device, dtype=target.dtype)
            )
    return int(checkpoint.get("step", 0)), 0


def save_checkpoint(
    path: Path,
    wrapper: HriMoTWrapper,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    norm_stats: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "trainable_params": trainable_state_dict(wrapper),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "norm_stats": norm_stats_for_checkpoint(norm_stats),
        },
        path,
    )


def create_metric_loggers(
    args: argparse.Namespace,
    out_dir: Path,
    *,
    rank: int,
) -> Tuple[Optional[TextIO], Optional[Any]]:
    if rank != 0:
        return None, None

    metrics_path = Path(args.metrics_jsonl) if args.metrics_jsonl else out_dir / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_mode = "a" if args.resume else "w"
    metrics_fp = open(metrics_path, metrics_mode, encoding="utf-8", buffering=1)
    print(f"metrics jsonl: {metrics_path}", flush=True)

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
        except Exception as exc:
            raise RuntimeError(
                "wandb logging requested but wandb is not importable. "
                "Install wandb or run with USE_WANDB=0."
            ) from exc

        wandb_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_name or out_dir.name,
            "entity": args.wandb_entity,
            "config": vars(args),
            "dir": str(out_dir),
            "resume": "allow" if args.resume else None,
        }
        if args.wandb_mode:
            wandb_kwargs["mode"] = args.wandb_mode
        wandb_run = wandb.init(**{k: v for k, v in wandb_kwargs.items() if v is not None})
        print(f"wandb run: {wandb_run.name}", flush=True)
    else:
        print("wandb disabled", flush=True)

    if args.tensorboard_dir:
        print("--tensorboard-dir is ignored; TensorBoard logging is disabled", flush=True)
    return metrics_fp, wandb_run


def write_training_metrics(
    *,
    metrics_fp: Optional[TextIO],
    wandb_run: Optional[Any],
    step: int,
    metrics: Dict[str, Any],
) -> None:
    if metrics_fp is not None:
        metrics_fp.write(json.dumps({"step": step, **metrics}, ensure_ascii=False) + "\n")
        metrics_fp.flush()

    if wandb_run is not None:
        wandb_run.log({f"train/{key}": value for key, value in metrics.items()}, step=step)


def cycle(loader, sampler=None):
    epoch = 0
    while True:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1


def main() -> None:
    args = parse_args()
    distributed, rank, local_rank, world_size, device = setup_distributed(args)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    seed_runtime_rngs(args.seed, rank)

    initialization_checkpoint = None
    initialization_norm_stats = None
    if args.init_from:
        loaded_checkpoint = load_checkpoint_cpu(args.init_from)
        validate_emotion_initialization_checkpoint(
            args, loaded_checkpoint
        )
        initialization_norm_stats = _coerce_norm_stats(
            loaded_checkpoint.get("norm_stats")
        )
        if initialization_norm_stats is None:
            raise ValueError(
                "Initialization checkpoint has no normalization statistics; "
                "conditioning fine-tuning must preserve the baseline normalization space"
            )
        if int(initialization_norm_stats.get("version", 1)) < 3:
            raise ValueError(
                "Initialization checkpoint normalization version is "
                f"{int(initialization_norm_stats.get('version', 1))}, expected "
                "co-speech v3 or newer"
            )
        initialization_checkpoint = {
            "step": int(loaded_checkpoint.get("step", 0)),
            "trainable_params": loaded_checkpoint["trainable_params"],
            "args": loaded_checkpoint["args"],
        }
        del loaded_checkpoint

    args.initialization_mode = (
        "resume"
        if args.resume
        else "emotion_finetune"
        if args.init_from and args.use_emotion_conditioning
        else "random"
    )
    args.initialization_source_checkpoint = (
        str(args.init_from) if args.init_from else None
    )
    args.initialization_source_step = (
        int(initialization_checkpoint["step"])
        if initialization_checkpoint is not None
        else None
    )
    args.initialization_weights_loaded = bool(args.init_from)
    args.emotion_mapping_version = (
        EMOTION_MAPPING_VERSION
        if args.use_emotion_conditioning
        else None
    )
    args.emotion_sampling_mode = (
        "uniform_six_emotions"
        if args.emotion_samples_per_class > 0
        else "uniform_dataset"
    )
    args.emotion_samples_per_epoch = (
        len(EMOTION_LABELS) * int(args.emotion_samples_per_class)
        if args.emotion_samples_per_class > 0
        else None
    )
    args.emotion_samples_per_rank = (
        args.emotion_samples_per_epoch // world_size
        if args.emotion_samples_per_epoch is not None
        and args.emotion_samples_per_epoch % world_size == 0
        else None
    )
    args.emotion_source_class_counts = None
    args.emotion_eligible_class_counts = None
    args.emotion_label_quotas = None
    args.emotion_effective_repetitions_per_epoch = None
    args.normalization_source = (
        f"init_checkpoint:{args.init_from}"
        if args.init_from
        else f"resume_checkpoint:{args.resume}"
        if args.resume
        else f"json:{args.norm_stats}"
        if args.norm_stats
        else "computed_from_training_dataset"
    )

    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "args.json", "w", encoding="utf-8") as fp:
            json.dump(vars(args), fp, indent=2, ensure_ascii=False)
    metrics_fp, wandb_run = create_metric_loggers(args, out_dir, rank=rank)
    if distributed:
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)

    data_config = HriMotDataConfig(
        train_index_paths=[args.train_index],
        max_episodes=args.max_episodes,
        strict_missing_files=True,
        include_partner_state=False,
        include_text_fields=True,
        text_chunk_time_mode=args.text_chunk_time_mode,
        audio_loader=args.audio_loader,
    )
    episode_index_cache = None
    if args.dataset_format == "episode_jsonl" and args.max_episodes is None:
        episode_index_cache = args.episode_index_cache or f"{args.train_index}.hri_mot_offsets.jsonl"

    def create_dataset():
        if args.dataset_format == "cospeech":
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
        return HriMotEpisodeDataset(
            data_config.train_index_paths,
            config=data_config,
            split="train",
            episode_index_cache=episode_index_cache,
        )

    if distributed and episode_index_cache is not None:
        if rank == 0:
            rank_zero_print(rank, f"building/loading episode offset cache: {episode_index_cache}")
            create_dataset()
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)

    dataset = create_dataset()
    if args.dataset_format == "cospeech":
        rank_zero_print(
            rank,
            "primary action supervision: keep pre-listening idle/source "
            "motion/exit transition; artificial terminal idle is excluded "
            "from the primary loss and trained by its separate weighted loss",
        )
    if args.emotion_samples_per_class > 0:
        sampler = DistributedBalancedEmotionSampler(
            dataset,
            samples_per_class=args.emotion_samples_per_class,
            num_replicas=world_size,
            rank=rank,
            seed=args.seed,
        )
        args.emotion_source_class_counts = dict(
            sampler.source_class_counts
        )
        args.emotion_eligible_class_counts = dict(
            sampler.eligible_class_counts
        )
        args.emotion_label_quotas = dict(sampler.label_quotas)
        args.emotion_effective_repetitions_per_epoch = dict(
            sampler.effective_repetitions_per_epoch
        )
        args.emotion_samples_per_rank = len(sampler)
        rank_zero_print(
            rank,
            "six-class emotion sampling without motion filtering; "
            f"eligible={sampler.eligible_class_counts}; "
            f"label_quotas={sampler.label_quotas}; "
            "effective_repetitions="
            f"{sampler.effective_repetitions_per_epoch}; "
            f"global_per_epoch={sampler.global_num_samples}; "
            f"per_rank={len(sampler)}",
        )
        if rank == 0:
            with open(out_dir / "args.json", "w", encoding="utf-8") as fp:
                json.dump(vars(args), fp, indent=2, ensure_ascii=False)
            if wandb_run is not None:
                wandb_run.config.update(
                    {
                        "emotion_source_class_counts": (
                            args.emotion_source_class_counts
                        ),
                        "emotion_eligible_class_counts": (
                            args.emotion_eligible_class_counts
                        ),
                        "emotion_label_quotas": args.emotion_label_quotas,
                        "emotion_effective_repetitions_per_epoch": (
                            args.emotion_effective_repetitions_per_epoch
                        ),
                        "emotion_samples_per_rank": (
                            args.emotion_samples_per_rank
                        ),
                    },
                    allow_val_change=True,
                )
    else:
        sampler = (
            DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )
            if distributed
            else None
        )
    loader = DataLoader(
        dataset,
        batch_size=args.episode_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
        collate_fn=HriMotEpisodeCollator(),
        drop_last=False,
    )
    if episode_index_cache is not None:
        rank_zero_print(rank, f"episode offset cache: {episode_index_cache}")
    rank_zero_print(rank, f"loaded episodes: {len(dataset)}")
    rank_zero_print(rank, f"dataset format: {args.dataset_format}")
    rank_zero_print(rank, f"distributed: {distributed} world_size={world_size}")
    rank_zero_print(rank, f"steps per sampling epoch: {len(loader)}")
    rank_zero_print(rank, f"text chunk time mode: {data_config.text_chunk_time_mode}")
    rank_zero_print(
        rank,
        f"audio input: loader={data_config.audio_loader} sample_rate={data_config.audio_sample_rate} "
        f"feature_mode={args.audio_feature_mode} enhancement=none",
    )
    if len(dataset) == 0:
        raise RuntimeError("No training episodes loaded")

    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu")
        resume_args = resume_checkpoint.get("args", {})
        checkpoint_response_time_embedding = bool(
            resume_args.get("response_time_embedding", False)
        )
        if checkpoint_response_time_embedding != bool(
            args.response_time_embedding
        ):
            raise ValueError(
                "Cannot change --response-time-embedding when resuming: "
                f"checkpoint={checkpoint_response_time_embedding}, "
                f"requested={bool(args.response_time_embedding)}. "
                "Start a new training run for this architecture change."
            )
        checkpoint_motion_conditioning = bool(
            resume_args.get("use_emotion_conditioning", False)
        )
        if checkpoint_motion_conditioning != bool(
            args.use_emotion_conditioning
        ):
            raise ValueError(
                "Cannot change --use-emotion-conditioning when resuming: "
                f"checkpoint={checkpoint_motion_conditioning}, "
                f"requested={bool(args.use_emotion_conditioning)}. "
                "Start a new training run for this architecture change."
            )
        checkpoint_emotion_samples_per_class = int(
            resume_args.get("emotion_samples_per_class", 0)
        )
        if checkpoint_emotion_samples_per_class != int(
            args.emotion_samples_per_class
        ):
            raise ValueError(
                "Cannot change --emotion-samples-per-class when resuming: "
                f"checkpoint={checkpoint_emotion_samples_per_class}, "
                f"requested={int(args.emotion_samples_per_class)}. "
                "Pass the original sampling quota when resuming."
            )
    norm_stats = initialization_norm_stats
    if norm_stats is None:
        norm_stats = _coerce_norm_stats(
            resume_checkpoint.get("norm_stats")
            if resume_checkpoint is not None
            else None
        )
    if initialization_norm_stats is not None:
        rank_zero_print(
            rank,
            "using robot state/action normalization from initialization "
            f"checkpoint: {args.init_from}",
        )
    if norm_stats is None and args.norm_stats and rank == 0:
        rank_zero_print(rank, f"loading robot state/action normalization stats: {args.norm_stats}")
        norm_stats = load_norm_stats_json(args.norm_stats)
    if norm_stats is None and rank == 0:
        rank_zero_print(rank, "computing robot state/action normalization stats")
        norm_stats = compute_action_state_norm_stats(dataset, std_floor=args.norm_std_floor)
    if rank == 0 and norm_stats is not None:
        required_norm_version = 3 if args.dataset_format == "cospeech" else 2
        if int(norm_stats.get("version", 1)) < required_norm_version:
            source = args.resume if resume_checkpoint is not None else args.norm_stats
            raise ValueError(
                f"Normalization stats from {source or 'runtime'} have version "
                f"{int(norm_stats.get('version', 1))}, but "
                f"dataset_format={args.dataset_format} requires version "
                f"{required_norm_version}. Co-speech v3 excludes artificial "
                "terminal idle from normalization and the primary valid-action "
                "loss; any terminal-idle training uses a separate weighted "
                "term. Regenerate stats and start a new run instead of "
                "resuming an older checkpoint."
            )
    norm_stats = broadcast_norm_stats(
        norm_stats,
        config=data_config,
        distributed=distributed,
        rank=rank,
        device=device,
    )
    normalizer = ActionStateNormalizer(norm_stats, device=device)
    if rank == 0:
        write_norm_stats_json(out_dir / "norm_stats.json", norm_stats)
    rank_zero_print(
        rank,
        "normalization: "
        f"state_count={norm_stats['state_count']} action_count={norm_stats['action_count']} "
        f"std_floor={norm_stats['std_floor']:g}",
    )

    base_model, tokenizer, qwen_omni_ckpt, audio_tower_ckpt = load_base_model_and_tokenizer(
        args.checkpoint_dir, device, args.seed
    )
    audio_encoder = None
    if not args.no_audio_features:
        audio_encoder = load_audio_encoder(qwen_omni_ckpt, audio_tower_ckpt, device)
        audio_encoder.eval()

    wrapper = HriMoTWrapper(
        base_model=base_model,
        decoder_path="_forward_module.transformer.h",
        robot_state_dim=data_config.robot_state_dim,
        action_dim=data_config.action_dim,
        action_hidden_size=args.action_hidden_size,
        action_intermediate_size=args.action_intermediate_size,
        robot_state_steps=data_config.robot_steps_per_chunk,
        speak_conditioning=args.speak_conditioning,
        use_response_time_embedding=bool(args.response_time_embedding),
        use_emotion_conditioning=bool(
            args.use_emotion_conditioning
        ),
        emotion_film_layers=args.emotion_film_layers,
    ).to(device)
    if initialization_checkpoint is not None:
        load_emotion_finetune_weights_strict(
            wrapper,
            initialization_checkpoint,
            device=device,
        )
        rank_zero_print(
            rank,
            "conditioning fine-tune initialization: restored all baseline action "
            f"parameters from {args.init_from} "
            f"(source step={args.initialization_source_step}); conditioning "
            "embedding is new; optimizer/global step restart from zero",
        )
        del initialization_checkpoint
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
        [param for param in train_model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    validate_lr_args(args)

    start_step = 0
    if args.resume:
        start_step, missing = load_trainable_state(raw_wrapper, args.resume, device)
        checkpoint = torch.load(args.resume, map_location=device)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        rank_zero_print(rank, f"resumed from {args.resume} at step={start_step}, missing_trainable_keys={missing}")

    seed_runtime_rngs(args.seed, rank)

    rank_zero_print(rank, f"base frozen: {raw_wrapper.base_parameters_frozen()}")
    rank_zero_print(rank, f"trainable params: {raw_wrapper.trainable_parameter_count():,}")
    rank_zero_print(
        rank,
        f"speak conditioning: {args.speak_conditioning} "
        f"loss_weight={args.speak_loss_weight:g} "
        f"response_time_embedding={args.response_time_embedding} "
        f"emotion={args.use_emotion_conditioning}",
    )
    rank_zero_print(
        rank,
        "action losses: valid=1 "
        f"terminal_idle={args.terminal_idle_action_loss_weight:g}",
    )
    rank_zero_print(rank, f"audio features: {not args.no_audio_features}")
    rank_zero_print(
        rank,
        f"lr schedule: {args.lr_schedule} peak={args.lr:g} min={args.min_lr:g} warmup={args.lr_warmup_steps}",
    )

    data_iter = cycle(loader, sampler=sampler)
    disable_tqdm = rank != 0 or args.disable_tqdm or not sys.stderr.isatty()
    progress = tqdm(
        range(start_step + 1, args.max_steps + 1),
        desc="hri-mot train",
        disable=disable_tqdm,
    )
    for step in progress:
        current_lr = learning_rate_for_step(args, step)
        set_optimizer_lr(optimizer, current_lr)
        train_model.train()
        raw_wrapper.base_model.eval()
        if audio_encoder is not None:
            audio_encoder.eval()

        batch = next(data_iter)
        dtype = next(raw_wrapper.base_model.parameters()).dtype

        batch_size, num_chunks = batch["target_action"].shape[:2]
        available_decisions = int(num_chunks) - 1
        if available_decisions <= 0:
            raise RuntimeError(
                "Speak conditioning requires at least two physical chunks so decision c "
                "has target_response_state[c+1]"
            )
        prefix_chunks = choose_prefix_chunks_distributed(
            num_chunks=available_decisions,
            chunks_per_step=args.chunks_per_step,
            sampling=args.temporal_prefix_sampling,
            distributed=distributed,
            device=device,
        )
        current_prefix_chunks = prefix_chunks
        robot_steps = int(batch["robot_state"].shape[2])
        action_horizon = int(batch["target_action"].shape[2])
        robot_token_count = int(prefix_chunks * robot_steps)
        speak_token_count = int(prefix_chunks)
        action_token_count = int(prefix_chunks * action_horizon)

        input_ids, context_valid_mask, context_chunk_ids, context_token_types, audio_positions = build_temporal_context_batch(
            tokenizer=tokenizer,
            batch=batch,
            prefix_chunks=prefix_chunks,
            max_context_seq_len=args.max_total_seq_len,
            device=device,
        )
        # -1 means truncation; -2 means no partner-audio block was inserted
        # intentionally while the target response was open.
        if audio_positions[:, -1, 0].eq(-1).any():
            raise RuntimeError(
                f"max_total_seq_len={args.max_total_seq_len} is too small for {prefix_chunks} audio chunks"
            )

        audio_features = None
        if audio_encoder is not None:
            partner_audio = batch["partner_audio"][:, :prefix_chunks]
            if args.audio_feature_mode == "episode_contiguous":
                audio_features = encode_episode_audio_feature_batch(
                    partner_audio,
                    audio_encoder,
                    device,
                    dtype,
                )
            elif args.audio_feature_mode == "independent_chunk":
                flat_audio = partner_audio.reshape(-1, partner_audio.shape[-1])
                audio_features = encode_audio_feature_batch(
                    flat_audio,
                    audio_encoder,
                    device,
                    dtype,
                )
                audio_features = audio_features.view(
                    batch_size, prefix_chunks, AUDIO_TOKENS_PER_CHUNK, -1
                )
            else:
                raise ValueError(f"Unknown audio_feature_mode: {args.audio_feature_mode}")

        robot_state = batch["robot_state"][:, :prefix_chunks].to(device=device)
        robot_state = normalizer.normalize_state(robot_state)

        target_action = batch["target_action"][:, :prefix_chunks].to(device=device)
        target_action = normalizer.normalize_action(target_action)
        batch_action_valid_mask = batch.get("target_action_valid_mask")
        if batch_action_valid_mask is None:
            if args.dataset_format == "cospeech":
                raise RuntimeError(
                    "Co-speech dataset must provide target_action_valid_mask"
                )
            target_action_valid_mask = torch.ones(
                target_action.shape[:3], device=device, dtype=torch.bool
            )
        else:
            target_action_valid_mask = batch_action_valid_mask[
                :, :prefix_chunks
            ].to(device=device, dtype=torch.bool)
            if tuple(target_action_valid_mask.shape) != tuple(
                target_action.shape[:3]
            ):
                raise RuntimeError(
                    "target_action_valid_mask must match target_action [B,C,H], "
                    f"got mask={tuple(target_action_valid_mask.shape)} "
                    f"action={tuple(target_action.shape)}"
                )
        speak_labels = batch["target_state_ids"][:, 1:prefix_chunks + 1].to(
            device=device, dtype=torch.long
        )
        if tuple(speak_labels.shape) != (batch_size, prefix_chunks):
            raise RuntimeError(
                f"Expected shifted speak labels {(batch_size, prefix_chunks)}, "
                f"got {tuple(speak_labels.shape)}"
            )
        emotion_labels = None
        emotion_chunk_ids = None
        if args.use_emotion_conditioning:
            sample_emotion = batch.get("target_emotion_id")
            if sample_emotion is None:
                raise RuntimeError(
                    "Emotion CoSpeech batch lacks "
                    "target_emotion_id; use an emotion-tagged index"
                )
            sample_emotion = sample_emotion.to(
                device=device, dtype=torch.long
            ).reshape(batch_size, 1)
            emotion_labels = torch.where(
                speak_labels.bool(),
                sample_emotion.expand(-1, prefix_chunks),
                torch.zeros_like(speak_labels),
            )
        response_elapsed_chunks = None
        if args.response_time_embedding:
            response_elapsed_chunks = shifted_response_elapsed_chunks(
                batch["target_state_ids"].to(device=device, dtype=torch.long),
                decisions=prefix_chunks,
            )
        action_dim = target_action.shape[3]
        decision_chunk_ids = torch.arange(prefix_chunks, device=device, dtype=torch.long)
        x_tau, tau, target_velocity = sample_flow_matching_pair(target_action)
        target_velocity = target_velocity.reshape(batch_size, prefix_chunks * action_horizon, action_dim)
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
        if args.loss_last_n_chunks > 0:
            loss_start_chunk = max(0, prefix_chunks - args.loss_last_n_chunks)
            selected_decision_mask = speak_chunk_ids >= loss_start_chunk
        else:
            selected_decision_mask = torch.ones_like(
                speak_chunk_ids, dtype=torch.bool
            )
        # Keep pre-response listening idle and the exit transition supervised;
        # their valid-frame loss remains unchanged. The artificial trailing
        # idle padding gets its own separately normalized, low-weight loss.
        action_loss_mask, terminal_idle_action_loss_mask = (
            split_valid_and_terminal_action_masks(
                target_action_valid_mask,
                selected_decision_mask,
            )
        )
        terminal_valid_decisions = target_action_valid_mask.any(dim=2)
        decision_loss_mask = selected_decision_mask & terminal_valid_decisions
        flow_loss = masked_mse_loss(
            outputs["action_velocity"],
            target_velocity,
            action_loss_mask,
        )
        terminal_idle_flow_loss = terminal_idle_masked_mse_loss(
            outputs["action_velocity"],
            target_velocity,
            terminal_idle_action_loss_mask,
            distributed=distributed,
            world_size=world_size,
        )
        action_loss = (
            flow_loss
            + args.terminal_idle_action_loss_weight * terminal_idle_flow_loss
        )
        if args.speak_conditioning == "predicted":
            speak_logits = outputs.get("speak_logits")
            if speak_logits is None:
                raise RuntimeError("predicted speak conditioning did not return speak_logits")
            speak_loss = masked_binary_cross_entropy_with_logits(
                speak_logits,
                speak_labels,
                decision_loss_mask,
            )
            with torch.no_grad():
                speak_correct = (speak_logits >= 0) == speak_labels.bool()
                speak_accuracy = speak_correct[decision_loss_mask].float().mean()
            loss = action_loss + args.speak_loss_weight * speak_loss
        else:
            speak_loss = flow_loss.new_zeros(())
            speak_accuracy = flow_loss.new_zeros(())
            loss = action_loss
        chunk_batch = batch_size * prefix_chunks
        total_seq_len = (
            int(input_ids.shape[1])
            + robot_token_count
            + speak_token_count
            + action_token_count
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [param for param in train_model.parameters() if param.requires_grad], args.grad_clip
            )
        emotion_grad_metrics: Dict[str, float] = {}
        if raw_wrapper.emotion_embedding is not None:
            emotion_grad = raw_wrapper.emotion_embedding.weight.grad
            if emotion_grad is not None:
                emotion_grad = emotion_grad.detach().float()
                emotion_grad_metrics = {
                    f"emotion_grad_{label}_norm": float(emotion_grad[index].norm().item())
                    for label, index in {"none": NO_EMOTION_ID, **EMOTION_LABEL_TO_CONDITION_ID}.items()
                }
        optimizer.step()

        log_loss = distributed_mean(loss, distributed, world_size)
        log_action_loss = distributed_mean(action_loss, distributed, world_size)
        log_flow_loss = distributed_mean(flow_loss, distributed, world_size)
        log_terminal_idle_flow_loss = distributed_mean(
            terminal_idle_flow_loss, distributed, world_size
        )
        log_speak_loss = distributed_mean(speak_loss, distributed, world_size)
        log_speak_accuracy = distributed_mean(speak_accuracy, distributed, world_size)
        valid_action_fraction = target_action_valid_mask.float().mean()
        optimized_action_fraction = action_loss_mask.float().mean()
        terminal_idle_action_fraction = (
            terminal_idle_action_loss_mask.float().mean()
        )
        total_optimized_action_mask = action_loss_mask
        if args.terminal_idle_action_loss_weight > 0:
            total_optimized_action_mask = (
                total_optimized_action_mask | terminal_idle_action_loss_mask
            )
        total_optimized_action_fraction = total_optimized_action_mask.float().mean()
        log_valid_action_fraction = distributed_mean(
            valid_action_fraction, distributed, world_size
        )
        log_optimized_action_fraction = distributed_mean(
            optimized_action_fraction, distributed, world_size
        )
        log_total_optimized_action_fraction = distributed_mean(
            total_optimized_action_fraction, distributed, world_size
        )
        log_terminal_idle_action_fraction = distributed_mean(
            terminal_idle_action_fraction, distributed, world_size
        )
        log_metrics = {
            "loss": float(log_loss.item()),
            "action_loss": float(log_action_loss.item()),
            "flow_loss": float(log_flow_loss.item()),
            "terminal_idle_flow_loss": float(
                log_terminal_idle_flow_loss.item()
            ),
            "terminal_idle_action_loss_weight": float(
                args.terminal_idle_action_loss_weight
            ),
            "speak_loss": float(log_speak_loss.item()),
            "speak_accuracy": float(log_speak_accuracy.item()),
            "terminal_valid_action_fraction": float(
                log_valid_action_fraction.item()
            ),
            "optimized_action_fraction": float(
                log_optimized_action_fraction.item()
            ),
            "total_optimized_action_fraction": float(
                log_total_optimized_action_fraction.item()
            ),
            "terminal_idle_action_fraction": float(
                log_terminal_idle_action_fraction.item()
            ),
            "sequence_chunks": int(current_prefix_chunks),
            "prefix_chunks": int(current_prefix_chunks),
            "chunk_batch": int(chunk_batch),
            "max_total_seq_len": int(args.max_total_seq_len),
            "total_seq_len": int(total_seq_len),
            "context_seq_len": int(input_ids.shape[1]),
            "robot_tokens": int(robot_token_count),
            "speak_tokens": int(speak_token_count),
            "action_tokens": int(action_token_count),
            "context_valid_tokens": int(context_valid_mask.sum().item()),
            "lr": float(current_lr),
        }
        if raw_wrapper.emotion_embedding is not None:
            emotion_weight = (
                raw_wrapper.emotion_embedding.weight.detach().float()
            )
            log_metrics.update(emotion_grad_metrics)
            log_metrics.update({
                f"emotion_{label}_norm": float(emotion_weight[index].norm().item())
                for label, index in {"none": NO_EMOTION_ID, **EMOTION_LABEL_TO_CONDITION_ID}.items()
            })
        if rank == 0:
            progress.set_postfix(
                loss=f"{log_loss.item():.4f}",
                flow=f"{log_flow_loss.item():.4f}",
                terminal=f"{log_terminal_idle_flow_loss.item():.4f}",
                speak=f"{log_speak_loss.item():.4f}",
                chunks=current_prefix_chunks,
                seq=total_seq_len,
            )
            write_training_metrics(
                metrics_fp=metrics_fp,
                wandb_run=wandb_run,
                step=step,
                metrics=log_metrics,
            )
        if rank == 0 and (step % args.log_interval == 0 or step == start_step + 1):
            print(
                f"step {step} loss={log_loss.item():.6f} flow={log_flow_loss.item():.6f} "
                f"terminal_idle={log_terminal_idle_flow_loss.item():.6f} "
                f"speak={log_speak_loss.item():.6f} speak_acc={log_speak_accuracy.item():.4f} "
                f"chunks={current_prefix_chunks} chunk_batch={chunk_batch} "
                f"total_seq_len={total_seq_len} context_seq_len={input_ids.shape[1]}",
                flush=True,
            )
        if rank == 0 and args.save_interval > 0 and step % args.save_interval == 0:
            save_checkpoint(out_dir / f"step-{step:06d}.pt", raw_wrapper, optimizer, step, args, norm_stats)

    if rank == 0:
        save_checkpoint(out_dir / "final.pt", raw_wrapper, optimizer, args.max_steps, args, norm_stats)
        if wandb_run is not None:
            wandb_run.finish()
        if metrics_fp is not None:
            metrics_fp.close()
        print(f"saved final checkpoint to {out_dir / 'final.pt'}", flush=True)
    if distributed:
        dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
