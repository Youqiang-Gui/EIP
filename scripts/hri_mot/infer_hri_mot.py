#!/usr/bin/env python3
"""Streaming HRI-MoT inference.

This script replays test episodes as 400 ms audio chunks:

1. append one audio chunk to the frozen AudioInteraction KV cache;
2. predict KEEP_SILENCE or TEXT_BEGIN with the base model;
3. sample one normalized action chunk with the HRI action expert from the
   current KV cache and current robot state;
4. denormalize the sampled action and save it as the downstream 35-dim pkl.

The action side matches the current training semantics:
    audio_c + state_c -> action_{c+1}
The decoded target text is also timestamped as target_{c+1}. Internally the
wrapper mask still uses decision chunk id c so action_{c+1} can attend to
state_c while current generated target text/control remains hidden from the
same decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audiointeraction.dataset.TOKENS import (  # noqa: E402
    ANGRY,
    ASSISTANT,
    AUDIO_BEGIN,
    ENGLISH,
    HAPPY,
    KEEP_SILENCE,
    NORMAL,
    ONLINE,
    PAD,
    SAD,
    SYSTEM,
    SURPRISE,
    TEXT_BEGIN,
    TEXT_END,
    URGENT,
)
from src.audiointeraction.hri_mot.data_config import (  # noqa: E402
    CONTEXT_TOKEN_PARTNER_AUDIO,
    CONTEXT_TOKEN_SYSTEM,
    CONTEXT_TOKEN_TARGET_CONTROL,
    CONTEXT_TOKEN_TARGET_TEXT,
    HriMotDataConfig,
    TEST_INDEX_PATH,
    TEXT_CHUNK_TIME_MODES,
    VALID_QPOS_DIMS,
)
from src.audiointeraction.hri_mot.episode_dataset import HriMotEpisodeDataset  # noqa: E402
from src.audiointeraction.hri_mot.motion_smoothing import (  # noqa: E402
    action_motion_stats,
    smooth_action_chunks_offline,
    stitch_action_chunk_c2,
)
from src.audiointeraction.hri_mot.video_postprocess import (  # noqa: E402
    add_video_postprocess_args,
    maybe_postprocess_prediction_video,
)
from src.audiointeraction.hri_mot.mot_wrapper import HriMoTWrapper  # noqa: E402
from src.audiointeraction.hri_mot.emotion_conditioning import (  # noqa: E402
    EMOTION_LABEL_TO_CONDITION_ID,
    EMOTION_TOKEN_ID_TO_CONDITION_ID,
    EMOTION_LABELS,
    NUM_EMOTION_CONDITIONS,
    EMOTION_MAPPING_VERSION,
    NO_EMOTION_ID,
    emotion_name,
)
from src.audiointeraction.hri_mot.streaming_episode_inference import (  # noqa: E402
    infer_episode_streaming_impl,
)
from src.audiointeraction.hri_mot.streaming_response import (  # noqa: E402
    ResponseStepResult,
    StreamingResponseController,
)
from src.audiointeraction.hri_mot.streaming_language import (  # noqa: E402
    CommittedTextFragment,
    PlannedResponse,
    commit_playback_fragment,
    plan_response_without_committing,
)
from src.audiointeraction.hri_mot.streaming_tts import (  # noqa: E402
    PlaybackFragment,
    ResponsePlaybackTimeline,
    SynthesizedUtterance,
    TTSClient,
)
from src.audiointeraction.hri_mot.streaming_tts_cli import (  # noqa: E402
    add_streaming_tts_args,
    build_tts_client_from_args,
    write_pcm16_wav,
)
from src.audiointeraction.generate.base import sample as sample_audiointeraction_logits  # noqa: E402
from src.audiointeraction.tokenizer import Tokenizer  # noqa: E402
from src.audiointeraction.utils import get_default_supported_precision  # noqa: E402
from utils import load_audio_encoder, load_model, resolve_checkpoint_paths, set_seed  # noqa: E402

from scripts.hri_mot.train_hri_mot import (  # noqa: E402
    AUDIO_TOKENS_PER_CHUNK,
    ActionStateNormalizer,
    DEFAULT_SYSTEM_PROMPT,
    _coerce_norm_stats,
    encode_audio_feature_batch,
    load_norm_stats_json,
    load_trainable_state,
)


DEFAULT_CHECKPOINT = "checkpoints/stage2/final.pt"

WEB_AUDIO_ENERGY_THRESHOLD = 0.012
WEB_AUDIO_INPUT_SAMPLE_RATE = 48_000
WEB_AUDIO_VAD_WINDOW_SAMPLES = 4096
WEB_AUDIO_TRAILING_REAL_CHUNKS = 2
WEB_AUDIO_SILENCE_SEC = 6.0

EMOTION_EMOJI = {
    HAPPY: "😊",
    SAD: "😢",
    ANGRY: "😠",
    SURPRISE: "😲",
    NORMAL: "😐",
    URGENT: "⚠️",
}

EMOTION_TOKEN_ID_TO_LABEL = {
    int(HAPPY): "happy",
    int(SAD): "sad",
    int(ANGRY): "angry",
    int(SURPRISE): "surprise",
    int(NORMAL): "normal",
    int(URGENT): "urgent",
}


def emotion_inference_config(wrapper: torch.nn.Module) -> Dict[str, Any]:
    """Return the checkpoint-resolved semantic-emotion contract for JSON logs."""

    enabled = bool(
        getattr(wrapper, "use_emotion_conditioning", False)
    )
    return {
        "emotion_conditioning": enabled,
        "emotion_film_layers": int(
            getattr(wrapper, "emotion_film_layers", 0)
        ),
        "emotion_source": (
            "generated_semantic_emotion_token_for_speaking_else_none"
            if enabled
            else "disabled"
        ),
        "emotion_mapping_version": (
            EMOTION_MAPPING_VERSION if enabled else None
        ),
        "emotion_ids": {
            "none": int(NO_EMOTION_ID),
            **EMOTION_LABEL_TO_CONDITION_ID,
        },
        "semantic_emotion_to_condition": {
            label: {
                "id": int(condition_id),
                "name": emotion_name(condition_id),
            }
            for label, condition_id in sorted(
                EMOTION_LABEL_TO_CONDITION_ID.items()
            )
        },
        "semantic_emotion_token_to_condition_id": {
            str(token_id): int(condition_id)
            for token_id, condition_id in sorted(
                EMOTION_TOKEN_ID_TO_CONDITION_ID.items()
            )
        },
        "semantic_emotion_token_action_kv_policy": (
            "legacy_commit_on_response_onset"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run streaming HRI-MoT inference and save predicted qpos pkl files.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HRI-MoT trainable checkpoint.")
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Training output directory containing args.json/norm_stats.json. Defaults to checkpoint parent.",
    )
    parser.add_argument("--checkpoint-dir", default=None, help="Base AudioInteraction checkpoint dir.")
    parser.add_argument("--test-index", default=TEST_INDEX_PATH)
    parser.add_argument("--episode-index-cache", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <train-dir>/streaming_inference.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=10)
    parser.add_argument(
        "--episode-selection",
        choices=("random", "first"),
        default="random",
        help=(
            "How to choose episodes after category filtering. random uses --seed and is reproducible; "
            "first preserves the previous index-order behavior. Exact episode/id_sample selection is unaffected."
        ),
    )
    episode_selector = parser.add_mutually_exclusive_group()
    episode_selector.add_argument(
        "--episode-id",
        default=None,
        help="Infer exactly one dataset episode, for example train_61771. Overrides category/max-episodes filtering.",
    )
    episode_selector.add_argument(
        "--id-sample",
        type=int,
        default=None,
        help="Infer exactly one dataset id_sample, for example 61771. Overrides category/max-episodes filtering.",
    )
    parser.add_argument(
        "--category",
        choices=("improvised", "naturalistic", "all"),
        default="improvised",
        help="Episode category to run. Defaults to all test episodes.",
    )
    parser.add_argument("--chunks-per-step", type=int, default=None, help="0 means all chunks.")
    parser.add_argument("--max-total-seq-len", type=int, default=None)
    parser.add_argument(
        "--speak-conditioning",
        choices=("auto", "none", "predicted", "teacher_forced"),
        default="teacher_forced",
        help=(
            "auto restores the mode recorded by the training checkpoint/args.json; "
            "an explicit mode must match the recorded checkpoint architecture."
        ),
    )
    parser.add_argument(
        "--emotion-override",
        choices=("auto", *EMOTION_LABELS),
        default="auto",
        help=(
            "Override only the Action Expert semantic emotion condition on speaking "
            "chunks. auto uses the generated six-class semantic emotion tag; "
            "silent chunks always keep the none condition."
        ),
    )
    parser.add_argument(
        "--text-chunk-time-mode",
        choices=TEXT_CHUNK_TIME_MODES,
        default="start",
        help="Dataset transcript slicing mode. Kept for compatibility with train/eval configs.",
    )
    parser.add_argument(
        "--norm-stats",
        default=None,
        help="Optional norm_stats.json. If omitted, checkpoint norm_stats or <train-dir>/norm_stats.json is used.",
    )
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--solver", choices=("euler", "heun"), default="euler")
    parser.add_argument("--initial-noise", choices=("random", "zeros"), default="random")
    parser.add_argument(
        "--state-source",
        choices=("rollout", "dataset"),
        default="rollout",
        help=(
            "rollout feeds each generated action chunk back as the next robot_state, "
            "matching autonomous streaming. dataset uses ground-truth states, useful for teacher-forced evaluation."
        ),
    )
    parser.add_argument(
        "--c2-bridge-frames",
        type=int,
        default=6,
        help="Frames replaced by a position/velocity/acceleration-continuous bridge at each action boundary.",
    )
    parser.add_argument(
        "--offline-lowpass-cutoff-hz",
        type=float,
        default=4.0,
        help="Zero-phase Butterworth cutoff for saved actions; 0 disables the offline low-pass filter.",
    )
    parser.add_argument("--max-response-tokens", type=int, default=48)
    parser.add_argument(
        "--text-temperature",
        type=float,
        default=0.0,
        help="Text/control sampling temperature; 0 enables greedy decoding by default.",
    )
    parser.add_argument(
        "--text-top-k",
        type=int,
        default=1,
        help="Text/control top-k; 1 keeps only the highest-logit token by default.",
    )
    parser.add_argument(
        "--text-top-p",
        type=float,
        default=0.0,
        help="Text/control nucleus threshold; 0 is used with temperature 0 for greedy decoding.",
    )
    parser.add_argument("--strict-missing-files", action="store_true")
    parser.add_argument("--no-audio-features", action="store_true")
    parser.add_argument(
        "--audio-feature-mode",
        choices=("streaming_latest", "independent_first"),
        default="streaming_latest",
        help=(
            "streaming_latest matches web/infer_wav.py: keep a PCM history queue, "
            "re-encode accumulated audio, and use the newest 10 audio tokens. "
            "independent_first encodes each chunk separately and uses its first 10 audio tokens."
        ),
    )
    parser.add_argument(
        "--audio-preprocess",
        choices=("none", "episode_rms", "episode_active_rms"),
        default="none",
        help=(
            "Process the complete episode waveform once before replaying 400 ms chunks. "
            "none matches web/infer_wav.py and keeps the whisper-decoded samples unchanged; "
            "the RMS modes normalize waveform amplitude."
        ),
    )
    parser.add_argument(
        "--audio-target-active-rms",
        type=float,
        default=0.10,
        help="Target robust active-frame RMS for episode_active_rms.",
    )
    parser.add_argument(
        "--audio-target-rms",
        type=float,
        default=0.06,
        help="Target complete-episode RMS for episode_rms.",
    )
    parser.add_argument(
        "--audio-active-quantile",
        type=float,
        default=0.70,
        help="Frame-RMS quantile defining active frames in episode_active_rms.",
    )
    parser.add_argument("--audio-frame-ms", type=float, default=25.0)
    parser.add_argument("--audio-hop-ms", type=float, default=10.0)
    parser.add_argument("--audio-min-gain-db", type=float, default=-12.0)
    parser.add_argument("--audio-max-gain-db", type=float, default=30.0)
    parser.add_argument("--audio-peak-limit", type=float, default=0.95)
    parser.add_argument(
        "--audio-silence-policy",
        choices=("web_vad_zero", "none"),
        default="web_vad_zero",
        help=(
            "web_vad_zero matches the web frontend's activity threshold and two real tail chunks, "
            "then replaces low-energy chunks with exact zeros while preserving the HRI action timeline. "
            "none forwards every processed chunk unchanged."
        ),
    )
    parser.add_argument(
        "--audio-energy-threshold",
        type=float,
        default=WEB_AUDIO_ENERGY_THRESHOLD,
        help="Short-window RMS activity threshold used by web_vad_zero.",
    )
    parser.add_argument(
        "--audio-trailing-real-chunks",
        type=int,
        default=WEB_AUDIO_TRAILING_REAL_CHUNKS,
        help="Number of inactive 400 ms chunks kept real before web_vad_zero starts sending exact zeros.",
    )
    parser.add_argument(
        "--audio-eof-flush-sec",
        type=float,
        default=WEB_AUDIO_SILENCE_SEC,
        help=(
            "Maximum exact-zero audio appended after the episode while waiting "
            "for a deferred response. These physical chunks generate synchronized "
            "actions and extend the prediction pkl; metrics remain limited to "
            "ground-truth episode chunks."
        ),
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--rank", type=int, default=None, help="Inference shard rank. Defaults to SLURM_PROCID/RANK.")
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="Number of inference shards. Defaults to SLURM_NTASKS/WORLD_SIZE.",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=None,
        help="Local GPU rank. Defaults to SLURM_LOCALID/LOCAL_RANK.",
    )
    parser.add_argument(
        "--metrics-wait-timeout",
        type=int,
        default=3600,
        help="Seconds rank 0 waits for per-rank metrics before writing aggregate metrics.json.",
    )
    add_streaming_tts_args(parser)
    add_video_postprocess_args(parser, allow_gt_override=False)
    return parser.parse_args()


def load_train_args(train_dir: Path) -> Dict[str, Any]:
    args_path = train_dir / "args.json"
    if not args_path.is_file():
        return {}
    with args_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_checkpoint_train_args(
    checkpoint: Path,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Read training args and infer the supported action architecture."""

    try:
        checkpoint_obj = torch.load(
            checkpoint,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except (TypeError, RuntimeError):
        checkpoint_obj = torch.load(checkpoint, map_location="cpu")
    if not isinstance(checkpoint_obj, dict):
        return {}, None

    raw_args = checkpoint_obj.get("args", {})
    checkpoint_args = dict(raw_args) if isinstance(raw_args, dict) else {}
    recorded_mode = checkpoint_args.get("speak_conditioning")
    params = checkpoint_obj.get("trainable_params", checkpoint_obj)
    if not isinstance(params, dict):
        return checkpoint_args, (
            str(recorded_mode) if recorded_mode is not None else None
        )

    param_names = tuple(str(name) for name in params)
    if any("motion_intensity_" in name for name in param_names):
        raise RuntimeError(
            "Checkpoint conditioning is incompatible with the six-emotion model. "
            "Use a Stage-I or six-emotion Stage-II checkpoint."
        )
    for name, tensor in params.items():
        if str(name).endswith("emotion_embedding.weight"):
            if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2 or tensor.shape[0] != NUM_EMOTION_CONDITIONS:
                raise RuntimeError("Emotion embedding must contain none plus six semantic-emotion rows")
    unsupported_names = [
        name
        for name in param_names
        if name.endswith("emotion_token_embedding.weight")
        or name.endswith("emotion_action_film.weight")
        or "emotion_low_residual_adapters." in name
    ]
    if unsupported_names:
        raise RuntimeError(
            "Checkpoint contains unsupported emotion-conditioning parameters: "
            f"{unsupported_names[:4]}"
        )

    has_emotion_embedding = any(
        name.endswith("emotion_embedding.weight")
        for name in param_names
    )
    film_pattern = re.compile(
        r"(?:^|\.)emotion_layer_film\.(\d+)\.weight$"
    )
    film_layers: Dict[int, torch.Tensor] = {}
    for raw_name, tensor in params.items():
        match = film_pattern.search(str(raw_name))
        if match is None:
            continue
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 2:
            raise RuntimeError(
                "Emotion layer FiLM weights must be 2D tensors: "
                f"{raw_name}"
            )
        if int(tensor.shape[0]) != NUM_EMOTION_CONDITIONS:
            raise RuntimeError(
                "Emotion layer FiLM must contain none plus six semantic-emotion rows: "
                f"{raw_name} has shape {tuple(tensor.shape)}"
            )
        film_layers[int(match.group(1))] = tensor

    has_layer_film = bool(film_layers)
    if has_emotion_embedding != has_layer_film:
        raise RuntimeError(
            "Emotion checkpoints must contain both the condition "
            "embedding and per-layer FiLM tables"
        )
    inferred_film_layers = len(film_layers)
    if has_layer_film:
        ordered_layers = sorted(film_layers)
        if ordered_layers != list(
            range(ordered_layers[0], ordered_layers[-1] + 1)
        ):
            raise RuntimeError(
                "Emotion FiLM block indices must be contiguous, got "
                f"{ordered_layers}"
            )

    recorded_emotion = checkpoint_args.get(
        "use_emotion_conditioning"
    )
    if (
        recorded_emotion is not None
        and bool(recorded_emotion)
        != has_emotion_embedding
    ):
        raise RuntimeError(
            "Checkpoint args/state disagree about emotion "
            "conditioning: "
            f"args={bool(recorded_emotion)}, "
            f"embedding_present={has_emotion_embedding}"
        )
    recorded_film_layers = checkpoint_args.get(
        "emotion_film_layers"
    )
    if (
        has_layer_film
        and recorded_film_layers is not None
        and int(recorded_film_layers) != inferred_film_layers
    ):
        raise RuntimeError(
            "Checkpoint args/state disagree about emotion FiLM "
            f"depth: args={recorded_film_layers}, state={inferred_film_layers}"
        )

    checkpoint_args["use_emotion_conditioning"] = bool(
        has_emotion_embedding
    )
    checkpoint_args["emotion_film_layers"] = int(
        inferred_film_layers
    )

    if recorded_mode is None:
        if any("speak_prediction_head" in name for name in param_names):
            recorded_mode = "predicted"
        else:
            speak_embedding = next(
                (
                    tensor
                    for name, tensor in params.items()
                    if str(name).endswith("speak_token_embedding.weight")
                ),
                None,
            )
            if (
                isinstance(speak_embedding, torch.Tensor)
                and speak_embedding.dim() == 2
            ):
                if int(speak_embedding.shape[0]) == 1:
                    recorded_mode = "predicted"
                elif int(speak_embedding.shape[0]) == 2:
                    recorded_mode = "teacher_forced"
            if recorded_mode is None:
                recorded_mode = "none"
    return checkpoint_args, str(recorded_mode)


def resolve_inference_train_args(
    *,
    checkpoint: Path,
    file_train_args: Dict[str, Any],
    requested_speak_conditioning: str,
) -> Tuple[Dict[str, Any], str]:
    checkpoint_args, checkpoint_mode = load_checkpoint_train_args(checkpoint)
    merged = dict(file_train_args)
    merged.update(checkpoint_args)
    file_mode = file_train_args.get("speak_conditioning")
    recorded_mode = checkpoint_mode or (str(file_mode) if file_mode is not None else None)
    requested = str(requested_speak_conditioning)
    valid = ("auto", "none", "predicted", "teacher_forced")
    if requested not in valid:
        raise ValueError(f"Invalid --speak-conditioning={requested!r}")
    if requested == "auto":
        resolved = recorded_mode or "none"
        source = "checkpoint" if checkpoint_mode is not None else "args.json" if file_mode is not None else "default"
    else:
        if recorded_mode is not None and requested != recorded_mode:
            raise ValueError(
                f"--speak-conditioning={requested!r} does not match checkpoint mode "
                f"{recorded_mode!r}; use the matching checkpoint or --speak-conditioning auto"
            )
        resolved = requested
        source = "explicit_cli"
    if resolved not in valid[1:]:
        raise ValueError(f"Invalid resolved speak_conditioning={resolved!r}")
    merged["speak_conditioning"] = resolved
    use_emotion = bool(
        merged.get("use_emotion_conditioning", False)
    )
    if use_emotion:
        recorded_mapping = merged.get("emotion_mapping_version")
        if recorded_mapping != EMOTION_MAPPING_VERSION:
            raise ValueError(
                "Emotion checkpoint mapping is missing or incompatible: "
                f"checkpoint={recorded_mapping!r}, "
                f"inference={EMOTION_MAPPING_VERSION!r}"
            )
        if resolved != "teacher_forced":
            raise ValueError(
                "Emotion checkpoints require teacher_forced SPEAK "
                f"conditioning, got {resolved!r}"
            )
    return merged, source



def resolve_rank_info(args: argparse.Namespace) -> Tuple[int, int, int]:
    world_size = args.world_size
    if world_size is None:
        world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    rank = args.rank
    if rank is None:
        rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    local_rank = args.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", str(rank))))
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    return int(rank), int(world_size), int(local_rank)


def resolve_device(device_arg: str, local_rank: int = 0) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            return torch.device(f"cuda:{local_rank % max(device_count, 1)}")
        return torch.device("cpu")
    return torch.device(device_arg)


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


def build_wrapper(
    *,
    base_model: torch.nn.Module,
    data_config: HriMotDataConfig,
    train_args: Dict[str, Any],
    device: torch.device,
) -> HriMoTWrapper:
    speak_conditioning = str(train_args.get("speak_conditioning", "none"))
    if speak_conditioning not in ("none", "predicted", "teacher_forced"):
        raise ValueError(
            f"Invalid speak_conditioning={speak_conditioning!r} in training args"
        )
    wrapper = HriMoTWrapper(
        base_model=base_model,
        decoder_path="_forward_module.transformer.h",
        robot_state_dim=data_config.robot_state_dim,
        action_dim=data_config.action_dim,
        action_hidden_size=int(train_args.get("action_hidden_size", 768)),
        action_intermediate_size=int(train_args.get("action_intermediate_size", 3072)),
        robot_state_steps=data_config.robot_steps_per_chunk,
        speak_conditioning=speak_conditioning,
        use_response_time_embedding=bool(
            train_args.get("response_time_embedding", False)
        ),
        use_emotion_conditioning=bool(
            train_args.get("use_emotion_conditioning", False)
        ),
        emotion_film_layers=int(
            train_args.get("emotion_film_layers", 4)
        ),
    ).to(device)
    wrapper.eval()
    wrapper.base_model.eval()
    return wrapper


def _sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: Optional[int],
    top_p: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if logits.dim() != 1:
        raise ValueError(f"logits must be 1D, got shape {tuple(logits.shape)}")
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1)
    logits = logits.float() / temperature
    if top_k is not None and top_k > 0:
        values, indices = torch.topk(logits, min(int(top_k), logits.numel()))
        filtered = torch.full_like(logits, float("-inf"))
        logits = filtered.scatter(0, indices, values)
    if top_p < 1.0:
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be in [0, 1], got {top_p}")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_indices, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(0)


def _system_prefix_ids(tokenizer: Tokenizer) -> List[int]:
    system_ids = tokenizer.encode(DEFAULT_SYSTEM_PROMPT, bos=False, eos=False).tolist()
    return [ONLINE, ENGLISH, SYSTEM, TEXT_BEGIN] + system_ids + [TEXT_END]


def _init_input_pos_maxp1(model: torch.nn.Module, prompt_size: int, device: torch.device) -> Optional[torch.Tensor]:
    if any(module.__class__.__name__ == "ThunderModule" for module in model.modules()):
        return None
    return torch.tensor(prompt_size, device=device)


class HriMotStreamingSession:
    def __init__(
        self,
        *,
        wrapper: HriMoTWrapper,
        tokenizer: Tokenizer,
        normalizer: ActionStateNormalizer,
        data_config: HriMotDataConfig,
        device: torch.device,
        max_total_seq_len: int,
        text_temperature: float,
        text_top_k: int,
        text_top_p: float,
        language_generator: Optional[torch.Generator],
        language_sampling_mode: str = "independent",
        debug_language_sampling: bool = False,
    ) -> None:
        self.wrapper = wrapper
        self.base_model = wrapper.base_model
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.data_config = data_config
        self.device = device
        self.max_total_seq_len = int(max_total_seq_len)
        self.text_temperature = float(text_temperature)
        self.text_top_k = int(text_top_k)
        self.text_top_p = float(text_top_p)
        self.language_generator = language_generator
        self.language_sampling_mode = str(language_sampling_mode)
        if self.language_sampling_mode not in ("independent", "web_legacy"):
            raise ValueError(
                "language_sampling_mode must be 'independent' or 'web_legacy', "
                f"got {self.language_sampling_mode!r}"
            )
        if self.language_sampling_mode == "independent" and language_generator is None:
            raise ValueError("independent language sampling requires a generator")
        self.debug_language_sampling = bool(debug_language_sampling)
        self.language_sample_index = 0

        self.pending_ids: torch.Tensor
        self.pending_pos: torch.Tensor
        self.pending_chunk_ids: List[int] = []
        self.pending_token_types: List[int] = []
        self.context_chunk_ids: List[int] = []
        self.context_token_types: List[int] = []
        self.pcm_history: List[np.ndarray] = []
        self.last_sample_debug: Dict[str, Any] = {}
        self.kv_context_length = 0
        self.input_pos_maxp1: Optional[torch.Tensor] = None
        self.reset()

    def reset(self) -> None:
        try:
            self.wrapper.clear_context_kv_cache()
        except Exception:
            pass
        cache_len = min(int(getattr(self.base_model, "max_seq_length", self.max_total_seq_len)), self.max_total_seq_len)
        dtype = next(self.base_model.parameters()).dtype
        self.wrapper.set_context_kv_cache(
            batch_size=1,
            max_seq_length=cache_len,
            device=self.device,
            dtype=dtype,
        )
        self.base_model.eval()
        self.wrapper.eval()

        prefix = _system_prefix_ids(self.tokenizer)
        self.pending_ids = torch.tensor(prefix, dtype=torch.long, device=self.device)
        self.pending_pos = torch.arange(len(prefix), dtype=torch.int64, device=self.device)
        self.pending_chunk_ids = [-1] * len(prefix)
        self.pending_token_types = [CONTEXT_TOKEN_SYSTEM] * len(prefix)
        self.context_chunk_ids = []
        self.context_token_types = []
        self.pcm_history = []
        self.last_sample_debug = {}
        self.kv_context_length = 0
        self.input_pos_maxp1 = _init_input_pos_maxp1(self.base_model, len(prefix), self.device)

    @torch.no_grad()
    def encode_streaming_audio_feature(
        self,
        *,
        audio_encoder: torch.nn.Module,
        pcm_chunk: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Match web/infer_wav.py: cache PCM history and return newest audio tokens."""
        import whisper

        pcm = pcm_chunk.detach().float().cpu().numpy().astype(np.float32).reshape(-1)
        self.pcm_history.append(pcm)
        full_audio = np.concatenate(self.pcm_history)

        mel = whisper.log_mel_spectrogram(full_audio, n_mels=128)
        mel_tensor = mel if torch.is_tensor(mel) else torch.from_numpy(mel)
        mel_tensor = mel_tensor.to(self.device)
        mel_frames = int(mel_tensor.shape[-1])
        chunks = mel_frames // 40
        feature_lens = torch.tensor([40] * chunks, device=self.device)
        aftercnn_lens = torch.tensor((mel_frames - 1) // 2 + 1, device=self.device)

        features = audio_encoder(mel_tensor, feature_lens, aftercnn_lens).last_hidden_state
        return features[-AUDIO_TOKENS_PER_CHUNK:].to(device=self.device, dtype=dtype)

    def _scheduled_length(self) -> int:
        if self.pending_pos.numel() == 0:
            return self.kv_context_length
        return int(self.pending_pos[-1].item()) + 1

    def _ensure_capacity(self, extra_tokens: int) -> None:
        scheduled = self._scheduled_length()
        required = scheduled + int(extra_tokens)
        model_limit = int(getattr(self.base_model, "max_seq_length", self.max_total_seq_len))
        limit = min(model_limit, self.max_total_seq_len)
        if required > limit:
            raise RuntimeError(
                f"Streaming context would reach {required} tokens, exceeding limit {limit}. "
                f"Decrease --max-response-tokens/--chunks-per-step or increase --max-total-seq-len."
            )

    def _schedule_pending_token(self, token_id: int, *, chunk_idx: int, token_type: int) -> None:
        next_pos = int(self.pending_pos[-1].item()) + 1 if self.pending_pos.numel() else self.kv_context_length
        self.pending_ids = torch.tensor([int(token_id)], dtype=torch.long, device=self.device)
        self.pending_pos = torch.tensor([next_pos], dtype=torch.int64, device=self.device)
        self.pending_chunk_ids = [int(chunk_idx)]
        self.pending_token_types = [int(token_type)]
        if self.input_pos_maxp1 is not None:
            self.input_pos_maxp1.add_(1)

    def _append_listening_block(self, *, chunk_idx: int) -> None:
        self._ensure_capacity(2 + AUDIO_TOKENS_PER_CHUNK)
        block_ids = [AUDIO_BEGIN] + [PAD] * AUDIO_TOKENS_PER_CHUNK + [ASSISTANT]
        new_tokens = torch.tensor(block_ids, dtype=torch.long, device=self.device)
        last_pos = int(self.pending_pos[-1].item()) if self.pending_pos.numel() else self.kv_context_length - 1
        new_pos = last_pos + torch.arange(1, len(block_ids) + 1, dtype=torch.int64, device=self.device)
        self.pending_ids = torch.cat([self.pending_ids, new_tokens], dim=0)
        self.pending_pos = torch.cat([self.pending_pos, new_pos], dim=0)
        self.pending_chunk_ids.extend([int(chunk_idx)] * len(block_ids))
        self.pending_token_types.extend([CONTEXT_TOKEN_PARTNER_AUDIO] * len(block_ids))
        if self.input_pos_maxp1 is not None:
            self.input_pos_maxp1.add_(len(block_ids))

    def _forward_pending_and_sample(
        self,
        *,
        audio_feature: Optional[torch.Tensor],
        next_chunk_idx: int,
        next_token_type: int,
        allowed_token_ids: Optional[Sequence[int]] = None,
        restrict_to_allowed_token_ids: bool = True,
        forced_next_token: Optional[int] = None,
    ) -> int:
        if self.pending_ids.numel() == 0:
            raise RuntimeError("No pending token to forward into the KV cache")
        input_ends_text_begin = bool(
            self.debug_language_sampling
            and self.pending_ids.numel() == 1
            and int(self.pending_ids.item()) == TEXT_BEGIN
            and int(next_token_type) == CONTEXT_TOKEN_TARGET_TEXT
        )
        input_pos_maxp1 = self.input_pos_maxp1
        logits = self.base_model(
            self.pending_ids.view(1, -1),
            None,
            1,
            audio_feature.to(self.device) if audio_feature is not None else None,
            self.pending_pos,
            input_pos_maxp1=input_pos_maxp1,
            audio_tokens_per_chunk=AUDIO_TOKENS_PER_CHUNK,
        )
        self.context_chunk_ids.extend(self.pending_chunk_ids)
        self.context_token_types.extend(self.pending_token_types)
        self.kv_context_length = len(self.context_chunk_ids)

        if forced_next_token is not None:
            token_id = int(forced_next_token)
            self.last_sample_debug = {}
        else:
            # web_legacy intentionally retains the base model's native logits
            # dtype and process-global RNG. independent retains the original
            # HRI behavior: float32 probabilities and a private RNG stream.
            last_logits = logits[0, -1]
            if self.language_sampling_mode == "independent":
                last_logits = last_logits.float()
            self.language_sample_index += 1
            debug: Dict[str, Any] = {}
            if allowed_token_ids is not None:
                allowed_ids = [int(token_id_item) for token_id_item in allowed_token_ids]
                allowed_logits = torch.stack([last_logits[token_id_item] for token_id_item in allowed_ids])
                allowed_probs = torch.softmax(allowed_logits, dim=0)
                debug = {
                    "allowed_token_ids": allowed_ids,
                    "allowed_logits": {
                        str(token_id_item): float(logit_item.item())
                        for token_id_item, logit_item in zip(allowed_ids, allowed_logits)
                    },
                    "allowed_probs": {
                        str(token_id_item): float(prob_item.item())
                        for token_id_item, prob_item in zip(allowed_ids, allowed_probs)
                    },
                }
                if restrict_to_allowed_token_ids:
                    restricted = torch.full_like(last_logits, float("-inf"))
                    for token_id_item in allowed_ids:
                        restricted[token_id_item] = last_logits[token_id_item]
                    last_logits = restricted
            rng_before = None
            emotion_debug = None
            if input_ends_text_begin:
                rng_before = hashlib.sha256(
                    self._language_rng_state().cpu().numpy().tobytes()
                ).hexdigest()[:16]
                full_probs = torch.softmax(last_logits, dim=-1)
                emotion_debug = {
                    str(token_id_item): {
                        "emoji": emoji,
                        "logit": float(last_logits[token_id_item].item()),
                        "prob": float(full_probs[token_id_item].item()),
                    }
                    for token_id_item, emoji in EMOTION_EMOJI.items()
                }
            if self.language_sampling_mode == "web_legacy":
                sampled = sample_audiointeraction_logits(
                    last_logits.view(1, 1, -1),
                    temperature=self.text_temperature,
                    top_k=(self.text_top_k if self.text_top_k > 0 else None),
                    top_p=self.text_top_p,
                ).squeeze(0)
            else:
                sampled = _sample_from_logits(
                    last_logits,
                    temperature=self.text_temperature,
                    top_k=self.text_top_k,
                    top_p=self.text_top_p,
                    generator=self.language_generator,
                )
            token_id = int(sampled.item())
            if input_ends_text_begin:
                rng_after = hashlib.sha256(
                    self._language_rng_state().cpu().numpy().tobytes()
                ).hexdigest()[:16]
                print(
                    "[language-sampling-debug] "
                    + json.dumps(
                        {
                            "sample_index": self.language_sample_index,
                            "kind": "emotion_after_text_begin",
                            "rng_before": rng_before,
                            "rng_after": rng_after,
                            "selected_token_id": token_id,
                            "emotion_tokens": emotion_debug,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            self.last_sample_debug = debug
        self._schedule_pending_token(token_id, chunk_idx=next_chunk_idx, token_type=next_token_type)
        return token_id

    def _language_rng_state(self) -> torch.Tensor:
        if self.language_sampling_mode == "independent":
            assert self.language_generator is not None
            return self.language_generator.get_state()
        if self.device.type == "cuda":
            return torch.cuda.get_rng_state(self.device)
        return torch.get_rng_state()

    @torch.no_grad()
    def step_audio_chunk(
        self,
        *,
        chunk_idx: int,
        audio_feature: Optional[torch.Tensor],
        allow_text_begin: bool = True,
    ) -> int:
        self._append_listening_block(chunk_idx=chunk_idx)
        # The released Audio-Interaction model already predicts the listening
        # controls itself. Do not change its distribution in Web-compat mode.
        restrict_controls = bool(
            not allow_text_begin or self.language_sampling_mode != "web_legacy"
        )
        return self._forward_pending_and_sample(
            audio_feature=audio_feature,
            next_chunk_idx=chunk_idx,
            next_token_type=CONTEXT_TOKEN_TARGET_CONTROL,
            allowed_token_ids=(KEEP_SILENCE, TEXT_BEGIN) if allow_text_begin else (KEEP_SILENCE,),
            restrict_to_allowed_token_ids=restrict_controls,
        )

    @torch.no_grad()
    def generate_response_text(self, *, chunk_idx: int, max_tokens: int) -> Tuple[List[int], str, bool]:
        produced: List[int] = []
        completed = False
        for _ in range(max(0, int(max_tokens))):
            token_id = self._forward_pending_and_sample(
                audio_feature=None,
                next_chunk_idx=chunk_idx,
                next_token_type=CONTEXT_TOKEN_TARGET_TEXT,
            )
            produced.append(token_id)
            if token_id == TEXT_END:
                completed = True
                break
        if not completed:
            # Forward the last generated token (or TEXT_BEGIN when max_tokens == 0)
            # and make TEXT_END the next pending token, so the following audio
            # chunk sees a structurally closed assistant turn.
            token_id = self._forward_pending_and_sample(
                audio_feature=None,
                next_chunk_idx=chunk_idx,
                next_token_type=CONTEXT_TOKEN_TARGET_TEXT,
                forced_next_token=TEXT_END,
            )
            produced.append(token_id)
            completed = True
        emotion = ""
        text_ids: List[int] = []
        for token_id in produced:
            if token_id in (TEXT_BEGIN, TEXT_END, KEEP_SILENCE):
                continue
            if not emotion and not text_ids and token_id in EMOTION_EMOJI:
                emotion = EMOTION_EMOJI[token_id]
                continue
            text_ids.append(token_id)
        text = self.tokenizer.decode(torch.tensor(text_ids)) if text_ids else ""
        text = f"{emotion} {text}".strip()
        return produced, text, completed

    def action_context_tensors(self) -> Dict[str, torch.Tensor]:
        if self.kv_context_length <= 0:
            raise RuntimeError("KV cache is empty; run at least one context forward before action sampling")
        return {
            "context_length": int(self.kv_context_length),
            "context_valid_mask": torch.ones((1, self.kv_context_length), dtype=torch.bool, device=self.device),
            "context_chunk_ids": torch.tensor([self.context_chunk_ids], dtype=torch.long, device=self.device),
            "context_token_types": torch.tensor([self.context_token_types], dtype=torch.long, device=self.device),
        }


@torch.no_grad()
def sample_action_chunk(
    *,
    session: HriMotStreamingSession,
    robot_state: torch.Tensor,
    chunk_idx: int,
    action_horizon: int,
    action_dim: int,
    sampling_steps: int,
    solver: str,
    initial_noise: str,
    device: torch.device,
    action_generator: torch.Generator,
    teacher_forced_speak_label: Optional[int] = None,
    response_elapsed_chunk: Optional[int] = None,
    emotion_token_id: Optional[int] = None,
) -> torch.Tensor:
    if sampling_steps <= 0:
        raise ValueError(f"sampling_steps must be positive, got {sampling_steps}")
    if initial_noise == "zeros":
        action = torch.zeros((1, action_horizon, action_dim), device=device, dtype=torch.float32)
    else:
        action = torch.randn(
            (1, action_horizon, action_dim),
            device=device,
            dtype=torch.float32,
            generator=action_generator,
        )

    robot_state = session.normalizer.normalize_state(robot_state.to(device=device).unsqueeze(0))
    state_chunk_ids = torch.full((1, robot_state.shape[1]), int(chunk_idx), dtype=torch.long, device=device)
    action_chunk_ids = torch.full((1, action_horizon), int(chunk_idx), dtype=torch.long, device=device)
    speak_mode = str(session.wrapper.speak_conditioning)
    speak_chunk_ids = None
    speak_labels = None
    response_elapsed_chunks = None
    emotion_labels = None
    emotion_chunk_ids = None
    emotion_id = None
    emotion_source = "disabled"
    semantic_emotion_label = None
    emotion_override = str(
        getattr(session, "emotion_override", "auto")
    )
    if emotion_override not in ("auto", *EMOTION_LABELS):
        raise ValueError(
            "emotion_override must be auto or one of the six emotion labels, got "
            f"{emotion_override!r}"
        )
    if speak_mode != "none":
        speak_chunk_ids = torch.full((1, 1), int(chunk_idx), dtype=torch.long, device=device)
    if speak_mode == "teacher_forced":
        if teacher_forced_speak_label not in (0, 1):
            raise ValueError(
                "teacher_forced inference requires a planned c+1 speak label"
            )
        speak_labels = torch.tensor(
            [[int(teacher_forced_speak_label)]], dtype=torch.long, device=device
        )
    if bool(
        getattr(session.wrapper, "use_emotion_conditioning", False)
    ):
        if int(teacher_forced_speak_label) == 1:
            if emotion_token_id not in EMOTION_TOKEN_ID_TO_CONDITION_ID:
                raise RuntimeError(
                    "Speaking action requires a generated semantic emotion token "
                    f"for emotion, got {emotion_token_id!r}"
                )
            semantic_emotion_label = EMOTION_TOKEN_ID_TO_LABEL[
                int(emotion_token_id)
            ]
            if emotion_override == "auto":
                emotion_id = EMOTION_TOKEN_ID_TO_CONDITION_ID[
                    int(emotion_token_id)
                ]
                emotion_source = "generated_semantic_emotion_token"
            else:
                emotion_id = EMOTION_LABEL_TO_CONDITION_ID[emotion_override]
                emotion_source = f"inference_override_{emotion_override}"
        else:
            if emotion_token_id is not None:
                raise RuntimeError(
                    "Silent action interval unexpectedly has an emotion token"
                )
            emotion_id = NO_EMOTION_ID
            emotion_source = "silent_interval"
        emotion_labels = torch.tensor(
            [[int(emotion_id)]], dtype=torch.long, device=device
        )
        emotion_chunk_ids = speak_chunk_ids
    elif emotion_override != "auto":
        raise ValueError(
            "--emotion-override requires a checkpoint with "
            "emotion conditioning"
        )
    if bool(getattr(session.wrapper, "use_response_time_embedding", False)):
        if response_elapsed_chunk is None:
            raise ValueError(
                "response time embedding requires response_elapsed_chunk"
            )
        elapsed = int(response_elapsed_chunk)
        expected_active = int(teacher_forced_speak_label) == 1
        if (elapsed >= 0) != expected_active:
            raise ValueError(
                "response_elapsed_chunk must be >= 0 exactly when the "
                "teacher-forced speak label is 1"
            )
        response_elapsed_chunks = torch.tensor(
            [[elapsed]], dtype=torch.long, device=device
        )
    session.last_action_speak_debug = {
        "mode": speak_mode,
        "teacher_forced_label": (
            int(teacher_forced_speak_label)
            if speak_mode == "teacher_forced"
            else None
        ),
        "response_elapsed_chunk": (
            int(response_elapsed_chunk)
            if response_elapsed_chunks is not None
            else None
        ),
        "semantic_emotion_token_id": (
            int(emotion_token_id) if emotion_token_id is not None else None
        ),
        "semantic_emotion_label": semantic_emotion_label,
        "emotion_conditioning": bool(
            getattr(
                session.wrapper,
                "use_emotion_conditioning",
                False,
            )
        ),
        "emotion_id": (
            int(emotion_id)
            if emotion_id is not None
            else None
        ),
        "emotion_name": (
            emotion_name(emotion_id)
            if emotion_id is not None
            else None
        ),
        "emotion_source": emotion_source,
        "emotion_override": emotion_override,
    }
    context = session.action_context_tensors()
    dt = 1.0 / float(sampling_steps)

    for step_idx in range(sampling_steps):
        tau_value = step_idx / float(sampling_steps)
        tau = torch.full((1,), tau_value, device=device, dtype=torch.float32)
        outputs = session.wrapper.forward_from_context_kv_cache(
            robot_state=robot_state,
            noisy_action=action,
            tau=tau,
            state_chunk_ids=state_chunk_ids,
            action_chunk_ids=action_chunk_ids,
            speak_labels=speak_labels,
            speak_chunk_ids=speak_chunk_ids,
            emotion_labels=emotion_labels,
            emotion_chunk_ids=emotion_chunk_ids,
            response_elapsed_chunks=response_elapsed_chunks,
            **context,
        )
        if step_idx == 0 and "speak_logits" in outputs:
            speak_logit = float(outputs["speak_logits"].reshape(-1)[0].float().item())
            session.last_action_speak_debug.update(
                {
                    "predicted_logit": speak_logit,
                    "predicted_probability": float(torch.sigmoid(torch.tensor(speak_logit)).item()),
                    "predicted_label": int(speak_logit >= 0.0),
                }
            )
        velocity = outputs["action_velocity"].float()
        if solver == "heun":
            action_euler = action + velocity * dt
            tau_next = torch.full((1,), min(1.0, tau_value + dt), device=device, dtype=torch.float32)
            outputs_next = session.wrapper.forward_from_context_kv_cache(
                robot_state=robot_state,
                noisy_action=action_euler,
                tau=tau_next,
                state_chunk_ids=state_chunk_ids,
                action_chunk_ids=action_chunk_ids,
                speak_labels=speak_labels,
                speak_chunk_ids=speak_chunk_ids,
                emotion_labels=emotion_labels,
                emotion_chunk_ids=emotion_chunk_ids,
                response_elapsed_chunks=response_elapsed_chunks,
                **context,
            )
            velocity = 0.5 * (velocity + outputs_next["action_velocity"].float())
        action = action + velocity * dt

    return session.normalizer.unnormalize_action(action).squeeze(0).float()


def action_boundary_stats(pred_action: torch.Tensor) -> Dict[str, float]:
    if pred_action.dim() != 3 or pred_action.shape[0] <= 1:
        return {"boundary_jump_mean": 0.0, "boundary_jump_max": 0.0}
    jumps = (pred_action[1:, 0, :] - pred_action[:-1, -1, :]).float().norm(dim=-1)
    return {
        "boundary_jump_mean": float(jumps.mean().item()),
        "boundary_jump_max": float(jumps.max().item()),
    }


def preprocess_episode_audio(
    audio_chunks: torch.Tensor,
    *,
    method: str,
    sample_rate: int,
    target_active_rms: float,
    target_rms: float,
    active_quantile: float,
    frame_ms: float,
    hop_ms: float,
    min_gain_db: float,
    max_gain_db: float,
    peak_limit: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Apply one peak-safe gain to a complete episode, then restore its chunk layout."""
    if audio_chunks.dim() != 2:
        raise ValueError(f"partner_audio must have shape [chunks, samples], got {tuple(audio_chunks.shape)}")
    if audio_chunks.numel() == 0:
        raise ValueError("Cannot preprocess an empty episode waveform")
    if not (0.0 < float(active_quantile) < 1.0):
        raise ValueError(f"audio_active_quantile must be in (0, 1), got {active_quantile}")
    if float(target_active_rms) <= 0.0 or float(target_rms) <= 0.0:
        raise ValueError("Audio RMS targets must be positive")
    if float(min_gain_db) > float(max_gain_db):
        raise ValueError("audio_min_gain_db must be <= audio_max_gain_db")
    if not (0.0 < float(peak_limit) <= 1.0):
        raise ValueError(f"audio_peak_limit must be in (0, 1], got {peak_limit}")

    original_shape = audio_chunks.shape
    flat = audio_chunks.detach().float().reshape(-1)
    raw_mean = flat.mean()
    raw_rms = flat.square().mean().sqrt()
    raw_peak = flat.abs().max()
    stats: Dict[str, Any] = {
        "method": str(method),
        "samples": int(flat.numel()),
        "duration_sec": float(flat.numel() / int(sample_rate)),
        "raw_mean": float(raw_mean.item()),
        "raw_rms": float(raw_rms.item()),
        "raw_peak": float(raw_peak.item()),
    }
    if method == "none":
        stats.update(
            {
                "reference_rms": float(raw_rms.item()),
                "requested_gain": 1.0,
                "requested_gain_db": 0.0,
                "applied_gain": 1.0,
                "applied_gain_db": 0.0,
                "peak_protection_applied": False,
                "processed_rms": float(raw_rms.item()),
                "processed_peak": float(raw_peak.item()),
            }
        )
        return audio_chunks.clone(), stats

    centered = flat - raw_mean
    eps = torch.finfo(centered.dtype).eps
    centered_rms = centered.square().mean().sqrt()
    centered_peak = centered.abs().max()
    stats["dc_removed"] = True
    stats["centered_rms"] = float(centered_rms.item())
    stats["centered_peak"] = float(centered_peak.item())

    if method == "episode_rms":
        reference_rms = centered_rms
        target = float(target_rms)
        stats["target_rms"] = target
    elif method == "episode_active_rms":
        frame_length = max(1, int(round(float(frame_ms) * int(sample_rate) / 1000.0)))
        hop_length = max(1, int(round(float(hop_ms) * int(sample_rate) / 1000.0)))
        if centered.numel() < frame_length:
            padded = F.pad(centered, (0, frame_length - centered.numel()))
            frames = padded.unsqueeze(0)
        else:
            frames = centered.unfold(0, frame_length, hop_length)
        frame_rms = frames.square().mean(dim=-1).sqrt()
        # Select an exact top-energy fraction instead of thresholding with >=.
        # With long digital silence, many frame RMS values are tied at zero and
        # thresholding would otherwise include the entire episode.
        active_count = max(1, int(np.ceil((1.0 - float(active_quantile)) * frame_rms.numel())))
        active_frame_rms = torch.topk(frame_rms, k=active_count, largest=True, sorted=False).values
        threshold = active_frame_rms.min()
        reference_rms = active_frame_rms.square().mean().sqrt()
        target = float(target_active_rms)
        stats.update(
            {
                "target_active_rms": target,
                "frame_ms": float(frame_ms),
                "hop_ms": float(hop_ms),
                "active_quantile": float(active_quantile),
                "active_threshold_rms": float(threshold.item()),
                "active_frames": int(active_frame_rms.numel()),
                "total_frames": int(frame_rms.numel()),
            }
        )
    else:
        raise ValueError(f"Unknown audio preprocess method: {method}")

    reference_value = float(reference_rms.item())
    stats["reference_rms"] = reference_value
    if reference_value <= float(eps) or float(centered_peak.item()) <= float(eps):
        requested_gain = 1.0
    else:
        requested_gain = target / reference_value
    min_gain = 10.0 ** (float(min_gain_db) / 20.0)
    max_gain = 10.0 ** (float(max_gain_db) / 20.0)
    bounded_gain = min(max(requested_gain, min_gain), max_gain)
    peak_safe_gain = float(peak_limit) / max(float(centered_peak.item()), float(eps))
    applied_gain = min(bounded_gain, peak_safe_gain)
    processed = centered * float(applied_gain)

    stats.update(
        {
            "requested_gain": float(requested_gain),
            "requested_gain_db": float(20.0 * np.log10(max(requested_gain, float(eps)))),
            "bounded_gain": float(bounded_gain),
            "peak_safe_gain": float(peak_safe_gain),
            "applied_gain": float(applied_gain),
            "applied_gain_db": float(20.0 * np.log10(max(applied_gain, float(eps)))),
            "peak_protection_applied": bool(applied_gain < bounded_gain),
            "processed_rms": float(processed.square().mean().sqrt().item()),
            "processed_peak": float(processed.abs().max().item()),
        }
    )
    return processed.reshape(original_shape).to(dtype=audio_chunks.dtype), stats


def apply_streaming_audio_silence_policy(
    audio_chunks: torch.Tensor,
    *,
    policy: str,
    sample_rate: int,
    energy_threshold: float,
    trailing_real_chunks: int,
) -> Tuple[torch.Tensor, List[str], Dict[str, Any]]:
    """Apply the web activity/tail/zero policy without shortening the HRI timeline.

    The browser drops inactive frames before its first activity trigger and after
    its timeout. HRI still needs one action for every 400 ms dataset chunk, so
    those dropped positions are represented by exact-zero audio instead. Text
    triggering is separately disarmed until a real activity chunk is observed.
    """
    if audio_chunks.dim() != 2:
        raise ValueError(f"audio_chunks must have shape [chunks, samples], got {tuple(audio_chunks.shape)}")
    if float(energy_threshold) < 0.0:
        raise ValueError(f"audio_energy_threshold must be >= 0, got {energy_threshold}")
    if int(trailing_real_chunks) < 0:
        raise ValueError(f"audio_trailing_real_chunks must be >= 0, got {trailing_real_chunks}")

    if policy == "none":
        modes = ["real/unmodified"] * int(audio_chunks.shape[0])
        return audio_chunks.clone(), modes, {
            "policy": policy,
            "time_axis_preserved": True,
            "activity_chunks": None,
            "real_tail_chunks": None,
            "zero_chunks": 0,
        }
    if policy != "web_vad_zero":
        raise ValueError(f"Unknown audio silence policy: {policy}")

    # The browser VAD evaluates 4096 input samples at a typical 48 kHz.
    # Convert that duration to the 16 kHz model waveform used here.
    window_samples = max(
        1,
        int(round(int(sample_rate) * WEB_AUDIO_VAD_WINDOW_SAMPLES / WEB_AUDIO_INPUT_SAMPLE_RATE)),
    )
    output = audio_chunks.clone()
    modes: List[str] = []
    sending = False
    trailing = 0
    activity_chunks = 0
    real_tail_count = 0
    zero_count = 0
    pre_activity_zero_count = 0

    for chunk_idx in range(int(audio_chunks.shape[0])):
        chunk = audio_chunks[chunk_idx].detach().float()
        triggered = False
        for start in range(0, int(chunk.numel()), window_samples):
            window = chunk[start : start + window_samples]
            if window.numel() == 0:
                continue
            rms = float(window.square().mean().sqrt().item())
            if rms >= float(energy_threshold):
                triggered = True
                break

        if triggered:
            sending = True
            trailing = int(trailing_real_chunks)
            activity_chunks += 1
            modes.append("real/activity")
        elif sending and trailing > 0:
            trailing -= 1
            real_tail_count += 1
            modes.append("real/tail")
        else:
            output[chunk_idx].zero_()
            zero_count += 1
            if sending:
                modes.append("zero/silence")
            else:
                pre_activity_zero_count += 1
                modes.append("zero/pre_activity")

    stats = {
        "policy": policy,
        "time_axis_preserved": True,
        "sample_rate": int(sample_rate),
        "energy_threshold": float(energy_threshold),
        "vad_window_samples": int(window_samples),
        "trailing_real_chunks": int(trailing_real_chunks),
        "total_chunks": int(audio_chunks.shape[0]),
        "activity_chunks": int(activity_chunks),
        "real_tail_chunks": int(real_tail_count),
        "zero_chunks": int(zero_count),
        "pre_activity_zero_chunks": int(pre_activity_zero_count),
    }
    return output, modes, stats


def _control_name(token_id: int) -> str:
    if int(token_id) == TEXT_BEGIN:
        return "response"
    if int(token_id) == KEEP_SILENCE:
        return "keep_silence"
    return "unexpected"


@torch.no_grad()
def infer_episode_streaming(
    *,
    sample: Dict[str, Any],
    wrapper: HriMoTWrapper,
    tokenizer: Tokenizer,
    normalizer: ActionStateNormalizer,
    audio_encoder: Optional[torch.nn.Module],
    data_config: HriMotDataConfig,
    num_chunks: int,
    max_total_seq_len: int,
    args: argparse.Namespace,
    device: torch.device,
    audio_frontend_modes: Sequence[str],
    language_generator: torch.Generator,
    action_generator: torch.Generator,
    tts_client: TTSClient,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], List[SynthesizedUtterance]]:
    return infer_episode_streaming_impl(
        sample=sample,
        wrapper=wrapper,
        tokenizer=tokenizer,
        normalizer=normalizer,
        audio_encoder=audio_encoder,
        data_config=data_config,
        num_chunks=num_chunks,
        max_total_seq_len=max_total_seq_len,
        args=args,
        device=device,
        audio_frontend_modes=audio_frontend_modes,
        language_generator=language_generator,
        action_generator=action_generator,
        tts_client=tts_client,
        session_class=HriMotStreamingSession,
        sample_action_fn=sample_action_chunk,
        encode_audio_feature_batch_fn=encode_audio_feature_batch,
        control_name_fn=_control_name,
        emotion_token_to_text=EMOTION_EMOJI,
    )



def episode_rows_from_metadata(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    index_path = metadata.get("index_path")
    row_offsets = metadata.get("row_offsets")
    if not index_path or row_offsets is None:
        raise ValueError("Episode metadata must include index_path and row_offsets")
    rows: List[Dict[str, Any]] = []
    with open(index_path, "r", encoding="utf-8") as fp:
        for offset in row_offsets:
            fp.seek(int(offset))
            line = fp.readline().strip()
            if line:
                rows.append(json.loads(line))
    return sorted(rows, key=lambda row: int(row.get("episode_chunk_idx", 0)))


def load_wav_like_web(path: Path, *, sample_rate: int = 16_000) -> np.ndarray:
    """Use the same unmodified WAV decoding/resampling path as web/infer_wav.py."""
    import whisper

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"WAV file not found: {path}")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Expected a .wav file, got: {path}")
    audio = whisper.load_audio(str(path), sr=int(sample_rate)).astype(np.float32)
    if audio.size == 0:
        raise ValueError(f"WAV file contains no audio samples: {path}")
    return audio


def load_episode_audio_like_web(
    *,
    metadata: Dict[str, Any],
    t_start: torch.Tensor,
    num_chunks: int,
    sample_rate: int,
    samples_per_chunk: int,
) -> Tuple[torch.Tensor, List[str]]:
    """Decode at 16 kHz with whisper, then slice the episode into 400 ms chunks.

    This intentionally bypasses the dataset's soundfile/torchaudio audio path so
    HRI inference receives the same decoded waveform as web/infer_wav.py.  No
    gain, normalization, DC removal, clipping, or other enhancement is applied.
    """
    rows = episode_rows_from_metadata(metadata)
    if len(rows) < int(num_chunks):
        raise ValueError(f"Episode has {len(rows)} index rows for {num_chunks} requested chunks")
    if int(t_start.numel()) < int(num_chunks):
        raise ValueError(f"Episode has {t_start.numel()} timestamps for {num_chunks} requested chunks")

    decoded_by_path: Dict[str, np.ndarray] = {}
    chunks: List[np.ndarray] = []
    source_paths: List[str] = []
    for chunk_idx, row in enumerate(rows[: int(num_chunks)]):
        chunk_meta = row.get("chunk", {})
        partner = chunk_meta.get("partner", {}) if isinstance(chunk_meta, dict) else {}
        wav_value = partner.get("wav") if isinstance(partner, dict) else None
        if not wav_value:
            raise ValueError(f"Missing partner WAV path for episode chunk {chunk_idx}")
        wav_path = str(Path(str(wav_value)).expanduser().resolve())
        if not Path(wav_path).is_file():
            raise FileNotFoundError(f"Partner WAV does not exist: {wav_path}")
        if wav_path not in decoded_by_path:
            decoded = load_wav_like_web(Path(wav_path), sample_rate=int(sample_rate))
            decoded_by_path[wav_path] = decoded
            source_paths.append(wav_path)

        decoded = decoded_by_path[wav_path]
        source_start = int(round(float(t_start[chunk_idx].item()) * int(sample_rate)))
        source_end = source_start + int(samples_per_chunk)
        output = np.zeros(int(samples_per_chunk), dtype=np.float32)
        copy_start = max(0, source_start)
        copy_end = min(int(decoded.size), source_end)
        if copy_end > copy_start:
            output_start = copy_start - source_start
            output[output_start : output_start + (copy_end - copy_start)] = decoded[copy_start:copy_end]
        chunks.append(output)

    return torch.from_numpy(np.stack(chunks, axis=0)).float(), source_paths


def episode_category(episode: Dict[str, Any]) -> Optional[str]:
    metadata = episode.get("metadata", {})
    category = metadata.get("category")
    if category:
        return str(category)
    index_path = episode.get("index_path") or metadata.get("index_path")
    row_offsets = episode.get("row_offsets") or metadata.get("row_offsets")
    if not index_path or not row_offsets:
        return None
    with open(index_path, "r", encoding="utf-8") as fp:
        fp.seek(int(row_offsets[0]))
        line = fp.readline().strip()
    if not line:
        return None
    row = json.loads(line)
    chunk = row.get("chunk", {})
    if isinstance(chunk, dict) and chunk.get("category"):
        return str(chunk["category"])
    if row.get("category"):
        return str(row["category"])
    return None


def filtered_episode_indices(
    dataset: HriMotEpisodeDataset,
    *,
    category: str,
    max_episodes: Optional[int],
    selection: str = "random",
    seed: int = 1337,
    episode_id: Optional[str] = None,
    id_sample: Optional[int] = None,
) -> List[int]:
    indices: List[int] = []
    category = str(category)
    exact_selection = episode_id is not None or id_sample is not None
    for idx, episode in enumerate(dataset.episodes):
        metadata = episode.get("metadata", {})
        if episode_id is not None:
            if str(episode.get("episode_id", metadata.get("episode_id", ""))) != str(episode_id):
                continue
        elif id_sample is not None:
            if str(metadata.get("id_sample", "")) != str(id_sample):
                continue
        else:
            if category != "all":
                ep_category = episode_category(episode)
                if ep_category != category:
                    continue
        indices.append(idx)
        if exact_selection:
            break
    if exact_selection or max_episodes is None or len(indices) <= int(max_episodes):
        return indices
    if int(max_episodes) < 0:
        raise ValueError(f"max_episodes must be >= 0 or None, got {max_episodes}")
    if selection == "first":
        return indices[: int(max_episodes)]
    if selection == "random":
        return random.Random(int(seed)).sample(indices, k=int(max_episodes))
    raise ValueError(f"Unknown episode selection mode: {selection}")


def infer_original_robot_pkl(metadata: Dict[str, Any]) -> Optional[str]:
    rows = episode_rows_from_metadata(metadata)
    for row in rows:
        target = row.get("chunk", {}).get("target", {})
        if isinstance(target, dict) and target.get("robot_pkl"):
            return str(target["robot_pkl"])
    return None


def infer_original_target_json(metadata: Dict[str, Any]) -> Optional[str]:
    rows = episode_rows_from_metadata(metadata)
    for row in rows:
        target = row.get("chunk", {}).get("target", {})
        if isinstance(target, dict) and target.get("json"):
            return str(target["json"])
    return None


def _as_qpos_array(qpos_list: Any) -> np.ndarray:
    qpos = np.asarray(qpos_list)
    if qpos.ndim != 2:
        raise ValueError(f"qpos_list must be 2D, got shape {qpos.shape}")
    return qpos


def _to_original_qpos_list(pred_qpos: np.ndarray, original_qpos_list: Optional[Any]) -> Any:
    dtype = np.float32 if original_qpos_list is None else np.asarray(original_qpos_list).dtype
    pred_qpos = pred_qpos.astype(dtype, copy=False)
    if isinstance(original_qpos_list, list) or original_qpos_list is None:
        return [frame.astype(dtype, copy=False) for frame in pred_qpos]
    return pred_qpos


def _compact_prediction_qpos(full_qpos: np.ndarray) -> np.ndarray:
    """Convert source qpos to the 35-dim output layout expected by downstream tools."""
    if full_qpos.ndim != 2:
        raise ValueError(f"full_qpos must be 2D, got shape {full_qpos.shape}")
    # Current datasets already use the compact 35-dim S1 layout.  Keep it as-is;
    # predictions have already been written into ``robot_valid_qpos_indices``.
    if full_qpos.shape[1] == 35:
        return full_qpos.copy()

    # Backward compatibility for legacy 59-dim retarget PKLs.
    if full_qpos.shape[1] <= 42:
        raise ValueError(
            "full_qpos must use either the compact 35-dim layout or the legacy "
            f">=43-dim layout, got shape {full_qpos.shape}"
        )

    zeros_after_19 = np.zeros((full_qpos.shape[0], 6), dtype=full_qpos.dtype)
    zeros_after_42 = np.zeros((full_qpos.shape[0], 6), dtype=full_qpos.dtype)
    compact = np.concatenate(
        [
            full_qpos[:, 0:3],
            full_qpos[:, 7:20],
            zeros_after_19,
            full_qpos[:, 36:43],
            zeros_after_42,
        ],
        axis=1,
    )
    if compact.shape[1] != 35:
        raise ValueError(f"Expected compact qpos dim 35, got {compact.shape[1]}")
    return compact


def _format_time_tag(t_start: float, t_end: float) -> str:
    return f"t{float(t_start):.3f}-{float(t_end):.3f}".replace(".", "p")


def write_episode_audio_wav(
    *,
    audio_chunks: torch.Tensor,
    output_root: Path,
    episode_id: str,
    method: str,
    audio_t_start: float,
    audio_t_end: float,
    sample_rate: int,
) -> Path:
    """Save the exact complete waveform that will be replayed as 400 ms chunks."""
    import soundfile as sf

    if audio_chunks.dim() != 2:
        raise ValueError(f"audio_chunks must have shape [chunks, samples], got {tuple(audio_chunks.shape)}")
    safe_episode_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(episode_id))
    safe_method = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(method))
    time_tag = _format_time_tag(audio_t_start, audio_t_end)
    out_path = output_root / "processed_audio" / f"{safe_episode_id}__{safe_method}__{time_tag}.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    waveform = audio_chunks.detach().float().cpu().reshape(-1).numpy()
    sf.write(out_path, waveform, int(sample_rate), subtype="FLOAT")
    return out_path


def _strip_known_dataset_prefix(path: Path) -> Path:
    parts = [part for part in path.parts if part not in (path.anchor, "")]
    for marker in ("seamless_interaction_retarget", "seamless_interaction_full"):
        if marker in parts:
            marker_idx = parts.index(marker)
            tail = parts[marker_idx + 1 :]
            if tail:
                return Path(*tail)
    if len(parts) >= 4:
        return Path(*parts[-4:])
    return Path(*parts) if parts else Path(path.name)


def _safe_rel_output_path(
    original_path: Optional[str],
    episode_id: str,
    *,
    audio_t_start: float,
    audio_t_end: float,
    default_suffix: str = ".pkl",
) -> Path:
    safe_episode_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id)
    time_tag = _format_time_tag(audio_t_start, audio_t_end)
    suffix = f"__{safe_episode_id}__{time_tag}"
    if original_path:
        rel_path = _strip_known_dataset_prefix(Path(original_path))
        return rel_path.with_name(f"{rel_path.stem}{suffix}{rel_path.suffix or default_suffix}")
    return Path(f"{safe_episode_id}__{time_tag}{default_suffix}")


def _slice_sequence_like(value: Any, start: int, length: int) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        end = min(start + length, value.shape[0])
        if value.ndim > 0 and 0 <= start < value.shape[0]:
            sliced = value[start:end].copy()
            if sliced.shape[0] == length:
                return sliced
        shape = (length,) + tuple(value.shape[1:])
        return np.zeros(shape, dtype=value.dtype)
    if isinstance(value, list):
        end = min(start + length, len(value))
        if 0 <= start < len(value):
            sliced = list(value[start:end])
            if len(sliced) < length:
                sliced.extend([None] * (length - len(sliced)))
            return sliced
        return [None] * length
    return value


def _derive_root_from_compact_qpos(qpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    root_pos = np.zeros((qpos.shape[0], 3), dtype=qpos.dtype)
    root_pos[:, :2] = qpos[:, :2]
    yaw = qpos[:, 2]
    half_yaw = yaw * 0.5
    root_rot = np.zeros((qpos.shape[0], 4), dtype=qpos.dtype)
    root_rot[:, 2] = np.sin(half_yaw)
    root_rot[:, 3] = np.cos(half_yaw)
    return root_pos, root_rot


def _array_dtype_or(value: Any, fallback: np.dtype) -> np.dtype:
    if value is None:
        return np.dtype(fallback)
    dtype = np.asarray(value).dtype
    return np.dtype(fallback) if dtype == np.dtype("O") else dtype


def write_prediction_pkl(
    *,
    original_pkl: Optional[str],
    output_root: Path,
    episode_id: str,
    audio_t_start: float,
    audio_t_end: float,
    action_t_start: float,
    action_t_end: float,
    pred_action: np.ndarray,
    data_config: HriMotDataConfig,
) -> Path:
    pred_action = np.asarray(pred_action, dtype=np.float32)
    if pred_action.ndim != 3:
        raise ValueError(f"pred_action must have shape [chunks, horizon, dim], got {pred_action.shape}")
    num_chunks, action_horizon, action_dim = pred_action.shape
    if action_dim != data_config.action_dim:
        raise ValueError(f"Expected action dim {data_config.action_dim}, got {action_dim}")

    if original_pkl is not None and os.path.exists(original_pkl):
        with open(original_pkl, "rb") as fp:
            data = pickle.load(fp)
        if not isinstance(data, dict):
            raise ValueError(f"Robot pkl must contain a dict, got {type(data).__name__}: {original_pkl}")
        output_data = dict(data)
        original_qpos_list = data[data_config.robot_source_key]
        original_qpos = _as_qpos_array(original_qpos_list)
        source_dim = int(original_qpos.shape[1])
        fps = int(data.get("fps", data_config.robot_fps))
    else:
        output_data = {
            "fps": data_config.robot_fps,
            "format": "hri_mot_prediction",
        }
        original_qpos_list = None
        fps = data_config.robot_fps
        # Predictions are saved in the current compact S1 qpos layout, including
        # fixed chassis and gripper slots that are not learned by the model.
        source_dim = 35

    frame_count = int(num_chunks * action_horizon)
    action_start_frame = int(round(float(action_t_start) * fps))
    if original_pkl is not None and os.path.exists(original_pkl):
        full_qpos = _slice_sequence_like(original_qpos, action_start_frame, frame_count)
        full_qpos = np.asarray(full_qpos, dtype=np.float32)
        if full_qpos.ndim != 2 or full_qpos.shape[1] != source_dim:
            raise ValueError(
                f"Sliced source qpos must have shape [{frame_count}, {source_dim}], "
                f"got {full_qpos.shape}"
            )
    else:
        full_qpos = np.zeros((frame_count, source_dim), dtype=np.float32)
    valid_indices = np.asarray(data_config.robot_valid_qpos_indices, dtype=np.int64)
    for chunk_idx, chunk_pred in enumerate(pred_action):
        start_frame = int(chunk_idx * action_horizon)
        for step_idx, frame_pred in enumerate(chunk_pred):
            frame_idx = start_frame + step_idx
            if 0 <= frame_idx < full_qpos.shape[0]:
                full_qpos[frame_idx, valid_indices] = frame_pred.astype(np.float32, copy=False)

    output_qpos = _compact_prediction_qpos(full_qpos)
    output_data[data_config.robot_source_key] = _to_original_qpos_list(output_qpos, original_qpos_list)
    output_data["fps"] = fps
    output_data.setdefault("format", "hri_mot_prediction")

    for key in ("human_frames", "local_body_pos", "link_body_list"):
        if key in output_data:
            output_data[key] = _slice_sequence_like(output_data[key], action_start_frame, frame_count)

    root_pos, root_rot = _derive_root_from_compact_qpos(output_qpos)
    root_pos_dtype = _array_dtype_or(output_data.get("root_pos"), root_pos.dtype)
    root_rot_dtype = _array_dtype_or(output_data.get("root_rot"), root_rot.dtype)
    dof_dtype = _array_dtype_or(output_data.get("dof_pos"), output_qpos.dtype)
    output_data["root_pos"] = root_pos.astype(root_pos_dtype, copy=False)
    output_data["root_rot"] = root_rot.astype(root_rot_dtype, copy=False)
    output_data["dof_pos"] = output_qpos[:, 3:].astype(dof_dtype, copy=False)

    output_data["hri_mot_prediction"] = {
        "episode_id": episode_id,
        "source_robot_pkl": original_pkl,
        "robot_source_key": data_config.robot_source_key,
        "robot_valid_qpos_indices": list(map(int, data_config.robot_valid_qpos_indices)),
        "predicted_shape": list(pred_action.shape),
        "output_frames": int(frame_count),
        "output_qpos_dim": int(output_qpos.shape[1]),
        "output_dof_dim": int(output_qpos.shape[1] - 3),
        "audio_t_start": float(audio_t_start),
        "audio_t_end": float(audio_t_end),
        "action_t_start": float(action_t_start),
        "action_t_end": float(action_t_end),
    }

    rel_path = _safe_rel_output_path(
        original_pkl,
        episode_id,
        audio_t_start=action_t_start,
        audio_t_end=action_t_end,
    )
    out_path = output_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fp:
        pickle.dump(output_data, fp)
    return out_path


def _split_decoded_words(text: str) -> List[str]:
    """Split decoded text into transcript words.

    The dataset transcript stores already-tokenized words. At inference time we
    do not have acoustic word alignment, so this conservative whitespace split
    keeps punctuation attached, matching the common source-json style such as
    "Oh." or "that.".
    """
    return [part for part in re.findall(r"\S+", str(text).strip()) if part]


def _decoded_text_segment(text: str, start: float, end: float) -> Optional[Dict[str, Any]]:
    words = _split_decoded_words(text)
    if not words:
        return None
    start = float(start)
    end = float(end)
    if end <= start:
        end = start + 1e-3
    duration = end - start
    step = duration / max(len(words), 1)
    timed_words: List[Dict[str, Any]] = []
    for idx, word in enumerate(words):
        word_start = start + idx * step
        word_end = end if idx == len(words) - 1 else start + (idx + 1) * step
        timed_words.append(
            {
                "word": word,
                "start": float(word_start),
                "end": float(word_end),
                "score": 1.0,
            }
        )
    return {
        "words": timed_words,
        "start": float(start),
        "end": float(end),
        "transcript": " ".join(words),
    }


def decoded_response_transcript(
    *,
    response_records: Sequence[Dict[str, Any]],
    t_start: np.ndarray,
    chunk_size_sec: float,
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    if len(t_start) == 0:
        return segments
    for record in response_records:
        if record.get("control") not in ("response", "speaking"):
            continue
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        if record.get("playback_t_start") is not None and record.get("playback_t_end") is not None:
            start = float(record["playback_t_start"])
            end = float(record["playback_t_end"])
        elif "target_t_start" in record and "target_t_end" in record:
            start = float(record["target_t_start"])
            end = float(record["target_t_end"])
        else:
            chunk_idx = int(record.get("target_chunk_idx", int(record.get("chunk_idx", 0)) + 1))
            if chunk_idx < 0:
                continue
            if chunk_idx < len(t_start):
                start = float(t_start[chunk_idx])
            elif len(t_start):
                start = float(t_start[-1]) + float(chunk_size_sec) * (chunk_idx - len(t_start) + 1)
            else:
                start = float(chunk_idx) * float(chunk_size_sec)
            end = start + float(chunk_size_sec)
        segment = _decoded_text_segment(text, start, end)
        if segment is not None:
            segments.append(segment)
    return segments


def write_decoded_text_json(
    *,
    original_json: Optional[str],
    output_root: Path,
    episode_id: str,
    audio_t_start: float,
    audio_t_end: float,
    target_t_start: float,
    target_t_end: float,
    response_records: Sequence[Dict[str, Any]],
    t_start: np.ndarray,
    data_config: HriMotDataConfig,
) -> Path:
    segments = decoded_response_transcript(
        response_records=response_records,
        t_start=t_start,
        chunk_size_sec=data_config.chunk_size_sec,
    )
    output_data: Dict[str, Any] = {
        "id": episode_id,
        "metadata:transcript": segments,
        "metadata:vad": [
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
            }
            for segment in segments
        ],
        "hri_mot_decoded_text": {
            "episode_id": episode_id,
            "source_target_json": original_json,
            "audio_t_start": float(audio_t_start),
            "audio_t_end": float(audio_t_end),
            "target_t_start": float(target_t_start),
            "target_t_end": float(target_t_end),
            "chunk_size_sec": float(data_config.chunk_size_sec),
            "timestamp_policy": "TTS playback text is committed in 400 ms target intervals; action c sees text committed through the previous playback interval",
            "num_segments": int(len(segments)),
            "predicted_responses": int(sum(str(item.get("control", "")) == "response" for item in response_records)),
            "predicted_keep_silence": int(
                sum(str(item.get("control", "")) == "keep_silence" for item in response_records)
            ),
            "response_records": list(response_records),
        },
    }
    rel_path = _safe_rel_output_path(
        original_json,
        episode_id,
        audio_t_start=target_t_start,
        audio_t_end=target_t_end,
        default_suffix=".json",
    )
    out_path = output_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(output_data, fp, indent=2, ensure_ascii=False)
    return out_path


def write_decoded_text_txt(
    *,
    decoded_text_json: Path,
    response_records: Sequence[Dict[str, Any]],
) -> Path:
    out_path = decoded_text_json.with_suffix(".txt")
    lines: List[str] = []
    for record in response_records:
        if str(record.get("control", "")) not in ("response", "speaking"):
            continue
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        start = float(record.get("playback_t_start", record.get("target_t_start", record.get("audio_t_start", 0.0))))
        end = float(record.get("playback_t_end", record.get("target_t_end", start)))
        lines.append(f"[{start:.3f}-{end:.3f}] {text}")
    if not lines:
        lines.append("[no response text generated; all decisions were KEEP_SILENCE]")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    return out_path


def write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)


def rank_metrics_path(output_dir: Path, rank: int, world_size: int) -> Path:
    return output_dir / "metrics.json" if world_size == 1 else output_dir / f"metrics_rank{rank:03d}_of{world_size:03d}.json"


def wait_and_write_aggregate_metrics(
    *,
    output_dir: Path,
    world_size: int,
    timeout: int,
    data_config: HriMotDataConfig,
) -> None:
    rank_paths = [rank_metrics_path(output_dir, idx, world_size) for idx in range(world_size)]
    deadline = time.time() + timeout
    while time.time() < deadline and not all(path.is_file() for path in rank_paths):
        time.sleep(5.0)
    missing = [str(path) for path in rank_paths if not path.is_file()]
    if missing:
        print(f"[rank 0/{world_size}] missing rank metrics, skip aggregate: {missing}", flush=True)
        return
    rank_metrics = []
    for path in rank_paths:
        with path.open("r", encoding="utf-8") as fp:
            rank_metrics.append(json.load(fp))
    total_values = sum(
        sum(int(ep["chunks"]) * data_config.action_horizon * data_config.action_dim for ep in item["episodes"])
        for item in rank_metrics
    )
    total_sq_error = sum(
        sum(float(ep["mse"]) * int(ep["chunks"]) * data_config.action_horizon * data_config.action_dim for ep in item["episodes"])
        for item in rank_metrics
    )
    aggregate = dict(rank_metrics[0])
    aggregate.pop("rank", None)
    aggregate["world_size"] = int(world_size)
    aggregate["num_episodes"] = int(sum(item["num_episodes"] for item in rank_metrics))
    aggregate["overall_mse"] = float(total_sq_error / max(total_values, 1))
    aggregate["rank_metrics"] = [str(path) for path in rank_paths]
    aggregate["episodes"] = sorted(
        [episode for item in rank_metrics for episode in item.get("episodes", [])],
        key=lambda item: str(item.get("episode_id", "")),
    )
    aggregate_path = output_dir / "metrics.json"
    write_metrics(aggregate_path, aggregate)
    print(f"[rank 0/{world_size}] aggregate_mse={aggregate['overall_mse']:.8f}", flush=True)
    print(f"[rank 0/{world_size}] aggregate metrics: {aggregate_path}", flush=True)


def load_inference_norm_stats(
    *,
    checkpoint: Path,
    train_dir: Path,
    norm_stats_path: Optional[str],
) -> Dict[str, Any]:
    try:
        checkpoint_obj = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint_obj = torch.load(checkpoint, map_location="cpu")
    stats = _coerce_norm_stats(checkpoint_obj.get("norm_stats") if isinstance(checkpoint_obj, dict) else None)
    if stats is not None:
        return stats
    if norm_stats_path:
        return load_norm_stats_json(norm_stats_path)
    default_path = train_dir / "norm_stats.json"
    if default_path.is_file():
        return load_norm_stats_json(str(default_path))
    raise RuntimeError(
        "Normalization stats are missing. Provide --norm-stats or use a checkpoint containing norm_stats."
    )


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = resolve_rank_info(args)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    train_dir = Path(args.train_dir) if args.train_dir else checkpoint.parent
    file_train_args = load_train_args(train_dir)
    train_args, speak_conditioning_source = resolve_inference_train_args(
        checkpoint=checkpoint,
        file_train_args=file_train_args,
        requested_speak_conditioning=args.speak_conditioning,
    )
    checkpoint_dir = args.checkpoint_dir or train_args.get("checkpoint_dir") or "checkpoints"
    seed = int(args.seed if args.seed is not None else train_args.get("seed", 1337))
    text_mode = args.text_chunk_time_mode or train_args.get("text_chunk_time_mode") or "start"
    max_total_seq_len = int(args.max_total_seq_len or train_args.get("max_total_seq_len", 4096))
    chunks_per_step = int(
        args.chunks_per_step if args.chunks_per_step is not None else train_args.get("chunks_per_step", 0)
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else train_dir / "streaming_inference"
    )

    rank_seed = seed + rank
    random.seed(rank_seed)
    np.random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
        torch.set_float32_matmul_precision("high")

    device = resolve_device(args.device, local_rank=local_rank)
    language_rng_seed = rank_seed
    action_rng_seed = rank_seed + 1
    language_generator = torch.Generator(device=device).manual_seed(language_rng_seed)
    action_generator = torch.Generator(device=device).manual_seed(action_rng_seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    tts_client = build_tts_client_from_args(args)
    print(f"[rank {rank}/{world_size}] checkpoint: {checkpoint}", flush=True)
    print(f"[rank {rank}/{world_size}] base checkpoint dir: {checkpoint_dir}", flush=True)
    print(f"[rank {rank}/{world_size}] test index: {args.test_index}", flush=True)
    print(f"[rank {rank}/{world_size}] output dir: {output_dir}", flush=True)
    print(f"[rank {rank}/{world_size}] device: {device} local_rank={local_rank}", flush=True)
    print(f"[rank {rank}/{world_size}] text chunk time mode: {text_mode}", flush=True)
    print(f"[rank {rank}/{world_size}] category filter: {args.category}", flush=True)
    print(
        f"[rank {rank}/{world_size}] episode selection: {args.episode_selection} "
        f"max_episodes={args.max_episodes} seed={seed}",
        flush=True,
    )
    print(
        f"[rank {rank}/{world_size}] exact episode selector: "
        f"episode_id={args.episode_id!r} id_sample={args.id_sample!r}",
        flush=True,
    )
    print(f"[rank {rank}/{world_size}] state source: {args.state_source}", flush=True)
    print(
        f"[rank {rank}/{world_size}] speak conditioning: "
        f"requested={args.speak_conditioning} resolved={train_args['speak_conditioning']} "
        f"source={speak_conditioning_source}",
        flush=True,
    )
    print(
        f"[rank {rank}/{world_size}] action postprocess: "
        f"c2_bridge_frames={args.c2_bridge_frames} "
        f"offline_lowpass_cutoff_hz={args.offline_lowpass_cutoff_hz}",
        flush=True,
    )
    print(
        f"[rank {rank}/{world_size}] video postprocess: enabled={not args.skip_video_postprocess} "
        f"render_script={args.render_script} render_xml={args.render_xml_path} "
        f"gt_root={args.gt_video_root} "
        f"layout=prediction_left_gt_right",
        flush=True,
    )
    print(
        f"[rank {rank}/{world_size}] audio preprocess: {args.audio_preprocess} "
        f"target_active_rms={args.audio_target_active_rms} target_rms={args.audio_target_rms} "
        f"gain_db=[{args.audio_min_gain_db}, {args.audio_max_gain_db}] peak={args.audio_peak_limit}",
        flush=True,
    )
    print(
        f"[rank {rank}/{world_size}] audio loader: whisper.load_audio(sr=16000), "
        "matching web/infer_wav.py",
        flush=True,
    )
    print(
        f"[rank {rank}/{world_size}] audio silence policy: {args.audio_silence_policy} "
        f"energy_threshold={args.audio_energy_threshold} "
        f"trailing_real_chunks={args.audio_trailing_real_chunks}",
        flush=True,
    )
    print(
        f"[rank {rank}/{world_size}] language sampling: temperature={args.text_temperature} "
        f"top_k={args.text_top_k} top_p={args.text_top_p} "
        f"silence_gate={args.min_reply_silence_sec}s",
        flush=True,
    )
    print(
        f"[rank {rank}/{world_size}] barge-in mode: {args.barge_in_mode} "
        f"min_reply_silence={args.min_reply_silence_sec}s "
        f"deferred_reply_prob_threshold={args.deferred_reply_prob_threshold}",
        flush=True,
    )
    print(
        f"[rank {rank}/{world_size}] independent RNG streams: "
        f"language_seed={language_rng_seed} action_seed={action_rng_seed}",
        flush=True,
    )
    print(
        f"[rank {rank}/{world_size}] EOF zero continuation: "
        f"max={args.audio_eof_flush_sec}s (generates synchronized actions and TTS)",
        flush=True,
    )

    data_config = HriMotDataConfig(
        test_index_path=args.test_index,
        text_chunk_time_mode=text_mode,
        max_episodes=None,
        strict_missing_files=args.strict_missing_files,
    )
    episode_index_cache = args.episode_index_cache or f"{args.test_index}.hri_mot_offsets.jsonl"
    dataset = HriMotEpisodeDataset(
        [args.test_index],
        config=data_config,
        split="test",
        episode_index_cache=episode_index_cache,
    )
    if len(dataset) == 0:
        raise RuntimeError("No test episodes loaded")
    selected_indices = filtered_episode_indices(
        dataset,
        category=args.category,
        max_episodes=args.max_episodes,
        selection=args.episode_selection,
        seed=seed,
        episode_id=args.episode_id,
        id_sample=args.id_sample,
    )
    if not selected_indices:
        selector = (
            f"episode_id={args.episode_id!r}"
            if args.episode_id is not None
            else f"id_sample={args.id_sample!r}"
            if args.id_sample is not None
            else f"category={args.category!r}"
        )
        raise RuntimeError(
            f"No test episodes matched {selector} "
            f"from dataset size {len(dataset)}"
        )
    shard_indices = selected_indices[rank::world_size]
    print(
        f"[rank {rank}/{world_size}] loaded episodes: {len(dataset)} "
        f"selected episodes: {len(selected_indices)} shard episodes: {len(shard_indices)}",
        flush=True,
    )
    if episode_index_cache:
        print(f"[rank {rank}/{world_size}] episode offset cache: {episode_index_cache}", flush=True)

    norm_stats = load_inference_norm_stats(
        checkpoint=checkpoint,
        train_dir=train_dir,
        norm_stats_path=args.norm_stats,
    )
    normalizer = ActionStateNormalizer(norm_stats, device=device)
    print(
        f"[rank {rank}/{world_size}] norm stats: "
        f"state_count={norm_stats['state_count']} action_count={norm_stats['action_count']}",
        flush=True,
    )
    base_model, tokenizer, qwen_omni_ckpt, audio_tower_ckpt = load_base_model_and_tokenizer(
        checkpoint_dir, device, seed
    )
    use_audio_features = not (args.no_audio_features or bool(train_args.get("no_audio_features", False)))
    audio_encoder = None
    if use_audio_features:
        audio_encoder = load_audio_encoder(qwen_omni_ckpt, audio_tower_ckpt, device)
        audio_encoder.eval()
    wrapper = build_wrapper(
        base_model=base_model,
        data_config=data_config,
        train_args=train_args,
        device=device,
    )
    step, missing = load_trainable_state(wrapper, str(checkpoint), device)
    wrapper.eval()
    wrapper.base_model.eval()
    if (
        args.emotion_override != "auto"
        and not wrapper.use_emotion_conditioning
    ):
        raise ValueError(
            "--emotion-override requires a checkpoint with "
            "emotion conditioning"
        )
    print(f"[rank {rank}/{world_size}] loaded trainable checkpoint step={step} missing_trainable_keys={missing}", flush=True)
    print(f"[rank {rank}/{world_size}] audio features: {use_audio_features}", flush=True)
    print(
        f"[rank {rank}/{world_size}] speak_conditioning={wrapper.speak_conditioning} "
        f"emotion={wrapper.use_emotion_conditioning} "
        f"TTS mode={args.tts_mode} language={args.tts_language} "
        f"speaker={args.tts_speaker}",
        flush=True,
    )
    if use_audio_features:
        print(f"[rank {rank}/{world_size}] audio feature mode: {args.audio_feature_mode}", flush=True)

    config_payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(step),
        "emotion_override": str(args.emotion_override),
        "test_index": args.test_index,
        "episode_id_filter": args.episode_id,
        "id_sample_filter": args.id_sample,
        "text_chunk_time_mode": text_mode,
        "category": args.category,
        "episode_selection": args.episode_selection,
        "selection_seed": int(seed),
        "sampling_steps": int(args.sampling_steps),
        "solver": args.solver,
        "initial_noise": args.initial_noise,
        "state_source": args.state_source,
        "c2_bridge_frames": int(args.c2_bridge_frames),
        "offline_lowpass_cutoff_hz": float(args.offline_lowpass_cutoff_hz),
        "video_postprocess": not bool(args.skip_video_postprocess),
        "video_postprocess_strict": bool(args.video_postprocess_strict),
        "render_script": str(Path(args.render_script).expanduser().resolve()),
        "render_conda_env": str(args.render_conda_env),
        "render_xml_path": (
            str(Path(args.render_xml_path).expanduser().resolve())
            if args.render_xml_path
            else None
        ),
        "gt_video_root": str(Path(args.gt_video_root).expanduser().resolve()),
        "comparison_layout": "prediction_left_gt_right",
        "max_response_tokens": int(args.max_response_tokens),
        "text_temperature": float(args.text_temperature),
        "text_top_k": int(args.text_top_k),
        "text_top_p": float(args.text_top_p),
        "independent_rng_streams": True,
        "language_rng_seed": int(language_rng_seed),
        "action_rng_seed": int(action_rng_seed),
        "speak_conditioning": str(wrapper.speak_conditioning),
        "speak_conditioning_requested": str(args.speak_conditioning),
        "speak_conditioning_source": str(speak_conditioning_source),
        **emotion_inference_config(wrapper),
        "tts_mode": str(args.tts_mode),
        "tts_url": str(args.tts_url),
        "tts_language": str(args.tts_language),
        "tts_speaker": str(args.tts_speaker),
        "tts_instruct": str(args.tts_instruct),
        "barge_in_mode": str(args.barge_in_mode),
        "min_reply_silence_sec": float(args.min_reply_silence_sec),
        "deferred_reply_prob_threshold": float(
            args.deferred_reply_prob_threshold
        ),
        "silence_gate": True,
        "silence_gate_policy": "defer_text_begin_until_vad_silence",
        "max_total_seq_len": int(max_total_seq_len),
        "chunks_per_step": int(chunks_per_step),
        "audio_features": bool(use_audio_features),
        "audio_feature_mode": args.audio_feature_mode if use_audio_features else "none",
        "audio_loader": "whisper.load_audio",
        "audio_sample_rate": int(data_config.audio_sample_rate),
        "audio_preprocess": args.audio_preprocess,
        "audio_target_active_rms": float(args.audio_target_active_rms),
        "audio_target_rms": float(args.audio_target_rms),
        "audio_active_quantile": float(args.audio_active_quantile),
        "audio_frame_ms": float(args.audio_frame_ms),
        "audio_hop_ms": float(args.audio_hop_ms),
        "audio_min_gain_db": float(args.audio_min_gain_db),
        "audio_max_gain_db": float(args.audio_max_gain_db),
        "audio_peak_limit": float(args.audio_peak_limit),
        "audio_silence_policy": args.audio_silence_policy,
        "audio_energy_threshold": float(args.audio_energy_threshold),
        "audio_trailing_real_chunks": int(args.audio_trailing_real_chunks),
        "audio_eof_flush_sec": float(args.audio_eof_flush_sec),
        "max_episodes_after_category_filter": int(args.max_episodes) if args.max_episodes is not None else None,
        "dataset_episodes": int(len(dataset)),
        "selected_episodes": int(len(selected_indices)),
        "selected_dataset_indices": [int(idx) for idx in selected_indices],
        "selected_episode_ids": [str(dataset.episodes[idx].get("episode_id", "")) for idx in selected_indices],
        "audio_to_action_offset_chunks": 1,
        "robot_valid_qpos_indices": list(map(int, VALID_QPOS_DIMS)),
    }
    if rank == 0:
        write_metrics(output_dir / "inference_config.json", config_payload)

    total_sq_error = 0.0
    total_values = 0
    episode_metrics: List[Dict[str, Any]] = []
    progress = tqdm(
        shard_indices,
        desc=f"hri-mot infer rank {rank}",
        disable=args.disable_tqdm or not sys.stderr.isatty() or rank != 0,
    )
    for dataset_idx in progress:
        sample = dataset[dataset_idx]
        num_chunks_total = int(sample["target_action"].shape[0])
        actual_chunks = num_chunks_total if chunks_per_step <= 0 or chunks_per_step >= num_chunks_total else chunks_per_step
        episode_id = str(sample["episode_id"])
        t_start_np = sample["t_start"][:actual_chunks].detach().cpu().numpy()
        audio_t_start = float(t_start_np[0]) if len(t_start_np) else 0.0
        audio_t_end = float(t_start_np[actual_chunks - 1] + data_config.chunk_size_sec) if actual_chunks else 0.0
        metadata = dict(sample["metadata"])
        if "row_offsets" not in metadata:
            metadata["row_offsets"] = dataset.episodes[dataset_idx]["row_offsets"]
        if "index_path" not in metadata:
            metadata["index_path"] = dataset.episodes[dataset_idx]["index_path"]

        samples_per_chunk = int(round(data_config.chunk_size_sec * data_config.audio_sample_rate))
        web_audio, audio_source_wavs = load_episode_audio_like_web(
            metadata=metadata,
            t_start=sample["t_start"],
            num_chunks=actual_chunks,
            sample_rate=data_config.audio_sample_rate,
            samples_per_chunk=samples_per_chunk,
        )
        inference_sample = dict(sample)
        processed_audio, audio_preprocess_stats = preprocess_episode_audio(
            web_audio,
            method=args.audio_preprocess,
            sample_rate=data_config.audio_sample_rate,
            target_active_rms=args.audio_target_active_rms,
            target_rms=args.audio_target_rms,
            active_quantile=args.audio_active_quantile,
            frame_ms=args.audio_frame_ms,
            hop_ms=args.audio_hop_ms,
            min_gain_db=args.audio_min_gain_db,
            max_gain_db=args.audio_max_gain_db,
            peak_limit=args.audio_peak_limit,
        )
        audio_preprocess_stats["loader"] = "whisper.load_audio"
        audio_preprocess_stats["source_wavs"] = audio_source_wavs
        model_audio, audio_frontend_modes, audio_silence_stats = apply_streaming_audio_silence_policy(
            processed_audio,
            policy=args.audio_silence_policy,
            sample_rate=data_config.audio_sample_rate,
            energy_threshold=args.audio_energy_threshold,
            trailing_real_chunks=args.audio_trailing_real_chunks,
        )
        audio_preprocess_stats["streaming_silence"] = audio_silence_stats
        audio_preprocess_stats["model_input_rms"] = float(model_audio.float().square().mean().sqrt().item())
        audio_preprocess_stats["model_input_peak"] = float(model_audio.float().abs().max().item())
        inference_sample["partner_audio"] = sample["partner_audio"].clone()
        inference_sample["partner_audio"][:actual_chunks] = model_audio
        saved_audio_method = (
            args.audio_preprocess
            if args.audio_silence_policy == "none"
            else f"{args.audio_preprocess}_{args.audio_silence_policy}"
        )
        processed_audio_wav = write_episode_audio_wav(
            audio_chunks=model_audio,
            output_root=output_dir,
            episode_id=episode_id,
            method=saved_audio_method,
            audio_t_start=audio_t_start,
            audio_t_end=audio_t_end,
            sample_rate=data_config.audio_sample_rate,
        )
        audio_preprocess_stats["saved_wav"] = str(processed_audio_wav)
        print(
            f"[rank {rank}/{world_size}] episode={episode_id} audio preprocess={args.audio_preprocess} "
            f"raw_rms={audio_preprocess_stats['raw_rms']:.6g} "
            f"gain_db={audio_preprocess_stats['applied_gain_db']:.3f} "
            f"processed_rms={audio_preprocess_stats['processed_rms']:.6g} "
            f"peak={audio_preprocess_stats['processed_peak']:.6g} wav={processed_audio_wav}",
            flush=True,
        )
        print(
            f"[rank {rank}/{world_size}] episode={episode_id} silence policy={args.audio_silence_policy} "
            f"activity={audio_silence_stats['activity_chunks']} "
            f"real_tail={audio_silence_stats['real_tail_chunks']} "
            f"zero={audio_silence_stats['zero_chunks']}",
            flush=True,
        )

        pred_action, response_records, tts_utterances = infer_episode_streaming(
            sample=inference_sample,
            wrapper=wrapper,
            tokenizer=tokenizer,
            normalizer=normalizer,
            audio_encoder=audio_encoder,
            data_config=data_config,
            num_chunks=actual_chunks,
            max_total_seq_len=max_total_seq_len,
            args=args,
            device=device,
            audio_frontend_modes=audio_frontend_modes,
            language_generator=language_generator,
            action_generator=action_generator,
            tts_client=tts_client,
        )
        tts_wav_paths: List[str] = []
        if not args.no_tts_save_audio:
            for utterance_index, utterance in enumerate(tts_utterances):
                tts_path = output_dir / episode_id / f"tts_response_{utterance_index:03d}.wav"
                write_pcm16_wav(tts_path, utterance.pcm, utterance.sample_rate)
                tts_wav_paths.append(str(tts_path))
        for record in response_records:
            utterance_index = record.get("tts_utterance_index")
            if utterance_index is not None and int(utterance_index) < len(tts_wav_paths):
                record["tts_wav"] = tts_wav_paths[int(utterance_index)]
        raw_boundary = action_boundary_stats(pred_action)
        raw_motion = action_motion_stats(pred_action)
        pred_action = smooth_action_chunks_offline(
            pred_action,
            bridge_frames=int(args.c2_bridge_frames),
            lowpass_cutoff_hz=float(args.offline_lowpass_cutoff_hz),
            fps=float(data_config.robot_fps),
        )
        smoothed_boundary = action_boundary_stats(pred_action)
        smoothed_motion = action_motion_stats(pred_action)
        target_action = sample["target_action"][:actual_chunks].to(device=device).float()
        evaluated_pred_action = pred_action[:actual_chunks]
        mse = F.mse_loss(evaluated_pred_action, target_action, reduction="mean")
        sq_error = (evaluated_pred_action - target_action).pow(2).sum().item()
        count = int(target_action.numel())
        total_sq_error += float(sq_error)
        total_values += count

        pred_np = pred_action.detach().cpu().numpy()
        predicted_chunk_count = int(pred_action.shape[0])
        processed_audio_t_end = float(
            audio_t_start + predicted_chunk_count * data_config.chunk_size_sec
        )
        action_t_start = audio_t_start + data_config.chunk_size_sec
        action_t_end = processed_audio_t_end + data_config.chunk_size_sec
        original_pkl = infer_original_robot_pkl(metadata)
        original_json = infer_original_target_json(metadata)
        out_pkl = write_prediction_pkl(
            original_pkl=original_pkl,
            output_root=output_dir,
            episode_id=episode_id,
            audio_t_start=audio_t_start,
            audio_t_end=processed_audio_t_end,
            action_t_start=action_t_start,
            action_t_end=action_t_end,
            pred_action=pred_np,
            data_config=data_config,
        )
        video_outputs = maybe_postprocess_prediction_video(
            skip=bool(args.skip_video_postprocess),
            strict=bool(args.video_postprocess_strict),
            prediction_pkl=out_pkl,
            action_t_start=action_t_start,
            source_robot_pkl=original_pkl,
            render_script=Path(args.render_script),
            render_conda_env=str(args.render_conda_env),
            render_xml_path=(Path(args.render_xml_path) if args.render_xml_path else None),
            gt_video_root=Path(args.gt_video_root),
        )
        out_text_json = write_decoded_text_json(
            original_json=original_json,
            output_root=output_dir,
            episode_id=episode_id,
            audio_t_start=audio_t_start,
            audio_t_end=processed_audio_t_end,
            target_t_start=action_t_start,
            target_t_end=action_t_end,
            response_records=response_records,
            t_start=t_start_np,
            data_config=data_config,
        )
        out_text_txt = write_decoded_text_txt(
            decoded_text_json=out_text_json,
            response_records=response_records,
        )
        controls = [str(item.get("control", "")) for item in response_records]
        eof_flush_chunks = int(sum(bool(item.get("language_only_flush", False)) for item in response_records))
        eof_action_flush_chunks = int(
            sum(
                str(item.get("audio_kind", "")) == "eof_action_flush"
                for item in response_records
            )
        )
        episode_metrics.append(
            {
                "episode_id": episode_id,
                "dataset_index": int(dataset_idx),
                "mse": float(mse.item()),
                "chunks": int(actual_chunks),
                "frames": int(actual_chunks * data_config.action_horizon),
                "audio_t_start": audio_t_start,
                "audio_t_end": audio_t_end,
                "processed_audio_t_end": processed_audio_t_end,
                "action_t_start": action_t_start,
                "action_t_end": action_t_end,
                "target_text_t_start": action_t_start,
                "target_text_t_end": action_t_end,
                "predicted_responses": int(sum(control == "response" for control in controls)),
                "predicted_keep_silence": int(sum(control == "keep_silence" for control in controls)),
                "barge_in_count": int(
                    sum(
                        bool(item.get("response_interrupted", False))
                        for item in response_records
                    )
                ),
                "language_input_chunks": int(len(response_records)),
                "action_chunks": predicted_chunk_count,
                "eof_language_flush_chunks": eof_flush_chunks,
                "eof_action_flush_chunks": eof_action_flush_chunks,
                "raw_boundary_jump_mean": raw_boundary["boundary_jump_mean"],
                "raw_boundary_jump_max": raw_boundary["boundary_jump_max"],
                "boundary_jump_mean": smoothed_boundary["boundary_jump_mean"],
                "boundary_jump_max": smoothed_boundary["boundary_jump_max"],
                "raw_motion_stats": raw_motion,
                "smoothed_motion_stats": smoothed_motion,
                "response_records": response_records,
                "prediction_pkl": str(out_pkl),
                "decoded_text_json": str(out_text_json),
                "decoded_text_txt": str(out_text_txt),
                "source_robot_pkl": original_pkl,
                "source_target_json": original_json,
                "processed_audio_wav": str(processed_audio_wav),
                "tts_wavs": tts_wav_paths,
                "tts_outputs": [
                    {
                        "index": int(index),
                        "text": utterance.text,
                        "duration_sec": float(utterance.duration_sec),
                        "sample_rate": int(utterance.sample_rate),
                        "backend": utterance.backend,
                        "alignment_method": utterance.alignment_method,
                        "wav": tts_wav_paths[index] if index < len(tts_wav_paths) else None,
                    }
                    for index, utterance in enumerate(tts_utterances)
                ],
                "audio_preprocess": audio_preprocess_stats,
                "video_outputs": video_outputs,
            }
        )
        progress.set_postfix(mse=f"{mse.item():.6f}")

    overall_mse = total_sq_error / max(total_values, 1)
    metrics = {
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(step),
        "test_index": args.test_index,
        "episode_id_filter": args.episode_id,
        "id_sample_filter": args.id_sample,
        "rank": int(rank),
        "world_size": int(world_size),
        "dataset_episodes": int(len(dataset)),
        "selected_episodes": int(len(selected_indices)),
        "num_episodes": len(episode_metrics),
        "overall_mse": float(overall_mse),
        "sampling_steps": int(args.sampling_steps),
        "solver": args.solver,
        "initial_noise": args.initial_noise,
        "state_source": args.state_source,
        "c2_bridge_frames": int(args.c2_bridge_frames),
        "offline_lowpass_cutoff_hz": float(args.offline_lowpass_cutoff_hz),
        "video_postprocess": not bool(args.skip_video_postprocess),
        "video_postprocess_strict": bool(args.video_postprocess_strict),
        "render_script": str(Path(args.render_script).expanduser().resolve()),
        "render_conda_env": str(args.render_conda_env),
        "render_xml_path": (
            str(Path(args.render_xml_path).expanduser().resolve())
            if args.render_xml_path
            else None
        ),
        "gt_video_root": str(Path(args.gt_video_root).expanduser().resolve()),
        "comparison_layout": "prediction_left_gt_right",
        "inference_mode": "kv_cache_streaming_chunk",
        "text_chunk_time_mode": text_mode,
        "category": args.category,
        "episode_selection": args.episode_selection,
        "selection_seed": int(seed),
        "audio_preprocess": args.audio_preprocess,
        "audio_target_active_rms": float(args.audio_target_active_rms),
        "audio_target_rms": float(args.audio_target_rms),
        "audio_active_quantile": float(args.audio_active_quantile),
        "audio_min_gain_db": float(args.audio_min_gain_db),
        "audio_max_gain_db": float(args.audio_max_gain_db),
        "audio_peak_limit": float(args.audio_peak_limit),
        "audio_silence_policy": args.audio_silence_policy,
        "audio_energy_threshold": float(args.audio_energy_threshold),
        "audio_trailing_real_chunks": int(args.audio_trailing_real_chunks),
        "audio_eof_flush_sec": float(args.audio_eof_flush_sec),
        "max_response_tokens": int(args.max_response_tokens),
        "text_temperature": float(args.text_temperature),
        "text_top_k": int(args.text_top_k),
        "text_top_p": float(args.text_top_p),
        "independent_rng_streams": True,
        "language_rng_seed": int(language_rng_seed),
        "action_rng_seed": int(action_rng_seed),
        "speak_conditioning": str(wrapper.speak_conditioning),
        "speak_conditioning_requested": str(args.speak_conditioning),
        "speak_conditioning_source": str(speak_conditioning_source),
        **emotion_inference_config(wrapper),
        "tts_mode": str(args.tts_mode),
        "tts_url": str(args.tts_url),
        "tts_language": str(args.tts_language),
        "tts_speaker": str(args.tts_speaker),
        "tts_instruct": str(args.tts_instruct),
        "barge_in_mode": str(args.barge_in_mode),
        "min_reply_silence_sec": float(args.min_reply_silence_sec),
        "deferred_reply_prob_threshold": float(
            args.deferred_reply_prob_threshold
        ),
        "silence_gate": True,
        "silence_gate_policy": "defer_text_begin_until_vad_silence",
        "max_total_seq_len": int(max_total_seq_len),
        "requested_chunks_per_episode": int(chunks_per_step),
        "robot_valid_qpos_indices": list(map(int, VALID_QPOS_DIMS)),
        "episodes": episode_metrics,
    }
    metrics_path = rank_metrics_path(output_dir, rank, world_size)
    write_metrics(metrics_path, metrics)
    print(f"[rank {rank}/{world_size}] overall_mse={overall_mse:.8f}", flush=True)
    print(f"[rank {rank}/{world_size}] metrics: {metrics_path}", flush=True)
    print(f"[rank {rank}/{world_size}] prediction output root: {output_dir}", flush=True)

    if world_size > 1 and rank == 0:
        wait_and_write_aggregate_metrics(
            output_dir=output_dir,
            world_size=world_size,
            timeout=args.metrics_wait_timeout,
            data_config=data_config,
        )


if __name__ == "__main__":
    main()
