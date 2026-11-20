#!/usr/bin/env python3
"""Run HRI-MoT streaming inference directly on one WAV file.

This is intentionally separate from ``infer_hri_mot.py``.  It follows the
400 ms streaming loop used by ``web/infer_wav.py``, but:

* the default audio frontend keeps two real tail chunks and then replaces
  low-energy chunks with exact zeros while preserving the action timeline;
* the HRI action expert samples one next-chunk action with Euler integration;
* the first robot-state window defaults to the training state mean, because a
  standalone WAV does not contain proprioception, and can optionally use the
  idle motion recorded in the training arguments;
* later state windows default to closed-loop rollout from the previous
  C2-stitched action prediction;
* decoded text, per-chunk control probabilities, and a downstream action pkl
  are written to a new output directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audiointeraction.hri_mot.emotion_conditioning import EMOTION_LABELS

from scripts.hri_mot.infer_hri_mot import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_SYSTEM_PROMPT,
    EMOTION_EMOJI,
    HriMotStreamingSession,
    WEB_AUDIO_ENERGY_THRESHOLD,
    WEB_AUDIO_SILENCE_SEC,
    WEB_AUDIO_TRAILING_REAL_CHUNKS,
    _control_name,
    action_boundary_stats,
    apply_streaming_audio_silence_policy,
    build_wrapper,
    load_base_model_and_tokenizer,
    load_wav_like_web,
    load_inference_norm_stats,
    load_train_args,
    emotion_inference_config,
    resolve_device,
    resolve_inference_train_args,
    sample_action_chunk,
    write_prediction_pkl,
)
from src.audiointeraction.hri_mot.streaming_tts import (  # noqa: E402
    SynthesizedUtterance,
    TTSClient,
)
from src.audiointeraction.hri_mot.streaming_tts_cli import (  # noqa: E402
    add_streaming_tts_args,
    build_tts_client_from_args,
    write_pcm16_wav,
)
from src.audiointeraction.hri_mot.streaming_wav_inference import infer_wav_impl  # noqa: E402
from src.audiointeraction.hri_mot.onset_condition_probe import OnsetConditionProbe  # noqa: E402
from src.audiointeraction.hri_mot.interaction_audio import (  # noqa: E402
    reconstruct_interaction_audio,
    reconstruct_realtime_interaction_audio,
)
from src.audiointeraction.hri_mot.motion_smoothing import (  # noqa: E402
    action_motion_stats,
    smooth_action_chunks_offline,
    stitch_action_chunk_c2,
)
from src.audiointeraction.hri_mot.realtime_profile import (  # noqa: E402
    build_realtime_profile_report,
)
from src.audiointeraction.hri_mot.video_postprocess import (  # noqa: E402
    DEFAULT_PAPER_RENDER_XML,
    add_video_postprocess_args,
    maybe_postprocess_prediction_video,
)
from scripts.hri_mot.train_hri_mot import (  # noqa: E402
    ActionStateNormalizer,
    encode_audio_feature_batch,
    load_trainable_state,
)
from src.audiointeraction.dataset.TOKENS import KEEP_SILENCE, TEXT_BEGIN  # noqa: E402
from src.audiointeraction.hri_mot.data_config import HriMotDataConfig  # noqa: E402
from src.audiointeraction.hri_mot.cospeech_episode_dataset import (  # noqa: E402
    _resample_matrix,
)
from utils import load_audio_encoder  # noqa: E402


DEFAULT_WAV = "examples/input.wav"
DEFAULT_BASE_YAW_DEG = 90.0


def summarize_profile_values(
    values: List[float], *, unit: str = "sec"
) -> Dict[str, Any]:
    if unit not in ("sec", "fps", "rtf", "x"):
        raise ValueError(f"Unsupported profile unit: {unit}")
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            f"mean_{unit}": None,
            f"p50_{unit}": None,
            f"p95_{unit}": None,
            f"p99_{unit}": None,
            f"max_{unit}": None,
        }
    return {
        "count": int(array.size),
        f"mean_{unit}": float(array.mean()),
        f"p50_{unit}": float(np.percentile(array, 50)),
        f"p95_{unit}": float(np.percentile(array, 95)),
        f"p99_{unit}": float(np.percentile(array, 99)),
        f"max_{unit}": float(array.max()),
    }


def parse_args(*, default_wav: str = DEFAULT_WAV) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer response text and Euler-sampled HRI actions directly from one WAV."
    )
    parser.add_argument("--wav", default=str(default_wav), help="Input WAV path.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Trained HRI-MoT checkpoint.")
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Directory containing args.json/norm_stats.json; defaults to checkpoint parent.",
    )
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
            "chunks; auto follows the generated semantic emotion tag."
        ),
    )
    parser.add_argument("--checkpoint-dir", default=None, help="Base AudioInteraction checkpoint directory.")
    parser.add_argument("--onset-condition-probe", action="store_true",
                        help="Fork matched six-emotion action rollouts at each speaking onset; normal inference stays auto.")
    parser.add_argument("--onset-probe-seconds", type=float, default=2.0,
                        help="Speaking seconds per paired rollout; must be a multiple of 0.4. Stops at interruptions.")
    parser.add_argument("--onset-probe-no-render", action="store_true",
                        help="Save paired actions only; run render_onset_condition_probe.py separately.")
    parser.add_argument("--norm-stats", default=None, help="Optional normalization stats JSON.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Defaults to <train-dir>/wav_inference/<wav-stem>; the realtime "
            "entry uses <train-dir>/wav_realtime_inference/<wav-stem>."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Language seed; defaults to the training checkpoint seed (usually 1337).",
    )
    parser.add_argument(
        "--debug-language-sampling",
        action="store_true",
        help="Print RNG/logit diagnostics for emotion-token sampling.",
    )
    parser.add_argument(
        "--language-sampling-mode",
        choices=("web_legacy", "independent"),
        default="web_legacy",
        help=(
            "web_legacy (default) reproduces web/infer_wav.py's native-dtype "
            "global-RNG language sampling; independent preserves the previous "
            "HRI private float32 RNG stream."
        ),
    )
    parser.add_argument("--max-total-seq-len", type=int, default=None)
    

    parser.add_argument(
        "--tail-silence-sec",
        type=float,
        default=0.0,
        help=(
            "Optional legacy zero audio that also generates actions. The default is 0; "
            "use --audio-eof-flush-sec for automatic Web-style EOF continuation."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Maximum total real+tail chunks; 0 processes all available chunks.",
    )
    parser.add_argument(
        "--stop-after-first-response",
        action="store_true",
        help=(
            "Stop after the first synthesized response finishes playback. By default the script preserves the language KV cache "
            "and continues through the remaining WAV chunks; PCM feature history restarts after each response."
        ),
    )
    parser.add_argument(
        "--profile-warmup-runs",
        type=int,
        default=0,
        help="Realtime-profile turns discarded before measured runs.",
    )
    parser.add_argument(
        "--profile-runs",
        type=int,
        default=1,
        help="Realtime-profile turns included in the latency distribution.",
    )

    parser.add_argument("--sampling-steps", type=int, default=10, help="Euler integration steps.")
    parser.add_argument("--initial-noise", choices=("random", "zeros"), default="random")
    parser.add_argument(
        "--state-source",
        choices=("rollout", "constant"),
        default="rollout",
        help="rollout feeds the previous C2-stitched action chunk back; constant uses the initial state window.",
    )
    parser.add_argument(
        "--initial-state-source",
        choices=("state_mean", "idle_pkl"),
        default="state_mean",
        help=(
            "Initial state used when --initial-state-npy is omitted. state_mean preserves the "
            "existing behavior; idle_pkl uses the idle motion recorded in the training arguments."
        ),
    )
    parser.add_argument(
        "--initial-state-npy",
        default=None,
        help=(
            "Optional float array shaped [20], [1,20], or [12,20]. "
            "This overrides --initial-state-source."
        ),
    )
    parser.add_argument(
        "--idle-motion-pkl",
        default=None,
        help=(
            "Optional idle motion PKL override for --initial-state-source idle_pkl. "
            "By default, use idle_motion_pkl from the checkpoint training arguments."
        ),
    )
    parser.add_argument(
        "--template-robot-pkl",
        default=None,
        help="Optional source robot pkl whose non-predicted qpos fields/layout should be preserved.",
    )
    parser.add_argument(
        "--default-base-yaw-deg",
        type=float,
        default=DEFAULT_BASE_YAW_DEG,
        help=(
            "Fixed chassis yaw written to a generated robot pkl when --template-robot-pkl is omitted. "
            "The default 90 degrees matches the retarget training-data convention."
        ),
    )

    parser.add_argument(
        "--max-response-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per response; 64 matches web/infer_wav.py.",
    )
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
    parser.add_argument(
        "--audio-feature-mode",
        choices=("streaming_latest", "independent_first"),
        default="streaming_latest",
        help="streaming_latest matches web/infer_wav.py feature extraction.",
    )
    parser.add_argument(
        "--audio-silence-policy",
        choices=("web_vad_zero", "none"),
        default="web_vad_zero",
        help=(
            "web_vad_zero keeps two real inactive tail chunks, then sends exact zeros while retaining "
            "one action per 400 ms chunk. none forwards the WAV chunks unchanged."
        ),
    )
    parser.add_argument(
        "--audio-energy-threshold",
        type=float,
        default=WEB_AUDIO_ENERGY_THRESHOLD,
        help="Short-window RMS threshold used by web_vad_zero.",
    )
    parser.add_argument(
        "--audio-trailing-real-chunks",
        type=int,
        default=WEB_AUDIO_TRAILING_REAL_CHUNKS,
        help="Inactive chunks retained as real audio before exact-zero substitution.",
    )
    parser.add_argument(
        "--audio-eof-flush-sec",
        type=float,
        default=WEB_AUDIO_SILENCE_SEC,
        help=(
            "Maximum exact-zero audio appended after the WAV while waiting for "
            "a deferred response. These physical chunks generate synchronized "
            "actions and extend the output pkl."
        ),
    )
    parser.add_argument("--no-audio-features", action="store_true")
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
        help="Zero-phase Butterworth cutoff for the saved action; 0 disables the offline low-pass filter.",
    )
    add_streaming_tts_args(parser)
    add_video_postprocess_args(
        parser,
        allow_gt_override=True,
        default_render_xml=DEFAULT_PAPER_RENDER_XML,
    )
    return parser.parse_args()


def load_wav_16k(path: Path) -> Tuple[np.ndarray, float]:
    """Decode/resample exactly like web/infer_wav.py, without enhancement."""
    audio = load_wav_like_web(path, sample_rate=16_000)
    return audio, float(audio.size / 16_000.0)


def capture_global_torch_rng_state(
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Capture the global RNG state used by legacy Web language sampling."""
    state = {"cpu": torch.get_rng_state().clone()}
    if device.type == "cuda":
        state["device"] = torch.cuda.get_rng_state(device).clone()
    return state


def restore_global_torch_rng_state(
    state: Dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    """Undo RNG consumed by HRI-only module construction before inference."""
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["device"], device)


def build_audio_chunks(
    audio: np.ndarray,
    *,
    samples_per_chunk: int,
    tail_silence_sec: float,
    max_chunks: int,
) -> Tuple[List[np.ndarray], int]:
    if tail_silence_sec < 0:
        raise ValueError(f"tail_silence_sec must be >= 0, got {tail_silence_sec}")
    chunks: List[np.ndarray] = []
    for start in range(0, int(audio.size), samples_per_chunk):
        chunk = audio[start : start + samples_per_chunk]
        if chunk.size < samples_per_chunk:
            chunk = np.pad(chunk, (0, samples_per_chunk - chunk.size))
        chunks.append(np.asarray(chunk, dtype=np.float32))
    real_chunks = len(chunks)
    tail_chunks = int(math.ceil(tail_silence_sec * 16_000 / samples_per_chunk - 1e-12))
    chunks.extend(np.zeros(samples_per_chunk, dtype=np.float32) for _ in range(tail_chunks))
    if max_chunks > 0:
        chunks = chunks[:max_chunks]
    return chunks, min(real_chunks, len(chunks))


def resolve_initial_state(
    *,
    path: Optional[str],
    source: str,
    idle_motion_path: Optional[str],
    idle_motion_duration_sec: float,
    normalizer: ActionStateNormalizer,
    data_config: HriMotDataConfig,
    steps: int,
    dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, str]:
    if path is not None:
        state_path = Path(path)
        if not state_path.is_file():
            raise FileNotFoundError(f"Initial state npy does not exist: {state_path}")
        array = np.asarray(np.load(state_path), dtype=np.float32)
        if array.shape == (dim,):
            array = np.repeat(array[None, :], steps, axis=0)
        elif array.shape == (1, dim):
            array = np.repeat(array, steps, axis=0)
        elif array.shape != (steps, dim):
            raise ValueError(
                f"Initial state must have shape {(dim,)}, {(1, dim)}, or {(steps, dim)}, got {array.shape}"
            )
        return torch.from_numpy(array).to(device=device), str(state_path)

    if source == "state_mean":
        state = normalizer.state_mean.detach().to(device=device, dtype=torch.float32)
        if tuple(state.shape) != (dim,):
            raise ValueError(f"state mean must have shape {(dim,)}, got {tuple(state.shape)}")
        return state.unsqueeze(0).repeat(steps, 1), "training_state_mean"

    if source != "idle_pkl":
        raise ValueError(f"Unknown initial state source: {source}")
    if not idle_motion_path:
        raise ValueError(
            "--initial-state-source idle_pkl requires idle_motion_pkl in the training "
            "arguments or an explicit --idle-motion-pkl"
        )

    idle_path = Path(idle_motion_path).expanduser().resolve()
    if not idle_path.is_file():
        raise FileNotFoundError(f"Idle motion pkl does not exist: {idle_path}")
    with open(idle_path, "rb") as fp:
        payload = pickle.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(f"Idle motion PKL must contain a dict: {idle_path}")

    qpos = np.asarray(payload.get(data_config.robot_source_key), dtype=np.float32)
    valid_indices = list(map(int, data_config.robot_valid_qpos_indices))
    max_valid_index = max(valid_indices) if valid_indices else None
    if (
        qpos.ndim != 2
        or max_valid_index is None
        or qpos.shape[1] <= max_valid_index
    ):
        raise ValueError(
            "Idle qpos must contain compact robot indices through "
            f"{max_valid_index}, got {qpos.shape}"
        )
    try:
        source_fps = float(payload.get("fps", data_config.robot_fps))
    except (TypeError, ValueError):
        source_fps = float(data_config.robot_fps)
    if not math.isfinite(source_fps) or source_fps <= 0.0:
        raise ValueError(f"Idle motion FPS must be positive, got {source_fps}")

    keep_frames = min(
        qpos.shape[0],
        max(1, int(round(float(idle_motion_duration_sec) * source_fps))),
    )
    idle = qpos[:keep_frames, valid_indices]
    idle = _resample_matrix(idle, source_fps, float(data_config.robot_fps))
    if idle.shape[1] != dim:
        raise ValueError(f"Idle selected qpos must have {dim} dims, got {idle.shape}")
    if idle.shape[0] < steps:
        idle = np.concatenate(
            [idle, np.repeat(idle[-1:, :], steps - idle.shape[0], axis=0)],
            axis=0,
        )
    initial = torch.from_numpy(idle[:steps].copy()).to(
        device=device, dtype=torch.float32
    )
    return initial, f"idle_pkl:{idle_path}"


def _control_probability(debug: Dict[str, Any], token_id: int) -> Optional[float]:
    value = debug.get("allowed_probs", {}).get(str(int(token_id)))
    return None if value is None else float(value)


def _format_probability(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def write_text_outputs(
    *,
    output_dir: Path,
    wav_path: Path,
    payload: Dict[str, Any],
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / f"{wav_path.stem}__response.txt"
    json_path = output_dir / f"{wav_path.stem}__response.json"
    response_texts = [
        str(record.get("planned_full_text", "")).strip()
        for record in payload["response_records"]
        if str(record.get("planned_full_text", "")).strip()
    ]
    if not response_texts:
        fragments = [
            str(record.get("text", "")).strip()
            for record in payload["response_records"]
            if record.get("control") in ("response", "speaking")
            and str(record.get("text", "")).strip()
        ]
        if fragments:
            response_texts = [" ".join(fragments)]
    text_path.write_text("\n".join(response_texts) + ("\n" if response_texts else "<KEEP_SILENCE>\n"), encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    return text_path, json_path


def write_model_input_wav(
    *,
    output_dir: Path,
    wav_path: Path,
    audio_chunks: List[np.ndarray],
    sample_rate: int,
) -> Path:
    """Save the exact waveform presented to the streaming audio encoder."""
    import soundfile as sf

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{wav_path.stem}__model_input.wav"
    waveform = np.concatenate(audio_chunks).astype(np.float32, copy=False)
    sf.write(output_path, waveform, int(sample_rate), subtype="FLOAT")
    return output_path


def apply_default_base_yaw_to_prediction_pkl(path: Path, *, yaw_deg: float) -> None:
    """Set a fixed planar-base yaw for standalone WAV predictions without a robot template."""
    if not math.isfinite(float(yaw_deg)):
        raise ValueError(f"default base yaw must be finite, got {yaw_deg}")

    with path.open("rb") as fp:
        motion = pickle.load(fp)
    if not isinstance(motion, dict):
        raise ValueError(f"Prediction pkl must contain a dict, got {type(motion).__name__}: {path}")

    raw_qpos = motion.get("qpos_list")
    qpos = np.asarray(raw_qpos)
    if qpos.ndim != 2 or qpos.shape[1] < 3:
        raise ValueError(f"qpos_list must have shape [frames, >=3], got {qpos.shape}: {path}")
    qpos = qpos.copy()
    yaw_rad = math.radians(float(yaw_deg))
    qpos[:, 2] = yaw_rad
    if isinstance(raw_qpos, list):
        motion["qpos_list"] = [frame.copy() for frame in qpos]
    else:
        motion["qpos_list"] = qpos

    raw_root_pos = motion.get("root_pos")
    root_pos = np.asarray(raw_root_pos) if raw_root_pos is not None else None
    if root_pos is None or root_pos.shape != (qpos.shape[0], 3):
        root_pos = np.zeros((qpos.shape[0], 3), dtype=qpos.dtype)
    else:
        root_pos = root_pos.copy()
    root_pos[:, :2] = qpos[:, :2]
    motion["root_pos"] = root_pos

    raw_root_rot = motion.get("root_rot")
    root_rot_dtype = np.asarray(raw_root_rot).dtype if raw_root_rot is not None else qpos.dtype
    root_rot = np.zeros((qpos.shape[0], 4), dtype=root_rot_dtype)
    root_rot[:, 2] = math.sin(yaw_rad * 0.5)
    root_rot[:, 3] = math.cos(yaw_rad * 0.5)
    motion["root_rot"] = root_rot

    prediction_metadata = dict(motion.get("hri_mot_prediction", {}))
    prediction_metadata["base_pose_source"] = "wav_default_fixed_yaw"
    prediction_metadata["default_base_yaw_deg"] = float(yaw_deg)
    motion["hri_mot_prediction"] = prediction_metadata
    with path.open("wb") as fp:
        pickle.dump(motion, fp)


@torch.no_grad()
def infer_wav(
    *,
    audio_chunks: List[np.ndarray],
    audio_frontend_modes: List[str],
    real_chunk_count: int,
    initial_state: torch.Tensor,
    wrapper: torch.nn.Module,
    tokenizer: Any,
    normalizer: ActionStateNormalizer,
    audio_encoder: Optional[torch.nn.Module],
    data_config: HriMotDataConfig,
    max_total_seq_len: int,
    args: argparse.Namespace,
    device: torch.device,
    language_generator: torch.Generator,
    action_generator: torch.Generator,
    tts_client: TTSClient,
    onset_probe: Optional[OnsetConditionProbe] = None,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], List[SynthesizedUtterance]]:
    return infer_wav_impl(
        audio_chunks=audio_chunks,
        audio_frontend_modes=audio_frontend_modes,
        real_chunk_count=real_chunk_count,
        initial_state=initial_state,
        wrapper=wrapper,
        tokenizer=tokenizer,
        normalizer=normalizer,
        audio_encoder=audio_encoder,
        data_config=data_config,
        max_total_seq_len=max_total_seq_len,
        args=args,
        device=device,
        language_generator=language_generator,
        action_generator=action_generator,
        tts_client=tts_client,
        session_class=HriMotStreamingSession,
        sample_action_fn=sample_action_chunk,
        encode_audio_feature_batch_fn=encode_audio_feature_batch,
        control_name_fn=_control_name,
        emotion_token_to_text=EMOTION_EMOJI,
        control_probability_fn=_control_probability,
        format_probability_fn=_format_probability,
        realtime_profile=bool(getattr(args, "realtime_profile", False)),
        onset_probe=onset_probe,
    )



def main(
    *,
    realtime_profile: bool = False,
    default_wav: str = DEFAULT_WAV,
) -> None:
    process_started = time.perf_counter()
    args = parse_args(default_wav=default_wav)
    args.realtime_profile = bool(realtime_profile)
    if args.profile_warmup_runs < 0:
        raise ValueError("--profile-warmup-runs must be >= 0")
    if args.profile_runs <= 0:
        raise ValueError("--profile-runs must be positive")
    if not realtime_profile and (
        args.profile_warmup_runs != 0 or args.profile_runs != 1
    ):
        raise ValueError(
            "--profile-warmup-runs/--profile-runs require "
            "infer_hri_mot_wav_realtime.py"
        )
    if args.onset_condition_probe:
        if realtime_profile or args.state_source != "rollout" or args.emotion_override != "auto":
            raise ValueError("Onset probe requires offline WAV inference, --state-source rollout and --emotion-override auto")
        if args.template_robot_pkl:
            raise ValueError("Onset probe rendering currently uses the standalone WAV fixed-base robot, without --template-robot-pkl")
        if not math.isfinite(args.onset_probe_seconds) or args.onset_probe_seconds <= 0 or not math.isclose(args.onset_probe_seconds / .4, round(args.onset_probe_seconds / .4), abs_tol=1e-6):
            raise ValueError("--onset-probe-seconds must be a positive multiple of 0.4")
    wav_path = Path(args.wav).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    train_dir = Path(args.train_dir).resolve() if args.train_dir else checkpoint.parent
    file_train_args = load_train_args(train_dir)
    train_args, speak_conditioning_source = resolve_inference_train_args(
        checkpoint=checkpoint,
        file_train_args=file_train_args,
        requested_speak_conditioning=args.speak_conditioning,
    )
    checkpoint_dir = args.checkpoint_dir or train_args.get("checkpoint_dir") or "checkpoints"
    seed = int(args.seed if args.seed is not None else train_args.get("seed", 1337))
    max_total_seq_len = int(args.max_total_seq_len or train_args.get("max_total_seq_len", 4096))
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else train_dir
        / ("wav_realtime_inference" if realtime_profile else "wav_inference")
        / wav_path.stem
    )
    if args.onset_condition_probe and not args.output_dir:
        output_dir = train_dir / "onset_condition_probe" / f"{wav_path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    if args.onset_condition_probe and (output_dir / "onset_probe").exists():
        raise FileExistsError(f"Choose a fresh --output-dir; probe already exists: {output_dir / 'onset_probe'}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")
    device = resolve_device(args.device)
    language_rng_seed = seed
    action_rng_seed = seed + 1
    language_generator = torch.Generator(device=device).manual_seed(language_rng_seed)
    action_generator = torch.Generator(device=device).manual_seed(action_rng_seed)

    data_config = HriMotDataConfig(text_chunk_time_mode=str(train_args.get("text_chunk_time_mode", "start")))
    samples_per_chunk = int(round(data_config.chunk_size_sec * data_config.audio_sample_rate))
    audio, source_duration_sec = load_wav_16k(wav_path)
    audio_chunks, real_chunk_count = build_audio_chunks(
        audio,
        samples_per_chunk=samples_per_chunk,
        tail_silence_sec=float(args.tail_silence_sec),
        max_chunks=int(args.max_chunks),
    )
    audio_chunk_tensor = torch.from_numpy(np.stack(audio_chunks, axis=0))
    model_audio_tensor, audio_frontend_modes, audio_silence_stats = apply_streaming_audio_silence_policy(
        audio_chunk_tensor,
        policy=args.audio_silence_policy,
        sample_rate=data_config.audio_sample_rate,
        energy_threshold=args.audio_energy_threshold,
        trailing_real_chunks=args.audio_trailing_real_chunks,
    )
    audio_chunks = [
        model_audio_tensor[idx].detach().cpu().numpy().astype(np.float32, copy=False)
        for idx in range(int(model_audio_tensor.shape[0]))
    ]

    norm_stats = load_inference_norm_stats(
        checkpoint=checkpoint,
        train_dir=train_dir,
        norm_stats_path=args.norm_stats,
    )
    normalizer = ActionStateNormalizer(norm_stats, device=device)
    initial_state, initial_state_source = resolve_initial_state(
        path=args.initial_state_npy,
        source=args.initial_state_source,
        idle_motion_path=args.idle_motion_pkl or train_args.get("idle_motion_pkl"),
        idle_motion_duration_sec=float(
            train_args.get("idle_motion_duration_sec", 9.0)
        ),
        normalizer=normalizer,
        data_config=data_config,
        steps=data_config.robot_steps_per_chunk,
        dim=data_config.robot_state_dim,
        device=device,
    )

    print(f"[config] wav: {wav_path}", flush=True)
    print(f"[config] source duration: {source_duration_sec:.3f}s", flush=True)
    print(
        f"[config] chunks: total={len(audio_chunks)} real={real_chunk_count} "
        f"tail={len(audio_chunks) - real_chunk_count}",
        flush=True,
    )
    print(
        f"[config] audio silence policy: {args.audio_silence_policy} "
        f"energy_threshold={args.audio_energy_threshold} "
        f"trailing_real_chunks={args.audio_trailing_real_chunks}",
        flush=True,
    )
    print(
        f"[config] audio frontend chunks: activity={audio_silence_stats['activity_chunks']} "
        f"real_tail={audio_silence_stats['real_tail_chunks']} "
        f"zero={audio_silence_stats['zero_chunks']}",
        flush=True,
    )
    print(
        f"[config] language sampling: temperature={args.text_temperature} "
        f"top_k={args.text_top_k} top_p={args.text_top_p} "
        f"silence_gate={args.min_reply_silence_sec}s",
        flush=True,
    )
    print(
        f"[config] RNG: language_mode={args.language_sampling_mode} "
        f"language_seed={language_rng_seed} action_seed={action_rng_seed}",
        flush=True,
    )
    print(
        f"[config] EOF zero continuation: max={args.audio_eof_flush_sec}s "
        "(generates synchronized actions and TTS)",
        flush=True,
    )
    print(
        f"[config] barge-in: mode={args.barge_in_mode} "
        f"min_reply_silence={args.min_reply_silence_sec}s "
        f"deferred_reply_prob_threshold={args.deferred_reply_prob_threshold}",
        flush=True,
    )
    print(f"[config] action solver: euler steps={args.sampling_steps}", flush=True)
    print(f"[config] initial state: {initial_state_source}", flush=True)
    print(
        f"[config] offline motion smoothing: c2_bridge_frames={args.c2_bridge_frames} "
        f"lowpass_cutoff_hz={args.offline_lowpass_cutoff_hz}",
        flush=True,
    )
    print(
        f"[config] video postprocess: enabled={not args.skip_video_postprocess} "
        f"render_script={args.render_script} render_xml={args.render_xml_path} "
        f"gt_root={args.gt_video_root} "
        f"layout=prediction_left_gt_right",
        flush=True,
    )
    print(
        f"[config] realtime wall-clock profile: enabled={realtime_profile} "
        f"input_chunk_period={data_config.chunk_size_sec}s",
        flush=True,
    )
    print(f"[config] output dir: {output_dir}", flush=True)
    print(
        f"[config] speak conditioning: requested={args.speak_conditioning} "
        f"resolved={train_args['speak_conditioning']} source={speak_conditioning_source}",
        flush=True,
    )
    print(
        f"[config] emotion override: {args.emotion_override}",
        flush=True,
    )
    tts_client = build_tts_client_from_args(args)
    print(
        f"[config] TTS: mode={args.tts_mode} url={args.tts_url} "
        f"language={args.tts_language} speaker={args.tts_speaker}",
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
    # This is the point where web/infer_wav.py has finished constructing the
    # base model and audio encoder. Its next stochastic operations are language
    # samples. Save that exact global state before HRI-only modules are built.
    web_legacy_rng_state = capture_global_torch_rng_state(device)
    wrapper = build_wrapper(
        base_model=base_model,
        data_config=data_config,
        train_args=train_args,
        device=device,
    )
    checkpoint_step, missing = load_trainable_state(wrapper, str(checkpoint), device)
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
    print(
        f"[config] checkpoint step={checkpoint_step} missing_trainable_keys={missing} "
        f"audio_features={use_audio_features} "
        f"mode={args.audio_feature_mode if use_audio_features else 'none'} "
        f"emotion={wrapper.use_emotion_conditioning} "
        f"emotion_film_layers="
        f"{wrapper.emotion_film_layers}",
        flush=True,
    )
    if args.language_sampling_mode == "web_legacy":
        restore_global_torch_rng_state(web_legacy_rng_state, device)
        print(
            "[config] restored post-base/audio global RNG for Web-compatible "
            "language sampling",
            flush=True,
        )

    onset_probe = None
    if args.onset_condition_probe:
        if not wrapper.use_emotion_conditioning:
            raise ValueError("Onset probe requires a checkpoint with six-emotion conditioning")
        onset_probe = OnsetConditionProbe(
            output_dir / "onset_probe", seconds=args.onset_probe_seconds,
            chunk_seconds=data_config.chunk_size_sec, fps=data_config.robot_fps,
            qpos_indices=data_config.robot_valid_qpos_indices,
            bridge_frames=args.c2_bridge_frames, base_yaw_deg=args.default_base_yaw_deg,
            metadata=dict(checkpoint=str(checkpoint), checkpoint_step=int(checkpoint_step),
                          wav=str(wav_path), language_seed=language_rng_seed,
                          action_seed=action_rng_seed, sampling_steps=args.sampling_steps,
                          initial_noise=args.initial_noise, initial_state_source=initial_state_source),
        )
    if realtime_profile and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    initialization_sec = float(time.perf_counter() - process_started)
    profile_run_reports: List[Dict[str, Any]] = []
    total_profile_runs = (
        args.profile_warmup_runs + args.profile_runs
        if realtime_profile
        else 1
    )
    pred_action_raw = None
    response_records = []
    tts_utterances = []
    for profile_run_idx in range(total_profile_runs):
        if total_profile_runs > 1:
            run_kind = (
                "warmup"
                if profile_run_idx < args.profile_warmup_runs
                else "measured"
            )
            print(
                f"[profile] {run_kind} run {profile_run_idx + 1}/"
                f"{total_profile_runs}",
                flush=True,
            )
        if args.language_sampling_mode == "web_legacy":
            restore_global_torch_rng_state(web_legacy_rng_state, device)
        run_language_generator = torch.Generator(device=device).manual_seed(
            language_rng_seed
        )
        run_action_generator = torch.Generator(device=device).manual_seed(
            action_rng_seed
        )
        pred_action_raw, response_records, tts_utterances = infer_wav(
            audio_chunks=audio_chunks,
            audio_frontend_modes=audio_frontend_modes,
            real_chunk_count=real_chunk_count,
            initial_state=initial_state,
            wrapper=wrapper,
            tokenizer=tokenizer,
            normalizer=normalizer,
            audio_encoder=audio_encoder,
            data_config=data_config,
            max_total_seq_len=max_total_seq_len,
            args=args,
            device=device,
            language_generator=run_language_generator,
            action_generator=run_action_generator,
            tts_client=tts_client,
            onset_probe=onset_probe,
        )
        if realtime_profile:
            run_report = build_realtime_profile_report(
                response_records=response_records,
                source_duration_sec=source_duration_sec,
                chunk_size_sec=data_config.chunk_size_sec,
                motion_fps=float(data_config.robot_fps),
            )
            run_report["run_index"] = int(profile_run_idx)
            run_report["warmup"] = bool(
                profile_run_idx < args.profile_warmup_runs
            )
            profile_run_reports.append(run_report)
    assert pred_action_raw is not None
    raw_boundary = action_boundary_stats(pred_action_raw)
    raw_motion = action_motion_stats(pred_action_raw)
    pred_action = smooth_action_chunks_offline(
        pred_action_raw,
        bridge_frames=int(args.c2_bridge_frames),
        lowpass_cutoff_hz=float(args.offline_lowpass_cutoff_hz),
        fps=float(data_config.robot_fps),
    )
    smoothed_boundary = action_boundary_stats(pred_action)
    smoothed_motion = action_motion_stats(pred_action)

    processed_chunks = int(pred_action.shape[0])
    audio_t_end = float(processed_chunks * data_config.chunk_size_sec)
    action_t_start = float(data_config.chunk_size_sec)
    action_t_end = float(audio_t_end + data_config.chunk_size_sec)
    template_robot_pkl = None
    if args.template_robot_pkl:
        template_path = Path(args.template_robot_pkl).resolve()
        if not template_path.is_file():
            raise FileNotFoundError(f"Template robot pkl does not exist: {template_path}")
        template_robot_pkl = str(template_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tts_wav_paths: List[str] = []
    if not args.no_tts_save_audio:
        for utterance_index, utterance in enumerate(tts_utterances):
            tts_path = output_dir / f"tts_response_{utterance_index:03d}.wav"
            write_pcm16_wav(tts_path, utterance.pcm, utterance.sample_rate)
            tts_wav_paths.append(str(tts_path))
    for record in response_records:
        utterance_index = record.get("tts_utterance_index")
        if utterance_index is not None and int(utterance_index) < len(tts_wav_paths):
            record["tts_wav"] = tts_wav_paths[int(utterance_index)]

    interaction_pcm, interaction_audio_meta = reconstruct_interaction_audio(
        partner_pcm=audio,
        sample_rate=data_config.audio_sample_rate,
        duration_sec=action_t_end,
        response_records=response_records,
        tts_utterances=tts_utterances,
    )
    interaction_audio_path = output_dir / f"{wav_path.stem}__interaction.wav"
    write_pcm16_wav(
        interaction_audio_path,
        interaction_pcm,
        data_config.audio_sample_rate,
    )
    interaction_audio_meta.update(
        {
            "wav": str(interaction_audio_path),
            "global_t_start": 0.0,
            "global_t_end": action_t_end,
            "video_audio_start_sec": action_t_start,
        }
    )
    realtime_profile_report = None
    realtime_profile_path = None
    realtime_interaction_audio_path = None
    realtime_interaction_audio_meta = None
    if realtime_profile:
        realtime_profile_report = profile_run_reports[-1]
        realtime_profile_report["system_prompt"] = DEFAULT_SYSTEM_PROMPT
        realtime_profile_report["system_prompt_source"] = (
            "scripts.hri_mot.train_hri_mot.DEFAULT_SYSTEM_PROMPT"
        )
        realtime_profile_report["initialization_sec"] = initialization_sec
        first_turn_response = next(
            (
                item
                for item in profile_run_reports[0]["responses"]
                if item.get("reply_action_latency_from_input_end_sec") is not None
            ),
            None,
        )
        measured_reply_latencies = []
        for run_report in profile_run_reports[args.profile_warmup_runs :]:
            response = next(
                (
                    item
                    for item in run_report["responses"]
                    if item.get("reply_action_latency_from_input_end_sec")
                    is not None
                ),
                None,
            )
            if response is not None:
                measured_reply_latencies.append(
                    float(response["reply_action_latency_from_input_end_sec"])
                )
        steady_reply_latency = (
            float(np.median(measured_reply_latencies))
            if measured_reply_latencies
            else None
        )
        measured_action_sampling_fps = [
            float(run_report["motion_throughput"]["action_sampling_fps"])
            for run_report in profile_run_reports[args.profile_warmup_runs :]
            if run_report["motion_throughput"]["action_sampling_fps"]
            is not None
        ]
        measured_audio_to_motion_fps = [
            float(run_report["motion_throughput"]["audio_to_motion_pipeline_fps"])
            for run_report in profile_run_reports[args.profile_warmup_runs :]
            if run_report["motion_throughput"]["audio_to_motion_pipeline_fps"]
            is not None
        ]
        measured_interaction_rtfs = [
            float(run_report["interaction_compute_rtf"])
            for run_report in profile_run_reports[args.profile_warmup_runs :]
            if run_report["interaction_compute_rtf"] is not None
        ]
        measured_interaction_compute_sec = sum(
            float(run_report["interaction_compute_sec"])
            for run_report in profile_run_reports[args.profile_warmup_runs :]
        )
        measured_interaction_timeline_sec = sum(
            float(run_report["interaction_timeline_sec"])
            for run_report in profile_run_reports[args.profile_warmup_runs :]
        )
        aggregate_interaction_rtf = (
            measured_interaction_compute_sec
            / measured_interaction_timeline_sec
            if measured_interaction_timeline_sec > 0.0
            else None
        )
        cold_turn_latency = (
            float(first_turn_response["reply_action_latency_from_input_end_sec"])
            if first_turn_response is not None
            else None
        )
        realtime_profile_report["primary_metric"] = {
            "name": "input_end_to_first_reply_action_ready",
            "steady_state_median_sec": steady_reply_latency,
            "steady_state_runs_sec": measured_reply_latencies,
            "steady_state": summarize_profile_values(
                measured_reply_latencies
            ),
            "warmup_runs": int(args.profile_warmup_runs),
            "measured_runs": int(args.profile_runs),
            "cold_start_sec": (
                initialization_sec + cold_turn_latency
                if cold_turn_latency is not None
                else None
            ),
            "cold_start_includes": (
                "WAV/config preparation, TTS client setup, base/audio/action "
                "model loading, then paced streaming through input end"
            ),
            "excluded": (
                "offline smoothing, PKL/WAV writes, rendering, video muxing, "
                "network/robot transport"
            ),
        }
        realtime_profile_report["motion_throughput_benchmark"] = {
            "action_sampling_fps": summarize_profile_values(
                measured_action_sampling_fps, unit="fps"
            ),
            "audio_to_motion_pipeline_fps": summarize_profile_values(
                measured_audio_to_motion_fps, unit="fps"
            ),
            "action_sampling_runs_fps": measured_action_sampling_fps,
            "audio_to_motion_pipeline_runs_fps": (
                measured_audio_to_motion_fps
            ),
        }
        realtime_profile_report["interaction_rtf_benchmark"] = {
            "definition": (
                "summed live-loop compute seconds divided by generated "
                "interaction-motion timeline seconds; paced waits excluded"
            ),
            "aggregate_rtf": aggregate_interaction_rtf,
            "aggregate_realtime_speedup": (
                1.0 / aggregate_interaction_rtf
                if aggregate_interaction_rtf is not None
                and aggregate_interaction_rtf > 0.0
                else None
            ),
            "distribution": summarize_profile_values(
                measured_interaction_rtfs, unit="rtf"
            ),
            "runs_rtf": measured_interaction_rtfs,
            "total_compute_sec": measured_interaction_compute_sec,
            "total_timeline_sec": measured_interaction_timeline_sec,
            "warmup_runs": int(args.profile_warmup_runs),
            "measured_runs": int(args.profile_runs),
        }
        (
            realtime_interaction_pcm,
            realtime_interaction_audio_meta,
        ) = reconstruct_realtime_interaction_audio(
            partner_pcm=audio,
            sample_rate=data_config.audio_sample_rate,
            minimum_duration_sec=max(
                source_duration_sec,
                float(realtime_profile_report["wall_clock_elapsed_sec"]),
            ),
            response_records=response_records,
            tts_utterances=tts_utterances,
        )
        realtime_interaction_audio_path = (
            output_dir / f"{wav_path.stem}__realtime_interaction.wav"
        )
        write_pcm16_wav(
            realtime_interaction_audio_path,
            realtime_interaction_pcm,
            data_config.audio_sample_rate,
        )
        realtime_interaction_audio_meta["wav"] = str(
            realtime_interaction_audio_path
        )
        realtime_profile_report["interaction_audio"] = (
            realtime_interaction_audio_meta
        )
        realtime_profile_path = (
            output_dir / f"{wav_path.stem}__realtime_profile.json"
        )
        with realtime_profile_path.open("w", encoding="utf-8") as fp:
            json.dump(
                realtime_profile_report,
                fp,
                indent=2,
                ensure_ascii=False,
            )

    model_audio_chunks_for_output = list(audio_chunks)
    extra_playback_chunks = max(0, processed_chunks - len(model_audio_chunks_for_output))
    tts_playback_tail_chunks = int(
        sum(
            str(record.get("audio_kind", "")) == "tts_playback_tail"
            for record in response_records
        )
    )
    eof_action_flush_chunks = int(
        sum(
            str(record.get("audio_kind", "")) == "eof_action_flush"
            for record in response_records
        )
    )
    model_audio_chunks_for_output.extend(
        np.zeros(samples_per_chunk, dtype=np.float32)
        for _ in range(extra_playback_chunks)
    )
    model_input_wav = write_model_input_wav(
        output_dir=output_dir,
        wav_path=wav_path,
        audio_chunks=model_audio_chunks_for_output,
        sample_rate=data_config.audio_sample_rate,
    )
    action_path = write_prediction_pkl(
        original_pkl=template_robot_pkl,
        output_root=output_dir,
        episode_id=wav_path.stem,
        audio_t_start=0.0,
        audio_t_end=audio_t_end,
        action_t_start=action_t_start,
        action_t_end=action_t_end,
        pred_action=pred_action.detach().cpu().numpy(),
        data_config=data_config,
    )
    if template_robot_pkl is None:
        apply_default_base_yaw_to_prediction_pkl(
            action_path,
            yaw_deg=float(args.default_base_yaw_deg),
        )
        print(
            f"[output] no robot template: fixed base yaw to {float(args.default_base_yaw_deg):.3f} deg",
            flush=True,
        )

    video_outputs = maybe_postprocess_prediction_video(
        skip=bool(args.skip_video_postprocess),
        strict=bool(args.video_postprocess_strict),
        prediction_pkl=action_path,
        action_t_start=action_t_start,
        source_robot_pkl=template_robot_pkl,
        render_script=Path(args.render_script),
        render_conda_env=str(args.render_conda_env),
        render_xml_path=(Path(args.render_xml_path) if args.render_xml_path else None),
        gt_video_root=Path(args.gt_video_root),
        explicit_gt_video=args.gt_video,
        interaction_audio=interaction_audio_path,
        interaction_audio_start_sec=action_t_start,
        interaction_audio_codec=str(args.interaction_audio_codec),
    )

    tts_outputs = [
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
    ]
    payload: Dict[str, Any] = {
        "wav": str(wav_path),
        "source_duration_sec": source_duration_sec,
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(checkpoint_step),
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "system_prompt_source": (
            "scripts.hri_mot.train_hri_mot.DEFAULT_SYSTEM_PROMPT"
        ),
        "speak_conditioning": str(wrapper.speak_conditioning),
        "speak_conditioning_requested": str(args.speak_conditioning),
        "speak_conditioning_source": str(speak_conditioning_source),
        "emotion_override": str(args.emotion_override),
        **emotion_inference_config(wrapper),
        "tts_mode": str(args.tts_mode),
        "tts_url": str(args.tts_url),
        "tts_language": str(args.tts_language),
        "tts_speaker": str(args.tts_speaker),
        "tts_instruct": str(args.tts_instruct),
        "tts_playback_tail_chunks": tts_playback_tail_chunks,
        "tts_outputs": tts_outputs,
        "barge_in_mode": str(args.barge_in_mode),
        "min_reply_silence_sec": float(args.min_reply_silence_sec),
        "deferred_reply_prob_threshold": float(
            args.deferred_reply_prob_threshold
        ),
        "barge_in_count": int(
            sum(
                bool(record.get("response_interrupted", False))
                for record in response_records
            )
        ),
        "silence_gate": True,
        "silence_gate_policy": "defer_text_begin_until_vad_silence",
        "audio_silence_policy": args.audio_silence_policy,
        "audio_energy_threshold": float(args.audio_energy_threshold),
        "audio_trailing_real_chunks": int(args.audio_trailing_real_chunks),
        "audio_eof_flush_sec": float(args.audio_eof_flush_sec),
        "audio_silence_stats": audio_silence_stats,
        "model_input_wav": str(model_input_wav),
        "interaction_audio_wav": str(interaction_audio_path),
        "interaction_audio": interaction_audio_meta,
        "realtime_profile_enabled": bool(realtime_profile),
        "realtime_profile_json": (
            str(realtime_profile_path)
            if realtime_profile_path is not None
            else None
        ),
        "realtime_profile_summary": (
            {
                key: value
                for key, value in realtime_profile_report.items()
                if key not in ("chunks", "interaction_audio")
            }
            if realtime_profile_report is not None
            else None
        ),
        "realtime_interaction_audio_wav": (
            str(realtime_interaction_audio_path)
            if realtime_interaction_audio_path is not None
            else None
        ),
        "realtime_interaction_audio": realtime_interaction_audio_meta,
        "template_robot_pkl": template_robot_pkl,
        "default_base_yaw_deg": float(args.default_base_yaw_deg) if template_robot_pkl is None else None,
        "audio_feature_mode": args.audio_feature_mode if use_audio_features else "none",
        "tail_silence_sec": float(args.tail_silence_sec),
        "text_temperature": float(args.text_temperature),
        "text_top_k": int(args.text_top_k),
        "text_top_p": float(args.text_top_p),
        "language_sampling_mode": str(args.language_sampling_mode),
        "language_logits_dtype_policy": (
            "native" if args.language_sampling_mode == "web_legacy" else "float32"
        ),
        "independent_rng_streams": bool(
            args.language_sampling_mode == "independent"
        ),
        "language_rng_seed": int(language_rng_seed),
        "action_rng_seed": int(action_rng_seed),
        "processed_chunks": processed_chunks,
        "language_input_chunks": int(len(response_records)),
        "eof_language_flush_chunks": int(
            sum(bool(record.get("language_only_flush", False)) for record in response_records)
        ),
        "eof_action_flush_chunks": eof_action_flush_chunks,
        "processed_audio_t_end": audio_t_end,
        "action_solver": "euler",
        "action_sampling_steps": int(args.sampling_steps),
        "state_source": args.state_source,
        "initial_state_selection": args.initial_state_source,
        "initial_state_source": initial_state_source,
        "initial_noise": args.initial_noise,
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
        "gt_video_override": str(Path(args.gt_video).expanduser().resolve()) if args.gt_video else None,
        "comparison_layout": "prediction_left_gt_right",
        "raw_boundary_stats": raw_boundary,
        "smoothed_boundary_stats": smoothed_boundary,
        "raw_motion_stats": raw_motion,
        "smoothed_motion_stats": smoothed_motion,
        "action_pkl": str(action_path),
        "video_outputs": video_outputs,
        "onset_condition_probe": str(onset_probe.output_dir / "manifest.json") if onset_probe is not None else None,
        "response_records": response_records,
    }
    text_path, json_path = write_text_outputs(
        output_dir=output_dir,
        wav_path=wav_path,
        payload=payload,
    )
    if onset_probe is not None:
        print(f"[onset-probe] manifest: {onset_probe.output_dir / 'manifest.json'}", flush=True)
        if not args.onset_probe_no_render:
            render_python = Path(sys.executable).resolve().parents[2] / args.render_conda_env / "bin/python"
            if not render_python.is_file():
                raise FileNotFoundError(f"Probe data saved. Render environment Python is missing: {render_python}")
            render_env = os.environ.copy()
            render_env["LD_LIBRARY_PATH"] = str(render_python.parent.parent / "lib") + os.pathsep + render_env.get("LD_LIBRARY_PATH", "")
            subprocess.run([
                str(render_python), str(ROOT / "scripts/hri_mot/render_onset_condition_probe.py"),
                "--probe-dir", str(onset_probe.output_dir), "--xml", str(args.render_xml_path or DEFAULT_PAPER_RENDER_XML),
                "--software",
            ], env=render_env, check=True)
    response_count = sum(record["control"] == "response" for record in response_records)
    print(f"[done] responses: {response_count}", flush=True)
    print(f"[done] text: {text_path}", flush=True)
    print(f"[done] details: {json_path}", flush=True)
    print(f"[done] model input wav: {model_input_wav}", flush=True)
    print(f"[done] full interaction wav: {interaction_audio_path}", flush=True)
    if realtime_interaction_audio_path is not None:
        print(
            f"[done] realtime interaction wav: "
            f"{realtime_interaction_audio_path}",
            flush=True,
        )
    if realtime_profile_path is not None:
        print(
            f"[done] realtime profile: {realtime_profile_path}",
            flush=True,
        )
        print(
            f"[done] deadline misses: "
            f"{realtime_profile_report['deadline_miss_count']}/"
            f"{realtime_profile_report['processed_chunks']} "
            f"({realtime_profile_report['deadline_miss_rate']:.1%})",
            flush=True,
        )
        print(
            f"[done] pipeline p95: "
            f"{realtime_profile_report['pipeline_compute']['p95_sec']:.3f}s "
            f"action-ready p95: "
            f"{realtime_profile_report['action_ready_latency']['p95_sec']:.3f}s",
            flush=True,
        )
        for response_profile in realtime_profile_report["responses"]:
            first_phrase_sec = response_profile.get(
                "first_phrase_decode_tts_and_control_sec"
            )
            if first_phrase_sec is not None:
                stage_summary = (
                    f"first_phrase_decode_tts_and_control="
                    f"{float(first_phrase_sec):.3f}s"
                )
            else:
                stage_summary = (
                    f"decode={response_profile['text_decode_sec']:.3f}s "
                    f"tts={response_profile['tts_synthesis_sec']:.3f}s"
                )
            print(
                f"[done] response {response_profile['utterance_index']}: "
                f"{stage_summary} "
                f"wall_start={response_profile['wall_playback_t_start']:.3f}s "
                f"from_last_activity_chunk="
                f"{response_profile['wall_latency_from_last_activity_chunk_end_sec']}",
                flush=True,
            )
        primary_metric = realtime_profile_report["primary_metric"]
        print(
            "[done] input-end -> first reply action: "
            f"steady_median={primary_metric['steady_state_median_sec']}s "
            f"cold={primary_metric['cold_start_sec']}s "
            f"initialization={initialization_sec:.3f}s",
            flush=True,
        )
        throughput = realtime_profile_report["motion_throughput"]
        print(
            "[done] motion throughput (400ms pacing excluded): "
            f"action_sampling={throughput['action_sampling_fps']:.2f} FPS "
            f"audio_to_motion={throughput['audio_to_motion_pipeline_fps']:.2f} FPS",
            flush=True,
        )
        interaction_rtf = realtime_profile_report["interaction_rtf_benchmark"]
        print(
            "[done] interaction compute RTF (400ms pacing excluded): "
            f"aggregate={interaction_rtf['aggregate_rtf']:.4f} "
            f"p50={interaction_rtf['distribution']['p50_rtf']:.4f} "
            f"p95={interaction_rtf['distribution']['p95_rtf']:.4f} "
            f"speedup={interaction_rtf['aggregate_realtime_speedup']:.2f}x",
            flush=True,
        )
    print(f"[done] action pkl: {action_path}", flush=True)
    for tts_wav in tts_wav_paths:
        print(f"[done] TTS wav: {tts_wav}", flush=True)
    if video_outputs.get("comparison_video"):
        print(f"[done] comparison video: {video_outputs['comparison_video']}", flush=True)
    elif video_outputs.get("prediction_video"):
        print(f"[done] prediction video: {video_outputs['prediction_video']}", flush=True)


if __name__ == "__main__":
    main()

# Example: python scripts/hri_mot/infer_hri_mot_wav.py --wav examples/input.wav
