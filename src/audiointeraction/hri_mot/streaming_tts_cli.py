"""CLI helpers shared by the two HRI streaming inference entry points."""

from __future__ import annotations

import argparse
from pathlib import Path
import wave

import numpy as np

from .streaming_tts import EstimatedTTSClient, HttpTTSClient, TTSClient


def add_streaming_tts_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("streaming robot TTS")
    group.add_argument(
        "--tts-mode",
        choices=("http", "estimated"),
        default="http",
        help=(
            "http uses an isolated HRI-compatible TTS service (Kokoro by default) "
            "and real synthesized PCM; "
            "estimated is an explicit no-audio timing fallback for tests."
        ),
    )
    group.add_argument("--tts-url", default="http://127.0.0.1:8765")
    group.add_argument("--tts-timeout-sec", type=float, default=120.0)
    group.add_argument("--tts-language", default="English")
    group.add_argument(
        "--tts-speaker",
        default="",
        help=(
            "Optional per-inference voice override. By default the speaker is "
            "selected only by the TTS server's --default-speaker argument."
        ),
    )
    group.add_argument(
        "--tts-instruct",
        default="",
        help="Optional backend instruction. Kokoro accepts this field but ignores it.",
    )
    group.add_argument("--tts-estimated-units-per-second", type=float, default=4.5)
    group.add_argument(
        "--no-tts-save-audio",
        action="store_true",
        help="Do not save synthesized robot speech WAV files beside inference outputs.",
    )
    turn_group = parser.add_argument_group("streaming turn taking")
    turn_group.add_argument(
        "--barge-in-mode",
        choices=("none", "immediate"),
        default="immediate",
        help=(
            "immediate stops active robot speech at the first new partner-activity "
            "chunk, then waits for VAD silence before allowing another response."
        ),
    )
    turn_group.add_argument(
        "--min-reply-silence-sec",
        type=float,
        default=0.8,
        help=(
            "Minimum VAD-inactive duration before releasing a deferred "
            "TEXT_BEGIN. The 0.8 second default matches web/infer_wav.py."
        ),
    )
    turn_group.add_argument(
        "--deferred-reply-prob-threshold",
        type=float,
        default=0.5,
        help=(
            "While a barge-in is waiting for silence, preserve a reply as pending "
            "when the model's TEXT_BEGIN probability reaches this threshold."
        ),
    )


def validate_streaming_tts_args(args: argparse.Namespace) -> None:
    if float(args.tts_timeout_sec) <= 0:
        raise ValueError("--tts-timeout-sec must be positive")
    if float(args.tts_estimated_units_per_second) <= 0:
        raise ValueError("--tts-estimated-units-per-second must be positive")
    if float(args.min_reply_silence_sec) < 0:
        raise ValueError("--min-reply-silence-sec must be non-negative")
    if not 0.0 <= float(args.deferred_reply_prob_threshold) <= 1.0:
        raise ValueError("--deferred-reply-prob-threshold must be in [0, 1]")


def build_tts_client_from_args(args: argparse.Namespace) -> TTSClient:
    validate_streaming_tts_args(args)
    mode = str(args.tts_mode)
    if mode == "http":
        client = HttpTTSClient(
            str(args.tts_url),
            timeout_sec=float(args.tts_timeout_sec),
        )
        health = client.health()
        if not bool(health.get("ok")):
            raise RuntimeError(f"TTS service is not healthy: {health}")
        return client
    if mode == "estimated":
        return EstimatedTTSClient(
            units_per_second=float(args.tts_estimated_units_per_second)
        )
    raise ValueError(f"Unknown tts_mode={mode!r}")


def write_pcm16_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(pcm, dtype=np.float32).reshape(-1), -1.0, 1.0)
    int16 = np.rint(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(int(sample_rate))
        output.writeframes(int16.tobytes())
