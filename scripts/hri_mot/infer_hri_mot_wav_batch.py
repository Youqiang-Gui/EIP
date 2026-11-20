#!/usr/bin/env python3
"""Run standalone WAV inference for every checkpoint in one training run.

The checkpoint directory is intentionally derived from ``DEFAULT_CHECKPOINT``
in ``infer_hri_mot.py``.  Point that constant at any ``step-XXXXXX.pt`` in the
desired training run, then invoke this script with a WAV path.  Full inference
artifacts are reused in the normal WAV output directory, while each final video
is copied to ``<checkpoint-step>.mp4`` before the next checkpoint runs.

Example:

    python -u scripts/hri_mot/infer_hri_mot_wav_batch.py \
        --wav /path/to/input.wav \
        --seed 2024

Additional ``infer_hri_mot_wav.py`` options can be placed after ``--``:

    python -u scripts/hri_mot/infer_hri_mot_wav_batch.py \
        --wav /path/to/input.wav -- --device cuda:1
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audiointeraction.hri_mot.emotion_conditioning import EMOTION_LABELS

INFERENCE_SCRIPT = Path(__file__).with_name("infer_hri_mot_wav.py")
INFER_HRI_MOT_SCRIPT = Path(__file__).with_name("infer_hri_mot.py")
CHECKPOINT_NAME_RE = re.compile(r"^step-(\d+)\.pt$")
PROTECTED_INFERENCE_OPTIONS = {
    "--checkpoint",
    "--output-dir",
    "--emotion-override",
    "--seed",
    "--skip-video-postprocess",
    "--speak-conditioning",
    "--train-dir",
    "--wav",
}


@dataclass(frozen=True)
class Checkpoint:
    step: int
    path: Path


def read_default_checkpoint() -> Path:
    """Read DEFAULT_CHECKPOINT without importing the heavyweight model module."""

    tree = ast.parse(
        INFER_HRI_MOT_SCRIPT.read_text(encoding="utf-8"),
        filename=str(INFER_HRI_MOT_SCRIPT),
    )
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_CHECKPOINT"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, str) or not value.strip():
            break
        return Path(value).expanduser().resolve()
    raise ValueError(
        f"Could not read a string DEFAULT_CHECKPOINT from {INFER_HRI_MOT_SCRIPT}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run infer_hri_mot_wav.py for every step-*.pt next to "
            "infer_hri_mot.DEFAULT_CHECKPOINT and save each video as <step>.mp4."
        )
    )
    parser.add_argument(
        "--wav",
        required=True,
        help="Input WAV used for every checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List checkpoints and commands without launching inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Inference seed used for every checkpoint. By default, each "
            "checkpoint uses the seed recorded during training."
        ),
    )
    parser.add_argument(
        "--emotion-override",
        choices=("auto", *EMOTION_LABELS),
        default="auto",
        help=(
            "Use auto semantic-tag conditioning or select a semantic emotion for every speaking "
            "chunk. Forced runs save videos with emotion-label suffixes."
        ),
    )
    parser.add_argument(
        "inference_args",
        nargs=argparse.REMAINDER,
        help="Extra infer_hri_mot_wav.py arguments, placed after --.",
    )
    args = parser.parse_args(argv)
    if args.inference_args[:1] == ["--"]:
        args.inference_args = args.inference_args[1:]
    _validate_extra_inference_args(args.inference_args, parser=parser)
    return args


def _validate_extra_inference_args(
    inference_args: Sequence[str],
    *,
    parser: argparse.ArgumentParser,
) -> None:
    for token in inference_args:
        option = token.split("=", 1)[0]
        if option in PROTECTED_INFERENCE_OPTIONS:
            parser.error(
                f"{option} is managed by the batch script and cannot be passed after --"
            )


def discover_checkpoints(checkpoint_dir: Path) -> list[Checkpoint]:
    checkpoints: list[Checkpoint] = []
    for path in checkpoint_dir.glob("step-*.pt"):
        match = CHECKPOINT_NAME_RE.fullmatch(path.name)
        if match is not None and path.is_file():
            checkpoints.append(
                Checkpoint(step=int(match.group(1)), path=path.resolve())
            )
    checkpoints.sort(key=lambda checkpoint: (checkpoint.step, checkpoint.path.name))
    if not checkpoints:
        raise FileNotFoundError(f"No step-*.pt checkpoints found in {checkpoint_dir}")

    duplicate_steps = [
        step
        for step in {checkpoint.step for checkpoint in checkpoints}
        if sum(checkpoint.step == step for checkpoint in checkpoints) > 1
    ]
    if duplicate_steps:
        raise ValueError(
            "Checkpoint filenames resolve to duplicate step numbers: "
            + ", ".join(str(step) for step in sorted(duplicate_steps))
        )
    return checkpoints


def build_inference_command(
    *,
    wav_path: Path,
    checkpoint: Checkpoint,
    train_dir: Path,
    output_dir: Path,
    seed: int | None,
    emotion_override: str,
    inference_args: Sequence[str],
) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(INFERENCE_SCRIPT),
        "--wav",
        str(wav_path),
        "--checkpoint",
        str(checkpoint.path),
        "--train-dir",
        str(train_dir),
        "--output-dir",
        str(output_dir),
        "--speak-conditioning",
        "auto",
        "--emotion-override",
        str(emotion_override),
        "--video-postprocess-strict",
    ]
    if seed is not None:
        command.extend(("--seed", str(seed)))
    command.extend(inference_args)
    return command


def read_video_path(
    *,
    response_json: Path,
    expected_step: int,
    output_dir: Path,
) -> Path:
    if not response_json.is_file():
        raise FileNotFoundError(
            f"Inference did not write response JSON: {response_json}"
        )
    with response_json.open("r", encoding="utf-8") as fp:
        loaded_payload: Any = json.load(fp)
    if not isinstance(loaded_payload, dict):
        raise RuntimeError(f"Response JSON must contain an object: {response_json}")
    payload: dict[str, Any] = loaded_payload

    actual_step = payload.get("checkpoint_step")
    if actual_step != expected_step:
        raise RuntimeError(
            f"Response JSON checkpoint step is {actual_step!r}, expected {expected_step}"
        )
    video_outputs = payload.get("video_outputs")
    if not isinstance(video_outputs, dict):
        raise RuntimeError(
            f"Response JSON has no video_outputs object: {response_json}"
        )
    video_value = video_outputs.get("comparison_video") or video_outputs.get(
        "prediction_video"
    )
    if not video_value:
        raise RuntimeError(
            "Inference completed without a rendered MP4: "
            f"status={video_outputs.get('status')!r}"
        )

    video_path = Path(str(video_value)).expanduser()
    if not video_path.is_absolute():
        video_path = output_dir / video_path
    video_path = video_path.resolve()
    if not video_path.is_file() or video_path.stat().st_size <= 48:
        raise RuntimeError(f"Rendered MP4 is missing or invalid: {video_path}")
    return video_path


def save_step_video(*, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return

    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    wav_path = Path(args.wav).expanduser().resolve()
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV does not exist: {wav_path}")

    default_checkpoint = read_default_checkpoint()
    checkpoint_dir = default_checkpoint.parent
    if not checkpoint_dir.is_dir():
        raise NotADirectoryError(
            "DEFAULT_CHECKPOINT parent does not exist: "
            f"DEFAULT_CHECKPOINT={default_checkpoint}"
        )
    checkpoints = discover_checkpoints(checkpoint_dir)
    output_dir = checkpoint_dir / "wav_inference" / wav_path.stem
    response_json = output_dir / f"{wav_path.stem}__response.json"
    print(f"[batch] DEFAULT_CHECKPOINT: {default_checkpoint}", flush=True)
    print(f"[batch] checkpoint directory: {checkpoint_dir}", flush=True)
    print(f"[batch] checkpoints: {len(checkpoints)}", flush=True)
    print(f"[batch] WAV: {wav_path}", flush=True)
    print(f"[batch] output directory: {output_dir}", flush=True)
    print(
        "[batch] seed: "
        + (str(args.seed) if args.seed is not None else "checkpoint default"),
        flush=True,
    )
    print(
        f"[batch] emotion override: {args.emotion_override}",
        flush=True,
    )

    saved_videos: list[Path] = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        condition_suffix = (
            ""
            if args.emotion_override == "auto"
            else f"__{args.emotion_override}"
        )
        destination = output_dir / f"{checkpoint.step}{condition_suffix}.mp4"
        command = build_inference_command(
            wav_path=wav_path,
            checkpoint=checkpoint,
            train_dir=checkpoint_dir,
            output_dir=output_dir,
            seed=args.seed,
            emotion_override=args.emotion_override,
            inference_args=args.inference_args,
        )
        print(
            f"\n[batch {index}/{len(checkpoints)}] step={checkpoint.step} "
            f"checkpoint={checkpoint.path.name}",
            flush=True,
        )
        if args.dry_run:
            print(f"[batch][dry-run] {shlex.join(command)}", flush=True)
            print(f"[batch][dry-run] video -> {destination}", flush=True)
            continue

        subprocess.run(command, cwd=ROOT, check=True)
        rendered_video = read_video_path(
            response_json=response_json,
            expected_step=checkpoint.step,
            output_dir=output_dir,
        )
        save_step_video(source=rendered_video, destination=destination)
        saved_videos.append(destination)
        print(f"[batch] saved video: {destination}", flush=True)

    if args.dry_run:
        print(f"\n[batch] dry run complete: {len(checkpoints)} checkpoints", flush=True)
    else:
        print(
            f"\n[batch] complete: saved {len(saved_videos)} videos in {output_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()



# python -u scripts/hri_mot/infer_hri_mot_wav_batch.py --wav examples/input.wav
