"""Render HRI-MoT predictions and build time-aligned GT comparison videos."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_RENDER_SCRIPT = Path("tools/render.sh")
DEFAULT_PAPER_RENDER_XML = Path("assets/robot_scene.xml")
DEFAULT_GT_VIDEO_ROOT = Path("data/gt_videos")
DEFAULT_RENDER_CONDA_ENV = "gmr_astribot"
DEFAULT_INTERACTION_AUDIO_CODEC = "aac"


def add_video_postprocess_args(
    parser: argparse.ArgumentParser,
    *,
    allow_gt_override: bool,
    default_render_xml: Optional[Path] = None,
) -> None:
    group = parser.add_argument_group("automatic video postprocess")
    group.add_argument(
        "--skip-video-postprocess",
        action="store_true",
        help="Do not render the prediction pkl or build a GT comparison video.",
    )
    group.add_argument(
        "--render-script",
        default=str(DEFAULT_RENDER_SCRIPT),
        help="render.sh used to convert a prediction pkl to mp4.",
    )
    group.add_argument(
        "--render-conda-env",
        default=DEFAULT_RENDER_CONDA_ENV,
        help="Conda environment used to run render.sh; use an empty string to inherit the current environment.",
    )
    group.add_argument(
        "--render-xml-path",
        default=str(default_render_xml) if default_render_xml is not None else None,
        help=(
            "Optional MuJoCo scene XML forwarded to the renderer. When omitted, "
            "the renderer's original default scene is used."
        ),
    )
    group.add_argument(
        "--gt-video-root",
        default=str(DEFAULT_GT_VIDEO_ROOT),
        help="Root containing GT videos with paths mirroring the retarget dataset.",
    )
    if allow_gt_override:
        group.add_argument(
            "--gt-video",
            default=None,
            help="Optional explicit GT mp4. Useful for standalone WAV inference without a dataset source pkl.",
        )
    group.add_argument(
        "--video-postprocess-strict",
        action="store_true",
        help="Fail inference when rendering, GT clipping, or comparison generation fails.",
    )
    group.add_argument(
        "--interaction-audio-codec",
        choices=("mp3", "aac"),
        default=DEFAULT_INTERACTION_AUDIO_CODEC,
        help=(
            "Audio codec embedded in interaction MP4 files. AAC is the default "
            "for standard MP4 and macOS playback; MP3 remains available as an override."
        ),
    )


def _dataset_relative_path(path: Path) -> Path:
    parts = [part for part in path.parts if part not in (path.anchor, "")]
    for marker in ("seamless_interaction_retarget", "seamless_interaction_full"):
        if marker in parts:
            marker_idx = parts.index(marker)
            tail = parts[marker_idx + 1 :]
            if tail:
                return Path(*tail)
    if len(parts) >= 5:
        return Path(*parts[-5:])
    return Path(*parts) if parts else Path(path.name)


def resolve_gt_video(
    *,
    source_robot_pkl: Optional[str],
    gt_video_root: Path,
    explicit_gt_video: Optional[str] = None,
) -> Optional[Path]:
    """Resolve the full GT mp4 corresponding to a source robot pkl."""
    if explicit_gt_video:
        explicit = Path(explicit_gt_video).expanduser().resolve()
        if not explicit.is_file():
            raise FileNotFoundError(f"Explicit GT video does not exist: {explicit}")
        return explicit
    if not source_robot_pkl:
        return None

    relative = _dataset_relative_path(Path(source_robot_pkl).expanduser())
    candidate = Path(gt_video_root).expanduser().resolve() / relative.with_suffix(".mp4")
    return candidate if candidate.is_file() else None


def _run(command: list[str], *, cwd: Optional[Path] = None) -> None:
    printable = " ".join(command)
    print(f"[video] run: {printable}", flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def render_prediction_video(
    *,
    prediction_pkl: Path,
    render_script: Path,
    render_conda_env: str,
    render_xml_path: Optional[Path] = None,
) -> Path:
    prediction_pkl = Path(prediction_pkl).expanduser().resolve()
    render_script = Path(render_script).expanduser().resolve()
    if not prediction_pkl.is_file():
        raise FileNotFoundError(f"Prediction pkl does not exist: {prediction_pkl}")
    if not render_script.is_file():
        raise FileNotFoundError(f"Render script does not exist: {render_script}")
    if render_xml_path is not None:
        render_xml_path = Path(render_xml_path).expanduser().resolve()
        if not render_xml_path.is_file():
            raise FileNotFoundError(
                f"MuJoCo render scene XML does not exist: {render_xml_path}"
            )

    prediction_video = prediction_pkl.with_suffix(".mp4")
    if render_conda_env:
        conda = shutil.which("conda")
        if conda is None:
            raise FileNotFoundError("conda is required when --render-conda-env is set")
        command = [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            str(render_conda_env),
            "bash",
            str(render_script),
        ]
    else:
        command = ["bash", str(render_script)]
    command.extend(["--pkl", str(prediction_pkl), "--video_path", str(prediction_video)])
    if render_xml_path is not None:
        command.extend(["--xml_path", str(render_xml_path)])
    _run(command, cwd=render_script.parent)
    if not prediction_video.is_file() or prediction_video.stat().st_size <= 48:
        raise RuntimeError(f"Renderer did not create a valid-looking mp4: {prediction_video}")
    return prediction_video


def _probe_video(video_path: Path) -> Dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("ffprobe is required for automatic video postprocessing")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames:format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {video_path}")
    stream = streams[0]
    fps_text = str(stream.get("avg_frame_rate") or "0/1")
    fps = float(Fraction(fps_text)) if fps_text != "0/0" else 0.0
    duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
    frames_text = stream.get("nb_frames")
    frames = int(frames_text) if frames_text not in (None, "N/A") else int(round(duration * fps))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frames": frames,
        "duration": duration,
    }


def _prediction_pkl_duration_sec(prediction_pkl: Path) -> float:
    """Return the physical duration represented by the prediction motion."""

    prediction_pkl = Path(prediction_pkl).expanduser().resolve()
    with prediction_pkl.open("rb") as fp:
        payload = pickle.load(fp)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Prediction pkl must contain a dict: {prediction_pkl}"
        )
    fps = float(payload.get("fps", 0.0) or 0.0)
    metadata = payload.get("hri_mot_prediction", {})
    frame_count = (
        int(metadata.get("output_frames", 0) or 0)
        if isinstance(metadata, dict)
        else 0
    )
    if frame_count <= 0:
        qpos = payload.get("qpos_list")
        if qpos is not None:
            frame_count = len(qpos)
    if fps <= 0.0 or frame_count <= 0:
        raise ValueError(
            f"Invalid prediction timing in {prediction_pkl}: "
            f"fps={fps}, frames={frame_count}"
        )
    return float(frame_count / fps)


def normalize_prediction_video_duration(
    *,
    prediction_video: Path,
    expected_duration_sec: float,
) -> Dict[str, Any]:
    """Force the rendered video to cover the complete prediction timeline."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for automatic video postprocessing")
    prediction_video = Path(prediction_video).expanduser().resolve()
    timing_before = _probe_video(prediction_video)
    fps = float(timing_before["fps"])
    if fps <= 0.0 or expected_duration_sec <= 0.0:
        raise ValueError(
            f"Invalid prediction video timing: video={timing_before}, "
            f"expected_duration_sec={expected_duration_sec}"
        )
    target_frames = max(1, int(round(float(expected_duration_sec) * fps)))
    if int(timing_before["frames"]) == target_frames:
        return {
            "expected_duration_sec": float(expected_duration_sec),
            "target_frames": int(target_frames),
            "frames_before": int(timing_before["frames"]),
            "frames_after": int(timing_before["frames"]),
            "normalized": False,
        }

    temporary = prediction_video.with_name(
        f"{prediction_video.stem}__duration-normalized.tmp.mp4"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(prediction_video),
        "-vf",
        (
            "tpad=stop_mode=clone:"
            f"stop_duration={float(expected_duration_sec):.9f},"
            "setpts=PTS-STARTPTS"
        ),
        "-frames:v",
        str(target_frames),
        "-r",
        f"{fps:.12g}",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        str(temporary),
    ]
    _run(command)
    normalized = _probe_video(temporary)
    if int(normalized["frames"]) != target_frames:
        raise RuntimeError(
            f"Normalized prediction has {normalized['frames']} frames, "
            f"expected {target_frames}"
        )
    temporary.replace(prediction_video)
    return {
        "expected_duration_sec": float(expected_duration_sec),
        "target_frames": int(target_frames),
        "frames_before": int(timing_before["frames"]),
        "frames_after": int(normalized["frames"]),
        "normalized": True,
    }


def mux_interaction_audio(
    *,
    video_path: Path,
    interaction_audio: Path,
    audio_start_sec: float = 0.0,
    loudness_lufs: Optional[float] = None,
    audio_codec: str = DEFAULT_INTERACTION_AUDIO_CODEC,
) -> Dict[str, Any]:
    """Attach the complete dialogue track without changing the video frame grid."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for automatic video postprocessing")
    video_path = Path(video_path).expanduser().resolve()
    interaction_audio = Path(interaction_audio).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if not interaction_audio.is_file():
        raise FileNotFoundError(
            f"Interaction audio does not exist: {interaction_audio}"
        )
    audio_start_sec = float(audio_start_sec)
    if audio_start_sec < 0.0:
        raise ValueError(
            f"Interaction audio start must be non-negative, got {audio_start_sec}"
        )
    if loudness_lufs is not None and not -70.0 <= float(loudness_lufs) <= -5.0:
        raise ValueError(
            "FFmpeg loudnorm target must be between -70 and -5 LUFS, "
            f"got {loudness_lufs}"
        )
    audio_codec = str(audio_codec).lower()
    audio_encoders = {
        "aac": "aac",
        "mp3": "libmp3lame",
    }
    if audio_codec not in audio_encoders:
        raise ValueError(
            f"Unsupported MP4 audio codec {audio_codec!r}; "
            f"expected one of {sorted(audio_encoders)}"
        )
    timing_before = _probe_video(video_path)
    if timing_before["fps"] <= 0.0 or timing_before["frames"] <= 0:
        raise ValueError(f"Invalid video timing for audio mux: {timing_before}")
    duration = float(timing_before["frames"] / timing_before["fps"])
    temporary = video_path.with_name(
        f"{video_path.stem}__audio-mux.tmp.mp4"
    )
    audio_filters = [
        f"atrim=start={audio_start_sec:.9f}",
        "asetpts=PTS-STARTPTS",
    ]
    if loudness_lufs is not None:
        audio_filters.append(
            f"loudnorm=I={float(loudness_lufs):.3f}:LRA=11:TP=-1.5"
        )
    if loudness_lufs is not None or audio_codec == "mp3":
        # Use a broadly supported rate for MP3-in-MP4 playback and after
        # loudnorm's internal true-peak oversampling.
        audio_filters.append("aresample=48000")
    audio_filters.append(f"apad=pad_dur={duration:.9f}")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(interaction_audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        audio_encoders[audio_codec],
        "-b:a",
        "192k",
        "-af",
        ",".join(audio_filters),
        "-t",
        f"{duration:.9f}",
        "-movflags",
        "+faststart",
        "-y",
        str(temporary),
    ]
    _run(command)
    timing_after = _probe_video(temporary)
    if int(timing_after["frames"]) != int(timing_before["frames"]):
        raise RuntimeError(
            f"Audio mux changed video frame count from {timing_before['frames']} "
            f"to {timing_after['frames']}"
        )
    temporary.replace(video_path)
    return {
        "video": str(video_path),
        "audio": str(interaction_audio),
        "duration_sec": duration,
        "audio_start_sec": audio_start_sec,
        "loudness_lufs": (
            float(loudness_lufs) if loudness_lufs is not None else None
        ),
        "true_peak_limit_dbtp": -1.5 if loudness_lufs is not None else None,
        "video_frames": int(timing_after["frames"]),
        "audio_codec": audio_codec,
    }


def clip_gt_video(
    *,
    gt_video: Path,
    output_path: Path,
    start_sec: float,
    reference_video: Path,
) -> Path:
    """Clip GT to prediction timing, freezing its final frame when too short."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for automatic video postprocessing")
    gt_video = Path(gt_video).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    reference_video = Path(reference_video).expanduser().resolve()
    if not gt_video.is_file():
        raise FileNotFoundError(f"GT video does not exist: {gt_video}")
    reference = _probe_video(reference_video)
    gt_probe = _probe_video(gt_video)
    if reference["fps"] <= 0.0 or reference["frames"] <= 0:
        raise ValueError(f"Invalid prediction video timing: {reference_video} -> {reference}")
    duration = reference["frames"] / reference["fps"]
    if start_sec < 0.0:
        raise ValueError(f"GT clip start must be >= 0, got {start_sec}")
    if gt_probe["duration"] <= start_sec:
        raise ValueError(
            f"GT video has no frame at clip start {start_sec:.6f}: "
            f"duration={gt_probe['duration']:.6f}, path={gt_video}"
        )
    needs_tail_padding = (
        gt_probe["duration"] + 0.5 / reference["fps"]
        < start_sec + duration
    )
    if needs_tail_padding:
        print(
            f"[video] GT ends before the full interaction; freezing its last "
            f"frame through {start_sec + duration:.6f}s",
            flush=True,
        )

    width = reference["width"]
    height = reference["height"]
    fps = reference["fps"]
    video_filter = (
        f"fps={fps:.12g},"
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,"
        f"tpad=stop_mode=clone:stop_duration={duration:.9f},"
        "setpts=PTS-STARTPTS"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{float(start_sec):.9f}",
        "-i",
        str(gt_video),
        "-vf",
        video_filter,
        "-frames:v",
        str(reference["frames"]),
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]
    _run(command)
    clipped = _probe_video(output_path)
    if clipped["frames"] != reference["frames"]:
        raise RuntimeError(
            f"GT clip frame count {clipped['frames']} does not match prediction {reference['frames']}"
        )
    return output_path


def stack_prediction_left_gt_right(
    *,
    prediction_video: Path,
    gt_clip_video: Path,
    output_path: Path,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for automatic video postprocessing")
    prediction_video = Path(prediction_video).expanduser().resolve()
    gt_clip_video = Path(gt_clip_video).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    prediction = _probe_video(prediction_video)
    gt_clip = _probe_video(gt_clip_video)
    if prediction["frames"] != gt_clip["frames"]:
        raise ValueError(
            f"Prediction and GT clip frame counts differ: {prediction['frames']} vs {gt_clip['frames']}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(prediction_video),
        "-i",
        str(gt_clip_video),
        "-filter_complex",
        "[0:v]setpts=PTS-STARTPTS[predict];[1:v]setpts=PTS-STARTPTS[gt];"
        "[predict][gt]hstack=inputs=2[outv]",
        "-map",
        "[outv]",
        "-frames:v",
        str(prediction["frames"]),
        "-r",
        f"{prediction['fps']:.12g}",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-y",
        str(output_path),
    ]
    _run(command)
    comparison = _probe_video(output_path)
    if comparison["frames"] != prediction["frames"]:
        raise RuntimeError(
            f"Comparison frame count {comparison['frames']} does not match prediction {prediction['frames']}"
        )
    return output_path


def postprocess_prediction_video(
    *,
    prediction_pkl: Path,
    action_t_start: float,
    source_robot_pkl: Optional[str],
    render_script: Path,
    render_conda_env: str,
    gt_video_root: Path,
    render_xml_path: Optional[Path] = None,
    explicit_gt_video: Optional[str] = None,
    interaction_audio: Optional[Path] = None,
    interaction_audio_start_sec: float = 0.0,
    interaction_audio_loudness_lufs: Optional[float] = None,
    interaction_audio_codec: str = DEFAULT_INTERACTION_AUDIO_CODEC,
) -> Dict[str, Any]:
    """Render prediction and, when GT is available, create a synchronized hstack."""
    prediction_pkl = Path(prediction_pkl).expanduser().resolve()
    prediction_video = render_prediction_video(
        prediction_pkl=prediction_pkl,
        render_script=render_script,
        render_conda_env=render_conda_env,
        render_xml_path=render_xml_path,
    )
    prediction_duration_sec = _prediction_pkl_duration_sec(prediction_pkl)
    prediction_duration = normalize_prediction_video_duration(
        prediction_video=prediction_video,
        expected_duration_sec=prediction_duration_sec,
    )
    result: Dict[str, Any] = {
        "status": "prediction_rendered",
        "layout": "prediction_left_gt_right",
        "prediction_video": str(prediction_video),
        "render_xml_path": (
            str(Path(render_xml_path).expanduser().resolve())
            if render_xml_path is not None
            else None
        ),
        "gt_video": None,
        "gt_clip_video": None,
        "comparison_video": None,
        "gt_clip_start_sec": float(action_t_start),
        "interaction_duration_sec": float(prediction_duration_sec),
        "prediction_duration": prediction_duration,
        "interaction_audio": (
            str(Path(interaction_audio).expanduser().resolve())
            if interaction_audio is not None
            else None
        ),
        "prediction_audio_mux": None,
        "comparison_audio_mux": None,
        "interaction_audio_start_sec": float(interaction_audio_start_sec),
        "interaction_audio_loudness_lufs": (
            float(interaction_audio_loudness_lufs)
            if interaction_audio_loudness_lufs is not None
            else None
        ),
        "interaction_audio_codec": str(interaction_audio_codec),
    }

    gt_video = resolve_gt_video(
        source_robot_pkl=source_robot_pkl,
        gt_video_root=gt_video_root,
        explicit_gt_video=explicit_gt_video,
    )
    if gt_video is None:
        if interaction_audio is not None:
            result["prediction_audio_mux"] = mux_interaction_audio(
                video_path=prediction_video,
                interaction_audio=interaction_audio,
                audio_start_sec=interaction_audio_start_sec,
                loudness_lufs=interaction_audio_loudness_lufs,
                audio_codec=interaction_audio_codec,
            )
        result["status"] = "prediction_rendered_gt_not_found"
        print(
            f"[video] GT not found; rendered prediction only. source_robot_pkl={source_robot_pkl!r}",
            flush=True,
        )
        return result

    gt_clip_video = prediction_pkl.with_name(f"{prediction_pkl.stem}__gt_clip.mp4")
    comparison_video = prediction_pkl.with_name(
        f"{prediction_pkl.stem}__predict-left_gt-right.mp4"
    )
    clip_gt_video(
        gt_video=gt_video,
        output_path=gt_clip_video,
        start_sec=float(action_t_start),
        reference_video=prediction_video,
    )
    stack_prediction_left_gt_right(
        prediction_video=prediction_video,
        gt_clip_video=gt_clip_video,
        output_path=comparison_video,
    )
    prediction_audio_mux = None
    comparison_audio_mux = None
    if interaction_audio is not None:
        prediction_audio_mux = mux_interaction_audio(
            video_path=prediction_video,
            interaction_audio=interaction_audio,
            audio_start_sec=interaction_audio_start_sec,
            loudness_lufs=interaction_audio_loudness_lufs,
            audio_codec=interaction_audio_codec,
        )
        comparison_audio_mux = mux_interaction_audio(
            video_path=comparison_video,
            interaction_audio=interaction_audio,
            audio_start_sec=interaction_audio_start_sec,
            loudness_lufs=interaction_audio_loudness_lufs,
            audio_codec=interaction_audio_codec,
        )
    result.update(
        {
            "status": "comparison_created",
            "gt_video": str(gt_video),
            "gt_clip_video": str(gt_clip_video),
            "comparison_video": str(comparison_video),
            "prediction_audio_mux": prediction_audio_mux,
            "comparison_audio_mux": comparison_audio_mux,
        }
    )
    print(f"[video] prediction: {prediction_video}", flush=True)
    print(f"[video] GT clip: {gt_clip_video}", flush=True)
    print(f"[video] comparison (prediction left, GT right): {comparison_video}", flush=True)
    return result


def maybe_postprocess_prediction_video(
    *,
    skip: bool,
    strict: bool,
    prediction_pkl: Path,
    action_t_start: float,
    source_robot_pkl: Optional[str],
    render_script: Path,
    render_conda_env: str,
    gt_video_root: Path,
    render_xml_path: Optional[Path] = None,
    explicit_gt_video: Optional[str] = None,
    interaction_audio: Optional[Path] = None,
    interaction_audio_start_sec: float = 0.0,
    interaction_audio_loudness_lufs: Optional[float] = None,
    interaction_audio_codec: str = DEFAULT_INTERACTION_AUDIO_CODEC,
) -> Dict[str, Any]:
    """Run video postprocessing without discarding inference outputs on optional failures."""
    if skip:
        return {
            "status": "skipped",
            "layout": "prediction_left_gt_right",
            "prediction_video": None,
            "render_xml_path": (
                str(Path(render_xml_path).expanduser().resolve())
                if render_xml_path is not None
                else None
            ),
            "gt_video": None,
            "gt_clip_video": None,
            "comparison_video": None,
            "gt_clip_start_sec": float(action_t_start),
            "interaction_audio": (
                str(Path(interaction_audio).expanduser().resolve())
                if interaction_audio is not None
                else None
            ),
            "interaction_audio_start_sec": float(interaction_audio_start_sec),
            "interaction_audio_loudness_lufs": (
                float(interaction_audio_loudness_lufs)
                if interaction_audio_loudness_lufs is not None
                else None
            ),
            "interaction_audio_codec": str(interaction_audio_codec),
        }
    try:
        return postprocess_prediction_video(
            prediction_pkl=prediction_pkl,
            action_t_start=action_t_start,
            source_robot_pkl=source_robot_pkl,
            render_script=render_script,
            render_conda_env=render_conda_env,
            gt_video_root=gt_video_root,
            render_xml_path=render_xml_path,
            explicit_gt_video=explicit_gt_video,
            interaction_audio=interaction_audio,
            interaction_audio_start_sec=interaction_audio_start_sec,
            interaction_audio_loudness_lufs=interaction_audio_loudness_lufs,
            interaction_audio_codec=interaction_audio_codec,
        )
    except Exception as exc:
        if strict:
            raise
        print(f"[video][warn] automatic video postprocess failed: {type(exc).__name__}: {exc}", flush=True)
        return {
            "status": "failed",
            "layout": "prediction_left_gt_right",
            "prediction_video": None,
            "render_xml_path": (
                str(Path(render_xml_path).expanduser().resolve())
                if render_xml_path is not None
                else None
            ),
            "gt_video": None,
            "gt_clip_video": None,
            "comparison_video": None,
            "gt_clip_start_sec": float(action_t_start),
            "interaction_audio": (
                str(Path(interaction_audio).expanduser().resolve())
                if interaction_audio is not None
                else None
            ),
            "interaction_audio_start_sec": float(interaction_audio_start_sec),
            "interaction_audio_loudness_lufs": (
                float(interaction_audio_loudness_lufs)
                if interaction_audio_loudness_lufs is not None
                else None
            ),
            "interaction_audio_codec": str(interaction_audio_codec),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
