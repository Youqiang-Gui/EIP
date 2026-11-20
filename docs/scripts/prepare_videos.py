#!/usr/bin/env python3
"""Refresh browser videos, posters and cache versions without changing originals.

Run from any directory: python /path/to/EIP/docs/scripts/prepare_videos.py
Requires ffmpeg and ffprobe on PATH. No third-party Python packages are needed.
"""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "assets/videos"
WEB_DIR = VIDEO_DIR / "web"
POSTER_DIR = ROOT / "assets/posters"
STATE_PATH = WEB_DIR / ".build-state.json"
RECIPE = "h264-1080p-crf23-fast-aac-frame1-v1"
# Source recording, web video, and poster filenames.
CLIPS = (
    ("crack.mp4", "crack.mp4", "crack-video.jpg"),
    ("cough.mp4", "cough.mp4", "cough-video.jpg"),
    ("emotion.mp4", "emotion.mp4", "emotion-video.jpg"),
    ("state adaptation.mp4", "state adaptation.mp4", "state-adaptation-video.jpg"),
    ("sim-01.mov", "sim-01.mp4", "sim-01-video.jpg"),
    ("sim-02.mov", "sim-02.mp4", "sim-02-video.jpg"),
)


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def atomic_text(path, text):
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False) as file:
        file.write(text)
        temporary = Path(file.name)
    temporary.replace(path)


def version_urls(text, versions):
    for path, version in versions.items():
        # Match complete local URLs only, retaining unrelated configuration edits.
        pattern = r'(["\'])' + re.escape(path) + r'(?:\?v=[^"\']*)?\1'
        text = re.sub(pattern, lambda match: f'{match[1]}{path}?v={version[:12]}{match[1]}', text)
    return text


def build_clip(source, web, poster):
    info = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(source)
    ]))
    stream = next(item for item in info["streams"] if item["codec_type"] == "video")
    audio = [item for item in info["streams"] if item["codec_type"] == "audio"]
    can_copy = (stream["codec_name"] == "h264" and stream.get("pix_fmt") == "yuv420p"
                and stream["width"] <= 1920 and stream["height"] % 2 == 0)
    # Temporary files are on the same filesystem, so replacement is atomic.
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=WEB_DIR) as folder:
        temp_web = Path(folder) / web.name
        temp_poster = Path(folder) / poster.name
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-threads", "2", "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?",
                   "-map_metadata", "-1"]
        if can_copy:
            command += ["-c:v", "copy"]
        else:
            command += ["-vf", "scale=w='min(1920,iw)':h=-2", "-c:v", "libx264",
                        "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "2"]
        command += (["-c:a", "copy"] if all(item["codec_name"] == "aac" for item in audio)
                    else ["-c:a", "aac", "-b:a", "128k"])
        subprocess.run(command + ["-movflags", "+faststart", str(temp_web)], check=True)
        timestamp = min(1.0, float(info["format"]["duration"]) / 2)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-threads", "2",
            "-ss", str(timestamp), "-i", str(temp_web), "-frames:v", "1",
            "-vf", "scale=w='min(960,iw)':h=-2", "-q:v", "3", "-threads", "2", str(temp_poster)
        ], check=True)
        temp_web.replace(web)
        temp_poster.replace(poster)


def main():
    for name in ("ffmpeg", "ffprobe"):
        if not shutil.which(name):
            raise SystemExit(f"Missing {name}; install FFmpeg before running this script.")
    for original, _, _ in CLIPS:
        if not (VIDEO_DIR / original).is_file():
            raise SystemExit(f"Missing original: {VIDEO_DIR / original}")
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    try:
        state = json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        state = {}
    versions = {}
    updated = 0
    for original, output, cover in CLIPS:
        source, web, poster = VIDEO_DIR / original, WEB_DIR / output, POSTER_DIR / cover
        source_hash = digest(source)
        previous = state.get(original, {})
        outputs = {str(path.relative_to(ROOT)): digest(path) for path in (web, poster) if path.exists()}
        current = (previous.get("source") == source_hash and previous.get("recipe") == RECIPE
                   and len(outputs) == 2 and previous.get("outputs") == outputs)
        if not current:
            print(f"Updating {original} ...", flush=True)
            build_clip(source, web, poster)
            outputs = {str(path.relative_to(ROOT)): digest(path) for path in (web, poster)}
            state[original] = {"source": source_hash, "recipe": RECIPE, "outputs": outputs}
            atomic_text(STATE_PATH, json.dumps(state, indent=2) + "\n")
            updated += 1
        else:
            print(f"Unchanged: {original}", flush=True)
        versions.update(outputs)

    config_path, index_path = ROOT / "site-config.js", ROOT / "index.html"
    atomic_text(config_path, version_urls(config_path.read_text(encoding="utf-8"), versions))
    versions["site-config.js"] = digest(config_path)
    atomic_text(index_path, version_urls(index_path.read_text(encoding="utf-8"), versions))
    print(f"Done: {updated} video(s) updated; posters and cache versions synchronized. Refresh the page.")


if __name__ == "__main__":
    main()
