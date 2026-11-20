#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for name in BASE_MODEL_DIR CHECKPOINT WAV_PATH; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

python "$PROJECT_ROOT/scripts/hri_mot/infer_hri_mot_wav.py" \
  --wav "$WAV_PATH" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-dir "$BASE_MODEL_DIR" \
  --emotion-override "${EMOTION_OVERRIDE:-auto}" \
  --tts-mode estimated \
  --no-tts-save-audio \
  --skip-video-postprocess \
  --output-dir "${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/wav_inference}"
