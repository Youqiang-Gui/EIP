#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for name in BASE_MODEL_DIR CHECKPOINT COSPEECH_INDEX; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

expanded_args=()
if [[ -n "${EXPANDED_INDEX:-}" ]]; then
  expanded_args=(--expanded-index "$EXPANDED_INDEX")
fi

python "$PROJECT_ROOT/scripts/hri_mot/infer_hri_mot_teacher_forcing.py" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-dir "$BASE_MODEL_DIR" \
  "${expanded_args[@]}" \
  --cospeech-index "$COSPEECH_INDEX" \
  --episodes-per-source "${EPISODES_PER_SOURCE:-5}" \
  --skip-video-postprocess \
  --output-dir "${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/teacher_forcing}"
