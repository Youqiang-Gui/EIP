#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
}

for name in BASE_MODEL_DIR STAGE1_CHECKPOINT COSPEECH_INDEX EMOTION_INDEX IDLE_MOTION_PKL; do
  require_env "$name"
done

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/stage2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"

torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
  "$PROJECT_ROOT/scripts/hri_mot/train_hri_mot_posttrain.py" \
  --checkpoint-dir "$BASE_MODEL_DIR" \
  --emotion-init-from "$STAGE1_CHECKPOINT" \
  --cospeech-train-index "$EMOTION_INDEX" \
  --cospeech-baseline-index "$COSPEECH_INDEX" \
  --idle-motion-pkl "$IDLE_MOTION_PKL" \
  --use-emotion-conditioning true \
  --emotion-film-layers 4 \
  --motion-unfreeze-action-layers 8 \
  --motion-unfreeze-response-time-encoder false \
  --emotion-samples-per-class 512 \
  --norm-stats-mode source \
  --posttrain-episode-batch-size 2 \
  --cospeech-episode-batch-size 2 \
  --cospeech-subset-size 0 \
  --listen-action-loss-weight 0.0 \
  --cospeech-speaking-loss-weight 1.0 \
  --cospeech-listening-loss-weight 0.0 \
  --cospeech-idle-loss-weight 0.0 \
  --epochs 100 \
  --lr 1e-4 \
  --min-lr 1e-5 \
  --pretrained-lr-scale 0.02 \
  --lr-warmup-steps 100 \
  --lr-schedule warmup_cosine \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --save-interval 500 \
  --seed 1337 \
  --max-total-seq-len 4096 \
  --text-chunk-time-mode start \
  --audio-loader whisper \
  --audio-feature-mode episode_contiguous \
  --num-workers "$NUM_WORKERS" \
  --output-dir "$OUTPUT_DIR"
