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

for name in BASE_MODEL_DIR ARCH_CHECKPOINT POSTTRAIN_INDEX COSPEECH_INDEX IDLE_MOTION_PKL; do
  require_env "$name"
done

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/stage1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NORM_STATS="${NORM_STATS:-${OUTPUT_DIR}/norm_stats_precompute/norm_stats.json}"

COMMON_ARGS=(
  --checkpoint-dir "$BASE_MODEL_DIR"
  --init-from "$ARCH_CHECKPOINT"
  --train-index "$POSTTRAIN_INDEX"
  --cospeech-train-index "$COSPEECH_INDEX"
  --cospeech-baseline-index "$COSPEECH_INDEX"
  --idle-motion-pkl "$IDLE_MOTION_PKL"
  --posttrain-episode-batch-size 2
  --cospeech-episode-batch-size 2
  --cospeech-subset-size 0
  --listen-action-loss-weight 2.0
  --cospeech-speaking-loss-weight 1.0
  --cospeech-listening-loss-weight 0.0
  --cospeech-idle-loss-weight 1.0
  --epochs 60
  --lr 1e-4
  --min-lr 1e-5
  --lr-warmup-steps 500
  --lr-schedule warmup_cosine
  --weight-decay 0.01
  --grad-clip 1.0
  --save-interval 1000
  --seed 1337
  --max-total-seq-len 4096
  --text-chunk-time-mode start
  --audio-loader whisper
  --audio-feature-mode episode_contiguous
  --num-workers "$NUM_WORKERS"
)

if [[ ! -f "$NORM_STATS" ]]; then
  mkdir -p "$(dirname "$NORM_STATS")"
  python "$PROJECT_ROOT/scripts/hri_mot/train_hri_mot_posttrain.py" \
    "${COMMON_ARGS[@]}" \
    --norm-stats-mode joint_supervised \
    --compute-norm-stats-only \
    --output-dir "$(dirname "$NORM_STATS")"
fi

torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" \
  "$PROJECT_ROOT/scripts/hri_mot/train_hri_mot_posttrain.py" \
  "${COMMON_ARGS[@]}" \
  --norm-stats-mode joint_supervised \
  --norm-stats "$NORM_STATS" \
  --output-dir "$OUTPUT_DIR"
