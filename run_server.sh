#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_KIND="${1:-smoke}"
if [[ "$RUN_KIND" == smoke ]]; then
  CANDIDATES=1
  EXTRA=(--smoke)
elif [[ "$RUN_KIND" == full ]]; then
  CANDIDATES=512
  EXTRA=()
else
  echo "Usage: bash run_server.sh smoke|full" >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS=16
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
OUTPUT="$ROOT/results/${RUN_KIND}_$(date +%Y%m%d_%H%M%S)_$$"
"$ROOT/.venv/bin/python" "$ROOT/run_inference.py" \
  --repo "$ROOT/upstream" \
  --checkpoint "$ROOT/resources/checkpoints/score_model.pth" \
  --dataset "$ROOT/resources/data/datasets/dataset_dental_vali_128.pkl" \
  --polygons "$ROOT/resources/data/polygons" \
  --extensions "$ROOT/extensions" \
  --sample-index 0 --candidates "$CANDIDATES" --steps 128 \
  --output "$OUTPUT" "${EXTRA[@]}"
echo "Results: $OUTPUT"

