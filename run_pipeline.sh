#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${1:?Usage: bash run_pipeline.sh NEW_OUTPUT_DIRECTORY}"
shift
STATUS=0
"$ROOT/.venv/bin/python" -u "$ROOT/run_benchmark.py" --root "$ROOT" --output "$OUTPUT" "$@" || STATUS=$?
"$ROOT/.venv/bin/python" "$ROOT/verify_benchmark.py" "$OUTPUT"
exit "$STATUS"
