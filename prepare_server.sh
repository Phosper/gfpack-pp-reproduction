#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PYTHON="${BASE_PYTHON:-python3}"
if [[ -e "$ROOT/.venv" ]]; then
  echo "Refusing to recreate existing environment: $ROOT/.venv" >&2
  exit 1
fi
"$BASE_PYTHON" -m venv --system-site-packages "$ROOT/.venv"
PYTHON="$ROOT/.venv/bin/python"
"$PYTHON" -m pip install --disable-pip-version-check --no-input \
  torch-geometric==2.6.1 pybind11==2.13.6 shapely==2.0.7 tensorboardX==2.6.2.2

bash "$ROOT/build_extensions.sh"

"$PYTHON" -m pip freeze > "$ROOT/environment_evidence/pip_freeze.txt"
"$PYTHON" -c 'import torch,torch_geometric,shapely,pybind11,tensorboardX; print(torch.__version__,torch.version.cuda,torch_geometric.__version__,shapely.__version__,pybind11.__version__)'
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total,memory.free,utilization.gpu --format=csv > "$ROOT/environment_evidence/gpu_before.csv"
g++ --version > "$ROOT/environment_evidence/compiler.txt"

