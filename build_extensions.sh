#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
mkdir -p "$ROOT/native_build" "$ROOT/extensions" "$ROOT/environment_evidence"
cp "$ROOT"/upstream/external/rmspacing_cpp/* "$ROOT/native_build/"
# Normalize only the build copies; preserve the pinned upstream checkout.
sed -i 's/\r$//' "$ROOT/native_build/build_util.sh" "$ROOT/native_build/build.sh"
export PATH="$ROOT/.venv/bin:$PATH"
# python3-config may be the system Python 3.10 while this environment is 3.11.
# Get both includes and ABI suffix from the actual interpreter used for inference.
read -r -a PY_INCLUDES <<< "$("$PYTHON" -m pybind11 --includes)"
PY_SUFFIX="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
COMMON_FLAGS=(-std=c++17 -O3 -march=native -finline-functions -funroll-loops -Wall -fPIC -fopenmp)
cd "$ROOT/native_build"
"${CXX:-g++}" "${COMMON_FLAGS[@]}" "${PY_INCLUDES[@]}" \
  util.cpp clipper.engine.cpp clipper.offset.cpp clipper.rectclip.cpp \
  -shared -o "calutil${PY_SUFFIX}"
"${CXX:-g++}" "${COMMON_FLAGS[@]}" "${PY_INCLUDES[@]}" \
  main.cpp clipper.engine.cpp clipper.offset.cpp clipper.rectclip.cpp \
  -shared -o "rmspacing${PY_SUFFIX}"
cp "calutil${PY_SUFFIX}" "rmspacing${PY_SUFFIX}" "$ROOT/extensions/"
