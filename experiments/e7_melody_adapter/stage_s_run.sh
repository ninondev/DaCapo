#!/usr/bin/env bash
# Run Stage S from the checkout root. GPU stages are for the owner's runtime.
set -euo pipefail
e7_here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
e7_root=$(git -C "$e7_here" rev-parse --show-toplevel)
cd "$e7_root"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$e7_root${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$e7_here/outputs/cache/matplotlib}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$e7_here/outputs/cache/numba}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
e7_python=${E7_PYTHON:-python}
stage=${1:-help}
if [ "$#" -gt 0 ]; then shift; fi
case "$stage" in
  pack)
    "$e7_python" -m experiments.e7_melody_adapter.registration --check
    exec "$e7_python" -m experiments.e7_melody_adapter.pack render "$@"
    ;;
  suno) exec "$e7_python" -m experiments.e7_melody_adapter.suno "$@" ;;
  check)
    "$e7_python" -m experiments.e7_melody_adapter.registration --check "$@"
    exec bash "$e7_here/precommit_check.sh"
    ;;
  help|-h|--help)
    echo 'Usage: bash experiments/e7_melody_adapter/stage_s_run.sh <pack|suno|check> [arguments]'
    echo 'pack: render the upload pack on CPU; suno: score on a prepared CUDA runtime.'
    ;;
  *) echo "Unknown stage: $stage. H1 training stages are not part of this delivery." >&2; exit 2 ;;
esac
