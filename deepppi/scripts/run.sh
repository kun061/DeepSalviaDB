#!/usr/bin/env bash
set -euo pipefail
PPI_PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PPI_PACKAGE_ROOT"
PPI_PACKAGE_PYTHON="${PPI_PACKAGE_PYTHON:-python}"
PPI_PACKAGE_MODE="${1:-verify}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ "$PPI_PACKAGE_MODE" == verify ]]; then
  exec "$PPI_PACKAGE_PYTHON" scripts/verify_package.py "$@"
fi
PPI_PACKAGE_GPU_LIBS="$("$PPI_PACKAGE_PYTHON" -c 'import glob,site;print(":".join(sorted(p for s in site.getsitepackages() for p in glob.glob(s+"/nvidia/*/lib"))))')"
if [[ -n "$PPI_PACKAGE_GPU_LIBS" ]]; then
  export LD_LIBRARY_PATH="$PPI_PACKAGE_GPU_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
PPI_PACKAGE_NVCC="$("$PPI_PACKAGE_PYTHON" -c 'import glob,site;print(":".join(sorted(p for s in site.getsitepackages() for p in glob.glob(s+"/nvidia/cuda_nvcc/bin"))))')"
if [[ -n "$PPI_PACKAGE_NVCC" ]]; then export PATH="$PPI_PACKAGE_NVCC:$PATH"; fi
export TF_DETERMINISTIC_OPS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TF_CPP_MIN_LOG_LEVEL=2
case "$PPI_PACKAGE_MODE" in
  evaluate) exec "$PPI_PACKAGE_PYTHON" -u scripts/reproduce.py "$@" ;;
  train) exec "$PPI_PACKAGE_PYTHON" -u training/train.py --prepared-dir data/D --protocols protein_disjoint --seeds 42 43 44 --epochs 30 --patience 5 --batch-size 128 --bootstrap 1000 --output-dir runs/retrain_D "$@" ;;
  smoke) mkdir -p runs; exec "$PPI_PACKAGE_PYTHON" -u training/train.py --smoke-only --smoke-output runs/gpu_smoke.json "$@" ;;
  *) printf 'Usage: bash scripts/run.sh {verify|smoke|evaluate|train}\n' >&2; exit 2 ;;
esac
