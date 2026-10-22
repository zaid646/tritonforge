#!/usr/bin/env bash
# Run every decoder backend at batch 1 and 8 on the pinned CPUs.
# Usage: RUNS=10 BATCHES="1 8" scripts/run_x_benchmarks.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS=${RUNS:-10}
WARMUP=${WARMUP:-2}
BATCHES=${BATCHES:-"1 8"}
OUT=${OUT:-results/x/bench}
ORT_PYTHON=${ORT_PYTHON:-/content/venv_ort/bin/python}
MODELS=${MODELS:-/content/models}
BACKENDS=${BACKENDS:-"pytorch_eager torch_compile onnxruntime ov_fp16 ov_int8 ov_int4_g128 ov_int4_awq_g128"}

mkdir -p "$OUT" artifacts/x/logs
for batch in $BATCHES; do
  for backend in $BACKENDS; do
    python_bin=python3
    if [ "$backend" = "onnxruntime" ]; then
      python_bin=$ORT_PYTHON
    fi
    if [[ "$backend" == ov_* ]] && [ ! -d "$MODELS/ov_${backend#ov_}" ]; then
      echo "skip $backend (model missing)"
      continue
    fi
    echo "=== $backend batch=$batch ==="
    TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/content/torch_cache} \
      taskset -c 0,1 "$python_bin" scripts/bench_backends.py \
        --backend "$backend" \
        --batch-size "$batch" \
        --runs "$RUNS" \
        --warmup "$WARMUP" \
        --models-dir "$MODELS" \
        --out-dir "$OUT" 2>&1 | tee "artifacts/x/logs/bench_${backend}_b${batch}.log"
  done
done
