#!/usr/bin/env bash
# Full GPU suite: correctness, sanitizer, kernel benchmark, e2e, report.
# Run this on the GPU host from the repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src

echo "== environment =="
python -c "import torch, triton, platform; print('torch', torch.__version__, 'triton', triton.__version__, 'py', platform.python_version())"
python -c "from tritonforge import env; print(env.describe())"

echo "== correctness =="
python -m pytest tests/ -q

echo "== compute-sanitizer =="
SAN="${SAN:-/usr/local/cuda/bin/compute-sanitizer}" ./scripts/sanitize.sh || echo "sanitizer reported issues"

echo "== kernel benchmark =="
python benchmarks/run_kernels.py

echo "== end-to-end benchmark =="
python benchmarks/run_e2e.py

echo "== report =="
python scripts/make_report.py

echo "== regression gate =="
python scripts/check_regression.py
