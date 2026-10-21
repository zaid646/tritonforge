#!/usr/bin/env bash
# Run NVIDIA compute-sanitizer over a single launch of each kernel.
# Writes raw logs into artifacts/sanitizer/ for the report.
set -uo pipefail

SAN="${SAN:-compute-sanitizer}"
PY="${PY:-python}"
OUT="${OUT:-artifacts/sanitizer}"
mkdir -p "$OUT"

if ! command -v "$SAN" >/dev/null 2>&1; then
  echo "compute-sanitizer not found; set SAN=/path/to/compute-sanitizer" >&2
  exit 2
fi

status=0
for tool in memcheck racecheck synccheck; do
  log="$OUT/${tool}.log"
  echo "=== compute-sanitizer --tool ${tool} ==="
  "$SAN" --tool "$tool" --target-processes all \
    "$PY" tests/sanitizer_run.py "$tool" > "$log" 2>&1
  code=$?
  tail -n 20 "$log"
  if [ "$code" -ne 0 ]; then
    echo "tool ${tool} exited ${code}"
    status=1
  fi
done

echo "logs in $OUT"
exit "$status"
