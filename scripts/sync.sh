#!/usr/bin/env bash
# Sync the local repo to the active GPU runtime over the colab SSH alias.
set -euo pipefail

REMOTE="${REMOTE:-colab-t4}"
DEST="${DEST:-/content/tritonforge}"

rsync -az --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.ruff_cache' \
  --exclude 'results/raw' \
  --exclude 'artifacts' \
  -e ssh \
  ./ "${REMOTE}:${DEST}/"

echo "synced -> ${REMOTE}:${DEST}"
