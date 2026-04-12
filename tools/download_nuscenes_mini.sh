#!/usr/bin/env bash
# Lightweight helper to download nuScenes mini (v1.0-mini).
# Requires a nuScenes API token from https://www.nuscenes.org (login → My Account → Create API Token).
# Usage: NUSC_TOKEN=your_token_here ./tools/download_nuscenes_mini.sh

set -euo pipefail

if [[ -z "${NUSC_TOKEN:-}" ]]; then
  echo "[error] Please set NUSC_TOKEN env var with your nuScenes API token." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/data/nuscenes"

mkdir -p "${TARGET_DIR}"

python -m nuscenes.scripts.download \
  -d "${TARGET_DIR}" \
  -v v1.0-mini \
  -k "${NUSC_TOKEN}" \
  --overwrite True

echo "[ok] nuScenes mini downloaded to ${TARGET_DIR}"