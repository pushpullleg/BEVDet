#!/usr/bin/env bash
# Download a BEVDet checkpoint to checkpoints/.
# Usage: CHECKPOINT_URL=... ./tools/download_bevdet_checkpoint.sh

set -euo pipefail

if [[ -z "${CHECKPOINT_URL:-}" ]]; then
  echo "[error] Please set CHECKPOINT_URL to the .pth URL (e.g., BEVDet-R50)." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/checkpoints"
mkdir -p "${TARGET_DIR}"

FILE_NAME="${CHECKPOINT_URL##*/}"
DEST_PATH="${TARGET_DIR}/${FILE_NAME}"

echo "[info] Downloading ${CHECKPOINT_URL} -> ${DEST_PATH}"
curl -fL "${CHECKPOINT_URL}" -o "${DEST_PATH}"

echo "[ok] Saved to ${DEST_PATH}"