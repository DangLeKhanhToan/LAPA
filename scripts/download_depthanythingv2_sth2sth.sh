#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$( cd -- "$( dirname -- "$SCRIPT_DIR" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"

DEPTH_ANYTHING_REPO_DIR="${DEPTH_ANYTHING_REPO_DIR:-$PROJECT_DIR/third_party/depth_anything_v2}"
DEPTH_ANYTHING_REPO_URL="${DEPTH_ANYTHING_REPO_URL:-https://github.com/DepthAnything/Depth-Anything-V2.git}"
DEPTH_ANYTHING_CHECKPOINT_DIR="${DEPTH_ANYTHING_CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints/depth_anything_v2_sth2sth}"
DEPTH_ANYTHING_CHECKPOINT="${DEPTH_ANYTHING_CHECKPOINT:-$DEPTH_ANYTHING_CHECKPOINT_DIR/depth_anything_v2_sth2sth.pth}"

mkdir -p "$(dirname "$DEPTH_ANYTHING_REPO_DIR")" "$DEPTH_ANYTHING_CHECKPOINT_DIR"

if [[ ! -d "$DEPTH_ANYTHING_REPO_DIR/.git" ]]; then
  echo "[depthanything] cloning $DEPTH_ANYTHING_REPO_URL -> $DEPTH_ANYTHING_REPO_DIR"
  git clone "$DEPTH_ANYTHING_REPO_URL" "$DEPTH_ANYTHING_REPO_DIR"
else
  echo "[depthanything] repo already exists: $DEPTH_ANYTHING_REPO_DIR"
fi

if [[ -n "${DEPTH_ANYTHING_CKPT_LOCAL:-}" ]]; then
  echo "[depthanything] copying local Sth2Sth checkpoint"
  cp "$DEPTH_ANYTHING_CKPT_LOCAL" "$DEPTH_ANYTHING_CHECKPOINT"
elif [[ -n "${DEPTH_ANYTHING_CKPT_URL:-}" ]]; then
  echo "[depthanything] downloading Sth2Sth checkpoint -> $DEPTH_ANYTHING_CHECKPOINT"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$DEPTH_ANYTHING_CKPT_URL" -o "$DEPTH_ANYTHING_CHECKPOINT"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$DEPTH_ANYTHING_CHECKPOINT" "$DEPTH_ANYTHING_CKPT_URL"
  else
    echo "ERROR: neither curl nor wget is available." >&2
    exit 1
  fi
else
  cat >&2 <<EOF
ERROR: DepthAnythingV2 Sth2Sth checkpoint source is missing.

Set one of:
  export DEPTH_ANYTHING_CKPT_LOCAL=/path/to/depth_anything_v2_sth2sth.pth
  export DEPTH_ANYTHING_CKPT_URL=https://.../depth_anything_v2_sth2sth.pth

Then rerun:
  bash scripts/download_depthanythingv2_sth2sth.sh
EOF
  exit 1
fi

echo "[depthanything] done"
echo "export DEPTH_ANYTHING_REPO_DIR=\"$DEPTH_ANYTHING_REPO_DIR\""
echo "export DEPTH_ANYTHING_CHECKPOINT=\"$DEPTH_ANYTHING_CHECKPOINT\""
