#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10 libero_90}"

for suite in $SUITES; do
  echo "[train-depth-all] starting $suite"
  SUITE="$suite" bash "$SCRIPT_DIR/train_lapa_depth_suite.sh"
done
