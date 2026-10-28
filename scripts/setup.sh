#!/usr/bin/env bash
# Fetch tau2-bench at the pinned commit and install everything into .venv.
set -euo pipefail
cd "$(dirname "$0")/.."

TAU2_COMMIT=122e2432b82d740e8f313f46f0a071729c404f39
if [ ! -d third_party/tau2-bench ]; then
  mkdir -p third_party
  git clone https://github.com/sierra-research/tau2-bench.git third_party/tau2-bench
fi
git -C third_party/tau2-bench fetch --depth 1 origin "$TAU2_COMMIT"
git -C third_party/tau2-bench checkout --quiet "$TAU2_COMMIT"

uv venv --python 3.12
uv pip install -e ".[dev]"
