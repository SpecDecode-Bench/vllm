#!/usr/bin/env bash
set -euo pipefail

# Set the env path
: "${ENV_DIR:?Usage: $0 <env-path>  (or set ENV_DIR, or run from a git repo)}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Create conda env if missing
if [[ ! -d "$ENV_DIR" ]]; then
  echo "[rebuild_env] creating conda env at $ENV_DIR (python=$PYTHON_VERSION)"
  conda create -y -p "$ENV_DIR" "python=$PYTHON_VERSION"
fi

# Ensure uv exists inside the env (safe even if already installed)
echo "[rebuild_env] ensuring uv exists"
conda run -p "$ENV_DIR" python -m pip install -U pip uv

# Install the branch (worktree) in editable mode
echo "[rebuild_env] uv pip install -e ."
conda run -p "$ENV_DIR" uv pip install --editable .