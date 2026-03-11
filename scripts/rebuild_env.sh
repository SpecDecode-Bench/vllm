#!/usr/bin/env bash
set -euo pipefail

: "${ENV_DIR:?Set ENV_DIR to the env path (e.g. /path/to/.envs/branch)}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# Create conda env if missing
if [[ ! -d "$ENV_DIR" ]]; then
  echo "[rebuild_env] creating conda env at $ENV_DIR (python=$PYTHON_VERSION)"
  conda create -y -p "$ENV_DIR" "python=$PYTHON_VERSION"
fi

# Install the branch (worktree) in editable mode
echo "[rebuild_env] python -m pip install --editable . "
conda run -p "$ENV_DIR" python -m pip install --editable .
