#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT_ROOT="$ROOT/.worktrees"
ENV_ROOT="$ROOT/.envs"
mkdir -p "$WT_ROOT" "$ENV_ROOT"

ensure_worktree() {
  local branch="$1"
  local dir="$WT_ROOT/$branch"
  if [[ ! -d "$dir" || ! -e "$dir/.git" && ! -f "$dir/.git" ]]; then
    echo "[orch] worktree add $branch -> $dir"
    git -C "$ROOT" worktree add "$dir" "$branch"
  fi
  echo "$dir"
}

run_branch_suite() {
  local branch="$1"
  local suite="$2"
  shift 2

  local dir env_dir
  dir="$(ensure_worktree "$branch")"
  env_dir="$ENV_ROOT/$branch"

  # Rebuild/sync env (conda + uv editable install), using the branch’s own logic
  (cd "$dir" && ENV_DIR="$env_dir" bash scripts/rebuild_env.sh)

  # Run suite inside env
  OUT_DIR="$ROOT/all_results/$branch/$(date +%Y%m%d_%H%M%S)" \
    (cd "$dir" && conda run -p "$env_dir" bash scripts/run_suite.sh "$suite" "$@")
}

# Example: run “profile” suite on two branches
run_branch_suite "perf/e2e-v0.10.1.1" "profile" "$@"
