#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SUITE="${1:-run}"
shift || true

OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/$(git rev-parse --short HEAD)/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_DIR"

pick_jobs() {
  case "$SUITE" in
    run)
      find "$SCRIPT_DIR" -maxdepth 1 -type f -name "run-*.sh" -perm -u+x | sort
      ;;
    profile)
      find "$SCRIPT_DIR" -maxdepth 1 -type f -name "profile-*.sh" -perm -u+x | sort
      ;;
    *)
      echo "Unknown suite: $SUITE" >&2
      exit 2
      ;;
  esac
}

mapfile -t JOBS < <(pick_jobs)

echo "[run_suite] suite=$SUITE jobs=${#JOBS[@]} out=$OUT_DIR"

for job in "${JOBS[@]}"; do
  name="$(basename "$job" .sh)"
  log="$OUT_DIR/$name.log"
  echo "[run_suite] >>> $name"
  (cd "$REPO_ROOT" && bash "$job" "$@") |& tee "$log"
done
