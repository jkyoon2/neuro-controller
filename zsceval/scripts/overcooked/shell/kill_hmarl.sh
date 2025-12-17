#!/usr/bin/env bash
set -euo pipefail

# Kill all train_hmarl runs (script + launched Python) across seeds.
# This targets process groups to ensure children are reaped and GPU memory is freed.

main() {
  local patterns=(
    "train_hmarl.py"
    "train_hmarl.sh"
  )

  local pids=()
  for pat in "${patterns[@]}"; do
    while IFS= read -r pid; do
      pids+=("$pid")
    done < <(pgrep -f "$pat" || true)
  done

  # Deduplicate
  readarray -t pids < <(printf "%s\n" "${pids[@]:-}" | sort -u)

  if [[ ${#pids[@]} -eq 0 ]]; then
    echo "No matching train_hmarl processes found."
    exit 0
  fi

  echo "Found PIDs: ${pids[*]}"

  # Kill whole process groups.
  for pid in "${pids[@]}"; do
    pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
    [[ -n "$pgid" ]] || continue
    echo "Sending SIGTERM to process group -$pgid (from PID $pid)"
    kill -TERM "-$pgid" 2>/dev/null || true
  done

  # Wait briefly, then force kill leftovers.
  sleep 3
  for pid in "${pids[@]}"; do
    if ps -p "$pid" >/dev/null 2>&1; then
      pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
      echo "Forcing SIGKILL to process group -$pgid (PID $pid still alive)"
      kill -KILL "-$pgid" 2>/dev/null || true
    fi
  done

  echo "Done. GPU memory should be released once processes exit."
}

main "$@"
