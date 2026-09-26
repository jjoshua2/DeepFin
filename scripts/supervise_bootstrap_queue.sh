#!/bin/bash
# Receipt-driven bounded queue. Never kills a group inferred from an old wrapper PID.
set -euo pipefail
LOOP=${1:?loop directory}
RUNTIME=${2:?arena runtime}
OPERATOR=$(dirname "$(readlink -f "$0")")/bootstrap_experiment_operator.py
exec 9>"$LOOP/supervisor.lock"
flock -n 9 || exit 0
while true; do
  say=$(python3 "$OPERATOR" --loop-dir "$LOOP" --runtime "$RUNTIME")
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$say" >> "$LOOP/supervisor.log"
  case "$say" in
    DEADLINE*|QUEUE_IDLE*) exit 0 ;;
  esac
  # Terminal receipts alone allow harvest. A lost wrapper is an explicit recovery
  # condition; never relaunch its work or signal a possibly reused process group.
  python3 - "$LOOP" <<'PY'
import json, pathlib, sys, time
root = pathlib.Path(sys.argv[1])
queue = json.loads((root / 'queue.json').read_text())
if any(x.get('status') in {'needs_recovery', 'launching'} for x in queue['items']):
    raise SystemExit('Unresolved job: inspect children before continuing')
running = [x for x in queue['items'] if x.get('status') == 'running']
if not running:
    raise SystemExit('No owned running job: resource/probe failure requires review')
deadline = float(json.loads((root / 'STATE.json').read_text())['deadline_unix'])
terminal = pathlib.Path(running[0]['out']) / 'parent_outer_terminal.json'
while not terminal.exists():
    remaining = deadline - time.time()
    if remaining <= 0:
        raise SystemExit('Deadline: no terminal receipt; preserve job for recovery')
    time.sleep(min(30, remaining))
PY
done
