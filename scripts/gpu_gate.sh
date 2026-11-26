#!/usr/bin/env bash
# GPU launch gate for shell launchers: refuses an ACCIDENTAL GPU launch beside a running GPU job.
# Rule, override and exit codes: chess_anti_engine/utils/gpu_gate.py (module docstring) and
# docs/operations.md "GPU launch gate". Not a security control.
#
# Usage, from a launcher, before its first GPU child:
#   _gate=$(bash scripts/gpu_gate.sh --tool NAME --for-pid $$ [--allow-busy-gpu "<reason>"]) || exit $?
#   [ -z "$_gate" ] || export "$_gate"    # CAE_GPU_GATE_PASSED=<pid>:<tool>: children pass as "OK inherited"
# Exit 0 pass (prints the marker on stdout, or nothing when cpu-only), 8 refused, 2 usage error.
set -u
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec env PYTHONPATH="$here/..${PYTHONPATH:+:$PYTHONPATH}" "${PYTHON:-python3}" -m chess_anti_engine.utils.gpu_gate "$@"
