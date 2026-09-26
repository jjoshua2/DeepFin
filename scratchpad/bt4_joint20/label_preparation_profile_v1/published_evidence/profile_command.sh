#!/usr/bin/env bash
set -euo pipefail
STATE=/home/josh/projects/chess/scratchpad/bt4_joint20/label_preparation_profile_v1
mkdir "$STATE/run"
cd /home/josh/projects/chess/.dev/worktree/wise-cloud
export CUDA_VISIBLE_DEVICES='' PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 BLOSC_NTHREADS=2
set +e
/usr/bin/time -o "$STATE/run/resources.txt" -v nice -n 19 ionice -c 3 taskset -c 0 \
  /usr/bin/timeout --signal=TERM --kill-after=5s 115s /usr/bin/python3 \
  "$STATE/profile_preparation.py" --plan "$STATE/plan.json" > "$STATE/run/execution.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$STATE/run/exit_code.txt"
exit "$rc"
