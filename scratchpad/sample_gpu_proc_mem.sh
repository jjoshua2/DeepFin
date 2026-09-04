#!/usr/bin/env bash
# Non-invasive half of the allocator diagnostic: sample per-PROCESS device memory
# over time. This is RESERVED-side (what the process holds from the driver), not
# torch's `memory_allocated`, so it cannot separate fragmentation from retention
# -- but FLAT vs GROWING already discriminates one hypothesis, and it needs no
# code change to the live trial.
set -uo pipefail
OUT=scratchpad/gpu_proc_mem.tsv
echo -e "epoch\tpid\tused_mib\ttotal_used_mib\tsm_clock\ttemp" > "$OUT"
for _ in $(seq 1 12); do
  TS=$(date +%s)
  TOT=$(nvidia-smi --query-gpu=memory.used,clocks.sm,temperature.gpu --format=csv,noheader,nounits | head -1)
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null |
    while IFS=, read -r pid mem; do
      echo -e "${TS}\t$(echo $pid|tr -d ' ')\t$(echo $mem|tr -d ' ')\t${TOT// /}"
    done >> "$OUT"
  sleep 60
done
echo "done" >> "$OUT"
