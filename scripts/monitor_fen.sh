#!/bin/bash
# Cadenced FEN/seed monitor for the LIVE trial (managed by scripts/train.sh).
#
# Every checkpoint >= $READ_EVERY iters past the last read: panel v1/v2 severity
# + value_regret (each paired vs the banked 512-swap baselines), then the seed
# retire/probation step (scripts/blindspot_retire_step.py — auto-retire 2x-AWARE
# seeds, re-feed retirees reading > -0.2). One summary line per read.
#
# Replaces the scratchpad-era scratchpad/live_read/monitor_fen.sh, which was
# hardwired to the retired 46M trial dir (5fac4) and a 46M-era iteration
# counter — after the 2026-07-11 512 swap it silently never fired again. This
# version auto-detects the NEWEST trial dir each cycle and keeps its last-read
# iteration in a state file, so trial swaps/restarts can't orphan it.
#
# Compute discipline: does NOTHING (no checkpoint copy, no GPU work) unless the
# trainer is actually running — seed logic must not burn GPU/CPU while training
# is stopped for a match or maintenance.
set -u
cd /home/josh/projects/chess
PIDFILE=/tmp/chess_training.pid
WORK_DIR="${TRAIN_WORK_DIR:-runs/pbt2_small}"
MON=scratchpad/live_read/monitor
STATE="$MON/monitor_last_read"          # holds "<trial_basename> <iter>"
READ_EVERY="${MONITOR_READ_EVERY:-10}"
# Banked 512-swap baselines (canary_512_iter20). Trend anchors only; the
# pre-committed readout rules live in docs/experiment_ledger.md.
BASE=scratchpad/canary_512_iter20
# Harvest-gate vetting (PR #182): CPU Stockfish + syzygy for deep-SF vetting of
# harvested severe seeds. Same binary/TB the miner uses; skipped if absent.
SF_BIN="${HARVEST_SF_BIN:-/home/josh/projects/chess/e2e_server/publish/stockfish}"
SYZYGY_PATH="${HARVEST_SYZYGY_PATH:-/home/josh/projects/chess/data/syzygy_3-4-5}"
mkdir -p "$MON"

trainer_running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null
}

while true; do
    if ! trainer_running; then sleep 300; continue; fi
    TRIAL=$(ls -dt "$WORK_DIR"/tune/train_trial_* 2>/dev/null | head -1)
    [ -n "$TRIAL" ] || { sleep 300; continue; }
    CK=$(ls -d "$TRIAL"/checkpoint_* 2>/dev/null | sort | tail -1)
    [ -n "$CK" ] && [ -f "$CK/trainer.pt" ] || { sleep 300; continue; }
    N=$((10#$(basename "$CK" | tr -dc '0-9')))
    read -r LAST_TRIAL LAST_N < "$STATE" 2>/dev/null || { LAST_TRIAL=""; LAST_N=-999; }
    # New trial dir (swap/salvage-restart) => reset the counter.
    [ "$(basename "$TRIAL")" = "$LAST_TRIAL" ] || LAST_N=-999
    if [ $((N - LAST_N)) -lt "$READ_EVERY" ]; then sleep 300; continue; fi

    sleep 120  # let the checkpoint write settle
    trainer_running || continue
    cp "$CK/trainer.pt" "$MON/ck_$N.pt" 2>/dev/null || { sleep 600; continue; }

    for P in v1 v2; do
        PYTHONPATH=. nice -n 15 python3 scripts/blindspot_panel.py --checkpoint "$MON/ck_$N.pt" \
            --panel "data/blindspot_panel_$P.jsonl" \
            --dump-per-position "$MON/paneldump_${P}_$N.jsonl" > "$MON/panel_${P}_$N.log" 2>&1
        PYTHONPATH=. python3 scripts/paired_compare.py \
            "$BASE/panel_${P}_live.jsonl" "$MON/paneldump_${P}_$N.jsonl" \
            --label-a swap_iter20 --label-b "ck_$N" > "$MON/pairpanel_${P}_${N}.log" 2>&1
    done
    PYTHONPATH=. nice -n 15 python3 scripts/value_regret.py --checkpoint "$MON/ck_$N.pt" \
        --max-positions 2000 --gpu-mem-fraction 0.15 \
        --dump-per-position "$MON/vdump_$N.jsonl" > "$MON/vregret_$N.log" 2>&1
    PYTHONPATH=. python3 scripts/paired_compare.py "$BASE/vdump_boot_swaptime.jsonl" \
        "$MON/vdump_$N.jsonl" --label-a boot512 --label-b "ck_$N" \
        > "$MON/paired_${N}_vs_boot.log" 2>&1

    # Seed retire + probation re-feed (PR #155): rewrites the active list +
    # repoints the live yaml, validated + reverted-on-error; fail-soft.
    RET=$(PYTHONPATH=. nice -n 15 python3 scripts/blindspot_retire_step.py \
        --checkpoint "$MON/ck_$N.pt" --tag "$N" --gpu-mem-fraction 0.15 \
        2>>"$MON/retire_$N.log" | grep '^retire:' | tail -1)

    # Auto-mine gate (PR #182): vet a bounded batch of NEW harvester severe seeds
    # (collect flywheel's middle) → STAGE survivors to data/harvest/staged_candidates.txt.
    # Staging only — promotion into the live pool stays a human/ledger-gated feed.
    # CPU Stockfish (nice, capped), safe concurrent with training; fail-soft.
    GATE=""
    if [ -x "$SF_BIN" ]; then
        GATE=$(PYTHONPATH=. nice -n 15 python3 scripts/harvest_gate_step.py \
            --sf-path "$SF_BIN" --syzygy-path "$SYZYGY_PATH" \
            --max-vet-per-run "${HARVEST_VET_PER_RUN:-30}" --stamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            2>>"$MON/harvest_gate_$N.log" | grep '^harvest_gate:' | tail -1)
    fi

    B1=$(grep -oE "BLIND \(net > -0.2\): [0-9]+/35" "$MON/panel_v1_$N.log" | tail -1)
    B2=$(grep -oE "BLIND \(net > -0.2\): [0-9]+/113" "$MON/panel_v2_$N.log" | tail -1)
    VAL=$(grep OVERALL "$MON/vregret_$N.log" | tail -1 | xargs)
    DELTA=$(grep "paired delta" "$MON/paired_${N}_vs_boot.log" | xargs)
    echo "[monitor $(date +%m-%d\ %H:%M)] trial=$(basename "$TRIAL") ckpt=$N | v1 $B1 | v2 $B2 | $VAL | vs_boot: $DELTA | $RET | $GATE" >> "$MON/monitor.log"
    rm -f "$MON/ck_$N.pt"
    echo "$(basename "$TRIAL") $N" > "$STATE"
done
