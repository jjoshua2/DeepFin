#!/bin/bash
# Cadenced FEN/seed monitor for the LIVE trial (managed by scripts/train.sh).
#
# Two cadences (see READ_EVERY / RETIRE_EVERY below): every RETIRE_EVERY
# checkpoints the cheap flywheel runs (seed retire/probation via
# scripts/blindspot_retire_step.py — auto-retire 2x-AWARE seeds, re-feed retirees
# reading > -0.2 — then the harvest gate + auto-feed); every READ_EVERY the
# expensive yardsticks also run (panel v1/v2 severity + value_regret, each paired
# vs the banked 512-swap baselines). One summary line per cycle.
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
cd "$(dirname "$0")/.." || exit 2
REPO_ROOT="$PWD"
PIDFILE=/tmp/chess_training.pid
WORK_DIR="${TRAIN_WORK_DIR:-runs/pbt2_small}"
MON=scratchpad/live_read/monitor
STATE="$MON/monitor_last_read"          # holds "<trial_basename> <iter>"
# Two decoupled cadences (in checkpoints). The cheap flywheel — retire (net_q,
# no SF) + gate (2M-SF vet of NEW candidates) + feed — runs every RETIRE_EVERY so
# learned seeds clear and new holes feed fast (tenure floor is 2x-consecutive
# AWARE, so a faster cadence retires learned seeds in ~2 reads instead of waiting
# on the slow one; probation re-feed is the safety net for the tighter
# confirmation). The expensive yardsticks (panels + value_regret — frozen-set GPU
# reads, no live SF) run every READ_EVERY to JUDGE experiments over day-plus
# windows. Keep READ_EVERY a multiple of RETIRE_EVERY (deep reads piggyback a
# flywheel cycle on the same checkpoint copy).
READ_EVERY="${MONITOR_READ_EVERY:-20}"
RETIRE_EVERY="${MONITOR_RETIRE_EVERY:-1}"   # checkpoints ~= iterations (~44min each);
                                            # 1 = run the flywheel every iteration.
# Banked 512-swap baselines (canary_512_iter20). Trend anchors only; the
# pre-committed readout rules live in docs/experiment_ledger.md.
BASE=scratchpad/canary_512_iter20
# Harvest-gate vetting (PR #182): CPU Stockfish + syzygy for deep-SF vetting of
# harvested severe seeds. Same binary/TB the miner uses; skipped if absent. Vetted
# survivors are promoted after each gate pass through the safe versioned feed step.
#
# ⚑⚑ NEITHER OF THESE IS IN THE CHECKOUT, so neither may be derived from
# $REPO_ROOT. `e2e_server/publish/` is untracked runtime output and
# `data/syzygy_*` is 151G of tablebases; both exist only in the checkout that
# published them, so a $REPO_ROOT-relative value is correct in the live tree and
# empty in every worktree and fresh clone. PR #441 made both of them
# $REPO_ROOT-relative. The binary at least fails visibly (the `-x` test below
# skips the gate); the TABLEBASES DO NOT — Stockfish accepts a nonexistent
# SyzygyPath, prints one `info string Found 0 WDL and 0 DTZ` into a redirected
# log, and answers `readyok`. Both now come from the shared discovery, which
# falls back to the MAIN checkout via `git rev-parse --git-common-dir`.
SF_BIN="${HARVEST_SF_BIN:-$(PYTHONPATH=. python3 -m chess_anti_engine.utils.engine_discovery 2>/dev/null)}"
SF_BIN="${SF_BIN:-$REPO_ROOT/e2e_server/publish/stockfish}"
# Match the in-loop label's TB coverage (games search 3-4-5 + 6-man, see
# pbt2_small.yaml syzygy_path) so the gate is never TB-blind where the ~700k
# label has exact truth — the gate must dominate the label on every axis (depth
# 2M>700k, MultiPV-1 concentrated, TB equal) to legitimately overrule it.
#
# ⚑ That claim is now ENFORCED rather than asserted: harvest_gate_step.py exits
# non-zero when any component of this path holds no .rtbw/.rtbz, so a TB-blind
# cycle produces `harvest_gate: FAILED (...)` in the monitor line instead of a
# quietly worse number. `--allow-missing-tablebases` is the deliberate opt-out.
SYZYGY_PATH="${HARVEST_SYZYGY_PATH:-$(PYTHONPATH=. python3 -m chess_anti_engine.utils.syzygy 2>/dev/null)}"
SYZYGY_PATH="${SYZYGY_PATH:-$REPO_ROOT/data/syzygy_3-4-5:$REPO_ROOT/data/syzygy_6}"
STAGED_SEEDS="${HARVEST_STAGED_SEEDS:-data/harvest/staged_candidates.txt}"
AUTO_FEED_DISABLED="$MON/auto_feed_disabled"
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
    # STATE: "<trial> <last_retire_ck> <last_deep_ck>". Older 2-field files
    # ("<trial> <ck>") load with an empty deep field -> backfilled below.
    read -r LAST_TRIAL LAST_RETIRE_N LAST_DEEP_N < "$STATE" 2>/dev/null || { LAST_TRIAL=""; LAST_RETIRE_N=-999; LAST_DEEP_N=-999; }
    [ -n "$LAST_DEEP_N" ] || LAST_DEEP_N="$LAST_RETIRE_N"
    # New trial dir (swap/salvage-restart) => reset both counters.
    [ "$(basename "$TRIAL")" = "$LAST_TRIAL" ] || { LAST_RETIRE_N=-999; LAST_DEEP_N=-999; }
    # Nothing due until the (faster) retire/flywheel cadence elapses.
    if [ $((N - LAST_RETIRE_N)) -lt "$RETIRE_EVERY" ]; then sleep 300; continue; fi

    sleep 120  # let the checkpoint write settle
    trainer_running || continue
    cp "$CK/trainer.pt" "$MON/ck_$N.pt" 2>/dev/null || { sleep 600; continue; }

    # Deep yardsticks (panels + value_regret) run only every READ_EVERY. They are
    # frozen-reference GPU reads (no live SF — value_regret scores against the
    # pre-labeled audit set) used to JUDGE experiments over day-plus windows, so
    # they lag the flywheel. Piggyback the same checkpoint copy.
    DO_DEEP=0
    [ $((N - LAST_DEEP_N)) -ge "$READ_EVERY" ] && DO_DEEP=1
    B1=""; B2=""; VAL=""; DELTA=""
    if [ "$DO_DEEP" = 1 ]; then
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
        B1=$(grep -oE "BLIND \(net > -0.2\): [0-9]+/35" "$MON/panel_v1_$N.log" | tail -1)
        B2=$(grep -oE "BLIND \(net > -0.2\): [0-9]+/113" "$MON/panel_v2_$N.log" | tail -1)
        VAL=$(grep OVERALL "$MON/vregret_$N.log" | tail -1 | xargs)
        DELTA=$(grep "paired delta" "$MON/paired_${N}_vs_boot.log" | xargs)
        LAST_DEEP_N=$N
    fi

    # Seed retire + probation re-feed (PR #155): rewrites the active list +
    # repoints the live yaml, validated + reverted-on-error; fail-soft. Net-only
    # (net_q, no SF) — cheap, so it rides the fast RETIRE_EVERY cadence.
    RET=$(PYTHONPATH=. nice -n 15 python3 scripts/blindspot_retire_step.py \
        --checkpoint "$MON/ck_$N.pt" --tag "$N" --gpu-mem-fraction 0.15 \
        2>>"$MON/retire_$N.log" | grep '^retire:' | tail -1)

    # Auto-mine gate (PR #182): vet a bounded batch of NEW harvester severe seeds
    # (collect flywheel's middle) and stage survivors. CPU Stockfish (nice, capped),
    # safe concurrent with training; fail-soft.
    GATE=""
    if [ -x "$SF_BIN" ]; then
        GATE=$(PYTHONPATH=. nice -n 15 python3 scripts/harvest_gate_step.py \
            --sf-path "$SF_BIN" --syzygy-path "$SYZYGY_PATH" \
            --max-vet-per-run "${HARVEST_VET_PER_RUN:-50}" \
            --sf-nodes "${HARVEST_SF_NODES:-2000000}" --multipv "${HARVEST_MULTIPV:-1}" \
            --stamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            2>>"$MON/harvest_gate_$N.log" | grep '^harvest_gate:' | tail -1)
    fi

    # Close the flywheel at the same checkpoint boundary: merge all staged,
    # deep-SF-vetted candidates that are neither active nor retired, write a fresh
    # path (the loader cache is path-keyed), and atomically repoint + validate the
    # live yaml. The feed step is idempotent against the cumulative staging file.
    FEED=""
    if [ -s "$STAGED_SEEDS" ] && [ ! -f "$AUTO_FEED_DISABLED" ]; then
        FEED_TAG="auto_ck${N}_$(date -u +%Y%m%dT%H%M%SZ)"
        FEED=$(PYTHONPATH=. nice -n 15 python3 scripts/blindspot_feed_step.py \
            --batch "$STAGED_SEEDS" --tag "$FEED_TAG" \
            2>>"$MON/feed_$N.log" | tail -1)
    elif [ -f "$AUTO_FEED_DISABLED" ]; then
        FEED="[feed] disabled by $AUTO_FEED_DISABLED"
    fi

    # Full yardstick line on deep cycles; compact flywheel line otherwise.
    if [ "$DO_DEEP" = 1 ]; then
        echo "[monitor $(date +%m-%d\ %H:%M)] trial=$(basename "$TRIAL") ckpt=$N | v1 $B1 | v2 $B2 | $VAL | vs_boot: $DELTA | $RET | $GATE | $FEED" >> "$MON/monitor.log"
    else
        echo "[monitor $(date +%m-%d\ %H:%M)] trial=$(basename "$TRIAL") ckpt=$N | flywheel | $RET | $GATE | $FEED" >> "$MON/monitor.log"
    fi
    rm -f "$MON/ck_$N.pt"
    echo "$(basename "$TRIAL") $N $LAST_DEEP_N" > "$STATE"
done
