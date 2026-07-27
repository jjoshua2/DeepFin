#!/usr/bin/env bash
# Daily strength ratchet — the promotion gate the in-loop one never was.
#
# WHY THIS EXISTS AND NOT `gate_games`. The in-loop gate
# (configs/pbt2_small.yaml gate_games/gate_threshold/gate_mcts_sims) is broken
# three independent ways: it runs at gate_mcts_sims=1 against 256-sim selfplay,
# it scores against STOCKFISH rather than the previous net, and it decides on a
# noisy 0.50 bar. Switching it on would report a verdict that cannot detect the
# regression we care about, and changing it needs a restart. This runs OUTSIDE
# the training process: no live-yaml edit, no restart, nothing to wedge.
#
# Two series per day, both paired at 32 matched sims (sims 1/32 arenas are the
# ones explicitly SAFE concurrent with training — never 256+, that OOM-crashed
# the run 2026-06-18):
#
#   vs YESTERDAY  -> which day a regression happened (what you asked for)
#   vs boot512    -> cumulative drift from the frozen anchor. This is the series
#                    that exposed the warm-start LR crash; the day-over-day one
#                    would have shown ~nothing, because -494 Elo arrived as many
#                    small daily steps.
#
# Ray PRUNES live checkpoints, so today's is copied out before it can vanish.
#
# This is the ONE-SHOT. It is not scheduled by cron (cron does not run in this
# WSL2 instance, and a ratchet that fires while training is stopped burns GPU
# the operator wanted free). scripts/ratchet_loop.sh drives it once per calendar
# day and is started/stopped WITH training by scripts/train.sh.
#
#   ./scripts/daily_gate_ratchet.sh            # run once, now
#   ./scripts/daily_gate_ratchet.sh --games 100
#
# THIS JOB RUNS ON A HARD WALL-CLOCK BUDGET (--max-minutes, default 30) SPLIT
# ACROSS BOTH SERIES. That is a deliberate inversion of how it was first built.
# The original sized itself by statistical power — 200 games because that
# resolves the effect — and let the clock fall where it may. On 2026-07-26 the
# clock fell at 4h32m and cost ~23 iterations of training (82% throughput loss;
# invariant L6 in docs/rl_loop_audit.md). A daily observer that eats half a day
# of the training it observes is worse than no observer.
#
# So the budget is the fixed quantity and the CI is what floats. A capped run
# records however many games finished, with the games column reporting the TRUE
# count rather than the requested one. --report-every 16 exists for the same
# reason: the arena's default RUNNING-Elo block only lands at 64 games, so a
# capped run could burn its whole budget and print nothing.
set -u
cd /home/josh/projects/chess
export PYTHONPATH=.

GAMES=200
SIMS=32
# Concurrent games. 512x16 search costs ~700MB of GPU per game, and this runs
# ALONGSIDE training — a 256+ sim arena concurrent with training OOM-crashed the
# run 2026-06-18. 8 keeps the arena near ~6GB of the ~19GB free, so it coexists
# with the trainer and monitor_fen's panel/value_regret reads while staying far
# from that OOM (which was 256 sims at 16 concurrent, not 32 at 8).
# 2026-07-26: raised 4 -> 16 (via 8). Measured, at 32 sims:
#     conc 16 (swap gate)  64 games / 194s
#     conc 4  (this job)   64 games / 2770s
# 14x slower for 4x less concurrency — SUPERLINEAR, because with only a few
# games in flight the GPU batches are tiny and the run is latency-bound on
# round-trips rather than compute. The box is also CPU-starved: Stockfish label
# generation holds ~27 of 32 cores (32 procs at ~698k nodes), leaving the arena
# ~0.9 of a core, so the win comes from making each GPU round-trip carry more
# work rather than from more CPU. 16 at 32 sims is exactly what the 2026-07-25
# swap-gate arena ran (400 games) without trouble; the 2026-06-18 OOM was 256
# sims, not 32.
CONC=16
# Hard total wall-clock budget for the whole job, split across the series that
# actually run. See the header: the budget is fixed, the CI floats.
BUDGET_MIN=30
REPORT_EVERY=16
SNAP_DIR=data/ratchet/snapshots
LOG=data/ratchet/ratchet.csv
ANCHOR=scratchpad/scaleup/gateread/boot_snap_recheck_0711_0404.pt

while [ $# -gt 0 ]; do
    case "$1" in
        --games) GAMES=$2; shift 2 ;;
        --sims)  SIMS=$2;  shift 2 ;;
        --max-concurrent-games) CONC=$2; shift 2 ;;
        --max-minutes) BUDGET_MIN=$2; shift 2 ;;
        --report-every) REPORT_EVERY=$2; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Deadline for the WHOLE job. Each series takes an equal share of whatever is
# left, so a slow first series cannot starve the second.
DEADLINE=$(( $(date +%s) + BUDGET_MIN * 60 ))
SERIES_LEFT=2

mkdir -p "$SNAP_DIR" "$(dirname "$LOG")"
[ -s "$LOG" ] || echo "date,iter,series,elo,ci_lo,ci_hi,score,games" > "$LOG"

today=$(date +%F)

# --- locate the live trial's newest checkpoint ------------------------------
trial=$(ls -td runs/pbt2_small/tune/train_trial_*/ 2>/dev/null | head -1)
if [ -z "$trial" ]; then echo "[ratchet] no trial dir — nothing to do"; exit 0; fi

ck=$(ls -td "$trial"checkpoint_* 2>/dev/null | head -1)
if [ -z "$ck" ]; then echo "[ratchet] no checkpoint under $trial"; exit 0; fi

src=$(find "$ck" -name "trainer.pt" | head -1)
if [ -z "$src" ]; then echo "[ratchet] no trainer.pt under $ck"; exit 0; fi

iter=$(basename "$ck" | sed 's/checkpoint_0*//')
snap="$SNAP_DIR/ck_${today}_iter${iter}.pt"

# Copy out BEFORE arena-ing: Ray can evict this checkpoint mid-run.
if [ ! -s "$snap" ]; then
    cp "$src" "$snap" || { echo "[ratchet] snapshot copy FAILED"; exit 1; }
    echo "[ratchet] snapshotted iter=$iter -> $snap"
fi

run_arena () {   # $1=reference  $2=series-label
    local ref="$1" series="$2" out
    out="data/ratchet/arena_${today}_${series}.log"
    if [ -s "$out" ] && grep -q "Elo:" "$out"; then
        echo "[ratchet] $series already done today"; return
    fi
    if [ ! -s "$ref" ]; then echo "[ratchet] $series: reference missing ($ref) — skip"; return; fi

    # Equal share of the time that is actually left, not of the original budget.
    local budget=$(( (DEADLINE - $(date +%s)) / SERIES_LEFT ))
    SERIES_LEFT=$(( SERIES_LEFT - 1 ))
    if [ "$budget" -lt 60 ]; then
        echo "[ratchet] $series: out of budget (${budget}s left) — skipping"; return
    fi

    echo "[ratchet] $series: iter$iter vs $(basename "$ref"), up to $GAMES games @${SIMS} sims, ${budget}s budget"
    # SIGTERM at the deadline, SIGKILL 20s later if it ignores it. A timed-out
    # arena is NOT an error here — it has been printing RUNNING Elo blocks every
    # $REPORT_EVERY games, and the parser below reads the last one.
    timeout -k 20 "${budget}s" \
        python3 scripts/arena_standard.py \
        --candidate "$snap" --reference "$ref" \
        --mode matched_sims --sims "$SIMS" --games "$GAMES" \
        --max-concurrent-games "$CONC" --report-every "$REPORT_EVERY" \
        --no-compile --device cuda --seed 42 \
        --label "ratchet_${today}_iter${iter}_${series}" > "$out" 2>&1
    local rc=$?
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
        echo "[ratchet] $series: hit the ${budget}s cap — recording the partial result"
    fi

    # arena_standard prints  [arena] Elo: -193.2  95% CI: [-248.7, -145.4]
    # ...but a POSITIVE result prints  Elo: +16.3  95% CI: [-70.3, +105.0].
    # The `\+?` below is load-bearing: without it the character class does not
    # match the leading '+', sed finds no match and passes the WHOLE LINE
    # through. That line contains commas, so it corrupts the CSV, and the
    # emptiness check below does NOT catch it because the value is non-empty.
    # Negative Elo parsed fine, so this failed ONLY when the net improved —
    # exactly the case this ratchet exists to detect. Fixed 2026-07-26.
    local line elo lo hi score played
    line=$(grep -E "^\[arena\] Elo:" "$out" | tail -1)
    elo=$(sed -E 's/.*Elo: *\+?([-0-9.]+).*/\1/'     <<<"$line")
    lo=$(sed  -E 's/.*CI: *\[ *\+?([-0-9.]+).*/\1/'  <<<"$line")
    hi=$(sed  -E 's/.*, *\+?([-0-9.]+) *\].*/\1/'    <<<"$line")
    score=$(grep -E "^\[arena\] score:" "$out" | tail -1 | sed -E 's/.*score: *([0-9.]+).*/\1/')
    # Reject anything non-numeric rather than writing it: a passthrough line
    # would silently corrupt the CSV for every later reader.
    #
    # 2026-07-26, SECOND ROUND: the first version of this guard checked `elo`
    # ONLY, and that was not enough. A one-sided `n/a` CI bound — which the
    # arena prints whenever the pentanomial is degenerate — passes the elo
    # check and then `hi` falls through as the WHOLE log line, writing a
    # 9-field row into an 8-field CSV. Reproduced exactly:
    #   [arena] Elo: +470.4  95% CI: [+311.5, n/a]
    #     -> elo=470.4  lo=311.5  hi=<the entire line, commas and all>
    # This is NOT exotic. PR #252 made capped runs routine, and with
    # REPORT_EVERY=16 the first RUNNING block lands at 8 pairs, where counts
    # like (6,2,0,0,0) produce exactly that line. Making the instrument cheaper
    # made its unhappy path common — the guard has to cover every field the
    # row is built from, not the one that failed first.
    local f
    for f in "${elo:-}" "${lo:-}" "${hi:-}"; do
        case "$f" in
            ''|*[!0-9.+-]*)
                echo "[ratchet] $series: unparseable Elo/CI (elo='${elo:-}' lo='${lo:-}' hi='${hi:-}') — see $out"
                return ;;
        esac
    done
    # A degenerate pentanomial (e.g. all draw-draw) gives se exactly 0, so the
    # arena prints a zero-width CI like [+0.0, +0.0]. That is arithmetically
    # correct and epistemically worthless — a row claiming perfect certainty
    # from a handful of pairs. Record it, but never silently.
    if [ "$lo" = "$hi" ]; then
        echo "[ratchet] $series: WARNING zero-width CI [$lo, $hi] — degenerate pentanomial, treat as no reading"
    fi
    # The games column must be what was PLAYED, not what was requested. Under a
    # wall-clock cap those differ, and a row claiming 200 games behind a 96-game
    # CI is exactly the class of bug this whole audit is about: a number that
    # does not mean what its name says. Both the final and the RUNNING blocks
    # print "[arena] N games (M opening pairs)", so the last one is the truth.
    played=$(grep -E "^\[arena\] [0-9]+ games \(" "$out" | tail -1 | sed -E 's/.*\] ([0-9]+) games.*/\1/')
    case "${played:-}" in ''|*[!0-9]*) played=$GAMES ;; esac
    echo "$today,$iter,$series,$elo,$lo,$hi,$score,$played" >> "$LOG"
    echo "[ratchet] $series: Elo $elo [$lo, $hi]  ($played games)"
}

# --- series 1: vs yesterday (most recent earlier snapshot) ------------------
prev=$(ls -t "$SNAP_DIR"/ck_*.pt 2>/dev/null | grep -v "$(basename "$snap")" | head -1)
if [ -n "$prev" ]; then
    run_arena "$prev" "vs_prev"
else
    echo "[ratchet] no earlier snapshot yet — vs_prev starts tomorrow"
fi

# --- series 2: vs the frozen boot512 anchor ---------------------------------
run_arena "$ANCHOR" "vs_boot512"

echo "[ratchet] done $(date -Is).  log: $LOG"
