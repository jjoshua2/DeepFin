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
#
# THE DAY'S BUDGET IS ATTEMPTS x --max-minutes, NOT UNBOUNDED. A zero-row run
# exits non-zero so ratchet_loop.sh retries instead of stamping the day done —
# but "retry until it works" against a failure that reproduces is a GPU retry
# storm, not a retry. At BUDGET_MIN=30 and a 600s poll that is ~18 GPU-hours a
# day, spent by the same script whose header blames a 4h32m arena for an 82%
# throughput loss, and it is self-reinforcing: contention makes runs produce no
# complete pairs, no pairs makes the loop retry, the retry adds contention. So:
#
#   * every run that reaches the arena appends one row to data/ratchet/attempts.csv
#     (date,iter,attempt,rc,rows,reason) — ATTEMPTED is recorded separately from
#     SUCCEEDED, and neither is inferred from the other;
#   * at most --max-attempts (3) attempts per calendar day may spend GPU;
#   * a failure whose REASON string repeats the previous attempt's is treated as
#     deterministic and not retried at all — the 2026-07-31 shape (880s of
#     model loading inside an 855s cap -> 0 pairs, every time) costs exactly two
#     attempts rather than the whole day;
#   * exhausted/deterministic failures exit $EXIT_NO_RETRY (5), which the loop
#     honours by stopping for the day WITHOUT stamping the day as done.
#
# Nothing here makes a failure quieter: the day still ends with no CSV row, a
# banner on stderr, and a durable attempts.csv record of every attempt.
set -u
# RATCHET_ROOT is a TEST SEAM, not an operator knob: tests/test_ratchet_search_shape.py
# runs this script for real against a sandbox tree with a stub arena. A guard
# that only greps this file cannot fail (that is exactly how the zero-row exit
# below shipped un-fireable), so the script has to be executable somewhere that
# is not the live run.
cd "${RATCHET_ROOT:-/home/josh/projects/chess}" || exit 2
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
# Attempt ledger: one row per run that reached the arena stage, whatever came
# of it. This is what bounds the day's GPU and what makes a dead instrument
# legible — "3 attempts, 0 rows" is a fact on disk, not an inference from a
# missing CSV row.
ATTEMPTS=data/ratchet/attempts.csv
ATTEMPTS_HEADER="date,iter,attempt,rc,rows,reason"
MAX_ATTEMPTS=3
# Exit statuses. ratchet_loop.sh branches on these; keep them in sync with the
# RATCHET_EXIT_NO_RETRY literal there (tests/test_ratchet_search_shape.py
# asserts the two files agree).
EXIT_RETRY=1      # no CSV row, but another attempt today might produce one
EXIT_NO_RETRY=5   # no CSV row, and no further attempt today can help
ANCHOR=scratchpad/scaleup/gateread/boot_snap_recheck_0711_0404.pt
# WHICH SEARCH THIS SERIES IS MEASURED WITH. arena_standard.py now REQUIRES this
# (it used to seed itself from PLAY_SEARCH_DEFAULTS silently), and the answer for
# this job is `training`: the ratchet exists to price the RL loop's own progress
# — scripts/ratchet_slope.py fits Elo against cumulative optimizer steps — so it
# has to rank nets under the search selfplay actually runs, not the tuned 8000-sim
# UCI shape. Judging the loop through the play shape is exactly what the
# 2026-07-28 ledger finding condemns.
#
# THIS IS AN INSTRUMENT BREAK. Rows written before 2026-07-29 were measured on
# the play shape at vloss_weight=0 — a ruler that no longer exists on either
# side of this flag — so they are NOT comparable with rows written after, and
# the migration below labels them `legacy_play_vloss0` rather than pretending
# they are `play`. ratchet_slope.py refuses to fit across two labels.
SHAPE=training
CSV_HEADER="date,iter,series,elo,ci_lo,ci_hi,score,games,search_shape"
LEGACY_SHAPE=legacy_play_vloss0

while [ $# -gt 0 ]; do
    case "$1" in
        --games) GAMES=$2; shift 2 ;;
        --sims)  SIMS=$2;  shift 2 ;;
        --max-concurrent-games) CONC=$2; shift 2 ;;
        --max-minutes) BUDGET_MIN=$2; shift 2 ;;
        --report-every) REPORT_EVERY=$2; shift 2 ;;
        --search-shape) SHAPE=$2; shift 2 ;;
        # 0 = unlimited, for an operator who has fixed the cause and wants the
        # day's reading back. The loop never passes this.
        --max-attempts) MAX_ATTEMPTS=$2; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# Deadline for the WHOLE job. Each series takes an equal share of whatever is
# left, so a slow first series cannot starve the second.
DEADLINE=$(( $(date +%s) + BUDGET_MIN * 60 ))
SERIES_LEFT=2
# How many series ended with a readable CSV row. The exit status is built from
# this, because ratchet_loop.sh stamps the calendar day DONE on exit 0: a run
# that wrote nothing and still exited 0 burned the day's only attempt and told
# nobody. That is how 2026-07-30 and 07-31 each became a silent hole in the
# series instead of a retry ten minutes later.
ROWS_WRITTEN=0
# One token per series describing how it ended, joined into the attempt
# ledger's `reason` field. Two consecutive attempts with the SAME reason is the
# signature of a failure that reproduces, which is the one thing a retry cannot
# fix. Tokens: row | already | noref | nosnap | nobudget | backstop | nopairs |
# unparseable | selfmatch.
REASONS=""
# Tokens that no later attempt today can change: the missing file will still be
# missing, the same net will still be the same net. A run whose every series
# ended this way is not retried at all — not even once.
STRUCTURAL_TOKENS=" noref nosnap selfmatch "
note_reason () { REASONS="${REASONS:+$REASONS|}$1:$2"; }

mkdir -p "$SNAP_DIR" "$(dirname "$LOG")"
[ -s "$LOG" ] || echo "$CSV_HEADER" > "$LOG"
# One-time migration of the pre-instrument-break CSV. Appending a 9th field to a
# file whose header has 8 would leave csv.DictReader stuffing the shape into the
# restkey and every old row claiming the new ruler by omission — a schema break
# that reads as agreement. Stamp the old rows with what they ACTUALLY measured
# instead. Idempotent: keyed on the header line, atomic via mv.
if [ "$(head -1 "$LOG")" != "$CSV_HEADER" ]; then
    if [ "$(head -1 "$LOG")" = "date,iter,series,elo,ci_lo,ci_hi,score,games" ]; then
        echo "[ratchet] migrating $LOG to the search_shape schema (old rows -> $LEGACY_SHAPE)"
        { echo "$CSV_HEADER"; tail -n +2 "$LOG" | sed "s/\$/,$LEGACY_SHAPE/"; } > "$LOG.tmp" \
            && mv "$LOG.tmp" "$LOG"
    else
        echo "[ratchet] REFUSING to write: $LOG header is neither the current nor the pre-2026-07-29 schema" >&2
        exit 1
    fi
fi

today=$(date +%F)

# --- locate the live trial's newest checkpoint ------------------------------
trial=$(ls -td runs/pbt2_small/tune/train_trial_*/ 2>/dev/null | head -1)
if [ -z "$trial" ]; then echo "[ratchet] no trial dir — nothing to do"; exit 0; fi

ck=$(ls -td "$trial"checkpoint_* 2>/dev/null | head -1)
if [ -z "$ck" ]; then echo "[ratchet] no checkpoint under $trial"; exit 0; fi

src=$(find "$ck" -name "trainer.pt" | head -1)
if [ -z "$src" ]; then echo "[ratchet] no trainer.pt under $ck"; exit 0; fi

# `sed 's/checkpoint_0*//'` yields the EMPTY STRING for checkpoint_000000 — the
# zeros it strips are the number. Strip the prefix, then leading zeros only down
# to the last digit, so iteration 0 stays "0" instead of becoming an empty
# snapshot name and an empty iter column in every row of that day.
iter=$(basename "$ck" | sed -E 's/^checkpoint_//; s/^0+([0-9])/\1/')
snap="$SNAP_DIR/ck_${today}_iter${iter}.pt"

# --- how much GPU may this calendar day still spend? ------------------------
[ -s "$ATTEMPTS" ] || echo "$ATTEMPTS_HEADER" > "$ATTEMPTS"
rows_today=$(grep -c "^$today,$iter," "$LOG" || true)
attempts_today=$(grep -c "^$today," "$ATTEMPTS" || true)
last_reason=$(grep "^$today," "$ATTEMPTS" | tail -1 | cut -d, -f6)
last_status=$(grep "^$today," "$ATTEMPTS" | tail -1 | cut -d, -f4)

give_up () {   # $1=why (one line, for the operator)
    echo "[ratchet] ================ RATCHET DOWN for $today ================" >&2
    echo "[ratchet] ${attempt_no:-$attempts_today} attempt(s) today, ZERO rows in $LOG." >&2
    echo "[ratchet] $1" >&2
    echo "[ratchet] This is NOT a null reading: $today has NO strength measurement." >&2
    echo "[ratchet] Attempts: $ATTEMPTS   arena logs: data/ratchet/arena_${today}_iter${iter}_*.log" >&2
    echo "[ratchet] Re-run by hand with --max-attempts 0 once the cause is fixed." >&2
    exit "$EXIT_NO_RETRY"
}

# The GPU bound, and the only part of it that matters: it is checked BEFORE the
# snapshot copy and before any arena starts, so a day that has already given up
# costs one `grep` per poll instead of 30 minutes of contention. Keyed on the
# attempts ledger, so it holds for a hand-run job too — the loop's own
# last_giveup_date only stops the loop.
if [ "$rows_today" -eq 0 ] && [ "$MAX_ATTEMPTS" -gt 0 ]; then
    if [ "${last_status:-}" = "$EXIT_NO_RETRY" ]; then
        give_up "attempt $attempts_today already concluded today is not retryable (${last_reason:-none})."
    fi
    if [ "$attempts_today" -ge "$MAX_ATTEMPTS" ]; then
        give_up "the day's budget of $MAX_ATTEMPTS attempt(s) x ${BUDGET_MIN}min is spent (last reason: ${last_reason:-none})."
    fi
fi

# Copy out BEFORE arena-ing: Ray can evict this checkpoint mid-run.
if [ ! -s "$snap" ]; then
    cp "$src" "$snap" || { echo "[ratchet] snapshot copy FAILED"; exit 1; }
    echo "[ratchet] snapshotted iter=$iter -> $snap"
fi

pick_prev_snapshot () {   # $1=snapshot dir  $2=today's snapshot PATH  $3=today's iter
    # rc 0 + path : a usable earlier reference
    # rc 3        : the newest earlier snapshot is the SAME NET — byte-identical
    #               to today's, or the same iteration under a different name.
    #               A self-match, true Elo exactly 0.
    # rc 1        : no earlier snapshot at all
    #
    # The old selector filtered by FILENAME only, so when training is down the
    # daily snapshot is byte-identical to yesterday's under a new date and the
    # `vs_prev` series quietly played the net against ITSELF. Confirmed by
    # sha256: ck_2026-07-31_iter478.pt and ck_2026-08-01_iter478.pt are the same
    # file (93db63e5…), and the 2026-08-01 vs_prev run wrote -43.7 Elo for a
    # comparison whose true value is 0.
    #
    # This was harmless while a short run wrote nothing; `--max-seconds` made
    # EVERY run write a row, and the ledger's own advice is to run the ratchet
    # in a training-down window — exactly when the self-match happens. So the
    # fix that made the instrument reliable is what made this consequential.
    #
    # SKIP rather than fall back to an older iteration: silently substituting a
    # 2-day-old reference would keep the `vs_prev` label on a row that no longer
    # means "vs yesterday", which is the defect class this whole PR is about.
    # THE COMPARISON IS ON CONTENT, not on the iteration parsed out of the name.
    # The name is a claim about the file; `cmp -s` is the file. They come apart
    # exactly when it matters — a snapshot re-copied under a new date, a
    # checkpoint saved twice, any renaming — and the failure is silent in the
    # direction that writes a bogus row. The iteration check stays as a second
    # net: a re-SAVED same-iteration checkpoint differs byte-wise (timestamps,
    # optimizer moments) while still being a self-match for Elo purposes.
    local prev pit
    prev=$(ls -t "$1"/ck_*.pt 2>/dev/null | grep -vF "$(basename "$2")" | head -1)
    [ -n "$prev" ] || return 1
    cmp -s "$prev" "$2" && return 3
    pit=$(basename "$prev" | sed 's/.*_iter//; s/\.pt$//')
    [ "$pit" = "$3" ] && return 3
    printf '%s\n' "$prev"
}

pick_log_path () {   # $1=date  $2=iter  $3=series  -> a path that does NOT exist
    # NEVER overwrite a previous attempt's log. The name used to be
    # arena_<date>_<series>.log opened with `>`, so the retry that the exit-1
    # path now makes routine would destroy the evidence of the attempt it is
    # retrying. That is not hypothetical: it is exactly how the 2026-07-27 00:09
    # failure log was lost — a later manual run that day reused the name, which
    # is why the first version of this fix mis-attributed that day to a flush
    # bug. Keyed on the ITERATION too, because a new checkpoint is a new
    # measurement rather than a retry.
    #
    # A standalone function so it can be extracted and executed by
    # tests/test_ratchet_search_shape.py: a guard that only greps this file for
    # the string cannot fail when the logic behind it is dead.
    local base="data/ratchet/arena_${1}_iter${2}_${3}"
    local path="${base}.log" attempt=2
    while [ -e "$path" ]; do
        path="${base}.attempt${attempt}.log"
        attempt=$(( attempt + 1 ))
    done
    printf '%s\n' "$path"
}

run_arena () {   # $1=reference  $2=series-label
    local ref="$1" series="$2" out
    # "Already done" must be keyed on THE CSV ROW, which is the only artifact
    # this job exists to produce. Two weaker versions were both wrong:
    #   `grep "Elo:" "$out"`        matched the RUNNING-block HEADER, i.e.
    #                               exactly what a SIGKILLed run leaves behind.
    #   `grep "^\[arena\] Elo:"`    means the LOG was parseable, which is still
    #                               not "a row exists": every field guard below
    #                               `return`s without writing, and the one-sided
    #                               `n/a` CI case ("[arena] Elo: +470.4  95% CI:
    #                               [+311.5, n/a]") is common at small pair
    #                               counts — arena_2026-07-29_vs_prev.log has one
    #                               at 5 pairs. A capped run now ALWAYS prints a
    #                               final summary, so that case got MORE likely,
    #                               not less.
    # Asking the CSV directly cannot drift from what the CSV contains.
    if grep -q "^$today,$iter,$series," "$LOG"; then
        echo "[ratchet] $series already done today"
        note_reason "$series" already
        ROWS_WRITTEN=$(( ROWS_WRITTEN + 1 )); return
    fi
    if [ ! -s "$ref" ]; then
        echo "[ratchet] $series: reference missing ($ref) — skip"
        note_reason "$series" noref; return
    fi

    out=$(pick_log_path "$today" "$iter" "$series")

    # Equal share of the time that is actually left, not of the original budget.
    local budget=$(( (DEADLINE - $(date +%s)) / SERIES_LEFT ))
    SERIES_LEFT=$(( SERIES_LEFT - 1 ))
    if [ "$budget" -lt 60 ]; then
        echo "[ratchet] $series: out of budget (${budget}s left) — skipping"
        note_reason "$series" nobudget; return
    fi

    echo "[ratchet] $series: iter$iter vs $(basename "$ref"), up to $GAMES games @${SIMS} sims, ${budget}s budget"
    # THE ARENA STOPS ITSELF (`--max-seconds`); `timeout` is only a backstop.
    #
    # It used to be the other way round, and that is why this job produced ZERO
    # rows on 2026-07-30 and 07-31. A SIGKILLed process loses whatever is still
    # in its block-buffered stdout, which is always exactly the LAST block --
    # each `flush=True` print pushes everything written before it. A run that
    # printed many blocks therefore kept all but the last (07-28/07-29 recorded
    # rows); a run slow enough to print only ONE lost everything, and those logs
    # end mid-report at "[arena] RUNNING Elo after 6 complete pairs:" with no
    # "[arena] Elo:" line for the parser below. Under --max-seconds the arena breaks
    # out of the play loop on its own clock, scores the pairs that FINISHED, and
    # prints + appends a normal record — so a capped run is a small sample
    # rather than no sample.
    #
    # The internal budget is 45s short of the external one so the arena has room
    # to finalize (summary + JSONL) before `timeout` would fire. The backstop
    # stays because the deadline is only checked between plies: a hang inside
    # the C search or in checkpoint loading still has to be killed from outside.
    local inner=$(( budget - 45 ))
    [ "$inner" -lt 30 ] && inner=30
    timeout -k 20 "${budget}s" \
        python3 scripts/arena_standard.py \
        --candidate "$snap" --reference "$ref" \
        --mode matched_sims --search-shape "$SHAPE" --sims "$SIMS" --games "$GAMES" \
        --max-concurrent-games "$CONC" --report-every "$REPORT_EVERY" \
        --max-seconds "$inner" \
        --no-compile --device cuda --seed 42 \
        --label "ratchet_${today}_iter${iter}_${series}" > "$out" 2>&1
    local rc=$?
    # The token recorded for this series if no row comes out of the log below.
    # `nopairs` (arena rc 3) is the 2026-07-31 shape and the reason the retry
    # storm is not hypothetical: the arena spent its entire budget before the
    # play loop ever completed a pair — arena_2026-07-31_vs_boot512.log is 4
    # lines ending at "SEARCH reference:" after 880s — so a retry under the same
    # cap replays the same 880s to the same end. One log cannot tell that from
    # transient contention, but two identical ones can, which is what the
    # repeat-reason rule at the bottom of this script keys on.
    local fail=unparseable
    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
        echo "[ratchet] $series: the ${budget}s BACKSTOP fired — --max-seconds ${inner}s did not stop it in time (hang, not slowness); any reading below came from a RUNNING block"
        fail=backstop
    elif [ "$rc" -eq 3 ]; then
        echo "[ratchet] $series: arena finished ${inner}s with NO COMPLETE PAIRS — see $out"
        fail=nopairs
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
                note_reason "$series" "$fail"; return ;;
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
    echo "$today,$iter,$series,$elo,$lo,$hi,$score,$played,$SHAPE" >> "$LOG"
    ROWS_WRITTEN=$(( ROWS_WRITTEN + 1 ))
    note_reason "$series" row
    echo "[ratchet] $series: Elo $elo [$lo, $hi]  ($played games, $SHAPE shape)"
}

# --- series 1: vs yesterday (most recent earlier snapshot) ------------------
prev=$(pick_prev_snapshot "$SNAP_DIR" "$snap" "$iter")
case $? in
    0) run_arena "$prev" "vs_prev" ;;
    3) echo "[ratchet] vs_prev SKIPPED: the newest earlier snapshot is the SAME NET" \
            "(byte-identical, or iter$iter again) — true Elo exactly 0, and the row" \
            "would read as a real day-over-day measurement. Training is probably down."
       note_reason vs_prev selfmatch ;;
    *) echo "[ratchet] no earlier snapshot yet — vs_prev starts tomorrow"
       note_reason vs_prev nosnap ;;
esac

# --- series 2: vs the frozen boot512 anchor ---------------------------------
run_arena "$ANCHOR" "vs_boot512"

echo "[ratchet] done $(date -Is).  log: $LOG"

# --- decide, record the attempt, then act ------------------------------------
# ATTEMPTED and SUCCEEDED are separate facts and neither is inferred from the
# other: this row is appended whatever happened, and ratchet_loop.sh's day stamp
# still moves only on exit 0.
attempt_no=$(( attempts_today + 1 ))
reason=${REASONS:-none}
status=0
detail=""
if [ "$ROWS_WRITTEN" -eq 0 ]; then
    status=$EXIT_RETRY
    # 1. Deterministic BY EVIDENCE: this attempt failed exactly the way the last
    #    one did, so the next one reproduces it. That is a loop, not a retry.
    if [ -n "$last_reason" ] && [ "$last_reason" = "$reason" ]; then
        status=$EXIT_NO_RETRY
        detail="attempt $attempt_no failed identically to attempt $attempts_today ($reason) — it reproduces, so it is not retryable."
    else
        # 2. Deterministic BY CONSTRUCTION: no later attempt today can change any
        #    of these outcomes (the reference file is still missing; the newest
        #    earlier snapshot is still the same net).
        transient=0
        for tok in ${reason//|/ }; do
            case "$STRUCTURAL_TOKENS" in *" ${tok#*:} "*) ;; *) transient=1 ;; esac
        done
        if [ "$transient" -eq 0 ]; then
            status=$EXIT_NO_RETRY
            detail="every series ended structurally ($reason) — no later attempt today can change that."
        # 3. The absolute cap, checked here as well as before the arena so the
        #    LAST attempt reports the give-up itself instead of leaving the next
        #    poll to discover it 10 minutes later.
        elif [ "$MAX_ATTEMPTS" -gt 0 ] && [ "$attempt_no" -ge "$MAX_ATTEMPTS" ]; then
            status=$EXIT_NO_RETRY
            detail="the day's budget of $MAX_ATTEMPTS attempt(s) x ${BUDGET_MIN}min is now spent."
        fi
    fi
fi
echo "$today,$iter,$attempt_no,$status,$ROWS_WRITTEN,$reason" >> "$ATTEMPTS"

[ "$status" -eq 0 ] && exit 0
echo "[ratchet] NO ROWS WRITTEN (attempt $attempt_no, reason $reason)" >&2
[ "$status" -eq "$EXIT_NO_RETRY" ] && give_up "$detail"
echo "[ratchet] retryable — up to $(( MAX_ATTEMPTS - attempt_no )) more attempt(s) today" >&2
exit "$EXIT_RETRY"
