#!/bin/bash
# Usage: ./scripts/train.sh [start|stop|status|log|restart|salvage-export|salvage-restart|best-save|best-list]
#
# salvage-export [--top-n N] [--out DIR] [--metric KEY] [--no-copy-replay] [--dry-run]
#   Export top-N trial seeds (checkpoint + replay) from the current tune run
#   into a salvage pool. Does not touch the running process. If --out is
#   omitted the pool lands under $WORK_DIR/salvage/<run-id>_<timestamp>/.
#   --dry-run prints selected trials/checkpoints without copying files.
#
# salvage-restart POOL_DIR [--no-pid] [--no-optimizer] [--reinit-volatility] [--donor-config]
#   Stop training, then start it again pointing at POOL_DIR. Defaults restore
#   the donor PID state and full trainer state but NOT the donor's LR/config.
#   Pass --no-pid / --no-optimizer to flip those defaults.
#
# best-save LABEL [--iter N]
#   Snapshot the active trial's best_regret dir (lowest regret unless --iter N
#   names a specific iteration) plus the current replay_shards into
#   data/best_pools/LABEL/ as a valid salvage pool — restore-compatible with
#   salvage-restart. Ray rotates trial dirs; this is the one persistent copy.
#
# best-list
#   Enumerate pools under data/best_pools/ with regret/iter/size metadata.
#
set -e

CONFIG="${TRAIN_CONFIG:-configs/pbt2_small.yaml}"
LOG="/tmp/chess_training.log"
PIDFILE="/tmp/chess_training.pid"
STOP_MARKER="/tmp/chess_training.intentional_stop"
WORK_DIR="${TRAIN_WORK_DIR:-runs/pbt2_small}"
BEST_POOLS_DIR="${TRAIN_BEST_POOLS_DIR:-data/best_pools}"
# Rolling top-N auto-save written by _update_best_regret_checkpoints. Matches
# TrialConfig.best_regret_checkpoints_dir default in trial_config.py.
AUTO_BEST_REGRET_DIR="${TRAIN_AUTO_BEST_REGRET_DIR:-data/best_regret_checkpoints}"

cd "$(dirname "$0")/.."

clear_pause_markers() {
    if [ "${TRAIN_KEEP_PAUSE_MARKERS:-0}" = "1" ]; then
        return 0
    fi
    local tune_dir="$WORK_DIR/tune"
    [ -d "$tune_dir" ] || return 0

    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local cleared=0
    local marker
    for marker in "$tune_dir/pause.txt" "$tune_dir"/train_trial_*/pause.txt; do
        [ -f "$marker" ] || continue
        mv "$marker" "$marker.cleared_$ts"
        cleared=$((cleared + 1))
    done
    if [ "$cleared" -gt 0 ]; then
        echo "Cleared $cleared stale pause marker(s) under $tune_dir."
    fi
}

check_c_extensions() {
    if [ "${TRAIN_SKIP_C_EXT_CHECK:-0}" = "1" ]; then
        return 0
    fi
    PYTHONPATH=. python3 scripts/check_c_extensions_fresh.py --quiet \
        --min-gcc-major 15 --require-production-recipe
}

start() {
    if running; then
        echo "Already running (PID $(cat "$PIDFILE"))"
        return 1
    fi
    # Strip our own --fresh flag (not passed to run.py) and detect --resume /
    # --salvage-seed-pool-dir so auto-resume only kicks in when the caller
    # didn't specify intent. This prevents silently starting a fresh trial and
    # dropping a live one when restarting after a stop.
    local extra_args=() has_resume=0 has_salvage=0 fresh=0
    for a in "$@"; do
        case "$a" in
            --fresh)                 fresh=1 ;;
            --resume)                has_resume=1; extra_args+=("$a") ;;
            --salvage-seed-pool-dir) has_salvage=1; extra_args+=("$a") ;;
            *)                       extra_args+=("$a") ;;
        esac
    done
    if [ "$fresh" = "0" ] && [ "$has_resume" = "0" ] && [ "$has_salvage" = "0" ]; then
        if ls "$WORK_DIR"/tune/experiment_state-*.json >/dev/null 2>&1; then
            echo "Detected prior tune state under $WORK_DIR/tune — auto-resuming."
            echo "  To start fresh instead, pass --fresh or rm -rf $WORK_DIR/tune."
            extra_args+=("--resume")
        fi
    fi
    check_c_extensions
    clear_pause_markers
    # Re-arm the watchdog flat-clock: a stale state file (rows + wall_time from a
    # prior stall) makes the first post-start poll read >90min-flat and false-fire
    # auto-recovery on the freshly-started trainer. Clearing it forces a fresh
    # first-observation baseline. The recovery cooldown stamp is NOT cleared:
    # recover_stall.sh calls this start path, so clearing it here would erase the
    # anti-flap cooldown after every auto-recovery and the "re-stall within
    # cooldown -> NEEDS HUMAN" escalation could never trigger.
    rm -f /tmp/chess_watchdog_state.json
    echo "Starting training with $CONFIG ${extra_args[*]:+(extra: ${extra_args[*]})}..."
    # Inductor compile parallelism — without these, autotune is single-threaded
    # and uses ~6% of available CPU. COMPILE_THREADS parallelizes codegen
    # across candidate kernels; AUTOTUNE_IN_SUBPROC isolates each candidate's
    # benchmark in a subprocess so a config that overflows shared memory
    # (RTX 5090 hits this with max-autotune block sizes) crashes its own
    # process instead of polluting the parent's CUDA context, and lets
    # candidates benchmark concurrently. FX_GRAPH_CACHE persists traced
    # graphs across runs alongside the kernel cache.
    PYTHONPATH=. \
    TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-16}" \
    TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC="${TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC:-1}" \
    TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}" \
    setsid nohup python3 -m chess_anti_engine.run \
        --config "$CONFIG" --mode tune "${extra_args[@]}" \
        < /dev/null \
        > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "Started PID $! — log: $LOG"
    rm -f "$STOP_MARKER"
    start_observers
}

# Observers resume WITH training so nothing is forgotten (2026-07-13, user):
# watchdog_loop (stall detection + AUTO-RECOVERY via recover_stall.sh on a
# confirmed stall; disable with WATCHDOG_AUTO_RECOVER=0; alerts and recovery
# suppressed while $STOP_MARKER exists) and monitor_fen (panels + value trend
# + seed retire/probation; internally idles while the trainer is down so seed
# logic never burns compute against a stopped run).
start_observers() {
    if ! pgrep -f "scripts/watchdog_loop.sh" >/dev/null; then
        setsid nohup bash scripts/watchdog_loop.sh < /dev/null \
            > /dev/null 2>&1 &
        echo "Started watchdog_loop (PID $!) — log: scratchpad/watchdog.log"
    fi
    if ! pgrep -f "scripts/monitor_fen.sh" >/dev/null; then
        setsid nohup bash scripts/monitor_fen.sh < /dev/null \
            > scratchpad/live_read/monitor_fen.out 2>&1 &
        echo "Started monitor_fen (PID $!) — log: scratchpad/live_read/monitor/monitor.log"
    fi
    if ! pgrep -f "scripts/ratchet_loop.sh" >/dev/null; then
        setsid nohup bash scripts/ratchet_loop.sh < /dev/null \
            > /dev/null 2>&1 &
        echo "Started ratchet_loop (PID $!) — log: scratchpad/ratchet_loop.log"
    fi
}

stop() {
    if ! running; then
        echo "Not running"
        return 0
    fi
    local pid=$(cat "$PIDFILE")
    # Mark the stop as intentional BEFORE killing so the watchdog never alerts
    # on a stop the operator asked for (it keeps watching; a later crash or a
    # forgotten restart still shows in its log, just not as an alert).
    touch "$STOP_MARKER"

    # ── Drain selfplay BEFORE anything is killed ─────────────────────────────
    # Workers are subprocess grandchildren of `ray::ImplicitFunc.train`, NOT of
    # $pid, so `kill "$pid"` below never reaches them (verified 2026-07-31:
    # ps -o ppid on all four live workers gives the trial actor, not the driver).
    # Every in-flight game was therefore discarded on an operator pause:
    # `resumed games=0` on all four workers, versus `resumed games=24
    # records=731` across a graceful session restart. Those discarded games are
    # what makes the next window length-truncated (only short decisive games
    # have finished), spiking winrate and driving the PID to tighten regret
    # ~-0.008 against a phantom.
    #
    # NOT the mechanism: the `pkill -9 ray::` / `raylet` lines below. Measured
    # 2026-07-31 against /proc/<pid>/cmdline, ZERO of the four workers match
    # either pattern, and the intersection of `pgrep -f 'ray::'` with the worker
    # pids is empty. Killing the trial actor ORPHANS the workers rather than
    # killing them; what actually reached them was the 5s SIGTERM->SIGKILL in
    # tune/_utils.py::terminate_process, and with no handler installed that was
    # simply fatal. An earlier revision of this comment asserted the pkill
    # mechanism -- it was wrong, and the second pass at the end of stop() exists
    # because of it.
    #
    # Signalling the workers directly lets their SIGTERM handler run the same
    # suspend path a reco restart-key change already uses. Games land in
    # <worker>/selfplay_resume/ and the next session picks them up.
    #
    # The wait is over the pid list captured ONCE, never a re-run of pgrep: the
    # driver's revive_dead_selfplay_processes() (distributed_runtime.py) relaunches
    # a worker that exits mid-iteration, so a `pgrep`-until-empty loop would never
    # converge — it would burn the whole grace period and then report a false
    # "still alive" warning. A revived worker is seconds old and has nothing in
    # flight worth suspending, so it is not worth waiting for HERE — but it is
    # NOT free: it reuses the dead worker's worker_index and therefore its
    # work_dir, so once it reaches a session (~60s) it claims-by-rename the very
    # selfplay_resume/ files this drain just banked. The second pass at the end
    # of stop() is what stops that, since nothing above kills it.
    local grace="${CAE_STOP_GRACE_SECONDS:-90}"
    # ONE definition, used by both drain passes. Duplicating the literal is how
    # this PR's own founding defect recurs: the second pass sits outside the
    # range tests/test_train_sh_worker_drain.py extracts, so a pattern edit
    # applied to one copy and not the other would leave the suite green while
    # silently reopening the orphan race. `test_the_worker_pattern_is_defined_once`
    # pins that. `--` is REQUIRED: the pattern begins with `-m`.
    #
    # -f is required (the module name is an argument, not the comm); the pattern
    # cannot match this script because it never appears in our own argv. Anchored
    # on `-m ... ( |$)` so it cannot also catch `chess_anti_engine.worker_pool`,
    # `.worker_config` or `.worker_buffer`. Verified 2026-07-31 against the live
    # run: anchored and unanchored select the identical four worker pids.
    local wpat='-m chess_anti_engine\.worker( |$)'
    local wpids
    wpids=$(pgrep -f -- "$wpat" 2>/dev/null || true)
    if [ -n "$wpids" ]; then
        echo "Draining selfplay workers (grace ${grace}s): $(echo "$wpids" | tr '\n' ' ')"
        # Snapshot each worker's --log-file and its CURRENT LENGTH before we
        # signal anything. The evidence that a worker banked its games is the
        # worker's OWN line
        #     selfplay resume: suspended games=N records=M ...
        # and that string is also emitted by every in-session reco restart, so
        # a naive grep of the whole log would find an hours-old line and read
        # it as proof that THIS drain worked. Recording the byte offset first
        # means we only ever read what this drain produced. (Same trap as the
        # negative-log checks elsewhere: a stale success line is worse than no
        # line at all.)
        local dstate w logf
        dstate=$(mktemp -d) || dstate=""
        if [ -n "$dstate" ]; then
            for w in $wpids; do
                logf=$(tr '\0' '\n' < "/proc/$w/cmdline" 2>/dev/null \
                       | awk '/^--log-file$/{getline; print; exit}')
                if [ -n "$logf" ] && [ -f "$logf" ]; then
                    printf '%s\n' "$logf" > "$dstate/$w.log"
                    wc -c < "$logf" > "$dstate/$w.off"
                fi
            done
        fi
        # shellcheck disable=SC2086
        kill -TERM $wpids 2>/dev/null || true
        local waited=0 alive w
        alive="$wpids"
        while [ "$waited" -lt "$grace" ] && [ -n "$alive" ]; do
            sleep 2
            waited=$((waited + 2))
            alive=""
            for w in $wpids; do
                # `if`, not `kill -0 ... && ...`. Measured 2026-07-31 against the
                # `set -e` at the top of this file: the && form itself is exempt
                # (it is an AND-list), but it leaves the for-loop's status
                # non-zero on the SUCCESS path — every worker gone — and a
                # for-loop that ends a function DOES abort under set -e. Today
                # that is only latent, because `echo "Stopping PID ..."` follows
                # this block; it becomes a live bug the moment the block moves or
                # something is appended after it, and it would fail in the one
                # case that matters (a clean drain).
                if kill -0 "$w" 2>/dev/null; then
                    alive="$alive $w"
                fi
            done
        done
        # LIVENESS IS NOT EVIDENCE ABOUT DATA. Until 2026-07-31 this reported
        # every still-running pid as "in-flight games will be DISCARDED", which
        # was measured false on the first real teardown: workers 01/02/03 each
        # logged "exiting after suspend" 4-40s BEFORE the 90s deadline and
        # banked 22-23 games (100/252/128 records), yet all three were named in
        # the warning because the probe caught them tearing down their CUDA
        # context. An operator reading that would either mourn a window of games
        # that survived, or raise CAE_STOP_GRACE_SECONDS to lengthen every
        # future stop against a non-problem — and, worst, could no longer tell
        # that case apart from a real discard. So the verdict is read off the
        # suspend line, and only a worker that banked NOTHING is a loss.
        local banked total=0 lost=""
        for w in $wpids; do
            banked=0
            if [ -n "$dstate" ] && [ -s "$dstate/$w.log" ]; then
                logf=$(cat "$dstate/$w.log")
                if [ -f "$logf" ]; then
                    banked=$(tail -c "+$(( $(cat "$dstate/$w.off" 2>/dev/null || echo 0) + 1 ))" \
                                 "$logf" 2>/dev/null \
                             | grep -oE 'suspended games=[0-9]+' \
                             | sed 's/.*=//' \
                             | awk '{s += $1} END {print s + 0}')
                fi
            fi
            case "${banked:-0}" in ''|*[!0-9]*) banked=0 ;; esac
            total=$((total + banked))
            if [ "$banked" -gt 0 ]; then
                echo "  worker $w: suspended ${banked} in-flight game(s)"
            elif kill -0 "$w" 2>/dev/null; then
                # Still running AND nothing banked: this is the real failure.
                lost="$lost $w"
            fi
        done
        # `if`, not `[ ... ] && ...`: an AND-list whose left side is false leaves
        # a non-zero status, which is the same set -e footgun documented above.
        if [ -n "$dstate" ]; then rm -rf "$dstate"; fi
        if [ -z "$lost" ]; then
            echo "Workers drained after ${waited}s (${total} in-flight game(s) suspended)"
        else
            echo "WARNING: worker(s) ($lost) still alive after ${grace}s with NO" \
                 "suspend recorded; their in-flight games are DISCARDED." \
                 "Raise CAE_STOP_GRACE_SECONDS if this recurs."
        fi
    fi

    echo "Stopping PID $pid ..."
    kill "$pid" 2>/dev/null || true
    # The driver still gets a short grace; the expensive drain already happened.
    local dwait=0
    while [ "$dwait" -lt 15 ] && kill -0 "$pid" 2>/dev/null; do
        sleep 1
        dwait=$((dwait + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "Force killing ..."
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
    ray stop 2>/dev/null || true
    sleep 1
    pkill -9 -f 'ray::' 2>/dev/null || true
    pkill -9 -f 'raylet' 2>/dev/null || true

    # ── Second drain pass: ORPHANS ───────────────────────────────────────────
    # Nothing above kills a worker. Verified 2026-07-31 against
    # /proc/<pid>/cmdline: no worker matches `ray::` or `raylet`, and the
    # intersection of `pgrep -f 'ray::'` with the worker pids is empty. Workers
    # are subprocess grandchildren, so tearing down ray ORPHANS any that the
    # first pass did not already drain -- including one the driver revived
    # DURING that pass, which reuses the dead worker's work_dir and would
    # otherwise reach a session in ~60s and claim-by-rename the selfplay_resume/
    # files we just banked.
    #
    # These get the same graceful SIGTERM: a survivor of the first pass may be
    # mid-suspend, and a revived one has a (small) table of its own. Only then
    # SIGKILL, so a hung worker cannot wedge `stop`.
    local orphans
    orphans=$(pgrep -f -- "$wpat" 2>/dev/null || true)
    if [ -n "$orphans" ]; then
        echo "Draining orphaned workers: $(echo "$orphans" | tr '\n' ' ')"
        # shellcheck disable=SC2086
        kill -TERM $orphans 2>/dev/null || true
        local owait=0 oalive="$orphans" o
        while [ "$owait" -lt "${CAE_ORPHAN_GRACE_SECONDS:-30}" ] && [ -n "$oalive" ]; do
            sleep 2
            owait=$((owait + 2))
            oalive=""
            for o in $orphans; do
                if kill -0 "$o" 2>/dev/null; then
                    oalive="$oalive $o"
                fi
            done
        done
        if [ -n "$oalive" ]; then
            echo "Force killing orphaned workers:$oalive"
            # shellcheck disable=SC2086
            kill -9 $oalive 2>/dev/null || true
        fi
    fi
    echo "Stopped"
}

running() {
    [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

status() {
    if running; then
        echo "Running (PID $(cat "$PIDFILE"))"
    else
        echo "Not running"
        rm -f "$PIDFILE"
    fi
}

salvage_export() {
    local top_n=3
    local out_dir=""
    local metric="opponent_strength"
    local copy_replay="--salvage-copy-replay"
    local dry_run=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --top-n) top_n="$2"; shift 2 ;;
            --out) out_dir="$2"; shift 2 ;;
            --metric) metric="$2"; shift 2 ;;
            --no-copy-replay) copy_replay="--no-salvage-copy-replay"; shift ;;
            --dry-run) dry_run="--salvage-dry-run"; shift ;;
            *) echo "Unknown salvage-export arg: $1"; return 1 ;;
        esac
    done

    local out_args=()
    [ -n "$out_dir" ] && out_args=(--salvage-out-dir "$out_dir")

    if [ -n "$dry_run" ]; then
        echo "Planning top-$top_n salvage seeds from $WORK_DIR (metric=$metric, dry-run)..."
    else
        echo "Exporting top-$top_n salvage seeds from $WORK_DIR (metric=$metric)..."
    fi
    PYTHONPATH=. python3 -m chess_anti_engine.run \
        --config "$CONFIG" --mode salvage \
        --work-dir "$WORK_DIR" \
        --salvage-top-n "$top_n" \
        --salvage-metric "$metric" \
        "${out_args[@]}" \
        "$copy_replay" \
        $dry_run
}

salvage_restart() {
    if [ $# -lt 1 ]; then
        echo "Usage: $0 salvage-restart POOL_DIR [--no-pid] [--no-optimizer] [--reinit-volatility] [--donor-config]"
        return 1
    fi
    local pool_dir="$1"; shift
    if [ ! -d "$pool_dir" ] || [ ! -f "$pool_dir/manifest.json" ]; then
        echo "Not a salvage pool: $pool_dir (missing manifest.json)"
        return 1
    fi
    # Ray trial workers run with cwd set to a per-trial tmp dir, so relative
    # paths break salvage loading. Resolve to absolute before passing to CLI.
    pool_dir="$(realpath "$pool_dir")"

    # Defaults: restore pid + full trainer, keep GPBT-sampled config, don't reinit volatility.
    local pid_flag="--salvage-restore-pid-state"
    local opt_flag="--salvage-restore-full-trainer-state"
    local donor_flag="--no-salvage-restore-donor-config"
    local volatility_flag="--no-salvage-reinit-volatility-heads"
    while [ $# -gt 0 ]; do
        case "$1" in
            --no-pid) pid_flag="--no-salvage-restore-pid-state"; shift ;;
            --no-optimizer) opt_flag="--no-salvage-restore-full-trainer-state"; shift ;;
            --reinit-volatility) volatility_flag="--salvage-reinit-volatility-heads"; shift ;;
            --donor-config) donor_flag="--salvage-restore-donor-config"; shift ;;
            *) echo "Unknown salvage-restart arg: $1"; return 1 ;;
        esac
    done

    stop
    start \
        --salvage-seed-pool-dir "$pool_dir" \
        "$pid_flag" "$opt_flag" "$donor_flag" "$volatility_flag"
}

_active_trial_dir() {
    ls -td "$WORK_DIR"/tune/train_trial_*/ 2>/dev/null | head -1 | sed 's:/$::'
}

_trial_replay_dir() {
    local trial_dir="$1"
    PYTHONPATH=. python3 - "$CONFIG" "$trial_dir" <<'PY'
import sys
from pathlib import Path

from chess_anti_engine.tune.replay_exchange import _trial_replay_shard_dir
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

cfg = flatten_run_config_defaults(load_yaml_file(sys.argv[1]))
print(_trial_replay_shard_dir(config=cfg, trial_dir=Path(sys.argv[2])))
PY
}

best_save() {
    if [ $# -lt 1 ]; then
        echo "Usage: $0 best-save LABEL [--iter N]"
        return 1
    fi
    local label="$1"; shift
    local want_iter=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --iter) want_iter="$2"; shift 2 ;;
            *) echo "Unknown best-save arg: $1"; return 1 ;;
        esac
    done

    local trial_dir
    trial_dir="$(_active_trial_dir)"
    if [ -z "$trial_dir" ] || [ ! -d "$trial_dir" ]; then
        echo "No active trial dir under $WORK_DIR/tune/ (needed for replay-shard copy)"
        return 1
    fi
    # Snapshots live under AUTO_BEST_REGRET_DIR (cross-trial, survives Ray rotation).
    # best-save promotes one of those auto-saved rolling snapshots into a permanent
    # pool and additionally copies the current replay shards alongside.
    local best_src=""
    if [ -n "$want_iter" ]; then
        best_src="$(ls -d "$AUTO_BEST_REGRET_DIR"/regret_*_iter"$want_iter" 2>/dev/null | head -1)"
        if [ -z "$best_src" ]; then
            echo "No best_regret snapshot with iter=$want_iter in $AUTO_BEST_REGRET_DIR/"
            return 1
        fi
    else
        # Pick the lowest-regret entry (ls sort treats regret_0.xxxx_... lexicographically, which matches numeric for the 4-digit fixed format)
        best_src="$(ls -d "$AUTO_BEST_REGRET_DIR"/regret_* 2>/dev/null | sort | head -1)"
        if [ -z "$best_src" ]; then
            echo "No best_regret snapshots in $AUTO_BEST_REGRET_DIR/"
            return 1
        fi
    fi
    if [ ! -f "$best_src/trainer.pt" ]; then
        echo "Snapshot missing trainer.pt: $best_src"
        return 1
    fi

    local replay_src
    replay_src="$(_trial_replay_dir "$trial_dir")"

    local pool="$BEST_POOLS_DIR/$label"
    if [ -d "$pool" ]; then
        echo "Pool already exists: $pool (choose a different LABEL or rm it first)"
        return 1
    fi
    mkdir -p "$pool/seeds/slot_000"
    cp "$best_src/trainer.pt" "$pool/seeds/slot_000/trainer.pt"
    [ -f "$best_src/pid_state.json" ] && cp "$best_src/pid_state.json" "$pool/seeds/slot_000/pid_state.json"
    [ -f "$best_src/rng_state.json" ] && cp "$best_src/rng_state.json" "$pool/seeds/slot_000/rng_state.json"
    [ -f "$best_src/meta.json" ] && cp "$best_src/meta.json" "$pool/seeds/slot_000/meta.json"

    local shards_copied=0
    if [ -d "$replay_src" ]; then
        mkdir -p "$pool/seeds/slot_000/replay_shards"
        cp -r "$replay_src"/. "$pool/seeds/slot_000/replay_shards/"
        shards_copied="$(ls "$pool/seeds/slot_000/replay_shards/" 2>/dev/null | wc -l)"
    fi

    # Read meta.json fields for the manifest
    local regret="null" iter_v="null" winrate="null" opp_str="null"
    if [ -f "$best_src/meta.json" ]; then
        regret=$(python3 -c "import json; print(json.load(open('$best_src/meta.json')).get('regret','null'))")
        iter_v=$(python3 -c "import json; print(json.load(open('$best_src/meta.json')).get('iter','null'))")
        winrate=$(python3 -c "import json; print(json.load(open('$best_src/meta.json')).get('ema_winrate','null'))")
        opp_str=$(python3 -c "import json; print(json.load(open('$best_src/meta.json')).get('opp_strength_ema','null'))")
    fi

    python3 - <<PY
import json, time
from pathlib import Path
p = Path("$pool/manifest.json")
p.write_text(json.dumps({
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "label": "$label",
    "source_trial_dir": "$trial_dir",
    "source_best_regret_snapshot": "$best_src",
    "metric": "wdl_regret",
    "top_n": 1,
    "entries": [{
        "slot": 0,
        "metric": $([ "$regret" = "null" ] && echo "null" || echo "$regret"),
        "training_iteration": $([ "$iter_v" = "null" ] && echo "null" || echo "$iter_v"),
        "seed_dir": "seeds/slot_000",
        "copied_replay_shards": $shards_copied,
        "result_row": {
            "wdl_regret": $([ "$regret" = "null" ] && echo "null" || echo "$regret"),
            "pid_ema_winrate": $([ "$winrate" = "null" ] && echo "null" || echo "$winrate"),
            "opponent_strength": $([ "$opp_str" = "null" ] && echo "null" || echo "$opp_str")
        }
    }]
}, indent=2, sort_keys=True))
PY
    echo "Saved best pool: $pool"
    echo "  regret=$regret iter=$iter_v winrate=$winrate shards=$shards_copied"
    echo "Restore with: ./scripts/train.sh salvage-restart $pool"
}

best_list() {
    local any=0
    # Permanent pools (manually promoted via best-save).
    if [ -d "$BEST_POOLS_DIR" ]; then
        echo "=== $BEST_POOLS_DIR (permanent pools, with replay shards) ==="
        for pool in "$BEST_POOLS_DIR"/*/; do
            [ -d "$pool" ] || continue
            any=1
            local label size
            label="$(basename "$pool")"
            size="$(du -sh "$pool" 2>/dev/null | awk '{print $1}')"
            if [ -f "$pool/manifest.json" ]; then
                python3 - <<PY
import json
m = json.load(open("$pool/manifest.json"))
e = (m.get("entries") or [{}])[0]
regret = e.get("metric")
it = e.get("training_iteration")
shards = e.get("copied_replay_shards", 0)
rr = e.get("result_row") or {}
winrate = rr.get("pid_ema_winrate")
print(f"  regret={regret}  iter={it}  winrate={winrate}  shards={shards}")
PY
            else
                echo "  (no manifest.json)"
            fi
            printf '%-30s %s\n' "$label" "$size"
        done
    fi

    # Rolling auto-save (top-N lowest regret, written every iter, no replay).
    # Usable directly with salvage-restart; its manifest.json lists all slots.
    if [ -d "$AUTO_BEST_REGRET_DIR" ] && [ -f "$AUTO_BEST_REGRET_DIR/manifest.json" ]; then
        any=1
        local size
        size="$(du -sh "$AUTO_BEST_REGRET_DIR" 2>/dev/null | awk '{print $1}')"
        echo
        echo "=== $AUTO_BEST_REGRET_DIR (rolling top-N auto-save, no replay shards) ==="
        python3 - <<PY
import json
m = json.load(open("$AUTO_BEST_REGRET_DIR/manifest.json"))
for e in (m.get("entries") or []):
    regret = e.get("metric")
    it = e.get("training_iteration")
    rr = e.get("result_row") or {}
    winrate = rr.get("pid_ema_winrate")
    seed = e.get("seed_dir") or "?"
    print(f"  regret={regret}  iter={it}  winrate={winrate}  slot={seed}")
PY
        printf '%-30s %s\n' "(auto-save dir)" "$size"
    fi

    if [ "$any" = "0" ]; then
        echo "No pools under $BEST_POOLS_DIR or $AUTO_BEST_REGRET_DIR"
    fi
    return 0
}

case "${1:-status}" in
    start)            shift; start "$@" ;;
    stop)             stop ;;
    restart)          shift; stop; start "$@" ;;
    status)           status ;;
    log)              tail -f "$LOG" ;;
    salvage-export)   shift; salvage_export "$@" ;;
    salvage-restart)  shift; salvage_restart "$@" ;;
    best-save)        shift; best_save "$@" ;;
    best-list)        best_list ;;
    *) echo "Usage: $0 {start|stop|restart|status|log|salvage-export|salvage-restart|best-save|best-list}" ;;
esac
