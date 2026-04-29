#!/bin/bash
# Usage: ./scripts/train.sh [start|stop|status|log|restart|salvage-export|salvage-restart|best-save|best-list]
#
# salvage-export [--top-n N] [--out DIR] [--metric KEY] [--no-copy-replay]
#   Export top-N trial seeds (checkpoint + replay) from the current tune run
#   into a salvage pool. Does not touch the running process. If --out is
#   omitted the pool lands under $WORK_DIR/salvage/<run-id>_<timestamp>/.
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
WORK_DIR="${TRAIN_WORK_DIR:-runs/pbt2_small}"
BEST_POOLS_DIR="${TRAIN_BEST_POOLS_DIR:-data/best_pools}"
# Rolling top-N auto-save written by _update_best_regret_checkpoints. Matches
# TrialConfig.best_regret_checkpoints_dir default in trial_config.py.
AUTO_BEST_REGRET_DIR="${TRAIN_AUTO_BEST_REGRET_DIR:-data/best_regret_checkpoints}"

cd "$(dirname "$0")/.."

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
    nohup python3 -m chess_anti_engine.run \
        --config "$CONFIG" --mode tune "${extra_args[@]}" \
        > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "Started PID $! — log: $LOG"
}

stop() {
    if ! running; then
        echo "Not running"
        return 0
    fi
    local pid=$(cat "$PIDFILE")
    echo "Stopping PID $pid ..."
    kill "$pid" 2>/dev/null || true
    sleep 2
    if kill -0 "$pid" 2>/dev/null; then
        echo "Force killing ..."
        kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
    ray stop 2>/dev/null || true
    sleep 1
    pkill -9 -f 'ray::' 2>/dev/null || true
    pkill -9 -f 'raylet' 2>/dev/null || true
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
    while [ $# -gt 0 ]; do
        case "$1" in
            --top-n) top_n="$2"; shift 2 ;;
            --out) out_dir="$2"; shift 2 ;;
            --metric) metric="$2"; shift 2 ;;
            --no-copy-replay) copy_replay="--no-salvage-copy-replay"; shift ;;
            *) echo "Unknown salvage-export arg: $1"; return 1 ;;
        esac
    done

    local out_args=()
    [ -n "$out_dir" ] && out_args=(--salvage-out-dir "$out_dir")

    echo "Exporting top-$top_n salvage seeds from $WORK_DIR (metric=$metric)..."
    PYTHONPATH=. python3 -m chess_anti_engine.run \
        --config "$CONFIG" --mode salvage \
        --work-dir "$WORK_DIR" \
        --salvage-top-n "$top_n" \
        --salvage-metric "$metric" \
        "${out_args[@]}" \
        "$copy_replay"
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

    local replay_src="$WORK_DIR/replay/$(basename "$trial_dir")/replay_shards"

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
