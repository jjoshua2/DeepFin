#!/bin/bash
# Monitor PBT2 run: check progress of the latest experiment
# Writes status to runs/pbt2_small/monitor.log

TUNE_DIR="/home/josh/projects/chess/runs/pbt2_small/tune"
LOG="/home/josh/projects/chess/runs/pbt2_small/monitor.log"

check_progress() {
    echo "========================================"
    echo "PBT2 Monitor Check: $(date)"
    echo "========================================"

    # Check if any python training process is running
    PIDS=$(pgrep -f "chess_anti_engine.run" 2>/dev/null)
    if [ -z "$PIDS" ]; then
        echo "WARNING: No chess_anti_engine.run process found!"
        echo "STATUS: DEAD"
        return 1
    fi
    echo "Process PIDs: $PIDS"

    # Check GPU utilization
    if command -v nvidia-smi &>/dev/null; then
        echo ""
        echo "GPU Status:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null
    fi

    # Find the latest experiment prefix (most recently modified trial dir)
    LATEST_DIR=$(ls -dt "$TUNE_DIR"/train_trial_*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_DIR" ]; then
        echo "WARNING: No trial directories found!"
        echo "STATUS: DEAD"
        return 1
    fi
    PREFIX=$(basename "$LATEST_DIR" | grep -oP 'train_trial_\K[a-f0-9]+(?=_)')

    echo ""
    echo "Experiment prefix: $PREFIX"
    echo "Per-trial status:"
    echo "---"

    TOTAL_TRIALS=0
    ACTIVE_TRIALS=0
    MAX_ITER=0
    BEST_LOSS=999999
    BEST_TRIAL=""

    for d in "$TUNE_DIR"/train_trial_${PREFIX}_*/; do
        [ -d "$d" ] || continue
        TOTAL_TRIALS=$((TOTAL_TRIALS + 1))

        trial=$(basename "$d" | grep -oP '000\d+')
        csv=$(find "$d" -name "progress.csv" -print -quit 2>/dev/null)

        if [ -z "$csv" ] || [ ! -s "$csv" ]; then
            echo "  Trial $trial: no progress data"
            continue
        fi

        # Read header to find column indices dynamically
        header=$(head -1 "$csv" 2>/dev/null)
        last=$(tail -1 "$csv" 2>/dev/null)

        # Use awk to extract by column name
        get_col() {
            echo "$header" | awk -F',' -v name="$1" '{for(i=1;i<=NF;i++) if($i==name) print i}'
        }

        col_iter=$(get_col "iter")
        col_loss=$(get_col "train_loss")
        col_wr=$(get_col "pid_ema_winrate")
        col_sf_acc=$(get_col "sf_move_acc")
        col_rand=$(get_col "random_move_prob")
        col_strength=$(get_col "opponent_strength")
        col_skill=$(get_col "skill_level")

        iter=$(echo "$last" | awk -F',' -v c="${col_iter:-1}" '{print $c}')
        loss=$(echo "$last" | awk -F',' -v c="${col_loss:-13}" '{print $c}')
        wr=$(echo "$last" | awk -F',' -v c="${col_wr:-12}" '{print $c}')
        sf_acc=$(echo "$last" | awk -F',' -v c="${col_sf_acc:-20}" '{print $c}')
        rand_prob=$([ -n "$col_rand" ] && echo "$last" | awk -F',' -v c="$col_rand" '{print $c}' || echo "n/a")
        strength=$([ -n "$col_strength" ] && echo "$last" | awk -F',' -v c="$col_strength" '{print $c}' || echo "n/a")
        skill=$([ -n "$col_skill" ] && echo "$last" | awk -F',' -v c="$col_skill" '{print $c}' || echo "n/a")

        # Track max iter
        if [ -n "$iter" ] && [ "$iter" -gt "$MAX_ITER" ] 2>/dev/null; then
            MAX_ITER=$iter
        fi

        # Check file modification time
        MOD_AGE=$(( $(date +%s) - $(stat -c %Y "$csv") ))

        STATUS="ok"
        if [ "$MOD_AGE" -gt 600 ]; then
            STATUS="STALE (${MOD_AGE}s since last write)"
        else
            ACTIVE_TRIALS=$((ACTIVE_TRIALS + 1))
        fi

        echo "  Trial $trial: iter=$iter loss=$loss wr=$wr rand=$rand_prob strength=$strength skill=$skill sf_acc=$sf_acc age=${MOD_AGE}s [$STATUS]"
    done

    echo ""
    echo "Summary: $ACTIVE_TRIALS/$TOTAL_TRIALS trials active, max_iter=$MAX_ITER"
    echo "Target: 2200 iterations, progress: $((MAX_ITER * 100 / 2200))%"

    if [ "$ACTIVE_TRIALS" -eq 0 ] && [ "$TOTAL_TRIALS" -gt 0 ]; then
        echo "WARNING: All trials appear stale!"
        echo "STATUS: STALLED"
        return 1
    fi

    echo "STATUS: OK"
    return 0
}

check_progress 2>&1 | tee -a "$LOG"
