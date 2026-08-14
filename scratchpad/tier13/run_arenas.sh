#!/usr/bin/env bash
# Tier-13 three-way round robin. Frozen commands from ledger 726c069a1 plus the
# --pgn-out added by amendment #2. Run back-to-back in one session so all three
# contrasts share one instrument (pin 3).
#
# Launched detached with setsid; progress goes to $LOG.
set -u

cd /home/josh/projects/chess || exit 1
B=scratchpad/tier13/banked
LOG=scratchpad/tier13/arena_run_$(date +%Y%m%d_%H%M).log
mkdir -p scratchpad/tier13/pgn

echo "=== yaml identity at start (pin 3) ===" | tee -a "$LOG"
sha256sum configs/pbt2_small.yaml | tee -a "$LOG"

# ⚑ --pgn-candidate-name / --pgn-reference-name are REQUIRED, not cosmetic.
# Every arm's checkpoint is named checkpoint_000099/trainer.pt, so the default
# engine identity (last two path components) collides across arms and the PGN
# writer refuses to run. Left to a default that did NOT refuse, Ordo would have
# pooled two arms into one player. The names are also the exact strings the
# pinned Ordo anchor uses (-A arm_A_iter100), so they must match it.
run_one () {
    local cand="$1" ref="$2" label="$3"
    echo "" | tee -a "$LOG"
    echo "=== $label  $(date -Is) ===" | tee -a "$LOG"
    PYTHONPATH=. python3 scripts/arena_standard.py \
        --candidate "$B/$cand/checkpoint_000099/trainer.pt" \
        --reference "$B/$ref/checkpoint_000099/trainer.pt" \
        --mode matched_sims --sims 32 --search-shape training \
        --games 1600 --seed 42 --max-concurrent-games 16 \
        --label "$label" \
        --pgn-candidate-name "$cand" --pgn-reference-name "$ref" \
        --pgn-out "scratchpad/tier13/pgn/$label.pgn" >> "$LOG" 2>&1
    local rc=$?
    echo "ARENA-DONE $label rc=$rc $(date -Is)" | tee -a "$LOG"
    if [ "$rc" -ne 0 ]; then
        echo "ARENA-FAILED $label rc=$rc — stopping the sequence rather than running the rest against a broken instrument" | tee -a "$LOG"
        return "$rc"
    fi
    tail -1 runs/arena_results.jsonl >> scratchpad/tier13/banked/arena_rows.jsonl
    return 0
}

run_one arm_B_iter100 arm_A_iter100 tier13_BvsA_iter100 || exit 1
run_one arm_C_iter100 arm_B_iter100 tier13_CvsB_iter100 || exit 1
run_one arm_C_iter100 arm_A_iter100 tier13_CvsA_iter100 || exit 1

echo "" | tee -a "$LOG"
echo "=== yaml identity at end (must equal the start value) ===" | tee -a "$LOG"
sha256sum configs/pbt2_small.yaml | tee -a "$LOG"
echo "ARENA-ALL-DONE $(date -Is)" | tee -a "$LOG"
