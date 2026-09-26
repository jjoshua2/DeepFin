# Matched exact-epoch schedule verification

`verify_matched_schedule.py` reads shard metadata only and reuses the pinned
wise-cloud planner. It never launches training or decodes features. Future runs
use two metadata workers, numeric/Torch thread caps of two, and Blosc one.

Current evidence: `matched_schedule_E0_S0.json` (366.19 seconds) verifies all
2,309 actual staged links and full ordered `game_id`/`has_game_id` columns against
18,910,484 source rows. E0's recomputed physical-path plan equals its saved plan
and realized hash. S0's metadata matches; S0 training was still incomplete.

The common canonical plan hash is:
`dc687fc333295dee565d19bb4f20da5aa95479dba3aacc5499c22a4004acc64f`.
Physical plan hashes differ across corpus paths; equality of each run's own
plan/realized hashes alone does **not** establish equal schedules across arms.

The receipt SHA256 is
`5cb4a1fdd2b2f6f148c3662c602b8d590a817b8b15119fa767d5e43d26658e1b`.
Its executed verifier is preserved as `verify_matched_schedule.executed_3fb25356.py`
(SHA256 `3fb25356a573641178c120b36ff17ee296e6b119e5ade9632dab534babc7ec73`).
The current verifier adds only an explicit Blosc cap and its own SHA to future
receipts (SHA256 `4e8e27e861021dd4c75a1a21404ff9aed0d55775e0212def585bc21b1d8cde18`).
Negative fixtures in `/tmp/deepfin-matched-schedule-negative-check.py` reject
swapped links, equal-count game-row permutations, and missing game identities.

## Later commands

Run only against **published** G20 directories. Do not inspect `.writing` as a
completed corpus. Use a new output filename for each observation; preserve older
receipts. No further E0 scan is needed: compare subsequent canonical hashes with
the banked E0 receipt above.

```bash
repo=/home/josh/projects/chess
schedule_root="$repo/scratchpad/bt4_joint20/global_run02"
corpus_root="$repo/data/nnue_derived/armB"
run_root="$repo/runs/armB"

# After G20T1 publication; repeat with G20T05 only after its publication.
arm=G20T1
nice -n 19 /usr/bin/python3 "$schedule_root/verify_matched_schedule.py" \
  --corpus "$arm=$corpus_root/qtemp_0.0005_hist_20m_bt4_global_$arm" \
  --output "$schedule_root/matched_schedule_published_$arm.json"

# After S0 and both global arms have completed training.
nice -n 19 /usr/bin/python3 "$schedule_root/verify_matched_schedule.py" \
  --run "S0=$run_root/qtemp_0.0005_hist_20m_bt4_global_S0_epoch_v2" \
  --run "G20T1=$run_root/qtemp_0.0005_hist_20m_bt4_global_G20T1_epoch_v2" \
  --run "G20T05=$run_root/qtemp_0.0005_hist_20m_bt4_global_G20T05_epoch_v2" \
  --output "$schedule_root/matched_schedule_completed_S0_G20.json"
```

`--corpus` verifies prospective staging only; `--run` verifies actual staged links
and, when present, saved summary completion/realized schedule evidence. Missing
summaries remain explicitly incomplete; this verifier does not replace checkpoint
validation by the frozen experiment driver.

## What this establishes and does not

Exactly equal normalized planner inputs, full ordered game columns, and the frozen
scheduler's independent seeded row RNG/stable segment order imply the same
`(original source, shard, row offset)` schedule under the same NumPy RNG runtime.
The observed runtime was Python 3.10.12, NumPy 1.26.2, Torch 2.11.0+cu128.
Historical E0 NumPy provenance is not independently stamped; reproducing its saved
plan does not separately measure its historical row-RNG stream.

This is a code-backed row-order inference, not a bank of realized row offsets.
It does not independently compare full feature/non-policy contents or detect
within-game content permutations that leave all game IDs unchanged. Separate
support is the pinned mixer's source copy followed by policy-only mutation and
`../global_run01/G20T1.first-shard-integrity.json`, which compares all 16 untouched
arrays on the first 8,192-row shard only. No whole-corpus untouched-array equality
measurement is claimed. G20 comparison and completion evidence remain missing
until the later checks succeed. No strength, winner, or promotion follows from
schedule equality.
