# Tier-13 arm C: iteration 100 → arenas. Ordered runbook.

Written at iter 90 so the sequence is decided before the moment arrives. Every step's ORDER matters;
the reasons are given because a step whose reason is forgotten gets skipped.

Ledger anchors: `726c069a1` (frozen commands + pins 1-6), amendment #2 and its two addenda
(`1c643d0ff`, `c082808fb`, `54480e119`, `7799b9323`), resume `34fdb2f59`.

---

## 1. Bank the training log — BEFORE anything else

    cp /tmp/chess_training.log scratchpad/tier13/chess_training_armC_iter100_$(date +%Y%m%d_%H%M).log

**Why first:** `/tmp/chess_training.log` is TRUNCATED by every `train.sh start`. If the next start
happens before this copy, the whole arm's log is gone. This has cost us a log before.

## 2. Stop

    ./scripts/train.sh stop

Expect workers to drain in ~45 s. ⚑ Read the teardown output for
`worker(s) (...) recorded NOTHING this teardown; their in-flight games were DISCARDED` — it fired on
the 22:15 stop. Record whether it fires again; if it does NOT, that is a data point for task #77/#78
(the C14b resume path capturing in-flight games), not a formality.

## 3. Bank the iter-100 milestone

    D=scratchpad/tier13/banked/arm_C_iter100
    mkdir -p $D
    T=$(ls -d runs/tier13_arm_C/tune/train_trial_*/ | head -1)
    cp -r $T/checkpoint_000099 $D/          # index = iter - 1
    cp $T/params.json $T/progress.csv $D/

**Verify before moving on:** `trial_meta.json` `global_iter == 990` (donor 890 + 100), and
`$D/checkpoint_000099/trainer.pt` exists at the top level of the copied dir (the arena commands point
at `trainer.pt`). ⚑ Ray keeps only 6 checkpoints — this copy is not optional and is not deferrable.

## 4. Confirm the GPU is actually released

    ps -eo pid,cmd --no-headers | grep -E 'chess_anti_engine|ray::|raylet|gcs_server' | grep -v grep
    for p in $(ls /proc | grep -E '^[0-9]+$'); do ls -l /proc/$p/fd 2>/dev/null | grep -q 'dxg\|nvidia' && echo "HOLDS GPU: $p"; done

⚑ **Do NOT use `nvidia-smi` memory as the ownership test.** Under WSL2 it reports the whole physical
card including Windows-host processes and cannot attribute them, so a non-zero reading is expected and
proves nothing. Zero hits from the `/proc` sweep is the test. (Measured 2026-08-13: 20.5 GB shown with
zero processes of ours.)

## 5. Release the CPU hold on PR #423

Tell both agents the box is quiet. The three outstanding gates are: full suite on the branch, full
suite on the merge result, and repo-wide `lint.sh` on the MERGE RESULT (CI covers the branch only and
never re-runs when the base advances). Require a real summary line
(`--override-ini=addopts=`), not counts derived from progress characters, and require the wrapper to
exit with pytest's status — a trailing `echo` makes a SIGTERM'd run notify as "exit code 0".

⚑ **They may run CONCURRENTLY with the arenas (step 8), but NOT with training.** The asymmetry is
real and is the #189 finding: under `--mode matched_sims` each side gets N simulations regardless of
speed, and there is no time-based decision anywhere in the arena (`--max-plies 300`, no
`--max-seconds`, Syzygy adjudication), so CPU contention costs wall clock and **cannot** move the Elo.
Training had no such protection — its Stockfish labelling sits at nice 19 on the critical path.
Condition: `CUDA_VISIBLE_DEVICES=""` and `nice -n 19`, so nothing competes for GPU memory
(paired/compiled arenas have OOMed this box twice).

## 6. Merge `origin/main` into the live branch — the yaml resolution is PREPARED

    git merge origin/main            # conflicts ONLY on configs/pbt2_small.yaml
    cp scratchpad/tier13/merge_prep/pbt2_small.RESOLVED.yaml configs/pbt2_small.yaml
    git add configs/pbt2_small.yaml && git commit

⚑ Valid ONLY for `live c082808fb × main 2c700dd70` (sha256
`b588d276f0626dc70b462aa5d54b7081868ad6907bae4d2b560e6ef491211acc`). **If either side has moved,
re-do the resolution by hand** — do not reuse the file.

**ACCEPTANCE CRITERION, pre-committed:** flatten the merged yaml and diff against the pre-merge live
yaml → **added 1 (`diff_focus_norm_shared`), removed 0, CHANGED 0.** Any non-zero `changed` or
`removed` means the resolution reverted a deployed value — stop and re-resolve. Neither `-X ours` nor
`-X theirs` is correct here: live has `diff_focus_norm_enabled: true` (deployed, #171/#173) and main
has the PR default `false`.

## 7. REBUILD THE C EXTENSION — hard blocker, not hygiene

    python3 scripts/build_production_extensions.py

#409 raises `_REQUIRED_MCTS_ABI` **3 → 4** (`mcts/gumbel_c.py:189`). Until this runs, the merged
Python meets an ABI-3 `.so` and **every arena and every training start dies at import**. NOT
`pip install -e .` — the .venv setuptools lacks PEP 660.

Then re-dry-run the merged yaml (`flatten_run_config_defaults` + `TrialConfig.from_dict`) and record
`sha256sum configs/pbt2_small.yaml` — pin 3 requires the three arena rows to carry an equal
`config_hash`.

## 8. The three arenas — frozen commands plus `--pgn-out`

Back-to-back, in one session, from the LIVE tree. `B=scratchpad/tier13/banked`.

    PYTHONPATH=. python3 scripts/arena_standard.py \
      --candidate $B/arm_B_iter100/trainer.pt --reference $B/arm_A_iter100/trainer.pt \
      --mode matched_sims --sims 32 --search-shape training \
      --games 1600 --seed 42 --max-concurrent-games 16 --label tier13_BvsA_iter100 \
      --pgn-out scratchpad/tier13/pgn/tier13_BvsA_iter100.pgn

C-vs-B and C-vs-A identical apart from paths and `--label` / `--pgn-out` name. All ten flags verified
present on the post-merge script (`7799b9323`). ~80-100 min each.

**After each:** `tail -1 runs/arena_results.jsonl` must show `games: 1600, pairs: 800,
truncated: false` and the two DISTINCT full checkpoint paths; copy the row into
`scratchpad/tier13/banked/arena_rows.jsonl` the same session.

## 9. Readout — dual instrument, rules already pinned

**PRIMARY: paired pentanomial**, per contrast. Every verdict is decided by the JSONL row and nothing
else. Trajectory ⇔ `elo_ci95[0] > 0` AND `elo < +40`; full win ⇔ `elo_ci95[0] > 0` AND `elo >= +40`;
kill ⇔ `elo < -15`. Final row only — never a rolling intermediate.

**SECONDARY: Ordo pooled fit** over the three PGNs + the pair-level block bootstrap
(`scripts/ordo_pooled_fit.py`). `-W -D -s 1000 -M -A arm_A_iter100 -a 0 -z 200.2409`.
⚑ `-M` is REQUIRED with `-D`+`-s` (≈1/30 datasets otherwise hang at 100% CPU, deterministically —
re-running will not clear it). ⚑ `-z 200.2409`, NOT the default 202.

⚑ **Ordo NEVER overturns the pentanomial.** A sign disagreement is an INSTRUMENT finding to resolve
before Tier-13 is quoted anywhere, not a licence to pick the friendlier number. Agreement criterion,
stated before the data: the pentanomial point estimate must lie inside Ordo's 95% interval on all
three contrasts.

**Free transitivity check:** (B−A) + (C−B) must reconcile with (C−A) on both instruments.

## 10. Confounds to carry into every verdict

- Arm C iters **80-86 excluded** from per-iteration comparisons — drain transient plus a SECOND
  per-thread diff-focus warm-up (~131k plies, ~37 min) that arms A and B each paid only once. ⚑ Not
  narrowed on the strength of timing recovering at iter 81; the window was pre-committed on the
  recording mechanism, not on wall clock.
- Not fully excludable: those rows reach arm C's iter-100 weights, which is what the arenas read.
- If C−A or C−B lands in the trajectory band, the pinned second-seed re-run launches **without an
  intervening pause**, so seed 2 carries one warm-up like A and B.
