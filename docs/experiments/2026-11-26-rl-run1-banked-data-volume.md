# 2026-11-26: RL run 1 — more data, used once, from the same actor (pre-registration)

**Status: pre-registration, revision 2 (after the statistics review), for review.** Nothing is launched. Launch needs:
- this record merged;
- the admission record merged (`2026-11-26-banked-collections-admission.md`, #1523);
- the pooled bank built from it and read back in staging;
- the real kC pool rebuilt from the merged admission record, the launch-time host-RAM floor and the GPU memory smoke (below);
- a free GPU.

Only CPU work has been done for this record.

## The question

Generation 2 trained `gen1_baked.pt` (sha256 `8dd6ac8c…`) on one 8,000-game collection that the same net played.
- vs gen 1: +1.14 [−5.04, +7.32] Elo.
- vs the SL base: +3.85 [−2.46, +10.17].
- Licensed claim: "no second measured step".

**Hypothesis: more data, used once, makes the step.** The same recipe, from the same parent, trained once over 3.47×
(or 2.48×) as many stored rows played by the same actor, makes a net that beats the parent.

**What varies, and what cannot be separated.** All rows are played by `gen1_baked.pt` at the collector pin
`0c795fb7a` with an identical collect recipe, and each stored row is used exactly once. Under exactly-once, more rows
means proportionally more optimizer updates at the same flat LR. So **the primary cell does not separate "more
distinct rows" from "more updates"**. The reading is about "more data, used once". A control that separates the two
is registered below, conditional on an "improves".

**The owner's rule, for the record:** start with exactly one use per stored row. At most two total views, mirrors
included, is the ceiling, not the starting recipe.

Everything else is generation 2's B4/B5 recipe, unchanged:
- parent `gen1_baked.pt`; Zq flags;
- `--presentations 1`, mirror 0.5, batch 512;
- LR 3e-6 flat, no warm-up, Aurora with fresh state;
- hold-out 0.05 by game, the same recal fit, baked files only in matches.

**Evidence log first.** I grepped `docs/experiments/2026-11-21-rl-loop-evidence-log.md` for volume, rows, views,
exact-once, second pass and 16,000.
- No row measures a volume dose from one actor.
- The pilot's 1,412 updates on 722,742 rows is a single volume, not a dose.
- The amendment's dose-response in games was planned and never run.
- So this is a new measurement.

## Design (revision 2)

**Deviation from the earlier outline, stated plainly.**
- Revision 1 registered kA (1.48×) as the cell and left kB / kC to addenda.
- The review showed that kA has only about an 8–21% chance of reading "improves" for plausible effects, and an
  800-game point on it adds nothing. **kA is dropped.**
- The design is now:
  - one primary cell at the highest admissible dose;
  - a fresh k1 match for a slope contrast;
  - descriptive checkpoints inside the one run.

| Arm | What | Matches |
|---|---|---|
| **primary** | `gen1_baked.pt` trained once over the pooled **kC** bank (2,767,035 rows, 3.47×). The coordinator chose kC on the measurement below. kB (1,972,644 rows, 2.48×) is used only if a quiet host still fails the launch-time floor | vs the parent, 12,800 games (800-game damage screen first); vs the SL base only if it improves vs the parent |
| **fresh k1** | the existing `gen2_baked.pt` (1×), no retrain | vs the parent, 12,800 games, same seed and size as the primary |
| checkpoints | the primary run's weights at 1,478 / 2,189 / 3,675 updates (and the final) | none registered; recal fits only (below) |

**kC vs kB was chosen by measurement: kC.** Measured 2026-11-26.
- **Method:**
  - The pin's own `learn.load_window(pool, 1, settings)` + `prepare_targets`, with generation 2's exact flags parsed
    by the pin's parser (`--seed 2026112601 --games 27840`).
  - CPU only, one thread, niced, a MemAvailable watchdog at 12 GiB that never fired.
  - The load ran on a kC pool built in staging (`rl-run1-volume.staging/rss/pool_kC`, 150 shards, 2,767,035 rows).
- **Result** (`~/chess-artifacts/operations/rl-run1-volume.staging/rss/rss_kC.json` [`7655061d`], log `~/chess-artifacts/operations/rl-run1-volume.staging/rss/rss_kC.log` [`aebd11e9`], script `~/chess-artifacts/operations/rl-run1-volume.staging/rss/rss_load.py` [`a8ef886e`]):
  - train rows 2,636,964, hold-out rows 130,071, planned updates 5,151;
  - **peak RSS (`ru_maxrss` = `VmHWM`) 60.46 GiB;**
  - **the loaded arrays total 72.14 GiB.** That is more than the peak RSS, because some preallocated pages are not
    touched by the load but are touched in training. **72.14 GiB is taken as the conservative footprint.**
  - Host MemAvailable was 90.3 GiB before the load, and fell to a minimum of 33.1 GiB with a 1-thread CPU check
    running beside it.
- **The staging pool is a measurement artefact, not the run's input.** Because the admission record had not merged,
  it was built under the tool's test override, bound to #1523's head file (sha256 `dfef9a7e…`), and its pool block
  says so. **The pool for the real run is rebuilt after #1523 merges,** with `ADMISSION_SHA256` set to the merged
  file's digest. Its counts and shard digests must equal the staging pool's (same sources, same order); any
  difference is a refusal to investigate, not to explain away.

**Host-RAM guards during the run:**
- **Launch-time floor: `MemAvailable` ≥ 86 GiB** (72.14 + 12 GiB margin + ≈ 2 GiB for the trainer process),
  checked by the launcher immediately before **the 3-update GPU smoke's** `.started` and again before the training
  step's (the smoke loads the same window).
  - A miss writes `HOLD_HOST_RAM`. The operator quiets the host and retries.
  - kB is used only if a quiet host still fails the floor. The switch to kB is recorded, with its own measured load.
- **No agent harness or test run may be active during the load phase.** The coordinator enforces this with the hold
  file.
- **A MemAvailable watchdog during training, and what the pin supports.**
  - The pin's `--pause-marker` is checked only between stages, so it cannot stop a training phase in flight.
  - The pin's `--snapshot-updates` keeps a resumable state (trainer, optimizer, sampler cursor, RNG) at the newest
    snapshot. A killed phase resumes from there with the same plan.
  - So the launcher runs its own watchdog. Below 6 GiB MemAvailable it writes `STOP` (the pause marker, which blocks
    every later stage) and sends SIGTERM to the training process itself. The OOM killer never chooses.
  - The phase then resumes **in place** from its newest snapshot (1,478 / 2,189 / 3,675 updates), losing at most
    one snapshot interval, about 8 GPU-min. The interrupted `train/` directory is **never moved aside**: the pin's
    `train_generation` always uses `<arm>/gen_0001/train`, and `resumable_snapshot` searches only that directory's
    `snapshots/step_*`. Moving it would restart from update zero.
  - Resume facts the statistics reviewer verified at the pin: the resume key hashes the batch plan; `snapshot.json`
    is written last, so a step without it is redone; the 17 resume tests pass; a mutant on the sampler cursor is
    killed.
  - A second watchdog stop is a HOLD for the owner.

**Updates are the pin's own plan, never a hand formula.**
- The trainer at the pin plans `ceil(trainable_rows / 512)` over the rows outside the game-hash hold-out
  (`learn.planned_phase_updates`), keeping a short final batch.
- On the sealed pools that is 3,675 updates (kB) and 5,151 (kC), as computed by the reviewer.
- The launcher's check: realised updates = `planned_phase_updates(arm run dir, 1, settings)`, evaluated at the pin on
  the sealed pool.
- Revision 1's `round(0.95 × rows / 512)` would have refused a correct run (2,183 vs 2,189 for kA). It is withdrawn.

**Every other equality gate was audited for the same defect** (a formula standing in for the code path):
- pool rows and games: taken from the sources' manifests, then re-counted from the copied meta sidecars (the pin's
  reader);
- the hold-out: `holdout_mask` from the pin on the pooled meta, compared with the same function on each source;
- the optimizer take-effect check: the realised Aurora fields against what the pin builds from the flags (generation
  2 revision 6's exact rule, which already reads the pin's code path);
- the reference acceptance: the pin's own `reference.reference_collect`, not a re-implementation;
- the snapshot steps: `learn.check_snapshot_plan` against the planned updates.

No remaining gate compares a realised value to a hand formula, or to another generation's realised value.

## The pooled bank (CPU, staging)

The trainer at the pin presents ONE collection per generation (`learn.phase_generations` with `--presentations 1`).
So the primary trains with `--collect-from` on one pooled source run dir.

**The tool:** `pool_collect.py` (staging; pure stdlib + numpy and the pin's manifest and hashing functions; modelled
on `simple_rl/retarget.py`). It writes `rl-run1-volume/pool_<cell>/`:
- `gen_0000/`: copied from the gen2 bank (`gen1_baked.pt` and its manifest, re-sealed at the pool's path).
- `gen_0001/collect/shards/`: every source shard copied byte-identically (no links), renumbered in the order gen2,
  succ_pilot, succ_cont, b01_r2 (kB), fresh_indep (kC).
- `games.jsonl` and `starts.txt`: concatenated in the same order.
- The manifest:
  - summed counts and recomputed outputs;
  - `seeds.base` = `config.settings.seed` = the pool seed 2026112601;
  - `config.settings.games` = the summed games;
  - a `pool` block. Per source it records run dir, manifest / shard-tree / games / starts sha256, rows, games, seed
    and openings file. It also records the admission record used.
- Everything is written in `.pool_<cell>.partial` and renamed into place. An existing output is refused.

**Cells are bound to the merged admission record, not to a list in the tool.** The tool:
- reads the admission record and requires its sha256 to equal `ADMISSION_SHA256`. That constant is empty until the
  launch addendum sets it to the merged file's digest, so every cell is refused today;
- requires every bank of the cell to be a row of the record's Admission table marked **yes**;
- writes the record's path, sha256 and admitted list into the pool block.

**Refusals** (non-zero exit, nothing published):
- P1: a source whose recorded sha256 no longer recomputes;
- P2: an actor other than `8dd6ac8c…`, a git sha other than the pin, or a dirty tree;
- P3: a settings difference from gen2's outside the admission record's skip list, or different realised evaluator
  caps;
- P4: a game id or an opening present in two sources;
- P5: an unknown cell, an unbound or mismatched admission record, or a bank the record does not admit;
- P6: an output inside a source or on another filesystem.

**The harness** (`test/pool_harness.py`, CPU, using the pin's own reader, reference and hold-out code). Its fixtures
are one re-sealed shard of each of the five banks.
- C1: the kC pool builds.
- C2: every pooled shard equals its source shard (all keys and meta), in source order.
- C3: the pooled ledger equals the sources' ledgers in order; game ids and rows agree with the manifest.
- **C18: the pooled `starts.txt` equals the sources' starts, in order.** New in revision 2: the reviewer's mutant
  that dropped the last source's starts survived revision 1's 20/20.
- C4–C6: the pin's `reference_collect` accepts the pool with its seed and games, and refuses any other seed or count.
- C7: the hold-out marks the same rows as on the sources.
- C16: the pool block and outputs are consistent.
- C17: every copy has link count 1.
- C8: an existing output is refused unchanged.
- P1–P6: each refuses with nothing published. This includes P5 for a bank the admission record marks **no**, and for
  a record whose sha256 does not match.
- C19: the pool block records the admission record, its sha256 and the test-override flag.

**Mutants:**
- each refusal disabled;
- a byte flipped in a copy;
- sources pooled in reverse;
- a source's last shard dropped;
- `games.jsonl` without the last source;
- **`starts.txt` without the last source**;
- the pool seed left as gen2's;
- `outputs.shards` copied from gen2;
- hard links instead of copies;
- the admission list ignored;
- the admission sha256 ignored.

**Staging status (2026-11-26, CPU).** Paths are under `~/chess-artifacts/operations/rl-run1-volume.staging/`.
- Revision 1's tool and harness (control 20/20, 15/15 mutants killed) are superseded.
- Revision 2:
  - `prelaunch/pool_collect.py` [`a8e824d7`];
  - `test/pool_harness.py` [`d3cc705d`], 5-bank fixtures (`fixture_v2/`);
  - **unmutated control 24/24**;
  - `test/mutants_pool.json` [`4ec53565`], run with `test/mutants.py` [`aaec0d68`]: **18 mutants, 18 killed**, each by
    a named FAIL line;
  - results in `test/mut_pool_v2.out` [`55232da5`] and `test/mut_pool_v2/mutant_results.json`.
- The starts mutant (M16) is killed by C18. The two admission mutants (M17, M18) are killed by the P5 cases.
- After the admission record merges, the real kC pool is built bound to the merged record and read back over every
  shard (`pool_harness.py real`). The launch addendum quotes those read-backs.

## Training, checkpoints and recalibration

One arm, `rl-run1-volume/arm_<cell>`. Its flags come from a copy of generation 2's `gen2_flags.sh`, pinned by sha256,
with these exceptions:
- `--seed 2026112601`;
- `--games` = the pool's games;
- `--collect-from pool_<cell>`;
- **`--snapshot-updates 1478,2189,3675`** for kC, or `1478,2189` for kB. The pin writes `train/snapshots/step_<N>/`
  (a weights-only `model.pt`), and training is unchanged by it.

**Recalibration at every checkpoint and at the final.**
- Each is fitted with `scripts/fit_value_recal.py` at the pin: `--net` = the checkpoint's `model.pt`, `--reference`
  = `gen1_baked.pt`, **`--collect-dir rl-run1-volume/pool_kC/gen_0001/collect/`** (the sealed collection directory,
  not the pool's run dir), **`--holdout-frac 0.05`**, `--device cuda`.
- All fits are banked.
- Only the final net is baked (`bake_recal.py`, then `verify`) and played.

**Recal bounds: generation 2's fixed box plus a stability rule** (revision 2's scaled box is withdrawn: at kC it
admitted T from 0.077 to 3.61, which tests nothing).
- **R1, at every checkpoint and the final:** T ∈ [0.4951, 1.4212] and b ∈ [−0.4542, +0.2271], generation 2's box
  unchanged.
- **R2, stability:** |ln T_final − ln T_1478| ≤ 0.10 and |b_final − b_1478| ≤ 0.10.

**Basis: the banked fits show the drift is set early, then flat.**
- The truncated-S series (one run, value-anchor screen) is flat across a 4× range of updates:

| Updates | T | b | File |
|---:|---:|---:|---|
| 353 | 0.8531 | −0.1273 | `~/chess-artifacts/operations/rl-run1-value-anchor/offline/cells/recal_S_trunc_353.json` [`1e32adfa`] |
| 635 | 0.8533 | −0.1300 | `~/chess-artifacts/operations/rl-run1-value-anchor/offline/cells/recal_S_trunc_635.json` [`0dd75c1d`] |
| 988 | 0.8508 | −0.1383 | `~/chess-artifacts/operations/rl-run1-value-anchor/offline/cells/recal_S_trunc_988.json` [`ad556efb`] |
| 1,412 | 0.853 | −0.120 | the pilot's banked S fit |

- Zq: generation 1 (T 0.7036, b −0.2271, 1,412 updates) and generation 2 (T 0.6997, b −0.2467, 1,478 updates).
- So under "more of the same training", the 1,478-update fit should already be close to the final, and both inside
  generation 2's box.
- **Caveat, stated:** the dose series is S, a different value target. Zq has two points, from two generations, not a
  dose series. Whether Zq's drift is flat in updates up to 5,151 is exactly what R2 tests. The statistics reviewer
  withdrew its "unmeetable at kC" finding on this basis.

**A miss of R1 or R2 is a HOLD, never a dead gate.** If any checkpoint's or the final's fit misses R1, or the final
misses R2:
- the launcher writes `HOLD_RECAL`, with all fits banked, and stops before any match;
- the operator reads the fits, and the owner decides whether the primary plays;
- no threshold is changed in flight.

**Are the checkpoints lower-dose cells? Judgement: valid in distribution, but not independent. No matches are
registered on them.**
- The batch plan is a uniform permutation of the pool's trainable rows. With a flat LR, no warm-up and fresh
  optimizer state, nothing in the update rule depends on the total planned length.
- So the net at update U has seen exactly the first U × 512 rows of a uniform shuffle: a uniform sample of the pool
  at that size, trained once. That is distributed like an independent run on a random subset of that size.
- But a checkpoint shares the primary's data prefix, seed, hold-out and GPU noise. A checkpoint-vs-final difference
  is a within-run contrast whose correlation the match design cannot remove.
- A 1,600-game match on the 2,189-update checkpoint (SE ≈ 9 Elo) cannot read an effect of the size at stake.
- **Default no:** the checkpoints are descriptive fit and calibration points only. Their (T, b), hold-out losses and
  drift are reported as a within-run dose curve, never as cells.

## Yardstick, screen and thresholds

**Matches.** All use the follow-up launcher, `arena_standard.py` matched sims, 64 v 64, plain seats, baked files, and
the match book `rl-run1-pilot/openings/eval.pgn`.
- Primary vs parent: seed **2026112611**, 12,800 games (6,400 per seat).
- Fresh k1 (`gen2_baked.pt`) vs parent: the same seed, 12,800 games.
- Primary vs SL base: seed **2026112612**, 12,800 games, **only if** the primary improves vs the parent.

**The estimator, exactly.**
- **Decision statistic:** the registered normal interval of `scripts/arena_seat_balanced_pool.py` over opening keys.
  Per opening key, d(o) = the seat-balanced per-game excess on the score scale; the estimate is d̄ = mean_o d(o) and
  SE = sd_o(d) / √K over the K opening clusters. LB95 / UB95 = d̄ ∓ 1.96 SE on the score scale. Elo, through
  Elo(x) = −400 log10(1 / (0.5 + x) − 1), is for reporting only.
- **Contrast:** Δ̂ = d̄_P − d̄_1 on the score scale (primary minus fresh k1, both vs the parent), with
  SE = √(SE_P² + SE_1²). It passes iff Δ̂ − 1.645·SE > 0, strictly.
- **Bootstrap, only as a check:** B = 20,000 resamples, seed 2026112613, the opening key as the unit, arms resampled
  independently. It never decides.
- **Missing games:** an incomplete pair is dropped. An opening key enters an arm only with both seats complete. The
  dropped counts are reported. `--on-max-plies` is as in generation 2 (its launcher's setting, unchanged).
- **Resume:** each match resumes per seat as in generation 2 revision 4. A finished seat (exit 0, exactly n games,
  unique (pair id, half), one header, one result row with the seed, both nets' sha256 and the pin) is never replayed;
  only a missing seat plays, then readback and pooling run again. No match is ever moved aside and rerun.

**No pairing claim is made.** The review measured an opening-level correlation of 0.026 between arms, so sharing the
seed buys nothing. The shared seed is kept only so that both arms play the same book draw.

**Screen (primary vs parent only):**
- defined by pair id, never by finishing order: the pairs with `pair_id` < 200 in each seat (400 pairs, 800 games in
  all), inside the 12,800-game schedule. The screen is read only when all of them are complete in both seats;
- the cell stops iff the screen's UB95 < 0 (damage). Otherwise the same match continues to 12,800, and the screen's
  games are counted once;
- it is one look with a damage-only exit;
- **a screen never decides a small difference** (SE ≈ 9 Elo at 800 games) and never licenses "improves".

**Primary reading (vs parent, 12,800 games): improves iff LB95 > 0.**
- Damage iff UB95 < 0; adoptable iff LB95 > −10; otherwise unread.
- Power: SE ≈ 2.29 Elo, so the 80%-power minimum detectable effect is ≈ 6.3–6.4 Elo.

**Pre-registered slope contrast, one-sided.**
- Δ̂ = d̄_P − d̄_1 (above), from the two 12,800-game matches.
- The arms are independent (no pairing assumed), so SE(Δ̂) ≈ 3.24 Elo-equivalent.
- **"Volume raised the step" iff Δ̂ − 1.645·SE > 0, strictly.** MDE80 ≈ 8.1 Elo.
- The banked generation-2 reading (+1.14) motivated this experiment, so it is **not reused** in the contrast
  (regression to the mean). The fresh k1 match replaces it.

| Outcome | Licenses |
|---|---|
| primary improves vs parent | a measured step from "more data, used once" at that dose. The base match runs, and the conditional control below is registered to run |
| primary improves and the slope contrast passes | additionally, that the larger dose beat 1× under the same recipe |
| primary does not improve | "more data, used once" up to 3.47× (or 2.48×) does not read as a step. Recorded as unread or damage, never "no effect" |
| a recal HOLD | no reading; the fits are banked and the owner decides |

**Conditional control (runs only after an "improves"): 1× rows, extra updates.** It separates rows from updates.
- Design: the gen2 rows alone, each presented twice (the owner's two-view ceiling, mirrors counted), giving
  2 × 1,478 = 2,956 updates.
- It plays 12,800 games vs the parent on seed 2026112611, under the same estimator.
- **Attribution needs a direct contrast, registered now:** Δ̂_PC = d̄_P − d̄_C (primary minus control, both vs the
  parent), SE = √(SE_P² + SE_C²). **"Distinct rows beat repeated rows" iff Δ̂_PC − 1.645·SE > 0, strictly.**
  Otherwise attribution is unread. A split of significance (primary improves, control does not) is not a difference
  and licenses nothing about attribution.
- Caveat: the update counts still differ (2,956 vs 5,151), because the ceiling forbids more passes.
- The pin has no exactly-twice-per-row mode for one collection (`--presentations` counts generations). The mechanism
  is specified and tested in the addendum that registers the control.

## Confounds (named, not removed)

- **UHO is out.** It has a different opening distribution. It is the next single variable, under its own record.
- **Volume and update count move together** (above). The conditional control addresses this.
- **The recalibration is fitted on more rows** (≈ 5% of the pool's games), drawn from several banks rather than one
  collection.
- **The batch plan differs** from gen2's (pool seed). Generation 2's own retrain moved the net (`5d019e5e…` vs
  `1db00eca…`), so arm-to-arm noise includes this.
- **The successor and HOLD banks' openings are different lines from the same book.** The stage batches excluded every
  booked id, so the book is the same but the selection is not gen2's `cycle` order.
- **Duplicate inputs across banks:** 0.25% of the 5-bank pool's rows share a stored-input hash with another row.
  These are positions reached independently in disjoint games. They are kept, since exactly-once is per stored row.

## Memory, GPU time and recovery

**Host RAM.** The stock loader holds every training row.
- Measured for kC: 60.46 GiB peak RSS, 72.14 GiB of loaded arrays (the conservative footprint). The guards are a
  launch floor of MemAvailable ≥ 86 GiB, the training watchdog and the load-phase hold (above).

**GPU memory.** Batch 512 and the model are unchanged (generation 2's smoke: 16.96 GiB peak reserved, flat from update
2 to 3). The primary still runs the 3-update GPU smoke on its own pool before training:
- it writes per-update peaks, and they are kept;
- it uses generation 2's abort fraction 0.85 and the flatness rule.

**GPU-h, from measured rates** (0.3164 s/update, from generation 2's 467.65 s / 1,478; 1.45 GPU-h per 6,400 games):

| Item | kC | kB |
|---|---:|---:|
| train (5,151 / 3,675 updates) | 0.45 | 0.32 |
| 4 / 3 recal fits + bake + verify | ≈ 0.4 | ≈ 0.3 |
| primary vs parent, 12,800 | 2.9 | 2.9 |
| fresh k1 vs parent, 12,800 | 2.9 | 2.9 |
| **unconditional total** | **≈ 6.7** | **≈ 6.4** |
| primary vs base (if improves) | +2.9 | +2.9 |
| conditional control (train + 12,800) | ≈ +3.2 | ≈ +3.2 |

A primary screened out costs ≈ 0.8–1.0 h (train, fits, 800 games), plus the k1 match if it was already played. Order: train, fits, screen, the
rest of the primary match, then the k1 match.

**Recovery.** Nothing here writes to a bank:
- pools and arms are new directories under `rl-run1-volume/`;
- the sources are re-hashed by P1 at build, and by the pin's lineage check at every stage;
- fresh_indep's `procs/01/` hard links are never opened (the admission record's hazard); the pool copies only
  `shards/`;
- a failed non-training, non-match stage is moved aside by name and rerun from its inputs. An interrupted training
  phase resumes in place from its snapshot. A match resumes per seat. Neither is moved aside;
- HOLD files stop the run before the next GPU step;
- no live config, training run or bank is touched.

## Carried from generation 2's bank (#1517 review MINORs), fixed in this bank's operator scripts

1. **R1:** one check case perturbs only `param_group_lr[0]` and one only `param_group_table[0].lr`. Each is run as a
   mutant of the take-effect check.
2. The operator script is idempotent: a second run with the same inputs is a no-op. It never overwrites a `.before_*`
   backup, and refuses if one exists.
3. Every sourced or `cd` step ends in `|| exit 3`.
4. The script pins its own sha256 and refuses to run on a mismatch.

The operator scripts are not written yet. When they are, the staging harness gets a case and a mutant per item, and
the launch addendum quotes the outcomes.

## Commands

Filled in by the launch addendum, from the staged scripts and their sha256. No command here runs before that
addendum merges.
