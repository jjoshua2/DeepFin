# 2026-11-26: RL run 1 — more data, used once, from the same actor (pre-registration)

**Status: pre-registration, revision 2 (after the statistics review), for review.** Nothing is launched. Launch needs:
- this record merged;
- the admission record merged (`2026-11-26-banked-collections-admission.md`, #1523);
- the pooled bank built from it and read back in staging;
- the real pool of the trained cell (kC; kB only after the fallback) rebuilt from the merged admission record, the launch-time host-RAM floor and the GPU memory smoke (below);
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
- After the admission record merges, the real pool of the trained cell (`pool_<cell>`, kC unless the fallback selects
  kB) is built bound to the merged record and read back over every
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
  = `gen1_baked.pt`, **`--collect-dir rl-run1-volume/pool_<cell>/gen_0001/collect/`** (`<cell>` = the trained cell:
  kC, or kB after the fallback; the sealed collection directory, not the pool's run dir), **`--holdout-frac 0.05`**,
  `--device cuda`.
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
- **Registered in "Conditional control addendum" at the end of this record** (mechanism, seed, checkpoints, gates).

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

All four are in `operator_volume.sh` and `opt_check.py`, with a harness case and a mutant each. The outcomes are in
the launch addendum below.

## Commands (launch addendum)

This addendum is the run's only command source. Nothing runs before it merges and the coordinator writes `OWNER_GO`.
Every Python stage runs at the pin `0c795fb7a` from its clean worktree `rl-run1-pilot-0c795fb7a`. The staged bank is
`~/chess-artifacts/operations/rl-run1-volume.staging/bank/`, and the operator copies it verbatim to the run's bank
directory, written `$V` below (`rl-run1-volume` beside the other `rl-run1-*` banks). The launcher, its library and
its checks are adapted from generation 2's revision 6 bank. What changed is listed per file below.

### What the bank does, step by step

One command (`operator_volume.sh`) verifies the bank and launches `launch_volume.sh`. The launcher runs the steps
below in order and stops at the first failure. Each step:
- writes `<step>.inputs` (the sha256 of the launcher, the checks library, the flags file, the prelaunch sums, the seed
  and every input file), then `<step>.started`, then `<step>.exit` ("<code> <time>");
- has a completion check that re-reads data and inputs, not file existence (`volume_checks.sh`, `prelaunch/stage_checks.py`).

On a relaunch a step is skipped only if all three hold: its exit is 0, its check passes, and its recorded inputs equal
the inputs now. A mismatch refuses (84). It never skips.

| Step | Device | Command (inside the launcher; `$COLLECT $TRAIN $ZQ` from `volume_flags.sh`) | Gate (refusal) |
|---|---|---|---|
| `pool` | CPU | `pool_collect.py kC $V/pool_kC`, bound to the merged admission record (sha256 `cdb1ab63`, the file on main); `test/pool_harness.py real $V/pool_kC kC`; `planned_updates.py $V/pool_kC plan.json --precheck -- $COLLECT $TRAIN $ZQ --snapshot-updates 1478,2189,3675`; `stage_checks.py pool` | read-back `RESULT … fail=0`; the PIN plan knowable and its precheck accepted; not built under a test override; admission sha256 = the pinned one; counts and the three digests = the staging measurement pool's; digests recomputed now; seed, games, actor and sources' manifests by sha256 (exit 1) |
| `smoke` | GPU | MemAvailable ≥ 86 GiB, else `HOLD_HOST_RAM` (89); `TRAIN_MEM_SMOKE_RUN=$V/pool_kC train_mem_smoke.py $V/smoke_mem 3 $G1 $COLLECT $TRAIN $ZQ` | PASS with rows = the plan's trainable rows, run = the pool, 3 updates; ABORT or any other exit → `HOLD_MEM` |
| `train` | GPU | MemAvailable ≥ 86 GiB (as above); `simple_rl_loop.py run --run-dir $V/arm_kC --collect-from $V/pool_kC --generations 1 $COLLECT $TRAIN $ZQ --snapshot-updates 1478,2189,3675 --pause-marker $V/STOP`; then `opt_check.py` (generation 2 revision 6's take-effect rule against generation 2's `arm_Zq2` fields) and `planned_updates.py` on the arm | optimizer check true; realised updates = the PIN's `planned_phase_updates` (5,151 expected, never a hand formula); snapshots exactly 1478, 2189, 3675; else `HOLD_TRAIN_CHECK` |
| `fits` | GPU | `fit_value_recal.py --net <step_N or final model.pt> --reference $G1 --collect-dir $V/pool_kC/gen_0001/collect/ --holdout-frac 0.05 --device cuda --out $V/recal_<N|final>.json` for N = 1478, 2189, 3675 and the final | each fit: T and b finite, `cli` = "T,b", the net's and the reference's sha256 now, the pool's collect dir, hold-out 0.05 |
| `bounds` | CPU | `recal_bounds.py $V $V/HOLD_RECAL 1478 2189 3675` | R1 (generation 2's box at every fit) and R2 (final vs 1,478, both ≤ 0.10); a miss → `HOLD_RECAL`, all fits banked |
| `bake` | CPU | `bake_recal.py bake` then `verify` → `$V/kC_baked.pt` | baked sha256 = json = verify; source = the final model; T, b = the final fit |
| `m_kC_vs_g1` | GPU | per seat, seed 2026112611: a 4-game smoke per seat, then seat A (6,400 games) to completion, then seat B (6,400) with the screen reader watching seat B's log | seat DONE (`seat_state.py`) both seats; result rows name the PIN and both nets' sha256; the PIN's `arena_seat_balanced_pool.py` → `pool.json`; screen damage → `STOP_SCREEN`, exit 80 |
| `m_k1_vs_g1` | GPU | the same for `gen2_baked.pt` vs the parent, seed 2026112611 | as above, no screen |
| `m_kC_vs_base` | GPU | only if `verdicts.py reading $V kC_vs_g1` prints `improves`; seed 2026112612 | as above |
| verdicts | CPU | `verdicts.py final $V` → `verdicts.json`, `summary.txt` (temp file + rename) | — |

Refusals before anything runs. None of these writes a marker:
- `OWNER_GO` missing (90);
- the guard missing (8) or not its pinned sha256 (93);
- `CUDA_VISIBLE_DEVICES` set to empty (94);
- the guard's own refusals (8): GPU memory outside the operator's idle ± 512 MiB, utilisation ≥ 10 % in 5 samples, a
  GPU job found by the process scan, or no `volume idle_mib=<MiB>` release line;
- the bank lock already held (9);
- any `HOLD_*`, `smoke/HOLD_*`, `STOP` or `STOP_SCREEN` file (81);
- a test-only variable in the environment (82);
- the checks library, flags file, prelaunch `SHA256SUMS` (and every file it lists), the operator gate, the match
  launcher, `pool_harness.py` or the PIN's `fit_value_recal.py` not its pinned sha256 (93);
- `PIN.txt` not the PIN (97); the PIN worktree not at the PIN, or dirty (96);
- a net not its registered sha256 (92);
- the resume prescan: 84, 86, 87, 88, or 95 for a pool or arm directory with no `.started`.

**Screen schedule** (the coordinator's decision):
- the primary match plays seat A to completion, then seat B;
- `screen.py` watches seat B's log and reads the screen once, as soon as every pair id < 200 has both halves in both
  seats. That is 800 games, chosen by pair id, never by finishing order;
- damage iff UB95 < 0. Damage writes `STOP_SCREEN` and SIGTERMs seat B's process group: the match stops (80), and
  `verdicts.json` records damage;
- otherwise seat B continues to 6,400, and the screen's games are counted once;
- an unreadable or inconsistent screen (headers not a seat swap, openings that differ, no interval) writes
  `HOLD_SCREEN` and stops seat B.

The screen's statistic is the PIN pool script's own `pool_looks`, imported, on the selected pairs.

**Decision:** the normal interval of `arena_seat_balanced_pool.py` over opening keys, per match. Readings:
- damage iff UB95 < 0;
- improves iff LB95 > 0;
- adoptable iff LB95 > x(−10 Elo);
- else unread.

Contrast: D − 1.645 SE > 0, strictly. The bootstrap (B = 20,000, seed 2026112613, the opening key as the unit) is
reported only and never decides.

**Watchdog and resume:**
- during smoke, train and fits, MemAvailable is sampled every 2 s. Below 6 GiB the watchdog appends to
  `watchdog.history`, writes `STOP` and SIGTERMs the step's process group; a second stop in the bank also writes
  `HOLD_WATCHDOG`;
- train is never moved aside. The relaunch reruns the same command, and the PIN resumes from the newest snapshot (1478,
  2189 or 3675). A kill before update 1,478 restarts the phase; a kill after it loses at most about 1,486 updates,
  ≈ 8 min;
- a match resumes per seat: a DONE seat is never replayed, and a PARTIAL seat stops until
  `move_aside.sh <step> --seat A|B`;
- any other failed or interrupted step stops until `move_aside.sh <step>`. It renames, never deletes.

### Staged files (sha256)

Every file below is in the staged bank. `SHA256SUMS` lists all of them; `prelaunch/SHA256SUMS` lists `prelaunch/`.
The launcher pins the guard, the checks library, the flags file, `prelaunch/SHA256SUMS`, `pool_harness.py`, the
operator gate, the match launcher and the PIN's `fit_value_recal.py`. `operator_volume.sh` pins its own sha256 and
verifies the whole bank against `SHA256SUMS`.

| File | sha256 | Origin |
|---|---|---|
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/SHA256SUMS` [`6c7e9545c466`] | `6c7e9545c46661679dcb6c6f0cdaec004e247bccd4867e88df7a981f72f1baab` | every file of the bank |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/gpu_guard.sh` [`29e85c52335d`] | `29e85c52335dd7f1e4f010b429f5d7e226156bc73c0d8863df4119523e7d6bad` | generation 2's guard v3, renamed to `VOLUME_*`, stage `volume`; `_SCAN_SHA` re-pinned to the v4z scan (follow-up below) |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/gpu_scan.sh` [`47c08152bd9e`] | `47c08152bd9e91dd1844018ab42dec2265a5dfb27e01924e206782e1fa369b84` | v4 plus only the full-width bank's v7 zombie hunk (follow-up below) |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/launch_volume.sh` [`7130c48bc46a`] | `7130c48bc46a6c482172ff195df8bc6dcb74f3122f7ab883ed317693c1281a3a` | generation 2's `launch_gen2.sh` v6: steps pool / smoke / train (snapshots, pause marker) / fits / bounds / bake / three matches / verdicts; RAM floor + watchdog; screen inside the primary; `bake()` runs `run env CUDA_VISIBLE_DEVICES= …` (follow-up below) |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/move_aside.sh` [`1a71108dffef`] | `1a71108dffef6b88e517319d7bb4f64ffb9cf4d07b51b677768faffb683ad9e3` | generation 2's v5: never train, never a whole match; `--seat A|B` |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/operator_volume.sh` [`f9e3f2b4133a`] | `f9e3f2b4133aad46ea973123f82e4b3fca103141c34e5d365e3ef9f83ea683ed` | the #1517 MINOR fixes 2-4 (idempotent, kept backups, `|| exit 3`, self sha256) |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/prelaunch/SHA256SUMS` [`f8e0abaebf1f`] | `f8e0abaebf1f170650363e04c4dc1d36395c6c07b9611062c397549c787d8483` | every file of `prelaunch/` |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/prelaunch/admission_record.md` [`cdb1ab63cdba`] | `cdb1ab63cdba4896c4c9677115484646c75fa5959d0784671c688e91f09f1b3d` | the merged admission record (#1523), byte-equal to main's |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/prelaunch/bake_recal.py` [`786bb537e0d1`] | `786bb537e0d1130ed2ae20bf81c7791239f4903095fd22a2f40840e3e8a06230` | generation 2's, unchanged |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/prelaunch/opt_check.py` [`4f8b306f72f8`] | `4f8b306f72f845dc7d5a4d687c41c74811d8f07c96bfd11dde4b2c7d991c55fc` | generation 2 revision 6's take-effect rule, moved into its own file |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/prelaunch/planned_updates.py` [`208995af153c`] | `208995af153ceacc3db7aa52e25cc914072394dd0fc089a3ba428bdc4fe901c4` | new: the PIN's `planned_phase_updates` / precheck / snapshot plan |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/prelaunch/pool_collect.py` [`66e2225c2006`] | `66e2225c2006878701dfa2e1b3dfcdb7922ce11b1ce3339aaa5d38b8a8209efe` | staging `a8e824d7` with `ADMISSION` / `ADMISSION_SHA256` set only |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/prelaunch/recal_bounds.py` [`c10c27e30b0b`] | `c10c27e30b0bf9fc5d957de19f8bb15429d1daaebd86cf71a80803bb53b631f7` | new: R1 (generation 2's box) + R2 |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/prelaunch/run_smoke.sh` [`8c3d545c7ba7`] | `8c3d545c7ba709ab0f6d7d9bfab5a5eda1ed83600bbe98e8ed3f6af284a8ad8b` | generation 2's runner, sources `volume_flags.sh` (standalone; the launcher runs the smoke inline) |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/prelaunch/screen.py` [`106f90becbe5`] | `106f90becbe5e881a7e1d529875aef0f681bd3caa6bf0a19f65050d47bc24989` | new: the damage-only screen reader |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/prelaunch/seat_state.py` [`a0f80af1286c`] | `a0f80af1286ca90d38eb6ddc77e84395292f0b8b0f73ad35654d579c842ceef0` | generation 2's, unchanged |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/prelaunch/stage_checks.py` [`a5ba3b131343`] | `a5ba3b131343e025a88794591bd3e8dac0751f05ab390ec961ea03afcc7b21e7` | generation 2's, plus `pool` / pool-chain `arm` / smoke / fits / bake / match modes |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/prelaunch/train_mem_smoke.py` [`2ce38774e589`] | `2ce38774e589c124266381b138ef9f7b689c2b9e2aace05e689c38d1a06d5c54` | generation 2's, unchanged |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/prelaunch/verdicts.py` [`f487d849300d`] | `f487d849300d89e2795d24a942f25fcfb2589c96a3e9e67e11792f51507c8118` | new: readings, contrast, bootstrap check |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/test/mutants.py` [`aaec0d680645`] | `aaec0d68064514be139c356c0a832d6d11f6dd281fee62510d7825d54263a6b8` | staging, unchanged |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/test/pool_harness.py` [`d3cc705d2757`] | `d3cc705d275707462bb8340e2e87ff32014317e1b76dfc9b1c1a5f85eefdd39e` | staging, unchanged (its `real` mode is the pool step's read-back) |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/volume_checks.sh` [`0676d7f46465`] | `0676d7f4646514f16392c6d9f422d4ac7f89623273c24fd9ca39f69ba7ad3bc1` | generation 2's `gen2_checks.sh`: paths, completion checks, inputs record, per-seat player with the screen sidecar |
| `~/chess-artifacts/operations/rl-run1-volume.staging/bank/volume_flags.sh` [`a837b3db627a`] | `a837b3db627af066bbfbce70c2c280b34abe4e742c0a1f7c0f6fb2af04ae18d9` | generation 2's `gen2_flags.sh` with `--seed 2026112601`, `--games 27840` (TRAIN, ZQ byte-identical) |

Read-only inputs, pinned by the launcher or the checks library:
- `gen1_baked.pt` `8dd6ac8c`, `gen2_baked.pt` `5881d2af` and `sl_base.pt` `a39f900c` (92);
- the operator gate `operator_gpu_check.sh` `2c4efb70` (93);
- `rl-run1-followup/run_followup_match.sh` `7c8fd065` (93);
- the PIN's `scripts/fit_value_recal.py` `37e27420` (93);
- the PIN's `scripts/arena_seat_balanced_pool.py` `cfd72395`.

### Follow-up: zombie-tolerant scan, `bake()` env (after #1527)

Two changes to the staged bank after #1527's review. The table above carries the new sha256s. The prior files are kept
outside the bank, so `SHA256SUMS` still lists exactly the bank:
`~/chess-artifacts/operations/rl-run1-volume.staging/prev/gpu_scan.sh.v4` [`949408e727e7`],
`.../gpu_guard.sh.v1` [`aaa51afcad66`], `.../launch_volume.sh.v1` [`f6da0bea755f`], `.../SHA256SUMS.v1`
[`424550da6c56`].

- **`gpu_scan.sh` v4z.** The full-width bank's scan v7 added a zombie rule: its v5+ scan refused on a non-dumpable
  zombie (`cannot read the fd links of pid … Permission denied`), and the host carries about 80 zombies left by a
  parent that does not reap them. This bank's scan is v4 plus exactly v7's hunk, taken from the diff of
  `~/chess-artifacts/operations/rl-run1-root-full-width/gpu_scan.sh.v6` [`b8197e6e9ca6`] against
  `.../gpu_scan.sh` [`d1aee351d16e`]: a `_gs_read_state` reader; State Z gives a NOTE and a skip; a process whose
  status and `/proc` entry are both gone is skipped; anything else unreadable still refuses. Nothing else from v5–v7
  is ported; the header line is reworded. `gpu_guard.sh`'s `_SCAN_SHA`, the launcher's `GUARD_SHA` and `SHA256SUMS`
  are re-pinned. `prelaunch/SHA256SUMS` and `operator_volume.sh` are unchanged; the operator verifies the bank
  against `SHA256SUMS`.
- **v4 was not the scan that refused.** v4 skips a process with an empty cmdline before the fd clause, and a zombie's
  cmdline is empty, so v4 already passed the host's zombies (argv and device clauses stubbed: rc 0). The hunk covers
  a process that becomes a zombie between its cmdline read and its fd listing. v4's empty-argv skip is the gap v5
  closed (a GPU process that cleared its argv escapes the device clause). It is not ported here; that is a separate
  decision.
- **`bake()`** runs both `bake_recal.py` calls as `run env CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY …`, like every
  other step. The old `VAR= run …` prefix on a shell function also reached the child, so this is consistency, not a
  behaviour change.

**Tests.** A direct scan-function test, run while a live GPU job held the host, so it starts only unlisted
`python3 -c` processes. The argv and device clauses are stubbed so the fd-readability clause decides:
`~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/zombie/zombie_test.sh` [`74853a74d355`], output
`.../out_v4z.txt` [`05bbbd53a3fe`], 6/6:

- Z1: a real non-dumpable process that turns zombie between its cmdline read and its fd listing is NOTEd and skipped.
- Z2: a live non-dumpable process still refuses beside it.
- Z3: a process that vanished after a denied listing is skipped.
- Z4: a live process with denied fd links still refuses.
- G1/G2: the guard's pin accepts this scan and refuses a tampered copy (the harness's tampered-scan case, run directly).

Mutants, each failing on its own target pid:

- the old v4 fails Z1, refusing on the zombie: `.../mut_gpu_scan.sh.v4.txt` [`bdee193e9fb2`];
- the Z branch disabled fails Z1: `.../mut_M1_no_zombie_branch.sh.txt` [`a21bb500fceb`];
- the vanished skip removed fails Z3: `.../mut_M2_no_vanished_skip.sh.txt` [`6f728aa73bd3`];
- every process read as a zombie fails Z2 and Z4: `.../mut_M3_everything_zombie.sh.txt` [`8cba93a1dc96`].

The harness's other scan case (a listed-name job refused) spawns a listed name, so it was not run while the live job
was up. The scan's argv clause is unchanged.

**Live host, read-only.**

- `.../live_unstubbed.txt` [`824cf167a014`]: the full scan refuses (rc 8) on the running full-width Stage 0 job,
  which is listed. That is correct while the job runs.
- `.../live_fdclause.txt` [`ae5d2c12657e`]: with the argv and device clauses stubbed, it returns 0 on 3 of 3 runs
  with 78 zombies present.

**Found, not changed (fails closed).** A live process that closes an fd between `find` listing it and `stat -L`
reading it makes the scan refuse (`cannot read the fd links of pid …`). The "exited mid-scan" exemption needs every
output line to be ENOENT, and the lines already listed are not. The code is the same in v4 and v7. The guard
re-checks every 5 s up to `VOLUME_REGUARD_WAIT_S` and only the last attempt decides, so this costs a retry, not a
wrong launch. These tests hit it repeatedly on Stage 0's busiest worker.

### Proofs on real fixtures (CPU; no GPU; torch-importing runs inside a private mount namespace)

Paths below are under `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/`.

| What | Fixture | Outcome |
|---|---|---|
| Guard / launcher harness: every refusal in the addendum's table, a full correct run, per-seat resume, the screen inside the match, the watchdog, move-aside and the operator script. Run under a fake HOME in a private user + PID + mount namespace with stub `nvidia-smi` and fake GPU stand-ins; the real `stage_checks.py`, `seat_state.py`, `screen.py`, `recal_bounds.py`, `opt_check.py`, `verdicts.py` and the PIN pool script | generation 2's real `arm_Zq2` metrics and manifest as the trained arm | `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/guard/out/unmutated.log` [`4e21316d6ec4`] (RESULT pass=202 fail=0) |
| Mutant series over the launcher, library, guard, move-aside, operator and every gate script; each mutant disables or shifts one refusal or comparison | the harnesses in this table | **148 of 152 killed (144 assertion, 4 error)**. The 4 survivors are equivalent, with the reason stated. Every control passed: `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/mutants/FINAL.txt` [`dbde83fc5ce9`]; runs `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/mutants/out_v1/SUMMARY.txt` [`e43027fbfbed`], `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/mutants/out_v2/SUMMARY.txt` [`a7c1025fa5e3`], `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/mutants/out_v3/SUMMARY.txt` [`3c8598642e96`] |
| Gates on generation 2's banked outputs: match, bake, fits, R1/R2, seat state, smoke rows. Each passes on the real file, and one wrong copy of each fails | generation 2's bank | `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/realfix/gates_real.log` [`ba848613741d`] (RESULT pass=14 fail=0) |
| Pool gate and the PIN's plan on the real pool, built by the pinned tool bound to the merged record | the five admitted banks | all 14 pool checks true; read-back 6/6; planned 5,151, trainable rows 2,636,964, precheck accepted; shard / games / starts digests `5cf545d7` / `774a61b6` / `a7c57baf` = the staging pool's: `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/realfix/realpool_build.out` [`2734b962272e`] (RESULT pass=6 fail=0), `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/realfix/realpool_digests.json` [`fa6d38918b70`] |
| PIN plan on the staging pool (5,151 updates, 2,636,964 trainable rows, precheck accepted); pilot plan rows = generation 2's banked smoke rows | the staging pool, the pilot collect | `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/realfix/plan_proofs.log` [`53ecae64e906`] |
| Take-effect check (generation 2 revision 6's rule, plus R1 MINOR fix 1: each of group 0's three lr places perturbed alone) | generation 2's `arm_Zq2` and its first attempt; `gen1_baked.pt`'s saved groups (torch.load) | `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/optcheck/out/unmutated.log` [`2cfc1945fbe0`] (RESULT pass=30 fail=0) |
| Screen reader: reproduces generation 2's `g2_vs_g1` and `g2_vs_base` pool estimate, SE, n and CI exactly; reads pair ids < 200, not the first finished games; partial line in flight; not readable; damage; HOLD | generation 2's match logs | `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/screen/out/unmutated.log` [`95a7125de463`] (RESULT pass=11 fail=0) |
| Verdicts: every reading and contrast branch (strict bounds, 1.645 one-sided, exact-zero contrast), screen-stop; d(o) rebuilt from the real logs reproduces pool.json; bootstrap deterministic | fabricated pools; generation 2's logs | `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/verdicts/out/unmutated.log` [`e641ae3a9d27`] (RESULT pass=34 fail=0) |
| CPU kill-and-relaunch resume of the training step at the PIN (tiny model; aurora, hparams from the init checkpoint, zclip on, the TRAIN/ZQ flag shape, the snapshot list, the pause marker): SIGTERM to the group inside the zclip warm-up, then the same argv relaunched | the PIN's own test fixtures | `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/resume/out/none.log` [`a4ff06f4fbc7`] (RESULT pass=23 fail=0); mutants `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/resume/out/zclip_not_restored.log` [`90acc8efee55`] (RESULT pass=7 fail=1), `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/resume/out/zclip_buffer_short.log` [`edca671641b0`] (RESULT pass=22 fail=1), `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/resume/out/resume_state_dropped.log` [`07e3be05f558`] (RESULT pass=21 fail=2) |
| The pool tool's staging suite after `ADMISSION_SHA256` was set: the control (24/24) and the two admission mutants | the 5-bank fixtures | control pass, 2 of 2 killed: `~/chess-artifacts/operations/rl-run1-volume.staging/test_launcher/pool_tool/mut_pool_v3_admission.out` [`854845080034`] |

The real pool was built in a scratch directory with the pool step's exact commands, never in `$V`, and then deleted.
The launcher builds its own pool. The resume test found that weights alone cannot see a lost zclip sample: the
`zclip_buffer_short` mutant leaves the final weights bit-identical and is caught only by the zclip-state comparison.
At the PIN, an exact resume refuses outright if the zclip state cannot be restored (`zclip_not_restored`).

Defects the harness found in this bank, both fixed before the shas above:
- `screen.py` was launched without `PYTHONPATH=.`, so every primary match would have stopped on `HOLD_SCREEN`;
- the watchdog's `sleep` inherited the bank lock's fd 9, which held the lock for up to 2 s after a launch ended.

### Operator runbook

1. **Copy and verify** (the run's bank must not exist yet):
   ```
   cp -a ~/chess-artifacts/operations/rl-run1-volume.staging/bank ~/chess-artifacts/operations/rl-run1-volume
   cd ~/chess-artifacts/operations/rl-run1-volume && sha256sum -c --quiet SHA256SUMS && echo BANK_OK
   ```
2. **Authorise.** The coordinator, never the operator script, writes one line, `<who> <ISO time> <message>`, to
   `$V/OWNER_GO`.
3. **Pre-flight** (verifies only, launches nothing; exit 0 expected):
   ```
   ~/chess-artifacts/operations/rl-run1-volume/operator_volume.sh --no-launch
   grep MemAvailable /proc/meminfo     # >= 90177536 kB (86 GiB) before the smoke and the training step
   ```
   Stop every other GPU job. Quiet the host: the training step needs MemAvailable ≥ 86 GiB, and the watchdog stops it
   below 6 GiB.
4. **Launch** (the one command):
   ```
   ~/chess-artifacts/operations/rl-run1-volume/operator_volume.sh
   ```
   It does the following:
   - verifies its own sha256, the bank against `SHA256SUMS` and `OWNER_GO`;
   - is a no-op (exit 0) if a launch holds the lock or the run already finished;
   - measures the idle `memory.used` and runs the pinned operator gate;
   - writes the release line `volume idle_mib=<MiB> <ISO time>` to `$V/GPU_RELEASED_BY_OPERATOR`. A line with the same
     idle value is kept; otherwise the old file is kept as `.before_<UTC stamp>`, and an existing kept file refuses (3);
   - starts `launch_volume.sh` under `setsid nohup` (with `VOLUME_TEST_MEMINFO` unset), appending to `$V/run.out`.

   Running it again is safe.
5. **Watch:** `$V/resume.log` (every run, skip and seat), `$V/run.out`, `$V/watchdog.history`, `$V/guard.log`.
6. **Stop and resume:**
   - to stop: `kill -TERM <launch pid>` (in `resume.log`). The launcher kills its tracked process groups and writes
     `$V/exit` 143;
   - to resume, rerun step 4. A finished step is skipped, train resumes from its newest snapshot and matches resume
     per seat.

   After a watchdog stop, quiet the host and move `$V/STOP` aside (never delete it), then rerun step 4. `move_aside.sh
   <step>` is only for a failed or interrupted non-training, non-match step, and `--seat A|B` only for one PARTIAL
   seat.
7. **Holds** are the owner's: read the file and change nothing. They are `HOLD_HOST_RAM`, `HOLD_MEM`,
   `HOLD_TRAIN_CHECK`, `HOLD_RECAL`, `HOLD_SCREEN`, `HOLD_WATCHDOG`, `HOLD_READBACK_<seat>` and `STOP_SCREEN` (a
   registered damage stop: the reading is recorded in `verdicts.json`).
8. **The end:**
   - `$V/exit` reads `0 <time>`;
   - `$V/summary.txt` gives the licensed claim, each match's reading, excess, SE and Elo, the contrast and the screen;
   - `$V/verdicts.json` holds the decision record, including the bootstrap check and the base match or why it was not
     run.

   The verdict goes in this record in the same session, by the pre-committed rule.

### Expected durations (measured rates; the GPU rates are generation 2's)

| Step | Duration | Basis |
|---|---:|---|
| pool | ≈ 15 min (measured: build 140 s, read-back 683–788 s, plan + check) | the real build in staging, below |
| smoke | ≈ 10 min | the kC load (540 s, `rss_kC.json`) + 3 updates |
| train | ≈ 36 min | 540 s load + 5,151 × 0.3164 s |
| fits + bounds + bake | ≈ 25 min | the record's ≈ 0.4 GPU-h |
| `m_kC_vs_g1` | ≈ 2.9 h | 1.45 h per 6,400-game seat; the screen is read ≈ 1.55 h in (seat A + ≈ 7 % of seat B: generation 2's g2_vs_g1 completed its pair ids < 200 after 458 seat-B games) |
| `m_k1_vs_g1` | ≈ 2.9 h | as above |
| **unconditional total** | **≈ 7.5 h** | |
| `m_kC_vs_base` (if improves) | + 2.9 h | |

A screen stop ends the run ≈ 3.0 h after launch.

### What is banked at each step (all under `$V`)

| Step | Banked |
|---|---|
| every step | `<step>.inputs`, `.started`, `.exit`; `resume.log`, `exit`, `exit.history` |
| pool | `pool_kC/` (sealed run dir, copied shards), `pool.log`, `pool_readback.log`, `plan.json`, `pool_check.json` |
| smoke | `smoke_mem.json`, `smoke_mem.jsonl` (per-update peaks), `smoke_mem.log` |
| train | `arm_kC/` (reference collect, `train/` with `model.pt`, `metrics.json`, `manifest.json`, `snapshots/step_{1478,2189,3675}/`), `train.log`, `train.opt.json`, `plan_arm.json`, `train.check.json` |
| fits, bounds | `recal_{1478,2189,3675,final}.json`, `fits.log`, `recal_bounds.json` |
| bake | `kC_baked.pt`, `kC_baked.pt.json`, `kC_baked.pt.verify.json`, `bake.log` |
| matches | `arenas/<label>/` and `smoke/arenas/<label>/`: per-seat `.games.jsonl`, `.pgn`, `.log`, `.exit`, `results.jsonl`, `pool.json`, `pool.log`; `screen.json` (+ `STOP_SCREEN` on damage) |
| verdicts | `verdicts.json`, `summary.txt` |

### Deviations from the launcher brief (conservative readings)

1. **The admission record the pool tool reads is a bank copy.** The copy is `prelaunch/admission_record.md`, not "a
   main checkout's copy". It is byte-equal to the merged file on main (sha256 `cdb1ab63`, checked against
   `origin/main` after #1523 merged), and the tool pins that sha256. With the copy, the bank is self-contained and
   verified by `SHA256SUMS`.
2. **The training command adds `--pause-marker $V/STOP`.** The brief's command does not have it. At the PIN the
   marker blocks stage starts only. A training phase already running is stopped by SIGTERM (the watchdog does this, or
   the operator).
3. **The smoke runs inline in the launcher.** `prelaunch/run_smoke.sh` is staged for a standalone smoke. It refuses
   while a launch holds the bank lock, so the launcher does not call it.
4. **Two refusals inside `step()` are kept as defence in depth.** They are "done but inputs differ" and "open step
   under other inputs". The prescan refuses the same conditions earlier, without writing a marker. The harness
   asserts the prescan's refusal, because the exit marker is left unchanged.
5. **The pool tool's staging suite was rerun after `ADMISSION_SHA256` was set,** reduced at the coordinator's
   request:
   - the control (24/24) and the two admission mutants (M17, M18) were rerun;
   - the other 16 mutants were not rerun. They test code that did not change, and the 18/18 series of revision 2
     stands.
6. **The per-refusal mutant series was run in full,** although the coordinator's re-scope arrived while it was
   running. It already covered the number-comparing gates.
7. **The bare `./scripts/lint.sh` exits 1 in the PR worktree,** on 6 errors and 1 warning in files this PR does not
   touch. The worktree uses the live venv. The errors are unresolved imports of `zstandard` and `msgspec`, which that
   venv lacks, and one torch-stub argument type. Ruff passes. This PR changes Markdown only, and CI's lint is its gate.

## Conditional control addendum (registered before any control launch)

**Status: registration, for non-author review.** Nothing of the control is launched. It launches only after this
addendum merges, and after its launcher PR is reviewed and merged.

**The trigger fired.** The primary read "improves", so the registered control runs. The readings are from the finished
bank's verdicts, `~/chess-artifacts/operations/rl-run1-volume/verdicts.json` [`fab91794d932`]:
- kC vs the parent: +5.08 Elo [+0.48, +9.67] (score excess 0.00730, SE 0.00337, 3,200 opening clusters);
- fresh k1 vs the parent: −5.13 [−9.47, −0.79];
- kC vs the SL base: +8.52 [+4.00, +13.05];
- the slope contrast passes (Δ̂ 0.01469, lower one-sided bound 0.00706 > 0).

### Decisions (the coordinator's, recorded as registered)

| Gap | Decision |
|---|---|
| G1 mechanism | **(a)**: one phase, one shuffled exact-once pass over the gen2 collection **copied twice**, with one continuous optimizer state and 2,956 planned updates from the pin's own plan. Both copies keep their game id, so the pin's game-hash hold-out puts them on the same side. The fallback was (b-continue): two sequential passes, a fresh permutation, `optimizer_state=continue`. (b-fresh) and sampling with replacement were excluded. **(a) was feasible (below), so the fallback is not used.** |
| G2 training seed | 2026112602 (the pool seed and `--seed`; the batch plan is the pin's `seed_for(2026112602, "train", 1)`) |
| G3 mirror | an independent 0.5 coin per presentation (the trainer's default, as in the primary). Two presentations are two views; mirrors are not extra views |
| G4 checkpoints and recal | snapshots at 1,478 and 2,956. Fits at 1,478, 2,956 and the final (the final is the 2,956-update net). R1 = this record's fixed box at every fit. R2: \|ln T_2956 − ln T_1478\| ≤ 0.10 and \|b_2956 − b_1478\| ≤ 0.10. A miss writes `HOLD_RECAL`. `--collect-dir` = gen2's sealed collection, `rl-run1-gen2/gen2/collect/gen_0001/collect/`, with `--holdout-frac 0.05` by game |
| G5 screen | none: all 12,800 games on seed 2026112611, resumable per seat as in the primary |

### Mechanism (a): what was checked, and the observation that decided it

The branch point was whether the pin accepts a collection in which every game id appears twice. The pin is fixed, so
everything had to work through the pool tool alone. Every check below ran on CPU at the pin `0c795fb7a`, inside a
private mount namespace with `/dev/dxg` covered and `CUDA_VISIBLE_DEVICES` empty.

**The tool.** `~/chess-artifacts/operations/rl-run1-volume-control.staging/feas/tools/pool_collect.py` [`d2d55334cf59`] is the volume bank's `pool_collect.py` (sha256
`66e2225c2006…`) with only these changes:
- one cell, `c2x` = (gen2, gen2): a bank listed twice is copied twice, in order;
- P4 allows a repeated game id or opening only between copies of the same bank. Between different banks it still
  refuses, and so does a repeat inside one copy;
- `POOL_SEED` = 2026112602;
- the pool block records the copies.

**On the real gen2 bank** (read only; `~/chess-artifacts/operations/rl-run1-gen2/gen2/collect/gen_0001/collect/manifest.json` [`778b7ecd7f4d`]):
- The tool built the doubled pool: 16,000 games, 1,592,682 rows (2 × 796,341), 84 shards. Log
  `~/chess-artifacts/operations/rl-run1-volume-control.staging/feas/pool.log` [`49569cf56fe6`]; manifest `~/chess-artifacts/operations/rl-run1-volume-control.staging/feas/pool_c2x/gen_0001/collect/manifest.json` [`30ad968c9762`].
- **The pin's own plan and acceptance**, through the volume bank's `planned_updates.py --precheck` with the control's
  flags (`~/chess-artifacts/operations/rl-run1-volume-control.staging/feas/plan.json` [`56183379a1ce`]):
  - `precheck_source` accepted the pool;
  - `planned_phase_updates` = **2,956**;
  - `phase_train_rows` = **1,513,422** (2 × gen2's 756,711);
  - the snapshot plan [1478, 2956] fits.
- **The pin's `reference_collect` sealed a reference to the pool, and lineage `verify_stage` accepted it.** The
  loader's shard list through the reference is the pool's 84 shards (`~/chess-artifacts/operations/rl-run1-volume-control.staging/feas/ref_check.json` [`1605ce080605`],
  script `~/chess-artifacts/operations/rl-run1-volume-control.staging/feas/tools/ref_check.py` [`03d2ae221523`]).
- **Exactly twice, counted from the pin's plan** (`~/chess-artifacts/operations/rl-run1-volume-control.staging/feas/tools/double_check.py` [`1eb6606d784c`] `plan`; result
  `~/chess-artifacts/operations/rl-run1-volume-control.staging/feas/plan_twice.json` [`eda3a8408137`]). Row ids are rebuilt as `learn.load_window` builds them, and the batch
  plan is the pin's `learn.plan_batches` with the pin's seed.
  - The two halves' meta sidecars are equal, in order.
  - **No game id is on both sides of the hold-out.**
  - The distinct position keys (game id, ply index) are exactly half the rows.
  - The plan is a permutation of the trainable rows, in 2,956 batches = the planned updates.
  - **Every one of the 756,711 trainable positions is in the plan exactly twice** (histogram {2: 756,711}). The 39,630
    held-out positions are in it never.
- **That rebuild is the pin's real order, not a model of it.** `double_check.py validate` on generation 2's finished
  arm: the rebuilt plan's rows, in order, equal its `presented.npz`, and the seed equals its recorded `seeds.train`
  (`~/chess-artifacts/operations/rl-run1-volume-control.staging/feas/validate_gen2.json` [`1f5836b55556`]).

**End to end through the real trainer and its ledger** (CPU, a tiny model, the pin's own test fixtures, as the volume
bank's resume test; `~/chess-artifacts/operations/rl-run1-volume-control.staging/feas/tools/ctl_e2e_test.py` [`fcdb483eb6b2`], log `~/chess-artifacts/operations/rl-run1-volume-control.staging/feas/e2e_unmutated.log`
[`8bed876d783b`], 15/15):
- the real tool built `c2x` from a tiny gen2 (its `ACTOR` constant set to the tiny parent);
- one `simple_rl_loop.py run --collect-from` phase exited 0, with realised updates = the pin's planned updates;
- the trainer's own `presented.npz` (the consumer's record, not the plan) shows every trainable position exactly twice;
- the rebuilt plan equals the realised order;
- **mutant:** the same phase on the single-copy source fails the ledger check.
- The tiny run held out no game. The hold-out side is shown by the real-pool plan check above (39,630 held positions,
  0 presentations, no shared game id).

**Deciding observation:** the pin keys its presentation ledger by (generation, row index), never by game id
(`learn.load_window`, `presentation_counts`). It keys the hold-out by game id alone (`holdout_mask`). And
`reference_collect` / `precheck_source` check sealed sha256s and settings, not id uniqueness. So duplicated ids pass
every pin check, and the copies stay on one side of the hold-out. This was observed above, not assumed.

### The control, exactly

- **Pool:** `pool_c2x`, built by the launcher's pool step in a new bank, `rl-run1-volume-control/` beside the
  finished one (it does not exist yet), from the gen2 bank. It is read back over every shard, and the pool step also passes `double_check.py plan`. A miss
  is a refusal, not a HOLD.
- **Train:** `gen1_baked.pt` (parent), one `simple_rl_loop.py run --collect-from pool_c2x --generations 1` phase.
  - Flags: the volume flags file with only `--seed 2026112602` and `--games 16000` changed (TRAIN and ZQ
    byte-identical: `--presentations 1`, mirror 0.5, batch 512, LR 3e-6 flat, no warm-up, Aurora, fresh optimizer
    state, hold-out 0.05).
  - `--snapshot-updates 1478,2956`.
  - Gates after training, each a `HOLD_TRAIN_CHECK` on a miss: the optimizer take-effect rule; realised updates = the
    pin's planned updates (2,956); **`double_check.py ledger` on the trainer's `presented.npz`** (every trainable
    position exactly twice, held-out never).
- **Host RAM:** the record's rule. The launch-time floor is the loaded-array footprint measured on the staged `c2x`
  pool, plus 12 GiB margin, plus 2 GiB for the trainer. The watchdog and the snapshot resume are unchanged. The launcher
  PR quotes the measured number.
- **GPU memory:** the 3-update smoke on the pool, generation 2's abort fraction 0.85.
- **Fits and bounds:** as G4. **Bake:** the final net only.
- **Match:** `ctl_vs_g1`, the control's baked net vs `gen1_baked.pt`, 12,800 games (6,400 per seat), seed
  2026112611, matched sims 64 v 64, the same book, no screen, per-seat resume.

### Readings it licenses (the estimator of this record, unchanged)

- **Control vs the parent:** improves iff LB95 > 0; damage iff UB95 < 0; adoptable iff LB95 > −10; otherwise unread.
- **Attribution contrast, registered in the record and computed here:** Δ̂_PC = d̄_kC − d̄_C on the score scale, with
  SE = √(SE_kC² + SE_C²). d̄_kC and SE_kC are the banked kC-vs-parent result
  (`~/chess-artifacts/operations/rl-run1-volume/arenas/kC_vs_g1/pool.json` [`f60ec7ca7d72`]). **kC is not replayed.**
  - **"Distinct rows beat repeated rows" iff Δ̂_PC − 1.645·SE > 0, strictly.** Otherwise attribution is unread.
  - A split of significance (one arm improves, the other does not) is not a difference and licenses nothing about
    attribution.
- **Caveats, stated:** the update counts still differ (2,956 vs 5,151); the batch plan differs (seed); the control's
  rows are a subset of kC's pool (gen2's rows).
- The verdict is recorded in the same session as the readout, by these rules.

**GPU time** (measured rates, as above):

| Item | GPU-h |
|---|---:|
| smoke + train (2,956 × 0.3164 s) | ≈ 0.3 |
| 3 fits + bake + verify | ≈ 0.3 |
| `ctl_vs_g1`, 12,800 games | ≈ 2.9 |
| **total** | **≈ 3.5** |

**Recovery:** everything is written under the new bank. The finished volume bank and the gen2 bank are read only,
pinned by sha256. Training resumes in place from its newest snapshot, and the match resumes per seat, as in the
primary.
