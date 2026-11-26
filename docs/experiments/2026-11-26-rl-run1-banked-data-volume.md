# 2026-11-26: RL run 1 — more data, used once, from the same actor (pre-registration)

**Status: pre-registration, revision 2 (after the statistics review), for review.** Nothing is launched. Launch needs:
- this record merged;
- the admission record merged (`2026-11-26-banked-collections-admission.md`, #1523);
- the pooled bank built from it and read back in staging;
- the measured host-RAM load and the GPU memory smoke (below);
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
| **primary** | `gen1_baked.pt` trained once over the pooled kC bank (2,767,035 rows, 3.47×), or kB (1,972,644 rows, 2.48×) if kC fails the host-RAM rule | vs the parent, 12,800 games (800-game damage screen first); vs the SL base only if it improves vs the parent |
| **fresh k1** | the existing `gen2_baked.pt` (1×), no retrain | vs the parent, 12,800 games, same seed and size as the primary |
| checkpoints | the primary run's weights at 1,478 / 2,189 / 3,675 updates (and the final) | none registered; recal fits only (below) |

**Choosing kC or kB, by measurement only.**
- Before launch, the pooled kC bank is loaded on the CPU exactly as the trainer loads it (the pin's `load_window` →
  `prepare_targets` over the pool, no GPU), and its peak RSS is measured.
- kC is the primary iff that peak plus 12 GiB fits the host's `MemAvailable` at the time, with the other GPU/CPU jobs
  of that window running.
- Otherwise the primary is kB, measured the same way.
- The measured peak and the choice go in the launch addendum. No projection decides it.

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

**Staging status.**
- Revision 1's tool and harness (control 20/20, 15/15 mutants killed) are superseded.
- Revision 2's changes are written but **not yet run**, because the coordinator's CPU/GPU hold is up. They are:
  5-bank fixtures, the kC/kB cells, the admission binding, C18, C19 and three new mutants (18 in all).
- They run when the hold clears. Their control and mutant outcomes, with sha256 stamps, go in the launch addendum.
- After the admission record merges, the real pool is built and read back over every shard with
  `pool_harness.py real`.

## Training, checkpoints and recalibration

One arm, `rl-run1-volume/arm_<cell>`. Its flags come from a copy of generation 2's `gen2_flags.sh`, pinned by sha256,
with these exceptions:
- `--seed 2026112601`;
- `--games` = the pool's games;
- `--collect-from pool_<cell>`;
- **`--snapshot-updates 1478,2189,3675`** for kC, or `1478,2189` for kB. The pin writes `train/snapshots/step_<N>/`
  (a weights-only `model.pt`), and training is unchanged by it.

**Recalibration at every checkpoint and at the final.**
- Each is fitted with `scripts/fit_value_recal.py` at the pin: the checkpoint's net, reference `gen1_baked.pt`,
  `--collect-dir` = the pool, hold-out 0.05.
- All fits are banked.
- Only the final net is baked (`bake_recal.py`, then `verify`) and played.

**Bounds that scale with the updates.** Revision 1's fixed bounds were generation 2's, and they are likely unmeetable
at 3,675–5,151 updates.
- The bounds come from the pilot's one-generation drift, (T1, b1) = (0.7036, −0.2271) after U1 = 1,412 updates.
  Generation 2 reproduced it: T 0.6997, b −0.2467 after 1,478 updates.
- For a net after U updates, with s = U / 1,412:
  - ln T ∈ [2 s ln T1, −s ln T1] = [−0.703 s, +0.352 s];
  - b ∈ [2 s b1, −s b1] = [−0.454 s, +0.227 s].
- At s = 1 these are generation 2's bounds, unchanged.

Derivation:
- With a flat LR and no schedule, the drift per update does not grow with the update count.
- So a linear envelope (twice the measured rate in the measured direction, or once the rate in the opposite
  direction) bounds "more of the same training". A fit outside it means drift faster than twice linear, which is not
  more of the same.
- Generation 2's fit sits at about half its envelope (s = 1.05).
- A linear projection to kC (s = 3.65) gives T ≈ 0.29 and b ≈ −0.86. That is inside the kC bounds,
  T ∈ [0.077, 3.61] and b ∈ [−1.66, +0.83], but it would be an extreme calibration. That is why every checkpoint is
  fitted.

**A bound violation is a HOLD, never a dead gate.** If any checkpoint's or the final's fit falls outside its bound:
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

**Confidence intervals:** a cluster bootstrap by opening (both colours of a pair resampled together).

**No pairing claim is made.** The review measured an opening-level correlation of 0.026 between arms, so sharing the
seed buys nothing. The shared seed is kept only so that both arms play the same book draw.

**Screen (primary vs parent only):**
- the first 800 games (400 pairs);
- the cell stops iff the screen's UB95 < 0 (damage). Otherwise the same match continues to 12,800, and the screen's
  games are counted once;
- it is one look with a damage-only exit;
- **a screen never decides a small difference** (SE ≈ 9 Elo at 800 games) and never licenses "improves".

**Primary reading (vs parent, 12,800 games): improves iff LB95 > 0.**
- Damage iff UB95 < 0; adoptable iff LB95 > −10; otherwise unread.
- Power: SE ≈ 2.29 Elo, so the 80%-power minimum detectable effect is ≈ 6.3–6.4 Elo.

**Pre-registered slope contrast, one-sided.**
- Δ = (primary vs parent) − (fresh k1 vs parent), from the two 12,800-game matches.
- The arms are independent (no pairing assumed), so SE(Δ) ≈ 3.24.
- **"Volume raised the step" iff Δ − 1.645 SE(Δ) > 0.** MDE80 ≈ 8.1 Elo.
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
- It plays 12,800 games vs the parent on seed 2026112611 and is read by the same rules.
- If the control also improves, the step is not attributable to distinct rows.
- If it does not improve while the primary did, that supports distinct rows. Caveat: the update counts still differ
  (2,956 vs 5,151 or 3,675), because the ceiling forbids more passes.
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
- The measured CPU peak RSS of the pool that will run decides kC vs kB (above).
- The pilot's launch record implies about 2.9 GiB per 100k rows, which projects kC at ≈ 73–84 GiB on a 98 GiB host.
  That projection decides nothing; the measurement does.

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
- a failed stage is moved aside by name and rerun from its inputs;
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
