# 2026-11-26: RL run 1 — does more data from the same actor improve the step? (pre-registration)

**Status: pre-registration, for review.** Nothing is launched. Launch needs:
- this record merged;
- the admission record merged (`2026-11-26-banked-collections-admission.md`);
- the pooled bank built and its harness passing in staging (below);
- a free GPU.

Only CPU work has been done for this record.

## The question, and the one variable

Generation 2 trained `gen1_baked.pt` (sha256 `8dd6ac8c…`) on one 8,000-game collection that the same net played.
- vs gen 1: +1.14 [−5.04, +7.32] Elo.
- vs the SL base: +3.85 [−2.46, +10.17].
- Licensed claim: "no second measured step".

**Hypothesis:** the step was data-limited. The same recipe, from the same parent, on more rows played by the same
actor, makes a net that beats the parent.

**The one variable is DATA VOLUME**: the number of distinct stored rows, all played by `gen1_baked.pt` at the
collector pin `0c795fb7a` with an identical collect recipe. Everything else is generation 2's B4/B5 recipe, unchanged:
- parent `gen1_baked.pt`;
- Zq flags, `--presentations 1` (exact-once), mirror 0.5, batch 512;
- LR 3e-6 flat, Aurora with fresh state;
- hold-out 0.05 by game, the same recal fit and bounds, the baked files only in matches.

**Evidence log first.** I grepped `docs/experiments/2026-11-21-rl-loop-evidence-log.md` for volume, rows, views,
exact-once, second pass and 16,000.
- No row measures a volume dose from one actor.
- The nearest are the pilot rows: 1,412 updates on 722,742 rows exact-once. They are one volume, not a dose.
- The amendment's "dose-response in games" was planned and never run.
- So this is a new measurement, not a rerun.

## Cells

| Cell | Rows (stored) | × gen 2 | Banks (all admitted or pending in the admission record) | Train updates ≈ | Status |
|---|---:|---:|---|---:|---|
| **k1** | 796,341 | 1.00 | gen2 | 1,478 (done) | **the existing `gen2_baked.pt`; no retrain**, its two banked matches are this cell's points |
| **kA** | 1,176,521 | 1.48 | gen2 + successor pilot + continuation | ≈ 2,183 | **registered now**: every bank is admissible on clean evidence |
| kB | 1,972,644 | 2.48 | kA + b01_r2 | ≈ 3,660 | **addendum**, only after the admission record admits b01_r2 |
| kC | 2,767,035 | 3.47 | kB + fresh_indep | ≈ 5,134 | **addendum**, only after both HOLD banks are admitted and kC passes the host-RAM rule (below) |

**Deviation from the brief, stated:**
- The brief asked for k = 1, 2 and ≈ 3–4. What is admissible on clean evidence now reaches only 1.48×.
- The coordinator's instruction is to write around what is admissible and let the HOLD banks join by addendum. So:
  - kA is the registered cell;
  - kB (2.48×) and kC (3.47×) are pre-specified here and are registered by an addendum. The addendum only names the
    admitted banks; recipe, seeds, rules and thresholds stay as written below.
- No cell at exactly 2× exists without subsampling a bank. Subsampling would add a selection rule, so it is not done.

Updates are `round(0.95 × rows / 512)`: exact-once over the training split. They are not set by hand. The trainer
computes them, and the launcher checks the realised count against this formula, derived from the cell's own rows,
never from another generation's realised number.

## The pooled bank (built in staging, CPU only)

The trainer at the pin presents ONE collection per generation. Exact-once with `--presentations 1` reads only
generation g's collect (`learn.phase_generations`). So each cell needs one pooled sealed collect.

**The tool:** `pool_collect.py`, in the record's staging directory, pure stdlib + numpy, no torch, modelled on
`simple_rl/retarget.py`'s full-run-dir output. It writes a source run dir, `rl-run1-volume/pool_<cell>/`:
- `gen_0000/` is copied from the gen2 bank (`gen1_baked.pt` and its manifest), so the arm's generation-0 check passes.
- `gen_0001/collect/shards/` holds every source shard **copied byte-identically** and renumbered in a fixed order:
  gen2, then succ_pilot, then succ_cont (then b01_r2, fresh_indep for the addenda). Each `.zarr` keeps its
  `.meta.npz` sidecar. Copies, not hard links, so nothing can write through to a bank.
- `games.jsonl` and `starts.txt` are concatenated in the same order.
- The manifest is gen2's with these replaced:
  - `counts` (summed);
  - `seeds.base` and `config.settings.seed` = the pool seed **2026112601**;
  - `config.settings.games` = the summed games;
  - `outputs` (recomputed with the pin's `sha256_tree` / `sha256_file`);
  - plus a `pool` block. Per source: run dir, manifest sha256, shard-tree / games / starts sha256, rows, games, the
    openings file and its sha.
- `inputs.train_openings` is the pilot train book (`1a98416d…`). Every opening id of every pooled bank is a line of
  it (admission census), and each successor stage batch is a subset of it, recorded in the `pool` block.
- It writes inside `.pool_<cell>.partial` and renames into place, and it refuses an existing output.

**Refusals**, each a non-zero exit before the rename:
- (P1) a source manifest whose recorded shard-tree, games or starts sha256 no longer recomputes equal;
- (P2) a source whose actor (`inputs.checkpoint.sha256`) is not `8dd6ac8c…`, or whose `git.sha` is not the pin, or
  is dirty;
- (P3) any `config.settings` difference from gen2's, outside the admission record's skip list;
- (P4) a game id present in two sources, or an opening position present in two sources;
- (P5) a source that is not in the cell's admitted list (the list is a constant in the tool, set per cell);
- (P6) a bank on a different filesystem from the output, or an output inside a bank directory.

**The harness** (`test/pool_harness.py`, CPU, the pin's own reader / reference / hold-out code) builds a pool from
small fixture sources (one re-sealed shard each of gen2, succ_pilot and succ_cont) and checks:
- reading the output through the pin's own `collect.gather_rows` gives, per source, arrays **byte-equal to that
  source's** (every key, every row), in the declared order;
- the union of meta game ids = the union of the ledgers' game ids with rows > 0;
- total rows = the sum of the sources;
- the pin's `reference.reference_collect` accepts the pool as a `--collect-from` source for an arm with the pool
  seed and summed games, and refuses it with any other seed or game count;
- `holdout_mask` marks the same games held out as in each source on its own (the hold-out is a hash of the game id);
- every pooled shard file has link count 1 (a copy, never a link into a bank);
- an existing output is refused unchanged, and each of P1–P6 refuses with nothing published (no output, no
  `.partial`).

Then the read-back checks run on the real pool before launch, reading every shard (`pool_harness.py real`). With
one shard per fixture source, the order of shards WITHIN a source is checked only by this real read-back.

**Mutants**, each killed or the harness is not trusted. Each one breaks one refusal or one check, the harness is run,
it must fail, and the mutant is reverted:
- each refusal disabled: P1, P2 (actor), P2 (pin / dirty), P3, P4 (game id), P4 (opening), P5, P6;
- a shard copied with one byte changed;
- the sources pooled in reverse order;
- a source's last shard dropped;
- `games.jsonl` concatenated without the last source;
- the pool seed left as gen2's;
- `outputs.shards` copied from gen2 instead of recomputed;
- hard links instead of copies.

**Staging status (2026-11-26, CPU).** Paths are under `~/chess-artifacts/operations/rl-run1-volume.staging/`:
- `prelaunch/pool_collect.py` [`eb05f7b5`];
- `test/pool_harness.py` [`632fe771`], whose unmutated control gives 20/20;
- `test/mutants_pool.json` [`798fb6a3`], run with `test/mutants.py` [`aaec0d68`]: **15 mutants, 15 killed**, each
  by a named FAIL line;
- results in `test/mut_pool.out` and `test/mut_pool/mutant_results.json`.

The real kA pool has not been built. It is built, and `pool_harness.py real` run on it, after this record merges.
The launch addendum quotes those read-backs.

## Training, recalibration and checks per cell

Per cell, one arm `rl-run1-volume/arm_<cell>`:
- the same flags as `gen2/gen2_flags.sh` (`COLLECT2` / `TRAIN` / `ZQ`), except `--seed 2026112601`, `--games` =
  the pool's games, and `--collect-from pool_<cell>`;
- sourced from a copy of that file pinned by sha256, as generation 2 did.

**Checks:**
- the optimizer take-effect check: the realised Aurora fields equal what the **flags** imply, at the pin's code
  path. Revision 6 of the generation-2 record has the exact rule;
- the realised updates equal the formula above;
- the train manifest's inputs chain to the pool manifest, and the pool manifest to each source manifest's sha256.

**No check compares a value to another generation's realised value.** This is a lesson of generation 2's revision 6.
For example, the arm's optimizer fields are not checked against arm_Zq2's.

**Recalibration and bounds (generation 2's B5, unchanged):**
- fit (T, b) with `scripts/fit_value_recal.py` at the pin: the cell's net, reference `gen1_baked.pt`, `--collect-dir`
  = the pool, hold-out 0.05;
- bake with `bake_recal.py` and `verify`;
- stop before any match unless T ∈ [0.4951, 1.4212] and b ∈ [−0.4542, +0.2271].

## Yardstick, screen and thresholds

**Matches** (the follow-up launcher, `arena_standard.py` matched sims, 64 v 64, plain seats, baked files, the
16-ply EVAL book):
- vs the parent `gen1_baked.pt`, seed **2026112283**;
- vs the SL base, seed **2026112284**.

These are generation 2's seeds, so every cell plays **the same opening sequence** as k1's banked matches. The cells
are paired on openings with each other and with k1.

**Screen, vs the parent only:**
- the first **800 games (400 opening pairs)** of the parent match;
- **rule: stop the cell iff the screen's UB95 < 0 (damage); otherwise extend the same match on the same paired
  openings to 6,400 games, then run the base match at 6,400.** The 6,400 includes the screen's games, played once.

What the screen can and cannot do:
- It is one look with a damage-only exit. It can only stop a cell that is losing, and it never licenses "improves".
- **A screen never decides a small difference.** At 800 games the SE is about 9 Elo, so a cell within ±15 Elo of the
  parent always continues.
- A cell stopped at the screen is reported as "screened out (damage at 800)", with its interval, and nothing more.

**Readings, each 6,400-game match (generation 2's rules):**
- damage iff UB95 < 0;
- adoptable iff LB95 > −10;
- **improves iff LB95 > 0**;
- otherwise unread.

The tie-break and seat handling are match 5's, and the result rows' `git_sha` and candidate/reference sha256 are read
back after smoke and real runs.

| Outcome | Licenses |
|---|---|
| a cell **improves** vs the parent | a measured RL step on banked data: generation 2's recipe repeats from `gen1_baked.pt` when given that cell's volume. The cell's net becomes the next generation's actor candidate under a new record |
| a cell improves vs the base, not vs the parent | a cumulative gain over the base only; not a step over the parent |
| no cell improves vs the parent | volume up to that cell's × does not rescue the step. Recorded as "unread" or "damage" per cell, never "no effect" |

**Dose-response is descriptive.** Cells are never ranked by point estimate. A difference between two cells is
claimed only from a head-to-head match registered by a later record.

Power, at SE ≈ 3.24 for 6,400 games: a +9.6 step (match 8's size) reads "improves" with probability 0.84, and a +5
step with 0.34.

## Confounds (named, not removed)

- **UHO is out.** Its opening distribution differs. It is the next single variable after volume, under its own
  record.
- **Volume and update count move together.** Exact-once means more rows = more updates at the same flat LR 3e-6, so
  "more distinct rows" and "a longer schedule" are not separated.
  - A separating control (k1's rows presented 1.48 times) would break the exact-once rule, so it is not run.
  - The reading is "more data, used once", which is the owner's rule.
- **The recalibration is fitted on more rows.** The pooled hold-out is ≈ 5% of the pool's games, about 59k rows for
  kA vs about 40k for k1. It is also drawn from several banks' games, not from one collection.
- **The batch plan differs.** The pool seed draws a different shuffled order from gen2's. Generation 2's own rerun
  showed that a retrain alone moves the net (`5d019e5e…` vs `1db00eca…`), so cell-to-cell noise includes this.
- **The successor openings are a different subset of the same book.** The stage batches excluded every id already
  booked, so kA's new games come from book lines gen2 did not use. The distribution is the same book. The selection
  is not gen2's `cycle` order.

## Memory, GPU time and recovery

**GPU memory.** Batch 512 and the model are unchanged, so the per-update GPU peak is generation 2's (smoke: 16.96 GiB
peak reserved, flat from update 2 to 3). Each cell still runs the 3-update memory smoke on its own pool before
training, with generation 2's abort fraction 0.85.

**Host RAM is what grows.** The loader holds all training rows: 20–23 GiB at ≈ 760k rows in the pilot's launch
record, about 2.9 GiB per 100k rows.

| Cell | Projected loader RAM | Rule |
|---|---|---|
| kA | ≈ 31–36 GiB | floor `MemAvailable` ≥ 50 GiB at launch |
| kB | ≈ 52–60 GiB | floor ≥ 70 GiB |
| kC | ≈ 73–84 GiB | **refused on this 98 GiB host by the stock loader** unless a measured CPU load of its pool (peak RSS) plus 10 GiB fits `MemAvailable`; otherwise kC waits for a streaming loader under its own record |

The floors are checked by the launcher before `.started`.

**GPU-h, from measured rates:**
- train: 0.3164 s/update (generation 2: 467.65 s / 1,478), giving kA ≈ 0.19 h, kB ≈ 0.32 h, kC ≈ 0.45 h;
- fit + bake + verify: ≈ 0.2 h;
- matches: 1.45 GPU-h per 6,400 games (match 8 and generation 2);
- per cell, through both matches: **≈ 3.3–3.6 GPU-h**. A cell screened out costs ≈ 0.6 h;
- kA alone: ≈ 3.3 GPU-h. All three cells: ≈ 10.3 GPU-h.

**Recovery.** Nothing here writes to a bank:
- the pools and arms are new directories under `rl-run1-volume/`;
- the sources are re-hashed by P1 before every build, and by the pin's lineage check at every stage;
- a failed stage is moved aside by name (generation 2's `move_aside.sh` pattern) and rerun from its inputs;
- no live config, training run or bank is touched;
- the only state that survives a failure is the arm directory, which is deleted only after the owner's word.

## Carried from generation 2's bank (#1517 review MINORs), fixed in this bank's operator scripts

1. **R1:** a check case that perturbs only `param_group_lr[0]`, and one that perturbs only
   `param_group_table[0].lr`. Each is run as a mutant of the take-effect check.
2. The operator script is idempotent. A second run with the same inputs is a no-op, and it never overwrites a
   `.before_*` backup: it refuses if one exists.
3. Every sourced or `cd` step ends in `|| exit 3` (generation 2's line 74 lacked one).
4. The script pins its own sha256 and refuses to run if the pinned value does not match.

The operator scripts are not written yet. When they are, the staging harness gets a case and a mutant for each of
the four, and the launch addendum quotes the outcomes.

## Commands

Filled in by the launch addendum, from the staged scripts and their sha256. No command here may run before that
addendum merges.
