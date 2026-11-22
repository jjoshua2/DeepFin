# 2026-11-21: RL run 1 — sibling-ordering value screen (`--w-sibling`) (pre-registration, revision 7)

**Status: registered design (revision 7), not launched.** Revision 6's first GPU step failed (step 1b, below), and
no cell has ever trained. **NO stage of this screen launches — no smoke, probe, training, Stage A or match — before
the launch addendum is reviewed and merged on main.** That addendum carries the built scripts' sha256, the sealed
strengths, and the memory smoke's thresholds as coded; this record alone authorises nothing. Revision 7 redesigns the cells around the owner's ask and the measurements made since.
It launches in two parts:
- **Part 1, on the pilot's frozen collection, after** this record is reviewed and merged, after generation 2
  ([gen-2 record](2026-11-22-rl-run1-generation2-zq-recal.md)) has released the GPU, and after the launch
  preconditions below are banked and reviewed.
- **Part 2, on a new full-width collection, after** Part 1's Stage A, after PR #1511 (forward chunking) is merged
  and its one GPU rerun has passed, and after a Part 2 addendum to this record pins the collection command, its
  cost and the root-list sidecar (below).

The screen plays no match against the SL base and changes nothing live. Revision 7's outputs go to a NEW bank,
`~/chess-artifacts/operations/rl-run1-sibling-r7/` (`SB7`). Revision 6's bank, `~/chess-artifacts/operations/
rl-run1-sibling/` (`SB`), is frozen and only read. Revision 6's full text, with every number that revisions 1–6
registered, is this file at main `ea56911fc`.

## Revisions

Revisions 2–6 applied the #1481 statistics reviews (evidence:
`~/chess-artifacts/operations/rl-claude-20261119/plan-review-1481/` and its `delta-*` folders). They are kept in git
history. In short, they:
- set head-to-heads against a no-term counterpart at the same code base, instead of matches against the base;
- added the `Zq_main` control, which exercises the WDL / Zq value-loss branch;
- fitted match 2's recalibration before any game, behind a fit control and a bounds stop;
- applied the Stage B label by script, and relabelled "unread";
- registered a page-cache warm-up of the sidecar bytes the cells read.

**Revision 7** (this one):
1. **Cells: the Zq family only**, five term-or-control cells across two parts: `Zq_main` (no term), `diff` at step-0
   share 0.50, `order` on pairs at 0.50, and `order` on root lists at 0.50 and 0.75, with a second no-term control,
   `Zq_main_fw`, on Part 2's collection. The P and S families and the three-strength ladder are dropped. What each
   loss costs is stated below.
2. **The order form** (#1501, #1503, #1510): a ranking loss on the sign of the children's value difference, gap
   weighted, with no target for its size. On root lists it covers every pair among a root's ≥ 2-visit children. This
   is the form the owner asked for (below).
3. **Training pin `TSHA` = `ea56911fc`** (the merge of #1510: list-mode ρ 0.08, `ORDER_FORM_VERSION` 3). Guards (a)
   and (b) are re-proved from the PIN to it, and guard (b)'s path list is restated.
4. **Memory:** the term forwards at most 64 pairs per chunk with a backward per chunk (`--sibling-chunk-pairs` 64).
   A 3-update GPU smoke writes peak memory after each update, and its 0.85 × total abort runs before any cell.
5. **Stage A for every term cell:** the primary is the value head's child-ordering accuracy against deep SF on the
   frozen audit set (clear pairs ≥ 50 cp), cell minus control: an INDEPENDENT ruler, and the gate to Stage B. Sign
   agreement with the search's own dq on hold-out pairs with |dq| ≥ t is a take-effect check only (it decides
   INERT, so a term that goes quiet by design is not called inert, but it is scored by the teacher itself).
6. **Equal child-forward budget** is a checked take-effect field. Active pairs, child positions and roots trained
   are recorded per step.
7. **Start net and control are stated against generation 2's Stage 0**, which read "adoptable, not replicated".
8. **Resume and legacy snapshot behaviour** after #1501 / #1503 / #1510 / #1514 is stated.
9. **The GPU check moves to the shared scan v5** (`8b4261b0…`), with the harness and wording fixes listed.
10. **Evidence-log rows** are added for step 1b, the memory diagnostic, the visit-class split, the full-width visit
    distribution, the two-visit and visits-per-child scopings, the order form's step-0 activity and the full-list ρ.

## Why

**The owner's ask, verbatim:** "As for the pairs vs ordering I was imaging we would be doing all children eval or at
least most of them but I think all will help and I was also thinking many moves are similar and order doesn't matter
but many have large gaps and order does matter as well as actual score differences but the calibration of each
individual score or the size of the differences doesn't really matter etc".

How revision 7 implements each part of it:
- **"All children eval or at least most of them":** the root-list unit takes every pair among a root's eligible
  children. A 1-visit child is not eligible, because its searched q is the collecting net's own value of that child
  and carries no ordering information (evidence below). So "all children" needs a collection in which nearly every
  legal child has ≥ 2 visits. That collection is `--root-full-width --topk 32` at 64 sims, which gives 21.4 of
  23.8 legal children ≥ 2 visits on the audit set. Part 2 trains on it.
- **"Many moves are similar and order doesn't matter":** pairs with |dq| < t = 0.0203 contribute nothing.
- **"Many have large gaps and order does matter":** each pair is weighted by min(|dq|, 0.62).
- **"The calibration of each individual score or the size of the differences doesn't really matter":** the loss is
  relu(min(m, |dq|) − sign(dq)·Δv). It is zero once the net orders the pair as the search does by the capped margin,
  and it never asks Δv to equal dq. The `diff` form (Huber on Δv − dq) does ask that, and it is kept as the
  comparison.

**What is measured** (`docs/experiments/2026-11-21-rl-loop-evidence-log.md`, Table 2; no run has trained the term):

| Evidence | Number | Bank (sha256 prefix) |
|---|---|---|
| The 64-sim search orders clear sibling pairs better than the net (vs deep SF 8M, both children ≥ 2 visits) | +0.0274 [+0.018, +0.037], 14,304 pairs | `RC/sibling-target-scoping/out/searched_order_vs_sf.json` [`5a267ec1`] |
| A pair with a 1-visit child | +0.0015 [−0.0044, +0.0071]: no information beyond the net's own | `RC/sibling-order-scoping/out/order_vs_sf_by_visit_class.json` [`f0d4a7ec`] |
| An exactly-2-visit child | (2, 2) informative: +0.0267 [+0.0086, +0.0444] (full-width topk 16), +0.0242 [+0.0069, +0.0430] (topk 64) | `RC/two-visit-children-scoping/out/analysis.json` [`b745c5f3`] |
| Children ≥ 2 visits per root (of 23.8 legal, audit set) | knob off 7.23; full-width topk 16 13.77; **full-width topk 32/64 21.42** | same; the pilot-scale count (off 7.69, on topk 16 14.55) is `RC/sibling-order-scoping/out/root_visits_fullwidth_on.json` [`cca594d7`] |
| Wide@64 vs more sims | wide@64 beats every equal-leaf knob-off control; 128 / 270 sims do not beat wide@64 | `RC/visits-per-child-scoping/out/analysis.json` [`f15c8937`] |
| Order form at step 0 (t 0.0203, m 0.0556) | active on 31.1% [29.1, 33.0] of pair weight; base orders 0.778 of clear weight as the search does | `RC/sibling-order-scoping/out/order_step0.json` [`1e7b723a`] |
| Within-root pair correlation ρ of the order loss on full lists | 0.0782 [0.0681, 0.0924] → 0.08 | `RC/sibling-fulllist-rho/out/rho.json` [`e75b7df5`] |
| S's root value loss broke sibling ordering; Zq's did not | S −0.0096 [−0.0129, −0.0064]; Zq −0.0007 [−0.0032, +0.0017] | `RC/pilot-value-diagnosis/out/sibling_order.json` [`bb8e5b16`] |
| Match 8: recalibrated Zq vs the SL base | +9.6 [+3.3, +16.0], selected as a winner | `~/chess-artifacts/operations/rl-run1-followup/arenas/zq_recal_vs_base/pool.json` [`3e407ba8`] |
| Generation 2, Stage 0: the same player, new seed | **+4.94 [−1.62, +11.50], "adoptable": consistent with match 8, does not replicate it** | [gen-2 results](2026-11-22-rl-run1-gen2-results.md) |
| Revision 6's step 1b: 255 pairs in one gradient-carrying forward | died on update 2 (OOM); reserved ≈ 16.95 + 0.0574 × pairs GiB | `~/chess-artifacts/operations/rl-run1-sibling/oprun/diag_mem/RESULT.md` [`0468d345`] |

## What revision 7 drops, and what that costs

- **The S family (three cells).** Its only role was a fallback when the Zq control failed. S is known damage
  (−20.0 vs base), and its value damage to ordering is already measured. **Cost:** the reading "does the term
  repair the ordering S breaks" is lost.
- **The P family (`--w-wdl 0`, three cells, `P0_main`).** **Cost:** the cleanest reading of "sibling training alone,
  with no root value loss" is lost. It was also the one family where the term was the value head's only signal.
- **The three-strength ladder per family.** **Costs:**
  - the dose-response reading: a monotone change in w is evidence that an effect is the term and not noise;
  - the choice of strength from data.
  The two list strengths (0.50, 0.75) keep a two-point dose check for the form the owner asked for. It is
  report-only: no match between them.
- **What stays:** one control per collection, the `diff` form as the comparison, and the `order` form in both units.

## Start net and control, given Stage 0 ("adoptable, not replicated")

**What the plan's rule gives:**
- **Start net:** every cell starts from the SL base (`sl_base.pt`, `a39f900c…`), as every pilot cell did.
- **Control:** the same Zq flags at the same pin with `--w-sibling 0`, trained from the same start on the same rows,
  batches, seed and optimizer: `Zq_main` (Part 1, pilot collection) and `Zq_main_fw` (Part 2, full-width
  collection). Each head-to-head is the term alone, plus GPU nondeterminism.
- **Why not generation 1's net (`gen1_baked.pt`, `8dd6ac8c…`) or generation 2's:**
  - The screen trains on frozen collections that the SL base played. Starting from another net would put an
    off-policy start and a second change into every contrast.
  - Generation 2 owns the RL line's GPU and its own record.

**What Stage 0 changes: what a win licenses, not the contrast.** Recalibrated Zq's standing against the base is now
"adoptable, not replicated" (+4.94 [−1.62, +11.50]; match 8's +9.6 was selected as a winner, so it is not pooled).
So:
- "The term adds" nominates **"Zq + term + per-net recalibration" over "Zq + per-net recalibration"** at the same
  start.
- It does NOT license "improves on the SL base". The control itself is not shown to improve on the base.
- Adoption into the RL line is a later record, under the RL line's own rules. It is never this screen's result
  alone.

## Collection: what runs now, what waits for #1511

| Part | Collection | Cells | Can run |
|---|---|---|---|
| **1** | the pilot's (`rl-run1-pilot/collect`, SL base, 64 sims, topk 16, knob off; 8,000 games) with the banked pair sidecar v2 (K = 2 pairs per row, both children ≥ 2 visits) | `Zq_main`, `Zq_diff_s50`, `Zq_order_s50` | after generation 2 and the preconditions |
| **2** | **a new full-width collection**: the pilot's COLLECT flags plus `--root-full-width --topk 32`, same openings, seed and net (SL base), 8,000 games; a root-list sidecar at `--min-visits 2` | `Zq_main_fw`, `Zq_list_s50`, `Zq_list_s75` | after #1511 merges and its GPU rerun passes, and the Part 2 addendum |

- **Why the list cells need the new collection:** on the pilot's collection only 7.69 of 27.65 legal children per
  root have ≥ 2 visits. Root lists there would be "the searched top 8", not "all or most".
- **Why topk 32:** at 64 sims the realised width is min(topk, legal, (sims + 1) // 2). So topk 32 is the identical
  search to topk 64, and it is the smallest topk that realises the cap. The collect's `check_vram` refuses topk 64
  at 3 × 128.
- **Why #1511:** full-width topk 32 at 3 × 128 hit the card ceiling within 30 s without forward chunking
  (`mem_wide2_20261122`). With #1511, the operator measured 10,208 MiB peak per process (topk 32, 3 × 128). #1511 is
  not merged, and one GPU rerun is due after generation 2.
- **Why the SL base collects it:** so that Part 2 differs from Part 1 only in the collection's search width, which
  is the one data-affecting change between the parts.
- **Part 2 needs its own control.** The full-width search changes the stored policy and value targets too (policy
  CE vs BT4 2.390 vs 2.402). A list cell vs Part 1's `Zq_main` would confound the term with the collection, so each
  list cell reads against `Zq_main_fw`. **No list-vs-pairs head-to-head is registered:** it would cross collections.
- **Part 2's sizes (estimates, to be measured in its addendum):**
  - collection ≈ 2.1 GPU-h (the pilot's 5,534 s for 8,000 games, divided by the 0.74 rows/s factor measured for
    full-width at a saturated GPU; #1511's chunking measured +7.5% rows/s, unpinned);
  - root-list sidecar ≈ 350 GB (≈ 21.5 KB per child, measured on the ρ run's 604,884 children; × 21.4 children
    per root × 761,979 rows) against 547 GB free on the artifacts disk on 2026-11-22.
  **The addendum holds** unless free space ≥ 1.3 × the materialiser's own size estimate at that time. Otherwise a
  materialiser option that writes only the roots the cells' plan touches (≈ 24 roots per batch × 1,412 batches,
  plus the hold-out games) is built first, under review.

## What `--w-sibling` computes at `TSHA` (`simple_rl/sibling.py`, read at `ea56911fc`)

- **`diff`** (`--sibling-loss diff`, revision 6's form): w · mean over pairs of (min(v_A, v_B) / 64) ·
  Huber_0.2(Δv − dq).
  - Δv is the difference of the two children's values, each L − W of the child's own WDL softmax, both through the
    same net in the same step.
  - dq is the searched difference of completed Q, in the parent's POV.
- **`order`** (`--sibling-loss order`, `ORDER_FORM_VERSION` 3): the per-pair term is
  (min v / 64) · 1[|dq| ≥ t] · min(|dq|, 0.62) · relu(min(m, |dq|) − sign(dq)·Δv), with t = 0.0203 and m = 0.0556.
  The margin is capped at |dq|, so a pair is never pushed apart by more than the search separates it.
  - **Pair unit** (`--sibling-unit pairs`): the term is the mean over the step's pairs.
  - **Root-list unit** (`--sibling-unit roots`): every pair among a root's children, from a root-list sidecar made
    with `scripts/simple_rl_sibling_pairs.py --unit roots --min-visits 2`. Each root's loss is the mean of its
    eligible pairs. Roots are combined with weight n_eff = n / (1 + (n − 1)·0.08) and normalised by the step's
    total weight.
  - PR 1's unweighted `order` (version 1) was never registered and is refused on restore. Version 2 (ρ 0.13) is
    refused too.
- **Rows:**
  - Pair unit: the first `--sibling-rows` (128) pair-carrying rows of each batch, all their pairs (≈ 254 per step,
    ≈ 507 child positions).
  - Root-list unit: the first `--sibling-rows` (R) roots that carry a list. **`--sibling-rows` counts roots that
    have a list, not roots with an eligible pair.** A root with no pair at |dq| ≥ t is neither forwarded nor
    counted. The trained-roots count per step is recorded (below) and stated per cell.
- **Chunks** (`--sibling-chunk-pairs` 64): pairs are forwarded at most 64 at a time (list unit: whole roots, at most
  128 child positions unless one root is larger), with a backward per chunk inside the penalty hook. This is
  gradient-equivalent to one forward up to float summation order (#1501's tests).
  - **Cost-probe phase note:** the term's backward now runs inside the Trainer's `fwd_loss_s` phase, so `bwd_s` no
    longer includes it.
- **Hold-out:** hold-out games' pairs and roots are never trained (the learner refuses otherwise).

## Code base: train at `TSHA`, match at the PIN

The term exists only on main. **Training runs at `TSHA` = `ea56911fc4e37b2814989633c5c56c6145c8d4ae`** in a
detached worktree. **Every match and every Stage A instrument except the take-effect statistic runs at the PIN
`0c795fb7a`**, as all pilot, follow-up and anchor-screen numbers did. Anything merged after `TSHA` is not used.
Part 2's collection is the one exception: it runs at the first main commit that contains #1511 (`CSHA`, pinned in
the Part 2 addendum).

**Training-path changes from revision 6's pin `9c0ff07dc` to `ea56911fc`** (`git diff --name-only 9c0ff07dc
ea56911fc -- chess_anti_engine scripts/simple_rl_loop.py`):

| Files | Change | On the training path? |
|---|---|---|
| `simple_rl/sibling.py`, `simple_rl/learn.py`, `scripts/simple_rl_loop.py` | #1501 chunking, #1501 `order` form, #1503 list unit and gap weight, #1510 ρ 0.08 and the snapshot refusals; every new `TrainSettings` field defaults to the earlier behaviour | yes. `learn.py`'s hunks are in `TrainSettings`, the resume-key and snapshot functions, `_rows_per_step` and `train_generation`'s term construction. **None is in `load_window`, `prepare_targets`, `plan_batches` or `ScheduledSampler`** |
| `simple_rl/collect.py` (#1483) | which clock fields a merge-only attempt reports | no: collect only |
| `mcts/_mcts_tree.c` / `.pyi` (#1499) | an added post-search transposition census method and one `#include`; additions only (0 removed lines) | no: search diagnostics; cells read stored `x` |
| `uci/{__main__,engine,protocol,search,walker_pool,wedge,multi_gpu_pucv_pool,root_parallel_gumbel}.py` | UCI clock, ponder, quit and pool fixes | no: the UCI engine; `uci/model_loader.py` is unchanged |

`train/` and `replay/` are unchanged from `9c0ff07dc` to `ea56911fc`.

**Guards, restated for PIN → `TSHA`:**
- **(a) The decoded training input is identical.** It was byte-equal at `9c0ff07dc` (`SB/prelaunch/
  guard_a_9c0ff07dc.verdict`). Since then no hunk touches the decode path (above), and `train/` and `replay/` are
  unchanged. **It is not re-run:** a run needs ≈ 37 GB RSS, and the diff shows nothing it could detect. The
  realised fields `train_check.py` checks, and the Zq code-base control (Stage A), cover the value-loss path on the
  real inputs.
- **(b) The encoder that wrote the stored `x` is identical, and the loader is the PIN's.** Revision 6's one-line
  check (`git diff --quiet` over `encoding/ mcts/_mcts_tree.c moves/ replay/packed_zarr.py model/ uci/
  inference.py`) is no longer empty at `ea56911fc`: #1499 added to `_mcts_tree.c`, and the UCI fixes touched
  `uci/`. Restated, checked mechanically in step 0:
  - `git diff --quiet 0c795fb7a $TSHA` over `encoding/`, `moves/`, `replay/packed_zarr.py`, `model/`,
    `inference.py` and `uci/model_loader.py` is empty (checked 2026-11-22);
  - `_mcts_tree.c`'s diff from the PIN has sha256 `532650fa7477…` and 0 removed lines (#1499's additions only);
  - `replay/shard.py`'s diff from the PIN is still exactly #1477's optional field (sha256 `418b6e96…`, unchanged).
- **(c) Default off is off on the real inputs:** `Zq_main` vs the banked Zq, paired, in Stage A (the Zq code-base
  control). Revision 6's `P0_main` control is dropped with the P family. `Zq_main` exercises everything `P0_main`
  did, plus the WDL / Zq value-loss branch.
- **(d) A main-written checkpoint loads and evaluates identically at the PIN.** `SB/prelaunch/check_load.py` on
  `Zq_main` and `Zq_main_fw` (both Stage B opponents), PIN vs `TSHA`, bit-equal or the screen stops. `model/`,
  `uci/model_loader.py` and `inference.py` are unchanged.

## Resume and legacy snapshots (what a re-run does)

Snapshots (`--snapshot-updates`) let a crashed cell resume. Since #1501 / #1503 / #1510 / #1514:
- **The chunk size is out of the resume key** and recorded per segment (`chunk_segments`). A phase killed at chunk
  64 and resumed at another chunk matches an uninterrupted run within float summation order (#1503's test; max
  |dw| 1.0e-6 where chunking really splits).
- **A real setting change never resumes.** Every key rebuilt at a recorded chunk is built from the CURRENT
  settings. The #1503 reviewer's mutant D9, which broke this, is killed by the author's tests alone.
- **Pre-#1501 sibling snapshots are refused** (they lack `chunk_segments`). A legacy sibling phase re-run with a
  real setting change stops with the move-aside message instead of restarting.
- **Other-version snapshots are refused:** an `order` snapshot of `ORDER_FORM_VERSION` 1 or 2.
- **An unreadable `resume_state.pt` or `resume_trainer.pt` is refused by name**, with nothing deleted. #1510 did
  this with the term on; #1514 (open) does it with the term off. Until #1514 merges, a damaged snapshot of
  `Zq_main` / `Zq_main_fw` surfaces as a bare traceback, which still stops the cell.
- **This screen's crash rule:** a crashed cell is renamed `arm_<cell>.crash1` and rerun ONCE, whole, from step 0,
  not from a snapshot. No cell passes `--snapshot-updates`, so none of the above can change a cell's data. It is
  stated so that an operator who resumes by hand knows what the code does.

## Strengths: the step-0 share rule, re-derived per form

**Rule (unchanged):** w is set so that at step 0 the sibling term contributes a share s of the combined value-head
gradient norm against the Zq root WDL term:
w = s / (1 − s) × ‖g_ZqWDL(value_wdl)‖ / ‖g_term(w = 1)(value_wdl)‖, with w rounded to an integer.

- **`diff`, s = 0.50: w = 122**, from revision 6's banked measurement (`SB/prelaunch/grad_scale.json`
  [`782d52a2…`]: Zq head 1.094, sibling head 0.00896; ratio 122.1). The chunked term is gradient-equivalent, so the
  value carries. The registered re-measurement below must reproduce 122 within its own ±28% noise, or the screen
  holds.
- **`order` on pairs, s = 0.50: w = `W_ORDER`**, measured before launch by `SB7/prelaunch/grad_scale.py` v2 (CPU,
  fp32, at the SL base, 4 batches of 512 pilot rows, revision 6's method). It uses the production term built by the
  same constructor the cell uses (precondition P2), `--sibling-loss order`, t 0.0203, m 0.0556, rows 128, chunk
  64. It writes `W_ORDER` to `SB7/prelaunch/strengths.json`. That file is sealed into `SHA256SUMS` before any GPU
  step, and the commands read w from it, never by hand.
- **`order` on root lists, s = 0.50 and 0.75: `W_LIST50`, `W_LIST75`**, measured the same way on Part 2's
  collection and root-list sidecar, at Part 2's R (below). Sealed in the Part 2 addendum before Part 2 trains.
- **The order form's step-0 share is a STARTING share.** A relu pair drops out once the net orders it by its
  margin. At step 0 the term is active on 31.1% of the pair weight, and that falls as training proceeds, so the
  share falls with it. `active_pairs` per step goes in the take-effect record, so the decay is visible. s = 0.75 is
  the second list strength because it is the side that keeps pressure on as the term decays.
- **The shares are labels, not measurements of what a cell experiences** (revision 6, unchanged): about ± 28% per
  w from four batches; coherence 0.63 on the head; Huber regime and weight tails as recorded.

## Cells

All cells train at `TSHA` from the SL base with the pilot's `$TRAIN` verbatim (it carries `--kl-wdl 0`) and Zq's
value flags (`$Z_V`: outcome weight 0.5, search weight 0.5, stored search WDL). Each cell differs from its part's
control by the term alone.

| Part | Cell | Term flags | Role |
|---|---|---|---|
| 1 | `Zq_main` | `--w-sibling 0` | control: Stage B opponent of the Part 1 cells; the Zq code-base control (Stage A) |
| 1 | `Zq_diff_s50` | `--sibling-loss diff --w-sibling 122` + pair sidecar v2 | the comparison form (calibrated differences) |
| 1 | `Zq_order_s50` | `--sibling-loss order --sibling-order-margin 0.0556 --sibling-order-min-dq 0.0203 --w-sibling W_ORDER` + pair sidecar v2 | the order form on the same pairs: the `order` vs `diff` contrast |
| 2 | `Zq_main_fw` | `--w-sibling 0`, Part 2 collection | control for the list cells |
| 2 | `Zq_list_s50` | `--sibling-loss order --sibling-unit roots ... --sibling-rows R --w-sibling W_LIST50` + root-list sidecar | **the owner's form**: every pair among each root's ≥ 2-visit children |
| 2 | `Zq_list_s75` | the same at `W_LIST75` | second dose of the owner's form (report-only dose check) |
| — | banked Zq (`rl-run1-pilot/arm_Zq`) | — | reference for the code-base control; not retrained |

Every sibling cell passes `--sibling-chunk-pairs 64` explicitly. The pair cells pass `--sibling-rows 128`.

**R for the list cells (equal budget).** R is the smallest integer for which the mean child positions forwarded per
step, over Part 2's own 1,412-batch plan, is ≥ 507: the pair cells' 2 × 253.7, from revision 6's warm plan of
358,198 pairs. It is computed on CPU from the plan and the sidecar (`root_pairs` at t, which drops roots without an
eligible pair, exactly as the term does) and sealed in the Part 2 addendum. **Estimate: R ≈ 24** (21.4 children per
root at full-width topk 32). On the pilot's collection the same rule would give ≈ 66 roots at 7.66 children.

## Take-effect (`SB7/stage_a/train_check.py` v3, one run, after every crash rerun)

A field mismatch writes `arm_<cell>/VOID`. **For every cell:**
- exit 0 and a sealed train manifest;
- init sha256 = the SL base's (`a39f900c…`);
- `realised_w_wdl` 1 and `realised_search_wdl_frac` 0.5;
- `updates` 1,412, and Part 1's `drawn_rows` 722,742 with `presented.npz` sha256 `63e227ac…` (Part 2: equal
  across its three cells);
- no KL anchor.

**For every sibling cell, additionally, read from `summary.sibling`:**
- `w` = the cell's sealed w; `loss`, `unit`, `order_margin`, `order_min_dq` and `order_form_version` (3 for
  `order`, null for `diff`) as the cell specifies;
- `sibling_rows` (128 or R), `chunk_pairs` 64, `chunk_segments` `[[0, 64]]`;
- `steps` = `updates`; one `per_step` entry (pairs, loss) and one `per_step_detail` entry per update (#1515:
  active pairs, child positions; the root unit adds roots trained); per-step pairs summing to `pairs` > 0, per-step
  active summing to `active_pairs`, and per-step child positions summing to `child_positions`;
- the sidecar input's sha256: Part 1 `bbd97a99…` (pair sidecar v2); Part 2 the addendum's.

**Equal budget (checked).** `child_positions_per_step` must lie in [0.8, 1.2] × 507 for every sibling cell, or the
cell is VOID ("budget not equal"). Expected: about 507 for the pair cells, by construction, and about 507 for the
list cells, by R's rule.

**Reported per cell (report-only):**
- the decay of `active_pairs / pairs` over the 1,412 steps (deciles);
- trained roots per step (root unit), against R;
- the learner's probe before and after (pairs mode: 1,024 hold-out pairs; roots mode: 128 hold-out roots), with
  mean |Δv − dq| and `sign_agree`. It comes from a handful of games, so it decides nothing.

`Zq_main` and `Zq_main_fw` must carry no `summary.sibling` and no sidecar input.

**Crash rule:** one whole rerun per cell, as above. A second crash is VOID.

## Deciding yardstick

**The one deciding yardstick: each surviving term cell's head-to-head against its part's control**: 6,400 games,
seat-balanced, both seats recalibrated, read by
`$PY $SB/stage_b/stage_b_verdict.py $SB7/arenas/sib_<cell>/pool.json`. The thresholds are pre-committed below.
Stage A only vetoes or declares a cell inert. It never selects between survivors.

### Stage A: the independent ordering ruler (primary), the take-effect check, controls

**Instruments:** the value-anchor screen's banked adapted instruments at the PIN, by sha256, exactly as revision 6
registered them. Their control (`rl-run1-value-anchor/offline/control/CONTROL_PASS`) is required, not rerun.
Ruler leakage: 0 shared game ids with the pilot collection. The Part 2 addendum repeats this check for its
collection.

**Per cell:**
- **VOID:** `train_check.py` failed the cell.
- **Primary: ordering accuracy against an INDEPENDENT reference.** On the frozen audit set the two scoping banks
  and the pilot diagnosis already score (2,433 leak-free rl-audit rows with full history, 190 games; deep SF 8M
  child scores; 38,705 "clear+" pairs, SF gap ≥ 50 cp, not decided; head ties 0.5), the accuracy with which the
  value head orders each pair of root children, **cell minus its part's control, paired by pair, 2,000
  game-clustered resamples (seed 20261121)**.
  - **Why it is the primary.** The term trains toward the SEARCH's dq. Agreement with that same dq on hold-out
    games is held out by game but scored by the teacher itself, so it cannot tell "learned the search's ordering"
    from "learned the search's errors", and a self-search ruler has inverted against an independent one here
    before (the net's own deep search scored the stored target +8.4 pp; BT4 scored it +0.7). Deep SF on the frozen
    audit set is a different engine, on a different collection (0 shared games with the pilot collection), and
    never changes.
  - **Pre-committed readings:** **"improves independent ordering" iff the 95% lower bound > 0; VETOED ("harms
    ordering") iff the upper bound < 0; otherwise "not separated".**
  - **Resolution and headroom.** Paired against the base on these pairs, the pilot arms' intervals were about
    ± 0.0025–0.0033 wide (S −0.0096 [−0.0129, −0.0064], Zq −0.0007 [−0.0032, +0.0017]). The headroom is the
    search's own advantage over the net, +0.0274 on the (≥ 2, ≥ 2) clear pairs (14,304 of the 38,705). A term that
    transferred all of it would move the all-clear+ accuracy by about +0.010; a transfer of about a third is at the
    edge of resolution. The (≥ 2, ≥ 2) subset is reported beside it, not decided on.
  - **Base-relative, report-only:** each net minus the base (the old veto's form,
    `sibling_order.json → paired_minus_base.all.clearplus_50plus`).
  - It applies unchanged to both parts: the audit set does not depend on the collection a cell trained on.
  - Built from the instrument's per-pair correctness (`sibling_children.npz` form) for every net, so the contrast
    with the control is paired by construction (precondition P3).
- **Take-effect check (demoted from revision 7's first draft; decides INERT).** The sign-agreement rate with the
  search on hold-out pairs with |dq| ≥ t = 0.0203, both children ≥ 2 visits, cell minus its part's control, paired by
  pair, with 2,000 game-bootstrap resamples (seed 20261121). It shows that the term reached the net; it is never
  evidence that the ordering is right:
  - Part 1: every hold-out pair of the pilot pair sidecar v2 (416 hold-out games);
  - Part 2: every pair among each hold-out root's ≥ 2-visit children, from Part 2's root-list sidecar (its 5%
    hold-out games).
  **"Takes effect" iff the 95% lower bound > 0. Otherwise INERT:** scored, never matched.
  - **Why this statistic:** an ordering loss goes quiet by design once pairs are ordered. Revision 6's statistic
    (mean |Δv − dq|) rewards calibrated differences, which the order form does not ask for. The same statistic is
    used for `diff`, so the two forms are judged by one rule. `diff`'s |Δv − dq| movement is reported beside it.
  - **Resolution, derived from the base's score and its bootstrap width:** on revision 6's CPU smoke (23 hold-out
    games), the base's clear-pair agreement was 0.752 with a 95% half-width ≈ 0.013. At 416 games that projects to
    ≈ 0.013 × √(23 / 416) ≈ 0.003. So a cell must move agreement by roughly 0.003 (0.4% of 0.752) to read "takes
    effect". That bar is small against the order form's 16.8% mis-ordered weight at step 0.
  - **Base-relative, report-only:** each net's agreement minus the base's.
- **Code-base control (Part 1; gates both parts).** kl_pair_Zq = mean over the hold-out rows of KL(p_Zq ‖ p_Zq_main)
  on the WDL softmax, PASS iff kl_pair_Zq ≤ max(2 × kl_rep, 0.05 × mean KL(p_base ‖ p_Zq)). Here kl_rep is the
  anchor screen's S replicate (`logits_S_trunc_final`). This is revision 6's rule and procedure, the bit-equality
  checks included. If the S replicate is void, it reads "unread: control undecidable (no replicate)", unless the
  floor alone passes.
  - **A failure means main trains differently from the PIN: no Stage B in either part**, verdict "unread: code-base
    control failed". Part 2 runs at the same `TSHA`, so it inherits the gate.
  - Part 2's control has no banked PIN counterpart on its collection. It needs none: its contrasts are within the
    part.
- **Report-only:** each cell's ΔWL and Δ draw mass vs its control, `KL_base_net_raw`, `share_of_gap_closed`,
  `KL_share_removed` by its own (T, b), FEN-only value regret and outcome distance (revision 6's list).

**To Stage B:** every term cell that takes effect AND reads **"improves independent ordering"** (primary LB > 0),
provided the code-base control passed. The hypothesis is that the term improves the value's ordering of children;
a cell that does not show that on the independent ruler has not shown the mechanism, and a 1.45 GPU-h match would
test a term whose mechanism is unseen. A "not separated" cell is reported, not matched. There is no selection by w
or by the size of the gain: up to four term matches, two per part. An offline gain still does not license a
strength claim (three fit→Elo transfer failures are on record); only Stage B does.

### Stage B: matches

| Match | Candidate | Opponent | Seed | Runs |
|---|---|---|---|---|
| B1 | `Zq_diff_s50` + its own (T, b) | `Zq_main` + its own (T, b) | **2026112371** | if B1's cell reached Stage B |
| B2 | `Zq_order_s50` + own (T, b) | `Zq_main` + own (T, b) | **2026112372** | if B2's cell reached Stage B |
| B3 | `Zq_list_s50` + own (T, b) | `Zq_main_fw` + own (T, b) | **2026112373** | if B3's cell reached Stage B |
| B4 | `Zq_list_s75` + own (T, b) | `Zq_main_fw` + own (T, b) | **2026112374** | if B4's cell reached Stage B |
| B5 (conditional) | `Zq_order_s50` + own (T, b) | `Zq_diff_s50` + own (T, b) | **2026112375** | only if both Part 1 term cells reached Stage B AND at most 3 of B1–B4 ran ("only if ≤ 3 matched") |

- Seeds 2026112371–2026112375 were checked unused on 2026-11-22: absent from `main`, from every open PR, and from
  the operations banks.
- **B5's timing:** it is decided when Part 2's Stage A is read, or when the owner declares Part 2 not launched (then
  B3 and B4 count as not run).
- **Order:** B1, B2 (after Part 1's Stage A), then B3, B4 (after Part 2's), then B5.
- **Design (each match):** revision 6's match 2 machinery, unchanged:
  - 6,400 games (3,200 per seat run, 1,600 openings), seat-balanced;
  - the follow-up launcher `rl-run1-followup/run_followup_match.sh` (`7c8fd065…`) at the PIN, output root `SB7`;
  - a 4-game smoke per seat first;
  - the result-row sha readback;
  - both seats hybrid (own value net + own recalibration), read back by `SB/stage_b/check_both_seats.py`;
  - `arena_seat_balanced_pool.py`, with the verdict printed by `SB/stage_b/stage_b_verdict.py`.
- **Recalibration, fixed before any game** (revision 6's procedure, unchanged):
  - **Fit control first:** `scripts/fit_value_recal.py` (`37e27420…`, PIN, `--device cuda`) on the pilot's Zq must
    reproduce match 8's (T, b) within 1e-3 each.
  - **Then each net's own fit:** toward the SL base, on the pilot collection's 416 hold-out games, which no cell
    trained on. Part 2's nets never saw the pilot collection at all.
  - **Bounds stop:** `SB/stage_b/recal_gate.py` v2: ln T in [−0.703, +0.352], b in [−0.4542, +0.2271], fit of the
    named net (its sha256), same reference, collect, 416 games and device. Outside them, that match stops before any
    game.
- **Rules, per match (Elo of the candidate over the opponent):**
  - **"the term adds"** (B5: "order beats diff") iff LB95 > 0;
  - **"the term harms"** (B5: "diff beats order") iff UB95 < 0;
  - otherwise **"unread: CI contains 0 (this design had 80% power only for |effect| ≥ ≈ 9)"**.
  - At SE ≈ 3.37 the 80%-power MDE is ≈ 9 Elo. P(adds) is 0.025 at a true 0, 0.32 at +5 and 0.84 at +10.
- **Multiplicity.** At most four Elo readings: B1–B4, or three of them plus B5 (B5 runs only if at most 3 of B1–B4
  ran). Revision 6 had at most two. Each
  match's false-"adds" rate at a true 0 is 0.025. **Across the readings the family-wise false-"adds" rate is ≤ 4 ×
  0.025 = 0.10 (Bonferroni bound; four matches at most run).** No correction is applied, as in revisions 1–6.
  This is why a reading NOMINATES and never adopts, and why competing nominations are settled by a fresh
  head-to-head, never by the best point estimate.

**What each outcome licenses:**

| Reading | Licenses |
|---|---|
| B1 / B2 / B3 / B4: the term adds | **NOMINATES** "Zq + that form + per-net recalibration" over "Zq + recalibration" at the same start, on that part's collection. It does not license "improves on the SL base": the control's own standing vs the base is "adoptable, not replicated" (Stage 0) |
| B3 or B4 adds | additionally: the owner's form (all ≥ 2-visit children, ranked) helps on a full-width collection. It does not separate the form from the collection; only B3 / B4 vs `Zq_main_fw` is read |
| any match: the term harms | **retires that form at that strength for the Zq recipe.** It may partly be worse calibratability under a 2-parameter recalibration (part of the treatment; see Confounds) |
| B5: order beats diff (or diff beats order) | the form choice for the pair unit, at s = 0.50, under recalibration |
| unread | nothing. The interval contains 0, which is consistent with true effects up to ≈ 13 Elo either way |
| Stage A: every cell of a form VOID, vetoed or inert | that form does not take effect at the registered dose; it is never matched |
| Stage A: takes effect, but "not separated" on the independent ruler | the term reached the net, but no improvement of the value's ordering against deep SF is shown at this resolution; not matched |
| Stage A: "improves independent ordering" | the mechanism is shown offline (the value orders children closer to deep SF); NOT a strength claim; it licenses the match |

**The two list strengths are not matched against each other.** Their Stage A agreement gains and their B3 / B4
readings are reported side by side as a two-point dose check. A difference between them is not a registered
reading.

**Competing nominations** (revision 6's rule, unchanged): if more than one recipe is nominated across the value
lines (the anchor screen, match 8, generation 2, this screen), the choice is made by a fresh head-to-head under the
next record, never by the best point estimate across screens.

## Confounds

- **Same games, two targets:** pairs come from the same games as the root rows. Hold-out means hold-out GAMES
  (`is_holdout_game` at 0.05), which no cell trains on. The veto ruler is a different collection.
- **Same search:** the sibling target is the same 64-sim search that Zq's root target came from.
- **Part 2's collection** differs from Part 1's in the search width, which also changes the stored policy and value
  targets. Every Part 2 contrast is within Part 2. **Any comparison of a list cell with a pair cell crosses
  collections and is not a registered reading.**
- **Recalibration:** each match's contrast includes how well each net calibrates under a 2-parameter recalibration,
  and the two fits' unshared sampling error on the same 416 games. Neither is in SE 3.37. The nominated recipe is
  "Zq + term + per-net recalibration", so calibratability is part of the treatment. Both seats' fit
  cross-entropies are reported.
- **Code base:** cells train at `TSHA`; the banked Zq trained at the PIN. Guards (a)–(d) and the code-base control
  license comparing them.
- **Generation 2 shares the GPU's calendar, not its data.** The screen launches after generation 2 releases the
  GPU. No cell reads generation 2's files.
- **One data-affecting change per contrast:** the term (and its sidecar). Same rows, batches, seed and optimizer
  within a part.

## Memory and cost

**The failure revision 7 fixes.** Revision 6's step 1b (the cold cost probe, `Zq_sib_w366`) completed update 1 and
died on update 2: `CUBLAS_STATUS_INTERNAL_ERROR`, then `CUDA error: out of memory`. The file is
`~/chess-artifacts/operations/rl-run1-sibling/oprun/driver.out` [`677d6436`], written 09:17. Six CPU review rounds
had never measured GPU memory. The operator's diagnostic (`oprun/diag_mem/RESULT.md` [`0468d345`]) measured peak
reserved = 16.95 + 0.0574 × pairs GiB per gradient-carrying forward:
- 16.96 GiB with no term;
- ≈ 31.6 GiB at the registered 255 pairs, on a 31.82 GiB card that also carries the Windows desktop;
- ≈ 20.6 GiB at a 64-pair chunk.
Update 1 takes 5–7 s (kernel warm-up); the probe keeps 50 updates.

**Registered memory gate: the per-update GPU smoke** (step 1b, before any cell and before the cost probe):
- `SB7/prelaunch/sib_mem_smoke.py` is generation 2's `prelaunch/train_mem_smoke/train_mem_smoke.py` (`2ce38774…`)
  reused, not rewritten. It has the same `load_window` over the whole window, `prepare_targets`, the real plan,
  `build_trainer`, and `train_steps` one update at a time. **After each update it appends and fsyncs one JSON line:**
  update, max allocated, max reserved, reserved now, device total, wall.
  - The one addition: it attaches the sibling term through the same constructor `train_generation` uses
    (precondition P2), so the smoke cannot run a different term from the cell.
  - It asserts that the term's own counters (`pairs` > 0, `child_positions` > 0, `chunk_pairs` 64) moved on every
    update. This proves the consumer received the flags.
- **3 updates per distinct forward shape:** one run for `Zq_order_s50` (pair unit, rows 128, chunk 64), one for
  `Zq_diff_s50`, and in Part 2 one for `Zq_list_s50` at R.
- **ABORT iff peak reserved over the 3 updates > 0.85 × the device total (27.0 GiB), or any update dies.** Then no
  cell trains and `SB7/HOLD_MEM` is written.
- **One registered fallback:** if a sibling smoke ABORTs at chunk 64, it is rerun ONCE at `--sibling-chunk-pairs
  32`. Chunk size is gradient-equivalent and out of the resume key, so this is not a data change. If that passes,
  every sibling cell of that part trains at 32. A second ABORT holds for the owner.
- Expected at chunk 64: ≈ 20.6 GiB (the diagnostic's fit), against the 27.0 GiB bar.

**The cost probe** (`SB7/prelaunch/cost_probe.py` revision 4 = revision 3 updated for the new `SiblingTerm`
arguments, `chunk_pairs` and `batch_size`, and building the term through P2). It runs with `Zq_order_s50`'s flags,
WARM only, after the warm-up. It also writes its per-update peaks incrementally (as the smoke does) and re-applies
the 0.85 abort. It decides the wall projection:
- **Per sibling cell:** projected wall = 419 + `sidecar_setup_s` + 1,412 × step median (updates 11–50). HOLD if the
  projected Part 1 training exceeds 2 × its estimate (2 × 1.13 GPU-h).
- **Per-cell loop budget:** (2 × 1.13 × 3,600 − 859) / 2 − 419 − `sidecar_setup_s` s. Here 859 s is the pilot Zq
  cell's process wall, charged once for `Zq_main`. It is checked after every sibling cell (`SB7/HOLD_BUDGET`). The
  realised-wall gate (1.5 × projection) applies to the first sibling cell.
- Revision 6's cold probe is dropped: it was report-only, and revision 6's CPU smoke already measured the cold read
  (1.60 s per step) that the warm-up removes.

**Page-cache plan: unchanged for Part 1, re-checked at the new pin.** Both Part 1 sibling cells read exactly the
pairs revision 6's touched set covers. The pair unit forwards every pair of the first 128 pair-carrying rows whatever
the form (the order form zeroes pairs below t; it does not skip reading them). The plan, seed and rows are
unchanged, and chunking changes the order of reads within a step, not the set. Step 1a re-derives the touched list
with `SB/prelaunch/warm_sidecar.py plan` at `TSHA`; its range-list sha256 must equal `c54a7872…`, or the screen
holds. The warm / check gates before each sibling cell are revision 6's. Part 2's touched set is derived in its
addendum.

**GPU-hours (estimates; the probe and the smoke replace them):**

| Item | Part 1 | Part 2 |
|---|---|---|
| Memory smokes (3 updates each, plus window load) | 2 × ≈ 0.1 | ≈ 0.1 |
| Cost probe (warm) | ≈ 0.15 | ≈ 0.15 |
| Collection | — (frozen) | ≈ 2.1 |
| Training: 2 sibling cells × ≈ 27 min + control ≈ 14 min | ≈ 1.13 | ≈ 1.13 |
| Stage A: 3 nets through the instruments, plus the take-effect statistic | ≈ 0.35 | ≈ 0.35 |
| Fits: fit control once, then one per net in a match | ≈ 0.15 | ≈ 0.1 |
| Matches: ≤ 4 in total across both parts (B1–B4, or 3 + B5), × ≈ 1.45 | ≤ ≈ 5.8 (shared) | |
| **Total, both parts** | **≤ ≈ 11.7** (Part 1 ≈ 2.0 + Part 2 ≈ 3.9 + matches ≤ 5.8) | |

**Against the settled envelope (≤ ≈ 9.05 GPU-h for the five-cell, one-collection sketch):** the collection
dependency adds Part 2's collection (≈ 2.1), its own control (≈ 0.25 training, plus a Stage A row and a fit) and
Part 2's smoke and probe; dropping revision 6's cold probe saves ≈ 0.15. **This is a stated deviation, forced by the
full-width requirement.** Without Part 2 (if #1511 does not land), Part 1 alone is ≤ ≈ 4.9 (two matches).

## GPU check (shared scan v5)

The operator runs `SB7/operator/operator_gpu_check.sh IDLE_MIB` before each GPU step block, as revision 6
registered (IDLE_MIB an integer ≤ 4,096; every memory line within IDLE_MIB ± 512 MiB; utilisation < 10% over 5
samples; no GPU job of this user by the shared scan). Fail-closed, read-only. Revision 7's changes, banked in
`SB7/operator/` (precondition P5):
- **Re-pin the scan to v5:** `gpu_scan.sh` sha256 `8b4261b0a6532430882233e774377e0b81c1cb27cbd0dbfd751abbec50a7f46a`
  (the root-full-width bank's staged file).
- **The one-line descendant guard.** In the descendant pass, `[ -n "${pp[$p]:-}" ]` is tested before
  `${desc[${pp[$p]}]}`, so a process whose parent pid could not be read no longer prints the bash
  "bad array subscript" message (the line-171 message of the harness output).
  - That message was fail-safe: the scan still refused or passed correctly.
  - Because the change alters the shared file, it ships as the scan owner's next version, under a new sha in all
    three banks at once. Until then v5 is pinned unchanged.
- **The harness:** the cases that kill the scan's v5 changes R1–R4. The runner asserts its case count (33) and exits
  non-zero on any mutant that is not KILLED.
  - The operator harness's ANCESTOR case is vacuous unless it runs under `env -u CUDA_VISIBLE_DEVICES`. It is
    rewritten so.
- **Wording fixes:**
  - `CUDA_VISIBLE_DEVICES=0` appears only on the GPU lines. In revision 6 that was the three cost-probe lines; in
    revision 7 it is the smoke, probe, training and fit lines.
  - The memory band's limit is stated as the measured cause: a loading job is invisible for minutes.
  - Other-uid processes are not scanned.
  - Before CUDA initialises, a process that sets the variable itself is covered only by the operator's release
    decision.
  - **Do not run guard harnesses while a real stage relies on a re-guard.**
- **Optional `--wait N`** on the operator check: re-run up to N times, 5 s apart, before refusing. The default is
  no wait.

## Launch preconditions (Part 1)

None of these exists yet. Each is built after this record's review, so that it is built to the reviewed design. Each
is tested on fabricated inputs and mutated (a kill counts only with a FAIL line and an unmutated control that
passes). Each is sealed in `SB7/*/SHA256SUMS`, and its sha256 is added to this record by an addendum PR before
launch.
- **P1.** `SB7/stage_a/cells.json` v6: the six cells, their forms and sealed w, the B1–B5 specs and seeds.
- **P2. One term constructor** (#1515): `learn.build_sibling_term(...)`, used by `train_generation`, the smoke, the
  cost probe and `grad_scale.py` v2. It adds `per_step` active pairs, child positions and roots trained, which the
  take-effect record needs. #1515 is open; it must merge, and `TSHA` moves to its merge commit by an addendum, with
  guards (a)/(b) re-checked by the same diff argument.
- **P3.** `SB7/stage_a/train_check.py` v3 (the fields above, the equal-budget VOID), `sibling_holdout.py` v2 (the
  take-effect sign-agreement statistic over |dq| ≥ t, pair and root units), `ordering_vs_control.py` (the primary:
  per-pair deep-SF clear+ correctness of each net on the frozen audit set, cell minus control, paired, game
  bootstrap), `stage_a_decide.py` v6 (VOID / VETO / INERT / not separated / improves, the Zq code-base gate on both
  parts, B1–B5 eligibility including B5's "≤ 3 matched").
- **P4.** `SB7/prelaunch/sib_mem_smoke.py`, `cost_probe.py` rev 4, `grad_scale.py` v2 → `strengths.json`.
- **P5.** `SB7/operator/`: the v5 re-pin, the harness and the wording above.
- **P6.** Generation 2 has released the GPU (its record's end), and the value-anchor screen's `offline/cells/DONE`
  and `offline/control/CONTROL_PASS` exist.

**Part 2 addendum (before Part 2 trains):** #1511 merged plus its GPU rerun; `CSHA`; the collection command (the
pilot's COLLECT plus `--root-full-width --topk 32`) and its gates; the root-list sidecar (sha256, size, disk rule
above); R; `W_LIST50` / `W_LIST75`; the touched set; ruler leakage for the new collection.

## Commands (Part 1)

The scripts named here are the preconditions above; their sha256 are added by the launch addendum. Everything not
shown is revision 6's command text at `ea56911fc` with `SB` → `SB7` for outputs.

```bash
SB=$HOME/chess-artifacts/operations/rl-run1-sibling; SB7=$HOME/chess-artifacts/operations/rl-run1-sibling-r7
P=$HOME/chess-artifacts/operations/rl-run1-pilot; AV=$HOME/chess-artifacts/operations/rl-run1-value-anchor
PIN=$(cat $P/PIN.txt); PINW=$HOME/projects/chess-worktrees/rl-run1-pilot-${PIN:0:9}
TSHA=ea56911fc4e37b2814989633c5c56c6145c8d4ae; TW=$HOME/projects/chess-worktrees/rl-run1-sibling-${TSHA:0:9}
PY=$HOME/projects/chess/.venv/bin/python; SL=$P/sl_base.pt
TB="$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6"
SF=$HOME/local_stockfish_linux_latest/stockfish/stockfish-ubuntu-x86-64-bmi2
SIDE=$HOME/chess-artifacts/operations/rl-claude-20261119/sibling-pairs-pilot/v2
# step 0: preconditions, worktree, guard (b) restated
[ -e $AV/offline/cells/DONE ] && [ -e $AV/offline/control/CONTROL_PASS ] || exit 92
for d in stage_a prelaunch operator; do (cd $SB7/$d && sha256sum -c --quiet SHA256SUMS) || exit 93; done
[ -d $TW ] || git -C $PINW worktree add --detach $TW $TSHA
cd $TW && [ "$(git rev-parse HEAD)" = "$TSHA" ] || exit 96
$PY scripts/build_production_extensions.py || exit 1
git diff --quiet 0c795fb7a $TSHA -- chess_anti_engine/encoding chess_anti_engine/moves chess_anti_engine/replay/packed_zarr.py \
  chess_anti_engine/model chess_anti_engine/inference.py chess_anti_engine/uci/model_loader.py || exit 1
[ "$(git diff 0c795fb7a $TSHA -- chess_anti_engine/mcts/_mcts_tree.c | sha256sum | cut -d' ' -f1)" = \
  532650fa74776bf0cdc27f4d9cb5f4875733d41e14d25d4b86c370e1de5e5f8b ] || exit 1
[ "$(git diff 0c795fb7a $TSHA -- chess_anti_engine/replay/shard.py | sha256sum | cut -d' ' -f1)" = \
  418b6e96e29716160dfaf439b632896191be501018a15f4412b2686d8d2901ff ] || exit 1
CUDA_VISIBLE_DEVICES= PYTHONPATH=. $PY -m pytest -q -p no:cacheprovider tests/test_simple_rl_sibling*.py || exit 1

# cell flags
COLLECT="--openings-dir $P/openings --init-checkpoint $SL --device cuda --seed 2026112100 \
  --games 8000 --collect-procs 3 --concurrency 128 --sims 64 --full-prob 1.0 \
  --topk 16 --c-scale 0.1 --c-visit 50 --max-plies 450 --syzygy $TB --sf-path $SF"
TRAIN="--presentations 1 --mirror-prob 0.5 --batch-size 512 --lr 3e-6 --lr-floor 3e-6 --lr-schedule flat \
  --warmup-steps 0 --optimizer aurora --optimizer-hparams-source init-checkpoint \
  --optimizer-state fresh --init-optimizer-state fresh --train-target-temp 1.0 \
  --kl-policy 0 --kl-wdl 0 --sl-mix-frac 0 --holdout-frac 0.05 --probe-rows 2048 --no-eval"
Z_V="--value-outcome-weight 0.5 --value-search-weight 0.5 --value-search-source stored_search_wdl"
W_ORDER=$($PY -c "import json;print(json.load(open('$SB7/prelaunch/strengths.json'))['W_ORDER'])")
SIBP="--sibling-dir $SIDE --sibling-rows 128 --sibling-chunk-pairs 64"
ORD="--sibling-loss order --sibling-order-margin 0.0556 --sibling-order-min-dq 0.0203"
declare -A CELL=(
  [Zq_main]="$Z_V --w-sibling 0"
  [Zq_diff_s50]="$Z_V $SIBP --sibling-loss diff --w-sibling 122"
  [Zq_order_s50]="$Z_V $SIBP $ORD --w-sibling $W_ORDER")

# step 1a: touched set re-derived at TSHA (CPU) -- must equal revision 6's
(cd $TW && PYTHONPATH=. $PY $SB/prelaunch/warm_sidecar.py plan $SB7/warm) || exit 1
[ "$($PY -c "import json;print(json.load(open('$SB7/warm/plan.json'))['range_list_sha256'])")" = \
  c54a78722450f8adca516cfef99aeddee752d6bcd1016bc4a379d3932a77b345 ] || exit 1

# step 1b: memory smokes (GPU alone), then the warm cost probe
$SB7/operator/operator_gpu_check.sh $IDLE_MIB || exit 8
for c in Zq_order_s50 Zq_diff_s50; do
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. $PY -u $SB7/prelaunch/sib_mem_smoke.py $SB7/mem_smoke_$c 3 $SL \
    $COLLECT $TRAIN ${CELL[$c]} || { echo "$c: smoke ABORT at chunk 64" > $SB7/HOLD_MEM; exit 1; }
done   # the registered chunk-32 fallback is a second, separately logged run of the same line with chunk 32
(cd $TW && PYTHONPATH=. $PY $SB/prelaunch/warm_sidecar.py warm $SB7/warm && \
  PYTHONPATH=. $PY $SB/prelaunch/warm_sidecar.py check $SB7/warm) || exit 1
CUDA_VISIBLE_DEVICES=0 COST_PROBE_CACHE=warm PYTHONPATH=. $PY -u $SB7/prelaunch/cost_probe.py $SB7/cost_probe_sib.json 50 \
  $COLLECT $TRAIN ${CELL[Zq_order_s50]} || exit 1

# step 2: training (revision 6's train / wall_gate / budget_check / warm_ok / crash-rerun loop, SB -> SB7)
CELLS="Zq_order_s50 Zq_diff_s50 Zq_main"
# ... then: $PY $SB7/stage_a/train_check.py $SB7
# step 3: guard (d) on Zq_main (check_load.py at the PIN and at TSHA, --compare)
# step 4: $SB7/stage_a/run_stage_a.sh $TW $TSHA   -> $SB7/offline/cells/stage_a_decision.json
# step 5: for each B-match the decision lists: fit control, both fits, both bounds stops, smoke, 2 x 3,200 games,
#         readback, pool, then the deciding yardstick:
$PY $SB/stage_b/stage_b_verdict.py $SB7/arenas/sib_$X/pool.json > $SB7/arenas/sib_$X/verdict.json
```

## Recovery plan

- **Nothing live moves.** Cells train on frozen collections into `SB7`; no live yaml, run or checkpoint is
  touched. A revert is deleting `SB7` outputs. Revision 6's bank stays frozen.
- **A HOLD file** (`HOLD_MEM`, `HOLD_WALL`, `HOLD_BUDGET`, `HOLD_WARM`, `HOLD_FIT`, `HOLD_BOUNDS`) stops the
  screen before the next GPU step, and the owner decides. No threshold is changed in flight. The one registered
  in-flight change is the chunk-32 memory fallback.
- **GPU contention:** the screen waits for generation 2. If the owner needs the GPU mid-screen, the running step is
  stopped. A training cell is then rerun whole under the crash rule; a match is rerun whole from game 0 with its
  seed, and its partial games are not read.
- **Part 2 never launches** (#1511 not merged, disk rule fails, or the owner declines): Part 1's readings stand on
  their own. B5's condition counts B3 and B4 as not run. The owner's form remains untested at full width, and the
  record says so.
