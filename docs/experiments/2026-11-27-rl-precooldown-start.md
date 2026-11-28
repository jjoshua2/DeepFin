# RL: start from before the SL learning-rate drop (pre-registration; Stages 0–1 launch-ready, Stage 2 design only)

Date: 2026-11-27. Status: registered, nothing launched. Stage 0 launches when this record is merged, the coordinator
writes `stage0/STAGE0_GO` and adds a `pc0` release line, and the value follow-up no longer holds the GPU. Stage 1
launches after `stage0/STAGE0_DONE`, `stage1/STAGE1_GO` and a `pc1` release line. Bank:
`~/chess-artifacts/operations/rl-precooldown-start/`. Code pin for every GPU step: `d08b03adb` (the deep/fast and value
follow-up pin), interpreter `~/projects/chess/.venv/bin/python`.

## Why

The owner's hypothesis, in his words: the SL run dropped the learning rate, "which makes it harder to continue on with
rl we maybe have to go back up and lose elo before we cam progress again"; "Maybe we should try continuing on from
before the [lr] drop then."

What the bank shows so far:
- **Every RL run trains at the annealed floor.** The SL base (end of SL epoch 2) finished at
  `actual_final_group_lrs` [6e-5, 3e-6, 3e-6, 3e-6] (its `training/RESULT.json`). Phase B's `TRAIN` flags are
  `--lr 3e-6 --lr-floor 3e-6 --lr-schedule flat`, with the matrix group at the imported ratio (`opt_C.json`: 6e-5).
- **Loop nets gain at 1 sim and not at 64.** They gain +75 to +90 Elo at 1 sim and −4 to −54 at 64 sims
  ([value follow-up](2026-11-27-rl-value-followup-after-deep-fast.md), "Why"; the
  [evidence log](2026-11-21-rl-loop-evidence-log.md)).
- **The second stock generation lost to its parent.** Phase B's C read −5.81 [−10.07, −1.55] against kC
  ([deep/fast record](2026-11-26-rl-deep-fast-root-allocation.md), "Phase B verdict"). One generation earlier, kC had
  read +5.08 [+0.48, +9.67].
- **The schedule's share of the SL gain is unknown.** Epoch 2 beat epoch 1 by +153.4 Elo [+124.0, +185.1] at 64 sims
  over 400 games
  (`~/chess-artifacts/operations/epoch2-vs-epoch1-fixed200-20261110-88e41e4ef670d523fc1951e2fbb83d8c/result.jsonl`).
  Epoch 2 was both another SL epoch of data and the cooldown, so the bank cannot say how much of that +153 the
  cooldown bought.

Prior evidence against raising the rate. All three items concern different nets, data and rates from what is tested
here:
- **July boot512.** An offline screen of 1,500 updates at aux 3e-4 / matrix 6e-3, ten times the donor's SL rate, read
  −193.2 [−248.7, −145.4] against boot512. At 3e-5 it read −3.5 (`docs/experiment_ledger.md`, the 2026-07-26
  controlled pre-test).
- **The fixed-G1 probe.** From the SL base, NadamW peak 3e-6 beat 3e-5 by 57.0 % [52.1, 62.0] in a 256-game
  head-to-head, on a 24,827-row bank seen about 5 times per row
  ([fixed-G1 probe](2026-11-13-fixed-g1-lr-g2-wdl.md)).
- **Neither tested Aurora at a start's own SL rate on about 2.8M fresh rows, each used once.** That is what this
  record tests.

## Hypothesis and what would falsify it

The owner's idea has two separable factors:
- **START:** the annealed SL base (E2) versus a checkpoint from before or during the cooldown (P). P is weaker, but
  may be in a region the RL loss can still move.
- **RATE:** the floor (6e-5 / 3e-6) versus P's own saved rate, which is higher.

**H (owner's claim, as tested here).** One generation from P at P's own rate gains more over its parent than one
generation from E2 at the floor gains over E2. Both are whole nets at 64 sims, under the same recipe, rows and
update count.
- **Falsified for this recipe and horizon** if the final reading of own = G(P, own rate) − G(E2, floor) has
  UB95 < 0.
- **Not supported** if the interval includes 0 at 12,800 games per arm. In that case the claim is unresolved, not
  refuted.

**Which factor carries it** is read from the full 2×2:
- the start's effect at the floor, G(P, floor) − G(E2, floor);
- the rate's effect on E2, G(E2, P's rate) − G(E2, floor);
- the rate's effect on P, G(P, own) − G(P, floor);
- the interaction, (rate effect on P) − (rate effect on E2).

These are reported with intervals. Only `own` decides.

**Terminology.** The brief called `own` "the interaction". In the 2×2 it is the diagonal contrast: the start and the
rate changed together. The interaction proper is the fifth line above, and it carries twice the variance of `own`.
`own` is the contrast the owner's claim names, so it is the one registered to decide.

**This stage cannot answer the real question.** That question is whether "RL at a higher rate, then anneal" ends above
the SL base. A one-generation gain over a weaker parent does not answer it. Stage 2 is the design that would.

## The starts (verified; receipts)

Every sha256 below was recomputed today (CPU, `sha256sum`) and equals the receipt.
- Recovery points: the `checkpoint_sha256` of the sibling `manifest.json`.
- E1 and E2: the fixed-200 match's source seal.

The saved optimizer rates were read from each checkpoint's pickled `opt.param_groups`, without torch. The method was a
`pickletools` disassembly of `data.pkl`, which is memo-aware and executes nothing.

| Name | Optimizer step | File (under `~/chess-artifacts/operations/`) | sha256 | Saved group lr [0, 1, 2, 3] |
|---|---:|---|---|---|
| E1 | 340,168 | `all-accumulated-174m-epoch1-evaluation-first-20261109-v1.resume-recovery333632-992b48793575f3ee/training/epoch-1/checkpoint.pt` | `6212930d316566eefe3f1b3b0ae7665a9c18d9e1b5cf2932bf9792b989eedd49` | 6e-4, 3e-5, 3e-5, 3e-5 |
| M | 475,738 | `all-accumulated-174m-epoch2-sqrt-cooldown-20261109-v1.resume-recovery403942-44d455e8b1b56826/training/epoch-2/recovery/step-000000475738/checkpoint.pt` | `d0d4a97b09702c0cf4a38bcd07ecdbf0a4b657ee0975aa98c139a7ffe774c298` | 1.4844000572e-4, 7.4220002862e-6 ×3 |
| (not used) | 356,239 | `…epoch2-sqrt-cooldown-20261109-v1.resume-2048e3cebc32df88/training/epoch-2/recovery/step-000000356239/checkpoint.pt` | `0d5b72671aca2328ef24429b331563337918dfa4fae4363792086a3a80b20373` | 2.4781900906e-4, 1.2390950453e-5 ×3 |
| E2 (the SL base) | 680,276 | `all-accumulated-174m-epoch2-sqrt-cooldown-20261109-v1.resume-recovery654213-a4349851abddea4a/training/epoch-2/checkpoint.pt` (byte-identical to `rl-claude-20261119/a39.pt`) | `a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0` | 6e-5, 3e-6, 3e-6, 3e-6 |

Every start's group-0 / group-2 ratio is 20.0 to float precision.

Step 475,738 sits 39.9 % of the way through epoch 2's rows: `run_rows_completed` 69,411,840 of 174,053,867. The
cooldown formula 6e-5 + 2.4e-4 × (1 − √p) gives 1.4844e-4 there, which matches the saved rate.

**The path names mislead.** M lives in the `resume-recovery403942` directory. The `resume-recovery475738` directory
holds steps 636,790 and 638,004.

## Stage 0: the strength ladder (no training; ≤ 1 GPU-h)

**Question.** What does each candidate start cost at 64 sims, before any collection is spent on it?

**Arms.** E1 and M, each against E2. Two seats over the same openings: seat A has the rung as the candidate, seat B as
the reference.

**Settings.** The value follow-up's match settings at the pin, through `common/run_v2_match.sh` (copied byte for byte):
- `matched_sims`, search shape `training`, no Gumbel noise, compile off;
- the pilot's `eval.pgn`, 16 opening plies, max plies 600 with abort;
- the production Syzygy pair, 128 concurrent games.

Seed 2026112801. 600 games per seat (1,200 per rung) at 64 sims, and the same at 1 sim (reported only).

**Precision.** The measured per-arm half-width is ±8.5 Elo at 3,200 games near 50 %. At 1,200 games that is ±13.9.
At E1's expected −150, where the score is about 0.30 and the slope is steeper, it is about ±16.5. Both are within
±20 Elo.

**The Elo map's bias at E1's distance.** The estimator is Step 1's `contrast_v.py`, which maps excess to Elo linearly
at 694.87 Elo per unit:
- at −150 it understates the logistic Elo by about 8 (−139 against −147);
- at the −75 threshold the difference is 1.4 Elo.

Neither changes the pick.

### The rule that picks P (`common/pick_p.py`, fixture-tested)

The rungs are ordered by saved rate, highest first: E1 (10× the floor), then M (2.47× the floor).
- **P = the first rung whose 64-sim point estimate vs E2 is ≥ −75 Elo.**
- If neither qualifies, P = the rung with the larger estimate, flagged `beyond_threshold`. Stage 1 still runs, with
  that caveat stated in its readout.
- The point estimate decides. The interval is reported beside it.

Why −75:
1. **Recovery budget.** Stage 2's horizon is 3–4 generations, then a cooldown. The best measured per-generation gain at
   64 sims is +5 to +8.5 Elo (kC), so 3–4 generations recover about 15–35 Elo. A start more than about 75 Elo down
   would need the closing cooldown to recover the rest: more than 40–60 Elo, about half or more of what epoch 2 plus
   its cooldown bought (+153). The bank has never measured a cooldown's share.
2. **Parent-strength confound.** The yardstick is gain over the arm's own parent. A much weaker parent has easier
   targets, so its gain is inflated independently of the rate. That confound grows with the gap.
3. **Resolvability.** E1's banked deficit is −153 [−185, −124]. At ±16.5 Elo per rung, the −75 threshold is not a
   close call for E1 unless the current-pin reading differs from the bank by about 60 Elo.

**My prior agrees with the coordinator's: M.** The rule picks E1 only if the ladder contradicts the +153 by about
60 Elo, in which case the higher-rate start is cheap and should be the one tested. Step 356,239 (2.48e-4, 4.1× the
floor) is inventoried but not laddered. It is a fallback to register in an addendum only if E1 qualifies and the
coordinator wants an intermediate point.

**Stage 0 GPU-hours.** Measured basis: phase B's `C_vs_kC` took 4,587 s per 6,400 games at 64 sims, about 0.72 s per
game. The 1-sim rate is the value follow-up's ≈ 0.05 h per 3,200 games.

| Item | Estimate |
|---|---|
| Smoke, 4 games per seat × 2 rungs | < 0.02 h |
| 64 sims, 2 × 1,200 games | ≈ 0.48 h (an upper bound: lopsided games end sooner) |
| 1 sim, 2 × 1,200 games | ≈ 0.04 h |
| **Total** | **≈ 0.55 h** |

## Stage 1: the crossed test

**Arms.** All four train one generation from their start's own fresh collection.

| Arm | Start (= parent = collect actor) | Rate: `--lr` = `--lr-floor` (groups 1–3); group 0 = 20 × |
|---|---|---|
| A | E2 | 3e-6 (floor; group 0 6e-5) |
| B | E2 | P's own: 7.4220002861554846e-06 if P = M (group 0 1.4844e-4); 3e-5 if P = E1 (group 0 6e-4) |
| C | P | P's own (as B) |
| D | P | 3e-6 (floor) |

**Recipe.** Every arm runs phase B's `TRAIN` and `ZQ` byte for byte. The harness compares them with the pinned
`phaseb/stage2_flags.sh`:
- presentations 1 (each row once), mirror probability 0.5, batch 512;
- flat schedule, warmup 0, Aurora with `--optimizer-hparams-source init-checkpoint`;
- **fresh optimizer moments** (`--init-optimizer-state fresh --optimizer-state fresh`);
- value 0.5 outcome / 0.5 stored search WDL, no KL anchors, no SL mix, hold-out 0.05.

Each arm then **appends** `--lr X --lr-floor X --matrix-lr-multiplier 20`, and argparse keeps the later value.
- **Why an explicit multiplier.** `init-checkpoint` would import the ratio 20.0 from every start anyway (verified
  above). Passing it makes group 0 = 20 × `--lr` exactly. Explicit knobs win (`simple_rl/learn.py`
  `resolve_optimizer_hparams`).
- **Why `--lr` on every arm.** Omitting it gives the CLI default 3e-5 (`scripts/simple_rl_loop.py`), so every arm
  sets it explicitly.

**Collections.** One per start, at the pin. The flags are deep/fast phase A's arm C:
- stock 64 sims, topk 16, full-prob 1.0;
- `--collect-procs 3 --concurrency 128 --eval-chunk-rows 512`;
- c_scale 0.1, c_visit 50, max plies 450;
- the production Syzygy pair and the pilot's openings.

Both collections use 27,955 games (phase A's `collect_C` dose: 2,785,627 rows from kC; the ≥ ~2.8M-row rule of the
data-volume record) and **the same seed, 2026112821**. Both therefore draw the same opening lines; only the actor
differs. Each start's two arms train on that start's collection through `--collect-from` with the same seed. Within
a start, the two rates therefore see the same rows in the same order, and the arms differ only in the rate.

**Equal exposure across starts.** E2's and P's games will differ in length, so the two collections will differ in
rows and in planned updates.
- The `plan` step runs the pin's own planner on both collects (`planned_updates.py`, phase B's tool, with the
  `--collect-from` precheck).
- It sets U = min(planned_E2, planned_P).
- Every arm writes a snapshot at exactly U updates (`--snapshot-updates U`).
- **Every arm's deciding net is its snapshot at U:** U × 512 rows, each at most once, the same count in all four arms.
- The final `model.pt` of the longer collection's arms (more updates) is banked and not read.

**Which banks could be reused, and why none is.**
- **E2's arms:** the only banked rows from an E2 actor are the pilot's: `rl-run1-pilot/collect/gen_0001/collect`,
  8,000 games, 761,979 rows, actor `a39f900c…` (verified in its manifest). Every other bank (gen2, the kC pool, the
  successor, UHO and fresh banks) was played by `gen1_baked.pt` (`8dd6ac8c…`). That is from the admission record, and
  the pilot's actor was re-read here.
- **The pilot bank fails three ways:**
  1. It holds 0.27 of the dose.
  2. It was collected at pin `0c795fb7`.
  3. The pin's `--collect-from` refuses it: its manifest has no `root_full_width` key, and `reference.settings_diff`
     back-fills only `collect_procs`. That was verified in source. So reusing it would need a code change.
- **P's arms cannot reuse any bank.** `reference_collect` requires the source collect's actor sha256 to equal the arm's
  generation-0 model (verified in source). Every banked row was played by E2 or by the annealed RL lineage, so the
  rows would be off-policy for P.
- **A fresh E2 collection** at the pin, with P's flags and seed, costs about 5.1 GPU-h. It removes a pin, settings
  and volume confound from every cross-start contrast.

**Carried SL moments: no fifth arm.** AdamW's betas are (0.9, 0.95) and Aurora's momentum is 0.95, so fresh and
carried moments differ only over roughly the first 20–100 of about 5,000 updates (< 2 %). A fifth arm would cost
about 0.55 h of training plus 0.64 h of reading, to resolve a transient the 64-sim match cannot see at this size. The
owner's phrase "continue on from before the drop" is honoured by the start and the rate. If Stage 2 runs, it can
carry moments from generation 2 on with `--optimizer-state continue`.

### Steps (`stage1/launch_stage1.sh`)

1. **`collect_E2`, `collect_P`.** Each is checked after it runs:
   - games = 27,955, and the clock is complete (no game-level resume);
   - the manifest's actor sha256 is the start's.

   **An interrupted collect is never resumed** (88): a resume keeps finished games, which is the survivorship bias of
   the evidence log. The owner moves it aside.
2. **`plan`.** U, as above. CPU only.
3. **`smoke_A..D`.** The 3-update GPU memory smoke with the arm's exact flags: `common/train_smoke_v.py`, copied, which
   records each group's realised lr after every update. Then a check after update 1:
   - group 0 within 1e-6 (relative) of 20 × lr;
   - groups 1–3 at lr.

   **A miss voids the arm:** `HOLD_LR_<X>`, and nothing trains. ABORT above 0.85 × device memory writes `HOLD_MEM`.
   Host-RAM floor 86 GiB, phase B's floor for a pool this size.
4. **`train_A..D`.** Then the take-effect check, read from the trainer's own `metrics.json` / `manifest.json`:
   - realised updates = planned;
   - the snapshot at U written;
   - realised lr per group = [20 lr, lr, lr, lr];
   - `base_lrs` the same;
   - `lr_schedule.eta_min` = lr;
   - settings `lr` and `matrix_lr_multiplier` as passed;
   - `init_optimizer_state` fresh;
   - realised search-WDL fraction 0.5.

   A miss writes `HOLD_TRAIN_CHECK` before the next arm trains.
5. **`fit_X`, `bake_X`.** Phase B's recalibration, applied the same way:
   - `scripts/fit_value_recal.py` at the pin (sha256 `37e27420…`, pinned) on the snapshot at U;
   - against the arm's **parent**, on the hold-out games of the arm's **own start's** collect;
   - then `bake_recal.py` bake and verify (copied byte for byte).

   **A deviation from the value follow-up: phase B's recal bounds are not a gate here.** Their box was fitted on
   floor-rate nets from the kC lineage. A high-rate arm outside it is an outcome to report, not a fault. The bake's
   verify remains the integrity gate: policy bit-identical and logits within tolerance. Each arm's (T, b) is reported.
6. **`msmoke_X`.** 4 games per seat: the baked net against its parent.
7. **`f64_X`.** Look 1: X_baked against its parent at 64 sims, 1,600 games per seat (3,200 per arm), seed 2026112811.
8. **`read64_1`.** The deciding yardstick (below).
9. **`x64_X`, `read64_2`.** Look 2, 4,800 per seat at seed 2026112812, for **all four** arms iff `read64_1` says
   extend.
10. **`h64_X`, `readh64`.** The head-swap: the parent's policy file with `--<seat>-value-from X_baked`, against the
    parent, 1,600 per seat. This is the secondary reading.
11. **`f16_X`, `f1_X`, `readrep`.** 800 per seat at 16 and 1 sims. Reported only.
12. **`stage1/verdict.json`, `stage1/STAGE1_DONE`.**

### The ONE deciding yardstick

Step `read64_1` runs, with B=`~/chess-artifacts/operations/rl-precooldown-start`, S=`$B/stage1/arenas`, E2 and P the
checkpoint paths above:

```
python3 $B/common/contrast_pc.py 64 --kind f --out $B/stage1/contrast_f64_looks1.json --look 2026112811 \
  A=$S/fA64/fA64_1_A.games.jsonl,$S/fA64/fA64_1_B.games.jsonl,<E2> \
  B=$S/fB64/fB64_1_A.games.jsonl,$S/fB64/fB64_1_B.games.jsonl,<E2> \
  C=$S/fC64/fC64_1_A.games.jsonl,$S/fC64/fC64_1_B.games.jsonl,<P> \
  D=$S/fD64/fD64_1_A.games.jsonl,$S/fD64/fD64_1_B.games.jsonl,<P>
```

Look 2 adds `--look 2026112812` with the `_2_` logs, written to `contrast_f64_looks12.json`.

**How the reading is computed.**
- **Per arm.** G_X is Step 1's seat-balanced per-game excess over the arm's own parent, per opening key
  (`contrast_v.arm_excess`, unchanged), averaged over keys.
- **The deciding contrast.** own = G_C − G_A, paired by opening key: all four arms play the same openings, because
  they share the seed. SE = sd over opening keys / √K. Elo through the slope at 0.5, 95 % at z = 1.96.
- **The decomposition.** The four 2×2 contrasts are reported the same way.
- **What `contrast_pc.py` refuses** (exit 2): a missing arm; A and B (or C and D) naming different parents, or the two
  parents being equal; an incomplete log; differing seeds, games, sims, opening maps or header settings; an arm on the
  wrong side of its own parent (`contrast_v2.check_orientation`); a head-swap log passed as a whole net.
- **The head-swap reading** (`--kind h`) uses the same bands, as a registered secondary reading.

**Pre-committed rule** (`contrast_pc.py` `rule()`). These are the value follow-up's bands. Its labels are stopping
labels, and every readout quotes the interval.

| Reading | Look 1 (3,200 games per arm) | Final (12,800 per arm) |
|---|---|---|
| own = G_C − G_A | \|own\| < 5: stop, "not resolved at 3,200 games". own ≥ +15 with LB > 0: P at its own rate gains more than E2 at the floor. own ≤ −15 with UB < 0: gains less. Anything else: extend. | LB > 0: more. UB < 0: less (H falsified for this recipe and horizon). Otherwise: not resolved at 12,800 games. |
| Each arm's G | ≥ +15 with LB > 0: gain over its parent. ≤ −15 with UB < 0: loss. Any other G ≥ +5: extend. Otherwise: not resolved. | LB > 0 gain, UB < 0 loss, else not resolved. |

**Extension is all or nothing.** It costs about 7.7 GPU-h. All four arms get look 2 iff look 1's `own` rule or any
arm's G rule says extend. That keeps the 2×2 balanced. The head-swap, 16-sim and 1-sim readings never extend and
decide nothing.

**Power.** These follow from the measured per-arm ±8.5 Elo at 3,200 games (value follow-up power table), with an
arm-to-arm correlation near 0, because the arms play different parents.

| Contrast | Half-width at 3,200 / arm | MDE (80 %, two-sided 5 %) | Half-width at 12,800 / arm | MDE |
|---|---|---|---|---|
| G_X (one arm) | ±8.5 | ≈ 12 | ±4.3 | ≈ 6 |
| own, and each single-factor contrast | ±12 | ≈ 17 | ±6 | ≈ 8.5 |
| interaction (four terms) | ±17 | ≈ 24 | ±8.5 | ≈ 12 |

**What the power table means for the readout.** An `own` of +10 Elo, a large effect by this bank's standards, is
detectable only at the final reading. The interaction is detectable only if it is about 12 Elo or more after
extension. **No claim about the interaction below that size will be made.**

**What is NOT a kill.**
- A first-generation dip of the high-rate arms (B, C) against the SL base is expected and decides nothing.
- So does P's own deficit against E2, which Stage 0 measures.
- Stage 1 never plays any arm against the SL base.
- Only the gain over each arm's own parent and `own` decide.

### Stage 1 GPU-hours

Measured basis:

| Item | Time |
|---|---|
| Phase A `collect_C` (stock 64, kC) | 18,375 s active for 27,955 games |
| Phase B smoke | 5.5 min |
| Phase B `train_C` | 33 min |
| Phase B fits | ≈ 5 min per fit |
| Plain 64-sim match | 0.72 s per game |
| Step 1's live hybrid | ≈ 3,370 games/h |

| Item | Estimate |
|---|---|
| Two collections | ≈ 10.2 h |
| Four smokes | ≈ 0.4 h |
| Four trainings | ≈ 2.2 h |
| Four fits and bakes | ≈ 0.45 h |
| Whole nets at 64 sims, look 1 (4 × 3,200) | ≈ 2.6 h |
| Head-swap at 64 sims (4 × 3,200) | ≈ 3.8 h |
| 16 and 1 sims (4 × 1,600 each) | ≈ 0.5 h |
| **Stage 1, look 1** | **≈ 20.1 h** |
| Plus look 2 if triggered (4 × 9,600 at 64 sims) | ≈ 7.7 h more (≈ 27.8 h in all) |

**Total registered here:**
- Stage 0 ≈ 0.55 h plus Stage 1 ≈ 20.1 h, so **≈ 20.7 GPU-h**;
- **≈ 28.4 GPU-h** if Stage 1 extends.

Disk: each collect is about 7.3 GB (phase A's `run_C`). At the time of writing the volume has 191 GB free.

**The brief's collection rate is wrong for this arm.** The brief quoted 219k stored rows per GPU-h at 64 sims. That is
deep/fast arm D's rate: 967,336 rows in 15,863 s, i.e. 219.5k rows/h. Stock 64 measured about 546k rows/GPU-h
(phase A `collect_C`: 2,785,627 rows in 5.10 h). This estimate uses the stock rate. At 219k rows/h, each collection
would take about 12.8 h.

## Stage 2 (design only; not launched by this record)

If Stage 1's final reading has `own` LB > 0, or the owner chooses to run it anyway, the winning start continues at its
rate. If `own` is unresolved, the owner decides.

- **The chain.** Generations g = 1 … 4 at the Stage 1 recipe. Each generation:
  - collects about 2.8M rows (27,955 games) from its own actor, the previous generation's baked net, at a fresh seed;
  - trains once at the start's rate (fresh moments, or `--optimizer-state continue` from generation 2: one choice,
    registered in the addendum);
  - is recalibrated against its parent and baked.

  Generation 1 is Stage 1's arm C, reused.
- **The readings, every generation.** Against the **SL base**:
  - 64 sims, 3,200 games (look 1; 12,800 at the end of the chain);
  - 16 sims and 1 sim, 1,600 games each.

  Against its own parent, at 64 sims and 3,200 games.
- **The cooldown at the end.** The last generation trains once more over a fresh collection from its own actor with
  `--lr-schedule cosine-phase --lr <rate> --lr-floor 3e-6`. That is verified in source to keep every group's ratio:
  `install_group_floor_cosine` puts group 0's floor at 20 × 3e-6 = 6e-5, not at 3e-6.
- **The deciding reading.** The cooled net against the SL base at 64 sims, 12,800 games. The question is whether
  "RL at a higher rate, then anneal" ends above the SL base: LB > 0.
- **The control.** E2 continued at the floor for the same number of generations.
  - Without it, a positive end reading cannot be credited to the rate.
  - It doubles the chain's cost.
  - Arm A's lineage is its natural first generation.
- **Cost.** About 6.5 GPU-h per generation per chain (collect 5.1, train 0.55, fits and bake 0.1, reads about 0.8).
  So about 26 h for one 4-generation chain plus about 7 h for the cooldown and its 12,800-game read. That is about
  60 h with the control chain.

## Confounds (named)

1. **The parent-strength confound.** Gain over a weaker parent is easier. The start main effect (D − A) carries it,
   and so does `own`. The P rule bounds the gap at 75 Elo, and Stage 0 reports the actual gap beside every Stage 1
   reading. The rate-within-start contrasts (C − D, B − A) do not carry it.
2. **The two starts' collections differ in actor and therefore in row content and game length.** That difference is
   part of "the start" by design: each start learns from its own play. Row count and update count are equalised at U.
3. **The recalibration is fitted per arm against that arm's parent, on that start's collect.** Two fits differ in
   their rows. The bake is part of the generation step as the current recipe defines it.
4. **Weight decay scales with the rate.** Decoupled decay exposure is lr × wd, so the high-rate arms also decay about
   2.47× (M) or 10× (E1) more per update. That is inseparable from "the rate" in this recipe.
5. **The snapshot at U.** For the longer collection's arms, the deciding net has not seen the last
   (planned − U) × 512 rows of its own shuffled plan. Those rows are a random subset, the same subset for both rates of
   that start.
6. **The search config is frozen.** Every match runs the value follow-up's settings at the pin, and every collect runs
   phase A's arm C settings.
7. **One training seed and one match seed per look.** No replication.

## Recovery and stop

Nothing here changes live training, the live yaml, any banked file or any other bank. Every output is a new path under
`~/chess-artifacts/operations/rl-precooldown-start/`. The SL checkpoints, the pilot's openings and phase B's flags are
read only.

- **Stop Stage 0:** `kill -TERM $(cat ~/chess-artifacts/operations/rl-precooldown-start/stage0/stage0.pid)`.
- **Stop Stage 1:** the same with `stage1/stage1.pid`.
- **After stopping:** check `ps` for leftover arena, collect or trainer processes whose command line names this bank.
  The trap kills the child's process group.
- **A relaunch:**
  - skips every step whose exit reads 0, and every DONE seat;
  - never re-rolls a seat;
  - stops on a PARTIAL (87) or STALE (86) seat, a `HOLD_*` or `STOP` file (81), or an interrupted collect (88).
  The owner moves the named files into a new `aborted_*` directory. Nothing is ever deleted.
- **Pause:** `stage1/STOP` is passed to the trainer as `--pause-marker`, and the launcher refuses while it exists.

## Bank and launchers

Every file is listed with its sha256 in `~/chess-artifacts/operations/rl-precooldown-start/SHA256SUMS` [`26fb2e36`].
Both launchers pin that file's sha256 and the guard's. Launchers and tests are listed in `SHA256SUMS.launch`
[`f5c988cd`].

| File | sha256 | Origin |
|---|---|---|
| `stage0/launch_stage0.sh` | `664da21c520d41e0aae43ab0d01b6abef2beb439499208e718e250f60b020c0a` | new: Step 2's launcher structure |
| `stage1/launch_stage1.sh` | `e75dae9d15db7e2063f38fbe55f5bc74ebdb102876c8d24df3c3df27121d850c` | new: Step 2's structure plus phase A's collect step |
| `common/gpu_guard_pc.sh` | `0bfcab5e…` | the value follow-up's `gpu_guard_v23.sh` [`99f51eb9`] with three stated changes (below) |
| `common/lib_pc.sh` | `db35650a…` | `common23/lib23.sh`'s run / ram floor / seat / match / step, with a per-seat parent |
| `common/pc_flags.sh` | `7ef64d49…` | the starts, rates, flags and seeds; `TRAIN` / `ZQ` byte for byte from phase B |
| `common/contrast_pc.py` | `d0609f7d…` | new; imports `contrast_v2.py` and `contrast_v.py` unchanged |
| `common/pick_p.py` | `40625ff0…` | new |
| `common/contrast_v2.py`, `run_v2_match.sh`, `train_smoke_v.py`, `planned_updates.py`, `bake_recal.py`; `contrast_v.py`, `seat_state_v.py`, `gpu_scan.sh` | as in the value follow-up's `SHA256SUMS.step23` | copied byte for byte |

**The guard's three changes.**
1. Stages `pc0` / `pc1`, released by a line `pc0 idle_mib=<MiB> <ISO time>` (or `pc1 …`) in this bank's
   `GPU_RELEASED_BY_OPERATOR`. The coordinator creates that file; it does not exist yet.
2. This bank's own lock and `guard.log`.
3. It also takes **the value follow-up bank's launch lock, read-only**, beside the deep/fast lock it already took.
   So it refuses (9) while any value follow-up or deep/fast launcher runs, and they refuse while it runs.

The memory band, the 5-sample utilisation, the process scan and the bounded re-guard are unchanged.

**Refusals, both launchers.**

| Exit | Refusal |
|---|---|
| 90 | no GO file |
| 93 | `SHA256SUMS`, a listed file, or the guard not as pinned |
| 94 | `CUDA_VISIBLE_DEVICES` empty |
| 82 | `VF_PY` / `VF_TEST_MEMINFO` set outside the harness, or a test-only smoke variable set |
| 81 | a `HOLD_*` / `STOP` file |
| 92 | a checkpoint not its receipt sha256 |
| 97 / 96 | the pin worktree not at `d08b03adb` / dirty |
| 84 | inputs changed |
| 8 / 9 | the guard |

Stage 1 also refuses with:
- 91: Stage 0 not done, or `P.json` naming no registered rung;
- 88: an interrupted collect, or a run dir of unknown origin;
- 89: the host-RAM floor;
- 85: a collect failing its check.

**Launch.** Stage 0 runs after the value follow-up has released the GPU, this record is merged, `stage0/STAGE0_GO`
exists and a `pc0` line is in the release file.
1. Check: `sha256sum ~/chess-artifacts/operations/rl-precooldown-start/stage0/launch_stage0.sh` must print
   `664da21c520d41e0aae43ab0d01b6abef2beb439499208e718e250f60b020c0a`.
2. Launch:
   `PC=~/chess-artifacts/operations/rl-precooldown-start/stage0; setsid nohup bash $PC/launch_stage0.sh >> $PC/stage0.log 2>&1 < /dev/null &`
3. Stop: `kill -TERM $(cat $PC/stage0.pid)`.

Stage 1 is the same with `stage1`, `STAGE1_GO`, a `pc1` line, and sha256
`e75dae9d15db7e2063f38fbe55f5bc74ebdb102876c8d24df3c3df27121d850c`.

**Both launchers must run with `VF_PY`, `VF_TEST_MEMINFO`, `PC_BANK`, `VF_BANK` and `DF_BANK` unset.** The first two
are refused outside the harness. The bank variables only re-point paths.

### Harness

**`~/chess-artifacts/operations/rl-precooldown-start/test/harness_pc.sh` [`7666fbf1`]: 43 of 43 pass**
(`test/harness_pc.out` [`351b5787`]).

**Setup.**
- A fake HOME with a copy of the bank.
- Read-only symlinks to the three real SL checkpoints, so the launchers' sha256 pins are checked against the real
  bytes; to phase B's `arm_C` train templates; and to the pin worktree.
- A stub interpreter that runs every stdlib tool for real (`contrast_pc.py`, `contrast_v2.py`, `contrast_v.py`,
  `seat_state_v.py`, `pick_p.py`, every inline python of the launchers), and `test/fake_pc.py` [`25809f25`] in place of
  each torch call: collect, train, planner, smoke, fit, bake, arena.
- No torch. `CUDA_VISIBLE_DEVICES` is empty for every python.
- **One stated difference from a real launch:** the fake bank's guard is a stub, with the copy's two pins rewritten.
  The real guard's changes are exercised separately, with a fake `nvidia-smi` and a stub scan.

**What it proves.**
- **Stage 0 refusals:** 90, 94, a tampered common file (93), a HOLD file (81), `VF_PY` outside the harness (82), and a
  checkpoint path re-pointed to another file (92).
- **A full Stage 0 pass:**
  - both rungs play seats against E2, with the rung as candidate in seat A;
  - the 1-sim ladder is read;
  - the fake biases give E1 < −75 < M, so `P.json` picks M without `beyond_threshold`;
  - the pid file and exit marker are written.
- **A Stage 0 relaunch** that plays nothing.
- **`pick_p.py` fixtures:** E1 at −60 picks E1; −150 / −90 picks M with `beyond_threshold`; exactly −75 qualifies; a
  1-sim ladder is refused.
- **Stage 1 refusals:** 90, Stage 0 not done (91), `P.json` naming E2 (91), a test-only variable (82), an interrupted
  collect (88).
- **The flag check:** `TRAIN` / `ZQ` equal phase B's pinned file.
- **The void:** group 0 at 1.1 × 20 lr gives a non-zero exit, `HOLD_LR_A`, no arm dir and no arena. Once the void is
  moved aside, a relaunch completes **without replaying the collects**: one collect call per start in all.
- **A full Stage 1 pass:**
  - U = min planned, with P's collect the shorter;
  - every train check is ok;
  - the realised lr is A / D 6e-5 / 3e-6 and B / C 1.4844e-4 / 7.422e-6;
  - the fits read the snapshot at U;
  - look 1 extended, and all four arms' look 2 were pooled;
  - `verdict.json` carries `own`, the decomposition and the head-swap marked secondary and non-extending;
  - the 16 / 1-sim reads are reported only.
- **`contrast_pc.py`:** the registered arms are accepted. Refused: an arm read against the wrong parent, head-swap logs
  passed as a whole net, a seat-swapped arm.
- **A Stage 1 relaunch** that collects, trains and plays nothing.
- **Nothing is written** in the fake HOME outside the fake bank.
- **The real guard:**
  - a value follow-up stage name is refused (8);
  - released and idle, it passes;
  - with the value follow-up lock held, it refuses (9);
  - while it holds, a value follow-up `flock` fails.

No mutant series, per the owner's rule.

### Not verifiable without torch (stated plainly)

These are left for the first unmutated GPU pass:
- **Loading the recovery checkpoints.** The arena and `simple_rl_loop.py collect --init-checkpoint` must load E1's and
  M's raw SL checkpoints. Stage 0's `smoke0` (4 games per seat) is the first step and checks this on the real files.
  M is a recovery bundle written by the same `Trainer.save` as E2, at 665,659,009 bytes (E2: 665,665,037).
- **The realised per-group lr** with `--matrix-lr-multiplier 20` on top of `init-checkpoint`. The source says explicit
  knobs win and group 0 = lr × multiplier (`train/trainer.py`). The smoke's update-1 check voids an arm if not.
- **That `init-checkpoint` accepts E1's and M's saved groups.** The checks are the Aurora layout flags, the fixed
  betas / eps / momentum, and the `opt_param_names` scope derivation. Codex reports a historical strict CPU load of E1
  and E2 under older code. That claim is unverified here, and the smoke tests it.
- **Collection throughput and game length for E1 / M actors.** The estimate assumes kC's.
- **Whether the high-rate arms' recalibration is sane.** It is reported, not gated.

## Codex's investigation: what was verified

| Claim | Status |
|---|---|
| The checkpoint inventory: steps, sha256s, rates for E1 340,168, M 475,738, 356,239, E2 680,276 | **verified**: hashes recomputed; rates read from the pickles; 475,738's rate recomputed from its manifest |
| E2's `actual_final_group_lrs` [6e-5, 3e-6 ×3] | **verified** (`RESULT.json`) |
| E2 vs E1 +153.44 [123.98, 185.14], 400 games, 64 sims | **verified** (`result.jsonl`) |
| Omitting `--lr` gives 3e-5 | **verified** (CLI default, `scripts/simple_rl_loop.py`) |
| `init-checkpoint` imports the ratio group 0 / group 2; explicit knobs win | **verified** (`resolve_optimizer_hparams`) |
| Trainer group 0 = lr × multiplier | **verified** (`train/trainer.py`, the Aurora branch) |
| `--collect-from` refuses the old pilot manifest (no `root_full_width`) | **verified** (`reference.settings_diff`, `SETTINGS_ADDED_SINCE`; the pilot manifest lacks the key) |
| The pilot's actor is the SL base; every later bank's is `gen1_baked` | **verified** (the pilot manifest; the admission record) |
| `reference_collect` refuses another actor | **verified** (source) |
| `--init-optimizer-state checkpoint` loads SL moments, with layout / name checks | **not verified** (not used: no fifth arm) |
| A historical strict CPU load of both endpoints | **not verified** (older code; the smoke tests it) |
| The 1,262-pair V5 estimate and other off-bank handoffs under `/mnt/c/...` | **not verified, not used** |
| July boot512 −193.2 / −3.5 and the fixed-G1 57.0 % | **verified** (ledger and record text) |

**Where this record departs from Codex's design.**
- **No sampling with replacement.** Codex proposed `--sample-with-replacement --updates 1412`. That breaks "one use per
  row". Equal exposure comes from the snapshot at U instead.
- **No fixed 8,000-game dose.** Codex used 8,000 games, about 0.27 of the dose that the data-volume record found
  necessary for a floor-rate gain. Its +10-Elo bar is replaced by the registered bands, with their MDEs stated.

## Open decisions made on the owner's behalf

1. **P rule threshold −75 Elo**, highest rate first, with M as the expected pick.
2. **27,955 games per start** (phase A's dose), not 8,000. That costs about 7 GPU-h more, buys the ≥ 2.8M-row
   regime, and makes Stage 1 about 20 GPU-h.
3. **Fresh E2 collection** instead of the banked pilot rows.
4. **No carried-moments fifth arm.**
5. **Recalibration applied, its bounds reported not gated.**
6. **Deciding net = the snapshot at U** (equal exposure); the extra updates of the longer collection are not read.
7. **The head-swap is a registered secondary reading** that never extends, so the brief's "plus the head-swap" does
   not add a second decision.
8. **16 / 1-sim reads at 800 per seat** (reported only).
9. **Step 356,239 is not laddered.**
10. **Seeds:** 2026112801 (Stage 0), 2026112821 (collects and training), 2026112811 / 2026112812 (Stage 1 looks).
