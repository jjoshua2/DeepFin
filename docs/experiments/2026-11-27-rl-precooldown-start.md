# RL: start from before the SL learning-rate drop (pre-registration; Stages 0–1 launch-ready, Stage 2 design only)

Date: 2026-11-27. Status: registered (revision 2, after the non-author review of #1555), nothing launched.
- **Stage 0** launches when this record is merged, the coordinator writes `stage0/STAGE0_GO`, and a `pc0` release line
  is written in a gap when no value follow-up launcher is live. "Bank and launchers" says how the guard prevents
  overlap and how to avoid blocking Step 3.
- **Stage 1** launches after `stage0/STAGE0_DONE`, `stage1/STAGE1_GO` and a `pc1` release line.
- **Revision 2:**
  - Stage 1 runs the P half first and buys the E2 half only if a registered gate passes;
  - the head-swap reading is dropped;
  - extension follows only the current half's deciding contrast;
  - the Stage 2 trigger also needs the parent-matched rate contrast;
  - the train check compares every non-LR optimizer field with phase B's record;
  - the harness asserts every contrast's value.
 Bank:
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
- If the final own = G(P, own rate) − G(E2, floor) has UB95 < 0, the reading is **"P at its own rate gains less than
  E2 at the floor"** in one generation. That is not a falsification of the owner's idea, which predicts a first-
  generation loss before progress ("go back up and lose elo before we can progress again"). Only the Stage 2 chain can
  test "dip, then gain".
- If the interval includes 0 at 12,800 games per arm, the reading is unresolved, not refuted.
- **The rate's own effect, separated from parent headroom,** is rate_P = G(P, own) − G(P, floor). It compares two arms
  with the same parent and the same rows. **It decides the P half and gates the E2 half** (Stage 1).

**Which factor carries it** is read from the full 2×2:
- the start's effect at the floor, G(P, floor) − G(E2, floor);
- the rate's effect on E2, G(E2, P's rate) − G(E2, floor);
- the rate's effect on P, G(P, own) − G(P, floor);
- the interaction, (rate effect on P) − (rate effect on E2).

rate_P decides the P half, and `own` decides the E2 half. The other contrasts are reported with intervals.

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
- **The −75 threshold only gates E1. It never stops Stage 1.** M is P whenever E1 fails, flagged `beyond_threshold`
  if M is also below −75, even if both rungs read −150.
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

## Stage 1: the crossed test, run as two halves (revision 2, after the #1555 review)

**Arms.** Each arm trains one generation from its start's own fresh collection.

| Arm | Start (= parent = collect actor) | Rate: `--lr` = `--lr-floor` (groups 1–3); group 0 = 20 × |
|---|---|---|
| A | E2 | 3e-6 (floor; group 0 6e-5) |
| B | E2 | P's own: 7.4220002861554846e-06 if P = M (group 0 1.4844e-4); 3e-5 if P = E1 (group 0 6e-4) |
| C | P | P's own (as B) |
| D | P | 3e-6 (floor) |

**Order and gate.** The P half runs first, then a gate decides whether the E2 half is bought.

1. **P half:** collect from P, train C and D, play both against P, and read
   **rate_P = G_C − G_D**. That contrast compares two arms with the same parent, the same rows and the same row order,
   differing only in the rate. It carries no parent-headroom term.
2. **The gate, registered:** the P half's final rate_P **point estimate ≥ 0**.
   - **If it fails,** Stage 1 stops with the registered reading **"at one generation the higher rate does not beat the
     floor from P"**. E2 is never collected, and A and B never train.
   - **If it passes,** the E2 half runs: collect from E2, train A and B, and read **own = G_C − G_A** (deciding) and the
     reported 2×2.

**The stop does not falsify "dip, then gain".** The owner's hypothesis predicts that the higher rate first loses Elo
("go back up and lose elo before we can progress again"). A failed gate is a one-generation result. Only the Stage 2
chain can test a dip followed by a gain. The gate exists because the E2 half cannot rescue that case. If the higher rate
does not beat the floor from P after one generation, `own` would at best show parent headroom (finding 2 of the
review). It would not show a rate effect, and the E2 half's ~8 GPU-h would buy no evidence about the rate.

**Recipe.** Every arm runs phase B's `TRAIN` and `ZQ` byte for byte. The harness compares them with the pinned
`phaseb/stage2_flags.sh` (`043b4810…`):
- presentations 1 (each row once), mirror probability 0.5, batch 512;
- flat schedule, warmup 0, Aurora with `--optimizer-hparams-source init-checkpoint`;
- **fresh optimizer moments** (`--init-optimizer-state fresh --optimizer-state fresh`);
- value 0.5 outcome / 0.5 stored search WDL, no KL anchors, no SL mix, hold-out 0.05.

Each arm then **appends** `--lr X --lr-floor X --matrix-lr-multiplier 20`, and argparse keeps the later value.
- **Why an explicit multiplier.** `init-checkpoint` would import the ratio 20.0 from every start anyway (verified
  above). Passing it makes group 0 = 20 × `--lr` exactly. Explicit knobs win (`resolve_optimizer_hparams` fills only
  unset ones).
- **Why `--lr` on every arm.** Omitting it gives the CLI default 3e-5, so every arm sets it explicitly.

**Collections.** One per start, at the pin. The flags are deep/fast phase A's arm C:
- stock 64 sims, topk 16, full-prob 1.0;
- `--collect-procs 3 --concurrency 128 --eval-chunk-rows 512`;
- c_scale 0.1, c_visit 50, max plies 450;
- the production Syzygy pair and the pilot's openings.

Both collections use 27,955 games and **the same seed, 2026112821**, so both draw the same opening lines.

Each collect step also runs two of phase A's monitors (copied):
- **The per-process memory log.** Phase A's `RSS_PY`, unchanged: every `DF_RSS_INTERVAL_S` seconds (default 600), one
  line per process ≥ 256 MiB in the collect's session, written to the launcher's log.
- **The GPU-holder sampler.** `common/gpu_holders.py`, byte for byte (`9991af8d`): it scans for other GPU holders every
  15 s. Its output, `collect_<s>.gpu_holders.jsonl`, and the `check` summary are banked. The sampler is never a gate.

**Dose and exposure.**
- **The dose is 27,955 games, phase A's `collect_C` count.** From kC that gave 99.6 rows per game, 2,785,627 rows and
  5,174 updates.
- **From an E2 actor the pilot measured 95.2 rows per game** (761,979 / 8,000, at pin `0c795fb7`). The E2 collection is
  therefore expected at about **2.66M rows**:
  - about 2.53M of them trainable after the 0.05 hold-out;
  - **U_E2 ≈ 4,940 updates**, against phase B's 5,174.
  - That is just under the ~2.8M rows the data-volume and phase B steps used. The record does not claim the 2.8M
    regime for E2's arms.
- **P's rows per game (M or E1 actor) are unmeasured.** U_P is whatever P's collect gives.
- **Equal exposure, per contrast. U is the minimum across the arms that contrast compares.**
  - **rate_P (C vs D)** uses one pool, so both arms run to **U_P** = P's planned updates. Their deciding nets are their
    snapshots at U_P.
  - **C and D also snapshot on a grid** every 100 updates from 100 × ⌈0.85 U_P / 100⌉ up to U_P.
  - **own (C vs A)** uses **U_own**:
    - U_own = U_P if E2's planned updates are ≥ U_P;
    - otherwise U_own is the largest grid point ≤ E2's planned updates;
    - if E2's planned updates fall below the grid's low end, `HOLD_DOSE` is written and the owner decides.
  - A and B snapshot at U_own.
  - When U_own < U_P, C's snapshot at U_own is recalibrated, baked and played as its own net, **Co**. `own` then
    compares A with Co. That costs about 0.8 GPU-h more.
  - With E2's expected 4,940 updates, U_own is about 4,900 if P's collect is the longer.
- **The reported 2×2 uses C and D at U_P and A and B at U_own.** When the two differ, its cross-start contrasts other
  than `own` mix exposures by at most (U_P − U_own) updates. They are reported, not decided.

**Which banks could be reused, and why none is.**
- **E2's arms:** the only banked rows from an E2 actor are the pilot's: `rl-run1-pilot/collect/gen_0001/collect`,
  8,000 games, 761,979 rows, actor `a39f900c…` (verified in its manifest). Every other bank was played by
  `gen1_baked.pt` (`8dd6ac8c…`). That is from the admission record, and the pilot's actor was re-read here.
- **The pilot bank fails three ways:**
  1. It holds 0.27 of the dose.
  2. It was collected at pin `0c795fb7`.
  3. The pin's `--collect-from` refuses it: no `root_full_width` key, and `reference.settings_diff` back-fills only
     `collect_procs`. That was verified in source. Reusing it would need a code change.
- **P's arms cannot reuse any bank.** `reference_collect` requires the source collect's actor sha256 to equal the arm's
  generation-0 model (verified in source).

**Carried SL moments: no fifth arm.** AdamW's betas are (0.9, 0.95) and Aurora's momentum is 0.95, so fresh and
carried moments differ only over roughly the first 20–100 of about 5,000 updates.

**No head-swap reading (dropped in revision 2).** It cost about 3.8 GPU-h and decided nothing.

### Steps (`stage1/launch_stage1.sh`)

**P half:**
1. **`collect_P`.** Checked after it runs:
   - games = 27,955, with a complete clock;
   - the manifest's actor sha256 is P's.

   An interrupted collect is never resumed (88).
2. **`plan_P`.** The pin's planner gives U_P, and the snapshot grid is re-checked. CPU only.
3. **`smoke_C`, `smoke_D`.** The 3-update GPU memory smoke, then the update-1 lr check: group 0 within 1e-6 of
   20 × lr, groups 1–3 at lr. **A miss voids the arm** (`HOLD_LR_<X>`). Host-RAM floor 86 GiB.
4. **`train_C`, `train_D`.** Then the take-effect check, **`common/opt_check_pc.py`**.
   - **Derived from** the value follow-up's `opt_check_v.py`, itself from phase B's `opt_check.py`.
   - **Its reference** is phase B's `arm_C` `metrics.json`, pinned by sha256 `1a99a9fa…` (refusal 92).
   - **Its checks:**
     - every non-LR field of `realised_optimizer` equals that reference's: weight decays, betas, eps, Aurora knobs,
       group sizes, built split, grad clip, scheduler class / T_0 / warmup;
     - the train manifest's `init_checkpoint` sha256 is the start's;
     - groups 1–3 = lr in all four places, and `base_lrs[0]` = lr × 20;
     - group 0 in its closed form at the last update;
     - **`lr_schedule.eta_min` = lr**, checked positively and stripped from the comparison, because it legitimately
       equals each arm's floor (the Step 3 blocker of #1553);
     - settings: flat schedule, warmup 0, accum 1, hparams from the init checkpoint, both optimizer states fresh, `lr`,
       `lr_floor` and multiplier as passed;
     - updates = planned, and the registered snapshot list with its files;
     - `realised_w_wdl` 1, search fraction 0.5, no anchor, policy target temperature 1, SL fraction 0.
   - **Validated on real data:** on phase B's own `arm_C`, with its imported multiplier 19.999999998663274, every
     check passes. With multiplier 20, or with lr 3e-5, it fails.
   - A miss writes `HOLD_TRAIN_CHECK` before the next arm trains.
5. **`fit_C`, `bake_C`, `fit_D`, `bake_D`.** Phase B's recalibration on the snapshot at U_P, against P, on P's
   collect's hold-out games. Bake and verify. Bounds are reported, not gated.
6. **`msmoke_C`, `msmoke_D`.** 4 games per seat.
7. **`f64_C`, `f64_D`.** Look 1: 1,600 games per seat at 64 sims, seed 2026112811.
8. **`readP_1`.** `contrast_pc.py --decide rate_P`.
9. **`x64_C`, `x64_D`, `readP_2`.** Look 2 (4,800 per seat, seed 2026112812), only if rate_P's look-1 rule says
   extend.
10. **`f16`, `f1` (C, D), `readrepP`.** 800 per seat, reported.
11. **The gate.** If it fails: `verdict.json` (branch `p_half_stop`), then `STAGE1_DONE`, exit 0.

**E2 half (only if the gate passes):**

12. **`collect_E2`, `plan_E2`.** U_own, or `HOLD_DOSE`.
13. **`smoke_A`, `smoke_B`; `train_A`, `train_B`** (snapshot at U_own), with the same checks.
14. **`fit` / `bake` for A and B,** on the snapshot at U_own, against E2, on E2's collect. Plus **`fit_Co`, `bake_Co`,
    `f64_Co`** when U_own < U_P.
15. **`msmoke_A`, `msmoke_B`; `f64_A`, `f64_B`.** Look 1.
16. **`readE_1`.** `contrast_pc.py --decide own` on A and C (or Co), with `--rate-p` the P half's final file.
17. **`x64_A`, `x64_<C|Co>`, `readE_2`.** Look 2, only if own's look-1 rule says extend. C's look 2 is reused if the P
    half already played it.
18. **`read2x2`.** Reported.
19. **`f16`, `f1` (A, B), `readrepE`.** Reported.
20. **`verdict.json`** (branch `full`), then **`STAGE1_DONE`**.

### The ONE deciding yardstick, per half

B=`~/chess-artifacts/operations/rl-precooldown-start`, S=`$B/stage1/arenas`, E2 and P the checkpoint paths above.

**P half (step `readP_1`):**
```
python3 $B/common/contrast_pc.py 64 --decide rate_P --out $B/stage1/contrast_rateP_looks1.json --look 2026112811 \
  C=$S/fC64/fC64_1_A.games.jsonl,$S/fC64/fC64_1_B.games.jsonl,<P> \
  D=$S/fD64/fD64_1_A.games.jsonl,$S/fD64/fD64_1_B.games.jsonl,<P>
```

**E2 half (step `readE_1`):**
```
python3 $B/common/contrast_pc.py 64 --decide own --rate-p <the P half's final file> \
  --out $B/stage1/contrast_own_looks1.json --look 2026112811 \
  A=$S/fA64/fA64_1_A.games.jsonl,$S/fA64/fA64_1_B.games.jsonl,<E2> \
  C=$S/f<C|Co>64/f<C|Co>64_1_A.games.jsonl,$S/f<C|Co>64/f<C|Co>64_1_B.games.jsonl,<P>
```

Look 2 adds `--look 2026112812` with the `_2_` logs, written to `*_looks12.json`.

**How each reading is computed.**
- **Per arm.** G_X is Step 1's seat-balanced per-game excess over the arm's own parent, per opening key
  (`contrast_v.arm_excess`, unchanged).
- **Contrasts.** Each contrast is a same-key mean of per-key differences, so it equals the corresponding combination of
  G's exactly. The harness asserts this.
- **Pairing.** By opening key: every arm plays the same openings, because they share the seed. SE = sd over keys / √K.
  Elo through the slope at 0.5, 95 % at z = 1.96.
- **What `contrast_pc.py` refuses:** the deciding contrast's arms missing; A and B (or C and D) with different parents;
  an E2-side parent equal to a P-side parent; incomplete logs; differing seeds, games, sims, opening maps or header
  settings; an arm on the wrong side of its own parent; a head-swap log; a `--rate-p` file that is not a final 64-sim
  rate_P reading.

**Pre-committed rule** (`contrast_pc.py` `band()` / `rule()`, edge-tested). These are stopping labels, and every
readout quotes the interval.

| Reading | Look 1 (3,200 games per arm) | Final (12,800 per arm) |
|---|---|---|
| rate_P = G_C − G_D (P half) | \|x\| < 5: stop, "not resolved at 3,200 games". x ≥ +15 with LB > 0: "the higher rate beats the floor from P". x ≤ −15 with UB < 0: "the higher rate loses to the floor from P". Anything else: extend C and D. | LB > 0 beats / UB < 0 loses / else not resolved. |
| **Gate** (on the P half's final reading) | **Pass iff the rate_P point estimate ≥ 0.** Otherwise stop: "at one generation the higher rate does not beat the floor from P". | |
| own = G_C − G_A (E2 half) | \|own\| < 5: stop, not resolved. own ≥ +15 with LB > 0: "P at its own rate gains more than E2 at the floor". own ≤ −15 with UB < 0: "P at its own rate gains less than E2 at the floor". Anything else: extend A and C. | LB > 0 more / UB < 0 less / else not resolved. |
| Each arm's G | Reported label only (gain / loss / not resolved). **It never extends.** | Same. |

**Extension.**
- Look 2 is played only when the current half's deciding contrast is unresolved at look 1, and only for that
  contrast's two arms. That costs about 3.8 GPU-h per half.
- An arm's own G never triggers it.
- The 2×2, 16-sim and 1-sim readings never extend and decide nothing.

**Stage 2 trigger** (registered; `contrast_pc.py` writes it into `stage2` of own's final reading):
- **The rate chain is triggered iff own's final LB > 0 AND the P half's final rate_P point estimate ≥ 0.**
- If own LB > 0 but rate_P < 0, the reading is **"start headroom, not rate"**, and the high rate is not carried into
  Stage 2.
- If own LB ≤ 0, there is no trigger.

**A limit of this trigger, stated rather than fixed.** The gate already requires rate_P ≥ 0, so whenever the E2 half
runs the trigger's rate_P clause holds. Through the launcher, "start headroom, not rate" is therefore unreachable, and
the trigger reduces to own LB > 0. A point estimate ≥ 0 with a ±12-Elo half-width does not exclude a pure headroom
effect (C − D ≈ 0, D − A > 0). A stricter rule (rate_P LB > 0, or rate_P ≥ some share of own) would close that gap at
the cost of power. It is not adopted here. Each readout quotes D − A and C − D beside own.

**Power.** These follow from the measured per-arm ±8.5 Elo at 3,200 games (value follow-up power table).

| Contrast | Half-width at 3,200 / arm | MDE (80 %, two-sided 5 %) | Half-width at 12,800 / arm | MDE |
|---|---|---|---|---|
| G_X (one arm) | ±8.5 | ≈ 12 | ±4.3 | ≈ 6 |
| rate_P, own, each single-factor contrast | ±12 | ≈ 17 | ±6 | ≈ 8.5 |
| interaction (four terms; look 1 only, reported) | ±17 | ≈ 24 | n/a | n/a |

- **rate_P** pairs two arms with the same parent and the same rows, so its arm-to-arm correlation may exceed 0 and its
  band narrow. Each readout reports the realised correlation.
- **A +10-Elo effect** on either deciding contrast is detectable only at the final reading.
- **No claim about the interaction** will be made: it is reported at look 1 only.

**What is NOT a kill.**
- C and D are never played against the SL base. A and B play it only as their own parent.
- P's deficit against E2 (Stage 0), and any first-generation dip of the high-rate arms against the SL base, decide
  nothing.
- **A failed gate is a one-generation reading,** consistent with the owner's predicted dip. It ends Stage 1's spending.
  It does not refute "dip, then gain".

### Stage 1 GPU-hours, by branch

Measured basis:

| Item | Time |
|---|---|
| Phase A `collect_C` | 18,375 s active for 27,955 games |
| Phase B smoke | 5.5 min |
| Phase B `train_C` | 33 min |
| Phase B fits | ≈ 5 min per fit |
| Plain 64-sim match | 0.72 s per game |

| Item | P half | E2 half |
|---|---|---|
| Collection | 5.1 | 5.1 |
| Two smokes | 0.2 | 0.2 |
| Two trainings (C / D carry about 8 extra weights-only snapshots each) | 1.1 | 1.1 |
| Fits and bakes | 0.25 | 0.25 (+0.1 for Co) |
| 64 sims, look 1 (2 × 3,200) | 1.3 | 1.3 (+0.65 for Co) |
| 16 and 1 sims (2 × 1,600 each) | 0.25 | 0.25 |
| **Half, look 1** | **≈ 8.2 h** | **≈ 8.2 h (+0.75 if U_own < U_P)** |
| Look 2 of the deciding contrast if triggered (2 × 9,600) | +3.8 | +3.8 (+1.9 if the P half already played C's look 2 and U_own = U_P) |

| Branch | Stage 0 + Stage 1 | Worst case with every extension |
|---|---|---|
| **Gate fails (stop after the P half)** | 0.55 + 8.2 ≈ **8.7 GPU-h** | ≈ 12.6 |
| **Gate passes (full)** | 0.55 + 8.2 + 8.2 ≈ **17.0 GPU-h** | ≈ 25.4 (with Co) |

Revision 1 was ≈ 20.7, or ≈ 28.4 with extension. Disk: about 7.3 GB per collect, plus the snapshot grids (about
16 × 0.25 GB). At the time of writing the volume has 191 GB free.

**The brief's collection rate is wrong for this arm.** The brief quoted 219k stored rows per GPU-h at 64 sims. That is
deep/fast arm D's rate: 967,336 rows in 15,863 s. Stock 64 measured about 546k rows/GPU-h (phase A `collect_C`). This
estimate uses the stock rate.

## Stage 2 (design only; not launched by this record)

**Triggered only by the registered Stage 2 trigger above:** own's final LB > 0 and rate_P ≥ 0. On "start headroom, not
rate", on a failed gate, or on an unresolved own, the owner decides. A dip-then-gain chain can still be run as a
deliberate choice. A failed gate does not forbid it; the gate only saves the E2 half.

- **The chain.** Generations g = 1 … 4 from P at P's rate, at the Stage 1 recipe. Each generation:
  - collects about 2.7–2.8M rows from its own actor, the previous generation's baked net;
  - trains once at P's rate (fresh moments, or `--optimizer-state continue` from generation 2: one choice, registered
    in the addendum);
  - is recalibrated against its parent and baked.

  Generation 1 is Stage 1's arm C, reused.
- **The readings, every generation.** Against the **SL base**:
  - 64 sims, 3,200 games (12,800 at the end of the chain);
  - 16 sims and 1 sim, 1,600 games each.

  Against its own parent, at 64 sims and 3,200 games.
- **The cooldown at the end.** The last generation trains once more over a fresh collection from its own actor with
  `--lr-schedule cosine-phase --lr <rate> --lr-floor 3e-6`. That is verified in source to keep every group's ratio:
  `install_group_floor_cosine` puts group 0's floor at 6e-5.
- **The deciding reading.** The cooled net against the SL base at 64 sims, 12,800 games, LB > 0.
- **The control.** E2 continued at the floor for the same number of generations. Arm A's lineage is its natural first
  generation. It doubles the chain's cost.
- **Cost.** About 6.5 GPU-h per generation per chain, so about 26 h for one chain plus about 7 h for the cooldown and
  its read. That is about 60 h with the control chain.

## Confounds (named)

1. **The parent-strength confound.**
   - It is carried by `own` and by the start main effect (D − A).
   - It is handled by deciding the P half on rate_P, which carries no headroom term, by the gate, and by the Stage 2
     trigger's rate_P clause.
   - It is not eliminated: see the trigger's stated limit. Stage 0 reports the actual gap.
2. **The two starts' collections differ in actor.** Row content and game length therefore differ. That is part of "the
   start" by design. Exposure is equalised per contrast at U_P or U_own.
3. **Recalibration is fitted per arm** against that arm's parent, on that start's collect.
4. **Weight decay scales with the rate.** Decoupled decay exposure is lr × wd, so the high-rate arms also decay 2.47×
   (M) or 10× (E1) more per update. This is inseparable from "the rate".
5. **Exposure differences in the reported 2×2** when U_own < U_P (above).
6. **The halves run at different times,** at the same pin and with the same settings.
7. **The search config is frozen.**
8. **One training seed and one match seed per look.** Two looks at an unadjusted z = 1.96 inflate α modestly, a
   property inherited from the value follow-up's bands.

## Recovery and stop

Nothing here changes live training, the live yaml, any banked file or any other bank. Every output is a new path under
`~/chess-artifacts/operations/rl-precooldown-start/`.

- **Stop:** `kill -TERM $(cat <bank>/stage<N>/stage<N>.pid)`. The trap kills the tracked child, the GPU-holder sampler
  and the memory logger. Then check `ps` for leftover processes naming this bank.
- **A relaunch:**
  - skips every step whose exit reads 0, and every DONE seat;
  - never re-rolls a seat;
  - stops on a PARTIAL (87) or STALE (86) seat, a `HOLD_*` or `STOP` file (81), or an interrupted collect (88).
  The owner moves the named files into a new `aborted_*` directory. Nothing is ever deleted.
- **`stage1/STOP`** pauses only inside the trainer (`--pause-marker`). The launcher refuses while it exists.

## Bank and launchers

Every runtime file is listed in `~/chess-artifacts/operations/rl-precooldown-start/SHA256SUMS` [`80965d6f`]. Both
launchers pin that file's sha256 and the guard's. Launchers, tests and outputs are listed in `SHA256SUMS.launch`
[`a1e836e0`].

| File | sha256 | Origin |
|---|---|---|
| `stage0/launch_stage0.sh` | `c09a90a17ec7a4dd79f6c5477635ee3f5f9087c34bcefc8e3fc4b3592a8db49e` | new: Step 2's launcher structure (only the `SHA256SUMS` pin changed in revision 2) |
| `stage1/launch_stage1.sh` | `5145a2168a03b47bb410a2f1f5233f7e377f9093f1827ed806a00b4276b9754a` | revision 2: P half, gate, E2 half |
| `common/opt_check_pc.py` | `e4210b43…` | derived from `opt_check_v.py` [`1c75c174`], changes in its header |
| `common/contrast_pc.py` | `43bf5557…` | revision 2: per-half deciding contrast, gate, Stage 2 trigger; imports `contrast_v2.py` / `contrast_v.py` unchanged |
| `common/gpu_holders.py` | `9991af8d…` | phase A's, byte for byte |
| `common/gpu_guard_pc.sh` | `0bfcab5e…` | the value follow-up's `gpu_guard_v23.sh` [`99f51eb9`] with three stated changes |
| `common/lib_pc.sh` | `db35650a…` | `common23/lib23.sh`'s run / ram floor / seat / match / step, with a per-seat parent |
| `common/pc_flags.sh` | `7ef64d49…` | the starts, rates, flags and seeds; `TRAIN` / `ZQ` byte for byte from phase B |
| `common/pick_p.py` | `40625ff0…` | Stage 0's P rule |
| `common/contrast_v2.py`, `run_v2_match.sh`, `train_smoke_v.py`, `planned_updates.py`, `bake_recal.py`; `contrast_v.py`, `seat_state_v.py`, `gpu_scan.sh` | as in the value follow-up's `SHA256SUMS.step23` | copied byte for byte |

**The guard's three changes.**
1. Stages `pc0` / `pc1`, released by a line `pc0 idle_mib=<MiB> <ISO time>` (or `pc1 …`) in this bank's
   `GPU_RELEASED_BY_OPERATOR`. The coordinator creates that file.
2. This bank's own lock and `guard.log`.
3. It also takes the value follow-up bank's launch lock, read-only, beside the deep/fast lock.

**How the guard prevents overlap with the value follow-up, and how to avoid blocking Step 3.**
- **Stage 0 may run in any gap when no value follow-up launcher is live.**
- **While a value follow-up launcher runs,** this guard refuses at once, before any marker:
  - with 9, because that launcher holds its bank lock on fd 9, which the review confirmed in `/proc/locks` for the live
    Step 2;
  - and with 8 at the GPU memory band.
- **While this launcher runs,** it holds the value follow-up and deep/fast locks exclusively. A value follow-up
  launcher then refuses at its own `flock -n 9` (exit 9) and spends nothing. Two launchers can never hold the GPU
  together.
- **The cost is that a Step 3 launch attempted during Stage 0 is refused.** So the coordinator writes the `pc0` line
  only in a gap of at least about 0.6 h before Step 3 is due: Stage 0 is ≈ 0.55 GPU-h. If Step 3 is refused anyway, it
  is relaunched unchanged once `stage0/exit` exists. Its refusal happens before any marker except an unchanged
  `inputs.txt`.
- **Stage 1 (~8–25 h) should get its `pc1` line only after Step 3 completes or is cancelled.**

**Refusals, both launchers.**

| Exit | Refusal |
|---|---|
| 90 | no GO file |
| 93 | `SHA256SUMS`, a listed file, or the guard not as pinned |
| 94 | `CUDA_VISIBLE_DEVICES` empty |
| 82 | `VF_PY` / `VF_TEST_MEMINFO` outside the harness, or a test-only smoke variable |
| 81 | `HOLD_*` / `STOP` |
| 92 | a checkpoint, or Stage 1's phase B `arm_C` reference, not its sha256 |
| 97 / 96 | the pin worktree not at the pin / dirty |
| 84 | inputs changed |
| 8 / 9 | the guard |

Stage 1 also refuses with:
- 91: Stage 0 not done, or `P.json` naming no registered rung;
- 88: an interrupted collect, or a run dir of unknown origin;
- 89: the host-RAM floor;
- 85: a collect failing its check.

`HOLD_DOSE` (81 on relaunch) stops it when E2's planned updates fall below C's snapshot grid.

**Launch.** Stage 0 runs after this record is merged, `stage0/STAGE0_GO` exists and a `pc0` line is written in a gap,
as above.
1. Check: `sha256sum ~/chess-artifacts/operations/rl-precooldown-start/stage0/launch_stage0.sh` must print
   `c09a90a17ec7a4dd79f6c5477635ee3f5f9087c34bcefc8e3fc4b3592a8db49e`.
2. Launch:
   `PC=~/chess-artifacts/operations/rl-precooldown-start/stage0; setsid nohup bash $PC/launch_stage0.sh >> $PC/stage0.log 2>&1 < /dev/null &`
3. Stop: `kill -TERM $(cat $PC/stage0.pid)`.

Stage 1 is the same with `stage1`, `STAGE1_GO`, a `pc1` line and sha256
`5145a2168a03b47bb410a2f1f5233f7e377f9093f1827ed806a00b4276b9754a`.

**Run both with `VF_PY`, `VF_TEST_MEMINFO`, `PC_BANK`, `VF_BANK`, `DF_BANK` and `DF_RSS_INTERVAL_S` unset.**

### Harness

**`~/chess-artifacts/operations/rl-precooldown-start/test/harness_pc.sh` [`5f6baa0f`]: 57 of 57 pass**
(`test/harness_pc.out` [`1497e23c`]).

**Setup.**
- A fake HOME with a copy of the bank.
- Read-only symlinks to the three real SL checkpoints (the sha256 pins are checked on the real bytes), to phase B's
  `arm_C` train templates and reference `metrics.json`, and to the pin worktree.
- A stub interpreter that runs every stdlib tool for real, and `test/fake_pc.py` [`73ad6fa3`] in place of each torch
  call. The fake collect gives E2 95 rows per game and P 100, so P's collect is the longer and the U_own < U_P / Co path
  is exercised.
- No torch. `CUDA_VISIBLE_DEVICES` is empty for every python.
- **One stated difference from a real launch:** the guard is a stub. The real guard's changes are tested separately.

**What it proves (beyond revision 1's refusal, Stage 0 and `pick_p.py` cases, all kept).**
- **`opt_check_pc.py` on phase B's real `arm_C`:** it passes with its own lr and imported multiplier, and fails with
  multiplier 20 or with lr 3e-5.
- **A non-LR optimizer field off phase B's** gives `HOLD_TRAIN_CHECK` after `train_C`, with
  `non_lr_fields_equal_phase_b_arm_C` false and nothing played.
- **The void on arm C.**
- **The stop branch** (C below D):
  - rate_P's look 1 says extend, so look 2 is played for C and D only;
  - rate_P equals G_C − G_D exactly, and is negative;
  - the gate fails with the registered reading;
  - E2 is never collected, and A and B never train or play;
  - a relaunch does nothing.
- **The full branch:**
  - the P half runs strictly before the E2 half, in `resume.log` order;
  - U_P, the grid and U_own follow the rule (E2 shorter, so U_own = the largest grid point ≤ E2's planned updates);
  - every train check is ok;
  - the realised lr per arm is correct;
  - C's snapshot at U_own is fitted, baked and played as Co;
  - own equals G_C − G_A exactly, and is positive;
  - **each of the 2×2's five contrasts equals its G combination exactly;**
  - the signs match what the biases imply;
  - the Stage 2 trigger fires, reading rate_P from the P half's final file;
  - own is resolved at look 1, so A gets no look 2;
  - no head-swap is played;
  - the collect monitors run;
  - the 16 / 1-sim reads are reported.
- **`contrast_pc.py` refusals:** the wrong parent, `rate_P` without C and D, a seat swap, and a non-final `--rate-p`.
- **18 fixtures of the bands, the gate and the trigger at their edges:** |x| = 4.99 / 5.0; +15 with LB ±0.1; 14.99;
  −15 with UB ±0.1; the look-2 rules; the gate at 0 and at −0.01; trigger, headroom and no-trigger.
- **Nothing is written** outside the fake bank.
- **The real guard's four cases.**

**The reviewer's mutant now dies, in both forms (each run on a scratch copy of the bank).**
- **`own_C_minus_A` computed as C − B** whenever B is present: `TOTAL PASS=56 WRONG=1`, the failing check being "the
  2×2's five contrasts each equal their G combination exactly" (`test/mutant_own_C_minus_B_guarded.out` [`d7464efe`]).
- **The literal mutant (always C − B):** `TOTAL PASS=48 WRONG=9`. The own read crashes, so the full branch fails
  (`test/mutant_own_C_minus_B_literal.out` [`908111f5`]).

No other mutant series, per the owner's rule.

### Not verifiable without torch (stated plainly)

These are left for the first unmutated GPU pass:
- **Loading E1 / M.** The arena loads them in Stage 0's `smoke0`; `simple_rl_loop.py collect` first loads them in
  `collect_P`. The review checked offline that their names, layout and fixed optimizer keys match kC.
- **The realised per-group lr** in the real trainer. The smoke's update-1 check voids the arm if it is wrong.
- **That the real `realised_optimizer` record for an SL-checkpoint init equals phase B's in every non-LR field.** The
  review found the saved groups identical offline. If anything differs, `HOLD_TRAIN_CHECK` stops the stage for the
  owner rather than reading a wrong arm.
- **Rows per game and throughput for P's actor.**
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
- **Codex's head-swap yardstick** is dropped in revision 2: it decides nothing and costs about 3.8 GPU-h.

## Open decisions made on the owner's behalf

1. **P rule threshold −75 Elo**, highest rate first, with M as the expected pick. It only gates E1.
2. **27,955 games per start** (phase A's dose), not 8,000. From E2 that is ≈ 2.66M rows and U ≈ 4,940, just under
   the 2.8M of the data-volume and phase B steps.
3. **Fresh E2 collection** instead of the banked pilot rows.
4. **No carried-moments fifth arm.**
5. **Recalibration applied, its bounds reported not gated.**
6. **Exposure equalised per contrast:** U_P for rate_P, U_own for own (via C's snapshot grid and Co).
7. **(Coordinator, revision 2) The P half first, gated on rate_P ≥ 0.** The head-swap is dropped. Extension follows
   only the deciding contrast. The Stage 2 trigger is own LB > 0 and rate_P ≥ 0.
8. **16 / 1-sim reads at 800 per seat** (reported only).
9. **Step 356,239 is not laddered.**
10. **Seeds:** 2026112801 (Stage 0), 2026112821 (collects and training), 2026112811 / 2026112812 (Stage 1 looks).
