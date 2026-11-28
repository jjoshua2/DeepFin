# RL: start from before the SL learning-rate drop (Stage 0 read: P = M; Stage 1 launch-ready, not launched; Stage 2 design only)

Date: 2026-11-27. Status: registered (revision 2, after the non-author review of #1555), plus **Amendment 1
(Stage 1 on banked rows)** and **Amendment 2 (Stage 1 scores max-plies games as draws, the value follow-up's
rule)**.
- **Stage 0: DONE and read (2026-11-28): P = M** (step 475,738), 64-sim −44.6 [−60.2, −28.9] vs the SL base; E1
  −155.2 [−169.0, −141.4]. See "Stage 0 verdict".
- **Stage 1 (banked): launch-ready at launcher `9191300c…25a8`, NOT launched.** Per the guard it follows value
  follow-up Step 2 (relaunched 2026-11-28 00:55 EST).
- **Stage 0 (as registered; done)** launches when this record is merged, the coordinator writes `stage0/STAGE0_GO`, and a `pc0` release line
  is written in a gap when no value follow-up launcher is live. "Bank and launchers" says how the guard prevents
  overlap and how to avoid blocking Step 3. Amendment 1 does not change Stage 0, its launcher or its pins.
- **Stage 1 (Amendment 1)** trains all four arms on the banked rows of deep/fast phase B's `pool_C`, with no
  collection. It launches with `stage1/launch_stage1_banked.sh` after `stage0/STAGE0_DONE`,
  `stage1/banked/STAGE1B_GO` and a `pc1` release line. The collecting launcher `stage1/launch_stage1.sh` is
  **superseded**: it stays in the bank, unrun, and the new launcher refuses while its GO file or state exists.
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

## Stage 0 verdict (read 2026-11-28)

**Run.**
- Launched 2026-11-28 00:18:47 EST, launcher pid 3567884, launcher sha256
  `c09a90a17ec7a4dd79f6c5477635ee3f5f9087c34bcefc8e3fc4b3592a8db49e` (from `stage0/resume.log` and
  `stage0/inputs.txt`).
- Exit 0 at 00:54:02 EST, then `STAGE0_DONE` (`stage0/exit`).
- 35 min wall, against an estimate of ≈ 0.55 GPU-h. Each 64-sim seat of 600 games took 412–428 s, about 0.70 s per
  game; each 1-sim seat took 41–52 s.

**Settings, as registered.**
- Pin `d08b03adb`, `matched_sims`, temperature 0.1, no Gumbel noise.
- The pilot's `eval.pgn` (sha256 `067675c2…`), 16 opening plies, max plies 600 with abort.
- Seed 2026112801, and 600 games per seat (two seats, 1,200 games per rung) over the same 300 opening keys.
- **Every one of the 8 seats:** 600 games, `max_plies_unresolved` 0, `max_plies_aborts_banked` 0. Terminations were by
  rules or Syzygy only. The longest game was 252 plies (`lE164_1_A`).

**Per seat.** Seat A has the rung as the candidate and seat B has the SL base as the candidate. W / D / L is from the
candidate's side, counted from each seat's game log (the results rows carry score and pentanomial, not W/D/L).

| Seat | Sims | Candidate | W / D / L | Score |
|---|---|---|---|---|
| `lE164_1_A` | 64 | E1 | 80 / 186 / 334 | 0.288 |
| `lE164_1_B` | 64 | SL base | 349 / 184 / 67 | 0.735 |
| `lM64_1_A` | 64 | M | 141 / 235 / 224 | 0.431 |
| `lM64_1_B` | 64 | SL base | 223 / 225 / 152 | 0.559 |
| `lE11_1_A` | 1 | E1 | 410 / 62 / 128 | 0.735 |
| `lE11_1_B` | 1 | SL base | 142 / 58 / 400 | 0.285 |
| `lM1_1_A` | 1 | M | 213 / 25 / 362 | 0.376 |
| `lM1_1_B` | 1 | SL base | 361 / 33 / 206 | 0.629 |

**Ladder** (`contrast_v2.py --control none --base E2`, seat-balanced per opening key, K = 300, linear map at 694.87
Elo per unit).

| Rung | 64 sims (decides) | 1 sim (reported only) |
|---|---|---|
| E1 (step 340,168) | **−155.2 [−169.0, −141.4]** | +156.3 [+137.1, +175.6] |
| M (step 475,738) | **−44.6 [−60.2, −28.9]** | −88.0 [−108.0, −68.0] |

**The registered rule** (`pick_p.py`: the first rung, highest rate first, whose 64-sim point is ≥ −75) gives:
- **E1 fails** (−155.2 < −75);
- **M qualifies** (−44.6 ≥ −75);
- **P = M, not `beyond_threshold`** (`stage0/P.json`).

This was the prior's expected pick. Stage 1 uses M at its own saved rate, 7.4220002861554846e-06 (group 0 1.4844e-4).

On the logistic map, E1's excess (−0.2233) gives −167.0 [about −184.6, −150.1]: about 12 Elo larger in magnitude than
the linear −155.2. The record's earlier "about 8" was the figure at −139 linear. M reads −44.8 on the logistic map. The
pick is unchanged.

**What this establishes.**
- **A strength ladder against the SL base at 64 sims, on one seed.** E1 sits about 155 Elo below the SL base and M
  about 45 below it.
- **The P choice for Stage 1.**

**What it does not establish.**
- **Any learning-rate result.** No net was trained. The ladder says nothing about whether a higher rate learns more
  from RL rows: that is Stage 1's question, and only Stage 2 asks whether anything ends above the SL base.
- It does not show that M's deficit is the cost of its higher rate. M is also earlier in epoch 2's rows.

**Consistent with the banked figure.** "Epoch 2 beat epoch 1 by +153.44 [+123.98, +185.14] at 64 sims, 400 games"
(`~/chess-artifacts/operations/epoch2-vs-epoch1-fixed200-20261110-88e41e4ef670d523fc1951e2fbb83d8c/result.jsonl`
[`5d4ba512`]).
- That figure is cited in this record's "Why", but it had **no evidence-log row**. A row is added with this readout.
- **It ran a different instrument.** Read from its results row:

  | Setting | 11-10 match | Stage 0 |
  |---|---|---|
  | Move temperature | **0.0** | 0.1 |
  | Openings | **200 FEN seeds** (`openings_kind` fen, no opening plies) | the pilot's `eval.pgn` at 16 plies |
  | Prior temperature (`policy_temp`) | **1.5** | 1.0 |
  | Syzygy path | a `syzygy_3-4-5-6` + `syzygy_6` pair | the production pair |
  | Elo map | logistic (its interval is asymmetric) | linear |
  | Seed / git / max plies | 42 / `7775477b` / 20,000 | 2026112801 / `d08b03adb` / 600 |

- **Stage 0's E1 reading is consistent with it, not a replication on the same instrument.** On a common logistic map,
  E1 reads −167.0 [about −184.6, −150.1], against the banked +153.44 [+123.98, +185.14] for E2 over E1, so the
  intervals overlap. Stage 0 used 3× the games.

**Observation, unexplained, with no causal claim: the 1-sim ladder inverts for E1.**
- At 1 sim, E1 beats the SL base by +156.3, while M loses by −88.0. Ordered by step along the SL trajectory, the 1-sim
  reading is E1 +156, M −88, E2 0. E2's 0 is the reference by definition, not a reading. The sequence is non-monotone,
  and E1's sign is the opposite of its 64-sim sign.
- **It is not a seat or sign error.** E1 scores 0.735 as the candidate, and the SL base scores 0.285 as the candidate.
  The two seats agree.
- Loop nets showed a related pattern earlier (gains at 1 sim, none at 64; evidence log, "Loop nets vs SL base at 1 /
  4 / 16 / 64 sims"). That pattern is not shown to explain this one.
- **What a 1-sim score measures, traced in code. This is a fact about the instrument, not a cause.** The source is a
  read-only Codex trace at pin `d08b03adb` (gumbel.py, gumbel_c.py, sampling.py). No checkpoint, torch or GPU was run.
  - At 1 sim with no Gumbel noise, the arena visits only the top-prior move.
  - Every unvisited move gets the completed Q (v + q₁)/2, where v is the root value and q₁ the visited child's value.
    After min–max normalisation, if q₁ < v, every other move receives the full root bonus S and the visited move none.
  - The move is then sampled from softmax((log prior + bonus) / T).
  - **The scale S depends on the search shape.** The trace gives S ≈ 47.6 under the `play` shape defaults. Stage 0 ran
    `--search-shape training`, and its results rows record c_scale 0.1, c_visit 50, so S = 0.1 × (50 + 1) = **5.1**
    here.
  - Even at 5.1, the next move outscores the top-prior move whenever the top prior is less than e^5.1 ≈ 164 times the
    next one. At T = 0.1, play then usually goes to that unvisited move.
  - **So a 1-sim score at T = 0.1 largely measures this value veto, not policy quality.**
  - The evidence log has a related control. P1 c12's 1-sim gain over the SL base was +240.8 at T = 0.1 (Table 1, "Loop
    nets vs SL base at 1 sim") and +79.1 [+59.7, +98.9] at T = 0 (Table 1, "T1 / T0 1-sim (H1: 1-sim gain is a veto
    artefact)"), on the same opening pairs.
  - This does not identify which mechanism changed between E1, M and E2.
- **The read-only investigation of the 1-sim move rule continues.** Until it reports, the 1-sim column is reported only
  and decides nothing.

**Files** (full sha256):
- `stage0/P.json`: `c14be5ca9024bb74a1371d08c0e315e57bbbc6ac20611eea6d1e875ab61955a2`
- `stage0/ladder64.json`: `b3eef3205ec162284ec0f0f068c0fe26719a0d75182b969bb47315b5d8d83596`
- `stage0/ladder1.json`: `278ccbb894cb14d7e23b9d7b86ec79bf9eeef153df6f3ce617919ee83c2a3eb1`
- results rows:
  - `stage0/arenas/lE164/results.jsonl`: `819bfe2f…`
  - `lM64/results.jsonl`: `0e4ef730…`
  - `lE11/results.jsonl`: `b79a7902…`
  - `lM1/results.jsonl`: `69a3fecc…`

**Next.** Stage 1 (banked, Amendments 1–2) is launch-ready at launcher `9191300c…25a8` and **not launched**. Per the
guard, it follows value follow-up Step 2, which was relaunched at 00:55 EST: its `pc1` line is written only once no
value follow-up launcher holds the GPU.

## Amendment 1 (2026-11-27): Stage 1 trains on banked rows

**The owner's decision**, in his words: "I think it would he ok to reuse those game even if they are off policy its
not too far off and its more games and its closer than bt4 thar was sl". This amendment replaces revision 2's
collections. The arms, rates, recipe, gate, deciding contrasts, bands, Stage 2 trigger and match protocol are
unchanged. Where the revision-2 Stage 1 section below conflicts with this one, this one holds.

### What changes

- **No collection.** `collect_P`, `collect_E2`, the E2-half collection and their monitors are dropped.
- **All four arms train on the same banked rows,** in the same order, each row once:
  - from P: C at P's rate, D at the floor;
  - from the SL base E2: A at the floor, B at P's rate.
- **The rows are `~/chess-artifacts/operations/rl-deep-fast/stage2/phaseb/pool_C`,** read only and never moved:
  - 27,955 games and 2,785,627 rows, actor `kC_baked.pt` (`c48a7b2c…`), stock 64 sims (search share 1.0);
  - collect manifest sha256 `f5ba4cb33d4851c87372125003bbb47f80b579e0b28b97feafa6d197e3edc3e6`, pinned by the launcher;
  - 2,648,696 trainable rows after the 0.05 hold-out.
- **Every arm uses the pool's seed, 2026112703, and its 27,955 games.** The pin's reference check does not exempt `seed`
  from the settings comparison, and the same seed draws the batch plan. So every arm sees the rows in phase B arm_C's
  order, with phase B's `TRAIN` / `ZQ` (presentations 1, uniform shuffle, mirror probability 0.5, batch 512).
- **Every arm runs U = 5,174 updates,** phase B's planned count on these rows (`phaseb/plan_C.json`, sha256
  `4530e556…`, pinned). Snapshots are taken at phase B's 2,587 and 5,174. The deciding net is the snapshot at 5,174,
  recalibrated against the arm's own parent on the pool's hold-out games, then baked. **Exposure is equal by
  construction,** so revision 2's snapshot grid, U_own, `HOLD_DOSE` and the Co net are all dropped. `own` compares A
  with C directly.
- **Seeds 2026112821 (collects and training) are retired.** The look seeds 2026112811 / 2026112812 are unchanged.

### Which pool, and why not both

There were two candidate pools.
1. **`pool_C`** (above), played by kC_baked.
2. **The SL base's own rows:** the pilot collect `rl-run1-pilot/collect/gen_0001/collect`. It holds 8,000 games and
   761,979 rows, actor `a39f900c…`, pin `0c795fb7`, seed 2026112100.

**`pool_C` alone is used.** The reasons:
- **It is the dose on its own.** At 2.79M rows it meets the ≥ ~2.8M-row regime of the data-volume and phase B steps.
  The pilot adds 0.27 of a dose.
- **Mixing the two would give two actors an asymmetric role.** 22 % of a combined pool would be exactly on-policy for
  A and B, the E2 starts, and off-policy for C and D. That tilts `own` and the start contrast toward E2 by
  construction. With `pool_C` alone, every arm is off-policy and the confound is one stated quantity.
- **A combined pool would be a new artifact with one actor field for rows from two nets.** The phase B pooling tool
  refuses mixed actors, and the reference record carries one `inputs.checkpoint`. It would also join rows from two pins
  and two seeds.
- **Compatibility was checked and is not the obstacle.** The two collects' shard key sets are identical (every key,
  including `search_wdl` / `has_search_wdl`). Their settings differ only in seed, games and two keys the pilot predates
  (`root_full_width`, `eval_chunk_rows`).

**Arm A cannot be reused from the bank.** No banked net is "SL base, floor rate, these rows, this order, this recipe".
- The evidence log has no SL-base-init training on `pool_C`.
- The only SL-base floor-rate arm, the pilot's Zq, trained on the pilot rows at another pin.
- The one net trained at the floor on exactly these rows, in this order and with this recipe, is phase B's `C_baked`,
  from kC: −5.81 [−10.07, −1.55] against kC. That is a different start, and it is quoted only as context.

So A is retrained.

### The trainer path for off-policy rows (verified in source at the pin and on the real pool)

**`--collect-from` refuses these rows for every arm.** `reference.reference_collect` receives the arm's generation-0
model as both `gen0_model` and `actor_checkpoint`. It refuses rows played by any other net: "its rows would be
off-policy for this arm". That is the guard the owner waived here.

**The new tool `common/ref_banked_pc.py` calls the same pin function** with the pool's own sealed generation-0 model,
kC (the net that really played the rows), as both arguments.
- **Every other check runs unchanged:**
  - the source is sealed and not itself a reference;
  - every output sha256 is recomputed against its manifest;
  - the TRAIN openings' sha256 matches;
  - the collect settings equal the source's (`precheck_source`, with seed and games not exempt).
- **The sealed reference records the true actor** (`inputs.checkpoint` = kC) and the `source_*` inputs, which
  `lineage.verify_stage` re-hashes before every later stage.
- **Each arm then runs `simple_rl_loop.py run --init-checkpoint <start>` with no `--collect-from`.** It creates
  generation 0 from the start, verifies and skips the sealed collect, and trains from the start.
- **Nothing in the train path compares the collect's actor with generation 0.** Checked in `learn.py` and
  `simple_rl_loop.py`: the only actor check is the one in `reference_collect`.

**Verified on the real pool today, with no torch.** A stub `torch` module (version string only) was put first on
`PYTHONPATH`, with `CUDA_VISIBLE_DEVICES` empty, and the output went to a scratch arm dir.
- The tool sealed the reference with arm A's real flags, and the pin's `reference_collect` accepted.
- It recorded 282 source shard files, 2,785,627 rows and actor kC.
- `verify_stage(collect)` then accepted.
- `planned_phase_updates` / `phase_train_rows` gave **5,174 / 2,648,696**, equal to `plan_C.json`.
- Nothing was written into the pool.

The transcript is `PC/test/ref_banked_real_validation.txt` [`56c8b8f9`].

**Not verifiable without torch:**
- `planned_updates.py`'s own `train_settings`, which does a `torch.load` of the init checkpoint;
- `validate_before_pinning`;
- the run itself.

The launcher re-runs the planner per arm and writes `HOLD_PLAN` unless it gives 5,174 / 2,648,696 and phase B's
snapshot plan.

### The confound, stated plainly

**The rows are off-policy for every arm.** They were played by kC_baked, which reads +8.52 [+4.00, +13.05] against
the SL base at 64 sims (evidence log). kC is stronger than E2 and further still from P, which Stage 0 measures below E2.
- So Stage 1 now reads **"does the pre-cooldown start, or the higher rate, learn more from the same data"**.
- It is **not a full on-policy generation.** No arm's targets come from its own search.
- **The distance is not symmetric between starts.** It is larger for C and D (from P) than for A and B (from E2). Any
  start contrast (`own`, D − A) carries "how far the teacher's rows are from this start" alongside parent headroom.
- **rate_P (C − D) is free of this,** because both arms share the start, the rows and the order.
- A positive `own` here would not show that P's own collections would train as well.

### Contrasts, rules, power and GPU-hours under Amendment 1

**Unchanged:**
- the P half first and its gate (rate_P point estimate ≥ 0);
- rate_P = G_C − G_D deciding the P half, and own = G_C − G_A deciding the E2 half;
- the bands, the extension rule, the Stage 2 trigger and its stated limit, and the 16 / 1-sim reports.

Every G_X is still the arm's baked net against its own parent at 64 sims.

**Power:** the same per-arm ±8.5 Elo at 3,200 games, ±12 (MDE ≈ 17) for each contrast at look 1, and ±6 (MDE ≈ 8.5)
at 12,800 per arm. All four arms now share the rows and the order, so `own`'s arm-to-arm correlation may exceed 0 too.
Each readout reports the realised correlation.

**GPU-hours.** Measured basis as in revision 2: phase B `train_C` 33 min on this pool, which already includes the
reference's re-hash; smoke 5.5 min; fit about 5 min; 0.72 s per 64-sim game. The references and plans are CPU only.

| Item | P half | E2 half |
|---|---|---|
| Two smokes | 0.2 | 0.2 |
| Two trainings | 1.1 | 1.1 |
| Fits and bakes | 0.25 | 0.25 |
| 64 sims, look 1 (2 × 3,200) | 1.3 | 1.3 |
| 16 and 1 sims (2 × 1,600 each) | 0.25 | 0.25 |
| **Half, look 1** | **≈ 3.1 h** | **≈ 3.1 h** |
| Look 2 if triggered (2 × 9,600) | +3.8 | +3.8, or +1.9 if C's look 2 was already played |

| Branch | Stage 0 + Stage 1 | Worst case with every extension |
|---|---|---|
| **Gate fails** | 0.55 + 3.1 ≈ **3.7 GPU-h** | ≈ 7.5 |
| **Gate passes (full)** | 0.55 + 3.1 + 3.1 ≈ **6.8 GPU-h** | ≈ 12.5 |

Revision 2 was ≈ 8.7 / 17.0, or 12.6 / 25.4 with extensions. The ~10.2 GPU-h of the two collections is gone. Disk:
about 4 × 0.5 GB of snapshots plus the baked nets, and no collect.

### Steps (`stage1/launch_stage1_banked.sh`)

Per half (P half: C, D; then the gate; E2 half: A, B):
1. **`ref_<X>`.** `ref_banked_pc.py` seals `arm_<X>/gen_0001/collect` as the pool reference. Then a check: it is a
   reference to the pinned pool manifest, its actor and source generation 0 are kC, and its seed is the pool's. CPU
   only.
2. **`plan_<X>`.** The pin's planner on the arm's own flags must give 5,174 / 2,648,696 and phase B's snapshot plan,
   else `HOLD_PLAN`. CPU only.
3. **`smoke_<X>`.** The 3-update GPU memory smoke, reading the pool read-only as phase B did, and the update-1 lr void
   (`HOLD_LR_<X>`). Host-RAM floor 86 GiB.
4. **`train_<X>`.** `simple_rl_loop.py run` without `--collect-from`. Then the reference re-check and
   `common/opt_check_pc.py` against phase B's `arm_C` (`HOLD_TRAIN_CHECK`), with U = 5,174 and snapshots 2,587,5,174.
5. **`fit_<X>`, `bake_<X>`.** The snapshot at 5,174, against the arm's parent, on the pool collect's hold-out games
   (`--collect-dir pool_C/gen_0001/collect/ --holdout-frac 0.05`).
6. **`msmoke_<X>`, `f64_<X>`, the read, `x64_*` (iff extend), `f16` / `f1`, `readrep*`, `read2x2`.** As in revision 2,
   with outputs under `stage1/banked/`.
7. **`verdict.json`, then `STAGE1B_DONE`.**

**The ONE deciding yardstick, per half (exact commands).** B=`~/chess-artifacts/operations/rl-precooldown-start`,
S=`$B/stage1/banked/arenas`, E2 and P the checkpoint paths above.

P half (`readP_1`):
```
python3 $B/common/contrast_pc.py 64 --decide rate_P --out $B/stage1/banked/contrast_rateP_looks1.json --look 2026112811 \
  C=$S/fC64/fC64_1_A.games.jsonl,$S/fC64/fC64_1_B.games.jsonl,<P> \
  D=$S/fD64/fD64_1_A.games.jsonl,$S/fD64/fD64_1_B.games.jsonl,<P>
```
E2 half (`readE_1`):
```
python3 $B/common/contrast_pc.py 64 --decide own --rate-p <the P half's final file> \
  --out $B/stage1/banked/contrast_own_looks1.json --look 2026112811 \
  A=$S/fA64/fA64_1_A.games.jsonl,$S/fA64/fA64_1_B.games.jsonl,<E2> \
  C=$S/fC64/fC64_1_A.games.jsonl,$S/fC64/fC64_1_B.games.jsonl,<P>
```
Look 2 adds `--look 2026112812` with the `_2_` logs, written to `*_looks12.json`.

### Launch (Amendment 1)

1. **Check:** `sha256sum ~/chess-artifacts/operations/rl-precooldown-start/stage1/launch_stage1_banked.sh` must print
   `9191300c32d11f2c7e6c8fa3f94e36ec23e05b13a26f166b40a957e11f1625a8` (Amendment 2; Amendment 1's file was
   `2e76e133…`).
2. **Prerequisites:**
   - `stage0/STAGE0_DONE` and `stage0/P.json` exist;
   - `stage1/banked/STAGE1B_GO` is written by the coordinator;
   - a `pc1 idle_mib=<MiB> <ISO time>` line is in the bank's `GPU_RELEASED_BY_OPERATOR`, written only after value
     follow-up Step 3 completes or is cancelled;
   - `stage1/STAGE1_GO` is never written.
3. **Launch:**
   `PC=~/chess-artifacts/operations/rl-precooldown-start/stage1; setsid nohup bash $PC/launch_stage1_banked.sh >> $PC/banked/stage1b.log 2>&1 < /dev/null &`
4. **Stop:** `kill -TERM $(cat ~/chess-artifacts/operations/rl-precooldown-start/stage1/banked/stage1b.pid)`, then
   check `ps` for leftovers naming the bank.

Run it with `VF_PY`, `VF_TEST_MEMINFO`, `PC_BANK`, `VF_BANK`, `DF_BANK` unset.

**Refusals.** Stage 0's codes apply (90 no GO, 93 sums or guard, 94, 82, 81, 84, 97 / 96, 8 / 9), plus:
- 91: Stage 0 not done, or `P.json` missing or naming no registered rung;
- 92: a checkpoint, the pool manifest, `plan_C.json` or phase B's `arm_C` reference not as pinned;
- **83: the superseded launcher's `stage1/STAGE1_GO`, `inputs.txt` or `stage1.pid` exists;**
- 89: the host-RAM floor;
- `HOLD_PLAN` / `HOLD_LR_<X>` / `HOLD_MEM` / `HOLD_TRAIN_CHECK` (81 on relaunch).

**Pins.**
- `SHA256SUMS` [`80965d6f`] is unchanged, so Stage 0's pin is untouched.
- The new runtime file is listed in **`SHA256SUMS.banked`** [`a6df8f67`], which the new launcher pins:
  `common/ref_banked_pc.py` [`4c79fb3c`].
- `SHA256SUMS.launch` (`ffc121ca` at Amendment 1; re-made by Amendment 2, see there) adds the new launcher, harness,
  outputs and validation transcript.

### Harness (Amendment 1)

**`test/harness_banked.sh` (Amendment 1's file, `05f27bce`): 33 of 33 pass** (its output `ec61a9e3`).

Both files were later replaced by Amendment 2's versions, which pass 40 of 40 and are stamped there. Amendment 2's re-run
overwrote the Amendment 1 output file, so `ec61a9e3` is no longer in the bank.

**Setup.** The same construction as `harness_pc.sh`:
- a fake HOME with a copy of the bank, a stub guard and a stub interpreter;
- `test/fake_pc.py` (`8d0f2928` at Amendment 1; Amendment 2's file is stamped there), now also standing in for
  `ref_banked_pc.py` and for a `run` without
  `--collect-from`;
- read-only symlinks to the real checkpoints, the pin worktree, phase B's `plan_C.json` and `arm_C` `metrics.json`,
  and **the real `pool_C`**.

**What it proves:**
- **Every refusal above,** with nothing run during them.
- **`HOLD_PLAN`** when the planned count is not 5,174, with nothing smoked, trained or played.
- **The stop branch:** A and B are never referenced, trained or played, there is no collect, and rate_P = G_C − G_D
  exactly.
- **The full branch:**
  - there is no collect;
  - the P half runs strictly before the E2 half;
  - every arm's collect is a reference to the real pool manifest, with actor kC and seed 2026112703;
  - every arm's `run` argv has no `--collect-from`, the pool's seed and games, its own start, mirror 0.5,
    presentations 1, its rate and multiplier 20, and snapshots 2587,5174;
  - every arm planned and ran 5,174 updates over 2,648,696 rows, and every train check is ok;
  - the realised lr per arm is correct;
  - each fit uses the snapshot at 5,174, against the arm's own parent, on the pool's collect;
  - own = G_C − G_A exactly, with no Co;
  - the 2×2's five identities hold;
  - the Stage 2 trigger fires;
  - the 16 / 1-sim reports are produced.
- **Relaunch is idempotent,** and the superseded launcher left no state.
- **Nothing is written** outside the fake bank or into the real pool: no file under it is newer than the harness start,
  and its manifest sha256 is unchanged.

**The original harness re-run after the `fake_pc.py` changes:** 57 of 57, with output byte-identical (`harness_pc.out`
still [`1497e23c`]).

**Mutant, run on a scratch copy of the bank.** Each arm's `--seed $POOL_SEED` was replaced with revision 2's
`--seed $SEED_COLLECT`, a realistic slip that would silently change the row order.
- **Result:** `TOTAL PASS=18 WRONG=15`. The launcher stops at `ref_C`'s `seed_is_the_pool_seed` check, so no arm
  trains (`test/mutant_banked_seed_SEED_COLLECT.out` [`0f9d16e2`]).
- In a real launch, the pin's `precheck_source` would also refuse it, because seed is not exempt.

## Amendment 2 (2026-11-28): max-plies games are scored as draws in Stage 1

**Why.** Value follow-up Step 2 exited 1 after 2 h of a 4,800-game seat. One game reached `--max-plies 600` unresolved,
and the seat ran under `--on-max-plies abort`. This bank's `common/run_v2_match.sh` carries the same flag, and Stage 1
plays up to about 40k games. Amendment 2 ports **the value follow-up's own rule, unchanged** (its record's
"Amendment 1: max-plies draws", PR #1557, under review when this was written). If that review changes the rule, this
amendment follows it in an addendum.

**The rule.**
- **Every Stage 1 seat** is played by **`common/run_v3_match.sh`** [`68856fbc`], the value follow-up's file byte for
  byte. It differs from `run_v2_match.sh` in two ways:
  - **`--on-max-plies draw`.** A game that reaches 600 plies unresolved, by rules or Syzygy, is written and scored as
    a draw with termination `max_plies`. The pin's arena counts it in the results row's `max_plies_unresolved`.
  - **It refuses (95) a seat that has an abort sidecar (`.games.jsonl.unresolved.jsonl`) beside it.**
- **Every Stage 1 read** goes through **`common/contrast_pc_mp.py`** [`f241050a`], which is `contrast_pc.py` with the
  value follow-up's amended reader. That reader, its `VF/common23/contrast_v2.py` [`ab514cae`], is copied byte for byte as
  `common/contrast_v2mp.py`. It is renamed because the unamended `contrast_v2.py` is pinned by Stage 0.
  - The header comparison leaves out `on_max_plies`.
  - Each run's results-row policy must be abort or draw and equal its header's.
  - The log's `max_plies` games must equal the row's `max_plies_unresolved`, which must be 0 under abort.
  - `max_plies_aborts_banked` must be 0.
  - Every run's `{on_max_plies, max_plies_draws, games}` is reported under `max_plies`.
- **The bound.** If any seat's max-plies draws exceed **0.25 % of its games**, the read is written to `OUT.HOLD.json`
  instead of `OUT` and exits 3. The launcher then writes `HOLD_MAXPLIES`, and the owner decides.
  - The bound is 4 of a 1,600-game look-1 seat, 12 of a 4,800-game look-2 seat, and 2 of an 800-game 16- or 1-sim
    seat.
- **Wiring.** `PC/common/lib_pc_mp.sh` [`cb5870bd`] is sourced after `lib_pc.sh`. It replaces `seat()` (`run_v3_match.sh`,
  and the PARTIAL message names the sidecar) and adds `read_c`, which maps exit 3 to `HOLD_MAXPLIES`. Nothing else in
  the launcher changed.

**The size.** One max-plies draw moves its seat's E by at most 0.5 / games × 694.87 Elo, and the seat-balanced arm
estimate by half that.

| Seat size | One draw moves the seat's E by at most | At the bound |
|---|---|---|
| 4,800 games | 0.07 Elo | 12 draws, ≤ 0.87 Elo on the seat, ≤ 0.43 on the arm |
| 1,600 games | 0.22 Elo | 4 draws, ≤ 0.87 Elo on the seat, ≤ 0.43 on the arm |

Both are far inside the ±8.5 / ±4.3-Elo per-arm half-widths. Above the bound, the games are not a rare tail and the
owner decides.

**Stage 0 is unchanged and stays on `abort`, as registered.** It was live when this amendment was written. No file in
`PC/SHA256SUMS` [`80965d6f`] and nothing `launch_stage0.sh` uses was modified.
- **Stage 1 reads no Stage 0 seat.** It takes only `stage0/P.json`.
- **The amended reader still accepts abort seats with 0 unresolved** together with draw seats. The harness checks this
  on a mixed look.
- **If a Stage 0 seat aborts on a max-plies game, the registered recovery** is the PARTIAL path (87):
  1. The owner moves that seat's four files, plus its `.unresolved.jsonl` sidecar, into a new `stage0/aborted_*` dir.
  2. The owner relaunches `launch_stage0.sh` unchanged. That replays the seat at the same seed and can hit the same
     game again.

  Running Stage 0's seats with `draw` would need its own addendum. Nothing here changes Stage 0.

**Pins.**
- **New launcher sha256: `9191300c32d11f2c7e6c8fa3f94e36ec23e05b13a26f166b40a957e11f1625a8`.** Amendment 1's launch
  check now expects this value. The only change from Amendment 1's `2e76e133…` is the `SHA256SUMS.banked` pin, the
  `lib_pc_mp.sh` source line, `read_c` in place of `contrast_pc.py`, and the header comment.
- **`PC/SHA256SUMS.banked` [`cf926c1c`]** lists `ref_banked_pc.py`, `run_v3_match.sh`, `contrast_v2mp.py`,
  `contrast_pc_mp.py` and `lib_pc_mp.sh`.
- `SHA256SUMS` is unchanged. `SHA256SUMS.launch` is now [`e810ad0d`].

**Launch and stop are unchanged:**
- launch: `PC=~/chess-artifacts/operations/rl-precooldown-start/stage1; setsid nohup bash $PC/launch_stage1_banked.sh >> $PC/banked/stage1b.log 2>&1 < /dev/null &`
- stop: `kill -TERM $(cat ~/chess-artifacts/operations/rl-precooldown-start/stage1/banked/stage1b.pid)`

The new refusal is `HOLD_MAXPLIES` (81 on relaunch).

**Harness.** `PC/test/harness_banked.sh` [`98704f69`] passes 40 of 40: Amendment 1's 33, plus 7 new checks.
- **New checks:**
  - **Over the bound:** 5 max-plies draws in a 1,600-game seat give exit 3, `HOLD_MAXPLIES`, and `OUT.HOLD.json` naming
    that seat. No `OUT` is written and the E2 half never starts. A relaunch is refused with 81.
  - **Every Stage 1 results row records `on_max_plies` = draw,** across every look and sims.
  - **Draws are read and counted per seat.** 4 draws is exactly the bound and does not hold. 1 draw is counted in the
    `own` read; the other seats read 0.
  - **A mixed look reads.** An abort seat pair with 0 unresolved, Stage 0's form, is read with draw seats, and each
    run's policy is reported.
  - **A row/log count mismatch is refused** (2).
  - **`run_v3_match.sh` refuses a seat with an abort sidecar** (95) and plays nothing.
- **Harness fixes:**
  - `PC/test/fake_pc.py` [`ade63d5d`]'s arena now honours `--on-max-plies`. `FAKE_MAXPLY` makes games of a named seat
    reach max plies: they become draws under draw, and under abort the run writes a sidecar and exits 1.
  - Both harnesses now copy only the launchers, never a stage dir's run state, because the real Stage 0 was live while
    they ran.
- **The original `PC/test/harness_pc.sh` [`fc6f83f4`]** was re-run after these changes: 57 of 57, with output byte-identical
  (`harness_pc.out` still [`1497e23c`]). `test/harness_banked.out` is [`9a183e6b`].

**Mutant: the draw flag ignored.** `run_v3_match.sh`'s `--on-max-plies draw` was set back to `abort` on a scratch copy
of the bank, with `SHA256SUMS.banked` and its pin re-made in the copy.
- **Result:** `TOTAL PASS=22 WRONG=18` (`PC/test/mutant_banked_maxplies_abort.out` [`d707ce19`]). `fC64_1_A`, which has 4 max-plies
  games, aborts, so the P half never reads. The results-row check reads `False`.

**Not verifiable without the GPU:**
- the pin arena's real `max_plies_unresolved` / `max_plies_aborts_banked` / header `on_max_plies` on a real game. The
  fields were read in the pin's `scripts/arena_standard.py`, and the value follow-up's Step 2 reruns exercise them
  first;
- how often 600 plies is reached at these arms' strength.

## Stage 1 as registered in revision 2 (SUPERSEDED by Amendment 1: fresh collections; kept for the record, not run)

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

  Generation 1 is Stage 1's arm C, reused. Under Amendment 1 arm C trained on kC's banked rows, not its own, so the
  chain's first generation is off-policy and generations 2 onward are on-policy. The Stage 2 addendum states this.
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
9. **(Amendment 1) Off-policy rows.** Every arm trains on rows played by kC_baked, a stronger net than either start.
   The distance is larger from P than from E2. Start contrasts carry it, and rate_P does not (see "The confound,
   stated plainly"). Under Amendment 1, items 2, 3 (the collect) and 5 no longer apply: one pool and one hold-out serve
   every arm, at equal U.

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
- **`stage1/STOP`** pauses only inside the trainer (`--pause-marker`). The launcher refuses while it exists. Under
  Amendment 1 the files are `stage1/banked/stage1b.pid` and `stage1/banked/STOP`.

## Bank and launchers

Every runtime file is listed in `~/chess-artifacts/operations/rl-precooldown-start/SHA256SUMS` [`80965d6f`]. Both
launchers pin that file's sha256 and the guard's. Launchers, tests and outputs are listed in
`PC/SHA256SUMS.launch` [`e810ad0d`] (`a1e836e0` at revision 2).

| File | sha256 | Origin |
|---|---|---|
| `stage0/launch_stage0.sh` | `c09a90a17ec7a4dd79f6c5477635ee3f5f9087c34bcefc8e3fc4b3592a8db49e` | new: Step 2's launcher structure (only the `SHA256SUMS` pin changed in revision 2) |
| `stage1/launch_stage1.sh` | `5145a2168a03b47bb410a2f1f5233f7e377f9093f1827ed806a00b4276b9754a` | revision 2: P half, gate, E2 half, fresh collections. **SUPERSEDED by Amendment 1; kept, never to be run** |
| `stage1/launch_stage1_banked.sh` | `2e76e13351f53df655b3a71232252ff04f0febda65fa5ddd184a02b012130511` | Amendment 1: the same halves on the banked `pool_C`; pins `SHA256SUMS` and `SHA256SUMS.banked` |
| `common/ref_banked_pc.py` | `4c79fb3c…` (in `SHA256SUMS.banked`) | Amendment 1: the pin's `reference_collect` with the pool's own actor |
| `common/opt_check_pc.py` | `e4210b43…` | derived from `VF/common23/opt_check_v.py` [`1c75c174`], changes in its header |
| `common/contrast_pc.py` | `43bf5557…` | revision 2: per-half deciding contrast, gate, Stage 2 trigger; imports `contrast_v2.py` / `contrast_v.py` unchanged |
| `common/gpu_holders.py` | `9991af8d…` | phase A's, byte for byte |
| `common/gpu_guard_pc.sh` | `0bfcab5e…` | the value follow-up's `VF/common23/gpu_guard_v23.sh` [`99f51eb9`] with three stated changes |
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

Stage 1: **use Amendment 1's launch section** (`stage1/launch_stage1_banked.sh`, `stage1/banked/STAGE1B_GO`, a `pc1`
line). Revision 2's `stage1/launch_stage1.sh` with `stage1/STAGE1_GO` is superseded, and `STAGE1_GO` must not be
written.

**Run both with `VF_PY`, `VF_TEST_MEMINFO`, `PC_BANK`, `VF_BANK`, `DF_BANK` and `DF_RSS_INTERVAL_S` unset.**

### Harness

**`PC/test/harness_pc.sh` [`fc6f83f4`]: 57 of 57 pass** (`PC/test/harness_pc.out` [`1497e23c`]). This is the
Amendment 2 file. Revision 2's file was `5f6baa0f`; the only change since is copying launchers instead of stage dirs.
The output is byte-identical.

**Setup.**
- A fake HOME with a copy of the bank.
- Read-only symlinks to the three real SL checkpoints (the sha256 pins are checked on the real bytes), to phase B's
  `arm_C` train templates and reference `metrics.json`, and to the pin worktree.
- A stub interpreter that runs every stdlib tool for real, and `test/fake_pc.py` (`73ad6fa3` at revision 2) in place of each torch
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
  2×2's five contrasts each equal their G combination exactly" (`PC/test/mutant_own_C_minus_B_guarded.out` [`d7464efe`]).
- **The literal mutant (always C − B):** `TOTAL PASS=48 WRONG=9`. The own read crashes, so the full branch fails
  (`PC/test/mutant_own_C_minus_B_literal.out` [`908111f5`]).

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
3. ~~Fresh E2 collection instead of the banked pilot rows.~~ **Replaced by the owner's decision (Amendment 1):** no
   collection. All four arms train on the banked `pool_C`, and the pilot rows are not mixed in (reasons in Amendment
   1).
4. **No carried-moments fifth arm.**
5. **Recalibration applied, its bounds reported not gated.**
6. **Exposure equalised per contrast:** U_P for rate_P, U_own for own (via C's snapshot grid and Co).
7. **(Coordinator, revision 2) The P half first, gated on rate_P ≥ 0.** The head-swap is dropped. Extension follows
   only the deciding contrast. The Stage 2 trigger is own LB > 0 and rate_P ≥ 0.
8. **16 / 1-sim reads at 800 per seat** (reported only).
9. **Step 356,239 is not laddered.**
10. **Seeds:** 2026112801 (Stage 0), 2026112821 (collects and training; retired by Amendment 1, which trains at the
    pool's 2026112703), 2026112811 / 2026112812 (Stage 1 looks).
11. **(Amendment 1) Equal U = 5,174 for every arm** (phase B's plan on these rows), with snapshots at phase B's 2,587 /
    5,174.
