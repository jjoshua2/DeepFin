# RL run 1 amendment: count games, not rows

**Status:** merged (#1452, `d3be0f4e1`). This is the registered rule set the pilot was judged by: the pilot (Step 1)
launched and was read on 2026-11-21 ([results](2026-11-21-rl-run1-pilot-results.md)). Step 2 onward awaits the owner's
decisions ([below](#decisions-for-the-owner)). When this record was written, nothing had been launched. Code base: main at 75d063c97, where #1447
(`--accum-steps`, `--snapshot-rows`) and #1446 (offline retarget) are merged; #1441,
#1444 and #1445 are open. This amends
[RL run 1](2026-11-20-rl-run1-multi-generation-plan.md): it replaces that plan's
Stage 1 (four policy-target cells on one ~100k-row collection) with a **pilot that
asks whether the value loss is shape, whether a soft value target avoids it, and
whether the policy pays at 64 sims with the value held exactly**, followed by **one
large generation read as a dose-response curve whose x-axis is games**. The
independent review of this record (PR #1452, evidence in `plan-review-1452/`) is
accepted; its findings are folded in below. The BT4 / Ceres / SF ruler columns stay, as report-only columns.
Neither raw ruler validated (the plan's positive control fails on BT4 raw and C3
raw, `rl-run1-prereq/a1_poscontrol_score.json`); the offline policy ruler is the
deep-SF regret on the leak-free rows, and value has no offline ruler (matches
only). This PR also corrects plan and takeover-record statements that later banked
evidence overtook ([below](#corrections-to-the-plan-and-the-takeover-record)).

Every number below is either read from a named bank file, derived (the arithmetic is
shown), or marked **guess**. Bank paths are relative to
`~/chess-artifacts/operations/rl-claude-20261119/` unless written in full.

## Why the plan changes

### What the earlier loop runs actually trained on

- **F12, 12 cycles: 2,304 games, 241,125 rows, 104.7 rows per game.** Counted from
  the 12 learners' draw journals
  (`~/chess-artifacts/operations/rl-week-one-20261112-v1/fresh-current-policy-A39-F12-cycle-NN-learn-results-v1/learner/draws.jsonl`,
  distinct `source_sha256` = games, distinct `(source_sha256, row_index)` = rows;
  each row was drawn once). Q50 and P1 are the same scale.
- Every row of a game carries that game's ONE outcome. With the outcome in the value
  target, each of the 2,304 outcomes was fitted about 105 times, from 105 strongly
  correlated positions. **The value target's outcome information is bounded by the
  number of games, whatever the row count.** For the outcome component that bound is
  exact: the label noise is one draw per game.
- **Effective sample size**, `n_eff = G · r / (1 + (r − 1) · ICC)` for G games and
  r rows per game, with the within-game ICC of each target's residual against the
  base (`search-compose/rows_per_game.json`, `search-compose/logs/rows_per_game.txt`,
  F12's rows):

  | Target residual | ICC | Design effect at 104.7 rows/game | n_eff of 241,125 rows |
  |---|---|---|---|
  | game outcome z | 0.600 | 63.2 | 3,813 (1.7 per game) |
  | F12's value target (0.9 outcome) | 0.593 | 62.4 | 3,862 |
  | Q50's value target (0.5 outcome) | 0.514 | 54.3 | 4,440 |
  | stored search WDL | 0.063 | 7.5 | 32,194 |
  | SF 8M (reference) | 0.134 | 14.8 | 16,245 |
  | policy KL(stored ‖ base prior) | 0.030 | 4.1 | 58,306 |
  | policy argmax ≠ base argmax | 0.004 | 1.4 | 175,433 |

  Derived from the same formula: with a z-heavy value target, **2 rows per game
  carry 76% and 4 rows 86%** of the value information all 104.7 rows carry
  (2 / (1 + 0.6) = 1.25 and 4 / (1 + 3 × 0.6) = 1.43, against 104.7 / 63.2 = 1.66
  per game). The search-WDL residual decorrelates within ~6–10 plies (pairwise
  correlation 0.48 at 1–2 plies, 0.10 at 6–10, 0.07 at 11–20), which supports about
  one value row per 8 plies (~12 per game). **Policy rows are nearly independent:
  keep every full-search row for policy.**

### The damage is not in the value's level: the leading hypothesis is WDL shape

From `search-compose/logs/rows_per_game.txt`, `search-compose/wdl_shape.json`,
`headswap/table.md` and `headswap/README.md`:

- On their own training games the trained heads did NOT overfit the outcome in
  level: their pull toward SF (F12 c01 +0.275, F12 c12 +0.375, Q50 c12 +0.491) is
  4–7× their pull toward z (+0.065, +0.066, +0.075), and their level error vs SF
  fell in every split. This argues against the variance mechanism as well.
- No offline value metric in the bank saw the 64-sim damage (chooser-regret nulls,
  level closer to SF; `value_prescreen.json` is itself *interim*, evidence log
  #1451).
- **The strongest shape evidence: recalibrating the FULL net.** Two parameters
  (temperature and draw bias on the WDL logits, fitted to the base,
  `value-recalibration/recalibration_params.json` `ref=base|temp_drawbias|fitCE`:
  T 0.635 / 0.545, b −0.59 / −0.64) take **F12 c12 −34.9 → −0.9 [−19.0, +17.2]**
  and **Q50 c12 −63.2 → −14.3 [−32.6, +3.9]** (`table.md` rows R4 / R3). Paired on
  the same openings: +34.0 [+8.4, +60.0] (recovers 0.97 [0.37, 1.56] of the
  deficit) and +48.5 [+25.3, +72.2] (0.77 [0.47, 1.06]) (`headswap/README.md`).
  On the value half alone, recalibration recovers most of Q50's loss (−55.2 → −8.2)
  but only about a third of F12's (−18.3 → −12.2). The two parameters fold exactly
  into the WDL head's last layer, so recalibration is also a loop lever with no
  runtime change.
- **The SL base was trained on a fully soft value target**: `search_wdl_frac 1.0`,
  `sf_wdl_frac 0.0`, 174,053,867 rows, batch 512, saved at optimizer step 680,276
  (the SL base's step), per
  `~/chess-artifacts/operations/all-accumulated-174m-epoch2-sqrt-cooldown-20261109-v1.resume-recovery654213-a4349851abddea4a/training/RESULT.json`
  (`identity.recipe`). No one-hot outcomes. F12 and Q50 switched to 90% and 50%
  one-hot z.
- **What that did to the head's shape** (`wdl_shape.json`, 2,433 rows): the heads
  got flatter (WDL entropy base 0.416 → F12 0.527, Q50 0.586: +0.11 / +0.17);
  |W − L| was compressed (0.440 → 0.378 / 0.341: −14% / −23%); draw mass moved
  toward z's conditional profile (|Q_base| > 0.3: base 0.192, F12 0.268, Q50 0.283,
  z 0.242); and the top-2 child-value gap shrank (mean 0.142 → 0.123 / 0.112;
  median 0.0386 → 0.0294 / 0.0288). Recalibration restores entropy and |W − L|
  (0.427 / 0.436; 0.424 / 0.397) but NOT the top-2 gap median
  (`child_recal.json`: F12 0.0291 → 0.0299, Q50 0.0299 → 0.0273; that file and
  `wdl_shape.json` disagree on Q50's raw median, 0.0299 vs 0.0288).
- **The stored 64-sim search WDL is NOT shape-preserving.** It is flatter than the
  base in the damage direction: entropy 0.450 vs 0.416 (+0.034, about 3/10 of F12's
  shift) and |W − L| 0.402 vs 0.440 (−0.038, about 6/10 of F12's). Only the draw
  mass matches: row by row it is the base's own to within 0.0013 on average (max
  0.0135); the mean W − L shift is +0.0003 and the mean absolute shift 0.085, with
  |W − L| shrinking on 50.5% of rows (`plan-review-1452/dcheck.txt`). So s = 1.0 is
  self-distillation in D plus a per-row W − L change with no net bias, and a |W − L|
  compression in the damage direction.
- **The bank's dose ordering points against a search-heavy target.** Q50 (half
  search WDL) lost MORE than F12 (a tenth): value half −55.2 vs −18.3, full net
  −63.2 vs −34.9, with learner boundaries identical except `outcome_wdl_frac` 0.5 vs
  0.9 (the review checked them). The value-recalibration README (line 41) also finds
  the base over-confident against the 800-sim survivor Q, so a "truer" target is
  flatter, and the 64-sim search has so far penalised flatter. **S's expected sign
  is unknown, possibly negative.**
- Every root-Q column is biased about −0.10 against deep SF at every sim count
  (R64 −0.102 … R800n −0.110, `value_prescreen.json`), which points to a convention
  artefact; until explained, no "deeper root value" target can be priced from the
  bank.

### Head-swap: trained policy on the base's value reads about 0

From `headswap/table.md` (64 sims, seat-balanced, against the SL base):

| Arm | Games | Elo [95% CI] |
|---|---|---|
| Q50 c12, full net | 800 | −63.2 [−81.1, −45.7] |
| Q50 policy + base value | 800 | +4.3 [−12.5, +21.2] |
| base policy + Q50 value | 800 | −55.2 [−73.6, −37.1] |
| F12 c12, full net | 800 | −34.9 [−53.1, −16.8] |
| F12 policy + base value | 800 | −15.7 [−33.8, +2.5] |
| base policy + F12 value | 800 | −18.3 [−36.2, −0.4] |
| base policy + Q50 value, recalibrated | 800 | −8.2 [−26.1, +9.6] |
| base policy + F12 value, recalibrated | 800 | −12.2 [−29.7, +5.3] |

The two halves add up to the full net within error (`table.md`'s additive check:
Q50 −50.8 vs −63.2, F12 −33.9 vs −34.9). With P1 c12 at −3.9, three nets say a
policy-target cell at ~240k rows reads about 0 at 64 sims, and the losses sit in the
value head. These are the head-swap harness's numbers; the registered matches read
F12 c12 −20.0 [−38.3, −1.8] and Q50 c12 −53.9, and F12 at cycles 3 and 6 read about
−1 ± 26 at 400 games (evidence log #1451): the damage is not monotone in dose and
its size at a new recipe is not predictable from the bank.

### Value pre-screen: no trained value head is a better chooser

From `search-compose/value_prescreen.json` (2,433 rows, reference SF 8M nodes,
cp-logistic slope 0.006; script `search-compose/value_prescreen.py`):

- **1-ply chooser regret vs deep SF** (`value_as_chooser`, 828 rows): base 35.4 cp;
  F12 c1 −2.3 [−6.4, +0.9], F12 c6 −0.5 [−4.0, +2.8], F12 c12 −2.0 [−6.2, +1.9],
  Q50 c12 +1.6 [−2.2, +5.8] cp vs the base. **All nulls.**
- **Level agreement is anti-aligned with play.** MAE vs SF: base 0.0657; Q50 c12
  0.0538 (the closest of all) and F12 c12 0.0543, yet Q50 is the −55 Elo value and
  F12 the −18 one. Closer level does not mean better play.
- **P1's value moved with no value loss:** P1 c12's MAE is 0.0758, +0.0100
  [+0.0068, +0.0132] vs the base, though P1 trained policy only. The trunk carries
  value drift, so a policy-only arm still needs an explicit value anchor.
- **Which search value matters.** The STORED search WDL is closer to SF than the
  base (above). The 64-sim root-node Q (`rootQ_R64`, a different quantity) sits at
  MAE 0.1157, bias −0.102, farther than the base. The driver's default
  `--value-search-source survivor_q` (the halving survivor's completed Q) was not
  measured; the pilot uses `stored_search_wdl`, the one that was.

### T1: the raw policy did learn

T1 (registered in `search-compose/PREREG.md`, temperature 0, 1 sim, P1 c12 vs the
base, both orientations, 800 games each; `search-compose/results/pool_t0s1_P1.json`):
**+80.9 [+60.3, +102.1]** and **+77.3 [+56.7, +98.4]** (orientation B, sign
flipped); pooled **+79.1 [+59.7, +98.9]**, "SUPPORTED by the registered rule"
(`search-compose/PREREG.sha256.addendum`, addendum 5). The ~240k rows moved the
raw policy by about +79 Elo at 1 sim; the gain
does not survive 64-sim search with the trained value (above).

### The owner's objection, verbatim

> "More data wont really help though if its already hurting it with the few steps
> we have? Maybe we are way over sampling and should only sample like 2 per game?
> This is why katago uses fast moves and doesn't train them to increase game output"

and, on the optimizer: "Just using aurora instead of Adam won't fix our loop … If we
have the same bad training target it won't help."

## What KataGo and lc0 do (verified)

- **KataGo** (Wu, *Accelerating Self-Play Learning in Go*, arXiv 1902.10565v5,
  read at <https://arxiv.org/html/1902.10565v5>, §3.1 and §2):
  - "On a small proportion p of turns, we perform a full search, stopping when the
    tree reaches a cap of N nodes, and for all other turns we perform a fast search
    with a much smaller cap of n<N."
  - "For KataGo's main 19-day run, we chose p=0.25 and (N,n)=(600,100) initially,
    annealing up to (1000,200) after the first two days of training."
  - "Only turns with a full search are recorded for training."
  - "Because most moves use a fast search, more games are played, improving value
    training." and "For fast searches, we also disable Dirichlet noise and other
    explorative settings, maximizing strength."
  - §2: "generated about 241 million training samples across 4.2 million games"
    (about 57 samples per game, derived).
  - Not verified: whether the paper discusses within-game correlation (the second
    half of the HTML page was not read verbatim).
- **lc0** (`LeelaChessZero/lczero-training`, branch `master`, read from
  raw.githubusercontent.com on 2026-11-21):
  - `tf/train.py` line 29: `SKIP = 32`, passed as `sample=SKIP` to the training
    `ChunkParser` (lines 162, 174).
  - `tf/chunkparser.py`: "sample_record() also skips most training records to avoid
    sampling over-correlated positions since they typically are from sequential
    positions in a game." and, in `sample_record`: "Downsample, using only 1/Nth of
    the items." (`if random.randint(0, self.sample - 1) != 0`).
  - So lc0 trains on about 1 in 32 positions, for the stated reason of within-game
    correlation. How often a kept record is reused across the training window was
    not checked.

Both projects train far fewer rows per game than we did (KataGo ~57 of its turns
recorded at p = 0.25; lc0 1 in 32), and KataGo spends the saved search on more
games. Our runs: every ply, ~105 rows per game.

## What the driver already has, and what is missing

| Need | On main (75d063c97) / open PRs | Status |
|---|---|---|
| Playout cap (full search on a share p of plies, fast elsewhere, fast plies store no row) | `--full-prob p`, `--fast-sims n`; `--fast-value-rows` keeps fast plies as value-only rows | **exists**; matches KataGo's "only full-search turns are recorded" |
| Fast plies without root noise (KataGo disables noise on fast searches) | not checked | **to check** in `collect.py` / the selfplay path before the pilot |
| Value rows per game r chosen at TRAIN time, policy on every row (the [variance follow-up](#follow-up-the-owners-2-per-game-idea-dose-matched)) | none: no per-game value-row subsampling | **missing**, follow-up only: `--value-rows-per-game r`, rows chosen by a hash of `game_id`, a per-row VALUE loss mask (the WDL term is masked by `is_network_turn` only today), recorded in the manifest |
| The pilot's recalibrated and grafted nets in a match | `headswap/headswap_arena.py` (bank scratch, monkeypatches the loader) | **the pilot's only prerequisite code**: promote a tested `--value-from CKPT` (policy from one net, value from another, two forwards) and `--value-recal T,b` option on `scripts/arena_standard.py`. Recalibration alone could instead be folded into a checkpoint's WDL head (exact), but the graft needs the hybrid |
| Outcome smoothed to the base's draw mass (a second shape-preserving value target) | none | **missing**; the pilot uses search WDL instead |
| Dose-response points at game counts | separate train runs, each on a game-prefix subset of the one collection | **no new schedule needed**: train each dose as its own run on the first G_i games (total ≈ 2× one full run). Independent runs also make the slope's independence assumption honest and complete the LR schedule at every dose. Selecting a game prefix needs a small train-side game filter (to check: `--collect-from` arms copy whole collects) |
| Window larger than RAM | the train window is loaded whole (`load_window`, `lazy=False`) | **blocker for the pilot on today's main**: the full `load_window` path peaks at ≈ 141.6 KiB per row (measured below), so the pilot's ≈ 758k rows need ≈ 102 GiB on a 98 GiB box (largest that fits in 88 GiB free: ≈ 0.65M rows). **The loader-RAM PR (in progress, collect agent) is a prerequisite**: it streams shards into preallocated arrays, ≈ 27.95 KiB per row + one decoded shard, so ≈ 20 GiB for the pilot and ≈ 3.3M rows in 88 GiB (≈ 1.9M with the actor prior mix, 49 KiB per row) |
| Multi-process collect, PAUSE marker | #1441 (open) | needed |
| Kill-safe resume | #1444 (open, stacked on #1441) | needed |
| SL base's Aurora split, its realised LRs | #1445 (open) | needed |
| Value blend | `--value-search-weight`, `--value-outcome-weight` (search share = s / (s + o)), `--value-search-source survivor_q` | exists |
| Policy sharpening guard | `--train-target-temp`, `--kl-policy` / `--kl-wdl` with `--kl-reference` | exists |
| SL-corpus share | `--sl-mix-dir`, `--sl-mix-frac`, `--sl-mix-max-rows` (default 1M, loaded in RAM) | exists; **to check**: an SL corpus in shard format with `search_wdl` when the value search weight > 0 (the loader refuses otherwise); 1M SL rows ≈ 28 GB RAM |
| Gradient accumulation | #1447 | not used by this design |
| Collect process shape for mostly-fast collects | #1441 picks the shape from `--sims` (3 × 128 above 16 sims) | **to measure** in the pilot: with p ≈ 0.02 most plies are 16-sim, where 8 × 64 was faster; pass `--collect-procs` / `--concurrency` explicitly if the VRAM estimate allows |

## Measured costs used below

- **Collect, 64 sims every ply:** 5,021 games/h at 3 × 128 (#1441, RTX 5090, SL base;
  re-measured 4,982 on head 347aaf08c); 95.0 rows per game
  (`collect-throughput/bench_c128`: 48,641 rows / 512 games; 94.8 in the re-measure).
- **Collect, 16 sims every ply:** 1.85M training rows/h at 8 × 64 (#1441); 96.0 rows
  per game (`bench_c256_s16_prof`: 49,158 / 512) → **≈ 19.3k games/h** (derived).
- **Collect, playout cap at 64 / 16 sims: measured** (3 × 128, 960 games per p, SL base, the pilot's flags;
  `collect-throughput/playout-cap/README.md`). Steady-state games/h: p = 0.5 9,038 (play measured, unsealed:
  killed by the bench's 15 GiB RSS cap at game 959/960); p = 0.25 12,208 (whole run 10,745); p = 0.125 15,077
  (whole run 13,197). Against the pilot's 5,204 at p = 1.0, that is ×1.74 / ×2.35 / ×2.90. The earlier
  eval-count guess, games/h ≈ 1 / (p / 5,021 + (1 − p) / 19,300), gave ≈ 11.3k (×2.3) at p = 0.25.
  - Policy rows per GPU-hour FALL with p: 434k / 299k / 183k per hour, against the pilot's 496k.
  - **The rule is on EFFECTIVE policy rows**, rows / (1 + (m − 1)·0.03) for m rows per game (the pilot launch record,
    "2. The owner's design as an equal-GPU-time arm"). At 1.6 GPU-h:

    | p | m (rows/game) | raw policy rows, steady | effective, steady | raw, whole run | effective, whole run |
    |---|---|---|---|---|---|
    | 1.0 (the pilot, measured) | 95.25 | 793k | 207k | — | — |
    | 0.5 (play measured, unsealed: killed by the bench's 15 GiB RSS cap at game 959/960) | 47.57 | 695k | 290k | — | — |
    | 0.25 | 23.99 | 478k | 283k | 412k | 244k |
    | 0.125 | 12.18 | 293k | 220k | 257k | 193k |

    Collection A's 198k is itself effective (≈ 758k raw at m = 95); re-derived from the measured pilot it
    is 207k.
  - **The steady rate is the applicable one.** A 1.6 GPU-h collection is ≈ 20–24k games over ≈ 5,800 s,
    with one ramp and one drain. The 960-game bench runs spend ≈ 12% of their wall (262 s at p = 0.125)
    filling and draining 384 slots: the whole-run rate is 87.5% of the steady rate at p = 0.125 and 88.0%
    at p = 0.25.
  - **By the rule as written, the smallest passing p is 0.125 on the steady rate, but only marginally:**
    220k against 198k (against 207k if A is re-measured). On the whole-run rate it FAILS (193k).
    p = 0.25 passes on both rates (283k / 244k).
  - Capped games are more decisive: 0.586 / 0.606 / 0.620 against 0.510 at p = 1.0 (SE about 0.016). Plies
    per game are unchanged.
  - A capped full-ply target rests on slightly fewer visits. Carried visits: mean 8.5 at p = 0.25 and 7.1 at
    p = 0.125, against 12.1. Realised child visits per target: 71.3 / 70.0, against 75.0.
  - 6 × 96 is refused by `check_vram` (41.8 GiB against 26.4 usable): the estimate uses the 64-sim peak even
    for a mostly-fast collect.
- **Disk:** 1.87 KB per row (`snapshots-arms-e2e-20261120/src_graphs`: 23.0 MB of
  shards for 12,299 rows; the collect agent measured 1.86 KB on head 347aaf08c).
- **RAM:** 28.0 KB per row resident for the training keys (float16 planes and
  targets). On today's main the full training load (`load_window`: the
  target-source keys, 42.9 KB per row while a generation loads; the per-shard
  arrays and their concatenation; the train / held-out selection copy and a second
  concatenation) peaks at **≈ 141.6 KiB per row** (collect agent, measured at
  11.9k and 47.6k rows; an earlier 85 KB figure covered `load_generation_rows`
  only). For the pilot's ≈ 758k rows that is **≈ 102 GiB: it does not fit**. With
  the loader-RAM PR (prerequisite): ≈ 27.95 KiB per row + one decoded shard
  (≈ 1.2 GiB at 20k-row shards), **≈ 20 GiB** for the pilot.
- **Kill / pause at the pilot's size** (collect agent, #1441 / #1444): without #1444
  an unfinished process restarts from its first game (a kill at 90% loses ≈ 1.45 h);
  with #1444 ≈ 4 min per process (`--rows-per-shard 5000`: ≈ 2 min). PAUSE holds all
  processes but keeps ≈ 23 GB of VRAM allocated; to free the GPU, kill and resume
  under #1444.
- **Training:** at least 770 rows/s (3.0 updates/s at batch 256, wall including
  start-up, `snapshots-arms-e2e-20261120/arm_wdl1.0`); a floor, not a rate at 512.
- **Match:** an 800-game 64-sim match vs the base has Elo SE ≈ 8.9 (the head-swap
  rows' CI half-widths ≈ 17.4 / 1.96). Plain net-vs-net matches ran 731–800 s per
  800 games (≈ 0.2 GPU-h); two-net hybrid arms (graft, recalibrated value via the
  head-swap harness) ran up to 2,527 s (≈ 0.7 GPU-h).

## Step 1: the pilot — is the loss shape, does a soft target avoid it, does the policy pay

Rewritten after the review (#1452). The earlier Z-2 / Z-all / S-all / P-anchor
design is withdrawn: Z-2 changed the AMOUNT of value training (2 of ~95 value rows,
about 2% of Z-all's value gradient), so any dose-proportional mechanism reads
"variance", and in simulation a variance world read "variance" only 15% of the time
(`plan-review-1452/power.txt`). The owner's 2-per-game idea moves to a
[dose-matched follow-up](#follow-up-the-owners-2-per-game-idea-dose-matched).

**Collection:** one, from the SL base, 64 sims on every ply (`--full-prob 1.0`),
**8,000 games** (8,000 / 5,021 ≈ 1.6 GPU-h; ≈ 758k rows; 1.4 GB disk; ≈ 21 GB
resident; ≈ 102 GiB peak at load on today's main, which does not fit, so the
loader-RAM PR is a prerequisite and brings it to ≈ 20 GiB; each training cell still
runs alone, never beside an arena). Flags: `--collect-procs 3 --concurrency
128` (the default at 64 sims, spelled out because it is pinned and sets the RNG
streams), `--rows-per-shard 5000` (a shorter kill window under #1444), the production
Syzygy pair, and the cells' train flags given at the first stage (it pins them).

**The pilot's own recipe** (pinned here; Step 2's items do not apply):

| Item | Value | Why |
|---|---|---|
| Init, optimizer | the SL base; Aurora split, batch 512, its END rates `--lr 3e-6` (matrices 6e-5), flat (`--optimizer aurora --optimizer-hparams-source init-checkpoint`, #1445) | the converged net's own rates |
| Presentations | `--presentations 1 --mirror-prob 0.5` | exact-once |
| SL mix | off | an SL share anchors the value and weakens the control; F12 / Q50 had none |
| Policy target | the stored search target, `--train-target-temp 1.0` (as F12 had it) | no second change to the data the policy sees |
| Value search source | `--value-search-source stored_search_wdl` on every cell with a search share | the driver default `survivor_q` would silently make a different blend than Q50's |
| KL anchors | none (`--kl-policy 0 --kl-wdl 0`) | nothing below is called "anchored" |

**Cells** (all on the same games):

| Cell | Training | Role |
|---|---|---|
| **Zq** | Q50's blend: `--value-outcome-weight 0.5 --value-search-weight 0.5` | the control: the largest measured damage (registered −53.9, head-swap −63.2) |
| **S** | `--value-search-weight 1 --value-outcome-weight 0` | the SL base's value-target family; expected sign unknown, possibly negative |
| **P0** | `--w-wdl 0` (policy only) | the policy dose, read with the base's value grafted on |

**Order of work** (the RAM and VRAM facts fix it): collect (≈ 1.6 GPU-h, kill /
resume under #1444 — a PAUSE keeps ≈ 23 GB of VRAM allocated, so a collect that
spans the owner's gaming is killed and resumed, not paused) → train the three cells
ONE AT A TIME with nothing else on the box (≈ 20 GiB load peak each with the
loader-RAM PR; ≈ 102 GiB without it, which does not fit) → fit the recalibration → matches 1–4.

**Matches** (registered, seat-balanced, 64 sims, common opening list from the
openings split's `eval.pgn`; SE = 8.9 × √(800 / games) Elo; played in this order;
each 1,600-game block is a kill / resume unit):

1. **Zq vs base — the gate, 1,600 games** (SE 6.3). "Damage reproduced" iff
   UB95 < 0. If not, match 2 is skipped and the record says the recipe did not
   reproduce the damage at this dose. That is NOT evidence the recipe is safe (see
   Step 2's adoption rule). Matches 3 and 4 are played either way.
2. **Zq-recalibrated vs Zq, head-to-head, 1,600 games** (SE 6.3). "The loss is
   shape" iff LB95 > 0; the recovered fraction (diff / −Elo(Zq)) is reported.
   T and b are fitted with `scripts/fit_value_recal.py` (the arena PR) against the
   base's WDL on **the pilot collect's hold-out games** (`is_holdout_game`, never
   trained on; TRAIN-split openings, so disjoint from the match's `eval.pgn` by the
   split's orbit rule) — pre-registered here, fitted before the match. A
   recalibration-only seat runs one forward per evaluation, so matches 1–2 run at
   plain speed (≈ 0.4 GPU-h per 1,600 games).
3. **S vs base, 800 games** (SE 8.9). Damage iff UB95 < 0. "S adoptable" iff
   LB95 > −10 (Step 2). 800 games is kept: at a true 0 S is adoptable only 20% of
   the time, and 1,600 games raises that only to 36%; S's expected sign is unknown
   and the bank leans negative, so the cheaper read decides whether S earns a
   larger one.
4. **P0's policy grafted onto the base's value vs base — group-sequential,
   pre-registered.** Looks at 1,600 / 3,200 / 4,800 games with O'Brien–Fleming-type
   boundaries z = 3.471 / 2.454 / 2.004 (est and SE from all games so far): stop
   "the policy pays" iff est − z·SE > 0; stop "does not pay +10" iff est + z·SE <
   +10; otherwise continue; after the third look, "unread". No other stopping.

**Why +10 sizes match 4** (`plan-review-1452-r2/sizing.txt`): the banked "trained
policy, sound value" reads sit at about 0 near 2.3k games (P1 −3.9, Q50-policy +4.3,
F12-policy −15.7); the pilot's 8k games are 1.8 doublings beyond, and Step 3's
smallest slope worth detecting is 5.3 Elo per doubling, so the effect that matters
here is 5.3 × 1.8 ≈ **+10 Elo**. An 80 / 80 fixed design at +10 needs SE 3.57
(≈ 5,000 games); the sequential design reaches the same error rates at an expected
≈ 4,070 games at a true 0 or +10 and ≈ 2,600 at +20. The 1-sim gain (+79) is the
wrong yardstick: 64-sim search absorbed it.

**What match 4 decides.** If the policy does not pay +10 at 64 sims at this dose
with the value held EXACTLY at the base's, no value-target fix can make the loop pay
in this form at Step 3's detectable rate. The next levers are then the policy side:
more sims in the target (the target is a 64-sim search), the policy dose (more
games), and recalibration-in-the-loop only as damage control.

**Secondary, pre-registered (report only):** the WDL-shape statistics of every net
on A1 (entropy, |W − L|, draw mass by the base's |W − L| bucket, top-2 child gap
where child values are computed), via the scorer (#1448). The child top-2 gap was
already at its c12 level by F12 c6 while c6 showed no Elo loss, so these do not
substitute for a match.

Also report-only, per cell: the policy KL from the base and the policy entropy on
A1. A comparison across optimizers or doses is meaningful only at matched KL (at
Aurora 512 the KL moved per row differs from NAdam 256), so P0's dose is compared
with P1 / F12 by KL as well as by games. Dose is NOT controlled by KL.

**64-sim matches are the only arbiter.** The diagnosis's regime split is one bit
(a shared sign), and no offline column reproduces an ordering among nets. Claims
this record does not rely on, in the strong form they have been quoted in:
- "Optimizer / batch is not a lever." What the screen licenses: no difference of
  about 0.045 or more in raw-policy changed-row precision at matched KL between
  batch 256 / 512 / 2048, on rulers that do not validate; the accumulation control
  was not a clean pass; 64-sim play was never measured. The pilot's Aurora split at
  512 stands on the owner's own tuning, not on this screen.
- "The tablebase move is better by SF" (28.5 vs 48.8 cp, CI [−5.0, +66.9]). What
  holds: tablebases change lc0's move on 1.9% of rows with no detectable quality
  difference by SF. Tablebases stay on for every engine because the owner requires
  it and it is correct, not because of this number.
- Sharpening is real and shared across optimizers (policy entropy 1.19 → about
  0.87 in every arm), but whether it costs 64-sim strength is untested (the
  softening control never ran). The pilot trains at temperature 1.0; a softening
  guard is a named later lever.

**Reading table, simulated** (`amendment-1452-sim/pilot_sim_v2.py`, 400k draws per
world, sizes as above; output `pilot_sim_v2.txt`; the superseded 800-game version is
in `v1/`). Columns: P(gate reads damage); P(Zq adoptable, LB95 > −10); P(match 2
reads shape | gate); S: P(damage), P(adoptable); match 4: P(pays / does not pay +10 /
unread) and expected games.

| True world (Zq, recal gain, S, graft) | Gate | Zq adopt | Shape \| gate | S dmg | S adopt | Match 4: pays / not +10 / unread | E[games] |
|---|---|---|---|---|---|---|---|
| shape loss, S mild, policy 0 (−55, +45, −10, 0) | 1.00 | 0.00 | 1.00 | 0.20 | 0.03 | 0.03 / 0.78 / 0.20 | 4,072 |
| shape loss, S safe, policy +10 (−55, +45, 0, +10) | 1.00 | 0.00 | 1.00 | 0.02 | 0.20 | 0.78 / 0.02 / 0.20 | 4,071 |
| loss not shape, S as bad (−55, 0, −55, 0) | 1.00 | 0.00 | 0.02 | 1.00 | 0.00 | 0.03 / 0.78 / 0.20 | 4,071 |
| partial shape (−55, +20, −20, 0) | 1.00 | 0.00 | 0.89 | 0.61 | 0.00 | 0.02 / 0.78 / 0.19 | 4,069 |
| small damage (−15, +12, 0, 0) | 0.67 | 0.00 | 0.48 | 0.03 | 0.20 | 0.03 / 0.78 / 0.20 | 4,070 |
| no damage (0, 0, 0, 0) | 0.02 | 0.35 | 0.02 | 0.02 | 0.20 | 0.02 / 0.78 / 0.20 | 4,068 |
| policy pays +20 (−55, +45, −10, +20) | 1.00 | 0.00 | 1.00 | 0.20 | 0.02 | 1.00 / 0.00 / 0.00 | 2,620 |
| policy +5 (−55, +45, −10, +5) | 1.00 | 0.00 | 1.00 | 0.20 | 0.02 | 0.27 / 0.27 / 0.45 | 4,493 |

So: a shape loss of the head-swap's size is read correctly almost always, a partial
one (+20) 89%; a false "shape" 2%; a small (−15) control damage passes the gate 67%;
and match 4 reads +10 correctly 78% of the time and a true 0 as "does not pay +10"
78%. A safe recipe is "adoptable" only 20–35% of the time: by design the record does
not adopt a value target on a failure to see damage.

**Cost** (GPU-h): collect 1.6; training three cells ≈ 0.8 (≈ 16 min each at
≥ 770 rows/s, run in sequence); recalibration fit < 0.1; matches: (1) 1,600 plain
≈ 0.4, (2) 1,600 plain (recal-only seat, one forward) ≈ 0.4, (3) 800 plain ≈ 0.2,
(4) 1,600–4,800 two-forward hybrid ≈ 0.45–0.7 per 800 games (head-swap bank), so
≈ 0.9–4.2. **Total ≈ 4.0–7.6 GPU-h**, expected ≈ 6.3 (match 4's expected
≈ 4,070 games).

**Prerequisite code (the only one): the tested arena option** (`--candidate-value-from`
/ `--candidate-value-recal`, mirrored for the reference) with these controls by name,
each a test in its PR: `--candidate-value-from <the candidate's own checkpoint>`
with `--candidate-value-recal 1,0` plays games byte-identical to the plain arena;
`--candidate-value-recal 1,0` alone likewise; a value with W and L swapped declines
material the plain value wins. The scratch head-swap harness must not be used for a
registered match: its loader monkeypatch is untested in the repo.

## Step 2: the recipe for the large generation

Set by the pilot's reading; items marked **guess** are listed for the owner.

- **Init:** as the pilot (SL base, Aurora split at 512, `--lr 3e-6`, matrices
  6e-5, flat). The SL run's last segment (epoch 2) ran from matrices 3e-4 (AdamW
  1.5e-5) down to 6e-5 / 3e-6 (`RESULT.json`: `epoch2_sqrt_policy.matrix_lr_start`
  3e-4, `matrix_lr_end` 6e-5, `actual_final_group_lrs` [6e-5, 3e-6, 3e-6, 3e-6]).
  The end rates continue a converged net; a cosine from the epoch-2 start rates
  (`--lr 1.5e-5 --lr-floor 3e-6 --lr-schedule cosine-phase`) is the alternative
  (**guess**, not tried).
- **Value target — adopted only on a shown margin, never on a failure to see
  damage.** A target is adopted only if its match against the base has
  **LB95 > −10 Elo** (it loses at most 10). A gate that does not read damage, or an
  S match that reads "null", is NOT safety: at a true −15 a 1,600-game gate fails
  to read damage 33% of the time and an 800-game S match's null band admits −15, and a per-generation
  −15 compounds over Stage 2. In order:
  1. S if match 3 has LB95 > −10;
  2. else Zq if match 1 has LB95 > −10;
  3. else, if match 2 read shape, Zq with the recalibration refitted and folded
     into the WDL head after each generation — and its first generation's match
     must itself show LB95 > −10 before a second generation runs;
  4. else stop for the owner.
- **Policy rows:** every full-search row (policy residual ICC 0.030; argmax-flip
  ICC 0.004).
- **Policy target:** as the pilot (train temp 1.0). A sharpening guard
  (`--train-target-temp 1.5` or `--kl-policy`) is a separate, later change.
- **SL share:** off unless the owner asks; if on, it needs an SL corpus in
  shard format with `search_wdl` while s > 0, ≈ 28 GB RAM per 1M SL rows.
- **Exact-once**, `--presentations 1 --mirror-prob 0.5`; a native batch of 512.

## Step 3: one large generation, read as a dose-response curve in games

Only with the value target and value rows the pilot supports.

**Collection.** ONE collect from the SL base, sized in GPU-hours (12 GPU-h
proposed). Policy wants full-search rows (near-independent); a z-based value wants
games. The share p of full-search plies trades the two; rates below are 12 × the
games/h above, and the playout-cap rates are **guesses** until measured:

| Design | Games | Full-search (policy) rows | Disk | RAM at load (loader-RAM PR, ≈ 28 KiB/row) |
|---|---|---|---|---|
| p = 1, 64 sims every ply | ≈ 60k | ≈ 5.7M | 10.7 GB | ≈ 153 GiB: does not fit |
| p = 0.25, 64 / 16 (KataGo's p) | ≈ 135k (guess) | ≈ 3.2M | 6.1 GB | ≈ 86 GiB: at the 88 GiB limit |
| p ≈ 0.021, 64 / 16 (≈ 2 full plies per game) | ≈ 218k (guess) | ≈ 0.44M | 0.8 GB | ≈ 13 GiB |
| 16 sims every ply (the alternative) | ≈ 232k | ≈ 22M | 41 GB | does not fit |

- **Proposal:** p = 1 at 64 sims (the pilot's own collect shape) unless the
  dose-matched follow-up reads variance, in which case games matter most and
  p = 0.25 is preferred. With the loader-RAM PR a window of ≈ 3.3M rows fits
  (≈ 7 GPU-h at p = 1, ≈ 35k games); a 12 GPU-h p = 1 collect still needs the lazy
  window (missing) or a smaller collection.
- The artifact disk has 366 GB free.
- Kill-safe through #1444's game-granularity resume; pausable with the PAUSE marker
  (#1441); one output directory under `$R`.

**Training.** One independent run per dose, each on the first G_i games of the one
collection, at G/32, G/16, G/8, G/4, G/2 and G (six, doubling; x = log2 G_i,
Σ(x − x̄)² = 17.5). Total training ≈ 2× one full run (< 1 h at the pilot's size,
more at Step 3's).

**Readout.**
- **Primary: the slope of Elo vs log2(games).** Each snapshot plays a registered,
  seat-balanced 64-sim match vs the base, 800 games (SE ≈ 8.9). Weighted least
  squares over the six points: SE(slope) = 8.9 / √17.5 ≈ **2.1 Elo per doubling**,
  so a slope of 2.49 × 2.1 ≈ **5.3 Elo per doubling** is detected at one-sided
  α = 0.05 with 80% power. Five separate 800-game verdicts would each need ≈ 25 Elo.
  **SE(slope) is match noise only**: training-seed noise is not in it (independent
  runs per dose make the points independent, but one seed per dose). Assumes a
  linear trend in log2 games; a curvature term is reported, not tested.
- **Match cost:** 6 × 800 = 4,800 plain games ≈ 1.2 GPU-h; arena share ≈ 8% of the
  experiment (12 h collect + ≈ 1–2 h training + 1.2 h matches).
- **Secondary:** deep-SF regret on the leak-free rows (the scorer, #1448) and the
  value pre-screen on every snapshot, report only;
  temperature-0 1-sim reads of every snapshot beside the searched curve, so the
  raw-policy and searched curves sit side by side. BT4 / Ceres / SF ruler columns
  are report-only.

**Pre-committed readings.**
- **The loop works at this scale:** the slope's one-sided 95% lower bound > 0 AND
  the final point's LB95 > 0 (a rising curve that stays below the base does not
  count). Report Elo per doubling of games, and extrapolate to Stage 2's budget.
- **Flat:** the slope's 95% CI inside [−5.3, +5.3] and the final point's CI
  containing 0. The loop in this form does not pay at this scale; the record says so
  and names the next options (more sims in the target; policy dose; a value target
  from a deeper teacher once the root-Q bias is explained).
- **Negative:** value damage. **Stop early** when two CONSECUTIVE doses each have
  UB95 < −15 Elo (damage need not be monotone in dose: F12 read about 0 at cycles 3
  and 6 and −20 at 12). No MAE conjunct: it cannot fire on the known failure (Q50 at
  −63 has the better MAE).
- **Unread, insufficient evidence:** anything else. At SE 2.13, a true slope of 0
  lands here 55% of the time and a true slope of 3 43% (`plan-review-1452/power.txt`);
  the record then says "unread", never "deferred".

## Stage 2, reframed

A generation = collect hours + train hours + readout hours. With the default above:
12 + ≈ 1–2 (training, one run per dose) + ≈ 1.2 (matches) ≈ **14–15 GPU-h**, so
about **11 generations per week** at full GPU use, fewer with the owner's gaming
hours. After the first, a generation needs one training run and one adoption match
against the previous generation sized to Step 2's margin (LB95 > −10 at 80% for a
true 0 needs ≈ 5,000 games, ≈ 1.3 GPU-h plain; a sequential design with the same
looks as match 4 cuts the expectation), ≈ 14 GPU-h each. These are design numbers; Stage 2's own record re-derives them from the
pilot's measured rates.

## Decisions for the owner

| # | Decision | Proposal | Owner's decision |
|---|---|---|---|
| O1 | Launch the pilot (collect 8k games; train Zq, S, P0; matches 1–4 as above) | Yes, once #1441, #1444, #1445, the loader-RAM PR and the arena option are merged | *(unfilled)* |
| O2 | Total budget | ≈ 4.0–7.6 GPU-h (expected ≈ 6.3), in kill / resume blocks of ≤ 1.6 h | *(unfilled)* |
| O3 | Changes to the three cells | none proposed (Zq = Q50's blend; S = search WDL only; P0 = policy only) | *(unfilled)* |

Step 3's collection size, its SL share and the rows-per-game follow-up are decided
after the pilot reads, in their own record.

## Before the pilot

**Superseded for execution by the [launch record](2026-11-21-rl-run1-pilot-launch-record.md)**, which drops #1444 from the
prerequisites, re-measures the match costs, and adds the max-plies and adoption rules. Where they differ, the launch record wins.

- The only prerequisite code: the tested `--candidate-value-from` /
  `--candidate-value-recal` arena option with its identity and swapped-W/L controls
  (above); the loader-RAM PR is a merge prerequisite for the training step. Step 3
  needs a train-side game-prefix filter and, above ≈ 3.3M rows, the lazy window. The fast-ply noise check precedes any playout-cap collect.
- #1441, #1444, #1445 and the loader-RAM PR merged; the pilot runs from a detached worktree of main.
- Main's CLAUDE.md experiment rule: this record names the hypotheses (the loss is
  shape; a soft target avoids it; the policy pays at 64 sims with the value held),
  the controls (Zq; the base), the deciding yardsticks (matches 1–4 with their
  thresholds; the curve's slope), and the recovery plan (the SL base is never
  overwritten; every cell is a new run directory).

## Follow-up: the owner's 2-per-game idea, dose-matched

The only design that isolates within-game variance from the amount of value
training: two arms with policy on all 8k games and value trained on about 16k rows
each — arm A 2 value rows from each of the 8k games, arm B every value row of a
~168-game subset (168 × 95 ≈ 16k). Variance predicts B much worse (n_eff ≈ 280 vs
≈ 10k at ICC 0.6); shape or dose predicts them equal. It needs the per-row value
mask (`--value-rows-per-game`). **Launch condition:** the pilot's gate reads damage
and match 2 does not read shape (the loss is not explained by shape), or the owner
asks.

## Corrections to the plan and the takeover record

Statements overtaken by later banked evidence (found by the evidence-log audit,
PR #1451), each re-read from its source and corrected in place:

| Was | Now | Source |
|---|---|---|
| Interim deep-SF table (n = 127): R64 16.3 cp beats no-rescale 25.5; stored 25.2; base 42.6; P1 c12 29.5 | Final leak-free labels (n = 2,433): R64 24.3, no-rescale 23.2, stored 22.8, base 44.6, P1 c12 38.6; "no-rescale is worse than rescaled R64" does not survive | `search-compose/logs/sf_parse_final.txt` |
| "T1 not read" | Read: +79.1 [+59.7, +98.9], supported | `search-compose/PREREG.sha256.addendum`, addendum 5 |
| "The local lc0 builds are CPU-only" | A CUDA lc0 build exists and is smoked with tablebases; Ceres with its own C3 net likewise | `~/local_engines/lc0-0.32.1-cuda/smoke_gpu.json`, `~/local_engines/ceres-0cb57d8/smoke_gpu.json` |
| Learner screen arms N1024 / N4096 / N4096L | Replaced by N256 / A512 / A2048 / N512 (addendum 6), now deprioritised | addendum 6 |
| Softening control "still running" | Never produced a game | `raw-net-1sim/BASE_cand_s64_pt1.5.log` and `BASE_ref_s64_pt1.5.log` stop at opening sampling; no games file |
| Ruler validation pending | Positive control fails on both raw rulers: BT4 raw +0.0074 [−0.0117, +0.0265], C3 raw +0.0181 [−0.0019, +0.0381]; neither validates. Offline ruler: deep-SF regret (policy); none for value | `rl-run1-prereq/a1_poscontrol_score.json` |
| Takeover D2: policy-only KL "not separately reported"; value-only and self-target arms omitted | Policy only 0.0384; value only KL 0.0171, mean abs ΔQ 0.0916; self-target KL 5.8e-5, mean abs ΔQ 0.0103 | `rl-audit/check6_policyonly.json`, `check6_valueonly.json`, `check6_selftarget.json` |
| "96.8 rows per game" (unsourced) | 95.0 (`collect-throughput/bench_c128` manifest, 48,641 / 512); F12's 12 cycles 104.7 | the manifests and draw journals named above |

Corrections to this record's own first draft, from its review (#1452): the
recalibration claim (F12's value recovers about a third, not most; the full-net
rows added); 86% not 87%; the SL start-rate claim (now read from `RESULT.json`);
one main sha; the hybrid-arm match cost; ~240k rows, not ~100k; S-all is not
shape-preserving; `value_prescreen.json` is interim.
