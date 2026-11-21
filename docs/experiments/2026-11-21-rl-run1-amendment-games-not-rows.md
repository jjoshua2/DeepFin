# RL run 1 amendment: count games, not rows (DRAFT)

**Status:** DRAFT, awaiting the owner's decisions ([below](#decisions-for-the-owner)).
Nothing has been launched. This amends
[RL run 1](2026-11-20-rl-run1-multi-generation-plan.md): it replaces that plan's
Stage 1 (four policy-target cells on one ~100k-row collection) with a **rows-per-game
pilot** followed by **one large generation read as a dose-response curve whose x-axis
is games**. The plan's Stage 0b rulers, its scorer (#1448) and its BT4 / Ceres / SF
ruler columns stay, as report-only columns.

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
- **The effective sample size for the other targets** (search policy, search value)
  is `n_eff = G · r / (1 + (r − 1) · ρ̄)` for G games, r rows per game and mean
  within-game correlation ρ̄. ρ̄ is being measured by the diagnosis agent; this record
  states n_eff for the earlier runs and for each design below **when that bank lands**.
  Until then the table carries the formula, not a number.

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
policy-target cell at ~100k rows reads about 0 at 64 sims, and the losses sit in the
value head.

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
- **The search value is not obviously a better target.** The 64-sim root Q
  (`rootQ_R64`) sits at MAE 0.1157, bias −0.102 (pessimistic), farther from SF than
  the base's raw value.

### T1: the raw policy did learn

T1 (registered in `search-compose/PREREG.md`, temperature 0, 1 sim, P1 c12 vs the
base, both orientations, 800 games each; `search-compose/results/pool_t0s1_P1.json`):
**+80.9 [+60.3, +102.1]** and **+77.3 [+56.7, +98.4]** (orientation B, sign
flipped). The ~240k rows moved the raw policy by about +79 Elo at 1 sim; the gain
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

| Need | On main (15d515cce) / open PRs | Status |
|---|---|---|
| Playout cap (full search on a share p of plies, fast elsewhere, fast plies store no row) | `--full-prob p`, `--fast-sims n`; `--fast-value-rows` keeps fast plies as value-only rows | **exists**; matches KataGo's "only full-search turns are recorded" |
| Fast plies without root noise (KataGo disables noise on fast searches) | not checked | **to check** in `collect.py` / the selfplay path before the pilot |
| Rows per game r chosen at TRAIN time from a full collection (the pilot's equal-games contrast) | none | **missing**: a `--rows-per-game r` train option, selection keyed by a hash of `game_id` (as `is_holdout_game` does), recorded in the manifest and the presentation ledger. Small, `learn.py` only |
| Dose-response snapshots at game counts in ONE run | `--snapshot-updates`; `--snapshot-rows` (#1447) | **missing**: a nested-prefix schedule (games in a seeded order; rows of games [G_{i−1}, G_i) shuffled within that segment) so the snapshot at G_i has trained on exactly the first G_i games. Without it, a shuffled snapshot at 1/8 of the rows has seen rows from ~all games |
| Window larger than RAM | the train window is loaded whole (`load_window`, `lazy=False`) | **blocker for r = all at scale**: 28.0 KB of RAM per training row (measured below); 98 GB RAM fits ~2M rows with headroom. Not a blocker at r = 2 |
| Multi-process collect, PAUSE marker | #1441 (open) | needed |
| Kill-safe resume | #1444 (open, stacked on #1441) | needed |
| SL base's Aurora split, its realised LRs | #1445 (open) | needed |
| Value blend | `--value-search-weight`, `--value-outcome-weight` (search share = s / (s + o)), `--value-search-source survivor_q` | exists |
| Policy sharpening guard | `--train-target-temp`, `--kl-policy` / `--kl-wdl` with `--kl-reference` | exists |
| SL-corpus share | `--sl-mix-dir`, `--sl-mix-frac`, `--sl-mix-max-rows` (default 1M, loaded in RAM) | exists; **to check**: an SL corpus in shard format with `search_wdl` when the value search weight > 0 (the loader refuses otherwise); 1M SL rows ≈ 28 GB RAM |
| Gradient accumulation | #1447 | not used by this design |
| Collect process shape for mostly-fast collects | #1441 picks the shape from `--sims` (3 × 128 above 16 sims) | **to measure** in the pilot: with p ≈ 0.02 most plies are 16-sim, where 8 × 64 was faster; pass `--collect-procs` / `--concurrency` explicitly if the VRAM estimate allows |

## Measured costs used below

- **Collect, 64 sims every ply:** 5,021 games/h at 3 × 128 (#1441, RTX 5090, SL base);
  95.0 rows per game (`collect-throughput/bench_c128`: 48,641 rows / 512 games).
- **Collect, 16 sims every ply:** 1.85M training rows/h at 8 × 64 (#1441); 96.0 rows
  per game (`bench_c256_s16_prof`: 49,158 / 512) → **≈ 19.3k games/h** (derived).
- **Collect, playout cap at 64 / 16 sims**, assuming GPU-bound cost proportional to
  the per-ply sim mix (**guess**, to be measured by the pilot's collect `stats.json`):
  games/h ≈ 1 / (p / 5,021 + (1 − p) / 19,300). p = 0.25: ≈ 11.3k games/h;
  p = 0.021 (≈ 2 full plies per ~96-ply game): ≈ 18.2k games/h.
- **Disk:** 1.87 KB per row (`snapshots-arms-e2e-20261120/src_graphs`: 23.0 MB of
  shards for 12,299 rows). **RAM:** 28.0 KB per row for the training keys (float16
  planes and targets, measured on the same shard).
- **Training:** at least 770 rows/s (3.0 updates/s at batch 256, wall including
  start-up, `snapshots-arms-e2e-20261120/arm_wdl1.0`); a floor, not a rate at 512.
- **Match:** an 800-game 64-sim match vs the base has Elo SE ≈ 8.9 (the head-swap
  rows' CI half-widths ≈ 17.4 / 1.96); wall 731–2,527 s per 800 games in the
  head-swap bank, so **0.2–0.7 GPU-h per 800 games**.

## Step 1: the rows-per-game pilot

The cheapest experiment that tests the owner's hypothesis directly.

**Hypothesis.** Fitting each game's outcome ~100 times from correlated positions is
what damages the value head; at r = 2 rows per game the same games do not damage it,
and at equal GPU time a playout-cap collect yields more games and a better net.

**Collections (both from the SL base, the plan's openings and seed scheme):**
- **P-all:** 64 sims on every ply (`--full-prob 1.0`), **8,000 games**
  (8,000 / 5,021 ≈ 1.6 GPU-h; ≈ 760k rows; 1.4 GB disk; 21 GB RAM for r = all).
- **P-cap:** playout cap at equal GPU time, `--sims 64 --full-prob 0.021
  --fast-sims 16` (no rows on fast plies), **1.6 GPU-h** → ≈ 29k games and ≈ 58k
  rows (**guess** from the rate above; the realised count is what is reported).

**Training arms** (one recipe, [Step 2](#step-2-the-recipe); value search share s):

| Arm | Data | r | s | Match games |
|---|---|---|---|---|
| A-all-0.75 | P-all | all (~95) | 0.75 | 800 |
| B-r2-0.75 | P-all | 2 | 0.75 | 800 |
| A-all-1.0 | P-all | all | 1.0 | 400 |
| B-r2-1.0 | P-all | 2 | 1.0 | 400 |
| C-cap-0.75 | P-cap | ≈2 (by the cap) | 0.75 | 400 |
| C-cap-1.0 | P-cap | ≈2 | 1.0 | 400 |

A and B share every game, so their contrast is rows per game at equal games; C vs A is
equal GPU time. B trains on ≈ 16k rows (≈ 31 updates at 512): a null there means "no
damage", not "no gain". r = 2 is the owner's default; the contrast is r = all. When
the within-game correlation bank lands, r for the big collection is proposed from it
(the smallest r whose n_eff is within 10% of r = all's, **rule is a guess**).

**Reads.**
- Offline, every arm (minutes, CPU): `search-compose/value_prescreen.py` on its rows
  (MAE / bias vs SF 8M; 1-ply chooser regret with game-clustered CI), held-out
  value loss on held-out games, and the plan's offline policy ruler (#1448).
- Temperature-0 1-sim reads, every arm (minutes): raw-policy movement.
- One registered seat-balanced 64-sim match per arm vs the base (games above; 3,200
  in all ≈ 0.8–2.8 GPU-h).

**Pre-committed readings.**
- **Rows per game matters** if Elo(B-r2-0.75) − Elo(A-all-0.75) ≥ 2 × 12.6 = 25.2
  (SE of the difference of two independent 800-game reads), or if A's chooser regret
  is worse than B's with a game-clustered 95% CI excluding 0. The big collection
  then uses r = 2 (or the correlation-derived r) and the playout cap.
- **It does not** if that difference is < 25.2 and A's value is not damaged either
  (A's 95% CI contains 0): the big collection uses whichever of A / C is better at
  equal GPU time, and the record says the owner's mechanism was not seen at 8k games.
- **The outcome share matters** if the s = 1.0 arms beat the s = 0.75 arms by
  ≥ 2 SE of the pooled contrast; the big collection then uses s = 1.0.
- **A broken recipe** (any arm < −40 Elo, or value MAE vs SF worse than the base by
  more than P1's +0.010) stops the plan before the big collection.

**Cost:** collect 3.2 GPU-h, training < 0.5 h (P-all r = all ≈ 760k rows ≥ 770
rows/s ≈ 16 min per arm, two arms; the rest are tiny), matches 0.8–2.8 GPU-h.
**≈ 4.5–6.5 GPU-h in all.**

## Step 2: the recipe

One recipe, not a grid. Items marked **guess** are listed for the owner.

- **Init:** the SL base; Aurora split at batch 512 as the SL base used it
  (`--optimizer aurora --matrix-optimizer-scope mlp_out --optimizer-hparams-source
  init-checkpoint`, #1445), its realised end-of-SL learning rates, flat schedule.
- **Heads:** all trained heads (policy and WDL); every other weight stays 0 as the
  driver pins it.
- **Value target:** s × search WDL (`--value-search-source survivor_q`) + (1 − s) ×
  game outcome. **s = 0.75 is a guess**; the measured negatives are F12 and Q50's
  outcome shares of 0.9 and 0.5 at ~105 rows per game, and the pilot reads s = 1.0
  against it. Risk to state: the 64-sim root Q is pessimistic vs SF (bias −0.10 in
  the pre-screen).
- **Policy target:** the standard stored search target with `--train-target-temp
  1.5` as the sharpening guard (**guess**; the alternative is `--kl-policy` to the
  base, which also anchors the trunk but costs a reference forward pass per batch).
- **SL share:** `--sl-mix-frac 0.25` of each batch from the SL corpus (**guess**,
  owner's yes/no), to guard against forgetting; needs an SL corpus in shard format
  with `search_wdl` while s > 0 (to check), ≈ 28 GB RAM per 1M SL rows.
- **Exact-once:** `--presentations 1` with `--mirror-prob 0.5` (a mirror is the same
  presentation): each row at most once, inside the owner's ≤ 2 presentations.
- **Batch:** a native 512; no accumulation.

## Step 3: one large generation, read as a dose-response curve in games

Only with the r, p and s the pilot supports.

**Collection.** ONE collect from the SL base, sized in GPU-hours:
- **Default proposal: 12 GPU-h with the playout cap** at 64 / 16 sims and r ≈ 2:
  ≈ 218k games (12 × 18.2k, **guess** until the pilot's rate), ≈ 437k rows,
  0.8 GB disk, 12 GB RAM.
- **If the pilot says r = all at 64 sims:** 12 GPU-h ≈ 60k games, 5.7M rows, 10.7 GB
  disk, **160 GB RAM, which does not fit** (98 GB): needs the lazy window (missing) or
  a smaller collection (≤ ~2M rows, ≈ 21k games).
- **O5's alternative, 16 sims every ply at equal GPU time:** ≈ 232k games, 22M rows,
  41 GB disk (366 GB free on the artifact disk), RAM does not fit at r = all.
- Kill-safe through #1444's game-granularity resume; pausable with the PAUSE marker
  (#1441); one output directory under `$R`.

**Training.** The recipe above, one run over the whole collection with the
nested-prefix schedule (missing code), so snapshot i has trained on exactly the first
G_i games.

**Snapshots** at games G/32, G/16, G/8, G/4, G/2 and G (six, doubling; x = log2 G_i,
Σ(x − x̄)² = 17.5).

**Readout.**
- **Primary: the slope of Elo vs log2(games).** Each snapshot plays a registered,
  seat-balanced 64-sim match vs the base, 800 games (SE ≈ 8.9). Weighted least
  squares over the six points: SE(slope) = 8.9 / √17.5 ≈ **2.1 Elo per doubling**,
  so a slope of 2.49 × 2.1 ≈ **5.3 Elo per doubling** is detected at one-sided
  α = 0.05 with 80% power. Five separate 800-game verdicts would each need ≈ 25 Elo.
  Assumes independent matches and a linear trend in log2 games (both stated in the
  readout; a curvature term is reported, not tested).
- **Match cost:** 6 × 800 = 4,800 games ≈ 1.2–4.2 GPU-h; arena share of the
  experiment ≈ 9–25% (of 12 h collect + < 0.5 h train + 1.2–4.2 h matches).
- **Secondary:** offline scorer (#1448) and the value pre-screen on every snapshot;
  temperature-0 1-sim reads of every snapshot beside the searched curve, so the
  raw-policy and searched curves sit side by side. BT4 / Ceres / SF ruler columns
  are report-only.

**Pre-committed readings.**
- **Slope > 0 with its one-sided 95% lower bound > 0:** the loop works at this
  scale. Report Elo per doubling of games, and extrapolate to Stage 2's budget.
- **Flat** (95% CI of the slope inside [−5.3, +5.3] and the final snapshot's CI
  containing 0): the loop in this form does not pay at this scale. The record says
  so and names the next options (value target from a deeper teacher; lc0-style
  far sparser sampling; a different value anchor).
- **Negative:** value damage. **Stop early** at the first snapshot whose 95% upper
  bound is below −15 Elo (damage of at least 15 Elo established), to save the
  remaining match compute. (Deviation from the requested wording "LB < −X": a lower
  bound below −X fires on noise, so the stop uses the upper bound.)

## Stage 2, reframed

A generation = collect hours + train hours + readout hours. With the default above:
12 + < 0.5 + 1.2–4.2 ≈ **14–17 GPU-h**, so about **10 generations per week** at full
GPU use (168 / 14–17), fewer with the owner's gaming hours. After the first, a
generation's readout can drop to one 800-game match (0.2–0.7 h), ≈ 12.5–13 GPU-h
each. These are design numbers; Stage 2's own record re-derives them from the
pilot's measured rates.

## Decisions for the owner

| # | Decision | Proposal | Owner's decision |
|---|---|---|---|
| O1 | Replace Stage 1's four policy-target cells with the pilot + one large generation | Yes | *(unfilled)* |
| O5 | Collection sims and hours (now first-order) | 12 GPU-h, playout cap 64 / 16 sims, r from the pilot; alternative 16 sims every ply at equal GPU time | *(unfilled)* |
| O6 | Value outcome share | s = 0.75 search share, read against s = 1.0 in the pilot | *(unfilled)* |
| O7 | SL-corpus share | yes, 0.25 of each batch | *(unfilled)* |
| O8 | Rows per game | r = 2 default (the owner's suggestion), r = all as the contrast; revisit with the within-game correlation bank | *(unfilled)* |

## Before the pilot

- The missing code above: `--rows-per-game`, the nested-prefix schedule; and the
  fast-ply noise check.
- #1441, #1444 and #1445 merged; the pilot runs from a detached worktree of main.
- The within-game correlation bank read, and n_eff filled into this record.
- Main's CLAUDE.md experiment rule: this record names the hypothesis, control, the
  deciding yardstick (the pilot's A vs B match difference; the curve's slope), the
  thresholds above and the recovery plan (the SL base is never overwritten; every
  arm is a new run directory).
