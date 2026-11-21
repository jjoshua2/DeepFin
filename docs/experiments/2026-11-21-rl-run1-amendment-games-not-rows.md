# RL run 1 amendment: count games, not rows (DRAFT)

**Status:** DRAFT, awaiting the owner's decisions ([below](#decisions-for-the-owner)).
Nothing has been launched. Code base: main at d678f97ed, where #1447
(`--accum-steps`, `--snapshot-rows`) and #1446 (offline retarget) are merged; #1441,
#1444 and #1445 are open. This amends
[RL run 1](2026-11-20-rl-run1-multi-generation-plan.md): it replaces that plan's
Stage 1 (four policy-target cells on one ~100k-row collection) with a **pilot that
separates value-row variance from value-target shape** followed by **one large generation read as a dose-response curve whose x-axis
is games**. The BT4 / Ceres / SF ruler columns stay, as report-only columns.
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
  carry 76% and 4 rows 87%** of the value information all 104.7 rows carry
  (2 / (1 + 0.6) = 1.25 and 4 / (1 + 3 × 0.6) = 1.43, against 104.7 / 63.2 = 1.66
  per game). The search-WDL residual decorrelates within ~6–10 plies (pairwise
  correlation 0.48 at 1–2 plies, 0.10 at 6–10, 0.07 at 11–20), which supports about
  one value row per 8 plies (~12 per game). **Policy rows are nearly independent:
  keep every full-search row for policy.**

### The damage is not in the value's level: leading hypothesis is WDL shape

From `search-compose/logs/rows_per_game.txt` and `headswap/table.md`:

- On their own training games the trained heads did NOT overfit the outcome in
  level: their pull toward SF (F12 c01 +0.275, F12 c12 +0.375, Q50 c12 +0.491) is
  4–6× their pull toward z (+0.065, +0.066, +0.075), and their level error vs SF
  fell in every split.
- No offline value metric in the bank saw the 64-sim damage (chooser regret nulls,
  level closer to SF), while the head-swap's two-parameter recalibration recovers
  most of it (base policy + Q50 value −55.2 → −8.2; F12 −18.3 → −12.2).
- **The SL base was trained on a fully soft value target**: `search_wdl_frac 1.0`,
  `sf_wdl_frac 0.0`, 174,053,867 rows, batch 512, saved at optimizer step 680,276
  (the SL base's step), per
  `~/chess-artifacts/operations/all-accumulated-174m-epoch2-sqrt-cooldown-20261109-v1.resume-recovery654213-a4349851abddea4a/training/RESULT.json`
  (`identity.recipe`). No one-hot outcomes. F12 and Q50 switched to 90% and 50%
  one-hot z.
- **What that did to the head's shape** (`search-compose/wdl_shape.json`, 2,433
  rows): the heads got flatter (WDL entropy base 0.416 → F12 0.527, Q50 0.586:
  +0.11 / +0.17); |W − L| was compressed (0.440 → 0.378 / 0.341: −14% / −23%);
  draw mass moved toward z's conditional profile (|Q_base| > 0.3: base 0.192, F12
  0.268, Q50 0.283, z 0.242); and the gap between the two best child values
  shrank (`child_spread.top2_gap`, mean 0.142 → 0.123 / 0.112: −14% / −21%;
  median 0.0386 → 0.0294 / 0.0288: −24% / −25%). The head-swap's recalibration
  (`value-recalibration/recalibration_params.json`, `ref=base|temp_drawbias|fitCE`:
  T 0.635 / 0.545, draw bias −0.59 / −0.64) restores the shape statistics
  (entropy 0.427 / 0.436; |W − L| 0.424 / 0.397), matching the Elo it recovers.
- **The stored 64-sim search WDL keeps the base's shape**: draw mass 0.507 (base
  0.507) with the same per-bucket profile, |W − L| 0.402 (base 0.440), entropy 0.450;
  residual ICC 0.063 (≈ 32k effective samples vs ≈ 3.8k for z); and its level is
  CLOSER to deep SF than the base's: MAE 0.0501, −0.0156 [−0.0191, −0.0124]
  (`value_prescreen.json`).
- Rows per game (variance) and value-target shape are separate mechanisms; the
  pilot separates them, and only a 64-sim match can read either.
- **Open:** the stored search WDL's draw mass equals the base net's own to three
  decimals (0.5066 vs 0.5066, every bucket), so its D appears to be the raw net's
  root draw and search moves only W − L. A search-WDL target is then partly
  self-distillation in D; the information it adds is search's W − L. Whether that is
  worth anything is not readable offline; only a 64-sim match scores it.

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

| Need | On main (15d515cce) / open PRs | Status |
|---|---|---|
| Playout cap (full search on a share p of plies, fast elsewhere, fast plies store no row) | `--full-prob p`, `--fast-sims n`; `--fast-value-rows` keeps fast plies as value-only rows | **exists**; matches KataGo's "only full-search turns are recorded" |
| Fast plies without root noise (KataGo disables noise on fast searches) | not checked | **to check** in `collect.py` / the selfplay path before the pilot |
| Value rows per game r chosen at TRAIN time, policy on every row (the pilot's variance contrast) | none: no per-game value-row subsampling | **missing**: `--value-rows-per-game r`, rows chosen by a hash of `game_id` (as `is_holdout_game` does), a per-row VALUE loss mask (the WDL term is masked by `is_network_turn` only today), recorded in the manifest. `learn.py` plus the loss mask |
| Outcome smoothed to the base's draw mass (a second shape-preserving value target) | none | **missing**; the pilot uses search WDL instead |
| Dose-response snapshots at game counts in ONE run | `--snapshot-updates`; `--snapshot-rows` (#1447) | **missing**: a nested-prefix schedule (games in a seeded order; rows of games [G_{i−1}, G_i) shuffled within that segment) so the snapshot at G_i has trained on exactly the first G_i games. Without it, a shuffled snapshot at 1/8 of the rows has seen rows from ~all games |
| Window larger than RAM | the train window is loaded whole (`load_window`, `lazy=False`) | **blocker above ~2M training rows**: 28.0 KB of RAM per training row (measured below); 98 GB RAM fits ~2M rows with headroom. Not a blocker for the pilot (≈ 760k rows) |
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
- **Collect, playout cap at 64 / 16 sims: UNMEASURED.** An eval-count model only
  (**guess**; the diagnosis's own model gives ~2.3× at p = 0.25, agreeing), until a
  throughput run measures it:
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

## Step 1: the pilot — variance, shape, or nothing

The smallest set of cells that separates "too many correlated value rows"
(variance) from "the value target's shape" (shape) from "nothing helps".

**Collection:** one, from the SL base, 64 sims on every ply (`--full-prob 1.0`),
**8,000 games** (8,000 / 5,021 ≈ 1.6 GPU-h; ≈ 760k rows; 1.4 GB disk; 21 GB RAM).
Every cell trains on these same games: the contrasts are at equal games.

**Cells** (all heads trained unless stated; policy uses every full-search row in
every cell; recipe otherwise as [Step 2](#step-2-the-recipe)):

| Cell | Value target | Value rows per game | Separates |
|---|---|---|---|
| **Z-all** | z-heavy blend: outcome 0.9, search 0.1 (F12's) | all (~95) | reproduces the measured damage at this size |
| **Z-2** | same blend | 2 | **variance**: Z-2 vs Z-all |
| **S-all** | s = 1.0, the stored search WDL only (`--value-search-weight 1 --value-outcome-weight 0 --value-search-source stored_search_wdl`): the value-target family the SL base was trained on | all (its ICC 0.063 supports ~12 per game; all is kept) | **shape**: S-all vs Z-all |
| **P-anchor** | none (`--w-wdl 0`), value anchored with `--kl-wdl` to the base | – | "policy only, value held": the reference for "nothing helps" |

**Why Z-all is re-run rather than borrowed from F12 / Q50:** a matched control is
required. F12 and Q50 trained on a different harness and optimizer, over 12
on-policy cycles; S-all vs F12 would confound the value target with all of that.
Z-all costs one ≈ 16 min training and one 800-game match. "2 value rows per game"
is a contrast only where z has weight (Z-2); for S-all the ICC supports all rows.

An outcome smoothed to the base's draw mass is the other shape-preserving target;
the driver cannot build it (missing), so search WDL stands for "shape-preserving".
The playout-cap collect at equal GPU time and the outcome-share grid are dropped
from the pilot: they come after the mechanism is known.

**Reads.** Each cell plays one registered, seat-balanced 64-sim match vs the base,
**800 games** (SE ≈ 8.9 Elo; SE of a difference between two cells ≈ 12.6). The
offline value pre-screen, held-out losses and 1-sim reads are recorded per cell,
report only: the bank shows offline value metrics are blind to this damage.

**Pre-committed readings** (a "difference" is ≥ 2 × 12.6 = 25.2 Elo):

| Z-2 − Z-all | S-all − Z-all | Reading | Big collection |
|---|---|---|---|
| ≥ 25.2 | < 25.2 | variance | value rows ≈ 2 per game (or the ICC-derived r) with the z-heavy blend allowed |
| < 25.2 | ≥ 25.2 | shape | search-WDL value target, value rows ~1 per 8 plies |
| ≥ 25.2 | ≥ 25.2 | both | search-WDL target AND sparse value rows |
| < 25.2 | < 25.2 | neither | if P-anchor ≥ every other cell and its CI contains or exceeds 0: "nothing helps the value at this size"; the record says so and the big collection runs as P-anchor (policy only, value anchored) or stops for the owner |

Z-all is expected near F12's −34.9; if Z-all's 95% CI contains 0, the damage did
not reproduce at this size and the pilot is read as "unread, insufficient
evidence" for the mechanism. Any cell < −40 Elo with a value MAE vs SF worse than
P1's +0.010 is a broken recipe and stops the plan.

**Missing code for this pilot:** per-row VALUE masking (`--value-rows-per-game r`):
choose r rows per game by a hash of `game_id` (as `is_holdout_game` does) and
mask only the WDL loss on the rest, so policy still trains on every row. That needs
a per-row value mask in the loss (today WDL uses `is_network_turn` only, which
would drop policy too). Under `--accum-steps` a partial value mask is refused by
#1447's check, but accumulation is not used here.

**Cost:** collect 1.6 GPU-h; training ≈ 16 min per cell at ≥ 770 rows/s (four
cells, ≈ 1.1 h); matches 4 × 800 = 3,200 games ≈ 0.8–2.8 GPU-h. **≈ 3.5–5.5 GPU-h.**

## Step 2: the recipe

One recipe, not a grid; the pilot fixes only the value target and the value rows
per game. Items marked **guess** are listed for the owner.

- **Init:** the SL base; Aurora split at batch 512 as the SL base used it
  (`--optimizer aurora --matrix-optimizer-scope mlp_out --optimizer-hparams-source
  init-checkpoint`, #1445), flat schedule at the SL run's END rates: `--lr 3e-6`
  (AdamW groups; the Aurora matrices at ×20 = 6e-5), which is what its saved
  groups hold. The SL run STARTED at `--lr 3e-5` (matrices 6e-4) and decayed to
  these. The end rates are chosen because this continues training a converged
  net on new data, and P1 / F12 / Q50 already damaged the value at 3e-6-scale
  rates. **Guess:** a cosine from the start rates (`--lr 3e-5 --lr-floor 3e-6
  --lr-schedule cosine-phase`, per-group floors as #1445 now builds them) is the
  alternative, not tried here.
- **Heads:** all trained heads (policy and WDL); every other weight stays 0 as the
  driver pins it.
- **Value target and value rows per game:** set by the pilot's reading (its table):
  shape → stored search WDL only (`--value-search-source stored_search_wdl`), value rows all (its ICC supports ~1 per
  8 plies); variance → the blend is allowed with ~2 value rows per game; neither →
  P-anchor. The expected reading, from the shape evidence above, is S-all
  (`stored_search_wdl`); its open point (D from the net itself) is judged only by
  the pilot's match.
- **Policy rows:** every full-search row, in every design (policy residual ICC
  0.030; argmax-flip ICC 0.004).
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

Only with the value target and value rows the pilot supports.

**Collection.** ONE collect from the SL base, sized in GPU-hours (12 GPU-h
proposed). Policy wants full-search rows (near-independent); a z-based value wants
games. The share p of full-search plies trades the two; rates below are 12 × the
games/h above, and the playout-cap rates are **guesses** until measured:

| Design | Games | Full-search (policy) rows | Disk | RAM for the window |
|---|---|---|---|---|
| p = 1, 64 sims every ply | ≈ 60k | ≈ 5.7M | 10.7 GB | 160 GB: does not fit |
| p = 0.25, 64 / 16 (KataGo's p) | ≈ 135k (guess) | ≈ 3.2M | 6.1 GB | 91 GB: does not fit with headroom |
| p ≈ 0.021, 64 / 16 (≈ 2 full plies per game) | ≈ 218k (guess) | ≈ 0.44M | 0.8 GB | 12 GB |
| 16 sims every ply (O5's alternative) | ≈ 232k | ≈ 22M | 41 GB | does not fit |

- **Proposal:** if the pilot reads **shape** (search-WDL value, ICC 0.063), value
  information grows with rows as much as games, so p = 1 or p = 0.25; if it reads
  **variance** (z-based value), games matter most, so p = 0.25. Either needs the
  lazy window (missing) for more than ~2M rows; without it the collection is capped
  at ~2M training rows (≈ 4.4 GPU-h at p = 1, ≈ 21k games).
- The artifact disk has 366 GB free.
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
- **Secondary:** deep-SF regret on the leak-free rows (the scorer, #1448) and the
  value pre-screen on every snapshot, report only;
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
| O5 | Collection sims, p and hours (now first-order) | 12 GPU-h at 64 / 16 sims, p from the pilot's reading (table in Step 3); alternative 16 sims every ply at equal GPU time | *(unfilled)* |
| O6 | Value target | set by the pilot: search WDL only (shape) or the blend with sparse value rows (variance) | *(unfilled)* |
| O7 | SL-corpus share | yes, 0.25 of each batch | *(unfilled)* |
| O8 | Rows per game | policy: every full-search row; value: 2 per game for a z-heavy target (76% of its information, the owner's suggestion), ~1 per 8 plies for search WDL | *(unfilled)* |

## Before the pilot

- The pilot's missing code: `--value-rows-per-game` (a per-row value mask). Step
  3's: the nested-prefix schedule, and the lazy window for more than ~2M rows. The
  fast-ply noise check before any playout-cap collect.
- #1441, #1444 and #1445 merged; the pilot runs from a detached worktree of main.
- The diagnosis's WDL-shape quantification read, if it has landed (it may sharpen
  the shape-preserving cell).
- Main's CLAUDE.md experiment rule: this record names the hypotheses (variance,
  shape), the control (P-anchor and the base), the deciding yardsticks (the pilot's
  Z-2 − Z-all and S-all − Z-all match differences; the curve's slope), the
  thresholds above and the recovery plan (the SL base is never overwritten; every
  cell is a new run directory).

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
