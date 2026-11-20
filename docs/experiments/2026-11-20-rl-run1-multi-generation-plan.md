# RL run 1: a measured multi-generation self-play run from the SL base (plan)

**Status:** DRAFT for owner approval. Nothing has been launched. No stage below
starts until the owner has approved this record and the
[decisions](#decisions-for-the-owner) are recorded in a follow-up commit
(CLAUDE.md "Experiments" rule 1).

**Why this record exists.** The owner set a new direction on 2026-11-20. In his
words:
- "i do want the RL loop to be the main path to strength";
- "i expect to be working on this project for years but i would hope to have a
  very strong engine within weeks";
- on 4,000 vs 16,000 games vs "1 hour": "they both seem equally made up to me … i
  wish we could be more rigorous instead of guessing";
- on search headroom: the SL base "gained at each more step and it should
  continue to do so as we increase the model; even lc0 and ceres do this at like
  1500 elo stronger base";
- on what is wrong: "i think the main problem is we dont know how to calibrate the
  sharpness and distribution of policy and value head to use the information. i
  think it shouldnt be that hard to get half of that 800 elo with like 100,000
  positions if we do it right";
- on why RL rather than more SL data: "search goes towards the holes where policy
  is worse so fixing those is very important".

So the one-generation experiment is no longer the goal. The plan is a staged
sequence:
- banked evidence (Stage 0);
- diagnostics that need no training (Stage 0c);
- a small-data calibration lab (Stage 1a);
- a dose and sims sweep (Stage 1b);
- the multi-generation run (Stage 2).

Doses come from measurements. Every number that is still a guess is listed under
[What is still a guess](#what-is-still-a-guess), with the measurement that would
replace it.

**Relation to the loop-exp1 record.** The
[loop-exp1 record](2026-11-20-loop-exp1-one-generation-from-sl-base.md) is
superseded as a standalone experiment. It is reused as the source of:
- the commands (the collection, `--collect-from` arms, planned updates);
- the collect gates G-C1–G-C11 and their threshold file;
- the model gates G-T1–G-T4 and the `gt1` / `gt2` verdict commands;
- the arena command and the seat-balanced pooling (`scripts/arena_seat_balanced_pool.py`);
- the net write-protection and the sha256 provenance rules.

"As loop-exp1" means that record's text together with #1437's additions:
- openings split step 0b and SL base copy step 0a;
- `gt1` / `gt2` with their expected-file counts;
- the pooling script's `--z`;
- the recalibrated G-C5 and G-T4 bounds.

#1437 must merge before launch.

Naming: "SL base" is the supervised starting net, sha256
`a39f900c944477c4ecf7f56cf0e1a6dd9486b67e5c2abb00973254e4bb6749b0`. The driver is
`scripts/simple_rl_loop.py` ([docs](../simple_rl_loop.md)).

## Decisions for the owner

Three decisions are the owner's. Everything else is set by a measurement, or by a
rule written below before any data.

| # | Decision | Recommended default | Alternative |
|---|---|---|---|
| O1 | **Time cap for Stage 1b's collection** (the lab's collection is about 20 min and needs no cap) | **One day of collection in total**, split equally between 64 and 256 sims: about 39,000 games at 64 sims, at the measured 3,284 games/h, plus about a quarter as many at 256 sims. Stage 1b in total is then about 42–52 GPU hours ([compute](#stage-1b-compute)) | Two days in total. This doubles the data behind every rung; arena time is unchanged |
| O2 | **Primary recipe (policy target, value target)** | **Decided by the lab's pre-written rule** ([Stage 1a](#stage-1a-calibration-lab)), over P (policy only), V (search value) and O (game outcome), crossed with policy-target temperatures | Name one recipe and skip that factor in the lab |
| O3 | **Sims per move for the collection: include the 256-sim factor in Stage 1b** | **Include it.** A 256-sim collection takes the same GPU time as the 64-sim one, and the two are compared at equal GPU time; the 256-sim teacher is about +272 Elo stronger (Stage 0) | 64 sims only. This halves the Stage 1b arms, and the whole collection goes to 64 sims |

## Stage 0: search ceiling — answered by banked matches

No new matches are run. The SL base's search gain is already banked, in
`~/chess-artifacts/operations/rl-claude-20261119/search-gain/results.jsonl`
(takeover record D1). The rows, verbatim:

| Label | Search vs search | Games | Score | Elo (95% CI) |
|---|---|---:|---:|---|
| `a39_s64_v_s1` | 64 vs 1 sim | 200 | 0.9925 | +848.7 (+690.7 to undefined) |
| `a39_s256_v_s64` | 256 vs 64 sims | 400 | 0.8275 | +272.4 (+243.1 to +305.3) |
| `a39_s800_v_s64` | 800 vs 64 sims | 200 | 0.9400 | +478.0 (+410.2 to +582.3) |

These matches were played in one orientation. A candidate-seat offset of up to
about 15 Elo would be negligible at these sizes; D4 round 2's pooled same-net read
was +6.4 (−4.2 to +17.1).

A 4× search is worth about +270 Elo to the SL base. So the question is not
whether there is anything to gain. It is whether training TRANSFERS the search's
strength into the net, in a form that still helps when the net searches.

## Stage 0b: raw-net transfer — banked, recomputed here

`~/chess-artifacts/operations/rl-claude-20261119/raw-net-1sim/` holds a diagnostic
banked on 2026-11-20. It was not pre-registered, and is marked NOT_A_CANDIDATE.
The banked loop nets were played against the SL base with BOTH sides at 1
simulation, on main `f51c51f20`:
- EVAL openings, seed 2026112050, `--sims 1`, no root noise, `--compile off`;
- seat-balanced, 800 games per orientation.

The pooled numbers below were recomputed from the banked game logs and result rows
with `scripts/arena_seat_balanced_pool.py` (`net_check: row_sha256`, both seats'
shas verified at pool time). The 64-sim column is from the earlier records.

| Net | 1 sim vs 1 sim: pooled Elo (95% CI, opening-clustered) | Seat offset | A/B correlation | 64 sims vs 64 sims |
|---|---|---|---:|---|
| P1 c12 | **+240.8** (+218.3 to +265.2) | −2.6 (−12.9 to +7.7) | +0.357 | −10.9 (−27.7 to +5.9) |
| F12 c12 | **+211.7** (+190.3 to +234.7) | +3.0 (−8.5 to +14.5) | +0.295 | −34.9 (D4) / −20.0 (own panel) |
| Q50 c12 | **+128.7** (+109.4 to +148.8) | −9.3 (−22.4 to +3.7) | +0.273 | −63.2 (−81.1 to −45.7) |
**The central fact: P1 c12's transfer curve.** P1 c12 played the SL base with
both sides at the same sims, seat-balanced. All rows were recomputed with the
pooling script:

| Sims (both sides) | Games | Pooled Elo (95% CI, opening-clustered) | Seat offset | A/B correlation |
|---:|---:|---|---|---:|
| 1 | 1,600 | **+240.8** (+218.3 to +265.2) | −2.6 | +0.357 |
| 4 | 800 | **+98.1** (+72.4 to +124.8) | −3.5 | +0.496 |
| 16 | 800 | **+20.0** (−2.2 to +42.3) | +7.8 | +0.287 |
| 64 | 800 | **−3.9** (−22.8 to +15.0) | −1.3 | +0.124 |

**What follows from it:**
- **Training moved the raw net a lot.** About 950 updates on about 2,300 games
  made the raw net 130–240 Elo stronger (the first table): roughly a quarter of the
  +849 64-vs-1 gap.
- **The gain decays with search and vanishes.** The advantage more than halves
  with each 4× of search. It reaches zero at 64 sims without going clearly
  negative.
- **The CIs here are wider than a hand pooling.** A hand pooling that treated the
  orientations as independent gave narrower intervals: 1 sim +241 (+221 to +262);
  16 sims +20.0 (+0.4 to +39.7), which excludes 0, where the clustered interval
  does not. The orientations of an opening are positively correlated (+0.12 to
  +0.50), so the opening-clustered CIs are the right ones, and sizing below allows
  for them.

**The central open question** is therefore no longer "does training move the net".
It is: **why does a +240-Elo raw-net gain not compose with search?**

### The softening test (running; result slot)

Labels `*_s64_pt1.5`, same bank:
- **Arm:** P1 c12 at 64 sims with search-time `policy_temp=1.5`, against the SL
  base at 1.0, seat-balanced.
- **Control:** the SL base at 1.5 against the SL base at 1.0.

> **[RESULT SLOT: the softening test. To be filled from the banked rows, pooled
> with `arena_seat_balanced_pool.py`, before this record is approved. Arm:
> ______ Elo (95% CI ______). Control: ______ Elo (95% CI ______). Reading
> chosen by the table below: ______.]**

Three readings, each written before the result:

| Reading | What the softening test shows | What it means | What the plan does |
|---|---|---|---|
| **R1: prior sharpness** | the arm gains (its CI above P1's 64-sim −3.9) and the control does not | training made the policy too sharp for search; the information is there but search cannot use it | the search-time prior temperature becomes a factor of every read, and of the lab's collection; the lab adds prior softening (`--train-target-temp`, `--policy-prior-mix`) |
| **R2: a general search-setting effect** | the arm and the control both gain, by similar amounts | `policy_temp=1.0` is not the best setting for either net; P1's own deficit is not about sharpness | the setting is retuned for both sides (a separate, recorded change to the yardstick, made before any lab read), and the composition question stays open |
| **R3: a shallow head start** | neither gains | the net learned what 64-sim search already finds (a head start that search catches up with), not new information | the loop must train on targets deeper than its own 64-sim result: more sims per move for the targets (the lab's 64 vs 256 factor), or value-target changes; more data of the same depth is not expected to help |

## The central question: why the raw-net gain does not compose with search

The softening test above is M1's first point. Each candidate mechanism below has
a test that needs no new training. These tests are Stage 0c. They run on banked nets (P1 c12 first, then F12 c12 and Q50 c12 where
stated), before any new collection.

**Common to every Stage 0c match:**
- seat-balanced against the SL base, 800 games (400 per orientation) on the
  200-line `diag` opening block;
- loop-exp1's arena flags (`--search-shape training --no-gumbel-noise
  --compile off`);
- pooled with the pooling script.

**Reading.** These are diagnostics, not registered claims. A mechanism is
"supported" when the named contrast moves by more than the 95% half-width of the
difference. That half-width is about 1.96 × √2 × 8.8 ≈ 24 Elo at 800 games per
arm, treating the arms as independent; that is conservative, since they share
openings.

| # | Mechanism | Test (no training) | Supported if |
|---|---|---|---|
| M1 | **The trained prior is too sharp for search.** Sequential halving trusts the prior's top-k, and a sharpened prior under-explores | P1 c12 at 64 sims with its search-time prior temperature raised on the candidate side only. `--cand-gumbel policy_temp=τ` (per-side knob of `arena_standard.py`) for τ ∈ {τ₀, 1.25 τ₀, 1.5 τ₀, 2 τ₀}, where τ₀ is the training shape's realised `policy_temp`, printed at startup | some τ > τ₀ beats τ₀ by more than 24 Elo |
| M2 | **Value drift misdirects search.** The trunk moved, so even a frozen value head reads differently | Head swap with the banked tool (`<bank>/headswap_arena.py`, takeover record D4): P1 c12's policy with the SL base's value, at 1 and at 64 sims. Already known for Q50: Q50 policy + base value reads +4.3 (−12.5 to +21.2) at 64 sims, so for Q50 the policy gain alone does not show at 64 sims | P1 policy + base value at 64 sims beats full P1 c12 (−10.9) by more than 24 Elo |
| M3 | **Redundancy.** The net learned what 64-sim search already finds, so its gain is a search saving rather than extra strength at 64 sims | Equal-strength sims ratio: P1 c12 at s sims (`--sims-candidate s --sims-reference 64`, s ∈ {16, 32, 48}) against the SL base at 64. Interpolate the s at which the pooled Elo is 0; the ratio is 64 / s | P1 matches the base at s < 64, with 64 / s CI above 1. A search saving is then real but does not add at equal sims |
| M4 | **Top-k / Gumbel root interaction.** A sharper prior changes which 16 children enter halving, or how σ ranks them | P1 c12 at 64 sims with `--cand-gumbel topk=32`, and separately with `--cand-gumbel c_scale=` at 2× the training value | either beats P1 at its defaults by more than 24 Elo |

**Compute:** about 8,000 64-sim games (M1 4 arms, M2 2 arms, M3 3 arms, M4 2
arms, plus 1-sim reads), about 4.4–6.3 GPU hours.

**What each outcome means for the plan:**
- **M1 supported:** the lab adds the search-time prior temperature as a factor, and
  Stage 2 collects with the measured τ.
- **M2 supported:** value targets and value calibration move to the front of the
  lab's grid.
- **M3 supported:** the gain is real but is a sims saving. The yardstick question
  becomes "strength at equal GPU time".
- **None supported:** the lab proceeds as designed, and the composition question
  stays open as a named gap.

Owner's hypothesis, which the lab is built to test: the problem is the calibration
of the policy's and value's sharpness and distribution, and "half of that 800 elo"
should be reachable from about 100,000 positions done right.

## The readout protocol: transfer curves

Every net is read the same way from now on.
- **The transfer curve.** The net plays the SL base seat-balanced with both sides
  at 1, 4, 16 and 64 sims, at the size each stage states, and pooled by the pooling
  script.
- **Success is judged at 64 sims** (and, later, on the tournament path). A 1-sim
  gain alone is not strength: P1 c12 is +241 at 1 sim and −3.9 at 64.
- **The curve's shape is the diagnostic.** For example, it can decay to 0, hold
  its level, or improve with sims.

| Sims | Cost per 800 games | Role |
|---:|---|---|
| 1 | about 1–2 min | transfer: did training move the raw net |
| 4, 16 | minutes | where the advantage is lost |
| 64 | 26–38 min (1,274–1,800 games/h) | **judged:** strength at the search the engine uses |

## Stage 1a: calibration lab

**Question.** On small collections, which choice of target search depth, policy-
target sharpness, value target and learning rate gives the strongest net at 64
sims? Its arms are chosen to discriminate between the three readings:
- **Target depth (64 vs 256 sims)** tests R3: deeper targets should help if the
  gain is a shallow head start.
- **Policy-target temperature** tests R1: softer targets should help if the prior
  is too sharp.
- **Value target** tests M2.

**Collections (equal GPU time, about 20 min each).** Both start from the SL base,
with loop-exp1's `$COLLECT`, at `--seed 2026112300`:
- `lab64/`: `--sims 64 --games 1040`, about 100,000 rows at 96.8 rows per game;
- `lab256/`: `--sims 256`, with `--games` = 1,040 × (256-sim rate / 64-sim rate)
  from P6, about 260 games and 25,000 rows if the rate is a quarter.

Both pass loop-exp1's collect gates (machinery gates only at 256 sims), and are
frozen once arms reference them. They share their first openings: same seed, same
cursor.

**Arms (36).** All arms reference their collection with `--collect-from`, with
fresh NAdamW, warmup 0 and a flat LR. The grid:

| Factor | Levels | Flags |
|---|---|---|
| Target sims (collection) | 64, 256, at equal GPU time | `lab64/` or `lab256/` |
| Policy-target temperature | 1, 2, 4 | `--train-target-temp T` (D3: stored targets 0.69 nats against the SL base's 1.12; T = 2 gives 1.32) |
| Value target | P: none; V: search value; O: game outcome | P `--w-wdl 0`; V `--w-wdl 1 --value-outcome-weight 0 --value-search-weight 1 --value-search-source survivor_q --value-draw-source net`; O `--w-wdl 1` (the driver's default outcome target) |
| Learning rate | 3e-6, 3e-5 | `--lr X --lr-floor X` |

Not in the grid:
- **A value-target temperature.** The driver has no such flag; it would need a code
  change before it could be a factor. The no-training analogue is the D4 value
  recalibration.
- **Multiple passes over the 100,000 rows** (`--sample-with-replacement
  --passes`). This is a candidate second lab factor.

**Training cost.** A `lab64/` arm is about 370 updates, and a `lab256/` arm about
95. All 36 arms take about 1.1–2.0 h.

**Reads, per arm (final model only): the transfer curve.** Each read is
seat-balanced against the SL base, with G-T2 and G-T4 as loop-exp1:

| Sims | Games | Block |
|---:|---:|---|
| 64 | 400 | `lab_64` (first 100 lines) |
| 16 | 400 | `lab_64` (first 100 lines) |
| 4 | 800 | `lab_1sim` (first 200 lines) |
| 1 | 1,600 | `lab_1sim` (400 lines) |

**Pre-written selection rule.**
1. The winner is the arm with the highest 64-sim pooled excess.
2. A tie is two arms within 15 Elo, about 1.2 SE at 400 games. Ties go, in order:
   - to the higher 16-sim excess;
   - then to the higher 1-sim excess;
   - then to the lower LR;
   - then to the lower T;
   - then to 64 target sims;
   - then in the order P, V, O.
3. The selection uses all 36 arms on shared openings. The winner's curse is handled
   by the confirmation below, never by the screen number.
4. **The lab also reports which reading its data favour,** without a test. The
   readings are the 256-vs-64 target contrast and the temperature contrast, each
   averaged over the other factors, compared at 64 and at 1 sim. This is
   descriptive, and it feeds the record's next step; it decides nothing on its own.

**Registered confirmation.** The winner plays the SL base at 64 sims:
- 2,000 games seat-balanced, on the 500-line `lab_confirm` block, which no other
  match uses;
- read with `--verdict-rule lb-gt-zero` at z 1.96;
- its 1-, 4- and 16-sim reads on the same block are reported, not judged.

The reading is pre-committed:
- **Lab success:** the lower 95% bound is above 0. Stage 1b uses the winner's
  policy temperature and value target, and centres its LR grid on the winner's LR.
- **Lab kill:** the upper bound is below 0.
- **Unread:** anything else.
- On a kill or unread, Stage 1b does not launch on this recipe. The next lab grid
  is chosen from Stage 0c's supported mechanisms (search-time τ in collection,
  prior mixing with `--policy-prior-mix`, value calibration), under an addendum.

**Power of the confirmation.** SE 5.6 Elo, assuming no orientation correlation.
At the +0.36 correlation measured at 1 sim it is about 6.5. The confirmation
detects +16 to +18 Elo with 80% power. The owner's hope (half of 800, about +400)
is far above that. One-twentieth of the raw-net gain (+12) is near the edge.

**Compute:**
- collect 0.7 h (two collections of about 20 min each);
- training 1.1–2.0 h;
- 64-sim reads, 36 × 400 = 14,400 games: 8.0–11.3 h;
- 16-sim reads: about a quarter of that, 2.0–2.8 h (a guess: arena cost assumed
  proportional to sims);
- 4- and 1-sim reads: about 1 h;
- confirmation: 1.1–1.6 h, plus its transfer curve.

Total about 14–19 GPU hours.

## Stage 1b: dose and sims sweep

Stage 1b launches only on a lab success. It asks how strength scales with GPU time
spent collecting, at 64 sims and at 256, for the lab's recipe and three LRs around
the lab's LR.

### Collections

Two collections from the SL base, each loop-exp1's `$COLLECT` with these changes:

| Collection | Sims | `--seed` | `--games` |
|---|---:|---|---|
| `c64/` | 64 | 2026112100 | half of O1 × the 64-sim rate measured on the day: about 39,000 at 3,284 games/h |
| `c256/` | 256 | 2026112100 | half of O1 × the 256-sim rate from pre-launch check P6: about 9,800 if the rate is a quarter |

- **Equal GPU time.** Each game count is rounded down to a multiple of 100. Both
  collections take the same GPU time, so the same fraction f of either collection
  is the same GPU time; the dose axis is GPU time.
- **Shared openings.** Same seed, same cursor: `c256/`'s lines are the first lines
  `c64/` plays too, so the two collections are paired on openings.
- **Size.** 96.8 rows per game. After the 5% holdout, at batch 256, N ≈ 14,200
  updates for `c64/` and N ≈ 3,550 for `c256/`. Each N is computed exactly by
  loop-exp1's step-2 snippet and recorded.
- **Disk.** About 0.26 MB per game: about 13 GB for both collections, plus models.
- **Gates.** Loop-exp1's collect gates. Its SL-base bands were measured at 64 sims,
  so on `c256/` only the machinery gates are gated (G-C1, G-C2, G-C3 ≥ +0.50, G-C7,
  G-C10); the rest are recorded. A failed gate voids that collection's arms.

### Arms (6)

The lab's recipe (policy-target temperature and value target), at three learning
rates — the lab's LR ÷ 3, the lab's LR, and the lab's LR × 3 — on each collection.
- Fresh NAdamW, warmup 0, flat LR.
- Probe KL and |ΔQ| are recorded as descriptors only
  ([why](#why-no-sizing-from-loss-curves)).

### Dose rungs

```bash
N=<planned updates from the arm's sealed collect>
SNAPS=$((N/64)),$((N/32)),$((N/16)),$((N/8)),$((N/4)),$((N/2))   # doses 1/64 … 1/2; the final model is dose 1
```

This gives 7 rungs, with x = log₂(fraction) ∈ {−6, …, 0}; rung x is the same GPU
time on both collections.

**A snapshot is a valid dose rung only because the schedule is exact-once,
shuffled and at a flat LR.** The rows seen by update N/64 are a uniform random
subset of the collection, at the LR a shorter run would use.

**Gates.** Loop-exp1's G-T1–G-T4 run at every rung. A failing snapshot ends its arm
there; the trend test then uses the earlier rungs, at least three of them, and
fewer than three reads "gate failed".

### Matches

Each rung of each arm is read with its transfer curve against the SL base,
seat-balanced, on the rung's own 100-line block:
- **64 sims (judged):** 400 games, two runs of 200, with `--openings-fen`,
  `--games 200` and 100 rows, so every row is played once;
- **16, 4 and 1 sims:** 400 games each, with the same command and `--sims 16`,
  `--sims 4`, `--sims 1`. These are reported, not judged.

The 64-sim command is loop-exp1's, with `--openings-fen` in place of `--openings`
and `--opening-plies`:

```bash
TB="$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6"
COMMON="--sims 64 --games 200 --mode matched_sims --search-shape training --no-gumbel-noise \
  --openings-fen $R/openings/blocks/rung_$((X+6)).fen --max-plies 300 --syzygy $TB \
  --compile off --seed 2026112101"
# $M = the rung's model (snapshot or final), $T = a tag such as c64_lr1_x-6
PYTHONPATH=. python -u scripts/arena_standard.py $COMMON --candidate $M --reference $R/sl_base.pt \
  --label s1_${T}_A --games-out $R/arenas/s1_${T}_A.games.jsonl --pgn-out $R/arenas/s1_${T}_A.pgn \
  --out $R/arenas/results.jsonl
PYTHONPATH=. python -u scripts/arena_standard.py $COMMON --candidate $R/sl_base.pt --reference $M \
  --label s1_${T}_B --games-out $R/arenas/s1_${T}_B.games.jsonl --pgn-out $R/arenas/s1_${T}_B.pgn \
  --out $R/arenas/results.jsonl
PYTHONPATH=. python -u scripts/arena_seat_balanced_pool.py --arm $M \
  --a-games $R/arenas/s1_${T}_A.games.jsonl --b-games $R/arenas/s1_${T}_B.games.jsonl \
  --a-result $R/arenas/results.jsonl --b-result $R/arenas/results.jsonl \
  --a-label s1_${T}_A --b-label s1_${T}_B --out $R/arenas/s1_${T}_pooled.json
```

Every arena record meets loop-exp1's "Required in every arena record" list. Every
net that plays is write-protected until its pooling is recorded.

### The registered test: trend of 64-sim Elo on log₂(dose)

For each arm, rung k gives three numbers:
- x_k, the dose;
- y_k, the pooled excess `result.pooled.excess` at 64 sims;
- s_k, its opening-clustered SE `result.pooled.se`, over 100 clusters.

**Estimator.** Weighted least squares, w_k = 1 / s_k²:
- x̄ = Σ w x / Σ w, and ȳ likewise;
- b = Σ w (x − x̄)(y − ȳ) / Σ w (x − x̄)²;
- SE(b) = 1 / √(Σ w (x − x̄)²), and z = b / SE(b);
- fitted excess at dose 1: ŷ₀ = ȳ − b x̄, with SE √(1 / Σ w + x̄² SE(b)²).

In Elo, the slope is about 694.9 × b per doubling (1600 / ln 10 is the slope of the
Elo map at x = 0).

**Test.** One-sided at 2.5%, family-wise over the m arms tested (Bonferroni).
- z_crit = Φ⁻¹(1 − 0.025 / m): 2.638 for six arms.
- The null is "Elo does not change with dose", at any level.
- A non-monotone response is read as its average trend. A departure from
  linearity costs power, not validity.

The same fit on the 1-, 4- and 16-sim reads is reported as the transfer trend.
It is not registered.

```bash
trend() {  # $1 = m (arms in the family); then x=pooled.json for each rung played
  python3 -c 'import json, math, sys
from statistics import NormalDist
m = int(sys.argv[1]); pts = []
for a in sys.argv[2:]:
    x, f = a.split("=", 1); p = json.load(open(f))["result"]["pooled"]
    pts.append((float(x), p["excess"], p["se"]))
if len(pts) < 3 or any(s is None or not s > 0 for _, _, s in pts):
    print("fewer than 3 rungs, or an unavailable SE: no trend test"); sys.exit(4)
w = [1 / s ** 2 for _, _, s in pts]; sw = sum(w)
xb = sum(wi * x for wi, (x, _, _) in zip(w, pts)) / sw
yb = sum(wi * y for wi, (_, y, _) in zip(w, pts)) / sw
sxx = sum(wi * (x - xb) ** 2 for wi, (x, _, _) in zip(w, pts))
b = sum(wi * (x - xb) * (y - yb) for wi, (x, y, _) in zip(w, pts)) / sxx
se = 1 / math.sqrt(sxx); zc = NormalDist().inv_cdf(1 - 0.025 / m); k = 1600 / math.log(10)
y0 = yb - b * xb; se0 = math.sqrt(1 / sw + xb ** 2 * se ** 2)
out = {"rungs": len(pts), "slope_excess": b, "se": se, "z": b / se, "z_crit": zc,
       "slope_elo_per_doubling": k * b, "se_elo_per_doubling": k * se,
       "fitted_excess_dose1": y0, "fitted_se_dose1": se0, "passed": b / se > zc}
print(json.dumps(out)); sys.exit(0 if out["passed"] else 3)' "$@"
  echo "exit=$?"
}
T=c64_lr1
trend 6 -6=$R/arenas/s1_${T}_x-6_pooled.json -5=$R/arenas/s1_${T}_x-5_pooled.json \
  -4=$R/arenas/s1_${T}_x-4_pooled.json -3=$R/arenas/s1_${T}_x-3_pooled.json \
  -2=$R/arenas/s1_${T}_x-2_pooled.json -1=$R/arenas/s1_${T}_x-1_pooled.json 0=$R/arenas/s1_${T}_x0_pooled.json
```

Exit codes: 0 = pass, 3 = not passed, 4 = no test.

The function was run verbatim on seven synthetic rungs, with a slope of 0.0072
excess per doubling and an SE of 0.01796 per rung (a 400-game rung):
- it printed an SE of 2.36 Elo per doubling and z = 2.12;
- that is not a pass at m = 16 (`exit=3`);
- it recovered the fitted dose-1 excess of 0.0100 (SE 0.0122).

An earlier form of the function returned z = 2.835 on five synthetic rungs, and
printed `exit=4` when given two.

### Power

The per-pair SD is 0.254, from the banked matches.
- **Per rung.** A 400-game rung has P = 200 pairs, so s = 0.254 / √200 = 0.0180
  excess (12.5 Elo) with uncorrelated orientations. A correlation r inflates this by
  √(1 + r). Stage 0b measured r = +0.27 to +0.50 at 1 sim, which is up to ×1.22. The
  test uses the realised clustered SE.
- **Slope.** With seven equal rungs, Σ (x − x̄)² = 28, so SE(b) = s / √28 = 2.36 Elo
  per doubling at r = 0.
- **Detectable effect.** The 80%-power effect is (z_crit + 0.842) × SE(b). At six
  arms that is 8.2 Elo per doubling at r = 0, and about 10 at r = +0.5.

The games per rung needed for 80% power are
2 × ((z_crit + 0.842) × 0.254 × 694.9 / (δ × √R_x))², with R_x = Σ (x − x̄)²:

| Rungs (R_x) | Arms tested (m) | δ = +10 Elo per doubling | δ = +25 Elo per doubling |
|---|---:|---:|---:|
| 5 (10) | 1 | 489 | 78 |
| 7 (28) | 1 | 175 | 28 |
| 7 (28) | 6 | 269 | 43 |
| 7 (28) | 16 | 321 | 51 |

**Design: 7 rungs × 400 games at 64 sims.** That is above the 269 games needed at
+10 Elo per doubling, which leaves room for the correlation inflation. 400 games
is also the floor for the normal approximation and for opening coverage. A smaller
true trend reads "not passed", which means unread, not "no effect".

### Stage 1b output: the pre-written rule

The rule is applied in this order:

1. **Eligible arms:** those whose 64-sim trend passes at the family-wise z_crit,
   using only rungs that passed every gate.
2. **LR and sims:** the eligible arm with the highest `fitted_excess_dose1`, which
   is the most Elo for the same collection GPU time.
   - A tie is two arms within 5 Elo.
   - Ties go first to the steeper slope, then to the lower LR, then to 64 sims.
3. **Rows per generation:** the smallest dose rung of the chosen arm whose 64-sim
   pooled 95% lower bound is above 0. Ties go to the smaller dose.
   - If no rung's lower bound is above 0 but the fitted dose-1 excess is positive,
     use dose 1.
   - Otherwise the reading is "trend without a gain": stop and report.
4. **Stage 2 starts** from the chosen arm's model at the chosen rung, with that arm's
   recipe, LR and sims. Generation size = the chosen dose's games.
5. **Kill / stop:** if no arm passes, Stage 2 does not launch.
   - **KILL** when every tested arm's one-sided 97.5% upper bound on the slope is
     below +5 Elo per doubling.
   - **"Unread, insufficient evidence"** otherwise.
   - The next lever is named from Stage 0c's supported mechanisms, under a new
     record.

The per-rung bound in rule 3 is a decision threshold, not a registered claim. The
claim Stage 1b registers is the trend test, and Stage 2's readouts re-measure on
fresh openings.

### Stage 1b compute

**Rates used:**
- collect: 3,284 games/h at 64 sims; 256 sims measured in P6;
- training: 0.47–0.84 s per update (F12 record);
- 64-sim arena: 1,274–1,800 games/h (D4's 800-game arms);
- 1-sim arena: about 800 games per 1–2 min (Stage 0b);
- one GPU job at a time.

| Item | 6 arms (default) | 3 arms (64 sims only) |
|---|---:|---:|
| Collect (O1 = one day in total) | 24 h | 24 h |
| Train `c64/` arms, ≈ 14,200 updates each | 5.6–9.9 h (3 arms) | 5.6–9.9 h |
| Train `c256/` arms, ≈ 3,550 updates each | 1.4–2.5 h (3 arms) | — |
| Model gates | ≈ 1 h | ≈ 0.5 h |
| 64-sim reads, 7 rungs × 400 games per arm | 9.3–13.2 h (16,800 games) | 4.7–6.6 h |
| 16-, 4- and 1-sim reads, 7 rungs × 400 games each per arm | ≈ 3–4.5 h | ≈ 1.5–2.3 h |
| **Total** | **≈ 44–56 h** | **≈ 36–43 h** |

## Openings: disjoint blocks

The EVAL split (4,000 lines) is cut once, in `eval.jsonl` order:

```bash
mkdir -p $R/openings/blocks
python3 -c 'import json, sys
d = sys.argv[1]; lines = [json.loads(l)["line"] for l in open(d + "/eval.jsonl")]
assert len(lines) == 4000
cuts = [("lab_64", 0, 200), ("lab_1sim", 200, 600), ("lab_confirm", 600, 1100), ("diag", 1100, 1300)]
cuts += [(f"rung_{k}", 1300 + 100 * k, 1400 + 100 * k) for k in range(7)]
cuts += [("readout_base", 2000, 2500), ("readout_prev", 2500, 3000), ("confirm", 3000, 3500), ("anchor", 3500, 4000)]
for name, a, b in cuts:
    open(f"{d}/blocks/{name}.fen", "w").write("".join(x + "\n" for x in lines[a:b]))
print(len(cuts), "blocks")' $R/openings
```

| Block | Lines | Used by |
|---|---:|---|
| `lab_64` / `lab_1sim` | 200 / 400 | Stage 1a screen reads |
| `lab_confirm` | 500 | Stage 1a confirmation only |
| `diag` | 200 | Stage 0c |
| `rung_0` … `rung_6` | 7 × 100 | Stage 1b rung x uses `rung_{x+6}` in every arm |
| `readout_base` / `readout_prev` | 500 / 500 | Stage 2 readouts |
| `confirm` | 500 | Stage 2's final confirmation only |
| `anchor` | 500 | the Stockfish anchor |

An earlier version of this cutter, with the same structure but different blocks,
was run verbatim on the banked split. In the arena's FEN loader, every block loaded
as unique rows, each with its move history. The cutter is re-run and its sha256
recorded at P2.

**Why disjoint and not shared.** One shared set would correlate the Stage 1b rungs
through the openings, by some correlation ρ.
- That would cut the slope's variance by about (1 − ρ).
- But it would need an opening-level slope estimator, which is not written. The
  plain weighted fit assumes independent rungs, so with shared openings its SE would
  be wrong.
- Disjoint blocks make the registered estimator exactly the pooling output plus a
  weighted fit. That is conservative: the (1 − ρ) gain is given up, and ρ is
  unmeasured.

Arms share the block at each rung, so comparisons between arms at a rung are paired
on openings.

## Stage 2: the multi-generation run

Stage 2 launches only on a Stage 1b pass. It takes the recipe, LR, sims, starting
model and generation size from the
[rule](#stage-1b-output-the-pre-written-rule).

### Driver settings

One run dir, `run --generations G`.

**Start.** `--init-checkpoint` = the Stage 1b model the rule chose.
- It is copied into the run dir, sha-verified and write-protected.
- Whether it is a weights-only snapshot or a final model, the first phase starts a
  fresh optimizer.
- Stage 2's generation 1 is the loop's second generation.

**Generation size.** The chosen dose's games, at the chosen sims, concurrency 128,
and Stage 1b's collect flags.

**Phase: `--presentations 1`.** Each phase trains exactly once on its own
generation's rows, which is the regime Stage 1 measured. `--presentations 2` and
`--sample-with-replacement --window` were not measured, and are not registered.

**Optimizer: `--optimizer-state fresh`** each generation, at the Stage 1b LR,
warmup 0, flat. `--optimizer-state continue` carries Adam moments across
generations. The driver supports it: it restores the optimizer, scheduler and
step, and refuses a changed `--lr` without `--reapply-optimizer-hparams`. But it
was not measured, and is not registered.

**Openings.** `--opening-sampling cycle` with `--seed 2026112200`, an independent
permutation of TRAIN.
- Stage 1 played up to about 26% of TRAIN with other seeds. That share of Stage 2's
  lines was also played in Stage 1, by the SL base.
- This is counted, not avoided: the driver cannot continue another run dir's
  cursor.

**Holdout / probe:** `--holdout-frac 0.05`, `--probe-rows 2048`. Probe drift is
measured against the previous generation.

**Evaluation:** `--no-eval`. Readouts run separately (below).

### Banked per generation (kept for the whole run)

- the collect directory;
- the Trainer checkpoint, with optimizer state;
- `presented.npz` and `metrics.json`;
- the gate outputs;
- each collect's content digest.

Disk use is about 0.26 MB per 64-sim game plus about 0.7 GB per checkpoint.
Nothing is pruned.

### Per-generation gates

Loop-exp1's bands describe SL-base collects, so in Stage 2 they split three ways.

**Gated (machinery).** A failure means the collect cannot be trusted:
- G-C1 = 0;
- G-C2 equal;
- G-C3 ≥ +0.30 on both root classes;
- G-C7 ≥ 80;
- G-C10 ≤ 0.05.

**Recorded, with a drift alarm:** G-C4, G-C5, G-C6, G-C8 and G-C9 against
loop-exp1's bands. Two consecutive generations outside a band pause the run for a
report. A pause voids nothing.

**Model gates:**
- G-T2 against the previous generation: KL > 1.5 or |ΔQ| > 0.4 fails.
- G-T1: mean |Q| ≥ 0.33 and P(draw) in 0.45–0.60. These bands are guesses.
- G-T4, as loop-exp1.

**On a gate failure:**
- **Collect gate:** one redo of that collect (`collect --redo --generation g`, which
  replays the same lines from the sealed cursor). If the redo fails too, the run
  stops at generation g − 1.
- **Model gate:** the run stops at g − 1, and the failed model is kept.
- **Restart:** either kind of stop needs an addendum to restart.

### Readouts

Every k generations, the newest net gets two transfer curves (1, 4, 16 and 64
sims). Both are seat-balanced, and each is read on its own block:
- **R-base**, against the SL base, on `readout_base`;
- **R-prev**, against the previous readout net, on `readout_prev`. At the first
  readout, that is Stage 2's start model.

The 64-sim reads (R-base-64, R-prev-64) are 2,000 games each, and so are the 1-sim
reads. The 4- and 16-sim reads are 800 games each.

**How the readouts are judged.** The stop rule and the deliverable are judged at
64 sims only. The rest of each curve is reported beside it: the shape of the curve
is the composition evidence. A 1-sim gain alone is not strength.

**Size.** A 2,000-game match has SE 5.6 Elo, or about 6.5 at the orientation
correlation measured at 1 sim. Its 80%-power effect is about 16–18 Elo. After Stage
1b, the realised clustered SEs replace these figures.

**Cadence.** k is the smallest integer with k × g₁ ≥ 18 Elo, clamped to 1–4. Here
g₁ is Stage 1b's point estimate of the chosen rung's 64-sim Elo against the SL base,
taken as the per-generation gain.

**Two anchors that do not move.** Anchored Elo is anchor-dependent here, by up to
82 Elo in one recorded three-contrast design (experiment ledger).
1. **The SL base**, through R-base-64 and R-base-1.
2. **Stockfish at fixed nodes**, through `scripts/match_vs_uci.py` with our engine
   as `python -m chess_anti_engine.uci --checkpoint <net>`.
   - The node count is fixed before Stage 2, where the SL base scores 25–75%.
   - Each readout plays 1,000 games, with `--nodes-b <fixed> --enforce-nodes-b
     --openings $R/openings/blocks/anchor.fen`.
   - The exact UCI command and per-move budget are a pre-Stage-2 check.

The transitivity check is that (net vs SF) − (SL base vs SF) agrees with
R-base-64 within the combined CI. A disagreement is reported, not averaged away.

### Stop rule (pre-committed)

The run stops at the first of:
1. two consecutive readouts with an R-prev-64 pooled upper 95% bound below 0
   (regression);
2. three consecutive readouts with R-prev-64 unread (plateau);
3. a gate failure that one re-collect does not clear;
4. the owner's time cap.

**Deliverable.** The deliverable is the readout net with the highest R-base-64
estimate, among those whose lower bound is above 0.
- This is a selection, so it carries winner's curse.
- The claim is re-measured once: 2,000 games at 64 sims, seat-balanced, against
  the SL base on `confirm`, read at 95%. A 1-sim read on the same block is reported.

### Recovery

- Every generation's model and Trainer state are kept.
- Resuming from the last passing generation g uses `--redo --redo-downstream` of
  generation g + 1's collect (later stages move to `superseded/`). Alternatively,
  start a new run dir with `--init-checkpoint` = generation g's model.
- The SL base, the live checkout, the production recipe and every earlier bank are
  untouched. Nothing is promoted to production.

### Compute per generation and per readout

With C the generation size in games:
- **A generation:** C / rate hours of collect, plus
  (C × 96.8 × 0.95 / 256) × 0.47–0.84 s of training, plus a few minutes of gates.
  - At 64 sims and C = 1,200: about 0.5 h, so about 40 generations a day before
    readouts.
  - At C = 39,000: about 14–15 h.
- **A readout:**
  - 4,000 64-sim games: 2.2–3.1 h;
  - 3,200 games at 4 and 16 sims: about 0.5–0.8 h (16 sims assumed a quarter of the
    64-sim cost);
  - 4,000 1-sim games: about 5–10 min;
  - the Stockfish anchor: 1,000 games, at a rate not yet measured.

### What success means at one week

**The one-week target, pre-committed:** within 7 days of Stage 2's start, the
deliverable's confirmation match reads a 95% lower bound above 0 with a point
estimate of at least +50 Elo at 64 sims.
- Short of +50, with LB > 0: "gain confirmed, below target".
- LB not above 0: "not shown at one week".

+50 Elo is a guess. Before Stage 2, Stage 1b replaces it with a projection:
generations in a week × g₁, discounted by half, recorded at the Stage 1b readout.

## Why no sizing from loss curves

The experiment ledger (2026-08-23) records three cases where a fit or audit gain
did not transfer to Elo:
- F-only: held-out CE fell 0.043 nats, for 0 cp;
- absorption: 11.6× the fit, for 0 Elo;
- v020: +3.1 cp on the deep-SF audit, for −9.56 Elo (−29.59 to +10.41) at 800
  games.

The ft595 fine-tune traded Elo for fit: −41.0 (−71.3 to −11.4). Stage 0b adds a
fourth pattern, at the level of play: a large raw-net (1-sim) gain that does not
show at 64 sims.

So every dose, LR, sims and recipe choice here is made on 64-sim arena Elo against
a fixed opponent. 1-sim Elo, loss, CE, probe KL and |ΔQ| are recorded, and are used
only as transfer descriptors or gross gates.

## What is still a guess

| Number | Where it is used | The measurement that replaces it |
|---|---|---|
| 3,284 games/h at 64 sims | collection sizes, compute | Stage 1a's collect (1,040 games), then Stage 1b's |
| The 256-sim rate (a quarter of 64-sim assumed) | `c256/` size, compute | pre-launch check P6 |
| 96.8 rows per game | updates, training time | the sealed collects (N is computed exactly) |
| 0.47–0.84 s per update | training time | Stage 1a's arms |
| 1,274–1,800 64-sim arena games/h; 800 1-sim games per 1–2 min | arena time | Stage 0c's first match |
| Per-pair SD 0.254; orientation correlation at 64 sims | all sizing | Stage 0c's and Stage 1a's realised clustered SEs (+0.27 to +0.50 at 1 sim, Stage 0b) |
| +24 Elo as Stage 0c's "supported" threshold | diagnostics | the realised SEs of the paired arms |
| The lab grid (T 1 / 2 / 4; P / V / O; LR 3e-6 / 3e-5) | Stage 1a | the lab itself; an optimum at an edge points the next grid outward |
| About 100,000 positions as the lab's size | Stage 1a | Stage 1b's dose trend |
| +8–10 Elo per doubling as the effect worth detecting | Stage 1b games per rung | none: a statement of what is worth detecting |
| Linear in log₂ dose | Stage 1b's trend test | the rung estimates (curvature is visible) |
| That a Stage 1b rung's gain (g₁) repeats each generation | readout cadence, the one-week projection | Stage 2's first readouts |
| `--presentations 1`, a fresh optimizer each generation | Stage 2 | a later comparison |
| +50 Elo at one week | success line | Stage 1b's projection |
| The Stockfish node count | the second anchor | the pre-Stage-2 calibration |
| Stage 2's G-T1 bands; the G-T2 bounds 1.5 / 0.4 | per-generation gates | Stage 1a and 1b's per-arm audit and probe reads |
| The arena's 64 sims as the yardstick | every judged match | not replaced: fixed so that matches stay comparable |
| The Gumbel noise and move-temperature schedules | collection | not measured here |
| The one-week horizon and Stage 1b's one-day cap | budgets | owner (O1) |

## Pre-launch checks

- **P1.** #1437 is merged. Every stage runs from a detached worktree at a pinned
  main sha, recorded here, with the extensions rebuilt.
- **P2.** The openings split is reproduced and checked (loop-exp1 step 0b). The
  blocks are cut, and their sha256 recorded.
- **P3.** The SL base is copied into the run dir and write-protected (loop-exp1 step
  0a).
- **P4.** Free disk: about 2 GB for Stage 1a, about 20 GB for Stage 1b, plus
  Stage 2's estimate.
- **P5.** `trend`, `gt1` and `gt2` are re-run on the dry run's banked files and on
  synthetic files, and print the outputs stated.
- **P6.** A 128-game collect at 256 sims (`$COLLECT` with `--games 128`, in a
  scratch run dir) measures the 256-sim rate and passes the machinery gates.
- **P7.** The softening test's result slot is filled from its banked rows,
  pooled with the pooling script, and the reading is named from the table. Done
  before approval.

## Confounds

- **The P1 c12 ladder shares its openings across sims levels.** Every run used
  seed 2026112050 on `eval.pgn`. The 4-, 16- and 64-sim runs played the same 200
  openings, and those are 200 of the 1-sim run's 400 (checked from the game
  logs). So the curve's points are paired on openings and positively correlated:
  differences between levels are estimated more precisely than the separate CIs
  suggest, and the points are not independent.
- **Shared collections.** Each collection's arms share one sample, so every arm
  verdict is conditional on it.
- **Snapshots, not independent fits.** Stage 1b's rungs are snapshots along one
  trajectory per arm. Their matches are on disjoint openings, but the models are
  not independent fits.
- **Paired collections.** `c64/` and `c256/` share their first openings, which
  pairs the comparison and does not bias it.
- **Stage 0b and 0c are on banked nets** that were trained under the old loop on
  F12's rows. Their mechanisms may not be the same as a new arm's.
- **Stage 2's start.** Stage 2's first generation starts from a net trained on part
  of a Stage 1b collection.
- **Inherited from loop-exp1:** the yardstick is not F12's instrument, and loop-exp1's
  confounds (twofold-repetition draws, the same-coloured-bishops gap) apply.
