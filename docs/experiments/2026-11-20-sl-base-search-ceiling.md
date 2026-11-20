# 2026-11-20: SL base search ceiling — how much stronger does more search make the SL base?

**Status:** measurement only (no training, no collection). Registered before any game
was played; results appended below the line marked RESULTS.

**What it is for.** Sizing the first loop experiment
([loop experiment 1](2026-11-20-loop-exp1-one-generation-from-sl-base.md)) from a
measurement instead of a guessed dose. It decides nothing about training: no arm,
dose or gate is chosen by it.

## Question

How much stronger does the SL base (sha256 `a39f900c944477c4…`) play when it searches
128, 256 or 512 sims per move instead of the loop's 64? A net that perfectly
distilled its own 64-sim search into its policy and value would, searched again at 64
sims, play roughly like the base at more sims. So these gaps size what one
self-play generation could at most hope to recover. That reading is a heuristic,
not a bound; see "What this does and does not bound" below.

## Already banked (read before running)

Searched `~/chess-artifacts/operations/rl-claude-20261119/` (rl-audit, headswap,
p1-policy-only, value-recalibration, search-gain, loop-smoke-gpu and the others), the
experiment records on main and the arena results log. Same-net, different-sims
matches of this net exist only in D1 of the
[2026-11-19 record](2026-11-19-rl-takeover-diagnostics-and-policy-only-ablation.md)
(`search-gain/results.jsonl`):

| Candidate vs reference sims | Games | Score | Elo (95% CI) | Settings |
|---|---:|---:|---|---|
| 64 vs 1 | 200 | 0.9925 | +849 (lower bound +691) | candidate = high sims in every game |
| 256 vs 64 | 400 | 0.8275 | +272 (+243 to +305) | same |
| 800 vs 64 | 200 | 0.94 | +478 (+410 to +582) | same |

Common settings: `matched_sims`, `--search-shape training --no-gumbel-noise`,
tablebases on, `--compile off`, seed 20261119, 128 games in flight, the default
147k-line opening book (NOT the loop's EVAL split), git `df55413a6`. None is
seat-balanced: the high-sims side was always the candidate seat. D4 round 2 found no
candidate-seat offset (+6.4, −4.2 to +17.1), small against these effects. No
`scripts/elo_vs_sims.py` output for this net was found.

So 256 vs 64 is roughly answered (±31 Elo) on a different opening set, and 128 vs 64
and 512 vs 64 are not. All three are run here on one ruler — the loop's EVAL
openings, seat-balanced, 1,600 games each — so that the dose curve is read from one
protocol. The banked 256 vs 64 is a cross-check.

## Design

- **Nets:** the SL base on both seats. The high-sims seat uses
  `search-ceiling/a39_hisims_seat.pt`, a byte copy of `a39.pt` (identical sha256),
  only so that the pooling script's `--arm` path names the high-sims seat; the arena
  and the pool both check the sha256s.
- **Comparisons:** 128 vs 64, 256 vs 64, 512 vs 64 (512 last, dropped only if the
  first two show it would exceed about 3 h).
- **Seat balance:** each comparison is two runs on the same seed, so the same 400
  opening pairs are played in both orientations:
  run A = high sims as candidate (`--sims-candidate N --sims-reference 64`),
  run B = high sims as reference (`--sims-candidate 64 --sims-reference N`).
  `scripts/arena_seat_balanced_pool.py` pools them; its settings check requires
  every `*_candidate` header key of A to equal B's `*_reference` key, so the sims
  fields must swap with the seats, which they do.
- **Settings (as loop experiment 1's arena):** `--mode matched_sims --search-shape
  training --no-gumbel-noise --compile off`, EVAL openings `loop-smoke-gpu/openings/eval.pgn`
  (sha256 `067675c264fb6b88…`), `--opening-plies 16` (its `eval_max_plies`),
  `--max-plies 300`, production Syzygy pair, `--seed 2026112070`, 128 games in
  flight, one arena process at a time on an otherwise idle GPU.
- **Games:** 800 per orientation, 1,600 per comparison (400 opening clusters).

```bash
R=~/chess-artifacts/operations/rl-claude-20261119; C=$R/search-ceiling
TB="$HOME/projects/chess/data/syzygy_3-4-5:$HOME/projects/chess/data/syzygy_6"
COMMON="--mode matched_sims --search-shape training --no-gumbel-noise --compile off \
  --games 800 --openings $R/loop-smoke-gpu/openings/eval.pgn --opening-plies 16 \
  --max-plies 300 --syzygy $TB --seed 2026112070 --device cuda --out $C/results.jsonl"
# for N in 128 256 512:
PYTHONPATH=. python -u scripts/arena_standard.py $COMMON \
  --candidate $C/a39_hisims_seat.pt --reference $R/a39.pt \
  --sims-candidate $N --sims-reference 64 --label ceiling_s${N}_A \
  --games-out $C/ceiling_s${N}_A.games.jsonl --pgn-out $C/ceiling_s${N}_A.pgn
PYTHONPATH=. python -u scripts/arena_standard.py $COMMON \
  --candidate $R/a39.pt --reference $C/a39_hisims_seat.pt \
  --sims-candidate 64 --sims-reference $N --label ceiling_s${N}_B \
  --games-out $C/ceiling_s${N}_B.games.jsonl --pgn-out $C/ceiling_s${N}_B.pgn
PYTHONPATH=. python -u scripts/arena_seat_balanced_pool.py --arm $C/a39_hisims_seat.pt \
  --a-games $C/ceiling_s${N}_A.games.jsonl --b-games $C/ceiling_s${N}_B.games.jsonl \
  --a-result $C/results.jsonl --b-result $C/results.jsonl \
  --a-label ceiling_s${N}_A --b-label ceiling_s${N}_B --out $C/ceiling_s${N}_pooled.json
```

**The one yardstick:** the pooled seat-balanced Elo of N sims over 64 sims, with its
95% interval, from `ceiling_sN_pooled.json`.

**Expected precision.** Clustered by opening (400 clusters). At a 50% score a
same-strength pair SD of 0.254 gives an SE of about 0.009 on the excess, roughly ±12
Elo at 95%. At the banked 256-vs-64 score (0.83) the score SD is smaller but each
unit of excess is worth more Elo, so the half-width is about ±15–25 Elo.

**Throughput.** Each run's wall time is logged by its driver script; games/h is
reported per comparison (each run mixes N-sim and 64-sim moves).

**How the result is used.** Sizing context for the loop experiment (how large a
one-generation gain could plausibly be, and how many games a match needs to see a
fraction of it). It is not a verdict on any arm or dose.

## Dose → movement already measured (context, banked)

Policy drift of the P1 (policy-only) and F12 (policy + value) twelve-cycle loops
from the SL base, on the fixed 4,735 held-out cycle-2 rows
(`p1-policy-only/logs/eval_per_cycle.json`, fields `kl_base_model` and
`mean_abs_dQ`; updates from each cycle's training log, batch 256 at lr 3e-6).

| Cycle | Cumulative updates | P1 KL | P1 \|ΔQ\| | F12 KL | F12 \|ΔQ\| |
|---:|---:|---:|---:|---:|---:|
| 1 | 69 | 0.038 | 0.020 | 0.039 | 0.075 |
| 2 | 145 | 0.094 | 0.034 | 0.088 | 0.079 |
| 3 | 223 | 0.130 | 0.048 | 0.122 | 0.098 |
| 4 | 303 | 0.160 | 0.053 | 0.151 | 0.111 |
| 6 | 460 | 0.215 | 0.048 | 0.198 | 0.106 |
| 8 | 621 | 0.250 | 0.063 | 0.238 | 0.112 |
| 10 | 785 | 0.275 | 0.061 | 0.266 | 0.109 |
| 12 | 948 | 0.299 | 0.066 | 0.288 | 0.107 |

Playing results at the end of those loops, against the SL base, for scale: P1 c12
−10.9 (−27.7 to +5.9) ([2026-11-19 record](2026-11-19-rl-takeover-diagnostics-and-policy-only-ablation.md));
F12 c12 47.125% (paired 95% CI 44.51–49.74%, about −20 Elo)
([2026-11-18 record](2026-11-18-fresh-current-policy-a39-twelve-cycle-baseline.md)).
Policy KL grows sublinearly and flattens: from cycle 1 to 6 it rises about as
updates^0.8, from cycle 6 to 12 about as updates^0.5. The policy-only arm's |ΔQ|
moves only through shared trunk weights, about 0.06 by cycle 8; F12's value moves
about 0.075 in the first cycle and stays near 0.11.

## What this does and does not bound

- It measures the base's own search at N sims against the same search at 64. It does
  not measure a trained net.
- A net that distilled the 64-sim search is then searched again at 64 sims. Its
  strength could exceed "base at N sims" (search on top of a better prior compounds)
  or fall short of it (distillation is lossy, the value target is the noisier part,
  and a sharper prior can make 64-sim search explore less). So the gap is a scale for
  sizing, not an upper or lower bound on a one-generation gain.
- Policy targets are 64-sim visit distributions, so a single generation distils
  roughly "64 sims of search". The N-vs-64 gaps are what further generations
  (iterated distillation) could in principle reach; the one-generation scale is
  closer to the 64-vs-1 gap (+849, banked) times the fraction of it the net absorbs.
- All on the EVAL openings with 64 sims as the reference; seat-balanced; same
  weights on both sides.

---

## RESULTS

(appended after the runs)
