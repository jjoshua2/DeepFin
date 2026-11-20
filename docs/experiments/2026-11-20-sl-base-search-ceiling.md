# 2026-11-20: SL base search ceiling — banked search gains and dose → drift (sizing context)

**Status:** context only, no new compute counted. It sizes the first loop experiment
([loop experiment 1](2026-11-20-loop-exp1-one-generation-from-sl-base.md)) from
measurements instead of guessed doses, and decides nothing about training.

## Search gain of the SL base (banked)

How much stronger the SL base (sha256 `a39f900c944477c4…`) plays with more search. A
net that perfectly distilled its own 64-sim search, searched again at 64 sims, would
play roughly like the base with more search. These rows are D1 of the
[2026-11-19 record](2026-11-19-rl-takeover-diagnostics-and-policy-only-ablation.md),
banked in `~/chess-artifacts/operations/rl-claude-20261119/search-gain/results.jsonl`:

| Candidate sims | Reference sims | Games | Score | Elo of candidate (95% CI) |
|---:|---:|---:|---:|---|
| 64 | 1 | 200 | 0.9925 | +849 (lower bound +691) |
| 256 | 64 | 400 | 0.8275 | +272.4 (+243.1 to +305.3) |
| 800 | 64 | 200 | 0.94 | +478.0 (+410.2 to +582.3) |

Settings: `scripts/arena_standard.py --mode matched_sims --search-shape training
--no-gumbel-noise --compile off`, tablebases on, seed 20261119, 128 games in flight,
the default 147k-line opening book (not the loop's EVAL split), git `df55413a6`.

**What the 1-sim side plays.** With one simulation and no root noise the candidate
width is 1: only the top-prior child is visited. The arena then samples its move at
temperature 0.1 from softmax(log prior + σ(completed Q)), where unvisited children
carry v_mix. That is essentially the raw policy prior sharpened by the T=0.1 power
and nudged by a single child value: neither a pure argmax of the policy nor a search.

**Seat handling.** Single orientation: the high-sims side was the candidate seat in
every game (colours are balanced by opening pairs). D4 round 2 found no
candidate-seat offset (+6.4, −4.2 to +17.1 Elo over 998 same-net pairs); an offset of
that size is negligible against effects of +272 to +849.

**A started rerun, stopped.** A seat-balanced 1,600-game rerun on the EVAL openings
was registered (commit `96ac13162` on this branch) and started, then stopped on the
owner's decision because the ceiling was already banked. One orientation of 128 vs
64 had logged 284 of 800 games (134 complete pairs) when it was interrupted. It is
banked as-is and marked incomplete
(`search-ceiling/INCOMPLETE.json`, `ceiling_s128_A.games.jsonl`); it wrote no result
row and is not a result. Its descriptive reading, +152 Elo (+122 to +184) on a
completion-ordered subset, is not used.

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
updates^0.9 (exponents 0.91 for P1, 0.86 for F12), from cycle 6 to 12 about as updates^0.5. The policy-only arm's |ΔQ|
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
- The banked rows use the default opening book, not the loop's EVAL split, and a
  single seat orientation; same weights on both sides.
