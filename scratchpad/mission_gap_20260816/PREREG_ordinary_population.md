# PREREG — SF instability on the ORDINARY population (the base-rate control)

Committed BEFORE any ordinary-population position is scored. Companion to the
2026-08-16 ledger entry "SF verdict instability, 75k -> 4M nodes", whose own
"Next" line pre-committed exactly this run.

## The question this answers, and the one it does NOT

The 123 banked rows are harvest **CANDIDATES** — positions already selected for
(net was wrong) AND (SF adjudicated them lost). Their 4.88% continuous
instability rate is therefore an **enriched** rate and cannot be used as a corpus
yield estimate. The decision the mission actually needs is:

> If we screened ORDINARY selfplay positions to build the Tier-A corpus, what
> fraction would clear the instability filter?

That is a **base rate**, not a contrast. It is measured directly here.

**NOT answered:** whether the harvest procedure enriches. That contrast is
reported as secondary and is UNDERPOWERED by construction — computed in advance,
Fisher one-sided, harvest 5/123 vs ordinary at n=1200: an ordinary rate of 2%
gives p = 0.168 and 3% gives p = 0.369. **⚑ No enrichment conclusion may be drawn
from this run unless the ordinary rate lands at or below 1%.** Stated now so it
cannot be read off post hoc.

## Instrument

`scripts/blindspot_deepsf_scaling.py`, unmodified. Production Syzygy pair. Cold
TT per budget (`ucinewgame`), i.e. independent convergence checks, not
incremental deepening.

**Ladder: `--nodes-list 75000,150000,500000`.** The 1M and 4M rungs are dropped:
the banked run established 500k..4M is stable on 122/123, so they cost 7.9x the
nodes for ~one extra event. This makes the ladder ~8x cheaper and buys ~10x the
sample, which is what the base-rate question needs.

**⚑ Dropping the rungs CHANGES THE RULER, so the comparator is re-derived on the
new ruler from the SAME banked rows — not carried over.** Re-derived from
`sf_scaling_75k_to_4M.jsonl` (all 5 budgets are banked, so this is free and
requires no re-running):

| ruler | harvest k/123 | rate | 95% CI |
|---|---|---|---|
| 75k vs 4M (the banked headline) | 6 | 4.88% | [1.81%, 10.32%] |
| **75k vs 500k (THIS run's ruler)** | **5** | **4.07%** | **[1.33%, 9.23%]** |

The two rulers are NOT nested: 4 positions move on both, 2 only at 4M, 1 only at
500k. So this is a different measurement that happens to be similar, and
**4.07% — never 4.88% — is the number this run compares against.**

## Population

**n = 1200 rows drawn from `data/audit_set_v1.jsonl` (4000 rows), uniformly
without replacement, `numpy.random.default_rng(20260816)`, indices banked.**

Why this is the right ordinary population: `scripts/build_audit_set.py` sources
positions from **real replay rows** and stores `board.fen()`. They are ordinary
selfplay positions, sampled with **no conditioning on SF behaviour** — which is
exactly the property the harvest set lacks.

**Two instrument facts verified before committing to this source:**

1. **History loss is INERT here.** `StockfishUCI` sends `position fen {fen}` and
   never a move stack (`chess_anti_engine/stockfish/uci.py:583`). The harvest
   lines' ~8 plies are used only to derive the terminal board via
   `seed_board_from_line`. **SF is shown the same kind of object in both arms**,
   so the audit set's known "0/4000 history" defect cannot bias this comparison.
   (It would badly bias a NET-side measurement. This is not one.)
2. **Castling is genuine, not stripped.** 556/4000 = 13.9% of audit rows carry
   castling rights; the "no castling" claim came from a 600-row subset draw.
   Positions without rights are late-game positions that really have none.

**CONFOUND, stated in advance with its direction.** The audit set is deliberately
phase-balanced (1332/1335/1333). Natural selfplay is not. If instability
concentrates in endgames — and the one morphology we have is a locked pawn
endgame — this **over-represents** the unstable phase and biases the measured
ordinary rate **UPWARD**. That is conservative for ABANDON and anti-conservative
for PROCEED: a PROCEED verdict here must be re-checked on a phase-matched draw
before any corpus is built, an ABANDON verdict needs no such re-check.

## Statistic

Per position: `|sq(75_000) - sq(500_000)| >= 0.20`, the ENDPOINT comparison
(not "any adjacent-rung flip" — the banked entry showed those differ, 2.44% vs
0.81%, because of non-monotone wobble that returns).

`k` = count of qualifying positions out of 1200. CI = Clopper-Pearson 95%.

## Pre-committed decision rule (bar = the Tier-A yield bar of 1%)

Computed before the run; the readout is a pure lookup, no judgement:

| observed k | rate | verdict |
|---|---|---|
| **k <= 5** | <= 0.417% | **ABANDON** — 95% upper bound < 1%, ordinary screening cannot feed a corpus |
| **6 <= k <= 19** | 0.5%–1.58% | **INCONCLUSIVE** — CI straddles the bar |
| **k >= 20** | >= 1.667% | **PROCEED** — 95% lower bound > 1% (subject to the phase re-check above) |

## ⚑ PREDICTED COUNT, recorded before running

**I predict k in the range 6–19, i.e. INCONCLUSIVE**, on the reasoning that the
harvest selects on net-error AND SF-lost adjudication, which should enrich by
roughly 3–8x over base, putting ordinary at ~0.5–1.4% — squarely in the band.

Recording this because a design whose own author expects it to return
INCONCLUSIVE is a design worth questioning BEFORE spending the compute. It is run
anyway, deliberately: **the CI is the deliverable, not the trichotomy.** Even an
inconclusive verdict replaces "unknown base rate" with a bounded one, which is
what the Tier-A prereg needs in order to size its screening budget. If the
outcome lands outside 6–19 that is itself informative about the harvest
mechanism, and the miss will be recorded rather than quietly re-narrated.

## Banking

Raw rows -> `scratchpad/mission_gap_20260816/sf_scaling_ordinary_n1200.jsonl`
(NEW path; `sf_scaling_75k_to_4M.jsonl` and the 2026-07-08
`scratchpad/harvest_fp/deepsf_scaling.jsonl` are READ-ONLY inputs and must not be
touched). Sampled indices -> `ordinary_sample_indices.json`.

Verdict recorded in `docs/experiment_ledger.md` the same session, judged by the
table above and not by post-hoc reading.
