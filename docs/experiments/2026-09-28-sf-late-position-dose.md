# Saved Stockfish policy dose by game ply

Status: **source-only readout; no training authorization**. This record reuses the
published G10 frozen-bank Tactical300 preview. It opens no corpus payload and
does not run a teacher, model, arena, or trainer. The question is whether that
fixed policy correction's *saved target change* is concentrated at late game
plies. It is not a test of playing strength or of a late-only training recipe.

The [September 11 constraints readout](2026-09-11-sf-negative-constraints-screen.md)
banked 262,079 unique rows from 32 shards and retained 1,331 source-qualified
game clusters. Of these rows, 187,628 have ordinary d9 scores; 74,451 are
mate-domain exclusions. The exact Tactical300 preview triggers on 13,495
ordinary rows. The table partitions the saved observations by ply; moved mass
is the sum of probability moved by the preview, not a count of positions or
Stockfish searches.

| Ply | All rows | Ordinary rows | Scientific triggers | Mean moved mass per ordinary row | Share of total moved mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| <40 | 53,024 | 48,490 | 2,583 | 0.043548% | 1.146% |
| 40–79 | 51,524 | 38,499 | 1,556 | 0.304511% | 6.364% |
| ≥80 | 157,531 | 100,639 | 9,356 | 1.692828% | **92.489%** |

The three moved-mass sums are 21.116653, 117.233513, and 1,703.644674;
the total is 1,841.994840 probability units. Applying *exactly this fixed
operator* only at ply ≥80 would retain 92.489% of its saved G10 target-change
mass while touching 53.638% of ordinary rows. Averaged across **all** ordinary
rows, that hypothetical late-only dose is 0.907991%. These are same-bank
arithmetic, not a forecast for the 58,090,688-row factorial corpus. Another
31,669 bit-level-only normalization differences are outside the 13,495
scientific triggers and do not represent additional interventions.

The constraints bank limits what this concentration means. At ply ≥80, the
ordinary d9 preview excludes 56,892 mate-domain rows, or 36.115% of all rows
in that slice. The narrowed deeper rosters leave **57.534%** of the late
>300-cp inferior BT4 policy mass unscored. Of the late inferior mass with an
adjudicable deeper comparison, **12.230%** is contradicted by the conservative
all-winner test. Conditional regret normalizes on the scored roster and cannot
resolve the missing-move bias. Ply measures game age, not remaining material;
this bank has no published ply-by-piece-count table. Accepted 58M source
summaries declare a minimum of seven pieces, but no decoded-row census proves
how many late positions have exactly seven pieces.

Strength evidence also remains separate. Tactical100 versus B100 was −13.6 Elo
[-52.4, +24.9], and Downside300 versus B100 was +17.66
[-17.97, +53.66]. Both 256-game, 400-simulation comparisons were unresolved,
used different policy operators, and had no tablebases. On the 58M factorial
corpus, the two policy contrasts were B–A +6.79 [-24.95, +38.63] and D–C
−16.30 [-46.88, +14.04] Elo. The two value contrasts were C–A +21.74
[-12.49, +56.40] and D–B +17.66 [-13.93, +49.54] Elo; adding Ceres value
also reduced both SF and BT4 shares, so it did not isolate SF. A later strict
rule50-aware six-man Selected-E versus D comparison gave −19.02
[-33.43, −4.68] Elo over 576 opening pairs, **inconclusive** under its
precommitted −30-Elo noninferiority gate. Its changed source, teacher routing,
and schedule prevent attribution to SF removal alone. None of these matches
tests a late-only Stockfish selector.

The next decision is a **bounded read-only cross-tab screen**, not a training
launch. Before opening registered 58M payload, freeze and authenticate the 35
cohorts, 7,108-shard base/overlay roster, 58,090,688-row count, raw SF and
native-teacher lineage, exact joins, and full-game identities. A reviewed
first-cohort pilot should set the whole-census limits and forecast. The proposed
hard ceilings are 24 hours, 1 TiB source reads, 16 GiB RSS, 8 GiB compact
output, two CPU threads, GPU hidden, and an explicit stop. If a pilot projects
an overrun, publish a weighted sample instead of claiming a complete census.

The fixed screen should cross source family and ply `<40`, `40–79`, `≥80` with
piece bins `7`, `8–13`, `14–22`, `≥23`, ordinary/mate score domain, check and
legal-count bands, and teacher availability. Keep the existing Tactical300
policy operator fixed; a value claim needs its own separately pinned operator.
Report legal-masked moved mass, scientific triggers, scored/unknown/contradicted
mass, and BT4/Ceres disagreement with complete source-qualified game clusters.
Compare a late selector against a source-matched game-cluster random selector
at equal **measured SF label-search wall** and label count. If per-row cost is
unavailable, an equal-count comparison is only an equal-count result. No
threshold or correction-weight sweep is authorized by this readout. A later
playing claim needs a separately registered matched training comparison and
strict rule50-aware six-man arena.

[Compact arithmetic and source hashes](evidence/sf-late-position-dose-20260928.json)
pin the exact saved slices. Historical training and evaluation constraints are
in the [Tactical100](2026-09-10-bt4-sf-tactical-training.md),
[Downside300](2026-09-12-sf-allmove-downside.md),
[factorial](2026-09-22-factorial-readout-next24h.md), and
[Syzygy audit](2026-09-23-bootstrap-syzygy-correctness-audit.md) records.
