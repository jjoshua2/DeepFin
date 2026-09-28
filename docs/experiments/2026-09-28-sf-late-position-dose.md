# Saved Stockfish policy dose by game ply

Status: **read-only saved-target readout; no training authorization**. The
initial source-only slice reused the published G10 frozen-bank Tactical300
preview; a later registered CPU cross-tab reopened only its pinned saved row
banks and raw source shards. Neither stage ran a teacher, model, arena, or
trainer. The question is where this fixed policy correction's *saved target
change* is concentrated, not whether it improves playing strength.

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
resolve the missing-move bias. Ply measures game age, not remaining material. The registered G10 follow-up
below cross-tabulates saved dose by raw-FEN piece count. Accepted 58M source
summaries declare a minimum of seven pieces, but no decoded-row census proves
how many late positions have exactly seven pieces in that separate corpus.

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

## Registered G10 ply-by-piece saved-target cross-tab, 2026-09-28

A separately reviewed, one-core, nice-19 CPU scan joined both exact saved G10 row banks to the 32 raw compressed shards, rechecked FEN-derived piece counts and source-qualified game identities, and read **1,270,054,769 compressed bytes**. It completed in **46.548 seconds** with 186,818,560 bytes peak observed child RSS. The registered terminal passed with zero training credit; independent readback rehashed the result and re-aggregated all nine cells from 1,331 source-qualified game clusters. The same 262,079 rows, 187,628 ordinary scores, 74,451 mate exclusions, 13,495 scientific triggers, and 1,841.994840 total moved-mass units reconcile with the earlier published ply-only readout.

| Ply and raw-FEN piece count | Rows | Ordinary d9 rows | Scientific triggers | Moved mass | Share of all moved mass |
| --- | ---: | ---: | ---: | ---: | ---: |
| <80, all material | 104,548 | 86,989 | 4,139 | 138.350166 | 7.511% |
| ≥80, exactly 7 | 25,816 | 17,708 | 2,471 | 523.451444 | **28.418%** |
| ≥80, 8–13 | 88,685 | 56,487 | 5,508 | 1,021.599856 | **55.462%** |
| ≥80, 14–22 | 42,411 | 25,951 | 1,364 | 158.239168 | 8.591% |
| ≥80, ≥23 | 619 | 493 | 13 | 0.354206 | 0.019% |

Thus late positions with **7–13 pieces account for 83.879%** of this fixed G10 Tactical300 moved mass. Exactly-seven-piece positions alone account for 28.418%; a seven-piece-only selector would omit the larger 8–13-piece share. A 2,000-resample bootstrap over the 1,331 source-qualified games gives a descriptive 95% interval of **82.278–85.311%** for the combined late 7–13-piece mass share. The intervals measure variation among these saved source games, not uncertainty about Elo or a different corpus.

This identifies where an already-defined SF policy correction changes *saved targets*. It does not show that shallow SF reaches Syzygy from 8–13 pieces, that deep SF labels help there, that the excluded mate domain is unimportant, or that the 58M/500M source mix has these piece frequencies. The preregistered 58M read-only pilot and any matched training/strict rule50 six-man arena remain separate decisions. [Compact independently re-aggregated evidence](evidence/g10-sf-ply-piece-cross-tab-20260928.json) has SHA-256 `4eb0232a9a05ee260290dd358deea0b2cef9e21e4346123c6335401103cb50a8`; its terminal/result and source packet SHA-256 values are `62e508d0899275615da26fda145067843a007617a1a245a6579a1ea3249702e8`, `4cf8b044eb4456b4a87defd8171ce7e8e3405d6a8dbe37662479ac8348d71dca`, and `a3c412b9c21c60dba447655e0880545b2a0700cb9d2d9fb2a0dece41b8b7bbba`.
