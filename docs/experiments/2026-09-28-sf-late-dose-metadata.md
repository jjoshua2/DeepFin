# 58M SF late-dose census: metadata receipt and bounded next step

**Status: source-only planning, no row-level census.** A bounded reader inspected
the frozen E manifest roster, 35 child-manifest headers, their referenced
derivation summaries, and three sample Zarr schemas. It read **4,713,721
metadata bytes** against an 8 MiB cap. It opened no corpus or teacher array
chunks, raw rows, model, GPU, trainer, or arena. The exact machine-readable
[receipt](evidence/sf-late-dose-metadata-20260928.json) has SHA-256
`30074c27cdc816cf4927bca13a8257275fc90b256246ca095a88eeb3f44c54da`.
The reader itself was sealed at SHA-256
`50f91a8cad15bc12b32c90b1f0428aff8d8f1fbc5c1f0fd321badb07fe7ca710`;
this record publishes its method and receipt, not an executable corpus scanner.

The reader verified the 10,797-byte E roster at SHA-256
`ca4e61bbae176ac0b6d1f32698aad9de299bc4f48de3035f2a430dac60ccf34c`.
It read **only the first 4 KiB of each large child manifest**, then checked
each referenced derivation-summary hash against that header. It did **not**
verify any complete child-manifest hash; the summary bindings are therefore
conditional on unauthenticated headers. The prior E receipt is explicitly
storage-only [qualification](2026-09-27-selected-e-sparse-source-audit.md),
not new source or teacher admission. The counts below reconcile roster entries
with those summary headers, but are not an independently authenticated
58M-row payload scan.

| SF source family | Corpus config SHA-256 | Cohorts | Retained rows | Derived shards | Saved SF policy observation |
| --- | --- | --- | ---: | ---: | --- |
| run03 original | `50fdfe58a5bc2275b4957b898811190fe91739d15dfa45ce2190749452c9bfc8` | 0 | 18,910,484 | 2,309 | latest-phase d9 |
| run06 G10 | `eda5eacfc84a5b9276d8822261a70eba67b4636a7886f49e0af67b72183a8187` | odd 1–33 | 25,827,015 | 3,160 | phase-zero d9 |
| run07 G10 | `79d4bb2e9f4302ee858e73cf2f4c4bdcce3ebb7c421dc2fbff310b194f87b29e` | even 2–34 | 13,353,189 | 1,639 | phase-zero d9 |
| **Total** | | **35** | **58,090,688** | **7,108** | source-dependent |

The sampled derived shards have `ply_index`, `game_id`, legal mask, policy,
WDL, and `x` float16 `[N,175,8,8]` chunked by 512 rows. They have no sampled
scalar `piece_count` column. Directly obtaining piece count from qualified
`x` requires the first 12 piece planes; because each compressed chunk spans
all 175 planes, this direct route would decode an estimated
**1,301,231,411,200 feature bytes (1.183 TiB)** over all 58,090,688 rows,
before other columns or raw scores. This is a decoded-volume estimate, not
measured disk I/O, elapsed time, or a lower bound for every possible census.
Raw schema-3 rows already carry `piece_count`. An authenticated raw-to-derived
join, checked against `x` on a capped sample, might support a cheaper exact
census; no such route has been qualified or costed here. G10 has per-row
`row_provenance.npz`; run03 lacks it and needs its original-order/no-result
streaming join checked separately. Join on physical source-qualified row
identity and cluster uncertainty by complete source-qualified game, not by
bare numeric game ID. The existing [Selected-E source audit](2026-09-27-selected-e-sparse-source-audit.md)
also shows that derived games may cross shard boundaries.

**Preregistered diagnostic, not a launch:** after full manifest, source,
sidecar, teacher, and join qualification, apply one fixed ordinary Tactical300
policy correction to D's mixed BT4/Ceres policy on the same retained rows.
Report source family × ply (`<40`, `40–79`, `≥80`) × piece-count (`7`,
`8–13`, `14–22`, `≥23`) denominators, ordinary/mate and legal-score
eligibility, corrected mass, and BT4/Ceres disagreement. Keep run03's
latest-phase d9 distinct from G10's phase-zero d9: the saved
[G10 constraint screen](2026-09-11-sf-negative-constraints-screen.md) is a
proxy and cannot supply a homogeneous 58M dose. Separately tabulate D–E WDL
target difference and Selected-E teacher route balance; neither is the D
policy correction. Every position was SF-generated, so a target change does
not test SF-free source generation or demonstrate playing strength.

Freeze control selection before reading dose: within each source family,
compare exactly-seven-piece rows with the same number sampled from late
`ply_index ≥80` and from all eligible rows, using a fixed hash of physical row
identity. Add a source-and-piece-bin-matched random control for the late
allocation from `ply_index <80` to test whether ply adds information beyond
material count.
The seed is SHA-256 of `58m-sf-late-dose-controls-v1`; refuse an allocation
if its eligible pool is too small. The primary diagnostic is the difference
in mean legal-masked policy L1 movement between late and source/piece-matched
random rows, with a 95% complete-game-cluster bootstrap interval. A lower
bound above zero would qualify only a separate measured-cost screen; it is
not a training trigger, and even a precise target-mass difference has no
known Elo value. Report each family separately and treat the other controls
as descriptive. Report natural late coverage separately. These are **equal-row-count**
allocations only: raw SF phase records bank nodes but no per-row search wall
time. A cost claim needs independently measured SF labeling wall, then a
new frozen equal-wall allocation. Target-mass movement remains a proxy;
only a prospective matched training and strict rule50 six-man arena can
support a strength claim.

The only proposed payload step is a separately reviewed **three-shard CPU
pilot**, one frozen first shard per source family, at most 24,576 retained
rows, 30 minutes inclusive wall, two numeric CPU threads, GPU hidden,
≤2 GiB peak RSS, ≤1 GiB source reads, and ≤64 MiB output. It must first
verify all relevant full child hashes and physical joins, then stop on any
identity, legal-support, piece-count, completeness, or resource-cap failure.
The pilot has **not** run; its purpose is to validate joins and measure cost
before deciding whether any full census or training comparison is feasible.

Source contracts: [A–E teacher/source audit](evidence/bootstrap-syzygy-audit-20260923/teacher-path-audit.md),
[Selected-E source audit](2026-09-27-selected-e-sparse-source-audit.md),
`scripts/corpus_row_provenance.py`, `scripts/sf_policy_rewrite.py`,
`scripts/derive_corpus_targets.py`, and `chess_anti_engine/model/transformer.py`.
