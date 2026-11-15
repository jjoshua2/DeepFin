# Retained-search WDL target census of the fixed G1 bank

Status at preparation: **prospective diagnostic; corpus execution UNRUN and UNADMITTED.** The standalone reader and 35 synthetic cases have source review, with local imports and tests still unrun. Final source/profile review, main publication and a fresh root CPU admission are required before corpus access. This record authorizes no training or checkpoint adoption.

The [512-update retained-search pilot](2026-11-15-retained-search-mainwdl-512-pilot.md) failed both registered playing gates: B scored .453 against matched A and .4555 against unchanged A39, each over 1,000 games / 500 pairs with an approximate paired 95% upper bound below .5. Those results motivate this diagnostic; they do not identify its answer. The bank has already been used for training and qualification. This plan fixes a new analysis before its corpus read; it does not make the bank unseen or independent.

## Question and complete population

Describe disagreement between retained-search WDL targets and genuine recorded outcomes, separating the W−L value axis from draw probability. Distinguish original F32 targets, physical F16 storage and the subsequent loss normalization. The deciding gate is technical integrity and complete coverage, not a favorable metric value. A qualified run reports whatever values occur.

Use all **256 original G1 physical TRAIN games / 24,827 rows**, generation manifest `35279e2ef19cdf808e4b4e25bb50fbb524750693ee035c1d44080d87a3db7878`. The prior producer records 114 natural and 142 strict-TB-adjudicated games. Prior full-bank qualification reports 24,827 available, nonidentical search targets. These are pre-existing receipt counts, not this census's measurements.

Authenticate the existing fixed-bank authority and original postclosure promotion, then its original audit/producer protocols, closed owners, inventories, generation, per-game wrappers and evidence. Join physical game and row IDs, original ply, player-to-move perspective, outcomes and masks. Check exact archived F32-to-stored-F16 quantization. Numerical game IDs or retained-row adjacency cannot substitute for those joins. The census inherits authenticated full-history, legality, encoding and resolved-outcome proofs; it does not rerun engines or tablebases or issue a new data acceptance.

Retain every game and row. Missing masks, invalid values, inconsistent identities or incomplete required coverage fail the technical gate rather than silently excluding observations. Original validity rules remain unchanged, including the F32 simplex check. Do not normalize malformed input into apparent validity. No relabeling, filtering, resampling, new generation, actor inference, book/pool access or role allocation is permitted.

## Representations and fixed descriptive functions

Report separately:

1. Original archived F32 search-WDL triples.
2. Physical stored F16 triples widened to F32, without normalization.
3. The existing source-formula nonnegative clamp and row-mass normalization of widened F16, computed in F32.

Then cast each representation to F64 for descriptive functions, sums and quantiles. The third representation is a source-formula reconstruction, not a new Torch/GPU replay or a bit-identity claim. Physical F16 vectors can have rounded row mass; their vector errors and negative sum of p log p are algebraic descriptions, not calibrated probabilities or normalized Shannon entropy.

With Q = W−L, recorded outcome z = +1/0/−1 and draw indicator d, report Q−z and D−d, their absolute values and squares, and the sum of squared component residuals against the genuine one-hot WDL outcome. Also report Q, D, row mass, finite/nonnegative/missing/invalid denominators, mass minima/maxima and negative sum of p log p with 0 log 0 = 0.

Boundary descriptions include exact zero W, zero L and their union, plus numerical boundary mass where abs(abs(Q)−(1−D)) ≤ 1e−6. These are **not clipping frequencies**. The archive does not retain unclipped bestQ, full original raw root WDL or search visits, so it cannot recover search improvement over the raw network or discarded uncertainty.

Use fixed unweighted finite-row quantiles at 0, .01, .05, .25, .5, .75, .95, .99 and 1 with NumPy's linear method. Report scalar functions/rates explicitly, without choosing quantiles after seeing their values. Equal-game weighting applies to means: average per-physical-game row means. Within a stratum, include games with at least one row there, reporting that contributing game count and the row denominator. Empty cells have null metrics and zero counts, not imputed zero outcomes.

## Strata and interpretation

Report overall, recorded natural versus strict-TB-adjudicated endings, genuine outcome class, exact plies-to-recorded-stop bins 0–4, 5–16, 17–64, 65+, and UNKNOWN, and the ending-class-by-distance cross.

Where metadata is available, require total_plies minus actual retained ply to equal the number of played moves minus move_offset. Inconsistent identities fail; unavailable distance metadata remains UNKNOWN. Never reconstruct distance from rounded moves_left or retained-row order. For strict TB games, the stop is adjudication. Eventual mate/end distance beyond that stop is unknown.

This is a census of one finite actor/path-conditioned training bank. Report row-weighted sums/counts and equal-game means, retaining per-game sums/counts. **No population confidence interval, independent-row inference or playing-strength selector is defined.** One resolved outcome is not the optimal value of a position. A target's closer agreement with that outcome does not establish that training on it improves search or timed play.

## Source, controls, budget and recovery

The standalone source and tests were first reviewed at `105d6eba50c4f3b032a31e0020d02e76b75e65c1`; final source/profile identities and the main publication proof are bound separately before execution. The reader uses narrow NumPy/Zarr/codec processing with bounded NPZ headers and authenticated inventories. It imports no Torch or project/native chess runtime and decodes only necessary target/identity columns. Only bounded scalar metric vectors survive each game for exact quantiles; no per-row payload is published.

Meaningful synthetic controls precede real corpus access in the same owner. They cover outcome/perspective signs, representation and quantization, masks, malformed or unsafe inputs, authority/source/row joins, sparse actual ply distances, and row versus equal-game denominators. Authored cases are not passed-test credit until executed. Final controls-to-census wrapper review remains required.

One combined root-owned CPU phase is capped at **600 outer / 540 inner seconds**, including controls, source/identity checks and the complete census. GPU budget is zero. Affinity is CPUs 20 and 22, with OMP/MKL/OpenBLAS at two threads. Count all actual failed wall time. Preserve the global week-one deadline, control markers, leases, neighboring jobs and the unchanged donor. No automatic retry, cap extension, fallback data or second numerical owner is authorized.

All input data remain read-only outside code checkouts. Bank compact aggregate and per-game sum/count reports with source/settings/input identities; keep raw arrays and bulk data outside Git. Technical failure emits UNQUALIFIED diagnostics and preserves the failed owner, with no clean partial statistics. A source correction or future admission needs its own review and remaining budget.

After a qualified readout, select any learning intervention separately and test it with fresh controlled playing evidence. This census does not select a dose, optimizer, teacher, target transform, checkpoint or an optimum RL formula. A39 remains retained and the broader strength objective remains unachieved.

External records use `~/chess-artifacts/operations/rl-week-one-20261112-v1/`:

| Record | SHA-256 |
| --- | --- |
| `core-retained-search-WDL-bank-decomposition-selection-v1/decision.json` | `833b057f35ec24ddf8feed5154daca32c4668338a6be3f5545ba1fec76fa82f7` |
| `core-retained-search-WDL-bank-decomposition-statistic-clarification-v1/decision.json` | `6fb0dd253f0239cffdbe900d33f9370c49d6f0c188383649a9882ff9affd4053` |
| `core-retained-search-mainWDL-512-pilot-bindings-v1/data-bank-authority.json` | `2d8570409214a331323d5c1c9b68e46f1308257608ee9a9b39244f24de8513e5` |
| `core-retained-search-mainWDL-target-decomposition-feasibility-v1/manifest.json` | `c675ee0a71d904b9ef8c93224c34a01aa00606a8739ec32476094e31235e5d95` |
| `core-retained-search-target-census-code-independent-source-review-v1/review.json` | `5b515645a4d8d21ad97c246edccb75142a985fa4798f1c75c33bfe3a766d4034` |
| `core-retained-search-target-census-source-preparation-v1/manifest.json` | `dbc25acf6428f3ca8349535c71f228617d7a2c0dbd6dfd8bc5df77c79f0550e5` |

The initial private profile names the reviewed code-only commit above. Any subsequent docs/main integration requires an explicit, reviewed source-head/path metadata transfer with those same reader/test bytes before final execution bindings. It does not change the selected population, functions, masks, strata or CPU ceiling. The separate publication record binds the final protocol, source and main proof; no publication digest is placed inside that protocol.
