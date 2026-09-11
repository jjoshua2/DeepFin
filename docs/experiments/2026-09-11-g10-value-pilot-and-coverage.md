# Saved G10 values and neural-teacher coverage

Status: 128-row saved-data integration pilot passed; metadata coverage refreshed. No new training or strength result.

The next SF value comparison reuses the recorded adaptive d10/d12 observations. It does not increase search depth or launch additional Stockfish inference over the future 100M-position corpus. The opt-in selector is implemented in [PR #633](https://github.com/jjoshua2/DeepFin/pull/633). The existing [census](2026-09-11-deeper-sf-value-census.md) establishes target differences, not improved accuracy.

## Native BT4 value coverage

The accepted common G10 cohort contains 16,404,093 rows. Existing completed direct-derived BT4 WDL banks cover 1,574,952 rows. A metadata join identifies another 889,544 accepted survivors across 108 raw shards whose native WDL is already banked. They need the existing typed adapter to put those values into derived-row order. The prospective disjoint union is 2,464,496 rows; that total is not yet qualified training data.

The raw banks contain 15,740,010 native-WDL physical rows at this snapshot. Most lie outside the currently accepted cohort, so the raw total must not be reported as available matched training coverage. Counts and source-receipt identity are retained in [compact evidence](evidence/g10-value-readiness-20260911/coverage.json).

The existing adapter already supports optional native WDL export and typed lineage. Reuse it; older recommendations to build a new adapter are stale. Partially covered batches need an explicit source-qualified subset. Do not silently drop uncovered rows from one experimental arm.

## Next decisions

The real-row integration pilot below passed. Next qualify a matched G10 cohort for baseline versus adaptive SF values, holding policy targets fixed. When testing neural value mixtures, keep teacher coverage and weights fixed across the SF-anchor comparison. The original20M Ceres bank is a separate corpus and cannot be treated as G10 coverage.

Ceres labeling and its registered policy/value experiments remain separate ongoing work. Any further deep SF diagnostic should use a bounded sample; bulk labeling cost remains a constraint at 100M–1B positions.

## Completed pilot

The frozen 128-row sample passed actual derivation, value writing and replay loading in 3.27 seconds. It exercised 102 d10 and 26 d12 selections. Row provenance and order matched between arms, and all 16 nonvalue arrays were byte-identical, including their compressed representation. The search-WDL target changed on 110 rows. See the [completed receipt](evidence/g10-value-readiness-20260911/pilot.json).

Two failed harness attempts remain preserved locally: v1 incorrectly passed the loader's `(arrays, metadata)` pair into the sample constructor; v2 reached the equality checks but parallel Blosc compression reordered identical logical blocks. Saved-output diagnosis confirmed identical decoded bytes for all nonvalue arrays. V3 corrected tuple unpacking and pinned Blosc to one thread; no scientific check was weakened. Parent-observed session 43345 terminated with exit code zero.

This purposive sample covers only three source-qualified games. It establishes integration behavior, not whole-corpus eligibility, improved value accuracy or playing strength. Absent fallback/single-move strata remain covered by the implementation tests rather than being invented in this real-row pilot.
