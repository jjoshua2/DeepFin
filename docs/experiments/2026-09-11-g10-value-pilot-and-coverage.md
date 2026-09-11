# Saved G10 values and neural-teacher coverage

Status: 128-row integration pilot and first 262,079-row matched derivation passed; metadata coverage refreshed. No new training or strength result.

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

## First matched G10 batch completed

The original run06 increment batch now has a matching adaptive-SF derivation: all 262,079 retained rows, 32 shards, identical source-qualified row order/provenance and all 16 nonvalue arrays equal in decoded bytes. The 4,400 missing-result and 12 policy-support exclusions match the original derivation exactly. Stored search-WDL values changed on 180,584 rows (68.90%). Selection counts were 186,261 ordinary d10, 74,187 d12 and 1,631 complete single-move d10; no fallback occurred in this retained batch.

Derivation took 474.91 seconds versus the historical baseline's 484.17 seconds; peak reported RSS was 874,160 KiB. The full derivation and array/provenance comparison took 564.34 seconds. This single timing shows the candidate is affordable on this batch, not a throughput improvement claim. The stage used two CPU cores, no GPU or teacher inference, and terminated successfully within its 40-minute limit. [Compact completed evidence](evidence/g10-value-readiness-20260911/matched-run06.json) pins the full local receipt.

Matching rows does not automatically authorize old teacher sidecars against a new source summary. The existing BT4 policy/WDL artifacts remain untouched and source-bound; an explicit identity bridge, target recipe and training schedule are still required. This is a data-preparation result, not a model-strength result.

Next, expand to the original run07 increment batch with the same checks. The complete 1,574,952-row native-WDL cohort spans those two increments plus the first 128 derived shards of the original run06 large batch. Reproducing the original large-batch layout requires its full 2,025,055-physical-row selector before taking that derived prefix. Do not substitute an approximate raw-row truncation.

Independent completed-result review passed: receipt hashes, all 32 shard records, provenance, omissions, settings, value identity and timing agreed with the preregistration. The review reused the completed array-comparison evidence rather than repeating the payload scan.
