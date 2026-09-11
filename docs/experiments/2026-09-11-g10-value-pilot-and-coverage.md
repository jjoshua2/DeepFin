# Saved G10 values and neural-teacher coverage

Status: two matched batches totaling 526,376 rows passed. The larger derivation stopped on a confirmed malformed saved SF roster. Historical native-BT4 value reuse is implemented; no new training or strength result.

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

The complete 1,574,952-row native-WDL cohort spans two increments plus the first 128 derived shards of the original run06 large batch. The large-stage plan conservatively used the full 2,025,055-physical-row selector to preserve that layout. A smaller selector or recovered output prefix requires an explicit row/provenance equivalence proof; approximate raw-row truncation is insufficient.

Independent completed-result review passed: receipt hashes, all 32 shard records, provenance, omissions, settings, value identity and timing agreed with the preregistration. The review reused the completed array-comparison evidence rather than repeating the payload scan.

## Second matched batch completed

Run07 passed with 264,297 retained rows in 33 shards. All 16 nonvalue arrays, row order and provenance match the original; exclusions remain 1,600 missing-result rows and one policy-support miss. Search WDL changed on 179,992 rows. Selections were 188,244 d10, 74,334 d12, 1,716 single-move d10, and three malformed-later-roster fallbacks to a valid original baseline. Derivation took 488.29 seconds; the whole comparison took 581.66 seconds. Independent completed-result review passed. Together the two increments provide **526,376 matched rows**, still requiring value-recipe and training admission.

A posthoc descriptive check of run06's stored WDL arrays found mean total variation 0.005240 and 6,335/262,079 rows (2.417%) with absolute change in Q = W − L of at least 0.05. This describes target changes, not improved accuracy or strength.

## Large batch stopped on a real source-label ambiguity

The large derivation exited 1 after 3,136.43 seconds, before its timeout. It encountered original derived shard 197, row 5,084: worker 2, game 99,518, ply 318, from raw `w02-00196.jsonl.zst` row 3,541. This is **outside** the intended native-WDL prefix of derived shards 0–127.

The frozen raw hash, original provenance and full-history input key match. Phase-zero d9 has all 14 unique legal moves and a best score of −625 cp. Later consumed d9 records repeat moves: phase two contains `d2d3` at −753, 0 and 0 cp. The historical dictionary construction keeps the final duplicate, turning the composite best score into 0 cp and producing a near-balanced WDL label. Later deeper rosters are malformed too. This is a real ambiguity in saved labels, not a parser mismatch, policy-support exclusion or evidence against adaptive SF values.

The new validator correctly rejected that ambiguous baseline. The failed output and exact source record remain preserved; no fallback rule was loosened and no new search was run. The next bounded investigation checks whether already-written prefix shards can be independently qualified, or whether an exact frozen source prefix can reproduce them. The failed whole stage remains failed even if a separately qualified subset is recovered. There is not yet a measured prevalence estimate for this defect.

[Compact evidence](evidence/g10-value-readiness-20260911/run07-and-large-integrity.json) pins the completed run07 receipt, failure and exact-row diagnosis. Do not count the intended 1.575M cohort as successfully prepared.

## Reusing native BT4 values without new inference

[PR #637](https://github.com/jjoshua2/DeepFin/pull/637) adds explicit historical native-WDL admission to the existing value writer. It preserves original G10 source bindings, producer identities and actual content/feed checks. Forty-nine focused tests, whole-repository lint and independent review passed.

The actual run06 manifest covers 262,079 rows, 32 shards and two historical collection invocations. Metadata admission and independent review passed; this does not replace payload verification during materialization. No new value corpus or training run has been produced. A separately qualified adaptive-SF value input and honest selection handling remain necessary for the matched SF-anchor comparison.

Ceres snapshot 13 independently qualified 11,537,684/18,910,484 original20M positions (61.01%) at the latest completed audit. This is separate from G10. The registered Ceres policy/value comparisons remain the next GPU experiments after collection and corpus qualification; no Ceres strength result is claimed.
