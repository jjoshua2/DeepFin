# Saved G10 values and neural-teacher coverage

Status: original B100 T0.5 policy and fixed 50% original-SF / 50% native-BT4 WDL value products now cover all 20 accepted common G10 cohorts: 16,404,093 rows. The remaining 19 value writers completed, including the four-root run06-large mapping. Matched saved-SF derivations remain a separate 1,574,952-row subset. Training admission and playing strength remain unresolved.

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

The new validator correctly rejected that ambiguous baseline. The failed output and exact source record remain preserved; no fallback rule was loosened and no new search was run. The follow-up below established an exact frozen source prefix and used a fresh derivation; the failed output contained only worker spills, with no completed final shards. The failed whole stage remains failed even if a separately qualified subset is recovered. There is not yet a measured prevalence estimate for this defect.

[Compact evidence](evidence/g10-value-readiness-20260911/run07-and-large-integrity.json) pins the completed run07 receipt, failure and exact-row diagnosis. At this failed-stage readout, only 526,376 matched rows had passed. The later exact-prefix result below supersedes that coverage count.

## Reusing native BT4 values without new inference

[PR #637](https://github.com/jjoshua2/DeepFin/pull/637) adds explicit historical native-WDL admission to the existing value writer. It preserves original G10 source bindings, producer identities and actual content/feed checks. Forty-nine focused tests, whole-repository lint and independent review passed.

The actual run06 manifest covers 262,079 rows, 32 shards and two historical collection invocations. Metadata admission and independent review passed; this does not replace payload verification during materialization. No new value corpus or training run has been produced. A separately qualified adaptive-SF value input and honest selection handling remain necessary for the matched SF-anchor comparison.

Ceres snapshot 13 independently qualified 11,537,684/18,910,484 original20M positions (61.01%) at the latest completed audit. This is separate from G10. The registered Ceres policy/value comparisons remain the next GPU experiments after collection and corpus qualification; no Ceres strength result is claimed.

## Matched SF value consumption and training readiness

The value writer now accepts an optional manifest for a complete matched adaptive-SF cohort. It reads the candidate only for the SF term, retaining the original B100 policy product and native BT4 source bindings. It checks actual provenance, all 16 nonvalue arrays, separately witnessed WDL bytes and input stability. The default path remains unchanged. The shared value identity names the SF selector, while cohort-specific proof hashes remain in provenance; baseline and adaptive recipes cannot silently join as one target.

Fifty-nine focused tests, whole-repository Ruff/basedpyright/Vulture and independent code review passed. The two real increment manifests also passed metadata admission and independent review. A 2.68-second WDL-only read supplied the missing content witnesses for their 526,376 rows; the older completion receipts did not record WDL hashes, so these are explicitly current witnesses. This is not a full writer run or new training corpus. The partial large-batch selection remains a separate consumer boundary.

A read-only training audit found that `lc0_control_train.py` currently starts a fresh model and has no checkpoint-donor CLI. Its multi-epoch option does not resume B100. Trainer checkpoints contain optimizer/scheduler state, but the existing tolerant restore behavior and unsaved RNG state require an explicit continuation contract. A deliberately matched optimizer reset could also be valid if specified for both arms. Neither path is qualified for this G10 comparison yet. The targeted inventory found no materialized B100 policy product for these complete G10 increments.

Finish data/consumer readiness and reassess expected value before adding a donor mode or allocating GPU time to a small G10 comparison. The registered Ceres comparisons retain priority. Relevant local evidence is pinned in [consumer readiness](evidence/g10-value-readiness-20260911/matched-consumer.json).

## Exact large-batch prefix completed

The fresh exact-prefix derivation passed for **1,048,576 retained rows in 128 shards**. Its physical limit of 1,056,591 ends at original raw `w02-00127.jsonl.zst` row 2,278; the original frozen 244-shard selector remains recorded. It has exactly 7,999 missing-result and 16 policy-support exclusions, with no envelope omissions. Original prefix provenance/order and all 16 nonvalue arrays match in decoded bytes. The deriver uses the same per-output-shard shuffle sequence; row-local search values do not require the rest of the game after the cutoff.

Search WDL changed on 718,464 rows. The selected values comprise 748,128 ordinary d10, 293,877 d12, 6,564 single-move d10 and seven malformed-later-roster fallbacks to valid original d9 baselines. Those fallbacks are retained rows, not exclusions or successful deeper selections.

Derivation took 1,934.20 seconds (32.24 minutes), with peak reported RSS 868,760 KiB. Including complete prefix matching, the stage took 2,287.40 seconds (38.12 minutes), within its 45-minute child and 60-minute total limits. Parent-observed session 86138 exited zero; independent completed-result review passed without repeating the payload scan. [Compact prefix result](evidence/g10-value-readiness-20260911/matched-prefix128.json) pins the receipts and source reference.

Together with the two increments, this completes matched derivation for **1,574,952 native-BT4-WDL-covered rows**. It does not make the whole original large batch valid: the previous full-stage failure and its malformed row remain preserved outside this prefix. No failed worker spills were reused, and no teacher inference was added.

The value writer currently supports the two complete increments; consuming this prefix still needs honest selected-source handling for its original full-source teacher bindings. The next completed stage below exercises the writer on one complete increment. No G10 value comparison has been trained. Keep the registered Ceres comparisons ahead of an unqualified small-corpus training comparison.

## First real matched SF–BT4 value products completed

The complete run06 increment now has an actual B100 policy product and two value products on the same **262,079 rows in 32 shards**. Policy uses normalized global BT4 at teacher temperature 0.5. Both value products use 50% normalized SF WDL and 50% of the same native BT4 WDL; only the SF source changes from the original labels to the matched adaptive labels.

All three existing producers exited zero. B100 policy materialization took 104.01 seconds, the original-SF value control 65.41 seconds, and the adaptive-SF value product 94.08 seconds. The full stage took 272.11 seconds (4 minutes 32 seconds), within the 20-minute cap. Peak reported stage RSS was 1,005,600 KiB. No teacher inference or training occurred.

Both value writers consumed the same actual post-publication B100 derive/policy hashes and original teacher metadata. Their source, native feed/content, provenance, nonvalue preservation and publication checks passed. All 32 paired output-shard WDL digests differ, confirming that the adaptive SF input reached the real output. This does **not** establish that every row differs: each writer's `changed_rows=262079` counter is relative to original B100, not to the other value product. Maximum stored WDL mass error was 0.0003662109375 for each product.

Independent completed-receipt review passed without repeating payload checks. [Compact materialization evidence](evidence/g10-value-readiness-20260911/value-materialization-run06.json) records all source/output summary pins and timings. The products remain integration artifacts, not a qualified training comparison or evidence of stronger values. Further G10 expansion or donor-mode implementation is deferred until its expected value warrants compute; Ceres collection and its registered policy/value comparisons retain priority.

## Original run06 large native BT4 WDL coverage completed, September 12

Native BT4 WDL labels now cover all **2,013,019 original run06 common-large rows
in 246 shards**. Four disjoint saved-output units cover shards 0–63, 64–127,
128–191 and 192–245. Together with the two earlier common increments
(526,376 rows), direct native-WDL coverage is **2,539,395 rows**. These counts
describe native teacher labels; matched adaptive-SF coverage remains the separate
1,574,952-row result above.

The final 54-shard unit contains 440,155 rows, including a final 5,979-row shard.
Collection exited zero in 317.01 seconds within the 600-second enclosing budget.
Its unchanged qualified producer retained batch 128, two threads and native
W/D/L probabilities. A terminal-only saved-output qualification exited zero in
7.13 seconds, checking all five arrays (28,169,920 decoded bytes), exact row and
teacher/source bindings, hashes and finite float32 probability mass. Maximum
unit-mass error was 1.4804e-7. Independent receipt review passed without another
array scan. No further GPU unit was launched.

The [consolidated readiness inventory](evidence/g10-value-readiness-20260911/native-wdl-run06-complete.json)
pins all four completions and saved-output qualifications and matches their
ordered union to the original source summary. This is not a consumer admission
manifest: the existing historical native-WDL reader requires one complete cohort
in one output directory, while these four units retain distinct invocation/output
namespaces. No files were moved or rebound to manufacture a merged lineage.
The inventory reuses prior qualified receipts without rereading their arrays.

This completes one source's native labels, not all G10 data, a new value target
corpus, or training admission. It does not repair the malformed adaptive-SF
baseline encountered outside the previously qualified prefix. Registered Ceres
strength comparisons retain priority; there is no new playing-strength result.

## Future generation: reject malformed candidate handoffs

A preserved failure in worker 2, game 99518, ply 318 explains why the full-large adaptive-value derivation stopped beyond its accepted prefix. The exact frozen source row has valid, unique legal phase-zero d9 scores (best −625 cp). Later phases contain repeated moves at different ranks. A d10 block marked complete by its rank IDs produced the next `searchmoves` roster `c3g7 d2d3 d2d3 d2d3`. The historical latest-phase value composition then accepted a duplicated d2d3 observation at 0 cp. The adaptive selector correctly rejected that ambiguous baseline. See [the source-bound diagnosis](evidence/g10-value-readiness-20260911/roster-handoff-diagnosis.json).

The future generator now checks a preceding block before using it to request another search: contiguous ranks, the requested full width, finite effective scores, and unique legal moves. Explicit `searchmoves` must match the preceding request's roster exactly; an unrestricted numeric first rung may return a legal top-k subset of its requested width. A failure stops extension and adds `extension_stop_reason: invalid_candidate_roster` to the observed phase; at the G10 decision it also records that gate reason. The valid phase-zero observations, malformed later observations, and parser anomaly counters remain available. Harmless repeated emissions of the same block still allow extension.

This does not redefine the parser's rank-based `complete` flag, deduplicate historical observations, or repair the frozen corpus. It prevents malformed candidate rosters from propagating into a subsequent search; it does not sanitize a malformed final phase with no further handoff or guarantee that every saved intermediate d9 block is valid. The latest-phase d9 derivation remains unchanged. No live generator restart or adoption is part of this change. The full failed derivation remains failed, the separately qualified prefix remains valid, and later native-BT4 coverage does not retroactively repair the SF baseline. One diagnosed row establishes the failure mechanism, not its frequency.

## Original run07 large native BT4 WDL coverage completed, September 12

The complete original run07 common-large cohort now has native BT4 WDL labels:
**2,008,952 rows in 246 shards**, with 1,912 rows in its final shard. Unlike the
four run06 collection units, this collection writes one complete output namespace.
Combined with run06 large (2,013,019 rows) and both earlier increments
(526,376 rows), direct G10 native-WDL coverage is now **4,548,347 rows**.

The existing qualified collector exited zero in 1,182.65 seconds (19m43s), within
a 2,100-second enclosing bound and 2,000-second child bound. It retained batch
128, two threads on CPUs 4–5, the same native probability teacher and mandatory
GPU lease. Its subsequent terminal-only qualification exited zero in 18.04 seconds
with 50,356 KiB reported peak RSS. All five saved arrays across every shard
(128,572,928 decoded bytes) passed their stored hashes, exact row/source/teacher
bindings and finite float32 probability checks; maximum unit-mass error was
1.4831e-7. Independent completion review passed without rereading the arrays.

Two earlier advisory preflights returned a busy lease before collection launched.
A bounded same-call probe subsequently acquired and released the lease; the
transient owner's cause remains unresolved. One actual collector attempt then
succeeded with its unchanged lease and resource guards. No lock was bypassed or
other process stopped. Launch preflight recorded 3,348 MiB GPU memory used, 6%
GPU utilization, no compute applications, 279.42 GiB free SSD space and
58,173,488 KiB available RAM; Ceres policy materialization was still running.
The [compact completion evidence](evidence/g10-value-readiness-20260911/native-wdl-run07-complete.json)
retains the failed preflights, successful probe, launch and both terminal receipts.

The single output directory makes the existing historical native-WDL consumer
route applicable in principle; its complete provenance manifest/admission and
writer content/feed checks have not been run for this cohort. This is completed
label readiness, not a new value target, training admission or scientific strength
result. The separate run06 adaptive-SF failure remains unchanged. No additional
GPU cohort was queued, and registered Ceres comparisons retain priority.

## September 12: complete next96 native-WDL cohort

The original run06 `G10_common_next96_v1` cohort now has native BT4 WDL for all
790,282 retained positions in 97 derived shards (last shard: 3,850 rows). Its 96
raw shards, `w03-00043` through `w03-00138`, contain 795,926 physical rows; the
existing qualification excludes 5,629 no-result rows and 15 policy-support misses.
Saved source selectors establish no raw-shard overlap with the already labeled
run06 large/increment cohorts; run07 has a separate source namespace. Direct G10
native-WDL coverage rises from 4,548,347 to **5,338,629 positions**. This count does
not include the separate original 18.91M native-WDL bank.

The unchanged collector completed in 382.22 seconds (382.35 seconds for its
operator). It used batch 128, two threads on CPUs 2–3, the same native probability
teacher and mandatory GPU lease. The 900-second inclusive allocation included an
800-second child bound and residual time for saved-output qualification. A 32 GiB
Linux available-memory guard and 150 GiB disk reserve remained active; the 8 GiB
ORT allowance is not a total-device or CPU RSS cap. The downside CPU rebuild on
CPUs 4–5 was preserved.

The terminal-only qualifier then passed in 3.11 seconds, within the original
deadline. All five arrays across all 97 shards—50,578,048 decoded bytes—passed
stored hashes, exact source/teacher/row bindings and finite float32 probability
checks. Maximum unit-mass error was 1.474e-7. A compact completion review checked
the retained proof records without reading arrays again. The initial metadata
attempt failed before admission because its working directory selected the wrong
Python package; the corrected working directory yielded one successful metadata
admission, with no inference or target changes.

[Compact completed evidence](evidence/g10-native-wdl-next96-20260912.json) pins the
original qualification through the collector plan, complete output, saved checks
and terminal handles. The single complete output directory supports the existing
consumer route in principle; its provenance-manifest admission and writer checks
have not been run. No new target, training admission, playing-strength result or
automatic further GPU cohort follows from this coverage milestone.

The [completed common-pool native-WDL batch](2026-09-12-conservative-research-recovery.md#completed-native-wdl-for-all-accepted-common-g10-positions) now supplies all 16,404,093 accepted positions. The completed consumer provenance admission is recorded below, including the older four-output run06 large cohort. No new training or strength result follows from this coverage.


## All 20 common-cohort native-WDL manifests admitted

The existing native-WDL consumer admitted all **16,404,093 positions across 20
cohorts and 2,012 derived shards** in one metadata-only pass. Nineteen cohorts use
the original single-directory route. The 2,013,019-row run06 large cohort uses
[PR #682](https://github.com/jjoshua2/DeepFin/pull/682)'s explicit four-directory
route, preserving each original invocation and output namespace. No labels were
moved, generated or rewritten.

The actual run finished successfully in **58.71 seconds**, with maximum recorded
RSS **263,156 KiB**, within a 600-second allocation on CPUs 6–7. GPU visibility was
disabled, numerical-library threads were limited to two, address space to 4 GiB,
and guards required 32 GiB available Linux memory and 150 GiB free disk. These
limits do not assert an aggregate Windows/GPU memory bound.

Each cohort made exactly one existing source-inventory call and one historical
native-WDL admission call. The 25 original invocation entries retain complete,
disjoint shard coverage, source qualification, native teacher/head identity,
separately pinned historical producers and accepted attribute hashes. The two
early pilot shards also matched their preserved storage fingerprints. A compact
completed review checked all 20 actual manifests, routing maps and admission
receipts without repeating the admission or reading array payloads. The
[complete compact evidence and per-cohort manifest map](evidence/g10-native-wdl-all20-admission-20260912.json)
pin the actual results and local artifacts.

This closes the historical manifest-admission gap. It does **not** create B100
policy products, blended-value targets or a training corpus. The full value writer
still checks cached contents, actual stored feeds and original SF/B100 lineage.
A targeted completed-product metadata inventory found the previously qualified
**262,079-row run06 increment B100 T0.5 product**. The other **19 cohorts,
16,142,014 rows**, still need their original-source B100 policy products before a
full value rewrite. The original 18.91M B100 corpus belongs to another source and
cannot substitute; the existing value-modified control/adaptive products cannot
substitute for the original B100 input either. No new mixture weight, target
materialization, training admission or strength claim is selected here.


## Remaining original B100 policy products launched

After the completed all-20-cohort native-WDL metadata admission, one sequential
CPU batch launched at **21:35:03 UTC on September 12** to build the remaining
**19 original-source B100 T0.5 policy products: 16,142,014 rows across 1,980
shards**. The previously completed 262,079-row run06 increment product is
preserved and excluded. Each cohort keeps its own original source and complete
output namespace; only one producer runs at a time. The recipe is unchanged
global B100 alpha 1, BT4 temperature 0.5, with original SF values and history.

The prior real 262,079-row B100 build took 104.01 seconds. Linear scaling suggests
**about 107 minutes**, with roughly 10.4 GiB logical / 12.3 GiB allocated output.
Those are one-cohort estimates, not a full-batch benchmark or guarantee. The
accepted allocation is **four hours inclusive**, with 300 seconds reserved for
final metadata checks and cleanup. The supervisor stops between cohorts if the
next complete cohort cannot fit; it reports only a completed prefix and does not
extend the budget or retry a failed producer automatically.

The actual launch uses CPUs **6–7**, two numerical-library threads, GPU hidden,
**4 GiB address space**, at least **32 GiB Linux available memory**, a **150 GiB
disk reserve**, and a **32 GiB sampled output cap**. Parent preflight reported
about 83 GiB available memory, zero swap use and 242 GiB free disk. These are
launch observations, not peaks. The independent input/output namespace permits
this one CPU producer alongside the existing downside and Ceres jobs; its guards
and cleanup affect only its owned process group.

The [compact launch evidence](evidence/g10-b100-remaining19-launch-20260912.json)
contains the exact plan, unchanged qualified producer runtime, independent
preparation review, per-cohort mapping and an immutable small launch snapshot.
This is **policy target building in progress**, not a completed corpus, new value
mixture, training admission or playing-strength result. No further job was
launched for this publication.


## All original B100 policy products complete

The remaining-19 batch completed at **2026-09-12T22:47:22.095072+00:00**,
with exit zero after **4,338.6108 seconds** (72.31 minutes). All **19 cohorts,
16,142,014 rows and 1,980 shards** completed within the four-hour allocation;
this is a full completion, not a budget-stopped prefix. Together with the
previously completed and retained **262,079-row / 32-shard** run06 increment
product, original-source B100 policy targets now cover all **20 common G10
cohorts: 16,404,093 rows / 2,012 shards**.

Each new product has the unchanged global recipe **alpha 1, BT4 temperature
0.5**, with `policy_target` as its only mutated array. Original SF value
columns, history and cohort lineage remain unchanged. Independent completion
review checked all 19 small recipe/derive summaries, ordered layouts, source
lineage, timing receipts and final proof metadata. Every stage returned zero;
the supervisor's terminal path rechecked input/output storage identities and
pinned evidence before declaring full completion. The reviewer did not repeat
array or storage-tree audits.

The largest individual producer peak reported by `/usr/bin/time` was
**1,011,244 KiB** (about 988 MiB). This measures one producer, not simultaneous
whole-host use. CPUs 6–7, two numerical-library threads, GPU hidden, the 4 GiB
address-space cap, 32 GiB host-memory floor, 150 GiB disk reserve and 32 GiB
sampled output cap stayed in the executed plan. No resource limit was expanded
to obtain this result.

The exact terminal status is
`COMPLETE_ALL19_B100_POLICY_NOT_VALUE_OR_TRAINING_ADMISSION`. This closes the
original B100 policy-product prerequisite for the full-cohort value writer.
It does **not** create a new SF/BT4 value blend, qualify a training corpus or
establish playing strength. The follow-on value writer must still verify
native cached contents, actual stored feeds and original SF/B100 lineage;
no follow-on launch was claimed at that completion snapshot.

[Compact completion evidence](evidence/g10-b100-all20-completed-20260912.json)
contains per-cohort source/output mappings, recipe and summary hashes, timings,
resource limits and independent review. Local terminal state and the 19 new
products are under
`scratchpad/bt4_joint20/g10_b100_remaining19_preparation_v1/execution/`;
the retained run06 increment product remains at
`scratchpad/bt4_joint20/g10_run06_value_materialization_v1/execution/B100`.
The parent's root session 99876 exited zero and sole observer 141 is closed.


## Remaining SF50/native BT4 value targets launched

After all 20 original B100 policy products completed, the fixed **50% original SF
plus 50% native BT4 WDL** writer launched at **22:54:39 UTC on September 12**.
One sequential CPU batch covers the remaining **19 cohorts / 16,142,014 rows /
1,980 shards**. The previously qualified **262,079-row** original-SF/native-BT4
control is preserved and excluded from the rewrite. This continues that existing
recipe; no dose sweep or adaptive-SF input is selected.

Every command binds the actual completed B100 derive and policy summary hashes.
B100 T0.5 policy remains unchanged, while the original SF corpus retains the
native teacher's source qualification and feed identity. The historical manifests
are not relabeled as B100 inputs. Run06-large uses its reviewed four-output
native-WDL mapping. The unchanged writer still checks cached probabilities,
actual stored feeds, row identities, original SF/B100 nonpolicy lineage and final
storage identities before publishing each complete target product.

The earlier 262,079-row original-value integration took **65.41 seconds**, with
**394,652 KiB** maximum reported process RSS. Linear scaling suggests **about
67 minutes** for the remaining rows; this is a one-cohort extrapolation, not a
full-batch throughput or memory guarantee. The actual allocation is **four hours
inclusive**, including 300 seconds reserved for final metadata checks. A cohort
starts only if its complete allowance plus that reserve remains. Failures and
budget stops preserve partial artifacts and report only completed products;
there is no automatic retry or extension.

The launch uses **CPUs 6–7**, two numerical-library threads, GPU hidden,
**4 GiB address space**, at least **32 GiB Linux available memory**, **150 GiB
free disk**, and a **32 GiB sampled output cap**. Parent preflight observed about
83 GiB available memory, zero swap use and 225 GiB free disk. These are startup
observations, not peaks or an aggregate Windows/GPU memory bound. Only one value
producer runs at a time.

The [compact launch evidence](evidence/g10-native-value50-remaining19-launch-20260912.json)
contains the exact plan and runtime, independent preparation review, actual
B100 summary bindings, all 19 output mappings and an immutable launch snapshot.
Parent root session **39995**, sole observer **184**, owns completion. This is
**target materialization in progress**, not completed full-cohort value targets,
training admission or a playing-strength result. No job was launched or polled
for this publication beyond the single small launch-state snapshot.


## All original-SF/native-BT4 value products completed

The remaining **19 cohorts / 16,142,014 rows / 1,980 shards** completed on
September 12 at **23:52:17 UTC**, in **3,458.751 seconds** (about 57.65 minutes).
All nineteen timed writers and the enclosing operator exited zero. Together with
the preserved, previously qualified 262,079-row / 32-shard control, fixed
**50% original SF + 50% native BT4 WDL** targets now cover **all 20 cohorts /
16,404,093 rows / 2,012 shards**. No existing control was regenerated.

Every new row's stored `search_wdl` changed. The writer normalized the two WDL
probability distributions, averaged them at equal weight and stored float16
results. The maximum recorded mass error was **0.000366211**. Only `search_wdl`
changed: sixteen other arrays, including B100 T0.5 policy, `wdl_target`, features
and row identities, passed the producer's unchanged-file checks. This retains
the original SF anchor; the adaptive d10/d12 selector is not enabled in this
recipe. It does not add a Ceres component.

The actual writer also completed the **2,013,019-row / 246-shard run06-large**
cohort through its schema-2 **four-root native-WDL mapping**. This advances beyond
metadata admission: the frozen writer checked native cached contents, actual
stored-feed digests, game/ply alignment and original source-array proofs while
writing. All 1,980 new shards have saved output-file-manifest, attribute,
source/sidecar storage and stored-value digests. Existing native predictions were
reused with **zero new teacher evaluations**. The independent readout verified
complete saved recipe, layout, native-manifest and proof metadata; it did not
repeat feature/target decoding or storage-tree scans.

The largest individual timed producer RSS was **402,160 KiB** (about 393 MiB).
The executed four-hour allocation, sequential CPU producers, CPUs 6–7, two
threads, hidden GPU, 4 GiB address-space limit, 32 GiB available-memory floor,
150 GiB disk reserve and 32 GiB sampled output cap were unchanged. An individual
process peak does not establish aggregate host memory usage.

Terminal status is
`COMPLETE_ALL19_SF50_NATIVE50_VALUE_NOT_TRAINING_ADMISSION`; parent root 39995
and sole observer 184 are closed with exit zero. The
[compact completion evidence](evidence/g10-native-value50-all20-completed-20260912.json)
contains per-cohort output locations, summary/native-manifest hashes, timings,
resource limits and independent review. It supersedes the preceding launch
snapshot for completion status. These are materialized targets, not an admitted
training corpus, trained checkpoint or demonstration that V50 improves playing
strength on G10. The next stage can qualify and schedule a matched comparison;
it need not regenerate these teachers or targets.
