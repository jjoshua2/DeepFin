# Weighted Ceres bootstrap preparation

Status: B100CeresV25 completed its registered seed-zero epoch. Its first fixed-400 match against B100 launched September 12 at 14:56:36 UTC after waiting for the shared GPU lease; no value playing result is available. The B100V50 comparison is also prepared and remains required. The CeresB50 policy result remains unresolved at +12.22 Elo [-22.88, +47.57].

## Selected question

The [tactical SF attenuation experiment](2026-09-10-bt4-sf-tactical-training.md) did not demonstrate a gain. Retain B100 (BT4 policy at teacher temperature 0.5, original SF values) and explore a distinct teacher mixture.

The first candidate averages independently normalized BT4 and Ceres policies with equal weights and teacher temperature 0.5 each, holding SF values fixed. Use raw BT4 probabilities and apply sharpening once. Complementary teacher mistakes motivate this test; disagreement alone is not evidence that averaging improves play.

The separate value candidate is 50% SF / 25% BT4 / 25% Ceres, with B100 policy fixed. Its producer and training admission are implemented and independently reviewed. Normalize native SF and BT4 WDL probabilities; convert Ceres raw primary and secondary heads separately with softmax temperatures 0.55 and 1.5, then combine them 60/40. The final target is therefore 50% SF, 25% BT4, 15% Ceres primary and 10% Ceres secondary. All distributions use win/draw/loss order from the side to move. This probability calculation is a mathematical analogue, not verified native Ceres FP16 parity.

## Collection allocation

The new validation chunk covered 16 distinct full shards: 131,072 positions, 4,096 fixed32 calls and zero padding. Independent review decoded and checked all 144 saved arrays, source metadata bindings, legal rosters, provider evidence and loaded libraries. Total driver time was 703.61 seconds, including 245.46 seconds of admission/possible GPU waiting; post-lease completion took 453.29 seconds and inference calls totaled 331.81 seconds. The saved bank occupied 21.34 MB. These are measured collection costs, not strength evidence.

The next allocation covers shards 32–2307 in 143 disjoint chunks, totaling 18,644,992 real rows and 582,656 calls. Reuse accepted shards 0–31 and final partial shard 2308 once. Successful completion would yield 18,910,484 distinct rows; a launched plan is not completed coverage.

Each chunk has a 1,800-second cap and at most 128 MiB of bank/state output. The driver has a 30-hour cap, 30-second inter-chunk pauses and a 150 GiB disk reserve. The collector owns the shared GPU lease, allowing existing BT4 labeling to continue between operations. Stop on failure; retain completed chunks without automatic retries. At observed costs, the remaining allocation projects to about 19.18 hours excluding variable waits or 29.11 hours using observed whole-chunk time, including pauses. It may reach the cap before finishing.

The backend remains the qualified fixed32 ORT1.29/CUDA13 approximation for C3-768-30-pre8-I8. It retains raw legal policy and both raw WDL heads. This does not establish native Ceres numerical parity or engine strength.

## Telemetry failure and recovery

The next chunk (32–47) completed and independently passed all 144 array checks,
source bindings, provider observations and first/final runtime identities. The
following chunk (48–63) stopped after 121.08 seconds because a memory query returned
exit code 3. Subsequent GPU queries succeeded and existing BT4 labeling continued;
this observation does not establish a GPU hardware failure.

Two shards from the failed invocation have intact arrays but lack final runtime
observations and completion receipts. Preserve those files without admitting them.
The independently reviewed recovery launched at shard 48 and retains qualified 0–47. It covers
18,513,920 rows in 142 chunks, with 107,297 execution seconds remaining from the
original 108,000-second allocation after charging the failed run's 702.85 seconds.
Repair time is excluded from that execution budget. Existing numerical, memory,
lease and disk guards remain unchanged.

The new adapter uses a bounded synchronous telemetry helper: retry only exit code 3
or subprocess timeout, at most three attempts of at most three seconds, within a
9.2-second logical deadline. It records every failed query and checks STOP/deadline
before and after attempts. It never substitutes a stale reading or retries the
collection workload. The enclosing process timeout remains the hard termination
bound. Helper behavior passed 17 mocked tests and independent review.

The first recovery chunk subsequently completed in 532.32 seconds and passed
independent checks of all 144 saved arrays and final runtime evidence. It recovered
one actual telemetry exit-code-3 failure without relaunching inference. A later
bounded snapshot audit qualified recovery shards 48–79 (262,144 rows), scanning only
the newly completed chunk and reusing unchanged pinned prior qualification. This
snapshot excludes in-progress chunks and is not full-corpus completion.

## Qualified collection progress, September 11 at 23:12 UTC

Snapshot 15 qualified recovery shards 48–2063: 16,515,072 rows. Together with
previously accepted shards 0–47 and final partial shard 2308, this gives
**16,911,636 of 18,910,484 positions (89.43%)**, across 2,065 shards. Audit session
19685 exited zero under the existing ten-minute, two-core, 2 GiB limit. Twenty new
chunks received saved-array checks; 106 unchanged qualified chunks reused their
pinned prior evidence. In-progress output is excluded. This is label
qualification, not completed mixture training or playing-strength evidence.

[Compact snapshot15 evidence](evidence/ceres-snapshot15-20260911.json) pins the
completed snapshot (`ca515de4a06cb2c60bb5ee822643eb81bca7409aa40c5a8071b17eaf353ab48c`),
its snapshot14 predecessor, and the unchanged audit helpers. The publication check
matched all 126 terminal chunk receipt hashes and checked the reused review chain
and exact coverage without repeating array verification. The earlier
[snapshot14 milestone](evidence/ceres-three-quarter-20260911.json) remains preserved.

Collection continues. The last host process observation at 23:00 UTC found driver
276374 and child 554254 collecting shards 2064–2079; those rows are outside this
qualified snapshot. This observation is not a continuous liveness claim. No
collector restart or inference was added by the audit. The next substantive work
remains full-bank materialization and the registered policy/value comparisons below.

## Full original-corpus saved bank, September 12 at 01:41 UTC

Snapshot 16 completes recovery coverage: **18,513,920 rows in 2,260 shards**
(48–2307), across 142 completed chunks. Adding the previously accepted 396,564
rows in shards 0–47 and final partial shard 2308 yields **18,910,484 distinct rows
in 2,309 shards: 100% of the original corpus**. Sixteen newly completed chunks
received saved-array verification; 126 retained their unchanged pinned prior
qualification. Audit session 37900 exited zero. The publication check matched all
142 terminal receipt hashes, the reused review records and exact recovery coverage
without repeating the payload audit.

The recovery driver completed in 81,733.77 seconds (22 hours 42 minutes), within
its remaining allocation. This elapsed time includes inter-chunk pauses and other
execution overhead; it is neither GPU kernel time nor a fresh throughput benchmark.
The failed earlier invocation remains excluded. The original collection driver recorded `COMPLETE`; its observer closed and the
host process was gone. Root session 47292 returned no captured exit code, so the
completion claim rests on the driver receipt. Collection has ended.

[Compact final coverage evidence](evidence/ceres-full-coverage-20260912.json)
pins snapshot 16 (`a156bef0bb4d997cffde3d9b06e49c23c0d1c45274c76d7363d2e21fbf9430a0`),
the terminal driver, unchanged auditors and snapshot 15 ancestry. Bulk per-shard
records remain at the pinned host paths. Earlier progress snapshots above are
historical observations, superseded by this completed coverage.

This completes the saved teacher bank, **not training-corpus admission or a strength
result**. Raw policy and both value heads are available under the existing qualified
approximate backend. Both complete producer manifests subsequently passed their actual producers’
manifest readers (assembly session 51620 exited zero). Their hashes and the
`COMPLETE_MANIFEST_ASSEMBLY_NOT_CORPUS_ADMISSION` receipt are included in the
compact evidence. This was metadata assembly, not target rewriting. The next work
is materializing the registered policy and value mixtures, followed by their existing
training and match allocation. No target weights or scientific promotion criteria
change because collection finished.

After plan and command review, the CeresB50 full-corpus materializer launched in
root session 61364 under the existing eight-hour bound, on CPU cores 0,1 with
the GPU hidden. Its plan SHA256 is
`1bc2189980f8551c765cbe9d97669d6c362be56a37e17af97395365175090011`.
This is a launch observation, not a completed corpus. The B100CeresV25 value
materialization plan was prepared but not yet launched at that snapshot; its subsequent concurrent launch is recorded below. Both original plan identities are
in the compact evidence; training admission remains a later step. A separate [G10 convenience pilot](2026-09-11-g10-ceres-pilot.md)
also completed collection and saved-output qualification; its diagnostic remains
separate from the original-corpus training anchors.

## Real-teacher policy pilot

One actual 8,192-row shard passed the equal-weight, temperature-0.5 policy mixture
pilot in 14.48 seconds on CPU. Original SF values and all source storage identities
were unchanged. Maximum stored probability-mass error was 0.0003602, maximum
rounding total-variation error was 0.0001761, with no lost support. The standalone
pilot array is not an admitted training corpus or a playing-strength result.

## Producer and training compatibility

The policy-only producer uses ordinary copied shards, preserving all 16 nonpolicy arrays. It checks complete source/teacher coverage, actual Ceres input/history and legal alignment, and atomic output completion. Legacy BT4 root-position keys do not newly prove historical input frames: the producer pins the original collection and records the inherited limitation. Stronger full-input hashes are checked wherever available.

Independent review identified that provenance distinction, which was fixed and rereviewed. The producer/admission tests exercise real shard rewriting, loader/collator/loss propagation and meaningful rejection cases. A separate 32-row fixture also passed the frozen Python3.10 / NumPy1.26.2 / Torch2.11 consumer: policy loss and gradients changed while SF value loss and all nonpolicy arrays remained identical. This qualifies a tiny fixture's consumption, not the full corpus or training epoch.

Before training, qualify the completed full-corpus manifest and realized targets, freeze producing-code identities, and verify the same historical initialization, row schedule, update budget and runtime as B100. Register the deciding match and its resource/stopping rule before launching. A new trainer or sampling regime would require a fresh matched control.

## Value-only implementation and pilot

The value producer copies the B100 corpus and changes only `search_wdl`, retaining
policy and the other 16 arrays. It preserves the original BT4 WDL collector's
historical identity, checks actual stored inputs and teacher feeds, and requires
complete source coverage before atomic publication. Teacher-array integrity does
not establish missing invocation-finalization evidence; only independently qualified
Ceres shards may enter the collection manifest.

Tests cover actual storage, loader/collator and loss propagation: policy gradients
remain identical while value gradients change. Twelve producer tests and 163
admission/one-epoch tests passed, with 14 focused admission tests passing after final
fixture cleanup. Independent review found no implementation blockers.

The real-teacher value pilot checked one 8,192-row shard against the actual historical
BT4 WDL and qualified Ceres dual-head output. It completed in 10.02 seconds on CPU,
using about 656 MiB peak RAM. All 8,192 targets changed; maximum stored mass error was
0.0003662, with zero support losses. Original source identities, B100 policy and
other input arrays remained unchanged. Its standalone value array does not admit a
full corpus or demonstrate a strength gain.

The historical consumer also passed a separate 32-row synthetic-teacher fixture
in 19.60 seconds overall, using the frozen Python 3.10 / NumPy 1.26 / Torch 2.11
runtime with GPUs hidden. Policy loss and gradients remained identical; value loss
and gradients changed. All 16 other stored arrays were byte-identical and all 69
imported project modules matched the unchanged historical runtime. This establishes
the exercised target-to-loss path, not full-corpus admission or strength.

## Remaining preparation handoff

The materialization supervisor and corpus qualifier now provide the two-profile
handoff to the existing training coordinator. Producer summaries bind each completed
output shard's storage identity and recheck it before atomic publication. The
qualifier uses those identities plus actual metadata and the successful terminal
receipt, avoiding another full payload/history scan. No target math, historical
trainer or training schedule changes are introduced.

This tooling does not qualify an unfinished teacher bank. Complete collection
coverage, accepted invocation evidence, concrete producer manifests, realized corpus
publication and the prospective schedule remain required before either registered
training launch. The earlier real-teacher and historical-consumer pilots remain
separate, pinned evidence for their exercised versions; they are not relabeled as
full-corpus runs.

## Registered next training and match allocation

This allocation begins only after the same complete 18,910,484-row Ceres manifest
is qualified. It does not treat a partially labeled subset as the original corpus.
Train the two already selected recipes separately:

| Candidate | Policy target | Search value target | Comparisons |
| --- | --- | --- | --- |
| CeresB50 | Equal BT4/Ceres policies, each teacher T=0.5 | Original SF | B100 |
| B100CeresV25 | Unchanged B100 policy | 50% SF / 25% BT4 / 25% Ceres, conversions above | B100 and B100V50 |

Run policy training first, then value training. Each uses the historical seed-zero
initialization and exact canonical row schedule: 18,910,484 rows, batch 512,
36,935 updates and 420 windows, with the same frozen trainer/runtime and non-target
settings as its controls. Each training has a 16,200-second cap; each enclosing
operation allows 21,630 seconds including waiting, cleanup and realized-schedule
verification. Use the training-only coordinator profiles, not a legacy automatic
multi-depth arena sequence. Admission must verify final corpus/producer identities,
prospective and realized schedules, actual runtime, and the final checkpoint.
If compatibility with an old control fails, stop that comparison and register a
fresh matched control before spending training compute; do not label an unmatched
historical comparison a recipe test.

Each of the three direct matches gets 128 swapped opening pairs (256 games),
400 simulations per side, search-prior temperature 1.0, maximum 300 plies and no
tablebases. Reuse the explicit training search shape and development opening panel
from the tactical screen (seed 20260909, panel hash below); only checkpoint identity
changes. The existing rolling match path permits up to 128 concurrent games and
batch 4096 under its qualified resource guards. Each match has a 5,400-second owned-stage
cap and a 10,230-second enclosing-operation cap, including lease waiting and
cleanup, matching the qualified launcher. These are common-search-setting development tests, not a measurement
of search scaling, optimized play settings or unseen-opening generalization.

Freeze the exact match package and output identities before each launch. All
completed swapped pairs enter the paired score/95% interval readout. A timeout or
incomplete registered match is not a completed fixed-horizon result. No automatic
game extension, additional depth or temperature sweep is allocated. Existing GPU
leases, STOP markers, RAM guards and 150 GiB disk reserve remain in force; collection
and source generation retain their existing recovery and scheduling contracts.

For allocation decisions, an estimated gain of at least 15 Elo earns consideration
for the next investigation; a lower interval bound above zero additionally supports
a win for the tested checkpoints. An upper bound below +15 stops ordinary refinement
of that candidate in this regime. Other outcomes remain unresolved, without demanding
more games on the same pair of models. These thresholds do not rank entire teacher
families or account for training-seed variability. The three comparisons are
exploratory; report them separately without treating their intervals as a joint
confirmation test. The value candidate faces both controls regardless of its first
match result, so Ceres's incremental contribution is not selected by a favorable
SF-baseline result.

Use the completed results to choose the next substantive experiment. Combining both
head changes, adjusting teacher weights or testing a pure-Ceres endpoint is not an
automatic queue. Fresh matched training seeds and fresh openings remain necessary
before a promotion claim, followed by transfer to the largest qualified common
corpus relevant to the intended 100M bootstrap.

Registered controls: B100 checkpoint SHA256
`b30ab345d0cf3acfb51bea6c90a91aef3c1dd5edb78da3c92d3a504fb2735d62`;
B100V50 checkpoint SHA256
`f813e47e43907f1dd7b716a191b69f7444d9b70bdce53cd90010d613d6e504a8`.
The development panel SHA256 is
`3c955d68a6c010e373b44418f3bbbf1500cf9192d04ca17ffdd3dcd191d7cbb1`.
Final launch manifests must verify these identities and pin newly produced artifacts;
this allocation is not itself a runnable or completed launch manifest.

## Concurrent registered materialization, September 12

B100CeresV25 preparation launched at 04:56:27 UTC alongside the active CeresB50
policy writer. This overlaps preparation of the two already-registered recipes;
it does not launch training or add a new target variant. Each corpus remains
18,910,484 original positions. Batch size stays 128, and every transitive value
producer file retains its previously qualified bytes.

The policy job remains on its original frozen runtime and CPUs 0–1 with the shared
preparation lock. The value job uses CPUs 2–3 and the fixed value-profile lock,
with supervisor-only runtime `0719ad87e763250ffee45aab63190a9fd9659fd6` based on
`f6b8b83aa0a17a4170a4c4e30a62b5ff08eb8a9e`. [PR #662](https://github.com/jjoshua2/DeepFin/pull/662)
adds these optional allocations while preserving legacy defaults. The two jobs
read shared immutable SF/Ceres inputs and write disjoint output/state directories;
no active runtime was edited. Both retain two numeric threads, no GPU, STOP,
150 GiB disk reserve, sampled 32 GiB output cap and eight-hour enclosing deadline
with owned-process cleanup. No RSS/address-space cap is implied.

Twenty-six focused subprocess/guard tests passed, scoped host type checking had
zero errors/warnings, and configured whole Ruff/Vulture checks passed. Independent
source and frozen-plan reviews passed; metadata-only producer admission exited zero.
Whole-project type checks were not repeated after related broad timeouts, and no
whole-type pass is claimed. Parent launch preflight recorded 294.316 GiB free disk,
56,392,632 KiB available memory and no applicable STOP markers.

The retained policy baseline showed 1,135 of 2,309 policy-stamped shard directories
after 10,866 seconds, approximately 0.10445 shards/second since launch. Those stamps
measure preparation progress, not completed corpus qualification. One check about
20 minutes into concurrent preparation compares the interval rate with this baseline.
If B50 clearly slows and its projected completion reaches within 30 minutes of its
original deadline or later, stop only the new value job through its own STOP/cleanup
path. Do not extend B50's deadline or automatically resume the value attempt.

The scheduled check observed 1,313 policy-stamped shards: 178 additional shards
in 1,489.49 seconds, or 0.11950 shards/second versus the 0.10445 baseline.
The projected policy completion retained approximately 135 minutes before its
original deadline, so neither stop condition held and both jobs continue. The
value output had 176 directories; this check did not qualify their completion
stamps. Free SSD space was 291.85 GiB and available RAM 53.59 GiB. The interval
includes about 230 seconds before value launch and overlaps BT4 labeling, so this
is a contention check, not evidence that concurrency caused a speedup. Projection
also excludes unmeasured final publication overhead.

While the GPU was available, one existing native BT4 WDL unit completed on CPUs
4–5: run06 common-large derived shards 128–191, exactly 524,288 rows in 359.51
seconds, within a 600-second enclosing bound. The unchanged qualified producer
used batch 128 and native W/D/L probabilities. A subsequent saved-output check
verified all five arrays across 64 shards (32 MiB decoded), exact row/teacher/source
bindings, stored hashes, finite float32 probabilities and unit mass; the largest
mass error was 1.465e-7. Independent receipt and checker-delta review passed. No
source feature or model arrays were reread for that check.
The run06 large bank now has 1,572,864 completed rows; including the earlier two
common increments, direct native-WDL coverage totals 2,099,240 rows. This is partial
scale preparation, not full-source or training admission, and it does not repair
the separate malformed adaptive-SF baseline outside the qualified prefix. No
additional GPU unit was queued.

The [compact progress evidence](evidence/ceres-concurrent-preparation-20260912.json)
retains launch, baseline, the complete co-run checkpoint and native-WDL completion
pins. Successful materialization and existing corpus admission still precede any
registered training; there is no new playing-strength result.

## Compact evidence identities

| Evidence | SHA256 |
| --- | --- |
| First new chunk independent readout | `b4317054250af4b410bb639c2233de9d75d0f899e4c424f1913db437ef0bb711` |
| First new collector completion | `74e79c222e96d4dc03b957050cbe18866901739267755b466259858245d15b40` |
| Remaining collection driver plan | `3a3e44f20ce2904a7bafd6d5b15ae73b4a4712188b1dca5b3a0c8a32bd61698b` |
| Remaining collection independent review | `f97cb4dbd6812c886f23ea9c7d06abd0167e4f34a2ae2f43c2120295555c9764` |
| Real-teacher value pilot | `0df10b9a7288385f2685881f734bc1ffd4bb0f8a2b8b62b5f04951847961da3b` |
| Recovery coverage review | `5487e2a3d7781cae23812deb680a825f0a1528251ad2693983b6cb7dc1cde9a8` |
| First recovery chunk independent readout | `ad02a3c2529af736b5abc5c580e4df500a642fe9a276147d4a4040d2700a030d` |
| Completed recovery snapshot 01 | `e20953e4a9f93d56baaa62379913a4186bb812ef022029521d5241c72662bd06` |
| Fresh recovery driver plan | `053c1365cbeee1919141007c0b4e7bf94997733f4c6e47b1c26f52cd180e36ae` |
| Real-teacher policy pilot | `b25ae0621f3e87e3d66bd6aa7b1437198f6917396531d082de4da5e49da5f922` |
| Frozen value consumer readout | `155adc4c1bfa24f9e565f3d03f12fee6569dff00e9def9206198ddc2733ab024` |
| Frozen consumer fixture readout | `89b625ac4d30eb89484a6cb1066803b058859a6279d498f8f47b44f36ecaf4e4` |

Bulk labels, executable manifests, logs and receipts remain in the host experiment storage. These identities bind the launch snapshot; future completion and training results belong in this record.

## CeresB50 corpus completed

The original batch-128 producer completed the full registered corpus in 5.57 hours. The separate frozen qualifier passed in 529.91 seconds with 381,260 KiB peak RSS, checking all 2,309 shard layouts, recipe attributes, and stable producer-bound source/teacher/output identities. It inherited the completed producer’s payload checks rather than decoding the corpus again. [Completion evidence](evidence/ceres-b50-training-handoff-20260912.json) binds the published summaries, actual COMPLETE receipt, qualification plan, and terminal.

The prospective schedule passed in 421.83 seconds: 18,910,484 rows and 36,935 batches match the frozen canonical seed-zero epoch. The independently reviewed training manifest retains Python 3.10.12, NumPy 1.26.2, Torch 2.11.0+cu128, batch 512, and 16/16 plan/load workers. The coordinator launched at 07:56:06 UTC and the actual trainer stage started three seconds later; its log reached 176 of 36,935 steps at this launch snapshot. The operator inherited CPUs 0–31, preserving the historical worker layout. Training has a 16,200-second cap and the enclosing operator a 21,630-second cap; the parent owns its completion observer. There is no completed model or playing result yet. The fixed recipe remains equal sharpened BT4/Ceres policy with original SF values; this preparation establishes corpus readiness, not a stronger network.

## Value corpus completed

B100CeresV25 materialization completed at 2026-09-12 09:19:35.364403 UTC, after 4.39 hours. The full 18,910,484-row, 2,309-shard corpus passed the existing qualifier in 722.56 seconds with 384,284 KiB peak RSS. The actual value-lane producer map and original SF/B100/BT4/Ceres lineage are retained in [compact evidence](evidence/ceres-value-training-readiness-20260912.json).

This prepares the registered 50% SF / 25% BT4 / 25% Ceres value mixture with unchanged B100 policy. Its prospective schedule passed in 463.960 seconds, matching all 18,910,484 rows and 36,935 batches of the canonical epoch. The final schema-3 manifest binds the actual 17-entry value-lane producer map, completed qualification, and schedule; the reviewed host operator preserves the 16,200-second training budget and 21,630-second outer bound. At that snapshot, value training had not launched and no playing result was claimed. At that value-readiness snapshot, the earlier CeresB50 training launch remained the latest policy-training observation.

## Policy epoch completed

CeresB50 completed the registered epoch with 11,055.006 seconds of charged training-stage time. Its realized schedule matches canonical `dc687fc3…`; checkpoint `5d7e1e81…` is bound to the completed training receipt and summary. [Compact completion evidence](evidence/ceres-b50-completed-match-20260912.json) retains the actual operator terminal, realized schedule, and independent review.

The historical `valid_control=false` flag remains: there is no held-out purity receipt, architecture/trainer assumptions use committed pins rather than a fresh live-file comparison, and game-epoch sampling differs from the replacement-sampled historical control. This qualifies the registered matched-epoch comparison, not a continuation of that older control protocol.

The next match is the registered 400-simulation comparison with B100 over 128 swapped opening pairs (256 games), at search-prior temperature 1.0 for both. Actual CPU package preparation passed: both models have 61,444,448 parameters and matching architecture, and the registered opening panel and search settings match the final contract. The match operator acquired the GPU lease after 19.168 seconds and launched its owned stage at 11:16:50 UTC. It uses rolling concurrency 128 and evaluation batch cap 4,096, with a 5,400-second stage cap and 10,230-second enclosing allocation. No game outcomes were read for this launch record; a completed paired bank is required before interpreting playing strength.

## Completed fixed-400 policy result and readout recovery

CeresB50 scored 51.758% over the registered 128 swapped pairs (256 games) against B100: estimated +12.22 Elo, nominal 95% paired interval [-22.88, +47.57]. The pentanomial counts are 18 / 23 / 51 / 22 / 14 in WW / WD+DW / DD+WL / LD+DL / LL order. Both packages used 400 simulations and search-prior temperature 1.0. This result establishes neither a gain nor equivalence; it does not select an optimal teacher mix or resolve variation across training seeds. The matched-epoch scope and historical-control limitations above remain. Keep B100 as the incumbent and CeresB50 as a competitive alternative; prioritize the registered value contrast rather than automatically extending this match or tuning a fine Ceres-dose grid.

The arena itself exited successfully and saved all games, but its enclosing operator exited 1 when the strict reader rejected a command observation. The saved child command exactly matched the recorded timeout supervisor argv, including its 5,370-second TERM deadline, 30-second KILL grace and exact workload suffix. This is consistent with sampling between fork and exec; the actual post-exec argv was not retrospectively observed. Original receipts and game banks remain unchanged.

The durable capture fix defers only that exact inherited wrapper for a bounded observation window, retaining unexpected commands for strict rejection. Missing observations remain possible; explicit null observations remain invalid. An explicit reader option admits this exact supervised snapshot only after supervisor/child identities, hard budget and elapsed time checks; all checkpoint, opening-history, settings, pair-completeness and score checks remain. The separately reviewed CPU recovery passed in 4.875 seconds with 235,824 KiB peak RSS, without model inference or game reruns. [Compact evidence](evidence/ceres-b50-match-recovered-value-launch-20260912.json) binds the original failure, unchanged process/bank, revised reader, completed result and independent review.

B100CeresV25's coordinator launched at 11:45:43 UTC and its actual trainer at 11:45:58 UTC after the policy match ended. It uses the already qualified 50% SF / 25% BT4 / 25% Ceres value target and fixed B100 policy, with the registered 16,200-second training and 21,630-second outer limits. This is a launch observation, not a completed value model or playing result. The capture correction does not change this already-running frozen training runtime.

## Value epoch completed

B100CeresV25 completed its registered epoch with 10,028.655 seconds of charged training-stage time. Checkpoint `3e2653d0…` is bound to the actual completed training receipt, run summary and realized canonical schedule `dc687fc3…`: 18,910,484 rows, 36,935 updates and 420 windows. The fixed value recipe remains 50% SF / 25% BT4 / 25% Ceres with unchanged B100 policy.

The same three historical `valid_control=false` limitations remain: no held-out purity receipt, committed architecture/trainer assumptions rather than a fresh live-file comparison, and game-epoch sampling rather than historical replacement sampling. Completion qualifies the registered matched-epoch comparison; it does not establish playing strength or erase those limitations. [Compact evidence](evidence/ceres-value-completed-match-20260912.json) binds the actual terminal, training receipt, summary, realized schedule and independent review.

Both registered controls remain B100 and B100V50, each over 128 swapped opening pairs (256 games), 400 simulations and prior temperature 1.0. Both comparisons remain required regardless of the first result. Bounded CPU preparation uses CPUs 2–3 to avoid the active downside producer on 4–5; no training or arena algorithm changes follow from that scheduling choice. Both actual CPU packages passed: candidate and controls each have 61,444,448 parameters and matching architecture, with CUDA uninitialized during preparation. Their stages took 127.598 and 118.547 seconds. No value result has been read.

The first standalone launch preflight stopped before invoking the launcher because an existing raw BT4 labeling job was using the GPU (8,883 MiB and 96% utilization observed). Both jobs use the same advisory lease. No process was interrupted or resource cap raised: the unchanged reviewed launcher entered its existing bounded lease wait instead. The match operator started at 14:52:30 UTC, acquired the lease after 246.688 seconds, and launched its actual arena stage at 14:56:36 UTC. This time the recorded child argv matches the requested arena command; the narrowly adopted future capture fix does not change arena code or search settings. The parent owns the sole match completion observer. The second B100V50 match is prepared for its slot after the first terminal, independently of the first result; the ongoing downside CPU preparation is preserved.
