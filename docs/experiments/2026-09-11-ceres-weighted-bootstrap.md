# Weighted Ceres bootstrap preparation

Status: qualified Ceres coverage now includes full shards 0–47 and the final partial shard. Collection stopped on a GPU telemetry query failure; an independently reviewed recovery is running. The policy mixer is merged and its real-teacher pilot passes. No Ceres-mixture training or playing-strength result is claimed.

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
