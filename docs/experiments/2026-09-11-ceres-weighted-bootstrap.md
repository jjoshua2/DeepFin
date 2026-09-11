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

## Compact evidence identities

| Evidence | SHA256 |
| --- | --- |
| First new chunk independent readout | `b4317054250af4b410bb639c2233de9d75d0f899e4c424f1913db437ef0bb711` |
| First new collector completion | `74e79c222e96d4dc03b957050cbe18866901739267755b466259858245d15b40` |
| Remaining collection driver plan | `3a3e44f20ce2904a7bafd6d5b15ae73b4a4712188b1dca5b3a0c8a32bd61698b` |
| Remaining collection independent review | `f97cb4dbd6812c886f23ea9c7d06abd0167e4f34a2ae2f43c2120295555c9764` |
| Real-teacher value pilot | `0df10b9a7288385f2685881f734bc1ffd4bb0f8a2b8b62b5f04951847961da3b` |
| Recovery coverage review | `5487e2a3d7781cae23812deb680a825f0a1528251ad2693983b6cb7dc1cde9a8` |
| Fresh recovery driver plan | `053c1365cbeee1919141007c0b4e7bf94997733f4c6e47b1c26f52cd180e36ae` |
| Real-teacher policy pilot | `b25ae0621f3e87e3d66bd6aa7b1437198f6917396531d082de4da5e49da5f922` |
| Frozen value consumer readout | `155adc4c1bfa24f9e565f3d03f12fee6569dff00e9def9206198ddc2733ab024` |
| Frozen consumer fixture readout | `89b625ac4d30eb89484a6cb1066803b058859a6279d498f8f47b44f36ecaf4e4` |

Bulk labels, executable manifests, logs and receipts remain in the host experiment storage. These identities bind the launch snapshot; future completion and training results belong in this record.
