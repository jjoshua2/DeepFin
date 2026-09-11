# Weighted Ceres bootstrap preparation

Status: the first new Ceres collection chunk passed independent validation. The remaining original-corpus labels are now collecting under a bounded, reviewed plan. The offline policy mixer is implemented and reviewed; no Ceres-mixture training or playing-strength result is claimed.

## Selected question

The [tactical SF attenuation experiment](2026-09-10-bt4-sf-tactical-training.md) did not demonstrate a gain. Retain B100 (BT4 policy at teacher temperature 0.5, original SF values) and explore a distinct teacher mixture.

The first candidate averages independently normalized BT4 and Ceres policies with equal weights and teacher temperature 0.5 each, holding SF values fixed. Use raw BT4 probabilities and apply sharpening once. Complementary teacher mistakes motivate this test; disagreement alone is not evidence that averaging improves play.

A later separate value candidate is 50% SF / 25% BT4 / 25% Ceres, with policy fixed. Its raw-head conversion must be registered explicitly. This is a direction, not an implemented value producer or unconditional experiment queue.

## Collection allocation

The new validation chunk covered 16 distinct full shards: 131,072 positions, 4,096 fixed32 calls and zero padding. Independent review decoded and checked all 144 saved arrays, source metadata bindings, legal rosters, provider evidence and loaded libraries. Total driver time was 703.61 seconds, including 245.46 seconds of admission/possible GPU waiting; post-lease completion took 453.29 seconds and inference calls totaled 331.81 seconds. The saved bank occupied 21.34 MB. These are measured collection costs, not strength evidence.

The next allocation covers shards 32–2307 in 143 disjoint chunks, totaling 18,644,992 real rows and 582,656 calls. Reuse accepted shards 0–31 and final partial shard 2308 once. Successful completion would yield 18,910,484 distinct rows; a launched plan is not completed coverage.

Each chunk has a 1,800-second cap and at most 128 MiB of bank/state output. The driver has a 30-hour cap, 30-second inter-chunk pauses and a 150 GiB disk reserve. The collector owns the shared GPU lease, allowing existing BT4 labeling to continue between operations. Stop on failure; retain completed chunks without automatic retries. At observed costs, the remaining allocation projects to about 19.18 hours excluding variable waits or 29.11 hours using observed whole-chunk time, including pauses. It may reach the cap before finishing.

The backend remains the qualified fixed32 ORT1.29/CUDA13 approximation for C3-768-30-pre8-I8. It retains raw legal policy and both raw WDL heads. This does not establish native Ceres numerical parity or engine strength.

## Producer and training compatibility

The policy-only producer uses ordinary copied shards, preserving all 16 nonpolicy arrays. It checks complete source/teacher coverage, actual Ceres input/history and legal alignment, and atomic output completion. Legacy BT4 root-position keys do not newly prove historical input frames: the producer pins the original collection and records the inherited limitation. Stronger full-input hashes are checked wherever available.

Independent review identified that provenance distinction, which was fixed and rereviewed. The producer/admission tests exercise real shard rewriting, loader/collator/loss propagation and meaningful rejection cases. A separate 32-row fixture also passed the frozen Python3.10 / NumPy1.26.2 / Torch2.11 consumer: policy loss and gradients changed while SF value loss and all nonpolicy arrays remained identical. This qualifies a tiny fixture's consumption, not the full corpus or training epoch.

Before training, qualify the completed full-corpus manifest and realized targets, freeze producing-code identities, and verify the same historical initialization, row schedule, update budget and runtime as B100. Register the deciding match and its resource/stopping rule before launching. A new trainer or sampling regime would require a fresh matched control.

## Compact evidence identities

| Evidence | SHA256 |
| --- | --- |
| First new chunk independent readout | `b4317054250af4b410bb639c2233de9d75d0f899e4c424f1913db437ef0bb711` |
| First new collector completion | `74e79c222e96d4dc03b957050cbe18866901739267755b466259858245d15b40` |
| Remaining collection driver plan | `3a3e44f20ce2904a7bafd6d5b15ae73b4a4712188b1dca5b3a0c8a32bd61698b` |
| Remaining collection independent review | `f97cb4dbd6812c886f23ea9c7d06abd0167e4f34a2ae2f43c2120295555c9764` |
| Frozen consumer fixture readout | `89b625ac4d30eb89484a6cb1066803b058859a6279d498f8f47b44f36ecaf4e4` |

Bulk labels, executable manifests, logs and receipts remain in the host experiment storage. These identities bind the launch snapshot; future completion and training results belong in this record.
