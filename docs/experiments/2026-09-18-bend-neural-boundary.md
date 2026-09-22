# Bend search / native neural boundary

## Decision and scope

Follow #777 rather than tune more perft. Baseline is
`c8bf8679fe68523effc7d971def5cd27ad35d4c9`; compiler remains
`57bc84edc0df32780e2c4dde44e3a2ee1a500cc9` with its source fingerprint.
No production code/configuration, training run, live checkout or perft-depth change.

Hypothesis: persistent Bend tree state can compose with DeepFin's real history
encoding, legal policy mapping and a native model evaluator without another
compiler modification. Use existing CBoard encoding and LibTorch/AOTI deliberately;
an all-Bend rewrite is not the success criterion.

The neural smoke is actual project TinyNet code with seeded **untrained** weights,
not a learned checkpoint. CPU float32 batch one, 175 channels, compact 1858 policy
logits plus WDL logits exported as a fixed tuple. Python coordinates encoding and
verification; the worker itself is C++ and the search itself is Bend.

## Confirmation contract, recorded before hosted execution

All four Bend modes must pass: generic C, portable U64 helpers, native CPU target,
and UBSan. Five root fixtures, two epochs each, 16 simulations each; every path,
leaf board, ordered legal policy mapping, final node state and best move checked.
Native logits compare to eager same-version weights at atol 2e-6 / rtol 2e-5;
legal-gather/WDL softmax compare independently with production policy expansion
and Torch softmax. Keep previous session cancellation, malformed-reply, capacity,
terminal and reset checks. No timing-based acceptance gate.

Input history's first 112 planes compare exactly to the Python encoder. Other
classical features tolerate only atol 6e-8 / rtol 2e-7 due float intermediate
rounding. All four root-oriented history/extra-feature combinations are checked,
including all legal promotions, EP, castling and repeated pre-root history.
Require identical boards reached with different paths to encode differently.
Six native malformed-input cases must fail with code 2 and no model response.

Budget: one focused hosted CPU confirmation (15-minute job bound), isolated
package/cache, one C++ compilation process and two inference threads; no GPU or
training. Additional attempts only to repair observed failures, not select a fast
runner. Recovery: discard the isolated change, preserve #777 and all earlier PRs.

## Local observations

Torch 2.10.0+cpu / Clang 17: actual native TinyNet execution and all four Bend
modes passed. 64 independent input-encoding checks; 40 neural search epochs with
640 native evaluator calls; all repeats deterministic. Maximum absolute native
logit difference from eager was 2.2351741790771484e-8. Six malformed native inputs
were rejected. These checks have no speed or trained-model significance.

Exploration found the older `legacy` history convention disagreed on repetition
planes after repeated knight histories. It is explicitly unsupported here; no
production encoder was changed to make the comparison pass. Separately, a few
classical graded features differed by 1.49e-8 from float intermediates, which
motivates the narrow feature-only tolerance rather than falsely claiming tensor
bit equality. CBoard's 8-bit rule50 storage is guarded against overflow.

## Hosted readout: PASS

[Confirmation run 35405756949](https://github.com/jjoshua2/DeepFin/actions/runs/35405756949),
job 105795067854, passed on the locked **Torch 2.14.0+cpu** environment. The exact
validated executable tree was published as commit
`28131ae90fa8c043f4a451c7b43cf07f76603c5c`, with parent #777's head. This later
readout changes only documentation. Core published blob hashes were checked
against the locally tested files.

- Four Bend build modes passed: generic, portable helpers, native target, UBSan.
- 64 independent encoding comparisons plus the equal-board/different-history check.
- 40 neural epochs / **640 real native AOTI calls**. Each 16-request epoch had
  16 distinct input tensors and 16 distinct output pairs; resets reproduced results.
- Maximum absolute native/eager logit difference: **2.2351741790771484e-8**.
- Every final search node/board/statistic and best move matched the diagnostic reference.
- All six malformed native inputs rejected with code 2 and no model response.
- Existing session regression passed **38 sessions per build**, including malformed
  replies, cooperative cancellation, capacity, reset, terminal and EOF behavior;
  python-chess checked all 207 distinct oracle positions.
- Focused Ruff, Basedpyright and **53 cheap tests** passed (38 new boundary cases
  plus 15 existing session tests). No suppression or depth increase.

Identical across build modes, two epochs per fixture:

| Root | Completed per epoch | Nodes | Best private key | Native requests |
| --- | ---: | ---: | ---: | ---: |
| Start | 16 | 338 | 1153 | 16 |
| Kiwipete | 16 | 733 | 204 | 16 |
| En passant | 16 | 102 | 2852 | 16 |
| Black promotion | 16 | 122 | 3324 | 16 |
| Repeated pre-root history | 16 | 338 | 1153 | 16 |

These moves reflect untrained weights, not good chess. Keys are private Bend
move words, not neural policy indices.

Artifact: `bend-neural-confirmation`, ID **10572258848** (30-day retention).
ZIP digest: `41ff02459e3fc0d9b5befd63debc50cf31d24d79a9bbc19af32461c5d2b63f91`.
Neural report SHA-256: `0785bf00eee638f926c4471c693c4d28b7553362ff3b4c501e4cd604765f1859`.
Session report SHA-256: `eff3dbfbc630b41f38539a1d5a3467d2afbbcea467ffd5d9f85015c3696e3435`.
Executed package SHA-256: `0cdae40230af2c8c0e70620227a5d15f210877757751c2bfe5af1c14f3c410b9`.
Package format is `deepfin-tuple-policy-wdl-cpu-f32-v1`; seed 20260918;
root-legacy-meta/v2_threats/repetition-fix-on. Smoke packages are temporary,
not trained artifacts and not committed. Re-exporting recreates the semantic
check; the package's archive hash is an identity, not a reproducible-build promise.

The first hosted attempt stopped at two harness lint findings before inference.
The rerun fixed those, clarified binary-stream annotations, made CPU/F32 explicit,
and strengthened the nonconstant-output assertion; it did not relax the numerical,
source-integrity or chess checks. No timing comparison was made between runners.

## Remaining decision

This pass qualifies the boundary design only. Next use a trusted real checkpoint
and its explicit input/model contract, then the existing CUDA parity bridge, and
then bounded batching/backpressure before measuring end-to-end speed.
No change to DeepFin production Gumbel semantics is implied by this diagnostic
PUCT tree. No root advance, full draws, async stop, GPU, universal verification
or independent reviewer is claimed. Broader repository CI is separate from this
focused confirmation; no merge or deployment occurred.

Reproduction and all protocol/manifest details:
[`native/bend_engine/neural_probe/README.md`](../../native/bend_engine/neural_probe/README.md).
