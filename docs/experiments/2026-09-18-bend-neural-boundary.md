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

## Hosted readout

Pending at publication preparation. Local smoke artifacts are disposable and
not committed. Hosted outputs will record the exact Torch version, package hash,
compiler revision, session observations and negative-test counts.

## Remaining decision

A passing smoke qualifies the boundary design only. Next use a trusted real
checkpoint and its explicit input/model contract, then the existing CUDA parity
bridge, and then bounded batching/backpressure before measuring end-to-end speed.
No change to DeepFin production Gumbel semantics is implied by this diagnostic
PUCT tree. No root advance, full draws, async stop, GPU, universal verification
or independent reviewer is claimed.

Reproduction and all protocol/manifest details:
[`native/bend_engine/neural_probe/README.md`](../../native/bend_engine/neural_probe/README.md).
