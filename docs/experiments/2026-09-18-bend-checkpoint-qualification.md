# Bend explicit checkpoint qualification

## Decision and hypothesis

Follow #779 (4ab0836c69f7d60ae41c84fa069527a8c731f047) with checkpoint and
transformer compatibility, not another perft optimization. Compiler remains
57bc84edc0df32780e2c4dde44e3a2ee1a500cc9. Bend chess/search, production model,
encoder, configuration and perft depths remain unchanged.

Hypothesis: strict self-describing checkpoints can drive the existing native
batched evaluator and Bend tree without more language/compiler features. Keep
shared architecture parsing and policy-output selection. Reject partial weight
loads, encoding guesses, stale file identity and ambiguous SWA selection.

## Confirmation plan (before hosted execution)

An explicitly untrained saved transformer fixture: 2 layers, width 32, 4 heads,
Smolgen, relation basis and dynamic relations, root-legacy-meta/v2_threats,
repetition fix enabled. Actual checkpoint deserialization and strict weight load;
not TinyNet and not a representative trained production checkpoint.

All four Bend modes; each has one-real-row control, normal batched searches and
queued/in-flight cancellation/recovery across five roots. Existing group contracts
check native logits against singleton eager execution at unchanged CPU F32 atol
2e-6 / rtol 2e-5, per-reply statistics and complete final trees. Retain all previous
session, single-row and batching regressions after changing the native worker.

Bounded 15-minute hosted CPU job, one compiler worker and two inference threads.
No timing acceptance gate, GPU allocation, training, checkpoint download or live
restart. No additional search/inference/compilation in ordinary pytest. Recovery:
discard this isolated branch; earlier PRs and production are untouched.

## CUDA boundary

V3 explicitly records CPU/F32 or CUDA/BF16 and a device index. CPU builds compile
the generic ATen device-selection path; this is NOT execution on CUDA. The runner
requires a real requested GPU and explicit predeclared CUDA tolerances. It fails
instead of silently using CPU. Native output checks require matching device,
float32 tuple policy/WDL and fixed shapes. Existing CUDA-package parity is separate.

No trained checkpoint or GPU was available in the local working environment.
The ability to accept a user-supplied file does not establish its provenance or
compatibility: actual execution and a successful report are required.

## Local preliminary observation

CPU Torch 2.10.0 and Clang 17: the native-target transformer checkpoint round trip
passed, including normal/control/cancellation groups and all five roots. Maximum
native/eager singleton logit difference 5.960464477539062e-7. Full-mode and inherited
regressions are being run separately. All 108 cheap contracts passed (28 new, 80
inherited); tests use tiny tensor dictionaries, not full-model export or search.

## Hosted readout

Pending. Record exact validated revision, toolchain/package/checkpoint identities,
per-mode outcomes, errors and remaining limitations after confirmation. No
throughput, strength, universal proof or independent review claim.
