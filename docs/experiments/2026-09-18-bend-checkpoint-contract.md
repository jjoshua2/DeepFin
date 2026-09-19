# Bend: explicit checkpoint and device contract

## Decision

Follow #779 on `4ab0836c69f7d60ae41c84fa069527a8c731f047`. Stop adding synthetic
scheduler features; make the existing search/batching path accept a specified,
self-describing checkpoint without random fallback or another compiler change.
The user need not rewrite C encoding or C++ inference in Bend.

No trained .pt/.pt2 artifact was present in the accessible repository snapshot;
the repository has no published releases. The local runtime is Torch 2.10.0+cpu
with no CUDA device. Therefore a trained-model/GPU execution claim is unavailable,
not replaced by presenting smoke weights as a trained checkpoint.

## Implementation / scope

Strict embedded architecture + state-dict loading, explicit history/policy
identity, frozen tuple-output package, runtime/dtype/CUDA-device identity checks,
and full-package bit parity precede interpreting the search result. The native
worker supports CPU F32 and explicitly selected CUDA F32/BF16. CPU mode keeps
prior package formats compatible. The existing bounded coordinator gains an
optional full-batch comparator; its ordinary smoke path is unchanged.

The checkpoint-specific command compares native and Python package outputs exactly,
then compares real rows to the original eager singleton model. Default maximum
legal-policy/WDL TV limits are 0.01 (engineering bounds, not strength calibration);
F32 also retains atol 2e-6 / rtol 2e-5 logits. Full control/batched/cancelled search
state agreement remains required. No benchmark or production Gumbel claim.

No live checkout, configuration, checkpoint, deployed package, compiler, chess core,
perft depth or routine native-test schedule is changed. New output directories
must not exist. A failure never publishes a qualification report.

## Local exploration

A temporary untrained reduced transformer has 32-wide/two-layer/four-head trunk,
split QKV, per-layer Smolgen and relation bases, ARC adapter and DeepNorm. It is
saved with full checkpoint metadata, then strictly reloaded. No training performed.
Native + UBSan Bend modes passed 34 search epochs (30 normal and four cancelled),
250 real inference rows. Native C++ outputs were bit-identical to Python loading
the same package. Maximum absolute eager logit error was 4.470348358154297e-7;
max legal policy TV 5.21540641784668e-8; max WDL TV 1.4901161193847656e-8.
All cancellation/recovery/control search comparisons passed.

An initial local orchestration timeout interrupted export; it produced no result
report and is not counted as successful validation. A subsequently bounded run
completed. No numerical threshold was relaxed. Focused Ruff and 113 small tests
passed after scalar guard/import cleanup; no model forward in ordinary pytest.

## Hosted confirmation plan (before hosted execution)

One bounded 15-minute CPU confirmation on the locked project environment. Run
Ruff/Basedpyright and focused cheap tests. Retain original four-mode session,
batch-one neural, and batch-four neural checks after changing the native worker.
Run the new reduced-transformer checkpoint smoke in all four Bend modes. Every
native/Python same-package output must match exactly, eager differences must pass
the stated gate, and complete search states must match. No timing acceptance gate.
Do not download trained weights, rent GPU capacity, or launch training.

Recovery: discard the isolated branch; preserve all prior PRs. CUDA remains an
explicit unexecuted gate. Self-review only unless an independent review is recorded.

## Hosted readout

Pending. Commands, artifact semantics and remaining limits are in
[`neural_probe/CHECKPOINTS.md`](../../native/bend_engine/neural_probe/CHECKPOINTS.md).
