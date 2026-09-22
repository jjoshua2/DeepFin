# Native CUDA/BF16 backend implementation

## Preregistration — September 22, 2026

PR3b implementation, based on #831 `9b0f57492f7c86fdb935da7523748fc088e056b8`.
No merge, production deployment, live GPU use or private trained-model upload.
There is no CUDA device in the authoring container. GPU execution must remain
explicitly **unqualified**, even if compilation and all CPU checks pass.

Hypothesis: the existing v3 CUDA/BF16 exporter can feed a Bend-owned synchronous
batch boundary using one pinned staging slot and an explicit stream/event lifetime,
without changing the CPU path or treating padding/delivery as useful search work.
No speed or strength hypothesis is tested here.

Scope: explicit CUDA-only binding/build gate, BF16 staging and strict output
contracts, CUDA execution/physical completion, a separate actual-GPU numerical
gate and failure-safe reports. Not scheduler, multi-worker search, CUDA Graphs,
FP32-versus-BF16 playing-strength qualification, larger arena or default adoption.

Controls and acceptance:
- Preserve all old CPU/singleton binding controls and numerical tolerances.
- Run the shared BF16 casting/clearing helper on CPU, including round-to-even,
  signed zero, ignored source tail and overflow/nonfinite rejection. Verify both
  output descriptors before publication; device type is not enough without index.
- Compile/link the real CUDA branch against the locked CUDA Torch build and a
  matching CUDA 13.0 toolkit. Explicitly test a missing GPU as a failed startup,
  not a skipped/passed CUDA-model run. Record compilation separately from execution.
- Reuse the saved untrained CPU singleton package; rerun its batch probe and the
  unchanged singleton selected-leaf verifier against the changed CPU bridge.
- Actual GPU qualification must have a real device, matching trusted package and
  exact checkpoint, independent BF16 singleton comparisons, full/partial/repeated
  rows, quiet and traced paths, exact BF16/padding bits, output-tail and audit
  controls, and **explicit predeclared** atol/rtol. No automatic CPU substitution.
  CPU-F32 fidelity and actual chess-input/strength evidence are separate gates.

Budget: bounded hosted CPU and CUDA-build-only lanes, one compiler job and two
Torch threads each. No model export, training or game arena. Reuse the saved
singleton fixture and existing generated Bend application source. New small Bend
probe generation is permitted; do not regenerate the unchanged chess engine.
Reports/patches/hashes only in the new qualification artifacts, not private traces
or models. Temporary staging workflows stay out of the feature PR.

Recovery: retain failed reports; fix the relevant layer; do not relax numerical
or parser gates or upgrade the compiler to hide failures. Publish a reviewable
opt-in implementation with explicit GPU limitations, not a claim that original
PR3's trained-5090 acceptance criterion is complete. Self-review is not independent
review; no independent reviewer is available in this execution environment.

## Reference contracts

The CUDA stream handle and explicit device-index constructor follow the upstream
[PyTorch 2.14 loader interface](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/inductor/aoti_package/model_package_loader.h).
Pinned-memory lifetime and synchronization follow the
[PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html).
The runtime makes one synchronous slot, not concurrent AOTI runners.

## Readout

Pending hosted qualification. Source implementation and local CPU tests alone
cannot certify native CUDA model execution. User-facing gates and remaining
adoption requirements are in `native/bend_engine/batch_backend/CUDA.md`.
