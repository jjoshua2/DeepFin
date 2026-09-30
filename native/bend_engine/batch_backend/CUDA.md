# Explicit CUDA/BF16 backend (PR3b implementation)

This opt-in execution path is **not yet numerically qualified on a CUDA device**.
Compilation, CPU casting tests and a no-device rejection test do not establish
GPU correctness, performance, trained-checkpoint fidelity or RTX 5090 readiness.
Use a disposable build/environment. Do not replace a running worker or consume a
live training GPU without the separate resource/adoption checks in project guidance.

## Ownership and execution

The existing v3 exporter already emits CUDA/BF16 checkpoint packages. This path
uses that contract; it never relabels a CPU package, guesses an accelerator,
converts an unsupported target to CPU, or changes the model's declared encoding.

`--cuda-bf16` selects a separate binding gate and enables CUDA compilation. Both
are required. Normal UCI binding remains CPU F32 batch one, and the ordinary batch
build remains CPU F32. A CUDA package cannot enter the singleton open API. The
bound logical device index (0..127) is passed to both CUDA guards and AOTI's loader;
CUDA_VISIBLE_DEVICES therefore needs to have the same intended mapping at export,
build qualification and execution. No runtime override silently changes the index.
The release Torch version and CUDA major/minor are checked. A nightly build is
not supported by this binding. These checks are compatibility prerequisites, not
numerical evidence or guarantees for different hardware.

One synchronous slot owns a non-default CUDA stream, a persistent BF16 input
on the device, a persistent F32 packed device output, pinned BF16 host input,
pinned F32 host output, and one completion event. The trace-only device-input
snapshot adds another pinned buffer. AOTI's temporary/output allocations are not
covered by the count of one persistent input allocation.

Bend provides real F32 rows while retaining ownership of its arrays. Host staging
uses round-to-nearest-even BF16 conversion and clears every physical padding row
to exact +0, including after a full batch. Nonfinite inputs or finite inputs that
overflow BF16 are rejected. The backend passes the actual stream handle to AOTI;
a current-stream guard alone is not treated as an execution contract.

Both returned tensor descriptors must have the exact bound CUDA device/index,
F32 dtype and `[batch,1858]` / `[batch,3]` shapes before packing. Only real rows are
copied back. Event synchronization completes H2D, model execution, output packing,
D2H and optional device-input snapshot before caller output bytes are published.
Temporary output tensors remain alive through completion. Exceptional execution
poisons the slot and attempts a stream drain; the foreign effect exits on failure,
not retry/reuse. Catastrophic driver failure is not recoverable in this process.

This is **not asynchronous search**: the call blocks until its event completes.
There is no multi-slot scheduler, CUDA Graph capture, search-leaf gathering, larger
arena, useful-EPS inference from delivery, or change to Gumbel/PUCT. PR4 must define
logical cancellation separately from physical completion before exposing overlap.

## Build and qualify on the intended GPU

Use the project's isolated locked `cu130` environment and a matching CUDA toolkit
and LibTorch. Keep CPU and CUDA builds in separate new output directories.
Obtain an exact trusted CUDA/BF16 v3 package through the existing
`neural_probe.checkpoint.export_checkpoint` with explicit device/index/batch. The
source checkpoint and exported sidecar must retain complete encoding metadata.

```sh
bash native/bend_engine/batch_backend/build_probe.sh \
  build/native_cuda_b4 /path/to/cuda_b4.pt2 /path/to/libtorch/share/cmake \
  /path/to/verified-bend-source --cuda-bf16

# ATOL/RTOL must be preregistered for this model BEFORE executing the gate.
# This command does not infer a tolerance or substitute a CPU run.
python -m native.bend_engine.batch_backend.verify_cuda \
  --binary build/native_cuda_b4/build/deepfin-bend-batch-probe \
  --package /path/to/cuda_b4.pt2 --checkpoint /path/to/checkpoint.pt \
  --atol "$ATOL" --rtol "$RTOL" --report /tmp/new-cuda-b4-report.json
```

The explicit GPU gate checks full, partial and repeated rows against independent
**eager BF16 singleton** outputs from the exact checkpoint. It also runs without
trace instrumentation, checks native buffer/stream audits, finite logical outputs,
untouched tail, input quantization/padding and invalid row/buffer rejection. A new
report is required; failures record `cuda_model_qualified=false` and exit 2.
A model-specific pass does not certify other packages, buckets, GPUs, all inputs,
or the accuracy loss of BF16 relative to FP32. Production adoption additionally
needs trained-checkpoint/actual-chess-input and separate FP32 policy/WDL fidelity
checks, memory/latency measurements and the original fixed-work/fixed-wall protocol.
No numerical tolerances are relaxed after seeing a failure.

## Trace and audit

CPU traces are unchanged. CUDA batch traces use version 3: eight little-endian U32s
`[0x44464333, sequence, batch, real_rows, channels, 1861, device_index, 16]`, followed
by the actual device-input snapshot as `batch*channels*64` BF16 words and only
`real_rows*1861` F32 outputs. The 16 denotes BF16 input storage bits, not IEEE half.
The dedicated parser rejects wrong versions, shapes, device indices or truncation.

The existing explicit trace option still requires a new owner-only file and may
contain private input/model information; do not upload raw traces by default.
`DEEPFIN_BEND_BUFFER_AUDIT=1` adds a CUDA audit alongside the original buffer audit.
Timing or allocation claims about AOTI internals cannot be inferred from that audit.
