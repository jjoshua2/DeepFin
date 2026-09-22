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

## Completed implementation checks — September 22, 2026

The opt-in runtime implementation passed the CPU-regression and CUDA-build-only
lanes in [run 35778534888](https://github.com/jjoshua2/DeepFin/actions/runs/35778534888).
**No CUDA model was executed. GPU numerical correctness, trained-model fidelity,
performance and RTX 5090 readiness remain unqualified.** This closes the code/build
slice of PR3b, not the original PR3 trained-GPU acceptance criterion.

### Source and environment

The executable-source feature commit is `16bef1eccf88614698a6d1d94f4beb7d833774ce`.
A subsequent readout commit changes only this Markdown record. The 16-file authored
code patch has SHA256
`888925f36bd6c39827bcb7861a103747c1f482ff38d4c238c95fb60e4a7bedeb`.
Both lanes reconstructed that exact patch on #831 at `9b0f574`, and their source
manifests match the locally authored files. During validation the parent advanced
to `33718f1` by incorporating other stack work, including table/proof changes and
a corpus-test correction. This feature does not overwrite those changes. The
recorded native qualification is on the pinned original base, not a claim that
the later parent or full merged head was requalified here. No compiler, Bend
application/search, native array effect, exporter, checkpoint, production config
or ordinary CI change is in the authored feature.

Compiler stays `jjoshua2/bend@aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`,
Bend 2.0.21 + U64, with the unchanged 84-file fingerprint. Bun 1.4.2,
Clang 18.1.3, locked Python 3.13.15, uv 0.12.10, Torch 2.14.0+cpu or
2.14.0+cu130 in separate environments. CUDA build uses the NVIDIA
`13.0.2-devel-ubuntu24.04` container and reports NVCC 13.0.88. Builds use one
compiler job; Torch is capped at two threads.

### Checks actually completed

- Whole-repository Ruff, Basedpyright and Vulture, plus explicit native-verifier
  lint, pass with zero type errors/warnings.
- 153 Python tests pass, including 40 new CUDA-target/parser/failure controls;
  zero skips. The existing CPU batch/accounting/benchmark/broker tests are included.
- 112 Bun binding/compiler tests pass, including 47 new CUDA binding controls.
  CPU gates still reject CUDA manifests; device indices and package/encoding
  identity are not inferred or overwritten.
- Shared BF16 staging helper executes under real CPU ATen: 41 valid cases and
  17 rejected cases. Checks cover all five supported batches, both input widths,
  full/partial reuse, exact round-to-even tie bits, signed zero, ignored poisoned
  source tails, nonfinite/overflow and malformed storage/row/device contracts.
- Existing ATen output-packing helper passes 5 valid noncontiguous cases and
  18 malformed-contract rejections. Neither set of CPU tests certifies GPU work.
- The CUDA-enabled target compiles and links **cuda_execution.cpp itself**, not a
  CPU substitute, against Torch 2.14.0+cu130. The executable has native CUDA/ATen
  dependencies and no libpython link. A CPU LibTorch build explicitly refuses
  `BEND_CUDA_MODEL=ON`.

The saved **untrained 5,043,005-parameter** 175-plane CPU-F32 batch-one fixture
was reused without export. The native batch-one probe passed 5 real forward rows,
four invalid-buffer controls, stable buffer addresses and one bridge input-tensor
allocation. Maximum absolute logit error was `4.76837158203125e-7`, within the
unchanged `2e-6` absolute / `2e-5` relative tolerances.

The relinked singleton engine passed 18 searches, 50 traced forwards/replies,
1,141 legal-prior comparisons, six automatic-draw replies and four zero-forward
terminal roots. Same-board/different-history inputs remain distinct; all three
startup-failure controls reject. Maximum logit error `7.152557373046875e-7` and
probability error `2.9802322387695312e-8` stay within the original tolerances.
This run did not execute a batch-four CPU model or a CUDA model.

CPU batch probe SHA256:
`ba587fb500539ee709653b555f574b9e9a362a7ddf4a8e7ce46d1af839498dd4`.
CPU singleton SHA256:
`85f801508504d23b0c1928ffbadaf73ed55ed923db8458fecfd46a726aefcd75`.
Saved model package SHA256:
`9654a1e334fb5f875d164b82068846bfeab3df2ef2a994737e44789c2623ee6e`.

### No-device failure evidence, not GPU qualification

The CUDA build-only lane has no GPU. It uses an explicitly labeled non-model
text canary with a matching digest and CUDA-target metadata only to instantiate
and compile the binding. This is **not** a relabeled CPU model or a CUDA export.
The native executable checks the requested CUDA device before trying to load
that canary and exits 2 with `requested CUDA device is unavailable; no CPU
fallback`. It produces no trace or output rows. The separate numerical verifier
also exits 2 and records `status=failed` and `cuda_model_qualified=false`.
The lane passes because both expected failures were observed, not because model
execution passed. The canary's zero atol/rtol values never evaluate any tensor
and do not establish tolerances for real models.

CUDA-enabled executable SHA256:
`29213e4904ebec5cfb01a8002499e9ae78ee08780d424d32af77fd63a1f0fa62`.
The no-device report, build commands, linker dependencies and source identities
are retained. No binary, model or raw neural trace is uploaded in these artifacts.

### Failure recovery and review

First [run 35777915315](https://github.com/jjoshua2/DeepFin/actions/runs/35777915315)
passed the locked static/Python/Bun gates but failed two temporary-workflow setup
steps: an artifact download path was misspelled, and Git rejected container
checkout ownership. Corrected that path and trusted only the exact isolated
checkout path. The source patch, numerical controls and compiler did not change.
No engine bug or tolerance was hidden by the workflow repair. The complete rerun
is separate from the initially passing static checks.

Self-review only, not independent review or formal proof. Local CPU staging,
parser and binding checks passed; the locked hosted checks supply version-matched
build evidence. Source hashes were reconciled across both lanes and publication.
The full singleton uses PR2's hash-verified unchanged generated C, recompiled and
linked with the final bridge; no expensive full Bend regeneration or model export.
No new perft, full UBSan stack or no-Python chroot run is claimed. Dependency
inspection is not process isolation.

### Artifacts and remaining gate

Run 35778534888 retains the following compact artifacts for 30 days:

| Artifact | ID | ZIP SHA256 |
| --- | --- | --- |
| deepfin-pr3b-cpu-regression | 10717641764 | `72bbff27351daff89e2b62ffa1a7891b6cbba57b1cc8b69b8d397ca6fc8132f0` |
| deepfin-pr3b-cuda-build-only | 10717661357 | `dde08a32776db08b24a01e149b0a47c1f7c5979838a74e063b6b7b2d41527820` |
| deepfin-pr3b-source-publication | 10717467074 | `4b0ea4dcf52b110b9d73fe237f6f313eb2afa0f5b39302bfa2ffbfc2b3d1b746` |

Reports explicitly distinguish CPU model execution, host data-contract tests,
CUDA compile/link evidence and intentionally failed no-GPU model qualification.
No model packages, binaries or raw traces are in these artifacts or the feature PR.
Ordinary PR CI is separate from the dedicated implementation checks above.

User-facing commands and limits are in
`native/bend_engine/batch_backend/CUDA.md`. The separate numerical gate requires
an actual GPU, exact trusted CUDA/BF16 package and checkpoint, and preregistered
finite atol/rtol. It compares full/partial/repeated real rows against independent
BF16 singletons, verifies actual BF16 device-input/padding bits, and exercises both
quiet/traced execution and invalid requests. Even a future pass is package/device/
test-input specific: trained chess inputs, FP32-versus-BF16 policy/WDL fidelity,
memory and latency remain separate production-adoption work.

Normal UCI remains CPU F32 batch one. This PR adds no search scheduler, async
submit/poll, multi-slot overlap, CUDA Graph capture, expanded arena, production
Gumbel parity, speed or Elo claim. Nothing was merged, deployed or run on a live
training GPU during this work.
