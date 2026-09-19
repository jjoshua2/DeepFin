# History-aware Bend / native neural boundary

This is an opt-in CPU **composition qualification**, stacked on the persistent
search-session probe. It runs an actual project `TinyNet`, with deterministic
**untrained smoke weights**, in a separate native C++ AOTInductor process. It is
not a trained chess-strength result, a production search replacement, or CUDA
qualification.

```
Bend chess + persistent PUCT tree
  -> leaf board + complete root-to-leaf move path + ordered legal keys
  -> external Python adapter: root history + path, existing CBoard encoder,
     shared DeepFin dense/compact policy mappings
  -> persistent native C++ CPU AOTI worker: one loaded .pt2, real forward passes
  -> legal-logit gather + softmax, WDL-logit softmax
  -> Bend reply validation, expansion, backup and best move
```

Bend continues to generate/apply its own moves and own its search tree. The
adapter intentionally reuses CBoard for **encoding**, not candidate search. Python
is an external coordinator and test oracle, not embedded in either executable.
The native worker links LibTorch, not libpython. Encoding in pure Bend and a
single-process all-native evaluator are not prerequisites or claims of this gate.

## Run

Use an isolated development checkout and the project's CPU environment. Requires
Bun 1.4.2, Clang/Clang++, CMake, the project native encoding extension, python-chess,
NumPy and PyTorch with CPU AOTInductor support. Use trusted locally built packages:
AOTI packages contain native executable code.

```sh
uv sync --locked --extra dev --extra cpu
bash native/bend_engine/bitboard_probe/install_toolchain.sh
uv run python -m native.bend_engine.neural_probe.run_probe \
  --report artifacts/bend-neural-boundary.json
```

The command builds one temporary fixed batch-one, float32, 175-plane CPU package
from the actual `ModelConfig(kind='tiny')` code with seed `20260918`. It compiles
the worker and four Bend modes (generic/portable/native/UBSan), checks outputs
against eager PyTorch, and removes its temporary packages/binaries afterward.
`--modes native` is a smaller explicit run. No network checkpoint is downloaded.

The sidecar format is `deepfin-tuple-policy-wdl-cpu-f32-v1`. It pins the package's
SHA-256, exact Torch version, input encoding, batch/shape and compact tuple-output
contract. The worker accepts **(policy logits [1,1858], WDL logits [1,3])**, not
arbitrary dictionary outputs or existing BF16 CUDA packages. `--smoke-package`
can reuse a package from `package.py --out PATH`; it still requires this command's
default seed/encoding and verifies the manifest. No existing package is overwritten.
Bend's source-fingerprinted compiler pin remains unchanged.

## Why the path is required

The session wire now includes `path LENGTH KEY...` after the leaf board, before
its action rows. The root has `path 0`. Bend follows its own parent links with
bounded fuel, requires decreasing parent IDs, and retains the tree through IO.
The receiver rejects missing/malformed paths and old binaries; this private
probe protocol is not UCI or a backwards-compatible production API.

The adapter starts from `root.copy(stack=True)`, preserving **pre-root history**,
then pushes the exact path. A leaf-only FEN cannot supply past repetitions or
history planes. A test constructs identical final boards by two move orders and
requires their history tensors to differ. There is deliberately no board-only
encoding cache. The model's process-global repetition regime is applied before
CBoard creation; unsafe flips are not waived. An existing uint8 CBoard rule50
clock limitation is guarded before a non-zeroing push could overflow 255.

Packed Bend keys are decoded and validated as legal orthodox moves. Shared
`move_to_index` and `FULL_TO_COMPACT_POLICY` map them to neural indices, preserving
**request order**. Legal sets must be complete, unique, and match CBoard. Softmax
happens **after gathering legal logits**; WDL also requires softmax and remains
in node-side-to-move W/D/L order. Finite tensor/shape checks precede a Bend reply.

Supported explicit encodings are `lc0_root` and `lc0_root_legacy_meta`, with `v1`
(146 planes) or `v2_threats` (175). Boundary tests cover all four combinations;
the actual neural-session smoke uses root-legacy-meta/v2 with repetition fix on.
The `legacy` history mode is rejected, not silently substituted: exploratory
repeated-knight-history checks found C/Python disagreement in that convention.
This change does not fix or change the existing production encoders. The first
112 history planes are checked exactly; remaining classical graded features allow
only `atol=6e-8, rtol=2e-7` for C float versus Python intermediate rounding.

## Qualification and failure handling

For each Bend mode, five roots (start, Kiwipete, EP, black promotion, repeated
pre-root history) each run two search epochs, 16 simulations per epoch. Every
request's path, board and legal set is checked. Native policy/WDL logits match
same-version eager TinyNet at `atol=2e-6, rtol=2e-5`; independent production policy
expansion/Torch softmax checks probability conversion. Native-derived replies
feed both Bend and the existing independent diagnostic PUCT reference. Every
final node, statistic and best move must agree. Reset must reproduce the result.
This remains diagnostic PUCT, not production Gumbel or C MCTSTree parity.

The native worker loads once, uses two CPU inference threads, and identifies each
request with a monotone sequence. Little-endian headers and IEEE F32 payloads have
fixed shape bounds. Six negative native tests cover wrong magic/sequence/count,
truncation, NaN and infinity. The host checks response sequence, dimensions,
finite values and bounded reads/writes. A failed worker cannot be reused; the
verifier closes its own processes. This is not asynchronous CUDA cancellation or
a production retry service. The inherited Bend bad-reply/cancellation tests remain
separate and must still pass after the path-protocol addition.

## Cost and remaining gates

No ordinary test compiles a model or runs search. The path-scoped native workflow
has a 15-minute limit. Existing perft depths, legal core and performance benchmarks
are unchanged. Cheap pytest contracts check parsing/mapping/manifests/deadlines.

Still unqualified: trained transformer weights, BF16/CUDA, batch buckets and
multiple outstanding requests, production feature/model compatibility, asynchronous
stop/backpressure, root advance/subtree reuse, draw adjudication, throughput,
playing strength, and full all-native integration. Export/runtime APIs are
version-sensitive; both the package and C++ worker are built from the same Torch.
Self-review is not independent review or a universal proof.

## Bounded batching across searches

The next opt-in gate uses **five independent native Bend search processes** and
one persistent native C++ batch-four evaluator. The external Python coordinator
owns `Batcher`; Bend's existing one-pending-request contract is unchanged. This
is not multiple simultaneous leaves from the same tree and adds no virtual loss.

```sh
uv run python -m native.bend_engine.neural_probe.batch_probe \
  --report artifacts/bend-neural-batching.json
```

The test runs a one-real-row padded control, ordinary batched sessions, and
queued/in-flight cancellation followed by reset in the same Bend processes. It
compares each real output row to eager singleton inference, then checks each
search against its independent diagnostic reference. All four Bend build modes
are exercised by default (`--modes native` is a smaller explicit run). The
control still executes batch four; call counts are **not** speed measurements.

`package.py --batch 4 --out PATH` exports the new static-batch smoke contract.
Existing v1 manifests stay batch-one only. Batched v2 manifests require explicit
row independence; zero padding is unsafe for a model that mixes across rows.
The package fingerprint, Torch version and encoding checks still apply. Supported
buckets are 1/2/4/8/16, but execution qualification currently covers only 1/4.
Existing BF16/CUDA packages cannot be passed to this CPU tuple-output interface.

`Batcher` has a fixed queue bound and one batch in flight, snapshots input tensors
and legal indices, and routes by `(session, epoch, request, node)`. Register a
new epoch only after its old pending request is completed/cancelled. One owner
thread calls `register`, `submit`, `expire`, `dispatch`, `complete` or `fail`;
only `NativeEvaluator.evaluate` runs on the worker thread. Concurrent calls on
that native stream fail rather than corrupt framing. Cancelled in-flight slots
remain reserved until the old batch returns, but its results cannot remove or
complete a newer epoch's request. Malformed output cannot partially commit.

The sample coordinator flushes partial batches after 2 ms, expires requests at
30 seconds, and retains the existing native IO deadlines. Verification and
encoding are not real-time tasks, so these settings are not latency guarantees.
Cancellation can deliver a cancellation reply while native work is outstanding;
it does not interrupt the native forward. The driver has a bounded set of
sessions and a group deadline; it is not a network-facing production server.

No change to perft depth, board/search implementation or compiler pin. The existing
path-scoped native-neural workflow covers this test; ordinary pytest only tests
broker and manifest contracts. Real trained/CUDA batching and throughput remain
separate gates. Evidence is in
[`docs/experiments/2026-09-18-bend-evaluator-batching.md`](../../../docs/experiments/2026-09-18-bend-evaluator-batching.md).

## Explicit checkpoint qualification

`checkpoint_probe` removes the seeded-TinyNet restriction. It requires a trusted,
immutable `trainer.pt` (or its directory) with embedded `arch`, explicit encoding
metadata and canonical `model` weights. `--weights-key swa_model` selects SWA
explicitly; it never silently falls back to normal weights. Missing/extra/shape-
mismatched tensors, nonfinite weights, conflicting tied tensors, unknown fields
and unsupported architecture versions fail before export. No nearby params.json,
architecture override, tolerant migration, or random replacement layers are used.

```sh
# A COPIED self-describing checkpoint; no production process is changed.
uv run python -m native.bend_engine.neural_probe.checkpoint_probe \
  --checkpoint /path/to/copied/trainer.pt --device cpu --batch 4 \
  --report artifacts/bend-neural-checkpoint.json

# Explicit UNTRAINED two-layer transformer fixture, not TinyNet or a learned net.
uv run python -m native.bend_engine.neural_probe.checkpoint_probe \
  --fixture-checkpoint --modes generic portable native ubsan \
  --report artifacts/bend-neural-checkpoint-fixture.json
```

The transformer returns `policy_own`, unlike TinyNet's `policy`. The wrapper uses
the existing production `_policy_output` helper and WDL logits, sets the existing
inference-only flag, and does not substitute auxiliary policy heads. The original
model architecture is reconstructed from the file, not a guessed current YAML.
A SHA-256 and the selected state key bind the package/report to that checkpoint;
step metadata is reported, never treated as proof of training provenance.

The coordinator derives its real-row limit from the selected bucket, with bounded
reservation `max(8, batch)`; the one-real-row control still pads that same package.

Each command exports one static package, checks native outputs against eager
singleton forwards, and runs five actual Bend searches with the existing batched
coordinator/control and cancellation recovery. Per-reply reference checks and
full final tree/visit/move comparisons remain mandatory. A numerical or search
mismatch remains a failure; this runner does not automatically increase tolerance.

CPU runs use F32 and the previous 2e-6 absolute / 2e-5 relative logit tolerances.
The v3 package format also supports an **experimental CUDA BF16 execution path**:
append `--device cuda --device-index 0 --atol A --rtol R`, replacing A/R with
numerical tolerances chosen BEFORE the experiment for that model. Neither
missing hardware nor missing explicit CUDA tolerances falls back to CPU. Run only
on an NVIDIA host with compatible CUDA PyTorch/LibTorch, during a budgeted or
training-paused window. No GPU is reserved by this command and no live training
is paused automatically. Compilation/inference can consume significant GPU memory.

The package and native worker are built from the same Torch version. Input wire
payloads remain F32; CUDA transfers/casts to BF16 before inference and returns F32
policy/WDL logits via a blocking host copy. Device identity and output dtype/shape
are checked. This is NOT zero-copy, CUDA graph, overlapping-transfer or latency
qualification. Existing BF16 dictionary-output packages are not interchangeable:
the v3 exporter creates the explicit tuple-output contract from the checkpoint.
The previous native CUDA parity command remains unchanged and independent.

Export alone does not qualify row independence or real chess strength. Tiny and
transformer model families are accepted, but only actually exercised shapes,
weights, targets and batch buckets are qualified by a successful report. Partial
batches, one-real-row controls and cancellation continue using the same physical
batch package. No end-to-end speedup is inferred. Failure reports record the stage;
no trained-weight or GPU pass is inferred from the synthetic CPU fixture.


## Preflight, retained evidence and package reuse

Before spending target-GPU time, use a trusted immutable copy of the checkpoint
and a separate development checkout. `--preflight-only` performs the strict CPU
weight load, encoding/config checks, source-pinned compiler verification, native
executable discovery and requested-device checks. It does **not** export, execute
a model forward, start a native worker, or search. A CUDA preflight may initialize
a CUDA context to query the device and free-memory snapshot; it neither reserves
memory nor guarantees that compilation/inference will fit beside another job.
Its report says `status: preflight_passed`, `qualification: not_run`, never a
neural/GPU PASS. CUDA still requires explicit tolerances even for this preflight.

```sh
# Run in the CUDA environment with a safe compute window; no automatic fallback.
# CHECKPOINT, ATOL and RTOL must be chosen before the experiment.
python -m native.bend_engine.neural_probe.checkpoint_probe \
  --checkpoint "$CHECKPOINT" --weights-key model \
  --device cuda --device-index 0 --batch 4 --atol "$ATOL" --rtol "$RTOL" \
  --preflight-only --report artifacts/bend-preflight.json

# Full qualification. This directory MUST NOT already exist.
python -m native.bend_engine.neural_probe.checkpoint_probe \
  --checkpoint "$CHECKPOINT" --weights-key model \
  --device cuda --device-index 0 --batch 4 --atol "$ATOL" --rtol "$RTOL" \
  --work-dir build/bend-qualification/run1 --report artifacts/bend-run1.json

# After correcting a later build/runtime problem, keep the exact exported model.
# Native worker and Bend executables are still rebuilt from the current checkout.
python -m native.bend_engine.neural_probe.checkpoint_probe \
  --checkpoint "$CHECKPOINT" --weights-key model \
  --device cuda --device-index 0 --batch 4 --atol "$ATOL" --rtol "$RTOL" \
  --reuse-package build/bend-qualification/run1/checkpoint.pt2 \
  --work-dir build/bend-qualification/run2 --report artifacts/bend-run2.json
```

The default without `--work-dir` remains disposable. Explicit work directories
retain exports, manifests, CMake/Bend builds and Inductor caches on both success
and failure; they can contain model weights and should not be committed or uploaded
indiscriminately. Existing directories are refused, not reset. Report aliases to
checkpoint/package/sidecar inputs and planned model artifacts are rejected before
writing. Progress is atomically saved at stage boundaries; completed groups survive
later failures. Stage times include checking/build/reference work and are diagnostics,
**not inference-throughput benchmarks**. Worker startup failures retain a bounded
stderr tail instead of merely reporting an EOF. Library calls restore the prior
Inductor cache environment after the run.

Reuse is opt-in and only for an immutable, trusted v3 package. The sidecar and
package SHA-256, exact Torch version, checkpoint file SHA-256, selected normal/SWA
weights, resolved model config, encoding, batch, device index and dtype must match.
A sidecar is not a signature or proof of training. Use compatible hardware (normally
the same host): these fields do not prove portability of generated machine code.
Reuse skips **export only**, not eager singleton comparison, full search/control
checks or cancellation recovery. The same numerical tolerances are recorded, and
a mismatch still fails; no automatic tolerance adjustment or stale-code reuse.
`--reuse-package` requires a real `--checkpoint`, not a regenerated fixture.

See [the preflight/retention record](../../../docs/experiments/2026-09-18-bend-checkpoint-preflight.md).
