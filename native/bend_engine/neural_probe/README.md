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

## A specified checkpoint rather than the smoke model

See [the checkpoint/device contract](CHECKPOINTS.md) for strict loading of an
embedded-architecture checkpoint, frozen CPU/CUDA export, and its explicit
batched-search qualification command. This is a separate v3 package contract;
existing v1/v2 TinyNet commands remain unchanged. No trained or CUDA pass is
implied by the reduced-transformer CPU fixture.
