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
