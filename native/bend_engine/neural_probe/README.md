# Bend search -> real history/policy -> native CPU AOTI

This is an opt-in **composition qualification**, not a production engine or speed
benchmark. It extends the persistent diagnostic PUCT session from #777 without
changing its search decisions, node storage, legal core, or compiler pin.

## What executes where

1. Bend generates the actual leaf, its complete legal move keys and root-to-leaf
   path, retaining its tree while waiting for a reply.
2. `features.c` reconstructs the immutable game root and path with production
   CBoard, validates the independently supplied Bend board and full legal set,
   encodes the real history/features, and maps requested actions to the canonical
   dense-4672 and compact-1858 policy spaces. This CBoard is an **input adapter**,
   not the candidate move generator or tree.
3. `aoti.cpp` loads an actual `.pt2` once, executes it through LibTorch AOTI, gathers
   compact logits in the requested order, and computes legal softmax and WDL.
   It has no libpython dependency. The generated model is the actual production
   `ChessNet` class, reduced to width32/layer1/head2/no Smolgen and **untrained**.
4. The Python test coordinator relays those native probabilities to Bend and
   independently checks encoding, policy mapping, eager neural outputs and the
   complete final tree. Python does not substitute its reference outputs.

**The coordinator is still Python.** This PR does not directly link the model
loader into the Bend executable or establish a Python-free production pipeline.
The two native libraries expose plain C ABIs for a subsequent direct integration.
The existing CUDA parity bridge is not removed or superseded.

## History is part of the input

A current board alone is insufficient. Callers explicitly provide a history-free
seed FEN and a chronological list of legal moves to the search root. Each Bend
request now adds `path <length> <packed keys...>` between `board` and `action`.
Native replay retains the real halfmove/ply counters and CBoard history semantics.
The fixed request limit is 32 search plies and 128 pre-root game moves. The seed
absolute ply is bounded to leave room for both. Longer games require a qualified
history snapshot interface; do not truncate history silently. Chess960 is rejected.

Both current-STM/root-STM encodings and legacy metadata modes (0/1/2), feature
counts 34/63/67 (total 146/175/179), and both repetition-history settings are tested
against the real Python-facing CBoard encoder. The adapter inherits CBoard's clock
and history storage behavior; it does not implement draw adjudication.

Move keys in Bend are NOT neural policy IDs. The boundary validates exact legal
sets and uniqueness, and preserves the supplied action order. Python's independent
`move_to_index` and canonical compact map check black orientation, castling, en
passant and all promotions; no guessed action-ID conversion is used.

## Package contract and failure behavior

Only an explicit `deepfin-bend-cpu-tuple-v1` contract is accepted: CPU, fixed batch1,
input planes/dtype/history flags, SHA-256 package and policy-map identities, and
`(compact_policy_logits, wdl_logits)` output treespec. The native loader checks the
package's actual treespec and output shape/device/type before committing output.
It requires `AOTI_RUNTIME_CHECK_INPUTS=1` before model loading; the coordinator sets
this explicitly. A deliberately inconsistent input-plane sidecar is rejected by
the package's own generated input checks, rather than trusted because its hash is
correct. Direct C-ABI consumers must preserve this requirement.
This is not automatic support for arbitrary existing DeepFin package output order.
A valid hash identifies an artifact; it does not establish its neural semantics.
The qualification therefore also compares full logits against eager execution.

Native feature/evaluator errors leave output buffers untouched. The tests reject
wrong board/path pairs, missing/duplicate/illegal actions, malformed paths,
nonfinite tensors, invalid compact indices and wrong input lengths, then run a
valid request again. Session backend-failure injection preserves completed visits
and a later epoch recovers without restarting the process or reloading the model.
Feature contexts are **serial only**, because production CBoard uses process-wide
encoding globals. The native AOTI handle serializes calls. Concurrency, batching,
model reload and asynchronous cancellation are deliberately not qualified here.

## Run

From an isolated checkout with the normal CPU development dependencies:

```sh
uv sync --locked --extra dev --extra cpu
bash native/bend_engine/bitboard_probe/install_toolchain.sh
CXX=g++ TORCHINDUCTOR_COMPILE_THREADS=1 uv run python \
  -m native.bend_engine.neural_probe.run_probe \
  --report artifacts/bend-neural-contract.json
```

Requires Bun, Clang/Clang++, GCC/G++, CMake and Linux `readelf`, plus the checked
compiler source. Compilation and inference are CPU-only, capped at two Torch
threads and one compiler worker. AOTI uses an isolated temporary cache/package;
no checkpoint, training process, or production package is touched.

The default qualification exports a seeded untrained 146-plane FP32 ChessNet.
`--planes 175|179` and `--dtype bfloat16` select additional smoke configurations;
results apply only to configurations actually run. `--package` accepts a saved
package from this same seeded smoke with its matching JSON manifest, not an
arbitrary checkpoint without a matching eager reference.

Ordinary pytest gains only cheap path/move/manifest checks. Native model export and
search execution have their own path-scoped workflow. **Perft depths and timing
benchmarks are unchanged.** See the [experiment record](../../../docs/experiments/2026-09-18-bend-neural-contract.md)
for evidence, scope and remaining gates.

## Next decision

This gate answers whether real feature tensors and policy identities can drive the
Bend tree through a native model runtime. It does not answer trained playing
strength, production Gumbel parity, CUDA execution, batch scheduling, cancellation
of an in-flight inference, subtree reuse or end-to-end throughput. Those require
separate representative checks, not further isolated perft tuning.
