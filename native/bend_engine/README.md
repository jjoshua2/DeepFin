# Experimental Bend engine

This directory is the staging area for a possible shared native search engine
for both DeepFin UCI play and distributed selfplay.

The first slice is intentionally tiny: `probe/` proves that a native Bend
program can call DeepFin's existing **pure-C CBoard** through a foreign effect,
receive real legal policy actions, perform deterministic PUCT-shaped F32 search
math in Bend, apply the selected move back in C, and agree with the existing
Python/C extension.

Nothing here is on a production UCI, selfplay, training, or inference path.

## Bend toolchain

CI installs whatever Bend `https://bend-lang.com/dl/latest.json` currently
names (sha256-checked). The parity probe is the compatibility gate; a new
Bend patch should fail that test rather than a version-string pin.

```bash
# bun is required to run the Bend compiler
curl -fsSL https://bun.sh/install | bash
native/bend_engine/install_bend.sh
export PATH="$PWD/build/bend_toolchain/bin:$PATH"
```

## Build and run the probe

From the DeepFin repository root:

```bash
native/bend_engine/probe/build_probe.sh
./build/bend_chess_probe/deepfin_bend_chess_probe
```

The build has two explicit stages:

1. Bend emits `probe.generated.c`.
2. Clang links that generated program with `chess_bridge.c`, which includes
   DeepFin's existing `encoding/_cboard_impl.h`.

This is deliberate. Emitting C is the integration seam we eventually expect to
use for machine-specific chess primitives and AOTInductor/CUDA inference.

PR 1 uses CBoard's portable ray-slider fallback. It makes **no move-generation
performance claim**. If the boundary works, a later benchmark can bring over
the production PEXT/magic build macros.

## What the probe checks

Five fixed positions cover:

- the standard initial position;
- castling;
- legal en-passant;
- promotion;
- an in-check evasion position.

Synthetic PUCT selects by action id, so the PUCT child may be a king walk.
Castling (`e1g1`) and en passant (`e5d6`) are also applied through the same
`Chess.push` path as extra `feature_*` lines so those encodings hit
`cboard_push_index` in the native binary.

For each position the Bend program:

1. obtains the sorted legal policy actions from CBoard;
2. scores them in Bend with deterministic synthetic PUCT-shaped statistics;
3. chooses the maximum-scoring action;
4. asks CBoard to clone/apply that action;
5. scans the child's legal set again; and
6. prints the counts, checksums, check state, and low 32 bits of the child
   Zobrist position hash.

`tests/test_bend_chess_probe.py` independently computes the same observations
through the existing `_lc0_ext.CBoard` extension and requires exact parity.

The large board representation never crosses the Bend boundary: Bend sees only
U32 board handles and policy action IDs. That is also the intended direction if
current Bend continues to lack a native U64 bitboard type.

## Deliberate non-goals for PR 1

This probe does **not** include:

- CUDA or AOTInductor;
- neural-network inference;
- a real MCTS tree;
- Gumbel sequential halving;
- UCI or worker integration;
- a Bend U64 compiler extension;
- production configuration flags.

Those belong in later PRs only after this smallest native boundary is proven.


## Native AOTInductor deployment probe

`aoti_probe/` validates the next boundary without changing production UCI or
selfplay code:

```text
Bend control flow
    -> foreign C effect
    -> native C++ / LibTorch
    -> AOTIModelPackageLoader
    -> .pt2 AOTInductor package
    -> float32 output words
    -> Bend checksum
```

The large tensor stays native. Bend receives only compact scalar output words,
which is the same ownership direction intended for a future engine where
CBoard/native encoding and the NN runtime own bulk buffers while Bend owns
search and scheduling.

GitHub CI builds a tiny **CPU** `.pt2` fixture at test time, then compiles the
Bend-emitted C and C++ AOTI bridge into one executable using Clang/Clang++ and
C++20. Python is used only to create the fixture and discover LibTorch at build
time; the resulting executable does not embed or start a Python interpreter.
The fixture is compiled with ``BEND_AOTI_PACKAGE_CXX`` or ``/usr/bin/g++`` so
the wrapper's libstdc++ matches the native process — a newer ``g++`` on PATH
can emit a `.so` this host cannot dlopen.

This proves the C++ AOTI deployment mechanism and Bend/native data path. It does
**not** yet prove that a production DeepFin CUDA package runs correctly or at
the desired throughput. A CUDA DeepFin-package parity test is the next gate.
