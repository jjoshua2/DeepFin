# Experimental Bend engine

This directory is the staging area for a possible shared native search engine
for both DeepFin UCI play and distributed selfplay.

The first slice is intentionally tiny: `probe/` proves that a native Bend
program can call DeepFin's existing **pure-C CBoard** through a foreign effect,
receive real legal policy actions, perform deterministic PUCT-shaped F32 search
math in Bend, apply the selected move back in C, and agree with the existing
Python/C extension.

Nothing here is on a production UCI, selfplay, training, or inference path.

## Qualified Bend toolchain

PR 1 is qualified against:

- Bend **2.0.4**
- bendlang/bend commit
  `8008146ab90abb98b496fa2a6ffe555da7fb0dd5`

See `BEND_VERSION`. Bend is young and moves quickly; when changing the pinned
version, rerun the parity probe rather than assuming source compatibility.

Install Bend using its upstream installer:

```bash
curl -fsSL https://bend-lang.com/install.sh | sh
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
