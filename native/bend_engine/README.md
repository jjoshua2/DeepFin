# Experimental Bend engine

This directory is the staging area for a possible shared native search engine
for both DeepFin UCI play and distributed selfplay.

The first slice is intentionally tiny: `probe/` proves that a native Bend
program can call DeepFin's existing **pure-C CBoard** through a foreign effect,
receive real legal policy actions, perform deterministic PUCT-shaped F32 search
math in Bend, apply the selected move back in C, and agree with the existing
Python/C extension.

Nothing here is on a production UCI, selfplay, training, or inference path.

## Branch workflow: Bend lives on `bend`

Since 2026-11-13 the Bend track is not on `main`. It continues on this
long-lived `bend` branch of the same repository, which carries all of `main`
plus Bend, until it is ready for its own repository. Tag `pre-slim-20261113` is
the last `main` that still carried Bend.

- **New work:** branch from `origin/bend` and open the PR against `bend`. The
  `bend-*` workflows trigger on pull requests targeting `bend` and on pushes to
  `bend` (each still limited by its path filter); they do not run for `main`.
- **Staying current with main:** merge `main` into `bend` periodically with
  `scripts/sync_bend_with_main.sh` from a clean checkout of `bend`, review the
  result, then push it (directly, or as a PR into `bend`):

  ```sh
  git fetch origin
  git switch bend && git merge --ff-only origin/bend
  scripts/sync_bend_with_main.sh            # merges origin/main
  git push origin bend
  ```

- **Why not a plain `git merge origin/main`:** on `main` the Bend files were
  deleted, so a plain merge deletes every Bend file that `bend` has not changed
  since the merge base. It does that silently, with no conflict. Bend files that
  `bend` did change show up as modify/delete conflicts ("deleted by them"). In
  both cases the resolution is the same: keep Bend's copy. The script merges
  without committing, restores every path listed in
  [`branch_paths.txt`](branch_paths.txt) from the pre-merge `bend` commit (this
  also resolves those conflicts), lists any other file the merge deletes, and
  commits only when no conflict is left. It exits 2 if a conflict outside the
  Bend paths remains; resolve that one by hand and `git commit --no-edit`.
  Once a merge that includes the removal from `main` is committed, later
  merges no longer try to delete Bend files, but keep using the script: it is
  a no-op on the Bend paths then.
- **Adding Bend files outside `native/bend_engine/`:** add the path or glob to
  `branch_paths.txt`, or the next sync will not protect it.
- **Known side effect of the first sync:** `main` removed the Bend records' lines
  from `docs/experiments/README.md`, and that edit merges into `bend` like any
  other. The records themselves stay; re-add the index lines on `bend` if they
  are wanted.

## Bend toolchain

CI reads `https://bend-lang.com/dl/latest.json` for the current version and
still sha256-checks the archive. The old feed carried `{ver, sha256, url}`.
The current feed is `{ver, notice}` only; the installer then uses the matching
`bendlang/bend` GitHub release asset and its `digest`. The parity probe is the
compatibility gate; a new Bend patch should fail that test rather than a
version-string pin.

```bash
# bun is required only for a source tarball; official releases ship a binary
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

## Pinned-U64 bitboard experiment

[`bitboard_probe/`](bitboard_probe/README.md) is a separate CPU-only integration
gate using the tested U64 compiler fork, pinned and source-verified rather than
installed from the moving release feed. Bend computes PEXT/magic slider indices
and reads `Array<U64>` attack tables, compared exhaustively with the existing
CBoard ray walker and production magic tables. It does not change the compiler
used by the earlier probes, and it does not replace production move generation.

## Bend-owned legal move generation and perft

[`legal_probe/`](legal_probe/README.md) builds on the pinned U64 slider probe.
Bend owns copy/make, king-safety filtering, castling, en passant, promotions,
perft/divide, and seeded random play; C only loads input and attack tables.
A separate CBoard executable and optional python-chess validate exact moves and
child states. This experiment is isolated from production and the CUDA/search
stack. Native CI is path-scoped, defaults to startpos depth 4 or less, and does
not increase existing perft test depths. See its README for the explicit depth-5
command and scope limits.
