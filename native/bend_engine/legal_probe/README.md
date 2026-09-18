# Bend orthodox legal moves and perft

This CPU-only experiment builds on `../bitboard_probe/` and the same
source-fingerprinted U64 compiler. It does not replace a production DeepFin
path or depend on the separate CUDA/MCTS probe stack.

## Boundary

- Python parses FEN into a small, private input record and runs validation.
- C validates the input shape and transfers the initial board and immutable
  attack tables once. Tables come from DeepFin's existing CBoard geometry;
  PEXT-order entries use carry-rippler enumeration and the retained ray walker.
- **Bend owns the board, pseudo-legal moves, copy/make, king-safety filtering,
  castling, en passant, all four promotions, perft/divide, and seeded play.**
  No C legal-move or make-move callback is used by the candidate engine.
- A separate executable uses the actual CBoard move generator and move
  application as the reference, including policy-index decoding. It never
  supplies candidate moves to the Bend executable.

`Chess.bend` threads one linear `Array<U64>` through PEXT lookups. Boards are
reusable Data values. The first implementation deliberately uses copy/make and
lists rather than incremental pin masks, unmake, or packed move buffers. Last
ply perft bulk-counts legal moves without constructing unvisited children.
This is an auditable correctness baseline, not an optimized C replacement.

The root and child comparison uses all six piece masks, both color occupancies,
side to move, castling rights, and raw en-passant metadata. Each move includes
source, destination, promotion and special-move flags. Promotion codes are local
piece IDs, not DeepFin policy indices. Never interchange those spaces.

## Run

From the repository root, with Bun, Clang and Python 3:

```sh
bash native/bend_engine/bitboard_probe/install_toolchain.sh
python -m native.bend_engine.legal_probe.run_probe \
  --report artifacts/bend-legal-perft.json
```

To additionally compare all fixed legal sets and child positions to python-chess:

```sh
python -m pip install 'python-chess==1.999'
python -m native.bend_engine.legal_probe.run_probe --python-chess
```

No Torch or Python extension build is needed. `--compiler-root`, `--bun`, `--cc`,
and `--modes` can select an existing isolated, verified toolchain. The produced
executables need no Python interpreter; FEN decoding is currently in the Python
harness, not a Bend FEN parser. The candidate executable consumes the documented
private numeric input in `support.h`, not UCI.

A single position, with per-root-move divide and CBoard comparison:

```sh
python -m native.bend_engine.legal_probe.run_probe --modes native \
  --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' --depth 5
```

A reproducible random-legal-move trace (numeric moves and full child boards):

```sh
python -m native.bend_engine.legal_probe.run_probe --modes native \
  --fen 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' \
  --random-plies 64 --seed 7
```

Seed zero is mapped to a fixed nonzero xorshift32 seed. Random play is a test
fixture, not a strength policy or a promise of perfectly uniform sampling.
`checkmate`, `stalemate`, and `ply_limit` are distinct trace endings.

## Validation contract

The path-scoped native workflow checks generic C, forced-portable U64 helpers,
native-target C, and UndefinedBehaviorSanitizer C. Each build checks:

- all legal moves and exact child states for 30 fixed positions;
- 33 unique perft/divide cases, including the six published canonical perft
  positions also used in `tests/test_perft.py`;
- explicit castling/transit/promotion/en-passant and conservative-blocker rule expectations;
- four repeatable 32-ply traces, validating every chosen move and child against
  CBoard, plus complete legal sets and depth-2 divides at 16 sampled positions;
- initially checkmated/stalemated traces, including the one-ply-budget boundary;
- malformed input, missing kings, invalid metadata/modes, excessive depth and a
  position where the nonmoving king is already attacked must fail explicitly.

The CI job adds python-chess as a second test-time rules oracle for the fixed
positions. Source fingerprints are verified before compilation. The input C
adapter checks the pinned Job constructor arity, uses low/high U32 storage for
U64, and checks allocation failures before writing into the runtime heap.

Ordinary pytest receives only cheap FEN/wire/report parser tests. Native
compilation and traversal are **not** part of every PR's general pytest suite.
The new workflow's highest default depth is **4 for startpos, 3 for the other
canonical positions, and 2 for edge/random samples**. Existing perft tests and
depths are unchanged. Depth 5 is available explicitly; this prototype rejects
larger depths to keep requests bounded. Timings include setup, output and
reference validation, so they are not move-generation throughput measurements.

## Limits

Only orthodox chess and structurally valid positions with consistent castling
and en-passant metadata are admitted. This is not a proof of historical
reachability for arbitrary FENs. Static input checks do not constitute a full
FIDE-position validator. Computed moves never capture a king and are filtered
against attacks on the moving side's king, including pinned enemy attacks.

Perft counts move sequences, not draw adjudication: repetition, fifty/seventy-
five moves and dead-position termination are intentionally absent. FEN counters
are syntax-checked but not stored. The random demo can therefore continue in
positions a complete game controller would draw, until its explicit ply cap.
No evaluation, search strength, UCI, NN inference, history encoding, GPU execution
or production integration is claimed.

All Bend definitions pass the checker without `@unsafe`; no complete formal
chess specification or universal legal-move equivalence theorem is supplied.
This change has a self-review plus executable parity checks, not an independent
human/model review. The fork's upstream source-token maintenance-budget issue
remains separate and unchanged.


## Opt-in perft timing and conservative king-safety fast path

Outside check, an ordinary non-king/non-EP move only needs the full make/check
filter when its source is the first occupied square on a sliding ray from its
king. Any newly exposed slider must pass through that first blocker. The target
remains occupied by our moving piece, including captures/promotions. This is a
conservative test, not an exact pin detector. King moves, en passant, and every
move while in check still use the full filter. Extra fixtures cover file/diagonal
pins, a second blocker, an unpinned first blocker and both colors.

Benchmark separately from validation (requires an otherwise idle CPU):

```sh
python -m native.bend_engine.legal_probe.benchmark \
  --report artifacts/bend-perft-benchmark.json
```

Defaults are startpos depth 5 and Kiwipete depth 4, three measured samples per
engine, one CPU thread, native-target Clang. This is not called by regular tests.
Each process initializes tables, runs a full warmup and verifies its count, then
measures one traversal with CLOCK_MONOTONIC. Compilation, initialization, table
marshalling, printing and final table destruction are outside the interval.
The Bend timer boundary includes a small IO-dispatch cost. On Linux the timing
processes inherit the same single-CPU affinity; the original affinity is restored.
Raw samples, median/min/max, counts, C slider backend, compiler flags and source
hashes are recorded. No relative-speed assertion is made part of CI.

`--mode generic` compares portable-target builds (CBoard uses magic, Bend uses
its runtime-dispatched PEXT). `--mode native` applies identical target flags to
both hot loops; read the C backend in the report rather than assuming PEXT.
The C executable uses the actual CBoard generator/copy/push path, not Python
recursion. Both perfts bulk-count legal moves at depth one; leaf counts are not
numbers of executed make-move operations.

To compare a previous core on the SAME compiler, clock, tables and flags:

```sh
git show 7e33ebe5c7aeada732fad63034181c0138ed4ca4:native/bend_engine/legal_probe/Chess.bend > /tmp/Chess.baseline.bend
python -m native.bend_engine.legal_probe.benchmark \
  --baseline-chess /tmp/Chess.baseline.bend --samples 3 \
  --report artifacts/bend-perft-baseline.json
```

Only the supplied core source differs. Its SHA-256 is recorded; it is not
silently downloaded or treated as another compiler version. This remains a
single-thread prototype benchmark, not an engine-strength result or a claim
that Bend beats optimized C. The experiment record is
[here](../../../docs/experiments/2026-09-18-bend-perft-baseline.md).


## Scalar branches and opt-in allocation counters

The hot core uses a typed `select_u64` match instead of generic `Bool.pick`:
on the pinned compiler this keeps the two U32 limbs scalar rather than boxing
both alternatives. `retain_move` constructs a list node only when retained,
without sharing the accumulator with an unused list alternative. These are
representation changes, not changes to move legality or ordering.

The opt-in counter diagnostic is separate from ordinary timing and CI:

```sh
python -m native.bend_engine.legal_probe.profile_allocations \
  --report artifacts/bend-allocations.json
```

It supports `--baseline-chess` exactly like the benchmark. Counts cover only
the post-warmup traversal and include runtime heap requests, reference-wrapper
creation/increments and destructor entries. These are instrumented runtime
function calls, NOT OS allocations, peak bytes, or time shares. Instrumented
timing is intentionally omitted; use `benchmark.py` for performance. Generated
C is modified only in a disposable directory; missing or ambiguous anchors
fail closed. The compiler pin and normal executable source remain unchanged.

See [the scalar-branch record](../../../docs/experiments/2026-09-18-bend-scalar-branches.md)
for the controlled old/new comparison and limitations.
