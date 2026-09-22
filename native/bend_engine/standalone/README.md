# Bend-owned standalone diagnostic engine

This is the first no-Python-runtime slice of the Bend-everywhere experiment.
It is a separate entry point, not a replacement for production DeepFin or the
Python neural/UCI scaffolding. The application is authored in Bend and compiled
to a directly launched executable. The evaluator is explicitly **material-only**,
not a neural model. No interpreter, subprocess, checkpoint or attack-table file
is needed at runtime.

## Ownership, not just a wrapper

| Responsibility | Implementation |
| --- | --- |
| UCI tokens, decimal limits, command/state dispatch | Text.bend, Protocol.bend, main.bend |
| FEN, castling/EP metadata, exact legal move replay | Position.bend and existing Chess.bend |
| Played board/move history and both clocks | Position.bend; retained in Bend state |
| Slider masks/subsets/PEXT table entries, leapers | Tables.bend, generated once at startup |
| Legal moves, king safety, promotions, perft | Existing pure Chess.bend |
| PUCT tree and uniform-policy/material replies | Existing Search.bend and main.bend |
| Poll cadence, search budgets, stop/busy/result decisions | main.bend |
| Raw nonblocking ASCII stdin records | poll.c (generic bounded byte transport only) |
| Stdout, monotonic clock, sleep, allocation/thread primitives | Bend Base/compiler native runtime |

`poll.c` contains no UCI keyword, chess board, evaluator, move selection or
application controller. It reads at most 8192 bytes per invocation and returns a
packet tagged empty/line/EOF/invalid. Partial lines do not block computation.
The compiler-generated C implementing Bend definitions is not a handwritten
C replacement for the Python application. `table_reference.c` and `verify.py`
are external tests; neither is imported or linked into the engine.

## Build and run (Linux x86-64)

Bun 1.4.2, Clang, Git and standard shell/hash tools are build dependencies, not
runtime dependencies. The build uses its OWN compiler pin and checks all 74
compiler/effect source files. Older probes and their pin are left unchanged.

```sh
bash native/bend_engine/standalone/build.sh
./build/bend_standalone/deepfin-bend --threads 1
```

The optional arguments are NEW_OUTPUT_DIRECTORY COMPILER_SOURCE_DIRECTORY MODE,
where MODE is generic (default), portable, native, ubsan or static. Existing output
directories are refused. A missing compiler checkout is fetched at the pinned
commit; an existing one must match `verify_compiler.js`'s complete source fingerprint exactly. BUN and CC can point
to explicit executables. Builds use -O1 and no floating-point contraction for this
correctness gate; no performance claim is made. Native-target binaries are not
hardware portable. Static needs the host's static C/maths libraries.

```text
uci
isready
position startpos moves e2e4 e7e5
go nodes 32
```

Supported subset: uci/isready, position startpos or six-field FEN with optional
legal moves, ucinewgame, go nodes/depth/movetime/infinite, stop, quit, clean EOF.
No setoption, clocks/increments, ponder, searchmoves, MultiPV or claim protocol.
Unsupported/invalid limits return an info string, never silently select a model.
A position is committed only if its full FEN and every replayed move pass. Invalid
later moves preserve the previous root/history. FEN structural/king-safety checks
are not an exhaustive reachability proof. Clocks in FEN must be <=1000000, fullmove
positive; played histories are capped at 512 plies. Records are bounded to 4096
ASCII bytes. Overflow, NUL and invalid records are rejected with subsequent input
still usable. Diagnostics `d` and `perft 0..5` are not part of standard UCI.

`d` prints the full current bitboards/clocks/history plus each prior board/move,
for external exact comparisons. `perft` is explicit and synchronous: it does NOT
promise stop responsiveness during that diagnostic traversal. Ordinary search
polls between bounded resumable steps. isready works while searching; busy root/go
changes are rejected. stop yields one bestmove, repeated stop does not duplicate
it. If no simulation completed, a legal fallback is labeled **unsearched**.

Search uses 1..256 simulations, 4096 nodes, horizon 1..32; defaults 64/4.
`info nodes` counts completed simulations, not allocated nodes. `movetime` accepts
1..60000 milliseconds and is checked between steps, not a hard real-time deadline.
`infinite` completes a bounded search and holds the result until stop. It does not
continuously deepen; infinite plus movetime is rejected. Startup/table generation,
position parsing, stdout backpressure and a pure search step are not preemptible.
A sustained input flood can postpone search steps. The outer event-loop fuel is
finite (one billion polls), preserving the language's termination discipline.

## Tests outside the executable

Python/python-chess are **only** an external oracle/client, not part of execution
or build.sh. The verifier compares boards, clocks and complete histories against
python-chess, including every legal child in selected special-move positions. It
also uses the independent python-chess UCI client and raw partial-line/stop tests.

```sh
python native/bend_engine/standalone/verify.py \
  --report artifacts/bend-standalone.json \
  --command ./build/bend_standalone/deepfin-bend --threads 1
```

The decisive deployment test uses a static build and an EMPTY chroot containing
ONLY `deepfin-bend`; run the verifier outside it with `--command sudo chroot DIR
/deepfin-bend --threads 1`. No Python, Bun, shell, shared libraries, data files or
repository are copied inside. This tests runtime dependencies, not a security
sandbox or a formal proof. The same commands and assertions run inside that root.

A separate test compiles `table_dump.bend` and `table_reference.c` plus the old
legal-probe support as two different executables and compares all **108160** used
table entries exactly. Reference CBoard/table code is never linked into the engine.
The standalone verifier is opt-in; no deeper perft or extra benchmark joins CI.

## Compiler finding

Startup table construction exposed a native `U64.from_u32(variable)` bug: alias
reuse could leave the operand as C u32 and generate `variable >> 32`. Fork PR #3
fixes this by explicitly widening first, with a reproducer that fails strict C
compilation before the patch and passes generic/portable/native/UBSan afterward.
This entry point pins `d9b9bce9ce4c02583ca1a548cfd239b368c3fc77`; it does not silently
modify the pinned compiler used by the older probes. No Bend checker change.

## What is NOT ported yet

This is not feature parity with the Python scaffolding. Bend retains game history
and clocks, but this executable does not yet apply automatic draw adjudication or
optional claims from that scaffolding. Mate/stalemate are handled by the existing
native search; the GUI remains responsible for played-game adjudication here.
There is no neural encoder, model loading/inference, batching, training, PGN export,
subtree reuse, production Gumbel parity, advanced time management, strength or speed
claim. A material evaluator makes this initial runtime-isolation test independent
of model export infrastructure. The next migration work belongs in Bend (history
rules and encoding), rather than adding another Python orchestration layer.

Self-reviewed, not independently reviewed or formally proven. No production entry
point, prior branch, compiler kernel, perft depth or existing test is replaced.
