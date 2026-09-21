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
| Repetition identity, automatic draw rules, search-leaf history | Rules.bend and SearchHistory.bend |
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
C replacement for the Python application. `table_reference.c`, `verify.py` and `verify_rules.py`
are external tests; neither is imported or linked into the engine.

## Build and run (Linux x86-64)

Bun 1.4.2, Clang, Git and standard shell/hash tools are build dependencies, not
runtime dependencies. The build uses its OWN compiler pin and checks all 84
compiler/effect source files. Older probes and their pin are left unchanged.

```sh
bash native/bend_engine/standalone/build.sh
./build/bend_standalone/deepfin-bend --threads 1
```

The optional arguments are NEW_OUTPUT_DIRECTORY COMPILER_SOURCE_DIRECTORY MODE,
where MODE is generic (default), portable, native, ubsan or static. Existing output
directories are refused. A missing compiler checkout is fetched at the pinned
commit; an existing one must match `toolchain.json`'s complete source fingerprint exactly. BUN and CC can point
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
still usable. Diagnostics `d`, `rules` and `perft 0..5` are not part of standard UCI.

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
The checked September 20 fork update includes that fix, Bend 2.0.21, and the
upstream cyclic-template termination fix.
This entry point now pins **`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`**, from
`jjoshua2/bend`'s `feat/u64-compact-reviewed` branch. The fork's `main` is
upstream-only; it does **not** contain the native U64 extension. Installing a
moving upstream release or copying just Base definitions is not equivalent.

`toolchain.json` is the single revision/fingerprint contract for this entry point.
The default source cache is `build/bend_standalone_toolchain/REVISION/source`, so
the old `d9b9bce9...` cache is neither reused nor reset. Explicit source directories
must match all 84 pinned compiler/effect files; stale or modified contents are
rejected before compilation/output creation. `build.txt` records the revision and
source fingerprint actually checked. Existing output directories remain protected.
To keep an old executable while building this update:

```sh
bash native/bend_engine/standalone/build.sh build/bend_standalone_20260920
./build/bend_standalone_20260920/deepfin-bend --threads 1

# Build-time contract tests, not deployed engine dependencies:
bun test native/bend_engine/standalone/verify_compiler.test.js
```

The older bitboard/legal/session/neural probes still use their explicit
`57bc84ed...` pin in `bitboard_probe/toolchain.json`; the release-based probes use
their separate installer. They are preserved historical integration references,
not the standalone engine's compiler. This update does not claim to have migrated
or requalified those other entry points against the new standalone compiler.

Fork PR #2 remains draft: its compiler source-size gate now passes, while strict
TypeScript reports diagnostics reproduced on exact upstream. Targeted source
proofs and standalone executable checks are separate from that strict gate.
No kernel change, budget relaxation, or upstream feature merge is implied.

## Automatic draws are Bend-owned

`Rules.bend` compares complete piece/color/turn/castling identity, ignoring clocks
for repetition and normalizing en-passant to a square only when an actual legal
EP capture exists. In particular, pinned EP is not a different repetition state.
Counting includes the current board plus stored history within the reversible
halfmove window. A FEN-only root cannot invent earlier repetitions. The parser
can load analysis histories beyond an outcome; that is not a claim that such a
game could legally continue in a tournament.

The automatic rules are fivefold repetition, 150 halfmoves without a pawn move
or capture, and a conservative material subset (bare kings, a sole minor, or only
same-color bishops). This is not exhaustive dead-position detection. The two-knight
and opposite-color-bishop cases are not collapsed to a draw. Native mate/stalemate
checks precede these tests, so a mate on halfmove 150 stays a win.

`SearchHistory.bend` walks the selected node's actual parent chain with 32-step
fuel, checks bounds/decreasing IDs, replays moves into the Bend-owned full history,
and verifies the resulting full board equals the requested leaf. No Python/CBoard
history reconstruction or C rules callback is involved. A confirmed rule draw uses
the existing Search terminal-zero reply; no children or material evaluation are
needed, and later visits use the cached value. Cached states are local to one
history path and fresh tree; transposition/subtree reuse would require a separate
history-safety review. Arena exhaustion can still stop before the rule query.

An optional threefold or fifty-move claim is NOT an automatic ending. This port
does not silently remove winning continuations by forcing such claims. It also
does not yet implement the Python prototype's optional search claim policy.

The read-only `rules` diagnostic prints `info string rules REASON REPETITIONS
HALFMOVE`. Each newly confirmed automatic search leaf prints `info string
rule_draw NODE REASON REPETITIONS HALFMOVE`. An automatically drawn root can still
have legal moves: its UCI response is explicitly labeled as a legal protocol
fallback, not a searched continuation or a claim. `bestmove 0000` remains reserved
for no-legal-move positions. The GUI owns final adjudication.

```sh
python -m native.bend_engine.standalone.verify_rules \
  --report artifacts/bend-owned-rules.json \
  --command ./build/bend_standalone/deepfin-bend --threads 1
```

This is an opt-in external oracle; neither Python nor a helper executable is
linked or launched by the engine. Use the same empty-chroot command prefix as
above for interpreter-free execution. `perft` remains strictly a legal-move count
and deliberately ignores draw adjudication. The extra history work has not been
performance-qualified, and does not imply a speedup or a complete rules proof.

## What is NOT ported yet

This is not feature parity with the Python scaffolding. Automatic history draws
now run in Bend; optional claim choices are not yet migrated into this entry point.
Mate/stalemate are handled by the existing native search. The GUI remains
responsible for played-game results/claims: UCI has no claim-action encoding.
There is no neural encoder, model loading/inference, batching, training, PGN export,
subtree reuse, production Gumbel parity, advanced time management, strength or speed
claim. A material evaluator makes this initial runtime-isolation test independent
of model export infrastructure. The next migration work belongs in Bend (claim choices
and neural encoding), rather than adding another Python orchestration layer.

Self-reviewed, not independently reviewed or formally proven. No production entry
point, prior branch, compiler kernel, perft depth or existing test is replaced.

## Bend-owned 112-plane neural history block (partial model input)

`HistoryEncoding.bend` constructs the common **112 x 8 x 8 float32 block** directly
from `Position.Game`, including all available prior positions. Both supported
layouts keep every history slot in the evaluated position's side-to-move point
of view (black flips ranks, not files):

- `lc0_root`: eight 13-plane history slots, castling Q/K us/them, black flag,
  raw halfmove clock, reserved zero plane, and ones.
- `lc0_root_legacy_meta`: the same layout except plane 109 is min(halfmove,100)/100
  and plane 110 is the FEN EP file. EP metadata is present even when no legal EP
  capture exists; **repetition identity**, separately, uses only legally available EP.

Repetition flags describe whether that individual frame had occurred previously
at its own point in history. A later irreversible move does not erase older
visible flags. This implements **history_rep_fix=true only**. Missing pre-FEN
frames stay zero, rather than repeating the first available board. Unknown layouts
and compatibility requests are rejected, not defaulted or approximated.

This is **not a complete 146/175-plane input**: the additional 34 v1 or 63
v2_threats classical-feature planes are not implemented here. They are NOT appended
as zeros. Policy-head index mapping, model execution and batching remain separate.
The existing material evaluator and default search path do not invoke this encoder.
There is no neural-play or throughput claim from comparing input values.

The typed `encode(layout, game, table)` returns the table owner and a native
`Array<F32>` with 8192 capacity; **only its first 7168 elements are the logical
[112,8,8] block**. Future inference integration must supply an explicit complete
model shape, not use the physical Array capacity as its channel count.

A read-only command exposes exact IEEE float32 bits for the external oracle:

```text
position startpos moves e2e4 e7e5
encode_history lc0_root_legacy_meta
encode_history lc0_root moves g1f3 b8c6
```

It returns one `history_encoding` header marked `partial_input`, 112 `history_plane`
rows of 64 decimal U32 bit patterns, and `history_encoding_end`. This is diagnostic
text, not a neural input wire ABI. Optional hypothetical moves are legality-checked,
limited to 32, and leave the accepted root/history unchanged. An invalid late move
rejects the whole request with no partial plane block. Busy searches reject the
command. Diagnostic replay can inspect analysis sequences after an automatic draw;
it does not redefine played-game adjudication. This synchronous diagnostic output
is not preemptible and may delay processing stop/readiness until it finishes.

Build/run still needs **no Python**. Tests use the original Python and optionally
C encoders outside the candidate executable, and never feed it reference planes:

```sh
python -m native.bend_engine.standalone.verify_encoding --require-c \
  --report artifacts/bend-history-encoding.json \
  --command ./build/bend_owned_encoding/deepfin-bend --threads 1
```

`--require-c` needs the existing `_lc0_ext` extension built in the external test
environment. It sets the repetition fix before CBoard construction. Every logical
F32 bit must match both oracles; positions beyond CBoard's uint8 clock range are
checked against Python only and counted explicitly. Omit `--require-c` for a
Python-only reference check, which the report labels accordingly. Native encoding
traversals remain opt-in; ordinary pytest/perft depths and workflows are unchanged.

## Checked slider-index laws

`LAWS.bend` and `PROOF.bend` connect the actual sequential table builder to the
actual PEXT lookup. They include generic inductive bridge/bit lemmas and an
exhaustive source-checked certificate for all 128 masks and 107,648 generated
occupancy subsets. Unwrapped addresses, U32 truncation and segment ordering are
checked explicitly. This is not a proof of the complete engine or C compiler.

The [proof guide](proofs/README.md) defines the exact scope, explicit
preconditions, pinned dependencies, commands and failure controls. The separate
`.github/workflows/bend-slider-laws.yml` source gate runs on relevant PR changes;
existing native/perft/encoding qualification remains opt-in and unchanged.
