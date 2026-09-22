# Bend-owned standalone engine

This is a no-Python-runtime entry point for the Bend-everywhere experiment, not
a replacement for production DeepFin. The default build uses a **material-only**
evaluator and needs no interpreter, subprocess, checkpoint or attack-table file.
The separate [native neural build](#bend-owned-search-leaves-with-a-native-model-backend)
keeps the controller, input and probabilities in Bend while executing a bound
model through native LibTorch/AOTI. That product additionally needs the package
and native libraries, but still no Python runtime. Model math and training are
not yet Bend-authored; the old Python tools remain external migration references.

## Ownership, not just a wrapper

| Responsibility | Implementation |
| --- | --- |
| UCI tokens, decimal limits, command/state dispatch | Text.bend, Protocol.bend, main.bend |
| FEN, castling/EP metadata, exact legal move replay | Position.bend and existing Chess.bend |
| Played board/move history and both clocks | Position.bend; retained in Bend state |
| Repetition identity, automatic draw rules, search-leaf history | Rules.bend and SearchHistory.bend |
| Complete 146/175-plane input construction | HistoryEncoding.bend, ClassicalEncoding.bend and ModelInput.bend |
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
The later September 20 fork update includes that fix and upstream Bend 2.0.21.
This entry point now pins **`aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae`**, from
`jjoshua2/bend`'s `feat/u64-compact-reviewed` branch. The fork's `main` is
upstream-only; it does **not** contain the native U64 extension. Installing a
moving upstream release or copying just Base definitions is not equivalent.

`toolchain.json` is the single revision/fingerprint contract for this entry point.
The default source cache is `build/bend_standalone_toolchain/REVISION/source`, so
older `d9b9bce9...` and `fd1df817...` caches are neither reused nor reset. Explicit source directories
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
or requalified those other entry points against Bend 2.0.21.

The pinned fork commit records passing repository source budgets and its U64 laws/
oracles, while inherited strict-TypeScript findings remain separate. This engine
qualification does not relabel the fork as universally gate-clean or prove compiler
correctness. No checker/kernel edit or budget relaxation is made by this PR.

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

This history-only API is **not a complete 146/175-plane input**. Use the complete
`ModelInput.encode` API below to append the implemented 34 v1 or 63 v2_threats
features. Policy-head index mapping, model execution and batching remain separate.
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


## Complete Bend-owned 146/175-plane inputs

`ModelInput.encode(layout, version, game, table)` joins the unchanged 112 history
planes with actual `ClassicalEncoding.bend` features. Supported versions are exactly
`v1` (146 channels) and `v2_threats` (175), with either supported root-oriented layout
and corrected repetition semantics. Unsupported v3 variants or repfix=false are
rejected, not approximated. No absent feature is supplied as placeholder zeros.

The feature code owns king zones, enemy attacks into those zones, full pin rays,
discovered attacks, passed/isolated/backward/connected pawns, piece mobility,
outposts and space. v2 adds piece-specific attack maps, saturated attacker counts,
hanging/cheaper-attacker masks, safe-check squares, control, tension and pawn storms.
These intentionally match DeepFin's existing training inputs: pseudo-attacks from
pinned pieces are still counted, and pawn mobility is not a legal-move count.

All square indexing uses the current side-to-move perspective. Attacker counts
saturate at seven before the existing clamp/normalization formulas; an eight-versus-
four attacker fixture checks this boundary. Pawn storms follow the C encoder's
float32 division/subtraction rather than Python's double intermediate. Thus every
output bit is expected to match C, with an a priori absolute allowance of 1.2e-7
only on Python's two storm planes (absolute indices 173 and 174). All other Python
planes, including history, mobility and control, require exact float32 bits.

The result carries an explicit channel count and contiguous `Array<F32>` values:
9344 or 11200 logical values. Its physical capacity is 16384; that is not a tensor
shape. No padded extra channels are part of the interface. The model, policy
mapping and batching have not been connected: ordinary search still uses material.

At idle, the read-only diagnostic is:

```text
position startpos moves e2e4 e7e5
encode_input lc0_root_legacy_meta v2_threats
encode_input lc0_root v1 moves g1f3 b8c6
```

The response is a `model_input` header with layout/version/channels and repfix=1,
one `input_plane` row of 64 decimal IEEE-F32 bit patterns per channel, and
`model_input_end`. It is diagnostic text, not an inference wire protocol.
Hypothetical moves (at most 32) are checked before encoding and never replace the
accepted root/history. Invalid commands emit no partial tensor; busy commands are
rejected. Output is synchronous, not a stop/readiness responsiveness guarantee.

```sh
# New output directory; no Python needed to build or run the engine:
bash native/bend_engine/standalone/build.sh build/bend_complete_input
./build/bend_complete_input/deepfin-bend --threads 1

# Optional external Python and C oracles, never linked/launched by the engine:
python -m native.bend_engine.standalone.verify_classical --require-c \
  --report artifacts/bend-complete-input.json \
  --command ./build/bend_complete_input/deepfin-bend --threads 1
```

The verifier requires the original CBoard extension in its external environment
for `--require-c`; its uint8 clock limits are reported as Python-only cases rather
than using wrapped C values. It compares complete tensors, preserves every root,
activates all 63 feature planes and checks invalid/busy/reset behavior. Existing
history/rule/perft verifiers stay unchanged. No model export, native traversal or
benchmark is added to ordinary pytest or recurring CI. See the complete-input
experiment record for the exact tested revisions, counts and limitations.

## Bend-owned policy maps and complete evaluation inputs

`Policy.bend` generates DeepFin's full 4672 / compact 1858 tables and square-pair
lookup from geometry. No generated Python lookup file is shipped. Native packed
move keys (with promotion/castle/EP flags), full search actions and model slots
are distinct spaces. Encoding consumes generated legal moves; reverse resolution
only searches the current board's legal entries. Geometry alone does not authorize
a move. Unknown IDs, padding slots, private claim keys and wrong flags fail instead
of selecting a fallback. Both color orientations, promotions and file-mirror
permutations follow the project's existing `moves/encode.py` ordering.

Read-only diagnostics at idle:

```text
policy legal
policy encode e2e4
policy decode lc0_1858 0
policy key 1804
policy tables
encode_request lc0_root_legacy_meta v2_threats moves e2e4 e7e5
```

`policy decode` may fail when that ID is not legal on the accepted board; it does
not play anything. `tables` dumps every geometric forward/reverse/mirror/pair entry.
These commands preserve the root/history and reject busy searches. Diagnostic
maps are currently disposable and rebuilt per command, not a throughput design.

`EvaluationInput.prepare(layout, version, game, table, maps)` is a typed pure-Bend
composition boundary: generate exact legal policies and complete 146/175-plane
features from the SAME validated `Position.Game`. Its returned value retains map
and table owners for a future caller to reuse; array capacity is not model width.
Terminal boards may have zero legal entries, not a synthetic move/claim. The API
expects a Game already validated by Protocol or reconstructed by SearchHistory.

`encode_request` exposes that paired value, with a legal-policy header/rows followed
by the existing complete input block. Up to 32 hypothetical moves are replayed
transactionally before any output. Bad layouts/versions or a late illegal move
produce no partial request. It is read-only diagnostic text, not an inference ABI.
No model runs, no logits are converted to probabilities, and normal material search
is unchanged. Actual search-ticket scheduling and inference remain to be connected.

The standalone keeps the checked aaeb9bc9 U64 pin from #802. Only application Bend
and external oracle files are added; there is no Python runtime or new foreign C.
Optional external checks (never loaded by the engine):

```sh
python -m native.bend_engine.standalone.verify_policy --require-c \
  --report artifacts/bend-policy.json --command ./build/bend_policy/deepfin-bend --threads 1
python -m native.bend_engine.standalone.verify_request --require-c \
  --report artifacts/bend-request.json --command ./build/bend_policy/deepfin-bend --threads 1
```

The policy prototype was recovered from `bf2d50ada2d0b3d718a987c4d24de7a010ce1ad2`;
its old compiler pin and older Protocol/Main were NOT copied over the complete
input work. This combined version must be validated on its own exact source.


## Bend-owned search leaves with a native model backend

The optional neural build now routes **actual selected leaves** through
`SearchHistory`, `EvaluationInput`, native tensor execution, and `LogitReply` back
to the existing `Search.resume`. It is not only an `encode_request` diagnostic.
Bend owns board/history validation, automatic draws, complete 146/175-plane input,
exact legal policy entries, legal-only stable softmax, WDL probabilities and the
reply identity. An automatic draw bypasses encoding and inference. A predicted
draw remains a neural value rather than a rule-terminal assertion.

`model_call.c` only marshals bounded float lists. `model_bridge.cpp` loads one
pre-exported package and invokes native LibTorch/AOTI, with strict tensor shape,
device and dtype checks. It has no chess, legal masking, probability conversion,
search algorithm, scheduler or Python calls. This is an explicitly **transitional
native model backend**, not Bend-authored transformer mathematics or training.
Python remains in the separate exporter and external references; it is absent
from the running engine. Build tools and exporter are not deployment dependencies.

The default `build.sh` still creates the material-only executable without
LibTorch. Neural execution is a separate build product, bound to one exact trusted
package at build time. The package sidecar must use the existing v3 checkpoint
format, CPU float32, **batch 1**, compact policy width 1858 and corrected
root-oriented encoding. Four layout/feature combinations can be bound; runtime
qualification in this milestone uses the existing **untrained** 175-plane
root-legacy-meta/v2_threats fixture. Do not mistake a binding test for execution
of another layout, trained model, GPU or batch shape.

The model package and sidecar come from the separate checkpoint exporter. Existing
batch-four packages are rejected, not padded or silently re-exported. The supplied
LibTorch CMake prefix must match the package's recorded version. Building the
engine needs Bun, Git, Clang/Clang++, CMake, OpenSSL development files and compatible
LibTorch development files; the build script itself does not invoke Python.

```sh
# Both output directories and generated headers must be new.
bash native/bend_engine/standalone/build_neural.sh \
  build/bend_neural /path/to/checkpoint.pt2 /path/to/libtorch/share/cmake

DEEPFIN_BEND_MODEL_PACKAGE=/path/to/checkpoint.pt2 \
  ./build/bend_neural/neural/deepfin-bend-neural --threads 1
```

The UCI subset is unchanged, for example `position startpos moves e2e4 e7e5` then
`go nodes 4 depth 2`. The native model stays loaded across normal searches and
position changes; trees remain fresh. The sibling `build/bend_neural/deepfin-bend`
is the material-only product, not a fallback selected when neural execution fails.

The build checks package hash and manifest consistency and records the checkpoint
identity declared by the trusted sidecar; it does not load the original checkpoint.
`model_contract.h` records this immutable model contract. Startup copies the package
once into a new private scratch directory while hashing those exact bytes, then
loads only that verified copy. A missing/changed package, incompatible library
version, malformed output, bad legal alignment or nonfinite logit is fatal; no
silent material replacement or fabricated bestmove is returned. Every output
logit is checked, including currently illegal slots. Finite illegal logits do not
enter the legal softmax. Hashes establish identity/integrity, not authentication:
use only trusted executable packages and the original trusted sidecar. This is
not a machine-code portability or numerical-PASS certificate.

`info string native_path` and `native_reply` expose the actual selected ancestor
path and probabilities returned to the search. Optional
`DEEPFIN_BEND_MODEL_TRACE=/new/path.trace` writes the actual input and raw output
float32 bits in an append-only binary stream (new file only, no symlink/overwrite).
Tracing is disabled by default; the new path/reply diagnostics are emitted per
forward. Traces can contain model inputs and outputs, so keep them private when
using private models. The diagnostic stream is not a remote inference protocol.

The deployment now needs **the executable, exact model package, native ELF
libraries and writable scratch**, not just one static file. The external
`isolate_neural` helper tests an otherwise newly created filesystem with only
those files and a read-only `/proc/cpuinfo` snapshot for native CPU dispatch.
It discovers package dependencies using the executable's resolved native library
directories, without copying Python, Bun, a shell or the application source tree.
The host kernel/stdio remain infrastructure, and this dependency test is not a
security sandbox. The original material build can still be statically deployed
as a single executable.

The verifier compares actual traced input to the unchanged C/Python encoders,
raw logits to eager inference, legal priors/WDL to the existing normalization
reference, and subsequent selected paths/result counts/best moves to diagnostic
PUCT. It does not inspect every final tree field or prove all possible positions.
All verifiers/exporters are external, not launched by the engine.

Limits remain deliberate: synchronous CPU batch-one forwards are **not
preemptible**. `isready`, `stop`, `quit` and time limits can wait for a forward,
encoding, or blocked diagnostic output. No hard-stop latency guarantee is made.
Policy maps are currently rebuilt per leaf, and tensor lists are copied across
the boundary; no speed claim or optimized memory/batching claim is implied.
The backend has a 65,536-forward process limit. No CUDA, batched scheduler,
subtree reuse, production Gumbel parity, trained-model strength or training
migration is established. Existing material-mode regressions and perft depths
remain unchanged; new native tests and model export are opt-in only.


## Source-proof coverage

The production `Tables.fill` step calls `Subsets.next`; its arithmetic is unchanged.
The opt-in [proof suite](proofs/README.md) discharges eight initial mask-membership,
recurrence/recovery and closed-boundary laws, reusing the pinned fork's 16 U64 laws.
It does **not** prove complete enumeration order, slider geometry, affine table
lookup, chess legality, neural numerics or the native compiler. See the durable
[migration/proof matrix](../../../docs/bend_migration_proofs.md) and
[subset evidence record](../../../docs/experiments/2026-09-21-bend-subset-source-laws.md).
Neither proof/native gate is added to ordinary pytest or existing perft budgets.
