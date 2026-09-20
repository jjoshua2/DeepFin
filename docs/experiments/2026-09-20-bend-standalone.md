# Bend-authored application, no Python runtime

## Revised question and predeclared hosted gate

Owner clarified Bend everywhere, not permanent Python orchestration. Base #797:
3b6047ddfc08d9aed43da37aff8f2db218f84d98. Preserve the older Python scaffolding as
reference, but implement the standalone application in Bend. No production/live
changes, neural checkpoint, GPU, training, perft increase or deployment.

Acceptance: compile one directly launched diagnostic executable. UCI/FEN parsing,
played-history ownership, table initialization, evaluator and search/control
choices must live in Bend. Native exceptions are limited to raw IO/clock/sleep and
the compiler runtime, not chess/application decisions. A static executable must
pass the external tests in an otherwise empty chroot. Test tools may use Python
outside that root; no interpreter/packages/launcher/data can be inside it.

Check exact full board/clocks/history for 137 legal transitions from 10 roots,
40 seeded history extensions, 51 bounded searches, invalid transactional inputs,
partial lines, readiness/stop during search, repeated stop and subsequent recovery.
Use an independent python-chess UCI client for eight played plies. Compare startpos
perft3=8902, Kiwipete3=97862 and canonical endgame4=43238 (no deeper default). Also
compare all108160 used Bend-built attack-table words with the existing C reference
in a separately linked test executable. Run generic, portable, native, UBSan and
static-isolated variants. No performance acceptance threshold or claim.

Budget: isolated Linux CPU, two concurrent compilation processes at most, native
threads=1, ten-minute hosted cap. No permanent new workflow or ordinary pytest
native execution; the verifier stays opt-in. Failure recovery is discarding this
isolated feature, not resetting user branches. Self-review only.

## Implementation and compiler regression

New standalone modules own ASCII token/checked-number parsing, structural FEN
validation, legal move replay, full played board/move history, clock updates,
command state, limits, material evaluator and uniform priors. Search/Chess remain
unchanged pure Bend modules. Tables.bend enumerates every occupancy and initializes
slider/leaper tables once, without a C callback or a table file. Only poll.c is
new handwritten runtime code: generic bounded nonblocking stdin packet transport.
Base provides output, monotonic time and sleep. No exec, CBoard, Python or model
worker call exists in the candidate. The test reference is separately linked.

This work exposed U64.from_u32's codegen bug on a variable u32: emit_alias could
reuse that C local and then extract its high half using a width-32 shift. Fork
PR #3 explicitly casts to u64 before splitting. Before/after hosted regression:
https://github.com/jjoshua2/bend/actions/runs/35510038201. Old code fails with
-Werror=shift-count-overflow; fixed normalization/JS and four C modes return eight
correct sign/wrap-boundary values. Existing U64/array tests remain green. New
standalone pin d9b9bce9ce4c02583ca1a548cfd239b368c3fc77, 74 source/effect hashes.
Other DeepFin probes keep their prior compiler pin and behavior.

## Local readout

Clang17, Bun1.4.2, pinned compiler with the exact widening fix. Dynamic and UBSan
executables pass the full raw-client suite. The static build passes the same
suite plus the independent python-chess UCI client inside an empty chroot whose
only file is the engine. There is no dynamic interpreter segment or libpython
requirement. The test driver runs outside the root through standard pipes.

Per variant: 10 root fixtures,137 exact legal transitions,51 bounded searches,
23 invalid position/go transactions, all three canonical perft counts, overlong/
NUL record rejection, partial-line readiness, one bestmove after repeated stop,
and new-root recovery. Eight real-client plies are legal and the engine keeps
running. They are material-evaluator moves, not learned-strength evidence.
All108160 Bend/C table words compare byte-for-byte, serialized table SHA-256
37cf5ae16f1709ef3220c30f60fbd142bc32be6315016dd037a6a8e631c5a663.
Ruff passes the external verifier. No assertion/tolerance was relaxed.

Two implementation errors were caught before publication: the starting-position
black-pawn constant initially occupied the wrong rank (perft caught it), and
adding fallback labeling required marking a reused Bend number as copyable (the
checker rejected it). Both were corrected before the final native tests.

## Hosted readout

Pending. Exact source hashes, compiler pin, runtime-root inventory, executable
identity, per-mode reports and table equality will be recorded after execution.
No local result is described as a hosted result. No target-GPU or model execution.

## Scope / next decision

The no-Python-runtime diagnostic milestone is narrower than a complete Python-free
DeepFin. History/clocks are owned by Bend but draw adjudication/claim options are
not ported into this entry point yet. Model encoding/inference/training and batched
scheduling are also absent. These remain Bend migration work, not permanent Python
exceptions. Old tests and scaffolding remain as independent migration references.

The UCI subset lacks clock/increment allocation, options, ponder/searchmoves and
continuous infinite search. Search is bounded to256 simulations/4096 nodes/horizon32.
Stop is cooperative between steps; initialization, FEN replay, synchronous explicit
perft and stdout backpressure are not preemptible. Finite event-loop fuel and
bounded input/history avoid pretending this is a production daemon. No latency,
throughput, optimal play, full UCI or proof-of-correctness claim. Static isolation
is a dependency demonstration, not a security sandbox.

Run/build/ownership details: native/bend_engine/standalone/README.md. Nothing is
merged/deployed; independent review is not claimed. No routine test depths change.
