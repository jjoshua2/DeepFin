# Native Bend board adapter for structural position IDs

`BoardIndex.intern(index, board)` and `BoardIndex.get(index, board)` accept the
existing `legal_probe/Chess.Board` directly. They derive canonical fields and
their key together, then call the unchanged collision-safe PositionIndex. This
removes a caller-supplied hash/identity pairing from the normal native-board API.
Create the underlying owner with `PositionIndex.new(bits)`; use only BoardIndex
operations for that owner's lifetime, not raw entries made with another key scheme.

## Structural identity, not repetition or neural-input identity

The adapter copies all eight bitboards and side to move, masks castling rights to
the four orthodox bits, and drops an EP square only when no side-to-move pawn
could capture there geometrically. This follows the existing CBoard structural
DAG convention on **valid orthodox boards**. A pinned EP capturer still preserves
the square. Repetition adjudication instead needs legal-EP normalization; the
existing Rules module is unchanged, and must not be replaced by this adapter.

The adapter is not a legality validator. Production callers must first establish
a valid Board through the existing parser/move-generation path or an equally
strong boundary. Malformed bitboard sets and arbitrary raw constructors remain
outside its contract. It never takes ownership of a Position.Game, alters FEN
clocks, or discards history; callers keep that context separately. Equal structural
IDs still do not permit reusing history/model-sensitive neural evaluations.

## Key domain and costs

The adapter uses a deterministic word-wise XOR/multiply fold over the eleven
canonical fields. It is **not CBoard's Zobrist key**, a byte-wise serialization
standard, a cryptographic hash, or a promise of persisted-key compatibility.
The mixer has no measured distribution or speed advantage. Full-field equality
in PositionIndex, not the fingerprint alone, decides a match. Changing the mixer
requires rebuilding an index; do not mix entries from different key domains.

PositionIndex's existing fixed record capacity, collision chains, single ownership,
local dense IDs and full-table behavior are retained. No edges, visits, neural
outputs, external node pointers, deletion or transaction API is added. This is a
native adapter and qualification application, not adoption by live search/UCI.

## End-to-end native check

`board_index.bend` receives position commands containing FEN/startpos and optional
UCI moves. The **existing Bend Protocol, Position, Chess and Tables modules** parse,
validate and legally apply those moves. No Python/CBoard-computed key or canonical
record is sent into the native application. It prints the derived fields/key,
original EP marker and game context, insertion outcome and immediate lookup.

The Python reference independently parses/applies with python-chess and keys its
ID dictionary by complete fields, not the hash function. A separate arithmetic
check validates the explicitly defined key fold. Fixed cases cover transpositions,
reversible history, clocks, unusable/capturable/pinned/edge-file EP for both sides,
all four orthodox castlings, both EP captures and all promotions for both sides.
With --chess, every recorded FEN from the unchanged CBoard legal-walk corpus is
reconstructed natively in bounded chunks, with endpoint repeats in each lifetime.
This does not replay every original move history: corpus FENs have no move stack;
the explicit move-command fixtures supply the history/transition checks.

Generic, portable-U64, explicit BMI2/POPCNT and UBSan run the same fixtures. Six
invalid native inputs must fail with the expected exit/diagnostic. Two deliberately
broken adapters preserve every raw EP marker or erase every EP marker; each must
compile and run normally, then fail the unchanged reference. Native crashes are
not passing semantic negative controls. No performance panel or timing gate runs.

```sh
python -m pytest tests/test_bend_map_board_index.py
python -m native.bend_engine.u64_map_probe.board_index \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --bun bun --cc clang-18 --output artifacts/NEW-board-index --chess
```

Use a fresh directory. Reports retain source/case hashes, every output and command
status, including failures. Timed-out command process groups are killed. CI also
triggers on changes to the newly used native chess/position/table dependencies;
it builds current source and does not need a historical artifact as an input.

Local validation: 37 Python cases passed with --noconftest. The unchanged pinned
compiler and Clang 17 built all four native modes. All 1,673 banked corpus FENs
plus endpoint repeats (1,731 observations in 29 chunks per mode) matched a separate
FEN/geometry reader. This is banked-data readback, not a fresh local CBoard build
or the hosted python-chess comparison. Fixed native fixtures and six rejection
paths also executed locally. Hosted all-suite results are recorded separately
on #876. No independent review, formal proof, new benchmark claim, engine change,
compiler change, merge or deployment. Self-review only.
