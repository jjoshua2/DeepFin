# Native CBoard perft

Perft counts **legal move sequences of exactly N plies**, not unique positions.
It tests move generation and move application, not neural evaluation or search
strength. DeepFin already has a Python-recursive CBoard perft oracle in
`tests/test_perft.py`. This runner adds native recursion and root divide output
without changing that oracle or the production move rules.

## Build and run

Use a **separate checkout**, not an environment serving an active training job.
Follow [development setup](development.md) for the pinned Python/uv versions:

```bash
uv sync --locked --extra dev --extra cpu
. .venv/bin/activate

# Start position: published depth-5 count.
python -m chess_anti_engine.encoding.perft 5 --expect 4865609

# Root-move breakdown, directly comparable with another engine's divide.
python -m chess_anti_engine.encoding.perft 4 --divide --expect 197281

# Kiwipete: castling, checks and tactical interactions.
python -m chess_anti_engine.encoding.perft 3 \
  --fen 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1' \
  --divide --expect 97862

# Descend into a suspicious branch without editing a FEN.
python -m chess_anti_engine.encoding.perft 3 --moves e2e4 e7e5 --divide --json

python -m pytest tests/test_perft.py tests/test_perft_cli.py
```

`--expect` exits 1 on a count mismatch; malformed input exits 2. `--json` emits
FEN, depth, total, elapsed seconds, nodes/second, the compiled slider backend,
and optional divide/expectation fields. Text divide output is sorted by UCI move.
To localize a discrepancy, compare root counts, append the differing move to
`--moves`, decrease depth by one, and repeat.

## API and semantics

```python
import chess
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.perft import perft, perft_divide

board = CBoard.from_board(chess.Board())
assert perft(board, 0) == 1
assert perft(board, 3) == 8902
assert sum(perft_divide(board, 3).values()) == 8902
```

The diagnostic `_perft_ext` extension includes the **same `_cboard_impl.h`** as
`_lc0_ext` and the C search tree, with the same fast-slider build macros. It uses
`cboard_legal_move_indices`, copies the CBoard struct, and calls
`cboard_push_index`; it does not introduce another move generator or move maker.
The separate module keeps diagnostic entry points out of training/search code.
A root CBoard's public position fields are copied once, without relying on its
Python object's private C layout. There is no Python move traversal. The caller's
board, hash and history are unchanged.

Depth 0 is one leaf, including at mate/stalemate. At positive depth, no legal moves
means zero leaves. **Repetition, 50/75-move draws, and insufficient material never
stop perft.** A private traversal history starts empty because those histories do
not change legal move counts. Moves still use the normal CBoard push operation.
Depth 1 uses bulk legal-move counting; no caching, pruning or neural evaluation is
performed. Divide requires depth >= 1 and includes zero-count root branches.

Counts use checked unsigned 64-bit arithmetic; overflow raises `OverflowError`
instead of wrapping. Depths are limited to 0..64 (divide: 1..64) to bound C stack
usage; this is a safety limit, not a claim that such deep traversals are practical.
Native traversal releases the GIL and periodically checks signals so Ctrl-C works.
Only valid **standard chess** positions are supported, not Chess960. The CLI checks
root validity and the legality of supplied UCI moves before entering native code.

## Validation and benchmarking

The original [published-count suite](https://www.chessprogramming.org/Perft_Results)
remains the primary oracle. Native totals and divide sums are checked against all
six existing positions. The tests also compare per-move divide counts to
python-chess for both colors' promotions/castling and legal/pinned en passant,
exercise draw-independent semantics and root immutability, and check API/CLI errors.
The native tests live in `tests/test_perft.py`, so the existing portable and PEXT
CI selections both exercise them. The reported backend must match `_lc0_ext`.

Timing excludes parsing the root FEN and output formatting, but includes the
native call's root snapshot and, for divide, construction of the result dictionary.
Use **non-divide runs** for throughput comparisons. Compare identical positions,
depths, compiler flags, slider backends and thread counts. This is single-threaded
perft even though its GIL release allows independent callers. Avoid comparing its
nodes/second with the old Python-recursive harness as though only movegen changed.
No measured speedup or correctness result is implied merely by adding the harness.
