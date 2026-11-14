# C Syzygy prober (`chess_anti_engine.tb._tb_ext`)

Status: **built and proven equal to python-chess; not used by any production
path.** Every tablebase call site still goes through
`chess_anti_engine/tablebase.py` and `chess.syzygy`. This extension exists so
those call sites can move off python-chess one at a time (the follow-up list
is at the end). It is not in the UCI startup gate's `REQUIRED_EXTENSIONS`,
because the engine does not import it yet.

## What was added

- `chess_anti_engine/tb/fathom/`: [Fathom](https://github.com/jdart1/Fathom)
  at commit `c9c6fef0`, vendored **unmodified** under its MIT licence
  (`LICENSE` kept beside it). `VENDORED.md` records the upstream commit and the
  sha256 of every file, and a test re-hashes them.
- `chess_anti_engine/tb/_tb_fathom.c`: the one translation unit that compiles
  Fathom. It `#include`s `fathom/src/tbprobe.c` so it can call Fathom's static
  `probe_wdl`/`probe_dtz` without patching the vendored file. Nothing Python
  in it.
- `chess_anti_engine/tb/_tb_ext.c`: the Python module: locking, a CBoard
  reader, a FEN reader.
- Build: a sixth `Extension` in `setup.py`. It is pure C, so the host's C++
  runtime mismatch does not apply. It is registered in
  `scripts/check_c_extensions_fresh.py` (the freshness table and build
  records) and is built by `scripts/build_production_extensions.py` like the
  others. It gets no fast-slider macros: `define_macros` apply to every source
  of an Extension, and Fathom defines its own `bishop_attacks`/`rook_attacks`.

## API

All results are from the side to move's point of view.

| call | returns |
|---|---|
| `init(path)` | Loads tables from a `:`-separated directory list and returns the largest man count found. Every directory is searched, and the first that holds a file wins. A second call replaces the table set; `init("")` clears it. |
| `largest()`, `path()`, `free()` | Introspection and release. |
| `probe_wdl(cboard)` / `probe_wdl_fen(fen)` | -2 loss, -1 blessed loss, 0 draw, 1 cursed win, 2 win |
| `probe_dtz(cboard)` / `probe_dtz_fen(fen)` | DTZ, python-chess `probe_dtz` convention |
| `probe_root(cboard, has_repeated=False, use_rule50=True)` / `probe_root_fen` | `[(uci, dtz, rank, score), ...]`, one entry per legal move; `[]` at a mate or stalemate root |
| `fen_bitboards(fen)`, `cboard_bitboards(cb)` | the parsed input, for tests |

`None` means the position cannot be probed here: it has castling rights, more
men than the loaded tables, or its table is missing from every directory.
python-chess raises `KeyError`/`MissingTableError` in the same cases. An
illegal position (Fathom's `is_valid`: wrong king count, side not to move in
check, pawn on a back rank, overlapping bitboards) or a malformed FEN raises
`ValueError`.

The CBoard is read in place, with no python-chess object involved. The
`PyCBoard` struct layout is shared with `_lc0_ext` the same way `_mcts_tree`
shares it: the layout comes from `_cboard_impl.h`, and `tp_basicsize` is
checked on first use.

## WDL / DTZ mapping

**Raw probes: identity. No transform, no tolerance.**

- `probe_wdl` is Fathom's internal `probe_wdl`, through the same path as
  `tb_probe_wdl_impl`. It returns -2..2 exactly as python-chess
  `Tablebase.probe_wdl` does.
- `probe_dtz` is Fathom's internal `probe_dtz`. Both libraries implement
  Stockfish's `probe_dtz`, and both return the same signed value:
  - 0 for a draw;
  - ±1 / ±101 when the best move zeroes the counter (a capture or pawn move);
  - otherwise the table distance plus that offset;
  - |n| > 100 for a cursed win or blessed loss;
  - -1 for a checkmated side to move.
- **The halfmove clock is ignored by both** raw probes. Both assume the
  counter was just reset. Fathom's public `tb_probe_wdl` refuses a nonzero
  `rule50`, so the wrapper calls the internal function and documents the
  python-chess contract instead.
- En passant: a legal ep capture is resolved inside both probers. An ep square
  with no legal capture changes nothing.

**Root ranking: 50-move aware.** `probe_root` returns, per legal move:

- the move's DTZ counted from the root (Fathom `tb_probe_root`'s per-move
  results);
- the rank and score from Fathom's `root_probe_dtz`, which use the board's
  halfmove clock.

The rank is

```
dtz > 0: 1000 if dtz + clock <= 99 else 1000 - (dtz + clock)
dtz < 0: -1000 if 2*|dtz| + clock < 100 else -1000 + (|dtz| + clock)
dtz = 0: 0                                   (has_repeated=False)
```

These are the same buckets as the repo's `tablebase._rule50_dtz_key`.

**The one root difference from the repo's python-chess root logic is
documented, not tolerated.** `tablebase._dtz_of_played_move` returns 0 for a
move after which `child.is_fifty_moves()` holds (root clock ≥ 99, a
non-zeroing, non-mating move). It treats that move as a fifty-move claim.
Fathom has no claim concept: it reports the table distance (±(|child dtz|+1))
and ranks the move in the "50-move draw in sight" band. The equivalence
harness checks that move against `_dtz_ignoring_fifty` (the same function
without that short-circuit), so the comparison is still exact. Whoever
switches the UCI root shortcut must keep the repo's claim rule on top of
Fathom's numbers.

## Thread safety

Fathom's tables are process-global, so `_tb_ext` is a process-wide singleton:

- Every probe releases the GIL and holds a shared (read) lock on the table
  set. WDL/DTZ probes run concurrently; Fathom protects its lazy table loading
  with its own mutex and atomics.
- `init()`/`free()` take that lock exclusively, so tables are never unmapped
  under an in-flight probe.
- `probe_root` also holds a module mutex. Fathom documents `tb_probe_root` as
  not thread-safe, and the lock makes concurrent calls serialise instead of
  relying on callers.

Upstream behaviours to know about: if `mmap` of a table fails, Fathom calls
`exit()`. It prints `info string Incomplete tablebase file` for a truncated
file and skips that file. Tables are opened `O_RDONLY` and mapped `PROT_READ`.

## Evidence

- `scripts/tb_prober_equivalence.py` banks every observation and a manifest
  under the bulk-artifact root, `tb-prober-equivalence-20261114/`. The
  observations are the FEN, both probers' WDL/DTZ, and per-move root rows.
  That run was against the production pair (3-6 man WDL, 3-5 man DTZ in the
  first directory, 6-man DTZ only in the second). Its totals and throughput
  are in the PR that added this file.
- `tests/test_tb_prober.py`: the table-free tests run in CI. The table tests
  skip in CI with a printed reason, because CI ships no tables. Wherever tables
  are found (`$CAE_SYZYGY_PATH`, the production pair, or
  `data/syzygy_3-4man`), they run, and a disagreement fails.

## Follow-ups (not in this PR)

Each switch is its own PR. Each one must prove bit-identical outputs on a
frozen input set, because a difference at a selfplay leaf is a target change.

1. **`SyzygyProbe.apply` leaf WDL** (`tablebase.py`). It builds
   `chess.Board(cb.fen())` for every eligible leaf, which puts python-chess on
   the search hot path. Replace it with `_tb_ext.probe_wdl(cb)`, keeping the
   cursed-as-draw mapping. Gate: identical override rows on a frozen leaf set.
2. **UCI root DTZ shortcut** (`try_tb_root_move` / `_select_dtz_move`).
   Replace it with `probe_root`, keeping the repo's fifty-move-claim rule and
   its distance tie-break inside Fathom's win bucket. When the engine first
   imports `_tb_ext`, add it to `uci/selfcheck.py`'s `REQUIRED_EXTENSIONS`.
3. **Strict match status** (`rule50_match_status`, `selfplay/strict_tb.py`).
4. **Selfplay adjudication and rescoring** (`probe_wdl`,
   `tb_adjudicate_result`, `rescore_game_samples`).
5. **Walker / `puct_tb` leaf hooks.**

**Do not unify or change the leaf eligibility rules while switching.**
Selfplay, arena and classic UCI Gumbel probe at any halfmove clock. The walker
and `puct_tb` probe only at clock 0, and strict mode has a third rule. Moving
any of these changes selfplay targets or engine behaviour, and needs an
experiment record and the owner's go-ahead
(`docs/search_and_encode_paths.md` §6).
