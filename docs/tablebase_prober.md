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
| `init(path)` | Loads tables from a `:`-separated directory list (`;` on Windows, Fathom's own separator) and returns the largest man count found. Every directory is searched, and the first that holds a file wins. A second call replaces the table set; `init("")` clears it. **Raises `ValueError`** if a component is empty or not a readable directory (the old table set stays loaded), or if the path loads no table at all (the table set is then empty and `path()` is `None`). Fathom alone would skip a missing directory silently. Each component is resolved with `realpath()` and Fathom gets absolute paths (it opens table files lazily, so a relative path would break after a `chdir`); `path()` reports that absolute form, **with symlinks resolved**. The production directories are symlinks, so `path()` names their targets, not the configured strings: compare it against `os.path.realpath` of each component, never against the raw path. |
| `largest()`, `path()`, `free()` | Introspection and release. |
| `probe_wdl(cboard)` / `probe_wdl_fen(fen)` | -2 loss, -1 blessed loss, 0 draw, 1 cursed win, 2 win |
| `probe_dtz(cboard)` / `probe_dtz_fen(fen)` | DTZ, python-chess `probe_dtz` convention |
| `probe_root(cboard, has_repeated=False, use_rule50=True)` / `probe_root_fen` | `[(uci, dtz, rank, score), ...]`, one entry per legal move; `[]` at a mate or stalemate root |
| `fen_bitboards(fen)`, `cboard_bitboards(cb)` | the parsed input, for tests |

`None` means the position cannot be probed here: it has castling rights, more
men than the loaded tables, or its table is missing from every directory.
Castling rights are read the way each input already carries them. A CBoard's
flags are used raw, as `CBoard.from_board` copied them. A FEN's castling field
is mapped to rook squares exactly as python-chess's FEN parser does
(`_set_castling_fen`), so the FEN path reaches the verdict `chess.Board(fen)`
reaches on clean, stale and absent rights. That parser is asymmetric for stale
letters: `Q`/`q` with no rook on the back rank adds nothing (the position
becomes probeable), while `K`/`k` with no such rook still sets the corner right
(the position stays unprobeable). A letter that lands on a non-corner rook
makes the FEN unprobeable (`None`; python-chess raises `KeyError`). The castling
field accepts `KQkq` letters only, uppercase before lowercase and at most two
per side (python-chess 1.11's `FEN_CASTLING_REGEX`); anything else is a malformed FEN (`ValueError`). python-chess also
accepts Shredder file letters (`A`-`H`), which this reader rejects.
python-chess raises `KeyError`/`MissingTableError` in the same cases. An
illegal position (Fathom's `is_valid`: wrong king count, side not to move in
check, pawn on a back rank, overlapping bitboards, or the impossible en-passant
square described below) or a malformed FEN raises `ValueError`. This holds from
import: the module builds Fathom's attack tables on load, so the legality check
works before the first `init()`.

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
- En passant. Fathom trusts the ep square it is given, and its ep capture
  removes whatever stands on ep∓8. An impossible square can therefore delete a
  king (a segfault in the probe) or invent a capture (a wrong DTZ). The wrapper
  handles every ep square in one gate shared by the FEN and CBoard paths:
  - **No side-to-move pawn can capture onto it** (python-chess's
    `generate_pseudo_legal_ep`): the square is dropped. python-chess's probe
    ignores it too, so values are equal. This covers the ordinary "double push
    with no capturer" and any impossible square nobody can capture onto.
  - **A capturer exists and an enemy pawn stands on ep∓8:** passed through.
    Both probers then capture that pawn, and their values are equal. This
    includes squares python-chess's `status()` calls invalid because the
    origin square ep±8 is occupied, or the square is on the wrong rank: the
    capture is still well defined (it removes a real pawn), python-chess
    probes it, and so does the C prober. The equivalence harness's
    `ep_origin_occupied` class checks this case.
  - **A capturer exists but ep∓8 is not an enemy pawn** (empty, or another
    piece): python-chess applies a nonsense capture, removing a knight or the
    king from ep∓8, and its answer is not a table value for any real position.
    The C prober raises `ValueError` instead. This is the only refused ep
    class. python-chess's own `status()` also flags these positions
    (`STATUS_INVALID_EP_SQUARE`).
  - A FEN ep field of `a1` means "no ep square" in both readers (python-chess
    stores square 0 and its ep logic treats 0 as none); a CBoard cannot carry
    an a1 ep square at all.

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
`has_repeated=True` drops a win out of the 1000 bucket (rank
`1000 - (dtz + clock)`). **`use_rule50` changes only the score, never the
rank.** With it off, every win scores as a mate and every loss as mated, but
the rank stays clock-aware. For example, `8/8/8/4k3/8/8/8/KBN5 w - - 90 1`
gives `a1a2` = `(60, 850, 450)` with `use_rule50=True` and
`(60, 850, 31744)` with `use_rule50=False`.

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

## Where the two probers answer differently

On every legal position within the loaded tables, the two probers return the
same WDL, DTZ and root values. They deliberately differ on these inputs:

1. **The refused ep class** above: a capturer exists but ep∓8 is not an enemy
   pawn. python-chess returns a value for a nonsense capture; C raises
   `ValueError`.
2. **Illegal positions** (Fathom's `is_valid`, listed under API): python-chess
   often returns values for them; C raises `ValueError`.
3. **Positions with more men than the loaded tables** (including 7+ men):
   python-chess sometimes still answers by resolving a capture that leads into
   the tables (its `probe_ab` capture cutoff), and C returns `None`. With
   only 3-4 man tables loaded this also hits 5-man positions with such a
   capture. The C prober never answers above `largest()`.
4. **Non-corner castling rights on a CBoard** (a right on a rook that is not
   on its corner, e.g. `K` with the rook on f1): python-chess raises
   `KeyError`, but `CBoard.from_board` keeps corner bits only, so the CBoard
   path probes the position without that right and returns a value. The FEN
   path refuses it (`None`), the same "no answer" as python-chess.
5. **Shredder castling letters** in a FEN: python-chess parses them, C raises
   `ValueError` (malformed FEN).
6. **FEN whitespace:** C requires exactly one space between FEN fields;
   python-chess accepts runs of spaces. A double space is a malformed FEN
   (`ValueError`) here.

Beyond these, the root's fifty-move-claim rule (above) is a difference in
convention, not in table values.

## Wiring rules for the switch PRs

- **Catch `ValueError` at the root probe and treat it as "no TB".** A UCI
  `position` command can deliver any of the refused or illegal inputs above;
  the engine must fall through to search, never crash.
- **Pass the production pair in full.** `init("<...>/syzygy_3-4-5")` alone
  returns 6 and raises nothing: 6-man WDL works (its WDL files are in that
  directory) and 6-man DTZ returns `None` (its DTZ files are in the second
  directory). The wiring must check DTZ availability for the largest class
  (for example, probe the DTZ of one known 6-man position after `init`) rather
  than trusting `largest()`.

## Thread safety, fork, platforms

Fathom's tables are process-global, so `_tb_ext` is a process-wide singleton:

- Every probe releases the GIL and holds a shared (read) lock on the table
  set. WDL/DTZ probes run concurrently; Fathom protects its lazy table loading
  with its own mutex and atomics.
- `init()`/`free()` take that lock exclusively, so tables are never unmapped
  under an in-flight probe.
- `probe_root` also holds a module mutex. Fathom documents `tb_probe_root` as
  not thread-safe, and the lock makes concurrent calls serialise instead of
  relying on callers.

Fork: there is no `pthread_atfork` handler. Do not fork while another thread
is probing, because the child could inherit a held lock. Call `init()` again in
a forked child before probing; the mappings are inherited, but nothing else
should be assumed. Windows is untested. The path separator follows Fathom's
(`;`), and the directory check uses POSIX `stat`/`access`.

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
- The committed crash fuzz (`scripts/tb_prober_fuzz.py`, and the fuzz test)
  only exercises probes when tables are present, so CI never runs it. What
  protects CI is the table-free refusal test: the illegal and refused-ep
  inputs raise `ValueError` before any table is touched, with no tables
  loaded.

## Follow-ups (not in this PR)

Each switch is its own PR. Each one must prove bit-identical outputs on a
frozen input set, because a difference at a selfplay leaf is a target change.

1. **`SyzygyProbe.apply` leaf WDL** (`tablebase.py`). It builds
   `chess.Board(cb.fen())` for every eligible leaf, which puts python-chess on
   the search hot path. Replace it with `_tb_ext.probe_wdl(cb)`, keeping the
   cursed-as-draw mapping. Gate: identical override rows on a frozen leaf set.
2. **UCI root DTZ shortcut** (`try_tb_root_move` / `_select_dtz_move`).
   Replace it with `probe_root`, keeping the repo's fifty-move-claim rule and
   its distance tie-break inside Fathom's win bucket. At clock ≥ 99, the
   repo's pick fell outside Fathom's top rank in 205 of the 544 positions with
   fifty-claim moves that the independent review measured. All 205 were in the
   same result band. Wiring rule: keep the repo's claim score and use Fathom's
   rank only as the tie-break. When the engine first
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
