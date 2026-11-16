# C Syzygy prober (`chess_anti_engine.tb._tb_ext`)

Status: **built and proven equal to python-chess; its production callers are
the UCI engine's tablebase probes**: the root DTZ shortcut
(`chess_anti_engine/uci/tb_root.py`, "UCI root shortcut" below) and the
search's leaf and root-eval WDL overrides (`chess_anti_engine/uci/tb_leaf.py`,
"UCI leaf probe" below). With `SyzygyPath` set the engine opens this one table
set and no python-chess tablebase. Every non-UCI call site (selfplay, arena,
worker, training labels) still goes through `chess_anti_engine/tablebase.py`
and `chess.syzygy`. This extension exists so those call sites can move off
python-chess one at a time (the follow-up list is at the end). The UCI startup gate and `deepfin --selfcheck` require it
(`uci/selfcheck.UCI_EXTRA_SYMBOLS`); the shared `REQUIRED_EXTENSIONS` that
also gates the worker and training does not, because they do not import it.

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
| `table_counts()` | `(n_wdl, n_dtz, largest_dtz)`: WDL tables loaded (a material and its colour mirror are one table), how many of them found a DTZ table in some directory, and the largest man count among those. `(0, 0, 0)` with nothing loaded. |
| `probe_wdl(cboard)` / `probe_wdl_fen(fen)` | -2 loss, -1 blessed loss, 0 draw, 1 cursed win, 2 win |
| `probe_wdl_batch(cboards, generation)` | `bytes`, one code per board: 0..4 = WDL -2..2, 5 = `probe_wdl`'s `None`, 6 = its `ValueError`. **`None`** (nothing probed) when the table set is no longer `generation`. One GIL release and one shared lock for the whole batch. |
| `generation()` | identity of the loaded table set; changes on every `init()` that reaches Fathom and every `free()` (a path refused before Fathom leaves it unchanged) |
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
- `probe_wdl_batch` compares the caller's `generation` with the loaded one
  under the same shared lock as its probes, and `init()`/`free()` bump the
  generation under the exclusive lock. A caller therefore can neither probe
  unmapped tables nor be answered from a table set other than the one it was
  configured for, whatever thread it runs on and however late it runs.

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

## UCI root shortcut (`uci/tb_root.py`)

The engine's root DTZ shortcut (`search._try_tb_root_bestmove`) calls
`RootTablebase.root_move`, which makes ONE `probe_root_fen` call for the per-move
DTZ of every legal root move and then reproduces
`tablebase.try_tb_root_move`'s choice (two differences, both "searched
instead of answered", are listed after this list):

- the claim rules of `_dtz_of_played_move` are applied from the python-chess
  board on top of Fathom's numbers: a mating move is distance 1; a move into
  stalemate, insufficient material, threefold repetition (from the board's move
  stack, which Fathom never sees) or, with `Syzygy50MoveRule` on, a run-out
  50-move counter is distance 0;
- moves are ranked with the repo's own `_rule50_dtz_key` / `_plain_dtz_key`
  over python-chess's legal-move order, first best wins. Fathom's rank and
  score are not used at all, so the tie-break inside a bucket is the repo's
  (shortest win / longest loss, then move order), not Fathom's;
- the reported WDL is `try_tb_root_move`'s (`_wdl_adjusted_for_rule50`).

Two differences from `try_tb_root_move`, both "python-chess answered, the
engine now searches"; neither changes a move that is answered:

1. More men than the loaded tables: python-chess sometimes resolved such a
   root through a capture into the tables, the C prober never answers above
   `largest()`. Example: `8/8/8/2br4/3P2k1/4B3/2p5/4K3 w - - 95 40` (7 men)
   with 6-man tables.
2. WDL table present, DTZ table absent: python-chess could still rank the
   root when every child resolved by zeroing moves; `probe_root` needs the
   DTZ table. Example: `7n/8/4k3/8/6P1/5p1K/1p6/8 w - - 5 40` with only the
   first directory of the production pair. The engine warns at load when WDL
   tables lack DTZ.

### Eager validation at load

Fathom initialises a table on the first probe of its material. A corrupt
file therefore used to fail on move 60 instead of at `setoption`; one random
tail overwrite of `KQvKR.rtbw` made `setup_pairs` malloc a garbage size and
write through NULL (SIGSEGV, forfeit). Now:

- `tb_root.load` initialises EVERY registered table's WDL and DTZ file
  (`_tb_ext.validate_table`, the same `init_table` the first probe would run).
  Cost on the production pair (510 WDL + 510 DTZ): 0.16 s with the files in
  page cache, 0.01 s for `init` itself; it reads only each file's header and
  symbol table. A failure refuses the path, naming the file.
- A small marked local patch to the vendored `tbprobe.c` (see
  `fathom/VENDORED.md`, "Local patch") makes `init_table` fail cleanly
  instead of crashing for the header-level corruptions: allocation failure,
  an implausible header, a symbol table that indexes past itself or loops,
  and index/size/data sections that do not fit in the file. A local fuzz
  (300 corrupted copies of smoke-set tables: random tails, header bytes,
  scattered bytes) crashed validation 0 times (237 refused, 63 accepted:
  those corruptions sit in compressed data, which no header check sees).
- Table files present in the directories that Fathom did not register (it
  drops a file whose size is not 16 mod 64 with only an `info string
  Incomplete tablebase file` on stdout) also refuse the path, named.
- **What is still possible:** a file whose corruption is in its compressed
  data passes validation and can still crash or mis-answer at probe time.
  Every Syzygy prober shares this. `deepfin --selfcheck
  --syzygy <path> --syzygy-require-checksums` hashes every file against a
  checksum list (`chess_anti_engine/tb/checksums.py`; docs/TCEC_SETUP.md,
  section 7) and runs the full validation in a child process.

`try_tb_root_move` itself no longer has a production caller; it is the
reference `tests/test_uci_tb_root_c.py` compares against: per-move DTZ, chosen
move and WDL, both `Syzygy50MoveRule` settings, over ~1,700 roots on the
production pair (en passant, clocks 90-99, cursed/blessed results, mates and
stalemates in one, threefold-repetition children, both colours to move). The
test SKIPs without tables.

Engine wiring (`Engine._install_tablebase`): `SyzygyPath` loads the C table set
first, validating every table (below). A refusal (a component that is not a
readable directory, no table at all, a corrupt table, or a table file Fathom
did not load) frees the C tables, clears BOTH the root shortcut and the
python-chess leaf probe, and prints `info string error: SyzygyPath ... REFUSED
by the C tablebase prober (...)`, repeated before every `readyok` while that
path stays configured. A good path prints `info string Syzygy root prober: C Fathom ...`
with the table counts, and a warning when some WDL tables have no DTZ (the
split production pair given by halves). `Syzygy50MoveRule` rebuilds the handle
on the same tables; a new `SyzygyPath` re-initialises them. A handle built for
an earlier table set (another engine in the same process re-loaded it) is
never probed: the search drops the shortcut with one `info string root
tablebase shortcut disabled ...` and searches the root normally.
`Engine.close()` and clearing `SyzygyPath` free the C tables only while they
are still the set this engine loaded.
`go searchmoves` skips the shortcut, as before. An input the C prober refuses
(`ValueError`) is "no TB answer" and the search plays.

## UCI leaf probe (`uci/tb_leaf.py`)

The search's in-tree probe (leaf WDL overrides on every wired UCI search path,
and the root eval in `SearchWorker._ensure_root_eval_cached`, which also runs
under `go searchmoves`) is `tb_leaf.CLeafProbe`. It is a subclass of
`tablebase.SyzygyProbe` so every search path's annotation accepts it, but it
never calls `SyzygyProbe.__init__` (which opens python-chess) and defines every
method the search uses. `Engine._install_tablebase` builds it from the table
set `tb_root.load` returned; `SyzygyProbe` itself is unchanged and is still
what selfplay, the arena and the worker construct.

`CLeafProbe.apply` is `SyzygyProbe.apply` with one `probe_wdl_batch` call in
place of a `chess.Board(cb.fen())` + `Tablebase.probe_wdl` per leaf:

- **Eligibility is the caller's, unchanged.** With `indices` the leaves were
  pre-filtered: classic Gumbel by the C tree (`get_pending_tb_leaves`: no
  castling, at most `max_pieces` men, any halfmove clock), the walker pool and
  `puct_tb` (PUCV, pool, RPG) by their clock-0 rule. Without `indices` (the
  root eval) the probe filters castling and piece count itself, as before.
  `max_pieces` is the largest WDL table's man count, the value `SyzygyProbe`
  derived from its `wdl` keys (`LoadedTables.largest_wdl`).
- **Value, unchanged.** One-hot WIN/DRAW/LOSS logits; with
  `Syzygy50MoveRule` on (`cursed_as_draw`) a cursed win / blessed loss is a
  draw, with it off it is decisive. `solved_out` gets 1 / -1 / 2. The
  halfmove clock is read by neither prober: neither the old nor the new probe
  considers the 50-move counter or DTZ at a leaf.
- **Counting, unchanged.** `probes` counts every leaf handed to the prober,
  `hits` every answered one; `tbhits` in `info` is `hits` (plus the root
  shortcut's 1). The counters take a lock: the GIL is released inside the
  batch call and walker / pool threads share one probe.
- **"No TB answer"**, where the leaf keeps the net's WDL and counts as a probe:
  every input python-chess raised on (castling rights, a table missing from
  every directory with no capture cutoff, illegal positions, `KeyError` /
  `IndexError`) is a C `None` or refusal. On the inputs where python-chess
  answered and C does not (the list under "Where the two probers answer
  differently": the refused ep class and illegal positions, which a search
  below a legal root never produces, and positions above `largest()`, which no
  caller hands over because every filter uses `max_pieces`), the leaf is "no TB
  answer". There is no fallback to python-chess.
- **No cache.** Neither probe caches answers; `override_leaf_values` and the
  walker pool write into a copy, classic Gumbel into the evaluator's fresh
  output (section 6 of `docs/search_and_encode_paths.md`), all unchanged.

Lifetime: the probe carries `LoadedTables.c_generation` and passes it to
`probe_wdl_batch`. After any re-init or free of the C set (another engine in
the process loading a different path, `SyzygyPath` cleared, `quit`), a batch
from that probe -- on any pool thread, including a worker abandoned after a
device fault that is still inside a call -- probes nothing and returns `None`;
the probe logs one warning and its leaves get no answer. `SearchWorker.run`
drops a stale probe before the next search, resets the tree (it was grown with
the old set's values) and prints `info string leaf tablebase probing disabled
(...)`; `setoption name SyzygyPath` installs a fresh one. `setoption` still
waits for the search to finish, as before.

Evidence (`tests/test_uci_tb_leaf_c.py`):

- Differential, production pair: 5,247 leaves (2- to 7-man; 181 with castling
  rights, 32 with an en-passant square, 845 at halfmove 0, 1,091 at 50-99,
  2,033 at 100+, 90 checkmates/stalemates, 24 cursed wins / blessed losses,
  1,010 7-man), the root-eval call and the pre-filtered leaf call, both
  `Syzygy50MoveRule` settings: rows written, solved statuses, hits and probes
  identical. 7-man leaves handed over as if pre-filtered (no caller does) are
  the only difference: python-chess answered 666 of them through a capture into
  the 6-man tables, C answers none. An incomplete set (KQvKR, KQvK, KNvK
  only) agrees on every leaf, including which ones get no answer.
- Search identity, production pair: a fixed-seed CPU search (classic Gumbel
  and the 1-device multi-GPU PUCT pool, both rule settings) from 7- and 8-man
  roots and `searchmoves` roots gives the same bestmove, score, PV, nodes,
  `tbhits` and root visit vector with either probe, and the same as `main`.
- Throughput, production pair, 4,000 5-6-man leaves in batches of 32 through
  `apply`, pages warm: python-chess 7,260 leaves/s on one thread and 6,835 on
  four; C 412,977 and 339,844 (57x and 50x). The raw `probe_wdl_batch` call
  is about 1.0 M leaves/s against 0.54 M for a `probe_wdl` loop.

## Follow-ups (not in this PR)

Each switch is its own PR. Each one must prove bit-identical outputs on a
frozen input set, because a difference at a selfplay leaf is a target change.

1. **`SyzygyProbe.apply` leaf WDL for selfplay / arena** (`tablebase.py`).
   Done for the UCI engine (`CLeafProbe`, section above, root eval included);
   selfplay and the arena still build `chess.Board(cb.fen())` per eligible
   leaf. Switching them is a selfplay-target question: experiment record and
   identical override rows on a frozen leaf set first.
2. ~~**UCI root DTZ shortcut**~~: done (section above).
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
