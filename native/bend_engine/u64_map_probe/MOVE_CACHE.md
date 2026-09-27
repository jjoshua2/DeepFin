# Bounded structural legal-move cache

`MoveCache` is a consumer of BoardIndex/PositionIndex and the existing native
`Chess.legal_moves` generator. It reuses a position's immutable move list; it
never stores neural outputs, draw adjudications, search visits or policy scores.
No existing search frontend is switched to it by this change.

## API

`new(bits)` returns an empty cache or None using the index's 1..16 exponent
contract. `legal_moves(tables, cache, board)` returns a Result owning the attack
tables and cache, plus the legal move list and one of these access tags:

| Access | Work performed |
| --- | --- |
| Hit | Returns the stored list, without calling the legal generator |
| Filled | Generates the list and stores it under the new structural ID |
| Bypassed | Generates normally because the record limit prevents admission |

An empty list is a valid cached result, not a missing entry. Checkmate and
stalemate therefore both support hits, but the list does not distinguish their
outcomes: callers must still check the current board and history. A full cache
never returns an invented empty list or evicts another entry; existing hits
remain usable and uncached positions continue to receive normal legal moves.
Move ordering is preserved, including castling, en-passant and promotion flags.

The cache owns its index and list-slot array together. Use only this public API
on a valid owner, the correct attack tables, and valid orthodox Chess.Board
values. Do not reconstruct raw storage, modify an extracted index, or mix caches
with different rule implementations. It is single-owner, with no concurrent
mutation, reset, eviction or recoverable native allocation-failure transaction.
The index publishes no partially prepared cache owner to another caller.

Position matching retains full identity checks even if every supplied hash
collides. BoardIndex canonicalizes unavailable en passant and masks castling
metadata; pinned EP remains distinct under the existing conservative rule.
Game clocks and repetition history stay with the caller and are not overwritten
by a cache hit. Legal moves alone do not authorize continuing after an automatic
draw, omitting a claim action, or sharing history/model-sensitive evaluations.

Storage adds one list slot per identity record and the retained move-list data,
on top of PositionIndex's arrays. At bits=16 the record limit is 32,768; this is
not a measured byte/RSS budget. Full-population tests of the underlying maps do
not establish full-population memory behavior for these additional list payloads.
There is no throughput or playing-strength claim and no default/live setting.

## Qualification

```sh
python -m pytest tests/test_bend_map_move_cache.py
python -m native.bend_engine.u64_map_probe.move_cache \
  --compiler-root build/bend_toolchain/sources/aaeb9bc91ff0ff0b3f58dba6a9744c6607e167ae \
  --bun bun --cc clang-18 --output artifacts/NEW-move-cache --chess
```

The native caller parses position commands and applies their legal transitions
through the unchanged Bend code. The verifier separately builds the checkout's
CBoard oracle: it validates each move, special flag and resulting child board.
A dictionary of complete structural identities independently predicts filled,
hit and bypassed outcomes. Side-to-move/pawn square geometry implements the
reference EP rule without copying the adapter's bit-shift expressions.

The diagnostic caller ALSO runs the uncached Bend generator for each request
and checks exact list ordering. That extra reference call is test infrastructure,
not part of MoveCache and not a timing measurement. Access tags describe routes;
this harness does not measure wall-time savings or count all process-wide calls.

Standalone cases include transposed move orders and a knight cycle, different
history/clock contexts, unavailable/capturable/pinned EP, special moves, empty
lists, interleaved hits and full-cache fallback. The --chess option adds all
observations from the unchanged fresh CBoard corpus in bounded cache lifetimes.
Repetition of first/last records checks reuse. These are generated legal walks,
not production cache-hit-rate measurements.

Four existing build modes run all cases. A separate UBSan control replaces only
BoardIndex's fingerprint with a constant and must pass the same oracle. Three
bad-cache variants must compile and run normally before failing comparison:
lost stored move lists, empty results on saturation, and reading the wrong slot.
Four invalid inputs have exact exit/diagnostic checks. Old rule, identity, scale,
ownership and registry suites remain in the persistent workflow.

Use a fresh output directory. Reports retain source/build identities, input/output
hashes, case routes, raw output, and expected negative diagnostics. Local checks
passed 27 Python cases with global conftest disabled and all 44 standalone cases
in each original mode plus the forced-collision mode; all three mutations were
rejected in individually completed commands. Outer tool limits interrupted the
combined local harness, so those invocations are not end-to-end passes. The full
entry point, four invalid inputs, fresh corpus and locked static gates are checked
separately by hosted CI; actual status is recorded on #876. Self-review only.
