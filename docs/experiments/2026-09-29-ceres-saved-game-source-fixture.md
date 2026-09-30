# Tracked Ceres saved-game source fixture, September 29, 2026

The Ceres full-prefix pilots use a pinned external actor/writer closure. This
first tracked producer slice gives the repository an executable CPU publication
and readback boundary for one complete saved natural game. It builds on the
qualified stored-feed helper in PR #935. The CLI takes a source ZIP path, its
expected physical SHA-256, a game ID and a fresh output root; no path or model
identity is hardcoded in the producer.

`load_saved_natural_game` hashes the bounded source ZIP, selects one whole-game
row span, reads its seven saved arrays and metadata, and validates the selected
game against main's board encoder. This checks all 175 stored feature planes,
all 137 Ceres feed bytes, source/history/row identities, legal compact and
Leela maps, played moves, natural terminal and result. `write_saved_game`
creates a fresh Zarr ZIP, performs a full physical and semantic readback before
publication, links the verified file into a previously absent final name, and
reopens that name for a second full readback. The retained receipt pins source,
model and output archive identities. The public `read_saved_game_archive` can
repeat the postpublication check independently of the writer call; the separate
`scripts/ceres_saved_game_readback.py --output ROOT` entrypoint did so on the
fresh saved-game output and reported 47 verified rows.

The actual saved32 source ZIP was
`/tmp/ceres-sustained-byteproof-integrated-abba-v2-20260926/run01/arm_00_A/bank/shard_000000.zarr.zip`
(1,530,905 bytes, SHA-256
`09bd17dd73a7adaefc9c2fbde289174eceba365c835b50cd9a069f4c15d94e67`).
Its game ID 3 contains 47 completed rows. A fresh invocation from the tracked
CLI produced `/tmp/ceres-tracked-saved-game-fixture-run02-20260929/game.zarr.zip`
with physical SHA-256
`a3f3b217a35279e2c58ed445fe566e62045906d114e9190d88e199646066c5fe`.
A prior invocation through the same semantic path produced a different physical
ZIP SHA-256, `7fd4ee010392f31487d526a16fd6b83dcb780b139da90163cad57e1a1606e34d`;
independent public readback found all seven arrays and row/game metadata equal.
Game ID 24 in the same source ZIP also passed at 56 rows, exercising nonzero
source row and legal-offset rebasing; its fresh output archive SHA-256 was
`5c3f5077e2331903b46fbf77cf8f82564118b421940db2190626f595060ddf81`.

The fixture is explicitly unqualified and grants zero generated-row, corpus or
throughput credit. It does not run Ceres inference, decide Syzygy outcomes,
exercise the rolling game scheduler or prove the one-use writer speedup.
Natural-game replay only is accepted in this tranche. The next producer change
can add the actual root actor and strict outcome gate against this tracked
archive/readback contract, then compare a complete fresh-root pilot to the
frozen external source. The existing full-prefix packet and live checkout were
not changed.
