# Fixed D-lite paired arena: durable resume contract

This is an opt-in route for the preregistered 576-opening, 1152-game paired
arena. The ordinary arena and its permissive recovery behavior stay unchanged.
The campaign uses one opening bank and one fixed-N result; no SPRT or
result-dependent stop is allowed. Each attempt uses internal
`--max-seconds 3200` and an external `timeout -k 10s 3300s` around the
**entire** process, including source checks, checkpoint loads, compilation,
play, and final readback. Thus
an interrupted attempt can lose at most 55 minutes; finished color-swapped
pairs survive each attempt.

Before the first attempt, freeze a Syzygy WDL/DTZ **metadata** inventory. The
receipt supplied as `--tablebase-catalog` has schema
`arena_syzygy_metadata_inventory_v1`, status
`PASS_METADATA_INVENTORY_ONLY`, the exact resolved ordered `roots`, and every
directly contained `.rtbw`/`.rtbz` file as an entry
`{path,bytes,mtime_ns}`. The source seal hashes the inventory receipt. Each
attempt compares exact names, sizes and mtimes and opens the strict six-man
WDL/DTZ, rule50-aware tablebase; missing required materials fail at probes.
This does **not** qualify tablebase file bytes. Prior factorial arena reviews
likewise authenticated metadata and actual terminal probes, not all tablebase
contents. A same-size content change with a restored mtime is outside the
per-attempt identity check. A separate full-byte audit can strengthen this
contract later without rehashing 235 GB on every attempt.
`scripts/prepare_arena_tb_inventory.py --syzygy <ordered-path> --output
<inventory.json>` creates this receipt from directory metadata only and
prints its whole-file SHA256. Inspect its roots, counts and file list before
freezing the arena seal.

From a clean, committed source checkout with its local native extensions
built, freeze the literal arena argument list (excluding only
`--durable-seal` and `--durable-seal-sha256`) as a JSON array of strings. It
must include explicit `--candidate`, `--reference`, `--openings-fen`,
`--games-out`, `--pgn-out`, `--pair-receipts-dir`, `--resume`, `--mode
matched_sims`, `--games 1152`, `--syzygy`, `--syzygy-max-pieces 6`,
`--max-seconds 3200`, and every search, compile, device, concurrency and
evaluator setting. Run `scripts/prepare_arena_durable_seal.py` with that
`--argv-json`, both checkpoint paths, the FEN bank, actual production config
path, frozen tablebase inventory, and Syzygy path. It writes a new fsynced
manifest and prints its SHA256; it refuses to overwrite an existing seal.
The manifest binds the exact argument list, tracked Git commit and clean
checkout, interpreter/numeric runtime, actual model/opening/config bytes,
and local board-encoding/features/MCTS native extension bytes. Its tablebase
field is only a hash of the metadata inventory receipt.

Launch `scripts/arena_standard.py` with the frozen argument list plus
`--durable-seal <path> --durable-seal-sha256 <printed-sha>` under the external
3300-second watchdog. Reuse the same JSONL, PGN, receipt directory and seal
on every attempt. The source gate runs before opening any result file or
loading either model. If model bytes change at the same path, a run setting
changes, or a catalog entry changes, the attempt refuses to resume. The
seal path and expected SHA are the only arguments omitted from the frozen
argument list, avoiding a self-referential manifest.

For each finished game, the PGN bytes are flushed and fsynced first; the
JSONL game row containing the PGN byte-span digest is flushed and fsynced
second. After both colorings, a small pair receipt names both exact JSONL
row and PGN spans and is atomically replaced and directory-fsynced. A crash
after the second JSONL commit but before receipt publication is recovered by
validating the two durable rows and PGN spans and publishing the missing
receipt. A one-coloring orphan is discarded and replayed in both colors.
If a crash lands between PGN and JSONL fsyncs, the append-only raw PGN can
also contain an uncommitted game with no JSONL row. Use the receipt-linked
PGN spans or the JSONL complete-pair union for analysis, not every raw PGN
record in the append-only file.
On each resume and final readback, every retained complete pair must match
its receipt. The final arena score and pentanomial confidence interval use
the full union of retained pairs, never a mean of per-attempt ratings. A
finished 1152-game campaign has exactly 576 pair receipts.

The CPU regression in `tests/test_arena_durable.py` covers a sealed pair,
an actual child-process SIGKILL after an orphan/PGN-only gap, a
crash-truncated JSONL tail, retained receipt byte/mtime identity, resumed
union scoring, missing receipt recovery, changed PGN/receipt refusal,
changed checkpoint bytes at the same path, and the opt-in `run_arena`
callback/resume wiring with mocked models and play. It does not prove GPU
numerical equivalence across process
restarts; the frozen source/profile plus a bounded CUDA warmup must be
reviewed before the arena campaign receives strength credit.
