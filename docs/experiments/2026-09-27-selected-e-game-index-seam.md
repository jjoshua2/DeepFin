# Selected-E complete-game index seam (NO-LAUNCH)

This packet adds `scripts/selected_e_game_index.py`, a CPU-only **small-fixture**
reader and deterministic complete-game sampler. It has no CLI or registered-E
path. The synthetic tests read only identity columns in temporary Zarr v2
fixtures. No E payload was scanned, no E row was admitted, and no throughput was
measured. The sample manifest's `status` is always `NO-LAUNCH`.

## Input and first seal

The caller supplies a `FixtureClosure` before any open: qualified-receipt and
manifests-roster SHA-256 pins; an ordered, closed shard roster; each shard's
cohort index (0–34), cohort-manifest digest, shard ordinal/name/row count,
canonical resolved base parent, and SHA-256 of the decoded `game_id` and
`has_game_id` columns; and the SHA-256 of the canonical closure as the
`first_sealed_identity_sha256`. The function checks that first seal *before*
opening a path. A self-asserted pin does **not** authenticate E: a trusted
upstream receipt must establish every field and the immutable fixture root.
Future E integration must bind the base-seal/overlay roster and the row-count
pins to this exact closure independently.

The fixture root must contain exactly the listed shard directories. Every
directory component is opened with `O_NOFOLLOW`; listed members are checked
with `stat(..., follow_symlinks=False)`. Symlinks, FIFOs, other special files,
unlisted files, missing chunks, non-Zarr-v2 metadata, unsupported codecs,
partial game IDs, and changed decoded hashes refuse. The only decoded arrays
are 1D little-endian int64 `game_id` and bool `has_game_id`; Blosc-zstd chunk
headers are checked against the declared decoded size before decoding. Limits:
128 shards, 50,000 rows, 50,000 games, 160,000 decoded bytes per column,
192 KiB per member, 4 KiB per `.zarray`, 512 chunks per column, 4,096 chunks
total and 64 MiB of aggregate reads. These limits intentionally make a
registered 7,108-shard E scan impossible through this interface. A future
large scan needs a separate reviewed tool and authority.

## Whole-game selection

Game identity is `(resolved_base_parent, game_id)` across every shard and
cohort in the closure. The numeric ID alone is insufficient; observed E IDs
are unsorted and games can cross shard boundaries. The index maps every game
to *all* `(shard-roster-index, row-offset)` pairs and checks that every fixture
row appears exactly once. It sorts games by SHA-256 of a domain tag, sampling
seed digest, and game key. It selects one deterministic game per required
cohort, then solves the remaining exact row total by a bitset subset-sum over
that frozen order. If the anchors exceed the target or no exact remainder is
found, it refuses. The first-ranked anchors are fixed; a refusal does not prove
that no other exact game subset exists. It never trims a game to manufacture
12,288 rows. The output orders row UIDs by roster index and offset and includes the closure
pins, source parent/game ID, all selected row offsets, selected-game count,
seed and algorithm, covered cohorts, and a canonical sample digest.

The current index has **no opening, phase, or legal-count data**. Thus it
cannot satisfy the frozen screen's stratum requirement by itself. A later
adapter needs authenticated per-row stratum metadata, a frozen assignment
rule at game level, and exact-input/context uniqueness and objective-mask
checks. It must use this sample only after first-sealed source authentication,
selected-shard byte verification, and independent row/payload qualification;
it must not infer those properties from game membership.

## Full E cost and authority

Historical receipts report 35 cohorts, 7,108 Zarr shards, 58,090,688 rows,
and 299,144 games, but no game-to-row offsets. Reading `game_id` (8 bytes)
and `has_game_id` (1 byte) across all rows would decode **522,816,192 bytes
(~499 MiB)** before Python arrays, hash maps, decompressor scratch, and
filesystem metadata. Two columns per shard imply at least 14,216 metadata
opens plus their chunk opens. If each column has one chunk, roughly 28,432
opens at 1–10 ms each imply about 28–284 seconds of open latency alone;
extra chunks, HDD seeks and decompression add time. This is a conditional
planning range, not measured E latency. The actual compressed-byte and seek
totals require a read-only metadata census. A compact `(source parent, game ID,
count)` index has about 299,144 records, with a lower bound of several MiB
for integers alone and potentially tens of MiB or more for Python maps. After
game selection, recovering offsets would require a second identity pass over
selected shards unless the first pass persists a larger per-row index. These
are planning estimates, **not** elapsed-time or throughput measurements.

A reviewed full-scan plan must pin the exact E receipts and manifest roster,
base-seal and overlay identities, shard count and row total; establish
read-only storage scope, byte/IO/CPU/RAM/wall ceilings and available host
capacity; independently verify column shapes/codecs and whole-game closure;
preserve the 35-cohort and strata protocol; and publish the first seal before
sampling or target/timing inspection. The historical E admission remains
storage-only with a WDL-only tablebase check. Its 299,144-game receipt does
not upgrade that source qualification.

Synthetic verification: `python3 -m pytest -q tests/test_selected_e_game_index.py`
checks unsorted/cross-shard games, source-scoped duplicate numeric IDs, exact
12,288 rows across 35 cohorts, deterministic output, missing IDs/rows,
inexact game totals, nonbinary bool bytes, first-seal/hash changes, symlink,
FIFO, size and chunk-count refusal.
