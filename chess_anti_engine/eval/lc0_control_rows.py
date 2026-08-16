"""Row identity and frozen row sets for the lc0 positive control.

⚑⚑ WHY THE ROW ID IS A CONTENT DIGEST AND NOT A NAME.

The held-out split for this arm is "the last 6 hourly tars by wall-clock". The
obvious purity check is "the train tar list and the held-out tar list are
disjoint", and it is worthless: it can only fail if someone types the same
filename twice. A gate that cannot fail is not a gate, and
`memory/exposure_recency_dominates_heldout_ce.md` records what a held-out set
that overlaps its train window actually costs — it measures forgetting and
recency, and reads as generalisation.

So the id is derived from THE ROW'S OWN CONTENT — its 175-plane input and its
1858-wide lc0 policy target — and never from the tar it arrived in, the shard
index, or the split it was assigned to. Two rows built from the same lc0 v6
record produce the same id no matter which conversion run produced them, so an
overlap is detectable; two rows from different records do not, so the check is
not trivially satisfied.

⚑ Two candidate ids were considered and one was rejected. `game_id` +
`ply_index` are stored in the shard and are cheap, but `game_id` is
`enumerate()`'s index over the games of ONE conversion invocation. Converting
train and held-out separately restarts it at 0, so those ids are disjoint BY
CONSTRUCTION and the assertion would pass on a set that is 100% contaminated.
That is the failure this module exists to avoid, not a cheaper way to reach it.

The RECORD id covers `x` and `policy_target` — the input and the label, i.e.
everything the yardstick reads. `x` alone would merge a transposition reached
with identical history; the policy floats come from a specific lc0 search and
break that tie. The claim that the pair discriminates is not left as an
argument: `frozen_row_set` reports `duplicate_ids`, the number of ids that
repeat WITHIN one split, and a non-zero count there is the id failing to
separate distinct records.

⚑⚑ BUT PURITY IS NOT A RECORD-IDENTITY QUESTION, IT IS AN EXPOSURE QUESTION,
AND THE RECORD ID IS THE WRONG INSTRUMENT FOR IT. "Did the net already see
this position" is answered by `x` ALONE. The same position reached in two
games gets two different lc0 visit distributions — search noise, a different
net generation — so the pair digest calls them distinct while the model
trained on exactly that input. Measured on the shipped smoke split:

    frozen held-out rows                          100,000
    INTERSECT by (x, policy_target) record id           0   <- reads PURE
    INTERSECT by x-only INPUT id                      450   <- actually exposed
    duplicate RECORD ids inside the held-out set         3
    duplicate INPUT ids inside the held-out set      3,153

450 exposed inputs at a train side of 48,360 rows — 0.06% of the full corpus.
The arm's entire readout is held-out generalisation, so an exposure gate that
under-reports biases the headline in the flattering direction.

So there are TWO ids and they answer different questions. `input_ids` is the
GATE (`PurityResult.is_pure`); `row_ids` remains the record identity used to
address a specific row when scoring. Both are reported, always, because the
gap between them is itself the diagnostic.

Cost, measured: ~31k rows/s single-threaded (blake2b-128 over 22.5 KB/row), so
~0.8 h for the full 87M-position corpus — against ~105 h for the conversion
that produces it.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays

# 128 bits. Over the full corpus (~8.7e7 rows) the birthday collision
# probability is ~1.4e-23, i.e. far below every other error source here.
ROW_ID_BYTES = 16

# Fields the digest covers, in this order. Changing this list changes every id
# and INVALIDATES every frozen set built with the old one, which is why the
# version below is stamped into the artifact and checked on load.
ROW_ID_FIELDS = ("x", "policy_target")
# ⚑ The EXPOSURE id. `x` alone, because "did the net already train on this
# input" does not depend on which label came with it.
INPUT_ID_FIELDS = ("x",)
ROW_ID_VERSION = "lc0_control_row_id_v2_x_policy_plus_x_only_blake2b128"

# ⚑ Ids live in numpy as fixed-width ASCII HEX (`S32`), never as the raw
# 16-byte digest. numpy's `S` dtype strips TRAILING NUL bytes, so a raw-digest
# array would render a digest ending in `\x00` as a 15-byte value and its
# `.hex()` would be 30 characters — silently unequal to the same id produced by
# `_content_ids`. Hex has no NUL, so the round trip is exact. The cost is 2x the
# bytes; the alternative is a 1-in-256 chance per row of a truncated id inside
# the purity gate, which is precisely this codebase's signature defect.
ID_DTYPE = np.dtype(f"S{2 * ROW_ID_BYTES}")

# ⚑ The two-id fast path in `_digest_pair` streams `x` into ONE blake2b, takes
# the INPUT id from it, then `.copy()`s the state and appends `policy_target`
# for the RECORD id. That is bit-identical to hashing `x` and `x || policy`
# separately ONLY while the input fields are a PREFIX of the record fields, so
# the relation is asserted at import rather than assumed by a reader. Reordering
# ROW_ID_FIELDS would otherwise change the ids while every test still passed.
if tuple(ROW_ID_FIELDS[: len(INPUT_ID_FIELDS)]) != tuple(INPUT_ID_FIELDS):
    raise RuntimeError(
        f"INPUT_ID_FIELDS {INPUT_ID_FIELDS} is no longer a prefix of "
        f"ROW_ID_FIELDS {ROW_ID_FIELDS}; the streaming digest in _digest_pair "
        "would no longer reproduce _content_ids.",
    )


def _content_ids(arrs: dict[str, np.ndarray], fields: Sequence[str]) -> list[str]:
    columns = [np.ascontiguousarray(arrs[name]) for name in fields]
    n = columns[0].shape[0]
    for name, col in zip(fields, columns, strict=True):
        if col.shape[0] != n:
            raise ValueError(
                f"shard column {name!r} has {col.shape[0]} rows, expected {n}",
            )
    return [
        hashlib.blake2b(
            b"".join(col[i].tobytes() for col in columns), digest_size=ROW_ID_BYTES,
        ).hexdigest()
        for i in range(n)
    ]


def row_ids(arrs: dict[str, np.ndarray]) -> list[str]:
    """RECORD ids — ``(x, policy_target)``. Addresses a row; not the gate."""
    return _content_ids(arrs, ROW_ID_FIELDS)


def input_ids(arrs: dict[str, np.ndarray]) -> list[str]:
    """EXPOSURE ids — ``x`` alone. This is what the purity gate counts.

    ⚑ Deliberately merges transpositions and re-searches of the same position.
    That merge is the point: they are the same input, and the model that
    trained on one has seen the other.
    """
    return _content_ids(arrs, INPUT_ID_FIELDS)


def _digest_pair(arrs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """``(record_ids, input_ids)`` as ``S32`` hex arrays, in one pass over ``x``.

    Equivalent to ``(_content_ids(arrs, ROW_ID_FIELDS),
    _content_ids(arrs, INPUT_ID_FIELDS))`` — pinned by
    ``test_digest_pair_matches_the_generic_content_id`` — and 1.9x cheaper,
    because `x` is 22,400 bytes of the 26,116 hashed per row and the generic
    path hashes it twice. On the 78.5M-row corpus that is the difference
    between a 3.1 h and a 1.7 h single-threaded scan.
    """
    columns = [np.ascontiguousarray(arrs[name]) for name in ROW_ID_FIELDS]
    n = columns[0].shape[0]
    for name, col in zip(ROW_ID_FIELDS, columns, strict=True):
        if col.shape[0] != n:
            raise ValueError(
                f"shard column {name!r} has {col.shape[0]} rows, expected {n}",
            )
    prefix = columns[: len(INPUT_ID_FIELDS)]
    suffix = columns[len(INPUT_ID_FIELDS):]
    record = np.empty(n, dtype=ID_DTYPE)
    inputs = np.empty(n, dtype=ID_DTYPE)
    for i in range(n):
        hasher = hashlib.blake2b(
            b"".join(col[i].tobytes() for col in prefix), digest_size=ROW_ID_BYTES,
        )
        inputs[i] = hasher.hexdigest()
        full = hasher.copy()
        for col in suffix:
            full.update(col[i].tobytes())
        record[i] = full.hexdigest()
    return record, inputs


def iter_shard_arrays(shard_dirs: Sequence[Path]) -> Iterator[tuple[Path, dict[str, np.ndarray]]]:
    """``(path, arrays)`` for every shard under every directory, in order.

    Directories are visited in the order given and shards within a directory in
    sorted order, so the traversal is deterministic and two runs over the same
    inputs produce the same id sequence.
    """
    for shard_dir in shard_dirs:
        for path in sorted(iter_shard_paths(Path(shard_dir))):
            arrs, _meta = load_shard_arrays(path)
            yield path, arrs


@dataclass
class SplitIds:
    """Every row id in a split, plus what it took to say that."""

    ids: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    rows: int = 0

    @property
    def unique_ids(self) -> int:
        return len(set(self.ids))

    @property
    def unique_inputs(self) -> int:
        return len(set(self.inputs))

    @property
    def duplicate_ids(self) -> int:
        """Rows whose RECORD id repeats within this split.

        ⚑ Report this, always. It is the id's own discrimination check: a
        content digest that collapses distinct records would inflate it, and
        the purity assertion downstream would then be comparing the wrong
        things while still reading "0 intersection".
        """
        return self.rows - self.unique_ids

    @property
    def duplicate_inputs(self) -> int:
        """Rows whose INPUT repeats within this split.

        ⚑ Always >= ``duplicate_ids``, and on real lc0 data by three orders of
        magnitude (3 vs 3,153 on the shipped smoke split). Reporting only the
        first invites reading "the id discriminates" as "the split has no
        repeated positions", which is a different and false claim.
        """
        return self.rows - self.unique_inputs


def collect_split_ids(shard_dirs: Sequence[Path]) -> SplitIds:
    """Record ids AND exposure (input) ids for every row under ``shard_dirs``."""
    split = SplitIds()
    for path, arrs in iter_shard_arrays(shard_dirs):
        ids = row_ids(arrs)
        split.ids.extend(ids)
        split.inputs.extend(input_ids(arrs))
        split.sources.append(str(path.name))
        split.rows += len(ids)
    return split


def _stratified_sample(
    per_source: list[list[tuple[str, str]]], *, sample: int, seed: int,
) -> list[tuple[str, str]]:
    """Take ``sample`` ids spread proportionally across sources.

    ⚑ A flat random draw over the pooled ids would still be *time-disjoint*
    from train, so it would not break purity — but it would let one hour
    dominate the frozen set by luck, and the prereg's whole reason for taking
    SIX hours is that a single hour is one net generation's worth of a
    correlated stream. Proportional-by-source keeps every hour represented.
    """
    total = sum(len(chunk) for chunk in per_source)
    if sample >= total:
        return [row for chunk in per_source for row in chunk]
    rng = np.random.default_rng(seed)
    picked: list[tuple[str, str]] = []
    remaining = sample
    for index, chunk in enumerate(per_source):
        sources_left = len(per_source) - index
        want = remaining // sources_left if sources_left > 1 else remaining
        want = min(want, len(chunk))
        take = rng.choice(len(chunk), size=want, replace=False)
        picked.extend(chunk[int(i)] for i in sorted(take))
        remaining -= want
    return picked


def frozen_row_set(
    shard_dirs: Sequence[Path], *, sample: int, seed: int,
) -> dict[str, Any]:
    """Build the frozen held-out artifact (ids + everything needed to audit it).

    The returned mapping is what gets written to disk and sha256'd. It carries
    the id VERSION so a later run cannot silently compare ids built under a
    different digest definition, and the per-source row counts so a reader can
    see the six hours are all present without re-reading the shards.
    """
    per_source: list[list[tuple[str, str]]] = []
    source_names: list[str] = []
    source_rows: list[int] = []
    all_ids: list[str] = []
    all_inputs: list[str] = []
    for shard_dir in shard_dirs:
        split = collect_split_ids([shard_dir])
        per_source.append(list(zip(split.ids, split.inputs, strict=True)))
        source_names.append(Path(shard_dir).name)
        source_rows.append(split.rows)
        all_ids.extend(split.ids)
        all_inputs.extend(split.inputs)
    pool = SplitIds(ids=all_ids, inputs=all_inputs, rows=len(all_ids))
    frozen = _stratified_sample(per_source, sample=sample, seed=seed)
    frozen_ids = [row_id for row_id, _ in frozen]
    frozen_inputs = [input_id for _, input_id in frozen]
    return {
        "row_id_version": ROW_ID_VERSION,
        "row_id_fields": list(ROW_ID_FIELDS),
        "input_id_fields": list(INPUT_ID_FIELDS),
        "sample_seed": seed,
        "sample_requested": sample,
        "pool_rows": pool.rows,
        "pool_unique_ids": pool.unique_ids,
        "pool_duplicate_ids": pool.duplicate_ids,
        "pool_unique_inputs": pool.unique_inputs,
        "pool_duplicate_inputs": pool.duplicate_inputs,
        "sources": source_names,
        "source_rows": source_rows,
        "frozen_rows": len(frozen),
        "frozen_unique_ids": len(set(frozen_ids)),
        "frozen_unique_inputs": len(set(frozen_inputs)),
        "row_ids": frozen_ids,
        "input_ids": frozen_inputs,
    }


def artifact_sha256(payload: dict[str, Any]) -> str:
    """sha256 of the artifact's canonical serialisation.

    Canonical = the exact bytes `write_frozen` puts on disk, so the recorded
    digest is a digest of the FILE and can be re-checked with `sha256sum`.
    """
    return hashlib.sha256(serialize_frozen(payload)).hexdigest()


def serialize_frozen(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def write_frozen(payload: dict[str, Any], path: Path) -> str:
    """Write the frozen set and return its sha256."""
    blob = serialize_frozen(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return hashlib.sha256(blob).hexdigest()


def load_frozen(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("row_id_version")
    if version != ROW_ID_VERSION:
        raise ValueError(
            f"frozen row set {path} was built with row id version {version!r}, "
            f"but this code builds {ROW_ID_VERSION!r}. The ids are not "
            "comparable; rebuild the frozen set or check out the code that "
            "made it. Silently proceeding would compare two different "
            "identities and report a clean purity check over nothing.",
        )
    if len(payload.get("input_ids", ())) != len(payload.get("row_ids", ())):
        raise ValueError(
            f"frozen row set {path} carries {len(payload.get('input_ids', ()))} "
            f"exposure (input) ids for {len(payload.get('row_ids', ()))} rows. "
            "The purity GATE is the x-only exposure count; a set without one "
            "can only be checked by record id, which under-reports exposure "
            "(measured 0 vs 450 on the shipped smoke split). Re-freeze.",
        )
    return payload


CACHE_FORMAT = "lc0_control_train_id_index_v1"


def train_shard_paths(train_shard_dirs: Sequence[Path]) -> list[Path]:
    """Every shard under ``train_shard_dirs``, in the traversal order used."""
    return [
        path
        for shard_dir in train_shard_dirs
        for path in sorted(iter_shard_paths(Path(shard_dir)))
    ]


@dataclass(frozen=True)
class CorpusFingerprint:
    """What the train corpus IS, reduced to one digest.

    ⚑ FAIL-CLOSED, and deliberately over-sensitive. The cached id set below is a
    pure function of the corpus, so reusing it is only sound while the corpus is
    byte-identical. The key therefore covers every file under every shard —
    relative path, size and mtime_ns — not the directory names, not the shard
    count, and not a "looks unchanged" heuristic. A re-converted shard, an
    appended directory, a truncated chunk and a touched file all change the key,
    and any mismatch rescans. The one thing this must never do is decide a stale
    set is *probably* fine: a purity gate reading a cache from a different
    corpus is a gate that cannot fail.
    """

    key: str
    shards: int
    files: int
    bytes: int


def corpus_fingerprint(train_shard_dirs: Sequence[Path]) -> CorpusFingerprint:
    """Digest the corpus's directory list AND every file's size + mtime."""
    hasher = hashlib.sha256()
    hasher.update(CACHE_FORMAT.encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(ROW_ID_VERSION.encode("utf-8"))
    hasher.update(b"\0")
    for shard_dir in train_shard_dirs:
        hasher.update(str(Path(shard_dir).resolve()).encode("utf-8"))
        hasher.update(b"\0")
    shards = 0
    files = 0
    total = 0
    for path in train_shard_paths(train_shard_dirs):
        shards += 1
        hasher.update(str(path.resolve()).encode("utf-8"))
        hasher.update(b"\0")
        entries: list[tuple[str, int, int]] = []
        for root, dirnames, filenames in os.walk(path):
            dirnames.sort()
            for name in sorted(filenames):
                full = Path(root) / name
                stat = full.stat()
                entries.append(
                    (str(full.relative_to(path)), int(stat.st_size), int(stat.st_mtime_ns)),
                )
        for rel, size, mtime_ns in sorted(entries):
            hasher.update(f"{rel}\0{size}\0{mtime_ns}\0".encode())
            files += 1
            total += size
    return CorpusFingerprint(
        key=hasher.hexdigest(), shards=shards, files=files, bytes=total,
    )


@dataclass(frozen=True)
class TrainIdIndex:
    """The train corpus's RECORD and INPUT ids, sorted and de-duplicated.

    This is the expensive half of the purity gate — 42.2 GB read and 2h40m of
    blake2b on the 122-directory corpus — and it is a pure function of that
    corpus, so it is cached to disk under ``CorpusFingerprint.key``. The point
    is not speed: it is that a re-verification costing 2h40m is the step that
    gets skipped under time pressure, and a gate that stops being run has
    already failed.
    """

    key: str
    rows: int
    record_ids: np.ndarray
    input_ids: np.ndarray
    from_cache: bool


class _UniqueAccumulator:
    """Collect id arrays and keep only the sorted unique ones, in bounded RAM."""

    def __init__(self, compact_rows: int = 8_000_000) -> None:
        self._unique: np.ndarray = np.empty(0, dtype=ID_DTYPE)
        self._pending: list[np.ndarray] = []
        self._pending_rows = 0
        self._compact_rows = int(compact_rows)

    def add(self, ids: np.ndarray) -> None:
        self._pending.append(np.asarray(ids, dtype=ID_DTYPE))
        self._pending_rows += int(ids.shape[0])
        if self._pending_rows >= self._compact_rows:
            self._compact()

    def _compact(self) -> None:
        if not self._pending:
            return
        self._unique = np.unique(np.concatenate([self._unique, *self._pending]))
        self._pending = []
        self._pending_rows = 0

    def result(self) -> np.ndarray:
        self._compact()
        return self._unique


def _scan_shard_ids(path_str: str) -> tuple[int, bytes, bytes]:
    """Worker body: ``(rows, record_hex_bytes, input_hex_bytes)`` for one shard.

    Returns raw buffers rather than arrays so the parallel path pickles 512 KB
    per shard instead of a numpy object graph.
    """
    arrs, _meta = load_shard_arrays(Path(path_str))
    record, inputs = _digest_pair(arrs)
    return int(record.shape[0]), record.tobytes(), inputs.tobytes()


def _iter_scanned_shards(
    paths: Sequence[Path], *, workers: int,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    if workers <= 1:
        for path in paths:
            rows, record, inputs = _scan_shard_ids(str(path))
            yield rows, np.frombuffer(record, dtype=ID_DTYPE), np.frombuffer(
                inputs, dtype=ID_DTYPE,
            )
        return
  # ⚑ "spawn", not the default fork. zarr's blosc decompressor keeps a thread
  # pool, and forking a process that already holds one is the classic
  # deadlock-after-fork. The result is order-independent (two de-duplicated
  # sets and a row COUNT), so `imap_unordered`-style completion order cannot
  # change the answer -- pinned by
  # `test_train_id_index_is_identical_in_parallel`.
    with ProcessPoolExecutor(
        max_workers=int(workers), mp_context=get_context("spawn"),
    ) as pool:
        for rows, record, inputs in pool.map(
            _scan_shard_ids, [str(p) for p in paths], chunksize=4,
        ):
            yield rows, np.frombuffer(record, dtype=ID_DTYPE), np.frombuffer(
                inputs, dtype=ID_DTYPE,
            )


def train_index_cache_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"train_ids_{key[:32]}.npz"


def load_train_id_index(cache_dir: Path, fingerprint: CorpusFingerprint) -> TrainIdIndex | None:
    """The cached index for this exact corpus, or ``None``. Never a near-miss."""
    path = train_index_cache_path(cache_dir, fingerprint.key)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as blob:
        if str(blob["cache_format"]) != CACHE_FORMAT:
            return None
        if str(blob["row_id_version"]) != ROW_ID_VERSION:
            return None
        if str(blob["key"]) != fingerprint.key:
  # Reachable when a digest collides with a file name, or when a cache
  # file is copied over another. The name is a hint; the stored key is
  # the check.
            return None
        return TrainIdIndex(
            key=fingerprint.key,
            rows=int(blob["rows"]),
            record_ids=np.asarray(blob["record_ids"], dtype=ID_DTYPE),
            input_ids=np.asarray(blob["input_ids"], dtype=ID_DTYPE),
            from_cache=True,
        )


def build_train_id_index(
    train_shard_dirs: Sequence[Path],
    *,
    cache_dir: Path | None = None,
    workers: int = 1,
    progress: Any = None,
) -> TrainIdIndex:
    """Every train id, from cache when the corpus is byte-identical.

    ``progress`` is called as ``progress(shards_done, shards_total, rows)`` so a
    2h40m scan can say where it is without this module importing a logger.
    """
    paths = train_shard_paths(train_shard_dirs)
    fingerprint = corpus_fingerprint(train_shard_dirs)
    if cache_dir is not None:
        cached = load_train_id_index(Path(cache_dir), fingerprint)
        if cached is not None:
            return cached
    records = _UniqueAccumulator()
    inputs = _UniqueAccumulator()
    rows = 0
    for done, (shard_rows, record_ids, input_ids_arr) in enumerate(
        _iter_scanned_shards(paths, workers=workers), start=1,
    ):
        rows += shard_rows
        records.add(record_ids)
        inputs.add(input_ids_arr)
        if progress is not None:
            progress(done, len(paths), rows)
    index = TrainIdIndex(
        key=fingerprint.key,
        rows=rows,
        record_ids=records.result(),
        input_ids=inputs.result(),
        from_cache=False,
    )
    if cache_dir is not None and rows > 0:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        path = train_index_cache_path(Path(cache_dir), fingerprint.key)
  # Written to a sibling and renamed: a cache half-written by an
  # interrupted 2h40m scan must not be loadable under the valid key.
  # `np.savez` appends `.npz` to a NAME that lacks it, so the temporary is
  # passed as an open file object instead.
        tmp = path.parent / (path.name + ".partial")
        with tmp.open("wb") as handle:
            np.savez(
                handle,
                cache_format=np.array(CACHE_FORMAT),
                row_id_version=np.array(ROW_ID_VERSION),
                key=np.array(fingerprint.key),
                rows=np.array(rows, dtype=np.int64),
                shards=np.array(fingerprint.shards, dtype=np.int64),
                record_ids=index.record_ids,
                input_ids=index.input_ids,
            )
        tmp.replace(path)
    return index


class EmptyTrainCorpus(ValueError):
    """The purity check found no training rows to compare against.

    ⚑ A "0 intersections" result over 0 scanned rows is not a clean split, it
    is an unperformed check — and it exits 0, prints PURE, and licenses a
    contaminated held-out set (Codex #6, reproduced verbatim by the reviewer).
    """


@dataclass(frozen=True)
class PurityResult:
    """Overlap between a frozen held-out set and a train corpus.

    ⚑ TWO instruments, deliberately. ``exposed_inputs`` is the GATE — the
    x-only count of held-out inputs the model would already have trained on.
    ``intersecting_ids`` is the record-level count and is reported for
    continuity; it under-reports exposure by construction, because the same
    position carrying a different lc0 policy target is a different record.
    """

    frozen_rows: int
    frozen_inputs: int
    train_rows: int
    train_unique_ids: int
    train_unique_inputs: int
    intersecting_ids: int
    exposed_inputs: int
    examples: tuple[str, ...]
  # ⚑ THE OPERANDS, not just their cardinalities. The 2026-08-16 run banked
  # `exposed_inputs: 5065` and five example hashes, so the subtraction that
  # repairs the split -- pure arithmetic on a set already computed -- could not
  # be performed without repeating a 2h40m, 42.2 GB scan. A count is a summary
  # of a set; keeping only the summary discards the thing the next step needs.
    exposed_input_ids: tuple[str, ...] = ()
    intersecting_row_ids: tuple[str, ...] = ()
    train_index_cached: bool = False

    @property
    def exposed_input_frac(self) -> float:
        if self.frozen_inputs <= 0:
            return 0.0
        return self.exposed_inputs / self.frozen_inputs

    @property
    def is_pure(self) -> bool:
        """Zero EXPOSURE. Record-id agreement is not sufficient."""
        return self.exposed_inputs == 0 and self.intersecting_ids == 0


def _members_of(wanted: Sequence[str], corpus: np.ndarray) -> tuple[str, ...]:
    """The distinct ``wanted`` ids that occur in the sorted ``corpus`` array."""
    values = list(wanted)
    too_long = [value for value in values if len(value) > ID_DTYPE.itemsize]
    if too_long:
  # `np.asarray(..., dtype="S32")` TRUNCATES silently, which would turn a
  # foreign id into a prefix match against a real one.
        raise ValueError(
            f"{len(too_long)} id(s) are longer than {ID_DTYPE.itemsize} "
            f"characters (first: {too_long[0]!r}); they are not "
            f"{ROW_ID_VERSION} ids.",
        )
    unique = np.unique(np.asarray(values, dtype=ID_DTYPE))
    if unique.size == 0 or corpus.size == 0:
        return ()
    hit = unique[np.isin(unique, corpus, assume_unique=True)]
    return tuple(value.decode("ascii") for value in hit.tolist())


def purity_against_train(
    frozen_ids: Sequence[str],
    train_shard_dirs: Sequence[Path],
    *,
    frozen_input_ids: Sequence[str],
    cache_dir: Path | None = None,
    workers: int = 1,
    progress: Any = None,
) -> PurityResult:
    """Count frozen rows the train corpus also contains, by record AND by input.

    Raises ``EmptyTrainCorpus`` when the train side turns out to hold no rows —
    a check that scanned nothing must not be able to report PURE.

    ``cache_dir`` reuses a previously banked id index, but only for a corpus
    whose ``corpus_fingerprint`` matches exactly; see ``CorpusFingerprint``.
    """
  # ⚑ No default on `frozen_input_ids`. An omitted exposure list would make
  # ``exposed_inputs`` 0 for every input and put the gate straight back where
  # F5 found it.
    index = build_train_id_index(
        train_shard_dirs, cache_dir=cache_dir, workers=workers, progress=progress,
    )
    if index.rows == 0:
        raise EmptyTrainCorpus(
            f"purity scanned {len(list(train_shard_dirs))} train directory(ies) "
            "and found ZERO rows. That is not a pure split, it is an "
            "unperformed check: with no train rows the intersection is 0 for "
            "any held-out set, including a 100% contaminated one. Check the "
            "--train-shards paths.",
        )
    hits = _members_of(frozen_ids, index.record_ids)
    exposed = _members_of(frozen_input_ids, index.input_ids)
    return PurityResult(
        frozen_rows=len(set(frozen_ids)),
        frozen_inputs=len(set(frozen_input_ids)),
        train_rows=index.rows,
        train_unique_ids=int(index.record_ids.size),
        train_unique_inputs=int(index.input_ids.size),
        intersecting_ids=len(hits),
        exposed_inputs=len(exposed),
        examples=tuple(sorted(set(hits) | set(exposed))[:5]),
        exposed_input_ids=exposed,
        intersecting_row_ids=hits,
        train_index_cached=index.from_cache,
    )


def exposed_rows(
    payload: dict[str, Any], exposed_input_ids: Sequence[str],
) -> list[dict[str, str]]:
    """The frozen rows carrying an exposed INPUT, in frozen order.

    ⚑ Rows, not inputs. 100,000 frozen rows carry 96,853 distinct inputs, so a
    list of exposed input hashes does not say which ROWS to drop — and the row
    is the unit `frozen_full.json`, every `.npz` score file and the paired
    comparison are all keyed on. Every id whose RECORD intersects train is
    necessarily in here too: a record digest is `blake2b(x || policy)`, so a
    record match implies an `x` match.
    """
    exposed = set(exposed_input_ids)
    return [
        {"row_id": row_id, "input_id": input_id}
        for row_id, input_id in zip(
            payload["row_ids"], payload["input_ids"], strict=True,
        )
        if input_id in exposed
    ]


def frozen_minus_exposed(
    payload: dict[str, Any], exposed_input_ids: Sequence[str],
) -> dict[str, Any]:
    """A new frozen payload with every EXPOSED-input row removed.

    Row ORDER and schema are preserved for the surviving rows, so the banked
    per-row score arrays can be restricted to it by masking rather than
    re-scored. The derived counts are recomputed from the surviving rows, and
    the provenance keys record what it was derived from — a trimmed set that
    could be mistaken for an original freeze would let the prereg's n=100,000
    constants be applied to a smaller set silently.
    """
    exposed = set(exposed_input_ids)
    kept = [
        (row_id, input_id)
        for row_id, input_id in zip(
            payload["row_ids"], payload["input_ids"], strict=True,
        )
        if input_id not in exposed
    ]
    kept_ids = [row_id for row_id, _ in kept]
    kept_inputs = [input_id for _, input_id in kept]
    trimmed = dict(payload)
    trimmed["row_ids"] = kept_ids
    trimmed["input_ids"] = kept_inputs
    trimmed["frozen_rows"] = len(kept_ids)
    trimmed["frozen_unique_ids"] = len(set(kept_ids))
    trimmed["frozen_unique_inputs"] = len(set(kept_inputs))
    trimmed["derived_from_rows"] = int(payload["frozen_rows"])
    trimmed["derived_by"] = "frozen_minus_exposed"
    trimmed["removed_rows"] = int(payload["frozen_rows"]) - len(kept_ids)
    trimmed["removed_inputs"] = len(exposed)
    return trimmed


@dataclass(frozen=True)
class ChanceLevel:
    """Chance top-1 accuracy over a set of variable-size legal-move sets."""

    rows: int
    mean_legal: float
    expected_inverse: float
    inverse_of_mean: float

    @property
    def jensen_ratio(self) -> float:
        """How far ``1/E[n]`` sits BELOW the true floor. Always >= 1."""
        if self.inverse_of_mean <= 0.0:
            return float("nan")
        return self.expected_inverse / self.inverse_of_mean


def chance_level(legal_counts: np.ndarray) -> ChanceLevel:
    """``E[1/n_legal]`` — the chance top-1 rate. NOT ``1/E[n_legal]``.

    ⚑ Jensen. A uniform-random guesser on a position with ``n`` legal moves is
    right with probability ``1/n``; its overall rate is the AVERAGE of those
    per-position probabilities, ``E[1/n]``. ``1/E[n]`` is the reciprocal of the
    average branching factor and is a strictly SMALLER number whenever ``n``
    varies at all — so using it puts the negative control's floor UNDER the
    real floor and a shuffled-label run can clear a gate it should fail. Both
    are returned, with their ratio, so the wrong one cannot be quoted by
    accident and the size of the error is on the record.
    """
    counts = np.asarray(legal_counts, dtype=np.float64).reshape(-1)
    positive = counts[counts > 0.0]
    if positive.size == 0:
        return ChanceLevel(rows=0, mean_legal=float("nan"),
                           expected_inverse=float("nan"), inverse_of_mean=float("nan"))
    mean_legal = float(positive.mean())
    return ChanceLevel(
        rows=int(positive.size),
        mean_legal=mean_legal,
        expected_inverse=float((1.0 / positive).mean()),
        inverse_of_mean=1.0 / mean_legal,
    )


def legal_counts_for_ids(
    shard_dirs: Sequence[Path], wanted: Sequence[str],
) -> np.ndarray:
    """Legal-move counts for exactly the rows named by ``wanted``.

    ⚑ Scoped to the FROZEN rows, not to the shards they live in. The chance
    floor has to describe the population the gate is read on; computing it over
    the whole held-out pool would be a different set with a different branching
    distribution, and the difference is invisible in the number itself.
    """
    want = set(wanted)
    found: dict[str, int] = {}
    for _path, arrs in iter_shard_arrays(shard_dirs):
        ids = row_ids(arrs)
        legal = np.asarray(arrs["legal_mask"]).astype(np.int64).sum(axis=1)
        for index, row_id in enumerate(ids):
            if row_id in want and row_id not in found:
                found[row_id] = int(legal[index])
    missing = len(want) - len(found)
    if missing:
        raise ValueError(
            f"{missing} of {len(want)} frozen row ids were not found in the "
            "given shard dirs. The frozen set and the shards do not match — "
            "recompute one of them rather than scoring a partial set.",
        )
    return np.array([found[row_id] for row_id in wanted], dtype=np.int64)
