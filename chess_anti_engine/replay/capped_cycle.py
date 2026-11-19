"""Cumulative stored-asset reservations, independent of a cycle permutation.

Original authenticated generation is retained in each row identity. The entry
authenticates it against producing evidence, never a user-renamed flat index.
Mirroring happens after reservation and does not create a second quota.
This is one authenticated learning lineage, not semantic board deduplication.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, cast

from .stored_row_once import canonical, integer, require, sha

SCHEMA = "CUMULATIVE_STORED_ROW_USAGE_V1"


def stamp(s: os.stat_result) -> tuple[int, ...]:
    return tuple(getattr(s, k) for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode"))


class Usage:
    """Reserved rows count even when a subsequent mirror/forward/update fails."""

    def __init__(self, *, maximum: int, max_rows: int) -> None:
        require(type(maximum) is int and 1 <= maximum <= 2, "cycle presentation cap must be1or2 total")
        self.maximum = maximum
        self.max_rows = integer(max_rows, 1)
        self.counts: Counter[tuple[int, str, int]] = Counter()
        self.batches = 0
        self.presentations = 0
        self._bytes = 0
        self._raw = hashlib.sha256()

    def validate_rows(self, values: Any) -> list[tuple[int, str, int]]:
        require(type(values) is list and bool(values), "empty/invalid cumulative row reservation")
        values = cast(list[Any], values)
        keys: list[tuple[int, str, int]] = []
        local: set[tuple[int, str, int]] = set()
        for value in values:
            require(type(value) is dict and set(value) == {"generation", "source_sha256", "row_index"},
                    "cumulative physical row identity schema")
            value = cast(dict[str, Any], value)
            generation = integer(value["generation"], 1)
            source, index = sha(value["source_sha256"]), integer(value["row_index"])
            key = (generation, source, index)
            require(key not in local and self.counts[key] < self.maximum, "stored-row cumulative presentation cap exceeded")
            local.add(key)
            keys.append(key)
        require(len(self.counts) + sum(key not in self.counts for key in keys) <= self.max_rows,
                "cumulative distinct-row memory cap")
        return keys

    def accept(self, values: list[dict[str, Any]]) -> None:
        keys = self.validate_rows(values)
        encoded = (canonical({"draw_batch": self.batches, "rows": values}) + "\n").encode()
        self._raw.update(encoded)
        self._bytes += len(encoded)
        self.counts.update(keys)
        self.batches += 1
        self.presentations += len(keys)

    def check_population(self, membership: Sequence[tuple[int, str, int]]) -> None:
        """Refuse spent original row identities before loading data/model.

        The caller supplies authenticated membership from the accepted bank;
        this counter is not an authority to relabel old rows as a new source.
        """
        pending = 0
        for generation, source, rows in membership:
            for index in range(rows):
                key = (generation, source, index)
                require(self.counts[key] < self.maximum, "cycle population contains already exhausted rows")
                pending += key not in self.counts
        require(len(self.counts) + pending <= self.max_rows, "cycle population exceeds cumulative row cap")

    def state(self) -> dict[str, Any]:
        h = hashlib.sha256()
        for (generation, source, index), count in sorted(self.counts.items()):
            h.update((canonical([generation, source, index, count]) + "\n").encode())
        return {"schema": SCHEMA, "maximum_presentations": self.maximum, "unique_rows": len(self.counts),
                "presentations": self.presentations, "batches": self.batches,
                "maximum_observed": max(self.counts.values(), default=0), "counts_sha256": h.hexdigest()}

    def ledger_checkpoint(self, path: str) -> dict[str, Any]:
        return {"path": path, "bytes": self._bytes, "sha256": self._raw.hexdigest()}


def read_usage(journals: Sequence[dict[str, str]], *, maximum: int, max_rows: int,
               batch_size: int, max_file_bytes: int, max_total_bytes: int,
               on_record: Callable[[Path, dict[str, Any]], None] | None = None) -> Usage:
    """Actual bounded sealed journal reads; caller must own this data-read scope.

    Not a payload-free Root metadata helper. No checkpoint/model/shard access.
    QuotaLedger separately authenticates the exact same concatenated bytes and
    refuses an uncheckpointed persistent-ledger tail before any new reservation.
    """
    usage = Usage(maximum=maximum, max_rows=max_rows)
    integer(batch_size, 1)
    integer(max_file_bytes, 1)
    integer(max_total_bytes, 1)
    total = 0
    paths: set[Path] = set()
    for ref in journals:
        require(type(ref) is dict and set(ref) == {"path", "sha256"}, "cycle journal reference")
        path = Path(ref["path"])
        require(path.is_absolute() and not path.is_symlink() and path.resolve() == path and path not in paths,
                "cycle journal canonical unique path")
        paths.add(path)
        before = path.stat()
        require(stat.S_ISREG(before.st_mode) and not before.st_mode & 0o222
                and before.st_size <= max_file_bytes, "cycle journal readonly file cap")
        total += before.st_size
        require(total <= max_total_bytes, "cycle journal aggregate cap")
        h = hashlib.sha256()
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            require(stamp(os.fstat(stream.fileno())) == stamp(before), "cycle journal changed before read")
            remaining = before.st_size
            while remaining:
                line = stream.readline(min(remaining, max_file_bytes))
                require(bool(line) and len(line) <= remaining and line.endswith(b"\n"), "cycle journal truncated")
                remaining -= len(line)
                value = json.loads(line)
                require(type(value) is dict and set(value) == {"draw_batch", "rows"}
                        and type(value["draw_batch"]) is int and value["draw_batch"] == usage.batches
                        and canonical(value).encode() + b"\n" == line, "cycle journal canonical batch order")
                value = cast(dict[str, Any], value)
                require(type(value["rows"]) is list and 0 < len(value["rows"]) <= batch_size,
                        "cycle journal actual row count")
                usage.accept(value["rows"])
                if on_record is not None:
                    on_record(path, value)
                h.update(line)
            require(stamp(os.fstat(stream.fileno())) == stamp(before), "cycle journal changed during read")
        require(stamp(path.stat()) == stamp(before) and h.hexdigest() == sha(ref["sha256"]), "cycle journal hash/custody")
    return usage


def row_dicts(rows: Any) -> list[dict[str, Any]]:
    return [asdict(row) for row in rows]
