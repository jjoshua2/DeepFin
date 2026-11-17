"""One closed-chain pass over stored row identities, before mirror augmentation.

Natural duplicate board contents are neither read nor deduplicated here. A failed
or interrupted draw taints this object. Only a closed, authenticated checkpoint
and its complete journal chain may continue; restarting a failed segment is not
supported. This is a quota within one declared lineage, not a worldwide exposure
registry or permission to restart a second lineage over already consumed rows.
"""
from __future__ import annotations

import bisect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import hashlib
import fcntl
import json
import os
from pathlib import Path
import stat
from typing import Any

from .generation import DrawnRow

SCHEMA = "STORED_ROW_EXACT_ONCE_CURSOR_V1"
SAMPLING = "STORED_ROW_EXACT_ONCE_PASS_V1"
ORDER = "sha256_seed_index_sort_v1"
IDENTITY = ["generation", "source_sha256", "row_index"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def integer(value: Any, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, "invalid stored-row integer")
    return value


def sha(value: Any) -> str:
    require(type(value) is str and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "invalid stored-row SHA256")
    return value


class StoredRowOnceReplay:
    """Adapter around a fully decoded GenerationReplay; no legacy RNG draw.

    The marker enables only Trainer serialization and no-discard/no-retry guards.
    It deliberately does not enable the different exact-epoch loss objective.
    Full batches only in this first family: an indivisible population is refused
    by construction, before Trainer/model construction. No row can wrap or pad.
    """

    stored_row_quota = True

    def __init__(self, replay: Any, *, seed: int, batch_size: int,
                 expected_rows: int, on_draw: Callable[[tuple[DrawnRow, ...]], None]) -> None:
        self._replay = replay
        self.rng = replay.rng
        self.membership = tuple(tuple(row) for row in replay.membership)
        require(bool(self.membership), "empty stored-row population")
        sources: set[str] = set()
        self._ends: list[int] = []
        total = 0
        for generation, source, count in self.membership:
            integer(generation, 1)
            sha(source)
            integer(count, 1)
            require(source not in sources, "repeated stored source identity")
            sources.add(source)
            total += count
            self._ends.append(total)
        require(total == len(replay) == integer(expected_rows, 1), "stored-row population mismatch")
        self.batch_size = integer(batch_size, 1)
        require(total % self.batch_size == 0, "last partial batch unsupported: refuse before first forward")
        self.seed = integer(seed)
        self.permutation = sorted(range(total), key=lambda index: (
            hashlib.sha256(f"{ORDER}:{self.seed}:{index}".encode()).digest(), index))
        self._cursor = self._batches = 0
        self._prefix = hashlib.sha256()
        self._tainted = False
        self._on_draw = on_draw

    def __len__(self) -> int:
        return self._ends[-1]

    @property
    def remaining(self) -> int:
        return len(self) - self._cursor

    def row_at(self, flat_index: int) -> DrawnRow:
        integer(flat_index)
        require(flat_index < len(self), "stored-row index outside population")
        chunk = bisect.bisect_right(self._ends, flat_index)
        generation, source, _ = self.membership[chunk]
        start = self._ends[chunk - 1] if chunk else 0
        return DrawnRow(generation, source, flat_index - start)

    def _update_prefix(self, rows: Sequence[DrawnRow]) -> None:
        for row in rows:
            self._prefix.update((canonical(asdict(row)) + "\n").encode())

    def abort_consumption(self) -> None:
        """Trainer faults keep already journaled rows spent; no checkpoint/retry."""
        self._tainted = True

    def sample_batch_arrays(self, batch_size: int) -> dict[str, Any]:
        require(not self._tainted, "tainted stored-row consumption: no retry")
        require(tuple(tuple(row) for row in self._replay.membership) == self.membership
                and len(self._replay) == len(self), "stored-row population changed after freezing")
        require(type(batch_size) is int and batch_size == self.batch_size, "stored-row batch size changed")
        require(self.remaining >= batch_size, "stored-row pass exhausted: no wraparound")
        indices = self.permutation[self._cursor:self._cursor + batch_size]
        rows = tuple(self.row_at(index) for index in indices)
        # Reserve before any callback/gather; failed journal/gather never releases quota.
        self._tainted = True
        self._cursor += batch_size
        self._batches += 1
        self._update_prefix(rows)
        self._on_draw(rows)  # Caller writes, flushes and fsyncs before returning.
        arrays = self._replay._buffer._gather_rows(indices)
        require(int(arrays["x"].shape[0]) == batch_size, "stored-row realized batch differs")
        self._tainted = False
        return arrays

    def state(self) -> dict[str, Any]:
        require(not self._tainted, "cannot checkpoint tainted stored-row consumption")
        return {"schema": SCHEMA, "identity": IDENTITY, "order": ORDER,
                "seed": self.seed, "membership": [list(row) for row in self.membership],
                "permutation": list(self.permutation), "batch_size": self.batch_size,
                "cursor": self._cursor, "batches": self._batches,
                "row_prefix_sha256": self._prefix.hexdigest(), "presentations_per_row": 1,
                "mirror_shares_quota": True, "tainted": False}

    def restore(self, state: Mapping[str, Any], journals: Sequence[dict[str, str]], *,
                max_file_bytes: int, max_total_bytes: int) -> None:
        require(self._cursor == self._batches == 0 and not self._tainted, "restore only into fresh adapter")
        expected = self.state()
        require(set(state) == set(expected), "stored-row checkpoint state schema")
        for key in set(expected) - {"cursor", "batches", "row_prefix_sha256"}:
            require(canonical(state[key]) == canonical(expected[key]),
                    f"stored-row checkpoint changed: {key}")
        cursor = integer(state["cursor"])
        batches = integer(state["batches"])
        require(cursor <= len(self) and cursor == batches * self.batch_size, "stored-row cursor/dose mismatch")
        sha(state["row_prefix_sha256"])
        # Validate full actual chain, not a caller assertion or only a checkpoint hash.
        observed = verify_journals(self, journals, max_file_bytes=max_file_bytes,
                                   max_total_bytes=max_total_bytes)
        require(observed == {"cursor": cursor, "batches": batches,
                             "row_prefix_sha256": state["row_prefix_sha256"]},
                "checkpoint/journal cursor mismatch: no tail rewind or redraw")
        self._cursor, self._batches = cursor, batches
        for index in self.permutation[:cursor]:
            self._update_prefix((self.row_at(index),))
        require(self._prefix.hexdigest() == state["row_prefix_sha256"], "stored-row saved prefix mismatch")


def _stamp(value: os.stat_result) -> tuple[int, ...]:
    # atime may change as a consequence of the authorized read.
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode)


def verify_journals(replay: StoredRowOnceReplay, journals: Sequence[dict[str, str]], *,
                    max_file_bytes: int, max_total_bytes: int) -> dict[str, Any]:
    """Read only bounded canonical JSONL in the admitted learner, never here at author time."""
    integer(max_file_bytes, 1)
    integer(max_total_bytes, 1)
    cursor = batches = total = 0
    prefix = hashlib.sha256()
    paths: set[Path] = set()
    for ref in journals:
        require(type(ref) is dict and set(ref) == {"path", "sha256"}, "stored-row journal ref schema")
        sha(ref["sha256"])
        path = Path(ref["path"])
        require(path.is_absolute() and not path.is_symlink(), "stored-row journal path")
        resolved = path.resolve(strict=True)
        require(resolved not in paths, "duplicate stored-row journal")
        paths.add(resolved)
        before = path.stat()
        require(stat.S_ISREG(before.st_mode) and before.st_mode & 0o222 == 0,
                "stored-row parent journal must be sealed regular file")
        require(before.st_size <= max_file_bytes, "stored-row journal file budget")
        total += before.st_size
        require(total <= max_total_bytes, "stored-row journal total budget")
        raw_hash = hashlib.sha256()
        with path.open("rb", buffering=0) as stream:
            require(_stamp(os.fstat(stream.fileno())) == _stamp(before), "stored-row journal changed before open")
            remaining = before.st_size
            while remaining:
                # Per-line bounded by the admitted file limit, never an unbounded readline.
                line = stream.readline(min(remaining, max_file_bytes))
                require(0 < len(line) <= remaining and line.endswith(b"\n"), "stored-row journal truncated/oversize")
                remaining -= len(line)
                raw_hash.update(line)
                value = json.loads(line)
                require(type(value) is dict and set(value) == {"draw_batch", "rows"}, "stored-row journal record schema")
                require(type(value["draw_batch"]) is int and value["draw_batch"] == batches,
                        "stored-row journal batch order")
                require(cursor + replay.batch_size <= len(replay), "stored-row journal overconsumes population")
                rows = [asdict(replay.row_at(i)) for i in replay.permutation[cursor:cursor + replay.batch_size]]
                require(canonical(value) + "\n" == line.decode("utf-8") and canonical(value["rows"]) == canonical(rows),
                        "stored-row journal identity/order/quota mismatch")
                for row in rows:
                    prefix.update((canonical(row) + "\n").encode())
                cursor += len(rows)
                batches += 1
            require(_stamp(os.fstat(stream.fileno())) == _stamp(before), "stored-row journal changed during read")
        require(_stamp(path.stat()) == _stamp(before) and raw_hash.hexdigest() == ref["sha256"], "stored-row journal custody/hash mismatch")
    return {"cursor": cursor, "batches": batches, "row_prefix_sha256": prefix.hexdigest()}


class QuotaLedger:
    """Exclusive persistent write-ahead reservation journal for one lineage.

    Checkpoint/journal tail equality is required before any new reservation.
    A failed sibling cannot be hidden by choosing a fresh output directory.
    Nothing deletes, rewinds, truncates or recreates an existing first ledger.
    """

    def __init__(self, path: Path, *, first: bool, expected_journals: Sequence[dict[str, str]],
                 max_file_bytes: int, max_total_bytes: int) -> None:
        self.path = path
        self.first = first
        self.expected_journals = expected_journals
        self.max_file_bytes = integer(max_file_bytes, 1)
        self.max_total_bytes = integer(max_total_bytes, 1)
        self._stream: Any = None
        self._bytes = 0
        self._raw = hashlib.sha256()

    def __enter__(self) -> QuotaLedger:
        require(self.path.is_absolute() and self.path.parent.resolve(strict=True) == self.path.parent
                and not self.path.is_symlink(), "quota ledger must be canonical and unlinked")
        fd = os.open(self.path, os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT | os.O_EXCL if self.first else 0), 0o600)
        self._stream = os.fdopen(fd, "r+b", buffering=0)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            opened = os.fstat(fd)
            if self.first:
                parent_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(parent_fd)
                finally:
                    os.close(parent_fd)
            require(stat.S_ISREG(opened.st_mode) and stat.S_IMODE(opened.st_mode) == 0o600
                    and _stamp(self.path.stat()) == _stamp(opened), "quota ledger file identity/mode")
            require(opened.st_size <= min(self.max_file_bytes, self.max_total_bytes), "quota ledger total byte cap")
            expected_raw = hashlib.sha256()
            expected_bytes = 0
            for ref in self.expected_journals:
                require(type(ref) is dict and set(ref) == {"path", "sha256"}, "quota parent journal ref")
                path = Path(ref["path"])
                require(path.is_absolute() and not path.is_symlink(), "quota parent journal path")
                before = path.stat()
                require(stat.S_ISREG(before.st_mode) and not before.st_mode & 0o222
                        and before.st_size <= self.max_file_bytes, "quota parent journal custody/cap")
                expected_bytes += before.st_size
                require(expected_bytes <= self.max_total_bytes, "quota parent journal chain cap")
                raw = hashlib.sha256()
                with path.open("rb", buffering=0) as stream:
                    require(_stamp(os.fstat(stream.fileno())) == _stamp(before), "quota parent journal changed on open")
                    remaining = before.st_size
                    while remaining:
                        chunk = stream.read(min(1 << 20, remaining))
                        require(bool(chunk), "quota parent journal truncated")
                        remaining -= len(chunk)
                        raw.update(chunk)
                        expected_raw.update(chunk)
                    require(_stamp(os.fstat(stream.fileno())) == _stamp(before), "quota parent journal changed during read")
                require(_stamp(path.stat()) == _stamp(before) and raw.hexdigest() == sha(ref["sha256"]),
                        "quota parent journal changed/hash mismatch")
            require(opened.st_size == expected_bytes, "quota ledger ahead/behind checkpoint: no tail rewind")
            remaining = opened.st_size
            while remaining:
                chunk = self._stream.read(min(1 << 20, remaining))
                require(bool(chunk), "quota ledger truncated")
                remaining -= len(chunk)
                self._raw.update(chunk)
            require(self._raw.hexdigest() == expected_raw.hexdigest()
                    and _stamp(os.fstat(fd)) == _stamp(opened) and _stamp(self.path.stat()) == _stamp(opened),
                    "quota ledger differs from checkpoint journal chain")
            self._bytes = opened.st_size
            return self
        except BaseException:
            self._stream.close()
            self._stream = None
            raise

    def reserve(self, encoded: bytes) -> None:
        stream = self._stream
        if stream is None:
            raise ValueError("quota ledger unavailable/record encoding")
        require(encoded.endswith(b"\n"), "quota ledger unavailable/record encoding")
        require(self._bytes + len(encoded) <= min(self.max_file_bytes, self.max_total_bytes), "quota ledger reservation byte cap")
        require(stream.write(encoded) == len(encoded), "partial quota reservation: no retry")
        stream.flush()
        os.fsync(stream.fileno())
        self._bytes += len(encoded)
        self._raw.update(encoded)
        require(_stamp(self.path.stat()) == _stamp(os.fstat(stream.fileno())), "quota ledger path changed")

    def checkpoint(self) -> dict[str, Any]:
        stream = self._stream
        if stream is None:
            raise ValueError("quota ledger changed size")
        require(os.fstat(stream.fileno()).st_size == self._bytes, "quota ledger changed size")
        before = os.fstat(stream.fileno())
        stream.seek(0)
        raw = hashlib.sha256()
        remaining = self._bytes
        while remaining:
            chunk = stream.read(min(1 << 20, remaining))
            require(bool(chunk), "quota ledger checkpoint truncated")
            raw.update(chunk)
            remaining -= len(chunk)
        require(raw.hexdigest() == self._raw.hexdigest()
                and _stamp(os.fstat(stream.fileno())) == _stamp(before)
                and _stamp(self.path.stat()) == _stamp(before), "quota ledger checkpoint content changed")
        return {"path": str(self.path), "bytes": self._bytes, "sha256": raw.hexdigest()}

    def __exit__(self, *_args: Any) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
