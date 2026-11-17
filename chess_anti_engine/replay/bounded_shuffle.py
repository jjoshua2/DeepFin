"""Opt-in approximate bounded shuffle; never the global-permutation baseline.

One producer owns selection/source I/O. Checkpoint joins it at a batch boundary.
Ready rows are not presentations; only the durable reserve callback spends them.
The borrowed consumer batch must be completed/released before checkpoint/next use.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import random
import threading
from typing import Any, Protocol

from .generation import DrawnRow

SCHEMA = "APPROXIMATE_BOUNDED_SHUFFLE_CURSOR_V1"
ORDER = "APPROXIMATE_BOUNDED_SHUFFLE_V1"
DRAW_SCHEMA = "APPROXIMATE_BOUNDED_SHUFFLE_RESERVATION_V1"


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def integer(value: Any, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, "invalid bounded-shuffle integer")
    return value


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def audit_selection(state: dict[str, Any], records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Pure closed-lineage order audit: symbolic IDs only, no payload or model.

    Caller authenticates raw immutable journal/checkpoint bytes and parses them
    strictly. This validates all lineage records plus the unspent ready frontier;
    it neither calls a source nor grants model/delivery/launch qualification.
    """
    require(state["schema"] == SCHEMA and state["order"] == ORDER
            and digest({k: v for k, v in state.items() if k != "sha256"}) == state["sha256"], "order audit state identity")
    n = sum(integer(row[2], 1) for row in state["membership"])
    size = integer(state["batch_size"], 1)
    capacity = integer(state["limits"]["buffer_rows"], size)
    rng = random.Random(integer(state["seed"]))
    slots: list[int] = []
    cursor = emitted = batches = 0
    prefix = "0" * 64
    spent = bytearray((n + 7) // 8)
    lookup: dict[str, tuple[int, int, int]] = {}
    start = 0
    for generation, sha, count in state["membership"]:
        require(sha not in lookup, "duplicate symbolic source")
        lookup[sha] = (integer(generation, 1), start, integer(count, 1))
        start += count

    def choose() -> int:
        nonlocal cursor
        while len(slots) < capacity and cursor < n:
            slots.append(cursor)
            cursor += 1
        require(bool(slots), "selection beyond finite source")
        slot = rng.randrange(len(slots))
        selected = slots.pop(slot)
        if cursor < n:
            slots.insert(slot, cursor)
            cursor += 1
        return selected

    for record in records:
        require(set(record) == {"schema", "order", "lineage", "identity_sha256", "batch", "emitted", "previous", "rows"}
                and record["schema"] == DRAW_SCHEMA and record["order"] == ORDER
                and record["lineage"] == state["lineage"]
                and record["identity_sha256"] == digest(state["identity"]) and record["previous"] == prefix,
                "reservation record family/prefix differs")
        require(type(record["rows"]) is list and len(record["rows"]) == min(size, n - emitted)
                and 0 < len(record["rows"]) <= state["emitted"] - emitted, "reservation record row count differs")
        for row in record["rows"]:
            require(type(row) is dict and set(row) == {"generation", "source_sha256", "row_index"}, "physical record fields differ")
            generation, base, count = lookup[row["source_sha256"]]
            local = integer(row["row_index"])
            selected = choose()
            require(type(row["generation"]) is int and row["generation"] == generation and local < count
                    and base + local == selected, "reservation order differs from approximate shuffle")
            spent[selected // 8] |= 1 << (selected % 8)
        emitted += len(record["rows"])
        batches += 1
        require(type(record["batch"]) is int and record["batch"] == batches
                and type(record["emitted"]) is int and record["emitted"] == emitted, "reservation ordinal differs")
        prefix = hashlib.sha256((canonical(record) + "\n").encode()).hexdigest()
    require(emitted == state["emitted"] and batches == state["batches"]
            and prefix == state["reservation_prefix"] and spent.hex() == state["spent_hex"], "closed journal frontier differs")
    require(type(state["ready"]) is list and len(state["ready"]) <= integer(state["limits"]["ready_batches"], 1),
            "ready frontier capacity differs")
    require(all(type(batch) is list and 0 < len(batch) <= size for batch in state["ready"])
            and all(len(batch) == size for batch in state["ready"][:-1]), "ready frontier batch shape differs")
    queued = 0
    for batch in state["ready"]:
        for index in batch:
            require(type(index) is int and index == choose(), "ready order differs from symbolic selection")
            queued += 1
    require(cursor == state["upstream"] and slots == state["slots"]
            and canonical(rng.getstate()) == canonical(state["shuffle_state"]), "source/slot/RNG frontier differs")
    return {"schema": "BOUNDED_SHUFFLE_SYMBOLIC_SELECTION_AUDIT_V1", "state_sha256": state["sha256"],
            "emitted": emitted, "queued_unspent": queued, "reservation_prefix": prefix,
            "payload_reads": 0, "model_qualification": False}


@dataclass(frozen=True)
class ReplayLimits:
    """Array/owned-row working storage only, not framework/manifest/whole RSS."""

    total_bytes: int
    source_bytes: int
    buffer_bytes: int
    ready_bytes: int
    consumer_bytes: int
    scratch_bytes: int
    buffer_rows: int
    ready_batches: int

    def __post_init__(self) -> None:
        for value in asdict(self).values():
            integer(value, 1)
        require(self.ready_batches <= 4, "at most four ready batches")
        require(sum((self.source_bytes, self.buffer_bytes, self.ready_bytes,
                     self.consumer_bytes, self.scratch_bytes)) <= self.total_bytes,
                "category caps exceed replay working-set target")


class RowSource(Protocol):
    @property
    def identity(self) -> dict[str, Any]: ...
    membership: tuple[tuple[int, str, int], ...]
    rng: Any  # Original augmentation RNG, never consumed by shuffle selection.
    maximum_row_bytes: int
    maximum_source_bytes: int

    def read_row(self, index: int) -> dict[str, Any]: ...
    def owned_bytes(self, payload: dict[str, Any]) -> int: ...
    def assemble_bound(self, count: int) -> int: ...
    def assemble(self, payloads: list[dict[str, Any]]) -> dict[str, Any]: ...
    def close(self) -> None: ...


class MemoryBudget:
    def __init__(self, limits: ReplayLimits):
        self.caps = {k: getattr(limits, k + "_bytes") for k in
                     ("source", "buffer", "ready", "consumer", "scratch")}
        self.used = dict.fromkeys(self.caps, 0)
        self.peak = 0

    def acquire(self, group: str, size: int) -> None:
        integer(size)
        require(self.used[group] + size <= self.caps[group], "bounded replay byte cap: " + group)
        self.used[group] += size
        self.peak = max(self.peak, sum(self.used.values()))

    def release(self, group: str, size: int) -> None:
        require(0 <= size <= self.used[group], "invalid replay byte release")
        self.used[group] -= size

    def transfer(self, old: str, new: str, size: int) -> None:
        # Check the destination before moving ownership; never allocate here.
        require(self.used[new] + size <= self.caps[new], "bounded replay byte cap: " + new)
        self.release(old, size)
        self.acquire(new, size)


@dataclass
class _Row:
    index: int
    payload: dict[str, Any]
    charge: int


class BoundedShuffleReplay:
    """Fill slots, random emit/replace, drain once; not a uniform permutation.

    No priority/skip/replacement of already emitted rows. Mirror uses source.rng,
    independent of the private random.Random. Caller authenticates checkpoint
    bytes and audited reservation-prefix metadata before restore; checksum alone
    is not external authority. Threads never invoke reserve or augmentation.
    """

    stored_row_quota = True

    def __init__(self, source: RowSource, *, seed: int, batch_size: int, limits: ReplayLimits,
                 lineage: str, reserve: Callable[[tuple[DrawnRow, ...], bytes], None]):
        self.source, self.rng = source, source.rng
        self._identity = copy.deepcopy(source.identity)
        self.membership: tuple[tuple[int, str, int], ...] = tuple(source.membership)
        require(bool(self.membership) and len({s[1] for s in self.membership}) == len(self.membership),
                "unique original source membership required")
        for generation, sha, rows in self.membership:
            integer(generation, 1)
            integer(rows, 1)
            require(type(sha) is str and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),
                    "invalid original source SHA")
        require(type(lineage) is str and bool(lineage), "explicit student lineage required")
        self.total_rows = sum(s[2] for s in self.membership)
        self.batch_size, self.seed = integer(batch_size, 1), integer(seed)
        self.limits, self.lineage, self._reserve = limits, lineage, reserve
        require(limits.buffer_rows >= self.batch_size and source.maximum_row_bytes > 0,
                "buffer must hold a nominal batch")
        require(source.maximum_row_bytes * limits.buffer_rows <= limits.buffer_bytes and
                source.maximum_row_bytes * self.batch_size * limits.ready_batches <= limits.ready_bytes and
                source.maximum_source_bytes <= limits.source_bytes and
                source.assemble_bound(self.batch_size) <= limits.consumer_bytes,
                "configuration cannot hold bounded source/buffer/queue/consumer")
        require(self.rng.bit_generator.__class__.__name__ == "PCG64", "explicit PCG64 augmentation state required")
        self._shuffle = random.Random(self.seed)
        self._budget = MemoryBudget(limits)
        self._budget.acquire("source", source.maximum_source_bytes)
        self._budget.acquire("scratch", limits.scratch_bytes)  # fixed declared scratch headroom
        self._slots: list[_Row] = []
        self._ready: deque[list[_Row]] = deque()
        self._upstream = self._emitted = self._batches = 0
        self._bits = bytearray((self.total_rows + 7) // 8)
        self._prefix = "0" * 64
        self._inflight: dict[str, Any] | None = None
        self._consumer_charge = 0
        self._tainted = self._closed = self._stop = False
        self._error: str | None = None
        self._callback_thread: int | None = None
        self._condition = threading.Condition()
        self._worker: threading.Thread | None = None

    def __len__(self) -> int:
        return self.total_rows

    @property
    def remaining(self) -> int:
        return self.total_rows - self._emitted

    @property
    def total_batches(self) -> int:
        return (self.total_rows + self.batch_size - 1) // self.batch_size

    @property
    def memory_accounting(self) -> dict[str, Any]:
        with self._condition:
            return {"used": dict(self._budget.used), "peak_bytes": self._budget.peak,
                    "total_cap_bytes": self.limits.total_bytes, "ready_batches": len(self._ready)}

    def row_at(self, index: int) -> DrawnRow:
        integer(index)
        require(index < self.total_rows, "row outside original population")
        for generation, sha, count in self.membership:
            if index < count:
                return DrawnRow(generation, sha, index)
            index -= count
        raise AssertionError("unreachable original-row mapping")

    def _copy(self, index: int, group: str) -> _Row:
        charge = self.source.maximum_row_bytes
        self._budget.acquire(group, charge)  # before any detached copy allocation
        try:
            payload = self.source.read_row(index)
            require(self.source.owned_bytes(payload) <= charge, "detached row exceeds reserved bytes")
            return _Row(index, payload, charge)
        except BaseException:
            self._budget.release(group, charge)
            raise

    def _produce(self) -> None:
        require(len(self._ready) < self.limits.ready_batches, "ready queue backpressure")
        while len(self._slots) < self.limits.buffer_rows and self._upstream < self.total_rows:
            self._slots.append(self._copy(self._upstream, "buffer"))
            self._upstream += 1
        output: list[_Row] = []
        while self._slots and len(output) < self.batch_size:
            slot = self._shuffle.randrange(len(self._slots))
            selected = self._slots.pop(slot)
            self._budget.transfer("buffer", "ready", selected.charge)
            output.append(selected)
            if self._upstream < self.total_rows:
                self._slots.insert(slot, self._copy(self._upstream, "buffer"))
                self._upstream += 1
        if output:
            self._ready.append(output)
        # This function is atomic under the condition, including one partial
        # producer batch. Checkpoint stop/join therefore has no hidden frontier.

    def _work(self) -> None:
        with self._condition:
            try:
                while not self._stop:
                    if len(self._ready) >= self.limits.ready_batches:
                        self._condition.wait()
                    elif self._upstream == self.total_rows and not self._slots:
                        return
                    else:
                        self._produce()
                        self._condition.notify_all()
            except BaseException as error:
                # Never retain an exception traceback that could pin decoded rows.
                self._error = type(error).__name__ + ": " + str(error)[:4096]
                self._tainted = self._stop = True
                self._condition.notify_all()

    def start_prefetch(self) -> None:
        with self._condition:
            require(not self._closed and not self._tainted, "closed/tainted replay")
            require(self._worker is None, "one prefetch worker only")
            self._stop = False
            self._worker = threading.Thread(target=self._work, name="bounded-replay-prefetch", daemon=False)
            self._worker.start()

    def stop_prefetch(self) -> None:
        require(self._callback_thread != threading.get_ident(), "reentrant replay control during reservation")
        with self._condition:
            self._stop = True
            self._condition.notify_all()
            worker = self._worker
        if worker is not None:
            require(worker is not threading.current_thread(), "prefetch worker cannot join itself")
            worker.join()  # source reads must have their own finite owner deadline
        with self._condition:
            self._worker = None

    def sample_batch_arrays(self, batch_size: int) -> dict[str, Any]:
        with self._condition:
            require(not self._closed and not self._tainted and self._error is None, "closed/tainted replay: no retry")
            require(canonical(self.source.identity) == canonical(self._identity)
                    and tuple(tuple(s) for s in self.source.membership) == self.membership, "source identity changed")
            require(type(batch_size) is int and batch_size == self.batch_size, "batch size changed")
            require(self._inflight is None, "complete borrowed batch before another draw")
            require(self.remaining > 0, "bounded shuffle pass exhausted: no wraparound")
            while not self._ready:
                if self._worker is None or not self._worker.is_alive():
                    try:
                        self._produce()
                    except BaseException:
                        self._tainted = True
                        raise
                    break
                self._condition.wait()
                require(not self._tainted, "prefetch failed: no retry")
            require(bool(self._ready), "source omitted unconsumed rows")
            rows = self._ready.popleft()
            physical = tuple(self.row_at(row.index) for row in rows)
            self._tainted = True  # stays true through any reservation/assembly fault
            for row in rows:
                byte, bit = divmod(row.index, 8)
                require(not self._bits[byte] & (1 << bit), "duplicate actual presentation")
                self._bits[byte] |= 1 << bit
            self._emitted += len(rows)
            self._batches += 1
            record = {"schema": DRAW_SCHEMA, "order": ORDER, "lineage": self.lineage,
                      "identity_sha256": digest(self._identity), "batch": self._batches,
                      "emitted": self._emitted, "previous": self._prefix,
                      "rows": [asdict(row) for row in physical]}
            encoded = (canonical(record) + "\n").encode()
            self._prefix = hashlib.sha256(encoded).hexdigest()
            try:
                self._callback_thread = threading.get_ident()
                try:
                    self._reserve(physical, encoded)  # durable ledger+journal BEFORE mirror/forward
                finally:
                    self._callback_thread = None
                charge = self.source.assemble_bound(len(rows))
                self._budget.acquire("consumer", charge)
                self._consumer_charge = charge
                arrays = self.source.assemble([row.payload for row in rows])
                require(self.source.owned_bytes(arrays) <= charge, "assembled batch exceeds byte reservation")
                require(int(arrays["x"].shape[0]) == len(rows), "actual final batch rows differ")
                self._inflight = arrays
                for row in rows:
                    self._budget.release("ready", row.charge)
                self._tainted = False
                self._condition.notify_all()
                return arrays
            except BaseException:
                self._stop = True
                self._condition.notify_all()
                raise

    def complete_batch(self) -> None:
        """Caller releases ALL borrowed payload references after optimizer/use.

        A Trainer/entry integration must add this finally-success seam. Until
        then this standalone opt-in engine is not deployable through old entry.
        """
        with self._condition:
            require(not self._tainted and self._inflight is not None, "no clean borrowed batch to complete")
            self._inflight = None
            self._budget.release("consumer", self._consumer_charge)
            self._consumer_charge = 0

    def abort_consumption(self) -> None:
        with self._condition:
            self._tainted = self._stop = True
            self._condition.notify_all()
        self.stop_prefetch()

    def _partition(self) -> None:
        self._partition_indices([row.index for row in self._slots], [[row.index for row in b] for b in self._ready])

    def _partition_indices(self, slots: list[int], ready: list[list[int]]) -> None:
        pending = list(slots) + [index for batch in ready for index in batch]
        require(all(type(batch) is list and 0 < len(batch) <= self.batch_size for batch in ready), "invalid ready batch")
        require(all(len(batch) == self.batch_size for batch in ready[:-1]), "partial nonfinal ready batch")
        if ready and len(ready[-1]) < self.batch_size:
            require(self._upstream == self.total_rows and not slots
                    and self._emitted + sum(len(b) for b in ready) == self.total_rows, "partial batch before actual EOF")
        require(self._emitted % self.batch_size == 0 or self._emitted == self.total_rows, "partial nonfinal reservation")
        require(len(pending) == len(set(pending)), "duplicate buffer/ready row")
        require(all(type(i) is int and 0 <= i < self._upstream for i in pending), "pending row beyond source cursor")
        pending_set = set(pending)
        used = 0
        for index in range(self.total_rows):
            spent = bool(self._bits[index // 8] & (1 << (index % 8)))
            used += int(spent)
            require(not (spent and index in pending_set), "reserved row still pending")
            require((spent or index in pending_set) == (index < self._upstream), "source partition lost/added row")
        require(used == self._emitted and self._batches == (used + self.batch_size - 1) // self.batch_size,
                "reservation counts differ")
        require(self.total_rows % 8 == 0 or not self._bits[-1] >> (self.total_rows % 8), "nonpopulation bitmap tail")

    def checkpoint(self) -> dict[str, Any]:
        self.stop_prefetch()
        with self._condition:
            require(not self._closed and not self._tainted and self._inflight is None, "cannot checkpoint tainted/inflight replay")
            self._partition()
            self.source.close()  # decoded source payload is reconstructible
            state = {"schema": SCHEMA, "order": ORDER, "identity": copy.deepcopy(self._identity),
                     "membership": [list(s) for s in self.membership], "lineage": self.lineage,
                     "seed": self.seed, "batch_size": self.batch_size, "limits": asdict(self.limits),
                     "upstream": self._upstream, "slots": [r.index for r in self._slots],
                     "ready": [[r.index for r in batch] for batch in self._ready],
                     "emitted": self._emitted, "batches": self._batches, "spent_hex": self._bits.hex(),
                     "reservation_prefix": self._prefix, "shuffle_state": self._shuffle.getstate(),
                     "mirror_state": copy.deepcopy(self.rng.bit_generator.state)}
            return {**state, "sha256": digest(state)}

    def restore(self, state: dict[str, Any], *, journal_rows: int, journal_prefix: str) -> None:
        """Only a fresh object; ANY failed restore permanently taints it."""
        try:
            self._restore(state, journal_rows=journal_rows, journal_prefix=journal_prefix)
        except BaseException:
            with self._condition:
                self._tainted = self._stop = True
                self._condition.notify_all()
            self.stop_prefetch()
            raise

    def _restore(self, state: dict[str, Any], *, journal_rows: int, journal_prefix: str) -> None:
        with self._condition:
            require(self._worker is None and not self._slots and not self._ready and not self._upstream
                    and not self._emitted and not self._closed, "restore requires pristine replay")
            expected = {"schema", "order", "identity", "membership", "lineage", "seed", "batch_size", "limits",
                        "upstream", "slots", "ready", "emitted", "batches", "spent_hex", "reservation_prefix",
                        "shuffle_state", "mirror_state", "sha256"}
            require(type(state) is dict and set(state) == expected and digest({k: v for k, v in state.items() if k != "sha256"}) == state["sha256"],
                    "corrupt bounded shuffle state")
            require(state["schema"] == SCHEMA and state["order"] == ORDER and canonical(state["identity"]) == canonical(self._identity)
                    and canonical(state["membership"]) == canonical([list(s) for s in self.membership]) and state["lineage"] == self.lineage
                    and type(state["seed"]) is int and state["seed"] == self.seed
                    and type(state["batch_size"]) is int and state["batch_size"] == self.batch_size
                    and canonical(state["limits"]) == canonical(asdict(self.limits)), "checkpoint family/config/identity differs")
            self._upstream = integer(state["upstream"])
            self._emitted, self._batches = integer(state["emitted"]), integer(state["batches"])
            require(self._upstream <= self.total_rows and self._emitted <= self.total_rows
                    and self._emitted == integer(journal_rows) and state["reservation_prefix"] == journal_prefix,
                    "checkpoint and audited journal frontier differ")
            require(type(state["spent_hex"]) is str and len(state["spent_hex"]) == 2 * len(self._bits), "bitmap size differs")
            self._bits = bytearray.fromhex(state["spent_hex"])
            require(type(journal_prefix) is str and len(journal_prefix) == 64
                    and all(c in "0123456789abcdef" for c in journal_prefix), "invalid audited reservation prefix")
            self._prefix = state["reservation_prefix"]
            require(type(state["slots"]) is list and len(state["slots"]) <= self.limits.buffer_rows
                    and type(state["ready"]) is list and len(state["ready"]) <= self.limits.ready_batches, "checkpoint queue/slot cap")
            self._partition_indices(state["slots"], state["ready"])  # refuse bad partitions BEFORE source I/O
            # Reconstruct payloads by exact original rowIDs; this consumes no
            # shuffle/mirror RNG and does not reserve presentations a second time.
            try:
                rs = state["shuffle_state"]
                require(isinstance(rs, (tuple, list)) and len(rs) == 3 and rs[0] == 3 and rs[2] is None
                        and isinstance(rs[1], (tuple, list)) and len(rs[1]) == 625
                        and all(type(v) is int and 0 <= v < 2**32 for v in rs[1][:-1])
                        and type(rs[1][-1]) is int and 0 <= rs[1][-1] <= 624, "invalid private shuffle RNG")
                self._shuffle.setstate((3, tuple(rs[1]), None))
                mirror = state["mirror_state"]
                require(type(mirror) is dict and set(mirror) == {"bit_generator", "state", "has_uint32", "uinteger"}
                        and mirror["bit_generator"] == "PCG64" and type(mirror["state"]) is dict
                        and set(mirror["state"]) == {"state", "inc"}
                        and all(type(v) is int and 0 <= v < 2**128 for v in mirror["state"].values())
                        and mirror["state"]["inc"] % 2 == 1
                        and type(mirror["has_uint32"]) is int and mirror["has_uint32"] in (0, 1)
                        and type(mirror["uinteger"]) is int and 0 <= mirror["uinteger"] < 2**32, "invalid mirror RNG")
                self.rng.bit_generator.state = copy.deepcopy(mirror)
                self._slots = [self._copy(integer(i), "buffer") for i in state["slots"]]
                for batch in state["ready"]:
                    require(type(batch) is list and 0 < len(batch) <= self.batch_size, "invalid ready batch")
                    self._ready.append([self._copy(integer(i), "ready") for i in batch])
                self._partition()
            except BaseException:
                self._tainted = True
                raise

    def close(self) -> None:
        self.stop_prefetch()
        with self._condition:
            self.source.close()
            self._slots.clear()
            self._ready.clear()
            self._inflight = None
            for key in self._budget.used:
                self._budget.used[key] = 0
            self._closed = True

    def __enter__(self) -> BoundedShuffleReplay:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
