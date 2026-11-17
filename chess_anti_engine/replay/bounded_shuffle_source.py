"""One checked source at a time and detached rows for bounded replay.

Full manifest/Python/codec/framework memory is not whole-RSS limited by this API.
Root/entry must authenticate metadata, source and owned I/O admission separately.
No metadata is permission to open historical shards outside an admitted owner.
"""
from __future__ import annotations

import bisect
from collections.abc import Callable
import copy
import json
import os
from typing import Any

import numpy as np

from .bounded_shuffle import canonical, integer, require
from .fixed_population import resident_array_bytes

ROW_OBJECT_HEADROOM = 32768


class SequentialArraySource:
    """Load sequentially once per uninterrupted pass; restore may reopen sources.

    The decoder must pre-reserve its declared maximum workspace before entry,
    authenticate source inventory and validate arrays. Only one decoded source
    is cached. All returned rows are owning, writable copies, never shard views.
    """

    def __init__(self, *, identity: dict[str, Any], membership: tuple[tuple[int, str, int], ...],
                 budgets: list[dict[str, int]], rng: Any,
                 decode: Callable[[int], dict[str, Any]]):
        require(len(membership) == len(budgets) > 0, "source budget roster differs")
        self._identity_json = canonical(identity)
        self.membership, self.rng, self._budgets, self._decode = membership, rng, copy.deepcopy(budgets), decode
        self._ends: list[int] = []
        total = 0
        for (_, _, rows), budget in zip(membership, budgets, strict=True):
            require(rows == budget["rows"], "source row budget differs")
            total += integer(rows, 1)
            self._ends.append(total)
        self.total_rows = total
        self.maximum_row_bytes = max(b["gather_row_bytes"] for b in budgets) + ROW_OBJECT_HEADROOM
        self.maximum_source_bytes = max(b["decoded_bytes"] + b["workspace_bytes"] for b in budgets)
        self._cache_index: int | None = None
        self._cache: dict[str, Any] | None = None

    @property
    def identity(self) -> dict[str, Any]:
        return json.loads(self._identity_json)

    def read_row(self, index: int) -> dict[str, Any]:
        integer(index)
        require(index < self.total_rows, "source row outside finite population")
        slot = bisect.bisect_right(self._ends, index)
        start = self._ends[slot - 1] if slot else 0
        if self._cache_index != slot:
            # Drop old source BEFORE allocating its successor, even on errors.
            self.close()
            arrays = self._decode(slot)
            require(resident_array_bytes(arrays) <= self._budgets[slot]["decoded_bytes"],
                    "source decoder exceeds reserved owning allocation")
            require(len(arrays) <= 128, "too many decoded row fields for object headroom")
            require(int(arrays["x"].shape[0]) == self.membership[slot][2], "decoded source row count differs")
            for value in arrays.values():
                value.setflags(write=False)
            self._cache, self._cache_index = arrays, slot
        require(self._cache is not None, "source cache absent")
        assert self._cache is not None
        local = index - start
        result = {key: np.array(value if value.ndim == 0 else value[local:local + 1], copy=True, order="C")
                  for key, value in self._cache.items()}
        require(self.owned_bytes(result) <= self.maximum_row_bytes, "detached source row exceeds reservation")
        return result

    def owned_bytes(self, payload: dict[str, Any]) -> int:
        return resident_array_bytes(payload) + ROW_OBJECT_HEADROOM

    def assemble_bound(self, count: int) -> int:
        integer(count, 1)
        # Full dense/sparse/gather copies, including nonzero index temporaries.
        # At most one row is sparsified at once; no declared NNZ assumption.
        return count * 80 * self.maximum_row_bytes + 65536

    def assemble(self, payloads: list[dict[str, Any]]) -> dict[str, Any]:
        from .buffer import ArrayReplayBuffer

        require(bool(payloads), "empty borrowed batch")
        buffer = ArrayReplayBuffer(len(payloads), rng=self.rng)
        for row in payloads:
            buffer.add_many_arrays(row)
        result = buffer._gather_rows(np.arange(len(payloads), dtype=np.int64))
        # Some scalar markers may originate in a row; detach the complete batch
        # so Trainer augmentation cannot alias queued rows/source cache/slots.
        return {key: np.array(value, copy=True, order="C") for key, value in result.items()}

    def close(self) -> None:
        self._cache = None
        self._cache_index = None


def from_fixed_manifest(architecture: dict[str, Any], manifest: dict[str, Any], manifest_ref: dict[str, str], *,
                        rng: Any) -> SequentialArraySource:
    """Actual future-owned source adapter; checked eager decode of ONE source.

    Metadata/hash/source admission remains caller-owned. This factory reads only
    named declarations initially. Decoder retains original pre/post raw inventory,
    producer-seal, codec, full target/history/TRAIN/identity validation.
    """
    from chess_anti_engine.model import infer_input_planes
    from scripts import core_finite_learner as core
    from .fixed_population import _checked_arrays, declared_memory_budget, manifest_identity, validate_manifest
    from .shard import load_shard_arrays

    validate_manifest(manifest)
    identity = manifest_identity(manifest, manifest_ref)
    require(manifest["architecture"] == architecture, "source architecture differs")
    # Freeze caller-owned mutable graphs before asynchronous selection/source I/O.
    sources = copy.deepcopy(manifest["sources"])
    arch = copy.deepcopy(architecture)
    planes = infer_input_planes(arch["input_extra_features"])
    budgets = [declared_memory_budget(s["shard"], s["rows"], planes) for s in sources]
    membership = tuple((s["generation"], s["source_sha256"], s["rows"]) for s in sources)

    def decode(slot: int) -> dict[str, Any]:
        source = sources[slot]
        path, before = core.verify_inventory(source["shard"])
        require(before == source["source_sha256"], "source inventory differs before stream decode")
        arrays, _ = load_shard_arrays(path, lazy=False, validate=True)
        _checked_arrays(arch, source, arrays)
        require(core.verify_inventory(source["shard"])[1] == before, "source changed during stream decode")
        return arrays

    return SequentialArraySource(identity=identity, membership=membership, budgets=budgets, rng=rng, decode=decode)


class BoundedReservationWriter:
    """Reuse QuotaLedger's exclusive write-ahead semantics, NEW record family.

    The existing QuotaLedger accepts raw encoded records; its file lock, fsync,
    immutable parent-journal hash/tail checks remain intact. Existing V2 global
    permutation journal validators MUST NOT be used for these records. Caller
    must supply a separately authenticated bounded-order prefix audit at resume.
    """

    def __init__(self, ledger: Any, journal: Any):
        self.ledger, self.journal = ledger, journal

    def __call__(self, rows: tuple[Any, ...], encoded: bytes) -> None:
        require(bool(rows) and encoded.endswith(b"\n"), "bounded reservation record required")
        self.ledger.reserve(encoded)
        require(self.journal.write(encoded) == len(encoded), "partial bounded draw journal: no retry")
        self.journal.flush()
        os.fsync(self.journal.fileno())
