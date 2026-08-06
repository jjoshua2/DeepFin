from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from chess_anti_engine.inference import BatchEvaluator

_CacheKey: TypeAlias = tuple[bytes, tuple[int, ...], str, bytes]
_CacheValue: TypeAlias = tuple[np.ndarray, np.ndarray]
_PucvKey: TypeAlias = tuple[int, int]
_PucvValue: TypeAlias = tuple[bytes, np.ndarray, np.ndarray]

# Everything that changes what an encoded row MEANS, or what an output row
# means, without necessarily changing either one's shape. `input_extra_features`
# does move the plane count (v1 146 vs v2_threats 175) and so is already
# implied by the row shape; `input_history_encoding` is the dangerous one --
# same 175 planes, different contents -- and `policy_encoding` changes the
# width and meaning of the cached OUTPUT (compact lc0_1858 vs 4672 action ids).
_ENCODING_ATTRS: tuple[str, ...] = (
    "input_extra_features",
    "input_history_encoding",
    "policy_encoding",
    "use_dynamic_relations",
)

# Attribute names evaluators use for the thing they wrap, walked in order to
# find the object that declares the encoding (production chain:
# ThreadSafeGPUDispatcher._eval -> DirectGPUEvaluator.model -> ChessNet).
_WRAPPED_ATTRS: tuple[str, ...] = ("model", "_model", "_inner", "_eval", "_evaluator")
_MAX_WRAPPER_DEPTH = 8


def resolve_encoding_source(obj: object) -> object:
    """First object in ``obj``'s wrapper chain that declares its encoding.

    Raises when nothing in the chain declares one: a cache that cannot name
    the encoding it is caching cannot promise not to serve entries across an
    encoding change, and quietly caching anyway is the failure this exists to
    prevent.
    """
    node = obj
    for _ in range(_MAX_WRAPPER_DEPTH):
        if any(hasattr(node, name) for name in _ENCODING_ATTRS):
            return node
        for name in _WRAPPED_ATTRS:
            inner = getattr(node, name, None)
            if inner is not None and inner is not node:
                node = inner
                break
        else:
            break
    raise ValueError(
        f"cannot resolve an encoding identity from {type(obj).__name__}: no object "
        f"in its wrapper chain declares any of {_ENCODING_ATTRS}. Pass "
        "encoding_source= explicitly (the model is the usual answer)."
    )


def encoding_identity(source: object) -> tuple[str, ...]:
    """Encoding version of ``source``, read live on every use.

    Read at call time rather than captured at construction so that a model
    reloaded or re-tagged behind the evaluator cannot keep serving entries
    computed under the previous encoding.
    """
    return tuple(
        f"{name}={getattr(source, name, None)!r}" for name in _ENCODING_ATTRS
    )


def _namespace(source: object, *, transport: str, relations: bool) -> bytes:
    """Key prefix isolating one (encoding, transport, relations mode).

    A prefix rather than a checked-on-hit field: entries under a different
    encoding are not compared and rejected, they are unreachable.
    """
    parts = (*encoding_identity(source), f"transport={transport}", f"rel={int(relations)}")
    return hashlib.blake2b("|".join(parts).encode(), digest_size=16).digest()


@dataclass(frozen=True)
class EncodedEvalCacheStats:
    entries: int
    hits: int
    misses: int
    evictions: int
    inner_calls: int
    rows_submitted: int

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        requests = self.requests
        if requests <= 0:
            return 0.0
        return self.hits / requests


def _coerce_encoded_batch(x: np.ndarray) -> np.ndarray:
    if x.ndim != 4:
        raise ValueError(f"expected encoded batch shape (B,C,H,W), got {x.shape!r}")
    if x.dtype is np.dtype(np.float32) and x.flags["C_CONTIGUOUS"]:
        return x
    return np.ascontiguousarray(x, dtype=np.float32)


def _encoded_row_key(
    namespace: bytes, row: np.ndarray, extra: bytes = b"",
) -> _CacheKey:
    if not row.flags["C_CONTIGUOUS"]:
        row = np.ascontiguousarray(row)
    hasher = hashlib.blake2b(digest_size=16)
    hasher.update(row.tobytes(order="C"))
    if extra:
        hasher.update(b"\x00")
        hasher.update(extra)
    return namespace, tuple(int(dim) for dim in row.shape), row.dtype.str, hasher.digest()


def _coerce_compact_batch(x: np.ndarray) -> np.ndarray:
    """Contiguity-only coercion: the compact transport also carries bf16 bits.

    ``_coerce_encoded_batch``'s float32 cast would silently reinterpret a
    uint16 bf16-bits batch, so the compact path keeps the caller's dtype and
    puts it in the key instead.
    """
    if x.ndim != 4:
        raise ValueError(f"expected encoded batch shape (B,C,H,W), got {x.shape!r}")
    return x if x.flags["C_CONTIGUOUS"] else np.ascontiguousarray(x)


class EncodedEvalCache:
    """LRU cache for immutable NN outputs keyed by exact encoded input rows.

    Covers both transports the search picks between: dense
    ``evaluate_encoded`` and compact ``evaluate_legal_bf16``. The compact one
    matters because ``__getattr__`` forwarding used to hand it straight to the
    inner evaluator, so enabling compact-bf16 UCI silently turned the cache
    inert while it went on reporting a clean 0/0 hit rate (audit I5).

    The PUCV in-place path uses ``PucvEvalCache`` below because it needs
    C-emitted leaf keys plus miss compaction.

    Every key is prefixed with the encoding identity of ``encoding_source``,
    re-read on each call, so entries can never outlive the encoding that
    produced them.
    """

    def __init__(
        self,
        inner: BatchEvaluator,
        *,
        max_entries: int = 131_072,
        encoding_source: object | None = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._inner = inner
        self._encoding_source = (
            resolve_encoding_source(inner)
            if encoding_source is None
            else encoding_source
        )
        self._max_entries = int(max_entries)
        self._cache: OrderedDict[_CacheKey, _CacheValue] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._inner_calls = 0
        self._rows_submitted = 0

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def stats(self) -> EncodedEvalCacheStats:
        with self._lock:
            return EncodedEvalCacheStats(
                entries=len(self._cache),
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                inner_calls=self._inner_calls,
                rows_submitted=self._rows_submitted,
            )

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._inner_calls = 0
            self._rows_submitted = 0

    @property
    def encoding_identity(self) -> tuple[str, ...]:
        """The encoding version currently namespacing every key."""
        return encoding_identity(self._encoding_source)

    @property
    def supports_legal_bf16(self) -> bool:
        """Advertise compact support only when the inner evaluator has it.

        Defined explicitly because ``evaluate_legal_bf16`` below exists
        unconditionally, so the search's ``hasattr`` half of the transport
        test can no longer speak for the inner evaluator.
        """
        if not hasattr(self._inner, "evaluate_legal_bf16"):
            return False
        return bool(getattr(self._inner, "supports_legal_bf16", True))

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
  # Relations are a pure function of the position, which the encoded row
  # determines — but the relations MODE is the caller's, and a row evaluated
  # without relations is a different computation, so it rides in the key
  # namespace rather than in a precondition comment.
        xb = _coerce_encoded_batch(x)
        bsz = int(xb.shape[0])
        if bsz == 0:
            return self._inner.evaluate_encoded(xb)

        ns = _namespace(
            self._encoding_source, transport="dense", relations=relations is not None,
        )
        keys = [_encoded_row_key(ns, xb[i]) for i in range(bsz)]
        cached: list[_CacheValue | None] = [None] * bsz
        miss_first_pos: dict[_CacheKey, int] = {}
        miss_rows: list[np.ndarray] = []
        miss_rels: list[np.ndarray] = []

        with self._lock:
            for i, key in enumerate(keys):
                got = self._cache.get(key)
                if got is not None:
                    self._cache.move_to_end(key)
                    cached[i] = got
                    self._hits += 1
                    continue
                self._misses += 1
                if key not in miss_first_pos:
                    miss_first_pos[key] = len(miss_rows)
                    miss_rows.append(xb[i])
                    if relations is not None:
                        miss_rels.append(relations[i])

        miss_values: dict[_CacheKey, _CacheValue] = {}
        if miss_rows:
            miss_batch = np.ascontiguousarray(np.stack(miss_rows, axis=0), dtype=np.float32)
            if miss_rels:
                pol_miss, wdl_miss = self._inner.evaluate_encoded(
                    miss_batch,
                    relations=np.ascontiguousarray(np.stack(miss_rels, axis=0)),
                )
            else:
                pol_miss, wdl_miss = self._inner.evaluate_encoded(miss_batch)
            if pol_miss.shape[0] != len(miss_rows) or wdl_miss.shape[0] != len(miss_rows):
                raise ValueError(
                    "inner evaluator returned mismatched batch dimensions: "
                    f"policy={pol_miss.shape!r}, wdl={wdl_miss.shape!r}, "
                    f"expected {len(miss_rows)} rows"
                )
            with self._lock:
                self._inner_calls += 1
                self._rows_submitted += len(miss_rows)
                for key, j in miss_first_pos.items():
                    value = (np.array(pol_miss[j], copy=True), np.array(wdl_miss[j], copy=True))
                    miss_values[key] = value
                    self._cache[key] = value
                    self._cache.move_to_end(key)
                while len(self._cache) > self._max_entries:
                    self._cache.popitem(last=False)
                    self._evictions += 1

        first_value = next((item for item in cached if item is not None), None)
        if first_value is None:
            first_value = miss_values[keys[0]]
        pol_out = np.empty((bsz, *first_value[0].shape), dtype=first_value[0].dtype)
        wdl_out = np.empty((bsz, *first_value[1].shape), dtype=first_value[1].dtype)

        for i, key in enumerate(keys):
            value = cached[i]
            if value is None:
                value = miss_values[key]
            pol_out[i] = value[0]
            wdl_out[i] = value[1]
        return pol_out, wdl_out

    def evaluate_legal_bf16(
        self, x: np.ndarray, legal_flat: np.ndarray, legal_counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compact-policy transport, cached per row.

        The cached policy for a row is that row's legal slice in the caller's
        order, so the row's legal indices are part of its key: two searches
        that enumerate the same position's moves differently must not share
        an entry.

        Returns ``(policy_bits, wdl)`` with the same layout the inner
        evaluator uses — policy concatenated over rows in row order, one
        entry per legal move.
        """
        inner_call = getattr(self._inner, "evaluate_legal_bf16", None)
        if inner_call is None:
            raise AttributeError(
                f"{type(self._inner).__name__} has no evaluate_legal_bf16; the "
                "compact transport is unavailable behind this cache"
            )
        xb = _coerce_compact_batch(x)
        bsz = int(xb.shape[0])
        counts = np.asarray(legal_counts, dtype=np.int32)
        if counts.ndim != 1 or counts.shape[0] != bsz:
            raise ValueError(
                f"legal_counts must be shape ({bsz},), got {counts.shape!r}"
            )
        flat = np.ascontiguousarray(np.asarray(legal_flat, dtype=np.int32))
        offsets = np.zeros(bsz + 1, dtype=np.int64)
        np.cumsum(counts.astype(np.int64), out=offsets[1:])
        total = int(offsets[-1])
        if total != int(flat.shape[0]):
            raise ValueError(
                f"legal_flat len {int(flat.shape[0])} != sum(legal_counts) {total}"
            )
        if bsz == 0:
            return inner_call(xb, flat, counts)

        ns = _namespace(self._encoding_source, transport="legal_bf16", relations=False)
        keys = [
            _encoded_row_key(ns, xb[i], flat[offsets[i]:offsets[i + 1]].tobytes())
            for i in range(bsz)
        ]
        cached: list[_CacheValue | None] = [None] * bsz
        miss_first_pos: dict[_CacheKey, int] = {}
        miss_rows: list[np.ndarray] = []
        miss_legal: list[np.ndarray] = []

        with self._lock:
            for i, key in enumerate(keys):
                got = self._cache.get(key)
                if got is not None:
                    self._cache.move_to_end(key)
                    cached[i] = got
                    self._hits += 1
                    continue
                self._misses += 1
                if key not in miss_first_pos:
                    miss_first_pos[key] = len(miss_rows)
                    miss_rows.append(xb[i])
                    miss_legal.append(flat[offsets[i]:offsets[i + 1]])

        miss_values: dict[_CacheKey, _CacheValue] = {}
        if miss_rows:
            miss_counts = np.array(
                [row.shape[0] for row in miss_legal], dtype=np.int32,
            )
            miss_offsets = np.zeros(len(miss_rows) + 1, dtype=np.int64)
            np.cumsum(miss_counts.astype(np.int64), out=miss_offsets[1:])
            pol_miss, wdl_miss = inner_call(
                np.ascontiguousarray(np.stack(miss_rows, axis=0)),
                np.concatenate(miss_legal).astype(np.int32, copy=False),
                miss_counts,
            )
            if int(wdl_miss.shape[0]) != len(miss_rows):
                raise ValueError(
                    "inner evaluator returned mismatched batch dimensions: "
                    f"wdl={wdl_miss.shape!r}, expected {len(miss_rows)} rows"
                )
            if int(pol_miss.shape[0]) != int(miss_offsets[-1]):
                raise ValueError(
                    "inner evaluator returned mismatched compact policy length: "
                    f"policy={pol_miss.shape!r}, expected {int(miss_offsets[-1])}"
                )
            with self._lock:
                self._inner_calls += 1
                self._rows_submitted += len(miss_rows)
                for key, j in miss_first_pos.items():
                    value = (
                        np.array(pol_miss[miss_offsets[j]:miss_offsets[j + 1]], copy=True),
                        np.array(wdl_miss[j], copy=True),
                    )
                    miss_values[key] = value
                    self._cache[key] = value
                    self._cache.move_to_end(key)
                while len(self._cache) > self._max_entries:
                    self._cache.popitem(last=False)
                    self._evictions += 1

        first_value = next((item for item in cached if item is not None), None)
        if first_value is None:
            first_value = miss_values[keys[0]]
        pol_out = np.empty(total, dtype=first_value[0].dtype)
        wdl_out = np.empty((bsz, *first_value[1].shape), dtype=first_value[1].dtype)
        for i, key in enumerate(keys):
            value = cached[i]
            if value is None:
                value = miss_values[key]
            pol_out[offsets[i]:offsets[i + 1]] = value[0]
            wdl_out[i] = value[1]
        return pol_out, wdl_out


@dataclass(frozen=True)
class PucvEvalCacheStats:
    entries: int
    hits: int
    misses: int
    stores: int
    evictions: int

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        requests = self.requests
        if requests <= 0:
            return 0.0
        return self.hits / requests


def _pucv_key(row: np.ndarray) -> _PucvKey:
    return int(row[0]), int(row[1])


def _encoded_row_digest(row: np.ndarray) -> bytes:
    if not row.flags["C_CONTIGUOUS"]:
        row = np.ascontiguousarray(row, dtype=np.float32)
    return hashlib.blake2b(row.tobytes(order="C"), digest_size=16).digest()


class PucvEvalCache:
    """LRU cache for PUCV in-place batches keyed by C-emitted row fingerprints.

    The C key is used as a fast prefilter. Hit candidates compare an exact
    encoded-row digest before reusing outputs, so a fingerprint collision is
    treated as a miss.
    """

    def __init__(self, *, max_entries: int) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = int(max_entries)
        self._cache: OrderedDict[_PucvKey, _PucvValue] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._stores = 0
        self._evictions = 0

    @property
    def max_entries(self) -> int:
        return self._max_entries

    def stats(self) -> PucvEvalCacheStats:
        with self._lock:
            return PucvEvalCacheStats(
                entries=len(self._cache),
                hits=self._hits,
                misses=self._misses,
                stores=self._stores,
                evictions=self._evictions,
            )

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._stores = 0
            self._evictions = 0

    def get(
        self,
        key_pair: np.ndarray,
        encoded_row: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        key = _pucv_key(key_pair)
        if key == (0, 0):
            return None
        with self._lock:
            got = self._cache.get(key)
        if got is None:
            with self._lock:
                self._misses += 1
            return None
        digest = _encoded_row_digest(encoded_row)
        stored_digest, pol, wdl = got
        if stored_digest != digest:
            with self._lock:
                self._misses += 1
            return None
        with self._lock:
            if self._cache.get(key) is got:
                self._cache.move_to_end(key)
            self._hits += 1
        return pol, wdl

    def put(
        self,
        key_pair: np.ndarray,
        encoded_row: np.ndarray,
        pol_row: np.ndarray,
        wdl_row: np.ndarray,
    ) -> None:
        key = _pucv_key(key_pair)
        if key == (0, 0):
            return
        value: _PucvValue = (
            _encoded_row_digest(encoded_row),
            np.array(pol_row, copy=True),
            np.array(wdl_row, copy=True),
        )
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            self._stores += 1
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)
                self._evictions += 1
