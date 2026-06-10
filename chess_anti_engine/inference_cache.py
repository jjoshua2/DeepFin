from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

from chess_anti_engine.inference import BatchEvaluator

_CacheKey: TypeAlias = tuple[tuple[int, ...], str, bytes]
_CacheValue: TypeAlias = tuple[np.ndarray, np.ndarray]
_PucvKey: TypeAlias = tuple[int, int]
_PucvValue: TypeAlias = tuple[bytes, np.ndarray, np.ndarray]


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


def _encoded_row_key(row: np.ndarray) -> _CacheKey:
    if not row.flags["C_CONTIGUOUS"]:
        row = np.ascontiguousarray(row, dtype=np.float32)
    digest = hashlib.blake2b(row.tobytes(order="C"), digest_size=16).digest()
    return tuple(int(dim) for dim in row.shape), row.dtype.str, digest


class EncodedEvalCache:
    """LRU cache for immutable NN outputs keyed by exact encoded input rows.

    This class intentionally caches only the plain ``evaluate_encoded``
    interface. The PUCV in-place path uses ``PucvEvalCache`` below because it
    needs C-emitted leaf keys plus miss compaction.
    """

    def __init__(self, inner: BatchEvaluator, *, max_entries: int = 131_072) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._inner = inner
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

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
  # Relations are a pure function of the position, which the encoded row
  # already determines — cache hits stay valid with or without them.
        xb = _coerce_encoded_batch(x)
        bsz = int(xb.shape[0])
        if bsz == 0:
            return self._inner.evaluate_encoded(xb)

        keys = [_encoded_row_key(xb[i]) for i in range(bsz)]
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
