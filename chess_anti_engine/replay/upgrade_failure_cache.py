"""Bounded, process-local memo for deterministic replay-plane repair failures.

No shard is changed or excluded. New buffer instances, changed decoded inputs,
history encodings, or target versions retry; eviction may also cause a retry.
This is not a durable quarantine or protection against a stream of new inputs.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import numpy as np

from .threat_upgrade import chunk_history_encoding


class UpgradeFailureCache:
    """Retain digests only, never caller arrays, paths, or exception objects."""

    def __init__(self, max_entries: int = 256) -> None:
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[bytes, None] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def fingerprint(
        arrs: dict[str, np.ndarray], target_planes: int, version: str,
    ) -> bytes | None:
        """Hash the actual transform inputs, not mutable filesystem metadata.

        The decoder implementation is fixed for this buffer's process lifetime;
        the explicit format tag versions this fingerprint's input contract.
        Unusual representations are not memoized. Noncontiguous inputs copy at
        most one bounded chess-position row, not an entire shard.
        """
        digest = hashlib.sha256()
        try:
            history_encoding = chunk_history_encoding(arrs)
        except (ValueError, TypeError):
            return None
        if history_encoding is not None and len(history_encoding) > 256:
            return None
        header = repr(("plane-upgrade-failure-v1", target_planes, version,
                       history_encoding)).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        for key in ("x", "x_lc0_root"):
            value = arrs.get(key)
            if value is None:
                digest.update(b"absent")
                continue
            try:
                arr = np.asarray(value)
            except (ValueError, TypeError):
                return None
            if (arr.ndim != 4 or arr.shape[2:] != (8, 8)
                    or arr.shape[1] > target_planes or arr.dtype.kind not in "fbiu"):
                return None
            metadata = repr((key, arr.shape, arr.dtype.str)).encode("ascii")
            digest.update(len(metadata).to_bytes(8, "big"))
            digest.update(metadata)
            if arr.size:
                if arr.flags.c_contiguous:
                    digest.update(arr.data.cast("B"))
                else:
                    for row in arr:
                        digest.update(np.ascontiguousarray(row).data.cast("B"))
        return digest.digest()

    def contains(self, key: bytes | None) -> bool:
        if key is None:
            return False
        with self._lock:
            if key not in self._entries:
                return False
            self._entries.move_to_end(key)
            return True

    def remember(self, key: bytes | None) -> None:
        if key is None:
            return
        with self._lock:
            self._entries[key] = None
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
