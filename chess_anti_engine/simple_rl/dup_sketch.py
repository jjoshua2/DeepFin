"""Constant-memory estimate of the cross-generation duplicate-row rate.

Threshold sampling on the 64-bit row identity ``x_hash`` (a blake2b digest of
the encoded input, so uniformly distributed): the sketch keeps EVERY earlier
hash below a threshold ``T``, in a sorted unique array of at most ``budget``
entries. When an update would exceed the budget, ``T`` is halved and the
hashes at or above it are dropped, so the array always holds exactly the set
``{earlier hashes} ∩ [0, T)``.

A generation's rows with ``hash < T`` are a sample whose inclusion depends
only on the hash value, not on whether the row repeats an earlier one, so
``hits / sampled`` estimates the share of the generation's rows that repeat
an earlier generation's row. Identical rows share a hash and enter or leave
the sample together, so the binomial error bar is a lower bound when rows
repeat within the generation too.

Resume safety: generation g reads the sketch written THROUGH generation g-1
and writes its own file; a redo of g reads the same g-1 file again and never
sees its own hashes. Files older than the previous generation are pruned.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

SKETCH_DIR = "dup_sketch"
FULL = np.uint64(0xFFFF_FFFF_FFFF_FFFF)
DEFAULT_BUDGET = 4_000_000  # ~32 MB of uint64
PLY_BUCKETS: tuple[tuple[int, int | None], ...] = ((0, 3), (4, 9), (10, 19), (20, 39), (40, None))


def bucket_name(lo: int, hi: int | None) -> str:
    return f"{lo}+" if hi is None else f"{lo}-{hi}"


class DupSketch:
    def __init__(self, hashes: np.ndarray | None = None, threshold: np.uint64 = FULL,
                 budget: int = DEFAULT_BUDGET) -> None:
        if int(budget) <= 0:
            raise ValueError("sketch budget must be positive")
        self.budget = int(budget)
        self.threshold = np.uint64(threshold)
        self.hashes = np.zeros(0, np.uint64) if hashes is None else np.asarray(hashes, dtype=np.uint64)

    def sampled(self, hashes: np.ndarray) -> np.ndarray:
        """Mask of the rows in the sample (hash below the current threshold;
        the full threshold samples every row)."""
        h = np.asarray(hashes, dtype=np.uint64)
        return np.ones(h.shape, bool) if self.threshold == FULL else h < self.threshold

    def contains(self, hashes: np.ndarray) -> np.ndarray:
        h = np.asarray(hashes, dtype=np.uint64)
        if self.hashes.size == 0:
            return np.zeros(h.shape, bool)
        idx = np.clip(np.searchsorted(self.hashes, h), 0, self.hashes.size - 1)
        return self.hashes[idx] == h

    def add(self, hashes: np.ndarray) -> None:
        h = np.asarray(hashes, dtype=np.uint64)
        h = h[self.sampled(h)]
        merged = np.union1d(self.hashes, h)
        while merged.size > self.budget:
            # Halve the threshold (from "everything" the first cut is 2**63).
            self.threshold = np.uint64(1 << 63) if self.threshold == FULL else self.threshold // np.uint64(2)
            merged = merged[merged < self.threshold]
        self.hashes = merged

    def estimate(self, hashes: np.ndarray, ply_after_book: np.ndarray | None = None) -> dict[str, Any]:
        """Duplicate rate of ``hashes`` against the sketch (earlier generations)."""
        h = np.asarray(hashes, dtype=np.uint64)
        s = self.sampled(h)
        hit = self.contains(h) & s
        out = _rate(int(s.sum()), int(hit.sum()))
        out.update(threshold_fraction=float(int(self.threshold) / float(int(FULL))), sketch_size=int(self.hashes.size),
                   budget=self.budget, rows=int(h.size))
        if ply_after_book is not None:
            ply = np.asarray(ply_after_book)
            by: dict[str, Any] = {}
            for lo, hi in PLY_BUCKETS:
                m = (ply >= lo) if hi is None else ((ply >= lo) & (ply <= hi))
                by[bucket_name(lo, hi)] = _rate(int((s & m).sum()), int((hit & m).sum()))
            out["by_ply_after_book"] = by
        return out

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(path).with_suffix(".tmp.npz")
        np.savez(tmp, hashes=self.hashes, threshold=np.asarray(self.threshold, dtype=np.uint64),
                 budget=np.asarray(self.budget))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path, *, budget: int) -> DupSketch:
        with np.load(path) as z:
            sketch = cls(z["hashes"], np.uint64(z["threshold"]), budget=int(z["budget"]))
        if sketch.budget != int(budget):
            sketch.budget = int(budget)
            sketch.add(np.zeros(0, np.uint64))  # re-apply a smaller budget
        return sketch


def _rate(sampled: int, hits: int) -> dict[str, Any]:
    if sampled == 0:
        return {"sampled_rows": 0, "hits": 0, "rate": math.nan, "se": math.nan}
    p = hits / sampled
    return {"sampled_rows": sampled, "hits": hits, "rate": p, "se": math.sqrt(max(p * (1 - p), 0.0) / sampled)}


def sketch_path(run_dir: Path, generation: int) -> Path:
    return Path(run_dir) / SKETCH_DIR / f"through_gen_{generation:04d}.npz"


def prune(run_dir: Path, keep_from: int) -> None:
    for p in (Path(run_dir) / SKETCH_DIR).glob("through_gen_*.npz"):
        if int(p.stem.rsplit("_", 1)[1]) < keep_from:
            p.unlink(missing_ok=True)
