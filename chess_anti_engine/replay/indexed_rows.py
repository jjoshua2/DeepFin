"""Pinned physical-row selection over unchanged replay shard paths.

No target construction, input encoding, random draws, or persistent array copies.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SelectedShard:
    source_rows: int
    indices: np.ndarray
    content_sha256: str


class IndexedArray:
    """Lazy row view; declaration inspection never decodes the wide arrays."""

    def __init__(self, source: Any, indices: np.ndarray) -> None:
        self.source = source
        self.indices = indices
        self.shape = (len(indices), *source.shape[1:])
        self.dtype = np.dtype(source.dtype)
        self.ndim = len(self.shape)
        raw_chunks = getattr(source, 'chunks', None)
        self.chunks = ((max(1, min(len(indices), raw_chunks[0])), *raw_chunks[1:])
                       if isinstance(raw_chunks, tuple) and raw_chunks else None)

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        value = self._read(self.indices)
        return np.array(value, dtype=dtype, copy=True) if copy else np.asarray(value, dtype=dtype)

    def _read(self, indices: Any) -> np.ndarray:
        if hasattr(self.source, 'get_orthogonal_selection'):
            return np.asarray(self.source.get_orthogonal_selection((indices, *[slice(None)] * (self.ndim - 1))))
        return np.asarray(self.source[indices])

    def __getitem__(self, key: Any) -> np.ndarray:
        # Row selection is translated before decode. Remaining axis operations
        # run on exactly those rows, never on a re-encoded or re-routed sample.
        if not isinstance(key, tuple):
            return self._read(self.indices[key])
        first, *rest = key
        value = self._read(self.indices[first])
        prefix = () if np.isscalar(self.indices[first]) else (slice(None),)
        return value[(*prefix, *rest)]


def selected_arrays(arrays: dict[str, Any], selection: SelectedShard, *, lazy: bool) -> dict[str, Any]:
    if int(arrays['x'].shape[0]) != selection.source_rows:
        raise ValueError('row-index source row count changed')
    result: dict[str, Any] = {}
    for name, value in arrays.items():
        shape = tuple(value.shape)
        if not shape:
            result[name] = value
        elif shape[0] == selection.source_rows:
            result[name] = (IndexedArray(value, selection.indices) if lazy
                            else np.take(value, selection.indices, axis=0))
        else:
            raise ValueError(f'row-index field {name} is not source-row aligned or scalar')
    return result


class RowIndexSelection:
    """Read the already-frozen two-half masks; retain original namespace paths."""

    def __init__(self, path: Path, sha256: str, arm: str, *, max_metadata_bytes: int = 1 << 30) -> None:
        if arm not in {'retained', 'union'}:
            raise ValueError('row-index arm must be retained or union')
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError('row-index manifest hash mismatch')
        manifest = json.loads(raw)
        if manifest.get('status') != 'PASS_FROZEN_NESTED_PHYSICAL_ROW_SELECTION_ONLY':
            raise ValueError('row-index manifest selection is incomplete')
        # Reserve persistent uint32 indices, masks/JSON, and the largest shard's
        # temporary intp flatnonzero result, which coexists with its uint32 cast.
        expected = int(manifest["small_rows" if arm == "retained" else "large_rows"])
        self.metadata_reserve_bytes = (4 * expected
            + 4 * max(int(c["source_rows"]) for c in manifest["cohorts"])
            + 8 * len(raw) + 4096 * sum(len(c["shards"]) for c in manifest["cohorts"])
            + 2 * max(int(c["bytes_per_half"]) for c in manifest["cohorts"])
            + np.dtype(np.intp).itemsize * max(
                int(shard["rows"]) for c in manifest["cohorts"] for shard in c["shards"]))
        if self.metadata_reserve_bytes >= max_metadata_bytes:
            raise ValueError("row-index metadata reserve exceeds working-set limit")
        self.manifest_sha256 = sha256
        self.arm = arm
        self.shards: dict[Path, SelectedShard] = {}
        total = 0
        for cohort in manifest['cohorts']:
            mask_path = Path(cohort['mask_path'])
            if mask_path.is_symlink():
                raise ValueError('row-index mask must not be a symlink')
            masks = mask_path.read_bytes()
            half = int(cohort['bytes_per_half'])
            rows = int(cohort['source_rows'])
            if rows <= 0 or half != (rows + 7) // 8 or len(masks) != 2 * half:
                raise ValueError('row-index mask length mismatch')
            if hashlib.sha256(masks).hexdigest() != cohort['sha256']:
                raise ValueError('row-index mask hash mismatch')
            retained = np.unpackbits(np.frombuffer(masks[:half], dtype=np.uint8), bitorder='little')
            additional = np.unpackbits(np.frombuffer(masks[half:], dtype=np.uint8), bitorder='little')
            if np.any(retained[rows:]) or np.any(additional[rows:]) or np.any(retained & additional):
                raise ValueError('row-index masks overlap or have nonzero padding')
            quota = int(cohort['rows_per_half'])
            if int(retained.sum()) != quota or int(additional.sum()) != quota:
                raise ValueError('row-index half quota mismatch')
            chosen = retained if arm == 'retained' else (retained | additional)
            offset = 0
            for shard in cohort['shards']:
                n = int(shard['rows'])
                if n < 0 or n > np.iinfo(np.uint32).max or int(shard['cohort_row_offset']) != offset or offset + n > rows:
                    raise ValueError('row-index shard offset/count mismatch')
                source = Path(shard['path']).resolve(strict=True)
                if source in self.shards:
                    raise ValueError('row-index duplicate resolved source shard')
                indices = np.flatnonzero(chosen[offset:offset + n]).astype('<u4')
                indices.setflags(write=False)
                self.shards[source] = SelectedShard(n, indices, shard['content_sha256'])
                total += len(indices)
                offset += n
            if offset != rows:
                raise ValueError('row-index source coverage mismatch')
        expected = int(manifest['small_rows' if arm == 'retained' else 'large_rows'])
        if total != expected:
            raise ValueError('row-index total quota mismatch')
        self.rows = total

    def check_paths(self, paths: list[Path]) -> None:
        resolved = [p.resolve(strict=True) for p in paths]
        if len(set(resolved)) != len(resolved) or set(resolved) != set(self.shards):
            raise ValueError('row-index staged shard coverage differs from manifest')

    def receipt(self) -> dict[str, Any]:
        return {'manifest_sha256': self.manifest_sha256, 'arm': self.arm, 'physical_rows': self.rows,
                'source_shards': len(self.shards), 'index_bytes': sum(s.indices.nbytes for s in self.shards.values()),
                'metadata_reserve_bytes': self.metadata_reserve_bytes,
                'persistent_array_copies': False, 'namespace': 'original_resolved_source_parent+game_id'}
