"""Bounded authenticated chunks for the collector's existing prior-exposure API.

Metadata only. No positions, arrays, engines, models or authority are created.
The legacy flat reference route remains in core_finite_collect.exclusion_inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import stat
from typing import Any

SCHEMA = 'core_exclusion_index_v1'
CHUNK_SCHEMA = 'core_exclusion_chunk_v1'
KINDS = ('history_sha256', 'stored_input_float16_sha256')


@dataclass(frozen=True)
class _Limits:
    manifest: int = 4 << 20
    chunk: int = 32 << 20
    chunks: int = 128
    serialized_chunks: int = 512 << 20


_LIMITS = _Limits()


def _require(value: Any, message: str) -> None:
    if not value:
        raise ValueError(message)


def _sha(value: Any) -> None:
    _require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None, 'invalid exclusion SHA256')


def _integer(value: Any, *, zero: bool = False) -> int:
    _require(type(value) is int and value >= (0 if zero else 1), 'invalid exclusion count/bytes')
    return value


def _stamp(path: Path) -> tuple[int, ...]:
    value = path.lstat()
    _require(stat.S_ISREG(value.st_mode) and not value.st_mode & 0o222, 'exclusion metadata must be readonly regular files')
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode


def _read(ref: dict[str, Any], cap: int) -> tuple[dict[str, Any], Path, tuple[int, ...]]:
    # Reuse the genuine collector reader, including duplicate-key/hash/fd checks.
    # Import is local to avoid a collector -> helper -> collector import cycle.
    from scripts import core_finite_collect as collect

    _require(type(ref) is dict and set(ref) == {'path', 'sha256'} and type(ref['path']) is str, 'invalid exclusion reference')
    _sha(ref['sha256'])
    path = Path(ref['path'])
    _require(path.is_absolute() and path.resolve() == path and not path.is_symlink()
             and not any((parent / '.git').exists() for parent in (path, *path.parents)), 'exclusion metadata must be canonical and external')
    before = _stamp(path)
    _require(before[2] <= cap, 'exclusion metadata byte cap')
    value = collect.checked(ref)
    _require(type(value) is dict and _stamp(path) == before, 'exclusion metadata changed during read')
    return value, path, before


def _load(envelope: dict[str, Any], limits: _Limits) -> dict[str, Any]:
    _require(type(envelope) is dict and set(envelope) == {'schema', 'manifest'} and envelope['schema'] == SCHEMA,
             'unsupported chunked exclusion envelope')
    _require(all(type(value) is int for value in (limits.manifest, limits.chunk, limits.chunks, limits.serialized_chunks))
             and 0 < limits.manifest <= _LIMITS.manifest and 0 < limits.chunk <= _LIMITS.chunk
             and 0 < limits.chunks <= _LIMITS.chunks and 0 < limits.serialized_chunks <= _LIMITS.serialized_chunks,
             'invalid exclusion limits')
    manifest, manifest_path, manifest_stamp = _read(envelope['manifest'], limits.manifest)
    _require(set(manifest) == {'schema', 'chunks', 'unique_counts', 'serialized_chunk_bytes', 'coverage_scope', 'donor_pretraining_exposure'}
             and manifest['schema'] == SCHEMA and type(manifest['coverage_scope']) is str and manifest['coverage_scope'].strip()
             and manifest['donor_pretraining_exposure'] == 'UNKNOWN', 'invalid exclusion manifest')
    chunks = manifest['chunks']
    counts = manifest['unique_counts']
    _require(type(chunks) is list and len(chunks) <= limits.chunks and type(counts) is dict and set(counts) == set(KINDS),
             'invalid exclusion chunk/count roster')
    for value in counts.values():
        _integer(value, zero=True)
    total = _integer(manifest['serialized_chunk_bytes'], zero=True)
    _require(total <= limits.serialized_chunks, 'exclusion serialized chunk total cap')
    paths: set[str] = set()
    declared_bytes = 0
    declared_counts = dict.fromkeys(KINDS, 0)
    previous_kind = 0
    for item in chunks:
        _require(type(item) is dict and set(item) == {'kind', 'path', 'sha256', 'bytes', 'count'}
                 and item['kind'] in KINDS, 'invalid exclusion chunk descriptor')
        kind = KINDS.index(item['kind'])
        _require(kind >= previous_kind, 'exclusion chunk kind order differs')
        previous_kind = kind
        _require(type(item['path']) is str and item['path'] not in paths and item['path'] != str(manifest_path),
                 'duplicate/recursive exclusion chunk path')
        paths.add(item['path'])
        _sha(item['sha256'])
        size = _integer(item['bytes'])
        _require(size <= limits.chunk, 'exclusion chunk byte cap')
        declared_bytes += size
        _require(declared_bytes <= limits.serialized_chunks, 'exclusion serialized chunk total cap')
        declared_counts[item['kind']] += _integer(item['count'])
    _require(declared_bytes == total and declared_counts == counts, 'exclusion declared totals differ')
    values: dict[str, list[str]] = {kind: [] for kind in KINDS}
    seen: set[str] = set()
    custody = [(manifest_path, manifest_stamp, envelope['manifest'])]
    for item in chunks:
        ref = {'path': item['path'], 'sha256': item['sha256']}
        body, path, before = _read(ref, min(limits.chunk, item['bytes']))
        _require(before[2] == item['bytes'] and set(body) == {'schema', 'kind', 'values'}
                 and body['schema'] == CHUNK_SCHEMA and body['kind'] == item['kind']
                 and type(body['values']) is list and len(body['values']) == item['count'], 'exclusion chunk schema/count/bytes differ')
        rows = body['values']
        _require(rows, 'empty exclusion chunk')
        for value in rows:
            _sha(value)
            _require(value not in seen and (not values[item['kind']] or values[item['kind']][-1] < value),
                     'duplicate/unsorted exclusion identity')
            seen.add(value)
            values[item['kind']].append(value)
        custody.append((path, before, ref))
    _require({kind: len(rows) for kind, rows in values.items()} == counts, 'exclusion observed unique totals differ')
    # No partial result escapes. Recheck every bounded input before returning the
    # same flat membership interface used by the old collision rules.
    for path, before, ref in custody:
        _require(_stamp(path) == before, 'exclusion metadata changed before return')
        _, observed_path, observed_stamp = _read(ref, before[2])
        _require(observed_path == path and observed_stamp == before, 'exclusion metadata changed before return')
    return {**values, 'coverage_scope': manifest['coverage_scope'], 'donor_pretraining_exposure': 'UNKNOWN'}


def load(envelope: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the complete fixed index; never truncate, dedup or fallback."""
    return _load(envelope, _LIMITS)
