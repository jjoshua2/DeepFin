"""Tiny mocked consumer plumbing; no generator, deriver, teacher or real corpus."""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import adapt_raw_bt4_sidecars as adapter
from scripts import corpus_row_provenance as provenance
from scripts.sidecar_cache import raw_identity_cache_bytes


class Group(dict[str, Any]):
    def __init__(self, path: Path, attrs: dict[str, Any] | None = None) -> None:
        super().__init__()
        self.path = path
        self.attrs = attrs or {}

    def create_dataset(self, name: str, *, data: np.ndarray, **_kwargs: Any) -> None:
        self[name] = data.copy()
        # Only an inert marker, not a valid sidecar or corpus.
        (self.path / (name + '.fixture')).write_bytes(b'mocked output')


def fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    shards: list[list[tuple[str, int]]], *, max_index_bytes: int = 100000,
) -> tuple[Path, Any, dict[Path, Group]]:
    source = tmp_path / 'derived'
    source.mkdir()
    records = np.zeros(2, dtype=provenance.RECORD_DTYPE)
    refs_by_path = {}
    groups: dict[Path, Group] = {}
    sources = {}
    for name in ('a', 'b', 'c'):
        root, labels = tmp_path / name, tmp_path / (name + '_labels')
        root.mkdir()
        labels.mkdir()
        (root / 'w00.jsonl.zst').write_bytes(b'not a raw corpus')
        label_path = labels / adapter.raw.sidecar_name('w00.jsonl.zst')
        label_path.mkdir()
        spec = SimpleNamespace(corpus_dir=root, out_dir=labels,
                               manifest={'config_sha256': name}, manifest_sha256='manifest')
        sources[str(root)] = spec, {'w00.jsonl.zst': {'positions': 2}}
        raw_group = Group(label_path)
        policy = np.zeros((2, adapter.mix.COMPACT_POLICY_SIZE), dtype=np.float32)
        policy[:, 0] = 1
        raw_group[adapter.raw.POLICY_FIELD] = SimpleNamespace(oindex=policy)
        groups[label_path] = raw_group
    summary: dict[str, Any] = {'row_provenance': {'path_in_shard': provenance.FILENAME}, 'shards': [],
                               'realized': {'rows_written': sum(map(len, shards))}}
    for number, refs in enumerate(shards):
        path = source / f'shard_{number:06d}.zarr'
        path.mkdir()
        ref_path = path / provenance.FILENAME
        ref_path.write_bytes(b'mocked provenance')
        stamp = {'schema': provenance.SCHEMA, 'rows': len(refs),
                 'record_bytes': provenance.RECORD_DTYPE.itemsize, 'path': provenance.FILENAME,
                 'sha256': adapter.file_sha256(ref_path)}
        summary['shards'].append({'path': path.name, 'rows': len(refs), 'row_provenance': stamp})
        group = Group(path, {'derive_row_provenance': stamp})
        group.update(x=np.zeros((len(refs), 1), dtype=np.float16),
                     game_id=np.zeros(len(refs), dtype=np.int64), ply_index=np.zeros(len(refs), dtype=np.int32),
                     legal_mask=np.ones((len(refs), adapter.mix.COMPACT_POLICY_SIZE), dtype=np.uint8))
        groups[path] = group
        refs_by_path[ref_path] = [{'source_dir': str(tmp_path / name), 'source_shard': 'w00.jsonl.zst',
            'source_namespace': adapter.namespace(tmp_path / name, name), 'source_config_sha256': name,
            'source_row': row, 'input_key': '0' * 32, 'stored_input_key': '0' * 32,
            'game_id': 0, 'ply': 0, 'worker_id': 0}
            for name, row in refs]
    summary_path = source / adapter.derive.SUMMARY_NAME
    summary_path.write_text(json.dumps(summary))
    manifest_path = tmp_path / 'adapter.json'
    manifest_path.write_text(json.dumps({'schema': 1, 'sources': [], 'teacher': {},
        'derived_summary': {'path': str(summary_path), 'sha256': adapter.file_sha256(summary_path)}}))
    # Retain the real cache/get/authentication path; replace only external sources.
    inputs: Any = object.__new__(adapter.RawInputs)
    inputs.wdl = None
    inputs.track_used_rows = False
    inputs.teacher = {'onnx': {'path': 'fake.onnx', 'sha256': 'a' * 64}, 'policy_output': 'policy',
                      'providers': ['CPUExecutionProvider'], 'remap': {}}
    inputs.remap = {}
    inputs.max_rows = 2
    inputs.max_index_bytes = max_index_bytes
    inputs.index_bytes = 0
    inputs.cache_dir = None
    inputs.disk_cache = {}
    inputs.feed_cache = {}
    inputs.sources = sources
    inputs.pins = []
    inputs.verified = {}
    inputs.stable = {}
    inputs.cache = OrderedDict()
    def make_inputs(*_args: Any, track_used_rows: bool = False, **_kwargs: Any) -> Any:
        inputs.track_used_rows = track_used_rows
        return inputs
    monkeypatch.setattr(adapter, 'RawInputs', make_inputs)
    def verify(_pending: Any, **kwargs: Any) -> dict[str, Any]:
        kwargs['identity_records'][:] = records
        return {}
    monkeypatch.setattr(adapter.raw, 'verify_shard', verify)
    monkeypatch.setattr(adapter.raw, 'receipt_from_attrs', lambda *_args: {'positions': 2})
    def open_group(path: str, *, mode: str) -> Group:
        key = Path(path)
        if mode == 'w':
            key.mkdir()
            groups[key] = Group(key)
        return groups[key]
    monkeypatch.setattr(adapter.zarr, 'open_group', open_group)
    monkeypatch.setattr(adapter.provenance, 'read', lambda path, **_kwargs: refs_by_path[path])
    monkeypatch.setattr(adapter.mix, '_sidecar_identity', lambda group, *_args, **_kwargs:
                        ('fixture', np.zeros((len(group['x']), 16), dtype=np.uint8), 'key', 'policy'))
    monkeypatch.setattr(adapter.mix, '_source_game_ply_sha', lambda *_args: None)
    monkeypatch.setattr(adapter.mix, '_validate_sidecar', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapter.corpus, 'input_tensor_key', lambda _x: '0' * 32)
    return manifest_path, inputs, groups


def run_fixture(manifest: Path, out: Path) -> dict[str, Any]:
    return adapter.adapt(manifest, expected_manifest_sha256=adapter.file_sha256(manifest), out=out)


@pytest.mark.parametrize('shards', [
    [[('a', 0)], [('a', 0)]], [[('a', 0), ('a', 0)]],
    [[('a', 0)], [('b', 0)], [('c', 0)], [('a', 0)]],
])
def test_duplicate_physical_rows_refused_across_outputs_and_cache_eviction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shards: list[list[tuple[str, int]]],
) -> None:
    manifest, _inputs, _groups = fixture(tmp_path, monkeypatch, shards)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    prior = tmp_path / 'unrelated-output'
    prior.mkdir()
    (prior / 'keep').write_bytes(b'prior output')
    out = tmp_path / 'adapted'
    with pytest.raises(ValueError, match='duplicate source-qualified derived row'):
        run_fixture(manifest, out)
    assert not out.exists()
    assert not (tmp_path / 'adapted.writing' / adapter.mix.SIDECAR_SUMMARY).exists()
    assert (prior / 'keep').read_bytes() == b'prior output'
    assert all(path.read_bytes() == data for path, data in before.items())
    if len(shards) > 1:
        staged = tmp_path / 'adapted.writing' / 'shard_000000.zarr'
        assert staged.is_dir()  # Retain earlier private staging; never publish a partial result.
        assert all(path.read_bytes() == b'mocked output' for path in staged.iterdir())


def test_distinct_rows_and_namespaces_with_identical_positions_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, inputs, _groups = fixture(tmp_path, monkeypatch,
        [[('a', 0)], [('b', 0)], [('c', 0)], [('a', 1)]])
    out = tmp_path / 'adapted'
    result = run_fixture(manifest, out)
    assert result['rows'] == 4
    assert result['sidecar_shards'] == 4
    assert inputs.index_bytes <= raw_identity_cache_bytes(6, 3)
    assert not (out / '._raw_identity_cache').exists()
    assert not out.with_name(out.name + '.writing').exists()
    assert (out / adapter.mix.SIDECAR_SUMMARY).is_file()


@pytest.mark.parametrize('row', [-1, 2])
def test_physical_row_bounds_preserved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, row: int) -> None:
    manifest, _inputs, _groups = fixture(tmp_path, monkeypatch, [[('a', row)]])
    out = tmp_path / 'adapted'
    with pytest.raises(ValueError, match='physical source row outside closed shard'):
        run_fixture(manifest, out)
    assert not out.exists()


def test_used_map_reserved_before_raw_verification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    old_budget = 2 * provenance.RECORD_DTYPE.itemsize + 4096
    manifest, inputs, _groups = fixture(tmp_path, monkeypatch, [[('a', 0)]], max_index_bytes=old_budget)
    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError('raw verification ran before quota rejection')
    monkeypatch.setattr(adapter.raw, 'verify_shard', unexpected)
    out = tmp_path / 'adapted'
    with pytest.raises(ValueError, match='raw identity disk-cache budget exhausted'):
        run_fixture(manifest, out)
    assert inputs.index_bytes == 0
    assert not out.exists()


def test_namespace_authentication_precedes_used_map_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, inputs, _groups = fixture(tmp_path, monkeypatch, [[('a', 0)]])
    original = adapter.provenance.read
    def wrong_namespace(path: Path, **kwargs: Any) -> Any:
        refs = original(path, **kwargs)
        refs[0]['source_namespace'] = adapter.namespace(tmp_path / 'b', 'b')
        return refs
    monkeypatch.setattr(adapter.provenance, 'read', wrong_namespace)
    with pytest.raises(ValueError, match='source namespace/config mismatch'):
        run_fixture(manifest, tmp_path / 'adapted')
    assert inputs.index_bytes == 0
    assert not inputs.disk_cache


@pytest.mark.parametrize('track_used_rows', [False, True])
@pytest.mark.parametrize('wdl', [False, True])
def test_cache_quota_and_allocation_with_optional_wdl_and_used_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, track_used_rows: bool, wdl: bool,
) -> None:
    _manifest, inputs, _groups = fixture(tmp_path, monkeypatch, [[('a', 0)]])
    inputs.track_used_rows = track_used_rows
    inputs.wdl = {} if wdl else None
    inputs.cache_dir = tmp_path / 'cache'
    inputs.cache_dir.mkdir()
    required = 2 * (provenance.RECORD_DTYPE.itemsize + int(track_used_rows)) + 4096
    required += 2 * 32 + 4096 if wdl else 0
    inputs.max_index_bytes = required - 1
    calls = []
    original = adapter.raw.verify_shard
    def verify(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(adapter.raw, 'verify_shard', verify)
    ref = {'source_dir': str(tmp_path / 'a'), 'source_shard': 'w00.jsonl.zst',
           'source_namespace': adapter.namespace(tmp_path / 'a', 'a'), 'source_config_sha256': 'a'}
    with pytest.raises(ValueError, match='raw identity disk-cache budget exhausted'):
        inputs.get(ref)
    assert not calls
    assert not list(inputs.cache_dir.iterdir())
    inputs.max_index_bytes = required
    inputs.get(ref)
    assert calls == [1]
    index_path, _digest = inputs.disk_cache[(str(tmp_path / 'a'), 'w00.jsonl.zst')]
    assert index_path.with_suffix('.used.npy').exists() is track_used_rows
    assert index_path.with_suffix('.feeds.npy').exists() is wdl
    assert inputs.index_bytes == sum(p.stat().st_size for p in inputs.cache_dir.iterdir())
    assert inputs.index_bytes <= required
    inputs.get(ref)
    assert calls == [1]  # Cache reuse does not reset a map or reserve another allocation.
