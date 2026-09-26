"""Real generator/deriver/writer paths, with CPU synthetic teacher logits only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import adapt_raw_bt4_sidecars as adapter
from scripts import bt4_policy_mix as mix
from scripts import bt4_raw_corpus_sidecar as raw
from scripts import corpus_row_provenance as provenance
from scripts.bt4_policy_dump import file_sha256, remap_provenance
from tests.test_derive_corpus_targets import history_row, run_derive, write_corpus, full_width_phase
from tests.test_derive_parallel import write_split_corpus
from tests.test_bt4_policy_mix import _write_audit_receipt


class Teacher:
    def __init__(self, offset: int = 0):
        self.offset = offset

    def run(self, outputs: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        rows = next(iter(feed.values())).shape[0]
        policy = [np.sin(np.arange(mix.COMPACT_POLICY_SIZE)[None, :] *
                       (np.arange(rows)[:, None] + self.offset + 1) * .01).astype(np.float32)]
        if outputs == ['policy', 'value']:
            win = (np.arange(rows) + self.offset + 1) / 32
            return [*policy, np.column_stack((win, np.full(rows, .25), .75-win)).astype('float32')]
        assert outputs == ['policy']
        return policy


def pinned(path: Path) -> dict[str, str]:
    return {'path': str(path.resolve()), 'sha256': file_sha256(path)}


def prepare(tmp_path: Path, *, offset: int = 0, three_shards: bool = False, wdl: bool = False) -> tuple[Path, Path, Path, dict[str, Any]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows = [history_row(game_id=i, result=None if i == 2 else 1.0) for i in range(9 if three_shards else 8)]
    for row in rows:
        block = adapter.derive.RowBank(row).full_width_block(9)
        assert block is not None
        row['phases'] = [full_width_phase(row['fen'], {9: {move: 100.0 if i < 2 else 0.0
                              for i, move in enumerate(block['order'])}})]
    source = write_split_corpus(tmp_path, rows, [3, 3, 3]) if three_shards else write_corpus(tmp_path, rows)
    derived = tmp_path / 'derived'
    run_derive(source, derived, 'uniform-d9', '--row-provenance', '--rows-per-shard', '7' if three_shards else '3',
               '--seed', '1' if three_shards else '9', temp=0.0005)
    sidecars = tmp_path / 'raw_labels'
    spec = raw.load_sources([f'source={source}'], sidecars)[0]
    spec.out_dir.mkdir(parents=True)
    remap = remap_provenance()
    teacher: dict[str, Any] = {'onnx': {'path': str(tmp_path / 'synthetic.onnx'), 'sha256': 'a' * 64},
               'policy_output': 'policy', 'providers': ['synthetic_CPU_fixture'], 'remap': remap}
    receipt_path = tmp_path / 'closed_receipts.jsonl'
    for path, count in zip(spec.inventory.shards, spec.inventory.shard_rows):
        pending = raw.PendingShard(spec, path, count, spec.out_dir / raw.sidecar_name(path.name))
        attrs = raw.label_shard(pending, sess=Teacher(offset), input_name='input', input_dtype=np.dtype(np.float32),
                                providers=teacher['providers'], policy_name='policy',
                                onnx_path=Path(teacher['onnx']['path']), onnx_sha256='a' * 64,
                                remap_stamp=remap, batch_size=128,
                                wdl_output={'output': 'value', 'kind': 'probabilities', 'dtype': 'float32'} if wdl else None)
        raw.append_receipt(receipt_path, raw.receipt_from_attrs(attrs, pending.target))
    manifest = {'schema': 1, 'derived_summary': pinned(derived / 'derive_targets_summary.json'),
                'teacher': teacher, 'sources': [{'source_dir': str(source), 'sidecar_dir': str(spec.out_dir),
                                               'manifest': pinned(source / 'manifest.json'),
                                               'receipts': pinned(receipt_path)}]}
    if wdl:
        manifest['wdl'] = {'output': 'value', 'kind': 'probabilities', 'dtype': 'float32'}
    manifest_path = tmp_path / 'adapter.json'
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, derived, spec.out_dir, manifest


def test_shuffled_dropped_rows_copy_exact_policy_and_pass_normal_admission(tmp_path: Path) -> None:
    manifest_path, derived, raw_dir, _manifest = prepare(tmp_path)
    out = tmp_path / 'adapted'
    assert adapter.main(['--manifest', str(manifest_path), '--expected-manifest-sha256', file_sha256(manifest_path),
                         '--out', str(out)]) == 0
    summary = json.loads((out / mix.SIDECAR_SUMMARY).read_text())
    assert summary['rows'] == 7
    assert summary['source_dir'] == str(derived)
    assert summary['source_shards'] == summary['sidecar_shards'] == 3
    order = []
    for path in sorted(derived.glob('shard_*.zarr')):
        source: Any = zarr.open_group(str(path), mode='r')
        refs = provenance.read(path / provenance.FILENAME, rows=len(source['x']))
        keys = mix._sidecar_identity(source, path)
        mix._validate_sidecar(out / path.name, source_path=path, source_keys=keys[1],
                              source_key_sha=keys[2], source_policy_sha=keys[3])
        target: Any = zarr.open_group(str(out / path.name), mode='r')
        for i, ref in enumerate(refs):
            original: Any = zarr.open_group(str(raw_dir / raw.sidecar_name(ref['source_shard'])), mode='r')
            np.testing.assert_array_equal(target[mix.SIDECAR_POLICY_FIELD][i], original[raw.POLICY_FIELD][ref['source_row']])
            assert ref['input_key'] != ref['stored_input_key']
            order.append(ref['game_id'])
    assert sorted(order) == [0, 1, 3, 4, 5, 6, 7]
    assert order != sorted(order)
    # Exercise the unchanged consumer's complete source/teacher admission path.
    assert mix.mix_corpus(argparse.Namespace(
        shards=derived, sidecar=out, out=tmp_path / 'mixed', alpha=1.0,
        scope='top-max-ties', bt4_temperature=1.0, near_max_ratio=0.5,
        expected_rows=7, expected_shards=3,
        expected_source_summary_sha256=file_sha256(derived / mix.DERIVE_SUMMARY),
        audit_receipt=_write_audit_receipt(tmp_path),
    )) == 0


def test_same_game_and_history_in_two_sources_do_not_collide(tmp_path: Path) -> None:
    a_path, a_derived, _a_raw, a = prepare(tmp_path / 'a', offset=0)
    b_path, b_derived, _b_raw, b = prepare(tmp_path / 'b', offset=9)
    # Reverse map order; neither bare game IDs nor identical input keys can select a source.
    for path, manifest in [(a_path, a), (b_path, b)]:
        manifest['sources'] = [*b['sources'], *a['sources']] if path == a_path else a['sources']
        path.write_text(json.dumps(manifest))
    outputs = []
    for name, path, derived in [('a', a_path, a_derived), ('b', b_path, b_derived)]:
        out = tmp_path / f'{name}_adapted'
        adapter.adapt(path, expected_manifest_sha256=file_sha256(path), out=out)
        first = sorted(derived.glob('shard_*.zarr'))[0]
        group: Any = zarr.open_group(str(out / first.name), mode='r')
        outputs.append(np.asarray(group[mix.SIDECAR_POLICY_FIELD][:]))
    assert not np.array_equal(outputs[0], outputs[1])


@pytest.mark.parametrize('field', ['input_key', 'stored_input_key'])
def test_consistently_pinned_but_wrong_history_reference_is_refused(tmp_path: Path, field: str) -> None:
    manifest_path, derived, _raw_dir, manifest = prepare(tmp_path)
    shard = sorted(derived.glob('shard_*.zarr'))[0]
    p = shard / provenance.FILENAME
    with np.load(p, allow_pickle=False) as archive:
        sources, records = archive['sources'], archive['records']
    records[field][0] = 0
    with p.open('wb') as stream:
        np.savez(stream, sources=sources, records=records)
    group: Any = zarr.open_group(str(shard), mode='r+')
    stamp = dict(group.attrs['derive_row_provenance'])
    stamp['sha256'] = file_sha256(p)
    group.attrs['derive_row_provenance'] = stamp
    summary_path = derived / 'derive_targets_summary.json'
    summary = json.loads(summary_path.read_text())
    summary['shards'][0]['row_provenance'] = stamp
    summary_path.write_text(json.dumps(summary))
    manifest['derived_summary'] = pinned(summary_path)
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / 'refused'
    with pytest.raises(ValueError, match='full-history key mismatch'):
        adapter.adapt(manifest_path, expected_manifest_sha256=file_sha256(manifest_path), out=out)
    assert not out.exists()


def test_corrupted_closed_receipt_is_refused(tmp_path: Path) -> None:
    manifest_path, _derived, _raw_dir, manifest = prepare(tmp_path)
    receipt_path = Path(manifest['sources'][0]['receipts']['path'])
    receipt = json.loads(receipt_path.read_text())
    receipt['source_sha256'] = 'f' * 64
    receipt_path.write_text(json.dumps(receipt) + '\n')
    manifest['sources'][0]['receipts'] = pinned(receipt_path)
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / 'refused'
    with pytest.raises(ValueError, match='raw receipt differs'):
        adapter.adapt(manifest_path, expected_manifest_sha256=file_sha256(manifest_path), out=out)
    assert not out.exists()


def test_raw_payload_change_after_verification_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path, _derived, _raw_dir, _manifest = prepare(tmp_path)
    verify = raw.verify_shard

    def changed(pending: Any, **kwargs: Any) -> Any:
        attrs = verify(pending, **kwargs)
        group: Any = zarr.open_group(str(pending.target), mode='r+')
        group[raw.POLICY_FIELD][0, 0] = 0.5
        return attrs

    monkeypatch.setattr(raw, 'verify_shard', changed)
    out = tmp_path / 'refused'
    with pytest.raises(ValueError, match='raw storage changed'):
        adapter.adapt(manifest_path, expected_manifest_sha256=file_sha256(manifest_path), out=out)
    assert not out.exists()


def test_source_policy_change_during_copy_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path, derived, _raw_dir, _manifest = prepare(tmp_path)
    verify = raw.verify_shard

    def changed(pending: Any, **kwargs: Any) -> Any:
        attrs = verify(pending, **kwargs)
        group: Any = zarr.open_group(str(sorted(derived.glob('shard_*.zarr'))[0]), mode='r+')
        group[mix.POLICY_FIELD][0, 0] = 0.5
        return attrs

    monkeypatch.setattr(raw, 'verify_shard', changed)
    out = tmp_path / 'refused'
    with pytest.raises(ValueError, match='derived input changed'):
        adapter.adapt(manifest_path, expected_manifest_sha256=file_sha256(manifest_path), out=out)
    assert not out.exists()


def test_three_interleaved_raw_shards_verify_once_across_output_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, derived, raw_dir, _manifest = prepare(tmp_path, three_shards=True)
    verify = raw.verify_shard
    calls: dict[str, int] = {}

    def counted(pending: Any, **kwargs: Any) -> Any:
        calls[pending.path.name] = calls.get(pending.path.name, 0) + 1
        return verify(pending, **kwargs)

    monkeypatch.setattr(raw, 'verify_shard', counted)
    out = tmp_path / 'adapted'
    adapter.adapt(manifest_path, expected_manifest_sha256=file_sha256(manifest_path), out=out)
    assert calls == {f'w00-{i:05d}.jsonl.zst': 1 for i in range(3)}
    paths = sorted(derived.glob('shard_*.zarr'))
    assert len(paths) == 2
    first_refs = provenance.read(paths[0] / provenance.FILENAME, rows=7)
    assert len({ref['source_shard'] for ref in first_refs}) == 3
    assert first_refs[0]['source_shard'] == 'w00-00002.jsonl.zst'
    for path in paths:
        source: Any = zarr.open_group(str(path), mode='r')
        refs = provenance.read(path / provenance.FILENAME, rows=len(source['x']))
        target: Any = zarr.open_group(str(out / path.name), mode='r')
        for i, ref in enumerate(refs):
            original: Any = zarr.open_group(str(raw_dir / raw.sidecar_name(ref['source_shard'])), mode='r')
            np.testing.assert_array_equal(target[mix.SIDECAR_POLICY_FIELD][i], original[raw.POLICY_FIELD][ref['source_row']])
    assert not (out / '._raw_identity_cache').exists()
