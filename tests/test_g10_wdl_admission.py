"""Real tiny derive/adapter/rank qualification; only teacher outputs are synthetic."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import adapt_raw_bt4_sidecars as adapter
from scripts import bt4_derived_wdl_sidecar as wdl
from scripts import bt4_raw_corpus_sidecar as raw
from scripts import common_input_batch as common
from scripts import sf_d9_rank_sidecar as rank
from tests.test_adapt_raw_bt4_sidecars import Teacher, pinned
from tests.test_bt4_derived_wdl_sidecar import Session, install, setup as complete_source_fixture
from tests.test_corpus_shard_selection import fixture as selected_fixture
from tests.test_derive_corpus_targets import run_derive


def write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, sort_keys=True))


def fixture(tmp_path: Path):
    source, selection, chosen = selected_fixture(tmp_path)
    # Canonical order is determined by the deriver, not manifest entry order.
    chosen['shards'].reverse()
    write(selection, chosen)
    derived = tmp_path / 'derived'
    summary = run_derive(source, derived, 'uniform-d9', '--source-shards', str(selection),
                         '--limit', '4', '--seed', '0', '--row-provenance', '--rows-per-shard', '2',
                         '--policy-observation', 'phase0', '--value-observation', 'latest-phase',
                         temp=0.0005)
    # Tiny source has a single complete d9 phase; tag the G10 parent metadata
    # for this admission fixture without pretending it is a real G10 search.
    summary['corpus']['staircase_gate'] = {'policy': 'g10'}
    write(derived / wdl.SUMMARY, summary)
    spec = raw.load_sources([f'fixture={source}'], tmp_path / 'raw_labels')[0]
    spec.out_dir.mkdir(parents=True)
    remap = raw.remap_provenance()
    teacher: dict[str, Any] = {'onnx': {'path': str(tmp_path / 'synthetic.onnx'), 'sha256': 'a' * 64},
               'policy_output': 'policy', 'providers': ['synthetic_CPU_fixture'], 'remap': remap}
    receipts = tmp_path / 'receipts.jsonl'
    names = {s['source_shard'] for s in chosen['shards']}
    for path, rows in zip(spec.inventory.shards, spec.inventory.shard_rows, strict=True):
        if path.name not in names:
            continue
        pending = raw.PendingShard(spec, path, rows, spec.out_dir / raw.sidecar_name(path.name))
        attrs = raw.label_shard(pending, sess=Teacher(), input_name='input', input_dtype=np.dtype('float32'),
                               providers=teacher['providers'], policy_name='policy',
                               onnx_path=Path(teacher['onnx']['path']), onnx_sha256='a' * 64,
                               remap_stamp=remap, batch_size=2)
        raw.append_receipt(receipts, raw.receipt_from_attrs(attrs, pending.target))
    manifest = tmp_path / 'adapter.json'
    write(manifest, {'schema': 1, 'derived_summary': pinned(derived / wdl.SUMMARY),
                    'teacher': teacher, 'sources': [{'source_dir': str(source),
                    'sidecar_dir': str(spec.out_dir), 'manifest': pinned(source / 'manifest.json'),
                    'receipts': pinned(receipts)}]})
    adapter.adapt(manifest, expected_manifest_sha256=wdl.file_sha256(manifest), out=tmp_path / 'bt4')
    assert rank.main(['--raw', str(source), '--source-shards', str(selection), '--shards', str(derived),
                      '--out', str(tmp_path / 'rank'), '--limit', '4', '--rows-per-shard', '2',
                      '--expected-rows', '3', '--expected-shards', '2',
                      '--expected-source-summary-sha256', wdl.file_sha256(derived / wdl.SUMMARY)]) == 0
    write(tmp_path / 'derived_identity.json', {'source_summary_sha256': wdl.file_sha256(derived / wdl.SUMMARY),
                                             'storage_identity': wdl.storage_identity(derived),
                                             'realized': summary['realized']})
    common.qualify({'derived_output': str(derived), 'adapted_output': str(tmp_path / 'bt4'),
                    'rank_output': str(tmp_path / 'rank'), 'source_dir': str(source),
                    'selection': pinned(selection), 'physical_rows': 4,
                    'support_drop_ceiling': 0, 'missing_result_count_ceiling': 1})
    model = tmp_path / 'teacher.onnx'
    model.write_bytes(b'synthetic value teacher')
    q = tmp_path / 'common_input_qualification.json'
    args = wdl.build_parser().parse_args([
        '--source', str(derived), '--out', str(tmp_path / 'wdl'), '--onnx', str(model),
        '--expected-source-summary-sha256', wdl.file_sha256(derived / wdl.SUMMARY),
        '--expected-onnx-sha256', wdl.file_sha256(model), '--wdl-output', 'value',
        '--wdl-output-kind', 'probabilities', '--max-shards', '2', '--batch-size', '2',
        '--minimum-free-gib', '0', '--g10-common-qualification', str(q),
        '--expected-g10-common-qualification-sha256', wdl.file_sha256(q)])
    Path(args.out).mkdir()
    args.invocation = str(tmp_path / 'invocation')
    Path(args.invocation).mkdir()
    return args, q


def test_qualified_nonprefix_g10_reaches_owned_child_without_policy_writes(tmp_path, monkeypatch):
    args, q = fixture(tmp_path)
    original = wdl.storage_identity(Path(args.source))
    policy = wdl.storage_identity(tmp_path / 'bt4')
    install(monkeypatch, Session())
    assert wdl.run(args) == 0
    receipt = json.loads(next((Path(args.out) / 'invocations').glob('*/completed.json')).read_text())
    assert receipt['rows'] == 3
    admitted = receipt['g10_common_admission']
    assert admitted['qualification'] == pinned(q)
    assert admitted['source_dir'] == args.source
    assert [s['source_shard'] for s in admitted['source_selection']['shards']] == [
        'w00-00001.jsonl.zst', 'w00-00003.jsonl.zst']
    import zarr
    for spec in receipt['selection']:
        output = zarr.open_group(str(Path(args.out) / spec['path']), mode='r')
        assert output.attrs['binding']['g10_common_admission'] == admitted
        assert set(output.array_keys()) == {'bt4_wdl_raw', 'row_index', 'game_id', 'ply_index', 'lc0_feed_sha256'}
    assert wdl.storage_identity(Path(args.source)) == original
    assert wdl.storage_identity(tmp_path / 'bt4') == policy


@pytest.mark.parametrize('bad', ['missing', 'missing_hash', 'hash', 'incomplete', 'summary',
                                 'selection', 'namespace', 'exclusions', 'provenance', 'storage'])
def test_refusal_before_session(tmp_path, monkeypatch, bad):
    args, path = fixture(tmp_path)
    q = json.loads(path.read_text())
    if bad == 'missing':
        args.g10_common_qualification = args.expected_g10_common_qualification_sha256 = None
    elif bad == 'missing_hash':
        args.expected_g10_common_qualification_sha256 = None
    elif bad == 'hash':
        args.expected_g10_common_qualification_sha256 = '0' * 64
    elif bad == 'summary':
        p = Path(args.source) / wdl.SUMMARY
        p.write_text(p.read_text() + '\n')
        args.expected_source_summary_sha256 = wdl.file_sha256(p)
    elif bad == 'storage':
        (Path(args.source) / 'unexpected').touch()
    else:
        if bad == 'incomplete':
            q['status'] = 'pending'
        elif bad == 'selection':
            q['source_selection']['shards'].reverse()
        elif bad == 'namespace':
            q['summary_pins'] = {k.replace('/derived/', '/foreign/'): v for k, v in q['summary_pins'].items()}
        elif bad == 'exclusions':
            q['omitted_rows'] += 1
        elif bad == 'provenance':
            q['source_qualified_input_sequence_sha256'] = None
        write(path, q)
        args.expected_g10_common_qualification_sha256 = wdl.file_sha256(path)
    def forbidden(*_a, **_kw):
        pytest.fail('bad admission reached a model session')
    monkeypatch.setattr(wdl, 'open_session', forbidden)
    with pytest.raises(ValueError, match=r'G10|incomplete'):
        wdl.produce(args)
    assert not list(Path(args.out).glob('shard_*.zarr*'))


def test_qualification_mutation_during_inference_refuses_publication(tmp_path, monkeypatch):
    args, q = fixture(tmp_path)
    session = Session()
    original = session.run
    def changed(names, feed):
        result = original(names, feed)
        q.write_text(q.read_text() + '\n')
        return result
    monkeypatch.setattr(session, 'run', changed)
    install(monkeypatch, session)
    with pytest.raises(ValueError, match='G10 qualification changed'):
        wdl.produce(args)
    assert not list(Path(args.out).glob('shard_*.zarr'))
    assert list(Path(args.out).glob('shard_*.writing'))


def test_output_namespace_refuses_another_batch_even_for_unwritten_ordinals(tmp_path, monkeypatch):
    a = tmp_path / 'a'
    a.mkdir()
    args, _ = fixture(a)
    args.max_shards = 1
    install(monkeypatch, Session())
    wdl.produce(args)
    b = tmp_path / 'b'
    b.mkdir()
    other, _ = fixture(b)
    other.out = args.out
    other.start_shard = 1  # Does not collide with the first batch's existing shard.
    with pytest.raises(ValueError, match='G10 output namespace differs'):
        wdl.produce(other)
    assert not (Path(args.out) / 'shard_000001.zarr').exists()



def test_output_namespace_refuses_default_mode_at_unwritten_ordinal(tmp_path, monkeypatch):
    selected = tmp_path / 'selected'
    selected.mkdir()
    args, _ = fixture(selected)
    args.max_shards = 1
    install(monkeypatch, Session())
    wdl.produce(args)
    before = wdl.storage_identity(Path(args.out))

    original = tmp_path / 'original'
    original.mkdir()
    other, _ = complete_source_fixture(original)
    other.out = args.out
    other.start_shard = 1
    summary, specs = wdl.source_inventory(other)
    assert summary['corpus']['corpus_complete'] is True
    assert other.g10_admission is None
    assert specs[0]['path'] == 'shard_000001.zarr'

    def forbidden(*_a, **_kw):
        pytest.fail('default mode reached a model session in a G10 output root')

    monkeypatch.setattr(wdl, 'open_session', forbidden)
    with pytest.raises(ValueError, match='G10 output namespace requires matching qualification'):
        wdl.produce(other)
    assert not (Path(args.out) / 'shard_000001.zarr').exists()
    assert not (Path(args.out) / 'shard_000001.zarr.writing').exists()
    assert wdl.storage_identity(Path(args.out)) == before

def test_historical_prefix_receipt_keeps_its_original_scope(tmp_path, monkeypatch):
    args, path = fixture(tmp_path)
    q = json.loads(path.read_text())
    # Represent the original prefix qualifier's schema: no selection/complement
    # fields, but pinned raw adapter inventory, rank accounting and storage proof.
    summary_path = Path(args.source) / wdl.SUMMARY
    summary = json.loads(summary_path.read_text())
    del summary['source_selection']
    write(summary_path, summary)
    rank_path = tmp_path / 'rank/sf_d9_rank_sidecar_summary.json'
    ranked = json.loads(rank_path.read_text())
    del ranked['source_selection']
    ranked['source_derive_summary_sha256'] = wdl.file_sha256(summary_path)
    write(rank_path, ranked)
    side_path = tmp_path / 'bt4/bt4_policy_sidecar_summary.json'
    side = json.loads(side_path.read_text())
    side['adapter']['derived_summary'] = pinned(summary_path)
    write(side_path, side)
    for key in ['source_selection', 'independent_rank_missing_result_rows',
                'verified_support_exclusion_rows', 'omitted_rows', 'complement_proof']:
        q.pop(key)
    q['summary_pins'] = {str(p): wdl.file_sha256(p) for p in [summary_path, rank_path, side_path]}
    q['unchanged_derived_storage_identity'] = wdl.storage_identity(Path(args.source))
    write(path, q)
    args.expected_source_summary_sha256 = wdl.file_sha256(summary_path)
    args.expected_g10_common_qualification_sha256 = wdl.file_sha256(path)
    install(monkeypatch, Session())
    wdl.produce(args)
    receipt = json.loads((Path(args.invocation) / 'child_completed.json').read_text())
    assert receipt['g10_common_admission']['source_selection'] is None
    assert receipt['g10_common_admission']['selected_raw_inventory_sha256']
