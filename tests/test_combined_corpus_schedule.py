from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts import combined_corpus_schedule as tool


def write(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return {'path': str(path), 'sha256': tool.sha(path)}


def cohort(tmp_path: Path, name: str, raw: str = 'w00-00000.jsonl.zst') -> dict[str, Any]:
    roots = {a: tmp_path / name / a for a in tool.ARMS}
    for root in roots.values():
        (root / 'shard_000000.zarr').mkdir(parents=True)
    source = {'shards': [{'path': 'shard_000000.zarr', 'rows': 3}], 'value_scheme': {'name': 'V0'}}
    sr = write(roots['source'] / 'derive_targets_summary.json', source)
    policy = {'kind': 'global', 'algorithm': 'legal-normalized-global-arithmetic-v1', 'alpha': 1.,
              'bt4_temperature': .5, 'rows': 3, 'expected_shards': 1, 'source_dir': str(roots['source']),
              'source_derive_summary_sha256': sr['sha256'], 'mutated_arrays': ['policy_target']}
    pr = write(roots['B100'] / 'bt4_policy_mix_summary.json', policy)
    b100 = dict(source, policy_target_postprocess=policy)
    br = write(roots['B100'] / 'derive_targets_summary.json', b100)
    value = {'schema': 1, 'status': 'COMPLETE', 'kind': 'bt4_value_rewrite',
             'algorithm': 'normalized-wdl-arithmetic-float16-v1', 'sf_weight': .5, 'bt4_weight': .5,
             'wdl_order': 'WDL', 'wdl_pov': 'side_to_move', 'wdl_kind': 'probabilities',
             'wdl_output': '/output/wdl', 'onnx_sha256': tool.TEACHER, 'rows': 3, 'shards': 1,
             'source_dir': str(roots['B100']), 'sf_source_dir': str(roots['source']),
             'source_derive_summary_sha256': br['sha256'], 'source_policy_summary_sha256': pr['sha256'],
             'sf_derive_summary_sha256': sr['sha256'], 'mutated_arrays': ['search_wdl'],
             'value_scheme': 'sf-bt4-native-alpha=0.5',
             'value_source': 'stored-sf-search-and-derived-bt4-wdl;onnx=' + tool.TEACHER
                 + ';output=/output/wdl;bt4_weight=0.5',
             'outputs': source['shards']}
    vr = write(roots['V50'] / 'bt4_value_rewrite_summary.json', value)
    derived = dict(b100, value_scheme={'name': value['value_scheme'], 'source': value['value_source']},
                   value_target_postprocess={k: v for k, v in value.items() if k != 'outputs'})
    dr = write(roots['V50'] / 'derive_targets_summary.json', derived)
    write(roots['V50'] / 'bt4_policy_mix_summary.json', policy)
    q = write(tmp_path / name / 'qualification.json', {'status': 'complete', 'rows': 3, 'per_raw_shard_survivors': {raw: 3}})
    namespace = hashlib.sha256(name.encode()).hexdigest()
    ident = {'status': 'METADATA_ONLY_DISJOINT_RAW_ROSTER_AND_SOURCE_IDENTITY_MAPPING',
             'cohorts': [{'cohort': name, 'source_namespace': namespace, 'derived_summary': sr,
                          'qualification': q, 'raw_shards': [raw], 'rows': 3}]}
    ir = write(tmp_path / name / 'identity.json', ident)
    return {'id': name, 'rows': 3, 'source_namespace': namespace, 'raw_shards': [raw],
            'roots': {a: {'summary': s} for a, s in zip(tool.ARMS, [sr, br, dr], strict=True)},
            'policy_recipe': pr, 'value_recipe': vr, 'source_qualification': q, 'identity_receipt': ir, 'identity_kind': 'qualified-g10-selection'}


def manifest(cohorts: list[dict[str, Any]]) -> dict[str, Any]:
    return {'schema': 1, 'kind': 'matched-b100-sf-native50-corpus-set', 'seed': 0, 'batch_size': 512,
            'game_identity_contract': 'disjoint-whole-game-raw-shard-selections-v1',
            'runtime_pins': dict(tool.FROZEN_PINS), 'cohorts': cohorts,
            'expected_rows': 3 * len(cohorts), 'expected_shards': len(cohorts)}


@dataclass
class Record:
    path: Path
    rows: int
    game_ids: Any
    game_keys: Any
    game_counts: Any


def test_two_sources_same_basename_and_game_id_remain_distinct(tmp_path: Path) -> None:
    m = manifest([cohort(tmp_path, 'run06'), cohort(tmp_path, 'run07')])
    mapping = tool.admit(m)
    # Same source-local integer, but two actual namespaces produce distinct keys.
    records = [Record(Path(row['paths']['V50']), 3, np.array([7]), np.array([i]), np.array([3]))
               for i, row in enumerate(mapping)]
    normalized = tool.canonical_records(records, mapping, 'V50')
    assert normalized[0].path.name == normalized[1].path.name
    assert normalized[0].path != normalized[1].path
    assert [int(r.game_keys[0]) for r in normalized] == [0, 1]
    source = [replace(r, path=Path(row['paths']['source'])) for r, row in zip(records, mapping, strict=True)]
    tool.same_records(normalized, source, np)
    with pytest.raises(ValueError, match='source order'):
        tool.canonical_records(records[::-1], mapping, 'V50')
    with pytest.raises(ValueError, match='roster count'):
        tool.canonical_records(records[:1], mapping, 'V50')
    source[1] = replace(source[1], game_keys=np.array([0]))
    with pytest.raises(ValueError, match='game grouping'):
        tool.same_records(normalized, source, np)


def test_one_source_uses_same_mapping_contract(tmp_path: Path) -> None:
    c = cohort(tmp_path, 'old')
    mapping = tool.admit(manifest([c]))
    assert len(mapping) == 1
    assert tool.ordered_paths(mapping, 'B100') == [Path(c['roots']['B100']['summary']['path']).parent / 'shard_000000.zarr']


def test_overlap_is_rejected_even_under_another_output_root(tmp_path: Path) -> None:
    a, b = cohort(tmp_path, 'first'), cohort(tmp_path, 'second')
    b['source_namespace'] = a['source_namespace']
    path = Path(b['identity_receipt']['path'])
    identity = json.loads(path.read_text())
    identity['cohorts'][0]['source_namespace'] = a['source_namespace']
    b['identity_receipt'] = write(path, identity)
    with pytest.raises(ValueError, match='overlapping physical'):
        tool.admit(manifest([a, b]))


@pytest.mark.parametrize(('field', 'value'), [('bt4_weight', .25), ('onnx_sha256', '0' * 64),
                                         ('source_policy_summary_sha256', '0' * 64)])
def test_wrong_value_teacher_dose_or_parent_rejected(tmp_path: Path, field: str, value: Any) -> None:
    c = cohort(tmp_path, 'one')
    p = Path(c['value_recipe']['path'])
    recipe = json.loads(p.read_text())
    recipe[field] = value
    c['value_recipe'] = write(p, recipe)
    with pytest.raises(ValueError, match='V50 recipe'):
        tool.admit(manifest([c]))


def test_b100_cannot_smuggle_modified_source_values(tmp_path: Path) -> None:
    c = cohort(tmp_path, 'one')
    p = Path(c['roots']['B100']['summary']['path'])
    d = json.loads(p.read_text())
    d['value_scheme'] = {'name': 'already-blended'}
    c['roots']['B100']['summary'] = write(p, d)
    with pytest.raises(ValueError, match='original source/value lineage'):
        tool.admit(manifest([c]))


def test_qualification_pin_and_whole_game_binding_not_ignored(tmp_path: Path) -> None:
    c = cohort(tmp_path, 'one')
    p = Path(c['source_qualification']['path'])
    p.write_text('{}')
    with pytest.raises(ValueError, match='input pin'):
        tool.admit(manifest([c]))


def test_alias_or_unpublished_root_rejected(tmp_path: Path) -> None:
    c = cohort(tmp_path, 'one')
    root = Path(c['roots']['V50']['summary']['path']).parent
    root.with_name(root.name + '.writing').mkdir()
    with pytest.raises(ValueError, match='unpublished'):
        tool.admit(manifest([c]))


def test_actual_cli_admits_two_sources_with_seed101_and_fresh_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = manifest([cohort(tmp_path, 'first'), cohort(tmp_path, 'second')])
    m['seed'] = 101
    pin = write(tmp_path / 'manifest.json', m)
    output = tmp_path / 'admission.json'
    cmd = [str(Path(tool.__file__)), '--manifest', pin['path'],
           '--expected-manifest-sha256', pin['sha256'], '--output', str(output),
           '--deadline-unix', str(time.time() + 60)]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.delenv('PYTHONOPTIMIZE', raising=False)
    monkeypatch.setattr(sys, 'argv', cmd)
    monkeypatch.setattr(tool, 'resource_guard', lambda _root: None)
    tool.main()
    report = json.loads(output.read_text())
    assert report['seed'] == 101
    assert report['rows'] == 6
    assert report['status'] == 'PASS_CORPUS_SET_METADATA_NOT_TRAINING'
    assert report['trainer_shards']['B100'] == [str(tool.root_path(c, 'B100')) for c in m['cohorts']]
    before = output.read_bytes()
    with pytest.raises(ValueError, match='fresh'):
        tool.main()
    assert output.read_bytes() == before


def test_extra_real_shard_cannot_be_hidden_from_trainer(tmp_path: Path) -> None:
    c = cohort(tmp_path, 'one')
    (tool.root_path(c, 'B100') / 'shard_000001.zarr').mkdir()
    with pytest.raises(ValueError, match='actual shard roster'):
        tool.admit(manifest([c]))


def test_scan_witness_reuses_decoded_columns_and_restores_loader(tmp_path: Path) -> None:
    from types import SimpleNamespace

    class Column:
        def __init__(self, values: Any):
            self.values = np.asarray(values)
            self.reads = 0

        def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
            self.reads += 1
            result = np.asarray(self.values, dtype=dtype)
            return result.copy() if copy else result

    game, present = Column([8, 8, 9]), Column([True, True, True])
    path = tmp_path / 'shard_000000.zarr'
    shape_only = SimpleNamespace(shape=(3, 175, 8, 8))
    def loader(_path: Path, *, lazy: bool) -> Any:
        assert lazy
        return {'x': shape_only, 'game_id': game, 'has_game_id': present}, {}
    epoch = SimpleNamespace(load_shard_arrays=loader)
    def scan(paths: list[Path], workers: int) -> Any:
        assert workers == 2
        arrays, _ = epoch.load_shard_arrays(paths[0], lazy=True)
        ids = np.asarray(arrays['game_id'], dtype=np.int64)
        flags = np.asarray(arrays['has_game_id'], dtype=bool)
        assert flags.all()
        games, counts = np.unique(ids, return_counts=True)
        return [Record(path, 3, games, games, counts)]
    epoch._scan_shards = scan
    records, columns = tool.scan_columns(epoch, [path], [{'rows': 3}], lambda: None)
    assert game.reads == 1
    assert present.reads == 1
    assert records[0].game_ids.tolist() == [8, 9]
    assert columns[0]['rows'] == 3
    assert epoch.load_shard_arrays is loader
    with pytest.raises(ValueError, match='incomplete'):
        tool.scan_columns(epoch, [path], [{'rows': 4}], lambda: None)
    assert epoch.load_shard_arrays is loader


def test_historical_source_and_g10_share_only_the_canonical_map(tmp_path: Path) -> None:
    old, g10 = cohort(tmp_path, 'historical'), cohort(tmp_path, 'g10')
    old['identity_kind'] = 'historical-single-source'
    old['raw_shards'] = []
    qualification: dict[str, Any] = {'status': 'PASS_REGISTERED_CORPUS_QUALIFICATION', 'rows': 3,
                     'corpus': str(tool.root_path(old, 'B100')),
                     'source': {'path': str(tool.root_path(old, 'source')),
                                'derive_sha256': tool.summary_ref(old, 'source')['sha256']}}
    old['source_qualification'] = write(Path(old['source_qualification']['path']), qualification)
    assert len(tool.admit(manifest([old, g10]))) == 2
    qualification['source']['path'] = str(tool.root_path(g10, 'source'))
    old['source_qualification'] = write(Path(old['source_qualification']['path']), qualification)
    with pytest.raises(ValueError, match='historical source qualification'):
        tool.admit(manifest([old, g10]))
