"""Actual value-writer outputs through the single-copy Downside path."""
import json
import hashlib
from pathlib import Path

import numpy as np
import pytest
import zarr

from scripts import bt4_derived_wdl_sidecar as labels
from scripts import bt4_value_rewrite as value
from scripts import sf_policy_rewrite as tool
from scripts import combined_corpus_schedule as schedule
from tests.test_sf_tactical_policy import fixture
from tests.test_bt4_derived_wdl_sidecar import Session, install


def prepared(tmp_path, monkeypatch, alpha):
    args, _ = fixture(tmp_path)
    args.tactical_recipe = 'allmove-downside300'
    model = tmp_path / 'fixture.onnx'
    model.write_bytes(b'fixture model identity; fake session, no inference')
    teacher = tool.file_sha256(model)
    monkeypatch.setattr(schedule, 'TEACHER', teacher)
    class NativeNames(Session):
        def get_outputs(self):
            outputs = super().get_outputs()
            for output in outputs:
                if output.name == 'value':
                    output.name = '/output/wdl'
            return outputs
    install(monkeypatch, NativeNames())
    lab = labels.build_parser().parse_args([
        '--source', args.source, '--out', str(tmp_path / 'labels'), '--onnx', str(model),
        '--expected-source-summary-sha256', args.expected_source_summary_sha256,
        '--expected-onnx-sha256', teacher, '--wdl-output', '/output/wdl',
        '--wdl-output-kind', 'probabilities', '--max-shards', '100', '--batch-size', '2',
        '--minimum-free-gib', '0'])
    lab.invocation = str(tmp_path / "invocation")
    Path(lab.invocation).mkdir()
    labels.produce(lab)
    target = tmp_path / 'value'
    va = value.build_parser().parse_args([
        '--source', args.tactical_bt4_source, '--sf-source', args.source,
        '--wdl', str(lab.out), '--out', str(target),
        '--expected-source-summary-sha256', args.expected_bt4_summary_sha256,
        '--expected-policy-summary-sha256', args.expected_bt4_mix_sha256,
        '--expected-sf-summary-sha256', args.expected_source_summary_sha256,
        '--expected-onnx-sha256', teacher, '--wdl-output', '/output/wdl',
        '--alpha', str(alpha), '--batch-size', '2', '--minimum-free-gib', '0'])
    value.rewrite(va)
    args.downside_value_source = str(target)
    args.expected_downside_value_summary_sha256 = tool.file_sha256(target / value.DERIVE_SUMMARY)
    args.expected_downside_value_recipe_sha256 = tool.file_sha256(target / value.SUMMARY)
    return args


@pytest.mark.parametrize('alpha', [.5, 1.])
def test_actual_value_writer_then_downside_preserves_selected_values(tmp_path, monkeypatch, alpha):
    args = prepared(tmp_path, monkeypatch, alpha)
    result = tool.rewrite(args)
    parent = Path(args.downside_value_source)
    actual = Path(args.out)
    summary = json.loads((actual / value.DERIVE_SUMMARY).read_text())
    old = json.loads((parent / value.DERIVE_SUMMARY).read_text())
    assert result['selected_value_parent']['profile'] == ('V50' if alpha == .5 else 'V100')
    assert summary['value_target_postprocess'] == old['value_target_postprocess']
    assert summary['value_scheme'] == old['value_scheme']
    for spec in result['outputs']:
        name = spec['path']
        for path in (parent / name).rglob('*'):
            if path.is_file() and path.relative_to(parent / name).parts[0] != 'policy_target' and path.name != '.zattrs':
                assert path.read_bytes() == (actual / name / path.relative_to(parent / name)).read_bytes()
    # Same actual raw join and policy bytes as the standalone Downside writer.
    args.downside_value_source = None
    args.expected_downside_value_summary_sha256 = None
    args.expected_downside_value_recipe_sha256 = None
    args.out = str(tmp_path / 'standalone')
    control = tool.rewrite(args)
    for spec in control['outputs']:
        name = spec['path']
        np.testing.assert_array_equal(zarr.open_group(str(actual / name), 'r')['policy_target'][:],
                                      zarr.open_group(str(Path(args.out) / name), 'r')['policy_target'][:])


@pytest.mark.parametrize('defect', ['pin', 'policy', 'value', 'missing_chunk', 'legacy', 'forged_policy', 'forged_x'])
def test_refuses_unqualified_or_mutated_value_parent(tmp_path, monkeypatch, defect):
    args = prepared(tmp_path, monkeypatch, .5)
    shard = next(Path(args.downside_value_source).glob('*.zarr'))
    if defect == 'pin':
        args.expected_downside_value_recipe_sha256 = '0' * 64
    elif defect == 'legacy':
        args.tactical_recipe = 'legacy'
    else:
        group = zarr.open_group(str(shard), 'a')
        if defect in ('policy', 'forged_policy'):
            group['policy_target'][0] = 0
        elif defect == 'forged_x':
            group['x'][0] = 0
        elif defect == 'value':
            group['search_wdl'][0] = [1, 0, 0]
        else:
            chunk = next(p for p in (shard / 'search_wdl').iterdir() if not p.name.startswith('.'))
            chunk.unlink()
    if defect.startswith('forged_'):
        from scripts.sf_downside_value import files
        parent = Path(args.downside_value_source)
        recipe_path = parent / value.SUMMARY
        recipe = json.loads(recipe_path.read_text())
        actual = files(shard)
        for item in recipe['outputs']:
            if item['path'] == shard.name:
                item['files_manifest_sha256'] = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()
        recipe_path.write_text(json.dumps(recipe))
        summary_path = parent / value.DERIVE_SUMMARY
        summary = json.loads(summary_path.read_text())
        summary['value_target_postprocess'] = {k: v for k, v in recipe.items() if k != 'outputs'}
        summary_path.write_text(json.dumps(summary))
        args.expected_downside_value_recipe_sha256 = tool.file_sha256(recipe_path)
        args.expected_downside_value_summary_sha256 = tool.file_sha256(summary_path)
    with pytest.raises(ValueError, match=r'pin|combined value source|value parent'):
        tool.rewrite(args)
    assert not Path(args.out).exists()
