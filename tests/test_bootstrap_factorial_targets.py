import copy
from typing import Any

import numpy as np
import pytest

from scripts.bootstrap_factorial_targets import mixed_targets, validate_base


def test_preserves_pretempered_bt4_and_calibrated_thirds():
    # BT4's stored 80/20 must remain 80/20, not become 94/6 by double sharpening.
    legal = np.array([[1, 1, 0]], dtype=np.uint8)
    p, w = mixed_targets(np.array([[.8, .2, 0]]), np.array([[.75, .25, 0]]),
                         legal, np.array([[0., 0., 100.]]),
                         np.zeros((1, 3)), np.zeros((1, 3)))
    np.testing.assert_allclose(p, [[.65, .35, 0]], atol=.0003)
    np.testing.assert_allclose(w, [[.5 + 1/9, 1/6 + 1/9, 1/9]], atol=.0003)


def test_illegal_ceres_mass_does_not_leak_and_value_heads_are_distinct():
    p, w = mixed_targets(np.array([[1., 0., 0.]]), np.array([[0., 0., 1.]]),
                         np.array([[1, 0, 1]]), np.array([[0., 1000., 0.]]),
                         np.array([[100., 0., 0.]]), np.array([[0., 100., 0.]]))
    np.testing.assert_allclose(p, [[.75, 0., .25]], atol=.0003)
    np.testing.assert_allclose(w, [[.2, .4/3, 2/3]], atol=.0003)


def test_rejects_non_v50_or_wrong_policy_temperature():
    s = {'policy_target_postprocess': {'kind': 'global', 'alpha': 1., 'bt4_temperature': .5},
         'value_target_postprocess': {'status': 'COMPLETE', 'kind': 'bt4_value_rewrite',
             'bt4_weight': .5, 'sf_weight': .5, 'mutated_arrays': ['search_wdl']}}
    validate_base(s)
    for section, key, bad in [('policy_target_postprocess', 'bt4_temperature', 1.),
                              ('value_target_postprocess', 'bt4_weight', 1.)]:
        other = copy.deepcopy(s)
        other[section][key] = bad
        with pytest.raises(ValueError, match="base must"):
            validate_base(other)


def _cohort(tmp_path, monkeypatch) -> dict[str, Any]:
    import json
    from pathlib import Path
    import zarr
    from scripts import bootstrap_factorial_targets as tool
    from scripts import target_overlay_storage
    from tests.test_ceres_derived_sidecar import Session, setup
    a = setup(tmp_path, monkeypatch)
    a.retain_value2 = True
    monkeypatch.setattr(tool.policy.ceres, 'open_teacher', lambda _: Session(tmp_path))
    tool.policy.ceres.produce(a)
    base = Path(a.source)
    summary_path = base / 'derive_targets_summary.json'
    summary = json.loads(summary_path.read_text())
    summary.update(policy_target_postprocess={'kind': 'global', 'alpha': 1., 'bt4_temperature': .5},
                   value_target_postprocess={'status': 'COMPLETE', 'kind': 'bt4_value_rewrite',
                     'bt4_weight': .5, 'sf_weight': .5, 'mutated_arrays': ['search_wdl']})
    summary_path.write_text(json.dumps(summary))
    shard = base / 'shard_000000.zarr'
    group = zarr.open_group(str(shard), mode='a')
    group['search_wdl'][:] = np.array([[.25, .5, .25]] * 32, dtype='float16')
    seal = tmp_path / 'seal.json'
    target_overlay_storage.seal_base(base, seal)
    bank_path = Path(a.out) / shard.name
    bank = zarr.open_group(str(bank_path), mode='r')
    def pin(p):
        return {'path': str(p), 'sha256': tool.policy.shared.file_sha256(p)}
    return {'base': str(base), 'base_summary': pin(summary_path), 'base_seal': pin(seal),
            'entries': [{'shard': shard.name, 'ceres': str(bank_path),
                         'ceres_binding': dict(bank.attrs)['binding']}]}


def test_real_cohort_changes_only_selected_targets(tmp_path, monkeypatch):
    from pathlib import Path
    from scripts import bootstrap_factorial_targets as tool
    from chess_anti_engine.replay.shard import load_shard_arrays
    manifest = _cohort(tmp_path, monkeypatch)
    out = tmp_path / 'outputs'
    result = tool.build_cohort(manifest, out)
    assert result['rows'] == 32
    original, meta = load_shard_arrays(Path(manifest['base']) / 'shard_000000.zarr')
    for arm, fields in tool.ARMS.items():
        actual, actual_meta = load_shard_arrays(out / arm / 'shard_000000.zarr', allow_target_overlay=True)
        assert meta == actual_meta
        for key in original:
            if key not in fields:
                np.testing.assert_array_equal(actual[key], original[key])
        assert not (out / arm / 'shard_000000.zarr' / 'x').exists()
    b, _ = load_shard_arrays(out / 'B/shard_000000.zarr', allow_target_overlay=True)
    c, _ = load_shard_arrays(out / 'C/shard_000000.zarr', allow_target_overlay=True)
    d, _ = load_shard_arrays(out / 'D/shard_000000.zarr', allow_target_overlay=True)
    np.testing.assert_array_equal(b['policy_target'], d['policy_target'])
    np.testing.assert_array_equal(c['search_wdl'], d['search_wdl'])


@pytest.mark.parametrize('mutation', ['during_read', 'missing_chunk'])
def test_teacher_corruption_never_completes(tmp_path, monkeypatch, mutation):
    from pathlib import Path
    import zarr
    from scripts import bootstrap_factorial_targets as tool
    manifest = _cohort(tmp_path, monkeypatch)
    cpath = Path(manifest['entries'][0]['ceres'])
    if mutation == 'missing_chunk':
        next(p for p in (cpath / 'value_logits').iterdir() if not p.name.startswith('.')).unlink()
    else:
        actual = tool.mixed_targets
        def corrupt(*args):
            result = actual(*args)
            bank = zarr.open_group(str(cpath), mode='a')
            bank['value_logits'][0] = [0, 0, 0]
            return result
        monkeypatch.setattr(tool, 'mixed_targets', corrupt)
    out = tmp_path / 'outputs'
    with pytest.raises(ValueError, match=r"changed|missing|digest|chunk"):
        tool.build_cohort(manifest, out)
    assert not (out / 'complete.json').exists()
