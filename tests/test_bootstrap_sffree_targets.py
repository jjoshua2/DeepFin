"""SF-free values use native teacher data and leave factorial policy unchanged."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import bootstrap_sffree_targets as tool
from scripts import bootstrap_factorial_targets as factorial


def test_policy_matches_d_and_value_has_no_sf_input():
    bp = np.array([[.8, .2, 0]])
    legal = np.array([[1, 1, 0]], dtype='uint8')
    logits = np.array([[0., 0., 100.]])
    bt4 = np.array([[.125, .25, .625]])
    heads = np.zeros((1, 3))
    p, w = tool.mixed_targets(bp, bt4, legal, logits, heads, heads)
    for v50 in [np.array([[1., 0, 0]]), np.array([[0., 0, 1.]])]:
        dp, dw = factorial.mixed_targets(bp, v50, legal, logits, heads, heads)
        np.testing.assert_array_equal(p, dp)
        assert not np.array_equal(w, dw)
    np.testing.assert_allclose(w, [[11/48, 7/24, 23/48]], atol=.0003)
    np.testing.assert_allclose(p, [[.65, .35, 0]], atol=.0003)


def cohort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    from tests.test_bootstrap_factorial_targets import _cohort
    from tests.test_bt4_derived_wdl_sidecar import Session
    manifest = _cohort(tmp_path, monkeypatch)
    base = Path(manifest['base'])
    out = tmp_path / 'native'
    out.mkdir()
    args = argparse.Namespace(source=str(base), out=str(out), onnx=str(tmp_path/'native.onnx'),
        expected_source_summary_sha256=manifest['base_summary']['sha256'],
        expected_onnx_sha256=tool.value.BT4_MODEL, wdl_output='/output/wdl',
        wdl_output_kind='probabilities', batch_size=16)
    spec: dict[str, Any] = {'path': 'shard_000000.zarr', 'rows': 32}
    tool.native.label_shard(args, spec, json.loads((base/'derive_targets_summary.json').read_text()),
        Session(), 'input', np.dtype('float32'),
        {'kind': 'probabilities', 'output': '/output/wdl', 'dtype': 'float16'}, lambda: None)
    path = out / spec['path']
    bank: Any = zarr.open_group(str(path), mode='r')
    manifest['schema'] = tool.SCHEMA
    manifest['entries'][0]['ceres_attrs_sha256'] = tool.native.file_sha256(Path(manifest['entries'][0]['ceres'])/'.zattrs')
    manifest['entries'][0].update(native_bt4=str(path), native_bt4_binding=dict(bank.attrs)['binding'],
        native_bt4_attrs_sha256=tool.native.file_sha256(path/'.zattrs'))
    return manifest


def test_real_overlay_preserves_inputs_masks_and_d_policy(tmp_path, monkeypatch):
    from chess_anti_engine.replay.shard import load_shard_arrays
    m = cohort(tmp_path, monkeypatch)
    original = tmp_path/'original'
    factorial.build_cohort(m, original)
    output = tmp_path/'sffree'
    result = tool.build_cohort(m, output, minimum_free_gib=0)
    assert result['rows'] == 32
    e, em = load_shard_arrays(output/'E/shard_000000.zarr', allow_target_overlay=True)
    d, dm = load_shard_arrays(original/'D/shard_000000.zarr', allow_target_overlay=True)
    assert em == dm
    for key in d:
        if key != 'search_wdl':
            np.testing.assert_array_equal(e[key], d[key])
    assert not np.array_equal(e['search_wdl'], d['search_wdl'])
    assert np.all(e['has_search_wdl'] == 1)
    assert not (output/'E/shard_000000.zarr/x').exists()
    assert result['recipe']['weights']['sf'] == 0


@pytest.mark.parametrize('defect', ['missing_chunk', 'wrong_teacher', 'changed_provenance', 'during_build'])
def test_bad_native_never_publishes_complete(tmp_path, monkeypatch, defect):
    m = cohort(tmp_path, monkeypatch)
    entry = m['entries'][0]
    path = Path(entry['native_bt4'])
    if defect == 'missing_chunk':
        next(p for p in (path/'bt4_wdl_raw').iterdir() if not p.name.startswith('.')).unlink()
    elif defect == 'wrong_teacher':
        entry['native_bt4_binding']['onnx_sha256'] = '0'*64
    elif defect == 'changed_provenance':
        bank: Any = zarr.open_group(str(path), mode='a')
        bank.attrs['unexpected'] = True
    else:
        mix = tool.mixed_targets
        def corrupt(*args):
            result = mix(*args)
            bank: Any = zarr.open_group(str(path), mode='a')
            bank['bt4_wdl_raw'][0] = [.25, .5, .25]
            return result
        monkeypatch.setattr(tool, 'mixed_targets', corrupt)
    output = tmp_path/'sffree'
    with pytest.raises(ValueError, match=r'chunk|contract|provenance|changed'):
        tool.build_cohort(m, output, minimum_free_gib=0)
    assert not (output/'complete.json').exists()


def test_matching_game_ids_do_not_excuse_wrong_input(tmp_path, monkeypatch):
    m = cohort(tmp_path, monkeypatch)
    entry = m['entries'][0]
    group: Any = zarr.open_group(str(Path(m['base'])/entry['shard']), mode='r')
    class WrongSource:
        def __getitem__(self, key):
            if key == 'x':
                x = np.asarray(group[key][:]).copy()
                x[0, 0, 0, 0] = 1 - x[0, 0, 0, 0]
                return x
            return group[key]
    with pytest.raises(ValueError, match='feed identity'):
        tool.verify_native(entry, WrongSource(), 32)


@pytest.mark.parametrize('mask', ['has_policy', 'has_search_wdl'])
def test_sealed_masked_base_cannot_publish_silent_unused_targets(tmp_path, monkeypatch, mask):
    from scripts import target_overlay_storage
    m = cohort(tmp_path, monkeypatch)
    group: Any = zarr.open_group(str(Path(m['base'])/'shard_000000.zarr'), mode='a')
    group[mask][0] = 0
    seal = tmp_path/'masked-base-seal.json'
    target_overlay_storage.seal_base(Path(m['base']), seal)
    m['base_seal'] = {'path': str(seal), 'sha256': tool.native.file_sha256(seal)}
    with pytest.raises(ValueError, match='full policy and value supervision'):
        tool.build_cohort(m, tmp_path/'masked-output', minimum_free_gib=0)
    assert not (tmp_path/'masked-output/complete.json').exists()
