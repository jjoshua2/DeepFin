"""Tiny real storage/consumer path; only ONNX inference is replaced by fake sessions."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import zarr

from scripts import ceres_value_mix as tool
from tests.test_ceres_derived_sidecar import Session as CeresSession, setup
from tests.test_bt4_derived_wdl_sidecar import Session as BT4Session, install


class ValueSession(BT4Session):
    def get_outputs(self):
        return [SimpleNamespace(name='/output/wdl', type='tensor(float16)', shape=[None, 3])]


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    ca = setup(tmp_path, monkeypatch)
    ca.pad_final_batch = True
    ca.retain_value2 = True
    monkeypatch.setattr(tool.ceres, 'open_teacher', lambda _: CeresSession(tmp_path))
    tool.ceres.produce(ca)
    wa = copy.deepcopy(ca)
    wa.gpu_mem_gb = 0  # Fake BT4 session is CPU-only; do not reacquire Ceres's retained child lease.
    wa.out = str(tmp_path / 'bt4_values')
    Path(wa.out).mkdir()
    wa.invocation = str(tmp_path / 'bt4_invocation')
    Path(wa.invocation).mkdir()
    wa.wdl_output, wa.wdl_output_kind = '/output/wdl', 'probabilities'
    install(monkeypatch, ValueSession())
    tool.wdl.produce(wa)
    sf = Path(ca.source)
    source = tmp_path / 'B100'
    shutil.copytree(sf, source)
    name = 'shard_000000.zarr'
    group: Any = zarr.open_group(str(source / name), mode='a')
    legal = group['legal_mask'][:]
    b = tool.policy.bt4._tempered_bt4_policy(
        np.broadcast_to(np.arange(1, 1859), legal.shape), legal, temperature=.5)
    group['policy_target'][:] = b.astype(np.float16)
    group.attrs.update(policy_target_mix_kind='global', policy_target_mix_alpha=1.,
                       policy_target_mix_bt4_temperature=.5)
    original: dict[str, Any] = json.loads((sf / tool.DERIVE_SUMMARY).read_text())
    recipe = {'kind': 'global', 'algorithm': 'legal-normalized-global-arithmetic-v1',
        'alpha': 1., 'bt4_temperature': .5, 'rows': 32, 'expected_shards': 1,
        'source_dir': str(sf), 'source_derive_summary_sha256': wa.expected_source_summary_sha256,
        'mutated_arrays': ['policy_target']}
    (source / tool.POLICY_SUMMARY).write_text(json.dumps(recipe))
    (source / tool.DERIVE_SUMMARY).write_text(json.dumps({**original, 'policy_target_postprocess': recipe}))
    bg: Any = zarr.open_group(str(Path(wa.out) / name), mode='r')
    cg: Any = zarr.open_group(str(Path(ca.out) / name), mode='r')
    binding = dict(bg.attrs)['binding']
    monkeypatch.setattr(tool, 'BT4_MODEL', binding['onnx_sha256'])
    # Fixture collector actually uses this checkout; do not rewrite its binding.
    monkeypatch.setattr(tool, 'HISTORICAL_BT4_PRODUCER', binding['producer'])
    cbinding = dict(cg.attrs)['binding']
    manifest: dict[str, Any] = {'schema': 1, 'source': str(source), 'sf_source': str(sf),
        'sf_summary_sha256': wa.expected_source_summary_sha256,
        'source_summary_sha256': tool.wdl.file_sha256(source / tool.DERIVE_SUMMARY),
        'source_policy_summary_sha256': tool.wdl.file_sha256(source / tool.POLICY_SUMMARY),
        'teachers': {'bt4': {'onnx': binding['onnx'], 'model_sha256': binding['onnx_sha256'],
                            'requested_wdl': binding['requested_wdl'], 'producer': binding['producer']},
                    'ceres': {k: cbinding[k] for k in ('model_sha256', 'profile', 'backend')}},
        'entries': [{'shard': name, 'bt4': str(Path(wa.out) / name),
                     'ceres': str(Path(ca.out) / name), 'ceres_binding': cbinding}]}
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    args = tool.build_parser().parse_args(['--manifest', str(path), '--expected-manifest-sha256',
        tool.wdl.file_sha256(path), '--out', str(tmp_path / 'value_mix'), '--minimum-free-gib', '0'])
    return args, manifest


def test_probability_level_value_mixture():
    sf, bt4 = np.array([[.1, .2, .7]]), np.array([[.8, .1, .1]])
    primary, secondary = np.array([[1., 0., -1.]]), np.array([[-1., 0., 1.]])
    def independently_softmax(x, temperature):
        exp = np.exp(x / temperature)
        return exp / exp.sum(1, keepdims=True)
    expected = .5 * sf + .25 * bt4 + .15 * independently_softmax(primary, .55) + .1 * independently_softmax(secondary, 1.5)
    np.testing.assert_allclose(tool.target(sf, bt4, primary, secondary), expected)
    wrong = .5 * sf + .25 * bt4 + .25 * independently_softmax(.6 * primary / .55 + .4 * secondary / 1.5, 1.)
    assert not np.allclose(wrong, expected)


def test_rewrite_changes_only_value_at_actual_loss(tmp_path, monkeypatch):
    from chess_anti_engine.replay.shard import load_shard_arrays
    from chess_anti_engine.replay.dataset import collate_arrays
    from chess_anti_engine.train.losses import compute_loss
    args, m = fixture(tmp_path, monkeypatch)
    result = tool.rewrite(args)
    assert result['rows'] == 32
    assert result['changed_rows'] > 0
    assert result['producer_sha256'] == tool.producer_pins()
    source, dest = Path(m['source']), Path(args.out)
    assert (dest / tool.POLICY_SUMMARY).read_bytes() == (source / tool.POLICY_SUMMARY).read_bytes()
    before = tool.copies.file_map(source / 'shard_000000.zarr')
    after = tool.copies.file_map(dest / 'shard_000000.zarr')
    assert tool.omit_array(before, 'search_wdl') == tool.omit_array(after, 'search_wdl')
    loaded = [load_shard_arrays(root / 'shard_000000.zarr') for root in (source, dest)]
    assert loaded[1][1]['derive_value_scheme'] == tool.VALUE_SCHEME
    assert loaded[1][1]['derive_value_source'] == tool.value_source()
    grads, losses = [], []
    for arrays, _ in loaded:
        batch = collate_arrays(arrays, device='cpu')
        p = torch.linspace(-2, 2, 1858).repeat(32, 1).requires_grad_()
        v = torch.tensor([[.6, -.2, .3]] * 32, requires_grad=True)
        loss = compute_loss({'policy': p, 'wdl': v}, batch, search_wdl_frac=1., sf_wdl_frac=0.)
        (loss['policy_ce'] + loss['wdl_ce']).backward()
        grads.append((p.grad, v.grad))
        losses.append(loss)
        assert loss['search_wdl_effective_rows'].item() == 32
    torch.testing.assert_close(grads[0][0], grads[1][0], rtol=0, atol=0)
    assert not torch.equal(grads[0][1], grads[1][1])
    torch.testing.assert_close(losses[0]['policy_ce'], losses[1]['policy_ce'], rtol=0, atol=0)
    assert losses[0]['wdl_ce'].item() != losses[1]['wdl_ce'].item()
    derived = json.loads((dest / tool.DERIVE_SUMMARY).read_text())
    assert derived['value_target_postprocess'] == {k: v for k, v in result.items() if k != 'outputs'}


@pytest.mark.parametrize('defect', ['secondary', 'order', 'pov', 'row', 'history', 'producer',
                                   'sf_changed', 'duplicate', 'ceres_shard'])
def test_reject_teacher_or_parent_mismatch(tmp_path, monkeypatch, defect):
    args, m = fixture(tmp_path, monkeypatch)
    entry = m['entries'][0]
    bg: Any = zarr.open_group(entry['bt4'], mode='a')
    if defect == 'secondary':
        cg: Any = zarr.open_group(entry['ceres'], mode='a')
        del cg['value2_logits']
    elif defect in ('order', 'pov'):
        metadata = dict(bg.attrs['wdl'])
        metadata[defect] = ['loss', 'draw', 'win'] if defect == 'order' else 'white'
        bg.attrs['wdl'] = metadata
    elif defect == 'row':
        bg['row_index'][0] = 1
    elif defect in ('history', 'producer'):
        binding = dict(bg.attrs['binding'])
        if defect == 'history':
            binding['history_lineage'] = 'zero history'
        else:
            binding['producer'] = {'changed.py': '0' * 64}
        bg.attrs['binding'] = binding
    elif defect == 'sf_changed':
        parent: Any = zarr.open_group(str(Path(m['source']) / entry['shard']), mode='a')
        parent['search_wdl'][0] = [.5, .25, .25]
    elif defect == 'duplicate':
        m['entries'].append(copy.deepcopy(entry))
    elif defect == 'ceres_shard':
        entry['ceres_binding']['shard'] = 'shard_000999.zarr'
    Path(args.manifest).write_text(json.dumps(m))
    args.expected_manifest_sha256 = tool.wdl.file_sha256(args.manifest)
    with pytest.raises(ValueError, match=r'differ|required|malformed|arrays'):
        tool.rewrite(args)
    assert not Path(args.out).exists()


def test_failed_rewrite_cannot_publish(tmp_path, monkeypatch):
    args, _ = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(tool, 'target', lambda *a: (_ for _ in ()).throw(ValueError('injected stop')))
    with pytest.raises(ValueError, match='injected stop'):
        tool.rewrite(args)
    assert not Path(args.out).exists()
    assert not (Path(args.out + '.writing') / tool.SUMMARY).exists()
