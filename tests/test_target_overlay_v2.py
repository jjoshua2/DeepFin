"""Real multi-root policy/value inheritance and exact training parity."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pytest
import zarr

from chess_anti_engine.replay import target_overlay as storage
from chess_anti_engine.replay import target_overlay_v2 as targets
from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer
from scripts import target_overlay_storage as cli
from tests.test_bt4_policy_mix import _write_source


@pytest.fixture(autouse=True)
def threads() -> Any:
    import numcodecs.blosc as blosc
    previous = blosc.set_nthreads(2)
    try:
        yield
    finally:
        blosc.set_nthreads(previous)


def fixture(tmp_path: Path, replacements: tuple[str, ...] = ('policy_target', 'search_wdl')) -> tuple[list[Path], list[Path], list[Path], dict[str, str]]:
    bases, roots, copies = [], [], []
    for i in range(2):
        work = tmp_path / str(i)
        work.mkdir()
        base, shard, _ = _write_source(work)
        seal = work / 'seal.json'
        cli.seal_base(base, seal)
        ref = {'path': str(seal), 'sha256': storage.sha(seal)}
        root = work / 'targets'
        root.mkdir()
        out = root / shard.name
        targets.begin_target_shard(shard, out, ref, replacements=replacements)
        src: Any = zarr.open_group(str(shard), mode='r')
        dst: Any = zarr.open_group(str(out), mode='a')
        legal = np.asarray(src['legal_mask'][:], dtype=np.float64)
        if 'policy_target' in replacements:
            dst['policy_target'][:] = (legal / legal.sum(axis=1, keepdims=True)).astype(np.float16)
        if 'search_wdl' in replacements:
            dst['search_wdl'][:] = np.array([[.125, .25, .625]] * 2, dtype=np.float16)
        targets.finish_target_shard(shard, out, ref, recipe={'test': 'balanced'})
        copied = work / 'copied'
        shutil.copytree(base, copied)
        c: Any = zarr.open_group(str(copied / shard.name), mode='a')
        for field in replacements:
            c[field][:] = dst[field][:]
        bases.append(base)
        roots.append(root)
        copies.append(copied)
    receipt = tmp_path / 'qualified.json'
    targets.qualify_target_roots(roots, receipt)
    return bases, roots, copies, {'path': str(receipt), 'sha256': storage.sha(receipt)}


@pytest.mark.parametrize('replacements', [('policy_target',), ('search_wdl',), ('policy_target', 'search_wdl')])
def test_multi_root_real_sampler_targets(tmp_path: Path, replacements: tuple[str, ...]) -> None:
    from scripts.lc0_control_train import stage_shards
    _, roots, copies, ref = fixture(tmp_path, replacements)
    stage, other = tmp_path / 'stage', tmp_path / 'other'
    stage_shards(roots, stage)
    stage_shards(copies, other)
    kwargs: dict[str, Any] = {'batch_size': 2, 'seed': 0, 'input_planes': 175,
        'input_history_encoding': 'lc0_root_legacy_meta', 'history_rep_fix': False,
        'mirror_augmentation': False, 'plan_workers': 2, 'load_workers': 2}
    left = GameAwareEpochBuffer(shard_dir=stage, overlay_storage_qualification=ref, **kwargs)
    right = GameAwareEpochBuffer(shard_dir=other, **kwargs)
    try:
        for _ in range(2):
            a, b = left.sample_batch_arrays(2), right.sample_batch_arrays(2)
            assert a.keys() == b.keys()
            for key in a:
                assert np.array_equal(a[key], b[key]), key
    finally:
        left.close()
        right.close()
    assert not (roots[0] / 'shard_000000.zarr' / 'x').exists()


@pytest.mark.parametrize('mutation', ['target', 'identity', 'row_history', 'membership', 'root_order'])
def test_mutations_refused(tmp_path: Path, mutation: str) -> None:
    bases, roots, _, ref = fixture(tmp_path)
    paths = [p for root in roots for p in storage.shard_paths(root)]
    if mutation == 'target':
        zarr.open_group(str(paths[0]), mode='a')['search_wdl'][0] = [1, 0, 0]
    elif mutation == 'identity':
        p = paths[0] / storage.MANIFEST
        m = json.loads(p.read_text())
        m['identity']['row_identity_sha256'] = '0' * 64
        p.write_text(json.dumps(m))
    elif mutation == 'row_history':
        zarr.open_group(str(next(bases[1].glob('shard*'))), mode='a')['game_id'][0] = 999
    elif mutation == 'membership':
        paths[1].rename(paths[1].with_name('shard_000001.zarr'))
    else:
        paths.reverse()
    with pytest.raises(ValueError, match=r"changed|differ"):
        storage.qualified_paths(ref, paths)


def test_no_chains_and_no_unapproved_arrays(tmp_path: Path) -> None:
    _, roots, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='chains'):
        cli.seal_base(roots[0], tmp_path / 'chain.json')
    with pytest.raises(ValueError, match='unsupported'):
        storage._names(['x'])


def test_real_training_matches_materialized_targets(tmp_path: Path) -> None:
    import torch
    from scripts import lc0_control_train as driver
    from tests.test_lc0_control_drivers import _tiny_config
    _, roots, copies, ref = fixture(tmp_path)
    common = ['--config', str(_tiny_config(tmp_path, selfplay={'history_rep_fix': False})),
        '--steps', '0', '--batch-size', '2', '--sampling-mode', 'game_epoch', '--epochs', '1',
        '--epoch-plan-workers', '2', '--epoch-load-workers', '2', '--train-window-steps', '1',
        '--device', 'cpu', '--no-compile', '--allow-arch-drift', '--allow-invalid-control']
    a, b = tmp_path / 'ordinary-run', tmp_path / 'overlay-run'
    assert driver.main([*common, '--shards', *map(str, copies), '--out-dir', str(a)]) == 0
    assert driver.main([*common, '--shards', *map(str, roots), '--out-dir', str(b),
        '--overlay-storage-qualification', ref['path'],
        '--expected-overlay-storage-qualification-sha256', ref['sha256']]) == 0
    left = torch.load(a / 'checkpoint.pt', map_location='cpu', weights_only=False)
    right = torch.load(b / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert all(torch.equal(value, right['model'][key]) for key, value in left['model'].items())
