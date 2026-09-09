"""Real tiny storage, producer and exact-consumer contracts; no teacher inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from chess_anti_engine.replay import target_overlay as storage
from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer, _shard_content_sha256
from chess_anti_engine.replay.shard import load_shard_arrays
from scripts import bt4_policy_mix as mixer
from scripts import target_overlay_storage as cli
from tests.test_bt4_policy_mix import _write_audit_receipt, _write_sidecar, _write_source


@pytest.fixture(autouse=True)
def two_blosc_threads() -> Any:
    import numcodecs.blosc as blosc
    previous = blosc.set_nthreads(2)
    try:
        yield
    finally:
        blosc.set_nthreads(previous)


def make_overlay(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    base, shard, _ = _write_source(tmp_path)
    side = _write_sidecar(tmp_path, base, shard)
    seal = tmp_path / 'base-seal.json'
    assert cli.main(['seal-base', '--shards', str(base), '--output', str(seal)]) == 0
    ref = {'path': str(seal), 'sha256': storage.sha(seal)}
    audit = _write_audit_receipt(tmp_path, scope='global')
    receipt = json.loads(audit.read_text())
    receipt['treatment_invariants']['mass_reference'] = 'normalized_total_legal_mass'
    audit.write_text(json.dumps(receipt))
    out = tmp_path / 'overlay'
    args = ['mix', '--shards', str(base), '--sidecar', str(side), '--out', str(out),
            '--scope', 'global', '--alpha', '1', '--bt4-temperature', '1',
            '--expected-rows', '2', '--expected-shards', '1',
            '--expected-source-summary-sha256', storage.sha(base / mixer.DERIVE_SUMMARY),
            '--audit-receipt', str(audit)]
    assert mixer.main([*args, '--output-storage', 'immutable-overlay',
                       '--base-storage-seal', str(seal),
                       '--expected-base-storage-seal-sha256', ref['sha256']]) == 0
    copied = tmp_path / 'copied'
    args[args.index('--out') + 1] = str(copied)
    assert mixer.main(args) == 0
    qualification = tmp_path / 'qualification.json'
    assert cli.main(['qualify-overlay', '--shards', str(out), '--output', str(qualification)]) == 0
    return base, out, copied, {'path': str(qualification), 'sha256': storage.sha(qualification)}


def test_direct_producer_composed_bytes_and_qualified_exact_epoch(tmp_path: Path) -> None:
    base, out, copied, ref = make_overlay(tmp_path)
    name = 'shard_000000.zarr'
    assert {p.name for p in (out / name).iterdir()} == {'.zgroup', '.zattrs', 'policy_target', storage.MANIFEST}
    assert not (out / name / 'x').exists()
    with pytest.raises(ValueError, match='opt-in'):
        load_shard_arrays(out / name)
    with pytest.raises(ValueError, match='admission'):
        _shard_content_sha256(out / name)
    arrays, meta = load_shard_arrays(out / name, allow_target_overlay=True)
    ordinary, ordinary_meta = load_shard_arrays(copied / name)
    assert meta == ordinary_meta
    assert arrays.keys() == ordinary.keys()
    for key in arrays:
        assert arrays[key].dtype == ordinary[key].dtype
        assert np.array_equal(arrays[key], ordinary[key]), key
    assert storage.verify_qualification(ref, out)['rows'] == 2
    kwargs: dict[str, Any] = {'batch_size': 2, 'seed': 0, 'input_planes': 175,
        'input_history_encoding': 'lc0_root_legacy_meta', 'history_rep_fix': False,
        'mirror_augmentation': False, 'plan_workers': 2, 'load_workers': 2}
    with pytest.raises(ValueError, match='admission'):
        GameAwareEpochBuffer(shard_dir=out, **kwargs)
    left = GameAwareEpochBuffer(shard_dir=out, overlay_storage_qualification=ref, **kwargs)
    right = GameAwareEpochBuffer(shard_dir=copied, **kwargs)
    assert left.plan.corpus_sha256 != right.plan.corpus_sha256
    a, b = left.sample_batch_arrays(2), right.sample_batch_arrays(2)
    assert a.keys() == b.keys()
    for key in a:
        assert np.array_equal(a[key], b[key]), key
    left.close()
    right.close()
    # This is inheritance, never hard-linking the original input file.
    assert storage.sha(base / name / '.zattrs') != storage.sha(out / name / '.zattrs')


@pytest.mark.parametrize('change', ['base_payload', 'base_history', 'target', 'manifest', 'seal', 'summary', 'membership'])
def test_qualified_dependency_mutation_refuses(tmp_path: Path, change: str) -> None:
    base, out, _, ref = make_overlay(tmp_path)
    name = 'shard_000000.zarr'
    if change == 'base_payload':
        zarr.open_group(str(base / name), mode='a')['x'][0, 0, 0, 0] = 1
    elif change == 'base_history':
        zarr.open_group(str(base / name), mode='a').attrs['history_rep_fix'] = True
    elif change == 'target':
        zarr.open_group(str(out / name), mode='a')['policy_target'][0, 0] = 0.5
    elif change == 'manifest':
        path = out / name / storage.MANIFEST
        value = json.loads(path.read_text())
        value['base_content_sha256'] = '0' * 64
        path.write_text(json.dumps(value))
    elif change == 'seal':
        (tmp_path / 'base-seal.json').write_text('{}')
    elif change == 'summary':
        (out / mixer.MIX_SUMMARY).write_text('{}')
    else:
        (out / name).rename(out / 'shard_000001.zarr')
    with pytest.raises(ValueError, match=r'changed|differs'):
        storage.verify_qualification(ref, out)


def test_real_train_cli_qualification_and_model_parity(tmp_path: Path) -> None:
    import torch
    from scripts import lc0_control_train as driver
    from tests.test_lc0_control_drivers import _tiny_config
    _, out, copied, ref = make_overlay(tmp_path)
    common = ['--config', str(_tiny_config(tmp_path, selfplay={'history_rep_fix': False})), '--steps', '0', '--batch-size', '2',
              '--sampling-mode', 'game_epoch', '--epochs', '2', '--epoch-plan-workers', '2',
              '--epoch-load-workers', '2', '--train-window-steps', '1', '--device', 'cpu',
              '--no-compile', '--allow-arch-drift', '--allow-invalid-control']
    ordinary_run, overlay_run = tmp_path / 'ordinary-run', tmp_path / 'overlay-run'
    assert driver.main([*common, '--shards', str(copied), '--out-dir', str(ordinary_run)]) == 0
    assert driver.main([*common, '--shards', str(out), '--out-dir', str(overlay_run),
                        '--overlay-storage-qualification', ref['path'],
                        '--expected-overlay-storage-qualification-sha256', ref['sha256']]) == 0
    ordinary_checkpoint = torch.load(ordinary_run / 'checkpoint.pt', map_location='cpu', weights_only=False)
    overlay_checkpoint = torch.load(overlay_run / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert all(torch.equal(value, overlay_checkpoint['model'][key])
               for key, value in ordinary_checkpoint['model'].items())
    summary = json.loads((overlay_run / 'summary.json').read_text())
    assert summary['overlay_storage_qualification'] == ref
    assert summary['steps_realized'] == 2
    assert summary['sampling']['complete'] is True
    assert summary['sampling']['rows_realized'] == 4
    assert summary['valid_control'] is False  # Storage qualification is not experiment qualification.


@pytest.mark.parametrize('mode', ['missing_pin', 'replacement', 'missing_optin', 'wrong_pin'])
def test_train_cli_refuses_unsupported_or_unqualified_storage(tmp_path: Path, mode: str) -> None:
    from scripts import lc0_control_train as driver
    from tests.test_lc0_control_drivers import _tiny_config
    _, out, _, ref = make_overlay(tmp_path)
    args = ['--config', str(_tiny_config(tmp_path, selfplay={'history_rep_fix': False})), '--shards', str(out),
            '--out-dir', str(tmp_path / 'refused-run'), '--steps', '0', '--device', 'cpu',
            '--sampling-mode', 'replacement' if mode == 'replacement' else 'game_epoch',
            '--allow-arch-drift', '--allow-invalid-control']
    if mode != 'missing_optin':
        args += ['--overlay-storage-qualification', ref['path']]
    if mode not in {'missing_pin', 'missing_optin'}:
        args += ['--expected-overlay-storage-qualification-sha256',
                 '0' * 64 if mode == 'wrong_pin' else ref['sha256']]
    with pytest.raises((ValueError, SystemExit)):
        driver.main(args)
    assert not (tmp_path / 'refused-run' / 'checkpoint.pt').exists()


def test_mutable_replay_refuses_instead_of_skipping_overlay(tmp_path: Path) -> None:
    from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
    _, out, _, _ = make_overlay(tmp_path)
    buf = object.__new__(DiskReplayBuffer)
    buf._shard_dir = out
    with pytest.raises(ValueError, match='mutable/replacement'):
        buf._scan_existing_shards()


def test_base_seal_refuses_invalid_identity_and_overlay_chains(tmp_path: Path) -> None:
    base, shard, _ = _write_source(tmp_path)
    group: Any = zarr.open_group(str(shard), mode='a')
    group['has_ply_index'][0] = 0
    with pytest.raises(ValueError, match='row identity'):
        cli.seal_base(base, tmp_path / 'seal.json')
    assert not (tmp_path / 'seal.json').exists()
    group['has_ply_index'][0] = 1
    (shard / storage.MANIFEST).write_text('{}')
    with pytest.raises(ValueError, match='chains'):
        cli.seal_base(base, tmp_path / 'seal.json')


def test_seal_stamps_refuse_same_bytes_new_inode_and_symlinks(tmp_path: Path) -> None:
    base, _, _, ref = make_overlay(tmp_path)
    target = base / 'shard_000000.zarr' / '.zattrs'
    payload = target.read_bytes()
    replacement = target.with_suffix('.replacement')
    replacement.write_bytes(payload)
    replacement.replace(target)
    with pytest.raises(ValueError, match='sealed base changed'):
        storage.verify_qualification(ref, tmp_path / 'overlay')
    target.unlink()
    target.symlink_to(tmp_path / 'copied' / 'shard_000000.zarr' / '.zattrs')
    with pytest.raises(ValueError, match='links'):
        storage.tree_stamp(base / 'shard_000000.zarr')


def test_failed_producer_retains_only_partial_no_published_overlay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    finish = storage.finish_policy_shard
    def fail_after_target(base: Path, output: Path, seal_ref: dict[str, str], *, seal: storage.BaseSeal | None = None) -> None:
        finish(base, output, seal_ref, seal=seal)
        raise RuntimeError('injected finalization failure')
    monkeypatch.setattr(storage, 'finish_policy_shard', fail_after_target)
    with pytest.raises(RuntimeError, match='injected'):
        make_overlay(tmp_path)
    assert not (tmp_path / 'overlay').exists()
    assert (tmp_path / 'overlay.writing').is_dir()
    with pytest.raises(ValueError, match='unpublished'):
        cli.qualify_overlay(tmp_path / 'overlay.writing', tmp_path / 'bad-qualification.json')
    assert not (tmp_path / 'bad-qualification.json').exists()


@pytest.mark.parametrize('mutation', ['nonfinite', 'zero_mass', 'row_shape', 'history'])
def test_rebound_malformed_target_still_fails_actual_qualification(tmp_path: Path, mutation: str) -> None:
    _, out, _, _ = make_overlay(tmp_path)
    shard = out / 'shard_000000.zarr'
    group: Any = zarr.open_group(str(shard), mode='a')
    if mutation == 'history':
        group.attrs['history_rep_fix'] = True
    elif mutation == 'row_shape':
        group['policy_target'].resize((1, 1858))
    elif mutation == 'nonfinite':
        group['policy_target'][0, 0] = np.nan
    else:
        group['policy_target'][0, :] = 0
    path = shard / storage.MANIFEST
    manifest = json.loads(path.read_text())
    manifest['target_content_sha256'] = storage._plain_content(shard / 'policy_target')
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r'changed|non-finite|sums|sum|NaN|Inf'):
        cli.qualify_overlay(out, tmp_path / 'second-qualification.json')
    assert not (tmp_path / 'second-qualification.json').exists()


def test_unsupported_producer_flags_refuse_before_creating_output(tmp_path: Path) -> None:
    args = argparse.Namespace(output_storage='immutable-overlay', scope='top-max-ties')
    with pytest.raises(ValueError, match='fresh global'):
        mixer.mix_corpus(args)
    args = argparse.Namespace(output_storage='copy', base_storage_seal=tmp_path / 'seal')
    with pytest.raises(ValueError, match='options require'):
        mixer.mix_corpus(args)


@pytest.mark.parametrize('where', ['base', 'target'])
def test_changed_dependency_after_planning_refuses_before_emission(tmp_path: Path, where: str) -> None:
    base, out, _, ref = make_overlay(tmp_path)
    buf = GameAwareEpochBuffer(shard_dir=out, overlay_storage_qualification=ref,
        batch_size=2, seed=0, input_planes=175, input_history_encoding='lc0_root_legacy_meta',
        history_rep_fix=False, mirror_augmentation=False, plan_workers=2, load_workers=2)
    path = (base if where == 'base' else out) / 'shard_000000.zarr'
    group: Any = zarr.open_group(str(path), mode='a')
    if where == 'base':
        group['x'][0, 0, 0, 0] = 1
    else:
        group['policy_target'][0, 0] = 0.25
    try:
        with pytest.raises(ValueError, match='changed'):
            buf.sample_batch_arrays(2)
    finally:
        buf.close()


@pytest.mark.parametrize('change', ['added_shard', 'staged_other_root', 'base_summary'])
def test_epoch_requires_exact_actual_qualification_and_staging(tmp_path: Path, change: str) -> None:
    import shutil
    base, out, copied, ref = make_overlay(tmp_path)
    staged = out
    if change == 'added_shard':
        shutil.copytree(copied / 'shard_000000.zarr', out / 'shard_000001.zarr')
    elif change == 'staged_other_root':
        staged = tmp_path / 'staging'
        staged.mkdir()
        (staged / 'shard_000000.zarr').symlink_to(copied / 'shard_000000.zarr')
    else:
        (base / mixer.DERIVE_SUMMARY).write_text('{}')
    with pytest.raises(ValueError, match=r'membership|paths/order|metadata changed'):
        GameAwareEpochBuffer(shard_dir=staged, overlay_storage_qualification=ref,
            batch_size=2, seed=0, input_planes=175, input_history_encoding='lc0_root_legacy_meta',
            history_rep_fix=False, mirror_augmentation=False, plan_workers=2, load_workers=2)


@pytest.mark.parametrize('entries', [1, 128])
def test_one_operation_parses_seal_once_and_indexes_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entries: int) -> None:
    root = tmp_path / 'base'
    seal_path = tmp_path / 'synthetic-seal.json'
    value = {'schema': 1, 'status': storage.BASE_STATUS, 'base': str(root),
             'shards': [{'name': f'shard_{i:06d}.zarr', 'storage_stamp': str(i)} for i in range(entries)]}
    seal_path.write_text(json.dumps(value))
    ref = {'path': str(seal_path), 'sha256': storage.sha(seal_path)}
    original = storage._read_pin
    reads = []
    def read_pin(pin: dict[str, str]) -> dict[str, Any]:
        reads.append(pin)
        return original(pin)
    monkeypatch.setattr(storage, '_read_pin', read_pin)
    # This is a metadata complexity fixture, not a fake dataset qualification.
    monkeypatch.setattr(storage, 'tree_stamp', lambda p: str(int(p.stem.split('_')[1])))
    context = storage.BaseSeal(ref)
    for _ in range(3):
        for i in range(entries):
            entry = storage.base_entry(ref, root / f'shard_{i:06d}.zarr', seal=context)
            assert entry['storage_stamp'] == str(i)
    assert len(reads) == 1
    # A current receipt identity check remains live even for identical bytes.
    replacement = tmp_path / 'replacement.json'
    replacement.write_bytes(seal_path.read_bytes())
    replacement.replace(seal_path)
    with pytest.raises(ValueError, match='seal identity changed'):
        storage.base_entry(ref, root / 'shard_000000.zarr', seal=context)


def test_preflight_and_epoch_share_operation_seal_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts.lc0_data_to_rows import shard_dir_search_wdl_coverage, shard_dir_sf_wdl_coverage
    _, out, _, ref = make_overlay(tmp_path)
    manifest = json.loads((out / 'shard_000000.zarr' / storage.MANIFEST).read_text())
    base_ref = manifest['base_seal']
    original = storage._read_pin
    reads = []
    def read_pin(pin: dict[str, str]) -> dict[str, Any]:
        if pin == base_ref:
            reads.append(pin)
        return original(pin)
    monkeypatch.setattr(storage, '_read_pin', read_pin)
    context = storage.BaseSeal(base_ref)
    assert shard_dir_sf_wdl_coverage(out, allow_target_overlay=True, overlay_seal=context) == (0, 2)
    assert shard_dir_search_wdl_coverage(out, allow_target_overlay=True, overlay_seal=context) == (2, 2)
    assert len(reads) == 1
    buf = GameAwareEpochBuffer(shard_dir=out, overlay_storage_qualification=ref,
        batch_size=2, seed=0, input_planes=175, input_history_encoding='lc0_root_legacy_meta',
        history_rep_fix=False, mirror_augmentation=False, plan_workers=2, load_workers=2)
    try:
        buf.sample_batch_arrays(2)
    finally:
        buf.close()
    assert len(reads) == 2  # A fresh, independent epoch context; no global cache.


def test_mutable_late_overlay_arrival_refuses_before_generic_skip(tmp_path: Path) -> None:
    from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
    _, out, _, _ = make_overlay(tmp_path)
    empty = tmp_path / 'initially-empty'
    empty.mkdir()
    buf = DiskReplayBuffer(4, shard_dir=empty, rng=np.random.default_rng(0), read_only=True)
    late = empty / 'shard_000000.zarr'
    late.symlink_to(out / late.name)
    with pytest.raises(ValueError, match='mutable/replacement'):
        buf._try_load_shard(late, context='late-arrival')
