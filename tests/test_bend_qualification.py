"""Cheap preflight/reuse/report contracts; no export, inference or chess traversal."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig
from native.bend_engine.neural_probe import backend, checkpoint_probe as probe
from native.bend_engine.neural_probe.adapter import Encoding
from native.bend_engine.neural_probe.backend import CHECKPOINT_FORMAT, NativeEvaluator
from native.bend_engine.neural_probe.checkpoint import LoadedCheckpoint
from native.bend_engine.neural_probe.qualification import (
    compilation_cache, protect_inputs, verified_package, workspace, write_report,
)


def loaded() -> LoadedCheckpoint:
    # Keep a tiny test-only module, but use the real v3 architecture schema.
    # No mocked package is passed to LibTorch or used for a model forward.
    cfg = ModelConfig(kind='tiny', input_history_encoding='lc0_root',
                      input_extra_features='v1', history_rep_fix=False)
    resolved = asdict(cfg)
    return LoadedCheckpoint(torch.nn.Linear(2, 2), Encoding('lc0_root', 'v1', False),
        {'checkpoint_sha256': 'a' * 64, 'weights_key': 'model',
         'arch': {'_schema_version': ARCH_SCHEMA_VERSION, **resolved},
         'resolved_model_config': resolved, 'parameter_count': 6})


def package_at(path: Path, value: LoadedCheckpoint) -> dict:
    path.write_bytes(b'Test payload; never pass this file to a real native loader.')
    manifest = {'format': CHECKPOINT_FORMAT, 'torch_version': str(torch.__version__),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'channels': 146, 'batch': 4, 'policy_width': 1858, 'row_independent': True,
                'device': 'cpu', 'dtype': 'float32', 'device_index': 0,
                'input_history_encoding': 'lc0_root', 'input_extra_features': 'v1',
                'history_rep_fix': False, **value.identity}
    path.with_suffix('.json').write_text(json.dumps(manifest))
    # Match a deserialized sidecar, without aliasing the loaded checkpoint dicts.
    return json.loads(path.with_suffix('.json').read_text())


@pytest.mark.parametrize('key', ['checkpoint_sha256', 'weights_key', 'resolved_model_config',
                                'batch', 'device', 'encoding', 'hash', 'torch_version'])
def test_reuse_rejects_other_contracts(tmp_path: Path, key: str) -> None:
    value, package = loaded(), tmp_path / 'saved.pt2'
    manifest = package_at(package, value)
    assert verified_package(package, value, batch=4, device='cpu', device_index=0) == manifest
    if key == 'checkpoint_sha256':
        manifest[key] = 'b' * 64
    elif key == 'weights_key':
        manifest[key] = 'swa_model'
    elif key == 'resolved_model_config':
        # Internally valid package metadata for a DIFFERENT checkpoint config.
        manifest['arch']['num_layers'] += 1
        manifest[key]['num_layers'] += 1
    elif key == 'batch':
        manifest[key] = 1
    elif key == 'device':
        manifest.update(device='cuda', dtype='bfloat16')
    elif key == 'encoding':
        for record in (manifest, manifest['arch'], manifest['resolved_model_config']):
            record['input_history_encoding'] = 'lc0_root_legacy_meta'
    elif key == 'hash':
        package.write_bytes(b'changed after export')
    else:
        manifest[key] = 'wrong'
    package.with_suffix('.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r'reused package|fingerprint|mismatch'):
        verified_package(package, value, batch=4, device='cpu', device_index=0)


def test_reuse_compares_serialized_configuration(tmp_path: Path) -> None:
    value = loaded()
    resolved = value.identity['resolved_model_config']
    assert isinstance(resolved, dict)
    thresholds = resolved['phase_piece_thresholds']
    assert isinstance(thresholds, tuple)
    package = tmp_path / 'saved.pt2'
    manifest = package_at(package, value)
    assert manifest['resolved_model_config']['phase_piece_thresholds'] == list(thresholds)
    assert verified_package(package, value, batch=4, device='cpu', device_index=0)['batch'] == 4


@pytest.mark.parametrize('alias', ['same', 'symlink', 'hardlink'])
def test_report_cannot_replace_package_or_manifest(tmp_path: Path, alias: str) -> None:
    for name in ('model.pt2', 'model.json'):
        source = tmp_path / name
        source.write_bytes(b'preserve this')
        report = tmp_path / ('alias-' + name)
        if alias == 'same':
            report = source
        elif alias == 'symlink':
            report.symlink_to(source)
        else:
            report.hardlink_to(source)
        with pytest.raises(ValueError, match='protected input'):
            protect_inputs(report, [source])
        assert source.read_bytes() == b'preserve this'


def test_retained_and_disposable_workspace_lifetimes(tmp_path: Path) -> None:
    kept = tmp_path / 'new-work'
    def fail_after_export():
        with workspace(kept) as work:
            (work / 'checkpoint.pt2').write_bytes(b'saved export')
            raise RuntimeError('simulated failure')
    with pytest.raises(RuntimeError, match='simulated failure'):
        fail_after_export()
    assert (kept / 'checkpoint.pt2').read_bytes() == b'saved export'
    with pytest.raises(FileExistsError), workspace(kept):
        pytest.fail('must not enter an existing directory')
    with workspace(None) as work:
        removed = work
        assert removed.exists()
    assert not removed.exists()


@pytest.mark.parametrize('existing', [None, 'existing-private-cache'])
def test_compile_environment_restored_on_failure(monkeypatch, tmp_path: Path, existing: str | None) -> None:
    if existing is None:
        monkeypatch.delenv('TORCHINDUCTOR_CACHE_DIR', raising=False)
    else:
        monkeypatch.setenv('TORCHINDUCTOR_CACHE_DIR', existing)
    def fail_in_cache():
        with compilation_cache(tmp_path):
            assert os.environ['TORCHINDUCTOR_CACHE_DIR'] == str(tmp_path / 'inductor')
            raise RuntimeError('export failed')
    with pytest.raises(RuntimeError, match='export failed'):
        fail_in_cache()
    assert os.environ.get('TORCHINDUCTOR_CACHE_DIR') == existing


def test_atomic_report_preserves_previous_result_on_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / 'report.json'
    write_report(path, {'status': 'running', 'results': [1]})
    with pytest.raises(ValueError, match='JSON compliant'):
        write_report(path, {'error': float('nan')})
    assert json.loads(path.read_text()) == {'status': 'running', 'results': [1]}
    assert not list(tmp_path.glob('.bend-report-*'))
    write_report(path, {'status': 'failed', 'results': [1, 2]})
    assert json.loads(path.read_text())['results'] == [1, 2]


def configure_main(monkeypatch, tmp_path: Path, extra: list[str]) -> tuple[Path, LoadedCheckpoint]:
    value = loaded()
    report = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['probe', '--checkpoint', str(tmp_path / 'trainer.pt'),
                                    '--report', str(report), *extra])
    monkeypatch.setattr(probe, 'load_checkpoint', lambda *a, **kw: value)
    monkeypatch.setattr(probe, 'device_snapshot', lambda *a: {'device': 'cpu', 'memory_is_reserved': False})
    monkeypatch.setattr(probe, 'tools', lambda *a: {'bun': 'bun', 'cc': 'clang', 'cxx': 'clang++'})
    monkeypatch.setattr(probe.sessions, 'check_compiler', lambda *a: {'revision': 'pinned-test'})
    return report, value


def test_preflight_cannot_claim_execution(monkeypatch, tmp_path: Path) -> None:
    report, _ = configure_main(monkeypatch, tmp_path, ['--preflight-only'])
    for name in ('export_checkpoint', 'reuse_reference', 'build_worker', 'NativeEvaluator', 'group'):
        monkeypatch.setattr(probe, name, lambda *a, **kw: pytest.fail('execution in preflight'))
    probe.main()
    result = json.loads(report.read_text())
    assert result['status'] == 'preflight_passed'
    assert result['qualification'] == 'not_run'
    assert result['results'] == []
    assert result['input_shape'] == [4, 146, 8, 8]


def test_bad_toolchain_fails_before_export_or_weight_load(monkeypatch, tmp_path: Path) -> None:
    report, _ = configure_main(monkeypatch, tmp_path, [])
    def broken(_path):
        raise ValueError('compiler source fingerprint mismatch')
    monkeypatch.setattr(probe.sessions, 'check_compiler', broken)
    monkeypatch.setattr(probe, 'load_checkpoint', lambda *a, **kw: pytest.fail('loaded too early'))
    monkeypatch.setattr(probe, 'export_checkpoint', lambda *a, **kw: pytest.fail('exported too early'))
    with pytest.raises(ValueError, match='fingerprint'):
        probe.main()
    result = json.loads(report.read_text())
    assert result['status'] == 'failed'
    assert result['failed_stage'] == 'preflight'
    assert result['qualification'] == 'not_run'


def test_export_is_retained_when_build_fails(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / 'retained'
    report, value = configure_main(monkeypatch, tmp_path, ['--work-dir', str(destination)])
    def export(_loaded, path, **_kwargs):
        package_at(path, value)
        return value.model
    def broken(*_args):
        raise RuntimeError('simulated C++ build failure')
    monkeypatch.setattr(probe, 'export_checkpoint', export)
    monkeypatch.setattr(probe, 'build_worker', broken)
    with pytest.raises(RuntimeError, match=r'C\+\+ build'):
        probe.main()
    result = json.loads(report.read_text())
    assert result['failed_stage'] == 'native-build'
    assert result['artifacts']['retained'] is True
    assert result['package']['checkpoint_sha256'] == value.identity['checkpoint_sha256']
    assert (destination / 'checkpoint.pt2').is_file()
    assert (destination / 'checkpoint.json').is_file()


def test_reuse_preflight_does_not_export_or_prepare_gpu_reference(monkeypatch, tmp_path: Path) -> None:
    package = tmp_path / 'reuse.pt2'
    report, value = configure_main(monkeypatch, tmp_path, ['--reuse-package', str(package), '--preflight-only'])
    package_at(package, value)
    for name in ('export_checkpoint', 'reuse_reference'):
        monkeypatch.setattr(probe, name, lambda *a, **kw: pytest.fail('executed in preflight'))
    probe.main()
    assert json.loads(report.read_text())['qualification'] == 'not_run'


def test_report_inside_new_workspace_is_supported(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / 'run'
    _, _value = configure_main(monkeypatch, tmp_path, ['--work-dir', str(destination)])
    report = destination / 'result.json'
    sys.argv[sys.argv.index('--report') + 1] = str(report)
    def fail(*_a, **_kw):
        raise RuntimeError('expected export failure')
    monkeypatch.setattr(probe, 'export_checkpoint', fail)
    with pytest.raises(RuntimeError, match='expected export'):
        probe.main()
    assert json.loads(report.read_text())['failed_stage'] == 'export'


def test_existing_workdir_is_not_modified(monkeypatch, tmp_path: Path) -> None:
    destination = tmp_path / 'exists'
    destination.mkdir()
    report, _ = configure_main(monkeypatch, tmp_path, ['--work-dir', str(destination)])
    with pytest.raises(FileExistsError):
        probe.main()
    assert list(destination.iterdir()) == []
    assert not report.exists()


def test_startup_failure_keeps_stderr_and_reaps_worker(monkeypatch, tmp_path: Path) -> None:
    value, package = loaded(), tmp_path / 'saved.pt2'
    package_at(package, value)
    binary = tmp_path / 'failing-worker'
    binary.write_text(f'#!{sys.executable}\nimport os\nos.write(2, b"diagnostic marker: missing GPU kernel\\n")\nraise SystemExit(2)\n')
    binary.chmod(0o755)
    original = subprocess.Popen
    spawned = []
    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        spawned.append(child)
        return child
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    with pytest.raises(RuntimeError, match='diagnostic marker: missing GPU kernel'):
        NativeEvaluator(binary, package, timeout=5)
    assert len(spawned) == 1
    assert spawned[0].poll() is not None
    assert spawned[0].stdin.closed
    assert spawned[0].stdout.closed


@pytest.mark.parametrize('name', ['checkpoint.pt2', 'checkpoint.json', 'untrained-transformer.pt'])
def test_report_cannot_collide_with_planned_model_artifacts(monkeypatch, tmp_path: Path, name: str) -> None:
    destination = tmp_path / 'new'
    configure_main(monkeypatch, tmp_path, ['--work-dir', str(destination)])
    report = destination / name
    sys.argv[sys.argv.index('--report') + 1] = str(report)
    with pytest.raises(ValueError, match='protected input'):
        probe.main()
    assert not report.exists()


@pytest.mark.parametrize('fault', ['pipe_setup', 'stderr'])
def test_startup_setup_failures_still_reap_worker(monkeypatch, tmp_path: Path, fault: str) -> None:
    value, package = loaded(), tmp_path / 'saved.pt2'
    package_at(package, value)
    binary = tmp_path / 'waiting-worker'
    binary.write_text(f'#!{sys.executable}\nimport time\ntime.sleep(60)\n')
    binary.chmod(0o755)
    original = subprocess.Popen
    spawned = []
    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        spawned.append(child)
        return child
    def pipe_failure(*_args):
        raise OSError('injected pipe setup failure')
    def handshake_failure(*_args):
        raise RuntimeError('injected handshake failure')
    def stderr_failure(_self):
        raise OSError('injected stderr read failure')
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    if fault == 'pipe_setup':
        monkeypatch.setattr(backend.os, 'set_blocking', pipe_failure)
    else:
        monkeypatch.setattr(backend, 'read_exact', handshake_failure)
        monkeypatch.setattr(NativeEvaluator, 'diagnostics', stderr_failure)
    try:
        with pytest.raises(RuntimeError, match='native evaluator startup failed: injected') as caught:
            NativeEvaluator(binary, package, timeout=5)
        if fault == 'stderr':
            assert 'injected handshake failure' in str(caught.value)
            assert 'injected stderr read failure' in str(caught.value)
        assert len(spawned) == 1
        child = spawned[0]
        assert child.poll() is not None
        assert child.stdin.closed
        assert child.stdout.closed
    finally:
        # Also reap the before-fix negative control; do not orphan its worker.
        for child in spawned:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
            child.stdin.close()
            child.stdout.close()


def test_qualifier_diagnostics_fault_still_closes_worker(monkeypatch, tmp_path: Path) -> None:
    report, value = configure_main(monkeypatch, tmp_path, [])
    closed = []
    class Worker:
        sequence = 1
        def diagnostics(self):
            raise OSError('injected final diagnostic failure')
        def close(self):
            closed.append(True)
    def export(_loaded, path, **_kwargs):
        package_at(path, value)
        return value.model
    def no_search(*_args, **_kwargs):
        raise RuntimeError('injected search failure')
    monkeypatch.setattr(probe, 'export_checkpoint', export)
    monkeypatch.setattr(probe, 'build_worker', lambda *_args: tmp_path / 'worker')
    monkeypatch.setattr(probe.sessions, 'build', lambda *_args: {'reference': tmp_path / 'oracle'})
    monkeypatch.setattr(probe.sessions, 'Oracle', no_search)
    monkeypatch.setattr(probe, 'NativeEvaluator', lambda *_args: Worker())
    with pytest.raises(OSError, match='injected final diagnostic failure'):
        probe.main()
    assert closed == [True]
    result = json.loads(report.read_text())
    assert result['status'] == 'failed'
    assert result['qualification'] == 'failed'
