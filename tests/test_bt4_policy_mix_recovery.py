from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import bt4_policy_mix as tool
from tests.test_bt4_policy_mix import (
    _write_audit_receipt, _write_global_audit_inputs, _write_rank_sidecar,
    _write_sidecar, _write_source,
)


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value) + '\n')


def _pin(path: Path) -> dict[str, str]:
    return {'path': str(path), 'sha256': tool.file_sha256(path)}


def _interrupted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], Path, Path]:
    source, first, _ = _write_source(tmp_path)
    source_group: Any = zarr.open_group(str(first), mode='a')
    source_group[tool.POLICY_FIELD][:, :3] = np.asarray(
        [[.375, .375, .25], [.625, .375, 0]], dtype=np.float16,
    )
    second = source / 'shard_000001.zarr'
    shutil.copytree(first, second)
    source_summary = json.loads((source / tool.DERIVE_SUMMARY).read_text())
    source_summary['realized']['realized_base_depth_histogram']['9'] = 4
    _write(source / tool.DERIVE_SUMMARY, source_summary)
    side = _write_sidecar(tmp_path, source, first)
    side_group: Any = zarr.open_group(str(side / first.name), mode='a')
    side_group[tool.SIDECAR_POLICY_FIELD][:, :3] = np.asarray(
        [[.25, .75, 0], [1/3, 1/3, 1/3]], dtype=np.float32,
    )
    shutil.copytree(side / first.name, side / second.name)
    zarr.open_group(str(side / second.name), mode='a').attrs['source_shard'] = second.name
    summary = json.loads((side / tool.SIDECAR_SUMMARY).read_text())
    summary.update(source_shards=2, sidecar_shards=2, rows=4)
    _write(side / tool.SIDECAR_SUMMARY, summary)
    ranks = _write_rank_sidecar(tmp_path, source, first)
    shutil.copytree(ranks / first.name, ranks / second.name)
    zarr.open_group(str(ranks / second.name), mode='a').attrs['source_shard'] = second.name
    summary = json.loads((ranks / tool.sf_ranks.SUMMARY_NAME).read_text())
    summary.update(rows=4, shards=2)
    summary['outputs'].append({**summary['outputs'][0], 'path': second.name})
    _write(ranks / tool.sf_ranks.SUMMARY_NAME, summary)
    common = ['--shards', str(source), '--sidecar', str(side), '--sf-rank-sidecar', str(ranks),
              '--expected-rows', '4', '--expected-shards', '2',
              '--expected-source-summary-sha256', tool.file_sha256(source / tool.DERIVE_SUMMARY)]
    parent = tmp_path / 'C20'
    audit = _write_audit_receipt(tmp_path, scope='sf-cp-window', bt4_temperature=.5,
                                 sf_rank_cap=3, sf_cp_window=20.)
    assert tool.main(['mix', *common, '--scope', 'sf-cp-window', '--alpha', '1',
                      '--bt4-temperature', '.5', '--sf-cp-window', '20',
                      '--audit-receipt', str(audit), '--out', str(parent)]) == 0
    inputs = _write_global_audit_inputs(tmp_path, monkeypatch)
    record = tmp_path / 'prereg.md'
    record.write_text('Bounded synthetic H20 recovery test.\n')
    treatment = ['--scope', 'c20-global', '--alpha', '.2', '--bt4-temperature', '.5',
                 '--sf-rank-cap', '3', '--sf-cp-window', '20', '--sf-audit-mode', 'descriptive',
                 '--experiment-record', str(record)]
    haudit = tmp_path / 'H20.audit.json'
    assert tool.main(['audit', *inputs, *treatment, '--json', str(haudit)]) == 0
    out = tmp_path / 'H20'
    args = ['mix', *common, *treatment, '--c20-parent', str(parent),
            '--expected-c20-summary-sha256', tool.file_sha256(parent / tool.DERIVE_SUMMARY),
            '--expected-c20-mix-sha256', tool.file_sha256(parent / tool.MIX_SUMMARY),
            '--audit-receipt', str(haudit), '--out', str(out)]
    original = tool._sidecar_identity

    def stop(group: Any, path: Path, **kwargs: Any) -> Any:
        if path.name == second.name:
            raise RuntimeError('synthetic producer interrupted after one complete shard')
        return original(group, path, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(tool, '_sidecar_identity', stop)
        with pytest.raises(RuntimeError, match='synthetic producer interrupted'):
            tool.main(args)
    partial = out.with_name(out.name + '.writing')
    # Test-only qualified producer: actual loop above completed all first-shard checks.
    producer = tmp_path / 'producer.py'
    shutil.copyfile(tool.__file__, producer)
    monkeypatch.setattr(tool, 'RECOVERABLE_H20_PRODUCER_SHA256', tool.file_sha256(producer))
    status = tmp_path / 'failed.status.json'
    _write(status, {'stage': 'mix', 'status': 'FAILED_OR_STOPPED', 'returncode': 124,
                    'completed_unix': 1., 'pid': 2147483646, 'supervisor_pid': 2147483645,
                    'argv': [str(producer), *args]})
    metadata = tmp_path / 'prefix.jsonl'
    _write(metadata, tool.h20_recovery_metadata(first, parent / first.name,
                                               partial / first.name, side / first.name, ranks / first.name))
    receipt = tmp_path / 'recovery.json'
    _write(receipt, {'schema': 1, 'kind': 'h20-completed-prefix-recovery', 'partial_dir': str(partial),
                     'original_mixer': _pin(producer), 'original_status': _pin(status),
                     'terminated_process_ids': [2147483646, 2147483645],
                     'completed_prefix_metadata': _pin(metadata)})
    return args, out, receipt


def _recover(args: list[str], receipt: Path) -> int:
    return tool.main([*args, '--recovery-receipt', str(receipt),
                      '--expected-recovery-receipt-sha256', tool.file_sha256(receipt)])


def test_recovery_preserves_prefix_bytes_and_reads_only_suffix_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, out, receipt = _interrupted(tmp_path, monkeypatch)
    prefix = out.with_name(out.name + '.writing') / 'shard_000000.zarr'
    before = {str(p.relative_to(prefix)): p.read_bytes() for p in prefix.rglob('*') if p.is_file()}
    original = zarr.Array.__getitem__

    def only_suffix(self: Any, key: Any) -> Any:
        assert 'shard_000000.zarr' not in str(self.store.path), 'recovery reread a prefix array'
        return original(self, key)

    with monkeypatch.context() as patch:
        patch.setattr(zarr.Array, '__getitem__', only_suffix)
        assert _recover(args, receipt) == 0
    final = out / prefix.name
    assert before == {str(p.relative_to(final)): p.read_bytes() for p in final.rglob('*') if p.is_file()}
    summary = json.loads((out / tool.MIX_SUMMARY).read_text())
    assert (summary['rows'], summary['shards']) == (4, 2)
    assert summary['changed_rows'] is None
    assert summary['mean_entropy_nats'] is None
    assert summary['mean_l1_from_source'] is None
    assert summary['selected_mass_abs_drift']['mean'] is None
    rec = summary['recovery']
    assert rec['completed_prefix_rows'] == 2
    assert rec['completed_prefix_shards'] == 1
    assert rec['suffix_statistics']['rows'] == 2
    assert rec['suffix_statistics']['shards'] == 1
    assert rec['suffix_statistics']['changed_rows'] > 0
    assert rec['mass_bound_certificate']['certified_row_abs_bound'] < 2**-10
    assert json.loads((tmp_path / 'failed.status.json').read_text())['returncode'] == 124
    derived = json.loads((out / tool.DERIVE_SUMMARY).read_text())
    assert derived['policy_target_postprocess'] == summary


@pytest.mark.parametrize('corruption', ['stamp', 'metadata', 'producer', 'active_pid', 'recipe', 'suffix'])
def test_recovery_refuses_unqualified_prefix_or_changed_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    args, out, receipt = _interrupted(tmp_path, monkeypatch)
    partial = out.with_name(out.name + '.writing')
    proof = json.loads(receipt.read_text())
    if corruption == 'stamp':
        group = zarr.open_group(str(partial / 'shard_000000.zarr'), mode='a')
        del group.attrs['policy_target_mix_c20_parent_policy_sha256']
    elif corruption == 'metadata':
        path = partial / 'shard_000000.zarr/x/.zarray'
        value = json.loads(path.read_text())
        value['dtype'] = '<f4'
        _write(path, value)
    elif corruption == 'producer':
        proof['original_mixer']['sha256'] = '0' * 64
    elif corruption == 'active_pid':
        proof['terminated_process_ids'].append(os.getpid())
    elif corruption == 'recipe':
        args[args.index('--alpha') + 1] = '.3'
    else:
        parent = Path(args[args.index('--c20-parent') + 1])
        group = zarr.open_group(str(parent / 'shard_000001.zarr'), mode='a')
        group['wdl_target'][0] = 1
    _write(receipt, proof)
    with pytest.raises((ValueError, SystemExit, KeyError)):
        _recover(args, receipt)
    assert not out.exists()
    assert partial.exists()
    assert not (partial / tool.MIX_SUMMARY).exists()


def test_h20_rounding_certificate_covers_normal_and_subnormal_targets() -> None:
    cert = tool.h20_mass_bound_certificate()
    rng = np.random.default_rng(20260908)
    rows = np.concatenate([rng.dirichlet(np.full(1858, .05), size=8),
                           np.full((1, 1858), 1/1858)], axis=0)
    stored = rows.astype(np.float32).astype(np.float16)
    drift = np.abs(stored.astype(np.float64).sum(axis=1) - 1.)
    assert float(drift.max()) <= cert['certified_row_abs_bound'] < 2**-10
