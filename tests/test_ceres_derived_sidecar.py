"""Actual tiny shuffled corpus and compact storage; fake session only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import ceres_derived_sidecar as tool
from tests.test_sf_policy_rewrite import raw_row, write_corpus, run


class Session:
    def __init__(self, root: Path, defect: str = ''):
        self.root = root
        self.defect = defect
        self.requests: list[list[str]] = []
        self.feeds: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []

    def get_providers(self):
        return tool.BACKEND['providers']

    def end_profiling(self):
        p = self.root / 'profile.json'
        provider = 'CPUExecutionProvider' if self.defect == 'cpu' else 'CUDAExecutionProvider'
        p.write_text(json.dumps([{'args': {'provider': provider, 'op_name': 'Gemm'}}]))
        return str(p)

    def run(self, names, feed):
        self.requests.append(names)
        x = feed['squares_byte']
        self.feeds.append(x.copy())
        assert x.dtype == np.uint8
        assert x.shape == (32, 64, 137)
        policy = (np.arange(1858)[None, :] / 1000 + np.arange(32)[:, None]).astype('float16')
        value = np.column_stack((np.arange(32), -np.arange(32), np.ones(32))).astype('float16')
        if self.defect == 'nan':
            value[0, 0] = np.nan
        elif self.defect == 'dtype':
            policy = policy.astype('float32')
        elif self.defect == 'shape':
            value = value[:, :2]
        self.policies.append(policy.copy())
        return [policy, value]


def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int = 32):
    rows = [raw_row(game_id=i) for i in range(n)]
    raw = write_corpus(tmp_path, rows, row_schema=3, staircase=[{'depth': 9, 'width': 'all'}])
    source = tmp_path / 'derived'
    run(raw, source, '--limit', str(n), temp=0.0005, rows_per_shard=n)
    model = tmp_path / 'fake.onnx'
    model.write_bytes(b'not a real model')
    monkeypatch.setattr(tool, 'MODEL_SHA', tool.shared.file_sha256(model))
    args = tool.build_parser().parse_args([
        '--source', str(source), '--out', str(tmp_path / 'bank'), '--onnx', str(model),
        '--expected-source-summary-sha256', tool.shared.file_sha256(source / tool.shared.SUMMARY),
        '--expected-onnx-sha256', tool.MODEL_SHA, '--wdl-output', 'value',
        '--wdl-output-kind', 'logits', '--max-shards', '1', '--gpu-lock', str(tmp_path / 'gpu.lock'),
        '--minimum-free-gib', '0'])
    Path(args.out).mkdir()
    args.invocation = str(tmp_path / 'invocation')
    Path(args.invocation).mkdir()
    return args


def test_actual_shuffled_native_compact_bank_and_cache(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    session = Session(tmp_path)
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: session)
    tool.produce(args)
    assert session.requests == [['policy', 'value']]
    source: Any = zarr.open_group(str(Path(args.source) / 'shard_000000.zarr'), mode='r')
    bank: Any = zarr.open_group(str(Path(args.out) / 'shard_000000.zarr'), mode='r')
    assert source['game_id'][:].tolist() != sorted(source['game_id'][:].tolist())
    np.testing.assert_array_equal(bank['game_id'][:], source['game_id'][:])
    np.testing.assert_array_equal(bank['ply_index'][:], source['ply_index'][:])
    np.testing.assert_array_equal(bank['value_logits'][:], np.column_stack((np.arange(32), -np.arange(32), np.ones(32))).astype('float16'))
    gather = tool.mapping.leela_gather_indices(*tool.tpg.ceres_tpg_gather_context(session.feeds[0]))
    offsets = bank['legal_offsets'][:]
    for row in range(32):
        indices = np.flatnonzero(source['legal_mask'][row])
        lo, hi = int(offsets[row]), int(offsets[row+1])
        np.testing.assert_array_equal(bank['legal_indices'][lo:hi], indices)
        np.testing.assert_array_equal(bank['policy_logits'][lo:hi], session.policies[0][row, gather[row, indices]])
    np.testing.assert_array_equal(bank['tpg_feed_sha256'][:], tool.shared.row_digests(session.feeds[0]))
    assert set(bank.array_keys()) == set(tool.DTYPES)
    # Actual produce cache branch skips the fake session and shared GPU lock entirely.
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: pytest.fail('cache opened session'))
    tool.produce(args)
    writable: Any = zarr.open_group(str(Path(args.out) / 'shard_000000.zarr'), mode='a')
    writable['policy_logits'][0] += 1
    with pytest.raises(ValueError, match='digest differs'):
        tool.produce(args)


@pytest.mark.parametrize('defect', ['nan', 'dtype', 'shape', 'cpu'])
def test_bad_session_never_publishes(tmp_path, monkeypatch, defect):
    args = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: Session(tmp_path, defect))
    with pytest.raises(ValueError, match=r'native outputs|CPU kernel'):
        tool.produce(args)
    assert not (Path(args.out) / 'shard_000000.zarr').exists()
    assert not (Path(args.invocation) / 'child_completed.json').exists()


def test_nondivisible_shard_refuses_before_session(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch, 33)
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: pytest.fail('partial shard reached session'))
    with pytest.raises(ValueError, match='divisible shards'):
        tool.produce(args)
    assert not (Path(args.out) / 'ceres_source.json').exists()


def test_legacy_bt4_refuses_ceres_namespace_before_source(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    (Path(args.out) / 'ceres_source.json').write_text('{}')
    monkeypatch.setattr(tool.shared, 'source_inventory', lambda _a: pytest.fail('namespace admitted'))
    with pytest.raises(ValueError, match='Ceres output namespace'):
        tool.shared.produce(args)


@pytest.mark.parametrize('event', [
    {'provider': 'CPUExecutionProvider', 'op_name': 'Cast', 'output_type_shape': [{'float': [1]}]},
    {'provider': 'CPUExecutionProvider', 'op_name': 'Shape', 'output_type_shape': [{'int64': [4097]}]},
    {'provider': 'other', 'op_name': 'Gemm'},
])
def test_provider_proof_refuses_unqualified_compute(event):
    with pytest.raises(ValueError, match=r'CPU|provider'):
        tool.provider_proof([{'args': {'provider': 'CUDAExecutionProvider', 'op_name': 'Gemm'}}, {'args': event}])


def test_cli_selects_ceres_child_and_revalidates_contract(monkeypatch):
    seen = []
    monkeypatch.setattr(tool.shared, 'run', lambda args, *, child_target: seen.append((args, child_target)) or 0)
    argv = ['--source', '/fake/source', '--out', '/fake/out', '--onnx', '/fake/model',
            '--expected-source-summary-sha256', 'a'*64, '--expected-onnx-sha256', tool.MODEL_SHA,
            '--wdl-output', 'value', '--wdl-output-kind', 'logits', '--max-shards', '1', '--gpu-lock', '/fake/lock']
    assert tool.main(argv) == 0
    assert seen[0][1] is tool.child
    with pytest.raises(ValueError, match='fixed32'):
        tool.main([*argv, '--batch-size', '16'])
    assert len(seen) == 1


def test_final_publication_refusal_preserves_partial(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: Session(tmp_path))
    verify = tool.verify_cached

    def stopped_after_readback(path, expected):
        verify(path, expected)
        raise ValueError('independent STOP before publication')

    monkeypatch.setattr(tool, 'verify_cached', stopped_after_readback)
    with pytest.raises(ValueError, match='STOP before publication'):
        tool.produce(args)
    assert not (Path(args.out) / 'shard_000000.zarr').exists()
    assert (Path(args.out) / 'shard_000000.zarr.writing').is_dir()
    assert not (Path(args.invocation) / 'child_completed.json').exists()


def test_other_source_namespace_refuses_before_session(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    (Path(args.out) / 'ceres_source.json').write_text(json.dumps({'profile': 'other source'}))
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: pytest.fail('wrong namespace reached teacher'))
    with pytest.raises(ValueError, match='namespace differs'):
        tool.produce(args)
