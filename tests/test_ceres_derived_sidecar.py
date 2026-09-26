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
        value += (len(self.requests) - 1) * 100
        secondary = value + np.float16(0.5)
        if self.defect == 'nan':
            value[0, 0] = np.nan
        elif self.defect == 'dtype':
            policy = policy.astype('float32')
        elif self.defect == 'shape':
            value = value[:, :2]
        self.policies.append(policy.copy())
        if self.defect == 'missing_value2':
            return [policy, value]
        if self.defect == 'value2_shape':
            secondary = secondary[:, :2]
        elif self.defect == 'value2_dtype':
            secondary = secondary.astype('float32')
        elif self.defect == 'value2_nan':
            secondary[0, 0] = np.nan
        elif self.defect == 'padded_nan':
            value[-1, 0] = np.nan
        return [policy, value, *([secondary] if 'value2' in names else [])]


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
    assert tool.main([*argv, '--pad-final-batch', '--retain-value2']) == 0
    assert seen[-1][0].pad_final_batch is True
    assert seen[-1][0].retain_value2 is True


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


@pytest.mark.parametrize('n', [1, 31, 32, 33])
@pytest.mark.parametrize('retain_value2', [False, True])
def test_padded_real_rows_and_optional_head(tmp_path, monkeypatch, n, retain_value2):
    args = setup(tmp_path, monkeypatch, n)
    args.pad_final_batch = True
    args.retain_value2 = retain_value2
    # Construct distinct valid counter planes so first-row padding cannot pass as last-row padding.
    constructed: Any = zarr.open_group(str(Path(args.source) / 'shard_000000.zarr'), mode='a')
    features = constructed['x'][:]
    for row in range(n):
        features[row, 109] = np.float16((row + 8) / 100)
    constructed['x'][:] = features
    session = Session(tmp_path)
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: session)
    tool.produce(args)
    source: Any = zarr.open_group(str(Path(args.source) / 'shard_000000.zarr'), mode='r')
    bank: Any = zarr.open_group(str(Path(args.out) / 'shard_000000.zarr'), mode='r')
    calls = (n + 31) // 32
    assert session.requests == [['policy', 'value', *(['value2'] if retain_value2 else [])]] * calls
    actual_feed = tool.tpg.stored_x_to_ceres_tpg_bytes(source['x'][:],
        input_history_encoding=tool.shared.HISTORY, history_rep_fix=True)
    for call, feed in enumerate(session.feeds):
        real = min(32, n - 32 * call)
        np.testing.assert_array_equal(feed[:real], actual_feed[call * 32:call * 32 + real])
        np.testing.assert_array_equal(feed[real:], np.repeat(feed[real-1:real], 32-real, axis=0))
    for name in ['game_id', 'ply_index']:
        np.testing.assert_array_equal(bank[name][:], source[name][:])
    np.testing.assert_array_equal(bank['row_index'][:], np.arange(n))
    np.testing.assert_array_equal(bank['tpg_feed_sha256'][:], tool.shared.row_digests(actual_feed))
    expected_value = np.concatenate([
        np.column_stack((np.arange(min(32, n - 32*c)), -np.arange(min(32, n - 32*c)),
                         np.ones(min(32, n - 32*c)))).astype('float16') + c*100
        for c in range(calls)])
    np.testing.assert_array_equal(bank['value_logits'][:], expected_value)
    if retain_value2:
        np.testing.assert_array_equal(bank['value2_logits'][:], expected_value + np.float16(0.5))
    assert set(bank.array_keys()) == set(tool.DTYPES) | ({'value2_logits'} if retain_value2 else set())
    gather = tool.mapping.leela_gather_indices(*tool.tpg.ceres_tpg_gather_context(actual_feed))
    offsets = bank['legal_offsets'][:]
    assert offsets.shape == (n + 1,)
    for row in range(n):
        indices = np.flatnonzero(source['legal_mask'][row])
        lo, hi = int(offsets[row]), int(offsets[row+1])
        np.testing.assert_array_equal(bank['legal_indices'][lo:hi], indices)
        np.testing.assert_array_equal(bank['policy_logits'][lo:hi],
            session.policies[row // 32][row % 32, gather[row, indices]])
    expected_counts = {'real_rows': n, 'padding_rows': calls*32-n, 'calls': calls, 'input_rows': calls*32}
    assert bank.attrs['collection_counts'] == expected_counts
    assert bank.attrs['source_array_sha256'] == {
        k: tool.shared.raw.sha_array(np.asarray(source[k][:])) for k in tool.COLUMNS}
    completed_path = Path(args.invocation) / 'child_completed.json'
    completed = json.loads(completed_path.read_text())
    assert completed['collection_counts'] == completed['new_collection_counts'] == expected_counts
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: pytest.fail('cache opened teacher'))
    tool.produce(args)
    resumed = json.loads(completed_path.read_text())
    assert resumed['collection_counts'] == expected_counts
    assert resumed['new_collection_counts'] == dict.fromkeys(expected_counts, 0)
    writable: Any = zarr.open_group(str(Path(args.out) / 'shard_000000.zarr'), mode='a')
    writable.attrs['collection_counts'] = {**expected_counts, 'padding_rows': 999}
    with pytest.raises(ValueError, match='collection counts differ'):
        tool.produce(args)


@pytest.mark.parametrize('defect', ['missing_value2', 'value2_shape', 'value2_dtype', 'value2_nan', 'padded_nan'])
def test_optional_head_or_padded_output_defect_refuses(tmp_path, monkeypatch, defect):
    args = setup(tmp_path, monkeypatch, 1)
    args.pad_final_batch = args.retain_value2 = True
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: Session(tmp_path, defect))
    with pytest.raises(ValueError, match=r'missing teacher outputs|invalid native outputs'):
        tool.produce(args)
    assert not (Path(args.out) / 'shard_000000.zarr').exists()
    assert not (Path(args.invocation) / 'child_completed.json').exists()


@pytest.mark.parametrize(('pad', 'secondary'), [(True, False), (False, True), (True, True)])
def test_old_profile_verification_and_optin_namespace_refusal(tmp_path, monkeypatch, pad, secondary):
    args = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: Session(tmp_path))
    tool.produce(args)
    path = Path(args.out) / 'shard_000000.zarr'
    writable: Any = zarr.open_group(str(path), mode='a')
    binding = dict(writable.attrs['binding'])
    assert binding['profile'] == tool.PROFILE
    assert binding['backend'] == tool.BACKEND
    # Historical v1 did not carry explicit per-shard accounting; same verifier still admits it.
    del writable.attrs['collection_counts']
    tool.verify_cached(path, binding)
    args.pad_final_batch, args.retain_value2 = pad, secondary
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: pytest.fail('profile mismatch opened teacher'))
    with pytest.raises(ValueError, match='namespace differs'):
        tool.produce(args)
    with pytest.raises(ValueError, match='binding differs'):
        tool.verify_cached(path, tool.expected_binding(args, {'path': path.name, 'rows': 32},
                                                      binding['source_storage_identity']))


def test_value2_without_padding_and_corrupt_secondary_cache(tmp_path, monkeypatch):
    args = setup(tmp_path, monkeypatch)
    args.retain_value2 = True
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: Session(tmp_path))
    tool.produce(args)
    path = Path(args.out) / 'shard_000000.zarr'
    bank: Any = zarr.open_group(str(path), mode='a')
    assert bank.attrs['binding']['backend']['remainder'] == 'refuse'
    assert bank.attrs['binding']['backend']['outputs'] == ['policy', 'value', 'value2']
    bank['value2_logits'][0, 0] += np.float16(1)
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: pytest.fail('corrupt cache opened teacher'))
    with pytest.raises(ValueError, match='digest differs'):
        tool.produce(args)


def test_cli_flags_reach_actual_owned_child(tmp_path, monkeypatch):
    def completed_child(args):
        assert args.pad_final_batch is True
        assert args.retain_value2 is True
        tool.shared.raw.atomic_json(Path(args.invocation) / 'child_completed.json',
            {'complete': True, 'pad_final_batch': args.pad_final_batch,
             'retain_value2': args.retain_value2})

    monkeypatch.setattr(tool, 'produce', completed_child)
    out = tmp_path / 'out'
    assert tool.main(['--source', str(tmp_path / 'source'), '--out', str(out),
        '--onnx', str(tmp_path / 'model'), '--expected-source-summary-sha256', 'a'*64,
        '--expected-onnx-sha256', tool.MODEL_SHA, '--wdl-output', 'value',
        '--wdl-output-kind', 'logits', '--max-shards', '1', '--gpu-lock', str(tmp_path / 'gpu'),
        '--minimum-free-gib', '0', '--max-seconds', '60', '--pad-final-batch', '--retain-value2']) == 0
    invocation, = (out / 'invocations').iterdir()
    assert json.loads((invocation / 'completed.json').read_text())['retain_value2'] is True
    assert json.loads((invocation / 'child_started.json').read_text())['profile'] == tool.EXTENDED_PROFILE


@pytest.mark.parametrize(('rows', 'chunk_rows'), [(83, 48), (97, 16), (70, 512)])
def test_source_block_cache_matches_batches_and_decodes_each_block_once(tmp_path, rows, chunk_rows):
    group = zarr.open_group(str(tmp_path / 'source'), mode='w')
    accesses = {k: [] for k in tool.COLUMNS}

    class Tracked:
        def __init__(self, name):
            self.name = name
            self.chunks = group[name].chunks

        def __getitem__(self, selection):
            accesses[self.name].append((selection.start, selection.stop))
            return group[self.name][selection]

    for k in tool.COLUMNS:
        values = np.arange(rows * 3).reshape(rows, 3) if k == 'x' else np.arange(rows)
        group.create_dataset(k, data=values, chunks=(chunk_rows, *values.shape[1:]))
    reader = tool.SourceBatchReader({k: Tracked(k) for k in tool.COLUMNS}, rows)
    for start in range(0, rows, 32):
        end = min(start + 32, rows)
        actual = reader.read(start, end)
        for k in tool.COLUMNS:
            np.testing.assert_array_equal(actual[k], group[k][start:end])
    expected = [(start, min(start + chunk_rows, rows)) for start in range(0, rows, chunk_rows)]
    assert accesses == dict.fromkeys(tool.COLUMNS, expected)


def test_source_block_cache_large_chunks_keep_direct_batch_reads(tmp_path):
    group = zarr.open_group(str(tmp_path / 'source'), mode='w')
    calls = []

    class Tracked:
        chunks = (2048,)

        def __init__(self, name):
            self.name = name

        def __getitem__(self, selection):
            calls.append((self.name, selection.start, selection.stop))
            return group[self.name][selection]

    for k in tool.COLUMNS:
        group.create_dataset(k, data=np.arange(2050), chunks=(2048,))
    reader = tool.SourceBatchReader({k: Tracked(k) for k in tool.COLUMNS}, 2050)
    for start, end in [(0, 32), (32, 64), (2048, 2050)]:
        batch = reader.read(start, end)
        for k in tool.COLUMNS:
            np.testing.assert_array_equal(batch[k], np.arange(start, end))
    assert calls == [(k, start, end) for start, end in [(0, 32), (32, 64), (2048, 2050)]
                     for k in tool.COLUMNS]
    assert not reader.block


def test_cached_source_batches_preserve_actual_writer_feeds_and_partial_output(tmp_path, monkeypatch):
    # Real collectors own the GPU lease until process exit. Both sessions here
    # are fake and run in one process to compare saved arrays directly.
    monkeypatch.setattr(tool.fcntl, 'flock', lambda _fd, _mode: None)
    args = setup(tmp_path, monkeypatch, 83)
    args.pad_final_batch = args.retain_value2 = True
    source = zarr.open_group(str(Path(args.source) / 'shard_000000.zarr'), mode='a')
    x = source['x'][:]
    del source['x']
    source.create_dataset('x', data=x, chunks=(48, *x.shape[1:]))
    cached_session = Session(tmp_path)
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: cached_session)
    tool.produce(args)
    cached = zarr.open_group(str(Path(args.out) / 'shard_000000.zarr'), mode='r')
    completion = json.loads((Path(args.invocation) / 'child_completed.json').read_text())
    timings = completion['stage_seconds_new_shards']
    assert all(np.isfinite(v) and v >= 0 for v in timings.values())
    assert timings['total'] == pytest.approx(sum(v for k, v in timings.items() if k != 'total'))
    assert timings['source_read'] > 0
    assert timings['session_run'] > 0

    class DirectReader:
        def __init__(self, group, _rows):
            self.group = group

        def read(self, start, end):
            return {k: np.asarray(self.group[k][start:end]) for k in tool.COLUMNS}

    monkeypatch.setattr(tool, 'SourceBatchReader', DirectReader)
    direct_session = Session(tmp_path)
    monkeypatch.setattr(tool, 'open_teacher', lambda _a: direct_session)
    args.out = str(tmp_path / 'direct')
    Path(args.out).mkdir()
    tool.produce(args)
    direct = zarr.open_group(str(Path(args.out) / 'shard_000000.zarr'), mode='r')
    assert len(cached_session.feeds) == len(direct_session.feeds) == 3
    for a, b in zip(cached_session.feeds, direct_session.feeds, strict=True):
        np.testing.assert_array_equal(a, b)
    assert set(cached.array_keys()) == set(direct.array_keys())
    for key in cached.array_keys():
        np.testing.assert_array_equal(cached[key][:], direct[key][:])
    assert cached.attrs['source_array_sha256'] == direct.attrs['source_array_sha256']
    assert cached.attrs['array_sha256'] == direct.attrs['array_sha256']
