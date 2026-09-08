"""Cache collection preserves the verifier and the ordinary adapter publication."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import adapt_raw_bt4_sidecars as adapter
from scripts import bt4_policy_mix as mix
from scripts import bt4_raw_corpus_sidecar as raw
from scripts import corpus_row_provenance as provenance
from tests.test_adapt_raw_bt4_sidecars import prepare


def pending_for(manifest: dict[str, Any]) -> tuple[raw.PendingShard, dict[str, Any]]:
    inputs = adapter.RawInputs(manifest, max_rows=1000, max_index_bytes=1024**2)
    source, receipts = next(iter(inputs.sources.values()))
    name, receipt = next(iter(receipts.items()))
    pending = raw.PendingShard(source, source.corpus_dir / name, int(receipt['positions']),
                               source.out_dir / raw.sidecar_name(name))
    teacher = manifest['teacher']
    kwargs = {'onnx_sha256': teacher['onnx']['sha256'], 'expected_policy_output': teacher['policy_output'],
              'expected_providers': teacher['providers'], 'expected_remap': inputs.remap, 'batch_size': 3}
    return pending, kwargs


def test_collected_index_equals_legacy_second_pass_and_default_attrs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, manifest = prepare(tmp_path)
    pending, kwargs = pending_for(manifest)
    original = raw.encode_rows
    calls = []
    def count(rows: Any, **kw: Any) -> Any:
        calls.append(len(rows))
        return original(rows, **kw)
    monkeypatch.setattr(raw, 'encode_rows', count)
    collected = np.zeros(pending.claimed_rows, dtype=provenance.RECORD_DTYPE)
    collected['source'] = 123  # Collection initializes the unused source/row fields too.
    attrs = raw.verify_shard(pending, **kwargs, identity_records=collected)
    assert calls == [3, 3, 2]
    assert sum(calls) == pending.claimed_rows
    expected = np.zeros_like(collected)
    for i, row in enumerate(adapter.derive.iter_corpus_rows(pending.path)):
        planes, _, keys, gids, plies = original([row], source=pending.source)
        expected['input_key'][i] = keys[0]
        expected['stored_input_key'][i] = np.frombuffer(bytes.fromhex(
            adapter.corpus.input_tensor_key(np.asarray(planes[0], dtype=np.float16))), dtype=np.uint8)
        expected['game_id'][i], expected['ply'][i], expected['worker_id'][i] = gids[0], plies[0], int(row['worker_id'])
    np.testing.assert_array_equal(collected, expected)
    assert np.any(collected['input_key'] != collected['stored_input_key'])
    assert raw.verify_shard(pending, **kwargs) == attrs


@pytest.mark.parametrize('bad_buffer', ['dtype', 'shape', 'readonly'])
def test_invalid_cache_is_rejected_before_encoding(
    tmp_path: Path, bad_buffer: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, manifest = prepare(tmp_path)
    pending, kwargs = pending_for(manifest)
    data = np.zeros(pending.claimed_rows, dtype=provenance.RECORD_DTYPE)
    if bad_buffer == 'dtype':
        data = np.zeros(pending.claimed_rows, dtype=np.float32)
    elif bad_buffer == 'shape':
        data = data[:-1]
    else:
        data.flags.writeable = False
    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError('invalid buffer reached reconstruction')
    monkeypatch.setattr(raw, 'encode_rows', unexpected)
    with pytest.raises(ValueError, match='identity cache must be writable'):
        raw.verify_shard(pending, **kwargs, identity_records=data)


@pytest.mark.parametrize('field', [raw.INPUT_KEY_FIELD, raw.SOURCE_KEY_FIELD, raw.GAME_ID_FIELD, raw.POLICY_FIELD])
def test_collection_keeps_existing_integrity_refusals(tmp_path: Path, field: str) -> None:
    _, _, _, manifest = prepare(tmp_path)
    pending, kwargs = pending_for(manifest)
    group: Any = zarr.open_group(str(pending.target), mode='r+')
    array = np.asarray(group[field][:])
    if field == raw.POLICY_FIELD:
        array[0, :] = np.nan
    else:
        array.flat[0] += 1
    group[field][:] = array
    with pytest.raises(ValueError, match=r'mismatch|invalid stored policy'):
        raw.verify_shard(pending, **kwargs,
                         identity_records=np.zeros(pending.claimed_rows, dtype=provenance.RECORD_DTYPE))


def test_actual_shuffled_adapter_encodes_each_raw_row_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, derived, raw_dir, _ = prepare(tmp_path, three_shards=True)
    original = raw.encode_rows
    encoded = 0
    def count(rows: Any, **kwargs: Any) -> Any:
        nonlocal encoded
        encoded += len(rows)
        return original(rows, **kwargs)
    monkeypatch.setattr(raw, 'encode_rows', count)
    out = tmp_path / 'adapted'
    summary = adapter.adapt(manifest_path, expected_manifest_sha256=adapter.file_sha256(manifest_path), out=out)
    assert encoded == 9  # Includes the resultless row, once, even across shuffled output shards.
    assert summary['rows'] == 8
    for shard in sorted(derived.glob('shard_*.zarr')):
        source: Any = zarr.open_group(str(shard), mode='r')
        output: Any = zarr.open_group(str(out / shard.name), mode='r')
        keys = mix._sidecar_identity(source, shard)
        mix._validate_sidecar(out / shard.name, source_path=shard, source_keys=keys[1],
                              source_key_sha=keys[2], source_policy_sha=keys[3])
        for i, ref in enumerate(provenance.read(shard / provenance.FILENAME, rows=len(source['x']))):
            raw_group: Any = zarr.open_group(str(raw_dir / raw.sidecar_name(ref['source_shard'])), mode='r')
            np.testing.assert_array_equal(output[mix.SIDECAR_POLICY_FIELD][i], raw_group[raw.POLICY_FIELD][ref['source_row']])
