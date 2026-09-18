"""Cheap neural boundary contracts: no model, native build or search execution."""
from __future__ import annotations

import json
from pathlib import Path

import chess
import pytest

from native.bend_engine.neural_probe.adapter import contract, decode_key, key_for, mapping_hash, sha256
from native.bend_engine.session_probe.run_probe import parse_path


@pytest.mark.parametrize(('line', 'expected'), [('path 0', []), ('path 2 123 456', [123, 456])])
def test_history_path_roundtrip(line: str, expected: list[int]) -> None:
    assert parse_path(line) == expected


@pytest.mark.parametrize('line', ['path', 'path 1', 'path 0 123', 'path -1', 'path 33 ' + '1 '*33,
                                  'path 1 4294967296', 'path 1 1.5', 'board 0'])
def test_history_path_rejects_bad_records(line: str) -> None:
    with pytest.raises(ValueError, match=r'malformed|nondecimal|exceeds'):
        parse_path(line)


@pytest.mark.parametrize('fen', [chess.STARTING_FEN,
    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
    'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1',
    '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
    '4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1',
    '4k3/P7/8/8/8/8/8/4K3 w - - 0 1',
    '4k3/8/8/8/8/8/p7/4K3 b - - 0 1'])
def test_packed_moves_preserve_special_flags_and_promotions(fen: str) -> None:
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    keys = [key_for(board, m) for m in legal]
    assert len(set(keys)) == len(legal)
    assert [decode_key(board, key) for key in keys] == legal


@pytest.mark.parametrize('key', [0, -1, 1 << 17, 0xffffffff, (5 << 12) | 12 | 28 << 6,
                                 12 | 28 << 6 | 1 << 15])
def test_packed_illegal_or_wrong_flag_rejected(key: int) -> None:
    with pytest.raises(ValueError, match=r'illegal|exceeds|promotion|flag'):
        decode_key(chess.Board(), key)


def manifest(tmp_path: Path) -> tuple[Path, dict]:
    package = tmp_path / 'model.pt2'
    package.write_bytes(b'test fixture only, never loaded as a model')
    leaf = {'type': None, 'context': None, 'children_spec': []}
    data = {'format': 'deepfin-bend-cpu-tuple-v1', 'device': 'cpu', 'batch': 1, 'planes': 146,
            'dtype': 'float32', 'history_mode': 1, 'history_rep_fix': True,
            'package_sha256': sha256(package), 'policy_map_sha256': mapping_hash(),
            'outputs': ['compact_policy_logits', 'wdl_logits'],
            'output_spec': json.dumps([1, {'type': 'builtins.tuple', 'context': 'null', 'children_spec': [leaf, leaf]}])}
    return package, data


def test_explicit_package_contract_and_fingerprint(tmp_path: Path) -> None:
    package, data = manifest(tmp_path)
    package.with_suffix('.json').write_text(json.dumps(data))
    assert contract(package) == data
    package.write_bytes(b'changed')
    with pytest.raises(ValueError, match='fingerprint'):
        contract(package)


@pytest.mark.parametrize(('field', 'value', 'message'), [
    ('device', 'cuda', 'unsupported'), ('batch', 2, 'unsupported'),
    ('planes', 112, 'unsupported'), ('dtype', 'float64', 'unsupported'),
    ('history_mode', 9, 'history'), ('history_mode', True, 'history'),
    ('history_rep_fix', 1, 'history'), ('policy_map_sha256', 'bad', 'fingerprint'),
    ('outputs', ['wdl_logits', 'compact_policy_logits'], 'output contract'),
    ('output_spec', '[]', 'tensor tuple'), ('output_spec', None, 'output contract'),
])
def test_incompatible_model_contract_rejected(tmp_path: Path, field: str, value: object, message: str) -> None:
    package, data = manifest(tmp_path)
    data[field] = value
    package.with_suffix('.json').write_text(json.dumps(data))
    with pytest.raises(ValueError, match=message):
        contract(package)
