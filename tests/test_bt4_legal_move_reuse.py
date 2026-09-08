"""Exact compatibility with the prior numerical projection, without inference."""
from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import zarr

from scripts import bt4_policy_dump as dump
from scripts import bt4_raw_corpus_sidecar as raw
from tests.test_bt4_policy_dump import KITCHEN_SINK, ORDER_SENSITIVE
from tests.test_bt4_raw_corpus_sidecar import make_source

FENS = [chess.STARTING_FEN, KITCHEN_SINK, ORDER_SENSITIVE,
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
        "4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1",
        "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"]


def legacy_policy(board: chess.Board, logits: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Frozen pre-change UCI round-trip and float64 normalization oracle."""
    ucis = [move.uci() for move in board.legal_moves]
    if not ucis:
        return [], np.zeros((0,), dtype=np.float64)
    idx = np.array([dump.leela_index_for_move(board, chess.Move.from_uci(u)) for u in ucis],
                   dtype=np.int64)
    if int((idx < 0).sum()):
        raise RuntimeError('unmapped legal move')
    values = logits[idx].astype(np.float64)
    values = np.where(np.isfinite(values), values, -1e9)
    probs = np.exp(values - values.max())
    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError('degenerate policy')
    return ucis, probs / total


def logits_for(nonfinite: bool = False) -> np.ndarray:
    logits = np.linspace(-4, 3, raw.COMPACT_POLICY_SIZE, dtype=np.float32)
    if nonfinite:
        logits[::3] = np.nan
        logits[1::3] = np.inf
        logits[2::3] = -np.inf
    return logits


@pytest.mark.parametrize('fen', FENS)
@pytest.mark.parametrize('nonfinite', [False, True])
def test_move_objects_preserve_exact_legacy_order_and_probabilities(fen: str, nonfinite: bool) -> None:
    board = chess.Board(fen)
    logits = logits_for(nonfinite)
    expected_uci, expected = legacy_policy(board, logits)
    moves, actual = dump.legal_move_probabilities(board, logits)
    assert moves == list(board.legal_moves)
    assert [m.uci() for m in moves] == expected_uci
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    ucis, public_probs = dump.legal_move_policy(board, logits)
    assert ucis == expected_uci
    np.testing.assert_array_equal(public_probs, expected)
    np.testing.assert_array_equal(actual.astype(np.float32), expected.astype(np.float32))


@pytest.mark.parametrize('index', [-1, raw.COMPACT_POLICY_SIZE])
def test_teacher_mapping_errors_remain_fatal(monkeypatch: pytest.MonkeyPatch, index: int) -> None:
    monkeypatch.setattr(dump, 'leela_index_for_move', lambda _board, _move: index)
    with pytest.raises((RuntimeError, IndexError)):
        dump.legal_move_probabilities(chess.Board(), logits_for())


def make_multiboard_source(tmp_path: Path) -> tuple[raw.SourceSpec, list[dict[str, Any]]]:
    source = make_source(tmp_path)
    path = source.inventory.shards[0]
    with gzip.open(path, 'rt') as stream:
        template = json.loads(stream.readline())
    rows = []
    for i, fen in enumerate(FENS[:-1]):
        board = chess.Board(fen)
        x = raw.encode_position(board, add_features=True,
                                input_history_encoding=raw.derive.INPUT_HISTORY_ENCODING,
                                input_extra_features=raw.derive.INPUT_EXTRA_FEATURES)
        rows.append({**template, 'fen': board.fen(), 'history_root_fen': board.fen(en_passant='fen'),
                     'stm': 'w' if board.turn else 'b', 'piece_count': board.occupied.bit_count(),
                     'game_id': i, 'input_key': raw.corpus.input_tensor_key(x)})
    with gzip.open(path, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    return replace(source, inventory=replace(source.inventory, shard_rows=(len(rows),),
                                             rows_claimed=len(rows))), rows


class FixedSession:
    def __init__(self, logits: np.ndarray) -> None:
        self.logits = logits

    def run(self, _outputs: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        return [np.broadcast_to(self.logits, (next(iter(feed.values())).shape[0], *self.logits.shape))]


def label(source: raw.SourceSpec, logits: np.ndarray) -> tuple[Path, dict[str, Any]]:
    path = source.inventory.shards[0]
    pending = raw.PendingShard(source, path, source.inventory.rows_claimed,
                               source.out_dir / raw.sidecar_name(path.name))
    attrs = raw.label_shard(pending, sess=FixedSession(logits), input_name='input',
        input_dtype=np.dtype(np.float32), providers=['fake'], policy_name='policy',
        onnx_path=source.out_dir / 'fake.onnx', onnx_sha256='fake-model-hash',
        remap_stamp={'commit': 'fixed', 'blobs': {}}, batch_size=3)
    return pending.target, attrs


def test_actual_label_shard_matches_legacy_content_and_maps_compact_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, rows = make_multiboard_source(tmp_path)
    planes, boards, input_keys, games, plies = raw.encode_rows(rows, source=source)
    logits = logits_for()
    expected = np.zeros((len(rows), raw.COMPACT_POLICY_SIZE), dtype=np.float32)
    entropy = top1 = 0.0
    legal_count = 0
    for i, board in enumerate(boards):
        ucis, probabilities = legacy_policy(board, logits)
        indices = [raw.compact_index_for_move(board, chess.Move.from_uci(u)) for u in ucis]
        probabilities = probabilities.astype(np.float32)
        expected[i, indices] = probabilities
        positive = probabilities > 0
        entropy += float(-np.sum(probabilities[positive] * np.log(probabilities[positive])))
        top1 += float(probabilities.max())
        legal_count += len(indices)
    original = raw.compact_index_for_move
    calls = []

    def counting(board: chess.Board, move: chess.Move) -> int:
        calls.append(move)
        return original(board, move)

    monkeypatch.setattr(raw, 'compact_index_for_move', counting)
    target, attrs = label(source, logits)
    assert len(calls) == legal_count
    group: Any = zarr.open_group(str(target), mode='r')
    keys = np.frombuffer(b''.join(raw.position_fingerprints(planes,
        input_history_encoding=raw.derive.INPUT_HISTORY_ENCODING)), dtype=np.uint8).reshape(-1, 16)
    for name, array in [(raw.POLICY_FIELD, expected), (raw.INPUT_KEY_FIELD, input_keys),
                        (raw.SOURCE_KEY_FIELD, keys), (raw.GAME_ID_FIELD, games), (raw.PLY_FIELD, plies)]:
        actual = np.asarray(group[name][:])
        assert actual.dtype == array.dtype
        np.testing.assert_array_equal(actual, array)
        assert attrs[f'{name}_sha256'] == raw.sha_array(array)
    assert attrs['bt4_entropy_sum'] == entropy
    assert attrs['bt4_top1_sum'] == top1
    assert attrs['legal_moves_sum'] == legal_count
    raw.validate_existing(raw.PendingShard(source, source.inventory.shards[0], len(rows), target),
                          onnx_sha256='fake-model-hash', policy_output='policy', providers=['fake'])


@pytest.mark.parametrize('corruption', ['duplicate', 'negative', 'out_of_range', 'logit_shape'])
def test_actual_labeler_refuses_bad_compact_mapping_or_logits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str,
) -> None:
    source, _rows = make_multiboard_source(tmp_path)
    if corruption != 'logit_shape':
        index = {'duplicate': 0, 'negative': -1, 'out_of_range': raw.COMPACT_POLICY_SIZE}[corruption]
        monkeypatch.setattr(raw, 'compact_index_for_move', lambda _board, _move: index)
    logits = logits_for() if corruption != 'logit_shape' else np.zeros((raw.COMPACT_POLICY_SIZE, 2))
    with pytest.raises(ValueError, match=r'legal policy mapping mismatch|invalid BT4 legal policy'):
        label(source, logits)
    assert not list(source.out_dir.glob('*.bt4.zarr'))
