"""Cheap boundary contracts only: no compilation, neural forward or tree traversal."""
from __future__ import annotations

import os
import hashlib
import json
from pathlib import Path
import time

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.moves.encode import COMPACT_TO_FULL_POLICY, move_to_index
from native.bend_engine.neural_probe.adapter import (
    Encoding, HistoryEncoder, board_position, decode_key, legal_keys, probabilities,
)
from native.bend_engine.neural_probe.backend import FORMAT, package_manifest, read_exact, write_all
from native.bend_engine.session_probe.run_probe import move_key, parse_path


@pytest.mark.parametrize('line', ['path', 'path 1', 'path 0 1', 'path 33', 'path -1',
                                    'path 1 131072', 'path x', 'path 1 4294967296'])
def test_bad_history_path(line: str) -> None:
    with pytest.raises(ValueError, match=r'path|malformed|nondecimal|exceeds'):
        parse_path(line)


def test_empty_and_full_history_path() -> None:
    assert parse_path('path 0') == []
    assert parse_path('path 32' + ' 1234' * 32) == [1234] * 32


@pytest.mark.parametrize('key', [-1, 1 << 17, 0, 7 << 12, 3 << 15, (28 << 6) | 12 | 1 << 15])
def test_bad_packed_move(key: int) -> None:
    with pytest.raises(ValueError, match=r'packed|illegal|flag'):
        decode_key(chess.Board(), key)


@pytest.mark.parametrize('fen', ['4k3/P7/8/8/8/8/8/4K3 w - - 0 1',
                                '4k3/8/8/8/8/8/p7/4K3 b - - 0 1',
                                'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1',
                                '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1'])
def test_legal_keys_preserve_special_moves(fen: str) -> None:
    board = chess.Board(fen)
    assert {decode_key(board, k) for k in legal_keys(board)} == set(board.legal_moves)


def test_compact_and_dense_logits_preserve_requested_order() -> None:
    board = chess.Board()
    full = np.array([move_to_index(chess.Move.from_uci(u), board) for u in ('g1f3', 'e2e4', 'a2a3')])
    compact = np.arange(1858, dtype=np.float32)[None] / 100
    dense = np.full((1, 4672), -1e9, dtype=np.float32)
    dense[:, COMPACT_TO_FULL_POLICY] = compact
    logits = np.array([[1, 2, -1]], dtype=np.float32)
    wdl, policy = probabilities(compact, logits, full)
    assert (wdl, policy) == probabilities(dense, logits, full)
    assert policy == list(reversed(probabilities(compact, logits, full[::-1])[1]))
    assert sum(wdl) == pytest.approx(1)
    assert wdl[1] > wdl[0] > wdl[2]


@pytest.mark.parametrize('bad', ['nan', 'shape', 'duplicate', 'index', 'nonintegral'])
def test_invalid_neural_output(bad: str) -> None:
    policy, wdl, full = np.zeros((1, 1858)), np.zeros((1, 3)), np.array([0, 1])
    if bad == 'nan':
        policy[0, 0] = np.nan
    elif bad == 'shape':
        wdl = wdl[:, :2]
    elif bad == 'duplicate':
        full = np.array([0, 0])
    elif bad == 'index':
        full = np.array([-1, 1])
    else:
        full = np.array([0.1, 1.0])
    with pytest.raises(ValueError, match=r'shape|nonfinite|indices'):
        probabilities(policy, wdl, full)


def test_history_adapter_rejects_wrong_board_and_missing_actions() -> None:
    root = chess.Board()
    # Do not change a process-global encoder regime over somebody else's boards.
    encoding = Encoding('lc0_root_legacy_meta', 'v1', rep_fix.current() or False)
    encoder = HistoryEncoder(root, encoding)
    key = move_key((12, 28, 0, 0))
    with pytest.raises(ValueError, match='does not match'):
        encoder.encode([key], board_position(root), legal_keys(root))
    with pytest.raises(ValueError, match='incomplete'):
        encoder.encode([], board_position(root), legal_keys(root)[:-1])
    with pytest.raises(ValueError, match='duplicate'):
        encoder.encode([], board_position(root), legal_keys(root) * 2)


def test_legacy_encoding_is_not_silently_substituted() -> None:
    with pytest.raises(ValueError, match='unsupported history'):
        Encoding('legacy', 'v1', True)


def test_native_pipe_deadline_and_truncation() -> None:
    r, w = os.pipe()
    with os.fdopen(r, 'rb', buffering=0) as reader, os.fdopen(w, 'wb', buffering=0) as writer:
        os.set_blocking(writer.fileno(), False)
        with pytest.raises(TimeoutError, match='deadline'):
            read_exact(reader, 4, time.monotonic())
        with pytest.raises(TimeoutError, match='deadline'):
            write_all(writer, b'1234', time.monotonic())
        write_all(writer, b'1234', time.monotonic() + 1)
        assert read_exact(reader, 4, time.monotonic() + 1) == b'1234'
    r, w = os.pipe()
    os.close(w)
    with os.fdopen(r, 'rb', buffering=0) as reader:
        with pytest.raises(RuntimeError, match='closed'):
            read_exact(reader, 4, time.monotonic() + 1)


def test_clock_overflow_is_rejected_before_cboard_push() -> None:
    root = chess.Board('4k3/8/8/8/8/8/8/4K3 w - - 255 1')
    encoder = HistoryEncoder(root, Encoding('lc0_root', 'v1', rep_fix.current() or False))
    move = next(iter(root.legal_moves))
    key = move_key((move.from_square, move.to_square, 0, 0))
    child = root.copy(stack=True)
    child.push(move)
    with pytest.raises(ValueError, match='rule50 exceeds'):
        encoder.encode([key], board_position(child), legal_keys(child))


@pytest.mark.parametrize('change', ['hash', 'format', 'torch_version', 'batch', 'policy_width',
                                    'channels', 'history_rep_fix', 'input_history_encoding', 'missing'])
def test_package_manifest_fails_closed(tmp_path: Path, change: str) -> None:
    package = tmp_path / 'fake.pt2'
    package.write_bytes(b'not executed by this parser test')
    manifest = {'format': FORMAT, 'torch_version': str(torch.__version__),
                'sha256': hashlib.sha256(package.read_bytes()).hexdigest(), 'batch': 1,
                'policy_width': 1858, 'channels': 175, 'history_rep_fix': True,
                'input_history_encoding': 'lc0_root_legacy_meta', 'input_extra_features': 'v2_threats'}
    package.with_suffix('.json').write_text(json.dumps(manifest))
    assert package_manifest(package)[1].channels == 175
    if change == 'hash':
        package.write_bytes(b'changed')
    elif change == 'missing':
        manifest.pop('input_extra_features')
    else:
        manifest[change] = 'invalid'
    package.with_suffix('.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=r'manifest|fingerprint|mismatch|encoding|batch'):
        package_manifest(package)
