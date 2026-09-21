"""External eager/encoder/tree oracle for the no-Python native neural engine.

Reference models and CBoard are never imported, invoked or linked by the candidate.
All numerical criteria are fixed before the hosted run. No performance/strength gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding.features import extra_feature_planes_fast
from chess_anti_engine.encoding.lc0 import encode_lc0_full
from native.bend_engine.neural_probe.checkpoint import OutputTuple, load_checkpoint
from native.bend_engine.session_probe import run_probe as sessions
from native.bend_engine.session_probe.root_protocol import board_position
from native.bend_engine.session_probe.search_draws import automatic_draw
from ..verify import Client, FENS, assert_state, words
from ..verify_policy import entry
from ..verify_rules import position


def floats(values: list[int]) -> np.ndarray:
    assert all(0 <= v <= 0xffffffff for v in values)
    result = np.asarray(values, dtype=np.uint32).view(np.float32)
    assert np.isfinite(result).all()
    return result


def softmax(values: np.ndarray) -> np.ndarray:
    x = values.astype(np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def leaf_board(root: chess.Board, ref: sessions.Reference, node: int) -> chess.Board:
    keys = []
    while node:
        keys.append(ref.nodes[node].key)
        node = ref.nodes[node].parent
    board = root.copy(stack=True)
    for key in reversed(keys):
        move = next(m for m in board.legal_moves if entry(board, m)[1] == key)
        board.push(move)
    return board


def verify(command: list[str], checkpoint: Path, oracle_path: Path) -> dict[str, object]:
    from chess_anti_engine.encoding._lc0_ext import CBoard
    rep_fix.apply(True)
    torch.set_num_threads(2)
    loaded = load_checkpoint(checkpoint)
    assert loaded.encoding.history_rep_fix
    model = OutputTuple(loaded.model).eval()
    channels = loaded.encoding.channels
    layout_id = ('lc0_root', 'lc0_root_legacy_meta').index(loaded.encoding.input_history_encoding) + 1
    oracle = sessions.Oracle(oracle_path, with_python_chess=True)
    os.environ['DEEPFIN_BEND_TRACE'] = '1'
    client = Client(command)
    searches: list[dict[str, object]] = []
    payload = hashlib.sha256()
    rows_checked = draws = sequence = 0
    error_logits = error_probabilities = 0.0

    def search(root: chess.Board, label: str) -> chess.Move | None:
        nonlocal rows_checked, draws, sequence, error_logits, error_probabilities
        assert client.sync(position(root)) == ['readyok']
        assert_state(client, root)
        before_sequence = sequence
        ref = sessions.Reference(board_position(root), oracle, cap=4096, depth=2, budget=4)
        client.send('go nodes 4 depth 2\n')
        text = client.until('bestmove ', timeout=90)
        i = 0
        while i < len(text):
            line = text[i]
            if line.startswith('info string rule_draw '):
                fields = line.split()[3:]
                wanted = ref.next()
                assert wanted is not None, line
                assert wanted == int(fields[0]), line
                board = leaf_board(root, ref, wanted)
                assert automatic_draw(board) is not None, (label, board.fen(), line)
                ref.accept_draw(wanted)
                draws += 1
                i += 1
            elif line.startswith('info string neural_leaf '):
                seq, epoch, request, node, width = map(int, line.split()[3:])
                sequence += 1
                assert seq == sequence
                assert epoch == 1
                assert width == channels
                wanted = ref.next()
                assert wanted == node
                assert request == ref.seq
                board = leaf_board(root, ref, node)
                assert automatic_draw(board) is None
                assert text[i + 1] == f'info string neural_board {words(board)} {board.halfmove_clock} {board.fullmove_number}'
                i += 2
                actions = []
                while text[i].startswith('info string neural_action '):
                    actions.append(tuple(map(int, text[i].split()[3:])))
                    i += 1
                assert sorted(actions) == sorted(entry(board, m)[1:] for m in board.legal_moves)
                assert len(actions) == len({a[2] for a in actions}) > 0
                data: list[int] = []
                for plane in range(channels):
                    fields = text[i].split()
                    assert fields[:4] == ['info', 'string', 'neural_input', str(plane)]
                    assert len(fields) == 68
                    data.extend(map(int, fields[4:]))
                    i += 1
                tensor = floats(data).reshape(channels, 8, 8)
                cb = CBoard.from_board(board)
                expected_tensor = cb.encode_full(layout_id, channels - 112)
                np.testing.assert_array_equal(tensor.view(np.uint32), expected_tensor.view(np.uint32))
                python_tensor = np.concatenate((encode_lc0_full(board, input_history_encoding=loaded.encoding.input_history_encoding),
                                                extra_feature_planes_fast(board, version=loaded.encoding.input_extra_features)), axis=0)
                exact_planes = min(channels, 173)
                np.testing.assert_array_equal(tensor[:exact_planes].view(np.uint32), python_tensor[:exact_planes].view(np.uint32))
                if channels == 175:
                    np.testing.assert_allclose(tensor[173:], python_tensor[173:], atol=1.2e-7, rtol=0)
                raw = []
                for slot in range(1861):
                    fields = text[i].split()
                    assert fields[:4] == ['info', 'string', 'neural_logit', str(slot)]
                    assert len(fields) == 5
                    raw.append(int(fields[4]))
                    i += 1
                logits = floats(raw)
                with torch.inference_mode():
                    eager_policy, eager_wdl = model(torch.from_numpy(tensor.copy()[None]))
                eager = np.concatenate((eager_policy[0].cpu().numpy(), eager_wdl[0].cpu().numpy()))
                np.testing.assert_allclose(logits, eager, atol=2e-6, rtol=2e-5)
                error_logits = max(error_logits, float(np.abs(logits - eager).max()))
                prob_fields = list(map(int, text[i].split()[3:]))
                assert text[i].startswith('info string neural_prob ')
                assert len(prob_fields) == 5
                status, wb, db, lb, count = prob_fields
                assert status == 0
                assert count == len(actions)
                wdl = floats([wb, db, lb])
                i += 1
                priors_bits = []
                for index in range(count):
                    fields2 = text[i].split()
                    assert fields2[:4] == ['info', 'string', 'neural_prior', str(index)]
                    priors_bits.append(int(fields2[4]))
                    i += 1
                priors = floats(priors_bits)
                expected_priors = softmax(logits[[a[2] for a in actions]])
                expected_wdl = softmax(logits[1858:])
                np.testing.assert_allclose(priors, expected_priors, atol=6e-7, rtol=2e-5)
                np.testing.assert_allclose(wdl, expected_wdl, atol=6e-7, rtol=2e-5)
                error_probabilities = max(error_probabilities, float(np.abs(priors - expected_priors).max()),
                                          float(np.abs(wdl - expected_wdl).max()))
                assert abs(float(priors.astype(np.float64).sum()) - 1) < 2e-6
                assert abs(float(wdl.astype(np.float64).sum()) - 1) < 2e-6
                assert text[i] == 'info string neural_end'
                ref.accept(node, [a[0] for a in actions], list(map(float, wdl)), list(map(float, priors)))
                payload.update(tensor.astype('<f4', copy=False).tobytes())
                payload.update(logits.astype('<f4', copy=False).tobytes())
                rows_checked += 1
                i += 1
            elif line.startswith('info string neural_tree '):
                epoch, n, used, stop, pending, request = map(int, line.split()[3:])
                assert ref.next() is None
                result = [epoch, n, used, stop, pending, request]
                nodes = []
                i += 1
                for _ in range(used):
                    assert text[i].startswith('info string neural_node ')
                    row = list(map(int, text[i].split()[3:]))
                    assert len(row) == 30
                    index, *rest = row
                    board_words = rest[:19]
                    key, parent, first, count, visits, w, prior, value, status, depth = rest[19:]
                    nodes.append([index, parent, key, depth, status, visits, w, prior, value, first, count, *board_words])
                    i += 1
                assert text[i] == 'info string neural_tree_end'
                children = [node for node in ref.nodes[1:] if node.parent == 0]
                best = min(children, key=lambda a: (-a.n, a.key)).key if children else sessions.SENTINEL
                ref.check_snapshot(nodes, result, best, epoch=1)
                i += 1
            else:
                i += 1
        assert any(row.startswith('info string neural_tree ') for row in text)
        assert any(row.startswith('info nodes 4 string ') for row in text)
        bestmove = text[-1].split()[1]
        choices = list(root.legal_moves)
        if choices:
            move = chess.Move.from_uci(bestmove)
            assert move in choices
            children = [node for node in ref.nodes[1:] if node.parent == 0]
            if children:
                key = min(children, key=lambda a: (-a.n, a.key)).key
                assert entry(root, move)[1] == key
                assert not any('fallback' in row for row in text)
        else:
            assert bestmove == '0000'
            move = None
        assert_state(client, root)
        if root.outcome(claim_draw=False) is not None:
            assert before_sequence == sequence, label
        searches.append({'label': label, 'bestmove': bestmove, 'nodes': len(ref.nodes),
                         'completed': ref.completed, 'native_forwards': sequence - before_sequence})
        return move

    try:
        root = chess.Board()
        for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2:
            root.push_uci(uci)
        for ply in range(4):
            move = search(root, f'played-{ply}')
            assert move is not None
            root.push(move)
        for index, fen in enumerate(FENS[1:]):
            search(chess.Board(fen), f'special-{index}')
        search(chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 150 1'), 'automatic-75')
        search(chess.Board('4k3/8/8/8/8/8/8/3BK3 w - - 0 1'), 'material')
        search(chess.Board(), 'rewind-start')
    finally:
        client.close()
    assert rows_checked > 30
    assert draws >= 2
    return {'status': 'passed', 'scope': 'Bend selected leaves -> complete inputs -> native AOTI -> Bend probabilities -> native tree',
            'checkpoint': loaded.identity, 'searches': searches, 'native_forward_rows': rows_checked,
            'automatic_draw_replies': draws, 'model_sequences': sequence,
            'max_absolute_logit_error': error_logits, 'max_absolute_probability_error': error_probabilities,
            'ordered_input_output_sha256': payload.hexdigest(), 'tree_snapshots_checked': len(searches),
            'runtime_python': False, 'inference_backend': 'C++ AOTI CPU float32; not a Bend-written transformer'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    report = verify(args.command, args.checkpoint, args.oracle)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
