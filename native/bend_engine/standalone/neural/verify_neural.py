"""External oracle for actual Bend-owned leaf/model/search composition.

Never imported by the engine. Trace mode records the actual native model inputs
and raw outputs; reference chess, encoders, eager model and PUCT run only here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.features import extra_feature_planes_fast
from chess_anti_engine.encoding.lc0 import encode_lc0_full
from chess_anti_engine.moves.encode import FULL_TO_COMPACT_POLICY, move_to_index
from native.bend_engine.legal_probe import run_probe as rules
from native.bend_engine.session_probe import run_probe as sessions
from native.bend_engine.session_probe.root_protocol import board_position, packed_move
from native.bend_engine.session_probe.search_draws import automatic_draw
from native.bend_engine.neural_probe.adapter import decode_key
from native.bend_engine.neural_probe.checkpoint import OutputTuple, load_checkpoint
from ..verify import Client, assert_state
from ..verify_rules import position


class Oracle(sessions.Oracle):
    """Independent pure-Python transitions, with CBoard inputs checked per call."""
    def __init__(self) -> None:
        super().__init__(Path('/not-executed'))

    def moves(self, board: rules.Position) -> dict[int, rules.Position]:
        b = board
        if b not in self.cache:
            root = chess.Board(None)
            for piece in range(6):
                for square in chess.scan_forward(b[piece]):
                    root.set_piece_at(square, chess.Piece(piece + 1, bool(b[6] & 1 << square)))
            root.turn = bool(b[8])
            root.castling_rights = sum(1 << sq for i, sq in enumerate((7, 0, 63, 56)) if b[9] & 1 << i)
            root.ep_square = None if b[10] == 64 else b[10]
            children = {}
            for m in root.legal_moves:
                child = root.copy(stack=False)
                child.push(m)
                children[packed_move(root, m)] = board_position(child)
            self.cache[b] = children
            if not children:
                self.terminal[b] = -1.0 if root.is_checkmate() else 0.0
        return self.cache[b]


def leaf_board(root: chess.Board, ref: sessions.Reference, node: int) -> chess.Board:
    path = []
    while node:
        path.append(ref.nodes[node].key)
        node = ref.nodes[node].parent
    leaf = root.copy(stack=True)
    for key in reversed(path):
        leaf.push(decode_key(leaf, key))
    return leaf


def traces(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray, int]]:
    words = np.frombuffer(path.read_bytes(), dtype='<u4')
    records = {}
    i = 0
    while i < len(words):
        assert len(words) - i >= 6
        magic, seq, channels, batch, count, out = map(int, words[i:i + 6])
        assert magic == 0x44464c31
        assert seq == len(records) + 1
        assert channels in (146, 175)
        assert batch in (1, 2, 4, 8, 16)
        assert count == channels * 64
        assert out == 1861
        i += 6
        assert len(words) - i >= count + out
        x = words[i:i + count].copy().view('<f4').reshape(channels, 8, 8)
        logits = words[i + count:i + count + out].copy().view('<f4')
        assert np.isfinite(x).all()
        assert np.isfinite(logits).all()
        records[seq] = x, logits, batch
        i += count + out
    return records


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    y = np.exp(x - x.max())
    return (y / y.sum()).astype(np.float32)


def fixtures() -> list[tuple[str, chess.Board]]:
    rows = [('start', chess.Board()),
            ('kiwipete', chess.Board(rules.CANONICAL[1][1])),
            ('ep', chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1')),
            ('promotion', chess.Board('4k3/P7/8/8/8/8/8/4K3 w - - 0 1')),
            ('black-promotion', chess.Board('4k3/8/8/8/8/8/p7/4K3 b - - 0 1')),
            ('near75', chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 149 1'))]
    root = chess.Board()
    for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2:
        root.push_uci(move)
    for label, text in [('history', ''), ('order-a', 'g1f3 g8f6 b1c3 b8c6'),
                        ('order-b', 'b1c3 b8c6 g1f3 g8f6')]:
        b = root.copy(stack=True)
        for move in text.split():
            b.push_uci(move)
        rows.append((label, b))
    repeated = chess.Board()
    for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 4:
        repeated.push_uci(move)
    rows += [('fivefold', repeated), ('75', chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 150 1')),
             ('material', chess.Board('4k3/8/8/8/8/8/8/3BK3 w - - 0 1')),
             ('mate', chess.Board('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')),
             ('stalemate', chess.Board('7k/5Q2/5K2/8/8/8/8/8 b - - 0 1'))]
    return rows


def verify(command: list[str], checkpoint: Path, package: Path, trace: Path) -> dict[str, object]:
    rep_fix.apply(True)
    torch.set_num_threads(2)
    loaded = load_checkpoint(checkpoint)
    eager = OutputTuple(loaded.model).eval()
    metadata = json.loads(package.with_suffix('.json').read_text())
    assert metadata['checkpoint_sha256'] == loaded.identity['checkpoint_sha256']
    history = metadata['input_history_encoding']
    features = metadata['input_extra_features']
    client = Client(command)
    jobs = []
    try:
        for label, root in fixtures():
            assert root.is_valid(), label
            assert client.sync(position(root)) == ['readyok']
            client.send('go nodes 8 depth 3\n')
            rows = client.until('bestmove ', timeout=60)
            assert any(r.startswith('info nodes 8 ') for r in rows), rows
            assert_state(client, root)
            jobs.append((label, root, rows))
        # Normal UCI stop/result holding and subsequent new-game search reuse the
        # same native model, without promising to interrupt a synchronous call.
        client.send('go infinite nodes 1\n')
        rows: list[str] = []
        client.send('isready\n')
        rows += client.until('readyok')
        assert not any(r.startswith('bestmove ') for r in rows)
        client.send('stop\nstop\n')
        client.until('bestmove ', timeout=30)
        assert client.sync('ucinewgame') == ['readyok']
        root = chess.Board()
        client.send('go nodes 8 depth 3\n')
        rows = client.until('bestmove ', timeout=60)
        jobs.append(('reset', root, rows))
        assert_state(client, root)
    finally:
        client.close()

    records = traces(trace)
    seen = set()
    oracle = Oracle()
    model_rows = rules_rows = 0
    max_logit_error = max_probability_error = max_storm_error = 0.0
    results = []
    root_inputs = {}
    for label, root, rows in jobs:
        ref = sessions.Reference(board_position(root), oracle, cap=4096, depth=3, budget=8)
        wanted = ref.next()
        i = calls = draws = 0
        while i < len(rows):
            line = rows[i]
            if line.startswith('info string rule_draw '):
                assert wanted is not None
                values = line.split()
                assert int(values[3]) == wanted
                leaf = leaf_board(root, ref, wanted)
                assert automatic_draw(leaf) is not None
                ref.accept_draw(wanted)
                rules_rows += 1
                draws += 1
                wanted = ref.next()
            elif line.startswith('info string model_reply '):
                assert wanted is not None
                seq, epoch, request, node, count, *wb = map(int, line.split()[3:])
                assert epoch == 1
                assert request == ref.seq
                assert node == wanted
                assert seq in records
                assert seq not in seen
                seen.add(seq)
                x, logits, batch = records[seq]
                assert batch == metadata['batch']
                leaf = leaf_board(root, ref, node)
                assert automatic_draw(leaf) is None
                cb = CBoard.from_board(leaf)
                cx = cb.encode_full(1 if history == 'lc0_root' else 2, x.shape[0] - 112)
                np.testing.assert_array_equal(x.view(np.uint32), cx.view(np.uint32))
                px = np.concatenate((encode_lc0_full(leaf, input_history_encoding=history),
                                     extra_feature_planes_fast(leaf, version=features)), axis=0)
                exact = 173 if x.shape[0] == 175 else 146
                np.testing.assert_array_equal(x[:exact].view(np.uint32), px[:exact].view(np.uint32))
                if exact == 173:
                    np.testing.assert_allclose(x[173:], px[173:], atol=1.2e-7, rtol=0)
                    max_storm_error = max(max_storm_error, float(np.abs(x[173:] - px[173:]).max()))
                with torch.inference_mode():
                    p, w = eager(torch.from_numpy(x.copy()).unsqueeze(0))
                expected_logits = np.concatenate((p[0].float().numpy(), w[0].float().numpy()))
                np.testing.assert_allclose(logits, expected_logits, atol=2e-6, rtol=2e-5)
                max_logit_error = max(max_logit_error, float(np.abs(logits - expected_logits).max()))
                keys, slots, prior_bits, fulls = [], [], [], []
                for _ in range(count):
                    i += 1
                    parts = rows[i].split()
                    assert parts[:3] == ['info', 'string', 'model_prior']
                    key, full, compact, bits = map(int, parts[3:])
                    move = decode_key(leaf, key)
                    expected_full = move_to_index(move, leaf)
                    assert full == expected_full
                    assert compact == int(FULL_TO_COMPACT_POLICY[full])
                    keys.append(key)
                    slots.append(compact)
                    prior_bits.append(bits)
                    fulls.append(full)
                i += 1
                assert rows[i] == 'info string model_reply_end'
                assert set(keys) == set(oracle.moves(board_position(leaf)))
                assert len(set(keys)) == count == len(set(slots))
                assert sorted(fulls) == sorted(map(int, cb.legal_move_indices()))
                priors = np.array(prior_bits, dtype=np.uint32).view(np.float32)
                wdl = np.array(wb, dtype=np.uint32).view(np.float32)
                expected_p, expected_w = softmax(logits[slots]), softmax(logits[1858:])
                np.testing.assert_allclose(priors, expected_p, atol=2e-6, rtol=2e-5)
                np.testing.assert_allclose(wdl, expected_w, atol=2e-6, rtol=2e-5)
                max_probability_error = max(max_probability_error, float(np.abs(priors - expected_p).max()),
                                            float(np.abs(wdl - expected_w).max()))
                if node == 0:
                    root_inputs[label] = x
                ref.accept(node, keys, list(map(float, wdl)), list(map(float, priors)))
                wanted = ref.next()
                model_rows += 1
                calls += 1
            i += 1
        assert wanted is None
        assert ref.completed == 8
        assert ref.stop == 0
        candidates = [n for n in ref.nodes[1:] if n.parent == 0]
        expected_key = max(candidates, key=lambda n: (n.n, -n.key)).key if candidates else min(
            oracle.moves(board_position(root)), default=sessions.SENTINEL)
        actual = rows[-1].split()[1]
        assert actual == ('0000' if expected_key == sessions.SENTINEL else decode_key(root, expected_key).uci())
        results.append({'label': label, 'simulations': ref.completed, 'model_calls': calls,
                        'rule_replies': draws, 'bestmove': actual})
    assert seen == set(records), 'unmatched or missing native trace rows'
    assert not np.array_equal(root_inputs['order-a'][:104], root_inputs['order-b'][:104])
    assert model_rows > 0
    assert rules_rows > 0
    return {'status': 'passed', 'scope': 'actual Bend search leaves, native CPU AOTI and Bend legal/WDL softmax',
            'fixture': 'existing untrained transformer; no strength or GPU claim',
            'results': results, 'model_rows': model_rows, 'rule_replies': rules_rows,
            'max_logit_error': max_logit_error, 'max_probability_error': max_probability_error,
            'max_python_storm_error': max_storm_error, 'same_board_history_difference': True,
            'model_sequence_persists_across_searches': True, 'torch_version': str(torch.__version__),
            'trace_sha256': hashlib.sha256(trace.read_bytes()).hexdigest(),
            'package_sha256': hashlib.sha256(package.read_bytes()).hexdigest()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = p.parse_args()
    if args.trace.exists() or args.report.exists():
        p.error('trace and report must be new paths')
    report = verify(args.command, args.checkpoint, args.package, args.trace)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
