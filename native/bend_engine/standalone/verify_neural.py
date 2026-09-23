"""External selected-leaf/native model oracle. Never launched by the engine.

Requires the exact saved checkpoint used to export/bind a CPU batch-one package.
Verifies actual runtime inputs/outputs via the optional append-only native trace,
then reconciles selected paths and best moves against the existing PUCT reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
from typing import BinaryIO
from unittest.mock import patch

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.lc0 import encode_lc0_full
from chess_anti_engine.encoding.features import extra_feature_planes_fast
from chess_anti_engine.moves.encode import FULL_TO_COMPACT_POLICY, move_to_index
from native.bend_engine.session_probe import run_probe as sessions
from native.bend_engine.session_probe.search_draws import automatic_draw
from native.bend_engine.neural_probe.adapter import board_position, decode_key, probabilities
from native.bend_engine.neural_probe.checkpoint import EagerReference, load_checkpoint
from .verify import Client, assert_state
from .verify_policy import key
from .verify_rules import position


def floats(words: list[int]) -> np.ndarray:
    return np.asarray(words, dtype=np.uint32).view(np.float32)


def trace_row(stream: BinaryIO, sequence: int, channels: int) -> tuple[np.ndarray, np.ndarray]:
    header = stream.read(16)
    assert len(header) == 16, 'missing actual native call'
    assert struct.unpack('<4I', header) == (0x44464c31, sequence, channels * 64, 1861)
    raw = stream.read((channels * 64 + 1861) * 4)
    assert len(raw) == (channels * 64 + 1861) * 4
    data = np.frombuffer(raw, dtype='<f4').copy()
    return data[:channels * 64].reshape(1, channels, 8, 8), data[channels * 64:]


def history_path(ref: sessions.Reference, node: int) -> list[int]:
    path = []
    while node:
        path.append(ref.nodes[node].key)
        node = ref.nodes[node].parent
    return path[::-1]


def descendant(root: chess.Board, path: list[int]) -> chess.Board:
    leaf = root.copy(stack=True)
    for k in path:
        leaf.push(decode_key(leaf, k))
    return leaf


def fixtures() -> list[chess.Board]:
    roots = [chess.Board(fen) for fen in [
        chess.STARTING_FEN,
        'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
        '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
        '4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1',
        '4k3/P7/8/8/8/8/8/4K3 w - - 0 1',
        '4k3/8/8/8/8/8/p7/4K3 b - - 0 1',
        '4k3/8/8/8/8/8/8/R3K3 w - - 149 1',
        '4k3/8/8/8/8/8/8/R3K3 w - - 150 1',
        '4k3/8/8/8/8/8/8/3BK3 w - - 0 1',
        'k7/1Q6/2K5/8/8/8/8/8 b - - 150 1',
        'k7/2Q5/2K5/8/8/8/8/8 b - - 0 1',
    ]]
    history = chess.Board()
    for uci in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2:
        history.push_uci(uci)
    roots.append(history)
    for moves in [('g1f3', 'g8f6', 'b1c3', 'b8c6'), ('b1c3', 'b8c6', 'g1f3', 'g8f6')]:
        root = chess.Board()
        for uci in moves:
            root.push_uci(uci)
        roots.append(root)
    return roots


def verify(command: list[str], package: Path, checkpoint: Path, oracle_binary: Path,
           trace_path: Path, *, runtime_package: str | None = None,
           runtime_trace: str | None = None) -> dict[str, object]:
    torch.set_num_threads(2)
    rep_fix.apply(True)
    manifest = json.loads(package.with_suffix('.json').read_text())
    assert manifest['batch'] == 1
    assert manifest['device'] == 'cpu'
    assert manifest['history_rep_fix']
    assert hashlib.sha256(package.read_bytes()).hexdigest() == manifest['sha256']
    loaded = load_checkpoint(checkpoint, weights_key=manifest['weights_key'])
    assert loaded.identity['checkpoint_sha256'] == manifest['checkpoint_sha256']
    eager = EagerReference(loaded.model, torch.device('cpu'), torch.float32)
    encoding = loaded.encoding
    oracle = sessions.Oracle(oracle_binary, with_python_chess=True)
    environment = {'DEEPFIN_BEND_MODEL_PACKAGE': runtime_package or str(package),
                   'DEEPFIN_BEND_MODEL_TRACE': runtime_trace or str(trace_path),
                   'DEEPFIN_BEND_NATIVE_DIAGNOSTICS': '1'}
    sequence = replies = searches = rule_draws = zeros = policy_values = 0
    max_logits = max_probabilities = 0.0
    root_tensors: list[np.ndarray] = []
    results: list[dict[str, object]] = []
    with patch.dict(os.environ, environment):
        client = Client(command)
    stream = trace_path.open('rb')
    try:
        def search(root: chess.Board, *, remember: bool = False) -> str:
            nonlocal sequence, replies, searches, rule_draws, zeros, max_logits, max_probabilities, policy_values
            assert client.sync(position(root)) == ['readyok']
            ref = sessions.Reference(board_position(root), oracle, cap=4096, depth=2, budget=4)
            client.send('go nodes 4 depth 2\n')
            lines = client.until('bestmove ', timeout=60)
            paths: dict[tuple[int, int, int], list[int]] = {}
            seen = 0
            for line in lines:
                fields = line.split()
                if fields[:3] == ['info', 'string', 'rule_draw']:
                    node = ref.next()
                    assert node is not None
                    assert node == int(fields[3])
                    leaf = descendant(root, history_path(ref, node))
                    assert automatic_draw(leaf) is not None
                    ref.accept_draw(node)
                    rule_draws += 1
                elif fields[:3] == ['info', 'string', 'native_path']:
                    epoch, request, node = map(int, fields[3:6])
                    identity = epoch, request, node
                    assert identity not in paths
                    paths[identity] = list(map(int, fields[6:]))
                elif fields[:3] == ['info', 'string', 'native_reply']:
                    values = list(map(int, fields[3:]))
                    epoch, request, node = values[:3]
                    assert (epoch, request, node) == (searches + 1, ref.seq, ref.next())
                    path = paths.pop((epoch, request, node))
                    assert path == history_path(ref, node)
                    leaf = descendant(root, path)
                    assert board_position(leaf) == ref.nodes[node].board
                    assert automatic_draw(leaf) is None
                    n = values[6]
                    assert 1 <= n <= 256
                    assert len(values) == 7 + 2 * n
                    action_keys, ps = values[7::2], floats(values[8::2])
                    assert set(action_keys) == {key(leaf, move) for move in leaf.legal_moves}
                    assert len(set(action_keys)) == n
                    observed_wdl = floats(values[3:6])
                    sequence += 1
                    x, output = trace_row(stream, sequence, encoding.channels)
                    cb = CBoard.from_board(leaf)
                    c_input = cb.encode_full(1 if encoding.input_history_encoding == 'lc0_root' else 2,
                                            encoding.channels - 112)
                    np.testing.assert_array_equal(x[0].view(np.uint32), c_input.view(np.uint32))
                    py = np.concatenate((encode_lc0_full(leaf, input_history_encoding=encoding.input_history_encoding),
                                         extra_feature_planes_fast(leaf, version=encoding.input_extra_features)))
                    exact = 173 if encoding.channels == 175 else 146
                    np.testing.assert_array_equal(x[0, :exact].view(np.uint32), py[:exact].view(np.uint32))
                    if exact == 173:
                        np.testing.assert_allclose(x[0, 173:], py[173:], atol=1.2e-7, rtol=0)
                    if node == 0 and remember:
                        root_tensors.append(x.copy())
                    with torch.no_grad():
                        expected = eager(torch.from_numpy(x))
                    expected_logits = np.concatenate((expected['policy'].numpy()[0], expected['wdl'].numpy()[0]))
                    np.testing.assert_allclose(output, expected_logits, atol=2e-6, rtol=2e-5)
                    max_logits = max(max_logits, float(np.abs(output - expected_logits).max()))
                    full = np.array([move_to_index(decode_key(leaf, k), leaf) for k in action_keys], dtype=np.int64)
                    assert len(set(FULL_TO_COMPACT_POLICY[full].tolist())) == n
                    expected_wdl, expected_ps = probabilities(output[:1858][None], output[1858:][None], full)
                    np.testing.assert_allclose(observed_wdl, expected_wdl, atol=2e-7, rtol=3e-6)
                    np.testing.assert_allclose(ps, expected_ps, atol=2e-7, rtol=3e-6)
                    max_probabilities = max(max_probabilities, float(np.abs(ps - expected_ps).max()),
                                            float(np.abs(observed_wdl - expected_wdl).max()))
                    assert abs(float(ps.sum()) - 1) < 1e-6
                    assert abs(float(observed_wdl.sum()) - 1) < 1e-6
                    ref.accept(node, action_keys, observed_wdl.tolist(), ps.tolist())
                    seen += 1
                    replies += 1
                    policy_values += n
            assert not paths
            assert ref.next() is None
            metrics = [json.loads(line.removeprefix('info string neural_work ')) for line in lines
                       if line.startswith('info string neural_work ')]
            assert len(metrics) == 1, 'missing/duplicate per-search work report'
            work = metrics[0]
            assert work['schema'] == 'deepfin.neural-work.v1'
            assert work['search_epoch'] == searches + 1
            assert work['completed_simulations'] == ref.completed
            assert work['executed_real_rows'] == work['accepted_neural_rows'] == work['forward_calls'] == seen
            assert work['dispatched_real_rows'] == seen
            assert work['padded_rows'] == work['rejected_rows'] == work['failed_forward_rows'] == 0
            assert work['phase_seconds']['gpu'] is None
            assert ref.completed == 4
            assert ref.stop == 0
            assert any(line.startswith('info nodes 4 ') for line in lines)
            children = [a for i, a in enumerate(ref.nodes) if i and a.parent == 0]
            if children:
                best = min(children, key=lambda a: (-a.n, a.key)).key
            else:
                best = min((key(root, m) for m in root.legal_moves), default=sessions.SENTINEL)
            expected_move = decode_key(root, best).uci() if best != sessions.SENTINEL else '0000'
            move = lines[-1].split()[1]
            assert move == expected_move, (root.fen(), move, expected_move)
            if not seen:
                assert root.is_game_over(claim_draw=False)
                zeros += 1
            else:
                assert not any('unsearched' in line for line in lines)
            searches += 1
            results.append({'root': root.fen(en_passant='fen'), 'history_plies': len(root.move_stack),
                            'native_calls': seen, 'nodes': len(ref.nodes), 'completed': ref.completed,
                            'bestmove': move})
            assert_state(client, root)
            return move

        for root in fixtures():
            assert root.is_valid()
            search(root, remember=len(root.move_stack) == 4)
        assert len(root_tensors) == 2
        assert not np.array_equal(root_tensors[0][:, :104], root_tensors[1][:, :104])
        # Successive new roots in one process, with a model loaded only at startup.
        game = chess.Board()
        for _ in range(4):
            move = search(game)
            game.push_uci(move)
        assert not stream.read(1), 'unaccounted model calls'
    finally:
        stream.close()
        client.close()
    return {'status': 'passed', 'scope': 'Bend-controlled native CPU neural leaves and UCI; untrained fixture when so exported',
            'searches': searches, 'native_calls': sequence, 'reconciled_replies': replies,
            'legal_priors_compared': policy_values, 'automatic_draw_replies': rule_draws,
            'zero_forward_terminal_searches': zeros, 'max_logit_absolute_error': max_logits,
            'max_probability_absolute_error': max_probabilities, 'same_board_history_distinct': True,
            'oracle_positions': len(oracle.cache), 'results': results,
            'package_sha256': manifest['sha256'], 'checkpoint_sha256': manifest['checkpoint_sha256'],
            'parameter_count': loaded.identity['parameter_count'], 'torch_version': str(torch.__version__)}


def failures(command: list[str], package: Path, directory: Path) -> int:
    original = hashlib.sha256(package.read_bytes()).hexdigest()
    bad = directory / 'bad.pt2'
    bad.write_bytes(b'not the bound package')
    environments = [None, str(bad), str(package)]
    for value in environments:
        env = os.environ.copy()
        env.pop('DEEPFIN_BEND_MODEL_PACKAGE', None)
        env.pop('DEEPFIN_BEND_MODEL_TRACE', None)
        if value is not None:
            env['DEEPFIN_BEND_MODEL_PACKAGE'] = value
        if value == str(package):
            env['DEEPFIN_BEND_MODEL_TRACE'] = str(package)  # must not truncate it
        p = subprocess.run(command, input='uci\nisready\n', text=True, capture_output=True, env=env, timeout=30, check=False)
        assert p.returncode == 2, (p.returncode, p.stderr)
        assert 'native model startup:' in p.stderr
        assert 'uciok' not in p.stdout
        assert 'bestmove' not in p.stdout
    assert hashlib.sha256(package.read_bytes()).hexdigest() == original
    return len(environments)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--oracle', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--trace-path', type=Path)
    p.add_argument('--runtime-package')
    p.add_argument('--runtime-trace')
    p.add_argument('--skip-startup-failures', action='store_true')
    p.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = p.parse_args()
    if not args.command:
        p.error('--command requires an executable')
    with tempfile.TemporaryDirectory(prefix='bend-neural-check-') as tmp:
        trace = args.trace_path or Path(tmp) / 'trace.bin'
        report = verify(args.command, args.package.resolve(), args.checkpoint.resolve(), args.oracle.resolve(), trace,
                        runtime_package=args.runtime_package, runtime_trace=args.runtime_trace)
        report['startup_failures_rejected'] = 0 if args.skip_startup_failures else failures(args.command, args.package.resolve(), Path(tmp))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
