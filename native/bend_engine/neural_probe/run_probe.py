#!/usr/bin/env python3
"""Opt-in Bend -> history-aware encoder -> native CPU AOTI -> search qualification.

The generated smoke package is actual project TinyNet code with seeded,
UNTRAINED weights. This is not trained-model, CUDA, throughput or strength evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import struct
import subprocess
from typing import TypedDict

import chess
import numpy as np
import torch

from chess_anti_engine.encoding.lc0 import encode_lc0_full
from chess_anti_engine.encoding.features import extra_feature_planes_fast
from chess_anti_engine.inference import _policy_output_full
from native.bend_engine.session_probe import run_probe as sessions
from native.bend_engine.legal_probe import run_probe as rules

from .adapter import Encoding, HistoryEncoder, board_position, decode_key, legal_keys, probabilities
from .backend import MAGIC, NativeEvaluator, build_worker, package_manifest
from .package import DEFAULT_ENCODING, SEED, export_smoke, smoke_model

ROOT = Path(__file__).resolve().parents[3]


def independently_encode(board: chess.Board, encoding: Encoding) -> np.ndarray:
    # Python history traversal, not a CBoard reconstructed from just the leaf FEN.
    lc0 = encode_lc0_full(board, input_history_encoding=encoding.input_history_encoding)
    extra = extra_feature_planes_fast(board, version=encoding.input_extra_features)
    return np.concatenate((lc0, extra))[None]


def check_encoding(x: np.ndarray, board: chess.Board, encoding: Encoding) -> None:
    expected = independently_encode(board, encoding)
    np.testing.assert_array_equal(x[:, :112], expected[:, :112])
    # Classical graded features use C float arithmetic versus Python's double
    # intermediates. Keep discrete/history planes exact; allow float rounding only.
    np.testing.assert_allclose(x[:, 112:], expected[:, 112:], atol=6e-8, rtol=2e-7)


def boundary_checks() -> dict[str, int]:
    # Fix remains enabled throughout this standalone process; never flip over live boards.
    root = chess.Board()
    for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8') * 2:
        root.push_uci(uci)
    boards = [root, chess.Board(), chess.Board(rules.CANONICAL[1][1])]
    for fen in ('4k3/P7/8/8/8/8/8/4K3 w - - 0 1',
                '4k3/8/8/8/8/8/p7/4K3 b - - 0 1',
                '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
                'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 7 1'):
        boards.append(chess.Board(fen))
    count = 0
    for extra in ('v1', 'v2_threats'):
        for history in ('lc0_root', 'lc0_root_legacy_meta'):
            encoding = Encoding(history, extra, True)
            for board in boards:
                enc = HistoryEncoder(board, encoding)
                actions = legal_keys(board)
                x, _, _ = enc.encode([], board_position(board), actions)
                check_encoding(x, board, encoding)
                # Every special move's child carries correctly updated history/metadata.
                for key in actions:
                    if key >> 12 == 0:
                        continue
                    child = board.copy(stack=True)
                    child.push(decode_key(child, key))
                    if not child.is_game_over(claim_draw=False):
                        y, _, _ = enc.encode([key], board_position(child), legal_keys(child))
                        check_encoding(y, child, encoding)
                        count += 1
                count += 1
    # Same board, different reversible history: a board-only cache is invalid.
    encoder = HistoryEncoder(chess.Board(), DEFAULT_ENCODING)
    values = []
    positions = []
    for moves in [('g1f3', 'g8f6', 'b1c3', 'b8c6'), ('b1c3', 'b8c6', 'g1f3', 'g8f6')]:
        b, path = chess.Board(), []
        for uci in moves:
            move = chess.Move.from_uci(uci)
            path.append(sessions.move_key((move.from_square, move.to_square, 0, 0)))
            b.push(move)
        positions.append(board_position(b))
        values.append(encoder.encode(path, positions[-1], legal_keys(b))[0])
    assert positions[0] == positions[1]
    assert not np.array_equal(values[0], values[1])
    return {'independent_encoding_checks': count, 'same_board_distinct_history': 1}


class SessionResult(TypedDict):
    requests: int
    completed: int
    nodes: int
    best: int
    unique_inputs: int
    max_logit_absolute_error: float


def worker_failures(binary: Path, package: Path, channels: int) -> int:
    """Native input guards, separate from model outputs and Bend reply guards."""
    count = channels * 64
    cases = [
        (struct.pack('<3I', 0, 1, count), 'magic'),
        (struct.pack('<3I', MAGIC, 2, count), 'sequence'),
        (struct.pack('<3I', MAGIC, 1, count - 1), 'size'),
        (struct.pack('<3I', MAGIC, 1, count), 'truncated'),
        (struct.pack('<3If', MAGIC, 1, count, float('nan')), 'nonfinite'),
        (struct.pack('<3If', MAGIC, 1, count, float('inf')), 'nonfinite'),
    ]
    for payload, message in cases:
        run = subprocess.run([str(binary), str(package), str(channels)], input=payload,
                             capture_output=True, timeout=30, check=False)
        if run.returncode != 2 or message not in run.stderr.decode(errors='replace'):
            raise AssertionError(f'native worker did not reject {message}: {run.stderr!r}')
        if run.stdout != struct.pack('<2I', MAGIC, 0):
            raise AssertionError('invalid worker input emitted a model reply')
    return len(cases)


def neural_session(peer: sessions.Peer, oracle: sessions.Oracle, root: chess.Board,
                   evaluator: NativeEvaluator, eager: torch.nn.Module, epoch: int, budget: int = 16) -> SessionResult:
    enc = HistoryEncoder(root, evaluator.encoding)
    board = board_position(root)
    ref = sessions.Reference(board, oracle, cap=4096, depth=4, budget=budget)
    peer.write(f'config {epoch:x} {budget:x} 1000 4\n')
    inputs: set[str] = set()
    maximum_error = 0.0
    requests = 0
    while True:
        wanted = ref.next()
        line = peer.line()
        if wanted is None:
            result = sessions.numbers(line, 'result', 6)
            rows = [sessions.numbers(peer.line(), 'node', 30) for _ in ref.nodes]
            best = sessions.numbers(peer.line(), 'best', 1)[0]
            ref.check_snapshot(rows, result, best, epoch)
            peer.expect('ready')
            if best != sessions.SENTINEL:
                decode_key(root, best)
            return {'requests': requests, 'completed': ref.completed, 'nodes': len(ref.nodes),
                    'best': best, 'unique_inputs': len(inputs), 'max_logit_absolute_error': maximum_error}
        header = sessions.numbers(line, 'eval', 4)
        if header[:3] != [epoch, ref.seq, wanted] or not 1 <= header[3] <= 256:
            raise AssertionError('bad neural request identity/count')
        supplied = sessions.position(sessions.numbers(peer.line(), 'board', 19))
        path = sessions.parse_path(peer.line())
        expected_path = []
        ancestor = wanted
        while ancestor:
            expected_path.append(ref.nodes[ancestor].key)
            ancestor = ref.nodes[ancestor].parent
        if path != list(reversed(expected_path)):
            raise AssertionError('neural request has wrong ancestor path')
        actions = [sessions.numbers(peer.line(), 'action', 1)[0] for _ in range(header[3])]
        peer.expect('end_eval')
        x, full, history_board = enc.encode(path, supplied, actions)
        check_encoding(x, history_board, evaluator.encoding)
        inputs.add(hashlib.sha256(x.tobytes()).hexdigest())
        policy_logits, wdl_logits = evaluator.evaluate(x)
        with torch.no_grad():
            expected = eager(torch.from_numpy(x))
        for observed, key in ((policy_logits, 'policy'), (wdl_logits, 'wdl')):
            expected_np = expected[key].detach().numpy()
            np.testing.assert_allclose(observed, expected_np, atol=2e-6, rtol=2e-5)
            maximum_error = max(maximum_error, float(np.abs(observed - expected_np).max()))
        wdl, priors = probabilities(policy_logits, wdl_logits, full)
        # Independently use production dense expansion, then legal gather, then Torch softmax.
        dense = _policy_output_full({'policy': torch.from_numpy(policy_logits)})
        np.testing.assert_allclose(priors, torch.softmax(dense[0, full], dim=-1).numpy(), atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(wdl, torch.softmax(torch.from_numpy(wdl_logits[0]), dim=-1).numpy(), atol=2e-7, rtol=2e-6)
        ref.accept(wanted, actions, wdl, priors)
        fields = [epoch, header[1], wanted, 0, *(sessions.bits(v) for v in wdl), len(priors), *(sessions.bits(v) for v in priors)]
        peer.write('reply ' + ' '.join(f'{x:x}' for x in fields) + '\n')
        requests += 1


def verify(binaries: dict[str, Path], evaluator: NativeEvaluator, eager: torch.nn.Module) -> list[dict[str, object]]:
    oracle = sessions.Oracle(binaries['reference'], with_python_chess=True)
    roots = [chess.Board(), chess.Board(rules.CANONICAL[1][1]),
             chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1'),
             chess.Board('4k3/8/8/8/8/8/p7/4K3 b - - 0 1')]
    history = chess.Board()
    for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8') * 2:
        history.push_uci(uci)
    roots.append(history)
    results = []
    for mode, binary in binaries.items():
        if mode == 'reference':
            continue
        for i, root in enumerate(roots):
            peer = sessions.Peer(binary, board_position(root))
            try:
                first = neural_session(peer, oracle, root, evaluator, eager, epoch=1)
                second = neural_session(peer, oracle, root, evaluator, eager, epoch=2)
                if first != second:
                    raise AssertionError('neural reset was not deterministic')
                if first['requests'] < 2 or first['unique_inputs'] < 2:
                    raise AssertionError('neural path failed to exercise nonconstant inputs')
                peer.finish()
                results.append({'mode': mode, 'fixture': i, 'epochs': 2, **first})
            finally:
                peer.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, default=ROOT / 'build/bend_u64_toolchain/source')
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--cxx', default=shutil.which('clang++'))
    parser.add_argument('--modes', nargs='+', choices=list(sessions.MODES), default=list(sessions.MODES))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--smoke-package', type=Path, help='Reuse a manifest-verified seeded smoke package from package.py')
    args = parser.parse_args()
    if not args.bun or not args.cc or not args.cxx:
        parser.error('Bun, Clang and Clang++ are required')
    torch.set_num_threads(2)
    report: dict[str, object] = {'scope': 'history/logits/native CPU inference composition; NOT trained weights, CUDA or speed evidence'}
    with tempfile.TemporaryDirectory(prefix='bend-neural-') as temp:
        work = Path(temp)
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(work / 'inductor')
        checks = boundary_checks()
        package = args.smoke_package or work / 'smoke.pt2'
        if args.smoke_package:
            manifest, encoding = package_manifest(package)
            if encoding != DEFAULT_ENCODING or manifest.get('weights') != 'seeded-untrained' or manifest.get('seed') != SEED:
                raise ValueError('qualification requires the default seeded smoke model and encoding')
            eager = smoke_model(encoding)
        else:
            eager = export_smoke(package)
        binaries = sessions.build(args.compiler_root, work / 'bend', args.bun, args.cc, args.modes)
        worker = build_worker(work / 'native', args.cxx)
        evaluator = NativeEvaluator(worker, package)
        try:
            # The eager reference here is only meaningful for this exact seeded model identity.
            if evaluator.manifest.get('weights') != 'seeded-untrained' or evaluator.manifest.get('seed') != SEED:
                raise ValueError('this qualification command requires the declared smoke weights')
            negative_cases = worker_failures(worker, package, evaluator.encoding.channels)
            report.update({'encoding_checks': checks, 'native_invalid_inputs_rejected': negative_cases, 'package': evaluator.manifest,
                           'compiler_revision': sessions.check_compiler(args.compiler_root)['revision'],
                           'sessions': verify(binaries, evaluator, eager),
                           'native_evaluator_calls': evaluator.sequence - 1, 'native_libpython_dependency': False})
            evaluator.finish()
        finally:
            evaluator.close()
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
