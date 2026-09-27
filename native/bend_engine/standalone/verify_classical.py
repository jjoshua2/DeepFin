"""External Python/C oracles for Bend-authored complete 146/175-plane inputs.

Nothing in this verifier is linked into or invoked by the engine. Native model
execution and performance are deliberately outside this diagnostic comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding.features import extra_feature_planes_fast
from chess_anti_engine.encoding.lc0 import encode_lc0_full
from .verify import Client, assert_state
from .verify_encoding import LAYOUTS, c_representable
from .verify_rules import fixtures, position

VERSIONS = {'v1': 146, 'v2_threats': 175}


def read_input(client: Client, layout: str, version: str, moves: list[str] | None = None) -> np.ndarray:
    text = f'encode_input {layout} {version}'
    if moves:
        text += ' moves ' + ' '.join(moves)
    client.send(text + '\n')
    lines = client.until('info string model_input_end', timeout=30)
    channels = VERSIONS[version]
    assert lines[0] == f'info string model_input {layout} {version} {channels} 8 8 repfix=1'
    assert len(lines) == channels + 2
    values = []
    for i, line in enumerate(lines[1:-1]):
        fields = line.split()
        assert fields[:4] == ['info', 'string', 'input_plane', str(i)]
        assert len(fields) == 68
        row = [int(v) for v in fields[4:]]
        assert all(0 <= v <= 0xffffffff for v in row)
        values.extend(row)
    out = np.array(values, dtype=np.uint32).view(np.float32).reshape(channels, 8, 8)
    assert np.isfinite(out).all()
    return out


def positions() -> list[tuple[str, chess.Board]]:
    out = fixtures()
    for piece in 'PNBRQ':
        out.append(('orthogonal-pin-' + piece, chess.Board(f'k3r3/8/8/8/8/8/4{piece}3/4K3 w - - 0 1')))
        out.append(('diagonal-pin-' + piece, chess.Board(f'7b/k7/8/8/8/8/1{piece}6/K7 w - - 0 1')))
    for name, fen in [
        ('discovery', '4k3/8/8/8/4N3/8/8/4R2K w - - 0 1'),
        ('single-check', '4k3/8/8/8/8/8/N7/4R2K b - - 0 1'),
        ('double-check', '4k3/8/8/1B6/8/8/N7/4R2K b - - 0 1'),
        ('revealed-check', '4k3/8/8/8/4R3/8/N7/4R2K b - - 0 1'),
        ('queen-mobility', '7k/8/8/8/3Q4/8/8/1K6 b - - 0 1'),
        ('pawn-structure', '7k/pp1p4/2p1p3/2P1P3/1P6/P7/3PP3/7K w - - 0 1'),
    ]:
        b = chess.Board(fen)
        assert b.is_valid(), name
        out += [(name, b), (name + '-mirror', b.mirror())]
    # Eight attackers must saturate at seven, not wrap the three-bit count.
    # Four enemy attackers make that saturation observable in the margin plane.
    crowded = chess.Board('3r3k/8/2N1N3/1Np1pN2/8/1N3N2/2N1N3/K5b1 w - - 0 1')
    assert len(crowded.attackers(chess.WHITE, chess.D4)) == 8
    assert len(crowded.attackers(chess.BLACK, chess.D4)) == 4
    out += [('saturated-count', crowded), ('saturated-count-mirror', crowded.mirror())]
    # Every storm distance is observed, with all meaningful file offsets.
    for rank in range(1, 7):
        for file in range(3):
            b = chess.Board(None)
            b.set_piece_at(chess.B1, chess.Piece(chess.KING, chess.WHITE))
            b.set_piece_at(chess.H8, chess.Piece(chess.KING, chess.BLACK))
            b.set_piece_at(chess.square(file, rank), chess.Piece(chess.PAWN, chess.BLACK))
            assert b.is_valid()
            out += [(f'storm-{file}-{rank}', b), (f'storm-{file}-{rank}-mirror', b.mirror())]
    return out


def verify(command: list[str], *, require_c: bool = False) -> dict[str, object]:
    cboard = None
    if require_c:
        from chess_anti_engine.encoding._lc0_ext import CBoard
        rep_fix.apply(True)
        cboard = CBoard
    client = Client(command)
    digest = hashlib.sha256()
    comparisons = c_comparisons = skipped = values = rejected = 0
    max_storm_error = 0.0
    seen = np.zeros(63, dtype=bool)

    def compare(board: chess.Board, layout: str, version: str, observed: np.ndarray) -> None:
        nonlocal comparisons, c_comparisons, skipped, values, max_storm_error
        expected = np.concatenate((encode_lc0_full(board, input_history_encoding=layout),
                                   extra_feature_planes_fast(board, version=version)), axis=0)
        exact = 173 if version == 'v2_threats' else 146
        np.testing.assert_array_equal(observed[:exact].view(np.uint32), expected[:exact].view(np.uint32))
        if version == 'v2_threats':
            # Predeclared C F32-vs-Python F64 intermediate discrepancy, ONLY storm.
            np.testing.assert_allclose(observed[173:], expected[173:], atol=1.2e-7, rtol=0)
            max_storm_error = max(max_storm_error, float(np.abs(observed[173:] - expected[173:]).max()))
            seen[:] |= np.any(observed[112:] != 0, axis=(1, 2))
        digest.update(observed.astype('<f4', copy=False).tobytes())
        comparisons += 1
        values += observed.size
        if cboard is not None:
            if c_representable(board):
                ref = cboard.from_board(board).encode_full(LAYOUTS.index(layout) + 1, VERSIONS[version] - 112)
                np.testing.assert_array_equal(observed.view(np.uint32), ref.view(np.uint32))
                c_comparisons += 1
            else:
                skipped += 1

    try:
        boards = positions()
        for label, board in boards:
            assert board.is_valid(), label
            assert client.sync(position(board)) == ['readyok'], label
            for layout in LAYOUTS:
                tensors = {}
                for version in VERSIONS:
                    got = read_input(client, layout, version)
                    try:
                        compare(board, layout, version, got)
                    except AssertionError as error:
                        raise AssertionError(f'{label} / {layout} / {version}: {error}') from error
                    tensors[version] = got
                np.testing.assert_array_equal(tensors['v1'].view(np.uint32), tensors['v2_threats'][:146].view(np.uint32))
            assert_state(client, board)
        root = chess.Board()
        for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2:
            root.push_uci(move)
        assert client.sync(position(root)) == ['readyok']
        for moves in [['e2e4', 'e7e5', 'g1f3', 'b8c6'], ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 8]:
            child = root.copy(stack=True)
            for move in moves:
                child.push_uci(move)
            for layout in LAYOUTS:
                for version in VERSIONS:
                    compare(child, layout, version, read_input(client, layout, version, moves))
            assert_state(client, root)
        for args in ['', 'lc0_root', 'legacy v1', 'lc0_root v3_checks',
                     'lc0_root v2_threats repfix=0', 'lc0_root v1 garbage',
                     'lc0_root v1 moves e2e5', 'lc0_root v1 moves e2e4 e7e5 a1a8',
                     'lc0_root v1 moves ' + ' '.join(['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 8 + ['g1f3'])]:
            lines = client.sync('encode_input ' + args)
            assert len(lines) == 2
            assert lines[0].startswith('info string encode_input requires ')
            assert lines[1] == 'readyok'
            assert_state(client, root)
            rejected += 1
        client.send('go infinite nodes 4\nencode_input lc0_root v1\nisready\n')
        lines = client.until('readyok')
        assert any('busy;' in row for row in lines)
        assert not any('model_input ' in row for row in lines)
        client.send('stop\n')
        client.until('bestmove ')
        assert_state(client, root)
        assert client.sync('ucinewgame') == ['readyok']
        for version in VERSIONS:
            compare(chess.Board(), LAYOUTS[0], version, read_input(client, LAYOUTS[0], version))
        assert_state(client, chess.Board())
    finally:
        client.close()
    assert seen.all(), f'feature planes never activated: {np.flatnonzero(~seen)}'
    return {'status': 'passed', 'scope': 'Bend complete 146/175 inputs only, no model/policy execution',
            'position_fixtures': len(boards), 'python_tensors': comparisons, 'c_tensors': c_comparisons,
            'python_only_clock_cases': skipped, 'python_f32_values': values, 'rejected_requests': rejected,
            'all_63_features_activated': bool(seen.all()), 'c_equality': 'exact F32 bits',
            'python_equality': 'exact F32 bits except storm absolute tolerance 1.2e-7',
            'max_storm_abs_error': max_storm_error, 'ordered_tensor_sha256': digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-c', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    if not args.command:
        parser.error('--command requires an executable')
    result = verify(args.command, require_c=args.require_c)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
