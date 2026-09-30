"""External oracle for composed Bend input/policy requests; never engine code.

The typed preparation must use the SAME validated Game for input history and
legal policy slots. No checkpoint, neural forward or probability conversion.
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
from .verify import Client, FENS, assert_state
from .verify_classical import VERSIONS, read_input
from .verify_encoding import LAYOUTS
from .verify_policy import entry, parse_entry, promotion_fixtures
from .verify_rules import position


def read_request(client: Client, layout: str, version: str, moves: list[str]) -> tuple[np.ndarray, list[tuple[str, int, int, int]]]:
    command = f'encode_request {layout} {version}'
    if moves:
        command += ' moves ' + ' '.join(moves)
    client.send(command + '\n')
    rows = client.until('info string evaluation_request_end', timeout=30)
    header = f'info string evaluation_request {layout} {version} lc0_1858=1858 legal='
    assert rows[0].startswith(header), rows[:1]
    count = int(rows[0][len(header):])
    assert 0 <= count <= 256
    channels = VERSIONS[version]
    assert len(rows) == count + channels + 5, (count, len(rows))
    entries = [parse_entry(row) for row in rows[1:1 + count]]
    assert rows[count + 1] == 'info string policy_end'
    assert rows[count + 2] == f'info string model_input {layout} {version} {channels} 8 8 repfix=1'
    words = []
    for plane, row in enumerate(rows[count + 3:-2]):
        fields = row.split()
        assert fields[:4] == ['info', 'string', 'input_plane', str(plane)]
        assert len(fields) == 68
        values = [int(x) for x in fields[4:]]
        assert all(0 <= x <= 0xffffffff for x in values)
        words.extend(values)
    assert rows[-2] == 'info string model_input_end'
    tensor = np.array(words, dtype=np.uint32).view(np.float32).reshape(channels, 8, 8)
    assert np.isfinite(tensor).all()
    return tensor, entries


def verify(command: list[str], *, require_c: bool = False) -> dict[str, object]:
    cboard_type = None
    if require_c:
        from chess_anti_engine.encoding._lc0_ext import CBoard
        rep_fix.apply(True)
        cboard_type = CBoard
    client = Client(command)
    comparisons = c_comparisons = legal_moves = empty = invalid = direct_comparisons = 0
    max_storm_error = 0.0
    digest = hashlib.sha256()

    def compare(root: chess.Board, moves: list[str], *, direct: bool = False) -> tuple[np.ndarray, list[tuple[str, int, int, int]]]:
        nonlocal comparisons, c_comparisons, legal_moves, empty, max_storm_error, direct_comparisons
        assert client.sync(position(root)) == ['readyok']
        leaf = root.copy(stack=True)
        for move in moves:
            leaf.push_uci(move)
        expected_entries = sorted(entry(leaf, move) for move in leaf.legal_moves)
        last = None
        for layout in LAYOUTS:
            for version, channels in VERSIONS.items():
                tensor, entries = read_request(client, layout, version, moves)
                assert sorted(entries) == expected_entries
                assert len({x[1] for x in entries}) == len(entries)
                assert len({x[2] for x in entries}) == len(entries)
                assert len({x[3] for x in entries}) == len(entries)
                assert all(0 <= x[2] < 4672 and 0 <= x[3] < 1858 for x in entries)
                reference = np.concatenate((encode_lc0_full(leaf, input_history_encoding=layout),
                                            extra_feature_planes_fast(leaf, version=version)), axis=0)
                exact = 173 if version == 'v2_threats' else channels
                np.testing.assert_array_equal(tensor[:exact].view(np.uint32), reference[:exact].view(np.uint32))
                if version == 'v2_threats':
                    # Same predeclared C-F32/Python-F64 exception as the full-input gate.
                    np.testing.assert_allclose(tensor[173:], reference[173:], atol=1.2e-7, rtol=0)
                    max_storm_error = max(max_storm_error, float(np.abs(tensor[173:] - reference[173:]).max()))
                if cboard_type is not None:
                    cb = cboard_type.from_board(leaf)
                    oracle = cb.encode_full(LAYOUTS.index(layout) + 1, channels - 112)
                    np.testing.assert_array_equal(tensor.view(np.uint32), oracle.view(np.uint32))
                    assert sorted(map(int, cb.legal_move_indices())) == sorted(x[2] for x in entries)
                    c_comparisons += 1
                if direct:
                    separate = read_input(client, layout, version, moves)
                    np.testing.assert_array_equal(tensor.view(np.uint32), separate.view(np.uint32))
                    direct_comparisons += 1
                comparisons += 1
                legal_moves += len(entries)
                empty += not entries
                digest.update(tensor.astype('<f4', copy=False).tobytes())
                digest.update(json.dumps(sorted(entries), separators=(',', ':')).encode())
                last = tensor, sorted(entries)
        assert_state(client, root)
        assert last is not None
        return last

    try:
        for fen in FENS:
            compare(chess.Board(fen), [], direct=True)
        for root in promotion_fixtures():
            compare(root, [])
        root = chess.Board()
        for move in ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 2:
            root.push_uci(move)
        a, pa = compare(root, ['g1f3', 'g8f6', 'b1c3', 'b8c6'], direct=True)
        b, pb = compare(root, ['b1c3', 'b8c6', 'g1f3', 'g8f6'], direct=True)
        assert pa == pb
        assert not np.array_equal(a[:104], b[:104])
        compare(root, ['e2e4', 'e7e5', 'g1f3', 'b8c6'], direct=True)
        compare(root, ['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 8, direct=True)
        for text in ['', 'lc0_root', 'legacy v1', 'lc0_root v3', 'lc0_root v1 repfix=0',
                     'lc0_root v1 nonsense', 'lc0_root v1 moves e2e5',
                     'lc0_root v1 moves e2e4 e7e5 a1a8',
                     'lc0_root v1 moves ' + ' '.join(['g1f3', 'g8f6', 'f3g1', 'f6g8'] * 8 + ['g1f3'])]:
            rows = client.sync('encode_request ' + text)
            assert len(rows) == 2, rows
            assert rows[0].startswith('info string encode_request requires '), rows
            assert rows[-1] == 'readyok'
            assert_state(client, root)
            invalid += 1
        client.send('go infinite nodes 4\nencode_request lc0_root v1\nisready\n')
        rows = client.until('readyok')
        assert any('busy;' in row for row in rows)
        assert not any('evaluation_request' in row for row in rows)
        client.send('stop\n')
        client.until('bestmove ')
        assert_state(client, root)
        assert client.sync('ucinewgame') == ['readyok']
        compare(chess.Board(), [], direct=True)
    finally:
        client.close()
    assert empty > 0
    return {'status': 'passed', 'scope': 'complete input plus exact legal policies from one Bend Game; no model forward',
            'composed_requests': comparisons, 'c_cross_checks': c_comparisons,
            'legal_entries_checked': legal_moves, 'terminal_empty_requests': empty,
            'separate_input_bitwise_checks': direct_comparisons, 'invalid_requests': invalid,
            'same_board_distinct_history': True, 'max_python_storm_error': max_storm_error,
            'ordered_request_sha256': digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-c', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    if not args.command:
        parser.error('--command requires an executable')
    report = verify(args.command, require_c=args.require_c)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
