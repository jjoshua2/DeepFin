"""Opt-in play/search lifecycle gate, without a trained model or subtree reuse.

Runs the existing session suite and advances real native roots in the same
process. The evaluator is the existing deterministic diagnostic evaluator. The
optional encoder check preserves full game history across confirmed advances;
it performs no neural inference and requires the existing C encoding extension.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import TYPE_CHECKING

import chess

from . import run_probe as sessions
from .root_protocol import advance_root, board_position, packed_move
from ..legal_probe import run_probe as rules

if TYPE_CHECKING:
    from collections.abc import Callable

ROOT = Path(__file__).resolve().parents[3]


def verify_roots(binaries: dict[str, Path], *, check_encoding: bool = False) -> dict[str, object]:
    oracle = sessions.Oracle(binaries['reference'], with_python_chess=True)
    encoding_checks = 0
    encode: Callable[[chess.Board], None] | None = None
    observer: Callable[[chess.Board], sessions.RequestObserver] | None = None
    if check_encoding:
        import numpy as np
        from ..neural_probe.adapter import Encoding, HistoryEncoder, legal_keys
        from ..neural_probe.run_probe import check_encoding as compare_encoding
        from chess_anti_engine.encoding import rep_fix
        rep_fix.apply(True)  # standalone process; never bypass the live-board flip guard

        def encode_history(root: chess.Board) -> None:
            nonlocal encoding_checks
            if not any(root.legal_moves):
                return
            for history in ('lc0_root', 'lc0_root_legacy_meta'):
                for features in ('v1', 'v2_threats'):
                    enc = Encoding(history, features, True)
                    x, _, _ = HistoryEncoder(root, enc).encode([], board_position(root), legal_keys(root))
                    compare_encoding(x, root, enc)
                    if root.move_stack:
                        fresh = chess.Board(root.fen(en_passant='fen'))
                        y, _, _ = HistoryEncoder(fresh, enc).encode([], board_position(fresh), legal_keys(fresh))
                        if np.array_equal(x[:, :112], y[:, :112]):
                            raise AssertionError('played-root history was lost during advancement')
                    encoding_checks += 1

        def history_observer(root: chess.Board) -> sessions.RequestObserver:
            enc = Encoding('lc0_root_legacy_meta', 'v2_threats', True)
            encoder = HistoryEncoder(root, enc)

            def check_request(supplied: rules.Position, path: list[int], actions: list[int]) -> None:
                nonlocal encoding_checks
                x, _, leaf = encoder.encode(path, supplied, actions)
                compare_encoding(x, leaf, enc)
                encoding_checks += 1
            return check_request

        encode, observer = encode_history, history_observer

    results = []
    for mode, binary in binaries.items():
        if mode == 'reference':
            continue
        searches = advances = rejections = 0

        def search(peer: sessions.Peer, root: chess.Board, epoch: int, *, cancel: bool = False) -> dict[str, int | str]:
            nonlocal searches
            result = sessions.session(peer, oracle, board_position(root), epoch=epoch,
                                      budget=4, depth=2, fault='cancel' if cancel else '',
                                      on_request=observer(root) if observer else None)
            searches += 1
            if cancel and result['completed'] != 2:
                raise AssertionError('cancelled search accounting changed before root advance')
            return result

        def play(peer: sessions.Peer, root: chess.Board, last: int, key: int, *, status: int = 0,
                 expected: int | None = None, new: int | None = None) -> tuple[chess.Board, int]:
            nonlocal advances, rejections
            next_epoch = last + 1 if new is None else new
            want = last if expected is None else expected
            before = root.fen(en_passant='fen'), list(root.move_stack)
            candidate, reply = advance_root(peer, root, expected_epoch=want, new_epoch=next_epoch, key=key)
            if reply.status != status or reply.current_epoch != (next_epoch if status == 0 else last):
                raise AssertionError('advance status/epoch changed unexpectedly')
            if before != (root.fen(en_passant='fen'), list(root.move_stack)):
                raise AssertionError('host source history mutated before commit')
            if status == 0:
                if oracle.moves(board_position(root)).get(key) != board_position(candidate):
                    raise AssertionError('Bend advanced root differs from native CBoard')
                if len(candidate.move_stack) != len(root.move_stack) + 1:
                    raise AssertionError('accepted advance did not append exactly one history move')
                advances += 1
                if encode:
                    encode(candidate)
            else:
                if candidate is not root:
                    raise AssertionError('rejected advance replaced host history')
                rejections += 1
            return candidate, reply.current_epoch

        # Independent engine-selected games: advance best moves and search again,
        # comparing every request and final node with the CBoard-backed reference.
        for start in (rules.START, rules.CANONICAL[1][1]):
            root = chess.Board(start)
            peer = sessions.Peer(binary, board_position(root))
            pid, last = peer.proc.pid, 0
            try:
                for _ in range(8):
                    row = search(peer, root, last + 1)
                    last += 1
                    key = int(row['best'])
                    if key == sessions.SENTINEL:
                        if any(root.legal_moves):
                            raise AssertionError('nonterminal root has no best move')
                        break
                    root, last = play(peer, root, last, key)
                    if peer.proc.pid != pid or peer.proc.poll() is not None:
                        raise AssertionError('root advancement restarted or lost the process')
                search(peer, root, last + 1)
                peer.finish()
            finally:
                peer.close()

        # Opponent/scripted moves need not be the search's preferred child.
        # These sequences cover checkmate, repeated history, castling and EP.
        scripts = [
            ('mate', rules.START, 'f2f3 e7e5 g2g4 d8h4'),
            ('history', rules.START, 'g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8 e2e4'),
            ('castle', 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'e1g1 e8c8'),
            ('ep', '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', 'e5d6'),
        ]
        for side in ('white', 'black'):
            for promotion in 'nbrq':
                fen = '4k3/P7/8/8/8/8/8/4K3 w - - 0 1' if side == 'white' else '4k3/8/8/8/8/8/p7/4K3 b - - 0 1'
                scripts.append((f'{side}_{promotion}', fen, ('a7a8' if side == 'white' else 'a2a1') + promotion))
        for name, fen, text in scripts:
            root = chess.Board(fen)
            peer = sessions.Peer(binary, board_position(root))
            last = 0
            try:
                for uci in text.split():
                    # Search before each externally chosen move; paths must restart
                    # at the NEW root, without carrying an old tree's ancestors.
                    search(peer, root, last + 1)
                    last += 1
                    m = chess.Move.from_uci(uci)
                    root, last = play(peer, root, last, packed_move(root, m))
                row = search(peer, root, last + 1)
                last += 1
                if name == 'mate':
                    if not root.is_checkmate() or row['exchanges'] != 0:
                        raise AssertionError('played checkmate requested an evaluator')
                    root, last = play(peer, root, last, 0, status=2)
                peer.finish()
            finally:
                peer.close()

        # Semantic rejections are atomic and do not consume a new epoch.
        for fen, illegal in [
            (rules.START, [0, sessions.SENTINEL, sessions.move_key((12, 28, 0, 1))]),
            ('4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1', [sessions.move_key((38, 45, 0, 1))]),
            ('4kr2/8/8/8/8/8/8/4K2R w K - 0 1', [sessions.move_key((4, 6, 0, 2))]),
            ('4k3/P7/8/8/8/8/8/4K3 w - - 0 1', [sessions.move_key((48, 56, 0, 0))]),
            ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', [sessions.move_key((36, 43, 0, 0))]),
        ]:
            root = chess.Board(fen)
            peer = sessions.Peer(binary, board_position(root))
            last = 0
            try:
                for key in illegal:
                    root, last = play(peer, root, last, key, status=2)
                # The original, unconsumed epoch works after all rejections.
                key = packed_move(root, next(iter(root.legal_moves)))
                root, last = play(peer, root, last, key)
                search(peer, root, last + 1)
                last += 1
                key = packed_move(root, next(iter(root.legal_moves)))
                root, last = play(peer, root, last, key, expected=last - 1, status=1)
                root, last = play(peer, root, last, key, expected=last + 1, status=1)
                root, last = play(peer, root, last, key, new=last, status=1)
                root, last = play(peer, root, last, key)
                # Replaying the accepted transaction cannot apply another move.
                root, last = play(peer, root, last, key, expected=last - 1, new=last, status=1)
                search(peer, root, last + 1)
                peer.finish()
            finally:
                peer.close()

        root = chess.Board()
        peer = sessions.Peer(binary, board_position(root))
        try:
            search(peer, root, 1, cancel=True)
            root, last = play(peer, root, 1, packed_move(root, chess.Move.from_uci('e2e4')))
            search(peer, root, last + 1)
            peer.finish()
        finally:
            peer.close()
        # U32 epoch exhaustion is a rejection, never a wrap back to zero.
        root = chess.Board()
        peer = sessions.Peer(binary, board_position(root))
        try:
            root, last = play(peer, root, 0, packed_move(root, chess.Move.from_uci('e2e4')), new=sessions.SENTINEL)
            root, last = play(peer, root, last, 0, new=1, status=1)
            peer.finish()
        finally:
            peer.close()
        bad_wire = [
            ('advance\n', 'invalid advance command'),
            ('advance 0 0 70c\n', 'invalid advance command'),
            ('advance 0 1\n', 'invalid advance command'),
            ('advance 0 1 70c 1\n', 'invalid advance command'),
            ('advance 0 1 100000000\n', 'invalid transport word'),
            ('advance 0 1 xyz\n', 'invalid transport word'),
            ('advance 0 1 70c', 'unterminated transport record'),
            ('reply 1 2 3\n', 'wrong transport record'),
            # Advancing a root while an evaluator is pending must not execute it.
            ('config 1 1 1000 2\nadvance 0 2 70c\n', 'wrong transport record'),
        ]
        for wire, message in bad_wire:
            run = subprocess.run([str(binary), '--threads', '1'], input=rules.request(board_position(chess.Board()), mode=1) + wire,
                                 text=True, capture_output=True, timeout=10, check=False)
            if run.returncode != 2 or message not in run.stderr or 'advance_result' in run.stdout:
                raise AssertionError(f'invalid command was not rejected: {wire!r}: {run.stderr}')
        results.append({'mode': mode, 'search_epochs': searches, 'accepted_advances': advances,
                        'semantic_rejections': rejections, 'invalid_wire_cases': len(bad_wire)})
    return {'scope': 'same-process play/search, exact legal root updates; fresh tree, no subtree reuse or full draw rules',
            'results': results, 'oracle_positions': len(oracle.cache), 'encoding_checks': encoding_checks}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, default=ROOT / 'build/bend_u64_toolchain/source')
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--modes', nargs='+', choices=list(sessions.MODES), default=list(sessions.MODES))
    parser.add_argument('--check-encoding', action='store_true')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if not args.bun or not args.cc:
        parser.error('Bun and Clang are required')
    with tempfile.TemporaryDirectory(prefix='bend-roots-') as tmp:
        binaries = sessions.build(args.compiler_root, Path(tmp), args.bun, args.cc, args.modes)
        report = {'original_sessions': sessions.verify(binaries, with_python_chess=True),
                  'root_advancement': verify_roots(binaries, check_encoding=args.check_encoding)}
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
