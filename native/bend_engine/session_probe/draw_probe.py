"""Opt-in automatic-draw leaf/cache/backup checks against actual native Bend.

Uses a controlled evaluator, not a trained model. History/rules live in the host;
Bend must validate the terminal reply, cache it, and back up zero exactly once per
simulation. Threefold/50-move claims are deliberately NOT forced here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import TypedDict

import chess

from . import run_probe as sessions
from .root_protocol import board_position, packed_move
from .search_draws import automatic_draw, draw_reply, reconstruct_leaf

ROOT = Path(__file__).resolve().parents[3]


def cycle_root(cycles: int, tail: tuple[str, ...] = ()) -> chess.Board:
    board = chess.Board()
    for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8') * cycles + tail:
        board.push_uci(uci)
    return board


class Readout(TypedDict):
    completed: int
    nodes: int
    stop: int
    neural_requests: int
    draw_replies: list[dict[str, object]]
    request_paths: list[list[int]]
    best: int
    root_w: float


def drive(binary: Path, oracle: sessions.Oracle, root: chess.Board, *, budget: int = 12,
          depth: int = 4, prefer: tuple[str, ...] = (), fault: str = '',
          adjudicate: bool = True, model_draw: bool = False) -> tuple[Readout, sessions.Reference]:
    ref = sessions.Reference(board_position(root), oracle, cap=4096, depth=depth, budget=budget)
    peer = sessions.Peer(binary, board_position(root))
    neural = 0
    draws: list[dict[str, object]] = []
    requests: list[list[int]] = []
    try:
        peer.write(f'config 1 {budget:x} 1000 {depth:x}\n')
        while True:
            wanted = ref.next()
            line = peer.line()
            if wanted is None:
                result = sessions.numbers(line, 'result', 6)
                rows = [sessions.numbers(peer.line(), 'node', 30) for _ in ref.nodes]
                best = sessions.numbers(peer.line(), 'best', 1)[0]
                ref.check_snapshot(rows, result, best, 1)
                peer.expect('ready')
                peer.finish()
                return ({'completed': ref.completed, 'nodes': len(ref.nodes), 'stop': ref.stop,
                         'neural_requests': neural, 'draw_replies': draws, 'request_paths': requests,
                         'best': best, 'root_w': ref.nodes[0].w}, ref)
            header = sessions.numbers(line, 'eval', 4)
            if header[:3] != [1, ref.seq, wanted]:
                raise AssertionError('draw probe request identity mismatch')
            supplied = sessions.position(sessions.numbers(peer.line(), 'board', 19))
            path = sessions.parse_path(peer.line())
            ancestors = []
            index = wanted
            while index:
                ancestors.append(ref.nodes[index].key)
                index = ref.nodes[index].parent
            if path != ancestors[::-1] or supplied != ref.nodes[wanted].board:
                raise AssertionError('draw probe path/board mismatch')
            actions = [sessions.numbers(peer.line(), 'action', 1)[0] for _ in range(header[3])]
            peer.expect('end_eval')
            # In the counterfactual depth-one control, we intentionally ignore
            # draw rules only to prove a neural value is not the terminal value.
            board = reconstruct_leaf(root, path, supplied, actions)
            reason = automatic_draw(board) if adjudicate else None
            requests.append(path)
            if reason is not None:
                reply = draw_reply(1, ref.seq, wanted)
                if fault:
                    words = [int(w, 16) for w in reply.split()[1:]]
                    if fault == 'epoch':
                        words[0] += 1
                    elif fault == 'request':
                        words[1] += 1
                    elif fault == 'node':
                        words[2] += 1
                    elif fault == 'win':
                        words[4] = sessions.bits(0.5)
                    elif fault == 'draw':
                        words[5] = sessions.bits(0.5)
                    elif fault == 'loss':
                        words[6] = sessions.bits(1.0)
                    elif fault == 'nan':
                        words[4] = sessions.bits(float('nan'))
                    elif fault == 'infinity':
                        words[5] = sessions.bits(float('inf'))
                    elif fault == 'policy':
                        words[7] = 1
                        words.append(0)
                    else:
                        raise ValueError('unknown fault')
                    ref.stop = 4
                    peer.write('reply ' + ' '.join(f'{w:x}' for w in words) + '\n')
                else:
                    ref.accept_draw(wanted)
                    peer.write(reply)
                    draws.append({'node': wanted, 'path': path, 'reason': reason})
            else:
                neural += 1
                wdl = [0.0, 1.0, 0.0] if model_draw else [0.0, 0.25, 0.75]
                priors = [1.0 / len(actions)] * len(actions)
                if len(path) < len(prefer):
                    move = chess.Move.from_uci(prefer[len(path)])
                    if move not in board.legal_moves:
                        raise AssertionError('scripted search preference is illegal')
                    key = packed_move(board, move)
                    priors = [float(k == key) for k in actions]
                priors = [sessions.f32(p) for p in priors]
                ref.accept(wanted, actions, wdl, priors)
                fields = [1, header[1], wanted, 0, *(sessions.bits(v) for v in wdl),
                          len(priors), *(sessions.bits(v) for v in priors)]
                peer.write('reply ' + ' '.join(f'{w:x}' for w in fields) + '\n')
    finally:
        peer.close()


def verify(binaries: dict[str, Path]) -> dict[str, object]:
    oracle = sessions.Oracle(binaries['reference'], with_python_chess=True)
    results = []
    for mode, binary in binaries.items():
        if mode == 'reference':
            continue
        cases: dict[str, object] = {}
        root75 = chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 150 1')
        for name, root in [('75-move', root75), ('fivefold', cycle_root(4)),
                           ('material', chess.Board('4k3/8/8/8/8/8/8/3BK3 w - - 0 1'))]:
            report, ref = drive(binary, oracle, root)
            assert report['neural_requests'] == 0
            assert len(report['draw_replies']) == 1
            assert ref.completed == 12
            assert len(ref.nodes) == 1
            assert (ref.nodes[0].status, ref.nodes[0].value, ref.nodes[0].w) == (2, 0.0, 0.0)
            cases[name] = report
        repeated = cycle_root(3, ('g1f3', 'g8f6', 'f3g1'))
        report, ref = drive(binary, oracle, repeated, depth=1, prefer=('f6g8',))
        assert report['neural_requests'] == 1
        assert len(report['draw_replies']) == 1
        terminal = next(n for n in ref.nodes[1:] if n.n)
        assert (terminal.status, terminal.value, terminal.n) == (2, 0.0, 11)
        assert ref.nodes[0].w == -0.75
        cases['below-root-fivefold'] = report
        # Same exact leaf board but no pre-root repetition: must NOT be terminal.
        fresh = chess.Board(repeated.fen(en_passant='fen'))
        control, c = drive(binary, oracle, fresh, budget=3, depth=1, prefer=('f6g8',))
        assert control['draw_replies'] == []
        assert next(n for n in c.nodes[1:] if n.n).status == 3
        cases['historyless-control'] = control
        # All root moves reach the 75-move boundary without a capture/pawn move.
        near = chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 149 1')
        report, ref = drive(binary, oracle, near, depth=1, prefer=('a1b1',))
        assert report['neural_requests'] == 1
        assert len(report['draw_replies']) == 1
        assert ref.nodes[0].w == -0.75
        cases['below-root-75'] = report
        control, c = drive(binary, oracle, near, budget=3, depth=1,
                           prefer=('a1b1',), adjudicate=False)
        assert c.nodes[0].w == 0.75
        assert control['draw_replies'] == []
        cases['neural-value-counterfactual'] = control
        report, ref = drive(binary, oracle, near, depth=1, prefer=('a1b1',), fault='policy')
        assert ref.completed == 1
        assert ref.stop == 4
        assert ref.nodes[0].w == -0.75
        assert all(n.status == 0 and n.n == 0 for n in ref.nodes[1:])
        cases['invalid-draw-after-completed-simulation'] = report
        capture = chess.Board('4k3/8/8/8/8/8/p7/R3K3 w - - 149 1')
        report, _ = drive(binary, oracle, capture, budget=2, prefer=('a1a2',))
        assert report['draw_replies'] == []
        assert report['neural_requests'] == 2
        cases['capture-resets-clock'] = report
        material = chess.Board('4k2n/8/8/8/8/2B5/8/4K3 w - - 0 1')
        report, ref = drive(binary, oracle, material, depth=1, prefer=('c3h8',))
        assert report['neural_requests'] == 1
        assert len(report['draw_replies']) == 1
        assert report['draw_replies'][0]['reason'] == 'insufficient_material'
        assert next(n for n in ref.nodes[1:] if n.n).value == 0.0
        cases['below-root-insufficient-material'] = report
        for name, root in [('threefold-not-forced', cycle_root(2)),
                           ('50-move-not-forced', chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 100 1'))]:
            report, _ = drive(binary, oracle, root, budget=2, depth=1)
            assert report['draw_replies'] == []
            assert report['neural_requests'] == 2
            cases[name] = report
        reset = chess.Board()
        reset.halfmove_clock = 149
        report, _ = drive(binary, oracle, reset, budget=2, prefer=('e2e4',))
        assert report['draw_replies'] == []
        assert report['neural_requests'] == 2
        cases['pawn-resets-clock'] = report
        mate = chess.Board('k7/2Q5/2K5/8/8/8/8/8 w - - 149 1')
        report, ref = drive(binary, oracle, mate, budget=3, prefer=('c7b7',))
        assert report['neural_requests'] == 1
        assert report['draw_replies'] == []
        assert next(n for n in ref.nodes[1:] if n.n).value == -1.0
        cases['mate-precedes-75'] = report
        report, ref = drive(binary, oracle, chess.Board(), budget=2, model_draw=True)
        assert ref.nodes[0].status == 1
        assert report['neural_requests'] == 2
        cases['neural-draw-is-not-terminal'] = report
        for fault in ('epoch', 'request', 'node', 'win', 'draw', 'loss', 'nan', 'infinity', 'policy'):
            report, ref = drive(binary, oracle, root75, fault=fault)
            assert ref.completed == 0
            assert len(ref.nodes) == 1
            assert ref.nodes[0].status == 0
            assert ref.stop == 4
            cases['invalid-' + fault] = report
        results.append({'mode': mode, 'cases': cases})
    return {'scope': 'automatic draw reply/cache/backup, host-owned exact history; no NN/GPU/speed claim',
            'results': results, 'oracle_positions': len(oracle.cache)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compiler-root', type=Path, default=ROOT / 'build/bend_u64_toolchain/source')
    parser.add_argument('--bun', default=shutil.which('bun'))
    parser.add_argument('--cc', default=shutil.which('clang'))
    parser.add_argument('--modes', nargs='+', choices=list(sessions.MODES), default=list(sessions.MODES))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if not args.bun or not args.cc:
        parser.error('Bun and Clang are required')
    with tempfile.TemporaryDirectory(prefix='bend-search-draws-') as temp:
        binaries = sessions.build(args.compiler_root, Path(temp), args.bun, args.cc, args.modes)
        report = {'draws': verify(binaries), 'original_sessions': sessions.verify(binaries, with_python_chess=True)}
    text = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
