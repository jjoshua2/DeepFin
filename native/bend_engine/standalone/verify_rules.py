"""External rule oracle only; the tested executable computes every rule in Bend.

Run with the same --command prefix as verify.py, including an empty chroot.
No model, compiler, library, table file or Python process belongs inside that root.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import TypedDict

import chess

from .verify import Client, assert_state


def position(board: chess.Board) -> str:
    root = board.root()
    moves = ' '.join(m.uci() for m in board.move_stack)
    return 'position fen ' + root.fen(en_passant='fen') + (' moves ' + moves if moves else '')


def key(board: chess.Board) -> tuple[object, ...]:
    return (board.pawns, board.knights, board.bishops, board.rooks, board.queens,
            board.kings, board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
            board.turn, board.clean_castling_rights(),
            board.ep_square if board.has_legal_en_passant() else None)


def facts(board: chess.Board) -> tuple[str, int, int]:
    outcome = board.outcome(claim_draw=False)
    reason = 'ongoing' if outcome is None else outcome.termination.name.lower()
    count = 1
    if any(board.legal_moves):
        target = key(board)
        old = board.copy(stack=True)
        for _ in range(min(board.halfmove_clock, len(board.move_stack))):
            old.pop()
            count += int(key(old) == target)
        # Cross-check our diagnostic count against the library's own repetition API.
        assert board.is_repetition(count)
        assert not board.is_repetition(count + 1)
    return reason, count, board.halfmove_clock


def cycle(fen: str, moves: str, repeats: int, tail: str = '') -> chess.Board:
    board = chess.Board(fen)
    for uci in (moves.split() * repeats + tail.split()):
        board.push_uci(uci)
    assert board.is_valid()
    return board


def fixtures() -> list[tuple[str, chess.Board]]:
    result = [(f'clock-{clock}', chess.Board(f'4k3/8/8/8/8/8/8/R3K3 w - - {clock} 1'))
              for clock in (0, 99, 100, 149, 150, 151, 999999)]
    for pieces, name in [('4K3', 'bare'), ('3BK3', 'bishop'), ('3NK3', 'knight'),
                         ('2NNK3', 'two-knights'), ('2BBK3', 'opposite-bishops')]:
        result.append((name, chess.Board(f'4k3/8/8/8/8/8/8/{pieces} w - - 0 1')))
    result += [
        ('same-color-bishops', chess.Board('4k3/8/8/8/8/8/1b6/2B1K3 w - - 0 1')),
        ('promoted-bishops', chess.Board('4k3/8/8/8/8/4B3/1b6/2B1K3 w - - 0 1')),
        ('minor-v-minor', chess.Board('4k2n/8/8/8/8/8/8/2B1K3 w - - 0 1')),
        ('mate-clock150', chess.Board('k7/1Q6/2K5/8/8/8/8/8 b - - 150 1')),
        ('stalemate-clock150', chess.Board('k7/2Q5/2K5/8/8/8/8/8 b - - 150 1')),
    ]
    histories = [
        ('knights', chess.STARTING_FEN, 'g1f3 g8f6 f3g1 f6g8'),
        ('lost-castling', '4k3/8/8/8/8/8/8/4K2R w K - 0 1', 'h1h2 e8e7 h2h1 e7e8'),
        ('legal-ep', '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', 'e1f1 e8f8 f1e1 f8e8'),
        ('pinned-ep', '4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1', 'h5h4 e8e7 h4h5 e7e8'),
        ('black-legal-ep', '4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1', 'e8f8 e1f1 f8e8 f1e1'),
    ]
    for name, fen, moves in histories:
        for count in range(6):
            result.append((f'{name}-{count}', cycle(fen, moves, count)))
    result += [
        ('long-history', cycle(chess.STARTING_FEN, 'g1f3 g8f6 f3g1 f6g8', 128)),
        ('pawn-reset', cycle(chess.STARTING_FEN, 'g1f3 g8f6 f3g1 f6g8', 4, 'e2e4')),
        ('capture-reset', cycle('4k3/8/8/8/8/8/p7/R3K3 w - - 149 1', '', 0, 'a1a2')),
        ('ep-reset', cycle('4k3/8/8/3pP3/8/8/8/4K3 w - d6 149 1', '', 0, 'e5d6')),
        ('promotion-reset', cycle('4k3/P7/8/8/8/8/8/4K3 w - - 149 1', '', 0, 'a7a8n')),
    ]
    rng = random.Random(20260920)
    board = chess.Board()
    for i in range(64):
        if board.is_game_over():
            break
        board.push(rng.choice(list(board.legal_moves)))
        result.append((f'seeded-{i}', board.copy(stack=True)))
    return result


def search(c: Client, board: chess.Board, nodes: int = 12, depth: int = 1) -> list[str]:
    assert c.sync(position(board)) == ['readyok']
    c.send(f'go nodes {nodes} depth {depth}\n')
    rows = c.until('bestmove ')
    assert any(x.startswith(f'info nodes {nodes} ') for x in rows), rows
    assert any('stop_code 0' in x for x in rows), rows
    best = chess.Move.from_uci(rows[-1].split()[1])
    assert (best in board.legal_moves) if any(board.legal_moves) else not best
    assert_state(c, board)
    return rows


def draws(rows: list[str]) -> list[list[str]]:
    return [x.split()[3:] for x in rows if x.startswith('info string rule_draw ')]


class RulesReport(TypedDict):
    rule_fact_cases: list[dict[str, object]]
    search_cases: dict[str, list[str]]
    perft_draw_position_unchanged: int
    root_and_history_unchanged_by_queries: bool
    scope: str


def verify(command: list[str]) -> RulesReport:
    c = Client(command)
    observed = []
    searches = {}
    try:
        for name, board in fixtures():
            assert board.is_valid(), (name, board.fen())
            assert c.sync(position(board)) == ['readyok']
            before = c.dump()
            output = c.sync('rules')
            expected = 'info string rules ' + ' '.join(map(str, facts(board)))
            assert output == [expected, 'readyok'], (name, expected, output)
            assert c.dump() == before
            assert_state(c, board)
            observed.append({'case': name, 'facts': facts(board)})
        # Deliberately expose identical boards with and without their real history.
        repeated = cycle(chess.STARTING_FEN, 'g1f3 g8f6 f3g1 f6g8', 4)
        rows = search(c, repeated)
        assert draws(rows) == [['0', 'fivefold_repetition', '5', '16']], rows
        assert any('no searched continuation' in x for x in rows)
        searches['root-fivefold-cached'] = rows
        fresh = chess.Board(repeated.fen(en_passant='fen'))
        rows = search(c, fresh, nodes=2)
        assert not draws(rows)
        searches['historyless-control'] = rows
        for name, fen, reason in [
            ('root-75', '4k3/8/8/8/8/8/8/R3K3 w - - 150 1', 'seventyfive_moves'),
            ('root-material', '4k3/8/8/8/8/8/8/3BK3 w - - 0 1', 'insufficient_material'),
        ]:
            rows = search(c, chess.Board(fen))
            assert len(draws(rows)) == 1
            assert draws(rows)[0][:2] == ['0', reason]
            searches[name] = rows
        # Minimal packed move b1a1 reaches a fifth repetition, not just a root draw.
        near = cycle('4k2r/8/8/8/8/8/8/K7 b - - 0 1',
                     'e8e7 a1b1 e7e8 b1a1', 3, 'e8e7 a1b1 e7e8')
        rows = search(c, near, nodes=2)
        assert len(draws(rows)) == 1, rows
        assert draws(rows)[0][0] != '0', rows
        assert draws(rows)[0][1:] == ['fivefold_repetition', '5', '16'], rows
        searches['leaf-fivefold'] = rows
        rows = search(c, chess.Board(near.fen(en_passant='fen')), nodes=2)
        assert not draws(rows)
        searches['leaf-historyless'] = rows
        rows = search(c, chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 149 1'))
        assert draws(rows), rows
        assert all(x[0] != '0' and x[1] == 'seventyfive_moves' and x[3] == '150' for x in draws(rows))
        assert len({x[0] for x in draws(rows)}) == len(draws(rows)), rows
        searches['leaf-75-cached'] = rows
        rows = search(c, chess.Board('4k3/8/8/8/8/8/1B6/n3K3 w - - 0 1'), nodes=2)
        assert len(draws(rows)) == 1, rows
        assert draws(rows)[0][1] == 'insufficient_material', rows
        searches['leaf-capture-material'] = rows
        for name, board in [('threefold-not-forced', cycle(chess.STARTING_FEN, 'g1f3 g8f6 f3g1 f6g8', 2)),
                             ('50-not-forced', chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 100 1'))]:
            rows = search(c, board, nodes=2)
            assert not draws(rows)
            searches[name] = rows
        mate = chess.Board('k7/1Q6/2K5/8/8/8/8/8 b - - 150 1')
        rows = search(c, mate)
        assert not draws(rows), rows
        assert rows[-1] == 'bestmove 0000'
        searches['mate-not-draw'] = rows
        # Perft counts legal moves, and must not be pruned by draw adjudication.
        c.sync(position(repeated))
        assert 'info string perft 3 0 8902' in c.sync('perft 3')
        before = c.dump()
        assert any('rules takes no arguments' in x for x in c.sync('rules bogus'))
        assert c.dump() == before
        c.sync('ucinewgame')
        assert c.sync('rules') == ['info string rules ongoing 1 0', 'readyok']
        assert_state(c, chess.Board())
        return {'rule_fact_cases': observed, 'search_cases': searches,
                'perft_draw_position_unchanged': 8902,
                'root_and_history_unchanged_by_queries': True,
                'scope': 'Bend rule facts/search integration; external Python oracle only'}
    finally:
        c.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    result = verify(args.command)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print('PASS:', len(result['rule_fact_cases']), 'rule positions;', len(result['search_cases']), 'search cases')


if __name__ == '__main__':
    main()
