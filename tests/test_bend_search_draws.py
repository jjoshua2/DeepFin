"""Cheap history, terminal-reply and local-broker contracts. No native execution."""
from __future__ import annotations

from pathlib import Path
from typing import cast

import chess
import numpy as np
import pytest

from native.bend_engine.session_probe import run_probe as sessions
from native.bend_engine.session_probe.root_protocol import board_position, packed_move
from native.bend_engine.session_probe.search_draws import automatic_draw, draw_reply, reconstruct_leaf
from native.bend_engine.neural_probe.batching import Batcher, Key


def cycle(board: chess.Board, moves: tuple[str, ...], times: int) -> chess.Board:
    root = board.copy(stack=True)
    for uci in moves * times:
        root.push_uci(uci)
    return root


KNIGHTS = ('g1f3', 'g8f6', 'f3g1', 'f6g8')


@pytest.mark.parametrize(('clock', 'expected'), [(99, None), (100, None), (149, None), (150, 'seventyfive_moves')])
def test_automatic_clock_is_not_a_claim(clock: int, expected: str | None) -> None:
    board = chess.Board(f'4k3/8/8/8/8/8/8/R3K3 w - - {clock} 1')
    assert automatic_draw(board) == expected


@pytest.mark.parametrize(('repeats', 'expected'), [(1, None), (2, None), (3, None), (4, 'fivefold_repetition')])
def test_only_fivefold_is_forced(repeats: int, expected: str | None) -> None:
    root = cycle(chess.Board(), KNIGHTS, repeats)
    stack, fen = root.move_stack.copy(), root.fen()
    assert automatic_draw(root) == expected
    assert root.move_stack == stack
    assert root.fen() == fen
    assert automatic_draw(chess.Board(fen)) is None


@pytest.mark.parametrize(('fen', 'material'), [
    ('4k3/8/8/8/8/8/8/4K3 w - - 0 1', True),
    ('4k3/8/8/8/8/8/8/3BK3 w - - 0 1', True),
    ('4k3/8/8/8/8/8/8/3NK3 w - - 0 1', True),
    ('4k3/8/8/8/8/8/8/2NNK3 w - - 0 1', False),
    ('4k3/8/8/8/8/8/8/3RK3 w - - 0 1', False),
])
def test_conservative_insufficient_material(fen: str, material: bool) -> None:
    assert (automatic_draw(chess.Board(fen)) == 'insufficient_material') == material


def test_mate_takes_precedence_over_clock() -> None:
    board = chess.Board('k7/2Q5/2K5/8/8/8/8/8 w - - 149 1')
    board.push_uci('c7b7')
    assert board.halfmove_clock == 150
    assert board.is_checkmate()
    assert automatic_draw(board) is None


def test_lost_castling_rights_do_not_count_as_identical_position() -> None:
    board = chess.Board('4k3/8/8/8/8/8/8/4K2R w K - 0 1')
    moves = ('h1h2', 'e8e7', 'h2h1', 'e7e8')
    assert automatic_draw(cycle(board, moves, 4)) is None
    assert automatic_draw(cycle(board, moves, 5)) == 'fivefold_repetition'


def test_legal_ep_right_distinguishes_repetitions() -> None:
    board = chess.Board('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1')
    moves = ('e1f1', 'e8f8', 'f1e1', 'f8e8')
    assert board.has_legal_en_passant()
    assert automatic_draw(cycle(board, moves, 4)) is None
    assert automatic_draw(cycle(board, moves, 5)) == 'fivefold_repetition'


def test_illegal_pinned_ep_does_not_distinguish_repetitions() -> None:
    board = chess.Board('4k3/8/8/r4pPK/8/8/8/8 w - f6 0 1')
    assert not board.has_legal_en_passant()
    assert automatic_draw(cycle(board, ('h5h4', 'e8e7', 'h4h5', 'e7e8'), 4)) == 'fivefold_repetition'


@pytest.mark.parametrize('uci', ['e2e4', 'd2d4'])
def test_pawn_move_resets_draw_clock(uci: str) -> None:
    root = chess.Board()
    root.halfmove_clock = 149
    move = chess.Move.from_uci(uci)
    child = root.copy(stack=True)
    child.push(move)
    got = reconstruct_leaf(root, [packed_move(root, move)], board_position(child),
                           [packed_move(child, m) for m in child.legal_moves])
    assert got.halfmove_clock == 0
    assert automatic_draw(got) is None
    assert root.move_stack == []


@pytest.mark.parametrize('fault', ['key', 'flag', 'board', 'missing', 'duplicate', 'length'])
def test_bad_request_is_rejected_before_a_rule_reply(fault: str) -> None:
    root = chess.Board()
    move = chess.Move.from_uci('e2e4')
    child = root.copy(stack=True)
    child.push(move)
    path = [packed_move(root, move)]
    position = board_position(child)
    actions = [packed_move(child, m) for m in child.legal_moves]
    if fault == 'key':
        path = [True]
    elif fault == 'flag':
        path[0] |= 1 << 15
    elif fault == 'board':
        position = board_position(root)
    elif fault == 'missing':
        actions.pop()
    elif fault == 'duplicate':
        actions.append(actions[0])
    else:
        path *= 33
    with pytest.raises(ValueError, match=r'history|actions|path'):
        reconstruct_leaf(root, path, position, actions)
    assert root.move_stack == []


def test_search_cannot_continue_past_an_automatic_draw() -> None:
    root = cycle(chess.Board(), KNIGHTS, 4)
    move = next(iter(root.legal_moves))
    with pytest.raises(ValueError, match='continues after'):
        reconstruct_leaf(root, [packed_move(root, move)], board_position(root), [])


@pytest.mark.parametrize(('epoch', 'sequence', 'node'), [(0, 1, 0), (1, 0, 0), (1, 1, 4096),
                                                      (True, 1, 0), (1, 1, -1), (2**32, 1, 0)])
def test_draw_reply_identity_is_bounded(epoch: int, sequence: int, node: int) -> None:
    with pytest.raises(ValueError, match='identity'):
        draw_reply(epoch, sequence, node)


def test_rule_reply_has_no_neural_policy() -> None:
    assert draw_reply(1, 2, 3) == 'reply 1 2 3 3 0 3f800000 0 0\n'


def test_local_rule_reply_uses_no_inference_capacity_but_consumes_identity() -> None:
    broker = Batcher(4, 175)
    broker.register(0, 1)
    broker.record_local(Key(0, 1, 1, 0))
    assert broker.reserved == 0
    assert not broker.pending
    assert broker.epochs[0] == (1, 1)
    with pytest.raises(ValueError, match='duplicate'):
        broker.record_local(Key(0, 1, 1, 0))
    broker.submit(Key(0, 1, 2, 1), np.zeros((1, 175, 8, 8)), np.array([0]), now=1, deadline=2)
    with pytest.raises(ValueError, match='outstanding'):
        broker.record_local(Key(0, 1, 3, 2))
    assert broker.epochs[0] == (1, 2)


def test_cancelled_old_row_cannot_overwrite_a_new_local_draw() -> None:
    broker = Batcher(4, 175)
    broker.register(0, 1)
    old = Key(0, 1, 1, 0)
    broker.submit(old, np.zeros((1, 175, 8, 8)), np.array([0]), now=1, deadline=3)
    flight = broker.dispatch(2)
    assert flight is not None
    broker.cancel(old)
    broker.register(0, 2)
    broker.record_local(Key(0, 2, 1, 0))
    assert broker.reserved == 1
    assert broker.complete(flight, np.zeros((4, 1858)), np.zeros((4, 3)), now=2) == []
    assert broker.reserved == 0
    assert broker.epochs[0] == (2, 1)


def test_actor_skips_encoder_and_model_queue_for_confirmed_rule_draw(monkeypatch) -> None:
    from chess_anti_engine.encoding import rep_fix
    from native.bend_engine.neural_probe.adapter import Encoding, HistoryEncoder
    from native.bend_engine.neural_probe.batch_probe import Actor
    root = chess.Board('4k3/8/8/8/8/8/8/R3K3 w - - 150 1')
    moves = list(root.legal_moves)
    keys = [packed_move(root, m) for m in moves]
    pos = board_position(root)
    words = [w for x in pos[:8] for w in (x >> 32, x & 0xffffffff)] + list(pos[8:])

    class Peer:
        sent: list[str]
        def __init__(self):
            self.sent = []
            self.lines = iter(['board ' + ' '.join(map(str, words)), 'path 0',
                               *['action ' + str(k) for k in keys], 'end_eval'])
        def write(self, text: str) -> None:
            self.sent.append(text)
        def line(self) -> str:
            return next(self.lines)
        def expect(self, expected: str) -> None:
            assert self.line() == expected

    encoder = HistoryEncoder(root, Encoding('lc0_root', 'v1', rep_fix.current() or False))
    def forbidden(*_args, **_kwargs):
        raise AssertionError('terminal rule draw must not encode a neural row')
    monkeypatch.setattr(encoder, 'encode', forbidden)
    oracle = sessions.Oracle(Path('/never-executed'))
    oracle.cache[pos] = dict.fromkeys(keys, pos)  # no children are expanded in this test
    peer = Peer()
    actor = Actor(cast(sessions.Peer, peer), root, encoder, oracle, 0, 4)
    broker = Batcher(4, 146)
    broker.register(0, 1)
    actor.receive(f'eval 1 1 0 {len(keys)}', broker)
    assert peer.sent[-1] == draw_reply(1, 1, 0)
    assert actor.ref.nodes[0].status == 2
    assert actor.ref.completed == 1
    assert actor.waiting is None
    assert broker.reserved == 0
    assert actor.draw_leaves[0]['reason'] == 'seventyfive_moves'
