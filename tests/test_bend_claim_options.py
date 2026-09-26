"""Cheap optional-claim evidence/controller checks; no native search or NN calls."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import rep_fix
from native.bend_engine.session_probe import run_probe as sessions
from native.bend_engine.session_probe.claims import Claim, claim_option
from native.bend_engine.session_probe.root_protocol import board_position, packed_move
from native.bend_engine.neural_probe.adapter import Encoding, HistoryEncoder, decode_key
from native.bend_engine.neural_probe.batch_probe import Actor, Observation
from native.bend_engine.neural_probe.batching import Batcher, Completion, Key
from native.bend_engine.neural_probe.game import GameActor, GameSpec, adjudicate


def repeated(n: int, tail: tuple[str, ...] = ()) -> chess.Board:
    b = chess.Board()
    for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8') * n + tail:
        b.push_uci(uci)
    return b


def fifty(clock: int) -> chess.Board:
    return chess.Board(f'4k3/8/8/8/8/8/8/R3K3 w - - {clock} 1')


@pytest.mark.parametrize(('clock', 'expected'), [(98, None), (99, 'a1a2'), (100, None), (149, None), (150, None)])
def test_fifty_claim_and_automatic_boundary(clock: int, expected: str | None) -> None:
    b = fifty(clock)
    before = b.fen(), list(b.move_stack)
    option = claim_option(b)
    if clock in (98, 150):
        assert option is None
    else:
        assert option == Claim('fifty_moves', expected)
    assert (b.fen(), b.move_stack) == before


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4])
def test_threefold_requires_history_and_excludes_fivefold(n: int) -> None:
    b = repeated(n)
    before = b.fen(), list(b.move_stack)
    assert claim_option(b) == (Claim('threefold_repetition') if n in (2, 3) else None)
    assert claim_option(chess.Board(b.fen())) is None
    assert (b.fen(), b.move_stack) == before


def test_intended_threefold_is_not_a_played_move() -> None:
    b = repeated(1, ('g1f3', 'g8f6', 'f3g1'))
    assert not b.is_repetition(3)
    before = b.fen(), list(b.move_stack)
    assert claim_option(b) == Claim('threefold_repetition', 'f6g8')
    assert (b.fen(), b.move_stack) == before


@pytest.mark.parametrize('fen', ['k7/1Q6/2K5/8/8/8/8/8 b - - 150 1',
                               'k7/2Q5/2K5/8/8/8/8/8 b - - 100 1',
                               '4k3/8/8/8/8/8/8/4K3 w - - 100 1'])
def test_automatic_endings_precede_optional_claims(fen: str) -> None:
    b = chess.Board(fen)
    assert b.outcome() is not None
    assert claim_option(b) is None


@pytest.mark.parametrize('board', [chess.Board(None), chess.Board(chess960=True)])
def test_invalid_variant_is_rejected(board: chess.Board) -> None:
    with pytest.raises(ValueError, match='orthodox'):
        claim_option(board)


def test_reserved_claim_is_not_a_move_or_a_policy_index() -> None:
    with pytest.raises(ValueError, match='packed'):
        decode_key(fifty(100), sessions.CLAIM_KEY)
    assert sessions.CLAIM_KEY >= 1 << 17


def encoding() -> Encoding:
    return Encoding('lc0_root', 'v1', rep_fix.current() or False)


def peer_for(board: chess.Board):
    peer = create_autospec(sessions.Peer, instance=True)
    peer.proc = SimpleNamespace(pid=123, poll=lambda: None)
    children = {}
    for move in board.legal_moves:
        child = board.copy(stack=True)
        child.push(move)
        children[packed_move(board, move)] = board_position(child)
    oracle = sessions.Oracle(Path('/not-executed'))
    oracle.cache[board_position(board)] = children
    return peer, oracle, children


@pytest.mark.parametrize('enabled', [False, True])
def test_claim_status_flows_through_real_actor_and_batch_scatter(enabled: bool) -> None:
    b = fifty(100)
    peer, oracle, children = peer_for(b)
    actor = Actor(peer, b, HistoryEncoder(b, encoding()), oracle, 0, 4, allow_claims=enabled)
    pos = board_position(b)
    words = [w for x in pos[:8] for w in (x >> 32, x & 0xffffffff)] + list(pos[8:])
    peer.line.side_effect = ['board ' + ' '.join(map(str, words)), 'path 0',
                             *['action ' + str(k) for k in children]]
    broker = Batcher(4, 146)
    broker.register(0, 1)
    actor.receive(f'eval 1 1 0 {len(children)}', broker)
    flight = broker.dispatch(1e20)
    assert flight is not None
    # A synthetic model response only: no native worker or neural forward.
    replies = broker.complete(flight, np.zeros((4, 1858)), np.zeros((4, 3)), now=0)
    assert len(replies) == 1
    actor.deliver(replies[0])
    text = peer.write.call_args.args[0]
    assert int(text.split()[4], 16) == (4 if enabled else 0)
    assert len(actor.ref.nodes) == len(children) + 1 + int(enabled)
    assert all(a.key in children for a in actor.ref.nodes[1:] if a.key != sessions.CLAIM_KEY)
    assert bool(actor.claim_options) == enabled
    assert actor.pending_claim is None
    assert broker.reserved == 0


def test_cancelled_claim_metadata_cannot_enable_an_option() -> None:
    b = fifty(100)
    peer, oracle, children = peer_for(b)
    actor = Actor(peer, b, HistoryEncoder(b, encoding()), oracle, 0, 4, allow_claims=True)
    key = Key(0, 1, 1, 0)
    actor.waiting, actor.actions, actor.pending_claim = key, list(children), Claim('fifty_moves')
    actor.deliver(Completion(key, 'cancelled'))
    assert actor.ref.completed == 0
    assert len(actor.ref.nodes) == 1
    assert not actor.claim_options
    assert actor.pending_claim is None
    with pytest.raises(AssertionError, match='wrong-session'):
        actor.deliver(Completion(key, 'ok', (0.0, 1.0, 0.0), tuple([1.0] * len(children))))
    actor.claim_options[0] = Claim('fifty_moves')
    actor.start()
    assert actor.epoch == 2
    assert actor.claim_options == {}


@pytest.mark.parametrize('prospective', [False, True])
def test_selected_claim_ends_without_advancing_witness_or_history(prospective: bool) -> None:
    b = repeated(1, ('g1f3', 'g8f6', 'f3g1')) if prospective else repeated(2)
    peer, oracle, _ = peer_for(b)
    actor = GameActor(peer, GameSpec('claim', b, claims='search_choice'), encoding(), oracle, 0, 4)
    assert not actor.done  # Unlike unconditional claim_available: actually searches.
    option = claim_option(b)
    assert option is not None
    actor.claim_options[0] = option
    actor.results = [Observation({'best': sessions.CLAIM_KEY}, [])]
    broker = Batcher(4, 146)
    broker.register(0, actor.epoch)
    actor.after_search(broker)
    assert actor.done
    assert actor.end is not None
    assert actor.end.claimed
    assert actor.end.intended_move == ('f6g8' if prospective else None)
    assert actor.root.move_stack == b.move_stack
    assert actor.moves == []
    assert actor.rebound_encoders == 0
    assert len(peer.write.call_args_list) == 1  # The initial config, no advance.
    assert '1/2-1/2' in str(actor.report()['pgn'])


def test_claim_policy_is_explicit_and_default_stays_automatic() -> None:
    b = repeated(2)
    assert adjudicate(b, 'search_choice') is None
    assert adjudicate(b, 'automatic') is None
    assert adjudicate(b, 'claim_available') is not None
    assert GameSpec('default', b).claims == 'automatic'


def test_bogus_root_claim_is_not_accepted_by_controller() -> None:
    b = chess.Board()
    peer, oracle, _ = peer_for(b)
    actor = GameActor(peer, GameSpec('bogus', b, claims='search_choice'), encoding(), oracle, 0, 4)
    actor.results = [Observation({'best': sessions.CLAIM_KEY}, [])]
    broker = Batcher(4, 146)
    broker.register(0, actor.epoch)
    with pytest.raises(AssertionError, match='root evidence'):
        actor.after_search(broker)
    assert actor.root.move_stack == []
    assert actor.moves == []
