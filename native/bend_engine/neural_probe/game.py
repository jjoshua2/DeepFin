"""Host-owned played-game lifecycle around the native Bend diagnostic search.

The host adjudicates played roots and automatic draw leaves; Bend caches their
zero terminal values. The opt-in search_choice policy retains claim actions. A stopped or
truncated experiment is unfinished (*), never a draw. This controller deliberately
uses the existing verified Actor and ACK protocol rather than another search.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import chess
import chess.pgn

from native.bend_engine.session_probe import run_probe as sessions
from native.bend_engine.session_probe.claims import claim_option
from native.bend_engine.session_probe.root_protocol import Wire, advance_root, packed_move
from .adapter import Encoding, HistoryEncoder, board_position, decode_key
from .batch_probe import Actor
from .batching import Batcher

CLAIM_POLICIES = ('automatic', 'claim_available', 'search_choice')


@dataclass(frozen=True)
class GameEnd:
    result: str
    reason: str
    claimed: bool = False
    intended_move: str | None = None


def adjudicate(board: chess.Board, claims: str) -> GameEnd | None:
    """python-chess rules; material-only dead-position detection, not a solver.

    'claim_available' deliberately elects to claim, including by an intended
    legal move. That move is reported but not played. 'automatic' never claims.
    Always preserve the supplied board and move stack.
    """
    if claims not in CLAIM_POLICIES:
        raise ValueError('unknown draw-claim policy')
    if board.chess960 or not board.is_valid():
        raise ValueError('game requires valid orthodox chess')
    outcome = board.outcome(claim_draw=claims == 'claim_available')
    if outcome is None:
        return None
    reason = outcome.termination.name.lower()
    claim = reason in ('fifty_moves', 'threefold_repetition')
    intended = None
    if claim:
        current = board.is_fifty_moves() if reason == 'fifty_moves' else board.is_repetition(3)
        if not current:
            for move in board.legal_moves:
                child = board.copy(stack=True)
                child.push(move)
                eligible = child.is_fifty_moves() if reason == 'fifty_moves' else child.is_repetition(3)
                if eligible:
                    intended = move.uci()
                    break
            if intended is None:
                raise RuntimeError('claim by intended move has no legal witness')
    return GameEnd(outcome.result(), reason, claim, intended)


@dataclass(frozen=True)
class GameSpec:
    name: str
    root: chess.Board
    max_plies: int = 8
    claims: str = 'automatic'
    scripted: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.max_plies) is not int or not 1 <= self.max_plies <= 128:
            raise ValueError('game ply limit must be 1..128')
        if not self.name:
            raise ValueError('game name must be nonempty')
        adjudicate(self.root, self.claims)  # validate without treating a draw as an error
        if len(self.scripted) > self.max_plies:
            raise ValueError('script exceeds the game ply limit')
        # Scripts are explicit opponent/fixture choices, never advertised as NN play.
        child = self.root.copy(stack=True)
        for uci in self.scripted:
            if adjudicate(child, self.claims) is not None:
                raise ValueError('script continues past a game result')
            move = chess.Move.from_uci(uci)
            if move not in child.legal_moves:
                raise ValueError('script contains an illegal move')
            child.push(move)


def game_end(board: chess.Board, claims: str, played: int, limit: int) -> GameEnd | None:
    # Mate/actual draws take precedence over the experiment's move budget.
    return adjudicate(board, claims) or (GameEnd('*', 'ply_limit') if played >= limit else None)


def retire_and_advance(peer: Wire, broker: Batcher, session: int,
                       root: chess.Board, epoch: int, key: int) -> tuple[chess.Board, int]:
    """At ready only. Cancelled flight rows may remain, live pending rows may not.

    A lost/corrupt ACK leaves the remote uncertain. The caller must close it;
    never retry or adopt a new local root before this whole operation succeeds.
    """
    if session in broker.pending:
        raise ValueError('cannot advance with a pending evaluator request')
    if broker.epochs.get(session, (0, 0))[0] != epoch:
        raise ValueError('game/broker epoch mismatch')
    if epoch >= sessions.SENTINEL - 1:
        raise ValueError('no epoch capacity for advance and next search')
    child, reply = advance_root(peer, root, expected_epoch=epoch, new_epoch=epoch + 1, key=key)
    if reply.status != 0:
        raise ValueError('native game advance was rejected')
    broker.register(session, reply.current_epoch)
    return child, reply.current_epoch


def pgn_text(root: chess.Board, end: GameEnd, name: str) -> str:
    game = chess.pgn.Game.from_board(root)
    game.headers['Event'] = 'Bend native evaluator qualification: ' + name
    game.headers['Result'] = end.result
    game.headers['Termination'] = end.reason
    return game.accept(chess.pgn.StringExporter(headers=True, variations=False, comments=False))


class GameActor(Actor):
    """Fresh tree per played move, with one immutable encoder per acknowledged root.

    Cancellation never implicitly plays the partial search's best move. A scripted
    next move can model an external opponent/fixture transition after cancellation;
    otherwise it ends the game as unfinished. Backend failure/expiry always abort.
    """
    def __init__(self, peer: sessions.Peer, spec: GameSpec, encoding: Encoding,
                 oracle: sessions.Oracle, session: int, budget: int):
        if type(budget) is not int or not 2 <= budget <= 64:
            raise ValueError('game simulation budget must be 2..64')
        self.spec = spec
        self.moves: list[dict[str, object]] = []
        self.cancelled_epochs: list[int] = []
        self.end: GameEnd | None = None
        self.pid = peer.proc.pid
        self.initial_history = len(spec.root.move_stack)
        self.rebound_encoders = 0
        root = spec.root.copy(stack=True)
        super().__init__(peer, root, HistoryEncoder(root, encoding), oracle, session, budget,
                         allow_claims=spec.claims == 'search_choice')

    def start(self) -> None:
        self.end = game_end(self.root, self.spec.claims, len(self.moves), self.spec.max_plies)
        if self.end is not None:
            self.done = True
            return
        if self.peer.proc.pid != self.pid or self.peer.proc.poll() is not None:
            raise RuntimeError('game peer was restarted or died')
        super().start()

    def after_search(self, broker: Batcher) -> None:
        if self.waiting is not None or self.session in broker.pending:
            raise AssertionError('finished search still owns a pending request')
        stopped = self.ref.stop
        scripted = len(self.moves) < len(self.spec.scripted)
        if stopped:
            if stopped != 2 or not scripted:
                self.end = GameEnd('*', 'cancelled' if stopped == 2 else 'search_stopped')
                self.done = True
                return
            self.cancelled_epochs.append(self.epoch)
        if scripted:
            move = chess.Move.from_uci(self.spec.scripted[len(self.moves)])
            key = packed_move(self.root, move)
        else:
            key = self.results[-1].summary['best']
            if key == sessions.CLAIM_KEY:
                option = claim_option(self.root)
                if self.spec.claims != 'search_choice' or option is None or option != self.claim_options.get(0):
                    raise AssertionError('selected claim lacks current root evidence')
                self.end = GameEnd('1/2-1/2', option.reason, True, option.intended_move)
                self.done = True
                return  # Claim BEFORE any witness move: no root advance or history push.
            if key == sessions.SENTINEL:
                raise AssertionError('nonterminal game has no best move')
            move = decode_key(self.root, key)
        previous = self.root
        child, epoch = retire_and_advance(self.peer, broker, self.session, previous, self.epoch, key)
        # ACK source/target board, move stack and retired work are checked before
        # constructing the next encoder. Never reuse a historyless or old encoder.
        if self.oracle.moves(board_position(previous)).get(key) != board_position(child):
            raise AssertionError("played root differs from native CBoard")
        encoder = HistoryEncoder(child, self.encoder.encoding)
        if child.move_stack != [*previous.move_stack, move]:
            raise AssertionError('game advance lost pre-root history')
        if encoder.root.move_stack != child.move_stack or encoder.root.fen(en_passant='fen') != child.fen(en_passant='fen'):
            raise AssertionError('next search encoder does not own the played history')
        self.moves.append({'uci': move.uci(), 'source': 'script' if scripted else 'search',
                           'key': key, 'search_epoch': self.epoch, 'advance_epoch': epoch,
                           'completed': self.ref.completed, 'nodes': len(self.ref.nodes),
                           'fen': child.fen(en_passant='fen'), 'history_plies': len(child.move_stack)})
        self.root, self.epoch, self.encoder = child, epoch, encoder
        self.rebound_encoders += 1
        self.start()
        if not self.done:
            broker.register(self.session, self.epoch)

    def report(self) -> dict[str, object]:
        if self.end is None:
            raise ValueError('cannot publish unfinished controller state as a game result')
        return {'name': self.spec.name, 'end': asdict(self.end), 'moves': self.moves,
                'pre_root_plies': self.initial_history, 'played_plies': len(self.moves),
                'claims': self.spec.claims,
                'final_fen': self.root.fen(en_passant='fen'),
                'cancelled_epochs': self.cancelled_epochs, 'rebound_encoders': self.rebound_encoders,
                'search_epochs': [r.summary for r in self.results],
                'automatic_draw_leaves': self.draw_leaves,
                'optional_claim_leaves': self.claim_leaves,
                'pgn': pgn_text(self.root, self.end, self.spec.name)}
