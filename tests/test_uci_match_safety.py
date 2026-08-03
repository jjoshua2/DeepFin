"""Match-play safety fixes from the wave-4 UCI audit (U1, U2, U3, U5, U6, U7).

Every test here is red on ``origin/main``. The theme is the house defect in its
match-play form — a value that is computed, retained, and then silently ignored
at exactly the moment it is needed — so the assertions are about what the
PRODUCTION path emits, not about internal state:

* U2: an in-search exception is answered from the cached root policy, and says
  so in an ``info string`` a match log can be grepped for.
* U1: ``run()`` on a root ``CBoard.is_game_over()`` reports (which includes
  CLAIMABLE draws that still have legal moves) returns a legal move instead of
  raising ``node_id out of range`` into the U2 handler.
* U3: ``movestogo 1`` gets the clock it is owed.
* U5/U6: a short FEN is parsed rather than silently replaced by the start
  position, and a truncated move list is announced.
* U7: the advertised ``Hash`` minimum is the enforced one.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import chess
import numpy as np
import pytest

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.moves import POLICY_SIZE, move_to_index
from chess_anti_engine.uci.engine import Engine, EngineOptions, _MIN_HASH_MB
from chess_anti_engine.uci.protocol import CmdPosition, CmdSetOption, parse_command
from chess_anti_engine.uci.search import SearchResult, SearchWorker
from chess_anti_engine.uci.time_manager import Deadline, SearchLimits, limits_from_go
from chess_anti_engine.uci.protocol import GoArgs

# python-chess iterates legal moves in square order, so this is what the old
# `next(iter(board.legal_moves))` fallback plays from the start position. It is
# a knight to the rim — the point of U2 is that it is effectively random.
_FIRST_LEGAL_FROM_STARTPOS = "g1h3"

# Roots `CBoard.is_game_over()` reports true for. The last two have LEGAL MOVES
# and are ordinary playable positions: only a *claim* would end them.
_DECLINED_ROOTS = {
    "checkmate": "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    "stalemate": "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    "fifty_move_with_legal_moves": "8/8/8/4k3/8/4K3/4R3/8 w - - 100 60",
    "insufficient_material_with_legal_moves": "8/8/8/4k3/8/2B1K3/8/8 w - - 0 1",
}


class _FixedEvaluator:
    """Returns a fixed policy/WDL for any batch; optionally raises after N calls.

    Deliberately NOT a ``MagicMock``: ``run_gumbel_root_many_c`` probes the
    evaluator with ``getattr`` for a dozen optional fast paths, and a mock
    answers "yes" to all of them.
    """

    def __init__(
        self,
        *,
        preferred: dict[int, float] | None = None,
        raise_after: int | None = None,
    ) -> None:
        self._logits = np.zeros(POLICY_SIZE, dtype=np.float32)
        for idx, value in (preferred or {}).items():
            self._logits[idx] = value
        self._raise_after = raise_after
        self.calls = 0

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations  # part of the evaluator signature; this stub ignores it
        self.calls += 1
        if self._raise_after is not None and self.calls > self._raise_after:
            raise RuntimeError("synthetic evaluator failure")
        n = int(np.asarray(x).shape[0])
        pol = np.repeat(self._logits[None, :], n, axis=0)
        wdl = np.zeros((n, 3), dtype=np.float32)
        return pol, wdl


def _worker(evaluator: Any, *, n_walkers: int = 1) -> SearchWorker:
    return SearchWorker(
        evaluator,
        device="cpu",
        gumbel_cfg=GumbelConfig(simulations=8, add_noise=False, temperature=0.0),
        chunk_sims=8,
        n_walkers=n_walkers,
    )


def _index(board: chess.Board, uci: str) -> int:
    return int(move_to_index(chess.Move.from_uci(uci), board))


# --- U2: the fallback plays the net's move, not the first legal one ---------


def test_search_exception_falls_back_to_root_policy_argmax(capsys) -> None:
    """The audit's repro: the evaluator raises AFTER the root eval is cached.

    Red on main: the engine answers `g1h3`. The distinguishing power is the
    whole point — `e2e4` and the first legal move are different moves, so a
    reverted fix cannot pass this by accident.
    """
    board = chess.Board()
    evaluator = _FixedEvaluator(preferred={_index(board, "e2e4"): 12.0}, raise_after=1)
    engine = Engine(worker=_worker(evaluator))
    limits = SearchLimits(deadline_ms=None, max_nodes=64, searchmoves=())

    result = engine._run_one_phase(limits, is_ponder=False, board=board)

    assert result.bestmove_uci == "e2e4"
    assert result.bestmove_uci != _FIRST_LEGAL_FROM_STARTPOS
    out = capsys.readouterr().out
    assert "info string search error: RuntimeError" in out
    assert "bestmove_fallback_used=1 source=root_policy move=e2e4" in out
    assert "exception=RuntimeError" in out
    assert engine.bestmove_fallback_used == 1


def test_bestmove_fallback_counter_is_zero_on_a_clean_search(capsys) -> None:
    """Negative control: nothing raises, so nothing is counted or printed."""
    worker = MagicMock()
    worker.run = MagicMock(return_value=SearchResult(
        bestmove_uci="e2e4", ponder_uci=None, nodes=99, pv=("e2e4",),
        score_cp=10, tbhits=0,
    ))
    engine = Engine(worker=worker)
    limits = SearchLimits(deadline_ms=None, max_nodes=None, searchmoves=())

    for _ in range(3):
        engine._run_one_phase(limits, is_ponder=False, board=chess.Board())

    assert engine.bestmove_fallback_used == 0
    assert "bestmove_fallback_used" not in capsys.readouterr().out


def test_fallback_counter_accumulates_across_moves(capsys) -> None:
    worker = MagicMock()
    worker.run = MagicMock(side_effect=RuntimeError("boom"))
    engine = Engine(worker=worker)
    limits = SearchLimits(deadline_ms=None, max_nodes=None, searchmoves=())

    for _ in range(3):
        engine._run_one_phase(limits, is_ponder=False, board=chess.Board())

    assert engine.bestmove_fallback_used == 3
    assert "bestmove_fallback_used=3" in capsys.readouterr().out


def test_fallback_rejects_a_worker_answer_that_is_illegal_here(capsys) -> None:
    """A fallback illegal in the real position is a forfeit — worse than random.

    The engine re-validates instead of trusting the worker, and the
    ``source=`` field reports which path actually supplied the move.
    """
    worker = MagicMock()
    worker.run = MagicMock(side_effect=RuntimeError("boom"))
    worker.root_policy_bestmove = MagicMock(return_value="e2e5")  # not legal
    engine = Engine(worker=worker)
    limits = SearchLimits(deadline_ms=None, max_nodes=None, searchmoves=())

    result = engine._run_one_phase(limits, is_ponder=False, board=chess.Board())

    assert result.bestmove_uci == _FIRST_LEGAL_FROM_STARTPOS
    assert "source=first_legal" in capsys.readouterr().out


def test_fallback_survives_a_worker_that_raises_in_the_policy_path(capsys) -> None:
    worker = MagicMock()
    worker.run = MagicMock(side_effect=RuntimeError("boom"))
    worker.root_policy_bestmove = MagicMock(side_effect=ValueError("no logits"))
    engine = Engine(worker=worker)
    limits = SearchLimits(deadline_ms=None, max_nodes=None, searchmoves=())

    result = engine._run_one_phase(limits, is_ponder=False, board=chess.Board())

    assert result.bestmove_uci == _FIRST_LEGAL_FROM_STARTPOS
    assert "root-policy fallback unavailable" in capsys.readouterr().out


def test_root_policy_bestmove_honours_searchmoves() -> None:
    board = chess.Board()
    evaluator = _FixedEvaluator(preferred={
        _index(board, "e2e4"): 12.0, _index(board, "d2d4"): 6.0,
    })
    worker = _worker(evaluator)
    worker._ensure_root_eval_cached(board, None)

    assert worker.root_policy_bestmove(board) == "e2e4"
    assert worker.root_policy_bestmove(board, searchmoves=("d2d4", "a2a3")) == "d2d4"
    assert worker.root_policy_bestmove(board, searchmoves=("a1a8",)) is None


def test_root_policy_bestmove_is_none_before_any_root_eval() -> None:
    """The failure happened before/inside the root eval — no prior to use."""
    worker = _worker(_FixedEvaluator())

    assert worker.root_policy_bestmove(chess.Board()) is None


# --- U1: roots the C Gumbel path declines ----------------------------------


@pytest.mark.parametrize(("name", "fen"), sorted(_DECLINED_ROOTS.items()))
@pytest.mark.parametrize("n_walkers", [1, 2])
def test_run_answers_declined_roots_without_raising(
    name: str, fen: str, n_walkers: int,
) -> None:
    """Red on main at ``n_walkers=1``: ``ValueError: node_id out of range``.

    Positions with legal moves must get one of them; a 0-legal-move position
    gets the UCI null move. Both arms of the audit's U1 yardstick.
    """
    import threading

    board = chess.Board(fen)
    worker = _worker(_FixedEvaluator(), n_walkers=n_walkers)

    result = worker.run(
        board,
        stop_event=threading.Event(),
        deadline=Deadline(400),
        max_nodes=64,
    )

    legal = {m.uci() for m in board.legal_moves}
    if legal:
        assert result.bestmove_uci in legal, f"{name}: {result.bestmove_uci}"
    else:
        assert result.bestmove_uci == "0000", name


def test_run_on_a_mated_root_returns_immediately() -> None:
    """0 legal moves must not burn the deadline (the walker pool used to spin
    every walker into the terminal root and report ~35k 'nodes')."""
    import threading

    board = chess.Board(_DECLINED_ROOTS["checkmate"])
    worker = _worker(_FixedEvaluator(), n_walkers=2)

    result = worker.run(
        board,
        stop_event=threading.Event(),
        deadline=Deadline(2000),
        max_nodes=None,
    )

    assert result.bestmove_uci == "0000"
    assert result.nodes == 0
    assert result.score_mate == 0


def test_declined_root_plays_the_policy_argmax_not_the_first_legal_move() -> None:
    """A claimable-draw root is playable, so the answer must be the net's move."""
    import threading

    board = chess.Board(_DECLINED_ROOTS["fifty_move_with_legal_moves"])
    legal = [m.uci() for m in board.legal_moves]
    target = next(m for m in legal if m != legal[0])
    worker = _worker(
        _FixedEvaluator(preferred={_index(board, target): 12.0}), n_walkers=1,
    )

    result = worker.run(
        board,
        stop_event=threading.Event(),
        deadline=Deadline(400),
        max_nodes=64,
    )

    assert result.bestmove_uci == target


# --- U3: movestogo 1 -------------------------------------------------------


def test_movestogo_one_gets_almost_the_whole_clock() -> None:
    """Red on main: 29970 ms of a 60 s clock (the 50%-of-remaining ceiling)."""
    limits = limits_from_go(
        GoArgs(wtime_ms=60_000, movestogo=1),
        side_to_move_is_white=True,
        move_overhead_ms=0,
    )

    assert limits.deadline_ms is not None
    assert limits.deadline_ms > 0.9 * 60_000
    assert limits.deadline_ms <= 60_000


def test_movestogo_above_one_keeps_the_half_clock_ceiling() -> None:
    """The ceiling still binds where it was designed to — a long horizon."""
    limits = limits_from_go(
        GoArgs(wtime_ms=60_000, movestogo=5),
        side_to_move_is_white=True,
        move_overhead_ms=0,
    )

    assert limits.deadline_ms is not None
    assert limits.deadline_ms <= 0.5 * 60_000


def test_movestogo_zero_is_not_treated_as_the_last_move() -> None:
    """`movestogo 0` means "no repeating control", not "one move left" — the
    allocation ignores it, so the long-horizon ceiling must still apply."""
    limits = limits_from_go(
        GoArgs(wtime_ms=60_000, movestogo=0),
        side_to_move_is_white=True,
        move_overhead_ms=0,
    )

    assert limits.deadline_ms is not None
    assert limits.deadline_ms <= 0.5 * 60_000


def test_no_movestogo_keeps_the_half_clock_ceiling() -> None:
    limits = limits_from_go(
        GoArgs(wtime_ms=60_000, winc_ms=600),
        side_to_move_is_white=True,
        move_overhead_ms=0,
    )

    assert limits.deadline_ms is not None
    assert limits.deadline_ms <= 0.5 * 60_000


# --- U5: short / bare / bad FENs -------------------------------------------


_MIDGAME_FEN_FIELDS = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq -"


@pytest.mark.parametrize("suffix", ["", " 3", " 3 4"])
def test_short_fens_are_parsed_not_replaced_by_the_start_position(suffix: str) -> None:
    """Red on main for the 4- and 5-field forms: both returned ``fen=None``,
    which MEANS startpos, so the engine answered for a different board."""
    cmd = parse_command(f"position fen {_MIDGAME_FEN_FIELDS}{suffix}")

    assert isinstance(cmd, CmdPosition)
    assert cmd.fen is not None
    assert chess.Board(cmd.fen).board_fen() == _MIDGAME_FEN_FIELDS.split()[0]


def test_bare_fen_without_the_fen_keyword_is_parsed() -> None:
    cmd = parse_command(f"position {_MIDGAME_FEN_FIELDS} 3 4")

    assert isinstance(cmd, CmdPosition)
    assert cmd.fen is not None
    assert chess.Board(cmd.fen).board_fen() == _MIDGAME_FEN_FIELDS.split()[0]


@pytest.mark.parametrize("suffix", ["", " 3", " 3 4"])
def test_short_fens_still_take_their_move_list(suffix: str) -> None:
    cmd = parse_command(f"position fen {_MIDGAME_FEN_FIELDS}{suffix} moves f8c5 e1g1")

    assert isinstance(cmd, CmdPosition)
    assert cmd.moves == ("f8c5", "e1g1")


def test_six_field_fen_is_unchanged() -> None:
    cmd = parse_command(f"position fen {_MIDGAME_FEN_FIELDS} 3 4 moves f8c5")

    assert isinstance(cmd, CmdPosition)
    assert cmd.fen == f"{_MIDGAME_FEN_FIELDS} 3 4"
    assert cmd.moves == ("f8c5",)


def test_startpos_is_unchanged() -> None:
    cmd = parse_command("position startpos moves e2e4 e7e5")

    assert isinstance(cmd, CmdPosition)
    assert cmd.fen is None
    assert cmd.moves == ("e2e4", "e7e5")


def test_unparseable_fen_is_announced_before_falling_back_to_startpos(capsys) -> None:
    engine = Engine(worker=MagicMock())

    engine._handle_position(CmdPosition(fen="not/a/fen w - - 0 1", moves=()))

    out = capsys.readouterr().out
    assert "info string position: rejected FEN" in out
    assert "using the start position" in out
    assert engine._board == chess.Board()


# --- U6: truncated move lists ----------------------------------------------


@pytest.mark.parametrize(
    ("moves", "applied"),
    [
        (("e2e4", "e7e5", "e1e8", "g1f3"), 2),  # illegal third move
        (("e7e8Q",), 0),                        # uppercase promotion
        (("e2e4", "0000", "e7e5"), 1),          # null move mid-list
    ],
)
def test_truncated_move_list_is_announced(
    moves: tuple[str, ...], applied: int, capsys,
) -> None:
    """Red on main: `break` with no diagnostic. The engine then searches a
    position n plies behind the game and emits a move that is likely illegal
    there — a forfeit, delivered silently."""
    engine = Engine(worker=MagicMock())

    engine._handle_position(CmdPosition(fen=None, moves=moves))

    out = capsys.readouterr().out
    assert "info string position: ignored" in out
    assert f"ignored {len(moves) - applied} of {len(moves)} move(s)" in out
    assert len(engine._board.move_stack) == applied


def test_complete_move_list_says_nothing(capsys) -> None:
    engine = Engine(worker=MagicMock())

    engine._handle_position(CmdPosition(fen=None, moves=("e2e4", "e7e5", "g1f3")))

    assert "info string position:" not in capsys.readouterr().out
    assert len(engine._board.move_stack) == 3


# --- U7: the advertised Hash floor is the enforced one ---------------------


@pytest.mark.parametrize("requested", ["16", "256", "-5", "0"])
def test_hash_below_the_advertised_minimum_is_clamped_up(requested: str, capsys) -> None:
    """Red on main: `Hash 256` was accepted verbatim, which kills cross-move
    tree reuse for the whole game and throttles every chunk."""
    worker = MagicMock()
    engine = Engine(worker=worker, options=EngineOptions())

    engine._handle_setoption(CmdSetOption(name="Hash", value=requested))

    assert engine._options.hash_mb == _MIN_HASH_MB
    worker.set_max_tree_mb.assert_called_with(_MIN_HASH_MB)
    assert "below the advertised minimum" in capsys.readouterr().out


def test_hash_at_or_above_the_minimum_is_taken_verbatim(capsys) -> None:
    worker = MagicMock()
    engine = Engine(worker=worker, options=EngineOptions())

    engine._handle_setoption(CmdSetOption(name="Hash", value="4096"))

    assert engine._options.hash_mb == 4096
    worker.set_max_tree_mb.assert_called_with(4096)
    assert "below the advertised minimum" not in capsys.readouterr().out


def test_advertised_hash_minimum_matches_the_enforced_one(capsys) -> None:
    """The two disagreeing is the defect; assert them against each other."""
    emit = capsys
    from chess_anti_engine.uci.engine import emit_handshake

    emit_handshake(EngineOptions())

    line = next(
        ln for ln in emit.readouterr().out.splitlines()
        if ln.startswith("option name Hash ")
    )
    assert f"min {_MIN_HASH_MB}" in line
