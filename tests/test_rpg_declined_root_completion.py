"""Actual parsed Engine controls for zero-work RPG root completion."""
from __future__ import annotations

import threading

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.moves import POLICY_SIZE, move_to_index
from chess_anti_engine.uci import engine as engine_module
from chess_anti_engine.uci.engine import Engine, EngineOptions
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.search import SearchWorker

torch.set_num_threads(2)
DRAW_POSITIONS = (
    "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 100 1",
    "position startpos moves b1c3 b8c6 c3b1 c6b8 b1c3 b8c6 c3b1 c6b8",
)
SINGLE_POSITION = "position fen k7/8/8/8/8/8/1Q6/K7 b - - 0 1"


class PreferredRoot:
    def evaluate_encoded(self, x, relations=None):
        del relations
        policy = np.zeros((len(x), POLICY_SIZE), np.float32)
        policy[:, int(move_to_index(chess.Move.from_uci("e2e4"), chess.Board()))] = 8.0
        # Deliberately nonzero NN value: a declined DRAW must keep oracle Q=0.
        wdl = np.tile(np.array([1.2, 0.0, -0.7], np.float32), (len(x), 1))
        return policy, wdl


class CpuSlots(PreferredRoot):
    n_slots = 2

    def __init__(self):
        self.calls = 0
        self.buffers = [np.zeros((8, 146, 8, 8), np.float32) for _ in range(2)]

    def get_input_buffer(self, bsz, slot=0):
        assert bsz <= 8
        return self.buffers[slot][:bsz]

    def evaluate_inplace_async(self, bsz, *, slot=0, relations=None):
        self.calls += 1
        policy, wdl = self.evaluate_encoded(self.buffers[slot][:bsz], relations)
        return policy, wdl, None


class ObservedWorker(SearchWorker):
    def __init__(self):
        super().__init__(PreferredRoot(), device="cpu", n_walkers=1, chunk_sims=3,
                         gumbel_cfg=GumbelConfig(topk=4, simulations=4, add_noise=False,
                                               temperature=0.0, input_extra_features="v1"))
        self.results = []
        self.returned = threading.Event()

    def run(self, board, **kwargs):
        result = super().run(board, **kwargs)
        self.results.append(result)
        self.returned.set()
        return result


@pytest.fixture
def session(monkeypatch):
    lines = []
    monkeypatch.setattr(engine_module, "_println", lines.append)
    worker = ObservedWorker()
    engine = Engine(worker, options=EngineOptions(hash_mb=64, max_batch=4, vl_gather=4,
                                                  ponder=True),
                    rebuild_multi_gpu_pucv_factories=lambda max_batch, gather: [CpuSlots, CpuSlots])
    engine.dispatch(parse_command("setoption name SearchParallel value gumbel"))
    assert worker.realized_search_path() == "rpg"
    try:
        yield engine, worker, lines
    finally:
        engine.close()


def send(engine, text):
    engine.dispatch(parse_command(text))


def finish(engine):
    thread = engine._search_thread
    if thread is not None:
        thread.join(5)
        assert not thread.is_alive(), "known root must return without exhausting a time cap"


def bestmoves(lines):
    return [line.split()[1] for line in lines if line.startswith("bestmove ")]


def assert_draw_result(engine, worker, lines, *, expected="e2e4"):
    result = worker.results[-1]
    assert result.nodes == 0
    assert result.bestmove_uci == expected
    assert result.score_cp == 0, "keep exact DRAW, not the nonzero cached NN value"
    assert result.search_incomplete is None
    assert result.root_declined is not None
    assert engine.bestmove_fallback_used == 0
    assert bestmoves(lines)[-1] == expected
    assert any("prior_only_root=1" in line and "nodes=0" in line for line in lines)


@pytest.mark.parametrize("position", DRAW_POSITIONS)
@pytest.mark.parametrize("go", ["go nodes 3", "go depth 3"])
def test_declined_draw_uses_cached_legal_policy_without_simulations(session, position, go):
    engine, worker, lines = session
    send(engine, position)
    assert not engine._board.is_game_over()
    assert engine._board.can_claim_draw()
    send(engine, go)
    finish(engine)
    assert_draw_result(engine, worker, lines)
    assert engine.prior_only_roots == 1
    stats = worker.last_root_parallel_gumbel_stats()
    assert stats is not None
    assert stats.root_finished
    assert stats.phases == ()
    assert sum(stats.group_sims) == 0
    assert stats.tree_nodes == 1
    pool = worker._rpg_pool
    assert pool is not None
    assert pool._state is not None
    assert pool._state.finished_value == 0.0
    assert pool._state.tree.node_q(pool._state.root_id) == 0.0
    assert sum(ev.calls for ev in pool._evals) == 0


@pytest.mark.parametrize("position", DRAW_POSITIONS)
def test_searchmoves_restricts_declined_policy(session, position):
    engine, worker, lines = session
    send(engine, position)
    send(engine, "go nodes 3 searchmoves d2d4")
    finish(engine)
    assert_draw_result(engine, worker, lines, expected="d2d4")
    stats = worker.last_root_parallel_gumbel_stats()
    assert stats is not None
    assert stats.target_sims == 0
    assert not stats.phases


@pytest.mark.parametrize("position", [DRAW_POSITIONS[0], SINGLE_POSITION])
def test_infinite_completed_root_waits_for_stop_once(session, position):
    engine, worker, lines = session
    send(engine, position)
    send(engine, "go infinite")
    assert worker.returned.wait(5), "finished root must not spin on a zero-work chunk"
    assert bestmoves(lines) == []
    assert worker.results[-1].nodes == 0
    assert worker.results[-1].search_incomplete is None
    send(engine, "stop")
    finish(engine)
    assert len(bestmoves(lines)) == 1
    assert engine.bestmove_fallback_used == 0
    stats = worker.last_root_parallel_gumbel_stats()
    assert stats is not None
    assert stats.root_finished
    assert not stats.phases


def test_ponderhit_declined_draw_retains_exact_value_and_single_output(session):
    engine, worker, lines = session
    send(engine, DRAW_POSITIONS[1])
    send(engine, "go ponder nodes 3")
    assert worker.returned.wait(5)
    assert bestmoves(lines) == []
    send(engine, "ponderhit")
    finish(engine)
    assert len(worker.results) == 2
    assert len(bestmoves(lines)) == 1
    assert_draw_result(engine, worker, lines)


def test_repeated_go_retains_declined_completion(session):
    engine, worker, lines = session
    send(engine, DRAW_POSITIONS[0])
    for _ in range(2):
        send(engine, "go nodes 3")
        finish(engine)
        assert_draw_result(engine, worker, lines)
    assert bestmoves(lines) == ["e2e4", "e2e4"]
    assert engine.prior_only_roots == 2


def test_single_legal_root_finishes_with_zero_work(session):
    engine, worker, lines = session
    send(engine, SINGLE_POSITION)
    send(engine, "go nodes 3")
    finish(engine)
    result = worker.results[-1]
    assert result.nodes == 0
    assert result.bestmove_uci == "a8a7"
    assert result.root_declined is None
    assert result.search_incomplete is None
    assert bestmoves(lines) == ["a8a7"]
    stats = worker.last_root_parallel_gumbel_stats()
    assert stats is not None
    assert stats.root_finished
    assert stats.phases == ()
    assert sum(stats.group_sims) == 0
    assert engine.prior_only_roots == 0


def test_ordinary_rpg_reports_actual_simulations(session):
    engine, worker, lines = session
    send(engine, "position startpos")
    send(engine, "go nodes 3")
    finish(engine)
    result = worker.results[-1]
    stats = worker.last_root_parallel_gumbel_stats()
    assert result.nodes == 3
    assert result.root_declined is None
    assert stats is not None
    assert not stats.root_finished
    assert sum(stats.group_sims) == sum(p.sims_completed for p in stats.phases) == 3
    assert chess.Move.from_uci(bestmoves(lines)[0]) in chess.Board().legal_moves
    assert engine.bestmove_fallback_used == engine.prior_only_roots == 0


@pytest.mark.parametrize(("fen", "mate"), [("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1", 0),
                                     ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", None)])
def test_no_legal_terminal_contract_unchanged(session, fen, mate):
    engine, worker, lines = session
    send(engine, "position fen " + fen)
    send(engine, "go nodes 3")
    finish(engine)
    result = worker.results[-1]
    assert result.bestmove_uci == "0000"
    assert result.nodes == 0
    assert result.score_mate == mate
    assert bestmoves(lines) == ["0000"]
    assert engine.prior_only_roots == 0


def test_insufficient_material_oracle_value_unchanged(session):
    engine, worker, lines = session
    send(engine, "position fen 7k/8/8/8/8/8/8/K7 w - - 0 1")
    send(engine, "go nodes 3")
    finish(engine)
    result = worker.results[-1]
    assert result.nodes == 0
    assert result.score_cp == 0
    assert result.root_declined == "insufficient material"
    assert result.search_incomplete is None
    assert chess.Move.from_uci(bestmoves(lines)[0]) in engine._board.legal_moves
    assert engine.bestmove_fallback_used == 0
    assert engine.prior_only_roots == 1
