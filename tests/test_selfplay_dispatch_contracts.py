"""Real CPU boards and production dispatch with inert search/engine boundaries."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import chess
import numpy as np
import pytest

from chess_anti_engine import worker
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask
from chess_anti_engine.selfplay import network_turn as nt
from chess_anti_engine.selfplay.config import DiffFocusConfig, GameConfig, SearchConfig, TemperatureConfig
from chess_anti_engine.selfplay.state import BatchStats, SelfplayState
from chess_anti_engine.stockfish.uci import StockfishTimeoutError


class Draws:
    def __init__(self, draws):
        self.draws = np.asarray(draws, dtype=float)
    def random(self, size=None):
        assert size == len(self.draws), size
        return self.draws.copy()

class InertEvaluator:
    pads_batches_internally = True
    def evaluate_encoded(self, xs, **_kwargs):
        return np.zeros((len(xs), POLICY_SIZE), np.float32), np.zeros((len(xs), 3), np.float32)

def search_output(boards):
    probabilities, actions, masks = [], [], []
    for board in boards:
        if hasattr(board, "legal_move_indices"):
            mask = np.zeros(POLICY_SIZE, dtype=bool)
            mask[board.legal_move_indices()] = True
        else:
            mask = legal_move_mask(board)
        legal = np.flatnonzero(mask)
        probabilities.append(mask.astype(np.float32) / len(legal))
        actions.append(int(legal[0]))
        masks.append(mask)
    return probabilities, actions, [0.0] * len(boards), masks

def network_budget_case(name, draws, mcts) -> dict[str, Any]:
    boards = [chess.Board(), chess.Board()]
    state: Any = SimpleNamespace(
        boards=boards, cboards=[CBoard.from_board(b) for b in boards],
        model=None, device="cpu", evaluator=InertEvaluator(), rng=Draws(draws),
        search=SearchConfig(mcts_type=mcts, simulations=128, fast_simulations=8, playout_cap_fraction=0.5),
        temp=TemperatureConfig(temperature=0.0, decay_moves=0, endgame=0.0),
        game=GameConfig(input_extra_features="v1", record_prior_top1=False),
        diff_focus=DiffFocusConfig(enabled=False), diff_focus_norm=None,
        batch_enc_146=None, batch_enc_146_bf16=None,
        batch_enc_146_lc0_root=None, batch_enc_146_lc0_root_bf16=None,
        batch_enc_146_lc0_root_legacy_meta=None, batch_enc_146_lc0_root_legacy_meta_bf16=None,
        has_c_ply=False, has_classify_c=False, c_temp_resample=None,
        mcts_tree=None, root_ids=[-1, -1], tb_probe=None,
        selfplay_arr=np.ones(2, np.int8), done_arr=np.zeros(2, np.int8),
        force_full_next=[False, False], last_net_full=[True, True],
        consecutive_low_winrate=[0, 0], move_idx_history=[[], []], samples_per_game=[[], []],
    )
    state.replay_board = lambda index: SelfplayState.replay_board(state, index)
    calls = []
    def puct(_model, sub_boards, **kwargs):
        calls.append({"runner": "puct", "boards": len(sub_boards),
                      "cfg_simulations": kwargs["cfg"].simulations,
                      "per_game_simulations": kwargs.get("per_game_simulations")})
        return search_output(sub_boards)
    def gumbel(_model, sub_boards, **kwargs):
        calls.append({"runner": "gumbel", "boards": len(sub_boards),
                      "cfg_simulations": kwargs["cfg"].simulations,
                      "per_game_simulations": kwargs.get("per_game_simulations")})
        return (*search_output(sub_boards), [None] * len(sub_boards))
    with patch.object(nt, "_HAS_C_TREE", True), patch.object(nt, "_HAS_GUMBEL_C", False), \
            patch.object(nt, "_run_mcts_many_c", puct), patch.object(nt, "run_gumbel_root_many", gumbel):
        nt.run_network_turn(state, [0, 1])
    return {"name": name, "draws": draws, "mcts": mcts, "calls": calls,
            "record_has_policy": [records[-1].has_policy for records in state.samples_per_game],
            "records": [len(records) for records in state.samples_per_game],
            "board_plies": [board.ply for board in state.cboards],
            "expected_per_game_budget": [128 if draw < 0.5 else 8 for draw in draws]}

class FakeSF:
    def __init__(self):
        self.closes = 0
    def close(self):
        self.closes += 1

def worker_failure_case(failed_tid) -> dict[str, Any]:
    starts = [threading.Event(), threading.Event()]
    failed_logged, release, driver_done = threading.Event(), threading.Event(), threading.Event()
    healthy_done = threading.Event()
    messages, result = [], {}
    class Log:
        def exception(self, msg, *args):
            messages.append(msg % args)
            failed_logged.set()
        def warning(self, msg, *args):
            messages.append(msg % args)
    session: Any = object.__new__(worker.WorkerSession)
    session.args = SimpleNamespace(threaded_selfplay=True, selfplay_threads=2)
    session.device, session.rng, session.log = "cpu", np.random.default_rng(21), Log()
    session._stop_selfplay = session._shutdown_requested = session._hold_selfplay = False
    session._resume_inflight_enabled = False
    session._live_states, session._pending_live_override = [], None
    session._live_states_lock = threading.Lock()
    session.inference_client, session._direct_evaluator = None, InertEvaluator()
    fake_sf = FakeSF()
    session.sf = fake_sf
    def fake_play_batch(_model, **kwargs):
        tid = 0 if kwargs["on_step"] is not None else 1
        starts[tid].set()
        assert starts[1 - tid].wait(2), "fixture second thread did not start"
        if tid == failed_tid:
            raise StockfishTimeoutError("bounded inert SF failure")
        while not kwargs["stop_fn"]():
            if release.wait(0.005):
                break
        healthy_done.set()
        return [], BatchStats(games=0, positions=0, w=0, d=0, l=0)
    def drive():
        try:
            result["returned_none"] = session._dispatch_selfplay_one_shard(
                games_per_batch=2, cfgs={"diff_focus": DiffFocusConfig()}, need_local_model=False) is None
        except BaseException as exc:
            result["unexpected_exception"] = repr(exc)
        finally:
            driver_done.set()
    thread = threading.Thread(target=drive, name="bounded-review-driver", daemon=True)
    with patch.object(worker, "play_batch", fake_play_batch):
        try:
            thread.start()
            assert failed_logged.wait(2), "fixture failure was not observed"
            if session._stop_selfplay:
                assert driver_done.wait(2), "cooperative peers did not stop"
            else:
                time.sleep(0.05)
            observation = {"failed_tid": failed_tid, "stop_latch_after_failure": session._stop_selfplay,
                           "driver_done_before_fixture_release": driver_done.is_set(),
                           "healthy_thread_done_before_fixture_release": healthy_done.is_set(),
                           "sf_closed_before_fixture_release": fake_sf.closes}
        finally:
            release.set()
            thread.join(2)
            assert not thread.is_alive(), "fixture did not clean up"
    return {**observation, "result_after_fixture_release": result,
            "sf_closes_after_fixture_release": fake_sf.closes, "messages": messages,
            "selfplay_threads": 2, "control_driver_threads": 1}


@pytest.mark.parametrize("draws", [[0.1, 0.9], [0.9, 0.1], [0.1, 0.1], [0.9, 0.9]])
@pytest.mark.parametrize("mcts", ["puct", "gumbel"])
def test_real_network_dispatch_honors_each_playout_budget(draws, mcts):
    row = network_budget_case("regression", draws, mcts)
    if mcts == "puct" and len(set(row["expected_per_game_budget"])) > 1:
        expected = [
            {"runner": "puct", "boards": 1, "cfg_simulations": 128, "per_game_simulations": None},
            {"runner": "gumbel", "boards": 1, "cfg_simulations": 8, "per_game_simulations": [8]},
        ] if draws[0] < 0.5 else [
            {"runner": "gumbel", "boards": 1, "cfg_simulations": 8, "per_game_simulations": [8]},
            {"runner": "puct", "boards": 1, "cfg_simulations": 128, "per_game_simulations": None},
        ]
        assert row["calls"] == expected
    else:
        assert len(row["calls"]) == 1
        call = row["calls"][0]
        if call["runner"] == "gumbel":
            assert call["per_game_simulations"] == row["expected_per_game_budget"]
        else:
            assert call["cfg_simulations"] == 128
    assert row["record_has_policy"] == [d < 0.5 for d in draws]
    assert row["records"] == [1, 1]
    assert row["board_plies"] == [1, 1]


@pytest.mark.parametrize("failed_tid", [0, 1])
def test_thread_failure_stops_peers_and_reaches_sf_recovery(failed_tid):
    row = worker_failure_case(failed_tid)
    assert row["stop_latch_after_failure"]
    assert row["driver_done_before_fixture_release"]
    assert row["healthy_thread_done_before_fixture_release"]
    assert row["sf_closed_before_fixture_release"] == 1
    assert row["result_after_fixture_release"] == {"returned_none": True}
    assert row["sf_closes_after_fixture_release"] == 1
