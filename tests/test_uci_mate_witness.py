"""Mate-stop selection must retain its witness, not merely a winning outcome."""
from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import chess
import numpy as np
import pytest

import chess_anti_engine.uci.search as search
from chess_anti_engine.moves import move_to_index

if TYPE_CHECKING:
    from chess_anti_engine.mcts._mcts_tree import MCTSTree


@pytest.mark.parametrize("other_mate", [3, None])
@pytest.mark.parametrize("gumbel", [False, True])
def test_mate_witness_overrides_competing_win(
    monkeypatch: pytest.MonkeyPatch, other_mate: int | None, gumbel: bool,
) -> None:
    board = chess.Board()
    short = int(move_to_index(chess.Move.from_uci("a2a3"), board))
    other = int(move_to_index(chess.Move.from_uci("b2b3"), board))

    class Tree:
        def get_solved_status(self, node):
            return 1 if node == 0 else -1

        def get_children_visits(self, node):
            return np.array([short, other]), np.array([5, 100])

        def find_child(self, node, action):
            return action

        def node_count(self):
            return 3

        def set_cpuct_scaling(self, factor, base):
            pass

    worker = object.__new__(search.SearchWorker)
    worker._tree = cast("MCTSTree", cast(object, Tree()))
    worker._root_id = 0
    worker._mate_check_key = None
    worker._mate_witness_action = None
    worker._last_gumbel_action_idx = other if gumbel else None
    distances = {short: 2, other: other_mate}
    monkeypatch.setattr(search, "_proven_line_mate", lambda tree, root, board, action, walk=None: distances[action])
    monkeypatch.setattr(search, "_pv_from_root_action", lambda tree, root, action: [action])

    def result(allowed):
        return worker._build_final_search_result(
            board=board, total_nodes=105, last_value=1.0,
            tb_probe=None, allowed_root_indices=allowed,
        )

    # Ordinary Gumbel/visit selection is unchanged when no mate stop applies.
    assert result(None).bestmove_uci == "b2b3"
    assert worker._proven_mate_within(board, 2, None) == 2
    assert worker._mate_witness_action == short
    final = result({worker._mate_witness_action})
    assert final.bestmove_uci == "a2a3"
    assert final.score_mate == 2
    assert final.pv[0] == "a2a3"

    # A new search cannot reuse the witness or escape its searchmoves support.
    worker._mate_check_key = None
    assert worker._proven_mate_within(board, 2, {other}) is None
    assert worker._mate_witness_action is None
    worker._mate_check_key = None
    assert worker._proven_mate_within(board, 1, None) is None
    assert worker._mate_witness_action is None

    # Exercise the real run loop: only expensive search/clock helpers are
    # stubbed. Do NOT manually narrow allowed_root_indices in this regression.
    worker._tree_fen = board.fen()
    worker._tb_probe = None
    worker._rpg_pool = None
    monkeypatch.setattr(worker, "_cfg", SimpleNamespace(cpuct_factor=0.0, cpuct_base=38739.0), raising=False)
    for name in (
        "_resolve_pending_compaction", "_reset_batch_timing",
        "_ensure_root_eval_cached", "_pre_expand_root_for_pool",
        "_sync_tree_growth_step", "_record_batch_ms", "_record_growth_rate",
    ):
        monkeypatch.setattr(worker, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_try_tb_shortcut", lambda *args, **kwargs: None)
    monkeypatch.setattr(worker, "_tree_node_count", lambda: 3)
    monkeypatch.setattr(worker, "_chunk_budget", lambda allowed: 1)
    monkeypatch.setattr(worker, "_time_capped_chunk", lambda chunk, *args: chunk)
    monkeypatch.setattr(worker, "_is_shared_tree_path", lambda allowed: False)
    monkeypatch.setattr(worker, "_tree_footprint_bytes", lambda: 0)
    monkeypatch.setattr(worker, "_root_visit_snapshot", lambda: None)
    monkeypatch.setattr(worker, "_maybe_emit_pv_info", lambda **kwargs: ([], 0, 0))
    monkeypatch.setattr(worker, "_clock_time_margin_ms", lambda *args, **kwargs: 0)
    monkeypatch.setattr(worker, "_abort_ready", lambda *args: False)
    monkeypatch.setattr(
        worker, "_should_stop_search",
        lambda **kwargs: "nodes" if kwargs["total_nodes"] >= 2 else None,
    )

    def chunk(*args, **kwargs):
        worker._last_gumbel_action_idx = other if gumbel else None
        return 1.0, 1

    monkeypatch.setattr(worker, "_run_one_chunk", chunk)
    deadline = cast(search.Deadline, SimpleNamespace(
        remaining_ms=lambda: 10000, elapsed_ms=lambda: 0, expired=lambda: False,
    ))
    final = worker.run(
        board, stop_event=threading.Event(), deadline=deadline, max_nodes=2,
        mate_moves=2, allow_terminal_shortcuts=False,
    )
    assert worker.last_stop_reason == "mate"
    assert final.nodes == 1
    assert final.bestmove_uci == "a2a3"
    assert final.score_mate == 2
    # The next unrestricted ordinary go retains visit/Gumbel behavior.
    ordinary = worker.run(
        board, stop_event=threading.Event(), deadline=deadline, max_nodes=2,
        allow_terminal_shortcuts=False,
    )
    assert ordinary.bestmove_uci == "b2b3"
