"""The move clock starts at ``go`` receipt, so pre-search work is charged.

Tree sync (``_sync_tree_root``: cross-move reuse, and before #1264's deferral
the subtree compaction) ran inside ``_handle_go`` BEFORE the search thread built
its ``Deadline``, so any time it took was invisible to the time manager.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import chess
import numpy as np

from chess_anti_engine.mcts.gumbel import GumbelConfig
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.uci.engine import Engine
from chess_anti_engine.uci.protocol import parse_command
from chess_anti_engine.uci.search import SearchResult, SearchWorker


class _Zero:
    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        del relations
        return np.zeros((len(x), POLICY_SIZE), np.float32), np.zeros((len(x), 3), np.float32)


class _ClockWorker(SearchWorker):
    def __init__(self) -> None:
        super().__init__(_Zero(), device="cpu", n_walkers=1, chunk_sims=8,
                         gumbel_cfg=GumbelConfig(topk=4, add_noise=False, temperature=0.0))
        self.elapsed_at_entry: list[int] = []
        self.ran = threading.Event()

    def run(self, board: chess.Board, **kwargs: Any) -> SearchResult:
        self.elapsed_at_entry.append(kwargs["deadline"].elapsed_ms())
        try:
            return super().run(board, **kwargs)
        finally:
            self.ran.set()


def test_pre_search_sync_time_is_charged_to_the_move_deadline(monkeypatch) -> None:
    worker = _ClockWorker()
    engine = Engine(worker)
    real_sync = engine._sync_tree_root

    def slow_sync(target_moves: list[chess.Move]) -> None:
        time.sleep(0.3)
        real_sync(target_moves)

    monkeypatch.setattr(engine, "_sync_tree_root", slow_sync)
    try:
        for line in ("position startpos", "go movetime 2000"):
            cmd = parse_command(line)
            assert cmd is not None
            engine.dispatch(cmd)
        assert worker.ran.wait(10)
        engine._wait_for_search()
    finally:
        engine.close()
    assert worker.elapsed_at_entry
    assert worker.elapsed_at_entry[0] >= 250, worker.elapsed_at_entry
