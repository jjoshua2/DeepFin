"""A STOP teardown must not recycle finished slots into seeds nobody plays.

``play_batch``'s STOP branch finalizes the games whose labels were still
running (continuous mode defers them). It used to finalize them with the same
recycling rule as the running loop, so every finished slot was immediately
re-opened by ``SelfplayState.recycle_slot`` -- which POPS the next seed off the
shared ``fen_dole_queue`` (or the SF-refute queue). The very next statement is
the teardown: ``suspend_inflight_games`` skips those fresh slots as
``empty_slot`` and the worker hands only the queue's REMAINDER to the next
session, so each such seed was taken and never played.

These tests drive the real ``play_batch`` + ``recycle_slot`` +
``suspend_inflight_games`` with an asynchronous Stockfish pool, so finished
games are genuinely deferred on their labels when the stop arrives.
"""
from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import Future
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.manager import CompletedGameBatch
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.resume import SuspendReport, suspend_inflight_games
from chess_anti_engine.selfplay.state import SelfplayState
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishResult
from tests.selfplay_helpers import FakeStockfish, UniformPolicyValueModel

FEN_WHITE = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2"
N_SEEDS = 64
N_SLOTS = 3


LABEL_NODES = 7777  # sf_label_nodes_floor: tells label queries from moves


class _LazyFuture(Future):
    """A label future that is never ``done()`` until someone demands it.

    The running loop only polls ``done()``, so a finished game stays deferred
    on its label for as long as it runs; finalize's flush calls ``result()``,
    which resolves it on the spot. No timing is involved.
    """

    def __init__(self, value: StockfishResult) -> None:
        super().__init__()
        self._value = value

    def result(self, timeout: float | None = None) -> StockfishResult:
        if not self.done():
            self.set_result(self._value)
        return super().result(timeout)


class _HeldLabelPool(StockfishPool):
    """Pool-shaped fake: moves resolve at once, labels only when demanded."""

    def __init__(self) -> None:  # pyright: ignore[reportMissingSuperCall]
        self.nodes = 1

    def submit(
        self, fen: str, *, nodes: int | None = None,
        syzygy_path: str | None = None, fresh: bool = False,
        searchmoves: Sequence[str] | None = None,
        root_fen: str | None = None,
        root_moves: Sequence[str] | None = None,
    ) -> Future[StockfishResult]:
        del syzygy_path, fresh, root_fen, root_moves
        board = chess.Board(fen)
        if searchmoves:
            move = chess.Move.from_uci(str(next(iter(searchmoves))))
        else:
            move = next(iter(board.legal_moves), chess.Move.null())
        result = StockfishResult(
            bestmove_uci=move.uci(),
            wdl=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            pvs=[],
        )
        if nodes == LABEL_NODES:
            return _LazyFuture(result)
        future: Future[StockfishResult] = Future()
        future.set_result(result)
        return future


def _opening(channel: str) -> OpeningConfig:
    if channel == "dole":
        return OpeningConfig(
            random_start_plies=0, opening_fen_list_path="dummy",
            opening_fen_dole_per_iter=1,
        )
    return OpeningConfig(
        random_start_plies=0, opening_fen_list_path="dummy",
        opening_fen_sf_refute_frac=1.0, opening_fen_sf_refute_plies=1,
    )


def _queues(channel: str) -> tuple[list[str], list[str]]:
    seeds = [FEN_WHITE] * N_SEEDS
    return (seeds, []) if channel == "dole" else ([], seeds)


def _finished_awaiting_finalize(state: SelfplayState) -> int:
    return sum(
        1 for i in range(state.batch_size)
        if state.done_arr[i] and not state.finalized_arr[i]
    )


@pytest.mark.parametrize("channel", ["dole", "sf_refute"])
def test_stop_teardown_does_not_consume_seeds_it_never_plays(
    tmp_path: Path, channel: str,
) -> None:
    """Every seed taken off a queue is either a delivered game or a persisted one.

    MUTATION: let the STOP branch recycle again (``recycle=True``) -- the
    finished slots are re-opened from the queue, ``suspend_inflight_games``
    reports them as empty slots, and ``consumed`` exceeds the games played.
    """
    states: list[SelfplayState] = []
    completed: list[CompletedGameBatch] = []
    reports: list[SuspendReport] = []
    at_stop: dict[str, int] = {}

    def _stop() -> bool:
        if not states:
            return False
        if "deferred" not in at_stop:
            deferred = _finished_awaiting_finalize(states[0])
            if deferred == 0:
                return False
            at_stop["deferred"] = deferred
            at_stop["completed"] = len(completed)
        return True

    def _on_suspend(state: SelfplayState) -> None:
        reports.append(suspend_inflight_games(
            state, out_dir=tmp_path / "resume", compat_fingerprint="fp",
            model_sha="m", model_step=0, trial_id="trial-0",
        ))

    dole, refute = _queues(channel)
    play_batch(
        UniformPolicyValueModel().eval(), device="cpu",
        rng=np.random.default_rng(3), stockfish=_HeldLabelPool(),
        games=N_SLOTS, target_games=0,
        stop_fn=_stop, on_state_ready=states.append,
        on_suspend=_on_suspend, on_game_complete=completed.append,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=_opening(channel),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(
            max_plies=2, selfplay_fraction=1.0, sf_label_nodes_floor=LABEL_NODES,
        ),
        fen_dole_queue=dole, fen_sf_refute_queue=refute,
    )

    queue = dole if channel == "dole" else refute
    consumed = N_SEEDS - len(queue)
    assert at_stop.get("deferred", 0) > 0, "stop must arrive while games await labels"
    # Every game started here must have drawn a seed, or the identity below
    # would be comparing seeded and unseeded games.
    assert queue, "seed queue exhausted; raise N_SEEDS"
    assert len(reports) == 1
    report = reports[0]
    # The deferred games are still finalized and delivered at STOP.
    assert len(completed) - at_stop["completed"] == at_stop["deferred"]
    assert all(cg.positions == len(cg.samples) > 0 for cg in completed)
    # And nothing was re-opened after them.
    assert report.empty_slots == 0, report.reasons
    assert consumed == len(completed) + report.persisted
    state = states[0]
    assert state.games_started == state.games_completed + report.persisted


def test_running_loop_still_recycles_finished_slots_from_the_queue() -> None:
    """Outside STOP, a finished slot is re-opened from the dole queue as before.

    MUTATION: disable recycling on the running-loop calls too -- no game past
    the first ``N_SLOTS`` is ever started and the completion target is never
    reached (the loop would spin), so ``stats.games`` and ``consumed`` fall.
    """
    completed: list[CompletedGameBatch] = []
    states: list[SelfplayState] = []
    dole = [FEN_WHITE] * N_SEEDS
    target = 3 * N_SLOTS
    _, stats = play_batch(
        UniformPolicyValueModel().eval(), device="cpu",
        rng=np.random.default_rng(4),
        stockfish=FakeStockfish([0.0, 1.0, 0.0]),
        games=N_SLOTS, target_games=0,
        stop_fn=lambda: len(completed) >= target or (
            bool(states) and states[0].games_started == N_SLOTS
            and len(completed) >= N_SLOTS
        ),
        on_state_ready=states.append, on_game_complete=completed.append,
        temp=TemperatureConfig(temperature=1.0),
        search=SearchConfig(simulations=1, playout_cap_fraction=1.0, fast_simulations=1),
        opening=_opening("dole"),
        diff_focus=DiffFocusConfig(enabled=False),
        game=GameConfig(max_plies=2, selfplay_fraction=1.0),
        fen_dole_queue=dole,
    )
    state = states[0]
    assert stats.games >= target
    # Synchronous labels finalize every game inside the running loop, so each
    # recycle there drew one seed: the queue drained exactly once per start.
    assert N_SEEDS - len(dole) == state.games_started
    assert state.games_started > N_SLOTS
