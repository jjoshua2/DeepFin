"""Live node floor/ceiling on a completed iteration with no curriculum W/D/L.

Production ``selfplay_fraction`` is 0.50. Selfplay games increment
``matching_games`` and do not increment ``total_w/d/l``
(``trainable_metrics._curriculum_winrate_raw_or_none``). The retry continue
in ``train_trial`` fires only when ``matching_games <= 0``, so an iteration
that finished selfplay games and zero curriculum games still calls
``_run_pid_and_eval``. PR #73 re-clamps ``sf_pid_min_nodes`` /
``sf_pid_max_nodes`` every iteration. PR #83/#85 hold the EMA, lever
history, and pooled sample when the curriculum sample is too small to
step; they do not hold the node-bound clamp.

``train_trial`` passes ``sf=None`` (workers own Stockfish). The next
iteration's budget is ``DifficultyState.from_pid`` →
``build_recommended_worker(sf_nodes=ds.sf_nodes)`` → the worker's
``set_nodes`` → ``_eff_sf_nodes`` on the curriculum move. ``_play_batch_kwargs``
is the trainer-side play-config consumer of that same ``DifficultyState``
(regret limit); the node budget rides beside it, not inside ``GameConfig``.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np

from chess_anti_engine.model import ModelConfig
from chess_anti_engine.selfplay.stockfish_turn import _eff_sf_nodes
from chess_anti_engine.selfplay.state import SelfplayState
from chess_anti_engine.stockfish.pid import DifficultyPID
from chess_anti_engine.stockfish.uci import StockfishUCI
from chess_anti_engine.tune.distributed_runtime import build_recommended_worker
from chess_anti_engine.tune.trainable_config_ops import _play_batch_kwargs
from chess_anti_engine.tune.trainable_phases import _run_pid_and_eval
from chess_anti_engine.tune.trial_config import DifficultyState, SelfplayResult, TrialConfig


class _NodeSpy:
    """Records ``set_nodes`` the way the trial's local engine would."""

    def __init__(self, nodes: int) -> None:
        self.nodes = int(nodes)
        self.calls: list[int] = []

    def set_nodes(self, n: int) -> None:
        self.calls.append(int(n))
        self.nodes = int(n)


def _pid() -> DifficultyPID:
    # Floor 50, current nodes 5000, ceiling well above the edited floor.
    # Regret stage is open, so a later observe cannot move the nodes lever.
    pid = DifficultyPID(
        initial_nodes=5000,
        min_nodes=50,
        max_nodes=100_000,
        target_winrate=0.50,
        ema_alpha=0.25,
        min_games_between_adjust=1,
        min_games_for_adjust=30,
        initial_wdl_regret=0.30,
        wdl_regret_min=0.01,
        wdl_regret_max=1.0,
        wdl_regret_stage_end=0.05,
    )
    pid._held_wins = 4
    pid._held_draws = 1
    pid._held_losses = 2
    pid._games_since_adjust = 7
    return pid


def _tc() -> TrialConfig:
    return TrialConfig.from_dict({
        "device": "cpu",
        "sf_nodes": 5000,
        "sf_pid_target_winrate": 0.50,
        "sf_pid_wdl_regret_max": 1.0,
    })


def _played_nodes(ds: DifficultyState, tc: TrialConfig) -> int:
    """Follow the next iteration's difficulty into the curriculum move budget.

    Publish uses ``ds.sf_nodes`` (``_publish_iteration_model``). The worker
    writes that onto the engine, and ``_eff_sf_nodes(for_move=True)`` reads
    ``state.base_nodes``. ``_play_batch_kwargs`` supplies the ``GameConfig``
    the same play call uses; with ``sf_move_nodes`` 0 the move budget is the
    published node count.
    """
    kw = _play_batch_kwargs(tc, ds)
    assert kw["opponent"].wdl_regret_limit == ds.wdl_regret
    reco = build_recommended_worker(
        config={},
        model_cfg=ModelConfig(),
        sf_nodes=int(ds.sf_nodes),
        mcts_simulations=int(tc.mcts_simulations),
        wdl_regret=float(ds.wdl_regret),
    )
    published_nodes = reco["sf_nodes"]
    assert isinstance(published_nodes, int)
    assert published_nodes == int(ds.sf_nodes)
    assert reco["opponent_wdl_regret_limit"] == ds.wdl_regret
    state = cast(SelfplayState, cast(object, SimpleNamespace(
        base_nodes=published_nodes,
        game=kw["game"],
        opening=kw["opening"],
        last_net_full=np.ones(1, dtype=np.int8),
    )))
    played = _eff_sf_nodes(state, 0, for_move=True)
    assert played is not None
    return int(played)


def _snapshot(pid: DifficultyPID) -> dict[str, object]:
    return {
        "ema": float(pid.ema_winrate),
        "regret": float(pid.wdl_regret),
        "held": (pid._held_wins, pid._held_draws, pid._held_losses),
        "games_since": int(pid._games_since_adjust),
        "nodes_hist": list(pid.nodes_lever.history),
        "regret_hist": list(pid.regret_lever.history),
        "slope": pid.last_regret_fit_slope,
        "metric_min": int(pid.metric_min_nodes),
        "metric_max": int(pid.metric_max_nodes),
    }


def test_zero_curriculum_outcome_applies_live_node_floor() -> None:
    """Floor 50 → 20000 at nodes 5000 must reach the next iteration's play.

    The curriculum W/D/L total is zero. Observation state stays put.
    """
    pid = _pid()
    tc = _tc()
    spy = _NodeSpy(5000)
    before = _snapshot(pid)
    ds_now = DifficultyState.from_pid(pid, spy, tc)
    assert ds_now.sf_nodes == 5000

    result = _run_pid_and_eval(
        tc=tc,
        config={"sf_pid_min_nodes": 20_000},
        pid=pid,
        sf=cast(StockfishUCI, cast(object, spy)),
        sp_result=SelfplayResult(
            total_w=0, total_d=0, total_l=0,
            matching_games=8,
            total_selfplay_games=8,
            total_game_plies=160,
        ),
        opp_strength_ema=0.4,
        opp_ema_alpha=0.2,
        ds=ds_now,
    )

    assert result.pid_update is None
    assert pid.min_nodes == 20_000
    assert pid.nodes == 20_000
    assert spy.calls == [20_000]
    assert spy.nodes == 20_000
    assert result.sf_nodes_next == 20_000
    ds_next = DifficultyState.from_pid(pid, spy, tc)
    assert ds_next.sf_nodes == 20_000
    assert ds_next.wdl_regret == before["regret"]
    assert _played_nodes(ds_next, tc) == 20_000
    after = _snapshot(pid)
    assert after["ema"] == before["ema"]
    assert after["regret"] == before["regret"]
    assert after["held"] == before["held"]
    assert after["games_since"] == before["games_since"]
    assert after["nodes_hist"] == before["nodes_hist"]
    assert after["regret_hist"] == before["regret_hist"]
    assert after["slope"] == before["slope"]
    assert after["metric_min"] == 50
    assert after["metric_max"] == before["metric_max"]


def test_zero_curriculum_outcome_applies_live_node_ceiling() -> None:
    """Dropping the ceiling below the current count clamps on a zero sample.

    Same gate as the floor. Lowering ``sf_pid_max_nodes`` is how an operator
    pulls a starved curriculum back inside the iteration window; that edit
    has to apply on the iteration that finished no curriculum games.
    """
    pid = _pid()
    pid.nodes = 80_000
    assert pid.nodes == 80_000
    tc = _tc()
    spy = _NodeSpy(80_000)
    before = _snapshot(pid)
    ds_now = DifficultyState.from_pid(pid, spy, tc)

    result = _run_pid_and_eval(
        tc=tc,
        config={"sf_pid_max_nodes": 20_000},
        pid=pid,
        sf=cast(StockfishUCI, cast(object, spy)),
        sp_result=SelfplayResult(total_w=0, total_d=0, total_l=0, matching_games=8, total_selfplay_games=8),
        opp_strength_ema=0.4,
        opp_ema_alpha=0.2,
        ds=ds_now,
    )

    assert result.pid_update is None
    assert pid.max_nodes == 20_000
    assert pid.nodes == 20_000
    assert spy.calls == [20_000]
    assert result.sf_nodes_next == 20_000
    ds_next = DifficultyState.from_pid(pid, spy, tc)
    assert _played_nodes(ds_next, tc) == 20_000
    after = _snapshot(pid)
    assert after["ema"] == before["ema"]
    assert after["held"] == before["held"]
    assert after["regret_hist"] == before["regret_hist"]
    assert after["nodes_hist"] == before["nodes_hist"]
    assert after["metric_max"] == 100_000


def test_positive_small_sample_clamps_floor_and_holds_levers() -> None:
    """A 2-game curriculum sample still clamps the floor and still holds.

    PR #83: below ``min_games_for_adjust`` the EMA and the held pool update
    and the levers do not step. That path already entered the old gate, so
    this is the control that the zero-outcome fix must not disturb.
    """
    pid = _pid()
    pid._held_wins = pid._held_draws = pid._held_losses = 0
    pid._games_since_adjust = 0
    tc = _tc()
    spy = _NodeSpy(5000)
    regret_before = float(pid.wdl_regret)
    ema_before = float(pid.ema_winrate)
    nodes_hist = list(pid.nodes_lever.history)
    regret_hist = list(pid.regret_lever.history)
    ds_now = DifficultyState.from_pid(pid, spy, tc)

    result = _run_pid_and_eval(
        tc=tc,
        config={"sf_pid_min_nodes": 20_000},
        pid=pid,
        sf=cast(StockfishUCI, cast(object, spy)),
        sp_result=SelfplayResult(
            total_w=2, total_d=0, total_l=0,
            matching_games=2,
            total_curriculum_games=2,
            total_game_plies=40,
            total_plies_win=40,
        ),
        opp_strength_ema=0.4,
        opp_ema_alpha=0.2,
        ds=ds_now,
    )

    assert result.pid_update is not None
    assert pid.min_nodes == 20_000
    assert pid.nodes == 20_000
    assert spy.calls == [20_000]
    assert result.sf_nodes_next == 20_000
    assert pid.wdl_regret == regret_before
    assert pid.ema_winrate != ema_before
    assert pid._held_wins == 2
    assert pid._held_draws == 0
    assert pid._held_losses == 0
    assert pid._games_since_adjust == 2
    assert list(pid.nodes_lever.history) == nodes_hist
    assert list(pid.regret_lever.history) == regret_hist
    assert pid.metric_min_nodes == 50
    ds_next = DifficultyState.from_pid(pid, spy, tc)
    assert _played_nodes(ds_next, tc) == 20_000
