"""A requested move budget must reach a route that can keep labels separate."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from chess_anti_engine.model import ModelConfig
from chess_anti_engine.selfplay import manager
from chess_anti_engine.selfplay.config import GameConfig
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishUCI
from chess_anti_engine.tune import distributed_runtime as runtime
from chess_anti_engine.worker import WorkerSession


class _ReachedSetup(Exception):
    """Stop at the next real setup boundary without creating games or engines."""


@pytest.mark.parametrize('move_nodes', [100, 5000])
def test_play_batch_rejects_single_engine_before_state_or_engine_work(
    monkeypatch: pytest.MonkeyPatch, move_nodes: int,
) -> None:
    def unexpected_state(**_kwargs: Any) -> None:
        pytest.fail('unsupported move budget reached SelfplayState creation')

    monkeypatch.setattr(manager.SelfplayState, 'create', unexpected_state)
    # No constructor is called: the object represents the actual non-pool route
    # and cannot start an engine or issue a query during this regression test.
    engine = object.__new__(StockfishUCI)
    with pytest.raises(ValueError, match='sf_move_nodes > 0 requires StockfishPool'):
        manager.play_batch(
            None, device='cpu', rng=np.random.default_rng(0), stockfish=engine,
            games=1, game=GameConfig(sf_move_nodes=move_nodes),
        )


@pytest.mark.parametrize(('pooled', 'move_nodes'), [(False, 0), (True, 0), (True, 100)])
def test_play_batch_preserves_default_and_pool_routes(
    monkeypatch: pytest.MonkeyPatch, pooled: bool, move_nodes: int,
) -> None:
    seen: list[dict] = []

    def reached_state(**kwargs: Any) -> None:
        seen.append(kwargs)
        raise _ReachedSetup

    monkeypatch.setattr(manager.SelfplayState, 'create', reached_state)
    engine = object.__new__(StockfishPool if pooled else StockfishUCI)
    game = GameConfig(sf_move_nodes=move_nodes)
    with pytest.raises(_ReachedSetup):
        manager.play_batch(
            None, device='cpu', rng=np.random.default_rng(0), stockfish=engine,
            games=1, game=game,
        )
    assert len(seen) == 1
    assert seen[0]['stockfish'] is engine
    assert seen[0]['game'] is game


@pytest.mark.parametrize(('driver_workers', 'worker_workers', 'move_nodes', 'refused'), [
    (8, 1, 100, True),
    (8, 1, 5000, True),  # Even equality to the initial base is not durable.
    (8, 1, 0, False),
    (1, 2, 100, False),
])
def test_distributed_worker_uses_effective_cli_topology_before_any_bringup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver_workers: int,
    worker_workers: int, move_nodes: int, refused: bool,
) -> None:
    monkeypatch.setattr(runtime, '_resolve_distributed_worker_auth',
                        lambda **_kwargs: ('test-worker', tmp_path / 'unused-password'))
    config = {
        'sf_workers': driver_workers, 'distributed_worker_sf_workers': worker_workers,
        'sf_move_nodes': move_nodes, 'distributed_server_url': 'http://127.0.0.1:9000',
        'distributed_server_root': str(tmp_path / 'server'), 'stockfish_path': 'not-started',
    }
    command = runtime._build_distributed_worker_cmd(
        config=config, trial_root=tmp_path / 'worker', trial_id='trial',
        worker_index=0, worker_log=tmp_path / 'worker.log',
    )
    effective_workers = int(command[command.index('--sf-workers') + 1])
    assert effective_workers == worker_workers
    reco = runtime.build_recommended_worker(
        config=config, model_cfg=ModelConfig(), sf_nodes=5000, mcts_simulations=8,
    )
    assert reco['sf_move_nodes'] == move_nodes
    session = object.__new__(WorkerSession)
    session.args = SimpleNamespace(sf_workers=effective_workers)
    session._shutdown_requested = False
    reached: list[dict] = []

    def after_guard(_self: WorkerSession, incoming: dict) -> dict:
        reached.append(incoming)
        raise _ReachedSetup

    monkeypatch.setattr(WorkerSession, '_snapshot_reco', after_guard)
    expected = ValueError if refused else _ReachedSetup
    with pytest.raises(expected) as caught:
        session._run_selfplay({'recommended_worker': reco})
    if refused:
        assert 'sf_move_nodes > 0 requires StockfishPool' in str(caught.value)
        assert reached == []
    else:
        assert reached == [reco]


def test_worker_local_move_budget_override_is_also_checked() -> None:
    session = object.__new__(WorkerSession)
    session.args = SimpleNamespace(sf_workers=1, sf_move_nodes=100)
    session._shutdown_requested = False
    with pytest.raises(ValueError, match='sf_move_nodes > 0 requires StockfishPool'):
        session._run_selfplay({'recommended_worker': {'sf_move_nodes': 0}})


def test_invalid_move_budget_does_not_reach_an_existing_engine() -> None:
    session = object.__new__(WorkerSession)
    session.args = SimpleNamespace(sf_workers=1)
    session._shutdown_requested = False
    calls: list[int] = []
    session.sf = cast(Any, SimpleNamespace(set_nodes=calls.append))
    with pytest.raises(ValueError, match='sf_move_nodes > 0 requires StockfishPool'):
        session._run_selfplay({'recommended_worker': {'sf_move_nodes': 100, 'sf_nodes': 5000}})
    assert calls == []
