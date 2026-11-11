from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from chess_anti_engine.tune import distributed_runtime as runtime
from chess_anti_engine.tune import harness
from chess_anti_engine.tune import trainable_phases as phases
from chess_anti_engine.tune.trainable_config_ops import _reload_yaml_into_config
from chess_anti_engine.tune.trial_config import DifficultyState, TrialConfig
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults
from chess_anti_engine.utils.sequential_selfplay import STOP_BEFORE_TRAINING_KEY as KEY


def test_yaml_to_trial_and_restart_only_reload(tmp_path: Path) -> None:
    parsed = flatten_run_config_defaults({'tune': {KEY: True}})
    assert TrialConfig.from_dict(parsed).distributed_stop_selfplay_before_training is True
    assert TrialConfig.from_dict({}).distributed_stop_selfplay_before_training is False
    yaml_path = tmp_path / 'config.yaml'
    yaml_path.write_text(f'tune:\n  {KEY}: true\n')
    config = {KEY: False}
    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert config[KEY] is False
    _reload_yaml_into_config(config, str(yaml_path), live_reload=False)
    assert config[KEY] is False
    # Topology is launch-fixed: a new config, rather than a live overlay, adopts it.
    assert TrialConfig.from_dict(parsed).distributed_stop_selfplay_before_training is True


@pytest.mark.parametrize('value', ['false', 'true', 0, 1])
def test_nonboolean_mode_is_rejected(value: Any) -> None:
    with pytest.raises(ValueError, match='boolean'):
        flatten_run_config_defaults({'tune': {KEY: value}})
    with pytest.raises(ValueError, match='boolean'):
        TrialConfig.from_dict({KEY: value})


@pytest.mark.parametrize('override', [
    {'distributed_inference_shared_broker': True},
    {'distributed_async_test_eval': True},
    {'distributed_server_host': '0.0.0.0'},
    {'distributed_server_host': '192.168.1.2'},
    {'distributed_server_url': 'http://other-host:9000'},
    {'distributed_server_public_url': 'https://example.com'},
    {'distributed_workers_per_trial': 0},
])
def test_unowned_or_overlapping_topology_is_rejected(override: dict) -> None:
    with pytest.raises(ValueError, match='requires'):
        TrialConfig.from_dict({KEY: True, **override})


def test_driver_refuses_remote_endpoint_before_launching_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_launch(**_kwargs: Any) -> None:
        pytest.fail('invalid sequential topology reached sidecar launch')
    monkeypatch.setattr(harness, '_launch_distributed_server', unexpected_launch)
    with pytest.raises(ValueError, match='loopback server host'):
        harness.run_tune(base_config={KEY: True, 'distributed_server_host': '0.0.0.0'},
                         work_dir=tmp_path, num_samples=1)


def test_barrier_waits_for_real_worker_broker_and_stamped_engine_exit() -> None:
    processes = [subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
                 for _ in range(2)]
    processes.append(subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(60)'],
        env=os.environ | {runtime.CAE_ENGINE_MARKER_ENV: str(processes[0].pid)},
    ))
    workers = [processes[0]]
    broker: list[subprocess.Popen[bytes] | None] = [processes[1]]
    try:
        runtime._stop_local_selfplay_for_training(
            worker_procs=workers, broker_proc_box=broker, expected_workers=1,
        )
        assert all(process.poll() is not None for process in processes)
        assert workers == []
        assert broker == [None]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize('stuck', ['worker', 'broker'])
def test_unconfirmed_shutdown_keeps_handles_and_refuses_training(
    monkeypatch: pytest.MonkeyPatch, stuck: str,
) -> None:
    worker = SimpleNamespace(pid=10001, poll=lambda: None if stuck == 'worker' else 0)
    broker = SimpleNamespace(pid=10002, poll=lambda: None if stuck == 'broker' else 0)
    workers, box = [worker], [broker]
    monkeypatch.setattr(runtime, '_stop_worker_processes', lambda _procs: None)
    monkeypatch.setattr(runtime, '_stop_process', lambda _proc: None)
    with pytest.raises(RuntimeError, match=r'refusing training.*remain alive'):
        runtime._stop_local_selfplay_for_training(
            worker_procs=cast(Any, workers), broker_proc_box=cast(Any, box), expected_workers=1,
        )
    assert workers == [worker]
    assert box == [broker]


def test_missing_worker_handle_refuses_unverifiable_collection() -> None:
    with pytest.raises(RuntimeError, match='cannot verify ownership'):
        runtime._stop_local_selfplay_for_training(
            worker_procs=[], broker_proc_box=[None], expected_workers=1,
        )


def test_unreaped_engine_also_refuses_training(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = SimpleNamespace(pid=10001, poll=lambda: 0)
    workers = [worker]
    monkeypatch.setattr(runtime, '_stop_worker_processes', lambda _procs: None)
    monkeypatch.setattr(runtime, 'list_pids_with_env', lambda *_args: [10002])
    with pytest.raises(RuntimeError, match='owned Stockfish engines remain alive'):
        runtime._stop_local_selfplay_for_training(
            worker_procs=cast(Any, workers), broker_proc_box=[None], expected_workers=1,
        )
    assert workers == [worker]


@pytest.mark.parametrize('enabled', [False, True])
def test_collection_returns_only_after_barrier_and_recreates_next_iteration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, enabled: bool,
) -> None:
    """Exercise the real successful collection consumer for two iterations."""
    created = []

    def process() -> SimpleNamespace:
        proc = SimpleNamespace(pid=10001 + len(created), returncode=None)
        proc.poll = lambda: proc.returncode
        created.append(proc)
        return proc

    def ensure_workers(**kwargs: Any) -> list:
        return kwargs['procs'] or [process()]

    def ensure_broker(**kwargs: Any) -> SimpleNamespace:
        return kwargs['proc'] or process()

    def stop_workers(procs: list) -> None:
        for proc in procs:
            proc.returncode = 0

    def stop_broker(proc: SimpleNamespace | None) -> None:
        if proc is not None:
            proc.returncode = 0

    monkeypatch.setattr(phases, '_ensure_distributed_workers', ensure_workers)
    monkeypatch.setattr(phases, '_ensure_inference_broker', ensure_broker)
    monkeypatch.setattr(phases, '_publish_iteration_model', lambda **_kwargs: 'donor')
    monkeypatch.setattr(phases, '_export_selfplay_shards_for_siblings', lambda **_kwargs: None)
    monkeypatch.setattr(phases, '_maybe_share_cross_trial_replay', lambda **_kwargs: {})
    monkeypatch.setattr(runtime, '_stop_worker_processes', stop_workers)
    monkeypatch.setattr(runtime, '_stop_process', stop_broker)
    summary = runtime._empty_ingest_summary()
    summary.update(matching_games=1, matching_positions=2, positions_replay_added=2)
    monkeypatch.setattr(phases, '_ingest_distributed_selfplay', lambda **_kwargs: summary)
    buf = SimpleNamespace(capacity=16, _snapshot_shards=list, flush=lambda: None)
    tc = TrialConfig.from_dict({KEY: enabled, 'games_per_iter': 1, 'replay_window_max': 16,
                               'replay_window_growth': 0, 'replay_window_growth_frac': 0.0})
    workers: list[Any] = []
    broker: list[Any] = [None]
    for iteration in (1, 2):
        result, _, _ = phases._run_selfplay_phase(
            tc=tc, config={KEY: enabled}, trainer=None, model_cfg=None,
            buf=buf, holdout_buf=None, holdout_frozen=False, rng=np.random.default_rng(0),
            distributed_dirs={'publish_dir': tmp_path / 'publish', 'inbox_dir': tmp_path / 'inbox',
                              'processed_dir': tmp_path / 'processed'},
            distributed_server_root=tmp_path, distributed_worker_procs=workers,
            broker_proc_box=broker, prev_published_model_sha='',
            ds=DifficultyState(wdl_regret=0.2, sf_nodes=1000), sims=1,
            iteration_idx=iteration, iteration_zero_based=iteration, trial_id='test',
            trial_dir=tmp_path, selfplay_shards_dir=tmp_path / 'siblings',
            replay_shard_dir=tmp_path / 'replay', current_window=16, in_salvage_startup_grace=False,
        )
        assert result.should_retry is False
        if enabled:
            assert workers == []
            assert broker == [None]
            assert all(proc.poll() == 0 for proc in created)
            assert len(created) == iteration * 2
        else:
            assert len(workers) == 1
            assert broker[0] is not None
            assert all(proc.poll() is None for proc in created)
            assert len(created) == 2
