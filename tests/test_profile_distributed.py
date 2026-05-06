from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import yaml

from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults


def _load_profile_module() -> Any:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "profile_distributed.py"
    spec = importlib.util.spec_from_file_location("profile_distributed_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_base_config_exists() -> None:
    module = _load_profile_module()

    assert module.BASE_CONFIG.exists()


def test_make_config_overrides_nested_sections(tmp_path: Path) -> None:
    module = _load_profile_module()

    base_config = {
        "work_dir": "runs/base",
        "selfplay": {
            "selfplay_batch": 64,
            "games_per_iter": 150,
            "games_per_iter_start": 32,
        },
        "tune": {
            "distributed_workers_per_trial": 2,
            "distributed_worker_sf_workers": 2,
            "max_concurrent_trials": 4,
            "num_samples": 6,
            "distributed_server_root_override": "/tmp/base_server",
            "tune_replay_root_override": "/tmp/base_replay",
        },
    }
    base_config_path = tmp_path / "base.yaml"
    base_config_path.write_text(yaml.safe_dump(base_config), encoding="utf-8")

    module.BASE_CONFIG = base_config_path
    module.WORK_DIR = tmp_path / "runs"
    module.SERVER_DIR = tmp_path / "server"
    module.REPLAY_DIR = tmp_path / "replay"

    out = module.make_config(workers=3, sf_workers=1, batch=24)
    cfg = yaml.safe_load(out.read_text(encoding="utf-8"))
    flat = flatten_run_config_defaults(cfg)

    assert cfg["selfplay"]["selfplay_batch"] == 24
    assert cfg["tune"]["distributed_workers_per_trial"] == 3
    assert cfg["tune"]["distributed_worker_sf_workers"] == 1
    assert cfg["tune"]["distributed_server_root_override"] == str(module.SERVER_DIR)
    assert cfg["tune"]["tune_replay_root_override"] == str(module.REPLAY_DIR)
    assert flat["selfplay_batch"] == 24
    assert flat["distributed_workers_per_trial"] == 3
    assert flat["distributed_worker_sf_workers"] == 1
    assert flat["distributed_server_root_override"] == str(module.SERVER_DIR)
    assert flat["tune_replay_root_override"] == str(module.REPLAY_DIR)


def test_cleanup_does_not_broad_kill_chess_processes(tmp_path: Path, monkeypatch) -> None:
    module = _load_profile_module()
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)

    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module.cleanup(
        work_dir=tmp_path / "work",
        server_dir=tmp_path / "server",
        replay_dir=tmp_path / "replay",
        stop_ray=True,
    )

    assert calls == [["ray", "stop", "--force"]]
    assert not any(isinstance(cmd, str) and "chess_anti_engine" in cmd for cmd in calls)
    assert (tmp_path / "server").is_dir()
    assert (tmp_path / "replay").is_dir()


def test_flatten_run_config_defaults_passes_shuffle_balance_knobs() -> None:
    cfg = {
        "tune": {
            "shuffle_draw_cap_frac": 0.75,
            "shuffle_wl_max_ratio": 1.2,
        }
    }

    flat = flatten_run_config_defaults(cfg)

    assert flat["shuffle_draw_cap_frac"] == 0.75
    assert flat["shuffle_wl_max_ratio"] == 1.2


def test_flatten_stockfish_new_style_flat_keys() -> None:
    """New-style YAML: keys inside stockfish: match flat config names directly."""
    cfg = {
        "stockfish": {
            "stockfish_path": "/usr/bin/stockfish",
            "sf_nodes": 5000,
            "sf_pid_ema_alpha": 0.50,
            "sf_pid_wdl_regret_max": 0.9,
        }
    }
    flat = flatten_run_config_defaults(cfg)
    assert flat["stockfish_path"] == "/usr/bin/stockfish"
    assert flat["sf_nodes"] == 5000
    assert flat["sf_pid_ema_alpha"] == 0.50
    assert flat["sf_pid_wdl_regret_max"] == 0.9


def test_flatten_stockfish_new_key_overrides_legacy() -> None:
    """When both old and new key names are present, new-style wins."""
    cfg = {
        "stockfish": {
            "nodes": 1000,       # legacy
            "sf_nodes": 5000,    # new style — should win
            "pid_ema_alpha": 0.03,  # legacy
            "sf_pid_ema_alpha": 0.50,  # new style — should win
        }
    }
    flat = flatten_run_config_defaults(cfg)
    assert flat["sf_nodes"] == 5000
    assert flat["sf_pid_ema_alpha"] == 0.50
