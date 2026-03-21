from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults


def _load_profile_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "profile_distributed.py"
    spec = importlib.util.spec_from_file_location("profile_distributed_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
