from chess_anti_engine.worker_config import load_worker_config, save_worker_config


def test_worker_config_roundtrip(tmp_path):
    p = tmp_path / "worker.yaml"
    cfg = {"server_url": "http://x", "username": "alice", "games_per_batch": 12}
    save_worker_config(p, cfg)
    out = load_worker_config(p)
    assert out["server_url"] == "http://x"
    assert out["username"] == "alice"
    assert int(out["games_per_batch"]) == 12


def test_worker_config_missing_file(tmp_path):
    p = tmp_path / "missing.yaml"
    out = load_worker_config(p)
    assert out == {}
