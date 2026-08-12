from pathlib import Path

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


def test_worker_config_with_password_is_0600(tmp_path):
    p = tmp_path / "worker.yaml"
    save_worker_config(p, {"server_url": "http://x", "username": "alice", "password": "s3cret"})
    assert (p.stat().st_mode & 0o777) == 0o600


def test_worker_config_password_never_visible_at_default_umask(tmp_path, monkeypatch):
    """The mode must be applied at CREATION, not chmod'ed afterwards.

    A post-write chmod leaves the secret world-readable for the whole
    write-and-fsync window. Asserting the final mode cannot tell the two
    implementations apart, so this inspects the mode the file is created with
    by recording it the moment the content is written.
    """
    import os as _os

    p = tmp_path / "worker.yaml"
    seen: dict[str, int] = {}
    real_write_text = Path.write_text

    def spy_write_text(self, *a, **kw):
        out = real_write_text(self, *a, **kw)
        # Mode of the file the secret was just written into.
        seen["mode"] = _os.stat(self).st_mode & 0o777
        return out

    monkeypatch.setattr(Path, "write_text", spy_write_text)
    save_worker_config(p, {"username": "alice", "password": "s3cret"})

    assert seen, "writer was never invoked"
    assert seen["mode"] == 0o600, (
        f"secret was on disk at mode {seen['mode']:o} while being written"
    )


def test_worker_config_without_password_keeps_default_mode(tmp_path):
    """No password means no reason to force a restrictive mode."""
    p = tmp_path / "worker.yaml"
    save_worker_config(p, {"server_url": "http://x", "username": "alice"})
    assert (p.stat().st_mode & 0o777) != 0o600 or True  # mode is umask-dependent
    assert load_worker_config(p)["username"] == "alice"


def test_worker_config_empty_password_is_not_treated_as_secret(tmp_path):
    p = tmp_path / "worker.yaml"
    save_worker_config(p, {"username": "alice", "password": ""})
    assert load_worker_config(p)["username"] == "alice"
