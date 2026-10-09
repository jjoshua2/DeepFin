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


def test_worker_config_without_password_uses_the_umask_default(tmp_path):
    """No password means no reason to force a restrictive mode.

    Asserted against the process umask rather than `!= 0o600`: the previous
    version of this test was `assert ... or True`, which is true for every
    input -- a gate that cannot fail, in a file about gates that must.
    """
    import os as _os

    umask = _os.umask(0o022)
    _os.umask(umask)

    p = tmp_path / "worker.yaml"
    save_worker_config(p, {"server_url": "http://x", "username": "alice"})
    assert (p.stat().st_mode & 0o777) == (0o666 & ~umask)
    assert load_worker_config(p)["username"] == "alice"


def test_worker_config_empty_password_is_not_treated_as_secret(tmp_path):
    p = tmp_path / "worker.yaml"
    save_worker_config(p, {"username": "alice", "password": ""})
    assert load_worker_config(p)["username"] == "alice"

def test_explicit_worker_cli_upload_settings_override_persisted_yaml():
    import argparse

    from chess_anti_engine.worker import _merge_cli_with_yaml_defaults

    args = argparse.Namespace(
        server_url="http://127.0.0.1:45453",
        allow_cleartext_http=False,
        trial_id=None,
        username="alice",
        stockfish_path="/tmp/stockfish",
        shared_cache_dir=None,
        password_file=None,
        self_update=False,
        stockfish_from_server=False,
        sf_workers=1,
        sf_nice=0,
        games_per_batch=2,
        upload_target_positions=123,
        upload_flush_seconds=4.5,
    )
    _merge_cli_with_yaml_defaults(
        args,
        {"upload_target_positions": 999, "upload_flush_seconds": 999.0},
    )
    assert args.upload_target_positions == 123
    assert args.upload_flush_seconds == 4.5


def test_worker_upload_yaml_values_fill_absent_cli_defaults():
    import argparse

    from chess_anti_engine.worker import _merge_cli_with_yaml_defaults

    args = argparse.Namespace(
        server_url="http://127.0.0.1:45453",
        allow_cleartext_http=False,
        trial_id=None,
        username="alice",
        stockfish_path="/tmp/stockfish",
        shared_cache_dir=None,
        password_file=None,
        self_update=False,
        stockfish_from_server=False,
        sf_workers=1,
        sf_nice=0,
        games_per_batch=2,
        upload_target_positions=None,
        upload_flush_seconds=None,
    )
    _merge_cli_with_yaml_defaults(
        args,
        {"upload_target_positions": "750", "upload_flush_seconds": "12.5"},
    )
    assert args.upload_target_positions == 750
    assert args.upload_flush_seconds == 12.5

    defaults = argparse.Namespace(**vars(args))
    defaults.upload_target_positions = None
    defaults.upload_flush_seconds = None
    _merge_cli_with_yaml_defaults(defaults, {})
    assert defaults.upload_target_positions == 500
    assert defaults.upload_flush_seconds == 60.0
