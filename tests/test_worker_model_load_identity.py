"""Model-load failure identity regressions without optional native imports."""
from __future__ import annotations

import ast
import dataclasses
import logging
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

import pytest


@dataclasses.dataclass
class _Config:
    use_gradient_checkpointing: bool = True


@pytest.fixture
def session(tmp_path: Path) -> Any:
    source = (Path(__file__).resolve().parents[1] / "chess_anti_engine/worker.py").read_text()
    tree = ast.parse(source)
    env: dict[str, Any] = {
        "dataclasses": dataclasses, "ModelConfig": _Config,
        "model_config_from_manifest_dict": lambda _d: _Config(),
        "nullcontext": nullcontext, "suppress": suppress,
        "_sha256_file": lambda _p: "new",
    }
    methods = {}
    for name in ("_sync_model", "_swap_model_from_manifest", "_periodic_manifest_poll", "_sync_assets"):
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        module = ast.Module(body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            node,
        ], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "<worker-methods>", "exec"), env)
        methods[name] = env[name]
    cls = type("ModelSyncSession", (), methods)
    s: Any = cls()
    s.model = object()
    s.model_cfg_active = _Config()
    s.model_sha = s.last_model_sha = "old"
    s.model_step = 2
    s.inference_client = None
    s.fixed_trial_id = "fixed"
    s.leased_trial_id = "trial"
    s._stop_selfplay = False
    s.log = logging.getLogger("test.worker_load_identity")
    s.cache_dir = tmp_path
    (tmp_path / "model_new.pt").write_bytes(b"generated-cache-hit")
    s._require_slot_planes_match_manifest = lambda _m: None
    s._flush_pre_swap_buffer_if_stale = lambda **_kw: None
    s._ensure_local_model_at_sha = lambda **_kw: tmp_path / "model_new.pt"
    s._sync_opening_books = lambda _m: None
    s._maybe_ingest_dole_flag = lambda _m: None
    s._reco_changed = lambda *_a, **_kw: False
    return s


def _manifest() -> dict[str, Any]:
    return {"model": {"sha256": "new"}, "trainer_step": 99, "model_config": {"test": True}}


def test_poll_failed_load_preserves_identity_and_direct_swap_retries(session: Any) -> None:
    old_model, old_cfg = session.model, session.model_cfg_active
    calls: list[str] = []

    def fail(*_a: Any, **_kw: Any) -> None:
        calls.append("load")
        raise RuntimeError("generated loader failure")

    session._load_and_compile_model = fail
    session._poll_manifest = _manifest
    session._periodic_manifest_poll()
    assert session.model is old_model
    assert session.model_cfg_active is old_cfg
    assert (session.model_sha, session.model_step, session.last_model_sha) == ("old", 2, "old")
    assert not session._stop_selfplay
    with pytest.raises(RuntimeError, match="generated loader failure"):
        session._swap_model_from_manifest(_manifest())
    assert calls == ["load", "load"]
    assert session.model is old_model
    assert session.model_cfg_active is old_cfg
    assert (session.model_sha, session.model_step, session.last_model_sha) == ("old", 2, "old")


def test_successful_sync_commits_model_cfg_identity_and_resyncs(session: Any) -> None:
    old_cfg = session.model_cfg_active
    new_model = object()
    resynced: list[Any] = []
    session._load_and_compile_model = lambda *_a, **_kw: new_model
    session._resync_evaluator_to_model = lambda: resynced.append(session.model)
    session._sync_model(_manifest())
    assert session.model is new_model
    assert session.model_cfg_active is not old_cfg
    assert not session.model_cfg_active.use_gradient_checkpointing
    assert (session.model_sha, session.model_step, session.last_model_sha) == ("new", 99, "new")
    assert resynced == [new_model]


@pytest.mark.parametrize("client_only", [False, True])
def test_no_load_identity_paths_are_preserved(session: Any, client_only: bool) -> None:
    old_model, old_cfg = session.model, session.model_cfg_active
    if client_only:
        session.inference_client = object()
    else:
        session.last_model_sha = "new"

    def fail(*_a: Any, **_kw: Any) -> None:
        raise AssertionError("no-load path called loader")

    session._load_and_compile_model = fail
    session._sync_model(_manifest())
    assert session.model is old_model
    assert session.model_cfg_active is old_cfg
    assert (session.model_sha, session.model_step) == ("new", 99)
    assert session.last_model_sha == ("old" if client_only else "new")


def test_absent_download_retains_existing_retry_signal(session: Any) -> None:
    old_model, old_cfg = session.model, session.model_cfg_active
    session._ensure_local_model_at_sha = lambda **_kw: None
    session._sync_model(_manifest())
    assert session.model is old_model
    assert session.model_cfg_active is old_cfg
    assert (session.model_sha, session.model_step, session.last_model_sha) == ("", 99, "old")

