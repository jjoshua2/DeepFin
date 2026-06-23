from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from chess_anti_engine.model import ModelConfig
import chess_anti_engine.worker as worker_mod
from chess_anti_engine.selfplay.manager import BatchStats
from chess_anti_engine.worker import WorkerSession
from chess_anti_engine.worker_buffer import _BufferedUpload


def _bare_worker_session() -> WorkerSession:
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.worker_model_update")
    session.leased_trial_id = "trial_00000"
    session.pause_selfplay_active = False
    session._stop_selfplay = False
    session._active_reco = {k: None for k in WorkerSession._RECO_RESTART_KEYS}
    session._active_reco["sf_nodes"] = 100
    session._last_manifest_poll_s = time.time()
    session._manifest_mtime = None
    session.model_sha = "old-sha"
    session.args = SimpleNamespace()
    session.opening_book_path = None
    session.opening_book_path_2 = None
    return session


def test_manifest_compat_accepts_compact_lc0_policy() -> None:
    session = _bare_worker_session()
    compat = WorkerSession._check_manifest_compat(
        session,
        {
            "protocol_version": worker_mod.PROTOCOL_VERSION,
            "encoding": {
                "input_planes": 146,
                "policy_size": 1858,
                "policy_encoding": "lc0_1858",
            },
        },
    )
    assert compat.protocol_mismatch is False


def test_mtime_model_update_swaps_before_reco_restart(tmp_path: Path) -> None:
    session = _bare_worker_session()
    manifest = {
        "model": {"sha256": "new-sha"},
        "trainer_step": 449,
        "recommended_worker": {"sf_nodes": 200},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    swaps: list[str] = []

    def _swap_model_from_manifest(manifest: dict) -> None:
        swaps.append(str(manifest["model"]["sha256"]))
        session.model_sha = str(manifest["model"]["sha256"])

    session._resolve_local_manifest_path = lambda: manifest_path
    cast(Any, session)._swap_model_from_manifest = _swap_model_from_manifest

    WorkerSession._check_model_update(session)

    assert session._stop_selfplay is True
    assert swaps == ["new-sha"]
    assert session.model_sha == "new-sha"


def test_periodic_manifest_poll_swaps_before_reco_restart() -> None:
    session = _bare_worker_session()
    manifest = {
        "task": {"type": "selfplay"},
        "model": {"sha256": "new-sha"},
        "trainer_step": 449,
        "recommended_worker": {"sf_nodes": 200},
    }

    swaps: list[str] = []

    def _swap_model_from_manifest(manifest: dict) -> None:
        swaps.append(str(manifest["model"]["sha256"]))
        session.model_sha = str(manifest["model"]["sha256"])

    session._poll_manifest = lambda: manifest
    cast(Any, session)._swap_model_from_manifest = _swap_model_from_manifest

    WorkerSession._periodic_manifest_poll(session)

    assert session._stop_selfplay is True
    assert swaps == ["new-sha"]
    assert session.model_sha == "new-sha"


def test_cp_wdl_recommendation_changes_restart_selfplay_session() -> None:
    session = _bare_worker_session()
    old = {
        "sf_nodes": 100,
        "sf_move_nodes": 0,
        "sf_wdl_use_cp_logistic": False,
        "sf_wdl_cp_slope": 0.010,
        "sf_wdl_cp_draw_width": 60.0,
    }
    session._active_reco = {k: old.get(k) for k in WorkerSession._RECO_RESTART_KEYS}

    changed = WorkerSession._reco_changed(
        session,
        {
            "recommended_worker": {
                **old,
                "sf_wdl_use_cp_logistic": True,
            }
        },
        source_tag="test",
    )

    assert changed is True
    assert session._stop_selfplay is True


def test_build_selfplay_configs_uses_manifest_policy_and_history_encoding() -> None:
    session = _bare_worker_session()

    cfgs, _sf_args = WorkerSession._build_selfplay_configs(
        session,
        {
            "sf_nodes": 5000,
            "policy_encoding": "lc0_1858",
            "input_history_encoding": "lc0_root_legacy_meta",
        },
    )

    game_cfg = cfgs["game"]
    assert game_cfg.policy_encoding == "lc0_1858"
    assert game_cfg.input_history_encoding == "lc0_root_legacy_meta"


def test_build_selfplay_configs_consumes_history_rep_fix() -> None:
    """The published history_rep_fix flag must reach the worker GameConfig —
    otherwise external workers silently record legacy repetition planes."""
    session = _bare_worker_session()

    cfgs, _sf_args = WorkerSession._build_selfplay_configs(
        session, {"sf_nodes": 5000, "history_rep_fix": True},
    )
    assert cfgs["game"].history_rep_fix is True

    cfgs, _sf_args = WorkerSession._build_selfplay_configs(session, {"sf_nodes": 5000})
    assert cfgs["game"].history_rep_fix is False

    # Flipping the flag must restart the selfplay session.
    assert "history_rep_fix" in WorkerSession._RECO_RESTART_KEYS


def test_build_selfplay_configs_consumes_categorical_blend_frac() -> None:
    """The published categorical_blend_frac must reach the worker GameConfig —
    otherwise distributed workers silently emit legacy ternary categorical
    targets and the experiment measures the control."""
    session = _bare_worker_session()

    cfgs, _sf_args = WorkerSession._build_selfplay_configs(
        session, {"sf_nodes": 5000, "categorical_blend_frac": 0.5},
    )
    assert cfgs["game"].categorical_blend_frac == 0.5

    cfgs, _sf_args = WorkerSession._build_selfplay_configs(session, {"sf_nodes": 5000})
    assert cfgs["game"].categorical_blend_frac == 0.0

    # Flipping the flag mid-session must restart selfplay so it takes effect.
    assert "categorical_blend_frac" in WorkerSession._RECO_RESTART_KEYS


def test_encoding_recommendation_changes_restart_selfplay_session() -> None:
    session = _bare_worker_session()
    old = {
        "policy_encoding": "az_4672",
        "input_history_encoding": "legacy",
    }
    session._active_reco = {k: old.get(k) for k in WorkerSession._RECO_RESTART_KEYS}

    changed = WorkerSession._reco_changed(
        session,
        {
            "recommended_worker": {
                **old,
                "input_history_encoding": "lc0_root",
            }
        },
        source_tag="test",
    )

    assert changed is True
    assert session._stop_selfplay is True


def test_aggregate_thread_stats_preserves_counted_float_sums() -> None:
    stats = WorkerSession._aggregate_thread_stats([
        BatchStats(
            games=1,
            positions=10,
            w=1,
            d=0,
            l=0,
            sf_eval_delta6=0.25,
            sf_eval_delta6_n=4,
            diff_focus_records=2,
            diff_focus_keep_prob_sum=1.5,
            diff_focus_sample_weight_sum=2.5,
            diff_focus_priority_sum=3.0,
            diff_focus_priority_sq_sum=5.0,
            diff_focus_priority_min=0.4,
            diff_focus_priority_max=2.0,
            gumbel_policy_diag_n=2,
            gumbel_policy_top_prob_sum=0.8,
            gumbel_policy_entropy_sum=1.2,
            outcome_stats={"book:a": 1},
        ),
        BatchStats(
            games=2,
            positions=20,
            w=0,
            d=1,
            l=1,
            sf_eval_delta6=0.75,
            sf_eval_delta6_n=2,
            diff_focus_records=3,
            diff_focus_keep_prob_sum=2.0,
            diff_focus_sample_weight_sum=3.5,
            diff_focus_priority_sum=7.0,
            diff_focus_priority_sq_sum=11.0,
            diff_focus_priority_min=0.2,
            diff_focus_priority_max=3.0,
            gumbel_policy_diag_n=3,
            gumbel_policy_top_prob_sum=1.1,
            gumbel_policy_entropy_sum=2.4,
            outcome_stats={"book:a": 2, "book:b": 1},
        ),
    ])

    assert stats.games == 3
    assert stats.positions == 30
    assert stats.sf_eval_delta6_n == 6
    assert stats.sf_eval_delta6 == pytest.approx((0.25 * 4 + 0.75 * 2) / 6)
    assert stats.diff_focus_keep_prob_sum == pytest.approx(3.5)
    assert stats.diff_focus_sample_weight_sum == pytest.approx(6.0)
    assert stats.diff_focus_priority_sum == pytest.approx(10.0)
    assert stats.diff_focus_priority_sq_sum == pytest.approx(16.0)
    assert stats.diff_focus_priority_min == pytest.approx(0.2)
    assert stats.diff_focus_priority_max == pytest.approx(3.0)
    assert stats.gumbel_policy_diag_n == 5
    assert stats.gumbel_policy_top_prob_sum == pytest.approx(1.9)
    assert stats.gumbel_policy_entropy_sum == pytest.approx(3.6)
    assert stats.outcome_stats == {"book:a": 3, "book:b": 1}


def test_threaded_local_model_swap_keeps_buffer_metadata_atomic(monkeypatch, tmp_path: Path) -> None:
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.worker_model_update")
    session.inference_client = None
    session.model_sha = "old-sha"
    session.model_step = 1
    session.last_model_sha = "old-sha"
    session.model_cfg_active = ModelConfig(kind="tiny")
    session.cache_dir = tmp_path
    session.pending_dir = tmp_path / "pending"
    session.pending_dir.mkdir()
    session.trial_api_prefix = "/v1/trials/trial_00000"
    session.leased_trial_id = "trial_00000"
    session.fixed_trial_id = ""
    session.args = SimpleNamespace(username="worker")
    session.upload_buf = _BufferedUpload(positions=1, games=1, samples=[])
    lock = threading.Lock()
    session._upload_buf_lock = lock

    new_sha = "new-sha"
    (tmp_path / f"model_{new_sha}.pt").write_bytes(b"checkpoint")
    new_model = torch.nn.Identity()
    evaluator = SimpleNamespace(model="old-model")
    cast(Any, session)._direct_evaluator = evaluator

    session._server_url_for = lambda endpoint: f"http://server{endpoint}"
    session._load_and_compile_model = lambda *_args, **_kwargs: new_model
    monkeypatch.setattr(worker_mod, "_sha256_file", lambda _path: new_sha)

    sync_events: list[tuple[bool, str, int]] = []
    upload_events: list[bool] = []

    def _fake_flush(**kwargs):
        assert lock.locked()
        buf = kwargs["buf"]
        assert buf.positions == 1
        buf.positions = 0
        buf.samples = []
        return tmp_path / "pending.zarr", 7.0

    def _fake_sync(evaluator_arg, model_arg) -> None:
        sync_events.append((lock.locked(), session.model_sha, session.upload_buf.positions))
        evaluator_arg.model = model_arg

    def _fake_upload_pending_shards(*, default_elapsed_s: float | None = None) -> float:
        upload_events.append(lock.locked())
        assert default_elapsed_s == 7.0
        return 123.0

    monkeypatch.setattr(worker_mod, "_flush_upload_buffer_to_pending", _fake_flush)
    monkeypatch.setattr(worker_mod, "_sync_evaluator_to_model", _fake_sync)
    cast(Any, session)._upload_pending_shards = _fake_upload_pending_shards

    WorkerSession._swap_model_from_manifest(
        session,
        {"model": {"sha256": new_sha}, "trainer_step": 2},
    )

    assert sync_events == [(True, "old-sha", 0)]
    assert upload_events == [False]
    assert evaluator.model is new_model
    assert session.model is new_model
    assert session.model_sha == new_sha
    assert session.model_step == 2
    assert session.last_model_sha == new_sha


def test_completed_game_metadata_mismatch_flushes_before_retry(monkeypatch, tmp_path: Path) -> None:
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.worker_model_update")
    session.model_sha = "new-sha"
    session.model_step = 2
    session.pending_dir = tmp_path
    session.leased_trial_id = "trial_00000"
    session.fixed_trial_id = ""
    session.args = SimpleNamespace(
        username="worker",
        upload_max_buffered_positions=0,
        upload_target_positions=999,
        upload_flush_seconds=999.0,
    )
    session.last_successful_send_s = 100.0
    session.upload_buf = _BufferedUpload(
        samples=cast(Any, [object()]),
        model_sha="old-sha",
        model_step=1,
        games=1,
        positions=1,
        first_buffered_at_s=50.0,
    )
    session._completion_telemetry_lock = threading.Lock()
    session._completion_games = 0
    session._completion_positions = 0
    session._completion_callback_s = 0.0
    session._completion_upload_s = 0.0
    session._completion_by_thread = {}

    flushed: list[tuple[str | None, int | None, int]] = []
    uploaded_elapsed: list[float | None] = []
    maybe_flush_seen: list[tuple[str | None, int | None, int]] = []

    def _fake_flush(**kwargs):
        buf = kwargs["buf"]
        flushed.append((buf.model_sha, buf.model_step, buf.positions))
        buf.reset()
        return tmp_path / "old-shard.zarr", 12.5

    def _fake_upload_pending_shards(*, default_elapsed_s: float | None = None) -> float:
        uploaded_elapsed.append(default_elapsed_s)
        return 200.0

    def _fake_maybe_flush(**kwargs):
        buf = kwargs["buf"]
        maybe_flush_seen.append((buf.model_sha, buf.model_step, buf.positions))
        return None, 0.0

    monkeypatch.setattr(worker_mod, "_flush_upload_buffer_to_pending", _fake_flush)
    monkeypatch.setattr(worker_mod, "_maybe_flush_upload_buffer", _fake_maybe_flush)
    cast(Any, session)._upload_pending_shards = _fake_upload_pending_shards

    game_batch = SimpleNamespace(samples=[object(), object()], positions=2, games=1, w=1)

    WorkerSession._on_completed_game(session, game_batch)

    assert flushed == [("old-sha", 1, 1)]
    assert uploaded_elapsed == [12.5]
    assert session.last_successful_send_s == 200.0
    assert maybe_flush_seen == [("new-sha", 2, 2)]
    assert session.upload_buf.model_sha == "new-sha"
    assert session.upload_buf.model_step == 2
    assert session.upload_buf.positions == 2
    assert session._completion_games == 1
    assert session._completion_positions == 2


def test_require_reco_raises_when_sf_nodes_set_nowhere() -> None:
    session = _bare_worker_session()  # args is an empty namespace
    with pytest.raises(worker_mod._MissingRequiredReco):
        WorkerSession._require_reco(session, {}, "sf_nodes", int)


def test_require_reco_reads_from_reco() -> None:
    session = _bare_worker_session()
    assert WorkerSession._require_reco(session, {"sf_nodes": 5000}, "sf_nodes", int) == 5000


def test_require_reco_cli_overrides_reco() -> None:
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=5000)
    assert WorkerSession._require_reco(session, {"sf_nodes": 9999}, "sf_nodes", int) == 5000


def test_build_selfplay_configs_requires_sf_nodes() -> None:
    """No fail-weak 2000 default: a manifest without sf_nodes must refuse to
    build SF rather than silently run the opponent at a guessed budget."""
    session = _bare_worker_session()
    with pytest.raises(worker_mod._MissingRequiredReco):
        WorkerSession._build_selfplay_configs(session, {})
    # Present (and positive) -> resolves normally.
    _cfgs, sf_args = WorkerSession._build_selfplay_configs(session, {"sf_nodes": 5000})
    assert sf_args[0] == 5000


def test_sf_fast_ply_node_scale_triggers_session_restart() -> None:
    """The GameConfig is built once per session, so changing the published
    sf_fast_ply_node_scale must restart selfplay to take effect — i.e. it must
    be in the reco-restart key set (same as sf_nodes / sf_move_nodes)."""
    assert "sf_fast_ply_node_scale" in WorkerSession._RECO_RESTART_KEYS
