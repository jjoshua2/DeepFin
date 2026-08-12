from __future__ import annotations

from collections import deque
import json
import logging
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from chess_anti_engine.replay import ReplaySample
from chess_anti_engine.replay.shard import load_shard_arrays
from chess_anti_engine.model import ModelConfig
import chess_anti_engine.worker as worker_mod
from chess_anti_engine.selfplay.manager import BatchStats
from chess_anti_engine.worker import WorkerSession
from chess_anti_engine.worker_buffer import _buffer_add_completed_game, _BufferedUpload


def _bare_worker_session() -> WorkerSession:
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.worker_model_update")
    session.leased_trial_id = "trial_00000"
    session.pause_selfplay_active = False
    session._stop_selfplay = False
    session._hold_selfplay = False
    session._hold_on_pause = True
    session._active_reco = dict.fromkeys(WorkerSession._RECO_RESTART_KEYS)
    session._active_reco["sf_nodes"] = 100
  # Test manifests carry no stockfish/opening-book assets, so the session-start
  # fingerprint is all-None; match it so the asset-change gate stays quiet.
    session._active_assets = (None, None, None, None)
    session._last_manifest_poll_s = time.time()
    session._manifest_mtime = None
    session._manifest_path = None
    session._model_watch_started = False
    session._model_watch_lock = threading.Lock()
    session._model_stale_since_s = None
    session._model_stale_alarmed = False
    session._selfplay_session_active = True
    session.model_sha = "old-sha"
    session.args = SimpleNamespace()
    session.opening_book_path = None
    session.opening_book_path_2 = None
    session.opening_fen_list_path = None
    session._pending_fen_dole = []
    session._live_dole_queue = None
    session._pending_sf_refute = []
    session._live_sf_refute_queue = None
    session._dole_lock = threading.Lock()
    session._live_states = []
    session._live_states_lock = threading.Lock()
    session._pending_live_override = None
    return session


def test_broker_stats_also_emit_selfplay_phase_stats(monkeypatch) -> None:
    class _FakeBroker:
        def __init__(self) -> None:
            self.stats = {
                "lifetime_requests": 10,
                "lifetime_positions": 100,
                "lifetime_legal_requests": 8,
                "lifetime_legal_positions": 90,
                "lifetime_wait_s": 0.2,
                "lifetime_roundtrip_s": 0.4,
                "slot_requests": [3, 3, 2, 2],
                "slots": 4,
                "max_inflight": 4,
                "available_slots": 2,
            }

    monkeypatch.setattr(worker_mod, "MultiSlotInferenceClient", _FakeBroker)
    session = object.__new__(WorkerSession)
    session_any = cast(Any, session)
    session_any.inference_client = _FakeBroker()
    session_any._last_broker_client_stats_log_s = 0.0
    session_any._last_broker_client_stats_snapshot = {}
    session_any.log = Mock()
    session_any._completion_telemetry_lock = threading.Lock()
    session_any._completion_games = 0
    session_any._completion_positions = 0
    session_any._completion_callback_s = 0.0
    session_any._completion_upload_s = 0.0
    session_any._completion_by_thread = {}
    session_any._last_completion_stats_snapshot = (0, 0, 0.0, 0.0, {})
    phase_stats = Mock()
    session_any._maybe_log_selfplay_phase_stats = phase_stats

    WorkerSession._maybe_log_broker_client_stats(session, 60.0)

    phase_stats.assert_called_once_with(60.0)


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


def test_build_selfplay_configs_consumes_slot_oversubscribe() -> None:
    session = _bare_worker_session()

    cfgs, _sf_args = WorkerSession._build_selfplay_configs(
        session, {"sf_nodes": 5000, "slot_oversubscribe": 2.0},
    )
    assert cfgs["slot_oversubscribe"] == 2.0

    cfgs, _sf_args = WorkerSession._build_selfplay_configs(session, {"sf_nodes": 5000})
    assert cfgs["slot_oversubscribe"] == 1.0
    assert "slot_oversubscribe" in WorkerSession._RECO_RESTART_KEYS


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
    session._evaluator_model_id = None  # _resync_evaluator_to_model reads this

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


def test_completed_game_metadata_mismatch_flushes_before_retry(tmp_path: Path) -> None:
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

    queued: list[tuple[str | None, int | None, int]] = []

    def _fake_try_upload_pending_shards(*, default_elapsed_s: float | None = None) -> float:
        del default_elapsed_s
        old_buf = session._pending_buffer_flushes[0][0]
        queued.append((old_buf.model_sha, old_buf.model_step, old_buf.positions))
        return 200.0

    cast(Any, session)._try_upload_pending_shards = _fake_try_upload_pending_shards

    game_batch = SimpleNamespace(samples=[object(), object()], positions=2, games=1, w=1)

    WorkerSession._on_completed_game(session, game_batch)

    assert queued == [("old-sha", 1, 1)]
    assert session.last_successful_send_s == 200.0
    assert session.upload_buf.model_sha == "new-sha"
    assert session.upload_buf.model_step == 2
    assert session.upload_buf.positions == 2
    assert session._completion_games == 1
    assert session._completion_positions == 2


def test_completed_game_does_not_wait_for_active_shard_upload(
    monkeypatch, tmp_path: Path,
) -> None:
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.worker_model_update")
    session.model_sha = "model-sha"
    session.model_step = 1
    session.pending_dir = tmp_path
    session.leased_trial_id = "trial_00000"
    session.fixed_trial_id = ""
    session.args = SimpleNamespace(
        username="worker",
        upload_max_buffered_positions=0,
        upload_target_positions=1,
        upload_flush_seconds=999.0,
    )
    session.last_successful_send_s = 100.0
    session.upload_buf = _BufferedUpload()
    upload_buf_lock = threading.Lock()
    session._upload_buf_lock = upload_buf_lock
    session._completion_telemetry_lock = threading.Lock()
    session._completion_games = 0
    session._completion_positions = 0
    session._completion_callback_s = 0.0
    session._completion_upload_s = 0.0
    session._completion_by_thread = {}

    upload_started = threading.Event()
    release_upload = threading.Event()
    materialized_positions: list[int] = []

    def _fake_flush(**kwargs):
        assert not upload_buf_lock.locked()
        materialized_positions.append(int(kwargs["buf"].positions))
        kwargs["buf"].reset()
        return tmp_path / "pending.zarr", 1.0

    def _fake_upload_pending_shards_locked(*, default_elapsed_s: float | None = None) -> float:
        assert default_elapsed_s == 1.0
        upload_started.set()
        assert release_upload.wait(timeout=2.0)
        return 200.0

    monkeypatch.setattr(worker_mod, "_flush_upload_buffer_to_pending", _fake_flush)
    session._pending_upload_lock = threading.Lock()
    cast(Any, session)._upload_pending_shards_locked = _fake_upload_pending_shards_locked
    game_batch = SimpleNamespace(samples=[object(), object()], positions=2, games=1, w=1)

    first = threading.Thread(target=WorkerSession._on_completed_game, args=(session, game_batch))
    first.start()
    assert upload_started.wait(timeout=2.0)

    second = threading.Thread(target=WorkerSession._on_completed_game, args=(session, game_batch))
    second.start()
    second.join(timeout=1.0)
    assert not second.is_alive(), "next completion waited for unrelated shard HTTP upload"
    assert session.upload_buf.positions == 0
    assert session._pending_buffer_positions == 2
    assert sum(materialized_positions) + session._pending_buffer_positions == 4

    release_upload.set()
    first.join(timeout=2.0)
    assert not first.is_alive()
    assert session._completion_games == 2
    assert session._completion_positions == 4


def test_failed_detached_buffer_materialization_stays_queued(monkeypatch, tmp_path: Path) -> None:
    session = object.__new__(WorkerSession)
    session.pending_dir = tmp_path
    session.args = SimpleNamespace(username="worker")
    session._upload_buf_lock = threading.Lock()
    queued = _BufferedUpload(samples=cast(Any, [object()]), positions=3, games=1)
    session._pending_buffer_flushes = deque([
        (queued, 100.0, "trial_00000", 3),
    ])
    session._pending_buffer_positions = 3

    def _fail_flush(**_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(worker_mod, "_flush_upload_buffer_to_pending", _fail_flush)

    with pytest.raises(OSError, match="disk full"):
        WorkerSession._materialize_queued_buffers_locked(session)

    assert len(session._pending_buffer_flushes) == 1
    assert session._pending_buffer_flushes[0][0] is queued
    assert queued.positions == 3
    assert session._pending_buffer_positions == 3


def test_detached_buffer_materializes_losslessly_to_pending_shard(tmp_path: Path) -> None:
    session = object.__new__(WorkerSession)
    session.pending_dir = tmp_path
    session.args = SimpleNamespace(username="worker")
    session.leased_trial_id = "trial_00000"
    session.fixed_trial_id = ""
    session._upload_buf_lock = threading.Lock()
    session._pending_buffer_flushes = deque()
    session._pending_buffer_positions = 0
    session.upload_buf = _BufferedUpload()
    policy = np.zeros((1858,), dtype=np.float32)
    policy[0] = 1.0
    sample = ReplaySample(
        x=np.zeros((175, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=1,
    )
    game_batch = SimpleNamespace(samples=[sample], positions=1, games=1, w=1)
    _buffer_add_completed_game(
        buf=session.upload_buf,
        game_batch=game_batch,
        now_s=100.0,
        model_sha="model-sha",
        model_step=7,
    )

    with session._upload_buf_lock:
        assert WorkerSession._queue_upload_buffer_locked(session, now_s=101.0)
    assert session.upload_buf.positions == 0
    assert session._pending_buffer_positions == 1

    elapsed_s = WorkerSession._materialize_queued_buffers_locked(session)

    assert elapsed_s == 1.0
    assert session._pending_buffer_positions == 0
    assert not session._pending_buffer_flushes
    shards = list(tmp_path.glob("*.zarr"))
    assert len(shards) == 1
    arrays, meta = load_shard_arrays(shards[0])
    assert int(arrays["x"].shape[0]) == 1
    assert int(meta["positions"]) == 1
    assert meta["model_sha256"] == "model-sha"
    assert meta["model_step"] == 7
    assert meta["run_id"] == "trial_00000"


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
    assert sf_args[2] == 16


def test_sf_node_knobs_are_live_applied_not_restart() -> None:
    """SF node budgets + the selfplay/curriculum mix are read fresh per
    move/recycle, so the worker applies them to the running SelfplayState in
    place (no session restart that would abandon the in-flight games). NOTE:
    sf_move_nodes is intentionally restart-only — see
    test_sf_move_nodes_is_restart_only_not_live."""
    for key in ("sf_fast_ply_node_scale", "sf_nodes",
                "selfplay_fraction", "opponent_wdl_regret_limit"):
        assert key in WorkerSession._RECO_LIVE_KEYS
        assert key not in WorkerSession._RECO_RESTART_KEYS


def test_sf_move_nodes_is_restart_only_not_live() -> None:
    """sf_move_nodes gates the curriculum SF query path (the 0-boundary switches
    move-futures into label-futures), so it must restart rather than live-apply
    to avoid writing low-node SF targets for in-flight positions."""
    assert "sf_move_nodes" in WorkerSession._RECO_RESTART_KEYS
    assert "sf_move_nodes" not in WorkerSession._RECO_LIVE_KEYS


def _live_state() -> Any:
    """A minimal stand-in for the live SelfplayState with real config
    dataclasses (so `dataclasses.replace` works) carrying a distinctive,
    session-fixed `max_plies` to prove untracked fields are not transplanted.
    Binds the REAL SelfplayState.apply_live_overrides / terminal_eval_nodes_for
    so the live-apply path is exercised against the actual implementation."""
    from chess_anti_engine.selfplay.config import GameConfig, OpponentConfig
    from chess_anti_engine.selfplay.state import SelfplayState

    class _FakeState:
        apply_live_overrides = SelfplayState.apply_live_overrides
        terminal_eval_nodes_for = staticmethod(SelfplayState.terminal_eval_nodes_for)

        def __init__(self) -> None:
            self.game = GameConfig(selfplay_fraction=0.30, max_plies=137)
            self.opponent = OpponentConfig(wdl_regret_limit=0.04)
            self.base_nodes = 5000
            self.terminal_eval_nodes = 25000

    return _FakeState()


def test_live_reco_change_applies_without_restart() -> None:
    """A live-only reco change (selfplay_fraction + sf_nodes) on an active
    session swaps the live fields in place and does NOT request a restart."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None)
    session._active_reco = session._snapshot_reco({"sf_nodes": 5000, "selfplay_fraction": 0.30})

    captured: dict[str, Any] = {}
    session._live_states = [_live_state()]
    cast(Any, session).sf = SimpleNamespace(set_nodes=lambda n: captured.update(nodes=n))

    changed = WorkerSession._reco_changed(
        session,
        {"recommended_worker": {"sf_nodes": 6000, "selfplay_fraction": 0.50}},
        source_tag="test",
    )

    assert changed is False
    assert session._stop_selfplay is False
    st = session._live_states[0]
    assert st.game.selfplay_fraction == 0.50
    assert st.base_nodes == 6000
    assert captured == {"nodes": 6000}
    assert session._active_reco["selfplay_fraction"] == 0.50


def test_apply_live_reco_transplants_only_live_fields() -> None:
    """_apply_live_reco swaps ONLY the live-safe fields onto the running config,
    never the whole rebuilt config — so even a (restart-keyed) session-fixed
    field like max_plies present in the reco never mutates the session value.
    Tested directly (not via _reco_changed) since a max_plies change now routes
    to a restart, not the live path."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None)
    session._live_states = [_live_state()]  # max_plies=137
    cast(Any, session).sf = SimpleNamespace(set_nodes=lambda _n: None)

    applied = WorkerSession._apply_live_reco(
        session,
        {"sf_nodes": 5000, "selfplay_fraction": 0.50, "max_plies": 999},
    )

    assert applied is True
    st = session._live_states[0]
    assert st.game.selfplay_fraction == 0.50  # live field applied
    assert st.game.max_plies == 137  # session-fixed field preserved


def test_apply_live_reco_falls_back_to_restart_on_build_error(caplog: Any) -> None:
    """Codex review: a malformed reco that makes config construction raise must
    NOT be swallowed silently — _apply_live_reco logs and returns False so the
    caller restarts (re-parsing the reco at bring-up), rather than pinning the
    worker at the old config."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None)
    session._live_states = [_live_state()]
    cast(Any, session).sf = SimpleNamespace(set_nodes=lambda _n: None)

    # mcts_simulations must be int(); a non-numeric string raises ValueError.
    with caplog.at_level(logging.WARNING):
        applied = WorkerSession._apply_live_reco(
            session, {"sf_nodes": 5000, "mcts_simulations": "not-a-number"},
        )
    assert applied is False
    assert any("falling back to restart" in r.message for r in caplog.records)


def test_every_reco_field_is_watched() -> None:
    """Completeness guard (Codex review): every recommended_worker field that
    _build_selfplay_configs consumes must be in _RECO_LIVE_KEYS or
    _RECO_RESTART_KEYS. Otherwise a mid-run change to it silently never reaches
    workers now that the PID levers are live (no incidental restart to flush it)."""
    import inspect
    # Scan both the config builder and the session-start path (games_per_batch
    # is read in _run_selfplay, not _build_selfplay_configs).
    src = inspect.getsource(WorkerSession._build_selfplay_configs)
    src += inspect.getsource(WorkerSession._run_selfplay)
    keys: set[str] = set()
    # \s* after each "(" so wrapped calls like `_resolve_reco(\n    reco, "key"...)`
    # are caught too (\s matches the newline) — otherwise a multi-line knob could
    # be silently unwatched while passing this guard.
    for pat in (
        r'reco\.get\(\s*["\']([a-z0-9_]+)["\']',
        r'_resolve_reco\(\s*reco,\s*["\']([a-z0-9_]+)["\']',
        r'_optional_reco\(\s*["\']([a-z0-9_]+)["\']',
        r'_require_reco\(\s*reco,\s*["\']([a-z0-9_]+)["\']',
    ):
        keys |= set(re.findall(pat, src))
    watched = set(WorkerSession._RECO_LIVE_KEYS) | set(WorkerSession._RECO_RESTART_KEYS)
    unwatched = sorted(keys - watched)
    assert not unwatched, f"reco fields neither live nor restart-keyed: {unwatched}"


def test_every_live_key_is_transplanted() -> None:
    """Drift guard (Codex review): every _RECO_LIVE_KEYS entry must actually
    land on the live SelfplayState when changed — catches a live key added to
    the tuple but forgotten in _apply_live_reco's transplant body."""
    # key -> (baseline value, changed value, accessor on the live state)
    cases: dict[str, tuple[Any, Any, Any]] = {
        "selfplay_fraction": (0.30, 0.55, lambda st: st.game.selfplay_fraction),
        "sf_fast_ply_node_scale": (0.25, 0.6, lambda st: st.game.sf_fast_ply_node_scale),
        "sf_label_nodes_cap": (0, 100_000, lambda st: st.game.sf_label_nodes_cap),
        "sf_label_nodes_floor": (0, 700_000, lambda st: st.game.sf_label_nodes_floor),
        "sf_label_escalate_q_gap": (
            0.0, 0.8, lambda st: st.game.sf_label_escalate_q_gap,
        ),
        "sf_label_escalate_nodes": (
            3_000_000, 5_000_000, lambda st: st.game.sf_label_escalate_nodes,
        ),
        "sf_label_escalate_max_per_game": (
            2, 4, lambda st: st.game.sf_label_escalate_max_per_game,
        ),
        "opponent_wdl_regret_limit": (0.04, 0.02, lambda st: st.opponent.wdl_regret_limit),
        "sf_nodes": (5000, 7000, lambda st: st.base_nodes),
    }
    assert set(cases) == set(WorkerSession._RECO_LIVE_KEYS), (
        "update test_every_live_key_is_transplanted for the new live key(s)"
    )
    for key, (base, changed, get) in cases.items():
        session = _bare_worker_session()
        session.args = SimpleNamespace(sf_nodes=None)
        session._live_states = [_live_state()]
        cast(Any, session).sf = SimpleNamespace(set_nodes=lambda _n: None)
        baseline = {"sf_nodes": 5000, key: base}
        applied = WorkerSession._apply_live_reco(session, {**baseline, key: changed})
        assert applied is True
        assert get(session._live_states[0]) == changed, f"live key {key} not transplanted"


def test_live_reco_log_line_names_the_label_floor(caplog) -> None:
    """PR #354 M1/R2: the live-reco log line is the deploy-verification
    instrument for sf_label_nodes_floor — the knob exists because an 11x
    teacher cut happened with no log line anywhere, so silently dropping
    `label_floor=` from the line must fail a test, log line or not."""
    import logging

    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None)
    session._live_states = [_live_state()]
    cast(Any, session).sf = SimpleNamespace(set_nodes=lambda _n: None)

    with caplog.at_level(logging.INFO):
        applied = WorkerSession._apply_live_reco(
            session, {"sf_nodes": 5000, "sf_label_nodes_floor": 700_000},
        )

    assert applied is True
    assert any("label_floor=700000" in r.getMessage() for r in caplog.records), (
        "the live-reco log line no longer reports the label floor"
    )


def test_sf_multipv_change_triggers_restart() -> None:
    """Codex #81: sf_multipv is applied only at engine (re)init, so a change
    must restart even when bundled with a live-only sf_nodes update."""
    assert "sf_multipv" in WorkerSession._RECO_RESTART_KEYS
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None)
    session._active_reco = session._snapshot_reco({"sf_nodes": 5000, "sf_multipv": 5})
    session._live_states = [_live_state()]

    changed = WorkerSession._reco_changed(
        session,
        {"recommended_worker": {"sf_nodes": 6000, "sf_multipv": 10}},
        source_tag="test",
    )

    assert changed is True
    assert session._stop_selfplay is True


def test_session_asset_change_triggers_restart() -> None:
    """Codex review: a session-start asset change (SF binary / opening-book SHA)
    bundled with only live reco keys cannot be live-applied to the running
    engine/book, so it must restart instead of being silently ignored."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None, stockfish_from_server=True)
    session._active_reco = session._snapshot_reco({"sf_nodes": 5000})
    # 4-tuple (sf, book, book2, fen_list) so the change under test is the SF sha
    # (sha_old -> sha_NEW), not the tuple length (Codex review, PR #108).
    session._active_assets = ("sha_old", None, None, None)
    session._live_states = [_live_state()]

    changed = WorkerSession._reco_changed(
        session,
        {
            "stockfish": {"sha256": "sha_NEW"},
            "recommended_worker": {"sf_nodes": 6000},  # only a live key changed
        },
        source_tag="test",
    )

    assert changed is True
    assert session._stop_selfplay is True


def test_sync_stockfish_reinits_on_binary_or_hash_change(monkeypatch: Any) -> None:
    """Distributed workers must rebuild for binary and TT-size changes."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(
        stockfish_path="sf_v1", stockfish_from_server=False, sf_workers=1, sf_nice=0,
    )
    cast(Any, session).sf = None
    session.sf_multipv_active = None
    session.sf_hash_mb_active = None
    session.sf_syzygy_path_active = None
    session.sf_path_active = None

    built: list[tuple[str, int]] = []

    class _FakeSF:
        def __init__(self, path: str, **kw: Any) -> None:
            built.append((path, int(kw["hash_mb"])))

        def close(self) -> None:
            pass

        def set_nodes(self, _n: int) -> None:
            pass

    monkeypatch.setattr(worker_mod, "StockfishUCI", _FakeSF)

    WorkerSession._sync_stockfish(session, {}, 5000, 5, 16)
    WorkerSession._sync_stockfish(session, {}, 5000, 5, 16)  # unchanged -> no rebuild
    assert built == [("sf_v1", 16)]
    WorkerSession._sync_stockfish(session, {}, 5000, 5, 32)
    assert built == [("sf_v1", 16), ("sf_v1", 32)]
    session.args.stockfish_path = "sf_v2"  # binary hot-swap
    WorkerSession._sync_stockfish(session, {}, 5000, 5, 32)
    assert built == [("sf_v1", 16), ("sf_v1", 32), ("sf_v2", 32)]


def test_sync_model_resyncs_evaluator(monkeypatch: Any) -> None:
    """Codex review: a mid-session model swap on the poll path must re-point the
    running evaluator at the new model — otherwise play_batch keeps using the old
    model while shards are tagged with the new SHA."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(username="t")
    session.inference_client = None
    session.last_model_sha = "old"
    session.model_sha = ""
    session.model_step = 0
    session.fixed_trial_id = "trial_00000"  # skip the lease-reset branch
    session.model_cfg_active = None
    cast(Any, session)._direct_evaluator = object()
    session._evaluator_model_id = 999

    new_model = object()
    synced: list[object] = []
    monkeypatch.setattr(session, "_flush_pre_swap_buffer_if_stale", lambda **_k: None)
    monkeypatch.setattr(session, "_ensure_local_model_at_sha", lambda **_k: Path("m.pt"))
    monkeypatch.setattr(session, "_load_and_compile_model", lambda *_a, **_k: new_model)
    monkeypatch.setattr(worker_mod, "model_config_from_manifest_dict", lambda _d: ModelConfig())
    monkeypatch.setattr(worker_mod, "_sync_evaluator_to_model", lambda _ev, m: synced.append(m))

    WorkerSession._sync_model(
        session, {"model": {"sha256": "new"}, "trainer_step": 1, "model_config": {}},
    )

    assert session.model is new_model
    assert synced == [new_model]
    assert session._evaluator_model_id == id(new_model)


def test_pinned_games_per_batch_does_not_restart() -> None:
    """Codex review: when games_per_batch is CLI-pinned, the pin wins over reco,
    so a server-side change to it must NOT force a (no-op) restart."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None)
    session.games_per_batch_local = 64
    session._active_reco = session._snapshot_reco({"sf_nodes": 5000, "games_per_batch": 8})

    changed = WorkerSession._reco_changed(
        session,
        {"recommended_worker": {"sf_nodes": 5000, "games_per_batch": 16}},
        source_tag="test",
    )
    assert changed is False
    assert session._stop_selfplay is False

    # Without the pin, the same change restarts (games_per_batch resizes slots).
    session2 = _bare_worker_session()
    session2.args = SimpleNamespace(sf_nodes=None)
    session2.games_per_batch_local = None
    session2._active_reco = session2._snapshot_reco({"sf_nodes": 5000, "games_per_batch": 8})
    changed2 = WorkerSession._reco_changed(
        session2,
        {"recommended_worker": {"sf_nodes": 5000, "games_per_batch": 16}},
        source_tag="test",
    )
    assert changed2 is True


def test_reco_restart_keys_have_no_duplicates() -> None:
    """The hand-maintained key tuples must stay duplicate-free and disjoint."""
    restart = WorkerSession._RECO_RESTART_KEYS
    assert len(restart) == len(set(restart)), "duplicate in _RECO_RESTART_KEYS"
    assert not (set(WorkerSession._RECO_LIVE_KEYS) & set(restart)), "live/restart overlap"


def test_apply_live_reco_updates_all_registered_states() -> None:
    """Threaded selfplay registers one state per thread; a live reco change must
    reach EVERY registered state — updating only one would leave the other
    threads generating under stale PID settings (the pre-fix behavior was a
    full session restart, which post pause-hold/oversubscribe abandons ~2x
    games_per_batch in-flight games on every PID lever move)."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None)
    states = [_live_state(), _live_state(), _live_state()]
    session._live_states = list(states)
    cast(Any, session).sf = SimpleNamespace(set_nodes=lambda _n: None)

    applied = WorkerSession._apply_live_reco(
        session, {"sf_nodes": 6000, "selfplay_fraction": 0.50},
    )

    assert applied is True
    for st in states:
        assert st.game.selfplay_fraction == 0.50
        assert st.base_nodes == 6000


def test_late_registration_receives_pending_live_override() -> None:
    """Review finding (PR #153): a thread that registers AFTER an in-session
    live apply must be transplanted with the applied values at registration —
    otherwise it silently runs the whole continuous session on session-start
    config while _active_reco already claims the new values."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None)
    session._live_states = [_live_state()]
    cast(Any, session).sf = SimpleNamespace(set_nodes=lambda _n: None)

    applied = WorkerSession._apply_live_reco(
        session, {"sf_nodes": 6000, "selfplay_fraction": 0.50},
    )
    assert applied is True

    late = _live_state()
    WorkerSession._register_live_state(session, late)

    st = late
    assert st.game.selfplay_fraction == 0.50
    assert st.base_nodes == 6000


def test_clear_live_states_drops_registry_and_pending_override() -> None:
    """After _clear_live_states (between sessions) an apply must return False
    (restart fallback), and a fresh registration must NOT inherit the previous
    session's override — the new session was built from the current reco."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None)
    session._live_states = [_live_state()]
    cast(Any, session).sf = SimpleNamespace(set_nodes=lambda _n: None)
    assert WorkerSession._apply_live_reco(
        session, {"sf_nodes": 6000, "selfplay_fraction": 0.50},
    ) is True
    assert session._pending_live_override is not None

    WorkerSession._clear_live_states(session)

    assert WorkerSession._apply_live_reco(
        session, {"sf_nodes": 7000, "selfplay_fraction": 0.60},
    ) is False
    fresh = _live_state()
    WorkerSession._register_live_state(session, fresh)
    assert fresh.game.selfplay_fraction != 0.50  # no stale override


def test_threaded_path_wires_state_registration() -> None:
    """Wiring guard: _run_selfplay_threaded must register every thread's state
    (the pre-#153 behavior skipped this, so live keys like PID regret caused a
    full session teardown abandoning all in-flight games)."""
    import inspect
    src = inspect.getsource(WorkerSession._run_selfplay_threaded)
    assert "on_state_ready=self._register_live_state" in src
    assert "_clear_live_states" in src


def test_live_reco_change_without_active_state_restarts() -> None:
    """A live-key change with no running session (between sessions — the
    registry is empty) must fall back to a session restart so the change
    isn't lost. Threaded mode now registers every thread's state, so this
    fallback no longer fires there while a session is live."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None)
    session._active_reco = session._snapshot_reco({"sf_nodes": 5000, "selfplay_fraction": 0.30})
    session._live_states = []

    changed = WorkerSession._reco_changed(
        session,
        {"recommended_worker": {"sf_nodes": 5000, "selfplay_fraction": 0.50}},
        source_tag="test",
    )

    assert changed is True
    assert session._stop_selfplay is True


# ── Server-doled FEN seeding (_maybe_ingest_dole_flag) ───────────────────────

_DOLE_FEN_A = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
_DOLE_FEN_B = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2"


def _dole_session_with_list(tmp_path: Path) -> WorkerSession:
    session = _bare_worker_session()
    fen_path = tmp_path / "seeds.txt"
    fen_path.write_text("\n".join([_DOLE_FEN_A, _DOLE_FEN_B]) + "\n", encoding="utf-8")
    session.opening_fen_list_path = str(fen_path)
    session._live_states = []
    session._pending_fen_dole = []
    session._pending_sf_refute = []
    # This fixture builds a session without running __init__, so new dole
    # instance state has to be seeded here as well.
    session._dole_claim_key = None
    session._dole_claim_id = ""
    session._applied_dole_seq = {}
    session._legacy_dole_seq = {}
    return session


def test_dole_flag_stashes_seeds_when_no_active_session(tmp_path: Path) -> None:
    session = _dole_session_with_list(tmp_path)
    session._maybe_ingest_dole_flag(
        {"dole_fen_seeds": True, "recommended_worker": {"opening_fen_dole_per_iter": 1}},
    )
    # No live state → stash for the next session start, in file order.
    assert session._pending_fen_dole == [_DOLE_FEN_A, _DOLE_FEN_B]


def test_dole_flag_refills_live_queue(tmp_path: Path) -> None:
    session = _dole_session_with_list(tmp_path)
    session._live_dole_queue = []  # a session is running (single or threaded)
    session._maybe_ingest_dole_flag(
        {"dole_fen_seeds": True, "recommended_worker": {"opening_fen_dole_per_iter": 1}},
    )
    assert session._live_dole_queue == [_DOLE_FEN_A, _DOLE_FEN_B]
    assert session._pending_fen_dole == []  # refilled the live queue, not stashed


def test_dole_flag_refill_supersedes_in_place(tmp_path: Path) -> None:
    # A fresh iteration's dole replaces any undrained backlog IN PLACE, so the
    # running drainers (single state or every thread) — which all hold this same
    # list object — see the new batch without a session restart.
    session = _dole_session_with_list(tmp_path)
    live = ["stale-undrained-seed"]
    session._live_dole_queue = live
    session._maybe_ingest_dole_flag(
        {"dole_fen_seeds": True, "recommended_worker": {"opening_fen_dole_per_iter": 1}},
    )
    assert session._live_dole_queue == [_DOLE_FEN_A, _DOLE_FEN_B]
    assert session._live_dole_queue is live  # same object → drainers observe the refill
    assert session._pending_fen_dole == []


def test_dole_flag_repeats_list_n_times(tmp_path: Path) -> None:
    session = _dole_session_with_list(tmp_path)
    session._maybe_ingest_dole_flag(
        {"dole_fen_seeds": True, "recommended_worker": {"opening_fen_dole_per_iter": 2}},
    )
    assert session._pending_fen_dole == [_DOLE_FEN_A, _DOLE_FEN_B, _DOLE_FEN_A, _DOLE_FEN_B]


def test_dole_flag_noop_when_false_or_disabled(tmp_path: Path) -> None:
    session = _dole_session_with_list(tmp_path)
    # Flag false → no-op.
    session._maybe_ingest_dole_flag(
        {"dole_fen_seeds": False, "recommended_worker": {"opening_fen_dole_per_iter": 1}},
    )
    assert session._pending_fen_dole == []
    # Flag true but dole disabled (0) → no-op.
    session._maybe_ingest_dole_flag(
        {"dole_fen_seeds": True, "recommended_worker": {"opening_fen_dole_per_iter": 0}},
    )
    assert session._pending_fen_dole == []


def test_dole_flag_noop_without_fen_list(tmp_path: Path) -> None:
    session = _dole_session_with_list(tmp_path)
    session.opening_fen_list_path = None  # no list available to this worker
    session._maybe_ingest_dole_flag(
        {"dole_fen_seeds": True, "recommended_worker": {"opening_fen_dole_per_iter": 1}},
    )
    assert session._pending_fen_dole == []


def test_promote_pending_dole_returns_live_object_when_empty(tmp_path: Path) -> None:
    # THE 2026-07-12 BUG: a pause-hold session starts before any dole arrives,
    # so the pending stash is empty. The session must still hold the live queue
    # OBJECT — `or None` here orphaned it, and every mid-session dole refilled a
    # list nobody drained (seed injection silently off since the 07-11 swap).
    session = _dole_session_with_list(tmp_path)
    queue, sf_queue = session._promote_pending_dole()
    assert queue == []
    assert sf_queue == []
    assert queue is session._live_dole_queue  # same object, refillable in place
    assert sf_queue is session._live_sf_refute_queue
    session._maybe_ingest_dole_flag(
        {"dole_fen_seeds": True, "recommended_worker": {"opening_fen_dole_per_iter": 1}},
    )
    assert queue == [_DOLE_FEN_A, _DOLE_FEN_B]  # the session's handle sees the refill


def test_promote_pending_dole_moves_stash_into_live_queue(tmp_path: Path) -> None:
    session = _dole_session_with_list(tmp_path)
    session._pending_fen_dole = [_DOLE_FEN_A]
    queue, sf_queue = session._promote_pending_dole()
    assert queue == [_DOLE_FEN_A]
    assert queue is session._live_dole_queue
    assert sf_queue is session._live_sf_refute_queue
    assert session._pending_fen_dole == []


def test_dole_per_iter_is_watched_and_restart_keyed() -> None:
    # Completeness: the OpeningConfig reco.get in _build_selfplay_configs must be
    # watched (test_every_reco_field_is_watched); it's a restart key (mode switch).
    assert "opening_fen_dole_per_iter" in WorkerSession._RECO_RESTART_KEYS
    assert "opening_fen_sf_refute_frac" in WorkerSession._RECO_RESTART_KEYS
    assert "opening_fen_sf_refute_plies" in WorkerSession._RECO_RESTART_KEYS


def _stall_session() -> WorkerSession:
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.worker_stall_watchdog")
    session._selfplay_session_active = False
    session._hold_selfplay = False
    session._last_selfplay_progress_s = 0.0
    return session


def test_stall_watchdog_ignores_inactive_session() -> None:
    session = _stall_session()
    session._selfplay_session_active = False
    session._last_selfplay_progress_s = 0.0
    # No active session → never a stall, regardless of elapsed time.
    assert session._selfplay_stalled(now=10_000.0, timeout_s=300.0) is False


def test_stall_watchdog_fires_on_active_idle_session() -> None:
    session = _stall_session()
    session._selfplay_session_active = True
    session._hold_selfplay = False
    session._last_selfplay_progress_s = 1_000.0
    assert session._selfplay_stalled(now=1_000.0 + 301.0, timeout_s=300.0) is True
    assert session._selfplay_stalled(now=1_000.0 + 299.0, timeout_s=300.0) is False


def test_stall_watchdog_exempts_pause_hold_and_refreshes_timer() -> None:
    # A pause-hold (distributed_pause_selfplay_during_training / graceful-restart)
    # is intentionally idle for arbitrarily long; the watchdog must NOT hard-exit
    # a healthy paused worker, and must leave the timer fresh for the resume.
    session = _stall_session()
    session._selfplay_session_active = True
    session._hold_selfplay = True
    session._last_selfplay_progress_s = 1_000.0
    # Far past the timeout while held: not a stall, and the timer was refreshed.
    assert session._selfplay_stalled(now=1_000.0 + 10_000.0, timeout_s=300.0) is False
    assert session._last_selfplay_progress_s == 1_000.0 + 10_000.0


def test_fen_list_change_does_not_restart_in_dole_mode() -> None:
    """A new blind-spot FEN list must NOT tear down the selfplay session when
    seeds arrive via the dole (opening_fen_prob == 0).

    Measured 2026-07-24: the retire step rewrites the list every iteration, and
    fingerprinting it unconditionally routed every rewrite into the restart
    branch. A restart abandons every in-flight game -- 58% of ALL started games
    and ~89% of curriculum games, which are long enough to essentially never
    survive a session boundary. The dole path already re-downloads the asset and
    refills the live queue in place, so the restart bought nothing.
    """
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None, stockfish_from_server=False)
    session._active_reco = session._snapshot_reco({"opening_fen_prob": 0.0})
    session._active_assets = (None, None, None, None)
    session._live_states = [_live_state()]

    changed = WorkerSession._reco_changed(
        session,
        {
            "opening_fen_list": {"sha256": "list_v198"},
            "recommended_worker": {"opening_fen_prob": 0.0},
        },
        source_tag="test",
    )

    assert changed is False
    assert session._stop_selfplay is False


def test_fen_list_change_still_restarts_when_sampling_path_is_on() -> None:
    """With opening_fen_prob > 0 the list is baked into OpeningConfig at session
    start, so a new list genuinely needs a rebuild."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None, stockfish_from_server=False)
    session._active_reco = session._snapshot_reco({"opening_fen_prob": 0.25})
    session._active_assets = (None, None, None, None)
    session._live_states = [_live_state()]

    changed = WorkerSession._reco_changed(
        session,
        {
            "opening_fen_list": {"sha256": "list_v198"},
            "recommended_worker": {"opening_fen_prob": 0.25},
        },
        source_tag="test",
    )

    assert changed is True
    assert session._stop_selfplay is True


def test_asset_fingerprint_ignores_fen_list_only_in_dole_mode() -> None:
    session = _bare_worker_session()
    session.args = SimpleNamespace(stockfish_from_server=False)
    manifest = {"opening_fen_list": {"sha256": "abc"}}

    dole = WorkerSession._asset_fingerprint(
        session, {**manifest, "recommended_worker": {"opening_fen_prob": 0.0}},
    )
    sampled = WorkerSession._asset_fingerprint(
        session, {**manifest, "recommended_worker": {"opening_fen_prob": 0.5}},
    )

    assert dole[3] is None
    assert sampled[3] == "abc"


def test_restart_log_names_the_trigger(caplog: Any) -> None:
    """A recurring restart must be attributable from the log alone."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None, stockfish_from_server=False)
    session._active_reco = session._snapshot_reco({"max_plies": 300})
    session._live_states = [_live_state()]

    with caplog.at_level(logging.INFO):
        changed = WorkerSession._reco_changed(
            session,
            {"recommended_worker": {"max_plies": 400}},
            source_tag="test",
        )

    assert changed is True
    assert "restart_keys=max_plies" in caplog.text


def _manifest_at(tmp_path: Path, sha: str) -> Path:
    """Write a minimal publish manifest and return its path."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"model": {"sha256": sha}}), encoding="utf-8")
    return path


def test_model_watch_thread_updates_the_tag_with_no_on_step_caller(tmp_path: Path) -> None:
    """The regression: freshness must not depend on any selfplay thread.

    The threaded path wires on_step to selfplay thread 0 only, and joins its
    future only at session end — so when that thread died, the worker uploaded
    under a frozen model_sha indefinitely and the trainer discarded the lot
    (2026-07-24). Here NOTHING calls on_step: the swap must come from the watch
    thread alone, driven end-to-end through _start_model_watch_thread.
    """
    session = _bare_worker_session()
    cast(Any, session).inference_client = object()  # broker-backed: tag-only swap
    session.upload_buf = _BufferedUpload()
    session._upload_buf_lock = threading.Lock()
    session._model_watch_started = False
    session._selfplay_session_active = True

    def _fake_swap(manifest: dict) -> None:
        sha = str(manifest.get("model", {}).get("sha256", ""))
        if sha and sha != session.model_sha:
            session.model_sha = sha

    cast(Any, session)._swap_model_from_manifest = _fake_swap
    cast(Any, session)._reco_changed = lambda *_a, **_k: False
    cast(Any, session)._periodic_manifest_poll = lambda: None
    cast(Any, session)._maybe_log_dispatcher_stats = lambda _n: None
    cast(Any, session)._maybe_log_broker_client_stats = lambda _n: None
    session._manifest_path = _manifest_at(tmp_path, "new-sha")

    import os as _os

    _os.environ["CAE_WORKER_MODEL_WATCH_S"] = "0.05"
    try:
        session._start_model_watch_thread()
        deadline = time.time() + 5.0
        while session.model_sha != "new-sha" and time.time() < deadline:
            time.sleep(0.02)
    finally:
        _os.environ.pop("CAE_WORKER_MODEL_WATCH_S", None)

    assert session.model_sha == "new-sha", (
        "watch thread did not pick up the published model with no on_step caller"
    )


def test_run_selfplay_starts_the_model_watch_thread() -> None:
    """Pin the wiring: the watch thread is useless if nothing starts it.

    _run_selfplay is too heavy to drive end-to-end here, and this line is the
    single point where a session gains its freshness guarantee — assert it at
    the source level rather than leaving it uncovered.
    """
    import inspect

    src = inspect.getsource(WorkerSession._run_selfplay)
    assert "self._start_model_watch_thread()" in src


def test_check_model_update_skips_a_reentrant_call() -> None:
    """A contended call returns immediately instead of blocking a selfplay thread.

    on_step (per ply) and the watch thread both call this; whoever holds the
    lock is already doing the identical work, and blocking would park a game
    thread behind a model download. Driven from a separate thread with a join
    timeout so that turning the acquire back into a blocking one fails fast
    here instead of hanging the suite.
    """
    session = _bare_worker_session()
    called: list[int] = []
    cast(Any, session)._check_model_update_locked = lambda: called.append(1)

    session._model_watch_lock.acquire()
    try:
        worker = threading.Thread(target=session._check_model_update, daemon=True)
        worker.start()
        worker.join(timeout=5.0)
        assert not worker.is_alive(), (
            "contended call blocked instead of skipping — acquire is not non-blocking"
        )
    finally:
        session._model_watch_lock.release()

    assert called == [], "held lock must skip the body, not queue behind it"


def test_check_model_update_releases_its_lock_on_failure() -> None:
    """A raising body must not wedge every future freshness check."""
    session = _bare_worker_session()

    def _boom() -> None:
        raise RuntimeError("poll exploded")

    session._check_model_update_locked = _boom
    with pytest.raises(RuntimeError, match="poll exploded"):
        session._check_model_update()

    assert session._model_watch_lock.acquire(blocking=False), "lock leaked"
    session._model_watch_lock.release()


def test_stale_model_tag_alarms_once_past_the_threshold(caplog) -> None:
    session = _bare_worker_session()
    session.model_sha = "old-sha"

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        session._manifest_path = _manifest_at(Path(td), "published-sha")
        # First observation only arms the timer — no alarm yet.
        session._check_model_freshness()
        assert session._model_stale_since_s is not None
        assert "STALE" not in caplog.text

        session._model_stale_since_s = time.time() - (worker_mod._MODEL_STALE_ALARM_S + 1.0)
        with caplog.at_level(logging.ERROR):
            session._check_model_freshness()
        assert "model tag STALE" in caplog.text

        # Latched: a second pass must not spam a line per poll.
        caplog.clear()
        with caplog.at_level(logging.ERROR):
            session._check_model_freshness()
        assert "model tag STALE" not in caplog.text


def test_matching_model_tag_clears_the_stale_timer(caplog) -> None:
    session = _bare_worker_session()
    session.model_sha = "published-sha"
    session._model_stale_since_s = time.time() - 10_000.0
    session._model_stale_alarmed = True

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        session._manifest_path = _manifest_at(Path(td), "published-sha")
        with caplog.at_level(logging.ERROR):
            session._check_model_freshness()

    assert session._model_stale_since_s is None
    assert session._model_stale_alarmed is False
    assert "STALE" not in caplog.text


def test_model_watch_thread_survives_a_failing_iteration(caplog) -> None:
    """One bad poll must not take down the watch thread.

    A thread that dies on its first exception would recreate the single point
    of failure this thread exists to remove.
    """
    session = _bare_worker_session()
    session._model_watch_started = False
    session._selfplay_session_active = True
    calls: list[int] = []

    def _flaky() -> None:
        calls.append(len(calls))
        if len(calls) == 1:
            raise RuntimeError("transient")

    cast(Any, session)._check_model_update = _flaky
    cast(Any, session)._check_model_freshness = lambda: None

    import os as _os

    _os.environ["CAE_WORKER_MODEL_WATCH_S"] = "0.05"
    try:
        with caplog.at_level(logging.WARNING):
            session._start_model_watch_thread()
            deadline = time.time() + 5.0
            while len(calls) < 3 and time.time() < deadline:
                time.sleep(0.02)
    finally:
        _os.environ.pop("CAE_WORKER_MODEL_WATCH_S", None)

    assert len(calls) >= 3, "watch thread stopped after the failing iteration"
    assert "model watch iteration failed" in caplog.text


def test_model_watch_loop_also_runs_the_freshness_alarm() -> None:
    """The alarm is the only symptom that survives every swallowed swap error.

    Without this, dropping _check_model_freshness from the loop would leave the
    unit test for the alarm itself passing while nothing ever calls it live.
    """
    session = _bare_worker_session()
    session._model_watch_started = False
    session._selfplay_session_active = True
    freshness_calls: list[int] = []
    cast(Any, session)._check_model_update = lambda: None
    cast(Any, session)._check_model_freshness = lambda: freshness_calls.append(1)

    import os as _os

    _os.environ["CAE_WORKER_MODEL_WATCH_S"] = "0.05"
    try:
        session._start_model_watch_thread()
        deadline = time.time() + 5.0
        while not freshness_calls and time.time() < deadline:
            time.sleep(0.02)
    finally:
        _os.environ.pop("CAE_WORKER_MODEL_WATCH_S", None)

    assert freshness_calls, "watch loop never ran the staleness alarm"


def test_model_watch_thread_idles_between_sessions() -> None:
    """Between sessions the outer run() loop owns the manifest — stay out."""
    session = _bare_worker_session()
    session._model_watch_started = False
    session._selfplay_session_active = False
    calls: list[int] = []
    session._check_model_update = lambda: calls.append(1)
    session._check_model_freshness = lambda: None

    import os as _os

    _os.environ["CAE_WORKER_MODEL_WATCH_S"] = "0.05"
    try:
        session._start_model_watch_thread()
        time.sleep(0.4)
    finally:
        _os.environ.pop("CAE_WORKER_MODEL_WATCH_S", None)

    assert calls == []


def test_model_watch_thread_starts_only_once() -> None:
    session = _bare_worker_session()
    session._model_watch_started = False
    session._selfplay_session_active = False
    session._check_model_update = lambda: None
    session._check_model_freshness = lambda: None

    before = threading.active_count()
    session._start_model_watch_thread()
    session._start_model_watch_thread()
    session._start_model_watch_thread()
    assert threading.active_count() - before == 1
