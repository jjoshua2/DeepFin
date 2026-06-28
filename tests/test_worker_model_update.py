from __future__ import annotations

import json
import logging
import re
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
  # Test manifests carry no stockfish/opening-book assets, so the session-start
  # fingerprint is all-None; match it so the asset-change gate stays quiet.
    session._active_assets = (None, None, None)
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
    session._active_state = _live_state()
    cast(Any, session).sf = SimpleNamespace(set_nodes=lambda n: captured.update(nodes=n))

    changed = WorkerSession._reco_changed(
        session,
        {"recommended_worker": {"sf_nodes": 6000, "selfplay_fraction": 0.50}},
        source_tag="test",
    )

    assert changed is False
    assert session._stop_selfplay is False
    st = cast(Any, session._active_state)
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
    session._active_state = _live_state()  # max_plies=137
    cast(Any, session).sf = SimpleNamespace(set_nodes=lambda _n: None)

    applied = WorkerSession._apply_live_reco(
        session,
        {"sf_nodes": 5000, "selfplay_fraction": 0.50, "max_plies": 999},
    )

    assert applied is True
    st = cast(Any, session._active_state)
    assert st.game.selfplay_fraction == 0.50  # live field applied
    assert st.game.max_plies == 137  # session-fixed field preserved


def test_apply_live_reco_falls_back_to_restart_on_build_error(caplog: Any) -> None:
    """Codex review: a malformed reco that makes config construction raise must
    NOT be swallowed silently — _apply_live_reco logs and returns False so the
    caller restarts (re-parsing the reco at bring-up), rather than pinning the
    worker at the old config."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None)
    session._active_state = _live_state()
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
        "opponent_wdl_regret_limit": (0.04, 0.02, lambda st: st.opponent.wdl_regret_limit),
        "sf_nodes": (5000, 7000, lambda st: st.base_nodes),
    }
    assert set(cases) == set(WorkerSession._RECO_LIVE_KEYS), (
        "update test_every_live_key_is_transplanted for the new live key(s)"
    )
    for key, (base, changed, get) in cases.items():
        session = _bare_worker_session()
        session.args = SimpleNamespace(sf_nodes=None)
        session._active_state = _live_state()
        cast(Any, session).sf = SimpleNamespace(set_nodes=lambda _n: None)
        baseline = {"sf_nodes": 5000, key: base}
        applied = WorkerSession._apply_live_reco(session, {**baseline, key: changed})
        assert applied is True
        assert get(cast(Any, session._active_state)) == changed, f"live key {key} not transplanted"


def test_sf_multipv_change_triggers_restart() -> None:
    """Codex #81: sf_multipv is applied only at engine (re)init, so a change
    must restart even when bundled with a live-only sf_nodes update."""
    assert "sf_multipv" in WorkerSession._RECO_RESTART_KEYS
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None)
    session._active_reco = session._snapshot_reco({"sf_nodes": 5000, "sf_multipv": 5})
    session._active_state = _live_state()

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
    session._active_assets = ("sha_old", None, None)
    session._active_state = _live_state()

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


def test_sync_stockfish_reinits_on_binary_change(monkeypatch: Any) -> None:
    """Codex review: a hot-swapped SF binary (new resolved path) must re-init the
    engine — otherwise the asset-fingerprint restart downloads the new binary but
    keeps running the old one."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(
        stockfish_path="sf_v1", stockfish_from_server=False, sf_workers=1, sf_nice=0,
    )
    cast(Any, session).sf = None
    session.sf_multipv_active = None
    session.sf_syzygy_path_active = None
    session.sf_path_active = None

    built: list[str] = []

    class _FakeSF:
        def __init__(self, path: str, **_kw: Any) -> None:
            built.append(path)

        def close(self) -> None:
            pass

        def set_nodes(self, _n: int) -> None:
            pass

    monkeypatch.setattr(worker_mod, "StockfishUCI", _FakeSF)

    WorkerSession._sync_stockfish(session, {}, 5000, 5)
    WorkerSession._sync_stockfish(session, {}, 5000, 5)  # same path -> no rebuild
    assert built == ["sf_v1"]
    session.args.stockfish_path = "sf_v2"  # binary hot-swap
    WorkerSession._sync_stockfish(session, {}, 5000, 5)
    assert built == ["sf_v1", "sf_v2"]


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


def test_live_reco_change_without_active_state_restarts() -> None:
    """A live-key change with no running session (threaded mode / between
    sessions) must fall back to a session restart so the change isn't lost."""
    session = _bare_worker_session()
    session.args = SimpleNamespace(sf_nodes=None)
    session._active_reco = session._snapshot_reco({"sf_nodes": 5000, "selfplay_fraction": 0.30})
    session._active_state = None

    changed = WorkerSession._reco_changed(
        session,
        {"recommended_worker": {"sf_nodes": 5000, "selfplay_fraction": 0.50}},
        source_tag="test",
    )

    assert changed is True
    assert session._stop_selfplay is True
