"""Strict finite producer contracts; synthetic tables/transport, no real TB/SF/NN.

These tests are preparation for separately admitted CPU validation. No test
needs a checkpoint, an engine executable, tablebase files or a CUDA device.
"""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypedDict, cast
from typing_extensions import NotRequired, Unpack

import chess
import chess.syzygy
import numpy as np
import pytest

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.inference import BatchEvaluator
from chess_anti_engine.model import ModelConfig
from chess_anti_engine.tune.distributed_runtime import build_recommended_worker
from chess_anti_engine.moves import POLICY_SIZE, move_to_index, move_to_index_for_encoding
from chess_anti_engine.selfplay import finalize as finalize_module
from chess_anti_engine.selfplay import manager, state as state_module
from chess_anti_engine.selfplay import stockfish_turn as sf_turn
from chess_anti_engine.selfplay.config import DiffFocusConfig, GameConfig, SearchConfig
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.selfplay.state import SelfplayState, _NetRecord
from chess_anti_engine.selfplay.provenance import GameProvenance
from chess_anti_engine.selfplay.strict_tb import (
    StrictSelfplayTablebase, full_history_board, sf_history_kwargs,
)
from chess_anti_engine.stockfish import pool as pool_module
from chess_anti_engine.stockfish.uci import StockfishUCI
from chess_anti_engine.tablebase import MatchTablebaseError
from chess_anti_engine.tune.trainable_config_ops import (
    _play_batch_kwargs, _reload_yaml_into_config, restart_required_config_keys,
)
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults
from chess_anti_engine.worker import WorkerSession
from tests.test_worker_model_update import _bare_worker_session


class _StrictSettings(TypedDict):
    syzygy_outcome_mode: str
    syzygy_path: str
    syzygy_in_search: bool
    syzygy_adjudicate: bool
    syzygy_adjudicate_fraction: float
    syzygy_rescore_policy: bool


STRICT: _StrictSettings = {
    "syzygy_outcome_mode": "rule50_match_v1", "syzygy_path": "synthetic-full-pair",
    "syzygy_in_search": True, "syzygy_adjudicate": True,
    "syzygy_adjudicate_fraction": 1.0, "syzygy_rescore_policy": True,
}
ROOT = "4k1n1/8/8/8/8/8/8/3QK1N1 w - - 0 1"
CYCLE = ("g1f3", "g8f6", "f3g1", "f6g8")


class SyntheticTables:
    def __init__(self, *, wdl=2, dtz=3):
        self.wdl: dict[str, object] = {"KQQQQvK": None}
        self.dtz: dict[str, object] = {"KQQQQvK": None}
        self.wdl_value, self.dtz_value = wdl, dtz
        self.calls: list[tuple[str, str]] = []
        self.closed = 0

    def probe_wdl(self, board):
        self.calls.append(("wdl", board.fen()))
        return self.wdl_value

    def probe_dtz(self, board):
        self.calls.append(("dtz", board.fen()))
        return self.dtz_value

    def close(self):
        self.closed += 1


def session(tables=None) -> Any:
    owner = object.__new__(StrictSelfplayTablebase)
    owner.tablebase = cast(chess.syzygy.Tablebase, cast(object, tables if tables is not None else SyntheticTables()))
    owner._closed = False
    owner.asset_inventory_sha256 = "a" * 64
    return owner


def history(tokens=CYCLE * 2, root=ROOT):
    board = chess.Board(root)
    for token in tokens:
        board.push_uci(token)
    return board


def state_for(board, *, opening=None, owner=None):
    start = board.copy(stack=True) if opening is None else opening.copy(stack=True)
    replay = start.copy(stack=True)
    actions = []
    for move in board.move_stack[len(start.move_stack):]:
        actions.append(move_to_index(move, replay))
        replay.push(move)
    return cast(Any, SimpleNamespace(
        starting_boards=[start], cboards=[CBoard.from_board(board)],
        move_idx_history=[actions], game=GameConfig(**STRICT),
        strict_tb=owner or session(), strict_tb_witnesses={},
        strict_tb_policy_abstained={}, batch_size=1, done_arr=[0], finalized_arr=[0],
        tb_result_arr=[None], pending_sf_moves={},
    ))


def record(board, *, offset=0):
    probs = np.zeros(POLICY_SIZE, np.float32)
    for move in board.legal_moves:
        probs[move_to_index(move, board)] = 1
    probs /= probs.sum()
    return _NetRecord(
        x=np.zeros((175, 8, 8), np.float32), policy_probs=probs,
        net_wdl_est=np.array([0, 1, 0], np.float32),
        search_wdl_est=np.array([0, 1, 0], np.float32), pov_color=board.turn,
        ply_index=board.ply(), has_policy=True, priority=1, sample_weight=1,
        keep_prob=1, move_offset=offset,
    )


def test_schema_trial_direct_and_worker_build_realize_the_same_mode():
    flat = flatten_run_config_defaults({"selfplay": STRICT})
    tc = TrialConfig.from_dict(flat)
    assert _play_batch_kwargs(tc)["game"].syzygy_outcome_mode == "rule50_match_v1"
    reco = build_recommended_worker(config=flat, model_cfg=ModelConfig(),
                                    sf_nodes=5000, mcts_simulations=64)
    assert reco["syzygy_outcome_mode"] == tc.syzygy_outcome_mode
    cfgs, sf_args = WorkerSession._build_selfplay_configs(
        _bare_worker_session(), reco,
    )
    assert cfgs["game"].syzygy_outcome_mode == tc.syzygy_outcome_mode
    assert sf_args[-1] == STRICT["syzygy_path"]
    assert "syzygy_outcome_mode" in WorkerSession._RECO_RESTART_KEYS
    assert "syzygy_outcome_mode" in restart_required_config_keys()


@pytest.mark.parametrize("bad", [True, 1, "rule50", "RULE50_MATCH_V1", ""])
def test_bad_mode_is_refused_by_yaml_and_direct_trial(bad):
    for parse in (flatten_run_config_defaults, TrialConfig.from_dict):
        with pytest.raises(ValueError, match="syzygy_outcome_mode"):
            parse({"syzygy_outcome_mode": bad})


def test_mode_reload_requires_restart_but_startup_overlay_applies(tmp_path, caplog):
    import yaml
    path = tmp_path / "strict.yaml"
    path.write_text(yaml.safe_dump({"selfplay": STRICT}))
    config = {"syzygy_outcome_mode": "theoretical_wdl"}
    _reload_yaml_into_config(config, str(path), live_reload=True)
    assert config["syzygy_outcome_mode"] == "theoretical_wdl"
    assert "syzygy_outcome_mode" in caplog.text
    assert "requires restart" in caplog.text
    _reload_yaml_into_config(config, str(path), live_reload=False)
    assert config["syzygy_outcome_mode"] == "rule50_match_v1"


@pytest.mark.parametrize("override", [
    {"syzygy_path": None}, {"syzygy_in_search": False}, {"syzygy_adjudicate": False},
    {"syzygy_in_search": "true"}, {"syzygy_adjudicate_fraction": .5},
    {"syzygy_adjudicate_fraction": True}, {"syzygy_rescore_policy": "false"},
    {"stockfish_syzygy_path": "wdl-only"},
])
def test_incoherent_strict_config_refuses_before_production(override):
    with pytest.raises(ValueError, match=r"rule50_match_v1|syzygy_"):
        TrialConfig.from_dict({**STRICT, **override})


def test_full_history_keeps_opening_claim_boundary_separate_from_encoder_replay():
    # Claim by the next move at seven plies, actual third occurrence at eight.
    opening = history(CYCLE + CYCLE[:2])
    final = history(CYCLE + CYCLE[:3])
    state = state_for(final, opening=opening)
    state.has_c_ply = True
    state.boards = [opening.copy()]
    full = full_history_board(state, 0)
    encoded_replay = SelfplayState.replay_board(state, 0)
    assert full.fen() == encoded_replay.fen()
    assert full.can_claim_threefold_repetition()
    assert not encoded_replay.can_claim_threefold_repetition()
    assert sf_history_kwargs(state, 0).get("root_moves") == CYCLE + CYCLE[:3]
    assert len(opening.move_stack) == 6  # reconstruction never mutates source
    state.cboards[0] = CBoard.from_board(chess.Board(final.fen().replace(" 7 4", " 0 1")))
    with pytest.raises(ValueError, match="clocks disagree"):
        full_history_board(state, 0)


@pytest.mark.parametrize("tokens", [CYCLE + CYCLE[:3], CYCLE * 2])
def test_claim_has_natural_precedence_without_a_tb_query(tokens):
    tables = SyntheticTables()
    witness = session(tables).resolve(history(tokens), move_offset=1)
    assert witness.origin == "NATURAL"
    assert witness.result == "1/2-1/2"
    assert witness.wdl is witness.dtz is None
    assert tables.calls == []


@pytest.mark.parametrize(("wdl", "dtz", "result"), [(0, 0, "1/2-1/2"), (1, 101, "1/2-1/2"),
                                          (-1, -101, "1/2-1/2"), (2, 3, "1-0"), (-2, -3, "0-1")])
def test_reset_clock_results_preserve_raw_classes_and_witness(wdl, dtz, result):
    witness = session(SyntheticTables(wdl=wdl, dtz=dtz)).resolve(chess.Board(ROOT), move_offset=0)
    assert witness.origin == "TB_ADJUDICATED"
    assert witness.result == result
    assert (witness.wdl, witness.dtz, witness.turn, witness.halfmove_clock) == (wdl, dtz, True, 0)
    assert witness.mode == "rule50_match_v1"
    assert witness.asset_inventory_sha256 == "a" * 64


def test_positive_clock_decisive_root_plays_on_in_actual_adjudicator():
    board = history(("g1f3",))
    state = state_for(board)
    assert manager._tb_adjudicate_active_games(state) == 0
    assert state.done_arr == [0]
    assert state.tb_result_arr == [None]
    state.strict_tb = session(SyntheticTables(wdl=-1, dtz=-101))
    assert manager._tb_adjudicate_active_games(state) == 1
    assert state.done_arr == [1]
    assert state.tb_result_arr == ["1/2-1/2"]
    assert state.strict_tb_witnesses[0].origin == "TB_ADJUDICATED"


@pytest.mark.parametrize("missing", ["wdl", "dtz"])
def test_missing_eligible_probe_never_becomes_fallback(missing):
    tables = SyntheticTables()
    setattr(tables, "probe_" + missing, lambda _board: (_ for _ in ()).throw(KeyError("missing")))
    with pytest.raises(MatchTablebaseError, match="missing eligible"):
        session(tables).resolve(chess.Board(ROOT), move_offset=0)


def test_zeroing_move_permits_a_decisive_result_after_unresolved_clock():
    owner = session()
    board = chess.Board("7k/8/8/8/8/8/P7/KQ6 w - - 12 8")
    assert owner.resolve(board, move_offset=0).result == "*"
    board.push_uci("a2a3")
    assert owner.resolve(board, move_offset=1).result == "0-1"  # synthetic STM win is black


def test_policy_missing_child_is_error_even_if_root_would_abstain(monkeypatch):
    board = history(("g1f3",))
    tables = SyntheticTables()
    original = tables.probe_dtz
    monkeypatch.setattr(tables, "probe_dtz", lambda child: original(child) if child.fen() == board.fen() else (_ for _ in ()).throw(KeyError("child")))
    with pytest.raises(MatchTablebaseError):
        session(tables).policy_move(board)


def test_policy_abstains_on_positive_clock_decisive_ambiguity():
    assert session().policy_move(history(("g1f3",))) is None


def test_strict_finalizer_bypasses_last_theoretical_result_and_keeps_policy_only(monkeypatch):
    board = chess.Board(ROOT)
    start = board.copy()
    rec = record(start)
    board.push_uci("g1f3")
    state = state_for(board, opening=start)
    move = chess.Move.from_uci("d1d2")
    state.strict_tb.policy_move = lambda _board: move
    monkeypatch.setattr(finalize_module, "rescore_game_samples", lambda *_a: pytest.fail("legacy outcome rewrite"))
    result, policies = finalize_module._rescore_with_syzygy(state, 0, board, [rec], "1-0")
    assert result == "1-0"
    assert policies[0].sum() == 1
    assert policies[0][move_to_index_for_encoding(move, start, policy_encoding=state.game.policy_encoding)] == 1
    state.game = replace(state.game, syzygy_rescore_policy=False)
    assert finalize_module._rescore_with_syzygy(state, 0, board, [rec], "0-1") == ("0-1", {})
    state.game = replace(state.game, syzygy_rescore_policy=True)
    rec.move_offset = 17
    with pytest.raises(ValueError, match="join mismatch"):
        finalize_module._rescore_with_syzygy(state, 0, board, [rec], "1-0")


class _FiniteOverrides(TypedDict, total=False):
    target_games: int
    stop_fn: Callable[[], bool]
    on_suspend: Callable[[SelfplayState], None]
    on_game_provenance: Callable[[GameProvenance], None] | None
    search: SearchConfig
    game: GameConfig


class _FiniteArgs(TypedDict):
    model: None
    evaluator: BatchEvaluator
    device: str
    rng: np.random.Generator
    stockfish: StockfishUCI
    games: int
    target_games: int
    on_game_provenance: Callable[[GameProvenance], None] | None
    search: SearchConfig
    opening: OpeningConfig
    diff_focus: DiffFocusConfig
    game: GameConfig
    on_state_ready: Callable[[SelfplayState], None] | None
    stop_fn: NotRequired[Callable[[], bool]]
    on_suspend: NotRequired[Callable[[SelfplayState], None]]


def _finite(monkeypatch, *, start=None, owner=None, ready: Callable[[SelfplayState], None] | None = None,
            **kwargs: Unpack[_FiniteOverrides]):
    owner = owner or session()
    monkeypatch.setattr(state_module, "StrictSelfplayTablebase", lambda _path: owner)
    start = start if start is not None else chess.Board()
    monkeypatch.setattr(state_module, "resolve_slot_opening", lambda **_kw: SimpleNamespace(board=start.copy(), source="strict-test"))
    sf = SimpleNamespace(
        nodes=1, syzygy_path=STRICT["syzygy_path"], syzygy_50_move_rule=True,
        syzygy_probe_limit=6, syzygy_ready_after_requests=True,
        search=lambda *_a, **_kw: pytest.fail("no SF queries expected"),
    )
    observed: list[GameProvenance] = []
    params: _FiniteArgs = {
        "model": None, "evaluator": cast(BatchEvaluator, cast(object, SimpleNamespace())), "device": "cpu", "rng": np.random.default_rng(7),
        "stockfish": cast(StockfishUCI, cast(object, sf)), "games": 1, "target_games": 1, "on_game_provenance": observed.append,
        "search": SearchConfig(mcts_type="gumbel", simulations=1),
        "opening": OpeningConfig(random_start_plies=0), "diff_focus": DiffFocusConfig(enabled=False),
        "game": GameConfig(**STRICT, max_plies=0, selfplay_fraction=1), "on_state_ready": ready,
    }
    params.update(kwargs)
    result = manager.play_batch(**params)
    return result, observed, owner


def test_cap_is_censored_not_default_draw_no_rows_no_sf_query(monkeypatch):
    (samples, stats), observed, owner = _finite(monkeypatch)
    assert samples == []
    assert stats.censored_games == 1
    assert stats.games == 1
    assert stats.w == stats.d == stats.l == 0
    audit = observed[0]
    assert audit.resolution == "unresolved"
    assert audit.result == audit.resolved_result == "*"
    assert audit.cap_reached
    assert audit.retained_samples == 0
    assert audit.tablebase_witness is not None
    assert audit.tablebase_witness.origin == "UNRESOLVED"
    assert owner.tablebase.closed == 1


def test_zero_row_tb_result_is_retained_and_not_replaced(monkeypatch):
    (samples, stats), observed, owner = _finite(monkeypatch, start=chess.Board(ROOT))
    assert samples == []
    assert stats.games == 1
    assert len(observed) == 1
    assert stats.censored_games == 0
    assert stats.tb_adjudicated_games == 1
    assert observed[0].resolution == "tablebase_adjudication"
    assert observed[0].natural_result == observed[0].native_result == "*"
    assert observed[0].tablebase_witness is not None
    assert observed[0].tablebase_witness.origin == "TB_ADJUDICATED"
    assert observed[0].tablebase_rescore_result is None
    assert owner.tablebase.closed == 1


def test_strict_call_never_fetches_or_closes_a_legacy_global_handle(monkeypatch):
    from chess_anti_engine import tablebase
    monkeypatch.setattr(tablebase, "get_tablebase", lambda *_a: pytest.fail("legacy cache accessed"))
    (_samples, _stats), observed, owner = _finite(monkeypatch, start=chess.Board(ROOT))
    assert observed[0].tablebase_witness is not None
    assert observed[0].tablebase_witness.origin == "TB_ADJUDICATED"
    assert owner.tablebase.closed == 1


def test_actual_finalizer_delivers_strict_terminal_wdl_in_each_record_pov(monkeypatch):
    start = chess.Board("7k/8/8/8/8/8/P7/KQ6 w - - 12 8")
    def inject(state):
        board = start.copy()
        for offset, token in enumerate(("b1c1", "h8g8", "a2a3")):
            state.samples_per_game[0].append(record(board, offset=offset))
            move = chess.Move.from_uci(token)
            action = move_to_index(move, board)
            state.move_idx_history[0].append(action)
            state.cboards[0].push_index(action)
            board.push(move)
        state.done_arr[0] = 1
    (samples, stats), observed, _owner = _finite(
        monkeypatch, start=start, ready=inject,
        game=replace(GameConfig(**STRICT), syzygy_rescore_policy=False, max_plies=10,
                     selfplay_fraction=1),
    )
    assert stats.censored_games == 0
    assert observed[0].result == "0-1"  # fake TB says final black STM wins
    assert observed[0].resolution == "tablebase_adjudication"
    assert observed[0].natural_result == "*"
    assert [sample.wdl_target for sample in samples] == [2, 0, 2]
    assert len(observed[0].records) == len(samples) == 3


@pytest.mark.parametrize("where", ["state_ready", "provenance", "probe"])
def test_owned_handle_closes_once_on_callback_or_probe_exception(monkeypatch, where):
    owner = session()
    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic exception")
    args = {}
    if where == "state_ready":
        args["ready"] = fail
    elif where == "provenance":
        args["on_game_provenance"] = fail
    else:
        owner.resolve = fail
    with pytest.raises(RuntimeError, match="synthetic"):
        _finite(monkeypatch, owner=owner, **args)
    assert owner.tablebase.closed == 1


def test_owned_handle_closes_if_state_construction_after_open_fails(monkeypatch):
    owner = session()
    def fail(*_args, **_kwargs):
        raise RuntimeError("probe constructor failed")
    monkeypatch.setattr(state_module, "SyzygyProbe", fail)
    with pytest.raises(RuntimeError, match="probe constructor"):
        _finite(monkeypatch, owner=owner)
    assert owner.tablebase.closed == 1


@pytest.mark.parametrize("kwargs", [
    {"target_games": 0, "stop_fn": lambda: False}, {"on_suspend": lambda _state: None},
    {"on_game_provenance": None}, {"search": SearchConfig(mcts_type="puct")},
])
def test_unsupported_routes_refuse_before_state_creation(monkeypatch, kwargs):
    monkeypatch.setattr(SelfplayState, "create", lambda **_kw: pytest.fail("opened state"))
    with pytest.raises(ValueError, match="rule50_match_v1"):
        _finite(monkeypatch, **kwargs)


def test_worker_continuous_mode_refuses_before_any_asset_sync():
    worker = _bare_worker_session()
    worker._install_shutdown_handlers = lambda: None
    worker._shutdown_requested = False
    worker._poll_manifest = lambda: {"recommended_worker": STRICT}
    worker._sync_assets = lambda manifest: pytest.fail(f"model/assets touched: {bool(manifest)}")
    worker._cleanup = lambda: None
    with pytest.raises(ValueError, match="finite play_batch only"):
        worker.run()


def test_tune_refuses_before_ray_or_process_launch(tmp_path, monkeypatch):
    from chess_anti_engine.tune import harness
    monkeypatch.setattr(harness, "_patch_ray_artifact_sync_excludes", lambda: pytest.fail("Ray touched"))
    work = tmp_path / "not-created"
    with pytest.raises(ValueError, match="finite play_batch only"):
        harness.run_tune(base_config=dict(STRICT), work_dir=work, num_samples=1)
    assert not work.exists()


def test_sf_queries_and_pending_label_capture_the_actual_full_history():
    calls = []
    sf = object.__new__(pool_module.StockfishPool)
    sf.submit = lambda fen, **kwargs: calls.append((fen, kwargs)) or Future()
    board = history(("g1f3", "g8f6"))
    state = state_for(board)
    state.stockfish = sf
    state.base_nodes = 5000
    state.last_net_full = [True]
    state.samples_per_game = [[record(history(("g1f3",)), offset=0)]]
    state.pending_sf_labels = []
    sf_turn.submit_async_sf_label_queries(state, [0])
    pending = state.pending_sf_labels[0]
    assert calls[0][1]["root_moves"] == ("g1f3", "g8f6")
    # Actual P1 remains immutable after another move is played in the slot.
    move = chess.Move.from_uci("f3g1")
    state.move_idx_history[0].append(move_to_index(move, board))
    state.cboards[0].push_index(state.move_idx_history[0][-1])
    assert sf_turn._pending_sf_history(pending).get("root_moves") == ("g1f3", "g8f6")
    assert sf_history_kwargs(state, 0).get("root_moves") == ("g1f3", "g8f6", "f3g1")
    state.game = replace(state.game, sf_label_escalate_q_gap=.1,
                         sf_label_escalate_nodes=20000, sf_label_escalate_max_per_game=1,
                         sf_wdl_use_cp_logistic=False)
    state.sf_label_escalations = [0]
    result = SimpleNamespace(wdl=np.array([1., 0., 0.]), cp=None, mate=None)
    assert sf_turn._maybe_submit_label_escalation(state, pending, result)
    assert calls[-1][1]["fresh"] is True
    assert calls[-1][1]["nodes"] == 20000
    assert calls[-1][1]["root_moves"] == ("g1f3", "g8f6")
    assert calls[-1][0] == pending.query_fen
    with pytest.raises(ValueError, match="resumed label"):
        sf_turn.queue_resumed_sf_labels(state, 0, [], originals={})


def test_sync_strict_transport_never_catches_typeerror_as_bare_fen_fallback():
    calls = []
    def legacy_only(fen, *, nodes=None, syzygy_path=None):
        calls.append((fen, nodes, syzygy_path))
    with pytest.raises(TypeError):
        sf_turn._search_stockfish_sync(
            SimpleNamespace(search=legacy_only), ROOT, nodes=5000, syzygy_path="full",
            root_fen=ROOT, root_moves=(),
        )
    assert calls == []


def test_pool_initial_and_replacement_engines_keep_explicit_tb_options(monkeypatch):
    engines = []
    class InertEngine:
        def __init__(self, _path, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            engines.append(self)
        def close(self):
            self.closed = True
    monkeypatch.setattr(pool_module, "StockfishUCI", InertEngine)
    pool = pool_module.StockfishPool(
        path="unused", nodes=5000, num_workers=1, syzygy_path="full",
        syzygy_50_move_rule=True, syzygy_probe_limit=6,
    )
    try:
        pool._replace_engine(engines[0])
        assert len(engines) == 2
        for engine in engines:
            assert engine.kwargs["syzygy_path"] == "full"
            assert engine.kwargs["syzygy_50_move_rule"] is True
            assert engine.kwargs["syzygy_probe_limit"] == 6
    finally:
        pool.close()


def test_strict_open_checks_canonical_inventory_and_closes_on_missing(monkeypatch, tmp_path):
    from chess_anti_engine.selfplay import strict_tb
    tables = SyntheticTables()
    tables.wdl = {"KQvK": SimpleNamespace(path=str(tmp_path / "KQvK.rtbw"))}
    tables.dtz = {}
    monkeypatch.setattr(strict_tb, "open_strict_match_tablebase", lambda *_a, **_kw: tables)
    monkeypatch.setattr(chess.syzygy, "tablenames", lambda **_kw: iter(["KQvK"]))
    with pytest.raises(MatchTablebaseError, match="missing DTZ"):
        StrictSelfplayTablebase("synthetic")
    assert tables.closed == 1


def test_owned_inventory_is_metadata_identity_and_close_is_idempotent(monkeypatch, tmp_path):
    from chess_anti_engine.selfplay import strict_tb
    tables = SyntheticTables()
    for kind in ("wdl", "dtz"):
        path = Path(tmp_path) / kind
        path.write_bytes(b"synthetic; never probed")
        # Real mappings contain mirrored aliases for the same file.
        setattr(tables, kind, {name: SimpleNamespace(path=str(path)) for name in ("KQvK", "KvKQ")})
    monkeypatch.setattr(strict_tb, "open_strict_match_tablebase", lambda *_a, **_kw: tables)
    monkeypatch.setattr(chess.syzygy, "tablenames", lambda **_kw: iter(["KQvK"]))
    owner = StrictSelfplayTablebase("synthetic")
    assert len(owner.asset_inventory_sha256) == 64
    owner.close()
    owner.close()
    assert tables.closed == 1


def test_legacy_mode_is_default_and_keeps_original_syzygy_rescore(monkeypatch):
    assert GameConfig().syzygy_outcome_mode == TrialConfig.from_dict({}).syzygy_outcome_mode == "theoretical_wdl"
    state = state_for(chess.Board(ROOT))
    state.strict_tb = None
    state.game = replace(state.game, syzygy_outcome_mode="theoretical_wdl", syzygy_rescore_policy=False)
    state.has_c_ply = True
    monkeypatch.setattr(finalize_module, "rescore_game_samples", lambda *_a: "0-1")
    assert finalize_module._rescore_with_syzygy(state, 0, chess.Board(ROOT), [], "1-0") == ("0-1", {})
    assert sf_history_kwargs(state, 0) == {}
    assert asdict(state.game)["syzygy_outcome_mode"] == "theoretical_wdl"
