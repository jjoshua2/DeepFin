"""SF budget knobs must reach a route that can honour them, or be refused.

``sf_move_nodes`` / ``sf_label_nodes_floor`` / ``sf_label_nodes_cap`` > 0 need
the ``StockfishPool`` route: on the single-engine route one query is both the
curriculum move and its label. The refusal lives at launch, on live reload
(declined, not fatal), in the worker (wait, not exit) and in ``play_batch``;
eval, which is always single-engine, neutralises the knobs instead.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
import re
import threading
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import yaml

from chess_anti_engine.model import ModelConfig
from chess_anti_engine.selfplay import manager
from chess_anti_engine.selfplay.config import GameConfig
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.uci import StockfishResult, StockfishUCI
from chess_anti_engine.tune import distributed_runtime as runtime
from chess_anti_engine.tune import trainable_init
from chess_anti_engine.tune import trainable_config_ops
from chess_anti_engine.tune.trainable_config_ops import (
    _reload_yaml_into_config,
    reject_sf_budget_topology,
)
from chess_anti_engine.tune.trainable_phases import _run_eval_games
from chess_anti_engine.tune.startup_intent import StartupIntentError
from chess_anti_engine.tune.trial_config import TrialConfig
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file
from chess_anti_engine import worker as worker_mod
from chess_anti_engine.worker import WorkerSession
from tests.selfplay_helpers import FakeStockfish, UniformPolicyValueModel

_REFUSAL = worker_mod._UnsupportedSfBudgetForTopology
_BUDGET_KEYS = (
    "sf_move_nodes", "sf_label_nodes_floor", "sf_label_nodes_cap", "sf_label_escalate_q_gap",
)


class _ReachedSetup(Exception):
    """Stop at the next real setup boundary without creating games or engines."""


# -- play_batch (direct callers) ---------------------------------------------

@pytest.mark.parametrize(('key', 'value'), [
    ('sf_move_nodes', 100), ('sf_move_nodes', 5000),
    ('sf_label_nodes_floor', 150_000), ('sf_label_nodes_cap', 200_000),
    ('sf_label_escalate_q_gap', 0.8),
])
def test_play_batch_rejects_single_engine_before_state_or_engine_work(
    monkeypatch: pytest.MonkeyPatch, key: str, value: int,
) -> None:
    def unexpected_state(**_kwargs: Any) -> None:
        pytest.fail('unsupported budget reached SelfplayState creation')

    monkeypatch.setattr(manager.SelfplayState, 'create', unexpected_state)
    # No constructor is called: the object represents the actual non-pool route
    # and cannot start an engine or issue a query during this regression test.
    engine = object.__new__(StockfishUCI)
    with pytest.raises(ValueError, match=f'{key} > 0 requires StockfishPool'):
        manager.play_batch(
            None, device='cpu', rng=np.random.default_rng(0), stockfish=engine,
            games=1, game=dataclasses.replace(GameConfig(), **{key: value}),
        )


@pytest.mark.parametrize(('pooled', 'overrides'), [
    (False, {}),
    (True, {}),
    (True, {'sf_move_nodes': 100}),
    (True, {'sf_label_nodes_floor': 150_000, 'sf_label_nodes_cap': 200_000}),
    (True, {'sf_label_escalate_q_gap': 0.8}),
])
def test_play_batch_preserves_default_and_pool_routes(
    monkeypatch: pytest.MonkeyPatch, pooled: bool, overrides: dict[str, int],
) -> None:
    seen: list[dict] = []

    def reached_state(**kwargs: Any) -> None:
        seen.append(kwargs)
        raise _ReachedSetup

    monkeypatch.setattr(manager.SelfplayState, 'create', reached_state)
    engine = object.__new__(StockfishPool if pooled else StockfishUCI)
    game = dataclasses.replace(GameConfig(), **overrides)
    with pytest.raises(_ReachedSetup):
        manager.play_batch(
            None, device='cpu', rng=np.random.default_rng(0), stockfish=engine,
            games=1, game=game,
        )
    assert len(seen) == 1
    assert seen[0]['stockfish'] is engine
    assert seen[0]['game'] is game


# -- eval (always single-engine) ---------------------------------------------

class _RecordingUCI(FakeStockfish):
    """A StockfishUCI (so play_batch takes the single-engine route) that
    records every node budget it is sent."""

    def __init__(self) -> None:
        super().__init__([0.3, 0.4, 0.3])
        self.nodes = 500
        self.sent: list[int | None] = []

    def search(
        self,
        fen: str,
        *,
        nodes: int | None = None,
        depth: int | None = None,
        syzygy_path: str | None = None,
        fresh: bool = False,
        searchmoves: Sequence[str] | None = None,
        root_fen: str | None = None,
        root_moves: Sequence[str] | None = None,
    ) -> StockfishResult:
        self.sent.append(nodes)
        return super().search(
            fen, nodes=nodes, depth=depth, syzygy_path=syzygy_path, fresh=fresh,
            searchmoves=searchmoves,
            root_fen=root_fen, root_moves=root_moves,
        )


def _eval_budgets(**budget: float) -> tuple[Counter, dict]:
    tc = TrialConfig.from_dict({
        'eval_games': 2, 'eval_max_plies': 8, 'eval_mcts_simulations': 1,
        'sf_fast_ply_node_scale': 1.0, **budget,
    })
    engine = _RecordingUCI()
    out = _run_eval_games(
        tc=tc, trainer=cast(Any, SimpleNamespace(model=UniformPolicyValueModel().eval())),
        device='cpu', rng=np.random.default_rng(0), eval_sf=engine,
    )
    return Counter(engine.sent), out


def test_eval_on_single_engine_ignores_training_budget_knobs_exactly() -> None:
    """Review #1220 MAJOR 1: eval reuses the training GameConfig on ONE
    StockfishUCI; positive budget knobs must neither kill the iteration nor
    change a single node budget versus the all-zero oracle (main's eval sent
    the same budgets because that route never read them)."""
    oracle_sent, oracle_out = _eval_budgets()
    sent, out = _eval_budgets(
        sf_move_nodes=100, sf_label_nodes_floor=50_000, sf_label_nodes_cap=60_000,
        sf_label_escalate_q_gap=0.8,
    )
    assert sum(oracle_sent.values()) > 0, 'eval issued no SF queries'
    assert sent == oracle_sent
    assert out == oracle_out


# -- launch ------------------------------------------------------------------

@pytest.mark.parametrize('key', _BUDGET_KEYS)
@pytest.mark.parametrize(('worker_sf', 'refused'), [(None, True), (1, True), (2, False), (8, False)])
def test_launch_refuses_single_engine_workers_with_a_budget_knob(
    key: str, worker_sf: int | None, refused: bool,
) -> None:
    config: dict[str, Any] = {key: 1000, 'sf_workers': 8}  # driver sf_workers never reaches workers
    if worker_sf is not None:
        config['distributed_worker_sf_workers'] = worker_sf
    if refused:
        with pytest.raises(ValueError, match=rf'refusing to start.*{key} > 0 requires StockfishPool'):
            reject_sf_budget_topology(config)
    else:
        reject_sf_budget_topology(config)
    reject_sf_budget_topology({**config, key: 0})


@pytest.mark.parametrize(('config', 'refused'), [
    ({'sf_label_nodes_floor': 150_000, 'distributed_worker_sf_workers': 1}, True),
    ({'sf_label_nodes_floor': 150_000, 'distributed_worker_sf_workers': 8}, False),
])
def test_startup_path_refuses_before_any_replay_or_worker_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config: dict, refused: bool,
) -> None:
    def reached(**_kwargs: Any) -> Path:
        raise _ReachedSetup

    monkeypatch.setattr(trainable_init, '_trial_replay_shard_dir', reached)
    expected = ValueError if refused else _ReachedSetup
    with pytest.raises(expected):
        trainable_init._init_replay_buffers(
            tc=cast(Any, SimpleNamespace(replay_window_start=1)), config=dict(config),
            restore=cast(Any, None), trial_dir=tmp_path, work_dir=tmp_path,
            rng=None, ckpt=None,
        )


# -- live reload -------------------------------------------------------------

_PRODUCTION = Path(__file__).resolve().parents[1] / 'configs' / 'pbt2_small.yaml'


def _set(doc: dict, key: str, value: object) -> None:
    for section in doc.values():
        if isinstance(section, dict) and key in section:
            section[key] = value
            return
    # Not shipped by the production yaml (e.g. the label floor): add it beside
    # sf_move_nodes so it lands in the section the loader expects.
    for section in doc.values():
        if isinstance(section, dict) and 'sf_move_nodes' in section:
            section[key] = value
            return
    raise AssertionError(f'{key} has no section in the production yaml')


def _reload(tmp_path: Path, base: dict, edits: dict, caplog) -> tuple[dict, dict, list[str]]:
    """Launch on ``base`` edits, then live-reload with ``edits`` on top."""
    raw = load_yaml_file(str(_PRODUCTION))
    for k, v in base.items():
        _set(raw, k, v)
    start = tmp_path / 'start.yaml'
    start.write_text(yaml.safe_dump(raw), encoding='utf-8')
    config = flatten_run_config_defaults(load_yaml_file(str(start)))
    launched = dict(config)
    doc = copy.deepcopy(raw)
    for k, v in edits.items():
        _set(doc, k, v)
    edited = tmp_path / 'edited.yaml'
    edited.write_text(yaml.safe_dump(doc), encoding='utf-8')
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger='chess_anti_engine.tune.trainable_config_ops'):
        _reload_yaml_into_config(config, str(edited), live_reload=True)
    TrialConfig.from_dict(config)  # the iteration loop's next call must not raise
    return launched, config, [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize(('base', 'bad_edit'), [
    ({'distributed_worker_sf_workers': 1}, {'sf_label_nodes_floor': 150_000}),
    ({'distributed_worker_sf_workers': 1}, {'sf_label_nodes_cap': 200_000}),
    ({'distributed_worker_sf_workers': 1}, {'sf_move_nodes': 100}),
    ({'distributed_worker_sf_workers': 1}, {'sf_label_escalate_q_gap': 0.8}),
    ({'distributed_worker_sf_workers': 8, 'sf_label_nodes_floor': 150_000},
     {'distributed_worker_sf_workers': 1}),
])
def test_live_reload_declines_the_bad_combination_and_applies_the_rest(
    tmp_path: Path, caplog, base: dict, bad_edit: dict,
) -> None:
    # An unrelated live key in the same save (`lr` would not do: it is
    # PB2-searched, so the reloader preserves it regardless).
    unrelated = {'sf_pid_target_winrate': 0.55}
    launched, config, messages = _reload(tmp_path, base, {**bad_edit, **unrelated}, caplog)
    assert launched['sf_pid_target_winrate'] != 0.55
    for k in bad_edit:
        assert config.get(k) == launched.get(k), f'{k} was applied'
    assert config['sf_pid_target_winrate'] == 0.55, 'an unrelated edit was declined too'
    assert any('NOT applied' in m and 'requires StockfishPool' in m for m in messages), messages
    assert not any(m.startswith('YAML reload failed') for m in messages), messages


def test_live_reload_applies_a_topology_and_budget_edit_that_is_valid_together(
    tmp_path: Path, caplog,
) -> None:
    edit = {'distributed_worker_sf_workers': 8, 'sf_label_nodes_floor': 150_000}
    _launched, config, messages = _reload(
        tmp_path, {'distributed_worker_sf_workers': 1}, edit, caplog,
    )
    assert {k: config[k] for k in edit} == edit
    assert not any('NOT applied' in m for m in messages), messages


def test_a_failing_decline_check_reverts_every_topology_key_and_says_so(
    tmp_path: Path, caplog, capsys, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed: the overlay has already written the save into config when
    the decline runs. If the check itself raises, every SF budget/topology key
    goes back to its pre-reload value (absent keys removed), unrelated edits
    stay applied, and an ERROR names the exception; nothing propagates."""
    def broken(*_a: Any, **_kw: Any) -> None:
        raise NameError('simulated decline failure')

    monkeypatch.setattr(trainable_config_ops, '_decline_sf_budget_topology_edit', broken)
    edit = {
        'distributed_worker_sf_workers': 1, 'sf_label_nodes_floor': 150_000,
        'sf_pid_target_winrate': 0.55,
    }
    launched, config, messages = _reload(
        tmp_path, {'distributed_worker_sf_workers': 8}, edit, caplog,
    )
    assert config['distributed_worker_sf_workers'] == launched['distributed_worker_sf_workers'] == 8
    assert 'sf_label_nodes_floor' not in launched
    assert 'sf_label_nodes_floor' not in config, 'an absent key was not removed again'
    assert config['sf_pid_target_winrate'] == 0.55, 'an unrelated edit was reverted'
    assert not any(m.startswith('YAML reload failed') for m in messages), messages
    errors = [
        r.getMessage() for r in caplog.records
        if r.levelno == logging.ERROR and 'REVERTED' in r.getMessage()
    ]
    assert len(errors) == 1, messages
    assert 'NameError: simulated decline failure' in errors[0]
    assert 'SF budget/topology edits REVERTED' in capsys.readouterr().out


def test_a_non_numeric_budget_value_is_declined_loudly_not_raised(
    tmp_path: Path, caplog,
) -> None:
    """A quoted number reaches the check as a string. The real reload must
    decline it with a message naming it (TrialConfig.from_dict, run by
    `_reload`, would otherwise raise mid-iteration), and launch must refuse
    it with the same message rather than an int() traceback."""
    launched, config, messages = _reload(
        tmp_path, {'distributed_worker_sf_workers': 8}, {'sf_label_nodes_floor': '1.5e5'}, caplog,
    )
    assert config.get('sf_label_nodes_floor') == launched.get('sf_label_nodes_floor')
    declined = [m for m in messages if 'NOT applied' in m]
    assert len(declined) == 1, messages
    assert "sf_label_nodes_floor='1.5e5' is not a number" in declined[0]
    with pytest.raises(ValueError, match=re.escape("sf_label_nodes_floor='1.5e5' is not a number")):
        reject_sf_budget_topology({'sf_label_nodes_floor': '1.5e5', 'distributed_worker_sf_workers': 8})
    with pytest.raises(ValueError, match="distributed_worker_sf_workers='eight' is not a number"):
        reject_sf_budget_topology({'distributed_worker_sf_workers': 'eight'})


def test_startup_intent_refusal_still_propagates_before_any_decline(
    tmp_path: Path, caplog,
) -> None:
    """main's `except StartupIntentError: raise` sits right after the decline
    call. A strict-intent refusal raised earlier in the try (here: the save
    adds `rl_startup_intent` without its contract) must propagate unchanged,
    with no overlay and no decline, even when the same save is also a bad SF
    combination."""
    raw = load_yaml_file(str(_PRODUCTION))
    _set(raw, 'distributed_worker_sf_workers', 1)
    start = tmp_path / 'start.yaml'
    start.write_text(yaml.safe_dump(raw), encoding='utf-8')
    config = flatten_run_config_defaults(load_yaml_file(str(start)))
    launched = dict(config)
    doc = copy.deepcopy(raw)
    _set(doc, 'sf_label_nodes_floor', 150_000)
    doc['rl_startup_intent'] = 'fresh_bootstrap'
    edited = tmp_path / 'edited.yaml'
    edited.write_text(yaml.safe_dump(doc), encoding='utf-8')
    with (
        caplog.at_level(logging.WARNING, logger='chess_anti_engine.tune.trainable_config_ops'),
        pytest.raises(StartupIntentError, match='strict startup'),
    ):
        _reload_yaml_into_config(config, str(edited), live_reload=True)
    assert config == launched
    assert not any('NOT applied' in r.getMessage() for r in caplog.records)


def test_live_reload_declines_every_key_of_a_jointly_bad_save_and_says_so(
    tmp_path: Path, caplog,
) -> None:
    """Review note: one save that drops the workers to one SF engine AND adds a
    floor declines BOTH, although each would be valid alone (conservative by
    design). The warning must name every declined key with attempted and
    kept values, and say all were declined together."""
    edit = {'distributed_worker_sf_workers': 1, 'sf_label_nodes_floor': 150_000}
    launched, config, messages = _reload(
        tmp_path, {'distributed_worker_sf_workers': 8}, edit, caplog,
    )
    assert config['distributed_worker_sf_workers'] == launched['distributed_worker_sf_workers'] == 8
    assert config.get('sf_label_nodes_floor') == launched.get('sf_label_nodes_floor')
    declined = [m for m in messages if 'NOT applied' in m]
    assert len(declined) == 1, messages
    warning = declined[0]
    assert '2 SF budget/topology edit(s) NOT applied, ALL of them declined together' in warning
    assert 'distributed_worker_sf_workers: 1 -> kept 8' in warning
    assert 'sf_label_nodes_floor: 150000 -> kept' in warning
    assert 'even one that would be valid on its own' in warning
    assert 'sf_label_nodes_floor > 0 requires StockfishPool' in warning


# -- worker ------------------------------------------------------------------

@pytest.mark.parametrize(('driver_workers', 'worker_workers', 'move_nodes', 'refused'), [
    (8, 1, 100, True),
    (8, 1, 5000, True),  # Even equality to the initial base is not durable.
    (8, 1, 0, False),
    (1, 2, 100, False),
])
def test_distributed_worker_uses_effective_cli_topology_before_any_bringup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver_workers: int,
    worker_workers: int, move_nodes: int, refused: bool,
) -> None:
    monkeypatch.setattr(runtime, '_resolve_distributed_worker_auth',
                        lambda **_kwargs: ('test-worker', tmp_path / 'unused-password'))
    config = {
        'sf_workers': driver_workers, 'distributed_worker_sf_workers': worker_workers,
        'sf_move_nodes': move_nodes, 'distributed_server_url': 'http://127.0.0.1:9000',
        'distributed_server_root': str(tmp_path / 'server'), 'stockfish_path': 'not-started',
    }
    command = runtime._build_distributed_worker_cmd(
        config=config, trial_root=tmp_path / 'worker', trial_id='trial',
        worker_index=0, worker_log=tmp_path / 'worker.log',
    )
    effective_workers = int(command[command.index('--sf-workers') + 1])
    assert effective_workers == worker_workers
    reco = runtime.build_recommended_worker(
        config=config, model_cfg=ModelConfig(), sf_nodes=5000, mcts_simulations=8,
    )
    assert reco['sf_move_nodes'] == move_nodes
    session = object.__new__(WorkerSession)
    session.args = SimpleNamespace(sf_workers=effective_workers)
    session._shutdown_requested = False
    reached: list[dict] = []

    def after_guard(_self: WorkerSession, incoming: dict) -> dict:
        reached.append(incoming)
        raise _ReachedSetup

    monkeypatch.setattr(WorkerSession, '_snapshot_reco', after_guard)
    expected = _REFUSAL if refused else _ReachedSetup
    with pytest.raises(expected) as caught:
        session._run_selfplay({'recommended_worker': reco})
    if refused:
        assert 'sf_move_nodes > 0 requires StockfishPool' in str(caught.value)
        assert reached == []
    else:
        assert reached == [reco]


# `args.sf_move_nodes` (and the floor/cap) are NOT tested as worker-local
# overrides: no worker CLI flag or worker.yaml field sets them, so
# `_resolve_reco` always reads the reco for these keys in production.

@pytest.mark.parametrize(('key', 'value'), [
    ('sf_label_nodes_floor', 150_000), ('sf_label_nodes_cap', 200_000),
    ('sf_label_escalate_q_gap', 0.8),
])
def test_invalid_budget_does_not_reach_an_existing_engine(key: str, value: float) -> None:
    session = object.__new__(WorkerSession)
    session.args = SimpleNamespace(sf_workers=1)
    session._shutdown_requested = False
    calls: list[int] = []
    session.sf = cast(Any, SimpleNamespace(set_nodes=calls.append))
    with pytest.raises(_REFUSAL, match=f'{key} > 0 requires StockfishPool'):
        session._run_selfplay({'recommended_worker': {key: value, 'sf_nodes': 5000}})
    assert calls == []


def test_worker_waits_instead_of_exiting_on_an_unusable_reco(caplog) -> None:
    """Review #1220 MAJOR 2(c): the refusal must not escape run(). An exit
    is relaunched by revive_dead_selfplay_processes into the same refusal
    forever; a wait keeps the worker alive and the reason in its log."""
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger('test.sf_budget_wait')
    session.args = SimpleNamespace(sf_workers=1, poll_seconds=0.0)
    session._shutdown_requested = False
    session._shutdown_signal = 0
    session.model_sha = 'sha'
    polls: list[int] = []
    manifest = {'recommended_worker': {'sf_nodes': 5000, 'sf_label_nodes_floor': 150_000}}

    def poll() -> dict:
        polls.append(1)
        if len(polls) >= 3:
            session._shutdown_requested = True
        return manifest

    s = cast(Any, session)
    s._install_shutdown_handlers = lambda: None
    s._poll_manifest = poll
    s._sync_assets = lambda _m: None
    s._maybe_ingest_dole_flag = lambda _m: None
    s._cleanup = lambda: None
    with caplog.at_level(logging.ERROR, logger='test.sf_budget_wait'):
        session.run()  # returns via the shutdown flag; the refusal never escapes
    assert len(polls) == 3, 'the worker stopped re-polling'
    errors = [r.getMessage() for r in caplog.records if 'NOT PLAYING' in r.getMessage()]
    assert len(errors) == 1, errors  # throttled: one line for an unchanged reason
    assert 'sf_label_nodes_floor > 0 requires StockfishPool' in errors[0]


@pytest.mark.parametrize(('key', 'value'), [
    ('sf_label_nodes_floor', 150_000), ('sf_label_escalate_q_gap', 0.8),
])
@pytest.mark.parametrize(('sf_workers', 'applied'), [(1, False), (8, True)])
def test_live_floor_change_is_not_transplanted_onto_a_single_engine(
    sf_workers: int, applied: bool, key: str, value: float,
) -> None:
    """floor/cap are LIVE reco keys, so they bypass the session-start check;
    _apply_live_reco must refuse them on a single engine (forcing the restart
    path, which waits) and transplant them on a pool."""
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger('test.sf_budget_live')
    session.args = SimpleNamespace(sf_nodes=None, sf_workers=sf_workers)
    state = SimpleNamespace(game=GameConfig(), base_nodes=5000)
    applied_overrides: list[Any] = []
    state.apply_live_overrides = lambda **kw: applied_overrides.append(kw)
    session._live_states = [state]
    session._live_states_lock = threading.Lock()
    cast(Any, session).sf = SimpleNamespace(set_nodes=lambda _n: None)
    reco = {'sf_nodes': 5000, key: value}
    built: list[dict] = []
    real_build = WorkerSession._build_selfplay_configs

    def recording_build(self: WorkerSession, incoming: dict) -> Any:
        built.append(incoming)
        return real_build(self, incoming)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(WorkerSession, '_build_selfplay_configs', recording_build)
        result = WorkerSession._apply_live_reco(session, reco)
    if applied:
        # Past the gate: configs were built for the transplant. (The transplant
        # body itself is pinned by test_worker_model_update.)
        assert built == [reco]
    else:
        assert result is False
        assert built == []
        assert applied_overrides == []
