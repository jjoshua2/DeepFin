"""``mcts: puct`` is refused at every selfplay/arena/puzzle boundary (2026-11-14).

Selfplay, arena and puzzle eval have one search, compiled Gumbel. ``"puct"``
stays a recognised word (``normalize_mcts_type``) so the refusal can say what
happened instead of "unknown value", but every config boundary that used to
route to the PUCT search now raises ``PUCT_RETIRED_MESSAGE``:

* ``SearchConfig`` / ``TrialConfig.from_dict`` / the published worker reco /
  the worker's reco resolver;
* ``pick_moves_for_boards`` / ``play_match_batch`` and the worker arena task;
* the ``--mcts`` options of ``run.py`` (``tests/test_run_relation_preflight.py``),
  the worker, ``arena.py``, the play-batch bench, ``scripts/bench_vs_sf.py`` and
  ``scripts/match_checkpoints.py``;
* ``run_puzzle_eval``.

A live yaml reload that brings ``puct`` is DECLINED (old value kept, the trial
continues); at launch it is fatal (``tests/test_search_type_boundaries.py``
parametrizes both over ``puct``). The UCI engine's PUCT pools do not read this
knob and are not covered here.
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.utils.search_type import (
    PUCT_RETIRED_MESSAGE,
    normalize_mcts_type,
    require_gumbel_search,
)
from tests.test_gumbel_default_entry_points import (
    _DummyModel,
    _one_puzzle_suite,
    _record_match_routes,
    _record_puzzle_routes,
    _run_worker_arena,
)

_PUCTS = ["puct", "PUCT", " Puct "]


def _assert_retired(exc: pytest.ExceptionInfo[BaseException]) -> None:
    assert str(exc.value) == PUCT_RETIRED_MESSAGE


def test_the_message_is_the_one_the_spec_fixed() -> None:
    assert PUCT_RETIRED_MESSAGE == (
        "mcts: puct was retired for selfplay/arena on 2026-11-14 (it recorded ~8% "
        "of rows under the default diff-focus filter and was never the production "
        "path); use gumbel"
    )


@pytest.mark.parametrize("value", _PUCTS)
def test_puct_is_still_a_word_but_not_a_search(value: str) -> None:
    assert normalize_mcts_type(value) == "puct"
    with pytest.raises(ValueError, match="retired") as exc:
        require_gumbel_search(value)
    _assert_retired(exc)
    # An unknown value keeps the generic message, not the retirement one.
    with pytest.raises(ValueError, match="must be 'gumbel'") as unknown:
        require_gumbel_search("gumbell")
    assert "puct" not in str(unknown.value)


# --------------------------------------------------------------------------
# Config boundaries
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", _PUCTS)
def test_search_config_refuses_puct(value: str) -> None:
    from chess_anti_engine.selfplay.config import SearchConfig

    with pytest.raises(ValueError, match="retired") as exc:
        SearchConfig(mcts_type=value)
    _assert_retired(exc)


@pytest.mark.parametrize("value", _PUCTS)
def test_trial_config_refuses_puct(value: str) -> None:
    from chess_anti_engine.tune.trial_config import TrialConfig

    with pytest.raises(ValueError, match="retired") as exc:
        TrialConfig.from_dict({"mcts": value})
    _assert_retired(exc)


def test_published_reco_refuses_puct() -> None:
    from chess_anti_engine.model import ModelConfig
    from chess_anti_engine.tune.distributed_runtime import build_recommended_worker

    with pytest.raises(ValueError, match="retired") as exc:
        build_recommended_worker(
            config={"mcts": "puct"}, model_cfg=ModelConfig(), sf_nodes=1, mcts_simulations=1,
        )
    _assert_retired(exc)


def test_worker_reco_resolver_refuses_puct(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale server publishing ``mcts: puct`` must not reach a worker search.

    ``SearchConfig`` refuses it too; it is swapped for a permissive stand-in so
    this test observes the resolver's own refusal, not the downstream one.
    """
    from chess_anti_engine import worker as worker_mod
    from chess_anti_engine.worker import WorkerSession

    monkeypatch.setattr(worker_mod, "SearchConfig", SimpleNamespace)

    session: Any = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.puct_retired.worker_selfplay")
    session.args = SimpleNamespace()
    session.opening_book_path = None
    session.opening_book_path_2 = None
    session.opening_fen_list_path = None
    session._dole_lock = threading.Lock()

    with pytest.raises(ValueError, match="retired") as exc:
        WorkerSession._build_selfplay_configs(session, {"sf_nodes": 1, "mcts": "puct"})
    _assert_retired(exc)


# --------------------------------------------------------------------------
# Match / arena
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", _PUCTS)
def test_pick_moves_refuses_puct(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    from chess_anti_engine.selfplay import match as match_mod

    calls = _record_match_routes(monkeypatch)
    with pytest.raises(ValueError, match="retired") as exc:
        match_mod.pick_moves_for_boards(
            _DummyModel().eval(), [chess.Board()], device="cpu",
            rng=np.random.default_rng(0), mcts_type=value, mcts_simulations=100,
            temperature=0.0, gumbel_add_noise=False,
        )
    _assert_retired(exc)
    assert calls == [], "a refused search type must not run any search"


def test_play_match_batch_refuses_puct_before_building_openings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chess_anti_engine.selfplay import match as match_mod

    calls = _record_match_routes(monkeypatch)

    def _no_openings(**_kwargs: Any) -> chess.Board:
        raise AssertionError("play_match_batch built an opening before refusing puct")

    monkeypatch.setattr(match_mod, "make_starting_board", _no_openings)
    model = _DummyModel().eval()
    with pytest.raises(ValueError, match="retired") as exc:
        match_mod.play_match_batch(
            model, model, device="cpu", rng=np.random.default_rng(0), mcts_type="puct",
            games=2, max_plies=2, mcts_simulations=100, temperature=0.0,
        )
    _assert_retired(exc)
    assert calls == []


@pytest.mark.parametrize("value", ["puct", "PUCT"])
def test_worker_arena_task_refuses_puct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, value: str,
) -> None:
    """Refused by the task itself, before any match is started.

    ``play_match_batch`` refuses puct too; replacing it with a tripwire makes
    this test observe the worker's own refusal.
    """
    from chess_anti_engine import worker as worker_mod

    def _tripwire(*_a: Any, **_k: Any) -> None:
        raise AssertionError("the worker arena task started a match with mcts=puct")

    monkeypatch.setattr(worker_mod, "play_match_batch", _tripwire)
    with pytest.raises(ValueError, match="retired") as exc:
        _run_worker_arena(monkeypatch, tmp_path, {"mcts": value})
    _assert_retired(exc)
    assert list(tmp_path.glob("*.json")) == [], "no arena result may be written"


# --------------------------------------------------------------------------
# CLIs
# --------------------------------------------------------------------------


def test_worker_cli_refuses_puct(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from chess_anti_engine import worker as worker_mod

    def _must_not_get_here(_args: Any) -> None:
        raise AssertionError("worker --mcts puct parsed")

    monkeypatch.setattr(worker_mod, "_setup_worker_logging", _must_not_get_here)
    monkeypatch.setattr(sys, "argv", ["worker", "--mcts", "puct"])
    with pytest.raises(SystemExit) as exc:
        worker_mod.main()
    assert exc.value.code == 2
    assert PUCT_RETIRED_MESSAGE in capsys.readouterr().err


def test_arena_cli_refuses_puct(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from chess_anti_engine import arena as arena_mod

    calls = _record_match_routes(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "arena", "--latest-model", "unused-a.pt", "--best-model", "unused-b.pt",
        "--device", "cpu", "--games", "2", "--max-plies", "2", "--mcts-simulations", "100",
        "--mcts", "PUCT",
    ])
    with pytest.raises(SystemExit) as exc:
        arena_mod.main()
    assert exc.value.code == 2
    assert PUCT_RETIRED_MESSAGE in capsys.readouterr().err
    assert calls == []


def test_play_batch_bench_refuses_puct(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from chess_anti_engine.bench import play_batch_timing

    monkeypatch.setattr(sys, "argv", ["play_batch_timing", "--mcts", "puct"])
    with pytest.raises(SystemExit) as exc:
        play_batch_timing.main()
    assert exc.value.code == 2
    assert PUCT_RETIRED_MESSAGE in capsys.readouterr().err

    with pytest.raises(ValueError, match="retired") as vexc:
        play_batch_timing._run_mcts_for_subset(
            _DummyModel(), [chess.Board()], mcts_type="puct", simulations=2,
            temperature=0.0, device="cpu", rng=np.random.default_rng(0),
        )
    _assert_retired(vexc)


@pytest.mark.parametrize("module", ["scripts.bench_vs_sf", "scripts.match_checkpoints"])
def test_script_clis_refuse_puct_at_parse_time(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], module: str,
) -> None:
    import importlib

    script = importlib.import_module(module)
    monkeypatch.setattr(sys, "argv", [module, "--mcts", "puct"])
    with pytest.raises(SystemExit) as exc:
        script.main()
    assert exc.value.code == 2
    assert PUCT_RETIRED_MESSAGE in capsys.readouterr().err


# --------------------------------------------------------------------------
# Puzzle eval
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["puct", "PUCT", "python_puct_reference"])
def test_run_puzzle_eval_refuses_every_old_puct_route(
    monkeypatch: pytest.MonkeyPatch, value: str,
) -> None:
    from chess_anti_engine.eval import puzzles as puzzles_mod

    calls = _record_puzzle_routes(monkeypatch)
    with pytest.raises(ValueError, match=r"retired|must be 'gumbel'") as exc:
        puzzles_mod.run_puzzle_eval(
            _DummyModel(), _one_puzzle_suite(), device="cpu", mcts_simulations=3,
            batch_size=1, search=value,
        )
    if value == "python_puct_reference":
        assert "must be 'gumbel'" in str(exc.value)
    else:
        _assert_retired(exc)
    assert calls == []


# --------------------------------------------------------------------------
# Live reload: declined, not fatal
# --------------------------------------------------------------------------


def _write_yaml(path: Path, body: str) -> str:
    path.write_text(body, encoding="utf-8")
    return str(path)


def test_live_reload_of_puct_into_an_empty_config_leaves_mcts_unset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture,
) -> None:
    """The ``before is missing`` branch of ``_decline_invalid_mcts_edit``.

    Starting from ``config = {}``, the reload writes ``mcts: puct`` into config
    and the decline must POP it (there is no old value to restore), so the
    trial's next ``from_dict`` realizes the Gumbel default instead of dying.
    """
    from chess_anti_engine.tune.trainable_config_ops import _reload_yaml_into_config
    from chess_anti_engine.tune.trial_config import TrialConfig

    config: dict[str, Any] = {}
    path = _write_yaml(tmp_path / "live.yaml", "selfplay:\n  mcts: puct\n  mcts_simulations: 77\n")
    with caplog.at_level("ERROR"):
        _reload_yaml_into_config(config, path, live_reload=True)

    assert "mcts" not in config
    assert config["mcts_simulations"] == 77
    assert TrialConfig.from_dict(config).mcts == "gumbel"
    out = capsys.readouterr().out
    assert "[trial] ERROR YAML reload: mcts='puct'" in out
    assert "<unset: default gumbel>" in out
    assert PUCT_RETIRED_MESSAGE in out
    assert any(r.levelname == "ERROR" and "mcts=" in r.getMessage() for r in caplog.records)


def test_live_reload_of_puct_keeps_the_running_gumbel(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    from chess_anti_engine.tune.trainable_config_ops import _reload_yaml_into_config
    from chess_anti_engine.tune.trial_config import TrialConfig

    config: dict[str, Any] = {"mcts": "gumbel", "mcts_simulations": 50}
    path = _write_yaml(tmp_path / "live.yaml", "selfplay:\n  mcts: PUCT\n  mcts_simulations: 77\n")
    _reload_yaml_into_config(config, path, live_reload=True)
    assert config["mcts"] == "gumbel"
    assert config["mcts_simulations"] == 77
    assert TrialConfig.from_dict(config).mcts == "gumbel"
    assert PUCT_RETIRED_MESSAGE in capsys.readouterr().out


def test_launch_with_puct_is_fatal(tmp_path: Path) -> None:
    from chess_anti_engine.tune.trainable_config_ops import _reload_yaml_into_config
    from chess_anti_engine.tune.trial_config import TrialConfig

    config: dict[str, Any] = {}
    _reload_yaml_into_config(config, _write_yaml(tmp_path / "launch.yaml", "selfplay:\n  mcts: puct\n"))
    assert config["mcts"] == "puct"
    with pytest.raises(ValueError, match="retired") as exc:
        TrialConfig.from_dict(config)
    _assert_retired(exc)
