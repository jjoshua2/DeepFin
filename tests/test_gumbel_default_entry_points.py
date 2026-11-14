"""Every search-type default resolves to Gumbel, observed at the consumer.

Each test drives a real entry point with the search type left UNSET and
records which search runner the production code actually called, so a default
that is flipped in one place and then overridden (or ignored) downstream fails
here. The runners themselves are stubbed with a legal-move answer: what is
under test is the route, not the search.

Entry points: the arena CLI (``chess_anti_engine/arena.py``) with no flags,
``play_match_batch`` with no ``mcts_type``, the worker's arena task with no
``arena.mcts``, the worker's selfplay config with no ``mcts`` in the reco,
``TrialConfig.from_dict`` / the published worker reco with no ``mcts`` key, and
the training puzzle eval (``_run_puzzle_eval_if_due``). The default
``SearchConfig`` dispatch is pinned in ``tests/test_selfplay_dispatch_contracts.py``.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.moves import POLICY_SIZE, legal_move_mask
from chess_anti_engine.selfplay import match as match_mod


class _DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bs = int(x.shape[0])
        return {
            "policy_own": torch.zeros((bs, POLICY_SIZE), dtype=torch.float32),
            "wdl": torch.zeros((bs, 3), dtype=torch.float32),
        }


def _legal_answer(boards: list[chess.Board]) -> tuple[list, list[int], list[float], list]:
    probs, actions, masks = [], [], []
    for board in boards:
        mask = legal_move_mask(board)
        legal = np.flatnonzero(mask)
        probs.append(mask.astype(np.float32) / len(legal))
        actions.append(int(legal[0]))
        masks.append(mask)
    return probs, actions, [0.0] * len(boards), masks


def _record_match_routes(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace every search runner ``pick_moves_for_boards`` can reach."""
    calls: list[str] = []

    def runner(name: str):
        def _run(_model, boards, **_kwargs):
            calls.append(name)
            return _legal_answer(list(boards))
        return _run

    monkeypatch.setattr(match_mod, "_HAS_GUMBEL_C", True)
    monkeypatch.setattr(match_mod, "_HAS_C_TREE", True)
    monkeypatch.setattr(match_mod, "_run_gumbel_root_many_c", runner("gumbel_c"), raising=False)
    monkeypatch.setattr(match_mod, "_run_mcts_many_c", runner("puct_c"), raising=False)
    monkeypatch.setattr(match_mod, "run_gumbel_root_many", runner("gumbel_python"))
    monkeypatch.setattr(match_mod, "run_mcts_many", runner("puct_python"))
    return calls


# --------------------------------------------------------------------------
# Arena
# --------------------------------------------------------------------------


def test_arena_cli_without_flags_plays_the_compiled_gumbel_search(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from chess_anti_engine import arena as arena_mod

    calls = _record_match_routes(monkeypatch)
    monkeypatch.setattr(arena_mod, "build_model", lambda _cfg: _DummyModel())
    monkeypatch.setattr(arena_mod, "load_state_dict_tolerant", lambda *_a, **_k: None)
    monkeypatch.setattr(arena_mod, "_load_state_dict", lambda _p: {})
    monkeypatch.setattr(sys, "argv", [
        "arena", "--latest-model", "unused-a.pt", "--best-model", "unused-b.pt",
        "--device", "cpu", "--games", "2", "--max-plies", "4", "--mcts-simulations", "1",
    ])

    arena_mod.main()

    assert calls, "the arena played no search"
    assert set(calls) == {"gumbel_c"}, calls
    assert "arena latest vs best" in capsys.readouterr().out


def test_arena_cli_explicit_puct_still_plays_puct(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from chess_anti_engine import arena as arena_mod

    calls = _record_match_routes(monkeypatch)
    monkeypatch.setattr(arena_mod, "build_model", lambda _cfg: _DummyModel())
    monkeypatch.setattr(arena_mod, "load_state_dict_tolerant", lambda *_a, **_k: None)
    monkeypatch.setattr(arena_mod, "_load_state_dict", lambda _p: {})
    monkeypatch.setattr(sys, "argv", [
        "arena", "--latest-model", "unused-a.pt", "--best-model", "unused-b.pt",
        "--device", "cpu", "--games", "2", "--max-plies", "2", "--mcts-simulations", "100",
        "--mcts", "puct",
    ])

    arena_mod.main()

    assert calls, "no search ran"
    assert set(calls) == {"puct_c"}, calls
    capsys.readouterr()


def test_play_match_batch_without_mcts_type_plays_the_compiled_gumbel_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _record_match_routes(monkeypatch)
    model = _DummyModel().eval()

    match_mod.play_match_batch(
        model, model, device="cpu", rng=np.random.default_rng(0),
        games=2, max_plies=4, mcts_simulations=100, temperature=0.0,
    )

    assert calls, "no search ran"
    assert set(calls) == {"gumbel_c"}, calls


def test_worker_arena_task_without_mcts_plays_the_compiled_gumbel_search(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from chess_anti_engine.worker import WorkerSession

    calls = _record_match_routes(monkeypatch)
    best_sha = "b" * 64
    session: Any = object.__new__(WorkerSession)
    session.model_sha = "a" * 64
    session.model = _DummyModel().eval()
    session.model_cfg_active = object()
    session._refused_best_sha = None
    session.last_best_sha = best_sha
    session.best_model = _DummyModel().eval()
    session.args = SimpleNamespace(poll_seconds=0.0, username="test")
    session.log = logging.getLogger("test.gumbel_default.worker_arena")
    session.device = "cpu"
    session.rng = np.random.default_rng(0)
    session.opening_book_path = None
    session.arena_pending_dir = tmp_path
    session.shared_cache_enabled = True
    session.leased_trial_id = None
    session.fixed_trial_id = "t"
    session.cache_dir = tmp_path
    monkeypatch.setattr("chess_anti_engine.worker.time.sleep", lambda _s: None)

    task = {"type": "arena", "arena": {"batch_games": 2, "max_plies": 4, "mcts_simulations": 100}}
    WorkerSession._run_arena(session, {"best_model": {"sha256": best_sha}}, task)

    assert calls, "no search ran"
    assert set(calls) == {"gumbel_c"}, calls
    (payload_path,) = list(tmp_path.glob("*.json"))
    assert json.loads(payload_path.read_text())["mcts"] == "gumbel"


# --------------------------------------------------------------------------
# Config defaults (training config, published reco, worker selfplay config)
# --------------------------------------------------------------------------


def test_trial_config_without_mcts_realizes_a_gumbel_selfplay_search() -> None:
    from chess_anti_engine.tune.trainable_config_ops import _play_batch_kwargs
    from chess_anti_engine.tune.trial_config import TrialConfig

    # Both construction paths: the loader (a dict without `mcts`) and the
    # dataclass default (direct construction, used by tests and tools).
    for tc in (TrialConfig.from_dict({}), TrialConfig()):
        assert tc.mcts == "gumbel"
        assert _play_batch_kwargs(tc)["search"].mcts_type == "gumbel"


def test_published_reco_without_mcts_is_gumbel_and_relations_pass_transport() -> None:
    from chess_anti_engine.model import ModelConfig
    from chess_anti_engine.tune.distributed_runtime import (
        build_recommended_worker,
        check_dynamic_relations_transport,
    )

    reco = build_recommended_worker(
        config={}, model_cfg=ModelConfig(), sf_nodes=1, mcts_simulations=1,
    )
    assert reco["mcts"] == "gumbel"
    # The transport guard reads the same default: no `mcts` key means Gumbel,
    # which transports relations.
    check_dynamic_relations_transport({"use_dynamic_relations": True})


def test_worker_selfplay_config_without_mcts_in_the_reco_is_gumbel() -> None:
    from chess_anti_engine.selfplay.config import SearchConfig
    from chess_anti_engine.worker import WorkerSession

    session: Any = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.gumbel_default.worker_selfplay")
    session.args = SimpleNamespace()
    session.opening_book_path = None
    session.opening_book_path_2 = None
    session.opening_fen_list_path = None
    session._dole_lock = threading.Lock()

    cfgs, _sf_args = WorkerSession._build_selfplay_configs(session, {"sf_nodes": 1})
    search = cfgs["search"]
    assert isinstance(search, SearchConfig)
    assert search.mcts_type == "gumbel"


# --------------------------------------------------------------------------
# Training puzzle eval
# --------------------------------------------------------------------------


def _record_puzzle_routes(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    from chess_anti_engine.eval import puzzles as puzzles_mod
    from chess_anti_engine.mcts import gumbel_c, puct_c

    calls: list[dict[str, Any]] = []

    def runner(name: str):
        def _run(_model, boards, **kwargs):
            calls.append({"route": name, **kwargs})
            return _legal_answer(list(boards))
        return _run

    monkeypatch.setattr(gumbel_c, "run_gumbel_root_many_c", runner("gumbel_c"))
    monkeypatch.setattr(puct_c, "run_mcts_many_c", runner("puct_c"))
    monkeypatch.setattr(puzzles_mod, "run_mcts_many", runner("puct_python"))
    return calls


def _one_puzzle_suite():
    from chess_anti_engine.eval import puzzles as puzzles_mod

    return puzzles_mod.PuzzleSuite(
        puzzles=[puzzles_mod.Puzzle(board=chess.Board(), best_moves=[chess.Move.from_uci("e2e4")])],
        name="unit",
    )


def test_run_puzzle_eval_default_is_the_compiled_gumbel_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chess_anti_engine.eval import puzzles as puzzles_mod

    calls = _record_puzzle_routes(monkeypatch)
    puzzles_mod.run_puzzle_eval(
        _DummyModel(), _one_puzzle_suite(), device="cpu", mcts_simulations=3, batch_size=1,
    )
    assert [c["route"] for c in calls] == ["gumbel_c"]
    assert calls[0]["cfg"].simulations == 3
    assert calls[0]["cfg"].temperature == 0.0
    assert calls[0]["cfg"].add_noise is False


def test_run_puzzle_eval_refuses_a_gumbel_knob_on_a_puct_route() -> None:
    from chess_anti_engine.eval import puzzles as puzzles_mod
    from chess_anti_engine.mcts.gumbel import GumbelConfig

    with pytest.raises(ValueError, match="only to search='gumbel'"):
        puzzles_mod.run_puzzle_eval(
            _DummyModel(), _one_puzzle_suite(), device="cpu",
            search="puct", gumbel_cfg=GumbelConfig(),
        )
    with pytest.raises(ValueError, match="unknown search"):
        puzzles_mod.run_puzzle_eval(_DummyModel(), _one_puzzle_suite(), device="cpu", search="mystery")


def _run_training_puzzle_phase(tc) -> None:
    from chess_anti_engine.tune.trainable_phases import _run_puzzle_eval_if_due

    out = _run_puzzle_eval_if_due(
        _DummyModel(), _one_puzzle_suite(), tc=tc, device="cpu",
        rng=np.random.default_rng(0), iteration_zero_based=0,
    )
    assert out["puzzle_total"] == 1


def test_training_puzzle_eval_runs_the_production_selfplay_search_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production yaml with puzzles switched on: the compiled Gumbel search,
    handed the exact shape selfplay builds (config and both runner args)."""
    import dataclasses

    import yaml

    from chess_anti_engine.selfplay.network_turn import build_selfplay_search_shape
    from chess_anti_engine.tune.trainable_config_ops import _play_batch_kwargs
    from chess_anti_engine.tune.trial_config import TrialConfig
    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

    repo = Path(__file__).resolve().parents[1]
    with open(repo / "configs" / "pbt2_small.yaml", encoding="utf-8") as fh:
        flat = flatten_run_config_defaults(yaml.safe_load(fh))
    tc = dataclasses.replace(TrialConfig.from_dict(flat), puzzle_interval=1, puzzle_simulations=16)

    calls = _record_puzzle_routes(monkeypatch)
    _run_training_puzzle_phase(tc)

    kw = _play_batch_kwargs(tc)
    shape = build_selfplay_search_shape(search=kw["search"], game=kw["game"], simulations=16)
    assert [c["route"] for c in calls] == ["gumbel_c"]
    call = calls[0]
    assert call["target_batch"] == shape.target_batch
    assert call["vloss_weight"] == shape.vloss_weight
    expected = dataclasses.replace(shape.cfg, temperature=0.0, add_noise=False)
    # The model has no encoding attributes, so the shape's own values survive.
    assert call["cfg"] == expected


@pytest.mark.parametrize(("mcts", "sims", "route"), [
    ("gumbel", 200, "gumbel_c"),
    ("puct", 200, "puct_c"),
    # Selfplay's dispatch runs Gumbel at or below max(64, fast) even under puct.
    ("puct", 32, "gumbel_c"),
])
def test_training_puzzle_eval_follows_the_selfplay_dispatch_rule(
    monkeypatch: pytest.MonkeyPatch, mcts: str, sims: int, route: str,
) -> None:
    from chess_anti_engine.tune.trial_config import TrialConfig

    tc = TrialConfig.from_dict({"mcts": mcts, "puzzle_interval": 1, "puzzle_simulations": sims})
    calls = _record_puzzle_routes(monkeypatch)
    _run_training_puzzle_phase(tc)
    assert [c["route"] for c in calls] == [route]
