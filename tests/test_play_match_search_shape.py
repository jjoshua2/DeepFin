"""Arena/gate matches must play the tuned PLAY shape, not the selfplay shape.

`PLAY_SEARCH_DEFAULTS` documents itself as the single source of truth for
"UCI, puzzle eval, the standardized arena, the training-gate match", but
`play_match_batch` never passed it, so the in-loop gate, the worker's arena
task and `scripts/match_checkpoints.py` all ran a bare `GumbelConfig()`.
"""

from __future__ import annotations

from typing import Any

import chess
import numpy as np
import torch

from chess_anti_engine.mcts.gumbel import PLAY_SEARCH_DEFAULTS, GumbelConfig
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.selfplay import match as match_mod
from chess_anti_engine.selfplay.match import play_match_batch


class _DummyModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        bs = int(x.shape[0])
        return {
            "policy_own": torch.zeros((bs, POLICY_SIZE), dtype=torch.float32),
            "wdl": torch.zeros((bs, 3), dtype=torch.float32),
        }


def _capture_cfgs(monkeypatch: Any) -> list[GumbelConfig]:
    """Record every GumbelConfig the search is actually called with."""
    seen: list[GumbelConfig] = []

    def fake_gumbel(_model, boards, *, cfg, **_kwargs):
        seen.append(cfg)
        n = len(boards)
        return (
            [np.zeros(POLICY_SIZE, dtype=np.float32)] * n,
            [next(iter(b.legal_moves)) and 0 for b in boards],
            [0.0] * n,
            [np.ones(POLICY_SIZE, dtype=bool)] * n,
        )

    monkeypatch.setattr(match_mod, "_HAS_GUMBEL_C", False)
    monkeypatch.setattr(match_mod, "run_gumbel_root_many", fake_gumbel)
    return seen


def _run(monkeypatch: Any, **kwargs: Any) -> list[GumbelConfig]:
    seen = _capture_cfgs(monkeypatch)
    model = _DummyModel().eval()
    play_match_batch(
        model, model,
        device="cpu",
        rng=np.random.default_rng(0),
        games=1,
        max_plies=1,
        mcts_type="gumbel",
        mcts_simulations=7,
        temperature=0.0,
        **kwargs,
    )
    assert seen, "search was never invoked"
    return seen


def test_defaults_to_the_play_shape(monkeypatch: Any) -> None:
    cfg = _run(monkeypatch)[0]
    for key, want in PLAY_SEARCH_DEFAULTS.items():
        assert getattr(cfg, key) == want, f"{key}: {getattr(cfg, key)} != {want}"
  # The regression this test exists for: these are the SELFPLAY values.
    assert cfg.c_scale != GumbelConfig().c_scale
    assert cfg.topk != GumbelConfig().topk


def test_caller_sim_budget_is_never_overridden(monkeypatch: Any) -> None:
    """PLAY_SEARCH_DEFAULTS carries no `simulations`; keep it that way."""
    assert "simulations" not in PLAY_SEARCH_DEFAULTS
    assert _run(monkeypatch)[0].simulations == 7


def test_explicit_overrides_win(monkeypatch: Any) -> None:
    cfg = _run(monkeypatch, gumbel_overrides={"c_scale": 0.5, "topk": 4})[0]
    assert cfg.c_scale == 0.5
    assert cfg.topk == 4


def test_empty_dict_forces_the_dataclass_defaults(monkeypatch: Any) -> None:
    """An explicit {} is the escape hatch back to bare GumbelConfig()."""
    cfg = _run(monkeypatch, gumbel_overrides={})[0]
    assert cfg.c_scale == GumbelConfig().c_scale
    assert cfg.topk == GumbelConfig().topk


def test_puct_matches_are_unaffected(monkeypatch: Any) -> None:
    """The overrides are Gumbel knobs; the PUCT path must not see them."""
    seen = _capture_cfgs(monkeypatch)
    model = _DummyModel().eval()
    stats = play_match_batch(
        model, model,
        device="cpu",
        rng=np.random.default_rng(0),
        games=2,
        max_plies=4,
        a_plays_white=[True, False],
        mcts_type="puct",
        mcts_simulations=1,
        temperature=0.0,
    )
    assert not seen, "PUCT match must not route through the Gumbel search"
    assert stats.games == 2


def test_board_fixture_is_sane() -> None:
    """Guard the fixture itself: a startpos must have legal moves to pick."""
    assert len(list(chess.Board().legal_moves)) == 20
