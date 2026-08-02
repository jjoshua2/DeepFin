"""The paired held-out CI must actually cluster by game, not merely accept a game id.

The failure this guards against is the project's signature defect: a value that is
accepted and then silently ignored. A `game_id` column that is read but not used
would still produce a plausible CI -- just the (too narrow) row-level one. The
discriminating observation is that clustering rows into FEWER games must WIDEN the
interval on the same numbers.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "holdout_paired_ci", Path(__file__).resolve().parents[1] / "scripts" / "holdout_paired_ci.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
paired = _MOD.paired


def _dump(ce: np.ndarray, game_id: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ce": ce.astype(np.float32),
        "agree": np.zeros_like(ce, dtype=np.float32),
        "base": np.ones_like(ce, dtype=np.float32),
        "game_id": game_id.astype(np.int64),
    }


def test_identical_dumps_give_exactly_zero() -> None:
    rng = np.random.default_rng(0)
    ce = rng.normal(1.0, 0.5, 400)
    gid = np.repeat(np.arange(20), 20)
    a = _dump(ce, gid)
    out = paired(a, a, "ce", n_boot=200, seed=0)
    assert out["delta"] == 0.0
    assert out["lo"] == 0.0
    assert out["hi"] == 0.0


def test_point_estimate_is_the_row_mean_difference() -> None:
    rng = np.random.default_rng(1)
    ce_a = rng.normal(1.0, 0.5, 300)
    ce_b = ce_a + rng.normal(0.05, 0.2, 300)
    gid = np.repeat(np.arange(15), 20)
    out = paired(_dump(ce_a, gid), _dump(ce_b, gid), "ce", n_boot=200, seed=0)
    assert out["delta"] == pytest.approx(float((ce_b - ce_a).mean()), abs=1e-5)
    assert out["n_rows"] == 300
    assert out["n_games"] == 15


def test_clustering_widens_the_interval_versus_singleton_games() -> None:
    """Same numbers, same estimand; only the cluster structure changes.

    Rows within a game share a large offset, so a row-level interval understates the
    uncertainty. If `game_id` were ignored, both calls would return the same width.
    """
    rng = np.random.default_rng(2)
    n_games, per_game = 40, 25
    offsets = rng.normal(0.0, 0.4, n_games).repeat(per_game)
    noise = rng.normal(0.0, 0.05, n_games * per_game)
    ce_a = np.zeros(n_games * per_game)
    ce_b = offsets + noise

    clustered_gid = np.repeat(np.arange(n_games), per_game)
    singleton_gid = np.arange(n_games * per_game)

    wide = paired(_dump(ce_a, clustered_gid), _dump(ce_b, clustered_gid), "ce", n_boot=4000, seed=0)
    narrow = paired(_dump(ce_a, singleton_gid), _dump(ce_b, singleton_gid), "ce", n_boot=4000, seed=0)

    assert wide["delta"] == pytest.approx(narrow["delta"], abs=1e-6)
    assert wide["n_games"] == n_games
    assert narrow["n_games"] == n_games * per_game
    assert (wide["hi"] - wide["lo"]) > 3.0 * (narrow["hi"] - narrow["lo"])


def test_row_order_mismatch_is_refused() -> None:
    gid = np.repeat(np.arange(4), 5)
    a = _dump(np.zeros(20), gid)
    b = _dump(np.zeros(20), gid[::-1])
    with pytest.raises(SystemExit):
        paired(a, b, "ce", n_boot=10, seed=0)
