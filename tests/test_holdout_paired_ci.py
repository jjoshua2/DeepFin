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


def test_base_zero_rows_are_excluded_from_the_estimate() -> None:
    """The `base` mask must gate the comparison, not merely ride along in the dump."""
    gid = np.repeat(np.arange(4), 5)
    a = _dump(np.zeros(20), gid)
    b = _dump(np.concatenate([np.ones(10), np.full(10, 100.0)]), gid)
    b["base"] = np.concatenate([np.ones(10), np.zeros(10)]).astype(np.float32)
    out = paired(a, b, "ce", n_boot=200, seed=0)
    assert out["n_rows"] == 10
    assert out["n_games"] == 2
    assert out["delta"] == pytest.approx(1.0, abs=1e-6)


def test_n_boot_controls_the_number_of_replicates() -> None:
    """A single replicate collapses the interval; a hard-coded n_boot could not."""
    rng = np.random.default_rng(5)
    gid = np.repeat(np.arange(30), 10)
    a = _dump(np.zeros(300), gid)
    b = _dump(rng.normal(0.1, 0.5, 300), gid)
    one = paired(a, b, "ce", n_boot=1, seed=0)
    assert one["lo"] == one["hi"]
    assert one["sd_boot"] == 0.0
    many = paired(a, b, "ce", n_boot=2000, seed=0)
    assert many["hi"] > many["lo"]
    assert many["sd_boot"] > 0.0


def test_row_order_mismatch_is_refused() -> None:
    gid = np.repeat(np.arange(4), 5)
    a = _dump(np.zeros(20), gid)
    b = _dump(np.zeros(20), gid[::-1])
    with pytest.raises(SystemExit):
        paired(a, b, "ce", n_boot=10, seed=0)


@pytest.mark.parametrize("seed", [0, 17])
@pytest.mark.parametrize("n_boot", [1, 11, 100])
def test_streamed_replicates_match_original_finite_draws(seed: int, n_boot: int) -> None:
    """Banked finite comparisons retain their exact game-clustered estimator."""
    gid = np.array([1, 1, 1, 2, 3, 3, 3, 3, 3, 3])
    a = _dump(np.arange(10) / 7, gid)
    b = _dump(np.arange(10) / 5 + np.arange(10) % 3, gid)
    a["base"][2] = 0
    b["base"][7] = 0
    mask = (a["base"] > 0) & (b["base"] > 0)
    games, inv = np.unique(gid[mask], return_inverse=True)
    sums = np.bincount(inv, weights=(b["ce"] - a["ce"])[mask])
    counts = np.bincount(inv).astype(np.float64)
    indices = np.random.default_rng(seed).integers(
        0, games.size, size=(n_boot, games.size),
    )
    boot = sums[indices].sum(1) / counts[indices].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    assert paired(a, b, "ce", n_boot=n_boot, seed=seed) == {
        "delta": float(sums.sum() / counts.sum()), "lo": float(lo),
        "hi": float(hi), "n_rows": int(mask.sum()), "n_games": int(games.size),
        "sd_boot": float(boot.std()),
    }


def test_bootstrap_draws_retain_only_one_replicate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse the old product-shaped allocation before it allocates anything."""
    gid = np.array([0, 0, 1, 2, 2])
    a = _dump(np.zeros(5), gid)
    b = _dump(np.arange(5), gid)
    rng = np.random.default_rng(7)
    calls: list[int] = []

    class BoundedGenerator:
        def integers(self, low: int, high: int, *, size: int) -> np.ndarray:
            assert size == 3, "bootstrap must allocate one game draw at a time"
            calls.append(size)
            return rng.integers(low, high, size=size)

    monkeypatch.setattr(_MOD.np.random, "default_rng", lambda seed: BoundedGenerator())
    result = paired(a, b, "ce", n_boot=13, seed=7)
    assert calls == [3] * 13
    assert result["n_rows"] == 5
    assert result["n_games"] == 3


@pytest.mark.parametrize("n_boot", [0, -1])
def test_nonpositive_replicate_count_is_refused(n_boot: int) -> None:
    a = _dump(np.zeros(2), np.array([0, 1]))
    with pytest.raises(ValueError, match="n_boot must be positive"):
        paired(a, a, "ce", n_boot=n_boot)


@pytest.mark.parametrize("empty", [False, True])
def test_no_paired_eligible_games_is_refused(empty: bool) -> None:
    n = 0 if empty else 3
    a = _dump(np.zeros(n), np.arange(n))
    b = _dump(np.ones(n), np.arange(n))
    b["base"][:] = 0
    with pytest.raises(ValueError, match="no paired eligible games"):
        paired(a, b, "ce", n_boot=2)
