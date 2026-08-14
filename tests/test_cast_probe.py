"""Tests for scripts/cast_probe.py.

Every test here is written to FAIL under a specific plausible mutation of the
probe, and the mutation is named in the test. A probe whose tests pass when its
sign is flipped or its parent/child join is swapped is not an instrument.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.script_loading import load_script_module

cast_probe = load_script_module("cast_probe.py")

WIDTH = 12


def _shard(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a minimal in-memory shard from per-row dicts."""
    n = len(rows)
    arrs: dict[str, Any] = {
        "policy_target": np.zeros((n, WIDTH), dtype=np.float32),
        "legal_mask": np.zeros((n, WIDTH), dtype=np.uint8),
        "sf_p0_regret": np.zeros((n, WIDTH), dtype=np.float32),
        "sf_multipv_raw": np.full((n, 8, 5), -1, dtype=np.int16),
        "sf_wdl": np.zeros((n, 3), dtype=np.float32),
        "game_id": np.zeros((n,), dtype=np.int64),
        "ply_index": np.zeros((n,), dtype=np.int32),
        "has_sf_wdl": np.zeros((n,), dtype=np.uint8),
        "has_sf_p0_regret": np.zeros((n,), dtype=np.uint8),
        "has_sf_multipv_raw": np.zeros((n,), dtype=np.uint8),
    }
    for i, r in enumerate(rows):
        arrs["game_id"][i] = r.get("game_id", 0)
        arrs["ply_index"][i] = r["ply_index"]
        legal = r.get("legal", list(range(WIDTH)))
        arrs["legal_mask"][i, legal] = 1
        pol = np.zeros((WIDTH,), dtype=np.float32)
        for idx, p in r.get("policy", {}).items():
            pol[idx] = p
        if pol.sum() == 0:
            pol[legal] = 1.0 / len(legal)
        arrs["policy_target"][i] = pol
        q = float(r.get("q", 0.0))
        # q = W - L; put the whole mass on W/L so the probe reads back exactly q.
        arrs["sf_wdl"][i] = [(1.0 + q) / 2.0, 0.0, (1.0 - q) / 2.0]
        arrs["has_sf_wdl"][i] = 1 if r.get("has_wdl", True) else 0
        covered = r.get("covered")
        if covered is not None:
            for k, (mi, regret) in enumerate(covered.items()):
                arrs["sf_multipv_raw"][i, k, 0] = mi
                arrs["sf_p0_regret"][i, mi] = regret
            arrs["has_sf_multipv_raw"][i] = 1
        regret_vec = r.get("regret")
        if regret_vec is not None:
            for mi, v in regret_vec.items():
                arrs["sf_p0_regret"][i, mi] = v
            arrs["has_sf_p0_regret"][i] = 1
    return arrs, {}


def _collect(rows: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> Any:
    shard = _shard(rows)
    monkeypatch.setattr(cast_probe, "load_shard_arrays", lambda _path: shard)
    scan: dict[str, Any] = {
        "rows_scanned": 0, "rows_sf_wdl": 0, "rows_sf_p0_regret": 0,
        "cast_pairs": 0, "cast_pairs_with_p0": 0,
        "skipped_shards": [], "skipped_shards_omitted": 0,
    }
    out = cast_probe.collect([Path("fake.zarr")], scan, np.random.default_rng(0))
    return out, scan


# --------------------------------------------------------------------------
# 1. POV / sign
# --------------------------------------------------------------------------

def test_advantage_pov_sign_adds_not_subtracts() -> None:
    """MUTATION: ``q_child - q_parent``.

    Both labels are already in their own record's mover POV, so a move that
    keeps the evaluation level must score 0. Under the subtraction mutation an
    even position after an even position scores +0.6 instead of 0.0.
    """
    # Parent mover was +0.3; after our reply the position is -0.3 from the
    # opponent's view, i.e. +0.3 held. Nothing was lost: A = 0.
    assert cast_probe.advantage(q_child=-0.3, q_parent=0.3) == pytest.approx(0.0)
    # We blundered: our own post-move eval is -0.5 while the root was +0.3.
    assert cast_probe.advantage(q_child=-0.5, q_parent=0.3) == pytest.approx(-0.2)
    # A solver-consistent teacher never hands out a positive advantage.
    assert cast_probe.advantage(q_child=-0.3, q_parent=0.3) <= 0.0


def test_blunder_is_negative_and_best_move_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end sign check through ``collect``.

    MUTATION: swapping the parent/child terms, or dropping the POV flip,
    makes the blunder row read POSITIVE.
    """
    rows = [
        {"ply_index": 4, "q": 0.4, "covered": {0: 0.0, 1: 0.05}},
        # holds the eval -> A = 0
        {"ply_index": 5, "q": -0.4, "regret": {0: 0.0}, "policy": {0: 1.0}},
        {"ply_index": 10, "q": 0.4, "covered": {0: 0.0, 1: 0.05}},
        # throws away 0.5 -> A = -0.5
        {"ply_index": 11, "q": -0.9, "regret": {0: 0.0}, "policy": {0: 1.0}},
    ]
    rows[2]["game_id"] = rows[3]["game_id"] = 1
    out, _ = _collect(rows, monkeypatch)
    assert out.adv == pytest.approx([0.0, -0.5])


# --------------------------------------------------------------------------
# 2. Adjacency must be exact
# --------------------------------------------------------------------------

def test_adjacency_requires_exact_previous_ply(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: joining on "nearest earlier row" instead of ``ply_index - 1``.

    A gap means the intervening ply was never stored, so the earlier label
    describes a DIFFERENT position and the advantage would be nonsense.
    """
    rows = [
        {"ply_index": 4, "q": 0.4, "covered": {0: 0.0}},
        {"ply_index": 7, "q": -0.4, "regret": {0: 0.0}, "policy": {0: 1.0}},
    ]
    out, scan = _collect(rows, monkeypatch)
    assert scan["cast_pairs"] == 0
    assert out.adv == []


def test_adjacency_does_not_cross_games(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: joining on ``ply_index`` alone, ignoring ``game_id``."""
    rows = [
        {"game_id": 0, "ply_index": 4, "q": 0.4, "covered": {0: 0.0}},
        {"game_id": 1, "ply_index": 5, "q": -0.4, "regret": {0: 0.0}, "policy": {0: 1.0}},
    ]
    out, scan = _collect(rows, monkeypatch)
    assert scan["cast_pairs"] == 0
    assert out.adv == []


# --------------------------------------------------------------------------
# 3. The covered set comes from the PARENT row
# --------------------------------------------------------------------------

def test_covered_set_is_read_from_the_parent_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: reading ``sf_multipv_raw`` from the CHILD row.

    ``sf_p0_regret`` at row t is built from row t-1's MultiPV, because SF's
    label search runs one ply late. The child's own MultiPV describes the
    position AFTER the move -- a different move set entirely.
    """
    rows = [
        # parent surfaced moves 0 and 1
        {"ply_index": 4, "q": 0.0, "covered": {0: 0.0, 1: 0.05}},
        # child's own MultiPV names DIFFERENT moves (5, 6); the probe must ignore it
        {"ply_index": 5, "q": 0.0, "covered": {5: 0.0, 6: 0.05},
         "regret": {0: 0.0, 1: 0.05}, "policy": {1: 1.0}},
    ]
    out, _ = _collect(rows, monkeypatch)
    assert out.n_covered == [2]
    # move 1 is inside the PARENT's set; under the mutation it would read False
    assert out.in_multipv == [True]


def test_played_move_outside_multipv_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {"ply_index": 4, "q": 0.0, "covered": {0: 0.0, 1: 0.05}},
        {"ply_index": 5, "q": 0.0, "regret": {0: 0.0, 1: 0.05, 7: 0.525},
         "policy": {7: 1.0}},
    ]
    out, _ = _collect(rows, monkeypatch)
    assert out.in_multipv == [False]
    assert out.regret_played == pytest.approx([0.525])


def test_sf_best_is_the_zero_regret_covered_move(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: ``argmax`` instead of ``argmin`` when locating SF's best move."""
    rows = [
        {"ply_index": 4, "q": 0.0, "covered": {3: 0.0, 1: 0.4}},
        {"ply_index": 5, "q": 0.0, "regret": {3: 0.0, 1: 0.4}, "policy": {3: 1.0}},
    ]
    out, _ = _collect(rows, monkeypatch)
    assert out.is_sf_best == [True]


# --------------------------------------------------------------------------
# 4. Imputed-tail accounting
# --------------------------------------------------------------------------

def test_imputed_tail_share_is_measured_against_the_legal_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The share of E_pi[regret] contributed by moves SF never surfaced.

    MUTATION: ``n_legal = mask.size`` (the full policy width) instead of the
    legal count. That is the headline denominator -- "what fraction of legal
    moves carries a fabricated regret" -- and the width version reports 99.8%
    for every position regardless of the board.

    Note the expected-regret sums are NOT sensitive to the legal mask, because
    ``probs`` is already masked before normalization; only the move COUNTS and
    the covered-mass share are. Guard what is actually reachable.
    """
    # 4 legal moves; SF covered 2 of them; the search puts 0.5 on a covered
    # move (regret 0) and 0.5 on an imputed one (regret 0.5).
    rows = [
        {"ply_index": 4, "q": 0.0, "legal": [0, 1, 2, 3], "covered": {0: 0.0, 1: 0.1}},
        {"ply_index": 5, "q": 0.0, "legal": [0, 1, 2, 3],
         "regret": {0: 0.0, 1: 0.1, 2: 0.55, 3: 0.55},
         "policy": {0: 0.5, 2: 0.5}},
    ]
    out, _ = _collect(rows, monkeypatch)
    assert out.n_legal == [4]
    assert out.n_covered == [2]
    assert out.er_total == pytest.approx([0.5 * 0.0 + 0.5 * 0.55])
    assert out.er_imputed == pytest.approx([0.5 * 0.55])
    assert out.mass_covered == pytest.approx([0.5])


def test_within_position_shuffle_preserves_the_regret_marginal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control must permute WITHIN the legal set, not resample it.

    MUTATION: drawing fresh random regrets would change the marginal and the
    control would then be able to "pass" for the wrong reason.
    """
    rows = [
        {"ply_index": 4, "q": 0.0, "legal": [0, 1, 2, 3], "covered": {0: 0.0, 1: 0.1}},
        {"ply_index": 5, "q": 0.0, "legal": [0, 1, 2, 3],
         "regret": {0: 0.0, 1: 0.1, 2: 0.55, 3: 0.55},
         "policy": {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}},
    ]
    out, _ = _collect(rows, monkeypatch)
    # Under a uniform search target the permutation cannot change E_pi[regret].
    assert out.er_total_shuf == pytest.approx(out.er_total)


# --------------------------------------------------------------------------
# 5. Calibration inversion
# --------------------------------------------------------------------------

def test_monotone_prefix_drops_the_saturation_foldback() -> None:
    """MUTATION: inverting the raw curve.

    The top regret bucket folds back (|A| shrinks) because those positions are
    already decided. Interpolating through a fold maps one advantage to two
    regrets and silently picks the wrong one.
    """
    xs = np.array([0.0, 0.02, 0.09, 0.17, 0.50])
    ys = np.array([-0.014, -0.044, -0.083, -0.160, -0.117])
    kx, ky = cast_probe.monotone_prefix(xs, ys)
    assert kx.tolist() == [0.0, 0.02, 0.09, 0.17]
    assert ky.tolist() == [-0.014, -0.044, -0.083, -0.160]
    assert np.all(np.diff(ky) < 0)


def test_invert_reads_the_curve_backwards() -> None:
    xs = np.array([0.0, 0.02, 0.09])
    ys = np.array([-0.01, -0.05, -0.09])
    assert cast_probe.invert(xs, ys, -0.05) == pytest.approx(0.02)
    # halfway between the -0.05 and -0.09 knots
    assert cast_probe.invert(xs, ys, -0.07) == pytest.approx(0.055)


def test_price_the_tail_reports_the_overstatement() -> None:
    """A synthetic population where the truth is known by construction.

    Outside-set moves are built to be worth exactly as much as the 0.02 bucket,
    while the shard assigns them 0.55. The probe must recover ~27x, not 1x.
    """
    rng = np.random.default_rng(0)
    n = 4000
    reg = rng.choice([0.0, 0.02, 0.09], size=n)
    adv = -0.5 * reg
    inside = np.ones((n,), dtype=bool)
    # 400 outside-set rows whose TRUE advantage matches the 0.02 bucket
    reg = np.concatenate([reg, np.full((400,), 0.55)])
    adv = np.concatenate([adv, np.full((400,), -0.01)])
    inside = np.concatenate([inside, np.zeros((400,), dtype=bool)])
    arr = {
        "adv": adv,
        "regret_played": reg,
        "in_multipv": inside,
        "pmax": np.ones_like(adv),
        "abs_q_parent": np.zeros_like(adv),
    }
    res = cast_probe.price_the_tail(arr, np.ones_like(inside), "all")
    assert res["implied_cp"] == pytest.approx(20.0, abs=2.0)
    assert res["assigned_cp"] == pytest.approx(550.0)
    assert res["overstatement"] > 20.0


def test_price_the_tail_is_flat_when_the_tail_is_priced_correctly() -> None:
    """NEGATIVE CONTROL for the pricing itself.

    If outside-set moves really are worth what the shard assigns them, the
    overstatement must come out ~1x. A probe that always reports a large
    overstatement is measuring its own arithmetic.
    """
    rng = np.random.default_rng(1)
    n = 4000
    reg = rng.choice([0.0, 0.1, 0.3, 0.55], size=n)
    adv = -0.5 * reg
    inside = np.ones((n,), dtype=bool)
    reg = np.concatenate([reg, np.full((400,), 0.55)])
    adv = np.concatenate([adv, np.full((400,), -0.275)])  # truly worth 0.55
    inside = np.concatenate([inside, np.zeros((400,), dtype=bool)])
    arr = {
        "adv": adv,
        "regret_played": reg,
        "in_multipv": inside,
        "pmax": np.ones_like(adv),
        "abs_q_parent": np.zeros_like(adv),
    }
    res = cast_probe.price_the_tail(arr, np.ones_like(inside), "all")
    assert res["overstatement"] == pytest.approx(1.0, abs=0.15)
