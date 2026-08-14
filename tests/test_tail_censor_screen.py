"""Tests for the MultiPV tail-censoring screen (issue #425).

The load-bearing one is `test_p0_misalignment_is_caught`: it sets
`P0_PARENT_PLY_OFFSET` to 0 — the exact defect that made the first run of this
screen report a confident wrong number — and asserts the screen REFUSES rather
than answering. Everything else here would pass under that mutation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.replay.shard import SF_CP_SENTINEL
from tests.script_loading import load_script_module

tcs = load_script_module("tail_censor_screen.py")

N_ACTIONS = 64


def _pad(rows: list[tuple[int, int, int]], width: int = 16) -> np.ndarray:
    """A `sf_multipv_raw` block: (move_index, cp, mate) rows, -1-padded."""
    out = np.full((width, 3), -1, dtype=np.int32)
    out[:, 1] = SF_CP_SENTINEL
    out[:, 2] = 0
    for i, (mi, cp, mate) in enumerate(rows):
        out[i] = (mi, cp, mate)
    return out


def _shard(
    *,
    parent_block: list[tuple[int, int, int]],
    child_block: list[tuple[int, int, int]],
    child_legal: list[int],
    child_policy: dict[int, float],
) -> dict[str, Any]:
    """Two rows of one game: ply 0 (parent) and ply 1 (the row under analysis).

    `parent_block` is Stockfish's read of the CHILD's position — that is the whole
    point of the P0 shift — and `child_block` is the row's own block, describing
    the position after the child's move. The two are deliberately different so an
    alignment error cannot produce the right answer by luck.
    """
    legal = np.zeros((2, N_ACTIONS), dtype=bool)
    legal[0, : max(len(parent_block), 8)] = True
    legal[1, child_legal] = True
    policy = np.zeros((2, N_ACTIONS), dtype=np.float32)
    policy[0, 0] = 1.0
    for mi, p in child_policy.items():
        policy[1, mi] = p
    return {
        "sf_multipv_raw": np.stack([_pad(parent_block), _pad(child_block)]),
        "has_sf_multipv_raw": np.array([True, True]),
        "legal_mask": legal,
        "policy_target": policy,
        "x": np.zeros((2, 4, 8, 8), dtype=np.float32),
        "game_id": np.array([7, 7], dtype=np.int64),
        "ply_index": np.array([0, 1], dtype=np.int64),
    }


def _collect(monkeypatch: pytest.MonkeyPatch, arrs: dict[str, Any], k: int = 2):
    monkeypatch.setattr(tcs, "load_shard_arrays", lambda _path: (arrs, {}))
    scan: dict[str, Any] = {
        "shard_names": [], "rows_scanned": 0, "coverage_sum": 0.0, "coverage_n": 0,
        "unscored_mass_sum": 0.0, "surfaced_not_legal": 0,
        "skipped_shards": [], "skipped_shards_omitted": 0,
    }
    rows, planes = tcs.collect([Path("shard_000.zarr")], scan, k)
    return rows, planes, scan


# A parent block whose 6 scored moves all live inside the child's legal set.
GOOD_PARENT = [(10, 40, 0), (11, 20, 0), (12, 0, 0), (13, -60, 0), (14, -140, 0), (15, -300, 0)]
CHILD_LEGAL = [10, 11, 12, 13, 14, 15, 16, 17]
# A own-row block that scores TEN moves, six of which the child cannot play.
BAD_CHILD = [(20 + i, 100 - 30 * i, 0) for i in range(10)]
FLAT_POLICY = dict.fromkeys(CHILD_LEGAL, 0.125)


def test_p0_misalignment_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """⚑ THE MUTATION TEST. Reading the row's OWN block must not be answerable.

    With the offset at 0 the screen weighs the child's policy by a block that
    describes a different position; here that surfaces as a coverage above 1,
    which is what the impossible-value guard exists to refuse.
    """
    arrs = _shard(
        parent_block=GOOD_PARENT, child_block=BAD_CHILD,
        child_legal=CHILD_LEGAL, child_policy=FLAT_POLICY,
    )
    rows, _, _ = _collect(monkeypatch, arrs)
    assert len(rows) == 1, "correct alignment must analyse the row"

    monkeypatch.setattr(tcs, "P0_PARENT_PLY_OFFSET", 0)
    with pytest.raises(AssertionError, match="coverage"):
        _collect(monkeypatch, arrs)


def test_unscored_mass_invariant_refuses() -> None:
    """The aggregate half of the guard: policy mass SF never scored."""
    scan = {"unscored_mass_sum": 0.9, "coverage_sum": 0.5, "coverage_n": 1}
    with pytest.raises(AssertionError, match="unscored policy mass"):
        tcs.check_invariants(scan, 1)
    ok = tcs.check_invariants({"unscored_mass_sum": 0.001, "coverage_sum": 0.5, "coverage_n": 1}, 1)
    assert ok[1] == pytest.approx(0.001)


def test_row_uses_parent_block_regrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """The analysed row's regrets must come from the PARENT block's scores."""
    arrs = _shard(
        parent_block=GOOD_PARENT, child_block=BAD_CHILD,
        child_legal=CHILD_LEGAL, child_policy=FLAT_POLICY,
    )
    rows, _, scan = _collect(monkeypatch, arrs)
    row = rows[0]
    assert row.surfaced == [10, 11]                      # k=2, best two by cp
    assert row.hidden == [12, 13, 14, 15]
    assert row.regret[11] == pytest.approx(20 / tcs.SF_OWN_REGRET_CAP_CP)
    assert row.r_k == pytest.approx(20 / tcs.SF_OWN_REGRET_CAP_CP)
    assert scan["coverage_sum"] / scan["coverage_n"] == pytest.approx(6 / 8)


def test_sentinel_and_pad_rows_are_not_scored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cp sentinel is 'no score', not a score of -32768."""
    parent = [*GOOD_PARENT, (16, SF_CP_SENTINEL, 0), (17, SF_CP_SENTINEL, 0)]
    arrs = _shard(
        parent_block=parent, child_block=BAD_CHILD,
        child_legal=CHILD_LEGAL, child_policy=FLAT_POLICY,
    )
    rows, _, _ = _collect(monkeypatch, arrs)
    assert set(rows[0].regret) == {10, 11, 12, 13, 14, 15}
    assert max(rows[0].regret.values()) == pytest.approx(340 / tcs.SF_OWN_REGRET_CAP_CP)


def test_mate_dominates_centipawns(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mate score must outrank every cp score, so it becomes rank 1."""
    parent = [*GOOD_PARENT[:5], (15, 0, 3)]
    arrs = _shard(
        parent_block=parent, child_block=BAD_CHILD,
        child_legal=CHILD_LEGAL, child_policy=FLAT_POLICY,
    )
    rows, _, _ = _collect(monkeypatch, arrs)
    assert rows[0].surfaced[0] == 15
    assert rows[0].regret[15] == 0.0


def test_regret_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = [*GOOD_PARENT[:5], (15, -5000, 0)]
    arrs = _shard(
        parent_block=parent, child_block=BAD_CHILD,
        child_legal=CHILD_LEGAL, child_policy=FLAT_POLICY,
    )
    rows, _, _ = _collect(monkeypatch, arrs)
    assert rows[0].regret[15] == pytest.approx(1.0)


def test_row_skipped_when_block_no_wider_than_k(monkeypatch: pytest.MonkeyPatch) -> None:
    arrs = _shard(
        parent_block=GOOD_PARENT[:2], child_block=BAD_CHILD,
        child_legal=CHILD_LEGAL, child_policy=FLAT_POLICY,
    )
    rows, _, _ = _collect(monkeypatch, arrs)
    assert rows == []


def test_tail_value_spans_both_live_rules() -> None:
    """alpha=0 is eval/audit's floor, alpha=1 is finalize's midpoint, alpha=2 is 1.0."""
    r_k = 0.3
    assert tcs.tail_value(r_k, 0.0) == pytest.approx(r_k)
    assert tcs.tail_value(r_k, 1.0) == pytest.approx((r_k + 1.0) / 2.0)
    assert tcs.tail_value(r_k, 2.0) == pytest.approx(1.0)


def test_gradient_is_centred() -> None:
    """dL/dz for a softmax sums to zero, and a constant regret gives no gradient."""
    p = np.array([0.5, 0.3, 0.2])
    r = np.array([0.1, 0.4, 0.9])
    g = tcs.gradient(p, r)
    assert g.sum() == pytest.approx(0.0)
    assert tcs.gradient(p, np.full(3, 0.7)) == pytest.approx(np.zeros(3))


def _analysable(regrets: dict[int, float], prior: np.ndarray, k: int = 2) -> Any:
    order = sorted(regrets, key=lambda m: regrets[m])
    row = tcs.Row(
        legal=np.array(order), regret=regrets, surfaced=order[:k], hidden=order[k:],
        r_k=max(regrets[m] for m in order[:k]),
    )
    row.prior = prior
    return row


def test_alpha_value_recovers_the_planted_tail() -> None:
    """alpha_value is fitted, so a tail PLANTED at a known alpha must be recovered."""
    for planted in (0.0, 0.4, 1.0, 1.7):
        r_k = 0.2
        fill = tcs.tail_value(r_k, planted)
        regrets = {0: 0.05, 1: r_k, 2: fill, 3: fill}
        row = _analysable(regrets, np.array([0.4, 0.3, 0.2, 0.1]))
        res = tcs.analyse([row], ("r_k censor", "midpoint (live)", "alpha_value", "alpha_grad"))
        assert res["alpha_value"] == pytest.approx(planted, abs=1e-9)


def test_true_censor_is_exact_when_nothing_is_hidden_below_r_k() -> None:
    """A tail that really sits at r_k must price at alpha=0 and cost no gradient."""
    r_k = 0.25
    regrets = {0: 0.1, 1: r_k, 2: r_k, 3: r_k}
    row = _analysable(regrets, np.array([0.4, 0.3, 0.2, 0.1]))
    res = tcs.analyse([row], ("r_k censor", "midpoint (live)"))
    assert res["alpha_value"] == pytest.approx(0.0)
    assert res["variants"]["r_k censor"]["cos"] == pytest.approx(1.0)
    assert res["variants"]["r_k censor"]["rel_l2_pooled"] == pytest.approx(0.0, abs=1e-12)
    assert res["variants"]["midpoint (live)"]["rel_l2_pooled"] > 0.1


def test_midpoint_overstates_a_benign_tail() -> None:
    """The live rule must read HIGH when the hidden moves are only mildly bad."""
    regrets = {0: 0.0, 1: 0.1, 2: 0.15, 3: 0.2}
    row = _analysable(regrets, np.array([0.25, 0.25, 0.25, 0.25]))
    res = tcs.analyse([row], ("r_k censor", "midpoint (live)"))
    assert res["midpoint_cp"] > res["true_cp"]
    assert res["alpha_value"] < 1.0
    # and it must push HARDER on the hidden moves than the truth warrants
    press = res["variants"]["midpoint (live)"]["tail_pressure"]
    assert press > res["true_pressure"] > 0.0


def test_alpha_grad_is_the_least_squares_optimum() -> None:
    """alpha_grad must beat every neighbouring alpha on pooled gradient L2."""
    rows = [
        _analysable({0: 0.0, 1: 0.2, 2: 0.35, 3: 0.9}, np.array([0.4, 0.3, 0.2, 0.1])),
        _analysable({0: 0.05, 1: 0.15, 2: 0.6, 3: 0.7}, np.array([0.1, 0.2, 0.3, 0.4])),
    ]
    grid = ("r_k censor", "midpoint (live)", "alpha_value", "alpha_grad")
    res = tcs.analyse(rows, grid)
    best = res["variants"]["alpha_grad"]["rel_l2_pooled"]
    for name in ("r_k censor", "midpoint (live)", "alpha_value"):
        assert best <= res["variants"][name]["rel_l2_pooled"] + 1e-12
    assert 0.0 <= res["alpha_grad"] <= 2.0


def test_alpha_grad_and_alpha_value_disagree_on_a_skewed_tail() -> None:
    """⚑ Matching the SCALAR mean is not matching the gradient — they are different fits.

    A tail whose mass sits on the mild moves but whose weight sits on the severe
    one separates them; if this ever collapses to equality the two fits have been
    accidentally wired to the same quantity.
    """
    rows = [_analysable({0: 0.0, 1: 0.1, 2: 0.12, 3: 0.95}, np.array([0.05, 0.05, 0.8, 0.1]))]
    res = tcs.analyse(rows, ("alpha_value", "alpha_grad"))
    assert abs(res["alpha_value"] - res["alpha_grad_raw"]) > 0.05


def test_pooled_l2_is_not_the_per_row_mean() -> None:
    """The near-zero-||g_true|| row that read 4557 must not move the pooled number.

    A row whose true regrets are flat has an (almost) zero reference gradient, so
    its per-row RELATIVE error is unbounded. Pooling by summed squares is what
    makes the reported figure stable, and this pins that difference.
    """
    normal = _analysable({0: 0.0, 1: 0.2, 2: 0.4, 3: 0.8}, np.array([0.4, 0.3, 0.2, 0.1]))
    flat = _analysable({0: 0.2, 1: 0.2, 2: 0.2 + 1e-9, 3: 0.2 + 1e-9},
                       np.array([0.25, 0.25, 0.25, 0.25]))
    res = tcs.analyse([normal, flat], ("midpoint (live)",))
    v = res["variants"]["midpoint (live)"]
    assert v["rel_l2_median"] > 10.0 * v["rel_l2_pooled"]
    assert v["rel_l2_pooled"] < 5.0
