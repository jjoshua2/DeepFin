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
        # production only builds sf_p0_regret on selfplay slots (finalize.py:867)
        "is_selfplay": np.array([True, True]),
        "has_is_selfplay": np.array([True, True]),
        "legal_mask": legal,
        "policy_target": policy,
        "x": np.zeros((2, 4, 8, 8), dtype=np.float32),
        "game_id": np.array([7, 7], dtype=np.int64),
        "ply_index": np.array([0, 1], dtype=np.int64),
    }


def _collect(monkeypatch: pytest.MonkeyPatch, arrs: dict[str, Any], k: int = 2):
    monkeypatch.setattr(tcs, "load_shard_arrays", lambda _path: (arrs, {}))
    scan: dict[str, Any] = {
        "shard_names": [], "rows_scanned": 0, "rows_selfplay": 0,
        "skipped_not_selfplay": 0, "desync_checked": 0, "desync_orphaned": 0,
        "desync_rows_rejected": 0, "coverage_sum": 0.0, "coverage_n": 0,
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


def test_non_selfplay_rows_are_excluded(monkeypatch: pytest.MonkeyPatch) -> None:
    """⚑ Production never builds this label off a curriculum slot.

    `finalize.py:867` gates `want_sf_p0_regret` on `is_selfplay_slot`, so pricing
    the tail on non-selfplay rows measures an imputation those rows never carry.
    Mutation: dropping the `selfplay[i]` guard makes this analyse the row.
    """
    arrs = _shard(
        parent_block=GOOD_PARENT, child_block=BAD_CHILD,
        child_legal=CHILD_LEGAL, child_policy=FLAT_POLICY,
    )
    arrs["is_selfplay"] = np.array([True, False])
    rows, _, scan = _collect(monkeypatch, arrs)
    assert rows == []
    assert scan["skipped_not_selfplay"] == 1


def test_flat_reference_rows_leave_the_fit() -> None:
    """⚑ A row with ||g_true|| == 0 must not steer alpha_grad.

    It contributes row_num = 0 with row_den > 0, dragging the fit toward zero
    while contributing NOTHING to the pooled L2 the fit claims to minimise. The
    earlier version selected rows in the fitting loop and dropped flat rows only
    in the metric loop; measured on a 2-row case it fitted 0.3637 against a true
    argmin of 0.7340. Mutation: admitting flat rows to the fit changes alpha_grad.
    """
    live = _analysable({0: 0.0, 1: 0.2, 2: 0.4, 3: 0.8}, np.array([0.4, 0.3, 0.2, 0.1]))
    flat = _analysable(dict.fromkeys(range(4), 0.25), np.array([0.25] * 4))
    only_live = tcs.analyse([live], ("alpha_grad",))
    with_flat = tcs.analyse([live, flat], ("alpha_grad",))
    assert with_flat["n_rows"] == only_live["n_rows"] == 1
    assert with_flat["alpha_grad_raw"] == pytest.approx(only_live["alpha_grad_raw"])


def test_n_rows_describes_the_population_the_variants_describe() -> None:
    """`n_rows` and every variant statistic must count the SAME rows."""
    live = _analysable({0: 0.0, 1: 0.2, 2: 0.4, 3: 0.8}, np.array([0.4, 0.3, 0.2, 0.1]))
    flat = _analysable(dict.fromkeys(range(4), 0.25), np.array([0.25] * 4))
    res = tcs.analyse([live, flat], ("midpoint (live)",))
    assert res["n_rows"] == 1, "the flat row is in neither the fit nor the metric"


def test_never_scored_moves_do_not_enter_the_reference() -> None:
    """⚑ A legal move the wide label never scored has NO observed regret.

    Filling it with r_k fed pure 'alpha = 0' evidence from moves nobody measured.
    Mutation: reinstating `row.regret.get(m, row.r_k)` over the full legal set
    pulls alpha_grad toward 0 here.
    """
    regrets = {0: 0.0, 1: 0.2, 2: 0.9}          # move 3 is legal but NEVER scored
    row = tcs.Row(legal=np.array([0, 1, 2, 3]), regret=regrets,
                  surfaced=[0, 1], hidden=[2], r_k=0.2)
    row.prior = np.array([0.3, 0.3, 0.2, 0.2])
    got = tcs.analyse([row], ("alpha_grad",))
    same_without = tcs.Row(legal=np.array([0, 1, 2]), regret=regrets,
                           surfaced=[0, 1], hidden=[2], r_k=0.2)
    same_without.prior = np.array([0.375, 0.375, 0.25])
    ref = tcs.analyse([same_without], ("alpha_grad",))
    assert got["alpha_grad_raw"] == pytest.approx(ref["alpha_grad_raw"], rel=1e-9)
    assert got["dropped_unscored_mass_mean"] == pytest.approx(0.2)


def test_bootstrap_resamples_games_not_plies() -> None:
    """⚑⚑ THE RESAMPLING UNIT IS THE GAME.

    Adjacent plies of one game share an opening, material and phase, so they are
    not independent replicas. Resampling rows treats them as if they were and
    returns an interval NARROWER than the population earns. This matters most for
    the borderline calls an interval is consulted for: the row bootstrap put
    alpha_value's lower bound at 0.1838, excluding the historical 0.1759 by 0.008
    and firing a pre-registered branch on a margin the wrong unit manufactured.

    Mutation: ignoring `groups` (per-row resampling) collapses the interval here.
    """
    # 40 rows that are really 2 games of 20 identical plies. Row resampling sees
    # 40 independent draws; game resampling correctly sees 2.
    a = np.tile(np.array([[0.30, 0.05, 0.50, 0.02, 0.10]]), (20, 1))
    b = np.tile(np.array([[0.05, 0.05, 0.50, 0.00, 0.10]]), (20, 1))
    per_row = np.vstack([a, b])
    games = np.array([0] * 20 + [1] * 20, dtype=np.int64)

    wide = tcs.bootstrap_alphas(per_row, games, n_boot=400)
    narrow = tcs.bootstrap_alphas(per_row, None, n_boot=400)
    w = wide["alpha_value"][1] - wide["alpha_value"][0]
    n = narrow["alpha_value"][1] - narrow["alpha_value"][0]
    assert w > 3.0 * n, (
        f"clustering must widen the interval: game {w:.4f} vs row {n:.4f} — if these "
        "are close, `groups` is being ignored and the CI is the wrong unit")


def test_bootstrap_clusters_are_all_or_nothing() -> None:
    """A sampled game contributes ALL its plies, never a subset.

    With one game there is exactly one block, so every draw reproduces the full
    sample and the interval must collapse to the point estimate. Mutation:
    resampling rows within a cluster gives a non-degenerate interval here.
    """
    per_row = np.array([[0.3, 0.05, 0.5, 0.02, 0.1], [0.1, 0.05, 0.5, 0.01, 0.1]])
    one = tcs.bootstrap_alphas(per_row, np.array([7, 7]), n_boot=200)
    lo, hi = one["alpha_value"]
    assert hi - lo == pytest.approx(0.0, abs=1e-12)


def test_desync_rows_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """⚑ A desynced row's block describes a DIFFERENT position.

    `cast_probe` already rejected these; this screen read the raw block
    unconditionally. Mutation: dropping the `orphaned[...]` guard analyses the row.
    """
    arrs = _shard(
        parent_block=GOOD_PARENT, child_block=BAD_CHILD,
        child_legal=CHILD_LEGAL, child_policy=FLAT_POLICY,
    )
    monkeypatch.setattr(tcs, "sf_eval_pv_orphan_flags",
                        lambda _a: (np.array([True, False]), np.array([True, True])))
    rows, _, scan = _collect(monkeypatch, arrs)
    assert rows == []
    assert scan["desync_rows_rejected"] == 1
