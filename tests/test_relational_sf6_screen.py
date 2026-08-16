"""Tests for the relational SF-6 gradient screen.

The load-bearing ones, in order of what they would catch:

* `test_top6_ceiling_actually_bounds_top6_objectives` — the ceiling is the whole
  experiment's verdict. If it is not a real bound the verdict is not a verdict.
* `test_observed_pairwise_is_exactly_g_true_when_nothing_is_hidden` — pins the
  identity `dL/dz_i = sum_j p_i p_j (r_i - r_j)` that explains every result.
* `test_every_objective_agrees_with_the_regret_gradient_on_a_clear_row` — a sign
  error would silently invert the whole screen and still print plausible cosines.
* `test_tail_order_limit_is_the_softplus_limit` — pins the claim that the
  unmargined order constraint BECOMES a constant tail, rather than escaping one.
* `test_hyperparameters_are_selected_on_fit_only` — fitting and scoring on the
  same rows measures itself; this plants a hyperparameter that separates them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.script_loading import load_script_module

rss = load_script_module("relational_sf6_screen.py")


def _row(
    q: list[float], p: list[float], n_tail: int = 3, game_id: int = 0,
    tail_r: float | None = None,
) -> Any:
    """A row with ``len(q)`` surfaced moves and ``n_tail`` hidden ones.

    Regrets are derived from ``q`` exactly as ``tail_censor_screen`` derives them,
    so the row is a faithful miniature rather than an unrelated fixture.
    """
    q_arr = np.array(q + [np.nan] * n_tail, dtype=np.float64)
    best = max(q)
    r_s = [min(max(best - x, 0.0), 1000.0) / 1000.0 for x in q]
    r_k = max(r_s)
    r_arr = np.array(r_s + [r_k if tail_r is None else tail_r] * n_tail, dtype=np.float64)
    p_arr = np.array(p, dtype=np.float64)
    p_arr = p_arr / p_arr.sum()
    s = np.array([True] * len(q) + [False] * n_tail)
    return rss.RowData(
        game_id=game_id, p=p_arr, z=np.log(p_arr), r=r_arr, q=q_arr, s=s, r_k=r_k,
    )


def _objectives() -> dict[str, Any]:
    """Every candidate, at a hyperparameter setting inside its screened range."""
    return {
        "hinge": lambda r: rss.grad_hinge(r, t=1.0),
        "margin_tanh": lambda r: rss.grad_margin(r, beta=8.0, kappa=200.0),
        "margin_clip": lambda r: rss.grad_margin(r, beta=0.05, kappa=8.0, shape="clip"),
        "indifference": lambda r: rss.grad_margin(r, beta=8.0, kappa=200.0, tau=50.0),
        "listwise": lambda r: rss.grad_listwise(r, temp_cp=100.0),
        "tail_order": lambda r: rss.grad_surfaced_over_tail(r, t=1.0),
        "tail_order_prior": lambda r: rss.grad_surfaced_over_tail(
            r, t=1.0, prior_weighted=True),
        "observed_pairwise": rss.grad_observed_pairwise,
        "borda": rss.grad_borda_limit,
        "tail_limit": rss.grad_tail_order_limit,
        "constant_tail": lambda r: rss.grad_constant_tail(r, alpha=0.14),
    }


# --------------------------------------------------------------- sign/orientation


def test_every_objective_agrees_with_the_regret_gradient_on_a_clear_row() -> None:
    """⚑ SIGN CHECK. SF says A is clearly best; the net puts its mass on B.

    The reference loss ``L = p.r`` wants A raised (``dL/dz_A < 0``) and B lowered
    (``dL/dz_B > 0``). Every candidate must agree, or the screen is measuring the
    negative of what it claims and every cosine below is inverted.
    """
    # Move 0 is SF's best; move 5 is SF's WORST surfaced move and is where the net
    # has parked 60% of its mass. ⚑ "B" must be worse than the PRIOR-WEIGHTED MEAN
    # regret, not merely worse than A: the reference gradient is centred on
    # ``E_p[r]``, so a second-best move can sit BELOW the mean and be pushed UP by
    # the very loss this screen references. The first cut of this fixture got that
    # wrong and the reference itself failed the assertion.
    row = _row(q=[300.0, 250.0, 200.0, 150.0, 100.0, -700.0],
               p=[0.05, 0.05, 0.05, 0.05, 0.05, 0.60, 0.05, 0.05, 0.05])
    ref = rss.gradient(row.p, row.r)
    assert ref[0] < 0.0 < ref[5], "the reference itself must want A up and B down"
    for name, fn in _objectives().items():
        g = fn(row)
        if name in ("tail_order", "tail_order_prior", "tail_limit"):
            # These make no claim BETWEEN surfaced moves; they order S above T.
            assert g[row.s].max() < 0.0 <= g[~row.s].min(), name
            continue
        assert g[0] < 0.0, f"{name} pushes SF's best move DOWN"
        assert g[5] > 0.0, f"{name} pushes the net's over-weighted bad move UP"


def test_every_objective_is_difference_based_so_its_gradient_sums_to_zero() -> None:
    """Depending on ``z`` only through differences is what makes the sum-zero
    ceiling the binding one; an objective that violated it would invalidate that."""
    row = _row(q=[300.0, 120.0, 0.0, -80.0, -220.0, -500.0],
               p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.06, 0.04, 0.02])
    for name, fn in _objectives().items():
        assert fn(row).sum() == pytest.approx(0.0, abs=1e-9), name


# ------------------------------------------------------------------- the ceilings


def test_top6_ceiling_actually_bounds_top6_objectives() -> None:
    """⚑⚑ THE VERDICT RESTS ON THIS. No vector supported on the six surfaced
    coordinates may exceed the reported ceiling's cosine against ``g_true``."""
    rng = np.random.default_rng(0)
    row = _row(q=[300.0, 120.0, 0.0, -80.0, -220.0, -500.0],
               p=[0.25, 0.2, 0.12, 0.1, 0.08, 0.05, 0.1, 0.06, 0.04],
               tail_r=0.9)
    ceil = rss.ceilings([row])
    bound = ceil["mean_per_row"]["top6_only"]
    g = row.g_true
    for _ in range(500):
        v = np.zeros_like(g)
        v[row.s] = rng.normal(size=int(row.s.sum()))
        c = float(v @ g / (np.linalg.norm(v) * np.linalg.norm(g)))
        assert c <= bound + 1e-9
    # and the bound is TIGHT: g_true restricted to S attains it exactly.
    v = np.where(row.s, g, 0.0)
    assert float(v @ g / (np.linalg.norm(v) * np.linalg.norm(g))) == pytest.approx(bound)


def test_sum_zero_ceiling_is_tighter_and_still_a_bound() -> None:
    """Every screened objective sums to zero, so this is the binding ceiling."""
    rng = np.random.default_rng(1)
    row = _row(q=[300.0, 120.0, 0.0, -80.0, -220.0, -500.0],
               p=[0.25, 0.2, 0.12, 0.1, 0.08, 0.05, 0.1, 0.06, 0.04], tail_r=0.9)
    ceil = rss.ceilings([row])
    tight = ceil["mean_per_row"]["top6_only_sumzero"]
    g = row.g_true
    # ⚑ STRICTLY tighter. This row's tail carries real gradient, so g[S] does NOT
    # sum to zero and the constraint must actually bind. Asserting only "<=" is
    # what let a mutation that skips the recentring survive.
    assert abs(g[row.s].sum()) > 1e-6
    assert tight < ceil["mean_per_row"]["top6_only"] - 1e-6
    assert rss._sum_zero_on(g, row.s).sum() == pytest.approx(0.0, abs=1e-12)
    for _ in range(500):
        v = np.zeros_like(g)
        v[row.s] = rng.normal(size=int(row.s.sum()))
        v[row.s] -= v[row.s].mean()          # sum-zero AND supported on S
        c = float(v @ g / (np.linalg.norm(v) * np.linalg.norm(g)))
        assert c <= tight + 1e-9


def test_order_tail_cone_projection_is_feasible_and_optimal() -> None:
    """The escape hatch's bound: tail coordinates may be pushed UP but never down,
    because "not surfaced" observes an ORDER and never a magnitude."""
    rng = np.random.default_rng(2)
    g = np.array([-0.4, -0.1, 0.05, 0.2, -0.03, 0.28])
    tail = np.array([False, False, False, True, True, True])
    v = rss.project_sum_zero_cone(g, tail)
    assert v[tail].min() >= -1e-12, "tail coordinates must not be driven negative"
    assert v.sum() == pytest.approx(0.0, abs=1e-9)
    best = float(v @ g / (np.linalg.norm(v) * np.linalg.norm(g)))
    for _ in range(2000):
        w = rng.normal(size=g.size)
        w[tail] = np.abs(w[tail])
        w -= w.mean()
        w[tail] = np.maximum(w[tail], 0.0)
        if np.linalg.norm(w) == 0.0 or abs(w.sum()) > 1e-9:
            continue
        assert float(w @ g / (np.linalg.norm(w) * np.linalg.norm(g))) <= best + 1e-8


# ------------------------------------------------------------- structural identities


def test_observed_pairwise_is_exactly_g_true_when_nothing_is_hidden() -> None:
    """⚑⚑ THE IDENTITY THE WHOLE SCREEN TURNS ON.

    ``dL/dz_i = p_i(r_i - E_p[r]) = sum_j p_i p_j (r_i - r_j)``. With every legal
    move surfaced the observed pairwise form must reproduce the reference gradient
    to machine precision — which is what makes its shortfall on real rows
    attributable to the MISSING ``S x T`` pairs and to nothing else.
    """
    row = _row(q=[300.0, 120.0, 0.0, -80.0, -220.0, -500.0],
               p=[0.3, 0.25, 0.2, 0.12, 0.08, 0.05], n_tail=0)
    assert rss.grad_observed_pairwise(row) == pytest.approx(row.g_true, abs=1e-15)


def test_tail_order_limit_is_the_softplus_limit() -> None:
    """The unmargined order constraint does not escape the constant tail — it
    BECOMES one. At large ``t`` the softplus form must align with the closed form."""
    row = _row(q=[300.0, 120.0, 0.0, -80.0, -220.0, -500.0],
               p=[0.25, 0.2, 0.12, 0.1, 0.08, 0.05, 0.1, 0.06, 0.04])
    a = rss.grad_surfaced_over_tail(row, t=1e6, prior_weighted=True)
    b = rss.grad_tail_order_limit(row)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos == pytest.approx(1.0, abs=1e-6)


def test_borda_limit_is_the_hinge_limit() -> None:
    """Every pure-order variant converges here, which is what closes the
    edge-pinning question analytically instead of by widening the grid again."""
    row = _row(q=[300.0, 120.0, 0.0, -80.0, -220.0, -500.0],
               p=[0.25, 0.2, 0.12, 0.1, 0.08, 0.05, 0.1, 0.06, 0.04])
    a = rss.grad_hinge(row, t=1e5)
    b = rss.grad_borda_limit(row)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    assert cos == pytest.approx(1.0, abs=1e-6)


def test_ties_contribute_nothing_to_an_order_only_objective() -> None:
    """SF asserting no order between two moves must not force the policy apart."""
    row = _row(q=[100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
               p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.06, 0.04, 0.02])
    assert rss.grad_hinge(row, t=1.0) == pytest.approx(np.zeros(9))
    assert rss.grad_borda_limit(row) == pytest.approx(np.zeros(9))
    assert rss.grad_observed_pairwise(row) == pytest.approx(np.zeros(9))


def test_indifference_weight_shrinks_a_negligible_distinction() -> None:
    """Hypothesis (3): a 2cp gap must not be supervised like a 400cp gap.

    ⚑ Stated on a row whose ONLY surfaced pair IS the negligible one. On a
    six-move row that pair's contribution is buried in fourteen others and damping
    it can move the TOTAL gradient either way — the first cut of this test
    asserted on the total and failed for exactly that reason, which made the
    assertion wrong rather than the weighting.
    """
    row = _row(q=[300.0, 298.0], p=[0.4, 0.3, 0.2, 0.1], n_tail=2)
    plain = rss.grad_margin(row, beta=8.0, kappa=200.0)
    damped = rss.grad_margin(row, beta=8.0, kappa=200.0, tau=100.0)
    w = min(1.0, abs(row.q[0] - row.q[1]) / 100.0)
    assert w == pytest.approx(0.02)
    assert damped == pytest.approx(w * plain)
    assert abs(damped[0]) < abs(plain[0])


# ------------------------------------------------------------------- methodology


def test_split_is_disjoint_by_game() -> None:
    """Rows inside one game are correlated; a row-level split leaks the fit set."""
    rows = [_row(q=[300.0, 120.0, 0.0, -80.0, -220.0, -500.0],
                 p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.06, 0.04, 0.02], game_id=i // 5)
            for i in range(200)]
    fit, held = rss.split_by_game(rows, 0.5)
    assert len(fit) + len(held) == len(rows)
    assert not ({r.game_id for r in fit} & {r.game_id for r in held})
    assert fit
    assert held


def test_hyperparameters_are_selected_on_fit_only() -> None:
    """⚑ Plant a knob that is best on FIT and worst on HELD-OUT.

    If selection ever peeked at the held-out rows the reported score would be the
    grid's held-out maximum; it must instead be the score of the FIT-chosen knob.
    """
    fit = [_row(q=[300.0, 120.0, 0.0, -80.0, -220.0, -500.0],
                p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.06, 0.04, 0.02], game_id=0)]
    held = [_row(q=[300.0, 120.0, 0.0, -80.0, -220.0, -500.0],
                 p=[0.05, 0.05, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1], game_id=1)]

    def build(k: int):
        # perfect on the game it names, exactly inverted on the other one, so the
        # two splits DISAGREE about the best knob. A knob whose optimum is the same
        # on both splits cannot detect selection leakage at all — the first cut of
        # this test used one and survived the mutation that selects on held-out.
        return lambda r: (1.0 if r.game_id == k else -1.0) * r.g_true

    res = rss.fit_and_score(fit, held, build, [{"k": 0}, {"k": 1}])
    assert res["hp"] == {"k": 0}, "selection must be made on the FIT rows"
    assert res["cos"] == pytest.approx(-1.0), (
        "and the held-out score must be the FIT-chosen knob's, not the grid's "
        "held-out maximum of +1.0"
    )


def test_rel_l2_is_scale_fitted_and_pooled_not_a_per_row_mean() -> None:
    """A variant that is ``g_true`` times a constant is DIRECTIONALLY perfect, and
    the reported L2 must say so once the single global scale is fitted out."""
    rows = [_row(q=[300.0, 120.0, 0.0, -80.0, -220.0, -500.0],
                 p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.06, 0.04, 0.02], game_id=i)
            for i in range(4)]
    def fn(r: Any) -> np.ndarray:
        return 1000.0 * r.g_true

    s = rss.score(rows, fn)
    c = s["dot"] / s["hat2"]
    assert c == pytest.approx(1e-3, rel=1e-9)
    assert rss.rel_l2(rows, fn, c) == pytest.approx(0.0, abs=1e-9)
    # unscaled it is enormous, which is exactly why the fit is not optional
    assert rss.rel_l2(rows, fn, 1.0) > 100.0


def test_pooled_l2_is_not_moved_by_a_near_zero_norm_row() -> None:
    """The failure that once made a per-row relative mean read 4557."""
    normal = _row(q=[300.0, 120.0, 0.0, -80.0, -220.0, -500.0],
                  p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.06, 0.04, 0.02])
    flat = _row(q=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                p=[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.04, 0.03, 0.03])
    assert np.linalg.norm(flat.g_true) < 1e-12
    fn = lambda r: rss.grad_constant_tail(r, alpha=1.0)  # noqa: E731
    assert rss.rel_l2([normal, flat], fn, 1.0) < 5.0


def test_edge_pinned_hyperparameters_are_reported() -> None:
    """A knob resting on the boundary of its own grid is a defect, not a result."""
    grid = [{"t": t} for t in (1.0, 2.0, 4.0)]
    assert rss._edge_pinned({"t": 4.0}, grid) == ["t=4"]
    assert rss._edge_pinned({"t": 1.0}, grid) == ["t=1"]
    assert rss._edge_pinned({"t": 2.0}, grid) == []


# --------------------------------------------------------------- negative control


def test_permutation_control_kills_order_and_spares_membership() -> None:
    """The control's own validity, in both directions.

    An objective reading SF's ORDERING among the six must collapse. One reading
    only SET MEMBERSHIP (surfaced vs not) must be EXACTLY unchanged — so the floor
    for a combination is that term's own cosine, never zero.
    """
    rows = [_row(q=[300.0, 120.0, 0.0, -80.0, -220.0, -500.0],
                 p=[0.3, 0.2, 0.15, 0.1, 0.08, 0.05, 0.06, 0.04, 0.02], game_id=i)
            for i in range(40)]
    sh = rss.permute_scores(rows, seed=0)
    assert [r.game_id for r in sh] == [r.game_id for r in rows]
    for a, b in zip(rows, sh):
        assert sorted(b.q[b.s]) == sorted(a.q[a.s]), "the SET of scores is preserved"
        assert np.array_equal(a.s, b.s), "surfaced membership must not move"
        assert np.array_equal(a.g_true, b.g_true), "the ANSWER must not be shuffled"
    permuted = rss.score(sh, rss.grad_tail_order_limit)["cos"]
    assert permuted == pytest.approx(rss.score(rows, rss.grad_tail_order_limit)["cos"])
    within = rss.score(rows, rss.grad_observed_pairwise)["cos"]
    assert abs(rss.score(sh, rss.grad_observed_pairwise)["cos"]) < abs(within)


# ------------------------------------------------------------------------- banking


def test_bank_roundtrip_preserves_every_field(tmp_path: Path) -> None:
    """A later session must be able to re-aggregate without the GPU."""
    tcs = load_script_module("tail_censor_screen.py")
    legal = np.array([3, 5, 7, 9, 11, 13, 15, 17])
    score_map = {3: 100.0, 5: 40.0, 7: 0.0, 9: -60.0, 11: -150.0, 13: -400.0}
    regret = {m: (100.0 - s) / 1000.0 for m, s in score_map.items()}
    row = tcs.Row(legal, regret, [3, 5, 7, 9, 11, 13], [], max(regret.values()),
                  game_id=42, score=score_map)
    row.prior = np.full(8, 0.125)
    row.logits = np.arange(8, dtype=np.float64)
    out = tmp_path / "b.npz"
    rss.bank_rows([row], {"k": 6, "checkpoint": "c", "shard_names": ["s"]}, out)
    got, meta = rss.load_bank(out)
    assert meta["k"] == 6
    assert len(got) == 1
    assert got[0].game_id == 42
    assert got[0].z == pytest.approx(np.arange(8, dtype=np.float64))
    assert int(got[0].s.sum()) == 6
    # ⚑ an unsurfaced move is NaN, not a score of zero — a zero would be a value
    # the screen never observed, which is the exact fabrication it refuses to make
    assert np.isnan(got[0].q[~got[0].s]).all()
