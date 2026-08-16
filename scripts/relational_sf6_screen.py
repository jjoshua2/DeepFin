#!/usr/bin/env python3
"""Does a RELATIONAL loss over SF's six surfaced moves beat the best constant tail?

The question. At MultiPV 6 we observe six scored moves and nothing else. The live
training rule invents a value for every unsurfaced move
(``finalize._build_sf_p0_regret_vector``'s ``(worst + 1) / 2``); the fitted
constant tail ``r_k + (alpha/2)(1 - r_k)`` invents a smaller one. A RELATIONAL
objective invents no value at all: it supervises ORDER (move i outranks move j),
MARGIN (by roughly this cp gap), and INDIFFERENCE (these two are within noise).
Does that recover more of the full-information policy gradient?

⚑ THIS IS A MEASUREMENT, NOT A TRAINING CHANGE. Nothing here touches the run.

THE REFERENCE. The live loss is ``L = sum_a p_a r_a`` with
``p = softmax(legal-masked policy_own)`` (``train/losses.py:888-890``), so
``dL/dz = p * (r - E_p[r])``. ``g_true`` uses the FULL MultiPV-40 regrets from the
wide-label era; every candidate is scored by cosine against it on the same rows,
via ``scripts/tail_censor_screen.py``'s collector and estimator so the numbers are
the same statistic on the same population.

⚑⚑ THE SUBSPACE CEILING COMES FIRST. A pairwise loss over only the six surfaced
logits has a gradient supported on six coordinates while ``g_true`` has mass on
all ~27 legal moves, so its cosine is bounded by
``||g_true restricted to S|| / ||g_true||`` before any design choice is made. That
bound is ESCAPABLE: "move X was not surfaced" is itself an observation — it means
SF ranked X below rank 6, i.e. ``r_X >= r_k``. An UNMARGINED surfaced-over-tail
constraint therefore places legitimate gradient on tail coordinates without
inventing a value, which relaxes the bound to a sign-constrained cone. Both
ceilings are computed and printed before any objective is scored.

⚑ EVERY OBJECTIVE HERE DEPENDS ON z ONLY THROUGH DIFFERENCES, so its gradient sums
to zero. ``g_true`` does too. The sum-zero variants of both ceilings are therefore
the binding ones; the unconstrained variants are printed as the looser bound.

Two-step, because the GPU belongs to a live training run:

    # ONE GPU touch, capped at 10% of the device: bank priors + logits to .npz,
    # and reproduce tail_censor_screen's four reference cosines on the way past.
    PYTHONPATH=. python3 scripts/relational_sf6_screen.py bank \\
        --replay-dir <wide-era>/replay_shards --checkpoint <ckpt> \\
        --max-shards 6 --k 6 --bank-out scratchpad/sf6_bank.npz

    # pure numpy on CPU, no model, repeatable for free
    PYTHONPATH=. python3 scripts/relational_sf6_screen.py screen \\
        --bank scratchpad/sf6_bank.npz --json-out scratchpad/relational.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np

from scripts.diagnostic_replay_utils import select_shards
from scripts.tail_censor_screen import (
    attach_prior,
    analyse,
    check_invariants,
    collect,
    gradient,
    tail_value,
)

REFERENCE_GRID = ("r_k censor", "midpoint (live)", "alpha_value", "alpha_grad")
# The incumbent constant-tail family, as named in this screen's own variant table.
REFERENCE_NAMES = ("r_k censor (a=0)", "midpoint (live, a=1)", "alpha_value",
                   "alpha_grad (HURDLE)")

# The four cosines the productionised screen reported on n=5059, checkpoint_000218,
# k=6. Reproducing them is the precondition for believing anything below.
REFERENCE_COSINES = {
    "midpoint (live)": 0.8552,
    "r_k censor": 0.8681,
    "alpha_value": 0.9023,
    "alpha_grad": 0.9054,
}
# The hurdle: the best a CONSTANT tail can do, by closed-form least squares.
HURDLE = REFERENCE_COSINES["alpha_grad"]


# --------------------------------------------------------------------------- bank


def bank_rows(rows: list[Any], meta: dict[str, Any], out: Path) -> None:
    """Flatten the ragged per-row arrays into one npz.

    ⚑ BANK THE DUMP, NOT JUST THE NUMBER. Everything downstream is pure numpy on
    this file, so a later session can re-aggregate — or disagree — without the GPU.
    """
    offsets = [0]
    legal: list[np.ndarray] = []
    prior: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    r_true: list[np.ndarray] = []
    q_cp: list[np.ndarray] = []
    surfaced: list[np.ndarray] = []
    game_id: list[int] = []
    r_k: list[float] = []
    for row in rows:
        if row.prior is None or row.logits is None:
            continue
        idx = np.asarray(row.legal, dtype=np.int64)
        s = np.array([int(m) in set(row.surfaced) for m in idx], dtype=bool)
        legal.append(idx.astype(np.int32))
        prior.append(np.asarray(row.prior, dtype=np.float64))
        logits.append(np.asarray(row.logits, dtype=np.float64))
        r_true.append(np.array([row.regret.get(int(m), row.r_k) for m in idx], dtype=np.float64))
        # NaN = "SF never scored this move", which is NOT the same as "scored badly".
        q_cp.append(np.array([row.score.get(int(m), np.nan) for m in idx], dtype=np.float64))
        surfaced.append(s)
        game_id.append(int(row.game_id))
        r_k.append(float(row.r_k))
        offsets.append(offsets[-1] + int(idx.size))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        offsets=np.asarray(offsets, dtype=np.int64),
        legal=np.concatenate(legal), prior=np.concatenate(prior),
        logits=np.concatenate(logits), r_true=np.concatenate(r_true),
        q_cp=np.concatenate(q_cp), surfaced=np.concatenate(surfaced),
        game_id=np.asarray(game_id, dtype=np.int64), r_k=np.asarray(r_k, dtype=np.float64),
        meta=np.array([json.dumps(meta, sort_keys=True)]),
    )


@dataclass
class RowData:
    """One position, as pure numpy. ``s`` marks SF's six surfaced moves."""

    game_id: int
    p: np.ndarray
    z: np.ndarray
    r: np.ndarray
    q: np.ndarray
    s: np.ndarray
    r_k: float
    g_true: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.g_true = gradient(self.p, self.r)


def load_bank(path: Path) -> tuple[list[RowData], dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    off = z["offsets"]
    meta = json.loads(str(z["meta"][0]))
    rows: list[RowData] = []
    for i in range(off.size - 1):
        a, b = int(off[i]), int(off[i + 1])
        rows.append(RowData(
            game_id=int(z["game_id"][i]), p=z["prior"][a:b], z=z["logits"][a:b],
            r=z["r_true"][a:b], q=z["q_cp"][a:b], s=z["surfaced"][a:b],
            r_k=float(z["r_k"][i]),
        ))
    return rows, meta


# ---------------------------------------------------------------------- ceilings


def project_sum_zero_cone(g: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``g`` onto ``{v : v[tail] >= 0, sum(v) == 0}``.

    KKT gives ``v_S = g_S - lam`` and ``v_T = max(g_T - lam, 0)``; the residual
    ``sum(v)`` is non-increasing in ``lam``, so one bisection finds it. The cone is
    the tightest honest description of what an order-only tail claim permits: the
    SIGN of the tail gradient is observed (unsurfaced means worse), its MAGNITUDE
    is not, and every difference-based objective sums to zero.
    """
    if g.size == 0:
        return g.copy()
    lo, hi = float(g.min()) - 1.0, float(g.max()) + 1.0

    def resid(lam: float) -> float:
        v = np.where(tail, np.maximum(g - lam, 0.0), g - lam)
        return float(v.sum())

    if not np.any(~tail):
        # Every coordinate is sign-constrained; sum-zero then forces v == 0.
        return np.zeros_like(g)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    return np.where(tail, np.maximum(g - lam, 0.0), g - lam)


def ceilings(rows: list[RowData]) -> dict[str, Any]:
    """The four subspace ceilings, per-row and pooled.

    ⚑ The hurdle table's statistic is the MEAN of per-row cosines, so the
    comparable ceiling is the MEAN of per-row restriction ratios. The
    sqrt-of-summed-squares pooling is reported alongside because it is the
    energy-weighted reading and the two answer different questions.
    """
    names = ("top6_only", "top6_only_sumzero", "top6_plus_order_tail",
             "top6_plus_order_tail_sumzero")
    per_row: dict[str, list[float]] = {n: [] for n in names}
    num: dict[str, float] = dict.fromkeys(names, 0.0)
    den = 0.0
    clipped_rows = 0
    tail_neg_energy = 0.0
    tail_energy = 0.0
    tail_mass: list[float] = []
    for row in rows:
        g = row.g_true
        n2 = float(g @ g)
        if n2 <= 0.0:
            continue
        den += n2
        tail = ~row.s
        tail_mass.append(float(row.p[tail].sum()))
        gt = g[tail]
        tail_energy += float(gt @ gt)
        neg = gt[gt < 0.0]
        tail_neg_energy += float(neg @ neg)
        if neg.size:
            clipped_rows += 1
        cand = {
            "top6_only": np.where(row.s, g, 0.0),
            "top6_only_sumzero": _sum_zero_on(g, row.s),
            "top6_plus_order_tail": np.where(row.s, g, np.maximum(g, 0.0)),
            "top6_plus_order_tail_sumzero": project_sum_zero_cone(g, tail),
        }
        for name, v in cand.items():
            e = float(v @ v)
            num[name] += e
            per_row[name].append((e / n2) ** 0.5)
    return {
        "n_rows": len(per_row["top6_only"]),
        "mean_prior_mass_on_tail": float(np.mean(tail_mass)),
        "median_prior_mass_on_tail": float(np.median(tail_mass)),
        "p90_prior_mass_on_tail": float(np.percentile(tail_mass, 90)),
        "mean_per_row": {n: float(np.mean(per_row[n])) for n in names},
        "median_per_row": {n: float(np.median(per_row[n])) for n in names},
        "p10_per_row": {n: float(np.percentile(per_row[n], 10)) for n in names},
        "pooled": {n: float((num[n] / den) ** 0.5) for n in names},
        "rows_with_negative_tail_component": clipped_rows,
        "tail_negative_energy_fraction": float(tail_neg_energy / max(tail_energy, 1e-300)),
    }


def _sum_zero_on(g: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Best sum-zero vector supported on ``keep``: recentre ``g[keep]``."""
    v = np.zeros_like(g)
    if not np.any(keep):
        return v
    v[keep] = g[keep] - float(g[keep].mean())
    return v


# -------------------------------------------------------------------- objectives


def _pair_matrices(row: RowData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(idx, D, A)``: surfaced indices, cp gaps ``q_i - q_j``, logit gaps."""
    idx = np.flatnonzero(row.s)
    qs = row.q[idx]
    zs = row.z[idx]
    return idx, qs[:, None] - qs[None, :], zs[:, None] - zs[None, :]


def _scatter(row: RowData, idx: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Sum an ANTISYMMETRIC pair matrix into a full-length gradient.

    ``grad[i] = sum_j M_ij`` reproduces the sum over unordered pairs exactly, and
    antisymmetry is what makes that identity hold — every objective here is a
    function of ``z_i - z_j``, so its pair matrix is antisymmetric by construction.
    """
    g = np.zeros_like(row.p)
    g[idx] = m.sum(axis=1)
    return g


def grad_hinge(row: RowData, *, t: float, prior_weighted: bool = False) -> np.ndarray:
    """(1) Pairwise hinge, ORDER ONLY: ``softplus(-sign(d_ij)(z_i - z_j)/t)``.

    A tie (``d_ij == 0``) contributes exactly nothing, which is the correct
    reading of "order only" — SF asserted no order between them.
    """
    idx, d, a = _pair_matrices(row)
    u = np.sign(d)
    m = -(u / t) * _sigmoid(-u * a / t)
    if prior_weighted:
        w = row.p[idx]
        m = m * (w[:, None] * w[None, :])
    np.fill_diagonal(m, 0.0)
    return _scatter(row, idx, m)


def grad_margin(
    row: RowData, *, beta: float, kappa: float, shape: str = "tanh",
    tau: float = 0.0, prior_weighted: bool = False,
) -> np.ndarray:
    """(2)/(3) Pairwise margin regression, optionally indifference-weighted.

    ``f(d) = beta*tanh(d/kappa)`` or ``clip(beta*d, +-kappa)``. ``tau > 0`` turns on
    ``w_ij = min(1, |d_ij|/tau)``, which is the OVER-SHARPENING hypothesis: a cp
    gap far below SF's own resolution should not force the policy to separate.
    """
    idx, d, a = _pair_matrices(row)
    f = beta * np.tanh(d / kappa) if shape == "tanh" else np.clip(beta * d, -kappa, kappa)
    m = 2.0 * (a - f)
    if tau > 0.0:
        m = m * np.minimum(1.0, np.abs(d) / tau)
    if prior_weighted:
        w = row.p[idx]
        m = m * (w[:, None] * w[None, :])
    np.fill_diagonal(m, 0.0)
    return _scatter(row, idx, m)


def grad_listwise(row: RowData, *, temp_cp: float) -> np.ndarray:
    """(4) Listwise softmax cross-entropy over the six.

    ⚑ FLAGGED AS NOT RELATIONAL. A temperature turns SF's scores into a
    DISTRIBUTION, which invents exactly the kind of magnitude claim the rest of
    this screen refuses to make. It is here as a reference point, not a candidate.
    """
    idx = np.flatnonzero(row.s)
    qs = row.q[idx]
    target = _softmax(qs / temp_cp)
    ps = _softmax(row.z[idx])
    g = np.zeros_like(row.p)
    g[idx] = ps - target
    return g


def grad_surfaced_over_tail(
    row: RowData, *, t: float, prior_weighted: bool = False,
) -> np.ndarray:
    """(5) UNMARGINED ``softplus(-(z_i - z_j)/t)`` for every surfaced/unsurfaced pair.

    ⚑ THIS IS THE ESCAPE FROM THE TOP-6 CEILING AND IT INVENTS NOTHING. MultiPV 6
    returning move i and not move j is an observation that ``r_j >= r_i``. Only the
    ORDER is claimed; no value is assigned to j.
    """
    si = np.flatnonzero(row.s)
    ti = np.flatnonzero(~row.s)
    g = np.zeros_like(row.p)
    if si.size == 0 or ti.size == 0:
        return g
    a = row.z[si][:, None] - row.z[ti][None, :]
    m = -_sigmoid(-a / t) / t
    if prior_weighted:
        m = m * (row.p[si][:, None] * row.p[ti][None, :])
    g[si] += m.sum(axis=1)
    g[ti] -= m.sum(axis=0)
    return g


def grad_observed_pairwise(row: RowData, *, scale: float = 1.0) -> np.ndarray:
    """(8) The reference gradient's OWN pairwise form, restricted to observed pairs.

    ⚑⚑ THE IDENTITY THAT EXPLAINS THE WHOLE SCREEN.
    ``dL/dz_i = p_i(r_i - E_p[r]) = sum_j p_i p_j (r_i - r_j)`` — the full-information
    gradient IS a pairwise object with weight ``p_i p_j`` and pair term the regret
    DIFFERENCE. For ``i, j`` both surfaced that difference is OBSERVED, so this
    variant is the exact ``S x S`` block of ``g_true`` with nothing invented. What it
    necessarily omits is the ``S x T`` block, whose pair term ``r_i - r_j`` needs a
    tail MAGNITUDE. That omission is the entire subject of this experiment.
    """
    idx = np.flatnonzero(row.s)
    rs = row.r[idx]
    ps = row.p[idx]
    m = scale * (ps[:, None] * ps[None, :]) * (rs[:, None] - rs[None, :])
    np.fill_diagonal(m, 0.0)
    return _scatter(row, idx, m)


def grad_borda_limit(row: RowData) -> np.ndarray:
    """The t -> infinity limit that EVERY pure-order objective here converges to.

    ``softplus(-u*a/t) -> log2 - u*a/(2t)``, so the hinge's gradient tends to
    ``-sign(d_ij)/(2t)``: a constant force per pair, i.e. minus the BORDA score of
    each surfaced move. A tanh margin with ``beta -> infinity`` degenerates to the
    same step. Reporting it closes the edge-pinning question analytically instead
    of chasing the grid: it is where those variants are heading, exactly.
    """
    idx = np.flatnonzero(row.s)
    qs = row.q[idx]
    m = -np.sign(qs[:, None] - qs[None, :])
    np.fill_diagonal(m, 0.0)
    return _scatter(row, idx, m)


def grad_tail_order_limit(row: RowData, *, prior_weighted: bool = True) -> np.ndarray:
    """The t -> infinity limit of the surfaced-over-tail constraint.

    ⚑⚑ THIS IS THE RESULT THAT EXPLAINS THE SCREEN. In the limit the prior-weighted
    constraint becomes ``-p_i * mass(T)`` on surfaced and ``+p_j * mass(S)`` on tail —
    a gradient proportional to the prior with ONE global magnitude, which is
    precisely the shape of a CONSTANT-tail rule. So the unmargined order constraint
    does not escape the constant tail asymptotically; it BECOMES one, with its
    single free scale playing the role of ``alpha``.
    """
    g = np.zeros_like(row.p)
    w = row.p if prior_weighted else np.ones_like(row.p)
    ms = float(w[row.s].sum())
    mt = float(w[~row.s].sum())
    g[row.s] = -w[row.s] * mt
    g[~row.s] = w[~row.s] * ms
    return g


def grad_constant_tail(row: RowData, *, alpha: float) -> np.ndarray:
    """The incumbent family: fill every unsurfaced move with ``r_tail(alpha)``."""
    fill = tail_value(row.r_k, alpha)
    return gradient(row.p, np.where(row.s, row.r, fill))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.clip(x, -60.0, 60.0)))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def combine(*parts: tuple[Callable[..., np.ndarray], dict[str, Any], float]):
    """Weighted sum of objective gradients — the (6) family."""

    def g(row: RowData) -> np.ndarray:
        out = np.zeros_like(row.p)
        for fn, kw, w in parts:
            out = out + w * fn(row, **kw)
        return out

    return g


# ------------------------------------------------------------------------ scoring


def score(rows: list[RowData], fn: Callable[[RowData], np.ndarray]) -> dict[str, Any]:
    """Per-row cosines plus the sums a scale fit and a pooled L2 need."""
    cos: list[float] = []
    dot = 0.0
    hat2 = 0.0
    true2 = 0.0
    press: list[float] = []
    per_row: list[float] = []
    for row in rows:
        g = row.g_true
        nt = float(np.linalg.norm(g))
        if nt <= 0.0:
            continue
        h = fn(row)
        nh = float(np.linalg.norm(h))
        c = float(h @ g / (nh * nt)) if nh > 0.0 else 0.0
        cos.append(c)
        per_row.append(c)
        dot += float(h @ g)
        hat2 += nh ** 2
        true2 += nt ** 2
        press.append(float(h[~row.s].sum()))
    return {
        "cos": float(np.mean(cos)) if cos else float("nan"),
        "cos_median": float(np.median(cos)) if cos else float("nan"),
        "n": len(cos), "dot": dot, "hat2": hat2, "true2": true2,
        "press": press, "per_row": per_row,
    }


def rel_l2(rows: list[RowData], fn: Callable[[RowData], np.ndarray], c: float) -> float:
    """Pooled relative L2 AFTER a single global scale ``c``.

    ⚑ Without the scale fit this number is meaningless across objectives whose
    natural magnitudes differ by orders of magnitude, and it would rank them by
    accidental scale rather than by direction. ``c`` is fitted on the FIT split.
    ⚑ Pooled by summed squares, never as a per-row mean: one near-zero-norm row
    once made that mean read 4557.
    """
    num = 0.0
    den = 0.0
    for row in rows:
        g = row.g_true
        n2 = float(g @ g)
        if n2 <= 0.0:
            continue
        d = c * fn(row) - g
        num += float(d @ d)
        den += n2
    return float((num / den) ** 0.5)


def split_by_game(rows: list[RowData], frac: float = 0.5) -> tuple[list[RowData], list[RowData]]:
    """DISJOINT split on ``game_id``. Rows inside one game are correlated, so a
    row-level split would let a fitted hyperparameter see its own test set."""
    fit: list[RowData] = []
    held: list[RowData] = []
    for row in rows:
        h = hashlib.sha256(str(row.game_id).encode()).digest()
        u = int.from_bytes(h[:8], "big") / float(1 << 64)
        (fit if u < frac else held).append(row)
    return fit, held


def fit_and_score(
    rows_fit: list[RowData], rows_held: list[RowData],
    build: Callable[..., Callable[[RowData], np.ndarray]], grid: list[dict[str, Any]],
    grid_rows: int = 0,
) -> dict[str, Any]:
    """Pick hyperparameters on FIT by mean cosine; report everything on HELD-OUT.

    ``grid_rows`` searches the grid on a deterministic SUBSAMPLE of the fit split so
    a grid wide enough that nothing pins at an edge stays affordable beside a live
    run. The subsample is still strictly inside the fit split, so the held-out rows
    remain untouched by the selection; only the selection's own precision drops.
    """
    search = rows_fit
    if 0 < grid_rows < len(rows_fit):
        step = len(rows_fit) / grid_rows
        search = [rows_fit[int(i * step)] for i in range(grid_rows)]
    best_hp: dict[str, Any] = {}
    best = -2.0
    for hp in grid:
        r = score(search, build(**hp))
        if r["cos"] > best:
            best, best_hp = r["cos"], hp
    fn = build(**best_hp)
    f = score(rows_fit, fn)
    c = f["dot"] / f["hat2"] if f["hat2"] > 0.0 else 0.0
    h = score(rows_held, fn)
    return {
        "hp": best_hp, "cos_fit": f["cos"], "cos": h["cos"], "cos_median": h["cos_median"],
        "n_held": h["n"], "scale": c, "rel_l2_pooled": rel_l2(rows_held, fn, c),
        "tail_pressure": float(np.mean(h["press"])) * c,
        "per_row": h["per_row"],
    }


def _edge_pinned(hp: dict[str, Any], grid: list[dict[str, Any]]) -> list[str]:
    """Hyperparameters the fit drove to the boundary of their own search range.

    ⚑ A pin is a REPORTABLE DEFECT, not a result: it says the optimum is outside
    the range searched, so the reported score is a lower bound on that objective
    and the shape of the winner may be an artifact of where the grid stopped.
    """
    pinned: list[str] = []
    for key, val in hp.items():
        vals = sorted({g[key] for g in grid})
        if len(vals) > 1 and (val == vals[0] or val == vals[-1]):
            pinned.append(f"{key}={val:g}")
    return pinned


def true_tail_pressure(rows: list[RowData]) -> float:
    v = [float(r.g_true[~r.s].sum()) for r in rows if float(r.g_true @ r.g_true) > 0.0]
    return float(np.mean(v))


# ------------------------------------------------------------------------ variants


def build_variants(alpha_value: float, alpha_grad: float) -> dict[str, Any]:
    """(objective name) -> (builder, hyperparameter grid). Empty grid = no fit."""
    # ⚑ beta is in LOGIT units and must be able to REACH the net's own logit gaps
    # (order 1-10 on a net this sharp). The first cut of this grid topped out at
    # 0.05, every margin variant pinned to that edge, and the objective degenerated
    # into "flatten every logit" — a force that ANTI-correlates with the reference
    # gradient. The negative control caught it; the grid bound was the defect.
    tt = [0.25, 1.0, 4.0, 16.0, 64.0, 256.0, 1024.0, 4096.0]
    beta = [0.05, 0.5, 2.0, 8.0, 32.0, 128.0, 512.0]
    kap = [12.5, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    taus = [10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    lam = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    v: dict[str, Any] = {
        "r_k censor (a=0)": (lambda: (lambda r: grad_constant_tail(r, alpha=0.0)), [{}]),
        "midpoint (live, a=1)": (lambda: (lambda r: grad_constant_tail(r, alpha=1.0)), [{}]),
        "alpha_value": (lambda: (lambda r: grad_constant_tail(r, alpha=alpha_value)), [{}]),
        "alpha_grad (HURDLE)": (lambda: (lambda r: grad_constant_tail(r, alpha=alpha_grad)), [{}]),
        "1 pairwise hinge": (
            lambda t: (lambda r: grad_hinge(r, t=t)), [{"t": t} for t in tt]),
        "2a margin tanh": (
            lambda beta, kappa: (lambda r: grad_margin(r, beta=beta, kappa=kappa)),
            [{"beta": b, "kappa": k} for b in beta for k in kap]),
        "2b margin clipped": (
            lambda beta, kappa: (
                lambda r: grad_margin(r, beta=beta, kappa=kappa, shape="clip")),
            [{"beta": b / 100.0, "kappa": k}
             for b in beta for k in (0.5, 2.0, 8.0, 32.0, 128.0)]),
        "3 indifference margin": (
            lambda beta, kappa, tau: (
                lambda r: grad_margin(r, beta=beta, kappa=kappa, tau=tau)),
            [{"beta": b, "kappa": k, "tau": t} for b in beta for k in kap for t in taus]),
        "4 listwise CE (invents)": (
            lambda temp_cp: (lambda r: grad_listwise(r, temp_cp=temp_cp)),
            [{"temp_cp": t} for t in (25.0, 50.0, 100.0, 200.0, 400.0)]),
        "5 surfaced>tail order": (
            lambda t: (lambda r: grad_surfaced_over_tail(r, t=t)), [{"t": t} for t in tt]),
        "8 observed pairwise (SxS)": (
            lambda: grad_observed_pairwise, [{}]),
        "1inf Borda limit (t->inf)": (lambda: grad_borda_limit, [{}]),
        "5inf tail-order limit": (
            lambda: grad_tail_order_limit, [{}]),
    }
    # Combination grids are coarser on the shared axes so the joint search stays
    # affordable next to a live run; the components' own grids are the fine ones.
    cb, ct, cw = [0.5, 8.0, 128.0], [1.0, 16.0, 256.0, 4096.0], lam
    v["6 = 3 + w*5"] = (
        lambda beta, kappa, tau, t, w: combine(
            (grad_margin, {"beta": beta, "kappa": kappa, "tau": tau}, 1.0),
            (grad_surfaced_over_tail, {"t": t}, w)),
        [{"beta": b, "kappa": 200.0, "tau": ta, "t": t, "w": w}
         for b in cb for ta in (25.0, 100.0) for t in ct for w in cw],
    )
    v["7 prior-weighted 3 + w*5"] = (
        lambda beta, kappa, tau, t, w: combine(
            (grad_margin,
             {"beta": beta, "kappa": kappa, "tau": tau, "prior_weighted": True}, 1.0),
            (grad_surfaced_over_tail, {"t": t, "prior_weighted": True}, w)),
        [{"beta": b, "kappa": 200.0, "tau": ta, "t": t, "w": w}
         for b in cb for ta in (25.0, 100.0) for t in ct for w in cw],
    )
    v["9 = 8 + w*5(prior)"] = (
        lambda t, w: combine(
            (grad_observed_pairwise, {}, 1.0),
            (grad_surfaced_over_tail, {"t": t, "prior_weighted": True}, w)),
        [{"t": t, "w": w} for t in tt for w in lam],
    )
    v["9inf = 8 + w*5inf"] = (
        lambda w: combine(
            (grad_observed_pairwise, {}, 1.0), (grad_tail_order_limit, {}, w)),
        [{"w": w} for w in lam],
    )
    # ⚑ ABLATION for hypothesis (3): the ONLY difference from "7" is tau, so the
    # delta between them isolates indifference weighting rather than confounding
    # it with the prior weighting and the tail term.
    v["7-noindiff (tau off)"] = (
        lambda beta, kappa, t, w: combine(
            (grad_margin, {"beta": beta, "kappa": kappa, "tau": 0.0,
                           "prior_weighted": True}, 1.0),
            (grad_surfaced_over_tail, {"t": t, "prior_weighted": True}, w)),
        [{"beta": b, "kappa": 200.0, "t": t, "w": w}
         for b in cb for t in ct for w in cw],
    )
    return v


def permute_scores(rows: list[RowData], seed: int = 0) -> list[RowData]:
    """NEGATIVE CONTROL: permute SF's six scores WITHIN each row.

    Everything else — the prior, the logits, the surfaced SET, ``g_true`` — is
    untouched, so any objective reading SF's ORDERING must collapse. An objective
    that does not collapse is reading structure that is not SF's ranking, and its
    result is void. ⚑ A term that does not depend on the within-set order (the
    surfaced-over-tail constraint) survives BY CONSTRUCTION, so the floor for a
    combination is that term's own cosine, not zero.
    """
    rng = np.random.default_rng(seed)
    out: list[RowData] = []
    for row in rows:
        idx = np.flatnonzero(row.s)
        perm = rng.permutation(idx.size)
        q = row.q.copy()
        r = row.r.copy()
        q[idx] = row.q[idx][perm]
        r[idx] = row.r[idx][perm]
        new = RowData(game_id=row.game_id, p=row.p, z=row.z, r=r, q=q, s=row.s, r_k=row.r_k)
        # ⚑ g_true must stay the TRUE gradient: the labels are shuffled, not the answer.
        new.g_true = row.g_true
        out.append(new)
    return out


# ---------------------------------------------------------------------------- cli


def cmd_bank(args: argparse.Namespace) -> int:
    shards: list[Path] = []
    for d in args.replay_dir:
        shards.extend(select_shards(Path(d), args.max_shards))
    scan: dict[str, Any] = {
        "shard_names": [p.name for p in shards], "rows_scanned": 0,
        "coverage_sum": 0.0, "coverage_n": 0, "unscored_mass_sum": 0.0,
        "surfaced_not_legal": 0, "skipped_shards": [], "skipped_shards_omitted": 0,
    }
    rows, planes = collect(shards, scan, args.k)
    if not rows:
        raise SystemExit("no rows with a parent MultiPV block wider than --k")
    coverage, unscored = check_invariants(scan, len(rows))
    print(f"shards {len(shards)}   rows scanned {scan['rows_scanned']}   analysed {len(rows)}")
    print(f"  coverage {coverage:.4f}   unscored policy mass {unscored:.6f}")
    device = attach_prior(rows, planes, args.checkpoint, args.batch)
    print(f"  priors from {args.checkpoint} on {device}")

    res = analyse(rows, REFERENCE_GRID)
    print("\nREPRODUCTION of scripts/tail_censor_screen.py (must match to 4 dp)")
    print(f"  n_rows {res['n_rows']}   alpha_value {res['alpha_value']:.4f}"
          f"   alpha_grad {res['alpha_grad']:.4f}")
    ok = True
    for name, expect in REFERENCE_COSINES.items():
        got = res["variants"][name]["cos"]
        hit = abs(got - expect) < 5e-5
        ok = ok and hit
        print(f"  {name:20s} {got:.4f}  expected {expect:.4f}  {'OK' if hit else 'MISMATCH'}")
    if not ok:
        print("\n⚑ REPRODUCTION FAILED — everything downstream of this is meaningless.")

    bank_rows(rows, {
        "shard_names": scan["shard_names"], "k": args.k, "checkpoint": args.checkpoint,
        "coverage": coverage, "unscored_mass": unscored, "device": device,
        "reference": {n: res["variants"][n]["cos"] for n in REFERENCE_GRID},
        "alpha_value": res["alpha_value"], "alpha_grad": res["alpha_grad"],
        "reproduced": ok, "n_rows_analysed": res["n_rows"],
    }, Path(args.bank_out))
    print(f"\nbanked -> {args.bank_out}")
    return 0 if ok else 1


def cmd_screen(args: argparse.Namespace) -> int:
    rows, meta = load_bank(Path(args.bank))
    print(f"bank {args.bank}: {len(rows)} rows, {len({r.game_id for r in rows})} games")
    print(f"  checkpoint {meta['checkpoint']}   k {meta['k']}   "
          f"shards {len(meta['shard_names'])}")
    print(f"  banked reference cosines {meta['reference']}   reproduced={meta['reproduced']}")

    ceil = ceilings(rows)
    print(f"\nSUBSPACE CEILINGS  (n={ceil['n_rows']}, the bound BEFORE any loss design)")
    print(f"  {'support':32s} {'mean/row':>9s} {'median':>8s} {'p10':>8s} {'pooled':>8s}")
    for name in ceil["mean_per_row"]:
        print(f"  {name:32s} {ceil['mean_per_row'][name]:9.4f}"
              f" {ceil['median_per_row'][name]:8.4f} {ceil['p10_per_row'][name]:8.4f}"
              f" {ceil['pooled'][name]:8.4f}")
    print(f"  rows with a NEGATIVE tail g_true component: "
          f"{ceil['rows_with_negative_tail_component']} / {ceil['n_rows']}"
          f"   (negative share of tail energy {ceil['tail_negative_energy_fraction']:.4f})")
    print(f"  prior mass OUTSIDE SF's six: mean {ceil['mean_prior_mass_on_tail']:.4f}"
          f"  median {ceil['median_prior_mass_on_tail']:.4f}"
          f"  p90 {ceil['p90_prior_mass_on_tail']:.4f}")
    print(f"  published HURDLE (full population): {HURDLE:.4f} — ⚑ NOT the held-out"
          " comparison; the constant tails are re-scored on the held-out rows below.")

    fit, held = split_by_game(rows, args.fit_frac)
    print(f"\nDISJOINT SPLIT by game_id: fit {len(fit)} rows / held-out {len(held)} rows")

    variants = build_variants(meta["alpha_value"], meta["alpha_grad"])
    out: dict[str, Any] = {}
    for name, (build, grid) in variants.items():
        out[name] = fit_and_score(fit, held, build, grid, args.grid_rows)
        edge = _edge_pinned(out[name]["hp"], grid)
        print(f"  scored {name:28s} {len(grid):5d} combos"
              f"{'   ⚑ EDGE-PINNED: ' + ','.join(edge) if edge else ''}")
        out[name]["edge_pinned"] = edge

    tp_true = true_tail_pressure(held)
    print(f"\nHELD-OUT SCORES  (fit on {len(fit)} rows, reported on {len(held)})")
    print(f"  {'variant':28s} {'cos':>8s} {'cos med':>8s} {'relL2':>8s}"
          f" {'tailpress':>10s} {'vs TRUE':>8s}")
    print(f"  {'TRUE (g_true)':28s} {1.0:8.4f} {1.0:8.4f} {0.0:8.4f}"
          f" {tp_true:10.6f} {1.0:7.2f}x")
    for name, r in sorted(out.items(), key=lambda kv: -kv[1]["cos"]):
        print(f"  {name:28s} {r['cos']:8.4f} {r['cos_median']:8.4f} {r['rel_l2_pooled']:8.4f}"
              f" {r['tail_pressure']:10.6f} {r['tail_pressure'] / tp_true:7.2f}x")

    # ⚑ THE HURDLE MUST BE RE-MEASURED ON THE HELD-OUT ROWS. Comparing a held-out
    # relational cosine against the published full-population 0.9054 would be the
    # same-name-different-population error: the constant tails read HIGHER on this
    # subset, so that comparison silently hands the relational side ~0.01 of hurdle.
    constant = {n: out[n]["cos"] for n in REFERENCE_NAMES}
    hurdle_held = max(constant.values())
    relational = {n: r["cos"] for n, r in out.items() if n not in REFERENCE_NAMES}
    winner = max(relational.items(), key=lambda kv: kv[1])
    print(f"\nBEST CONSTANT TAIL on THESE held-out rows: {hurdle_held:.4f}"
          f"   (published full-population figure {HURDLE:.4f})")
    print(f"BEST RELATIONAL: {winner[0]}  cos {winner[1]:.4f}"
          f"  hp {out[winner[0]]['hp']}")
    beats = winner[1] > hurdle_held
    print(f"VERDICT: relational {'BEATS' if beats else 'DOES NOT BEAT'} the best constant"
          f" tail  (delta {winner[1] - hurdle_held:+.4f})")
    print(f"  top-6-only ceiling {ceil['mean_per_row']['top6_only']:.4f}"
          f" vs constant tail {hurdle_held:.4f}: a purely top-6-supported objective is"
          f" {'BOUNDED BELOW' if ceil['mean_per_row']['top6_only'] < hurdle_held else 'not bounded below'}"
          " the incumbent before any design choice.")

    # NEGATIVE CONTROL on the winner and on a pure within-S objective.
    shuffled = permute_scores(held, args.seed)
    control: dict[str, Any] = {}
    for name in (winner[0], "3 indifference margin", "8 observed pairwise (SxS)",
                 "5 surfaced>tail order", "alpha_grad (HURDLE)"):
        build, _ = variants[name]
        fn = build(**out[name]["hp"])
        control[name] = {
            "cos_true_labels": out[name]["cos"],
            "cos_permuted": score(shuffled, fn)["cos"],
        }
    print("\nNEGATIVE CONTROL — SF's six scores PERMUTED within each row")
    print(f"  {'variant':28s} {'true':>8s} {'permuted':>9s} {'collapse':>9s}")
    for name, c in control.items():
        print(f"  {name:28s} {c['cos_true_labels']:8.4f} {c['cos_permuted']:9.4f}"
              f" {c['cos_true_labels'] - c['cos_permuted']:9.4f}")

    res = {
        "meta": meta, "ceilings": ceil, "hurdle": HURDLE,
        "n_fit": len(fit), "n_held": len(held),
        "n_games": len({r.game_id for r in rows}),
        "true_tail_pressure": tp_true,
        "variants": {n: {k: v for k, v in r.items() if k != "per_row"}
                     for n, r in out.items()},
        "negative_control": control,
        "hurdle_held_out": hurdle_held,
        "best_relational": winner[0],
        "best_relational_cos": winner[1],
        "beats_hurdle": bool(beats),
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(res, indent=2, sort_keys=True))
        print(f"\nbanked -> {args.json_out}")
    if args.dump_per_row:
        p = Path(args.dump_per_row)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, names=np.array(list(out)),
            cos=np.stack([np.asarray(out[n]["per_row"]) for n in out]),
            game_id=np.asarray([r.game_id for r in held if float(r.g_true @ r.g_true) > 0.0]),
        )
        print(f"per-row cosines -> {p}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bank", help="ONE capped GPU pass: priors + logits -> npz")
    b.add_argument("--replay-dir", type=Path, required=True, action="append")
    b.add_argument("--checkpoint", type=str, required=True)
    b.add_argument("--k", type=int, default=6)
    b.add_argument("--max-shards", type=int, default=6)
    b.add_argument("--batch", type=int, default=128)
    b.add_argument("--bank-out", type=Path, required=True)
    b.set_defaults(func=cmd_bank)

    s = sub.add_parser("screen", help="pure-numpy CPU screen over the banked rows")
    s.add_argument("--bank", type=Path, required=True)
    s.add_argument("--fit-frac", type=float, default=0.5)
    s.add_argument("--grid-rows", type=int, default=800,
                   help="rows of the FIT split used for the grid search (0 = all)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--json-out", type=Path, default=None)
    s.add_argument("--dump-per-row", type=Path, default=None)
    s.set_defaults(func=cmd_screen)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
