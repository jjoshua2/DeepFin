#!/usr/bin/env python3
"""What is the MultiPV-6 censored tail actually worth? (issue #425)

Takes the wide-MultiPV era's own labels, HIDES everything below rank ``--k``, and
scores the candidate tail representations against the values that were withheld.
No CAST, no played-move proxy, no calibration inversion — the hidden ranks are
observations, so this is a direct measurement rather than an inference.

WHAT IS BEING SCORED. ``selfplay/finalize.py::_build_sf_p0_regret_vector`` gives
every legal move Stockfish did not surface ``(worst_surfaced + 1) / 2``.
``eval/audit.py::move_regrets`` gives them ``worst_surfaced`` as an explicit
optimistic FLOOR. Those are the two ends of one family:

    r_tail(alpha) = r_k + alpha * (r_mid - r_k) = r_k + (alpha/2) * (1 - r_k)

``alpha = 0`` is the audit ruler, ``alpha = 1`` is the live training rule, and
``alpha = 2`` is a maximally bad tail at regret 1.0. ⚑ The range is **[0, 2] and
NOT [0, 1]**: capping at 1 would assume the midpoint is too pessimistic, which is
the very question being asked.

⚑⚑ WEIGHT BY THE NET'S OWN PRIOR, NOT ``policy_target``. The loss this informs is
``sum(softmax(legal-masked policy_own) * r)`` (``train/losses.py:888-890``) — the
network's prior, not the stored search target. Pass ``--checkpoint`` for the
training-relevant answer; without it the screen falls back to ``policy_target``
and says so, which answers a DIFFERENT question.

⚑⚑ P0 ALIGNMENT IS THE TRAP THAT PRODUCED A WRONG ANSWER HERE. A row's OWN
``sf_multipv_raw`` describes the position AFTER that row's move — the opponent's
move set. Stockfish's read of THIS position is the PREVIOUS ply's block. The
first run of this screen weighted row *t*'s policy by row *t*'s own MultiPV and
reported a confident, wrong number. It was caught only by an impossible-valued
diagnostic (a coverage fraction of 1.04), which is why ``--k`` aside, this script
ASSERTS its invariants at runtime instead of trusting a unit test:

  * coverage (scored moves / legal moves) must be <= 1
  * policy mass on legal moves the wide label never scored must be small

MATCHING A SCALAR MEAN IS NOT MATCHING THE GRADIENT. For ``L = p·r`` the logit
gradient is ``dL/dz = p * (r - E_p[r])``, so the screen also scores each censor
against a full-information reference gradient. ``g(alpha)`` is LINEAR in alpha,
so the gradient-optimal alpha is closed form and no search is needed.

Usage:

    PYTHONPATH=. python3 scripts/tail_censor_screen.py \\
        --replay-dir runs/pbt2_small/replay/<wide-era-trial>/replay_shards \\
        --checkpoint data/tail_screen_20260814/checkpoint_000218 \\
        --max-shards 10 --json-out scratchpad/tail_screen.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.replay.shard import SF_CP_SENTINEL, load_shard_arrays
from chess_anti_engine.stockfish.wdl import mate_to_effective_cp
from scripts.diagnostic_replay_utils import record_skipped_shard, select_shards

# Mirrors selfplay/finalize.py. Importing finalize pulls the compiled extension,
# which a plain analysis environment need not have.
SF_OWN_REGRET_CAP_CP = 1000.0

# ⚑ How many plies back the MultiPV block that describes THIS row's position lives.
# Named rather than inlined so ``tests/test_tail_censor_screen.py`` can set it to 0
# and prove that the wrong alignment is caught rather than silently mis-answered.
P0_PARENT_PLY_OFFSET = 1

# A coverage fraction above this is structurally impossible and means the MultiPV
# block being read describes a different position than the legal mask.
MAX_COVERAGE = 1.0 + 1e-9
# Policy mass on legal moves the WIDE label never scored. Measured at 1.1e-4 on
# the wide era; a large value means the same misalignment.
MAX_UNSCORED_MASS = 0.05


def move_score(cp: int, mate: int) -> float | None:
    """Per-move ordering score, mirroring ``finalize._sf_move_score`` exactly."""
    if mate != 0:
        return mate_to_effective_cp(int(mate))
    if cp == SF_CP_SENTINEL:
        return None
    return float(cp)


def tail_value(r_k: float, alpha: float) -> float:
    """``r_tail(alpha)`` — the one-parameter family spanning both live rules."""
    return r_k + (alpha / 2.0) * (1.0 - r_k)


def gradient(p: np.ndarray, r: np.ndarray) -> np.ndarray:
    """``dL/dz`` for ``L = p·r`` with ``p = softmax(z)`` over the legal set."""
    return p * (r - float(p @ r))


class Row:
    """One analysed position: the wide truth, the surfaced set, and the prior."""

    __slots__ = ("hidden", "legal", "prior", "r_k", "regret", "surfaced")

    def __init__(
        self, legal: np.ndarray, regret: dict[int, float],
        surfaced: list[int], hidden: list[int], r_k: float,
    ) -> None:
        self.legal = legal
        self.regret = regret
        self.surfaced = surfaced
        self.hidden = hidden
        self.r_k = r_k
        self.prior: np.ndarray | None = None


def collect(
    shards: list[Path], scan: dict[str, Any], k: int,
) -> tuple[list[Row], list[np.ndarray]]:
    """Join each row to its PARENT's MultiPV block and hide everything below k."""
    rows: list[Row] = []
    planes: list[np.ndarray] = []
    for shard in shards:
        try:
            arrs, _ = load_shard_arrays(shard)
        except (OSError, ValueError, KeyError) as exc:
            record_skipped_shard(scan, shard, exc)
            continue
        raw = np.asarray(arrs["sf_multipv_raw"])
        has_raw = np.asarray(arrs["has_sf_multipv_raw"]).astype(bool)
        legal_mask = np.asarray(arrs["legal_mask"]).astype(bool)
        pol = np.asarray(arrs["policy_target"]).astype(np.float64)
        x = np.asarray(arrs["x"])
        gid = np.asarray(arrs["game_id"]).astype(np.int64)
        ply = np.asarray(arrs["ply_index"]).astype(np.int64)
        n_rows_shard = int(raw.shape[0])
        # ⚑⚑ PRODUCTION ONLY BUILDS THIS LABEL ON SELFPLAY SLOTS.
        # `finalize.py:867` gates `want_sf_p0_regret` on `is_selfplay_slot`, so a
        # curriculum row NEVER receives the midpoint imputation. Screening on
        # unfiltered rows prices a tail that a large minority of the population
        # never carries -- and the two eras differ in exactly this composition
        # (wide 0.72 selfplay vs live MPV6 0.91), which is the dimension the
        # "the historical alpha transfers" claim compares across.
        has_sp = np.asarray(arrs.get("has_is_selfplay", np.zeros(n_rows_shard))).astype(bool)
        is_sp = np.asarray(arrs.get("is_selfplay", np.zeros(n_rows_shard))).astype(bool)
        selfplay = has_sp & is_sp
        scan["rows_scanned"] += n_rows_shard
        scan["rows_selfplay"] += int(selfplay.sum())
        index = {(int(gid[i]), int(ply[i])): i for i in range(raw.shape[0])}
        for i in range(raw.shape[0]):
            # ⚑ P0 SHIFT. Row i's own block is the position AFTER row i's move.
            if not selfplay[i]:
                scan["skipped_not_selfplay"] += 1
                continue
            parent = index.get((int(gid[i]), int(ply[i]) - P0_PARENT_PLY_OFFSET))
            if parent is None or not has_raw[parent]:
                continue
            scored: list[tuple[int, float]] = []
            for r in raw[parent].tolist():
                mi = int(r[0])
                if mi < 0:
                    continue
                sc = move_score(int(r[1]), int(r[2]))
                if sc is not None:
                    scored.append((mi, sc))
            if len(scored) <= k:
                continue
            mask = legal_mask[i]
            n_legal = int(mask.sum())
            if n_legal <= k:
                continue

            coverage = len(scored) / n_legal
            if coverage > MAX_COVERAGE:
                raise AssertionError(
                    f"coverage {coverage:.4f} > 1 in {shard.name} row {i}: the MultiPV "
                    "block does not describe the same position as the legal mask. The "
                    "usual cause is reading the row's OWN sf_multipv_raw instead of its "
                    "parent's (the P0 shift).",
                )
            scan["coverage_sum"] += coverage
            scan["coverage_n"] += 1

            scored.sort(key=lambda t: -t[1])
            best = scored[0][1]
            regret = {
                mi: min(max(best - sc, 0.0), SF_OWN_REGRET_CAP_CP) / SF_OWN_REGRET_CAP_CP
                for mi, sc in scored
            }
            surfaced = [mi for mi, _ in scored[:k]]
            hidden = [mi for mi, _ in scored[k:]]
            if not all(mask[mi] for mi in surfaced):
                scan["surfaced_not_legal"] += 1
                continue
            legal_idx = np.flatnonzero(mask)
            p = pol[i][legal_idx]
            if p.sum() <= 0:
                continue
            p = p / p.sum()
            unscored = float(p[[j for j, m in enumerate(legal_idx) if int(m) not in regret]].sum())
            scan["unscored_mass_sum"] += unscored
            rows.append(Row(legal_idx, regret, surfaced, hidden, max(regret[m] for m in surfaced)))
            rows[-1].prior = p
            planes.append(x[i])
    return rows, planes


def attach_prior(rows: list[Row], planes: list[np.ndarray], checkpoint: str, batch: int) -> str:
    """Replace the stored search target with the NET'S OWN legal-masked prior."""
    import torch

    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        # Live training owns this GPU; cap rather than compete for it.
        torch.cuda.set_per_process_memory_fraction(0.10)
    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            xb = torch.from_numpy(np.stack(planes[start:start + batch])).float().to(device)
            out = model(xb)
            logits = out["policy"] if "policy" in out else out["policy_own"]
            probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            for row, pr in zip(chunk, probs):
                sub = pr[row.legal]
                total = float(sub.sum())
                row.prior = sub / total if total > 0 else None
    return device


N_BOOT = 1000


def bootstrap_alphas(
    per_row: np.ndarray, *, n_boot: int = N_BOOT, seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Percentile CIs for both alphas by resampling ROWS.

    ⚑ A point estimate with no interval is what let the earlier version of this
    work report a tail price whose bounds varied only the outside population's
    mean. Both alphas are ratios of row-additive sums, so the resample is exact:
    columns are (true_tail, mass*r_k, mass*(1-r_k), grad_num, grad_den).

    The interval is over ROWS, which is the sampling unit. It does not cover
    Stockfish's own noise at a fixed position — two runs of the same search on
    the same row are not independent draws here.
    """
    if per_row.size == 0:
        return {"alpha_value": (float("nan"), float("nan")),
                "alpha_grad": (float("nan"), float("nan"))}
    rng = np.random.default_rng(seed)
    n = per_row.shape[0]
    vals: list[float] = []
    grads: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        c = per_row[idx].sum(axis=0)
        if c[2] > 0:
            vals.append(2.0 * (c[0] - c[1]) / c[2])
        if c[4] > 0:
            grads.append(c[3] / c[4])

    def pct(xs: list[float]) -> tuple[float, float]:
        if len(xs) < n_boot // 4:
            return (float("nan"), float("nan"))
        return (float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5)))

    return {"alpha_value": pct(vals), "alpha_grad": pct(grads)}


def analyse(rows: list[Row], alpha_grid: tuple[str, ...]) -> dict[str, Any]:
    """Fit both alphas and score every censor against the reference gradient.

    ⚑ ONE POPULATION, DETERMINED BEFORE ANY FITTING. An earlier version selected
    rows in the fitting loop and then dropped `‖g_true‖ == 0` rows in the metric
    loop, so `alpha_grad` was the least-squares optimum of a statistic OTHER than
    the pooled L2 printed beside it: a flat row contributes `row_num = 0` with
    `row_den > 0`, dragging the fit toward 0 while contributing nothing to the
    metric. Measured on a 2-row case, fitted 0.3637 against a true argmin of
    0.7340 — 33% worse on the very number it claimed to minimise. Both passes now
    iterate exactly `usable`.

    ⚑ NO FABRICATED TRUTH IN THE REFERENCE. A legal move the wide label never
    scored at ANY rank has no observed regret. Filling it with `r_k` (the old
    behaviour) fed pure "alpha = 0" evidence into the gradient fit from moves
    nobody measured. The support is now restricted to scored moves and the prior
    renormalised over it; the dropped mass is reported, not assumed negligible.
    """
    usable: list[Row] = []
    prep: list[dict[str, Any]] = []
    dropped_mass: list[float] = []
    for row in rows:
        p_full = row.prior
        if p_full is None:
            continue
        pos = {int(m): j for j, m in enumerate(row.legal)}
        hid = [pos[m] for m in row.hidden if m in pos]
        if not hid:
            continue
        # Restrict to the moves the wide label actually scored. `hidden` is
        # already scored-by-construction; `surfaced` likewise.
        scored = np.array([int(m) in row.regret for m in row.legal])
        if not scored.any():
            continue
        dropped_mass.append(float(p_full[~scored].sum()))
        p = p_full[scored]
        total = float(p.sum())
        if total <= 0:
            continue
        p = p / total
        legal_s = row.legal[scored]
        r_true = np.array([row.regret[int(m)] for m in legal_s])
        known = np.array([int(m) in set(row.surfaced) for m in legal_s])
        if known.all():
            continue
        mass = float(p[~known].sum())
        if mass <= 0:
            continue
        g_true = gradient(p, r_true)
        norm = float(np.linalg.norm(g_true))
        if norm <= 0:
            # A flat reference gradient cannot discriminate between censors, and
            # including it biases the fit while contributing nothing measurable.
            continue
        usable.append(row)
        prep.append({"p": p, "r_true": r_true, "known": known, "g_true": g_true,
                     "norm": norm, "mass": mass, "r_k": row.r_k})

    s_t = s_m = s_m_rk = s_m_1mrk = 0.0
    num_g = den_g = 0.0
    # ⚑ BOTH alphas are ratios of SUMS OVER ROWS, so keeping each row's five
    # contributions makes the bootstrap exact rather than approximate: resampling
    # rows and re-summing reproduces the estimator on the resampled population
    # without re-running it.
    per_row: list[tuple[float, float, float, float, float]] = []
    for q in prep:
        p, r_true, known, r_k = q["p"], q["r_true"], q["known"], q["r_k"]
        mass = q["mass"]
        true_tail = float(p[~known] @ r_true[~known])
        s_t += true_tail
        s_m += mass
        s_m_rk += mass * r_k
        s_m_1mrk += mass * (1.0 - r_k)
        # g(alpha) = g0 + alpha * d, so the gradient-optimal alpha is closed form.
        r0 = np.where(known, r_true, r_k)
        u = np.where(known, 0.0, (1.0 - r_k) / 2.0)
        d = p * (u - float(p @ u))
        row_num = float(d @ (q["g_true"] - gradient(p, r0)))
        row_den = float(d @ d)
        num_g += row_num
        den_g += row_den
        per_row.append((true_tail, mass * r_k, mass * (1.0 - r_k), row_num, row_den))

    alpha_value = 2.0 * (s_t - s_m_rk) / s_m_1mrk
    alpha_grad = num_g / den_g
    alphas = {"alpha_value": alpha_value, "alpha_grad": float(np.clip(alpha_grad, 0.0, 2.0))}
    ci = bootstrap_alphas(np.asarray(per_row, dtype=np.float64))

    cos: dict[str, list[float]] = {n: [] for n in alpha_grid}
    rel: dict[str, list[float]] = {n: [] for n in alpha_grid}
    num2: dict[str, float] = dict.fromkeys(alpha_grid, 0.0)
    press: dict[str, list[float]] = {n: [] for n in (*alpha_grid, "true")}
    den2 = 0.0
    for q in prep:
        p, r_true, known = q["p"], q["r_true"], q["known"]
        g_true, norm, r_k = q["g_true"], q["norm"], q["r_k"]
        den2 += norm ** 2
        press["true"].append(float(g_true[~known].sum()))
        for name in alpha_grid:
            fill = {
                "r_k censor": r_k,
                "midpoint (live)": (r_k + 1.0) / 2.0,
                "alpha_value": tail_value(r_k, alphas["alpha_value"]),
                "alpha_grad": tail_value(r_k, alphas["alpha_grad"]),
            }[name]
            g_hat = gradient(p, np.where(known, r_true, fill))
            cos[name].append(float(g_hat @ g_true / (np.linalg.norm(g_hat) * norm + 1e-12)))
            delta = float(np.linalg.norm(g_hat - g_true))
            rel[name].append(delta / norm)
            num2[name] += delta ** 2
            press[name].append(float(g_hat[~known].sum()))

    cap = SF_OWN_REGRET_CAP_CP
    return {
        "n_rows": len(usable),
        "dropped_unscored_mass_mean": float(np.mean(dropped_mass)) if dropped_mass else 0.0,
        "alpha_value_ci": ci["alpha_value"],
        "alpha_grad_ci": ci["alpha_grad"],
        "alpha_value": alpha_value,
        "alpha_grad_raw": alpha_grad,
        "alpha_grad": alphas["alpha_grad"],
        "true_cp": cap * s_t / s_m,
        "r_k_cp": cap * s_m_rk / s_m,
        "midpoint_cp": cap * (s_m_rk + 0.5 * s_m_1mrk) / s_m,
        "true_pressure": float(np.mean(press["true"])),
        "variants": {
            name: {
                "cos": float(np.mean(cos[name])),
                "rel_l2_pooled": float((num2[name] / den2) ** 0.5),
                "rel_l2_median": float(np.median(rel[name])),
                "tail_pressure": float(np.mean(press[name])),
            }
            for name in alpha_grid
        },
    }


def check_invariants(scan: dict[str, Any], n_rows: int) -> tuple[float, float]:
    """Runtime guards on values that CANNOT occur under correct P0 alignment.

    The per-row coverage guard already fired inside ``collect``; this is the
    aggregate half. Both exist because the first version of this screen produced a
    confident wrong answer that no unit test would have caught — the numbers were
    plausible, and only their impossibility gave them away.
    """
    coverage = scan["coverage_sum"] / max(scan["coverage_n"], 1)
    unscored = scan["unscored_mass_sum"] / max(n_rows, 1)
    if unscored > MAX_UNSCORED_MASS:
        raise AssertionError(
            f"unscored policy mass {unscored:.4f} > {MAX_UNSCORED_MASS}: the MultiPV block "
            "and the policy are almost certainly describing different positions (P0 shift).",
        )
    return coverage, unscored


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--replay-dir", type=Path, required=True, action="append",
                    help="wide-MultiPV shard directory; repeatable")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="weight by THIS net's prior (the training-relevant answer). "
                         "Without it the stored policy_target is used, which answers "
                         "a different question.")
    ap.add_argument("--k", type=int, default=6, help="truncate to this many PVs")
    ap.add_argument("--max-shards", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    shards: list[Path] = []
    for d in args.replay_dir:
        shards.extend(select_shards(Path(d), args.max_shards))
    scan: dict[str, Any] = {
        "shard_names": [p.name for p in shards], "rows_scanned": 0,
        "rows_selfplay": 0, "skipped_not_selfplay": 0,
        "coverage_sum": 0.0, "coverage_n": 0, "unscored_mass_sum": 0.0,
        "surfaced_not_legal": 0, "skipped_shards": [], "skipped_shards_omitted": 0,
    }
    rows, planes = collect(shards, scan, args.k)
    if not rows:
        raise SystemExit("no rows with a parent MultiPV block wider than --k")

    coverage, unscored = check_invariants(scan, len(rows))
    print(f"shards {len(shards)}   rows scanned {scan['rows_scanned']}"
          f"   selfplay {scan['rows_selfplay']}   analysed {len(rows)}")
    print(f"  ⚑ non-selfplay rows dropped: {scan['skipped_not_selfplay']}"
          f"   (production never builds sf_p0_regret on them -- finalize.py:867)")
    print(f"  wide label covers {coverage:.4f} of legal moves")
    print(f"  policy mass on moves the wide label never scored: {unscored:.6f}")
    # ⚑ THE COVERAGE ASSERT IS STRUCTURALLY INERT WHEN THE BLOCK IS NARROWER THAN
    # THE LEGAL COUNT (e.g. MultiPV 6 vs 30 legal): a misaligned run then silently
    # DISCARDS most rows instead of tripping, printing a healthy coverage from the
    # handful that survive. `surfaced_not_legal` is the detector that still has
    # power there, so it must be visible rather than buried in the JSON.
    snl = int(scan["surfaced_not_legal"])
    frac_snl = snl / max(scan["rows_selfplay"], 1)
    print(f"  ⚑ surfaced-move-not-legal rows: {snl} ({frac_snl:.4f} of selfplay rows)"
          f"   -- the P0 detector that still fires at narrow MultiPV")
    if frac_snl > 0.02:
        raise AssertionError(
            f"surfaced_not_legal {frac_snl:.4f} > 0.02: SF's own top move is not legal in "
            "the row it was joined to. That is a position mismatch (the P0 shift), not noise.",
        )
    kept = len(rows) / max(scan["rows_selfplay"], 1)
    print(f"  rows kept / selfplay rows: {kept:.4f}")
    if kept < 0.10:
        raise AssertionError(
            f"kept only {kept:.4f} of selfplay rows: a misaligned join discards most rows "
            "while leaving the coverage and unscored-mass diagnostics looking healthy.",
        )

    weighting = "policy_target (search target — NOT what the loss weights by)"
    if args.checkpoint:
        device = attach_prior(rows, planes, args.checkpoint, args.batch)
        weighting = f"net prior from {args.checkpoint} on {device}"
    print(f"  weighting: {weighting}")

    grid = ("r_k censor", "midpoint (live)", "alpha_value", "alpha_grad")
    res = analyse(rows, grid)
    res["scan"] = scan
    res["weighting"] = weighting

    print(f"\nPOLICY-WEIGHTED hidden-tail regret  (n={res['n_rows']})")
    print(f"  TRUE (withheld wide-SF)      {res['true_cp']:6.1f} cp")
    print(f"  r_k censor      (alpha=0)    {res['r_k_cp']:6.1f} cp"
          f"   bias {res['r_k_cp'] - res['true_cp']:+7.1f}")
    print(f"  midpoint (live) (alpha=1)    {res['midpoint_cp']:6.1f} cp"
          f"   bias {res['midpoint_cp'] - res['true_cp']:+7.1f}"
          f"   = {res['midpoint_cp'] / res['true_cp']:.2f}x")
    vlo, vhi = res["alpha_value_ci"]
    glo, ghi = res["alpha_grad_ci"]
    print(f"  alpha_value {res['alpha_value']:.4f} [{vlo:.4f}, {vhi:.4f}]"
          f"   alpha_grad {res['alpha_grad_raw']:.4f} [{glo:.4f}, {ghi:.4f}]"
          f"  (clipped {res['alpha_grad']:.4f})")
    print(f"  95% CIs over {N_BOOT} row-level bootstrap draws; they do NOT cover "
          "Stockfish's own\n  per-position noise, only the sampling of rows.")
    print("\nGRADIENT FIDELITY  dL/dz = p*(r - E_p[r])")
    print("  ⚑ pooled rel L2 is the reportable one: rows with ||g_true|| ~ 0 make the")
    print("    per-row MEAN explode (one row read 4557). Median shown alongside.")
    print(f"  {'variant':17s} {'cos':>8s} {'relL2 pool':>11s} {'relL2 med':>10s}"
          f" {'tail press':>11s} {'vs TRUE':>8s}")
    print(f"  {'TRUE':17s} {1.0:8.4f} {0.0:11.4f} {0.0:10.4f}"
          f" {res['true_pressure']:11.6f} {1.0:7.2f}x")
    for name in grid:
        v = res["variants"][name]
        print(f"  {name:17s} {v['cos']:8.4f} {v['rel_l2_pooled']:11.4f}"
              f" {v['rel_l2_median']:10.4f} {v['tail_pressure']:11.6f}"
              f" {v['tail_pressure'] / res['true_pressure']:7.2f}x")
    print("\n⚑ Hidden SF RANK does not exist at live MultiPV 6. The rank structure this")
    print("  screen exposes explains why a constant tail fails; it is NOT a usable target.")
    print("  Implementable successors are prior-aware/residual tails, an adaptive alpha on")
    print("  observable root features, or buying the information with targeted SF queries.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(res, indent=2, sort_keys=True))
        print(f"\nbanked -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
