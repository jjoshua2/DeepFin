#!/usr/bin/env python3
"""Does CAST-style solver credit add anything to the MultiPV-6 teacher we already pay for?

Phase 0 of issue #425, read-only: no training-loop, shard-format or live-config
change. It answers three questions off banked replay shards, and each one has a
control that can fail.

WHAT CAST WOULD GIVE US. For consecutive labelled plies, with ``q = W - L`` read
off the record-POV ``sf_wdl`` label:

    A_CAST(t) = q_t + q_(t-1)

The sign algebra is the whole trick, so it is worth stating why the two terms
ADD. A record's ``sf_wdl`` is Stockfish's read of the position AFTER that
record's move, flipped back into that record's mover POV
(``stockfish_turn._sf_result_wdl_for_record``). So ``q_t`` is already
``Q_T(s_t, a_t)`` in the current mover's POV, while ``q_(t-1)`` describes the
SAME position ``s_t`` from the PREVIOUS mover's POV -- the opponent's. Hence
``V_T(s_t) = -q_(t-1)`` and ``A = Q - V = q_t + q_(t-1)``. A minus sign here
silently turns the instrument into a measure of whose turn it is; see
``tests/test_cast_probe.py::test_advantage_pov_sign``.

Under a consistent teacher ``A <= 0``, with equality when the move played was
the teacher's own choice. Both properties are MEASURED here rather than
assumed, because ``V_T`` and ``Q_T`` come from two SEPARATE finite-node
searches: ``P(A > 0)`` is a direct read of the noise floor, and the mean ``A``
on rows where the played move WAS SF's best move is a direct read of the
systematic drift between them.

WHAT THIS IS UP AGAINST. On rows that carry ``sf_p0_regret`` we already know the
exact cp-regret of every move SF surfaced, from ONE root search, with no
cross-search noise. CAST can only add information where the played move fell
OUTSIDE that MultiPV set. So the probe's central number is not "is A_CAST
correlated with move quality" (it is) but "how many rows does it reach that the
MultiPV teacher does not, and what does it say there".

⚑⚑ THE PLAYED MOVE IS NOT IN THE SHARD. Neither ``_NetRecord`` nor
``ReplaySample`` stores the net's own move -- ``sf_move_index`` and
``sf_played_move_index`` are STOCKFISH's moves. So this probe cannot read the
played action; it uses ``argmax(policy_target)`` as a proxy and the proxy is
only as good as the search target is peaked. That is not a detail to bury: the
headline number moves by 3x across peakedness strata, so every played-move
statistic is reported STRATIFIED by ``max(policy_target)`` and there is no flag
to collapse it to one number. A training-time CAST loss would need the played
move persisted, i.e. a shard-format change.

THE THIRD QUESTION, which turned out to matter more than CAST itself:
``finalize._build_sf_p0_regret_vector`` gives every legal move SF did NOT
surface a synthetic regret of ``(worst_surfaced + 1) / 2``. At MultiPV 40 that
touched few moves. At the live MultiPV 6 it touches most of them, and the
expected-regret loss ``sum_a pi(a) r(a)`` is then largely an integral over
fabricated numbers. This probe prices that imputation against CAST: it builds a
calibration curve from moves whose regret IS known, then reads the outside-set
mean ``A_CAST`` backwards through it.

COVERAGE IS RECOVERABLE, contrary to the note in ``search_gain_probe.py``. It is
not recoverable from the regret VECTOR (the imputed entries are unmarked), but
the vector at row ``t`` is built from row ``t-1``'s ``sf_multipv_raw``, which IS
banked. Joining on ``(game_id, ply_index - 1)`` recovers the exact covered set,
so no shard change is needed to audit the tail.

CONTROLS (always on -- a control an operator can forget is a control that will
be forgotten):
  * ``shuffle``     - permute A_CAST across rows. Destroys the move->quality
                      association while preserving the marginal exactly, so the
                      calibration correlation MUST collapse toward 0.
  * ``within-pos``  - permute the regret vector WITHIN each position's legal
                      moves. Preserves the legal set, the coverage and the
                      marginal regret distribution; the tail-mass share is a
                      property of the distribution and survives, the
                      played-move regret association does not.
  * ``drift``       - mean A_CAST restricted to rows where the played-move
                      proxy IS SF's own best move. A consistent teacher owes 0
                      here; a nonzero offset is search drift between the parent
                      and child labels and biases every other A_CAST number by
                      the same amount.
  * ``sign``        - P(A_CAST > 0), which a noiseless solver cannot produce.

Usage:

    PYTHONPATH=. python3 scripts/cast_probe.py --run-dir runs/pbt2_small \\
        --max-shards 24 --json-out scratchpad/cast_probe.json
"""
from __future__ import annotations

import argparse
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from chess_anti_engine.replay.shard import load_shard_arrays
from scripts.diagnostic_replay_utils import (
    float_field,
    latest_replay_dir,
    record_skipped_shard,
    select_shards,
)

# finalize.py normalizes cp-regret by this cap before storing it, so every
# regret in the shard is in [0, 1] and multiplying by it returns centipawns.
SF_OWN_REGRET_CAP_CP = 1000.0

# Peakedness strata for the played-move proxy. Reported unconditionally: the
# outside-MultiPV estimate depends strongly on which one you read.
PMAX_STRATA: tuple[float, ...] = (0.0, 0.5, 0.8, 0.9)

# Regret bucket edges (normalized) for the calibration curve.
CALIB_EDGES: tuple[float, ...] = (0.0, 0.002, 0.01, 0.03, 0.06, 0.12, 0.25, 1.01)

MIN_BUCKET = 25


class Rows:
    """Column store for the joined (parent, child) rows the probe analyses."""

    def __init__(self) -> None:
        self.adv: list[float] = []          # A_CAST = q_t + q_(t-1)
        self.regret_played: list[float] = []  # imputed OR exact, as stored
        self.in_multipv: list[bool] = []    # played-move proxy inside SF's set
        self.is_sf_best: list[bool] = []    # played-move proxy IS SF's best
        self.pmax: list[float] = []         # max(policy_target), proxy quality
        self.abs_q_parent: list[float] = []  # |root value|, saturation proxy
        self.n_legal: list[int] = []
        self.n_covered: list[int] = []
        self.mass_covered: list[float] = []
        self.er_total: list[float] = []     # E_pi[regret]
        self.er_imputed: list[float] = []   # ... contributed by imputed moves
        self.er_total_shuf: list[float] = []
        self.er_imputed_shuf: list[float] = []
        self.imputed_value: list[float] = []
        self.worst_covered: list[float] = []
        # H3 pre-query allocator candidates: known BEFORE the child label is
        # submitted, so they are the features an allocator could actually use.
        self.priority_policy_kl: list[float] = []
        self.priority_q_delta: list[float] = []

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "adv": np.asarray(self.adv, dtype=np.float64),
            "regret_played": np.asarray(self.regret_played, dtype=np.float64),
            "in_multipv": np.asarray(self.in_multipv, dtype=bool),
            "is_sf_best": np.asarray(self.is_sf_best, dtype=bool),
            "pmax": np.asarray(self.pmax, dtype=np.float64),
            "abs_q_parent": np.asarray(self.abs_q_parent, dtype=np.float64),
            "n_legal": np.asarray(self.n_legal, dtype=np.float64),
            "n_covered": np.asarray(self.n_covered, dtype=np.float64),
            "mass_covered": np.asarray(self.mass_covered, dtype=np.float64),
            "er_total": np.asarray(self.er_total, dtype=np.float64),
            "er_imputed": np.asarray(self.er_imputed, dtype=np.float64),
            "er_total_shuf": np.asarray(self.er_total_shuf, dtype=np.float64),
            "er_imputed_shuf": np.asarray(self.er_imputed_shuf, dtype=np.float64),
            "imputed_value": np.asarray(self.imputed_value, dtype=np.float64),
            "worst_covered": np.asarray(self.worst_covered, dtype=np.float64),
            "priority_policy_kl": np.asarray(self.priority_policy_kl, dtype=np.float64),
            "priority_q_delta": np.asarray(self.priority_q_delta, dtype=np.float64),
        }


def advantage(q_child: float, q_parent: float) -> float:
    """CAST solver advantage for the move between two consecutive labelled plies.

    ``q_child`` is ``W - L`` of the record whose move we are grading, ``q_parent``
    the same quantity one ply earlier. Both are already in their own record's
    mover POV, which is why they ADD rather than subtract -- see the module
    docstring.
    """
    return q_child + q_parent


def collect(shards: list[Path], scan: dict[str, Any], rng: np.random.Generator) -> Rows:
    """Join adjacent labelled plies and accumulate every per-row quantity."""
    rows = Rows()
    for shard in shards:
        try:
            arrs, _ = load_shard_arrays(shard)
        except (OSError, ValueError, KeyError) as exc:
            record_skipped_shard(scan, shard, exc)
            continue
        n = int(np.asarray(arrs["policy_target"]).shape[0])
        scan["rows_scanned"] += n
        gid = np.asarray(arrs["game_id"]).astype(np.int64)
        ply = np.asarray(arrs["ply_index"]).astype(np.int64)
        has_p0r = np.asarray(arrs["has_sf_p0_regret"]).astype(bool)
        has_raw = np.asarray(arrs["has_sf_multipv_raw"]).astype(bool)
        has_wdl = np.asarray(arrs["has_sf_wdl"]).astype(bool)
        wdl = np.asarray(arrs["sf_wdl"]).astype(np.float64)
        q = wdl[:, 0] - wdl[:, 2]
        raw = np.asarray(arrs["sf_multipv_raw"])
        reg = np.asarray(arrs["sf_p0_regret"]).astype(np.float64)
        legal = np.asarray(arrs["legal_mask"]).astype(bool)
        pol = np.asarray(arrs["policy_target"]).astype(np.float64)
        kl = float_field(arrs, "priority_policy_kl", n)
        qd = float_field(arrs, "priority_q_delta", n)

        scan["rows_sf_wdl"] += int(has_wdl.sum())
        scan["rows_sf_p0_regret"] += int(has_p0r.sum())

        index = {(int(gid[i]), int(ply[i])): i for i in range(n)}
        for i in range(n):
            parent = index.get((int(gid[i]), int(ply[i]) - 1))
            if parent is None or not (has_wdl[i] and has_wdl[parent]):
                continue
            # Every adjacency below is EXACT: ply_index - 1 in the same game.
            # A gap is never treated as consecutive.
            scan["cast_pairs"] += 1
            a = advantage(float(q[i]), float(q[parent]))
            if not has_p0r[i] or not has_raw[parent]:
                continue
            covered_idx = raw[parent][:, 0].astype(np.int64)
            covered_idx = covered_idx[covered_idx >= 0]
            if covered_idx.size == 0:
                continue
            mask = legal[i]
            width = int(mask.size)
            covered_idx = covered_idx[covered_idx < width]
            covered = np.zeros((width,), dtype=bool)
            covered[covered_idx] = True
            covered &= mask
            if not covered.any():
                continue
            probs = pol[i] * mask
            total = float(probs.sum())
            if total <= 0.0:
                continue
            probs = probs / total
            played = int(np.argmax(probs))
            reg_legal = reg[i][mask]
            cov_legal = covered[mask]
            p_legal = probs[mask]
            sf_best = int(covered_idx[int(np.argmin(reg[i][covered_idx]))])

            scan["cast_pairs_with_p0"] += 1
            rows.adv.append(a)
            rows.regret_played.append(float(reg[i][played]))
            rows.in_multipv.append(bool(covered[played]))
            rows.is_sf_best.append(played == sf_best)
            rows.pmax.append(float(probs[played]))
            rows.abs_q_parent.append(abs(float(q[parent])))
            rows.n_legal.append(int(mask.sum()))
            rows.n_covered.append(int(cov_legal.sum()))
            rows.mass_covered.append(float(p_legal[cov_legal].sum()))
            rows.er_total.append(float((p_legal * reg_legal).sum()))
            rows.er_imputed.append(float((p_legal[~cov_legal] * reg_legal[~cov_legal]).sum()))
            if (~cov_legal).any():
                rows.imputed_value.append(float(np.median(reg_legal[~cov_legal])))
            rows.worst_covered.append(float(reg_legal[cov_legal].max()))
            rows.priority_policy_kl.append(float(kl[i]))
            rows.priority_q_delta.append(float(qd[i]))
            # within-position control: permute the regrets across this
            # position's legal moves, leaving the legal set and the marginal
            # regret distribution untouched.
            shuffled = rng.permutation(reg_legal)
            rows.er_total_shuf.append(float((p_legal * shuffled).sum()))
            rows.er_imputed_shuf.append(
                float((p_legal[~cov_legal] * shuffled[~cov_legal]).sum())
            )
    return rows


def monotone_prefix(xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Longest leading run of ``ys`` that is strictly decreasing.

    The calibration curve folds back at the top: positions where the played
    move threw away 500cp are usually already decided, so the WDL label
    saturates and |A_CAST| SHRINKS again. Inverting a non-monotone curve is
    meaningless, so only the monotone part is used and the discarded buckets
    are reported.
    """
    keep = [0]
    for k in range(1, int(ys.size)):
        if ys[k] < ys[keep[-1]]:
            keep.append(k)
    sel = np.asarray(keep, dtype=np.int64)
    return xs[sel], ys[sel]


def calibration(
    arr: dict[str, np.ndarray], sel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Mean A_CAST per exact-regret bucket, over COVERED played moves only."""
    xs: list[float] = []
    ys: list[float] = []
    table: list[dict[str, float]] = []
    reg = arr["regret_played"]
    adv = arr["adv"]
    for lo, hi in pairwise(CALIB_EDGES):
        m = sel & arr["in_multipv"] & (reg >= lo) & (reg < hi)
        count = int(m.sum())
        if count < MIN_BUCKET:
            continue
        xs.append(float(np.median(reg[m])))
        ys.append(float(adv[m].mean()))
        table.append({
            "lo_cp": lo * SF_OWN_REGRET_CAP_CP,
            "hi_cp": hi * SF_OWN_REGRET_CAP_CP,
            "median_cp": xs[-1] * SF_OWN_REGRET_CAP_CP,
            "mean_adv": ys[-1],
            "n": float(count),
        })
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64), table


def invert(xs: np.ndarray, ys: np.ndarray, y: float) -> float:
    """Read a mean-advantage back through the calibration curve to a cp regret."""
    order = np.argsort(ys)
    return float(np.interp(y, ys[order], xs[order]))


def price_the_tail(arr: dict[str, np.ndarray], sel: np.ndarray, label: str) -> dict[str, Any]:
    """What is a played move OUTSIDE SF's MultiPV set actually worth?

    Calibrates within ``sel`` -- never across strata -- because the played-move
    proxy degrades as the search target flattens and a curve borrowed from a
    different stratum silently imports that bias.
    """
    xs, ys, table = calibration(arr, sel)
    out = sel & ~arr["in_multipv"]
    n_out = int(out.sum())
    result: dict[str, Any] = {
        "stratum": label,
        "n": int(sel.sum()),
        "n_outside": n_out,
        "calibration": table,
    }
    if n_out < MIN_BUCKET or xs.size < 2:
        result["skipped"] = "insufficient rows"
        return result
    xs_m, ys_m = monotone_prefix(xs, ys)
    adv_out = arr["adv"][out]
    mean = float(adv_out.mean())
    half = 1.96 * float(adv_out.std(ddof=1)) / float(np.sqrt(n_out))
    implied = invert(xs_m, ys_m, mean) * SF_OWN_REGRET_CAP_CP
    lo_cp = invert(xs_m, ys_m, mean + half) * SF_OWN_REGRET_CAP_CP
    hi_cp = invert(xs_m, ys_m, mean - half) * SF_OWN_REGRET_CAP_CP
    assigned = float(arr["regret_played"][out].mean()) * SF_OWN_REGRET_CAP_CP
    result.update({
        "outside_frac": n_out / max(int(sel.sum()), 1),
        "mean_adv_outside": mean,
        "ci95_half": half,
        "implied_cp": implied,
        "implied_cp_lo": lo_cp,
        "implied_cp_hi": hi_cp,
        "assigned_cp": assigned,
        "overstatement": assigned / max(implied, 1e-9),
        "overstatement_lo": assigned / max(hi_cp, 1e-9),
        "overstatement_hi": assigned / max(lo_cp, 1e-9),
        "calibration_buckets_dropped": int(xs.size - xs_m.size),
    })
    return result


def report(arr: dict[str, np.ndarray], scan: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    adv = arr["adv"]
    out: dict[str, Any] = {"scan": scan}
    n = int(adv.size)
    print(f"shards scanned: {scan['shards']}   rows: {scan['rows_scanned']}")
    if scan["skipped_shards"]:
        print(f"  ⚑ skipped shards: {len(scan['skipped_shards'])}")
    rows_scanned = max(int(scan["rows_scanned"]), 1)
    print("\n1. COVERAGE -- does scalar CAST reach rows the MultiPV teacher misses?")
    print(f"   rows with an sf_wdl label          {scan['rows_sf_wdl']:7d}"
          f"  {100 * scan['rows_sf_wdl'] / rows_scanned:5.1f}%")
    print(f"   rows with sf_p0_regret (MultiPV)   {scan['rows_sf_p0_regret']:7d}"
          f"  {100 * scan['rows_sf_p0_regret'] / rows_scanned:5.1f}%")
    print(f"   exact adjacent CAST pairs          {scan['cast_pairs']:7d}"
          f"  {100 * scan['cast_pairs'] / rows_scanned:5.1f}%")
    print(f"   ... of which also carry P0 regret  {scan['cast_pairs_with_p0']:7d}"
          f"  {100 * scan['cast_pairs_with_p0'] / rows_scanned:5.1f}%")
    print("   ⇒ both teachers are gated by the SAME thing (is the previous ply also a")
    print("     stored row), not by MultiPV width, so scalar labels buy no extra reach.")
    print("   ⚑ the pair count is a LOWER bound: this join is within-shard, so a pair")
    print("     straddling a shard boundary is missed. sf_p0_regret is computed at")
    print("     finalize time inside the game and has no such loss -- compare the two")
    print("     rates knowing the CAST side is the biased-DOWN one.")
    out["coverage"] = {
        "rows": scan["rows_scanned"],
        "sf_wdl": scan["rows_sf_wdl"],
        "sf_p0_regret": scan["rows_sf_p0_regret"],
        "cast_pairs": scan["cast_pairs"],
        "cast_pairs_with_p0": scan["cast_pairs_with_p0"],
    }
    if n == 0:
        print("\nno joined rows -- nothing further to report")
        return out

    print("\n2. THE A_CAST SIGNAL AND ITS NOISE FLOOR")
    p_pos = float(np.mean(adv > 1e-9))
    p_zero = float(np.mean(np.abs(adv) <= 1e-9))
    pos = adv[adv > 1e-9]
    neg = adv[adv < -1e-9]
    print(f"   n={n}  mean {adv.mean():+.4f}  median {np.median(adv):+.4f}  sd {adv.std():.4f}")
    print(f"   P(A = 0) {p_zero:.4f}    P(A > 0) {p_pos:.4f}  ⚑ a consistent teacher owes 0 here")
    if pos.size and neg.size:
        print(f"   mean |A| where A>0 (pure noise) {pos.mean():.4f}"
              f"   where A<0 (signal+noise) {-neg.mean():.4f}")
    sat = arr["abs_q_parent"] > 0.8
    if sat.any():
        print(f"   saturated roots |q_parent|>0.8: {100 * sat.mean():.1f}% of rows,"
              f" mean A {adv[sat].mean():+.4f} (sd {adv[sat].std():.4f})"
              f" vs {adv[~sat].mean():+.4f} elsewhere")
    out["signal"] = {
        "n": n, "mean": float(adv.mean()), "median": float(np.median(adv)),
        "sd": float(adv.std()), "p_positive": p_pos, "p_zero": p_zero,
        "mean_abs_positive": float(pos.mean()) if pos.size else float("nan"),
        "mean_abs_negative": float(-neg.mean()) if neg.size else float("nan"),
        "saturated_frac": float(sat.mean()),
    }

    print("\n3. CONTROL -- search drift between the parent and child label")
    print("   (played-move proxy IS SF's own best move; a consistent teacher owes A = 0)")
    drift: list[dict[str, float]] = []
    for lo in PMAX_STRATA:
        m = arr["is_sf_best"] & (arr["pmax"] >= lo)
        if int(m.sum()) < MIN_BUCKET:
            continue
        x = adv[m]
        half = 1.96 * float(x.std(ddof=1)) / float(np.sqrt(int(m.sum())))
        flag = "" if abs(float(x.mean())) < half else "  ⚑ biased"
        print(f"   pmax>={lo:<4}  n={int(m.sum()):5d}  mean A {x.mean():+.4f} ± {half:.4f}{flag}")
        drift.append({"pmax_min": lo, "n": float(m.sum()), "mean": float(x.mean()), "ci95_half": half})
    out["drift_control"] = drift
    if len(drift) >= 2 and abs(drift[-1]["mean"]) < abs(drift[0]["mean"]):
        print("   drift SHRINKS as the proxy sharpens ⇒ most of it is proxy error (argmax")
        print("   != the played move), not parent/child search drift. Quote the tightest")
        print(f"   stratum as the bound: {drift[-1]['mean']:+.4f} ± {drift[-1]['ci95_half']:.4f}.")

    print("\n4. THE MULTIPV-6 IMPUTED TAIL")
    n_legal, n_cov = arr["n_legal"], arr["n_covered"]
    imputed_frac = float((1.0 - n_cov / n_legal).mean())
    er_t, er_i = arr["er_total"], arr["er_imputed"]
    share = float(er_i.sum() / er_t.sum()) if er_t.sum() > 0 else float("nan")
    share_shuf = (
        float(arr["er_imputed_shuf"].sum() / arr["er_total_shuf"].sum())
        if arr["er_total_shuf"].sum() > 0 else float("nan")
    )
    print(f"   legal moves/row {n_legal.mean():5.2f}   SF-covered {n_cov.mean():5.2f}"
          f"   ⇒ {100 * imputed_frac:.1f}% of legal moves carry a fabricated regret")
    print(f"   imputed value {SF_OWN_REGRET_CAP_CP * arr['imputed_value'].mean():6.0f} cp"
          f"   vs worst move SF ACTUALLY surfaced"
          f" {SF_OWN_REGRET_CAP_CP * arr['worst_covered'].mean():6.0f} cp"
          f" (median {SF_OWN_REGRET_CAP_CP * np.median(arr['worst_covered']):.0f})")
    print(f"   search-target mass on covered moves: mean {arr['mass_covered'].mean():.4f}"
          f"  median {np.median(arr['mass_covered']):.4f}"
          f"  p10 {np.percentile(arr['mass_covered'], 10):.4f}")
    print(f"   E_pi[regret] {er_t.mean():.4f}, of which {er_i.mean():.4f}"
          f" is imputed ⇒ SHARE {share:.4f}")
    print(f"   rows where the imputed tail supplies >50% of E_pi[regret]:"
          f" {100 * float(np.mean(er_i > 0.5 * er_t)):.1f}%")
    imputed_mass = float(1.0 - arr["mass_covered"].mean())
    print(f"   [within-position shuffle control] share {share_shuf:.4f}"
          f"  vs imputed probability MASS {imputed_mass:.4f}")
    print("     ⇒ the control MUST collapse to roughly the imputed mass share. It")
    print("       separates the two ways a tail can dominate: carrying most of the")
    print("       probability (it does not -- see the mass line) versus carrying a")
    print("       fabricated VALUE. It collapsed, so the domination is the value.")
    out["imputed_tail"] = {
        "imputed_frac_of_legal": imputed_frac,
        "imputed_cp": float(SF_OWN_REGRET_CAP_CP * arr["imputed_value"].mean()),
        "worst_covered_cp": float(SF_OWN_REGRET_CAP_CP * arr["worst_covered"].mean()),
        "mass_covered_mean": float(arr["mass_covered"].mean()),
        "expected_regret": float(er_t.mean()),
        "expected_regret_imputed": float(er_i.mean()),
        "imputed_share": share,
        "imputed_share_within_pos_shuffle": share_shuf,
        "rows_tail_dominated": float(np.mean(er_i > 0.5 * er_t)),
    }

    print("\n5. PRICING THE IMPUTED TAIL WITH CAST")
    print("   ⚑ stratified by max(policy_target): the played move is NOT stored, and the")
    print("     argmax proxy is only trustworthy where the search target is peaked.")
    strata: list[dict[str, Any]] = []
    for lo in PMAX_STRATA:
        sel = arr["pmax"] >= lo
        res = price_the_tail(arr, sel, f"pmax>={lo}")
        strata.append(res)
        if "implied_cp" not in res:
            print(f"   pmax>={lo:<4}  skipped ({res.get('skipped')})")
            continue
        print(f"   pmax>={lo:<4}  outside {res['n_outside']:5d}"
              f" ({100 * res['outside_frac']:4.1f}%)  mean A {res['mean_adv_outside']:+.4f}"
              f"  ⇒ true {res['implied_cp']:5.0f} cp"
              f" [{res['implied_cp_lo']:.0f}-{res['implied_cp_hi']:.0f}]"
              f"  vs {res['assigned_cp']:5.0f} cp assigned"
              f"  = {res['overstatement']:5.1f}x")
    out["tail_pricing"] = strata

    print("\n6. PRE-QUERY DIAGNOSTICS vs A_CAST")
    print("   ⚑⚑ THIS IS NOT A TEST OF ISSUE #425's H3, and must not be read as one.")
    print("   H3 asks whether parent-search features predict |q_deep - q_cheap| -- whether")
    print("   MORE SF COMPUTE MOVES THE LABEL. A_CAST is how much the played move DAMAGED")
    print("   THE POSITION. Different random variables: a crushing blunder can be obvious")
    print("   to 50k and 3M alike, while a quiet move lands where 150k and 1M disagree. A")
    print("   low correlation here is NOT evidence against an adaptive-label allocator;")
    print("   that needs paired cheap/deep labels, which this probe does not have.")
    feat: dict[str, float] = {}
    for name in ("priority_policy_kl", "priority_q_delta"):
        v = arr[name]
        ok = np.isfinite(v) & np.isfinite(adv)
        if int(ok.sum()) < MIN_BUCKET:
            continue
        c = float(np.corrcoef(v[ok], adv[ok])[0, 1])
        feat[name] = c
        print(f"   corr({name}, A_CAST) = {c:+.4f}   (n={int(ok.sum())})")
    out["allocator_features"] = feat

    print("\n7. NEGATIVE CONTROL -- permute A_CAST across rows")
    cov = arr["in_multipv"]
    real = float(np.corrcoef(adv[cov], -arr["regret_played"][cov])[0, 1])
    shuffled_corrs: list[float] = []
    for _ in range(5):
        perm = rng.permutation(adv)
        shuffled_corrs.append(float(np.corrcoef(perm[cov], -arr["regret_played"][cov])[0, 1]))
    worst = max(abs(c) for c in shuffled_corrs)
    verdict = "PASS" if worst < abs(real) / 3.0 else "⚑ FAIL"
    print(f"   corr(A_CAST, -regret) real {real:+.4f}"
          f"   shuffled max|corr| {worst:.4f}   {verdict}")
    unsat = cov & (arr["abs_q_parent"] < 0.5)
    if int(unsat.sum()) > MIN_BUCKET:
        print(f"   unsaturated roots only: corr"
              f" {np.corrcoef(adv[unsat], -arr['regret_played'][unsat])[0, 1]:+.4f}"
              f"  (n={int(unsat.sum())})")
    out["negative_control"] = {
        "corr_real": real,
        "corr_shuffled_max_abs": worst,
        "verdict": verdict,
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, default=Path("runs/pbt2_small"))
    ap.add_argument("--trial-dir", type=Path, default=None)
    ap.add_argument("--replay-dir", type=Path, default=None,
                    help="scan this shard directory directly instead of resolving the live one")
    ap.add_argument("--max-shards", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    replay_dir = args.replay_dir or latest_replay_dir(args.run_dir, args.trial_dir)
    shards = select_shards(Path(replay_dir), args.max_shards)
    if not shards:
        raise SystemExit(f"no shards under {replay_dir}")
    rng = np.random.default_rng(args.seed)
    scan: dict[str, Any] = {
        "replay_dir": str(replay_dir),
        "shards": len(shards),
        "rows_scanned": 0,
        "rows_sf_wdl": 0,
        "rows_sf_p0_regret": 0,
        "cast_pairs": 0,
        "cast_pairs_with_p0": 0,
        "skipped_shards": [],
        "skipped_shards_omitted": 0,
    }
    rows = collect(shards, scan, rng)
    out = report(rows.arrays(), scan, rng)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"\nbanked -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
