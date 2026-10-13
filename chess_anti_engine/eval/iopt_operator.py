"""Frozen-root Gumbel policy counterfactuals; no training or search-path effects.

The I-opt arm uses a deliberately simple independent-Q-error variance *proxy*.
Unvisited mixed-value estimates actually share correlated errors. This is a
mechanism screen, not a faithful/calibrated implementation of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RootCounterfactual:
    prior: np.ndarray
    visits: np.ndarray
    completed_q: np.ndarray
    reference_cp: np.ndarray
    policies: dict[str, np.ndarray]
    q_range: float
    parity_max_abs: float
    matched_kl_reachable: bool
    reliability: np.ndarray


def _softmax(log_prior: np.ndarray, adjustment: np.ndarray) -> np.ndarray:
    z = np.asarray(log_prior + adjustment, dtype=np.float64)
    z -= z.max()
    exp = np.exp(z)
    return exp / exp.sum()


def kl_from_prior(policy: np.ndarray, prior: np.ndarray) -> float:
    used = policy > 0.0
    return float(
        np.sum(
            policy[used]
            * (np.log(policy[used]) - np.log(prior[used]))
        )
    )


def completed_q_values(
    prior: np.ndarray,
    visits: np.ndarray,
    visited_q: np.ndarray,
    root_q: float,
) -> np.ndarray:
    """Mirror mctx mixed-value completion (not a new FPU rule)."""
    visited = visits > 0
    if np.any(visited):
        p_sum = float(prior[visited].sum())
        weighted_q = (
            float(np.dot(prior[visited], visited_q[visited]) / p_sum)
            if p_sum > 0.0 else root_q
        )
    else:
        weighted_q = root_q
    total = float(visits.sum())
    mixed = (root_q + total * weighted_q) / (1.0 + total)
    return np.where(visited, visited_q, mixed)


def match_kl(
    prior: np.ndarray,
    advantage: np.ndarray,
    reference_policy: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Match baseline KL per root, isolating *action-wise shape*, if reachable.

    This is deliberately NOT used for the unscaled-Q arm: unscaled vs
    min-max Q is only one positive scalar multiplier of the same vector,
    so matching KL per root would make the two policies identical.
    """
    logp = np.log(prior)
    target = kl_from_prior(reference_policy, prior)
    if target <= 1e-12:
        return prior.copy(), True

    # Center BEFORE multiplication, then remove the arbitrary score scale.
    # Otherwise large offsets lose the prior and tiny spreads exhaust a fixed
    # doubling budget even though the same policy family is reachable.
    values = np.asarray(advantage, dtype=np.float64)
    with np.errstate(over="ignore"):
        centered = values - values.max()
    if not np.all(np.isfinite(centered)):
        # Finite opposite-sign scores can overflow during subtraction.
        values = values / np.max(np.abs(values))
        centered = values - values.max()
    spread = float(np.max(np.abs(centered)))
    if spread == 0.0:
        return prior.copy(), False
    score = centered / spread
    winners = score == 0.0
    winner_mass = float(prior[winners].sum())
    limit = np.where(winners, prior / winner_mass, 0.0)
    limit_kl = kl_from_prior(limit, prior)
    if target > limit_kl + 1e-10:
        return limit, False
    if abs(target - limit_kl) <= 1e-10:
        return limit, True

    def tilted(multiplier: float) -> np.ndarray:
        # Nonpositive centered scores may become -inf at the float limit;
        # their zero probability is well-defined and the maxima stay finite.
        with np.errstate(over="ignore"):
            return _softmax(logp, multiplier * score)

    lo, hi = 0.0, 1.0
    while kl_from_prior(tilted(hi), prior) < target:
        lo = hi
        if hi > np.finfo(np.float64).max / 2.0:
            return tilted(hi), False
        hi *= 2.0
    for _ in range(64):
        mid = lo + (hi - lo) * 0.5
        if kl_from_prior(tilted(mid), prior) < target:
            lo = mid
        else:
            hi = mid
    matched = tilted(hi)
    return matched, abs(kl_from_prior(matched, prior) - target) <= 1e-8



def reconstruct(
    trace: dict[str, object],
    *,
    beta: float,
    tau2: float,
    variance_scale: float,
    unvisited_multiplier: float,
    parity_tolerance: float = 3e-4,
) -> RootCounterfactual:
    """Reconstruct policies ONLY if native-policy parity passes.

    The JSON capture contains legal moves in ONE canonical order; unknown
    Stockfish labels and unvisited child-Q entries are null, never invented.
    """
    params = (beta, tau2, variance_scale, unvisited_multiplier, parity_tolerance)
    if not all(np.isfinite(v) for v in params) or not (
        beta > 0.0 and tau2 > 0.0 and variance_scale > 0.0
        and unvisited_multiplier >= 1.0 and parity_tolerance >= 0.0
    ):
        raise ValueError("operator parameters must be finite, positive and valid")
    prior = np.asarray(trace["prior"], dtype=np.float64)
    visit_values = trace["visits"]
    if not isinstance(visit_values, list) or not all(
        type(v) is int and 0 <= v <= np.iinfo(np.int64).max for v in visit_values
    ):
        raise ValueError("child visits must be nonnegative integer counts")
    visits_raw = np.asarray(visit_values)
    if visits_raw.ndim != 1 or not np.issubdtype(visits_raw.dtype, np.integer):
        raise ValueError("child visits must be a vector of integers")
    visits = visits_raw.astype(np.int64)
    q_raw = trace["visited_q"]
    ref_raw = trace["reference_cp"]
    observed = np.asarray(trace["observed_policy"], dtype=np.float64)
    if not isinstance(q_raw, list) or not isinstance(ref_raw, list):
        raise ValueError("visited_q and reference_cp must be lists")
    if prior.ndim != 1 or observed.ndim != 1:
        raise ValueError("policies must be one-dimensional vectors")
    n = prior.size
    if n < 2 or any(v.size != n for v in (visits, observed)) or len(q_raw) != n or len(ref_raw) != n:
        raise ValueError("illegal trace vector lengths")
    legal_uci = trace["legal_uci"]
    legal_actions = trace["legal_actions"]
    if not (
        isinstance(legal_uci, list) and len(legal_uci) == n
        and all(isinstance(move, str) and bool(move) for move in legal_uci)
        and len(set(legal_uci)) == n
        and isinstance(legal_actions, list) and len(legal_actions) == n
        and all(type(action) is int and action >= 0 for action in legal_actions)
        and len(set(legal_actions)) == n
    ):
        raise ValueError("legal move/action identities must align with trace vectors")
    if not (
        np.all(np.isfinite(prior)) and np.all(prior > 0)
        and np.isclose(prior.sum(), 1.0, atol=1e-5)
        and np.all(visits >= 0)
        and np.all(np.isfinite(observed))
        and np.all(observed >= 0)
        and np.isclose(observed.sum(), 1.0, atol=1e-5)
    ):
        raise ValueError("invalid normalized policy or visits")
    if any((q is None) != (int(v) == 0) for q, v in zip(q_raw, visits, strict=True)):
        raise ValueError("visited-Q presence disagrees with child visits")
    visited_q = np.array(
        [float(q) if q is not None else np.nan for q in q_raw],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(visited_q[visits > 0])):
        raise ValueError("nonfinite visited Q")
    root_raw = trace["root_q"]
    scale_raw = trace["root_q_scale"]
    if (
        not isinstance(root_raw, int | float) or isinstance(root_raw, bool)
        or not isinstance(scale_raw, int | float) or isinstance(scale_raw, bool)
    ):
        raise ValueError("root value and q scale must be numeric")
    root_q = float(root_raw)
    q_scale = float(scale_raw)
    if not np.isfinite(root_q) or not np.isfinite(q_scale) or q_scale < 0:
        raise ValueError("invalid root value or q scale")
    cp = np.array(
        [float(c) if c is not None else np.nan for c in ref_raw], dtype=np.float64
    )
    if any(c is not None and not np.isfinite(float(c)) for c in ref_raw):
        raise ValueError("nonfinite directly observed reference label")
    completed = completed_q_values(prior, visits, visited_q, root_q)
    spread = float(np.ptp(completed))
    if not np.all(np.isfinite(completed)) or not np.isfinite(spread):
        raise ValueError("nonfinite completed Q")
    logp = np.log(prior)

    # DeepFin's C and Python root implementation use min-max COMPLETED Q.
    baseline = _softmax(
        logp, q_scale * (completed - completed.min()) / max(spread, 1e-8)
    )
    parity = float(np.max(np.abs(baseline - observed)))
    if not np.isfinite(parity):
        raise ValueError("nonfinite root-policy reconstruction")
    if parity > parity_tolerance:
        raise ValueError(
            f"native root-policy reconstruction mismatch {parity:.5g} > {parity_tolerance}"
        )

    # Unscaled completed Q: same raw Q, same q_scale, but no min-max stretch.
    unscaled = _softmax(logp, q_scale * (completed - completed.min()))

    # Independent-action error approximation to Eq. (10); shared mixed-value
    # completion and search bias violate its independence premise. PROXY ONLY.
    variance = variance_scale / np.maximum(visits.astype(np.float64), 1.0)
    variance = np.where(visits > 0, variance, variance_scale * unvisited_multiplier)
    mean_variance = float(np.dot(prior**2, variance))
    variance_advantage = np.maximum(
        0.0, mean_variance + (1.0 - 2.0 * prior) * variance
    )
    reliability = tau2 / (tau2 + variance_advantage)
    advantage = completed - float(np.dot(prior, completed))
    iopt_score = reliability * advantage
    iopt = _softmax(logp, iopt_score / beta)
    iopt_matched, reachable = match_kl(prior, iopt_score, baseline)

    policies = {
        "current": baseline,
        "unscaled": unscaled,
        "iopt_proxy": iopt,
        "iopt_matched_kl": iopt_matched,
    }
    if not all(np.all(np.isfinite(p)) for p in policies.values()):
        raise ValueError("nonfinite counterfactual policy")

    return RootCounterfactual(
        prior=prior, visits=visits, completed_q=completed, reference_cp=cp,
        policies=policies,
        q_range=spread,
        parity_max_abs=parity,
        matched_kl_reachable=reachable,
        reliability=reliability,
    )


def policy_metrics(
    p: np.ndarray, prior: np.ndarray, cp: np.ndarray
) -> dict[str, float | int | None]:
    """Unknown reference moves stay unknown; never substitute MultiPV floors."""
    known = np.isfinite(cp)
    j = int(np.argmax(p))
    ref_best = float(np.max(cp[known])) if np.any(known) else None
    # Same per-action cap as audit.AUDIT_REGRET_CAP_CP; raw labels stay intact.
    regrets = np.clip(ref_best - cp, 0.0, 1000.0) if ref_best is not None else None
    return {
        "top1": j,
        "top1_regret_cp": (
            float(regrets[j]) if known[j] and regrets is not None else None
        ),
        "expected_regret_cp": (
            float(np.dot(p, regrets))
            if np.all(known) and regrets is not None else None
        ),
        "unknown_mass": float(p[~known].sum()),
        "kl_prior": kl_from_prior(p, prior),
        "entropy": float(-np.sum(p[p > 0] * np.log(p[p > 0]))),
    }
