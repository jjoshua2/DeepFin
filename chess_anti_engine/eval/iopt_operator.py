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
    if np.ptp(advantage) <= 1e-12:
        return prior.copy(), False
    lo, hi = 0.0, 1.0
    for _ in range(40):
        if kl_from_prior(_softmax(logp, hi * advantage), prior) >= target:
            break
        hi *= 2.0
    else:
        return _softmax(logp, hi * advantage), False
    for _ in range(54):
        mid = (lo + hi) * 0.5
        if kl_from_prior(_softmax(logp, mid * advantage), prior) < target:
            lo = mid
        else:
            hi = mid
    return _softmax(logp, hi * advantage), True


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
    if not (
        beta > 0.0 and tau2 > 0.0 and variance_scale > 0.0
        and unvisited_multiplier >= 1.0
    ):
        raise ValueError("beta/tau2/variance_scale must be positive; unvisited_multiplier >= 1")
    prior = np.asarray(trace["prior"], dtype=np.float64)
    visits = np.asarray(trace["visits"], dtype=np.int64)
    q_raw = trace["visited_q"]
    ref_raw = trace["reference_cp"]
    observed = np.asarray(trace["observed_policy"], dtype=np.float64)
    if not isinstance(q_raw, list) or not isinstance(ref_raw, list):
        raise ValueError("visited_q and reference_cp must be lists")
    n = prior.size
    if n < 2 or any(v.size != n for v in (visits, observed)) or len(q_raw) != n or len(ref_raw) != n:
        raise ValueError("illegal trace vector lengths")
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
    root_q = float(trace["root_q"])
    q_scale = float(trace["root_q_scale"])
    if not np.isfinite(root_q) or not np.isfinite(q_scale) or q_scale < 0:
        raise ValueError("invalid root value or q scale")
    cp = np.array(
        [float(c) if c is not None else np.nan for c in ref_raw], dtype=np.float64
    )
    if not np.all(np.isfinite(cp[np.isfinite(cp)])):
        raise ValueError("invalid reference")
    completed = completed_q_values(prior, visits, visited_q, root_q)
    spread = float(np.ptp(completed))
    logp = np.log(prior)

    # DeepFin's C and Python root implementation use min-max COMPLETED Q.
    baseline = _softmax(
        logp, q_scale * (completed - completed.min()) / max(spread, 1e-8)
    )
    parity = float(np.max(np.abs(baseline - observed)))
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

    return RootCounterfactual(
        prior=prior, visits=visits, completed_q=completed, reference_cp=cp,
        policies={
            "current": baseline,
            "unscaled": unscaled,
            "iopt_proxy": iopt,
            "iopt_matched_kl": iopt_matched,
        },
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
    return {
        "top1": j,
        "top1_regret_cp": (
            float(ref_best - cp[j]) if known[j] and ref_best is not None else None
        ),
        "expected_regret_cp": (
            float(np.dot(p, ref_best - cp))
            if np.all(known) and ref_best is not None else None
        ),
        "unknown_mass": float(p[~known].sum()),
        "kl_prior": kl_from_prior(p, prior),
        "entropy": float(-np.sum(p[p > 0] * np.log(p[p > 0]))),
    }
