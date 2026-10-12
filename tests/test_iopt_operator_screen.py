"""CPU-only I-opt frozen-root operator screen: guard against fake improvements."""

from __future__ import annotations

import json

import numpy as np
import pytest

from chess_anti_engine.eval.iopt_operator import (
    completed_q_values,
    kl_from_prior,
    match_kl,
    policy_metrics,
    reconstruct,
)
from scripts.iopt_operator_screen import run_screen


def _trace(
    *,
    prior: tuple[float, ...] = (0.5, 0.3, 0.2),
    visits: tuple[int, ...] = (50, 2, 0),
    q: tuple[float | None, ...] = (0.51, 0.52, None),
    root_q: float = 0.48,
    scale: float = 2.0,
    cp: tuple[float | None, ...] = (35.0, 0.0, None),
) -> dict[str, object]:
    p = np.array(prior, dtype=np.float64)
    v = np.array(visits, dtype=np.int64)
    qarr = np.array([float(x) if x is not None else np.nan for x in q])
    completed = completed_q_values(p, v, qarr, root_q)
    logits = np.log(p) + scale * (
        completed - completed.min()
    ) / max(float(np.ptp(completed)), 1e-8)
    target = np.exp(logits - logits.max())
    target /= target.sum()
    return {
        "prior": list(prior),
        "visits": list(visits),
        "visited_q": list(q),
        "reference_cp": list(cp),
        "root_q": root_q,
        "root_q_scale": scale,
        "observed_policy": target.tolist(),
        "legal_uci": ["e2e4", "d2d4", "g1f3"],
        "legal_actions": [1, 2, 3],
        "label_kind": "native_deep_sf_multipv_partial",
    }


def test_unvisited_q_is_mixed_not_zero_or_unvisited_network_q() -> None:
    prior = np.array([0.5, 0.3, 0.2])
    visits = np.array([50, 2, 0])
    completed = completed_q_values(
        prior, visits, np.array([0.51, 0.52, np.nan]), 0.48
    )
    mean = (0.5 * 0.51 + 0.3 * 0.52) / 0.8
    expected = (0.48 + 52 * mean) / 53
    assert completed[2] == pytest.approx(expected)
    assert completed[0] == pytest.approx(0.51)


def test_baseline_parity_and_uncertainty_operator_not_identical() -> None:
    trace = _trace()
    result = reconstruct(
        trace, beta=0.1, tau2=0.08,
        variance_scale=0.04, unvisited_multiplier=4,
    )
    assert result.parity_max_abs < 1e-12
    assert result.visits.tolist() == [50, 2, 0]
    assert result.reliability[0] > result.reliability[1]
    assert result.reliability[1] > result.reliability[2]
    for p in result.policies.values():
        assert p.sum() == pytest.approx(1)
        assert np.all(p >= 0)
    assert not np.allclose(result.policies["current"], result.policies["unscaled"])
    assert not np.allclose(result.policies["current"], result.policies["iopt_proxy"])


def test_match_kl_is_a_shape_control_and_unscaled_only_differs_by_temperature() -> None:
    root = reconstruct(
        _trace(), beta=0.1, tau2=0.08,
        variance_scale=0.04, unvisited_multiplier=4,
    )
    assert root.matched_kl_reachable
    assert kl_from_prior(root.policies["current"], root.prior) == pytest.approx(
        kl_from_prior(root.policies["iopt_matched_kl"], root.prior),
        abs=1e-8,
    )
    # The Q ordering is unchanged; differences in policy argmax are due to
    # a single global multiplier. Matching that per root is a tautology.
    rematched, reachable = match_kl(
        root.prior, root.completed_q, root.policies["current"]
    )
    assert reachable
    assert rematched == pytest.approx(root.policies["current"], abs=1e-10)


def test_reconstruction_rejects_wrong_q_or_native_policy() -> None:
    trace = _trace()
    trace["observed_policy"] = [0.5, 0.3, 0.2]
    with pytest.raises(ValueError, match="native root-policy reconstruction mismatch"):
        reconstruct(
            trace, beta=0.1, tau2=0.08,
            variance_scale=0.04, unvisited_multiplier=4,
        )
    trace = _trace()
    trace["visited_q"] = [None, 0.52, None]
    with pytest.raises(ValueError, match="presence disagrees"):
        reconstruct(
            trace, beta=0.1, tau2=0.08,
            variance_scale=0.04, unvisited_multiplier=4,
        )


def test_no_fake_multi_pv_tail_regret_or_full_expected_regret() -> None:
    root = reconstruct(
        _trace(), beta=0.1, tau2=0.08,
        variance_scale=0.04, unvisited_multiplier=4,
    )
    for name, p in root.policies.items():
        result = policy_metrics(p, root.prior, root.reference_cp)
        assert result["expected_regret_cp"] is None, name
        assert result["unknown_mass"] == pytest.approx(p[2])


def test_full_coverage_enables_expected_regret() -> None:
    root = reconstruct(
        _trace(cp=(35.0, 0.0, -30.0)),
        beta=0.1, tau2=0.08,
        variance_scale=0.04, unvisited_multiplier=4,
    )
    p = root.policies["current"]
    metric = policy_metrics(p, root.prior, root.reference_cp)
    assert metric["expected_regret_cp"] == pytest.approx(
        p[1] * 35.0 + p[2] * 65.0
    )


def test_complete_offline_jsonl_readout(tmp_path) -> None:
    source = tmp_path / "roots.jsonl"
    rows = [
        {
            "_meta": True,
            "checkpoint": "frozen.pt",
            "shape": {"label": "test"},
            "operator_trace": True,
            "operator_trace_format": 1,
        },
        {
            "sims": 128,
            "fen": "test-position-1",
            "phase": 1,
            "operator_trace": _trace(cp=(35.0, 0.0, -30.0)),
        },
        {
            "sims": 128,
            "fen": "test-position-2",
            "phase": 1,
            "operator_trace": _trace(cp=(35.0, 0.0, None)),
        },
    ]
    source.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    report = run_screen(source)
    assert report["counts"]["rows_qualified"] == 2
    bucket = report["by_simulations"]["128"]
    assert bucket["qualified_roots"] == 2
    assert bucket["arms"]["current"]["n_FULL_legal_reference"] == 1
    assert bucket["failure_flags"]["missing_ANY_reference_move"] == 1
    assert bucket["paired_gain_positive_is_better"]["unscaled"]["top1_regret_gain_cp"]


def test_invalid_header_refused(tmp_path) -> None:
    source = tmp_path / "invalid.jsonl"
    source.write_text(json.dumps({"_meta": True, "operator_trace": False}) + "\n")
    with pytest.raises(ValueError, match="not captured"):
        run_screen(source)
