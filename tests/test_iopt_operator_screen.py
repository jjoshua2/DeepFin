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


def _unreachable_trace() -> dict[str, object]:
    trace = _trace(
        prior=(0.06293350762907712, 0.2858852391641012, 0.07807522190427352,
               0.4371186552426282, 0.13598737605992),
        visits=(36, 39, 1, 11, 28),
        q=(0.017319845837524595, 0.03903011926885706, 0.04717259008727228,
           0.0034942981378395313, 0.003374214105294302),
        root_q=-0.032733272425827076, scale=10.0,
        cp=(100.0, 20.0, 0.0, -20.0, -40.0),
    )
    trace['legal_uci'] = ['e2e4', 'd2d4', 'g1f3', 'c2c4', 'b1c3']
    trace['legal_actions'] = [1, 2, 3, 4, 5]
    trace['observed_policy'] = [0.0005621490177532028, 0.36299557666163285,
                                0.6362257536014934, 0.00016621101160653237,
                                0.00005030970751406481]
    return trace


def test_default_proxy_kl_unreachable_fixture() -> None:
    root = reconstruct(_unreachable_trace(), beta=0.1, tau2=0.08,
                       variance_scale=0.04, unvisited_multiplier=4)
    assert not root.matched_kl_reachable
    assert kl_from_prior(root.policies['current'], root.prior) == pytest.approx(1.41705, abs=1e-5)
    assert kl_from_prior(root.policies['iopt_matched_kl'], root.prior) == pytest.approx(1.25216, abs=1e-5)
    assert root.policies['iopt_matched_kl'] == pytest.approx([0, 1, 0, 0, 0])


@pytest.mark.parametrize('mixed', [False, True])
def test_screen_excludes_unreachable_only_from_matched_comparison(tmp_path, mixed) -> None:
    traces = [_unreachable_trace()]
    if mixed:
        traces.append(_trace(cp=(35.0, 0.0, -30.0)))
    rows: list[dict[str, object]] = [
        {'_meta': True, 'operator_trace': True, 'operator_trace_format': 1}
    ]
    rows.extend({'sims': 128, 'fen': f'root-{i}', 'operator_trace': trace}
                for i, trace in enumerate(traces))
    source = tmp_path / 'roots.jsonl'
    source.write_text('\n'.join(json.dumps(row) for row in rows))
    bucket = run_screen(source)['by_simulations']['128']
    assert bucket['qualified_roots'] == len(traces)
    assert bucket['arms']['current']['n_top1_covered'] == len(traces)
    assert bucket['arms']['iopt_proxy']['n_top1_covered'] == len(traces)
    matched = bucket['arms']['iopt_matched_kl']
    baseline = bucket['arms']['current_matched_kl_subset']
    assert matched['n_top1_covered'] == baseline['n_top1_covered'] == int(mixed)
    pairs = bucket['paired_gain_positive_is_better']['iopt_matched_kl']
    assert pairs['n_excluded_KL_unreachable'] == 1
    assert pairs['n_eligible_rows'] == int(mixed)
    if mixed:
        assert pairs['top1_regret_gain_cp']['n_roots'] == 1
        assert baseline['top1_regret_cp_mean_covered'] == pytest.approx(35.0)
    else:
        assert pairs['top1_regret_gain_cp'] is None
        assert pairs['FULL_reference_expected_regret_gain_cp'] is None
        assert matched['mean_kl_prior'] is None
        assert baseline['top1_regret_cp_mean_covered'] is None


def test_mate_regret_cap_preserves_unknown_labels_and_raw_cp() -> None:
    cp = np.array([100000.0, 0.0, np.nan])
    p = np.array([0.1, 0.8, 0.1])
    metrics = policy_metrics(p, p, cp)
    assert metrics['top1_regret_cp'] == 1000.0
    assert metrics['expected_regret_cp'] is None
    assert metrics['unknown_mass'] == pytest.approx(0.1)
    assert cp[0] == 100000.0
    assert np.isnan(cp[2])


def test_full_mate_regret_caps_each_action_before_expectation() -> None:
    cp = np.array([100000.0, 0.0, -100000.0])
    p = np.array([0.5, 0.3, 0.2])
    metrics = policy_metrics(p, p, cp)
    assert metrics['expected_regret_cp'] == pytest.approx(500.0)
    assert cp.tolist() == [100000.0, 0.0, -100000.0]


@pytest.mark.parametrize(("scale", "offset"), [(1.0, 0.0), (1e-200, 0.0), (1.0, 1e12)])
def test_kl_matching_preserves_score_scale_and_offset(scale, offset) -> None:
    prior = np.array([0.6, 0.3, 0.1])
    reference = np.array([0.05, 0.15, 0.8])
    matched, reachable = match_kl(prior, np.array([0.0, 1.0, 2.0]) * scale + offset, reference)
    assert reachable
    assert np.all(np.isfinite(matched))
    assert matched.sum() == pytest.approx(1.0)
    assert kl_from_prior(matched, prior) == pytest.approx(kl_from_prior(reference, prior), abs=1e-8)
    stable, _ = match_kl(prior, np.array([0.0, 1.0, 2.0]), reference)
    assert matched == pytest.approx(stable, abs=1e-10)


def test_kl_tied_maximum_limit_retains_prior_ratios() -> None:
    prior = np.array([0.2, 0.3, 0.5])
    limit = np.array([0.4, 0.6, 0.0])
    matched, reachable = match_kl(prior, np.array([1.0, 1.0, 0.0]), limit)
    assert reachable
    assert matched == pytest.approx(limit)
    unreachable, reachable = match_kl(prior, np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert not reachable
    assert unreachable == pytest.approx(limit)


def test_positive_kl_cannot_be_matched_by_constant_scores() -> None:
    prior = np.array([0.5, 0.3, 0.2])
    matched, reachable = match_kl(prior, np.ones(3), np.array([1.0, 0.0, 0.0]))
    assert not reachable
    assert matched == pytest.approx(prior)


def test_kl_matching_handles_finite_opposite_extreme_scores() -> None:
    prior = np.array([0.6, 0.3, 0.1])
    reference = np.array([0.9, 0.05, 0.05])
    matched, reachable = match_kl(prior, np.array([1e308, 0.0, -1e308]), reference)
    expected, _ = match_kl(prior, np.array([1.0, 0.0, -1.0]), reference)
    assert reachable
    assert np.all(np.isfinite(matched))
    assert matched == pytest.approx(expected)


@pytest.mark.parametrize(("key", "value"), [
    ("visits", [50.5, 2.0, 0.0]),
    ("visits", [2**80, 2, 0]),
    ("visits", [True, True, False]),
    ("visits", [True, 2, 0]),
    ("visits", [50, -2, 0]),
    ("prior", [[0.5, 0.3, 0.2]]),
    ("observed_policy", [[0.5, 0.3, 0.2]]),
    ("legal_uci", ["e2e4"]),
    ("legal_uci", ["e2e4", "e2e4", "g1f3"]),
    ("legal_actions", [1, 2]),
    ("legal_actions", [1, 1, 3]),
    ("root_q", True),
])
def test_malformed_trace_vectors_and_identities_are_refused(key, value) -> None:
    trace = _trace()
    trace[key] = value
    with pytest.raises(ValueError, match=r"visits|vectors|identities|numeric"):
        reconstruct(trace, beta=0.1, tau2=0.08, variance_scale=0.04, unvisited_multiplier=4)


def test_bad_move_identity_excludes_one_root_without_aborting_bank(tmp_path) -> None:
    bad = _trace()
    bad["legal_uci"] = ["e2e4"]
    rows = [{"_meta": True, "operator_trace": True, "operator_trace_format": 1},
            {"sims": 128, "fen": "bad", "operator_trace": bad},
            {"sims": 128, "fen": "good", "operator_trace": _trace()}]
    source = tmp_path / "roots.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows))
    report = run_screen(source)
    assert report["counts"]["rows_qualified"] == 1
    assert report["counts"]["skipped"] == {"invalid_or_unsupported_trace": 1}
    json.dumps(report, allow_nan=False)


def test_nonfinite_calculated_policy_cannot_pass_nan_parity_gate() -> None:
    trace = _trace(q=(0.0, 2.0, None))
    trace["root_q_scale"] = 1e308
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError, match="nonfinite"):
        reconstruct(trace, beta=0.1, tau2=0.08, variance_scale=0.04, unvisited_multiplier=4)


def test_nonfinite_proxy_policy_is_excluded_before_strict_json(tmp_path) -> None:
    rows = [{"_meta": True, "operator_trace": True, "operator_trace_format": 1},
            {"sims": 128, "fen": "root", "operator_trace": _trace()}]
    source = tmp_path / "roots.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows))
    with np.errstate(over="ignore", invalid="ignore"):
        report = run_screen(source, beta=1e-320)
    assert report["counts"]["rows_qualified"] == 0
    assert report["counts"]["skipped"] == {"invalid_or_unsupported_trace": 1}
    json.dumps(report, allow_nan=False)
