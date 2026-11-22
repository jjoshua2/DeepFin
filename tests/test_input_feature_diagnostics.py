from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.input_feature_diagnostics import (
    FEATURE_GROUPS,
    compare_outputs,
    perturb_inputs,
    summarize_metrics,
)


def test_perturbations_preserve_history_and_do_not_mutate_input() -> None:
    x = np.arange(2 * 175 * 64, dtype=np.float32).reshape(2, 175, 8, 8)
    before = x.copy()
    identity = perturb_inputs(x, family="extras_all", mode="identity", row_ids=["a", "b"], seed=7)
    np.testing.assert_array_equal(identity, x)
    assert not np.shares_memory(identity, x)
    zero = perturb_inputs(x, family="attacks_typed", mode="zero", row_ids=["a", "b"], seed=7)
    np.testing.assert_array_equal(zero[:, :148], x[:, :148])
    np.testing.assert_array_equal(zero[:, 160:], x[:, 160:])
    assert np.count_nonzero(zero[:, 148:160]) == 0
    for family in FEATURE_GROUPS:
        shifted = perturb_inputs(x, family=family, mode="spatial_shuffle", row_ids=["a", "b"], seed=7)
        np.testing.assert_array_equal(shifted[:, :112], x[:, :112])
    np.testing.assert_array_equal(x, before)


def test_spatial_shuffle_preserves_histograms_channel_alignment_and_row_identity() -> None:
    x = np.arange(3 * 175 * 64, dtype=np.float32).reshape(3, 175, 8, 8)
    ids = ["source-a/game-1/ply-1", "source-a/game-1/ply-2", "source-b/game-1/ply-1"]
    result = perturb_inputs(x, family="threats_all", mode="spatial_shuffle", row_ids=ids, seed=17)
    assert not np.array_equal(result[:, 146:], x[:, 146:])
    np.testing.assert_array_equal(result[:, :146], x[:, :146])
    np.testing.assert_array_equal(np.sort(result[:, 146:].reshape(3, 29, 64), axis=2),
                                  x[:, 146:].reshape(3, 29, 64))
    # Every channel is an offset of the same spatial ramp. The shared
    # permutation must retain those cross-channel differences at every square.
    np.testing.assert_array_equal(result[:, 147] - result[:, 146], np.full((3, 8, 8), 64))
    order = [2, 0, 1]
    reordered = perturb_inputs(x[order], family="threats_all", mode="spatial_shuffle",
                               row_ids=[ids[i] for i in order], seed=17)
    np.testing.assert_array_equal(reordered, result[order])
    for index in range(3):
        singleton = perturb_inputs(x[index:index + 1], family="attacks_typed", mode="spatial_shuffle",
                                    row_ids=[ids[index]], seed=17)
        np.testing.assert_array_equal(singleton[0, 148:160], result[index, 148:160])


@pytest.mark.parametrize("fault", ["shape", "nan", "family", "mode", "identity", "seed"])
def test_invalid_perturbation_contract_is_refused(fault: str) -> None:
    x = np.zeros((1, 175, 8, 8), dtype=np.float32)
    if fault == "shape":
        x = x[:, :146]
    elif fault == "nan":
        x[0, 0, 0, 0] = np.nan
    message = {"shape": "x must", "nan": "x must", "family": "feature family",
               "mode": "perturbation mode", "identity": "row_ids", "seed": "seed must"}[fault]
    with pytest.raises(ValueError, match=message):
        perturb_inputs(x, family="unknown" if fault == "family" else "extras_all",
                       mode="unknown" if fault == "mode" else "zero",
                       row_ids=[" " if fault == "identity" else "row"],
                       seed=-1 if fault == "seed" else 0)


def _output_pair() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    legal = np.zeros((1, 1858), dtype=np.bool_)
    legal[:, :2] = True
    ref_policy = np.full((1, 1858), 1000.0)
    cand_policy = np.full((1, 1858), -1000.0)
    ref_policy[0, :2] = np.log([0.75, 0.25])
    cand_policy[0, :2] = np.log([0.25, 0.75])
    reference = {"policy": ref_policy, "wdl": np.log([[0.6, 0.3, 0.1]])}
    candidate = {"policy": cand_policy, "wdl": np.log([[0.2, 0.5, 0.3]])}
    return reference, candidate, legal


def test_metrics_match_known_distributions_and_ignore_illegal_logits() -> None:
    reference, candidate, legal = _output_pair()
    result = compare_outputs(reference, candidate, legal)
    assert result["policy_kl"][0] == pytest.approx(0.5 * math.log(3))
    assert result["top1_changed"].tolist() == [True]
    assert result["policy_total_variation"][0] == pytest.approx(0.5)
    assert result["signed_q_change"][0] == pytest.approx(-0.6)
    assert result["abs_q_change"][0] == pytest.approx(0.6)
    assert result["wdl_total_variation"][0] == pytest.approx(0.4)
    for value in compare_outputs(reference, reference, legal).values():
        np.testing.assert_array_equal(value, np.zeros(1))
    # Adding constants to each row does not alter the softmax distribution.
    shifted = {key: value + 1000 for key, value in candidate.items()}
    for name, value in compare_outputs(reference, shifted, legal).items():
        np.testing.assert_allclose(value, result[name], atol=1e-12)


def test_forced_move_and_extreme_finite_logits_do_not_make_nan_metrics() -> None:
    reference, candidate, legal = _output_pair()
    reference["policy"][0, :2] = [1e4, -1e4]
    candidate["policy"][0, :2] = [-1e4, 1e4]
    result = compare_outputs(reference, candidate, legal)
    assert result["policy_kl"][0] == pytest.approx(2e4)
    legal[0, 1] = False
    result = compare_outputs(reference, candidate, legal)
    assert result["policy_kl"][0] == 0
    assert not result["top1_changed"][0]
    assert result["policy_total_variation"][0] == 0


@pytest.mark.parametrize("fault", ["legal_empty", "legal_fraction", "legal_shape", "aux_only", "wdl_shape", "nan"])
def test_invalid_output_contract_is_refused(fault: str) -> None:
    reference, candidate, legal = _output_pair()
    if fault == "legal_empty":
        legal[:] = False
    elif fault == "legal_fraction":
        legal = legal.astype(float)
        legal[0, 0] = 0.5
    elif fault == "legal_shape":
        legal = legal[:, :2]
    elif fault == "aux_only":
        candidate["policy_sf"] = candidate.pop("policy")
    elif fault == "wdl_shape":
        candidate["wdl"] = candidate["wdl"][:, :2]
    elif fault == "nan":
        candidate["policy"][0, 12] = np.nan
    message = {"legal_empty": "legal move", "legal_fraction": "legal must", "legal_shape": "legal must",
               "aux_only": "main policy", "wdl_shape": r"candidate\.wdl", "nan": r"candidate\.policy"}[fault]
    with pytest.raises(ValueError, match=message):
        compare_outputs(reference, candidate, legal)


def test_summary_distinguishes_row_and_game_estimands_and_clusters_source_qualified_games() -> None:
    values = np.array([0.0, 0.0, 0.0, 1.0])
    keys = ["source-a/game-1"] * 3 + ["source-b/game-1"]
    report = summarize_metrics({"sensitivity": values}, cluster_keys=keys, bootstrap=1000, seed=3)
    assert report["rows"] == 4
    assert report["games"] == 2
    assert report["few_cluster_caution"] is True
    assert report["metrics"]["sensitivity"] == {
        "row_mean": 0.25, "equal_game_mean": 0.5, "equal_game_ci95": [0.0, 1.0],
    }
    order = [3, 1, 0, 2]
    reordered = summarize_metrics({"sensitivity": values[order]}, cluster_keys=[keys[i] for i in order],
                                  bootstrap=1000, seed=3)
    assert reordered == report


def test_one_game_has_no_cluster_interval_and_zero_bootstrap_is_explicit() -> None:
    values = {"metric": np.array([0.0, 1.0])}
    one = summarize_metrics(values, cluster_keys=["source/game"] * 2)
    assert one["interval_status"] == "unavailable_one_game"
    assert one["metrics"]["metric"]["equal_game_ci95"] is None
    disabled = summarize_metrics(values, cluster_keys=["source/a", "source/b"], bootstrap=0)
    assert disabled["interval_status"] == "disabled"
    assert disabled["metrics"]["metric"]["equal_game_ci95"] is None


@pytest.mark.parametrize("fault", ["blank_key", "metric_shape", "nan", "negative_bootstrap"])
def test_invalid_summary_contract_is_refused(fault: str) -> None:
    value = np.array([1.0, np.nan if fault == "nan" else 0.0])
    if fault == "metric_shape":
        value = value[:1]
    message = {"blank_key": "cluster_keys", "metric_shape": "metric", "nan": "metric",
               "negative_bootstrap": "bootstrap must"}[fault]
    with pytest.raises(ValueError, match=message):
        summarize_metrics({"metric": value}, cluster_keys=["source/a", " " if fault == "blank_key" else "source/b"],
                          bootstrap=-1 if fault == "negative_bootstrap" else 10)
