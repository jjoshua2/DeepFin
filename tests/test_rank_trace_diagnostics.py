"""Tiny rank and trace-reader controls; no stored measurements or simulator run."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scratchpad.sfpolicy_compare import arm_contrast_closed_form as contrast
from scripts import sim_gather_policy as gather


def test_average_ranks_assign_identical_values_identical_mean_rank() -> None:
    values = np.array([4.0, 1.0, 4.0, 2.0, 1.0])
    np.testing.assert_array_equal(contrast._average_ranks(values), [3.5, 0.5, 3.5, 2.0, 0.5])


def test_spearman_ties_match_known_coefficient_and_pair_permutation() -> None:
    a = np.tile([0.0, 0.0, 1.0, 2.0], 25)
    b = np.tile([0.0, 1.0, 1.0, 2.0], 25)
    assert contrast._spearman(a, b) == pytest.approx(5.0 / 6.0)
    permutation = np.arange(100).reshape(25, 4)[:, [2, 0, 3, 1]].ravel()[::-1]
    assert contrast._spearman(a[permutation], b[permutation]) == pytest.approx(5.0 / 6.0)


@pytest.mark.parametrize("constant_axis", [0, 1])
def test_spearman_constant_input_is_unavailable(constant_axis: int) -> None:
    arrays = [np.arange(100.0), np.arange(100.0)]
    arrays[constant_axis] = np.ones(100)
    assert math.isnan(contrast._spearman(*arrays))


@pytest.mark.parametrize("size", [0, 99, 100])
def test_spearman_retains_minimum_finite_pair_count(size: int) -> None:
    a = np.arange(float(size))
    result = contrast._spearman(a, a)
    if size < 100:
        assert math.isnan(result)
    else:
        assert result == pytest.approx(1.0)


def test_spearman_filters_nonfinite_pairs_before_ranking_and_counting() -> None:
    a = np.tile([0.0, 0.0, 1.0, 2.0], 25)
    b = np.tile([0.0, 1.0, 1.0, 2.0], 25)
    invalid_a = np.array([float("nan"), 7.0, float("inf"), 9.0])
    invalid_b = np.array([7.0, float("nan"), 9.0, float("-inf")])
    assert contrast._spearman(np.concatenate([a, invalid_a]),
                               np.concatenate([b, invalid_b])) == pytest.approx(5.0 / 6.0)
    assert math.isnan(contrast._spearman(np.concatenate([a[:-1], invalid_a]),
                                         np.concatenate([b[:-1], invalid_b])))


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_spearman_tied_monotone_inputs_keep_perfect_correlation(sign: float) -> None:
    a = np.repeat([0.0, 1.0, 2.0], [40, 30, 30])
    assert contrast._spearman(a, sign * a) == pytest.approx(sign)


@pytest.mark.parametrize("pair", [
    [float("nan"), 1], [float("inf"), 1], [float("-inf"), 1],
    [10**400, 1], [1.0, float("inf")], [1.0, float("nan")],
    [1.0, 0], [1.0, -1], [None, 1], ["bad", 1], [1.0, None], [1.0], "bad", None,
])
def test_trace_loader_skips_invalid_arrivals_with_explicit_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], pair: Any,
) -> None:
    path = tmp_path / "synthetic.jsonl"
    path.write_text(json.dumps({"arrivals": [[100.0, 2], pair, [101.0, 3]]}) + "\n", encoding="utf-8")
    assert gather.load_trace(str(path)) == [(100.0, 2), (101.0, 3)]
    text = capsys.readouterr().err
    assert f"skipped 1 malformed arrivals at {path}:1" in text


def test_trace_loader_keeps_sorted_valid_records_and_reports_bad_lines(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "mixed.jsonl"
    path.write_text("\nnot JSON\n{}\n" + json.dumps({"arrivals": [[102, 3], [100, 2]]}) + "\n", encoding="utf-8")
    assert gather.load_trace(str(path)) == [(100.0, 2), (102.0, 3)]
    text = capsys.readouterr().err
    assert "skip malformed JSON" in text
    assert "missing/invalid 'arrivals'" in text


def test_relative_time_origin_is_first_valid_arrival(capsys: pytest.CaptureFixture[str]) -> None:
    stream = [(float("nan"), 1), (float("inf"), 2), (99.0, 0), (100.0, 2), (101.0, 3)]
    assert gather._to_ms_stream(stream) == [(0.0, 2), (1000.0, 3)]
    assert "skipped 3 malformed arrivals converting to milliseconds" in capsys.readouterr().err


def test_relative_time_skips_conversion_overflow(capsys: pytest.CaptureFixture[str]) -> None:
    assert gather._to_ms_stream([(-1e308, 1), (1e308, 2)]) == [(0.0, 1)]
    assert "skipped 1 malformed arrivals converting to milliseconds" in capsys.readouterr().err


def test_direct_stream_catches_timestamp_and_row_conversion_overflow(capsys: pytest.CaptureFixture[str]) -> None:
    malformed: Any = [(10**400, 1), (100.0, float("inf")), (101.0, 3)]
    assert gather._to_ms_stream(malformed) == [(0.0, 3)]
    assert "skipped 2 malformed arrivals converting to milliseconds" in capsys.readouterr().err


def test_all_invalid_arrivals_do_not_fabricate_time_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert gather._to_ms_stream([(float("nan"), 1), (100.0, 0)]) == []
    assert "skipped 2 malformed arrivals" in capsys.readouterr().err


def test_empty_and_valid_relative_stream_are_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    assert gather._to_ms_stream([]) == []
    assert gather._to_ms_stream([(100.0, 2), (101.0, 3)]) == [(0.0, 2), (1000.0, 3)]
    assert capsys.readouterr().err == ""
