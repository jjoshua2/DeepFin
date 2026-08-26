from __future__ import annotations

from copy import deepcopy

from scripts.bench_slider_backends import regression_gate


def _report() -> dict[str, object]:
    return {
        "pack_sha256": "same-pack",
        "movegen": {"positions": 20000, "moves": 400000},
        "qsearch": {
            "rows": 400,
            "nodes": 12345,
            "nnue_evals": 6789,
            "us_per_node": 5.0,
        },
    }


def test_regression_gate_accepts_identical_or_faster_work() -> None:
    baseline = _report()
    current = deepcopy(baseline)
    assert regression_gate(current, baseline) == []
    qsearch = current["qsearch"]
    assert isinstance(qsearch, dict)
    qsearch["us_per_node"] = 4.0
    assert regression_gate(current, baseline) == []


def test_regression_gate_fails_any_deterministic_work_change() -> None:
    baseline = _report()
    current = deepcopy(baseline)
    qsearch = current["qsearch"]
    assert isinstance(qsearch, dict)
    qsearch["nodes"] = 12346
    failures = regression_gate(current, baseline)
    assert any("qsearch.nodes changed" in failure for failure in failures)


def test_regression_gate_fails_more_than_three_percent_per_node() -> None:
    baseline = _report()
    current = deepcopy(baseline)
    qsearch = current["qsearch"]
    assert isinstance(qsearch, dict)
    qsearch["us_per_node"] = 5.151
    failures = regression_gate(current, baseline)
    assert any("us/node regressed" in failure for failure in failures)


def test_regression_gate_accepts_exact_three_percent_boundary() -> None:
    baseline = _report()
    current = deepcopy(baseline)
    qsearch = current["qsearch"]
    assert isinstance(qsearch, dict)
    qsearch["us_per_node"] = 5.15
    assert regression_gate(current, baseline) == []


def test_regression_gate_refuses_pack_changes() -> None:
    baseline = _report()
    current = deepcopy(baseline)
    current["pack_sha256"] = "different-pack"
    failures = regression_gate(current, baseline)
    assert "NNUE pack SHA differs between baseline and current" in failures
