"""Cheap report-contract tests only; native compilation lives in the opt-in probe."""
from __future__ import annotations

import pytest

from native.bend_engine.bitboard_probe.run_probe import parse_report, relevant_bits


def good_report() -> str:
    return "\n".join(
        f"bishop={bishop} square={square} cases={2 * (1 << relevant_bits(bishop, square)) + 64} "
        "failures=0 first=0 sum_hi=0 sum_lo=0"
        for bishop in (0, 1) for square in range(64)
    )


def test_slider_report_requires_all_cases() -> None:
    rows = parse_report(good_report())
    assert len(rows) == 128
    assert sum(row["cases"] for row in rows) == 223488
    assert relevant_bits(0, 0) == 12
    assert relevant_bits(1, 27) == 9


@pytest.mark.parametrize(("text", "message"), [
    ("", "expected 128"),
    ("bishop=0", "malformed slider"),
    (good_report().splitlines()[0], "expected 128"),
    (good_report() + "\n" + good_report().splitlines()[0], "duplicate"),
    (good_report().replace("cases=8256", "cases=0", 1), "incomplete fixture"),
    (good_report().replace("failures=0", "failures=1", 1), "incorrect slider"),
    (good_report().replace("sum_hi=0", "sum_hi=4294967296", 1), "checksum limbs"),
])
def test_bad_slider_reports_are_rejected(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_report(text)
