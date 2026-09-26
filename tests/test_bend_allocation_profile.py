"""Cheap instrumentation/parser contracts only; no compilation or perft in pytest."""
from __future__ import annotations

import pytest

from native.bend_engine.legal_probe.profile_allocations import (
    COUNTERS, instrument, parse_counters, replace_once,
)


def test_counter_row_accepts_zero_reference_wrappers() -> None:
    assert parse_counters("profile heap_alloc=42 rfc_wrap=0 rfc_bump=0 term_drop=3\n") == {
        "heap_alloc": 42, "rfc_wrap": 0, "rfc_bump": 0, "term_drop": 3,
    }


@pytest.mark.parametrize(("row", "reason"), [
    ("", "malformed"),
    ("profile heap_alloc=4 rfc_wrap=1 term_drop=2", "malformed"),
    ("profile heap_alloc=-4 rfc_wrap=1 rfc_bump=0 term_drop=2", "malformed"),
    ("profile heap_alloc=4 rfc_wrap=5 rfc_bump=0 term_drop=2", "exceeds"),
    (f"profile heap_alloc={1 << 64} rfc_wrap=1 rfc_bump=0 term_drop=2", "overflow"),
    ("profile heap_alloc=4 rfc_wrap=1 rfc_bump=0 term_drop=2\nextra", "malformed"),
])
def test_invalid_counter_rows_rejected(row: str, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        parse_counters(row)


@pytest.mark.parametrize("text", ["unrelated", "anchor anchor"])
def test_anchor_must_be_present_exactly_once(text: str) -> None:
    with pytest.raises(ValueError, match="anchor missing or ambiguous"):
        replace_once(text, "anchor", "replacement")


def test_instrumentation_lives_only_inside_measured_interval() -> None:
    # Synthetic boundary contract: source anchors, not a performance test.
    source = "\n".join(COUNTERS.values()) + '''
    perft_clock_start();
    perft_clock_finish("bend-pext",
    /* Keep the last table owner alive until AFTER the measurement. */
'''
    result = instrument(source)
    assert result.index("profile_active = 1;") > result.index("perft_clock_start();")
    assert result.index("profile_active = 0;") < result.index('perft_clock_finish("bend-pext",')
    assert result.index('fprintf(stderr, "profile ') > result.index('perft_clock_finish("bend-pext",')
    assert result.index('fprintf(stderr, "profile ') < result.index("/* Keep the last table")
    for name in COUNTERS:
        assert f"profile_{name} = 0;" in result
