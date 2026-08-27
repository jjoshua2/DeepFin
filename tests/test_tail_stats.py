"""`scripts/tail_stats.py` must be able to address the quantity a prereg names.

The defect these tests pin: the script's whole argument surface was
(dump_a, dump_b, --raw-top1, --tail-cp), and `--raw-top1` reads candidate (a)
`cand.raw.top1` -- the net's own policy, which has ZERO dependence on
`--stockfish`. A prereg that pre-commits on candidate (c) `cand.sf_soft.top1`
and reads it with `--raw-top1` therefore compares two byte-identical columns:
the paired flip count is identically 0 and the verdict is INCONCLUSIVE by
construction, for any teacher, after four 4000-position labeling runs.

So the load-bearing assertion is not "--field parses" but "--field
cand.sf_soft.top1 and --raw-top1 return DIFFERENT numbers on the same pair of
dumps".
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from chess_anti_engine.utils.audit_cache_format import (
    AUDIT_CACHE_FORMAT,
    ROW_COUNT_KEY,
    STAMP_FORMAT_KEY,
)
from scripts.tail_stats import (
    DEFAULT_FIELD,
    load,
    report,
    resolve_field,
    resolve_field_arg,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _row(key: str, phase: int, *, raw: float, sf_soft: float) -> dict:
    """One `audit_targets --dump-per-position` row, trimmed to what we read."""
    return {
        "key": key,
        "phase": phase,
        "cand": {
            "raw": {"top1": raw, "exp": raw / 2.0},
            "sf_soft": {"top1": sf_soft, "exp": sf_soft / 2.0},
        },
    }


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture
def arm_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two arms differing ONLY in the SF-derived candidate (c).

    This is the shape of the Stockfish-upgrade 2x2: candidate (a) comes off the
    checkpoint and is identical between arms, candidate (c) is the SF MultiPV
    soft target and is the thing that moved.
    """
    old = [_row(f"k{i}", i % 2, raw=100.0 * i, sf_soft=100.0 * i) for i in range(8)]
    new = []
    for i in range(8):
        # (a) identical to OLD; (c) two positions cross the 300cp tail boundary.
        sf = 100.0 * i
        if i in (1, 2):
            sf = 900.0
        new.append(_row(f"k{i}", i % 2, raw=100.0 * i, sf_soft=sf))
    return _write(tmp_path / "old.jsonl", old), _write(tmp_path / "new.jsonl", new)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/tail_stats.py", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
    )


def test_raw_top1_cannot_see_an_sf_only_difference(arm_pair: tuple[Path, Path]) -> None:
    """The bug, pinned: candidate (a) reads a flat 0 across an SF change."""
    old, new = arm_pair
    a = load(str(old), "cand.raw.top1")
    b = load(str(new), "cand.raw.top1")
    assert a == b, "candidate (a) must be identical between arms in this fixture"


def test_field_selector_addresses_the_sf_soft_candidate(
    arm_pair: tuple[Path, Path],
) -> None:
    """--field reaches candidate (c), which --raw-top1 structurally cannot."""
    old, new = arm_pair
    a = load(str(old), "cand.sf_soft.top1")
    b = load(str(new), "cand.sf_soft.top1")
    flips = sum(1 for k in a if b[k][0] > 300.0 >= a[k][0])
    assert flips == 2, "the SF-only tail flips must be visible on cand.sf_soft.top1"
    assert a != b


def test_cli_field_and_raw_top1_disagree_on_the_same_dumps(
    arm_pair: tuple[Path, Path],
) -> None:
    """End-to-end: the two readouts print DIFFERENT paired flip lines.

    If this ever passes with identical lines, the field selector is decorative
    and the prereg's deciding yardstick is back to being unable to fire.
    """
    old, new = arm_pair
    raw = _run([str(old), str(new), "--raw-top1"])
    soft = _run([str(old), str(new), "--field", "cand.sf_soft.top1"])
    assert raw.returncode == 0, raw.stderr
    assert soft.returncode == 0, soft.stderr
    assert "new-in-B 0, fixed-in-B 0" in raw.stdout, raw.stdout
    assert "new-in-B 2, fixed-in-B 0" in soft.stdout, soft.stdout
    assert "[field] cand.sf_soft.top1" in soft.stdout
    assert "[field] cand.raw.top1" in raw.stdout


def test_raw_top1_is_an_alias_not_a_second_mechanism() -> None:
    assert resolve_field_arg(None, True) == "cand.raw.top1"
    assert resolve_field_arg("cand.raw.top1", True) == "cand.raw.top1"
    assert resolve_field_arg(None, False) == "value"
    assert resolve_field_arg("cand.train.exp", False) == "cand.train.exp"


def test_conflicting_field_and_alias_is_refused() -> None:
    with pytest.raises(SystemExit) as e:
        resolve_field_arg("cand.sf_soft.top1", True)
    assert "conflicts" in str(e.value)


def test_a_mis_addressed_field_is_an_error_not_an_empty_null(tmp_path: Path) -> None:
    """A typo'd field must not read as 'no difference'.

    Returning {} makes the paired flip count 0, which is exactly what a real
    null looks like -- the failure this whole file exists for.
    """
    p = _write(tmp_path / "d.jsonl", [_row("k0", 0, raw=1.0, sf_soft=2.0)])
    with pytest.raises(SystemExit) as e:
        load(str(p), "cand.sf_softt.top1")
    assert "0 of 1 rows" in str(e.value)


def test_report_survives_a_phase_that_matches_nothing(tmp_path: Path) -> None:
    """np.percentile on an empty array raises IndexError; the readout must not."""
    p = _write(tmp_path / "d.jsonl", [_row("k0", 0, raw=1.0, sf_soft=2.0)])
    d = load(str(p), "cand.raw.top1")
    report("d.jsonl", d, "middlegame")  # phase 0 is endgame -> empty selection


def test_cli_survives_an_empty_phase_selection(arm_pair: tuple[Path, Path]) -> None:
    old, new = arm_pair
    r = _run([str(old), str(new), "--field", "cand.raw.top1", "--phase", "opening"])
    assert r.returncode == 0, r.stderr
    assert "no rows in this phase" in r.stdout


def test_resolve_field_returns_none_on_a_non_dict_hop() -> None:
    assert resolve_field({"cand": {"raw": 3}}, "cand.raw.top1") is None
    assert resolve_field({"cand": [1, 2]}, "cand.raw") is None
    assert resolve_field({"value": 7.5}, "value") == 7.5


def test_booleans_are_not_scored_as_numbers(tmp_path: Path) -> None:
    """`top1_agree` is a bool sitting next to `top1`; bool is an int subclass."""
    p = _write(tmp_path / "d.jsonl", [
        {"key": "k0", "phase": 0, "cand": {"raw": {"top1_agree": True}}},
    ])
    with pytest.raises(SystemExit):
        load(str(p), "cand.raw.top1_agree")


# --------------------------------------------------------------------------
# The provenance-stamped path. Every fixture above is UNSTAMPED, which made
# the header-skipping guard untestable by them: PR #440 review B2 showed a
# mutant that calls `iter_data_rows`, discards it, and restores a per-line
# `json.loads` SURVIVES all 142 tests in the four most relevant files while
# genuinely scoring the stamp as a position. `tests/test_paired_compare_gate_
# is_wired.py` asserts the CALL through the AST, which cannot see that.
# These tests exercise the behaviour on a real stamped dump instead.
# --------------------------------------------------------------------------


def _stamp(rows: int) -> dict:
    """A provenance header of the shape `write_audit_cache` emits."""
    return {
        STAMP_FORMAT_KEY: AUDIT_CACHE_FORMAT,
        "policy_map_version": "pmv",
        "audit_ruler_version": "arv",
        "audit_set": "s",
        "audit_set_digest": "d",
        ROW_COUNT_KEY: rows,
    }


def _write_stamped(path: Path, rows: list[dict]) -> Path:
    lines = [json.dumps(_stamp(len(rows)), sort_keys=True)]
    lines += [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_the_stamp_is_not_scored_as_a_position(tmp_path: Path) -> None:
    """⚑ A stamped dump must yield the DATA rows only, COUNTED.

    A reader that parses every line scores the header too, so `n` is one too
    high and every statistic is contaminated by a row that is not a position.

    ⚑⚑ THE FIELD HERE IS DELIBERATELY ONE THE STAMP AND THE ROWS BOTH CARRY,
    and that is the whole design of this test. Asserted against
    `cand.raw.top1` it would be VACUOUS: the stamp has no `cand`, so a
    header-scoring reader drops the header on the numeric test anyway and
    returns the same 3 rows — the "immune by accident" property #442 named.
    A test that passes for both the right and the wrong reader pins nothing.
    Giving the data rows a key the stamp also has makes the two readers
    disagree (3 vs 4) and the assertion load-bearing.
    """
    rows = [_row(f"k{i}", 0, raw=100.0 * i, sf_soft=100.0 * i) for i in range(3)]
    for i, r in enumerate(rows):
        r[ROW_COUNT_KEY] = float(i)  # collides with the stamp's own `rows`
    got = load(str(_write_stamped(tmp_path / "stamped.jsonl", rows)), ROW_COUNT_KEY)
    assert len(got) == 3, f"expected the 3 data rows, got {len(got)}: {sorted(got)}"
    assert set(got) == {"k0", "k1", "k2"}


def test_a_field_that_resolves_only_on_the_stamp_is_refused(tmp_path: Path) -> None:
    """⚑ `--field` ENDS the "immune by accident" property #442 relied on.

    The old reader survived the header because the stamp carries no `value` and
    no `cand`. An arbitrary dotted path has no such luck: the stamp carries
    `rows`, an int. Scoring it would report a clean n=1 readout whose entire
    population is the provenance header.
    """
    rows = [_row(f"k{i}", 0, raw=100.0 * i, sf_soft=100.0 * i) for i in range(3)]
    path = str(_write_stamped(tmp_path / "stamped.jsonl", rows))
    with pytest.raises(SystemExit) as exc:
        load(path, ROW_COUNT_KEY)
    assert "0 of 3 rows" in str(exc.value), str(exc.value)


def test_a_dump_with_no_data_rows_is_refused_not_reported_as_zero(
    tmp_path: Path,
) -> None:
    """⚑ PR #440 review B1 — the regression this pins.

    The guard was `if n_rows and not rows`. Once `n_rows` counts DATA rows the
    header no longer pads it, so a header-only/empty/truncated dump drove the
    count to 0 and short-circuited the refusal: `net +0 blowups` at exit 0,
    which is indistinguishable from a real null. `write_audit_cache(path, [])`
    produces exactly this file. Guard on the OUTPUT, never on a count the
    reader's own skipping can zero.
    """
    header_only = str(_write_stamped(tmp_path / "hdr.jsonl", []))
    with pytest.raises(SystemExit) as exc:
        load(header_only, DEFAULT_FIELD)
    assert "0 data rows" in str(exc.value), str(exc.value)

    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        load(str(empty), DEFAULT_FIELD)
    assert "0 data rows" in str(exc.value), str(exc.value)


def test_an_unstamped_dump_still_loads(tmp_path: Path) -> None:
    """Back-compat: dumps banked before the stamp existed must still read."""
    rows = [_row(f"k{i}", 0, raw=100.0 * i, sf_soft=100.0 * i) for i in range(3)]
    got = load(str(_write(tmp_path / "bare.jsonl", rows)), "cand.raw.top1")
    assert len(got) == 3
