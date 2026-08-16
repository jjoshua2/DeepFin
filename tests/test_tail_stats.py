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

from scripts.tail_stats import load, report, resolve_field, resolve_field_arg

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
