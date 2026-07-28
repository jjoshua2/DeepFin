from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.paired_compare import paired_bootstrap_ci


def test_bootstrap_ci_covers_known_shift() -> None:
    rng = np.random.default_rng(7)
    deltas = rng.normal(loc=-2.0, scale=5.0, size=2000)
    lo, hi = paired_bootstrap_ci(deltas, n_boot=4000, seed=1)
    assert lo < -2.0 < hi or (lo < deltas.mean() < hi)
    # width ~ 2*1.96*5/sqrt(2000) ≈ 0.44
    assert 0.2 < (hi - lo) < 1.0
    assert hi < 0  # a real 2cp shift at n=2000 is clearly significant


def test_bootstrap_ci_null_is_not_significant() -> None:
    rng = np.random.default_rng(3)
    deltas = rng.normal(loc=0.0, scale=5.0, size=2000)
    lo, hi = paired_bootstrap_ci(deltas, n_boot=4000, seed=2)
    assert lo < 0 < hi


def test_load_dump_audit_shape(tmp_path) -> None:
    from scripts.paired_compare import load_dump

    p = tmp_path / "audit.jsonl"
    rows = [
        {"key": "k1", "phase": "middlegame",
         "cand": {"search": {"exp": 12.5, "top1": 30.0}, "net": {"exp": 20.0}}},
        {"key": "k2", "phase": 1,
         "cand": {"search": {"exp": 0.0, "top1": 0.0}}},
        {"key": "k3", "cand": {"search": {"exp": None}}},  # null metric -> dropped
        {"key": "k4", "cand": {}},                          # missing path -> dropped
    ]
    with p.open("w") as f:
        for r in rows:
            f.write(__import__("json").dumps(r) + "\n")

    d = load_dump(str(p), join_key="key", field="cand.search.exp")
    assert set(d.rows) == {"k1", "k2"}
    assert d.rows["k1"] == (12.5, "middlegame")
    assert d.rows["k2"] == (0.0, "middlegame")  # int phase index mapped to name
    assert d.unusable == 2  # k3 null, k4 missing path

    top1 = load_dump(str(p), join_key="key", field="cand.search.top1")
    assert top1.rows["k1"] == (30.0, "middlegame")


def _write_jsonl(path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# Two ways this tool used to print a confident answer that was not the answer.
# Both matter more than an ordinary bug because every kill/hold decision in
# docs/experiment_ledger.md is read off this output: a comparison that launders
# a KILL into a HOLD is the worst defect class here.
# ---------------------------------------------------------------------------

def test_report_counts_null_rows_that_never_entered_the_join(tmp_path, capsys) -> None:
    """The reported drop count used to exclude nulls entirely.

    ``dropped`` was computed from the two INDEXES, so a row the scorer failed
    on -- absent from both -- was invisible. Demonstrated on main: eight rows
    per side, three of them null on both sides, printed "dropped 0". A dump the
    scorer had largely failed on read as a complete join.
    """
    from scripts.paired_compare import load_dump, report

    rows_a: list[dict[str, object]] = [{"fen": f"p{i}", "value": 10.0} for i in range(5)]
    rows_b: list[dict[str, object]] = [{"fen": f"p{i}", "value": 12.0} for i in range(5)]
    for i in range(3):  # measured on neither side -> in neither index
        rows_a.append({"fen": f"n{i}", "value": None})
        rows_b.append({"fen": f"n{i}", "value": None})

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", rows_a))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", rows_b))
    assert (a.unusable, b.unusable) == (3, 3)

    report(a, b, label_a="A", label_b="B", n_boot=200)

    out = capsys.readouterr().out
    assert "paired positions: 5\n" in out
    assert "A: 8 rows, 3 unusable, 0 unmatched   B: 8 rows, 3 unusable, 0 unmatched" in out


def test_report_counts_unmatched_and_unusable_separately(tmp_path, capsys) -> None:
    """The two losses are different problems and must not be summed away."""
    from scripts.paired_compare import load_dump, report

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        {"fen": "p1", "value": 10.0}, {"fen": "p2", "value": 20.0},
        {"fen": "only_a", "value": 5.0}, {"fen": "bad_a", "value": None},
    ]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        {"fen": "p1", "value": 13.0}, {"fen": "p2", "value": 23.0},
    ]))

    report(a, b, label_a="A", label_b="B", n_boot=200)

    out = capsys.readouterr().out
    assert "paired positions: 2\n" in out
    assert "A: 4 rows, 1 unusable, 1 unmatched   B: 2 rows, 0 unusable, 0 unmatched" in out
    assert "paired delta (A-B): -3.00" in out


def test_one_nan_row_cannot_launder_a_kill_into_a_hold(tmp_path, capsys) -> None:
    """``isinstance(v, (int, float))`` admits NaN, and one NaN poisons everything.

    numpy's ``mean`` and ``percentile`` propagate NaN, so the delta and both CI
    bounds print as ``nan``; ``nan < 0`` and ``nan > 0`` are both False, so the
    verdict falls through to "NOT significant". Measured on main with exactly
    this input: a clean -5.0 delta over 50 rows became "verdict at 95%: NOT
    significant" -- a KILL silently reported as a HOLD.
    """
    from scripts.paired_compare import load_dump, report

    rows_a = [{"fen": f"q{i}", "value": 100.0} for i in range(50)]
    rows_b = [{"fen": f"q{i}", "value": 105.0} for i in range(50)]
    rows_b[7]["value"] = float("nan")
    assert json.loads(json.dumps(rows_b[7]))["value"] != rows_b[7]["value"], \
        "NaN must survive the JSON round-trip or this test proves nothing"

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", rows_a))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", rows_b))
    assert b.unusable == 1

    report(a, b, label_a="A", label_b="B", n_boot=500)

    out = capsys.readouterr().out
    assert "nan" not in out.lower()
    assert "paired positions: 49\n" in out
  # The one lost position is visible on BOTH sides and counted once on each:
  # B could not measure it, so A has it unpartnered. Summing the two into a
  # single "dropped 2" would claim two positions were lost.
    assert "A: 50 rows, 0 unusable, 1 unmatched   B: 50 rows, 1 unusable, 0 unmatched" in out
    assert "paired delta (A-B): -5.00" in out
    assert "verdict at 95%: A better" in out


def test_infinite_metric_is_dropped_like_a_null(tmp_path) -> None:
    """``inf`` is numeric and finite-looking to ``isinstance`` too."""
    from scripts.paired_compare import load_dump

    d = load_dump(_write_jsonl(tmp_path / "inf.jsonl", [
        {"fen": "a", "value": 1.0},
        {"fen": "b", "value": float("inf")},
        {"fen": "c", "value": float("-inf")},
    ]))

    assert set(d.rows) == {"a"}
    assert d.unusable == 2


def test_total_scorer_failure_is_not_reported_as_a_schema_mismatch(tmp_path) -> None:
    """The empty-join message used to hide ``unusable`` entirely.

    Total scorer failure on one side is the extreme of the same defect the rest
    of this file pins: 50 rows on A, all 50 non-finite on B printed only "no
    joinable rows (A has 50, B has 0) -- check --join-key/--field against the
    dump schema". That is indistinguishable from an empty file or a wrong
    ``--field``, so it sends the operator after a config typo when in fact
    every position was scored and every score came back NaN.
    """
    from scripts.paired_compare import load_dump, report

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        {"fen": f"p{i}", "value": float(i)} for i in range(50)
    ]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        {"fen": f"p{i}", "value": float("nan")} for i in range(50)
    ]))
    assert (len(a.rows), a.unusable) == (50, 0)
    assert (len(b.rows), b.unusable) == (0, 50)

    with pytest.raises(SystemExit) as exc:
        report(a, b, label_a="A", label_b="B", n_boot=200)

    message = str(exc.value)
    assert "no joinable rows" in message
    # The number the operator needs, in the same per-side shape as the success
    # path. Without it the message reads as a schema problem.
    assert "A: 50 rows, 0 unusable, 50 indexed" in message
    assert "B: 50 rows, 50 unusable, 0 indexed" in message
    assert "scorer failed" in message
    # Name the side that indexed nothing. It is B here, and A indexed all 50 --
    # a message reading "A ... indexed nothing" would contradict the accounting
    # printed immediately before it on the same line.
    assert "B indexed nothing" in message
    assert "A indexed nothing" not in message


def test_empty_join_over_two_healthy_dumps_still_blames_the_join_key(tmp_path) -> None:
    """Both sides indexed fine and simply share no key -- that IS a schema/key
    problem, and the message must keep saying so rather than blaming the scorer.
    """
    from scripts.paired_compare import load_dump, report

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        {"fen": "a1", "value": 1.0}, {"fen": "a2", "value": 2.0},
    ]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        {"fen": "b1", "value": 1.0}, {"fen": "b2", "value": 2.0},
    ]))

    with pytest.raises(SystemExit) as exc:
        report(a, b, label_a="A", label_b="B", n_boot=200)

    message = str(exc.value)
    assert "A: 2 rows, 0 unusable, 2 indexed" in message
    assert "B: 2 rows, 0 unusable, 2 indexed" in message
    assert "--join-key/--field" in message
    assert "scorer failed" not in message
    assert "indexed nothing" not in message


def test_both_sides_all_unusable_names_both_sides(tmp_path) -> None:
    """When neither side indexed anything, say so for both rather than one."""
    from scripts.paired_compare import load_dump, report

    a = load_dump(_write_jsonl(tmp_path / "a.jsonl", [
        {"fen": "p1", "value": None}, {"fen": "p2", "value": None},
    ]))
    b = load_dump(_write_jsonl(tmp_path / "b.jsonl", [
        {"fen": "p1", "value": float("nan")}, {"fen": "p2", "value": float("nan")},
    ]))

    with pytest.raises(SystemExit) as exc:
        report(a, b, label_a="A", label_b="B", n_boot=200)

    message = str(exc.value)
    assert "A: 2 rows, 2 unusable, 0 indexed" in message
    assert "B: 2 rows, 2 unusable, 0 indexed" in message
    assert "A and B indexed nothing" in message


def test_load_dump_refuses_duplicate_join_keys(tmp_path) -> None:
    """Audit L14: duplicates used to last-win and stay out of ``dropped``.

    The caller then read a clean-looking join over a silently smaller and
    silently biased sample. Refusing is the only honest answer -- there is no
    principled winner between two rows claiming the same position.
    """
    from scripts.paired_compare import load_dump

    p = tmp_path / "dupes.jsonl"
    _write_jsonl(p, [
        {"fen": "a", "value": 1.0},
        {"fen": "b", "value": 2.0},
        {"fen": "a", "value": 99.0},
    ])

    with pytest.raises(SystemExit) as exc:
        load_dump(str(p))

    message = str(exc.value)
    assert "duplicate" in message
    assert "'fen'" in message
    assert "'a'" in message


def test_duplicate_detection_ignores_rows_that_never_entered_the_index(tmp_path) -> None:
    """A repeated key on unusable rows leaves the join unambiguous.

    Null/non-finite metrics are skipped before indexing, so two unusable rows
    sharing a key must not trip the refusal -- otherwise a dump with a couple of
    failed positions becomes unreadable.
    """
    from scripts.paired_compare import load_dump

    d = load_dump(_write_jsonl(tmp_path / "nulls.jsonl", [
        {"fen": "a", "value": 1.0},
        {"fen": "b", "value": None},
        {"fen": "b"},
    ]))

    assert set(d.rows) == {"a"}
    assert d.unusable == 2
