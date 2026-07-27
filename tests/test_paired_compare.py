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
    assert set(d) == {"k1", "k2"}
    assert d["k1"] == (12.5, "middlegame")
    assert d["k2"] == (0.0, "middlegame")  # int phase index mapped to name

    top1 = load_dump(str(p), join_key="key", field="cand.search.top1")
    assert top1["k1"] == (30.0, "middlegame")


def _write_jsonl(path, rows: list[dict]) -> str:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


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


def test_load_dump_duplicate_detection_ignores_dropped_rows(tmp_path) -> None:
    """A repeated key on rows that never enter the index is not a duplicate.

    Null/missing metrics are skipped before indexing, so two unusable rows
    sharing a key leave the join unambiguous and must not trip the refusal.
    """
    from scripts.paired_compare import load_dump

    p = tmp_path / "nulls.jsonl"
    _write_jsonl(p, [
        {"fen": "a", "value": 1.0},
        {"fen": "b", "value": None},
        {"fen": "b"},
    ])

    assert set(load_dump(str(p))) == {"a"}


def test_report_join_is_exact_when_no_duplicates(tmp_path, capsys) -> None:
    """The clean path is unchanged: unmatched rows still counted in ``dropped``."""
    from scripts.paired_compare import load_dump, report

    a = _write_jsonl(tmp_path / "a.jsonl", [
        {"fen": "p1", "value": 10.0}, {"fen": "p2", "value": 20.0},
        {"fen": "only_a", "value": 5.0},
    ])
    b = _write_jsonl(tmp_path / "b.jsonl", [
        {"fen": "p1", "value": 13.0}, {"fen": "p2", "value": 23.0},
    ])

    report(load_dump(a), load_dump(b), label_a="A", label_b="B", n_boot=200)

    out = capsys.readouterr().out
    assert "paired positions: 2 (dropped 1 unmatched/null)" in out
    assert "paired delta (A-B): -3.00" in out
