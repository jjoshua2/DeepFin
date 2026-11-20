"""Evaluate-stage guards: complete arenas only, an idempotent ledger, finite
tripwires, opening plies that cover every EVAL line, and the split / stage
re-run refusals of the CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.simple_rl import evaluate as ev
from scripts import simple_rl_loop


def _kw(run: Path) -> dict[str, Any]:
    for g in (0, 1):
        (run / f"gen_{g:04d}").mkdir(parents=True, exist_ok=True)
    for p in ("a.pt", "b.pt", "c.pt", "o.pgn"):
        (run / p).write_text(p)
    return {"run_dir": run, "generation": 1, "candidate": run / "a.pt", "previous": run / "b.pt",
            "reference": run / "c.pt", "openings_pgn": run / "o.pgn", "settings": ev.EvalSettings(games=2)}


def test_a_kill_between_the_two_matches_leaves_no_duplicate_ledger_rows(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review item 4 (= lead D1), the reviewer's ledger_dup probe as a test."""
    calls = {"n": 0}

    def fake(cmd: list[str], *, out: Path, log: Path) -> dict[str, Any]:
        del cmd, log
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("killed during vs_reference")
        out.write_text(json.dumps({"score": 0.5}) + "\n")
        return {"score": 0.5, "games": 2, "games_requested": 2, "truncated": False}

    monkeypatch.setattr(ev, "run_arena", fake)
    kw = _kw(tmp_path)
    (tmp_path / ev.LEDGER).write_text(json.dumps({"generation": 0, "match": "older"}) + "\n")
    with pytest.raises(KeyboardInterrupt):
        ev.evaluate_generation(**kw)
    ev.evaluate_generation(**kw)
    ev.evaluate_generation(**kw)  # a full redo of a finished generation replaces, never appends
    rows = [json.loads(r) for r in (tmp_path / ev.LEDGER).read_text().splitlines()]
    assert [(r["generation"], r["match"]) for r in rows] == [(0, "older"), (1, "vs_previous"), (1, "vs_reference")]


@pytest.mark.parametrize("record", [
    {"games": 6, "games_requested": 8, "truncated": True},
    {"games": 8, "games_requested": 8, "truncated": True},
    {"games": 6, "games_requested": 8, "truncated": False},
    {"games": 8, "truncated": False},
])
def test_an_incomplete_arena_record_is_refused(tmp_path: Path, record: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError, match="incomplete"):
        ev.check_complete(record, tmp_path / "arena.jsonl")
    ev.check_complete({"games": 8, "games_requested": 8, "truncated": False}, tmp_path / "arena.jsonl")


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_tripwire_limit_is_refused(limit: float) -> None:
    rows = [{"generation": 1, "draw_rate": 0.99}]
    with pytest.raises(ValueError, match="finite"):
        ev.check_tripwires(rows, {"max_draw_rate": limit})
    assert ev.check_tripwires(rows, {"max_draw_rate": 0.5})


def test_eval_opening_plies_follow_the_split_and_refuse_less(tmp_path: Path) -> None:
    (tmp_path / "summary.json").write_text(json.dumps({"eval_max_plies": 16}))

    def ns(plies: int | None) -> argparse.Namespace:
        return argparse.Namespace(openings_dir=str(tmp_path), eval_opening_plies=plies)

    assert simple_rl_loop.eval_opening_plies(ns(None)) == 16
    assert simple_rl_loop.eval_opening_plies(ns(20)) == 20
    with pytest.raises(SystemExit, match="mid-line"):
        simple_rl_loop.eval_opening_plies(ns(8))


def test_a_split_rerun_with_other_settings_is_refused(tmp_path: Path) -> None:
    from chess_anti_engine.simple_rl import common

    out = tmp_path / "split"
    out.mkdir()
    book = tmp_path / "b.pgn"
    common.write_json_atomic(out / common.MANIFEST, {"config": {
        "eval_lines": 4000, "salt": "simple-rl-v1", "books": [{"path": str(book), "max_plies": 16, "book": 1}]}})
    assert simple_rl_loop.main(["split-openings", "--out", str(out), "--book", f"{book}:16"]) == 0
    with pytest.raises(SystemExit, match="different settings"):
        simple_rl_loop.main(["split-openings", "--out", str(out), "--book", f"{book}:16", "--salt", "other"])
    with pytest.raises(SystemExit, match="different settings"):
        simple_rl_loop.main(["split-openings", "--out", str(out), "--book", f"{book}:8"])
