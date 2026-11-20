"""Collect's CUDA evaluator batch cap must cover the search's leaf buffer.

gumbel_c caps its leaf buffer at the evaluator's ``_max_batch``; a leaf that
finds the buffer full is backed up as a pseudo-terminal carrying the root's Q
instead of being evaluated. At the old fixed 512 and 64-128 concurrent games
(topk 16) that hit most first-round children and corrupted the reply side's
search (2026-11-20 diagnostic). The cap now defaults to the search's own
leaf-buffer size and a smaller explicit value is refused.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.mcts.gumbel_c import leaf_buffer_rows
from chess_anti_engine.simple_rl import collect as collect_mod
from chess_anti_engine.simple_rl.collect import (
    CollectSettings, collect_generation, realised_eval_max_batch, required_leaf_rows, validate_settings,
)


def test_the_default_cap_is_the_searchs_leaf_buffer_at_this_concurrency() -> None:
    s = CollectSettings(games=256, concurrency=128, topk=16)
    assert required_leaf_rows(s) == leaf_buffer_rows(128, topk=16, pipelined=False) == 4096
    assert realised_eval_max_batch(s) == 4096
    small = CollectSettings(games=3, concurrency=128, topk=16)  # never more boards than games
    assert required_leaf_rows(small) == leaf_buffer_rows(3, topk=16, pipelined=False)
    assert realised_eval_max_batch(replace(s, eval_max_batch=8192)) == 8192


def test_an_explicit_cap_below_the_leaf_buffer_is_refused() -> None:
    s = CollectSettings(games=256, concurrency=128, topk=16, eval_max_batch=512)
    with pytest.raises(ValueError, match="below the 4096 leaf rows"):
        realised_eval_max_batch(s)


def test_the_pre_pin_validation_refuses_a_leaf_dropping_cap() -> None:
    # The loop driver calls validate_settings BEFORE pinning run_config.json,
    # so a refused cap must be refused there, not only inside the collect.
    bad = CollectSettings(games=256, concurrency=128, topk=16, device="cuda", eval_max_batch=512)
    with pytest.raises(ValueError, match="--eval-max-batch 512 is below"):
        validate_settings(bad)
    validate_settings(replace(bad, eval_max_batch=None))
    validate_settings(replace(bad, device="cpu"))  # CPU builds no capped evaluator


def test_the_evaluator_the_search_consumes_gets_the_realised_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    import chess_anti_engine.inference as inference

    seen: dict[str, Any] = {}

    class Spy:
        def __init__(self, _model: Any, **kwargs: Any) -> None:
            seen.update(kwargs)
            self._max_batch = kwargs["max_batch"]

    monkeypatch.setattr(inference, "DirectGPUEvaluator", Spy)
    ev = collect_mod._evaluator(object(), CollectSettings(games=64, concurrency=64, topk=16, device="cuda"))
    assert seen["max_batch"] == leaf_buffer_rows(64, topk=16, pipelined=False) == 2048
    assert collect_mod.evaluator_graph_record(ev)["eval_max_batch_realised"] == 2048


def test_collect_refuses_a_leaf_dropping_cap_before_any_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_s: Any) -> Any:
        raise AssertionError("Stockfish was opened before the cap was checked")

    monkeypatch.setattr(collect_mod, "_open_stockfish", boom)
    openings = tmp_path / "train.jsonl"
    openings.write_text(json.dumps({"id": "o0", "book": 1, "line": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w "
                                                               "KQkq - 0 1 | e2e4"}) + "\n")
    settings = CollectSettings(games=128, concurrency=128, topk=16, device="cuda", eval_max_batch=512)
    with pytest.raises(ValueError, match="--eval-max-batch 512 is below"):
        collect_generation(gen_dir=tmp_path / "gen_0001" / "collect", generation=1,
                           checkpoint=tmp_path / "missing.pt", train_jsonl=openings, settings=settings)


def test_the_loop_flag_defaults_to_auto_and_reaches_the_settings(tmp_path: Path) -> None:
    import argparse

    from scripts import simple_rl_loop as loop

    parser = argparse.ArgumentParser()
    loop._common(parser)
    base = ["--run-dir", str(tmp_path), "--openings-dir", str(tmp_path)]
    assert parser.parse_args(base).eval_max_batch is None
    assert loop.collect_settings(parser.parse_args(base)).eval_max_batch is None
    assert loop.collect_settings(parser.parse_args([*base, "--eval-max-batch", "8192"])).eval_max_batch == 8192


def test_the_loop_refuses_a_leaf_dropping_cap_before_pinning(tmp_path: Path) -> None:
    from scripts import simple_rl_loop as loop

    args = ["collect", "--run-dir", str(tmp_path), "--openings-dir", str(tmp_path), "--device", "cuda",
            "--concurrency", "128", "--games", "256", "--generation", "1"]
    a = loop.parser().parse_args([*args, "--eval-max-batch", "512"])
    with pytest.raises(SystemExit, match="--eval-max-batch 512 is below"):
        loop.validate_before_pinning(a, tmp_path, {"collect"})
    assert not (tmp_path / "run_config.json").exists()
