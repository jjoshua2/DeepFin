"""Collect's CUDA evaluator batch cap must cover the search's leaf buffer.

gumbel_c caps its leaf buffer at the evaluator's ``_max_batch``; a leaf that
finds the buffer full is backed up as a pseudo-terminal carrying the root's Q
instead of being evaluated: wrong-signed at every odd depth, fabricated at every
even one. At the old fixed 512 and 64-128 concurrent games
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


def test_the_eval_cap_boundary_is_accepted() -> None:
    s = CollectSettings(games=256, concurrency=128, topk=16, device="cuda")
    need = required_leaf_rows(s)
    assert realised_eval_max_batch(replace(s, eval_max_batch=need)) == need
    validate_settings(replace(s, eval_max_batch=need))
    with pytest.raises(ValueError, match="below"):
        realised_eval_max_batch(replace(s, eval_max_batch=need - 1))


def test_a_cpu_collect_records_the_cap_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_simple_rl_resume import _argv, _setup

    from scripts import simple_rl_loop

    openings, model_config = _setup(tmp_path, monkeypatch)
    argv = _argv(tmp_path / "run", openings, model_config)
    argv[argv.index("--generations") + 1] = "1"
    assert simple_rl_loop.main(argv) == 0
    stats = json.loads((tmp_path / "run" / "gen_0001" / "collect" / "stats.json").read_text())
    assert stats["leaf_rows_required"] == leaf_buffer_rows(4, topk=16, pipelined=False)
    assert stats["eval_max_batch_realised"] is None  # CPU: no capped evaluator was built


def _manifest(**settings: Any) -> dict[str, Any]:
    return {"config": {"settings": {"games": 256, "concurrency": 128, "topk": 16, "device": "cuda", **settings}}}


def test_capped_collect_record_reads_old_format_manifests() -> None:
    from chess_anti_engine.simple_rl.collect import capped_collect_record

    old = capped_collect_record(_manifest())  # no eval_max_batch key, no stats: the legacy fixed 512
    assert old == {"eval_max_batch": 512, "eval_max_batch_source": "legacy_fixed_512", "boards": 128,
                   "topk": 16, "leaf_request_per_rep": 2048}
    assert capped_collect_record(_manifest(device="cpu")) is None
    assert capped_collect_record(_manifest(eval_max_batch=2048)) is None  # boundary: one rep fits exactly
    just_below = capped_collect_record(_manifest(eval_max_batch=2047))
    assert just_below is not None
    assert just_below["eval_max_batch_source"] == "settings"
    assert capped_collect_record(_manifest(concurrency=16)) is None  # F12/Q50 shape: 16 x 16 = 256 <= 512
    stats_win = {**_manifest(eval_max_batch=None), "stats": {"eval_max_batch_realised": 4096}}
    assert capped_collect_record(stats_win) is None


def test_training_on_capped_rows_is_refused_unless_allowed_and_then_recorded(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_simple_rl_resume import _argv, _setup

    from scripts import simple_rl_loop

    openings, model_config = _setup(tmp_path, monkeypatch)
    run = tmp_path / "run"
    argv = _argv(run, openings, model_config)
    argv[argv.index("--generations") + 1] = "1"
    assert simple_rl_loop.main(argv) == 0
    # Pretend generation 1 was a CUDA collect at the old fixed cap (CPU cannot produce one).
    real = collect_mod.capped_collect_record
    monkeypatch.setattr(collect_mod, "capped_collect_record",
                        lambda m: real(_manifest()) if m.get("generation") == 1 else real(m))
    argv[argv.index("--generations") + 1] = "2"  # gen 2 re-presents gen 1 (--presentations 2)
    before = (run / "run_config.json").read_bytes()
    with pytest.raises(SystemExit, match=r"generation\(s\) \[1\] were collected at an evaluator cap"):
        simple_rl_loop.main(argv)
    assert (run / "run_config.json").read_bytes() == before  # refused before pinning
    assert not (run / "gen_0002").exists()
    assert simple_rl_loop.main([*argv, "--allow-capped-collect-rows", "--allow-config-change"]) == 0
    summary = json.loads((run / "gen_0002" / "train" / "metrics.json").read_text())["summary"]
    assert summary["trained_on_capped_rows"] is True
    assert summary["capped_collect_rows"]["1"]["eval_max_batch"] == 512
    ledger = json.loads((run / "presentation_ledger.json").read_text())
    assert ledger["trained_on_capped_rows"] is True


def test_a_capped_ancestor_outside_the_window_still_taints_and_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Gen 2 at --presentations 1 never presents gen 1's rows, but its model
    descends from gen 1's training: the taint is read from the whole lineage's
    collect manifests, not from a field older metrics never wrote."""
    from tests.test_simple_rl_resume import _argv, _setup

    from scripts import simple_rl_loop

    openings, model_config = _setup(tmp_path, monkeypatch)
    run = tmp_path / "run"
    argv = _argv(run, openings, model_config)
    argv[argv.index("--generations") + 1] = "1"
    argv[argv.index("--presentations") + 1] = "1"
    assert simple_rl_loop.main(argv) == 0
    metrics = run / "gen_0001" / "train" / "metrics.json"
    old = json.loads(metrics.read_text())  # pre-PR metrics carry no taint fields
    old["summary"].pop("trained_on_capped_rows")
    metrics.write_text(json.dumps(old))
    real = collect_mod.capped_collect_record
    monkeypatch.setattr(collect_mod, "capped_collect_record",
                        lambda m: real(_manifest()) if m.get("generation") == 1 else real(m))
    argv[argv.index("--generations") + 1] = "2"
    with pytest.raises(SystemExit, match=r"generation\(s\) \[1\] were collected at an evaluator cap"):
        simple_rl_loop.main(argv)
    assert not (run / "gen_0002").exists()
    assert simple_rl_loop.main([*argv, "--allow-capped-collect-rows", "--allow-config-change"]) == 0
    summary = json.loads((run / "gen_0002" / "train" / "metrics.json").read_text())["summary"]
    assert summary["window_generations"] == [2]
    assert summary["capped_collect_rows"] == {}  # nothing capped IN the window
    assert summary["trained_on_capped_rows"] is True  # inherited from gen 1
    assert summary["capped_lineage_collects"]["1"]["eval_max_batch"] == 512
    assert json.loads((run / "presentation_ledger.json").read_text())["capped_lineage_collects"]["1"]
    # The phase itself refuses too (not only the driver's pre-pin check).
    from chess_anti_engine.simple_rl.learn import TrainSettings, train_generation

    model = run / "gen_0001" / "train" / "model.pt"
    assert model.is_file()
    with pytest.raises(ValueError, match=r"generation\(s\) \[1\] were collected at an evaluator cap"):
        train_generation(run_dir=run, generation=2, init_checkpoint=model, prev_trainer_checkpoint=None,
                         settings=TrainSettings(device="cpu", presentations=1))
