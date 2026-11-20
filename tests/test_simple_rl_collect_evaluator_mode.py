"""The collect evaluator has ONE fast path: CUDA graphs of its eager dense
forward on CUDA, eager only where graphs cannot run (CPU). There is no flag;
the realised mode and the graph counters are banked in stats and manifest, and
a run pinned with the retired ``collect_cuda_graphs`` flag still resumes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from chess_anti_engine.simple_rl import collect as collect_mod
from chess_anti_engine.simple_rl.collect import CollectSettings, collect_generation
from tests.simple_rl_helpers import QuietStockfish, save_tiny_checkpoint

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")

LINES = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | e2e4 e7e5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | d2d4 d7d5 c2c4",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | g1f3",
]


def _collect(tmp_path: Path, name: str, monkeypatch: pytest.MonkeyPatch, **overrides: object) -> Path:
    monkeypatch.setattr(collect_mod, "_open_stockfish", lambda _s: QuietStockfish([0.0, 1.0, 0.0]))
    ckpt = tmp_path / "model.pt"
    if not ckpt.exists():
        save_tiny_checkpoint(ckpt, seed=5)
    openings = tmp_path / "train.jsonl"
    if not openings.exists():
        openings.write_text("".join(json.dumps({"id": f"o{i}", "book": 1, "line": ln}) + "\n"
                                    for i, ln in enumerate(LINES)))
    settings = replace(CollectSettings(games=3, concurrency=3, sims=8, max_plies=24, seed=9, book_mix=(1.0,)),
                       **overrides)
    out = tmp_path / name / "gen_0001" / "collect"
    collect_generation(gen_dir=out, generation=1, checkpoint=ckpt, train_jsonl=openings, settings=settings)
    return out


def test_cpu_runs_eager_and_banks_the_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = _collect(tmp_path, "cpu", monkeypatch)
    stats = json.loads((out / collect_mod.STATS).read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert (stats["cuda_graphs"], stats["evaluator_mode"], stats["cuda_graph_stats"]) == (False, "eager_cpu", None)
    assert manifest["stats"]["evaluator_mode"] == "eager_cpu"
    assert "cuda_graphs" not in manifest["config"]["settings"]


@needs_cuda
def test_cuda_always_replays_graphs_and_banks_its_counters(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = _collect(tmp_path, "cuda", monkeypatch, device="cuda")
    stats = json.loads((out / collect_mod.STATS).read_text())
    assert (stats["cuda_graphs"], stats["evaluator_mode"]) == (True, "cuda_graphs")
    assert stats["cuda_graph_stats"]["replays"] > 0
    assert json.loads((out / "manifest.json").read_text())["stats"]["cuda_graph_stats"] == stats["cuda_graph_stats"]


def test_the_flag_is_gone_and_old_run_configs_still_load(tmp_path: Path) -> None:
    from scripts import simple_rl_loop as loop

    parser = argparse.ArgumentParser()
    loop._common(parser)
    base = ["--run-dir", str(tmp_path), "--openings-dir", str(tmp_path)]
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--collect-cuda-graphs"])
    a = parser.parse_args(base)
    a.allow_config_change = False
    assert "collect_cuda_graphs" not in vars(a)
    loop.check_run_config(a, tmp_path)
    pinned = json.loads((tmp_path / loop.RUN_CONFIG).read_text())
    assert "collect_cuda_graphs" not in pinned["settings"]
    # Runs pinned before the flag was retired, with it on or off, resume unchanged.
    for value in (False, True):
        old = {**pinned, "settings": {**pinned["settings"], "collect_cuda_graphs": value}}
        (tmp_path / loop.RUN_CONFIG).write_text(json.dumps(old))
        loop.check_run_config(a, tmp_path)
        assert json.loads((tmp_path / loop.RUN_CONFIG).read_text()) == old  # nothing rewritten
    # A genuine change is still refused next to the retired key.
    a.sims = 128
    with pytest.raises(SystemExit, match=r"run settings differ.*sims"):
        loop.check_run_config(a, tmp_path)


@pytest.mark.parametrize(("bad", "reason"), [
    (["--eval-opening-plies", "1"], "longest EVAL line"),                       # below the split's
    ([], "exactly one of --init-checkpoint / --random-init"),                   # no gen-0 source
])
def test_a_refused_setting_is_never_pinned(tmp_path: Path, bad: list[str], reason: str) -> None:
    from scripts import simple_rl_loop as loop

    openings = tmp_path / "openings"
    openings.mkdir()
    (openings / "summary.json").write_text(json.dumps({"eval_max_plies": 6}))
    run_dir = tmp_path / "run"
    init = [] if not bad else ["--random-init"]
    no_eval = [] if "--eval-opening-plies" in bad else ["--no-eval"]
    with pytest.raises(SystemExit, match=reason):
        loop.main(["run", "--run-dir", str(run_dir), "--openings-dir", str(openings), "--device", "cpu",
                   "--generations", "1", *init, *no_eval, *bad])
    assert not (run_dir / loop.RUN_CONFIG).exists()


def test_validate_settings_refuses_each_settings_only_fault() -> None:
    from chess_anti_engine.simple_rl.collect import CollectSettings, validate_settings

    validate_settings(CollectSettings(games=1))
    with pytest.raises(ValueError, match="opening_sampling"):
        validate_settings(CollectSettings(games=1, opening_sampling="shuffled"))
    validate_settings(CollectSettings(games=1, device="cuda"))  # settings only: no GPU touched


def test_run_validates_eval_settings_only_while_an_evaluation_remains(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """A re-invoked `run` on a fully sealed run must not be refused over eval
    settings it will never use (e.g. the openings summary.json since removed)."""
    import argparse

    from scripts import simple_rl_loop as loop

    for g in (1, 2):
        for kind in ("collect", "train", "eval"):
            (tmp_path / f"gen_{g:04d}" / kind).mkdir(parents=True)
            (tmp_path / f"gen_{g:04d}" / kind / "manifest.json").write_text("{}")
    a = loop.parser().parse_args(["run", "--run-dir", str(tmp_path), "--generations", "2",
                                  "--openings-dir", str(tmp_path / "no_openings")])
    assert (a.no_eval, a.eval_opening_plies) == (False, None)
    assert loop.run_stages_left(a, tmp_path) == set()
    (tmp_path / "gen_0002" / "eval" / "manifest.json").unlink()
    assert loop.run_stages_left(a, tmp_path) == {"eval"}
    assert loop.run_stages_left(argparse.Namespace(generations=2, no_eval=True), tmp_path) == set()
    with pytest.raises(SystemExit, match=r"summary\.json missing"):
        loop.validate_before_pinning(a, tmp_path, loop.run_stages_left(a, tmp_path))
    (tmp_path / "gen_0002" / "eval" / "manifest.json").write_text("{}")
    loop.validate_before_pinning(a, tmp_path, loop.run_stages_left(a, tmp_path))  # nothing left: no eval check
    # Through `run` itself: every stage sealed (lineage checks stubbed: the
    # manifests are placeholders), no summary.json, no --eval-opening-plies.
    (tmp_path / "gen_0000").mkdir()
    (tmp_path / "gen_0000" / "manifest.json").write_text("{}")
    monkeypatch.setattr(loop, "verify_stage", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "tripwire_exit", lambda *_a, **_k: 0)
    assert loop.main(["run", "--run-dir", str(tmp_path), "--generations", "2",
                      "--openings-dir", str(tmp_path / "no_openings")]) == 0
