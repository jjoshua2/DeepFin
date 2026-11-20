"""``--collect-cuda-graphs``: refused off CUDA, pinned, banked, and the same shards.

The collect evaluator is already on dense fp32 transport, so replaying CUDA
graphs of its eager forward must leave every stored row unchanged. Off CUDA
the flag is refused rather than dropped, and the realised flag plus the graph
counters are banked in the collect stage's stats and manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from chess_anti_engine.simple_rl import collect as collect_mod
from chess_anti_engine.simple_rl.collect import CollectSettings, collect_generation, load_generation_rows
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


def _rows_digest(collect_dir: Path) -> str:
    arrays, meta = load_generation_rows(collect_dir)
    h = hashlib.sha256()
    for part in (arrays, meta):
        for key in sorted(part):
            value = np.ascontiguousarray(part[key])
            h.update(key.encode())
            h.update(str(value.dtype).encode() + str(value.shape).encode())
            h.update(value.tobytes())
    h.update((collect_dir / collect_mod.GAMES).read_bytes())
    return h.hexdigest()


def test_refused_without_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="cuda_graphs needs a CUDA device"):
        _collect(tmp_path, "cpu", monkeypatch, device="cpu", cuda_graphs=True)


def test_off_is_banked_as_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = _collect(tmp_path, "off", monkeypatch)
    stats = json.loads((out / collect_mod.STATS).read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert stats["cuda_graphs"] is False
    assert stats["cuda_graph_stats"] is None
    assert manifest["config"]["settings"]["cuda_graphs"] is False
    assert manifest["stats"]["cuda_graphs"] is False


def test_the_loop_flag_reaches_the_settings_and_is_pinned(tmp_path: Path) -> None:
    from scripts import simple_rl_loop as loop

    parser = argparse.ArgumentParser()
    loop._common(parser)
    base = ["--run-dir", str(tmp_path), "--openings-dir", str(tmp_path)]
    on = parser.parse_args([*base, "--collect-cuda-graphs"])
    off = parser.parse_args(base)
    assert loop.collect_settings(on).cuda_graphs is True
    assert loop.collect_settings(off).cuda_graphs is False
    for ns in (on, off):
        ns.allow_config_change = False
    loop.check_run_config(off, tmp_path)
    pinned = json.loads((tmp_path / loop.RUN_CONFIG).read_text())
    assert pinned["settings"]["collect_cuda_graphs"] is False
    with pytest.raises(SystemExit, match="collect_cuda_graphs"):
        loop.check_run_config(on, tmp_path)
    # A run pinned before the flag existed resumes at its default, refuses otherwise.
    del pinned["settings"]["collect_cuda_graphs"]
    (tmp_path / loop.RUN_CONFIG).write_text(json.dumps(pinned))
    loop.check_run_config(off, tmp_path)
    with pytest.raises(SystemExit, match="collect_cuda_graphs"):
        loop.check_run_config(on, tmp_path)
    # A run pinned ON must not resume OFF: the stored value wins over the
    # added-since-pinning default.
    on_dir = tmp_path / "pinned_on"
    on_dir.mkdir()
    loop.check_run_config(on, on_dir)
    assert json.loads((on_dir / loop.RUN_CONFIG).read_text())["settings"]["collect_cuda_graphs"] is True
    with pytest.raises(SystemExit, match="collect_cuda_graphs"):
        loop.check_run_config(off, on_dir)
    loop.check_run_config(on, on_dir)


@pytest.mark.parametrize(("bad", "reason"), [
    (["--collect-cuda-graphs"], "cuda_graphs needs a CUDA device"),             # device cpu
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


@needs_cuda
def test_graphs_on_gives_the_same_shards_and_banks_its_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    off = _collect(tmp_path, "off", monkeypatch, device="cuda")
    on = _collect(tmp_path, "on", monkeypatch, device="cuda", cuda_graphs=True)
    assert _rows_digest(on) == _rows_digest(off)
    stats_on = json.loads((on / collect_mod.STATS).read_text())
    stats_off = json.loads((off / collect_mod.STATS).read_text())
    assert stats_off["cuda_graphs"] is False
    assert stats_on["cuda_graphs"] is True
    assert stats_on["cuda_graph_stats"]["replays"] > 0
    manifest = json.loads((on / "manifest.json").read_text())
    assert manifest["config"]["settings"]["cuda_graphs"] is True
    assert manifest["stats"]["cuda_graph_stats"] == stats_on["cuda_graph_stats"]
