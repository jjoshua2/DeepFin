"""simple_rl ``run`` resumes at stage boundaries with an identical result.

Run A goes straight through two generations. Run B is killed between gen-1
collect and train, then killed again in the middle of gen-2 collect (after a
shard is on disk), then resumed to the end. Both must produce identical
generation statistics, training losses and final weights.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from chess_anti_engine.replay import shard as shard_mod
from chess_anti_engine.simple_rl import collect as collect_mod
from scripts import simple_rl_loop
from tests.simple_rl_helpers import TINY_ARCH, QuietStockfish

_LINES = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | e2e4 e7e5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | d2d4 d7d5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | c2c4 e7e5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | g1f3 d7d5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | e2e4 c7c5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | b1c3 g8f6",
]


class Killed(Exception):
    pass


def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(collect_mod, "_open_stockfish", lambda _s: QuietStockfish([0.0, 1.0, 0.0]))
    openings = tmp_path / "openings"
    openings.mkdir(exist_ok=True)
    (openings / "train.jsonl").write_text("".join(
        json.dumps({"id": f"o{i}", "book": 1, "line": ln}) + "\n" for i, ln in enumerate(_LINES)))
    model_config = tmp_path / "tiny.json"
    model_config.write_text(json.dumps(TINY_ARCH))
    return openings, model_config


def _argv(run_dir: Path, openings: Path, model_config: Path) -> list[str]:
    return ["run", "--run-dir", str(run_dir), "--openings-dir", str(openings), "--random-init",
            "--model-config", str(model_config), "--generations", "2", "--games", "4", "--concurrency", "4",
            "--sims", "8", "--max-plies", "30", "--rows-per-shard", "25", "--book-mix", "1.0",
            "--batch-size", "16", "--passes", "1", "--lr", "1e-3", "--holdout-frac", "0.25",
            "--optimizer-state", "continue", "--window", "2", "--no-eval", "--seed", "77"]


_VOLATILE = {"wall_seconds", "games_per_hour", "rows_per_second", "updates_per_second"}


def _result(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for g in (1, 2):
        stats = json.loads((run_dir / f"gen_{g:04d}" / "collect" / "stats.json").read_text())
        out[f"collect{g}"] = {k: v for k, v in stats.items() if k not in _VOLATILE}
        summary = json.loads((run_dir / f"gen_{g:04d}" / "train" / "metrics.json").read_text())["summary"]
        out[f"train{g}"] = {k: summary[k] for k in ("losses", "heldout_before", "heldout_after",
                                                    "probe_vs_previous", "window_rows", "end_step")}
    return out


def _weights(run_dir: Path) -> dict[str, torch.Tensor]:
    return torch.load(run_dir / "gen_0002" / "train" / "model.pt", map_location="cpu", weights_only=True)["model"]


def test_resume_after_kills_is_identical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openings, model_config = _setup(tmp_path, monkeypatch)
    straight = tmp_path / "straight"
    assert simple_rl_loop.main(_argv(straight, openings, model_config)) == 0

    resumed = tmp_path / "resumed"
    argv = _argv(resumed, openings, model_config)

    # Kill 1: after gen-1 collect has finished, before its training starts.
    real_train = simple_rl_loop.do_train

    def die_in_train(*_a: Any, **_k: Any) -> None:
        raise Killed

    monkeypatch.setattr(simple_rl_loop, "do_train", die_in_train)
    with pytest.raises(Killed):
        simple_rl_loop.main(argv)
    assert (resumed / "gen_0001" / "collect" / "manifest.json").exists()
    assert not (resumed / "gen_0001" / "train" / "manifest.json").exists()
    monkeypatch.setattr(simple_rl_loop, "do_train", real_train)

    # Kill 2: in the middle of gen-2 collect, after its first shard is written.
    real_save = shard_mod.save_local_shard_arrays
    calls = {"n": 0}

    def die_after_first_gen2_shard(path: Any, **kwargs: Any) -> Any:
        result = real_save(path, **kwargs)
        if "gen_0002" in str(path):
            calls["n"] += 1
            raise Killed
        return result

    monkeypatch.setattr(shard_mod, "save_local_shard_arrays", die_after_first_gen2_shard)
    with pytest.raises(Killed):
        simple_rl_loop.main(argv)
    assert calls["n"] == 1
    assert (resumed / "gen_0002" / "collect" / "shards" / "shard_00000.zarr").exists()
    assert not (resumed / "gen_0002" / "collect" / "manifest.json").exists()
    monkeypatch.setattr(shard_mod, "save_local_shard_arrays", real_save)

    assert simple_rl_loop.main(argv) == 0
    assert _result(resumed) == _result(straight)
    a, b = _weights(straight), _weights(resumed)
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_stop_file_stops_at_the_next_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openings, model_config = _setup(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "STOP").write_text("")
    assert simple_rl_loop.main(_argv(run_dir, openings, model_config)) == simple_rl_loop.EXIT_BLOCKED
    assert not (run_dir / "gen_0000").exists()
    marker = tmp_path / "gpu_window_marker"
    marker.write_text("")
    (run_dir / "STOP").unlink()
    argv = [*_argv(run_dir, openings, model_config), "--pause-marker", str(marker)]
    assert simple_rl_loop.main(argv) == simple_rl_loop.EXIT_BLOCKED


def test_a_changed_setting_is_refused_on_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openings, model_config = _setup(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    argv = _argv(run_dir, openings, model_config)
    argv[argv.index("--generations") + 1] = "0"  # initialise only
    assert simple_rl_loop.main(argv) == 0
    changed = list(argv)
    changed[changed.index("--lr") + 1] = "2e-3"
    with pytest.raises(SystemExit, match="run settings differ"):
        simple_rl_loop.main(changed)
    assert simple_rl_loop.main([*changed, "--allow-config-change"]) == 0
    record = json.loads((run_dir / "run_config.json").read_text())
    assert record["changes"][0]["diff"]["lr"] == {"was": 1e-3, "now": 2e-3}
