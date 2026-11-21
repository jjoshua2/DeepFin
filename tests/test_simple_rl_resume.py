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

import numpy as np
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
            "--batch-size", "16", "--lr", "1e-3", "--holdout-frac", "0.25",
            "--optimizer-state", "continue", "--presentations", "2", "--no-eval", "--seed", "77"]


_VOLATILE = {"wall_seconds", "games_per_hour", "rows_per_second", "updates_per_second", "active_seconds",
             "paused_seconds", "pause_intervals"}


def _result(run_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for g in (1, 2):
        stats = json.loads((run_dir / f"gen_{g:04d}" / "collect" / "stats.json").read_text())
        out[f"collect{g}"] = {k: v for k, v in stats.items() if k not in _VOLATILE and k not in ("procs_resumed", "timing_independent")}
        # Per-process records: everything but their own clocks.
        # (and the attempt history, checked separately)
        out[f"collect{g}"]["procs"] = [{k: v for k, v in p.items() if k not in _VOLATILE and k != "resumes"}
                                       for p in stats["procs"]]
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
    # Each collect process writes under procs/<k>/ until the merge (K = 1 here).
    assert (resumed / "gen_0002" / "collect" / "procs" / "00" / "shards" / "shard_00000.zarr").exists()
    assert not (resumed / "gen_0002" / "collect" / "manifest.json").exists()
    monkeypatch.setattr(shard_mod, "save_local_shard_arrays", real_save)

    # Kill 3: gen-2 collect redone to its very end (stats, duplicate sketch and
    # opening record written) but killed before its manifest seals it.
    real_manifest = collect_mod.write_manifest

    def die_before_gen2_collect_manifest(stage_dir: Any, **kwargs: Any) -> Any:
        if "gen_0002" in str(stage_dir):
            raise Killed
        return real_manifest(stage_dir, **kwargs)

    monkeypatch.setattr(collect_mod, "write_manifest", die_before_gen2_collect_manifest)
    with pytest.raises(Killed):
        simple_rl_loop.main(argv)
    assert (resumed / "dup_sketch" / "through_gen_0002.npz").exists()
    monkeypatch.setattr(collect_mod, "write_manifest", real_manifest)

    assert simple_rl_loop.main(argv) == 0
    assert _result(resumed) == _result(straight)
    # Kill 2 struck before the first shard's commit record: nothing was kept, so the
    # process restarted on its own seed (hence the identical rows), and says so.
    gen2 = json.loads((resumed / "gen_0002" / "collect" / "stats.json").read_text())
    (resume,) = gen2["procs"][0]["resumes"]
    assert (resume["kept_games"], resume["discarded"]) == (0, ["shard_00000.zarr"])
    assert gen2["procs_game_resumed"] == {}
    # Openings cycle across generations: 8 games over a 6-line book reuse no
    # line before the 7th game, and the pass counter and earlier-play share say so.
    ids = [json.loads(r)["opening_id"] for g in (1, 2)
           for r in (resumed / f"gen_{g:04d}" / "collect" / "games.jsonl").read_text().splitlines()]
    starts = [ln for g in (1, 2) for ln in (resumed / f"gen_{g:04d}" / "collect" / "starts.txt").read_text().splitlines()]
    assert len(set(starts[:6])) == 6
    assert len(ids) == 8
    s2 = json.loads((resumed / "gen_0002" / "collect" / "stats.json").read_text())
    assert s2["book_pass"] == {"1": {"start": 1, "end": 2}}
    gen1_ids = set(ids[:4])
    assert s2["openings_played_in_earlier_generation_share"] == pytest.approx(
        sum(i in gen1_ids for i in ids[4:]) / 4)
    m2 = json.loads((resumed / "gen_0002" / "collect" / "manifest.json").read_text())
    assert m2["opening_cursor"]["books"]["1"] == {"pass": 2, "pos": 2, "lines": 6}
    assert m2["opening_cursor_before"]["books"]["1"]["pos"] == 4
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


def _presented(run_dir: Path, generations: int) -> dict[int, np.ndarray]:
    from chess_anti_engine.simple_rl.learn import _generation_meta, holdout_mask, presentation_counts

    counts, _ = presentation_counts(run_dir, before_generation=generations + 1)
    out = {}
    for g in range(1, generations + 1):
        meta = _generation_meta(run_dir, g)
        c = counts.get(g, np.zeros(len(meta["game_id"]), np.int64))
        held = holdout_mask(meta, 0.25)
        assert (c[held] == 0).all(), "a held-out row was trained on"
        out[g] = c[~held]
    return out


@pytest.mark.parametrize("presentations", [1, 2])
def test_every_row_is_presented_exactly_p_times_across_a_kill_and_resume(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, presentations: int) -> None:
    from chess_anti_engine.simple_rl import learn

    openings, model_config = _setup(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    argv = _argv(run_dir, openings, model_config)
    argv[argv.index("--generations") + 1] = "3"
    argv[argv.index("--presentations") + 1] = str(presentations)

    # Kill gen-2 training at the worst moment: after its presentation record
    # and model are on disk, before its manifest. The redo must not count it.
    real_write = learn.write_json_atomic
    fired = {"n": 0}

    def die_once(path: Any, payload: Any) -> Any:
        if str(path).endswith("gen_0002/train/metrics.json") and not fired["n"]:
            fired["n"] += 1
            raise Killed
        return real_write(path, payload)

    monkeypatch.setattr(learn, "write_json_atomic", die_once)
    with pytest.raises(Killed):
        simple_rl_loop.main(argv)
    assert (run_dir / "gen_0002" / "train" / learn.PRESENTED).exists()
    assert not (run_dir / "gen_0002" / "train" / "manifest.json").exists()
    assert simple_rl_loop.main(argv) == 0

    counts = _presented(run_dir, 3)
    for g in (1, 2, 3):
        lifetime_over = g <= 3 - presentations + 1
        expected = presentations if lifetime_over else presentations - 1
        assert counts[g].size > 0
        assert set(counts[g].tolist()) == {expected}, (g, np.unique(counts[g], return_counts=True))
    ledger = json.loads((run_dir / learn.LEDGER).read_text())
    for g in ("1", "2", "3"):
        h = ledger["generations"][g]
        if h["lifetime_complete"]:
            assert h["histogram"] == {str(presentations): h["trainable_rows"]}
    summary = json.loads((run_dir / "gen_0003" / "train" / "metrics.json").read_text())["summary"]
    assert summary["sampling"] == "scheduled_exact_once"
    gens = [1, 2, 3]
    windows = [json.loads((run_dir / f"gen_{g:04d}" / "train" / "metrics.json").read_text())["summary"][
        "window_generations"] for g in gens]
    assert windows == ([[1], [2], [3]] if presentations == 1 else [[1], [1, 2], [2, 3]])
    # Presentation ORDER: shuffled within each phase (never the load order),
    # and a generation's rows come in a different order in its second phase.
    order = {}
    for g in gens:
        with np.load(run_dir / f"gen_{g:04d}" / "train" / learn.PRESENTED) as z:
            pairs = np.stack([z["gen"], z["row"]], axis=1)
        load_order = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
        assert not np.array_equal(pairs, load_order), f"phase {g} presented rows in load order"
        order[g] = pairs
    if presentations == 2:
        for g in (1, 2):
            first = order[g][order[g][:, 0] == g, 1]
            second = order[g + 1][order[g + 1][:, 0] == g, 1]
            assert sorted(first.tolist()) == sorted(second.tolist())
            assert not np.array_equal(first, second), f"generation {g} replayed in the same order"


def test_more_than_two_presentations_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openings, model_config = _setup(tmp_path, monkeypatch)
    argv = _argv(tmp_path / "run", openings, model_config)
    argv[argv.index("--presentations") + 1] = "3"
    with pytest.raises(ValueError, match="cap is 2"):
        simple_rl_loop.main(argv)
    assert not (tmp_path / "run" / "gen_0000").exists()


def _stage_argv(argv: list[str], command: str, generation: int, *extra: str) -> list[str]:
    out = [command, *argv[1:]]
    i = out.index("--generations")
    del out[i:i + 2]
    return [*out, "--generation", str(generation), *extra]


def test_standalone_stages_are_held_to_the_run_config_and_load_only_sealed_models(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leads D2 / item 7: standalone subcommands go through check_run_config,
    skip a complete stage unless --redo, `device` is pinned, and a model whose
    sha256 does not match its manifest is refused rather than consumed."""
    openings, model_config = _setup(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    argv = _argv(run_dir, openings, model_config)
    argv[argv.index("--generations") + 1] = "1"
    assert simple_rl_loop.main(argv) == 0
    model = run_dir / "gen_0001" / "train" / "model.pt"
    before = model.read_bytes()

    assert simple_rl_loop.main(_stage_argv(argv, "train", 1)) == 0  # complete: skipped
    assert model.read_bytes() == before
    changed = _stage_argv(argv, "train", 1, "--redo")
    changed[changed.index("--lr") + 1] = "5e-4"
    with pytest.raises(SystemExit, match="run settings differ"):
        simple_rl_loop.main(changed)
    with pytest.raises(SystemExit, match="run settings differ"):
        simple_rl_loop.main([*argv, "--device", "meta"])  # device is pinned, not exempt

    model.write_bytes(before[:-1] + bytes([before[-1] ^ 1]))  # a stray overwrite of sealed weights
    with pytest.raises(SystemExit, match="does not match the sha256"):
        simple_rl_loop.main(_stage_argv(argv, "collect", 2))
    model.write_bytes(before)
    assert simple_rl_loop.main(_stage_argv(argv, "train", 1, "--redo")) == 0  # same settings: allowed, resealed
    manifest = json.loads((run_dir / "gen_0001" / "train" / "manifest.json").read_text())
    assert manifest["outputs"]["model"]["sha256"]
    assert "probe" in manifest["inputs"]
    assert manifest["config"]["realised"]["amp"] is False


def test_redo_of_an_earlier_stage_refuses_or_unseals_downstream_and_rebuilds_lineage(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reviewer N1: `train --generation 1 --redo` while generation 2 is sealed
    used to be accepted, leaving gen 2 played by a model that no longer exists."""
    from chess_anti_engine.simple_rl.common import sha256_tree

    openings, model_config = _setup(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    argv = _argv(run_dir, openings, model_config)
    assert simple_rl_loop.main(argv) == 0
    redo = _stage_argv(argv, "train", 1, "--redo", "--allow-config-change")
    redo[redo.index("--lr") + 1] = "5e-4"
    with pytest.raises(SystemExit, match="later stages are sealed"):
        simple_rl_loop.main(redo)
    assert (run_dir / "gen_0002" / "collect" / "manifest.json").exists()

    sketch2 = run_dir / "dup_sketch" / "through_gen_0002.npz"
    assert sketch2.exists()
    assert simple_rl_loop.main([*redo, "--redo-downstream"]) == 0
    superseded = list((run_dir / "superseded").glob("*/gen_0002/collect/manifest.json"))
    assert len(superseded) == 1  # moved, rows kept
    assert not (run_dir / "gen_0002").exists() or not any((run_dir / "gen_0002").glob("*/manifest.json"))
    assert not sketch2.exists()  # side files from the first unsealed collect on are rewound
    assert not (run_dir / "openings_seen" / "through_gen_0002.npy").exists()

    rerun = list(argv)
    rerun[rerun.index("--lr") + 1] = "5e-4"
    assert simple_rl_loop.main(rerun) == 0
    gen1 = sha256_tree(run_dir / "gen_0001" / "train" / "model.pt")
    played = json.loads((run_dir / "gen_0002" / "collect" / "manifest.json").read_text())["inputs"]["checkpoint"]
    assert played["sha256"] == gen1
    # the opening cursor rewound with the collect manifests: gen 2 continues gen 1's cursor again
    m2 = json.loads((run_dir / "gen_0002" / "collect" / "manifest.json").read_text())
    m1 = json.loads((run_dir / "gen_0001" / "collect" / "manifest.json").read_text())
    assert m2["opening_cursor_before"] == m1["opening_cursor"]


def test_a_sealed_stage_whose_upstream_changed_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    openings, model_config = _setup(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    argv = _argv(run_dir, openings, model_config)
    assert simple_rl_loop.main(argv) == 0
    assert simple_rl_loop.main(argv) == 0  # unchanged lineage: every stage is skipped
    manifest = run_dir / "gen_0001" / "collect" / "manifest.json"
    payload = json.loads(manifest.read_text())
    payload["tampered"] = True  # what any redo of gen-1 collect would do to its sealed record
    manifest.write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="no longer match their upstream"):
        simple_rl_loop.main(argv)
    with pytest.raises(SystemExit, match="no longer match their upstream"):
        simple_rl_loop.main(_stage_argv(argv, "train", 2))  # standalone, complete-and-skipped


def test_a_stale_diagnostics_file_is_rebuilt_not_trusted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.simple_rl.dup_sketch import DupSketch

    openings, model_config = _setup(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    argv = _argv(run_dir, openings, model_config)
    assert simple_rl_loop.main(argv) == 0
    stats_path = run_dir / "gen_0002" / "collect" / "stats.json"
    stats_before = json.loads(stats_path.read_text())
    before = stats_before["repeat_rate_vs_previous_estimate"]
    DupSketch(budget=10).save(run_dir / "dup_sketch" / "through_gen_0001.npz")  # stale / foreign file
    np.save(run_dir / "openings_seen" / "through_gen_0001.npy", np.asarray([], dtype=str))
    assert simple_rl_loop.main(_stage_argv(argv, "collect", 2, "--redo", "--redo-downstream")) == 0
    after = json.loads(stats_path.read_text())
    assert after["repeat_rate_vs_previous_estimate"] == before
    assert (after["openings_played_in_earlier_generation_share"]
            == stats_before["openings_played_in_earlier_generation_share"] > 0)


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize(("command", "generation", "bad", "reason"), [
    ("collect", 1, ["--device", "cuda", "--eval-max-batch", "1"], "leaf rows one search call"),  # settings only
    ("train", 1, ["--presentations", "3"], "presentations must be 1 or 2"),
    ("evaluate", 1, ["--eval-opening-plies", "1"], "longest EVAL line"),
])
def test_a_refused_standalone_stage_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    command: str, generation: int, bad: list[str],
                                                    reason: str) -> None:
    """A standalone stage validates its settings BEFORE pinning run_config.json,
    unsealing downstream stages to superseded/ or unlinking its own manifest:
    refused, the run directory is byte-identical, with or without --redo-downstream."""
    openings, model_config = _setup(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    argv = _argv(run_dir, openings, model_config)
    assert simple_rl_loop.main(argv) == 0
    (openings / "summary.json").write_text(json.dumps({"eval_max_plies": 6}))
    before = _tree_snapshot(run_dir)
    assert "gen_0002/train/manifest.json" in before
    for extra in (["--redo"], ["--redo", "--redo-downstream"]):
        with pytest.raises((SystemExit, ValueError), match=reason):
            simple_rl_loop.main(_stage_argv(argv, command, generation, *extra, "--allow-config-change", *bad))
        assert _tree_snapshot(run_dir) == before, extra
    assert not (run_dir / "superseded").exists()


def test_a_refused_standalone_stage_on_a_new_run_writes_nothing(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    openings, model_config = _setup(tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    argv = _stage_argv(_argv(run_dir, openings, model_config), "collect", 1, "--device", "cuda",
                       "--eval-max-batch", "1")
    with pytest.raises(SystemExit, match="leaf rows one search call"):
        simple_rl_loop.main(argv)
    assert not run_dir.exists() or _tree_snapshot(run_dir) == {}
