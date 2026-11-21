"""simple_rl collect with K processes (``--collect-procs``).

K processes each play a disjoint slice of the generation's openings with their
own seed and one manifest seals the merged collect. Covered here: the split and
seeds, a deterministic merged digest for a fixed (seed, K), resume after two
kills keeping finished processes, children (and the engines they start) dying
with the driver, the pause marker in every process, K pinned in run_config and
matched by arms, the VRAM refusal and the collect-level evaluator record.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.simple_rl import collect as collect_mod
from chess_anti_engine.simple_rl.collect import CollectSettings
from scripts import simple_rl_loop
from scripts.simple_rl_collect_gates import content_digest
from tests.simple_rl_helpers import TINY_ARCH, quiet_stockfish, save_tiny_checkpoint, scripted_stockfish

_LINES = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | e2e4 e7e5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | d2d4 d7d5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | c2c4 e7e5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | g1f3 d7d5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | e2e4 c7c5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | b1c3 g8f6",
]
_REPO = Path(__file__).resolve().parents[1]


def _openings(tmp_path: Path) -> Path:
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps({"id": f"o{i}", "book": 1, "line": ln}) + "\n" for i, ln in enumerate(_LINES)))
    return path


def _settings(procs: int, games: int = 4) -> CollectSettings:
    return CollectSettings(games=games, concurrency=2, sims=8, max_plies=24, rows_per_shard=25, book_mix=(1.0,),
                           collect_procs=procs, seed=77)


def _collect(tmp_path: Path, run: str, procs: int, settings: CollectSettings | None = None,
             **kw: Any) -> tuple[dict[str, Any], Path]:
    ckpt = tmp_path / "gen0.pt"
    if not ckpt.exists():
        save_tiny_checkpoint(ckpt)
    gen = tmp_path / run / "gen_0001" / "collect"
    stats = collect_mod.collect_generation(gen_dir=gen, generation=1, checkpoint=ckpt,
                                           train_jsonl=_openings(tmp_path),
                                           settings=settings or _settings(procs), **kw)
    return stats, gen


def _games(gen: Path) -> list[dict[str, Any]]:
    return [json.loads(r) for r in (gen / "games.jsonl").read_text().splitlines() if r.strip()]


def test_split_is_disjoint_and_k1_keeps_the_one_process_seed() -> None:
    rows = [{"id": f"o{i}"} for i in range(10)]
    parts = [collect_mod.proc_openings(rows, k, 3) for k in range(3)]
    assert sorted(r["id"] for p in parts for r in p) == sorted(r["id"] for r in rows)
    assert all(not ({r["id"] for r in a} & {r["id"] for r in b}) for i, a in enumerate(parts) for b in parts[i + 1:])
    one = _settings(1)
    assert collect_mod.proc_seed(one, 5, 0) == collect_mod.seed_for(77, "collect", 5)
    three = _settings(3, games=6)
    seeds = {collect_mod.proc_seed(three, 5, k) for k in range(3)}
    assert len(seeds) == 3
    assert collect_mod.seed_for(77, "collect", 5) not in seeds


@pytest.mark.parametrize("procs", [0, 5])
def test_procs_outside_one_to_games_are_refused(procs: int) -> None:
    with pytest.raises(ValueError, match="--collect-procs"):
        collect_mod.validate_settings(_settings(procs))


def test_k2_collect_merges_deterministically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    a_stats, a = _collect(tmp_path, "a", 2)
    _, b = _collect(tmp_path, "b", 2)
    da, db = content_digest(a), content_digest(b)
    assert (da["content_sha256"], da["games_sha256"]) == (db["content_sha256"], db["games_sha256"])
    manifest = json.loads((a / "manifest.json").read_text())
    assert manifest["collect_procs"] == 2
    assert manifest["config"]["settings"]["collect_procs"] == 2
    assert manifest["seeds"]["collect_procs"] == [collect_mod.proc_seed(_settings(2), 1, k) for k in range(2)]
    assert [p["proc"] for p in a_stats["procs"]] == [0, 1]
    assert all(p["games"] == 2 and p["evaluator_mode"] == "eager_cpu" for p in a_stats["procs"])
    assert a_stats["buffer_full_flushes"] == 0
    assert a_stats["evaluator_mode"] == "eager_cpu"
    starts = (a / "starts.txt").read_text().splitlines()
    played = sorted(g["opening_id"] for g in _games(a))
    assert len(starts) == 4
    assert played == sorted(f"o{_LINES.index(ln)}" for ln in starts)
    assert len(set(played)) == 4  # no opening played twice across the processes
    assert not (a / "procs").exists()
    assert not (a / "plan.json").exists()
    # The cursor advances by --games exactly as for one process.
    assert manifest["opening_cursor"]["books"]["1"]["pos"] == 4
    # K = 1 is the same opening draw, a different RNG stream per game.
    _, one = _collect(tmp_path, "one", 1)
    assert (one / "starts.txt").read_text() == (a / "starts.txt").read_text()
    assert content_digest(one)["content_sha256"] != da["content_sha256"]


def test_a_killed_k2_collect_resumes_keeping_finished_processes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    _, straight = _collect(tmp_path, "straight", 2)

    monkeypatch.setattr(collect_mod, "_open_stockfish", scripted_stockfish)
    gen = tmp_path / "resumed" / "gen_0001" / "collect"
    done0 = gen / "procs" / "00" / "done.json"
    # Kill 1: process 1 dies once process 0 has finished.
    monkeypatch.setenv("SIMPLE_RL_TEST_SF_SCRIPT", json.dumps({"1": {"after": str(done0), "do": "die"}}))
    with pytest.raises(RuntimeError, match="collect process"):
        _collect(tmp_path, "resumed", 2)
    assert done0.is_file()
    assert not (gen / "procs" / "01" / "done.json").exists()
    kept = done0.read_text()
    # Kill 2: process 1 dies again at once; process 0 is not replayed.
    monkeypatch.setenv("SIMPLE_RL_TEST_SF_SCRIPT", json.dumps({"1": {"do": "die"}}))
    with pytest.raises(RuntimeError, match=r"\(1, 1\)"):
        _collect(tmp_path, "resumed", 2)
    assert done0.read_text() == kept
    # Kill 3: both processes finished, the parent dies before the manifest seals it.
    monkeypatch.delenv("SIMPLE_RL_TEST_SF_SCRIPT")
    real_manifest = collect_mod.write_manifest

    def die(*_a: Any, **_k: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(collect_mod, "write_manifest", die)
    with pytest.raises(KeyboardInterrupt):
        _collect(tmp_path, "resumed", 2)
    # That attempt kept process 0 and played only process 1: its stats (written
    # before the manifest) rate process 1's games over its own active time.
    partial = json.loads((gen / "stats.json").read_text())
    assert partial["procs_resumed"] == [0]
    fresh_games = partial["procs"][1]["games_this_attempt"]  # < 2 if kill 1 left committed games
    assert 0 < fresh_games <= 2
    assert partial["games_per_hour"] == pytest.approx(3600.0 * fresh_games / partial["active_seconds"])
    monkeypatch.setattr(collect_mod, "write_manifest", real_manifest)
    stats, _ = _collect(tmp_path, "resumed", 2)
    assert stats["procs_resumed"] == [0, 1]
    # no process played in this attempt: no rate to report (not games / merge time)
    assert stats["games_per_hour"] is None
    assert stats["rows_per_second"] is None
    assert stats["timing_independent"] is False
    if stats["procs_game_resumed"]:
        # Process 1 had committed games when it died: they were kept and the rest
        # replayed on a fresh seed (test_simple_rl_collect_game_resume), so only the
        # gates and the openings can match an uninterrupted collect.
        assert sorted(g["opening_id"] for g in _games(gen)) == sorted(g["opening_id"] for g in _games(straight))
    else:  # nothing kept: a plain restart on its own seed reproduces the rows
        assert content_digest(gen)["content_sha256"] == content_digest(straight)["content_sha256"]
        assert content_digest(gen)["games_sha256"] == content_digest(straight)["games_sha256"]


def test_a_changed_plan_redoes_the_interrupted_collect_whole(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", scripted_stockfish)
    gen = tmp_path / "r" / "gen_0001" / "collect"
    monkeypatch.setenv("SIMPLE_RL_TEST_SF_SCRIPT", json.dumps(
        {"1": {"after": str(gen / "procs" / "00" / "done.json"), "do": "die"}}))
    with pytest.raises(RuntimeError):
        _collect(tmp_path, "r", 2)
    monkeypatch.delenv("SIMPLE_RL_TEST_SF_SCRIPT")
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    stats, _ = _collect(tmp_path, "r", 1)  # K changed: nothing of the K = 2 attempt is kept
    assert stats["procs_resumed"] == []
    assert stats["collect_procs"] == 1


def _pause_once(gen: Path, procs: int, run_dir: Path, hold_s: float, log: dict[str, float]) -> threading.Thread:
    """Create <run>/PAUSE after the first finished game, wait until every process
    acknowledges (procs/<k>/PAUSED), hold ``hold_s``, then remove it."""
    def go() -> None:
        first = gen / "procs" / "00" / "games.jsonl"
        while not (first.exists() and first.read_text().strip()):
            time.sleep(0.02)
        (run_dir / "PAUSE").write_text("")
        acks = [gen / "procs" / f"{k:02d}" / "PAUSED" for k in range(procs)]
        while not all(a.exists() for a in acks):
            time.sleep(0.02)
        log["held"] = time.time()
        time.sleep(hold_s)
        log["released"] = time.time()
        (run_dir / "PAUSE").unlink()

    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


@pytest.mark.parametrize("procs", [1, 2])
def test_a_pause_holds_every_process_and_resumes_to_identical_rows(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, procs: int) -> None:
    from chess_anti_engine.selfplay import network_turn

    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    # Two waves or more per process, so games remain after the first ones end.
    settings = _settings(procs, games=8)
    _, plain = _collect(tmp_path, "plain", procs, settings)
    calls: list[float] = []
    real = network_turn._run_gumbel_root_many_c

    def counting(*a: Any, **k: Any) -> Any:
        calls.append(time.time())
        return real(*a, **k)

    monkeypatch.setattr(network_turn, "_run_gumbel_root_many_c", counting)  # seen in-process (K = 1)
    log: dict[str, float] = {}
    gen = tmp_path / "paused" / "gen_0001" / "collect"
    pauser = _pause_once(gen, procs, tmp_path / "paused", 2.0, log)
    stats, _ = _collect(tmp_path, "paused", procs, settings)
    pauser.join(5)
    assert content_digest(gen)["content_sha256"] == content_digest(plain)["content_sha256"]
    assert content_digest(gen)["games_sha256"] == content_digest(plain)["games_sha256"]
    assert stats["paused_seconds"] >= 2.0
    # the processes pause together: the union of their intervals, not the sum
    assert stats["paused_seconds"] < 2.0 + 1.5
    assert stats["timing_independent"] is False
    assert all(len(p["pause_intervals"]) == 1 for p in stats["procs"])
    assert stats["active_seconds"] <= stats["wall_seconds"] - 2.0
    if procs == 1:
        assert calls, "the counting search wrapper was not on the collect's path"
        assert not [c for c in calls if log["held"] <= c <= log["released"]], "a search ran during the pause"


def test_a_marker_present_at_start_holds_before_any_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    marker = tmp_path / "gpu_window"
    marker.write_text("")

    def release() -> None:
        ack = tmp_path / "p" / "gen_0001" / "collect" / "procs" / "00" / "PAUSED"
        while not ack.exists():
            time.sleep(0.02)
        time.sleep(1.0)
        marker.unlink()

    from chess_anti_engine.uci import model_loader

    loads: list[float] = []
    real_load = model_loader.load_model_from_checkpoint

    def recording_load(*a: Any, **k: Any) -> Any:
        loads.append(time.time())
        return real_load(*a, **k)

    monkeypatch.setattr(model_loader, "load_model_from_checkpoint", recording_load)
    threading.Thread(target=release, daemon=True).start()
    stats, _ = _collect(tmp_path, "p", 1, pause_markers=(marker,))
    assert stats["paused_seconds"] >= 1.0
    assert stats["games"] == 4
    (proc,) = stats["procs"]
    (start, end), = proc["pause_intervals"]
    # the pre-load hold is inside the process clock, so its rate is not inflated
    assert proc["paused_seconds"] <= proc["wall_seconds"]
    assert start >= proc["started_at"]
    assert proc["games_per_hour"] < 3600.0 * 4 / 0.5
    assert loads, "the recording loader was not on the collect's path"
    assert min(loads) >= end, "the model was loaded (GPU work) before the pause ended"


_LAUNCHER = textwrap.dedent("""
    import sys
    from pathlib import Path
    sys.path.insert(0, {repo!r})
    from chess_anti_engine.simple_rl import collect as collect_mod
    from tests.simple_rl_helpers import save_tiny_checkpoint, scripted_stockfish


    def main() -> None:
        collect_mod._open_stockfish = scripted_stockfish
        tmp = Path({tmp!r})
        save_tiny_checkpoint(tmp / "gen0.pt")
        settings = collect_mod.CollectSettings(games=4, concurrency=2, sims=8, max_plies=24, rows_per_shard=25,
                                               book_mix=(1.0,), collect_procs=2, seed=77)
        collect_mod.collect_generation(gen_dir=tmp / "run" / "gen_0001" / "collect", generation=1,
                                       checkpoint=tmp / "gen0.pt", train_jsonl=tmp / "train.jsonl",
                                       settings=settings)


    if __name__ == "__main__":  # spawned children re-import this file as __mp_main__
        main()
""")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = Path(f"/proc/{pid}/stat")
    return stat.exists() and stat.read_text().split(")")[-1].split()[0] != "Z"


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="parent-death signal is Linux-only")
@pytest.mark.parametrize("how", ["SIGKILL", "SIGTERM", "child_fails"])
def test_no_collect_process_or_engine_outlives_the_driver(tmp_path: Path, how: str) -> None:
    _openings(tmp_path)
    pid_dir = tmp_path / "pids"
    pid_dir.mkdir()
    hang = {"do": "hang", "pid_dir": str(pid_dir)}
    script = {"0": hang, "1": ({"do": "die", "after": str(pid_dir / "go")} if how == "child_fails" else hang)}
    launcher = tmp_path / "launch.py"
    launcher.write_text(_LAUNCHER.format(repo=str(_REPO), tmp=str(tmp_path)))
    env = {**os.environ, "SIMPLE_RL_TEST_SF_SCRIPT": json.dumps(script), "PYTHONPATH": str(_REPO)}
    driver = subprocess.Popen([sys.executable, str(launcher)], env=env, cwd=str(_REPO))
    want = 1 if how == "child_fails" else 2
    deadline = time.monotonic() + 240
    while len(list(pid_dir.glob("sentinel_*"))) < want:
        assert driver.poll() is None, "the driver exited before its processes started"
        assert time.monotonic() < deadline, "collect processes never started their engines"
        time.sleep(0.2)
    pids = [int(p.read_text()) for p in pid_dir.iterdir() if p.name.startswith(("collect_", "sentinel_"))]
    assert all(_alive(p) for p in pids)
    if how == "child_fails":
        (pid_dir / "go").write_text("")
        assert driver.wait(timeout=120) != 0
    else:
        driver.send_signal(getattr(signal, how))
        driver.wait(timeout=60)
    deadline = time.monotonic() + 30
    while any(_alive(p) for p in pids) and time.monotonic() < deadline:
        time.sleep(0.2)
    survivors = [p for p in pids if _alive(p)]
    for p in survivors:  # never leave them behind, even when failing
        os.kill(p, signal.SIGKILL)
    assert not survivors, f"processes outlived the driver ({how}): {survivors}"


# --------------------------------------------------------------------------
# driver: default, pinning, arms; VRAM refusal; combined evaluator record
# --------------------------------------------------------------------------

def _driver_args(run_dir: Path, *extra: str, device: str = "cpu") -> Any:
    a = simple_rl_loop.parser().parse_args(["collect", "--run-dir", str(run_dir), "--openings-dir", "x",
                                            "--generation", "1", "--device", device, *extra])
    return a


def test_default_shape_follows_sims_and_a_pinned_value_wins(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert simple_rl_loop.resolve_collect_shape(_driver_args(run_dir)) == (1, 64)
    assert simple_rl_loop.resolve_collect_shape(_driver_args(run_dir, device="cuda")) == (3, 128)
    assert simple_rl_loop.resolve_collect_shape(_driver_args(run_dir, "--sims", "16", device="cuda")) == (8, 64)
    assert simple_rl_loop.resolve_collect_shape(
        _driver_args(run_dir, "--sims", "16", "--concurrency", "32", device="cuda")) == (8, 32)
    assert simple_rl_loop.resolve_collect_procs(_driver_args(run_dir, "--collect-procs", "2", device="cuda")) == 2
    run_dir.mkdir()
    # A run pinned before the flag existed resumes at the one process it ran with.
    (run_dir / simple_rl_loop.RUN_CONFIG).write_text(json.dumps({"settings": {"concurrency": 64}, "changes": []}))
    assert simple_rl_loop.resolve_collect_shape(_driver_args(run_dir, "--sims", "16", device="cuda")) == (1, 64)
    (run_dir / simple_rl_loop.RUN_CONFIG).write_text(json.dumps({"settings": {"collect_procs": 2}, "changes": []}))
    assert simple_rl_loop.resolve_collect_procs(_driver_args(run_dir, device="cuda")) == 2
    # An arm without the flag takes its source's value.
    source = tmp_path / "source"
    source.mkdir()
    (source / simple_rl_loop.RUN_CONFIG).write_text(json.dumps({"settings": {}, "changes": []}))
    arm = _driver_args(tmp_path / "arm", "--collect-from", str(source), device="cuda")
    assert simple_rl_loop.resolve_collect_procs(arm) == 1


def test_k_is_pinned_and_a_change_is_refused_on_resume(tmp_path: Path) -> None:
    openings = tmp_path / "openings"
    openings.mkdir()
    _openings(openings)
    model_config = tmp_path / "tiny.json"
    model_config.write_text(json.dumps(TINY_ARCH))
    run_dir = tmp_path / "run"
    argv = ["run", "--run-dir", str(run_dir), "--openings-dir", str(openings), "--random-init",
            "--model-config", str(model_config), "--generations", "0", "--games", "4", "--book-mix", "1.0",
            "--no-eval", "--seed", "77"]
    assert simple_rl_loop.main(argv) == 0
    assert json.loads((run_dir / "run_config.json").read_text())["settings"]["collect_procs"] == 1
    with pytest.raises(SystemExit, match="collect_procs"):
        simple_rl_loop.main([*argv, "--collect-procs", "2"])
    assert simple_rl_loop.main([*argv, "--collect-procs", "2", "--allow-config-change"]) == 0
    assert json.loads((run_dir / "run_config.json").read_text())["settings"]["collect_procs"] == 2
    assert simple_rl_loop.main(argv) == 0  # no flag: the pinned 2


def test_an_arm_must_match_the_source_k() -> None:
    from chess_anti_engine.simple_rl.reference import settings_diff

    old_source = {"config": {"settings": {k: v for k, v in collect_mod.asdict(_settings(1)).items()
                                          if k != "collect_procs"}}}
    old_source["config"]["settings"]["book_mix"] = [1.0]
    assert settings_diff(old_source, _settings(1)) == {}
    assert settings_diff(old_source, _settings(2)) == {"collect_procs": {"source": 1, "arm": 2}}


def test_vram_refusal_uses_free_memory_and_the_usable_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    gib = 2**30
    monkeypatch.setattr(collect_mod, "_gpu_memory", lambda *_a: (10 * gib, 32 * gib))
    settings = replace(_settings(3, games=512), concurrency=128, sims=64, device="cuda")
    with pytest.raises(ValueError, match=r"--collect-procs 3 x --concurrency 128 .* pass --collect-procs 1 explicitly"):
        collect_mod.check_vram(settings)
    monkeypatch.setattr(collect_mod, "_gpu_memory", lambda *_a: (31 * gib, 32 * gib))
    record = collect_mod.check_vram(settings)  # measured 22.1 GB: fits
    assert record is not None
    assert record["usable_gib"] == pytest.approx(0.83 * 32)
    # 16 sims: the default 8 x 64 (measured 22.9 GB) fits; 8 x 96 (31.0 GB, ran
    # 22% slower) is past the usable limit although the card has 31 GiB free.
    assert collect_mod.check_vram(replace(settings, collect_procs=8, concurrency=64, sims=16)) is not None
    with pytest.raises(ValueError, match="usable limit"):
        collect_mod.check_vram(replace(settings, collect_procs=8, concurrency=96, sims=16))
    # One process at 64 sims: 1 x 256 (measured 12.1 GB) fits; 1 x 512 peaked at
    # 29.9 GB, past the usable limit, which the linear segment alone under-predicted.
    one = replace(settings, collect_procs=1)
    assert collect_mod.check_vram(replace(one, concurrency=256)) is not None
    with pytest.raises(ValueError, match="usable limit"):
        collect_mod.check_vram(replace(one, concurrency=512))
    # A shape no process count fits is not told to use --collect-procs 1.
    with pytest.raises(ValueError, match="no process count fits at --concurrency 1024: lower") as err:
        collect_mod.check_vram(replace(one, concurrency=1024, games=2048))
    assert "pass --collect-procs" not in str(err.value)


def test_the_driver_of_k_processes_reads_gpu_memory_without_a_cuda_context(
        monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    import torch

    def no_context(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("torch.cuda.mem_get_info would create a CUDA context in the driver")

    calls: list[list[str]] = []

    def smi(cmd: list[str], **_k: Any) -> Any:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="20480, 32607\n", stderr="")

    monkeypatch.setattr(torch.cuda, "mem_get_info", no_context)
    monkeypatch.setattr(subprocess, "run", smi)
    settings = replace(_settings(3, games=512), device="cuda:0")
    assert collect_mod._gpu_memory(settings) == (20480 * 2**20, 32607 * 2**20)
    assert calls
    assert calls[0][0] == "nvidia-smi"
    assert "--id=0" in calls[0]
    # K = 1 plays in the driver, which needs the context anyway
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a: (5, 9))
    assert collect_mod._gpu_memory(replace(settings, collect_procs=1)) == (5, 9)


def test_default_shape_table_by_sims() -> None:
    assert collect_mod.default_collect_shape("cuda", 16) == (8, 64)
    assert collect_mod.default_collect_shape("cuda", 8) == (8, 64)
    assert collect_mod.default_collect_shape("cuda", 32) == (3, 128)
    assert collect_mod.default_collect_shape("cuda", 64) == (3, 128)
    assert collect_mod.default_collect_shape("cpu", 16) == (1, 64)


def test_combined_evaluator_record_fails_closed_on_disagreement() -> None:
    graph_stats = {"replays": 3, "eager_over_max_rows": 1, "max_rows": 512, "graphs": 2}
    base = {"cuda_graphs": True, "evaluator_mode": "cuda_graphs", "eval_max_batch_realised": 4096,
            "cuda_graph_stats": graph_stats}
    same = collect_mod._combined_evaluator_record([base, dict(base)])
    assert same["eval_max_batch_realised"] == 4096
    assert same["evaluator_mode"] == "cuda_graphs"
    # counters sum; the per-process captured-shape count does not
    assert same["cuda_graph_stats"] == {"replays": 6, "eager_over_max_rows": 2, "max_rows": 512, "graphs": 2}
    other = {**base, "cuda_graph_stats": {**graph_stats, "graphs": 3}}
    assert collect_mod._combined_evaluator_record([base, other])["cuda_graph_stats"]["graphs"] == [2, 3]
    odd = collect_mod._combined_evaluator_record([base, {**base, "eval_max_batch_realised": 2048,
                                                         "evaluator_mode": "eager", "cuda_graphs": False}])
    assert odd["eval_max_batch_realised"] is None
    assert odd["evaluator_mode"] == "mixed"
    assert not odd["cuda_graphs"]


def test_a_kept_process_is_reverified_by_its_games_digest(tmp_path: Path) -> None:
    spec = collect_mod.ProcSpec(
        proc=0, gen_dir=str(tmp_path), generation=1, checkpoint="", starts_path="", settings=_settings(1),
        openings=(), seed=0, sf_factory=None, torch_threads=1, run_dir=str(tmp_path), pause_markers=(),
        parent_pid=os.getpid())
    (spec.proc_dir / "shards").mkdir(parents=True)
    games = spec.proc_dir / "games.jsonl"
    games.write_text('{"game_id": 1}\n')
    (spec.proc_dir / "done.json").write_text(json.dumps(
        {"proc": 0, "shards": 0, "games_sha256": collect_mod.sha256_file(games)}))
    assert collect_mod._proc_done(spec) is not None
    games.write_text('{"game_id": 1}\n{"game_id": 2}\n')  # tampered after done.json
    assert collect_mod._proc_done(spec) is None


def test_overlapping_or_missing_openings_across_processes_are_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    # Both processes get the first half: half the drawn openings twice, half never.
    monkeypatch.setattr(collect_mod, "proc_openings", lambda rows, _p, k: rows[:len(rows) // k])
    with pytest.raises(ValueError, match="openings played differ"):
        _collect(tmp_path, "overlap", 2)
