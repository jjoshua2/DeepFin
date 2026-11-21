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


def _collect(tmp_path: Path, run: str, procs: int, **kw: Any) -> tuple[dict[str, Any], Path]:
    ckpt = tmp_path / "gen0.pt"
    if not ckpt.exists():
        save_tiny_checkpoint(ckpt)
    gen = tmp_path / run / "gen_0001" / "collect"
    stats = collect_mod.collect_generation(gen_dir=gen, generation=1, checkpoint=ckpt,
                                           train_jsonl=_openings(tmp_path), settings=_settings(procs), **kw)
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
    monkeypatch.setattr(collect_mod, "write_manifest", real_manifest)
    stats, _ = _collect(tmp_path, "resumed", 2)
    assert stats["procs_resumed"] == [0, 1]
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


@pytest.mark.parametrize("procs", [1, 2])
def test_every_process_honours_the_pause_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, procs: int) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    marker = tmp_path / "gpu_window"
    marker.write_text("")
    with pytest.raises(collect_mod.CollectBlocked):
        _collect(tmp_path, "p", procs, pause_markers=(marker,))
    assert not list((tmp_path / "p").rglob("done.json"))


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


def test_default_is_three_on_cuda_one_on_cpu_and_a_pinned_value_wins(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    assert simple_rl_loop.resolve_collect_procs(_driver_args(run_dir)) == 1
    assert simple_rl_loop.resolve_collect_procs(_driver_args(run_dir, device="cuda")) == 3
    assert simple_rl_loop.resolve_collect_procs(_driver_args(run_dir, "--collect-procs", "2", device="cuda")) == 2
    run_dir.mkdir()
    # A run pinned before the flag existed resumes at the one process it ran with.
    (run_dir / simple_rl_loop.RUN_CONFIG).write_text(json.dumps({"settings": {"lr": 1e-3}, "changes": []}))
    assert simple_rl_loop.resolve_collect_procs(_driver_args(run_dir, device="cuda")) == 1
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


def test_insufficient_free_vram_is_refused_never_stepped_down(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a: (10 * 2**30, 32 * 2**30))
    settings = replace(_settings(3, games=512), concurrency=128, device="cuda")
    with pytest.raises(ValueError, match=r"--collect-procs 3 needs about .* pass --collect-procs 1 explicitly"):
        collect_mod.check_vram(settings)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_a: (31 * 2**30, 32 * 2**30))
    record = collect_mod.check_vram(settings)
    assert record is not None
    assert record["need_gib"] < 31


def test_combined_evaluator_record_fails_closed_on_disagreement() -> None:
    base = {"cuda_graphs": True, "evaluator_mode": "cuda_graphs", "eval_max_batch_realised": 4096,
            "cuda_graph_stats": {"replays": 3, "eager_over_max_rows": 1, "max_rows": 512}}
    same = collect_mod._combined_evaluator_record([base, dict(base)])
    assert same["eval_max_batch_realised"] == 4096
    assert same["evaluator_mode"] == "cuda_graphs"
    assert same["cuda_graph_stats"] == {"replays": 6, "eager_over_max_rows": 2, "max_rows": 512}
    odd = collect_mod._combined_evaluator_record([base, {**base, "eval_max_batch_realised": 2048,
                                                         "evaluator_mode": "eager", "cuda_graphs": False}])
    assert odd["eval_max_batch_realised"] is None
    assert odd["evaluator_mode"] == "mixed"
    assert not odd["cuda_graphs"]
