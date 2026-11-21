"""Game-granularity resume of a killed collect process.

A killed process keeps the games whose rows are in committed shards (each
shard's ``.commit.json`` is written last, atomically) and replays only the rest
of its openings with a fresh seed. Before anything is appended, the directory is
repaired on disk: torn or uncommitted shards, temporary zarr stores and
games.jsonl lines whose rows were never flushed are deleted. Kills are
simulated in-process (K = 1) at each step of the write path; one test kills
twice. A resumed collect that kept games is not digest-identical to an
unkilled one, and says so.
"""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from chess_anti_engine.replay import shard as shard_mod
from chess_anti_engine.simple_rl import collect as collect_mod
from chess_anti_engine.simple_rl.collect import CollectSettings
from tests.simple_rl_helpers import quiet_stockfish, save_tiny_checkpoint

_LINES = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | e2e4 e7e5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | d2d4 d7d5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | c2c4 e7e5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | g1f3 d7d5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | e2e4 c7c5",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | b1c3 g8f6",
]
GAMES = 6


class Killed(BaseException):
    """A simulated SIGKILL: nothing after the raise point runs except ``finally``s."""


def _settings() -> CollectSettings:
    # A small shard: every finished game flushes one.
    return CollectSettings(games=GAMES, concurrency=2, sims=8, max_plies=24, rows_per_shard=5,
                           book_mix=(1.0,), collect_procs=1, seed=91)


def _collect(tmp_path: Path, run: str) -> tuple[dict[str, Any], Path]:
    ckpt = tmp_path / "gen0.pt"
    if not ckpt.exists():
        save_tiny_checkpoint(ckpt)
    train = tmp_path / "train.jsonl"
    train.write_text("".join(json.dumps({"id": f"o{i}", "book": 1, "line": ln}) + "\n" for i, ln in enumerate(_LINES)))
    gen = tmp_path / run / "gen_0001" / "collect"
    stats = collect_mod.collect_generation(gen_dir=gen, generation=1, checkpoint=ckpt, train_jsonl=train,
                                           settings=_settings())
    return stats, gen


def _games(path: Path) -> list[dict[str, Any]]:
    return [json.loads(r) for r in path.read_text().splitlines() if r.strip()]


def _nth(n: int, before: Callable[..., None], real: Callable[..., Any],
         when: Callable[..., bool] = lambda *a, **k: True) -> Callable[..., Any]:
    """Call ``real`` normally, except the n-th matching call runs ``before`` and dies."""
    seen = [0]

    def wrapped(*a: Any, **k: Any) -> Any:
        if when(*a, **k):
            seen[0] += 1
            if seen[0] == n:
                before(*a, **k)
                raise Killed
        return real(*a, **k)

    return wrapped


def _torn_line(path: Path, value: Any) -> None:
    with Path(path).open("a") as fh:
        fh.write(json.dumps(value)[:17])  # half a line, no newline


def _torn_meta(path: Path, _arrays: Any) -> None:
    Path(path).write_bytes(b"PK\x03\x04torn")


def _torn_zarr_tmp(path: Any, **_k: Any) -> None:
    p = Path(path)
    tmp = p.with_name(f"._tmp_1_deadbeef_{p.name}")
    tmp.mkdir(parents=True)
    (tmp / ".zgroup").write_text("{")


def _nothing(*_a: Any, **_k: Any) -> None:
    return None


KILLS: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    # games.jsonl: a line half written (game 3)
    "torn_games_line": lambda mp: mp.setattr(
        collect_mod, "append_jsonl", _nth(3, _torn_line, collect_mod.append_jsonl)),
    # shard flush 3: the zarr is renamed into place, the meta file is torn
    "flush_torn_meta": lambda mp: mp.setattr(
        collect_mod, "_write_shard_meta", _nth(3, _torn_meta, collect_mod._write_shard_meta)),
    # shard flush 3: killed while writing the zarr (its temporary store is left)
    "flush_torn_zarr": lambda mp: mp.setattr(
        shard_mod, "save_local_shard_arrays", _nth(3, _torn_zarr_tmp, shard_mod.save_local_shard_arrays)),
    # shard flush 3: zarr and meta written, killed before the commit record
    "flush_before_commit": lambda mp: mp.setattr(
        collect_mod, "write_json_atomic", _nth(3, _nothing, collect_mod.write_json_atomic,
                                               when=lambda p, *_a, **_k: str(p).endswith(".commit.json"))),
    # every game played and flushed, killed before done.json
    "before_done": lambda mp: mp.setattr(
        collect_mod, "write_json_atomic", _nth(1, _nothing, collect_mod.write_json_atomic,
                                               when=lambda p, *_a, **_k: Path(p).name == "done.json")),
}


def _check_resumed(gen: Path, stats: dict[str, Any], kept_before: list[str]) -> list[dict[str, Any]]:
    """The gates a resumed collect must still pass, and the resume's record."""
    games = _games(gen / "games.jsonl")
    starts = (gen / "starts.txt").read_text().splitlines()
    by_line = {ln: f"o{i}" for i, ln in enumerate(_LINES)}
    # every drawn opening played exactly once -- none twice, none dropped
    assert Counter(g["opening_id"] for g in games) == Counter(by_line[ln] for ln in starts)
    assert stats["games"] == GAMES
    assert stats["duplicate_game_ids"] == 0
    assert stats["rows"] == sum(int(g["rows"]) for g in games)
    # the kept games are the committed lines of the killed attempt, byte for byte, first
    merged = (gen / "games.jsonl").read_text().splitlines(keepends=True)
    assert merged[:len(kept_before)] == kept_before
    manifest = json.loads((gen / "manifest.json").read_text())
    resumes = manifest["game_resumes"]["0"]
    assert resumes == stats["procs"][0]["resumes"]
    last = resumes[-1]
    assert last["kept_games"] == len(kept_before)
    assert sorted(last["kept_opening_ids"] + last["replayed_opening_ids"]) == sorted(
        by_line[ln] for ln in starts)
    if kept_before:
        assert stats["procs_game_resumed"] == {0: len(kept_before)}
        assert stats["timing_independent"] is False
        assert last["seed"] != stats["procs"][0]["seed"]  # re-seeded
    return resumes


def _committed_lines(gen: Path) -> list[str]:
    return collect_mod._committed_prefix(gen / "procs" / "00")[2]


@pytest.mark.parametrize("kill", sorted(KILLS))
def test_a_killed_process_keeps_its_committed_games_and_replays_the_rest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kill: str) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    with monkeypatch.context() as mp:
        KILLS[kill](mp)
        with pytest.raises(Killed):
            _collect(tmp_path, "r")
    gen = tmp_path / "r" / "gen_0001" / "collect"
    kept = _committed_lines(gen)
    on_disk = _games(gen / "procs" / "00" / "games.jsonl") if kill != "torn_games_line" else None
    stats, _ = _collect(tmp_path, "r")
    resumes = _check_resumed(gen, stats, kept)
    assert len(resumes) == 1
    expected_kept = {"torn_games_line": 2, "flush_torn_meta": 2, "flush_torn_zarr": 2,
                     "flush_before_commit": 2, "before_done": GAMES}[kill]
    assert len(kept) == expected_kept
    discarded = resumes[0]["discarded"]
    if kill == "flush_torn_meta":
        assert discarded == ["shard_00002.meta.npz", "shard_00002.zarr"]
    elif kill == "flush_torn_zarr":
        assert discarded == ["._tmp_1_deadbeef_shard_00002.zarr"]
    elif kill == "flush_before_commit":
        assert discarded == ["shard_00002.meta.npz", "shard_00002.zarr"]
    else:
        assert discarded == []
    if on_disk is not None and kill != "before_done":
        # the killed attempt had written a line for a game whose rows never reached a
        # committed shard; that line was dropped and its opening replayed
        assert len(on_disk) > len(kept)
    if kill == "before_done":
        assert resumes[0]["replayed_opening_ids"] == []


def test_two_kills_then_a_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    gen = tmp_path / "r" / "gen_0001" / "collect"
    with monkeypatch.context() as mp:
        KILLS["flush_torn_meta"](mp)  # dies on flush 3: keeps 2 games
        with pytest.raises(Killed):
            _collect(tmp_path, "r")
    first = _committed_lines(gen)
    with monkeypatch.context() as mp:
        KILLS["torn_games_line"](mp)  # dies on the 3rd game line of attempt 2: keeps 2 + 2
        with pytest.raises(Killed):
            _collect(tmp_path, "r")
    second = _committed_lines(gen)
    assert second[:len(first)] == first
    assert len(second) == 4
    stats, _ = _collect(tmp_path, "r")
    resumes = _check_resumed(gen, stats, second)
    assert [r["attempt"] for r in resumes] == [1, 2]
    assert [r["kept_games"] for r in resumes] == [2, 4]
    assert len({r["seed"] for r in resumes}) == 2


def test_a_kill_before_anything_is_committed_restarts_on_the_own_seed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    from scripts.simple_rl_collect_gates import content_digest

    _, plain = _collect(tmp_path, "plain")
    with monkeypatch.context() as mp:
        mp.setattr(collect_mod, "_write_shard_meta", _nth(1, _torn_meta, collect_mod._write_shard_meta))
        with pytest.raises(Killed):
            _collect(tmp_path, "r")
    stats, gen = _collect(tmp_path, "r")
    assert stats["procs"][0]["resumes"][0]["kept_games"] == 0
    assert stats["procs_game_resumed"] == {}
    # nothing kept = a plain restart: the same rows as an uninterrupted collect
    assert content_digest(gen)["content_sha256"] == content_digest(plain)["content_sha256"]
