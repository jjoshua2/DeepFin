"""Game-granularity resume of a killed collect process.

A killed process keeps the games whose rows are in committed shards (each
shard's ``.commit.json`` is written last, atomically) and replays only the rest
of its openings. Every game draws from its own RNG stream keyed by (process
seed, dole index), so the replayed games are the ones an uninterrupted run
plays: the resumed collect holds the SAME games (by content-hash game id), not
a sample selected by which games happened to finish before the kill. Before anything is appended, the directory is
repaired on disk: torn or uncommitted shards, temporary zarr stores and
games.jsonl lines whose rows were never flushed are deleted. Kills are
simulated in-process (K = 1) at each step of the write path; one test kills
twice. A resumed collect that kept games still differs from an unkilled one in
shard order (so its digest differs), and says so.
"""
from __future__ import annotations

import json
import sys
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


def _same_games(gen: Path, plain: Path) -> None:
    """The resumed collect holds exactly the uninterrupted collect's games."""
    def key(g: dict[str, Any]) -> tuple[Any, ...]:
        return (g["game_id"], g["dole_index"], g["opening_id"], g["result"], g["total_plies"], g["rows"])

    assert sorted(map(key, _games(gen / "games.jsonl"))) == sorted(map(key, _games(plain / "games.jsonl")))


def _check_resumed(gen: Path, stats: dict[str, Any], kept_before: list[str],
                   plain: Path) -> list[dict[str, Any]]:
    """The gates a resumed collect must still pass, and the resume's record."""
    _same_games(gen, plain)
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
    assert last["replayed_share"] == pytest.approx(len(last["replayed_dole_indices"]) / GAMES)
    assert sorted(int(json.loads(ln)["dole_index"]) for ln in kept_before) == sorted(
        set(range(GAMES)) - set(last["replayed_dole_indices"]))
    if kept_before:
        assert stats["procs_game_resumed"] == {0: len(kept_before)}
        assert stats["timing_independent"] is False
    return resumes


def _committed_lines(gen: Path) -> list[str]:
    return collect_mod._committed_prefix(gen / "procs" / "00")[2]


@pytest.mark.parametrize("kill", sorted(KILLS))
def test_a_killed_process_keeps_its_committed_games_and_replays_the_rest(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kill: str) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    _, plain = _collect(tmp_path, "plain")
    with monkeypatch.context() as mp:
        KILLS[kill](mp)
        with pytest.raises(Killed):
            _collect(tmp_path, "r")
    gen = tmp_path / "r" / "gen_0001" / "collect"
    kept = _committed_lines(gen)
    on_disk = _games(gen / "procs" / "00" / "games.jsonl") if kill != "torn_games_line" else None
    stats, _ = _collect(tmp_path, "r")
    resumes = _check_resumed(gen, stats, kept, plain)
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
    _, plain = _collect(tmp_path, "plain")
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
    resumes = _check_resumed(gen, stats, second, plain)
    assert [r["attempt"] for r in resumes] == [1, 2]
    assert [r["kept_games"] for r in resumes] == [2, 4]
    assert [r["replayed_share"] for r in resumes] == pytest.approx([4 / GAMES, 2 / GAMES])


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


def test_replayed_games_are_the_uninterrupted_games_under_every_random_draw(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Playout-cap randomisation on (a full/fast draw per ply) as well as the
    default root Gumbel noise and move temperature: every draw of a replayed
    game comes from its own (process seed, dole index) stream."""
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    import dataclasses

    capped = dataclasses.replace(_settings(), full_prob=0.5, fast_sims=4, fast_value_rows=True)
    assert capped.temp_start > 0
    assert capped.root_noise_scale > 0
    monkeypatch.setattr(sys.modules[__name__], "_settings", lambda: capped)
    _, plain = _collect(tmp_path, "plain")
    with monkeypatch.context() as mp:
        KILLS["flush_before_commit"](mp)
        with pytest.raises(Killed):
            _collect(tmp_path, "r")
    gen = tmp_path / "r" / "gen_0001" / "collect"
    kept = _committed_lines(gen)
    stats, _ = _collect(tmp_path, "r")
    resumes = _check_resumed(gen, stats, kept, plain)
    assert 0 < resumes[-1]["replayed_share"] < 1


def _corrupt_after_a_kill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                          corrupt: Callable[[Path, int], None]) -> tuple[dict[str, Any], int, Path, Path]:
    """Kill after every game is committed, corrupt the process dir by hand,
    resume. Returns (the resume record, committed shards before the damage, the
    resumed collect, the uninterrupted one)."""
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    _, plain = _collect(tmp_path, "plain")
    with monkeypatch.context() as mp:
        KILLS["before_done"](mp)
        with pytest.raises(Killed):
            _collect(tmp_path, "r")
    gen = tmp_path / "r" / "gen_0001" / "collect"
    proc = gen / "procs" / "00"
    n = len(list((proc / "shards").glob("shard_*.commit.json")))
    assert n >= 4
    corrupt(proc, n)
    after_repair: dict[str, Any] = {}
    real_repair = collect_mod._repair_proc_dir

    def repair(spec: Any) -> Any:  # what is ON DISK right after the repair, before any append
        out = real_repair(spec)
        after_repair["shards"] = sorted(p.name for p in (spec.proc_dir / "shards").iterdir())
        after_repair["lines"] = (spec.proc_dir / "games.jsonl").read_text().splitlines(keepends=True)
        return out

    monkeypatch.setattr(collect_mod, "_repair_proc_dir", repair)
    stats, _ = _collect(tmp_path, "r")
    _same_games(gen, plain)
    rec = stats["procs"][0]["resumes"][-1]
    assert after_repair["shards"] == sorted(f for i in range(rec["kept_shards"]) for f in _shard_files(i))
    assert len(after_repair["lines"]) == rec["kept_games"]
    return rec, n, gen, plain


def _shard_files(i: int) -> list[str]:
    return [f"shard_{i:05d}.commit.json", f"shard_{i:05d}.meta.npz", f"shard_{i:05d}.zarr"]


def _lines_of(proc: Path, shard: int) -> int:
    return int(json.loads((proc / "shards" / f"shard_{shard:05d}.commit.json").read_text())["games_lines"])


def test_a_zarr_deleted_under_its_commit_record_ends_the_committed_prefix(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import shutil

    seen: dict[str, int] = {}

    def corrupt(proc: Path, _n: int) -> None:
        seen["lines"] = _lines_of(proc, 1)
        shutil.rmtree(proc / "shards" / "shard_00002.zarr")

    rec, n, _, _ = _corrupt_after_a_kill(tmp_path, monkeypatch, corrupt)
    assert rec["kept_shards"] == 2
    assert rec["kept_games"] == seen["lines"]
    # the repair removed every file from the damaged shard on
    assert rec["discarded"] == sorted(f for i in range(2, n) for f in _shard_files(i) if f != "shard_00002.zarr")


def test_an_altered_games_line_ends_the_committed_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, int] = {}

    def corrupt(proc: Path, _n: int) -> None:
        seen["lines"] = _lines_of(proc, 1)
        lines = (proc / "games.jsonl").read_text().splitlines(keepends=True)
        g = json.loads(lines[seen["lines"]])  # the first game of shard 2
        g["result"] = "1-0" if g["result"] != "1-0" else "0-1"
        lines[seen["lines"]] = json.dumps(g) + "\n"
        (proc / "games.jsonl").write_text("".join(lines))

    rec, n, _, _ = _corrupt_after_a_kill(tmp_path, monkeypatch, corrupt)
    assert rec["kept_shards"] == 2  # the games digest of shard 2 no longer matches
    assert rec["kept_games"] == seen["lines"]
    assert rec["discarded"] == sorted(f for i in range(2, n) for f in _shard_files(i))


def test_a_meta_with_another_row_count_ends_the_committed_prefix(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    seen: dict[str, int] = {}

    def corrupt(proc: Path, _n: int) -> None:
        seen["lines"] = _lines_of(proc, 1)
        path = proc / "shards" / "shard_00002.meta.npz"
        with np.load(path) as z:
            arrays = {k: z[k][:-1] for k in z.files}  # one row short of the commit record
        collect_mod._write_shard_meta(path, arrays)

    rec, n, _, _ = _corrupt_after_a_kill(tmp_path, monkeypatch, corrupt)
    assert rec["kept_shards"] == 2
    assert rec["kept_games"] == seen["lines"]
    assert rec["discarded"] == sorted(f for i in range(2, n) for f in _shard_files(i))


def test_a_stale_uncommitted_store_past_the_prefix_is_removed(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def corrupt(proc: Path, n: int) -> None:
        stale = proc / "shards" / f"shard_{n:05d}.zarr"
        stale.mkdir()
        (stale / ".zgroup").write_text("{}")
        (proc / "shards" / f"shard_{n:05d}.meta.npz").write_bytes(b"stale")

    rec, n, _, _ = _corrupt_after_a_kill(tmp_path, monkeypatch, corrupt)
    assert rec["kept_shards"] == n
    assert rec["discarded"] == [f"shard_{n:05d}.meta.npz", f"shard_{n:05d}.zarr"]


# ---- move journal ---------------------------------------------------------------------------------
def _journal_entries(path: Path) -> list[dict[str, Any]]:
    return collect_mod._read_journal(path)


def _kill_journal_write(n: int, *, torn: bool) -> Callable[[pytest.MonkeyPatch], None]:
    """Die at the n-th journal write; ``torn`` leaves half a line behind first."""
    real = collect_mod._Journal.write
    seen = [0]

    def write(self: Any, entries: list[dict[str, Any]]) -> None:
        if entries:
            seen[0] += 1
            if seen[0] == n:
                if torn:
                    self.fh.write(json.dumps(entries)[:23])
                    self.fh.flush()
                raise Killed
        real(self, entries)

    return lambda mp: mp.setattr(collect_mod._Journal, "write", write)


def _check_forced(gen: Path, at_kill: list[dict[str, Any]], kept_ids: set[int]) -> int:
    """Every in-flight game of the kill replayed exactly its journaled moves."""
    games = {int(g["dole_index"]): g for g in _games(gen / "games.jsonl")}
    n = 0
    for e in at_kill:
        if int(e["i"]) in kept_ids:
            continue
        g = games[int(e["i"])]
        assert g["moves"][int(e["p"]) - int(g["start_ply"])] == int(e["m"])
        n += 1
    return n


def test_a_torn_journal_tail_is_repaired_and_in_flight_games_replay_their_journaled_moves(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    _, plain = _collect(tmp_path, "plain")
    with monkeypatch.context() as mp:
        _kill_journal_write(20, torn=True)(mp)
        with pytest.raises(Killed):
            _collect(tmp_path, "r")
    gen = tmp_path / "r" / "gen_0001" / "collect"
    proc = gen / "procs" / "00"
    raw = (proc / "journal.jsonl").read_bytes()
    assert not raw.endswith(b"\n")  # the kill left a torn record
    at_kill = _journal_entries(proc / "journal.jsonl")
    kept_ids = {int(json.loads(ln)["dole_index"]) for ln in _committed_lines(gen)}
    after_repair: dict[str, bytes] = {}
    real_repair = collect_mod._repair_journal

    def repair(proc_dir: Path, *a: Any, **k: Any) -> Any:
        out = real_repair(proc_dir, *a, **k)
        after_repair["journal"] = (proc_dir / "journal.jsonl").read_bytes()
        return out

    monkeypatch.setattr(collect_mod, "_repair_journal", repair)
    stats, _ = _collect(tmp_path, "r")
    # repaired ON DISK before anything was appended: complete records only
    assert after_repair["journal"] == b"" or after_repair["journal"].endswith(b"\n")
    rec = stats["procs"][0]["resumes"][-1]
    assert rec["journaled_games"] > 0
    assert _check_forced(gen, at_kill, kept_ids) == rec["journaled_plies"] > 0
    _same_games(gen, plain)


def test_two_kills_during_journaled_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    _, plain = _collect(tmp_path, "plain")
    gen = tmp_path / "r" / "gen_0001" / "collect"
    with monkeypatch.context() as mp:
        _kill_journal_write(20, torn=True)(mp)
        with pytest.raises(Killed):
            _collect(tmp_path, "r")
    with monkeypatch.context() as mp:
        _kill_journal_write(3, torn=False)(mp)  # killed again while replaying the forced prefixes
        with pytest.raises(Killed):
            _collect(tmp_path, "r")
    stats, _ = _collect(tmp_path, "r")
    assert [r["attempt"] for r in stats["procs"][0]["resumes"]] == [1, 2]
    _same_games(gen, plain)


def test_a_kill_during_the_journal_repair_repairs_the_same_way(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    _, plain = _collect(tmp_path, "plain")
    gen = tmp_path / "r" / "gen_0001" / "collect"
    with monkeypatch.context() as mp:
        _kill_journal_write(20, torn=True)(mp)
        with pytest.raises(Killed):
            _collect(tmp_path, "r")
    real_write = collect_mod._write_journal

    def die_mid_rewrite(path: Path, entries: list[dict[str, Any]]) -> None:
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps(entries)[:40])  # the temporary file torn, the journal untouched
        raise Killed

    with monkeypatch.context() as mp:
        mp.setattr(collect_mod, "_write_journal", die_mid_rewrite)
        with pytest.raises(Killed):
            _collect(tmp_path, "r")
    monkeypatch.setattr(collect_mod, "_write_journal", real_write)
    _collect(tmp_path, "r")
    _same_games(gen, plain)


def test_a_journal_entry_that_disagrees_with_a_committed_game_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    with monkeypatch.context() as mp:
        KILLS["before_done"](mp)
        with pytest.raises(Killed):
            _collect(tmp_path, "r")
    proc = tmp_path / "r" / "gen_0001" / "collect" / "procs" / "00"
    g = json.loads(_committed_lines(tmp_path / "r" / "gen_0001" / "collect")[0])
    wrong = next(m for m in range(4672) if m != g["moves"][0])
    with (proc / "journal.jsonl").open("a") as fh:
        fh.write(json.dumps([{"i": g["dole_index"], "p": g["start_ply"], "m": wrong, "f": True,
                              "s0": None, "s1": None}]) + "\n")
    with pytest.raises(ValueError, match="disagrees with committed game"):
        _collect(tmp_path, "r")


# ---- drain on STOP -------------------------------------------------------------------------------
def test_stop_drains_the_in_flight_games_and_a_rerun_continues_dealing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collect_mod, "_open_stockfish", quiet_stockfish)
    _, plain = _collect(tmp_path, "plain")
    stop = tmp_path / "r" / "STOP"
    real_append = collect_mod.append_jsonl
    lines = [0]

    def append(path: Path, value: Any) -> None:
        real_append(path, value)
        if Path(path).name == "games.jsonl":
            lines[0] += 1
            if lines[0] == 2:
                stop.write_text("")  # an operator's STOP after the 2nd finished game

    with monkeypatch.context() as mp:
        mp.setattr(collect_mod, "append_jsonl", append)
        with pytest.raises(collect_mod.CollectDrained, match="drained"):
            _collect(tmp_path, "r")
    gen = tmp_path / "r" / "gen_0001" / "collect"
    assert not (gen / "manifest.json").exists()  # not sealed
    drained = json.loads((gen / "procs" / "00" / "drained.json").read_text())
    finished = _games(gen / "procs" / "00" / "games.jsonl")
    # every dealt game finished: the 2 in flight at STOP (concurrency 2) plus the
    # ones finished before it, and none was dealt after it
    assert drained["dealt"] == drained["games"] == len(finished) < GAMES
    assert len(_committed_lines(gen)) == len(finished)  # all committed: nothing for a journal to force
    assert collect_mod._read_journal(gen / "procs" / "00" / "journal.jsonl") == []
    stop.unlink()
    stats, _ = _collect(tmp_path, "r")
    rec = stats["procs"][0]["resumes"][-1]
    assert rec["previous_stop"] == "drain"
    assert rec["dealt_before"] == drained["dealt"]
    assert rec["replayed_dole_indices"] == list(range(drained["dealt"], GAMES))  # next dole index on
    assert rec["journaled_games"] == 0
    _same_games(gen, plain)
