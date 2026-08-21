"""Crash-resilient per-game persistence + --resume for the two match drivers.

The failure these exist for (2026-08-21): a 128-game compiled arena died of
CUDA OOM at ply 20 with ZERO games persisted, and the relaunch lost its first
minutes the same way. Hours of GPU time, nothing scoreable, because the only
durable artifact was written after the last game.

The load-bearing property is not "a file appears". It is that a run which
crashed and was resumed produces THE SAME NUMBER as one that never crashed —
same pentanomial, same Elo, same CI — and that a resume onto different settings
is refused instead of silently averaging two populations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chess
import pytest

from chess_anti_engine.utils.game_log import (
    GameLogWriter,
    fingerprint_differences,
    latest_rows_by_key,
    read_game_log,
    settings_fingerprint,
)
from tests.script_loading import load_script_module

# Four openings = four pairs = eight games. The per-pair candidate scores are
# chosen so that BOTH plausible ways of botching the half-pair filter change
# the answer: pair 2 is (1.0, 0.0), so counting its orphan half twice gives 2.0
# and imputing the missing half as a draw gives 1.5, against a true 1.0.
_OPENING_FENS = (
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/5N2/8/8/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
)
_PAIR_PLAN: dict[int, tuple[float, float]] = {
    0: (1.0, 1.0),   # WW
    1: (0.5, 0.0),   # LD_DL
    2: (1.0, 0.0),   # DD_WL  <- the pair the simulated crash cuts in half
    3: (0.0, 0.0),   # LL
}
_EXPECTED_PENTANOMIAL = {"WW": 1, "WD_DW": 0, "DD_WL": 1, "LD_DL": 1, "LL": 1}


class _SimulatedCrash(RuntimeError):
    """Stands in for the CUDA OOM that started all this."""


def _result_for(score: float, *, a_is_white: bool) -> str:
    """PGN result string (White POV) giving the candidate ``score``."""
    if score == 0.5:
        return "1/2-1/2"
    return "1-0" if ((score == 1.0) == a_is_white) else "0-1"


def _openings_file(tmp_path: Path, n: int = 4) -> Path:
    path = tmp_path / "openings.fen"
    path.write_text("\n".join(_OPENING_FENS[:n]) + "\n")
    return path


# ---------------------------------------------------------------------------
# game_log module
# ---------------------------------------------------------------------------

def test_fingerprint_is_order_independent_and_value_sensitive() -> None:
    a = settings_fingerprint({"seed": 42, "sims": 64})
    b = settings_fingerprint({"sims": 64, "seed": 42})
    c = settings_fingerprint({"seed": 42, "sims": 65})
    assert a == b, "key order must not change the fingerprint"
    assert a != c


def test_fingerprint_refuses_a_value_json_cannot_serialize() -> None:
    # With ``default=str`` this would hash a repr carrying a memory address, so
    # the same invocation would fingerprint differently every run and --resume
    # would refuse a log it wrote itself. Fail loudly instead.
    with pytest.raises(TypeError):
        settings_fingerprint({"board": chess.Board()})


def test_read_game_log_tolerates_a_truncated_tail_but_not_a_middle_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a.games.jsonl"
    with GameLogWriter(path, driver="t", settings={"k": 1}) as log:
        log.write_game({"game_index": 0})
        log.write_game({"game_index": 1})
    # A crash mid-write leaves exactly this: a half-written FINAL line.
    with path.open("a") as fh:
        fh.write('{"kind": "game", "game_ind')
    parsed = read_game_log(path)
    assert [row["game_index"] for row in parsed.games] == [0, 1]
    assert parsed.truncated_tail is True

    corrupt = tmp_path / "b.games.jsonl"
    corrupt.write_text(
        json.dumps({"kind": "header", "settings": {}, "fingerprint": "x"}) + "\n"
        + "{not json}\n"
        + json.dumps({"kind": "game", "game_index": 0}) + "\n"
    )
    with pytest.raises(ValueError, match="corrupt"):
        read_game_log(corrupt)


def test_writer_flushes_every_game_before_the_next_one(tmp_path: Path) -> None:
    """The whole point: a game is on disk before the next one starts."""
    path = tmp_path / "flush.games.jsonl"
    log = GameLogWriter(path, driver="t", settings={"k": 1})
    log.write_game({"game_index": 0})
    # Read with a SEPARATE handle while the writer is still open, exactly as a
    # post-mortem would after the process was killed.
    assert len(read_game_log(path).games) == 1
    log.write_game({"game_index": 1})
    assert len(read_game_log(path).games) == 2
    log.close()


def test_latest_rows_by_key_keeps_the_last_write() -> None:
    rows = [
        {"game_index": 0, "result": "1-0"},
        {"game_index": 1, "result": "0-1"},
        {"game_index": 1, "result": "1/2-1/2"},  # replayed after a crash
    ]
    out = latest_rows_by_key(rows, key=lambda r: int(r["game_index"]))
    assert out[1]["result"] == "1/2-1/2"


def test_fingerprint_differences_names_added_and_removed_keys() -> None:
    diffs = fingerprint_differences({"seed": 1, "gone": 2}, {"seed": 2, "new": 3})
    joined = "\n".join(diffs)
    assert "seed" in joined, joined
    assert "gone" in joined, "a key the log HAS and this run lacks must be named"
    assert "new" in joined, "a key this run HAS and the log lacks must be named"



def test_no_header_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    path = tmp_path / "headerless.jsonl"
    path.write_text(json.dumps({"kind": "game", "game_index": 0}) + "\n")
    with pytest.raises(ValueError, match="no header row"):
        read_game_log(path)


# ---------------------------------------------------------------------------
# scripts/arena_standard.py
# ---------------------------------------------------------------------------

def _fake_matched_time(
    *, crash_after: int | None = None, on_crash: Any = None,
):
    """Stand-in for the UCI play loop: deterministic, and emits per game.

    Deliberately NOT a stub that only returns pair scores: the property under
    test is that each FINISHED game reaches the sink (and therefore the disk)
    while the loop is still running, so the fake emits and only then keeps
    playing — and, when asked, dies mid-pair like the real OOM did.
    """
    # Counted across CALLS, not within one: the chunked arena loop calls the
    # play function once per chunk, so a per-call counter would never reach a
    # crash threshold that spans chunks.
    emitted = 0

    def _play(
        _cand: str, _ref: str, openings: list[chess.Board], *,
        pgn_sink: Any = None, pair_ids: list[int] | None = None, **_kw: Any,
    ) -> list[float]:
        nonlocal emitted
        ids = list(range(len(openings))) if pair_ids is None else list(pair_ids)
        pair_scores: list[float] = []
        for k, opening in enumerate(openings):
            scores: list[float] = []
            for half, a_is_white in ((0, True), (1, False)):
                score = _PAIR_PLAN[ids[k]][half]
                if pgn_sink is not None:
                    pgn_sink(
                        pair_id=ids[k], half=half, a_is_white=a_is_white,
                        start_fen=opening.fen(), moves=(),
                        result=_result_for(score, a_is_white=a_is_white),
                        termination="rules", plies=10 + ids[k],
                        duration_s=0.5, chunk=None, loop="matched_time",
                    )
                emitted += 1
                scores.append(score)
                if crash_after is not None and emitted >= crash_after:
                    if on_crash is not None:
                        # Snapshot the log from INSIDE the doomed process: after
                        # the exception unwinds, garbage collection closes the
                        # writer and flushes it, which would hide a missing
                        # per-game flush from every in-process assertion.
                        on_crash()
                    raise _SimulatedCrash("simulated CUDA OOM")
            pair_scores.append(scores[0] + scores[1])
        return pair_scores
    return _play


def _run_arena(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    log_path: Path,
    crash_after: int | None = None,
    on_crash: Any = None,
    resume: bool = False,
    seed: int = 7,
    n_openings: int = 4,
    candidate: str = "cand.pt",
    pgn_out: Path | None = None,
    openings_file: Path | None = None,
) -> dict:
    import scripts.arena_standard as arena

    monkeypatch.setattr(
        arena, "play_paired_games_matched_time",
        _fake_matched_time(crash_after=crash_after, on_crash=on_crash),
    )
    fen_file = (
        openings_file if openings_file is not None
        else _openings_file(tmp_path, n_openings)
    )
    return arena.run_arena(
        candidate=candidate, reference="ref.pt",
        games=2 * n_openings,
        openings_path=None, openings_fen=fen_file,
        opening_plies=16, mode="matched_time",
        sims_candidate=32, sims_reference=32,
        ms_per_move=100, max_plies=300, temperature=0.1,
        gumbel_add_noise=True, device="cpu", seed=seed, out_path=None,
        game_log_path=log_path, resume=resume, pgn_out=pgn_out,
    )


def test_every_finished_game_is_on_disk_when_the_run_dies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The motivating failure, inverted: a crash must leave the finished games."""
    log_path = tmp_path / "crash.games.jsonl"
    at_crash: list[int] = []
    with pytest.raises(_SimulatedCrash):
        _run_arena(
            monkeypatch, tmp_path, log_path=log_path, crash_after=5,
            on_crash=lambda: at_crash.append(len(read_game_log(log_path).games)),
        )
    assert at_crash == [5], (
        "the finished games must already be readable from another handle "
        "WHILE the doomed process is still running — a buffered write that "
        f"only lands on close is exactly the 128-game loss; saw {at_crash}"
    )

    parsed = read_game_log(log_path)
    assert len(parsed.games) == 5, (
        "the five games that finished before the crash must be on disk; the "
        f"log holds {len(parsed.games)}"
    )
    row = parsed.games[0]
    # Everything a later reader (or a resume) needs about ONE game.
    for key in (
        "pair_id", "half", "opening_index", "opening_fen", "a_is_white",
        "result", "score_candidate", "plies", "termination", "seed", "chunk",
        "loop",
    ):
        assert key in row, f"game row is missing {key!r}"
    assert row["seed"] == 7
    assert row["opening_fen"] == _OPENING_FENS[0]
    assert row["loop"] == "matched_time"


def test_resumed_arena_equals_an_uninterrupted_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """THE test: crash mid-pair, resume, get the uninterrupted answer.

    The interrupted run dies after five games, i.e. two complete pairs plus one
    ORPHAN half. The resume must keep the two pairs, discard and replay the
    orphan, play the untouched pair, and score all four as one pentanomial.
    """
    clean = _run_arena(
        monkeypatch, tmp_path, log_path=tmp_path / "clean.games.jsonl",
    )
    assert clean["pentanomial"] == _EXPECTED_PENTANOMIAL, "fixture drifted"

    log_path = tmp_path / "resumed.games.jsonl"
    with pytest.raises(_SimulatedCrash):
        _run_arena(monkeypatch, tmp_path, log_path=log_path, crash_after=5)
    resumed = _run_arena(monkeypatch, tmp_path, log_path=log_path, resume=True)

    assert resumed["pentanomial"] == clean["pentanomial"]
    assert resumed["pairs"] == clean["pairs"] == 4
    assert resumed["games"] == clean["games"] == 8
    assert resumed["elo"] == clean["elo"]
    assert resumed["elo_ci95"] == clean["elo_ci95"]
    assert resumed["score"] == clean["score"]
    assert resumed["truncated"] is False
    # Provenance: the row must SAY it is a splice of two processes.
    assert resumed["resumed_pairs"] == 2
    assert resumed["resumed_orphan_pairs"] == 1
    assert clean["resumed_pairs"] == 0
    assert resumed["game_log_agrees"] is True

    # 5 crashed rows + 4 replayed/new rows, collapsing to the 8 real games.
    parsed = read_game_log(log_path)
    assert len(parsed.games) == 9
    unique = latest_rows_by_key(
        parsed.games, key=lambda r: (int(r["pair_id"]), int(r["half"])),
    )
    assert len(unique) == 8


def test_resume_with_nothing_left_to_play_scores_the_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A complete log + --resume must not replay a single game."""
    log_path = tmp_path / "done.games.jsonl"
    clean = _run_arena(monkeypatch, tmp_path, log_path=log_path)

    def _must_not_run(*_a: object, **_kw: object) -> list[float]:
        raise AssertionError("a fully-resumed run played a game")

    import scripts.arena_standard as arena
    monkeypatch.setattr(arena, "play_paired_games_matched_time", _must_not_run)
    again = arena.run_arena(
        candidate="cand.pt", reference="ref.pt", games=8,
        openings_path=None, openings_fen=_openings_file(tmp_path),
        opening_plies=16, mode="matched_time",
        sims_candidate=32, sims_reference=32, ms_per_move=100, max_plies=300,
        temperature=0.1, gumbel_add_noise=True, device="cpu", seed=7,
        out_path=None, game_log_path=log_path, resume=True,
    )
    assert again["pentanomial"] == clean["pentanomial"]
    assert again["resumed_pairs"] == 4


def test_existing_game_log_without_resume_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    log_path = tmp_path / "twice.games.jsonl"
    _run_arena(monkeypatch, tmp_path, log_path=log_path)
    with pytest.raises(SystemExit) as exc:
        _run_arena(monkeypatch, tmp_path, log_path=log_path)
    message = str(exc.value)
    # The refusal must name BOTH ways out, or the next person deletes the file
    # to get past it.
    assert "--resume" in message, message
    assert "--games-out" in message, message


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"seed": 8}, "seed"),
        ({"candidate": "a_different_net.pt"}, "candidate"),
        ({"n_openings": 3}, "games"),
    ],
)
def test_resume_refuses_when_a_recorded_setting_moved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    kwargs: dict[str, Any], expected: str,
) -> None:
    """A resume onto different settings would average two populations."""
    log_path = tmp_path / "mismatch.games.jsonl"
    with pytest.raises(_SimulatedCrash):
        _run_arena(monkeypatch, tmp_path, log_path=log_path, crash_after=3)
    with pytest.raises(SystemExit) as exc:
        _run_arena(
            monkeypatch, tmp_path, log_path=log_path, resume=True, **kwargs,
        )
    assert expected in str(exc.value)


def test_resume_refuses_when_the_schedule_does_not_reproduce(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The run-time proof that resume rests on a reproducible schedule.

    Same settings fingerprint (the openings PATH is unchanged), but the file
    behind it now holds different positions. Nothing about the fingerprint can
    see that, so the per-pair ``opening_fen`` check is the only thing standing
    between a resume and two different opening sets in one pentanomial.
    """
    log_path = tmp_path / "swapped.games.jsonl"
    with pytest.raises(_SimulatedCrash):
        _run_arena(monkeypatch, tmp_path, log_path=log_path, crash_after=3)
    swapped = list(_OPENING_FENS[:4])
    swapped[0], swapped[1] = swapped[1], swapped[0]
    # Same PATH (so the fingerprint is unchanged), different contents. The
    # cache_clear stands in for the fresh PROCESS a real resume runs in —
    # `_load_fen_list` is lru_cached per path, so without it this test would
    # re-read its own first load and the guard would never be reached.
    from chess_anti_engine.selfplay.opening import _load_fen_list

    (tmp_path / "openings.fen").write_text("\n".join(swapped) + "\n")
    _load_fen_list.cache_clear()
    with pytest.raises(SystemExit, match="NOT reproducible"):
        _run_arena(
            monkeypatch, tmp_path, log_path=log_path, resume=True,
            openings_file=tmp_path / "openings.fen",
        )


def test_arena_opening_schedule_is_a_pure_function_of_seed_and_book(
    tmp_path: Path,
) -> None:
    """Both opening paths must regenerate identically for a fixed seed."""
    import numpy as np

    from scripts.arena_standard import load_fen_openings

    fen_file = _openings_file(tmp_path)
    first = [b.fen() for b in load_fen_openings(
        fen_file, n_pairs=3, rng=np.random.default_rng(11))]
    second = [b.fen() for b in load_fen_openings(
        fen_file, n_pairs=3, rng=np.random.default_rng(11))]
    other = [b.fen() for b in load_fen_openings(
        fen_file, n_pairs=3, rng=np.random.default_rng(12))]
    assert first == second, "same seed must give the same subsample"
    assert len(first) == 3
    # Not asserting first != other: a 3-of-4 subsample can legitimately repeat.
    assert sorted(other) == sorted(set(other))


def test_resumed_pgn_tags_the_replayed_pair(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The orphan game cannot be unwritten, so mark its replacements.

    The PGN is append-only and already holds the discarded half-game. A pooled
    fit that block-bootstraps on PairId would otherwise see a three-game pair
    with no way to tell which game is the stale one. Rule: for any PairId
    carrying ``ResumeReplay``, the games WITHOUT the tag are the orphans.
    """
    log_path = tmp_path / "pgn.games.jsonl"
    pgn = tmp_path / "match.pgn"
    with pytest.raises(_SimulatedCrash):
        _run_arena(
            monkeypatch, tmp_path, log_path=log_path, crash_after=5, pgn_out=pgn,
        )
    _run_arena(
        monkeypatch, tmp_path, log_path=log_path, resume=True, pgn_out=pgn,
    )
    text = pgn.read_text()
    assert text.count('[PairId "2"]') == 3, "the orphan half is still on file"
    assert text.count('[ResumeReplay "1"]') == 2, (
        "both replayed halves of the orphan pair must be tagged"
    )
    # The untouched pair is present and carries no replay tag.
    assert '[PairId "3"]' in text
    assert '[ResumeReplay "1"]' not in text.split('[PairId "3"]')[1]


# ---------------------------------------------------------------------------
# scripts/match_vs_uci.py
# ---------------------------------------------------------------------------

_MATCH_RESULTS = ("1-0", "1/2-1/2", "0-1", "1-0", "1/2-1/2", "0-1", "1-0", "1-0")
_SUMMARY_PREFIXES = (
    "  A wins", "  draws", "  A losses", "  A score", "  A as W/B",
    "  Score 95% CI", "  Elo",
)


def _load_match_module():
    return load_script_module("match_vs_uci.py", "match_vs_uci_resume_module")


class _FakeEngine:
    def quit(self) -> None:
        return None


def _drive_match(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    argv: list[str],
    *,
    crash_after: int | None = None,
) -> list[int]:
    """Run ``main()`` against fake engines; return the game indices PLAYED."""
    played: list[int] = []

    def _fake_pairing(*, start_board: chess.Board, game: object, **_kw: Any):
        index = int(str(game)) - 1
        played.append(index)
        if crash_after is not None and len(played) > crash_after:
            raise _SimulatedCrash("simulated engine death")
        return module.GameRecord(
            result=_MATCH_RESULTS[index],
            plies=20 + index,
            start_board=start_board.copy(stack=False),
            moves=(),
            move_records=(),
            termination="rules",
        )

    monkeypatch.setattr(
        module, "_open_warm_engine", lambda *_a, **_kw: _FakeEngine(),
    )
    monkeypatch.setattr(module, "_play_one_pairing", _fake_pairing)
    monkeypatch.setattr("sys.argv", ["match_vs_uci.py", *argv])
    module.main()
    return played


def _match_argv(tmp_path: Path, log_path: Path, *, extra: list[str]) -> list[str]:
    return [
        "--engine-a", "engine-a", "--engine-b", "engine-b",
        "--label-a", "A", "--label-b", "B",
        "--games", "8", "--nodes", "100",
        "--openings", str(_openings_file(tmp_path)),
        "--games-out", str(log_path),
        *extra,
    ]


def _summary(capsys: pytest.CaptureFixture[str]) -> list[str]:
    out = capsys.readouterr().out
    return [
        line for line in out.splitlines()
        if line.startswith(_SUMMARY_PREFIXES)
    ]


def test_resumed_match_equals_an_uninterrupted_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same tabulation — W/D/L, score, colour balance, CI — across a crash."""
    module = _load_match_module()
    clean_log = tmp_path / "clean.games.jsonl"
    _drive_match(monkeypatch, module, _match_argv(tmp_path, clean_log, extra=[]))
    clean = _summary(capsys)
    assert clean, "the match printed no summary"

    log_path = tmp_path / "resumed.games.jsonl"
    with pytest.raises(_SimulatedCrash):
        _drive_match(
            monkeypatch, module, _match_argv(tmp_path, log_path, extra=[]),
            crash_after=5,
        )
    capsys.readouterr()
    assert len(read_game_log(log_path).games) == 5, (
        "the games finished before the crash must be on disk"
    )
    _drive_match(
        monkeypatch, module,
        _match_argv(tmp_path, log_path, extra=["--resume"]),
    )
    assert _summary(capsys) == clean


def test_resumed_match_replays_only_the_unfinished_games(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    module = _load_match_module()
    log_path = tmp_path / "partial.games.jsonl"
    with pytest.raises(_SimulatedCrash):
        _drive_match(
            monkeypatch, module, _match_argv(tmp_path, log_path, extra=[]),
            crash_after=5,
        )
    played = _drive_match(
        monkeypatch, module,
        _match_argv(tmp_path, log_path, extra=["--resume"]),
    )
    assert played == [5, 6, 7], (
        f"only the unfinished games may be replayed; played {played}"
    )
    rows = latest_rows_by_key(
        read_game_log(log_path).games, key=lambda r: int(r["game_index"]),
    )
    assert sorted(rows) == list(range(8))
    assert len(read_game_log(log_path).games) == 8


def test_match_pgn_and_move_log_survive_a_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A resume must APPEND to the PGN, not truncate away the crashed games."""
    module = _load_match_module()
    log_path = tmp_path / "pgn.games.jsonl"
    pgn = tmp_path / "out.pgn"
    with pytest.raises(_SimulatedCrash):
        _drive_match(
            monkeypatch, module,
            _match_argv(tmp_path, log_path, extra=["--pgn-out", str(pgn)]),
            crash_after=5,
        )
    assert pgn.read_text().count("[Event ") == 5
    _drive_match(
        monkeypatch, module,
        _match_argv(tmp_path, log_path, extra=["--pgn-out", str(pgn), "--resume"]),
    )
    assert pgn.read_text().count("[Event ") == 8


def test_match_existing_log_without_resume_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    module = _load_match_module()
    log_path = tmp_path / "twice.games.jsonl"
    _drive_match(monkeypatch, module, _match_argv(tmp_path, log_path, extra=[]))
    with pytest.raises(SystemExit) as exc:
        _drive_match(monkeypatch, module, _match_argv(tmp_path, log_path, extra=[]))
    assert "--resume" in str(exc.value)


def test_match_resume_refuses_when_the_opponent_changed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The refusal that matters most: same schedule, different opponent."""
    module = _load_match_module()
    log_path = tmp_path / "opponent.games.jsonl"
    with pytest.raises(_SimulatedCrash):
        _drive_match(
            monkeypatch, module, _match_argv(tmp_path, log_path, extra=[]),
            crash_after=4,
        )
    argv = _match_argv(tmp_path, log_path, extra=["--resume"])
    argv[argv.index("--engine-b") + 1] = "a-different-engine"
    with pytest.raises(SystemExit) as exc:
        _drive_match(monkeypatch, module, argv)
    assert "engine_b" in str(exc.value)


def test_match_schedule_is_a_pure_function_of_the_game_index() -> None:
    """No RNG anywhere in this driver: index -> (colour, opening) and nothing else."""
    module = _load_match_module()
    openings = [chess.Board(fen) for fen in _OPENING_FENS]
    schedule = [module._game_schedule(i, openings) for i in range(8)]
    assert schedule == [
        (True, 0), (False, 0), (True, 1), (False, 1),
        (True, 2), (False, 2), (True, 3), (False, 3),
    ]


@pytest.mark.parametrize("rolling", [True, False])
def test_resume_reaches_the_matched_sims_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rolling: bool,
) -> None:
    """The path the OOM actually crashed: matched_sims, rolling and chunked.

    Every other arena test here drives matched_time, which is the cheap loop to
    stub — and a resume that reached only THAT loop would be a knob accepted and
    then ignored on the production path. This asserts the play function receives
    the REMAINING openings and their GLOBAL pair ids, and that both loops still
    produce the uninterrupted pentanomial.
    """
    import scripts.arena_standard as arena

    log_path = tmp_path / f"sims_{rolling}.games.jsonl"
    side = arena.resolve_search_shape("training")
    monkeypatch.setattr(
        "chess_anti_engine.uci.model_loader.load_model_from_checkpoint",
        lambda *_a, **_kw: object(),
    )
    seen: list[dict[str, Any]] = []

    def _install(crash_after: int | None) -> None:
        play = _fake_matched_time(crash_after=crash_after)

        def _spy(model_c: object, model_r: object, openings: list[chess.Board],
                 **kw: Any) -> list[float]:
            seen.append({
                "n_openings": len(openings),
                "pair_ids": list(kw.get("pair_ids") or []),
                "chunk": kw.get("chunk"),
                "prior": list(kw.get("prior_pair_scores") or []),
            })
            return play(model_c, model_r, openings, **kw)  # pyright: ignore[reportArgumentType]

        for name in (
            "play_paired_games_matched_sims",
            "play_paired_games_matched_sims_rolling",
        ):
            monkeypatch.setattr(arena, name, _spy)

    def _run(*, crash_after: int | None, resume: bool) -> dict:
        _install(crash_after)
        return arena.run_arena(
            candidate="cand.pt", reference="ref.pt", games=8,
            openings_path=None, openings_fen=_openings_file(tmp_path),
            opening_plies=16, mode="matched_sims",
            sims_candidate=8, sims_reference=8, ms_per_move=0, max_plies=40,
            temperature=0.1, gumbel_add_noise=True, device="cpu", seed=7,
            out_path=None, game_log_path=log_path, resume=resume,
            compile_models=False, rolling=rolling, max_concurrent_games=2,
            search_candidate=side, search_reference=side,
        )

    with pytest.raises(_SimulatedCrash):
        _run(crash_after=5, resume=False)
    seen.clear()
    record = _run(crash_after=None, resume=True)

    assert seen, "the resumed run never called a matched_sims play loop"
    played_ids = [pid for call in seen for pid in call["pair_ids"]]
    assert played_ids == [2, 3], (
        "the resumed run must replay the orphan pair and the untouched one, "
        f"with their GLOBAL ids; got {played_ids}"
    )
    assert sum(call["n_openings"] for call in seen) == 2
    if rolling:
        assert seen[0]["prior"] == [2.0, 0.5], (
            "the rolling RUNNING-Elo block must count the resumed pairs"
        )
    else:
        assert [call["chunk"] for call in seen] == [0, 1]
    assert record["pentanomial"] == _EXPECTED_PENTANOMIAL
    assert record["resumed_pairs"] == 2


# ---------------------------------------------------------------------------
# What a resume is allowed to MIX, and what it must say about it
# ---------------------------------------------------------------------------

def _run_arena_matched_sims(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    log_path: Path,
    compile_models: bool,
    crash_after: int | None = None,
    resume: bool = False,
    out_path: Path | None = None,
) -> dict:
    """matched_sims with the two checkpoint loads and torch.compile stubbed.

    The compile-mix tests have to run THIS path: ``compile_models`` reaches
    nothing on the matched_time loop (UCI subprocesses build their own engine),
    which is exactly why a matched_time row records ``"n/a"``.
    """
    import scripts.arena_standard as arena

    side = arena.resolve_search_shape("training")
    monkeypatch.setattr(
        "chess_anti_engine.uci.model_loader.load_model_from_checkpoint",
        lambda *_a, **_kw: object(),
    )
    monkeypatch.setattr("torch.compile", lambda model, **_kw: model)
    play = _fake_matched_time(crash_after=crash_after)
    for name in (
        "play_paired_games_matched_sims",
        "play_paired_games_matched_sims_rolling",
    ):
        monkeypatch.setattr(arena, name, play)
    return arena.run_arena(
        candidate="cand.pt", reference="ref.pt", games=8,
        openings_path=None, openings_fen=_openings_file(tmp_path),
        opening_plies=16, mode="matched_sims",
        sims_candidate=8, sims_reference=8, ms_per_move=0, max_plies=40,
        temperature=0.1, gumbel_add_noise=True, device="cpu", seed=7,
        out_path=out_path, game_log_path=log_path, resume=resume,
        compile_models=compile_models, rolling=True, max_concurrent_games=2,
        search_candidate=side, search_reference=side,
    )


def test_a_resume_across_a_compile_change_is_recorded_as_mixed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """daily_gate_ratchet.sh's retry, exactly: compiled attempt, eager retry.

    The ratchet re-derives ``--compile on|off`` from free VRAM on EVERY attempt
    and always passes --resume, so the two halves of one reported Elo can come
    from two different inference paths. compile is deliberately NOT in the
    resume fingerprint (refusing that retry is worse), so the whole guard is
    that the mix is SAID rather than swallowed.
    """
    log_path = tmp_path / "mix.games.jsonl"
    with pytest.raises(_SimulatedCrash):
        _run_arena_matched_sims(
            monkeypatch, tmp_path, log_path=log_path, compile_models=True,
            crash_after=4,
        )
    assert [row["compile"] for row in read_game_log(log_path).games] == ["on"] * 4
    capsys.readouterr()

    record = _run_arena_matched_sims(
        monkeypatch, tmp_path, log_path=log_path, compile_models=False,
        resume=True,
    )
    err = capsys.readouterr().err
    assert "MIXED torch.compile" in err, err
    assert record["mixed_compile"] is True
    assert record["compile_values"] == ["off", "on"], record["compile_values"]
    assert record["compile"] == "off", "`compile` is what THIS process ran"
    # The mix is real and per-game, not a whole-file property.
    tags = [row["compile"] for row in read_game_log(log_path).games]
    assert tags == ["on"] * 4 + ["off"] * 4


def test_an_unmixed_resume_is_not_flagged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The flag must be silent on the ordinary retry, or it means nothing."""
    log_path = tmp_path / "same.games.jsonl"
    with pytest.raises(_SimulatedCrash):
        _run_arena_matched_sims(
            monkeypatch, tmp_path, log_path=log_path, compile_models=False,
            crash_after=4,
        )
    capsys.readouterr()
    record = _run_arena_matched_sims(
        monkeypatch, tmp_path, log_path=log_path, compile_models=False,
        resume=True,
    )
    assert "MIXED torch.compile" not in capsys.readouterr().err
    assert record["mixed_compile"] is False
    assert record["compile_values"] == ["off"]


def test_matched_time_rows_do_not_invent_a_compile_mix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """compile never reaches a UCI subprocess, so it cannot differ between two.

    ``compile_models`` still arrives at run_arena on this path (main() resolves
    it from free VRAM before it knows the mode), so recording the raw flag
    would report a mix between two attempts that played identically.
    """
    log_path = tmp_path / "uci.games.jsonl"
    record = _run_arena(monkeypatch, tmp_path, log_path=log_path)
    assert [row["compile"] for row in read_game_log(log_path).games] == ["n/a"] * 8
    assert record["mixed_compile"] is False
    assert record["compile_values"] == ["n/a"]


def test_rows_written_before_the_compile_field_existed_read_as_unknown(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An older log must resume, and must still say its provenance is unknown."""
    log_path = tmp_path / "legacy.games.jsonl"
    with pytest.raises(_SimulatedCrash):
        _run_arena_matched_sims(
            monkeypatch, tmp_path, log_path=log_path, compile_models=True,
            crash_after=4,
        )
    # Strip the field the way a log written before it existed would look.
    kept: list[str] = []
    for line in log_path.read_text().splitlines():
        row = json.loads(line)
        row.pop("compile", None)
        kept.append(json.dumps(row, separators=(",", ":")))
    log_path.write_text("\n".join(kept) + "\n")
    capsys.readouterr()

    record = _run_arena_matched_sims(
        monkeypatch, tmp_path, log_path=log_path, compile_models=True,
        resume=True,
    )
    err = capsys.readouterr().err
    assert record["pentanomial"] == _EXPECTED_PENTANOMIAL, "the resume must work"
    assert record["mixed_compile"] is True
    assert record["compile_values"] == ["on", "unknown"]
    assert err.count("MIXED torch.compile") == 1, (
        f"one warning per resume, not one per legacy row; got\n{err}"
    )


# ---------------------------------------------------------------------------
# Write ordering: the JSONL is the commit record
# ---------------------------------------------------------------------------

def test_a_pgn_write_that_dies_leaves_no_jsonl_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A crash BETWEEN the two writes must cost a replay, not a lost game.

    With the JSONL written first, a death in that window leaves a game the
    resume reads as complete and never replays, so the PGN is permanently one
    game short and nothing anywhere can detect it. Writing the JSONL LAST makes
    it the commit record: the window costs a duplicated PGN game (tagged
    ResumeReplay), which is visible and recoverable.
    """
    from chess_anti_engine.eval.arena_pgn import ArenaPgnWriter

    log_path = tmp_path / "order.games.jsonl"
    pgn = tmp_path / "order.pgn"
    real_write = ArenaPgnWriter.write_game
    calls = {"n": 0}

    def _dies_on_the_third(self: ArenaPgnWriter, game: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 3:
            raise _SimulatedCrash("the PGN write died")
        real_write(self, game)

    monkeypatch.setattr(ArenaPgnWriter, "write_game", _dies_on_the_third)
    with pytest.raises(_SimulatedCrash):
        _run_arena(monkeypatch, tmp_path, log_path=log_path, pgn_out=pgn)

    rows = read_game_log(log_path).games
    keys = [(int(r["pair_id"]), int(r["half"])) for r in rows]
    assert keys == [(0, 0), (0, 1)], (
        "the game whose PGN write died must NOT be in the JSONL — the JSONL is "
        f"the commit record and anything in it is never replayed; got {keys}"
    )
    assert pgn.read_text().count("[Event ") == 2

    # And therefore the pair genuinely replays.
    monkeypatch.setattr(ArenaPgnWriter, "write_game", real_write)
    record = _run_arena(
        monkeypatch, tmp_path, log_path=log_path, resume=True, pgn_out=pgn,
    )
    assert record["pentanomial"] == _EXPECTED_PENTANOMIAL
    assert record["resumed_pairs"] == 1
    assert pgn.read_text().count("[Event ") == 8


# ---------------------------------------------------------------------------
# game_log_agrees, read off the DISK
# ---------------------------------------------------------------------------

def test_game_log_agrees_sees_a_row_that_never_reached_the_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The check must touch the file, or it cannot fail for its own reason.

    Comparing two in-memory tallies built at the same call site agrees with
    itself no matter what the disk holds — a guard that cannot fire, wearing
    the name of the thing it does not do.
    """
    import scripts.arena_standard as arena

    log_path = tmp_path / "shortened.games.jsonl"
    play = _fake_matched_time()

    def _play_then_lose_the_last_row(*args: Any, **kwargs: Any) -> list[float]:
        scores = play(*args, **kwargs)
        lines = log_path.read_text().splitlines(keepends=True)
        log_path.write_text("".join(lines[:-1]))
        return scores

    monkeypatch.setattr(
        arena, "play_paired_games_matched_time", _play_then_lose_the_last_row,
    )
    record = arena.run_arena(
        candidate="cand.pt", reference="ref.pt", games=8,
        openings_path=None, openings_fen=_openings_file(tmp_path),
        opening_plies=16, mode="matched_time",
        sims_candidate=32, sims_reference=32, ms_per_move=100, max_plies=300,
        temperature=0.1, gumbel_add_noise=True, device="cpu", seed=7,
        out_path=None, game_log_path=log_path, resume=False,
    )
    assert record["game_log_agrees"] is False, (
        "the log is one game short on disk; the in-memory tally cannot see that"
    )
    assert "does not hold what this run scored" in capsys.readouterr().err
    # The played result still stands — the games happened.
    assert record["pentanomial"] == _EXPECTED_PENTANOMIAL


def test_game_log_agrees_is_true_on_a_clean_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The other half of the guard: it must not fire on a healthy run."""
    record = _run_arena(
        monkeypatch, tmp_path, log_path=tmp_path / "clean_agrees.games.jsonl",
    )
    assert record["game_log_agrees"] is True


# ---------------------------------------------------------------------------
# A no-op resume must not double-count in the shared aggregate
# ---------------------------------------------------------------------------

def test_a_fully_complete_resume_does_not_re_append_the_aggregate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--out is shared and append-only; a replayed no-op would double-count.

    The ratchet passes --resume unconditionally, so re-running a finished
    series is the ordinary case, not an odd one.
    """
    import scripts.arena_standard as arena

    log_path = tmp_path / "aggregate.games.jsonl"
    aggregate = tmp_path / "arena_results.jsonl"
    kwargs: dict[str, Any] = {
        "candidate": "cand.pt", "reference": "ref.pt", "games": 8,
        "openings_path": None, "openings_fen": _openings_file(tmp_path),
        "opening_plies": 16, "mode": "matched_time",
        "sims_candidate": 32, "sims_reference": 32, "ms_per_move": 100,
        "max_plies": 300, "temperature": 0.1, "gumbel_add_noise": True,
        "device": "cpu", "seed": 7,
        "out_path": aggregate, "game_log_path": log_path,
    }
    monkeypatch.setattr(
        arena, "play_paired_games_matched_time", _fake_matched_time(),
    )
    first = arena.run_arena(resume=False, **kwargs)
    assert len(aggregate.read_text().splitlines()) == 1

    def _must_not_run(*_a: object, **_kw: object) -> list[float]:
        raise AssertionError("a fully-resumed run played a game")

    monkeypatch.setattr(arena, "play_paired_games_matched_time", _must_not_run)
    capsys.readouterr()
    again = arena.run_arena(resume=True, **kwargs)
    out = capsys.readouterr().out

    assert len(aggregate.read_text().splitlines()) == 1, (
        "a resume that played zero games appended a second row for one arena"
    )
    assert "not re-appended" in out, out
    assert "[arena] Elo:" in out, "the recomputed summary must still print"
    assert again["pentanomial"] == first["pentanomial"]


# ---------------------------------------------------------------------------
# A log with a header and no games is not two runs
# ---------------------------------------------------------------------------

def test_a_header_only_log_does_not_block_a_fresh_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The crash window before the first game: checkpoint load, compile, warmup.

    The header is written before the first game finishes, so a death in that
    window leaves a log holding zero games. Refusing the retry there protects
    games that do not exist — and the operator's way out was deleting a file,
    which is the habit that loses real logs.
    """
    import scripts.arena_standard as arena

    log_path = tmp_path / "headeronly.games.jsonl"

    def _dies_before_any_game(*_a: object, **_kw: object) -> list[float]:
        raise _SimulatedCrash("died loading the checkpoint")

    monkeypatch.setattr(
        arena, "play_paired_games_matched_time", _dies_before_any_game,
    )
    with pytest.raises(_SimulatedCrash):
        arena.run_arena(
            candidate="cand.pt", reference="ref.pt", games=8,
            openings_path=None, openings_fen=_openings_file(tmp_path),
            opening_plies=16, mode="matched_time",
            sims_candidate=32, sims_reference=32, ms_per_move=100,
            max_plies=300, temperature=0.1, gumbel_add_noise=True,
            device="cpu", seed=7, out_path=None, game_log_path=log_path,
            resume=False,
        )
    parsed = read_game_log(log_path)
    assert parsed.games == [], "fixture: the log must hold a header and no games"

    record = _run_arena(monkeypatch, tmp_path, log_path=log_path)
    assert record["pentanomial"] == _EXPECTED_PENTANOMIAL
    assert record["resumed_pairs"] == 0
    assert len(read_game_log(log_path).games) == 8


def test_a_header_only_log_from_another_run_is_still_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Zero games is not a licence to take over someone else's file."""
    log_path = tmp_path / "foreign.games.jsonl"
    log_path.write_text(
        json.dumps({
            "kind": "header", "version": 1, "driver": "arena_standard",
            "fingerprint": "0123456789ab", "settings": {"mode": "something"},
        }) + "\n"
    )
    with pytest.raises(SystemExit) as exc:
        _run_arena(monkeypatch, tmp_path, log_path=log_path)
    message = str(exc.value)
    assert "ZERO games" in message, message
    assert "0123456789ab" in message, message
    assert log_path.read_text(), "a refused run must not have cleared the file"


# ---------------------------------------------------------------------------
# The refusal must only claim what it measured
# ---------------------------------------------------------------------------

def test_the_collision_refusal_does_not_claim_an_unmeasured_fingerprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """--games-out was compared against nothing before this refusal fired."""
    log_path = tmp_path / "explicit.games.jsonl"
    _run_arena(monkeypatch, tmp_path, log_path=log_path)
    with pytest.raises(SystemExit) as exc:
        _run_arena(monkeypatch, tmp_path, log_path=log_path)
    message = str(exc.value)
    assert "given explicitly" in message, message
    assert "fingerprint matches" not in message, (
        "nothing compared this path's settings to the invocation's; claiming a "
        f"match sends the reader hunting a difference never measured:\n{message}"
    )
    assert "--resume" in message, message
    assert "--games-out" in message, message


def test_the_default_path_refusal_does_claim_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """On the DEFAULT path the claim is true: the name is built from the hash."""
    import scripts.arena_standard as arena

    monkeypatch.setattr(arena, "DEFAULT_GAME_LOG_DIR", tmp_path / "arena_games")
    kwargs: dict[str, Any] = {
        "candidate": "cand.pt", "reference": "ref.pt", "games": 8,
        "openings_path": None, "openings_fen": _openings_file(tmp_path),
        "opening_plies": 16, "mode": "matched_time",
        "sims_candidate": 32, "sims_reference": 32, "ms_per_move": 100,
        "max_plies": 300, "temperature": 0.1, "gumbel_add_noise": True,
        "device": "cpu", "seed": 7,
        "out_path": None, "game_log_path": None, "label": "ladder",
    }
    monkeypatch.setattr(
        arena, "play_paired_games_matched_time", _fake_matched_time(),
    )
    arena.run_arena(resume=False, **kwargs)
    with pytest.raises(SystemExit) as exc:
        arena.run_arena(resume=False, **kwargs)
    message = str(exc.value)
    assert "keyed on this invocation's settings fingerprint" in message, message


# ---------------------------------------------------------------------------
# A PGN that cannot hold the whole match
# ---------------------------------------------------------------------------

def test_arena_warns_when_the_resumed_pgn_lost_the_earlier_games(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Appending to a missing PGN yields a partial file nothing can detect."""
    log_path = tmp_path / "lostpgn.games.jsonl"
    pgn = tmp_path / "lost.pgn"
    with pytest.raises(_SimulatedCrash):
        _run_arena(
            monkeypatch, tmp_path, log_path=log_path, crash_after=5, pgn_out=pgn,
        )
    pgn.unlink()
    capsys.readouterr()
    _run_arena(
        monkeypatch, tmp_path, log_path=log_path, resume=True, pgn_out=pgn,
    )
    err = capsys.readouterr().err
    assert "missing or empty" in err, err
    assert str(pgn) in err
    # A warning, not a refusal: the run still produces its games.
    assert pgn.read_text().count("[Event ") == 4


def test_arena_does_not_warn_when_the_pgn_carries_the_earlier_games(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    log_path = tmp_path / "keptpgn.games.jsonl"
    pgn = tmp_path / "kept.pgn"
    with pytest.raises(_SimulatedCrash):
        _run_arena(
            monkeypatch, tmp_path, log_path=log_path, crash_after=5, pgn_out=pgn,
        )
    capsys.readouterr()
    _run_arena(
        monkeypatch, tmp_path, log_path=log_path, resume=True, pgn_out=pgn,
    )
    assert "missing or empty" not in capsys.readouterr().err


def test_match_warns_when_the_resumed_pgn_lost_the_earlier_games(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_match_module()
    log_path = tmp_path / "matchpgn.games.jsonl"
    pgn = tmp_path / "match_lost.pgn"
    with pytest.raises(_SimulatedCrash):
        _drive_match(
            monkeypatch, module,
            _match_argv(tmp_path, log_path, extra=["--pgn-out", str(pgn)]),
            crash_after=5,
        )
    pgn.unlink()
    capsys.readouterr()
    _drive_match(
        monkeypatch, module,
        _match_argv(tmp_path, log_path, extra=["--pgn-out", str(pgn), "--resume"]),
    )
    err = capsys.readouterr().err
    assert "missing or empty" in err, err
    assert pgn.read_text().count("[Event ") == 3


def test_match_header_only_log_does_not_block_a_fresh_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    module = _load_match_module()
    log_path = tmp_path / "match_headeronly.games.jsonl"

    def _dies_before_any_game(**_kw: Any) -> object:
        raise _SimulatedCrash("engine died during warmup")

    monkeypatch.setattr(module, "_open_warm_engine", lambda *_a, **_kw: _FakeEngine())
    monkeypatch.setattr(module, "_play_one_pairing", _dies_before_any_game)
    monkeypatch.setattr(
        "sys.argv", ["match_vs_uci.py", *_match_argv(tmp_path, log_path, extra=[])],
    )
    with pytest.raises(_SimulatedCrash):
        module.main()
    assert read_game_log(log_path).games == []

    played = _drive_match(
        monkeypatch, module, _match_argv(tmp_path, log_path, extra=[]),
    )
    assert played == list(range(8))


# ---------------------------------------------------------------------------
# elo_vs_sims: a deliberate repeat of the same ladder
# ---------------------------------------------------------------------------

def _ladder_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, extra: list[str],
) -> list[str]:
    import scripts.elo_vs_sims as ladder

    seen: list[str] = []

    def _spy(**kwargs: Any) -> dict[str, Any]:
        seen.append(str(kwargs["label"]))
        return {"elo": 1.0, "elo_ci95": [0.0, 2.0], "pairs": 4}

    monkeypatch.setattr(ladder, "run_arena", _spy)
    monkeypatch.setattr("sys.argv", [
        "elo_vs_sims.py", "--checkpoint", "ck.pt",
        "--search-shape", "training", "--sims", "32,64",
        "--openings", str(_openings_file(tmp_path)),
        *extra,
    ])
    ladder.main()
    return seen


def test_elo_vs_sims_label_suffixes_every_rung(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Without this a repeat ladder can only start by deleting the first one's logs.

    Each rung's game log is named from its label, so two ladders with identical
    settings collide by construction — and elo_vs_sims exposed neither
    --games-out nor --label to break the tie.
    """
    assert _ladder_labels(monkeypatch, tmp_path, extra=[]) == ["elo_vs_sims:64v32"]
    assert _ladder_labels(
        monkeypatch, tmp_path, extra=["--label", "repeat2"],
    ) == ["elo_vs_sims:64v32:repeat2"]


# ---------------------------------------------------------------------------
# Appending after a crash-truncated tail
# ---------------------------------------------------------------------------

def test_a_resume_repairs_a_half_written_final_line(tmp_path: Path) -> None:
    """Appending onto a partial line would fuse two records into a corrupt one.

    read_game_log tolerates a truncated FINAL line and refuses a corrupt MIDDLE
    one, so an append that fuses them turns "lose the game in flight" into
    "lose the whole log" — on exactly the crash the resume exists for.
    """
    path = tmp_path / "partial.games.jsonl"
    with GameLogWriter(path, driver="t", settings={"k": 1}) as log:
        log.write_game({"game_index": 0})
    with path.open("a") as fh:
        fh.write('{"kind": "game", "game_ind')

    with GameLogWriter(
        path, driver="t", settings={"k": 1}, resuming=True,
    ) as log:
        assert log.repaired_truncated_tail is True
        log.write_game({"game_index": 1})

    parsed = read_game_log(path)
    assert [row["game_index"] for row in parsed.games] == [0, 1]
    assert parsed.truncated_tail is False


def test_a_complete_log_is_not_touched_by_the_repair(tmp_path: Path) -> None:
    path = tmp_path / "whole.games.jsonl"
    with GameLogWriter(path, driver="t", settings={"k": 1}) as log:
        log.write_game({"game_index": 0})
    before = path.read_bytes()
    with GameLogWriter(
        path, driver="t", settings={"k": 1}, resuming=True,
    ) as log:
        assert log.repaired_truncated_tail is False
    assert path.read_bytes() == before
