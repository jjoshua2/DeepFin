from __future__ import annotations

import json
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves import POLICY_ENCODING_LC0_1858, move_to_index
from scripts.arena_standard import (
    append_result,
    build_result_record,
    complete_pair_scores,
    game_scores_to_pair_scores,
    pentanomial_counts,
    play_paired_games_matched_sims,
    resolve_search_shape,
    summarize_pentanomial,
)

ROOT = Path(__file__).resolve().parent.parent


def _first_legal_actions(boards: list[chess.Board]) -> list[int]:
    """Stand-in for a search: the first legal move of each board, as an action id.

    A constant such as 0 used to work here because the arena laundered an
    undecodable id into the first legal move anyway. The arena is strict now, so
    a stub must hand back ids that really decode.
    """
    return [int(move_to_index(next(iter(b.legal_moves)), b)) for b in boards]


def test_game_scores_collapse_to_pair_scores():
    # candidate: W,W | W,D | D,D | L,D | L,L | W,L
    games = [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0]
    assert game_scores_to_pair_scores(games) == [2.0, 1.5, 1.0, 0.5, 0.0, 1.0]
    assert pentanomial_counts([2.0, 1.5, 1.0, 0.5, 0.0, 1.0]) == (1, 1, 2, 1, 1)


def test_game_scores_validation():
    with pytest.raises(ValueError, match="even number of games"):
        game_scores_to_pair_scores([1.0])  # odd number of games
    with pytest.raises(ValueError, match="per-game score must be"):
        game_scores_to_pair_scores([1.0, 0.3])  # not a chess score
    with pytest.raises(ValueError, match="pair score must be"):
        pentanomial_counts([1.25])  # not a pair score


def test_pentanomial_golden_symmetric():
    # Hand-constructed symmetric vector: zero Elo, known CI from the
    # pentanomial pair variance.
    summary = summarize_pentanomial((10, 20, 40, 20, 10))
    assert summary.pairs == 100
    assert summary.games == 200
    assert summary.score == pytest.approx(0.5)
    assert summary.score_se == pytest.approx(0.0275240941, abs=1e-9)
    assert summary.elo == pytest.approx(0.0, abs=1e-9)
    lo, hi = summary.elo_ci95
    assert lo == pytest.approx(-37.632858, abs=1e-4)
    assert hi == pytest.approx(+37.632858, abs=1e-4)


def test_pentanomial_golden_asymmetric():
    summary = summarize_pentanomial((30, 30, 20, 15, 5))
    assert summary.score == pytest.approx(0.6625)
    assert summary.score_se == pytest.approx(0.0300199849, abs=1e-9)
    assert summary.elo == pytest.approx(117.164842, abs=1e-4)
    lo, hi = summary.elo_ci95
    assert lo == pytest.approx(73.090400, abs=1e-4)
    assert hi == pytest.approx(165.225436, abs=1e-4)


def test_pentanomial_degenerate_score_has_no_elo():
    summary = summarize_pentanomial((4, 0, 0, 0, 0))  # candidate won every pair
    assert summary.score == pytest.approx(1.0)
    assert summary.elo is None
    assert summary.elo_ci95[1] is None


def _tiny_model() -> torch.nn.Module:
    cfg = ModelConfig(
        kind="transformer",
        embed_dim=32,
        num_layers=1,
        num_heads=2,
        use_smolgen=False,
        policy_encoding=POLICY_ENCODING_LC0_1858,
    )
    return build_model(cfg).eval()


def test_smoke_arena_matched_sims_cpu(tmp_path):
    """4-game paired arena with a tiny random-init lc0_1858 model on CPU."""
    torch.manual_seed(0)
    model_a = _tiny_model()
    torch.manual_seed(1)
    model_b = _tiny_model()

    e4 = chess.Board()
    e4.push_uci("e2e4")
    d4 = chess.Board()
    d4.push_uci("d2d4")
    openings = [e4, d4]

    # NOTE: this now hard-requires the compiled gumbel extension. The `play`
    # shape carries vloss_weight=3, and `pick_moves_for_boards` refuses to run
    # the Python reference with a virtual loss it cannot honour rather than
    # silently dropping it. Consistent with the repo's no-skips rule (the test
    # suite hard-requires every extra); it means a missing `.so` fails here
    # loudly instead of quietly measuring a different search.
    side = resolve_search_shape("play")
    pair_scores = play_paired_games_matched_sims(
        model_a, model_b, openings,
        device="cpu", rng=np.random.default_rng(0),
        sims_candidate=2, sims_reference=2,
        max_plies=10, temperature=1.0, gumbel_add_noise=True,
        search_candidate=side, search_reference=side,
    )
    assert len(pair_scores) == 2
    assert all(s in (0.0, 0.5, 1.0, 1.5, 2.0) for s in pair_scores)

    counts = pentanomial_counts(pair_scores)
    assert sum(counts) == 2
    summary = summarize_pentanomial(counts)
    assert summary.games == 4
    assert 0.0 <= summary.score <= 1.0

    record = build_result_record(
        summary,
        mode="matched_sims",
        candidate="cand.pt",
        reference="ref.pt",
        openings_path="book.pgn.zip",
        opening_plies=16,
        sims_candidate=2,
        sims_reference=2,
        ms_per_move=None,
        temperature=1.0,
        gumbel_add_noise=True,
        max_plies=10,
        seed=0,
        device="cpu",
        duration_s=0.0,
    )
    out = tmp_path / "arena_results.jsonl"
    append_result(record, out)

    loaded = json.loads(out.read_text().splitlines()[-1])
    assert loaded["mode"] == "matched_sims"
    assert loaded["candidate"] == "cand.pt"
    assert loaded["reference"] == "ref.pt"
    assert loaded["games"] == 4
    assert loaded["pairs"] == 2
    assert sum(loaded["pentanomial"].values()) == 2
    assert set(loaded["pentanomial"]) == {"WW", "WD_DW", "DD_WL", "LD_DL", "LL"}
    for key in ("ts", "git_sha", "config_hash", "score", "elo", "elo_ci95", "seed"):
        assert key in loaded


@pytest.mark.parametrize("play_fn_name", [
    "play_paired_games_matched_sims",
    "play_paired_games_matched_sims_rolling",
])
def test_both_arena_paths_pass_each_side_its_own_vloss_and_target_batch(
    monkeypatch, play_fn_name: str,
) -> None:
    """Both arena loops must hand each side ITS OWN vloss_weight/target_batch.

    The rolling loop is the DEFAULT (``run_arena(rolling=True)``) and is what
    ``daily_gate_ratchet.sh`` runs, but every other test here drives the chunked
    loop. A review of #286 mutated the rolling loop to drop both kwargs and the
    whole suite stayed green -- the guard covered the path that does not run.

    Asserting per-side (play on one side, training on the other, which differ in
    vloss_weight 3 vs 1) also catches the subtler bug where both sides are fed
    one side's value.
    """
    import scripts.arena_standard as arena

    cand = resolve_search_shape("play")       # vloss_weight 3
    ref = resolve_search_shape("training")    # vloss_weight 1
    assert cand.vloss_weight != ref.vloss_weight, "shapes must differ or the test is vacuous"

    # (vloss_weight, target_batch_present, target_batch, topk) per call.
    seen: list[tuple[int, bool, int, float]] = []

    def _recording_pick(_model, boards, **kw):
        missing = [
            k for k in ("gumbel_vloss_weight", "gumbel_target_batch")
            if k not in kw
        ]
        # Report the DROPPED kwarg by name rather than dying on a KeyError --
        # this assertion is the whole point of the test, so it must say what
        # broke rather than leaving the next reader a traceback to decode.
        assert not missing, (
            f"{play_fn_name} did not pass {missing} to pick_moves_for_boards; "
            "the side's search shape is silently not reaching the search"
        )
        seen.append((
            int(kw["gumbel_vloss_weight"]),
            True,
            int(kw["gumbel_target_batch"]),
            float((kw.get("gumbel_overrides") or {})["topk"]),
        ))
        return _first_legal_actions(boards)

    monkeypatch.setattr(
        "chess_anti_engine.selfplay.match.pick_moves_for_boards", _recording_pick,
    )

    e4 = chess.Board()
    e4.push_uci("e2e4")
    d4 = chess.Board()
    d4.push_uci("d2d4")

    play_fn = getattr(arena, play_fn_name)
    kwargs: dict[str, object] = {
        "device": "cpu",
        "rng": np.random.default_rng(0),
        "sims_candidate": 2,
        "sims_reference": 2,
        "max_plies": 6,
        "temperature": 1.0,
        "gumbel_add_noise": False,
        "search_candidate": cand,
        "search_reference": ref,
    }
    if play_fn_name.endswith("_rolling"):
        kwargs["pool_size"] = 4
        kwargs["report_every"] = 1000
    play_fn(None, None, [e4, d4], **kwargs)

    assert seen, f"{play_fn_name} never called pick_moves_for_boards"
    by_vloss = sorted({row[0] for row in seen})
    expected_vloss = sorted({int(cand.vloss_weight), int(ref.vloss_weight)})
    assert by_vloss == expected_vloss, (
        f"{play_fn_name}: each side must get its OWN vloss_weight; "
        f"saw {by_vloss}, expected {expected_vloss}"
    )
    # target_batch must be threaded too. Both shapes use 0 today, so assert the
    # kwarg is PRESENT -- an omitted kwarg would also read as 0 downstream.
    assert all(row[1] for row in seen), (
        f"{play_fn_name}: gumbel_target_batch was not passed at all"
    )
    assert all(row[2] == 0 for row in seen)
    # each call's vloss must pair with the topk of the shape it came from, which
    # catches feeding one side's vloss to both.
    for vloss, _present, _tb, topk in seen:
        expect = cand if vloss == int(cand.vloss_weight) else ref
        assert topk == float(expect.gumbel["topk"]), (
            f"{play_fn_name}: side mismatch -- vloss {vloss} came with "
            f"topk {topk}, expected {expect.gumbel['topk']}"
        )


# ---------------------------------------------------------------------------
# A capped run must survive as DATA. 2026-07-30 and 07-31: the ratchet produced
# no CSV row at all, because the arena was SIGKILLed by the caller's `timeout`
# and the one block it had computed died in the block-buffered stdout pipe.
# Buffering loses exactly the LAST block (each flushed line pushes what came
# before it), so only a run too slow to print a second block loses everything --
# which is why 07-28/07-29 recorded rows off the same code.
# ---------------------------------------------------------------------------

def test_print_summary_survives_an_unflushed_process_death(tmp_path):
    """``print_summary`` must flush, or a killed run reports nothing.

    Reproduces the exact failure: stdout redirected to a FILE (block-buffered,
    not line-buffered), one RUNNING-Elo block computed, then the process dies
    without stdio cleanup. ``os._exit`` is used because that is what SIGKILL
    does to the buffer.

    The last byte of data/ratchet/arena_2026-07-31_vs_prev.log is the ``RUNNING
    Elo after 6 complete pairs:`` header -- the reading was computed and thrown
    away. Negative control: dropping ``flush=True`` from print_summary's last
    line makes this test fail.

    ``137`` here is chosen by the child's own ``os._exit(137)`` -- it stands for
    the SIGKILL the ratchet's ``timeout -k`` delivers, but nothing in this test
    is timed, so the exit code is deterministic and any other value means the
    child died before it got there. The child therefore gets an explicit
    ``PYTHONPATH``: it is launched by path, so ``sys.path[0]`` is ``tmp_path``
    and ``cwd`` alone does NOT make ``scripts`` importable. Relying on the
    ambient ``PYTHONPATH`` made this pass only under ``PYTHONPATH=. pytest``
    and fail under the bare ``python -m pytest`` CI runs.
    """
    import os
    import subprocess
    import sys

    script = tmp_path / "die_mid_summary.py"
    script.write_text(
        "import os\n"
        "from scripts.arena_standard import (\n"
        "    pentanomial_counts, print_summary, summarize_pentanomial,\n"
        ")\n"
        "ready = [2.0, 1.0, 1.0, 0.0, 1.0, 2.0]\n"
        "print('[arena] RUNNING Elo after %d complete pairs:' % len(ready), flush=True)\n"
        "print_summary(summarize_pentanomial(pentanomial_counts(ready)))\n"
        "os._exit(137)\n"
    )
    log = tmp_path / "arena.log"
    with log.open("w") as fh:
        rc = subprocess.call(
            [sys.executable, str(script)], stdout=fh, stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env={"PYTHONPATH": str(ROOT), "PATH": os.environ.get("PATH", ""),
                 "HOME": os.environ.get("HOME", "")},
        )
    text = log.read_text()
    # Report what the child actually said; a bare `assert rc == 137` reads like
    # a timing flake when it is really an import that never reached os._exit.
    assert rc == 137, f"child exited {rc} instead of reaching os._exit(137):\n{text}"
    # This is the literal pattern daily_gate_ratchet.sh greps for.
    elo_lines = [ln for ln in text.splitlines() if ln.startswith("[arena] Elo:")]
    assert elo_lines, (
        "print_summary's block did not reach the file before the process died, "
        "so a wall-clock-capped arena yields NO reading:\n" + text
    )


def test_complete_pair_scores_drops_unfinished_pairs_rather_than_imputing():
    """An unfinished game must remove its pair, never be scored 0.5.

    The rolling loop used to return ``[s if s is not None else 0.5 ...]``, which
    is harmless only while every game finishes. Under ``--max-seconds`` it would
    turn in-flight and never-started games into draws and report the FULL
    requested pair count -- a games column that does not mean what its name says.
    """
    # pair 0 complete (W,L), pair 1 half-played, pair 2 complete (W,W),
    # pair 3 never started.
    scores: list[float | None] = [1.0, 0.0, 1.0, None, 1.0, 1.0, None, None]
    assert complete_pair_scores(scores) == [1.0, 2.0]
    assert complete_pair_scores([None] * 8) == []
    # Agrees with the strict helper whenever nothing is missing.
    full = [1.0, 0.0, 0.5, 0.5]
    assert complete_pair_scores(list(full)) == game_scores_to_pair_scores(full)


def test_rolling_loop_stops_at_the_deadline_with_no_imputed_pairs(monkeypatch):
    """``deadline`` must stop the rolling loop and return only FINISHED pairs.

    A deadline already in the past exits before a single ply, so the honest
    answer is "no pairs". The pre-fix code would have returned ``[1.0, 1.0]``
    (both openings imputed as draw/draw), which is exactly the fabricated
    reading this guard exists to prevent.
    """
    import time as _time

    import scripts.arena_standard as arena

    calls: list[int] = []

    def _never_should_run(_model, boards, **_kw):
        calls.append(len(boards))
        return _first_legal_actions(boards)

    monkeypatch.setattr(
        "chess_anti_engine.selfplay.match.pick_moves_for_boards", _never_should_run,
    )

    e4 = chess.Board()
    e4.push_uci("e2e4")
    d4 = chess.Board()
    d4.push_uci("d2d4")
    side = resolve_search_shape("training")

    pair_scores = arena.play_paired_games_matched_sims_rolling(
        None, None, [e4, d4],
        device="cpu", rng=np.random.default_rng(0),
        sims_candidate=2, sims_reference=2,
        max_plies=200, temperature=1.0, gumbel_add_noise=False,
        search_candidate=side, search_reference=side,
        pool_size=4, report_every=1000,
        deadline=_time.time() - 1.0,
    )
    assert pair_scores == [], (
        "an expired deadline must yield zero pairs, not imputed draws"
    )
    assert not calls, "the loop played a ply after its deadline had passed"


def test_run_arena_records_truncation_so_a_capped_row_is_readable():
    """``games``/``pairs`` are what was scored; the record must say so.

    A 40-game row that requested 200 is a valid small sample. Without
    ``games_requested``/``truncated`` a later reader cannot tell it from a
    200-game claim -- the ratchet CSV hit exactly this on 2026-07-26.
    """
    summary = summarize_pentanomial((2, 3, 5, 3, 2))
    record = build_result_record(
        summary,
        mode="matched_sims",
        candidate="cand.pt",
        reference="ref.pt",
        openings_path="book.pgn.zip",
        opening_plies=16,
        sims_candidate=32,
        sims_reference=32,
        ms_per_move=None,
        temperature=0.1,
        gumbel_add_noise=True,
        max_plies=300,
        seed=42,
        device="cuda",
        duration_s=855.0,
        games_requested=200,
        max_seconds=855.0,
        truncated=True,
    )
    assert record["games"] == 30
    assert record["games_requested"] == 200
    assert record["truncated"] is True
    assert record["max_seconds"] == 855.0


def test_deadline_is_checked_after_the_reap_not_before(monkeypatch):
    """Games that finished on the last played ply must still be scored.

    The deadline check originally sat at the TOP of the loop, before the reap,
    so up to ``pool_size`` already-decided games were thrown away with the
    process: the 2026-07-31 proof run banked 100 finished games and scored 96.
    Reaping first cannot fabricate anything -- ``_record`` still only runs for a
    board with a real result -- so the recovery is free.

    Pair 0 starts already stalemated (both colorings decide on arrival); pair 1
    is a live position that cannot finish. With an expired deadline the honest
    answer is therefore exactly one pair, scored 0.5+0.5. Checking the deadline
    before the reap returns ``[]`` instead.
    """
    import time as _time

    import scripts.arena_standard as arena

    def _pick(_model, boards, **_kw):
        return _first_legal_actions(boards)

    monkeypatch.setattr(
        "chess_anti_engine.selfplay.match.pick_moves_for_boards", _pick,
    )

    finished = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")  # stalemate
    assert finished.is_game_over(claim_draw=True), "fixture must start decided"
    live = chess.Board()
    live.push_uci("e2e4")
    side = resolve_search_shape("training")

    pair_scores = arena.play_paired_games_matched_sims_rolling(
        None, None, [finished, live],
        device="cpu", rng=np.random.default_rng(0),
        sims_candidate=2, sims_reference=2,
        max_plies=200, temperature=1.0, gumbel_add_noise=False,
        search_candidate=side, search_reference=side,
        pool_size=4, report_every=1000,
        deadline=_time.time() - 1.0,
    )
    assert pair_scores == [1.0], (
        "the decided pair must be reaped and scored before the deadline breaks "
        f"the loop; got {pair_scores}"
    )


def _run_arena_matched_time(monkeypatch, tmp_path, *, games: int, max_seconds=None):
    """Drive run_arena's matched_time path with the UCI match stubbed out.

    matched_time loads no models, so this exercises the real orchestration
    (openings, deadline, truncation, record) without spawning engines.
    """
    import scripts.arena_standard as arena

    fen_file = tmp_path / "openings.fen"
    fen_file.write_text(
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\n"
        "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1\n"
    )
    seen: dict[str, object] = {}

    def _stub(_cand, _ref, openings, **kw):
        seen.update(kw)
        seen["n_openings"] = len(openings)
        return [1.0] * len(openings)

    monkeypatch.setattr(arena, "play_paired_games_matched_time", _stub)
    record = arena.run_arena(
        candidate="cand.pt", reference="ref.pt", games=games,
        openings_path=None, openings_fen=fen_file, opening_plies=16,
        mode="matched_time", sims_candidate=32, sims_reference=32,
        ms_per_move=100, max_plies=300, temperature=0.1,
        gumbel_add_noise=True, device="cpu", seed=0, out_path=None,
        max_seconds=max_seconds,
    )
    return record, seen


def test_max_seconds_reaches_the_matched_time_loop(monkeypatch, tmp_path):
    """`--max-seconds` must not be accepted and then ignored.

    matched_time REFUSES the search-shape family because those change the ruler
    and cannot apply to UCI subprocesses. A wall-clock budget is different: it
    changes nothing about the ruler, so it is honoured rather than rejected --
    but only if it actually reaches the loop. Accepting it and running unbounded
    would be the accepted-then-ignored defect this script was just fixed for.
    """
    _record, seen = _run_arena_matched_time(
        monkeypatch, tmp_path, games=4, max_seconds=900.0,
    )
    assert "deadline" in seen, (
        "run_arena did not pass a deadline to play_paired_games_matched_time; "
        "--max-seconds is inert under matched_time"
    )
    assert seen["deadline"] is not None
    _record2, seen2 = _run_arena_matched_time(monkeypatch, tmp_path, games=4)
    assert seen2["deadline"] is None, "no --max-seconds must mean no deadline"


def test_truncated_is_measured_against_the_openings_actually_loaded(
    monkeypatch, tmp_path,
):
    """A short FEN list is not a truncated run.

    `load_fen_openings` uses ALL rows when the file holds fewer than
    ``games // 2``, so comparing the completed pair count with ``n_pairs``
    stamped ``truncated: True`` on every complete `--openings-fen` run with a
    small seed file -- a flag that would have said "small sample, distrust me"
    about a run that played every position it was given.
    """
    record, seen = _run_arena_matched_time(monkeypatch, tmp_path, games=100)
    assert seen["n_openings"] == 2, "fixture must supply fewer rows than games//2"
    assert record["pairs"] == 2
    assert record["games_requested"] == 100
    assert record["truncated"] is False, (
        "every opening in the file was played; that is a complete run"
    )
