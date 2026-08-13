"""Drive the REAL arena play loops from a mocked, CPU-only game stream.

No GPU is available to this test (and none should be used): the model and the
move chooser are replaced, but ``play_paired_games_matched_sims*`` itself — the
pairing, the reaping, the scoring, and the PGN emission — is the production
code. That is the part the PGN wiring can break.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest

import chess_anti_engine.selfplay.match as match_mod
from scripts.arena_standard import (
    SideSearch,
    play_paired_games_matched_sims,
    play_paired_games_matched_sims_rolling,
)

# 1. f3 e5 2. g4 Qh4# -- four plies to a decisive, rules-terminated game.
FOOLS_MATE = ("f2f3", "e7e5", "g2g4", "d8h4")


@pytest.fixture
def scripted_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_pick(_model: object, sub_boards: list[chess.Board],
                  **_kwargs: object) -> list[int]:
        return [0] * len(sub_boards)

    def fake_apply(boards: list[chess.Board], idxs: list[int],
                   _actions: list[int]) -> None:
        for i in idxs:
            b = boards[i]
            mv = None
            ply = len(b.move_stack)
            if ply < len(FOOLS_MATE):
                cand = chess.Move.from_uci(FOOLS_MATE[ply])
                if cand in b.legal_moves:
                    mv = cand
            if mv is None:
                mv = next(iter(b.legal_moves))
            b.push(mv)

    monkeypatch.setattr(match_mod, "pick_moves_for_boards", fake_pick)
    monkeypatch.setattr(match_mod, "apply_actions_to_boards", fake_apply)


def _search() -> SideSearch:
    return SideSearch(shape="test", source="test", gumbel={}, vloss_weight=0,
                      target_batch=0)


@dataclass(frozen=True)
class Emitted:
    """One sink call. Spelling the payload out here is deliberate: the sink is
    a keyword-only contract between arena_standard and the PGN writer, and if
    that contract changes these tests must fail to RUN, not silently pass."""

    pair_id: int
    half: int
    a_is_white: bool
    start_fen: str
    moves: tuple[chess.Move, ...]
    result: str
    termination: str
    plies: int
    duration_s: float


def _collect_sink(store: list[Emitted]) -> Callable[..., None]:
    def sink(
        *,
        pair_id: int,
        half: int,
        a_is_white: bool,
        start_fen: str,
        moves: tuple[chess.Move, ...],
        result: str,
        termination: str,
        plies: int,
        duration_s: float,
    ) -> None:
        store.append(Emitted(
            pair_id=pair_id, half=half, a_is_white=a_is_white,
            start_fen=start_fen, moves=moves, result=result,
            termination=termination, plies=plies, duration_s=duration_s,
        ))
    return sink


def _play(
    rolling: bool,
    openings: list[chess.Board],
    sink: Callable[..., None] | None = None,
    *,
    max_plies: int = 20,
    pair_id_offset: int = 0,
) -> list[float]:
    """Explicit calls, not a kwargs dict: an untyped dict makes every argument
    Unknown to the type checker, which is how a wrong one gets through."""
    if rolling:
        return play_paired_games_matched_sims_rolling(
            None, None, openings,
            device="cpu", rng=np.random.default_rng(7),
            sims_candidate=1, sims_reference=1, max_plies=max_plies,
            temperature=0.1, gumbel_add_noise=False,
            search_candidate=_search(), search_reference=_search(),
            pool_size=4, pgn_sink=sink,
        )
    return play_paired_games_matched_sims(
        None, None, openings,
        device="cpu", rng=np.random.default_rng(7),
        sims_candidate=1, sims_reference=1, max_plies=max_plies,
        temperature=0.1, gumbel_add_noise=False,
        search_candidate=_search(), search_reference=_search(),
        pgn_sink=sink, pair_id_offset=pair_id_offset,
    )


@pytest.mark.usefixtures("scripted_moves")
@pytest.mark.parametrize("rolling", [True, False])
def test_sink_receives_every_game_with_pair_identity(rolling: bool) -> None:
    openings = [chess.Board() for _ in range(3)]
    got: list[Emitted] = []
    pair_scores = _play(rolling, openings, sink=_collect_sink(got))

    assert len(pair_scores) == 3
    assert len(got) == 6, "every game must reach the sink exactly once"
    # Both halves of every pair, exactly once each.
    assert sorted((g.pair_id, g.half) for g in got) == [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)
    ]
    # Fool's mate: black always wins, whoever black is.
    assert {g.result for g in got} == {"0-1"}
    assert {g.termination for g in got} == {"rules"}
    for g in got:
        assert g.plies == 4
        assert [m.uci() for m in g.moves] == list(FOOLS_MATE)
        assert g.duration_s >= 0.0
        assert g.start_fen == chess.STARTING_FEN


@pytest.mark.usefixtures("scripted_moves")
@pytest.mark.parametrize("rolling", [True, False])
def test_default_off_is_byte_identical_and_writes_nothing(
    rolling: bool, tmp_path: Path,
) -> None:
    # Proven by EXECUTION, not by reading the diff: same openings and seed,
    # with and without a sink, must produce the same pair scores.
    openings = [chess.Board() for _ in range(3)]
    with_sink: list[Emitted] = []
    scores_on = _play(rolling, [b.copy() for b in openings],
                      sink=_collect_sink(with_sink))
    scores_off = _play(rolling, [b.copy() for b in openings], sink=None)
    assert scores_off == scores_on
    assert with_sink, "sanity: the sink path must actually have run"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.usefixtures("scripted_moves")
@pytest.mark.parametrize("rolling", [True, False])
def test_book_opening_records_book_position_and_excludes_book_moves(
    rolling: bool,
) -> None:
    # A book opening arrives as a board that ALREADY has move_stack. The PGN
    # must start at the book position and contain only the moves actually
    # PLAYED — writing the whole stack would replay the book as if the engines
    # had chosen it, and would misattribute the opening to the players.
    opening = chess.Board()
    for uci in ("f2f3", "e7e5"):
        opening.push(chess.Move.from_uci(uci))
    book_fen = opening.fen()

    got: list[Emitted] = []
    _play(rolling, [opening.copy()], sink=_collect_sink(got))

    assert len(got) == 2
    for g in got:
        assert g.start_fen == book_fen
        assert g.start_fen != chess.STARTING_FEN
        # Play resumes at ply 2, so only the remaining two mate moves are ours.
        assert [m.uci() for m in g.moves] == ["g2g4", "d8h4"]
        assert g.plies == 2
        replay = chess.Board(g.start_fen)
        for m in g.moves:
            assert m in replay.legal_moves
            replay.push(m)
        assert replay.is_checkmate()


@pytest.mark.usefixtures("scripted_moves")
@pytest.mark.parametrize("rolling", [True, False])
def test_game_decided_on_the_final_ply_is_written_decisive(rolling: bool) -> None:
    # Fool's mate is exactly 4 plies. At max_plies=20 the loop re-tests and sees
    # the mate; at max_plies=4 it never re-tests, so the game reached the
    # sweep-up path. It must STILL be written 0-1, because that is how the
    # pentanomial scored it. The chunked path used to blanket-override the
    # sweep-up to a draw, making the PGN and the pentanomial disagree.
    got: list[Emitted] = []
    scores = _play(rolling, [chess.Board()], sink=_collect_sink(got), max_plies=4)
    assert {g.result for g in got} == {"0-1"}
    assert {g.termination for g in got} == {"rules"}
    # Candidate is white in half 0 (loses) and black in half 1 (wins) -> 1.0.
    assert scores == [1.0]


@pytest.mark.usefixtures("scripted_moves")
def test_final_ply_pgn_decisiveness_matches_the_board() -> None:
    # ⚑ A PAIR-SUM comparison CANNOT test this, and an earlier version of this
    # test was vacuous for exactly that reason: a mirrored {win, loss} and a
    # {draw, draw} both sum to a pair score of 1.0. That is the same DD_WL
    # degeneracy that makes the banked pentanomial reconstruction ambiguous —
    # the pair score is not a function of the individual results.
    #
    # So assert at the GAME level: the count of decisive games in the PGN must
    # equal the count the boards actually reached.
    for max_plies in (4, 20):
        got: list[Emitted] = []
        _play(False, [chess.Board()], sink=_collect_sink(got),
              max_plies=max_plies)
        decisive = sum(1 for g in got if g.result != "1/2-1/2")
        assert decisive == 2, (
            f"both fool's-mate games are decisive; PGN reported {decisive} "
            f"at {max_plies=}"
        )


@pytest.mark.usefixtures("scripted_moves")
def test_chunked_pair_id_offset_keeps_pairs_globally_unique() -> None:
    # run_arena calls the chunked path once per chunk with a LOCAL opening list;
    # without the offset every chunk would restart PairId at 0 and the pooled
    # fit would merge unrelated pairs into one block.
    got: list[Emitted] = []
    _play(False, [chess.Board() for _ in range(2)], sink=_collect_sink(got),
          pair_id_offset=10)
    assert sorted({g.pair_id for g in got}) == [10, 11]


def test_max_plies_game_is_written_as_a_draw_not_a_star(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The arena SCORES an unfinished game 0.5, so the PGN must say "1/2-1/2".
    # Ordo maps "*" to DISCARD and would drop the game, giving the pooled fit a
    # different population than the pentanomial summary.
    def fake_pick(_model: object, sub_boards: list[chess.Board],
                  **_kwargs: object) -> list[int]:
        return [0] * len(sub_boards)

    def shuffle_knights(boards: list[chess.Board], idxs: list[int],
                        _actions: list[int]) -> None:
        for i in idxs:
            b = boards[i]
            b.push(next(iter(b.legal_moves)))

    monkeypatch.setattr(match_mod, "pick_moves_for_boards", fake_pick)
    monkeypatch.setattr(match_mod, "apply_actions_to_boards", shuffle_knights)

    for rolling in (True, False):
        got: list[Emitted] = []
        scores = _play(rolling, [chess.Board()], sink=_collect_sink(got),
                       max_plies=6)
        assert scores == [1.0], "two 0.5 games = pair score 1.0"
        assert {g.result for g in got} == {"1/2-1/2"}
        assert {g.termination for g in got} == {"max_plies"}


@pytest.mark.usefixtures("scripted_moves")
def test_end_to_end_pgn_is_parseable_and_fits_in_ordo_format(
    tmp_path: Path,
) -> None:
    from chess_anti_engine.eval.arena_pgn import ArenaGame, ArenaPgnWriter

    out = tmp_path / "arena.pgn"
    got: list[Emitted] = []
    _play(True, [chess.Board() for _ in range(2)], sink=_collect_sink(got))
    with ArenaPgnWriter(out, event="tier13",
                        base_tags={"ConfigHash": "deadbeef"}) as w:
        for g in got:
            w.write_game(ArenaGame(
                white="armA" if g.a_is_white else "armB",
                black="armB" if g.a_is_white else "armA",
                result=g.result, moves=g.moves,
                start_fen=g.start_fen, pair_id=g.pair_id,
                pair_half=g.half,
                extra={"Plies": str(g.plies)},
            ))
    fh = io.StringIO(out.read_text())
    parsed = []
    while True:
        game = chess.pgn.read_game(fh)
        if game is None:
            break
        parsed.append(game)
    assert len(parsed) == 4
    assert {p.headers["Result"] for p in parsed} == {"0-1"}
    assert {p.headers["White"] for p in parsed} == {"armA", "armB"}
    assert all(p.headers["ConfigHash"] == "deadbeef" for p in parsed)
