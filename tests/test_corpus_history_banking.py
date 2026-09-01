"""THE EQUIVALENCE GATE: a reconstructed corpus row encodes exactly like live play.

⚑⚑ WHAT THIS FILE IS FOR.  The NNUE-bootstrap corpus banks a position; the net
that trains on it plays with a search whose root carries eight real history
frames.  Measured 2026-09-01 (``docs/experiment_ledger.md``, the history-
materiality probe) the champion flips **46.5%** of its top-1 moves and loses
**+20.2 cp** of regret between those two input distributions -- so a corpus
built on the zero-history one is a corpus built on inputs production never sees.
Row schema 2 banks the move window that closes the gap, and this file is the
proof that it actually closes it: not "the history is plausible", but

    deriver planes of the RECONSTRUCTED board
      ==  encode_cboard(CBoard.from_board(LIVE board), ...)

bit for bit over the complete (175, 8, 8) tensor -- every history frame, every
repetition plane, castling, side-to-move, rule-50, ep and every feature plane.

⚑ THE RIGHT-HAND SIDE IS THE PLAY PATH, not the python encoder.  The UCI search
encodes its root with exactly ``encode_cboard(CBoard.from_board(board), ...)``
(``chess_anti_engine/uci/root_parallel_gumbel.py``, ``uci/search.py``), whose
repetition planes come from per-slot flags recorded in C.  The deriver calls
``encode_position``, whose repetition planes come from python-chess's own scan.
Those are two different implementations of the same claim, and pinning them
here is what lets the deriver keep the cheaper call.

⚑ Live SELFPLAY rows are encoded from a THIRD construction -- a CBoard built
once with ``from_board`` at the opening and advanced with ``push_index``
(``selfplay/state.py``, ``selfplay/stockfish_turn.py``) -- so "the play path" is
not one object.  ``test_the_reference_encoding_is_also_what_selfplay_rows_are
_written_from`` measures that the pushed and the reconstructed constructions
agree, which is what makes the reference above the right one.

⚑ THE NEGATIVE HALF IS NOT OPTIONAL.  ``test_the_schema_1_bare_fen_path_differs
_from_live_on_every_position_with_history`` asserts the OLD behaviour still
fails the same comparison -- a gate that passes both ways is a gate that cannot
fail.

The case list is the spec's, and each case is asserted to actually EXERCISE what
it is named for (a "3-fold repetition" case whose repetition planes are all zero
would pass every equality above while testing nothing).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import rep_fix
from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.moves.encode import move_to_index
from chess_anti_engine.selfplay.opening import OpeningConfig, sample_starting_board
from scripts import derive_corpus_targets as derive
from scripts import gen_sf_rooted_corpus as corpus

#: The 8 encoded frames are 12 piece planes + 1 repetition plane each.
_PLANES_PER_SLOT = 13
_PIECE_PLANES_PER_SLOT = 12
_HISTORY_SLOTS = 8
#: Plane index of slot ``i``'s repetition plane.
REP_PLANES = [slot * _PLANES_PER_SLOT + _PIECE_PLANES_PER_SLOT
              for slot in range(_HISTORY_SLOTS)]
#: LC0-root legacy-meta ep-file plane, written from the RAW ``ep_square`` with
#: no legality test -- which is why the banked root keeps a pseudo-legal square.
EP_PLANE = 110


@pytest.fixture(autouse=True)
def production_rep_fix() -> None:
    """``history_rep_fix: true``, which is what production selfplay encodes under.

    The flag lives as a C global per extension and is recorded at board
    construction time, so it is applied before any ``CBoard`` in this file is
    built (``encoding/rep_fix.py``'s ordering contract).  NOT torn down: it is
    the production value, the flip guard refuses an unsafe restore anyway, and
    every other test that cares sets its own.
    """
    rep_fix.apply(True)


# ── the two encoders ─────────────────────────────────────────────────────────


def live_planes(board: chess.Board) -> np.ndarray:
    """EXACTLY what live search encodes for this board -- the C path, verbatim."""
    return np.asarray(
        encode_cboard(
            CBoard.from_board(board),
            input_history_encoding=derive.INPUT_HISTORY_ENCODING,
            input_extra_features=derive.INPUT_EXTRA_FEATURES,
        ),
        dtype=np.float32,
    )


def deriver() -> derive.TargetDeriver:
    """A ``TargetDeriver`` whose only used surface here is ``_encode``."""
    return derive.TargetDeriver(
        derive.DeriveOptions(
            scheme=derive.parse_scheme("uniform-d9"),
            temp=1.0,
            cp_slope=1.0,
            cp_draw_width=1.0,
            limit=0,
            seed=0,
            rows_per_shard=8,
            max_envelope_misses=0,
        ),
    )


DERIVER = deriver()


def derived_planes(board: chess.Board) -> np.ndarray:
    """The planes the corpus would carry for this board."""
    return DERIVER._encode(board)


# ── the schema-2 round trip, through the production helpers ──────────────────


def row_for(board: chess.Board, *, ply: int = 0) -> dict[str, Any]:
    """A schema-2 row built by the GENERATOR's own helper, not a copy of it."""
    history = corpus.history_for(board)
    return {
        "schema": corpus.ROW_SCHEMA,
        "fen": board.fen(),
        "game_id": 0,
        "ply": ply,
        **history.as_row_fields(),
    }


def round_trip(board: chess.Board) -> chess.Board:
    """Bank ``board`` as the generator does, rebuild it as the deriver does."""
    return derive.board_from_row(row_for(board))


# ── the case list ────────────────────────────────────────────────────────────


def positions(start: str | chess.Board, moves: str) -> list[chess.Board]:
    """Every position of a line, INCLUDING the start, each with its own stack."""
    board = chess.Board(start) if isinstance(start, str) else start.copy(stack=True)
    out = [board.copy(stack=True)]
    for uci in moves.split():
        board.push_uci(uci)
        out.append(board.copy(stack=True))
    return out


#: ⚑⚑ A REPETITION WHOSE EARLIER OCCURRENCE IS MORE THAN 7 PLIES BACK, which is
#: the case the whole root definition exists for.  Each side cycles TWO knights
#: out and back, so one cycle is EIGHT plies: the position after ply 10 has
#: occurred before at ply 2 -- outside the 8 encoded frames but inside the
#: reversible run -- and after ply 18 it is a 3-fold.  The opening ``e4 e5``
#: puts an irreversible move at the front, so the root is a halfmove-clock-0
#: position rather than the game start.  A window of "the last 7 moves" cannot
#: see the partner at all, which is exactly what the first mutant checks.
FAR_REPETITION = (
    "e2e4 e7e5 "
    "g1f3 g8f6 b1c3 b8c6 f3g1 f6g8 c3b1 c6b8 "
    "g1f3 g8f6 b1c3 b8c6 f3g1 f6g8 c3b1 c6b8"
)

#: A repetition INSIDE the 7-ply window (a 4-ply shuffle), so the near case and
#: the far case are separate rather than one case standing in for both.
NEAR_REPETITION = (
    "e2e4 e7e5 g1f3 g8f6 f1c4 f8c5 e1g1 e8g8 f3g5 f6g4 g5f3 g4f6 "
    "f3g5 f6g4 g5f3 g4f6"
)

#: Castling: white short and black short, then a queenside pair.
CASTLE_KINGSIDE = "e2e4 e7e5 g1f3 g8f6 f1c4 f8c5 e1g1 e8g8"
CASTLE_QUEENSIDE = "d2d4 d7d5 b1c3 b8c6 c1f4 c8f5 d1d2 d8d7 e1c1 e8c8"

#: En passant actually PLAYED (exd6 e.p.), and the same line stopping one ply
#: earlier so the ep right exists and is declined.
EP_CAPTURED = "e2e4 a7a6 e4e5 d7d5 e5d6 a6a5"
EP_DECLINED_LEGAL = "e2e4 a7a6 e4e5 d7d5"

#: Irreversible moves INSIDE the 7-ply window: a capture chain plus pawn moves,
#: so the root's halfmove-clock walk starts from a clock that is not the row's.
IRREVERSIBLE_IN_WINDOW = "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5 g1f3 g8f6 f1c4 c8g4"

#: ⚑ A ROOT RIGHT AFTER A DOUBLE PAWN PUSH WITH NO LEGAL EP CAPTURE.  White's
#: e-pawn double-steps with no black pawn on d4/f4, so python-chess sets
#: ``ep_square`` but ``fen()`` under its DEFAULT ``en_passant="legal"`` policy
#: prints ``-``.  Plane 110 reads ``ep_square`` RAW, so a root written with the
#: default policy loses a plane -- the "default fen()" mutant's target.
PSEUDO_LEGAL_EP_FEN = "4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1"

#: The other way an ep right can be pseudo-legal only: the capturer is PINNED.
#: White has just played d2-d4; ``exd3 e.p.`` would open the e-file onto the
#: black king, so python-chess sets ``ep_square`` and still prints ``-``.
PSEUDO_LEGAL_EP_PINNED_FEN = "4k3/8/8/8/3Pp3/8/8/4R1K1 b - d3 0 1"


def named_cases() -> dict[str, list[chess.Board]]:
    """``{case name: every position of that case}``.

    Includes both opening sources the generator can start from: the production
    book/startpos sampler (``sample_starting_board``, whose board carries its
    book moves on the stack) and the blind-spot FEN-list branch (a bare FEN with
    an EMPTY stack, whose early plies are the short-history case).
    """
    cases: dict[str, list[chess.Board]] = {
        "far_repetition": positions(chess.STARTING_FEN, FAR_REPETITION),
        "near_repetition": positions(chess.STARTING_FEN, NEAR_REPETITION),
        "castle_kingside": positions(chess.STARTING_FEN, CASTLE_KINGSIDE),
        "castle_queenside": positions(chess.STARTING_FEN, CASTLE_QUEENSIDE),
        "ep_captured": positions(chess.STARTING_FEN, EP_CAPTURED),
        "ep_declined_legal": positions(chess.STARTING_FEN, EP_DECLINED_LEGAL),
        "irreversible_in_window": positions(
            chess.STARTING_FEN, IRREVERSIBLE_IN_WINDOW,
        ),
        "pseudo_legal_ep_root": positions(PSEUDO_LEGAL_EP_FEN, "e8d8 e1d1"),
        "pseudo_legal_ep_pinned": positions(
            PSEUDO_LEGAL_EP_PINNED_FEN, "e8d8 g1g2 d8e8 g2g1",
        ),
        "bare_fen_start": positions(
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "e1g1 g8f6 d2d3 f8c5 b1c3 e8g8 c1g5 h7h6",
        ),
        # ⚑ THE PRODUCTION SAMPLER, whose board arrives with its opening moves
        # already PUSHED -- so "the game start" is behind them and a row at the
        # game's own ply 0 still has 7 frames of real history.  `random` rather
        # than `book1` only because the book file is runtime output; both
        # branches hand back a board with a non-empty stack, which is the
        # property under test.
        "sampled_start": played_out(
            sample_starting_board(
                rng=np.random.default_rng(11),
                cfg=OpeningConfig(random_start_plies=6),
            ).board,
            plies=10,
            seed=5,
        ),
    }
    cases.update(random_games())
    return cases


def played_out(
    start: chess.Board, *, plies: int, seed: int,
) -> list[chess.Board]:
    """``start`` and every position of a seeded random-legal continuation.

    Written this way rather than as a hardcoded move list because ``start``
    itself comes from the sampler and is not fixed by this file.
    """
    rng = random.Random(seed)
    board = start.copy(stack=True)
    out = [board.copy(stack=True)]
    for _ in range(plies):
        if board.is_game_over(claim_draw=True):
            break
        board.push(rng.choice(list(board.legal_moves)))
        out.append(board.copy(stack=True))
    return out


def random_games(*, games: int = 8, plies: int = 40) -> dict[str, list[chess.Board]]:
    """Seeded random-legal games, for the cases nobody thought to name."""
    rng = random.Random(20260901)
    out: dict[str, list[chess.Board]] = {}
    for game in range(games):
        board = chess.Board()
        seen = [board.copy(stack=True)]
        for _ in range(plies):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
            seen.append(board.copy(stack=True))
        out[f"random_game_{game}"] = seen
    return out


CASES = named_cases()


# ── THE GATE ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_reconstructed_row_encodes_exactly_like_live_play(case: str) -> None:
    """⚑⚑ THE DELIVERABLE.  Complete-tensor equality against the C play path."""
    for index, live in enumerate(CASES[case]):
        rebuilt = round_trip(live)
        label = f"{case}[{index}] {live.fen()}"
        assert rebuilt.fen() == live.fen(), label
        want = live_planes(live)
        got = derived_planes(rebuilt)
        if not np.array_equal(want, got):
            differing = sorted(
                int(p) for p in np.where(np.any(want != got, axis=(1, 2)))[0]
            )
            pytest.fail(f"{label}: planes differ at {differing}")
        # The search's own terminal test, not just the planes it encodes.
        assert (
            CBoard.from_board(rebuilt).is_repetition()
            == CBoard.from_board(live).is_repetition()
        ), label


def test_the_deriver_and_the_live_c_encoder_agree_on_the_live_board_too() -> None:
    """The equality above is a claim about the WINDOW, not about the encoders.

    Without this, a run in which both encoders were wrong in the same way would
    pass ``test_a_reconstructed_row_encodes_exactly_like_live_play`` -- and a
    run in which they disagreed would blame the window.  Separating the two
    claims is what makes a failure legible: this one fails if the python and C
    repetition implementations drift apart, the one above if the window does.
    """
    checked = 0
    for boards in CASES.values():
        for live in boards:
            assert np.array_equal(live_planes(live), derived_planes(live)), live.fen()
            checked += 1
    assert checked > 400, checked


def test_the_reference_encoding_is_also_what_selfplay_rows_are_written_from() -> None:
    """⚑ THE RIGHT-HAND SIDE OF THE GATE IS THE RIGHT ONE, checked not assumed.

    The gate compares against ``CBoard.from_board(board)``, which is how the UCI
    search builds its root (``uci/root_parallel_gumbel.py``, ``uci/search.py``).
    Live SELFPLAY rows, though, are encoded from a CBoard that was built once at
    the opening and PUSHED forward (``selfplay/state.py`` +
    ``push_index``) -- a different constructor, with its per-slot repetition
    flags recorded at push time rather than by scanning a stack.  If those two
    disagreed, the gate would be pinning the corpus to something production does
    not write.  They do not, and this is the reading that says so.
    """
    board = chess.Board()
    cb = CBoard.from_board(board)
    checked = 0
    for uci in FAR_REPETITION.split():
        move = chess.Move.from_uci(uci)
        cb.push_index(int(move_to_index(move, board)))
        board.push(move)
        pushed = np.asarray(
            encode_cboard(
                cb,
                input_history_encoding=derive.INPUT_HISTORY_ENCODING,
                input_extra_features=derive.INPUT_EXTRA_FEATURES,
            ),
            dtype=np.float32,
        )
        assert np.array_equal(pushed, live_planes(board)), board.fen()
        checked += 1
    assert checked == len(FAR_REPETITION.split())


def test_the_history_rep_fix_stamp_is_inert_on_the_construction_this_tool_uses(
) -> None:
    """⚑ MEASURED, because the deriver STAMPS ``history_rep_fix: true`` into
    every shard as replay identity and never applies the flag.

    On a zero-history corpus that stamp was vacuous (nothing can repeat).  On
    schema 2 it is a real claim about the planes, so it needs a reading rather
    than an argument: over the repetition cases, a ``from_board`` CBoard encodes
    IDENTICALLY under both regimes.  The flag exists for CBoards built by
    ``cboard_push``, whose ``hash_stack`` is cleared at an irreversible move;
    ``from_board`` rebuilds that stack from the python stack, so the default
    path has the same look-back the fix records.  If that ever stops being true
    this fails, and the deriver then has to APPLY the flag it stamps.
    """
    boards = [*CASES["far_repetition"], *CASES["near_repetition"]]
    with_fix = [live_planes(b) for b in boards]
    rep_fix.apply(False, boards_discarded=True)
    try:
        without_fix = [live_planes(b) for b in boards]
    finally:
        rep_fix.apply(True, boards_discarded=True)
    for board, want, got in zip(boards, with_fix, without_fix):
        assert np.array_equal(want, got), board.fen()
    # And the deriver's own encoder agrees with both.
    for board, want in zip(boards, with_fix):
        assert np.array_equal(derived_planes(board), want), board.fen()


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_schema_1_bare_fen_path_differs_from_live_on_every_position_with_history(
    case: str,
) -> None:
    """⚑ THE NEGATIVE.  A gate that passes both ways is not a gate.

    The old (schema-1) reconstruction is ``chess.Board(row["fen"])``.  For every
    position whose live history is non-empty it must produce DIFFERENT planes --
    otherwise the equality above is being satisfied by something other than the
    banked window.
    """
    with_history = 0
    for index, live in enumerate(CASES[case]):
        if not live.move_stack:
            continue
        with_history += 1
        bare = derive.board_from_row({
            "schema": 1, "fen": live.fen(), "game_id": 0, "ply": index,
        })
        assert bare.fen() == live.fen()
        assert not np.array_equal(live_planes(live), derived_planes(bare)), (
            f"{case}[{index}] {live.fen()}: the bare-FEN path matched live, so "
            "this case cannot detect a dropped window"
        )
    assert with_history, f"{case} has no position with history to compare"


# ── the cases are what they are named ────────────────────────────────────────


def test_every_named_case_exercises_what_it_claims_to() -> None:
    """⚑ A case list that does not reach its own feature is a list of no-ops.

    Each assertion below reads the PLANES or the board, so a case that stopped
    producing (say) a repetition -- because a move was retyped, or python-chess
    changed -- fails here rather than passing the gate vacuously.
    """
    def any_rep_plane(boards: Sequence[chess.Board]) -> int:
        return sum(
            1 for b in boards if bool(np.any(live_planes(b)[REP_PLANES]))
        )

    # 2-fold at ply 10 and 3-fold at ply 18, with EVERY earlier occurrence more
    # than HISTORY_WINDOW_PLIES back -- read off python-chess, then off the
    # planes, then off the window length.
    far = CASES["far_repetition"]
    assert far[10].is_repetition(2), "the 2-fold case no longer 2-folds"
    assert far[18].is_repetition(3), "the 3-fold case no longer 3-folds"
    for index in (10, 18):
        earlier = [
            j for j in range(index)
            if far[j]._transposition_key() == far[index]._transposition_key()
        ]
        assert earlier, index
        assert index - max(earlier) > corpus.HISTORY_WINDOW_PLIES, (
            f"far_repetition[{index}]'s nearest earlier occurrence is only "
            f"{index - max(earlier)} plies back, so it is inside the 8 encoded "
            "frames and a last-7-moves banker would still find it"
        )
        assert bool(np.any(live_planes(far[index])[REP_PLANES[0]])), (
            f"far_repetition[{index}] sets no current-position repetition plane"
        )
    assert any_rep_plane(far) >= 8, "the repetition planes never fire"
    assert any_rep_plane(CASES["near_repetition"]) >= 4, (
        "the near-repetition case never repeats"
    )
    windows = [corpus.history_for(b) for b in far]
    assert max(w.plies for w in windows) > corpus.HISTORY_WINDOW_PLIES
    assert corpus.HISTORY_ROOT_IRREVERSIBLE in {w.reason for w in windows}, (
        "no window in the far-repetition case is rooted at an irreversible move"
    )

    # Castling, both wings, both colours -- read off the rights that VANISH.
    ks = CASES["castle_kingside"][-1]
    assert not ks.has_castling_rights(chess.WHITE)
    assert not ks.has_castling_rights(chess.BLACK)
    qs = CASES["castle_queenside"][-1]
    assert qs.king(chess.WHITE) == chess.C1
    assert qs.king(chess.BLACK) == chess.C8

    # En passant, captured and declined.
    ep_line = CASES["ep_captured"]
    assert ep_line[4].ep_square == chess.D6, "the ep right is gone from the line"
    assert chess.Move.from_uci("e5d6") in ep_line[4].legal_moves
    declined = CASES["ep_declined_legal"][-1]
    assert declined.has_legal_en_passant(), "the declined-ep case has no ep right"

    # ⚑ Pseudo-legal-only ep: the square exists on the board and DOES NOT
    # survive python-chess's default fen(). This is the whole reason the banked
    # root is written with en_passant="fen".
    for name, square in (
        ("pseudo_legal_ep_root", chess.E3), ("pseudo_legal_ep_pinned", chess.D3),
    ):
        pseudo = CASES[name][0]
        assert pseudo.ep_square == square, name
        assert not pseudo.has_legal_en_passant(), name
        assert pseudo.fen().split()[3] == "-", pseudo.fen()
        assert corpus.history_for(pseudo).root_fen.split()[3] != "-", (
            f"{name}: the banked root dropped the pseudo-legal ep square"
        )
        assert bool(np.any(live_planes(pseudo)[EP_PLANE])), (
            f"{name}: the ep plane is not set, so losing the square is invisible"
        )

    # Irreversible moves inside the 7-ply window: some position's own clock is
    # smaller than the clock 7 plies back, which is what the two-stage walk is
    # for.
    irr = CASES["irreversible_in_window"]
    assert any(b.halfmove_clock == 0 for b in irr[-corpus.HISTORY_WINDOW_PLIES:])

    # Short histories at both kinds of start.
    assert CASES["bare_fen_start"][0].move_stack == []
    assert len(CASES["sampled_start"][0].move_stack) > 0, (
        "the production sampler pushed no book moves, so the 'game start is "
        "behind the book' case is not covered"
    )
    for name in ("bare_fen_start", "castle_kingside"):
        plies = [corpus.history_for(b).plies for b in CASES[name][:7]]
        assert min(plies) < corpus.HISTORY_WINDOW_PLIES, name


# ── the window itself ────────────────────────────────────────────────────────


def test_the_window_reaches_the_last_irreversible_move_and_says_which() -> None:
    """``history_root_reason`` is a reading of the root, not a label on it."""
    for case, boards in CASES.items():
        for index, board in enumerate(boards):
            window = corpus.history_for(board)
            root = chess.Board(window.root_fen)
            label = f"{case}[{index}]"
            if window.reason == corpus.HISTORY_ROOT_IRREVERSIBLE:
                assert root.halfmove_clock == 0, label
                assert window.plies >= min(
                    corpus.HISTORY_WINDOW_PLIES, len(board.move_stack),
                ), label
            else:
                assert window.reason == corpus.HISTORY_ROOT_GAME_START, label
                assert window.plies == len(board.move_stack), label


def test_the_window_covers_every_frames_own_reversible_run() -> None:
    """The root definition's REASON, checked rather than restated.

    Walk the live board back frame by frame; for each of the 8 encoded frames,
    the start of that frame's own reversible run must be at or after the banked
    root.  That is the property the equality gate depends on, and it is checked
    here on its own so a failure names the window rather than a plane index.
    """
    for case, boards in CASES.items():
        for index, board in enumerate(boards):
            window = corpus.history_for(board)
            root_ply = board.ply() - window.plies
            # ⚑ CLAMPED AT THE GAME START.  A board built from a bare FEN with
            # a nonzero halfmove clock reports a reversible run reaching back
            # before its own first position; there is no history there to
            # cover, and live play has none either.
            game_start_ply = board.ply() - len(board.move_stack)
            walk = board.copy(stack=True)
            for frame in range(_HISTORY_SLOTS):
                if frame and not walk.move_stack:
                    break
                if frame:
                    walk.pop()
                run_start = max(walk.ply() - walk.halfmove_clock, game_start_ply)
                assert run_start >= root_ply, (
                    f"{case}[{index}] frame {frame}: its reversible run starts "
                    f"at ply {run_start}, before the banked root at {root_ply}"
                )


def test_a_window_that_does_not_reproduce_its_position_is_refused() -> None:
    """Both ends refuse: the generator on the way in, the deriver on the way out."""
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push_uci(uci)
    row = row_for(board)

    tampered = dict(row, history_uci=[*row["history_uci"][:-1], "b8a6"])
    with pytest.raises(derive.CorpusIntegrityError, match="not the row's own"):
        derive.board_from_row(tampered)

    # ⚑ AN ILLEGAL-BUT-PARSEABLE MOVE.  ``Board.push`` does NOT check legality
    # (that is python-chess's documented contract), so the FEN equality -- not
    # the push -- is what refuses it.  Asserted here so the reason the replay
    # uses the cheap ``push`` is a tested claim rather than a comment.
    illegal = dict(row, history_uci=[*row["history_uci"], "a1a8"])
    with pytest.raises(derive.CorpusIntegrityError, match="not the row's own"):
        derive.board_from_row(illegal)

    unparseable = dict(row, history_uci=[*row["history_uci"], "zz99"])
    with pytest.raises(derive.CorpusIntegrityError, match="does not replay"):
        derive.board_from_row(unparseable)

    bad_root = dict(row, history_root_fen=chess.STARTING_FEN.replace("w", "b", 1))
    with pytest.raises(derive.CorpusIntegrityError):
        derive.board_from_row(bad_root)


def test_the_generator_refuses_to_bank_a_window_it_cannot_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``history_for``'s own self-check, driven by breaking the replay.

    The check exists so a row that cannot reproduce its own FEN is never
    WRITTEN; without a mutation there is no way to see it fire.
    """
    board = chess.Board()
    for uci in ("d2d4", "d7d5", "c2c4"):
        board.push_uci(uci)

    real_board = chess.Board

    class DroppingBoard(real_board):  # type: ignore[misc, valid-type]
        def push(self, move: chess.Move) -> None:
            if move.uci() != "c2c4":
                super().push(move)

    monkeypatch.setattr(corpus.chess, "Board", DroppingBoard)
    with pytest.raises(RuntimeError, match="does not reproduce its own position"):
        corpus.history_for(board)


# ── the label path ───────────────────────────────────────────────────────────


def test_the_position_command_carries_the_window() -> None:
    """⚑ The exact bytes, because this is what Stockfish is told."""
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push_uci(uci)
    window = corpus.history_for(board)
    assert corpus.position_command(window) == (
        f"position fen {chess.STARTING_FEN} moves e2e4 e7e5 g1f3"
    )
    empty = corpus.RowHistory(
        fen=chess.STARTING_FEN, root_fen=chess.STARTING_FEN, uci=(),
        reason=corpus.HISTORY_ROOT_GAME_START,
    )
    assert corpus.position_command(empty) == f"position fen {chess.STARTING_FEN}"


def test_the_generator_sends_no_position_line_it_did_not_build() -> None:
    """Every ``position`` in the generator goes through ONE helper.

    Grep-level, on purpose: a second speller added later is exactly how the
    label path goes history-blind again, and it would not show up in any test
    that only drives the engine through today's call sites.
    """
    source = Path(corpus.__file__).read_text(encoding="utf-8")
    sends = [
        line.strip() for line in source.splitlines()
        if "_send(" in line and "position" in line
    ]
    assert sends == ["self.engine._send(position_command(history))"], sends
