"""Evaluation must not launder an undecodable action id into a legal move.

The laundering had two layers. `apply_actions_to_boards` re-checked the decoded
move against `board.legal_moves` and substituted the first legal move if it
failed — but that guard could never fire, because every return path of
`index_to_move_fast` is already legal: the substitution happens one level down,
inside `moves/encode.py`, on two branches (no LUT entry for this side to move,
and a constructed move that is illegal with no legal move encoding back to the
id). So a broken action space produced arena games that were scored normally.

These tests pin both layers: the strict decoder raises, the resilient one
substitutes AND counts, and the evaluation call sites ask for strict.
"""

from __future__ import annotations

import ast
from pathlib import Path

import chess
import numpy as np
import pytest

from chess_anti_engine.moves import (
    ActionDecodeError,
    decode_fallback_count,
    index_to_move_fast,
    index_to_move_strict,
    move_to_index,
)
from chess_anti_engine.moves import encode
from chess_anti_engine.moves.encode import _INDEX_TO_MOVE_LUT
from chess_anti_engine.selfplay import match as match_mod
from chess_anti_engine.selfplay.match import apply_actions_to_boards, play_match_batch
from chess_anti_engine.selfplay.opening import OpeningConfig

ROOT = Path(__file__).resolve().parent.parent

# From-square a1 (oriented) with plane 21: a queen ray that leaves the board, so
# the reverse LUT holds nothing for it — for either side to move, since the LUT
# is built in oriented coordinates. Undecodable in EVERY position.
NO_LUT_ENTRY_ID = 21
# a1a2 from White's side: the LUT does hold an entry, but the rook is blocked by
# its own pawn in the start position and no legal move there encodes back to it.
BLOCKED_ROOK_ID = 0

# The move python-chess yields first in the start position; what both resilient
# decoders substitute for the two ids above.
FIRST_LEGAL_STARTPOS = chess.Move.from_uci("g1h3")


def test_fixture_ids_really_are_the_two_failure_branches() -> None:
    """Pin the fixtures to the branches they stand for.

    A LUT change could quietly make either id decodable, and both tests below
    would then pass while measuring nothing.
    """
    board = chess.Board()
    for turn in (chess.WHITE, chess.BLACK):
        assert int(_INDEX_TO_MOVE_LUT[int(turn), NO_LUT_ENTRY_ID][0]) < 0

    entry = _INDEX_TO_MOVE_LUT[int(board.turn), BLOCKED_ROOK_ID]
    assert int(entry[0]) >= 0, "this id must HAVE a LUT entry — that is the point"
    assert chess.Move(int(entry[0]), int(entry[1])) == chess.Move.from_uci("a1a2")
    assert int(move_to_index(chess.Move.from_uci("a1a2"), board)) == BLOCKED_ROOK_ID
    assert chess.Move.from_uci("a1a2") not in board.legal_moves
    assert next(iter(board.legal_moves)) == FIRST_LEGAL_STARTPOS


@pytest.mark.parametrize(
    ("action_id", "detail"),
    [(NO_LUT_ENTRY_ID, "no LUT entry"), (BLOCKED_ROOK_ID, "a1a2")],
)
def test_strict_decode_raises_naming_the_id_and_the_position(
    action_id: int, detail: str,
) -> None:
    board = chess.Board()
    with pytest.raises(ActionDecodeError) as excinfo:
        index_to_move_strict(action_id, board)

    message = str(excinfo.value)
    assert str(action_id) in message, "the message must name the action id"
    assert board.fen() in message, "the message must carry the position"
    assert detail in message
    assert excinfo.value.index == action_id
    assert excinfo.value.fen == board.fen()
    assert excinfo.value.turn is True
    assert board.move_stack == [], "decoding must not mutate the board"


@pytest.mark.parametrize("action_id", [NO_LUT_ENTRY_ID, BLOCKED_ROOK_ID])
def test_fast_decode_substitutes_first_legal_and_counts_it(action_id: int) -> None:
    board = chess.Board()
    before = decode_fallback_count()

    assert index_to_move_fast(action_id, board) == FIRST_LEGAL_STARTPOS
    assert decode_fallback_count() == before + 1, (
        "the substitution must be counted — an uncounted one is invisible"
    )


def test_first_substitution_is_announced_on_stderr_then_only_counted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The count is for whoever asks; the one-shot line is for whoever does not.

    Worker stderr is captured into the run log (``stderr=subprocess.STDOUT`` in
    ``tune/distributed_runtime.py``, ``> "$LOG" 2>&1`` in ``scripts/train.sh``),
    so this line is what makes a live substitution visible without anyone
    thinking to look. stdout is the UCI protocol channel and must stay clean.
    """
    monkeypatch.setattr(encode, "_DECODE_FALLBACK_WARNED", False)
    board = chess.Board()

    index_to_move_fast(BLOCKED_ROOK_ID, board)
    first = capsys.readouterr()
    assert first.out == "", "stdout is the UCI protocol channel"
    assert str(BLOCKED_ROOK_ID) in first.err
    assert board.fen() in first.err
    assert "SUBSTITUTED" in first.err

    index_to_move_fast(BLOCKED_ROOK_ID, board)
    assert capsys.readouterr().err == "", "further occurrences are counted, not logged"


def test_apply_actions_strict_propagates_and_leaves_the_board_alone() -> None:
    boards = [chess.Board()]
    with pytest.raises(ActionDecodeError):
        apply_actions_to_boards(boards, [0], [BLOCKED_ROOK_ID], strict=True)
    assert boards[0].move_stack == []


def test_apply_actions_non_strict_substitutes_and_counts() -> None:
    boards = [chess.Board()]
    before = decode_fallback_count()

    apply_actions_to_boards(boards, [0], [BLOCKED_ROOK_ID], strict=False)

    assert boards[0].move_stack == [FIRST_LEGAL_STARTPOS]
    assert decode_fallback_count() == before + 1


def test_legal_moves_round_trip_identically_through_fast_and_strict() -> None:
    """Every legal move decodes back to itself, and neither decoder falls back."""
    rng = np.random.default_rng(20260823)
    before = decode_fallback_count()
    positions = 0
    moves_checked = 0

    for _game in range(40):
        board = chess.Board()
        for _ply in range(24):
            if board.is_game_over():
                break
            legal = list(board.legal_moves)
            positions += 1
            for move in legal:
                index = int(move_to_index(move, board))
                assert index_to_move_strict(index, board) == move
                assert index_to_move_fast(index, board) == move
                moves_checked += 1
            board.push(legal[int(rng.integers(0, len(legal)))])

    assert positions >= 400, f"walk must cover a few hundred positions, got {positions}"
    assert moves_checked >= 10_000
    assert decode_fallback_count() == before, (
        "a legal move must never take the substitution path"
    )


def _legal_actions(boards: list[chess.Board]) -> list[int]:
    return [int(move_to_index(next(iter(b.legal_moves)), b)) for b in boards]


@pytest.mark.parametrize("a_plays_white", [True, False])
def test_play_match_batch_propagates_a_decode_failure(
    monkeypatch: pytest.MonkeyPatch, a_plays_white: bool,
) -> None:
    """The paired match is a gate/eval instrument, so it must not play on.

    ``max_plies=1`` deliberately: only the side to move decodes, so each
    parametrisation pins ONE of the two `apply_actions_to_boards` call sites.
    Two plies would let the other call site's raise cover for a lenient one.
    """
    def _undecodable(_model: object, sub_boards: list[chess.Board], **_kw: object) -> list[int]:
        return [NO_LUT_ENTRY_ID] * len(sub_boards)

    monkeypatch.setattr(match_mod, "pick_moves_for_boards", _undecodable)

    with pytest.raises(ActionDecodeError):
        play_match_batch(
            None, None,  # pyright: ignore[reportArgumentType] - the search is stubbed
            device="cpu",
            rng=np.random.default_rng(0),
            games=1,
            max_plies=1,
            a_plays_white=[a_plays_white],
            mcts_type="gumbel",
            mcts_simulations=1,
            opening_cfg=OpeningConfig(random_start_plies=0),
        )


@pytest.mark.parametrize("rolling", [False, True])
def test_arena_call_sites_ask_for_strict(monkeypatch: pytest.MonkeyPatch, rolling: bool) -> None:
    """Both `arena_standard` play loops must decode strictly.

    Behavioural, not a source scan: the recorder sits where the arena actually
    calls, so flipping either call site to `strict=False` fails this.
    """
    import scripts.arena_standard as arena

    seen: list[bool] = []
    real_apply = match_mod.apply_actions_to_boards

    def _record(
        boards: list[chess.Board], idxs: list[int], actions: list[int], *, strict: bool,
    ) -> None:
        seen.append(bool(strict))
        real_apply(boards, idxs, actions, strict=strict)

    def _pick(_model: object, sub_boards: list[chess.Board], **_kw: object) -> list[int]:
        return _legal_actions(sub_boards)

    monkeypatch.setattr(match_mod, "apply_actions_to_boards", _record)
    monkeypatch.setattr(match_mod, "pick_moves_for_boards", _pick)

    side = arena.resolve_search_shape("training")
    common: dict[str, object] = {
        "device": "cpu",
        "rng": np.random.default_rng(0),
        "sims_candidate": 1,
        "sims_reference": 1,
        "max_plies": 4,
        "temperature": 0.1,
        "gumbel_add_noise": False,
        "search_candidate": side,
        "search_reference": side,
    }
    play = (
        arena.play_paired_games_matched_sims_rolling
        if rolling
        else arena.play_paired_games_matched_sims
    )
    play(None, None, [chess.Board()], **common)  # pyright: ignore[reportArgumentType]

    assert seen, "the arena never applied an action — the test measured nothing"
    assert all(seen), "an arena play loop decoded actions non-strictly"


def _strict_kwargs_at_call_sites(path: Path) -> list[ast.keyword | None]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[ast.keyword | None] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name != "apply_actions_to_boards":
            continue
        found.append(next((k for k in node.keywords if k.arg == "strict"), None))
    return found


@pytest.mark.parametrize(
    "relpath",
    ["scripts/arena_standard.py", "chess_anti_engine/selfplay/match.py"],
)
def test_every_evaluation_call_site_passes_strict_true(relpath: str) -> None:
    """A NEW evaluation call site must not inherit the resilient mode.

    `strict` is required, so omitting it is a TypeError; this catches the other
    half — a site that passes a non-literal or `False`.
    """
    keywords = _strict_kwargs_at_call_sites(ROOT / relpath)
    assert keywords, f"no apply_actions_to_boards call found in {relpath}"
    for keyword in keywords:
        assert keyword is not None, f"{relpath}: call site omits strict="
        assert isinstance(keyword.value, ast.Constant), (
            f"{relpath}: strict= must be a literal at an evaluation call site"
        )
        assert keyword.value.value is True, f"{relpath}: evaluation call site is not strict"
