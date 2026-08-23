"""Ply→side-to-move parity for the continuation reader (scripts/blindspot_continuation.py)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.blindspot_continuation import load_game_rows_from_jsonl


# `wdl_target` convention: 0 = win / 1 = draw / 2 = loss, side-to-move POV.
_WIN, _LOSS = 0, 2

# A black-to-move root at an ODD absolute ply. This is the shape a FEN-seeded
# (`opening_fen_*`) game produces, and the one the old relative-ply formula
# `root_white == (ply % 2 == 0)` inverts.
_BLACK_ROOT_PLY_137 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 69"
_WHITE_ROOT_PLY_136 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 69"


def _write(tmp_path: Path, name: str, record: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return str(path)


def _record(root_fen: str, plies: list[int], result: str) -> dict:
    return {
        "game_id": 7, "root_fen": root_fen, "result": result,
        "selfplay": False, "moves": [], "seed_plies": [],
        "plies": [{"ply": p, "hp": True, "nq": 0.1, "sq": 0.2} for p in plies],
    }


def test_black_to_move_root_gets_absolute_ply_parity(tmp_path: Path) -> None:
    """⚑ An ABSOLUTE ply is even exactly when White is to move — always, and
    independently of the root's colour. The formula that stood here,
    ``root_white == (ply % 2 == 0)``, is the RELATIVE-ply one: for this
    black-to-move root it reports White to move at ply 137 and Black at 138,
    which flips ``outcome`` on every row of the game.

    White wins, so at an ODD ply (Black to move) the side-to-move LOSES and at
    an EVEN ply it WINS. Both parities are asserted so a formula that is merely
    constant cannot pass.
    """
    path = _write(
        tmp_path, "b.games.p1.jsonl",
        _record(_BLACK_ROOT_PLY_137, [137, 138, 139], "1-0"),
    )
    rows = load_game_rows_from_jsonl([path])[7]

    assert [r.ply for r in rows] == [137, 138, 139]
    assert [r.outcome for r in rows] == [_LOSS, _WIN, _LOSS]


def test_white_to_move_root_is_unchanged(tmp_path: Path) -> None:
    """The white-to-move root is where the two formulas AGREE — which is why
    the old "0/2324 mismatches" check could not have caught the black-root
    inversion. Pinned so the fix is not mistaken for a global sign flip."""
    path = _write(
        tmp_path, "w.games.p1.jsonl",
        _record(_WHITE_ROOT_PLY_136, [136, 137], "1-0"),
    )
    rows = load_game_rows_from_jsonl([path])[7]

    assert [r.outcome for r in rows] == [_WIN, _LOSS]


def test_a_line_without_a_usable_root_fen_is_skipped(tmp_path: Path) -> None:
    """root_fen no longer feeds the parity, but it is still the malformed-line
    guard — dropping the parse would silently admit truncated records."""
    good = _record(_WHITE_ROOT_PLY_136, [136], "1-0")
    bad_missing = {k: v for k, v in good.items() if k != "root_fen"}
    bad_missing["game_id"] = 8
    bad_short = dict(good, game_id=9, root_fen="8/8/8/8/8/8/8/8")
    bad_turn = dict(good, game_id=10, root_fen="8/8/8/8/8/8/8/8 x - - 0 1")

    path = tmp_path / "mixed.games.p1.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in (good, bad_missing, bad_short, bad_turn)) + "\n",
        encoding="utf-8",
    )
    games = load_game_rows_from_jsonl([str(path)])

    assert set(games) == {7}
