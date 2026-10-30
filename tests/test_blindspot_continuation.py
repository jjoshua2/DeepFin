"""Ply→side-to-move parity for the continuation reader (scripts/blindspot_continuation.py)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.blindspot_continuation import GameRow, Seed, Verdict, classify, load_game_rows_from_jsonl


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


def _verdict(rows: list[GameRow] | None) -> Verdict:
    seed = Seed("fixture", 0.1, -0.7, 1, "fixture", 10)
    return classify(seed, rows, recover_to=-0.2, still_lost=-0.5,
                    confirm_h=8, horizons=[2, 4, 8], tol=0.03)


def _row(gap: int, sf_q: float, *, outcome: int = 2) -> GameRow:
    return GameRow(10 + gap, 0.1, sf_q, outcome, False)


@pytest.mark.parametrize("label", [-0.7, 0.1])
@pytest.mark.parametrize("recorded_gap", [8, 40])
def test_missing_horizon_labels_do_not_reuse_stale_verdict(label: float, recorded_gap: int) -> None:
    verdict = _verdict([_row(0, -0.7), _row(2, label), _row(recorded_gap, float("nan"))])
    assert verdict.bucket == "INCONCLUSIVE"
    assert verdict.profile == {2: label, 4: None, 8: None}
    assert verdict.reached_terminal is (recorded_gap <= 8)
    assert verdict.outcome == 2


@pytest.mark.parametrize(("early", "horizon", "expected"), [
    (0.1, -0.7, "CONFIRMED_LOST"), (-0.7, 0.1, "REFUTED"),
    (-0.7, -0.3, "INCONCLUSIVE"),
])
def test_fully_labelled_horizon_keeps_asof_and_transient_rules(
    early: float, horizon: float, expected: str,
) -> None:
    verdict = _verdict([_row(0, -0.7), _row(2, early), _row(8, horizon), _row(40, float("nan"))])
    assert verdict.bucket == expected
    assert verdict.profile == {2: early, 4: early, 8: horizon}
    assert verdict.reached_terminal is False


@pytest.mark.parametrize(("label", "expected"), [(-0.7, "CONFIRMED_LOST"), (0.1, "REFUTED")])
def test_short_recorded_game_retains_last_label_fallback(label: float, expected: str) -> None:
    verdict = _verdict([_row(0, -0.7, outcome=0), _row(2, label, outcome=0)])
    assert verdict.bucket == expected
    assert verdict.profile == {2: label, 4: label, 8: label}
    assert verdict.reached_terminal is True


@pytest.mark.parametrize("gap", [4, 40])
def test_all_unlabelled_forward_rows_stay_inconclusive(gap: int) -> None:
    verdict = _verdict([_row(0, -0.7), _row(gap, float("nan"))])
    assert verdict.bucket == "INCONCLUSIVE"
    assert verdict.profile == {2: None, 4: None, 8: None}
    assert verdict.reached_terminal is (gap <= 8)


@pytest.mark.parametrize(("outcome", "expected"), [(2, "CONFIRMED_LOST"), (0, "INCONCLUSIVE")])
def test_no_forward_rows_preserve_existing_terminal_fallback(outcome: int, expected: str) -> None:
    verdict = _verdict([_row(0, -0.7, outcome=outcome)])
    assert verdict.bucket == expected
    assert verdict.profile == {2: None, 4: None, 8: None}
    assert verdict.reached_terminal is False


def test_odd_ply_label_keeps_seed_side_pov_without_filling_missing_horizon() -> None:
    rows = [_row(0, -0.7), _row(1, 0.7), _row(8, float("nan"))]
    verdict = _verdict(rows)
    assert verdict.bucket == "INCONCLUSIVE"
    assert verdict.profile == {2: None, 4: None, 8: None}
    # A later actual label supplies coverage; the existing as-of/POV rule stays.
    verdict = _verdict([*rows, _row(9, 0.7)])
    assert verdict.bucket == "CONFIRMED_LOST"
    assert verdict.profile == {2: -0.7, 4: -0.7, 8: -0.7}


@pytest.mark.parametrize("rows", [None, []])
def test_unrecoverable_rows_remain_unrecoverable(rows: list[GameRow] | None) -> None:
    verdict = _verdict(rows)
    assert verdict.bucket == "UNRECOVERABLE"
    assert verdict.profile == {}


def test_jsonl_null_label_is_unknown_at_classification_boundary(tmp_path: Path) -> None:
    record = _record(_WHITE_ROOT_PLY_136, [136, 138, 144], "0-1")
    record["plies"] = [
        {"ply": 136, "nq": 0.1, "sq": -0.7},
        {"ply": 138, "nq": 0.1, "sq": -0.7},
        {"ply": 144, "nq": 0.1, "sq": None},
    ]
    path = _write(tmp_path, "fixture.jsonl", record)
    rows = load_game_rows_from_jsonl([path])[7]
    seed = Seed("fixture", 0.1, -0.7, 7, "fixture", 136)
    verdict = classify(seed, rows, recover_to=-0.2, still_lost=-0.5,
                       confirm_h=8, horizons=[2, 4, 8], tol=0.03)
    assert verdict.bucket == "INCONCLUSIVE"
    assert verdict.profile[8] is None


def test_cli_depth_report_excludes_uncovered_profile(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    from scripts import blindspot_continuation as module

    seed = Seed("fixture", 0.1, -0.7, 1, "fixture", 10)
    rows = [_row(0, -0.7), _row(2, -0.7), _row(8, float("nan"))]
    monkeypatch.setattr(module.glob, "glob", lambda _pattern: [])
    monkeypatch.setattr(module, "parse_seeds", lambda _paths: [seed])
    monkeypatch.setattr(module, "load_game_rows_from_jsonl", lambda _paths, _wanted: {1: rows})

    def no_replay(*_args, **_kwargs):
        pytest.fail("synthetic saved rows must not trigger replay reads")

    monkeypatch.setattr(module, "load_game_rows", no_replay)
    monkeypatch.setattr("sys.argv", ["blindspot_continuation.py", "--horizons", "2,4,8"])
    module.main()
    output = capsys.readouterr().out
    assert "2pl" in output
    assert "4pl" not in output
    assert "8pl" not in output
    assert "+8:--" in output
    assert "--- INCONCLUSIVE" in output
