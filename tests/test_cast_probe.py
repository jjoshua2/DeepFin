"""Tests for scripts/cast_probe.py.

Every test is written to FAIL under a specific plausible mutation of the probe,
and the mutation is named in the test. A probe whose tests pass when its sign is
flipped, its parent/child join is swapped, or its action reconstruction ignores
the canonical colour mirror is not an instrument.

The shard fixtures carry REAL encoded position planes, because the probe now
reconstructs the played move from consecutive positions. A fixture of synthetic
zeros would let a broken mirror pass.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding.encode import encode_position
from chess_anti_engine.moves.encode import move_to_index_for_encoding
from tests.script_loading import load_script_module

cast_probe = load_script_module("cast_probe.py")

ENC = "lc0_root_legacy_meta"
POLICY_ENC = "lc0_1858"
WIDTH = 1858


def _planes(board: chess.Board) -> np.ndarray:
    return encode_position(board, input_history_encoding=ENC, input_extra_features="v1")


def _canon_move(mv: chess.Move, turn: chess.Color) -> chess.Move:
    """The same move seen from the side-to-move-canonical (white) frame."""
    if turn == chess.WHITE:
        return mv
    return chess.Move(
        chess.square_mirror(mv.from_square),
        chess.square_mirror(mv.to_square),
        promotion=mv.promotion,
    )


def _canon_index(board: chess.Board, mv: chess.Move) -> int:
    canon_b = board if board.turn == chess.WHITE else board.mirror()
    return int(move_to_index_for_encoding(
        _canon_move(mv, board.turn), canon_b, policy_encoding=POLICY_ENC,
    ))


def _line(sans: list[str], start: chess.Board | None = None) -> list[chess.Board]:
    """Boards after each half-move, including the start position."""
    b = (start or chess.Board()).copy()
    out = [b.copy()]
    for san in sans:
        b.push_san(san)
        out.append(b.copy())
    return out


def _shard(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a minimal in-memory shard from per-row dicts."""
    n = len(rows)
    arrs: dict[str, Any] = {
        "policy_target": np.zeros((n, WIDTH), dtype=np.float32),
        "legal_mask": np.zeros((n, WIDTH), dtype=np.uint8),
        "sf_p0_regret": np.zeros((n, WIDTH), dtype=np.float32),
        "sf_multipv_raw": np.full((n, 8, 5), -1, dtype=np.int16),
        "sf_wdl": np.zeros((n, 3), dtype=np.float32),
        "x": np.zeros((n, 146, 8, 8), dtype=np.float32),
        "game_id": np.zeros((n,), dtype=np.int64),
        "ply_index": np.zeros((n,), dtype=np.int32),
        "has_sf_wdl": np.zeros((n,), dtype=np.uint8),
        "has_sf_p0_regret": np.zeros((n,), dtype=np.uint8),
        "has_sf_multipv_raw": np.zeros((n,), dtype=np.uint8),
        "_input_history_encoding": np.array(ENC),
        "_policy_encoding": np.array(POLICY_ENC),
    }
    for i, r in enumerate(rows):
        board: chess.Board = r["board"]
        arrs["x"][i] = _planes(board)
        arrs["game_id"][i] = r.get("game_id", 0)
        arrs["ply_index"][i] = r["ply_index"]
        canon = board if board.turn == chess.WHITE else board.mirror()
        legal = [
            int(move_to_index_for_encoding(m, canon, policy_encoding=POLICY_ENC))
            for m in canon.legal_moves
        ]
        arrs["legal_mask"][i, legal] = 1
        pol = np.zeros((WIDTH,), dtype=np.float32)
        for idx, prob in r.get("policy", {}).items():
            pol[idx] = prob
        if pol.sum() == 0:
            pol[legal] = 1.0 / max(len(legal), 1)
        arrs["policy_target"][i] = pol
        q = float(r.get("q", 0.0))
        arrs["sf_wdl"][i] = [(1.0 + q) / 2.0, 0.0, (1.0 - q) / 2.0]
        arrs["has_sf_wdl"][i] = 1
        covered = r.get("covered")
        if covered is not None:
            for k, (mi, regret) in enumerate(covered.items()):
                arrs["sf_multipv_raw"][i, k, 0] = mi
                arrs["sf_multipv_raw"][i, k, 1] = round(-1000.0 * regret)
                arrs["sf_p0_regret"][i, mi] = regret
            arrs["has_sf_multipv_raw"][i] = 1
        if r.get("regret") is not None:
            for mi, v in r["regret"].items():
                arrs["sf_p0_regret"][i, mi] = v
            arrs["has_sf_p0_regret"][i] = 1
    return arrs, {}


def _collect(rows: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> Any:
    shard = _shard(rows)
    monkeypatch.setattr(cast_probe, "load_shard_arrays", lambda _: shard)
    scan: dict[str, Any] = {
        "rows_scanned": 0, "rows_sf_wdl": 0, "rows_sf_p0_regret": 0,
        "cast_pairs": 0, "cast_pairs_with_p0": 0, "action_recovered": 0,
        "action_unrecovered": 0, "action_illegal": 0, "no_successor": 0,
        "skipped_shards": [], "skipped_shards_omitted": 0,
    }
    out = cast_probe.collect([Path("fake.zarr")], scan, np.random.default_rng(0))
    return out, scan


# --------------------------------------------------------------------------
# 1. Played-move reconstruction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("sans", [
    ["e4"], ["e4", "e5"], ["e4", "e5", "Nf3"], ["e4", "d5", "exd5"],
    ["Nf3", "Nf6", "g3", "g6", "Bg2", "Bg7"],
])
def test_recover_played_move_round_trip(sans: list[str]) -> None:
    """MUTATION: dropping ``.mirror()`` in the candidate comparison.

    The child row is stored side-to-move canonical, so it is the parent's
    successor MIRRORED. Comparing without the mirror matches nothing and the
    probe silently loses every row.
    """
    boards = _line(sans)
    parent, child = boards[-2], boards[-1]
    mv = child.peek()
    got = cast_probe.recover_played_move(
        _planes(parent), _planes(child),
        input_history_encoding=ENC, policy_encoding=POLICY_ENC,
    )
    assert got == _canon_index(parent, mv)


def test_recover_played_move_fails_closed_on_unrelated_positions() -> None:
    """A child not reachable in one move must return None, not a guess."""
    a = chess.Board()
    b = chess.Board("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1")
    b.push_san("d5")
    assert cast_probe.recover_played_move(
        _planes(a), _planes(b), input_history_encoding=ENC, policy_encoding=POLICY_ENC,
    ) is None


def test_recover_played_move_handles_en_passant_square() -> None:
    """A double pawn push creating a legal EP right must still be recovered.

    MUTATION: comparing EP under an encoding that DROPS it (plain ``lc0_root``)
    would reject every such move. Under ``lc0_root_legacy_meta`` the EP plane
    exists, so it belongs in the key; the probe selects on the encoding.
    """
    boards = _line(["e4", "a6", "e5", "d5"])  # d5 creates an EP target on d6
    parent, child = boards[-2], boards[-1]
    assert child.peek() == chess.Move.from_uci("d7d5")
    got = cast_probe.recover_played_move(
        _planes(parent), _planes(child),
        input_history_encoding=ENC, policy_encoding=POLICY_ENC,
    )
    assert got == _canon_index(parent, child.peek())


def test_recover_played_move_under_an_encoding_that_drops_en_passant() -> None:
    """MUTATION: requiring EP to match unconditionally.

    Plain ``lc0_root`` has no EP plane, so the decoded child of a double pawn
    push reports ``ep_square is None`` while the generated candidate knows the
    EP right. Comparing EP there rejects the move and the row is lost as
    "unrecoverable" — a lossy-encoding artefact indistinguishable, in the
    counters, from a broken canonical flip.
    """
    plain = "lc0_root"
    boards = _line(["e4", "a6", "e5", "d5"])
    parent, child = boards[-2], boards[-1]
    from chess_anti_engine.eval.audit import decode_board_from_planes
    decoded_child = decode_board_from_planes(
        encode_position(child, input_history_encoding=plain, input_extra_features="v1"),
        input_history_encoding=plain,
    )
    assert decoded_child is not None
    assert decoded_child.ep_square is None  # the plane simply is not there
    got = cast_probe.recover_played_move(
        encode_position(parent, input_history_encoding=plain, input_extra_features="v1"),
        encode_position(child, input_history_encoding=plain, input_extra_features="v1"),
        input_history_encoding=plain, policy_encoding=POLICY_ENC,
    )
    assert got == _canon_index(parent, child.peek())


# --------------------------------------------------------------------------
# 2. POV / sign
# --------------------------------------------------------------------------

def test_advantage_pov_sign_adds_not_subtracts() -> None:
    """MUTATION: ``q_child - q_parent``.

    Both labels are already in their own record's mover POV, so a move that
    holds the evaluation must score 0. Under subtraction it scores +0.6.
    """
    assert cast_probe.advantage(q_child=-0.3, q_parent=0.3) == pytest.approx(0.0)
    assert cast_probe.advantage(q_child=-0.5, q_parent=0.3) == pytest.approx(-0.2)
    assert cast_probe.advantage(q_child=-0.3, q_parent=0.3) <= 0.0


def test_blunder_is_negative_and_best_move_is_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end sign check through ``collect``."""
    boards = _line(["e4", "e5", "Nf3", "Nc6"])
    # The covered set for the analysed row (ply 2) comes from its PARENT (ply 1),
    # so those indices must be legal moves of the parent position.
    par = [_canon_index(boards[1], m) for m in list(boards[1].legal_moves)[:2]]
    rows = [
        {"board": boards[0], "ply_index": 0, "q": 0.4},
        {"board": boards[1], "ply_index": 1, "q": -0.4,
         "covered": {par[0]: 0.0, par[1]: 0.05}},
        {"board": boards[2], "ply_index": 2, "q": 0.4,
         "regret": {par[0]: 0.0, par[1]: 0.05}},
        {"board": boards[3], "ply_index": 3, "q": -0.9},
    ]
    out, _ = _collect(rows, monkeypatch)
    # Row at ply 2 is the analysed one: parent ply 1 (q=-0.4) + itself (q=+0.4).
    assert out.adv == pytest.approx([0.0])


# --------------------------------------------------------------------------
# 3. Adjacency must be exact
# --------------------------------------------------------------------------

def test_adjacency_requires_exact_previous_ply(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: joining on "nearest earlier row" instead of ``ply_index - 1``."""
    boards = _line(["e4", "e5", "Nf3"])
    rows = [
        {"board": boards[0], "ply_index": 0, "q": 0.4, "covered": {0: 0.0}},
        {"board": boards[2], "ply_index": 7, "q": -0.4, "regret": {0: 0.0}},
    ]
    out, scan = _collect(rows, monkeypatch)
    assert scan["cast_pairs"] == 0
    assert out.adv == []


def test_adjacency_does_not_cross_games(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: joining on ``ply_index`` alone, ignoring ``game_id``."""
    boards = _line(["e4", "e5"])
    rows = [
        {"board": boards[0], "game_id": 0, "ply_index": 4, "q": 0.4, "covered": {0: 0.0}},
        {"board": boards[1], "game_id": 1, "ply_index": 5, "q": -0.4, "regret": {0: 0.0}},
    ]
    out, scan = _collect(rows, monkeypatch)
    assert scan["cast_pairs"] == 0
    assert out.adv == []


# --------------------------------------------------------------------------
# 4. Covered set
# --------------------------------------------------------------------------

def test_covered_set_is_read_from_the_parent_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """MUTATION: reading ``sf_multipv_raw`` from the CHILD row.

    ``sf_p0_regret`` at row t is built from row t-1's MultiPV, because SF's
    label search runs one ply late.
    """
    boards = _line(["e4", "e5", "Nf3", "Nc6"])
    parent_moves = [_canon_index(boards[1], m) for m in list(boards[1].legal_moves)[:2]]
    child_moves = [_canon_index(boards[2], m) for m in list(boards[2].legal_moves)[:3]]
    rows = [
        {"board": boards[1], "ply_index": 1, "q": 0.0,
         "covered": {parent_moves[0]: 0.0, parent_moves[1]: 0.05}},
        {"board": boards[2], "ply_index": 2, "q": 0.0,
         "covered": dict.fromkeys(child_moves, 0.0),
         "regret": {parent_moves[0]: 0.0, parent_moves[1]: 0.05}},
        {"board": boards[3], "ply_index": 3, "q": 0.0},
    ]
    out, _ = _collect(rows, monkeypatch)
    # 2 covered from the PARENT; the child's own 3-move block must be ignored.
    assert out.n_covered == [2]


def test_scored_multipv_indices_skips_the_sentinel() -> None:
    """MUTATION: treating every non-padding raw index as covered.

    ``_build_sf_p0_regret_vector`` skips a PV row whose cp is the sentinel with
    no mate, leaving that entry at the IMPUTED default — so counting it as
    covered turns an imputed value into a claimed exact observation.
    """
    rows = np.array([
        [10, -20, 0, 0, 0],
        [11, cast_probe.SF_CP_SENTINEL, 0, 0, 0],  # unscored -> not covered
        [12, 0, 3, 0, 0],                          # mate score -> covered
        [-1, 0, 0, 0, 0],                          # padding
    ], dtype=np.int16)
    assert cast_probe.scored_multipv_indices(rows, WIDTH).tolist() == [10, 12]


# --------------------------------------------------------------------------
# 5. Calibration
# --------------------------------------------------------------------------

def test_monotone_prefix_stops_at_the_first_fold() -> None:
    """MUTATION: skipping the folded bucket and resuming (the original bug).

    ``ys=[-0.01,-0.05,-0.04,-0.08]`` must return the leading prefix ``[0,1]``.
    The old loop returned ``[0,1,3]``, splicing the pre-fold and post-fold
    branches so an advantage could be inverted through the saturation branch
    the guard claims to discard.
    """
    xs = np.array([0.0, 0.02, 0.09, 0.17])
    ys = np.array([-0.01, -0.05, -0.04, -0.08])
    kx, ky = cast_probe.monotone_prefix(xs, ys)
    assert kx.tolist() == [0.0, 0.02]
    assert ky.tolist() == [-0.01, -0.05]


def test_monotone_prefix_keeps_a_fully_monotone_curve() -> None:
    xs = np.array([0.0, 0.02, 0.09, 0.17, 0.50])
    ys = np.array([-0.014, -0.044, -0.083, -0.160, -0.117])
    kx, ky = cast_probe.monotone_prefix(xs, ys)
    assert kx.tolist() == [0.0, 0.02, 0.09, 0.17]
    assert np.all(np.diff(ky) < 0)


def test_invert_reads_the_curve_backwards() -> None:
    xs = np.array([0.0, 0.02, 0.09])
    ys = np.array([-0.01, -0.05, -0.09])
    assert cast_probe.invert(xs, ys, -0.05) == pytest.approx(0.02)
    assert cast_probe.invert(xs, ys, -0.07) == pytest.approx(0.055)


def test_price_the_tail_reports_the_overstatement() -> None:
    """Outside moves built to be worth the 0.02 bucket while assigned 0.55."""
    rng = np.random.default_rng(0)
    n = 4000
    reg = rng.choice([0.0, 0.02, 0.09], size=n)
    adv = -0.5 * reg
    inside = np.ones((n,), dtype=bool)
    reg = np.concatenate([reg, np.full((400,), 0.55)])
    adv = np.concatenate([adv, np.full((400,), -0.01)])
    inside = np.concatenate([inside, np.zeros((400,), dtype=bool)])
    arr = {
        "adv": adv, "regret_played": reg, "in_multipv": inside,
        "pmax": np.ones_like(adv), "abs_q_parent": np.zeros_like(adv),
    }
    res = cast_probe.price_the_tail(
        arr, np.ones_like(inside), "all", np.random.default_rng(0),
    )
    assert res["implied_cp"] == pytest.approx(20.0, abs=2.0)
    assert res["assigned_cp"] == pytest.approx(550.0)
    assert res["overstatement"] > 20.0
    assert res["n_bootstrap"] > 100


def test_price_the_tail_is_flat_when_the_tail_is_priced_correctly() -> None:
    """NEGATIVE CONTROL: a probe that always reports a large overstatement is
    measuring its own arithmetic."""
    rng = np.random.default_rng(1)
    n = 4000
    reg = rng.choice([0.0, 0.1, 0.3, 0.55], size=n)
    adv = -0.5 * reg
    inside = np.ones((n,), dtype=bool)
    reg = np.concatenate([reg, np.full((400,), 0.55)])
    adv = np.concatenate([adv, np.full((400,), -0.275)])
    inside = np.concatenate([inside, np.zeros((400,), dtype=bool)])
    arr = {
        "adv": adv, "regret_played": reg, "in_multipv": inside,
        "pmax": np.ones_like(adv), "abs_q_parent": np.zeros_like(adv),
    }
    res = cast_probe.price_the_tail(
        arr, np.ones_like(inside), "all", np.random.default_rng(2),
    )
    assert res["overstatement"] == pytest.approx(1.0, abs=0.15)
