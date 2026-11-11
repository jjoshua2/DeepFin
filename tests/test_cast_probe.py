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
        "has_game_id": np.ones((n,), dtype=np.uint8),
        "has_ply_index": np.ones((n,), dtype=np.uint8),
        "has_legal_mask": np.ones((n,), dtype=np.uint8),
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
    scan = cast_probe.new_scan("fake-dir", [Path("fake.zarr")])
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


@pytest.mark.parametrize(("chosen", "expected_cp"), [
    # Multiplicities change both calibration medians/means and the tail mean.
    ([0, 0, 1, 3, 3, 4, 6, 6, 8], 1000 * 0.011 * (0.16 / 0.36)),
    # Repeated draws still count toward the two-row calibration/tail minimum.
    ([0, 0, 3, 3, 6, 6, 6, 6, 6], 1000 * 0.011 * (0.03 / 0.09)),
    # Negative control: without repeated draws, the estimate is unchanged.
    (list(range(9)), 1000 * (0.001 + 0.019 * (0.06 / 0.18))),
])
def test_tail_bootstrap_preserves_sample_multiplicities(
    monkeypatch: pytest.MonkeyPatch, chosen: list[int], expected_cp: float,
) -> None:
    """A boolean membership mask loses replacement counts and changes the CI."""
    monkeypatch.setattr(cast_probe, "MIN_BUCKET", 2)
    monkeypatch.setattr(cast_probe, "N_BOOT", 4)
    sel = np.zeros(18, dtype=bool)
    sel[1::2] = True
    arr = {
        "regret_played": np.full(18, 999.0),
        "adv": np.full(18, 999.0),
        "in_multipv": np.zeros(18, dtype=bool),
    }
    arr["regret_played"][sel] = [0.0, 0.001, 0.0015, 0.011, 0.02, 0.025, 0.55, 0.55, 0.55]
    arr["adv"][sel] = [-0.01, -0.02, -0.03, -0.10, -0.20, -0.30, -0.04, -0.08, -0.12]
    arr["in_multipv"][sel] = [True] * 6 + [False] * 3
    original = {key: value.copy() for key, value in arr.items()}

    class FixedDraws:
        calls = 0

        def integers(self, low: int, high: int, size: int) -> np.ndarray:
            assert (low, high, size) == (0, 9, 9)
            self.calls += 1
            return np.asarray(chosen)

    rng = FixedDraws()
    result = cast_probe.price_the_tail(arr, sel, "selected", rng)
    assert rng.calls == 4
    assert result["n_bootstrap"] == 4
    assert result["implied_cp_lo"] == pytest.approx(expected_cp)
    assert result["implied_cp_hi"] == pytest.approx(expected_cp)
    assert result["implied_cp"] == pytest.approx(1000 * (0.001 + 0.019 * (0.06 / 0.18)))
    assert result["n"] == 9
    assert result["n_outside"] == 3
    for key, value in original.items():
        np.testing.assert_array_equal(arr[key], value)


@pytest.mark.parametrize("omitted", ["game_id", "ply_index", "sf_wdl", "sf_p0_regret", "sf_multipv_raw", "legal_mask"])
def test_collect_pruned_optional_arrays(tmp_path: Path, omitted: str) -> None:
    """Real writer/validated loader must admit pruned optional storage."""
    from chess_anti_engine.replay import ReplaySample
    from chess_anti_engine.replay.shard import load_shard_arrays, samples_to_arrays, save_local_shard_arrays

    samples = []
    for i, board in enumerate(_line(["e4", "e5"])):
        policy = np.full(WIDTH, 1.0 / WIDTH, dtype=np.float32)
        sample = ReplaySample(x=_planes(board), policy_target=policy, wdl_target=1)
        sample.input_history_encoding = ENC
        sample.game_id = 7
        sample.ply_index = i
        sample.sf_wdl = np.array([0.2, 0.6, 0.2], dtype=np.float32)
        samples.append(sample)
    arrs = samples_to_arrays(samples)
    arrs[f"has_{omitted}"] = np.zeros(3, dtype=np.uint8)
    path = save_local_shard_arrays(tmp_path / "shard_000001.zarr", arrs=arrs, meta={"positions": 3})
    loaded, _ = load_shard_arrays(path)
    assert omitted not in loaded
    scan = cast_probe.new_scan(tmp_path, [path])
    result = cast_probe.collect([path], scan, np.random.default_rng(0))
    assert scan["rows_scanned"] == 3
    assert scan["rows_sf_wdl"] == (0 if omitted == "sf_wdl" else 3)
    assert scan["rows_missing_join_metadata"] == (3 if omitted in {"game_id", "ply_index"} else 0)
    assert scan["cast_pairs"] == (0 if omitted in {"game_id", "ply_index", "sf_wdl"} else 2)
    assert scan["cast_pairs_with_p0"] == 0
    assert result.adv == []


def test_collect_partial_join_metadata_never_invents_adjacency(monkeypatch: pytest.MonkeyPatch) -> None:
    boards = _line(["e4", "e5"])
    arrs, meta = _shard([{"board": b, "ply_index": i} for i, b in enumerate(boards)])
    arrs["has_game_id"][1] = 0
    monkeypatch.setattr(cast_probe, "load_shard_arrays", lambda _: (arrs, meta))
    scan = cast_probe.new_scan("fixture", [Path("fixture.zarr")])
    cast_probe.collect([Path("fixture.zarr")], scan, np.random.default_rng(0))
    assert scan["rows_scanned"] == 3
    assert scan["rows_sf_wdl"] == 3
    assert scan["rows_missing_join_metadata"] == 1
    assert scan["cast_pairs"] == 0


def test_collect_rejects_claimed_present_missing_array(tmp_path: Path) -> None:
    from chess_anti_engine.replay import ReplaySample
    from chess_anti_engine.replay.shard import samples_to_arrays

    policy = np.full(WIDTH, 1.0 / WIDTH, dtype=np.float32)
    arrs = samples_to_arrays([ReplaySample(x=_planes(chess.Board()), policy_target=policy, wdl_target=1)])
    arrs["has_sf_wdl"] = np.ones(1, dtype=np.uint8)
    arrs.pop("sf_wdl", None)
    path = tmp_path / "shard_000001.npz"
    np.savez(path, allow_pickle=False, **arrs)
    scan = cast_probe.new_scan(tmp_path, [path])
    result = cast_probe.collect([path], scan, np.random.default_rng(0))
    assert scan["rows_scanned"] == 0
    assert "has_sf_wdl is set but sf_wdl is missing" in scan["skipped_shards"][0]["reason"]
    assert result.adv == []


@pytest.mark.parametrize("omitted", [None, "sf_wdl", "sf_p0_regret", "sf_multipv_raw", "legal_mask"])
def test_collect_only_reads_eligible_optional_fields(monkeypatch: pytest.MonkeyPatch, omitted: str | None) -> None:
    boards = _line(["e4", "e5", "Nf3"])
    covered = {_canon_index(boards[1], m): 0.05 for m in list(boards[1].legal_moves)[:2]}
    arrs, meta = _shard([
        {"board": b, "ply_index": i, "covered": covered, "regret": covered}
        for i, b in enumerate(boards)
    ])
    if omitted is not None:
        del arrs[omitted]
        arrs[f"has_{omitted}"][:] = 0
    monkeypatch.setattr(cast_probe, "load_shard_arrays", lambda _: (arrs, meta))
    scan = cast_probe.new_scan("fixture", [Path("fixture.zarr")])
    result = cast_probe.collect([Path("fixture.zarr")], scan, np.random.default_rng(0))
    assert scan["cast_pairs"] == (0 if omitted == "sf_wdl" else 3)
    if omitted is None:
        assert scan["cast_pairs_with_p0"] > 0
        assert result.adv
    else:
        assert scan["cast_pairs_with_p0"] == 0
        assert result.adv == []


def test_collect_mixed_shards_preserves_usable_rows_and_bounds_skip_details(monkeypatch: pytest.MonkeyPatch) -> None:
    boards = _line(["e4", "e5", "Nf3"])
    covered = {_canon_index(boards[1], m): 0.05 for m in list(boards[1].legal_moves)[:2]}
    valid, meta = _shard([
        {"board": b, "ply_index": i, "covered": covered, "regret": covered}
        for i, b in enumerate(boards)
    ])
    absent = dict(valid)
    del absent["game_id"]
    del absent["has_game_id"]
    good = Path("good.zarr")
    monkeypatch.setattr(cast_probe, "load_shard_arrays", lambda p: (valid if p == good else absent, meta))
    expected = cast_probe.collect([good], cast_probe.new_scan("fixture", [good]), np.random.default_rng(0))
    paths = [Path(f"absent-{i}.zarr") for i in range(25)] + [good]
    scan = cast_probe.new_scan("fixture", paths)
    result = cast_probe.collect(paths, scan, np.random.default_rng(0))
    for key, values in expected.arrays().items():
        np.testing.assert_array_equal(result.arrays()[key], values)
    assert result.adv
    assert scan["rows_scanned"] == 104
    assert scan["rows_missing_join_metadata"] == 100
    assert scan["rows_sf_wdl"] == 104
    assert scan["cast_pairs"] == 3
    assert len(scan["skipped_shards"]) == 20
    assert scan["skipped_shards_omitted"] == 5


def _fixed_draw_tail_fixture() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Nine selected rows: six inside (three regret buckets), three outside."""
    sel = np.zeros(18, dtype=bool)
    sel[1::2] = True
    arr = {
        "regret_played": np.full(18, 999.0),
        "adv": np.full(18, 999.0),
        "in_multipv": np.zeros(18, dtype=bool),
    }
    arr["regret_played"][sel] = [0.0, 0.001, 0.0015, 0.011, 0.02, 0.025, 0.55, 0.55, 0.55]
    arr["adv"][sel] = [-0.01, -0.02, -0.03, -0.10, -0.20, -0.30, -0.04, -0.08, -0.12]
    arr["in_multipv"][sel] = [True] * 6 + [False] * 3
    return arr, sel


class _ScriptedDraws:
    def __init__(self, picks: list[list[int]]) -> None:
        self.picks = list(picks)

    def integers(self, low: int, high: int, size: int) -> np.ndarray:
        assert (low, high, size) == (0, 9, 9)
        return np.asarray(self.picks.pop(0))


# A pick that survives both gates, and one with NO outside rows, which the
# MIN_BUCKET gate on the outside set drops.
_SURVIVES = list(range(9))
_DROPPED = [0, 1, 2, 3, 4, 5, 0, 1, 2]


@pytest.mark.parametrize(("picks", "n_kept", "reliable"), [
    ([_SURVIVES] * 4, 4, True),
    ([_SURVIVES] * 3 + [_DROPPED], 3, False),   # 75%: CI finite but conditional
    ([_SURVIVES] + [_DROPPED] * 3, 1, False),   # 25%: still >= N_BOOT // 4
    ([_DROPPED] * 4, 0, False),                 # no interval at all
])
def test_tail_bootstrap_reports_survival_beside_the_interval(
    monkeypatch: pytest.MonkeyPatch, picks: list[list[int]], n_kept: int, reliable: bool,
) -> None:
    """Dropped draws make the percentile interval CONDITIONAL on survival; the
    result must say how many survived and flag a low share, without changing
    the point estimate or which draws build the interval."""
    monkeypatch.setattr(cast_probe, "MIN_BUCKET", 2)
    monkeypatch.setattr(cast_probe, "N_BOOT", 4)
    arr, sel = _fixed_draw_tail_fixture()
    res = cast_probe.price_the_tail(arr, sel, "selected", _ScriptedDraws(picks))
    point = 1000 * (0.001 + 0.019 * (0.06 / 0.18))
    assert res["implied_cp"] == pytest.approx(point)
    assert res["n_bootstrap"] == n_kept
    assert res["n_bootstrap_attempted"] == 4
    assert res["bootstrap_survival_frac"] == pytest.approx(n_kept / 4)
    assert res["ci_reliable"] is reliable
    if n_kept:
        # The surviving draws all replay the full sample, so the interval is
        # the point estimate -- dropped draws are not used, survivors are.
        assert res["implied_cp_lo"] == pytest.approx(point)
        assert res["implied_cp_hi"] == pytest.approx(point)
    else:
        assert np.isnan(res["implied_cp_lo"])


def _report_arrays(n: int = 60) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)

    def f() -> np.ndarray:
        return rng.random(n)

    return {
        "adv": -f(), "regret_played": f(), "in_multipv": np.arange(n) % 3 != 0,
        "is_sf_best": np.arange(n) % 2 == 0, "pmax": f(), "abs_q_parent": f(),
        "n_legal": np.full(n, 20.0), "n_covered": np.full(n, 6.0), "mass_covered": f(),
        "er_total": f() + 0.1, "er_imputed": f() * 0.1, "er_total_shuf": f() + 0.1,
        "er_imputed_shuf": f() * 0.1, "imputed_value": f(), "worst_covered": f(),
        "priority_policy_kl": f(), "priority_q_delta": f(),
        "played_is_argmax": np.arange(n) % 4 != 0, "ply": np.arange(n) % 40,
        "mean_regret_row": f() + 0.1,
    }


def test_report_prints_and_banks_bootstrap_survival(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The printed tail-pricing row carries `boot k/N (p%)` and marks a
    low-survival interval; the banked JSON carries the same fields."""
    import json

    def fake_price(arr: Any, sel: Any, label: str, rng: Any) -> dict[str, Any]:
        del arr, sel, rng
        kept = 400 if label == "pmax>=0.0" else 180
        return {
            "stratum": label, "n": 100, "n_outside": 30, "calibration": [],
            "outside_frac": 0.3, "mean_adv_outside": -0.01, "implied_cp": 20.0,
            "implied_cp_lo": 15.0, "implied_cp_hi": 25.0, "assigned_cp": 550.0,
            "overstatement": 27.5, "inversion_unstable": False,
            "n_bootstrap": kept, "n_bootstrap_attempted": 400,
            "bootstrap_survival_frac": kept / 400, "ci_reliable": kept == 400,
        }

    monkeypatch.setattr(cast_probe, "price_the_tail", fake_price)
    scan = cast_probe.new_scan("replay", [])
    out = cast_probe.report(_report_arrays(), scan, np.random.default_rng(0))
    text = capsys.readouterr().out
    rows = {line.split()[0]: line for line in text.splitlines() if "assigned" in line}
    assert "boot 400/400 (100%)" in rows["pmax>=0.0"]
    assert "CI UNRELIABLE" not in rows["pmax>=0.0"]
    assert "boot 180/400 (45%)" in rows["pmax>=0.5"]
    assert "CI UNRELIABLE" in rows["pmax>=0.5"]
    assert "⚑⚑ CI UNRELIABLE means fewer than 95%" in text
    banked = json.loads(json.dumps(out))["tail_pricing"]
    assert [r["ci_reliable"] for r in banked] == [True, False, False, False]
    assert banked[1]["bootstrap_survival_frac"] == pytest.approx(0.45)


def test_report_runs_the_real_tail_pricing_with_survival_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End to end through the real `price_the_tail`: every priced stratum
    carries the survival fields and the printed row shows them."""
    rng = np.random.default_rng(0)
    n = 4000
    reg = np.concatenate([rng.choice([0.0, 0.02, 0.09], size=n), np.full(400, 0.55)])
    adv = np.concatenate([-0.5 * reg[:n], np.full(400, -0.01)])
    m = n + 400
    arr = _report_arrays(m)
    arr.update({"adv": adv, "regret_played": reg,
                "in_multipv": np.concatenate([np.ones(n, bool), np.zeros(400, bool)]),
                "pmax": np.ones(m)})
    out = cast_probe.report(arr, cast_probe.new_scan("replay", []), np.random.default_rng(0))
    text = capsys.readouterr().out
    priced = [r for r in out["tail_pricing"] if "implied_cp" in r]
    assert priced
    for r in priced:
        assert r["n_bootstrap_attempted"] == cast_probe.N_BOOT
        assert r["bootstrap_survival_frac"] == pytest.approx(r["n_bootstrap"] / cast_probe.N_BOOT)
        assert f"boot {r['n_bootstrap']}/{cast_probe.N_BOOT}" in text
