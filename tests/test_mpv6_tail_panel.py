"""Tests for scripts/mpv6_tail_panel.py.

The panel's whole value is that its three arms are DIFFERENT searches, so the
tests concentrate on the places where the arms' numbers could quietly get mixed:
which arm supplies which regret, and whether the cross-arm confound check can
actually report a confound. The fixture arms disagree on the SAME move by
construction, so no test here can pass on a cross-wired arm by coincidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import chess
import numpy as np
import pytest

from tests.script_loading import load_script_module

panel = load_script_module("mpv6_tail_panel.py")
CAP = panel.SF_OWN_REGRET_CAP_CP


@dataclass
class FakePV:
    move_uci: str
    cp: int | None = None
    mate: int | None = None


START = chess.Board()
ENC = "lc0_1858"


def _idx(uci: str, board: chess.Board = START) -> int:
    from chess_anti_engine.moves.encode import move_to_index_for_encoding
    return int(move_to_index_for_encoding(chess.Move.from_uci(uci), board, policy_encoding=ENC))


def test_score_of_mirrors_the_finalize_scorer() -> None:
    assert panel.score_of(FakePV("e2e4", cp=35)) == 35.0
    assert panel.score_of(FakePV("e2e4", cp=None)) is None
    assert panel.score_of(FakePV("e2e4", cp=10, mate=0)) == 10.0
    # A mate must DOMINATE any centipawn score, and a faster mate must outrank a slower.
    # (mate 3 reads 99700 — well past the 1000 cp regret cap, which is the point.)
    assert panel.score_of(FakePV("e2e4", mate=3)) > 30_000
    assert panel.score_of(FakePV("e2e4", mate=1)) > panel.score_of(FakePV("e2e4", mate=5))
    assert panel.score_of(FakePV("e2e4", mate=-3)) < -30_000


def test_regrets_from_sorts_best_first_and_drops_unscored() -> None:
    pvs = [FakePV("a2a3", cp=-40), FakePV("e2e4", cp=35),
           FakePV("d2d4", cp=None), FakePV("b1c3", cp=10)]
    scored = panel.regrets_from(pvs, START, ENC)
    assert [uci for uci, _ in
            [(m, s) for m, s in scored]] == [_idx("e2e4"), _idx("b1c3"), _idx("a2a3")]
    assert len(scored) == 3, "the cp-less PV must not become a scored candidate"


def test_regrets_from_drops_illegal_moves() -> None:
    """⚑ The encoder is NOT a legality check, so `regrets_from` has to be one.

    `move_to_index_for_encoding(e7e5, <white to move>)` returns index 1531
    without complaint. If the panel relied on the encoder to reject off-position
    moves, a board/FEN desync would enter the calibration as plausible indices.
    Removing the `board.is_legal` guard makes this test read 2.
    """
    scored = panel.regrets_from([FakePV("e2e4", cp=0), FakePV("e7e5", cp=-5)], START, ENC)
    assert len(scored) == 1
    assert scored[0][0] == _idx("e2e4")


def test_to_regret_is_relative_to_that_arms_own_best() -> None:
    scored = [(1, 100.0), (2, 40.0), (3, -900.0)]
    reg = panel.to_regret(scored)
    assert reg[1] == 0.0
    assert reg[2] == pytest.approx(60 / CAP)
    assert reg[3] == pytest.approx(1.0), "1000 cp is the cap"


def _result(
    prod: list[tuple[int, float]],
    matched: list[tuple[int, float]],
    deep: list[tuple[int, float]],
    prior: np.ndarray, legal: list[int], *, in_tb: bool = False,
) -> dict[str, Any]:
    return {
        "prod_scored": prod, "matched_scored": matched, "deep_scored": deep,
        "prior": prior, "legal_idx": np.array(legal), "in_tb": in_tb,
        "n_legal": len(legal),
        "prod_width": len(prod), "matched_width": len(matched), "deep_width": len(deep),
    }


# The three arms deliberately disagree on the SAME move 11: PROD calls the gap
# 50 cp, MATCHED 60 cp, DEEP 150 cp. Any cross-wiring of the arms moves a number.
PROD = [(10, 0.0), (11, -50.0)]
MATCHED = [(10, 0.0), (11, -60.0), (12, -300.0), (13, -500.0)]
DEEP = [(10, 0.0), (11, -150.0), (12, -400.0), (13, -600.0)]
PRIOR = np.array([0.4, 0.3, 0.2, 0.1])
LEGAL = [10, 11, 12, 13]


def _one(**kw: Any) -> dict[str, Any]:
    return _result(PROD, MATCHED, DEEP, PRIOR, LEGAL, **kw)


def test_substituted_keeps_prod_regrets_for_surfaced_moves() -> None:
    """⚑ Mutation: sourcing the SURFACED regrets from the truth arm must change this.

    `r_k` is the censoring point production actually applies, so it has to be the
    PROD arm's 50 cp — not MATCHED's 60 nor DEEP's 150 reading of the same move.
    """
    row = panel.build_rows([_one()], "matched_scored", substitute=True)[0]
    assert row.r_k == pytest.approx(50 / CAP)
    assert row.regret[11] == pytest.approx(50 / CAP), "surfaced move keeps the PROD value"
    assert row.regret[12] == pytest.approx(300 / CAP), "hidden move takes MATCHED's value"
    assert row.surfaced == [10, 11]
    assert sorted(row.hidden) == [12, 13]


def test_the_truth_arm_selects_which_search_prices_the_tail() -> None:
    """⚑ The primary and the truth panel must not be reading the same arm."""
    m = panel.build_rows([_one()], "matched_scored", substitute=True)[0]
    d = panel.build_rows([_one()], "deep_scored", substitute=True)[0]
    assert m.regret[12] == pytest.approx(300 / CAP)
    assert d.regret[12] == pytest.approx(400 / CAP)
    assert m.r_k == pytest.approx(d.r_k), "both keep PROD's censoring point"


def test_non_substituted_takes_every_regret_from_the_truth_arm() -> None:
    row = panel.build_rows([_one()], "deep_scored", substitute=False)[0]
    assert row.r_k == pytest.approx(150 / CAP), "DEEP's reading of the surfaced move"
    assert row.regret[11] == pytest.approx(150 / CAP)


def test_substituted_and_all_modes_disagree_on_this_fixture() -> None:
    """If these ever coincide, one mode has been wired to the other's source."""
    sub = panel.build_rows([_one()], "deep_scored", substitute=True)[0]
    alld = panel.build_rows([_one()], "deep_scored", substitute=False)[0]
    assert sub.r_k != pytest.approx(alld.r_k)


def test_row_without_a_hidden_tail_is_dropped() -> None:
    """Nothing to price when the truth arm surfaced nothing new."""
    r = _result(PROD, PROD, PROD, PRIOR[:2], [10, 11])
    assert panel.build_rows([r], "matched_scored", substitute=True) == []


def test_row_without_a_prior_is_dropped() -> None:
    r = _one()
    r["prior"] = None
    assert panel.build_rows([r], "matched_scored", substitute=True) == []


def test_cross_arm_check_reports_the_confound_it_exists_for() -> None:
    """The guard must SEE a deeper arm scoring the same moves differently.

    This fired for real on the first smoke run: MultiPV 64 at 200k nodes reached
    depth 10 against the prod arm's 12, and the ratio read 1.87 — that "wide" arm
    was not a ruler at all. A check that cannot produce such a number is decorative.
    """
    xa = panel.cross_arm_check([_one()], "prod_scored", "deep_scored")
    assert xa["n_moves"] == 2, "only the moves BOTH arms surfaced"
    assert xa["a_mean_cp"] == pytest.approx(25.0)     # (0 + 50) / 2
    assert xa["b_mean_cp"] == pytest.approx(75.0)     # (0 + 150) / 2
    assert xa["ratio"] == pytest.approx(3.0)
    assert xa["bestmove_agreement"] == pytest.approx(1.0)


def test_cross_arm_check_is_directional() -> None:
    """⚑ Swapping the arms must invert the ratio, not leave it alone.

    A check that reported the same number either way would be comparing something
    other than the pair it was asked about.
    """
    fwd = panel.cross_arm_check([_one()], "prod_scored", "deep_scored")["ratio"]
    rev = panel.cross_arm_check([_one()], "deep_scored", "prod_scored")["ratio"]
    assert rev == pytest.approx(1.0 / fwd)


def test_cross_arm_ratio_is_one_when_the_arms_agree() -> None:
    same = [(10, 0.0), (11, -50.0), (12, -400.0)]
    r = _result(PROD, same, same, PRIOR[:3], [10, 11, 12])
    assert panel.cross_arm_check([r], "prod_scored", "matched_scored")["ratio"] == pytest.approx(1.0)


def test_cross_arm_check_sees_a_bestmove_disagreement() -> None:
    flipped = [(11, 0.0), (10, -30.0), (12, -400.0)]
    r = _result(PROD, flipped, flipped, PRIOR[:3], [10, 11, 12])
    xa = panel.cross_arm_check([r], "prod_scored", "matched_scored")
    assert xa["bestmove_agreement"] == pytest.approx(0.0)


def test_tablebase_limit_matches_syzygy() -> None:
    """Syzygy tops out at 7 men, and the two arms carry DIFFERENT tablebase sets."""
    assert panel.TB_PIECE_LIMIT == 7


def test_prod_arm_defaults_are_the_realized_live_settings() -> None:
    """⚑ These are measurements, not config reads.

    `sf_label_nodes_floor` is 150000 and `sf_label_nodes_cap` is 200000; the
    realized median of `sf_label_meta[:,0]` on the live shards is 150071, i.e.
    pinned at the FLOOR. Defaulting to the cap would make the prod arm stronger
    than production and shrink the measured tail.
    """
    assert panel.PROD_NODES == 150_000
    assert panel.PROD_MULTIPV == 6
    assert panel.PROD_HASH_MB == 17
    # ⚑ stockfish_syzygy_path is the 3-4-5 directory ALONE. syzygy_path -- the
    # colon-separated pair -- is what the PLAYING engines get, not the labeller.
    assert ":" not in panel.PROD_SYZYGY
    assert ":" in panel.DEEP_SYZYGY


def _fake_shard_arrays(n_rows: int) -> dict[str, Any]:
    """A shard of `n_rows` DISTINCT positions with plenty of legal moves.

    Built by walking a real game so `decode_board_from_planes` round-trips and
    `position_key` sees genuinely different positions — a fixture of repeats
    would let a broken deduper look like a working quota.
    """
    from chess_anti_engine.encoding.encode import encode_position

    boards: list[chess.Board] = []
    b = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5",
             "c2c3", "d7d6", "b2b4", "c5b6", "a2a4", "a7a6", "b1d2", "e8g8"]
    for uci in moves:
        boards.append(b.copy(stack=False))
        b.push(chess.Move.from_uci(uci))
    boards = (boards * 10)[:n_rows]
    xs = np.stack([
        encode_position(bd, input_history_encoding="lc0_root_legacy_meta",
                        input_extra_features="v1")
        for bd in boards
    ])
    return {
        "x": xs,
        "legal_mask": np.ones((n_rows, 1858), dtype=bool),
        "_input_history_encoding": np.array(["lc0_root_legacy_meta"]),
        "_policy_encoding": np.array([ENC]),
    }


def test_sampling_spreads_across_shards_instead_of_draining_the_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """⚑ Mutation: filling from the front must not satisfy this.

    A shard is one worker's upload batch — a few dozen games over a few minutes.
    Draining shard 0 would turn N "independent" positions into a handful of
    correlated games and hand the row bootstrap an interval it never earned.
    Setting the per-shard quota to `want` (i.e. no quota) makes this read 1.
    """
    arrs = _fake_shard_arrays(40)
    monkeypatch.setattr(panel, "load_shard_arrays", lambda _p: (arrs, {}))
    shards = [tmp_path / f"shard_{i:03d}.zarr" for i in range(4)]
    scan: dict[str, Any] = {"rows_scanned": 0, "skipped_narrow": 0,
                            "skipped_shards": [], "skipped_shards_omitted": 0}
    out = panel.sample_positions(shards, scan, 8, 0)
    assert len(out) == 8
    assert scan["sampled_shards"] == 4, "each shard must contribute its quota"
    assert len({o["shard"] for o in out}) == 4


def test_sampling_backfills_when_a_shard_is_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any,
) -> None:
    """A quota is a target, not a cap on the total: short shards must be covered."""
    arrs = _fake_shard_arrays(40)
    monkeypatch.setattr(panel, "load_shard_arrays", lambda _p: (arrs, {}))
    shards = [tmp_path / f"shard_{i:03d}.zarr" for i in range(2)]
    scan: dict[str, Any] = {"rows_scanned": 0, "skipped_narrow": 0,
                            "skipped_shards": [], "skipped_shards_omitted": 0}
    # 16 distinct positions exist in total; asking for 30 must return all of them
    # rather than stopping at the first shard's quota of 15.
    out = panel.sample_positions(shards, scan, 30, 0)
    assert len(out) == 16
    assert scan["sampled_shards"] == 2
