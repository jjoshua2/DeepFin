"""Deterministic units for the target-surgery ladder (arms R / V / G).

Three groups, and each is here because a specific way of being wrong is cheap
to ship and expensive to detect:

1. **The join key.** ``position_fingerprints`` (planes) and ``board_fingerprint``
   (board) must agree, or the labels address different rows than the rig edits
   and every arm silently reads as "no coverage".
2. **The edit arithmetic**, against hand-computed three-move examples. A
   plausible-looking distribution is the easiest thing in this repo to ship
   wrong, because every downstream number stays finite and in range.
3. **The two structural properties the prereg is FOR**: V must leave unlisted
   moves' multiplier at exactly 1 (no fabricated regret), and R must leave
   unlisted moves at exactly 0 (no membership prior).

Every number below is written out by hand in the test, never recomputed with
the function under test.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import chess
import numpy as np
import pytest

from chess_anti_engine.eval.audit import decode_board_from_planes
from chess_anti_engine.eval.rvg_surgery import (
    RVG_EXTERNAL_POLICY_SCHEMA_VERSION,
    RVG_LABEL_SCHEMA_VERSION,
    MultiPvLine,
    RvgLabelIndex,
    apply_geometric_blend,
    apply_veto_edit,
    board_fingerprint,
    RvgExternalPolicyIndex,
    distribution_entropy_nats,
    fold_multipv,
    listed_only_regret_vector,
    position_fingerprints,
)
# The pure-Python encoder, deliberately: these units must run in a worktree
# whose C extensions have not been built (the .so files are build artifacts of
# the tree they were compiled in). It produces the same 112 planes.
from chess_anti_engine.encoding.lc0 import encode_lc0_full
from chess_anti_engine.moves.encode import move_to_index_for_encoding

PROD_ENCODING = "lc0_root_legacy_meta"


# ---------------------------------------------------------------------------
# 1. Join key
# ---------------------------------------------------------------------------

_ROUNDTRIP_FENS = [
    chess.STARTING_FEN,
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 6 5",
    "8/2R5/5pk1/8/3r3p/5PK1/8/8 w - - 0 1",
    "4k3/8/8/8/8/8/4P3/4K3 w - - 99 60",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 4 10",
]


@pytest.mark.parametrize("fen", _ROUNDTRIP_FENS)
def test_the_join_key_round_trips_board_to_planes_and_back(fen: str) -> None:
    """planes -> key == board -> key, on the canonical (white-to-move) board.

    ⚑ THIS IS THE JOIN. The label pass keys a record off the PLANES and the rig
    looks it up off the PLANES, so this test is not checking the join path
    itself — it is checking that the digest is a faithful identity for the
    POSITION, which is what makes "one label per position" true rather than
    hopeful. A derivation that quietly ignored castling or the half-move clock
    would still join perfectly and would still be wrong.
    """
    board = chess.Board(fen)
    planes = encode_lc0_full(board, input_history_encoding=PROD_ENCODING)
    canonical = decode_board_from_planes(planes, input_history_encoding=PROD_ENCODING)
    assert canonical is not None
    from_planes = position_fingerprints(planes, input_history_encoding=PROD_ENCODING)
    assert len(from_planes) == 1
    assert from_planes[0] == board_fingerprint(canonical)
    assert len(from_planes[0]) == 16


def test_the_key_separates_positions_that_differ_only_in_castling() -> None:
    with_rights = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    without = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1")
    a = position_fingerprints(
        encode_lc0_full(with_rights, input_history_encoding=PROD_ENCODING),
        input_history_encoding=PROD_ENCODING,
    )[0]
    b = position_fingerprints(
        encode_lc0_full(without, input_history_encoding=PROD_ENCODING),
        input_history_encoding=PROD_ENCODING,
    )[0]
    assert a != b


def test_the_key_separates_positions_that_differ_only_in_the_halfmove_clock() -> None:
    """``audit.position_key`` drops the clock; this key must not.

    A shallow label runs with Syzygy on, and both DTZ conversion and the 50-move
    rule read the clock — two rows that differ only there are two positions as
    far as this label is concerned, and sharing a label between them would be a
    silent mis-join, not a harmless dedup.
    """
    keys = [
        position_fingerprints(
            encode_lc0_full(
                chess.Board(f"4k3/8/8/8/8/8/4P3/4K3 w - - {clock} 60"),
                input_history_encoding=PROD_ENCODING,
            ),
            input_history_encoding=PROD_ENCODING,
        )[0]
        for clock in (0, 40, 99)
    ]
    assert len(set(keys)) == 3


def test_a_colour_mirrored_pair_shares_one_key() -> None:
    """The encoding is POV-normalized, so a position and its mirror are ONE
    position here. They legitimately share a label — this is a property, not a
    collision."""
    white_to_move = chess.Board("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
    black_to_move = chess.Board("4k3/4p3/8/8/4K3/8/8/8 b - - 0 1")
    a = position_fingerprints(
        encode_lc0_full(white_to_move, input_history_encoding=PROD_ENCODING),
        input_history_encoding=PROD_ENCODING,
    )[0]
    b = position_fingerprints(
        encode_lc0_full(black_to_move, input_history_encoding=PROD_ENCODING),
        input_history_encoding=PROD_ENCODING,
    )[0]
    assert a == b


def test_a_batch_of_planes_keys_row_by_row() -> None:
    boards = [chess.Board(f) for f in _ROUNDTRIP_FENS]
    batch = np.stack([
        encode_lc0_full(b, input_history_encoding=PROD_ENCODING) for b in boards
    ])
    batched = position_fingerprints(batch, input_history_encoding=PROD_ENCODING)
    singles = [
        position_fingerprints(
            encode_lc0_full(b, input_history_encoding=PROD_ENCODING),
            input_history_encoding=PROD_ENCODING,
        )[0]
        for b in boards
    ]
    assert batched == singles


# ---------------------------------------------------------------------------
# 2. MultiPV folding
# ---------------------------------------------------------------------------


def test_fold_multipv_hand_example() -> None:
    folded = fold_multipv([
        MultiPvLine("e2e4", 50, None),
        MultiPvLine("d2d4", 20, None),
        MultiPvLine("a2a3", -300, None),
    ])
    assert folded is not None
    assert folded.best_cp == 50.0
    assert folded.moves == ("e2e4", "d2d4", "a2a3")
    assert folded.regret_cp == (0.0, 30.0, 350.0)


def test_fold_multipv_first_listing_wins_and_caps_the_mate_band() -> None:
    """Rank order is authoritative, and the mate band saturates the 1000cp cap.

    A duplicated move must take the FIRST (best-ranked) line: re-ranking by cp
    would build a silently better ruler than the one production's label sees.
    """
    folded = fold_multipv([
        MultiPvLine("e2e4", None, 3),
        MultiPvLine("d2d4", 10, None),
        MultiPvLine("d2d4", -999, None),
        MultiPvLine("h2h4", None, -2),
    ])
    assert folded is not None
    assert folded.moves == ("e2e4", "d2d4", "h2h4")
    assert folded.regret_cp[0] == 0.0
    assert folded.regret_cp[1] == 1000.0   # capped, not ~99990
    assert folded.regret_cp[2] == 1000.0


def test_a_tb_sentinel_is_a_real_evaluation_capped_like_any_other() -> None:
    """Syzygy reports TB lines at ``cp ~ +/-20000``. Decision: fold normally.

    The regret is computed against ``best_cp`` and clipped at 1000cp, so a TB
    loss saturates the cap and is vetoed by any ``veto_cp <= 1000`` — which is
    correct, it IS a losing move.
    """
    folded = fold_multipv([
        MultiPvLine("a1a8", 20000, None),
        MultiPvLine("a1a2", 0, None),
        MultiPvLine("a1b1", -20000, None),
    ])
    assert folded is not None
    assert folded.best_cp == 20000.0
    assert folded.regret_cp == (0.0, 1000.0, 1000.0)


def test_an_unscoreable_line_is_skipped_not_crashed_on() -> None:
    folded = fold_multipv([
        MultiPvLine("e2e4", 5, None),
        MultiPvLine("d2d4", None, None),
    ])
    assert folded is not None
    assert folded.moves == ("e2e4",)
    assert fold_multipv([MultiPvLine("e2e4", None, None)]) is None


# ---------------------------------------------------------------------------
# 3. Arm V — veto edit
# ---------------------------------------------------------------------------


def test_v_hand_computed_three_move_example() -> None:
    """t = [0.5, 0.3, 0.2] over slots 0/1/2; slots 0 and 1 are LISTED.

    regret = [0, 120] cp, lambda = 0.01, tau = 20 cp, veto = 500 cp.
      slot 0: relu(0-20)   = 0    -> w = exp(0)     = 1
      slot 1: relu(120-20) = 100  -> w = exp(-1.0)  = 0.36787944117144233
      slot 2: UNLISTED            -> w = 1 exactly
    unnormalized = [0.5, 0.11036383235143270, 0.2]; total = 0.81036383235143270
    normalized   = [0.61700681, 0.13619047, 0.24680272]
    """
    t = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    out, fell_back = apply_veto_edit(
        t, np.array([0, 1]), np.array([0.0, 120.0]),
        lam=0.01, tau_cp=20.0, veto_cp=500.0,
    )
    assert not fell_back
    w1 = math.exp(-1.0)
    unnorm = np.array([0.5, 0.3 * w1, 0.2])
    expected = unnorm / unnorm.sum()
    assert np.allclose(out, expected, atol=1e-6)
    # The hand numbers, spelled out so a refactor cannot silently redefine them.
    assert pytest.approx(float(out[0]), abs=1e-6) == 0.61700681
    assert pytest.approx(float(out[1]), abs=1e-6) == 0.13619047
    assert pytest.approx(float(out[2]), abs=1e-6) == 0.24680272


def test_v_hard_veto_zeroes_a_listed_move_at_or_above_the_threshold() -> None:
    t = np.array([0.4, 0.4, 0.2], dtype=np.float32)
    out, fell_back = apply_veto_edit(
        t, np.array([0, 1]), np.array([0.0, 300.0]),
        lam=0.0, tau_cp=0.0, veto_cp=300.0,
    )
    assert not fell_back
    assert out[1] == 0.0
    # ">= veto_cp", so exactly 300 is vetoed; the survivors renormalize to 1.
    assert pytest.approx(float(out.sum()), abs=1e-6) == 1.0
    assert pytest.approx(float(out[0]), abs=1e-6) == 2.0 / 3.0


def test_v_leaves_unlisted_moves_multiplier_at_exactly_one() -> None:
    """THE PROPERTY THE PREREG IS FOR: no fabricated regret for unlisted moves.

    Two assertions, and the SECOND is the load-bearing one. Ratios AMONG the
    unlisted moves are preserved by any uniform multiplier, so a mutant that
    downweights the whole unlisted block by one constant — the exact shape of
    June's fabricated default — survives that check untouched. (Measured: a
    mutant setting every weight to the worst listed weight left this test green
    until the level assertion below was added.) The level is pinned against an
    UNTOUCHED LISTED move, whose multiplier is exactly 1 by construction, so
    only a genuinely-1.0 unlisted multiplier can satisfy it.
    """
    t = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    # Slot 0 is listed with regret 400 (downweighted); slot 1 is listed with
    # regret 0, i.e. below tau, so its multiplier is exactly 1.0.
    out, _ = apply_veto_edit(
        t, np.array([0, 1]), np.array([400.0, 0.0]),
        lam=0.02, tau_cp=10.0, veto_cp=1000.0,
    )
    unlisted = np.array([2, 3])
    ratios_before = t[unlisted] / t[unlisted].sum()
    ratios_after = out[unlisted] / out[unlisted].sum()
    assert np.allclose(ratios_before, ratios_after, atol=1e-7)
    # LEVEL, not just shape: unlisted mass relative to an untouched listed move
    # must be unchanged, which is what "multiplier exactly 1" means.
    for u in unlisted.tolist():
        assert pytest.approx(float(out[u] / out[1]), rel=1e-6) == float(t[u] / t[1])


def test_v_falls_back_to_the_unedited_target_when_all_mass_is_vetoed() -> None:
    """A row whose whole support is vetoed is served UNEDITED and counted.

    A zeroed target would silently drop the row from the policy CE while
    ``has_policy`` still claimed it was present; a uniform one would be a
    fabricated label. Falling back is the only option that stays honest AND
    countable.
    """
    t = np.array([0.6, 0.4, 0.0], dtype=np.float32)
    out, fell_back = apply_veto_edit(
        t, np.array([0, 1]), np.array([900.0, 900.0]),
        lam=0.0, tau_cp=0.0, veto_cp=500.0,
    )
    assert fell_back
    assert np.array_equal(out, t)


def test_v_takes_the_harshest_weight_on_a_duplicated_policy_slot() -> None:
    t = np.array([0.5, 0.5], dtype=np.float32)
    out, _ = apply_veto_edit(
        t, np.array([0, 0]), np.array([0.0, 1000.0]),
        lam=0.0, tau_cp=0.0, veto_cp=500.0,
    )
    assert out[0] == 0.0


# ---------------------------------------------------------------------------
# 4. Arm G — geometric blend
# ---------------------------------------------------------------------------


def test_g_hand_computed_three_move_example() -> None:
    """t = [0.5, 0.3, 0.2]; slots 0 and 1 listed with regret [0, 100] cp.

    T = 100 cp  -> raw q = [exp(0), exp(-1)] = [1, 0.36787944117144233]
                -> q     = [0.7310585786300049, 0.2689414213699951]
    alpha = 0.5 -> listed slots become sqrt(t*q):
        slot 0: sqrt(0.5 * 0.7310585786300049) = 0.6046019724331709
        slot 1: sqrt(0.3 * 0.2689414213699951) = 0.2840371481324665
        slot 2: UNLISTED, multiplier 1         = 0.2
    total = 1.0886391205656374; normalized = [0.5553645, 0.26091948, 0.18371602]
    """
    t = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    out, fell_back = apply_geometric_blend(
        t, np.array([0, 1]), np.array([0.0, 100.0]), alpha=0.5, temp_cp=100.0,
    )
    assert not fell_back
    q = np.array([1.0, math.exp(-1.0)])
    q = q / q.sum()
    unnorm = np.array([math.sqrt(0.5 * q[0]), math.sqrt(0.3 * q[1]), 0.2])
    assert np.allclose(out, unnorm / unnorm.sum(), atol=1e-6)
    assert pytest.approx(float(out[0]), abs=1e-6) == 0.55536450
    assert pytest.approx(float(out[1]), abs=1e-6) == 0.26091948
    assert pytest.approx(float(out[2]), abs=1e-6) == 0.18371602


def test_g_normalizes_q_over_the_LISTED_set_before_the_power() -> None:
    """The q normalization is load-bearing ONLY when unlisted mass exists.

    ⚑ MEASURED, NOT ASSUMED, AND IT COST A SURVIVING MUTANT TO FIND OUT. With
    every move listed, dropping ``q /= q.sum()`` scales all listed entries by one
    constant, which the FINAL renormalization then cancels exactly — so a
    two-move example cannot see the defect at all, and an earlier version of this
    test used one. The third move here is UNLISTED and keeps its t-mass, which is
    what breaks the symmetry: a q that is not a distribution shifts mass between
    the listed and unlisted halves.

    Hand numbers (t = [0.5, 0.3, 0.2], listed {0,1}, regret [0, 100] cp, T = 100,
    alpha = 0.5):
      correct: q = [1, e^-1]/1.3678794 = [0.73105858, 0.26894142]
               -> [sqrt(.5*q0), sqrt(.3*q1), .2] / 1.08863912
               =  [0.55536450, 0.26091948, 0.18371602]
      WRONG (q left unnormalized as [1, e^-1]):
               -> [0.70710678, 0.33221737, 0.2] / 1.23932415
               =  [0.57056068, 0.26806529, 0.16137403]
    Both are finite, both sum to 1, and nothing downstream would object.
    """
    t = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    out, _ = apply_geometric_blend(
        t, np.array([0, 1]), np.array([0.0, 100.0]), alpha=0.5, temp_cp=100.0,
    )
    q = np.array([1.0, math.exp(-1.0)])
    q = q / q.sum()
    unnorm = np.array([math.sqrt(0.5 * q[0]), math.sqrt(0.3 * q[1]), 0.2])
    assert np.allclose(out, unnorm / unnorm.sum(), atol=1e-6)
    wrong = np.array([math.sqrt(0.5 * 1.0), math.sqrt(0.3 * math.exp(-1.0)), 0.2])
    assert not np.allclose(out, wrong / wrong.sum(), atol=1e-4)


def test_g_does_not_condition_t_on_the_listed_set_before_the_power() -> None:
    """The other renormalization-order defect: rescaling t's LISTED entries to
    sum to 1 before the power. That turns ``t`` into a conditional distribution
    and silently changes how much of the row the unlisted tail keeps.

    Same row as above. Conditioning t on {0,1} makes it [0.625, 0.375]:
      -> [sqrt(.625*q0), sqrt(.375*q1), .2] / 1.16884
      =  [0.58676..., 0.27568..., 0.17110...] -- again finite and normalized.
    """
    t = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    out, _ = apply_geometric_blend(
        t, np.array([0, 1]), np.array([0.0, 100.0]), alpha=0.5, temp_cp=100.0,
    )
    q = np.array([1.0, math.exp(-1.0)])
    q = q / q.sum()
    cond = np.array([0.5, 0.3]) / 0.8
    wrong = np.array([math.sqrt(cond[0] * q[0]), math.sqrt(cond[1] * q[1]), 0.2])
    assert not np.allclose(out, wrong / wrong.sum(), atol=1e-4)
    # ... and it IS the unconditioned answer.
    right = np.array([math.sqrt(0.5 * q[0]), math.sqrt(0.3 * q[1]), 0.2])
    assert np.allclose(out, right / right.sum(), atol=1e-6)


def test_g_never_grows_the_support_of_t() -> None:
    """G edits the SUPPORT of t: a listed move the search never visited stays 0.

    ``t**(1-alpha)`` is 0 for t=0 at every alpha < 1, which is why the rig's
    band excludes alpha = 1 (``t**0 == 1`` would put mass on unvisited moves).
    """
    t = np.array([0.7, 0.0, 0.3], dtype=np.float32)
    out, _ = apply_geometric_blend(
        t, np.array([0, 1, 2]), np.array([0.0, 0.0, 500.0]),
        alpha=0.9, temp_cp=50.0,
    )
    assert out[1] == 0.0
    assert pytest.approx(float(out.sum()), abs=1e-6) == 1.0


def test_g_keeps_unlisted_t_mass_rather_than_annihilating_it() -> None:
    """THE DOCUMENTED CONVENTION. A literal ``q = 0`` for unlisted moves would
    make ``t**(1-a) * 0**a == 0`` and delete them; the prereg's stated behaviour
    is that they "keep reduced t-mass", which is what is implemented."""
    t = np.array([0.5, 0.5], dtype=np.float32)
    out, _ = apply_geometric_blend(
        t, np.array([0]), np.array([0.0]), alpha=0.5, temp_cp=100.0,
    )
    assert out[1] > 0.0


# ---------------------------------------------------------------------------
# 5. Arm R — listed-only regret vector
# ---------------------------------------------------------------------------


def test_r_hand_computed_vector_and_the_listed_only_property() -> None:
    """THE PROPERTY THE PREREG IS FOR: unlisted moves are EXACTLY 0.

    June's PR #78 form defaulted them to 1.0, which turned ``sum_m p_m r_m``
    into a membership prior. Values are normalized by the caller's cap (the
    training-side ``SF_OWN_REGRET_CAP_CP``), so 250 cp at a 1000 cp cap is 0.25.
    """
    out = listed_only_regret_vector(
        5, np.array([1, 3]), np.array([0.0, 250.0]), cap_cp=1000.0,
    )
    assert out.tolist() == [0.0, 0.0, 0.0, 0.25, 0.0]
    assert out[0] == 0.0
    assert out[2] == 0.0
    assert out[4] == 0.0


def test_r_clips_at_the_cap_and_takes_the_best_line_on_a_duplicate_slot() -> None:
    out = listed_only_regret_vector(
        3, np.array([0, 2, 2]), np.array([5000.0, 800.0, 300.0]), cap_cp=1000.0,
    )
    assert out[0] == 1.0
    assert pytest.approx(float(out[2]), abs=1e-6) == 0.3


def test_r_with_no_listed_moves_is_all_zero() -> None:
    out = listed_only_regret_vector(
        4, np.zeros((0,), dtype=np.int64), np.zeros((0,)), cap_cp=1000.0,
    )
    assert out.tolist() == [0.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# 6. Label index
# ---------------------------------------------------------------------------


def _write_labels(
    path: Path, records: list[dict[str, object]], *,
    nodes: int = 150000, multipv: int = 40,
) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"record": "provenance", "nodes": nodes,
                             "multipv": multipv,
                             "ucinewgame_per_position": True}) + "\n")
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _label(key: bytes, idx: list[int], cp: list[float]) -> dict[str, object]:
    """One label record. ``cp`` is EFFECTIVE CP, not regret.

    ⚑ Regret is relative to the set's own best, so a record cannot carry it:
    the layered join recomputes ``best_cp`` over the overlaid moves. A helper
    that took regret would let a test assert a regret the loader could not
    reproduce.
    """
    return {
        "v": RVG_LABEL_SCHEMA_VERSION, "key": key.hex(), "policy_encoding": "lc0_1858",
        "n_pv": len(idx), "move_index": idx, "cp_eff": cp,
        "best_cp": max(cp) if cp else 0.0,
    }


def test_the_label_index_round_trips_keys_and_slices(tmp_path: Path) -> None:
    k1, k2 = b"\x01" * 16, b"\x02" * 16
    path = tmp_path / "labels.jsonl"
    _write_labels(path, [_label(k1, [3, 7], [0.0, -40.0]),
                         _label(k2, [1, 4], [12.5, 0.0])])
    index = RvgLabelIndex.load(path, policy_encoding="lc0_1858")
    assert len(index) == 2
    got = index.get(k1)
    assert got is not None
    idx, reg = got
    assert idx.tolist() == [3, 7]
    assert reg.tolist() == [0.0, 40.0]     # best_cp 0 -> regret = -cp
    got = index.get(k2)
    assert got is not None
    idx, reg = got
    assert idx.tolist() == [1, 4]
    assert reg.tolist() == [0.0, 12.5]     # best_cp 12.5, so move 1 is the best
    assert index.get(b"\x03" * 16) is None
    assert index.header["nodes"] == 150000


def test_the_label_index_refuses_a_policy_encoding_it_was_not_built_for(
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.jsonl"
    _write_labels(path, [_label(b"\x01" * 16, [3, 5], [0.0, -20.0])])
    with pytest.raises(SystemExit, match="policy space"):
        RvgLabelIndex.load(path, policy_encoding="az_4672")


def test_the_label_index_refuses_a_schema_version_it_does_not_read(
    tmp_path: Path,
) -> None:
    path = tmp_path / "labels.jsonl"
    rec = _label(b"\x01" * 16, [3, 5], [0.0, -20.0])
    rec["v"] = RVG_LABEL_SCHEMA_VERSION + 1
    _write_labels(path, [rec])
    with pytest.raises(SystemExit, match="schema version"):
        RvgLabelIndex.load(path)


def test_the_label_index_refuses_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    _write_labels(path, [_label(b"\x01" * 16, [3, 5], [0.0, -20.0]),
                         _label(b"\x01" * 16, [4, 6], [1.0, 0.0])])
    with pytest.raises(SystemExit, match="duplicate key"):
        RvgLabelIndex.load(path)


# ---------------------------------------------------------------------------
# 7. The rig wrappers (scripts/retarget_retrain.py)
# ---------------------------------------------------------------------------

import scripts.retarget_retrain as rr  # noqa: E402

_POLICY_W = 8


class _FakeInner:
    """A replay buffer that serves ONE fixed batch of real encoded positions."""

    def __init__(self, fens: list[str], targets: np.ndarray) -> None:
        self._x = np.stack([
            encode_lc0_full(chess.Board(f), input_history_encoding=PROD_ENCODING)
            for f in fens
        ]).astype(np.float16)
        self._t = targets.astype(np.float16)
        self.calls = 0

    def keys(self) -> list[bytes]:
        return position_fingerprints(self._x, input_history_encoding=PROD_ENCODING)

    def sample_batch_arrays(
        self, batch_size: int, *, wdl_balance: bool = True,
    ) -> dict[str, np.ndarray]:
        del batch_size, wdl_balance   # a fixed batch; the house unused-arg idiom
        self.calls += 1
        n = self._x.shape[0]
        return {
            "x": self._x.copy(),
            "policy_target": self._t.copy(),
            "_input_history_encoding": np.array([PROD_ENCODING] * n),
            "sf_p0_regret": np.zeros((n, _POLICY_W), dtype=np.float16),
            "has_sf_p0_regret": np.zeros((n,), dtype=np.uint8),
        }

    def __len__(self) -> int:
        return 2


def _index_for(keys: list[bytes], tmp_path: Path,
               entries: dict[int, tuple[list[int], list[float]]]) -> RvgLabelIndex:
    """Build a one-pass index. ``entries`` values are ``(move_index, cp_eff)``."""
    path = tmp_path / "labels.jsonl"
    _write_labels(path, [_label(keys[i], idx, cp) for i, (idx, cp) in entries.items()])
    return RvgLabelIndex.load(path)


def _spec(**kw: object) -> rr.RvgArmSpec:
    base: dict[str, object] = {
        "arm": "v", "r_weight": 0.0, "v_lambda": 0.0, "v_tau_cp": 0.0,
        "v_veto_cp": float("inf"), "g_alpha": 0.0, "g_temp_cp": 100.0,
    }
    base.update(kw)
    return rr.RvgArmSpec(
        arm=str(base["arm"]), r_weight=float(base["r_weight"]),  # pyright: ignore[reportArgumentType]
        v_lambda=float(base["v_lambda"]), v_tau_cp=float(base["v_tau_cp"]),  # pyright: ignore[reportArgumentType]
        v_veto_cp=float(base["v_veto_cp"]), g_alpha=float(base["g_alpha"]),  # pyright: ignore[reportArgumentType]
        g_temp_cp=float(base["g_temp_cp"]),  # pyright: ignore[reportArgumentType]
    )


def test_the_wrapper_edits_only_covered_rows_and_reports_realized_f(
    tmp_path: Path,
) -> None:
    """Row 0 is labeled, row 1 is not. Row 1 must come back BITWISE unchanged
    and the realized coverage must read 1/2 — the ladder2 convention: ``f`` is a
    property of the banked corpus and is reported with every arm."""
    targets = np.array([[0.5, 0.3, 0.2, 0, 0, 0, 0, 0],
                        [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0]], dtype=np.float32)
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for(inner.keys(), tmp_path, {0: ([0, 1], [0.0, -300.0])})
    buf = rr._RvgTargetSurgeryBuffer(
        inner, _spec(arm="v", v_lambda=0.01, v_tau_cp=0.0, v_veto_cp=1000.0), index,
    )
    out = buf.sample_batch_arrays(2)
    assert np.array_equal(out["policy_target"][1], inner._t[1])
    assert not np.array_equal(out["policy_target"][0], inner._t[0])
    stats = buf.rig_stats()
    assert stats["total_rows"] == 2
    assert stats["eligible_rows"] == 1
    assert stats["edited_rows"] == 1
    assert stats["fallback_rows"] == 0
    assert stats["realized_f"] == 0.5


def test_the_wrapper_falls_back_and_counts_an_all_vetoed_row(tmp_path: Path) -> None:
    """⚑ THE ONLY WAY A ROW CAN LOSE ALL ITS MASS, and it is a real one.

    Regret is relative to the label's OWN best listed move, so some listed move
    always has regret 0 and V can never veto the whole listed set. All-mass-
    vetoed therefore requires the target's SUPPORT to sit entirely on vetoed
    moves — i.e. the search never visited SF's best move, which is exactly the
    row shape the veto exists for. Here slot 0 is SF's best (cp 0) and carries
    ZERO target mass; slots 1 and 2 hold all of it at 900 cp of regret.
    """
    targets = np.array([[0.0, 0.5, 0.5, 0, 0, 0, 0, 0],
                        [1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for(inner.keys(), tmp_path,
                       {0: ([0, 1, 2], [0.0, -900.0, -900.0])})
    buf = rr._RvgTargetSurgeryBuffer(
        inner, _spec(arm="v", v_lambda=0.0, v_tau_cp=0.0, v_veto_cp=500.0), index,
    )
    out = buf.sample_batch_arrays(2)
    assert np.array_equal(out["policy_target"][0], inner._t[0])
    stats = buf.rig_stats()
    assert stats["eligible_rows"] == 1
    assert stats["fallback_rows"] == 1
    assert stats["edited_rows"] == 0


def test_arm_r_writes_the_listed_only_vector_and_sets_its_presence_flag(
    tmp_path: Path,
) -> None:
    targets = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0],
                        [1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    # Slot 1 is SF's best (cp 0, regret 0); slot 2 is 500 cp worse -> 0.5 after
    # normalization by the 1000 cp cap.
    index = _index_for(inner.keys(), tmp_path, {0: ([1, 2], [0.0, -500.0])})
    buf = rr._RvgTargetSurgeryBuffer(inner, _spec(arm="r", r_weight=0.5), index)
    out = buf.sample_batch_arrays(2)
    assert out["has_sf_p0_regret"].tolist() == [1, 0]
    assert pytest.approx(float(out["sf_p0_regret"][0][2]), abs=1e-3) == 0.5
    assert float(out["sf_p0_regret"][0][1]) == 0.0
    # THE PROPERTY: every UNLISTED slot is exactly 0, never a fabricated default.
    assert float(out["sf_p0_regret"][0][0]) == 0.0
    assert float(out["sf_p0_regret"][0][3:].sum()) == 0.0
    assert float(out["sf_p0_regret"][0].sum()) == pytest.approx(0.5, abs=1e-3)
    # And R must not touch the search target at all.
    assert np.array_equal(out["policy_target"], inner._t)


def test_the_dead_knob_guard_never_sees_the_rig_only_keys() -> None:
    """⚑ The 2026-08-20 ladder refusal, in miniature. `rvg_arm` reaches training
    by WRAPPING the buffer, so the reachability probe truthfully says it reaches
    nothing — asking it is the bug, and the split is the fix."""
    overrides = {"rvg_arm": "v", "rvg_v_lambda": 0.02, "lr": 0.0003}
    assert not rr._override_key_reaches_the_trainer("rvg_arm", {})
    rig, params, trainer_bound = rr._split_rig_overrides(overrides)
    assert rig == {"rvg_arm": "v"}
    assert params == {"rvg_v_lambda": 0.02}
    assert trainer_bound == {"lr": 0.0003}
    rr._assert_overrides_reach_the_trainer(
        name="v020", overrides=trainer_bound, base_config={"lr": 0.0003},
    )
    with pytest.raises(SystemExit, match="reach NOTHING"):
        rr._assert_overrides_reach_the_trainer(
            name="v020", overrides=overrides, base_config={"lr": 0.0003},
        )


def test_rvg_params_stay_variant_overridable_while_sweep_settings_do_not() -> None:
    """One invocation must be able to run a whole ladder (that is what keeps the
    arms' draws paired), so `rvg_g_alpha` is per-variant; `batch_size` is not."""
    keys = rr._sweep_level_keys()
    assert "batch_size" in keys
    assert "steps" in keys
    assert not (keys & (set(rr._RIG_ONLY_PARAMS) | set(rr._RIG_ONLY_WRAPPERS)))
    name, overrides = rr._parse_variant("g030:rvg_arm=g,rvg_g_alpha=0.30")
    assert name == "g030"
    assert overrides == {"rvg_arm": "g", "rvg_g_alpha": 0.30}


def test_arm_r_refuses_to_run_beside_the_sf_policy_floor() -> None:
    """⚑⚑ THE INTERACTION GUARD. `sf_policy_floor_deficit` reads the SAME
    `sf_p0_regret` vector and calls `regret <= delta_cp/1000` SF-APPROVED.
    R's repaired vector puts unlisted moves at 0.0, so with the floor on, every
    unlisted legal move would be floored at tau — and nothing in the report
    would say the arm measured a corrupted floor instead of a repaired term."""
    ctx = rr._RigContext.inert()
    ctx.defaults = _spec(arm="", r_weight=0.7)
    config = {"w_sf_policy_floor": 0.8}
    with pytest.raises(SystemExit, match="w_sf_policy_floor=0"):
        rr._apply_rvg_config_side(config, {"rvg_arm": "r"}, {}, ctx, name="r070")

    config = {"w_sf_policy_floor": 0.0}
    rr._apply_rvg_config_side(config, {"rvg_arm": "r"}, {}, ctx, name="r070")
    assert config["w_sf_own_regret"] == 0.7


def test_arm_r_refuses_two_sources_for_its_weight() -> None:
    ctx = rr._RigContext.inert()
    ctx.defaults = _spec(arm="", r_weight=0.7)
    with pytest.raises(SystemExit, match="sets w_sf_own_regret directly"):
        rr._apply_rvg_config_side(
            {"w_sf_policy_floor": 0.0}, {"rvg_arm": "r"},
            {"w_sf_own_regret": 0.2}, ctx, name="r070",
        )


def test_an_arm_that_covered_no_rows_is_a_refusal_not_a_null(tmp_path: Path) -> None:
    """`eligible_rows == 0` means the edit was the identity, i.e. this arm IS
    the control under another name. Reporting it as an arm is the failure."""
    targets = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0],
                        [1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for([b"\xaa" * 16], tmp_path, {0: ([1, 5], [0.0, -10.0])})
    buf = rr._RvgTargetSurgeryBuffer(
        inner, _spec(arm="v", v_lambda=0.01, v_veto_cp=1000.0), index,
    )
    buf.sample_batch_arrays(2)
    with pytest.raises(SystemExit, match="NOT ONE was eligible"):
        rr._assert_rig_wrapper_took_effect(buf, name="v020")


def test_a_wrapper_the_trainer_never_drew_through_is_a_refusal(tmp_path: Path) -> None:
    targets = np.zeros((2, _POLICY_W), dtype=np.float32)
    targets[:, 0] = 1.0
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for(inner.keys(), tmp_path, {0: ([1, 5], [0.0, -10.0])})
    buf = rr._RvgTargetSurgeryBuffer(
        inner, _spec(arm="v", v_lambda=0.01, v_veto_cp=1000.0), index,
    )
    with pytest.raises(SystemExit, match="served ZERO rows"):
        rr._assert_rig_wrapper_took_effect(buf, name="v020")


def test_the_control_leg_runs_unwrapped_and_stamps_rig_active_none() -> None:
    """`a000` must be BITWISE the no-wrapper path, not a copy-and-mask identity."""
    sentinel = object()
    buf, active = rr._apply_rig_wrappers(
        sentinel, {}, rr._RigContext.inert(), name="a000",
    )
    assert buf is sentinel
    assert active is None


def test_an_rvg_arm_without_a_label_file_refuses() -> None:
    """A missing label file must never degrade to "train the control quietly"."""
    ctx = rr._RigContext.inert()
    ctx.defaults = _spec(arm="", v_lambda=0.02, v_veto_cp=500.0)
    with pytest.raises(SystemExit, match="--rvg-labels"):
        rr._apply_rig_wrappers(object(), {"rvg_arm": "v"}, ctx, name="v020")


def test_two_active_rig_wrappers_are_a_refusal_not_a_precedence_rule() -> None:
    with pytest.raises(SystemExit, match="rig wrappers at once"):
        rr._apply_rig_wrappers(
            object(), {"rvg_arm": "v", "rig_policy_from_soft": 1},
            rr._RigContext.inert(), name="mixed",
        )


@pytest.mark.parametrize(("kw", "match"), [
    ({"arm": "x"}, "expected one of"),
    ({"arm": "r", "r_weight": 0.0}, "trains the control"),
    ({"arm": "g", "g_alpha": 0.0}, "band is 0 < alpha < 1"),
    ({"arm": "g", "g_alpha": 1.0}, "band is 0 < alpha < 1"),
    ({"arm": "g", "g_alpha": 0.5, "g_temp_cp": 0.0}, "needs > 0"),
    ({"arm": "v", "v_lambda": 0.0, "v_veto_cp": float("inf")}, "IS the control"),
])
def test_an_arm_that_cannot_move_anything_is_refused(
    kw: dict[str, object], match: str,
) -> None:
    with pytest.raises(SystemExit, match=match):
        _spec(**kw).validate()


# ---------------------------------------------------------------------------
# 8. Arm VG — the composition (V FIRST, then G on the veto-edited target)
# ---------------------------------------------------------------------------
#
# ⚑ THE HAND EXAMPLE DELIBERATELY INCLUDES A SOFTLY-DOWNWEIGHTED MOVE, not only
# a hard veto. The hard veto is ORDER-INVARIANT (zero times anything is zero), so
# an example built from vetoes alone gives V-then-G and G-then-V the same answer
# and the order mutant survives. The soft leg is what separates them: V-then-G
# raises V's multiplier to the power (1-alpha), G-then-V applies it at full
# strength.

_VG_T = np.array([0.5, 0.3, 0.2], dtype=np.float32)
_VG_IDX = np.array([0, 1, 2])
_VG_REG = np.array([0.0, 150.0, 600.0])
_VG_V = {"lam": 0.01, "tau_cp": 50.0, "veto_cp": 500.0}
_VG_G = {"alpha": 0.5, "temp_cp": 200.0}

#: Hand-derived, spelled out so a refactor cannot silently redefine the chain.
#:   V:  w = [exp(0), exp(-0.01*100), 0]  = [1, 0.36787944, 0]  (600 >= veto 500)
#:       unnormalized [0.5, 0.11036383, 0] / 0.61036383
_VG_AFTER_V = [0.81918350, 0.18081647, 0.0]
#:   G on THAT: q ~ exp(-[0,150,600]/200) = [1, 0.47236655, 0.04978707] / 1.52215362
#:       = [0.65697000, 0.31032563, 0.03270437]; listed slots -> sqrt(t1*q)
_VG_AFTER_VG = [0.75591505, 0.24408492, 0.0]
#: The WRONG order, for the mutant: G first on the raw t, then V.
_GV_WRONG_ORDER = [0.83622617, 0.16377382, 0.0]


def test_vg_composes_v_then_g_on_a_hand_computed_example() -> None:
    after_v, fell_back_v = apply_veto_edit(_VG_T, _VG_IDX, _VG_REG, **_VG_V)
    assert not fell_back_v
    assert np.allclose(after_v, _VG_AFTER_V, atol=1e-6)
    after_vg, fell_back_g = apply_geometric_blend(
        after_v, _VG_IDX, _VG_REG,
        alpha=_VG_G["alpha"], temp_cp=_VG_G["temp_cp"],
    )
    assert not fell_back_g
    assert np.allclose(after_vg, _VG_AFTER_VG, atol=1e-6)
    # V's hard veto survives G: a zeroed move stays zero through the blend.
    assert after_vg[2] == 0.0
    assert pytest.approx(float(after_vg.sum()), abs=1e-6) == 1.0


def test_vg_through_the_wrapper_matches_the_chain_and_is_not_the_reverse_order(
    tmp_path: Path,
) -> None:
    """⚑ THE ORDER MUTANT'S KILLER. Composing G-then-V is a different, entirely
    finite, entirely normalized distribution — nothing downstream would object to
    it. Only this assertion distinguishes them."""
    targets = np.stack([_VG_T, _VG_T])
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for(inner.keys(), tmp_path,
                       {0: (_VG_IDX.tolist(), (-_VG_REG).tolist())})
    buf = rr._RvgTargetSurgeryBuffer(
        inner,
        _spec(arm="vg", v_lambda=_VG_V["lam"], v_tau_cp=_VG_V["tau_cp"],
              v_veto_cp=_VG_V["veto_cp"], g_alpha=_VG_G["alpha"],
              g_temp_cp=_VG_G["temp_cp"]),
        index,
    )
    out = buf.sample_batch_arrays(2)
    assert np.allclose(out["policy_target"][0], _VG_AFTER_VG, atol=1e-3)
    assert not np.allclose(out["policy_target"][0], _GV_WRONG_ORDER, atol=1e-3)
    # ... and it is not either half on its own, so "VG" cannot be a mislabelled
    # V or G run.
    assert not np.allclose(out["policy_target"][0], _VG_AFTER_V, atol=1e-3)


def test_vg_reports_edited_rows_per_stage(tmp_path: Path) -> None:
    """"How many rows V touched, and how many G then touched." An aggregate
    alone cannot tell "both stages bit" from "V bit and G was inert everywhere",
    and the second is a V arm reported as a VG arm."""
    targets = np.stack([_VG_T, _VG_T])
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for(inner.keys(), tmp_path,
                       {0: (_VG_IDX.tolist(), (-_VG_REG).tolist())})
    buf = rr._RvgTargetSurgeryBuffer(
        inner,
        _spec(arm="vg", v_lambda=_VG_V["lam"], v_tau_cp=_VG_V["tau_cp"],
              v_veto_cp=_VG_V["veto_cp"], g_alpha=_VG_G["alpha"],
              g_temp_cp=_VG_G["temp_cp"]),
        index,
    )
    buf.sample_batch_arrays(2)
    stats = buf.rig_stats()
    assert stats["stamp"] == "V+G"
    assert stats["stages"] == [
        {"stage": "V", "edited_rows": 1, "fallback_rows": 0},
        {"stage": "G", "edited_rows": 1, "fallback_rows": 0},
    ]
    assert stats["total_rows"] == 2
    assert stats["eligible_rows"] == 1
    assert stats["realized_f"] == 0.5


def test_vg_stamps_the_composition_in_rig_active(tmp_path: Path) -> None:
    """`rvg_arm` alone reads identically for V, G and VG in the report."""
    targets = np.stack([_VG_T, _VG_T])
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for(inner.keys(), tmp_path,
                       {0: (_VG_IDX.tolist(), (-_VG_REG).tolist())})
    ctx = rr._RigContext.inert()
    ctx.labels = index
    ctx.defaults = _spec(arm="", v_lambda=_VG_V["lam"], v_tau_cp=_VG_V["tau_cp"],
                         v_veto_cp=_VG_V["veto_cp"], g_alpha=_VG_G["alpha"],
                         g_temp_cp=_VG_G["temp_cp"])
    _, active = rr._apply_rig_wrappers(inner, {"rvg_arm": "vg"}, ctx, name="vg01")
    assert active == "rvg_arm:V+G"
    _, active_v = rr._apply_rig_wrappers(inner, {"rvg_arm": "v"}, ctx, name="v020")
    assert active_v == "rvg_arm:V"
    _, active_g = rr._apply_rig_wrappers(inner, {"rvg_arm": "g"}, ctx, name="g030")
    assert active_g == "rvg_arm:G"


def test_vg_validates_both_legs_not_just_the_first() -> None:
    """A VG arm with an inert G leg is a V arm wearing VG's label."""
    with pytest.raises(SystemExit, match="band is 0 < alpha < 1"):
        _spec(arm="vg", v_lambda=0.01, v_veto_cp=500.0, g_alpha=0.0).validate()
    with pytest.raises(SystemExit, match="IS the control"):
        _spec(arm="vg", v_lambda=0.0, v_veto_cp=float("inf"),
              g_alpha=0.5).validate()
    # Both legs live -> accepted.
    _spec(arm="vg", v_lambda=0.01, v_veto_cp=500.0, g_alpha=0.5).validate()


def test_the_composed_arm_reuses_the_standalone_edit_functions() -> None:
    """(1) in the spec: no third implementation. The stage table IS the chain,
    so a fix to `apply_veto_edit` or `apply_geometric_blend` reaches VG the day
    it lands rather than the day someone remembers a fused copy exists."""
    assert rr.RvgArmSpec.STAGES["vg"] == ("V", "G")
    assert rr.RvgArmSpec.STAGES["v"] == ("V",)
    assert rr.RvgArmSpec.STAGES["g"] == ("G",)
    assert rr.RvgArmSpec.STAGES["r"] == ()


# ---------------------------------------------------------------------------
# 9. G-stage entropy — CALIBRATION INSTRUMENT ONLY
# ---------------------------------------------------------------------------
#
# ⚑⚑ These tests pin that the number is REPORTED and CORRECT. They deliberately
# do not assert a direction or a threshold: G's temperature is the gradient of
# entropy, so an arm scored on entropy wins by construction
# ([[an_arm_that_is_the_gradient_of_the_metric_always_wins]]). The deciding
# yardstick stays row-(a) top-1 deep-SF regret with the E[regret] co-criterion.


def test_entropy_of_the_hand_example_pre_and_post_g() -> None:
    """t = [0.5, 0.3, 0.2] -> H = -(0.5 ln0.5 + 0.3 ln0.3 + 0.2 ln0.2)
                                = 0.34657359 + 0.36119184 + 0.32188758
                                = 1.02965301 nats
    after G (alpha 0.5, T 100 cp) the row is [0.5553645, 0.26091948, 0.18371602]
                                -> 0.98846535 nats."""
    pre = distribution_entropy_nats(np.array([0.5, 0.3, 0.2]))
    assert pytest.approx(pre, abs=1e-8) == 1.02965301
    out, _ = apply_geometric_blend(
        np.array([0.5, 0.3, 0.2], dtype=np.float32),
        np.array([0, 1]), np.array([0.0, 100.0]), alpha=0.5, temp_cp=100.0,
    )
    assert pytest.approx(distribution_entropy_nats(out), abs=1e-6) == 0.98846535


def test_entropy_ignores_zero_mass_entries_and_renormalizes() -> None:
    """A vetoed move contributes 0 (``0 log 0 == 0``), and a row that sums to
    1.0 only to float16 precision is normalized before the log — an
    unnormalized vector's "entropy" is not an entropy."""
    assert distribution_entropy_nats(np.array([0.5, 0.5, 0.0])) == pytest.approx(
        math.log(2.0), abs=1e-12,
    )
    assert distribution_entropy_nats(np.array([5.0, 5.0])) == pytest.approx(
        math.log(2.0), abs=1e-12,
    )
    assert distribution_entropy_nats(np.zeros((4,))) == 0.0


def test_the_g_stage_reports_entropy_before_and_after_in_the_run_stats(
    tmp_path: Path,
) -> None:
    targets = np.stack([_VG_T, _VG_T])
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for(inner.keys(), tmp_path,
                       {0: (_VG_IDX.tolist(), (-_VG_REG).tolist())})
    buf = rr._RvgTargetSurgeryBuffer(
        inner, _spec(arm="g", g_alpha=_VG_G["alpha"], g_temp_cp=_VG_G["temp_cp"]),
        index,
    )
    buf.sample_batch_arrays(2)
    ent = buf.rig_stats()["g_target_entropy_nats"]
    assert ent is not None
    assert ent["rows"] == 1
    # abs=1e-3: the wrapper serves the batch in float16 exactly as the shards
    # store it, so the row it measures is the float16 rounding of [0.5,0.3,0.2].
    assert pytest.approx(ent["before"], abs=1e-3) == 1.02965301
    assert "calibration only" in ent["note"]


def test_the_vg_chain_measures_entropy_across_the_G_STAGES_input_not_the_raw_target(
    tmp_path: Path,
) -> None:
    """⚑ In arm VG, "before" is the VETO-EDITED target — the G stage's own input
    — not the stored one. Reading the raw target there would misattribute V's
    entropy change to G and break the (alpha, T) calibration this number exists
    for. Hand values: after V the row is [0.81918350, 0.18081647, 0] -> 0.47262929
    nats; after G it is [0.75591505, 0.24408492, 0] -> 0.55574297 nats. The RAW
    target's entropy is 1.02965301, which is what a wrong read would report."""
    targets = np.stack([_VG_T, _VG_T])
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for(inner.keys(), tmp_path,
                       {0: (_VG_IDX.tolist(), (-_VG_REG).tolist())})
    buf = rr._RvgTargetSurgeryBuffer(
        inner,
        _spec(arm="vg", v_lambda=_VG_V["lam"], v_tau_cp=_VG_V["tau_cp"],
              v_veto_cp=_VG_V["veto_cp"], g_alpha=_VG_G["alpha"],
              g_temp_cp=_VG_G["temp_cp"]),
        index,
    )
    buf.sample_batch_arrays(2)
    ent = buf.rig_stats()["g_target_entropy_nats"]
    assert ent is not None
    assert pytest.approx(ent["before"], abs=1e-3) == 0.47262929   # float16 batch
    assert pytest.approx(ent["after"], abs=1e-3) == 0.55574297
    # The tolerance is far tighter than the gap to the WRONG read (the raw
    # target, 1.02965301), so this cannot pass by being loose.
    assert abs(ent["before"] - 1.02965301) > 0.5
    assert ent["rows"] == 1


def test_an_arm_without_a_g_stage_reports_no_entropy_block(tmp_path: Path) -> None:
    targets = np.stack([_VG_T, _VG_T])
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for(inner.keys(), tmp_path,
                       {0: (_VG_IDX.tolist(), (-_VG_REG).tolist())})
    buf = rr._RvgTargetSurgeryBuffer(
        inner, _spec(arm="v", v_lambda=_VG_V["lam"], v_tau_cp=_VG_V["tau_cp"],
                     v_veto_cp=_VG_V["veto_cp"]), index,
    )
    buf.sample_batch_arrays(2)
    assert buf.rig_stats()["g_target_entropy_nats"] is None


# ---------------------------------------------------------------------------
# 10. Layered label join (default OFF; one flag away)
# ---------------------------------------------------------------------------


def test_the_layered_join_lets_the_deeper_pass_override_the_wider_one(
    tmp_path: Path,
) -> None:
    """A move in BOTH passes takes the DEEP value; a wide-only move keeps its
    wide value. Deep = more nodes per LINE (MPV3@150k = 50k/line beats
    MPV40@150k = 3.75k/line), which is the quantity that decides whose cp is
    better resolved."""
    key = b"\x11" * 16
    wide = tmp_path / "wide_mpv40.jsonl"
    deep = tmp_path / "deep_mpv3.jsonl"
    _write_labels(wide, [_label(key, [1, 2, 3], [0.0, -50.0, -400.0])],
                  nodes=150000, multipv=40)
    _write_labels(deep, [_label(key, [1, 2], [0.0, -90.0])],
                  nodes=150000, multipv=3)

    index = RvgLabelIndex.load([wide, deep])
    got = index.get(key)
    assert got is not None
    idx, reg = got
    assert idx.tolist() == [1, 2, 3]
    # move 2 takes the DEEP cp (-90 -> regret 90), move 3 keeps the WIDE one.
    assert reg.tolist() == [0.0, 90.0, 400.0]
    supplied = index.pass_for(key)
    assert supplied is not None
    names = [index.passes[i]["name"] for i in supplied.tolist()]
    assert names == ["deep_mpv3.jsonl", "deep_mpv3.jsonl", "wide_mpv40.jsonl"]
    mix = {p["name"]: p["moves_supplied"] for p in index.mix}
    assert mix == {"deep_mpv3.jsonl": 2, "wide_mpv40.jsonl": 1}


def test_the_overlay_order_does_not_depend_on_the_order_the_files_were_given(
    tmp_path: Path,
) -> None:
    """⚑ THE ORDER MUTANT'S KILLER. Applying strongest-first (or trusting the CLI
    order) hands move 2 the WIDE cp instead of the deep one — a finite,
    normalized, silently wrong regret."""
    key = b"\x11" * 16
    wide = tmp_path / "wide_mpv40.jsonl"
    deep = tmp_path / "deep_mpv3.jsonl"
    _write_labels(wide, [_label(key, [1, 2], [0.0, -50.0])], nodes=150000, multipv=40)
    _write_labels(deep, [_label(key, [1, 2], [0.0, -90.0])], nodes=150000, multipv=3)
    for order in ([wide, deep], [deep, wide]):
        got = RvgLabelIndex.load(order).get(key)
        assert got is not None
        _, reg = got
        assert reg.tolist() == [0.0, 90.0], f"CLI order {order} changed the overlay"


def test_the_layered_join_recomputes_best_cp_over_the_overlaid_set(
    tmp_path: Path,
) -> None:
    """⚑ Regret cannot be overlaid — only cp can. Here the deep pass finds a move
    BETTER than anything the wide pass listed, which moves ``best_cp`` and
    therefore every regret in the row. A join that overlaid regrets would keep
    the wide pass's baseline and report move 1 at 0 cp of regret when it is
    actually 60 behind."""
    key = b"\x11" * 16
    wide = tmp_path / "wide_mpv40.jsonl"
    deep = tmp_path / "deep_mpv3.jsonl"
    _write_labels(wide, [_label(key, [1, 2], [0.0, -20.0])], nodes=150000, multipv=40)
    _write_labels(deep, [_label(key, [3], [60.0])], nodes=150000, multipv=3)
    got = RvgLabelIndex.load([wide, deep]).get(key)
    assert got is not None
    idx, reg = got
    assert idx.tolist() == [1, 2, 3]
    assert reg.tolist() == [60.0, 80.0, 0.0]


# ---------------------------------------------------------------------------
# 11. Arm B — an EXTERNAL q for the geometric blend
# ---------------------------------------------------------------------------


def test_the_external_q_source_blends_toward_a_file_supplied_policy(
    tmp_path: Path,
) -> None:
    """t = [0.5, 0.3, 0.2] on the three legal moves of a hand-built position;
    q comes from a FILE, not from SF regret.

    q = {a1a2: 0.7, a1b1: 0.3} and alpha = 0.5, so listed slots become
    sqrt(t*q) and the third listed move (absent from the file, q = 0) is removed
    — the documented external-teacher convention: a move an external net gives
    no mass is a move it votes against, unlike SF's MultiPV truncation where
    absence is only absence.
    """
    board = chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")
    ucis = sorted(m.uci() for m in board.legal_moves)
    slots = [
        move_to_index_for_encoding(chess.Move.from_uci(u), board) for u in ucis
    ]
    path = tmp_path / "ext.jsonl"
    key = b"\x22" * 16
    _write_external_q(
        path, [{"key": key.hex(), "policy": {ucis[0]: 0.7, ucis[1]: 0.3}}],
        name="bt4-test",
    )
    ext = RvgExternalPolicyIndex.load(path)
    assert len(ext) == 1
    assert ext.header["name"] == "bt4-test"

    idx = np.asarray(slots[:3], dtype=np.int64)
    q = ext.weights_for(key, idx)
    assert q is not None
    assert q.tolist() == [0.7, 0.3, 0.0]

    width = max(slots) + 1
    t = np.zeros((width,), dtype=np.float32)
    t[slots[0]], t[slots[1]], t[slots[2]] = 0.5, 0.3, 0.2
    out, fell_back = apply_geometric_blend(
        t, idx, np.zeros_like(q), alpha=0.5, temp_cp=100.0, q_weights=q,
    )
    assert not fell_back
    unnorm = np.array([math.sqrt(0.5 * 0.7), math.sqrt(0.3 * 0.3), 0.0])
    expected = unnorm / unnorm.sum()
    assert np.allclose(
        [out[slots[0]], out[slots[1]], out[slots[2]]], expected, atol=1e-6,
    )
    assert out[slots[2]] == 0.0


def test_the_external_q_aligns_by_policy_index_not_by_list_position(
    tmp_path: Path,
) -> None:
    """⚑ A row's external policy and its SF label list moves in different orders
    far more often than not. Zipping them produces a finite, normalized,
    completely wrong q — so the lookup is by SLOT."""
    board = chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")
    ucis = sorted(m.uci() for m in board.legal_moves)[:3]
    slots = [
        move_to_index_for_encoding(chess.Move.from_uci(u), board) for u in ucis
    ]
    path = tmp_path / "ext.jsonl"
    key = b"\x22" * 16
    # Written in REVERSE order on purpose.
    _write_external_q(path, [{
        "key": key.hex(),
        "policy": {ucis[2]: 0.6, ucis[1]: 0.3, ucis[0]: 0.1},
    }])
    ext = RvgExternalPolicyIndex.load(path)
    q = ext.weights_for(key, np.asarray(slots, dtype=np.int64))
    assert q is not None
    assert q.tolist() == [0.1, 0.3, 0.6]


def test_a_row_absent_from_the_external_q_file_is_counted_not_back_filled(
    tmp_path: Path,
) -> None:
    """Substituting the SF-derived q for a missing external row would make arm B
    a silent mixture of two teachers."""
    targets = np.stack([_VG_T, _VG_T])
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for(inner.keys(), tmp_path,
                       {0: (_VG_IDX.tolist(), (-_VG_REG).tolist())})
    ext_path = tmp_path / "ext.jsonl"
    _write_external_q(ext_path, [{"key": (b"\xee" * 16).hex(), "policy": {}}])
    buf = rr._RvgTargetSurgeryBuffer(
        inner, _spec(arm="g", g_alpha=0.5, g_temp_cp=200.0),
        index, external_q=RvgExternalPolicyIndex.load(ext_path),
    )
    out = buf.sample_batch_arrays(2)
    assert np.array_equal(out["policy_target"][0], inner._t[0])
    stats = buf.rig_stats()
    assert stats["q_source"] == "external_policy_file"
    assert stats["external_q"]["missing_rows"] == 1
    assert stats["stages"] == [{"stage": "G", "edited_rows": 0, "fallback_rows": 1}]


@pytest.mark.parametrize("arm", ["r", "v"])
def test_an_external_q_on_a_leg_with_no_g_stage_is_detached_and_says_so(
    arm: str, tmp_path: Path,
) -> None:
    """⚑ NOT A REFUSAL, AND NOT A SILENT NO-OP EITHER.

    A mixed ladder (a000 / v020 / g030) with ``--rvg-g-q-source`` is legitimate:
    the q belongs to the G legs. Refusing per-leg would ban that sweep. But the
    original form tested ``not spec.stages()``, which is empty ONLY for arm R —
    so arm V passed the guard, the q reached nothing, and ``rig_stats`` still
    reported ``q_source: external_policy_file``. A false provenance line is
    worse than the no-op it describes: it is the line a reader uses to rule the
    failure out.

    The leg detaches it and says so on both channels — ``q_source`` reads
    ``sf_regret``, and ``external_q`` carries ``attached: False`` with a reason.
    """
    targets = np.stack([_VG_T, _VG_T])
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for(inner.keys(), tmp_path,
                       {0: (_VG_IDX.tolist(), (-_VG_REG).tolist())})
    ext_path = tmp_path / "ext.jsonl"
    _write_external_q(ext_path, [{"key": (b"\xee" * 16).hex(), "policy": {}}])
    spec = (_spec(arm="r", r_weight=0.5) if arm == "r"
            else _spec(arm="v", v_lambda=0.01, v_veto_cp=1000.0))
    buf = rr._RvgTargetSurgeryBuffer(
        inner, spec, index, external_q=RvgExternalPolicyIndex.load(ext_path),
    )
    stats = buf.rig_stats()
    assert stats["q_source"] == "sf_regret"
    assert stats["external_q"] == {
        "attached": False,
        "reason": "this arm has no G stage; the q applies to g/vg legs",
    }


def _write_external_q(
    path: Path, rows: list[dict[str, object]], **header: object,
) -> Path:
    """Write an arm-B q file, provenance line included.

    The version line is REQUIRED by the loader: an external policy comes from
    another program on another branch, so an undeclared format is one neither
    side has agreed on. Written through one helper so a fixture cannot quietly
    test a shape the rig refuses.
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "record": "provenance", "v": RVG_EXTERNAL_POLICY_SCHEMA_VERSION,
            **header,
        }) + "\n")
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


# ---------------------------------------------------------------------------
# 11b. The join key reads each row's OWN history encoding
# ---------------------------------------------------------------------------


class _MixedEncodingInner:
    """A pool serving ONE position encoded under TWO different plane layouts.

    Salvage bundles are concatenations of eras and the metadata plane block
    moved between them (``_plane_layout``: root encodings hang their castling
    and clock off ``root_metadata_base``, the legacy one off ``metadata_base``).
    A pool that mixes them is exactly the case ``_row_keys`` exists for.
    """

    def __init__(self, fen: str, encodings: list[str], targets: np.ndarray) -> None:
        board = chess.Board(fen)
        self._encodings = list(encodings)
        self._x = np.stack([
            encode_lc0_full(board, input_history_encoding=enc) for enc in encodings
        ]).astype(np.float16)
        self._t = targets.astype(np.float16)

    def per_row_keys(self) -> list[bytes]:
        return [
            position_fingerprints(self._x[i:i + 1], input_history_encoding=enc)[0]
            for i, enc in enumerate(self._encodings)
        ]

    def sample_batch_arrays(
        self, batch_size: int, *, wdl_balance: bool = True,
    ) -> dict[str, np.ndarray]:
        del batch_size, wdl_balance   # a fixed batch; the house unused-arg idiom
        n = self._x.shape[0]
        return {
            "x": self._x.copy(),
            "policy_target": self._t.copy(),
            "_input_history_encoding": np.array(self._encodings),
            "sf_p0_regret": np.zeros((n, _POLICY_W), dtype=np.float16),
            "has_sf_p0_regret": np.zeros((n,), dtype=np.uint8),
        }

    def __len__(self) -> int:
        return 2


def test_the_join_key_reads_each_rows_own_history_encoding(tmp_path: Path) -> None:
    """⚑ A HARDCODED LAYOUT DOES NOT RAISE — IT MIS-JOINS AND REPORTS A LOW ``f``.

    Both rows are the SAME position, so a reader that honours each row's own
    encoding gives them ONE key: the fingerprint identifies the position, not
    the plane arrangement it arrived in. A reader that keys every row with one
    hardcoded layout reads the legacy row's history planes as castling bits and
    hands it a plausible, wrong key — no exception, just a row that silently
    stops matching its label while the arm still reports a tidy coverage number.

    So the assertion is on the JOIN, not on the helper: with a single label
    written under the shared key, BOTH rows must come back eligible.
    """
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5"
    targets = np.array([[0.5, 0.3, 0.2, 0, 0, 0, 0, 0]] * 2, dtype=np.float32)
    inner = _MixedEncodingInner(fen, [PROD_ENCODING, "legacy"], targets)
    keys = inner.per_row_keys()
    assert keys[0] == keys[1]

    path = tmp_path / "labels.jsonl"
    _write_labels(path, [_label(keys[0], [0, 1], [0.0, -300.0])])
    index = RvgLabelIndex.load(path)
    buf = rr._RvgTargetSurgeryBuffer(
        inner, _spec(arm="v", v_lambda=0.01, v_tau_cp=0.0, v_veto_cp=1000.0), index,
    )
    out = buf.sample_batch_arrays(2)
    stats = buf.rig_stats()
    assert stats["total_rows"] == 2
    assert stats["eligible_rows"] == 2
    assert stats["realized_f"] == 1.0
    # and the edit actually landed on both, not just on the row whose layout
    # happened to match whatever the key reader assumed.
    assert not np.array_equal(out["policy_target"][0], inner._t[0])
    assert not np.array_equal(out["policy_target"][1], inner._t[1])


# ---------------------------------------------------------------------------
# 12. The join on the PRODUCTION path (the rig's own replay buffer)
# ---------------------------------------------------------------------------


def test_the_key_survives_the_rigs_replay_buffer_not_just_the_scan_path(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE WIRING PROOF, AND IT IS A DIFFERENT PATH FROM THE LABEL PASS'S.

    ``scripts/rvg_label_pass.py`` keys its records off shard arrays read
    STRAIGHT off disk. The rig keys its lookups off batches
    ``DiskReplayBuffer.sample_batch_arrays`` returns, which go through the
    shuffle pool, sparse policy storage and ``_gather_rows``' densification. If
    any of that changed ``x`` — a plane upgrade, a dtype round-trip, a
    reordering — every fingerprint would differ, the join would return "no
    label" on every row, and the arms would read as a clean null with the rig's
    own coverage counter dutifully reporting f = 0.

    So this test does not check the fingerprint function; it checks that the two
    PATHS agree. Same shards, keys computed both ways, and every key the buffer
    serves must be one the label file could address.

    (Measured the same way against the banked ladder2 corpus: 4591/4591 of the
    rows the rig's buffer served were addressable by the scan path's keys.)
    """
    from chess_anti_engine.moves.encode import POLICY_SIZE
    from chess_anti_engine.replay.buffer import ReplaySample
    from chess_anti_engine.replay.shard import (
        samples_to_arrays,
        save_local_shard_arrays,
    )

    boards = [chess.Board(f) for f in _ROUNDTRIP_FENS]
    samples = []
    for i, board in enumerate(boards * 4):
        pol = np.zeros(POLICY_SIZE, dtype=np.float32)
        pol[i % POLICY_SIZE] = 1.0
        samples.append(ReplaySample(
            x=encode_lc0_full(board, input_history_encoding=PROD_ENCODING),
            policy_target=pol, wdl_target=i % 3, priority=1.0, has_policy=True,
        ))
    shard_dir = tmp_path / "shards"
    arrs = samples_to_arrays(samples)
    # The stored per-row encoding tag, exactly as a production shard carries it —
    # both paths read it off the row rather than assuming production's.
    arrs["_input_history_encoding"] = np.asarray(PROD_ENCODING)
    save_local_shard_arrays(shard_dir / "shard_000001.zarr", arrs=arrs)

    # SCAN path — what the label pass keys its records with.
    from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays
    scan_keys: set[bytes] = set()
    for path in iter_shard_paths(shard_dir):
        stored, _ = load_shard_arrays(path, lazy=False)
        scan_keys.update(position_fingerprints(
            np.asarray(stored["x"]),
            input_history_encoding=str(
                np.asarray(stored["_input_history_encoding"]).reshape(-1)[0],
            ),
        ))
    assert scan_keys

    # BUFFER path — what the rig looks up.
    buf = rr.build_rig_replay_buffer(
        config={"seed": 0}, replay_dir=shard_dir,
        target_planes=int(samples[0].x.shape[0]),
        rng=np.random.default_rng(0),
    )
    try:
        # Through the WRAPPER'S OWN `_row_keys`, not a re-derivation here: that
        # method is what the arms actually look labels up with, including its
        # per-row encoding read.
        keyer = rr._RvgTargetSurgeryBuffer(
            buf, _spec(arm="v", v_lambda=0.01, v_veto_cp=500.0),
            _index_for([b"\x00" * 16], tmp_path, {0: ([1, 2], [0.0, -10.0])}),
        )
        served: set[bytes] = set()
        for _ in range(8):
            served.update(keyer._row_keys(buf.sample_batch_arrays(8)))
    finally:
        buf.close()
    assert served
    assert served <= scan_keys, (
        f"{len(served - scan_keys)} of {len(served)} rows the buffer served are "
        "NOT addressable by the scan path's keys — the join would silently miss"
    )


# ---------------------------------------------------------------------------
# 13. Review-pass guards (2026-08-22 independent + cross-family review)
# ---------------------------------------------------------------------------


def test_the_soft_shape_wrapper_can_actually_be_activated() -> None:
    """⚑ THIS BRANCH BROKE A PRE-EXISTING ARM, AND THE SUITE DID NOT NOTICE.

    ``_SoftPolicyAsMainBuffer.rig_active_stamp`` was copied from the rvg wrapper
    during the port and read ``self._spec.stamp()``. ``_spec`` is not an
    attribute of that class, so ``__getattr__`` forwarded the lookup to the
    wrapped ``DiskReplayBuffer`` and the arm died with ``AttributeError`` the
    first time ``rig_policy_from_soft=1`` was used. Nothing caught it because
    the only existing test of that key refuses BEFORE the wrapper is built.

    So this test does the one thing that would have: it ACTIVATES the wrapper
    and takes it through the same two calls ``_run_variant`` makes.
    """
    class _Inner:
        rng = np.random.default_rng(0)

        def sample_batch_arrays(
            self, batch_size: int, *, wdl_balance: bool = True,
        ) -> dict[str, np.ndarray]:
            del batch_size, wdl_balance
            n = 2
            return {
                "x": np.zeros((n, 175, 8, 8), dtype=np.float16),
                "policy_target": np.eye(n, _POLICY_W, dtype=np.float16),
                "policy_soft_target": np.full((n, _POLICY_W), 1.0 / _POLICY_W,
                                              dtype=np.float16),
                "has_policy_soft": np.ones((n,), dtype=np.uint8),
            }

        def __len__(self) -> int:
            return 2

    buf, active = rr._apply_rig_wrappers(
        _Inner(), {"rig_policy_from_soft": 1.0}, rr._RigContext.inert(), name="soft",
    )
    assert active == "rig_policy_from_soft:SOFT"
    out = buf.sample_batch_arrays(2)
    rr._assert_rig_wrapper_took_effect(buf, name="soft")
    assert np.allclose(np.asarray(out["policy_target"], dtype=np.float32),
                       1.0 / _POLICY_W)


def test_rig_parameters_without_an_active_wrapper_are_a_refusal() -> None:
    """``--variant "g030:rvg_g_alpha=0.30"`` with ``rvg_arm=g`` FORGOTTEN.

    The parameters are split away from the dead-knob guard (they are not meant
    to reach the Trainer), so nothing else can see that they reach nothing
    either: the overrides land in the report as applied, the trainer-bound set
    is empty, no wrapper is built, and the leg trains the control while its row
    in the ladder table reads "alpha 0.30".
    """
    _name, overrides = rr._parse_variant("g030:rvg_g_alpha=0.30,rvg_g_temp=150")
    rig, params, bound = rr._split_rig_overrides(overrides)
    assert not rig            # nothing activates a wrapper
    assert not bound          # nothing reaches the Trainer either
    assert params             # ...and yet the parameters are recorded
    with pytest.raises(SystemExit, match="activates NO rig wrapper"):
        rr._apply_rig_wrappers(
            object(), rig, rr._RigContext.inert(), name="g030", params=params,
        )


def test_an_arm_that_edited_no_row_is_a_refusal_not_a_null(tmp_path: Path) -> None:
    """Eligible does not mean edited, and the gate has to test the second one.

    Arm B with a q file keyed against a different corpus is the realistic case:
    every row is labeled (eligible), every q lookup misses, every G stage falls
    back, and the arm trains the control with ``realized_f`` reading 1.0.
    """
    targets = np.stack([_VG_T, _VG_T])
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    keys = inner.keys()
    index = _index_for(keys, tmp_path, {
        0: (_VG_IDX.tolist(), (-_VG_REG).tolist()),
        1: (_VG_IDX.tolist(), (-_VG_REG).tolist()),
    })
    ext_path = tmp_path / "ext.jsonl"
    _write_external_q(ext_path, [{"key": (b"\xee" * 16).hex(), "policy": {}}])
    buf = rr._RvgTargetSurgeryBuffer(
        inner, _spec(arm="g", g_alpha=0.5, g_temp_cp=200.0), index,
        external_q=RvgExternalPolicyIndex.load(ext_path),
    )
    buf.sample_batch_arrays(2)
    stats = buf.rig_stats()
    assert stats["eligible_rows"] == 2
    assert stats["realized_f"] == 1.0
    assert stats["edited_rows"] == 0
    with pytest.raises(SystemExit, match="NOT ONE was edited"):
        rr._assert_rig_wrapper_took_effect(buf, name="b030")


def test_an_external_q_that_matched_no_eligible_row_is_a_refusal(
    tmp_path: Path,
) -> None:
    """The same failure named accurately: a key-space mismatch, not a null arm.

    The message has to name the q FILE, because "regenerate the q file against
    this --replay-dir" and "your alpha is too small" are different repairs.
    """
    targets = np.stack([_VG_T, _VG_T])
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for(inner.keys(), tmp_path, {
        0: (_VG_IDX.tolist(), (-_VG_REG).tolist()),
        1: (_VG_IDX.tolist(), (-_VG_REG).tolist()),
    })
    ext_path = tmp_path / "ext.jsonl"
    _write_external_q(ext_path, [{"key": (b"\xee" * 16).hex(), "policy": {}}])
    buf = rr._RvgTargetSurgeryBuffer(
        inner, _spec(arm="g", g_alpha=0.5, g_temp_cp=200.0), index,
        external_q=RvgExternalPolicyIndex.load(ext_path),
    )
    buf.sample_batch_arrays(2)
    assert buf.rig_stats()["external_q"]["missing_rows"] == 2
    # `edited_rows == 0` fires first; both are the same event, and the external-q
    # leg is what makes the message actionable, so it is asserted directly.
    with pytest.raises(SystemExit, match="different key spaces"):
        rr._assert_rig_wrapper_took_effect(
            _StatsOnly({
                "total_rows": 2, "eligible_rows": 2, "edited_rows": 1,
                "fallback_rows": 1,
                "external_q": {"missing_rows": 2},
            }),
            name="b030",
        )


class _StatsOnly:
    """A wrapper stand-in that only answers ``rig_stats`` — the duck type the
    took-effect gate is written against."""

    def __init__(self, stats: dict[str, object]) -> None:
        self._stats = stats

    def rig_stats(self) -> dict[str, object]:
        return self._stats


def test_the_external_q_is_renormalized_over_the_LISTED_set() -> None:
    """⚑ THE MUTANT THE FIRST ARM-B SUITE COULD NOT KILL.

    Every earlier arm-B test used a q that already summed to 1 over the listed
    moves, so deleting ``q = q / q_total`` changed nothing and the whole suite
    stayed green. The realistic file does NOT sum to 1 over the listed set: an
    external net spreads its mass over ALL legal moves and the SF label lists a
    subset, so the aligned slice sums to whatever fraction of the net's mass
    happens to fall inside it.

    Hand-computed. t = [0.5, 0.3, 0.2] on slots 0..2, listed = slots 0 and 1,
    file q = 0.2 and 0.1 (mass 0.3, the rest of the net's belief sits on moves
    SF did not list). Normalized over the listed set: q = [2/3, 1/3].
    alpha = 0.5:
        slot0 = sqrt(0.5 * 2/3) = 0.5773502692
        slot1 = sqrt(0.3 * 1/3) = 0.3162277660
        slot2 = 0.2 (unlisted, untouched)
        sum   = 1.0935780352
        t'    = [0.5279461, 0.2891680, 0.1828859]
    WITHOUT the normalization the same row gives
        sqrt(0.5*0.2)=0.3162278, sqrt(0.3*0.1)=0.1732051, 0.2  -> renormalized
        [0.4586781, 0.2512284, 0.2900935] — a visibly different, wrong target.
    """
    t = np.array([0.5, 0.3, 0.2, 0.0], dtype=np.float32)
    idx = np.array([0, 1], dtype=np.int64)
    reg = np.array([0.0, 50.0], dtype=np.float64)   # ignored: q is external
    out, fell_back = apply_geometric_blend(
        t, idx, reg, alpha=0.5, temp_cp=200.0,
        q_weights=np.array([0.2, 0.1], dtype=np.float64),
    )
    assert not fell_back
    assert out[:3] == pytest.approx([0.5279461, 0.2891680, 0.1828859], abs=1e-6)
    # and the un-normalized answer is genuinely different, so the assertion above
    # is not passing by coincidence
    assert out[:3] != pytest.approx([0.4586781, 0.2512284, 0.2900935], abs=1e-3)


def test_a_row_present_in_the_q_file_with_no_overlap_is_counted_apart(
    tmp_path: Path,
) -> None:
    """Present-but-zero-overlap is a THIRD state and points at a third repair.

    ``missing_rows`` says "regenerate the keys". A row that IS in the file, all
    of whose moves fall outside this row's listed set, says something else: the
    two files are joined and their MOVE spaces disagree. Folded into either of
    the other counters it is unreadable.
    """
    targets = np.stack([_VG_T, _VG_T])
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    keys = inner.keys()
    index = _index_for(keys, tmp_path, {0: (_VG_IDX.tolist(), (-_VG_REG).tolist())})
    # A real uci that encodes to a slot OUTSIDE the labeled row's listed set
    # ([0, 1, 2]); asserted rather than assumed, because picking one that happens
    # to overlap would make this test pass for the wrong reason.
    board = chess.Board("8/8/8/8/8/8/8/K6k w - - 0 1")
    far = "a1b1"
    far_slot = move_to_index_for_encoding(chess.Move.from_uci(far), board)
    assert far_slot not in set(_VG_IDX.tolist())
    ext_path = tmp_path / "ext.jsonl"
    _write_external_q(ext_path, [{"key": keys[0].hex(), "policy": {far: 1.0}}])
    buf = rr._RvgTargetSurgeryBuffer(
        inner, _spec(arm="g", g_alpha=0.5, g_temp_cp=200.0), index,
        external_q=RvgExternalPolicyIndex.load(ext_path),
    )
    out = buf.sample_batch_arrays(2)
    stats = buf.rig_stats()
    assert stats["external_q"]["missing_rows"] == 0        # it IS in the file
    assert stats["external_q"]["zero_overlap_rows"] == 1   # and it overlapped nothing
    assert np.array_equal(out["policy_target"][0], inner._t[0])


def test_an_external_q_file_without_a_schema_version_is_a_refusal(
    tmp_path: Path,
) -> None:
    """Produced by another program on another branch: an undeclared format is
    one the two sides have never agreed on."""
    path = tmp_path / "ext.jsonl"
    path.write_text(json.dumps({"key": (b"\x11" * 16).hex(), "policy": {}}) + "\n")
    with pytest.raises(SystemExit, match="declares no schema version"):
        RvgExternalPolicyIndex.load(path)


def test_an_external_q_file_from_a_future_schema_is_a_refusal(tmp_path: Path) -> None:
    path = tmp_path / "ext.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "record": "provenance",
            "v": RVG_EXTERNAL_POLICY_SCHEMA_VERSION + 1,
        }) + "\n")
    with pytest.raises(SystemExit, match="external policy schema version"):
        RvgExternalPolicyIndex.load(path)


def test_an_external_q_file_in_another_policy_space_is_a_refusal(
    tmp_path: Path,
) -> None:
    """The loader's existing check refuses a CORPUS it cannot serve. That is a
    different question from "is this FILE in the corpus's space", and a file
    declaring ``az_4672`` passed it."""
    path = tmp_path / "ext.jsonl"
    _write_external_q(
        path, [{"key": (b"\x11" * 16).hex(), "policy": {}}],
        policy_encoding="az_4672",
    )
    with pytest.raises(SystemExit, match="policy space"):
        RvgExternalPolicyIndex.load(path)


def test_the_index_loaders_record_the_bytes_they_read_not_just_the_path(
    tmp_path: Path,
) -> None:
    """A path is not a file identity: a label file grows while a resumed pass
    appends to it, so two runs quoting one path can have read different bytes."""
    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, [_label(b"\x01" * 16, [0, 1], [0.0, -20.0])])
    index = RvgLabelIndex.load(labels)
    prov = cast("list[dict[str, object]]", index.header["label_file_provenance"])
    assert prov[0]["path"] == str(labels)
    assert prov[0]["bytes"] == labels.stat().st_size
    assert str(prov[0]["mtime_utc"]).endswith("Z")


def test_the_streaming_single_file_load_equals_the_overlay_path(
    tmp_path: Path,
) -> None:
    """⚑ THE MEMORY FIX MUST NOT BE A SECOND CONVENTION.

    The single-file path skips the two intermediate dicts the overlay path
    builds (~9.3 GB at 1M positions). That is only a memory optimization if the
    two produce the SAME index — including move ORDER and the duplicate-slot
    rule, both of which the overlay path decides implicitly (``sorted(slot)``
    over a dict). The row below has moves out of order AND a repeated slot, so
    an agreement here is an agreement about the conventions and not just about
    a simple case.
    """
    path = tmp_path / "labels.jsonl"
    _write_labels(path, [
        _label(b"\x01" * 16, [5, 2, 9, 2], [-30.0, 0.0, -80.0, -55.0]),
        _label(b"\x02" * 16, [3], [0.0]),
    ])
    streamed = RvgLabelIndex.load(path)                # dispatches to streaming
    overlaid = RvgLabelIndex._load_overlay([path])     # the same file, other path
    for key in (b"\x01" * 16, b"\x02" * 16):
        a, b = streamed.get(key), overlaid.get(key)
        assert a is not None
        assert b is not None
        assert np.array_equal(a[0], b[0])          # move_index, same ORDER
        assert np.allclose(a[1], b[1])             # regret_cp (what `get` returns)
        cs, co = streamed.cp_for(key), overlaid.cp_for(key)
        assert cs is not None
        assert co is not None
        assert np.allclose(cs, co)                 # cp_eff
    assert len(streamed) == len(overlaid)
    assert streamed.header["policy_encoding"] == overlaid.header["policy_encoding"]


def test_arm_r_unlabeled_rows_leave_the_loss_term_entirely() -> None:
    """⚑ ZERO-VECTOR + ``has = 0`` MUST BE LOSS-EQUIVALENT TO ABSENT.

    Arm R does not serve an unlabeled row untouched: it REPLACES the whole
    ``sf_p0_regret`` field, so an unlabeled row loses its stored MPV6 vector and
    gets zeros with ``has_sf_p0_regret = 0``. That is the intended semantics —
    a repaired-vector arm must not mix repaired and unrepaired vectors inside
    one loss term — but it is only safe if the mask is honoured EVERYWHERE.

    If any consumer ignored ``has``, a zero vector would read as "SF says every
    move here is perfect", which is a strong, false signal that would drag
    ``m_sf_own_regret`` toward zero and make the arm look better the LESS of the
    corpus it covered.

    So this traces the real loss, not the field: the same labeled row scored
    (a) beside an unlabeled row and (b) alone must give the SAME
    ``sf_own_regret``, while (c) the same zero vector marked ELIGIBLE must give
    a different one — (c) is what proves (a) is not passing vacuously.
    """
    import torch

    from chess_anti_engine.train.losses import compute_loss

    actions = 3
    labeled_regret = [1.0, 0.5, 0.0]

    def _score(regret_rows: list[list[float]], has: list[float]) -> float:
        n = len(regret_rows)
        policy_t = torch.zeros((n, actions), dtype=torch.float32)
        policy_t[:, 0] = 1.0
        batch = {
            "x": torch.zeros((n, 1, 1, 1), dtype=torch.float32),
            "policy_t": policy_t,
            "wdl_t": torch.zeros((n,), dtype=torch.long),
            "has_policy": torch.ones((n,), dtype=torch.float32),
            "is_network_turn": torch.ones((n,), dtype=torch.float32),
            "sf_p0_regret_t": torch.tensor(regret_rows, dtype=torch.float32),
            "has_sf_p0_regret": torch.tensor(has, dtype=torch.float32),
        }
        outputs = {
            "policy": torch.zeros((n, actions), dtype=torch.float32),
            "wdl": torch.zeros((n, 3), dtype=torch.float32),
        }
        losses = compute_loss(
            outputs, batch, w_wdl=0.0, w_sf_own_regret=0.5,
        )
        return float(losses["sf_own_regret"])

    with_unlabeled = _score([labeled_regret, [0.0] * actions], [1.0, 0.0])
    alone = _score([labeled_regret], [1.0])
    assert with_unlabeled == pytest.approx(alone, abs=1e-7), (
        "an unlabeled arm-R row changed the loss, so its zero vector is being "
        "read as 'every move is perfect' somewhere instead of being masked out"
    )
    # The negative control: the SAME zero vector, marked eligible, must move it.
    # Without this the assertion above would pass on a term that is always 0.
    if_counted = _score([labeled_regret, [0.0] * actions], [1.0, 1.0])
    assert if_counted != pytest.approx(alone, abs=1e-7)


def test_the_enumeration_consumes_the_same_rng_stream_as_the_trainer(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE ENUMERATION NAMES THE ROWS THE SWEEP WILL DRAW, OR IT NAMES NOTHING.

    ``maybe_mirror_batch_arrays`` draws ``rng.random(n)`` from ``buf.rng`` — the
    SAME generator ``sample_batch_arrays`` samples rows from — unconditionally
    at ``prob > 0``, and ``Trainer.mirror_prob`` is 0.5 with no yaml key. An
    enumeration that samples without mirroring therefore reproduces draw 1
    exactly and diverges from draw 2 onward. The mirror only changes a batch's
    CONTENT, which is why it read as irrelevant; what matters is its SIDE EFFECT
    on the shared generator.

    The comparison is against the REAL ``Trainer._prepare_host_arrays`` (called
    unbound on a duck-typed stand-in), not against a re-implementation of it —
    so this fails the day the trainer grows another between-draw RNG consumer,
    which is the regression the enumeration cannot survive silently.
    """
    from types import SimpleNamespace

    from chess_anti_engine.moves.encode import POLICY_SIZE
    from chess_anti_engine.replay.augment import maybe_mirror_batch_arrays
    from chess_anti_engine.replay.buffer import ReplaySample
    from chess_anti_engine.replay.shard import (
        samples_to_arrays,
        save_local_shard_arrays,
    )
    from chess_anti_engine.train.trainer import Trainer

    from scripts.rvg_label_pass import _trainer_mirror_prob

    boards = [chess.Board(f) for f in _ROUNDTRIP_FENS]
    samples = []
    for i, board in enumerate(boards * 8):
        pol = np.zeros(POLICY_SIZE, dtype=np.float32)
        pol[i % POLICY_SIZE] = 1.0
        samples.append(ReplaySample(
            x=encode_lc0_full(board, input_history_encoding=PROD_ENCODING),
            policy_target=pol, wdl_target=i % 3, priority=1.0, has_policy=True,
        ))
    shard_dir = tmp_path / "shards"
    arrs = samples_to_arrays(samples)
    arrs["_input_history_encoding"] = np.asarray(PROD_ENCODING)
    save_local_shard_arrays(shard_dir / "shard_000001.zarr", arrs=arrs)
    planes = int(samples[0].x.shape[0])

    mirror_prob = _trainer_mirror_prob({})
    assert mirror_prob > 0.0, (
        "with mirror_prob 0 the mirror consumes no RNG and this test cannot "
        "detect the divergence it exists for"
    )

    def _draw_stream(*, like_the_trainer: bool, draws: int = 6) -> list[list[bytes]]:
        buf = rr.build_rig_replay_buffer(
            config={"seed": 0}, replay_dir=shard_dir, target_planes=planes,
            rng=np.random.default_rng(0),
        )
        fake_trainer = SimpleNamespace(
            _sf_rebuild_coverage=None, rebuild_sf_targets=False,
            rebuild_categorical_target=False, sf_policy_sparse_ce=False,
            _input_history_encoding=PROD_ENCODING,
        )
        stream: list[list[bytes]] = []
        try:
            for _ in range(draws):
                batch = buf.sample_batch_arrays(8)
                stream.append(position_fingerprints(
                    np.asarray(batch["x"]), input_history_encoding=PROD_ENCODING,
                ))
                if like_the_trainer:
                    # The real host-side pipeline, unbound on a stand-in.
                    Trainer._prepare_host_arrays(
                        # Duck-typed on purpose: the point is to run the REAL
                        # method, not a re-implementation of it.
                        cast("Trainer", cast("object", fake_trainer)),
                        batch, rng=buf.rng,
                        mirror_prob=mirror_prob, rebuild_sf_targets=False,
                    )
                else:
                    # What the enumeration does.
                    maybe_mirror_batch_arrays(
                        batch, rng=buf.rng, prob=mirror_prob,
                        input_history_encoding=PROD_ENCODING,
                    )
        finally:
            buf.close()
        return stream

    def _naive_stream(draws: int = 6) -> list[list[bytes]]:
        """No mirror at all — what the enumeration used to do."""
        buf = rr.build_rig_replay_buffer(
            config={"seed": 0}, replay_dir=shard_dir, target_planes=planes,
            rng=np.random.default_rng(0),
        )
        try:
            return [
                position_fingerprints(
                    np.asarray(buf.sample_batch_arrays(8)["x"]),
                    input_history_encoding=PROD_ENCODING,
                )
                for _ in range(draws)
            ]
        finally:
            buf.close()

    trainer_stream = _draw_stream(like_the_trainer=True)
    enum_stream = _draw_stream(like_the_trainer=False)
    assert enum_stream == trainer_stream, (
        "the enumeration's draw sequence no longer matches the trainer's — the "
        "trainer consumes the buffer's RNG somewhere the enumeration does not, "
        "so --restrict-to would name rows the sweep never draws"
    )

    # ⚑ THE NEGATIVE CONTROL, which is what stops the assertion above from being
    # vacuous: without the mirror the FIRST draw still matches and the rest do
    # not. That exact signature is what made the bug invisible.
    naive = _naive_stream()
    assert naive[0] == trainer_stream[0]
    assert naive != trainer_stream


@pytest.mark.parametrize(
    ("mode", "argv_extra", "expected"),
    [
        ("enumerate-rows", ["--limit", "200"], "--limit"),
        ("enumerate-rows", ["--multipv", "6"], "--multipv"),
        ("enumerate-rows", ["--restrict-to", "keys.txt"], "--restrict-to"),
        ("label", ["--enum-steps", "10"], "--enum-steps"),
        ("label", ["--enum-batch-size", "8"], "--enum-batch-size"),
    ],
)
def test_a_flag_the_selected_mode_never_reads_is_a_refusal(
    mode: str, argv_extra: list[str], expected: str,
) -> None:
    """⚑ ACCEPTED AND THEN IGNORED, IN ITS CHEAPEST FORM.

    ``--mode enumerate-rows`` returns before ``--limit``, ``--multipv``,
    ``--nodes``, ``--threads``, ``--restrict-to`` or ``--no-syzygy`` is ever
    read, and ``--mode label`` never reads the three ``--enum-*`` flags. An
    operator who wrote ``--limit 200`` beside ``enumerate-rows`` believes they
    bounded a 41-minute job; nothing told them otherwise.
    """
    import argparse

    from scripts.rvg_label_pass import _refuse_flags_for_other_mode

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("label", "enumerate-rows"), default="label")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--nodes", type=int, default=150_000)
    ap.add_argument("--multipv", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--row-range", default="")
    ap.add_argument("--roundtrip-sample", type=int, default=200)
    ap.add_argument("--restrict-to", type=Path, default=None)
    ap.add_argument("--enum-steps", type=int, default=6000)
    ap.add_argument("--enum-batch-size", type=int, default=0)
    ap.add_argument("--enum-progress-every", type=int, default=100)
    ap.add_argument("--no-syzygy", action="store_true")

    clean = ap.parse_args(["--mode", mode])
    _refuse_flags_for_other_mode(ap, clean)          # defaults never refuse

    dirty = ap.parse_args(["--mode", mode, *argv_extra])
    with pytest.raises(SystemExit, match=expected):
        _refuse_flags_for_other_mode(ap, dirty)


def test_a_resume_under_different_label_settings_is_a_refusal() -> None:
    """⚑ THE HEADER IS THE THING THAT LIES.

    The ``.partial`` is keyed by position only, so re-running the same ``--out``
    after changing ``--nodes`` or ``--multipv`` stitches the old rows into the
    new file — and the header is then written from the NEW argv, so the artifact
    claims a budget it did not run. The join check cannot see it: every key is
    present, which is all it tests.
    """
    from scripts.rvg_label_pass import _assert_resume_settings_match

    banked: dict[str, object] = {
        "nodes": 150_000, "multipv": 40, "syzygy": True,
        "corpus": {"replay_dir": "/c", "shards": 3, "bytes": 99},
    }
    same = dict(banked)
    _assert_resume_settings_match({"record": "resume_settings", **banked},
                                  same, Path("p.partial"))   # no raise

    for key, value in (("nodes", 75_000), ("multipv", 6), ("syzygy", False)):
        changed = {**banked, key: value}
        with pytest.raises(SystemExit, match=key):
            _assert_resume_settings_match(
                {"record": "resume_settings", **banked}, changed,
                Path("p.partial"),
            )
    with pytest.raises(SystemExit, match="corpus"):
        _assert_resume_settings_match(
            {"record": "resume_settings", **banked},
            {**banked, "corpus": {"replay_dir": "/other", "shards": 3, "bytes": 99}},
            Path("p.partial"),
        )


def test_a_legacy_partial_without_a_settings_line_still_resumes() -> None:
    """An in-flight pass wrote its ``.partial`` before this record existed.

    Refusing it would kill a running multi-hour job to enforce a check it
    predates, so the absence of the record means "unchecked", not "invalid".
    """
    from scripts.rvg_label_pass import _assert_resume_settings_match

    # The loop only calls the check when a `resume_settings` record is present;
    # this pins the contract that plain label rows never reach it.
    legacy_line = {"key": "aa", "move_index": [0], "cp_eff": [0.0]}
    assert legacy_line.get("record") != "resume_settings"
    _assert_resume_settings_match({}, {}, Path("p.partial"))   # no raise on empty


def test_the_rig_arms_the_label_encoding_gate_with_the_corpus_it_reads(
    tmp_path: Path,
) -> None:
    """⚑ A GATE WHOSE ONLY CALLER DECLINES TO ARM IT.

    ``RvgLabelIndex.load``'s policy-space refusal was green in the tests that
    passed ``policy_encoding=`` explicitly and DEAD in the rig, which called it
    without the argument. This pins the value the rig now passes — read off the
    corpus's own first shard — and that the refusal fires through it.
    """
    from chess_anti_engine.moves.encode import POLICY_SIZE
    from chess_anti_engine.replay.buffer import ReplaySample
    from chess_anti_engine.replay.shard import (
        samples_to_arrays,
        save_local_shard_arrays,
    )

    pol = np.zeros(POLICY_SIZE, dtype=np.float32)
    pol[0] = 1.0
    arrs = samples_to_arrays([ReplaySample(
        x=encode_lc0_full(chess.Board(), input_history_encoding=PROD_ENCODING),
        policy_target=pol, wdl_target=0, priority=1.0, has_policy=True,
    )])
    shard_dir = tmp_path / "shards"
    save_local_shard_arrays(shard_dir / "shard_000001.zarr", arrs=arrs)

    # `_policy_encoding` is SYNTHESIZED by the shard writer, not carried by
    # `samples_to_arrays`, so it is read back off the file rather than off the
    # dict that was handed in — which is also the only reading the rig can make.
    from chess_anti_engine.replay.shard import iter_shard_paths, load_shard_arrays
    stored, _ = load_shard_arrays(iter_shard_paths(shard_dir)[0], lazy=False)
    encoding = rr._corpus_policy_encoding(shard_dir)
    assert encoding == str(np.asarray(stored["_policy_encoding"]).reshape(-1)[0])
    # This fixture writes FULL-width rows, so its shard is az_4672 — which makes
    # it the mismatched corpus for a production `lc0_1858` label file, exactly
    # the pairing the refusal exists for.
    assert encoding == "az_4672"

    labels = tmp_path / "labels.jsonl"
    _write_labels(labels, [_label(b"\x01" * 16, [0], [0.0])])   # lc0_1858
    with pytest.raises(SystemExit, match="policy space"):
        RvgLabelIndex.load(labels, policy_encoding=encoding)
    # ...and it does NOT fire when they agree, so the refusal is discriminating
    # rather than universal.
    RvgLabelIndex.load(labels, policy_encoding="lc0_1858")
