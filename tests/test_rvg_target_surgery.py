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

import chess
import numpy as np
import pytest

from chess_anti_engine.eval.audit import decode_board_from_planes
from chess_anti_engine.eval.rvg_surgery import (
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

    The renormalization is a single global constant, so the RATIO between any
    two unlisted moves — and their ratio to the untouched listed move — is
    preserved exactly. Any per-move default on the unlisted set breaks this.
    """
    t = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    out, _ = apply_veto_edit(
        t, np.array([0]), np.array([400.0]),
        lam=0.02, tau_cp=10.0, veto_cp=1000.0,
    )
    unlisted = np.array([1, 2, 3])
    ratios_before = t[unlisted] / t[unlisted].sum()
    ratios_after = out[unlisted] / out[unlisted].sum()
    assert np.allclose(ratios_before, ratios_after, atol=1e-7)


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


def test_g_powers_before_it_renormalizes() -> None:
    """Renormalizing q or t AFTER the power is a different (and wrong) function.

    Hand check with a deliberately UNNORMALIZED-looking listed pair: with
    alpha = 0.5 the correct answer is proportional to sqrt(t*q) with q already
    normalized over the listed set. A version that raised the powers and then
    rescaled q would land somewhere else, and both answers are finite
    distributions — nothing downstream would complain.
    """
    t = np.array([0.8, 0.2], dtype=np.float32)
    out, _ = apply_geometric_blend(
        t, np.array([0, 1]), np.array([0.0, 200.0]), alpha=0.5, temp_cp=100.0,
    )
    q = np.array([1.0, math.exp(-2.0)])
    q = q / q.sum()
    unnorm = np.sqrt(np.array([0.8, 0.2]) * q)
    assert np.allclose(out, unnorm / unnorm.sum(), atol=1e-6)


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

    def sample_batch_arrays(self, batch_size: int, **kw: object) -> dict:
        del batch_size, kw   # a fixed batch; the house idiom for an unused arg
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
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"record": "provenance", "name": "bt4-test"}) + "\n")
        fh.write(json.dumps({
            "key": key.hex(), "policy": {ucis[0]: 0.7, ucis[1]: 0.3},
        }) + "\n")
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
    with open(path, "w", encoding="utf-8") as fh:
        # Written in REVERSE order on purpose.
        fh.write(json.dumps({
            "key": key.hex(),
            "policy": {ucis[2]: 0.6, ucis[1]: 0.3, ucis[0]: 0.1},
        }) + "\n")
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
    ext_path.write_text(json.dumps({"key": (b"\xee" * 16).hex(), "policy": {}}) + "\n")
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


def test_an_external_q_on_an_arm_with_no_g_stage_is_a_refusal(tmp_path: Path) -> None:
    targets = np.stack([_VG_T, _VG_T])
    inner = _FakeInner([_ROUNDTRIP_FENS[0], _ROUNDTRIP_FENS[3]], targets)
    index = _index_for(inner.keys(), tmp_path,
                       {0: (_VG_IDX.tolist(), (-_VG_REG).tolist())})
    ext_path = tmp_path / "ext.jsonl"
    ext_path.write_text(json.dumps({"key": (b"\xee" * 16).hex(), "policy": {}}) + "\n")
    with pytest.raises(SystemExit, match="no G stage"):
        rr._RvgTargetSurgeryBuffer(
            inner, _spec(arm="r", r_weight=0.5), index,
            external_q=RvgExternalPolicyIndex.load(ext_path),
        )
