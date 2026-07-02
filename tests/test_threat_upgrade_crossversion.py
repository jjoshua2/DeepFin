"""Cross-version replay-upgrade pipeline (v1/v2/v3) for ``threat_upgrade.py``.

``upgrade_arrays_to_planes`` normalizes stored shards across every registered
input-plane width (146 v1 / 175 v2_threats / 179 v3_checks / 181 v3_xray /
177 v3_see / 183 v3_passers). It recomputes the FULL extra block from the 112
LC0 base planes — reading only planes below 146 (preserved identically across
versions) and validating the recomputed v1 block against the stored
``[112:146]`` prefix — so a 146/175/179/… chunk can all be recomputed to any
target version without ever zero-padding. A silent regression here corrupts
training inputs for the whole replay window, so these tests pin:

  1. cross-version recompute bit-matches a NATIVE encode at the wider version;
  2. equal-width zero-pad rows are repaired (not passed through);
  3. ``version_for_input_planes`` round-trips and rejects unknown widths;
  4. ``x_lc0_root`` is carried to the target width consistently with ``x``;
  5. the stored-width > target case is refused (no truncation).

Mirrors the random-board / chunk-build helpers in ``test_v2_threat_upgrade.py``
and ``test_threat_planes.py``.
"""
from __future__ import annotations

import random

import chess
import numpy as np
import pytest

from chess_anti_engine.encoding import (
    encode_position,
    input_plane_count,
    version_for_input_planes,
)
from chess_anti_engine.encoding.features import (
    EXTRA_FEATURE_VERSIONS,
    EXTRA_FEATURES_V1,
    EXTRA_FEATURES_V2_THREATS,
    EXTRA_FEATURES_V3_CHECKS,
    EXTRA_FEATURES_V3_PASSERS,
    EXTRA_FEATURES_V3_SEE,
    EXTRA_FEATURES_V3_XRAY,
)
from chess_anti_engine.encoding.lc0 import (
    LC0_HISTORY_LEGACY,
    LC0_HISTORY_ROOT,
)
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.replay.threat_upgrade import (
    V1_INPUT_PLANES,
    _LC0_PLANES,
    upgrade_arrays_to_planes,
)

# Every registered version paired with its full input-plane width.
_VERSION_WIDTHS = {v: input_plane_count(v) for v in EXTRA_FEATURE_VERSIONS}

# A few stored→target pairs where stored_width < target_width, spanning the
# v1→v2 jump and several v2/v3→v3 cross-version recomputes (the path that must
# NOT zero-pad). 146→175 lands on the v2 delegate; the rest take the
# version-general recompute branch.
_UPGRADE_PAIRS = (
    (EXTRA_FEATURES_V1, EXTRA_FEATURES_V2_THREATS),       # 146 -> 175 (v2 delegate)
    (EXTRA_FEATURES_V1, EXTRA_FEATURES_V3_CHECKS),        # 146 -> 179
    (EXTRA_FEATURES_V2_THREATS, EXTRA_FEATURES_V3_CHECKS),  # 175 -> 179
    (EXTRA_FEATURES_V2_THREATS, EXTRA_FEATURES_V3_XRAY),    # 175 -> 181
    (EXTRA_FEATURES_V3_SEE, EXTRA_FEATURES_V3_PASSERS),     # 177 -> 183
    (EXTRA_FEATURES_V3_CHECKS, EXTRA_FEATURES_V3_XRAY),     # 179 -> 181
)


def _random_positions(seed: int, n: int) -> list[chess.Board]:
    rng = random.Random(seed)
    out: list[chess.Board] = []
    while len(out) < n:
        b = chess.Board()
        for _ in range(rng.randint(2, 80)):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(rng.choice(moves))
        if not b.is_game_over():
            out.append(b)
    return out


def _encode_rows(boards: list[chess.Board], enc: str, version: str) -> np.ndarray:
    rows = [
        encode_position(b, input_history_encoding=enc, input_extra_features=version)
        for b in boards
    ]
    return np.stack(rows).astype(np.float16)


def _chunk(x: np.ndarray, enc: str) -> dict[str, np.ndarray]:
    n = x.shape[0]
    policy = np.zeros((n, POLICY_SIZE), dtype=np.float16)
    policy[:, :8] = 0.125
    return {
        "x": x,
        "policy_target": policy,
        "wdl_target": np.zeros((n,), dtype=np.int8),
        "priority": np.ones((n,), dtype=np.float32),
        "has_policy": np.ones((n,), dtype=np.uint8),
        "_input_history_encoding": np.asarray(enc),
    }


# ---------------------------------------------------------------------------
# 1. Cross-version recompute round-trip vs native encode (key correctness)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("stored_v", "target_v"), _UPGRADE_PAIRS)
def test_crossversion_recompute_matches_native_encode(stored_v, target_v):
    """Stored-at-narrower → upgrade-to-wider must equal a native wider encode.

    The recompute-from-112-LC0-base path must reproduce the native wider extra
    block exactly, proving the upgrade never zero-pads or mis-derives across
    versions (e.g. a stored 179 chunk loaded by a 181 model).
    """
    enc = LC0_HISTORY_LEGACY
    boards = _random_positions(seed=4242, n=24)
    stored = _encode_rows(boards, enc, stored_v)
    target_width = _VERSION_WIDTHS[target_v]
    native_wider = _encode_rows(boards, enc, target_v)
    assert stored.shape[1] < target_width  # genuinely an upgrade

    out, stats = upgrade_arrays_to_planes(_chunk(stored, enc), target_planes=target_width)

    assert out["x"].shape[1] == target_width
    assert stats.upgraded_rows == len(boards)
    assert stats.dropout_rows == 0
    np.testing.assert_array_equal(out["x"], native_wider)
    # The recomputed extra block carries real signal (not zero-padded).
    assert out["x"][:, _LC0_PLANES:].any()


def test_crossversion_recompute_with_black_to_move_rows():
    """Black-to-move rows decode via the color-flipped mirror; the recompute
    must still bit-match the native wider encode (side-to-move oriented)."""
    enc = LC0_HISTORY_LEGACY
    boards = _random_positions(seed=99, n=30)
    btm = [b for b in boards if b.turn == chess.BLACK]
    assert len(btm) >= 5  # the seed actually exercises black-to-move rows
    stored = _encode_rows(btm, enc, EXTRA_FEATURES_V2_THREATS)
    native_wider = _encode_rows(btm, enc, EXTRA_FEATURES_V3_CHECKS)
    out, _ = upgrade_arrays_to_planes(
        _chunk(stored, enc), target_planes=_VERSION_WIDTHS[EXTRA_FEATURES_V3_CHECKS],
    )
    np.testing.assert_array_equal(out["x"], native_wider)


# ---------------------------------------------------------------------------
# 2. Equal-width zero-pad repair
# ---------------------------------------------------------------------------

def _zero_pad_to(x_v1: np.ndarray, target_width: int) -> np.ndarray:
    """Mimic the legacy zero-pad load path: v1 prefix + all-zero extra block."""
    pad = np.zeros(
        (x_v1.shape[0], target_width - V1_INPUT_PLANES, 8, 8), dtype=x_v1.dtype,
    )
    return np.concatenate([x_v1, pad], axis=1)


@pytest.mark.parametrize(
    "target_v",
    [EXTRA_FEATURES_V3_CHECKS, EXTRA_FEATURES_V3_XRAY, EXTRA_FEATURES_V3_PASSERS],
)
def test_equal_width_zero_pad_repaired_to_native(target_v):
    """A chunk already at the target v3 width but with an all-zero post-v1
    extra block (the legacy zero-pad artifact) must be REPAIRED in place to the
    native v3 block — not left zero, and not an unconditional passthrough."""
    enc = LC0_HISTORY_LEGACY
    target_width = _VERSION_WIDTHS[target_v]
    boards = _random_positions(seed=7, n=16)
    x_v1 = _encode_rows(boards, enc, EXTRA_FEATURES_V1)
    x_padded = _zero_pad_to(x_v1, target_width)
    x_native = _encode_rows(boards, enc, target_v)

    # Mix in one already-native row: it must pass through bit-identical.
    x_padded[3] = x_native[3]
    # Pre-repair: only the seeded native row carries signal beyond v1; all the
    # zero-pad rows have an all-zero extra block.
    live_extra = x_padded[:, V1_INPUT_PLANES:].any(axis=(1, 2, 3))
    assert int(live_extra.sum()) == 1
    assert live_extra[3]

    out, stats = upgrade_arrays_to_planes(
        _chunk(x_padded, enc), target_planes=target_width,
    )
    assert out["x"].shape[1] == target_width
    # All zero-pad rows recomputed; the pre-native row is not re-touched.
    assert stats.upgraded_rows == len(boards) - 1
    np.testing.assert_array_equal(out["x"], x_native)
    assert out["x"][:, V1_INPUT_PLANES:].any()  # repaired, not left zero

    # Repaired chunk is now fully native: a rerun is a no-op passthrough.
    out2, stats2 = upgrade_arrays_to_planes(out, target_planes=target_width)
    assert out2 is out
    assert stats2.upgraded_rows == 0


def test_equal_width_native_chunk_passthrough():
    """A natively-encoded equal-width chunk must pass through untouched."""
    enc = LC0_HISTORY_LEGACY
    target_width = _VERSION_WIDTHS[EXTRA_FEATURES_V3_XRAY]
    boards = _random_positions(seed=31, n=10)
    x_native = _encode_rows(boards, enc, EXTRA_FEATURES_V3_XRAY)
    chunk = _chunk(x_native, enc)
    out, stats = upgrade_arrays_to_planes(chunk, target_planes=target_width)
    assert out is chunk
    assert stats.upgraded_rows == 0


# ---------------------------------------------------------------------------
# 3. version_for_input_planes round-trips + unknown-width raise
# ---------------------------------------------------------------------------

def test_version_for_input_planes_round_trips():
    """Every registered width maps back to its version (146/175/179/181/177/183)."""
    for version, width in _VERSION_WIDTHS.items():
        assert version_for_input_planes(width) == version
    # Pin the exact documented counts so a silent registry change is caught.
    assert _VERSION_WIDTHS == {
        EXTRA_FEATURES_V1: 146,
        EXTRA_FEATURES_V2_THREATS: 175,
        EXTRA_FEATURES_V3_CHECKS: 179,
        EXTRA_FEATURES_V3_XRAY: 181,
        EXTRA_FEATURES_V3_SEE: 177,
        EXTRA_FEATURES_V3_PASSERS: 183,
    }


@pytest.mark.parametrize("bad_width", [0, 112, 145, 147, 176, 200, 999])
def test_version_for_input_planes_unknown_width_raises(bad_width):
    with pytest.raises(ValueError, match="no extra-features version"):
        version_for_input_planes(bad_width)


# ---------------------------------------------------------------------------
# 4. x_lc0_root carry-over
# ---------------------------------------------------------------------------

def test_x_lc0_root_carried_to_target_width():
    """When the stored arrs include ``x_lc0_root``, it must be recomputed to the
    same target width as ``x`` and match a native wider encode (no crash)."""
    enc = LC0_HISTORY_LEGACY
    target_v = EXTRA_FEATURES_V3_CHECKS
    target_width = _VERSION_WIDTHS[target_v]
    boards = _random_positions(seed=2024, n=14)
    x_stored = _encode_rows(boards, enc, EXTRA_FEATURES_V2_THREATS)
    # x_lc0_root uses the ROOT history encoding but the same positions/frame.
    root_stored = _encode_rows(boards, LC0_HISTORY_ROOT, EXTRA_FEATURES_V2_THREATS)
    root_native = _encode_rows(boards, LC0_HISTORY_ROOT, target_v)

    chunk = _chunk(x_stored, enc)
    chunk["x_lc0_root"] = root_stored
    chunk["has_x_lc0_root"] = np.ones((len(boards),), dtype=np.uint8)

    out, _ = upgrade_arrays_to_planes(chunk, target_planes=target_width)
    assert out["x"].shape[1] == target_width
    assert out["x_lc0_root"].shape[1] == target_width
    np.testing.assert_array_equal(out["x_lc0_root"], root_native)


def test_x_lc0_root_unflagged_zero_rows_stay_zero():
    """Unflagged ``x_lc0_root`` rows are stored as all-zero; the all-zero live
    gate must keep their extra block zero rather than recompute garbage."""
    enc = LC0_HISTORY_LEGACY
    target_v = EXTRA_FEATURES_V3_CHECKS
    target_width = _VERSION_WIDTHS[target_v]
    boards = _random_positions(seed=555, n=8)
    x_stored = _encode_rows(boards, enc, EXTRA_FEATURES_V2_THREATS)
    root_stored = _encode_rows(boards, LC0_HISTORY_ROOT, EXTRA_FEATURES_V2_THREATS)
    root_native = _encode_rows(boards, LC0_HISTORY_ROOT, target_v)

    has = np.ones((len(boards),), dtype=np.uint8)
    has[2] = 0
    root_stored[2] = 0
    root_native[2] = 0

    chunk = _chunk(x_stored, enc)
    chunk["x_lc0_root"] = root_stored
    chunk["has_x_lc0_root"] = has

    out, _ = upgrade_arrays_to_planes(chunk, target_planes=target_width)
    np.testing.assert_array_equal(out["x_lc0_root"], root_native)
    assert not out["x_lc0_root"][2].any()  # unflagged row stays fully zero


# ---------------------------------------------------------------------------
# 5. Guard: stored width != target — recognized vs unrecognized widths
# ---------------------------------------------------------------------------

def test_stored_wider_than_target_recomputes_not_truncates():
    """A wider→narrower request (181→179) is NOT a plane-dropping truncation:
    the recompute reads only the version-independent LC0 base + the validated
    v1 prefix, never the stored wider extra block, so it rebuilds a valid
    narrower block from scratch that bit-matches a native narrower encode.

    (The hard "refusing to truncate" guard lives one layer up in
    ``DiskReplayBuffer.add_many_arrays``, not in this recompute function — so
    this asserts the safe recompute, not a raise.)"""
    enc = LC0_HISTORY_LEGACY
    boards = _random_positions(seed=17, n=6)
    x_wide = _encode_rows(boards, enc, EXTRA_FEATURES_V3_XRAY)  # 181
    narrow_target = _VERSION_WIDTHS[EXTRA_FEATURES_V3_CHECKS]   # 179
    native_narrow = _encode_rows(boards, enc, EXTRA_FEATURES_V3_CHECKS)
    assert x_wide.shape[1] > narrow_target

    out, stats = upgrade_arrays_to_planes(_chunk(x_wide, enc), target_planes=narrow_target)
    assert out["x"].shape[1] == narrow_target
    assert stats.upgraded_rows == len(boards)
    np.testing.assert_array_equal(out["x"], native_narrow)


def test_unrecognized_stored_width_raises():
    """A stored chunk at a width no registered version produces has no
    trustworthy v1 prefix to validate against → hard error."""
    enc = LC0_HISTORY_LEGACY
    boards = _random_positions(seed=88, n=4)
    # Trim a 181-wide v3_xray chunk to 176 (an unregistered width).
    x_xray = _encode_rows(boards, enc, EXTRA_FEATURES_V3_XRAY)
    bad = _chunk(x_xray[:, :176], enc)
    assert bad["x"].shape[1] == 176
    with pytest.raises(ValueError, match="not a recognized version"):
        upgrade_arrays_to_planes(
            bad, target_planes=_VERSION_WIDTHS[EXTRA_FEATURES_V3_CHECKS],
        )


def test_unknown_target_planes_raises():
    """An unrecognized TARGET width cannot be mapped to a version → raises before
    any recompute is attempted."""
    enc = LC0_HISTORY_LEGACY
    boards = _random_positions(seed=3, n=4)
    x_v1 = _encode_rows(boards, enc, EXTRA_FEATURES_V1)
    with pytest.raises(ValueError, match="no extra-features version"):
        upgrade_arrays_to_planes(_chunk(x_v1, enc), target_planes=200)
