"""v2_threats input planes: C/Python parity, orientation, goldens, warm-start.

Plane indices below are within the extra block (absolute = 112 + idx); see
the layout table in chess_anti_engine/encoding/features.py.
"""
from __future__ import annotations

import random

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.encoding import (
    encode_position,
    encode_position_into,
    input_plane_count,
)
from chess_anti_engine.encoding.features import (
    EXTRA_FEATURE_VERSIONS,
    EXTRA_FEATURES_V1,
    EXTRA_FEATURES_V2_THREATS,
    extra_feature_plane_count,
    extra_feature_planes_c,
    extra_feature_planes_fast,
    normalize_extra_features_encoding,
)
from chess_anti_engine.model import ModelConfig, build_model, load_state_dict_tolerant

# Extra-block plane indices (v2_threats family).
_ATTACKED_BY_US = 34
_ATTACKED_BY_THEM = 35
_HANGING_US = 50
_HANGING_THEM = 51


def test_new_model_defaults_to_v2_threat_planes() -> None:
    assert ModelConfig().input_extra_features == EXTRA_FEATURES_V2_THREATS
    assert input_plane_count(ModelConfig().input_extra_features) == 175
    assert input_plane_count(EXTRA_FEATURES_V1) == 146
_CHEAPER_US = 52
_CHEAPER_THEM = 53
_SAFE_CHECK_N = 54
_SAFE_CHECK_B = 55
_SAFE_CHECK_R = 56
_SAFE_CHECK_Q = 57
_CONTROL_MARGIN = 58
_TENSION_US = 59
_TENSION_THEM = 60
_STORM_VS_US = 61


def _random_board(rng: random.Random, plies: int) -> chess.Board:
    b = chess.Board()
    for _ in range(plies):
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(rng.choice(moves))
    return b


def _sq(plane: np.ndarray, square: str) -> float:
    """Plane value at a square for a WHITE-to-move position (row == rank)."""
    sq = chess.parse_square(square)
    return float(plane[chess.square_rank(sq), chess.square_file(sq)])


def test_normalize_and_plane_counts():
    assert normalize_extra_features_encoding(None) == EXTRA_FEATURES_V1
    assert normalize_extra_features_encoding("v1") == EXTRA_FEATURES_V1
    assert normalize_extra_features_encoding("V2_Threats") == EXTRA_FEATURES_V2_THREATS
    assert extra_feature_plane_count("v1") == 34
    assert extra_feature_plane_count("v2_threats") == 63
    assert input_plane_count("v1") == 146
    assert input_plane_count("v2_threats") == 175
    with pytest.raises(ValueError, match="unknown input_extra_features"):
        normalize_extra_features_encoding("v3_nope")


def test_c_python_parity_random_positions():
    rng = random.Random(1234)
    for i in range(220):
        b = _random_board(rng, rng.randint(0, 100))
        py = extra_feature_planes_fast(b, version="v2_threats")
        cc = extra_feature_planes_c(b, version="v2_threats")
        assert py.shape == (63, 8, 8)
        assert cc.shape == (63, 8, 8)
        np.testing.assert_allclose(
            py, cc, atol=1e-6,
            err_msg=f"parity mismatch at position {i}: {b.fen()}",
        )


@pytest.mark.parametrize("version", EXTRA_FEATURE_VERSIONS)
def test_c_python_parity_all_versions(version):
    """C-accelerated recompute must match the pure-Python recompute for EVERY
    supported extra-feature version (v1, v2_threats, and the v3 families:
    v3_see, v3_checks, v3_xray, v3_passers).

    Mirrors ``test_c_python_parity_random_positions`` but parametrized over the
    full version list — this catches per-version C/Python divergences such as
    the SEE least-valuable-attacker tie-break (knight vs equal-valued bishop).
    """
    rng = random.Random(20260617)
    n_planes = extra_feature_plane_count(version)
    for i in range(220):
        b = _random_board(rng, rng.randint(0, 100))
        py = extra_feature_planes_fast(b, version=version)
        cc = extra_feature_planes_c(b, version=version)
        assert py.shape == (n_planes, 8, 8)
        assert cc.shape == (n_planes, 8, 8)
        np.testing.assert_allclose(
            py, cc, atol=1e-6,
            err_msg=f"parity mismatch ({version}) at position {i}: {b.fen()}",
        )


def test_v2_does_not_change_v1_planes():
    rng = random.Random(7)
    for _ in range(25):
        b = _random_board(rng, rng.randint(0, 60))
        v1 = extra_feature_planes_fast(b, version="v1")
        v2 = extra_feature_planes_fast(b, version="v2_threats")
        np.testing.assert_array_equal(v1, v2[:34])
        v1c = extra_feature_planes_c(b, version="v1")
        v2c = extra_feature_planes_c(b, version="v2_threats")
        np.testing.assert_array_equal(v1c, v2c[:34])


def test_orientation_mirror_invariance():
    """A position and its color-flipped mirror must produce identical extra
    planes — everything is side-to-move oriented."""
    rng = random.Random(99)
    checked = 0
    for _ in range(60):
        b = _random_board(rng, rng.randint(0, 60))
        if b.is_game_over():
            continue
        m = b.mirror()
        for fn in (extra_feature_planes_fast, extra_feature_planes_c):
            np.testing.assert_allclose(
                fn(b, version="v2_threats"), fn(m, version="v2_threats"),
                atol=1e-6, err_msg=f"orientation mismatch ({fn.__name__}): {b.fen()}",
            )
        checked += 1
    assert checked >= 40


@pytest.mark.parametrize("impl", [extra_feature_planes_fast, extra_feature_planes_c])
class TestGoldenPositions:
    def test_hanging_piece(self, impl):
        # White rook a5 attacked by black queen a8 and undefended; the queen
        # is itself attacked by the rook, undefended, AND rook < queen.
        b = chess.Board("q3k3/8/8/R7/8/8/8/4K3 w - - 0 1")
        f = impl(b, version="v2_threats")
        assert _sq(f[_HANGING_US], "a5") == 1.0
        assert _sq(f[_HANGING_THEM], "a8") == 1.0
        assert _sq(f[_CHEAPER_THEM], "a8") == 1.0   # queen attacked by cheaper rook
        assert _sq(f[_CHEAPER_US], "a5") == 0.0     # rook attacked by pricier queen

    def test_pawn_attacks_queen(self, impl):
        # White pawn c4 attacks black queen d5 (pawn → queen threat).
        b = chess.Board("4k3/8/8/3q4/2P5/8/8/4K3 w - - 0 1")
        f = impl(b, version="v2_threats")
        assert _sq(f[_CHEAPER_THEM], "d5") == 1.0
        assert _sq(f[_ATTACKED_BY_US], "d5") == 1.0

    def test_safe_checks_each_piece_type(self, impl):
        # Lone kings e1/e8: check-origin squares are derived from the enemy
        # king square per piece type, minus enemy-attacked and own-occupied
        # squares — independent of which pieces we actually have.
        b = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        f = impl(b, version="v2_threats")
        # Knight: d6/f6 check e8 and aren't touched by the e8 king.
        assert _sq(f[_SAFE_CHECK_N], "d6") == 1.0
        assert _sq(f[_SAFE_CHECK_N], "f6") == 1.0
        # g7 checks e8 but is adjacent-attacked? g7 is NOT attacked by e8.
        assert _sq(f[_SAFE_CHECK_N], "g7") == 1.0
        # Bishop: b5 sees e8 via b5-c6-d7-e8 diagonal; d7 is king-adjacent → unsafe.
        assert _sq(f[_SAFE_CHECK_B], "b5") == 1.0
        assert _sq(f[_SAFE_CHECK_B], "d7") == 0.0
        # Rook: e2 checks down the e-file; e7/d8/f8 are king-adjacent → unsafe.
        assert _sq(f[_SAFE_CHECK_R], "e2") == 1.0
        assert _sq(f[_SAFE_CHECK_R], "e7") == 0.0
        assert _sq(f[_SAFE_CHECK_R], "a8") == 1.0
        # Queen: union of bishop + rook check squares.
        assert _sq(f[_SAFE_CHECK_Q], "e2") == 1.0
        assert _sq(f[_SAFE_CHECK_Q], "b5") == 1.0
        assert _sq(f[_SAFE_CHECK_Q], "e7") == 0.0

    def test_pawn_tension(self, impl):
        # White pawn d4 and black pawn e5 can capture each other.
        b = chess.Board("4k3/8/8/4p3/3P4/8/8/4K3 w - - 0 1")
        f = impl(b, version="v2_threats")
        assert _sq(f[_TENSION_US], "d4") == 1.0
        assert _sq(f[_TENSION_THEM], "e5") == 1.0
        # Control margin on e5: white pawn d4 attacks it, black king doesn't.
        assert _sq(f[_CONTROL_MARGIN], "e5") == pytest.approx(0.25)

    def test_pawn_storm(self, impl):
        # Black pawn h4 storming the white king on g1: adjacent file,
        # rank distance 3 → 1 - 3/7.
        b = chess.Board("4k3/8/8/8/7p/8/8/6K1 w - - 0 1")
        f = impl(b, version="v2_threats")
        assert _sq(f[_STORM_VS_US], "h4") == pytest.approx(1.0 - 3.0 / 7.0)
        # A far-file pawn contributes nothing.
        b2 = chess.Board("4k3/8/8/8/p7/8/8/6K1 w - - 0 1")
        f2 = impl(b2, version="v2_threats")
        assert float(f2[_STORM_VS_US].sum()) == 0.0

    def test_attacked_by_them_includes_king_ring(self, impl):
        b = chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
        f = impl(b, version="v2_threats")
        assert _sq(f[_ATTACKED_BY_THEM], "d7") == 1.0
        assert _sq(f[_ATTACKED_BY_THEM], "e7") == 1.0
        assert _sq(f[_ATTACKED_BY_THEM], "e6") == 0.0


def test_encode_position_paths_agree_v2():
    b = chess.Board()
    b.push_san("e4")
    b.push_san("c5")
    b.push_san("Nf3")
    x = encode_position(b, input_extra_features="v2_threats")
    assert x.shape == (175, 8, 8)
    out = np.zeros((175, 8, 8), dtype=np.float32)
    encode_position_into(b, out, input_extra_features="v2_threats")
    np.testing.assert_allclose(x, out, atol=1e-6)
    np.testing.assert_array_equal(
        x[112:], extra_feature_planes_c(b, version="v2_threats"),
    )


def test_feature_dropout_covers_threat_planes():
    b = chess.Board()
    x = encode_position(
        b, input_extra_features="v2_threats",
        feature_dropout_p=1.0, rng=np.random.default_rng(0),
    )
    assert float(np.abs(x[112:]).sum()) == 0.0   # whole extra block zeroed
    assert float(np.abs(x[:112]).sum()) > 0.0


def _tiny_cfg(version: str) -> ModelConfig:
    return ModelConfig(
        embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False,
        input_extra_features=version,
    )


def test_v1_checkpoint_warm_starts_v2_bit_identical():
    torch.manual_seed(0)
    v1 = build_model(_tiny_cfg("v1")).eval()
    v2 = build_model(_tiny_cfg("v2_threats")).eval()
    load_state_dict_tolerant(v2, v1.state_dict(), label="test-warmstart")

    b = chess.Board()
    b.push_san("d4")
    b.push_san("Nf6")
    x1 = torch.from_numpy(encode_position(b)).unsqueeze(0)
    x2 = torch.from_numpy(
        encode_position(b, input_extra_features="v2_threats")
    ).unsqueeze(0)
    assert float(x2[0, 146:].abs().sum()) > 0.0  # new planes carry signal
    with torch.no_grad():
        o1 = v1(x1)
        o2 = v2(x2)
    for key in o1:
        assert torch.equal(o1[key], o2[key]), f"head {key} not bit-identical"


def test_old_shards_zero_padded_into_v2_buffer(tmp_path):
    from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer

    rng = np.random.default_rng(0)
    buf = DiskReplayBuffer(
        1000, shard_dir=tmp_path, rng=rng, read_only=False, input_planes=175, shuffle_cap=100,
    )
    try:
        n = 8
        policy = np.zeros((n, 4672), dtype=np.float16)
        policy[:, 0] = 1.0
        old_chunk = {
            "x": rng.random((n, 146, 8, 8)).astype(np.float16),
            "policy_target": policy,
            "wdl_target": np.zeros((n,), dtype=np.int8),
            "priority": np.ones((n,), dtype=np.float32),
            "has_policy": np.ones((n,), dtype=np.uint8),
        }
        buf.add_many_arrays(dict(old_chunk))
        out = buf.sample_batch_arrays(4, wdl_balance=False)
        assert out["x"].shape[1] == 175
        assert float(np.abs(out["x"][:, 146:]).sum()) == 0.0
        assert float(np.abs(out["x"][:, :146]).sum()) > 0.0
    finally:
        buf.close()


def test_v2_shards_into_v1_buffer_rejected(tmp_path):
    from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer

    rng = np.random.default_rng(0)
    buf = DiskReplayBuffer(
        1000, shard_dir=tmp_path, rng=rng, read_only=False, input_planes=146, shuffle_cap=100,
    )
    try:
        n = 2
        policy = np.zeros((n, 4672), dtype=np.float16)
        policy[:, 0] = 1.0
        chunk = {
            "x": np.zeros((n, 175, 8, 8), dtype=np.float16),
            "policy_target": policy,
            "wdl_target": np.zeros((n,), dtype=np.int8),
            "priority": np.ones((n,), dtype=np.float32),
            "has_policy": np.ones((n,), dtype=np.uint8),
        }
        with pytest.raises(ValueError, match="refusing to truncate"):
            buf.add_many_arrays(chunk)
    finally:
        buf.close()


# ---------------------------------------------------------------------------
# Production-surface coverage at 175 planes (review follow-up)
# ---------------------------------------------------------------------------

def _midgame_board() -> chess.Board:
    b = chess.Board()
    for san in ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6"):
        b.push_san(san)
    return b


def test_cboard_encode_full_v2_and_v1_compat():
    from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast

    b = _midgame_board()
    cb = cboard_from_board_fast(b)
    x2 = cb.encode_full(0, 63)
    assert x2.shape == (175, 8, 8)
    np.testing.assert_allclose(
        x2[112:], extra_feature_planes_fast(b, version="v2_threats"), atol=1e-6,
    )
    # v1 path stays byte-identical to the legacy method.
    np.testing.assert_array_equal(cb.encode_full(0, 34), cb.encode_146())
    np.testing.assert_array_equal(x2[:112], cb.encode_146()[:112])


def test_batch_encode_at_175_planes():
    from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast
    from chess_anti_engine.mcts._mcts_tree import batch_encode_146, batch_encode_146_bf16

    b = _midgame_board()
    cbs = [cboard_from_board_fast(b) for _ in range(3)]
    ref = cbs[0].encode_full(0, 63)

    out = np.empty((3, 175, 8, 8), dtype=np.float32)
    batch_encode_146(cbs, out)
    for i in range(3):
        np.testing.assert_allclose(out[i], ref, atol=1e-6)

    out16 = np.empty((3, 175, 8, 8), dtype=np.uint16)
    batch_encode_146_bf16(cbs, out16)
    as_f32 = np.frombuffer(
        (out16[0].astype(np.uint32) << 16).tobytes(), dtype=np.float32,
    ).reshape(175, 8, 8)
    np.testing.assert_allclose(as_f32, ref, atol=0.01)  # bf16 precision


def test_batch_process_ply_v2_planes():
    from chess_anti_engine.encoding.cboard_encode import cboard_from_board_fast
    from chess_anti_engine.mcts._mcts_tree import batch_process_ply

    b = _midgame_board()
    cbs = [cboard_from_board_fast(b) for _ in range(2)]
    expected_feat = extra_feature_planes_fast(b, version="v2_threats")
    legal = cbs[0].legal_move_indices()
    action = int(legal[0])
    n = 2
    pol = np.zeros((n, 4672), dtype=np.float32)
    wdl = np.zeros((n, 3), dtype=np.float32)
    actions = np.full((n,), action, dtype=np.int32)
    values = np.zeros((n,), dtype=np.float64)
    probs = np.zeros((n, 4672), dtype=np.float32)
    probs[:, action] = 1.0

    result = batch_process_ply(
        cbs, pol, wdl, actions, values, probs,
        0, 0.0, 0.0, 1.0, 0.0,   # diff-focus disabled
        0,                        # full-history encoding
        63,                       # n_extra: v2_threats
    )
    x = result[0]
    assert x.shape == (n, 175, 8, 8)
    np.testing.assert_allclose(x[0, 112:], expected_feat, atol=1e-6)


def test_warm_start_bit_identical_with_arc_pos_encoding():
    """Arc pos-encoding appends channels AFTER the plane block; the zero-column
    insertion at 146 must leave those columns aligned."""
    def cfg(version: str) -> ModelConfig:
        return ModelConfig(
            embed_dim=32, num_layers=1, num_heads=2, use_smolgen=False,
            input_pos_encoding="arc", input_extra_features=version,
        )

    torch.manual_seed(3)
    v1 = build_model(cfg("v1")).eval()
    v2 = build_model(cfg("v2_threats")).eval()
    load_state_dict_tolerant(v2, v1.state_dict(), label="test-arc-warmstart")

    b = _midgame_board()
    x1 = torch.from_numpy(encode_position(b)).unsqueeze(0)
    x2 = torch.from_numpy(
        encode_position(b, input_extra_features="v2_threats")
    ).unsqueeze(0)
    with torch.no_grad():
        o1 = v1(x1)
        o2 = v2(x2)
    for key in o1:
        # Warm-start makes the two forwards mathematically equal. The arc
        # pos-encoding's extra channels change the matmul reduction order, so the
        # output is not bit-identical across BLAS/torch builds (it is on some) —
        # a real misalignment bug would perturb values by O(1), not the last bit,
        # so a tight allclose checks the equivalence without depending on the build.
        assert torch.allclose(o1[key], o2[key], atol=1e-5, rtol=0.0), \
            f"head {key} not equivalent (arc)"


def test_optimizer_state_migrates_v1_to_v2():
    """A v1 trainer checkpoint's AdamW moments must zero-pad into a v2 model
    so the first opt.step() after warm-start doesn't shape-crash."""
    from chess_anti_engine.model import migrate_optimizer_input_plane_state

    torch.manual_seed(4)
    v1 = build_model(_tiny_cfg("v1"))
    opt1 = torch.optim.AdamW(v1.parameters(), lr=1e-3)
    b = _midgame_board()
    x1 = torch.from_numpy(encode_position(b)).unsqueeze(0)
    out = v1(x1)
    torch.stack([v.float().sum() for v in out.values()]).sum().backward()
    opt1.step()

    v2 = build_model(_tiny_cfg("v2_threats"))
    load_state_dict_tolerant(v2, v1.state_dict(), label="test-opt-warmstart")
    opt2 = torch.optim.AdamW(v2.parameters(), lr=1e-3)
    # torch accepts the mismatched moment tensors silently...
    opt2.load_state_dict(opt1.state_dict())
    migrated = migrate_optimizer_input_plane_state(opt2)
    assert migrated >= 2  # exp_avg + exp_avg_sq of the input embed

    x2 = torch.from_numpy(
        encode_position(b, input_extra_features="v2_threats")
    ).unsqueeze(0)
    out2 = v2(x2)
    torch.stack([v.float().sum() for v in out2.values()]).sum().backward()
    opt2.step()  # ...and without migration this raises a 146-vs-175 RuntimeError


def test_x_lc0_root_width_follows_x():
    from chess_anti_engine.replay.buffer import ReplaySample
    from chess_anti_engine.replay.shard import samples_to_arrays, validate_array_declarations

    rng = np.random.default_rng(0)
    pol = np.zeros(4672, dtype=np.float32)
    pol[0] = 1.0
    samples = [
        ReplaySample(
            x=rng.random((175, 8, 8)).astype(np.float32),
            policy_target=pol,
            wdl_target=1,
            x_lc0_root=rng.random((175, 8, 8)).astype(np.float32),
        )
        for _ in range(2)
    ]
    arrs = samples_to_arrays(samples)
    assert arrs["x_lc0_root"].shape == (2, 175, 8, 8)
    validate_array_declarations(arrs)  # must accept the v2 width


def test_write_buffer_accepts_mixed_width_chunks(tmp_path):
    from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer

    rng = np.random.default_rng(0)
    buf = DiskReplayBuffer(
        1000, shard_dir=tmp_path, rng=rng, read_only=False, input_planes=175,
        shuffle_cap=100, shard_size=8,
    )
    try:
        def chunk(planes: int, n: int) -> dict:
            policy = np.zeros((n, 4672), dtype=np.float16)
            policy[:, 0] = 1.0
            return {
                "x": rng.random((n, planes, 8, 8)).astype(np.float16),
                "policy_target": policy,
                "wdl_target": np.zeros((n,), dtype=np.int8),
                "priority": np.ones((n,), dtype=np.float32),
                "has_policy": np.ones((n,), dtype=np.uint8),
            }

        # A stale v1-width chunk and a v2-width chunk land in the same write
        # accumulator; the flush at shard_size=8 must concatenate cleanly.
        buf.add_many_arrays(chunk(146, 5))
        buf.add_many_arrays(chunk(175, 5))
        assert len(buf) >= 8
        out = buf.sample_batch_arrays(4, wdl_balance=False)
        assert out["x"].shape[1] == 175
    finally:
        buf.close()
