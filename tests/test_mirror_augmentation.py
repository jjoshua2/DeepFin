import numpy as np

from chess_anti_engine.moves.encode import (
    COMPACT_MIRROR_POLICY_MAP,
    COMPACT_POLICY_SIZE,
    COMPACT_TO_FULL_POLICY,
    MIRROR_POLICY_MAP,
    POLICY_SIZE,
    mirror_policy_index,
)
from chess_anti_engine.replay.augment import maybe_mirror_batch_arrays, mirror_sample, mirror_x
from chess_anti_engine.replay.buffer import ReplaySample


def test_mirror_policy_map_is_permutation_and_involution():
    m = np.asarray(MIRROR_POLICY_MAP)
    assert m.shape == (POLICY_SIZE,)

    # Permutation check
    assert len(set(map(int, m.tolist()))) == POLICY_SIZE

    # Involution check
    for i in [0, 1, 2, 123, 4096, POLICY_SIZE - 1]:
        assert mirror_policy_index(mirror_policy_index(i)) == i

    # Spot-check all indices in a vectorized way
    mm = m[m]
    assert np.array_equal(mm, np.arange(POLICY_SIZE, dtype=mm.dtype))


def test_mirror_x_swaps_legacy_castling_planes():
    x = np.zeros((146, 8, 8), dtype=np.float32)
    x[96, :, :] = 1.0
    x[97, :, :] = 2.0
    x[98, :, :] = 3.0
    x[99, :, :] = 4.0

    mirrored = mirror_x(x)

    assert np.all(mirrored[96] == 2.0)
    assert np.all(mirrored[97] == 1.0)
    assert np.all(mirrored[98] == 4.0)
    assert np.all(mirrored[99] == 3.0)


def test_mirror_x_swaps_lc0_root_castling_planes():
    x = np.zeros((146, 8, 8), dtype=np.float32)
    x[104, :, :] = 1.0
    x[105, :, :] = 2.0
    x[106, :, :] = 3.0
    x[107, :, :] = 4.0

    mirrored = mirror_x(x, input_history_encoding="lc0_root")

    assert np.all(mirrored[104] == 2.0)
    assert np.all(mirrored[105] == 1.0)
    assert np.all(mirrored[106] == 4.0)
    assert np.all(mirrored[107] == 3.0)


def test_mirror_batch_swaps_castling_planes_for_x_and_lc0_root():
    batch = {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "x_lc0_root": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "policy_target": np.zeros((2, POLICY_SIZE), dtype=np.float32),
    }
    batch["x"][:, 96, :, :] = 1.0
    batch["x"][:, 97, :, :] = 2.0
    batch["x_lc0_root"][:, 104, :, :] = 3.0
    batch["x_lc0_root"][:, 105, :, :] = 4.0

    mirrored = maybe_mirror_batch_arrays(batch, rng=np.random.default_rng(1), prob=1.0)

    assert np.all(mirrored["x"][:, 96] == 2.0)
    assert np.all(mirrored["x"][:, 97] == 1.0)
    assert np.all(mirrored["x_lc0_root"][:, 104] == 4.0)
    assert np.all(mirrored["x_lc0_root"][:, 105] == 3.0)


def test_mirror_batch_preserves_policy_storage_dtype_and_values():
    for width, mirror_map in (
        (COMPACT_POLICY_SIZE, COMPACT_MIRROR_POLICY_MAP),
        (POLICY_SIZE, MIRROR_POLICY_MAP),
    ):
        for dtype in (np.float16, np.float32):
            policy = np.arange(2 * width, dtype=np.float32).reshape(2, width).astype(dtype)
            batch = {
                "x": np.zeros((2, 146, 8, 8), dtype=np.float16),
                "policy_target": policy,
            }

            mirrored = maybe_mirror_batch_arrays(
                batch, rng=np.random.default_rng(1), prob=1.0,
            )

            assert mirrored["policy_target"].dtype == dtype
            assert mirrored["x"].flags.c_contiguous
            assert all(stride > 0 for stride in mirrored["x"].strides)
            np.testing.assert_array_equal(
                mirrored["policy_target"], policy[:, mirror_map],
            )


def test_mirror_batch_uses_configured_history_for_primary_x():
    batch = {
        "x": np.zeros((1, 146, 8, 8), dtype=np.float32),
        "policy_target": np.zeros((1, POLICY_SIZE), dtype=np.float32),
    }
    batch["x"][:, 96, :, :] = 9.0
    batch["x"][:, 97, :, :] = 10.0
    batch["x"][:, 104, :, :] = 1.0
    batch["x"][:, 105, :, :] = 2.0
    batch["x"][:, 106, :, :] = 3.0
    batch["x"][:, 107, :, :] = 4.0

    mirrored = maybe_mirror_batch_arrays(
        batch,
        rng=np.random.default_rng(1),
        prob=1.0,
        input_history_encoding="lc0_root",
    )

    assert np.all(mirrored["x"][:, 96] == 9.0)
    assert np.all(mirrored["x"][:, 97] == 10.0)
    assert np.all(mirrored["x"][:, 104] == 2.0)
    assert np.all(mirrored["x"][:, 105] == 1.0)
    assert np.all(mirrored["x"][:, 106] == 4.0)
    assert np.all(mirrored["x"][:, 107] == 3.0)


def test_mirror_sample_is_involution():
    rng = np.random.default_rng(0)

    x = rng.normal(size=(18, 8, 8)).astype(np.float32)

    p = rng.random(size=(POLICY_SIZE,)).astype(np.float32)
    p /= float(p.sum())

    ps = rng.random(size=(POLICY_SIZE,)).astype(np.float32)
    ps /= float(ps.sum())

    fp = rng.random(size=(POLICY_SIZE,)).astype(np.float32)
    fp /= float(fp.sum())

    s = ReplaySample(
        x=x,
        policy_target=p,
        wdl_target=2,
        priority=1.7,
        priority_policy_kl=0.12,
        priority_q_delta=-0.08,
        priority_sf_search_gap=0.04,
        game_id=123456789,
        ply_index=18,
        has_policy=True,
        sf_wdl=np.array([0.2, 0.3, 0.5], dtype=np.float32),
        sf_move_index=int(rng.integers(0, POLICY_SIZE)),
        sf_played_move_index=int(rng.integers(0, POLICY_SIZE)),
        sf_played_rank=3,
        sf_played_regret=0.0125,
        moves_left=0.25,
        is_network_turn=True,
        input_history_encoding="lc0_root_legacy_meta",
        categorical_target=rng.random(size=(32,)).astype(np.float32),
        policy_soft_target=ps,
        future_policy_target=fp,
        has_future=True,
        volatility_target=np.array([0.01, 0.02, 0.03], dtype=np.float32),
        has_volatility=True,
        sf_volatility_target=np.array([0.04, 0.05, 0.06], dtype=np.float32),
        has_sf_volatility=True,
        future_sf_regret_sum=0.25,
        future_sf_regret_d95=0.20,
        future_sf_regret_d98=0.23,
        future_sf_regret_max=0.05,
        future_sf_regret_h4=0.11,
        future_sf_regret_h6=0.12,
        future_sf_regret_h12=0.13,
        future_sf_regret_h24=0.14,
        future_sf_regret_h50=0.15,
        future_sf_regret_count=7,
    )

    s2 = mirror_sample(mirror_sample(s))

    assert np.array_equal(s2.x, s.x)
    assert np.allclose(s2.policy_target, s.policy_target)
    assert s2.wdl_target == s.wdl_target
    assert float(s2.priority) == float(s.priority)
    assert s2.priority_policy_kl == s.priority_policy_kl
    assert s2.priority_q_delta == s.priority_q_delta
    assert s2.priority_sf_search_gap == s.priority_sf_search_gap
    assert s2.game_id == s.game_id
    assert s2.ply_index == s.ply_index
    assert bool(s2.has_policy) == bool(s.has_policy)
    assert s2.input_history_encoding == s.input_history_encoding
    assert s2.sf_played_rank == s.sf_played_rank
    assert s2.sf_played_regret == s.sf_played_regret
    assert s2.future_sf_regret_sum == s.future_sf_regret_sum
    assert s2.future_sf_regret_d95 == s.future_sf_regret_d95
    assert s2.future_sf_regret_d98 == s.future_sf_regret_d98
    assert s2.future_sf_regret_max == s.future_sf_regret_max
    assert s2.future_sf_regret_h4 == s.future_sf_regret_h4
    assert s2.future_sf_regret_h6 == s.future_sf_regret_h6
    assert s2.future_sf_regret_h12 == s.future_sf_regret_h12
    assert s2.future_sf_regret_h24 == s.future_sf_regret_h24
    assert s2.future_sf_regret_h50 == s.future_sf_regret_h50
    assert s2.future_sf_regret_count == s.future_sf_regret_count

    # Narrow the optional fields — the test set all of them, mirror_sample must
    # round-trip all of them. Plain asserts both satisfy pyright and catch a
    # future silent-None regression in mirror_sample. Per-reference (not a
    # loop) because pyright narrowing is per-expression, not transitive.
    assert (s.sf_wdl is not None)
    assert (s.sf_move_index is not None)
    assert (s.sf_played_move_index is not None)
    assert (s.moves_left is not None)
    assert (s.is_network_turn is not None)
    assert (s.categorical_target is not None)
    assert (s.policy_soft_target is not None)
    assert (s.future_policy_target is not None)
    assert (s.volatility_target is not None)
    assert (s.sf_volatility_target is not None)
    assert (s2.sf_wdl is not None)
    assert (s2.sf_move_index is not None)
    assert (s2.sf_played_move_index is not None)
    assert (s2.moves_left is not None)
    assert (s2.is_network_turn is not None)
    assert (s2.categorical_target is not None)
    assert (s2.policy_soft_target is not None)
    assert (s2.future_policy_target is not None)
    assert (s2.volatility_target is not None)
    assert (s2.sf_volatility_target is not None)

    assert np.allclose(s2.sf_wdl, s.sf_wdl)
    assert int(s2.sf_move_index) == int(s.sf_move_index)
    assert int(s2.sf_played_move_index) == int(s.sf_played_move_index)
    assert float(s2.moves_left) == float(s.moves_left)
    assert bool(s2.is_network_turn) == bool(s.is_network_turn)

    assert np.allclose(s2.categorical_target, s.categorical_target)
    assert np.allclose(s2.policy_soft_target, s.policy_soft_target)
    assert np.allclose(s2.future_policy_target, s.future_policy_target)
    assert bool(s2.has_future) == bool(s.has_future)

    assert np.allclose(s2.volatility_target, s.volatility_target)
    assert bool(s2.has_volatility) == bool(s.has_volatility)

    assert np.allclose(s2.sf_volatility_target, s.sf_volatility_target)
    assert bool(s2.has_sf_volatility) == bool(s.has_sf_volatility)


def test_mirror_batch_arrays_is_involution():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 18, 8, 8)).astype(np.float32)

    policy = rng.random(size=(4, POLICY_SIZE)).astype(np.float32)
    policy /= policy.sum(axis=1, keepdims=True)

    soft = rng.random(size=(4, POLICY_SIZE)).astype(np.float32)
    soft /= soft.sum(axis=1, keepdims=True)

    legal_mask = (rng.random(size=(4, POLICY_SIZE)) > 0.5).astype(np.uint8)
    sf_move_index = rng.integers(0, POLICY_SIZE, size=(4,), dtype=np.int32)
    sf_played_move_index = rng.integers(0, POLICY_SIZE, size=(4,), dtype=np.int32)

    batch = {
        "x": x,
        "policy_target": policy,
        "sf_policy_target": policy.copy(),
        "policy_soft_target": soft,
        "future_policy_target": soft.copy(),
        "legal_mask": legal_mask,
        "sf_move_index": sf_move_index,
        "sf_played_move_index": sf_played_move_index,
    }

    mirrored = maybe_mirror_batch_arrays(batch, rng=np.random.default_rng(1), prob=1.0)
    unmirrored = maybe_mirror_batch_arrays(mirrored, rng=np.random.default_rng(2), prob=1.0)

    for key, value in batch.items():
        assert np.array_equal(unmirrored[key], value)


def test_mirror_sample_keeps_sf_move_index_in_compact_policy_space():
    compact_idx = 10
    compact_policy = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float32)
    compact_policy[compact_idx] = 1.0
    sf_legal_mask = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.uint8)
    sf_legal_mask[compact_idx] = 1
    sample = ReplaySample(
        x=np.zeros((18, 8, 8), dtype=np.float32),
        policy_target=compact_policy,
        wdl_target=1,
        sf_move_index=compact_idx,
        sf_played_move_index=compact_idx,
        sf_policy_target=None,
        sf_legal_mask=sf_legal_mask,
    )

    mirrored = mirror_sample(sample)

    assert mirrored.sf_move_index == int(COMPACT_MIRROR_POLICY_MAP[compact_idx])
    assert mirrored.sf_played_move_index == int(COMPACT_MIRROR_POLICY_MAP[compact_idx])
    assert mirrored.sf_legal_mask is not None
    assert mirrored.sf_move_index is not None
    assert int(mirrored.sf_legal_mask[int(mirrored.sf_move_index)]) == 1


def test_mirror_sample_uses_full_map_from_policy_width_with_compact_sf_policy():
    full_idx = int(COMPACT_TO_FULL_POLICY[10])
    compact_sf = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float32)
    compact_sf[10] = 1.0
    policy = np.zeros((POLICY_SIZE,), dtype=np.float32)
    policy[full_idx] = 1.0
    sample = ReplaySample(
        x=np.zeros((18, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=1,
        sf_policy_target=compact_sf,
        sf_move_index=full_idx,
        sf_played_move_index=full_idx,
    )

    mirrored = mirror_sample(sample)

    assert mirrored.sf_move_index == mirror_policy_index(full_idx)
    assert mirrored.sf_played_move_index == mirror_policy_index(full_idx)


def test_mirror_batch_uses_compact_map_from_policy_width_without_sf_policy():
    compact_idx = 10
    batch = {
        "x": np.zeros((1, 18, 8, 8), dtype=np.float32),
        "policy_target": np.zeros((1, COMPACT_POLICY_SIZE), dtype=np.float32),
        "sf_legal_mask": np.zeros((1, COMPACT_POLICY_SIZE), dtype=np.uint8),
        "sf_move_index": np.array([compact_idx], dtype=np.int32),
        "sf_played_move_index": np.array([compact_idx], dtype=np.int32),
    }
    batch["policy_target"][0, compact_idx] = 1.0
    batch["sf_legal_mask"][0, compact_idx] = 1

    mirrored = maybe_mirror_batch_arrays(batch, rng=np.random.default_rng(1), prob=1.0)

    assert int(mirrored["sf_move_index"][0]) == int(COMPACT_MIRROR_POLICY_MAP[compact_idx])
    assert int(mirrored["sf_played_move_index"][0]) == int(COMPACT_MIRROR_POLICY_MAP[compact_idx])
    assert int(mirrored["sf_legal_mask"][0, int(mirrored["sf_move_index"][0])]) == 1


def test_mirror_batch_uses_full_map_from_policy_width_with_compact_sf_policy():
    full_idx = int(COMPACT_TO_FULL_POLICY[10])
    batch = {
        "x": np.zeros((1, 18, 8, 8), dtype=np.float32),
        "policy_target": np.zeros((1, POLICY_SIZE), dtype=np.float32),
        "sf_policy_target": np.zeros((1, COMPACT_POLICY_SIZE), dtype=np.float32),
        "sf_move_index": np.array([full_idx], dtype=np.int32),
        "sf_played_move_index": np.array([full_idx], dtype=np.int32),
    }
    batch["policy_target"][0, full_idx] = 1.0
    batch["sf_policy_target"][0, 10] = 1.0

    mirrored = maybe_mirror_batch_arrays(batch, rng=np.random.default_rng(1), prob=1.0)

    assert int(mirrored["sf_move_index"][0]) == mirror_policy_index(full_idx)
    assert int(mirrored["sf_played_move_index"][0]) == mirror_policy_index(full_idx)


def _build_per_head_masked_batch(n: int = 8, seed: int = 0):
    """Batch where each head's target indices lie in a disjoint legal-index set,
    so head-mask misalignment after mirror is immediately observable.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 18, 8, 8)).astype(np.float32)

    def _build(target_is_hard: bool):
        mask = np.zeros((n, POLICY_SIZE), dtype=np.uint8)
        target = np.zeros((n, POLICY_SIZE), dtype=np.float32)
        sf_idx = np.zeros((n,), dtype=np.int32)
        for i in range(n):
            legal = rng.choice(POLICY_SIZE, size=rng.integers(5, 30), replace=False)
            mask[i, legal] = 1
            pick = rng.choice(legal, size=min(3, legal.size), replace=False)
            if target_is_hard:
                sf_idx[i] = int(pick[0])
                target[i, pick[0]] = 1.0
            else:
                target[i, pick] = 1.0 / pick.size
        return mask, target, sf_idx

    own_mask, own_target, _ = _build(target_is_hard=False)
    sf_mask, sf_target, sf_idx = _build(target_is_hard=True)
    fut_mask, fut_target, _ = _build(target_is_hard=False)
    return {
        "x": x,
        "policy_target": own_target,
        "legal_mask": own_mask,
        "sf_policy_target": sf_target,
        "sf_legal_mask": sf_mask,
        "sf_move_index": sf_idx,
        "future_policy_target": fut_target,
        "future_legal_mask": fut_mask,
    }


def _assert_target_legal(target: np.ndarray, mask: np.ndarray, tag: str) -> None:
    """Every nonzero target index must fall on a mask==1 slot."""
    assert target.shape == mask.shape, f"{tag}: shape mismatch"
    illegal_mass = target * (mask == 0)
    bad_rows = np.where(illegal_mass.sum(axis=1) > 0)[0]
    assert bad_rows.size == 0, f"{tag}: rows with target mass on illegal moves: {bad_rows.tolist()[:5]}"


def test_mirror_preserves_per_head_mask_alignment():
    """Regression: after batch mirror, each head's target indices must still be
    legal under its own mirrored mask. Guards against the bug where only a
    subset of masks get mirrored (Codex adversarial review finding)."""
    batch = _build_per_head_masked_batch()

    # Sanity: pre-mirror batch is already well-aligned.
    _assert_target_legal(batch["policy_target"], batch["legal_mask"], "own pre-mirror")
    _assert_target_legal(batch["sf_policy_target"], batch["sf_legal_mask"], "sf pre-mirror")
    _assert_target_legal(batch["future_policy_target"], batch["future_legal_mask"], "future pre-mirror")

    # prob=1.0 so every row is mirrored deterministically.
    m = maybe_mirror_batch_arrays(batch, rng=np.random.default_rng(1), prob=1.0)

    _assert_target_legal(m["policy_target"], m["legal_mask"], "own mirrored")
    _assert_target_legal(m["sf_policy_target"], m["sf_legal_mask"], "sf mirrored")
    _assert_target_legal(m["future_policy_target"], m["future_legal_mask"], "future mirrored")

    # sf_move_index was mirrored; it must land on sf_legal_mask.
    n = batch["x"].shape[0]
    for i in range(n):
        assert m["sf_legal_mask"][i, int(m["sf_move_index"][i])] == 1


def test_mirror_sample_preserves_per_head_mask_alignment():
    """Same invariant for the sample-list path (mirror_sample)."""
    rng = np.random.default_rng(42)
    legal = rng.choice(POLICY_SIZE, size=20, replace=False)
    sf_legal = rng.choice(POLICY_SIZE, size=18, replace=False)
    fut_legal = rng.choice(POLICY_SIZE, size=16, replace=False)

    own_mask = np.zeros(POLICY_SIZE, dtype=np.uint8)
    own_mask[legal] = 1
    sf_mask = np.zeros(POLICY_SIZE, dtype=np.uint8)
    sf_mask[sf_legal] = 1
    fut_mask = np.zeros(POLICY_SIZE, dtype=np.uint8)
    fut_mask[fut_legal] = 1

    own = np.zeros(POLICY_SIZE, dtype=np.float32)
    own[legal[:3]] = 1.0 / 3
    sf = np.zeros(POLICY_SIZE, dtype=np.float32)
    sf[sf_legal[:4]] = 1.0 / 4
    fut = np.zeros(POLICY_SIZE, dtype=np.float32)
    fut[fut_legal[:2]] = 0.5
    sf_idx = int(sf_legal[0])

    s = ReplaySample(
        x=rng.normal(size=(18, 8, 8)).astype(np.float32),
        policy_target=own,
        wdl_target=0,
        legal_mask=own_mask,
        sf_policy_target=sf,
        sf_legal_mask=sf_mask,
        sf_move_index=sf_idx,
        future_policy_target=fut,
        future_legal_mask=fut_mask,
        has_future=True,
    )
    m = mirror_sample(s)
    assert m.sf_policy_target is not None
    assert m.sf_legal_mask is not None
    assert m.future_policy_target is not None
    assert m.future_legal_mask is not None
    assert m.sf_move_index is not None

    assert (m.policy_target * (m.legal_mask == 0)).sum() == 0
    assert (m.sf_policy_target * (m.sf_legal_mask == 0)).sum() == 0
    assert (m.future_policy_target * (m.future_legal_mask == 0)).sum() == 0
    assert int(m.sf_legal_mask[int(m.sf_move_index)]) == 1


# ---------------------------------------------------------------------------
# prior_top1_index: the generating net's raw-prior top-1 MOVE INDEX.
#
# ⚑ The whole hazard of this field is that it looks like a scalar and is not:
# it lives in policy space, so a mirrored row must have it REMAPPED. A copy
# raises nothing, changes no shape, and leaves a plausible in-range index --
# it is simply the wrong move on every mirrored row. These tests exist to fail
# on exactly that, in both mirror paths.
#
# Every entry of both mirror maps is self-distinct (mirror(i) != i for all i),
# so `expected != original` holds for any index and the assertions below cannot
# go vacuous by accidentally picking a fixed point.
# ---------------------------------------------------------------------------


def test_mirror_sample_remaps_prior_top1_index_compact():
    compact_idx = 10
    expected = int(COMPACT_MIRROR_POLICY_MAP[compact_idx])
    assert expected != compact_idx  # a copy would be indistinguishable otherwise
    policy = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float32)
    policy[compact_idx] = 1.0
    sample = ReplaySample(
        x=np.zeros((18, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=1,
        prior_top1_index=compact_idx,
        prior_top1_prob=0.75,
    )

    mirrored = mirror_sample(sample)

    assert mirrored.prior_top1_index == expected
    # The probability rides along unchanged: mass on a move is mirror-invariant.
    assert mirrored.prior_top1_prob == 0.75


def test_mirror_sample_remaps_prior_top1_index_full_width():
    full_idx = int(COMPACT_TO_FULL_POLICY[10])
    expected = mirror_policy_index(full_idx)
    assert expected != full_idx
    policy = np.zeros((POLICY_SIZE,), dtype=np.float32)
    policy[full_idx] = 1.0
    sample = ReplaySample(
        x=np.zeros((18, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=1,
        prior_top1_index=full_idx,
        prior_top1_prob=0.5,
    )

    mirrored = mirror_sample(sample)

    assert mirrored.prior_top1_index == expected


def test_mirror_sample_prior_top1_lands_on_the_mirrored_legal_move():
    """Consistency check with a co-mirrored mask.

    Stronger than comparing against the map: it re-derives the answer from a
    field mirrored by a DIFFERENT mechanism (legal_mask goes through
    mirror_policy, the index through the index remap). A copied index lands on
    an illegal square of the mirrored position and this fails.
    """
    compact_idx = 10
    policy = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float32)
    policy[compact_idx] = 1.0
    legal_mask = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.uint8)
    legal_mask[compact_idx] = 1
    sample = ReplaySample(
        x=np.zeros((18, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=1,
        legal_mask=legal_mask,
        prior_top1_index=compact_idx,
        prior_top1_prob=0.9,
    )

    mirrored = mirror_sample(sample)

    assert mirrored.legal_mask is not None
    assert mirrored.prior_top1_index is not None
    assert int(mirrored.legal_mask[int(mirrored.prior_top1_index)]) == 1
    # ...and the pre-mirror slot is no longer the legal one, so the assertion
    # above is not satisfied by a mask that simply stayed put.
    assert int(mirrored.legal_mask[compact_idx]) == 0


def test_mirror_batch_remaps_prior_top1_index():
    compact_idx = 10
    expected = int(COMPACT_MIRROR_POLICY_MAP[compact_idx])
    assert expected != compact_idx
    batch = {
        "x": np.zeros((1, 18, 8, 8), dtype=np.float32),
        "policy_target": np.zeros((1, COMPACT_POLICY_SIZE), dtype=np.float32),
        "prior_top1_index": np.array([compact_idx], dtype=np.int32),
        "prior_top1_prob": np.array([0.75], dtype=np.float16),
    }
    batch["policy_target"][0, compact_idx] = 1.0

    mirrored = maybe_mirror_batch_arrays(batch, rng=np.random.default_rng(1), prob=1.0)

    assert int(mirrored["prior_top1_index"][0]) == expected
    assert float(mirrored["prior_top1_prob"][0]) == 0.75


def test_mirror_batch_leaves_inactive_prior_top1_rows_alone():
    """A row with the has_ flag off carries -1 and must pass through untouched:
    -1 is not a move, and remapping it would index the mirror map out of range.
    """
    batch = {
        "x": np.zeros((1, 18, 8, 8), dtype=np.float32),
        "policy_target": np.zeros((1, COMPACT_POLICY_SIZE), dtype=np.float32),
        "prior_top1_index": np.array([-1], dtype=np.int32),
    }
    batch["policy_target"][0, 0] = 1.0

    mirrored = maybe_mirror_batch_arrays(batch, rng=np.random.default_rng(1), prob=1.0)

    assert int(mirrored["prior_top1_index"][0]) == -1


# Every SCALAR INTEGER optional shard field that is NOT a move index, with the
# reason. This is the exemption half of
# ``test_no_scalar_int_shard_field_escapes_the_move_index_classification``: the
# test walks ``_OPTIONAL_FIELD_SPECS`` — the one place a new field is actually
# added — and demands that each such field be EITHER registered in
# ``POLICY_INDEX_FIELDS`` (⇒ range-checked and mirror-remapped) OR listed here.
#
# ⚑ Adding a name here is a decision, not paperwork: it asserts the value is not
# an index into policy space, and therefore that mirroring it unchanged is
# correct. Get that wrong and every mirrored row carries the wrong move, with no
# exception and no shape change. If you are adding a field and are unsure, it
# belongs in POLICY_INDEX_FIELDS — a remap of a non-index is loud (it will not
# round-trip in that field's own tests), a copy of an index is silent.
_NOT_A_MOVE_INDEX: dict[str, str] = {
    "game_id": "opaque game identifier",
    "ply_index": "ply counter — '_index' in the name, not an index into policy space",
    "seed_id": "blind-spot seed identifier",
    "seed_family_id": "blind-spot seed family identifier",
    "opening_source_code": "enum of where the opening came from",
    "sf_played_rank": "1-based rank WITHIN the MultiPV list, not a move id",
    "future_sf_regret_count": "how many future plies the regret aggregate covers",
    "is_network_turn": "boolean",
    "is_selfplay": "boolean",
}


def test_no_scalar_int_shard_field_escapes_the_move_index_classification():
    """⚑ The completeness half. It walks the specs, NOT the registry.

    The forgettable step when adding a move-index field is REGISTERING it: the
    author adds an ``_OptFieldSpec``, a ``_SCALAR_FIELDS`` row and a
    ``ReplaySample`` attribute, and the value then flows to disk and back
    without ever passing through ``POLICY_INDEX_FIELDS`` — so nothing
    range-checks it and both mirror paths COPY it, which is the wrong move on
    every mirrored row.

    ``test_every_registered_policy_index_field_is_remapped_by_both_paths``
    below cannot see that, by construction: it iterates the registry, and being
    in the registry is precisely what the forgotten step would have done. It is
    a guard on the mirror paths, not on the schema. Its docstring used to claim
    the schema case; a `foo_move_index` added to ``_OPTIONAL_FIELD_SPECS`` and
    omitted from the registry left seven suites green.

    So derive the field set from the place a field is actually ADDED, and
    require every scalar integer spec to be classified exactly once: registered
    as a move index, or listed in ``_NOT_A_MOVE_INDEX`` with a reason. Adding a
    field and classifying it nowhere fails here.

    Scope note: scalar (``shape == ()``) and integer-dtyped. A move index cannot
    be a float, and the non-scalar policy-space fields are covered by
    ``POLICY_SIZED_FIELDS`` and its own guards.
    """
    from chess_anti_engine.replay.shard import (
        _OPTIONAL_FIELD_SPECS,
        POLICY_INDEX_FIELDS,
    )

    int_dtypes = {np.dtype(np.uint8), np.dtype(np.int32), np.dtype(np.int64)}
    registered = dict(POLICY_INDEX_FIELDS)
    specs = {s.arr: s for s in _OPTIONAL_FIELD_SPECS}

    scalar_ints = [
        s for s in _OPTIONAL_FIELD_SPECS if s.shape == () and s.dtype in int_dtypes
    ]
    # Non-vacuity: this test is worthless if the filter selects nothing, and it
    # must at least reach the field this guard was written for.
    assert len(scalar_ints) >= 10
    assert "prior_top1_index" in {s.arr for s in scalar_ints}

    unclassified = [
        s.arr for s in scalar_ints
        if s.arr not in registered and s.arr not in _NOT_A_MOVE_INDEX
    ]
    assert not unclassified, (
        "scalar integer shard field(s) classified neither as a move index nor as "
        f"a non-index: {unclassified}. Add each to replay/shard.POLICY_INDEX_FIELDS "
        "(if it holds a move id — it will then be range-checked by validate_arrays "
        "and mirror-remapped by both augmentation paths) or to _NOT_A_MOVE_INDEX "
        "in this file with the reason it is not one."
    )

    # Both directions: no name may be classified twice, and nothing may be
    # exempted that does not exist (a rename would otherwise leave a dead
    # exemption behind that silently re-opens the hole under the old name).
    both = sorted(set(registered) & set(_NOT_A_MOVE_INDEX))
    assert not both, f"classified as both an index and a non-index: {both}"
    stale = sorted(set(_NOT_A_MOVE_INDEX) - {s.arr for s in scalar_ints})
    assert not stale, f"_NOT_A_MOVE_INDEX names no longer in the schema: {stale}"

    # And the registry must name real specs with their real presence flags —
    # validate_arrays' range check and the mirror lookup are both keyed on this
    # pair, so a typo in either half is a check that silently never applies.
    for name, flag in registered.items():
        assert name in specs, f"POLICY_INDEX_FIELDS names a non-existent field: {name}"
        assert specs[name].flag == flag, (
            f"{name}: registry flag {flag!r} != schema flag {specs[name].flag!r}"
        )


def test_every_registered_policy_index_field_is_remapped_by_both_paths():
    """Behaviour guard on the two MIRROR PATHS, driven by the registry.

    ``replay/shard.POLICY_INDEX_FIELDS`` is the schema's statement of which
    scalars hold a move index. Anything on that list that a mirror path COPIES
    instead of remapping is a silent corruption of that field, so this drives
    the list itself and exercises both paths on every entry.

    ⚑ What it does NOT catch, and what its docstring used to claim it did: a
    field added to ``_OPTIONAL_FIELD_SPECS`` and never registered. Iterating the
    registry cannot see an absence from the registry. That case is
    ``test_no_scalar_int_shard_field_escapes_the_move_index_classification``
    above; the two are complementary and neither replaces the other.
    """
    from chess_anti_engine.replay.shard import POLICY_INDEX_FIELDS

    idx = 10
    expected = int(COMPACT_MIRROR_POLICY_MAP[idx])
    assert expected != idx

    for name, _flag in POLICY_INDEX_FIELDS:
        policy = np.zeros((COMPACT_POLICY_SIZE,), dtype=np.float32)
        policy[idx] = 1.0
        sample = ReplaySample(
            x=np.zeros((18, 8, 8), dtype=np.float32),
            policy_target=policy,
            wdl_target=1,
        )
        setattr(sample, name, idx)
        assert getattr(mirror_sample(sample), name) == expected, f"mirror_sample: {name}"

        batch = {
            "x": np.zeros((1, 18, 8, 8), dtype=np.float32),
            "policy_target": np.zeros((1, COMPACT_POLICY_SIZE), dtype=np.float32),
            name: np.array([idx], dtype=np.int32),
        }
        batch["policy_target"][0, idx] = 1.0
        out = maybe_mirror_batch_arrays(batch, rng=np.random.default_rng(1), prob=1.0)
        assert int(out[name][0]) == expected, f"maybe_mirror_batch_arrays: {name}"
