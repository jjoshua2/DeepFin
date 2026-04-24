import numpy as np

from chess_anti_engine.moves.encode import (
    MIRROR_POLICY_MAP,
    POLICY_SIZE,
    mirror_policy_index,
)
from chess_anti_engine.replay.augment import maybe_mirror_batch_arrays, mirror_sample
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
        has_policy=True,
        sf_wdl=np.array([0.2, 0.3, 0.5], dtype=np.float32),
        sf_move_index=int(rng.integers(0, POLICY_SIZE)),
        moves_left=0.25,
        is_network_turn=True,
        categorical_target=rng.random(size=(32,)).astype(np.float32),
        policy_soft_target=ps,
        future_policy_target=fp,
        has_future=True,
        volatility_target=np.array([0.01, 0.02, 0.03], dtype=np.float32),
        has_volatility=True,
        sf_volatility_target=np.array([0.04, 0.05, 0.06], dtype=np.float32),
        has_sf_volatility=True,
    )

    s2 = mirror_sample(mirror_sample(s))

    assert np.array_equal(s2.x, s.x)
    assert np.allclose(s2.policy_target, s.policy_target)
    assert s2.wdl_target == s.wdl_target
    assert float(s2.priority) == float(s.priority)
    assert bool(s2.has_policy) == bool(s.has_policy)

    # Narrow the optional fields — the test set all of them, mirror_sample must
    # round-trip all of them. Plain asserts both satisfy pyright and catch a
    # future silent-None regression in mirror_sample. Per-reference (not a
    # loop) because pyright narrowing is per-expression, not transitive.
    assert (s.sf_wdl is not None and s.sf_move_index is not None
            and s.moves_left is not None and s.is_network_turn is not None
            and s.categorical_target is not None and s.policy_soft_target is not None
            and s.future_policy_target is not None and s.volatility_target is not None
            and s.sf_volatility_target is not None)
    assert (s2.sf_wdl is not None and s2.sf_move_index is not None
            and s2.moves_left is not None and s2.is_network_turn is not None
            and s2.categorical_target is not None and s2.policy_soft_target is not None
            and s2.future_policy_target is not None and s2.volatility_target is not None
            and s2.sf_volatility_target is not None)

    assert np.allclose(s2.sf_wdl, s.sf_wdl)
    assert int(s2.sf_move_index) == int(s.sf_move_index)
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

    batch = {
        "x": x,
        "policy_target": policy,
        "sf_policy_target": policy.copy(),
        "policy_soft_target": soft,
        "future_policy_target": soft.copy(),
        "legal_mask": legal_mask,
        "sf_move_index": sf_move_index,
    }

    mirrored = maybe_mirror_batch_arrays(batch, rng=np.random.default_rng(1), prob=1.0)
    unmirrored = maybe_mirror_batch_arrays(mirrored, rng=np.random.default_rng(2), prob=1.0)

    for key, value in batch.items():
        assert np.array_equal(unmirrored[key], value)


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
    assert m.sf_policy_target is not None and m.sf_legal_mask is not None
    assert m.future_policy_target is not None and m.future_legal_mask is not None
    assert m.sf_move_index is not None

    assert (m.policy_target * (m.legal_mask == 0)).sum() == 0
    assert (m.sf_policy_target * (m.sf_legal_mask == 0)).sum() == 0
    assert (m.future_policy_target * (m.future_legal_mask == 0)).sum() == 0
    assert int(m.sf_legal_mask[int(m.sf_move_index)]) == 1
