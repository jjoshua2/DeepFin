from __future__ import annotations

import numpy as np

from chess_anti_engine.replay.shard import INPUT_HISTORY_ENCODING_ARRAY_KEY, POLICY_ENCODING_ARRAY_KEY
from chess_anti_engine.tune._utils import concat_array_batches, slice_array_batch


def _minimal_batch(n: int) -> dict[str, np.ndarray]:
    policy = np.zeros((n, 4672), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "x": np.zeros((n, 146, 8, 8), dtype=np.float32),
        "policy_target": policy,
        "wdl_target": np.ones((n,), dtype=np.int8),
    }


def test_concat_array_batches_preserves_optional_fields_across_mixed_schema():
    full = _minimal_batch(2)
    full["sf_wdl"] = np.array(
        [[0.2, 0.7, 0.1], [0.4, 0.5, 0.1]], dtype=np.float32,
    )
    full["has_sf_wdl"] = np.ones((2,), dtype=np.uint8)
    minimal = _minimal_batch(3)

    out = concat_array_batches([full, minimal])

    assert out["x"].shape[0] == 5
    assert out["sf_wdl"].shape == (5, 3)
    assert out["has_sf_wdl"].tolist() == [1, 1, 0, 0, 0]
    np.testing.assert_allclose(out["sf_wdl"][:2], full["sf_wdl"])
    np.testing.assert_allclose(out["sf_wdl"][2:], 0.0)


def test_concat_array_batches_synthesizes_legacy_required_defaults():
    batch = _minimal_batch(2)

    out = concat_array_batches([batch])

    assert out["priority"].tolist() == [1.0, 1.0]
    assert out["has_policy"].tolist() == [1, 1]


def test_slice_array_batch_preserves_scalar_metadata():
    batch = _minimal_batch(3)
    batch["_policy_size"] = np.array(4672, dtype=np.int32)

    out = slice_array_batch(batch, np.array([2, 0], dtype=np.int64))

    assert out["x"].shape[0] == 2
    assert out["_policy_size"].shape == ()
    assert int(out["_policy_size"]) == 4672


def test_concat_array_batches_preserves_matching_history_metadata():
    a = _minimal_batch(2)
    b = _minimal_batch(3)
    a[INPUT_HISTORY_ENCODING_ARRAY_KEY] = np.asarray("lc0_root")
    b[INPUT_HISTORY_ENCODING_ARRAY_KEY] = np.asarray("lc0_root")

    out = concat_array_batches([a, b])

    assert out[INPUT_HISTORY_ENCODING_ARRAY_KEY].shape == ()
    assert str(out[INPUT_HISTORY_ENCODING_ARRAY_KEY]) == "lc0_root"


def test_concat_array_batches_rejects_missing_and_concrete_history_metadata():
    tagged = _minimal_batch(2)
    tagged[INPUT_HISTORY_ENCODING_ARRAY_KEY] = np.asarray("lc0_root")

    with np.testing.assert_raises_regex(ValueError, "mixed replay metadata"):
        concat_array_batches([tagged, _minimal_batch(3)])


def test_concat_array_batches_preserves_matching_policy_metadata():
    a = _minimal_batch(2)
    b = _minimal_batch(3)
    a[POLICY_ENCODING_ARRAY_KEY] = np.asarray("az_4672")
    b[POLICY_ENCODING_ARRAY_KEY] = np.asarray("az_4672")

    out = concat_array_batches([a, b])

    assert out[POLICY_ENCODING_ARRAY_KEY].shape == ()
    assert str(out[POLICY_ENCODING_ARRAY_KEY]) == "az_4672"


def test_concat_array_batches_rejects_missing_and_concrete_policy_metadata():
    tagged = _minimal_batch(2)
    tagged[POLICY_ENCODING_ARRAY_KEY] = np.asarray("az_4672")

    with np.testing.assert_raises_regex(ValueError, "mixed replay metadata"):
        concat_array_batches([tagged, _minimal_batch(3)])
