from __future__ import annotations

import numpy as np

from chess_anti_engine.tune._utils import concat_array_batches


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
