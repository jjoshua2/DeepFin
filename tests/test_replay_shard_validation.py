import numpy as np
import pytest

from chess_anti_engine.replay.shard import validate_arrays


def test_validate_rejects_wrong_policy_size():
    arrs = {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "policy_target": np.ones((2, 10), dtype=np.float32),
        "wdl_target": np.array([0, 1], dtype=np.int8),
    }
    try:
        validate_arrays(arrs)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "A mismatch" in str(e)


def test_validate_rejects_negative_policy():
    arrs = {
        "x": np.zeros((1, 146, 8, 8), dtype=np.float32),
        "policy_target": np.zeros((1, 4672), dtype=np.float32),
        "wdl_target": np.array([1], dtype=np.int8),
    }
    arrs["policy_target"][0, 0] = -0.1
    try:
        validate_arrays(arrs)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "negative" in str(e)


def _minimal_valid_arrays() -> dict[str, np.ndarray]:
    policy = np.zeros((2, 4672), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "policy_target": policy,
        "wdl_target": np.array([0, 1], dtype=np.int8),
    }


def test_validate_rejects_present_optional_flag_without_value():
    arrs = _minimal_valid_arrays()
    arrs["has_sf_wdl"] = np.array([1, 0], dtype=np.uint8)

    with pytest.raises(ValueError, match="has_sf_wdl.*sf_wdl"):
        validate_arrays(arrs)


def test_validate_rejects_optional_value_shape_mismatch():
    arrs = _minimal_valid_arrays()
    arrs["has_search_wdl"] = np.array([1, 1], dtype=np.uint8)
    arrs["search_wdl"] = np.zeros((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="search_wdl shape mismatch"):
        validate_arrays(arrs)


def test_validate_rejects_active_zero_optional_distribution():
    arrs = _minimal_valid_arrays()
    arrs["has_search_wdl"] = np.array([1, 0], dtype=np.uint8)
    arrs["search_wdl"] = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="search_wdl active rows have non-positive sum"):
        validate_arrays(arrs)


def test_validate_rejects_active_negative_optional_distribution():
    arrs = _minimal_valid_arrays()
    arrs["has_policy_soft"] = np.array([1, 0], dtype=np.uint8)
    arrs["policy_soft_target"] = np.zeros((2, 4672), dtype=np.float32)
    arrs["policy_soft_target"][0, 0] = -0.1
    arrs["policy_soft_target"][0, 1] = 1.1

    with pytest.raises(ValueError, match="policy_soft_target active rows contain negative values"):
        validate_arrays(arrs)


def test_validate_allows_missing_optional_value_when_flag_is_absent():
    arrs = _minimal_valid_arrays()
    arrs["has_sf_wdl"] = np.array([0, 0], dtype=np.uint8)

    validate_arrays(arrs)
