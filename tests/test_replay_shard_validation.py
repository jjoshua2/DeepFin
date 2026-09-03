import numpy as np
import pytest

from chess_anti_engine.replay.shard import (
    POLICY_ENCODING_ARRAY_KEY,
    validate_active_optional_values_present,
    validate_array_declarations,
    validate_arrays,
)


class _DeclaredArray:
    def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.shape = shape
        self.dtype = dtype

    def __array__(self):
        raise AssertionError("declaration validation should not materialize arrays")


def test_validate_rejects_wrong_policy_size():
    arrs = {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "policy_target": np.ones((2, 10), dtype=np.float32),
        "wdl_target": np.array([0, 1], dtype=np.int8),
    }
    with pytest.raises(ValueError, match="A mismatch"):
        validate_arrays(arrs)


def test_validate_rejects_negative_policy():
    arrs = {
        "x": np.zeros((1, 146, 8, 8), dtype=np.float32),
        "policy_target": np.zeros((1, 4672), dtype=np.float32),
        "wdl_target": np.array([1], dtype=np.int8),
    }
    arrs["policy_target"][0, 0] = -0.1
    with pytest.raises(ValueError, match="negative"):
        validate_arrays(arrs)


def _minimal_valid_arrays() -> dict[str, np.ndarray]:
    policy = np.zeros((2, 4672), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "policy_target": policy,
        "wdl_target": np.array([0, 1], dtype=np.int8),
    }


def test_validate_accepts_compact_policy_size():
    policy = np.zeros((2, 1858), dtype=np.float32)
    policy[:, 0] = 1.0
    validate_arrays({
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "policy_target": policy,
        "wdl_target": np.array([0, 1], dtype=np.int8),
    })


def test_validate_rejects_policy_size_attr_mismatch():
    arrs = _minimal_valid_arrays()
    arrs["_policy_size"] = np.array(999_999, dtype=np.int32)

    with pytest.raises(ValueError, match="_policy_size mismatch"):
        validate_arrays(arrs)


def test_validate_rejects_policy_encoding_width_mismatch():
    arrs = _minimal_valid_arrays()
    arrs[POLICY_ENCODING_ARRAY_KEY] = np.asarray("lc0_1858")

    with pytest.raises(ValueError, match="_policy_encoding"):
        validate_arrays(arrs)


def test_validate_rejects_policy_index_out_of_declared_range():
    arrs = _minimal_valid_arrays()
    arrs["sf_move_index"] = np.array([4672, 999999], dtype=np.int32)
    arrs["has_sf_move"] = np.array([1, 0], dtype=np.uint8)

    with pytest.raises(ValueError, match="sf_move_index active rows out of range"):
        validate_arrays(arrs)


def test_validate_declarations_rejects_huge_lazy_shard_without_materializing():
    arrs = {
        "x": _DeclaredArray((100_001, 146, 8, 8), np.dtype(np.float16)),
        "policy_target": _DeclaredArray((100_001, 4672), np.dtype(np.float16)),
        "wdl_target": _DeclaredArray((100_001,), np.dtype(np.int8)),
        "has_moves_left": _DeclaredArray((100_001,), np.dtype(np.uint8)),
    }

    with pytest.raises(ValueError, match="too many positions"):
        validate_array_declarations(arrs, max_positions=50_000)


def test_validate_declarations_rejects_uncompressed_size_without_materializing():
    arrs = {
        "x": _DeclaredArray((10, 146, 8, 8), np.dtype(np.float16)),
        "policy_target": _DeclaredArray((10, 4672), np.dtype(np.float16)),
        "wdl_target": _DeclaredArray((10,), np.dtype(np.int8)),
    }

    with pytest.raises(ValueError, match="uncompressed arrays too large"):
        validate_array_declarations(arrs, max_uncompressed_bytes=1)


def test_validate_rejects_present_optional_flag_without_value():
    arrs = _minimal_valid_arrays()
    arrs["has_sf_wdl"] = np.array([1, 0], dtype=np.uint8)

    with pytest.raises(ValueError, match=r"has_sf_wdl.*sf_wdl"):
        validate_arrays(arrs)


def test_validate_active_optional_values_rejects_flag_without_value():
    arrs = _minimal_valid_arrays()
    arrs["has_moves_left"] = np.array([1, 0], dtype=np.uint8)

    with pytest.raises(ValueError, match=r"has_moves_left.*moves_left"):
        validate_active_optional_values_present(arrs)


def test_validate_active_optional_values_allows_inactive_flag_without_value():
    arrs = _minimal_valid_arrays()
    arrs["has_moves_left"] = np.zeros(2, dtype=np.uint8)

    validate_active_optional_values_present(arrs)


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
