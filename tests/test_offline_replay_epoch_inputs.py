from __future__ import annotations

import numpy as np

from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY, FULL_TO_COMPACT_POLICY
from scripts.offline_replay_epoch import (
    _concat,
    _convert_policy_targets,
    _select_configured_input_history,
    _select_recorded_input_history,
)


def test_lc0_root_legacy_meta_copies_rule50_and_ep_planes() -> None:
    legacy = np.zeros((2, 146, 8, 8), dtype=np.float32)
    root = np.zeros((2, 146, 8, 8), dtype=np.float32)

    legacy[:, 100, :, 4] = 1.0
    legacy[0, 102, :, :] = 0.37
    legacy[1, 102, :, :] = 0.82
    root[:, 109, :, :] = 37.0
    root[:, 110, :, :] = 0.0
    root[:, 108, :, :] = 1.0

    out = _select_recorded_input_history(
        {
            "x": legacy,
            "x_lc0_root": root,
            "has_x_lc0_root": np.ones((2,), dtype=np.uint8),
        },
        input_history_encoding="lc0_root",
        prefer_recorded_lc0_root=True,
        lc0_root_legacy_meta=True,
    )

    np.testing.assert_allclose(out["x"][:, 109], legacy[:, 102])
    np.testing.assert_array_equal(out["x"][:, 110], legacy[:, 100])
    np.testing.assert_array_equal(out["x"][:, 108], root[:, 108])


def test_lc0_root_legacy_meta_is_opt_in() -> None:
    legacy = np.zeros((1, 146, 8, 8), dtype=np.float32)
    root = np.zeros((1, 146, 8, 8), dtype=np.float32)
    legacy[:, 102, :, :] = 0.5
    root[:, 109, :, :] = 50.0

    out = _select_recorded_input_history(
        {
            "x": legacy,
            "x_lc0_root": root,
            "has_x_lc0_root": np.ones((1,), dtype=np.uint8),
        },
        input_history_encoding="lc0_root",
        prefer_recorded_lc0_root=True,
        lc0_root_legacy_meta=False,
    )

    np.testing.assert_array_equal(out["x"], root)


def test_configured_lc0_root_uses_recorded_planes_without_extra_flags() -> None:
    legacy = np.zeros((1, 146, 8, 8), dtype=np.float32)
    root = np.ones((1, 146, 8, 8), dtype=np.float32)

    out = _select_configured_input_history(
        {
            "x": legacy,
            "x_lc0_root": root,
            "has_x_lc0_root": np.ones((1,), dtype=np.uint8),
        },
        input_history_encoding="lc0_root",
        prefer_recorded_lc0_root=False,
        synthetic_lc0_root_history=False,
        lc0_root_legacy_meta=False,
    )

    np.testing.assert_array_equal(out["x"], root)


def test_convert_policy_targets_preserves_compact_sf_move_indices() -> None:
    arrs = {
        "policy_target": np.zeros((2, 1858), dtype=np.float32),
        "_policy_size": np.array(1858, dtype=np.int32),
        "sf_move_index": np.array([10, 1000], dtype=np.int64),
        "has_sf_move": np.ones((2,), dtype=np.float32),
    }

    out = _convert_policy_targets(arrs, policy_encoding="lc0_1858")

    np.testing.assert_array_equal(out["sf_move_index"], arrs["sf_move_index"])
    np.testing.assert_array_equal(out["has_sf_move"], arrs["has_sf_move"])


def test_convert_policy_targets_keeps_compact_legal_masks_binary() -> None:
    mask = np.zeros((1, 4672), dtype=np.uint8)
    mask[0, COMPACT_TO_FULL_POLICY[[2, 5, 11]]] = 1
    arrs = {
        "policy_target": mask.astype(np.float32),
        "legal_mask": mask,
        "_policy_size": np.array(4672, dtype=np.int32),
    }

    out = _convert_policy_targets(arrs, policy_encoding="lc0_1858")

    assert out["legal_mask"].dtype == np.uint8
    assert set(np.unique(out["legal_mask"]).tolist()) == {0, 1}
    assert int(out["legal_mask"].sum()) == 3


def test_convert_policy_targets_expands_compact_targets_for_az() -> None:
    compact_policy = np.zeros((2, 1858), dtype=np.float32)
    compact_policy[0, 10] = 1.0
    compact_policy[1, 1000] = 1.0
    compact_mask = (compact_policy > 0).astype(np.uint8)
    arrs = {
        "policy_target": compact_policy,
        "legal_mask": compact_mask,
        "_policy_size": np.array(1858, dtype=np.int32),
        "sf_move_index": np.array([10, 1000], dtype=np.int64),
        "has_sf_move": np.ones((2,), dtype=np.float32),
    }

    out = _convert_policy_targets(arrs, policy_encoding="az_4672")

    assert out["policy_target"].shape == (2, 4672)
    assert out["legal_mask"].shape == (2, 4672)
    assert int(out["legal_mask"].sum()) == 2
    np.testing.assert_array_equal(
        out["sf_move_index"],
        COMPACT_TO_FULL_POLICY[np.asarray(arrs["sf_move_index"], dtype=np.int64)],
    )
    np.testing.assert_array_equal(
        FULL_TO_COMPACT_POLICY[out["sf_move_index"]],
        arrs["sf_move_index"],
    )


def test_concat_preserves_optional_eval_targets_when_some_chunks_lack_them() -> None:
    policy_a = np.zeros((1, 1858), dtype=np.float32)
    policy_b = np.zeros((2, 1858), dtype=np.float32)
    full = {
        "x": np.zeros((1, 146, 8, 8), dtype=np.float32),
        "policy_target": policy_a,
        "wdl_target": np.zeros((1,), dtype=np.int8),
        "priority": np.ones((1,), dtype=np.float32),
        "has_policy": np.ones((1,), dtype=np.uint8),
        "sf_wdl": np.ones((1, 3), dtype=np.float16),
        "has_sf_wdl": np.ones((1,), dtype=np.uint8),
        "future_policy_target": policy_a.copy(),
        "has_future": np.ones((1,), dtype=np.uint8),
        "_policy_size": np.array(1858, dtype=np.int32),
    }
    minimal = {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "policy_target": policy_b,
        "wdl_target": np.zeros((2,), dtype=np.int8),
        "priority": np.ones((2,), dtype=np.float32),
        "has_policy": np.ones((2,), dtype=np.uint8),
        "_policy_size": np.array(1858, dtype=np.int32),
    }

    out = _concat([full, minimal])

    assert out["sf_wdl"].shape == (3, 3)
    assert out["future_policy_target"].shape == (3, 1858)
    np.testing.assert_array_equal(out["has_sf_wdl"], np.array([1, 0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(out["has_future"], np.array([1, 0, 0], dtype=np.uint8))
    assert int(out["_policy_size"].item()) == 1858
