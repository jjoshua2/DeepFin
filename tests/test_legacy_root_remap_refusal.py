"""The POV-lossy legacy→lc0_root remap must fail loudly, not silently.

docs/rl_loop_audit.md M12: the synthetic remap sets plane 108 (side-to-move)
to 0 unconditionally because the legacy layout stores no absolute colour, so
every converted row reads as white-to-move. It fires when legacy shards
re-enter the window — a salvage restore from a pre-switch pool — which is
exactly when nobody is watching the value head.
"""
from __future__ import annotations

import numpy as np
import pytest

from chess_anti_engine.encoding.lc0 import LC0_FULL
from chess_anti_engine.replay.sample import ReplaySample
from chess_anti_engine.train.trainer import (
    _legacy_x_to_synthetic_lc0_root,
    select_input_history_arrays,
    select_input_history_samples,
)

_SIDE_TO_MOVE_PLANE = LC0_FULL.root_metadata_base + 4  # 108


def _legacy_rows(n: int) -> np.ndarray:
    x = np.zeros((n, 146, 8, 8), dtype=np.float32)
    x[:, 0, :, :] = 1.0
    x[:, LC0_FULL.metadata_base + 5, :, :] = 1.0  # legacy "color to move": always 1
    return x


def test_synthetic_remap_loses_side_to_move() -> None:
    """The documented reason this path must not run silently."""
    out = _legacy_x_to_synthetic_lc0_root(_legacy_rows(3))
    assert np.all(out[:, _SIDE_TO_MOVE_PLANE] == 0.0)


def test_arrays_refuse_the_lossy_remap_by_default() -> None:
    with pytest.raises(ValueError, match="CANNOT recover side-to-move"):
        select_input_history_arrays(
            {"x": _legacy_rows(2)},
            input_history_encoding="lc0_root_legacy_meta",
        )


def test_arrays_convert_when_the_caller_opts_in() -> None:
    out = select_input_history_arrays(
        {"x": _legacy_rows(2)},
        input_history_encoding="lc0_root_legacy_meta",
        allow_lossy_legacy_remap=True,
    )
    assert np.asarray(out["x"]).shape == (2, 146, 8, 8)


def test_recorded_root_rows_need_no_opt_in() -> None:
    """A recorded x_lc0_root is lossless, so it must not trip the refusal."""
    legacy = _legacy_rows(2)
    recorded = np.zeros_like(legacy)
    recorded[:, _SIDE_TO_MOVE_PLANE, :, :] = 1.0

    out = select_input_history_arrays(
        {
            "x": legacy,
            "x_lc0_root": recorded,
            "has_x_lc0_root": np.array([1, 1], dtype=np.uint8),
        },
        input_history_encoding="lc0_root",
    )
    assert np.all(np.asarray(out["x"])[:, _SIDE_TO_MOVE_PLANE] == 1.0)


def test_partially_recorded_rows_still_refuse() -> None:
    legacy = _legacy_rows(2)
    recorded = np.zeros_like(legacy)

    with pytest.raises(ValueError, match="1 of 2 replay rows"):
        select_input_history_arrays(
            {
                "x": legacy,
                "x_lc0_root": recorded,
                "has_x_lc0_root": np.array([1, 0], dtype=np.uint8),
            },
            input_history_encoding="lc0_root",
        )


def test_rows_already_in_the_target_encoding_are_untouched() -> None:
    root = _legacy_rows(2)
    out = select_input_history_arrays(
        {"x": root, "_input_history_encoding": np.asarray("lc0_root")},
        input_history_encoding="lc0_root",
    )
    np.testing.assert_array_equal(np.asarray(out["x"]), root)


def _sample(x: np.ndarray) -> ReplaySample:
    return ReplaySample(
        x=x,
        policy_target=np.zeros((1858,), dtype=np.float32),
        wdl_target=1,
    )


def test_samples_refuse_the_lossy_remap_by_default() -> None:
    samples = [_sample(row) for row in _legacy_rows(2)]
    with pytest.raises(ValueError, match="CANNOT recover side-to-move"):
        select_input_history_samples(samples, input_history_encoding="lc0_root")


def test_samples_convert_when_the_caller_opts_in() -> None:
    samples = [_sample(row) for row in _legacy_rows(2)]
    out = select_input_history_samples(
        samples,
        input_history_encoding="lc0_root",
        allow_lossy_legacy_remap=True,
    )
    assert all(s.input_history_encoding == "lc0_root" for s in out)
