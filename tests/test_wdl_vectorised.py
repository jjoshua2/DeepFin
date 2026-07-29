"""`cp_to_wdl` / `mate_to_effective_cp` vs their vectorised twins.

The batch SF-target rebuild runs the array versions on every MultiPV row of
every sampled batch, so a divergence would silently retarget training. The
scalar functions are the definition; these pin the twins to them BITWISE,
including the float32 cast order (`cp_to_wdl` casts the triple to float32 and
only then divides by the float64 total).
"""
from __future__ import annotations

import numpy as np
import pytest

from chess_anti_engine.stockfish.wdl import (
    cp_to_wdl,
    cp_to_wdl_array,
    mate_to_effective_cp,
    mate_to_effective_cp_array,
)

_MATES = [-120, -50, -49, -7, -1, 0, 1, 7, 49, 50, 120]
_CPS = [-32000, -5000, -1000, -121, -120, -119, -1, 0, 1, 119, 120, 121, 1000, 32000]


def test_mate_to_effective_cp_array_matches_scalar() -> None:
    mates = np.array(_MATES, dtype=np.int64)
    got = mate_to_effective_cp_array(mates)
    want = np.array([mate_to_effective_cp(int(m)) for m in mates], dtype=np.float64)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize(("slope", "draw"), [(0.0060, 120.0), (0.010, 60.0), (0.03, 0.0)])
def test_cp_to_wdl_array_matches_scalar_on_cp(slope: float, draw: float) -> None:
    eff = np.array(_CPS, dtype=np.float64)
    got = cp_to_wdl_array(eff, slope=slope, draw_width_cp=draw)
    want = np.stack([
        cp_to_wdl(float(c), None, slope=slope, draw_width_cp=draw) for c in eff
    ])
    assert got.dtype == want.dtype == np.float32
    np.testing.assert_array_equal(got, want)


def test_cp_to_wdl_array_matches_scalar_on_mate() -> None:
    mates = np.array(_MATES, dtype=np.int64)
    eff = mate_to_effective_cp_array(mates)
    got = cp_to_wdl_array(eff, slope=0.0060, draw_width_cp=120.0)
    want = np.stack([
        cp_to_wdl(None, int(m), slope=0.0060, draw_width_cp=120.0) for m in mates
    ])
    np.testing.assert_array_equal(got, want)


def test_cp_to_wdl_array_preserves_shape_and_validates() -> None:
    eff = np.zeros((3, 4), dtype=np.float64)
    assert cp_to_wdl_array(eff, slope=0.01, draw_width_cp=60.0).shape == (3, 4, 3)
    with pytest.raises(ValueError, match="slope>0"):
        cp_to_wdl_array(eff, slope=0.0, draw_width_cp=60.0)
    with pytest.raises(ValueError, match="draw_width_cp>=0"):
        cp_to_wdl_array(eff, slope=0.01, draw_width_cp=-1.0)
