from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

def compute_extra_features(
    pieces_us: NDArray[np.uint64],
    pieces_them: NDArray[np.uint64],
    occupied: int,
    king_sq_us: int,
    king_sq_them: int,
    turn_white: bool | int,
    ep_square: int,
) -> NDArray[np.float32]: ...
