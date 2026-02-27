from __future__ import annotations

from typing import Optional

import numpy as np
import chess

from .lc0 import encode_lc0_full, encode_lc0_reduced
from .features import extra_feature_planes


def encode_position(
    board: chess.Board,
    *,
    add_features: bool = True,
    feature_dropout_p: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    use_full_lc0: bool = True,
) -> np.ndarray:
    """Encode a position into (C, 8, 8) float32.

    By default this produces the spec target shape:
    - LC0 full 112-plane input (8 history steps)
    - plus 34 additional classical feature planes
    => total 146 planes.

    `use_full_lc0=False` keeps the earlier reduced encoder for debugging.

    Parameters
    - feature_dropout_p: probability of zeroing ALL extra feature planes.
    """
    base = encode_lc0_full(board) if use_full_lc0 else encode_lc0_reduced(board)

    if not add_features:
        return base

    feats = extra_feature_planes(board)
    feat_arr = np.stack(feats, axis=0).astype(np.float32, copy=False)

    if feature_dropout_p > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        if float(rng.random()) < float(feature_dropout_p):
            feat_arr[...] = 0.0

    return np.concatenate([base, feat_arr], axis=0)
