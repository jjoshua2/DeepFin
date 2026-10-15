"""Pure sampling calculations shared by the memory and disk replay buffers."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def priority_probabilities(priority: np.ndarray) -> np.ndarray | None:
    """Normalize a private float64 copy, or request uniform sampling.

    Nonfinite and negative priorities contribute zero. A zero or overflowing
    total retains the callers' uniform fallback without changing their RNG use.
    """
    pri = priority.astype(np.float64, copy=True)
    np.nan_to_num(pri, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.maximum(pri, 0.0, out=pri)
    total = float(pri.sum())
    if total <= 0.0 or not np.isfinite(total):
        return None
    return pri / total


def wdl_batch_counts(
    batch_size: int,
    *,
    size: int,
    wins: int,
    draws: int,
    losses: int,
    draw_cap_frac: float,
    wl_max_ratio: float,
) -> tuple[int, int, int]:
    """Return draw, win and loss quotas when both decisive classes exist.

    Keep the total pool size as the draw denominator, including unknown labels.
    Python's tie-to-even round and the existing floor/ceil ratio correction are
    part of the sampling contract, including for very small batches.
    """
    p_draw = float(draws) / float(max(1, size))
    n_draw = round(batch_size * p_draw)
    n_draw = min(n_draw, int(np.floor(draw_cap_frac * batch_size)))
    n_draw = max(0, min(batch_size, n_draw))

    bs_decisive = batch_size - n_draw
    n_win = 0
    n_loss = 0
    if bs_decisive > 0:
        p_win = float(wins) / float(wins + losses)
        n_win = round(bs_decisive * p_win)
        n_win = max(0, min(bs_decisive, n_win))
        n_loss = bs_decisive - n_win
        r = wl_max_ratio
        if n_win > int(np.floor(r * n_loss)):
            n_loss = int(np.ceil(bs_decisive / (1.0 + r)))
            n_win = bs_decisive - n_loss
        elif n_loss > int(np.floor(r * n_win)):
            n_win = int(np.ceil(bs_decisive / (1.0 + r)))
            n_loss = bs_decisive - n_win
    return n_draw, n_win, n_loss


def sample_pool_indices(
    pool: np.ndarray, k: int, *, rng: np.random.Generator,
    surprise_mix: float, priority: Callable[[], np.ndarray],
) -> np.ndarray:
    """Sample uniform then weighted rows, reading priorities only when needed."""
    if k <= 0 or pool.size == 0:
        return np.zeros((0,), dtype=np.int64)
    pool = np.asarray(pool, dtype=np.int64)
    k_uni = round(k * (1.0 - surprise_mix))
    k_pri = k - k_uni
    picks: list[np.ndarray] = []
    if k_uni > 0:
        chosen = rng.choice(pool.shape[0], size=k_uni, replace=True)
        picks.append(pool[np.asarray(chosen, dtype=np.int64)])
    if k_pri > 0:
        p = priority_probabilities(priority()[pool])
        chosen = rng.choice(pool.shape[0], size=k_pri, replace=True, p=p)
        picks.append(pool[np.asarray(chosen, dtype=np.int64)])
    if not picks:
        return np.zeros((0,), dtype=np.int64)
    if len(picks) == 1:
        return picks[0]
    return np.concatenate(picks, axis=0)
