from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ReplaySample:
    x: np.ndarray  # (C,8,8) float32
    policy_target: np.ndarray  # (POLICY_SIZE,) float32 distribution
    wdl_target: int  # 0/1/2

    # Sampling priority (KataGo-style surprise weighting)
    priority: float = 1.0
    has_policy: bool = True

    # Optional auxiliary targets (for spec completeness; not all are trained yet)
    #
    # NOTE: With the "train on network turns only" scheme, SF targets (policy + eval)
    # are attached to the *network-turn* sample, representing Stockfish's reply to the
    # network's move and the evaluation after that reply.
    sf_wdl: Optional[np.ndarray] = None  # (3,) float32
    sf_move_index: Optional[int] = None  # action index for SF chosen move
    sf_policy_target: Optional[np.ndarray] = None  # (POLICY_SIZE,) float32 SF reply distribution
    moves_left: Optional[float] = None
    is_network_turn: Optional[bool] = None

    categorical_target: Optional[np.ndarray] = None  # (num_bins,) float32

    policy_soft_target: Optional[np.ndarray] = None  # (POLICY_SIZE,) float32
    future_policy_target: Optional[np.ndarray] = None  # (POLICY_SIZE,) float32
    has_future: Optional[bool] = None

    volatility_target: Optional[np.ndarray] = None  # (3,) float32
    has_volatility: Optional[bool] = None

    sf_volatility_target: Optional[np.ndarray] = None  # (3,) float32
    has_sf_volatility: Optional[bool] = None

    # LC0-style illegal move masking: 1=legal, 0=illegal, shape (POLICY_SIZE,).
    # Applied to policy logits before softmax during training to avoid wasting
    # probability mass on illegal moves. None for old shards (masking skipped).
    legal_mask: Optional[np.ndarray] = None  # (POLICY_SIZE,) bool/uint8


def balance_wdl(
    samples: list[ReplaySample],
    rng: np.random.Generator,
    *,
    max_ratio: float = 1.5,
) -> list[ReplaySample]:
    """Downsample the majority WDL class to prevent value head collapse.

    If one WDL outcome dominates (e.g. 90% losses early in training), the
    value head learns to predict that outcome everywhere, which poisons MCTS.

    This caps any single WDL class to at most ``max_ratio`` times the size of
    the smallest non-empty class. Default 1.5 keeps roughly balanced data
    while still allowing the model to see a natural skew.

    Returns a new list (does not modify the input).
    """
    if not samples:
        return samples

    buckets: dict[int, list[ReplaySample]] = {0: [], 1: [], 2: []}
    for s in samples:
        wdl = int(s.wdl_target)
        if wdl in buckets:
            buckets[wdl].append(s)

    sizes = [len(v) for v in buckets.values() if len(v) > 0]
    if len(sizes) <= 1:
        return samples  # only one class present, nothing to balance

    min_size = min(sizes)
    cap = max(1, int(min_size * max_ratio))

    out: list[ReplaySample] = []
    for wdl_class in (0, 1, 2):
        bucket = buckets[wdl_class]
        if len(bucket) <= cap:
            out.extend(bucket)
        else:
            idxs = rng.choice(len(bucket), size=cap, replace=False)
            out.extend([bucket[int(i)] for i in idxs])

    rng.shuffle(out)  # type: ignore[arg-type]
    return out


class ReplayBuffer:
    def __init__(self, capacity: int, *, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self._data: list[ReplaySample] = []
        self._pos = 0

        # Surprise weighting: 50% uniform, 50% proportional to priority
        self.surprise_mix = 0.5

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data = []
        self._pos = 0

    def add_many(self, samples: list[ReplaySample]) -> None:
        for s in samples:
            self.add(s)

    def add(self, sample: ReplaySample) -> None:
        if len(self._data) < self.capacity:
            self._data.append(sample)
        else:
            self._data[self._pos] = sample
            self._pos = (self._pos + 1) % self.capacity

    def sample_batch(self, batch_size: int, *, wdl_balance: bool = True) -> list[ReplaySample]:
        n = len(self._data)
        if n == 0:
            raise ValueError("ReplayBuffer is empty")

        bs = int(batch_size)

        if not wdl_balance:
            return self._sample_raw(bs)

        # WDL balancing (light-touch):
        # - Cap draws to avoid batches that are almost entirely drawn games.
        # - Keep win/loss roughly balanced (cap max(win,loss) <= wl_max_ratio * min(win,loss)).
        #
        # This is intentionally weaker than equalizing W/D/L. The goal is to prevent
        # value head collapse / spurious correlations (e.g. volatility => loss) while
        # preserving the natural draw rate that self-play produces.
        draw_cap_frac = 0.90
        wl_max_ratio = 1.5

        buckets: dict[int, list[int]] = {0: [], 1: [], 2: []}
        for i, s in enumerate(self._data):
            wdl = int(s.wdl_target)
            if wdl in buckets:
                buckets[wdl].append(i)

        win_idx = buckets[2]
        draw_idx = buckets[1]
        loss_idx = buckets[0]

        # Only apply balancing when we have *both* decisive outcomes available.
        # If one of win/loss is missing, we can't enforce a ratio anyway, and capping
        # draws could distort early training; fall back to original sampling.
        if len(win_idx) == 0 or len(loss_idx) == 0:
            return self._sample_raw(bs)

        def _sample_from_indices(idxs: list[int], k: int) -> list[ReplaySample]:
            if k <= 0:
                return []
            # Same surprise mix as _sample_raw, but restricted to a bucket.
            k_uni = int(round(k * (1.0 - self.surprise_mix)))
            k_pri = k - k_uni

            out_local: list[ReplaySample] = []

            if k_uni > 0:
                chosen = self.rng.choice(len(idxs), size=k_uni, replace=True)
                out_local.extend([self._data[idxs[int(i)]] for i in chosen])

            if k_pri > 0:
                pri = np.array(
                    [max(0.0, float(self._data[j].priority)) for j in idxs],
                    dtype=np.float64,
                )
                ps = float(pri.sum())
                if ps <= 0:
                    chosen = self.rng.choice(len(idxs), size=k_pri, replace=True)
                    out_local.extend([self._data[idxs[int(i)]] for i in chosen])
                else:
                    p = pri / ps
                    chosen = self.rng.choice(np.arange(len(idxs)), size=k_pri, replace=True, p=p)
                    out_local.extend([self._data[idxs[int(i)]] for i in chosen])

            return out_local

        # (1) Decide draw count based on buffer's draw rate, with a hard cap.
        p_draw = float(len(draw_idx)) / float(max(1, n))
        n_draw = int(round(bs * p_draw))
        n_draw_cap = int(np.floor(draw_cap_frac * bs))
        n_draw = min(n_draw, n_draw_cap)
        n_draw = max(0, min(bs, n_draw))

        # If there are no draws in the buffer, force n_draw=0.
        if len(draw_idx) == 0:
            n_draw = 0

        bs_decisive = bs - n_draw

        # (2) Split decisive slots into wins/losses, keeping ratio bounded.
        n_win = 0
        n_loss = 0
        if bs_decisive > 0 and (len(win_idx) > 0 or len(loss_idx) > 0):
            if len(win_idx) == 0:
                n_loss = bs_decisive
            elif len(loss_idx) == 0:
                n_win = bs_decisive
            else:
                # Start from natural decisive win-rate.
                p_win = float(len(win_idx)) / float(len(win_idx) + len(loss_idx))
                n_win = int(round(bs_decisive * p_win))
                n_win = max(0, min(bs_decisive, n_win))
                n_loss = bs_decisive - n_win

                # Enforce max ratio (oversample minority if needed).
                r = float(wl_max_ratio)
                if n_win > int(np.floor(r * n_loss)):
                    # Wins are majority.
                    n_loss = int(np.ceil(bs_decisive / (1.0 + r)))
                    n_win = bs_decisive - n_loss
                elif n_loss > int(np.floor(r * n_win)):
                    # Losses are majority.
                    n_win = int(np.ceil(bs_decisive / (1.0 + r)))
                    n_loss = bs_decisive - n_win

        out: list[ReplaySample] = []
        out.extend(_sample_from_indices(draw_idx, n_draw))
        out.extend(_sample_from_indices(win_idx, n_win))
        out.extend(_sample_from_indices(loss_idx, n_loss))

        # Any rounding issues: top up from full buffer using original sampler.
        if len(out) < bs:
            out.extend(self._sample_raw(bs - len(out)))
        elif len(out) > bs:
            out = out[:bs]

        self.rng.shuffle(out)  # type: ignore[arg-type]
        return out

    def _sample_raw(self, batch_size: int) -> list[ReplaySample]:
        """Sample without WDL balancing (original surprise-weighted method)."""
        n = len(self._data)
        bs = int(batch_size)
        k_uni = int(round(bs * (1.0 - self.surprise_mix)))
        k_pri = bs - k_uni

        out: list[ReplaySample] = []

        if k_uni > 0:
            idxs = self.rng.integers(0, n, size=k_uni)
            out.extend([self._data[int(i)] for i in idxs])

        if k_pri > 0:
            pri = np.array([max(0.0, float(s.priority)) for s in self._data], dtype=np.float64)
            ps = float(pri.sum())
            if ps <= 0:
                idxs = self.rng.integers(0, n, size=k_pri)
                out.extend([self._data[int(i)] for i in idxs])
            else:
                p = pri / ps
                idxs = self.rng.choice(np.arange(n), size=k_pri, replace=True, p=p)
                out.extend([self._data[int(i)] for i in idxs])

        return out
