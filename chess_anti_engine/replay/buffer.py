from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
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
    sf_wdl: np.ndarray | None = None  # (3,) float32
    sf_move_index: int | None = None  # action index for SF chosen move
    sf_policy_target: np.ndarray | None = None  # (POLICY_SIZE,) float32 SF reply distribution
    moves_left: float | None = None
    is_network_turn: bool | None = None
    is_selfplay: bool | None = None

    categorical_target: np.ndarray | None = None  # (num_bins,) float32

    policy_soft_target: np.ndarray | None = None  # (POLICY_SIZE,) float32
    future_policy_target: np.ndarray | None = None  # (POLICY_SIZE,) float32
    has_future: bool | None = None

    volatility_target: np.ndarray | None = None  # (3,) float32
    has_volatility: bool | None = None

    sf_volatility_target: np.ndarray | None = None  # (3,) float32
    has_sf_volatility: bool | None = None

  # LC0-style illegal move masking: 1=legal, 0=illegal, shape (POLICY_SIZE,).
  # Applied to policy logits before softmax during training to avoid wasting
  # probability mass on illegal moves. None for old shards (masking skipped).
    legal_mask: np.ndarray | None = None  # (POLICY_SIZE,) bool/uint8 — legal at t, net POV
  # Legal mask at t+1 (opponent POV) for policy_sf head.
    sf_legal_mask: np.ndarray | None = None
  # Legal mask at t+2 (net POV, next own move) for policy_future head.
    future_legal_mask: np.ndarray | None = None


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

    rng.shuffle(out)  # type: ignore[arg-type] # numpy expects ArrayLike; list[dataclass] works at runtime
    return out


class ReplayBuffer:
    def __init__(self, capacity: int, *, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self._array = ArrayReplayBuffer(self.capacity, rng=rng)

    @property
    def surprise_mix(self) -> float:
        return float(self._array.surprise_mix)

    @surprise_mix.setter
    def surprise_mix(self, value: float) -> None:
        self._array.surprise_mix = float(value)

    def __len__(self) -> int:
        return len(self._array)

    def clear(self) -> None:
        self._array.clear()

    def add_many(self, samples: list[ReplaySample]) -> None:
        self._array.add_many(samples)

    def add_many_arrays(self, arrs: dict[str, np.ndarray]) -> None:
        self._array.add_many_arrays(arrs)

    def add(self, sample: ReplaySample) -> None:
        self._array.add(sample)

    def sample_batch_arrays(self, batch_size: int, *, wdl_balance: bool = True) -> dict[str, np.ndarray]:
        return self._array.sample_batch_arrays(batch_size, wdl_balance=wdl_balance)

    def sample_batch(self, batch_size: int, *, wdl_balance: bool = True) -> list[ReplaySample]:
        from .shard import arrays_to_samples

        return arrays_to_samples(self.sample_batch_arrays(batch_size, wdl_balance=wdl_balance))


class ArrayReplayBuffer:
    def __init__(self, capacity: int, *, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self._chunks: deque[dict[str, np.ndarray]] = deque()
        self._chunk_sizes: deque[int] = deque()
        self._size = 0
        self._priority = np.zeros((0,), dtype=np.float32)
        self._wdl = np.zeros((0,), dtype=np.int8)
        self.surprise_mix = 0.5

    def __len__(self) -> int:
        return int(self._size)

    def clear(self) -> None:
        self._chunks = deque()
        self._chunk_sizes = deque()
        self._size = 0
        self._priority = np.zeros((0,), dtype=np.float32)
        self._wdl = np.zeros((0,), dtype=np.int8)

    def _append_arrays(self, arrs: dict[str, np.ndarray]) -> None:
        from .shard import sparsify_chunk
        n = int(np.asarray(arrs["x"]).shape[0])
        if n <= 0:
            return
        self._chunks.append(sparsify_chunk({k: np.asarray(v) for k, v in arrs.items()}))
        self._chunk_sizes.append(n)
        self._size += n
        self._priority = np.concatenate([self._priority, np.asarray(arrs["priority"], dtype=np.float32)], axis=0)
        self._wdl = np.concatenate([self._wdl, np.asarray(arrs["wdl_target"], dtype=np.int8)], axis=0)

    def _drop_oldest(self, count: int) -> None:
        drop = min(max(0, int(count)), self._size)
        if drop <= 0:
            return
        remaining = drop
        while remaining > 0 and self._chunks:
            first_n = self._chunk_sizes[0]
            if remaining >= first_n:
                self._chunks.popleft()
                self._chunk_sizes.popleft()
                self._size -= first_n
                remaining -= first_n
                continue
            chunk = self._chunks[0]
            for k in tuple(chunk.keys()):
                chunk[k] = chunk[k][remaining:]
            self._chunk_sizes[0] = first_n - remaining
            self._size -= remaining
            remaining = 0
        self._priority = self._priority[drop:]
        self._wdl = self._wdl[drop:]

    def _enforce_capacity(self) -> None:
        overflow = self._size - self.capacity
        if overflow > 0:
            self._drop_oldest(overflow)

    def add(self, sample: ReplaySample) -> None:
        self.add_many([sample])

    def add_many(self, samples: list[ReplaySample]) -> None:
        from .shard import prune_storage_arrays, samples_to_arrays

        if not samples:
            return
        self.add_many_arrays(prune_storage_arrays(samples_to_arrays(samples)))

    def add_many_arrays(self, arrs: dict[str, np.ndarray]) -> None:
        from .shard import prune_storage_arrays

        self._append_arrays(prune_storage_arrays(arrs))
        self._enforce_capacity()

    def _sample_indices(self, pool: np.ndarray, k: int) -> np.ndarray:
        if k <= 0 or pool.size == 0:
            return np.zeros((0,), dtype=np.int64)
        pool = np.asarray(pool, dtype=np.int64)
        k_uni = int(round(k * (1.0 - self.surprise_mix)))
        k_pri = k - k_uni
        picks: list[np.ndarray] = []
        if k_uni > 0:
            chosen = self.rng.choice(pool.shape[0], size=k_uni, replace=True)
            picks.append(pool[np.asarray(chosen, dtype=np.int64)])
        if k_pri > 0:
            pri = self._priority[pool].astype(np.float64, copy=True)
            np.nan_to_num(pri, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.maximum(pri, 0.0, out=pri)
            ps = float(pri.sum())
            if ps <= 0.0 or not np.isfinite(ps):
                chosen = self.rng.choice(pool.shape[0], size=k_pri, replace=True)
                picks.append(pool[np.asarray(chosen, dtype=np.int64)])
            else:
                p = pri / ps
                chosen = self.rng.choice(np.arange(pool.shape[0]), size=k_pri, replace=True, p=p)
                picks.append(pool[np.asarray(chosen, dtype=np.int64)])
        if not picks:
            return np.zeros((0,), dtype=np.int64)
        if len(picks) == 1:
            return picks[0]
        return np.concatenate(picks, axis=0)

    def _sample_all_indices(self, batch_size: int) -> np.ndarray:
        n = int(self._size)
        bs = int(batch_size)
        if bs <= 0 or n <= 0:
            return np.zeros((0,), dtype=np.int64)
        k_uni = int(round(bs * (1.0 - self.surprise_mix)))
        k_pri = bs - k_uni
        picks: list[np.ndarray] = []
        if k_uni > 0:
            picks.append(np.asarray(self.rng.integers(0, n, size=k_uni), dtype=np.int64))
        if k_pri > 0:
            pri = self._priority.astype(np.float64, copy=True)
            np.nan_to_num(pri, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.maximum(pri, 0.0, out=pri)
            ps = float(pri.sum())
            if ps <= 0.0 or not np.isfinite(ps):
                picks.append(np.asarray(self.rng.integers(0, n, size=k_pri), dtype=np.int64))
            else:
                picks.append(
                    np.asarray(self.rng.choice(n, size=k_pri, replace=True, p=(pri / ps)), dtype=np.int64)
                )
        if not picks:
            return np.zeros((0,), dtype=np.int64)
        if len(picks) == 1:
            return picks[0]
        return np.concatenate(picks, axis=0)

    def _gather_rows(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        from chess_anti_engine.moves import POLICY_SIZE

        from .shard import densify_chunk
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if not self._chunks:
            raise ValueError("ArrayReplayBuffer is empty")
        if idx.size == 0:
            x_planes = int(np.asarray(self._chunks[0]["x"]).shape[1])
            return {
                "x": np.empty((0, x_planes, 8, 8), dtype=np.float16),
                "policy_target": np.empty((0, POLICY_SIZE), dtype=np.float16),
                "wdl_target": np.empty((0,), dtype=np.int8),
                "priority": np.empty((0,), dtype=np.float32),
                "has_policy": np.empty((0,), dtype=np.uint8),
            }
  # Densify each chunk's selected rows, then merge into output.
  # This avoids shape mismatches when chunks have different sparse K values.
        selected: list[tuple[dict[str, np.ndarray], np.ndarray]] = []
        all_keys: set[str] = set()
        start = 0
        for chunk, chunk_n in zip(self._chunks, self._chunk_sizes):
            end = start + chunk_n
            mask = (idx >= start) & (idx < end)
            if np.any(mask):
                local = idx[mask] - start
                rows = {k: v[local] for k, v in chunk.items()}
                dense_rows = densify_chunk(rows, policy_size=POLICY_SIZE)
                selected.append((dense_rows, mask))
                all_keys.update(dense_rows.keys())
            start = end
  # Build prototype from ALL selected chunks so optional fields present
  # in any chunk are allocated (not just those in the first chunk).
        proto: dict[str, np.ndarray] = {}
        for dense_rows, _ in selected:
            for k, v in dense_rows.items():
                if k not in proto:
                    proto[k] = v
        out = {
            k: np.zeros((idx.shape[0], *proto[k].shape[1:]), dtype=proto[k].dtype)
            for k in sorted(all_keys) if k in proto
        }
        for dense_rows, mask in selected:
            for k, value in dense_rows.items():
                if k in out:
                    out[k][mask] = value
        return out

    def _sample_raw_indices(self, batch_size: int) -> np.ndarray:
        return self._sample_all_indices(batch_size)

    def sample_batch_arrays(self, batch_size: int, *, wdl_balance: bool = True) -> dict[str, np.ndarray]:
        if self._size <= 0:
            raise ValueError("ArrayReplayBuffer is empty")
        bs = int(batch_size)
        if not wdl_balance:
            return self._gather_rows(self._sample_raw_indices(bs))

        draw_cap_frac = 0.90
        wl_max_ratio = 1.5
        win_idx = np.flatnonzero(self._wdl == 0)
        draw_idx = np.flatnonzero(self._wdl == 1)
        loss_idx = np.flatnonzero(self._wdl == 2)
        if win_idx.size == 0 or loss_idx.size == 0:
            return self._gather_rows(self._sample_raw_indices(bs))

        p_draw = float(draw_idx.size) / float(max(1, self._size))
        n_draw = int(round(bs * p_draw))
        n_draw = min(n_draw, int(np.floor(draw_cap_frac * bs)))
        n_draw = max(0, min(bs, n_draw))
        if draw_idx.size == 0:
            n_draw = 0

        bs_decisive = bs - n_draw
        n_win = 0
        n_loss = 0
        if bs_decisive > 0:
            p_win = float(win_idx.size) / float(win_idx.size + loss_idx.size)
            n_win = int(round(bs_decisive * p_win))
            n_win = max(0, min(bs_decisive, n_win))
            n_loss = bs_decisive - n_win
            r = float(wl_max_ratio)
            if n_win > int(np.floor(r * n_loss)):
                n_loss = int(np.ceil(bs_decisive / (1.0 + r)))
                n_win = bs_decisive - n_loss
            elif n_loss > int(np.floor(r * n_win)):
                n_win = int(np.ceil(bs_decisive / (1.0 + r)))
                n_loss = bs_decisive - n_win

        picks = [
            self._sample_indices(draw_idx, n_draw),
            self._sample_indices(win_idx, n_win),
            self._sample_indices(loss_idx, n_loss),
        ]
        chosen = np.concatenate([p for p in picks if p.size > 0], axis=0) if any(p.size > 0 for p in picks) else np.zeros((0,), dtype=np.int64)
        if chosen.shape[0] < bs:
            extra = self._sample_raw_indices(bs - chosen.shape[0])
            chosen = np.concatenate([chosen, extra], axis=0) if chosen.size > 0 else extra
        elif chosen.shape[0] > bs:
            chosen = chosen[:bs]
        self.rng.shuffle(chosen)
        return self._gather_rows(chosen)

    def sample_batch(self, batch_size: int, *, wdl_balance: bool = True) -> list[ReplaySample]:
        from .shard import arrays_to_samples

        return arrays_to_samples(self.sample_batch_arrays(batch_size, wdl_balance=wdl_balance))
