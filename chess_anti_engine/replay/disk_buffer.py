"""Disk-backed replay buffer with in-memory shuffle buffer.

Stores training data as NPZ shards on disk. Keeps a small shuffle buffer
in memory (~20k samples) for efficient sampling. This reduces per-trial
memory from ~4-10 GB to ~1 GB, enabling 10+ concurrent Ray Tune trials
on a 128 GB machine.

Drop-in replacement for ReplayBuffer — same interface expected by
Trainer.train_steps(): sample_batch(), add_many(), __len__(), .rng.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .buffer import ReplaySample
from .shard import load_npz, save_npz


class DiskReplayBuffer:
    """Disk-backed replay buffer with small in-memory shuffle buffer."""

    def __init__(
        self,
        capacity: int,
        *,
        shard_dir: Path,
        rng: np.random.Generator,
        shuffle_cap: int = 20_000,
        shard_size: int = 1000,
        refresh_interval: int = 5,
        refresh_shards: int = 3,
    ):
        self.capacity = int(capacity)
        self.rng = rng
        self._shard_dir = Path(shard_dir)
        self._shard_dir.mkdir(parents=True, exist_ok=True)

        self._shuffle_cap = int(shuffle_cap)
        self._shard_size = int(shard_size)
        self._refresh_interval = int(refresh_interval)
        self._refresh_shards = int(refresh_shards)

        # In-memory shuffle buffer (small subset of all data).
        self._shuffle_buf: list[ReplaySample] = []

        # Write buffer: accumulates until shard_size, then flushes to disk.
        self._write_buf: list[ReplaySample] = []

        # Disk shard tracking (ordered oldest-first).
        self._shard_paths: list[Path] = []
        self._shard_sizes: list[int] = []
        self._total_positions = 0
        self._shard_index = 0

        # Counter for scheduling shuffle buffer refreshes.
        self._sample_count = 0

        # KataGo-style surprise weighting: 50% uniform, 50% priority.
        self.surprise_mix = 0.5

        # Scan existing shards on disk (for resume).
        self._scan_existing_shards()

    def _scan_existing_shards(self) -> None:
        """Discover shards already on disk (e.g. after trial restart)."""
        existing = sorted(self._shard_dir.glob("shard_*.npz"))
        if not existing:
            return
        for p in existing:
            # Extract shard index from filename.
            try:
                idx = int(p.stem.split("_")[1])
                self._shard_index = max(self._shard_index, idx + 1)
            except (IndexError, ValueError):
                pass
            # Read sample count from file without fully deserializing.
            try:
                with np.load(str(p), allow_pickle=False) as z:
                    n = z["x"].shape[0] if "x" in z.files else 0
            except Exception:
                n = 0
            self._shard_paths.append(p)
            self._shard_sizes.append(n)
            self._total_positions += n

        # Seed shuffle buffer from most recent shards.
        if self._shard_paths:
            n_seed = min(len(self._shard_paths), self._refresh_shards * 2)
            for sp in self._shard_paths[-n_seed:]:
                try:
                    samples, _ = load_npz(sp)
                    self._shuffle_buf.extend(samples)
                except Exception:
                    pass
            # Trim to capacity.
            if len(self._shuffle_buf) > self._shuffle_cap:
                self._shuffle_buf = self._shuffle_buf[-self._shuffle_cap:]

    def __len__(self) -> int:
        """Total positions on disk + in write buffer."""
        return self._total_positions + len(self._write_buf)

    def add(self, sample: ReplaySample) -> None:
        self.add_many([sample])

    def add_many(self, samples: list[ReplaySample]) -> None:
        """Add samples: into shuffle buffer immediately, flush to disk when full."""
        # Add to shuffle buffer (newest data always available for training).
        self._shuffle_buf.extend(samples)
        if len(self._shuffle_buf) > self._shuffle_cap:
            # Keep the most recent samples.
            self._shuffle_buf = self._shuffle_buf[-self._shuffle_cap:]

        # Add to write buffer, flush to disk when we have enough.
        self._write_buf.extend(samples)
        while len(self._write_buf) >= self._shard_size:
            self._flush_shard(self._write_buf[:self._shard_size])
            self._write_buf = self._write_buf[self._shard_size:]

        # Enforce sliding window.
        self._enforce_window()

    def flush(self) -> None:
        """Force-write any remaining samples in write buffer to disk."""
        if self._write_buf:
            self._flush_shard(self._write_buf)
            self._write_buf = []
            self._enforce_window()

    def _flush_shard(self, samples: list[ReplaySample]) -> None:
        """Write a shard to disk."""
        path = self._shard_dir / f"shard_{self._shard_index:06d}.npz"
        save_npz(path, samples=samples)
        self._shard_paths.append(path)
        self._shard_sizes.append(len(samples))
        self._total_positions += len(samples)
        self._shard_index += 1

    def _enforce_window(self) -> None:
        """Delete oldest shards when total exceeds capacity."""
        while self._total_positions > self.capacity and self._shard_paths:
            oldest = self._shard_paths.pop(0)
            n = self._shard_sizes.pop(0)
            self._total_positions -= n
            try:
                oldest.unlink(missing_ok=True)
            except Exception:
                pass

    def sample_batch(self, batch_size: int, *, wdl_balance: bool = True) -> list[ReplaySample]:
        """Sample a batch from the shuffle buffer."""
        n = len(self._shuffle_buf)
        if n == 0:
            raise ValueError("DiskReplayBuffer shuffle buffer is empty")

        self._sample_count += 1

        # Periodically refresh shuffle buffer from disk shards.
        if (self._sample_count % self._refresh_interval == 0
                and len(self._shard_paths) > 0):
            self._refresh_shuffle_buf()

        bs = int(batch_size)

        if not wdl_balance:
            return self._sample_raw(bs)

        # WDL-balanced sampling (same logic as ReplayBuffer).
        draw_cap_frac = 0.90
        wl_max_ratio = 1.5

        buckets: dict[int, list[int]] = {0: [], 1: [], 2: []}
        for i, s in enumerate(self._shuffle_buf):
            wdl = int(s.wdl_target)
            if wdl in buckets:
                buckets[wdl].append(i)

        win_idx = buckets[2]
        draw_idx = buckets[1]
        loss_idx = buckets[0]

        if len(win_idx) == 0 or len(loss_idx) == 0:
            return self._sample_raw(bs)

        def _sample_from_indices(idxs: list[int], k: int) -> list[ReplaySample]:
            if k <= 0:
                return []
            k_uni = int(round(k * (1.0 - self.surprise_mix)))
            k_pri = k - k_uni
            out_local: list[ReplaySample] = []

            if k_uni > 0:
                chosen = self.rng.choice(len(idxs), size=k_uni, replace=True)
                out_local.extend([self._shuffle_buf[idxs[int(i)]] for i in chosen])

            if k_pri > 0:
                pri = np.array(
                    [max(0.0, float(self._shuffle_buf[j].priority)) for j in idxs],
                    dtype=np.float64,
                )
                ps = float(pri.sum())
                if ps <= 0:
                    chosen = self.rng.choice(len(idxs), size=k_pri, replace=True)
                    out_local.extend([self._shuffle_buf[idxs[int(i)]] for i in chosen])
                else:
                    p = pri / ps
                    chosen = self.rng.choice(np.arange(len(idxs)), size=k_pri, replace=True, p=p)
                    out_local.extend([self._shuffle_buf[idxs[int(i)]] for i in chosen])
            return out_local

        p_draw = float(len(draw_idx)) / float(max(1, n))
        n_draw = int(round(bs * p_draw))
        n_draw_cap = int(np.floor(draw_cap_frac * bs))
        n_draw = min(n_draw, n_draw_cap)
        n_draw = max(0, min(bs, n_draw))
        if len(draw_idx) == 0:
            n_draw = 0

        bs_decisive = bs - n_draw

        n_win = 0
        n_loss = 0
        if bs_decisive > 0:
            p_win = float(len(win_idx)) / float(len(win_idx) + len(loss_idx))
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

        out: list[ReplaySample] = []
        out.extend(_sample_from_indices(draw_idx, n_draw))
        out.extend(_sample_from_indices(win_idx, n_win))
        out.extend(_sample_from_indices(loss_idx, n_loss))

        if len(out) < bs:
            out.extend(self._sample_raw(bs - len(out)))
        elif len(out) > bs:
            out = out[:bs]

        self.rng.shuffle(out)  # type: ignore[arg-type]
        return out

    def _sample_raw(self, batch_size: int) -> list[ReplaySample]:
        """Sample without WDL balancing."""
        n = len(self._shuffle_buf)
        bs = int(batch_size)
        k_uni = int(round(bs * (1.0 - self.surprise_mix)))
        k_pri = bs - k_uni

        out: list[ReplaySample] = []

        if k_uni > 0:
            idxs = self.rng.integers(0, n, size=k_uni)
            out.extend([self._shuffle_buf[int(i)] for i in idxs])

        if k_pri > 0:
            pri = np.array(
                [max(0.0, float(s.priority)) for s in self._shuffle_buf],
                dtype=np.float64,
            )
            ps = float(pri.sum())
            if ps <= 0:
                idxs = self.rng.integers(0, n, size=k_pri)
                out.extend([self._shuffle_buf[int(i)] for i in idxs])
            else:
                p = pri / ps
                idxs = self.rng.choice(np.arange(n), size=k_pri, replace=True, p=p)
                out.extend([self._shuffle_buf[int(i)] for i in idxs])

        return out

    def _refresh_shuffle_buf(self) -> None:
        """Replace half the shuffle buffer with samples from random disk shards."""
        if not self._shard_paths:
            return

        n_shards = len(self._shard_paths)
        n_pick = min(self._refresh_shards, n_shards)

        # Weight toward recent shards (linear: shard 0 gets weight 1, last gets weight n).
        weights = np.arange(1, n_shards + 1, dtype=np.float64)
        weights /= weights.sum()

        chosen_idxs = self.rng.choice(n_shards, size=n_pick, replace=False, p=weights)

        new_samples: list[ReplaySample] = []
        for idx in chosen_idxs:
            try:
                samples, _ = load_npz(self._shard_paths[int(idx)])
                new_samples.extend(samples)
            except Exception:
                pass

        if not new_samples:
            return

        # Replace oldest half of shuffle buffer with new samples.
        half = len(self._shuffle_buf) // 2
        self._shuffle_buf = self._shuffle_buf[half:] + new_samples

        # Trim to capacity.
        if len(self._shuffle_buf) > self._shuffle_cap:
            self._shuffle_buf = self._shuffle_buf[-self._shuffle_cap:]

    def clear(self) -> None:
        """Remove all data (disk and memory)."""
        self._shuffle_buf = []
        self._write_buf = []
        for p in self._shard_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        self._shard_paths = []
        self._shard_sizes = []
        self._total_positions = 0
