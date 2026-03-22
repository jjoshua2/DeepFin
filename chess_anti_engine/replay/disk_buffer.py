"""Disk-backed replay buffer with array-backed hot shuffle storage.

Stores training data as NPZ shards on disk and keeps a small hot subset in
memory for efficient sampling. The hot path stays columnar/array-backed so the
trainer can batch directly from NumPy arrays instead of inflating tens of
thousands of ``ReplaySample`` Python objects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .buffer import ReplaySample
from .shard import (
    arrays_to_samples,
    delete_shard_path,
    iter_shard_paths,
    load_shard_arrays,
    local_shard_path,
    prune_storage_arrays,
    samples_to_arrays,
    save_local_shard_arrays,
    shard_index,
    shard_positions,
)


_ARRAY_FIELD_ORDER = (
    "x",
    "policy_target",
    "wdl_target",
    "priority",
    "has_policy",
    "sf_wdl",
    "has_sf_wdl",
    "sf_move_index",
    "has_sf_move",
    "sf_policy_target",
    "has_sf_policy",
    "moves_left",
    "has_moves_left",
    "is_network_turn",
    "has_is_network_turn",
    "categorical_target",
    "has_categorical",
    "policy_soft_target",
    "has_policy_soft",
    "future_policy_target",
    "has_future",
    "volatility_target",
    "has_volatility",
    "sf_volatility_target",
    "has_sf_volatility",
    "legal_mask",
    "has_legal_mask",
)


def _compact_f16(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    return np.array(arr, dtype=np.float16, copy=True, order="C")


def _compact_u8(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    return np.array(arr, dtype=np.uint8, copy=True, order="C")


def _compact_sample_inplace(sample: ReplaySample) -> ReplaySample:
    sample.x = _compact_f16(sample.x)
    sample.policy_target = _compact_f16(sample.policy_target)
    sample.sf_wdl = _compact_f16(sample.sf_wdl)
    sample.sf_policy_target = _compact_f16(sample.sf_policy_target)
    sample.categorical_target = _compact_f16(sample.categorical_target)
    sample.policy_soft_target = _compact_f16(sample.policy_soft_target)
    sample.future_policy_target = _compact_f16(sample.future_policy_target)
    sample.volatility_target = _compact_f16(sample.volatility_target)
    sample.sf_volatility_target = _compact_f16(sample.sf_volatility_target)
    sample.legal_mask = _compact_u8(sample.legal_mask)
    return sample


def _zeros_for_missing_field(
    name: str,
    *,
    n: int,
    policy_size: int,
    x_planes: int,
) -> np.ndarray:
    if name == "x":
        return np.zeros((n, x_planes, 8, 8), dtype=np.float16)
    if name == "policy_target":
        return np.zeros((n, policy_size), dtype=np.float16)
    if name == "wdl_target":
        return np.zeros((n,), dtype=np.int8)
    if name == "priority":
        return np.ones((n,), dtype=np.float32)
    if name == "has_policy":
        return np.ones((n,), dtype=np.uint8)
    if name == "sf_wdl":
        return np.zeros((n, 3), dtype=np.float16)
    if name == "has_sf_wdl":
        return np.zeros((n,), dtype=np.uint8)
    if name == "sf_move_index":
        return np.zeros((n,), dtype=np.int32)
    if name == "has_sf_move":
        return np.zeros((n,), dtype=np.uint8)
    if name == "sf_policy_target":
        return np.zeros((n, policy_size), dtype=np.float16)
    if name == "has_sf_policy":
        return np.zeros((n,), dtype=np.uint8)
    if name == "moves_left":
        return np.zeros((n,), dtype=np.float16)
    if name == "has_moves_left":
        return np.zeros((n,), dtype=np.uint8)
    if name == "is_network_turn":
        return np.zeros((n,), dtype=np.uint8)
    if name == "has_is_network_turn":
        return np.zeros((n,), dtype=np.uint8)
    if name == "categorical_target":
        return np.zeros((n, 32), dtype=np.float16)
    if name == "has_categorical":
        return np.zeros((n,), dtype=np.uint8)
    if name == "policy_soft_target":
        return np.zeros((n, policy_size), dtype=np.float16)
    if name == "has_policy_soft":
        return np.zeros((n,), dtype=np.uint8)
    if name == "future_policy_target":
        return np.zeros((n, policy_size), dtype=np.float16)
    if name == "has_future":
        return np.zeros((n,), dtype=np.uint8)
    if name == "volatility_target":
        return np.zeros((n, 3), dtype=np.float16)
    if name == "has_volatility":
        return np.zeros((n,), dtype=np.uint8)
    if name == "sf_volatility_target":
        return np.zeros((n, 3), dtype=np.float16)
    if name == "has_sf_volatility":
        return np.zeros((n,), dtype=np.uint8)
    if name == "legal_mask":
        return np.zeros((n, policy_size), dtype=np.uint8)
    if name == "has_legal_mask":
        return np.zeros((n,), dtype=np.uint8)
    raise KeyError(f"unknown replay field {name!r}")


def _normalize_arrays(arrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    x = arrs["x"]
    policy_target = arrs["policy_target"]
    n = int(x.shape[0])
    policy_size = int(policy_target.shape[1])
    x_planes = int(x.shape[1])

    out: dict[str, np.ndarray] = {}
    for name in _ARRAY_FIELD_ORDER:
        if name in arrs:
            out[name] = arrs[name]
        else:
            out[name] = _zeros_for_missing_field(
                name,
                n=n,
                policy_size=policy_size,
                x_planes=x_planes,
            )
    return out


def _batch_dims(arrs: dict[str, np.ndarray]) -> tuple[int, int, int]:
    x = arrs["x"]
    policy_target = arrs["policy_target"]
    return int(x.shape[0]), int(policy_target.shape[1]), int(x.shape[1])


def _slice_array_batch(arrs: dict[str, np.ndarray], idxs: np.ndarray) -> dict[str, np.ndarray]:
    ii = np.asarray(idxs, dtype=np.int64).reshape(-1)
    return {k: np.array(v[ii], copy=True, order="C") for k, v in arrs.items()}


def _concat_sparse_batches(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ValueError("no chunks to concatenate")
    if len(chunks) == 1:
        return chunks[0]

    out: dict[str, np.ndarray] = {}
    for name in _ARRAY_FIELD_ORDER:
        parts: list[np.ndarray] = []
        for chunk in chunks:
            n, chunk_policy_size, chunk_x_planes = _batch_dims(chunk)
            if name in chunk:
                parts.append(np.asarray(chunk[name]))
            else:
                parts.append(
                    _zeros_for_missing_field(
                        name,
                        n=n,
                        policy_size=chunk_policy_size,
                        x_planes=chunk_x_planes,
                    )
                )
        merged = np.concatenate(parts, axis=0)
        # Keep required fields explicit; drop optional fields that are uniformly absent.
        if any(name in chunk for chunk in chunks) or name in ("x", "policy_target", "wdl_target", "priority", "has_policy"):
            out[name] = merged
            continue
        if name.startswith("has_"):
            if np.any(merged):
                out[name] = merged
        else:
            # value arrays are only needed when the corresponding has_* flag is present
            flag_name = None
            if name == "sf_wdl":
                flag_name = "has_sf_wdl"
            elif name == "sf_move_index":
                flag_name = "has_sf_move"
            elif name == "sf_policy_target":
                flag_name = "has_sf_policy"
            elif name == "moves_left":
                flag_name = "has_moves_left"
            elif name == "is_network_turn":
                flag_name = "has_is_network_turn"
            elif name == "categorical_target":
                flag_name = "has_categorical"
            elif name == "policy_soft_target":
                flag_name = "has_policy_soft"
            elif name == "future_policy_target":
                flag_name = "has_future"
            elif name == "volatility_target":
                flag_name = "has_volatility"
            elif name == "sf_volatility_target":
                flag_name = "has_sf_volatility"
            elif name == "legal_mask":
                flag_name = "has_legal_mask"
            if flag_name is not None and flag_name in out:
                out[name] = merged
    return out


class DiskReplayBuffer:
    """Disk-backed replay buffer with array-backed hot shuffle storage."""

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

        # In-memory shuffle buffer as chunked arrays rather than ReplaySample objects.
        # Optional arrays may be omitted and synthesized only when gathering a batch.
        self._shuffle_buf: list[dict[str, np.ndarray]] = []
        self._shuffle_sizes: list[int] = []
        self._shuffle_size_total = 0
        self._shuffle_priority = np.zeros((0,), dtype=np.float32)
        self._shuffle_wdl = np.zeros((0,), dtype=np.int8)

        # Write buffer: chunked arrays that accumulate until shard_size, then flush.
        self._write_buf: list[dict[str, np.ndarray]] = []
        self._write_buf_sizes: list[int] = []
        self._write_buf_rows = 0

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

    def _effective_shuffle_cap(self) -> int:
        return max(1, min(int(self._shuffle_cap), int(self.capacity)))

    def _shuffle_len(self) -> int:
        return int(self._shuffle_size_total)

    def _append_shuffle_arrays(self, arrs: dict[str, np.ndarray]) -> None:
        n, _, _ = _batch_dims(arrs)
        if n <= 0:
            return
        self._shuffle_buf.append(dict(arrs))
        self._shuffle_sizes.append(n)
        self._shuffle_size_total += n
        self._shuffle_priority = np.concatenate(
            [self._shuffle_priority, np.asarray(arrs["priority"], dtype=np.float32)],
            axis=0,
        )
        self._shuffle_wdl = np.concatenate(
            [self._shuffle_wdl, np.asarray(arrs["wdl_target"], dtype=np.int8)],
            axis=0,
        )

    def _drop_oldest_from_shuffle(self, count: int) -> None:
        drop = int(max(0, count))
        if drop <= 0 or self._shuffle_size_total <= 0:
            return
        drop = min(drop, self._shuffle_size_total)
        remaining = drop
        while remaining > 0 and self._shuffle_buf:
            first_n = self._shuffle_sizes[0]
            if remaining >= first_n:
                self._shuffle_buf.pop(0)
                self._shuffle_sizes.pop(0)
                self._shuffle_size_total -= first_n
                remaining -= first_n
                continue
            chunk = self._shuffle_buf[0]
            for name, value in list(chunk.items()):
                chunk[name] = value[remaining:]
            self._shuffle_sizes[0] = first_n - remaining
            self._shuffle_size_total -= remaining
            remaining = 0
        self._shuffle_priority = self._shuffle_priority[drop:]
        self._shuffle_wdl = self._shuffle_wdl[drop:]

    def _trim_shuffle_buf(self) -> None:
        cap = self._effective_shuffle_cap()
        overflow = self._shuffle_len() - cap
        if overflow > 0:
            self._drop_oldest_from_shuffle(overflow)

    def _append_write_arrays(self, arrs: dict[str, np.ndarray]) -> None:
        n, _, _ = _batch_dims(arrs)
        if n <= 0:
            return
        self._write_buf.append(dict(arrs))
        self._write_buf_sizes.append(n)
        self._write_buf_rows += n

    def _take_write_prefix(self, count: int) -> dict[str, np.ndarray]:
        take = int(max(0, count))
        if take <= 0 or self._write_buf_rows <= 0:
            raise ValueError("write buffer is empty")
        take = min(take, self._write_buf_rows)
        gathered: list[dict[str, np.ndarray]] = []
        remaining = take
        while remaining > 0 and self._write_buf:
            first_n = self._write_buf_sizes[0]
            chunk = self._write_buf[0]
            if remaining >= first_n:
                gathered.append(chunk)
                self._write_buf.pop(0)
                self._write_buf_sizes.pop(0)
                self._write_buf_rows -= first_n
                remaining -= first_n
                continue
            gathered.append(_slice_array_batch(chunk, np.arange(remaining, dtype=np.int64)))
            for name, value in list(chunk.items()):
                chunk[name] = value[remaining:]
            self._write_buf_sizes[0] = first_n - remaining
            self._write_buf_rows -= remaining
            remaining = 0
        if len(gathered) == 1:
            return gathered[0]
        return _concat_sparse_batches(gathered)

    def _scan_existing_shards(self) -> None:
        """Discover shards already on disk (e.g. after trial restart)."""
        existing = iter_shard_paths(self._shard_dir)
        if not existing:
            return
        for p in existing:
            idx = shard_index(p)
            if idx >= 0:
                self._shard_index = max(self._shard_index, idx + 1)
            n = shard_positions(p)
            self._shard_paths.append(p)
            self._shard_sizes.append(n)
            self._total_positions += n

        if self._shard_paths:
            n_seed = min(len(self._shard_paths), self._refresh_shards * 2)
            for sp in self._shard_paths[-n_seed:]:
                try:
                    arrs, _ = load_shard_arrays(sp, lazy=False)
                    self._append_shuffle_arrays(arrs)
                except Exception:
                    pass
            self._trim_shuffle_buf()

    def __len__(self) -> int:
        """Total positions on disk + in write buffer."""
        return self._total_positions + self._write_buf_rows

    def add(self, sample: ReplaySample) -> None:
        self.add_many([sample])

    def add_many(self, samples: list[ReplaySample]) -> None:
        """Add samples: into shuffle buffer immediately, flush to disk when full."""
        if not samples:
            return
        compacted = [_compact_sample_inplace(s) for s in samples]
        arrs = prune_storage_arrays(samples_to_arrays(compacted))

        # Keep newest data available for training immediately in the hot buffer.
        self._append_shuffle_arrays(arrs)
        self._trim_shuffle_buf()

        self._append_write_arrays(arrs)
        while self._write_buf_rows >= self._shard_size:
            self._flush_shard_arrays(self._take_write_prefix(self._shard_size))

        self._enforce_window()

    def add_many_arrays(self, arrs: dict[str, np.ndarray]) -> None:
        """Add array-backed samples without materializing ReplaySample objects."""
        sparse = prune_storage_arrays(arrs)
        n = int(sparse["x"].shape[0])
        if n <= 0:
            return

        self._append_shuffle_arrays(sparse)
        self._trim_shuffle_buf()

        self._append_write_arrays(sparse)
        while self._write_buf_rows >= self._shard_size:
            self._flush_shard_arrays(self._take_write_prefix(self._shard_size))

        self._enforce_window()

    def flush(self) -> None:
        """Force-write any remaining samples in write buffer to disk."""
        if self._write_buf_rows > 0:
            self._flush_shard_arrays(self._take_write_prefix(self._write_buf_rows))
            self._enforce_window()

    def _flush_shard_arrays(self, arrs: dict[str, np.ndarray]) -> None:
        """Write a shard to disk."""
        path = local_shard_path(self._shard_dir, self._shard_index)
        saved_path = save_local_shard_arrays(path, arrs=arrs)
        self._shard_paths.append(saved_path)
        n = int(arrs["x"].shape[0])
        self._shard_sizes.append(n)
        self._total_positions += n
        self._shard_index += 1

    def _enforce_window(self) -> None:
        """Delete oldest shards when total exceeds capacity."""
        if self._total_positions > self.capacity and not self._shard_paths:
            print(
                f"[disk_buf] BUG: total_pos={self._total_positions} > cap={self.capacity} "
                f"but no tracked shards to delete!"
            )
        deleted = 0
        while self._total_positions > self.capacity and self._shard_paths:
            oldest = self._shard_paths.pop(0)
            n = self._shard_sizes.pop(0)
            self._total_positions -= n
            deleted += 1
            try:
                delete_shard_path(oldest)
            except Exception as e:
                print(f"[disk_buf] WARNING: failed to delete {oldest}: {e}")
        if deleted:
            print(
                f"[disk_buf] enforce_window: deleted {deleted} shards, "
                f"total_pos={self._total_positions}, cap={self.capacity}, "
                f"tracked={len(self._shard_paths)}"
            )

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
            pri = np.maximum(0.0, self._shuffle_priority[pool].astype(np.float64, copy=False))
            ps = float(pri.sum())
            if ps <= 0.0:
                chosen = self.rng.choice(pool.shape[0], size=k_pri, replace=True)
                picks.append(pool[np.asarray(chosen, dtype=np.int64)])
            else:
                p = pri / ps
                chosen = self.rng.choice(pool.shape[0], size=k_pri, replace=True, p=p)
                picks.append(pool[np.asarray(chosen, dtype=np.int64)])
        if not picks:
            return np.zeros((0,), dtype=np.int64)
        if len(picks) == 1:
            return picks[0]
        return np.concatenate(picks, axis=0)

    def _sample_all_indices(self, batch_size: int) -> np.ndarray:
        n = self._shuffle_len()
        bs = int(batch_size)
        if bs <= 0 or n <= 0:
            return np.zeros((0,), dtype=np.int64)

        k_uni = int(round(bs * (1.0 - self.surprise_mix)))
        k_pri = bs - k_uni
        picks: list[np.ndarray] = []
        if k_uni > 0:
            picks.append(np.asarray(self.rng.integers(0, n, size=k_uni), dtype=np.int64))
        if k_pri > 0:
            pri = np.maximum(0.0, self._shuffle_priority.astype(np.float64, copy=False))
            ps = float(pri.sum())
            if ps <= 0.0:
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
        if not self._shuffle_buf:
            raise ValueError("DiskReplayBuffer shuffle buffer is empty")
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1:
            idx = idx.reshape(-1)
        template = self._shuffle_buf[0]
        _, policy_size, x_planes = _batch_dims(template)
        if idx.size == 0:
            return {
                name: _zeros_for_missing_field(
                    name,
                    n=0,
                    policy_size=policy_size,
                    x_planes=x_planes,
                )
                for name in ("x", "policy_target", "wdl_target", "priority", "has_policy")
            }

        if np.any(idx < 0) or np.any(idx >= self._shuffle_len()):
            raise IndexError("sample index out of shuffle-buffer range")

        selected: list[tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]] = []
        required = {"x", "policy_target", "wdl_target", "priority", "has_policy"}
        present_optional: set[str] = set()
        start = 0
        for chunk, chunk_n in zip(self._shuffle_buf, self._shuffle_sizes):
            end = start + chunk_n
            mask = (idx >= start) & (idx < end)
            if np.any(mask):
                local = idx[mask] - start
                selected.append((chunk, mask, local))
                present_optional.update(set(chunk.keys()) - required)
            start = end

        out = {
            name: _zeros_for_missing_field(
                name,
                n=int(idx.shape[0]),
                policy_size=policy_size,
                x_planes=x_planes,
            )
            for name in sorted(required | present_optional)
        }
        for chunk, mask, local in selected:
            for name, value in chunk.items():
                out[name][mask] = value[local]
        return out

    def sample_batch_arrays(self, batch_size: int, *, wdl_balance: bool = True) -> dict[str, np.ndarray]:
        """Sample a batch as compact arrays without materializing ReplaySample objects."""
        self._trim_shuffle_buf()
        n = self._shuffle_len()
        if n == 0:
            raise ValueError("DiskReplayBuffer shuffle buffer is empty")

        self._sample_count += 1
        if (self._sample_count % self._refresh_interval == 0) and self._shard_paths:
            self._refresh_shuffle_buf()
            n = self._shuffle_len()

        bs = int(batch_size)
        if not wdl_balance:
            return self._sample_raw_arrays(bs)

        draw_cap_frac = 0.90
        wl_max_ratio = 1.5

        win_idx = np.flatnonzero(self._shuffle_wdl == 0)
        draw_idx = np.flatnonzero(self._shuffle_wdl == 1)
        loss_idx = np.flatnonzero(self._shuffle_wdl == 2)

        if win_idx.size == 0 or loss_idx.size == 0:
            return self._sample_raw_arrays(bs)

        p_draw = float(draw_idx.size) / float(max(1, n))
        n_draw = int(round(bs * p_draw))
        n_draw_cap = int(np.floor(draw_cap_frac * bs))
        n_draw = min(n_draw, n_draw_cap)
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
        """Compatibility path returning ReplaySample objects."""
        return arrays_to_samples(self.sample_batch_arrays(batch_size, wdl_balance=wdl_balance))

    def _sample_raw_indices(self, batch_size: int) -> np.ndarray:
        return self._sample_all_indices(batch_size)

    def _sample_raw_arrays(self, batch_size: int) -> dict[str, np.ndarray]:
        return self._gather_rows(self._sample_raw_indices(batch_size))

    def _refresh_shuffle_buf(self) -> None:
        """Replace oldest shuffle samples with rows from randomly chosen disk shards."""
        if not self._shard_paths:
            return

        n_shards = len(self._shard_paths)
        n_pick = min(self._refresh_shards, n_shards)

        weights = np.arange(1, n_shards + 1, dtype=np.float64)
        weights /= weights.sum()
        chosen_idxs = self.rng.choice(n_shards, size=n_pick, replace=False, p=weights)

        loaded: list[dict[str, np.ndarray]] = []
        loaded_n = 0
        for idx in chosen_idxs:
            try:
                arrs, _ = load_shard_arrays(self._shard_paths[int(idx)], lazy=False)
                loaded.append(arrs)
                loaded_n += int(arrs["x"].shape[0])
            except Exception:
                pass

        if loaded_n <= 0:
            return

        replace_n = min(self._shuffle_len(), loaded_n)
        if replace_n > 0:
            self._drop_oldest_from_shuffle(replace_n)
        for arrs in loaded:
            self._append_shuffle_arrays(arrs)
        self._trim_shuffle_buf()

    def clear(self) -> None:
        """Remove all data (disk and memory)."""
        self._shuffle_buf = []
        self._shuffle_sizes = []
        self._shuffle_size_total = 0
        self._shuffle_priority = np.zeros((0,), dtype=np.float32)
        self._shuffle_wdl = np.zeros((0,), dtype=np.int8)
        self._write_buf = []
        self._write_buf_sizes = []
        self._write_buf_rows = 0
        for p in self._shard_paths:
            try:
                delete_shard_path(p)
            except Exception:
                pass
        self._shard_paths = []
        self._shard_sizes = []
        self._total_positions = 0
