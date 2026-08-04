from __future__ import annotations

from collections import deque

import numpy as np

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.replay.sample import ReplaySample as ReplaySample

# Scalar encoding-identity markers. Two chunks that disagree on any of these
# describe DIFFERENT input or policy encodings, so the rows cannot be merged
# into one array without picking a winner. `samples_to_arrays` hard-fails on a
# mixed batch "so the buffer's scalar-metadata merge can hard-fail on mixed
# chunks" (replay/shard.py) and `_scalar_metadata_string` raises
# `mixed replay metadata ...` downstream -- but nothing reached that guard from
# here, because the merge below homogenised the markers to one value first.
# `_history_rep_fix` is mapped rather than merely compared: shard.py
# materializes it on every write path and documents "a missing shard attr
# provably means the flag was off", so an ABSENT marker beside a "true" one is
# a mixed set, not an unknown. The two encoding-NAME markers have no such
# default (`_scalar_metadata_string` returns None for an absent key and
# `_policy_encoding_for_size` infers from the width), so absence there is
# ignored exactly as it is downstream.
_REP_FIX_MARKER_KEY = "_history_rep_fix"
_ENCODING_NAME_MARKER_KEYS: tuple[str, ...] = (
    "_input_history_encoding",
    "_policy_encoding",
)


def _refuse_mixed_identity_markers(
    selected: list[tuple[dict[str, np.ndarray], np.ndarray]],
) -> None:
    """Raise when the selected chunks disagree about an encoding identity.

    The failure this prevents is not a crash, it is a silent agreement: the
    row-merge below broadcasts each chunk's scalar marker over that chunk's
    rows and then the shard sidecar promotes row 0's value to the whole set,
    so a holdout holding both `history_rep_fix` true and false chunks saved
    and reloaded as uniformly true. The set that does that is the best-model
    ruler's own (M4-3), and `history_rep_fix` changes the repetition planes
    under an unchanged encoding NAME -- there is no other signal that the
    ruler is scoring two encodings at once.

    Raises with the same message shape as `shard._scalar_metadata_string`, so
    the guard here and the guard the merge used to hide read identically.
    """
    if len(selected) <= 1:
        return

    def _values(rows: dict[str, np.ndarray], key: str) -> set[str]:
        if key not in rows:
            return set()
        arr = np.asarray(rows[key])
        return {str(v) for v in arr.reshape(-1).tolist() if str(v)}

    for key in _ENCODING_NAME_MARKER_KEYS:
        values: set[str] = set()
        for rows, _ in selected:
            values |= _values(rows, key)
        if len(values) > 1:
            raise ValueError(f"mixed replay metadata {key}: {sorted(values)}")

  # Compared through the SAME predicate the reader uses
  # (`shard.history_rep_fix_from_arrays`: strip, lower, == "true"), so the
  # guard cannot fire on a spelling the reader treats as identical.
    rep_fix: set[str] = set()
    for rows, _ in selected:
        found = _values(rows, _REP_FIX_MARKER_KEY)
        rep_fix |= {
            "true" if v.strip().lower() == "true" else "false" for v in found
        } or {"false"}
    if len(rep_fix) > 1:
        raise ValueError(
            f"mixed replay metadata {_REP_FIX_MARKER_KEY}: {sorted(rep_fix)}",
        )


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
                arr = np.asarray(chunk[k])
                chunk[k] = arr if arr.ndim == 0 else arr[remaining:]
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
        k_uni = round(k * (1.0 - self.surprise_mix))
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
        if int(batch_size) <= 0 or n <= 0:
            return np.zeros((0,), dtype=np.int64)
        return self._sample_indices(np.arange(n, dtype=np.int64), int(batch_size))

    def _gather_rows(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        from .shard import densify_chunk
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        if not self._chunks:
            raise ValueError("ArrayReplayBuffer is empty")
        policy_size = int(np.asarray(self._chunks[0].get("_policy_size", POLICY_SIZE)).item())
        if idx.size == 0:
            x_planes = int(np.asarray(self._chunks[0]["x"]).shape[1])
            return {
                "x": np.empty((0, x_planes, 8, 8), dtype=np.float16),
                "policy_target": np.empty((0, policy_size), dtype=np.float16),
                "wdl_target": np.empty((0,), dtype=np.int8),
                "priority": np.empty((0,), dtype=np.float32),
                "has_policy": np.empty((0,), dtype=np.uint8),
            }
  # Densify each chunk's selected rows, then merge into output.
  # This avoids shape mismatches when chunks have different sparse K values.
        selected_sparse: list[tuple[dict[str, np.ndarray], np.ndarray]] = []
        selected_policy_sizes: set[int] = set()
        all_keys: set[str] = set()
        start = 0
        for chunk, chunk_n in zip(self._chunks, self._chunk_sizes, strict=True):
            end = start + chunk_n
            mask = (idx >= start) & (idx < end)
            if np.any(mask):
                local = idx[mask] - start
                rows = {k: (v if np.asarray(v).ndim == 0 else v[local]) for k, v in chunk.items()}
                selected_policy_sizes.add(int(np.asarray(rows.get("_policy_size", POLICY_SIZE)).item()))
                selected_sparse.append((rows, mask))
            start = end
        if len(selected_policy_sizes) > 1:
            raise ValueError(f"mixed replay policy sizes in array buffer: {sorted(selected_policy_sizes)}")
        if not selected_policy_sizes:
            raise ValueError(
                "sample indices did not match any replay chunk; "
                f"indices range [{int(idx.min())}, {int(idx.max())}], size={self._size}",
            )
        policy_size = next(iter(selected_policy_sizes))
        selected: list[tuple[dict[str, np.ndarray], np.ndarray]] = []
        for rows, mask in selected_sparse:
            dense_rows = densify_chunk(rows, policy_size=policy_size)
            selected.append((dense_rows, mask))
            all_keys.update(dense_rows.keys())
  # Build prototype from ALL selected chunks so optional fields present
  # in any chunk are allocated (not just those in the first chunk).
        proto: dict[str, np.ndarray] = {}
        for dense_rows, _ in selected:
            for k, v in dense_rows.items():
                if k not in proto:
                    proto[k] = v
        _refuse_mixed_identity_markers(selected)
  # Widen each field's dtype across ALL selected chunks rather than taking
  # the prototype's. numpy's fixed-width string dtypes TRUNCATE on
  # assignment, so a `<U4` prototype ("true") silently stored the scalar
  # marker "false" as "fals" -- which then read back as rep_fix=False from a
  # chunk that had it ON. `np.result_type` is a no-op for the numeric
  # fields, whose dtypes are fixed by the shard schema.
        dtypes: dict[str, np.dtype] = {}
        for dense_rows, _ in selected:
            for k, v in dense_rows.items():
                arr_dtype = np.asarray(v).dtype
                prev = dtypes.get(k)
                dtypes[k] = arr_dtype if prev is None else np.result_type(prev, arr_dtype)
        out = {
            k: np.zeros((idx.shape[0], *proto[k].shape[1:]), dtype=dtypes[k])
            for k in sorted(all_keys) if k in proto
        }
        for dense_rows, mask in selected:
            for k, value in dense_rows.items():
                if k in out:
                    out[k][mask] = value
        return out

    def _sample_raw_indices(self, batch_size: int) -> np.ndarray:
        return self._sample_all_indices(batch_size)

    def export_arrays(self) -> dict[str, np.ndarray]:
        """Every row currently held, densified, oldest first.

        The inverse of :meth:`add_many_arrays`: the result is shard-shaped, so
        it round-trips through the shard writer/reader and can be fed straight
        back into a fresh buffer to reconstitute this one. Returns ``{}`` when
        the buffer is empty — an empty buffer has no plane count or policy
        width to describe, so there is nothing meaningful to write.
        """
        if self._size <= 0:
            return {}
        return self._gather_rows(np.arange(self._size, dtype=np.int64))

    def rows_slice_arrays(self, start: int, stop: int) -> dict[str, np.ndarray]:
        """Densified rows ``[start, stop)`` in buffer order, oldest first.

        The deterministic read path: no rng, no reweighting, no class
        rebalance. Out-of-range bounds are clamped rather than raising, so a
        caller that computed its own chunk boundaries cannot walk off the end
        when the buffer changed size underneath it.
        """
        n = int(self._size)
        lo = max(0, min(int(start), n))
        hi = max(lo, min(int(stop), n))
        return self._gather_rows(np.arange(lo, hi, dtype=np.int64))

    def batch_row_bounds(self, batch_size: int) -> list[tuple[int, int]]:
        """``[start, stop)`` pairs covering EVERY row exactly once, in order.

        The chunking of a deterministic full pass, defined once here so the
        consumer that prefetches and the consumer that does not cannot disagree
        about which rows land in which batch.

        The deterministic counterpart to :meth:`sample_batch_arrays`, and the
        reason that method cannot serve: all three of its behaviours make an
        evaluation a function of the rng state instead of a function of the set
        (docs/rl_loop_audit.md G14) -- it draws WITH REPLACEMENT, it
        WDL-rebalances every batch away from the set's own class mix, and it
        draws ``surprise_mix`` of each batch proportional to ``priority``.
        Measured on the live 2000-row frozen holdout that made it an effective
        sample of 1051 rows (ESS 52.6%) and gave `test_loss` a noise floor of
        sd 0.0522 nats -- larger than every effect any yardstick reading it was
        pre-registered to detect.

        The LAST pair is RAGGED whenever the row count is not a multiple of
        ``batch_size`` (production: 2000 rows at 512 = 3 x 512 + 464). A
        consumer that averages per-batch means without weighting by row count
        overweights that tail -- 1.078x at the production shape, i.e. the tail
        gets 1/4 = 0.2500 of the weight instead of the 464/2000 = 0.2320 its
        row count earns.
        """
        n = int(self._size)
        bs = int(batch_size)
        if n <= 0 or bs <= 0:
            return []
        return [(s, min(s + bs, n)) for s in range(0, n, bs)]

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
        n_draw = round(bs * p_draw)
        n_draw = min(n_draw, int(np.floor(draw_cap_frac * bs)))
        n_draw = max(0, min(bs, n_draw))
        if draw_idx.size == 0:
            n_draw = 0

        bs_decisive = bs - n_draw
        n_win = 0
        n_loss = 0
        if bs_decisive > 0:
            p_win = float(win_idx.size) / float(win_idx.size + loss_idx.size)
            n_win = round(bs_decisive * p_win)
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


# Public name. ``ReplayBuffer`` was previously a pure-delegation wrapper; the
# wrapper-as-API existed for historical reasons and added no behavior. Aliased
# here so external imports keep working.
ReplayBuffer = ArrayReplayBuffer
