"""Single-thread batched virtual-loss PUCT with 2-slot async pipeline.

Drives N sims by alternating between two pinned-input slots so the GPU
forward of chunk K runs concurrently with the CPU descent of chunk K+1.
Bench at gather=512 hits ~54k nps on a 10-layer 384-dim model on RTX 5090,
matching the gumbel walkers=1 path while giving classic PUCT visit counts.

The pipeline lives inside ``run`` — caller hands in (tree, root_id,
root_cboard, target_sims) and gets back when all sims are integrated. No
threading, no walker pool: the throughput comes from CPU/GPU overlap, not
parallelism. Buffers are pre-allocated once in ``__init__`` and reused
across calls.

Evaluator must expose the inplace-async API:
    n_slots                        — must be ≥ 2
    get_input_buffer(bsz, slot)   — returns the slot's pinned input view
    evaluate_inplace_async(bsz, slot)
                                  — returns (pol_t, wdl_t, cuda_event).
                                    Caller waits on the event before reading
                                    pol_t / wdl_t.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from chess_anti_engine.encoding._lc0_ext import CBoard as _CBoard
from chess_anti_engine.encoding.features import RELATION_COUNT
from chess_anti_engine.inference_cache import PucvEvalCache, PucvEvalCacheStats
from chess_anti_engine.mcts._mcts_tree import MCTSTree

# Stride constants matching the C extension's per-leaf write layout.
_MAX_PATH = 512
_MAX_LEGAL = 256
_PLANES = 146


def _alloc_buffers(
    gather: int, planes: int = _PLANES, *, with_relations: bool = False,
) -> dict[str, np.ndarray]:
    bufs: dict[str, np.ndarray] = {
        "leaf_ids": np.empty(gather, dtype=np.int32),
        "path_buf": np.empty(gather * _MAX_PATH, dtype=np.int32),
        "path_lens": np.empty(gather, dtype=np.int32),
        "legal_buf": np.empty(gather * _MAX_LEGAL, dtype=np.int32),
        "legal_lens": np.empty(gather, dtype=np.int32),
        "term_qs": np.empty(gather, dtype=np.float64),
        "is_term": np.empty(gather, dtype=np.int8),
        "cache_keys": np.empty((gather, 2), dtype=np.uint64),
        "enc_rows": np.empty((gather, planes, 8, 8), dtype=np.float32),
    }
    if with_relations:
        # zeros (not empty): batch_descend_puct zeroes terminal rows itself,
        # but a fresh buffer should never carry garbage into the model.
        bufs["rel"] = np.zeros(
            (gather, RELATION_COUNT, 64, 64), dtype=np.uint8,
        )
    return bufs


class PucvChunker:
    """Pipelined batched-VL PUCT runner. One instance per ``SearchWorker``.

    Pre-allocates 2× per-slot CPU metadata buffers and reuses them across
    every ``run`` call. Encoding lands directly in the evaluator's pinned
    input via ``get_input_buffer`` — no extra copy.
    """

    def __init__(
        self,
        evaluator: Any,
        *,
        gather: int = 512,
        c_puct: float = 1.4,
        fpu_at_root: float = 0.0,
        fpu_reduction: float = 0.2,
        vloss_weight: int = 3,
        vloss_mode: int = 0,
        eval_cache_entries: int = 0,
        input_planes: int = _PLANES,
        compute_relations: bool = False,
    ) -> None:
        if not hasattr(evaluator, "evaluate_inplace_async"):
            raise TypeError(
                "evaluator must expose evaluate_inplace_async (async-pipeline path)",
            )
        if getattr(evaluator, "n_slots", 1) < 2:
            raise ValueError(
                f"evaluator needs n_slots >= 2 for pipelining, got {evaluator.n_slots}",
            )
        self._ev = evaluator
        self._gather = max(1, int(gather))
        self._c_puct = float(c_puct)
        self._fpu_root = float(fpu_at_root)
        self._fpu_red = float(fpu_reduction)
        self._vloss = int(vloss_weight)
        self._vloss_mode = int(vloss_mode)
        self._cache = (
            PucvEvalCache(max_entries=int(eval_cache_entries))
            if eval_cache_entries > 0
            else None
        )
  # Ping-pong CPU metadata buffer sets, one per slot. Encoding for slot
  # k goes into ev.get_input_buffer(_, slot=k); we don't dual that.
        self._planes = int(input_planes)
        self._relations = bool(compute_relations)
        self._bufs = [
            _alloc_buffers(self._gather, self._planes, with_relations=self._relations),
            _alloc_buffers(self._gather, self._planes, with_relations=self._relations),
        ]

    def cache_stats(self) -> PucvEvalCacheStats | None:
        return None if self._cache is None else self._cache.stats()

    def run(
        self,
        tree: MCTSTree,
        root_id: int,
        root_cboard: _CBoard,
        target_sims: int,
    ) -> int:
        """Drive ``target_sims`` simulations on (tree, root_id) under root_cboard.

        Returns the number of sims actually run (== target_sims unless the
        tree is solved/terminal at root, in which case 0).
        """
        if target_sims <= 0:
            return 0

        ev = self._ev
        gather = self._gather
        sims = 0
        pending_slot = -1
        pending_n = 0
        pending_buf_idx = -1
        pending_handle: Any = None

        while sims < target_sims or pending_handle is not None:
            n = min(gather, target_sims - sims) if sims < target_sims else 0

            if n > 0:
                next_slot = 0 if pending_slot < 0 else (1 - pending_slot)
                next_buf_idx = (
                    0 if pending_buf_idx < 0 else (1 - pending_buf_idx)
                )
                inp = ev.get_input_buffer(n, slot=next_slot)
                inp_np = inp.numpy() if hasattr(inp, "numpy") else inp
                enc_view = np.asarray(inp_np).reshape(n, self._planes, 8, 8)
                b = self._bufs[next_buf_idx]
                rel = b["rel"] if self._relations else None
                tree.batch_descend_puct(
                    root_id, root_cboard, n,
                    self._c_puct, self._fpu_root, self._fpu_red, self._vloss,
                    enc_view,
                    b["leaf_ids"], b["path_buf"], b["path_lens"],
                    b["legal_buf"], b["legal_lens"],
                    b["term_qs"], b["is_term"], self._vloss_mode,
                    b["cache_keys"] if self._cache is not None else None,
                    rel,
                )
                if self._cache is None:
                    if rel is None:
                        handle = ev.evaluate_inplace_async(n, slot=next_slot)
                    else:
                        handle = ev.evaluate_inplace_async(
                            n, slot=next_slot, relations=rel[:n],
                        )
                    next_handle = ("gpu_full", next_buf_idx, n, handle)
                else:
                    next_handle = self._prepare_cached_submit(
                        ev, next_slot, next_buf_idx, n, enc_view, b, rel,
                    )
            else:
                next_slot = -1
                next_buf_idx = -1
                next_handle = None

            if pending_handle is not None:
                b_cur, pending_n, pol, wdl = self._finish_pending(
                    pending_handle,
                )
                tree.batch_integrate_leaves(
                    pending_n,
                    b_cur["path_buf"], b_cur["path_lens"],
                    b_cur["legal_buf"], b_cur["legal_lens"],
                    b_cur["is_term"], pol, wdl, self._vloss,
                )

            if next_handle is not None:
                pending_slot = next_slot
                pending_buf_idx = next_buf_idx
                pending_n = n
                pending_handle = next_handle
                sims += n
            else:
                pending_slot = -1
                pending_buf_idx = -1
                pending_n = 0
                pending_handle = None

        return sims

    def _prepare_cached_submit(
        self,
        ev: Any,
        slot: int,
        buf_idx: int,
        n: int,
        enc_view: np.ndarray,
        b: dict[str, np.ndarray],
        rel: np.ndarray | None,
    ) -> tuple[Any, ...]:
        cache = self._cache
        assert cache is not None
        hit_values: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        miss_indices: list[int] = []
        for i in range(n):
            if int(b["is_term"][i]) != 0:
                continue
            hit = cache.get(b["cache_keys"][i], enc_view[i])
            if hit is None:
                miss_indices.append(i)
            else:
                hit_values[i] = hit

        if not miss_indices:
            return ("cache_only", buf_idx, n, hit_values)

        miss_idx = np.asarray(miss_indices, dtype=np.int32)
        miss_n = int(miss_idx.size)
        b["enc_rows"][:n] = enc_view[:n]
        enc_view[:miss_n] = enc_view[miss_idx]
        if rel is None:
            handle = ev.evaluate_inplace_async(miss_n, slot=slot)
        else:
            # Compact relation rows alongside the encoding rows. Cache hits
            # stay valid: relations are a pure function of the position, so
            # position-keyed cached outputs already include the bias effect.
            rel[:miss_n] = rel[miss_idx]
            handle = ev.evaluate_inplace_async(
                miss_n, slot=slot, relations=rel[:miss_n],
            )
        return ("gpu_miss", buf_idx, n, miss_idx, hit_values, handle)

    def _finish_pending(
        self,
        pending_handle: tuple[Any, ...],
    ) -> tuple[dict[str, np.ndarray], int, np.ndarray, np.ndarray]:
        kind = pending_handle[0]
        if kind == "gpu_full":
            _, buf_idx, n, handle = pending_handle
            pol_t, wdl_t, evt = handle
            if evt is not None:
                evt.synchronize()
            pol = pol_t.numpy() if hasattr(pol_t, "numpy") else np.asarray(pol_t)
            wdl = wdl_t.numpy() if hasattr(wdl_t, "numpy") else np.asarray(wdl_t)
            return self._bufs[buf_idx], int(n), pol, wdl

        if kind == "cache_only":
            _, buf_idx, n, hit_values = pending_handle
            b = self._bufs[buf_idx]
            pol, wdl = self._materialize_cached_outputs(int(n), hit_values, None, None)
            return b, int(n), pol, wdl

        _, buf_idx, n, miss_idx, hit_values, handle = pending_handle
        pol_t, wdl_t, evt = handle
        if evt is not None:
            evt.synchronize()
        pol_m = pol_t.numpy() if hasattr(pol_t, "numpy") else np.asarray(pol_t)
        wdl_m = wdl_t.numpy() if hasattr(wdl_t, "numpy") else np.asarray(wdl_t)
        b = self._bufs[buf_idx]
        cache = self._cache
        assert cache is not None
        for j, original_i in enumerate(miss_idx):
            i = int(original_i)
            cache.put(b["cache_keys"][i], b["enc_rows"][i], pol_m[j], wdl_m[j])
        pol, wdl = self._materialize_cached_outputs(
            int(n), hit_values, miss_idx, (pol_m, wdl_m),
        )
        return b, int(n), pol, wdl

    def _materialize_cached_outputs(
        self,
        n: int,
        hit_values: dict[int, tuple[np.ndarray, np.ndarray]],
        miss_idx: np.ndarray | None,
        miss_outputs: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        first_value = next(iter(hit_values.values()), None)
        if first_value is None and miss_outputs is not None:
            first_value = (miss_outputs[0][0], miss_outputs[1][0])
        if first_value is None:
            return (
                np.zeros((n, 4672), dtype=np.float32),
                np.zeros((n, 3), dtype=np.float32),
            )
        pol = np.zeros((n, *first_value[0].shape), dtype=first_value[0].dtype)
        wdl = np.zeros((n, *first_value[1].shape), dtype=first_value[1].dtype)
        for i, (pol_row, wdl_row) in hit_values.items():
            pol[i] = pol_row
            wdl[i] = wdl_row
        if miss_idx is not None and miss_outputs is not None:
            pol_m, wdl_m = miss_outputs
            for j, original_i in enumerate(miss_idx):
                i = int(original_i)
                pol[i] = pol_m[j]
                wdl[i] = wdl_m[j]
        return pol, wdl
