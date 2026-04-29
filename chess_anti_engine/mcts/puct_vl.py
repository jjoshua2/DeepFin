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
from chess_anti_engine.mcts._mcts_tree import MCTSTree

# Stride constants matching the C extension's per-leaf write layout.
_MAX_PATH = 512
_MAX_LEGAL = 256
_PLANES = 146


def _alloc_buffers(gather: int) -> dict[str, np.ndarray]:
    return {
        "leaf_ids": np.empty(gather, dtype=np.int32),
        "path_buf": np.empty(gather * _MAX_PATH, dtype=np.int32),
        "path_lens": np.empty(gather, dtype=np.int32),
        "legal_buf": np.empty(gather * _MAX_LEGAL, dtype=np.int32),
        "legal_lens": np.empty(gather, dtype=np.int32),
        "term_qs": np.empty(gather, dtype=np.float64),
        "is_term": np.empty(gather, dtype=np.int8),
    }


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
  # Ping-pong CPU metadata buffer sets, one per slot. Encoding for slot
  # k goes into ev.get_input_buffer(_, slot=k); we don't dual that.
        self._bufs = [_alloc_buffers(self._gather), _alloc_buffers(self._gather)]

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
                enc_view = np.asarray(inp_np).reshape(n, _PLANES, 8, 8)
                b = self._bufs[next_buf_idx]
                tree.batch_descend_puct(
                    root_id, root_cboard, n,
                    self._c_puct, self._fpu_root, self._fpu_red, self._vloss,
                    enc_view,
                    b["leaf_ids"], b["path_buf"], b["path_lens"],
                    b["legal_buf"], b["legal_lens"],
                    b["term_qs"], b["is_term"],
                )
                next_handle = ev.evaluate_inplace_async(n, slot=next_slot)
            else:
                next_slot = -1
                next_buf_idx = -1
                next_handle = None

            if pending_handle is not None:
                pol_t, wdl_t, evt = pending_handle
                if evt is not None:
                    evt.synchronize()
                pol = pol_t.numpy() if hasattr(pol_t, "numpy") else np.asarray(pol_t)
                wdl = wdl_t.numpy() if hasattr(wdl_t, "numpy") else np.asarray(wdl_t)
                b_cur = self._bufs[pending_buf_idx]
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
