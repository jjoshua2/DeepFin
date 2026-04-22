"""Thread-safe dispatcher around a single GPU evaluator.

Wraps ``DirectGPUEvaluator`` (or any evaluator matching the ``BatchEvaluator``
protocol) behind a lock so multiple walker threads can submit ``evaluate_encoded``
concurrently without corrupting the shared pinned buffers. Calls serialize on
the lock — the dispatcher does NOT batch across callers (walkers already
produce halving-round-sized batches of ~256+ leaves on their own).

Future (multi-GPU): replace the single-evaluator + lock with a pool of
per-device evaluators and a round-robin / least-loaded dispatch. Callers keep
the same ``evaluate_encoded`` entry point.

The dispatcher deliberately does NOT expose ``evaluate_encoded_async``.
``gumbel_c.py`` gates its pipelined path on ``hasattr(..., 'evaluate_encoded_async')``,
and that path is for cross-game pipelining at ``n_boards >= 64`` — irrelevant
for single-game UCI. Forcing the sync path also makes the lock contract
straightforward: one caller's result is safely copied out before the next
caller's buffer write begins.
"""
from __future__ import annotations

import threading

import numpy as np

from chess_anti_engine.inference import BatchEvaluator


class ThreadSafeGPUDispatcher:
    def __init__(self, evaluator: BatchEvaluator) -> None:
        self._eval = evaluator
        self._lock = threading.Lock()

    def evaluate_encoded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._eval.evaluate_encoded(x)

    def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
        # Proxy read-only attrs (device, _max_batch, model, etc.) to the
        # underlying evaluator. Only attribute lookup — mutations still go
        # through the wrapped object directly if the caller holds a reference.
        return getattr(self._eval, name)
