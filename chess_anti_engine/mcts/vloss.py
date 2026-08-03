"""Release virtual loss that a failed descend/evaluate/integrate cycle left behind.

Every batched-VL search in this repo splits one atomic operation across two C
calls with arbitrary Python — including a network evaluation — in between::

    tree.walker_descend_puct(...)      # APPLIES vloss along each path
    pol, wdl = evaluator.evaluate(...) # <-- can raise (broker TimeoutError)
    tree.walker_integrate_leaf(...)    # REMOVES vloss, then backprops

If the middle line raises, the vloss stays on the tree. The tree is
caller-owned and outlives the failed run (``SearchWorker._tree`` is reused
across chunks and across plies), so the leak biases selection away from those
subtrees for the rest of the game — silently, since nothing ever reads virtual
loss back out. Measured at 8-48 leaked units per failed run over 5/5 runs
(``sb_cwalk_vloss_leak_on_eval_error.py``, SHARED_BROKER_AUDIT B7); the raising
evaluator is the documented one, ``SlotInferenceClient``'s I1 TimeoutError.

``MCTSTree.remove_vloss_path`` has existed and been exported the whole time and
was called from nowhere in ``chess_anti_engine/``. These helpers are its callers.

Both run inside ``finally`` blocks during exception unwinding, so neither may
raise: an exception here would REPLACE the evaluator error the caller is about
to report, turning a diagnosable broker timeout into a confusing tree error.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

_log = logging.getLogger(__name__)

# Row stride of the flat ``path_buf`` that ``batch_descend_puct`` fills. Must
# match MCTS_MAX_PATH in _mcts_tree.c; the batch helpers below index with it.
MAX_PATH = 512


def release_paths(tree: Any, paths: Sequence[Any], *, vloss_weight: int) -> int:
    """Remove vloss from each path in ``paths``. Returns how many were released.

    ``paths`` must contain ONLY paths that were descended and not integrated:
    ``walker_integrate_leaf``/``batch_integrate_leaves`` already remove vloss
    for what they consume, and removing twice would push the counters below the
    real in-flight vloss of a concurrent walker.
    """
    if vloss_weight <= 0 or not paths:
        return 0
    released = 0
    for path in paths:
        try:
            tree.remove_vloss_path(path)
        except Exception:  # see module docstring: must not raise
            _log.exception("failed to release virtual loss on a pending path")
        else:
            released += 1
    if released:
        # A guard nobody can observe is indistinguishable from one that never
        # fires, and this one only ever runs on a path that is already
        # unwinding an exception -- so there is no other trace that it worked.
        # Bounded by the failure rate: the pools set stop_event on the first
        # error, so a broker outage costs roughly one line per walker thread.
        _log.warning(
            "released virtual loss on %d un-integrated leaf path(s) after a "
            "failed descend/evaluate/integrate cycle (SHARED_BROKER_AUDIT B7)",
            released,
        )
    return released


def release_batch_rows(
    tree: Any,
    bufs: dict[str, np.ndarray],
    n: int,
    *,
    vloss_weight: int,
    start: int = 0,
) -> int:
    """Release vloss for rows ``start:n`` of a ``batch_descend_puct`` buffer set.

    Terminal rows are skipped: ``batch_descend_puct`` backprops them inline and
    never applies vloss to them, so ``is_term`` is the same discriminator
    ``batch_integrate_leaves`` uses.
    """
    if vloss_weight <= 0 or n <= start:
        return 0
    try:
        path_buf = np.asarray(bufs["path_buf"]).reshape(-1)
        path_lens = np.asarray(bufs["path_lens"])
        is_term = np.asarray(bufs["is_term"])
    except Exception:  # see module docstring: must not raise
        _log.exception("failed to read pending descend buffers for vloss release")
        return 0
    paths = []
    for i in range(int(start), int(n)):
        if int(is_term[i]):
            continue
        plen = int(path_lens[i])
        if plen <= 0:
            continue
        base = i * MAX_PATH
        paths.append(np.array(path_buf[base:base + plen], dtype=np.int32))
    return release_paths(tree, paths, vloss_weight=vloss_weight)
