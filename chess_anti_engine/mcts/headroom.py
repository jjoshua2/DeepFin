"""Arena headroom for lock-free concurrent descents on one shared ``MCTSTree``.

The walker pool and the multi-device PUCV pool descend the tree from several
threads without ``tree_lock`` while other threads expand leaves under it. An
expansion that outgrows the node or child arrays reallocs them, and a thread
mid-descent then reads freed memory. ``reserve_concurrent_headroom`` grows the
arrays up front so that a run of ``sims`` simulations cannot realloc.
"""
from __future__ import annotations

from typing import Any

# One simulation adds at most this many nodes (and child edges): the leaf's
# legal moves (at most 218) plus, on the walker path, up to 16 forced
# single-child expansions along the descent (218 + 16 = 234), with margin.
MAX_NODES_PER_SIM = 256


def reserve_concurrent_headroom(tree: Any, sims: int) -> None:
    """Make room for ``sims`` simulations' worst-case growth before any
    concurrent descent starts. A no-op when the caller (the UCI search, which
    bounds the chunk by the Hash cap first) already reserved it; raises
    ``MemoryError`` rather than letting a run realloc under a descending thread.
    """
    need = max(0, int(sims)) * MAX_NODES_PER_SIM
    if need <= 0:
        return
    stats = tree.memory_stats()
    tree.reserve(int(stats["node_count"]) + need, int(stats["child_count"]) + need)
