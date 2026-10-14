"""History-mode arguments for native PUCT, with legacy binary compatibility."""
from __future__ import annotations

from chess_anti_engine.encoding.lc0 import c_input_history_mode
from chess_anti_engine.mcts import _mcts_tree

_PUCT_HISTORY_ABI = 9


def puct_history_args(input_history_encoding: str | None) -> tuple[int, ...]:
    """Normalize checkpoint mode; rooted leaves require the new native argument.

    Omit the argument for legacy mode so callers can still use the older native
    signature. Fail before spawning pools or touching a tree for rooted modes.
    """
    mode = c_input_history_mode(input_history_encoding)
    if mode == 0:
        return ()
    abi = getattr(_mcts_tree, "ABI_VERSION", 0)
    if abi < _PUCT_HISTORY_ABI:
        raise RuntimeError(
            f"compiled _mcts_tree ABI_VERSION={abi} < required {_PUCT_HISTORY_ABI} "
            "for PUCT input-history modes; rebuild the C extension: "
            "python3 scripts/build_production_extensions.py"
        )
    return (mode,)
