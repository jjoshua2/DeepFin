"""The one validator for the selfplay/arena search-type knob (`mcts` / `mcts_type`).

Every consumer compares the value with ``== "gumbel"``, while the relations
transport guard compared ``.lower()``. So before this existed, ``mcts: Gumbel``
passed the guard and then dispatched to PUCT above 64 sims: accepted, then
silently ignored. Normalising once at each config boundary closes that.
"""
from __future__ import annotations

MCTS_TYPES: tuple[str, ...] = ("gumbel", "puct")


def normalize_mcts_type(value: object) -> str:
    """Return ``value`` as a lower-case member of ``MCTS_TYPES`` or raise.

    Case and surrounding whitespace are normalised (``"Gumbel"`` -> ``"gumbel"``);
    anything else raises ``ValueError``.
    """
    text = str(value).strip().lower()
    if text not in MCTS_TYPES:
        raise ValueError(f"mcts/mcts_type must be one of {MCTS_TYPES} (any case), got {value!r}")
    return text
