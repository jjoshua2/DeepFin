"""The one validator for the selfplay/arena search-type knob (`mcts` / `mcts_type`).

Every consumer compares the value with ``== "gumbel"``, while the relations
transport guard compared ``.lower()``. So before this existed, ``mcts: Gumbel``
passed the guard and then dispatched to PUCT above 64 sims: accepted, then
silently ignored. Normalising once at each config boundary closes that.

Since 2026-11-14 selfplay, arena and puzzle eval have one search, compiled
Gumbel. ``"puct"`` is still a recognised word (so the error names the real
problem instead of "unknown value"), but ``require_gumbel_search`` refuses it
at every selfplay/arena/puzzle config boundary. The UCI engine's PUCT walker,
pucv and multi-GPU pools are a separate path and do not read this knob.
"""
from __future__ import annotations

import argparse

MCTS_TYPES: tuple[str, ...] = ("gumbel", "puct")

PUCT_RETIRED_MESSAGE = (
    "mcts: puct was retired for selfplay/arena on 2026-11-14 (it recorded ~8% of "
    "rows under the default diff-focus filter and was never the production path); "
    "use gumbel"
)


def normalize_mcts_type(value: object) -> str:
    """Return ``value`` as a lower-case member of ``MCTS_TYPES`` or raise.

    Case and surrounding whitespace are normalised (``"Gumbel"`` -> ``"gumbel"``);
    anything else raises ``ValueError``.
    """
    text = str(value).strip().lower()
    if text not in MCTS_TYPES:
        raise ValueError(f"mcts/mcts_type must be one of {MCTS_TYPES} (any case), got {value!r}")
    return text


def require_gumbel_search(value: object) -> str:
    """Normalise ``value`` and refuse anything but ``"gumbel"``.

    The config-boundary check for selfplay, arena and puzzle eval. Raises
    ``ValueError`` with ``PUCT_RETIRED_MESSAGE`` for ``puct`` (any case) and the
    ``normalize_mcts_type`` message for an unrecognised value.
    """
    text = normalize_mcts_type(value)
    if text != "gumbel":
        raise ValueError(PUCT_RETIRED_MESSAGE)
    return text


def gumbel_search_cli_arg(value: str) -> str:
    """argparse ``type=`` for ``--mcts``: ``require_gumbel_search`` with argparse's error.

    argparse also applies ``type`` to a string default, so a yaml-supplied
    default (``run.py``'s ``set_defaults``) is refused with the same message.
    """
    try:
        return require_gumbel_search(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
