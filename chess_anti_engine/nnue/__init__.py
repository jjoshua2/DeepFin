"""Native big-net Stockfish-NNUE evaluator.

The C evaluator lives in ``_nnue_ext`` (built from ``_nnue_ext.c`` +
``_nnue_impl.h``); the same implementation header is compiled into the MCTS tree
extension, where it is reached through the value-provider seam rather than
called directly.

Weights are a RUNTIME ARTIFACT: build a pack with ``scripts/nnue_pack.py`` from
a ``.nnue`` file and pass its path to ``load()``. Nothing here hardcodes a path.
"""

from __future__ import annotations

__all__ = ["_nnue_ext"]
