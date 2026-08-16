"""Shared Stockfish binary discovery for tests that need a real engine."""
from __future__ import annotations

import os
import sys
from pathlib import Path

SF_CANDIDATES = [
    # The checkout's own published engine, derived from this file rather than
    # written absolute — an absolute path only ever named one machine's copy.
    str(Path(__file__).resolve().parents[1] / "e2e_server" / "publish" / "stockfish"),
    "/usr/bin/stockfish",
    "/usr/games/stockfish",
]


def find_stockfish() -> str | None:
    # engine-backed tests only run inside WSL/Linux (startswith defeats
    # basedpyright's sys.platform literal narrowing, which would otherwise
    # flag the loop below as unreachable on non-linux check configs)
    if not sys.platform.startswith("linux"):
        return None
    for p in SF_CANDIDATES:
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None
