"""Shared Stockfish binary discovery for tests that need a real engine.

⚑⚑ THIS MODULE DECIDES WHETHER THE E2E SUITE RUNS AT ALL. `test_e2e_smoke.py`,
`test_selfplay_resume.py` and `test_sparse_multipv_labels.py` each do
`pytestmark = pytest.mark.skipif(SF_PATH is None, ...)`, so a discovery that
returns None turns the project's only end-to-end wiring check into 18 silent
skips -- a gate that cannot fail. CLAUDE.md MANDATES running the e2e smoke for
selfplay/distributed changes and a standing rule mandates running it IN THE
WORKTREE, so discovery must work from a worktree, not only from the main
checkout.

⚑ The logic itself lives in `chess_anti_engine.utils.engine_discovery`, not
here. The `scripts/blindspot_*` tools need the identical multi-root lookup for
the identical reason -- `e2e_server/publish/` is UNTRACKED runtime output, so a
checkout-relative candidate resolves in one tree and nowhere else -- and three
copies of a path constant is how the copies drift. This module is the tests'
view of it: the names below are re-exported so `pytest.MonkeyPatch` has one
obvious place to patch.
"""
from __future__ import annotations

from chess_anti_engine.utils import engine_discovery
from chess_anti_engine.utils.engine_discovery import (
    DISTRO_CANDIDATES,
    ENV_VAR,
    PUBLISHED,
    find_stockfish,
    main_checkout,
    stockfish_candidates,
)

__all__ = [
    "DISTRO_CANDIDATES",
    "ENV_VAR",
    "PUBLISHED",
    "SF_CANDIDATES",
    "engine_discovery",
    "find_stockfish",
    "main_checkout",
    "stockfish_candidates",
]

#: Retained for readers/debuggers; `find_stockfish` uses the live list so the
#: env override is honoured at call time.
SF_CANDIDATES = stockfish_candidates()
