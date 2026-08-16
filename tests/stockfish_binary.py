"""Shared Stockfish binary discovery for tests that need a real engine.

⚑⚑ THIS MODULE DECIDES WHETHER THE E2E SUITE RUNS AT ALL. `test_e2e_smoke.py`,
`test_selfplay_resume.py` and `test_sparse_multipv_labels.py` each do
`pytestmark = pytest.mark.skipif(SF_PATH is None, ...)`, so a discovery that
returns None turns the project's only end-to-end wiring check into 18 silent
skips -- a gate that cannot fail. CLAUDE.md MANDATES running the e2e smoke for
selfplay/distributed changes and a standing rule mandates running it IN THE
WORKTREE, so discovery must work from a worktree, not only from the main
checkout.

⚑ `e2e_server/publish/` is UNTRACKED runtime output (`git ls-files e2e_server`
returns nothing). It exists only in the checkout the server actually ran in, so
a checkout-relative candidate resolves in that one tree and nowhere else --
which is why the main checkout and the shared PATH install are both consulted.
The env override is the seam every other path-derived default in this repo got.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

#: Explicit override, checked first. The same seam as RATCHET_ROOT /
#: WATCHDOG_ROOT / HARVEST_SF_BIN: a machine that keeps its engine somewhere
#: unusual says so, instead of the suite silently degrading to zero coverage.
ENV_VAR = "CAE_STOCKFISH"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PUBLISHED = Path("e2e_server") / "publish" / "stockfish"


def _main_checkout() -> Path | None:
    """The checkout this worktree was created from, or None.

    `git rev-parse --git-common-dir` answers with the MAIN repository's `.git`
    directory even when called inside a linked worktree, which is exactly the
    lookup needed: the published engine lives in the main checkout because it
    is untracked.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=_REPO_ROOT, capture_output=True, text=True, check=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    common = Path(raw)
    if not common.is_absolute():
        common = (_REPO_ROOT / common).resolve()
    parent = common.parent
    return parent if parent.is_dir() else None


def stockfish_candidates() -> list[str]:
    """Every place a real engine may be, most specific first.

    Recomputed per call rather than frozen at import: the env override must be
    settable by a test (and by CI) without reloading the module.
    """
    candidates: list[str] = []
    override = os.environ.get(ENV_VAR, "").strip()
    if override:
        candidates.append(override)
    candidates.append(str(_REPO_ROOT / _PUBLISHED))
    main = _main_checkout()
    if main is not None and main != _REPO_ROOT:
        candidates.append(str(main / _PUBLISHED))
    on_path = shutil.which("stockfish")
    if on_path:
        candidates.append(on_path)
    # Kept after `which` as a last resort: a PATH-less environment (some CI
    # containers) still finds a distro package here.
    candidates.extend(["/usr/bin/stockfish", "/usr/games/stockfish"])
    # Order-preserving dedup, so a duplicate candidate is not reported twice
    # when `which` happens to resolve to one of the literals below it.
    seen: set[str] = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


#: Retained for readers/debuggers; `find_stockfish` uses the live list so the
#: env override is honoured at call time.
SF_CANDIDATES = stockfish_candidates()


def find_stockfish() -> str | None:
    # engine-backed tests only run inside WSL/Linux (startswith defeats
    # basedpyright's sys.platform literal narrowing, which would otherwise
    # flag the loop below as unreachable on non-linux check configs)
    if not sys.platform.startswith("linux"):
        return None
    for p in stockfish_candidates():
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None
