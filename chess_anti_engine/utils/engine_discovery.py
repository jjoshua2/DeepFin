"""Find the Stockfish binary this checkout should use.

⚑⚑ THE PUBLISHED ENGINE IS UNTRACKED. `git ls-files e2e_server` returns nothing:
`e2e_server/publish/stockfish` is runtime output of whichever checkout actually
ran the server. So a candidate derived from `__file__` resolves in that ONE tree
and nowhere else — and CLAUDE.md mandates doing branch work in a `git worktree`,
where it therefore resolves to nothing.

That is not hypothetical. PR #441 replaced a hardcoded absolute engine path with
a checkout-relative one, which is right for a public repo and, on its own, turned
`tests/stockfish_binary.find_stockfish()` into a function that returns `None` in
every worktree — silently skipping the 18 end-to-end tests that guard the
selfplay wiring. The fix is not to go back to an absolute path; it is to look in
more than one place:

    $CAE_STOCKFISH  ->  this checkout  ->  the MAIN checkout  ->  PATH  ->  distro

The main checkout is reached through `git rev-parse --git-common-dir`, which
answers with the MAIN repository's `.git` even when called inside a linked
worktree — exactly the lookup needed, since the engine lives where it was
published.

This module is the ONE definition. `tests/stockfish_binary.py` and the
`scripts/blindspot_*` tools both consume it rather than each carrying a copy;
three separate copies of a path constant is how the copies drift.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

#: Explicit override, checked first. The same seam as RATCHET_ROOT /
#: WATCHDOG_ROOT / HARVEST_SF_BIN: a machine that keeps its engine somewhere
#: unusual says so, instead of the caller silently degrading to zero coverage.
ENV_VAR = "CAE_STOCKFISH"

#: Where the server publishes the engine, relative to a checkout root.
PUBLISHED = Path("e2e_server") / "publish" / "stockfish"

#: Distro package locations, tried last. A module constant rather than a literal
#: inside `stockfish_candidates`, so a test that wants a checkout with NO engine
#: anywhere can remove them — mocking `shutil.which` does not, and a host with a
#: distro Stockfish would otherwise discover the real system engine and fail
#: every negative assertion.
DISTRO_CANDIDATES: tuple[str, ...] = ("/usr/bin/stockfish", "/usr/games/stockfish")

REPO_ROOT = Path(__file__).resolve().parents[2]


def main_checkout(repo_root: Path | None = None) -> Path | None:
    """The checkout `repo_root` was created from, or None if it is not a worktree."""
    root = REPO_ROOT if repo_root is None else repo_root
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=root, capture_output=True, text=True, check=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    raw = result.stdout.strip()
    if not raw:
        return None
    common = Path(raw)
    if not common.is_absolute():
        common = (root / common).resolve()
    parent = common.parent
    return parent if parent.is_dir() else None


def stockfish_candidates(repo_root: Path | None = None) -> list[str]:
    """Every place a real engine may be, most specific first.

    Recomputed per call rather than frozen at import: the env override must be
    settable by a test (and by CI) without reloading the module.
    """
    root = REPO_ROOT if repo_root is None else repo_root
    candidates: list[str] = []
    override = os.environ.get(ENV_VAR, "").strip()
    if override:
        candidates.append(override)
    candidates.append(str(root / PUBLISHED))
    main = main_checkout(root)
    if main is not None and main != root:
        candidates.append(str(main / PUBLISHED))
    on_path = shutil.which("stockfish")
    if on_path:
        candidates.append(on_path)
    candidates.extend(DISTRO_CANDIDATES)
    # Order-preserving dedup, so a duplicate candidate is not reported twice
    # when `which` happens to resolve to one of the literals above.
    seen: set[str] = set()
    return [c for c in candidates if not (c in seen or seen.add(c))]


def find_stockfish(repo_root: Path | None = None) -> str | None:
    """The first candidate that is an executable file, or None."""
    # engine-backed callers only run inside WSL/Linux (startswith defeats
    # basedpyright's sys.platform literal narrowing, which would otherwise
    # flag the loop below as unreachable on non-linux check configs)
    if not sys.platform.startswith("linux"):
        return None
    for p in stockfish_candidates(repo_root):
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def default_stockfish(repo_root: Path | None = None) -> str:
    """A `--stockfish` default: the discovered engine, else the checkout path.

    ⚑ NEVER None. An argparse default of None turns "no engine found" into a
    `TypeError` deep inside an engine constructor; falling back to the
    checkout-relative path keeps the failure legible ("no such file: <path>")
    and keeps `--help` output stable and machine-independent.
    """
    root = REPO_ROOT if repo_root is None else repo_root
    return find_stockfish(root) or str(root / PUBLISHED)
