"""Find the Syzygy tablebases this checkout should use — and REFUSE a blind one.

⚑⚑ A MISSING TABLEBASE PATH IS SILENT. Measured against the production binary:

    setoption name SyzygyPath value <a directory that does not exist>
    isready
    info string Found 0 WDL and 0 DTZ tablebase files (up to 0-man).
    readyok                                  # exit 0, no error, no warning

That single `info string` is the entire trace, and in every caller here it goes
into a redirected log nobody reads. A missing ENGINE fails loudly ("no such
file"); a missing TABLEBASE just makes the answers quietly worse. So a gate whose
whole claim is that it out-resolves the in-loop label — `scripts/monitor_fen.sh`'s
harvest gate says it "must dominate the label on every axis ... TB equal" — can
run completely tablebase-blind and report success. That is a value accepted and
then ignored, which is this repository's signature defect.

PR #441 armed it: `monitor_fen.sh`'s `SYZYGY_PATH` was an absolute path (always
right on the one machine) and became `$REPO_ROOT/data/syzygy_*`, which resolves
to nothing in the `git worktree` CLAUDE.md mandates for branch work. Production
is unaffected — the live tree resolves — but every worktree and every fresh
clone silently lost the tablebases.

Two answers here, and they are separate on purpose:

* `default_syzygy_path()` LOOKS IN MORE THAN ONE PLACE, like
  `engine_discovery.default_stockfish()`: `$CAE_SYZYGY_PATH`, then this
  checkout, then the main checkout reached through `--git-common-dir`.
* `require_tablebases()` FAILS LOUDLY when a component holds no tablebases, so
  the caller that cannot honestly claim "TB equal" says so with a non-zero exit
  instead of producing a number.

⚑ THE DIRECTORY NAMES LIE — read the CONTENTS. Per CLAUDE.md, production's
`data/syzygy_3-4-5` actually holds 3-6 man WDL plus 3-5 man DTZ, and the 6-man
DTZ lives in the separate `data/syzygy_6`. So a check that a *name* looks right
proves nothing; every check here counts `.rtbw` / `.rtbz` files.
"""
from __future__ import annotations

import os
from pathlib import Path

from chess_anti_engine.utils.engine_discovery import main_checkout

#: Explicit override, checked first — the same seam as `CAE_STOCKFISH`.
ENV_VAR = "CAE_SYZYGY_PATH"

#: What Stockfish separates `SyzygyPath` components with. Not `os.pathsep`:
#: the engine's own parser is `:` on every platform this runs on.
SEPARATOR = ":"

#: The production pair, relative to a checkout root, in production's order.
#: `configs/pbt2_small.yaml`'s `syzygy_path` is exactly these two joined by
#: `SEPARATOR`, and BOTH halves are local (CLAUDE.md, "Production Syzygy").
SYZYGY_DIRS: tuple[str, ...] = ("data/syzygy_3-4-5", "data/syzygy_6")

#: A tablebase file, by extension: WDL and DTZ.
TABLEBASE_SUFFIXES: tuple[str, ...] = (".rtbw", ".rtbz")

REPO_ROOT = Path(__file__).resolve().parents[2]


def count_tablebases(directory: str | Path, *, limit: int = 1) -> int:
    """How many tablebase files ``directory`` holds, counting at most ``limit``.

    Early-exits: the production directories hold 875 files between them and the
    only question any caller asks is "any at all", so a full listing would be
    pure I/O for nothing.
    """
    found = 0
    try:
        with os.scandir(str(directory)) as entries:
            for entry in entries:
                if entry.name.endswith(TABLEBASE_SUFFIXES) and entry.is_file():
                    found += 1
                    if found >= limit:
                        break
    except OSError:
        return 0
    return found


def has_tablebases(directory: str | Path) -> bool:
    return count_tablebases(directory) > 0


def missing_tablebase_dirs(syzygy_path: str | None) -> list[str]:
    """Components of ``syzygy_path`` that hold no tablebases.

    An empty or absent path returns a single `""` entry rather than `[]`: "no
    path at all" is the blindest case there is, and returning "nothing missing"
    for it is exactly the shape of gate this module exists to stop.
    """
    if not syzygy_path or not syzygy_path.strip():
        return [""]
    return [
        part for part in syzygy_path.split(SEPARATOR)
        if not part.strip() or not has_tablebases(part.strip())
    ]


def require_tablebases(syzygy_path: str | None, *, what: str = "SyzygyPath") -> None:
    """Raise unless every component of ``syzygy_path`` really holds tablebases.

    ⚑ The caller is expected to turn this into a NON-ZERO EXIT. Stockfish will
    happily accept the same value and report `Found 0 WDL and 0 DTZ`.
    """
    missing = missing_tablebase_dirs(syzygy_path)
    if not missing:
        return
    named = ", ".join(repr(m) for m in missing) or "(empty)"
    raise ValueError(
        f"{what} is tablebase-BLIND: {named} hold no .rtbw/.rtbz files. "
        "Stockfish accepts this silently (`info string Found 0 WDL and 0 DTZ`, "
        "then `readyok`), so a gate that claims TB parity with the in-loop "
        f"label would be claiming it falsely. Set ${ENV_VAR} (or pass the path "
        "explicitly) to a checkout that has them, or run from the checkout that "
        "does."
    )


def syzygy_roots(repo_root: Path | None = None) -> list[Path]:
    """Checkout roots to look in, most specific first."""
    root = REPO_ROOT if repo_root is None else repo_root
    roots = [root]
    main = main_checkout(root)
    if main is not None and main != root:
        roots.append(main)
    return roots


def default_syzygy_path(repo_root: Path | None = None) -> str:
    """A `--syzygy-path` default: the production pair, from wherever it exists.

    ⚑ NEVER empty, and never partially silent. Each half is resolved
    independently against the same root order, so a checkout that published only
    one of them still gets the other from the main checkout; a half that is
    nowhere resolves to this checkout's path, which `require_tablebases` then
    names in the failure.
    """
    override = os.environ.get(ENV_VAR, "").strip()
    if override:
        return override
    root = REPO_ROOT if repo_root is None else repo_root
    roots = syzygy_roots(root)
    parts: list[str] = []
    for rel in SYZYGY_DIRS:
        chosen = next((r / rel for r in roots if has_tablebases(r / rel)), root / rel)
        parts.append(str(chosen))
    return SEPARATOR.join(parts)


if __name__ == "__main__":  # `SYZYGY_PATH=$(python3 -m ...utils.syzygy)` in shell
    print(default_syzygy_path())
