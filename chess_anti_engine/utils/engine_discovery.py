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

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TextIO

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
#:
#: `/usr/local/bin` is here because `scripts/bench_vs_sf.py` carried its own
#: private candidate list including that path; folding it in is what made
#: deleting that copy lossless.
DISTRO_CANDIDATES: tuple[str, ...] = (
    "/usr/bin/stockfish", "/usr/games/stockfish", "/usr/local/bin/stockfish",
)

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


#: Source labels for `stockfish_sources`, in the order they are tried. These are
#: the strings that reach an artifact, so they are part of the record format:
#: `env` and `checkout` mean the caller got what it asked for, `main_checkout`
#: means the worktree fallback fired, and `path` / `distro` mean **a binary this
#: repository did not publish**.
SOURCE_ENV = "env"
SOURCE_CHECKOUT = "checkout"
SOURCE_MAIN_CHECKOUT = "main_checkout"
SOURCE_PATH = "path"
SOURCE_DISTRO = "distro"
SOURCE_MISSING = "missing"

#: Resolutions that are NOT the engine this checkout publishes. Reaching one of
#: these is a silent substitution unless something says so out loud — see
#: `announce_engine`.
SUBSTITUTED_SOURCES: frozenset[str] = frozenset({SOURCE_PATH, SOURCE_DISTRO})


def stockfish_sources(repo_root: Path | None = None) -> list[tuple[str, str]]:
    """`(path, source)` for every place a real engine may be, most specific first.

    Recomputed per call rather than frozen at import: the env override must be
    settable by a test (and by CI) without reloading the module.

    ⚑ The SOURCE is the half that matters and the half that used to be thrown
    away. `find_stockfish` returning `/usr/games/stockfish` and returning
    `<checkout>/e2e_server/publish/stockfish` are the same type and the same
    shape, and the deep-SF label tools recorded neither — so a distro engine of
    a different version could relabel the audit set and the artifact would look
    identical. (#441 review N3; same shape as the recorded burn where an SF
    cache key omitted engine identity.)
    """
    root = REPO_ROOT if repo_root is None else repo_root
    candidates: list[tuple[str, str]] = []
    override = os.environ.get(ENV_VAR, "").strip()
    if override:
        candidates.append((override, SOURCE_ENV))
    candidates.append((str(root / PUBLISHED), SOURCE_CHECKOUT))
    main = main_checkout(root)
    if main is not None and main != root:
        candidates.append((str(main / PUBLISHED), SOURCE_MAIN_CHECKOUT))
    on_path = shutil.which("stockfish")
    if on_path:
        candidates.append((on_path, SOURCE_PATH))
    candidates.extend((p, SOURCE_DISTRO) for p in DISTRO_CANDIDATES)
    # Order-preserving dedup, so a duplicate candidate is not reported twice
    # when `which` happens to resolve to one of the literals above. The FIRST
    # label wins, which is what keeps a distro path that is also on `$PATH`
    # attributed to the more specific source.
    seen: set[str] = set()
    return [(p, s) for p, s in candidates if not (p in seen or seen.add(p))]


def stockfish_candidates(repo_root: Path | None = None) -> list[str]:
    """Every place a real engine may be, most specific first (paths only)."""
    return [p for p, _ in stockfish_sources(repo_root)]


def resolve_stockfish(repo_root: Path | None = None) -> tuple[str | None, str]:
    """The first executable candidate and WHICH source produced it.

    Returns `(None, "missing")` when nothing resolves, so a caller can tell
    "not found" apart from "found somewhere unexpected" — two outcomes that
    `find_stockfish`'s `str | None` collapses into one.
    """
    # engine-backed callers only run inside WSL/Linux (startswith defeats
    # basedpyright's sys.platform literal narrowing, which would otherwise
    # flag the loop below as unreachable on non-linux check configs)
    if not sys.platform.startswith("linux"):
        return None, SOURCE_MISSING
    for p, source in stockfish_sources(repo_root):
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p, source
    return None, SOURCE_MISSING


def find_stockfish(repo_root: Path | None = None) -> str | None:
    """The first candidate that is an executable file, or None."""
    return resolve_stockfish(repo_root)[0]


def default_stockfish(repo_root: Path | None = None) -> str:
    """A `--stockfish` default: the discovered engine, else the checkout path.

    ⚑ NEVER None. An argparse default of None turns "no engine found" into a
    `TypeError` deep inside an engine constructor; falling back to the
    checkout-relative path keeps the failure legible ("no such file: <path>")
    and keeps `--help` output stable and machine-independent.
    """
    root = REPO_ROOT if repo_root is None else repo_root
    return find_stockfish(root) or str(root / PUBLISHED)


def engine_identity(path: str | None, repo_root: Path | None = None) -> dict[str, object]:
    """A record of WHICH binary a run used, for the artifact it produces.

    The path alone is not an identity — the same path holds a different engine
    after a rebuild, and that is exactly how a stale cache key produced wrong
    labels here before. So the record carries a content hash as well, cheap
    against the multi-minute deep-SF analysis that follows it (one ~50MB
    sha256, ~0.1s, once per run).

    ⚑ NEVER RAISES. A tool that cannot hash its engine must still produce its
    result; the record says `sha256: null` and the reader knows why, which is
    strictly more than the nothing that was recorded before.
    """
    record: dict[str, object] = {
        "path": path,
        "source": SOURCE_MISSING,
        "sha256": None,
        "size": None,
    }
    if not path:
        return record
    known = dict(stockfish_sources(repo_root))
    record["source"] = known.get(path, "explicit")
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        record["sha256"] = digest.hexdigest()
        record["size"] = os.path.getsize(path)
    except OSError:
        pass
    return record


def announce_engine(
    tag: str,
    path: str | None,
    repo_root: Path | None = None,
    *,
    stream: TextIO | None = None,
) -> dict[str, object]:
    """Print which engine was resolved, and return the record to store with it.

    ⚑ EVERY engine-backed tool calls this, not just the one that happened to
    have a `print`. Before #441 review N3, `blindspot_deepsf_calibrate.py`
    printed `stockfish=...` and `blindspot_deepsf_gate.py`,
    `blindspot_netside_vet.py` and `blindspot_value_gap.py` — all three of them
    deep-SF *label and gate* tools — printed nothing and stored nothing. The
    binary was accepted and then silently ignored, which is this repository's
    signature defect.

    A substituted source (`path`/`distro`) is called out explicitly, because
    the pre-#441 literal had exactly ONE possible value and the derived default
    does not.
    """
    out = sys.stderr if stream is None else stream
    record = engine_identity(path, repo_root)
    source = str(record["source"])
    sha = record["sha256"]
    short = str(sha)[:12] if sha else "unhashed"
    print(f"[{tag}] stockfish={path} source={source} sha256={short}", file=out)
    if source in SUBSTITUTED_SOURCES:
        print(
            f"[{tag}] ⚑ this engine is NOT the one this checkout publishes "
            f"({PUBLISHED}); it came from {source}. Deep-SF labels and gate "
            f"verdicts depend on the engine — set {ENV_VAR} to pin it.",
            file=out,
        )
    return record


def write_engine_record(out_path: str | Path, record: dict[str, object]) -> Path:
    """Store ``record`` beside an artifact as ``<artifact>.engine.json``.

    A SIDECAR rather than a row inside the artifact: these tools write JSONL
    whose rows are consumed positionally by downstream scripts, and adding a
    heterogeneous header row would break them. The sidecar cannot break any
    existing reader and is what makes the artifact self-describing.
    """
    side = Path(str(out_path) + ".engine.json")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return side


if __name__ == "__main__":  # `SF_BIN=$(python3 -m ...utils.engine_discovery)` in shell
    print(default_stockfish())
