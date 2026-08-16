"""No TRACKED file may carry an absolute `/home/<user>/...` path.

This repository is PUBLIC. An absolute home path leaks the maintainer's
username and local directory layout into every clone and every diff, and it is
also a plain defect: a default that names one machine's filesystem either fails
outright on any other checkout or -- worse -- silently resolves to something
stale. `docs/REVIEW_BUG_HUNT.md` findings F035/F036/F037 are three separate
rounds of exactly that, each fixed the same way (derive the repo root from the
file's own location, allow an env override), and each time the paths grew back
somewhere else because nothing failed when they did. This is the something.

⚑ THE FILE SET COMES FROM `git ls-files`, not a glob, for the same reason
`test_worker_secret_sources.py` gives: what makes a path disclosed is that it
is TRACKED. A `scripts/**` + `docs/**` glob would have scanned neither the
top-level `TRAINING_LOG.md` nor `CLAUDE.md`, both of which carried hits.

What the pattern deliberately does NOT match
--------------------------------------------
`/home/<user>/...` -- the angle-bracket placeholder. Several prose passages are
*about* hardcoded home paths ("`scripts/monitor_pbt.sh` hardcoded ...") and
their meaning does not survive being rewritten into a repo-relative form, so
they keep the shape and lose only the username. `<` is not in the character
class, so the placeholder reads as documentation rather than as a leak. `~/...`
is likewise fine and is what the ledger's recorded commands now use: it is
shell-expandable, copy-pasteable, and names no user.

Only `/home/` is covered -- not `/Users/` (macOS) or `C:\\Users\\` (Windows).
Neither has ever appeared here, and a pattern with no possible trigger is a
gate that cannot fail; add one when there is something for it to catch.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# A real home directory: `/home/` followed by at least one name character.
# `<` is excluded so the `/home/<user>/` placeholder is not a finding.
_ABS_HOME = re.compile(r"/home/[A-Za-z0-9._-]+")

# Built by concatenation ON PURPOSE. This file is scanned by its own sweep
# (excluding it would leave a hole exactly where the pattern is defined), so a
# literal leak string here would make the guard report itself. Split, the text
# `/home/" + "` cannot match: the character after `/home/` is a quote.
_POSITIVE_CONTROL = "/home/" + "exampleuser" + "/projects/chess/data/syzygy_6"
_PLACEHOLDER_CONTROL = "/home/" + "<user>" + "/projects/chess"

# Paths that still carry an absolute home path, each for a stated reason.
# EXACT names, never a `configs/` prefix rule: a prefix rule would let a NEW
# config land with a fresh leak and stay green.
_ALLOWED: dict[str, str] = {
    # `syzygy_path` and friends point OUTSIDE the checkout (the 151G 6-man
    # tablebases are not in the repo), so there is no repo-relative form, and
    # these files are edited on the LIVE branch while a run re-reads them every
    # iteration -- a merge conflict here reverts live experiments. Scrubbing
    # them is a separate, operator-scheduled change.
    "configs/pbt2_small.yaml": "production config, edited on the live branch",
    "configs/scratch_pc.yaml": "research config, default off",
    "configs/bt4_aurora_asha.yaml": "research config, default off",
    "configs/exp_cat_search.yaml": "research config, default off",
    "configs/exp_categorical_continuous.yaml": "research config, default off",
    "configs/exp_dynamic_relations.yaml": "research config, default off",
    "configs/exp_label_escalation.yaml": "research config, default off",
    "configs/exp_repetition_fix.yaml": "research config, default off",
    "configs/exp_soft_policy_ablation.yaml": "research config, default off",
    "configs/exp_soft_policy_divergent_only.yaml": "research config, default off",
    "configs/exp_threat_planes.yaml": "research config, default off",
    "configs/exp_throughput_views.yaml": "research config, default off",
    "configs/exp_v3_checks.yaml": "research config, default off",
    "configs/exp_v3_passers.yaml": "research config, default off",
    "configs/exp_v3_see.yaml": "research config, default off",
    "configs/exp_v3_xray.yaml": "research config, default off",
    "configs/exp_volatility_search.yaml": "research config, default off",
    # PINNED BY A TEST, so it cannot be scrubbed on its own:
    # `test_param_count.py::test_claude_md_syzygy_pair_matches_the_production_config`
    # requires CLAUDE.md to quote `configs/pbt2_small.yaml`'s `syzygy_path`
    # VERBATIM, and to quote no other pair. The doc is downstream of the config;
    # it goes when the config goes.
    "CLAUDE.md": "must quote the production syzygy_path verbatim (test_param_count)",
}


def _tracked_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return sorted(REPO_ROOT / name for name in listed.split("\0") if name)


def _hits(path: Path) -> list[tuple[int, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []  # binary or unreadable: nothing to leak in text form
    return [
        (lineno, line.strip())
        for lineno, line in enumerate(text.splitlines(), 1)
        if _ABS_HOME.search(line)
    ]


def test_the_pattern_can_actually_fire() -> None:
    """The positive control, so a broken regex cannot pass the sweep silently."""
    assert _ABS_HOME.search(_POSITIVE_CONTROL), "the guard's own pattern matches nothing"
    assert not _ABS_HOME.search(_PLACEHOLDER_CONTROL), (
        "the `/home/<user>/` placeholder must NOT be a finding — the prose that "
        "documents this defect class is written in it"
    )
    # And the split-literal trick above must really be invisible to the sweep,
    # or this file reports itself and every reader learns to ignore the guard.
    assert not _hits(Path(__file__)), (
        "this guard file matches its own pattern; the controls must stay split"
    )


def test_no_tracked_file_carries_an_absolute_home_path() -> None:
    tracked = _tracked_files()
    assert len(tracked) > 100, (
        f"git ls-files returned only {len(tracked)} files — the sweep is vacuous"
    )

    offenders: list[str] = []
    scanned = 0
    for path in tracked:
        rel = str(path.relative_to(REPO_ROOT))
        if rel in _ALLOWED:
            continue
        scanned += 1
        offenders.extend(f"{rel}:{lineno}: {line}" for lineno, line in _hits(path))

    assert scanned > 100, f"only {scanned} files were actually scanned"
    assert not offenders, (
        "these TRACKED files carry an absolute /home/<user> path, in a PUBLIC "
        f"repository: {offenders}. Derive the path from the file's own location "
        "instead — `Path(__file__).resolve().parents[1]` in Python, "
        '`cd \"$(dirname \"$0\")/..\"` in shell — or, if it is prose rather than '
        "a path the code uses, write it `~/...` (a recorded command) or "
        "`/home/<user>/...` (prose about this defect itself)."
    )


def test_every_allowlist_entry_is_still_needed() -> None:
    """An allowlist that outlives its reason is how the next leak stays green.

    Each entry is an IDENTITY pin, not a count: when a config is finally
    scrubbed, this fails and names the line to delete, so the exemption cannot
    quietly become permission for a file that no longer needs it.
    """
    stale = [
        rel for rel in sorted(_ALLOWED)
        if not _hits(REPO_ROOT / rel)
    ]
    assert not stale, (
        f"{stale} no longer contain an absolute home path — delete their entries "
        "from _ALLOWED so the files are covered by the sweep again"
    )

    missing = [rel for rel in sorted(_ALLOWED) if not (REPO_ROOT / rel).is_file()]
    assert not missing, f"_ALLOWED names files that do not exist: {missing}"
