"""No TRACKED file may leak the maintainer's identity or local layout.

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

⚑⚑ THREE SHAPES, NOT ONE. The first revision of this guard matched `/home/<user>`
only, and announced that "the maintainer's absolute home path" was gone -- while
two OTHER shapes of the SAME disclosure stayed tracked and green, because the
pattern structurally could not match them:

* `session_scratch` -- the agent scratchpad root `/tmp/claude-<uid>/-home-<user>-...`.
  The home path is FLATTENED into a single directory name, so there is no `/home/`
  substring anywhere in it. 7 tracked instances (6 in `docs/experiment_ledger.md`,
  1 in `docs/lc0_adapter_probe_run.md`) survived the first sweep and are scrubbed
  to `<session-scratch>/` by the same commit that adds this shape.
* `worker_username` -- `distributed_worker_username: <name>` in 17 configs. Not a
  path at all, so no path pattern will ever see it.

A guard that reports "clean" while naming only the shape it happens to match is
worse than no guard, because it retires the question.

What the patterns deliberately do NOT match
-------------------------------------------
`/home/<user>/...` -- the angle-bracket placeholder. Several prose passages are
*about* hardcoded home paths ("`scripts/monitor_pbt.sh` hardcoded ...") and
their meaning does not survive being rewritten into a repo-relative form, so
they keep the shape and lose only the username. `<` is not in any character
class here, so every placeholder form reads as documentation rather than as a
leak -- including `/tmp/claude-1000/<session-id>/` and
`distributed_worker_username: <name>`. `~/...` is likewise fine and is what the
ledger's recorded commands now use: it is shell-expandable, copy-pasteable, and
names no user.

Only `/home/` is covered -- not `/Users/` (macOS) or `C:\\Users\\` (Windows).
Neither has ever appeared here, and a pattern with no possible trigger is a
gate that cannot fail; add one when there is something for it to catch.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ⚑ BYTES, not str. The first revision read files as UTF-8 text and returned []
# on UnicodeDecodeError, with the comment "binary or unreadable: nothing to leak
# in text form" -- which is untrue of a pickle, an .npz or any container that
# stores paths as ASCII inside a binary envelope, and it exempted every tracked
# binary from the sweep BY CONSTRUCTION. Scanning bytes has no such hole and
# costs nothing.
_PATTERNS: dict[str, re.Pattern[bytes]] = {
    # A real home directory: `/home/` followed by at least one name character.
    "abs_home": re.compile(rb"/home/[A-Za-z0-9._-]+"),
    # The agent-session scratchpad root, where the home path is flattened into
    # one directory name (`/-home-<user>-projects-chess`) and so contains no
    # `/home/` substring. Anchored on the leading `/-home-` to stay specific.
    "session_scratch": re.compile(rb"/-home-[A-Za-z0-9]"),
    # Not a path: the fleet's SSH username, carried by every distributed config.
    "worker_username": re.compile(rb"distributed_worker_username:[ \t]*(?!<)[A-Za-z0-9._-]+"),
}

# Built by concatenation ON PURPOSE. This file is scanned by its own sweep
# (excluding it would leave a hole exactly where the patterns are defined), so a
# literal leak string here would make the guard report itself.
_CONTROLS: dict[str, str] = {
    "abs_home": "/home/" + "exampleuser" + "/projects/chess/data/syzygy_6",
    "session_scratch": "/tmp/claude-1000" + "/-home-" + "exampleuser" + "-projects-chess",
    "worker_username": "distributed_worker_username:" + " exampleuser",
}
_PLACEHOLDER_CONTROLS: dict[str, str] = {
    "abs_home": "/home/" + "<user>" + "/projects/chess",
    "session_scratch": "/tmp/claude-1000/" + "<session-id>" + "/scratchpad",
    "worker_username": "distributed_worker_username:" + " <name>",
}

_LIVE_CONFIG_REASON = "production config, edited on the live branch"
_RESEARCH_REASON = "research config, default off"

_CONFIGS = (
    "configs/pbt2_small.yaml",
    "configs/scratch_pc.yaml",
    "configs/bt4_aurora_asha.yaml",
    "configs/exp_cat_search.yaml",
    "configs/exp_categorical_continuous.yaml",
    "configs/exp_dynamic_relations.yaml",
    "configs/exp_label_escalation.yaml",
    "configs/exp_repetition_fix.yaml",
    "configs/exp_soft_policy_ablation.yaml",
    "configs/exp_soft_policy_divergent_only.yaml",
    "configs/exp_threat_planes.yaml",
    "configs/exp_throughput_views.yaml",
    "configs/exp_v3_checks.yaml",
    "configs/exp_v3_passers.yaml",
    "configs/exp_v3_see.yaml",
    "configs/exp_v3_xray.yaml",
    "configs/exp_volatility_search.yaml",
)

# Paths that still leak, each for a stated reason, EXEMPT PER SHAPE.
#
# ⚑ Per shape, not per file. The first revision skipped an allowlisted file
# WHOLLY, so `configs/pbt2_small.yaml` could grow an unrelated new leak and stay
# green -- an exemption for one known reason silently became an exemption for
# everything. A file listed here is still swept for every shape it does not name.
#
# ⚑ EXACT names, never a `configs/` prefix rule: a prefix rule would let a NEW
# config land with a fresh leak and stay green.
_ALLOWED: dict[str, dict[str, str]] = {
    # `syzygy_path` and friends point OUTSIDE the checkout (the 151G 6-man
    # tablebases are not in the repo), so there is no repo-relative form, and
    # these files are edited on the LIVE branch while a run re-reads them every
    # iteration -- a merge conflict here reverts live experiments. Scrubbing
    # them is a separate, operator-scheduled change.
    #
    # `distributed_worker_username` is the SSH login the fleet actually uses.
    # It is a real value, not a path, so there is no derived form either; it
    # goes when these configs are scrubbed, in the same operator-scheduled
    # change, and `pbt2_small.yaml` in particular is RESTART-GATED.
    **{
        rel: {
            "abs_home": _LIVE_CONFIG_REASON if rel == "configs/pbt2_small.yaml" else _RESEARCH_REASON,
            "worker_username": "fleet SSH login; scrubbed with the config, restart-gated",
        }
        for rel in _CONFIGS
    },
    # PINNED BY A TEST, so it cannot be scrubbed on its own:
    # `test_param_count.py::test_claude_md_syzygy_pair_matches_the_production_config`
    # requires CLAUDE.md to quote `configs/pbt2_small.yaml`'s `syzygy_path`
    # VERBATIM, and to quote no other pair. The doc is downstream of the config;
    # it goes when the config goes.
    "CLAUDE.md": {"abs_home": "must quote the production syzygy_path verbatim (test_param_count)"},
}


def _tracked_files() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return sorted(REPO_ROOT / name for name in listed.split("\0") if name)


def _hits(path: Path, shapes: frozenset[str] | None = None) -> list[tuple[str, int, str]]:
    """(shape, line number, line) for every leak in ``path``.

    Reads BYTES so a binary file is scanned rather than exempted; the offending
    line is decoded only to render the failure message.
    """
    try:
        blob = path.read_bytes()
    except OSError:
        return []
    wanted = _PATTERNS if shapes is None else {k: v for k, v in _PATTERNS.items() if k in shapes}
    found: list[tuple[str, int, str]] = []
    for lineno, line in enumerate(blob.split(b"\n"), 1):
        for shape, pattern in wanted.items():
            if pattern.search(line):
                found.append((shape, lineno, line.decode("utf-8", "replace").strip()[:200]))
    return found


def test_every_pattern_can_actually_fire() -> None:
    """Positive AND placeholder controls, per shape.

    A shape with no control is a shape nobody has proved can match -- which is
    how `session_scratch` and `worker_username` went unnoticed in the first
    place.
    """
    assert set(_CONTROLS) == set(_PATTERNS) == set(_PLACEHOLDER_CONTROLS), (
        "every shape needs both a positive and a placeholder control"
    )
    for shape, pattern in _PATTERNS.items():
        assert pattern.search(_CONTROLS[shape].encode()), (
            f"the {shape!r} pattern matches nothing"
        )
        assert not pattern.search(_PLACEHOLDER_CONTROLS[shape].encode()), (
            f"the {shape!r} placeholder must NOT be a finding — the prose that "
            "documents this defect class is written in it"
        )
    # And the split-literal trick above must really be invisible to the sweep,
    # or this file reports itself and every reader learns to ignore the guard.
    assert not _hits(Path(__file__)), (
        "this guard file matches its own patterns; the controls must stay split"
    )


def test_a_binary_file_is_scanned_not_exempted(tmp_path: Path) -> None:
    """The UTF-8 hole, closed and pinned.

    A pickle or .npz stores ASCII paths inside a binary envelope. Reading as
    text and swallowing UnicodeDecodeError exempted every such file BY
    CONSTRUCTION while the docstring claimed there was "nothing to leak in text
    form".
    """
    blob = tmp_path / "fixture.npz"
    blob.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe" + _CONTROLS["abs_home"].encode() + b"\x00\xff")
    hits = _hits(blob)
    assert [shape for shape, _, _ in hits] == ["abs_home"], hits


def test_no_tracked_file_leaks() -> None:
    tracked = _tracked_files()
    assert len(tracked) > 100, (
        f"git ls-files returned only {len(tracked)} files — the sweep is vacuous"
    )

    offenders: list[str] = []
    scanned = 0
    for path in tracked:
        rel = str(path.relative_to(REPO_ROOT))
        exempt = frozenset(_ALLOWED.get(rel, {}))
        shapes = frozenset(_PATTERNS) - exempt
        if not shapes:
            continue
        scanned += 1
        offenders.extend(
            f"{rel}:{lineno}: [{shape}] {line}" for shape, lineno, line in _hits(path, shapes)
        )

    assert scanned > 100, f"only {scanned} files were actually scanned"
    assert not offenders, (
        "these TRACKED files leak the maintainer's identity or local layout, in "
        f"a PUBLIC repository: {offenders}. Derive paths from the file's own "
        "location instead — `Path(__file__).resolve().parents[1]` in Python, "
        '`cd \"$(dirname \"$0\")/..\"` in shell — or, if it is prose rather than '
        "a path the code uses, write it `~/...` (a recorded command), "
        "`<session-scratch>/...` (an agent scratchpad bank) or "
        "`/home/<user>/...` (prose about this defect itself)."
    )


def test_every_allowlist_entry_is_still_needed() -> None:
    """An allowlist that outlives its reason is how the next leak stays green.

    Each entry is an IDENTITY pin PER SHAPE, not a count: when a config is
    finally scrubbed, this fails and names the exact line to delete, so the
    exemption cannot quietly become permission for a file that no longer needs
    it -- nor for a shape it never needed.
    """
    stale: list[str] = []
    for rel, shapes in sorted(_ALLOWED.items()):
        present = {shape for shape, _, _ in _hits(REPO_ROOT / rel)}
        stale.extend(f"{rel}[{shape}]" for shape in sorted(shapes) if shape not in present)
    assert not stale, (
        f"{stale} no longer leak that shape — delete those entries from _ALLOWED "
        "so the files are covered by the sweep again"
    )

    missing = [rel for rel in sorted(_ALLOWED) if not (REPO_ROOT / rel).is_file()]
    assert not missing, f"_ALLOWED names files that do not exist: {missing}"


def test_an_allowlisted_file_is_still_swept_for_other_shapes(tmp_path: Path) -> None:
    """The second structural gap, pinned.

    `configs/pbt2_small.yaml` is exempt for two shapes and must remain covered
    for the third, or the exemption is a licence rather than an exception.
    """
    exempt = frozenset(_ALLOWED["configs/pbt2_small.yaml"])
    assert frozenset(_PATTERNS) - exempt, (
        "pbt2_small.yaml is exempt from EVERY shape — it is no longer swept at all"
    )
    poisoned = tmp_path / "pbt2_small.yaml"
    poisoned.write_bytes(_CONTROLS["session_scratch"].encode())
    assert _hits(poisoned, frozenset(_PATTERNS) - exempt), (
        "a new leak of a non-exempt shape in an allowlisted file must still be found"
    )
