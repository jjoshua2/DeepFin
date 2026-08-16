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

import os
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
    # one directory name (`-home-<user>-projects-chess`) and so contains no
    # `/home/` substring.
    #
    # ⚑ NOT anchored on a leading `/`. The first revision was (`/-home-`), and
    # the reviewer got the bare token past it: `-home-<user>-projects-chess`
    # with no slash in front is exactly how the flattened directory NAME appears
    # in `ls`, `tar` and `rsync` output and in prose that names a scratchpad
    # bank. The boundary is "not preceded by a name character", which still
    # refuses to fire inside an ordinary hyphenated word (`multi-home-server`).
    "session_scratch": re.compile(rb"(?<![A-Za-z0-9])-home-[A-Za-z0-9]"),
    # Not a path: the fleet's SSH username, carried by every distributed config.
    #
    # ⚑⚑ THE QUOTES ARE THE WHOLE POINT. The first revision opened its character
    # class immediately after the whitespace, so a QUOTED yaml scalar never
    # matched — `"` is not `<`, so the negative lookahead passed, and then
    # `[A-Za-z0-9._-]` failed on the quote. MEASURED by the independent reviewer:
    # a new tracked config carrying `distributed_worker_username: "<the real
    # login>"` swept GREEN, while the same file unquoted went red. YAML authors
    # habitually quote strings, so the shape most likely to land was the one
    # shape the guard could not see.
    #
    # The optional quote sits BEFORE the `(?!<)` lookahead, so the documentation
    # placeholder stays exempt whether it is written `<name>` or `"<name>"` —
    # the lookahead is applied to the first character of the VALUE, not to the
    # quote. The key may also be quoted (a JSON dump of the same config) and may
    # carry spaces before its colon (`key : value` is legal YAML).
    # The value is CAPTURED so it can be checked against `_NEUTRAL_USERNAMES`
    # below; the trailing lookahead keeps a Python EXPRESSION (`str(username)`
    # in `tune/harness.py`, which names a variable rather than a person) from
    # reading as a scalar.
    "worker_username": re.compile(
        rb"distributed_worker_username[\"']?[ \t]*:[ \t]*"
        rb"[\"']?(?!<)([A-Za-z0-9._-]+)(?![A-Za-z0-9._(-])",
    ),
}

#: Login names that are NOT anybody's account — test fixtures and the guard's
#: own controls.
#:
#: ⚑ DENY BY DEFAULT. The shape is a finding unless its value is named here, so
#: a config carrying a real login is red without anyone having to predict it.
#: The inverse — permit everything, list the bad names — cannot be written in a
#: public repo at all, because the list would BE the disclosure.
#:
#: ⚑ This list is what caught the leak the widened pattern was written for:
#: `tests/test_worker_secret_sources.py` carried the maintainer's actual login
#: in seven tracked places, and the narrower pre-review pattern matched none of
#: them. Scrubbed to `fleetuser` in the same commit.
#:
#: Limit, stated: a real account that happens to be called `worker` would pass.
#: That is the price of a list that can be written down publicly.
_NEUTRAL_USERNAMES: frozenset[bytes] = frozenset({
    b"worker", b"tune_worker_old", b"fleetuser", b"exampleuser",
})

# Built by concatenation ON PURPOSE. This file is scanned by its own sweep
# (excluding it would leave a hole exactly where the patterns are defined), so a
# literal leak string here would make the guard report itself.
_CONTROLS: dict[str, str] = {
    "abs_home": "/home/" + "exampleuser" + "/projects/chess/data/syzygy_6",
    "session_scratch": "/tmp/claude-1000" + "/-home-" + "exampleuser" + "-projects-chess",
    "worker_username": "distributed_worker_username:" + " exampleuser",
}

#: ⚑ EVERY SHAPE THE LEAK CAN TAKE, not just the one shape someone happened to
#: write. A control proves a pattern CAN fire; these prove it fires on the
#: variants a human actually produces. Each entry below was GREEN under some
#: earlier revision of the pattern above:
#:   - the quoted username swept green past the whole guard (review P1);
#:   - the bare flattened scratchpad token swept green past it too (review P2).
_VARIANT_CONTROLS: dict[str, tuple[str, ...]] = {
    "abs_home": (
        "/home/" + "exampleuser",
        "cd /home/" + "exampleuser" + "/projects/chess",
        '"/home/' + 'exampleuser' + '/projects/chess/data"',
    ),
    "session_scratch": (
        "/tmp/claude-1000" + "/-home-" + "exampleuser" + "-projects-chess",
        # No leading slash: how the flattened directory NAME appears bare.
        "-home-" + "exampleuser" + "-projects-chess/scratchpad/x.json",
        "see " + "-home-" + "exampleuser" + "-projects-chess",
    ),
    "worker_username": (
        "distributed_worker_username:" + " exampleuser",
        'distributed_worker_username:' + ' "exampleuser"',      # ← the P1 hole
        "distributed_worker_username:" + " 'exampleuser'",
        "distributed_worker_username:" + "exampleuser",
        "distributed_worker_username" + " : exampleuser",
        '"distributed_worker_username":' + ' "exampleuser"',     # JSON dump
    ),
}

_PLACEHOLDER_CONTROLS: dict[str, tuple[str, ...]] = {
    "abs_home": ("/home/" + "<user>" + "/projects/chess",),
    "session_scratch": (
        "/tmp/claude-1000/" + "<session-id>" + "/scratchpad",
        # An ordinary hyphenated word must not become a finding.
        "multi-home-server", "the -homeostasis- note",
    ),
    "worker_username": (
        "distributed_worker_username:" + " <name>",
        # ⚑ The QUOTED placeholder too: prose about this defect gets quoted the
        # same way real yaml does, and documentation must not read as a leak.
        'distributed_worker_username:' + ' "<name>"',
        "distributed_worker_username:" + " '<name>'",
    ),
}

# ⚑⚑ AN EXEMPTION MUST ARGUE INFEASIBILITY, NOT UNIMPORTANCE. Until the #441
# review, 16 of these 17 read "research config, default off" — which is a reason
# nobody would MIND the leak, not a reason it cannot be removed, and it would
# have justified exempting anything anyone did not care about. The real blocker
# is below, and it applies to all 17 equally.
_YAML_HAS_NO_SELF_LOCATION = (
    "yaml cannot locate itself: every functional scrub in this repo works by "
    "deriving the root from __file__ / $0, and a config has neither. The "
    "alternatives are a relative path (meaning then depends on the loader's cwd "
    "— unproven against TrialConfig.from_dict and the tune harness) or ~ (which "
    "the loader does not expand). Several paths also point OUTSIDE the checkout "
    "(syzygy_path's 151G 6-man tablebases) and have no repo-relative form at all"
)
_LIVE_CONFIG_REASON = (
    f"{_YAML_HAS_NO_SELF_LOCATION}; AND this one is production, re-read every "
    "iteration on the live branch, and key changes are restart-gated"
)
_RESEARCH_REASON = _YAML_HAS_NO_SELF_LOCATION

# (path, number of `abs_home` hits present when the exemption was written)
_CONFIGS = (
    ("configs/pbt2_small.yaml", 11),
    ("configs/scratch_pc.yaml", 10),
    ("configs/bt4_aurora_asha.yaml", 9),
    ("configs/exp_cat_search.yaml", 9),
    ("configs/exp_categorical_continuous.yaml", 9),
    ("configs/exp_dynamic_relations.yaml", 9),
    ("configs/exp_label_escalation.yaml", 11),
    ("configs/exp_repetition_fix.yaml", 9),
    ("configs/exp_soft_policy_ablation.yaml", 9),
    ("configs/exp_soft_policy_divergent_only.yaml", 9),
    ("configs/exp_threat_planes.yaml", 9),
    ("configs/exp_throughput_views.yaml", 9),
    ("configs/exp_v3_checks.yaml", 9),
    ("configs/exp_v3_passers.yaml", 9),
    ("configs/exp_v3_see.yaml", 9),
    ("configs/exp_v3_xray.yaml", 9),
    ("configs/exp_volatility_search.yaml", 9),
)

# Paths that still leak, each for a stated reason, EXEMPT PER SHAPE AND PER COUNT.
#
# ⚑ Per shape, not per file. The first revision skipped an allowlisted file
# WHOLLY, so `configs/pbt2_small.yaml` could grow an unrelated new leak and stay
# green -- an exemption for one known reason silently became an exemption for
# everything. A file listed here is still swept for every shape it does not name.
#
# ⚑⚑ AND PER COUNT, since the #441 review. Exempting the SHAPE was still too
# coarse: it subtracted the whole detector before the file was scanned, so a
# config could gain a SECOND, unrelated absolute path -- or a fleet login where
# it previously had none -- and stay green for as long as one old exempt
# occurrence survived. These 17 configs are the files most likely to acquire a
# fresh machine-specific path, which made them the worst possible place to turn
# the detector off. The number below is the number of hits that were there when
# the exemption was written; hit N+1 is a finding, and hit N-1 fails the
# staleness pin. (Codex inline review.)
#
# ⚑ EXACT names, never a `configs/` prefix rule: a prefix rule would let a NEW
# config land with a fresh leak and stay green.
_ALLOWED: dict[str, dict[str, tuple[int, str]]] = {
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
            "abs_home": (
                n_abs_home,
                _LIVE_CONFIG_REASON if rel == "configs/pbt2_small.yaml" else _RESEARCH_REASON,
            ),
            "worker_username": (
                1, "fleet SSH login; scrubbed with the config, restart-gated",
            ),
        }
        for rel, n_abs_home in _CONFIGS
    },
    # PINNED BY A TEST, so it cannot be scrubbed on its own:
    # `test_param_count.py::test_claude_md_syzygy_pair_matches_the_production_config`
    # requires CLAUDE.md to quote `configs/pbt2_small.yaml`'s `syzygy_path`
    # VERBATIM, and to quote no other pair. The doc is downstream of the config;
    # it goes when the config goes.
    "CLAUDE.md": {
        "abs_home": (1, "must quote the production syzygy_path verbatim (test_param_count)"),
    },
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

    ⚑ A TRACKED SYMLINK IS ITS TARGET STRING, not the file it points at. Git
    stores the target verbatim in the tree, so `ln -s /home/<user>/private x`
    publishes that path to every clone — while `read_bytes()` FOLLOWS the link
    and scans something else entirely (or raises, and the `except OSError`
    below returned no hits, so the disclosure swept green). `os.readlink` reads
    what git actually stored. Codex inline review, #441.
    """
    if path.is_symlink():
        try:
            blob = os.readlink(path).encode("utf-8", "surrogateescape")
        except OSError:
            return []
    else:
        try:
            blob = path.read_bytes()
        except OSError:
            return []
    wanted = _PATTERNS if shapes is None else {k: v for k, v in _PATTERNS.items() if k in shapes}
    found: list[tuple[str, int, str]] = []
    for lineno, line in enumerate(blob.split(b"\n"), 1):
        for shape, pattern in wanted.items():
            match = pattern.search(line)
            if match is None:
                continue
            if shape == "worker_username" and match.group(1) in _NEUTRAL_USERNAMES:
                continue
            found.append((shape, lineno, line.decode("utf-8", "replace").strip()[:200]))
    return found


def _offenders(paths: list[Path], *, root: Path = REPO_ROOT) -> list[str]:
    """Leaks in ``paths`` that the allowlist does not budget for.

    ⚑ EVERY shape is scanned on EVERY file, including allowlisted ones. The
    exemption is applied to the HITS afterwards, as a BUDGET, not to the
    detector beforehand as an off switch — see `_ALLOWED`. Factored out of the
    sweep so the budget arithmetic itself can be tested on a fixture rather than
    only on a tree that is (correctly) clean.
    """
    offenders: list[str] = []
    for path in paths:
        rel = str(path.relative_to(root))
        allowed = _ALLOWED.get(rel, {})
        hits = _hits(path)
        for shape in sorted(_PATTERNS):
            of_shape = [(lineno, line) for sh, lineno, line in hits if sh == shape]
            budget = allowed.get(shape, (0, ""))[0]
            if len(of_shape) <= budget:
                continue
            # Which specific lines are the NEW ones cannot be known from the
            # count alone, so the whole set is named with both numbers and the
            # operator reads the diff.
            offenders.append(
                f"{rel}: [{shape}] {len(of_shape)} hits, {budget} allowed — "
                f"e.g. line {of_shape[-1][0]}: {of_shape[-1][1]}"
            )
    return offenders


def test_every_pattern_can_actually_fire() -> None:
    """Positive AND placeholder controls, per shape.

    A shape with no control is a shape nobody has proved can match -- which is
    how `session_scratch` and `worker_username` went unnoticed in the first
    place.
    """
    assert (set(_CONTROLS) == set(_PATTERNS) == set(_PLACEHOLDER_CONTROLS)
            == set(_VARIANT_CONTROLS)), (
        "every shape needs a positive control, its realistic variants, and a "
        "placeholder control"
    )
    for shape, pattern in _PATTERNS.items():
        assert pattern.search(_CONTROLS[shape].encode()), (
            f"the {shape!r} pattern matches nothing"
        )
        for variant in _VARIANT_CONTROLS[shape]:
            assert pattern.search(variant.encode()), (
                f"the {shape!r} pattern misses a REALISTIC form of the same "
                f"leak: {variant!r}. A leak that only the tidiest spelling "
                "trips is a guard that the untidy spelling walks past."
            )
        for placeholder in _PLACEHOLDER_CONTROLS[shape]:
            assert not pattern.search(placeholder.encode()), (
                f"the {shape!r} placeholder {placeholder!r} must NOT be a "
                "finding — the prose that documents this defect class is "
                "written in it"
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

    offenders = _offenders(tracked)
    scanned = len(tracked)
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

    Each entry is an EXACT-COUNT pin PER SHAPE: when a config is partly or
    wholly scrubbed, this fails and names the number to correct, so the
    exemption cannot quietly become permission for a file that no longer needs
    it -- nor for a shape it never needed, nor for MORE hits than it had.

    ⚑ The over-budget direction is enforced by `test_no_tracked_file_leaks`;
    this is the under-budget half. Both are needed: an entry that is too LARGE
    is a hole, an entry that outlives its reason is clutter that the next
    reader trusts.
    """
    stale: list[str] = []
    for rel, shapes in sorted(_ALLOWED.items()):
        hits = _hits(REPO_ROOT / rel)
        for shape, (budget, _reason) in sorted(shapes.items()):
            n = sum(1 for sh, _, _ in hits if sh == shape)
            if n < budget:
                stale.append(f"{rel}[{shape}]: {n} hits remain, {budget} exempted")
    assert not stale, (
        f"{stale} — the exemption is now larger than the leak. Lower the count "
        "(or delete the entry) so the file is covered by the sweep again"
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


def test_one_more_hit_in_an_allowlisted_file_is_still_a_finding(tmp_path: Path) -> None:
    """⚑ The Codex finding, pinned: the exemption is a BUDGET, not an off switch.

    While `_ALLOWED` subtracted the whole detector, the 17 exempt configs — the
    files most likely to acquire a fresh machine-specific path — could gain any
    number of NEW absolute paths and stay green for as long as one old exempt
    occurrence survived. `test_every_allowlist_entry_is_still_needed` could not
    see it either: it only asked whether at least one hit of the shape remained.
    """
    rel = "configs/pbt2_small.yaml"
    budget, _reason = _ALLOWED[rel]["abs_home"]
    real = (REPO_ROOT / rel).read_bytes()
    fixture_root = tmp_path / "root"
    (fixture_root / "configs").mkdir(parents=True)
    poisoned = fixture_root / rel

    # Exactly the budget: green, or the sweep would fail on the real tree.
    poisoned.write_bytes(real)
    assert not _offenders([poisoned], root=fixture_root)

    # One more, and only one more: red.
    poisoned.write_bytes(real + b"\nsome_new_key: " + _CONTROLS["abs_home"].encode() + b"\n")
    found = _offenders([poisoned], root=fixture_root)
    assert found, (
        f"a {budget + 1}th abs_home hit in an allowlisted config was not a "
        "finding — the exemption is an off switch again"
    )
    assert f"{budget + 1} hits, {budget} allowed" in found[0], found


def test_a_tracked_symlink_is_scanned_as_its_TARGET_STRING(tmp_path: Path) -> None:
    """⚑ Codex finding: git stores the target verbatim, so the target IS the leak.

    `Path.read_bytes()` follows the link. If the target does not exist on the
    machine running the sweep — which is the whole point of a leaked absolute
    home path — the `OSError` branch returned no hits and the disclosure swept
    green. If it DOES exist, some unrelated file's contents were scanned
    instead, which is no better.
    """
    target = _CONTROLS["abs_home"]            # a path that does not exist here
    link = tmp_path / "some_link"
    link.symlink_to(target)
    assert not link.exists(), "fixture broken: the target must be absent"
    hits = _hits(link)
    assert [shape for shape, _, _ in hits] == ["abs_home"], hits
    assert target in hits[0][2]
