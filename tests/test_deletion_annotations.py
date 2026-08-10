"""A `# DELETED <key>` annotation that cites a blank line is worse than no annotation.

When an inert key is removed from a config, the `# DELETED <key>: <reason>
(file:line)` comment left in its place is the ONLY surviving record of why the
removal was safe. It is not decoration — it is the artifact a future reader
follows before re-adding the key, and CLAUDE.md's standing bias applies to it
directly: documentation that says something the code does not.

These citations rot fast, and that is measured rather than assumed:

* 18 of 24 citations in the first revision of this annotation block pointed at
  unrelated code — blank lines, SODA weight-decay grouping, `_build_pb2` where
  GPBT was meant — because they were written against one revision of the tree
  and shipped against another.
* `train/trainer.py` moved by 19 lines between two revisions of `main` whose
  `configs/pbt2_small.yaml` was byte-identical, so a config-only diff gives no
  hint that its own citations have gone stale.

So the rot happens with no edit to the annotated file at all, which means no
review of the config can catch it. This test is the guard: it re-derives every
citation against the tree and requires the code to actually say what the
annotation claims. It reads the whole annotation block, not a diff, so it keeps
working long after the PR that introduced the block has merged.

The count assertions matter as much as the content check: without them a
citation could be quietly deleted — or an annotation dropped whole — and every
remaining citation would still pass.

"The citation resolves" and "the claim is true" are different guarantees, and
the first does not imply the second: three annotations shipped
``DEAD KEY - repo-wide grep outside tests finds only the allowlist`` while
``run.py`` defined a CLI flag for each, and every one of those citations
resolved cleanly because the allowlist line does mention the key. So the
``DEAD KEY`` claim — the strongest claim any annotation here makes — is
re-derived rather than checked for self-consistency:
``test_dead_key_annotations_cite_every_surface_that_mentions_the_key`` runs the
grep itself, in **both** spellings, and fails on any mention the annotation does
not already name. Searching only the source spelling is what produced the false
statements; argparse renames ``foo_bar`` to ``--foo-bar``.

Citations name a FILE, not a line, and that is deliberate
--------------------------------------------------------
A citation is a repo path in square brackets — ``[tune/trial_config.py]``. A
path written without brackets is prose, not a claim, which is what lets
``bootstrap_dir`` say "NOT run.py's argparse-resolved config" without asserting
that ``run.py`` reads the key. The line number is resolved here, at test time,
by searching the cited file for the token the annotation is about; the check
fails when that token is ABSENT from the file, which is the condition that means
the claim died.

An earlier revision pinned ``file:line`` and was measured to be unusable. Taking
the citation set that was exactly correct at ``ef401b93f`` and replaying it
backwards over ``main``: 40/46 resolve one commit earlier, 28/46 five commits
earlier, 21/46 after three days, 9/46 after five. 171 of ``main``'s last 852
commits touch a cited file. A full set of line numbers is therefore correct only
at the single commit it was written against, and ``main`` moves ~14 commits a
day — so the guard was red essentially always, on PRs whose authors did not
cause it and could not read it. Worse, it made the LIVE production config the
thing every unrelated PR had to edit to get green, and CLAUDE.md makes edits to
that file restart-gated and run-fatal when they go wrong. A guard that routinely
demands edits to the live training config to stay green is worse than the rot it
detects. Content pinning keeps every property the guard is for — a renamed key,
a wrong file, a missed surface, and a deleted reader *whose key name goes with
it* all still fail — while drift alone no longer fails anything.

The bound on "a deleted reader still fails" is exact and worth stating, because
the unbounded version is false: what fails is the key's NAME vanishing from the
cited file. Delete the reader but leave the key named anywhere else in that file
— a historical comment, a docstring, an unrelated dict literal — and this check
stays green where a line pin would have gone red. That is the residual price of
dropping line numbers, alongside the one-line-off case below.

The consequence to know: a one-line-off citation and a citation that has merely
drifted are formally the same observation, so nothing here can distinguish them.
Line-level precision is instead recovered where it carries the real claim, by
``DEAD_KEY_EXPECTED_MENTIONS`` — see that pin.

What this file does NOT verify
------------------------------
Stated so the coverage is not overread:

* the annotation's REASON. Every surface being cited is not the sentence about
  that surface being true; a false prose reason built from correct citations
  passes here.
* the 24 annotations that carry no ``DEAD KEY`` string. They get the
  citation-content check only, never the completeness re-derivation. This hole
  shipped for real: ``lr_T0``/``lr_T_mult`` were annotated "read ONLY by the
  cosine branch, unreachable at ``lr_schedule: sqrt_release``" while
  ``scripts/train_bootstrap.py`` read them off the yaml flat dict and never
  forwarded ``lr_schedule``, so the cosine branch was exactly what ran there.
  Both keys are RETAINED rather than deleted as a result.
* claims phrased without the literal token ``DEAD KEY``. The completeness check
  keys off that string, so the identical claim in other words is not re-derived.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# A citation may name something OTHER than the annotated key in exactly two
# situations: it points at the GATE that makes the key inert (a gate is a
# different key by construction), or the cited line spells the key differently
# from the yaml (argparse hyphens, a dropped prefix). Each exception is
# registered here as (annotated key, repo-relative file) -> the token that line
# must contain, so the set is auditable and cannot grow by accident. Everything
# not listed must mention its own key.
#
# Keyed by file rather than by line so that ordinary code movement does not
# force an edit here — the CONTENT check still has to pass, which is the part
# that catches rot.
GATE_TOKENS: dict[tuple[str, str], str] = {
  # The three adjusted_wdl_regret_* keys are dead behind one gate flag.
    ("adjusted_wdl_regret_source", "chess_anti_engine/train/losses.py"): "use_adjusted_wdl_target",
    ("adjusted_wdl_regret_scale", "chess_anti_engine/train/losses.py"): "use_adjusted_wdl_target",
    ("adjusted_wdl_regret_cap", "chess_anti_engine/train/losses.py"): "use_adjusted_wdl_target",
  # 1.0 is the identity; the softening is skipped by a bare `temperature` compare.
    ("sf_wdl_temperature", "chess_anti_engine/train/losses.py"): "temperature != 1.0",
  # The buffer field drops the `replay_` prefix the yaml key carries.
    ("replay_sf_gap_priority_weight", "chess_anti_engine/replay/disk_buffer.py"): "sf_gap_priority_weight",
  # argparse spells the flag with hyphens. The three entries after the first are
  # exactly the surfaces the source-name grep missed; see the module docstring.
    ("games_per_iter_start", "chess_anti_engine/run.py"): "games-per-iter-start",
    ("bootstrap_max_positions", "chess_anti_engine/run.py"): "bootstrap-max-positions",
    ("bootstrap_train_steps", "chess_anti_engine/run.py"): "bootstrap-train-steps",
    ("min_replay_size", "chess_anti_engine/run.py"): "min-replay-size",
  # Ray's scheduler attribute, not the yaml key that feeds it.
    ("gpbt_resample_probability", "chess_anti_engine/tune/gpbt.py"): "_resample_probability",
}

# Pinned so that a citation (or a whole annotation) cannot vanish unnoticed.
# Raise deliberately when adding a citation; never lower to make the test pass.
EXPECTED_CITATIONS = 48

# Pinned BY NAME, not by count. A count cannot express the guarantee this test
# claims to give: swap one annotation block for a different one and the total is
# unchanged, every surviving citation still resolves, and the absent-key test
# only ever iterates the blocks it discovered — so the annotation that vanished
# is checked by nothing. The set is 24 deleted keys plus the 3 that the audit
# deliberately did NOT delete, which ship a `⚑ RETAINED` warning instead.
EXPECTED_ANNOTATED_KEYS: frozenset[str] = frozenset({
  # Deleted: inert on every resolution path, reason and citations in the config.
    "adjusted_wdl_regret_cap",
    "adjusted_wdl_regret_scale",
    "adjusted_wdl_regret_source",
    "bootstrap_dir",
    "bootstrap_max_positions",
    "bootstrap_train_steps",
    "drift_threshold",
    "gpbt_inertia_weight",
    "gpbt_quantile_fraction",
    "gpbt_resample_probability",
    "gpbt_winner_weight",
    "min_replay_size",
    "no_amp",
    "opening_fen_prob",
    "pb2_perturbation_interval",
    "replay_sf_gap_priority_weight",
    "resid_channel_balance_weight",
    "resid_channel_dropout",
    "search_optimizer_choices",
    "sf_search_dampen_sf_high",
    "sf_search_dampen_sf_low",
    "sf_wdl_temperature",
    "shared_shards_dir",
    "use_nla",
  # Retained: a live consumer whose fallback differs from the shipped value, so
  # the key is protected by being present rather than by a comment.
    "games_per_iter_start",
    "lr_T0",
    "lr_T_mult",
})

# Where a `DEAD KEY` claim is re-derived. Code only: another yaml SETTING the key
# is a writer, not a reader, so the other 16 configs/*.yaml that still set these
# keys are deliberately out of scope, and so is prose in docs/.
#
# Blind spots, stated rather than implied: this cannot see a reader in `.c`,
# `.h` or `.pyx`, in a `.sh` outside `scripts/`, or in `.toml`/`.json`. That is
# verified moot for all 27 keys annotated today, but the next `DEAD KEY` will be
# about a different key — widen the globs rather than trusting this comment.
# The scan is also filesystem-based, not git-based, so an UNTRACKED scratch
# script under `scripts/` that names a key fails this locally while CI is green.
_DEAD_KEY_SOURCE_GLOBS = (
    "chess_anti_engine/**/*.py",
    "scripts/**/*.py",
    "scripts/**/*.sh",
)

# Citations name a file, so a second mention appearing INSIDE an already-cited
# file would otherwise be invisible. Pinning the mention count per key restores
# that precision without pinning a line number: drift cannot change a count, but
# a new surface always does. Raise a number here only together with the
# annotation sentence that explains the new surface.
DEAD_KEY_EXPECTED_MENTIONS: dict[str, int] = {
    "bootstrap_max_positions": 2,  # config_yaml.py allowlist + run.py argparse flag
    "bootstrap_train_steps": 2,  # ditto
    "min_replay_size": 2,  # ditto
}

# A cited path is written relative to the package or to the repo root. Both are
# tried and exactly one must resolve, so an ambiguous path is an error rather
# than a coin flip.
_SEARCH_ROOTS = (REPO_ROOT, REPO_ROOT / "chess_anti_engine")

# Both markers name their key, so the parser never has to guess one. An earlier
# revision hardcoded the single RETAINED key here; that stops working the moment
# a second key is retained, which is exactly what happened.
_ANNOTATION_START = re.compile(
    r"^#\s*(?:⚑ RETAINED, do not delete|DELETED) ([A-Za-z_0-9]+):"
)
# Continuation lines use the "#" + 3-space wrap format. A plain "# ..." comment
# ENDS the annotation: absorbing it would silently attribute a neighbouring
# comment's citations to this key, which is how a checker like this fakes its
# own coverage.
_CONTINUATION = re.compile(r"^#\s{3,}\S")
# A citation is a bracketed repo path. The brackets are what separate a CLAIM
# from a passing prose mention of a file — see the module docstring.
_CITATION = re.compile(r"\[([A-Za-z_0-9/]+\.py)\]")


def _annotation_blocks(text: str) -> list[tuple[str, str]]:
    """Return ``(annotated_key, joined_comment_text)`` for each annotation."""
    blocks: list[tuple[str, str]] = []
    key: str | None = None
    buf: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        start = _ANNOTATION_START.match(line)
        if start:
            if key is not None:
                blocks.append((key, " ".join(buf)))
            key = start.group(1)
            buf = [line]
        elif key is not None and _CONTINUATION.match(line):
            buf.append(line)
        elif key is not None:
            blocks.append((key, " ".join(buf)))
            key, buf = None, []
    if key is not None:
        blocks.append((key, " ".join(buf)))
    return blocks


def _resolve(rel: str) -> Path:
    matches = [root / rel for root in _SEARCH_ROOTS if (root / rel).is_file()]
    if len(matches) != 1:
        pytest.fail(
            f"cited path {rel!r} resolves to {len(matches)} files under "
            f"{[str(r) for r in _SEARCH_ROOTS]}; it must resolve to exactly one"
        )
    return matches[0]


def _annotated_configs() -> list[Path]:
    return sorted(
        p for p in CONFIG_DIR.glob("*.yaml")
        if _ANNOTATION_START.search(p.read_text("utf-8"), 0) or "# DELETED " in p.read_text("utf-8")
    )


def _all_blocks() -> list[tuple[Path, str, str]]:
    out: list[tuple[Path, str, str]] = []
    for cfg in _annotated_configs():
        for key, body in _annotation_blocks(cfg.read_text("utf-8")):
            out.append((cfg, key, body))
    return out


def test_every_deletion_annotation_citation_points_at_the_code_it_claims() -> None:
    """Each cited file must still contain the key, or its registered gate.

    The line is resolved here rather than pinned in the config; what is asserted
    is that the token has not VANISHED from the file it is claimed to be in.
    That is the condition under which the annotation became a lie — a deleted
    reader, a renamed key, or a citation naming the wrong file. Code merely
    moving inside the file is not a defect and does not fail.
    """
    failures: list[str] = []
    checked = 0

    for cfg, key, body in _all_blocks():
        for rel in _CITATION.findall(body):
            checked += 1
            path = _resolve(rel)
            relpath = str(path.relative_to(REPO_ROOT))
            lines = path.read_text("utf-8").splitlines()
            expected = GATE_TOKENS.get((key, relpath), key)
            if not any(expected in src for src in lines):
                failures.append(
                    f"{cfg.name}: {key} cites [{rel}] expecting {expected!r}, "
                    f"but no line of that file contains it"
                )

    assert not failures, (
        "config deletion annotations cite code that no longer says what they "
        "claim. The cited file no longer contains the token at all, so this is "
        "not code movement — re-derive the claim, do not repoint the "
        "citation:\n  " + "\n  ".join(failures)
    )
    assert checked == EXPECTED_CITATIONS, (
        f"expected {EXPECTED_CITATIONS} citations across the annotation blocks, "
        f"found {checked}. A citation was added or removed; update "
        f"EXPECTED_CITATIONS deliberately."
    )


def test_the_annotation_block_still_has_every_key_it_is_supposed_to() -> None:
    """A whole annotation disappearing must not pass silently."""
    blocks = _all_blocks()
    keys = [key for _, key, _ in blocks]

    assert len(keys) == len(set(keys)), f"duplicate annotated keys: {sorted(keys)}"
    moved = [f"annotation vanished: {k!r}" for k in sorted(EXPECTED_ANNOTATED_KEYS - set(keys))]
    moved += [f"annotation not in the pin: {k!r}" for k in sorted(set(keys) - EXPECTED_ANNOTATED_KEYS)]
    assert not moved, (
        "the annotated key set moved. Edit EXPECTED_ANNOTATED_KEYS deliberately "
        "— a block may not be swapped for another one silently:\n  "
        + "\n  ".join(moved)
    )

    for cfg, key, body in blocks:
        assert _CITATION.search(body), (
            f"{cfg.name}: annotation for {key!r} carries no (file:line) citation; "
            f"an unsourced reason is the thing this guard exists to prevent"
        )


def test_the_annotated_key_is_actually_absent_from_the_config() -> None:
    """`# DELETED foo` while `foo:` is still set would be a lie in the file.

    Which keys are exempt is read off the annotation itself rather than listed
    here: a `⚑ RETAINED` block asserts the opposite — the key MUST still be set.
    """
    for cfg, key, body in _all_blocks():
        if body.startswith("# ⚑ RETAINED"):
            assert re.search(rf"^\s*{re.escape(key)}:", cfg.read_text("utf-8"), re.M), (
                f"{cfg.name}: annotation says {key!r} is RETAINED, but the key "
                f"is not set in the file"
            )
            continue
        assert not re.search(rf"^\s*{re.escape(key)}:", cfg.read_text("utf-8"), re.M), (
            f"{cfg.name}: annotation says {key!r} was DELETED, but the key is "
            f"still set in the file"
        )
        assert body.startswith("# DELETED"), body[:60]


def _dead_key_sources() -> list[tuple[str, list[str]]]:
    """Repo-relative path + lines for every source file a reader could live in."""
    paths: set[Path] = set()
    for pattern in _DEAD_KEY_SOURCE_GLOBS:
        paths.update(p for p in REPO_ROOT.glob(pattern) if p.is_file())
    return [
        (str(p.relative_to(REPO_ROOT)), p.read_text("utf-8", errors="replace").splitlines())
        for p in sorted(paths)
        if "tests" not in p.relative_to(REPO_ROOT).parts
    ]


def test_dead_key_annotations_cite_every_surface_that_mentions_the_key() -> None:
    """`DEAD KEY` asserts "nothing reads this" — so re-run the grep, both spellings.

    The citation check above only asks whether a cited file still mentions the
    key; it passes happily over a *false* reason built from correct citations.
    This one asks the question the annotation actually answers, and it is the
    check that fails when a surface is missed: a source-name-only grep is
    invisible to the reader of a config, and it is what put three false
    ``DEAD KEY`` statements into this block.

    Scope, so a red result here is diagnosable: only code is scanned
    (``_DEAD_KEY_SOURCE_GLOBS``). Another yaml SETTING the key is a writer, not
    a reader, so the 16 other ``configs/*.yaml`` that still set these keys are
    deliberately excluded — including them would force each annotation to cite
    16 lines that prove nothing. Prose in ``docs/`` is excluded for the same
    reason. The glob's blind spots are listed above the glob.
    """
    sources = _dead_key_sources()
    assert sources, "no source files scanned — the glob list is wrong, not the repo"

    failures: list[str] = []
    for cfg, key, body in _all_blocks():
        if "DEAD KEY" not in body:
            continue
        cited = {str(_resolve(rel).relative_to(REPO_ROOT)) for rel in _CITATION.findall(body)}
        spellings = (key, key.replace("_", "-"))
        seen = 0
        for relpath, lines in sources:
            for lineno, src in enumerate(lines, start=1):
                if not any(s in src for s in spellings):
                    continue
                seen += 1
                if relpath not in cited:
                    failures.append(
                        f"{cfg.name}: {key!r} is annotated DEAD KEY, but "
                        f"{relpath}:{lineno} mentions it and the annotation does "
                        f"not cite that file: {src.strip()!r}"
                    )

        expected = DEAD_KEY_EXPECTED_MENTIONS.get(key)
        if expected is None:
            failures.append(
                f"{cfg.name}: {key!r} is annotated DEAD KEY but has no "
                f"DEAD_KEY_EXPECTED_MENTIONS entry; pin the surface count"
            )
        elif seen != expected:
            failures.append(
                f"{cfg.name}: {key!r} is annotated DEAD KEY with "
                f"{expected} pinned surfaces, but {seen} lines mention it now. "
                f"A surface was added or removed inside an already-cited file"
            )

    assert not failures, (
        "a DEAD KEY annotation does not account for every surface that names the "
        "key. Either the claim is false, or the annotation must name the surface "
        "and say why it is not a reader:\n  " + "\n  ".join(failures)
    )


def test_registered_gate_tokens_are_all_in_use() -> None:
    """A stale exception is a hole in the check, so unused ones must be removed."""
    used = {
        (key, str(_resolve(rel).relative_to(REPO_ROOT)))
        for _, key, body in _all_blocks()
        for rel in _CITATION.findall(body)
    }
    unused = sorted(set(GATE_TOKENS) - used)
    assert not unused, (
        f"GATE_TOKENS entries no longer cited by any annotation: {unused}. "
        f"Delete them — an exception that guards nothing only widens what the "
        f"checker will accept."
    )
