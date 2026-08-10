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
review of the config can catch it. This test is the guard: it opens every cited
line and requires it to actually mention the thing the annotation claims is
there. It reads the whole annotation block, not a diff, so it keeps working
long after the PR that introduced the block has merged.

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
# Raise these deliberately when adding an annotation; never lower them to make
# the test pass.
EXPECTED_CITATIONS = 46
EXPECTED_ANNOTATED_KEYS = 27  # 26 deleted keys + the one RETAINED warning block

# Where a `DEAD KEY` claim is re-derived. Code only: another yaml setting the key
# is not a reader (17 configs/*.yaml still carry these keys and are out of scope
# for this PR), and neither is prose in docs/.
_DEAD_KEY_SOURCE_GLOBS = (
    "chess_anti_engine/**/*.py",
    "scripts/**/*.py",
    "scripts/**/*.sh",
)

# A cited path is written relative to the package or to the repo root. Both are
# tried and exactly one must resolve, so an ambiguous path is an error rather
# than a coin flip.
_SEARCH_ROOTS = (REPO_ROOT, REPO_ROOT / "chess_anti_engine")

_ANNOTATION_START = re.compile(
    r"^#\s*(?:⚑ RETAINED, do not delete:|DELETED ([A-Za-z_0-9]+):)"
)
# Continuation lines use the "#" + 3-space wrap format. A plain "# ..." comment
# ENDS the annotation: absorbing it would silently attribute a neighbouring
# comment's citations to this key, which is how a checker like this fakes its
# own coverage.
_CONTINUATION = re.compile(r"^#\s{3,}\S")
_CITATION = re.compile(r"([A-Za-z_0-9/]+\.py):(\d+)")


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
          # The RETAINED block has no key in the marker; it is the one key the
          # audit deliberately did NOT delete.
            key = start.group(1) or "games_per_iter_start"
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
    """Open each cited line; it must mention the key, or its registered gate."""
    failures: list[str] = []
    checked = 0

    for cfg, key, body in _all_blocks():
        for rel, lineno in _CITATION.findall(body):
            checked += 1
            path = _resolve(rel)
            relpath = str(path.relative_to(REPO_ROOT))
            lines = path.read_text("utf-8").splitlines()
            n = int(lineno)
            if not 1 <= n <= len(lines):
                failures.append(
                    f"{cfg.name}: {key} cites {rel}:{n}, but that file has "
                    f"{len(lines)} lines"
                )
                continue
            src = lines[n - 1]
            expected = GATE_TOKENS.get((key, relpath), key)
            if expected not in src:
                failures.append(
                    f"{cfg.name}: {key} cites {rel}:{n} expecting {expected!r}, "
                    f"but that line is {src.strip()!r}"
                )

    assert not failures, (
        "config deletion annotations cite code that no longer says what they "
        "claim — re-derive the line numbers against HEAD (do not just bump "
        "them):\n  " + "\n  ".join(failures)
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
    assert len(keys) == EXPECTED_ANNOTATED_KEYS, (
        f"expected {EXPECTED_ANNOTATED_KEYS} annotated keys "
        f"(26 deleted + 1 RETAINED), found {len(keys)}: {sorted(keys)}"
    )

    for cfg, key, body in blocks:
        assert _CITATION.search(body), (
            f"{cfg.name}: annotation for {key!r} carries no (file:line) citation; "
            f"an unsourced reason is the thing this guard exists to prevent"
        )


def test_the_annotated_key_is_actually_absent_from_the_config() -> None:
    """`# DELETED foo` while `foo:` is still set would be a lie in the file."""
    retained = {"games_per_iter_start"}
    for cfg, key, body in _all_blocks():
        if key in retained:
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

    The citation check above only asks whether a cited line mentions the key; it
    passes happily over a *false* reason built from correct citations. This one
    asks the question the annotation actually answers, and it is the check that
    fails when a surface is missed: a source-name-only grep is invisible to the
    reader of a config, and it is what put three false ``DEAD KEY`` statements
    into this block.
    """
    sources = _dead_key_sources()
    assert sources, "no source files scanned — the glob list is wrong, not the repo"

    failures: list[str] = []
    for cfg, key, body in _all_blocks():
        if "DEAD KEY" not in body:
            continue
        cited = {
            (str(_resolve(rel).relative_to(REPO_ROOT)), int(line))
            for rel, line in _CITATION.findall(body)
        }
        spellings = (key, key.replace("_", "-"))
        for relpath, lines in sources:
            for lineno, src in enumerate(lines, start=1):
                if not any(s in src for s in spellings):
                    continue
                if (relpath, lineno) not in cited:
                    failures.append(
                        f"{cfg.name}: {key!r} is annotated DEAD KEY, but "
                        f"{relpath}:{lineno} mentions it and the annotation does "
                        f"not cite that line: {src.strip()!r}"
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
        for rel, _ in _CITATION.findall(body)
    }
    unused = sorted(set(GATE_TOKENS) - used)
    assert not unused, (
        f"GATE_TOKENS entries no longer cited by any annotation: {unused}. "
        f"Delete them — an exception that guards nothing only widens what the "
        f"checker will accept."
    )
