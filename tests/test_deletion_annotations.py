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

The identity pins matter as much as the content check: without them a citation
could be quietly deleted — or an annotation dropped whole — and every remaining
citation would still pass. They are IDENTITY pins, not counts, because a count
has a substitution hole: move a citation from one block to another, or swap one
annotation for a different key's, and the total never moves.

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

Completeness is re-derived for EVERY key, over the string-literal surface
------------------------------------------------------------------------
The ``DEAD KEY`` phrase gate used to be the only completeness check, and twice
in a row the miss lived exactly where the gate was not: ``lr_T0``/``lr_T_mult``
had a yaml-flat consumer in ``scripts/train_bootstrap.py``, and both
``sf_search_dampen_*`` keys are REQUIRED to be present by
``scripts/value_optimism.py``. Neither annotation carried the phrase, so nothing
re-derived either claim.

Widening the ``DEAD KEY`` re-derivation itself — every raw-text mention of the
key must be cited by the annotation — was measured before being rejected:
**318 line-surfaces over 156 (key, file) pairs** for the 27 annotated keys, with
``use_nla`` alone at **37 lines across 21 files**, most of them bench scripts
passing the name through as a kwarg. An annotation that must cite 21 files is
not read, so that widening buys nothing.

What is used instead is the STRING-LITERAL surface, and it is not a weaker
approximation of the raw-text one — it is the exact one. A yaml key can only be
read out of the flattened config by NAME, so every consumer that can be broken
by deleting the key contains the key as a ``str`` constant. Attribute reads
(``tc.use_nla``), argparse ``--flag`` prose and comments cannot be. Measured:
**179 literal occurrences over 104 (key, file) pairs**, ``use_nla`` down to
5 files. ``KEY_LITERAL_SURFACES`` pins that set per key, so a new file that
starts naming a deleted key goes red and the annotation is re-judged.

The specific shape that got past every name-based scan is handled separately by
``MODULE_LEVEL_KEY_COLLECTIONS``. ``scripts/value_optimism.py`` never writes
``flat.get("sf_search_dampen_sf_low", ...)``; it lists the key in a module-level
tuple and reaches it through a LOOP VARIABLE, raising ``SystemExit`` when the key
is absent. A scan looking for ``.get(key, default)`` and ``tc.<field>`` is blind
to that by construction, which is why it is enumerated from the AST rather than
grepped, and why each collection carries an explicit "does absence break this"
verdict.

What this file does NOT verify
------------------------------
Stated so the coverage is not overread:

* the annotation's REASON, in prose. Every surface being cited is not the
  sentence about that surface being true. Two parts of the reason ARE now
  re-derived — the resolved-value claim (``EXPECTED_VALUE_CLAIMS``) and the
  whole retention rationale for the five retained keys
  (``RETENTION_MECHANISMS``) — but the sentences around those numbers are not.
* which annotation block belongs to which key, beyond the value claim. Swapping
  the header keys of two blocks whose stated values differ is caught by
  ``EXPECTED_VALUE_CLAIMS``; swapping two that state the SAME value (both GPBT
  weights resolve at 1.0) is not.
* claims phrased without the literal token ``DEAD KEY`` still skip the
  raw-text completeness scan; they get the literal-surface pin instead.
* ``.c``/``.h``/``.pyx``/``.toml``/``.json`` and ``.sh`` outside ``scripts/``,
  for both scans. Verified moot for all 27 keys annotated today.
"""

from __future__ import annotations

import ast
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

# Pinned as (annotated key -> the files it cites), not as a total. A scalar
# count has a substitution hole that is the exact analogue of the one the key
# count had: move a citation from one annotation block to another and the total
# is unchanged, both copies resolve, and the second-consumer clause the first
# block existed to carry is gone. Duplicates are preserved — `lr_T0` really does
# cite [train/trainer.py] twice, in two different clauses.
EXPECTED_CITATIONS: dict[str, tuple[str, ...]] = {
    "adjusted_wdl_regret_cap": ("chess_anti_engine/train/losses.py", "chess_anti_engine/train/trainer.py"),
    "adjusted_wdl_regret_scale": ("chess_anti_engine/train/losses.py", "chess_anti_engine/train/trainer.py"),
    "adjusted_wdl_regret_source": ("chess_anti_engine/train/losses.py", "chess_anti_engine/train/trainer.py"),
    "bootstrap_dir": ("scripts/train_bootstrap.py",),
    "bootstrap_max_positions": ("chess_anti_engine/run.py", "chess_anti_engine/utils/config_yaml.py"),
    "bootstrap_train_steps": ("chess_anti_engine/run.py", "chess_anti_engine/utils/config_yaml.py"),
    "drift_threshold": ("chess_anti_engine/tune/trainable.py", "chess_anti_engine/tune/trial_config.py"),
    "games_per_iter_start": ("chess_anti_engine/run.py", "chess_anti_engine/tune/trial_config.py"),
    "gpbt_inertia_weight": ("chess_anti_engine/tune/harness.py",),
    "gpbt_quantile_fraction": ("chess_anti_engine/tune/harness.py",),
    "gpbt_resample_probability": ("chess_anti_engine/tune/gpbt.py", "chess_anti_engine/tune/harness.py"),
    "gpbt_winner_weight": ("chess_anti_engine/tune/harness.py",),
    "lr_T0": ("chess_anti_engine/train/trainer.py", "chess_anti_engine/train/trainer.py", "scripts/train_bootstrap.py"),
    "lr_T_mult": ("chess_anti_engine/train/trainer.py", "scripts/train_bootstrap.py"),
    "min_replay_size": ("chess_anti_engine/run.py", "chess_anti_engine/utils/config_yaml.py"),
    "no_amp": ("chess_anti_engine/run.py", "scripts/train_bootstrap.py"),
    "opening_fen_prob": ("chess_anti_engine/selfplay/opening.py", "chess_anti_engine/tune/trial_config.py", "chess_anti_engine/worker.py"),
    "pb2_perturbation_interval": ("chess_anti_engine/tune/harness.py",),
    "replay_sf_gap_priority_weight": ("chess_anti_engine/replay/disk_buffer.py", "chess_anti_engine/tune/trial_config.py"),
    "resid_channel_balance_weight": ("chess_anti_engine/model/transformer.py", "chess_anti_engine/train/trainer.py"),
    "resid_channel_dropout": ("chess_anti_engine/model/transformer.py", "chess_anti_engine/train/trainer.py"),
    "search_optimizer_choices": ("chess_anti_engine/tune/harness.py",),
    "sf_search_dampen_sf_high": ("chess_anti_engine/train/losses.py", "chess_anti_engine/train/trainer.py", "scripts/value_optimism.py"),
    "sf_search_dampen_sf_low": ("chess_anti_engine/train/losses.py", "chess_anti_engine/train/trainer.py", "scripts/value_optimism.py"),
    "sf_wdl_temperature": ("chess_anti_engine/train/losses.py", "chess_anti_engine/train/trainer.py", "scripts/value_optimism.py"),
    "shared_shards_dir": ("chess_anti_engine/tune/trial_config.py",),
    "use_nla": ("chess_anti_engine/model/__init__.py",),
}

# Pinned BY NAME **and by class**, not by count. A count cannot express the
# guarantee this test claims to give: swap one annotation block for a different
# one and the total is unchanged, every surviving citation still resolves, and
# the absent-key test only ever iterates the blocks it discovered — so the
# annotation that vanished is checked by nothing. The CLASS is pinned for the
# same reason one step down: names alone let `lr_T0`'s marker flip
# `⚑ RETAINED` -> `# DELETED`, the key be removed, and the absent-key test
# simply take its other arm — silently restoring the 999999 -> 2000 bootstrap
# drift this PR retained the key to prevent.
EXPECTED_ANNOTATIONS: dict[str, str] = {
  # DELETED: inert on every resolution path, reason and citations in the config.
    "adjusted_wdl_regret_cap": "DELETED",
    "adjusted_wdl_regret_scale": "DELETED",
    "adjusted_wdl_regret_source": "DELETED",
    "bootstrap_dir": "DELETED",
    "bootstrap_max_positions": "DELETED",
    "bootstrap_train_steps": "DELETED",
    "drift_threshold": "DELETED",
    "gpbt_inertia_weight": "DELETED",
    "gpbt_quantile_fraction": "DELETED",
    "gpbt_resample_probability": "DELETED",
    "gpbt_winner_weight": "DELETED",
    "min_replay_size": "DELETED",
    "no_amp": "DELETED",
    "opening_fen_prob": "DELETED",
    "pb2_perturbation_interval": "DELETED",
    "replay_sf_gap_priority_weight": "DELETED",
    "resid_channel_balance_weight": "DELETED",
    "resid_channel_dropout": "DELETED",
    "search_optimizer_choices": "DELETED",
    "sf_wdl_temperature": "DELETED",
    "shared_shards_dir": "DELETED",
    "use_nla": "DELETED",
  # RETAINED: a live consumer that either falls back to a DIFFERENT value than
  # the one shipped, or refuses to run without the key at all. Protected by
  # being present, not by a comment. Each one's mechanism is re-derived from
  # source by RETENTION_MECHANISMS below.
    "games_per_iter_start": "RETAINED",
    "lr_T0": "RETAINED",
    "lr_T_mult": "RETAINED",
    "sf_search_dampen_sf_high": "RETAINED",
    "sf_search_dampen_sf_low": "RETAINED",
}

# The one falsifiable NUMBER an annotation states: what the key resolves to once
# it is gone. Pinned per key so that swapping two annotation blocks' header keys
# — which leaves the count, the class map and every citation intact — is caught
# whenever the two blocks claim different values. It does not catch a swap
# between two blocks stating the SAME value; see the docstring.
EXPECTED_VALUE_CLAIMS: dict[str, tuple[str, ...]] = {
    "drift_threshold": ("0.0",),
    "games_per_iter_start": ("0",),
    "gpbt_inertia_weight": ("1.0",),
    "gpbt_quantile_fraction": ("0.25",),
    "gpbt_resample_probability": ("0.05",),
    "gpbt_winner_weight": ("1.0",),
    "pb2_perturbation_interval": ("10",),
    "search_optimizer_choices": ("None",),
}
_VALUE_CLAIM = re.compile(r"(?:[Rr]esolves unchanged at|resolves to(?: the argparse default)?) ([^\s,]+)")

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
# that precision without pinning a line number.
#
# Stated accurately, because the earlier wording ("drift cannot change a count,
# only a new surface does") overclaimed: what cannot change the count is code
# MOVING. Anything that adds or removes a line NAMING the key does change it —
# a new reader, a deleted reader, a renamed local, a reworded comment, a
# reformatted call that splits one mention across two lines. So an unrelated PR
# CAN be forced red here, and the honest cost is that the fix then lands in
# `configs/pbt2_small.yaml`, the live production config, because the annotation
# sentence and the number have to move together. That cost is bounded to the
# three `DEAD KEY` keys, whose whole claim is "nothing reads this"; every other
# pin in this file lives in the test, where an unrelated PR can edit it safely.
# `_resolve()` has the same shape at file granularity: MOVING a cited file
# renames the citation, and that edit is in the config too.
#
# Raise a number here only together with the annotation sentence that explains
# the new surface.
DEAD_KEY_EXPECTED_MENTIONS: dict[str, int] = {
    "bootstrap_max_positions": 2,  # config_yaml.py allowlist + run.py argparse flag
    "bootstrap_train_steps": 2,  # ditto
    "min_replay_size": 2,  # ditto
}

# A cited path is written relative to the package or to the repo root. Both are
# tried and exactly one must resolve, so an ambiguous path is an error rather
# than a coin flip.
_SEARCH_ROOTS = (REPO_ROOT, REPO_ROOT / "chess_anti_engine")

# ---------------------------------------------------------------------------
# The literal surface: every place a deleted key could still be READ.
# ---------------------------------------------------------------------------
# Python only, because this is an AST scan and the claim it supports is exact:
# a flattened-config key is reached by NAME, so a consumer that breaks when the
# key disappears necessarily contains the key as a `str` constant. `.sh` is
# excluded here (it is covered by the raw-text DEAD KEY scan) because it has no
# AST and its raw mentions are the noise this scan exists to drop.
_LITERAL_SOURCE_GLOBS = (
    "chess_anti_engine/**/*.py",
    "scripts/**/*.py",
)

# Pinned per key: every code file that names it as a string literal. Adding a
# reader of a key this config no longer sets goes RED here, and the fix is to
# re-judge the annotation, not to extend the tuple reflexively.
#
# This pin lives in the TEST, deliberately, and that is the answer to "a guard
# must not make the live production config the file unrelated PRs have to edit":
# a PR that adds a consumer edits this file, never `configs/pbt2_small.yaml`.
KEY_LITERAL_SURFACES: dict[str, tuple[str, ...]] = {
    "adjusted_wdl_regret_cap": ("chess_anti_engine/train/trainer.py", "chess_anti_engine/tune/trainable_config_ops.py", "chess_anti_engine/utils/config_yaml.py", "scripts/diagnose_target_calibration.py", "scripts/offline_replay_epoch.py", "scripts/shrink_ffn_checkpoint.py"),
    "adjusted_wdl_regret_scale": ("chess_anti_engine/train/trainer.py", "chess_anti_engine/tune/trainable_config_ops.py", "chess_anti_engine/utils/config_yaml.py", "scripts/diagnose_target_calibration.py", "scripts/offline_replay_epoch.py", "scripts/shrink_ffn_checkpoint.py"),
    "adjusted_wdl_regret_source": ("chess_anti_engine/train/trainer.py", "chess_anti_engine/tune/trainable_config_ops.py", "chess_anti_engine/utils/config_yaml.py", "scripts/diagnose_target_calibration.py", "scripts/offline_replay_epoch.py", "scripts/shrink_ffn_checkpoint.py"),
    "bootstrap_dir": ("chess_anti_engine/utils/config_yaml.py", "scripts/train_bootstrap.py"),
    "bootstrap_max_positions": ("chess_anti_engine/utils/config_yaml.py",),
    "bootstrap_train_steps": ("chess_anti_engine/utils/config_yaml.py",),
    "drift_threshold": ("chess_anti_engine/tune/trial_config.py", "chess_anti_engine/utils/config_yaml.py"),
    "games_per_iter_start": ("chess_anti_engine/tune/trial_config.py", "chess_anti_engine/utils/config_yaml.py", "scripts/audit_realized_config.py", "scripts/profile_distributed.py"),
    "gpbt_inertia_weight": ("chess_anti_engine/tune/harness.py", "chess_anti_engine/tune/trainable_config_ops.py", "chess_anti_engine/utils/config_yaml.py"),
    "gpbt_quantile_fraction": ("chess_anti_engine/tune/harness.py", "chess_anti_engine/tune/trainable_config_ops.py", "chess_anti_engine/utils/config_yaml.py"),
    "gpbt_resample_probability": ("chess_anti_engine/tune/harness.py", "chess_anti_engine/tune/trainable_config_ops.py", "chess_anti_engine/utils/config_yaml.py"),
    "gpbt_winner_weight": ("chess_anti_engine/tune/harness.py", "chess_anti_engine/tune/trainable_config_ops.py", "chess_anti_engine/utils/config_yaml.py"),
    "lr_T0": ("chess_anti_engine/train/trainer.py", "chess_anti_engine/utils/config_yaml.py", "scripts/train_bootstrap.py"),
    "lr_T_mult": ("chess_anti_engine/train/trainer.py", "chess_anti_engine/utils/config_yaml.py", "scripts/train_bootstrap.py"),
    "min_replay_size": ("chess_anti_engine/utils/config_yaml.py",),
    "no_amp": ("chess_anti_engine/utils/config_yaml.py", "scripts/train_bootstrap.py"),
    "opening_fen_prob": ("chess_anti_engine/tune/distributed_runtime.py", "chess_anti_engine/tune/trial_config.py", "chess_anti_engine/utils/config_yaml.py", "chess_anti_engine/worker.py", "scripts/loop_health.py"),
    "pb2_perturbation_interval": ("chess_anti_engine/tune/harness.py", "chess_anti_engine/tune/trainable_config_ops.py", "chess_anti_engine/utils/config_yaml.py"),
    "replay_sf_gap_priority_weight": ("chess_anti_engine/tune/trial_config.py", "chess_anti_engine/utils/config_yaml.py"),
    "resid_channel_balance_weight": ("chess_anti_engine/model/transformer.py", "chess_anti_engine/train/trainer.py", "chess_anti_engine/tune/trainable_config_ops.py", "chess_anti_engine/utils/config_yaml.py", "scripts/offline_replay_epoch.py"),
    "resid_channel_dropout": ("chess_anti_engine/model/transformer.py", "chess_anti_engine/train/trainer.py", "chess_anti_engine/tune/trainable_config_ops.py", "chess_anti_engine/utils/config_yaml.py", "scripts/offline_replay_epoch.py"),
    "search_optimizer_choices": ("chess_anti_engine/run.py", "chess_anti_engine/tune/harness.py", "chess_anti_engine/tune/trainable_config_ops.py", "chess_anti_engine/utils/config_yaml.py"),
    "sf_search_dampen_sf_high": ("chess_anti_engine/config_keys.py", "chess_anti_engine/train/trainer.py", "chess_anti_engine/tune/trial_config.py", "scripts/diagnose_sf_search_disagreements.py", "scripts/diagnose_target_calibration.py", "scripts/diagnose_wdl_by_age.py", "scripts/shrink_ffn_checkpoint.py", "scripts/value_optimism.py"),
    "sf_search_dampen_sf_low": ("chess_anti_engine/config_keys.py", "chess_anti_engine/train/trainer.py", "chess_anti_engine/tune/trial_config.py", "scripts/diagnose_sf_search_disagreements.py", "scripts/diagnose_target_calibration.py", "scripts/diagnose_wdl_by_age.py", "scripts/shrink_ffn_checkpoint.py", "scripts/value_optimism.py"),
    "sf_wdl_temperature": ("chess_anti_engine/config_keys.py", "chess_anti_engine/train/trainer.py", "chess_anti_engine/tune/trainable_report.py", "chess_anti_engine/tune/trial_config.py", "scripts/diagnose_target_calibration.py", "scripts/diagnose_wdl_by_age.py", "scripts/shrink_ffn_checkpoint.py", "scripts/value_optimism.py"),
    "shared_shards_dir": ("chess_anti_engine/tune/trial_config.py", "chess_anti_engine/utils/config_yaml.py"),
    "use_nla": ("chess_anti_engine/arena.py", "chess_anti_engine/model/__init__.py", "chess_anti_engine/tune/harness.py", "chess_anti_engine/tune/trial_config.py", "chess_anti_engine/utils/config_yaml.py"),
}

# ---------------------------------------------------------------------------
# The shape no name-based scan can see.
# ---------------------------------------------------------------------------
# A consumer does not have to write `flat.get("<key>", default)`. It can list the
# key in a module-level constant collection and reach it through a loop variable
# — and then the only text near the read is the loop variable's name. That is
# how `scripts/value_optimism.py` requiring `sf_search_dampen_*` survived a
# generalised scan of all 26 deleted keys: the scan classified consumers by the
# shapes `.get(key, default)` and `tc.<field>`, and this is neither.
#
# So the collections are enumerated from the AST instead of being grepped, and
# each one carries an explicit verdict on the only question that matters:
# does the key's ABSENCE change what this code does?
MODULE_LEVEL_KEY_COLLECTIONS: dict[tuple[str, str], tuple[str, ...]] = {
  # The flattener's own schema. Absence from the YAML is exactly the case these
  # allowlists exist to permit; absence from the ALLOWLIST is the launch-fatal
  # unknown-key path, and deleting a yaml key does not touch it.
    ("chess_anti_engine/utils/config_yaml.py", "_CORE_KEYS"): ("bootstrap_dir", "bootstrap_max_positions", "bootstrap_train_steps", "shared_shards_dir"),
    ("chess_anti_engine/utils/config_yaml.py", "_MODEL_PASSTHROUGH"): ("use_nla",),
    ("chess_anti_engine/utils/config_yaml.py", "_SELFPLAY_KEYS"): ("games_per_iter_start", "opening_fen_prob"),
    ("chess_anti_engine/utils/config_yaml.py", "_TRAIN_KEYS"): ("adjusted_wdl_regret_cap", "adjusted_wdl_regret_scale", "adjusted_wdl_regret_source", "lr_T0", "lr_T_mult", "no_amp", "resid_channel_balance_weight", "resid_channel_dropout"),
    ("chess_anti_engine/utils/config_yaml.py", "_TUNE_KEYS"): ("drift_threshold", "gpbt_inertia_weight", "gpbt_quantile_fraction", "gpbt_resample_probability", "gpbt_winner_weight", "min_replay_size", "pb2_perturbation_interval", "replay_sf_gap_priority_weight", "search_optimizer_choices"),
  # Runtime-mutable trainer attributes. Both consumers loop `if k in config`
  # (trainable_config_ops live sync) or `if k in donor_cfg` (trainable_init's
  # donor overlay), so absence is a no-op on each.
    ("chess_anti_engine/config_keys.py", "TRAINER_WEIGHT_KEYS"): ("sf_search_dampen_sf_high", "sf_search_dampen_sf_low", "sf_wdl_temperature"),
  # Membership tests over the driver/tune config. Absence removes a member from
  # a set intersection; no value is read.
    ("chess_anti_engine/run.py", "_TUNE_CONFIG_DENYLIST"): ("search_optimizer_choices",),
    ("chess_anti_engine/tune/trainable_config_ops.py", "_DRIVER_LAUNCH_FIXED_KEYS"): ("gpbt_inertia_weight", "gpbt_quantile_fraction", "gpbt_resample_probability", "gpbt_winner_weight", "pb2_perturbation_interval", "search_optimizer_choices"),
    ("scripts/audit_realized_config.py", "_RECO_SERVER_RESOLVED"): ("games_per_iter_start",),
  # Read off progress.csv / compared against an already-resolved dict, never off
  # the yaml flat dict — so `sf_wdl_temperature` can be deleted despite sitting
  # in both of these.
    ("scripts/value_optimism.py", "_NEUTRAL_BLEND_KNOBS"): ("sf_search_dampen_sf_high", "sf_search_dampen_sf_low", "sf_wdl_temperature"),
    ("scripts/value_optimism.py", "_REALIZED_FROM_PROGRESS"): ("sf_wdl_temperature",),
  # ⚑ REQUIRES PRESENCE. See PRESENCE_REQUIRING_COLLECTIONS.
    ("scripts/value_optimism.py", "_CROSSCHECK_YAML_VS_PARAMS"): ("sf_search_dampen_sf_high", "sf_search_dampen_sf_low"),
}

# Collections whose consumer RAISES when a member key is absent from the config.
# No key in one of these may be annotated DELETED — that is the whole gate, and
# it is what round 6 shipped wrong.
PRESENCE_REQUIRING_COLLECTIONS: frozenset[tuple[str, str]] = frozenset({
  # resolve_blend_knobs: `for key in _CROSSCHECK_YAML_VS_PARAMS: if key not in
  # flat: raise SystemExit(...)`, on flatten_run_config_defaults(--config),
  # before any model or data work. Deliberate: the docstring refuses to let an
  # absent key fall back to the neutral value.
    ("scripts/value_optimism.py", "_CROSSCHECK_YAML_VS_PARAMS"),
})

# ---------------------------------------------------------------------------
# Why each retained key is retained — re-derived, not asserted.
# ---------------------------------------------------------------------------
# `⚑ RETAINED` blocks were the one part of this annotation set with no guard at
# all: invert the measurement in the prose and every test stayed green. Now the
# mechanism is read back out of the source. Three kinds, because there are three
# real mechanisms:
#
#   flat_get           the consumer reads the yaml flat dict with a fallback
#                      that DIFFERS from the shipped value
#   argparse_default   run.py materialises the key at a default that DIFFERS
#                      from the shipped value, so absence is not "unset"
#   requires_presence  the consumer raises when the key is absent
#
# (kind, repo-relative consumer file, expected fallback as source text).
RETENTION_MECHANISMS: dict[str, tuple[str, str, str]] = {
    "lr_T0": ("flat_get", "scripts/train_bootstrap.py", "2000"),
    "lr_T_mult": ("flat_get", "scripts/train_bootstrap.py", "2"),
    "games_per_iter_start": ("argparse_default", "chess_anti_engine/run.py", "0"),
    "sf_search_dampen_sf_low": ("requires_presence", "scripts/value_optimism.py", ""),
    "sf_search_dampen_sf_high": ("requires_presence", "scripts/value_optimism.py", ""),
}

# Both markers name their key, so the parser never has to guess one. An earlier
# revision hardcoded the single RETAINED key here; that stops working the moment
# a second key is retained, which is exactly what happened.
#
# re.MULTILINE is load-bearing, not decoration: `_annotated_configs` searches the
# WHOLE file text, and without it `^` anchors at offset 0 only — so a config
# whose only annotation is a `⚑ RETAINED` block below its header was silently
# not scanned at all, and its key, citations and required setting were validated
# by nothing. `_annotation_blocks` matches line by line and is unaffected.
_ANNOTATION_START = re.compile(
    r"^#\s*(?:⚑ RETAINED, do not delete|DELETED) ([A-Za-z_0-9]+):",
    re.MULTILINE,
)
_RETAINED_PREFIX = "# ⚑ RETAINED"
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
    return sorted(p for p in CONFIG_DIR.glob("*.yaml") if _ANNOTATION_START.search(p.read_text("utf-8")))


def _all_blocks() -> list[tuple[Path, str, str]]:
    out: list[tuple[Path, str, str]] = []
    for cfg in _annotated_configs():
        for key, body in _annotation_blocks(cfg.read_text("utf-8")):
            out.append((cfg, key, body))
    return out


def _annotation_class(body: str) -> str:
    return "RETAINED" if body.startswith(_RETAINED_PREFIX) else "DELETED"


def _prose(body: str) -> str:
    """The block's sentences with the comment markers taken back out.

    A wrapped clause joins as ``"the argparse #   default 10"``, so any regex
    read over the raw block sees a ``#`` in the middle of a sentence and matches
    the wrong token. Every claim read out of an annotation goes through here.
    """
    return re.sub(r"\s+", " ", body.replace("#", " ")).strip()


def _literal_source_files() -> list[tuple[str, ast.Module]]:
    """Repo-relative path + parsed AST for every file a key could be READ in."""
    paths: set[Path] = set()
    for pattern in _LITERAL_SOURCE_GLOBS:
        paths.update(p for p in REPO_ROOT.glob(pattern) if p.is_file())
    out: list[tuple[str, ast.Module]] = []
    for p in sorted(paths):
        if "tests" in p.relative_to(REPO_ROOT).parts:
            continue
        try:
            out.append((str(p.relative_to(REPO_ROOT)), ast.parse(p.read_text("utf-8", errors="replace"))))
        except SyntaxError:  # a scratch file that does not parse is not a consumer
            continue
    return out


def _spellings(keys: set[str]) -> dict[str, str]:
    """Every literal that names one of *keys*, mapped back to the yaml spelling."""
    return {s: k for k in keys for s in (k, k.replace("_", "-"))}


def _module_level_collections(rel: str, mod: ast.Module, names: dict[str, str]) -> dict[str, set[str]]:
    """``constant name -> annotated keys named inside it``, module scope only."""
    del rel
    out: dict[str, set[str]] = {}
    for node in mod.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        bound = [t.id for t in targets if isinstance(t, ast.Name)]
        if not bound or node.value is None:
            continue
        found = {
            names[n.value]
            for n in ast.walk(node.value)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in names
        }
        if found:
            out.setdefault(bound[0], set()).update(found)
    return out


def _yaml_value(cfg: Path, key: str) -> str:
    m = re.search(rf"^\s*{re.escape(key)}:\s*(\S+)", cfg.read_text("utf-8"), re.M)
    assert m is not None, f"{cfg.name}: {key!r} is not set in the file"
    return m.group(1)


def _flat_get_fallback(rel: str, mod: ast.Module, key: str) -> str | None:
    """The literal default of a ``<dict>.get("<key>", <const>)`` call, if any."""
    del rel
    for node in ast.walk(mod):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "get" or len(node.args) != 2:
            continue
        first, second = node.args
        if isinstance(first, ast.Constant) and first.value == key and isinstance(second, ast.Constant):
            return str(second.value)
    return None


def _argparse_default(rel: str, mod: ast.Module, key: str) -> str | None:
    """The literal ``default=`` of ``add_argument("--<key>", ...)``, if any."""
    del rel
    flag = "--" + key.replace("_", "-")
    for node in ast.walk(mod):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and first.value == flag):
            continue
        for kw in node.keywords:
            if kw.arg == "default" and isinstance(kw.value, ast.Constant):
                return str(kw.value.value)
    return None


def test_every_deletion_annotation_citation_points_at_the_code_it_claims() -> None:
    """Each cited file must still contain the key, or its registered gate.

    The line is resolved here rather than pinned in the config; what is asserted
    is that the token has not VANISHED from the file it is claimed to be in.
    That is the condition under which the annotation became a lie — a deleted
    reader, a renamed key, or a citation naming the wrong file. Code merely
    moving inside the file is not a defect and does not fail.
    """
    failures: list[str] = []
    found: dict[str, tuple[str, ...]] = {}

    for cfg, key, body in _all_blocks():
        cited: list[str] = []
        for rel in _CITATION.findall(body):
            path = _resolve(rel)
            relpath = str(path.relative_to(REPO_ROOT))
            cited.append(relpath)
            lines = path.read_text("utf-8").splitlines()
            expected = GATE_TOKENS.get((key, relpath), key)
            if not any(expected in src for src in lines):
                failures.append(
                    f"{cfg.name}: {key} cites [{rel}] expecting {expected!r}, "
                    f"but no line of that file contains it"
                )
        found[key] = tuple(sorted(cited))

    assert not failures, (
        "config deletion annotations cite code that no longer says what they "
        "claim. The cited file no longer contains the token at all, so this is "
        "not code movement — re-derive the claim, do not repoint the "
        "citation:\n  " + "\n  ".join(failures)
    )

    moved = [
        f"{k}: pinned {EXPECTED_CITATIONS.get(k)} but the annotation cites {found.get(k)}"
        for k in sorted(set(found) | set(EXPECTED_CITATIONS))
        if found.get(k) != tuple(sorted(EXPECTED_CITATIONS.get(k, ())))
    ]
    assert not moved, (
        "the citation set moved. Pinning the TOTAL would not have caught this: a "
        "citation relocated from one annotation block to another keeps the count, "
        "keeps resolving, and drops the clause it was carrying. Edit "
        "EXPECTED_CITATIONS deliberately:\n  " + "\n  ".join(moved)
    )


def test_the_annotation_block_still_has_every_key_and_class_it_is_supposed_to() -> None:
    """A whole annotation disappearing — or changing class — must not pass silently.

    The CLASS matters as much as the name. `⚑ RETAINED lr_T0` -> `# DELETED
    lr_T0` plus deleting the key leaves the name set and every citation intact,
    and the absent-key test below simply takes its other arm — which is the
    bootstrap-schedule drift this PR retained the key to prevent, restored in
    silence.
    """
    blocks = _all_blocks()
    keys = [key for _, key, _ in blocks]

    assert len(keys) == len(set(keys)), f"duplicate annotated keys: {sorted(keys)}"
    found = {key: _annotation_class(body) for _, key, body in blocks}
    moved = [
        f"{k!r}: pinned {EXPECTED_ANNOTATIONS.get(k)!r}, file says {found.get(k)!r}"
        for k in sorted(set(found) | set(EXPECTED_ANNOTATIONS))
        if found.get(k) != EXPECTED_ANNOTATIONS.get(k)
    ]
    assert not moved, (
        "the annotated key set or its DELETED/RETAINED classes moved. Edit "
        "EXPECTED_ANNOTATIONS deliberately — a block may not be swapped for "
        "another one, or reclassified, silently:\n  " + "\n  ".join(moved)
    )

    for cfg, key, body in blocks:
        assert _CITATION.search(body), (
            f"{cfg.name}: annotation for {key!r} carries no (file:line) citation; "
            f"an unsourced reason is the thing this guard exists to prevent"
        )


def test_each_annotation_states_the_resolved_value_it_is_pinned_to() -> None:
    """The one number in the prose that is falsifiable, pinned per key.

    This is what stops two annotation blocks having their header keys swapped:
    the count, the class map and every citation survive that mutation, but
    `gpbt_quantile_fraction` then tells an operator it resolves at 1.0 when it
    resolves at 0.25.
    """
    failures: list[str] = []
    for cfg, key, body in _all_blocks():
        stated = tuple(v.rstrip(".") for v in _VALUE_CLAIM.findall(_prose(body)))
        expected = EXPECTED_VALUE_CLAIMS.get(key, ())
        if stated != expected:
            failures.append(f"{cfg.name}: {key!r} states {stated}, pinned {expected}")
    assert not failures, (
        "a resolved-value claim moved. Either the annotation was reworded, or a "
        "block's header key was swapped with another block's:\n  " + "\n  ".join(failures)
    )


def test_the_annotated_key_is_actually_absent_from_the_config() -> None:
    """`# DELETED foo` while `foo:` is still set would be a lie in the file.

    Which keys are exempt is read off the annotation itself rather than listed
    here: a `⚑ RETAINED` block asserts the opposite — the key MUST still be set.
    """
    for cfg, key, body in _all_blocks():
        if _annotation_class(body) == "RETAINED":
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


def test_every_annotated_key_has_its_literal_surface_pinned() -> None:
    """The completeness re-derivation, for EVERY key, over the literal surface.

    The ``DEAD KEY`` phrase gate re-derives three keys. This re-derives all 27,
    and it is the check that would have caught ``scripts/value_optimism.py``
    reading ``sf_search_dampen_sf_low``: that file is in the key's literal
    surface, so pinning the surface forces someone to look at it.

    Why literals and not raw text: a flattened-config key is reached by NAME, so
    a consumer that breaks when the key is deleted must contain the key as a
    ``str`` constant. The raw-text surface is 318 lines over 156 (key, file)
    pairs and is dominated by attribute reads, ``--flag`` prose and comments,
    none of which can break; the literal surface is 179 over 104 and is exact.
    """
    keys = set(EXPECTED_ANNOTATIONS)
    names = _spellings(keys)
    sources = _literal_source_files()
    assert sources, "no source files scanned — the glob list is wrong, not the repo"

    found: dict[str, set[str]] = {k: set() for k in keys}
    for rel, mod in sources:
        for node in ast.walk(mod):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in names:
                found[names[node.value]].add(rel)

    moved = [
        f"{k}: pinned {tuple(sorted(KEY_LITERAL_SURFACES.get(k, ())))}, tree says {tuple(sorted(found.get(k, set())))}"
        for k in sorted(keys | set(KEY_LITERAL_SURFACES))
        if found.get(k, set()) != set(KEY_LITERAL_SURFACES.get(k, ()))
    ]
    assert not moved, (
        "the set of code files that name an annotated key as a string literal "
        "moved. A NEW file naming a key this config no longer sets is a consumer "
        "the annotation has never been judged against — re-judge it, do not "
        "extend the pin reflexively:\n  " + "\n  ".join(moved)
    )


def test_no_deleted_key_sits_in_a_collection_whose_consumer_requires_it() -> None:
    """The shape a name-based scan cannot see: constant collection + loop variable.

    ``scripts/value_optimism.py`` does not write ``flat.get("sf_search_dampen_sf_low",
    0.0)``. It puts the key in a module-level tuple and loops over it, raising
    ``SystemExit`` when the key is absent from the config. Six rounds of scans
    for ``.get(key, default)`` and ``tc.<field>`` shapes never saw it. So the
    collections are enumerated from the AST and each carries a verdict.
    """
    keys = set(EXPECTED_ANNOTATIONS)
    names = _spellings(keys)
    found: dict[tuple[str, str], set[str]] = {}
    for rel, mod in _literal_source_files():
        for const, ks in _module_level_collections(rel, mod, names).items():
            found[(rel, const)] = ks

    moved = [
        f"{loc}: pinned {tuple(sorted(MODULE_LEVEL_KEY_COLLECTIONS.get(loc, ())))}, tree says {tuple(sorted(found.get(loc, set())))}"
        for loc in sorted(set(found) | set(MODULE_LEVEL_KEY_COLLECTIONS))
        if found.get(loc, set()) != set(MODULE_LEVEL_KEY_COLLECTIONS.get(loc, ()))
    ]
    assert not moved, (
        "the module-level constant collections naming an annotated key moved. "
        "Each one needs an explicit verdict on whether the key's ABSENCE changes "
        "what the consumer does, and a presence-requiring one must be registered "
        "in PRESENCE_REQUIRING_COLLECTIONS:\n  " + "\n  ".join(moved)
    )

    unregistered = sorted(PRESENCE_REQUIRING_COLLECTIONS - set(MODULE_LEVEL_KEY_COLLECTIONS))
    assert not unregistered, (
        f"PRESENCE_REQUIRING_COLLECTIONS names collections that no longer exist: "
        f"{unregistered}. A stale entry guards nothing."
    )

    violations = [
        f"{key!r} is annotated DELETED but {loc[0]}::{loc[1]} requires it to be present"
        for loc in sorted(PRESENCE_REQUIRING_COLLECTIONS)
        for key in sorted(MODULE_LEVEL_KEY_COLLECTIONS[loc])
        if EXPECTED_ANNOTATIONS.get(key) == "DELETED"
    ]
    assert not violations, (
        "a key was deleted from the config while a consumer refuses to run "
        "without it. This is not a documentation defect — the script exits at "
        "startup:\n  " + "\n  ".join(violations)
    )


def test_every_retained_key_has_its_retention_mechanism_re_derived() -> None:
    """A `⚑ RETAINED` block's reason is checked against source, not trusted.

    Retention rationales were the last unguarded prose here: inverting the
    measurement so the block claimed the value was unchanged left every test
    green. Now the fallback is read out of the consumer's AST, compared against
    what the yaml actually ships, and both numbers must appear in the block.
    """
    blocks = {key: (cfg, body) for cfg, key, body in _all_blocks()}
    retained = sorted(k for k, v in EXPECTED_ANNOTATIONS.items() if v == "RETAINED")
    missing = [k for k in retained if k not in RETENTION_MECHANISMS]
    assert not missing, (
        f"retained keys with no re-derivable mechanism: {missing}. A key kept "
        f"only because a comment says so is exactly what this guard refuses."
    )
    stale = sorted(set(RETENTION_MECHANISMS) - set(retained))
    assert not stale, f"RETENTION_MECHANISMS entries for non-retained keys: {stale}"

    parsed = dict(_literal_source_files())
    failures: list[str] = []
    for key in retained:
        kind, rel, expected_fallback = RETENTION_MECHANISMS[key]
        cfg, body = blocks[key]
        shipped = _yaml_value(cfg, key)
        if rel not in parsed:
            failures.append(f"{key}: consumer {rel} is not in the scanned source set")
            continue
        if f"[{rel}]" not in body and f"[{rel.removeprefix('chess_anti_engine/')}]" not in body:
            failures.append(f"{key}: the retention block does not cite its consumer [{rel}]")

        if kind == "requires_presence":
            locs = [
                loc for loc in PRESENCE_REQUIRING_COLLECTIONS
                if loc[0] == rel and key in MODULE_LEVEL_KEY_COLLECTIONS.get(loc, ())
            ]
            if not locs:
                failures.append(
                    f"{key}: annotated as retained because {rel} requires it, but no "
                    f"presence-requiring collection there names it"
                )
            continue

        actual = (
            _flat_get_fallback(rel, parsed[rel], key) if kind == "flat_get"
            else _argparse_default(rel, parsed[rel], key)
        )
        if actual is None:
            failures.append(f"{key}: no {kind} consumer for it in {rel}; the retention reason died")
            continue
        if actual != expected_fallback:
            failures.append(f"{key}: {rel} falls back to {actual!r}, pinned {expected_fallback!r}")
            continue
        if actual == shipped:
            failures.append(
                f"{key}: fallback {actual!r} EQUALS the shipped value {shipped!r}, so the "
                f"stated reason for retaining it is false"
            )
        prose = _prose(body)
        failures.extend(
            f"{key}: the retention block never states the value {number!r}"
            for number in (shipped, actual)
            if not re.search(rf"(?<![\w.]){re.escape(number)}(?!\w)", prose)
        )

    assert not failures, (
        "a `⚑ RETAINED` block's reason no longer matches the code it is about:\n  "
        + "\n  ".join(failures)
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
