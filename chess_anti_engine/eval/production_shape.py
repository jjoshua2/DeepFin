"""Prove an instrument measured PRODUCTION, and fail loudly when it did not.

This module exists because of a repeated, measured failure mode: an offline
instrument hand-lists the production settings it cares about, the hand-list
drifts as the live yaml is tuned, and the instrument then reports a confident
number about a configuration production does not run. The number is not noisy
— it is precise, reproducible and about the wrong object, which is why it
survives review.

The two rules that shape everything here:

* **A presence check is not a value read.** ``"gumbel_c_scale" in flat`` and
  ``flat["gumbel_c_scale"] == production's value`` are different assertions.
  Every helper below compares VALUES.
* **A guard must share the criterion's instrument.** These helpers do not
  re-derive production's search shape from a parallel hand-list — that is the
  defect, one level up. They call the SAME function production selfplay calls
  (``selfplay.network_turn.build_selfplay_search_shape``) through the SAME
  config chain (``TrialConfig.from_dict`` ->
  ``trainable_config_ops._play_batch_kwargs``). A field added to that builder
  is therefore picked up automatically; a field an instrument overrides
  without declaring it shows up as a diff and stops the run.
* **⚑ The compared object is what reaches the CONSUMER, not what one
  dataclass declares.** The first revision of this module diffed
  ``GumbelConfig`` field-complete and called that exhaustive. It was not:
  ``gumbel_vloss_weight`` and ``gumbel_target_batch`` reach the C runner as
  plain keyword arguments and are not ``GumbelConfig`` fields, so a config
  differing from live on ``gumbel_vloss_weight`` alone was not refused and was
  stamped authoritative. "Exhaustive over <schema>" is only ever as complete
  as <schema>; the fix is to move the boundary to the runner's own argument
  set (``SelfplaySearchShape``), which production builds and unpacks, and not
  to pick a roomier dataclass.

What this module still does NOT cover, stated rather than implied: the
per-ply root-noise schedule (``per_game_add_noise`` /
``per_game_gumbel_scale``, from ``_scheduled_gumbel_scale``). It is a function
of the move number, not a scalar, so no field comparison can express it. An
instrument whose search runs with noise on must say what it does about the
schedule; ``shape_coverage_note`` is the one line to print.

Which file counts as "production" is deliberately explicit. The in-tree
``configs/pbt2_small.yaml`` is stale by construction on every branch except
the live one, because the live branch's working copy is the only writer and
its edits are frequently uncommitted. So the live path is named by the
``CHESS_ANTI_ENGINE_LIVE_CONFIG`` environment variable, and when that is unset
the fallback to the in-tree copy is reported as a non-authoritative
provenance rather than silently substituted. An instrument that hard-crashes
when run off this machine would be a regression, so the degradation is loud,
not fatal.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.selfplay.network_turn import SelfplaySearchShape

# The live production yaml is named by this variable. It holds an absolute
# path on the training host; the repo is public, so the path itself must not
# be committed anywhere.
LIVE_CONFIG_ENV = "CHESS_ANTI_ENGINE_LIVE_CONFIG"

# Fallback, relative to the repo root. Correct ONLY when the instrument is run
# from the live working tree; everywhere else this file is a stale snapshot.
DEFAULT_RELATIVE_CONFIG = Path("configs") / "pbt2_small.yaml"

# "The config does not have this key at all." A shared SENTINEL rather than a
# per-caller `.get(key, <some plausible default>)`, because a plausible default
# is how a probe reports MATCHES against a config that never mentions the key —
# the same shape as the stale hard-coded `3.0` that started all of this. Every
# caller must branch on it; none may treat it as a value.
CONFIG_ABSENT = "<absent>"


def repo_root() -> Path:
    """The checkout this module was imported from."""
    return Path(__file__).resolve().parent.parent.parent


@dataclasses.dataclass(frozen=True)
class LiveConfig:
    """A resolved production config, carrying WHERE it came from.

    ``authoritative`` is the whole point: a caller must be able to tell "this
    is the file the live trainer re-reads" from "this is the in-tree copy,
    which may be stale". Both are usable; only one of them settles an argument
    about what production runs.
    """

    path: Path
    flat: dict[str, Any]
    sha256: str
    provenance: str
    authoritative: bool

    def header(self) -> str:
        """One line naming the file, its digest and whether it is the live one."""
        mark = "LIVE" if self.authoritative else "NOT-LIVE"
        return (
            f"[shape] production config [{mark}]: {self.path} "
            f"sha256={self.sha256[:12]} ({self.provenance})"
        )


def resolve_live_config_path() -> tuple[Path, str, bool]:
    """``(path, provenance, authoritative)`` for the production config.

    Never raises: a missing file is reported by the caller, because the right
    reaction differs between an instrument that cannot proceed without the
    live shape and one that only wants to annotate its header.
    """
    env = os.environ.get(LIVE_CONFIG_ENV, "").strip()
    if env:
        return Path(env).expanduser(), f"${LIVE_CONFIG_ENV}", True
    return (
        repo_root() / DEFAULT_RELATIVE_CONFIG,
        f"in-tree fallback; set ${LIVE_CONFIG_ENV} to name the live file",
        False,
    )


def load_config_file(
    path: Path, *, provenance: str, authoritative: bool,
) -> tuple[LiveConfig | None, str]:
    """``(config, reason it is unavailable)`` for one explicit path.

    Exactly one of the two is meaningful: on success the reason is ``""``, on
    failure the config is ``None`` and the reason NAMES the failure.

    ⚑ The reason string is not decoration. Three different states used to
    collapse into a bare ``None`` — the variable is unset, the file it names
    does not exist, and the file exists but will not flatten — and they call
    for three different operator actions (export it / fix the path / rebase
    onto the branch whose schema defines the live yaml's new key). A caller
    that prints "unset or unreadable" for all three sends the reader to the
    wrong one two times out of three.
    """
    from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

    if not path.is_file():
        return None, f"{path} does not exist ({provenance})"
    raw_bytes = path.read_bytes()
    try:
        flat = flatten_run_config_defaults(load_yaml_file(str(path)))
    except Exception as exc:
      # Broad on purpose: yaml syntax, an unknown key rejected by the schema,
      # a bad value. They are NOT interchangeable to the reader, so the
      # exception's own text is carried out rather than swallowed — an unknown
      # key in particular means this checkout's schema predates the running
      # config, and the fix is to rebase, never to delete the key from the
      # live file.
        return None, f"{path} does not flatten: {type(exc).__name__}: {exc}"
    return LiveConfig(
        path=path,
        flat=dict(flat),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        provenance=provenance,
        authoritative=authoritative,
    ), ""


def load_live_config_or_reason() -> tuple[LiveConfig | None, str]:
    """``load_live_config()``, plus WHY it is unavailable when it is.

    Callers that print a degradation warning should use this one, so the
    warning distinguishes the three failure states instead of announcing the
    union of them.
    """
    path, provenance, authoritative = resolve_live_config_path()
    return load_config_file(path, provenance=provenance, authoritative=authoritative)


def load_live_config() -> LiveConfig | None:
    """Load and flatten the production config, or ``None`` if it is unreadable.

    Returns ``None`` rather than raising, and every caller in the tree degrades
    loudly on it: these instruments are run off the training host often enough
    (foreign nets, historical checkpoints, CI) that a hard requirement would be
    a regression. A caller that degrades must SAY so in its report — a silent
    substitution is the failure this module exists to stop.

    ⚑ There is deliberately no ``required=True`` mode. One was written and
    removed unused: a flag whose only branch nothing takes is untested code
    that reads as a safety property. What a caller that *cannot* proceed on a
    stale config does instead is refuse on ``authoritative`` being False —
    see ``scripts/audit_targets.py``.

    Use ``load_live_config_or_reason`` when the ``None`` is going to be
    reported to a human: this wrapper drops the diagnosis.
    """
    return load_live_config_or_reason()[0]


# ---------------------------------------------------------------------------
# The production search shape, via production's own builder
# ---------------------------------------------------------------------------


def production_selfplay_gumbel_config(
    flat: dict[str, Any], *, simulations: int,
) -> GumbelConfig:
    """The ``GumbelConfig`` production SELFPLAY builds from ``flat``.

    Deliberately routed through the production chain rather than reassembled
    from ``flat.get(...)`` calls:

        flat -> TrialConfig.from_dict -> _play_batch_kwargs -> SearchConfig
             -> build_selfplay_gumbel_config -> GumbelConfig

    Every link is the one the worker uses. A knob that stops reaching the
    search in production therefore stops reaching it here too, so an
    instrument checking itself against this cannot certify a wiring that is
    broken on both sides — it fails with production instead of diverging
    silently from it.
    """
    return production_search_shape(flat, simulations=simulations).cfg


def production_search_shape(
    flat: dict[str, Any], *, simulations: int,
) -> SelfplaySearchShape:
    """EVERYTHING config-derived that production SELFPLAY hands its runner.

    Same chain as ``production_selfplay_gumbel_config``, one level out:
    ``build_selfplay_search_shape`` rather than ``build_selfplay_gumbel_config``.
    That extra level is the whole F1 fix — ``vloss_weight`` and
    ``target_batch`` live there and nowhere in ``GumbelConfig``.

    This is the object every guard in this module compares. Use it, not the
    inner ``GumbelConfig``, whenever the question is "is this the search
    production runs".
    """
    from chess_anti_engine.selfplay.network_turn import build_selfplay_search_shape
    from chess_anti_engine.tune.trainable_config_ops import _play_batch_kwargs
    from chess_anti_engine.tune.trial_config import TrialConfig

    tc = TrialConfig.from_dict(flat)
    groups = _play_batch_kwargs(tc)
    return build_selfplay_search_shape(
        search=groups["search"], game=groups["game"], simulations=int(simulations),
    )


def production_input_encoding(flat: dict[str, Any]) -> dict[str, str]:
    """``{input_history_encoding, input_extra_features}`` production selfplay uses.

    Read off the config production's builder produces, not off ``flat`` keys by
    name — the yaml key and the field are not always spelled the same, and a
    name-based lookup is how an instrument ends up comparing against a key
    nothing consumes.
    """
    cfg = production_selfplay_gumbel_config(flat, simulations=1)
    return {
        "input_history_encoding": str(cfg.input_history_encoding),
        "input_extra_features": str(cfg.input_extra_features),
    }


@dataclasses.dataclass(frozen=True)
class FieldDiff:
    """One field on which a realized config differs from production's."""

    field: str
    realized: object
    production: object

    def __str__(self) -> str:
        return f"{self.field}: realized {self.realized!r} vs production {self.production!r}"


# The runner arguments that are NOT ``GumbelConfig`` fields. Named here so
# every consumer of this module compares the same surface, and so the two
# names appear in exactly one place rather than in each instrument.
#
# ⚑ This is a hand-list, and it is the ONLY one left — but it is a hand-list
# over ``SelfplaySearchShape``'s own non-cfg fields, which is checkable:
# ``tests/test_production_shape_guard.py`` derives the expected set from
# ``dataclasses.fields(SelfplaySearchShape)`` and fails if a field is added to
# the shape without landing here. A list that a test regenerates is not the
# drift mechanism #227 was about.
RUNNER_ARG_FIELDS: tuple[str, ...] = ("target_batch", "vloss_weight")

# What a clean ``shape_field_diff`` does and does not prove. Printed by
# instruments beside the affirmative line, because "matches production" with
# no scope is how a reader concludes more than was checked.
SHAPE_COVERAGE_NOTE = (
    "covers every GumbelConfig field plus "
    f"{list(RUNNER_ARG_FIELDS)} (the complete argument set production's "
    "selfplay hands its search runner). NOT covered: the per-ply root-noise "
    "schedule (per_game_add_noise / per_game_gumbel_scale), which is a "
    "function of the move number and has no field to compare."
)


def shape_coverage_note(prefix: str = "[shape]") -> str:
    """One line naming exactly what a clean shape comparison proved."""
    return f"{prefix}   scope: {SHAPE_COVERAGE_NOTE}"


def shape_field_diff(
    realized: SelfplaySearchShape,
    production: SelfplaySearchShape,
    *,
    exempt: dict[str, str],
) -> list[FieldDiff]:
    """Every difference across the runner's COMPLETE argument set.

    ``GumbelConfig`` fields plus ``RUNNER_ARG_FIELDS``. The second half is the
    F1 fix: a config that differs from live only on ``gumbel_vloss_weight``
    used to produce an empty diff and an affirmative "matches production"
    line, because the comparison object was the dataclass rather than the
    consumer.
    """
    out = gumbel_field_diff(realized.cfg, production.cfg, exempt=exempt)
    for name in RUNNER_ARG_FIELDS:
        if name in exempt:
            continue
        got = getattr(realized, name)
        want = getattr(production, name)
        if got != want:
            out.append(FieldDiff(field=name, realized=got, production=want))
    return out


def gumbel_field_diff(
    realized: GumbelConfig,
    production: GumbelConfig,
    *,
    exempt: dict[str, str],
) -> list[FieldDiff]:
    """Fields where ``realized`` differs from ``production``, minus ``exempt``.

    ⚑ Not a certification on its own — it sees only ``GumbelConfig``. Callers
    asking "is this production's search" want ``shape_field_diff``.

    ``exempt`` maps field name -> the REASON the instrument deliberately
    deviates. It is a dict rather than a set so that an undocumented deviation
    is impossible to add: there is nowhere to put the field without also
    writing down why.

    Note what this does NOT do — it never asks whether a field is "set from
    the yaml". Production's builder may pass a literal, a default or a config
    value, and from the instrument's side the distinction is irrelevant: what
    matters is whether the search it is about to run agrees with the search
    production runs.
    """
    out: list[FieldDiff] = []
    for f in dataclasses.fields(production):
        if f.name in exempt:
            continue
        got = getattr(realized, f.name)
        want = getattr(production, f.name)
        if isinstance(want, float) and isinstance(got, (int, float)):
            if abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want))):
                continue
        elif got == want:
            continue
        out.append(FieldDiff(field=f.name, realized=got, production=want))
    return out


def format_shape_table(
    realized: SelfplaySearchShape,
    production: SelfplaySearchShape,
    *,
    exempt: dict[str, str],
    prefix: str = "[shape]",
) -> str:
    """Realized-vs-production for every RUNNER ARGUMENT, deviations called out.

    Printed unconditionally by callers, including on the success path. A guard
    that only speaks up when it fails cannot be distinguished from a guard that
    is not running — and "not running" is this repo's signature defect.

    Ends with ``shape_coverage_note``: the table is complete over the runner's
    arguments and silent about the per-ply noise schedule, and a reader who is
    not told the scope will assume the wider one.
    """
    lines: list[str] = []
    pairs = [
        (f.name, getattr(realized.cfg, f.name), getattr(production.cfg, f.name))
        for f in dataclasses.fields(production.cfg)
    ] + [
        (name, getattr(realized, name), getattr(production, name))
        for name in RUNNER_ARG_FIELDS
    ]
    for name, got, want in pairs:
        if name in exempt:
            if got != want:
                lines.append(
                    f"{prefix}   {name}: {got!r} (production {want!r}) "
                    f"— DELIBERATE: {exempt[name]}"
                )
            continue
        if got != want:
            lines.append(f"{prefix}   {name}: {got!r} != production {want!r}  <-- DRIFT")
        else:
            lines.append(f"{prefix}   {name}: {got!r}")
    lines.append(shape_coverage_note(prefix))
    return "\n".join(lines)


def assert_matches_production(
    realized: SelfplaySearchShape,
    production: SelfplaySearchShape,
    *,
    exempt: dict[str, str],
    where: str,
) -> None:
    """Refuse to run when the realized search shape is not production's.

    The failing input is concrete and easy to produce: set any production
    search key in the live yaml that the instrument does not carry, and this
    raises. That is the mutation ``tests/test_production_shape_guard.py``
    runs — including on ``gumbel_vloss_weight``, which is a runner argument
    rather than a ``GumbelConfig`` field and which the pre-F1 comparison could
    not see.
    """
    diffs = shape_field_diff(realized, production, exempt=exempt)
    if not diffs:
        return
    detail = "\n  ".join(str(d) for d in diffs)
    raise SystemExit(
        f"[shape] {where}: the search this run would score is NOT the search "
        f"production runs:\n  {detail}\n"
        "  Either carry the field through to this instrument, or add it to the "
        "exempt map with the reason it deliberately differs. Refusing to run — "
        "the numbers would be precise and about the wrong configuration."
    )


# ---------------------------------------------------------------------------
# Value-level comparison between two configs
# ---------------------------------------------------------------------------


def compare_config_values(
    realized: dict[str, Any],
    production: dict[str, Any],
    keys: tuple[str, ...],
) -> list[FieldDiff]:
    """VALUE-compare selected keys across two flattened configs.

    Absence is a difference, not a pass: a key missing from ``realized`` while
    production sets it is exactly the drift being hunted, so it is reported
    with a ``<absent>`` sentinel rather than skipped. This is the "a presence
    check is not a value read" rule in its most literal form.
    """
    out: list[FieldDiff] = []
    for key in keys:
        got = realized.get(key, CONFIG_ABSENT)
        want = production.get(key, CONFIG_ABSENT)
        if isinstance(want, float) and isinstance(got, (int, float)):
            if abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want))):
                continue
        elif got == want:
            continue
        out.append(FieldDiff(field=key, realized=got, production=want))
    return out
