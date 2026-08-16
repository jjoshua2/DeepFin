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
  (``selfplay.network_turn.build_selfplay_gumbel_config``) through the SAME
  config chain (``TrialConfig.from_dict`` ->
  ``trainable_config_ops._play_batch_kwargs``). A field added to that builder
  is therefore picked up automatically; a field an instrument overrides
  without declaring it shows up as a diff and stops the run.

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

# The live production yaml is named by this variable. It holds an absolute
# path on the training host; the repo is public, so the path itself must not
# be committed anywhere.
LIVE_CONFIG_ENV = "CHESS_ANTI_ENGINE_LIVE_CONFIG"

# Fallback, relative to the repo root. Correct ONLY when the instrument is run
# from the live working tree; everywhere else this file is a stale snapshot.
DEFAULT_RELATIVE_CONFIG = Path("configs") / "pbt2_small.yaml"


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


def load_live_config() -> LiveConfig | None:
    """Load and flatten the production config, or ``None`` if it is unreadable.

    Returns ``None`` rather than raising, and every caller in the tree degrades
    loudly on it: these instruments are run off the training host often enough
    (foreign nets, historical checkpoints, CI) that a hard requirement would be
    a regression. A caller that degrades must SAY so in its report — a silent
    substitution is the failure this module exists to stop.

    ⚑ There is deliberately no ``required=True`` mode. One was written and
    removed unused: a flag whose only branch nothing takes is untested code
    that reads as a safety property.

    A yaml that does not FLATTEN is treated the same way, and it usually means
    something specific: the live yaml carries a key this checkout's schema does
    not define, i.e. the instrument's code predates the running config. The fix
    is to rebase onto the branch that defines the key, never to delete the key
    from the live file.
    """
    from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

    path, provenance, authoritative = resolve_live_config_path()
    if not path.is_file():
        return None
    raw_bytes = path.read_bytes()
    try:
        flat = flatten_run_config_defaults(load_yaml_file(str(path)))
    except Exception:
      # Broad on purpose: yaml syntax, an unknown key rejected by the schema,
      # a bad value — every one of them means "this file cannot tell us what
      # production runs", and the caller's response is identical.
        return None
    return LiveConfig(
        path=path,
        flat=dict(flat),
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        provenance=provenance,
        authoritative=authoritative,
    )


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
    from chess_anti_engine.selfplay.network_turn import build_selfplay_gumbel_config
    from chess_anti_engine.tune.trainable_config_ops import _play_batch_kwargs
    from chess_anti_engine.tune.trial_config import TrialConfig

    tc = TrialConfig.from_dict(flat)
    groups = _play_batch_kwargs(tc)
    return build_selfplay_gumbel_config(
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


def gumbel_field_diff(
    realized: GumbelConfig,
    production: GumbelConfig,
    *,
    exempt: dict[str, str],
) -> list[FieldDiff]:
    """Fields where ``realized`` differs from ``production``, minus ``exempt``.

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
    realized: GumbelConfig,
    production: GumbelConfig,
    *,
    exempt: dict[str, str],
    prefix: str = "[shape]",
) -> str:
    """Realized-vs-production for every field, with deviations called out.

    Printed unconditionally by callers, including on the success path. A guard
    that only speaks up when it fails cannot be distinguished from a guard that
    is not running — and "not running" is this repo's signature defect.
    """
    lines: list[str] = []
    for f in dataclasses.fields(production):
        got = getattr(realized, f.name)
        want = getattr(production, f.name)
        if f.name in exempt:
            if got != want:
                lines.append(
                    f"{prefix}   {f.name}: {got!r} (production {want!r}) "
                    f"— DELIBERATE: {exempt[f.name]}"
                )
            continue
        if got != want:
            lines.append(f"{prefix}   {f.name}: {got!r} != production {want!r}  <-- DRIFT")
        else:
            lines.append(f"{prefix}   {f.name}: {got!r}")
    return "\n".join(lines)


def assert_matches_production(
    realized: GumbelConfig,
    production: GumbelConfig,
    *,
    exempt: dict[str, str],
    where: str,
) -> None:
    """Refuse to run when the realized search shape is not production's.

    The failing input is concrete and easy to produce: set any production
    search key in the live yaml that the instrument does not carry, and this
    raises. That is the mutation ``tests/test_production_shape_guard.py``
    runs.
    """
    diffs = gumbel_field_diff(realized, production, exempt=exempt)
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
    absent = "<absent>"
    out: list[FieldDiff] = []
    for key in keys:
        got = realized.get(key, absent)
        want = production.get(key, absent)
        if isinstance(want, float) and isinstance(got, (int, float)):
            if abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want))):
                continue
        elif got == want:
            continue
        out.append(FieldDiff(field=key, realized=got, production=want))
    return out
