"""Two-world classification for keys the LIVE yaml arms and ``main``'s does not.

⚑⚑ ``configs/pbt2_small.yaml`` IS TWO DIFFERENT FILES DEPENDING ON THE BRANCH.
On ``main`` it is a copy that has not moved since the live branch diverged; on
``ops/live-20260725`` it IS the file the running trial re-reads every iteration.
A pin written as "no config ships this key" is therefore true on one branch and
false on the other, and a test that states it unconditionally is not asserting a
property of the repository — it is asserting which branch it happens to be on.

The pattern every consumer here uses instead: a key may be in exactly TWO
states, and any third state fails.

  * **absent** — ``main``'s world. Not one key of the bundle is set anywhere in
    the file. This is the state the original single-world assertions pinned.
  * **armed** — ``ops/live-20260725``'s world. EVERY key of the bundle is set,
    exactly once, at the value recorded by the caller, which is the value a
    named, ledger-pre-registered restart commit put there.

Everything else — a partially armed bundle, a key bound twice, a key bound to
something other than its pin — is the third state, and it fails with the key
named. That is the point: the pin stays the arbiter, and the live file is
accepted only when it EQUALS the pinned world rather than merely resembling it.

⚑ Presence is read from the PARSED yaml, not from a substring scan of the text.
The live file documents these bundles at length in comments directly above the
keys, so ``"era_probe_path" in path.read_text()`` reports the prose as an
offender. Values are read from the raw mapping rather than through
``flatten_run_config_defaults`` for the opposite reason: the flattener fills
absent keys with defaults, so it cannot tell "absent" from "set to the default",
which is the exact distinction the two worlds turn on.

⚑ Paths are pinned by ``PathEndingIn`` rather than in full. THE REPO IS PUBLIC
and must not accumulate local absolute paths; the part of an artifact path that
identifies WHICH ruler was armed is the trailing ``data/<dir>/<file>``, and the
machine-specific prefix identifies nothing.

⚑⚑ THE RESIDUAL A TWO-WORLD PIN CANNOT CLOSE, stated rather than implied
away: a WHOLE-FILE overwrite of the live yaml with main's copy flips every
bundle to ABSENT consistently, and each pin then reads a legitimate world.
No unit test can tell "this tree is the live branch" without a lineage
oracle, and the yaml deliberately carries no such key (adding one is a live
key edit). The protections that DO cover it are the cross-bundle same-world
consistency test (a PARTIAL merge — the realistic accident — disarms some
bundles and not others, and fails there), the standing never-``git
checkout``-in-the-live-tree rule the overwrite would have to violate, and
the fact that a full overwrite also reverts 55 realized keys and cannot
survive the first iteration's regime lines unnoticed.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
PRODUCTION_CONFIG = CONFIGS_DIR / "pbt2_small.yaml"

ABSENT = "absent"
ARMED = "armed"


@dataclasses.dataclass(frozen=True)
class PathEndingIn:
    """A pin on a path-valued key that names the artifact and not the machine.

    ``value.endswith(suffix)`` — so ``data/era_probe/era_20260804.npz`` pins
    WHICH frozen set was armed (the thing a re-cut would change, and the thing
    the column's meaning depends on) while leaving the operator's home directory
    out of a public repository.
    """

    suffix: str

    def matches(self, value: object) -> bool:
        return isinstance(value, str) and value.endswith(self.suffix)

    def __str__(self) -> str:
        return f"<a path ending in {self.suffix!r}>"


@dataclasses.dataclass(frozen=True)
class ArmingState:
    """The classification, plus the diagnostics for a third state.

    ``problems`` is empty iff the file is in one of the two known worlds. It is
    returned rather than asserted so each caller keeps its own protective-intent
    message in its own test module, where pytest can rewrite the assertion.
    """

    world: str
    problems: list[str]
    values: dict[str, list[Any]]

    @property
    def is_armed(self) -> bool:
        return self.world == ARMED


def _bindings_in_node(node: yaml.Node, key: str) -> list[yaml.Node]:
    """Every value node ``key`` is bound to, on the COMPOSED yaml tree.

    ⚑ The composed tree, not ``safe_load``'s dicts, because ``safe_load``
    collapses duplicate keys within one mapping to the last value BEFORE any
    caller can look — so "bound exactly once" checked on the loaded dict is
    vacuous for the classic copy-paste double binding (caught in review of
    PR #488). Node-level walking preserves every occurrence.

    Recursive rather than section-scoped so a key that MOVES between
    ``tune:`` and ``selfplay:`` (or into a section the flattener ignores) is
    still SEEN — whether a sighting counts as armed is the flattener's call,
    made in ``classify_production_arming``.
    """
    found: list[yaml.Node] = []
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
                found.append(value_node)
            else:
                found.extend(_bindings_in_node(value_node, key))
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            found.extend(_bindings_in_node(item, key))
    return found


def _construct(node: yaml.Node) -> Any:
    loader = yaml.SafeLoader("")
    try:
        return loader.construct_object(node, deep=True)
    finally:
        loader.dispose()


def yaml_bindings_from_text(text: str, key: str) -> list[Any]:
    """Every value ``key`` is bound to anywhere in the document — duplicates
    within one mapping included."""
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    if root is None:
        return []
    return [_construct(node) for node in _bindings_in_node(root, key)]


def yaml_bindings(node: object, key: str) -> list[Any]:
    """Every value ``key`` is bound to anywhere in a PARSED yaml tree.

    ⚑ Kept for callers that already hold a loaded dict, and honest about its
    blind spot: a duplicate binding inside ONE mapping was collapsed by the
    loader before this function ran. ``yaml_bindings_from_text`` sees those;
    the classifier uses it.
    """
    found: list[Any] = []
    if isinstance(node, Mapping):
        for name, value in node.items():
            if name == key:
                found.append(value)
            else:
                found.extend(yaml_bindings(value, key))
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
        for item in node:
            found.extend(yaml_bindings(item, key))
    return found


def _matches(pin: object, value: object) -> bool:
    if isinstance(pin, PathEndingIn):
        return pin.matches(value)
    if isinstance(pin, bool) or isinstance(value, bool):
        return pin is value
    if isinstance(pin, (int, float)) and isinstance(value, (int, float)):
        return float(pin) == float(value)
    return pin == value


def classify_production_arming(
    pinned: Mapping[str, object], *, config: Path | None = None,
) -> ArmingState:
    """Classify ``config`` against a bundle of pinned armed values.

    Returns ``ABSENT`` when the file sets none of the keys, ``ARMED`` when it
    sets all of them exactly once at their pins, and a state carrying
    ``problems`` otherwise.

    ⚑ ``config`` is resolved HERE and not in the signature's default. A default
    bound at def-time cannot be redirected, and a test whose reference file is
    fixed at import is a test whose mutant can only be run by editing the LIVE
    yaml — which, on the live branch, is editing a file a running trial re-reads
    every iteration. Callers pass their own module-level ``PRODUCTION_CONFIG``
    so a mutation harness has exactly one name to redirect per module.
    """
    path = PRODUCTION_CONFIG if config is None else config
    text = path.read_text(encoding="utf-8")
    values = {key: yaml_bindings_from_text(text, key) for key in pinned}
    problems: list[str] = []

    present = sorted(key for key, bound in values.items() if bound)
    if not present:
        return ArmingState(world=ABSENT, problems=problems, values=values)

    missing = sorted(key for key, bound in values.items() if not bound)
    if missing:
        problems.append(
            f"{path.name} arms {present} but not {missing} — a PARTIALLY armed "
            "bundle is neither of the two known worlds"
        )

    for key in present:
        bound = values[key]
        if len(bound) != 1:
            problems.append(
                f"{path.name}: {key!r} is bound {len(bound)} times ({bound!r}); "
                "which one wins depends on parse order"
            )
            continue
        if not _matches(pinned[key], bound[0]):
            problems.append(
                f"{path.name}: {key!r} is {bound[0]!r}, pinned {pinned[key]!s}"
            )

    # ⚑ THE EFFECTIVENESS CROSS-CHECK, using the production flattener as the
    # oracle. A binding sighted by the raw walk can sit somewhere the loader
    # never reads (nested inside another key's list value, say) — this
    # codebase's signature defect is a value that is accepted and then
    # silently ignored, and a guard that counts an INERT binding as armed
    # would be that defect inside the instrument built to catch it (review of
    # PR #488). ARMED therefore additionally requires each pinned value to
    # SURVIVE ``flatten_run_config_defaults`` — the same function the trial
    # boots through — so "armed" always means "reaches the run", never merely
    # "appears in the file".
    if not problems:
        from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults

        flat = flatten_run_config_defaults(yaml.safe_load(text))
        for key in present:
            realized = flat.get(key)
            if not _matches(pinned[key], realized):
                problems.append(
                    f"{path.name}: {key!r} is bound at its pin in the file but "
                    f"realizes {realized!r} through flatten_run_config_defaults "
                    "— the binding sits somewhere the loader does not read, so "
                    "the file LOOKS armed while the run is not"
                )

    return ArmingState(
        world=ARMED if not problems else "mixed", problems=problems, values=values,
    )


def production_bindings(key: str, *, config: Path | None = None) -> list[Any]:
    """Every value the production yaml binds ``key`` to, RAW.

    For the knobs whose loader COERCES — ``bool(v)`` turns ``0.5`` on, and
    ``max(0, int(v))`` turns ``0.5`` off — the realized value alone cannot say
    which world the file is in, because two very different yaml lines realize
    the same run. This is the other half of that pair.
    """
    path = PRODUCTION_CONFIG if config is None else config
    return yaml_bindings_from_text(path.read_text(encoding="utf-8"), key)


def other_configs_mentioning(keys: Sequence[str]) -> list[str]:
    """``name:key`` for every config OTHER than production that names a key.

    Deliberately a text scan and deliberately "mentions", not "sets": the
    research configs have no business documenting these bundles either, and the
    stricter reading is the one the single-world assertions already applied to
    them. Production is excluded because it is the file the two worlds are about
    — and because on the live branch it explains each armed bundle in prose
    directly above the keys.
    """
    paths = sorted(CONFIGS_DIR.glob("*.yaml"))
    assert paths, "no configs found; the glob is wrong and the caller is vacuous"
    return sorted(
        f"{path.name}:{key}"
        for path in paths
        if path != PRODUCTION_CONFIG
        for key in keys
        if key in path.read_text(encoding="utf-8")
    )
