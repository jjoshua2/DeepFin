"""⚑⚑ EVERY two-world bundle must read the SAME world — the cross-bundle gate.

``tests/live_yaml_arming.py`` classifies each armed bundle independently, and
its module docstring states the residual that independence leaves open: a
WHOLE-FILE overwrite of ``configs/pbt2_small.yaml`` flips every bundle at once
and each per-bundle pin then reads a legitimate world (review of PR #488,
finding 7). No unit test can close that without a lineage oracle the yaml
deliberately does not carry.

What a test CAN close is the realistic accident: a PARTIAL merge. A cherry-pick
of one restart commit, a conflict resolution that keeps one section and drops
another, a hand-revert of a single bundle — each of those disarms SOME bundles
and not others, and the file then claims two branches at once. Per-bundle
classification cannot see it, because each bundle it left alone still reads a
legitimate world in isolation. This test is the instrument for exactly that
state: every discriminator below must agree on ONE world, and any split fails
naming the bundles on each side.

The discriminators are IMPORTED from their consumer test modules, never
re-declared, so a re-pin there (with its ledger note) propagates here without a
second edit — a stale copy of a pin would make this gate disagree with the
per-bundle gates about what "armed" means, which is drift inside the drift
detector.

⚑ The arming commits are DIFFERENT per bundle (``67191f995`` armed the era
probe and the recency exponent; ``c62eb8ff2`` armed the untempered prior and
the sigma cap; the bt4heads promotion is a third lineage), so "all agree" is
not vacuous: nothing about the file's history forces these to move together
EXCEPT the fact that they were each armed on the same live branch — which is
precisely the property being checked.
"""
from __future__ import annotations

from collections.abc import Mapping

from tests.live_yaml_arming import (
    ABSENT,
    ARMED,
    PRODUCTION_CONFIG,
    classify_production_arming,
    production_bindings,
)


def _era_probe_world() -> tuple[str, list[str]]:
    from tests.test_era_forgetting_probe import ERA_PROBE_ARMED

    state = classify_production_arming(ERA_PROBE_ARMED, config=PRODUCTION_CONFIG)
    return state.world, state.problems


def _recency_exponent_world() -> tuple[str, list[str]]:
    from tests.test_replay_shard_recency_exponent import _ARMED_EXPONENT, _KEY

    state = classify_production_arming(
        {_KEY: _ARMED_EXPONENT}, config=PRODUCTION_CONFIG,
    )
    return state.world, state.problems


def _sigma_cap_world() -> tuple[str, list[str]]:
    """OFF admits the consumer's own spellings — absent, or a binding whose
    ``max(0, int(v))`` realizes 0 (an explicit ``0``, or ``0.5``).

    That is ``test_target_sigma_decoupling``'s recorded contract; refusing here
    what the knob's own consumer accepts would make this gate disagree with the
    per-bundle gate about what OFF means (codex round 2 on PR #488).
    """
    import yaml

    from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults
    from tests.test_target_sigma_decoupling import (
        _ARMED_TARGET_MAX_VISIT_CAP,
        _CAP_KEY,
    )

    bound = production_bindings(_CAP_KEY, config=PRODUCTION_CONFIG)
    if bound and len(bound) == 1:
        flat = flatten_run_config_defaults(
            yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
        )
        if int(flat.get(_CAP_KEY, 0) or 0) == 0:
            return ABSENT, []
    state = classify_production_arming(
        {_CAP_KEY: _ARMED_TARGET_MAX_VISIT_CAP}, config=PRODUCTION_CONFIG,
    )
    return state.world, state.problems


def _untempered_prior_world() -> tuple[str, list[str]]:
    """OFF admits two spellings — absent, or an explicit ``false``.

    That is the consumer's own convention (``test_untempered_target_prior``,
    citing c62eb8ff2's body), so it is honoured here rather than re-litigated:
    an explicit ``false`` is a DISARM, which is ``main``'s world as far as
    "which branch is this file from" is concerned.
    """
    from tests.test_untempered_target_prior import _KNOB

    bound = production_bindings(_KNOB, config=PRODUCTION_CONFIG)
    if bound in ([], [False]):
        return ABSENT, []
    state = classify_production_arming({_KNOB: True}, config=PRODUCTION_CONFIG)
    return state.world, state.problems


def _bt4heads_world() -> tuple[str, list[str]]:
    """The model-section discriminator, read the way test_lc0_control_arch does.

    Not through ``classify_production_arming``: the bt4heads keys live under
    ``model:`` and their two-world contract (whole-section equality with
    ``LIVE_ARCH_PIN`` when carried) is already enforced in
    ``tests/test_lc0_control_arch.py``. Here only the WORLD is read — all three
    keys present (live), none (main), anything else a named problem.
    """
    import yaml

    from chess_anti_engine.eval.lc0_control_arch import model_section
    from tests.test_lc0_control_arch import LIVE_ONLY_MODEL_KEYS

    in_tree = model_section(
        yaml.safe_load(PRODUCTION_CONFIG.read_text(encoding="utf-8"))
    )
    carried = sorted(key for key in LIVE_ONLY_MODEL_KEYS if key in in_tree)
    if not carried:
        return ABSENT, []
    if len(carried) == len(LIVE_ONLY_MODEL_KEYS):
        return ARMED, []
    missing = sorted(set(LIVE_ONLY_MODEL_KEYS) - set(carried))
    return "mixed", [
        f"model: section carries {carried} of the bt4heads bundle but not "
        f"{missing}"
    ]


def _promotion_gate_world() -> tuple[str, list[str]]:
    """The gate bundle, read off its switch key ``gate_mode``.

    The live yaml arms the whole ``gate_*`` bundle (``gate_mode: shadow`` and
    its operating-point keys); ``main``'s copy ships none and
    ``gate_config_from_dict`` realizes the dataclass defaults. A partial merge
    that strips the bundle while the other bundles stay armed silently drops
    ``gate_mode`` from ``shadow`` to the default — exactly the split this
    module exists to catch, and the bundle was missing from its first revision
    (codex round 2 on PR #488). Internal consistency of the bundle's VALUES is
    the consumer tests' job (``tests/test_promotion_gate.py`` pins
    ``gate_min_games_per_side`` against the dataclass default); here only the
    WORLD is read, off the one key that switches the machinery on.
    """
    bound = production_bindings("gate_mode", config=PRODUCTION_CONFIG)
    if not bound:
        return ABSENT, []
    if bound == ["shadow"]:
        return ARMED, []
    return "mixed", [
        f"gate_mode is bound as {bound!r} — neither absent (main's world) nor "
        "the single ledger'd `shadow`; a mode change is a restart-time "
        "decision with its own ledger note"
    ]


_BUNDLES = {
    "era_probe": _era_probe_world,
    "replay_shard_recency_exponent": _recency_exponent_world,
    "gumbel_target_max_visit_cap": _sigma_cap_world,
    "gumbel_target_untempered_prior": _untempered_prior_world,
    "bt4heads_model_keys": _bt4heads_world,
    "promotion_gate": _promotion_gate_world,
}


def _disagreements(worlds: Mapping[str, str]) -> list[str]:
    """The assertion's logic, factored out so the negative controls below can
    exercise the failing direction without editing the production yaml."""
    problems = [
        f"{name} is in a third state ({world!r})"
        for name, world in worlds.items()
        if world not in (ABSENT, ARMED)
    ]
    distinct = sorted(set(worlds.values()))
    if len(distinct) > 1:
        by_world = {
            world: sorted(name for name, w in worlds.items() if w == world)
            for world in distinct
        }
        problems.append(
            "the bundles disagree on which branch this yaml is from: "
            + "; ".join(f"{world}={names}" for world, names in by_world.items())
        )
    return problems


def test_every_armed_bundle_reads_the_same_world() -> None:
    worlds: dict[str, str] = {}
    bundle_problems: list[str] = []
    for name, read in _BUNDLES.items():
        world, problems = read()
        worlds[name] = world
        bundle_problems.extend(f"{name}: {p}" for p in problems)

    assert not bundle_problems, (
        "a bundle is individually broken — fix it in its own consumer test "
        "first, this gate only compares worlds:\n  "
        + "\n  ".join(bundle_problems)
    )
    split = _disagreements(worlds)
    assert not split, (
        "configs/pbt2_small.yaml claims TWO branches at once:\n  "
        + "\n  ".join(split)
        + "\nA partial merge or cherry-pick disarmed some bundles and not "
        "others. The fix is in the yaml (finish or revert the merge), never "
        "in a pin — and if this is deliberate, it is a restart-time decision "
        "that needs its own ledger note and a re-pin in the CONSUMER module."
    )


def test_the_gate_fails_on_a_split_and_names_both_sides() -> None:
    """Negative control: the checker itself, on the states it must refuse.

    A same-world gate that cannot fail is a constant; per this repo's rule a
    new test is vacuous until its failing direction is watched. The production
    yaml cannot be edited to produce a split (it is the LIVE file), so the
    factored ``_disagreements`` is driven directly.
    """
    agree_armed: dict[str, str] = dict.fromkeys(_BUNDLES, ARMED)
    agree_absent: dict[str, str] = dict.fromkeys(_BUNDLES, ABSENT)
    assert _disagreements(agree_armed) == []
    assert _disagreements(agree_absent) == []

    split = dict(agree_armed, era_probe=ABSENT)
    problems = _disagreements(split)
    assert len(problems) == 1
    assert "era_probe" in problems[0], problems
    assert ARMED in problems[0], problems

    third = dict(agree_armed, bt4heads_model_keys="mixed")
    problems = _disagreements(third)
    assert any("third state" in p and "bt4heads" in p for p in problems), problems
