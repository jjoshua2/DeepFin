"""Derive the startup-only config keys from the source, instead of listing them.

Audit T2. ``trainable_config_ops`` classifies a yaml key as construction-only
when its only consumer runs before the iteration loop, because the alternative
is worse than being ignored: the value lands in ``config``, echoes back correct
in the result row, and ``scripts/audit_realized_config.py`` reports it under
"the running value is correct" while the object using it kept the launch value.

That classification was hand-maintained, and it had been finished for the
replay buffer and the era probes and never swept over the rest of the startup
block. A hand list drifts the moment somebody adds a key; this module derives
the class from the AST so the drift fails a test instead.

THE DERIVATION, and what it can and cannot see:

  * ``train_trial`` is split into its STARTUP region (everything outside the
    ``while``) and its LOOP region (the ``while`` subtree).
  * From each region, calls are followed transitively through every function
    defined anywhere in ``chess_anti_engine``, by NAME -- both ``f(...)`` and
    ``obj.method(...)``, because the loop reaches its live consumers through
    both (``pid.refresh_live_params(config)`` is a method call, and missing it
    would report the whole live PID surface as startup-only).
  * A key read from ``config``/``cfg`` by a string literal, or off ``tc``/
    ``trial_config`` by attribute, counts as a read of that key.
  * ``TrialConfig.from_dict`` is EXCLUDED from the closure. It reads every key
    there is; parsing a key into the dataclass is not consuming it, and the
    loop rebuilds ``tc`` every iteration, so including it would make every key
    look live.
  * Startup-only = read in the startup scope, not read anywhere in the loop
    scope.

Two blind spots, both handled by the annotated residue below rather than
pretended away: a key read through a CONSTANT (``for k in TRAINER_WEIGHT_KEYS:
config[k]``) is invisible to a literal-key screen and shows up as a false
startup-only; and name-based call resolution merges same-named functions from
different modules, which can only ever ADD reachability (a false LIVE), never
remove it.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from chess_anti_engine.tune.trainable_config_ops import (
    _STARTUP_ONLY_TRIAL_KEYS,
    construction_only_config_keys,
    dead_config_keys,
    restart_required_config_keys,
)
from chess_anti_engine.utils.config_yaml import _FLAT_ALLOWLIST

_PKG_ROOT = Path(__file__).resolve().parents[1] / "chess_anti_engine"

# Reads inside this function are PARSING, not consuming: see the module
# docstring.
_NOT_A_CONSUMER = frozenset({"from_dict"})

_CONFIG_NAMES = frozenset({"config", "cfg"})
_TRIAL_CONFIG_NAMES = frozenset({"tc", "trial_config"})


def _module_asts() -> list[ast.Module]:
    return [
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in sorted(_PKG_ROOT.rglob("*.py"))
    ]


def _function_index(modules: list[ast.Module]) -> dict[str, list[ast.AST]]:
    index: dict[str, list[ast.AST]] = defaultdict(list)
    for module in modules:
        for node in ast.walk(module):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                index[node.name].append(node)
    return index


def _called_names(nodes: Sequence[ast.AST]) -> set[str]:
    out: set[str] = set()
    for node in nodes:
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            if isinstance(sub.func, ast.Name):
                out.add(sub.func.id)
            elif isinstance(sub.func, ast.Attribute):
                out.add(sub.func.attr)
    return out


def _reachable(seed: Sequence[ast.AST], index: dict[str, list[ast.AST]]) -> set[str]:
    seen: set[str] = set()
    frontier = _called_names(seed) - _NOT_A_CONSUMER
    while frontier:
        nxt: set[str] = set()
        for name in frontier:
            if name in seen:
                continue
            seen.add(name)
            nxt |= _called_names(index.get(name, []))
        frontier = (nxt - seen) - _NOT_A_CONSUMER
    return seen


def _config_key_reads(nodes: Sequence[ast.AST]) -> set[str]:
    out: set[str] = set()
    for node in nodes:
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Attribute)
                and isinstance(sub.value, ast.Name)
                and sub.value.id in _TRIAL_CONFIG_NAMES
            ):
                out.add(sub.attr)
            elif (
                isinstance(sub, ast.Subscript)
                and isinstance(sub.value, ast.Name)
                and sub.value.id in _CONFIG_NAMES
                and isinstance(sub.slice, ast.Constant)
                and isinstance(sub.slice.value, str)
            ):
                out.add(sub.slice.value)
            elif (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Attribute)
                and sub.func.attr == "get"
                and isinstance(sub.func.value, ast.Name)
                and sub.func.value.id in _CONFIG_NAMES
                and sub.args
                and isinstance(sub.args[0], ast.Constant)
                and isinstance(sub.args[0].value, str)
            ):
                out.add(sub.args[0].value)
    return out


def _region_keys(region: Sequence[ast.AST], index: dict[str, list[ast.AST]]) -> set[str]:
    nodes: list[ast.AST] = list(region)
    for name in _reachable(region, index):
        nodes.extend(index.get(name, []))
    return _config_key_reads(nodes)


def derive_startup_only_keys() -> frozenset[str]:
    """Yaml keys ``train_trial`` reads only before its iteration loop."""
    modules = _module_asts()
    index = _function_index(modules)
    train_trial = next(
        node for module in modules for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "train_trial"
    )
    loop = next(n for n in ast.walk(train_trial) if isinstance(n, ast.While))
    loop_nodes = {id(n) for n in ast.walk(loop)}
    startup = [
        n for n in ast.walk(train_trial)
        if isinstance(n, ast.stmt) and id(n) not in loop_nodes
    ]
    startup_keys = _region_keys(startup, index)
    loop_keys = _region_keys([loop], index)
    return frozenset((startup_keys - loop_keys) & _FLAT_ALLOWLIST)


# Derived startup-only keys this PR does NOT classify, each group with the
# reason. The test below pins the residue EXACTLY, so a key that becomes
# startup-only tomorrow -- or one of these that gets a live consumer -- fails
# here until somebody says which it is. That is the whole point: the previous
# arrangement had no way to notice.
_UNCLASSIFIED_STARTUP_ONLY: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "PID lever gains: genuinely construction-only inside `pid_from_config` "
        "-- `refresh_live_params` re-reads only ~10 of the family. Freezing "
        "~35 knobs of the LIVE difficulty controller changes the control "
        "surface of a running experiment, so it needs its own ledger entry, "
        "not an observability PR.",
        frozenset({
            "sf_pid_nodes_crash_ease_min_frac", "sf_pid_nodes_crash_ease_z",
            "sf_pid_nodes_deadband_sigma", "sf_pid_nodes_degen_step_frac",
            "sf_pid_nodes_emergency_ease_mult",
            "sf_pid_nodes_emergency_ease_step", "sf_pid_nodes_max_step",
            "sf_pid_nodes_max_step_frac", "sf_pid_nodes_recency_half_life",
            "sf_pid_nodes_safety_floor", "sf_pid_nodes_tighten_gain",
            "sf_pid_nodes_tighten_streak_after", "sf_pid_nodes_tighten_streak_gain",
            "sf_pid_nodes_window",
            "sf_pid_regret_crash_ease_min_frac", "sf_pid_regret_crash_ease_z",
            "sf_pid_regret_deadband_sigma", "sf_pid_regret_degen_step_frac",
            "sf_pid_regret_ease_step_frac", "sf_pid_regret_emergency_ease_mult",
            "sf_pid_regret_emergency_ease_step", "sf_pid_regret_max_step",
            "sf_pid_regret_max_step_frac", "sf_pid_regret_recency_half_life",
            "sf_pid_regret_safety_floor", "sf_pid_regret_tighten_gain",
            "sf_pid_regret_tighten_streak_after", "sf_pid_regret_tighten_streak_gain",
            "sf_pid_regret_window", "sf_pid_wdl_regret_start",
        }),
    ),
    (
        "Trainer construction arguments (`trainer_kwargs_from_config`). Same "
        "class as the replay-buffer keys and a candidate for the same "
        "treatment, but the Trainer's construction surface deserves one sweep "
        "of its own rather than a partial one taken from this file's call "
        "graph.",
        frozenset({
            "gradient_checkpointing", "rebuild_categorical_target",
            "sf_policy_sparse_ce", "warmup_lr_start",
            "fdp_king_safety", "fdp_mobility", "fdp_outposts", "fdp_pawns",
            "fdp_pins",
        }),
    ),
    (
        "FALSE POSITIVE of the screen: pushed onto the trainer every iteration "
        "by `_apply_lr_gamma_weights`, which iterates the constant "
        "`TRAINER_WEIGHT_KEYS` instead of naming the key -- a literal-key AST "
        "read cannot see it. Freezing it would break a live knob.",
        frozenset({"w_sf_volatility"}),
    ),
    (
        "Startup RESTORE decisions: they choose what this process starts FROM, "
        "so 'live' has no meaning for them and a warning saying 'restart "
        "required' would be noise. `salvage_seed_pool_dir` is the dangerous "
        "one and its danger (audit T5) is stickiness in the SAVED trial "
        "config, which freezing the reload does not touch.",
        frozenset({
            "bootstrap_checkpoint", "bootstrap_reinit_volatility_heads",
            "bootstrap_zero_policy_heads", "salvage_reinit_volatility_heads",
            "salvage_restore_donor_config", "salvage_restore_full_trainer_state",
            "salvage_restore_pid_state", "salvage_seed_pool_dir",
            "search_optimizer", "shared_shards_dir", "warm_start_lr_max_ratio",
        }),
    ),
    (
        "The cross-trial exploit family, structurally inert at "
        "`num_samples: 1` (audit S1 -- all five shared_* columns 0.0 on "
        "478/478 rows). Classifying dead-at-this-topology code would say more "
        "about the topology than about the key.",
        frozenset({
            "exploit_replay_donor_shards",
            "exploit_replay_local_keep_older_fraction",
            "exploit_replay_local_keep_recent_fraction",
            "exploit_replay_refresh_enabled",
        }),
    ),
)

# `eval_games` is classified by hand and NOT derived, on purpose: the loop does
# read it (`tc.eval_games <= 0 or eval_sf is None`), so the screen sees a live
# consumer. That read can only ever DISABLE, because `eval_sf` is built once at
# startup and is None forever when the launch value was 0 -- enabling eval live
# is the silent no-op, which is exactly the class being frozen.
_CLASSIFIED_BUT_NOT_DERIVED = frozenset({"eval_games"})


def test_the_derivation_finds_the_keys_the_audit_found_by_hand() -> None:
    """Sanity + negative control on the instrument itself. A derivation that
    returned everything, or nothing, would pass the classification test below
    vacuously."""
    derived = derive_startup_only_keys()
    assert "iterations" in derived
    assert "sf_pid_enabled" in derived
    assert "puzzle_epd" in derived
    assert "pause_file" in derived
    assert "eval_sf_nodes" in derived
    # Keys with a real live consumer must NOT be derived: `log_level` reaches
    # the worker launch signature, `lr` and the PID's live half are re-read
    # every iteration.
    assert "log_level" not in derived
    assert "lr" not in derived
    assert "sf_pid_ema_alpha" not in derived
    assert 20 < len(derived) < 200, len(derived)


def test_every_derived_startup_only_key_is_classified_or_named() -> None:
    """THE GUARD. A key whose only consumer runs at startup must be in the
    frozen classes -- or in the annotated residue above, with the reason it is
    not. Adding a startup-only key to the schema and forgetting to classify it
    fails here."""
    derived = derive_startup_only_keys()
    classified = restart_required_config_keys() | dead_config_keys()
    named = frozenset().union(*(keys for _reason, keys in _UNCLASSIFIED_STARTUP_ONLY))
    unaccounted = derived - classified - named
    assert not unaccounted, (
        "these keys are read only before train_trial's iteration loop and are "
        "in no reload class, so a live edit to one lands in config, reads back "
        "correct in the result row, and does nothing: "
        f"{sorted(unaccounted)}"
    )
    # And the residue must not rot: an entry that has since been classified, or
    # that no longer exists, has to be removed rather than left as a claim
    # nobody re-checked.
    stale = (named & classified) | (named - derived)
    assert not stale, f"stale entries in _UNCLASSIFIED_STARTUP_ONLY: {sorted(stale)}"


def test_the_hand_maintained_startup_only_set_is_derivable() -> None:
    """The other direction: every key frozen by hand must be one the source
    actually shows is startup-only, so the set cannot grow by assertion."""
    derived = derive_startup_only_keys()
    invented = _STARTUP_ONLY_TRIAL_KEYS - derived - _CLASSIFIED_BUT_NOT_DERIVED
    assert not invented, (
        "declared startup-only but the source shows a live consumer: "
        f"{sorted(invented)}"
    )
    assert _STARTUP_ONLY_TRIAL_KEYS.issubset(restart_required_config_keys())
    assert _STARTUP_ONLY_TRIAL_KEYS.issubset(construction_only_config_keys())


def test_a_startup_only_key_can_no_longer_be_called_a_running_value() -> None:
    """ASSERT THE AXIS STATE, not an exit code. The provenance audit routes a
    key in none of the classes to its live-reloadable branch, where a
    yaml-vs-row difference reads as a rejected reload and a params.json
    difference reads as "the running value is correct". `sf_pid_enabled` is a
    boolean an operator would flip live during a diagnosis, and flipping it
    does nothing at all."""
    from scripts.audit_realized_config import classify_config_provenance

    report, findings = classify_config_provenance(
        params={"sf_pid_enabled": False},
        flat_yaml={"sf_pid_enabled": True},
        realized={"sf_pid_enabled": False},
        restart_keys=restart_required_config_keys(),
        construction_only_keys=construction_only_config_keys(),
        dead_keys={},
    )
    joined = "\n".join(report)
    assert "PENDING-RESTART sf_pid_enabled" in joined, joined
    assert any("restart required" in f for f in findings), findings
    assert "RELOAD-NOT-APPLIED" not in joined, joined
