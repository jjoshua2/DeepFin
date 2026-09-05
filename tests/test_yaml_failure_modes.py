"""YAML failure modes across launch, live reload and trial validation.

These structural checks protect the operational distinctions described in
``docs/operations.md`` without requiring literal values in project guidance.
They inspect exception boundaries without starting or terminating a live trial.
Policy-temperature runtime boundaries are covered by
``test_selfplay_policy_temp_plumbing.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]


def _module(rel: str) -> ast.Module:
    return ast.parse((_REPO / rel).read_text(encoding="utf-8"))


def _function(rel: str, name: str) -> ast.FunctionDef:
    for node in ast.walk(_module(rel)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    pytest.fail(f"{rel} no longer defines {name}()")


def _enclosing_tries(fn: ast.FunctionDef, line: int) -> list[ast.Try]:
    return [
        node for node in ast.walk(fn)
        if isinstance(node, ast.Try) and node.lineno <= line <= (node.end_lineno or -1)
    ]


def _call_lines(fn: ast.FunctionDef, name: str) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Name) and func.id == name) or (
            isinstance(func, ast.Attribute) and func.attr == name
        ):
            lines.append(node.lineno)
    return lines


def test_mid_run_reload_swallows_the_unknown_key_error() -> None:
    """Category (a), MID-RUN: the reload is rejected and the trial survives.

    The whole body of ``_reload_yaml_into_config`` sits under one ``try`` with a
    catch-all handler, which is what turns an unknown key into a warning plus
    "keep the old config" instead of an exception the caller has to survive.
    """
    fn = _function("chess_anti_engine/tune/trainable_config_ops.py", "_reload_yaml_into_config")
    tries = [node for node in fn.body if isinstance(node, ast.Try)]
    assert len(tries) == 1, "the overlay is meant to be wrapped in exactly one try"
    handlers = tries[0].handlers
    assert handlers, "no except => an unknown key would propagate and kill the trial"
    assert any(
        h.type is None or (isinstance(h.type, ast.Name) and h.type.id in {"Exception", "BaseException"})
        for h in handlers
    ), "the handler must be broad enough to catch the loader's ValueError"


def test_launch_path_flattens_outside_any_try() -> None:
    """An unknown launch key raises before a trial or fallback config exists."""
    fn = _function("chess_anti_engine/run.py", "main")
    lines = _call_lines(fn, "flatten_run_config_defaults")
    assert lines, "run.py:main no longer flattens the yaml"
    for line in lines:
        assert not _enclosing_tries(fn, line), (
            "flatten_run_config_defaults is inside a try in run.py:main; "
            "an unknown launch key must propagate instead of silently using defaults"
        )


def test_trial_iteration_loop_has_no_except() -> None:
    """Per-iteration validation errors run cleanup and propagate out of the trial."""
    fn = _function("chess_anti_engine/tune/trainable.py", "train_trial")
    lines = _call_lines(fn, "from_dict")
    assert lines, "train_trial no longer rebuilds TrialConfig per iteration"
    enclosing = [t for line in lines for t in _enclosing_tries(fn, line)]
    assert enclosing, "the from_dict call is no longer inside the iteration-loop try"
    assert all(t.handlers == [] for t in enclosing), (
        "the iteration loop grew an except; a bad TrialConfig value no longer "
        "propagates out of the trial; reassess its failure and cleanup behavior"
    )
    assert any(t.finalbody for t in enclosing), "the cleanup finally is gone"


def _yaml_schema_keys() -> set[str]:
    import chess_anti_engine.utils.config_yaml as config_yaml

    schema: set[str] = set()
    for name in dir(config_yaml):
        value = getattr(config_yaml, name)
        if name.isupper() and isinstance(value, tuple):
            schema |= {v for v in value if isinstance(v, str)}
    assert schema, "no uppercase tuple in config_yaml looks like the key schema any more"
    return schema


def _read_keys(node: ast.AST) -> set[str]:
    """Every literal config key read under *node*, by ANY access form.

    ⚑ A ``.get("<key>")``-only scan is the trap this helper exists to close: the
    constructor reads ``lr`` as ``config["lr"]``, a SUBSCRIPT, so a ``.get``-only
    scan reports the single most consequential key in the yaml as unread and any
    "this key is never read" assertion built on it passes vacuously. Subscripts
    and ``"<key>" in config`` membership tests count as reads too.
    """
    read: set[str] = set()
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call) and child.args
            and isinstance(child.args[0], ast.Constant)
            and isinstance(child.args[0].value, str)
            and (
                (isinstance(child.func, ast.Attribute) and child.func.attr == "get")
                or (isinstance(child.func, ast.Name) and child.func.id == "_get")
            )
        ):
            read.add(child.args[0].value)
        elif (
            isinstance(child, ast.Subscript)
            and isinstance(child.value, ast.Name) and child.value.id == "config"
            and isinstance(child.slice, ast.Constant) and isinstance(child.slice.value, str)
        ):
            read.add(child.slice.value)
        elif (
            isinstance(child, ast.Compare)
            and isinstance(child.left, ast.Constant) and isinstance(child.left.value, str)
            and any(isinstance(op, ast.In) for op in child.ops)
        ):
            read.add(child.left.value)
    return read


def _from_dict_read_keys() -> set[str]:
    return _read_keys(_function("chess_anti_engine/tune/trial_config.py", "from_dict"))


@pytest.mark.parametrize(
    ("form", "source"),
    [
        ("dict.get", 'x = config.get("k", 1)'),
        ("_get helper", 'x = _get("k", 1)'),
        ("subscript", 'x = config["k"]'),
        ("membership", 'x = 1 if "k" in config else 2'),
    ],
)
def test_read_key_scan_sees_every_access_form(form: str, source: str) -> None:
    """Each access form must independently register a config read.

    The real lr constructor uses both subscripting and membership, so checking
    that field alone would not expose a scanner that missed either form.
    """
    assert _read_keys(ast.parse(source)) == {"k"}, f"the scan no longer sees {form} reads"


@pytest.mark.parametrize("key", ["w_wdl", "zclip_max_norm"])
def test_schema_keys_that_from_dict_never_reads(key: str) -> None:
    """Some schema-known settings bypass TrialConfig and reach other consumers."""
    assert key in _yaml_schema_keys(), f"{key} is no longer a yaml schema key"
    assert key not in _from_dict_read_keys(), (
        f"{key} is now read by TrialConfig.from_dict, so it is no longer the "
        "schema-only case; reassess its validation and consumer path"
    )


def test_lr_is_read_by_from_dict_and_unvalidated() -> None:
    """Reading a TrialConfig field does not imply range validation.

    lr is coerced to float but an unusually large numeric value is passed on.
    """
    import dataclasses

    from chess_anti_engine.tune.trial_config import TrialConfig

    assert "lr" in _yaml_schema_keys()
    assert "lr" in {f.name for f in dataclasses.fields(TrialConfig)}
    assert "lr" in _from_dict_read_keys()

    cfg = TrialConfig.from_dict({"lr": 0.3})
    assert cfg.lr == 0.3, (
        "lr now goes through a validator or clamp; reassess live reload behavior"
    )


def test_sf_pid_keys_split_between_read_and_unread() -> None:
    """PID configuration is divided between TrialConfig and other consumers."""
    read = _from_dict_read_keys()
    pid = {k for k in _yaml_schema_keys() if k.startswith("sf_pid_")}
    assert pid, "sf_pid_* keys have left the yaml schema"
    expected = {
        "sf_pid_enabled",
        "sf_pid_ema_alpha",
        "sf_pid_target_winrate",
        "sf_pid_wdl_regret_max",
    }
    assert pid & read == expected, (
        f"expected {sorted(expected)} as the sf_pid_* keys "
        f"TrialConfig.from_dict reads; the code now reads {sorted(pid & read)}"
    )
    assert len(pid - read) > len(expected), "the 'most sf_pid_* are unread' claim is stale"


def test_most_of_the_train_section_is_unread_by_from_dict() -> None:
    """Most train settings bypass TrialConfig; its validation is incomplete."""
    import yaml

    cfg = yaml.safe_load((_REPO / "configs" / "pbt2_small.yaml").read_text(encoding="utf-8"))
    train_keys = set(cfg["train"])
    unread = train_keys - _from_dict_read_keys()
    assert len(unread) * 2 > len(train_keys), (
        f"only {len(unread)}/{len(train_keys)} train: keys are unread by from_dict; "
        "reassess which train settings bypass TrialConfig validation"
    )
