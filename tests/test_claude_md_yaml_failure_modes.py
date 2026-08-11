"""CLAUDE.md's three yaml-failure modes must still be the code's behaviour.

The rule they document is operational: which of the three categories a live
yaml key falls into decides whether a bad edit is rejected (trial lives), fatal
at launch (process never boots), or fatal mid-iteration (trial dies). Every one
of the three is a *structural* property of a specific call site --- a bare
``except``-less ``try``, a call placed outside any ``try``, a validator invoked
inside ``from_dict`` --- and structure is exactly what a refactor moves without
anyone re-reading the doc. CLAUDE.md has already been through three rounds of
stale-claim removal; this is the pin that stops a fourth.

These tests read the source with ``ast``, not the running process, on purpose:
the mid-iteration mode can only be *observed* by killing a trial.
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
    """Category (a), AT LAUNCH: the same ValueError is fatal.

    ``main()`` calls ``flatten_run_config_defaults`` outside any ``try``, so an
    unknown key raises before the trial exists --- there is no old config to
    fall back to, and the process never starts. Wrapping this call in a
    ``try`` would silently downgrade "fails to boot" to "boots on stale
    defaults", which is the failure CLAUDE.md tells operators to expect.
    """
    fn = _function("chess_anti_engine/run.py", "main")
    lines = _call_lines(fn, "flatten_run_config_defaults")
    assert lines, "run.py:main no longer flattens the yaml"
    for line in lines:
        assert not _enclosing_tries(fn, line), (
            "flatten_run_config_defaults is inside a try in run.py:main; "
            "CLAUDE.md documents an unknown key at launch as fatal"
        )


def test_trial_iteration_loop_has_no_except() -> None:
    """Category (b): a validator ValueError tears the trial down mid-iteration.

    ``TrialConfig.from_dict`` is called per iteration inside a ``try`` that has
    a ``finally`` and no handler at all, so any validator raise runs
    ``_cleanup_trial_resources`` and ends the trial. Adding an ``except`` here
    would be a behaviour change big enough to rewrite the doc for.
    """
    fn = _function("chess_anti_engine/tune/trainable.py", "train_trial")
    lines = _call_lines(fn, "from_dict")
    assert lines, "train_trial no longer rebuilds TrialConfig per iteration"
    enclosing = [t for line in lines for t in _enclosing_tries(fn, line)]
    assert enclosing, "the from_dict call is no longer inside the iteration-loop try"
    assert all(t.handlers == [] for t in enclosing), (
        "the iteration loop grew an except; a bad TrialConfig value no longer "
        "kills the trial and CLAUDE.md's category (b) is stale"
    )
    assert any(t.finalbody for t in enclosing), "the cleanup finally is gone"


def test_validated_field_band_is_the_documented_one() -> None:
    """Category (b)'s measured example must still be the band CLAUDE.md quotes."""
    from chess_anti_engine.tune.trial_config import POLICY_TEMP_MAX, POLICY_TEMP_MIN

    text = (_REPO / "CLAUDE.md").read_text(encoding="utf-8")
    assert f"`[{POLICY_TEMP_MIN}, {POLICY_TEMP_MAX}]`" in text, (
        f"CLAUDE.md must quote the live gumbel_policy_temp band "
        f"[{POLICY_TEMP_MIN}, {POLICY_TEMP_MAX}]"
    )


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
    """Unit-test the scanner itself against each access form in ISOLATION.

    Asserting ``"lr" in _from_dict_read_keys()`` against the real file does NOT
    pin subscript detection: ``from_dict`` also writes ``"lr" in config``, so the
    membership branch alone satisfies it and deleting the subscript branch leaves
    the suite green. Measured --- that mutation was run and passed. Each form
    therefore gets its own one-line module, where nothing else can cover for a
    branch that has been removed. A ``.get``-only scan is what let the first
    version of this file certify a false claim in CLAUDE.md.
    """
    assert _read_keys(ast.parse(source)) == {"k"}, f"the scan no longer sees {form} reads"


@pytest.mark.parametrize("key", ["w_wdl", "zclip_max_norm"])
def test_schema_keys_that_from_dict_never_reads(key: str) -> None:
    """Category (c), first sub-shape: in the schema, never read by ``from_dict``.

    The named keys are the examples CLAUDE.md gives for silent wrongness. If one
    of them ever starts being read *and validated* it moves to category (b) --- a
    bad value would start killing the trial instead of being applied quietly ---
    and the doc's example list has to move with it.
    """
    assert key in _yaml_schema_keys(), f"{key} is no longer a yaml schema key"
    assert key not in _from_dict_read_keys(), (
        f"{key} is now read by TrialConfig.from_dict, so it is no longer the "
        "'never read at all' category (c) example CLAUDE.md uses"
    )


def test_lr_is_read_by_from_dict_and_unvalidated() -> None:
    """Category (c), second sub-shape: READ, applied, and range-checked by nothing.

    CLAUDE.md's warning is that "not validated" must not be read as "inert".
    ``lr`` is a ``TrialConfig`` field, ``from_dict`` reads it, and the only
    coercion is ``float()`` --- so a live ``lr: 0.3`` is accepted and handed to
    the running trainer. If ``lr`` ever gains a band it becomes category (b) and
    the doc must say so.
    """
    import dataclasses

    from chess_anti_engine.tune.trial_config import TrialConfig

    assert "lr" in _yaml_schema_keys()
    assert "lr" in {f.name for f in dataclasses.fields(TrialConfig)}
    assert "lr" in _from_dict_read_keys()

    cfg = TrialConfig.from_dict({"lr": 0.3})
    assert cfg.lr == 0.3, (
        "lr now goes through a validator or clamp; CLAUDE.md lists it as the "
        "category (c) key that is read but NOT range-checked"
    )


def test_sf_pid_keys_split_between_read_and_unread() -> None:
    """CLAUDE.md names the exact 4 ``sf_pid_*`` keys ``from_dict`` reads.

    The superseded text said "every ``sf_pid_*``" was unread, which was false for
    four of the forty. The doc now enumerates them, so the enumeration is pinned.
    """
    read = _from_dict_read_keys()
    pid = {k for k in _yaml_schema_keys() if k.startswith("sf_pid_")}
    assert pid, "sf_pid_* keys have left the yaml schema"
    documented = {
        "sf_pid_enabled",
        "sf_pid_ema_alpha",
        "sf_pid_target_winrate",
        "sf_pid_wdl_regret_max",
    }
    assert pid & read == documented, (
        f"CLAUDE.md enumerates {sorted(documented)} as the sf_pid_* keys "
        f"TrialConfig.from_dict reads; the code now reads {sorted(pid & read)}"
    )
    assert len(pid - read) > len(documented), "the 'most sf_pid_* are unread' claim is stale"


def test_most_of_the_train_section_is_unread_by_from_dict() -> None:
    """CLAUDE.md: "most of the live yaml's ``train:`` section" is category (c)."""
    import yaml

    cfg = yaml.safe_load((_REPO / "configs" / "pbt2_small.yaml").read_text(encoding="utf-8"))
    train_keys = set(cfg["train"])
    unread = train_keys - _from_dict_read_keys()
    assert len(unread) * 2 > len(train_keys), (
        f"only {len(unread)}/{len(train_keys)} train: keys are unread by from_dict; "
        "CLAUDE.md's 'most of train:' is no longer true"
    )
