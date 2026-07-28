"""Every worker-affecting yaml key must actually REACH a distributed worker.

The defect this file exists to prevent (`docs/rl_loop_audit.md` E13/A7):
``soft_policy_temp`` sat in the production yaml as 3.0 from 2026-03-02, but the
key was named in neither ``build_recommended_worker`` nor
``WorkerSession._build_selfplay_configs``. So it was never published, never
consumed, and ``GameConfig`` kept its dataclass default of 2.0 — measured on
live shards as a fitted target exponent of 0.50000 against a configured 1/3.
``soft_policy_ce`` is ~41% of weighted trunk gradient, so five months of
training ran the second-largest loss term at the wrong sharpness.

The instrument missed it for a structural reason worth stating: the A4 check
diffed the published reco against the yaml over the INTERSECTION of the two key
sets, and a key absent from the reco never enters an intersection. The tests
below therefore come in two kinds — one that follows a single key end to end,
and one that diffs over the UNION so that "absent" is a state a test can fail on.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import yaml

from chess_anti_engine.model import ModelConfig
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.tune.distributed_runtime import build_recommended_worker
from chess_anti_engine.utils.config_yaml import SELFPLAY_CONFIG_KEYS
from chess_anti_engine.worker import WorkerSession
from scripts.audit_realized_config import (
    _RECO_WORKER_DEFAULT,
    diff_reco_coverage,
    flatten_yaml_config,
)

_PRODUCTION_YAML = Path(__file__).resolve().parents[1] / "configs" / "pbt2_small.yaml"

# EMPTY on purpose, and the name is kept rather than deleted.
#
# diff_focus used to be the one entry here: the distributed worker never builds
# a ``DiffFocusConfig`` from the reco, so it runs the dataclass defaults, and the
# yaml used to ask for the Run-4 sweep winners instead. That made it a live
# DIVERGENCE rather than a cosmetic gap, which is why it was tracked separately
# from ``_RECO_WORKER_DEFAULT``.
#
# The 2026-07-26 ledger entry pinned the yaml to the realized values as config
# honesty, which closed the divergence — so on 2026-07-28 the group moved into
# ``_RECO_WORKER_DEFAULT``, where the allowlist is SELF-INVALIDATING: if anyone
# edits a diff_focus value off the default it stops being covered and becomes a
# finding again, instead of silently doing nothing. That is a strictly stronger
# guard than this set provided.
#
# The name stays because ``test_production_config_publishes_every_worker_affecting_key``
# compares against it, so a NEWLY unpublished key still fails loudly instead of
# quietly widening an allowlist that no longer exists.
_KNOWN_UNPUBLISHED: frozenset[str] = frozenset()


def _bare_session() -> WorkerSession:
    """Minimal ``WorkerSession``: ``_build_selfplay_configs`` reads only these."""
    session = object.__new__(WorkerSession)
    session.log = logging.getLogger("test.reco_coverage")
    session.args = SimpleNamespace()
    session.opening_book_path = None
    session.opening_book_path_2 = None
    session.opening_fen_list_path = None
    session._dole_lock = threading.Lock()
    return session


def _as_int(raw: object, default: int) -> int:
    """Narrow a ``dict[str, object]`` config value for the publisher's int args."""
    return int(raw) if isinstance(raw, (int, float, str)) else default


def _reco_from(config: dict[str, object]) -> dict[str, object]:
    """The reco the server would publish for ``config``.

    ``sf_nodes`` and ``mcts_simulations`` are passed in by the publisher rather
    than read from the config (PID budget / simulation ramp), so mirror the
    config's own values here — otherwise the diff reports the test's arguments
    as a divergence.
    """
    return build_recommended_worker(
        config=config,
        model_cfg=ModelConfig(),
        sf_nodes=_as_int(config.get("sf_nodes"), 5000),
        mcts_simulations=_as_int(config.get("mcts_simulations"), 32),
    )


# --------------------------------------------------------------------------
# soft_policy_temp, end to end
# --------------------------------------------------------------------------


def test_soft_policy_temp_is_published_to_workers() -> None:
    """Server side: the yaml value must appear in recommended_worker."""
    assert _reco_from({"soft_policy_temp": 3.5})["soft_policy_temp"] == 3.5


def test_soft_policy_temp_reaches_game_config_from_the_reco() -> None:
    """Worker side: the published value must land on GameConfig, not the default.

    This is the regression test for E13. If either half of the plumbing is
    removed, the assert reads back GameConfig's 2.0 and this fails.
    """
    reco = _reco_from({"soft_policy_temp": 3.5})
    cfgs, _sf_args = WorkerSession._build_selfplay_configs(_bare_session(), reco)

    assert cfgs["game"].soft_policy_temp == 3.5
    assert cfgs["game"].soft_policy_temp != GameConfig().soft_policy_temp


def test_soft_policy_temp_falls_back_to_the_dataclass_default() -> None:
    """An old server's manifest has no such key; the worker must not invent one."""
    cfgs, _sf_args = WorkerSession._build_selfplay_configs(
        _bare_session(), {"sf_nodes": 5000},
    )

    assert cfgs["game"].soft_policy_temp == GameConfig().soft_policy_temp


def test_soft_policy_temp_change_restarts_the_selfplay_session() -> None:
    """It is baked into a frozen GameConfig at session start, like sf_policy_temp.

    Unwatched, a mid-run edit would keep producing targets at the old exponent
    until some unrelated key happened to force a restart.
    """
    assert "soft_policy_temp" in WorkerSession._RECO_RESTART_KEYS
    assert "soft_policy_temp" not in WorkerSession._RECO_LIVE_KEYS


def test_production_soft_policy_temp_is_the_value_the_net_trained_on() -> None:
    """Behaviour pin, not a preference.

    The plumbing above makes this yaml key real for the first time. It reads 2.0
    because 2.0 is what every worker has produced since 2026-03-02; publishing a
    3.0 would have silently re-targeted the second-largest loss term on the next
    restart. Moving it is a data-affecting experiment and needs a ledger entry
    with a pre-committed yardstick — update this test in that same change.
    """
    raw = yaml.safe_load(_PRODUCTION_YAML.read_text(encoding="utf-8"))
    flat = flatten_yaml_config(raw)

    assert flat["soft_policy_temp"] == 2.0
    assert flat["soft_policy_temp"] == GameConfig().soft_policy_temp


# --------------------------------------------------------------------------
# The instrument: union, not intersection
# --------------------------------------------------------------------------


def test_union_diff_reports_a_key_absent_from_the_reco() -> None:
    """The E13 shape: configured in the yaml, missing from the published reco."""
    reco = {"sf_policy_temp": 0.012}
    flat = {"sf_policy_temp": 0.012, "soft_policy_temp": 3.0}

    _report, findings = diff_reco_coverage(
        reco, flat, selfplay_keys=("sf_policy_temp", "soft_policy_temp"),
    )

    assert len(findings) == 1
    assert "soft_policy_temp" in findings[0]
    # ...and the reason the old check could not see it: the key is in exactly
    # one of the two sets, so an intersection-based diff compares nothing.
    assert not (set(reco) & set(flat)) - {"sf_policy_temp"}


def test_intersection_of_the_same_inputs_is_silent() -> None:
    """Pins the actual defect in the old instrument, so it cannot come back."""
    reco = {"sf_policy_temp": 0.012}
    flat = {"sf_policy_temp": 0.012, "soft_policy_temp": 3.0}

    shared = set(reco) & set(flat)
    assert all(reco[k] == flat[k] for k in shared), "old check would report 'all exact'"


def test_union_diff_accepts_a_key_parked_on_the_worker_default() -> None:
    reco: dict[str, object] = {}
    flat: dict[str, object] = {"fpu_reduction": 1.2}

    _report, findings = diff_reco_coverage(reco, flat, selfplay_keys=("fpu_reduction",))

    assert findings == []


def test_union_diff_flags_a_worker_default_key_moved_off_its_default() -> None:
    """The allowlist is self-invalidating: editing the yaml revokes it.

    "Unpublished but equal to the default" is harmless. "Unpublished and NOT
    equal to the default" is a request the loop will never honour.
    """
    reco: dict[str, object] = {}
    flat: dict[str, object] = {"fpu_reduction": 0.7}

    _report, findings = diff_reco_coverage(reco, flat, selfplay_keys=("fpu_reduction",))

    assert len(findings) == 1
    assert "fpu_reduction" in findings[0]
    assert "1.2" in findings[0]


def test_union_diff_reports_a_shared_key_whose_values_disagree() -> None:
    _report, findings = diff_reco_coverage(
        {"sf_policy_temp": 0.25}, {"sf_policy_temp": 0.012},
        selfplay_keys=("sf_policy_temp",),
    )

    assert len(findings) == 1
    assert "sf_policy_temp" in findings[0]


def test_union_diff_allows_the_documented_sf_nodes_divergence() -> None:
    """sf_nodes yaml = PID floor, reco = the live ramped budget. Not a finding."""
    _report, findings = diff_reco_coverage(
        {"sf_nodes": 698289}, {"sf_nodes": 5000}, selfplay_keys=("sf_nodes",),
    )

    assert findings == []


def test_union_diff_ignores_yaml_int_vs_json_float() -> None:
    """A manifest round trip turns 32 into 32.0; that is not a divergence."""
    _report, findings = diff_reco_coverage(
        {"gumbel_topk": 16.0}, {"gumbel_topk": 16}, selfplay_keys=("gumbel_topk",),
    )

    assert findings == []


def test_reco_worker_defaults_match_the_real_dataclasses() -> None:
    """The allowlist stores numbers; make them the dataclass's numbers.

    Without this, a change to (say) ``SearchConfig.fpu_reduction`` would leave
    the audit comparing the yaml against a value nothing uses, and the check
    would pass while the config lied.
    """
    yaml_to_field: dict[str, tuple[type, str]] = {
        "fpu_reduction": (SearchConfig, "fpu_reduction"),
        "fpu_at_root": (SearchConfig, "fpu_at_root"),
        "temperature_drop_plies": (TemperatureConfig, "drop_plies"),
        "temperature_after": (TemperatureConfig, "after"),
        "categorical_bins": (GameConfig, "categorical_bins"),
        "hlgauss_sigma": (GameConfig, "hlgauss_sigma"),
        "diff_focus_enabled": (DiffFocusConfig, "enabled"),
        "diff_focus_q_weight": (DiffFocusConfig, "q_weight"),
        "diff_focus_pol_scale": (DiffFocusConfig, "pol_scale"),
        "diff_focus_slope": (DiffFocusConfig, "slope"),
        "diff_focus_min": (DiffFocusConfig, "min_keep"),
    }
    assert set(yaml_to_field) == set(_RECO_WORKER_DEFAULT)

    for yaml_key, (cls, field_name) in yaml_to_field.items():
        actual = getattr(cls(), field_name)
        assert _RECO_WORKER_DEFAULT[yaml_key] == actual, (
            f"{yaml_key}: audit allowlist says {_RECO_WORKER_DEFAULT[yaml_key]!r}, "
            f"{cls.__name__}.{field_name} is {actual!r}"
        )


# --------------------------------------------------------------------------
# The CI guard: run the union diff on the production config
# --------------------------------------------------------------------------


def test_production_config_publishes_every_worker_affecting_key() -> None:
    """The A7 invariant, checkable at merge time instead of only on a live run.

    Builds the reco the server WOULD publish for the production yaml and diffs
    it against that same yaml over the union. Any new unpublished knob fails
    here; the one known gap is pinned by name so it stays a decision rather
    than a habit.
    """
    raw = yaml.safe_load(_PRODUCTION_YAML.read_text(encoding="utf-8"))
    flat = flatten_yaml_config(raw)
    reco = _reco_from(flat)

    _report, findings = diff_reco_coverage(reco, flat)
    offenders = {f.split(":", 1)[0] for f in findings}

    assert offenders == _KNOWN_UNPUBLISHED, (
        "reco coverage changed. Newly unpublished keys are a defect; a key that "
        "disappeared from _KNOWN_UNPUBLISHED was fixed — update the constant.\n"
        + "\n".join(f"  - {f}" for f in findings)
    )


def _diff_focus_field(yaml_key: str) -> str:
    """``diff_focus_min`` -> ``min_keep``; every other suffix is 1:1."""
    suffix = yaml_key.removeprefix("diff_focus_")
    return "min_keep" if suffix == "min" else suffix


_DIFF_FOCUS_KEYS = (
    "diff_focus_enabled", "diff_focus_q_weight",
    "diff_focus_pol_scale", "diff_focus_slope", "diff_focus_min",
)


def test_diff_focus_yaml_still_equals_what_the_worker_actually_runs() -> None:
    """Replaces the old divergence test, and is a stronger guard than it was.

    The worker never reads these keys — it constructs ``DiffFocusConfig()``. So
    a yaml value differing from the dataclass default is not a tuning knob, it
    is a lie: someone edits it, the keep-probability and row priority do not
    move, and the null result gets read as a verdict.

    The test that used to live here asserted the four tuned keys DIVERGED, which
    was true until the 2026-07-26 ledger entry pinned the yaml to the realized
    values as config honesty. It then failed exactly as its own docstring said it
    should ("parks the whole group on the defaults … this fails then, on
    purpose"), and the group moved into ``_RECO_WORKER_DEFAULT``. This asserts
    the pin HOLDS.
    """
    raw = yaml.safe_load(_PRODUCTION_YAML.read_text(encoding="utf-8"))
    flat = flatten_yaml_config(raw)
    running = DiffFocusConfig()

    diverging = {
        key: (flat[key], getattr(running, _diff_focus_field(key)))
        for key in _DIFF_FOCUS_KEYS
        if flat[key] != getattr(running, _diff_focus_field(key))
    }

    assert not diverging, (
        "the yaml asks for diff_focus values the worker will never read: "
        + ", ".join(f"{k} yaml={y!r} running={r!r}" for k, (y, r) in diverging.items())
        + ". Either pin the yaml back to the running value, or plumb the group "
        "through the reco — but plumbing changes the selfplay keep-probability "
        "and row priority, so it needs its own ledger entry and kill rule."
    )


def test_every_diff_focus_key_is_a_real_knob_not_a_typo() -> None:
    """A stale name here would make the guard above vacuously pass."""
    raw = yaml.safe_load(_PRODUCTION_YAML.read_text(encoding="utf-8"))
    flat = flatten_yaml_config(raw)
    df_fields = {f.name for f in fields(DiffFocusConfig)}

    for key in _DIFF_FOCUS_KEYS:
        assert key in flat, f"{key} is guarded but not in the production yaml"
        assert key in SELFPLAY_CONFIG_KEYS, f"{key} is not a declared selfplay key"
        assert _diff_focus_field(key) in df_fields, (
            f"{key} does not map onto a DiffFocusConfig field"
        )
    assert {f.name for f in fields(DiffFocusConfig)} == {
        _diff_focus_field(k) for k in _DIFF_FOCUS_KEYS
    }, "a DiffFocusConfig field exists with no yaml key guarding it"
