"""The control must SAMPLE from the buffer production is running.

⚑⚑ THE THIRD AXIS OF "OUR EXACT STACK", AND IT HAD NO INSTRUMENT AT ALL.
`test_lc0_control_arch.py` pins the network, `test_lc0_control_trainer.py` pins
`trainer_kwargs_from_config`. Neither can see `DiskReplayBuffer`, which
`tune/trainable_init.py` builds from `TrialConfig` fields that
`trainer_kwargs_from_config` does not read — so `scripts/lc0_control_train.py`
hand-wrote three buffer kwargs, took the constructor defaults for everything
else, and BOTH of those modules' gates passed while the buffer sat SEVEN axes
off production, `shuffle_cap` 20,000 against 100,000 among them.

The arm's hypothesis is *H_stack*. A plateau produced by 5x less hot-pool
diversity is a plateau of the RIG, and in the held-out top-1 slope it is
indistinguishable from the plateau the arm is looking for.

Same three propositions this file's two siblings test, one axis over: the pin is
fresh, the drift check can fail, and the check prefers the live file over any
in-tree copy.
"""
from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any

import pytest
import yaml

from chess_anti_engine.eval.lc0_control_arch import (
    LIVE_CONFIG_ENV,
    live_production_config_path,
)
from chess_anti_engine.eval.lc0_control_replay import (
    CONFIG_KWARGS,
    LC0_REPLAY_DEVIATIONS,
    LIVE_REPLAY_PIN,
    PER_RUN_KWARGS,
    ControlReplayDrift,
    apply_control_deviations,
    assert_buffer_kwargs_are_classified,
    assert_control_matches_live_replay,
    replay_kwargs_drift,
    replay_kwargs_signature,
)
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

REPO = Path(__file__).resolve().parent.parent
PRODUCTION = REPO / "configs" / "pbt2_small.yaml"
CONTROL = REPO / "configs" / "lc0_positive_control.yaml"


def _flat(path: Path) -> dict[str, Any]:
    return flatten_run_config_defaults(load_yaml_file(str(path)))


def test_the_recorded_pin_still_matches_the_live_file_when_one_is_named() -> None:
    """⚑⚑ THE PIN IS A COMMITTED COPY TOO — THIS IS WHAT KEEPS IT HONEST.

    Skips rather than passes where the live file is not named, so "not checked"
    never reads as "checked and fine".
    """
    live = live_production_config_path()
    if live is None:
        pytest.skip(
            f"${LIVE_CONFIG_ENV} is not set — LIVE_REPLAY_PIN's freshness "
            "against the LIVE production yaml was NOT checked by this run",
        )
    drift = replay_kwargs_drift(
        replay_kwargs_signature(_flat(live)), LIVE_REPLAY_PIN["kwargs"],
    )
    assert drift == {}, (
        f"LIVE_REPLAY_PIN is STALE against {live}: {drift}. Regenerate it with "
        "scripts/lc0_control_arch_pin.py --axis replay --live-config <live "
        "yaml>, then decide whether the arm follows production. ⚑ Hot-pool "
        "size and refresh cadence are training-affecting; that decision needs "
        "a ledger entry, not a pin edit."
    )


def test_the_shipped_control_matches_the_pin() -> None:
    """The gate must be capable of PASSING, or it is a constant, not a check.

    ⚑ And it must pass with NO declared deviation applied: the control config
    carries production's value for both overrides deliberately, so the driver's
    override is visible AS an override rather than hidden in the yaml.
    """
    provenance = assert_control_matches_live_replay(_flat(CONTROL), context="unit")
    assert "recorded replay pin" in provenance
    signature = replay_kwargs_signature(_flat(CONTROL))
    for key in LC0_REPLAY_DEVIATIONS:
        assert signature[key] == LIVE_REPLAY_PIN["kwargs"][key], (
            f"the control config carries its OWN value for {key!r}. The "
            "deviation is applied by the driver; putting it in the yaml makes "
            "it invisible to this guard."
        )


def test_a_control_on_the_buffer_defaults_is_refused() -> None:
    """⚑⚑ THE DEFECT ITSELF, as an input the gate must reject: the state PR
    #438's driver was in, where every unlisted kwarg fell back to
    `disk_buffer.py`'s constructor default."""
    raw = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    raw.pop("tune", None)
    raw["selfplay"].pop("diff_focus_pol_scale", None)
    raw["selfplay"].pop("diff_focus_q_weight", None)
    with pytest.raises(ControlReplayDrift) as excinfo:
        assert_control_matches_live_replay(
            flatten_run_config_defaults(raw), context="unit",
        )
    message = str(excinfo.value)
    for key in ("shuffle_cap", "refresh_interval", "refresh_shards"):
        assert key in message
    assert "H_stack" in message, (
        "the message must say WHY a buffer deviation voids this arm, not only "
        "that two numbers differ"
    )


def test_the_pin_records_a_buffer_the_in_tree_production_config_also_has() -> None:
    """⚑ THE ONE AXIS WHERE `main` IS NOT STALE, ASSERTED RATHER THAN ASSUMED.

    The arch and trainer pins exist because `main`'s `configs/pbt2_small.yaml`
    is behind the live file. On the REPLAY axis the two agree today — so the
    pin's justification here is only "the live file is the authority", not "the
    in-tree copy is wrong". Recording that keeps the next reader from inferring
    a staleness that is not there; if `main` and live ever diverge on a buffer
    key, this breaks and says so.
    """
    assert replay_kwargs_drift(
        replay_kwargs_signature(_flat(PRODUCTION)), LIVE_REPLAY_PIN["kwargs"],
    ) == {}


def test_the_live_file_beats_the_pin_when_both_are_available(tmp_path: Path) -> None:
    """A pin that wins over a readable live file is a pin nobody can correct."""
    live = tmp_path / "live.yaml"
    raw = yaml.safe_load(CONTROL.read_text(encoding="utf-8"))
    raw["tune"]["shuffle_buffer_size"] = 777_000
    live.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ControlReplayDrift, match="777000"):
        assert_control_matches_live_replay(_flat(CONTROL), live_config=live)


def test_a_caller_supplied_pin_is_the_one_judged_against() -> None:
    pin = copy.deepcopy(LIVE_REPLAY_PIN)
    pin["kwargs"]["shuffle_cap"] = 12_345
    with pytest.raises(ControlReplayDrift, match="12345"):
        assert_control_matches_live_replay(_flat(CONTROL), pin=pin)


def test_an_empty_reference_is_refused_not_agreed_with() -> None:
    """⚑ An empty reference makes every comparison vacuously true — a gate that
    cannot fail, which is the shape every finding on this arm has had."""
    with pytest.raises(ControlReplayDrift, match="EMPTY"):
        assert_control_matches_live_replay(
            _flat(CONTROL), pin={"kwargs": {}, "recorded": "never"},
        )


def test_every_disk_replay_buffer_parameter_is_classified() -> None:
    """⚑⚑ THE ANTI-DRIFT HALF. Without it, the next knob added to
    `DiskReplayBuffer` is absent from the mapping, absent from the signature,
    absent from the pin, and the control takes its default while every gate
    reports green — the same defect one turn of the crank later."""
    assert_buffer_kwargs_are_classified()
    parameters = set(inspect.signature(DiskReplayBuffer.__init__).parameters) - {"self"}
    assert set(CONFIG_KWARGS) <= parameters
    assert parameters >= PER_RUN_KWARGS


def test_a_new_buffer_parameter_breaks_the_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check above is vacuous until shown to fire. Removing a mapping entry
    is the same event as the constructor gaining a parameter."""
    mapping = dict(CONFIG_KWARGS)
    mapping.pop("refresh_shards")
    monkeypatch.setattr(
        "chess_anti_engine.eval.lc0_control_replay.CONFIG_KWARGS", mapping,
    )
    with pytest.raises(ControlReplayDrift, match="refresh_shards"):
        assert_buffer_kwargs_are_classified()


def test_a_stale_classification_entry_is_also_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other direction: a mapping naming a parameter the buffer no longer
    has is a silently-dropped kwarg at the construction site."""
    mapping = dict(CONFIG_KWARGS)
    mapping["shuffle_cap_v2"] = "shuffle_buffer_size"
    monkeypatch.setattr(
        "chess_anti_engine.eval.lc0_control_replay.CONFIG_KWARGS", mapping,
    )
    with pytest.raises(ControlReplayDrift, match="no longer has"):
        assert_buffer_kwargs_are_classified()


def test_every_declared_deviation_is_a_real_buffer_kwarg() -> None:
    """A deviation named for a kwarg that does not exist waives nothing and
    reads as if it waived something."""
    parameters = set(inspect.signature(DiskReplayBuffer.__init__).parameters)
    assert set(LC0_REPLAY_DEVIATIONS) <= parameters
    for key, reason in LC0_REPLAY_DEVIATIONS.items():
        assert len(reason) > 80, f"{key}'s recorded reason is a placeholder"


def test_the_driver_deviations_are_exactly_the_declared_ones() -> None:
    """⚑ `apply_control_deviations` is the ONE place the driver's overrides
    live, so the set it applies is checkable against the mapping the guard
    reads. Anything else is a hand-written override next to a check of
    different values — review F6's defect."""
    applied = apply_control_deviations(LIVE_REPLAY_PIN["kwargs"])
    changed = {
        key for key in LIVE_REPLAY_PIN["kwargs"]
        if applied[key] != LIVE_REPLAY_PIN["kwargs"][key]
    }
    assert changed == set(LC0_REPLAY_DEVIATIONS)


def test_an_override_with_no_recorded_reason_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "chess_anti_engine.eval.lc0_control_replay.LC0_REPLAY_DEVIATIONS",
        {"deterministic_refresh": "..."},
    )
    with pytest.raises(ControlReplayDrift, match="shard_recency_exponent"):
        apply_control_deviations(LIVE_REPLAY_PIN["kwargs"])


def test_the_signature_reads_trialconfig_defaults_not_a_local_guess() -> None:
    """⚑ A key absent from a yaml still has a realized production value, and it
    is `TrialConfig`'s default — not whatever a `.get(key, <guess>)` at the read
    site would supply. `shuffle_draw_cap_frac` is such a key in BOTH files."""
    raw = yaml.safe_load(PRODUCTION.read_text(encoding="utf-8"))
    assert "shuffle_draw_cap_frac" not in raw.get("tune", {})
    assert replay_kwargs_signature(_flat(PRODUCTION))["draw_cap_frac"] == 0.9
