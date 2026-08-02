"""A yaml key that reaches NO production consumer must be refused, not accepted.

``replay_sf_gap_priority_signed`` is in the yaml allowlist
(``utils/config_yaml.py``), parsed into ``TrialConfig``
(``trial_config.py``), and supported by
``DiskReplayBuffer.sf_gap_priority_signed`` — and never passed to the buffer
the trial actually trains from. The production construction site
(``trainable_init._init_replay_buffers``) hands over ``sf_gap_priority_weight``
alone, and the per-iteration live push in ``trainable.py`` pushes the same four
live knobs and not this one. Its only real consumer is the OFFLINE
``scripts/retarget_retrain.py``.

That is the ``soft_policy_temp`` / E13 shape: a knob that reads back correctly
from the trial's own config and does nothing. It is refused rather than wired,
on the precedent of ``promotion_gate.gate_config_from_dict``'s ``gate_games``
check, because the gap-priority family was KILLED by experiment #104 and
because ``signed`` mode is not a refinement of the default — it swaps the boost
SOURCE from the stored ``priority_sf_search_gap`` column to a recomputed signed
difference gated on different ``has_`` flags, so a chunk carrying one and not
the other silently gets no shaping at all.

The premise is pinned too, not just the guard: if somebody later WIRES the key,
``test_the_dead_key_is_still_dead`` fails and tells them to delete the entry.
An entry that has stopped being true is how a guard turns into folklore.
"""
from __future__ import annotations

import io
import logging
import tokenize
from pathlib import Path

import pytest

from chess_anti_engine.tune.trainable_config_ops import (
    _DEAD_CONFIG_KEY_INERT_VALUES,
    _reload_yaml_into_config,
    reject_dead_config_keys,
)
from chess_anti_engine.utils.config_yaml import _FLAT_ALLOWLIST

_REPO = Path(__file__).resolve().parents[1]
_KEY = "replay_sf_gap_priority_signed"


def _executable_source(path: Path) -> str:
    """Module source with comments and docstrings removed.

    The dead-key check below has to read CODE. A plain substring search over the
    file would be satisfied by the comment that explains WHY the key is dead --
    i.e. writing the explanation would break the guard, which is the fastest way
    to get the explanation deleted.
    """
    out: list[str] = []
    prev_type = tokenize.INDENT
    for tok in tokenize.generate_tokens(
        io.StringIO(path.read_text(encoding="utf-8")).readline,
    ):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev_type in (
            tokenize.INDENT, tokenize.DEDENT, tokenize.NEWLINE, tokenize.NL,
        ):
            continue  # docstring
        if tok.type not in (tokenize.NL, tokenize.NEWLINE):
            prev_type = tok.type
        out.append(tok.string)
    return " ".join(out)


# ---------------------------------------------------------------------------
# The premise: the key really is dead, and stays checked
# ---------------------------------------------------------------------------


def test_the_dead_key_is_still_dead() -> None:
    """No production file may pass or push ``replay_sf_gap_priority_signed``.

    If this fails, the key has been wired: remove it from
    ``_DEAD_CONFIG_KEY_INERT_VALUES`` (and add the ledger entry experiment #104
    requires) rather than relaxing the assertion.
    """
    for rel in (
        "chess_anti_engine/tune/trainable_init.py",
        "chess_anti_engine/tune/trainable.py",
    ):
        src = _executable_source(_REPO / rel)
        assert _KEY not in src, (
            f"{rel} now references {_KEY} in CODE; the key may no longer be dead"
        )
    # It is still ACCEPTED, which is why a refusal is needed rather than a
    # deletion: dropping it from the allowlist would make the all-or-nothing
    # live-yaml validator reject the WHOLE reload for any yaml carrying it.
    assert _KEY in _FLAT_ALLOWLIST
    # ...and it still has real offline consumers, which is why the key is not
    # deleted from TrialConfig. BOTH are pinned: an unpinned consumer can be
    # deleted as dead, and the refusal message would then name a file that no
    # longer uses the key.
    for offline_rel in ("scripts/retarget_retrain.py", "scripts/holdout_policy_screen.py"):
        offline = (_REPO / offline_rel).read_text(encoding="utf-8")
        assert f"sf_gap_priority_signed=tc.{_KEY}" in offline, offline_rel


def test_the_inert_value_is_the_realized_default() -> None:
    """The tolerated value must be the one production actually runs at.

    A guard whose "safe" value is not the realized one would fire on the live
    config the first time anyone restarted.
    """
    from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
    from chess_anti_engine.tune.trial_config import TrialConfig

    inert = _DEAD_CONFIG_KEY_INERT_VALUES[_KEY]
    assert inert is False
    assert TrialConfig().replay_sf_gap_priority_signed is False
    import inspect

    sig = inspect.signature(DiskReplayBuffer.__init__)
    assert sig.parameters["sf_gap_priority_signed"].default is False
    # And no shipped config sets it, so the guard cannot fire on today's yaml.
    for cfg in sorted((_REPO / "configs").glob("*.yaml")):
        assert _KEY not in cfg.read_text(encoding="utf-8"), cfg.name


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_refuses_the_live_value_and_names_the_key() -> None:
    with pytest.raises(ValueError, match=_KEY) as excinfo:
        reject_dead_config_keys({_KEY: True})
    message = str(excinfo.value)
    assert _KEY in message
    assert "reaches no production code path" in message
    # Naming the offline consumer is what stops the next reader concluding the
    # key is simply obsolete and deleting it out from under retarget_retrain.
    assert "retarget_retrain" in message
    assert "holdout_policy_screen" in message


@pytest.mark.parametrize("truthy", [True, 1, "yes"])
def test_refuses_every_truthy_spelling(truthy: object) -> None:
    """YAML says ``true``/``1``/``yes`` for one intent; all must be refused."""
    with pytest.raises(ValueError, match=_KEY):
        reject_dead_config_keys({_KEY: truthy})


@pytest.mark.parametrize("config", [{}, {_KEY: False}, {_KEY: 0}])
def test_tolerates_absence_and_the_inert_value(config: dict) -> None:
    """Deleting a key from a live yaml is itself a reload risk.

    So the operator must be able to leave it in place at the value the buffer
    already runs at, exactly as ``gate_threshold`` is tolerated beside
    ``gate_games``.
    """
    reject_dead_config_keys(config)


def test_the_guard_is_on_the_production_construction_path() -> None:
    """It must be called where the argument goes missing, not in a banner.

    Asserted against the source of ``_init_replay_buffers`` so a refactor that
    moves the buffer construction without moving the check is caught. A banner
    check would be skippable by a resume that takes a different startup branch.
    """
    import inspect

    from chess_anti_engine.tune import trainable_init

    src = inspect.getsource(trainable_init._init_replay_buffers)
    assert "reject_dead_config_keys(config)" in src
    assert src.index("reject_dead_config_keys(config)") < src.index(
        "DiskReplayBuffer(",
    ), "the refusal must precede the construction it describes"


def test_live_reload_declines_the_value_and_says_a_restart_will_refuse(
    tmp_path, caplog,
) -> None:
    """The live path must not apply it, and must not say 'requires restart'.

    Restarting does not honour the value, it makes the trial refuse to start,
    so the generic restart message would be the same lie relocated. Both halves
    are asserted in the SAME log record: matching them separately in
    ``caplog.text`` is satisfied by two unrelated records.
    """
    yaml_path = tmp_path / "live.yaml"
    yaml_path.write_text(f"{_KEY}: true\n", encoding="utf-8")
    config: dict = {_KEY: False}

    with caplog.at_level(logging.WARNING):
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)

    assert config[_KEY] is False, "the dead value must NOT be overlaid"
    hits = [
        r.getMessage() for r in caplog.records
        if _KEY in r.getMessage() and "reaches no production code path" in r.getMessage()
    ]
    assert len(hits) == 1, caplog.text
    assert "REFUSE to start" in hits[0]
    assert "requires restart" not in hits[0]


def test_startup_reload_still_applies_it_so_the_refusal_can_fire(tmp_path) -> None:
    """The non-live overlay must NOT swallow the value.

    If startup silently dropped it too, the operator's edit would vanish with
    no error anywhere -- the guard would be unreachable and this whole module
    would be testing a branch production never takes.
    """
    yaml_path = tmp_path / "live.yaml"
    yaml_path.write_text(f"{_KEY}: true\n", encoding="utf-8")
    config: dict = {_KEY: False}
    _reload_yaml_into_config(config, str(yaml_path), live_reload=False)
    assert config[_KEY] is True
    with pytest.raises(ValueError, match=_KEY):
        reject_dead_config_keys(config)


# ---------------------------------------------------------------------------
# The FOURTH reload class: not overlaid, and not restart-required either
# ---------------------------------------------------------------------------


def test_a_dead_key_is_its_own_reload_class() -> None:
    """Declined live like a restart-required key, but a restart REFUSES it.

    Reporting it as restart-required would send an operator into the one action
    that turns a silent no-op into a refusal to start, so it must not be in
    that set -- and it must be reachable on its own, or a provenance tool has
    no way to name the case.
    """
    from chess_anti_engine.tune.trainable_config_ops import (
        construction_only_config_keys,
        dead_config_keys,
        restart_required_config_keys,
    )

    assert _KEY in dead_config_keys()
    assert _KEY not in restart_required_config_keys()
    assert _KEY not in construction_only_config_keys()
    assert not (dead_config_keys() & restart_required_config_keys())


def test_the_provenance_audit_names_the_dead_key_instead_of_calling_it_healthy() -> None:
    """Without the fourth class it falls through every branch and reads clean.

    The startup overlay puts the yaml value into the realized row, so
    yaml == realized -- which every remaining branch treats as agreement. The
    finding has to be raised on the VALUE, not on a diff.
    """
    from scripts.audit_realized_config import classify_config_provenance
    from chess_anti_engine.tune.trainable_config_ops import dead_config_keys

    params = {_KEY: False}
    yaml_cfg = {_KEY: True}
    realized = {_KEY: True}  # the startup overlay applied it; nothing consumed it

    report, findings = classify_config_provenance(
        params, yaml_cfg, realized,
        restart_keys=frozenset(), construction_only_keys=frozenset(),
        dead_keys=dead_config_keys(),
    )
    assert any(line.startswith(f"  DEAD-KEY {_KEY}:") for line in report), report
    assert len(findings) == 1, findings
    assert _KEY in findings[0]
    assert "REFUSE" in findings[0]
    # It must NOT be described as restart-required -- that is the wrong advice.
    assert "restart required" not in findings[0].lower()

    # ...and the inert value is silent, or the audit would fire on every run.
    report_ok, findings_ok = classify_config_provenance(
        {_KEY: False}, {_KEY: False}, {_KEY: False},
        restart_keys=frozenset(), construction_only_keys=frozenset(),
        dead_keys=dead_config_keys(),
    )
    assert findings_ok == []
    assert not [line for line in report_ok if "DEAD-KEY" in line]


def test_the_audit_binds_the_dead_set_to_the_real_one() -> None:
    """The script must pass the production set, not an empty default.

    An injected argument that no caller supplies is a branch that cannot fire
    -- the failure mode the construction-only work already had to fix once.
    """
    import inspect

    import scripts.audit_realized_config as audit

    src = inspect.getsource(audit)
    assert "dead_keys=dead_config_keys()," in src
