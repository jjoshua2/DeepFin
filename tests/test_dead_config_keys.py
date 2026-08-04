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


def test_live_reload_declines_a_NUMERIC_dead_key_too(tmp_path, caplog) -> None:
    """The same branch, driven by a key whose inert value is not falsey.

    The test above uses a boolean dead key, where ``bool(value) !=
    bool(inert)`` and ``value != inert`` agree, so it cannot see whether the
    live-reload branch shares the startup refusal's type-aware predicate.
    ``gate_interval`` is inert at ``1``: a truthiness-only comparison calls
    ``5`` identical to it and the branch is never entered, which is exactly the
    silent overlay this module exists to prevent. It is also the ONLY case
    where the reload's ordering matters -- these three keys were in the
    restart-required skip set before this change, and a reordering would send
    the operator the "requires restart" lie for a value a restart refuses.
    """
    key = "gate_interval"
    yaml_path = tmp_path / "live.yaml"
    yaml_path.write_text(f"{key}: 5\n", encoding="utf-8")
    config: dict = {key: 1}

    with caplog.at_level(logging.WARNING):
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)

    assert config[key] == 1, "the dead value must NOT be overlaid"
    hits = [
        r.getMessage() for r in caplog.records
        if key in r.getMessage() and "reaches no production code path" in r.getMessage()
    ]
    assert len(hits) == 1, caplog.text
    assert "REFUSE to start" in hits[0]
    assert "requires restart" not in hits[0]
    # And the inert value passes through the branch without a warning, so an
    # operator can leave the corpse in the yaml at its realized value.
    caplog.clear()
    yaml_path.write_text(f"{key}: 1\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert not [
        r for r in caplog.records
        if key in r.getMessage() and "reaches no production code path" in r.getMessage()
    ], caplog.text


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


def _classify(params: dict, yaml_cfg: dict, realized: dict):
    from chess_anti_engine.tune.trainable_config_ops import (
        dead_config_key_inert_values,
    )
    from scripts.audit_realized_config import classify_config_provenance

    return classify_config_provenance(
        params, yaml_cfg, realized,
        restart_keys=frozenset(), construction_only_keys=frozenset(),
        dead_keys=dead_config_key_inert_values(),
    )


def test_the_audit_fires_on_a_LIVE_ADD_where_the_row_lacks_the_key() -> None:
    """The only operator action that can make a dead key live, and the one the
    intersection could not see.

    No file under ``configs/`` carries a dead key, so the sole route to a live
    value is an operator ADDING one to the yaml. ``_reload_yaml_into_config``
    then DECLINES to overlay it, so it never enters the trial config and never
    reaches the result row -- and a sweep over ``yaml & realized`` drops it and
    prints "ok every shared yaml key has reached the running trial".

    ``realized`` deliberately does NOT contain the key here. The previously
    pinned state (key present in the row at a live value) is unreachable on the
    production path anyway, because ``reject_dead_config_keys`` raises at
    startup before such a row could ever be written.
    """
    report, findings = _classify({}, {_KEY: True}, {"lr": 3e-5})

    assert any(line.startswith(f"  DEAD-KEY {_KEY}:") for line in report), report
    assert len(findings) == 1, findings
    assert _KEY in findings[0]
    assert "REFUSE" in findings[0]
    assert "DECLINES" in findings[0], (
        "the finding must say the key is ABSENT from the running config, not "
        "merely stale -- that distinction is why it needs its own sweep"
    )
    # It must NOT be described as restart-required -- that is the wrong advice.
    assert "restart required" not in findings[0].lower()


def test_the_audit_also_fires_when_only_the_ROW_carries_it() -> None:
    """The other half of the union: yaml cleaned up, row still carrying it."""
    _, findings = _classify({}, {}, {_KEY: True})
    assert len(findings) == 1, findings
    assert "running=True" in findings[0]


def test_the_audit_is_silent_at_the_inert_value_on_every_shape() -> None:
    """Or it would fire on every run of a healthy trial."""
    for params, yaml_cfg, realized in (
        ({}, {_KEY: False}, {"lr": 3e-5}),
        ({_KEY: False}, {_KEY: False}, {_KEY: False}),
        ({}, {}, {_KEY: 0}),
        ({}, {}, {"lr": 3e-5}),
    ):
        report, findings = _classify(params, yaml_cfg, realized)
        assert findings == [], (params, yaml_cfg, realized, findings)
        assert not [line for line in report if "DEAD-KEY" in line]


def test_the_audit_and_the_refusal_share_ONE_predicate() -> None:
    """A guard must share the criterion's instrument.

    The audit says "safe" and the trial says "refuse" about the same value only
    if the two derive that answer from the same code. They did not: the audit
    used ``bool(value)`` while the refusal compared ``bool(value) != bool(inert)``,
    which agree only while every inert value is falsey -- true today, and a
    silent trap the first time a dead key is added that is inert at True.

    Driven over a value matrix rather than argued: for every value, "the audit
    raised a finding" must equal "the trial would refuse to start".
    """
    from chess_anti_engine.tune.trainable_config_ops import (
        dead_config_key_inert_values,
        reject_dead_config_keys,
    )

    for value in (True, False, 1, 0, "yes", "", None, 2):
        _, findings = _classify({}, {_KEY: value}, {"lr": 3e-5})
        try:
            reject_dead_config_keys({_KEY: value})
            refused = False
        except ValueError:
            refused = True
        assert bool(findings) == refused, (
            f"{value!r}: audit findings={len(findings)} but refusal={refused}"
        )

    # ...and the same agreement asserted for EVERY dead key rather than for
    # the one that exists today, at its inert value and at one that is not.
    #
    # This USED TO BE `assert not bool(inert)` for every key -- a tripwire for
    # the bool()-only predicate, which agreed with the refusal only while every
    # inert value was falsey. The gate_* corpses (audit G3-11) are inert at
    # 1 / 0.50, where truthiness cannot tell `gate_interval: 5` from the inert
    # `1`, so the tripwire's premise is gone and the agreement itself is
    # asserted instead -- over a value that IS the inert one and a value that
    # is not, for each key.
    for key, inert in dead_config_key_inert_values().items():
        if isinstance(inert, bool):
            live = not inert
        elif isinstance(inert, (int, float)):
            live = type(inert)(inert + 1)
        else:
            live = f"{inert}_CHANGED"
        for value, expect_refusal in ((inert, False), (live, True)):
            _, findings = _classify({}, {key: value}, {"lr": 3e-5})
            try:
                reject_dead_config_keys({key: value})
                refused = False
            except ValueError:
                refused = True
            assert refused is expect_refusal, (
                f"{key}={value!r}: refusal={refused}, expected {expect_refusal}"
            )
            assert bool(findings) == refused, (
                f"{key}={value!r}: audit findings={len(findings)} but "
                f"refusal={refused}"
            )


def test_the_shared_predicate_holds_for_a_key_that_is_inert_at_TRUE(
    monkeypatch,
) -> None:
    """The case every ``bool(value)`` shortcut gets right by luck today.

    Both the audit and the refusal currently agree with a bare truthiness test,
    because the one dead key in existence is inert at ``False``. That is a
    property of today's key set, not of the code, and a test that only exercises
    that key cannot tell a correct implementation from a lucky one -- so a
    SYNTHETIC dead key inert at ``True`` is registered here, and both sides must
    then treat ``True`` as the safe value and ``False`` as the violation.

    Registered on the module's own mapping (restored by ``monkeypatch``) so both
    the accessor the audit reads and the global the refusal reads see it --
    otherwise the two would be driven from different sources and the test would
    be asserting the very split it exists to forbid.
    """
    from chess_anti_engine.tune import trainable_config_ops as ops

    synthetic = "synthetic_dead_key_inert_at_true"
    monkeypatch.setitem(ops._DEAD_CONFIG_KEY_INERT_VALUES, synthetic, True)

    # The INERT value is True here: silent on both sides.
    _, findings = _classify({}, {synthetic: True}, {"lr": 3e-5})
    assert findings == [], findings
    ops.reject_dead_config_keys({synthetic: True})  # must not raise

    # ...and False is the violation, on both sides.
    _, findings = _classify({}, {synthetic: False}, {"lr": 3e-5})
    assert len(findings) == 1, findings
    assert "inert=True" in _classify({}, {synthetic: False}, {"lr": 3e-5})[0][0]
    with pytest.raises(ValueError, match=synthetic):
        ops.reject_dead_config_keys({synthetic: False})


def test_the_audit_binds_the_dead_set_to_the_real_one() -> None:
    """The script must pass the production set, not an empty default.

    An injected argument that no caller supplies is a branch that cannot fire
    -- the failure mode the construction-only work already had to fix once.
    """
    import inspect

    import scripts.audit_realized_config as audit

    src = inspect.getsource(audit)
    assert "dead_keys=dead_config_key_inert_values()," in src
    # ...and swept over the UNION, not the intersection: the live-ADD case has
    # no realized value at all.
    assert "for key in sorted(set(flat_yaml) | set(realized)):" in src


def test_the_removed_1sim_gate_knobs_are_refused_at_a_live_value() -> None:
    """MUTATION (audit G3-11): drop the three ``gate_*`` corpses from the set.

    ``gate_interval`` / ``gate_threshold`` / ``gate_mcts_sims`` are the knobs
    of the REMOVED 1-sim vs-Stockfish gate. Each appears in exactly three
    places -- the yaml allowlist, the live-reload skip set, and ``TrialConfig``,
    which parses them and never reads them -- and ``gate_config_from_dict``
    looks at none of them. Only ``gate_games`` refused, and it refused because
    a non-zero value asks for BEHAVIOUR that was deleted; these three were left
    "tolerated at any value" on the grounds that they are inert scalars, which
    is true of the value and false of the operator's belief. ``gate_interval: 5``
    started a run, read back correctly from the trial's own config, and did
    nothing at all.

    THE TRUTHINESS TRAP IS THE POINT: their inert values are 1 / 0.50 / 1, so a
    ``bool(value) != bool(inert)`` predicate cannot tell ``gate_interval: 5``
    from the inert ``1``. That is why the shared predicate is type-aware.
    """
    from chess_anti_engine.tune.trainable_config_ops import (
        dead_config_key_inert_values,
        reject_dead_config_keys,
        restart_required_config_keys,
    )

    inert = dead_config_key_inert_values()
    for key, live in (("gate_interval", 5), ("gate_threshold", 0.55),
                      ("gate_mcts_sims", 32)):
        assert key in inert, key
        # The inert value is tolerated: deleting a key from a live yaml is
        # itself a reload risk, so an operator must be able to leave it.
        reject_dead_config_keys({key: inert[key]})
        with pytest.raises(ValueError, match=key) as exc:
            reject_dead_config_keys({key: live})
        assert key in str(exc.value)
        assert "REMOVED 1-sim" in str(exc.value), (
            "the refusal must say what happened to the knob, not merely that "
            "it is dead"
        )
        # A dead key is NOT restart-required: a restart refuses it rather than
        # applying it, and saying otherwise sends an operator into a crash.
        assert key not in restart_required_config_keys(), key
        # ...while a truthiness-only predicate would call 5 and 1 the same.
        assert bool(live) == bool(inert[key]), (
            "if this ever stops holding, this test has stopped covering the "
            "trap it was written for"
        )

    # gate_games keeps its OWN refusal, with its own message, and is therefore
    # deliberately not in the dead set.
    assert "gate_games" not in inert

    # THE PRODUCTION YAML MUST STILL START. Its three values are the inert ones.
    from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file
    flat = flatten_run_config_defaults(load_yaml_file(
        str(_REPO / "configs" / "pbt2_small.yaml")))
    reject_dead_config_keys(flat)
    for key in ("gate_interval", "gate_threshold", "gate_mcts_sims"):
        assert key in flat, f"{key} is in the shipped yaml; keep it inert there"
