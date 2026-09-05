"""The control must build the architecture PRODUCTION IS RUNNING.

⚑⚑ THE IN-TREE `configs/pbt2_small.yaml` IS NOT NECESSARILY THE FILE PRODUCTION
READS. The live run reads the yaml in the LIVE working tree, on the live branch,
and that file moves independently of whatever a given branch carries. Measured
2026-08-16 from `main` and unchanged by the bt4heads merge: in-tree 63,084,128
trainable params, live 61,444,448, the gap being exactly `aux_policy_head_dim:
128`, `categorical_head_coupled: true`, `policy_embedding_mode: linear` — which
`main` has in its SCHEMA (PR #439) but not in its committed production yaml.

So this file tests the OTHER instrument: the pin, the drift check, and the
fact that the check prefers the live file over any in-tree copy.

⚑ TWO OF THESE TESTS WERE MERGE-GUARDS, WRITTEN SO THAT THE SITUATION
IMPROVING BROKE THEM. It improved on 2026-08-16, so they are INVERTED, not
deleted: `test_this_tree_builds_the_pinned_live_architecture` used to require
the flattener to RAISE and now requires it to build the pinned 61,444,448, and
`test_the_control_config_matches_the_recorded_pin` used to require the control
to be REFUSED and now requires it to be accepted — while still proving that
dropping any one bt4heads key is refused, so the gate can still fail.

⚑⚑ `test_the_pin_names_the_bt4heads_keys_the_in_tree_config_lacks` IS NOW A
TWO-WORLD CONTRACT, for the third time this file has had to learn the same
lesson. It was written on `main`, where "the in-tree production yaml is stale by
exactly these three keys" is a fact; on `ops/live-20260725` the in-tree yaml IS
the live yaml, the same sentence is false, and the test went red for being
right. The accepted states are now named:

* **STATE MAIN** — the in-tree `model:` section carries NONE of
  `LIVE_ONLY_MODEL_KEYS`. The pin is the reference because the file next door is
  behind.
* **STATE LIVE** — the in-tree `model:` section equals `LIVE_ARCH_PIN["model"]`
  key for key and value for value. The file next door IS the live file; the pin
  is a faithful second copy of it.

A PARTIAL bundle fails, and that is the point: it is the only state in which
`tests/test_lc0_control_config.py`'s "judge the architecture against the pin"
would read as justified while describing a file that is half-live. ⚑ The pin
stays the arbiter in both worlds — STATE LIVE is accepted because the file
EQUALS the pin, never because a mismatch was read as progress.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from chess_anti_engine.eval.lc0_control_arch import (
    LIVE_ARCH_PIN,
    LIVE_CONFIG_ENV,
    ControlArchitectureDrift,
    assert_control_matches_live_architecture,
    live_production_config_path,
    model_section,
    model_section_drift,
    unique_storage_param_count,
)
from chess_anti_engine.model import build_model, model_config_from_flat_config
from chess_anti_engine.utils import flatten_run_config_defaults

REPO = Path(__file__).resolve().parent.parent
CONTROL = REPO / "configs" / "lc0_positive_control.yaml"
PRODUCTION = REPO / "configs" / "pbt2_small.yaml"

# The bt4heads bundle, promoted to LIVE production 2026-08-15. Two of the three
# touch the POLICY HEAD, which is the arm's only yardstick.
LIVE_ONLY_MODEL_KEYS = (
    "aux_policy_head_dim",
    "categorical_head_coupled",
    "policy_embedding_mode",
)


def _raw(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_this_tree_builds_the_pinned_live_architecture() -> None:
    """⚑⚑ THE INVERTED MERGE-GUARD. Until 2026-08-16 this asserted the OPPOSITE.

    It used to require ``flatten_run_config_defaults`` to RAISE on
    ``aux_policy_head_dim``: the key was not in ``utils/config_yaml.py``'s
    schema, and CLAUDE.md category (a) makes an unknown ``model:`` key FATAL AT
    LAUNCH, so "re-point the control at the live model: section" was not a
    one-line config edit — it needed the promotion on this branch first. PR
    #439 landed it, so the assertion is INVERTED, not deleted.

    Three teeth, each of which has been watched to fail under mutation:

    * the flatten must SUCCEED — a bt4heads key leaving ``main``'s schema again
      puts the arm back where it started, and that must be loud;
    * each key must survive into the ``ModelConfig`` — "in the schema but never
      read by the builder" is this repo's signature defect, and a schema-only
      key would let the control config claim an architecture it does not build;
    * the BUILT net must land on the pin's 61,444,448 unique-storage params and
      77,173,088 state_dict numel. This is a RE-MEASUREMENT of the pin, not a
      restatement of it: it runs this tree's builder over the pinned ``model:``
      section, so the pin can no longer be a number nothing checks.
    """
    live_like = copy.deepcopy(_raw(PRODUCTION))
    live_like["model"] = dict(LIVE_ARCH_PIN["model"])
    flat = flatten_run_config_defaults(live_like)

    model_cfg = model_config_from_flat_config(flat)
    for key in LIVE_ONLY_MODEL_KEYS:
        assert getattr(model_cfg, key) == LIVE_ARCH_PIN["model"][key], (
            f"{key} is in the schema but does not reach ModelConfig — the "
            "control config can carry it and the built net will not have it"
        )

    built = build_model(model_cfg)
    assert unique_storage_param_count(built) == (
        LIVE_ARCH_PIN["trainable_params_unique_storage"]
    ), "this tree builds a different net from the pinned model: section"
    assert sum(v.numel() for v in built.state_dict().values()) == (
        LIVE_ARCH_PIN["state_dict_numel_sum"]
    )


def test_the_pin_names_the_bt4heads_keys_the_in_tree_config_lacks() -> None:
    """⚑⚑ THE TWO-WORLD FORM. IT IS THE LOAD-BEARING TEST OF THIS FILE.

    PR #439 gave ``main`` the SCHEMA for the bt4heads keys; it did not put them
    in ``main``'s committed ``configs/pbt2_small.yaml``. The live branch's copy
    of that same path IS the live file and carries them. So "the in-tree config
    lacks these three keys" is branch-dependent, and asserting it unconditionally
    asserts which branch the checkout is on rather than anything about the
    reference decision it exists to justify.

    This is the RECORDED REASON that
    ``tests/test_lc0_control_config.py::test_architecture_is_identical_to_production``
    judges against ``LIVE_ARCH_PIN`` rather than against the yaml sitting next
    to it, and that reason holds in exactly two states:

    * **STATE MAIN** — none of the three keys is in the in-tree ``model:``
      section. The pin is the reference because the file next door is behind.
    * **STATE LIVE** — the in-tree ``model:`` section EQUALS the pin, whole:
      same keys, same values. Checked across the entire section rather than the
      three bt4heads keys, because "carries the bundle" and "is the file the pin
      was cut from" are different claims and only the second one licenses
      calling the in-tree copy faithful.

    Anything in between fails and names the key. ⚑ Do not fix such a failure by
    editing this line: a half-adopted bundle means the tree is claiming a
    reference it does not have, and the fix is in the yaml or the pin.
    """
    in_tree = model_section(_raw(PRODUCTION))
    pinned = dict(LIVE_ARCH_PIN["model"])
    for key in LIVE_ONLY_MODEL_KEYS:
        assert key in pinned, f"the pin lost {key}"

    carried = sorted(key for key in LIVE_ONLY_MODEL_KEYS if key in in_tree)
    if not carried:
        return  # STATE MAIN, the state this test was written in.

    absent = sorted(set(LIVE_ONLY_MODEL_KEYS) - set(in_tree))
    assert not absent, (
        f"the in-tree production config carries {carried} of the bt4heads "
        f"bundle but not {absent}. A PARTIAL bundle is neither `main`'s world "
        "nor LIVE's: it builds a net nothing has measured, while "
        "tests/test_lc0_control_config.py goes on describing the file as "
        "known-stale. Finish the sync or drop the partial keys."
    )

  # Not `model_section_drift`: its lines are labelled control=/live=, and both
  # sides here are references. A mislabelled diff is worse than no diff.
    drift = [
        f"{key}: in_tree={in_tree.get(key, '<absent>')!r} "
        f"pin={pinned.get(key, '<absent>')!r}"
        for key in sorted(set(in_tree) | set(pinned))
        if in_tree.get(key, "<absent>") != pinned.get(key, "<absent>")
    ]
    assert drift == [], (
        "the in-tree production config carries the whole bt4heads bundle, so it "
        "is claiming to BE the live file — but it does not equal LIVE_ARCH_PIN:"
        "\n  " + "\n  ".join(drift) + "\n"
        "Regenerate the pin with scripts/lc0_control_arch_pin.py --live-config "
        "<live yaml>, then revisit the reference decision in "
        "tests/test_lc0_control_config.py."
    )


def test_the_control_config_matches_the_recorded_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ THE INVERTED FINDING. It used to assert the arm was REFUSED.

    The shipped control config carried none of the bt4heads keys, so the arm
    would have trained ``main``'s 63,084,128-param net rather than production's
    61,444,448-param one, and this test pinned that refusal so the state could
    not be forgotten. The config now carries them, so the refusal is inverted
    into an acceptance — with the failing direction kept, because "it passes"
    on its own is a constant, not a gate.

    Both directions here: the shipped config is ACCEPTED against the pin, and
    dropping ANY ONE of the three keys is still refused with that key named and
    the "our stack" warning intact.
    """
    monkeypatch.delenv(LIVE_CONFIG_ENV, raising=False)
    control = _raw(CONTROL)
    provenance = assert_control_matches_live_architecture(control)
    assert "recorded pin" in provenance
    assert "THE LIVE FILE WAS NOT READ" in provenance

    for key in LIVE_ONLY_MODEL_KEYS:
        crippled = copy.deepcopy(control)
        del crippled["model"][key]
        with pytest.raises(ControlArchitectureDrift) as excinfo:
            assert_control_matches_live_architecture(crippled)
        message = str(excinfo.value)
        assert key in message, f"the drift report must name {key}"
        assert "may be quoted as 'our stack'" in message
        assert "recorded pin" in message, (
            "the message must name what it judged against"
        )


def test_the_recorded_pin_still_matches_the_live_file_when_one_is_named() -> None:
    """⚑⚑ THE PIN IS A COMMITTED COPY TOO — THIS IS WHAT KEEPS IT HONEST.

    Every other gate here judges against ``LIVE_ARCH_PIN``, which is the same
    shape of object as the in-tree production yaml: a copy that can go stale
    while the thing it describes moves. The pin's defence is that it is
    REGENERABLE from the live file, and this is the test that regenerates it.
    It can only run where the live file exists, so when it cannot it says what
    was NOT checked rather than reporting a pass.

    ⚑ It does NOT move any other test's verdict. The control is judged against
    the pin whether or not ``$CHESS_LIVE_PRODUCTION_CONFIG`` is set — a guard
    whose verdict follows the operator's shell is review F6's defect. This is a
    SEPARATE proposition ("the pin is not stale") that only the operator can
    supply the instrument for.
    """
    live = live_production_config_path()
    if live is None:
        pytest.skip(
            f"${LIVE_CONFIG_ENV} is not set — the pin's freshness against the "
            "LIVE production yaml was NOT checked by this run",
        )
    live_model = model_section(_raw(live))
    pinned = dict(LIVE_ARCH_PIN["model"])
  # Not `model_section_drift`: its lines are labelled control=/live=, and both
  # sides here are references. A mislabelled diff is worse than no diff.
    drift = [
        f"{key}: live={live_model.get(key, '<absent>')!r} "
        f"pin={pinned.get(key, '<absent>')!r}"
        for key in sorted(set(live_model) | set(pinned))
        if live_model.get(key, "<absent>") != pinned.get(key, "<absent>")
    ]
    assert drift == [], (
        f"LIVE_ARCH_PIN is STALE against {live}:\n  " + "\n  ".join(drift) + "\n"
        "Regenerate it with scripts/lc0_control_arch_pin.py --live-config "
        "<live yaml>, then decide whether the control follows production."
    )


def test_the_check_passes_when_the_model_section_matches_the_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard must be capable of PASSING, or it is a constant, not a check.

    ⚑ REVIEW F6: this test used to FAIL whenever ``$CHESS_LIVE_PRODUCTION_CONFIG``
    was set — i.e. in the operator's own shell, in the mode the rig's docs
    prescribe — because ``live_config=None`` meant "no live config" here and
    "go read the env" inside the function. The env is now resolved at the CALL
    SITE, so ``None`` has exactly one meaning. The ``setenv`` below proves it:
    the verdict must not move.
    """
    monkeypatch.setenv(LIVE_CONFIG_ENV, "/nonexistent/live/pbt2_small.yaml")
    matching = {"model": dict(LIVE_ARCH_PIN["model"])}
    provenance = assert_control_matches_live_architecture(matching, live_config=None)
    assert "recorded pin" in provenance


def test_a_control_matching_the_stale_pin_still_fails_against_the_live_file(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE POINT OF THE MODULE: the reference is a file OUTSIDE this tree.

    A control that matches the recorded pin EXACTLY must still fail when the
    live file says something else — otherwise the check is pinned to a
    committed copy and can never notice production moving, which is the exact
    blindness the in-tree config diff has.
    """
    live = tmp_path / "live_pbt2_small.yaml"
    moved = dict(LIVE_ARCH_PIN["model"])
    moved["embed_dim"] = 640
    live.write_text(yaml.safe_dump({"model": moved}), encoding="utf-8")
    with pytest.raises(ControlArchitectureDrift, match="embed_dim"):
        assert_control_matches_live_architecture(
            {"model": dict(LIVE_ARCH_PIN["model"])}, live_config=live,
        )


def test_a_caller_supplied_pin_is_the_one_judged_against() -> None:
    """⚑⚑ REVIEW F4 — ``pin=`` WAS ACCEPTED AND SILENTLY IGNORED.

    Only ``pin["trainable_params_unique_storage"]`` was read; the reference
    ``model:`` section came from the MODULE GLOBAL, so a caller's pin judged
    nothing and the raised message reported ``LIVE_ARCH_PIN``'s provenance as
    though it were the caller's. That is this repo's signature defect inside
    the module written to close an instance of it.

    Both directions here: a control matching the caller's pin PASSES (it would
    have raised against the global), and one that does not FAILS.
    """
    pin: dict[str, Any] = {
        "recorded": "2020-01-01",
        "live_branch": "toy",
        "live_commit": "deadbeef",
        "trainable_params_unique_storage": 1234,
        "model": {"embed_dim": 64, "num_layers": 4},
    }
    provenance = assert_control_matches_live_architecture(
        {"model": dict(pin["model"])}, pin=pin,
    )
    assert "CALLER-SUPPLIED pin" in provenance, (
        "the provenance must not claim to be the recorded pin"
    )
    assert "deadbeef" in provenance
    with pytest.raises(ControlArchitectureDrift, match="num_layers"):
        assert_control_matches_live_architecture(
            {"model": {"embed_dim": 64, "num_layers": 5}}, pin=pin,
        )


def test_the_pinned_parameter_count_is_checked_against_a_built_model() -> None:
    """⚑ REVIEW F4, second half: ``model=`` had no caller either, so the pinned
    61,444,448 gated nothing on any path.

    A control whose ``model:`` section matches the pin EXACTLY, built by code
    that produces a different net, must still be refused — that is the only
    check that can see "same yaml keys, different builder", which is precisely
    the state this branch is in relative to the live tree.
    """
    pin: dict[str, Any] = {
        "recorded": "2020-01-01", "live_branch": "toy", "live_commit": "deadbeef",
        "trainable_params_unique_storage": 105,
        "model": {"embed_dim": 64},
    }
    right = torch.nn.Linear(10, 10)  # 100 weights + 5 would be 105; this is 110
    assert unique_storage_param_count(right) == 110
    with pytest.raises(ControlArchitectureDrift, match="trainable params"):
        assert_control_matches_live_architecture(
            {"model": dict(pin["model"])}, model=right, pin=pin,
        )
    pin["trainable_params_unique_storage"] = 110
    assert assert_control_matches_live_architecture(
        {"model": dict(pin["model"])}, model=right, pin=pin,
    )


def test_a_stale_pin_is_REPORTED_rather_than_silently_disabling_the_count(
    tmp_path: Path,
) -> None:
    """⚑⚑ THE CROSSCHECK USED TO SWITCH ITSELF OFF IN THE DRIFT CASE.

    ``expected_params`` is ``None`` for exactly one reason in live-file mode:
    the live ``model:`` section is not the one the pin measured. That is the
    STALE-PIN state, and it is the one state where the count is the only
    instrument left — ``model_section_drift`` compares the CONTROL against the
    LIVE file, so it is silent whenever the two moved together. The guard
    therefore returned a clean provenance string for a pin describing a net the
    tree no longer builds (PR #438 review, finding 1).

    The setup is that exact state: control == live, both away from the pin.
    """
    pin: dict[str, Any] = {
        "recorded": "2020-01-01", "live_branch": "toy", "live_commit": "deadbeef",
        "trainable_params_unique_storage": 110,
        "model": {"embed_dim": 64},
    }
    live = tmp_path / "live.yaml"
    live.write_text(yaml.safe_dump({"model": {"embed_dim": 128}}), encoding="utf-8")
    control: dict[str, Any] = {"model": {"embed_dim": 128}}
    model = torch.nn.Linear(10, 10)

  # The precondition, asserted rather than assumed: key-level drift is EMPTY,
  # so nothing but the count could have caught this.
    reference = model_section(yaml.safe_load(live.read_text(encoding="utf-8")))
    assert model_section_drift(model_section(control), reference) == []

    with pytest.raises(
        ControlArchitectureDrift, match="parameter crosscheck COULD NOT RUN",
    ) as excinfo:
        assert_control_matches_live_architecture(
            control, model=model, live_config=live, pin=pin,
        )
  # It must name the key that went stale, not just complain.
    assert "embed_dim" in str(excinfo.value)

  # ⚑ AND IT MUST STILL PASS WHEN THE PIN IS FRESH — otherwise this test would
  # be satisfied by a guard that refuses everything, and the fix would be a
  # regression rather than a repair.
    fresh: dict[str, Any] = dict(pin, model={"embed_dim": 128})
    assert assert_control_matches_live_architecture(
        control, model=model, live_config=live, pin=fresh,
    )


def test_an_empty_reference_model_section_is_refused_not_agreed_with(
    tmp_path: Path,
) -> None:
    """⚑ REVIEW F7. A live yaml with no ``model:`` section made both sides
    ``{}``, the drift list empty, and — in live-file mode — the parameter
    crosscheck off as well. The guard agreed with a file that said nothing."""
    empty = tmp_path / "no_model.yaml"
    empty.write_text(yaml.safe_dump({"train": {"lr": 0.0003}}), encoding="utf-8")
    with pytest.raises(ControlArchitectureDrift, match="reference architecture is EMPTY"):
        assert_control_matches_live_architecture(
            _raw(CONTROL), live_config=empty,
        )


def test_the_provenance_says_when_the_live_file_was_not_read() -> None:
    """"Judged against the pin" and "judged against production" are different
    claims and must not print the same string."""
    provenance = assert_control_matches_live_architecture(
        {"model": dict(LIVE_ARCH_PIN["model"])},
    )
    assert "THE LIVE FILE WAS NOT READ" in provenance
    assert "LIVE file" not in provenance
    assert model_section_drift(
        model_section({"model": dict(LIVE_ARCH_PIN["model"])}),
        LIVE_ARCH_PIN["model"],
    ) == []


def test_a_missing_live_config_path_is_an_error_not_a_silent_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """⚑ A typo'd env var must not quietly downgrade to the weaker instrument."""
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(tmp_path / "nope.yaml"))
    with pytest.raises(ControlArchitectureDrift, match="does not name a readable"):
        live_production_config_path()


def test_unique_storage_count_does_not_double_count_a_tied_tensor() -> None:
    """⚑ CLAUDE.md: 16 `layer_smolgens.N.gen_weight.weight` keys are ONE tensor.

    `sum(v.numel() for v in state_dict().values())` reads 77,173,088 on the
    live net against a true 61,444,448. This builds the same trap in miniature
    so the counter's dedup is tested rather than asserted.
    """
    shared = torch.nn.Parameter(torch.zeros(10, 10))
    model = torch.nn.Module()
    for index in range(4):
        block = torch.nn.Module()
        block.register_parameter("weight", shared)
        model.add_module(f"block{index}", block)
    model.register_parameter("own", torch.nn.Parameter(torch.zeros(5)))

    state_sum = sum(v.numel() for v in model.state_dict().values())
    assert state_sum == 4 * 100 + 5, "the trap must actually be present"
    assert unique_storage_param_count(model) == 105


def test_the_pin_matches_the_measurement_it_claims() -> None:
    """Internal consistency: the pin's two counts must be the ones recorded.

    Not the re-measurement — that is
    ``test_this_tree_builds_the_pinned_live_architecture``, which became
    possible on 2026-08-16 — but a cheap independent stop on a hand-edit that
    changes one number and not the other, since the tied-tensor identity below
    only holds for the pair actually measured.
    """
    assert LIVE_ARCH_PIN["trainable_params_unique_storage"] == 61_444_448
    assert LIVE_ARCH_PIN["state_dict_numel_sum"] == 77_173_088
    tied = (
        LIVE_ARCH_PIN["state_dict_numel_sum"]
        - LIVE_ARCH_PIN["trainable_params_unique_storage"]
    )
  # 15 extra copies of the one shared 1024x1024 smolgen generator.
    assert tied == 15 * 1_048_576
