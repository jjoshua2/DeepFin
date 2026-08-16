"""The control must build the architecture PRODUCTION IS RUNNING.

⚑⚑ `tests/test_lc0_control_config.py` pins the control's diff against the
IN-TREE `configs/pbt2_small.yaml`. That is the right design and it is
structurally blind: the live run reads the yaml in the LIVE working tree, on
the live branch, and that file moves independently of whatever this branch
carries. Measured 2026-08-16 — in-tree 63,084,128 trainable params, live
61,444,448, and `aux_policy_head_dim` is not in this tree's schema at all, so
this tree's code cannot even BUILD the live network.

So this file tests the OTHER instrument: the pin, the drift check, and the
fact that the check prefers the live file over any in-tree copy. Every test
here is written so that the situation improving BREAKS it — the pin going
stale, or bt4heads reaching `main` — because a stale pin that keeps passing is
the same blindness one level up.
"""
from __future__ import annotations

import copy
from pathlib import Path

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


def test_the_pin_records_an_architecture_this_tree_cannot_build() -> None:
    """⚑ THIS TEST IS SUPPOSED TO FAIL WHEN bt4heads REACHES `main`.

    ``aux_policy_head_dim`` is not in ``utils/config_yaml.py``'s schema here,
    and CLAUDE.md category (a) says an unknown key is FATAL AT LAUNCH:
    ``flatten_run_config_defaults`` raises before ``run.py``'s parser exists.
    That is why "re-point the control at the live model: section" is not a
    one-line config edit — it needs the promotion on this branch first.

    When it starts failing, regenerate ``LIVE_ARCH_PIN`` with
    ``scripts/lc0_control_arch_pin.py`` and re-point the control config. Do not
    delete the assertion.
    """
    live_like = copy.deepcopy(_raw(PRODUCTION))
    live_like["model"] = dict(LIVE_ARCH_PIN["model"])
    with pytest.raises(ValueError, match="aux_policy_head_dim"):
        flatten_run_config_defaults(live_like)


def test_the_pin_names_the_bt4heads_keys_the_in_tree_config_lacks() -> None:
    in_tree = model_section(_raw(PRODUCTION))
    for key in LIVE_ONLY_MODEL_KEYS:
        assert key in LIVE_ARCH_PIN["model"], f"the pin lost {key}"
        assert key not in in_tree, (
            f"{key} is now in the in-tree production config — the arm can "
            "follow production; regenerate the pin and re-point the control"
        )


def test_the_control_config_is_currently_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """⚑⚑ THE FINDING ITSELF, AS A GATE. The arm as shipped would train
    `main`'s 63,084,128-param net, not production's 61,444,448-param one."""
    monkeypatch.delenv(LIVE_CONFIG_ENV, raising=False)
    with pytest.raises(ControlArchitectureDrift) as excinfo:
        assert_control_matches_live_architecture(_raw(CONTROL))
    message = str(excinfo.value)
    for key in LIVE_ONLY_MODEL_KEYS:
        assert key in message, f"the drift report must name {key}"
    assert "may be quoted as 'our stack'" in message
    assert "recorded pin" in message, "the message must name what it judged against"


def test_the_check_passes_when_the_model_section_matches_the_reference() -> None:
    """The guard must be capable of PASSING, or it is a constant, not a check."""
    matching = {"model": dict(LIVE_ARCH_PIN["model"])}
    provenance = assert_control_matches_live_architecture(matching, live_config=None)
    assert "recorded pin" in provenance


def test_a_control_matching_the_stale_pin_still_fails_against_the_live_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setenv(LIVE_CONFIG_ENV, str(live))
    with pytest.raises(ControlArchitectureDrift, match="embed_dim"):
        assert_control_matches_live_architecture({"model": dict(LIVE_ARCH_PIN["model"])})


def test_the_provenance_says_when_the_live_file_was_not_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """"Judged against the pin" and "judged against production" are different
    claims and must not print the same string."""
    monkeypatch.delenv(LIVE_CONFIG_ENV, raising=False)
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

    Not a re-measurement — this tree cannot build the live net — but it stops a
    hand-edit that changes one number and not the other.
    """
    assert LIVE_ARCH_PIN["trainable_params_unique_storage"] == 61_444_448
    assert LIVE_ARCH_PIN["state_dict_numel_sum"] == 77_173_088
    tied = (
        LIVE_ARCH_PIN["state_dict_numel_sum"]
        - LIVE_ARCH_PIN["trainable_params_unique_storage"]
    )
  # 15 extra copies of the one shared 1024x1024 smolgen generator.
    assert tied == 15 * 1_048_576
