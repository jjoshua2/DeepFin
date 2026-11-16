"""Actual synthetic caller/continuation and explicit anchored actor metadata."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from chess_anti_engine.model import model_config_from_manifest_dict
from chess_anti_engine.policy_anchor_spec import CHECKPOINT_KEY as REFERENCE_KEY
from chess_anti_engine.train import frozen_policy_anchor as runtime
from scripts import core_finite_collect as collect
from scripts import core_finite_learner as core
from scripts import core_generation_bank_learner as bank_learner
from scripts import core_generation_bank_policy_anchor as anchored
from scripts.bootstrap_checkpoint_resume import require_equal
from scripts.offline_epoch_resume import _restore_rng, _rng_state
from tests.test_core_finite_learner import frozen
from tests.test_core_generation_bank_learner import bank, contract, parent
from tests.test_frozen_policy_anchor import disabled, reference


def anchored_contract(tmp_path: Path, *, zero: bool = False) -> tuple[dict[str, Any], dict[str, str]]:
    c, _ = contract(tmp_path)
    c.update(schema=anchored.CONTRACT, policy_anchor=disabled() if zero else reference(tmp_path, model_config_from_manifest_dict(c["architecture"])))
    c["source_files"] = {name: core.file_sha(core.REPO / name) for name in anchored.SOURCE_FILES}
    return c, frozen(tmp_path / "anchored-contract.json", c)


def stage(tmp_path: Path, cref: dict[str, str], banks: list[dict[str, Any]], prior: dict[str, Any] | None, name: str) -> dict[str, str]:
    return frozen(tmp_path / (name + "-stage.json"), {"schema": anchored.STAGE, "contract_sha256": cref["sha256"],
        "segment": len(banks), "banks": banks, "parent": prior, "output": str(tmp_path / name)})


def closed_parent(tmp_path: Path, cref: dict[str, str], boundary: dict[str, Any]) -> dict[str, Any]:
    result = parent(tmp_path, cref, boundary)
    result["command"][3] = "scripts.core_generation_bank_policy_anchor"
    _, terminal = core.read_ref(result["owner_terminal"])
    terminal["command"] = result["command"]
    result["owner_terminal"] = frozen(tmp_path / "anchor-owner-terminal.json", terminal)
    return result


def test_actual_public_cli_saves_distinct_family_reference_and_diagnostics(tmp_path: Path):
    c, cref = anchored_contract(tmp_path)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000)
    sref = stage(tmp_path, cref, [binding], None, "first")
    assert anchored.main(["--contract", cref["path"], "--contract-sha256", cref["sha256"], "--stage", sref["path"], "--stage-sha256", sref["sha256"]]) == 0
    _, boundary = core.read_ref({"path": str(tmp_path / "first/boundary.json"), "sha256": core.file_sha(tmp_path / "first/boundary.json")})
    assert boundary["status"] == anchored.BOUNDARY
    assert boundary["logical_generation"] == 2
    assert boundary["segment"] == 1
    assert boundary["initial_actor_step"] == 9000
    assert boundary["step_start"] == 0
    assert boundary["step"] == boundary["applied_updates"] == 2
    assert boundary["policy_anchor"] == c["policy_anchor"]
    assert boundary["policy_anchor_diagnostics"]["reference_forwards"] == 2
    assert boundary["policy_anchor_diagnostics"]["eligible_rows"] == 4
    assert boundary["policy_anchor_diagnostics"]["mean_kl"] > 0
    saved = torch.load(boundary["checkpoint"]["path"], map_location="cpu", weights_only=True)
    assert saved[anchored.CHECKPOINT_KEY] == bank_learner.checkpoint_identity(c, cref, 1, _anchored=True)
    assert saved[REFERENCE_KEY] == c["policy_anchor"]
    assert bank_learner.CHECKPOINT_KEY not in saved
    assert set(saved["model"]) == set(torch.load(c["initial_model"]["path"], map_location="cpu", weights_only=True)["model"])


def test_zero_public_caller_matches_real_unanchored_state_and_draws(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    c, cref = anchored_contract(tmp_path, zero=True)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000)
    baseline_c = {key: value for key, value in c.items() if key != "policy_anchor"}
    baseline_c["schema"] = bank_learner.CONTRACT
    baseline_ref = frozen(tmp_path / "baseline-contract.json", baseline_c)
    baseline_stage = frozen(tmp_path / "baseline-stage.json", {"schema": bank_learner.STAGE, "contract_sha256": baseline_ref["sha256"],
        "segment": 1, "banks": [binding], "parent": None, "output": str(tmp_path / "baseline")})
    baseline = bank_learner.execute(baseline_ref, baseline_stage)
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("zero public caller touched a reference")
    monkeypatch.setattr(runtime, "FrozenPolicyAnchor", forbidden)
    monkeypatch.setattr(runtime, "_checkpoint", forbidden)
    actual = anchored.execute(cref, stage(tmp_path, cref, [binding], None, "zero"))
    a = torch.load(baseline["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(actual["checkpoint"]["path"], map_location="cpu", weights_only=True)
    for key in ("model", "opt", "scheduler", "zclip", "step", "peak_lr", "fresh_rl_donor_provenance"):
        require_equal(a[key], b[key], "zero public caller " + key)
    assert actual["rng"] == baseline["rng"]
    assert actual["draws_sha256"] == baseline["draws_sha256"]
    assert actual["membership"] == baseline["membership"]
    assert actual["policy_anchor_diagnostics"] is None
    assert b[REFERENCE_KEY] == disabled()
    assert REFERENCE_KEY not in a
    for key in ("loss", "policy_loss", "wdl_loss", "train_steps_done", "train_samples_seen"):
        assert actual["metrics"][key] == baseline["metrics"][key]


def test_adjacent_anchored_resume_matches_real_direct_full_state_continuation(tmp_path: Path):
    c, cref = anchored_contract(tmp_path)
    first_bank = bank(tmp_path, c, 2, c["initial_model"], 9000)
    first = anchored.execute(cref, stage(tmp_path, cref, [first_bank], None, "first"))
    second_bank = bank(tmp_path, c, 3, first["checkpoint"], first["step"])
    prior = closed_parent(tmp_path, cref, first)
    second_stage = stage(tmp_path, cref, [first_bank, second_bank], prior, "second")
    _, s = core.read_ref(second_stage)
    generations, _, _ = bank_learner.metadata(c, s, cref, _anchored=True)
    expected_trainer = core.make_trainer(bank_learner.settings(c), tmp_path / "oracle")
    expected_trainer.load(Path(first["checkpoint"]["path"]), exact_resume=True)
    expected_trainer.configure_frozen_policy_anchor(c["policy_anchor"])
    rng = np.random.default_rng()
    replay = core.load_replay(bank_learner.settings(c), generations, rng, None, first_generation=2)
    _restore_rng(first["rng"], rng, cast(Any, np.random))
    expected_trainer.train_steps(replay, batch_size=2, steps=2)
    expected = {"model": copy.deepcopy(expected_trainer.model.state_dict()), "opt": copy.deepcopy(expected_trainer.opt.state_dict()),
        "scheduler": copy.deepcopy(expected_trainer._scheduler.state_dict()), "zclip": copy.deepcopy(expected_trainer.zclip_state_dict())}
    rng_expected = _rng_state(rng, cast(Any, np.random))
    expected_trainer.writer.close()
    second = anchored.execute(cref, second_stage)
    saved = torch.load(second["checkpoint"]["path"], map_location="cpu", weights_only=True)
    for key, value in expected.items():
        require_equal(saved[key], value, "anchored full continuation " + key)
    assert second["rng"] == rng_expected
    assert second["step_start"] == 2
    assert second["step"] == 4
    assert second["segment"] == 2
    assert second["logical_generation"] == 3
    assert saved[REFERENCE_KEY] == c["policy_anchor"]


@pytest.mark.parametrize("fault", ["old_entry", "new_entry", "unknown", "stage"])
def test_crossed_family_schemas_and_unknown_fields_refuse(tmp_path: Path, fault: str):
    c, cref = anchored_contract(tmp_path, zero=True)
    if fault == "old_entry":
        sref = stage(tmp_path, cref, [], None, "old-refused")
        with pytest.raises(ValueError, match="contract"):
            bank_learner.execute(cref, sref)
        return
    if fault == "new_entry":
        c["schema"] = bank_learner.CONTRACT
        del c["policy_anchor"]
    elif fault == "unknown":
        c["allow_anchor_alias"] = True
    cref = frozen(tmp_path / "crossed-contract.json", c)
    s = {"schema": bank_learner.STAGE, "contract_sha256": cref["sha256"], "segment": 1, "banks": [], "parent": None, "output": str(tmp_path / "bad")}
    with pytest.raises(ValueError, match=r"contract|stage"):
        anchored.execute(cref, frozen(tmp_path / "crossed-stage.json", s))


def test_changed_reference_contract_cannot_resume_closed_anchored_parent(tmp_path: Path):
    c, cref = anchored_contract(tmp_path)
    first_bank = bank(tmp_path, c, 2, c["initial_model"], 9000)
    first = anchored.execute(cref, stage(tmp_path, cref, [first_bank], None, "first"))
    second_bank = bank(tmp_path, c, 3, first["checkpoint"], first["step"])
    prior = closed_parent(tmp_path, cref, first)
    changed = copy.deepcopy(c)
    changed["policy_anchor"]["coefficient"] = 0.4
    new_ref = frozen(tmp_path / "changed-contract.json", changed)
    with pytest.raises(ValueError, match="parent"):
        anchored.execute(new_ref, stage(tmp_path, new_ref, [first_bank, second_bank], prior, "refused"))
    assert not (tmp_path / "refused/learner.pt").exists()


def test_actual_new_family_parent_projects_to_adjacent_collector_without_reference_io(tmp_path: Path):
    c, cref = anchored_contract(tmp_path)
    binding = bank(tmp_path, c, 2, c["initial_model"], 9000)
    boundary = anchored.execute(cref, stage(tmp_path, cref, [binding], None, "first"))
    prior = closed_parent(tmp_path, cref, boundary)
    identity = bank_learner.checkpoint_identity(c, cref, 1, _anchored=True)
    actor = {"kind": collect.ANCHOR_BANK_KIND, **boundary["checkpoint"], "expected_checkpoint_step": 2,
        "architecture": c["architecture"], "predecessor": prior,
        "lineage": {"schema": collect.ANCHOR_BANK_LINEAGE_SCHEMA, **identity}}
    # Removing the real test reference proves serving-lineage checks never reopen
    # it. The student's real closed checkpoint and family identity remain intact.
    Path(c["policy_anchor"]["reference"]["checkpoint"]["path"]).unlink()
    result = collect.closed_bank_predecessor({"generation": 3, "actor": actor})
    assert result["logical_generation"] == 2
    assert collect.bank_lineage_identity(actor) == identity
    actor["kind"] = "bank_learner"
    with pytest.raises(ValueError, match="lineage"):
        collect.closed_bank_predecessor({"generation": 3, "actor": actor})
