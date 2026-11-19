"""Real Tiny frozen-anchor/union/optimizer/resume integration; fixtures are not authority."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
from typing import Any

import pytest
import torch

from chess_anti_engine.policy_anchor_spec import CHECKPOINT_KEY as REFERENCE_KEY, DIRECTION, MASK, SCHEMA
from scripts import core_finite_learner as core
from scripts import core_selfplay_union_policy_anchor_stored_row_once as entry
from scripts import core_selfplay_union_policy_anchor_stored_row_once_proof as proof
from scripts import core_selfplay_union_stored_row_once as baseline
from scripts.bootstrap_checkpoint_resume import require_equal
from tests.test_core_finite_learner import frozen
from tests.test_core_selfplay_union_stored_row_once import contract as base_contract


def selected_method(tmp_path: Path, c: dict[str, Any], name: str) -> dict[str, str]:
    _, old = core.read_ref(c["method"])
    value = copy.deepcopy(old)
    value["training"]["closed_segments_updates"] = c["updates_per_segment"]
    value["training"]["expected_optimizer_updates"] = sum(c["updates_per_segment"])
    value["training"]["policy_anchor_coefficient"] = c["policy_anchor"]["coefficient"]
    value["training"]["policy_anchor"] = c["policy_anchor"]
    return frozen(tmp_path / (name + ".method.json"), value)


def contract(tmp_path: Path, *, coefficient: float = 1.0, split: bool = False) -> tuple[dict[str, Any], dict[str, str]]:
    c, _ = base_contract(tmp_path, split=split)
    c["schema"] = entry.CONTRACT
    c["source_files"] = {name: core.file_sha(core.REPO / name) for name in entry.SOURCE_FILES}
    c["policy_anchor"] = {"schema": SCHEMA, "direction": DIRECTION, "mask": MASK,
        "coefficient": coefficient, "reference": None if coefficient == 0 else {
            "checkpoint": c["initial_model"], "architecture": c["architecture"],
            "checkpoint_step": c["initial_model_step"], "historical_source_revision": None}}
    c["method"] = selected_method(tmp_path, c, "anchor")
    return c, frozen(tmp_path / "anchor.contract.json", c)


def stage(tmp_path: Path, cref: dict[str, str], c: dict[str, Any], segment: int,
          parent: dict[str, Any] | None, name: str) -> dict[str, str]:
    return frozen(tmp_path / (name + ".stage.json"), {"schema": entry.STAGE, "contract_sha256": cref["sha256"],
        "segment": segment, "population": c["population"], "parent": parent, "output": str(tmp_path / name)})


def closed(tmp_path: Path, cref: dict[str, str], value: dict[str, Any], *, observed: bool = True) -> dict[str, Any]:
    module = "scripts.core_selfplay_union_policy_anchor_stored_row_once_proof" if observed else "scripts.core_selfplay_union_policy_anchor_stored_row_once"
    command = [sys.executable, "-B", "-m", module, "--contract", cref["path"], "--contract-sha256", cref["sha256"],
        "--stage", value["stage"]["path"], "--stage-sha256", value["stage"]["sha256"]]
    terminal = frozen(tmp_path / (Path(value["checkpoint"]["path"]).parent.name + ".terminal.json"), {
        "status": "PASS", "returncode": 0, "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED",
        "supervisor_sha256": "f" * 64, "command": command})
    path = Path(value["checkpoint"]["path"]).parent / "boundary.json"
    return {"boundary": {"path": str(path), "sha256": core.file_sha(path)}, "owner_terminal": terminal,
        "owner_sha256": "f" * 64, "command": command}


def test_absent_baseline_and_zero_null_anchor_do_not_construct_teacher_and_match_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from chess_anti_engine.train.frozen_policy_anchor import FrozenPolicyAnchor

    c, cref = contract(tmp_path, coefficient=0)
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("zero/absent anchor constructed a teacher")
    monkeypatch.setattr(FrozenPolicyAnchor, "__init__", forbidden)
    zero = proof.execute(cref, stage(tmp_path, cref, c, 1, None, "zero"))
    ordinary = copy.deepcopy(c)
    ordinary["schema"] = baseline.CONTRACT
    ordinary.pop("policy_anchor")
    ordinary["sampling"]["quota_ledger"] = str(tmp_path / "ordinary.quota.jsonl")
    _, method = core.read_ref(c["method"])
    method["training"].pop("policy_anchor")
    ordinary["method"] = frozen(tmp_path / "ordinary.method.json", method)
    ordinary_ref = frozen(tmp_path / "ordinary.contract.json", ordinary)
    from tests.test_core_selfplay_union_stored_row_once import stage as ordinary_stage

    plain = baseline.execute(ordinary_ref, ordinary_stage(tmp_path, ordinary_ref, ordinary, 1, None, "plain"))
    a = torch.load(zero["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(plain["checkpoint"]["path"], map_location="cpu", weights_only=True)
    for key in ("model", "opt", "scheduler", "zclip", "step", "fresh_rl_donor_provenance"):
        require_equal(a[key], b[key], "zero versus unchanged global baseline " + key)
    assert zero["rng"] == plain["rng"]
    assert zero["quota_state"] == plain["quota_state"]
    assert zero["own_live_certificate"]["policy_anchor"]["actual_reference_forwards"] == 0
    assert zero["own_live_certificate"]["policy_anchor"]["initial_reference"] is None
    assert a[REFERENCE_KEY] == c["policy_anchor"]
    assert REFERENCE_KEY not in b


def test_positive_anchor_reaches_real_optimizer_and_observes_fixed_teacher_without_extra_forwards(tmp_path: Path) -> None:
    c, cref = contract(tmp_path, coefficient=10)
    observed = proof.execute(cref, stage(tmp_path, cref, c, 1, None, "positive"))
    cert = observed["own_live_certificate"]
    anchor = cert["policy_anchor"]
    assert core.canonical(anchor["identity"]) == core.canonical(c["policy_anchor"])
    assert anchor["actual_reference_forwards"] == cert["actual_calls"]["forward"] == 3
    assert anchor["diagnostics"]["eligible_rows"] == anchor["diagnostics"]["total_rows"] == 5
    assert anchor["diagnostics"]["kl_row_sum"] > 0
    assert anchor["diagnostics"]["weighted_row_sum"] == 10 * anchor["diagnostics"]["kl_row_sum"]
    assert core.digest(anchor["initial_reference"]) == anchor["final_reference_sha256"]
    assert anchor["teacher_unchanged"] is True
    assert anchor["no_optimizer_ownership"] is True
    assert anchor["no_reference_gradients"] is True
    assert anchor["all_reference_modules_eval"] is True
    assert cert["save_before_equals_after"] is True
    assert cert["extra_forwards"] == 0
    assert observed["quota_state"]["last_batch_rows"] == 1
    result = entry.validate_closed_model_delivery(cref, closed(tmp_path, cref, observed))
    assert core.canonical(result["policy_anchor"]) == core.canonical(c["policy_anchor"])
    zero = copy.deepcopy(c)
    zero["policy_anchor"].update(coefficient=0, reference=None)
    zero["sampling"]["quota_ledger"] = str(tmp_path / "zero.quota.jsonl")
    zero["method"] = selected_method(tmp_path, zero, "zero")
    zero_ref = frozen(tmp_path / "zero.contract.json", zero)
    control = entry.execute(zero_ref, stage(tmp_path, zero_ref, zero, 1, None, "control"))
    a = torch.load(observed["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(control["checkpoint"]["path"], map_location="cpu", weights_only=True)
    assert any(not torch.equal(value, b["model"][key]) for key, value in a["model"].items())
    assert observed["rng"] == control["rng"]
    assert observed["quota_state"] == control["quota_state"]


def test_positive_anchor_full_vs_closed_resume_preserves_optimizer_rng_rows_and_reference(tmp_path: Path) -> None:
    c, cref = contract(tmp_path, split=True)
    first = proof.execute(cref, stage(tmp_path, cref, c, 1, None, "first"))
    second = proof.execute(cref, stage(tmp_path, cref, c, 2, closed(tmp_path, cref, first), "second"))
    full_c = copy.deepcopy(c)
    full_c["updates_per_segment"] = [3]
    full_c["sampling"]["quota_ledger"] = str(tmp_path / "full.quota.jsonl")
    full_c["method"] = selected_method(tmp_path, full_c, "full")
    full_ref = frozen(tmp_path / "full.contract.json", full_c)
    full = proof.execute(full_ref, stage(tmp_path, full_ref, full_c, 1, None, "full"))
    a = torch.load(second["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(full["checkpoint"]["path"], map_location="cpu", weights_only=True)
    for key in ("model", "opt", "scheduler", "zclip", "step", REFERENCE_KEY):
        require_equal(a[key], b[key], "anchored full vs adjacent resume " + key)
    assert second["rng"] == full["rng"]
    assert second["quota_state"] == full["quota_state"]
    assert second["quota_ledger_checkpoint"]["sha256"] == full["quota_ledger_checkpoint"]["sha256"]
    assert core.digest(first["own_live_certificate"]["policy_anchor"]["initial_reference"]) == second["own_live_certificate"]["policy_anchor"]["final_reference_sha256"]
    assert second["own_live_certificate"]["actual_calls"]["optimizer_post"] == 2
    assert second["own_live_certificate"]["policy_anchor"]["actual_reference_forwards"] == 2


@pytest.mark.parametrize("fault", ["negative", "nonfinite", "zero_reference", "wrong_reference", "wrong_step", "method"])
def test_anchor_reference_and_method_tampering_refused_before_student_constructor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    c, _ = contract(tmp_path)
    if fault == "negative":
        c["policy_anchor"]["coefficient"] = -1
    elif fault == "nonfinite":
        c["policy_anchor"]["coefficient"] = float("inf")
    elif fault == "zero_reference":
        c["policy_anchor"]["coefficient"] = 0
    elif fault == "wrong_reference":
        c["policy_anchor"]["reference"]["checkpoint"] = {**c["initial_model"], "sha256": "0" * 64}
    elif fault == "wrong_step":
        c["policy_anchor"]["reference"]["checkpoint_step"] += 1
    else:
        _, method = core.read_ref(c["method"])
        method["training"]["policy_anchor_coefficient"] += 1
        c["method"] = frozen(tmp_path / "bad.method.json", method)
    calls: list[bool] = []
    monkeypatch.setattr(core, "make_trainer", lambda *_args: calls.append(True))
    if fault == "nonfinite":
        with pytest.raises(ValueError, match="invalid frozen policy anchor coefficient"):
            entry._validate_contract_metadata(c)
    else:
        cref = frozen(tmp_path / "bad.contract.json", c)
        expected = {"negative": "invalid frozen policy anchor coefficient", "zero_reference": "zero anchor requires null reference",
            "wrong_reference": "selected model-only bootstrap reference", "wrong_step": "selected model-only bootstrap reference",
            "method": "actual contract differs from caller-pinned method"}[fault]
        with pytest.raises(ValueError, match=expected):
            entry.execute(cref, stage(tmp_path, cref, c, 1, None, "refused"))
    assert not calls
    assert not (tmp_path / "refused").exists()


def test_saved_anchor_identity_tamper_cannot_publish_observed_model(tmp_path: Path) -> None:
    c, cref = contract(tmp_path)

    class Tampered(proof.OwnLiveProof):
        def after_save(self, context: dict[str, Any]) -> None:
            super().after_save(context)
            path = context["checkpoint"]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload[REFERENCE_KEY]["coefficient"] += 1
            torch.save(payload, path)

    with pytest.raises(ValueError, match="saved payload differs"):
        entry.execute(cref, stage(tmp_path, cref, c, 1, None, "failed"), observer=Tampered())
    assert not (tmp_path / "failed/boundary.json").exists()
    assert not (tmp_path / "failed/boundary.pending.json").exists()
