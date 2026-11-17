"""Actual Tiny passive own-live proof, full resume, and saved-payload tampering."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import torch

from chess_anti_engine.model import infer_input_planes

from scripts import core_selfplay_union_stored_row_once as entry
from scripts import core_selfplay_union_stored_row_once_proof as proof
from scripts.bootstrap_checkpoint_resume import require_equal
from tests.test_core_finite_learner import frozen
from tests.test_core_selfplay_union_stored_row_once import closed_parent, contract, stage


def test_actual_passive_observer_adds_no_forward_or_rng_or_training_state_effect(tmp_path: Path) -> None:
    c, cref = contract(tmp_path, split=False)
    observed = proof.execute(cref, stage(tmp_path, cref, c, 1, None, "observed"))
    certificate = observed["own_live_certificate"]
    assert certificate["schema"] == proof.SCHEMA
    assert certificate["status"] == proof.STATUS
    assert certificate["actual_calls"] == {"forward": 3, "optimizer_pre": 3, "optimizer_post": 3, "save": 1}
    assert certificate["step"] == 3
    assert certificate["actual_views"] == certificate["cumulative_views"] == 5
    assert certificate["extra_forwards"] == 0
    assert certificate["additional_final_CPU_checkpoint_decodes"] == 1
    assert certificate["saved_model"]["state_entries"] > 0
    assert certificate["saved_model"]["input_planes"] == infer_input_planes(c["architecture"]["input_extra_features"])
    assert certificate["saved_model"]["unique_parameter_numel"] > 0
    assert certificate["initial"]["model"]["key_shape_dtype_sha256"] == certificate["saved_model"]["key_shape_dtype_sha256"]
    assert certificate["save_before_equals_after"] is True
    assert certificate["checkpoint"] == observed["checkpoint"]
    direct = copy.deepcopy(c)
    direct["sampling"]["quota_ledger"] = str(tmp_path / "direct-quota.jsonl")
    direct_ref = frozen(tmp_path / "direct.contract.json", direct)
    bare = entry.execute(direct_ref, stage(tmp_path, direct_ref, direct, 1, None, "bare"))
    a = torch.load(observed["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(bare["checkpoint"]["path"], map_location="cpu", weights_only=True)
    for key in ("model", "opt", "scheduler", "zclip", "step", "fresh_rl_donor_provenance"):
        require_equal(a[key], b[key], "passive union proof " + key)
    assert observed["rng"] == bare["rng"]
    assert observed["quota_state"] == bare["quota_state"]
    assert "own_live_certificate" not in bare


def test_actual_passive_proof_uses_cumulative_optimizer_slots_after_closed_resume(tmp_path: Path) -> None:
    c, cref = contract(tmp_path)
    first = proof.execute(cref, stage(tmp_path, cref, c, 1, None, "first"))
    second = proof.execute(cref, stage(tmp_path, cref, c, 2, closed_parent(tmp_path, cref, first, proof=True), "second"))
    certificate = second["own_live_certificate"]
    assert certificate["step_start"] == 1
    assert certificate["step"] == 3
    assert certificate["applied_updates"] == 2
    assert certificate["actual_views"] == 3
    assert certificate["cumulative_views"] == 5
    assert certificate["actual_calls"]["optimizer_post"] == 2
    assert certificate["active_optimizer_slots"] > 0
    assert certificate["saved_model"]["key_shape_dtype_sha256"] == first["own_live_certificate"]["saved_model"]["key_shape_dtype_sha256"]
    delivery = entry.validate_closed_model_delivery(cref, closed_parent(tmp_path, cref, second, proof=True))
    assert delivery["pass_complete"] is True
    assert delivery["own_live_certificate"] == certificate


@pytest.mark.parametrize("fault", ["model", "optimizer_step", "family", "architecture"])
def test_real_final_saved_payload_tamper_refuses_closed_delivery(tmp_path: Path, fault: str) -> None:
    c, cref = contract(tmp_path, split=False)

    class Tampered(proof.OwnLiveProof):
        def after_save(self, context: dict[str, Any]) -> None:
            super().after_save(context)
            path = context["checkpoint"]
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if fault == "model":
                next(iter(payload["model"].values())).add_(1)
            elif fault == "optimizer_step":
                next(iter(payload["opt"]["state"].values()))["step"] += 1
            elif fault == "architecture":
                payload["arch"]["history_rep_fix"] = not payload["arch"]["history_rep_fix"]
            else:
                payload["fresh_rl_donor_provenance"]["sha256"] = "0" * 64
            torch.save(payload, path)

    with pytest.raises(ValueError, match="saved payload differs"):
        entry.execute(cref, stage(tmp_path, cref, c, 1, None, "failed"), observer=Tampered())
    assert not (tmp_path / "failed/boundary.json").exists()
    assert not (tmp_path / "failed/boundary.pending.json").exists()
