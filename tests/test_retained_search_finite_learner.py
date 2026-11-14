"""Synthetic admissions only; real finite worker/Trainer/loss effect witnesses."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.replay.shard import load_shard_arrays, save_local_shard_arrays
from chess_anti_engine.train import trainer as trainer_module
from scripts import core_finite_learner as core
from scripts import retained_search_finite_learner as finite
from scripts.bootstrap_checkpoint_resume import require_equal
from tests import test_core_finite_learner as old


def read(ref: dict[str, str]) -> dict[str, Any]:
    return json.loads(Path(ref["path"]).read_bytes())


def setup(tmp_path: Path, share: float = 1, mirror: float = .5, *, stored: bool = True) -> tuple[dict[str, Any], dict[str, str], dict[str, str], dict[str, str]]:
    bindings = tmp_path / "bindings"
    bindings.mkdir()
    c, _ = old.contract(bindings)
    c["mirror_prob"] = mirror
    c["source_files"].update({name: core.file_sha(core.REPO / name) for name in finite.EXTRA_SOURCE})
    genref = old.generation(bindings, c, 1, c["donor"]["sha256"])
    gen = read(genref)
    game = gen["games"][0]
    if stored:
        arrays, _ = load_shard_arrays(game["shard"]["path"], lazy=False, validate=True)
        arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
        arrays["search_wdl"] = np.array([[.8, .1, .1], [.6, .25, .15], [.1, .2, .7]], dtype=np.float16)
        arrays["has_search_wdl"] = np.array([0, 1, 0], dtype=np.uint8)
        shard = tmp_path / "stored-search.zarr"
        save_local_shard_arrays(shard, arrs=arrays, meta={"games": 1, "positions": 3})
        core.seal(shard)
        files = [{"path": str(p.relative_to(shard)), "bytes": p.stat().st_size, "sha256": core.file_sha(p)}
                 for p in sorted(shard.rglob("*")) if p.is_file()]
        game["shard"] = {"path": str(shard), "files": files}
        proof = read(game["producer_proof"])
        proof["shard_inventory_sha256"] = core.digest(files)
        game["producer_proof"] = old.frozen(bindings / "stored-search-proof.json", proof)
    gen["qualification_scope"] = "SYNTHETIC_NOT_REAL_AUTHORITY"
    genref = old.frozen(bindings / "stored-search-generation.json", gen)
    legacy = old.frozen(bindings / "legacy.json", c)
    c = {**c, "schema": finite.CONTRACT, "run_uuid": "22222222-2222-4222-8222-222222222222",
         "updates_per_segment": c["updates_per_generation"], "retained_search": {
             "schema": finite.PROFILE, "scope": "SYNTHETIC_ONLY", "search_share": share,
             "data_identity": {"run_uuid": c["run_uuid"], "source_revision": c["source_revision"]},
             "generation_manifest": genref}}
    del c["updates_per_generation"]
    return c, old.frozen(bindings / "search-contract.json", c), legacy, genref


def stage(tmp_path: Path, cref: dict[str, str], genref: dict[str, str], segment: int = 1,
          parent: Any = None, label: str | None = None) -> dict[str, str]:
    label = label or f"segment-{segment}"
    return old.frozen(tmp_path / "bindings" / f"{label}.json", {"schema": finite.STAGE,
        "contract_sha256": cref["sha256"], "segment": segment, "generations": [genref],
        "parent": parent, "output": str(tmp_path / label)})


def parent_binding(tmp_path: Path, cref: dict[str, str], result: dict[str, Any]) -> dict[str, Any]:
    command = [sys.executable, "-B", "-m", finite.MODULE, "--contract", cref["path"],
               "--contract-sha256", cref["sha256"], "--stage", result["stage"]["path"],
               "--stage-sha256", result["stage"]["sha256"]]
    terminal = old.frozen(tmp_path / "bindings" / "closed-parent.json", {
        "status": "PASS", "returncode": 0, "owned_processes_closed": True,
        "custody": "OWNED_PROCESSES_CLOSED", "supervisor_sha256": "b" * 64, "command": command,
        "qualification_scope": "SYNTHETIC_NOT_REAL_AUTHORITY"})
    path = Path(result["checkpoint"]["path"]).parent / "boundary.json"
    return {"boundary": {"path": str(path), "sha256": core.file_sha(path)},
            "owner_terminal": terminal, "owner_sha256": "b" * 64, "command": command}


def test_zero_public_worker_matches_legacy_state_rng_and_unwrapped_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, cref, legacy, gen = setup(tmp_path, share=0)
    baseline = core.execute(legacy, old.stage(tmp_path, legacy, [gen], None, "legacy"))
    monkeypatch.setattr(finite.SearchProfile, "validate_arrays", lambda *_: pytest.fail("zero inspected search arrays"))
    actual = finite.execute(cref, stage(tmp_path, cref, gen))
    assert actual["rng"] == baseline["rng"]
    assert actual["draws_sha256"] == baseline["draws_sha256"]
    assert actual["membership"] == baseline["membership"]
    assert actual["retained_search_target_mass"] == 0
    assert actual["genuine_outcome_target_mass"] == actual["drawn_rows"] == 4
    a = torch.load(actual["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(baseline["checkpoint"]["path"], map_location="cpu", weights_only=True)
    assert a.pop(finite.CHECKPOINT_KEY) == finite.checkpoint_identity(c, cref)
    require_equal(a, b, "zero all saved state except explicit family metadata")
    assert core.CORE_KWARGS["search_wdl_frac"] == core.CORE_KWARGS["sf_wdl_frac"] == 0
    legacy_parent = old.parent_binding(tmp_path, legacy, "legacy")
    with pytest.raises(ValueError, match="same-family contract/profile/data parent mismatch"):
        finite.execute(cref, stage(tmp_path, cref, gen, 2, legacy_parent, "foreign-parent"))
    search_parent = parent_binding(tmp_path, cref, actual)
    with pytest.raises(ValueError, match="parent original stage mismatch"):
        core.closed_parent(search_parent, read(legacy), legacy,
                           {"generation": 2, "generations": [gen, gen]})


@pytest.mark.parametrize("mirror", [0.0, 1.0])
def test_actual_worker_host_loss_has_search_ce_gradient_and_single_draw_effect(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                            mirror: float) -> None:
    c, cref, _, gen = setup(tmp_path, mirror=mirror)
    original = trainer_module.compute_loss
    calls: list[dict[str, Any]] = []

    def observed(outputs: Any, batch: Any, **kwargs: Any) -> Any:
        assert kwargs["search_wdl_frac"] == 1
        assert kwargs["sf_wdl_frac"] == 0
        assert kwargs["w_policy"] == kwargs["w_wdl"] == 1
        assert all(v == 0 for k, v in kwargs.items() if k.startswith("w_") and k not in {"w_policy", "w_wdl"})
        assert kwargs["sf_policy_floor"].w == kwargs["sf_shape"].w == 0
        actual = original(outputs, batch, **kwargs)
        raw = batch["search_wdl"].float()
        normalized = raw / raw.sum(-1, keepdim=True)
        onehot = torch.nn.functional.one_hot(batch["wdl_t"], 3).float()
        mask = batch["has_search_wdl"].bool().unsqueeze(1)
        target = torch.where(mask, normalized, onehot)
        expected_ce = -(target * torch.log_softmax(outputs["wdl"].float(), -1)).sum(-1).mean()
        torch.testing.assert_close(actual["blended_wdl_ce"], expected_ce)
        torch.testing.assert_close(actual["total"], actual["policy_ce"] + expected_ce)
        grad = torch.autograd.grad(actual["blended_wdl_ce"], outputs["wdl"], retain_graph=True)[0]
        expected_grad = (torch.softmax(outputs["wdl"].float(), -1) - target) / target.shape[0]
        torch.testing.assert_close(grad, expected_grad)
        control_kwargs: dict[str, Any] = {**kwargs, "search_wdl_frac": 0.0}
        control = original(outputs, batch, **control_kwargs)
        control_grad = torch.autograd.grad(control["blended_wdl_ce"], outputs["wdl"], retain_graph=True)[0]
        if bool(mask.any()):
            assert not torch.equal(actual["blended_wdl_ce"], control["blended_wdl_ce"])
            assert not torch.equal(grad, control_grad)
        calls.append({key: batch[key].detach().cpu().numpy().copy()
                      for key in ("search_wdl", "has_search_wdl", "wdl_t", "policy_t")})
        return actual

    monkeypatch.setattr(trainer_module, "compute_loss", observed)
    result = finite.execute(cref, stage(tmp_path, cref, gen))
    draws = [json.loads(line) for line in (tmp_path / "segment-1/draws.jsonl").read_text().splitlines()]
    expected_rng = np.random.default_rng(c["seed"])
    qualified = 0
    source_sha = core.digest(read(gen)["games"][0]["shard"]["files"])
    values = np.array([[.8, .1, .1], [.6, .25, .15], [.1, .2, .7]], dtype=np.float16).astype(np.float32)
    assert len(draws) == len(calls) == 2
    for draw, batch in zip(draws, calls, strict=True):
        indices = [row["row_index"] for row in draw["rows"]]
        np.testing.assert_array_equal(indices, expected_rng.choice(3, size=2, replace=True))
        if mirror:
            expected_rng.random(2)
        assert all(row["generation"] == 1 and row["source_sha256"] == source_sha for row in draw["rows"])
        np.testing.assert_array_equal(batch["wdl_t"], indices)
        np.testing.assert_array_equal(batch["has_search_wdl"], np.array(indices) == 1)
        np.testing.assert_array_equal(batch["search_wdl"], values[indices])  # L/R mirror never swaps W/L.
        qualified += indices.count(1)
    assert 0 < qualified < 4  # Positive search AND unavailable-outcome branches actually reach loss.
    assert result["drawn_rows"] == result["applied_views"] == 4
    assert result["retained_search_target_mass"] == pytest.approx(qualified)
    assert result["genuine_outcome_target_mass"] == pytest.approx(4 - qualified)
    assert result["rng"]["numpy"] == expected_rng.bit_generator.state


@pytest.mark.parametrize("fault", ["no_mask", "nonbinary", "nonfinite", "negative", "empty", "precision", "claimed_absent"])
def test_invalid_qualified_storage_refuses(fault: str) -> None:
    arrays: dict[str, Any] = {"search_wdl": np.array([[.6, .25, .15]], dtype=np.float16), "has_search_wdl": np.ones(1, dtype=np.uint8)}
    messages = {"no_mask": "availability mask", "nonbinary": "availability mask", "nonfinite": "invalid qualified",
                "negative": "invalid qualified", "empty": "invalid qualified", "precision": "shape/precision",
                "claimed_absent": "claimed label is absent"}
    if fault == "no_mask":
        del arrays["has_search_wdl"]
    elif fault == "nonbinary":
        arrays["has_search_wdl"][0] = 2
    elif fault == "nonfinite":
        arrays["search_wdl"][0, 0] = np.nan
    elif fault == "negative":
        arrays["search_wdl"][0] = [-.1, .5, .6]
    elif fault == "empty":
        arrays["search_wdl"][0] = 0
    elif fault == "precision":
        arrays["search_wdl"] = arrays["search_wdl"].astype(np.float32)
    else:
        del arrays["search_wdl"]
    with pytest.raises(ValueError, match=messages[fault]):
        finite.SearchProfile().validate_arrays(arrays, 1)


def test_absent_and_unclaimed_nonfinite_labels_preserve_explicit_fallback() -> None:
    profile = finite.SearchProfile()
    profile.validate_arrays({}, 1)
    profile.validate_arrays({"has_search_wdl": np.zeros(1, dtype=np.uint8)}, 1)
    profile.validate_arrays({"search_wdl": np.full((1, 3), np.nan, dtype=np.float16),
                             "has_search_wdl": np.zeros(1, dtype=np.uint8)}, 1)


@pytest.mark.parametrize("share", [False, True, -.1, .2, float("nan"), 2])
def test_only_explicit_zero_and_one_are_accepted(share: Any) -> None:
    with pytest.raises(ValueError, match="share must be zero or one"):
        finite.validate_profile({"schema": finite.PROFILE, "scope": "SYNTHETIC_ONLY", "search_share": share,
                                 "data_identity": {}, "generation_manifest": {}})


def test_old_and_new_public_contracts_refuse_cross_family(tmp_path: Path) -> None:
    c, cref, legacy, gen = setup(tmp_path)
    with pytest.raises(ValueError, match="unknown or missing contract fields"):
        core.execute(cref, stage(tmp_path, cref, gen))
    with pytest.raises(ValueError, match="not the retained-search learning family"):
        finite.execute(legacy, old.stage(tmp_path, legacy, [gen], None, "legacy"))
    poisoned = {**c, "sf_p0": {"alpha": 1}}
    combined_ref = old.frozen(tmp_path / "bindings/combined-contract.json", poisoned)
    combined_stage = stage(tmp_path, combined_ref, gen, label="combined-stage")
    with pytest.raises(ValueError, match="unknown or missing contract fields"):
        finite.execute(combined_ref, combined_stage)


def test_actual_same_family_continuation_keeps_bank_and_restores_full_state(tmp_path: Path) -> None:
    c, cref, _, gen = setup(tmp_path)
    first = finite.execute(cref, stage(tmp_path, cref, gen))
    parent = parent_binding(tmp_path, cref, first)
    second = finite.execute(cref, stage(tmp_path, cref, gen, 2, parent))
    assert first["segment"] == 1
    assert second["segment"] == 2
    assert first["step"] == 2
    assert second["step"] == 4
    assert first["membership"] == second["membership"]
    assert second["generation_manifests"] == [gen]
    assert all(row[0] == 1 for row in second["membership"])
    state = torch.load(second["checkpoint"]["path"], map_location="cpu", weights_only=True)
    assert state[finite.CHECKPOINT_KEY] == finite.checkpoint_identity(c, cref)
    assert second["retained_search_profile"] == c["retained_search"]
    changed = copy.deepcopy(c)
    changed["retained_search"]["search_share"] = 0
    newref = old.frozen(tmp_path / "bindings/changed-share.json", changed)
    with pytest.raises(ValueError, match="same-family contract/profile/data parent mismatch"):
        finite.execute(newref, stage(tmp_path, newref, gen, 2, parent, "changed"))


@pytest.mark.parametrize("fault", ["family", "profile", "missing"])
def test_checkpoint_metadata_refuses_before_optimizer_restoration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    _c, cref, _, gen = setup(tmp_path)
    first = finite.execute(cref, stage(tmp_path, cref, gen))
    path = Path(first["checkpoint"]["path"])
    state = torch.load(path, map_location="cpu", weights_only=True)
    if fault == "family":
        state[finite.CHECKPOINT_KEY]["family"] = "core_finite_learning_v1"
    elif fault == "profile":
        state[finite.CHECKPOINT_KEY]["profile"]["search_share"] = 0
    else:
        del state[finite.CHECKPOINT_KEY]
    path.chmod(0o644)
    torch.save(state, path)
    path.chmod(0o444)
    boundary_path = path.parent / "boundary.json"
    boundary = json.loads(boundary_path.read_text())
    boundary["checkpoint"]["sha256"] = boundary["final_checkpoint"]["sha256"] = core.file_sha(path)
    boundary_path.chmod(0o644)
    boundary_path.write_text(core.canonical(boundary))
    boundary_path.chmod(0o444)
    parent = parent_binding(tmp_path, cref, boundary)
    monkeypatch.setattr(trainer_module.Trainer, "load", lambda *_a, **_k: pytest.fail("restored optimizer before family check"))
    with pytest.raises(ValueError, match="checkpoint family/contract/profile mismatch"):
        finite.execute(cref, stage(tmp_path, cref, gen, 2, parent))


def test_wholly_absent_search_component_uses_outcome_in_actual_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _c, cref, legacy, gen = setup(tmp_path, stored=False)
    baseline = core.execute(legacy, old.stage(tmp_path, legacy, [gen], None, "legacy"))
    original = trainer_module.compute_loss
    seen: list[bool] = []

    def observed(outputs: Any, batch: Any, **kwargs: Any) -> Any:
        assert kwargs["search_wdl_frac"] == 1
        assert "search_wdl" not in batch
        assert "has_search_wdl" not in batch
        actual = original(outputs, batch, **kwargs)
        target = torch.nn.functional.one_hot(batch["wdl_t"], 3).float()
        expected = -(target * torch.log_softmax(outputs["wdl"].float(), -1)).sum(-1).mean()
        torch.testing.assert_close(actual["blended_wdl_ce"], expected)
        torch.testing.assert_close(actual["total"], actual["policy_ce"] + expected)
        grad = torch.autograd.grad(actual["blended_wdl_ce"], outputs["wdl"], retain_graph=True)[0]
        expected_grad = (torch.softmax(outputs["wdl"].float(), -1) - target) / target.shape[0]
        torch.testing.assert_close(grad, expected_grad)
        seen.append(True)
        return actual

    monkeypatch.setattr(trainer_module, "compute_loss", observed)
    actual = finite.execute(cref, stage(tmp_path, cref, gen))
    assert len(seen) == 2
    assert actual["retained_search_target_mass"] == 0
    assert actual["genuine_outcome_target_mass"] == actual["drawn_rows"] == 4
    assert actual["draws_sha256"] == baseline["draws_sha256"]
    assert actual["rng"] == baseline["rng"]
    a = torch.load(actual["checkpoint"]["path"], map_location="cpu", weights_only=True)
    b = torch.load(baseline["checkpoint"]["path"], map_location="cpu", weights_only=True)
    for key in ("model", "opt", "scheduler"):
        require_equal(a[key], b[key], key)
