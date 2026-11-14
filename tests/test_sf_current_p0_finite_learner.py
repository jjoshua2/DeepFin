"""Small serialized authority fictions and actual CPU Trainer/loss path controls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.train import trainer as trainer_module
from scripts import core_finite_learner as core
from scripts import sf_current_p0_finite_learner as finite
from scripts import sf_current_p0_sidecar as carrier
from scripts import sf_current_p0_sidecar_schema as schema
from scripts.bootstrap_checkpoint_resume import require_equal
from tests import test_core_finite_learner as old


def replace(ref: dict[str, str], value: Any) -> dict[str, str]:
    path = Path(ref["path"])
    path.chmod(0o644)
    path.write_text(core.canonical(value))
    path.chmod(0o444)
    return {"path": str(path), "sha256": core.file_sha(path)}


def read(ref: dict[str, str]) -> dict[str, Any]:
    return json.loads(Path(ref["path"]).read_bytes())


def setup(tmp_path: Path, alpha: float = .5, variant: str = "native", mirror: float = .5) -> tuple[dict[str, Any], dict[str, str], dict[str, str], dict[str, str]]:
    bindings = tmp_path / "bindings"
    bindings.mkdir()
    c, _ = old.contract(bindings)
    c["source_files"].update({name: core.file_sha(core.REPO / name) for name in finite.EXTRA_SOURCE})
    c["mirror_prob"] = mirror
    genref = old.generation(bindings, c, 1, c["donor"]["sha256"])
    gen = read(genref)
    gen["qualification_scope"] = "SYNTHETIC_NOT_REAL_AUTHORITY"
    genref = replace(genref, gen)
    legacyref = old.frozen(bindings / "legacy-contract.json", c)
    new = {**c, "schema": finite.CONTRACT, "run_uuid": "22222222-2222-4222-8222-222222222222",
           "updates_per_segment": c["updates_per_generation"], "sf_p0": {
               "schema": finite.PROFILE, "scope": "SYNTHETIC_ONLY", "alpha": alpha, "value_variant": variant,
               "data_identity": {"run_uuid": c["run_uuid"], "source_revision": c["source_revision"]},
               "generation_manifest": genref,
               "limits": {"metadata_bytes": 8 << 20, "game_decoded_bytes": 8 << 20, "resident_bytes": 8 << 20}}}
    del new["updates_per_generation"]
    newref = old.frozen(bindings / "sf-contract.json", new)
    return new, newref, legacyref, genref


def accepted_carrier(tmp_path: Path, c: dict[str, Any], genref: dict[str, str]) -> dict[str, Any]:
    """Known fake three-row root/validation documents; never real authority."""
    from chess_anti_engine.replay.shard import load_shard_arrays

    b = tmp_path / "bindings"
    gen = read(genref)
    game = gen["games"][0]
    source, _ = load_shard_arrays(game["shard"]["path"], lazy=False, validate=True)
    n = game["rows"]
    root = tmp_path / "carrier"
    root.mkdir()
    folder = root / "game-000000"
    folder.mkdir()
    arrays = schema.arrays_for_rows(n)
    arrays["legal_mask_u8"][:] = source["legal_mask"]
    arrays["stored_record_wdl_f16"][1] = [.15, .25, .6]
    arrays["stored_query_wdl_f16"][1] = [.6, .25, .15]
    arrays["native_wdl_permille_i16"][1] = [700, 200, 100]
    for key in ("qualified_stored_wdl", "qualified_native_wdl", "observed_predecessor_meta", "observed_predecessor_pv"):
        arrays[key][1] = 1
    arrays["predecessor_meta_i32"][1] = [50, 2, 7, 0, 700, 200]
    arrays["predecessor_pv_i16"][1, 0] = [5, 7, 0, 700, 200]
    rows = []
    for i, status in enumerate(("UNAVAILABLE", "SOURCE_CONTEXT_REDUCED_LABEL", "INVALID_REDUCED_LABEL")):
        rows.append({"physical_row_id": game["physical_game_id"] + ":" + str(i),
                     "physical_game_id": game["physical_game_id"], "game_id": game["game_id"], "original_row_index": i,
                     "input_identity": {"ply": i, "input_F16_sha256": schema.array_digest(source["x"][i])},
                     "legal_mask_sha256": schema.array_digest(source["legal_mask"][i]),
                     "source_join": {"status": status},
                     "qualified_flags": {"native_wdl": i == 1, "stored_wdl": i == 1, "policy": False}})
    old.frozen(folder / "rows.json", {"original_shard": game["shard"], "producer_proof": game["producer_proof"],
               "physical_game_id": game["physical_game_id"], "game_id": game["game_id"], "rows": n, "rows_detail": rows})
    for name, array in arrays.items():
        np.save(folder / (name + ".npy"), array, allow_pickle=False)
    files = [carrier.entry(root, p) for p in sorted(folder.iterdir())]
    runtime = {"owner": {"sha256": "b" * 64}, "owner_seconds": 180}
    produce = {"schema": carrier.PROTOCOL, "phase": "produce", "scope": "SYNTHETIC_ONLY", "runtime": runtime,
               "source": {"revision": c["source_revision"], "files": c["source_files"]},
               "selection": {"path": str(b / "fiction-selection.json"), "sha256": "d" * 64},
               "census_readout": {"path": str(b / "fiction-census.json"), "sha256": "e" * 64},
               "output_root": str(root), "limits": {"metadata_bytes": 8 << 20, "carrier_total_bytes": 8 << 20, "inventory_files": 100},
               "candidate": None, "producer_owner": None}

    def owner(phase: str, protocol: dict[str, Any]) -> dict[str, Any]:
        pref = old.frozen(b / (phase + ".json"), protocol)
        aref = old.frozen(b / (phase + "-admission.json"), {"status": "ROOT_ADMITS_CURRENT_P0_SF_SIDECAR_PHASE_V1",
            "protocol": pref, "phase": phase, "scope": protocol["scope"], "source_revision": c["source_revision"],
            "source_files_sha256": core.digest(c["source_files"]), "runtime": runtime,
            "selection": protocol["selection"], "census_readout": protocol["census_readout"]})
        command = ["python", "-m", "scripts.sf_current_p0_sidecar", "--protocol", pref["path"], "--protocol-sha256",
                   pref["sha256"], "--admission", aref["path"], "--admission-sha256", aref["sha256"]]
        terminal = old.frozen(b / (phase + "-terminal.json"), {"status": "PASS", "returncode": 0, "resource": "cpu",
            "seconds": 180, "command": command, "supervisor_sha256": "b" * 64, "owned_processes_closed": True,
            "custody": "OWNED_PROCESSES_CLOSED", "known_processes": {"999999991": 1}})
        return {"protocol": pref, "admission": aref, "terminal": terminal, "command": command, "owner_sha256": "b" * 64}

    p = owner("produce", produce)
    candidate = old.frozen(root / "manifest.json", {"schema": schema.SCHEMA, "status": "CANDIDATE_CURRENT_P0_SF_SIDECAR_V1",
        "training_admitted": False, "root_postclosure_acceptance_required": True, "scope": "SYNTHETIC_ONLY",
        "source": produce["source"], "producer_protocol": p["protocol"], "execution_admission": p["admission"],
        "generation_manifest": genref, "rows": n, "physical_games": 1,
        "games": [{"physical_game_id": game["physical_game_id"], "rows": n, "directory": folder.name}],
        "files": files, "bytes": sum(f["bytes"] for f in files), "counts": {"rows": n}})
    core.seal(root)
    validation_p = {**produce, "phase": "validate", "candidate": candidate,
                    "producer_owner": {k: p[k] for k in ("terminal", "command", "owner_sha256")}}
    v = owner("validate", validation_p)
    validation = old.frozen(b / "validation.json", {"status": "VALIDATION_CANDIDATE_CURRENT_P0_SF_SIDECAR_V1",
        "candidate": candidate, "protocol": v["protocol"], "execution_admission": v["admission"],
        "source_payload_and_vector_bytes_recomputed": True, "rows": n, "physical_games": 1})
    post = old.frozen(b / "post-closure.json", {"qualification_scope": "SYNTHETIC_NOT_REAL_AUTHORITY"})
    root_receipt = old.frozen(b / "root-carrier.json", {"status": "ROOT_ACCEPTS_CLOSED_CURRENT_P0_SF_SIDECAR_DATA_ONLY",
        "qualification_scope": "SYNTHETIC_NOT_REAL_AUTHORITY", "scope": "SYNTHETIC_ONLY", "training_admitted": False,
        "generation_manifest": genref, "candidate": candidate, "validation": validation,
        "source": produce["source"], "counts": {"rows": n}, "rows": n, "physical_games": 1,
        "produce": {k: p[k] for k in ("protocol", "admission", "command", "terminal")},
        "validate": {k: v[k] for k in ("protocol", "admission", "command", "terminal")}, "post_closure": post})
    review = old.frozen(b / "independent.json", {"status": "ACCEPT_CLOSED_CURRENT_P0_SF_SIDECAR_DATA_ONLY_NO_FINDINGS",
        "findings": [], "candidate": candidate, "validation": validation,
        "scope": "SYNTHETIC_ONLY", "root_acceptance": root_receipt})
    return {"root_acceptance": root_receipt, "independent_review": review}


def stage(tmp_path: Path, cref: dict[str, str], gen: dict[str, str], sidecar: dict[str, Any] | None,
          *, segment: int = 1, parent: Any = None) -> dict[str, str]:
    return old.frozen(tmp_path / "bindings" / f"stage-{segment}.json", {"schema": finite.STAGE,
        "contract_sha256": cref["sha256"], "segment": segment, "generations": [gen], "parent": parent,
        "sf_p0_sidecars": None if sidecar is None else [sidecar], "output": str(tmp_path / f"segment-{segment}")})


def test_zero_actual_engine_matches_legacy_without_any_sidecar_io_or_wrapper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _c, cref, legacyref, gen = setup(tmp_path, alpha=0)
    legacy_stage = old.stage(tmp_path, legacyref, [gen], None, "legacy")
    baseline = core.execute(legacyref, legacy_stage)
    monkeypatch.setattr(finite.P0Profile, "prepare", lambda *_: pytest.fail("zero opened sidecar authority"))
    monkeypatch.setattr(finite.P0Profile, "wrap", lambda *_: pytest.fail("zero copied/wrapped replay"))
    monkeypatch.setattr(schema, "load_npy", lambda *_: pytest.fail("zero opened sidecar arrays"))
    actual = finite.execute(cref, stage(tmp_path, cref, gen, None))
    assert actual["rng"] == baseline["rng"]
    assert actual["draws_sha256"] == baseline["draws_sha256"]
    assert actual["membership"] == baseline["membership"]
    assert actual["sf_p0_teacher_mass"] == 0
    assert actual["sf_p0_outcome_mass"] == actual["drawn_rows"] == 4
    a = torch.load(actual["checkpoint"]["path"], map_location="cpu", weights_only=False)
    b = torch.load(baseline["checkpoint"]["path"], map_location="cpu", weights_only=False)
    for key in ("model", "opt", "scheduler"):
        require_equal(a[key], b[key], key)
    poison_ref = stage(tmp_path, cref, gen, None, segment=2, parent={})
    poisoned = read(poison_ref)
    poisoned["sf_p0_sidecars"] = [{"path": "/nonexistent/poison", "sha256": "f" * 64}]
    with pytest.raises(ValueError, match="zero SF-P0 must bypass sidecars"):
        finite.execute(cref, replace(poison_ref, poisoned))


@pytest.mark.parametrize("alpha", [.5, 1.0])
@pytest.mark.parametrize("variant", ["native", "stored"])
@pytest.mark.parametrize("mirror", [0.0, 1.0])
def test_actual_worker_host_loss_preserves_scale_masks_fallback_and_single_draw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                           alpha: float, variant: str, mirror: float) -> None:
    c, cref, _, gen = setup(tmp_path, alpha, variant, mirror)
    accepted = accepted_carrier(tmp_path, c, gen)
    calls = []
    original = trainer_module.compute_loss

    def observed(outputs: Any, batch: Any, **kwargs: Any) -> Any:
        loss = original(outputs, batch, **kwargs)
        core.require(kwargs["sf_wdl_frac"] == alpha and kwargs["search_wdl_frac"] == 0, "wrong active blend")
        assert kwargs["w_policy"] == kwargs["w_wdl"] == 1
        assert all(value == 0 for name, value in kwargs.items() if name.startswith("w_") and name not in {"w_policy", "w_wdl"})
        assert kwargs["sf_policy_floor"].w == kwargs["sf_shape"].w == 0
        target = torch.nn.functional.one_hot(batch["wdl_t"], 3).float()
        teacher = batch["sf_wdl"].float()
        teacher = teacher / teacher.sum(-1, keepdim=True)
        mask = batch["has_sf_wdl"].float().unsqueeze(1)
        expected = (1 - alpha) * target + alpha * (mask * teacher + (1 - mask) * target)
        expected_ce = -(expected * torch.log_softmax(outputs["wdl"].float(), -1)).sum(-1).mean()
        torch.testing.assert_close(loss["blended_wdl_ce"], expected_ce)
        torch.testing.assert_close(loss["total"], loss["policy_ce"] + loss["blended_wdl_ce"])
        assert "sf_multipv_raw" not in batch  # Existing host health-before-prune route.
        torch.testing.assert_close(batch["sf_eval_pv_checked"], batch["has_sf_wdl"].float())
        assert torch.count_nonzero(batch["sf_eval_pv_orphan"]) == 0
        calls.append({key: batch[key].detach().cpu().numpy().copy() for key in ("wdl_t", "has_sf_wdl", "sf_wdl", "policy_t")})
        return loss

    monkeypatch.setattr(trainer_module, "compute_loss", observed)
    result = finite.execute(cref, stage(tmp_path, cref, gen, accepted))
    journal = [json.loads(line) for line in (tmp_path / "segment-1/draws.jsonl").read_text().splitlines()]
    assert len(journal) == len(calls) == 2
    expected_rng = np.random.default_rng(c["seed"])
    qualified = 0
    for draw, batch in zip(journal, calls, strict=True):
        indices = [row["row_index"] for row in draw["rows"]]
        np.testing.assert_array_equal(indices, expected_rng.choice(3, size=2, replace=True))
        if mirror > 0:
            expected_rng.random(2)  # One existing mirror mask; no overlay RNG.
        np.testing.assert_array_equal(batch["wdl_t"], indices)
        np.testing.assert_array_equal(batch["has_sf_wdl"], np.array(indices) == 1)
        qualified += indices.count(1)
        for index, row in enumerate(indices):
            expected = ([.7, .2, .1] if variant == "native" else np.array([.6, .25, .15], dtype=np.float16).astype(np.float32)) if row == 1 else np.eye(3)[row]
            np.testing.assert_allclose(batch["sf_wdl"][index], expected, atol=1e-7)
    assert result["sf_p0_sample_counts"]["rows"] == result["drawn_rows"] == 4
    assert 0 < qualified < 4  # A real teacher row AND outcome fallback reached loss.
    assert result["sf_p0_teacher_mass"] == qualified * alpha
    assert result["sf_p0_outcome_mass"] + result["sf_p0_teacher_mass"] == 4
    assert result["segment"] == 1
    assert "generation" not in result
    assert result["rng"]["numpy"] == expected_rng.bit_generator.state


@pytest.mark.parametrize("fault", ["candidate_only", "pending", "review", "failed_owner", "foreign_generation"])
def test_unaccepted_sidecars_refuse_before_trainer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    c, cref, _, gen = setup(tmp_path)
    accepted = accepted_carrier(tmp_path, c, gen)
    value = read(accepted["root_acceptance"])
    expected_exception: type[Exception] = ValueError
    expected_message = "not root-postclosure accepted SF-P0 carrier"
    if fault == "candidate_only":
        accepted["root_acceptance"] = value["candidate"]
    elif fault == "pending":
        value["status"] = "PENDING"
        accepted["root_acceptance"] = replace(accepted["root_acceptance"], value)
    elif fault == "review":
        accepted["independent_review"] = {"path": str(tmp_path / "missing-review"), "sha256": "0" * 64}
        expected_exception, expected_message = FileNotFoundError, "missing-review"
    elif fault == "failed_owner":
        term = read(value["produce"]["terminal"])
        term["status"] = "FAILED"
        value["produce"]["terminal"] = replace(value["produce"]["terminal"], term)
        accepted["root_acceptance"] = replace(accepted["root_acceptance"], value)
        review = read(accepted["independent_review"])
        review["root_acceptance"] = accepted["root_acceptance"]
        accepted["independent_review"] = replace(accepted["independent_review"], review)
        expected_message = "not exact closed CPU owner"
    else:
        value["generation_manifest"] = {**gen, "sha256": "0" * 64}
        accepted["root_acceptance"] = replace(accepted["root_acceptance"], value)
    monkeypatch.setattr(core, "make_trainer", lambda *_a, **_k: pytest.fail("constructed model before authority refusal"))
    with pytest.raises(expected_exception, match=expected_message):
        finite.execute(cref, stage(tmp_path, cref, gen, accepted))
    assert not (tmp_path / "segment-1").exists()


def test_families_and_unknown_profile_fields_remain_strict(tmp_path: Path) -> None:
    c, _, legacyref, gen = setup(tmp_path, alpha=0)
    with pytest.raises(ValueError, match="unknown or missing contract fields"):
        core.validate_contract(c)
    legacy = read(legacyref)
    s = {"schema": "core_finite_stage_v1"}
    with pytest.raises(ValueError, match="not the SF-P0 learning family"):
        finite.P0Profile().validate_and_project(legacy, s)
    p = copy.deepcopy(c["sf_p0"])
    p["policy_alpha"] = .2
    with pytest.raises(ValueError, match="profile fields/schema"):
        finite.validate_profile(p)
    p = copy.deepcopy(c["sf_p0"])
    p.update(scope="FIXED_ORIGINAL_G1", generation_manifest=gen)
    with pytest.raises(ValueError, match="fixed original G1 identity"):
        finite.validate_profile(p)


def parent_binding(tmp_path: Path, cref: dict[str, str], segment: int) -> dict[str, Any]:
    path = tmp_path / f"segment-{segment}" / "boundary.json"
    boundary = json.loads(path.read_bytes())
    command = ["python", "-B", "-m", finite.MODULE, "--contract", cref["path"], "--contract-sha256", cref["sha256"],
               "--stage", boundary["stage"]["path"], "--stage-sha256", boundary["stage"]["sha256"]]
    terminal = old.frozen(tmp_path / "bindings" / f"learner-terminal-{segment}.json", {"status": "PASS", "returncode": 0,
        "resource": "cpu", "seconds": 300, "supervisor_sha256": "b" * 64, "command": command,
        "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED", "known_processes": {"999999992": 1}})
    return {"boundary": {"path": str(path), "sha256": core.file_sha(path)}, "owner_terminal": terminal,
            "owner_sha256": "b" * 64, "command": command}


def test_same_family_full_resume_fixed_data_and_legacy_parent_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, cref, legacyref, gen = setup(tmp_path, alpha=.5)
    accepted = accepted_carrier(tmp_path, c, gen)
    first = finite.execute(cref, stage(tmp_path, cref, gen, accepted))
    checkpoint = torch.load(first["checkpoint"]["path"], map_location="cpu", weights_only=False)
    parent = parent_binding(tmp_path, cref, 1)
    second_ref = stage(tmp_path, cref, gen, accepted, segment=2, parent=parent)
    original_make = core.make_trainer

    def checked_make(*args: Any, **kwargs: Any) -> Any:
        trainer = original_make(*args, **kwargs)
        original_train = trainer.train_steps

        def checked_train(replay: Any, **training: Any) -> Any:
            from scripts.offline_epoch_resume import _rng_state
            assert trainer.step == first["step"]
            require_equal(trainer.model.state_dict(), checkpoint["model"], "restored model")
            require_equal(trainer.opt.state_dict(), checkpoint["opt"], "restored optimizer")
            require_equal(trainer._scheduler.state_dict(), checkpoint["scheduler"], "restored scheduler")
            assert _rng_state(replay.rng, np.random) == first["rng"]
            return original_train(replay, **training)

        trainer.train_steps = checked_train
        return trainer

    monkeypatch.setattr(core, "make_trainer", checked_make)
    second = finite.execute(cref, second_ref)
    monkeypatch.setattr(core, "make_trainer", original_make)
    assert first["step"] == 2
    assert second["step"] == second["final_step"] == 4
    assert second["segment"] == 2
    assert first["generation_manifests"] == second["generation_manifests"] == [gen]
    assert first["sf_p0_profile"] == second["sf_p0_profile"] == c["sf_p0"]
    assert second["parent"] == parent
    assert {g for g, _, _ in second["membership"]} == {1}
    assert second["rng"]["numpy"] != first["rng"]["numpy"]
    with pytest.raises(ValueError, match="parent original stage mismatch"):
        core.closed_parent(parent, c, legacyref, {"generation": 2, "generations": [gen, gen]})
    legacy = read(legacyref)
    core.execute(legacyref, old.stage(tmp_path, legacyref, [gen], None, "legacy"))
    legacy_parent = old.parent_binding(tmp_path, legacyref, "legacy")
    _, projected = finite.P0Profile().validate_and_project(c, read(second_ref))
    with pytest.raises(ValueError, match="same-family contract/profile/data parent mismatch"):
        finite.P0Profile().closed_parent(legacy_parent, legacy, cref, projected)


@pytest.mark.parametrize("fault", ["profile", "command", "failure_marker"])
def test_same_family_parent_drift_refuses(tmp_path: Path, fault: str) -> None:
    c, cref, _, gen = setup(tmp_path, alpha=0)
    finite.execute(cref, stage(tmp_path, cref, gen, None))
    parent = parent_binding(tmp_path, cref, 1)
    second_ref = stage(tmp_path, cref, gen, None, segment=2, parent=parent)
    if fault == "profile":
        c = copy.deepcopy(c)
        c["sf_p0"]["value_variant"] = "stored"
    elif fault == "command":
        parent["command"][-9] = "scripts.core_finite_learner"
    else:
        core.failure_marker(tmp_path / "segment-1").write_text("{}")
    engine = finite.P0Profile()
    c, s = engine.validate_and_project(c, read(second_ref))
    with pytest.raises(ValueError, match=r"parent publication failed|same-family|parent exact command"):
        engine.closed_parent(parent, c, cref, s)


def test_real_draw_wrapper_preserves_rng_and_source_and_rejects_foreign_rows(tmp_path: Path) -> None:
    c, cref, _, gen = setup(tmp_path)
    accepted = accepted_carrier(tmp_path, c, gen)
    sref = stage(tmp_path, cref, gen, accepted)
    profile = finite.P0Profile()
    c, s = profile.validate_and_project(c, read(sref))
    generations, _ = core.metadata(c, s, cref, _sf_p0=profile)
    profile.prepare(c, s, generations)
    seen = []

    def on_draw(rows: Any) -> None:
        seen.append(rows)
        profile.on_draw(rows)

    seed = 78
    replay = core.load_replay(c, generations, np.random.default_rng(seed), on_draw)
    baseline = core.load_replay(c, generations, np.random.default_rng(seed), None)
    wrapper = profile.wrap(replay, c)
    expected = baseline.sample_batch_arrays(6)
    batch = wrapper.sample_batch_arrays(6)
    assert len(seen) == 1
    assert replay.rng.bit_generator.state == baseline.rng.bit_generator.state
    assert wrapper.rng is replay.rng
    for key in ("x", "policy_target", "wdl_target", "game_id", "ply_index", "legal_mask"):
        np.testing.assert_array_equal(batch[key], expected[key])
    source = next(iter(profile.index.values()))
    assert not source.arrays["native_wdl_permille_i16"].flags.writeable
    from chess_anti_engine.replay.generation import DrawnRow
    with pytest.raises(ValueError, match="foreign SF-P0 physical draw"):
        profile.overlay(expected, [DrawnRow(1, "0" * 64, 0)] * 6)
    changed = dict(expected)
    changed["game_id"] = np.full(6, 9000)
    with pytest.raises(ValueError, match="sampled game/ply identity"):
        profile.overlay(changed, seen[0])


def test_sidecar_resident_cap_refuses_before_npy_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, cref, _, gen = setup(tmp_path)
    accepted = accepted_carrier(tmp_path, c, gen)
    profile = finite.P0Profile()
    c, s = profile.validate_and_project(c, read(stage(tmp_path, cref, gen, accepted)))
    generations, _ = core.metadata(c, s, cref, _sf_p0=profile)
    profile.prepare(c, s, generations)
    replay = core.load_replay(c, generations, np.random.default_rng(1), profile.on_draw)
    c["sf_p0"]["limits"]["resident_bytes"] = 1
    monkeypatch.setattr(schema, "load_npy", lambda *_: pytest.fail("loaded before resident cap"))
    with pytest.raises(ValueError, match="pre-load decoded byte cap"):
        profile.wrap(replay, c)


def test_forced_ids_cover_both_fallback_classes_in_actual_host_and_loss(tmp_path: Path) -> None:
    from chess_anti_engine.replay.generation import DrawnRow
    from chess_anti_engine.replay.shard import load_shard_arrays

    c, cref, _, gen = setup(tmp_path, alpha=.5, variant="native", mirror=1)
    accepted = accepted_carrier(tmp_path, c, gen)
    profile = finite.P0Profile()
    c, s = profile.validate_and_project(c, read(stage(tmp_path, cref, gen, accepted)))
    generations, _ = core.metadata(c, s, cref, _sf_p0=profile)
    profile.prepare(c, s, generations)
    replay = core.load_replay(c, generations, np.random.default_rng(7), profile.on_draw)
    profile.wrap(replay, c)
    game = generations[0]["games"][0]
    arrays, _ = load_shard_arrays(game["shard"]["path"], lazy=False, validate=True)
    original = {key: value.tobytes() for key, value in arrays.items()}
    source = core.digest(game["shard"]["files"])
    batch = profile.overlay(arrays, tuple(DrawnRow(1, source, i) for i in range(3)))
    assert [row["source_join"]["status"] for row in profile.index[source].rows] == [
        "UNAVAILABLE", "SOURCE_CONTEXT_REDUCED_LABEL", "INVALID_REDUCED_LABEL"]
    trainer = core.make_trainer(c, tmp_path / "forced-log", sf_p0_alpha=.5)
    try:
        host = trainer._prepare_host_arrays(batch, rng=np.random.default_rng(7), mirror_prob=1)
        tensors = trainer._host_batch_to_tensors(host)
        output = trainer.model(tensors["x"])
        losses = trainer_module.compute_loss(output, tensors, **trainer._loss_kwargs)
        np.testing.assert_array_equal(tensors["has_sf_wdl"].cpu().numpy(), [0, 1, 0])
        np.testing.assert_array_equal(tensors["wdl_t"].cpu().numpy(), [0, 1, 2])
        expected = torch.tensor([[1, 0, 0], [.35, .6, .05], [0, 0, 1]], dtype=torch.float32)
        ce = -(expected * torch.log_softmax(output["wdl"].float(), -1)).sum(-1).mean()
        torch.testing.assert_close(losses["blended_wdl_ce"], ce)
        torch.testing.assert_close(losses["total"], losses["policy_ce"] + ce)
        np.testing.assert_array_equal(tensors["sf_eval_pv_checked"].cpu().numpy(), [0, 1, 0])
        assert original == {key: value.tobytes() for key, value in arrays.items()}
        assert profile.counts["qualified"] == 1
        assert profile.counts["outcome_fallback_rows"] == 2
    finally:
        trainer.writer.close()
