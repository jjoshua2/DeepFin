"""Fictional software fixtures; no real study, bank or receipt authority.

Exercises the actual CPU math, strict public model loader, actual Zarr decoder,
fixed scoring body and streaming CPU reconstruction; production GPU entry remains
subject to separate source/runtime/native/model-delivery qualification.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model, infer_input_planes
from chess_anti_engine.replay.shard import save_local_shard_arrays
from scripts import core_finite_collect as collect
from scripts import core_finite_learner as core
from scripts import core_generation_holdout_metrics as metrics
from scripts import core_generation_holdout_read as reader
from scripts import core_generation_holdout_ruler as ruler
from scripts import core_generation_holdout_score as score
from tests.test_core_finite_learner import frozen

LIMITS = {"max_rows": 1000, "game_decoded_bytes": 16 << 20, "total_decoded_bytes": 64 << 20,
          "reference_cache_bytes": 8 << 20, "row_log_bytes": 4 << 20}


def tensor_case():
    policy = torch.zeros(3, 1858)
    policy[:, :2] = torch.tensor([0.25, 0.75])
    legal = torch.zeros_like(policy)
    legal[:, :2] = 1
    logits = torch.full_like(policy, 999)
    logits[:, :2] = torch.tensor([math.log(0.25), math.log(0.75)])
    wdl = torch.tensor([[2.0, 1.0, -1.0], [-1.0, 2.0, 0.0], [0.0, -1.0, 3.0]])
    return {"policy_own": logits, "wdl": wdl}, legal, policy.half(), torch.arange(3)


def test_actual_policy_ce_wdl_perspective_and_legal_forward_kl():
    outputs, legal, policy, outcome = tensor_case()
    with torch.no_grad():
        values, logs, wdl = metrics.row_metrics(outputs, legal, policy, outcome)
        second, _, _ = metrics.row_metrics(outputs, legal, policy, outcome, logs)
    expected = -torch.log_softmax(outputs["wdl"].float(), -1).diagonal()
    torch.testing.assert_close(values["wdl_ce"], expected, rtol=0, atol=0)
    torch.testing.assert_close(wdl.exp().sum(-1), torch.ones(3))
    torch.testing.assert_close(values["policy_ce"], torch.full((3,), -0.25 * math.log(0.25) - 0.75 * math.log(0.75)))
    torch.testing.assert_close(second["reference_forward_kl"], torch.zeros(3), rtol=0, atol=0)
    assert logs[:, 2:].eq(0).all()
    probability = outputs["wdl"].softmax(-1)
    torch.testing.assert_close(values["brier"], ((probability - torch.eye(3)) ** 2).sum(-1), rtol=0, atol=0)
    torch.testing.assert_close(values["score_mse"], (probability[:, 0] + 0.5 * probability[:, 1] - torch.tensor([1.0, 0.5, 0.0])) ** 2, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["missing_policy", "none_policy", "missing_wdl", "float_result", "censored", "empty_legal", "illegal_target", "nan", "wrong_width"])
def test_metric_input_refusals(fault: str):
    outputs, legal, policy, outcome = tensor_case()
    if fault == "missing_policy":
        del outputs["policy_own"]
    elif fault == "none_policy":
        outputs["policy"] = None
    elif fault == "missing_wdl":
        del outputs["wdl"]
    elif fault == "float_result":
        outcome = outcome.float()
    elif fault == "censored":
        outcome[0] = -1
    elif fault == "empty_legal":
        legal[0] = 0
    elif fault == "illegal_target":
        policy[0, 2] = 0.1
    elif fault == "nan":
        outputs["wdl"][0, 0] = torch.nan
    else:
        outputs["policy_own"] = outputs["policy_own"][:, :-1]
    with pytest.raises(ValueError):
        metrics.row_metrics(outputs, legal, policy, outcome)


def game_sums(n: int, base: float, delta: float, index: int) -> dict[str, Any]:
    models = {label: dict.fromkeys(metrics.METRICS, n * base) for label in metrics.LABELS}
    models["C"]["wdl_ce"] = n * (base + delta)
    return {"physical_game_id": "collection:" + str(index), "parent_id": "parent-" + str(index),
            "rows": n, "row_identity_sha256": "a" * 64, "models": models}


def test_equal_parent_not_pooled_rows():
    report = metrics.summarize([game_sums(1, 1, 0.1, 0), game_sums(9, 3, 0.3, 1)])
    assert report["models"]["A39"]["row_weighted"]["wdl_ce"] == 2.8
    assert report["models"]["A39"]["equal_parent"]["wdl_ce"] == 2
    assert report["parents"] == 2


@pytest.mark.parametrize(("delta", "veto"), [(0.3, True), (0.1, False), (-0.3, False), (0.0, False)])
def test_fixed_parent_bootstrap_degenerate_local_rng(delta: float, veto: bool):
    before = np.random.get_state()
    games = [game_sums(50, 1, delta, index) for index in range(48)]
    result = metrics.parent_health(games)
    interval = result["C_minus_A39_equal_parent_WDL_CE"]
    assert interval["point"] == pytest.approx(delta)
    assert interval["lower95"] == pytest.approx(delta)
    assert interval["upper95"] == pytest.approx(delta)
    assert result["bootstrap"]["degenerate"] is True
    assert result["severe_damage_veto"] is veto
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


@pytest.mark.parametrize("fault", ["few", "empty", "few_rows", "nonfinite", "shared_parent", "missing_model"])
def test_unread_incomplete_parent_estimator(fault: str):
    games = [game_sums(50, 1, 0.3, index) for index in range(48)]
    if fault == "few":
        games.pop()
    elif fault == "empty":
        games.clear()
    elif fault == "few_rows":
        games = [game_sums(1, 1, 0, index) for index in range(48)]
    elif fault == "nonfinite":
        games[0]["models"]["C"]["wdl_ce"] = math.nan
    elif fault == "shared_parent":
        games[0]["parent_id"] = games[1]["parent_id"]
    else:
        del games[0]["models"]["C"]
    with pytest.raises(ValueError):
        metrics.parent_health(games)


def synthetic_models(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = ModelConfig(kind="tiny", input_extra_features="v1", history_rep_fix=True)
    architecture = {"_schema_version": ARCH_SCHEMA_VERSION, **asdict(cfg)}
    bindings = {}
    for label in metrics.LABELS:
        model = build_model(cfg)
        path = tmp_path / (label + ".pt")
        tag = {"contract_sha256": "a" * 64, "run_uuid": "11111111-1111-4111-8111-111111111111", "source_revision": "b" * 40,
               "segment": 1, "logical_generation": 1, "policy_anchor": {"fixture": label}}
        torch.save({"model": model.state_dict(), "arch": architecture, "step": 2,
                    ruler.ANCHOR_KEY: tag, "frozen_reference_policy_anchor_v1": tag["policy_anchor"], "opt": {"do_not_restore": True}}, path)
        path.chmod(0o444)
        bindings[label] = {"checkpoint": {"path": str(path), "sha256": core.file_sha(path)}, "architecture": architecture,
                           "step": 2, "tag": tag, "model_state_sha256": collect.state_sha256(model.state_dict())}
    return bindings, architecture


def synthetic_games(tmp_path: Path, architecture: dict[str, Any], sizes=(3, 5)):
    games = []
    for index, n in enumerate(sizes):
        policy = np.zeros((n, 1858), dtype=np.float16)
        policy[:, 0] = 1
        arrays = {"x": np.ones((n, infer_input_planes(architecture["input_extra_features"]), 8, 8), dtype=np.float16),
                  "policy_target": policy, "legal_mask": np.ones_like(policy, dtype=np.uint8), "wdl_target": np.arange(n, dtype=np.int8) % 3,
                  "game_id": np.full(n, index, dtype=np.int64), "ply_index": np.arange(n, dtype=np.int32), "priority": np.ones(n, dtype=np.float32),
                  "_input_history_encoding": np.asarray(architecture["input_history_encoding"]), "_policy_encoding": np.asarray(architecture["policy_encoding"]),
                  "_history_rep_fix": np.asarray("true")}
        for key in ("has_policy", "has_legal_mask", "is_network_turn", "has_is_network_turn", "is_selfplay", "has_is_selfplay", "has_game_id", "has_ply_index"):
            arrays[key] = np.ones(n, dtype=np.uint8)
        path = tmp_path / ("fixture-" + str(index) + ".zarr")
        save_local_shard_arrays(path, arrs=arrays, meta={"games": 1, "positions": n})
        core.seal(path)
        inventory = [{"path": str(p.relative_to(path)), "bytes": p.stat().st_size, "sha256": core.file_sha(p)} for p in sorted(path.rglob("*")) if p.is_file()]
        games.append({"physical_game_id": "fiction:" + str(index), "game_id": index, "rows": n, "bank": "holdout", "label_class": "NATURAL",
                      "producer_proof": {"path": "/fiction/game-" + str(index) + ".json", "sha256": "c" * 64}, "shard": {"path": str(path), "files": inventory}})
    return SimpleNamespace(identity={"bank": "holdout", "architecture": architecture}, games=tuple(games))


def test_real_public_loader_fixed_all_row_scoring_and_streaming_reader(tmp_path: Path):
    models, architecture = synthetic_models(tmp_path)
    view = synthetic_games(tmp_path, architecture)
    games = score.load_games(view, LIMITS)
    parents = {game["physical_game_id"]: "parent-" + str(index) for index, (game, _) in enumerate(games)}
    output = tmp_path / "observations"
    output.mkdir()
    before = score.snapshot_rng()
    result = score.score_games(games, parents, models, output, device="cpu", autocast=False, limits=LIMITS)
    assert score.same_rng(before, score.snapshot_rng())
    result.update(eligible_games=len(games), eligible_rows=sum(game["rows"] for game, _ in games))
    reconstructed = reader.reconcile_rows(result, view, parents, LIMITS["row_log_bytes"])
    assert reconstructed == result["games"]
    assert result["aggregate"]["rows"] == 8
    assert all(evidence["optimizer_constructed_or_restored"] is False for evidence in result["models"].values())
    assert all(evidence["real_decodes"] == evidence["real_constructors"] == 1 for evidence in result["models"].values())
    for descriptor in result["row_logs"].values():
        rows = list(reader.read_rows(descriptor, LIMITS["row_log_bytes"]))
        assert len(rows) == 8
        assert all(len(row["wdl_log_probs"]) == 3 and set(row["metrics"]) == set(metrics.METRICS) for row in rows)
        assert not any("policy_vector" in key for row in rows for key in row)


@pytest.mark.parametrize("fault", ["family", "reference", "step", "architecture", "state"])
def test_corrupt_real_decoded_checkpoint_refused_before_real_constructor(tmp_path: Path, monkeypatch, fault: str):
    from chess_anti_engine.uci import model_loader

    models, _ = synthetic_models(tmp_path)
    binding = copy.deepcopy(models["C"])
    original = torch.load
    builds = []
    build = model_loader.build_model
    def real_load_with_fault(*args, **kwargs):
        payload = original(*args, **kwargs)
        if fault == "family":
            del payload[ruler.ANCHOR_KEY]
        elif fault == "reference":
            payload["frozen_reference_policy_anchor_v1"] = None
        elif fault == "step":
            payload["step"] += 1
        elif fault == "architecture":
            payload["arch"]["history_rep_fix"] = False
        else:
            key = next(iter(payload["model"]))
            payload["model"][key] = payload["model"][key] + 1
        return payload
    def counted(config):
        builds.append(1)
        return build(config)
    monkeypatch.setattr(torch, "load", real_load_with_fault)
    monkeypatch.setattr(model_loader, "build_model", counted)
    with pytest.raises(ValueError):
        score.load_model(binding, device="cpu")
    assert not builds


@pytest.mark.parametrize("fault", ["drop", "order", "bool_outcome", "pair", "identity", "metric", "nonfinite"])
def test_rehashed_observation_faults_refused(tmp_path: Path, fault: str):
    models, architecture = synthetic_models(tmp_path)
    view = synthetic_games(tmp_path, architecture, sizes=(2,))
    games = score.load_games(view, LIMITS)
    parents = {"fiction:0": "parent-0"}
    output = tmp_path / "observations"
    output.mkdir()
    result = score.score_games(games, parents, models, output, device="cpu", autocast=False, limits=LIMITS)
    result.update(eligible_games=1, eligible_rows=2)
    descriptor = result["row_logs"]["C"]
    path = Path(descriptor["path"])
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if fault == "drop":
        rows.pop()
    elif fault == "order":
        rows.reverse()
    elif fault == "bool_outcome":
        rows[0]["outcome"] = True
    elif fault == "pair":
        rows[0]["ply_index"] += 10
    elif fault == "identity":
        rows[0]["producer_proof"]["sha256"] = "d" * 64
    elif fault == "metric":
        rows[0]["metrics"]["wdl_ce"] += 1
    else:
        rows[0]["wdl_log_probs"][0] = math.nan
    path.chmod(0o644)
    path.write_bytes(b"".join((json.dumps(row, allow_nan=True) + "\n").encode() for row in rows))
    path.chmod(0o444)
    descriptor.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size, rows=len(rows))
    with pytest.raises(ValueError):
        reader.reconcile_rows(result, view, parents, LIMITS["row_log_bytes"])


@pytest.mark.parametrize("fault", ["game_id", "float_result", "censored", "duplicate_ply", "network_mask", "policy_mask", "empty_legal", "illegal_mass", "history", "nonfinite"])
def test_actual_decode_then_strict_hold_payload_refusal(tmp_path: Path, monkeypatch, fault: str):
    from chess_anti_engine.replay import shard

    models, architecture = synthetic_models(tmp_path)
    view = synthetic_games(tmp_path, architecture, sizes=(2,))
    real_decode = shard.load_shard_arrays
    def decoded_fault(*args, **kwargs):
        arrays, metadata = real_decode(*args, **kwargs)
        arrays = {key: value.copy() for key, value in arrays.items()}
        if fault == "game_id":
            arrays["game_id"][0] += 1
        elif fault == "float_result":
            arrays["wdl_target"] = arrays["wdl_target"].astype(np.float32)
        elif fault == "censored":
            arrays["wdl_target"][0] = -1
        elif fault == "duplicate_ply":
            arrays["ply_index"][1] = arrays["ply_index"][0]
        elif fault == "network_mask":
            arrays["is_network_turn"][0] = 0
        elif fault == "policy_mask":
            arrays["has_policy"][0] = 0
        elif fault == "empty_legal":
            arrays["legal_mask"][0] = 0
        elif fault == "illegal_mass":
            arrays["legal_mask"][0, 0] = 0
        elif fault == "history":
            arrays["_input_history_encoding"] = np.asarray("wrong")
        else:
            arrays["x"][0, 0, 0, 0] = np.nan
        return arrays, metadata
    monkeypatch.setattr(shard, "load_shard_arrays", decoded_fault)
    with pytest.raises(ValueError):
        score.load_games(view, LIMITS)
    assert models  # Actual fixture checkpoints are unrelated to data admission.


def parent_graph(tmp_path: Path, fault=""):
    collection = "11111111-1111-4111-8111-111111111111"
    design = frozen(tmp_path / "design.json", {"study_uuid": collection})
    plans = [{"physical_game_id": collection + ":" + str(index), "bank": "holdout"} for index in range(64)]
    rows = [{"physical_game_id": row["physical_game_id"], "parent_id": "parent-" + str(index),
             "parent_member_ids": ["member-" + str(index)]} for index, row in enumerate(plans)]
    if fault == "duplicate_parent":
        rows[1]["parent_id"] = rows[0]["parent_id"]
    elif fault == "shared_member":
        rows[1]["parent_member_ids"] = rows[0]["parent_member_ids"]
    elif fault == "missing_attempt":
        rows.pop()
    elif fault == "train":
        plans[0]["bank"] = "train"
    elif fault == "physical":
        rows[0]["physical_game_id"] = collection + ":64"
    roster = tmp_path / "roster.json"
    roster.write_bytes(ruler.canonical(plans))
    roster.chmod(0o444)
    pref = frozen(tmp_path / "producer.json", {"roster": ruler.reference(roster)})
    audit = frozen(tmp_path / "audit.json", {"producer_protocol": pref})
    hroot = frozen(tmp_path / "hold-root.json", {"audit_protocol": audit})
    mapping = frozen(tmp_path / "parent-map.json", {"study_uuid": collection, "collection_uuid": collection, "games": rows})
    root = {"status": ruler.ROSTER_ROOT, "study_uuid": collection, "design": design,
            "producer_protocol": pref, "producer_roster": ruler.reference(roster), "parent_map": mapping}
    rref = frozen(tmp_path / "parents-root.json", root)
    review = {**root, "status": ruler.ROSTER_REVIEW, "root_acceptance": rref, "findings": []}
    if fault == "review":
        review["findings"] = ["unresolved"]
    iref = frozen(tmp_path / "parents-ind.json", review)
    p = {"study_uuid": collection, "design": design, "parents": {"root": rref, "independent": iref}, "holdout": {"root": hroot}}
    view = SimpleNamespace(identity={"collection_uuid": collection}, games=({"physical_game_id": plans[0]["physical_game_id"]},))
    return p, view


def test_all_planned_attempt_parents_bound_without_outcome_inference(tmp_path: Path):
    p, view = parent_graph(tmp_path)
    m = ruler.Metadata(1 << 20, 8 << 20)
    mapping = ruler.parent_map(m, p, view)
    assert len(mapping) == 64
    assert mapping[view.games[0]["physical_game_id"]] == "parent-0"
    m.finish()


@pytest.mark.parametrize("fault", ["duplicate_parent", "shared_member", "missing_attempt", "train", "physical", "review"])
def test_rehashed_parent_graph_refusals(tmp_path: Path, fault: str):
    p, view = parent_graph(tmp_path, fault)
    with pytest.raises(ValueError):
        ruler.parent_map(ruler.Metadata(1 << 20, 8 << 20), p, view)


@pytest.mark.parametrize("entry", [score, reader])
def test_public_entries_refuse_train_or_legacy_schema_before_runtime(tmp_path: Path, entry):
    pref = frozen(tmp_path / "legacy.json", {"schema": "CORE_FINITE_ADMISSION_V1"})
    ad = frozen(tmp_path / "admission.json", {})
    with pytest.raises(ValueError, match="wrong ruler protocol"):
        entry.main(["--protocol", pref["path"], "--protocol-sha256", pref["sha256"], "--admission", ad["path"], "--admission-sha256", ad["sha256"]])


def test_bounded_row_log_never_emits_partial_qualified_descriptor(tmp_path: Path):
    path = tmp_path / "small.jsonl"
    with path.open("xb") as stream:
        log = score.RowLog(path, 10, stream)
        with pytest.raises(ValueError, match="byte cap"):
            log.add({"too_large": "more than ten bytes"})
        assert log.rows == 0


def test_actual_hold_auditor_readonly_acceptance_feeds_strict_decoder(tmp_path: Path, monkeypatch):
    from scripts import core_generation_holdout_admit as hold_audit
    from scripts import core_generation_holdout_acceptance as hold_view
    from tests.test_core_finite_admit import entry_case, bind_data_case
    from tests.test_core_generation_holdout import accepted_audit

    case = entry_case(tmp_path, monkeypatch, holdout=True)
    case.protocol["runtime"] = {"owner": {"path": "/fictional-owner.py", "sha256": "e" * 64}, "seconds": 60,
        "inner_seconds": 30, "cpu_affinity": [20, 22], "python": "/python"}
    pref = bind_data_case(tmp_path, case)
    assert hold_audit.main(["--protocol", pref["path"], "--protocol-sha256", pref["sha256"]]) == 0
    result = json.loads((Path(case.protocol["output_root"]) / "readout.json").read_bytes())
    root, review = accepted_audit(tmp_path, case, pref, result)
    view = hold_view.validate_accepted_holdout(root, review, max_file_bytes=1 << 20, max_total_bytes=8 << 20, max_games=3)
    with pytest.raises(ValueError, match="invalid root acceptance"):
        ruler.training_population(ruler.Metadata(1 << 20, 8 << 20), {"stage": {"banks": [{"root_acceptance": root, "independent_review": review}]}})
    loaded = score.load_games(view, LIMITS)
    assert sum(game["rows"] for game, _ in loaded) == result["eligible_rows"]
    assert all(game["bank"] == "holdout" for game, _ in loaded)
    assert all(not array.flags.writeable for _, arrays in loaded for array in arrays.values())


def owner_fixture(tmp_path: Path, suffix: list[str], resource: str, name: str):
    command = ["/fiction/python", "-B", *suffix]
    terminal = {"status": "PASS", "returncode": 0, "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED",
                "resource": resource, "seconds": 900, "wall_seconds": 1.0, "command": command,
                "source_revision": "f" * 40, "cwd": "/fiction/owner", "supervisor_sha256": "e" * 64,
                "supervisor_pid": 2147483645, "known_processes": {"2147483646": 1}}
    tref = frozen(tmp_path / (name + "-terminal.json"), terminal)
    aref = frozen(tmp_path / (name + "-admission.json"), {"command": command, "resource": resource, "seconds": 900, "owner": {"sha256": "e" * 64}})
    return {"terminal": tref, "admission": aref, "command": command, "source_revision": "f" * 40, "cwd": "/fiction/owner",
            "closure": {"supported_eof": True, "exit_code": 0, "supervisor_pid": 2147483645, "supervisor_closed": True,
                        "owned_births_closed": True, "known_processes": terminal["known_processes"]}}


def delivery_graph(tmp_path: Path, fault=""):
    from tests.test_core_generation_bank_policy_anchor import anchored_contract

    c, _ = anchored_contract(tmp_path, zero=True)
    c.update(first_generation=1, updates_per_segment=[250], batch_size=32)
    cref = frozen(tmp_path / "selected-contract.json", c)
    sref = frozen(tmp_path / "selected-stage.json", {"schema": "core_generation_bank_policy_anchor_stage_v1",
        "contract_sha256": cref["sha256"], "segment": 1, "parent": None, "banks": ["fictional accepted TRAIN refs"], "output": str(tmp_path / "learner")})
    checkpoint = {"path": str(tmp_path / "learner/learner.pt"), "sha256": "a" * 64}
    boundary = {"status": ruler.BOUNDARY, "contract_sha256": cref["sha256"], "stage": sref, "checkpoint": checkpoint,
        "source_revision": c["source_revision"], "run_uuid": c["run_uuid"], "logical_generation": 1, "segment": 1,
        "initial_model": c["initial_model"], "initial_actor_step": c["initial_actor_step"], "parent": None,
        "step_start": 0, "step": 250, "applied_updates": 250, "applied_views": 8000, "drawn_rows": 8000,
        "policy_anchor": c["policy_anchor"], "membership": [[1, "b" * 64, 8000]], "accepted_banks": ["fictional accepted TRAIN refs"],
        "bank_identities": [{"run_uuid": "44444444-4444-4444-8444-444444444444", "collection_uuid": "55555555-5555-4555-8555-555555555555",
                             "generation": 1, "source_revision": c["source_revision"], "architecture": c["architecture"],
                             "actor_sha256": c["initial_model"]["sha256"], "actor_checkpoint_step": c["initial_actor_step"]}]}
    if fault == "legacy":
        boundary["status"] = "COMPLETED_CORE_GENERATION_V1"
    elif fault == "counter":
        boundary["step"] = 249
    elif fault == "holdout":
        boundary["bank_identities"][0]["bank"] = "holdout"
    bref = frozen(tmp_path / "selected-boundary.json", boundary)
    learner_suffix = ["-m", "scripts.core_generation_bank_policy_anchor", "--contract", cref["path"], "--contract-sha256", cref["sha256"],
                      "--stage", sref["path"], "--stage-sha256", sref["sha256"]]
    learner_owner = owner_fixture(tmp_path, learner_suffix, "gpu", "learner")
    cpu_suffix = ["-m", "scripts.fictional_future_saved_state_audit", "--boundary", bref["path"]]
    audit_owner = owner_fixture(tmp_path, cpu_suffix, "cpu", "audit")
    readout = {"status": "CORE_BANK_LEARNER_SAVED_STATE_AUDIT_PENDING_ROOT_CLOSURE_V1", "contract": cref, "stage": sref,
        "boundary": bref, "checkpoint": checkpoint, "source_revision": c["source_revision"], "architecture": c["architecture"],
        "step": 250, "model_state_sha256": "c" * 64, "command_suffix": cpu_suffix,
        "reconciled": dict.fromkeys(("model", "optimizer", "scheduler", "zclip", "counters", "rng", "journal", "membership", "policy_anchor"), True)}
    if fault == "saved_rng":
        readout["reconciled"]["rng"] = False
    elif fault == "saved_checkpoint":
        readout["checkpoint"] = {"path": checkpoint["path"], "sha256": "d" * 64}
    elif fault == "closure":
        learner_owner["closure"]["owned_births_closed"] = False
    root = {"status": ruler.DELIVERY_ROOT, "contract": cref, "stage": sref, "boundary": bref, "checkpoint": checkpoint,
        "source_revision": c["source_revision"], "source_checkout": str(core.REPO), "learner_owner": learner_owner,
        "payload_audit": {"readout": frozen(tmp_path / "saved-audit.json", readout), "owner": audit_owner}}
    rref = frozen(tmp_path / "delivery-root.json", root)
    review = {"status": ruler.DELIVERY_REVIEW, "root_acceptance": rref, "findings": [],
              **{key: root[key] for key in ("contract", "stage", "boundary", "checkpoint", "payload_audit")}}
    if fault == "independent":
        review["findings"] = ["missing"]
    return {"root": rref, "independent": frozen(tmp_path / "delivery-ind.json", review)}


def test_closed_delivery_distinguishes_owner_and_learner_revisions(tmp_path: Path):
    binding = delivery_graph(tmp_path)
    m = ruler.Metadata(1 << 20, 8 << 20)
    result = ruler.model_delivery(m, binding, "A")
    assert result["step"] == 250
    assert result["tag"]["logical_generation"] == 1
    assert result["tag"]["source_revision"] != "f" * 40
    assert result["checkpoint"]["path"].endswith("learner.pt")
    m.finish()


@pytest.mark.parametrize("fault", ["legacy", "counter", "holdout", "saved_rng", "saved_checkpoint", "closure", "independent"])
def test_rehashed_delivery_and_saved_state_refusals(tmp_path: Path, fault: str):
    binding = delivery_graph(tmp_path, fault)
    with pytest.raises(ValueError):
        ruler.model_delivery(ruler.Metadata(1 << 20, 8 << 20), binding, "A")
