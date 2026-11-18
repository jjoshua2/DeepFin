"""Actual Tiny two-bank continuation and cumulative durable quota controls.

All Root/source authorities are explicitly fictional fixture metadata. These
tests do not claim real collection, a qualified actor, or a research result.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import pytest
import torch

from chess_anti_engine.replay.capped_cycle import Usage, read_usage
from chess_anti_engine.replay.generation import DrawnRow
from scripts import core_capped_on_policy_cycle as entry
from scripts import core_capped_on_policy_cycle_proof as proof
from scripts import core_closed_once_actor as bridge
from scripts import core_finite_learner as core
from scripts import core_selfplay_union_stored_row_once as old
from scripts import core_selfplay_union_stored_row_once_proof as old_proof
from chess_anti_engine.replay.fixed_population import manifest_identity
from tests.test_core_finite_learner import frozen
from tests.test_core_generation_bank_learner import bank
from tests.test_core_selfplay_union_manifest import compiled, plan
from tests.test_core_selfplay_union_stored_row_once import contract as old_contract, stage as old_stage


def delivery(tmp_path: Path, c: dict[str, Any], cref: dict[str, str], boundary: dict[str, Any], *, cycle: bool) -> dict[str, Any]:
    anchored = not cycle and c.get("policy_anchor") is not None
    module = ("scripts.core_capped_on_policy_cycle_proof" if cycle else
              "scripts.core_selfplay_union_policy_anchor_stored_row_once_proof" if anchored else
              "scripts.core_selfplay_union_stored_row_once_proof")
    command = [sys.executable, "-B", "-m", module, "--contract", cref["path"], "--contract-sha256", cref["sha256"],
               "--stage", boundary["stage"]["path"], "--stage-sha256", boundary["stage"]["sha256"]]
    terminal = frozen(tmp_path / "terminal.json", {"status": "PASS", "returncode": 0, "owned_processes_closed": True,
        "custody": "OWNED_PROCESSES_CLOSED", "supervisor_sha256": "f" * 64, "command": command})
    bref = {"path": str(Path(boundary["checkpoint"]["path"]).parent / "boundary.json"),
            "sha256": core.file_sha(Path(boundary["checkpoint"]["path"]).parent / "boundary.json")}
    binding = {"boundary": bref, "owner_terminal": terminal, "owner_sha256": "f" * 64, "command": command}
    owned = {"scripts/core_capped_on_policy_cycle.py": {"path": str(core.REPO / "scripts/core_capped_on_policy_cycle.py"),
              "sha256": core.file_sha(core.REPO / "scripts/core_capped_on_policy_cycle.py")}}
    source = frozen(tmp_path / "source.json", {"owned_source_files": owned})
    review = frozen(tmp_path / "source-review.json", {"status": ("ACCEPT_CAPPED_ON_POLICY_CYCLE_SOURCE_ONLY_NO_FINDINGS_V1" if cycle
        else "ACCEPT_GLOBAL_UNION_POLICY_ANCHOR_INTEGRATION_SOURCE_ONLY_NO_FINDINGS_V1" if anchored
        else "ACCEPT_HISTORICAL_SELFPLAY_UNIFORM_ONCE_ENTRY_SOURCE_ONLY_NO_FINDINGS_V1"),
        "source_manifest": source, "owned_source_files": owned, "findings": []})
    from scripts import core_selfplay_union_policy_anchor_stored_row_once as anchored_entry

    old_entry = anchored_entry if anchored else old
    tag = (entry.checkpoint_tag(c, cref, boundary) if cycle else {"identity": old_entry.checkpoint_identity(c, cref, boundary["segment"]),
        **{k: boundary[k] for k in ("quota_state", "draw_journals", "quota_ledger_checkpoint", "population_identity", "source_origin_binding")}})
    model = {"checkpoint": boundary["checkpoint"], "architecture": c["architecture"], "step": boundary["step"],
        "model_state_sha256": boundary["own_live_certificate"]["saved_model_state_sha256"],
        "model_state_hash_schema": "union_typed_tensor_tree_sha256_v1", "label": "fictional software fixture",
        "tag_key": entry.CHECKPOINT_KEY if cycle else old_entry.CHECKPOINT_KEY, "tag": tag}
    root = {"status": ("ROOT_ACCEPTS_CLOSED_CAPPED_ON_POLICY_CYCLE_MODEL_DELIVERY_V1" if cycle
                        else "ROOT_ACCEPTS_CLOSED_U2_FULL_PASS_MODEL_DELIVERY_V1" if anchored
                        else "ROOT_ACCEPTS_CLOSED_U1_FULL_PASS_MODEL_DELIVERY_V1"),
        "study_uuid": c["study_uuid"], "method": c["method"], "contract": cref, "boundary": bref,
        "closed_binding": binding, "entry_source": source, "entry_source_review": review, "model": model}
    rref = frozen(tmp_path / "root.json", root)
    iref = frozen(tmp_path / "independent.json", dict(root, status=("ACCEPT_CLOSED_CAPPED_ON_POLICY_CYCLE_MODEL_DELIVERY_NO_FINDINGS_V1"
        if cycle else "ACCEPT_CLOSED_U2_FULL_PASS_MODEL_DELIVERY_NO_FINDINGS_V1" if anchored
        else "ACCEPT_CLOSED_U1_FULL_PASS_MODEL_DELIVERY_NO_FINDINGS_V1"), root_acceptance=rref, findings=[]))
    return {"schema": bridge.SCHEMA, "family": "capped_cycle" if cycle else "policy_anchor" if anchored else "global_union",
            "delivery": {"root": rref, "independent": iref}}


def origin(tmp_path: Path, *, coefficient: float | None = None, batch_size: int = 2) -> tuple[dict[str, Any], dict[str, Any]]:
    if coefficient is None:
        c, cref = old_contract(tmp_path, split=False)
        if batch_size != c["batch_size"]:
            from tests.test_core_selfplay_union_stored_row_once import method

            c["batch_size"] = batch_size
            c["updates_per_segment"] = [(c["population"]["rows"] + batch_size - 1) // batch_size]
            c["method"] = method(tmp_path, c, "different-batch")
            cref = frozen(tmp_path / "different-batch.contract.json", c)
        boundary = old_proof.execute(cref, old_stage(tmp_path, cref, c, 1, None, "origin"))
    else:
        from scripts import core_selfplay_union_policy_anchor_stored_row_once_proof as anchored_proof
        from tests.test_core_selfplay_union_policy_anchor import contract as anchored_contract, stage as anchored_stage

        c, cref = anchored_contract(tmp_path, coefficient=coefficient)
        boundary = anchored_proof.execute(cref, anchored_stage(tmp_path, cref, c, 1, None, "origin"))
    return c, delivery(tmp_path, c, cref, boundary, cycle=False)


def cycle_contract(tmp_path: Path, oc: dict[str, Any], initial: dict[str, Any], parent: dict[str, Any] | None, number: int) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    tmp_path.mkdir()
    parent_model = ({"checkpoint": oc["initial_model"], "step": oc["initial_model_step"]} if parent is None
                    else bridge.closed_delivery(parent)["model"])
    accepted = bank(tmp_path, {"architecture": oc["architecture"]}, 10 + number,
                    parent_model["checkpoint"], parent_model["step"])
    from scripts import core_generation_bank_acceptance as admission

    view = admission.validate_accepted_generation(accepted["root_acceptance"], accepted["independent_review"],
        max_file_bytes=4 << 20, max_total_bytes=16 << 20, max_games=3)
    n, sources = sum(g["rows"] for g in view.games), len(view.games)
    pref = plan(tmp_path, [{"kind": "bank", **accepted}], expected_sources=sources, expected_rows=n, architecture=oc["architecture"])
    manifest, mref = compiled(tmp_path, pref)
    population = {**manifest_identity(manifest, mref), "plan": pref}
    c = {k: copy.deepcopy(oc[k]) for k in ("study_uuid", "run_uuid", "source_revision", "architecture", "initial_model",
        "initial_model_step", "batch_size", "seed", "runtime", "lr", "mirror_prob")}
    c.update(schema=entry.CONTRACT, source_files={n: core.file_sha(core.REPO / n) for n in entry.SOURCE_FILES},
        cycle=number, origin=initial, parent=parent, population=population, bank=accepted,
        updates_per_segment=[(n + c["batch_size"] - 1) // c["batch_size"]],
        limits={"max_rows": n, "max_decoded_bytes": 64 << 20, "max_file_bytes": 8 << 20,
                "max_total_metadata_bytes": 256 << 20, "max_lineage_rows": 100},
        bank_limits={"max_file_bytes": 4 << 20, "max_total_bytes": 16 << 20, "max_games": 3},
        policy_anchor=copy.deepcopy(oc.get("policy_anchor")), retained_search_share=0.0, maximum_presentations=2)
    seed = 300 + number
    c["sampling"] = {**oc["sampling"], "order_seed": seed, "expected_rows": n,
                      "permutation_sha256": core.digest(entry.permutation_for(n, seed))}
    c["method"] = frozen(tmp_path / "method.json", {"schema": "CORE_CAPPED_ON_POLICY_CYCLE_METHOD_V1",
        "study_uuid": c["study_uuid"], "origin": initial, "cycle_recipe": entry.recipe(c)})
    cref = frozen(tmp_path / "contract.json", c)
    sref = frozen(tmp_path / "stage.json", {"schema": entry.STAGE, "contract_sha256": cref["sha256"],
        "segment": 1, "cycle": number, "population": population, "parent": parent, "output": str(tmp_path / "out")})
    return c, cref, sref


def test_actual_two_fresh_banks_preserve_full_state_rng_and_cumulative_quota(tmp_path: Path) -> None:
    oc, initial = origin(tmp_path, batch_size=3)
    parent = initial
    previous = bridge.closed_delivery(initial)["boundary"]
    for number in (1, 2):
        folder = tmp_path / f"cycle-{number}"
        c, cref, sref = cycle_contract(folder, oc, initial, parent, number)
        result = proof.execute(cref, sref)
        cert = result["own_live_certificate"]
        assert cert["initial"]["state"] == previous["own_live_certificate"]["own_live_state_fingerprint"]
        assert cert["initial"]["rng"] == previous["rng"]
        assert cert["actual_calls"]["forward"] == c["updates_per_segment"][0]
        assert result["step_start"] == previous["step"]
        assert result["step"] == previous["step"] + c["updates_per_segment"][0]
        assert result["quota_state"]["cursor"] == result["applied_views"] == c["population"]["rows"]
        assert result["quota_state"]["last_batch_rows"] == (c["population"]["rows"] % c["batch_size"] or c["batch_size"])
        records = [json.loads(line) for line in Path(result["draw_journals"][-1]["path"]).read_text().splitlines()]
        assert [len(record["rows"]) for record in records] == [3, 1]
        actual_ids = [(row["generation"], row["source_sha256"], row["row_index"]) for record in records for row in record["rows"]]
        expected_ids = {(generation, source, i) for generation, source, rows in result["membership"] for i in range(rows)}
        assert len(actual_ids) == len(set(actual_ids)) == 4
        assert set(actual_ids) == expected_ids
        assert result["draw_journals"][:-1] == previous["draw_journals"]
        assert result["usage_state"]["presentations"] == previous.get("usage_state", {}).get("presentations", oc["population"]["rows"]) + c["population"]["rows"]
        assert result["usage_state"]["maximum_observed"] <= 2
        assert result["state_handoff"] == "adjacent_full_state_continuation"
        parent = delivery(folder, c, cref, result, cycle=True)
        public = proof.validate_closed_model_delivery(cref, bridge.closed_delivery(parent)["root"]["closed_binding"])
        assert public["pass_complete"] is True
        assert public["own_live_certificate"] == result["own_live_certificate"]
        saved = torch.load(result["checkpoint"]["path"], map_location="cpu", weights_only=True)
        assert saved[entry.CHECKPOINT_KEY]["usage_state"] == result["usage_state"]
        for slot in saved["opt"]["state"].values():
            assert float(slot["step"]) == result["step"]
        previous = result


def test_actual_model_only_origin_starts_at_zero_then_two_banks_continue_without_optimizer_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    oc, _ = old_contract(tmp_path, split=False)  # Tiny donor includes a deliberately unusable historical opt state.
    oc["batch_size"] = 3
    initial = {"schema": "CORE_CAPPED_CYCLE_MODEL_ONLY_ORIGIN_V1", "family": "model_only",
        "model": oc["initial_model"], "architecture": oc["architecture"], "recorded_step": oc["initial_model_step"]}
    parent = None
    previous = None
    for number in (1, 2):
        folder = tmp_path / f"fresh-cycle-{number}"
        c, cref, sref = cycle_contract(folder, oc, initial, parent, number)
        if number == 2:
            def forbidden(*_args: Any, **_kwargs: Any) -> None:
                pytest.fail("adjacent cycle silently reset optimizer/model state")
            monkeypatch.setattr(entry.bank, "fresh_model", forbidden)
        result = proof.execute(cref, sref)
        certificate = result["own_live_certificate"]
        if previous is None:
            assert result["step_start"] == 0
            assert result["state_handoff"] == "model_only_fresh_optimizer"
            assert certificate["initial"]["fresh_optimizer_slots"] == 0
            assert certificate["initial"]["initial_student_step"] == 0
            assert certificate["parent_initial_state_matches"] is None
            donor = torch.load(oc["initial_model"]["path"], map_location="cpu", weights_only=True)
            assert certificate["initial"]["model"]["model_state_sha256"] == proof.fingerprint(donor["model"], torch)
        else:
            assert result["step_start"] == previous["step"]
            assert certificate["initial"]["state"] == previous["own_live_certificate"]["own_live_state_fingerprint"]
            assert certificate["initial"]["rng"] == previous["rng"]
            assert result["draw_journals"][:-1] == previous["draw_journals"]
            assert result["state_handoff"] == "adjacent_full_state_continuation"
        parent = delivery(folder, c, cref, result, cycle=True)
        validated = proof.validate_closed_model_delivery(cref, bridge.closed_delivery(parent)["root"]["closed_binding"])
        assert validated["model"]["checkpoint"] == result["checkpoint"]
        assert validated["model"]["model_state_sha256"] == certificate["saved_model_state_sha256"]
        saved = torch.load(result["checkpoint"]["path"], map_location="cpu", weights_only=True)
        assert "must_not_load" not in saved["opt"]
        assert saved["step"] == result["step"]
        assert result["usage_state"]["maximum_observed"] <= 2
        previous = result


@pytest.mark.parametrize("coefficient", [0.0, 1.0])
def test_actual_anchor_origin_continuation_keeps_reference_and_no_extra_forwards(tmp_path: Path, coefficient: float) -> None:
    oc, initial = origin(tmp_path, coefficient=coefficient)
    parent = bridge.closed_delivery(initial)["boundary"]
    c, cref, sref = cycle_contract(tmp_path / "cycle", oc, initial, initial, 1)
    result = proof.execute(cref, sref)
    cert = result["own_live_certificate"]
    assert cert["initial"]["state"] == parent["own_live_certificate"]["own_live_state_fingerprint"]
    assert cert["initial"]["rng"] == parent["rng"]
    assert cert["extra_forwards"] == 0
    assert cert["actual_calls"]["forward"] == result["applied_updates"]
    anchor = cert["policy_anchor"]
    assert anchor["identity"] == c["policy_anchor"] == oc["policy_anchor"]
    assert anchor["teacher_unchanged"] is True
    assert anchor["no_optimizer_ownership"] is True
    assert anchor["no_reference_gradients"] is True
    if coefficient:
        assert anchor["actual_reference_forwards"] == result["applied_updates"]
        assert anchor["diagnostics"]["eligible_rows"] == result["applied_views"]
        assert anchor["initial_reference"] == parent["own_live_certificate"]["policy_anchor"]["initial_reference"]
    else:
        assert anchor["actual_reference_forwards"] == 0
        assert anchor["initial_reference"] is None


def test_original_plus_mirror_is_one_reservation_and_third_use_is_refused() -> None:
    usage = Usage(maximum=2, max_rows=4)
    row = asdict(DrawnRow(7, "a" * 64, 0))
    usage.accept([row])
    # Orientation is not part of row identity; a mirror cannot make new quota.
    usage.accept([row])
    assert usage.state()["presentations"] == 2
    with pytest.raises(ValueError, match="presentation cap exceeded"):
        usage.accept([row])
    assert usage.state()["presentations"] == 2


@pytest.mark.parametrize("fault", ["ledger", "cycle", "parent", "targets", "method", "membership", "actor_step", "lr"])
def test_cumulative_identity_tampering_refuses_before_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    oc, initial = origin(tmp_path)
    c, _, sref = cycle_contract(tmp_path / "cycle", oc, initial, initial, 1)
    if fault == "ledger":
        c["sampling"]["quota_ledger"] = str(tmp_path / "fresh-but-wrong-ledger")
    elif fault == "cycle":
        c["cycle"] = 2
    elif fault == "parent":
        c["parent"]["delivery"]["root"]["sha256"] = "0" * 64
    elif fault == "targets":
        c["retained_search_share"] = 0.1
    elif fault == "method":
        c["method"]["sha256"] = "0" * 64
    elif fault == "membership":
        c["population"]["membership_sha256"] = "0" * 64
    elif fault == "lr":
        c["lr"].update(peak=2e-4, floor=2e-4, warmup_start=2e-4)
    else:
        folder = tmp_path / "wrong-actor"
        folder.mkdir()
        model = bridge.closed_delivery(initial)["model"]
        c["bank"] = bank(folder, {"architecture": c["architecture"]}, 99, model["checkpoint"], model["step"] + 1)
    # Update selected method for identity faults so the intended downstream
    # lineage guard, rather than a stale-method alias, is actually exercised.
    if fault != "method":
        c["method"] = frozen(tmp_path / "changed-method.json", {"schema": "CORE_CAPPED_ON_POLICY_CYCLE_METHOD_V1",
            "study_uuid": c["study_uuid"], "origin": c["origin"], "cycle_recipe": entry.recipe(c)})
    cref = frozen(tmp_path / "changed-contract.json", c)
    _, stage = core.read_ref(sref)
    stage.update(contract_sha256=cref["sha256"], cycle=c["cycle"], population=c["population"], parent=c["parent"])
    sref = frozen(tmp_path / "changed-stage.json", stage)
    calls: list[bool] = []
    monkeypatch.setattr(core, "make_trainer", lambda *_a, **_k: calls.append(True))
    with pytest.raises(ValueError, match=r"cycle|metadata hash|bank"):
        proof.execute(cref, sref)
    assert calls == []


@pytest.mark.parametrize("fault", ["lr", "scheduler", "group"])
def test_real_resume_cannot_silently_replace_constructor_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    from scripts import bootstrap_checkpoint_resume as bootstrap

    oc, initial = origin(tmp_path)
    c, cref, sref = cycle_contract(tmp_path / "cycle", oc, initial, initial, 1)
    actual = bootstrap.resume_bootstrap
    def changed(trainer: Any, *args: Any, **kwargs: Any) -> Any:
        result = actual(trainer, *args, **kwargs)
        if fault == "lr":
            trainer.opt.param_groups[0]["lr"] *= 2
        elif fault == "scheduler":
            trainer._scheduler.eta_min *= 2
        else:
            trainer.opt.param_groups[0]["weight_decay"] += 0.1
        return result
    monkeypatch.setattr(bootstrap, "resume_bootstrap", changed)
    before = Path(c["sampling"]["quota_ledger"]).read_bytes()
    with pytest.raises(ValueError, match="actual cycle optimizer/scheduler recipe changed"):
        proof.execute(cref, sref)
    assert Path(c["sampling"]["quota_ledger"]).read_bytes() == before
    assert (tmp_path / "cycle/out/draws.jsonl").read_bytes() == b""
    assert not (tmp_path / "cycle/out/boundary.json").exists()


def test_actual_uncheckpointed_ledger_tail_never_restarts_or_builds_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    oc, initial = origin(tmp_path)
    c, cref, sref = cycle_contract(tmp_path / "cycle", oc, initial, initial, 1)
    with Path(c["sampling"]["quota_ledger"]).open("ab") as stream:
        stream.write(b"uncheckpointed failed sibling tail\n")
    calls: list[bool] = []
    monkeypatch.setattr(core, "make_trainer", lambda *_a, **_k: calls.append(True))
    with pytest.raises(ValueError, match="ahead/behind checkpoint"):
        proof.execute(cref, sref)
    assert calls == []
    assert not (tmp_path / "cycle/out/boundary.json").exists()


@pytest.mark.parametrize("fault", ["forward", "optimizer", "nonfinite"])
def test_actual_training_fault_spends_reservation_and_never_publishes_clean_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    oc, initial = origin(tmp_path)
    c, cref, sref = cycle_contract(tmp_path / "cycle", oc, initial, initial, 1)
    actual = core.make_trainer
    calls: list[str] = []
    def make(*args: Any, **kwargs: Any) -> Any:
        trainer = actual(*args, **kwargs)
        trainer.prefetch_batches = True
        if fault == "forward":
            def fail_forward(*_args: Any, **_kwargs: Any) -> Any:
                calls.append("forward")
                raise RuntimeError("CUDA owned synthetic forward failure")
            monkeypatch.setattr(trainer.model, "forward", fail_forward)
        elif fault == "optimizer":
            def fail_optimizer(*_args: Any, **_kwargs: Any) -> Any:
                calls.append("optimizer")
                raise RuntimeError("CUDA owned synthetic optimizer failure")
            monkeypatch.setattr(trainer.opt, "step", fail_optimizer)
        else:
            def nonfinite(grad: Any) -> Any:
                calls.append("gradient")
                return grad * float("inf")
            next(trainer.model.parameters()).register_hook(nonfinite)
        return trainer
    monkeypatch.setattr(core, "make_trainer", make)
    before = Path(c["sampling"]["quota_ledger"]).stat().st_size
    with pytest.raises(ValueError, match="retries forbidden") as caught:
        proof.execute(cref, sref)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert calls == ["gradient" if fault == "nonfinite" else fault]
    assert Path(c["sampling"]["quota_ledger"]).stat().st_size > before
    assert not (tmp_path / "cycle/out/boundary.json").exists()
    assert len((tmp_path / "cycle/out/draws.jsonl").read_text().splitlines()) == 1


def test_sealed_usage_journal_count_hash_and_cap_tampering_is_detected(tmp_path: Path) -> None:
    row = asdict(DrawnRow(4, "b" * 64, 0))
    path = tmp_path / "draws.jsonl"
    path.write_text(core.canonical({"draw_batch": 0, "rows": [row]}) + "\n")
    path.chmod(0o444)
    ref = {"path": str(path), "sha256": core.file_sha(path)}
    observed = read_usage([ref], maximum=2, max_rows=4, batch_size=2, max_file_bytes=1 << 20, max_total_bytes=1 << 20)
    assert observed.state()["unique_rows"] == 1
    assert observed.ledger_checkpoint(str(tmp_path / "ledger"))["sha256"] == ref["sha256"]
    with pytest.raises(ValueError, match="canonical unique path"):
        read_usage([ref, ref], maximum=2, max_rows=4, batch_size=2, max_file_bytes=1 << 20, max_total_bytes=1 << 20)


def test_actual_accepted_actor_snapshot_does_not_reopen_population_or_parent_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    oc, initial = origin(tmp_path)
    c, cref, sref = cycle_contract(tmp_path / "cycle", oc, initial, initial, 1)
    result = proof.execute(cref, sref)
    pair = delivery(tmp_path / "cycle", c, cref, result, cycle=True)
    actual = core.read_ref
    def metadata_only(ref: dict[str, str], *args: Any, **kwargs: Any) -> Any:
        assert ref not in (c["population"]["manifest"], c["parent"]["delivery"]["root"], c["parent"]["delivery"]["independent"])
        return actual(ref, *args, **kwargs)
    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("accepted actor transport reconstructed a virtual order or bank")
    monkeypatch.setattr(core, "read_ref", metadata_only)
    monkeypatch.setattr(entry, "metadata", forbidden)
    monkeypatch.setattr(entry, "validate_contract", forbidden)
    joined = bridge.closed_delivery(pair)
    assert joined["model"]["checkpoint"] == result["checkpoint"]
    assert joined["boundary"]["usage_state"] == result["usage_state"]


def test_synthetic_large_current_tag_and_hundreds_of_journal_refs_stay_compact() -> None:
    # Synthetic metadata only: no population payload, model, random order or
    # game-independence claim. Prior rows/counts/models are not embedded.
    ref = {"path": "/synthetic/accepted.json", "sha256": "a" * 64}
    c = {"study_uuid": "s", "run_uuid": "r", "source_revision": "b" * 40,
        "cycle": 501, "method": ref, "origin": {"family":"global_union", "delivery":{"root":ref,"independent":ref}},
        "parent": {"family":"capped_cycle", "delivery":{"root":ref,"independent":ref}},
        "sampling": {"order_seed": 9}, "maximum_presentations": 2, "policy_anchor": None, "retained_search_share": 0.0}
    b = {"quota_state": {"permutation":list(range(405013))},
        "draw_journals": [{"path":f"/synthetic/cycle-{i}/draws.jsonl", "sha256":"c" * 64} for i in range(500)],
        "quota_ledger_checkpoint": {"path":"/synthetic/quota.jsonl", "bytes":1 << 30,"sha256":"d" * 64},
        "population_identity": {"manifest":ref}, "source_origin_binding": {"manifest":ref},
        "usage_state": {"presentations":405013 * 501,"counts_sha256":"e" * 64}}
    tag = entry.checkpoint_tag(c, ref, b)
    assert entry.compact_metadata(tag) < 4 << 20
    assert set(tag["identity"]["parent"]["delivery"]) == {"root", "independent"}
    assert set(tag["usage_state"]) == {"presentations", "counts_sha256"}
    with pytest.raises(ValueError, match="closed metadata exceeds4MiB"):
        entry.compact_metadata({"oversized": "x" * (4 << 20)})


def test_real_closed_consumer_rejects_rehashed_counter_membership_and_alias_changes(tmp_path: Path) -> None:
    oc, initial = origin(tmp_path)
    c, cref, sref = cycle_contract(tmp_path / "cycle", oc, initial, initial, 1)
    result = proof.execute(cref, sref)
    pair = delivery(tmp_path / "cycle", c, cref, result, cycle=True)
    binding = bridge.closed_delivery(pair)["root"]["closed_binding"]
    path = Path(binding["boundary"]["path"])
    original = path.read_bytes()
    for fault in ("membership", "usage", "cursor", "step", "ledger", "initial_state"):
        value = json.loads(original)
        if fault == "membership":
            value["membership"][0][2] += 1
        elif fault == "usage":
            value["usage_state"]["presentations"] += 1
        elif fault == "cursor":
            value["quota_state"]["cursor"] -= 1
        elif fault == "step":
            value["step_start"] -= 1
        elif fault == "ledger":
            value["quota_ledger_checkpoint"]["sha256"] = "0" * 64
        else:
            value["own_live_certificate"]["initial"]["state"] = "0" * 64
        # Intentional fixture-only unseal/edit/reseal: production outputs remain
        # immutable. Preserve every original byte for the next negative case.
        path.chmod(0o644)
        path.write_text(core.canonical(value) + "\n")
        path.chmod(0o444)
        changed = copy.deepcopy(binding)
        changed["boundary"]["sha256"] = core.file_sha(path)
        with pytest.raises(ValueError, match="cycle"):
            proof.validate_closed_model_delivery(cref, changed)
        path.chmod(0o644)
        path.write_bytes(original)
        path.chmod(0o444)
