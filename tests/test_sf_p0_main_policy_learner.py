"""Authored synthetic-only authorities; actual finite worker/host/main-CE controls."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import chess
import numpy as np
import pytest
import torch

from chess_anti_engine.moves import COMPACT_MIRROR_POLICY_INV, move_to_index, move_to_index_for_encoding
from chess_anti_engine.replay.shard import load_shard_arrays, save_local_shard_arrays
from chess_anti_engine.train import trainer as trainer_module
from scripts import core_finite_learner as core
from scripts import sf_current_p0_sidecar as carrier
from scripts import sf_current_p0_sidecar_schema as schema
from scripts import sf_p0_main_policy_eligibility as eligibility
from scripts import sf_p0_main_policy_learner as finite
from scripts.bootstrap_checkpoint_resume import require_equal
from tests import test_core_finite_learner as old
from tests import test_sf_current_p0_finite_learner as fixture


def setup(tmp_path: Path, share: int = 1, mirror: float = .5, *, tb: bool = False, tb_marker: str = "applied") -> tuple[Any, ...]:
    """Real legal histories, invented data/receipts; never actual TRAIN authority."""
    _, _, _, old_gen = fixture.setup(tmp_path, alpha=0, mirror=mirror)
    b = tmp_path / "bindings"
    base = fixture.read({"path": str(b / "legacy-contract.json"), "sha256": "unused"})
    base["source_files"].update({name: core.file_sha(core.REPO / name) for name in finite.EXTRA_SOURCE})
    gen = fixture.read(old_gen)
    game = gen["games"][0]
    raw, _ = load_shard_arrays(game["shard"]["path"], lazy=False, validate=True)
    raw = {k: np.array(v, copy=True) for k, v in raw.items()}
    root = chess.Board("4k3/7p/8/8/1b6/8/r2B4/R3K1N1 w - - 0 1" if tb else
                       "4k3/7p/8/8/8/8/Pr6/R3KBN1 w - - 0 1")
    moves = ["a1a2", "b4d2", "e1d2"] if tb else ["g1f3", "b2a2", "a1a2"]
    board = root.copy(stack=True)
    positions, actions, details, records = [], [], [], []
    for i, uci in enumerate(moves):
        positions.append(board.copy(stack=True))
        rec = {"ply_index": i, "move_offset": i, "pov_color": board.turn, "has_policy": True, "is_sf_refute_opp": False}
        records.append(rec)
        legal = np.zeros(1858, dtype=np.uint8)
        for move in board.legal_moves:
            legal[move_to_index_for_encoding(move, board, policy_encoding="lc0_1858")] = 1
        raw["legal_mask"][i] = legal
        raw["policy_target"][i] = 0
        raw["policy_target"][i, int(np.flatnonzero(legal)[0])] = 1
        details.append({"physical_row_id": game["physical_game_id"] + ":" + str(i), "ply": i, "move_offset": i,
                        "history_sha256": core.digest([root.fen(), moves[:i]]),
                        "stored_input_float16_sha256": schema.array_digest(raw["x"][i])})
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves
        actions.append(move_to_index(move, board))
        board.push(move)
    shard = tmp_path / "main-policy-source.zarr"
    save_local_shard_arrays(shard, arrs=raw, meta={"games": 1, "positions": 3})
    core.seal(shard)
    files = [{"path": str(p.relative_to(shard)), "bytes": p.stat().st_size, "sha256": core.file_sha(p)}
             for p in sorted(shard.rglob("*")) if p.is_file()]
    game["shard"] = {"path": str(shard), "files": files}
    wrapper = fixture.read(game["producer_proof"])
    provenance = fixture.read(wrapper["evidence"]["provenance"])
    actual = {"game_id": game["game_id"], "start": {"root_fen": root.fen(), "opening_moves": [], "start_fen": root.fen(), "start_ply": 0, "is_selfplay": True},
              "records": records, "played_move_indices": actions, "final_fen": board.fen(),
              "tablebase_policy_applied_plies": [2] if tb and tb_marker == "applied" else [],
              "tablebase_policy_abstained_plies": [2] if tb and tb_marker == "abstained" else []}
    provenance.update(actual=actual, bank="train")
    outcome = fixture.read(wrapper["evidence"]["outcome_proof"])
    outcome["bank"] = "train"
    verified = outcome["verification"]
    verified.update(schema="CORE_ACTUALLY_VERIFIED_OUTCOME_V1", audit_sha256=core.digest(actual),
                    policy_applied_plies=[2] if tb and tb_marker == "applied" else [],
                    policy_abstained_plies=[2] if tb and tb_marker == "abstained" else [],
                    policy_all_eligible_raw_records_reprobed=True, policy_censored_bypass=False,
                    policy_eligible_raw_records=int(tb))
    outcome["proof_sha256"] = core.digest(verified)
    validation = fixture.read(wrapper["evidence"]["validation"])
    validation.update(bank="train", full_history_verified=True, usable_shard=True, raw_targets_verified=True,
                      rows_detail=details, verified_outcome_sha256=outcome["proof_sha256"])
    wrapper["evidence"] = {"provenance": old.frozen(b / "main-provenance.json", provenance),
                           "outcome_proof": old.frozen(b / "main-outcome.json", outcome),
                           "validation": old.frozen(b / "main-validation.json", validation)}
    wrapper["shard_inventory_sha256"] = core.digest(files)
    gen["audit_protocol"] = old.frozen(b / "fiction-audit.json", {"qualification_scope": "SYNTHETIC_NOT_REAL_AUTHORITY"})
    wrapper["audit_protocol"] = gen["audit_protocol"]
    game["producer_proof"] = old.frozen(b / "main-game-proof.json", wrapper)
    genref = old.frozen(b / "main-generation.json", gen)
    legacy = old.frozen(b / "main-legacy.json", base)
    c = {**base, "schema": finite.CONTRACT, "run_uuid": "33333333-3333-4333-8333-333333333333",
         "updates_per_segment": base["updates_per_generation"], "sf_p0_main_policy": {
             "schema": finite.PROFILE, "scope": "SYNTHETIC_ONLY", "teacher_share": share,
             "eligibility": eligibility.SCHEMA, "data_identity": {"run_uuid": gen["run_uuid"], "source_revision": gen["source_revision"]},
             "generation_manifest": genref, "data_authority": old.frozen(b / "main-bank-authority.json", {
                 "status": "SYNTHETIC_NOT_REAL_AUTHORITY", "generation_manifest": genref,
                 "data_identity": {"run_uuid": gen["run_uuid"], "source_revision": gen["source_revision"]}}), "limits": {"metadata_bytes": 8 << 20, "game_decoded_bytes": 8 << 20, "resident_bytes": 8 << 20}}}
    del c["updates_per_generation"]
    cref = old.frozen(b / "main-contract.json", c)
    return c, cref, legacy, genref, raw, positions, details


def accepted(tmp_path: Path, c: dict[str, Any], genref: dict[str, str], positions: Any, details: Any,
             *, invalid: bool = True, bad_vector: str | None = None, context_fault: str | None = None) -> dict[str, Any]:
    refs = fixture.accepted_carrier(tmp_path, c, genref)
    root = fixture.read(refs["root_acceptance"])
    candidate = fixture.read(root["candidate"])
    folder = tmp_path / "carrier/game-000000"
    folder.parent.chmod(0o755)
    folder.chmod(0o755)
    metadata = json.loads((folder / "rows.json").read_text())
    policy = np.zeros((3, 1858), dtype=np.float16)
    for i, row in enumerate(metadata["rows_detail"]):
        board = positions[i]
        row["input_identity"].update(ply=i, move_offset=i, STM_white=board.turn, fen=board.fen(),
            halfmove_clock=board.halfmove_clock, fullmove_number=board.fullmove_number,
            history_sha256=details[i]["history_sha256"])
        row["source_join"]["status"] = ("UNAVAILABLE" if i == 0 else
            "INVALID_REDUCED_LABEL" if i == 2 and invalid else "SOURCE_CONTEXT_REDUCED_LABEL")
        row["qualified_flags"]["policy"] = i > 0 and not (i == 2 and invalid)
        row["qualified_flags"]["native_wdl"] = False
        row["qualified_flags"]["stored_wdl"] = False
        if i:
            legal = np.flatnonzero(np.asarray(np.load(folder / "legal_mask_u8.npy", allow_pickle=False))[i])
            policy[i, legal[0]], policy[i, legal[1]] = .25, .75
    row = metadata["rows_detail"][1]
    if context_fault in {"fen", "history_sha256", "input_F16_sha256"}:
        row["input_identity"][context_fault] = "wrong"
    elif context_fault == "legal":
        row["legal_mask_sha256"] = "0" * 64
    elif context_fault == "ply":
        row["input_identity"]["ply"] = 900
    elif context_fault == "row":
        row["physical_row_id"] = "foreign:1"
    if bad_vector == "nan":
        policy[1, 0] = np.nan
    elif bad_vector == "negative":
        policy[1, 0] = -1
    elif bad_vector == "empty":
        policy[1] = 0
    elif bad_vector == "illegal":
        mask = np.asarray(np.load(folder / "legal_mask_u8.npy", allow_pickle=False))[1]
        policy[1] = 0
        policy[1, np.flatnonzero(mask == 0)[0]] = 1
    fixture.replace({"path": str(folder / "rows.json"), "sha256": "unused"}, metadata)
    # Policy qualification is deliberately independent of BOTH WDL variants.
    for name in ("stored_record_wdl_f16", "stored_query_wdl_f16", "qualified_native_wdl", "qualified_stored_wdl",
                 "predecessor_meta_i32", "observed_predecessor_meta", "observed_predecessor_pv"):
        path = folder / (name + ".npy")
        value = np.zeros_like(np.load(path, allow_pickle=False))
        path.chmod(0o644)
        np.save(path, value, allow_pickle=False)
    for name, value in [("native_wdl_permille_i16", np.full((3, 3), -1, dtype=np.int16)),
                        ("predecessor_pv_i16", np.tile(np.array([-1, -32768, 0, -1, -1], dtype=np.int16), (3, 48, 1))),
                        ("policy_f16", policy), ("qualified_policy", np.array([0, 1, int(not invalid)], dtype=np.uint8))]:
        path = folder / (name + ".npy")
        path.chmod(0o644)
        np.save(path, value, allow_pickle=False)
    for path in folder.iterdir():
        path.chmod(0o444)
    folder.chmod(0o555)
    candidate["files"] = [carrier.entry(folder.parent, p) for p in sorted(folder.iterdir())]
    candidate["bytes"] = sum(f["bytes"] for f in candidate["files"])
    candref = fixture.replace(root["candidate"], candidate)
    core.seal(folder.parent)
    v = root["validate"]
    protocol = fixture.read(v["protocol"])
    protocol["candidate"] = candref
    vp = fixture.replace(v["protocol"], protocol)
    ad = fixture.read(v["admission"])
    ad["protocol"] = vp
    va = fixture.replace(v["admission"], ad)
    command = [*v["command"][:-8], "--protocol", vp["path"], "--protocol-sha256", vp["sha256"],
               "--admission", va["path"], "--admission-sha256", va["sha256"]]
    terminal = fixture.read(v["terminal"])
    terminal["command"] = command
    vt = fixture.replace(v["terminal"], terminal)
    validation = fixture.read(root["validation"])
    validation.update(candidate=candref, protocol=vp, execution_admission=va)
    valref = fixture.replace(root["validation"], validation)
    root.update(candidate=candref, validation=valref, validate={"protocol": vp, "admission": va, "command": command, "terminal": vt})
    rr = fixture.replace(refs["root_acceptance"], root)
    review = fixture.read(refs["independent_review"])
    review.update(root_acceptance=rr, candidate=candref, validation=valref)
    return {"root_acceptance": rr, "independent_review": fixture.replace(refs["independent_review"], review)}


def stage(tmp_path: Path, cref: dict[str, str], gen: dict[str, str], refs: Any = None,
          *, segment: int = 1, parent: Any = None, label: str = "main-segment") -> dict[str, str]:
    return old.frozen(tmp_path / "bindings" / (label + ".json"), {"schema": finite.STAGE,
        "contract_sha256": cref["sha256"], "segment": segment, "generations": [gen], "parent": parent,
        "main_policy_sidecars": None if refs is None else [refs], "output": str(tmp_path / label)})


def test_zero_public_worker_is_unwrapped_and_matches_legacy_with_poisoned_sidecar_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c, cref, legacy, gen, *_ = setup(tmp_path, share=0)
    before = core.execute(legacy, old.stage(tmp_path, legacy, [gen], None, "legacy-zero"))
    monkeypatch.setattr(finite.MainPolicyProfile, "prepare", lambda *_: pytest.fail("zero sidecar authority IO"))
    monkeypatch.setattr(finite.MainPolicyProfile, "wrap", lambda *_: pytest.fail("zero wrapped replay"))
    monkeypatch.setattr(schema, "load_npy", lambda *_: pytest.fail("zero payload IO"))
    after = finite.execute(cref, stage(tmp_path, cref, gen))
    assert after["rng"] == before["rng"]
    assert after["draws_sha256"] == before["draws_sha256"]
    assert after["membership"] == before["membership"]
    assert after["main_policy_batch_counts"] == []
    assert after["main_policy_replaced_rows"] == 0
    assert after["main_policy_original_rows"] == after["drawn_rows"] == 4
    new = torch.load(after["checkpoint"]["path"], map_location="cpu", weights_only=True)
    old_state = torch.load(before["checkpoint"]["path"], map_location="cpu", weights_only=True)
    assert new.pop(finite.CHECKPOINT_KEY) == finite.checkpoint_identity(c, cref)
    require_equal(new, old_state, "zero state/RNG/model/optimizer/scheduler except explicit family tag")
    poisoned = fixture.read(stage(tmp_path, cref, gen, label="poison-zero"))
    poisoned["main_policy_sidecars"] = [{"path": "/never-open-this-poison", "sha256": "f" * 64}]
    with pytest.raises(ValueError, match="zero main-policy must bypass sidecar"):
        finite.execute(cref, old.frozen(tmp_path / "bindings/poisoned-stage.json", poisoned))


@pytest.mark.parametrize("mirror", [0.0, 1.0])
def test_actual_worker_host_main_ce_analytic_gradient_fallback_and_single_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                              mirror: float) -> None:
    c, cref, _, gen, source, positions, details = setup(tmp_path, mirror=mirror)
    refs = accepted(tmp_path, c, gen, positions, details)
    metadata = json.loads((tmp_path / "carrier/game-000000/rows.json").read_text())
    assert metadata["rows_detail"][1]["qualified_flags"] == {"policy": True, "stored_wdl": False, "native_wdl": False}
    for flag in ("qualified_native_wdl", "qualified_stored_wdl"):
        assert not np.load(tmp_path / "carrier/game-000000" / (flag + ".npy"), allow_pickle=False).any()
    calls = []
    original = trainer_module.compute_loss

    def observed(outputs: Any, batch: Any, **kwargs: Any) -> Any:
        assert kwargs["sf_wdl_frac"] == kwargs["search_wdl_frac"] == 0
        assert kwargs["w_policy"] == kwargs["w_wdl"] == 1
        assert all(v == 0 for k, v in kwargs.items() if k.startswith("w_") and k not in {"w_policy", "w_wdl"})
        assert kwargs["sf_policy_floor"].w == kwargs["sf_shape"].w == 0
        actual = original(outputs, batch, **kwargs)
        logits = outputs["policy"]
        legal = batch["legal_mask"].bool()
        target = batch["policy_t"].float()
        target = target / target.sum(-1, keepdim=True)
        masked = logits.float().masked_fill(~legal, -1e9)
        expected_ce = -(target * torch.log_softmax(masked, -1)).sum(-1).mean()
        torch.testing.assert_close(actual["policy_ce"], expected_ce)
        grad = torch.autograd.grad(actual["policy_ce"], logits, retain_graph=True)[0]
        expected_grad = (torch.softmax(masked, -1) - target) * legal / target.shape[0]
        torch.testing.assert_close(grad, expected_grad)
        indices = batch["ply_index"].detach().cpu().numpy()
        control_target = source["policy_target"][indices].astype(np.float32)
        if mirror:
            control_target = control_target[:, COMPACT_MIRROR_POLICY_INV]
        control_batch = {**batch, "policy_t": torch.as_tensor(control_target, device=logits.device)}
        control = original(outputs, control_batch, **kwargs)
        control_grad = torch.autograd.grad(control["policy_ce"], logits, retain_graph=True)[0]
        if bool((indices == 1).any()):
            assert not torch.equal(grad, control_grad)
        outcome = torch.nn.functional.cross_entropy(outputs["wdl"].float(), batch["wdl_t"])
        torch.testing.assert_close(actual["blended_wdl_ce"], outcome)
        torch.testing.assert_close(actual["total"], expected_ce + outcome)
        calls.append({k: batch[k].detach().cpu().numpy().copy() for k in ("policy_t", "has_policy", "legal_mask", "wdl_t", "ply_index")})
        return actual

    monkeypatch.setattr(trainer_module, "compute_loss", observed)
    result = finite.execute(cref, stage(tmp_path, cref, gen, refs))
    journal = [json.loads(x) for x in (tmp_path / "main-segment/draws.jsonl").read_text().splitlines()]
    teacher = np.load(tmp_path / "carrier/game-000000/policy_f16.npy", allow_pickle=False)
    expected_rng = np.random.default_rng(c["seed"])
    replaced = 0
    receipts = result["main_policy_batch_counts"]
    assert len(receipts) == 2
    for number, (draw, batch, receipt) in enumerate(zip(journal, calls, receipts, strict=True)):
        indices = [r["row_index"] for r in draw["rows"]]
        np.testing.assert_array_equal(indices, expected_rng.choice(3, size=2, replace=True))
        if mirror:
            expected_rng.random(2)
        expected = source["policy_target"][indices].astype(np.float32)
        for i, row in enumerate(indices):
            if row == 1:
                expected[i] = teacher[row]
        legal = source["legal_mask"][indices]
        if mirror:
            expected = expected[:, COMPACT_MIRROR_POLICY_INV]
            legal = legal[:, COMPACT_MIRROR_POLICY_INV]
        np.testing.assert_array_equal(batch["policy_t"], expected)
        np.testing.assert_array_equal(batch["legal_mask"], legal)
        np.testing.assert_array_equal(batch["has_policy"], source["has_policy"][indices])
        np.testing.assert_array_equal(batch["wdl_t"], source["wdl_target"][indices])
        expected_replaced = indices.count(1)
        assert receipt["draw_batch"] == number
        assert receipt["rows"] == receipt["original_policy_supervised"] == 2
        assert receipt["qualified_teacher"] == receipt["replaced"] == expected_replaced
        assert receipt["structurally_eligible"] == indices.count(0) + indices.count(1)
        assert receipt["original_fallback"] == 2 - expected_replaced
        assert receipt["replaced_normalized_f32_changed"] == expected_replaced
        replaced += expected_replaced
    assert len(calls) == len(journal) == 2
    assert 0 < replaced < 4  # Both actual teacher and exact original fallback reach the loss.
    assert result["main_policy_replaced_rows"] == replaced
    assert result["main_policy_original_rows"] == 4 - replaced
    assert result["rng"]["numpy"] == expected_rng.bit_generator.state
    assert core.CORE_KWARGS["w_policy"] == core.CORE_KWARGS["w_wdl"] == 1
    assert core.CORE_KWARGS["sf_wdl_frac"] == core.CORE_KWARGS["search_wdl_frac"] == 0


@pytest.mark.parametrize("marker", ["applied", "abstained"])
def test_tb_markers_and_material_keep_original_bytes_and_mask_without_resampling(tmp_path: Path, marker: str) -> None:
    c, _, _, gen, source, positions, details = setup(tmp_path, tb=True, tb_marker=marker)
    refs = accepted(tmp_path, c, gen, positions, details, invalid=False)
    profile = finite.MainPolicyProfile()
    profile.prepare(c, {"generations": [gen], "main_policy_sidecars": [refs]}, [fixture.read(gen)])
    from chess_anti_engine.replay.generation import DrawnRow

    source_sha = core.digest(fixture.read(gen)["games"][0]["shard"]["files"])
    rows = [DrawnRow(1, source_sha, i) for i in range(3)]
    original_bytes = source["policy_target"].tobytes()
    result = profile.overlay(source, rows)
    assert source["policy_target"].tobytes() == original_bytes
    assert result["policy_target"].tobytes() == original_bytes
    assert result["has_policy"] is source["has_policy"]
    assert result["wdl_target"] is source["wdl_target"]
    assert profile.counts["drawn_rows"] == profile.counts["original_fallback"] == 3
    assert profile.counts["TB_candidate"] == 1
    assert profile.counts["material_le7"] == 2
    assert profile.counts["replaced"] == 0
    # A correctly authenticated value-only row is not made policy-supervised.
    item = profile.index[source_sha]
    assert item[2][2]["TB_applied"] is (marker == "applied")
    assert item[2][2]["TB_abstained"] is (marker == "abstained")
    item[2][1]["original_has_policy"] = False
    source["has_policy"][1] = 0
    result = profile.overlay(source, rows)
    assert result["has_policy"][1] == 0
    assert result["policy_target"][1].tobytes() == source["policy_target"][1].tobytes()


@pytest.mark.parametrize("fault", ["missing", "foreign", "duplicate", "disagreement", "overlap", "not_reprobed"])
def test_strict_marker_lists_fail_closed(fault: str) -> None:
    actual = {"tablebase_policy_applied_plies": [5], "tablebase_policy_abstained_plies": [6]}
    verified = {"policy_applied_plies": [5], "policy_abstained_plies": [6], "policy_eligible_raw_records": 2,
                "policy_all_eligible_raw_records_reprobed": True, "policy_censored_bypass": False}
    if fault == "missing":
        del actual["tablebase_policy_applied_plies"]
    elif fault == "foreign":
        actual["tablebase_policy_applied_plies"] = verified["policy_applied_plies"] = [9]
    elif fault == "duplicate":
        actual["tablebase_policy_applied_plies"] = [5, 5]
    elif fault == "disagreement":
        verified["policy_applied_plies"] = []
    elif fault == "overlap":
        actual["tablebase_policy_abstained_plies"] = verified["policy_abstained_plies"] = [5]
    else:
        verified["policy_all_eligible_raw_records_reprobed"] = False
    if fault == "missing":
        with pytest.raises(KeyError, match="tablebase_policy_applied_plies"):
            eligibility.marker_sets(actual, verified, {5, 6})
    else:
        with pytest.raises(ValueError, match="original strict TB marker"):
            eligibility.marker_sets(actual, verified, {5, 6})


@pytest.mark.parametrize("share", [False, True, -.1, .2, 2, float("nan")])
def test_only_explicit_zero_or_full_replacement_profile(share: Any) -> None:
    with pytest.raises(ValueError, match="teacher share must be zero or one"):
        finite.validate_profile({"schema": finite.PROFILE, "scope": "SYNTHETIC_ONLY", "teacher_share": share,
            "eligibility": eligibility.SCHEMA, "data_identity": {}, "generation_manifest": {}, "data_authority": {}, "limits": {}})


def test_old_public_contracts_do_not_admit_new_family(tmp_path: Path) -> None:
    _, cref, legacy, gen, *_ = setup(tmp_path, share=0)
    with pytest.raises(ValueError, match="unknown or missing contract fields"):
        core.execute(cref, old.stage(tmp_path, cref, [gen], None, "old-refusal"))
    with pytest.raises(ValueError, match="not the main-policy learning family"):
        finite.execute(legacy, stage(tmp_path, legacy, gen))


@pytest.mark.parametrize("fault", ["nan", "negative", "empty", "illegal"])
def test_bad_true_qualified_vector_refuses_before_model_construction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                   fault: str) -> None:
    c, cref, _, gen, _, positions, details = setup(tmp_path)
    refs = accepted(tmp_path, c, gen, positions, details, bad_vector=fault)
    monkeypatch.setattr(core, "make_trainer", lambda *_args, **_kwargs: pytest.fail("invalid teacher reached model"))
    with pytest.raises(ValueError, match="invalid qualified teacher vector"):
        finite.execute(cref, stage(tmp_path, cref, gen, refs))


@pytest.mark.parametrize("fault", ["fen", "history_sha256", "input_F16_sha256", "legal", "ply", "row"])
def test_authenticated_but_crossed_context_refuses_before_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             fault: str) -> None:
    c, cref, _, gen, _, positions, details = setup(tmp_path)
    refs = accepted(tmp_path, c, gen, positions, details, context_fault=fault)
    monkeypatch.setattr(core, "make_trainer", lambda *_args, **_kwargs: pytest.fail("crossed context reached model"))
    with pytest.raises(ValueError, match=r"(main-policy row identities|SF-P0 row identities|eligibility|carrier legal mask)"):
        finite.execute(cref, stage(tmp_path, cref, gen, refs))


def test_same_family_continuation_tags_and_cross_contract_refusal(tmp_path: Path) -> None:
    c, cref, _, gen, *_ = setup(tmp_path, share=0)
    result = finite.execute(cref, stage(tmp_path, cref, gen))
    command = [sys.executable, "-B", "-m", finite.MODULE, "--contract", cref["path"], "--contract-sha256", cref["sha256"],
               "--stage", result["stage"]["path"], "--stage-sha256", result["stage"]["sha256"]]
    terminal = old.frozen(tmp_path / "bindings/parent-terminal.json", {"status": "PASS", "returncode": 0,
        "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED", "supervisor_sha256": "b" * 64,
        "command": command, "qualification_scope": "SYNTHETIC_NOT_REAL_AUTHORITY"})
    boundary_path = tmp_path / "main-segment/boundary.json"
    parent = {"boundary": {"path": str(boundary_path), "sha256": core.file_sha(boundary_path)},
              "owner_terminal": terminal, "owner_sha256": "b" * 64, "command": command}
    continued = finite.execute(cref, stage(tmp_path, cref, gen, segment=2, parent=parent, label="continuation"))
    assert continued["step"] == 4
    changed = copy.deepcopy(c)
    changed["sf_p0_main_policy"]["teacher_share"] = 1
    changed_ref = old.frozen(tmp_path / "bindings/changed-contract.json", changed)
    # Authentication fails on the contract before any parent optimizer restoration.
    with pytest.raises(ValueError, match="same-family contract/profile/data parent mismatch"):
        finite.MainPolicyProfile().closed_parent(parent, changed, changed_ref,
            {"generation": 2, "generations": [gen], "main_policy_sidecars": None})
    checkpoint = Path(continued["checkpoint"]["path"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state[finite.CHECKPOINT_KEY] == finite.checkpoint_identity(c, cref)
    assert state["step"] == 4


def test_marker_sets_preserve_both_authenticated_classes() -> None:
    actual = {"tablebase_policy_applied_plies": [5], "tablebase_policy_abstained_plies": [6]}
    verified = {"policy_applied_plies": [5], "policy_abstained_plies": [6], "policy_eligible_raw_records": 2,
                "policy_all_eligible_raw_records_reprobed": True, "policy_censored_bypass": False}
    assert eligibility.marker_sets(actual, verified, {5, 6}) == ({5}, {6})
