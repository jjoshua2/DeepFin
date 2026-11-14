"""Known synthetic evidence; no real data, engine, model or newly issued authority."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy
import struct
from typing import Any

import numpy as np
import pytest
import zarr

from scripts import sf_banked_p0_census as audit
from scripts import sf_current_p0_sidecar as carrier
from scripts import sf_current_p0_sidecar_schema as schema


def ref(path: Path, value: Any) -> dict[str, str]:
    path.write_bytes(audit.canonical(value))
    path.chmod(0o444)
    return carrier.reference(path)


def replace(reference: dict[str, str], value: Any) -> dict[str, str]:
    path = Path(reference["path"])
    path.chmod(0o644)
    return ref(path, value)


def read(reference: dict[str, str]) -> Any:
    return json.loads(Path(reference["path"]).read_bytes())


def fiction(tmp_path: Path, *, kept: tuple[int, ...] = (0, 1, 2, 3), corrupt: str = "",
            no_reduced_native: bool = False) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], list[dict[str, Any]]]:
    helpers = runpy.run_path(str(Path(__file__).with_name("test_sf_banked_p0_census.py")))
    base = tmp_path / "fiction"
    base.mkdir()
    p, generation, _producer = helpers["fixture"](base, kept=kept, corrupt=corrupt)
    item = generation["games"][0]
    if no_reduced_native:
        shard = Path(item["shard"]["path"])
        for path in [shard, *shard.rglob("*")]:
            path.chmod(0o755 if path.is_dir() else 0o644)
        group = zarr.open_group(str(shard), mode="a")
        for flag in ("has_sf_policy", "has_sf_wdl", "has_sf_multipv_raw"):
            group[flag][0] = 0
        helpers["sealed"](shard)
        item["shard"]["files"] = helpers["inventory"](shard)
        wrapper = read(item["producer_proof"])
        wrapper["shard_inventory_sha256"] = audit.digest(item["shard"]["files"])
        item["producer_proof"] = replace(item["producer_proof"], wrapper)
        p["generation_manifest"] = replace(p["generation_manifest"], generation)
        selected = read(p["selection"])
        selected["population"]["generation_manifest"] = p["generation_manifest"]
        p["selection"] = replace(p["selection"], selected)
        authority = read(p["input_authority"])
        authority["selection"] = p["selection"]
        authority["generation_manifest"] = p["generation_manifest"]
        source_readout = read(authority["source_audit"]["readout"])
        source_readout["generation_manifest"] = p["generation_manifest"]
        authority["source_audit"]["readout"] = replace(authority["source_audit"]["readout"], source_readout)
        p["input_authority"] = replace(p["input_authority"], authority)
    files = {name: hashlib.sha256((carrier.REPO / name).read_bytes()).hexdigest() for name in (
        "scripts/sf_banked_p0_census.py", "chess_anti_engine/moves/encode.py", "chess_anti_engine/stockfish/wdl.py")}
    p["auditor_source"] = {"revision": "2" * 40, "files": files}
    p["qualification_scope"] = "SYNTHETIC_NOT_REAL_AUTHORITY"
    census_protocol = ref(tmp_path / "fiction-census-protocol.json", p)
    audit.census(p, protocol_ref=census_protocol)
    report_ref = carrier.reference(Path(p["output_root"]) / "readout.json")
    command = ["python", "-m", "scripts.sf_banked_p0_census", "--protocol", census_protocol["path"],
               "--protocol-sha256", census_protocol["sha256"]]
    census_terminal = ref(tmp_path / "fiction-census-terminal.json", {
        "status": "PASS", "returncode": 0, "resource": "cpu", "seconds": 900,
        "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED", "command": command,
        "supervisor_sha256": "b" * 64, "known_processes": {"999999991": 1}})
    p["runtime"] = {"owner": {"sha256": "b" * 64}, "owner_seconds": 900}
    # Rebind the fictional protocol/report: no actual owner or authority is claimed.
    census_protocol = replace(census_protocol, p)
    report = read(report_ref)
    report["protocol"] = census_protocol
    report_ref = replace(report_ref, report)
    command[-1] = census_protocol["sha256"]
    terminal = read(census_terminal)
    terminal["command"] = command
    census_terminal = replace(census_terminal, terminal)
    acceptance = ref(tmp_path / "fiction-root-acceptance.json", {
        "status": "ROOT_ACCEPTS_CLOSED_FIXED_G1_CENSUS_DATA_ONLY", "scope": "SYNTHETIC_NOT_REAL_AUTHORITY",
        "readout": report_ref, "protocol": census_protocol, "terminal": census_terminal,
        "rows": report["per_row_evidence"]})
    review = ref(tmp_path / "fiction-review.json", {"status": "ACCEPT_ACTUAL_METADATA_NO_FINDINGS_DATA_ONLY",
        "findings": [], "readout": report_ref, "protocol": census_protocol, "root_acceptance": acceptance})
    selection = ref(tmp_path / "fiction-direction.json", {"status": carrier.SELECTION,
        "execution_admitted": False, "training_recipe_selected": False,
        "evidence": {"census_root": acceptance, "census_independent": review,
                     "metadata_readout": {"path": str(tmp_path / "unused-metadata.json"), "sha256": "d" * 64},
                     "metadata_independent": {"path": str(tmp_path / "unused-metadata-review.json"), "sha256": "e" * 64}},
        "scope": "SYNTHETIC_NOT_REAL_AUTHORITY"})
    protocol = {"schema": carrier.PROTOCOL, "scope": "SYNTHETIC_ONLY", "phase": "produce", "selection": selection,
        "census_protocol": census_protocol, "census_readout": report_ref, "census_acceptance": acceptance,
        "census_review": review, "census_terminal": census_terminal, "census_command": command,
        "source": {"revision": "3" * 40, "files": files}, "runtime": {"owner_seconds": 300,
            "owner": {"path": str(tmp_path / "fiction-owner.py"), "sha256": "b" * 64}},
        "limits": {"metadata_bytes": 32 << 20, "game_decoded_bytes": 32 << 20, "inventory_files": 1000,
                   "row_bytes": 1 << 20, "game_metadata_bytes": 8 << 20,
                   "carrier_game_bytes": 8 << 20, "carrier_total_bytes": 32 << 20},
        "candidate": None, "producer_owner": None, "output_root": str(tmp_path / "carrier")}
    protocol_ref = ref(tmp_path / "fiction-produce.json", protocol)
    old_rows = [json.loads(line) for line in Path(report["per_row_evidence"]["path"]).read_text().splitlines()]
    return protocol, protocol_ref, generation, old_rows


def fiction_admission(p: dict[str, Any], p_ref: dict[str, str], path: Path) -> dict[str, str]:
    if path.exists():
        path.chmod(0o644)
    return ref(path, {"status": "ROOT_ADMITS_CURRENT_P0_SF_SIDECAR_PHASE_V1", "protocol": p_ref,
        "phase": p["phase"], "scope": p["scope"], "source_revision": p["source"]["revision"],
        "source_files_sha256": audit.digest(p["source"]["files"]), "runtime": p["runtime"],
        "selection": p["selection"], "census_readout": p["census_readout"],
        "synthetic_authority_notice": "Known serialized test fiction, not actual execution admission."})


def run_producer(p: dict[str, Any], p_ref: dict[str, str], final_guard: Any = None) -> dict[str, Any]:
    p_ref = replace(p_ref, p)
    a_ref = fiction_admission(p, p_ref, Path(p_ref["path"]).with_name("fiction-produce-admission.json"))
    return carrier.produce(p, p_ref, a_ref, final_guard=final_guard)


def run_validator(p: dict[str, Any], p_ref: dict[str, str]) -> dict[str, Any]:
    p_ref = replace(p_ref, p)
    a_ref = fiction_admission(p, p_ref, Path(p_ref["path"]).with_name("fiction-validate-admission.json"))
    return carrier.validate(p, p_ref, a_ref)


def completed_candidate(tmp_path: Path, **kwargs: Any) -> tuple[dict[str, Any], dict[str, str]]:
    p, p_ref, _, _ = fiction(tmp_path, **kwargs)
    run_producer(p, p_ref)
    candidate = carrier.reference(Path(p["output_root"]) / "manifest.json")
    command = ["python", "-m", "scripts.sf_current_p0_sidecar", "--protocol", p_ref["path"],
               "--protocol-sha256", p_ref["sha256"]]
    produce_admission = carrier.reference(tmp_path / "fiction-produce-admission.json")
    command.extend(["--admission", produce_admission["path"], "--admission-sha256", produce_admission["sha256"]])
    terminal = ref(tmp_path / "fiction-produce-terminal.json", {"status": "PASS", "returncode": 0,
        "resource": "cpu", "seconds": 300, "command": command, "supervisor_sha256": "b" * 64,
        "known_processes": {"999999993": 1}, "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED"})
    v = copy.deepcopy(p)
    v.update(phase="validate", candidate=candidate, output_root=str(tmp_path / "validation"),
             producer_owner={"terminal": terminal, "command": command, "owner_sha256": "b" * 64})
    return v, ref(tmp_path / "fiction-validate.json", v)


def test_exact_policy_bytes_distinct_value_povs_and_complete_denominator(tmp_path: Path) -> None:
    p, p_ref, generation, _ = fiction(tmp_path)
    result = run_producer(p, p_ref)
    folder = Path(p["output_root"]) / "game-000000"
    arrays = {k: schema.load_npy(folder / (k + ".npy"), k, 4, 8 << 20) for k in schema.FIELDS}
    original = zarr.open_group(generation["games"][0]["shard"]["path"], mode="r")
    assert arrays["policy_f16"][1].tobytes() == np.asarray(original["sf_policy_target"][0]).tobytes()
    np.testing.assert_array_equal(arrays["native_wdl_permille_i16"][1], [700, 200, 100])
    np.testing.assert_array_equal(arrays["stored_record_wdl_f16"][1], np.array([.1, .2, .7], dtype=np.float16))
    np.testing.assert_array_equal(arrays["stored_query_wdl_f16"][1], np.array([.7, .2, .1], dtype=np.float16))
    metadata = json.loads((folder / "rows.json").read_text())
    assert metadata["rows_detail"][1]["stored_value_relation"] == "NATIVE_PERMILLE_QUANTIZATION_AND_POV_FLIP_F16"
    assert metadata["source_value_identity"]["sf_wdl_use_cp_logistic"] is False
    assert arrays["observed_predecessor_meta"][1] == 1
    assert arrays["observed_predecessor_pv"][1] == 1
    assert result["rows"] == 4
    assert result["counts"]["qualified_policy:False"] == 1
    assert result["counts"]["qualified_policy:True"] == 3
    assert result["training_admitted"] is False
    assert result["root_postclosure_acceptance_required"] is True


def test_separate_validator_reopens_original_and_complete_carrier(tmp_path: Path) -> None:
    p, p_ref = completed_candidate(tmp_path)
    result = run_validator(p, p_ref)
    assert result["status"] == "VALIDATION_CANDIDATE_CURRENT_P0_SF_SIDECAR_V1"
    assert result["rows"] == 4
    assert result["source_payload_and_vector_bytes_recomputed"] is True
    assert result["training_admitted"] is False


def test_real_direction_shape_without_audit_uuid_and_distinct_synthetic_chain(tmp_path: Path) -> None:
    p, _, _, _ = fiction(tmp_path)
    selected = read(p["selection"])
    assert "audit_uuid" not in selected
    assert set(selected["evidence"]) == {"census_root", "census_independent", "metadata_readout", "metadata_independent"}
    carrier.direction_selection(selected, {**p, "scope": "FIXED_ORIGINAL_G1"})
    # The real direction points at real-G1 receipts, whereas the synthetic
    # input has its own accepted fictional census chain. They are not aliases.
    selected["evidence"]["census_root"] = {"path": str(tmp_path / "real-G1-root.json"), "sha256": "a" * 64}
    selected["evidence"]["census_independent"] = {"path": str(tmp_path / "real-G1-review.json"), "sha256": "b" * 64}
    p["selection"] = replace(p["selection"], selected)
    # Complete metadata consumer also sees the real shape, rather than a test-
    # invented audit_uuid. Tiny source remains explicitly fictional/synthetic.
    _, census, report, generation, _ = carrier.inputs(p)
    assert census["audit_uuid"] == report["audit_uuid"]
    assert len(generation["games"]) == 1


@pytest.mark.parametrize("key", ["census_root", "census_independent"])
def test_foreign_direction_bridge_refuses_before_source_arrays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    p, p_ref, _, _ = fiction(tmp_path)
    selected = read(p["selection"])
    selected["evidence"][key] = {"path": str(tmp_path / "foreign.json"), "sha256": "f" * 64}
    p["selection"] = replace(p["selection"], selected)
    p["scope"] = "FIXED_ORIGINAL_G1"
    monkeypatch.setattr(audit, "open_arrays", lambda *_: pytest.fail("payload before direction bridge"))
    with pytest.raises(ValueError, match="direction census acceptance/review bridge"):
        run_producer(p, p_ref)
    assert not Path(p["output_root"]).exists()


def test_thinned_retained_rows_are_not_actual_ply_predecessors(tmp_path: Path) -> None:
    p, p_ref, _, _ = fiction(tmp_path, kept=(0, 2, 3))
    result = run_producer(p, p_ref)
    assert result["rows"] == 3
    assert result["counts"]["status:UNAVAILABLE"] == 2
    assert result["counts"]["qualified_policy:True"] == 1


@pytest.mark.parametrize("corrupt", ["native_flip", "missing_flag", "duplicate_PV"])
def test_invalid_rows_keep_observations_but_no_qualified_masks(tmp_path: Path, corrupt: str) -> None:
    p, p_ref, _, _ = fiction(tmp_path, corrupt=corrupt)
    result = run_producer(p, p_ref)
    metadata = json.loads((Path(p["output_root"]) / "game-000000/rows.json").read_text())
    row = metadata["rows_detail"][1]
    assert row["source_join"]["status"] == "INVALID_REDUCED_LABEL"
    assert row["source_invalid_reasons"]
    assert not any(row["qualified_flags"].values())
    assert row["observed_flags"]["policy"] is True
    assert result["rows"] == 4


def test_no_reduced_label_can_still_supply_native_observation(tmp_path: Path) -> None:
    p, p_ref, _, _ = fiction(tmp_path, no_reduced_native=True)
    result = run_producer(p, p_ref)
    metadata = json.loads((Path(p["output_root"]) / "game-000000/rows.json").read_text())
    row = metadata["rows_detail"][1]
    assert row["source_join"]["status"] == "SOURCE_CONTEXT_NO_REDUCED_LABEL"
    assert row["qualified_flags"] == {"policy": False, "stored_wdl": False, "native_wdl": True}
    assert result["rows"] == 4


def test_cp_logistic_stored_value_is_not_native_permille(tmp_path: Path) -> None:
    p, p_ref, _, _ = fiction(tmp_path, corrupt="cp_logistic")
    run_producer(p, p_ref)
    folder = Path(p["output_root"]) / "game-000000"
    native = schema.load_npy(folder / "native_wdl_permille_i16.npy", "native_wdl_permille_i16", 4, 8 << 20)
    stored = schema.load_npy(folder / "stored_query_wdl_f16.npy", "stored_query_wdl_f16", 4, 8 << 20)
    np.testing.assert_array_equal(native[1], [700, 200, 100])
    assert not np.allclose(stored[1], native[1] / 1000, atol=.002)
    metadata = json.loads((folder / "rows.json").read_text())
    assert metadata["rows_detail"][1]["stored_value_relation"] == "CP_LOGISTIC_THEN_POV_FLIP_F16"


@pytest.mark.parametrize("fault", ["acceptance", "review", "command", "hash", "role"])
def test_authority_or_role_refusal_precedes_array_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    p, p_ref, _, _ = fiction(tmp_path)
    if fault == "acceptance":
        p["census_acceptance"] = replace(p["census_acceptance"], {"status": "PENDING"})
    elif fault == "review":
        p["census_review"] = replace(p["census_review"], {"status": "PENDING"})
    elif fault == "command":
        p["census_command"] = ["foreign"]
    elif fault == "hash":
        p["census_readout"] = {**p["census_readout"], "sha256": "0" * 64}
    else:
        p["scope"] = "FIXED_ORIGINAL_G1"
    monkeypatch.setattr(audit, "open_arrays", lambda *_: pytest.fail("arrays opened before refusal"))
    with pytest.raises((ValueError, KeyError)):
        run_producer(p, p_ref)
    assert not Path(p["output_root"]).exists()


@pytest.mark.parametrize("fault", ["missing", "wrong_protocol", "wrong_source", "wrong_phase", "pending"])
def test_exact_phase_admission_precedes_source_arrays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    p, p_ref, _, _ = fiction(tmp_path)
    a_ref = fiction_admission(p, p_ref, tmp_path / "fiction-bad-admission.json")
    admission = read(a_ref)
    if fault == "missing":
        a_ref = {"path": str(tmp_path / "missing.json"), "sha256": "0" * 64}
    else:
        if fault == "wrong_protocol":
            admission["protocol"] = {**p_ref, "sha256": "0" * 64}
        elif fault == "wrong_source":
            admission["source_revision"] = "4" * 40
        elif fault == "wrong_phase":
            admission["phase"] = "validate"
        else:
            admission["status"] = "PENDING"
        a_ref = replace(a_ref, admission)
    monkeypatch.setattr(audit, "open_arrays", lambda *_: pytest.fail("source payload before phase admission"))
    with pytest.raises((ValueError, FileNotFoundError)):
        carrier.produce(p, p_ref, a_ref)
    assert not Path(p["output_root"]).exists()


def test_source_inventory_mutation_is_hard_failure(tmp_path: Path) -> None:
    p, _, generation, old_rows = fiction(tmp_path)
    _, producer = audit.generation_inputs(read(p["census_protocol"]))
    path = next(Path(generation["games"][0]["shard"]["path"]).rglob(".zarray"))
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    path.chmod(0o444)
    with pytest.raises(ValueError, match="hash/size"):
        carrier.expected_game(audit, generation["games"][0], generation, producer, p["limits"], old_rows)


def test_wrong_game_history_and_input_mirror_identity_refuse(tmp_path: Path) -> None:
    p, _, generation, old_rows = fiction(tmp_path)
    _, producer = audit.generation_inputs(read(p["census_protocol"]))
    for key, changed in [("game_id", 88), ("input_identity", {**old_rows[0]["input_identity"], "history_sha256": "0" * 64}),
                         ("input_identity", {**old_rows[0]["input_identity"], "STM_white": False}),
                         ("input_identity", {**old_rows[0]["input_identity"], "halfmove_clock": 1}),
                         ("input_identity", {**old_rows[0]["input_identity"], "input_F16_sha256": "f" * 64})]:
        corrupt = copy.deepcopy(old_rows)
        corrupt[0][key] = changed
        with pytest.raises(ValueError, match="census/source"):
            carrier.expected_game(audit, generation["games"][0], generation, producer, p["limits"], corrupt)


def test_semantic_vector_tamper_refuses_even_with_rehashed_inventory(tmp_path: Path) -> None:
    p, _ = completed_candidate(tmp_path)
    manifest_path = Path(p["candidate"]["path"])
    candidate = read(p["candidate"])
    path = manifest_path.parent / "game-000000/policy_f16.npy"
    value = np.load(path, allow_pickle=False)
    value[1] = np.roll(value[1], 1)
    path.chmod(0o644)
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    path.chmod(0o444)
    for item in candidate["files"]:
        if item["path"] == "game-000000/policy_f16.npy":
            item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    p["candidate"] = replace(p["candidate"], candidate)
    p_ref = ref(tmp_path / "fiction-rehashed-validate.json", p)
    with pytest.raises(ValueError, match=r"qualified carrier policy|vector bytes"):
        run_validator(p, p_ref)
    assert Path(p["output_root"] + ".failed.json").exists()


@pytest.mark.parametrize("failure", ["failed_owner", "late_marker", "unknown_file"])
def test_candidate_owner_and_seal_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    p, _ = completed_candidate(tmp_path)
    root = Path(p["candidate"]["path"]).parent
    if failure == "failed_owner":
        terminal = read(p["producer_owner"]["terminal"])
        terminal["status"] = "FAILED"
        p["producer_owner"]["terminal"] = replace(p["producer_owner"]["terminal"], terminal)
        monkeypatch.setattr(carrier, "candidate_inventory", lambda *_: pytest.fail("carrier payload before owner closure"))
    elif failure == "late_marker":
        root.with_name(root.name + ".failed.json").write_text("{}")
    else:
        root.chmod(0o755)
        (root / "unexpected").write_text("unlisted")
        root.chmod(0o555)
    expected_message = {"failed_owner": "not exact closed CPU owner",
                        "late_marker": "candidate seal/failure marker",
                        "unknown_file": "candidate complete inventory"}[failure]
    with pytest.raises(ValueError, match=expected_message):
        run_validator(p, ref(tmp_path / "fiction-failure-validate.json", p))
    assert not Path(p["output_root"]).exists()


def test_matching_terminal_owner_cannot_override_producer_runtime_pin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p, p_ref = completed_candidate(tmp_path)
    owner = p["producer_owner"]
    terminal = read(owner["terminal"])
    owner["owner_sha256"] = "c" * 64
    terminal["supervisor_sha256"] = "c" * 64
    owner["terminal"] = replace(owner["terminal"], terminal)
    assert terminal["supervisor_sha256"] == owner["owner_sha256"]
    monkeypatch.setattr(carrier, "candidate_inventory", lambda *_: pytest.fail("carrier payload before runtime-owner join"))
    with pytest.raises(ValueError, match="pinned runtime owner"):
        run_validator(p, p_ref)
    assert not Path(p["output_root"]).exists()


def test_unsafe_npy_header_refuses_before_numpy_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "bad.npy"
    header = repr({"descr": "|O", "fortran_order": False, "shape": (1, 1858)}).encode() + b"\n"
    p.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
    monkeypatch.setattr(np, "load", lambda *_a, **_k: pytest.fail("unsafe load reached"))
    with pytest.raises(ValueError, match="header contract"):
        schema.load_npy(p, "policy_f16", 1, 8 << 20)


def test_decoded_carrier_cap_precedes_new_numeric_allocation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p, _, generation, old_rows = fiction(tmp_path)
    _, producer = audit.generation_inputs(read(p["census_protocol"]))
    cap = {**p["limits"], "carrier_game_bytes": 1}
    monkeypatch.setattr(schema, "arrays_for_rows", lambda *_: pytest.fail("carrier allocation exceeded cap"))
    with pytest.raises(ValueError, match="allocation cap"):
        carrier.expected_game(audit, generation["games"][0], generation, producer, cap, old_rows)


@pytest.mark.parametrize("marker_kind", ["directory", "file"])
def test_output_inside_another_checkout_refuses_before_mkdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker_kind: str) -> None:
    p, p_ref, _, _ = fiction(tmp_path)
    checkout = tmp_path / "different-checkout"
    checkout.mkdir()
    marker = checkout / ".git"
    if marker_kind == "directory":
        marker.mkdir()
    else:
        marker.write_text("gitdir: /synthetic/external/common/worktrees/other\n")
    output = checkout / "uncreated-parent" / "new-carrier"
    p["output_root"] = str(output)
    monkeypatch.setattr(audit, "open_arrays", lambda *_: pytest.fail("source arrays before checkout-output refusal"))
    with pytest.raises(ValueError, match="inside a Git checkout"):
        run_producer(p, p_ref)
    assert not output.exists()
    assert not output.parent.exists()


def test_output_input_overlap_and_late_failure_marker(tmp_path: Path) -> None:
    p, p_ref, generation, _ = fiction(tmp_path)
    p["output_root"] = str(Path(generation["games"][0]["shard"]["path"]) / "new-output")
    with pytest.raises(ValueError, match="overlap"):
        run_producer(p, p_ref)
    p["output_root"] = str(tmp_path / "late-seal")
    def fail() -> None:
        raise RuntimeError("controlled final guard refusal")
    with pytest.raises(RuntimeError, match="controlled"):
        run_producer(p, p_ref, final_guard=fail)
    assert Path(p["output_root"] + ".failed.json").exists()
    assert not (Path(p["output_root"]) / "manifest.json").exists()
