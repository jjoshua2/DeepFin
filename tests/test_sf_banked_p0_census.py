"""Known synthetic producer documents; no engine, model or outcome re-probe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chess
import numpy as np
import pytest
import zarr

from chess_anti_engine.moves import move_to_index, move_to_index_for_encoding
from scripts import sf_banked_p0_census as audit


def write_ref(path: Path, value: Any) -> dict[str, str]:
    path.write_bytes(audit.canonical(value))
    path.chmod(0o444)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def inventory(root: Path) -> list[dict[str, Any]]:
    return [{"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(root.rglob("*")) if p.is_file()]


def sealed(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def fixture(tmp_path: Path, *, kept: tuple[int, ...] = (0, 1, 2, 3),
            corrupt: str = "") -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Fool's mate: all actions/history known; synthetic zero input fingerprints.

    These are serialized accepted-document fixtures, not newly issued TB proofs.
    The census authenticates prior proof bytes; it does not reissue their meaning.
    """
    root = tmp_path / "input"
    root.mkdir()
    directory = root / "game-0"
    directory.mkdir()
    shard = directory / "shard"
    board = chess.Board()
    actions, records, positions, queries, played = [], [], [], [], []
    for offset, uci in enumerate(("f2f3", "e7e5", "g2g4", "d8h4")):
        positions.append(board.copy(stack=True))
        records.append({"move_offset": offset, "ply_index": board.ply(), "pov_color": board.turn,
                        "has_policy": True, "is_sf_refute_opp": False})
        move = chess.Move.from_uci(uci)
        actions.append(move_to_index(move, board))
        board.push(move)
        queries.append(board.copy(stack=True))
        played.append(uci)
    assert board.is_checkmate()
    n = len(kept)
    arrays: dict[str, np.ndarray] = {
        "x": np.zeros((n, 175, 8, 8), dtype=np.float16),
        "legal_mask": np.zeros((n, 1858), dtype=np.uint8),
        "game_id": np.full(n, 7, dtype=np.int64),
        "ply_index": np.array(kept, dtype=np.int32),
    }
    for key in ("has_game_id", "has_ply_index", "has_policy", "has_legal_mask", "is_network_turn",
                "has_is_network_turn", "is_selfplay", "has_is_selfplay"):
        arrays[key] = np.ones(n, dtype=np.uint8)
    for name, (flag, dtype, tail) in audit.SF_FIELDS.items():
        arrays[name] = np.zeros((n, *tail), dtype=dtype)
        arrays[flag] = np.zeros(n, dtype=np.uint8)
    arrays["sf_multipv_raw"][:] = audit.PAD
    rows = []
    for i, offset in enumerate(kept):
        pos = positions[offset]
        legal = [move_to_index_for_encoding(m, pos) for m in pos.legal_moves]
        arrays["legal_mask"][i, legal] = 1
        qlegal = [move_to_index_for_encoding(m, queries[offset]) for m in queries[offset].legal_moves]
        if qlegal:
            for flag in ("has_sf_wdl", "has_sf_policy", "has_sf_legal_mask", "has_sf_multipv_raw", "has_sf_label_meta", "has_sf_move"):
                arrays[flag][i] = 1
            arrays["sf_wdl"][i] = [.1, .2, .7]  # Record POV; query native is [.7,.2,.1].
            arrays["sf_policy_target"][i, qlegal[0]] = 1
            arrays["sf_legal_mask"][i, qlegal] = 1
            arrays["sf_multipv_raw"][i, 0] = [qlegal[0], 137, 0, 700, 200]
            arrays["sf_label_meta"][i] = [5000, 12, 137, 0, 700, 200]
            arrays["sf_move_index"][i] = qlegal[0]
        if offset > 0:
            arrays["has_sf_p0_regret"][i] = 1
            arrays["sf_p0_regret"][i] = .5  # Regret is not a probability distribution.
        rows.append({"physical_row_id": "synthetic:0:" + str(i), "ply": offset, "move_offset": offset,
                     "history_sha256": audit.digest([chess.STARTING_FEN, played[:offset]]),
                     "stored_input_float16_sha256": hashlib.sha256(arrays["x"][i].tobytes()).hexdigest()})
    if corrupt == "game":
        arrays["game_id"][0] = 8
    elif corrupt == "NET":
        arrays["is_network_turn"][0] = 0
    elif corrupt == "legal":
        arrays["legal_mask"][0] = 0
    elif corrupt == "pov":
        records[kept[0]]["pov_color"] = False
    elif corrupt == "history":
        rows[0]["history_sha256"] = "0" * 64
    elif corrupt == "input":
        arrays["x"][0, 0, 0, 0] = 1
    elif corrupt == "label_dtype":
        arrays["sf_wdl"] = arrays["sf_wdl"].astype(np.float32)
    elif corrupt == "fallback":
        arrays["sf_multipv_raw"][0] = audit.PAD
    elif corrupt == "missing":
        del arrays["sf_wdl"]
    elif corrupt == "duplicate_PV":
        arrays["sf_multipv_raw"][0, 1] = arrays["sf_multipv_raw"][0, 0]
    elif corrupt == "native_flip":
        arrays["sf_wdl"][0] = [.7, .2, .1]
    elif corrupt == "absent_all":
        for flag, _, _ in audit.SF_FIELDS.values():
            arrays[flag][0] = 0
    elif corrupt == "duplicate_row":
        rows[1]["ply"] = rows[0]["ply"]
    elif corrupt == "cp_logistic":
        from chess_anti_engine.stockfish.wdl import cp_to_wdl

        arrays["sf_wdl"][0] = cp_to_wdl(137, None, slope=.006, draw_width_cp=120)[::-1]
    group = zarr.open_group(str(shard), mode="w")
    group.attrs.update(policy_encoding="lc0_1858", input_history_encoding="lc0_root_legacy_meta", history_rep_fix=True)
    for name, value in arrays.items():
        group.create_dataset(name, data=value, compressor=None)
    sealed(shard)
    files = inventory(shard)
    common = {"run_uuid": "synthetic-run", "source_revision": "1" * 40, "generation": 1,
              "actor_sha256": "a" * 64, "physical_game_id": "synthetic:0", "game_id": 7,
              "rows": n, "bank": "train", "served_state_sha256": "c" * 64}
    actual = {"game_id": 7, "start": {"root_fen": chess.STARTING_FEN, "opening_moves": [],
              "start_fen": chess.STARTING_FEN, "start_ply": 0, "is_selfplay": True},
              "played_move_indices": actions, "records": records, "final_fen": board.fen(),
              "retained_samples": n, "result": "0-1"}
    verification = {**common, "resolved": True, "retained_rows": n, "audit_sha256": audit.digest(actual),
                    "scientific_class": "NATURAL"}
    proof = {**common, "schema": "CORE_PRODUCING_OUTCOME_PROOF_V1", "verification": verification,
             "proof_sha256": audit.digest(verification), "scientific_class": "NATURAL"}
    provenance = {**common, "actual": actual, "planned": {"physical_game_id": "synthetic:0"}}
    validation = {**common, "full_history_verified": True, "raw_targets_verified": True,
                  "usable_shard": True, "rows_detail": rows, "verified_outcome_sha256": proof["proof_sha256"]}
    evidence = {"outcome_proof": write_ref(directory / "outcome-proof.json", proof),
                "provenance": write_ref(directory / "provenance.json", provenance),
                "validation": write_ref(directory / "validation.json", validation)}
    wrapper = {**common, "status": "VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1", "resolved": True,
               "full_history_verified": True, "raw_targets_verified": True,
               "shard_inventory_sha256": audit.digest(files), "evidence": evidence}
    item = {"physical_game_id": "synthetic:0", "game_id": 7, "bank": "train", "label_class": "NATURAL",
            "rows": n, "producer_proof": write_ref(root / "game-0.json", wrapper),
            "shard": {"path": str(shard), "files": files}}
    generation: dict[str, Any] = {**{k: common[k] for k in ("run_uuid", "source_revision", "generation", "actor_sha256")},
                                  "schema": "ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1", "games": [item]}
    generation["actor_serving"] = write_ref(root / "serving.json", {"actor_sha256": common["actor_sha256"],
                                   "pre_state_sha256": "c" * 64, "post_state_sha256": "c" * 64,
                                   "checkpoint_model_state_sha256": "c" * 64})
    gen_ref = write_ref(root / "generation.json", generation)
    game = {"policy_encoding": "lc0_1858", "history_rep_fix": True, "selfplay_fraction": 1.0,
            "sf_wdl_use_cp_logistic": corrupt == "cp_logistic", "sf_wdl_cp_slope": .006, "sf_wdl_cp_draw_width": 120.,
            "record_sf_p0_policy": False, "record_sf_p0_regret": True}
    producer = {"source": {"revision": common["source_revision"]}, "run_uuid": common["run_uuid"], "generation": 1,
                "actor": {"sha256": common["actor_sha256"]}, "configs": {"game": game},
                "owner": {"sha256": "b" * 64, "seconds": 3600}, "output_root": str(root)}
    producer_ref = write_ref(root / "producer.json", producer)
    admission_ref = write_ref(root / "core-admission.json", {"status": "ROOT_ADMITS_CORE_GENERATION_V1",
                              "protocol_sha256": producer_ref["sha256"], "actor_sha256": common["actor_sha256"]})
    command = ["python", "-m", "scripts.core_finite_collect", "--execute", "--protocol", producer_ref["path"],
               "--protocol-sha256", producer_ref["sha256"], "--admission", admission_ref["path"], "--admission-sha256", admission_ref["sha256"]]
    terminal = {"status": "PASS", "returncode": 0, "owned_processes_closed": True, "custody": "OWNED_PROCESSES_CLOSED",
                "command": command, "supervisor_sha256": "b" * 64, "seconds": 3600, "resource": "gpu",
                "known_processes": {"999999991": 1}}
    protocol = {"schema": "SF_BANKED_P0_CENSUS_V1", "audit_uuid": "f47bab16-fca5-46b8-bb7b-d1d2d3dff172",
                "generation_manifest": gen_ref, "producer_protocol": producer_ref,
                "expected": {**{k: common[k] for k in ("run_uuid", "source_revision", "generation")},
                             "physical_games": 1, "rows": n},
                "producer_owner": {"terminal": write_ref(root / "terminal.json", terminal),
                                   "owner_sha256": "b" * 64, "command": command},
                "output_root": str(tmp_path / "report"),
                "limits": {"metadata_bytes": 32 << 20, "game_decoded_bytes": 32 << 20,
                           "inventory_files": 1000, "report_bytes": 8 << 20}}
    return protocol, generation, producer


def rows_of(protocol: dict[str, Any], generation: dict[str, Any], producer: dict[str, Any]) -> list[dict[str, Any]]:
    return list(audit.inspect_game(generation["games"][0], generation, producer, protocol["limits"]))


def test_actual_ply_gap_is_not_retained_row_adjacency(tmp_path: Path) -> None:
    protocol, generation, producer = fixture(tmp_path, kept=(0, 2, 3))
    rows = rows_of(protocol, generation, producer)
    assert rows[1]["retained_predecessor_P1_join"]["status"] == "UNAVAILABLE"
    assert rows[1]["raw_actual_ply_predecessor"] is True
    assert rows[1]["explicit_stored_P0"]["regret_present"] is True
    assert rows[2]["retained_predecessor_P1_join"]["predecessor_row"] == "synthetic:0:1"


def test_native_permille_and_pov_flip_are_distinct(tmp_path: Path) -> None:
    protocol, generation, producer = fixture(tmp_path)
    row = rows_of(protocol, generation, producer)[0]
    assert row["own_record_P1"]["native_query_STM_WDL_permille"] == [700, 200, 100]
    assert row["own_record_P1"]["WDL_relation"] == "NATIVE_PERMILLE_QUANTIZATION_AND_POV_FLIP_F16"
    assert row["own_record_P1"]["invalid_reasons"] == []
    assert row["own_record_P1"]["bound_flags"] == audit.UNKNOWN
    assert row["own_record_P1"]["multipv"]["full_legal_support"] is False


def test_declared_existing_cp_logistic_relation_is_not_native_WDL(tmp_path: Path) -> None:
    protocol, generation, producer = fixture(tmp_path, corrupt="cp_logistic")
    label = rows_of(protocol, generation, producer)[0]["own_record_P1"]
    assert label["WDL_relation"] == "CP_LOGISTIC_THEN_POV_FLIP_F16"
    assert "sf_wdl:cp_logistic_flip_inconsistent" not in label["invalid_reasons"]
    assert label["native_query_STM_WDL_permille"] == [700, 200, 100]


def test_retained_predecessor_without_payload_is_not_coverage(tmp_path: Path) -> None:
    protocol, generation, producer = fixture(tmp_path, corrupt="absent_all")
    rows = rows_of(protocol, generation, producer)
    assert rows[1]["retained_predecessor_P1_join"]["status"] == "SOURCE_CONTEXT_NO_REDUCED_LABEL"
    assert rows[1]["retained_predecessor_P1_join"]["stored_value_available"] is False


@pytest.mark.parametrize(("corrupt", "match"), [("game", "ply/POV/game"), ("NET", "NET/selfplay"),
                                         ("legal", "legal mask"), ("pov", "raw-record ply/POV"),
                                         ("history", "history fingerprint"), ("input", "input/history"),
                                         ("duplicate_row", "duplicate retained row")])
def test_identity_corruption_is_technical_failure(tmp_path: Path, corrupt: str, match: str) -> None:
    protocol, generation, producer = fixture(tmp_path, corrupt=corrupt)
    with pytest.raises(ValueError, match=match):
        rows_of(protocol, generation, producer)


@pytest.mark.parametrize(("corrupt", "reason"), [("missing", "flag_without_payload"), ("label_dtype", "storage_dtype"),
                                          ("duplicate_PV", "duplicate_move"), ("native_flip", "flip_inconsistent")])
def test_invalid_labels_keep_the_denominator(tmp_path: Path, corrupt: str, reason: str) -> None:
    protocol, generation, producer = fixture(tmp_path, corrupt=corrupt)
    rows = rows_of(protocol, generation, producer)
    assert len(rows) == 4
    assert any(reason in r for r in rows[0]["own_record_P1"]["invalid_reasons"])


def test_fallback_possible_is_not_wire_provenance(tmp_path: Path) -> None:
    protocol, generation, producer = fixture(tmp_path, corrupt="fallback")
    label = rows_of(protocol, generation, producer)[0]["own_record_P1"]
    assert "policy_without_legal_PV:fallback_possible_not_proven" in label["reasons"]
    assert label["fallback_provenance"] == audit.UNKNOWN
    assert label["multipv"]["support"] == 0


def test_wrong_hash_refuses_before_arrays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol, generation, producer = fixture(tmp_path)
    generation["games"][0]["producer_proof"]["sha256"] = "0" * 64
    monkeypatch.setattr(audit, "open_arrays", lambda *_: pytest.fail("arrays opened before proof authentication"))
    with pytest.raises(ValueError, match="hash mismatch"):
        rows_of(protocol, generation, producer)


@pytest.mark.parametrize("change", ["role", "source", "duplicate"])
def test_generation_population_refuses_before_output(tmp_path: Path, change: str) -> None:
    protocol, generation, _ = fixture(tmp_path)
    if change == "role":
        generation["games"][0]["bank"] = "holdout"
    elif change == "source":
        protocol["expected"]["source_revision"] = "2" * 40
    else:
        generation["games"].append(generation["games"][0])
        protocol["expected"].update(physical_games=2, rows=8)
    path = Path(protocol["generation_manifest"]["path"])
    path.chmod(0o644)
    protocol["generation_manifest"] = write_ref(path, generation)
    with pytest.raises(ValueError, match=r"nontraining|identity/source|duplicate"):
        audit.census(protocol)
    assert not Path(protocol["output_root"]).exists()


def test_complete_census_and_pergame_counts(tmp_path: Path) -> None:
    protocol, _, _ = fixture(tmp_path)
    report = audit.census(protocol)
    assert report["rows"] == 4
    assert report["physical_games"] == 1
    assert report["games"][0]["overlapping_reason_and_distribution_counts"]["rows"] == 4
    assert report["games"][0]["structure"]["raw_record_count"] == 4
    assert report["overlapping_reason_counts"]["join:SOURCE_CONTEXT_REDUCED_LABEL"] == 3
    assert sum(v for k, v in report["overlapping_reason_counts"].items() if k.startswith("join:")) == 4
    assert len(Path(report["per_row_evidence"]["path"]).read_text().splitlines()) == 4
    assert report["training_admitted"] is False
    assert Path(protocol["output_root"]).stat().st_mode & 0o222 == 0


def test_late_guard_failure_preserves_unqualified_attempt(tmp_path: Path) -> None:
    protocol, _, _ = fixture(tmp_path)
    def decline() -> None:
        raise ValueError("changed source")
    with pytest.raises(ValueError, match="changed source"):
        audit.census(protocol, final_guard=decline)
    output = Path(protocol["output_root"])
    assert not (output / "readout.json").exists()
    assert len((output / "rows.jsonl").read_text().splitlines()) == 4
    assert output.with_name(output.name + ".failed.json").exists()


def test_writable_or_changed_shard_refuses(tmp_path: Path) -> None:
    protocol, generation, producer = fixture(tmp_path)
    shard = generation["games"][0]["shard"]
    path = Path(shard["path"]) / shard["files"][-1]["path"]
    path.chmod(0o644)
    with pytest.raises(ValueError, match="writable"):
        rows_of(protocol, generation, producer)


def test_changed_payload_hash_refuses_without_decoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol, generation, producer = fixture(tmp_path)
    shard = generation["games"][0]["shard"]
    path = Path(shard["path"]) / shard["files"][-1]["path"]
    path.chmod(0o644)
    path.write_bytes(b"changed")
    path.chmod(0o444)
    monkeypatch.setattr(audit, "open_arrays", lambda *_: pytest.fail("changed inventory reached decoder"))
    with pytest.raises(ValueError, match="hash/size differs"):
        rows_of(protocol, generation, producer)


def test_clipping_padding_and_partial_support_remain_explicit() -> None:
    raw = np.tile(np.array(audit.PAD, dtype=np.int16), (48, 1))
    raw[0] = [10, 32000, 127, -1, -1]
    result = audit.sparse_evidence(raw, {10, 11})
    assert result["support"] == 1
    assert result["full_legal_support"] is False
    assert result["candidates"][0]["cp_at_clip_boundary"] is True
    assert result["candidates"][0]["native_query_STM_WDL_permille"] is None
    assert result["bounds"] == audit.UNKNOWN


def test_object_codec_refuses_before_zarr_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protocol, generation, _ = fixture(tmp_path)
    shard = generation["games"][0]["shard"]
    path = Path(shard["path"]) / "sf_wdl" / ".zarray"
    value = json.loads(path.read_bytes())
    value["filters"] = [{"id": "pickle"}]
    path.chmod(0o644)
    path.write_text(json.dumps(value))
    path.chmod(0o444)
    monkeypatch.setattr(zarr, "open_group", lambda *_args, **_kwargs: pytest.fail("unsafe metadata reached decoder"))
    with pytest.raises(ValueError, match="unsafe dtype/filter"):
        audit.open_arrays(shard, 4, protocol["limits"]["game_decoded_bytes"])


def test_input_allocation_cap_refuses_before_zarr_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, generation, _ = fixture(tmp_path)
    monkeypatch.setattr(zarr, "open_group", lambda *_args, **_kwargs: pytest.fail("oversized allocation reached decoder"))
    with pytest.raises(ValueError, match="allocation cap"):
        audit.open_arrays(generation["games"][0]["shard"], 4, 1)
