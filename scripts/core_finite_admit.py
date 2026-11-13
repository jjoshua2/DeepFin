"""Independent CPU audit of one closed finite collector generation.

Outputs are accepted-schema CANDIDATES. Root promotes their pinned bytes only
after this audit's own owner closes PASS; the child cannot attest future closure.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace
from typing import Any
import zipfile

from scripts import core_finite_learner as learning

require = learning.require
canonical = learning.canonical
digest = learning.digest
METADATA_CAP = 128 << 20


def ref(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": learning.file_sha(path)}


def check_owner(binding: dict[str, Any], protocol_ref: dict[str, str], producer: dict[str, Any]) -> dict[str, Any]:
    require(set(binding) == {"terminal", "owner_sha256", "command"}, "invalid producer owner binding")
    _, terminal = learning.read_ref(binding["terminal"], readonly=False)
    command = binding["command"]
    suffix = ["-m", "scripts.core_finite_collect", "--execute", "--protocol", protocol_ref["path"],
              "--protocol-sha256", protocol_ref["sha256"]]
    # --admission is mandatory on the real collector and is itself hash-pinned.
    require(isinstance(command, list) and all(isinstance(x, str) and x for x in command), "invalid producer command")
    require(len(command) > len(suffix) + 4 and command[-4] == "--admission" and
            command[-2] == "--admission-sha256" and command[-(len(suffix) + 4):-4] == suffix,
            "producer command does not invoke the exact protocol")
    _, admission = learning.read_ref({"path": command[-3], "sha256": command[-1]})
    require(admission["status"] == "ROOT_ADMITS_CORE_GENERATION_V1" and
            admission["protocol_sha256"] == protocol_ref["sha256"], "producer stage admission changed")
    require(terminal.get("status") == "PASS" and type(terminal.get("returncode")) is int
            and terminal["returncode"] == 0 and terminal.get("owned_processes_closed") is True
            and terminal.get("custody") == "OWNED_PROCESSES_CLOSED"
            and terminal.get("supervisor_sha256") == binding["owner_sha256"]
            and binding["owner_sha256"] == producer["owner"]["sha256"]
            and terminal.get("resource") == "gpu" and terminal.get("seconds") == producer["owner"]["seconds"]
            and terminal.get("command") == command, "producer owner not exact closed PASS")
    known = terminal["known_processes"]
    require(isinstance(known, dict) and bool(known), "producer owned-birth inventory absent")
    for pid, birth in known.items():
        require(str(pid).isdigit() and type(birth) is int and birth > 0, "invalid producer birth identity")
        try:
            current = int(Path("/proc", str(pid), "stat").read_text().rsplit(")", 1)[1].split()[19])
        except FileNotFoundError:
            current = None
        require(current != birth, "producer birth still alive")
    return admission


def gates(protocol: dict[str, Any], contract: dict[str, Any], producer: dict[str, Any],
          eligible: list[dict[str, Any]] | None) -> None:
    value = protocol["gates"]
    require(set(value) == {"attempts", "min_resolved_games", "min_total_rows"}, "invalid admission gates")
    for key, number in value.items():
        learning.integer(number, key)
    require(value["attempts"] == producer["attempts"] == contract["max_games_per_generation"]
            and value["min_resolved_games"] == contract["games_per_generation"], "attempt/min/max contract mismatch")
    if eligible is None:
        return
    require(value["min_resolved_games"] <= len(eligible) <= value["attempts"], "insufficient resolved nonempty games")
    require(sum(game["rows"] for game in eligible) >= value["min_total_rows"], "insufficient total eligible rows")


def runtime_guard(protocol: dict[str, Any]) -> None:
    runtime = protocol["runtime"]
    require(set(runtime) == {"owner", "seconds", "inner_seconds", "cpu_affinity", "python"}, "invalid CPU runtime")
    require(0 < learning.integer(runtime["inner_seconds"], "inner seconds") <
            learning.integer(runtime["seconds"], "owner seconds") <= 7200, "invalid finite CPU bounds")
    owner = learning.external(runtime["owner"]["path"])
    require(learning.file_sha(owner) == runtime["owner"]["sha256"], "audit owner source changed")
    args = Path("/proc", str(os.getppid()), "cmdline").read_bytes().decode().split("\0")
    require(str(owner) in args and "--resource" in args and "--seconds" in args
            and args[args.index("--resource") + 1] == "cpu"
            and args[args.index("--seconds") + 1] == str(runtime["seconds"]), "wrong finite audit owner")
    require(Path(sys.executable).resolve() == Path(runtime["python"]).resolve()
            and sorted(os.sched_getaffinity(0)) == runtime["cpu_affinity"] and len(runtime["cpu_affinity"]) == 2
            and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "2"
            and __debug__ and os.environ.get("PYTHONOPTIMIZE", "0") == "0",
            "audit CPU/visibility context changed")


def restore_audit(value: dict[str, Any]) -> Any:
    from chess_anti_engine.selfplay.provenance import GameProvenance, GameStartProvenance, NetRecordProvenance
    from chess_anti_engine.selfplay.strict_tb import TablebaseWitness

    fields = dict(value)
    start = dict(fields["start"])
    start["opening_moves"] = tuple(start["opening_moves"])
    fields["start"] = GameStartProvenance(**start)
    fields["records"] = tuple(NetRecordProvenance(**row) for row in fields["records"])
    for key in ("played_move_indices", "tablebase_policy_applied_plies", "tablebase_policy_abstained_plies"):
        fields[key] = tuple(fields[key])
    witness = dict(fields["tablebase_witness"])
    witness["moves"] = tuple(witness["moves"])
    fields["tablebase_witness"] = TablebaseWitness(**witness)
    audit = GameProvenance(**fields)
    require(canonical(asdict(audit)) == canonical(value), "audit reconstruction defaulted or changed fields")
    return audit


def load_raw(path: Path, cap: int) -> dict[str, Any]:
    import numpy as np

    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == 8 and len({entry.filename for entry in entries}) == 8
                and sum(entry.file_size for entry in entries) <= cap, "raw archive decoded bound exceeded")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    require(set(result) == {"physical_row_id", "policy_target_raw", "outcome_wdl", "pov_white", "search_wdl_raw",
                            "has_search_raw", "categorical_raw", "has_categorical_raw"}, "raw archive key set differs")
    require(not any(array.dtype.hasobject for array in result.values())
            and sum(array.nbytes for array in result.values()) <= cap, "raw archive object/decoded bound")
    return result


def shard_decoded_bound(path: Path, rows: int, cap: int) -> None:
    """Bound physical Zarr arrays before the public eager loader allocates them."""
    import numpy as np

    require(path.is_dir() and (path / ".zgroup").is_file(), "producer shard must be public Zarr layout")
    total = 0
    for metadata in path.rglob(".zarray"):
        value = json.loads(metadata.read_bytes())
        shape = value["shape"]
        require(isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape), "invalid shard shape")
        count = 1
        for size in shape:
            count *= size
        dtype = np.dtype(value["dtype"])
        require(not dtype.hasobject, "object shard refused")
        chunks = value["chunks"]
        require(len(chunks) == len(shape) and all(type(size) is int and size > 0 for size in chunks), "invalid shard chunk layout")
        chunk_count = 1
        for size, width in zip(shape, chunks, strict=True):
            chunk_count *= ((size + width - 1) // width) * width
        total += max(count, chunk_count) * dtype.itemsize
        require(total <= cap, "declared shard decoded bound exceeded")
        if metadata.parent.name == "x":
            require(shape == [rows, 175, 8, 8] and dtype == np.dtype("float16"), "stored input shape/precision differs")


def exact_arrays(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    require(actual.keys() == expected.keys(), "raw array keys differ")
    for key, value in expected.items():
        array = actual[key]
        require(array.dtype == value.dtype and array.shape == value.shape
                and array.tobytes(order="C") == value.tobytes(order="C"), "raw array bytes differ: " + key)


def verify_game(item: dict[str, Any], planned: dict[str, Any], producer: dict[str, Any],
                verifier: Any, game: Any, adapter: Any, caps: dict[str, int]) -> tuple[dict[str, Any], int]:
    import chess
    import numpy as np
    from chess_anti_engine.core_generation_data import capture_raw_targets, validate_stored_targets
    from chess_anti_engine.encoding import encode_cboard_for_model
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.moves import index_to_move_strict, move_to_index_for_encoding
    from chess_anti_engine.replay.augment import _mirror_x_batch
    from chess_anti_engine.replay.shard import load_shard_arrays, validate_arrays

    directory = learning.external(item["path"])
    learning.verify_inventory({"path": str(directory), "files": item["inventory"]})
    documents = {}
    for key, evidence in item["evidence"].items():
        require(Path(evidence["path"]).parent == directory, "foreign game evidence")
        _, documents[key] = learning.read_ref(evidence)
    require(set(documents) == {"provenance", "validation", "outcome_proof", "raw_metadata"}, "incomplete producer evidence")
    provenance, validation, original, raw_metadata = (documents[key] for key in
                                                     ("provenance", "validation", "outcome_proof", "raw_metadata"))
    require(provenance["planned"] == planned and validation == item["validation"], "planned/readout validation differs")
    audit_dict = provenance["actual"]
    proof = verifier.verify(audit_dict, planned, game)
    issued = proof.metadata()
    common = {"run_uuid": producer["run_uuid"], "collection_uuid": producer["collection_uuid"],
              "generation": producer["generation"], "source_revision": producer["source"]["revision"],
              "actor_sha256": producer["actor"]["sha256"], "served_state_sha256": verifier.served_state_sha256,
              "physical_game_id": planned["physical_game_id"], "bank": "train", "game_id": audit_dict["game_id"],
              "rows": audit_dict["retained_samples"], "scientific_class": issued["scientific_class"]}
    require(all(all(doc.get(key) == value and type(doc.get(key)) is type(value) for key, value in common.items())
                for doc in [item, *documents.values()]), "source/game/actor/common identity differs")
    require(original["schema"] == "CORE_PRODUCING_OUTCOME_PROOF_V1"
            and original["verification"] == issued and original["proof_sha256"] == proof.proof_sha256,
            "fresh outcome verification differs from original issued proof")
    require(validation["schema"] == "VERIFIED_CORE_GAME_METADATA_V1"
            and validation["full_history_verified"] is True and validation["resolved"] is issued["resolved"]
            and validation["verified_outcome_sha256"] == proof.proof_sha256, "validation/proof differs")
    audit = restore_audit(audit_dict)
    raw_path = directory / "raw-targets.npz"
    raw = load_raw(raw_path, caps["max_raw_decoded_bytes_per_game"])
    rows = validation["rows_detail"]
    n = common["rows"]
    require(type(n) is int and n >= 0 and len(rows) == n, "retained row count differs")
    eligible = [record for record in audit.records if record.has_policy and not record.is_sf_refute_opp]
    require(not any(record.is_sf_refute_opp for record in audit.records), "raw SF-refute record in pure selfplay")
    require(n == (len(eligible) if issued["resolved"] else 0), "eligible policy rows were dropped or fabricated")
    require([row["ply"] for row in rows] == ([record.ply_index for record in eligible] if issued["resolved"] else []),
            "retained raw-record chronological order differs")
    samples = []
    for i, row in enumerate(rows):
        require(row["physical_row_id"] == planned["physical_game_id"] + ":" + str(i), "physical row identity differs")
        samples.append(SimpleNamespace(game_id=common["game_id"], ply_index=row["ply"], has_policy=True,
            is_network_turn=True, is_selfplay=True, wdl_target=int(raw["outcome_wdl"][i]),
            policy_target=raw["policy_target_raw"][i],
            search_wdl=raw["search_wdl_raw"][i] if raw["has_search_raw"][i] else None,
            categorical_target=raw["categorical_raw"][i] if raw["has_categorical_raw"][i] else None))
    rebuilt, rebuilt_meta = capture_raw_targets(samples, audit, [row["physical_row_id"] for row in rows], proof=proof, game=game)
    exact_arrays(raw, rebuilt)
    require(all(raw_metadata[key] == value for key, value in rebuilt_meta.items()), "raw metadata/proof differs")
    decoded = sum(value.nbytes for value in raw.values())
    if n == 0:
        require(item["shard"] is None and not (directory / "shard").exists()
                and validation["usable_shard"] is False and validation["raw_targets_verified"] is False,
                "zero-row attempt has a usable shard")
        return {"eligible": False, "physical_game_id": planned["physical_game_id"],
                "scientific_class": issued["scientific_class"], "rows": 0, "proof_sha256": proof.proof_sha256}, decoded
    require(issued["resolved"] is True and validation["usable_shard"] is True and
            validation["raw_targets_verified"] is True, "nonempty unresolved or unqualified game")
    shard = item["shard"]
    require(Path(shard["path"]) == directory / "shard", "foreign shard")
    shard_path, inventory_sha = learning.verify_inventory(shard)
    require(inventory_sha == validation["shard_inventory_sha256"], "shard inventory identity differs")
    shard_decoded_bound(shard_path, n, caps["max_shard_decoded_bytes_per_game"])
    arrays, _ = load_shard_arrays(shard_path, lazy=False, validate=True)
    decoded_arrays = sum(value.nbytes for value in arrays.values())
    require(decoded_arrays <= caps["max_shard_decoded_bytes_per_game"], "loaded shard decoded bound exceeded")
    decoded += decoded_arrays
    validate_arrays(arrays)
    validate_stored_targets(raw, arrays, proof=proof, metadata=raw_metadata, game_id=common["game_id"], plies=[row["ply"] for row in rows])
    require(str(np.asarray(arrays["_input_history_encoding"]).item()) == adapter.input_history_encoding
            and str(np.asarray(arrays["_policy_encoding"]).item()) == game.policy_encoding
            and str(np.asarray(arrays["_history_rep_fix"]).item()).lower() == "true", "stored encoding metadata differs")
    require(arrays["x"].dtype == np.dtype("float16"), "stored inputs are not F16")
    require(arrays["legal_mask"].dtype == np.dtype("uint8") and arrays["legal_mask"].shape == (n, 1858), "stored legal mask layout differs")
    mirrored = _mirror_x_batch(arrays["x"], np.ones(n, dtype=bool), input_history_encoding=adapter.input_history_encoding)
    board = chess.Board(audit.start.root_fen)
    for token in audit.start.opening_moves:
        board.push_uci(token)
    wanted = {row["move_offset"]: i for i, row in enumerate(rows)}
    require(len(wanted) == n and len({row["ply"] for row in rows}) == n, "duplicate retained move/ply")
    played = []
    records = {record.ply_index: record for record in audit.records}
    for offset, action in enumerate(audit.played_move_indices):
        if offset in wanted:
            i = wanted[offset]
            row = rows[i]
            record = records[row["ply"]]
            require(record.move_offset == offset and record.ply_index == board.ply() and
                    record.pov_color == board.turn and record.has_policy and not record.is_sf_refute_opp,
                    "retained full-history/POV/record join differs")
            expected = np.asarray(encode_cboard_for_model(adapter, CBoard.from_board(board)), dtype=np.float16)
            require(expected.shape == (175, 8, 8) and np.isfinite(expected).all()
                    and arrays["x"][i].tobytes() == expected.tobytes(), "stored full-history F16 input differs")
            legal = np.zeros(1858, dtype=np.uint8)
            for move in board.legal_moves:
                legal[move_to_index_for_encoding(move, board, policy_encoding=game.policy_encoding)] = 1
            require(np.array_equal(arrays["legal_mask"][i], legal) and
                    not raw["policy_target_raw"][i][legal == 0].any(), "stored legal mask/policy differs")
            verifier.verify_policy_sample(proof, samples[i], board, game.policy_encoding)
            require(row["history_sha256"] == digest([audit.start.root_fen, list(audit.start.opening_moves) + played])
                    and row["stored_input_float16_sha256"] == hashlib.sha256(expected.astype("<f2").tobytes()).hexdigest()
                    and row["stored_input_float16_mirror_sha256"] == hashlib.sha256(mirrored[i].astype("<f2").tobytes()).hexdigest(),
                    "retained history/raw/mirror fingerprint differs")
        move = index_to_move_strict(action, board)
        require(move in board.legal_moves, "illegal replay action")
        played.append(move.uci())
        board.push(move)
    require(all(0 <= offset < len(audit.played_move_indices) for offset in wanted), "retained row outside played trace")
    require(learning.verify_inventory(shard)[1] == inventory_sha, "shard changed while audited")
    learning.verify_inventory({"path": str(directory), "files": item["inventory"]})
    return {"eligible": True, **common, "label_class": "TB_ADJUDICATED" if
            issued["scientific_class"] == "STRICT_TB_ADJUDICATED" else "NATURAL",
            "shard": shard, "evidence": item["evidence"], "shard_inventory_sha256": inventory_sha,
            "proof_sha256": proof.proof_sha256}, decoded


def execute(protocol_ref: dict[str, str]) -> dict[str, Any]:
    require(Path(__file__).resolve() == learning.REPO / "scripts/core_finite_admit.py", "foreign auditor entrypoint")
    _, p = learning.read_ref(protocol_ref)
    require(set(p) == {"schema", "learner_contract", "producer_protocol", "producer_readout", "producer_inventory",
                       "producer_owner", "gates", "runtime", "output_root", "limits"}
            and p["schema"] == "CORE_FINITE_ADMISSION_V1", "invalid audit protocol")
    runtime_guard(p)
    caps = p["limits"]
    require(set(caps) == {"max_metadata_bytes", "max_inventory_files", "max_inventory_bytes",
                         "max_raw_decoded_bytes_per_game", "max_shard_decoded_bytes_per_game", "max_total_decoded_bytes"}, "invalid audit limits")
    for key, value in caps.items():
        learning.integer(value, key)
    require(caps["max_metadata_bytes"] <= METADATA_CAP, "audit metadata cap exceeds128MiB")
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("finite CPU audit deadline")))
    signal.alarm(p["runtime"]["inner_seconds"])
    output = learning.external(p["output_root"])
    require(not learning.failure_marker(output).exists(), "previous audit failed")
    output.mkdir(exist_ok=False)
    verifier = None
    try:
        from scripts import core_finite_collect as collection
        from chess_anti_engine.encoding import rep_fix
        from chess_anti_engine.verified_outcome import VerifiedOutcomeVerifier

        _, contract = learning.read_ref(p["learner_contract"])
        learning.validate_contract(contract)
        require("scripts/core_finite_admit.py" in contract["source_files"], "audit source is not pinned")
        _, producer = learning.read_ref(p["producer_protocol"])
        require(contract["run_uuid"] == producer["run_uuid"] and contract["source_revision"] == producer["source"]["revision"]
                and Path(producer["source"]["checkout"]).resolve() == learning.REPO,
                "collector and learner must use the same final integrated source")
        gates(p, contract, producer, None)
        root_stage = check_owner(p["producer_owner"], p["producer_protocol"], producer)
        require(all(root_stage[key] == producer[value] for key, value in
                    (("run_uuid", "run_uuid"), ("generation", "generation"))) and
                root_stage["source_revision"] == producer["source"]["revision"] and
                root_stage["actor_sha256"] == producer["actor"]["sha256"], "producer admission identities differ")
        roster = collection.checked(producer["roster"])
        require(learning.file_sha(Path(producer["starts"]["path"])) == producer["starts"]["sha256"], "starts changed")
        collection.validate_protocol(producer, roster)
        native = collection.source_guard(producer)
        for image in native["native_images"]:
            module = importlib.import_module(image["module"])
            filename = getattr(module, "__file__", None)
            require(isinstance(filename, str), "audit native has no file origin")
            actual_path = Path(str(filename))
            require(actual_path.resolve() == Path(image["path"]).resolve()
                    and learning.file_sha(actual_path) == image["sha256"], "audit native origin differs")
        # No CBoard has been constructed in this fresh CLI. Set the regime
        # before descriptor replay, verifier reconstruction or native encoding.
        require(rep_fix.current() in (None, True), "unexpected preexisting repetition regime")
        rep_fix.apply(True)
        cfg, profile = collection.build_configs(producer, roster)
        require(digest(profile) == producer["resolved_profile_sha256"], "realized collector profile changed")
        producer_root = learning.external(producer["output_root"])
        require(not learning.failure_marker(producer_root).exists() and not (producer_root / "failure.json").exists()
                and not list(producer_root.glob(".attempt-*")) and not list(producer_root.glob("failed-*.json")), "failed/partial producer publication")
        _, full_inventory = learning.read_ref(p["producer_inventory"], max_bytes=caps["max_metadata_bytes"])
        require(full_inventory["path"] == str(producer_root) and len(full_inventory["files"]) <= caps["max_inventory_files"]
                and sum(row["bytes"] for row in full_inventory["files"]) <= caps["max_inventory_bytes"], "producer inventory budget/root differs")
        learning.verify_inventory(full_inventory)
        readout_path, readout = learning.read_ref(p["producer_readout"], max_bytes=caps["max_metadata_bytes"])
        require(readout_path == producer_root / "readout.json" and readout["status"] == "COLLECTED_CORE_GENERATION_V1"
                and readout["protocol_sha256"] == p["producer_protocol"]["sha256"] and
                all(readout[key] == producer[key] for key in ("run_uuid", "collection_uuid", "generation", "attempts")) and
                readout["source_revision"] == contract["source_revision"] and readout["actor_sha256"] == producer["actor"]["sha256"]
                and readout["all_planned_attempts_accounted"] is True and readout["training_steps"] == 0
                and readout["root_acceptance_issued"] is False and readout["learning_admitted"] is False,
                "producer readout identity/status differs")
        _, realized = learning.read_ref(ref(producer_root / "resolved-profile.json"))
        require(realized == profile, "saved realized profile differs")
        _, serving = learning.read_ref(readout["actor_serving"])
        served = serving["pre_state_sha256"]
        require(serving["actor_sha256"] == producer["actor"]["sha256"] and served == serving["post_state_sha256"] ==
                serving["checkpoint_model_state_sha256"] and len(served) == 64 and serving["state_tensors"] == 494
                and serving["model_build_count"] == 1 and serving["checkpoint_step"] == producer["actor"]["expected_checkpoint_step"]
                and serving["logical_generation"] == producer["generation"] and serving["forwards"]["calls"] > 0, "actor serving evidence differs")
        actor = learning.external(producer["actor"]["path"])
        require(learning.file_sha(actor) == producer["actor"]["sha256"] and collection.file_stamp(actor) == serving["checkpoint_stat"], "actor file changed")
        _, lifecycle = learning.read_ref(ref(producer_root / "stockfish-lifecycle.json"))
        require(lifecycle["all_observed_births_closed"] is True and readout["pool_closed"] is True and
                readout["verification_handle_closed"] is True, "producer engine/verifier closure absent")
        completed = readout["completed"]
        require(set(completed) == {str(i) for i in range(producer["attempts"])}, "not every fixed attempt accounted")
        adapter = SimpleNamespace(input_history_encoding=contract["architecture"]["input_history_encoding"],
                                  input_extra_features=contract["architecture"]["input_extra_features"])
        require(contract["architecture"] == producer["actor"]["architecture"], "actor/learner architecture differs")
        verifier = VerifiedOutcomeVerifier(path=producer["tablebase"]["path"],
            asset_inventory_sha256=producer["tablebase"]["asset_inventory_sha256"], collection_uuid=producer["collection_uuid"],
            source_revision=contract["source_revision"], run_uuid=producer["run_uuid"], generation=producer["generation"],
            concurrent_slots=producer["concurrent_slots"], actor_sha256=producer["actor"]["sha256"], served_state_sha256=served)
        if "torch" in sys.modules:
            require(not sys.modules["torch"].cuda.is_initialized(), "CPU audit initialized CUDA")
        audited, eligible, decoded = [], [], 0
        for i, planned in enumerate(roster):
            item = completed[str(i)]
            _, receipt = learning.read_ref(ref(producer_root / f"completed-{i}.json"))
            require(receipt == item and Path(item["path"]) == producer_root / f"game-{i}", "completed receipt/path differs")
            checked, size = verify_game(item, planned, producer, verifier, cfg["game"], adapter, caps)
            decoded += size
            require(decoded <= caps["max_total_decoded_bytes"], "total decoded budget exceeded")
            audited.append(checked)
            if checked["eligible"]:
                eligible.append(checked)
        counts = {kind: sum(game["scientific_class"] == kind for game in audited)
                  for kind in ("NATURAL", "STRICT_TB_ADJUDICATED", "UNRESOLVED")}
        require(counts == readout["class_counts"] and len(eligible) == readout["resolved_nonempty_games"], "producer class/usable counts differ")
        overlap = collection.exclusions_gate({int(key): value for key, value in completed.items()}, producer)
        _, original_overlap = learning.read_ref(ref(producer_root / "input-exclusions.json"))
        require(overlap == original_overlap and overlap["qualified"] is True and readout["playing_and_exposure_gate_qualified"] is True, "playing/exposure gate differs or failed")
        gates(p, contract, producer, eligible)
        verifier.close()
        verifier = None
        collection.source_guard(producer)
        learning.import_custody()
        require(learning.file_sha(actor) == producer["actor"]["sha256"], "actor changed during CPU audit")
        learning.verify_inventory(full_inventory)
        games = []
        for game in eligible:
            path = output / ("game-" + game["physical_game_id"].rsplit(":", 1)[1] + ".json")
            wrapper = {"status": "VERIFIED_RESOLVED_CORE_TRAIN_GAME_V1", **{key: game[key] for key in
                       ("run_uuid", "generation", "source_revision", "actor_sha256", "served_state_sha256", "physical_game_id",
                        "game_id", "bank", "rows", "label_class", "evidence", "shard_inventory_sha256")},
                       "resolved": True, "full_history_verified": True, "raw_targets_verified": True,
                       "audit_protocol": protocol_ref}
            learning.write_json(path, wrapper)
            games.append({key: game[key] for key in ("physical_game_id", "game_id", "bank", "rows", "label_class", "shard")} | {"producer_proof": ref(path)})
        manifest = {"schema": "ROOT_ACCEPTED_CORE_TRAIN_GENERATION_V1", "run_uuid": producer["run_uuid"],
                    "source_revision": contract["source_revision"], "generation": producer["generation"],
                    "actor_sha256": producer["actor"]["sha256"], "actor_serving": readout["actor_serving"], "games": games,
                    "audit_protocol": protocol_ref, "root_postclosure_promotion_required": True}
        learning.write_json(output / "generation.json", manifest)
        result = {"status": "AUDITED_CORE_TRAIN_GENERATION_CANDIDATE_PENDING_ROOT_CLOSURE_V1", "audit_protocol": protocol_ref,
                  "generation_manifest": ref(output / "generation.json"), "attempts": len(audited), "audited": audited,
                  "eligible_games": len(eligible), "eligible_rows": sum(game["rows"] for game in eligible),
                  "class_counts": counts, "decoded_bytes": decoded, "root_postclosure_promotion_required": True,
                  "models_constructed": 0, "forwards": 0, "optimizer_updates": 0,
                  "side_effect_count_scope": "source call graph; not instrumented constructor/forward counters",
                  "policy_limit": "original F32 observation plus legal/strictTB recheck; no independent Gumbel replay"}
        learning.write_json(output / "readout.json", result)
        learning.seal(output)
        learning.fsync_directory(output)
        learning.fsync_directory(output.parent)
        return result
    except BaseException as error:
        marker = learning.failure_marker(output)
        learning.write_json(marker, {"status": "FAILED_CORE_ADMISSION_V1", "error": repr(error), "protocol": protocol_ref,
                                    "root_postclosure_promotion_required": True, "learning_admitted": False})
        marker.chmod(0o444)
        learning.fsync_directory(marker.parent)
        raise
    finally:
        signal.alarm(0)
        if verifier is not None:
            verifier.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    args = parser.parse_args(argv)
    result = execute({"path": args.protocol, "sha256": args.protocol_sha256})
    print(canonical({"status": result["status"], "generation_manifest": result["generation_manifest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
