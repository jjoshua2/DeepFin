"""Produce or revalidate a full-denominator current-P0 SF evidence carrier, CPU only."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
SELECTION = "ROOT_SELECTS_BANKED_CURRENT_P0_SF_SIDECAR_ENGINEERING_DIRECTION_NOT_TRAINING_RECIPE_OR_EXECUTION"
PROTOCOL = "sf_current_p0_sidecar_v1"


def require(value: Any, message: str) -> None:
    if not value:
        raise ValueError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def reference(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class NoForbiddenImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname == "torch" or fullname.startswith(("torch.", "onnxruntime", "chess_anti_engine.train")):
            raise ImportError("forbidden sidecar import: " + fullname)
        return None


def module_guard() -> None:
    for name, module in list(sys.modules.items()):
        require(not (name == "torch" or name.startswith(("torch.", "onnxruntime", "chess_anti_engine.train"))),
                "forbidden sidecar runtime module")
        if name.startswith(("chess_anti_engine", "scripts.sf_")):
            origin = getattr(module, "__file__", None)
            if origin:
                path = Path(origin).resolve()
                require(path.is_relative_to(REPO), "foreign project source")
                require(not any(str(path).endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES),
                        "project native module loaded")


def runtime_guard(p: dict[str, Any]) -> None:
    module_guard()
    require(__debug__ and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
            and os.environ.get("OMP_NUM_THREADS") == os.environ.get("MKL_NUM_THREADS") == "2", "CPU environment")
    r = p["runtime"]
    require(sorted(os.sched_getaffinity(0)) == [20, 22] == r["cpu_affinity"], "CPU affinity")
    require(type(r["inner_seconds"]) is int and type(r["owner_seconds"]) is int
            and 0 < r["inner_seconds"] < r["owner_seconds"] <= 900, "finite sidecar caps")
    owner = Path(r["owner"]["path"])
    require(hashlib.sha256(owner.read_bytes()).hexdigest() == r["owner"]["sha256"], "owner source identity")
    parent = Path("/proc", str(os.getppid()), "cmdline").read_bytes().decode().split("\0")
    require(str(owner) in parent and parent.count("--seconds") == parent.count("--resource") == 1
            and parent[parent.index("--seconds") + 1] == str(r["owner_seconds"])
            and parent[parent.index("--resource") + 1] == "cpu", "not direct declared CPU owner")
    source = p["source"]
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    require(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, env=env, text=True).strip()
            == source["revision"], "source revision")
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, env=env, text=True), "dirty source")
    require(not list(REPO.rglob("*.so")), "sidecar runtime must isolate project native images")
    needed = {"scripts/sf_current_p0_sidecar.py", "scripts/sf_current_p0_sidecar_schema.py", "scripts/sf_banked_p0_census.py",
              "chess_anti_engine/moves/encode.py", "chess_anti_engine/stockfish/wdl.py", "chess_anti_engine/ext_check.py"}
    require(needed <= source["files"].keys(), "missing source pins")
    for name, sha in source["files"].items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts
                and not (REPO / path).is_symlink() and (REPO / path).resolve().is_relative_to(REPO)
                and hashlib.sha256((REPO / path).read_bytes()).hexdigest() == sha, "consumed source changed")


def limits(p: dict[str, Any]) -> None:
    require(set(p) == {"schema", "scope", "phase", "selection", "census_protocol", "census_readout", "census_acceptance",
                       "census_review", "census_terminal", "census_command", "source", "runtime", "limits", "candidate",
                       "producer_owner", "output_root"} and p["schema"] == PROTOCOL, "sidecar protocol keys/schema")
    require(p["scope"] in {"FIXED_ORIGINAL_G1", "SYNTHETIC_ONLY"} and p["phase"] in {"produce", "validate"}, "phase/scope")
    cap = p["limits"]
    require(set(cap) == {"metadata_bytes", "game_decoded_bytes", "inventory_files", "row_bytes", "game_metadata_bytes",
                        "carrier_game_bytes", "carrier_total_bytes"}, "sidecar limits")
    require(all(type(x) is int and x > 0 for x in cap.values()), "positive integer limits")
    require(cap["metadata_bytes"] <= 128 << 20 and cap["game_decoded_bytes"] <= 64 << 20
            and cap["inventory_files"] <= 128000 and cap["row_bytes"] <= 1 << 20
            and cap["game_metadata_bytes"] <= 32 << 20 and cap["carrier_game_bytes"] <= 64 << 20
            and cap["carrier_total_bytes"] <= 1 << 30, "sidecar bounded limits")


def closed(terminal: dict[str, Any], command: list[str], owner_sha: str, seconds: int) -> None:
    require(terminal["status"] == "PASS" and type(terminal["returncode"]) is int and terminal["returncode"] == 0
            and terminal["resource"] == "cpu" and terminal["owned_processes_closed"] is True
            and terminal["custody"] == "OWNED_PROCESSES_CLOSED" and terminal["command"] == command
            and terminal["supervisor_sha256"] == owner_sha and terminal["seconds"] == seconds, "not exact closed CPU owner")
    require(bool(terminal["known_processes"]), "missing owner births")
    for pid, birth in terminal["known_processes"].items():
        path = Path("/proc", str(pid), "stat")
        require(not path.exists() or int(path.read_text().rsplit(")", 1)[1].split()[19]) != birth, "owned child still live")


def phase_authority(p: dict[str, Any], protocol_ref: dict[str, str], admission_ref: dict[str, str]) -> None:
    """The separate admission can bind protocol SHA without a reference cycle."""
    from scripts import sf_banked_p0_census as audit

    require(audit.checked(protocol_ref, 4 << 20) == p, "sidecar protocol reference/content mismatch")
    admission = audit.checked(admission_ref, p["limits"]["metadata_bytes"])
    require(admission["status"] == "ROOT_ADMITS_CURRENT_P0_SF_SIDECAR_PHASE_V1"
            and admission["protocol"] == protocol_ref and admission["phase"] == p["phase"]
            and admission["scope"] == p["scope"] and admission["source_revision"] == p["source"]["revision"]
            and admission["source_files_sha256"] == audit.digest(p["source"]["files"])
            and admission["runtime"] == p["runtime"] and admission["selection"] == p["selection"]
            and admission["census_readout"] == p["census_readout"], "sidecar phase not exactly root-admitted")


def inputs(p: dict[str, Any]) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    from scripts import sf_banked_p0_census as audit

    limits(p)
    cap = p["limits"]["metadata_bytes"]
    selected = audit.checked(p["selection"], cap)
    require(selected["status"] == SELECTION and selected["execution_admitted"] is False
            and selected["training_recipe_selected"] is False, "engineering selection scope")
    census = audit.checked(p["census_protocol"], cap)
    report = audit.checked(p["census_readout"], cap)
    acceptance = audit.checked(p["census_acceptance"], cap)
    review = audit.checked(p["census_review"], cap)
    synthetic = p["scope"] == "SYNTHETIC_ONLY"
    if synthetic:
        require(census["qualification_scope"] == "SYNTHETIC_NOT_REAL_AUTHORITY"
                and census["expected"]["run_uuid"] == "synthetic-run"
                and census["expected"]["source_revision"] == "1" * 40
                and census["expected"]["physical_games"] == 1
                and 0 < census["expected"]["rows"] <= 4, "known fictional fixture scope")
    if synthetic and acceptance["status"] == "ROOT_ACCEPTS_CLOSED_FIXED_SYNTHETIC_DIRECT_CLI_RUNTIME_QUALIFICATION":
        require(acceptance["scope"] == "SYNTHETIC_NOT_REAL_AUTHORITY"
                and acceptance["synthetic_readout"] == p["census_readout"]
                and acceptance["synthetic_rows"] == report["per_row_evidence"]
                and acceptance["protocol"] == p["census_protocol"]
                and acceptance["terminal"] == p["census_terminal"], "closed synthetic CLI acceptance")
        receipt = review["root_receipt"]
        require(review["status"] == "ACCEPT_CLOSED_SYNTHETIC_DIRECT_CLI_POSITIVE_RUNTIME_QUALIFICATION_NO_FINDINGS"
                and not review["findings"]
                and {k: receipt[k] for k in ("path", "sha256")} == p["census_acceptance"],
                "synthetic independent CLI acceptance")
    else:
        require(not synthetic or acceptance["scope"] == "SYNTHETIC_NOT_REAL_AUTHORITY", "fictional receipt marker")
        require(acceptance["status"] == "ROOT_ACCEPTS_CLOSED_FIXED_G1_CENSUS_DATA_ONLY"
                and acceptance["readout"] == p["census_readout"] and acceptance["protocol"] == p["census_protocol"]
                and acceptance["terminal"] == p["census_terminal"] and acceptance["rows"] == report["per_row_evidence"], "census root acceptance")
        require(review["status"] == "ACCEPT_ACTUAL_METADATA_NO_FINDINGS_DATA_ONLY" and not review["findings"]
                and review["readout"] == p["census_readout"] and review["protocol"] == p["census_protocol"]
                and review["root_acceptance"] == p["census_acceptance"], "census independent acceptance")
    terminal = audit.checked(p["census_terminal"], cap, readonly=False)
    closed(terminal, p["census_command"], census["runtime"]["owner"]["sha256"], census["runtime"]["owner_seconds"])
    require(p["census_command"][-4:] == ["--protocol", p["census_protocol"]["path"], "--protocol-sha256", p["census_protocol"]["sha256"]], "census command binding")
    require(report["status"] == "COMPLETE_SF_BANKED_P0_EVIDENCE_CENSUS_DATA_ONLY"
            and report["protocol"] == p["census_protocol"] and report["generation_manifest"] == census["generation_manifest"]
            and report["audit_uuid"] == census["audit_uuid"] == selected["audit_uuid"]
            and report["rows"] == census["expected"]["rows"]
            and report["physical_games"] == census["expected"]["physical_games"], "fixed census report")
    require(audit.file_sha(REPO / "scripts/sf_banked_p0_census.py", readonly=False)
            == report["auditor_source"]["files"]["scripts/sf_banked_p0_census.py"], "census algorithm changed")
    for name, historical_sha in report["auditor_source"]["files"].items():
        # Current main removed one optional-symbol allowlist entry in ext_check.
        # Its NEW bytes are independently pinned by runtime_guard; that is not
        # historical whole-file equality. Encoding/move/value bytes stay exact.
        if name != "chess_anti_engine/ext_check.py":
            require(name in p["source"]["files"] and p["source"]["files"][name] == historical_sha
                    and audit.file_sha(REPO / name, readonly=False) == historical_sha,
                    "historical census dependency changed: " + name)
    if p["scope"] == "FIXED_ORIGINAL_G1":
        require(census["expected"]["physical_games"] == 256 and census["expected"]["rows"] == 24827
                and census["generation_manifest"]["sha256"] == "35279e2ef19cdf808e4b4e25bb50fbb524750693ee035c1d44080d87a3db7878", "original G1 population")
    generation, producer = audit.generation_inputs(census)
    require(len(report["games"]) == len(generation["games"])
            and [g["physical_game_id"] for g in report["games"]] == [g["physical_game_id"] for g in generation["games"]], "census game order")
    return audit, census, report, generation, producer


def fresh_output(p: dict[str, Any], generation: dict[str, Any], report: dict[str, Any]) -> Path:
    output = Path(p["output_root"])
    require(output.is_absolute() and not output.exists() and output.resolve() == output
            and not output.with_name(output.name + ".failed.json").exists()
            and not output.is_relative_to(REPO) and not REPO.is_relative_to(output), "fresh external ordinary output")
    inputs_roots = [Path(g["shard"]["path"]).parent for g in generation["games"]]
    inputs_roots.extend(Path(g["producer_proof"]["path"]).parent for g in generation["games"])
    inputs_roots.extend(Path(report[k]["path"]).parent for k in (
        "per_row_evidence", "protocol", "generation_manifest", "producer_protocol", "input_authority", "selection"))
    inputs_roots.extend(Path(p[k]["path"]).parent for k in (
        "selection", "census_protocol", "census_readout", "census_acceptance", "census_review", "census_terminal"))
    if p["candidate"] is not None:
        inputs_roots.append(Path(p["candidate"]["path"]).parent)
    require(all(not output.is_relative_to(root) and not root.is_relative_to(output) for root in inputs_roots), "output/input overlap")
    return output


def expected_game(audit: Any, item: dict[str, Any], generation: dict[str, Any], producer: dict[str, Any],
                  cap: dict[str, int], census_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fresh full-history validation, then original same-predecessor byte copies."""
    import numpy as np
    from scripts import sf_current_p0_sidecar_schema as schema

    fresh = list(audit.inspect_game(item, generation, producer, cap))
    require(canonical(fresh) == canonical(census_rows), "accepted census/source rows changed")
    n = item["rows"]
    require(schema.decoded_bytes(n) <= cap["carrier_game_bytes"], "carrier per-game allocation cap")
    before = audit.verify_inventory(item["shard"], cap["inventory_files"])
    group = audit.open_arrays(item["shard"], n, cap["game_decoded_bytes"])
    needed = {"sf_wdl", "sf_policy_target", "sf_legal_mask", "sf_multipv_raw", "sf_label_meta"}
    names = needed | {audit.SF_FIELDS[k][0] for k in needed}
    arrays = {name: np.asarray(group[name][:]) for name in names if name in group}
    arrays["x"] = group["x"]  # Shape only, already authenticated/decoded by inspect_game.
    current_legal = np.asarray(group["legal_mask"][:])
    result = schema.arrays_for_rows(n)
    result["legal_mask_u8"][:] = current_legal
    by_id = {r["physical_row_id"]: i for i, r in enumerate(fresh)}
    row_metadata = []
    for i, row in enumerate(fresh):
        join = row["retained_predecessor_P1_join"]
        status = join["status"]
        observations: dict[str, Any] = {"policy": None, "stored_wdl": None, "native_wdl": None}
        source_reasons, invalid = [], []
        storage_presence = None
        stored_relation = "UNAVAILABLE_NO_JOIN"
        if status != "UNAVAILABLE":
            previous = by_id.get(join["predecessor_row"])
            require(previous is not None and fresh[previous]["input_identity"]["ply"] + 1 == row["input_identity"]["ply"], "actual predecessor identity")
            query = fresh[previous]["own_P1_query_context"]
            require(all(query[k] == row["input_identity"][k] for k in ("fen", "STM_white", "history_sha256")), "P0 query/fullhistory")
            values = {name: audit.field(arrays, name, previous)[0] for name in needed}
            label = fresh[previous]["own_record_P1"]
            source_reasons, invalid = label["reasons"], label["invalid_reasons"]
            storage_presence = label["stored_field_contract"]
            stored_relation = label["WDL_relation"]
            observations = {"policy": label["policy_present"], "stored_wdl": label["stored_sf_wdl_record_POV"] is not None,
                            "native_wdl": label["native_query_STM_WDL_permille"] is not None}
            require(all(join[k] is observations[name] for k, name in [("policy_available", "policy"),
                        ("stored_value_available", "stored_wdl"), ("native_value_available", "native_wdl")]), "raw presence census join")
            valid = status != "INVALID_REDUCED_LABEL"
            if values["sf_legal_mask"] is not None:
                require(not valid or np.array_equal(values["sf_legal_mask"], current_legal[i]),
                        "predecessor/current legal-map identity")
            policy = values["sf_policy_target"]
            if policy is not None:
                result["policy_f16"][i] = policy
                result["qualified_policy"][i] = int(valid)
            wdl = values["sf_wdl"]
            if wdl is not None:
                result["stored_record_wdl_f16"][i] = wdl
                result["stored_query_wdl_f16"][i] = wdl[::-1]
                result["qualified_stored_wdl"][i] = int(valid)
            native = label["native_query_STM_WDL_permille"]
            if native is not None:
                result["native_wdl_permille_i16"][i] = native
                result["qualified_native_wdl"][i] = int(valid)
            for src, dst, flag in [("sf_label_meta", "predecessor_meta_i32", "observed_predecessor_meta"),
                                  ("sf_multipv_raw", "predecessor_pv_i16", "observed_predecessor_pv")]:
                if values[src] is not None:
                    result[dst][i] = values[src]
                    result[flag][i] = 1
        row_metadata.append({"physical_row_id": row["physical_row_id"], "physical_game_id": item["physical_game_id"],
                             "game_id": item["game_id"], "original_row_index": i, "input_identity": row["input_identity"],
                             "legal_mask_sha256": schema.array_digest(result["legal_mask_u8"][i]),
                             "source_join": join, "observed_flags": observations, "source_reasons": source_reasons,
                             "source_storage_presence": storage_presence,
                             "stored_value_relation": stored_relation,
                             "source_invalid_reasons": invalid,
                             "source_census_row_sha256": audit.digest(row),
                             "explicit_stored_P0": row["explicit_stored_P0"],
                             "qualified_flags": {key: bool(result["qualified_" + key][i]) for key in ("policy", "stored_wdl", "native_wdl")},
                             "observation_sha256": {key: schema.array_digest(result[key][i]) for key in schema.FIELDS},
                             "wire_authenticated": False})
    schema.check_arrays(result, n, cap["carrier_game_bytes"])
    require(before == audit.verify_inventory(item["shard"], cap["inventory_files"]), "source changed while copying carrier")
    metadata = {"schema": schema.SCHEMA, "physical_game_id": item["physical_game_id"], "game_id": item["game_id"], "rows": n,
                "original_shard": item["shard"], "producer_proof": item["producer_proof"], "rows_detail": row_metadata,
                "source_value_identity": {k: producer["configs"]["game"][k] for k in ("sf_wdl_use_cp_logistic", "sf_wdl_cp_slope", "sf_wdl_cp_draw_width")},
                "native_value": "query-STM permille W,D and reconstructed L; not stored CP-logistic",
                "stored_value": "original record-POV F16 plus exactly one W/L swap into query/current STM",
                "padding": "zero float and -1 native padding with false qualified flags is not an observation",
                "training_admitted": False}
    return metadata, result


def census_games(audit: Any, report: dict[str, Any], generation: dict[str, Any], p: dict[str, Any]) -> Any:
    """Read one accepted game's metadata at a time, never an all-population list."""
    path = Path(report["per_row_evidence"]["path"])
    require(audit.stamp(path)[2] <= 256 << 20
            and audit.file_sha(path) == report["per_row_evidence"]["sha256"], "census row stream cap/SHA")
    seen = set()
    with path.open("rb") as stream:
        for game in generation["games"]:
            rows, size = [], 0
            for _ in range(game["rows"]):
                raw = stream.readline(p["limits"]["row_bytes"] + 1)
                require(raw.endswith(b"\n") and len(raw) <= p["limits"]["row_bytes"], "census row line cap/EOF")
                size += len(raw)
                require(size <= p["limits"]["game_metadata_bytes"], "game row metadata cap")
                row = json.loads(raw, object_pairs_hook=audit.unique_object)
                require(row["physical_game_id"] == game["physical_game_id"] and row["physical_row_id"] not in seen, "census order/duplicate row")
                seen.add(row["physical_row_id"])
                rows.append(row)
            yield game, rows
        require(not stream.read(1) and len(seen) == report["rows"], "census stream denominator/trailing data")
    require(audit.file_sha(path) == report["per_row_evidence"]["sha256"], "census stream changed")


def write_json(path: Path, value: Any) -> None:
    with path.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def seal(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
        fd = os.open(path, os.O_RDONLY | (os.O_DIRECTORY if path.is_dir() else 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    fd = os.open(root, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    root.chmod(0o555)


def entry(root: Path, path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def preserve_failure(output: Path, error: BaseException) -> None:
    if output.exists():
        write_json(output.with_name(output.name + ".failed.json"), {
            "status": "FAILED_CURRENT_P0_SIDECAR_NO_ADMISSION", "error": f"{type(error).__name__}: {error}"})


def produce(p: dict[str, Any], protocol_ref: dict[str, str], admission_ref: dict[str, str],
            final_guard: Any = None) -> dict[str, Any]:
    import numpy as np
    from scripts import sf_current_p0_sidecar_schema as schema

    phase_authority(p, protocol_ref, admission_ref)
    audit, census, report, generation, producer = inputs(p)
    require(p["phase"] == "produce" and p["candidate"] is None and p["producer_owner"] is None, "producer phase")
    output = fresh_output(p, generation, report)
    output.mkdir()
    try:
        games, files, counts, total_bytes = [], [], Counter(), 0
        for ordinal, (item, old_rows) in enumerate(census_games(audit, report, generation, p)):
            metadata, arrays = expected_game(audit, item, generation, producer, p["limits"], old_rows)
            metadata_size = len(canonical(metadata)) + 1
            require(metadata_size <= p["limits"]["game_metadata_bytes"], "carrier game metadata cap")
            # NPY headers accepted by the validator are at most 4096 bytes,
            # plus the 12-byte v2 prefix. Reserve before writing the game.
            reserve = metadata_size + schema.decoded_bytes(item["rows"]) + len(schema.FIELDS) * 4108
            require(total_bytes + reserve <= p["limits"]["carrier_total_bytes"], "carrier output reservation cap")
            folder = output / f"game-{ordinal:06d}"
            folder.mkdir()
            write_json(folder / "rows.json", metadata)
            require((folder / "rows.json").stat().st_size <= p["limits"]["game_metadata_bytes"], "carrier game metadata cap")
            for key, array in arrays.items():
                with (folder / (key + ".npy")).open("xb") as stream:
                    np.save(stream, array, allow_pickle=False)
                    stream.flush()
                    os.fsync(stream.fileno())
            new_files = [entry(output, f) for f in sorted(folder.iterdir())]
            files.extend(new_files)
            total_bytes += sum(f["bytes"] for f in new_files)
            require(total_bytes <= p["limits"]["carrier_total_bytes"], "carrier total byte cap")
            games.append({"physical_game_id": item["physical_game_id"], "rows": item["rows"], "directory": folder.name})
            for row in metadata["rows_detail"]:
                counts["rows"] += 1
                counts["status:" + row["source_join"]["status"]] += 1
                for field, value in row["qualified_flags"].items():
                    counts["qualified_" + field + ":" + str(value)] += 1
            del arrays, metadata, old_rows
        require(counts["rows"] == report["rows"], "carrier full denominator")
        if final_guard is not None:
            final_guard()
        result = {"schema": schema.SCHEMA, "status": "CANDIDATE_CURRENT_P0_SF_SIDECAR_V1", "scope": p["scope"],
                  "producer_protocol": protocol_ref, "execution_admission": admission_ref,
                  "selection": p["selection"], "census_protocol": p["census_protocol"],
                  "census_readout": p["census_readout"], "generation_manifest": census["generation_manifest"],
                  "source": p["source"], "physical_games": len(games), "rows": report["rows"], "games": games,
                  "files": files, "bytes": total_bytes, "counts": dict(sorted(counts.items())),
                  "wire_authenticated": False, "training_admitted": False, "root_postclosure_acceptance_required": True}
        require(total_bytes + len(canonical(result)) + 1 <= p["limits"]["carrier_total_bytes"],
                "carrier manifest/output byte cap")
        write_json(output / "manifest.json", result)
        seal(output)
        return result
    except BaseException as error:
        preserve_failure(output, error)
        raise


def candidate_inventory(audit: Any, ref: dict[str, str], cap: dict[str, int]) -> dict[str, Any]:
    from scripts import sf_current_p0_sidecar_schema as schema

    candidate = audit.checked(ref, cap["metadata_bytes"])
    root = Path(ref["path"]).parent
    require(Path(ref["path"]).name == "manifest.json" and root.resolve() == root
            and not root.stat().st_mode & 0o222
            and not root.with_name(root.name + ".failed.json").exists(), "candidate seal/failure marker")
    require(candidate["schema"] == schema.SCHEMA and candidate["status"] == "CANDIDATE_CURRENT_P0_SF_SIDECAR_V1"
            and candidate["training_admitted"] is False and candidate["root_postclosure_acceptance_required"] is True,
            "candidate is not an admitted dataset")
    files = candidate["files"]
    require(sum(f["bytes"] for f in files) == candidate["bytes"]
            and candidate["bytes"] + audit.stamp(Path(ref["path"]))[2] <= cap["carrier_total_bytes"],
            "candidate byte accounting")
    listed = [f["path"] for f in files]
    require(len(listed) == len(set(listed)) and len(files) <= cap["inventory_files"], "candidate inventory duplicate/cap")
    actual = [f for f in root.rglob("*") if f.is_file()]
    require(not any(f.is_symlink() for f in root.rglob("*")) and {str(f.relative_to(root)) for f in actual} == set(listed) | {"manifest.json"}, "candidate complete inventory")
    directories = [f for f in root.rglob("*") if f.is_dir()]
    require({str(f.relative_to(root)) for f in directories} == {g["directory"] for g in candidate["games"]}
            and not any(f.stat().st_mode & 0o222 for f in directories), "candidate directory inventory/seal")
    expected_files = {str(Path(g["directory"]) / name) for g in candidate["games"]
                      for name in ["rows.json", *(key + ".npy" for key in schema.FIELDS)]}
    require(set(listed) == expected_files, "unknown/missing carrier payload")
    for f in files:
        name = Path(f["path"])
        require(not name.is_absolute() and ".." not in name.parts and "\\" not in str(name), "candidate unsafe member")
        path = root / name
        require(audit.stamp(path)[2] == f["bytes"] and audit.file_sha(path) == f["sha256"], "candidate payload hash")
    return candidate


def validate(p: dict[str, Any], protocol_ref: dict[str, str], admission_ref: dict[str, str],
             final_guard: Any = None) -> dict[str, Any]:
    import numpy as np
    from scripts import sf_current_p0_sidecar_schema as schema

    phase_authority(p, protocol_ref, admission_ref)
    audit, census, report, generation, producer = inputs(p)
    require(p["phase"] == "validate" and p["candidate"] is not None and p["producer_owner"] is not None, "validator phase")
    # Authenticate compact lineage and CLOSED producer before touching carrier
    # numeric payloads, including their inventory hashes.
    candidate = audit.checked(p["candidate"], p["limits"]["metadata_bytes"])
    producer_p = audit.checked(candidate["producer_protocol"], p["limits"]["metadata_bytes"])
    require(producer_p["phase"] == "produce"
            and Path(producer_p["output_root"]) / "manifest.json" == Path(p["candidate"]["path"]),
            "candidate producer output/protocol")
    phase_authority(producer_p, candidate["producer_protocol"], candidate["execution_admission"])
    for key in p:
        if key not in {"phase", "candidate", "producer_owner", "output_root", "runtime"}:
            require(p[key] == producer_p[key], "validator changed producing source/context contract: " + key)
    owner = p["producer_owner"]
    terminal = audit.checked(owner["terminal"], p["limits"]["metadata_bytes"], readonly=False)
    closed(terminal, owner["command"], owner["owner_sha256"], producer_p["runtime"]["owner_seconds"])
    require(owner["command"][-8:] == ["--protocol", candidate["producer_protocol"]["path"],
            "--protocol-sha256", candidate["producer_protocol"]["sha256"], "--admission",
            candidate["execution_admission"]["path"], "--admission-sha256", candidate["execution_admission"]["sha256"]],
            "producer command protocol/admission")
    require(candidate_inventory(audit, p["candidate"], p["limits"]) == candidate,
            "candidate changed before payload validation")
    require(candidate["scope"] == p["scope"] and candidate["census_readout"] == p["census_readout"]
            and candidate["census_protocol"] == p["census_protocol"] and candidate["source"] == p["source"]
            and candidate["selection"] == p["selection"]
            and candidate["generation_manifest"] == census["generation_manifest"], "candidate source lineage")
    require(candidate["rows"] == report["rows"] and candidate["physical_games"] == len(generation["games"])
            and len(candidate["games"]) == len(generation["games"]), "candidate full population")
    root = Path(p["candidate"]["path"]).parent
    output = fresh_output(p, generation, report)
    output.mkdir()
    try:
        rows, counts = 0, Counter()
        for ordinal, (item, old_rows) in enumerate(census_games(audit, report, generation, p)):
            info = candidate["games"][ordinal]
            require(info == {"physical_game_id": item["physical_game_id"], "rows": item["rows"], "directory": f"game-{ordinal:06d}"}, "candidate game order/identity")
            expected, arrays = expected_game(audit, item, generation, producer, p["limits"], old_rows)
            folder = root / info["directory"]
            member = next(f for f in candidate["files"] if f["path"] == str(Path(info["directory"]) / "rows.json"))
            actual = audit.checked({"path": str(folder / "rows.json"), "sha256": member["sha256"]},
                                   p["limits"]["game_metadata_bytes"])
            require(canonical(actual) == canonical(expected), "carrier identity/status/value provenance mismatch")
            loaded = {name: schema.load_npy(folder / (name + ".npy"), name, item["rows"], p["limits"]["carrier_game_bytes"])
                      for name in schema.FIELDS}
            schema.check_arrays(loaded, item["rows"], p["limits"]["carrier_game_bytes"])
            for name in schema.FIELDS:
                require(np.array_equal(loaded[name], arrays[name]) and loaded[name].tobytes(order="C") == arrays[name].tobytes(order="C"), "source/carrier vector bytes differ: " + name)
            rows += item["rows"]
            for row in actual["rows_detail"]:
                counts["rows"] += 1
                counts["status:" + row["source_join"]["status"]] += 1
                for field, value in row["qualified_flags"].items():
                    counts["qualified_" + field + ":" + str(value)] += 1
            del arrays, loaded, expected, actual, old_rows
        require(rows == candidate["rows"] and dict(sorted(counts.items())) == candidate["counts"], "candidate counts/denominator")
        require(candidate_inventory(audit, p["candidate"], p["limits"]) == candidate, "candidate custody changed")
        if final_guard is not None:
            final_guard()
        result = {"status": "VALIDATION_CANDIDATE_CURRENT_P0_SF_SIDECAR_V1", "scope": p["scope"],
                  "protocol": protocol_ref, "execution_admission": admission_ref,
                  "candidate": p["candidate"], "rows": rows, "physical_games": len(candidate["games"]),
                  "source_payload_and_vector_bytes_recomputed": True, "wire_authenticated": False,
                  "training_admitted": False, "root_postclosure_acceptance_required": True,
                  "limits": "Independent process revalidation, shared source algorithms; not an independently authored target-quality proof."}
        write_json(output / "readout.json", result)
        seal(output)
        return result
    except BaseException as error:
        preserve_failure(output, error)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(REPO))
    sys.meta_path.insert(0, NoForbiddenImports())
    from scripts import sf_banked_p0_census as audit

    protocol_ref = {"path": str(args.protocol), "sha256": args.protocol_sha256}
    admission_ref = {"path": str(args.admission), "sha256": args.admission_sha256}
    p = audit.checked(protocol_ref, 4 << 20)
    runtime_guard(p)
    limits(p)
    phase_authority(p, protocol_ref, admission_ref)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("fixed sidecar CPU deadline")))
    signal.alarm(p["runtime"]["inner_seconds"])
    try:
        def final_guard() -> None:
            require(audit.checked(protocol_ref, 4 << 20) == p, "sidecar protocol changed")
            phase_authority(p, protocol_ref, admission_ref)
            runtime_guard(p)
        function = produce if p["phase"] == "produce" else validate
        function(p, protocol_ref, admission_ref, final_guard=final_guard)
    finally:
        signal.alarm(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
