#!/usr/bin/env python3
"""CPU-only BT4 own-game teacher source; no inference or SF provenance.

Publishes the saved root policy and native WDL in separate teacher channels.
It deliberately does not produce replay ``policy_target`` or ``search_wdl``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import chess
import numpy as np
import zarr
from numcodecs import Blosc

from chess_anti_engine.eval.rvg_surgery import position_fingerprints
from chess_anti_engine.moves.leela_index import compact_index_for_move

SCHEMA = "bt4_own_teacher_source_v1"
_COMPRESSOR = Blosc(cname="zstd", clevel=2, shuffle=Blosc.BITSHUFFLE)
MAX_SOURCE_ROWS = 100_000  # bounded first adapter; large-corpus streaming needs a separate design


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_audit(bank: Path, summary_sha256: str, rows: int,
                 audit_path: Path, audit_sha256: str) -> dict[str, str]:
    audit_path = audit_path.resolve(strict=True)
    require(sha_file(audit_path) == audit_sha256, "full-bank audit pin differs")
    audit = json.loads(audit_path.read_text())
    status = audit.get("status")
    if status == "PASS_SAVED_BT4_TWO_GAME_BANK_AUDIT":
        require(audit.get("bank") == str(bank) and audit.get("facts", {}).get("summary_sha256") == summary_sha256
                and audit.get("facts", {}).get("accepted_rows") == rows,
                "saved-bank independent audit does not bind this bank")
    elif status == "PASS_INDEPENDENT_BT4_V4_READBACK":
        require(audit.get("new_summary_sha256") == summary_sha256
                and audit.get("new_accepted_rows") == rows
                and audit.get("overall_status") == "PASS_BT4_ROOT_CUDA_7_TO_6_CONSERVATIVE_CONTINUATION"
                and bool(audit.get("source_origins_verified")),
                "v4 independent audit does not bind this bank")
    else:
        raise ValueError("unsupported full-bank audit receipt; source cannot be qualified")
    return {"path": str(audit_path), "sha256": audit_sha256, "status": status}


def inspect_bank(bank: Path, expected_summary_sha256: str,
                 audit_path: Path, audit_sha256: str) -> dict[str, Any]:
    bank = bank.resolve(strict=True)
    summary_path = bank / "summary.json"
    require(sha_file(summary_path) == expected_summary_sha256, "bank summary pin differs")
    summary = json.loads(summary_path.read_text())
    require(summary.get("schema") == "bt4_root_policy_games_v1"
            and summary.get("status") == "complete", "bank is not complete")
    require(isinstance(summary.get("game_files"), list), "missing game receipts")
    launch_path = bank / "launch.json"
    require(sha_file(launch_path) == summary.get("launch_sha256"), "launch pin differs")
    launch = json.loads(launch_path.read_text())
    require(summary.get("games") == launch.get("games") and type(launch.get("games")) is int
            and launch["games"] > 0, "summary/launch game count differs")
    require(launch.get("schema") == "bt4_root_policy_games_v1"
            and launch.get("outcome_mode") == "rule50_match_v1"
            and launch.get("history_rep_fix") is True
            and launch.get("input_extra_features") == "v2_threats"
            and launch.get("teacher_observation") ==
            "root_inference_compact_t1_policy_and_native_wdl_unmodified_by_outcome",
            "launch source/label contract differs")
    model = launch.get("model", {})
    require(isinstance(model, dict) and len(str(model.get("sha256", ""))) == 64,
            "launch model pin missing")
    if model.get("requested_provider") == "cuda":
        proof_path = bank / "provider_proof.json"
        require(sha_file(proof_path) == summary.get("provider_proof_sha256"), "CUDA proof pin differs")
        proof = json.loads(proof_path.read_text())
        providers_after = proof.get("providers_after_first_call")
        require(proof.get("cuda_neural_nodes", 0) >= 1
                and isinstance(providers_after, list) and bool(providers_after)
                and providers_after[0] == "CUDAExecutionProvider",
                "saved CUDA provider proof differs")
    else:
        require(model.get("requested_provider") == "cpu", "unsupported provider")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    seen_games: set[int] = set()
    seen_names: set[str] = set()
    skipped = 0
    attempted = 0
    discarded_by_termination: dict[str, int] = {}
    teacher_contract: dict[str, Any] | None = None
    for receipt in summary["game_files"]:
        name = receipt["path"]
        require(isinstance(name, str) and Path(name).name == name, "unsafe game name")
        game_id = receipt["game_id"]
        require(type(game_id) is int and 0 <= game_id < launch["games"]
                and game_id not in seen_games and name == f"game_{game_id:08d}.npz"
                and name not in seen_names, "duplicate or unexpected game receipt")
        seen_games.add(game_id)
        seen_names.add(name)
        path = bank / "games" / name
        require(path.is_file() and not path.is_symlink(), "missing or aliased game")
        require(sha_file(path) == receipt["sha256"], "game receipt SHA differs")
        with np.load(path, allow_pickle=False) as payload:
            require(set(payload.files) == {"x", "policy_t1", "wdl_raw", "metadata"}, "game arrays differ")
            meta = json.loads(payload["metadata"].tobytes())
            source_rows = meta["rows"]
            require(meta["schema"] == "bt4_root_policy_games_v1", "game schema differs")
            require(meta["game_id"] == receipt["game_id"], "game ID differs")
            require(meta["status"] == receipt["status"], "game status differs")
            require(meta["initial_fen"] == launch["initial_fen"], "game initial FEN differs from launch")
            require(len(source_rows) == receipt["rows"], "row count differs")
            require(meta["outcome_provenance"]["mode"] == "rule50_match_v1", "outcome mode differs")
            if meta["status"] == "discarded":
                require(not source_rows and receipt["rows"] == 0
                        and meta["discarded_rows"] == receipt["discarded_rows"] == meta["attempted_plies"],
                        "discard accounting differs")
                require(all(payload[key].shape[0] == 0 for key in ("x", "policy_t1", "wdl_raw")),
                        "discarded game still exposes teacher arrays")
                skipped += 1
                attempted += int(meta["attempted_plies"])
                term = str(meta["termination"])
                discarded_by_termination[term] = discarded_by_termination.get(term, 0) + 1
                continue
            require(meta["status"] == "completed" and meta["result"] in ("1-0", "0-1", "1/2-1/2"),
                    "game has no terminal result")
            require(meta["attempted_plies"] == receipt["rows"]
                    and meta["discarded_rows"] == receipt["discarded_rows"] == 0,
                    "completed-game accounting differs")
            attempted += int(meta["attempted_plies"])
            board = chess.Board(meta["initial_fen"])
            x, policy, wdl = (payload[key] for key in ("x", "policy_t1", "wdl_raw"))
            n = len(source_rows)
            require(x.shape == (n, 175, 8, 8) and x.dtype == np.dtype("float32"), "x layout differs")
            require(policy.shape == (n, 1858) and policy.dtype == np.dtype("float32"), "policy layout differs")
            require(wdl.shape == (n, 3) and wdl.dtype == np.dtype("float32"), "WDL layout differs")
            require(np.isfinite(x).all() and np.isfinite(policy).all() and np.isfinite(wdl).all(),
                    "nonfinite teacher payload")
            require(bool(np.all(policy >= 0)) and bool(np.allclose(policy.sum(1, dtype=np.float64), 1, atol=2e-6, rtol=0)),
                    "policy is not a distribution")
            require(bool(np.all(wdl >= 0)) and bool(np.allclose(wdl.sum(1, dtype=np.float64), 1, atol=2e-6, rtol=0)),
                    "native WDL is not a probability distribution")
            source_keys = position_fingerprints(x, input_history_encoding=launch["input_history_encoding"])
            for i, row in enumerate(source_rows):
                teacher = row["teacher"]
                require(row["index"] == i and row["ply_index"] == board.ply(),
                        "row/absolute-ply order differs")
                require(board.fen() == row["fen"] and board.turn == row["pov_white"],
                        "actor history/FEN differs")
                require(teacher["kind"] == "root_inference_no_search", "teacher kind differs")
                require(teacher["policy_encoding"] == "lc0_1858_compact", "policy encoding differs")
                require(teacher["wdl_kind"] == "probabilities" and teacher["wdl_dtype"] == "float32",
                        "native WDL contract differs")
                require(teacher["wdl_order"] == ["win", "draw", "loss"] and teacher["wdl_pov"] == "side_to_move",
                        "native WDL axes differ")
                require(teacher["input_extra_features"] == "v2_threats" and teacher["history_rep_fix"] is True,
                        "input representation differs")
                require(teacher["input_history_encoding"] == launch["input_history_encoding"]
                        and teacher["model_sha256"] == model["sha256"],
                        "teacher model/input differs from launch")
                observed_contract = {key: teacher[key] for key in (
                    "model_sha256", "policy_output", "wdl_output", "wdl_kind", "wdl_dtype",
                    "input_name", "input_dtype", "input_history_encoding",
                    "input_extra_features", "history_rep_fix", "policy_encoding")}
                if teacher_contract is None:
                    teacher_contract = observed_contract
                require(observed_contract == teacher_contract, "game rows mix teacher contracts")
                require(hashlib.blake2b(np.ascontiguousarray(x[i], dtype=np.float32).tobytes(), digest_size=16).hexdigest()
                        == row["input_key"], "saved x/input_key differs")
                require(source_keys[i].hex() == row["source_key"], "saved x/source_key differs")
                legal = np.zeros((1858,), dtype=np.uint8)
                for move in board.legal_moves:
                    legal[compact_index_for_move(board, move)] = 1
                require(bool(np.all(policy[i][legal == 0] == 0)), "teacher policy has illegal mass")
                move = chess.Move.from_uci(row["move_uci"])
                require(move in board.legal_moves, "actor move is illegal")
                require(row["wdl_target"] in (0, 1, 2), "outcome target differs")
                expected_outcome = (1 if meta["result"] == "1/2-1/2" else
                                    0 if (meta["result"] == "1-0") == row["pov_white"] else 2)
                require(row["wdl_target"] == expected_outcome, "outcome target/result POV differs")
                identity = (int(meta["game_id"]), i)
                require(identity not in seen, "duplicate game/ply identity")
                seen.add(identity)
                board.push(move)
                rows.append({
                    "source_kind": "bt4_own_game_npz", "game_id": identity[0],
                    "ply_index": int(row["ply_index"]),
                    "game_file": str(path), "game_file_sha256": receipt["sha256"], "row_index": i,
                    "input_key": row["input_key"], "source_key": row["source_key"],
                    "fen": row["fen"], "temperature": row["temperature"],
                    "teacher_policy_field": "policy_t1", "teacher_wdl_field": "wdl_raw",
                    "teacher_model_sha256": teacher["model_sha256"],
                    "teacher_policy_output": teacher["policy_output"],
                    "teacher_wdl_output": teacher["wdl_output"],
                    "teacher_wdl_kind": teacher["wdl_kind"],
                    "actor_move_uci": row["move_uci"], "game_result": meta["result"],
                    "outcome_wdl_target": row["wdl_target"],
                })
    require(seen_games == set(range(launch["games"]))
            and len(seen_names) == len(summary["game_files"]) == launch["games"],
            "incomplete game receipt coverage")
    require({p.name for p in (bank / "games").iterdir()} == seen_names,
            "game file set differs from receipts")
    require(len(rows) == summary["rows_emitted"] and attempted == summary["rows_attempted"]
            and launch["games"] - skipped == summary["completed"]
            and discarded_by_termination == summary["discarded"],
            "summary game/row accounting differs")
    audit = verify_audit(bank, expected_summary_sha256, len(rows), audit_path, audit_sha256)
    return {
        "schema": "bt4_own_teacher_bank_v1", "training_ready": False,
        "bank": str(bank), "summary_sha256": expected_summary_sha256,
        "launch_sha256": summary["launch_sha256"],
        "provider_proof_sha256": summary.get("provider_proof_sha256"),
        "full_bank_audit": audit,
        "teacher_contract": teacher_contract,
        "outcome_mode": launch["outcome_mode"],
        "completed_rows": len(rows), "discarded_games": skipped,
        "mapping": {"bt4_policy": "policy_t1", "bt4_wdl_raw": "wdl_raw",
                    "terminal_outcome": "metadata.rows[].wdl_target"},
        "rows": rows,
    }


def write_source(banks: list[dict[str, Any]], out: Path) -> dict[str, Any]:
    """Publish a source-only Zarr cohort; never synthesize train targets."""
    require(bool(banks) and all(int(bank["completed_rows"]) > 0 for bank in banks),
            "each source bank needs completed rows")
    require(sum(int(bank["completed_rows"]) for bank in banks) <= MAX_SOURCE_ROWS,
            "source exceeds bounded first-adapter row cap")
    contract = banks[0]["teacher_contract"]
    require(all(bank["teacher_contract"] == contract and bank["outcome_mode"] == "rule50_match_v1"
                for bank in banks), "banks mix teacher/history/outcome contracts")
    namespaces = [str(bank["summary_sha256"]) for bank in banks]
    require(len(set(namespaces)) == len(namespaces), "duplicate bank summary namespace")
    out = out.absolute()
    writing = out.with_name(out.name + ".writing")
    require(not out.exists() and not writing.exists(), "output or incomplete stage already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    writing.mkdir()
    (writing / "incomplete.json").write_text(json.dumps({
        "schema": SCHEMA, "status": "incomplete", "source_summaries": namespaces,
    }, sort_keys=True) + "\n")
    shards: list[dict[str, object]] = []
    all_uids: set[bytes] = set()
    for bank_number, bank in enumerate(banks):
        bank_rows = bank["rows"]
        assert isinstance(bank_rows, list)
        n = len(bank_rows)
        group_name = f"bank_{bank_number:04d}.zarr"
        group: zarr.Group = zarr.open_group(str(writing / group_name), mode="w")
        chunk = min(128, n)
        specs = {
            "x": ((n, 175, 8, 8), "f4"),
            "bt4_policy": ((n, 1858), "f4"),
            "bt4_wdl_raw": ((n, 3), "f4"),
            "source_key": ((n, 16), "u1"),
            "input_key": ((n, 16), "u1"),
            "row_uid": ((n, 32), "u1"),
            "game_id": ((n,), "i8"),
            "ply_index": ((n,), "i4"),
            "outcome_wdl_target": ((n,), "i1"),
        }
        for name, (shape, dtype) in specs.items():
            group.create_dataset(name, shape=shape, dtype=dtype,
                                 chunks=(chunk, *shape[1:]), compressor=_COMPRESSOR)
        refs_name = f"bank_{bank_number:04d}.rows.jsonl"
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for offset, row in enumerate(bank_rows):
            assert isinstance(row, dict)
            grouped.setdefault(str(row["game_file"]), []).append((offset, row))
        with (writing / refs_name).open("w") as refs:
            for game_path_str, entries in grouped.items():
                game_path = Path(game_path_str)
                expected_game_sha = str(entries[0][1]["game_file_sha256"])
                require(sha_file(game_path) == expected_game_sha, "game changed before materialization")
                positions = [position for position, _ in entries]
                require(positions == list(range(positions[0], positions[-1] + 1)),
                        "game rows are not contiguous")
                with np.load(game_path, allow_pickle=False) as payload:
                    count = len(entries)
                    require(count == len(payload["policy_t1"]), "game row reference coverage differs")
                    first, last = positions[0], positions[-1] + 1
                    group["x"][first:last] = payload["x"]
                    group["bt4_policy"][first:last] = payload["policy_t1"]
                    group["bt4_wdl_raw"][first:last] = payload["wdl_raw"]
                    for stored, original in (("x", "x"), ("bt4_policy", "policy_t1"),
                                             ("bt4_wdl_raw", "wdl_raw")):
                        require(np.array_equal(group[stored][first:last], payload[original]),
                                f"written {stored} differs from saved teacher")
                    for position, row in entries:
                        row_index = int(row["row_index"])
                        require(row_index == position - first, "game row order differs")
                        uid = hashlib.sha256(bytes.fromhex(str(bank["summary_sha256"]))
                                             + bytes.fromhex(expected_game_sha)
                                             + row_index.to_bytes(8, "big")).digest()
                        require(uid not in all_uids, "cross-bank row identity collision")
                        all_uids.add(uid)
                        group["source_key"][position] = np.frombuffer(bytes.fromhex(str(row["source_key"])), dtype=np.uint8)
                        group["input_key"][position] = np.frombuffer(bytes.fromhex(str(row["input_key"])), dtype=np.uint8)
                        group["row_uid"][position] = np.frombuffer(uid, dtype=np.uint8)
                        group["game_id"][position] = int(row["game_id"])
                        group["ply_index"][position] = int(row["ply_index"])
                        group["outcome_wdl_target"][position] = int(row["outcome_wdl_target"])
                        refs.write(json.dumps({**row, "row_uid": uid.hex()}, sort_keys=True) + "\n")
                require(sha_file(game_path) == expected_game_sha, "game changed during materialization")
        array_sha256: dict[str, str] = {}
        for name in specs:
            digest = hashlib.sha256()
            for first in range(0, n, chunk):
                digest.update(np.ascontiguousarray(group[name][first:first + chunk]).tobytes())
            array_sha256[name] = digest.hexdigest()
        group.attrs.update({
            "schema": SCHEMA, "bank_summary_sha256": bank["summary_sha256"],
            "bank_launch_sha256": bank["launch_sha256"],
            "provider_proof_sha256": bank["provider_proof_sha256"],
            "teacher_contract": contract, "outcome_mode": "rule50_match_v1",
            "positions": n, "training_ready": False,
            "teacher_evaluations_per_position": 1, "search_nodes": 0,
            "policy_semantics": "BT4 root T1 prior, not search improved",
            "wdl_semantics": "BT4 native root probabilities, not search_wdl",
            "array_sha256": array_sha256,
        })
        shards.append({"path": group_name, "rows": n, "row_references": refs_name,
                       "row_references_sha256": sha_file(writing / refs_name),
                       "bank_summary_sha256": bank["summary_sha256"],
                       "array_sha256": array_sha256})
    manifest: dict[str, Any] = {
        "schema": SCHEMA, "status": "complete", "training_ready": False,
        "source_kind": "bt4_own_games", "producer_sha256": sha_file(Path(__file__)),
        "teacher_contract": contract, "outcome_mode": "rule50_match_v1",
        "teacher_fields": {"policy": "bt4_policy", "native_wdl": "bt4_wdl_raw"},
        "actor_outcome_fields": ["game_id", "ply_index", "outcome_wdl_target"],
        "missing_for_sf_factorial": ["sf_source_policy", "sf_search_wdl", "sf_raw_row_provenance", "ceres_labels"],
        "sources": [{key: bank[key] for key in ("bank", "summary_sha256", "launch_sha256",
                                                  "provider_proof_sha256", "full_bank_audit",
                                                  "completed_rows", "discarded_games")}
                    for bank in banks],
        "shards": shards, "rows": sum(int(bank["completed_rows"]) for bank in banks),
    }
    (writing / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    for bank in banks:
        root = Path(str(bank["bank"]))
        require(sha_file(root / "summary.json") == bank["summary_sha256"]
                and sha_file(root / "launch.json") == bank["launch_sha256"],
                "bank summary/launch changed during publication")
        audit = bank["full_bank_audit"]
        assert isinstance(audit, dict)
        require(sha_file(Path(audit["path"])) == audit["sha256"],
                "full-bank audit changed during publication")
        game_pins = {str(row["game_file"]): str(row["game_file_sha256"])
                     for row in bank["rows"]}
        for game_path, expected_sha in game_pins.items():
            require(sha_file(Path(game_path)) == expected_sha,
                    "bank game changed before publication")
    require(sha_file(Path(__file__)) == manifest["producer_sha256"], "producer changed during publication")
    (writing / "incomplete.json").unlink()
    os.replace(writing, out)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", action="append", type=Path, required=True)
    parser.add_argument("--summary-sha256", action="append", required=True)
    parser.add_argument("--full-bank-audit", action="append", type=Path, required=True)
    parser.add_argument("--audit-sha256", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    require(len(args.bank) == len(args.summary_sha256) == len(args.full_bank_audit) == len(args.audit_sha256),
            "one summary and audit pin required per bank")
    banks = [inspect_bank(path, pin, audit, audit_pin)
             for path, pin, audit, audit_pin in zip(args.bank, args.summary_sha256,
                                                    args.full_bank_audit, args.audit_sha256, strict=True)]
    result = write_source(banks, args.out)
    print(json.dumps({"shards": len(result["shards"]), "rows": result["rows"]}, sort_keys=True))


if __name__ == "__main__":
    main()
