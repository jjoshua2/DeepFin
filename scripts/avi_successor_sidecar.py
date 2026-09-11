#!/usr/bin/env python3
"""Collect frozen-network root and all-legal one-ply WDL on derived SF rows.

The collector is deliberately provenance-gated. A replay shard's stored float16 input
is not enough to reconstruct legal children with authentic history, so every selected
row must carry ``row_provenance.npz`` pointing to its original schema-3 raw corpus row.
The raw row is replayed from ``history_root_fen`` + ``history_uci`` and its original
``input_key`` is reverified before any neural label is accepted.

This tool banks raw observations only. It never rewrites training targets. Use
``avi_value_rewrite.py`` for a separately pinned root-distillation or successor-backup
corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import chess
import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_anti_engine.encoding import encode_positions_batch
from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.mcts.one_ply import evaluate_wdl_probabilities, one_ply_value_backups
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint
from scripts import bt4_derived_wdl_sidecar as derived
from scripts import corpus_row_provenance as provenance
from scripts import gen_sf_rooted_corpus as corpus
from scripts.adapt_raw_bt4_sidecars import storage_identity
from scripts.bt4_policy_dump import file_sha256
from scripts.sf_policy_rewrite import require

SUMMARY = "avi_successor_sidecar_summary.json"
SCHEMA = 1
KIND = "avi_successor_sidecar"
SUFFIX = ".avi_values.npz"


def sidecar_name(shard_name: str) -> str:
    require(shard_name.startswith("shard_") and shard_name.endswith(".zarr"), "invalid shard name")
    return shard_name.removesuffix(".zarr") + SUFFIX


def _namespace(source_dir: Path, config_sha256: str) -> str:
    payload = json.dumps([str(source_dir.resolve()), config_sha256], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _raw_row_board(row: dict[str, Any], ref: dict[str, Any]) -> chess.Board:
    """Replay and authenticate one raw history row referenced by derived provenance."""
    require(int(row.get("schema", 0)) >= 3, "AVI requires schema-3 raw history rows")
    run = row.get("run")
    require(isinstance(run, dict), "raw row missing run identity")
    require(run.get("config_sha256") == ref["source_config_sha256"], "raw config differs")
    require(
        int(row.get("worker_id", -1)) == int(ref["worker_id"])
        and int(row.get("game_id", -1)) == int(ref["game_id"])
        and int(row.get("ply", -1)) == int(ref["ply"]),
        "raw worker/game/ply differs",
    )
    require(row.get("input_key") == ref["input_key"], "raw input_key differs")
    root = row.get("history_root_fen")
    moves = row.get("history_uci")
    require(isinstance(root, str) and isinstance(moves, list), "raw history window missing")
    require(int(row.get("history_plies", -1)) == len(moves), "raw history length differs")
    board = chess.Board(root)
    for uci in moves:
        require(isinstance(uci, str), "non-string raw history move")
        move = chess.Move.from_uci(uci)
        require(move in board.legal_moves, "illegal move in raw history window")
        board.push(move)
    require(board.fen() == row.get("fen"), "raw history replay does not reproduce FEN")
    corpus.apply_history_rep_fix()
    require(corpus.row_key(board) == ref["input_key"], "replayed history input_key differs")
    return board


def _requested_raw_rows(refs: list[dict[str, Any]]) -> tuple[list[chess.Board], dict[str, str]]:
    """Resolve provenance to raw rows, scanning each referenced raw shard once."""
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
    for output_index, ref in enumerate(refs):
        source_dir = Path(str(ref["source_dir"]))
        require(source_dir.is_absolute() and source_dir == source_dir.resolve(), "raw source path is not canonical")
        require(
            ref["source_namespace"] == _namespace(source_dir, str(ref["source_config_sha256"])),
            "raw source namespace differs",
        )
        shard = str(ref["source_shard"])
        require(Path(shard).name == shard, "unsafe raw shard name")
        grouped.setdefault((str(source_dir), shard), []).append((output_index, ref))

    boards: list[chess.Board | None] = [None] * len(refs)
    identities: dict[str, str] = {}
    for (source_dir_text, shard_name), requests in grouped.items():
        source_dir = Path(source_dir_text)
        path = source_dir / shard_name
        require(path.is_file(), f"raw source shard missing: {path}")
        state = storage_identity(path)
        identities[str(path)] = state
        wanted = {int(ref["source_row"]): (index, ref) for index, ref in requests}
        require(len(wanted) == len(requests), "duplicate raw physical row requested")
        remaining = set(wanted)
        for raw_index, row in enumerate(corpus.iter_shard_rows(path)):
            if raw_index not in remaining:
                continue
            output_index, ref = wanted[raw_index]
            boards[output_index] = _raw_row_board(row, ref)
            remaining.remove(raw_index)
            if not remaining:
                break
        require(not remaining, f"raw source rows missing from {path.name}: {sorted(remaining)[:8]}")
        require(storage_identity(path) == state, "raw source changed during history replay")
    require(all(board is not None for board in boards), "unresolved raw history rows")
    return [board for board in boards if board is not None], identities


def _sha_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _root_feed_matches_source(
    boards: list[chess.Board],
    stored_x: np.ndarray,
    *,
    history_encoding: str | None,
    extra_features: str | None,
) -> None:
    encoded = encode_positions_batch(
        boards,
        input_history_encoding=history_encoding,
        input_extra_features=extra_features,
    )
    require(encoded.shape == stored_x.shape, "replayed root input shape differs from source x")
    require(
        np.array_equal(np.asarray(encoded, dtype=np.float16), stored_x),
        "replayed root input differs from stored source x after float16 quantization",
    )


def collect(args: argparse.Namespace) -> dict[str, Any]:
    require(type(args.parent_batch) is int and args.parent_batch > 0, "parent batch must be positive")
    require(type(args.batch_size) is int and args.batch_size > 0, "batch size must be positive")
    require(float(args.minimum_free_gib) >= 0, "minimum free GiB must be non-negative")
    source = Path(args.source).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    out = Path(args.out).resolve()
    writing = out.with_name(out.name + ".writing")
    require(checkpoint.is_file(), "checkpoint missing")
    require(file_sha256(checkpoint) == args.expected_checkpoint_sha256, "checkpoint SHA256 differs")
    require(not out.exists() and not writing.exists(), "new output required")
    require(out != source and source not in out.parents and out not in source.parents, "output overlaps source")

    summary, specs = derived.source_inventory(args)
    source_rows = int(summary["realized"]["rows_written"])
    selected_rows = sum(int(spec["rows"]) for spec in specs)
    require(selected_rows > 0, "empty shard selection")

    model = load_model_from_checkpoint(str(checkpoint), device=str(args.device))
    model.eval()
    history_encoding = getattr(model, "input_history_encoding", None)
    extra_features = getattr(model, "input_extra_features", None)
    compute_relations = bool(getattr(model, "use_dynamic_relations", False))
    require(
        history_encoding == summary["input"].get("input_history_encoding")
        and extra_features == summary["input"].get("input_extra_features"),
        "teacher checkpoint input encoding differs from source corpus",
    )
    evaluator = LocalModelEvaluator(model, device=str(args.device), use_amp=True)

    writing.mkdir(parents=True)
    source_summary = source / derived.SUMMARY
    pins = {source_summary: args.expected_source_summary_sha256, checkpoint: args.expected_checkpoint_sha256}
    source_states = {source / spec["path"]: storage_identity(source / spec["path"]) for spec in specs}
    producer = {
        str(Path(__file__).resolve()): file_sha256(Path(__file__).resolve()),
        str(Path(sys.modules["chess_anti_engine.mcts.one_ply"].__file__).resolve()): file_sha256(
            Path(sys.modules["chess_anti_engine.mcts.one_ply"].__file__).resolve()
        ),
        str(Path(provenance.__file__).resolve()): file_sha256(Path(provenance.__file__).resolve()),
        str(Path(corpus.__file__).resolve()): file_sha256(Path(corpus.__file__).resolve()),
    }
    outputs: list[dict[str, Any]] = []
    started = time.monotonic()
    total_children = total_network_children = total_terminal_children = 0
    try:
        for spec in specs:
            if (writing / "STOP").exists() or (out.parent / "STOP").exists():
                raise RuntimeError("STOP requested")
            require(
                shutil.disk_usage(writing).free >= float(args.minimum_free_gib) * 1024**3,
                "disk reserve breached",
            )
            name, rows = str(spec["path"]), int(spec["rows"])
            path = source / name
            require(storage_identity(path) == source_states[path], "source shard changed")
            group: Any = zarr.open_group(str(path), mode="r")
            stamp = dict(group.attrs).get("derive_row_provenance")
            require(isinstance(stamp, dict), "source shard has no row provenance")
            require(
                stamp.get("schema") == provenance.SCHEMA
                and stamp.get("path") == provenance.FILENAME
                and int(stamp.get("rows", -1)) == rows,
                "row provenance stamp differs",
            )
            provenance_path = path / provenance.FILENAME
            require(file_sha256(provenance_path) == stamp.get("sha256"), "row provenance SHA256 differs")
            refs = provenance.read(provenance_path, rows=rows)
            stored_x = np.asarray(group["x"][:])
            require(stored_x.dtype == np.float16 and len(stored_x) == rows, "source x layout differs")
            for i, ref in enumerate(refs):
                require(
                    ref["stored_input_key"] == corpus.input_tensor_key(stored_x[i]),
                    "row provenance does not bind stored x",
                )
            boards, raw_states = _requested_raw_rows(refs)
            _root_feed_matches_source(
                boards,
                stored_x,
                history_encoding=history_encoding,
                extra_features=extra_features,
            )

            root_parts: list[np.ndarray] = []
            backup_parts: list[np.ndarray] = []
            chosen: list[str] = []
            chosen_q: list[float] = []
            legal_counts: list[int] = []
            terminal_counts: list[int] = []
            evaluated_counts: list[int] = []
            selected_terminal: list[bool] = []
            for start in range(0, rows, args.parent_batch):
                chunk = boards[start : start + args.parent_batch]
                root_parts.append(
                    evaluate_wdl_probabilities(
                        chunk,
                        evaluator,
                        batch_size=args.batch_size,
                        input_history_encoding=history_encoding,
                        input_extra_features=extra_features,
                        compute_relations=compute_relations,
                    )
                )
                backups = one_ply_value_backups(
                    chunk,
                    evaluator,
                    batch_size=args.batch_size,
                    input_history_encoding=history_encoding,
                    input_extra_features=extra_features,
                    compute_relations=compute_relations,
                    claim_draws=bool(args.claim_draws),
                )
                backup_parts.append(np.stack([item.wdl for item in backups]))
                chosen.extend(item.move.uci() for item in backups)
                chosen_q.extend(item.q for item in backups)
                legal_counts.extend(item.legal_moves for item in backups)
                terminal_counts.extend(item.terminal_children for item in backups)
                evaluated_counts.extend(item.evaluated_children for item in backups)
                selected_terminal.extend(item.selected_terminal for item in backups)
            root_wdl = np.concatenate(root_parts).astype(np.float32, copy=False)
            backup_wdl = np.concatenate(backup_parts).astype(np.float32, copy=False)
            require(root_wdl.shape == backup_wdl.shape == (rows, 3), "collected WDL shape differs")
            require(
                np.isfinite(root_wdl).all()
                and np.isfinite(backup_wdl).all()
                and np.allclose(root_wdl.sum(axis=1), 1, atol=2e-6, rtol=0)
                and np.allclose(backup_wdl.sum(axis=1), 1, atol=2e-6, rtol=0),
                "collected WDL mass differs",
            )
            sidecar = writing / sidecar_name(name)
            with sidecar.open("xb") as handle:
                np.savez_compressed(
                    handle,
                    root_wdl=root_wdl,
                    backup_wdl=backup_wdl,
                    chosen_move=np.asarray(chosen, dtype="S5"),
                    chosen_q=np.asarray(chosen_q, dtype=np.float32),
                    legal_moves=np.asarray(legal_counts, dtype=np.int16),
                    terminal_children=np.asarray(terminal_counts, dtype=np.int16),
                    evaluated_children=np.asarray(evaluated_counts, dtype=np.int16),
                    selected_terminal=np.asarray(selected_terminal, dtype=np.bool_),
                )
            total_children += sum(legal_counts)
            total_network_children += sum(evaluated_counts)
            total_terminal_children += sum(terminal_counts)
            outputs.append(
                {
                    "source_shard": name,
                    "rows": rows,
                    "path": sidecar.name,
                    "sha256": file_sha256(sidecar),
                    "source_storage_identity": source_states[path],
                    "row_provenance_sha256": stamp["sha256"],
                    "raw_storage_identities": raw_states,
                    "root_wdl_sha256": _sha_array(root_wdl),
                    "backup_wdl_sha256": _sha_array(backup_wdl),
                    "legal_children": int(sum(legal_counts)),
                    "network_evaluated_children": int(sum(evaluated_counts)),
                    "terminal_children": int(sum(terminal_counts)),
                    "selected_terminal_rows": int(sum(selected_terminal)),
                }
            )
            require(storage_identity(path) == source_states[path], "source shard changed after collection")
            for raw_path, state in raw_states.items():
                require(storage_identity(Path(raw_path)) == state, "raw history shard changed after collection")

        for path, digest in pins.items():
            require(file_sha256(path) == digest, "pinned input changed during collection")
        for path, state in source_states.items():
            require(storage_identity(path) == state, "source changed before publication")
        payload = {
            "schema": SCHEMA,
            "status": "COMPLETE_SELECTION",
            "kind": KIND,
            "source_dir": str(source),
            "source_summary_sha256": args.expected_source_summary_sha256,
            "source_rows": source_rows,
            "selected_rows": selected_rows,
            "selected_shards": len(specs),
            "start_shard": int(args.start_shard),
            "max_shards": int(args.max_shards),
            "full_source_coverage": selected_rows == source_rows and len(specs) == len(summary["shards"]),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": args.expected_checkpoint_sha256,
            "device": str(args.device),
            "batch_size": int(args.batch_size),
            "parent_batch": int(args.parent_batch),
            "claim_draws": bool(args.claim_draws),
            "wdl_order": "WDL",
            "wdl_pov": "side_to_move_at_root_after_backup",
            "input_history_encoding": history_encoding,
            "input_extra_features": extra_features,
            "compute_relations": compute_relations,
            "legal_children": total_children,
            "network_evaluated_children": total_network_children,
            "terminal_children": total_terminal_children,
            "elapsed_seconds": float(time.monotonic() - started),
            "producer_sha256": producer,
            "outputs": outputs,
        }
        (writing / SUMMARY).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(writing, out)
        return payload
    except BaseException as error:
        (writing / "failed.json").write_text(json.dumps({"complete": False, "error": str(error)}) + "\n")
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-summary-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--parent-batch", type=int, default=128)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--minimum-free-gib", type=float, default=150)
    parser.add_argument(
        "--claim-draws",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match DeepFin's declared threefold/50-move-as-draw convention; twofold remains nonterminal",
    )
    return parser


if __name__ == "__main__":
    collect(build_parser().parse_args())
