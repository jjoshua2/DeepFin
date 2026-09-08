#!/usr/bin/env python3
"""Rewrite only policy on a pinned legacy uniform-d9 corpus, without inference.

The default q/.0005 mode is an identity control. Effective-cp/10 uses original
float64 scores (including mate encoding), never rounded rank-sidecar gaps.
Source history/value lineage is inherited, not freshly reconstructed or promoted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np
import zarr

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import compact_index_for_move
from scripts import derive_corpus_targets as derive
from scripts import sf_d9_rank_sidecar as rank
from scripts.bt4_policy_dump import file_sha256

SUMMARY = "sf_policy_rewrite_summary.json"
ARRAYS = frozenset(
    {
        "x",
        "policy_target",
        "legal_mask",
        "game_id",
        "ply_index",
        "wdl_target",
        "search_wdl",
        "priority",
        "is_selfplay",
        "is_network_turn",
        "has_game_id",
        "has_ply_index",
        "has_policy",
        "has_legal_mask",
        "has_search_wdl",
        "has_is_selfplay",
        "has_is_network_turn",
    }
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass
class Observation:
    game: int
    ply: int
    indices: np.ndarray
    scores: np.ndarray
    input_key: str


def observation(row: dict[str, Any], config_sha: str) -> Observation:
    """Full legal phase-zero roster, preserving original score/order precision."""
    derive._check_row_identity(row, config_sha)
    require(derive.row_schema_of(row) == 3, "raw row is not history schema3")
    derive.require_row_regime(row)
    key = row.get("input_key")
    if not isinstance(key, str) or len(key) != 32:
        raise ValueError("missing original history key")
    bytes.fromhex(key)
    board = chess.Board(row["fen"])
    require(row.get("stm") == ("w" if board.turn else "b"), "raw stm differs from FEN")
    require(
        row.get("piece_count") == chess.popcount(board.occupied),
        "raw piece count differs",
    )
    require(len(row["phases"]) == 1, "requires original single-phase d9 source")
    phase = row["phases"][0]
    legal_moves = list(board.legal_moves)
    width = len(legal_moves)
    require(
        width > 0
        and type(phase.get("index")) is int
        and phase["index"] == 0
        and phase.get("width_requested") == "all"
        and phase.get("searchmoves") is None,
        "phase0 is not full-width",
    )
    require(
        all(
            type(phase.get(k)) is int and phase[k] == width
            for k in ("width_realized", "width_streamed")
        ),
        "raw full-width counts differ",
    )
    require(phase.get("depth_requested") == 9, "source requested depth differs")
    blocks = [block for block in phase["per_depth"] if block.get("depth") == 9]
    require(
        len(blocks) == 1
        and type(blocks[0]["depth"]) is int
        and blocks[0].get("complete") is True,
        "malformed d9 completion",
    )
    lines = rank.d9_lines(row)
    require(
        len(lines) == width
        and all(
            len(line) == 4
            and type(line[0]) is int
            and line[0] == i
            and type(line[2]) in (int, float)
            for i, line in enumerate(lines, 1)
        ),
        "malformed full d9 roster",
    )
    # Existing rank validation checks order, finiteness, range and duplicates.
    ranked = rank.rank_observation(row, top_k=width)
    require(
        {str(line[1]) for line in lines} == {m.uci() for m in legal_moves},
        "incomplete legal d9 support",
    )
    indices = np.asarray(
        [compact_index_for_move(board, chess.Move.from_uci(line[1])) for line in lines],
        dtype=np.int64,
    )
    require(
        np.array_equal(indices, ranked.indices), "compact map differs from rank mapping"
    )
    scores = np.asarray([line[2] for line in lines], dtype=np.float64)
    require(bool(np.isfinite(scores).all()), "nonfinite raw score")
    return Observation(int(row["game_id"]), int(row["ply"]), indices, scores, key)


def target(obs: Observation, score_space: str, temperature: float) -> np.ndarray:
    require(score_space in {"q", "effective-cp"}, "unsupported policy score space")
    temperature = derive.validate_temp(temperature)
    values = (
        derive.gate.q_from_effective_cp(obs.scores, slope=0.006, draw_width_cp=120.0)
        if score_space == "q"
        else obs.scores
    )
    probabilities = derive.shard_stored(
        derive.softmax_at_temp(values, temp=temperature)
    )
    result = np.zeros(COMPACT_POLICY_SIZE, dtype=np.float16)
    result[obs.indices] = probabilities
    require(
        bool(np.isfinite(result).all() and np.all(result >= 0)), "invalid stored policy"
    )
    require(
        abs(float(result.astype(np.float64).sum()) - 1.0) <= 2**-10,
        "stored policy mass error",
    )
    return result


def source_contract(summary: dict[str, Any], record: derive.CorpusRecord) -> None:
    rows = summary["realized"]["rows_written"]
    require(
        record.corpus_complete and record.facts["row_schema"] == 3,
        "requires closed original schema3 corpus",
    )
    require(
        summary["realized"]["rows_read"] == summary["limit_requested"]
        and type(summary["limit_requested"]) is int
        and summary["limit_requested"] > 0
        and type(summary["rows_per_shard"]) is int
        and summary["rows_per_shard"] > 0,
        "invalid raw prefix/shard bounds",
    )
    require(
        summary["corpus"]["dir"] == record.shards[0].parent.name,
        "raw source directory differs",
    )
    require(
        summary.get("policy_target_postprocess") is None,
        "requires unmodified original SF policy",
    )
    require(
        summary["scheme"]["canonical"] == "uniform-d9"
        and summary["scheme"]["kind"] == "uniform"
        and summary["scheme"]["depth"] == 9
        and summary["scheme"].get("value_depth") is None
        and summary["scheme"]["value_source"] == "deepest_phase_covering",
        "source policy/value scheme differs",
    )
    require(
        summary["temp_requested"] == 0.0005 and summary["floor_requested"] == 0.0,
        "source q temperature/floor differs",
    )
    require(
        summary["cp_map"]["cp_slope"] == 0.006
        and summary["cp_map"]["cp_draw_width"] == 120.0,
        "source cp map differs",
    )
    require(summary["value_scheme"]["name"] == "search", "source value scheme differs")
    require(
        summary["input"]["input_history_encoding"] == "lc0_root_legacy_meta"
        and summary["input"]["history_rep_fix"] is True
        and summary["input"]["input_extra_features"] == "v2_threats"
        and summary["input"]["zero_history"] is False,
        "source history regime differs",
    )
    require(
        summary["realized"]["input_key_verified"] == rows
        and summary["realized"]["support_checks"] == rows
        and summary["realized"]["phases_per_row"] == {"1": rows},
        "missing original full-row identity/support proof",
    )
    require(
        summary["realized"].get("rows_dropped_envelope", 0) == 0
        and summary["realized"].get("rows_dropped_policy_support", 0) == 0,
        "unsupported source omissions",
    )
    require(
        summary.get("source_selection") is None and not summary.get("row_provenance"),
        "only legacy ordered source is supported",
    )
    require(
        summary["seed_effect"]
        == "permutes rows WITHIN each shard; changes no target value",
        "unsupported shuffle contract",
    )
    require(
        summary["corpus"]["config_sha256"] == record.facts["config_sha256"],
        "source/raw configuration differs",
    )
    require(
        record.facts["staircase_parsed"] == [{"depth": 9, "width": "all"}],
        "raw staircase differs",
    )
    require(
        summary["corpus"]["staircase_parsed"] == record.facts["staircase_parsed"]
        and summary["corpus"]["run_id"] == record.facts["run_id"]
        and summary["corpus"]["corpus_complete"] is True
        and summary["corpus"]["corpus_record_detail"]["shards_adopted"]
        == len(record.shards)
        and summary["corpus"]["corpus_record_detail"]["rows_claimed_by_inventory"]
        == record.rows_claimed,
        "source/raw inventory lineage differs",
    )


def shard_contract(attrs: dict[str, Any], summary: dict[str, Any], rows: int) -> None:
    expected = {
        "derive_state": "committed",
        "derive_run_finalized": True,
        "derive_corpus_config_sha256": summary["corpus"]["config_sha256"],
        "derive_scheme": "uniform-d9",
        "derive_temp": 0.0005,
        "derive_floor": 0.0,
        "derive_cp_slope": 0.006,
        "derive_cp_draw_width": 120.0,
        "derive_value_scheme": "search",
        "derive_value_source": "deepest_phase_covering",
        "derive_corpus_row_schema": 3,
        "derive_history_rep_fix": True,
        "history_rep_fix": True,
        "zero_history": False,
        "input_history_encoding": "lc0_root_legacy_meta",
        "policy_encoding": "lc0_1858",
        "policy_size": COMPACT_POLICY_SIZE,
        "positions": rows,
    }
    require(
        all(attrs.get(k) == v for k, v in expected.items()),
        "source shard identity differs",
    )
    require(
        not any(k.startswith(("policy_target_", "bt4_")) for k in attrs),
        "source shard already postprocessed",
    )


def copy_shard(source: Path, destination: Path) -> dict[str, str]:
    """Copy regular files; verify compressed non-policy bytes without decoding x."""
    nonpolicy: dict[str, str] = {}

    def copy_file(src: str, dst: str) -> str:
        path = Path(src)
        require(not path.is_symlink() and path.is_file(), "nonregular source storage")
        before = rank._file_identity(path)
        shutil.copy2(src, dst)
        if path.relative_to(source).parts[0] != "policy_target":
            digest = file_sha256(path)
            require(file_sha256(Path(dst)) == digest, "copied bytes differ")
            nonpolicy[str(path.relative_to(source))] = digest
        require(rank._file_identity(path) == before, "source changed during copy")
        return dst

    shutil.copytree(source, destination, copy_function=copy_file)
    return nonpolicy


def rewrite(args: argparse.Namespace) -> dict[str, Any]:
    temperature = derive.validate_temp(float(args.temperature))
    require(
        math.isfinite(args.minimum_free_gib) and args.minimum_free_gib >= 0,
        "invalid disk reserve",
    )
    raw_dir = Path(args.raw).resolve()
    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    writing = out.with_name(out.name + ".writing")
    require(
        all(
            out != root and root not in out.parents and out not in root.parents
            for root in (source, raw_dir)
        ),
        "output overlaps source",
    )
    require(
        not out.exists() and not writing.exists(), "output or partial already exists"
    )
    source_summary_path = source / derive.SUMMARY_NAME
    summary_bytes = source_summary_path.read_bytes()
    require(
        hashlib.sha256(summary_bytes).hexdigest()
        == args.expected_source_summary_sha256,
        "source summary pin differs",
    )
    summary = json.loads(summary_bytes)
    metadata = {
        p: file_sha256(p) for p in (raw_dir / "manifest.json", raw_dir / "summary.json")
    }
    metadata[source_summary_path] = args.expected_source_summary_sha256
    record = derive.read_corpus_record(raw_dir)
    source_contract(summary, record)
    manifest = derive.corpus.read_launch_manifest(raw_dir)
    require(
        all(
            manifest[k] == record.facts[k]
            for k in ("config_sha256", "row_schema", "staircase_parsed")
        ),
        "raw manifest/summary identity differs",
    )
    specs = summary["shards"]
    require(
        [p.name for p in sorted(source.glob("shard_*.zarr"))]
        == [s["path"] for s in specs],
        "source shard membership differs",
    )
    source_states = {
        source / s["path"]: rank._storage_identity(source / s["path"]) for s in specs
    }
    producer_hashes = {
        str(p): file_sha256(p)
        for p in (Path(__file__), Path(derive.__file__), Path(rank.__file__))
    }
    writing.mkdir(parents=True)
    start = time.monotonic()
    raw_rows = dropped = rows_written = 0
    rng = np.random.Generator(np.random.PCG64(int(summary["seed"])))
    pending: list[Observation] = []
    outputs: list[dict[str, Any]] = []
    raw_proofs = {}
    last_by_worker: dict[int, tuple[int, int]] = {}
    changed = 0
    max_mass_error = 0.0
    keys = hashlib.sha256()

    def guard() -> None:
        require(
            not (writing / "STOP").exists() and not (out.parent / "STOP").exists(),
            "STOP requested",
        )
        require(
            shutil.disk_usage(writing).free >= args.minimum_free_gib * 1024**3,
            "free disk reserve breached",
        )

    def flush() -> None:
        nonlocal rows_written, changed, max_mass_error
        guard()
        index = len(outputs)
        require(index < len(specs), "too many source rows")
        spec = specs[index]
        src = source / spec["path"]
        dst = writing / spec["path"]
        require(len(pending) == spec["rows"], "source shard row count differs")
        order = rng.permutation(len(pending))
        aligned = [pending[int(i)] for i in order]
        g: Any = zarr.open_group(str(src), mode="r")
        require(frozenset(g.array_keys()) == ARRAYS, "expected original17-array source")
        shard_contract(dict(g.attrs), summary, len(aligned))
        require(
            np.array_equal(g["game_id"][:], [r.game for r in aligned])
            and np.array_equal(g["ply_index"][:], [r.ply for r in aligned]),
            "raw/source shuffled identity differs",
        )
        old_policy = np.asarray(g["policy_target"][:])
        legal = np.asarray(g["legal_mask"][:])
        require(
            old_policy.dtype == np.float16 and legal.shape == old_policy.shape,
            "stored source policy schema differs",
        )
        new_policy = np.empty_like(old_policy)
        for i, obs in enumerate(aligned):
            mask = np.zeros(COMPACT_POLICY_SIZE, dtype=np.uint8)
            mask[obs.indices] = 1
            require(np.array_equal(mask, legal[i]), "stored/raw legal support differs")
            require(
                np.array_equal(target(obs, "q", 0.0005), old_policy[i]),
                "original q-policy reconstruction differs",
            )
            new_policy[i] = target(obs, args.score_space, temperature)
            keys.update(bytes.fromhex(obs.input_key))
        require(
            rank._storage_identity(src) == source_states[src],
            "source changed before copy",
        )
        copied = copy_shard(src, dst)
        dest: Any = zarr.open_group(str(dst), mode="a")
        dest["policy_target"][:] = new_policy
        require(
            np.array_equal(dest["policy_target"][:], new_policy),
            "policy readback differs",
        )
        recipe = {
            "score_space": args.score_space,
            "temperature": temperature,
            "temperature_units": "centipawns"
            if args.score_space == "effective-cp"
            else "q",
            "source_summary_sha256": args.expected_source_summary_sha256,
            "storage": "float64 softmax -> float32 -> float16",
            "mutated_arrays": ["policy_target"],
        }
        dest.attrs["policy_target_rewrite"] = recipe
        for rel, digest in copied.items():
            if rel != ".zattrs":
                require(file_sha256(dst / rel) == digest, "nonpolicy copy changed")
        changed += int(np.count_nonzero(np.any(new_policy != old_policy, axis=1)))
        max_mass_error = max(
            max_mass_error,
            float(np.max(np.abs(new_policy.astype(np.float64).sum(axis=1) - 1))),
        )
        rows_written += len(aligned)
        outputs.append(
            {
                "path": spec["path"],
                "rows": len(aligned),
                "source_storage_identity": source_states[src],
                "copied_file_hashes": copied,
                "policy_sha256": rank._sha_arrays(new_policy),
            }
        )
        pending.clear()

    try:
        for path in record.shards:
            if raw_rows >= summary["limit_requested"]:
                break
            guard()
            require(not path.is_symlink() and path.is_file(), "nonregular raw storage")
            before = rank._file_identity(path)
            read = 0
            for raw in derive.iter_corpus_rows(path):
                if raw_rows >= summary["limit_requested"]:
                    break
                raw_rows += 1
                read += 1
                derive._check_row_identity(raw, str(record.facts["config_sha256"]))
                require(
                    all(
                        type(raw[k]) is int and raw[k] >= 0
                        for k in ("worker_id", "game_id", "ply")
                    ),
                    "invalid raw physical identity",
                )
                worker = int(raw["worker_id"])
                key = (int(raw["game_id"]), int(raw["ply"]))
                require(
                    worker not in last_by_worker or key > last_by_worker[worker],
                    "duplicate/nonmonotone original worker identity",
                )
                last_by_worker[worker] = key
                if raw.get("result") is None:
                    dropped += 1
                    continue
                pending.append(observation(raw, str(record.facts["config_sha256"])))
                if len(pending) == summary["rows_per_shard"]:
                    flush()
            raw_proofs[str(path)] = {
                "identity": before,
                "sha256": file_sha256(path),
                "rows_consumed": read,
            }
            require(rank._file_identity(path) == before, "raw file changed during read")
        if pending:
            flush()
        require(
            raw_rows == summary["limit_requested"]
            and rows_written == summary["realized"]["rows_written"]
            and dropped == summary["realized"]["rows_dropped_no_result"]
            and len(outputs) == len(specs),
            "final source complement differs",
        )
        guard()
        for path, state in source_states.items():
            require(
                rank._storage_identity(path) == state,
                "source changed before publication",
            )
        for path, proof in raw_proofs.items():
            require(
                rank._file_identity(Path(path)) == proof["identity"],
                "raw source changed before publication",
            )
        for path, digest in metadata.items():
            require(file_sha256(path) == digest, "source metadata changed")
        result = {
            "schema": 1,
            "status": "COMPLETE",
            "kind": "sf_policy_score_rewrite",
            "score_space": args.score_space,
            "temperature": temperature,
            "source_dir": str(source),
            "raw_dir": str(raw_dir),
            "raw_limit": raw_rows,
            "rows": rows_written,
            "shards": len(outputs),
            "rows_dropped_no_result": dropped,
            "changed_rows": changed,
            "stored_mass_error_max": max_mass_error,
            "mutated_arrays": ["policy_target"],
            "nonpolicy_arrays_copied": 16,
            "producer_sha256": producer_hashes,
            "source_derive_summary_sha256": args.expected_source_summary_sha256,
            "metadata_sha256": {str(p): h for p, h in metadata.items()},
            "raw_files": raw_proofs,
            "raw_history_keys_sha256_in_emitted_order": keys.hexdigest(),
            "outputs": outputs,
            "elapsed_seconds": time.monotonic() - start,
            "history_lineage": "Inherited original input_key_verified/source x; no fresh history re-encoding.",
            "limitations": "Original historical control/provenance limitations unchanged; no valid-control promotion, inference, training or strength claim.",
        }
        rank._atomic_json(writing / SUMMARY, result)
        derived = dict(summary)
        derived["policy_target_postprocess"] = {
            k: v for k, v in result.items() if k != "outputs"
        }
        rank._atomic_json(writing / derive.SUMMARY_NAME, derived)
        guard()
        os.replace(writing, out)
        return result
    except BaseException as exc:
        rank._atomic_json(
            writing / "failed.json",
            {
                "error": repr(exc),
                "partial_output_preserved": True,
                "rows_written": rows_written,
                "raw_rows_read": raw_rows,
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["raw", "source", "out", "expected-source-summary-sha256"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--score-space", choices=["q", "effective-cp"], default="q")
    p.add_argument("--temperature", type=float, default=0.0005)
    p.add_argument("--minimum-free-gib", type=float, default=150.0)
    return p


def main(argv: list[str] | None = None) -> int:
    rewrite(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
