#!/usr/bin/env python3
"""Calibrate Tactical300 against already-recorded G10 d10/d12 observations.

This is a CPU-only retrospective diagnostic.  It joins a provenance-qualified
raw BT4 policy bank to the exact derived G10 rows described by the existing
``adapt_raw_bt4_sidecars`` manifest, reconstructs B100's T=0.5 one-node policy,
and asks a narrow question:

    when d9 Stockfish strongly prefers one move/set but BT4 prefers another,
    how often does the later recorded d10/d12 search prefer the BT4 choice?

The later G10 searches are narrowed and adaptively selected.  Missing moves are
therefore reported as unadjudicable rather than assigned invented scores.  The
output is calibration evidence, not ground truth, a training admission, or a new
teacher-label collection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import chess
import numpy as np
import zarr

from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE
from chess_anti_engine.moves.leela_index import compact_index_for_move
from chess_anti_engine.stockfish.wdl import SF_CP_CLAMP_CP
from scripts import adapt_raw_bt4_sidecars as adapter
from scripts import adaptive_sf_value as adaptive
from scripts import bt4_policy_mix as mix
from scripts import bt4_raw_corpus_sidecar as raw
from scripts import corpus_row_provenance as provenance
from scripts import derive_corpus_targets as derive
from scripts import sf_policy_rewrite as sf_rewrite
from scripts.bt4_policy_dump import file_sha256

SUMMARY = "tactical300_calibration.json"
SCHEMA = 1
THRESHOLDS_CP = (100.0, 200.0, 300.0, 500.0, 1000.0)
SELECTED_THRESHOLD_CP = 300.0
BLOCK_RATE = 0.05
MIN_ADJUDICABLE = 1000
BT4_TEMPERATURE = 0.5


def require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hex64(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _confidence_bucket(probability: float) -> str:
    require(math.isfinite(probability) and 0 <= probability <= 1, "invalid BT4 confidence")
    if probability < 0.25:
        return "lt_0.25"
    if probability < 0.50:
        return "0.25_to_0.50"
    if probability < 0.75:
        return "0.50_to_0.75"
    return "ge_0.75"


def _d9_scores(row: dict[str, Any], legal: set[str]) -> dict[str, float]:
    """Return the complete phase-zero d9 roster after G10 baseline validation."""
    adaptive.validate_baseline(row, legal)
    phase = row["phases"][0]
    blocks = [block for block in phase["per_depth"] if block["depth"] == 9]
    require(len(blocks) == 1, "ambiguous phase0 d9 block")
    lines = blocks[0]["lines"]
    require(len(lines) == len(legal), "phase0 d9 roster is not full width")
    values = {str(line[1]): float(line[2]) for line in lines}
    require(
        set(values) == legal and all(math.isfinite(v) for v in values.values()),
        "phase0 d9 legal roster differs",
    )
    return values


def _validate_score_domain(scores: dict[str, float]) -> None:
    values = np.asarray(list(scores.values()), dtype=np.float64)
    require(bool(np.isfinite(values).all()), "nonfinite SF score")
    mate = np.abs(values) > SF_CP_CLAMP_CP
    require(
        bool(np.isin(np.abs(values[mate]), sf_rewrite.MATE_SCORE_VALUES).all()),
        "SF score outside cp or shared mate domain",
    )


def _policy_view(
    board: chess.Board,
    policy: np.ndarray,
) -> tuple[set[str], float, np.ndarray]:
    """Return B100-T0.5 top set, its probability and the full tempered policy."""
    values = np.asarray(policy, dtype=np.float32)
    require(values.shape == (COMPACT_POLICY_SIZE,), "BT4 policy width differs")
    legal_mask = np.zeros(COMPACT_POLICY_SIZE, dtype=bool)
    move_to_index: dict[str, int] = {}
    for move in board.legal_moves:
        index = compact_index_for_move(board, move)
        require(index not in move_to_index.values(), "duplicate compact legal index")
        legal_mask[index] = True
        move_to_index[move.uci()] = index
    require(bool(legal_mask.any()), "position has no legal moves")
    tempered = mix._tempered_bt4_policy(
        values[None, :], legal_mask[None, :], temperature=BT4_TEMPERATURE
    )[0]
    top_probability = float(np.max(tempered[legal_mask]))
    top = {
        move
        for move, index in move_to_index.items()
        if float(tempered[index]) == top_probability
    }
    require(bool(top), "BT4 top set is empty")
    return top, top_probability, tempered


def _ordinary_d9_best(
    scores: dict[str, float],
) -> tuple[set[str], float | None]:
    values = np.asarray(list(scores.values()), dtype=np.float64)
    require(
        not bool(np.any(np.abs(values) > SF_CP_CLAMP_CP)),
        "ordinary d9 helper received mate-domain score",
    )
    best_score = float(values.max())
    best = {move for move, score in scores.items() if score == best_score}
    lower = [score for score in scores.values() if score < best_score]
    gap = best_score - max(lower) if lower else None
    return best, gap


def _set_presence(target: set[str], scores: dict[str, float]) -> tuple[bool, bool]:
    keys = set(scores)
    return target <= keys, bool(target & keys)


def _deeper_comparison(
    final_scores: dict[str, float] | None,
    d9_best: set[str],
    bt4_top: set[str],
) -> dict[str, Any]:
    if final_scores is None:
        return {
            "roster_case": "fallback_no_deeper_scores",
            "pairwise": None,
            "global_best": None,
        }
    d9_complete, d9_any = _set_presence(d9_best, final_scores)
    bt4_complete, bt4_any = _set_presence(bt4_top, final_scores)
    if d9_complete and bt4_complete:
        d9_score = max(final_scores[move] for move in d9_best)
        bt4_score = max(final_scores[move] for move in bt4_top)
        pairwise = (
            "bt4" if bt4_score > d9_score else "d9" if d9_score > bt4_score else "tie"
        )
        final_max = max(final_scores.values())
        global_set = {move for move, score in final_scores.items() if score == final_max}
        d9_global = bool(global_set & d9_best)
        bt4_global = bool(global_set & bt4_top)
        if d9_global and bt4_global:
            global_best = "tie_d9_bt4"
        elif d9_global:
            global_best = "d9"
        elif bt4_global:
            global_best = "bt4"
        else:
            global_best = "third"
        return {
            "roster_case": "both_complete",
            "pairwise": pairwise,
            "global_best": global_best,
        }
    if bt4_complete and not d9_complete:
        case = "bt4_complete_d9_partial" if d9_any else "bt4_only_scored"
    elif d9_complete and not bt4_complete:
        case = "d9_complete_bt4_partial" if bt4_any else "d9_only_scored"
    elif d9_any or bt4_any:
        case = "both_incomplete_or_partial"
    else:
        case = "neither_scored"
    return {"roster_case": case, "pairwise": None, "global_best": None}


def analyze_row(row: dict[str, Any], raw_bt4_policy: np.ndarray) -> dict[str, Any]:
    """Analyze one authenticated G10 row against its one-node BT4 policy."""
    board = chess.Board(str(row["fen"]))
    legal = {move.uci() for move in board.legal_moves}
    require(bool(legal), "terminal row is outside calibration scope")
    d9 = _d9_scores(row, legal)
    _validate_score_domain(d9)
    bt4_top, bt4_probability, _tempered = _policy_view(board, raw_bt4_policy)
    final_scores, final_depth, final_reason = adaptive.select(row, legal)
    if final_scores is not None:
        _validate_score_domain(final_scores)

    winning_mates = {move for move, score in d9.items() if score > SF_CP_CLAMP_CP}
    losing_mates = {move for move, score in d9.items() if score < -SF_CP_CLAMP_CP}
    base: dict[str, Any] = {
        "bt4_top": sorted(bt4_top),
        "bt4_top_probability": bt4_probability,
        "bt4_confidence_bucket": _confidence_bucket(bt4_probability),
        "final_depth": final_depth,
        "final_reason": final_reason,
    }
    if winning_mates:
        final_winning = (
            {
                move
                for move, score in final_scores.items()
                if score > SF_CP_CLAMP_CP
            }
            if final_scores is not None
            else set()
        )
        return {
            **base,
            "kind": "winning_mate",
            "d9_winning_mates": sorted(winning_mates),
            "bt4_agrees_with_d9_winning_mate": bool(bt4_top & winning_mates),
            "final_has_winning_mate": bool(final_winning),
            "final_preserves_d9_winning_mate": bool(final_winning & winning_mates),
            "bt4_top_is_final_winning_mate": bool(final_winning & bt4_top),
        }
    if losing_mates:
        return {
            **base,
            "kind": "losing_mate_present",
            "d9_losing_mates": len(losing_mates),
        }

    d9_best, gap = _ordinary_d9_best(d9)
    disagreement = d9_best.isdisjoint(bt4_top)
    comparison = (
        _deeper_comparison(final_scores, d9_best, bt4_top)
        if disagreement
        else {
            "roster_case": "bt4_agrees_d9",
            "pairwise": None,
            "global_best": None,
        }
    )
    return {
        **base,
        "kind": "ordinary",
        "d9_best": sorted(d9_best),
        "d9_gap_cp": gap,
        "disagreement": disagreement,
        **comparison,
    }


def _empty_cell() -> dict[str, int]:
    return {
        "eligible_gap": 0,
        "disagreements": 0,
        "adjudicable": 0,
        "bt4_pairwise_wins": 0,
        "d9_pairwise_wins": 0,
        "pairwise_ties": 0,
        "global_bt4": 0,
        "global_d9": 0,
        "global_tie_d9_bt4": 0,
        "global_third": 0,
        "bt4_only_scored": 0,
        "d9_only_scored": 0,
        "bt4_complete_d9_partial": 0,
        "d9_complete_bt4_partial": 0,
        "both_incomplete_or_partial": 0,
        "neither_scored": 0,
        "fallback_no_deeper_scores": 0,
    }


def _update_cell(
    cell: dict[str, int], result: dict[str, Any], *, disagreement: bool
) -> None:
    cell["eligible_gap"] += 1
    if not disagreement:
        return
    cell["disagreements"] += 1
    roster = str(result["roster_case"])
    if roster == "both_complete":
        cell["adjudicable"] += 1
        pairwise = str(result["pairwise"])
        cell[
            {"bt4": "bt4_pairwise_wins", "d9": "d9_pairwise_wins", "tie": "pairwise_ties"}[
                pairwise
            ]
        ] += 1
        global_best = str(result["global_best"])
        cell[
            {
                "bt4": "global_bt4",
                "d9": "global_d9",
                "tie_d9_bt4": "global_tie_d9_bt4",
                "third": "global_third",
            }[global_best]
        ] += 1
    else:
        require(roster in cell, "unknown roster calibration class")
        cell[roster] += 1


def _new_aggregate() -> dict[str, Any]:
    return {
        "rows": 0,
        "ordinary_rows": 0,
        "winning_mate_rows": 0,
        "losing_mate_rows": 0,
        "mate": {
            "bt4_agrees_d9_winning_mate": 0,
            "final_has_winning_mate": 0,
            "final_preserves_d9_winning_mate": 0,
            "bt4_top_is_final_winning_mate": 0,
            "fallback_no_deeper_scores": 0,
        },
        "thresholds": {str(int(value)): _empty_cell() for value in THRESHOLDS_CP},
        "threshold300_by_confidence": {
            name: _empty_cell()
            for name in ("lt_0.25", "0.25_to_0.50", "0.50_to_0.75", "ge_0.75")
        },
        "threshold300_by_depth": {
            "10": _empty_cell(),
            "12": _empty_cell(),
            "9_fallback": _empty_cell(),
        },
    }


def aggregate_row(aggregate: dict[str, Any], result: dict[str, Any]) -> None:
    aggregate["rows"] += 1
    kind = result["kind"]
    if kind == "winning_mate":
        aggregate["winning_mate_rows"] += 1
        mate = aggregate["mate"]
        for key in (
            "bt4_agrees_with_d9_winning_mate",
            "final_has_winning_mate",
            "final_preserves_d9_winning_mate",
            "bt4_top_is_final_winning_mate",
        ):
            target = (
                "bt4_agrees_d9_winning_mate"
                if key == "bt4_agrees_with_d9_winning_mate"
                else key
            )
            mate[target] += int(bool(result[key]))
        mate["fallback_no_deeper_scores"] += int(result["final_depth"] == 9)
        return
    if kind == "losing_mate_present":
        aggregate["losing_mate_rows"] += 1
        return
    require(kind == "ordinary", "unknown calibration row kind")
    aggregate["ordinary_rows"] += 1
    gap = result["d9_gap_cp"]
    if gap is None:
        return
    disagreement = bool(result["disagreement"])
    for threshold in THRESHOLDS_CP:
        if float(gap) > threshold:
            _update_cell(
                aggregate["thresholds"][str(int(threshold))],
                result,
                disagreement=disagreement,
            )
    if float(gap) > SELECTED_THRESHOLD_CP:
        bucket = str(result["bt4_confidence_bucket"])
        _update_cell(
            aggregate["threshold300_by_confidence"][bucket],
            result,
            disagreement=disagreement,
        )
        depth_key = (
            str(result["final_depth"])
            if result["final_depth"] in (10, 12)
            else "9_fallback"
        )
        _update_cell(
            aggregate["threshold300_by_depth"][depth_key],
            result,
            disagreement=disagreement,
        )


def decision_from_aggregate(
    aggregate: dict[str, Any], *, full_source_coverage: bool
) -> dict[str, Any]:
    selected = aggregate["thresholds"][str(int(SELECTED_THRESHOLD_CP))]
    denominator = int(selected["adjudicable"])
    numerator = int(selected["bt4_pairwise_wins"])
    rate = numerator / denominator if denominator else None
    if not full_source_coverage:
        verdict = "PARTIAL_NO_DECISION"
    elif denominator < MIN_ADJUDICABLE:
        verdict = "INSUFFICIENT_ADJUDICABILITY"
    elif rate is not None and rate >= BLOCK_RATE:
        verdict = "BLOCK_D9_ONLY_REQUIRE_DEPTH_STABILITY"
    else:
        verdict = "NO_5PCT_BLOCK_CALIBRATION_STILL_REQUIRED_FOR_ADMISSION"
    return {
        "selected_threshold_cp": SELECTED_THRESHOLD_CP,
        "bt4_pairwise_wins": numerator,
        "pairwise_adjudicable": denominator,
        "bt4_pairwise_win_rate": rate,
        "block_rate": BLOCK_RATE,
        "minimum_adjudicable": MIN_ADJUDICABLE,
        "verdict": verdict,
        "training_admission": False,
    }


def _raw_rows(path: Path, offsets: set[int]) -> dict[int, dict[str, Any]]:
    require(bool(offsets) and min(offsets) >= 0, "invalid requested raw offsets")
    result: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(derive.iter_corpus_rows(path)):
        if index in offsets:
            result[index] = row
            if len(result) == len(offsets):
                break
    require(set(result) == offsets, "raw physical rows missing from pinned source shard")
    return result


def calibrate(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    out: Path,
    start_shard: int = 0,
    max_shards: int | None = None,
    max_raw_rows: int = 100000,
    max_index_bytes: int = 8 * 1024**3,
) -> dict[str, Any]:
    """Run the provenance-qualified retrospective diagnostic without new inference."""
    require(_hex64(expected_manifest_sha256), "adapter manifest SHA256 required")
    require(
        start_shard >= 0 and (max_shards is None or max_shards > 0),
        "invalid shard slice",
    )
    manifest_path = manifest_path.resolve()
    manifest_pin = {"path": str(manifest_path), "sha256": expected_manifest_sha256}
    manifest = json.loads(adapter.pin(manifest_pin).read_text())
    require(
        set(manifest)
        in (
            {"schema", "derived_summary", "teacher", "sources"},
            {"schema", "derived_summary", "teacher", "sources", "wdl"},
        )
        and manifest["schema"] == 1,
        "adapter manifest fields/schema differ",
    )
    summary_path = adapter.pin(manifest["derived_summary"])
    require(summary_path.name == derive.SUMMARY_NAME, "expected derived corpus summary")
    source = summary_path.parent
    summary = json.loads(summary_path.read_text())
    require(
        summary.get("row_provenance", {}).get("path_in_shard") == provenance.FILENAME,
        "derivation has no row provenance",
    )
    paths = sorted(source.glob("shard_*.zarr"))
    written = {entry["path"]: entry for entry in summary["shards"]}
    require(
        bool(paths) and set(written) == {path.name for path in paths},
        "derived shard inventory differs from summary",
    )
    stop = (
        len(paths)
        if max_shards is None
        else min(len(paths), start_shard + max_shards)
    )
    selected = paths[start_shard:stop]
    require(bool(selected), "selected shard slice is empty")
    full_source_coverage = start_shard == 0 and len(selected) == len(paths)

    inputs = adapter.RawInputs(manifest, max_raw_rows, max_index_bytes)
    out = out.resolve()
    writing = out.with_name(out.name + ".writing")
    protected = [source, manifest_path, *[Path(value) for value in inputs.sources]]
    protected += [spec.out_dir for spec, _receipts in inputs.sources.values()]
    require(
        all(
            out != path and out not in path.parents and path not in out.parents
            for path in protected
        ),
        "output overlaps inputs",
    )
    require(not out.exists() and not writing.exists(), "new output required; no adoption")
    writing.mkdir(parents=True)
    inputs.cache_dir = writing / "._raw_identity_cache"
    inputs.cache_dir.mkdir()

    aggregate = _new_aggregate()
    identity = hashlib.sha256()
    derived_stable: dict[Path, str] = {}
    rows_analyzed = 0
    started = time.monotonic()
    shard_receipts: list[dict[str, Any]] = []
    try:
        for path in selected:
            derived_stable[path] = adapter.storage_identity(path)
            group: Any = zarr.open_group(str(path), mode="r")
            x = np.asarray(group["x"][:])
            rows = len(x)
            require(
                rows == written[path.name]["rows"] and rows > 0,
                "derived row count differs",
            )
            stamp = dict(group.attrs).get("derive_row_provenance")
            if not isinstance(stamp, dict):
                raise ValueError("provenance attribute is not a mapping")
            require(
                stamp == written[path.name].get("row_provenance"),
                "provenance summary/attribute pin mismatch",
            )
            require(
                stamp["schema"] == provenance.SCHEMA
                and stamp["rows"] == rows
                and stamp["record_bytes"] == provenance.RECORD_DTYPE.itemsize
                and stamp["path"] == provenance.FILENAME,
                "provenance stamp format differs",
            )
            provenance_path = path / provenance.FILENAME
            require(
                file_sha256(provenance_path) == stamp["sha256"],
                "corrupted row provenance",
            )
            refs = provenance.read(provenance_path, rows=rows)
            game_ids = np.asarray(group["game_id"][:])
            ply_indices = np.asarray(group["ply_index"][:])
            derived_legal = np.asarray(group["legal_mask"][:]) != 0
            grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = {}
            seen: set[tuple[str, str, int]] = set()
            for index, ref in enumerate(refs):
                source_identity = (
                    ref["source_namespace"],
                    ref["source_shard"],
                    int(ref["source_row"]),
                )
                require(
                    source_identity not in seen,
                    "duplicate source-qualified derived row",
                )
                seen.add(source_identity)
                grouped.setdefault(
                    (str(ref["source_dir"]), str(ref["source_shard"])), []
                ).append((index, ref))

            for requests in grouped.values():
                first_ref = requests[0][1]
                raw_group, records = inputs.get(first_ref)
                offsets = {int(ref["source_row"]) for _index, ref in requests}
                raw_path = Path(str(first_ref["source_dir"])) / str(
                    first_ref["source_shard"]
                )
                source_rows = _raw_rows(raw_path, offsets)
                raw_offsets = np.asarray(
                    [int(ref["source_row"]) for _index, ref in requests],
                    dtype=np.int64,
                )
                policies = np.asarray(
                    raw_group[raw.POLICY_FIELD].oindex[raw_offsets, :]
                )
                require(
                    policies.shape == (len(requests), COMPACT_POLICY_SIZE),
                    "raw BT4 policy shape differs",
                )

                for request_index, (derived_index, ref) in enumerate(requests):
                    offset = int(ref["source_row"])
                    require(
                        0 <= offset < len(records),
                        "physical source row outside closed shard",
                    )
                    record = records[offset]
                    require(
                        ref["input_key"] == record["input_key"].tobytes().hex(),
                        "raw full-history key mismatch",
                    )
                    require(
                        ref["stored_input_key"]
                        == record["stored_input_key"].tobytes().hex()
                        == adapter.corpus.input_tensor_key(x[derived_index]),
                        "stored quantized full-history key mismatch",
                    )
                    require(
                        all(
                            int(ref[field]) == int(record[field])
                            for field in ("worker_id", "game_id", "ply")
                        ),
                        "raw physical row/game/ply/worker identity mismatch",
                    )
                    require(
                        int(game_ids[derived_index]) == int(ref["game_id"])
                        and int(ply_indices[derived_index]) == int(ref["ply"]),
                        "derived game/ply alignment mismatch",
                    )
                    raw_row = source_rows[offset]
                    derive._check_row_identity(
                        raw_row, str(ref["source_config_sha256"])
                    )
                    require(
                        int(raw_row["worker_id"]) == int(ref["worker_id"])
                        and int(raw_row["game_id"]) == int(ref["game_id"])
                        and int(raw_row["ply"]) == int(ref["ply"])
                        and raw_row["input_key"] == ref["input_key"],
                        "raw row identity differs from provenance",
                    )

                    board = chess.Board(str(raw_row["fen"]))
                    expected_legal = np.zeros(COMPACT_POLICY_SIZE, dtype=bool)
                    for move in board.legal_moves:
                        expected_legal[compact_index_for_move(board, move)] = True
                    require(
                        np.array_equal(
                            expected_legal, derived_legal[derived_index]
                        ),
                        "raw board legal support differs from derived row",
                    )
                    result = analyze_row(raw_row, policies[request_index])
                    aggregate_row(aggregate, result)
                    identity.update(
                        json.dumps(
                            [
                                str(ref["source_namespace"]),
                                str(ref["source_shard"]),
                                offset,
                                str(ref["input_key"]),
                                sorted(result.get("bt4_top", [])),
                            ],
                            separators=(",", ":"),
                        ).encode()
                    )
                    rows_analyzed += 1

            require(
                adapter.storage_identity(path) == derived_stable[path],
                "derived source changed during calibration",
            )
            require(
                file_sha256(provenance_path) == stamp["sha256"],
                "row provenance changed during calibration",
            )
            shard_receipts.append(
                {
                    "path": path.name,
                    "rows": rows,
                    "row_provenance_sha256": stamp["sha256"],
                    "source_storage_identity": derived_stable[path],
                }
            )

        require(rows_analyzed == aggregate["rows"], "aggregate row count differs")
        require(
            all(
                adapter.storage_identity(path) == stamp
                for path, stamp in (inputs.stable | derived_stable).items()
            ),
            "verified input storage changed before publication",
        )
        for item in [manifest_pin, manifest["derived_summary"], *inputs.pins]:
            adapter.pin(item)

        decision = decision_from_aggregate(
            aggregate, full_source_coverage=full_source_coverage
        )
        final = {
            "schema": SCHEMA,
            "status": "COMPLETE_DIAGNOSTIC_NOT_TRAINING_ADMISSION",
            "kind": "tactical300-bt4-vs-deeper-sf-calibration",
            "adapter_manifest": manifest_pin,
            "derived_summary": manifest["derived_summary"],
            "teacher": {
                "onnx": inputs.teacher["onnx"],
                "policy_output": inputs.teacher["policy_output"],
                "providers": inputs.teacher["providers"],
                "bt4_temperature": BT4_TEMPERATURE,
                "teacher_evaluations_per_position": 1,
                "search_nodes": 0,
                "new_teacher_inference": 0,
            },
            "thresholds_cp": list(THRESHOLDS_CP),
            "selected_threshold_cp": SELECTED_THRESHOLD_CP,
            "start_shard": start_shard,
            "selected_shards": len(selected),
            "source_shards": len(paths),
            "full_source_coverage": full_source_coverage,
            "rows": rows_analyzed,
            "identity_sha256": identity.hexdigest(),
            "aggregate": aggregate,
            "decision": decision,
            "verified_raw_shards": inputs.verified,
            "shards": shard_receipts,
            "producer_sha256": {
                str(Path(__file__).resolve()): file_sha256(Path(__file__).resolve()),
                str(Path(adaptive.__file__).resolve()): file_sha256(
                    Path(adaptive.__file__).resolve()
                ),
                str(Path(adapter.__file__).resolve()): file_sha256(
                    Path(adapter.__file__).resolve()
                ),
                str(Path(mix.__file__).resolve()): file_sha256(
                    Path(mix.__file__).resolve()
                ),
            },
            "elapsed_seconds": time.monotonic() - started,
            "limits": [
                "Later G10 searches are narrowed and d12 is selected by the d10 gate; they are not optimal-play truth.",
                "Historical searches may share transposition-table state.",
                "A move absent from a later narrowed roster receives no invented score and cannot enter the pairwise denominator.",
                "The 5% rule is an experiment-allocation threshold, not an accuracy confidence bound.",
                "Even a non-blocking result does not admit Tactical300 materialization or training by itself.",
            ],
        }
        inputs.cache.clear()
        assert inputs.cache_dir is not None
        shutil.rmtree(inputs.cache_dir)
        mix._atomic_json(writing / SUMMARY, final)
        os.replace(writing, out)
        return final
    except BaseException as exc:
        inputs.cache.clear()
        if writing.exists():
            mix._atomic_json(
                writing / "failed.json",
                {
                    "error": repr(exc),
                    "rows_analyzed": rows_analyzed,
                    "partial_output_preserved": True,
                },
            )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-raw-rows", type=int, default=100000)
    parser.add_argument("--max-index-bytes", type=int, default=8 * 1024**3)
    args = parser.parse_args(argv)
    calibrate(
        args.manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
        out=args.out,
        start_shard=args.start_shard,
        max_shards=args.max_shards,
        max_raw_rows=args.max_raw_rows,
        max_index_bytes=args.max_index_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
