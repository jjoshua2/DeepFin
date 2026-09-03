#!/usr/bin/env python3
"""Test whether node-budget context improves a search continuation controller.

The input is the accumulating-tree bank produced by
``backtest_chunk_trajectory.py``.  Each adjacent chunk pair is one decision:
buy the next fixed-size node tranche or stop.  The signed label is
``regret_before - regret_after``; regressions therefore remain negative rather
than disappearing behind a positive-only clamp.

M0 is a deterministic ridge model over search-settledness features.  M1 adds
the current and hard node horizons plus interactions.  Every reported
prediction is out of sample twice: the target horizon is absent from training,
and the target game is in an outer held-out group.  Ridge strength is selected
inside the remaining games with grouped inner CV.

This tool deliberately tests fixed NODE horizons only.  Trajectory banks do
not contain real game clocks, soft budgets, or time-forfeit observations, so a
positive result can justify collecting a clock bank but cannot justify a
clock-conditioned production controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.moves import index_to_move

_SCHEMA = "deepfin.chunk_trajectory.v2"
_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
_COMPLEXITY_VISIT_GAP = 0.25
_COMPLEXITY_STABLE_CHUNKS = 2
_MIN_DECISION_GRADE_BOOTSTRAP_SAMPLES = 1000
_PRODUCTION_WDL_FILES = 875
_PRODUCTION_DTZ_FILES = 510
_PRODUCTION_TB_COMPONENTS = ((365, 365), (510, 145))
_PRODUCTION_GSS_HALVING_REV = 3
_ACTIVE_PARAMETER_KEYS = {
    "walker_puct": {
        "c_puct", "cpuct_factor", "cpuct_base", "fpu_reduction",
        "vloss_weight", "walker_gather", "policy_temp",
    },
    "gumbel": {
        "c_scale", "c_visit", "c_visit_root", "c_scale_root",
        "q_visit_exp_root", "topk", "policy_temp", "halving_div",
        "root_noise_scale", "vloss_weight", "minibatch_size",
    },
}


@dataclass(frozen=True)
class Transition:
    key: str
    group_id: str
    horizon: int
    cost: int
    gain: float
    regret_before: float
    regret_after: float
    complexity_continue: bool
    state: dict[str, float]


@dataclass(frozen=True)
class PolicyResult:
    selected: int
    spend: int
    signed_gain: float
    capture_over_random: float | None
    regret_mean: float
    regret_p95: float
    regret_p99: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, rendered: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp_path.open("x") as fh:
            fh.write(rendered)
            fh.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _require_safe_output_path(
    input_path: Path, meta_path: Path, output_path: Path | None,
) -> None:
    if output_path is None:
        return
    output = output_path.expanduser().resolve()
    protected = {input_path.expanduser().resolve(), meta_path.expanduser().resolve()}
    if output in protected:
        raise ValueError("--out must not overwrite the input bank or its manifest")


def _require_bootstrap_resolution(samples: int, methodology_smoke: bool) -> None:
    if not methodology_smoke and samples < _MIN_DECISION_GRADE_BOOTSTRAP_SAMPLES:
        raise ValueError(
            "decision-grade analysis requires at least "
            f"{_MIN_DECISION_GRADE_BOOTSTRAP_SAMPLES} bootstrap samples; "
            "use --methodology-smoke only for a smaller pipeline check"
        )


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _strict_finite(row: dict[str, Any], name: str) -> float:
    value = row.get(name)
    if value is None or isinstance(value, bool):
        raise ValueError(f"{row.get('key')}: {name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{row.get('key')}: {name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{row.get('key')}: {name} must be a finite number")
    return number


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower())
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _artifact_provenance_complete(artifact: Any) -> bool:
    return bool(
        isinstance(artifact, dict)
        and isinstance(artifact.get("path"), str)
        and artifact.get("path")
        and _positive_int(artifact.get("size"))
        and _nonnegative_int(artifact.get("mtime_ns"))
        and _valid_sha256(artifact.get("sha256"))
    )


def _compatible_native_extension(artifact: Any) -> bool:
    if not isinstance(artifact, dict):
        return False
    abi = artifact.get("abi_version")
    required = artifact.get("required_abi_version")
    halving_rev = artifact.get("gss_halving_rev")
    if (
        not isinstance(abi, int) or isinstance(abi, bool) or abi <= 0
        or not isinstance(required, int) or isinstance(required, bool) or required <= 0
    ):
        return False
    return (
        _artifact_provenance_complete(artifact)
        and str(artifact.get("path", "")).endswith((".so", ".pyd"))
        and abi >= required
        and halving_rev == _PRODUCTION_GSS_HALVING_REV
    )


def _sum_manifest_ints(rows: Any, field: str) -> int:
    if (
        not isinstance(rows, list)
        or any(not isinstance(row, dict) for row in rows)
        or any(not _nonnegative_int(row.get(field)) for row in rows)
    ):
        return -1
    return sum(int(row[field]) for row in rows)


def _tablebase_component_counts(rows: Any) -> tuple[tuple[int, int], ...]:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return ()
    try:
        return tuple(sorted(
            (int(row["rtbw_count"]), int(row["rtbz_count"])) for row in rows
        ))
    except (KeyError, TypeError, ValueError):
        return ()


def _update_stability(
    last_best: int,
    stable_chunks: int,
    *,
    emitted_action: int,
    visit_gap: float,
    action_count: int,
) -> tuple[int, int]:
    """Mirror the visit-gated streak update in ``SearchWorker._abort_ready``."""
    if visit_gap <= 0.0 and action_count != 1:
        return last_best, 0
    if emitted_action == last_best:
        return last_best, stable_chunks + 1
    return emitted_action, 0


def _complexity_continue(
    *, stable_chunks: int, visit_gap: float, action_count: int,
) -> bool:
    """Mirror the clock-free stability branch, including forced moves."""
    return not (
        stable_chunks >= _COMPLEXITY_STABLE_CHUNKS
        and (action_count == 1 or visit_gap >= _COMPLEXITY_VISIT_GAP)
    )


def _emitted_visit_gap(row: dict[str, Any]) -> float | None:
    actions = row.get("root_actions")
    shares = row.get("root_visit_shares")
    emitted = row.get("emitted_action")
    if not isinstance(actions, list) or not isinstance(shares, list) or emitted is None:
        return None
    if len(actions) != len(shares) or not actions:
        raise ValueError(f"{row.get('key')}: malformed root action/share arrays")
    try:
        index = [int(action) for action in actions].index(int(emitted))
    except ValueError as exc:
        raise ValueError(f"{row.get('key')}: emitted action is absent from root actions") from exc
    values = np.asarray(shares, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{row.get('key')}: invalid root visit shares")
    others = np.delete(values, index)
    return float(values[index] - (others.max() if others.size else 0.0))


def _source_group(row: dict[str, Any]) -> str | None:
    source_dir = row.get("source_dir")
    shard = row.get("shard")
    game_id = row.get("game_id")
    if (
        not isinstance(source_dir, str) or not source_dir
        or not isinstance(shard, str) or not shard
        or not _nonnegative_int(game_id)
    ):
        return None
    return "\0".join((source_dir, shard, str(game_id)))


def _validate_decision_grade_row(row: dict[str, Any], line_number: int) -> None:
    key = row.get("key")
    if not isinstance(key, str) or not key:
        raise ValueError(f"line {line_number}: position key is missing")
    fen = row.get("fen")
    if not isinstance(fen, str) or not fen:
        raise ValueError(f"{key}: root FEN is missing")
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"{key}: root FEN is invalid") from exc
    group = _source_group(row)
    if group is None or row.get("group_id") != group:
        raise ValueError(f"{key}: group_id is not (source_dir, shard, game_id)")
    if not _positive_int(row.get("chunk")) or not _positive_int(row.get("nodes")):
        raise ValueError(f"{key}: chunk and nodes must be positive integers")
    for name in (
        "elapsed_ms", "regret_cp", "regret_score", "regret_vs_final_cp",
        "visit_gap", "visit_entropy", "root_q",
    ):
        _strict_finite(row, name)
    if float(row["elapsed_ms"]) < 0.0:
        raise ValueError(f"{key}: elapsed_ms must be non-negative")
    for name in ("piece_count", "legal_move_count"):
        if not _positive_int(row.get(name)):
            raise ValueError(f"{key}: {name} must be a positive integer")
    if (
        row["piece_count"] != chess.popcount(board.occupied)
        or row["legal_move_count"] != board.legal_moves.count()
    ):
        raise ValueError(f"{key}: board morphology disagrees with root FEN")
    phase = row.get("phase")
    if not isinstance(phase, int) or isinstance(phase, bool) or phase not in (0, 1, 2):
        raise ValueError(f"{key}: phase must be 0, 1, or 2")
    if not _nonnegative_int(row.get("stable_chunks")):
        raise ValueError(f"{key}: stable_chunks must be a non-negative integer")
    if not isinstance(row.get("bestmove_flip"), bool):
        raise ValueError(f"{key}: bestmove_flip must be boolean")
    if not isinstance(row.get("changes_to_final"), bool):
        raise ValueError(f"{key}: changes_to_final must be boolean")
    if not isinstance(row.get("complexity_predicate_continue"), bool):
        raise ValueError(f"{key}: complexity predicate must be boolean")
    if not isinstance(row.get("emitted_action"), int) or isinstance(
        row.get("emitted_action"), bool,
    ):
        raise ValueError(f"{key}: emitted_action must be an integer")
    if not isinstance(row.get("uci"), str) or not row.get("uci"):
        raise ValueError(f"{key}: emitted UCI move is missing")
    q_gap = row.get("q_gap")
    if q_gap is not None:
        _strict_finite(row, "q_gap")
    for name in ("q_drift", "visit_churn"):
        value = row.get(name)
        if value is None and row["chunk"] == 1:
            continue
        _strict_finite(row, name)
    actions = row.get("root_actions")
    visits = row.get("root_visits")
    shares = row.get("root_visit_shares")
    child_q = row.get("root_child_q")
    if (
        not isinstance(actions, list) or not actions
        or any(not isinstance(action, int) or isinstance(action, bool) for action in actions)
        or len(set(actions)) != len(actions)
        or not isinstance(visits, list) or len(visits) != len(actions)
        or any(not _nonnegative_int(visit) for visit in visits)
        or not isinstance(shares, list) or len(shares) != len(actions)
        or not isinstance(child_q, list) or len(child_q) != len(actions)
    ):
        raise ValueError(f"{key}: malformed root action/visit/share/Q arrays")
    values = np.asarray(shares, dtype=np.float64)
    q_values = np.asarray(child_q, dtype=np.float64)
    visit_values = np.asarray(visits, dtype=np.float64)
    visit_total = float(visit_values.sum())
    expected_shares = (
        visit_values / visit_total if visit_total > 0.0 else np.zeros_like(visit_values)
    )
    if (
        not np.isfinite(values).all() or (values < 0.0).any()
        or (visit_total <= 0.0 and len(actions) != 1)
        or not np.allclose(values, expected_shares, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError(f"{key}: root visit shares disagree with integer visits")
    if not np.isfinite(q_values).all():
        raise ValueError(f"{key}: root child Q values must be finite")
    try:
        legal_root_actions = {
            action
            for action in actions
            if index_to_move(action, board) in board.legal_moves
        }
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError(f"{key}: root action cannot be decoded") from exc
    if len(legal_root_actions) != len(actions):
        raise ValueError(f"{key}: root action is illegal for the recorded FEN")
    pv_actions = row.get("pv_actions")
    pv_uci = row.get("pv_uci")
    if (
        not isinstance(pv_actions, list) or not pv_actions
        or any(
            not isinstance(action, int) or isinstance(action, bool)
            for action in pv_actions
        )
        or not isinstance(pv_uci, list) or len(pv_uci) != len(pv_actions)
        or any(not isinstance(move, str) or not move for move in pv_uci)
        or pv_actions[0] != row.get("emitted_action")
        or pv_uci[0] != row.get("uci")
    ):
        raise ValueError(f"{key}: malformed emitted-move PV provenance")
    pv_board = board.copy(stack=False)
    for action, uci in zip(pv_actions, pv_uci):
        try:
            move = index_to_move(action, pv_board)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(f"{key}: PV action cannot be decoded") from exc
        if move not in pv_board.legal_moves or move.uci() != uci:
            raise ValueError(f"{key}: PV action/UCI is illegal for the recorded FEN")
        pv_board.push(move)


def _require_manifest(
    input_path: Path, meta_path: Path, methodology_smoke: bool,
) -> tuple[dict[str, Any], bool]:
    if not meta_path.is_file():
        if methodology_smoke:
            return {}, False
        raise ValueError(
            f"decision-grade analysis requires {meta_path}; pass --methodology-smoke "
            "only to exercise the estimator on a legacy bank"
        )
    manifest = json.loads(meta_path.read_text())
    failures: list[str] = []
    if manifest.get("schema") != _SCHEMA:
        failures.append(f"schema={manifest.get('schema')!r}")
    if manifest.get("complete") is not True:
        failures.append("complete is not true")
    if manifest.get("analysis_scope") != "fixed_node_horizons_only":
        failures.append(f"analysis_scope={manifest.get('analysis_scope')!r}")
    if manifest.get("clock_conditioning_available") is not False:
        failures.append("clock scope is ambiguous")
    output = manifest.get("output")
    if (
        not isinstance(output, dict)
        or output.get("sha256") != _sha256(input_path)
        or output.get("size") != input_path.stat().st_size
    ):
        failures.append("input digest does not match manifest")
    producer_sha = manifest.get("producer_git_sha")
    if (
        not isinstance(producer_sha, str)
        or len(producer_sha) != 40
        or any(char not in "0123456789abcdef" for char in producer_sha.lower())
    ):
        failures.append("producer_git_sha is not a full commit id")
    if manifest.get("producer_git_dirty") is not False:
        failures.append("producer checkout was dirty")
    failures.extend(
        f"{name} artifact provenance is incomplete"
        for name in ("producer_script", "checkpoint", "audit_set", "matched_rows")
        if not _artifact_provenance_complete(manifest.get(name))
    )
    checkpoint_params = manifest.get("checkpoint_params")
    if "checkpoint_params" not in manifest or (
        checkpoint_params is not None
        and not _artifact_provenance_complete(checkpoint_params)
    ):
        failures.append("checkpoint architecture provenance is incomplete")
    mcts_extension = manifest.get("mcts_extension")
    if not _compatible_native_extension(mcts_extension):
        failures.append("native MCTS extension provenance is incomplete")
    if manifest.get("game_group_kind") != "source_dir:shard:game_id":
        failures.append(f"game_group_kind={manifest.get('game_group_kind')!r}")
    if manifest.get("root_position_history") != "fen_only_from_audit_fen":
        failures.append(f"root_position_history={manifest.get('root_position_history')!r}")
    if manifest.get("complexity_predicate") != {
        "kind": "clock_free_visit_gap_and_stability",
        "minimum_stable_chunks": _COMPLEXITY_STABLE_CHUNKS,
        "minimum_visit_gap": _COMPLEXITY_VISIT_GAP,
        "single_legal_move_is_decided": True,
    }:
        failures.append("clock-free complexity predicate provenance is incomplete")
    syzygy = manifest.get("syzygy")
    directories = syzygy.get("directories") if isinstance(syzygy, dict) else None
    directory_wdl_count = _sum_manifest_ints(directories, "rtbw_count")
    directory_dtz_count = _sum_manifest_ints(directories, "rtbz_count")
    syzygy_paths = (
        tuple(
            str(Path(value.strip()).expanduser().resolve())
            for value in syzygy["path"].split(os.pathsep)
        )
        if isinstance(syzygy, dict) and isinstance(syzygy.get("path"), str)
        else ()
    )
    directory_paths = (
        tuple(str(row.get("path", "")) for row in directories)
        if isinstance(directories, list) and all(isinstance(row, dict) for row in directories)
        else ()
    )
    if (
        not isinstance(syzygy, dict)
        or not isinstance(syzygy.get("path"), str)
        or not syzygy.get("path")
        or not isinstance(directories, list)
        or len(directories) < 2
        or syzygy_paths != directory_paths
        or _tablebase_component_counts(directories) != _PRODUCTION_TB_COMPONENTS
        or not isinstance(syzygy.get("rtbw_count"), int)
        or int(syzygy.get("rtbw_count", 0)) < _PRODUCTION_WDL_FILES
        or syzygy.get("rtbw_count") != directory_wdl_count
        or not isinstance(syzygy.get("rtbz_count"), int)
        or int(syzygy.get("rtbz_count", 0)) < _PRODUCTION_DTZ_FILES
        or syzygy.get("rtbz_count") != directory_dtz_count
        or any(
            not isinstance(directory, dict)
            or not isinstance(directory.get("path"), str)
            or not _positive_int(directory.get("rtbw_count"))
            or not _positive_int(directory.get("rtbz_count"))
            or not _positive_int(directory.get("total_bytes"))
            or not _valid_sha256(directory.get("inventory_sha256"))
            for directory in (directories or [])
        )
    ):
        failures.append("production Syzygy provenance is incomplete")
    requested = manifest.get("requested_search")
    realized = manifest.get("realized_search")
    if not isinstance(requested, dict) or not isinstance(realized, dict):
        failures.append("requested/realized search provenance is missing")
    else:
        active_path = requested.get("active_path")
        active = requested.get("active_parameters")
        expected_keys = _ACTIVE_PARAMETER_KEYS.get(str(active_path))
        mismatch = (
            active_path != realized.get("concurrency_mode")
            or requested.get("walkers") != realized.get("concurrency_workers")
            or requested.get("chunk_sims") != realized.get("chunk_sims")
            or not str(requested.get("device", "")).startswith("cuda")
            or not isinstance(active, dict)
            or expected_keys is None
            or set(active) != expected_keys
        )
        if isinstance(active, dict):
            mismatch = mismatch or any(realized.get(name) != value for name, value in active.items())
        if mismatch:
            failures.append("requested search does not match realized active parameters")
    requested_evaluator = manifest.get("requested_evaluator")
    realized_evaluator = manifest.get("realized_evaluator")
    compile_info = manifest.get("compile")
    compile_valid = (
        isinstance(compile_info, dict)
        and compile_info.get("enabled") is True
        and compile_info.get("mode") == "max-autotune"
        and isinstance(compile_info.get("cache_dir"), str)
        and bool(compile_info.get("cache_dir"))
        and isinstance(compile_info.get("torchinductor_cache_dir"), str)
        and bool(compile_info.get("torchinductor_cache_dir"))
        and isinstance(compile_info.get("triton_cache_dir"), str)
        and bool(compile_info.get("triton_cache_dir"))
    )
    expected_stack = (
        "BatchCoalescingDispatcher(ThreadSafeGPUDispatcher(DirectGPUEvaluator))"
        if isinstance(requested, dict) and requested.get("active_path") == "walker_puct"
        else "CUDAOwnerDispatcher(ThreadSafeGPUDispatcher(DirectGPUEvaluator))"
    )
    if (
        not compile_valid
        or not isinstance(requested_evaluator, dict)
        or not isinstance(realized_evaluator, dict)
        or requested_evaluator != realized_evaluator
        or realized_evaluator.get("stack") != expected_stack
        or not _positive_int(realized_evaluator.get("direct_max_batch"))
        or realized_evaluator.get("outer_max_batch")
        != realized_evaluator.get("direct_max_batch")
        or realized_evaluator.get("n_slots") != 2
        or not isinstance(realized_evaluator.get("input_bf16"), bool)
        or not isinstance(realized_evaluator.get("legal_bf16"), bool)
        or realized_evaluator.get("compiled") is not True
        or realized_evaluator.get("model_wrapper_type") != "OptimizedModule"
    ):
        failures.append("requested evaluator does not match realized evaluator stack")
    realized_tablebase = manifest.get("realized_tablebase")
    leaf_probe_expected = (
        isinstance(requested, dict) and requested.get("active_path") == "gumbel"
    )
    if (
        not isinstance(realized_tablebase, dict)
        or realized_tablebase.get("installed") is not True
        or realized_tablebase.get("cursed_as_draw") is not True
        or realized_tablebase.get("root_probe_active") is not True
        or realized_tablebase.get("leaf_probe_active") is not leaf_probe_expected
        or not isinstance(realized_tablebase.get("n_wdl"), int)
        or int(realized_tablebase.get("n_wdl", 0)) < 510
        or not isinstance(realized_tablebase.get("n_dtz"), int)
        or int(realized_tablebase.get("n_dtz", 0)) < 510
        or not isinstance(realized_tablebase.get("max_pieces"), int)
        or int(realized_tablebase.get("max_pieces", 0)) < 6
    ):
        failures.append("production Syzygy probe was not realized")
    row_count = manifest.get("row_count")
    chunk_count = manifest.get("chunk_count")
    position_count = manifest.get("position_count")
    requested_positions = manifest.get("requested_position_count")
    excluded_positions = manifest.get("excluded_position_count")
    excluded_details = manifest.get("excluded_positions")
    counts_are_ints = (
        _positive_int(row_count)
        and _positive_int(chunk_count)
        and _positive_int(position_count)
        and _positive_int(requested_positions)
        and _nonnegative_int(excluded_positions)
    )
    if not counts_are_ints:
        failures.append("row/position/chunk accounting is inconsistent")
    else:
        assert isinstance(row_count, int)
        assert isinstance(chunk_count, int)
        assert isinstance(position_count, int)
        assert isinstance(requested_positions, int)
        assert isinstance(excluded_positions, int)
        if (
            row_count != chunk_count * position_count
            or requested_positions != position_count + excluded_positions
            or not isinstance(excluded_details, list)
            or len(excluded_details) != excluded_positions
            or any(
                not isinstance(entry, dict)
                or not isinstance(entry.get("key"), str)
                or not entry.get("key")
                or not _nonnegative_int(entry.get("chunks_observed"))
                or entry.get("chunks_required") != chunk_count
                or not isinstance(entry.get("reason"), str)
                or not entry.get("reason")
                for entry in (excluded_details or [])
            )
            or len({entry["key"] for entry in (excluded_details or [])})
            != len(excluded_details or [])
        ):
            failures.append("row/position/chunk accounting is inconsistent")
    decision_grade = manifest.get("decision_grade") is True and not failures
    if (failures or not decision_grade) and not methodology_smoke:
        detail = ", ".join(failures) if failures else "manifest is non-decision-grade"
        raise ValueError(f"trajectory provenance is not decision-grade: {detail}")
    return manifest, decision_grade


def _recomputed_trajectory_state(
    trajectory: Sequence[dict[str, Any]],
    *,
    methodology_smoke: bool,
) -> list[tuple[float, int]]:
    """Recompute the production stability/gap predicate for every row."""
    last_best = -1
    stable = 0
    states: list[tuple[float, int]] = []
    final = trajectory[-1]
    for index, row in enumerate(trajectory):
        key = str(row.get("key"))
        observed_gap = _emitted_visit_gap(row)
        if observed_gap is None:
            if not methodology_smoke:
                raise ValueError(f"{key}: emitted-action visit provenance is missing")
            observed_gap = _finite(row.get("visit_gap"))
            if index == 0 or bool(row.get("bestmove_flip")):
                stable = 0
            else:
                stable += 1
        elif not math.isclose(
            observed_gap,
            _finite(row.get("visit_gap")),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{key}: visit_gap is not the emitted action's gap")
        else:
            last_best, stable = _update_stability(
                last_best,
                stable,
                emitted_action=int(row["emitted_action"]),
                visit_gap=observed_gap,
                action_count=len(row["root_actions"]),
            )
        recorded_stable = row.get("stable_chunks")
        if recorded_stable is not None and int(recorded_stable) != stable:
            raise ValueError(f"{key}: stable_chunks disagrees with emitted move history")
        expected_continue = _complexity_continue(
            stable_chunks=stable,
            visit_gap=observed_gap,
            action_count=len(row.get("root_actions", [])),
        )
        if (
            not methodology_smoke
            and row.get("complexity_predicate_continue") is not expected_continue
        ):
            raise ValueError(f"{key}: complexity predicate disagrees with search state")
        if not methodology_smoke:
            if row["fen"] != trajectory[0]["fen"]:
                raise ValueError(f"{key}: root FEN changes within trajectory")
            shares = np.asarray(row["root_visit_shares"], dtype=np.float64)
            positive_shares = shares[shares > 0.0]
            expected_entropy = (
                float(-(positive_shares * np.log(positive_shares)).sum())
                if positive_shares.size else 0.0
            )
            if not math.isclose(
                float(row["visit_entropy"]),
                expected_entropy,
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{key}: visit_entropy disagrees with raw visits")
            action_index = row["root_actions"].index(row["emitted_action"])
            child_q = [float(value) for value in row["root_child_q"]]
            expected_q_gap = None
            if len(child_q) >= 2:
                other_q = child_q[:action_index] + child_q[action_index + 1:]
                expected_q_gap = child_q[action_index] - max(other_q)
            recorded_q_gap = row.get("q_gap")
            q_gap_matches = expected_q_gap is None and recorded_q_gap is None
            if expected_q_gap is not None and recorded_q_gap is not None:
                q_gap_matches = math.isclose(
                    float(recorded_q_gap), expected_q_gap,
                    rel_tol=1e-10, abs_tol=1e-12,
                )
            if not q_gap_matches:
                raise ValueError(f"{key}: q_gap disagrees with raw child Q values")
            previous = trajectory[index - 1] if index > 0 else None
            expected_flip = bool(
                previous is not None
                and row["emitted_action"] != previous["emitted_action"]
            )
            if row["bestmove_flip"] is not expected_flip:
                raise ValueError(f"{key}: bestmove_flip disagrees with raw actions")
            expected_q_drift = (
                None if previous is None
                else abs(float(row["root_q"]) - float(previous["root_q"]))
            )
            if expected_q_drift is None:
                if row.get("q_drift") is not None:
                    raise ValueError(f"{key}: first-chunk q_drift must be null")
            elif not math.isclose(
                float(row["q_drift"]), expected_q_drift,
                rel_tol=1e-10, abs_tol=1e-12,
            ):
                raise ValueError(f"{key}: q_drift disagrees with raw root Q")
            if previous is None:
                if row.get("visit_churn") is not None:
                    raise ValueError(f"{key}: first-chunk visit_churn must be null")
            else:
                current_shares = dict(zip(row["root_actions"], shares.tolist()))
                previous_shares = dict(zip(
                    previous["root_actions"], previous["root_visit_shares"],
                ))
                expected_churn = 0.5 * sum(
                    abs(
                        current_shares.get(action, 0.0)
                        - float(previous_shares.get(action, 0.0))
                    )
                    for action in current_shares.keys() | previous_shares.keys()
                )
                if not math.isclose(
                    float(row["visit_churn"]), expected_churn,
                    rel_tol=1e-10, abs_tol=1e-12,
                ):
                    raise ValueError(f"{key}: visit_churn disagrees with raw visits")
            if row["changes_to_final"] is not (row["uci"] != final["uci"]):
                raise ValueError(f"{key}: changes_to_final disagrees with final move")
            expected_final_regret = float(row["regret_cp"]) - float(final["regret_cp"])
            if not math.isclose(
                float(row["regret_vs_final_cp"]), expected_final_regret,
                rel_tol=1e-10, abs_tol=1e-12,
            ):
                raise ValueError(f"{key}: regret_vs_final_cp disagrees with final row")
            if previous is not None and float(row["elapsed_ms"]) < float(previous["elapsed_ms"]):
                raise ValueError(f"{key}: elapsed_ms decreases within trajectory")
        states.append((observed_gap, stable))
    return states


def load_transitions(
    input_path: Path,
    *,
    meta_path: Path | None = None,
    methodology_smoke: bool = False,
) -> tuple[list[Transition], dict[str, Any]]:
    """Load, validate, and convert trajectory rows to adjacent-chunk decisions."""
    actual_meta = meta_path or Path(str(input_path) + ".meta.json")
    manifest, decision_grade = _require_manifest(input_path, actual_meta, methodology_smoke)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(input_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not methodology_smoke and row.get("schema") != _SCHEMA:
            raise ValueError(f"line {line_number}: missing {_SCHEMA} row schema")
        if not methodology_smoke:
            _validate_decision_grade_row(row, line_number)
        rows.append(row)
    if not rows:
        raise ValueError("trajectory bank is empty")
    expected_rows = manifest.get("row_count")
    if not methodology_smoke and expected_rows != len(rows):
        raise ValueError(f"manifest says {expected_rows} rows, found {len(rows)}")

    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[str(row["key"])].append(row)
    if not methodology_smoke and len(by_key) != manifest.get("position_count"):
        raise ValueError("unique trajectory count disagrees with manifest position_count")
    excluded_keys = {
        str(entry["key"])
        for entry in manifest.get("excluded_positions", [])
        if isinstance(entry, dict) and "key" in entry
    }
    if not methodology_smoke and excluded_keys.intersection(by_key):
        raise ValueError("a trajectory is both completed and excluded")
    transitions: list[Transition] = []
    has_score_regret = all("regret_score" in row for row in rows)
    if not has_score_regret and not methodology_smoke:
        raise ValueError("expected-score regret is missing")
    metric = "regret_score" if has_score_regret else "regret_cp"
    for key, trajectory in sorted(by_key.items()):
        trajectory.sort(key=lambda row: int(row["chunk"]))
        chunks = [int(row["chunk"]) for row in trajectory]
        expected_chunk_count = manifest.get("chunk_count")
        if (
            chunks != list(range(1, len(trajectory) + 1))
            or len(trajectory) < 2
            or (not methodology_smoke and len(trajectory) != expected_chunk_count)
        ):
            raise ValueError(
                f"{key}: chunks must match the complete consecutive manifest horizon"
            )
        if not methodology_smoke:
            requested_search = manifest["requested_search"]
            chunk_sims = int(requested_search["chunk_sims"])
            if any(
                int(row["nodes"]) != int(row["chunk"]) * chunk_sims
                for row in trajectory
            ):
                raise ValueError(f"{key}: node horizons disagree with fixed chunk size")
        recomputed_states = _recomputed_trajectory_state(
            trajectory,
            methodology_smoke=methodology_smoke,
        )
        for index, (lower, upper) in enumerate(pairwise(trajectory)):
            observed_gap, stable = recomputed_states[index]
            game_id = lower.get("game_id")
            if game_id is None or int(game_id) < 0:
                if not methodology_smoke:
                    raise ValueError(f"{key}: source game_id is missing")
                game_id = int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)
            group_id = lower.get("group_id")
            expected_group = _source_group(lower)
            if group_id is None:
                if not methodology_smoke:
                    raise ValueError(f"{key}: source-scoped game group is missing")
                group_id = f"smoke:{game_id}"
            elif not methodology_smoke and group_id != expected_group:
                raise ValueError(
                    f"{key}: group_id is not (source_dir, shard, game_id)"
                )
            if upper.get("group_id", group_id) != group_id:
                raise ValueError(f"{key}: source-scoped game group changes within trajectory")
            lo_regret = (
                _strict_finite(lower, metric)
                if not methodology_smoke else _finite(lower.get(metric))
            )
            hi_regret = (
                _strict_finite(upper, metric)
                if not methodology_smoke else _finite(upper.get(metric))
            )
            lo_nodes, hi_nodes = int(lower["nodes"]), int(upper["nodes"])
            if hi_nodes <= lo_nodes:
                raise ValueError(f"{key}: node horizons are not strictly increasing")
            state = {
                "visit_gap": observed_gap,
                "visit_entropy": _finite(lower.get("visit_entropy")),
                "q_gap": _finite(lower.get("q_gap")),
                "q_gap_missing": float(lower.get("q_gap") is None),
                "bestmove_flip": float(bool(lower.get("bestmove_flip"))),
                "stable_chunks": float(stable),
                "q_drift": _finite(lower.get("q_drift")),
                "visit_churn": _finite(lower.get("visit_churn")),
                "root_q": _finite(lower.get("root_q")),
                "phase": float(int(lower.get("phase", 0))),
                "piece_count": _finite(lower.get("piece_count")),
                "legal_move_count": _finite(lower.get("legal_move_count")),
                "nodes": float(lo_nodes),
            }
            complexity_continue = lower.get("complexity_predicate_continue")
            if complexity_continue is None:
                if not methodology_smoke:
                    raise ValueError(f"{key}: clock-free complexity predicate is missing")
                complexity_continue = lower.get("current_gate_continue")
                if complexity_continue is None:
                    complexity_continue = _complexity_continue(
                        stable_chunks=stable,
                        visit_gap=observed_gap,
                        action_count=len(lower.get("root_actions", [])),
                    )
            transitions.append(Transition(
                key=key,
                group_id=str(group_id),
                horizon=hi_nodes,
                cost=hi_nodes - lo_nodes,
                gain=lo_regret - hi_regret,
                regret_before=lo_regret,
                regret_after=hi_regret,
                complexity_continue=bool(complexity_continue),
                state=state,
            ))
    costs = {transition.cost for transition in transitions}
    if len(costs) != 1:
        raise ValueError(
            "matched-spend comparison requires one fixed node tranche; "
            f"observed costs={sorted(costs)}"
        )
    info = {
        "decision_grade": decision_grade and not methodology_smoke,
        "methodology_smoke": methodology_smoke,
        "metric": metric,
        "analysis_scope": "fixed_node_horizons_only",
        "clock_conditioning_tested": False,
        "manifest": manifest,
    }
    return transitions, info


_M0_FEATURES = (
    "visit_gap", "visit_entropy", "q_gap", "q_gap_missing", "bestmove_flip",
    "stable_chunks", "q_drift", "visit_churn", "root_q", "piece_count",
    "legal_move_count",
)


def _design(transitions: Sequence[Transition], model: str) -> np.ndarray:
    rows: list[list[float]] = []
    for transition in transitions:
        state = transition.state
        base = [state[name] for name in _M0_FEATURES]
        phase = int(state["phase"])
        base.extend([float(phase == 1), float(phase == 2)])
        if model == "M1":
            log_nodes = math.log1p(state["nodes"])
            log_horizon = math.log1p(transition.horizon)
            remaining_fraction = transition.cost / transition.horizon
            context = [log_nodes, log_horizon, remaining_fraction]
            interactions = [
                value * remaining_fraction for value in (
                    state["visit_gap"], state["visit_entropy"], state["bestmove_flip"],
                    state["q_drift"], state["visit_churn"],
                )
            ]
            interactions.extend([
                state["visit_gap"] * log_horizon,
                state["visit_entropy"] * log_horizon,
            ])
            base.extend(context + interactions)
        elif model != "M0":
            raise ValueError(f"unknown model {model!r}")
        rows.append(base)
    return np.asarray(rows, dtype=np.float64)


@dataclass(frozen=True)
class _Ridge:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        standardized = (x - self.mean) / self.scale
        return np.column_stack([np.ones(len(x)), standardized]) @ self.coef


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> _Ridge:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    augmented = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(augmented.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(augmented.T @ augmented + penalty, augmented.T @ y)
    return _Ridge(mean=mean, scale=scale, coef=coef)


def grouped_folds(groups: Sequence[str], n_folds: int) -> list[np.ndarray]:
    """Deterministic group folds, balanced greedily by row count."""
    values, counts = np.unique(np.asarray(groups, dtype=str), return_counts=True)
    if len(values) < 2:
        raise ValueError("grouped CV needs at least two source games")
    n_actual = min(max(2, int(n_folds)), len(values))
    buckets: list[list[str]] = [[] for _ in range(n_actual)]
    loads = [0] * n_actual
    order = sorted(zip(values.tolist(), counts.tolist()), key=lambda item: (-item[1], item[0]))
    for group, count in order:
        target = min(range(n_actual), key=lambda index: (loads[index], index))
        buckets[target].append(group)
        loads[target] += count
    arr = np.asarray(groups, dtype=str)
    return [np.flatnonzero(np.isin(arr, bucket)) for bucket in buckets]


def _inner_alpha(
    transitions: Sequence[Transition], model: str, n_folds: int,
) -> float:
    if len({transition.group_id for transition in transitions}) < 3:
        return 1.0
    folds = grouped_folds([transition.group_id for transition in transitions], n_folds)
    x = _design(transitions, model)
    y = np.asarray([transition.gain for transition in transitions], dtype=np.float64)
    losses: dict[float, list[float]] = {alpha: [] for alpha in _ALPHAS}
    all_indices = np.arange(len(transitions))
    for valid in folds:
        train = np.setdiff1d(all_indices, valid, assume_unique=True)
        if len(train) < 2 or not len(valid):
            continue
        for alpha in _ALPHAS:
            pred = _fit_ridge(x[train], y[train], alpha).predict(x[valid])
            losses[alpha].append(float(np.mean((pred - y[valid]) ** 2)))
    return min(_ALPHAS, key=lambda alpha: (np.mean(losses[alpha]), alpha))


def held_horizon_predictions(
    transitions: Sequence[Transition], model: str, *, n_folds: int = 5,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Nested group-OOF predictions with each target horizon absent from fitting."""
    predictions = np.full(len(transitions), np.nan, dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    horizons = sorted({transition.horizon for transition in transitions})
    groups = np.asarray([transition.group_id for transition in transitions], dtype=str)
    for horizon in horizons:
        target_indices = np.flatnonzero(
            np.asarray([transition.horizon == horizon for transition in transitions])
        )
        target_groups = groups[target_indices]
        for test_local in grouped_folds(target_groups.tolist(), n_folds):
            test = target_indices[test_local]
            test_group_set = set(groups[test].tolist())
            train = np.asarray([
                index for index, transition in enumerate(transitions)
                if transition.horizon != horizon and transition.group_id not in test_group_set
            ], dtype=np.int64)
            if len(train) < 2:
                raise ValueError(f"not enough training rows with horizon {horizon} held out")
            train_rows = [transitions[index] for index in train]
            alpha = _inner_alpha(train_rows, model, n_folds)
            fitted = _fit_ridge(
                _design(train_rows, model),
                np.asarray([row.gain for row in train_rows], dtype=np.float64),
                alpha,
            )
            predictions[test] = fitted.predict(_design([transitions[index] for index in test], model))
            diagnostics.append({
                "horizon": horizon,
                "test_groups": sorted(test_group_set),
                "train_groups": sorted(set(groups[train].tolist())),
                "train_horizons": sorted({transitions[index].horizon for index in train}),
                "alpha": alpha,
            })
    if not np.isfinite(predictions).all():
        raise AssertionError("cross-validation failed to predict every transition")
    return predictions, diagnostics


def _top_indices(scores: np.ndarray, keys: Sequence[str], count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    key_order = np.asarray(keys, dtype=str)
    return np.lexsort((key_order, -scores))[:count]


def _capture(gain: float, random_gain: float, oracle_gain: float) -> float | None:
    denominator = oracle_gain - random_gain
    if denominator <= 0.0:
        return None
    return (gain - random_gain) / denominator


def _policy_result(
    transitions: Sequence[Transition], selected: np.ndarray,
    *, random_gain: float, oracle_gain: float,
) -> PolicyResult:
    mask = np.zeros(len(transitions), dtype=bool)
    mask[selected] = True
    gains = np.asarray([row.gain for row in transitions], dtype=np.float64)
    before = np.asarray([row.regret_before for row in transitions], dtype=np.float64)
    after = np.asarray([row.regret_after for row in transitions], dtype=np.float64)
    signed_gain = float(gains[mask].sum())
    realized_regret = np.where(mask, after, before)
    return PolicyResult(
        selected=int(mask.sum()),
        spend=int(sum(row.cost for row, choose in zip(transitions, mask, strict=True) if choose)),
        signed_gain=signed_gain,
        capture_over_random=_capture(signed_gain, random_gain, oracle_gain),
        regret_mean=float(realized_regret.mean()),
        regret_p95=float(np.quantile(realized_regret, 0.95)),
        regret_p99=float(np.quantile(realized_regret, 0.99)),
    )


def _complexity_predicate_indices(
    transitions: Sequence[Transition], keys: Sequence[str], count: int,
) -> np.ndarray:
    """Rank exact continues first, then apply a deterministic hardness tie-break."""
    continues = np.asarray([row.complexity_continue for row in transitions], dtype=np.int8)
    stable = np.asarray([row.state["stable_chunks"] for row in transitions])
    gaps = np.asarray([row.state["visit_gap"] for row in transitions])
    key_order = np.asarray(keys, dtype=str)
    return np.lexsort((key_order, gaps, stable, -continues))[:count]


def evaluate_horizon(
    transitions: Sequence[Transition], m0_scores: np.ndarray, m1_scores: np.ndarray,
    *, selected_count: int | None = None,
) -> dict[str, Any]:
    """Compare every policy at exactly the same count of fixed-size tranches."""
    natural_count = sum(row.complexity_continue for row in transitions)
    count = natural_count if selected_count is None else int(selected_count)
    if not 0 <= count <= len(transitions):
        raise ValueError("selected_count is outside the available positions")
    keys = [row.key for row in transitions]
    gains = np.asarray([row.gain for row in transitions], dtype=np.float64)
    complexity = _complexity_predicate_indices(transitions, keys, count)
    oracle = _top_indices(gains, keys, count)
    random_gain = float(count / len(transitions) * gains.sum())
    oracle_gain = float(gains[oracle].sum())
    policies = {
        "random": {
            "selected": count,
            "spend": int(count * transitions[0].cost),
            "signed_gain": random_gain,
            "capture_over_random": 0.0 if oracle_gain > random_gain else None,
        },
        "oracle": asdict(_policy_result(
            transitions, oracle, random_gain=random_gain, oracle_gain=oracle_gain,
        )),
        "complexity_predicate": asdict(_policy_result(
            transitions, complexity, random_gain=random_gain, oracle_gain=oracle_gain,
        )),
        "M0": asdict(_policy_result(
            transitions, _top_indices(m0_scores, keys, count),
            random_gain=random_gain, oracle_gain=oracle_gain,
        )),
        "M1": asdict(_policy_result(
            transitions, _top_indices(m1_scores, keys, count),
            random_gain=random_gain, oracle_gain=oracle_gain,
        )),
    }
    return {
        "n": len(transitions),
        "extension_fraction": count / len(transitions),
        "complexity_predicate_natural_extension_fraction": natural_count / len(transitions),
        "complexity_predicate_baseline": (
            "exact clock-free predicate selection" if count == natural_count
            else "clock-free predicate, then stability and emitted-gap tie-break"
        ),
        "signed_gain_mean_if_all": float(gains.mean()),
        "oracle_over_random_headroom_mean": (oracle_gain - random_gain) / len(transitions),
        "corrections": int((gains > 0).sum()),
        "regressions": int((gains < 0).sum()),
        "unchanged": int((gains == 0).sum()),
        "policies": policies,
    }


def _weighted_capture_delta(
    transitions: Sequence[Transition], m0: np.ndarray, m1: np.ndarray,
    allocation_fraction: float, min_oracle_headroom: float = 0.0,
) -> float | None:
    numerator = 0.0
    weight = 0
    for horizon in sorted({row.horizon for row in transitions}):
        indices = np.flatnonzero([row.horizon == horizon for row in transitions])
        rows = [transitions[index] for index in indices]
        count = min(len(rows), max(1, round(allocation_fraction * len(rows))))
        result = evaluate_horizon(
            rows, m0[indices], m1[indices], selected_count=count,
        )
        c0 = result["policies"]["M0"]["capture_over_random"]
        c1 = result["policies"]["M1"]["capture_over_random"]
        headroom = float(result["oracle_over_random_headroom_mean"])
        if c0 is None or c1 is None or headroom < min_oracle_headroom:
            return None
        numerator += len(rows) * (float(c1) - float(c0))
        weight += len(rows)
    return numerator / weight if weight else None


def _refit_oob_predictions(
    transitions: Sequence[Transition],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    model: str,
    n_folds: int,
) -> np.ndarray:
    """Select alpha and fit on one draw; predict only its untouched OOB games."""
    predictions = np.full(len(test_indices), np.nan, dtype=np.float64)
    for horizon in sorted({transitions[index].horizon for index in test_indices}):
        target_local = np.flatnonzero([
            transitions[index].horizon == horizon for index in test_indices
        ])
        train = np.asarray([
            index for index in train_indices
            if transitions[index].horizon != horizon
        ], dtype=np.int64)
        if len(train) < 2 or not len(target_local):
            raise ValueError("bootstrap replicate cannot fit every held horizon")
        train_rows = [transitions[int(index)] for index in train]
        test_rows = [transitions[int(test_indices[int(index)])] for index in target_local]
        alpha = _inner_alpha(train_rows, model, n_folds)
        fitted = _fit_ridge(
            _design(train_rows, model),
            np.asarray([row.gain for row in train_rows], dtype=np.float64),
            alpha,
        )
        predictions[target_local] = fitted.predict(_design(test_rows, model))
    if not np.isfinite(predictions).all():
        raise ValueError("bootstrap replicate left OOB rows unpredicted")
    return predictions


def cluster_bootstrap_delta(
    transitions: Sequence[Transition],
    *,
    allocation_fraction: float,
    samples: int,
    seed: int,
    n_folds: int = 5,
    min_oracle_headroom: float = 0.0,
) -> dict[str, float | int | None]:
    """Refitted game-cluster bootstrap with untouched out-of-bag evaluation."""
    groups = sorted({row.group_id for row in transitions})
    by_group = {
        group: np.flatnonzero([row.group_id == group for row in transitions])
        for group in groups
    }
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        drawn_set = {str(group) for group in drawn}
        oob_groups = [group for group in groups if group not in drawn_set]
        if len(oob_groups) < 2:
            continue
        train_indices = np.concatenate([by_group[str(group)] for group in drawn])
        test_indices = np.concatenate([by_group[group] for group in oob_groups])
        rows = [transitions[index] for index in test_indices]
        try:
            m0 = _refit_oob_predictions(
                transitions, train_indices, test_indices,
                model="M0", n_folds=n_folds,
            )
            m1 = _refit_oob_predictions(
                transitions, train_indices, test_indices,
                model="M1", n_folds=n_folds,
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
        delta = _weighted_capture_delta(
            rows, m0, m1, allocation_fraction, min_oracle_headroom,
        )
        if delta is not None and math.isfinite(delta):
            values.append(delta)
    if not values:
        return {
            "requested_samples": samples,
            "valid_samples": 0,
            "valid_fraction": 0.0,
            "mean": None,
            "lower_95": None,
            "upper_95": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "requested_samples": samples,
        "valid_samples": len(values),
        "valid_fraction": len(values) / samples,
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(array, 0.025)),
        "upper_95": float(np.quantile(array, 0.975)),
    }


def analyze(
    transitions: Sequence[Transition], *, n_folds: int, bootstrap_samples: int,
    seed: int, allocation_fraction: float, min_capture_gain: float,
    min_oracle_headroom: float, min_bootstrap_valid_fraction: float,
) -> dict[str, Any]:
    m0, m0_diagnostics = held_horizon_predictions(transitions, "M0", n_folds=n_folds)
    m1, m1_diagnostics = held_horizon_predictions(transitions, "M1", n_folds=n_folds)
    horizons: dict[str, Any] = {}
    weighted_m0 = 0.0
    weighted_m1 = 0.0
    weight = 0
    tail_ok = True
    every_horizon_eligible = True
    every_horizon_positive = True
    for horizon in sorted({row.horizon for row in transitions}):
        indices = np.flatnonzero([row.horizon == horizon for row in transitions])
        rows = [transitions[index] for index in indices]
        count = min(len(rows), max(1, round(allocation_fraction * len(rows))))
        result = evaluate_horizon(
            rows, m0[indices], m1[indices], selected_count=count,
        )
        c0 = result["policies"]["M0"]["capture_over_random"]
        c1 = result["policies"]["M1"]["capture_over_random"]
        headroom = float(result["oracle_over_random_headroom_mean"])
        eligible = (
            c0 is not None and c1 is not None
            and headroom >= min_oracle_headroom
        )
        if c0 is not None and c1 is not None:
            delta = float(c1) - float(c0)
            weighted_m0 += len(rows) * float(c0)
            weighted_m1 += len(rows) * float(c1)
            weight += len(rows)
        else:
            delta = None
        every_horizon_eligible = every_horizon_eligible and eligible
        every_horizon_positive = every_horizon_positive and bool(
            eligible and delta is not None and delta > 0.0
        )
        tail_ok = tail_ok and (
            result["policies"]["M1"]["regret_p95"]
            <= result["policies"]["M0"]["regret_p95"] + 1e-12
            and result["policies"]["M1"]["regret_p99"]
            <= result["policies"]["M0"]["regret_p99"] + 1e-12
        )
        result["advance_eligible_headroom"] = eligible
        result["M1_minus_M0_oracle_capture"] = delta
        horizons[str(horizon)] = result
    mean_m0 = weighted_m0 / weight if weight else None
    mean_m1 = weighted_m1 / weight if weight else None
    capture_gain = (
        mean_m1 - mean_m0
        if mean_m0 is not None and mean_m1 is not None else None
    )
    bootstrap = cluster_bootstrap_delta(
        transitions,
        allocation_fraction=allocation_fraction,
        samples=bootstrap_samples, seed=seed, n_folds=n_folds,
        min_oracle_headroom=min_oracle_headroom,
    )
    bootstrap_resolution_ok = (
        bootstrap_samples >= _MIN_DECISION_GRADE_BOOTSTRAP_SAMPLES
    )
    advance = bool(
        capture_gain is not None
        and capture_gain >= min_capture_gain
        and every_horizon_eligible and every_horizon_positive
        and bootstrap_resolution_ok
        and float(bootstrap["valid_fraction"] or 0.0) >= min_bootstrap_valid_fraction
        and bootstrap["lower_95"] is not None and float(bootstrap["lower_95"]) > 0.0
        and tail_ok
    )
    return {
        "preregistered_rule": {
            "allocation_fraction": allocation_fraction,
            "minimum_M1_minus_M0_oracle_capture": min_capture_gain,
            "minimum_oracle_over_random_headroom_mean": min_oracle_headroom,
            "M1_minus_M0_positive_on_every_held_horizon": True,
            "refitted_OOB_game_cluster_bootstrap_lower_95_above_zero": True,
            "bootstrap_samples": bootstrap_samples,
            "minimum_decision_grade_bootstrap_samples": (
                _MIN_DECISION_GRADE_BOOTSTRAP_SAMPLES
            ),
            "minimum_bootstrap_valid_fraction": min_bootstrap_valid_fraction,
            "M1_p95_and_p99_regret_not_worse_than_M0": True,
        },
        "verdict": "ADVANCE_TO_CLOCK_BANK" if advance else "KILL_BUDGET_CONTEXT",
        "scope": "fixed_node_horizons_only",
        "clock_controller_authorized": False,
        "weighted_capture_M0": mean_m0,
        "weighted_capture_M1": mean_m1,
        "M1_minus_M0_oracle_capture": capture_gain,
        "every_horizon_eligible": every_horizon_eligible,
        "every_horizon_positive": every_horizon_positive,
        "tail_rule_passed": tail_ok,
        "bootstrap_resolution_passed": bootstrap_resolution_ok,
        "bootstrap_refit_oob_M1_minus_M0": bootstrap,
        "horizons": horizons,
        "cv_diagnostics": {"M0": m0_diagnostics, "M1": m1_diagnostics},
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="analyze_chunk_controller")
    parser.add_argument("--in", dest="input_path", type=Path, required=True)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--methodology-smoke", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allocation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--min-capture-gain", type=float, default=0.05,
        help="minimum M1-minus-M0 fraction of oracle-over-random headroom",
    )
    parser.add_argument(
        "--min-oracle-headroom", type=float, default=1e-4,
        help="minimum per-row oracle-over-random regret headroom at every horizon",
    )
    parser.add_argument(
        "--min-bootstrap-valid-fraction", type=float, default=0.95,
        help="minimum fraction of cluster replicates with eligible headroom",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if (
        args.folds < 2 or args.bootstrap_samples < 1
        or not 0.0 < args.allocation_fraction < 1.0
        or args.min_capture_gain < 0.0
        or args.min_oracle_headroom < 0.0
        or not 0.0 < args.min_bootstrap_valid_fraction <= 1.0
    ):
        raise SystemExit(
            "--folds must be >=2, --bootstrap-samples >=1, and "
            "--allocation-fraction in (0,1), non-negative gain/headroom gates, "
            "and --min-bootstrap-valid-fraction in (0,1]"
        )
    actual_meta = args.meta or Path(str(args.input_path) + ".meta.json")
    try:
        _require_bootstrap_resolution(args.bootstrap_samples, args.methodology_smoke)
        _require_safe_output_path(args.input_path, actual_meta, args.out)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    transitions, info = load_transitions(
        args.input_path, meta_path=actual_meta, methodology_smoke=args.methodology_smoke,
    )
    result = analyze(
        transitions,
        n_folds=args.folds,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        allocation_fraction=args.allocation_fraction,
        min_capture_gain=args.min_capture_gain,
        min_oracle_headroom=args.min_oracle_headroom,
        min_bootstrap_valid_fraction=args.min_bootstrap_valid_fraction,
    )
    if not info["decision_grade"]:
        result["verdict"] = "METHODOLOGY_SMOKE_ONLY"
        result["clock_controller_authorized"] = False
    payload = {"input": info, "analysis": result}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.out, rendered)
    print(rendered)


if __name__ == "__main__":
    main()
