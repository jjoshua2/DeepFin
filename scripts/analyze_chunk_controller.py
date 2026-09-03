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
import platform
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

import chess
import numpy as np

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.eval.audit import (
    AUDIT_REGRET_CAP_CP,
    legal_full_indices,
    phase_bucket,
    position_key,
)
from chess_anti_engine.mcts.search_options import SEARCH_OPTIONS
from chess_anti_engine.moves import ActionDecodeError, POLICY_SIZE, index_to_move_strict
from scripts.reachable_oracle import solve_reachable_oracle
from scripts.repo_output_guard import repo_controlled_output

_SCHEMA = "deepfin.chunk_trajectory.v3"
_CP_TO_SCORE_C = 300.0
_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
_COMPLEXITY_VISIT_GAP = 0.25
_COMPLEXITY_STABLE_CHUNKS = 2
_MIN_DECISION_GRADE_BOOTSTRAP_SAMPLES = 1000
_PRODUCTION_WDL_FILES = 875
_PRODUCTION_DTZ_FILES = 510
_PRODUCTION_TB_COMPONENTS = ((510, 145), (365, 365))
_PRODUCTION_GSS_HALVING_REV = 3
_PRODUCTION_WALKERS = 2
_MIN_DECISION_GRADE_CHUNKS = 4
_CANONICAL_FOLDS = 5
_CANONICAL_BOOTSTRAP_SAMPLES = 2000
_CANONICAL_SEED = 0
_CANONICAL_ALLOCATION_FRACTION = 0.2
_CANONICAL_MIN_CAPTURE_GAIN = 0.05
_CANONICAL_MIN_ORACLE_HEADROOM = 1e-4
_CANONICAL_MIN_BOOTSTRAP_VALID_FRACTION = 0.95
_PREREGISTRATION_SCHEMA = "deepfin.chunk_controller_preregistration.v1"
_NATIVE_MODULES = [
    "chess_anti_engine.encoding._lc0_ext",
    "chess_anti_engine.mcts._mcts_tree",
]
_ACTIVE_PARAMETER_KEYS = {
    "walker_puct": {
        "c_puct", "cpuct_factor", "cpuct_base", "fpu_reduction",
        "vloss_weight", "walker_gather",
    },
    "gumbel": {
        "c_scale", "c_visit", "c_visit_root", "c_scale_root",
        "q_visit_exp_root", "topk", "policy_temp", "halving_div",
        "root_noise_scale", "vloss_weight", "minibatch_size",
    },
}


def _score(cp: float) -> float:
    exponent = float(cp) * math.log(10.0) / _CP_TO_SCORE_C
    if exponent >= 0.0:
        return 1.0 / (1.0 + math.exp(-exponent))
    exp_value = math.exp(exponent)
    return exp_value / (1.0 + exp_value)


@dataclass(frozen=True)
class Transition:
    key: str
    group_id: str
    horizon: int
    hard_horizon: int
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


def _git_state() -> tuple[str, bool]:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            [
                "git", "-C", str(repo_root), "status", "--porcelain",
                "--untracked-files=normal",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return sha, bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _artifact_snapshot(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(resolved),
    }


def _analyzer_provenance(
    start_artifact: dict[str, Any], start_git_sha: str, start_git_dirty: bool,
) -> dict[str, Any]:
    end_artifact = _artifact_snapshot(Path(__file__))
    end_git_sha, end_git_dirty = _git_state()
    stable = (
        start_artifact == end_artifact
        and len(start_git_sha) == 40
        and all(char in "0123456789abcdef" for char in start_git_sha.lower())
        and end_git_sha == start_git_sha
        and not start_git_dirty
        and not end_git_dirty
    )
    return {
        "decision_grade": stable,
        "git_sha": start_git_sha,
        "git_dirty": start_git_dirty,
        "final_git_sha": end_git_sha,
        "final_git_dirty": end_git_dirty,
        "script": start_artifact,
        "script_stable": start_artifact == end_artifact,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": str(np.__version__),
        "python_chess_version": str(chess.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


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
    input_path: Path,
    meta_path: Path,
    output_path: Path | None,
    *,
    manifest: dict[str, Any] | None = None,
) -> None:
    if output_path is None:
        return
    output = output_path.expanduser().resolve()
    protected = {
        input_path.expanduser().resolve(),
        meta_path.expanduser().resolve(),
        Path(__file__).resolve(),
        Path(solve_reachable_oracle.__code__.co_filename).resolve(),
    }
    if repo_controlled_output(output_path, Path(__file__).resolve().parents[1]):
        raise ValueError("--out must not overwrite a tracked or repository-control path")
    if manifest is not None:
        for name in (
            "producer_script", "checkpoint", "checkpoint_params", "audit_set",
            "matched_rows", "preregistration", "mcts_extension", "lc0_extension",
        ):
            artifact = manifest.get(name)
            if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
                protected.add(Path(artifact["path"]).expanduser().resolve())
    if output in protected:
        raise ValueError("--out must not overwrite a consumed input artifact")
    syzygy = manifest.get("syzygy") if manifest is not None else None
    directories = syzygy.get("directories") if isinstance(syzygy, dict) else None
    for row in directories if isinstance(directories, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            continue
        try:
            output.relative_to(Path(row["path"]).expanduser().resolve())
        except ValueError:
            continue
        raise ValueError("--out must not be inside a consumed Syzygy directory")


def _require_bootstrap_resolution(samples: int, methodology_smoke: bool) -> None:
    if not methodology_smoke and samples < _MIN_DECISION_GRADE_BOOTSTRAP_SAMPLES:
        raise ValueError(
            "decision-grade analysis requires at least "
            f"{_MIN_DECISION_GRADE_BOOTSTRAP_SAMPLES} bootstrap samples; "
            "use --methodology-smoke only for a smaller pipeline check"
        )


def _is_canonical_decision_rule(
    *, n_folds: int, bootstrap_samples: int, seed: int,
    allocation_fraction: float, min_capture_gain: float,
    min_oracle_headroom: float, min_bootstrap_valid_fraction: float,
) -> bool:
    """Whether a rule exactly matches the frozen decision-grade preregistration."""
    return bool(
        n_folds == _CANONICAL_FOLDS
        and bootstrap_samples == _CANONICAL_BOOTSTRAP_SAMPLES
        and seed == _CANONICAL_SEED
        and allocation_fraction == _CANONICAL_ALLOCATION_FRACTION
        and min_capture_gain == _CANONICAL_MIN_CAPTURE_GAIN
        and min_oracle_headroom == _CANONICAL_MIN_ORACLE_HEADROOM
        and min_bootstrap_valid_fraction == _CANONICAL_MIN_BOOTSTRAP_VALID_FRACTION
    )


def _canonical_analysis_contract() -> dict[str, int | float]:
    """Exact analysis knobs that a decision-grade collection must freeze."""
    return {
        "folds": _CANONICAL_FOLDS,
        "bootstrap_samples": _CANONICAL_BOOTSTRAP_SAMPLES,
        "seed": _CANONICAL_SEED,
        "allocation_fraction": _CANONICAL_ALLOCATION_FRACTION,
        "min_capture_gain": _CANONICAL_MIN_CAPTURE_GAIN,
        "min_oracle_headroom": _CANONICAL_MIN_ORACLE_HEADROOM,
        "min_bootstrap_valid_fraction": _CANONICAL_MIN_BOOTSTRAP_VALID_FRACTION,
    }


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


def _canonical_cuda_device_string(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("cuda:"):
        return False
    suffix = value.removeprefix("cuda:")
    return suffix.isdigit() and str(int(suffix)) == suffix


def _cuda_device_matches_resolved(raw: Any, resolved: Any) -> bool:
    return bool(
        _canonical_cuda_device_string(resolved)
        and (raw == "cuda" or raw == resolved)
    )


def _artifact_provenance_complete(artifact: Any) -> bool:
    return bool(
        isinstance(artifact, dict)
        and isinstance(artifact.get("path"), str)
        and artifact.get("path")
        and _positive_int(artifact.get("size"))
        and _nonnegative_int(artifact.get("mtime_ns"))
        and _valid_sha256(artifact.get("sha256"))
    )


def _git_file_at_commit(commit: str, relative_path: str) -> bytes | None:
    """Read a tracked file exactly as it existed in ``commit``."""
    relative = PurePosixPath(relative_path)
    if (
        len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit.lower())
        or not relative_path
        or relative.is_absolute()
        or ":" in relative_path
        or any(part in ("", ".", "..") for part in relative.parts)
    ):
        return None
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative.as_posix()}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def _preregistration_is_only_source_change(
    source_sha: str, producer_sha: str, relative_path: str,
) -> bool:
    """Prove the collection commit only adds the generated plan to its source SHA."""
    if _git_file_at_commit(source_sha, relative_path) is not None:
        return False
    repo_root = Path(__file__).resolve().parents[1]
    try:
        ancestor = subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor",
             source_sha, producer_sha],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        changed = subprocess.run(
            ["git", "-C", str(repo_root), "diff", "--name-only", "-z",
             f"{source_sha}..{producer_sha}", "--"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return False
    return bool(
        ancestor.returncode == 0
        and changed.returncode == 0
        and changed.stdout == relative_path.encode() + b"\0"
    )


def _strict_json_object(document: str) -> dict[str, Any]:
    """Parse a JSON object while rejecting duplicate keys."""
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(document, object_pairs_hook=no_duplicates)
    if not isinstance(value, dict):
        raise ValueError("preregistration must be a JSON object")
    return value


def _preregistration_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build the exact producer and analysis contract a plan must contain."""
    def artifact_sha(name: str) -> Any:
        value = manifest.get(name)
        return value.get("sha256") if isinstance(value, dict) else None

    compile_info = manifest.get("compile")
    return {
        "schema": _PREREGISTRATION_SCHEMA,
        "producer": {
            "source_git_sha": manifest.get("producer_git_sha"),
            "checkpoint_sha256": artifact_sha("checkpoint"),
            "checkpoint_params_sha256": artifact_sha("checkpoint_params"),
            "audit_set_sha256": artifact_sha("audit_set"),
            "matched_rows_sha256": artifact_sha("matched_rows"),
            "max_positions": manifest.get("requested_max_positions"),
            "requested_search": manifest.get("requested_search"),
            "requested_model_search_contract": manifest.get(
                "requested_model_search_contract"
            ),
            "requested_evaluator": manifest.get("requested_evaluator"),
            "compile": {
                "enabled": (
                    compile_info.get("enabled")
                    if isinstance(compile_info, dict) else None
                ),
                "mode": (
                    compile_info.get("mode")
                    if isinstance(compile_info, dict) else None
                ),
            },
            "syzygy": manifest.get("syzygy"),
        },
        "analysis": _canonical_analysis_contract(),
    }


def _preregistered_design_failures(manifest: dict[str, Any]) -> list[str]:
    """Authenticate and compare the frozen producer plus analyzer contract."""
    artifact = manifest.get("preregistration")
    document = manifest.get("preregistration_document")
    producer_sha = manifest.get("producer_git_sha")
    if not _artifact_provenance_complete(artifact) or not isinstance(artifact, dict):
        return ["preregistration artifact provenance is incomplete"]
    relative_path = artifact.get("repo_relative_path")
    if not isinstance(document, str) or not isinstance(relative_path, str):
        return ["preregistration document or tracked path is missing"]
    try:
        document_bytes = document.encode("utf-8", errors="strict")
    except UnicodeError:
        return ["preregistration document is not valid UTF-8"]
    if (
        artifact.get("size") != len(document_bytes)
        or artifact.get("sha256") != hashlib.sha256(document_bytes).hexdigest()
    ):
        return ["preregistration document does not match its artifact identity"]
    if (
        not isinstance(producer_sha, str)
        or _git_file_at_commit(producer_sha, relative_path) != document_bytes
    ):
        return ["preregistration was not tracked verbatim by the producer commit"]
    try:
        payload = _strict_json_object(document)
    except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
        return [f"preregistration document is invalid: {exc}"]

    payload_producer = payload.get("producer")
    source_sha = (
        payload_producer.get("source_git_sha")
        if isinstance(payload_producer, dict) else None
    )
    if (
        not isinstance(source_sha, str)
        or not _preregistration_is_only_source_change(
            source_sha, producer_sha, relative_path,
        )
    ):
        return [
            "producer commit is not the source snapshot plus only its preregistration"
        ]
    expected = _preregistration_payload(manifest)
    expected_producer = expected.get("producer")
    if isinstance(expected_producer, dict):
        expected_producer["source_git_sha"] = source_sha
    if payload != expected:
        return ["realized collection does not match the preregistered design"]
    return []


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
        and artifact.get("freshness_check") == {
            "modules": _NATIVE_MODULES,
            "minimum_gcc_major": 15,
            "production_recipe_required": True,
            "passed": True,
            "issues": [],
        }
    )


def _compatible_lc0_extension(artifact: Any) -> bool:
    return bool(
        _artifact_provenance_complete(artifact)
        and str(artifact.get("path", "")).endswith((".so", ".pyd"))
        and artifact.get("cboard_encode_full") is True
        and artifact.get("freshness_check") == {
            "modules": _NATIVE_MODULES,
            "minimum_gcc_major": 15,
            "production_recipe_required": True,
            "passed": True,
            "issues": [],
        }
    )


_MODEL_SEARCH_CONTRACT_KEYS = {
    "model_input_history_encoding",
    "model_input_extra_features",
    "model_policy_encoding",
    "model_compute_relations",
    "search_input_history_encoding",
    "search_input_extra_features",
    "search_policy_encoding",
    "search_compute_relations",
    "evaluator_input_planes",
    "walker_input_planes",
    "walker_compute_relations",
}


def _valid_model_search_contract(contract: Any, *, walker: bool) -> bool:
    if not isinstance(contract, dict) or set(contract) != _MODEL_SEARCH_CONTRACT_KEYS:
        return False
    string_fields = (
        "model_input_history_encoding", "model_input_extra_features",
        "model_policy_encoding", "search_input_history_encoding",
        "search_input_extra_features", "search_policy_encoding",
    )
    if any(not isinstance(contract.get(name), str) or not contract[name] for name in string_fields):
        return False
    if not isinstance(contract.get("model_compute_relations"), bool):
        return False
    if not isinstance(contract.get("search_compute_relations"), bool):
        return False
    if (
        contract["model_input_history_encoding"]
        != contract["search_input_history_encoding"]
        or contract["model_input_extra_features"]
        != contract["search_input_extra_features"]
        or contract["model_policy_encoding"] != contract["search_policy_encoding"]
        or contract["model_compute_relations"]
        is not contract["search_compute_relations"]
    ):
        return False
    try:
        planes = input_plane_count(contract["model_input_extra_features"])
    except (TypeError, ValueError):
        return False
    if contract.get("evaluator_input_planes") != planes:
        return False
    if walker:
        return (
            contract.get("walker_input_planes") == planes
            and contract.get("walker_compute_relations")
            is contract["search_compute_relations"]
        )
    return (
        contract.get("walker_input_planes") is None
        and contract.get("walker_compute_relations") is None
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
        return tuple(
            (int(row["rtbw_count"]), int(row["rtbz_count"])) for row in rows
        )
    except (KeyError, TypeError, ValueError):
        return ()


def _valid_root_tb_control(value: Any) -> bool:
    control_fen = "7k/8/8/8/8/8/8/KQ6 w - - 0 1"
    if not isinstance(value, dict) or value.get("fen") != control_fen:
        return False
    bestmove = value.get("bestmove_uci")
    if not isinstance(bestmove, str):
        return False
    try:
        move = chess.Move.from_uci(bestmove)
    except ValueError:
        return False
    return (
        move in chess.Board(control_fen).legal_moves
        and value.get("nodes") == 1
        and value.get("tbhits") == 1
        and value.get("root_declined") is None
        and value.get("tree_created") is False
        and value.get("passed") is True
    )


def _update_stability(
    last_best: int,
    stable_chunks: int,
    *,
    emitted_action: int,
    visit_gap: float,
    action_count: int,
) -> tuple[int, int]:
    """Mirror the full stability path through ``SearchWorker._abort_ready``."""
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


def _active_search_values_valid(active_path: str, values: Any) -> bool:
    if not isinstance(values, dict):
        return False
    registry_path = "walker" if active_path == "walker_puct" else active_path
    options = {option.field: option for option in SEARCH_OPTIONS}
    integer_fields = {
        "vloss_weight", "walker_gather", "topk", "halving_div", "minibatch_size",
    }
    for field, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        numeric = float(value)
        if not math.isfinite(numeric):
            return False
        if field in integer_fields and not isinstance(value, int):
            return False
        if field == "walker_gather" and int(value) < 1:
            return False
        option = options.get(field)
        if option is None or registry_path not in option.live_in:
            continue
        if option.lo is not None and numeric < option.lo:
            return False
        if option.hi is not None and numeric > option.hi:
            return False
    return True


def _validate_decision_grade_row(
    row: dict[str, Any], line_number: int, *, require_full_root_support: bool,
) -> None:
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
    if not board.turn:
        raise ValueError(f"{key}: audit root is not side-to-move canonical")
    if key != position_key(board):
        raise ValueError(f"{key}: position key disagrees with root FEN")
    group = _source_group(row)
    if group is None or row.get("group_id") != group:
        raise ValueError(f"{key}: group_id is not (source_dir, shard, game_id)")
    if not _positive_int(row.get("chunk")) or not _positive_int(row.get("nodes")):
        raise ValueError(f"{key}: chunk and nodes must be positive integers")
    if (
        not _nonnegative_int(row.get("tb_probes"))
        or not _nonnegative_int(row.get("tb_hits"))
        or int(row["tb_hits"]) > int(row["tb_probes"])
    ):
        raise ValueError(f"{key}: tablebase probe counters are invalid")
    for name in (
        "elapsed_ms", "regret_cp", "regret_score", "regret_vs_final_cp",
        "deep_reference_best_cp", "visit_gap", "visit_entropy", "root_q",
    ):
        _strict_finite(row, name)
    if not 0.0 <= float(row["regret_cp"]) <= AUDIT_REGRET_CAP_CP:
        raise ValueError(f"{key}: regret_cp is outside the audit regret domain")
    if not 0.0 <= float(row["regret_score"]) <= 1.0:
        raise ValueError(f"{key}: regret_score is outside [0, 1]")
    if not -1.0 <= float(row["visit_gap"]) <= 1.0:
        raise ValueError(f"{key}: visit_gap is outside [-1, 1]")
    if not -1.0 <= float(row["root_q"]) <= 1.0:
        raise ValueError(f"{key}: root_q is outside [-1, 1]")
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
    if (
        not isinstance(phase, int) or isinstance(phase, bool)
        or phase != phase_bucket(int(row["piece_count"]))
    ):
        raise ValueError(f"{key}: phase disagrees with root FEN morphology")
    source = row.get("source")
    if not isinstance(source, int) or isinstance(source, bool) or source not in (0, 1):
        raise ValueError(f"{key}: source must be audit source 0 or 1")
    if not _nonnegative_int(row.get("stable_chunks")):
        raise ValueError(f"{key}: stable_chunks must be a non-negative integer")
    if int(row["stable_chunks"]) >= int(row["chunk"]):
        raise ValueError(f"{key}: stable_chunks cannot reach the current chunk number")
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
        if not -2.0 <= float(q_gap) <= 2.0:
            raise ValueError(f"{key}: q_gap is outside [-2, 2]")
    for name in ("q_drift", "visit_churn"):
        value = row.get(name)
        if value is None and row["chunk"] == 1:
            continue
        numeric = _strict_finite(row, name)
        upper = 2.0 if name == "q_drift" else 1.0
        if not 0.0 <= numeric <= upper:
            raise ValueError(f"{key}: {name} is outside [0, {upper:g}]")
    actions = row.get("root_actions")
    visits = row.get("root_visits")
    shares = row.get("root_visit_shares")
    child_q = row.get("root_child_q")
    child_q_observed = row.get("root_child_q_observed")
    action_regret = row.get("root_action_regret_cp")
    action_reference_cp = row.get("root_action_reference_cp")
    action_reference_listed = row.get("root_action_reference_listed")
    deep_reference_move_cp = row.get("deep_reference_move_cp")
    if (
        not isinstance(actions, list) or not actions
        or any(not isinstance(action, int) or isinstance(action, bool) for action in actions)
        or len(set(actions)) != len(actions)
        or not isinstance(visits, list) or len(visits) != len(actions)
        or any(not _nonnegative_int(visit) for visit in visits)
        or not isinstance(shares, list) or len(shares) != len(actions)
        or not isinstance(child_q, list) or len(child_q) != len(actions)
        or not isinstance(child_q_observed, list)
        or len(child_q_observed) != len(actions)
        or any(not isinstance(observed, bool) for observed in child_q_observed)
        or not isinstance(action_regret, list) or len(action_regret) != len(actions)
        or not isinstance(action_reference_cp, list)
        or len(action_reference_cp) != len(actions)
        or not isinstance(action_reference_listed, list)
        or len(action_reference_listed) != len(actions)
        or any(not isinstance(listed, bool) for listed in action_reference_listed)
        or not isinstance(deep_reference_move_cp, dict)
        or not deep_reference_move_cp
        or any(not isinstance(uci, str) or not uci for uci in deep_reference_move_cp)
    ):
        raise ValueError(f"{key}: malformed root action/visit/share/Q arrays")
    values = np.asarray(shares, dtype=np.float64)
    q_values = np.asarray(child_q, dtype=np.float64)
    action_regret_values = np.asarray(action_regret, dtype=np.float64)
    action_reference_values = np.asarray(action_reference_cp, dtype=np.float64)
    visit_values = np.asarray(visits, dtype=np.float64)
    visit_total = float(visit_values.sum())
    expected_shares = (
        visit_values / visit_total if visit_total > 0.0 else np.zeros_like(visit_values)
    )
    unvisited_forced_gumbel = bool(
        not require_full_root_support
        and board.legal_moves.count() == 1
        and len(actions) == 1
        and visit_total == 0.0
    )
    if (
        not np.isfinite(values).all() or (values < 0.0).any()
        or (visit_total <= 0.0 and len(actions) != 1)
        or (visit_total != int(row["nodes"]) and not unvisited_forced_gumbel)
        or not np.allclose(values, expected_shares, rtol=1e-10, atol=1e-12)
    ):
        raise ValueError(f"{key}: root visits/shares disagree with completed nodes")
    if not np.isfinite(q_values).all() or (np.abs(q_values) > 1.0).any():
        raise ValueError(f"{key}: root child Q values must be finite and inside [-1, 1]")
    if child_q_observed != [int(visit) > 0 for visit in visits]:
        raise ValueError(f"{key}: child-Q observation mask disagrees with visits")
    if (
        not np.isfinite(action_regret_values).all()
        or (action_regret_values < 0.0).any()
        or (action_regret_values > AUDIT_REGRET_CAP_CP).any()
    ):
        raise ValueError(
            f"{key}: root action regrets must be finite and inside the audit cap"
        )
    if not np.isfinite(action_reference_values).all():
        raise ValueError(f"{key}: root action reference CP values must be finite")
    try:
        move_cp = {
            str(uci): float(value) for uci, value in deep_reference_move_cp.items()
            if not isinstance(value, bool)
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}: deep reference move CP values must be finite") from exc
    if len(move_cp) != len(deep_reference_move_cp) or not all(
        math.isfinite(value) for value in move_cp.values()
    ):
        raise ValueError(f"{key}: deep reference move CP values must be finite")
    legal_uci = {move.uci() for move in board.legal_moves}
    if not set(move_cp).issubset(legal_uci):
        raise ValueError(f"{key}: deep reference contains an illegal move")
    if not math.isclose(
        float(row["deep_reference_best_cp"]), max(move_cp.values()),
        rel_tol=1e-10, abs_tol=1e-12,
    ):
        raise ValueError(f"{key}: deep reference best CP disagrees with move values")
    try:
        if any(not 0 <= action < POLICY_SIZE for action in actions):
            bad = next(action for action in actions if not 0 <= action < POLICY_SIZE)
            raise ActionDecodeError(bad, board, "outside the native action space")
        root_moves = [index_to_move_strict(action, board) for action in actions]
        legal_root_actions = {
            action for action, move in zip(actions, root_moves, strict=True)
            if move in board.legal_moves
        }
    except ActionDecodeError as exc:
        raise ValueError(f"{key}: root action cannot be decoded") from exc
    if len(legal_root_actions) != len(actions):
        raise ValueError(f"{key}: root action is illegal for the recorded FEN")
    worst_listed_cp = min(move_cp.values())
    expected_listed = [move.uci() in move_cp for move in root_moves]
    expected_reference = np.asarray([
        move_cp.get(move.uci(), worst_listed_cp) for move in root_moves
    ], dtype=np.float64)
    if action_reference_listed != expected_listed or not np.allclose(
        action_reference_values, expected_reference, rtol=1e-10, atol=1e-12,
    ):
        raise ValueError(f"{key}: action references disagree with raw deep reference")
    if require_full_root_support:
        expected_actions = {int(action) for action in legal_full_indices(board)[1]}
        if set(actions) != expected_actions:
            raise ValueError(f"{key}: walker root does not contain every legal action")
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
            if not 0 <= action < POLICY_SIZE:
                raise ActionDecodeError(action, pv_board, "outside the native action space")
            move = index_to_move_strict(action, pv_board)
        except ActionDecodeError as exc:
            raise ValueError(f"{key}: PV action cannot be decoded") from exc
        if move not in pv_board.legal_moves or move.uci() != uci:
            raise ValueError(f"{key}: PV action/UCI is illegal for the recorded FEN")
        pv_board.push(move)


def _require_manifest(
    input_path: Path, meta_path: Path, methodology_smoke: bool,
    *,
    input_bytes: bytes | None = None,
    meta_bytes: bytes | None = None,
) -> tuple[dict[str, Any], bool, bool]:
    if meta_bytes is None and not meta_path.is_file():
        if methodology_smoke:
            return {}, False, False
        raise ValueError(
            f"decision-grade analysis requires {meta_path}; pass --methodology-smoke "
            "only to exercise the estimator on a legacy bank"
        )
    consumed_input = input_path.read_bytes() if input_bytes is None else input_bytes
    consumed_meta = meta_path.read_bytes() if meta_bytes is None else meta_bytes
    manifest = json.loads(consumed_meta)
    failures: list[str] = []
    if manifest.get("schema") != _SCHEMA:
        failures.append(f"schema={manifest.get('schema')!r}")
    if manifest.get("complete") is not True:
        failures.append("complete is not true")
    if manifest.get("analysis_scope") != "fixed_node_horizons_only":
        failures.append(f"analysis_scope={manifest.get('analysis_scope')!r}")
    if manifest.get("clock_conditioning_available") is not False:
        failures.append("clock scope is ambiguous")
    if manifest.get("elapsed_measurement") != {
        "kind": "callback_instrumented_wall_time",
        "usable_for_controller_or_cost_analysis": False,
    }:
        failures.append("elapsed-time instrumentation scope is ambiguous")
    output = manifest.get("output")
    if (
        not isinstance(output, dict)
        or output.get("sha256") != hashlib.sha256(consumed_input).hexdigest()
        or output.get("size") != len(consumed_input)
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
    preregistration_failures = _preregistered_design_failures(manifest)
    failures.extend(preregistration_failures)
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
    if not _compatible_lc0_extension(manifest.get("lc0_extension")):
        failures.append("native CBoard encoding extension provenance is incomplete")
    artifact_stability = manifest.get("artifact_stability")
    if (
        not isinstance(artifact_stability, dict)
        or artifact_stability.get("passed") is not True
        or artifact_stability.get("changed") != []
        or artifact_stability.get("final_git_sha") != manifest.get("producer_git_sha")
        or artifact_stability.get("final_git_dirty") is not False
    ):
        failures.append("consumed artifacts or producer checkout changed during collection")
    warmup = manifest.get("search_warmup")
    if (
        not isinstance(warmup, dict)
        or warmup.get("completed") is not True
        or warmup.get("excluded_from_timing") is not True
        or not _positive_int(warmup.get("requested_nodes"))
        or not _positive_int(warmup.get("realized_nodes"))
        or warmup.get("realized_nodes") != warmup.get("requested_nodes")
        or warmup.get("tree_reset_after") is not True
        or warmup.get("tablebase_counters_reset_after") is not True
    ):
        failures.append("production search warmup was not completed and isolated")
    if manifest.get("game_group_kind") != "source_dir:shard:game_id":
        failures.append(f"game_group_kind={manifest.get('game_group_kind')!r}")
    if manifest.get("root_position_history") != "fen_only_from_audit_fen":
        failures.append(f"root_position_history={manifest.get('root_position_history')!r}")
    if manifest.get("root_tree_state") != "fresh_per_position_no_cross_move_reuse":
        failures.append(f"root_tree_state={manifest.get('root_tree_state')!r}")
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
        or tuple(Path(path).name for path in directory_paths)
        != ("syzygy_3-4-5", "syzygy_6")
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
            or active_path != "walker_puct"
            or requested.get("walkers") != realized.get("concurrency_workers")
            or requested.get("walkers") != _PRODUCTION_WALKERS
            or requested.get("chunk_sims") != realized.get("chunk_sims")
            or not _positive_int(requested.get("max_chunks"))
            or int(requested.get("max_chunks", 0)) < _MIN_DECISION_GRADE_CHUNKS
            or requested.get("max_chunks") != manifest.get("chunk_count")
            or not _positive_int(requested.get("chunk_sims"))
            or not 32 <= int(requested.get("chunk_sims", 0)) <= 1_048_576
            or not _positive_int(requested.get("walkers"))
            or not 1 <= int(requested.get("walkers", 0)) <= 64
            or not (
                requested.get("device") == "cuda"
                or _canonical_cuda_device_string(requested.get("device"))
            )
            or not _active_search_values_valid(str(active_path), active)
            or expected_keys is None
            or not isinstance(active, dict)
            or set(active) != expected_keys
        )
        if isinstance(active, dict):
            mismatch = mismatch or any(realized.get(name) != value for name, value in active.items())
        if mismatch:
            failures.append("requested search does not match realized active parameters")
    requested_contract = manifest.get("requested_model_search_contract")
    realized_contract = manifest.get("realized_model_search_contract")
    walker_contract = (
        isinstance(requested, dict) and requested.get("active_path") == "walker_puct"
    )
    if (
        requested_contract != realized_contract
        or not _valid_model_search_contract(requested_contract, walker=walker_contract)
    ):
        failures.append("requested model/search encoding contract was not realized")
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
    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, dict)
        or any(
            not isinstance(runtime.get(name), str) or not runtime.get(name)
            for name in (
                "python_version", "python_implementation", "python_executable",
                "numpy_version", "python_chess_version", "platform", "machine",
                "torch_version", "nvidia_driver_version",
            )
        )
        or not isinstance(runtime.get("torch_version"), str)
        or not runtime.get("torch_version")
        or not isinstance(runtime.get("torch_cuda_version"), str)
        or not runtime.get("torch_cuda_version")
        or runtime.get("torch_cuda_version") == "None"
        or not _positive_int(runtime.get("cudnn_version"))
        or not isinstance(requested, dict)
        or runtime.get("requested_device") != requested.get("device")
        or not isinstance(runtime.get("model_parameter_devices"), list)
        or not runtime.get("model_parameter_devices")
        or not _canonical_cuda_device_string(runtime.get("resolved_requested_device"))
        or not _cuda_device_matches_resolved(
            runtime.get("requested_device"), runtime.get("resolved_requested_device"),
        )
        or not _cuda_device_matches_resolved(
            runtime.get("evaluator_device"), runtime.get("resolved_evaluator_device"),
        )
        or any(
            not _cuda_device_matches_resolved(
                device, runtime.get("resolved_requested_device"),
            )
            for device in runtime.get("model_parameter_devices", [])
        )
        or runtime.get("resolved_evaluator_device")
        != runtime.get("resolved_requested_device")
        or not isinstance(runtime.get("resolved_model_parameter_devices"), list)
        or not runtime.get("resolved_model_parameter_devices")
        or any(
            device != runtime.get("resolved_requested_device")
            for device in runtime.get("resolved_model_parameter_devices", [])
        )
        or not isinstance(runtime.get("cuda_device_name"), str)
        or not runtime.get("cuda_device_name")
        or not isinstance(runtime.get("cuda_device_capability"), list)
        or len(runtime.get("cuda_device_capability", [])) != 2
        or not _positive_int(runtime.get("cuda_device_capability", [None])[0])
        or not _nonnegative_int(runtime.get("cuda_device_capability", [None, None])[1])
    ):
        failures.append("CUDA runtime/device provenance is incomplete")
    realized_tablebase = manifest.get("realized_tablebase")
    root_tb_control = (
        realized_tablebase.get("root_shortcut_positive_control")
        if isinstance(realized_tablebase, dict) else None
    )
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
        or realized_tablebase.get("positive_control") != {
            "fen": "7k/8/8/8/8/8/8/KQ6 w - - 0 1",
            "probes": 1,
            "hits": 1,
            "apply_return": 1,
            "passed": True,
        }
        or not _valid_root_tb_control(root_tb_control)
    ):
        failures.append("production Syzygy probe was not realized")
    row_count = manifest.get("row_count")
    chunk_count = manifest.get("chunk_count")
    position_count = manifest.get("position_count")
    requested_positions = manifest.get("requested_position_count")
    requested_max_positions = manifest.get("requested_max_positions")
    excluded_positions = manifest.get("excluded_position_count")
    excluded_details = manifest.get("excluded_positions")
    counts_are_ints = (
        _positive_int(row_count)
        and _positive_int(chunk_count)
        and _positive_int(position_count)
        and _positive_int(requested_positions)
        and _positive_int(requested_max_positions)
        and _nonnegative_int(excluded_positions)
    )
    if not counts_are_ints:
        failures.append("row/position/chunk accounting is inconsistent")
    else:
        assert isinstance(row_count, int)
        assert isinstance(chunk_count, int)
        assert isinstance(position_count, int)
        assert isinstance(requested_positions, int)
        assert isinstance(requested_max_positions, int)
        assert isinstance(excluded_positions, int)
        if (
            row_count != chunk_count * position_count
            or requested_positions != position_count + excluded_positions
            or requested_positions > requested_max_positions
            or manifest.get("incomplete_exclusion_count") != 0
            or not isinstance(excluded_details, list)
            or len(excluded_details) != excluded_positions
            or any(
                not isinstance(entry, dict)
                or not isinstance(entry.get("key"), str)
                or not entry.get("key")
                or entry.get("chunks_observed") != 0
                or entry.get("chunks_required") != chunk_count
                or entry.get("reason") != "production_terminal_shortcut"
                or not isinstance(entry.get("search_result"), dict)
                or not _nonnegative_int(entry["search_result"].get("nodes"))
                or int(entry["search_result"].get("nodes", 2)) > 1
                or not _nonnegative_int(entry["search_result"].get("tbhits"))
                or entry["search_result"].get("root_declined") is not None
                or (
                    entry["search_result"].get("score_mate") is None
                    and int(entry["search_result"].get("tbhits", 0)) <= 0
                    and entry["search_result"].get("board_game_over") is not True
                )
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
    return manifest, decision_grade, not preregistration_failures


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
    first = trajectory[0]
    invariant_fields = (
        "key", "fen", "source_dir", "shard", "game_id", "group_id",
        "phase", "source", "piece_count", "legal_move_count",
        "deep_reference_best_cp", "deep_reference_move_cp",
    )
    for index, row in enumerate(trajectory):
        key = str(row.get("key"))
        if not methodology_smoke:
            changed = [
                field for field in invariant_fields
                if row.get(field) != first.get(field)
            ]
            if changed:
                raise ValueError(
                    f"{key}: trajectory-invariant fields change between chunks: "
                    f"{', '.join(changed)}"
                )
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
            best_cp = float(row["deep_reference_best_cp"])
            reference_cp = np.asarray(
                row["root_action_reference_cp"], dtype=np.float64,
            )
            expected_action_regret = np.clip(
                best_cp - reference_cp, 0.0, AUDIT_REGRET_CAP_CP,
            )
            if not np.allclose(
                np.asarray(row["root_action_regret_cp"], dtype=np.float64),
                expected_action_regret,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    f"{key}: capped regrets disagree with raw reference CP values"
                )
            expected_regret_cp = float(expected_action_regret[action_index])
            if not math.isclose(
                float(row["regret_cp"]), expected_regret_cp,
                rel_tol=1e-10, abs_tol=1e-12,
            ):
                raise ValueError(f"{key}: regret_cp disagrees with raw reference regrets")
            expected_regret_score = (
                _score(best_cp) - _score(float(reference_cp[action_index]))
            )
            if not math.isclose(
                float(row["regret_score"]), expected_regret_score,
                rel_tol=1e-10, abs_tol=1e-12,
            ):
                raise ValueError(f"{key}: regret_score disagrees with raw reference values")
            child_q = [float(value) for value in row["root_child_q"]]
            child_q_observed = [bool(value) for value in row["root_child_q_observed"]]
            expected_q_gap = None
            if child_q_observed[action_index]:
                other_q = [
                    value for i, value in enumerate(child_q)
                    if i != action_index and child_q_observed[i]
                ]
            else:
                other_q = []
            if other_q:
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
            if previous is not None:
                if set(row["root_actions"]) != set(previous["root_actions"]):
                    raise ValueError(f"{key}: root action support changes within trajectory")
                previous_visits = dict(zip(
                    previous["root_actions"], previous["root_visits"], strict=True,
                ))
                current_visits = dict(zip(
                    row["root_actions"], row["root_visits"], strict=True,
                ))
                if any(
                    int(current_visits[action]) < int(previous_visits[action])
                    for action in current_visits
                ):
                    raise ValueError(f"{key}: root visits decrease within trajectory")
                if (
                    int(row["tb_probes"]) < int(previous["tb_probes"])
                    or int(row["tb_hits"]) < int(previous["tb_hits"])
                ):
                    raise ValueError(
                        f"{key}: tablebase counters decrease within trajectory"
                    )
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
    # Authenticate and parse the same immutable byte buffers. Reading the
    # paths independently for hashing and parsing leaves a replacement window
    # in which the manifest can attest to bytes other than those analyzed.
    input_bytes = input_path.read_bytes()
    meta_bytes = actual_meta.read_bytes() if actual_meta.is_file() else None
    manifest, decision_grade, preregistered_design = _require_manifest(
        input_path,
        actual_meta,
        methodology_smoke,
        input_bytes=input_bytes,
        meta_bytes=meta_bytes,
    )
    requested_search = manifest.get("requested_search", {})
    require_full_root_support = (
        not methodology_smoke
        and isinstance(requested_search, dict)
        and requested_search.get("active_path") == "walker_puct"
    )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(input_bytes.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not methodology_smoke and row.get("schema") != _SCHEMA:
            raise ValueError(f"line {line_number}: missing {_SCHEMA} row schema")
        if not methodology_smoke:
            _validate_decision_grade_row(
                row,
                line_number,
                require_full_root_support=require_full_root_support,
            )
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
                "q_drift_missing": float(lower.get("q_drift") is None),
                "visit_churn": _finite(lower.get("visit_churn")),
                "visit_churn_missing": float(lower.get("visit_churn") is None),
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
                hard_horizon=int(trajectory[-1]["nodes"]),
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
        "preregistered_design": preregistered_design and not methodology_smoke,
        "methodology_smoke": methodology_smoke,
        "metric": metric,
        "analysis_scope": "fresh_tree_fixed_node_horizons_only",
        "clock_conditioning_tested": False,
        "cross_move_tree_reuse_tested": False,
        "manifest": manifest,
    }
    return transitions, info


_M0_FEATURES = (
    "visit_gap", "visit_entropy", "q_gap", "q_gap_missing", "bestmove_flip",
    "stable_chunks", "q_drift", "q_drift_missing", "visit_churn",
    "visit_churn_missing", "root_q", "piece_count", "legal_move_count",
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
            remaining_fraction = (
                transition.hard_horizon - state["nodes"]
            ) / transition.hard_horizon
            context = [log_nodes, remaining_fraction]
            interactions = [
                value * remaining_fraction for value in (
                    state["visit_gap"], state["visit_entropy"], state["bestmove_flip"],
                    state["q_drift"], state["visit_churn"],
                )
            ]
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


def _rollout_layout(
    transitions: Sequence[Transition],
) -> tuple[list[int], list[str], dict[int, np.ndarray]]:
    """Return the complete rectangular key-by-horizon trajectory layout."""
    horizons = sorted({row.horizon for row in transitions})
    keys = sorted({row.key for row in transitions})
    if not horizons or not keys:
        raise ValueError("reachable rollout requires at least one trajectory stage")
    by_horizon: dict[int, np.ndarray] = {}
    for horizon in horizons:
        indices = np.flatnonzero([row.horizon == horizon for row in transitions])
        stage_keys = [transitions[int(index)].key for index in indices]
        if len(indices) != len(keys) or len(set(stage_keys)) != len(keys):
            raise ValueError("reachable rollout requires one row per key at every horizon")
        if set(stage_keys) != set(keys):
            raise ValueError("reachable rollout horizons contain different position keys")
        by_horizon[horizon] = indices
    return horizons, keys, by_horizon


def _stage_counts(n_positions: int, n_stages: int, fraction: float) -> list[int]:
    """Geometric, nested tranche counts with exact rounded total spend."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("allocation_fraction must lie in [0, 1]")
    target = round(fraction * n_positions * n_stages)
    if target <= 0:
        return [0] * n_stages
    if target >= n_positions * n_stages:
        return [n_positions] * n_stages
    low, high = 0.0, 1.0
    for _ in range(80):
        rate = (low + high) / 2.0
        total = sum(n_positions * rate ** (stage + 1) for stage in range(n_stages))
        if total < target:
            low = rate
        else:
            high = rate
    rate = (low + high) / 2.0
    ideals = [n_positions * rate ** (stage + 1) for stage in range(n_stages)]
    counts = [math.floor(value) for value in ideals]
    while sum(counts) < target:
        candidates = [
            stage for stage in range(n_stages)
            if counts[stage] < n_positions
            and (stage == 0 or counts[stage] < counts[stage - 1])
        ]
        if not candidates:
            raise AssertionError("could not realize nested allocation schedule")
        stage = max(candidates, key=lambda index: (ideals[index] - counts[index], -index))
        counts[stage] += 1
    if any(later > earlier for earlier, later in pairwise(counts)):
        raise AssertionError("allocation schedule is not reachable")
    return counts


def _rollout_selected_indices(
    transitions: Sequence[Transition],
    scores: np.ndarray,
    stage_counts: Sequence[int],
    *,
    complexity: bool = False,
) -> np.ndarray:
    """Select nested prefixes; a stopped key can never re-enter later."""
    horizons, keys, by_horizon = _rollout_layout(transitions)
    if len(stage_counts) != len(horizons) or len(scores) != len(transitions):
        raise ValueError("rollout scores/counts do not match the trajectory bank")
    eligible = set(keys)
    selected: list[int] = []
    for stage, horizon in enumerate(horizons):
        indices = np.asarray([
            int(index) for index in by_horizon[horizon]
            if transitions[int(index)].key in eligible
        ], dtype=np.int64)
        count = int(stage_counts[stage])
        if count > len(indices):
            raise ValueError("stage allocation exceeds the reachable position set")
        stage_rows = [transitions[int(index)] for index in indices]
        stage_keys = [row.key for row in stage_rows]
        if complexity:
            local = _complexity_predicate_indices(stage_rows, stage_keys, count)
        else:
            local = _top_indices(scores[indices], stage_keys, count)
        chosen = indices[local]
        selected.extend(int(index) for index in chosen)
        eligible = {transitions[int(index)].key for index in chosen}
    return np.asarray(selected, dtype=np.int64)


def _reachable_oracle_selected_indices(
    transitions: Sequence[Transition], stage_counts: Sequence[int],
) -> tuple[np.ndarray, float]:
    """Return the exact hindsight policy under nested stop-depth capacities."""
    horizons, keys, by_horizon = _rollout_layout(transitions)
    if len(stage_counts) != len(horizons):
        raise ValueError("oracle stage counts do not match the trajectory bank")
    index_by_stage_key = {
        (stage, transitions[int(index)].key): int(index)
        for stage, horizon in enumerate(horizons)
        for index in by_horizon[horizon]
    }
    gains = [
        [transitions[index_by_stage_key[(stage, key)]].gain for stage in range(len(horizons))]
        for key in keys
    ]
    solution = solve_reachable_oracle(keys, gains, stage_counts)
    selected = np.asarray([
        index_by_stage_key[(stage, key)]
        for stage, stage_keys in enumerate(solution.selected_keys_by_stage)
        for key in stage_keys
    ], dtype=np.int64)
    realized = math.fsum(transitions[int(index)].gain for index in selected)
    if not math.isclose(realized, solution.objective, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("reachable oracle selection disagrees with its objective")
    return selected, float(solution.objective)


def _rollout_policy_result(
    transitions: Sequence[Transition],
    selected: np.ndarray,
    *,
    random_gain: float,
    oracle_gain: float,
) -> PolicyResult:
    horizons, keys, _ = _rollout_layout(transitions)
    first_horizon = horizons[0]
    final_regret = {
        row.key: row.regret_before for row in transitions if row.horizon == first_horizon
    }
    selected_set = {int(index) for index in selected}
    for index, row in sorted(
        enumerate(transitions), key=lambda item: (item[1].horizon, item[1].key),
    ):
        if index in selected_set:
            final_regret[row.key] = row.regret_after
    regrets = np.asarray([final_regret[key] for key in keys], dtype=np.float64)
    signed_gain = float(sum(transitions[index].gain for index in selected_set))
    return PolicyResult(
        selected=len(selected_set),
        spend=int(sum(transitions[index].cost for index in selected_set)),
        signed_gain=signed_gain,
        capture_over_random=_capture(signed_gain, random_gain, oracle_gain),
        regret_mean=float(regrets.mean()),
        regret_p95=float(np.quantile(regrets, 0.95)),
        regret_p99=float(np.quantile(regrets, 0.99)),
    )


def _reachable_stage_diagnostics(
    transitions: Sequence[Transition],
    horizons: Sequence[int],
    stage_counts: Sequence[int],
    oracle_selected: np.ndarray,
    m0_selected: np.ndarray,
    m1_selected: np.ndarray,
    *,
    n_positions: int,
) -> list[dict[str, Any]]:
    """Compare signed gains at each decision rung under reachable policies."""
    oracle_set = {int(index) for index in oracle_selected}
    m0_set = {int(index) for index in m0_selected}
    m1_set = {int(index) for index in m1_selected}
    diagnostics: list[dict[str, Any]] = []
    for horizon, count in zip(horizons, stage_counts, strict=True):
        stage_indices = [
            index for index, row in enumerate(transitions) if row.horizon == horizon
        ]
        m0_gain = math.fsum(
            transitions[index].gain for index in stage_indices if index in m0_set
        )
        m1_gain = math.fsum(
            transitions[index].gain for index in stage_indices if index in m1_set
        )
        oracle_gain = math.fsum(
            transitions[index].gain for index in stage_indices if index in oracle_set
        )
        random_gain = (
            int(count) / n_positions
            * math.fsum(transitions[index].gain for index in stage_indices)
        )
        headroom = (oracle_gain - random_gain) / n_positions
        diagnostics.append({
            "horizon": horizon,
            "selected": int(count),
            "M0_signed_gain": m0_gain,
            "M1_signed_gain": m1_gain,
            "random_expected_signed_gain": random_gain,
            "oracle_signed_gain": oracle_gain,
            "oracle_over_random_headroom_mean": headroom,
            "M1_minus_M0_signed_gain_mean": (m1_gain - m0_gain) / n_positions,
            "eligible": int(count) > 0 and headroom > 0.0,
        })
    return diagnostics


def evaluate_reachable_rollout(
    transitions: Sequence[Transition],
    m0_scores: np.ndarray,
    m1_scores: np.ndarray,
    *,
    allocation_fraction: float,
) -> dict[str, Any]:
    """Evaluate policies as nested stop/continue trajectories at matched spend."""
    horizons, keys, by_horizon = _rollout_layout(transitions)
    counts = _stage_counts(len(keys), len(horizons), allocation_fraction)
    random_gain = 0.0
    relaxed_oracle_gain = 0.0
    for count, horizon in zip(counts, horizons, strict=True):
        indices = by_horizon[horizon]
        gains = np.asarray([transitions[int(index)].gain for index in indices])
        random_gain += float(count / len(keys) * gains.sum())
        relaxed_oracle_gain += float(
            gains[_top_indices(gains, [transitions[int(i)].key for i in indices], count)].sum()
        )
    oracle_selected, oracle_gain = _reachable_oracle_selected_indices(
        transitions, counts,
    )
    m0_selected = _rollout_selected_indices(transitions, m0_scores, counts)
    m1_selected = _rollout_selected_indices(transitions, m1_scores, counts)
    complexity_selected = _rollout_selected_indices(
        transitions, np.zeros(len(transitions)), counts, complexity=True,
    )
    reachable_stages = _reachable_stage_diagnostics(
        transitions, horizons, counts, oracle_selected, m0_selected, m1_selected,
        n_positions=len(keys),
    )
    policies = {
        "random": {
            "selected": sum(counts),
            "spend": int(sum(
                count * transitions[int(by_horizon[horizon][0])].cost
                for count, horizon in zip(counts, horizons, strict=True)
            )),
            "signed_gain": random_gain,
            "capture_over_random": (
                0.0 if oracle_gain > random_gain else None
            ),
        },
        "oracle": asdict(_rollout_policy_result(
            transitions, oracle_selected,
            random_gain=random_gain, oracle_gain=oracle_gain,
        )),
        "complexity_predicate": asdict(_rollout_policy_result(
            transitions, complexity_selected,
            random_gain=random_gain, oracle_gain=oracle_gain,
        )),
        "M0": asdict(_rollout_policy_result(
            transitions, m0_selected,
            random_gain=random_gain, oracle_gain=oracle_gain,
        )),
        "M1": asdict(_rollout_policy_result(
            transitions, m1_selected,
            random_gain=random_gain, oracle_gain=oracle_gain,
        )),
    }
    return {
        "n_positions": len(keys),
        "n_stages": len(horizons),
        "horizons": horizons,
        "stage_continue_counts": counts,
        "target_allocation_fraction": allocation_fraction,
        "realized_allocation_fraction": sum(counts) / (len(keys) * len(horizons)),
        "selection_semantics": "nested_prefix_no_reentry",
        "oracle_semantics": "exact_nested_stop_depth_assignment",
        "reachable_oracle_signed_gain": oracle_gain,
        "relaxed_oracle_signed_gain": relaxed_oracle_gain,
        "relaxation_gap": relaxed_oracle_gain - oracle_gain,
        "oracle_over_random_headroom_mean": (
            oracle_gain - random_gain
        ) / len(keys),
        "reachable_stage_diagnostics": reachable_stages,
        "policies": policies,
    }


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


def _minimum_reachable_rung_gain_delta(
    transitions: Sequence[Transition], m0: np.ndarray, m1: np.ndarray,
    allocation_fraction: float, min_oracle_headroom: float = 0.0,
) -> float | None:
    result = evaluate_reachable_rollout(
        transitions, m0, m1, allocation_fraction=allocation_fraction,
    )
    headroom = float(result["oracle_over_random_headroom_mean"])
    stages = result["reachable_stage_diagnostics"]
    if (
        headroom < min_oracle_headroom
        or not stages
        or any(
            not stage["eligible"]
            or float(stage["oracle_over_random_headroom_mean"])
            < min_oracle_headroom
            for stage in stages
        )
    ):
        return None
    return min(float(stage["M1_minus_M0_signed_gain_mean"]) for stage in stages)


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
    """Bootstrap the worst reachable rung's signed M1-minus-M0 gain."""
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
        delta = _minimum_reachable_rung_gain_delta(
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
        delta = (
            float(c1) - float(c0)
            if c0 is not None and c1 is not None else None
        )
        result["diagnostic_only"] = True
        result["eligible_headroom"] = (
            c0 is not None and c1 is not None and headroom >= min_oracle_headroom
        )
        result["M1_minus_M0_oracle_capture"] = delta
        horizons[str(horizon)] = result
    rollout = evaluate_reachable_rollout(
        transitions, m0, m1, allocation_fraction=allocation_fraction,
    )
    capture_m0 = rollout["policies"]["M0"]["capture_over_random"]
    capture_m1 = rollout["policies"]["M1"]["capture_over_random"]
    capture_gain = (
        float(capture_m1) - float(capture_m0)
        if capture_m0 is not None and capture_m1 is not None else None
    )
    rollout_headroom = float(rollout["oracle_over_random_headroom_mean"])
    rollout_eligible = (
        capture_m0 is not None and capture_m1 is not None
        and rollout_headroom >= min_oracle_headroom
    )
    tail_ok = bool(
        rollout["policies"]["M1"]["regret_p95"]
        <= rollout["policies"]["M0"]["regret_p95"] + 1e-12
        and rollout["policies"]["M1"]["regret_p99"]
        <= rollout["policies"]["M0"]["regret_p99"] + 1e-12
    )
    reachable_stage_rule_passed = bool(
        rollout["reachable_stage_diagnostics"]
        and all(
            stage["eligible"]
            and float(stage["oracle_over_random_headroom_mean"])
            >= min_oracle_headroom
            and float(stage["M1_minus_M0_signed_gain_mean"]) > 0.0
            for stage in rollout["reachable_stage_diagnostics"]
        )
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
        and capture_m1 is not None and float(capture_m1) > 0.0
        and rollout_eligible
        and reachable_stage_rule_passed
        and bootstrap_resolution_ok
        and float(bootstrap["valid_fraction"] or 0.0) >= min_bootstrap_valid_fraction
        and bootstrap["lower_95"] is not None and float(bootstrap["lower_95"]) > 0.0
        and tail_ok
    )
    return {
        "evaluated_rule": {
            "grouped_cv_folds": n_folds,
            "bootstrap_seed": seed,
            "allocation_fraction": allocation_fraction,
            "minimum_M1_minus_M0_oracle_capture": min_capture_gain,
            "minimum_oracle_over_random_headroom_mean": min_oracle_headroom,
            "reachable_selection_required": "nested_prefix_no_reentry",
            "per_horizon_tables_are_diagnostic_only": True,
            "M1_reachable_rollout_capture_above_random": True,
            "M1_minus_M0_reachable_signed_gain_positive_at_every_rung": True,
            "refitted_OOB_game_cluster_bootstrap_worst_rung_lower_95_above_zero": True,
            "bootstrap_samples": bootstrap_samples,
            "minimum_decision_grade_bootstrap_samples": (
                _MIN_DECISION_GRADE_BOOTSTRAP_SAMPLES
            ),
            "minimum_bootstrap_valid_fraction": min_bootstrap_valid_fraction,
            "M1_p95_and_p99_regret_not_worse_than_M0": True,
        },
        "statistical_gate_passed": advance,
        "scope": "fresh_tree_fixed_node_horizons_only",
        "clock_controller_authorized": False,
        "cross_move_tree_reuse_tested": False,
        "reachable_rollout_capture_M0": capture_m0,
        "reachable_rollout_capture_M1": capture_m1,
        "M1_minus_M0_oracle_capture": capture_gain,
        "reachable_rollout_eligible": rollout_eligible,
        "reachable_stage_rule_passed": reachable_stage_rule_passed,
        "tail_rule_passed": tail_ok,
        "bootstrap_resolution_passed": bootstrap_resolution_ok,
        "bootstrap_refit_oob_worst_reachable_rung_M1_minus_M0_signed_gain": bootstrap,
        "reachable_rollout": rollout,
        "stage_conditional_diagnostics": horizons,
        "cv_diagnostics": {"M0": m0_diagnostics, "M1": m1_diagnostics},
    }


def _evidence_verdict(
    *, evidence_inputs_decision_grade: bool, canonical_preregistered_rule: bool,
    statistical_gate_passed: bool,
) -> str:
    """Name only the next experiment authorized by this deliberately narrow bank."""
    if not evidence_inputs_decision_grade:
        return "METHODOLOGY_SMOKE_ONLY"
    if not canonical_preregistered_rule:
        return "NONCANONICAL_RULE_DIAGNOSTIC_ONLY"
    if statistical_gate_passed:
        return "ADVANCE_TO_CLOCK_HISTORY_REUSED_TREE_BANK"
    return "NO_ADVANCE_FROM_FRESH_TREE_FIXED_NODE_SCREEN"


def main() -> None:
    parser = argparse.ArgumentParser(prog="analyze_chunk_controller")
    parser.add_argument("--in", dest="input_path", type=Path, required=True)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--methodology-smoke", action="store_true")
    parser.add_argument("--folds", type=int, default=_CANONICAL_FOLDS)
    parser.add_argument(
        "--bootstrap-samples", type=int, default=_CANONICAL_BOOTSTRAP_SAMPLES,
    )
    parser.add_argument("--seed", type=int, default=_CANONICAL_SEED)
    parser.add_argument(
        "--allocation-fraction", type=float,
        default=_CANONICAL_ALLOCATION_FRACTION,
    )
    parser.add_argument(
        "--min-capture-gain", type=float, default=_CANONICAL_MIN_CAPTURE_GAIN,
        help="minimum M1-minus-M0 fraction of oracle-over-random headroom",
    )
    parser.add_argument(
        "--min-oracle-headroom", type=float, default=_CANONICAL_MIN_ORACLE_HEADROOM,
        help="minimum per-row oracle-over-random regret headroom at every horizon",
    )
    parser.add_argument(
        "--min-bootstrap-valid-fraction", type=float,
        default=_CANONICAL_MIN_BOOTSTRAP_VALID_FRACTION,
        help="minimum fraction of cluster replicates with eligible headroom",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    analyzer_start_artifact = _artifact_snapshot(Path(__file__))
    analyzer_start_git_sha, analyzer_start_git_dirty = _git_state()
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
    manifest = info.get("manifest")
    _require_safe_output_path(
        args.input_path,
        actual_meta,
        args.out,
        manifest=manifest if isinstance(manifest, dict) else None,
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
    analyzer = _analyzer_provenance(
        analyzer_start_artifact, analyzer_start_git_sha, analyzer_start_git_dirty,
    )
    manifest_producer_sha = (
        manifest.get("producer_git_sha") if isinstance(manifest, dict) else None
    )
    analyzer_matches_producer_commit = bool(
        analyzer["git_sha"] == manifest_producer_sha
        and analyzer["final_git_sha"] == manifest_producer_sha
    )
    evidence_inputs_decision_grade = bool(
        info["decision_grade"]
        and analyzer["decision_grade"]
        and analyzer_matches_producer_commit
    )
    canonical_analysis_rule = _is_canonical_decision_rule(
        n_folds=args.folds,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        allocation_fraction=args.allocation_fraction,
        min_capture_gain=args.min_capture_gain,
        min_oracle_headroom=args.min_oracle_headroom,
        min_bootstrap_valid_fraction=args.min_bootstrap_valid_fraction,
    )
    canonical_rule = bool(canonical_analysis_rule and info["preregistered_design"])
    decision_grade = evidence_inputs_decision_grade and canonical_rule
    result["canonical_analysis_rule"] = canonical_analysis_rule
    result["canonical_preregistered_rule"] = canonical_rule
    result["analyzer_matches_producer_commit"] = analyzer_matches_producer_commit
    result["evidence_decision_grade"] = decision_grade
    result["verdict"] = _evidence_verdict(
        evidence_inputs_decision_grade=evidence_inputs_decision_grade,
        canonical_preregistered_rule=canonical_rule,
        statistical_gate_passed=bool(result["statistical_gate_passed"]),
    )
    payload = {"input": info, "analyzer": analyzer, "analysis": result}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        _require_safe_output_path(
            args.input_path,
            actual_meta,
            args.out,
            manifest=manifest if isinstance(manifest, dict) else None,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(args.out, rendered)
    print(rendered)


if __name__ == "__main__":
    main()
