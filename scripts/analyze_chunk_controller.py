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

Allocation ranks those predictions only within the outer fold that produced
them.  Deterministic nested fold quotas sum to the preregistered global spend;
random and exact reachable-oracle comparisons use the same quotas.

This tool deliberately tests fixed NODE horizons only.  Trajectory banks do
not contain real game clocks, soft budgets, or time-forfeit observations, so a
positive result can justify collecting a clock bank but cannot justify a
clock-conditioned production controller.

Decision-grade provenance requires direct execution as
``PYTHONPATH=. python3 scripts/analyze_chunk_controller.py``. Module-mode execution
preloads ``scripts/__init__.py`` before the entrypoint can snapshot project sources.
"""
# ruff: noqa: E402  # Provenance snapshot must precede non-stdlib imports.
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import stat
import subprocess
import sys
import types
from collections import defaultdict
from collections.abc import Collection, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any


_PYTHON_PREIMPORT_SCHEMA = "deepfin.python_preimport.v1"


def _preimport_python_file_artifact(
    path: Path, *, expected_oid: str, object_format: str,
) -> dict[str, Any]:
    """Read one source without accepting a change during the read."""
    resolved = path.resolve()
    try:
        before = resolved.lstat()
        content = resolved.read_bytes()
        after = resolved.lstat()
    except OSError as exc:
        return {
            "path": str(resolved),
            "git_blob_oid": expected_oid,
            "error": f"{type(exc).__name__}: {exc}",
            "stable_read": False,
            "matches_git_revision": False,
        }
    identity = (
        int(before.st_mode), int(before.st_size), int(before.st_mtime_ns),
        int(before.st_ctime_ns), int(before.st_dev), int(before.st_ino),
    )
    stable_read = bool(
        identity == (
            int(after.st_mode), int(after.st_size), int(after.st_mtime_ns),
            int(after.st_ctime_ns), int(after.st_dev), int(after.st_ino),
        )
        and stat.S_ISREG(before.st_mode)
        and len(content) == before.st_size
    )
    blob = f"blob {len(content)}\0".encode("ascii") + content
    observed_oid = hashlib.new(object_format, blob).hexdigest()
    return {
        "path": str(resolved),
        "size": len(content),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "sha256": hashlib.sha256(content).hexdigest(),
        "git_blob_oid": expected_oid,
        "observed_git_blob_oid": observed_oid,
        "stable_read": stable_read,
        "matches_git_revision": bool(stable_read and observed_oid == expected_oid),
    }


def _preimport_python_source_snapshot() -> dict[str, Any]:
    """Snapshot tracked Python before any project or third-party import.

    The executing entry script and the Python process itself are the trust boundary.
    This proof detects concurrent checkout mutation; it does not claim to defend
    against a hostile process that can replace this already-executing guard.
    """
    repo_root = Path(__file__).resolve().parents[1]
    preexisting = sorted(
        name for name in sys.modules
        if name in ("chess_anti_engine", "scripts")
        or name.startswith(("chess_anti_engine.", "scripts."))
    )
    try:
        git_sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        object_format = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--show-object-format"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        tree = subprocess.check_output(
            ["git", "-C", str(repo_root), "ls-tree", "-r", "-z", git_sha],
            stderr=subprocess.DEVNULL,
        )
        final_git_sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "schema": _PYTHON_PREIMPORT_SCHEMA,
            "git_sha": "unknown",
            "repo_root": str(repo_root),
            "entrypoint": str(Path(__file__).resolve()),
            "trust_boundary": "already_executing_entry_script_and_python_process",
            "preexisting_project_modules": preexisting,
            "files": {},
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if object_format not in hashlib.algorithms_available:
        return {
            "schema": _PYTHON_PREIMPORT_SCHEMA,
            "git_sha": git_sha,
            "repo_root": str(repo_root),
            "entrypoint": str(Path(__file__).resolve()),
            "trust_boundary": "already_executing_entry_script_and_python_process",
            "preexisting_project_modules": preexisting,
            "files": {},
            "passed": False,
            "error": f"unsupported Git object format: {object_format}",
        }
    files: dict[str, dict[str, Any]] = {}
    for raw_entry in tree.split(b"\0"):
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        relative_path = raw_path.decode("utf-8", errors="surrogateescape")
        if not relative_path.endswith(".py"):
            continue
        artifact = _preimport_python_file_artifact(
            repo_root / relative_path,
            expected_oid=oid,
            object_format=object_format,
        )
        artifact["git_mode"] = mode
        artifact["git_kind"] = kind
        files[relative_path] = artifact
    surface_rows = [
        [relative_path, artifact.get("git_blob_oid"), artifact.get("sha256")]
        for relative_path, artifact in sorted(files.items())
    ]
    surface_digest = hashlib.sha256(json.dumps(
        surface_rows, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()
    source_tree_matches_revision = bool(files) and all(
        artifact.get("matches_git_revision") is True for artifact in files.values()
    )
    return {
        "schema": _PYTHON_PREIMPORT_SCHEMA,
        "git_sha": git_sha,
        "final_git_sha": final_git_sha,
        "git_object_format": object_format,
        "repo_root": str(repo_root),
        "entrypoint": str(Path(__file__).resolve()),
        "snapshot_stage": "before_project_or_third_party_imports",
        "trust_boundary": "already_executing_entry_script_and_python_process",
        "preexisting_project_modules": preexisting,
        "tracked_python_file_count": len(files),
        "tracked_python_surface_sha256": surface_digest,
        "source_tree_matches_revision": source_tree_matches_revision,
        "files": files,
        "passed": bool(
            source_tree_matches_revision
            and final_git_sha == git_sha
            and not preexisting
        ),
    }


_PREIMPORT_PYTHON_SOURCES = _preimport_python_source_snapshot()


def _install_authenticated_source_only_import(
    snapshot: Mapping[str, Any],
) -> Any:
    """Compile the import guard's authenticated source, never its cached bytecode."""
    module_name = "scripts.source_only_import"
    relative_path = "scripts/source_only_import.py"
    expected_files = snapshot.get("files")
    expected = (
        expected_files.get(relative_path)
        if isinstance(expected_files, dict) else None
    )
    if not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str):
        raise ImportError("source-only import guard is absent from pre-import snapshot")
    existing = sys.modules.get(module_name)
    if existing is None:
        source_path = Path(str(snapshot["repo_root"])) / relative_path
        source = source_path.read_bytes()
        digest = hashlib.sha256(source).hexdigest()
        if digest != expected["sha256"]:
            raise ImportError("source-only import guard changed after pre-import snapshot")
        module = types.ModuleType(module_name)
        module.__file__ = str(source_path.resolve())
        module.__package__ = "scripts"
        sys.modules[module_name] = module
        try:
            exec(
                compile(source, str(source_path), "exec", dont_inherit=True),
                module.__dict__,
            )
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
        setattr(module, "BOOTSTRAP_SOURCE_SHA256", digest)
    else:
        module = existing
        if getattr(module, "BOOTSTRAP_SOURCE_SHA256", None) != expected["sha256"]:
            raise ImportError("existing source-only import guard is not authenticated")
    return module.install(snapshot)


_SOURCE_ONLY_IMPORT_GUARD = _install_authenticated_source_only_import(
    _PREIMPORT_PYTHON_SOURCES
)


def _preimport_python_surface_status(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-read the exact pre-import surface and report any identity/content drift."""
    files = snapshot.get("files")
    object_format = snapshot.get("git_object_format")
    git_sha = snapshot.get("git_sha")
    repo_root = Path(str(snapshot.get("repo_root", "")))
    if (
        not isinstance(files, dict)
        or not isinstance(object_format, str)
        or object_format not in hashlib.algorithms_available
        or not isinstance(git_sha, str)
        or not repo_root.is_dir()
    ):
        return {"passed": False, "changed": ["invalid_preimport_snapshot"]}
    changed: list[str] = []
    for relative_path, initial in sorted(files.items()):
        if not isinstance(relative_path, str) or not isinstance(initial, dict):
            changed.append(str(relative_path))
            continue
        expected_oid = initial.get("git_blob_oid")
        if not isinstance(expected_oid, str):
            changed.append(relative_path)
            continue
        current = _preimport_python_file_artifact(
            repo_root / relative_path,
            expected_oid=expected_oid,
            object_format=object_format,
        )
        identity_keys = (
            "path", "size", "mtime_ns", "ctime_ns", "device", "inode",
            "sha256", "git_blob_oid", "observed_git_blob_oid", "stable_read",
            "matches_git_revision",
        )
        if any(current.get(key) != initial.get(key) for key in identity_keys):
            changed.append(relative_path)
    try:
        current_git_sha = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        current_git_sha = "unknown"
    if current_git_sha != git_sha:
        changed.append("analyzer_checkout_revision")
    return {
        "passed": not changed,
        "changed": sorted(set(changed)),
        "git_sha": current_git_sha,
        "tracked_python_file_count": len(files),
        "tracked_python_surface_sha256": snapshot.get(
            "tracked_python_surface_sha256"
        ),
    }

import chess
import numpy as np

from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.eval.audit import (
    AUDIT_REGRET_CAP_CP,
    legal_full_indices,
    phase_bucket,
    position_key,
)
from chess_anti_engine.mcts import search_options as search_options_module
from chess_anti_engine.moves import ActionDecodeError, POLICY_SIZE, index_to_move_strict
from scripts.approved_syzygy import (
    APPROVED_SYZYGY_CHECKSUM_CATALOG_DTZ_COUNT,
    APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRIES_SHA256,
    APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRY_COUNT,
    APPROVED_SYZYGY_CHECKSUM_CATALOG_NAME,
    APPROVED_SYZYGY_CHECKSUM_CATALOG_RAW_SHA256,
    APPROVED_SYZYGY_CHECKSUM_CATALOG_SIZE,
    APPROVED_SYZYGY_CHECKSUM_CATALOG_WDL_COUNT,
    APPROVED_SYZYGY_COMPONENTS,
    APPROVED_SYZYGY_FILENAME_SIZE_ENCODING,
    APPROVED_SYZYGY_LAYOUT_SCHEMA,
    checksum_catalog_entries_sha256,
    filename_size_sha256,
)
from scripts.check_c_extensions_fresh import (
    native_build_attestation,
    native_build_dependency_paths,
)
from scripts.reachable_oracle import solve_reachable_oracle
from scripts.repo_output_guard import repo_controlled_output, reserved_output_path

_SCHEMA = "deepfin.chunk_trajectory.v6"
_CP_TO_SCORE_C = 300.0
_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
_COMPLEXITY_VISIT_GAP = 0.25
_COMPLEXITY_STABLE_CHUNKS = 2
_MIN_DECISION_GRADE_BOOTSTRAP_SAMPLES = 1000
_PRODUCTION_WDL_FILES = 875
_PRODUCTION_DTZ_FILES = 510
_PRODUCTION_TB_COMPONENTS = ((510, 145), (365, 365))
_TABLEBASE_INVENTORY_SCHEMA = "deepfin.syzygy_inventory.v4"
_APPROVED_SYZYGY_COMPONENTS = APPROVED_SYZYGY_COMPONENTS
_TABLEBASE_FILE_IDENTITY_FIELDS = (
    "name", "size", "mtime_ns", "ctime_ns", "device", "inode",
)
_TABLEBASE_PATH_COMPONENT_IDENTITY_FIELDS = (
    "path", "device", "inode", "mtime_ns", "ctime_ns",
)
_PRODUCTION_GSS_HALVING_REV = 3
_PRODUCTION_WALKERS = 2
_MIN_DECISION_GRADE_CHUNKS = 4
# Keep the preregistered source-game floor; the fixed-fold bootstrap below no
# longer makes its validity depend on random OOB retention.
_MIN_DECISION_GRADE_SOURCE_GAMES = 9
_CANONICAL_FOLDS = 5
_CANONICAL_BOOTSTRAP_SAMPLES = 2000
_CANONICAL_SEED = 0
_CANONICAL_ALLOCATION_FRACTION = 0.2
_CANONICAL_MIN_CAPTURE_GAIN = 0.05
_CANONICAL_MIN_ORACLE_HEADROOM = 1e-4
_CANONICAL_MIN_BOOTSTRAP_VALID_FRACTION = 0.95
_FOLD_SELECTION_SEMANTICS = "fold_local_nested_prefix_no_reentry"
_FOLD_ORACLE_SEMANTICS = "exact_fold_local_nested_stop_depth_assignment"
_BOOTSTRAP_RESAMPLING_SEMANTICS = (
    "global_source_game_clusters_with_recomputed_evaluation_folds"
)
_BOOTSTRAP_INTERVAL_SEMANTICS = (
    "unconditional_requested_replicates_with_invalid_mass_in_lower_tail_v1"
)
_PREREGISTRATION_SCHEMA = "deepfin.chunk_controller_preregistration.v3"
_DEEP_REFERENCE_EVIDENCE_SCHEMA = "deepfin.chunk_deep_reference_evidence.v2"
_MODEL_INPUT_CONSUMPTION_SCHEMA = "deepfin.model_input_consumption.v2"
_PARAMS_CANDIDATE_INVENTORY_SCHEMA = "deepfin.params_candidate_inventory.v1"
# data/audit_set_v1.jsonl, whose sibling README freezes the requested
# unhandicapped Stockfish budget at 1,000,000 nodes and MultiPV 10. Some
# forced-mate rows legitimately terminate early, so the artifact identity—not
# a false per-row observed-node floor—is the authority for that request.
_APPROVED_AUDIT_SET_SHA256 = (
    "d8e26efa0b010450abf9374693afc45027db6d146571785ab897af5061144df2"
)
_DEEP_REFERENCE_REQUESTED_NODES = 1_000_000
_DEEP_REFERENCE_MIN_MULTIPV = 10
_DEEP_REFERENCE_COVERAGE = "min(minimum_multipv,legal_move_count)_unique_scored_moves"
_PANEL_SELECTION_STRATEGY = "joint_audit_source_phase_piece_round_robin_v1"
_PANEL_REQUIRED_SOURCES = (0, 1)
_PANEL_SOURCE_BALANCE_MAX_DIFFERENCE = 1
SEARCH_OPTIONS = search_options_module.SEARCH_OPTIONS
_NATIVE_MODULES = [
    "chess_anti_engine.encoding._features_ext",
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


def _trajectory_reference_censoring(
    key: str, rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Describe finite-MultiPV censoring without discarding any observations."""
    listed_rows: list[bool] = []
    unlisted_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        actions = row.get("root_actions", row.get("actions"))
        listed = row.get(
            "root_action_reference_listed", row.get("action_reference_listed"),
        )
        emitted = row.get("emitted_action")
        if (
            not isinstance(actions, list)
            or not isinstance(listed, list)
            or len(actions) != len(listed)
            or any(not isinstance(value, bool) for value in listed)
            or not isinstance(emitted, int)
            or isinstance(emitted, bool)
        ):
            raise ValueError(f"{key}: malformed emitted-reference censoring evidence")
        try:
            emitted_index = actions.index(emitted)
        except ValueError as exc:
            raise ValueError(f"{key}: emitted action is absent from root actions") from exc
        emitted_listed = listed[emitted_index]
        explicit = row.get("emitted_reference_listed")
        if not isinstance(explicit, bool) or explicit is not emitted_listed:
            raise ValueError(f"{key}: emitted-reference censoring flag disagrees with root data")
        listed_rows.append(emitted_listed)
        if not emitted_listed:
            unlisted_rows.append({
                "chunk": int(row.get("chunk", index)),
                "nodes": int(row["nodes"]),
                "emitted_action": emitted,
                "uci": str(row["uci"]),
            })
    censored_transitions = [
        {
            "from_chunk": int(lower.get("chunk", index)),
            "to_chunk": int(upper.get("chunk", index + 1)),
            "horizon_nodes": int(upper["nodes"]),
        }
        for index, (lower, upper) in enumerate(pairwise(rows), start=1)
        if not listed_rows[index - 1] or not listed_rows[index]
    ]
    if not unlisted_rows:
        return None
    return {
        "key": key,
        "unlisted_emitted_rows": unlisted_rows,
        "censored_transitions": censored_transitions,
    }


def _reference_censoring_summary(
    affected_trajectories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return deterministic bank-level diagnostics for censored decision labels."""
    affected = sorted(
        (dict(entry) for entry in affected_trajectories),
        key=lambda entry: str(entry["key"]),
    )
    unlisted_rows = sum(len(entry["unlisted_emitted_rows"]) for entry in affected)
    censored_transitions = sum(len(entry["censored_transitions"]) for entry in affected)
    return {
        "kind": "finite_multipv_unlisted_emitted_move",
        "scope": "completed_trajectory_decision_labels",
        "decision_labels_require_listed_emitted_moves": True,
        "passed": not affected,
        "affected_trajectory_count": len(affected),
        "unlisted_emitted_row_count": unlisted_rows,
        "censored_transition_count": censored_transitions,
        "affected_trajectories": affected,
    }


def _valid_reference_censoring(value: Any) -> bool:
    """Validate the self-contained finite-MultiPV censoring audit trail."""
    if (
        not isinstance(value, dict)
        or value.get("kind") != "finite_multipv_unlisted_emitted_move"
        or value.get("scope") != "completed_trajectory_decision_labels"
        or value.get("decision_labels_require_listed_emitted_moves") is not True
        or not isinstance(value.get("passed"), bool)
        or not _nonnegative_int(value.get("affected_trajectory_count"))
        or not _nonnegative_int(value.get("unlisted_emitted_row_count"))
        or not _nonnegative_int(value.get("censored_transition_count"))
        or not isinstance(value.get("affected_trajectories"), list)
    ):
        return False
    affected = value["affected_trajectories"]
    if (
        value["affected_trajectory_count"] != len(affected)
        or value["passed"] is not (len(affected) == 0)
    ):
        return False
    keys: list[str] = []
    unlisted_count = 0
    transition_count = 0
    for trajectory in affected:
        if (
            not isinstance(trajectory, dict)
            or not isinstance(trajectory.get("key"), str)
            or not trajectory.get("key")
            or not isinstance(trajectory.get("unlisted_emitted_rows"), list)
            or not trajectory["unlisted_emitted_rows"]
            or not isinstance(trajectory.get("censored_transitions"), list)
        ):
            return False
        keys.append(trajectory["key"])
        for row in trajectory["unlisted_emitted_rows"]:
            if (
                not isinstance(row, dict)
                or not _positive_int(row.get("chunk"))
                or not _positive_int(row.get("nodes"))
                or not isinstance(row.get("emitted_action"), int)
                or isinstance(row.get("emitted_action"), bool)
                or not isinstance(row.get("uci"), str)
                or not row.get("uci")
            ):
                return False
        for transition in trajectory["censored_transitions"]:
            if (
                not isinstance(transition, dict)
                or not _positive_int(transition.get("from_chunk"))
                or not _positive_int(transition.get("to_chunk"))
                or not _positive_int(transition.get("horizon_nodes"))
            ):
                return False
            if int(transition["to_chunk"]) != int(transition["from_chunk"]) + 1:
                return False
        unlisted_count += len(trajectory["unlisted_emitted_rows"])
        transition_count += len(trajectory["censored_transitions"])
    return bool(
        len(keys) == len(set(keys))
        and keys == sorted(keys)
        and value["unlisted_emitted_row_count"] == unlisted_count
        and value["censored_transition_count"] == transition_count
    )


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
        "ctime_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "sha256": _sha256(resolved),
    }


def _analyzer_source_artifacts() -> dict[str, dict[str, Any]]:
    """Snapshot every loaded project Python module used by the analyzer."""
    source_paths = {
        "analyzer": Path(__file__),
        "chess_anti_engine.encoding.encode": Path(input_plane_count.__code__.co_filename),
        "chess_anti_engine.eval.audit": Path(phase_bucket.__code__.co_filename),
        "chess_anti_engine.mcts.search_options": Path(
            str(search_options_module.__file__)
        ),
        "chess_anti_engine.moves.encode": Path(index_to_move_strict.__code__.co_filename),
        "scripts.check_c_extensions_fresh": Path(
            native_build_attestation.__code__.co_filename
        ),
        "scripts.reachable_oracle": Path(solve_reachable_oracle.__code__.co_filename),
        "scripts.repo_output_guard": Path(repo_controlled_output.__code__.co_filename),
    }
    unauthenticated_loaded: dict[str, str | None] = {}
    verified_native = _SOURCE_ONLY_IMPORT_GUARD.verified_native_modules
    for module_name, module in sorted(sys.modules.items()):
        if not (
            module_name in ("chess_anti_engine", "scripts")
            or module_name.startswith(("chess_anti_engine.", "scripts."))
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str) and Path(module_file).suffix == ".py":
            source_paths[module_name] = Path(module_file)
        elif module_name not in verified_native:
            unauthenticated_loaded[module_name] = (
                module_file if isinstance(module_file, str) else None
            )
    preimport_files = _PREIMPORT_PYTHON_SOURCES.get("files")
    if not isinstance(preimport_files, dict):
        preimport_files = {}
    artifacts: dict[str, dict[str, Any]] = {}
    repo_root = Path(__file__).resolve().parents[1]
    for name, path in sorted(source_paths.items()):
        artifact = _artifact_snapshot(path)
        try:
            relative_path = Path(artifact["path"]).relative_to(repo_root).as_posix()
        except ValueError:
            relative_path = None
        preimport = (
            preimport_files.get(relative_path)
            if relative_path is not None else None
        )
        artifact["repo_relative_path"] = relative_path
        artifact["matches_preimport_snapshot"] = bool(
            isinstance(preimport, dict)
            and all(
                artifact.get(key) == preimport.get(key)
                for key in (
                    "path", "size", "mtime_ns", "ctime_ns", "device", "inode",
                    "sha256",
                )
            )
        )
        source_record = _SOURCE_ONLY_IMPORT_GUARD.verified_modules.get(name)
        artifact["source_only_import_verified"] = bool(
            name == "analyzer"
            or (
                relative_path is not None
                and _SOURCE_ONLY_IMPORT_GUARD.module_verified(name, relative_path)
            )
        )
        artifact["source_execution"] = (
            "entrypoint_trust_boundary"
            if name == "analyzer"
            else (
                source_record.get("execution")
                if isinstance(source_record, dict) else None
            )
        )
        artifacts[name] = artifact
    for name, module_file in sorted(unauthenticated_loaded.items()):
        artifacts[name] = {
            "path": module_file,
            "repo_relative_path": None,
            "matches_preimport_snapshot": False,
            "source_only_import_verified": False,
            "source_execution": "unauthenticated_loaded_project_module",
        }
    return artifacts


def _source_revision_bindings(
    sources: dict[str, dict[str, Any]], git_sha: str,
) -> dict[str, dict[str, Any]]:
    """Bind loaded project sources to tracked bytes in the reported revision."""
    repo_root = Path(__file__).resolve().parents[1]
    bindings: dict[str, dict[str, Any]] = {}
    for name, artifact in sources.items():
        raw_path = artifact.get("path")
        relative_path: str | None = None
        if isinstance(raw_path, str):
            try:
                relative_path = Path(raw_path).resolve().relative_to(repo_root).as_posix()
            except (OSError, ValueError):
                pass
        committed = (
            _git_file_at_commit(git_sha, relative_path)
            if relative_path is not None else None
        )
        matches_revision = bool(
            committed is not None
            and artifact.get("size") == len(committed)
            and artifact.get("sha256") == hashlib.sha256(committed).hexdigest()
        )
        matches_preimport = artifact.get("matches_preimport_snapshot") is True
        source_only_verified = artifact.get("source_only_import_verified") is True
        bindings[name] = {
            "repo_relative_path": relative_path,
            "matches_reported_git_revision": matches_revision,
            "matches_preimport_snapshot": matches_preimport,
            "source_only_import_verified": source_only_verified,
            "passed": bool(
                matches_revision and matches_preimport and source_only_verified
            ),
        }
    return bindings


def _analyzer_provenance(
    start_sources: dict[str, dict[str, Any]],
    start_git_sha: str,
    start_git_dirty: bool,
    *, preimport_start_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    end_sources = _analyzer_source_artifacts()
    end_git_sha, end_git_dirty = _git_state()
    start_preimport = (
        dict(preimport_start_status)
        if preimport_start_status is not None
        else _preimport_python_surface_status(_PREIMPORT_PYTHON_SOURCES)
    )
    end_preimport = _preimport_python_surface_status(_PREIMPORT_PYTHON_SOURCES)
    source_only_import = _SOURCE_ONLY_IMPORT_GUARD.status()
    source_only_import_passed = bool(
        source_only_import.get("schema") == "deepfin.source_only_import.v2"
        and source_only_import.get("active") is True
        and source_only_import.get("installed") is True
        and source_only_import.get("first_finder") is True
        and source_only_import.get("git_sha") == start_git_sha
        and source_only_import.get("tracked_python_surface_sha256")
        == _PREIMPORT_PYTHON_SOURCES.get("tracked_python_surface_sha256")
        and source_only_import.get("project_scope")
        == ["chess_anti_engine", "scripts"]
        and source_only_import.get("execution")
        == "compile_authenticated_source_bytes"
        and source_only_import.get("bytecode_cache_reads") is False
        and source_only_import.get("native_extension_loading")
        == "default_deny_exact_preimport_artifact_authenticated_loader"
        and source_only_import.get("permitted_native_modules") == _NATIVE_MODULES
        and source_only_import.get("authorized_native_modules") == []
        and source_only_import.get("authorized_native_artifacts") == {}
        and source_only_import.get("verified_native_modules") == {}
        and isinstance(source_only_import.get("loaded_project_modules"), dict)
        and source_only_import["loaded_project_modules"].get("passed") is True
        and source_only_import["loaded_project_modules"].get("unverified_modules")
        == []
        and source_only_import.get("failures") == []
    )
    source_bindings = _source_revision_bindings(start_sources, start_git_sha)
    sources_match_git_revision = bool(source_bindings) and all(
        row["passed"] for row in source_bindings.values()
    )
    stable = (
        _PREIMPORT_PYTHON_SOURCES.get("passed") is True
        and _PREIMPORT_PYTHON_SOURCES.get("git_sha") == start_git_sha
        and start_preimport.get("passed") is True
        and end_preimport.get("passed") is True
        and source_only_import_passed
        and start_sources == end_sources
        and sources_match_git_revision
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
        "script": start_sources["analyzer"],
        "sources": start_sources,
        "sources_stable": start_sources == end_sources,
        "source_revision_bindings": source_bindings,
        "sources_match_git_revision": sources_match_git_revision,
        "python_preimport": {
            **_PREIMPORT_PYTHON_SOURCES,
            "start_check": start_preimport,
            "post_run_check": end_preimport,
            "source_only_import": source_only_import,
        },
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": str(np.__version__),
        "python_chess_version": str(chess.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


@dataclass(frozen=True)
class _AnchoredOutputTarget:
    lexical_path: Path
    parent_fd: int
    parent_identity: tuple[int, int]
    input_path: Path
    meta_path: Path
    manifest: dict[str, Any] | None
    consumed_artifacts: tuple[Mapping[str, Any], ...]

    @property
    def name(self) -> str:
        return self.lexical_path.name

    @property
    def staging_name(self) -> str:
        return f".{self.name}.tmp-{os.getpid()}"


def _directory_identity(stat_result: os.stat_result) -> tuple[int, int]:
    return int(stat_result.st_dev), int(stat_result.st_ino)


_PROC_SELF_FD = Path("/proc/self/fd")
_MAX_OUTPUT_ANCESTOR_DEPTH = 1024


def _strict_descriptor_path(fd: int, *, kind: str) -> Path:
    """Resolve one procfs descriptor link and prove that it names ``fd``."""
    descriptor_link = _PROC_SELF_FD / str(fd)
    try:
        resolved = descriptor_link.resolve(strict=True)
        descriptor_stat = os.stat(descriptor_link)
        resolved_stat = os.stat(resolved)
        fd_stat = os.fstat(fd)
    except OSError as exc:
        raise RuntimeError(
            f"safe analyzer I/O requires an intact procfs descriptor mapping for {kind}"
        ) from exc
    expected_kind = stat.S_ISDIR if kind == "directory" else stat.S_ISREG
    if (
        not expected_kind(fd_stat.st_mode)
        or not expected_kind(descriptor_stat.st_mode)
        or not expected_kind(resolved_stat.st_mode)
        or _directory_identity(descriptor_stat) != _directory_identity(fd_stat)
        or _directory_identity(resolved_stat) != _directory_identity(fd_stat)
    ):
        raise RuntimeError(f"procfs {kind} descriptor mapping disagrees with fstat")
    return resolved


def _descriptor_ancestor_identities(fd: int) -> frozenset[tuple[int, int]]:
    """Return directory ancestry without converting ``fd`` back to a pathname."""
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if nofollow == 0 or directory == 0:
        raise RuntimeError("safe analyzer output requires O_NOFOLLOW/O_DIRECTORY")
    flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | directory
    current_fd = os.dup(fd)
    identities: set[tuple[int, int]] = set()
    try:
        for _depth in range(_MAX_OUTPUT_ANCESTOR_DEPTH):
            current_stat = os.fstat(current_fd)
            if not stat.S_ISDIR(current_stat.st_mode):
                raise RuntimeError("analyzer output ancestor is not a directory")
            current_identity = _directory_identity(current_stat)
            if current_identity in identities:
                raise RuntimeError("analyzer output directory ancestry contains a cycle")
            identities.add(current_identity)
            parent_fd = os.open("..", flags, dir_fd=current_fd)
            try:
                parent_stat = os.fstat(parent_fd)
                if not stat.S_ISDIR(parent_stat.st_mode):
                    raise RuntimeError("analyzer output ancestor is not a directory")
                parent_identity = _directory_identity(parent_stat)
            except BaseException:
                os.close(parent_fd)
                raise
            if parent_identity == current_identity:
                os.close(parent_fd)
                break
            os.close(current_fd)
            current_fd = parent_fd
        else:
            raise RuntimeError("analyzer output directory ancestry is too deep")
        return frozenset(identities)
    except OSError as exc:
        raise RuntimeError(
            "could not authenticate analyzer output directory ancestry"
        ) from exc
    finally:
        os.close(current_fd)


def _stable_file_identity(stat_result: os.stat_result) -> tuple[int, ...]:
    return (
        int(stat_result.st_mode), int(stat_result.st_size),
        int(stat_result.st_mtime_ns), int(stat_result.st_ctime_ns),
        int(stat_result.st_dev), int(stat_result.st_ino),
    )


def _read_consumed_artifact(path: Path, *, role: str) -> tuple[bytes, dict[str, Any]]:
    """Read exact bytes while retaining their descriptor-authenticated identity."""
    lexical_path = path.expanduser().absolute()
    fd = os.open(lexical_path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        canonical_before = _strict_descriptor_path(fd, kind="file")
        with os.fdopen(os.dup(fd), "rb") as fh:
            content = fh.read()
        after = os.fstat(fd)
        canonical_after = _strict_descriptor_path(fd, kind="file")
        if (
            _stable_file_identity(before) != _stable_file_identity(after)
            or canonical_before != canonical_after
            or len(content) != int(after.st_size)
        ):
            raise RuntimeError(f"consumed {role} changed during authenticated read")
        return content, {
            "role": role,
            "lexical_path": str(lexical_path),
            "canonical_path": str(canonical_after),
            "size": int(after.st_size),
            "mtime_ns": int(after.st_mtime_ns),
            "ctime_ns": int(after.st_ctime_ns),
            "device": int(after.st_dev),
            "inode": int(after.st_ino),
            "sha256": hashlib.sha256(content).hexdigest(),
            "stable_read": True,
            "descriptor_authenticated": True,
        }
    finally:
        os.close(fd)


def _open_directory_anchored(path: Path, *, create: bool) -> int:
    """Open an absolute directory without following a path-component symlink."""
    if not path.is_absolute():
        raise ValueError("anchored output parent must be absolute")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    if nofollow == 0 or directory == 0:
        raise RuntimeError("safe analyzer output requires O_NOFOLLOW/O_DIRECTORY")
    flags = os.O_RDONLY | os.O_CLOEXEC | nofollow | directory
    current_fd = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise
                try:
                    os.mkdir(component, dir_fd=current_fd)
                except FileExistsError:
                    # A concurrent creator is acceptable only if the no-follow
                    # open below proves that it created a real directory.
                    pass
                next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _require_anchored_output_stable(target: _AnchoredOutputTarget) -> None:
    """Revalidate both the lexical binding and the descriptor-bound destination."""
    try:
        current_parent = os.stat(target.lexical_path.parent)
        descriptor_parent = os.fstat(target.parent_fd)
    except OSError as exc:
        raise RuntimeError("analyzer output parent changed during publication") from exc
    if (
        not stat.S_ISDIR(current_parent.st_mode)
        or not stat.S_ISDIR(descriptor_parent.st_mode)
        or _directory_identity(current_parent) != target.parent_identity
        or _directory_identity(descriptor_parent) != target.parent_identity
    ):
        raise RuntimeError("analyzer output parent changed during publication")
    descriptor_parent_path = _strict_descriptor_path(
        target.parent_fd, kind="directory",
    )
    descriptor_ancestors = _descriptor_ancestor_identities(target.parent_fd)
    _require_safe_output_path(
        target.input_path,
        target.meta_path,
        target.lexical_path,
        manifest=target.manifest,
        consumed_artifacts=target.consumed_artifacts,
        _resolved_output=descriptor_parent_path / target.name,
        _anchored_ancestor_identities=descriptor_ancestors,
    )


@contextmanager
def _anchored_output_target(
    input_path: Path,
    meta_path: Path,
    output_path: Path,
    *,
    manifest: dict[str, Any] | None = None,
    consumed_artifacts: Sequence[Mapping[str, Any]] = (),
) -> Generator[_AnchoredOutputTarget, None, None]:
    """Pin the validated output parent for the entire atomic publication."""
    _require_safe_output_path(
        input_path,
        meta_path,
        output_path,
        manifest=manifest,
        consumed_artifacts=consumed_artifacts,
    )
    lexical_path = output_path.expanduser().absolute()
    resolved_parent = lexical_path.parent.resolve()
    parent_fd = _open_directory_anchored(resolved_parent, create=True)
    target = _AnchoredOutputTarget(
        lexical_path=lexical_path,
        parent_fd=parent_fd,
        parent_identity=_directory_identity(os.fstat(parent_fd)),
        input_path=input_path,
        meta_path=meta_path,
        manifest=manifest,
        consumed_artifacts=tuple(consumed_artifacts),
    )
    try:
        _require_anchored_output_stable(target)
        _require_fresh_output_leaf(target)
        yield target
    finally:
        os.close(parent_fd)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return bool(left.st_dev == right.st_dev and left.st_ino == right.st_ino)


def _require_fresh_output_leaf(target: _AnchoredOutputTarget) -> None:
    try:
        os.stat(target.name, dir_fd=target.parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise FileExistsError(
        f"analyzer output already exists; choose a fresh --out path: "
        f"{target.lexical_path}"
    )


def _unlink_owned_output_entry(
    target: _AnchoredOutputTarget,
    name: str,
    expected: os.stat_result,
) -> bool:
    """Remove an analyzer-created entry, never a different observed inode."""
    try:
        current = os.stat(name, dir_fd=target.parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not _same_file_identity(current, expected):
        return False
    try:
        os.unlink(name, dir_fd=target.parent_fd)
    except FileNotFoundError:
        return False
    return True


def _write_json_atomic(target: _AnchoredOutputTarget, rendered: str) -> None:
    """Publish fresh JSON without clobbering through the validated directory fd."""
    staging_fd: int | None = None
    staging_identity: os.stat_result | None = None
    destination_is_ours = False
    try:
        _require_anchored_output_stable(target)
        _require_fresh_output_leaf(target)
        staging_fd = os.open(
            target.staging_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o666,
            dir_fd=target.parent_fd,
        )
        staging_identity = os.fstat(staging_fd)
        with os.fdopen(os.dup(staging_fd), "w", encoding="utf-8") as fh:
            fh.write(rendered)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        staged = os.stat(
            target.staging_name, dir_fd=target.parent_fd, follow_symlinks=False,
        )
        if not _same_file_identity(staged, staging_identity):
            raise RuntimeError("analyzer output staging entry changed before publication")
        _require_anchored_output_stable(target)
        try:
            os.link(
                target.staging_name,
                target.name,
                src_dir_fd=target.parent_fd,
                dst_dir_fd=target.parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                f"analyzer output appeared during publication; choose a fresh "
                f"--out path: {target.lexical_path}"
            ) from exc
        destination = os.stat(
            target.name, dir_fd=target.parent_fd, follow_symlinks=False,
        )
        if not _same_file_identity(destination, staging_identity):
            raise RuntimeError("published analyzer output differs from its staging file")
        destination_is_ours = True
        _unlink_owned_output_entry(target, target.staging_name, staging_identity)
        os.fsync(target.parent_fd)
        _require_anchored_output_stable(target)
        final_destination = os.stat(
            target.name, dir_fd=target.parent_fd, follow_symlinks=False,
        )
        if not _same_file_identity(final_destination, staging_identity):
            raise RuntimeError("published analyzer output changed after publication")
    except BaseException:
        directory_changed = False
        if destination_is_ours and staging_identity is not None:
            directory_changed = _unlink_owned_output_entry(
                target, target.name, staging_identity,
            )
        if staging_identity is not None:
            directory_changed = (
                _unlink_owned_output_entry(
                    target, target.staging_name, staging_identity,
                )
                or directory_changed
            )
        if directory_changed:
            os.fsync(target.parent_fd)
        raise
    finally:
        if staging_fd is not None:
            os.close(staging_fd)


def _add_protected_path(protected: set[Path], raw_path: str | Path) -> None:
    """Protect both the authenticated spelling and its current resolution."""
    lexical = Path(raw_path).expanduser().absolute()
    protected.add(lexical)
    protected.add(lexical.resolve())


def _recorded_identity(record: Mapping[str, Any]) -> tuple[int, int] | None:
    device = record.get("device")
    inode = record.get("inode")
    if (
        isinstance(device, int) and not isinstance(device, bool)
        and isinstance(inode, int) and not isinstance(inode, bool)
        and device >= 0 and inode > 0
    ):
        return device, inode
    return None


def _manifest_object_identities(value: Any) -> set[tuple[int, int]]:
    """Collect every explicit device/inode pair in authenticated manifest data."""
    identities: set[tuple[int, int]] = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            identity = _recorded_identity(current)
            if identity is not None:
                identities.add(identity)
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return identities


def _manifest_directory_identities(
    manifest: Mapping[str, Any] | None,
) -> set[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    if manifest is None:
        return identities
    syzygy = manifest.get("syzygy")
    directories = syzygy.get("directories") if isinstance(syzygy, dict) else None
    for directory in directories if isinstance(directories, list) else []:
        if not isinstance(directory, dict):
            continue
        root_identity = directory.get("root_identity")
        if isinstance(root_identity, dict):
            identity = _recorded_identity(root_identity)
            if identity is not None:
                identities.add(identity)
    origin = manifest.get("matched_row_origin_verification")
    snapshot = origin.get("snapshot_inventory") if isinstance(origin, dict) else None
    if isinstance(snapshot, dict):
        root_identity = snapshot.get("root_identity")
        if isinstance(root_identity, dict):
            identity = _recorded_identity(root_identity)
            if identity is not None:
                identities.add(identity)
        shards = snapshot.get("shards")
        for shard in shards if isinstance(shards, list) else []:
            if isinstance(shard, dict):
                identity = _recorded_identity(shard)
                if identity is not None:
                    identities.add(identity)
    return identities


def _tablebase_file_identities(
    manifest: Mapping[str, Any] | None,
) -> set[tuple[int, int]]:
    identities: set[tuple[int, int]] = set()
    syzygy = manifest.get("syzygy") if manifest is not None else None
    directories = syzygy.get("directories") if isinstance(syzygy, dict) else None
    for directory in directories if isinstance(directories, list) else []:
        rows = directory.get("file_identities") if isinstance(directory, dict) else None
        for row in rows if isinstance(rows, list) else []:
            if (
                isinstance(row, list) and len(row) == 6
                and isinstance(row[4], int) and not isinstance(row[4], bool)
                and isinstance(row[5], int) and not isinstance(row[5], bool)
                and row[4] >= 0 and row[5] > 0
            ):
                identities.add((row[4], row[5]))
    return identities


def _require_safe_output_path(
    input_path: Path,
    meta_path: Path,
    output_path: Path | None,
    *,
    manifest: dict[str, Any] | None = None,
    consumed_artifacts: Sequence[Mapping[str, Any]] = (),
    _resolved_output: Path | None = None,
    _anchored_ancestor_identities: Collection[tuple[int, int]] | None = None,
) -> None:
    if output_path is None:
        return
    if reserved_output_path(output_path):
        raise ValueError("--out must not use the output lock/staging namespace")
    output = (
        _resolved_output.expanduser().resolve()
        if _resolved_output is not None
        else output_path.expanduser().resolve()
    )
    protected: set[Path] = set()
    for path in (
        input_path,
        meta_path,
        Path(__file__),
        Path(solve_reachable_oracle.__code__.co_filename),
    ):
        _add_protected_path(protected, path)
    consumed_identities: set[tuple[int, int]] = set()
    for record in consumed_artifacts:
        for field in ("lexical_path", "canonical_path"):
            raw_path = record.get(field)
            if isinstance(raw_path, str) and raw_path:
                _add_protected_path(protected, raw_path)
        identity = _recorded_identity(record)
        if identity is not None:
            consumed_identities.add(identity)
    protected_object_identities = {
        *consumed_identities,
        *_manifest_object_identities(manifest),
        *_tablebase_file_identities(manifest),
    }
    protected_directory_identities = _manifest_directory_identities(manifest)
    repo_root = Path(__file__).resolve().parents[1]
    if (
        repo_controlled_output(output_path, repo_root)
        or (
            _resolved_output is not None
            and repo_controlled_output(_resolved_output, repo_root)
        )
    ):
        raise ValueError("--out must not overwrite a tracked or repository-control path")
    if manifest is not None:
        for name in (
            "output", "producer_script", "publication_helper", "checkpoint",
            "checkpoint_params", "audit_set", "matched_rows", "matched_rows_report",
            "preregistration", "features_extension", "mcts_extension",
            "lc0_extension",
        ):
            artifact = manifest.get(name)
            if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
                _add_protected_path(protected, artifact["path"])
        producer_sources = manifest.get("producer_sources")
        if isinstance(producer_sources, dict):
            for artifact in producer_sources.values():
                if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
                    _add_protected_path(protected, artifact["path"])
        preimport = manifest.get("python_preimport")
        preimport_files = preimport.get("files") if isinstance(preimport, dict) else None
        if isinstance(preimport_files, dict):
            for artifact in preimport_files.values():
                if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
                    _add_protected_path(protected, artifact["path"])
        params_inventory = manifest.get("params_candidate_inventory")
        params_candidates = (
            params_inventory.get("candidates")
            if isinstance(params_inventory, dict) else None
        )
        if isinstance(params_candidates, list):
            for candidate in params_candidates:
                if isinstance(candidate, dict) and isinstance(candidate.get("path"), str):
                    _add_protected_path(protected, candidate["path"])
    if output in protected:
        raise ValueError("--out must not overwrite a consumed input artifact")
    try:
        output_identity = _directory_identity(os.stat(
            _resolved_output if _resolved_output is not None else output_path,
        ))
    except FileNotFoundError:
        output_identity = None
    if output_identity is not None and output_identity in protected_object_identities:
        raise ValueError("--out must not replace an alias of a consumed input artifact")
    syzygy = manifest.get("syzygy") if manifest is not None else None
    directories = syzygy.get("directories") if isinstance(syzygy, dict) else None
    for row in directories if isinstance(directories, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            continue
        directory_paths: set[Path] = set()
        _add_protected_path(directory_paths, row["path"])
        if any(output == path or output.is_relative_to(path) for path in directory_paths):
            raise ValueError("--out must not be inside a consumed Syzygy directory")
    origin_verification = (
        manifest.get("matched_row_origin_verification")
        if manifest is not None else None
    )
    snapshot = (
        origin_verification.get("snapshot_inventory")
        if isinstance(origin_verification, dict) else None
    )
    snapshot_path = snapshot.get("path") if isinstance(snapshot, dict) else None
    if isinstance(snapshot_path, str) and snapshot_path:
        snapshot_paths: set[Path] = set()
        _add_protected_path(snapshot_paths, snapshot_path)
        if any(output == path or output.is_relative_to(path) for path in snapshot_paths):
            raise ValueError(
                "--out must not be inside an authenticated replay-snapshot directory"
            )
    if _anchored_ancestor_identities is None:
        output_parent = (
            _resolved_output.parent if _resolved_output is not None
            else output_path.expanduser().absolute().parent
        )
        output_ancestor_identities: set[tuple[int, int]] = set()
        for parent in (output_parent, *output_parent.parents):
            try:
                output_ancestor_identities.add(_directory_identity(os.stat(parent)))
            except FileNotFoundError:
                continue
    else:
        output_ancestor_identities = set(_anchored_ancestor_identities)
    if output_ancestor_identities & protected_directory_identities:
        raise ValueError("--out must not be inside a consumed protected directory")


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


def _canonical_analysis_contract() -> dict[str, int | float | str]:
    """Exact analysis knobs that a decision-grade collection must freeze."""
    return {
        "folds": _CANONICAL_FOLDS,
        "bootstrap_samples": _CANONICAL_BOOTSTRAP_SAMPLES,
        "seed": _CANONICAL_SEED,
        "allocation_fraction": _CANONICAL_ALLOCATION_FRACTION,
        "min_capture_gain": _CANONICAL_MIN_CAPTURE_GAIN,
        "min_oracle_headroom": _CANONICAL_MIN_ORACLE_HEADROOM,
        "min_bootstrap_valid_fraction": _CANONICAL_MIN_BOOTSTRAP_VALID_FRACTION,
        "reachable_selection_semantics": _FOLD_SELECTION_SEMANTICS,
        "reachable_oracle_semantics": _FOLD_ORACLE_SEMANTICS,
        "bootstrap_resampling_semantics": _BOOTSTRAP_RESAMPLING_SEMANTICS,
        "bootstrap_interval_semantics": _BOOTSTRAP_INTERVAL_SEMANTICS,
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


def _canonical_reference_cp(value: Any, *, key: str, field: str) -> str:
    """Canonicalize one finite ruler score for content-sensitive evidence hashing."""
    if value is None or isinstance(value, bool):
        raise ValueError(f"{key}: {field} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}: {field} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{key}: {field} must be a finite number")
    # Numeric JSON spellings such as 1, 1.0, and 1e0 carry the same CP value.
    # Float.hex is locale-independent and exact; collapse signed zero as well.
    if number == 0.0:
        number = 0.0
    return number.hex()


def _deep_reference_evidence_summary(
    positions: Sequence[Mapping[str, Any]], *, audit_set_sha256: Any,
) -> dict[str, Any]:
    """Summarize the raw teacher budget and finite-MultiPV coverage per position.

    The approved audit-set digest authenticates how the teacher was invoked.
    Actual nodes may be below the requested million on forced terminal searches,
    so they are banked as diagnostics rather than incorrectly treated as a
    per-row minimum. ``minimum_multipv`` is a coverage requirement: positions
    with fewer than ten legal moves must list every legal move.
    """
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position in positions:
        key = position.get("key")
        fen = position.get("fen")
        nodes = position.get("deep_reference_nodes")
        depth = position.get("deep_reference_depth")
        recorded_count = position.get("deep_reference_scored_multipv")
        move_cp = position.get("deep_reference_move_cp")
        best_cp = position.get("deep_reference_best_cp")
        if not isinstance(key, str) or not key:
            raise ValueError("deep-reference evidence has no position key")
        if not isinstance(fen, str) or not fen:
            raise ValueError(f"{key}: deep-reference evidence has no FEN")
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            raise ValueError(f"{key}: deep-reference evidence has an invalid FEN") from exc
        if not _nonnegative_int(nodes) or not _nonnegative_int(depth):
            raise ValueError(f"{key}: deep-reference nodes/depth must be integers")
        if not _nonnegative_int(recorded_count) or not isinstance(move_cp, dict):
            raise ValueError(f"{key}: deep-reference MultiPV evidence is malformed")
        assert isinstance(nodes, int)
        assert isinstance(depth, int)
        assert isinstance(recorded_count, int)
        scored_count = len(move_cp)
        if any(not isinstance(uci, str) or not uci for uci in move_cp):
            raise ValueError(f"{key}: deep-reference move key is malformed")
        canonical_move_cp: list[list[str]] = []
        for uci, cp in sorted(move_cp.items()):
            canonical_move_cp.append([
                uci,
                _canonical_reference_cp(
                    cp, key=key, field=f"deep_reference_move_cp[{uci!r}]",
                ),
            ])
        canonical_best_cp = _canonical_reference_cp(
            best_cp, key=key, field="deep_reference_best_cp",
        )
        legal_move_count = board.legal_moves.count()
        required_count = min(_DEEP_REFERENCE_MIN_MULTIPV, legal_move_count)
        record = {
            "key": key,
            "nodes": nodes,
            "depth": depth,
            "scored_multipv": scored_count,
            "recorded_scored_multipv": recorded_count,
            "legal_move_count": legal_move_count,
            "required_scored_multipv": required_count,
            "deep_reference_best_cp": canonical_best_cp,
            "deep_reference_move_cp": canonical_move_cp,
        }
        records.append(record)
        reasons: list[str] = []
        if nodes <= 0:
            reasons.append("nodes_not_positive")
        if depth <= 0:
            reasons.append("depth_not_positive")
        if recorded_count != scored_count:
            reasons.append("recorded_multipv_count_mismatch")
        if scored_count < required_count:
            reasons.append("insufficient_multipv_coverage")
        if reasons:
            failures.append({"key": key, "reasons": reasons})

    records.sort(key=lambda row: str(row["key"]))
    failures.sort(key=lambda row: str(row["key"]))
    evidence_bytes = json.dumps(
        records, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return {
        "schema": _DEEP_REFERENCE_EVIDENCE_SCHEMA,
        "label_source": "frozen_unhandicapped_stockfish_audit_set",
        "approved_audit_set_sha256": _APPROVED_AUDIT_SET_SHA256,
        "audit_set_sha256": audit_set_sha256,
        "audit_set_identity_passed": audit_set_sha256 == _APPROVED_AUDIT_SET_SHA256,
        "requested_nodes": _DEEP_REFERENCE_REQUESTED_NODES,
        "observed_nodes_semantics": "actual_nodes_may_stop_early_on_forced_terminal_search",
        "positive_depth_required": True,
        "minimum_multipv": _DEEP_REFERENCE_MIN_MULTIPV,
        "multipv_coverage_requirement": _DEEP_REFERENCE_COVERAGE,
        "per_position_fields": [
            "deep_reference_nodes",
            "deep_reference_depth",
            "deep_reference_scored_multipv",
            "deep_reference_best_cp",
            "deep_reference_move_cp",
        ],
        "position_count": len(records),
        "position_evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "minimum_observed_nodes": min(
            (int(row["nodes"]) for row in records), default=0,
        ),
        "minimum_observed_depth": min(
            (int(row["depth"]) for row in records), default=0,
        ),
        "minimum_observed_scored_multipv": min(
            (int(row["scored_multipv"]) for row in records), default=0,
        ),
        "positions_below_requested_nodes": [
            str(row["key"])
            for row in records
            if int(row["nodes"]) < _DEEP_REFERENCE_REQUESTED_NODES
        ],
        "failing_position_count": len(failures),
        "failing_positions": failures,
        "passed": bool(
            records
            and audit_set_sha256 == _APPROVED_AUDIT_SET_SHA256
            and not failures
        ),
    }


def _valid_deep_reference_evidence(value: Any) -> bool:
    """Validate the versioned ruler contract before trajectory rows are read."""
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "label_source",
        "approved_audit_set_sha256",
        "audit_set_sha256",
        "audit_set_identity_passed",
        "requested_nodes",
        "observed_nodes_semantics",
        "positive_depth_required",
        "minimum_multipv",
        "multipv_coverage_requirement",
        "per_position_fields",
        "position_count",
        "position_evidence_sha256",
        "minimum_observed_nodes",
        "minimum_observed_depth",
        "minimum_observed_scored_multipv",
        "positions_below_requested_nodes",
        "failing_position_count",
        "failing_positions",
        "passed",
    }:
        return False
    failures = value.get("failing_positions")
    early_stops = value.get("positions_below_requested_nodes")
    if (
        value.get("schema") != _DEEP_REFERENCE_EVIDENCE_SCHEMA
        or value.get("label_source")
        != "frozen_unhandicapped_stockfish_audit_set"
        or value.get("approved_audit_set_sha256") != _APPROVED_AUDIT_SET_SHA256
        or value.get("audit_set_sha256") != _APPROVED_AUDIT_SET_SHA256
        or value.get("audit_set_identity_passed") is not True
        or value.get("requested_nodes") != _DEEP_REFERENCE_REQUESTED_NODES
        or value.get("observed_nodes_semantics")
        != "actual_nodes_may_stop_early_on_forced_terminal_search"
        or value.get("positive_depth_required") is not True
        or value.get("minimum_multipv") != _DEEP_REFERENCE_MIN_MULTIPV
        or value.get("multipv_coverage_requirement") != _DEEP_REFERENCE_COVERAGE
        or value.get("per_position_fields") != [
            "deep_reference_nodes",
            "deep_reference_depth",
            "deep_reference_scored_multipv",
            "deep_reference_best_cp",
            "deep_reference_move_cp",
        ]
        or not _positive_int(value.get("position_count"))
        or not _valid_sha256(value.get("position_evidence_sha256"))
        or not _nonnegative_int(value.get("minimum_observed_nodes"))
        or not _nonnegative_int(value.get("minimum_observed_depth"))
        or not _nonnegative_int(value.get("minimum_observed_scored_multipv"))
        or not isinstance(early_stops, list)
        or any(not isinstance(key, str) or not key for key in early_stops)
        or early_stops != sorted(set(early_stops))
        or not _nonnegative_int(value.get("failing_position_count"))
        or not isinstance(failures, list)
        or value.get("failing_position_count") != len(failures)
        or not isinstance(value.get("passed"), bool)
        or value.get("passed") is not (len(failures) == 0)
    ):
        return False
    previous_key = ""
    for failure in failures:
        if (
            not isinstance(failure, dict)
            or set(failure) != {"key", "reasons"}
            or not isinstance(failure.get("key"), str)
            or not failure.get("key")
            or failure["key"] <= previous_key
            or not isinstance(failure.get("reasons"), list)
            or not failure["reasons"]
            or any(
                reason not in {
                    "nodes_not_positive",
                    "depth_not_positive",
                    "recorded_multipv_count_mismatch",
                    "insufficient_multipv_coverage",
                }
                for reason in failure["reasons"]
            )
        ):
            return False
        previous_key = failure["key"]
    return True


def _panel_key_digest(keys: Sequence[str]) -> str:
    payload = json.dumps(
        sorted(keys), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _panel_piece_bucket(piece_count: int) -> int:
    return min(32, max(2, int(piece_count))) // 4


def _panel_source_counts(value: Any) -> dict[int, int] | None:
    if not isinstance(value, list):
        return None
    counts: dict[int, int] = {}
    for row in value:
        if not isinstance(row, dict) or set(row) != {"source", "count"}:
            return None
        source, count = row["source"], row["count"]
        if (
            not isinstance(source, int)
            or isinstance(source, bool)
            or source in counts
            or not _nonnegative_int(count)
        ):
            return None
        counts[source] = count
    if any(source not in counts for source in _PANEL_REQUIRED_SOURCES):
        return None
    return counts


def _panel_stratum_counts(value: Any) -> dict[tuple[int, int, int], int] | None:
    if not isinstance(value, list):
        return None
    counts: dict[tuple[int, int, int], int] = {}
    for row in value:
        if not isinstance(row, dict) or set(row) != {
            "source", "phase", "piece_bucket", "count",
        }:
            return None
        source = row["source"]
        phase = row["phase"]
        piece_bucket = row["piece_bucket"]
        count = row["count"]
        if (
            not isinstance(source, int)
            or isinstance(source, bool)
            or not isinstance(phase, int)
            or isinstance(phase, bool)
            or phase not in (0, 1, 2)
            or not isinstance(piece_bucket, int)
            or isinstance(piece_bucket, bool)
            or not 0 <= piece_bucket <= 8
            or not _positive_int(count)
        ):
            return None
        key = (source, phase, piece_bucket)
        if key in counts:
            return None
        counts[key] = count
    return counts


def _valid_panel_selection(
    value: Any, *, requested_max_positions: Any, requested_position_count: Any,
) -> bool:
    if not isinstance(value, dict):
        return False
    available_count = value.get("available_position_count")
    selected_count = value.get("selected_position_count")
    requested_limit = value.get("requested_max_positions")
    available_sources = _panel_source_counts(value.get("available_source_counts"))
    selected_sources = _panel_source_counts(value.get("selected_source_counts"))
    available_strata = _panel_stratum_counts(value.get("available_stratum_counts"))
    selected_strata = _panel_stratum_counts(value.get("selected_stratum_counts"))
    balance = value.get("source_balance")
    if (
        value.get("strategy") != _PANEL_SELECTION_STRATEGY
        or value.get("stratum_fields") != ["source", "phase", "piece_bucket"]
        or value.get("piece_bucket_definition")
        != "clamp_2_32_then_floor_divide_by_4"
        or value.get("within_stratum_order")
        != "sha256_position_key_then_position_key"
        or value.get("source_order") != list(_PANEL_REQUIRED_SOURCES)
        or not _nonnegative_int(requested_limit)
        or requested_limit != requested_max_positions
        or not _nonnegative_int(available_count)
        or not _nonnegative_int(selected_count)
        or selected_count != requested_position_count
        or available_sources is None
        or selected_sources is None
        or available_strata is None
        or selected_strata is None
        or not _valid_sha256(value.get("available_position_keys_sha256"))
        or not _valid_sha256(value.get("selected_position_keys_sha256"))
        or value.get("available_keys_unique") is not True
        or value.get("phase_morphology_passed") is not True
        or not isinstance(balance, dict)
        or set(balance) != {"maximum_difference", "observed_difference", "passed"}
    ):
        return False
    assert isinstance(requested_limit, int)
    assert isinstance(available_count, int)
    assert isinstance(selected_count, int)
    assert available_sources is not None
    assert selected_sources is not None
    assert available_strata is not None
    assert selected_strata is not None
    assert isinstance(balance, dict)
    expected_selected = (
        available_count
        if requested_limit <= 0 or requested_limit >= available_count
        else requested_limit
    )
    expected_mode = (
        "full_set"
        if expected_selected == available_count
        else "truncated_joint_stratified"
    )
    available_by_stratum_source: defaultdict[int, int] = defaultdict(int)
    selected_by_stratum_source: defaultdict[int, int] = defaultdict(int)
    for (source, _phase, _piece_bucket), count in available_strata.items():
        available_by_stratum_source[source] += count
    for stratum, count in selected_strata.items():
        selected_by_stratum_source[stratum[0]] += count
        if count > available_strata.get(stratum, 0):
            return False
    source_domain_passed = all(
        source in _PANEL_REQUIRED_SOURCES or count == 0
        for source, count in available_sources.items()
    )
    source_difference = abs(
        selected_sources[_PANEL_REQUIRED_SOURCES[0]]
        - selected_sources[_PANEL_REQUIRED_SOURCES[1]]
    )
    source_balance_passed = bool(
        all(
            source in _PANEL_REQUIRED_SOURCES or count == 0
            for source, count in selected_sources.items()
        )
        and source_difference <= _PANEL_SOURCE_BALANCE_MAX_DIFFERENCE
    )
    decision_grade_passed = bool(
        selected_count > 0
        and source_domain_passed
        and source_balance_passed
    )
    return bool(
        selected_count == expected_selected
        and value.get("selection_mode") == expected_mode
        and sum(available_sources.values()) == available_count
        and sum(selected_sources.values()) == selected_count
        and dict(available_by_stratum_source) == {
            source: count for source, count in available_sources.items() if count
        }
        and dict(selected_by_stratum_source) == {
            source: count for source, count in selected_sources.items() if count
        }
        and all(
            selected_sources.get(source, 0) <= count
            for source, count in available_sources.items()
        )
        and value.get("source_domain_passed") is source_domain_passed
        and balance.get("maximum_difference")
        == _PANEL_SOURCE_BALANCE_MAX_DIFFERENCE
        and balance.get("observed_difference") == source_difference
        and balance.get("passed") is source_balance_passed
        and value.get("decision_grade_passed") is decision_grade_passed
    )


def _panel_selection_matches_observations(
    manifest: dict[str, Any], by_key: dict[str, list[dict[str, Any]]],
) -> bool:
    selection = manifest.get("panel_selection")
    if not isinstance(selection, dict):
        return False
    observations = [trajectory[0] for trajectory in by_key.values()]
    excluded = manifest.get("excluded_positions")
    if not isinstance(excluded, list) or not all(isinstance(row, dict) for row in excluded):
        return False
    observations.extend(excluded)
    keys: list[str] = []
    source_counts: defaultdict[int, int] = defaultdict(int)
    stratum_counts: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    for row in observations:
        key, fen = row.get("key"), row.get("fen")
        source, phase = row.get("source"), row.get("phase")
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(fen, str)
            or not fen
            or not isinstance(source, int)
            or isinstance(source, bool)
            or source not in _PANEL_REQUIRED_SOURCES
            or not isinstance(phase, int)
            or isinstance(phase, bool)
        ):
            return False
        try:
            piece_count = chess.popcount(chess.Board(fen).occupied)
        except ValueError:
            return False
        if phase != phase_bucket(piece_count):
            return False
        keys.append(key)
        source_counts[source] += 1
        stratum_counts[(source, phase, _panel_piece_bucket(piece_count))] += 1
    selected_sources = _panel_source_counts(selection.get("selected_source_counts"))
    selected_strata = _panel_stratum_counts(selection.get("selected_stratum_counts"))
    if selected_sources is None or selected_strata is None:
        return False
    observed_sources = {
        source: int(source_counts.get(source, 0))
        for source in sorted(set(_PANEL_REQUIRED_SOURCES) | set(source_counts))
    }
    return bool(
        len(keys) == len(set(keys)) == selection.get("selected_position_count")
        and _panel_key_digest(keys) == selection.get("selected_position_keys_sha256")
        and observed_sources == selected_sources
        and dict(stratum_counts) == selected_strata
    )


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


def _model_input_consumption_complete(manifest: Mapping[str, Any]) -> bool:
    """Validate that attested model inputs are the objects the loader consumed."""
    checkpoint = manifest.get("checkpoint")
    params = manifest.get("checkpoint_params")
    inventory = manifest.get("params_candidate_inventory")
    proof = manifest.get("model_input_consumption")

    def stable_artifact(value: Any, consumption: str) -> bool:
        return bool(
            _artifact_provenance_complete(value)
            and isinstance(value, dict)
            and isinstance(value.get("lexical_path"), str)
            and value.get("lexical_path")
            and _nonnegative_int(value.get("ctime_ns"))
            and _nonnegative_int(value.get("device"))
            and _positive_int(value.get("inode"))
            and value.get("stable_read") is True
            and value.get("consumption") == consumption
        )

    checkpoint_ok = stable_artifact(
        checkpoint, "torch_load_from_same_open_file_description",
    )
    params_ok = (
        params is None
        or stable_artifact(params, "json_decode_from_exact_authenticated_bytes")
    )
    inventory_ok = _params_candidate_inventory_complete(
        inventory, checkpoint=checkpoint, params=params,
    )
    inventory_digest = (
        inventory.get("inventory_sha256") if isinstance(inventory, dict) else None
    )
    selected_index = (
        inventory.get("selected_index") if isinstance(inventory, dict) else None
    )
    selected_path = (
        inventory.get("selected_path") if isinstance(inventory, dict) else None
    )
    return bool(
        checkpoint_ok
        and params_ok
        and inventory_ok
        and isinstance(proof, dict)
        and proof.get("schema") == _MODEL_INPUT_CONSUMPTION_SCHEMA
        and proof.get("checkpoint_open") == "absolute_lexical_path_o_nofollow"
        and proof.get("checkpoint")
        == "torch_load_from_same_open_file_description"
        and proof.get("checkpoint_path_reopened_by_loader") is False
        and proof.get("checkpoint_identity_verified_before_search") is True
        and proof.get(
            "checkpoint_sha256_streamed_from_same_open_file_description"
        ) is True
        and proof.get("params")
        == (
            "json_decode_from_exact_authenticated_bytes"
            if params is not None else "no_params_json"
        )
        and proof.get("params_open")
        == (
            "absolute_lexical_path_o_nofollow"
            if params is not None else "no_params_json"
        )
        and proof.get("params_path_reopened_by_loader") is False
        and proof.get("params_identity_verified_before_search") is True
        and proof.get("params_selection")
        == "first_is_file_in_checkpoint_ancestor_order"
        and proof.get("params_candidate_inventory_schema")
        == _PARAMS_CANDIDATE_INVENTORY_SCHEMA
        and proof.get("params_candidate_inventory_sha256") == inventory_digest
        and proof.get("params_candidate_inventory_verified_before_load") is True
        and proof.get("params_candidate_inventory_verified_after_load") is True
        and proof.get("params_selected_index") == selected_index
        and proof.get("params_selected_path") == selected_path
        and proof.get("passed") is True
    )


def _params_candidate_inventory_complete(
    inventory: Any, *, checkpoint: Any, params: Any,
) -> bool:
    """Validate the bounded lookup's positive and negative provenance."""
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema") != _PARAMS_CANDIDATE_INVENTORY_SCHEMA
        or inventory.get("search_limit") != 6
        or inventory.get("selection_policy")
        != "first_is_file_in_checkpoint_ancestor_order"
        or not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("lexical_path"), str)
    ):
        return False
    payload = {name: value for name, value in inventory.items() if name != "inventory_sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")
    if inventory.get("inventory_sha256") != hashlib.sha256(encoded).hexdigest():
        return False

    trainer = Path(checkpoint["lexical_path"])
    if inventory.get("trainer_pt") != str(trainer):
        return False
    expected_paths: list[str] = []
    current = trainer.parent
    for _ in range(6):
        expected_paths.append(str(current / "params.json"))
        if current.parent == current:
            break
        current = current.parent
    candidates = inventory.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(expected_paths):
        return False

    first_eligible: int | None = None
    for index, (candidate, expected_path) in enumerate(zip(candidates, expected_paths)):
        if (
            not isinstance(candidate, dict)
            or candidate.get("index") != index
            or candidate.get("path") != expected_path
            or candidate.get("state") not in {
                "absent", "regular", "symlink", "nonregular",
            }
            or not isinstance(candidate.get("resolves_to_regular_file"), bool)
        ):
            return False
        state = candidate["state"]
        identity = candidate.get("identity")
        if state == "absent":
            if identity is not None or candidate["resolves_to_regular_file"]:
                return False
        elif not _params_identity_complete(identity):
            return False
        if state == "regular" and candidate["resolves_to_regular_file"] is not True:
            return False
        if state == "nonregular" and candidate["resolves_to_regular_file"] is not False:
            return False
        components = candidate.get("parent_path_components")
        if not _params_path_components_complete(components, Path(expected_path).parent):
            return False
        if first_eligible is None and candidate["resolves_to_regular_file"]:
            first_eligible = index

    selected_index = inventory.get("selected_index")
    selected_path = inventory.get("selected_path")
    if first_eligible is None:
        return bool(selected_index is None and selected_path is None and params is None)
    selected = candidates[first_eligible]
    return bool(
        selected_index == first_eligible
        and selected_path == expected_paths[first_eligible]
        and selected.get("state") == "regular"
        and isinstance(params, dict)
        and params.get("lexical_path") == selected_path
        and all(
            params.get(name) == selected.get("identity", {}).get(name)
            for name in (
                "mode", "size", "mtime_ns", "ctime_ns", "device", "inode",
            )
        )
    )


def _params_identity_complete(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and _positive_int(value.get("mode"))
        and _nonnegative_int(value.get("size"))
        and _nonnegative_int(value.get("mtime_ns"))
        and _nonnegative_int(value.get("ctime_ns"))
        and _nonnegative_int(value.get("device"))
        and _positive_int(value.get("inode"))
    )


def _params_path_components_complete(value: Any, parent: Path) -> bool:
    if not isinstance(value, list) or not value:
        return False
    expected: list[str] = []
    current = Path(parent.anchor)
    for index, component in enumerate(parent.parts):
        if index:
            current /= component
        expected.append(str(current))
    if len(value) != len(expected):
        return False
    for component, expected_path in zip(value, expected):
        if (
            not isinstance(component, dict)
            or component.get("path") != expected_path
            or component.get("kind") not in {
                "directory", "symlink", "non_directory",
            }
            or not _params_identity_complete(component)
        ):
            return False
    return True


def _producer_source_matches_revision(
    artifact: Any, producer_git_sha: Any, relative_path: str,
) -> bool:
    """Authenticate producer Python bytes against their reported Git revision."""
    if (
        not isinstance(artifact, dict)
        or not isinstance(artifact.get("path"), str)
        or not artifact.get("path")
        or not _nonnegative_int(artifact.get("size"))
        or not _nonnegative_int(artifact.get("mtime_ns"))
        or not _valid_sha256(artifact.get("sha256"))
        or artifact.get("repo_relative_path") != relative_path
        or artifact.get("matches_producer_git_revision") is not True
        or artifact.get("matches_preimport_snapshot") is not True
        or artifact.get("source_only_import_verified") is not True
        or not isinstance(producer_git_sha, str)
    ):
        return False
    committed = _git_file_at_commit(producer_git_sha, relative_path)
    return bool(
        committed is not None
        and artifact.get("size") == len(committed)
        and artifact.get("sha256") == hashlib.sha256(committed).hexdigest()
    )


def _producer_sources_match_revision(sources: Any, producer_git_sha: Any) -> bool:
    """Verify the complete loaded producer Python surface, not only its entrypoint."""
    if not isinstance(sources, dict) or not sources:
        return False
    required = {
        "producer_script": "scripts/backtest_chunk_trajectory.py",
        "scripts.chunk_trajectory_publication": (
            "scripts/chunk_trajectory_publication.py"
        ),
        "scripts.analyze_chunk_controller": "scripts/analyze_chunk_controller.py",
        "scripts.repo_output_guard": "scripts/repo_output_guard.py",
        "scripts.match_audit_rows": "scripts/match_audit_rows.py",
        "scripts.approved_syzygy": "scripts/approved_syzygy.py",
        "chess_anti_engine.eval.audit": "chess_anti_engine/eval/audit.py",
        "chess_anti_engine.mcts.search_options": (
            "chess_anti_engine/mcts/search_options.py"
        ),
        "chess_anti_engine.uci.search": "chess_anti_engine/uci/search.py",
        "chess_anti_engine.uci.model_loader": (
            "chess_anti_engine/uci/model_loader.py"
        ),
    }
    if any(
        not _producer_source_matches_revision(
            sources.get(name), producer_git_sha, relative_path,
        )
        for name, relative_path in required.items()
    ):
        return False
    return all(
        isinstance(artifact, dict)
        and isinstance(artifact.get("repo_relative_path"), str)
        and _producer_source_matches_revision(
            artifact, producer_git_sha, artifact["repo_relative_path"],
        )
        for artifact in sources.values()
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


def _git_python_tree_at_commit(commit: str) -> dict[str, tuple[str, str, str]] | None:
    """Return mode, kind, and blob id for the complete tracked Python surface."""
    if (
        len(commit) != 40
        or any(char not in "0123456789abcdef" for char in commit.lower())
    ):
        return None
    repo_root = Path(__file__).resolve().parents[1]
    try:
        raw_tree = subprocess.check_output(
            ["git", "-C", str(repo_root), "ls-tree", "-r", "-z", commit],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    tree: dict[str, tuple[str, str, str]] = {}
    for raw_entry in raw_tree.split(b"\0"):
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", 1)
        relative_path = raw_path.decode("utf-8", errors="surrogateescape")
        if relative_path.endswith(".py"):
            mode, kind, oid = metadata.decode("ascii").split()
            tree[relative_path] = (mode, kind, oid)
    return tree


def _producer_preimport_matches_revision(proof: Any, producer_sha: Any) -> bool:
    """Authenticate the producer's before-import full-Python source snapshot."""
    if (
        not isinstance(proof, dict)
        or proof.get("schema") != _PYTHON_PREIMPORT_SCHEMA
        or proof.get("git_sha") != producer_sha
        or proof.get("final_git_sha") != producer_sha
        or proof.get("snapshot_stage") != "before_project_or_third_party_imports"
        or proof.get("trust_boundary")
        != "already_executing_entry_script_and_python_process"
        or proof.get("preexisting_project_modules") != []
        or proof.get("source_tree_matches_revision") is not True
        or proof.get("passed") is not True
        or not _valid_sha256(proof.get("tracked_python_surface_sha256"))
        or not isinstance(proof.get("files"), dict)
        or not isinstance(producer_sha, str)
    ):
        return False
    files = proof["files"]
    source_only = proof.get("source_only_import")
    if (
        not isinstance(source_only, dict)
        or source_only.get("schema") != "deepfin.source_only_import.v2"
        or source_only.get("active") is not True
        or source_only.get("installed") is not True
        or source_only.get("first_finder") is not True
        or source_only.get("git_sha") != producer_sha
        or source_only.get("tracked_python_surface_sha256")
        != proof.get("tracked_python_surface_sha256")
        or source_only.get("project_scope") != ["chess_anti_engine", "scripts"]
        or source_only.get("execution") != "compile_authenticated_source_bytes"
        or source_only.get("bytecode_cache_reads") is not False
        or source_only.get("native_extension_loading")
        != "default_deny_exact_preimport_artifact_authenticated_loader"
        or source_only.get("permitted_native_modules") != _NATIVE_MODULES
        or source_only.get("authorized_native_modules") != _NATIVE_MODULES
        or not isinstance(source_only.get("authorized_native_artifacts"), dict)
        or set(source_only["authorized_native_artifacts"]) != set(_NATIVE_MODULES)
        or source_only.get("failures") != []
        or not isinstance(source_only.get("verified_modules"), dict)
        or not isinstance(source_only.get("verified_native_modules"), dict)
        or set(source_only["verified_native_modules"]) != set(_NATIVE_MODULES)
        or not isinstance(source_only.get("loaded_project_modules"), dict)
        or source_only["loaded_project_modules"].get("passed") is not True
        or source_only["loaded_project_modules"].get("unverified_modules") != []
    ):
        return False
    tree = _git_python_tree_at_commit(producer_sha)
    if (
        not files
        or tree is None
        or set(files) != set(tree)
        or proof.get("tracked_python_file_count") != len(files)
    ):
        return False
    surface_rows: list[list[Any]] = []
    for relative_path, expected in sorted(tree.items()):
        artifact = files.get(relative_path)
        mode, kind, oid = expected
        if (
            not isinstance(artifact, dict)
            or artifact.get("git_mode") != mode
            or artifact.get("git_kind") != kind
            or artifact.get("git_blob_oid") != oid
            or artifact.get("observed_git_blob_oid") != oid
            or artifact.get("stable_read") is not True
            or artifact.get("matches_git_revision") is not True
            or not _nonnegative_int(artifact.get("size"))
            or not _nonnegative_int(artifact.get("mtime_ns"))
            or not _nonnegative_int(artifact.get("ctime_ns"))
            or not _nonnegative_int(artifact.get("device"))
            or not _positive_int(artifact.get("inode"))
            or not _valid_sha256(artifact.get("sha256"))
            or not isinstance(artifact.get("path"), str)
        ):
            return False
        surface_rows.append([relative_path, oid, artifact["sha256"]])
    surface_digest = hashlib.sha256(json.dumps(
        surface_rows, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")).hexdigest()
    if surface_digest != proof.get("tracked_python_surface_sha256"):
        return False
    for module, row in source_only["verified_modules"].items():
        if (
            not isinstance(module, str)
            or not isinstance(row, dict)
            or not isinstance(row.get("repo_relative_path"), str)
            or row["repo_relative_path"] not in files
            or row.get("sha256") != files[row["repo_relative_path"]].get("sha256")
            or row.get("execution") not in (
                "compiled_authenticated_source_bytes",
                "compiled_authenticated_bootstrap_source_bytes",
            )
            or row.get("bytecode_cache_read") is not False
        ):
            return False
    loaded_modules = source_only["loaded_project_modules"].get("loaded_modules")
    expected_loaded = set(source_only["verified_modules"]) | set(_NATIVE_MODULES)
    if not isinstance(loaded_modules, list) or set(loaded_modules) != expected_loaded:
        return False
    for module, row in source_only["verified_native_modules"].items():
        authorized = source_only["authorized_native_artifacts"].get(module)
        if (
            not isinstance(row, dict)
            or not isinstance(authorized, dict)
            or row.get("execution") != "authenticated_canonical_extension_loader"
            or row.get("preimport_artifact_authenticated") is not True
            or not isinstance(row.get("path"), str)
            or row.get("lexical_path") != row.get("path")
            or not _positive_int(row.get("size"))
            or not _nonnegative_int(row.get("mtime_ns"))
            or not _nonnegative_int(row.get("ctime_ns"))
            or not _nonnegative_int(row.get("device"))
            or not _positive_int(row.get("inode"))
            or not _valid_sha256(row.get("sha256"))
            or authorized.get("stable_read") is not True
            or any(
                row.get(name) != authorized.get(name)
                for name in (
                    "path", "lexical_path", "size", "mtime_ns", "ctime_ns",
                    "device", "inode", "sha256", "stable_read",
                )
            )
        ):
            return False
    for check_name in ("start_check", "post_import_check", "post_run_check"):
        check = proof.get(check_name)
        if (
            not isinstance(check, dict)
            or check.get("passed") is not True
            or check.get("changed") != []
            or check.get("git_sha") != producer_sha
            or check.get("tracked_python_file_count") != len(files)
            or check.get("tracked_python_surface_sha256") != surface_digest
        ):
            return False
    return True


def _producer_native_imports_match_manifest(
    proof: Any, manifest: Mapping[str, Any],
) -> bool:
    """Tie pre-import native authorization to the published extension records."""
    if not isinstance(proof, dict):
        return False
    source_only = proof.get("source_only_import")
    if not isinstance(source_only, dict):
        return False
    authorized = source_only.get("authorized_native_artifacts")
    verified = source_only.get("verified_native_modules")
    if not isinstance(authorized, dict) or not isinstance(verified, dict):
        return False
    manifest_names = {
        "chess_anti_engine.encoding._features_ext": "features_extension",
        "chess_anti_engine.encoding._lc0_ext": "lc0_extension",
        "chess_anti_engine.mcts._mcts_tree": "mcts_extension",
    }
    identity_names = (
        "path", "lexical_path", "size", "mtime_ns", "ctime_ns", "device",
        "inode", "sha256", "stable_read",
    )
    for module, manifest_name in manifest_names.items():
        initial = authorized.get(module)
        loaded = verified.get(module)
        published = manifest.get(manifest_name)
        if (
            not isinstance(initial, dict)
            or not isinstance(loaded, dict)
            or not isinstance(published, dict)
        ):
            return False
        if any(
            initial.get(name) != loaded.get(name)
            or loaded.get(name) != published.get(name)
            for name in identity_names
        ):
            return False
    return True


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
            "model_input_consumption": manifest.get("model_input_consumption"),
            "params_candidate_inventory_sha256": (
                manifest.get("params_candidate_inventory", {}).get("inventory_sha256")
                if isinstance(manifest.get("params_candidate_inventory"), dict)
                else None
            ),
            "audit_set_sha256": artifact_sha("audit_set"),
            "matched_rows_sha256": artifact_sha("matched_rows"),
            "matched_rows_report_sha256": artifact_sha("matched_rows_report"),
            "matched_rows_snapshot_inventory_sha256": (
                manifest.get("matched_row_origin_verification", {})
                .get("snapshot_inventory", {})
                .get("inventory_sha256")
                if isinstance(manifest.get("matched_row_origin_verification"), dict)
                else None
            ),
            "max_positions": manifest.get("requested_max_positions"),
            "panel_selection": manifest.get("panel_selection"),
            "deep_reference_evidence": manifest.get("deep_reference_evidence"),
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


def _native_build_matches_revision(
    artifact: Any, producer_git_sha: Any, module: str,
) -> bool:
    """Independently recompute the embedded native-input stamp from Git bytes."""
    if not isinstance(artifact, dict) or not isinstance(producer_git_sha, str):
        return False
    build_attestation = artifact.get("build_attestation")
    if not isinstance(build_attestation, dict):
        return False
    schema = build_attestation.get("schema")
    if not isinstance(schema, str):
        return False
    dependency_bytes: dict[str, bytes] = {}
    try:
        dependencies = native_build_dependency_paths(schema, module)
    except ValueError:
        return False
    for relative_path in dependencies:
        committed = _git_file_at_commit(producer_git_sha, relative_path)
        if committed is None:
            return False
        dependency_bytes[relative_path] = committed
    try:
        expected = native_build_attestation(
            module, producer_git_sha, dependency_bytes, schema=schema,
        )
    except ValueError:
        return False
    return build_attestation == {
        **expected,
        "current_inputs_match_revision": True,
        "matches_producer_revision": True,
    }


def _compatible_native_extension(artifact: Any, producer_git_sha: Any) -> bool:
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
        and _native_build_matches_revision(
            artifact, producer_git_sha, "chess_anti_engine.mcts._mcts_tree",
        )
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


def _compatible_lc0_extension(artifact: Any, producer_git_sha: Any) -> bool:
    return bool(
        _artifact_provenance_complete(artifact)
        and _native_build_matches_revision(
            artifact, producer_git_sha, "chess_anti_engine.encoding._lc0_ext",
        )
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


def _compatible_features_extension(artifact: Any, producer_git_sha: Any) -> bool:
    return bool(
        _artifact_provenance_complete(artifact)
        and _native_build_matches_revision(
            artifact, producer_git_sha, "chess_anti_engine.encoding._features_ext",
        )
        and str(artifact.get("path", "")).endswith((".so", ".pyd"))
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


def _canonical_manifest_syzygy_paths(value: Any) -> tuple[str, ...]:
    """Return exact canonical absolute components, or empty on any ambiguity."""
    if not isinstance(value, str) or not value:
        return ()
    components = value.split(os.pathsep)
    if not components:
        return ()
    for component in components:
        if (
            not component
            or "\0" in component
            or not os.path.isabs(component)
            or component.startswith("//")
            or os.path.expanduser(component) != component
            or os.path.normpath(component) != component
            or str(Path(component)) != component
        ):
            return ()
    return tuple(components)


def _valid_tablebase_directory_inventory(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    root_identity = value.get("root_identity")
    path_components = value.get("path_components")
    file_identities = value.get("file_identities")
    approved_layout = value.get("approved_layout")
    if (
        not isinstance(value.get("path"), str)
        or not value.get("path")
        or not isinstance(root_identity, dict)
        or not _nonnegative_int(root_identity.get("device"))
        or not _positive_int(root_identity.get("inode"))
        or not _nonnegative_int(root_identity.get("mtime_ns"))
        or not _nonnegative_int(root_identity.get("ctime_ns"))
        or value.get("path_component_identity_fields")
        != list(_TABLEBASE_PATH_COMPONENT_IDENTITY_FIELDS)
        or not isinstance(path_components, list)
        or not path_components
        or value.get("file_identity_fields")
        != list(_TABLEBASE_FILE_IDENTITY_FIELDS)
        or not isinstance(file_identities, list)
        or value.get("file_identity_count") != len(file_identities)
        or not _positive_int(value.get("rtbw_count"))
        or not _positive_int(value.get("rtbz_count"))
        or not _positive_int(value.get("total_bytes"))
        or not _valid_sha256(value.get("inventory_sha256"))
        or not isinstance(approved_layout, dict)
    ):
        return False
    expected_parent: Path | None = None
    for index, component in enumerate(path_components):
        component_path = component.get("path") if isinstance(component, dict) else None
        if (
            not isinstance(component, dict)
            or not isinstance(component_path, str)
            or not component_path
            or not Path(component_path).is_absolute()
            or str(Path(component_path)) != component_path
            or not _nonnegative_int(component.get("device"))
            or not _positive_int(component.get("inode"))
            or not _nonnegative_int(component.get("mtime_ns"))
            or not _nonnegative_int(component.get("ctime_ns"))
        ):
            return False
        current = Path(component_path)
        if index == 0:
            if current != Path(os.sep):
                return False
        elif expected_parent is None or current.parent != expected_parent:
            return False
        expected_parent = current
    if (
        expected_parent != Path(str(value["path"]))
        or any(
            path_components[-1].get(field) != root_identity.get(field)
            for field in ("device", "inode", "mtime_ns", "ctime_ns")
        )
    ):
        return False
    seen_names: set[str] = set()
    wdl_count = 0
    dtz_count = 0
    total_bytes = 0
    for row in file_identities:
        if not isinstance(row, list) or len(row) != len(_TABLEBASE_FILE_IDENTITY_FIELDS):
            return False
        name, size, mtime_ns, ctime_ns, device, inode = row
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in seen_names
            or Path(name).suffix not in (".rtbw", ".rtbz")
            or not _positive_int(size)
            or not _nonnegative_int(mtime_ns)
            or not _nonnegative_int(ctime_ns)
            or not _nonnegative_int(device)
            or not _positive_int(inode)
        ):
            return False
        seen_names.add(name)
        if Path(name).suffix == ".rtbw":
            wdl_count += 1
        else:
            dtz_count += 1
        total_bytes += int(size)
    if (
        value["file_identity_count"] != wdl_count + dtz_count
        or value["rtbw_count"] != wdl_count
        or value["rtbz_count"] != dtz_count
        or value["total_bytes"] != total_bytes
    ):
        return False
    expected_by_name = {
        component.directory_name: component
        for component in _APPROVED_SYZYGY_COMPONENTS
    }
    component_name = Path(str(value["path"])).name
    expected = expected_by_name.get(component_name)
    portable_digest = filename_size_sha256(
        (str(row[0]), int(row[1])) for row in file_identities
    )
    if expected is None or approved_layout != {
        "schema": APPROVED_SYZYGY_LAYOUT_SCHEMA,
        "component": component_name,
        "canonical_encoding": APPROVED_SYZYGY_FILENAME_SIZE_ENCODING,
        "rtbw_count": expected.rtbw_count,
        "rtbz_count": expected.rtbz_count,
        "file_count": expected.file_count,
        "total_bytes": expected.total_bytes,
        "filename_size_sha256": expected.filename_size_sha256,
        "passed": True,
    } or portable_digest != expected.filename_size_sha256:
        return False
    identity_document = json.dumps(
        {
            "root_identity": root_identity,
            "path_component_identity_fields": list(
                _TABLEBASE_PATH_COMPONENT_IDENTITY_FIELDS
            ),
            "path_components": path_components,
            "file_identity_fields": list(_TABLEBASE_FILE_IDENTITY_FIELDS),
            "file_identities": file_identities,
        },
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(identity_document).hexdigest() == value["inventory_sha256"]


def _valid_tablebase_checksum_catalog(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    entries = value.get("entries")
    if (
        value.get("schema") != "deepfin.syzygy_checksum_catalog.v1"
        or value.get("component") != _APPROVED_SYZYGY_COMPONENTS[-1].directory_name
        or value.get("name") != APPROVED_SYZYGY_CHECKSUM_CATALOG_NAME
        or value.get("size") != APPROVED_SYZYGY_CHECKSUM_CATALOG_SIZE
        or not _nonnegative_int(value.get("mtime_ns"))
        or not _nonnegative_int(value.get("ctime_ns"))
        or not _nonnegative_int(value.get("device"))
        or not _positive_int(value.get("inode"))
        or value.get("raw_sha256")
        != APPROVED_SYZYGY_CHECKSUM_CATALOG_RAW_SHA256
        or value.get("algorithm") != "md5"
        or value.get("entry_count")
        != APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRY_COUNT
        or value.get("rtbw_count")
        != APPROVED_SYZYGY_CHECKSUM_CATALOG_WDL_COUNT
        or value.get("rtbz_count")
        != APPROVED_SYZYGY_CHECKSUM_CATALOG_DTZ_COUNT
        or value.get("canonical_entries_sha256")
        != APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRIES_SHA256
        or value.get("approved") is not True
        or not isinstance(entries, list)
        or len(entries) != APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRY_COUNT
    ):
        return False
    parsed: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    for row in entries:
        if not isinstance(row, list) or len(row) != 2:
            return False
        name, digest = row
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or Path(name).suffix not in (".rtbw", ".rtbz")
            or name in seen_names
            or not isinstance(digest, str)
            or len(digest) != 32
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            return False
        seen_names.add(name)
        parsed.append((name, digest))
    return bool(
        parsed == sorted(parsed)
        and sum(name.endswith(".rtbw") for name, _digest in parsed)
        == APPROVED_SYZYGY_CHECKSUM_CATALOG_WDL_COUNT
        and sum(name.endswith(".rtbz") for name, _digest in parsed)
        == APPROVED_SYZYGY_CHECKSUM_CATALOG_DTZ_COUNT
        and checksum_catalog_entries_sha256(parsed)
        == APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRIES_SHA256
    )


def _valid_tablebase_content_verification(
    value: Any, directories: list[dict[str, Any]], catalog: dict[str, Any],
) -> bool:
    if not isinstance(value, dict):
        return False
    entries = catalog.get("entries")
    if not isinstance(entries, list):
        return False
    expected_md5 = {
        str(row[0]): str(row[1])
        for row in entries
        if isinstance(row, list) and len(row) == 2
    }
    try:
        verification_rows = [
            [
                str(directory["approved_layout"]["component"]),
                str(identity[0]),
                int(identity[1]),
                int(identity[2]),
                int(identity[3]),
                int(identity[4]),
                int(identity[5]),
                expected_md5[str(identity[0])],
            ]
            for directory in directories
            for identity in directory["file_identities"]
        ]
    except (KeyError, TypeError, ValueError):
        return False
    document = json.dumps(
        verification_rows, sort_keys=False, ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return value == {
        "schema": "deepfin.syzygy_content_verification.v1",
        "method": "single_pass_md5_against_approved_catalog",
        "identity_binding_fields": [
            "component", "name", "size", "mtime_ns", "ctime_ns", "device",
            "inode", "approved_md5",
        ],
        "file_count": sum(int(row["file_identity_count"]) for row in directories),
        "bytes_hashed": sum(int(row["total_bytes"]) for row in directories),
        "file_identity_checksum_sha256": hashlib.sha256(document).hexdigest(),
        "passed": True,
    }


def _valid_tablebase_inventory(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    directories = value.get("directories")
    checksum_catalog = value.get("checksum_catalog")
    expected_component_order = [
        component.directory_name for component in _APPROVED_SYZYGY_COMPONENTS
    ]
    if (
        value.get("schema") != _TABLEBASE_INVENTORY_SCHEMA
        or value.get("identity_method")
        != (
            "approved_filename_size_plus_no_follow_path_components_and_"
            "file_device_inode_size_mtime_ctime"
        )
        or value.get("path_anchor_semantics")
        != "absolute_root_and_each_lexical_directory_component_no_follow"
        or not isinstance(directories, list)
        or len(directories) != len(_APPROVED_SYZYGY_COMPONENTS)
        or not all(_valid_tablebase_directory_inventory(row) for row in directories)
        or value.get("approved_layout_schema") != APPROVED_SYZYGY_LAYOUT_SCHEMA
        or value.get("approved_component_order") != expected_component_order
        or value.get("approved_layout_passed") is not True
        or value.get("checksum_catalog_covers_logical_table_names") is not True
        or not _valid_tablebase_checksum_catalog(checksum_catalog)
        or not isinstance(checksum_catalog, dict)
        or not _valid_tablebase_content_verification(
            value.get("content_verification"), directories, checksum_catalog,
        )
        or not _valid_sha256(value.get("inventory_sha256"))
    ):
        return False
    observed_names = {
        str(identity[0])
        for directory in directories
        for identity in directory["file_identities"]
    }
    catalog_names = {
        str(row[0]) for row in checksum_catalog["entries"]
        if isinstance(row, list) and len(row) == 2
    }
    if observed_names != catalog_names:
        return False
    payload = dict(value)
    payload.pop("inventory_sha256", None)
    document = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(document).hexdigest() == value["inventory_sha256"]


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
    """Return the game cluster; one buffered game may span adjacent shards."""
    source_dir = row.get("source_dir")
    shard = row.get("shard")
    game_id = row.get("game_id")
    if (
        not isinstance(source_dir, str) or not source_dir
        or not isinstance(shard, str) or not shard
        or not _nonnegative_int(game_id)
    ):
        return None
    return "\0".join((source_dir, str(game_id)))


def _matched_origin_proofs(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate banked replay readback and recompute every game grouping claim."""
    verification = manifest.get("matched_row_origin_verification")
    report_artifact = manifest.get("matched_rows_report")
    if not isinstance(verification, dict):
        raise ValueError("matched-row origin verification is missing")
    if (
        verification.get("schema") != "deepfin.matched_audit_rows.v1"
        or verification.get("passed") is not True
    ):
        raise ValueError("matched-row origin verification did not pass")
    banked_report = verification.get("report")
    if (
        not _artifact_provenance_complete(report_artifact)
        or not _artifact_provenance_complete(banked_report)
        or not isinstance(report_artifact, dict)
        or not isinstance(banked_report, dict)
        or any(
            report_artifact.get(field) != banked_report.get(field)
            for field in ("path", "size", "sha256")
        )
    ):
        raise ValueError("matched-row report identity is not bound to origin verification")
    report_builder = verification.get("report_builder")
    builder_script = (
        report_builder.get("script") if isinstance(report_builder, dict) else None
    )
    builder_sha = (
        report_builder.get("git_sha") if isinstance(report_builder, dict) else None
    )
    if (
        not isinstance(report_builder, dict)
        or report_builder.get("git_dirty") is not False
        or not isinstance(builder_script, dict)
        or builder_script.get("repo_relative_path") != "scripts/match_audit_rows.py"
        or builder_script.get("matches_git_revision") is not True
        or not isinstance(builder_sha, str)
    ):
        raise ValueError("matched-row builder revision provenance is incomplete")
    committed_builder = _git_file_at_commit(builder_sha, "scripts/match_audit_rows.py")
    if (
        committed_builder is None
        or builder_script.get("size") != len(committed_builder)
        or builder_script.get("sha256")
        != hashlib.sha256(committed_builder).hexdigest()
    ):
        raise ValueError("matched-row builder source is not bound to its Git revision")
    for proof_field, manifest_field in (
        ("report_audit_set", "audit_set"),
        ("report_output", "matched_rows"),
    ):
        proof_artifact = verification.get(proof_field)
        manifest_artifact = manifest.get(manifest_field)
        if (
            not isinstance(proof_artifact, dict)
            or not isinstance(manifest_artifact, dict)
            or any(
                proof_artifact.get(field) != manifest_artifact.get(field)
                for field in ("path", "size", "sha256")
            )
        ):
            raise ValueError("matched-row report input/output bindings are inconsistent")
    if verification.get("report_input_stability") != {
        "audit_set_unchanged": True,
        "snapshot_unchanged": True,
        "builder_checkout_unchanged": True,
    }:
        raise ValueError("matched-row report did not prove stable builder inputs")
    snapshot = verification.get("snapshot_inventory")
    root_identity = snapshot.get("root_identity") if isinstance(snapshot, dict) else None
    if (
        not isinstance(snapshot, dict)
        or not isinstance(snapshot.get("path"), str)
        or not snapshot.get("path")
        or not isinstance(root_identity, dict)
        or any(
            not _nonnegative_int(root_identity.get(field))
            for field in ("device", "inode", "mtime_ns", "ctime_ns")
        )
        or not _positive_int(snapshot.get("shard_count"))
        or not _valid_sha256(snapshot.get("inventory_sha256"))
        or not isinstance(snapshot.get("shards"), list)
        or len(snapshot["shards"]) != snapshot["shard_count"]
    ):
        raise ValueError("matched-row replay snapshot inventory is incomplete")
    shard_names: set[str] = set()
    for shard in snapshot["shards"]:
        if (
            not isinstance(shard, dict)
            or not isinstance(shard.get("name"), str)
            or not str(shard["name"]).endswith(".zarr")
            or shard["name"] in shard_names
            or not _nonnegative_int(shard.get("device"))
            or not _nonnegative_int(shard.get("inode"))
            or not _nonnegative_int(shard.get("mtime_ns"))
            or not _nonnegative_int(shard.get("ctime_ns"))
            or not _nonnegative_int(shard.get("file_count"))
            or not _nonnegative_int(shard.get("total_bytes"))
            or not _valid_sha256(shard.get("entries_identity_sha256"))
        ):
            raise ValueError("matched-row replay shard inventory is malformed")
        shard_names.add(str(shard["name"]))
    inventory_document = json.dumps({
        "root_identity": root_identity,
        "shards": snapshot["shards"],
    }, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(inventory_document).hexdigest() != snapshot["inventory_sha256"]:
        raise ValueError("matched-row replay snapshot inventory digest is inconsistent")
    rows = verification.get("rows")
    if not isinstance(rows, list):
        raise ValueError("matched-row origin proof rows are missing")
    result: dict[str, dict[str, Any]] = {}
    for proof in rows:
        if not isinstance(proof, dict):
            raise ValueError("matched-row origin proof is not an object")
        key = proof.get("key")
        source_dir = proof.get("source_dir")
        selected = proof.get("selected_origin")
        occurrences = proof.get("occurrences")
        if (
            not isinstance(key, str)
            or not key
            or key in result
            or source_dir != snapshot["path"]
            or not isinstance(selected, dict)
            or not isinstance(occurrences, list)
            or not occurrences
            or selected != occurrences[0]
            or proof.get("duplicate_count") != len(occurrences)
            or proof.get("source_cluster_ambiguous") is not False
            or proof.get("source_cluster_unique") is not True
        ):
            raise ValueError("matched-row selected-origin proof is inconsistent")
        selected_game = selected.get("game_id")
        if selected.get("has_game_id") is not True or not _nonnegative_int(selected_game):
            raise ValueError(f"{key}: selected replay origin lacks a source game id")
        for occurrence in occurrences:
            if (
                not isinstance(occurrence, dict)
                or occurrence.get("position_key") != key
                or not isinstance(occurrence.get("shard"), str)
                or occurrence.get("shard") not in shard_names
                or not _nonnegative_int(occurrence.get("row"))
                or not _valid_sha256(occurrence.get("stored_x_sha256"))
                or occurrence.get("has_game_id") is not True
                or occurrence.get("game_id") != selected_game
            ):
                raise ValueError(f"{key}: duplicate-origin proof does not establish one game")
        result[key] = proof
    expected_count = verification.get("selected_position_count")
    expected_digest = verification.get("selected_position_keys_sha256")
    actual_digest = hashlib.sha256(json.dumps(
        sorted(result), ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    if expected_count != len(result) or expected_digest != actual_digest:
        raise ValueError("matched-row origin proof key inventory is inconsistent")
    return result


def _row_matches_origin_proof(row: Mapping[str, Any], proof: Mapping[str, Any]) -> bool:
    selected = proof.get("selected_origin")
    if not isinstance(selected, dict):
        return False
    source_dir = proof.get("source_dir")
    game_id = selected.get("game_id")
    return bool(
        row.get("key") == proof.get("key")
        and row.get("source_dir") == source_dir
        and row.get("shard") == selected.get("shard")
        and row.get("game_id") == game_id
        and row.get("group_id") == "\0".join((str(source_dir), str(game_id)))
    )


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
        raise ValueError(f"{key}: group_id is not (source_dir, game_id)")
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
    if not _positive_int(row.get("deep_reference_nodes")):
        raise ValueError(f"{key}: deep reference observed nodes must be positive")
    if not _positive_int(row.get("deep_reference_depth")):
        raise ValueError(f"{key}: deep reference depth must be a positive integer")
    if not _positive_int(row.get("deep_reference_scored_multipv")):
        raise ValueError(f"{key}: deep reference MultiPV count must be positive")
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
    required_multipv = min(_DEEP_REFERENCE_MIN_MULTIPV, board.legal_moves.count())
    if int(row["deep_reference_scored_multipv"]) != len(move_cp):
        raise ValueError(f"{key}: deep reference MultiPV count disagrees with move values")
    if len(move_cp) < required_multipv:
        raise ValueError(
            f"{key}: deep reference has {len(move_cp)} scored unique moves; "
            f"the frozen ruler requires {required_multipv}"
        )
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
    try:
        emitted_index = actions.index(int(row["emitted_action"]))
    except ValueError as exc:
        raise ValueError(f"{key}: emitted action is absent from root actions") from exc
    emitted_reference_listed = row.get("emitted_reference_listed")
    if (
        not isinstance(emitted_reference_listed, bool)
        or emitted_reference_listed is not expected_listed[emitted_index]
    ):
        raise ValueError(f"{key}: emitted-reference censoring flag is invalid")
    if not emitted_reference_listed:
        raise ValueError(
            f"{key}: emitted move is absent from the finite MultiPV deep reference; "
            "the decision label is censored"
        )
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
    meta_path: Path, methodology_smoke: bool,
    *,
    input_bytes: bytes,
    meta_bytes: bytes | None,
) -> tuple[dict[str, Any], bool, bool]:
    if meta_bytes is None:
        if methodology_smoke:
            return {}, False, False
        raise ValueError(
            f"decision-grade analysis requires {meta_path}; pass --methodology-smoke "
            "only to exercise the estimator on a legacy bank"
        )
    manifest = json.loads(meta_bytes)
    failures: list[str] = []
    if manifest.get("schema") != _SCHEMA:
        failures.append(f"schema={manifest.get('schema')!r}")
    if manifest.get("complete") is not True:
        failures.append("complete is not true")
    if manifest.get("analysis_scope") != "fixed_node_horizons_only":
        failures.append(f"analysis_scope={manifest.get('analysis_scope')!r}")
    if manifest.get("clock_conditioning_available") is not False:
        failures.append("clock scope is ambiguous")
    reference_censoring = manifest.get("reference_censoring")
    if not _valid_reference_censoring(reference_censoring):
        failures.append("reference censoring provenance is incomplete")
    elif (
        manifest.get("decision_grade") is True
        and reference_censoring.get("passed") is not True
    ):
        failures.append("finite-MultiPV censoring affects decision labels")
    deep_reference_evidence = manifest.get("deep_reference_evidence")
    if not _valid_deep_reference_evidence(deep_reference_evidence):
        failures.append("deep-reference ruler evidence is incomplete")
    elif (
        manifest.get("decision_grade") is True
        and deep_reference_evidence.get("passed") is not True
    ):
        failures.append("selected positions do not satisfy the deep-reference ruler")
    if manifest.get("elapsed_measurement") != {
        "kind": "callback_instrumented_wall_time",
        "usable_for_controller_or_cost_analysis": False,
    }:
        failures.append("elapsed-time instrumentation scope is ambiguous")
    output = manifest.get("output")
    if (
        not isinstance(output, dict)
        or output.get("sha256") != hashlib.sha256(input_bytes).hexdigest()
        or output.get("size") != len(input_bytes)
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
    if not _producer_preimport_matches_revision(
        manifest.get("python_preimport"), producer_sha,
    ):
        failures.append(
            "producer pre-import Python source provenance is incomplete"
        )
    preregistration_failures = _preregistered_design_failures(manifest)
    failures.extend(preregistration_failures)
    failures.extend(
        f"{name} artifact provenance is incomplete"
        for name in (
            "producer_script", "publication_helper", "checkpoint", "audit_set",
            "matched_rows", "matched_rows_report",
        )
        if not _artifact_provenance_complete(manifest.get(name))
    )
    failures.extend(
        f"{name} is not bound to the producer Git revision"
        for name, relative_path in (
            ("producer_script", "scripts/backtest_chunk_trajectory.py"),
            ("publication_helper", "scripts/chunk_trajectory_publication.py"),
        )
        if not _producer_source_matches_revision(
            manifest.get(name), producer_sha, relative_path,
        )
    )
    if not _producer_sources_match_revision(
        manifest.get("producer_sources"), producer_sha,
    ):
        failures.append("loaded producer Python sources are not bound to the revision")
    checkpoint_params = manifest.get("checkpoint_params")
    if "checkpoint_params" not in manifest or (
        checkpoint_params is not None
        and not _artifact_provenance_complete(checkpoint_params)
    ):
        failures.append("checkpoint architecture provenance is incomplete")
    if not _model_input_consumption_complete(manifest):
        failures.append(
            "checkpoint or params consumption is not bound to authenticated inputs"
        )
    mcts_extension = manifest.get("mcts_extension")
    if not _compatible_native_extension(mcts_extension, producer_sha):
        failures.append("native MCTS extension provenance is incomplete")
    if not _compatible_features_extension(
        manifest.get("features_extension"), producer_sha,
    ):
        failures.append("native feature encoding extension provenance is incomplete")
    if not _compatible_lc0_extension(manifest.get("lc0_extension"), producer_sha):
        failures.append("native CBoard encoding extension provenance is incomplete")
    if not _producer_native_imports_match_manifest(
        manifest.get("python_preimport"), manifest,
    ):
        failures.append(
            "native extension imports are not bound to their pre-import snapshots"
        )
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
    if manifest.get("game_group_kind") != "source_dir:game_id":
        failures.append(f"game_group_kind={manifest.get('game_group_kind')!r}")
    try:
        _matched_origin_proofs(manifest)
    except ValueError as exc:
        failures.append(str(exc))
    source_game_group_count = manifest.get("source_game_group_count")
    minimum_source_games = manifest.get("minimum_decision_grade_source_games")
    source_group_resolution_passed = manifest.get("source_group_resolution_passed")
    source_group_count_valid = _nonnegative_int(source_game_group_count)
    source_group_resolution_expected = bool(
        source_group_count_valid
        and int(source_game_group_count) >= _MIN_DECISION_GRADE_SOURCE_GAMES
    )
    if (
        not source_group_count_valid
        or minimum_source_games != _MIN_DECISION_GRADE_SOURCE_GAMES
        or source_group_resolution_passed is not source_group_resolution_expected
    ):
        failures.append("source-game resolution provenance is inconsistent")
    elif (
        manifest.get("decision_grade") is True
        and source_group_resolution_passed is not True
    ):
        failures.append("decision-grade bank has insufficient source-game resolution")
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
    syzygy_paths = _canonical_manifest_syzygy_paths(
        syzygy.get("path") if isinstance(syzygy, dict) else None
    )
    directory_paths = (
        tuple(str(row.get("path", "")) for row in directories)
        if isinstance(directories, list) and all(isinstance(row, dict) for row in directories)
        else ()
    )
    if (
        not _valid_tablebase_inventory(syzygy)
        or not isinstance(syzygy, dict)
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
    if (
        manifest.get("decision_grade") is True
        and _nonnegative_int(excluded_positions)
        and int(excluded_positions) > 0
    ):
        failures.append("decision-grade bank contains selected-position exclusions")
    if not _valid_panel_selection(
        manifest.get("panel_selection"),
        requested_max_positions=requested_max_positions,
        requested_position_count=requested_positions,
    ):
        failures.append("audit panel selection provenance is inconsistent")
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
        "deep_reference_nodes", "deep_reference_depth",
        "deep_reference_scored_multipv",
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
    input_bytes, input_artifact = _read_consumed_artifact(
        input_path, role="trajectory_bank",
    )
    meta_artifact: dict[str, Any] | None = None
    try:
        meta_bytes, meta_artifact = _read_consumed_artifact(
            actual_meta, role="trajectory_manifest",
        )
    except FileNotFoundError:
        meta_bytes = None
    manifest, decision_grade, preregistered_design = _require_manifest(
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
    if not methodology_smoke:
        censoring_details = [
            detail
            for key, trajectory in sorted(by_key.items())
            if (
                detail := _trajectory_reference_censoring(
                    key, sorted(trajectory, key=lambda row: int(row["chunk"])),
                )
            ) is not None
        ]
        recomputed_censoring = _reference_censoring_summary(censoring_details)
        if recomputed_censoring != manifest.get("reference_censoring"):
            raise ValueError(
                "finite-MultiPV reference censoring disagrees with raw trajectory rows"
            )
        reference_positions = [
            trajectory[0] for trajectory in by_key.values()
        ]
        excluded_reference_positions = manifest.get("excluded_positions")
        if not isinstance(excluded_reference_positions, list):
            raise ValueError("deep-reference ruler exclusions are malformed")
        reference_positions.extend(excluded_reference_positions)
        audit_artifact = manifest.get("audit_set")
        recomputed_deep_reference = _deep_reference_evidence_summary(
            reference_positions,
            audit_set_sha256=(
                audit_artifact.get("sha256")
                if isinstance(audit_artifact, dict) else None
            ),
        )
        if recomputed_deep_reference != manifest.get("deep_reference_evidence"):
            raise ValueError(
                "deep-reference ruler evidence disagrees with raw selected-position rows"
            )
    excluded_keys = {
        str(entry["key"])
        for entry in manifest.get("excluded_positions", [])
        if isinstance(entry, dict) and "key" in entry
    }
    if not methodology_smoke and excluded_keys.intersection(by_key):
        raise ValueError("a trajectory is both completed and excluded")
    origin_proofs: dict[str, dict[str, Any]] = {}
    if not methodology_smoke:
        origin_proofs = _matched_origin_proofs(manifest)
        selected_keys = set(by_key) | excluded_keys
        if selected_keys != set(origin_proofs):
            raise ValueError(
                "matched-row origin proofs do not cover exactly the selected positions"
            )
        for entry in manifest.get("excluded_positions", []):
            key = str(entry["key"])
            if not _row_matches_origin_proof(entry, origin_proofs[key]):
                raise ValueError(f"{key}: excluded source group disagrees with row readback")
    if not methodology_smoke and not _panel_selection_matches_observations(
        manifest, by_key,
    ):
        raise ValueError(
            "unique-position source/phase counts disagree with panel selection provenance"
        )
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
        if (
            not methodology_smoke
            and any(
                not _row_matches_origin_proof(row, origin_proofs[key])
                for row in trajectory
            )
        ):
            raise ValueError(f"{key}: trajectory source group disagrees with row readback")
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
                    f"{key}: group_id is not (source_dir, game_id)"
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
        "reference_censoring": manifest.get("reference_censoring"),
        "analyzer_consumed_inputs": [
            artifact
            for artifact in (input_artifact, meta_artifact)
            if artifact is not None
        ],
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


def _evaluation_fold_ids(
    transitions: Sequence[Transition], n_folds: int,
) -> np.ndarray:
    """Assign every trajectory for a position to one deterministic game fold."""
    horizons, _keys, by_horizon = _rollout_layout(transitions)
    anchor_indices = by_horizon[horizons[0]]
    anchor_groups = [transitions[int(index)].group_id for index in anchor_indices]
    result = np.full(len(transitions), -1, dtype=np.int64)
    group_to_fold: dict[str, int] = {}
    for fold, local_indices in enumerate(grouped_folds(anchor_groups, n_folds)):
        for local_index in local_indices:
            group = anchor_groups[int(local_index)]
            group_to_fold[group] = fold
    for index, transition in enumerate(transitions):
        try:
            result[index] = group_to_fold[transition.group_id]
        except KeyError as error:
            raise ValueError(
                "every source game must occur at the anchor horizon"
            ) from error
    if (result < 0).any():
        raise AssertionError("evaluation fold assignment is incomplete")
    return result


def _grouped_analysis_preflight(
    transitions: Sequence[Transition], n_folds: int,
) -> dict[str, Any]:
    """Explain whether grouped held-horizon rollout analysis is structurally valid."""
    reasons: list[str] = []
    groups = np.asarray([transition.group_id for transition in transitions], dtype=str)
    if len(set(groups.tolist())) < 2:
        reasons.append("insufficient_source_game_groups")
    horizons = sorted({transition.horizon for transition in transitions})
    if len(horizons) < 2:
        reasons.append("insufficient_held_horizons")
    if transitions:
        try:
            _rollout_layout(transitions)
        except ValueError:
            reasons.append("nonrectangular_key_by_horizon_layout")
    if not reasons:
        for horizon in horizons:
            target_indices = np.flatnonzero(
                np.asarray([transition.horizon == horizon for transition in transitions])
            )
            try:
                folds = grouped_folds(groups[target_indices].tolist(), n_folds)
            except ValueError:
                reasons.append("insufficient_horizon_source_game_groups")
                break
            for test_local in folds:
                test_groups = set(groups[target_indices[test_local]].tolist())
                training_rows = sum(
                    transition.horizon != horizon
                    and transition.group_id not in test_groups
                    for transition in transitions
                )
                if training_rows < 2:
                    reasons.append("insufficient_held_horizon_training_rows")
                    break
            if reasons:
                break
    return {
        "passed": not reasons,
        "reasons": reasons,
        "source_game_group_count": len(set(groups.tolist())),
        "horizon_count": len(horizons),
    }


def _grouped_analysis_possible(
    transitions: Sequence[Transition], n_folds: int,
) -> bool:
    """Whether held-horizon grouped fitting has a valid rollout layout and splits."""
    return bool(_grouped_analysis_preflight(transitions, n_folds)["passed"])


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
    fold_ids = _evaluation_fold_ids(transitions, n_folds)
    for horizon in horizons:
        target_indices = np.flatnonzero(
            np.asarray([transition.horizon == horizon for transition in transitions])
        )
        for fold in sorted(set(fold_ids.tolist())):
            test = target_indices[fold_ids[target_indices] == fold]
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
                "fold": fold,
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


def _apportion_fold_count(
    count: int, fold_sizes: Sequence[int], capacities: Sequence[int],
) -> list[int]:
    """Split one global quota deterministically without exceeding fold capacity."""
    if len(fold_sizes) != len(capacities) or not fold_sizes:
        raise ValueError("fold sizes and capacities must be non-empty and aligned")
    if count < 0 or count > sum(capacities):
        raise ValueError("global quota exceeds the available fold capacity")
    total_positions = sum(fold_sizes)
    if total_positions <= 0:
        raise ValueError("fold-local allocation requires at least one position")
    ideals = [count * size / total_positions for size in fold_sizes]
    allocated = [
        min(int(capacity), math.floor(ideal))
        for ideal, capacity in zip(ideals, capacities, strict=True)
    ]
    while sum(allocated) < count:
        candidates = [
            fold for fold, capacity in enumerate(capacities)
            if allocated[fold] < capacity
        ]
        if not candidates:
            raise AssertionError("could not apportion the fold-local allocation quota")
        fold = max(
            candidates,
            key=lambda index: (ideals[index] - allocated[index], -index),
        )
        allocated[fold] += 1
    return allocated


def _fold_stage_counts(
    fold_sizes: Sequence[int], n_stages: int, fraction: float,
) -> tuple[list[int], list[list[int]]]:
    """Make nested per-fold quotas whose column sums equal the global schedule."""
    global_counts = _stage_counts(sum(fold_sizes), n_stages, fraction)
    capacities = list(fold_sizes)
    by_fold = [[0] * n_stages for _ in fold_sizes]
    for stage, count in enumerate(global_counts):
        allocated = _apportion_fold_count(count, fold_sizes, capacities)
        for fold, fold_count in enumerate(allocated):
            by_fold[fold][stage] = fold_count
        capacities = allocated
    if any(
        sum(by_fold[fold][stage] for fold in range(len(fold_sizes))) != count
        for stage, count in enumerate(global_counts)
    ):
        raise AssertionError("fold-local quotas do not preserve the global schedule")
    if any(
        later > earlier
        for counts in by_fold
        for earlier, later in pairwise(counts)
    ):
        raise AssertionError("fold-local quotas are not nested")
    return global_counts, by_fold


def _fold_partitions(
    transitions: Sequence[Transition], fold_ids: Sequence[int] | np.ndarray | None,
) -> list[np.ndarray]:
    """Return stable row partitions and reject a position split across folds."""
    if fold_ids is None:
        return [np.arange(len(transitions), dtype=np.int64)]
    folds = np.asarray(fold_ids, dtype=np.int64)
    if len(folds) != len(transitions) or (folds < 0).any():
        raise ValueError("fold ids must contain one non-negative id per transition")
    key_folds: dict[str, int] = {}
    for transition, fold in zip(transitions, folds, strict=True):
        existing = key_folds.setdefault(transition.key, int(fold))
        if existing != int(fold):
            raise ValueError("one position trajectory cannot be split across folds")
    return [
        np.flatnonzero(folds == fold)
        for fold in sorted(set(folds.tolist()))
    ]


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
    random_stage_gains: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Compare signed gains at each decision rung under reachable policies."""
    oracle_set = {int(index) for index in oracle_selected}
    m0_set = {int(index) for index in m0_selected}
    m1_set = {int(index) for index in m1_selected}
    diagnostics: list[dict[str, Any]] = []
    if random_stage_gains is not None and len(random_stage_gains) != len(horizons):
        raise ValueError("random stage gains do not match the rollout horizons")
    for stage, (horizon, count) in enumerate(
        zip(horizons, stage_counts, strict=True)
    ):
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
            float(random_stage_gains[stage])
            if random_stage_gains is not None
            else int(count) / n_positions
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
    fold_ids: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate nested policies without comparing scores from different fits."""
    horizons, keys, _by_horizon = _rollout_layout(transitions)
    if len(m0_scores) != len(transitions) or len(m1_scores) != len(transitions):
        raise ValueError("rollout scores do not match the trajectory bank")
    partitions = _fold_partitions(transitions, fold_ids)
    fold_sizes: list[int] = []
    fold_layouts: list[tuple[list[Transition], np.ndarray]] = []
    for indices in partitions:
        rows = [transitions[int(index)] for index in indices]
        fold_horizons, fold_keys, _ = _rollout_layout(rows)
        if fold_horizons != horizons:
            raise ValueError("every evaluation fold must contain every horizon")
        fold_sizes.append(len(fold_keys))
        fold_layouts.append((rows, indices))
    counts, fold_counts = _fold_stage_counts(
        fold_sizes, len(horizons), allocation_fraction,
    )
    random_stage_gains = [0.0] * len(horizons)
    relaxed_oracle_gain = 0.0
    oracle_gain = 0.0
    selected_by_policy: dict[str, list[np.ndarray]] = {
        "oracle": [], "complexity_predicate": [], "M0": [], "M1": [],
    }
    fold_diagnostics: list[dict[str, Any]] = []
    for fold, ((rows, indices), local_counts) in enumerate(
        zip(fold_layouts, fold_counts, strict=True)
    ):
        _fold_horizons, fold_keys, fold_by_horizon = _rollout_layout(rows)
        local_oracle, local_oracle_gain = _reachable_oracle_selected_indices(
            rows, local_counts,
        )
        local_m0 = _rollout_selected_indices(rows, m0_scores[indices], local_counts)
        local_m1 = _rollout_selected_indices(rows, m1_scores[indices], local_counts)
        local_complexity = _rollout_selected_indices(
            rows, np.zeros(len(rows)), local_counts, complexity=True,
        )
        for name, local_selected in (
            ("oracle", local_oracle),
            ("complexity_predicate", local_complexity),
            ("M0", local_m0),
            ("M1", local_m1),
        ):
            selected_by_policy[name].append(indices[local_selected])
        oracle_gain += local_oracle_gain
        fold_relaxed_gain = 0.0
        for stage, (count, horizon) in enumerate(
            zip(local_counts, horizons, strict=True)
        ):
            stage_indices = fold_by_horizon[horizon]
            gains = np.asarray([rows[int(index)].gain for index in stage_indices])
            random_stage_gains[stage] += float(count / len(fold_keys) * gains.sum())
            fold_relaxed_gain += float(
                gains[
                    _top_indices(
                        gains,
                        [rows[int(index)].key for index in stage_indices],
                        count,
                    )
                ].sum()
            )
        relaxed_oracle_gain += fold_relaxed_gain
        fold_diagnostics.append({
            "fold": fold,
            "n_positions": len(fold_keys),
            "stage_continue_counts": local_counts,
            "reachable_oracle_signed_gain": local_oracle_gain,
            "relaxed_oracle_signed_gain": fold_relaxed_gain,
        })
    selected = {
        name: np.concatenate(parts).astype(np.int64, copy=False)
        for name, parts in selected_by_policy.items()
    }
    random_gain = math.fsum(random_stage_gains)
    oracle_selected = selected["oracle"]
    m0_selected = selected["M0"]
    m1_selected = selected["M1"]
    complexity_selected = selected["complexity_predicate"]
    reachable_stages = _reachable_stage_diagnostics(
        transitions, horizons, counts, oracle_selected, m0_selected, m1_selected,
        n_positions=len(keys),
        random_stage_gains=random_stage_gains,
    )
    policies = {
        "random": {
            "selected": sum(counts),
            "spend": int(sum(counts) * transitions[0].cost),
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
    expected_selected = sum(counts)
    expected_spend = expected_selected * transitions[0].cost
    if any(
        policy["selected"] != expected_selected
        or policy["spend"] != expected_spend
        for policy in policies.values()
    ):
        raise AssertionError("reachable policies are not evaluated at matched spend")
    return {
        "n_positions": len(keys),
        "n_stages": len(horizons),
        "horizons": horizons,
        "stage_continue_counts": counts,
        "target_allocation_fraction": allocation_fraction,
        "realized_allocation_fraction": sum(counts) / (len(keys) * len(horizons)),
        "selection_semantics": (
            _FOLD_SELECTION_SEMANTICS
            if fold_ids is not None else "nested_prefix_no_reentry"
        ),
        "oracle_semantics": (
            _FOLD_ORACLE_SEMANTICS
            if fold_ids is not None else "exact_nested_stop_depth_assignment"
        ),
        "evaluation_fold_count": len(partitions),
        "fold_diagnostics": fold_diagnostics,
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
    fold_ids: Sequence[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Compare fixed-size tranches without pooling scores across fitted folds."""
    if len(m0_scores) != len(transitions) or len(m1_scores) != len(transitions):
        raise ValueError("scores do not match the horizon rows")
    natural_count = sum(row.complexity_continue for row in transitions)
    count = natural_count if selected_count is None else int(selected_count)
    if not 0 <= count <= len(transitions):
        raise ValueError("selected_count is outside the available positions")
    gains = np.asarray([row.gain for row in transitions], dtype=np.float64)
    partitions = _fold_partitions(transitions, fold_ids)
    fold_counts = _apportion_fold_count(
        count, [len(indices) for indices in partitions],
        [len(indices) for indices in partitions],
    )
    selected_by_policy: dict[str, list[np.ndarray]] = {
        "oracle": [], "complexity_predicate": [], "M0": [], "M1": [],
    }
    random_gain = 0.0
    for indices, fold_count in zip(partitions, fold_counts, strict=True):
        rows = [transitions[int(index)] for index in indices]
        keys = [row.key for row in rows]
        local_gains = gains[indices]
        random_gain += float(fold_count / len(rows) * local_gains.sum())
        for name, local in (
            ("oracle", _top_indices(local_gains, keys, fold_count)),
            (
                "complexity_predicate",
                _complexity_predicate_indices(rows, keys, fold_count),
            ),
            ("M0", _top_indices(m0_scores[indices], keys, fold_count)),
            ("M1", _top_indices(m1_scores[indices], keys, fold_count)),
        ):
            selected_by_policy[name].append(indices[local])
    selected = {
        name: np.concatenate(parts).astype(np.int64, copy=False)
        for name, parts in selected_by_policy.items()
    }
    oracle = selected["oracle"]
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
            transitions, selected["complexity_predicate"],
            random_gain=random_gain, oracle_gain=oracle_gain,
        )),
        "M0": asdict(_policy_result(
            transitions, selected["M0"],
            random_gain=random_gain, oracle_gain=oracle_gain,
        )),
        "M1": asdict(_policy_result(
            transitions, selected["M1"],
            random_gain=random_gain, oracle_gain=oracle_gain,
        )),
    }
    expected_spend = count * transitions[0].cost
    if any(
        policy["selected"] != count or policy["spend"] != expected_spend
        for policy in policies.values()
    ):
        raise AssertionError("horizon policies are not evaluated at matched spend")
    return {
        "n": len(transitions),
        "extension_fraction": count / len(transitions),
        "complexity_predicate_natural_extension_fraction": natural_count / len(transitions),
        "complexity_predicate_baseline": (
            "exact clock-free predicate selection"
            if fold_ids is None and count == natural_count
            else "fold-local clock-free predicate with deterministic tie-break"
            if fold_ids is not None
            else "clock-free predicate, then stability and emitted-gap tie-break"
        ),
        "selection_semantics": (
            "fold_local_matched_quota" if fold_ids is not None else "pooled_matched_quota"
        ),
        "evaluation_fold_count": len(partitions),
        "fold_selected_counts": fold_counts,
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
    *, fold_ids: Sequence[int] | np.ndarray | None = None,
) -> float | None:
    result = evaluate_reachable_rollout(
        transitions, m0, m1, allocation_fraction=allocation_fraction,
        fold_ids=fold_ids,
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


def _refit_fold_predictions(
    transitions: Sequence[Transition], fold_ids: Sequence[int] | np.ndarray,
    *, model: str, n_folds: int,
) -> np.ndarray:
    """Refit every held-fold/held-horizon model on a cluster resample."""
    folds = np.asarray(fold_ids, dtype=np.int64)
    if len(folds) != len(transitions):
        raise ValueError("bootstrap fold ids do not match the resampled bank")
    predictions = np.full(len(transitions), np.nan, dtype=np.float64)
    horizons = sorted({transition.horizon for transition in transitions})
    for fold in sorted(set(folds.tolist())):
        for horizon in horizons:
            test = np.flatnonzero([
                int(row_fold) == fold and transition.horizon == horizon
                for transition, row_fold in zip(transitions, folds, strict=True)
            ])
            train = np.flatnonzero([
                int(row_fold) != fold and transition.horizon != horizon
                for transition, row_fold in zip(transitions, folds, strict=True)
            ])
            if len(train) < 2 or not len(test):
                raise ValueError("bootstrap replicate cannot fit every held fold/horizon")
            train_rows = [transitions[int(index)] for index in train]
            test_rows = [transitions[int(index)] for index in test]
            alpha = _inner_alpha(train_rows, model, n_folds)
            fitted = _fit_ridge(
                _design(train_rows, model),
                np.asarray([row.gain for row in train_rows], dtype=np.float64),
                alpha,
            )
            predictions[test] = fitted.predict(_design(test_rows, model))
    if not np.isfinite(predictions).all():
        raise ValueError("bootstrap replicate left held-fold rows unpredicted")
    return predictions


def _resample_source_game_clusters(
    transitions: Sequence[Transition], rng: np.random.Generator,
) -> list[Transition]:
    """Globally resample source games while preserving complete trajectories."""
    indices_by_group: dict[str, list[int]] = {}
    for index, transition in enumerate(transitions):
        indices_by_group.setdefault(transition.group_id, []).append(index)
    groups = sorted(indices_by_group)
    if not groups:
        raise ValueError("cluster bootstrap requires at least one source game")
    drawn = rng.choice(groups, size=len(groups), replace=True)
    sampled: list[Transition] = []
    for occurrence, group in enumerate(drawn.tolist()):
        for index in indices_by_group[str(group)]:
            transition = transitions[index]
            sampled.append(replace(
                transition,
                # Each occurrence is a distinct sampled position trajectory,
                # while the source-game identity deliberately remains shared.
                # Recomputed grouped folds therefore cannot put identical
                # copies of one empirical cluster in train and test.
                key=f"{transition.key}\0bootstrap:{occurrence}",
            ))
    return sampled


@contextmanager
def _bootstrap_blas_limit() -> Generator[None, None, None]:
    """Keep tiny bootstrap ridge solves on one BLAS thread, then restore the pool."""
    try:
        from threadpoolctl import threadpool_limits
    except ModuleNotFoundError as error:
        if error.name != "threadpoolctl":
            raise
        raise RuntimeError(
            'controller bootstrap requires the analysis dependencies; install '
            '`pip install -e ".[analysis]"`'
        ) from error
    with threadpool_limits(limits=1, user_api="blas"):
        yield


def cluster_bootstrap_delta(
    transitions: Sequence[Transition],
    *,
    allocation_fraction: float,
    samples: int,
    seed: int,
    n_folds: int = 5,
    min_oracle_headroom: float = 0.0,
) -> dict[str, Any]:
    """Refit and bootstrap the fold-local worst-rung M1-minus-M0 gain.

    A replicate that cannot be fitted or lacks preregistered oracle headroom is
    part of the requested sampling distribution, not missing-at-random data.
    Such draws occupy the lower tail so the reported lower bound cannot become
    optimistic by conditioning on only the successful/headroom-eligible draws.
    """
    if samples < 1:
        raise ValueError("bootstrap requires at least one requested sample")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    invalid_samples = 0
    ineligible_samples = 0
    # The trajectory producer imports this module for shared validation but
    # never enters this lazy, scoped analysis-only dependency.
    with _bootstrap_blas_limit():
        for _ in range(samples):
            try:
                rows = _resample_source_game_clusters(transitions, rng)
                fold_ids = _evaluation_fold_ids(rows, n_folds)
                m0 = _refit_fold_predictions(
                    rows, fold_ids, model="M0", n_folds=n_folds,
                )
                m1 = _refit_fold_predictions(
                    rows, fold_ids, model="M1", n_folds=n_folds,
                )
                delta = _minimum_reachable_rung_gain_delta(
                    rows, m0, m1, allocation_fraction, min_oracle_headroom,
                    fold_ids=fold_ids,
                )
            except (ValueError, np.linalg.LinAlgError):
                invalid_samples += 1
                continue
            if delta is None:
                ineligible_samples += 1
            elif math.isfinite(delta):
                values.append(delta)
            else:
                invalid_samples += 1
    lower_tail_failure_samples = invalid_samples + ineligible_samples
    lower_tail_failure_fraction = lower_tail_failure_samples / samples
    valid_fraction = len(values) / samples

    def unconditional_quantile(quantile: float) -> float | None:
        # Model every invalid/ineligible requested draw as worse than every
        # numeric draw.  Once that mass reaches a requested quantile there is
        # no finite bound at that quantile; otherwise map the unconditional
        # rank into the conditional numeric sample without dropping the mass.
        if not values or lower_tail_failure_fraction >= quantile:
            return None
        conditional_quantile = (
            quantile - lower_tail_failure_fraction
        ) / valid_fraction
        return float(np.quantile(
            np.asarray(values, dtype=np.float64), conditional_quantile,
        ))

    if not values:
        return {
            "resampling_semantics": _BOOTSTRAP_RESAMPLING_SEMANTICS,
            "interval_semantics": _BOOTSTRAP_INTERVAL_SEMANTICS,
            "selection_semantics": _FOLD_SELECTION_SEMANTICS,
            "oracle_semantics": _FOLD_ORACLE_SEMANTICS,
            "requested_samples": samples,
            "valid_samples": 0,
            "invalid_samples": invalid_samples,
            "ineligible_samples": ineligible_samples,
            "lower_tail_failure_samples": lower_tail_failure_samples,
            "lower_tail_failure_fraction": lower_tail_failure_fraction,
            "valid_fraction": 0.0,
            "mean": None,
            "lower_95": None,
            "upper_95": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "resampling_semantics": _BOOTSTRAP_RESAMPLING_SEMANTICS,
        "interval_semantics": _BOOTSTRAP_INTERVAL_SEMANTICS,
        "selection_semantics": _FOLD_SELECTION_SEMANTICS,
        "oracle_semantics": _FOLD_ORACLE_SEMANTICS,
        "requested_samples": samples,
        "valid_samples": len(values),
        "invalid_samples": invalid_samples,
        "ineligible_samples": ineligible_samples,
        "lower_tail_failure_samples": lower_tail_failure_samples,
        "lower_tail_failure_fraction": lower_tail_failure_fraction,
        "valid_fraction": valid_fraction,
        "mean": float(array.mean()),
        "lower_95": unconditional_quantile(0.025),
        "upper_95": unconditional_quantile(0.975),
    }


def analyze(
    transitions: Sequence[Transition], *, n_folds: int, bootstrap_samples: int,
    seed: int, allocation_fraction: float, min_capture_gain: float,
    min_oracle_headroom: float, min_bootstrap_valid_fraction: float,
) -> dict[str, Any]:
    fold_ids = _evaluation_fold_ids(transitions, n_folds)
    m0, m0_diagnostics = held_horizon_predictions(transitions, "M0", n_folds=n_folds)
    m1, m1_diagnostics = held_horizon_predictions(transitions, "M1", n_folds=n_folds)
    horizons: dict[str, Any] = {}
    for horizon in sorted({row.horizon for row in transitions}):
        indices = np.flatnonzero([row.horizon == horizon for row in transitions])
        rows = [transitions[index] for index in indices]
        count = min(len(rows), max(1, round(allocation_fraction * len(rows))))
        result = evaluate_horizon(
            rows, m0[indices], m1[indices], selected_count=count,
            fold_ids=fold_ids[indices],
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
        fold_ids=fold_ids,
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
            "reachable_selection_required": _FOLD_SELECTION_SEMANTICS,
            "reachable_oracle_required": _FOLD_ORACLE_SEMANTICS,
            "bootstrap_resampling_required": _BOOTSTRAP_RESAMPLING_SEMANTICS,
            "bootstrap_interval_required": _BOOTSTRAP_INTERVAL_SEMANTICS,
            "fold_score_comparison": "within_held_out_fold_only",
            "per_horizon_tables_are_diagnostic_only": True,
            "M1_reachable_rollout_capture_above_random": True,
            "M1_minus_M0_reachable_signed_gain_positive_at_every_rung": True,
            "fold_refit_game_cluster_bootstrap_worst_rung_lower_95_above_zero": True,
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
        "bootstrap_fold_refit_worst_reachable_rung_M1_minus_M0_signed_gain": bootstrap,
        "reachable_rollout": rollout,
        "stage_conditional_diagnostics": horizons,
        "cv_diagnostics": {"M0": m0_diagnostics, "M1": m1_diagnostics},
    }


def _evidence_verdict(
    *, evidence_inputs_decision_grade: bool, canonical_preregistered_rule: bool,
    source_group_resolution_passed: bool, statistical_gate_passed: bool,
) -> str:
    """Name only the next experiment authorized by this deliberately narrow bank."""
    if not evidence_inputs_decision_grade:
        return "METHODOLOGY_SMOKE_ONLY"
    if not canonical_preregistered_rule:
        return "NONCANONICAL_RULE_DIAGNOSTIC_ONLY"
    if not source_group_resolution_passed:
        return "INSUFFICIENT_SOURCE_GAME_GROUPS"
    if statistical_gate_passed:
        return "ADVANCE_TO_CLOCK_HISTORY_REUSED_TREE_BANK"
    return "NO_ADVANCE_FROM_FRESH_TREE_FIXED_NODE_SCREEN"


def _decision_grade_evidence_inputs(
    *, bank_decision_grade: Any, analyzer_provenance: dict[str, Any],
) -> bool:
    """Authenticate frozen observations and estimator revision independently."""
    return bool(
        bank_decision_grade
        and analyzer_provenance.get("decision_grade") is True
    )


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
    analyzer_preimport_start_status = _preimport_python_surface_status(
        _PREIMPORT_PYTHON_SOURCES
    )
    if (
        not args.methodology_smoke
        and (
            _PREIMPORT_PYTHON_SOURCES.get("passed") is not True
            or analyzer_preimport_start_status.get("passed") is not True
        )
    ):
        raise SystemExit(
            "decision-grade controller analysis requires a clean tracked-Python "
            "snapshot taken before project imports; run the script as a fresh process "
            "from a clean checkout"
        )
    analyzer_start_sources = _analyzer_source_artifacts()
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
    consumed_inputs = info.get("analyzer_consumed_inputs")
    if not isinstance(consumed_inputs, list) or not all(
        isinstance(record, dict) for record in consumed_inputs
    ):
        if args.out is not None:
            raise RuntimeError("analyzer input consumption identity is unavailable")
        consumed_inputs = []
    _require_safe_output_path(
        args.input_path,
        actual_meta,
        args.out,
        manifest=manifest if isinstance(manifest, dict) else None,
        consumed_artifacts=consumed_inputs,
    )
    source_game_group_count = len({row.group_id for row in transitions})
    source_group_resolution_passed = (
        source_game_group_count >= _MIN_DECISION_GRADE_SOURCE_GAMES
    )
    result: dict[str, Any]
    grouped_analysis_preflight = _grouped_analysis_preflight(transitions, args.folds)
    grouped_analysis_possible = bool(grouped_analysis_preflight["passed"])
    if grouped_analysis_possible:
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
    else:
        reasons = grouped_analysis_preflight["reasons"]
        result = {
            "statistical_gate_passed": False,
            "scope": "fresh_tree_fixed_node_horizons_only",
            "clock_controller_authorized": False,
            "cross_move_tree_reuse_tested": False,
            "analysis_skipped": reasons[0] if reasons else "grouped_analysis_unavailable",
        }
    analyzer = _analyzer_provenance(
        analyzer_start_sources, analyzer_start_git_sha, analyzer_start_git_dirty,
        preimport_start_status=analyzer_preimport_start_status,
    )
    manifest_producer_sha = (
        manifest.get("producer_git_sha") if isinstance(manifest, dict) else None
    )
    analyzer_matches_producer_commit = bool(
        analyzer["git_sha"] == manifest_producer_sha
        and analyzer["final_git_sha"] == manifest_producer_sha
    )
    evidence_inputs_decision_grade = _decision_grade_evidence_inputs(
        bank_decision_grade=info["decision_grade"],
        analyzer_provenance=analyzer,
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
    decision_grade = bool(
        evidence_inputs_decision_grade
        and canonical_rule
        and source_group_resolution_passed
    )
    result["canonical_analysis_rule"] = canonical_analysis_rule
    result["canonical_preregistered_rule"] = canonical_rule
    result["analyzer_matches_producer_commit"] = analyzer_matches_producer_commit
    result["source_game_group_count"] = source_game_group_count
    result["grouped_analysis_possible"] = grouped_analysis_possible
    result["grouped_analysis_preflight"] = grouped_analysis_preflight
    result["reference_censoring"] = info.get("reference_censoring")
    result["minimum_decision_grade_source_games"] = _MIN_DECISION_GRADE_SOURCE_GAMES
    result["source_group_resolution_passed"] = source_group_resolution_passed
    result["evidence_decision_grade"] = decision_grade
    result["verdict"] = _evidence_verdict(
        evidence_inputs_decision_grade=evidence_inputs_decision_grade,
        canonical_preregistered_rule=canonical_rule,
        source_group_resolution_passed=source_group_resolution_passed,
        statistical_gate_passed=bool(result["statistical_gate_passed"]),
    )
    payload = {"input": info, "analyzer": analyzer, "analysis": result}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        output_manifest = manifest if isinstance(manifest, dict) else None
        with _anchored_output_target(
            args.input_path,
            actual_meta,
            args.out,
            manifest=output_manifest,
            consumed_artifacts=consumed_inputs,
        ) as output_target:
            _write_json_atomic(output_target, rendered)
    print(rendered)


if __name__ == "__main__":
    main()
