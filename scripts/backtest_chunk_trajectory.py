#!/usr/bin/env python3
"""Per-chunk search trajectory on the REAL accumulating tree, for the time-management
predictor: at chunk k, can we predict whether more chunks change/improve the move?

Runs the production SearchWorker path (walker PUCT at the shipped Threads=2 default)
one position at a time with abort disabled. Methodology-smoke runs may explicitly select
classic Gumbel with ``--walkers 1``, but those banks are never decision-grade. It
snapshots root state after every chunk via the run() on_chunk hook — the node-horizon
states the clock-free complexity predicate sees. Each row carries
the chosen move + its
deep-SF regret, plus the STATIC settledness (visit-gap, entropy, q-gap) and the DYNAMIC
"is the search still moving" signals (bestmove_flip / q_drift / visit_churn vs the
previous chunk). Downstream: label each chunk by whether the move changes / regret drops
in later chunks, and learn which features predict it — the abort/extend rule.

Unlike backtest_time_value.py (independent one-shot Gumbel per budget), this preserves
one accumulating tree. It does not contain a real clock or the production controller's
provable visit-lead branch, so a positive screen authorizes only a real-clock bank—not a
deployed controller. Slower (single board), so run a few hundred positions.

Usage:
  PYTHONPATH=. python3 scripts/backtest_chunk_trajectory.py --checkpoint <ckpt> \\
    --preregistration <tracked-plan.json> --device cuda --chunk-sims 2048 \
    --max-chunks 8 --max-positions 200

Decision-grade provenance requires this direct script form. ``python -m scripts...``
executes ``scripts/__init__.py`` before the entrypoint can snapshot project sources.
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
import threading
import time
import types
from collections import Counter, deque
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import IO, Any


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
        changed.append("producer_checkout_revision")
    return {
        "passed": not changed,
        "changed": sorted(set(changed)),
        "git_sha": current_git_sha,
        "tracked_python_file_count": len(files),
        "tracked_python_surface_sha256": snapshot.get(
            "tracked_python_surface_sha256"
        ),
    }

from scripts import chunk_trajectory_publication as publication_module

if __name__ == "__main__" and "--recover-publication" in sys.argv[1:]:
    publication_module.recover_publication_cli(
        sys.argv[1:], repo_root=Path(__file__).resolve().parents[1],
    )
    raise SystemExit(0)

# These binaries are imported indirectly by the project modules below. Capture
# their on-disk identity first, then prove after import that the mapped module's
# path and bytes still match this snapshot.
import chess
import numpy as np

from scripts.native_import_guard import artifact as _artifact
from scripts.native_import_guard import PREIMPORT_NATIVE_ARTIFACTS
_SOURCE_ONLY_IMPORT_GUARD.authorize_native(PREIMPORT_NATIVE_ARTIFACTS)
from scripts.chunk_trajectory_publication import (
    CHUNK_TRAJECTORY_SCHEMA,
    _acquire_output_lock as _acquire_output_lock,
    _acquire_output_locks,
    _durably_prepare_output_artifact,
    _entry_name_exists,
    _git_ignored_or_outside as _git_ignored_or_outside,
    _invalid_manifest_path as _invalid_manifest_path,
    _open_staged_output_file,
    _output_lock_path as _output_lock_path,
    _pending_manifest_path,
    _pending_output_path,
    _prepared_output_artifact as _prepared_output_artifact,
    _publish_evidence_pair,
    _publish_output as _publish_output,
    _require_new_output_pair,
    _require_safe_output_paths,
    _write_json_atomic,
    _write_json_staged as _write_json_staged,
    _write_invalid_recovery_diagnostic,
)
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
from scripts.repo_output_guard import repo_controlled_output, reserved_output_path
from scripts.match_audit_rows import (
    MATCHED_ROWS_REPORT_SCHEMA,
    default_matched_rows_report_path,
    snapshot_inventory as _matched_snapshot_inventory,
    verify_selected_row_origins,
)

from chess_anti_engine.eval.audit import (
    AuditPosition,
    legal_full_indices,
    move_regrets,
    parse_audit_record,
    phase_bucket,
)
from chess_anti_engine.eval.audit_history import MatchedAuditRows, default_matched_rows_path
from chess_anti_engine.encoding import input_plane_count
from chess_anti_engine.mcts.gumbel import (
    PLAY_PUCT_DEFAULTS,
    PLAY_SEARCH_DEFAULTS,
    PLAY_SEARCH_TARGET_BATCH,
    PLAY_SEARCH_VLOSS_WEIGHT,
    validate_gumbel_config,
)
from chess_anti_engine.mcts.search_options import SEARCH_OPTIONS
from chess_anti_engine.moves import (
    ActionDecodeError,
    POLICY_SIZE,
    index_to_move_strict,
)
from chess_anti_engine.uci.engine import EngineOptions
from chess_anti_engine.uci.search import (
    _ABORT_MIN_STABLE_CHUNKS,
    _ABORT_VISIT_GAP_MARGIN,
    _pv_from_root_action,
    _visit_gap,
)
from chess_anti_engine.utils.syzygy import SEPARATOR, default_syzygy_path, require_tablebases
from scripts.analyze_chunk_controller import (
    _APPROVED_AUDIT_SET_SHA256,
    _MIN_DECISION_GRADE_SOURCE_GAMES,
    _PRODUCTION_WALKERS,
    _canonical_cuda_device_string,
    _complexity_continue,
    _deep_reference_evidence_summary,
    _preregistration_payload,
    _preregistered_design_failures,
    _reference_censoring_summary,
    _score,
    _trajectory_reference_censoring,
    _update_stability,
)
from scripts.check_c_extensions_fresh import (
    NATIVE_BUILD_ATTESTATION_SCHEMA,
    check_extensions,
    native_build_attestation,
    native_build_dependency_paths,
    require_current_native_build_attestation_schema,
)

_SCHEMA = CHUNK_TRAJECTORY_SCHEMA
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
_MIN_DECISION_GRADE_CHUNKS = 4
_PANEL_SELECTION_STRATEGY = "joint_audit_source_phase_piece_round_robin_v1"
_PANEL_REQUIRED_SOURCES = (0, 1)
_PANEL_SOURCE_BALANCE_MAX_DIFFERENCE = 1
_MODEL_INPUT_CONSUMPTION_SCHEMA = "deepfin.model_input_consumption.v2"
_PARAMS_CANDIDATE_INVENTORY_SCHEMA = "deepfin.params_candidate_inventory.v1"
_PARAMS_SEARCH_LIMIT = 6
_REQUIRED_PRODUCER_SOURCE_MODULES = {
    "producer_script",
    "scripts.chunk_trajectory_publication",
    "scripts.analyze_chunk_controller",
    "scripts.repo_output_guard",
    "scripts.match_audit_rows",
    "scripts.approved_syzygy",
    "chess_anti_engine.eval.audit",
    "chess_anti_engine.mcts.search_options",
    "chess_anti_engine.uci.search",
    "chess_anti_engine.uci.model_loader",
}
_ACTIVE_PENDING_EVIDENCE: Any = None


def _panel_piece_bucket(piece_count: int) -> int:
    return min(32, max(2, int(piece_count))) // 4


def _panel_key_digest(keys: list[str]) -> str:
    payload = json.dumps(
        sorted(keys), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _panel_count_rows(
    counts: Mapping[tuple[int, int, int], int],
) -> list[dict[str, int]]:
    return [
        {
            "source": source,
            "phase": phase,
            "piece_bucket": piece_bucket,
            "count": int(count),
        }
        for (source, phase, piece_bucket), count in sorted(counts.items())
    ]


def _panel_source_count_rows(counts: Mapping[int, int]) -> list[dict[str, int]]:
    sources = sorted(set(_PANEL_REQUIRED_SOURCES) | set(counts))
    return [
        {"source": source, "count": int(counts.get(source, 0))}
        for source in sources
    ]


def _panel_position_metadata(position: AuditPosition) -> tuple[int, int, int, bool]:
    piece_count = chess.popcount(chess.Board(position.fen).occupied)
    realized_phase = phase_bucket(piece_count)
    return (
        int(position.source),
        int(position.phase),
        _panel_piece_bucket(piece_count),
        int(position.phase) == realized_phase,
    )


def _deep_reference_position_fields(position: AuditPosition) -> dict[str, Any]:
    """Raw frozen-ruler evidence copied into every banked position record."""
    return {
        "deep_reference_nodes": int(position.sf_nodes),
        "deep_reference_depth": int(position.sf_depth),
        "deep_reference_scored_multipv": len(position.move_cp),
        "deep_reference_best_cp": float(position.best_cp),
        "deep_reference_move_cp": {
            str(uci): float(cp) for uci, cp in position.move_cp.items()
        },
    }


def _round_robin_panel_source(
    buckets: Mapping[tuple[int, int, int], list[AuditPosition]],
) -> deque[AuditPosition]:
    ordered = {
        stratum: deque(sorted(
            positions,
            key=lambda position: (
                hashlib.sha256(position.key.encode("utf-8")).digest(),
                position.key,
            ),
        ))
        for stratum, positions in sorted(buckets.items())
    }
    result: deque[AuditPosition] = deque()
    while any(ordered.values()):
        for stratum in ordered:
            if ordered[stratum]:
                result.append(ordered[stratum].popleft())
    return result


def _select_audit_panel(
    positions: list[AuditPosition], limit: int,
) -> tuple[list[AuditPosition], dict[str, Any]]:
    """Select an order-independent panel jointly balanced by source and morphology."""
    position_metadata = [
        _panel_position_metadata(position) for position in positions
    ]
    strata: dict[tuple[int, int, int], list[AuditPosition]] = {}
    available_sources: Counter[int] = Counter()
    available_strata: Counter[tuple[int, int, int]] = Counter()
    for position, position_row in zip(positions, position_metadata, strict=True):
        source, phase, piece_bucket, _phase_matches = position_row
        stratum = (source, phase, piece_bucket)
        strata.setdefault(stratum, []).append(position)
        available_sources[source] += 1
        available_strata[stratum] += 1

    truncated = 0 < limit < len(positions)
    if not truncated:
        selected = positions
        selection_mode = "full_set"
    else:
        by_source: dict[int, dict[tuple[int, int, int], list[AuditPosition]]] = {}
        for stratum, bucket in strata.items():
            by_source.setdefault(stratum[0], {})[stratum] = bucket
        source_queues = {
            source: _round_robin_panel_source(source_buckets)
            for source, source_buckets in sorted(by_source.items())
        }
        selected = []
        while len(selected) < limit and any(source_queues.values()):
            for source in source_queues:
                if source_queues[source] and len(selected) < limit:
                    selected.append(source_queues[source].popleft())
        selection_mode = "truncated_joint_stratified"

    selected_sources: Counter[int] = Counter()
    selected_strata: Counter[tuple[int, int, int]] = Counter()
    for position in selected:
        source, phase, piece_bucket, _phase_matches = _panel_position_metadata(position)
        selected_sources[source] += 1
        selected_strata[(source, phase, piece_bucket)] += 1
    source_difference = abs(
        selected_sources[_PANEL_REQUIRED_SOURCES[0]]
        - selected_sources[_PANEL_REQUIRED_SOURCES[1]]
    )
    unique_keys = len({position.key for position in positions}) == len(positions)
    source_domain_passed = set(available_sources).issubset(_PANEL_REQUIRED_SOURCES)
    phase_morphology_passed = all(row[3] for row in position_metadata)
    source_balance_passed = bool(
        set(selected_sources).issubset(_PANEL_REQUIRED_SOURCES)
        and source_difference <= _PANEL_SOURCE_BALANCE_MAX_DIFFERENCE
    )
    decision_grade_passed = bool(
        selected
        and unique_keys
        and source_domain_passed
        and phase_morphology_passed
        and source_balance_passed
    )
    selection = {
        "strategy": _PANEL_SELECTION_STRATEGY,
        "selection_mode": selection_mode,
        "stratum_fields": ["source", "phase", "piece_bucket"],
        "piece_bucket_definition": "clamp_2_32_then_floor_divide_by_4",
        "within_stratum_order": "sha256_position_key_then_position_key",
        "source_order": list(_PANEL_REQUIRED_SOURCES),
        "requested_max_positions": int(limit),
        "available_position_count": len(positions),
        "selected_position_count": len(selected),
        "available_position_keys_sha256": _panel_key_digest(
            [position.key for position in positions]
        ),
        "selected_position_keys_sha256": _panel_key_digest(
            [position.key for position in selected]
        ),
        "available_keys_unique": unique_keys,
        "source_domain_passed": source_domain_passed,
        "phase_morphology_passed": phase_morphology_passed,
        "available_source_counts": _panel_source_count_rows(available_sources),
        "selected_source_counts": _panel_source_count_rows(selected_sources),
        "available_stratum_counts": _panel_count_rows(available_strata),
        "selected_stratum_counts": _panel_count_rows(selected_strata),
        "source_balance": {
            "maximum_difference": _PANEL_SOURCE_BALANCE_MAX_DIFFERENCE,
            "observed_difference": source_difference,
            "passed": source_balance_passed,
        },
        "decision_grade_passed": decision_grade_passed,
    }
    return selected, selection


def _require_decision_grade_panel_selection(
    selection: Mapping[str, Any], *, methodology_smoke: bool,
) -> None:
    if methodology_smoke or selection.get("decision_grade_passed") is True:
        return
    raise SystemExit(
        "decision-grade trajectory banks require a unique, source-balanced audit "
        "panel with valid source and phase morphology"
    )


def _entropy(shares: np.ndarray) -> float:
    p = shares[shares > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def _strict_uci_pv(board: chess.Board, pv_actions: list[int]) -> list[str]:
    current = board.copy(stack=False)
    moves: list[str] = []
    for action in pv_actions:
        if not 0 <= int(action) < POLICY_SIZE:
            raise ActionDecodeError(action, current, "outside the native action space")
        move = index_to_move_strict(action, current)
        moves.append(move.uci())
        current.push(move)
    return moves


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


def _producer_git_file_at_commit(commit: str, relative_path: str) -> bytes | None:
    """Read producer-repository bytes without trusting another project module."""
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


def _artifact_if_file(path: Path) -> dict[str, Any] | None:
    """Best-effort end-of-run snapshot; disappearance is evidence, not an abort."""
    try:
        if not path.expanduser().resolve().is_file():
            return None
        return _artifact(path, require_file=True)
    except OSError:
        return None


def _load_audit_set_snapshot(
    path: Path, *, require_approved: bool,
) -> tuple[list[AuditPosition], dict[str, Any]]:
    """Hash and parse one immutable snapshot of the audit-set bytes.

    The old flow hashed the path and then reopened it through ``load_audit_set``.
    A replace/read/restore race could therefore authenticate one file while using
    rows from another.  This function reads one stable file descriptor into an
    immutable ``bytes`` object; that exact object supplies both the digest and
    every parsed position.
    """
    try:
        resolved = path.expanduser().resolve(strict=True)
        with resolved.open("rb") as stream:
            before = os.fstat(stream.fileno())
            content = stream.read()
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise SystemExit(f"cannot read audit set {path}: {exc}") from exc
    identity = (
        int(before.st_mode), int(before.st_size), int(before.st_mtime_ns),
        int(before.st_ctime_ns), int(before.st_dev), int(before.st_ino),
    )
    stable_read = bool(
        identity == (
            int(after.st_mode), int(after.st_size), int(after.st_mtime_ns),
            int(after.st_ctime_ns), int(after.st_dev), int(after.st_ino),
        )
        and stat.S_ISREG(after.st_mode)
        and len(content) == after.st_size
    )
    if not stable_read:
        raise SystemExit("audit set changed while its immutable snapshot was read")
    digest = hashlib.sha256(content).hexdigest()
    if require_approved and digest != _APPROVED_AUDIT_SET_SHA256:
        raise SystemExit(
            "decision-grade trajectory banks require the approved frozen audit set "
            f"SHA256 {_APPROVED_AUDIT_SET_SHA256}; observed {digest}"
        )
    try:
        document = content.decode("utf-8", errors="strict")
        positions = [
            parse_audit_record(line)
            for raw_line in document.splitlines()
            if (line := raw_line.strip())
        ]
    except (UnicodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"cannot parse audit-set snapshot {resolved}: {exc}") from exc
    artifact = {
        "path": str(resolved),
        "size": len(content),
        "mtime_ns": int(after.st_mtime_ns),
        "ctime_ns": int(after.st_ctime_ns),
        "device": int(after.st_dev),
        "inode": int(after.st_ino),
        "sha256": digest,
        "stable_read": True,
        "consumption": "sha256_and_positions_from_same_immutable_byte_snapshot",
    }
    return positions, artifact


def _producer_python_source_artifacts(
    producer_git_sha: str, *, require_tracked: bool,
) -> dict[str, dict[str, Any]]:
    """Bind every loaded project module to the revision and pre-import bytes."""
    import_status = _SOURCE_ONLY_IMPORT_GUARD.status()
    loaded_status = import_status.get("loaded_project_modules")
    import_guard_passed = bool(
        import_status.get("active") is True
        and import_status.get("first_finder") is True
        and isinstance(loaded_status, dict)
        and loaded_status.get("passed") is True
        and loaded_status.get("unverified_modules") == []
    )
    if require_tracked and not import_guard_passed:
        unverified = (
            loaded_status.get("unverified_modules", [])
            if isinstance(loaded_status, dict) else []
        )
        names = ", ".join(
            str(row.get("module"))
            for row in unverified
            if isinstance(row, dict)
        )
        raise SystemExit(
            "decision-grade producer loaded an unauthenticated project module or "
            f"lost source-guard precedence: {names or 'unknown module'}"
        )
    repo_root = Path(__file__).resolve().parents[1]
    preimport_files = _PREIMPORT_PYTHON_SOURCES.get("files")
    if not isinstance(preimport_files, dict):
        preimport_files = {}
    source_paths = {"producer_script": Path(__file__)}
    for module_name, module in sorted(sys.modules.items()):
        if not (
            module_name in ("chess_anti_engine", "scripts")
            or module_name.startswith(("chess_anti_engine.", "scripts."))
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if isinstance(module_file, str) and Path(module_file).suffix == ".py":
            source_paths[module_name] = Path(module_file)
    artifacts: dict[str, dict[str, Any]] = {}
    unbound: list[str] = []
    for name, path in sorted(source_paths.items()):
        resolved = path.expanduser().resolve()
        try:
            relative_path = resolved.relative_to(repo_root).as_posix()
        except ValueError:
            relative_path = None
        artifact = _artifact(resolved, require_file=True)
        committed = (
            _producer_git_file_at_commit(producer_git_sha, relative_path)
            if relative_path is not None else None
        )
        matches_commit = bool(
            committed is not None
            and artifact.get("size") == len(committed)
            and artifact.get("sha256") == hashlib.sha256(committed).hexdigest()
        )
        preimport = (
            preimport_files.get(relative_path)
            if relative_path is not None else None
        )
        matches_preimport = bool(
            isinstance(preimport, dict)
            and artifact.get("path") == preimport.get("path")
            and artifact.get("size") == preimport.get("size")
            and artifact.get("mtime_ns") == preimport.get("mtime_ns")
            and artifact.get("ctime_ns") == preimport.get("ctime_ns")
            and artifact.get("device") == preimport.get("device")
            and artifact.get("inode") == preimport.get("inode")
            and artifact.get("sha256") == preimport.get("sha256")
        )
        source_record = _SOURCE_ONLY_IMPORT_GUARD.verified_modules.get(name)
        source_only_execution = (
            "entrypoint_trust_boundary"
            if name == "producer_script"
            else (
                source_record.get("execution")
                if isinstance(source_record, dict) else None
            )
        )
        source_only_verified = bool(
            name == "producer_script"
            or (
                relative_path is not None
                and _SOURCE_ONLY_IMPORT_GUARD.module_verified(name, relative_path)
            )
        )
        artifacts[name] = {
            **artifact,
            "repo_relative_path": relative_path,
            "matches_producer_git_revision": matches_commit,
            "matches_preimport_snapshot": matches_preimport,
            "source_only_import_verified": source_only_verified,
            "source_execution": source_only_execution,
        }
        if not matches_commit or not matches_preimport or not source_only_verified:
            unbound.append(name)
    if require_tracked and unbound:
        raise SystemExit(
            "decision-grade producer Python sources are not bound to the pre-import "
            f"producer revision snapshot: {', '.join(unbound)}"
        )
    return artifacts


def _artifact_identity(artifact: Any) -> dict[str, Any] | None:
    if not isinstance(artifact, dict):
        return None
    return {
        name: artifact.get(name)
        for name in (
            "path", "size", "mtime_ns", "ctime_ns", "device", "inode", "sha256",
        )
    }


def _stat_identity(file_stat: os.stat_result) -> tuple[int, ...]:
    return (
        int(file_stat.st_mode),
        int(file_stat.st_size),
        int(file_stat.st_mtime_ns),
        int(file_stat.st_ctime_ns),
        int(file_stat.st_dev),
        int(file_stat.st_ino),
    )


def _open_authenticated_input(path: Path) -> tuple[IO[bytes], dict[str, Any]]:
    """Open one regular input without following its final path component."""
    lexical = Path(os.path.abspath(path.expanduser()))
    descriptor: int | None = None
    try:
        lexical_stat = lexical.lstat()
        if stat.S_ISLNK(lexical_stat.st_mode):
            raise SystemExit(
                f"decision-grade model input cannot be a symlink: {lexical}"
            )
        nofollow = getattr(os, "O_NOFOLLOW", None)
        if nofollow is None:
            raise SystemExit("decision-grade model inputs require O_NOFOLLOW support")
        flags = os.O_RDONLY | os.O_CLOEXEC | nofollow
        descriptor = os.open(lexical, flags)
        opened = os.fstat(descriptor)
        post_open_lexical_stat = lexical.lstat()
        resolved = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
        resolved_stat = resolved.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or _stat_identity(lexical_stat) != _stat_identity(opened)
            or _stat_identity(post_open_lexical_stat) != _stat_identity(opened)
            or _stat_identity(resolved_stat) != _stat_identity(opened)
        ):
            raise SystemExit(
                f"decision-grade model input is not a stable regular file: {resolved}"
            )
        stream = os.fdopen(descriptor, "rb")
        descriptor = None
    except OSError as exc:
        raise SystemExit(f"cannot safely open model input {lexical}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return stream, {
        "path": str(resolved),
        "lexical_path": str(lexical),
        "mode": int(opened.st_mode),
        "size": int(opened.st_size),
        "mtime_ns": int(opened.st_mtime_ns),
        "ctime_ns": int(opened.st_ctime_ns),
        "device": int(opened.st_dev),
        "inode": int(opened.st_ino),
    }


def _read_authenticated_bytes(
    stream: IO[bytes], opened: Mapping[str, Any],
) -> bytes:
    """Read immutable bytes from an authenticated open file description."""
    stream.seek(0)
    content = stream.read()
    after = os.fstat(stream.fileno())
    if (
        _stat_identity(after)
        != tuple(
            int(opened[name])
            for name in (
                "mode", "size", "mtime_ns", "ctime_ns", "device", "inode",
            )
        )
        or len(content) != int(opened["size"])
    ):
        raise SystemExit("model input changed while its bytes were read")
    return content


def _finish_authenticated_input(
    stream: IO[bytes],
    opened: Mapping[str, Any],
    *,
    consumed_bytes: bytes | None,
    consumption: str,
) -> dict[str, Any]:
    """Verify one loader input and hash the same open file description."""
    expected = tuple(
        int(opened[name])
        for name in ("mode", "size", "mtime_ns", "ctime_ns", "device", "inode")
    )
    try:
        before_hash = os.fstat(stream.fileno())
        lexical_stat = Path(str(opened["lexical_path"])).lstat()
        resolved_stat = Path(str(opened["path"])).lstat()
    except OSError as exc:
        raise SystemExit(f"model input disappeared while being loaded: {exc}") from exc
    if (
        _stat_identity(before_hash) != expected
        or _stat_identity(lexical_stat) != expected
        or _stat_identity(resolved_stat) != expected
    ):
        raise SystemExit("model input identity changed while being loaded")
    if consumed_bytes is None:
        stream.seek(0)
        digest = hashlib.sha256()
        byte_count = 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
        sha256 = digest.hexdigest()
    else:
        byte_count = len(consumed_bytes)
        sha256 = hashlib.sha256(consumed_bytes).hexdigest()
    after_hash = os.fstat(stream.fileno())
    try:
        final_lexical_stat = Path(str(opened["lexical_path"])).lstat()
        final_resolved_stat = Path(str(opened["path"])).lstat()
    except OSError as exc:
        raise SystemExit(f"model input disappeared while being hashed: {exc}") from exc
    if (
        _stat_identity(after_hash) != expected
        or _stat_identity(final_lexical_stat) != expected
        or _stat_identity(final_resolved_stat) != expected
        or byte_count != int(opened["size"])
    ):
        raise SystemExit("model input changed while being loaded or hashed")
    return {
        name: opened[name]
        for name in (
            "path", "lexical_path", "mode", "size", "mtime_ns", "ctime_ns", "device",
            "inode",
        )
    } | {
        "sha256": sha256,
        "stable_read": True,
        "consumption": consumption,
    }


def _authenticated_input_artifact_if_file(
    path: Path, *, consumption: str,
) -> dict[str, Any] | None:
    """Best-effort streaming recheck without buffering a checkpoint in RAM."""
    stream: IO[bytes] | None = None
    try:
        stream, opened = _open_authenticated_input(path)
        return _finish_authenticated_input(
            stream,
            opened,
            consumed_bytes=None,
            consumption=consumption,
        )
    except (OSError, SystemExit):
        return None
    finally:
        if stream is not None:
            stream.close()


def _load_authenticated_model_inputs(
    checkpoint_path: Path,
    expected_params_inventory: Mapping[str, Any],
    *,
    loader: Callable[..., Any],
    device: str,
    require_complete: bool,
) -> tuple[Any, dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    """Select and load the exact checkpoint/params objects recorded in provenance."""
    params_inventory_before = _params_candidate_inventory(checkpoint_path)
    if params_inventory_before != expected_params_inventory:
        raise SystemExit("params.json candidate inventory changed before model load")
    checkpoint_params_path = _selected_params_candidate(params_inventory_before)
    checkpoint_stream, checkpoint_opened = _open_authenticated_input(checkpoint_path)
    params_stream: IO[bytes] | None = None
    params_opened: dict[str, Any] | None = None
    params_bytes: bytes | None = None
    try:
        if checkpoint_params_path is not None:
            params_stream, params_opened = _open_authenticated_input(
                checkpoint_params_path
            )
            params_bytes = _read_authenticated_bytes(params_stream, params_opened)
        model = loader(
            checkpoint_stream,
            checkpoint_path=checkpoint_path,
            params_json=params_bytes,
            params_path=checkpoint_params_path,
            device=device,
            require_complete=require_complete,
        )
        checkpoint_artifact = _finish_authenticated_input(
            checkpoint_stream,
            checkpoint_opened,
            consumed_bytes=None,
            consumption="torch_load_from_same_open_file_description",
        )
        params_artifact = (
            _finish_authenticated_input(
                params_stream,
                params_opened,
                consumed_bytes=params_bytes,
                consumption="json_decode_from_exact_authenticated_bytes",
            )
            if params_stream is not None
            and params_opened is not None
            and params_bytes is not None
            else None
        )
        if params_artifact is not None:
            selected_index = params_inventory_before["selected_index"]
            candidates = params_inventory_before["candidates"]
            if (
                not isinstance(selected_index, int)
                or isinstance(selected_index, bool)
                or not isinstance(candidates, list)
                or not isinstance(candidates[selected_index], dict)
                or any(
                    params_artifact.get(name)
                    != candidates[selected_index].get("identity", {}).get(name)
                    for name in (
                        "mode", "size", "mtime_ns", "ctime_ns", "device", "inode",
                    )
                )
            ):
                raise SystemExit(
                    "loaded params.json does not match the selected candidate identity"
                )
        params_inventory_after = _params_candidate_inventory(checkpoint_path)
        if params_inventory_after != params_inventory_before:
            raise SystemExit("params.json candidate inventory changed during model load")
    finally:
        checkpoint_stream.close()
        if params_stream is not None:
            params_stream.close()
    proof = {
        "schema": _MODEL_INPUT_CONSUMPTION_SCHEMA,
        "checkpoint_open": "absolute_lexical_path_o_nofollow",
        "checkpoint": "torch_load_from_same_open_file_description",
        "checkpoint_path_reopened_by_loader": False,
        "checkpoint_identity_verified_before_search": True,
        "checkpoint_sha256_streamed_from_same_open_file_description": True,
        "params": (
            "json_decode_from_exact_authenticated_bytes"
            if params_artifact is not None else "no_params_json"
        ),
        "params_open": (
            "absolute_lexical_path_o_nofollow"
            if params_artifact is not None else "no_params_json"
        ),
        "params_path_reopened_by_loader": False,
        "params_identity_verified_before_search": True,
        "params_selection": "first_is_file_in_checkpoint_ancestor_order",
        "params_candidate_inventory_schema": _PARAMS_CANDIDATE_INVENTORY_SCHEMA,
        "params_candidate_inventory_sha256": params_inventory_before[
            "inventory_sha256"
        ],
        "params_candidate_inventory_verified_before_load": True,
        "params_candidate_inventory_verified_after_load": True,
        "params_selected_index": params_inventory_before["selected_index"],
        "params_selected_path": params_inventory_before["selected_path"],
        "passed": True,
    }
    return model, checkpoint_artifact, params_artifact, proof


def _loaded_native_build_attestation(
    module: Any, module_name: str, producer_git_sha: str,
) -> dict[str, Any]:
    """Bind one loaded binary's embedded stamp to exact producer-revision inputs."""
    require_current_native_build_attestation_schema()
    dependency_bytes: dict[str, bytes] = {}
    current_inputs_match_revision = True
    for relative_path in native_build_dependency_paths(
        NATIVE_BUILD_ATTESTATION_SCHEMA, module_name,
    ):
        committed = _producer_git_file_at_commit(producer_git_sha, relative_path)
        try:
            current = (Path(__file__).resolve().parents[1] / relative_path).read_bytes()
        except OSError:
            current = None
        if committed is None:
            current_inputs_match_revision = False
            continue
        dependency_bytes[relative_path] = committed
        if current != committed:
            current_inputs_match_revision = False
    expected: dict[str, Any] | None = None
    try:
        expected = native_build_attestation(
            module_name, producer_git_sha, dependency_bytes,
        )
    except ValueError:
        current_inputs_match_revision = False
    observed = {
        "schema": getattr(module, "BUILD_ATTESTATION_SCHEMA", None),
        "module": getattr(module, "BUILD_MODULE_NAME", None),
        "source_git_sha": getattr(module, "BUILD_SOURCE_GIT_SHA", None),
        "input_sha256": getattr(module, "BUILD_INPUT_SHA256", None),
    }
    expected_core = (
        {
            name: expected[name]
            for name in ("schema", "module", "source_git_sha", "input_sha256")
        }
        if expected is not None else None
    )
    return {
        **observed,
        "dependencies": expected.get("dependencies", {}) if expected is not None else {},
        "current_inputs_match_revision": current_inputs_match_revision,
        "matches_producer_revision": bool(
            current_inputs_match_revision and observed == expected_core
        ),
    }


def _read_tracked_preregistration(
    path: Path, producer_git_sha: str,
) -> tuple[dict[str, Any], str]:
    """Read one stable preregistration that is committed at the producer SHA."""
    repo_root = Path(__file__).resolve().parents[1]
    resolved = path.expanduser().resolve()
    try:
        relative_path = resolved.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise SystemExit("--preregistration must be a tracked file inside the repo") from exc
    before = _artifact(resolved, require_file=True)
    try:
        document_bytes = resolved.read_bytes()
    except OSError as exc:
        raise SystemExit(f"cannot read --preregistration {resolved}: {exc}") from exc
    after = _artifact(resolved, require_file=True)
    if (
        _artifact_identity(before) != _artifact_identity(after)
        or before.get("size") != len(document_bytes)
        or before.get("sha256") != hashlib.sha256(document_bytes).hexdigest()
    ):
        raise SystemExit("--preregistration changed while it was being read")
    if _producer_git_file_at_commit(producer_git_sha, relative_path) != document_bytes:
        raise SystemExit(
            "--preregistration must be tracked verbatim by the clean producer commit"
        )
    try:
        document = document_bytes.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise SystemExit("--preregistration must be valid UTF-8 JSON") from exc
    return {**before, "repo_relative_path": relative_path}, document


def _canonical_syzygy_directory_path(path: str | Path) -> Path:
    """Reject spellings whose kernel traversal can differ after normalization."""
    raw_path = os.fspath(path)
    if (
        not os.path.isabs(raw_path)
        or raw_path.startswith("//")
        or os.path.expanduser(raw_path) != raw_path
        or os.path.normpath(raw_path) != raw_path
    ):
        raise SystemExit(
            "Syzygy directories must use canonical absolute paths without '.', '..', "
            "'~', duplicate separators, or trailing separators"
        )
    return Path(raw_path)


def _open_directory_no_follow(
    path: Path,
) -> tuple[int, Path, list[dict[str, int | str]]]:
    """Open and bind every directory component without traversing a symlink."""
    lexical = _canonical_syzygy_directory_path(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    current_fd = os.open(os.sep, flags)
    try:
        anchor_stat = os.fstat(current_fd)
        if not stat.S_ISDIR(anchor_stat.st_mode):
            raise SystemExit("Syzygy absolute-root anchor is not a directory")
        path_components: list[dict[str, int | str]] = [{
            "path": os.sep,
            **_tablebase_directory_identity(anchor_stat),
        }]
        current_path = Path(os.sep)
        for component in lexical.parts[1:]:
            try:
                component_stat = os.stat(
                    component, dir_fd=current_fd, follow_symlinks=False,
                )
            except OSError as exc:
                raise SystemExit(
                    f"cannot inspect Syzygy directory component {lexical}: {exc}"
                ) from exc
            if stat.S_ISLNK(component_stat.st_mode):
                raise SystemExit(
                    f"Syzygy directory path must not contain a symlink: {lexical}"
                )
            if not stat.S_ISDIR(component_stat.st_mode):
                raise SystemExit(
                    f"Syzygy directory path component is not a directory: {lexical}"
                )
            try:
                next_fd = os.open(component, flags, dir_fd=current_fd)
            except OSError as exc:
                raise SystemExit(
                    f"cannot open Syzygy directory without following symlinks "
                    f"{lexical}: {exc}"
                ) from exc
            opened_stat = os.fstat(next_fd)
            observed_identity = (
                int(component_stat.st_mode), int(component_stat.st_dev),
                int(component_stat.st_ino),
            )
            opened_identity = (
                int(opened_stat.st_mode), int(opened_stat.st_dev),
                int(opened_stat.st_ino),
            )
            if opened_identity != observed_identity:
                os.close(next_fd)
                raise SystemExit(
                    f"Syzygy directory changed while it was being opened: {lexical}"
                )
            os.close(current_fd)
            current_fd = next_fd
            current_path /= component
            path_components.append({
                "path": str(current_path),
                **_tablebase_directory_identity(opened_stat),
            })
        return current_fd, lexical, path_components
    except BaseException:
        os.close(current_fd)
        raise


def _tablebase_directory_identity(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _tablebase_checksum_catalog(
    directory_fd: int,
    directory: Path,
    entry_name: str,
    entry_stat: os.stat_result,
) -> tuple[dict[str, Any], tuple[tuple[str, str], ...]]:
    """Read the small official checksum catalog from one stable no-follow FD."""
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        file_fd = os.open(entry_name, file_flags, dir_fd=directory_fd)
    except OSError as exc:
        raise SystemExit(
            f"cannot open Syzygy checksum catalog without following symlinks "
            f"{directory / entry_name}: {exc}"
        ) from exc
    try:
        opened_before = os.fstat(file_fd)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(file_fd)
    finally:
        os.close(file_fd)
    identity = (
        int(entry_stat.st_mode), int(entry_stat.st_size),
        int(entry_stat.st_mtime_ns), int(entry_stat.st_ctime_ns),
        int(entry_stat.st_dev), int(entry_stat.st_ino),
    )
    if identity != (
        int(opened_before.st_mode), int(opened_before.st_size),
        int(opened_before.st_mtime_ns), int(opened_before.st_ctime_ns),
        int(opened_before.st_dev), int(opened_before.st_ino),
    ) or identity != (
        int(opened_after.st_mode), int(opened_after.st_size),
        int(opened_after.st_mtime_ns), int(opened_after.st_ctime_ns),
        int(opened_after.st_dev), int(opened_after.st_ino),
    ):
        raise SystemExit(
            f"Syzygy checksum catalog changed while it was read: "
            f"{directory / entry_name}"
        )
    content = b"".join(chunks)
    try:
        lines = content.decode("ascii", errors="strict").splitlines()
    except UnicodeError as exc:
        raise SystemExit("Syzygy checksum catalog is not ASCII") from exc
    rows: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 2:
            raise SystemExit(
                "Syzygy checksum catalog row is malformed at line "
                f"{line_number}"
            )
        digest, name = fields
        if (
            len(digest) != 32
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
            or not name
            or Path(name).name != name
            or Path(name).suffix not in (".rtbw", ".rtbz")
            or name in seen_names
        ):
            raise SystemExit(
                "Syzygy checksum catalog row is invalid at line "
                f"{line_number}"
            )
        seen_names.add(name)
        rows.append((name, digest.lower()))
    canonical_rows = tuple(sorted(rows))
    raw_sha256 = hashlib.sha256(content).hexdigest()
    entries_sha256 = checksum_catalog_entries_sha256(canonical_rows)
    wdl_count = sum(name.endswith(".rtbw") for name, _digest in canonical_rows)
    dtz_count = sum(name.endswith(".rtbz") for name, _digest in canonical_rows)
    approved = bool(
        raw_sha256 == APPROVED_SYZYGY_CHECKSUM_CATALOG_RAW_SHA256
        and len(canonical_rows) == APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRY_COUNT
        and len(content) == APPROVED_SYZYGY_CHECKSUM_CATALOG_SIZE
        and wdl_count == APPROVED_SYZYGY_CHECKSUM_CATALOG_WDL_COUNT
        and dtz_count == APPROVED_SYZYGY_CHECKSUM_CATALOG_DTZ_COUNT
        and entries_sha256 == APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRIES_SHA256
    )
    return ({
        "schema": "deepfin.syzygy_checksum_catalog.v1",
        "component": directory.name,
        "name": entry_name,
        "size": len(content),
        "mtime_ns": int(opened_after.st_mtime_ns),
        "ctime_ns": int(opened_after.st_ctime_ns),
        "device": int(opened_after.st_dev),
        "inode": int(opened_after.st_ino),
        "raw_sha256": raw_sha256,
        "algorithm": "md5",
        "entry_count": len(canonical_rows),
        "rtbw_count": wdl_count,
        "rtbz_count": dtz_count,
        "canonical_entries_sha256": entries_sha256,
        "entries": [list(row) for row in canonical_rows],
        "approved": approved,
    }, canonical_rows)


def _content_verification_rows(
    directories: list[dict[str, Any]],
    expected_md5: Mapping[str, str],
) -> list[list[object]]:
    """Bind every approved checksum to the exact physical file identity."""
    return [
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


def _verify_tablebase_contents(
    directories: list[dict[str, Any]],
    catalog_rows: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    """Stream each physical table once and compare it with the approved MD5."""
    expected_md5 = dict(catalog_rows)
    verification_rows = _content_verification_rows(directories, expected_md5)
    bytes_hashed = 0
    files_hashed = 0
    for directory in directories:
        directory_fd, opened_path, path_components = _open_directory_no_follow(
            Path(str(directory["path"])),
        )
        try:
            if (
                opened_path != Path(str(directory["path"]))
                or path_components != directory["path_components"]
                or _tablebase_directory_identity(os.fstat(directory_fd))
                != directory["root_identity"]
            ):
                raise SystemExit(
                    f"Syzygy directory changed before content verification: "
                    f"{directory['path']}"
                )
            for identity in directory["file_identities"]:
                name = str(identity[0])
                expected_identity = tuple(int(value) for value in identity[1:])
                try:
                    file_fd = os.open(
                        name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                except OSError as exc:
                    raise SystemExit(
                        f"cannot open Syzygy file for content verification "
                        f"{opened_path / name}: {exc}"
                    ) from exc
                try:
                    before = os.fstat(file_fd)
                    observed_identity = (
                        int(before.st_size), int(before.st_mtime_ns),
                        int(before.st_ctime_ns), int(before.st_dev),
                        int(before.st_ino),
                    )
                    if (
                        not stat.S_ISREG(before.st_mode)
                        or observed_identity != expected_identity
                    ):
                        raise SystemExit(
                            "Syzygy file changed before content verification: "
                            f"{opened_path / name}"
                        )
                    digest = hashlib.md5(usedforsecurity=False)
                    file_bytes = 0
                    while True:
                        chunk = os.read(file_fd, 8 * 1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        file_bytes += len(chunk)
                    after = os.fstat(file_fd)
                finally:
                    os.close(file_fd)
                if (
                    (
                        int(after.st_size), int(after.st_mtime_ns),
                        int(after.st_ctime_ns), int(after.st_dev), int(after.st_ino),
                    )
                    != expected_identity
                    or file_bytes != expected_identity[0]
                ):
                    raise SystemExit(
                        f"Syzygy file changed during content verification: "
                        f"{opened_path / name}"
                    )
                expected_digest = expected_md5.get(name)
                if expected_digest is None or digest.hexdigest() != expected_digest:
                    raise SystemExit(
                        f"Syzygy content checksum mismatch: {opened_path / name}"
                    )
                bytes_hashed += file_bytes
                files_hashed += 1
        finally:
            os.close(directory_fd)
    document = json.dumps(
        verification_rows, sort_keys=False, ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": "deepfin.syzygy_content_verification.v1",
        "method": "single_pass_md5_against_approved_catalog",
        "identity_binding_fields": [
            "component", "name", "size", "mtime_ns", "ctime_ns", "device",
            "inode", "approved_md5",
        ],
        "file_count": files_hashed,
        "bytes_hashed": bytes_hashed,
        "file_identity_checksum_sha256": hashlib.sha256(document).hexdigest(),
        "passed": True,
    }


def _reuse_tablebase_content_verification(
    directories: list[dict[str, Any]],
    catalog_rows: tuple[tuple[str, str], ...],
    prior: Mapping[str, Any],
) -> dict[str, Any]:
    """Revalidate one-pass evidence against the final metadata inventory."""
    expected_md5 = dict(catalog_rows)
    try:
        verification_rows = _content_verification_rows(directories, expected_md5)
    except KeyError as exc:
        raise SystemExit("Syzygy checksum catalog does not cover every table") from exc
    document = json.dumps(
        verification_rows, sort_keys=False, ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    expected = {
        "schema": "deepfin.syzygy_content_verification.v1",
        "method": "single_pass_md5_against_approved_catalog",
        "identity_binding_fields": [
            "component", "name", "size", "mtime_ns", "ctime_ns", "device",
            "inode", "approved_md5",
        ],
        "file_count": sum(
            int(directory["file_identity_count"]) for directory in directories
        ),
        "bytes_hashed": sum(int(directory["total_bytes"]) for directory in directories),
        "file_identity_checksum_sha256": hashlib.sha256(document).hexdigest(),
        "passed": True,
    }
    if dict(prior) != expected:
        raise SystemExit(
            "Syzygy filesystem identities no longer match the initial content "
            "verification"
        )
    return expected


def _tablebase_inventory(
    path_value: str, *, require_approved: bool = False,
    verify_contents: bool = False,
    prior_content_verification: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """No-follow filesystem identity for the 220 GiB production tablebases.

    Full byte hashing would read the corpus twice per run (roughly 440 GiB) and
    interfere with the search being measured.  Instead, the initial and final
    inventories bind the absolute-root anchor, every traversed directory, and
    every table file's device, inode, size, mtime, and ctime.  Ctime deliberately
    makes an in-place same-size edit or swap-and-restore visible even if mtime is
    restored.  Decision-grade calls additionally require the portable sorted
    filename/size digest and official checksum catalog pinned in
    ``scripts.approved_syzygy``.  A decision-grade run streams every physical
    table once before search; its final inventory reuses that identity-bound
    proof instead of performing a second corpus-sized pass.
    """
    directories: list[dict[str, Any]] = []
    checksum_catalogs: list[
        tuple[dict[str, Any], tuple[tuple[str, str], ...]]
    ] = []
    approved_by_name = {
        component.directory_name: component
        for component in _APPROVED_SYZYGY_COMPONENTS
    }
    for raw_directory in path_value.split(SEPARATOR):
        if not raw_directory or raw_directory != raw_directory.strip():
            raise SystemExit("Syzygy path contains an empty directory component")
        canonical_directory = _canonical_syzygy_directory_path(raw_directory)
        directory_fd, directory, path_components = _open_directory_no_follow(
            canonical_directory,
        )
        try:
            root_before = os.fstat(directory_fd)
            root_identity = _tablebase_directory_identity(root_before)
            if any(
                path_components[-1].get(field) != root_identity[field]
                for field in ("device", "inode", "mtime_ns", "ctime_ns")
            ):
                raise SystemExit(
                    f"Syzygy directory changed while it was being opened: {directory}"
                )

            def scan_table_files(
                directory_fd: int = directory_fd,
                directory: Path = directory,
            ) -> tuple[
                dict[str, int], int, list[list[str | int]],
                tuple[dict[str, Any], tuple[tuple[str, str], ...]] | None,
            ]:
                counts = {"rtbw": 0, "rtbz": 0}
                total_bytes = 0
                identities: list[list[str | int]] = []
                checksum_catalog = None
                try:
                    with os.scandir(directory_fd) as iterator:
                        entries = sorted(iterator, key=lambda entry: entry.name)
                except OSError as exc:
                    raise SystemExit(
                        f"cannot inventory Syzygy directory {directory}: {exc}"
                    ) from exc
                for entry in entries:
                    try:
                        entry_stat = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        raise SystemExit(
                            f"cannot inspect Syzygy directory entry "
                            f"{directory / entry.name}: {exc}"
                        ) from exc
                    if stat.S_ISLNK(entry_stat.st_mode):
                        raise SystemExit(
                            "Syzygy directory entries must not be symlinks: "
                            f"{directory / entry.name}"
                        )
                    if not stat.S_ISREG(entry_stat.st_mode):
                        raise SystemExit(
                            "Syzygy directory entries must be regular files: "
                            f"{directory / entry.name}"
                        )
                    if entry.name == APPROVED_SYZYGY_CHECKSUM_CATALOG_NAME:
                        checksum_catalog = _tablebase_checksum_catalog(
                            directory_fd, directory, entry.name, entry_stat,
                        )
                    suffix = Path(entry.name).suffix
                    if suffix not in (".rtbw", ".rtbz"):
                        continue
                    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
                    try:
                        file_fd = os.open(
                            entry.name, file_flags, dir_fd=directory_fd,
                        )
                    except OSError as exc:
                        raise SystemExit(
                            f"cannot open Syzygy file without following symlinks "
                            f"{directory / entry.name}: {exc}"
                        ) from exc
                    try:
                        opened_stat = os.fstat(file_fd)
                    finally:
                        os.close(file_fd)
                    entry_identity = (
                        int(entry_stat.st_mode), int(entry_stat.st_size),
                        int(entry_stat.st_mtime_ns), int(entry_stat.st_ctime_ns),
                        int(entry_stat.st_dev), int(entry_stat.st_ino),
                    )
                    opened_identity = (
                        int(opened_stat.st_mode), int(opened_stat.st_size),
                        int(opened_stat.st_mtime_ns), int(opened_stat.st_ctime_ns),
                        int(opened_stat.st_dev), int(opened_stat.st_ino),
                    )
                    if opened_identity != entry_identity or not stat.S_ISREG(
                        opened_stat.st_mode,
                    ):
                        raise SystemExit(
                            f"Syzygy file changed while it was being inventoried: "
                            f"{directory / entry.name}"
                        )
                    counts[suffix[1:]] += 1
                    total_bytes += int(opened_stat.st_size)
                    identities.append([
                        entry.name,
                        int(opened_stat.st_size),
                        int(opened_stat.st_mtime_ns),
                        int(opened_stat.st_ctime_ns),
                        int(opened_stat.st_dev),
                        int(opened_stat.st_ino),
                    ])
                return counts, total_bytes, identities, checksum_catalog

            counts, total_bytes, file_identities, checksum_catalog = scan_table_files()
            repeated_scan = scan_table_files()
            if repeated_scan != (
                counts, total_bytes, file_identities, checksum_catalog,
            ):
                raise SystemExit(
                    f"Syzygy files changed while they were inventoried: {directory}"
                )
            root_after = os.fstat(directory_fd)
            if _tablebase_directory_identity(root_after) != root_identity:
                raise SystemExit(
                    f"Syzygy directory changed while it was inventoried: {directory}"
                )
            filename_sizes = [
                (str(row[0]), int(row[1])) for row in file_identities
            ]
            layout_sha256 = filename_size_sha256(filename_sizes)
            approved_component = approved_by_name.get(directory.name)
            approved_layout = {
                "schema": APPROVED_SYZYGY_LAYOUT_SCHEMA,
                "component": directory.name,
                "canonical_encoding": APPROVED_SYZYGY_FILENAME_SIZE_ENCODING,
                "rtbw_count": counts["rtbw"],
                "rtbz_count": counts["rtbz"],
                "file_count": len(file_identities),
                "total_bytes": total_bytes,
                "filename_size_sha256": layout_sha256,
                "passed": bool(
                    approved_component is not None
                    and counts["rtbw"] == approved_component.rtbw_count
                    and counts["rtbz"] == approved_component.rtbz_count
                    and len(file_identities) == approved_component.file_count
                    and total_bytes == approved_component.total_bytes
                    and layout_sha256 == approved_component.filename_size_sha256
                ),
            }
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
            directories.append({
                "path": str(directory),
                "root_identity": root_identity,
                "path_component_identity_fields": list(
                    _TABLEBASE_PATH_COMPONENT_IDENTITY_FIELDS
                ),
                "path_components": path_components,
                "rtbw_count": counts["rtbw"],
                "rtbz_count": counts["rtbz"],
                "file_identity_count": len(file_identities),
                "file_identity_fields": list(_TABLEBASE_FILE_IDENTITY_FIELDS),
                "file_identities": file_identities,
                "total_bytes": total_bytes,
                "approved_layout": approved_layout,
                "inventory_sha256": hashlib.sha256(identity_document).hexdigest(),
            })
            if checksum_catalog is not None:
                checksum_catalogs.append(checksum_catalog)
        finally:
            os.close(directory_fd)
    if verify_contents and prior_content_verification is not None:
        raise SystemExit(
            "Syzygy content verification cannot be both recomputed and reused"
        )
    catalog_artifacts = [artifact for artifact, _rows in checksum_catalogs]
    catalog_rows = checksum_catalogs[0][1] if len(checksum_catalogs) == 1 else ()
    observed_names = {
        str(identity[0])
        for directory in directories
        for identity in directory["file_identities"]
    }
    catalog_names = {name for name, _digest in catalog_rows}
    checksum_catalog = catalog_artifacts[0] if len(catalog_artifacts) == 1 else None
    checksum_catalog_passed = bool(
        isinstance(checksum_catalog, dict)
        and checksum_catalog.get("approved") is True
        and observed_names == catalog_names
    )
    approved_component_order = tuple(
        str(directory["approved_layout"]["component"])
        for directory in directories
    )
    expected_component_order = tuple(
        component.directory_name for component in _APPROVED_SYZYGY_COMPONENTS
    )
    approved_layout_passed = bool(
        approved_component_order == expected_component_order
        and all(
            directory["approved_layout"]["passed"] is True
            for directory in directories
        )
        and checksum_catalog_passed
    )
    if require_approved and not approved_layout_passed:
        raise SystemExit(
            "decision-grade trajectory banks require the exact approved production "
            "Syzygy filename/size layout and checksum catalog"
        )
    content_verification = None
    if approved_layout_passed and verify_contents:
        content_verification = _verify_tablebase_contents(
            directories, catalog_rows,
        )
    elif approved_layout_passed and prior_content_verification is not None:
        content_verification = _reuse_tablebase_content_verification(
            directories, catalog_rows, prior_content_verification,
        )
    if require_approved and not (
        isinstance(content_verification, dict)
        and content_verification.get("passed") is True
    ):
        raise SystemExit(
            "decision-grade trajectory banks require a one-pass content checksum "
            "verification bound to the current Syzygy file identities"
        )
    canonical_path = SEPARATOR.join(str(row["path"]) for row in directories)
    result = {
        "schema": _TABLEBASE_INVENTORY_SCHEMA,
        "identity_method": (
            "approved_filename_size_plus_no_follow_path_components_and_"
            "file_device_inode_size_mtime_ctime"
        ),
        "path_anchor_semantics": (
            "absolute_root_and_each_lexical_directory_component_no_follow"
        ),
        "path": canonical_path,
        "directories": directories,
        "rtbw_count": sum(int(row["rtbw_count"]) for row in directories),
        "rtbz_count": sum(int(row["rtbz_count"]) for row in directories),
        "approved_layout_schema": APPROVED_SYZYGY_LAYOUT_SCHEMA,
        "approved_component_order": list(approved_component_order),
        "approved_layout_passed": approved_layout_passed,
        "checksum_catalog": checksum_catalog,
        "checksum_catalog_covers_logical_table_names": checksum_catalog_passed,
        "content_verification": content_verification,
    }
    inventory_document = json.dumps(
        result, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")
    result["inventory_sha256"] = hashlib.sha256(inventory_document).hexdigest()
    return result


def _checkpoint_file(path: Path) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    try:
        input_stat = lexical.lstat()
    except OSError as exc:
        raise SystemExit(f"checkpoint cannot be inspected: {lexical}: {exc}") from exc
    if stat.S_ISLNK(input_stat.st_mode):
        raise SystemExit(f"checkpoint path cannot be a symlink: {lexical}")
    candidate = lexical / "trainer.pt" if stat.S_ISDIR(input_stat.st_mode) else lexical
    try:
        candidate_stat = candidate.lstat()
    except OSError as exc:
        raise SystemExit(f"checkpoint has no trainer.pt: {lexical}") from exc
    if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISREG(candidate_stat.st_mode):
        raise SystemExit(f"checkpoint has no regular trainer.pt: {lexical}")
    return candidate


def _checkpoint_params_candidates(trainer_pt: Path) -> tuple[Path, ...]:
    """Return the complete ordered candidate set used by the UCI loader."""
    current = trainer_pt.parent
    candidates: list[Path] = []
    for _ in range(_PARAMS_SEARCH_LIMIT):
        candidates.append(current / "params.json")
        if current.parent == current:
            break
        current = current.parent
    return tuple(candidates)


def _params_path_component_inventory(directory: Path) -> list[dict[str, Any]]:
    """Bind each lexical ancestor whose entry can redirect a candidate lookup."""
    lexical = Path(os.path.abspath(directory.expanduser()))
    components: list[dict[str, Any]] = []
    current = Path(lexical.anchor)
    for index, component in enumerate(lexical.parts):
        if index:
            current /= component
        try:
            observed = current.lstat()
        except OSError as exc:
            raise SystemExit(
                f"cannot inspect params.json path component {current}: {exc}"
            ) from exc
        kind = (
            "directory" if stat.S_ISDIR(observed.st_mode)
            else "symlink" if stat.S_ISLNK(observed.st_mode)
            else "non_directory"
        )
        components.append({
            "path": str(current),
            "kind": kind,
            "mode": int(observed.st_mode),
            "size": int(observed.st_size),
            "mtime_ns": int(observed.st_mtime_ns),
            "ctime_ns": int(observed.st_ctime_ns),
            "device": int(observed.st_dev),
            "inode": int(observed.st_ino),
        })
    return components


def _params_candidate_inventory(trainer_pt: Path) -> dict[str, Any]:
    """Snapshot every params candidate, including negative lookup evidence."""
    candidates: list[dict[str, Any]] = []
    selected_index: int | None = None
    selected_path: str | None = None
    for index, candidate in enumerate(_checkpoint_params_candidates(trainer_pt)):
        lexical = Path(os.path.abspath(candidate.expanduser()))
        try:
            observed = lexical.lstat()
        except FileNotFoundError:
            observed = None
        except OSError as exc:
            raise SystemExit(f"cannot inspect params.json candidate {lexical}: {exc}") from exc
        if observed is None:
            state = "absent"
            identity: dict[str, Any] | None = None
            resolves_to_regular = False
        else:
            state = (
                "regular" if stat.S_ISREG(observed.st_mode)
                else "symlink" if stat.S_ISLNK(observed.st_mode)
                else "nonregular"
            )
            identity = {
                "mode": int(observed.st_mode),
                "size": int(observed.st_size),
                "mtime_ns": int(observed.st_mtime_ns),
                "ctime_ns": int(observed.st_ctime_ns),
                "device": int(observed.st_dev),
                "inode": int(observed.st_ino),
            }
            try:
                resolves_to_regular = lexical.is_file()
            except OSError:
                resolves_to_regular = False
        if selected_index is None and resolves_to_regular:
            selected_index = index
            selected_path = str(lexical)
        candidates.append({
            "index": index,
            "path": str(lexical),
            "state": state,
            "resolves_to_regular_file": resolves_to_regular,
            "identity": identity,
            "parent_path_components": _params_path_component_inventory(lexical.parent),
        })
    payload: dict[str, Any] = {
        "schema": _PARAMS_CANDIDATE_INVENTORY_SCHEMA,
        "search_limit": _PARAMS_SEARCH_LIMIT,
        "selection_policy": "first_is_file_in_checkpoint_ancestor_order",
        "trainer_pt": str(Path(os.path.abspath(trainer_pt.expanduser()))),
        "candidates": candidates,
        "selected_index": selected_index,
        "selected_path": selected_path,
    }
    document = json.dumps(
        payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")
    return {**payload, "inventory_sha256": hashlib.sha256(document).hexdigest()}


def _selected_params_candidate(inventory: Mapping[str, Any]) -> Path | None:
    selected = inventory.get("selected_path")
    if selected is None:
        return None
    if not isinstance(selected, str) or not selected:
        raise SystemExit("params.json candidate inventory has an invalid selection")
    candidates = inventory.get("candidates")
    index = inventory.get("selected_index")
    if (
        not isinstance(candidates, list)
        or not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= len(candidates)
        or not isinstance(candidates[index], dict)
        or candidates[index].get("path") != selected
        or candidates[index].get("resolves_to_regular_file") is not True
    ):
        raise SystemExit("params.json candidate inventory selection is inconsistent")
    if candidates[index].get("state") != "regular":
        raise SystemExit(
            "selected params.json candidate must be a regular non-symlink file"
        )
    return Path(selected)


def _nvidia_driver_version(device_index: int) -> str | None:
    try:
        lines = subprocess.check_output(
            [
                "nvidia-smi", f"--id={device_index}",
                "--query-gpu=driver_version", "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    version = lines[0].strip() if lines else ""
    return version or None


def _require_analyzable_source_groups(
    group_ids: Mapping[str, str | None], *, methodology_smoke: bool,
) -> None:
    if methodology_smoke:
        return
    if _source_game_group_count(group_ids) < _MIN_DECISION_GRADE_SOURCE_GAMES:
        raise SystemExit(
            "decision-grade trajectory banks require at least "
            f"{_MIN_DECISION_GRADE_SOURCE_GAMES} distinct source games for the "
            "canonical OOB bootstrap"
        )


def _source_game_group_count(group_ids: Mapping[str, str | None]) -> int:
    """Count distinct source-game clusters without treating missing IDs as a group."""
    return len({group_id for group_id in group_ids.values() if group_id})


def _source_group_resolution_passed(group_ids: Mapping[str, str | None]) -> bool:
    """Whether a completed bank retains enough clusters for decision-grade OOB CIs."""
    return _source_game_group_count(group_ids) >= _MIN_DECISION_GRADE_SOURCE_GAMES


def _excluded_position_evidence(
    position: Any,
    *,
    source_dir: str | None,
    source_shard: str | None,
    game_id: int | None,
    group_id: str | None,
    chunks_required: int,
    snapshots: list[dict[str, Any]],
    reason: str,
    search_result: dict[str, Any] | None = None,
    collection_error: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Retain raw callback observations even when a position cannot be analyzed."""
    evidence: dict[str, Any] = {
        "key": position.key,
        "fen": position.fen,
        "source_dir": source_dir,
        "shard": source_shard,
        "game_id": game_id,
        "group_id": group_id,
        "phase": position.phase,
        "source": position.source,
        **_deep_reference_position_fields(position),
        "chunks_observed": len(snapshots),
        "chunks_required": chunks_required,
        "reason": reason,
        "partial_observations": [dict(snapshot) for snapshot in snapshots],
    }
    if search_result is not None:
        evidence["search_result"] = search_result
    if collection_error is not None:
        evidence["collection_error"] = collection_error
    return evidence


def _require_nvidia_driver_provenance(
    driver_version: str | None, *, methodology_smoke: bool,
) -> None:
    if not methodology_smoke and not driver_version:
        raise RuntimeError(
            "decision-grade trajectory banks require readable NVIDIA driver provenance"
        )


def _write_preregistration_plan(
    output_path: Path,
    payload: dict[str, Any],
    *,
    protected_files: list[Path],
    protected_directories: list[Path],
) -> Path:
    """Create a new untracked in-repo plan without replacing any evidence input."""
    output = _require_safe_preregistration_path(
        output_path,
        protected_files=protected_files,
        protected_directories=protected_directories,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output, payload)
    return output


def _require_safe_preregistration_path(
    output_path: Path,
    *,
    protected_files: list[Path],
    protected_directories: list[Path],
) -> Path:
    """Validate a plan destination before model load or any output write."""
    repo_root = Path(__file__).resolve().parents[1]
    if reserved_output_path(output_path):
        raise SystemExit(
            "--write-preregistration must not use the output lock/staging namespace"
        )
    output = output_path.expanduser().resolve()
    if output in {path.expanduser().resolve() for path in protected_files}:
        raise SystemExit("--write-preregistration aliases a consumed input artifact")
    for directory in protected_directories:
        try:
            output.relative_to(directory.expanduser().resolve())
        except ValueError:
            continue
        raise SystemExit(
            "--write-preregistration must not be inside a Syzygy or authenticated "
            "replay-snapshot directory"
        )
    try:
        output.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit("--write-preregistration must be inside the repository") from exc
    if output.exists():
        raise SystemExit(f"refusing to overwrite preregistration plan {output}")
    if repo_controlled_output(output, repo_root):
        raise SystemExit("--write-preregistration must target a new untracked file")
    return output


def _validate_registry_search_values(
    active_path: str, values: dict[str, float | int | bool],
) -> None:
    """Apply the same advertised bounds as the UCI startup surface."""
    registry_path = "walker" if active_path == "walker_puct" else active_path
    by_field = {option.field: option for option in SEARCH_OPTIONS}
    problems: list[str] = []
    for field, value in values.items():
        option = by_field.get(field)
        if option is None or registry_path not in option.live_in:
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            problems.append(f"{field}={value!r} is not finite")
        elif option.lo is not None and numeric < option.lo:
            problems.append(f"{field}={value!r} is below {option.lo}")
        elif option.hi is not None and numeric > option.hi:
            problems.append(f"{field}={value!r} is above {option.hi}")
    if problems:
        raise SystemExit("search values violate the production UCI registry: " + "; ".join(problems))


def _require_search_take_effect(
    *,
    expected_mode: str,
    expected_workers: int,
    active_parameters: dict[str, float | int | bool],
    realized: dict[str, float | int | bool | str],
) -> None:
    """Fail closed if the worker did not install the requested live search shape."""
    failures: list[str] = []
    if realized.get("concurrency_mode") != expected_mode:
        failures.append(
            f"mode requested={expected_mode!r} realized="
            f"{realized.get('concurrency_mode')!r}"
        )
    if realized.get("concurrency_workers") != expected_workers:
        failures.append(
            f"workers requested={expected_workers!r} realized="
            f"{realized.get('concurrency_workers')!r}"
        )
    for name, requested in active_parameters.items():
        actual = realized.get(name)
        if actual != requested:
            failures.append(f"{name} requested={requested!r} realized={actual!r}")
    if failures:
        raise RuntimeError("search-path take-effect check failed: " + "; ".join(failures))


def _evaluator_stack_name(evaluator: Any) -> str:
    """Read the wrapper chain from the objects that will actually evaluate leaves."""
    name = type(evaluator).__name__
    if name in ("BatchCoalescingDispatcher", "CUDAOwnerDispatcher"):
        return f"{name}({_evaluator_stack_name(evaluator._inner)})"
    if name == "ThreadSafeGPUDispatcher":
        return f"{name}({_evaluator_stack_name(evaluator._eval)})"
    return name


def _find_direct_evaluator(evaluator: Any) -> Any:
    current = evaluator
    while type(current).__name__ != "DirectGPUEvaluator":
        if hasattr(current, "_inner"):
            current = current._inner
        elif hasattr(current, "_eval"):
            current = current._eval
        else:
            raise RuntimeError(
                "production evaluator stack has no DirectGPUEvaluator: "
                f"{_evaluator_stack_name(evaluator)}"
            )
    return current


def _publish_collected_evidence_pair(
    pending_output: Path,
    output: Path,
    pending_manifest: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    """Publish only through the exact bank and parent retained at collection."""
    state = _ACTIVE_PENDING_EVIDENCE
    if not isinstance(state, dict):
        raise RuntimeError("complete collection lacks retained publication evidence")
    retained_output_fd = state.get("retained_output_fd")
    retained_output_parent_fd = state.get("retained_output_parent_fd")
    retained_output_artifact = state.get("output_artifact")
    if (
        state.get("collection_complete") is not True
        or state.get("pending_output") != pending_output
        or state.get("output") != output
        or state.get("manifest") != manifest_path
        or state.get("pending_manifest") != pending_manifest
        or not isinstance(retained_output_fd, int)
        or retained_output_fd < 0
        or not isinstance(retained_output_parent_fd, int)
        or retained_output_parent_fd < 0
        or not isinstance(retained_output_artifact, dict)
    ):
        raise RuntimeError("complete collection lacks retained publication evidence")
    state["publication_manifest"] = manifest
    _publish_evidence_pair(
        pending_output,
        output,
        pending_manifest,
        manifest_path,
        manifest,
        retained_output_fd=retained_output_fd,
        retained_output_parent_fd=retained_output_parent_fd,
        retained_output_artifact=retained_output_artifact,
    )


def _main() -> None:
    global _ACTIVE_PENDING_EVIDENCE
    ap = argparse.ArgumentParser(prog="backtest_chunk_trajectory")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument(
        "--preregistration", type=Path, default=None,
        help=(
            "tracked JSON freezing input hashes, panel size, full search shape, "
            "Syzygy inventory, and canonical analyzer rule; required unless "
            "--methodology-smoke"
        ),
    )
    ap.add_argument(
        "--write-preregistration", type=Path, default=None,
        help="write the realized pre-search contract and exit; commit it before collection",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    ap.add_argument("--chunk-sims", type=int, default=2048)
    ap.add_argument("--max-chunks", type=int, default=8, help="chunks per position (8 -> up to 16k sims)")
    ap.add_argument("--max-positions", type=int, default=200)
    ap.add_argument(
        "--c-scale", type=float, default=PLAY_SEARCH_DEFAULTS["c_scale"],
        help="classic-Gumbel only; rejected as inert when --walkers >1",
    )
    ap.add_argument(
        "--c-visit", type=float, default=PLAY_SEARCH_DEFAULTS["c_visit"],
        help="classic-Gumbel only; rejected as inert when --walkers >1",
    )
    ap.add_argument(
        "--c-visit-root", type=float, default=PLAY_SEARCH_DEFAULTS["c_visit_root"],
        help="classic-Gumbel only; rejected as inert when --walkers >1",
    )
    ap.add_argument("--c-scale-root", type=float, default=PLAY_SEARCH_DEFAULTS["c_scale_root"],
                    help="ROOT-ONLY c_scale (descent keeps --c-scale). Default matches the "
                         "production play search (root-log); <0 = legacy linear root.")
    ap.add_argument("--q-visit-exp-root", type=float, default=PLAY_SEARCH_DEFAULTS["q_visit_exp_root"],
                    help="ROOT-ONLY value-transform exponent. Default matches the production play "
                         "search (root-log); >=90 = legacy linear root.")
    ap.add_argument(
        "--gumbel-topk", type=int, default=PLAY_SEARCH_DEFAULTS["topk"],
        help="classic-Gumbel only; rejected as inert when --walkers >1",
    )
    ap.add_argument(
        "--walkers", type=int, default=EngineOptions().threads,
        help="walker threads; default is the production UCI Threads default",
    )
    ap.add_argument(
        "--matched-rows", type=Path, default=None,
        help="audit-to-game index (default: <audit-set>.matched_rows.npz)",
    )
    ap.add_argument(
        "--matched-rows-report", type=Path, default=None,
        help="authenticated index report (default: <matched-rows>.report.json)",
    )
    ap.add_argument(
        "--syzygy-path", default=default_syzygy_path(),
        help="production Syzygy directory pair; required for decision-grade banks",
    )
    ap.add_argument("--compile", dest="compile", action="store_true")
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.set_defaults(compile=True)
    ap.add_argument("--compile-mode", default="max-autotune")
    ap.add_argument(
        "--compile-cache-dir",
        default=os.environ.get(
            "DEEPFIN_COMPILE_CACHE", os.path.expanduser("~/.cache/deepfin/worker_cache"),
        ),
    )
    ap.add_argument(
        "--methodology-smoke", action="store_true",
        help="allow missing game groups; output is stamped non-decision-grade",
    )
    ap.add_argument(
        "--recover-publication", action="store_true",
        help="publish a fully prepared interrupted pair without reloading search inputs",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("runs/backtest/chunk_trajectory.jsonl"))
    args = ap.parse_args()

    preimport_start_status = _preimport_python_surface_status(
        _PREIMPORT_PYTHON_SOURCES
    )
    meta_path = Path(str(args.out) + ".meta.json")
    pending_meta_path = _pending_manifest_path(meta_path)
    if args.recover_publication:
        if args.write_preregistration is not None:
            raise SystemExit(
                "--recover-publication and --write-preregistration are mutually exclusive"
            )
        _require_safe_output_paths(
            args.out, meta_path, protected_files=[], protected_directories=[],
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        recovery_locks = _acquire_output_locks(args.out, meta_path)
        try:
            recovered = _require_new_output_pair(
                args.out, meta_path, overwrite=bool(args.overwrite),
            )
        finally:
            for recovery_lock in reversed(recovery_locks):
                recovery_lock.close()
        if not recovered:
            raise SystemExit(f"no fully prepared evidence pair to recover for {args.out}")
        print(f"[traj] recovered evidence pair -> {args.out}")
        print(f"[traj] provenance -> {meta_path}")
        return
    requested_path = "walker_puct" if int(args.walkers) > 1 else "gumbel"
    _validate_registry_search_values(
        requested_path, {"chunk_sims": int(args.chunk_sims)},
    )
    if (
        not args.methodology_smoke
        and (
            _PREIMPORT_PYTHON_SOURCES.get("passed") is not True
            or preimport_start_status.get("passed") is not True
        )
    ):
        raise SystemExit(
            "decision-grade trajectory collection requires a clean tracked-Python "
            "snapshot taken before project imports; run the script as a fresh process "
            "from a clean checkout"
        )
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --recover-publication is used")
    matched_path = args.matched_rows or default_matched_rows_path(args.audit_set)
    matched_report_path = (
        args.matched_rows_report
        or default_matched_rows_report_path(matched_path)
    )
    checkpoint_path = _checkpoint_file(Path(args.checkpoint))
    checkpoint_params_candidates = list(_checkpoint_params_candidates(checkpoint_path))
    producer_git_sha, producer_git_dirty = _git_state()
    if (
        not args.methodology_smoke
        and producer_git_sha != _PREIMPORT_PYTHON_SOURCES.get("git_sha")
    ):
        raise SystemExit(
            "producer Git revision changed after the pre-import Python snapshot"
        )
    if producer_git_dirty and not args.methodology_smoke:
        raise SystemExit(
            "decision-grade trajectory banks require a clean producer checkout; "
            "commit or stash changes, or pass --methodology-smoke"
        )
    initial_producer_sources = _producer_python_source_artifacts(
        producer_git_sha,
        require_tracked=not args.methodology_smoke,
    )
    if args.preregistration is not None and args.write_preregistration is not None:
        raise SystemExit(
            "--preregistration and --write-preregistration are mutually exclusive"
        )
    if (
        args.preregistration is None
        and args.write_preregistration is None
        and not args.methodology_smoke
    ):
        raise SystemExit(
            "decision-grade trajectory banks require a tracked --preregistration"
        )
    preregistration_path = (
        args.preregistration.expanduser().resolve()
        if args.preregistration is not None else None
    )
    preregistration_artifact: dict[str, Any] | None = None
    preregistration_document: str | None = None
    if preregistration_path is not None:
        preregistration_artifact, preregistration_document = (
            _read_tracked_preregistration(preregistration_path, producer_git_sha)
        )
    audit_positions, audit_set_artifact = _load_audit_set_snapshot(
        args.audit_set, require_approved=not args.methodology_smoke,
    )
    initial_input_artifacts = {
        "producer_script": initial_producer_sources["producer_script"],
        "publication_helper": initial_producer_sources[
            "scripts.chunk_trajectory_publication"
        ],
        # Checkpoint and params artifacts are filled from the exact open objects
        # consumed by the model loader immediately before search construction.
        "checkpoint": None,
        "checkpoint_params": None,
        "audit_set": audit_set_artifact,
        "matched_rows": (
            _artifact(matched_path, require_file=True) if matched_path.is_file() else None
        ),
        "matched_rows_report": (
            _artifact(matched_report_path, require_file=True)
            if matched_report_path.is_file() else None
        ),
        "preregistration": preregistration_artifact,
    }
    syzygy_directories = (
        [
            _canonical_syzygy_directory_path(part)
            for part in str(args.syzygy_path).split(SEPARATOR)
        ]
        if args.syzygy_path else []
    )
    _require_safe_output_paths(
        args.out,
        meta_path,
        protected_files=[
            args.audit_set, matched_path, matched_report_path, checkpoint_path,
            *checkpoint_params_candidates,
            Path(__file__),
            Path(publication_module.__file__),
            *([preregistration_path] if preregistration_path is not None else []),
        ],
        protected_directories=syzygy_directories,
    )
    if args.chunk_sims <= 0 or args.max_chunks < 2 or args.max_positions < 0:
        raise SystemExit("--chunk-sims must be >0, --max-chunks >=2, --max-positions >=0")
    if not args.methodology_smoke and args.max_chunks < _MIN_DECISION_GRADE_CHUNKS:
        raise SystemExit(
            "decision-grade trajectory banks require --max-chunks >= "
            f"{_MIN_DECISION_GRADE_CHUNKS} so every held rung trains on several others"
        )
    if not 1 <= args.walkers <= 64:
        raise SystemExit("--walkers must be inside the production Threads range [1, 64]")
    if int(EngineOptions().threads) != _PRODUCTION_WALKERS:
        raise SystemExit(
            "controller analyzer production-walker constant no longer matches "
            "EngineOptions.threads"
        )
    if not args.methodology_smoke and args.walkers != _PRODUCTION_WALKERS:
        raise SystemExit(
            "decision-grade trajectory banks require the shipped production "
            f"Threads={_PRODUCTION_WALKERS} walker-PUCT path"
        )
    if not args.methodology_smoke and args.max_positions == 0:
        raise SystemExit("decision-grade trajectory banks require --max-positions >0")
    if not args.methodology_smoke and not (
        args.device == "cuda" or _canonical_cuda_device_string(args.device)
    ):
        raise SystemExit("decision-grade trajectory banks require the production CUDA path")
    if not args.methodology_smoke and (
        not args.compile or args.compile_mode != "max-autotune"
    ):
        raise SystemExit(
            "decision-grade trajectory banks require production "
            "torch.compile mode max-autotune"
        )
    syzygy_inventory = (
        _tablebase_inventory(
            str(args.syzygy_path), require_approved=not args.methodology_smoke,
            verify_contents=not args.methodology_smoke,
        )
        if args.syzygy_path else None
    )
    syzygy_components = (
        tuple(
            (int(row["rtbw_count"]), int(row["rtbz_count"]))
            for row in syzygy_inventory["directories"]
        )
        if isinstance(syzygy_inventory, dict) else ()
    )
    if (
        not args.methodology_smoke
        and (
            not isinstance(syzygy_inventory, dict)
            or int(syzygy_inventory["rtbw_count"]) < _PRODUCTION_WDL_FILES
            or int(syzygy_inventory["rtbz_count"]) < _PRODUCTION_DTZ_FILES
            or syzygy_components != _PRODUCTION_TB_COMPONENTS
            or syzygy_inventory.get("approved_layout_passed") is not True
        )
    ):
        raise SystemExit(
            "decision-grade trajectory banks require the complete production "
            f"Syzygy pair ({_PRODUCTION_WDL_FILES} WDL and "
            f"{_PRODUCTION_DTZ_FILES} DTZ files)"
        )
    if isinstance(syzygy_inventory, dict):
        # The engine and final inventory must consume exactly the no-follow,
        # canonical paths whose identities were approved above—not a raw
        # spelling with different kernel traversal semantics.
        args.syzygy_path = syzygy_inventory["path"]
    if args.syzygy_path or not args.methodology_smoke:
        try:
            require_tablebases(args.syzygy_path, what="trajectory --syzygy-path")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    gumbel_defaults = {
        "c_scale": float(PLAY_SEARCH_DEFAULTS["c_scale"]),
        "c_visit": float(PLAY_SEARCH_DEFAULTS["c_visit"]),
        "c_visit_root": float(PLAY_SEARCH_DEFAULTS["c_visit_root"]),
        "c_scale_root": float(PLAY_SEARCH_DEFAULTS["c_scale_root"]),
        "q_visit_exp_root": float(PLAY_SEARCH_DEFAULTS["q_visit_exp_root"]),
        "gumbel_topk": int(PLAY_SEARCH_DEFAULTS["topk"]),
    }
    inert_overrides = [
        name for name, default in gumbel_defaults.items()
        if getattr(args, name) != default
    ]
    if int(args.walkers) > 1 and inert_overrides:
        raise SystemExit(
            "--walkers >1 selects production walker PUCT; these classic-Gumbel "
            "overrides would be inert: " + ", ".join(inert_overrides)
        )
    positions, panel_selection = _select_audit_panel(
        audit_positions, int(args.max_positions),
    )
    _require_decision_grade_panel_selection(
        panel_selection, methodology_smoke=bool(args.methodology_smoke),
    )
    audit_set_artifact = initial_input_artifacts["audit_set"]
    assert isinstance(audit_set_artifact, dict)
    deep_reference_evidence = _deep_reference_evidence_summary([
        {
            "key": position.key,
            "fen": position.fen,
            **_deep_reference_position_fields(position),
        }
        for position in positions
    ], audit_set_sha256=audit_set_artifact.get("sha256"))
    if not args.methodology_smoke and deep_reference_evidence["passed"] is not True:
        failures = deep_reference_evidence["failing_positions"]
        first = failures[0] if isinstance(failures, list) and failures else None
        raise SystemExit(
            "decision-grade trajectory banks require every selected audit row "
            "to come from the approved frozen 1M-node/MultiPV-10 audit set and "
            "carry positive observed nodes/depth plus complete MultiPV coverage "
            f"evidence (first failure: {first!r})"
        )
    matched: MatchedAuditRows | None = None
    if matched_path.exists():
        matched = MatchedAuditRows(matched_path)
        matched.require_index_layout(require_game_ids=not args.methodology_smoke)
    elif not args.methodology_smoke:
        raise SystemExit(
            f"decision-grade trajectory banks require game groups from {matched_path}; "
            "build it with scripts/match_audit_rows.py, or pass --methodology-smoke "
            "for a non-decision-grade pipeline check"
        )
    if not args.methodology_smoke:
        if not matched_report_path.is_file():
            raise SystemExit(
                "decision-grade trajectory banks require the authenticated matched-row "
                f"report {matched_report_path}; rebuild the index with "
                "scripts/match_audit_rows.py"
            )
        matched_origin_verification = verify_selected_row_origins(
            audit_set=args.audit_set,
            matched_rows=matched_path,
            report_path=matched_report_path,
            selected=[{"key": pos.key, "fen": pos.fen} for pos in positions],
        )
    else:
        matched_origin_verification = {
            "schema": MATCHED_ROWS_REPORT_SCHEMA,
            "passed": False,
            "reason": "methodology_smoke_does_not_authenticate_source_clusters",
            "rows": [],
        }
    origin_snapshot = matched_origin_verification.get("snapshot_inventory")
    matched_snapshot_directories = (
        [Path(origin_snapshot["path"])]
        if isinstance(origin_snapshot, dict)
        and isinstance(origin_snapshot.get("path"), str)
        and origin_snapshot["path"]
        else []
    )
    if not args.methodology_smoke and not matched_snapshot_directories:
        raise SystemExit(
            "decision-grade matched-row verification lacks an authenticated snapshot path"
        )
    origin_protected_files = [
        args.audit_set,
        matched_path,
        matched_report_path,
        checkpoint_path,
        *checkpoint_params_candidates,
        Path(__file__),
        Path(publication_module.__file__),
        *([preregistration_path] if preregistration_path is not None else []),
    ]
    origin_protected_directories = [
        *syzygy_directories,
        *matched_snapshot_directories,
    ]
    # The authenticated snapshot path is only known after the independent
    # matched-row readback. Re-run the complete destination check before output
    # locks, staging, model load, or publication so neither --overwrite nor a
    # derived manifest/lock path can damage the evidence that established the
    # game clusters.
    _require_safe_output_paths(
        args.out,
        meta_path,
        protected_files=origin_protected_files,
        protected_directories=origin_protected_directories,
    )
    if args.write_preregistration is not None:
        _require_safe_preregistration_path(
            args.write_preregistration,
            protected_files=origin_protected_files,
            protected_directories=origin_protected_directories,
        )
    post_load_artifacts = {
        "audit_set": _artifact_if_file(args.audit_set),
        "matched_rows": _artifact_if_file(matched_path),
        "matched_rows_report": _artifact_if_file(matched_report_path),
    }
    input_load_changes = sorted(
        name for name in ("audit_set", "matched_rows", "matched_rows_report")
        if _artifact_identity(initial_input_artifacts[name])
        != _artifact_identity(post_load_artifacts[name])
    )
    if input_load_changes and not args.methodology_smoke:
        raise SystemExit(
            "decision-grade inputs changed while being loaded: "
            + ", ".join(input_load_changes)
        )
    game_ids: dict[str, int | None] = {}
    source_dirs: dict[str, str | None] = {}
    source_shards: dict[str, str | None] = {}
    group_ids: dict[str, str | None] = {}
    raw_origin_proofs = matched_origin_verification.get("rows")
    if not isinstance(raw_origin_proofs, list):
        raw_origin_proofs = []
    origin_proofs = {
        str(row["key"]): row
        for row in raw_origin_proofs
        if isinstance(row, dict) and "key" in row
    }
    for pos in positions:
        proof = origin_proofs.get(pos.key)
        selected_origin = proof.get("selected_origin") if isinstance(proof, dict) else None
        if not args.methodology_smoke and not isinstance(selected_origin, dict):
            raise SystemExit(f"selected audit key {pos.key!r} lacks origin readback proof")
        gid = (
            int(selected_origin["game_id"])
            if isinstance(selected_origin, dict)
            else matched.game_id(pos.key)
            if matched is not None and pos.key in matched else None
        )
        source_dir = (
            str(proof["source_dir"])
            if isinstance(proof, dict)
            else matched.snapshot
            if matched is not None and pos.key in matched else None
        )
        source_shard = (
            str(selected_origin["shard"])
            if isinstance(selected_origin, dict)
            else matched.source_shard(pos.key)
            if matched is not None and pos.key in matched else None
        )
        source_cluster_unique = bool(
            proof.get("source_cluster_unique")
            if isinstance(proof, dict)
            else matched is not None
            and pos.key in matched
            and matched.source_cluster_is_unique(pos.key)
        )
        if (
            (
                gid is None or gid < 0 or not source_dir or not source_shard
                or not source_cluster_unique
            )
            and not args.methodology_smoke
        ):
            raise SystemExit(
                "decision-grade trajectory bank lacks an unambiguous "
                f"(source_dir, game_id) cluster and source shard for audit key {pos.key!r}"
            )
        game_ids[pos.key] = gid
        source_dirs[pos.key] = source_dir
        source_shards[pos.key] = source_shard
        # DiskReplayBuffer flushes fixed-size prefixes, so one add_many(game)
        # call can straddle a shard boundary. The shard identifies the row's
        # origin but cannot be part of the statistical game cluster.
        group_ids[pos.key] = (
            None
            if gid is None or not source_dir or not source_shard
            else "\0".join((source_dir, str(gid)))
        )
    _require_analyzable_source_groups(
        group_ids, methodology_smoke=bool(args.methodology_smoke),
    )

    output_locks: tuple[IO[bytes], ...] = ()
    if args.write_preregistration is None:
        # Reserve collection destinations before importing/loading the CUDA model.
        # Plan generation writes a distinct tracked artifact and is intentionally exempt.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        output_locks = _acquire_output_locks(args.out, meta_path)
        try:
            resumed_publication = _require_new_output_pair(
                args.out, meta_path, overwrite=bool(args.overwrite),
            )
        except BaseException:
            for output_lock in reversed(output_locks):
                output_lock.close()
            raise
        if resumed_publication:
            for output_lock in reversed(output_locks):
                output_lock.close()
            print(f"[traj] recovered evidence pair -> {args.out}")
            print(f"[traj] provenance -> {meta_path}")
            return

    # Take the authoritative negative/positive lookup snapshot only after this
    # process's output-directory and lock creation.  From here through model
    # loading and final publication, every candidate and redirecting ancestor
    # must remain byte-for-byte identical.
    initial_params_candidate_inventory = _params_candidate_inventory(checkpoint_path)

    provenance: dict[str, Any] = {
        "schema": _SCHEMA,
        "decision_grade": not args.methodology_smoke,
        "analysis_scope": "fixed_node_horizons_only",
        "clock_conditioning_available": False,
        "root_position_history": "fen_only_from_audit_fen",
        "root_tree_state": "fresh_per_position_no_cross_move_reuse",
        "game_group_kind": "source_dir:game_id",
        "panel_selection": panel_selection,
        "deep_reference_evidence": deep_reference_evidence,
        "complexity_predicate": {
            "kind": "clock_free_visit_gap_and_stability",
            "minimum_stable_chunks": int(_ABORT_MIN_STABLE_CHUNKS),
            "minimum_visit_gap": float(_ABORT_VISIT_GAP_MARGIN),
            "single_legal_move_is_decided": True,
        },
        "producer_git_sha": producer_git_sha,
        "producer_git_dirty": producer_git_dirty,
        "python_preimport": {
            **_PREIMPORT_PYTHON_SOURCES,
            "start_check": preimport_start_status,
            "source_only_import": _SOURCE_ONLY_IMPORT_GUARD.status(),
        },
        "producer_script": initial_input_artifacts["producer_script"],
        "publication_helper": initial_input_artifacts["publication_helper"],
        "checkpoint": initial_input_artifacts["checkpoint"],
        "checkpoint_params": initial_input_artifacts["checkpoint_params"],
        "params_candidate_inventory": initial_params_candidate_inventory,
        "audit_set": initial_input_artifacts["audit_set"],
        "matched_rows": initial_input_artifacts["matched_rows"],
        "matched_rows_report": initial_input_artifacts["matched_rows_report"],
        "matched_row_origin_verification": matched_origin_verification,
        "preregistration": initial_input_artifacts["preregistration"],
        "preregistration_document": preregistration_document,
        "syzygy": syzygy_inventory,
    }

    import torch

    from chess_anti_engine.encoding import _features_ext as features_extension
    from chess_anti_engine.encoding import _lc0_ext as lc0_extension
    from chess_anti_engine.encoding.cboard_encode import CBoard
    from chess_anti_engine.mcts.gumbel import GumbelConfig
    from chess_anti_engine.mcts import _mcts_tree as mcts_extension
    from chess_anti_engine.mcts.gumbel_c import _REQUIRED_MCTS_ABI
    from chess_anti_engine.tablebase import SyzygyProbe
    from chess_anti_engine.uci.__main__ import _make_evaluator_factory
    from chess_anti_engine.uci.model_loader import (
        load_model_from_checkpoint_artifacts,
    )
    from chess_anti_engine.uci.search import SearchWorker
    from chess_anti_engine.uci.time_manager import Deadline
    from chess_anti_engine.worker import _configure_shared_compile_cache

    producer_sources = _producer_python_source_artifacts(
        producer_git_sha,
        require_tracked=not args.methodology_smoke,
    )
    python_post_import_status = _preimport_python_surface_status(
        _PREIMPORT_PYTHON_SOURCES
    )
    provenance["python_preimport"]["post_import_check"] = python_post_import_status
    provenance["python_preimport"][
        "source_only_import"
    ] = _SOURCE_ONLY_IMPORT_GUARD.status()
    source_guard_status = provenance["python_preimport"]["source_only_import"]
    loaded_project_status = source_guard_status.get("loaded_project_modules")
    if not (
        source_guard_status.get("active") is True
        and source_guard_status.get("first_finder") is True
        and isinstance(loaded_project_status, dict)
        and loaded_project_status.get("passed") is True
        and loaded_project_status.get("unverified_modules") == []
    ) and not args.methodology_smoke:
        raise SystemExit(
            "decision-grade producer loaded an unauthenticated project module or "
            "lost source-guard precedence"
        )
    if (
        not args.methodology_smoke
        and python_post_import_status.get("passed") is not True
    ):
        raise SystemExit(
            "tracked Python sources changed while producer modules imported: "
            + ", ".join(python_post_import_status.get("changed", []))
        )
    missing_producer_sources = sorted(
        _REQUIRED_PRODUCER_SOURCE_MODULES - producer_sources.keys()
    )
    if missing_producer_sources and not args.methodology_smoke:
        raise SystemExit(
            "decision-grade producer source inventory is incomplete: "
            + ", ".join(missing_producer_sources)
        )
    producer_source_import_changes = sorted(
        name for name, artifact in initial_producer_sources.items()
        if _artifact_identity(artifact)
        != _artifact_identity(producer_sources.get(name))
    )
    if producer_source_import_changes and not args.methodology_smoke:
        raise SystemExit(
            "decision-grade producer sources changed while runtime modules imported: "
            + ", ".join(producer_source_import_changes)
    )
    provenance["producer_sources"] = producer_sources
    loaded_halving_rev = int(getattr(mcts_extension, "GSS_HALVING_REV", 0))
    loaded_mcts_artifact = _artifact(
        Path(mcts_extension.__file__), require_file=True,
    )
    loaded_features_artifact = _artifact(
        Path(features_extension.__file__), require_file=True,
    )
    loaded_lc0_artifact = _artifact(
        Path(lc0_extension.__file__), require_file=True,
    )
    features_module = "chess_anti_engine.encoding._features_ext"
    mcts_module = "chess_anti_engine.mcts._mcts_tree"
    lc0_module = "chess_anti_engine.encoding._lc0_ext"
    native_modules = [features_module, lc0_module, mcts_module]
    native_build_attestations = {
        features_module: _loaded_native_build_attestation(
            features_extension, features_module, producer_git_sha,
        ),
        lc0_module: _loaded_native_build_attestation(
            lc0_extension, lc0_module, producer_git_sha,
        ),
        mcts_module: _loaded_native_build_attestation(
            mcts_extension, mcts_module, producer_git_sha,
        ),
    }
    native_import_changes = sorted(
        module for module, loaded in (
            (features_module, loaded_features_artifact),
            (lc0_module, loaded_lc0_artifact),
            (mcts_module, loaded_mcts_artifact),
        )
        if _artifact_identity(PREIMPORT_NATIVE_ARTIFACTS[module])
        != _artifact_identity(loaded)
    )
    extension_issues = check_extensions(
        Path(__file__).resolve().parents[1],
        min_gcc_major=15,
        require_production_recipe=True,
        modules=native_modules,
        loaded_paths={
            features_module: Path(features_extension.__file__),
            lc0_module: Path(lc0_extension.__file__),
            mcts_module: Path(mcts_extension.__file__),
        },
    )
    extension_issues.extend(
        f"{module} changed between pre-import snapshot and loaded-path readback"
        for module in native_import_changes
    )
    extension_issues.extend(
        f"{module} was not built from producer revision {producer_git_sha}"
        for module, attestation in native_build_attestations.items()
        if attestation.get("matches_producer_revision") is not True
    )
    if extension_issues and not args.methodology_smoke:
        raise SystemExit(
            "decision-grade trajectory banks require a fresh production-built "
            "MCTS extension: " + "; ".join(extension_issues)
        )
    _require_safe_output_paths(
        args.out,
        meta_path,
        protected_files=[
            args.audit_set,
            matched_path,
            matched_report_path,
            checkpoint_path,
            *checkpoint_params_candidates,
            Path(__file__),
            Path(publication_module.__file__),
            Path(features_extension.__file__),
            Path(mcts_extension.__file__),
            Path(lc0_extension.__file__),
            *([preregistration_path] if preregistration_path is not None else []),
        ],
        protected_directories=origin_protected_directories,
    )
    if not args.methodology_smoke and loaded_halving_rev != _PRODUCTION_GSS_HALVING_REV:
        raise SystemExit(
            "decision-grade trajectory banks require MCTS "
            f"GSS_HALVING_REV={_PRODUCTION_GSS_HALVING_REV}, loaded "
            f"{loaded_halving_rev} from {mcts_extension.__file__}"
        )

    (
        model,
        checkpoint_artifact,
        checkpoint_params_artifact,
        model_input_consumption,
    ) = _load_authenticated_model_inputs(
        checkpoint_path,
        initial_params_candidate_inventory,
        loader=load_model_from_checkpoint_artifacts,
        device=args.device,
        require_complete=not args.methodology_smoke,
    )
    provenance["checkpoint"] = checkpoint_artifact
    provenance["checkpoint_params"] = checkpoint_params_artifact
    provenance["model_input_consumption"] = model_input_consumption
    checkpoint_params_path = _selected_params_candidate(
        initial_params_candidate_inventory
    )
    model.eval()
    # The helper raises before this point unless the exact file description
    # consumed by torch and the exact params bytes remained stable. Avoid a
    # redundant multi-gigabyte pathname re-hash here; final run provenance
    # still rechecks the live pathname after collection.
    checkpoint_load_changes: list[str] = []
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    pol_enc = str(getattr(model, "policy_encoding", "lc0_1858"))
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    engine_options = EngineOptions()
    cfg = GumbelConfig(
        simulations=int(args.chunk_sims), add_noise=False, temperature=0.0,
        input_history_encoding=hist, input_extra_features=extra, policy_encoding=pol_enc,
        compute_relations=use_rel, topk=int(args.gumbel_topk),
        c_scale=float(args.c_scale), c_visit=float(args.c_visit),
        c_visit_root=float(args.c_visit_root),
        c_scale_root=float(args.c_scale_root), q_visit_exp_root=float(args.q_visit_exp_root),
        policy_temp=float(PLAY_SEARCH_DEFAULTS["policy_temp"]),
        halving_div=int(engine_options.halving_div),
        c_puct=float(PLAY_PUCT_DEFAULTS["c_puct"]),
        cpuct_factor=float(PLAY_PUCT_DEFAULTS["cpuct_factor"]),
        cpuct_base=float(PLAY_PUCT_DEFAULTS["cpuct_base"]),
        fpu_reduction=float(PLAY_PUCT_DEFAULTS["fpu_reduction"]),
    )
    try:
        validate_gumbel_config(cfg, where="backtest_chunk_trajectory")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    compile_mode = str(args.compile_mode) if args.compile else None
    if compile_mode is not None:
        _configure_shared_compile_cache(
            cache_dir=Path(args.compile_cache_dir).expanduser(),
        )
    evaluator_factory = _make_evaluator_factory(
        [model], (str(args.device),), True, int(args.walkers),
        int(engine_options.leaf_gather), compile_mode=compile_mode,
    )
    evaluator = evaluator_factory(int(engine_options.max_batch))
    direct = _find_direct_evaluator(evaluator)
    compact_bf16 = os.environ.get("CAE_UCI_COMPACT_BF16", "0") == "1"
    worker = SearchWorker(
        evaluator, device=args.device,
        gumbel_cfg=cfg, chunk_sims=int(args.chunk_sims), n_walkers=int(args.walkers),
        vloss_weight=int(PLAY_SEARCH_VLOSS_WEIGHT),
        walker_gather=int(engine_options.leaf_gather),
    )
    worker.set_max_tree_mb(int(engine_options.hash_mb))
    if int(args.walkers) == 1:
        worker.set_minibatch_size(int(PLAY_SEARCH_TARGET_BATCH))
    worker.set_root_noise_scale(float(engine_options.root_noise_scale))
    tb_probe = SyzygyProbe(str(args.syzygy_path)) if args.syzygy_path else None
    worker.set_tb_probe(tb_probe)
    installed_tb_probe = getattr(worker, "_tb_probe", None)
    if installed_tb_probe is not tb_probe:
        worker.close()
        raise RuntimeError("tablebase take-effect check failed")
    concurrency_mode, concurrency_workers = worker.concurrency_profile()
    positive_control: dict[str, Any] | None = None
    root_shortcut_control: dict[str, Any] | None = None
    if installed_tb_probe is not None:
        control_fen = "7k/8/8/8/8/8/8/KQ6 w - - 0 1"
        control_board = chess.Board(control_fen)
        installed_tb_probe.reset_counts()
        root_result = worker.run(
            control_board,
            stop_event=threading.Event(),
            deadline=Deadline(deadline_ms=None),
            max_nodes=1,
            optimum_ms=None,
            allow_terminal_shortcuts=True,
        )
        try:
            root_move = chess.Move.from_uci(root_result.bestmove_uci)
        except ValueError:
            root_move = None
        root_shortcut_control = {
            "fen": control_fen,
            "bestmove_uci": root_result.bestmove_uci,
            "nodes": int(root_result.nodes),
            "tbhits": int(root_result.tbhits),
            "root_declined": root_result.root_declined,
            "tree_created": getattr(worker, "_tree", None) is not None,
            "passed": bool(
                root_move is not None
                and root_move in control_board.legal_moves
                and int(root_result.nodes) == 1
                and int(root_result.tbhits) == 1
                and root_result.root_declined is None
                and getattr(worker, "_tree", None) is None
            ),
        }
        worker.reset_tree()
        control_wdl = np.zeros((1, 3), dtype=np.float32)
        installed_tb_probe.reset_counts()
        control_return = int(installed_tb_probe.apply(
            [CBoard.from_board(chess.Board(control_fen))], control_wdl,
        ))
        positive_control = {
            "fen": control_fen,
            "probes": int(installed_tb_probe.probes),
            "hits": int(installed_tb_probe.hits),
            "apply_return": control_return,
            "passed": bool(
                control_return == 1
                and int(installed_tb_probe.probes) == 1
                and int(installed_tb_probe.hits) == 1
            ),
        }
        installed_tb_probe.reset_counts()
        if (
            not positive_control["passed"]
            or not root_shortcut_control["passed"]
        ) and not args.methodology_smoke:
            worker.close()
            raise RuntimeError(
                "production Syzygy root/leaf positive controls did not take effect"
            )
    realized_tablebase = {
        "installed": installed_tb_probe is not None,
        "cursed_as_draw": bool(
            getattr(installed_tb_probe, "_cursed_as_draw", False)
        ),
        "n_wdl": int(getattr(installed_tb_probe, "n_wdl", 0)),
        "n_dtz": int(getattr(installed_tb_probe, "n_dtz", 0)),
        "max_pieces": int(getattr(installed_tb_probe, "max_pieces", 0)),
        "root_probe_active": bool(
            root_shortcut_control is not None and root_shortcut_control["passed"]
        ),
        "leaf_probe_active": (
            installed_tb_probe is not None and concurrency_mode == "gumbel"
        ),
        "positive_control": positive_control,
        "root_shortcut_positive_control": root_shortcut_control,
    }
    expected_mode = "walker_puct" if int(args.walkers) > 1 else "gumbel"
    active_parameters: dict[str, float | int | bool] = (
        {
            **{name: float(value) for name, value in PLAY_PUCT_DEFAULTS.items()},
            "vloss_weight": int(PLAY_SEARCH_VLOSS_WEIGHT),
            "walker_gather": int(engine_options.leaf_gather),
        }
        if concurrency_mode == "walker_puct"
        else {
            "c_scale": float(args.c_scale),
            "c_visit": float(args.c_visit),
            "c_visit_root": float(args.c_visit_root),
            "c_scale_root": float(args.c_scale_root),
            "q_visit_exp_root": float(args.q_visit_exp_root),
            "topk": int(args.gumbel_topk),
            "policy_temp": float(PLAY_SEARCH_DEFAULTS["policy_temp"]),
            "halving_div": int(engine_options.halving_div),
            "root_noise_scale": float(engine_options.root_noise_scale),
            "vloss_weight": int(PLAY_SEARCH_VLOSS_WEIGHT),
            "minibatch_size": int(PLAY_SEARCH_TARGET_BATCH),
        }
    )
    requested_search = {
        "device": str(args.device),
        "chunk_sims": int(args.chunk_sims),
        "max_chunks": int(args.max_chunks),
        "walkers": int(args.walkers),
        "active_path": concurrency_mode,
        "active_parameters": active_parameters,
    }
    _validate_registry_search_values(
        concurrency_mode,
        {"chunk_sims": int(args.chunk_sims), **active_parameters},
    )
    realized_search: dict[str, float | int | bool | str] = {
        **worker.realized_search_values(),
        "concurrency_mode": concurrency_mode,
        "concurrency_workers": concurrency_workers,
    }
    if concurrency_mode == "walker_puct":
        walker_pool = getattr(worker, "_walker_pool", None)
        pool_config = getattr(walker_pool, "_cfg", None)
        realized_search["walker_gather"] = int(getattr(pool_config, "gather", 0))
    expected_evaluator_stack = (
        "BatchCoalescingDispatcher(ThreadSafeGPUDispatcher(DirectGPUEvaluator))"
        if int(args.walkers) > 1
        else (
            "CUDAOwnerDispatcher(ThreadSafeGPUDispatcher(DirectGPUEvaluator))"
            if compile_mode is not None
            else "ThreadSafeGPUDispatcher(DirectGPUEvaluator)"
        )
    )
    realized_evaluator = {
        "stack": _evaluator_stack_name(evaluator),
        "direct_max_batch": int(direct._max_batch),
        "outer_max_batch": int(getattr(evaluator, "_max_batch", direct._max_batch)),
        "n_slots": int(direct.n_slots),
        "input_bf16": bool(direct.supports_input_bf16_bits),
        "legal_bf16": bool(direct.supports_legal_bf16),
        "compiled": hasattr(direct.model, "_orig_mod"),
        "model_wrapper_type": type(direct.model).__name__,
    }
    live_model = getattr(direct.model, "_orig_mod", direct.model)
    expected_planes = int(input_plane_count(extra))
    expected_model_search_contract = {
        "model_input_history_encoding": hist,
        "model_input_extra_features": extra,
        "model_policy_encoding": pol_enc,
        "model_compute_relations": use_rel,
        "search_input_history_encoding": hist,
        "search_input_extra_features": extra,
        "search_policy_encoding": pol_enc,
        "search_compute_relations": use_rel,
        "evaluator_input_planes": expected_planes,
        "walker_input_planes": expected_planes if concurrency_mode == "walker_puct" else None,
        "walker_compute_relations": use_rel if concurrency_mode == "walker_puct" else None,
    }
    pool_config = getattr(getattr(worker, "_walker_pool", None), "_cfg", None)
    realized_model_search_contract = {
        "model_input_history_encoding": str(
            getattr(live_model, "input_history_encoding", "legacy")
        ),
        "model_input_extra_features": str(
            getattr(live_model, "input_extra_features", "v1")
        ),
        "model_policy_encoding": str(
            getattr(live_model, "policy_encoding", "lc0_1858")
        ),
        "model_compute_relations": bool(
            getattr(live_model, "use_dynamic_relations", False)
        ),
        "search_input_history_encoding": str(worker._cfg.input_history_encoding),
        "search_input_extra_features": str(worker._cfg.input_extra_features),
        "search_policy_encoding": str(worker._cfg.policy_encoding),
        "search_compute_relations": bool(worker._cfg.compute_relations),
        "evaluator_input_planes": int(direct._pinned_inputs[0].shape[1]),
        "walker_input_planes": (
            int(getattr(pool_config, "input_planes", -1))
            if concurrency_mode == "walker_puct" else None
        ),
        "walker_compute_relations": (
            bool(getattr(pool_config, "compute_relations", False))
            if concurrency_mode == "walker_puct" else None
        ),
    }
    if realized_model_search_contract != expected_model_search_contract:
        worker.close()
        raise RuntimeError(
            "model/search encoding take-effect check failed: requested "
            f"{expected_model_search_contract!r}, realized "
            f"{realized_model_search_contract!r}"
        )
    model_devices = sorted({str(parameter.device) for parameter in direct.model.parameters()})
    def resolve_cuda_device(value: str) -> str:
        device = torch.device(value)
        if device.type != "cuda":
            return str(device)
        index = getattr(device, "index", None)
        if index is None:
            index = int(torch.cuda.current_device())
        return f"cuda:{index}"

    resolved_requested_device = resolve_cuda_device(str(args.device))
    resolved_evaluator_device = resolve_cuda_device(str(direct.device))
    resolved_model_devices = sorted({resolve_cuda_device(value) for value in model_devices})
    nvidia_driver_version = (
        _nvidia_driver_version(int(resolved_requested_device.split(":", 1)[1]))
        if resolved_requested_device.startswith("cuda:") else None
    )
    try:
        _require_nvidia_driver_provenance(
            nvidia_driver_version, methodology_smoke=bool(args.methodology_smoke),
        )
    except RuntimeError:
        worker.close()
        raise
    runtime = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": str(Path(sys.executable).resolve()),
        "numpy_version": str(np.__version__),
        "python_chess_version": str(chess.__version__),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_driver_version": nvidia_driver_version,
        "requested_device": str(args.device),
        "evaluator_device": str(direct.device),
        "model_parameter_devices": model_devices,
        "resolved_requested_device": resolved_requested_device,
        "resolved_evaluator_device": resolved_evaluator_device,
        "resolved_model_parameter_devices": resolved_model_devices,
        "cuda_device_name": (
            str(torch.cuda.get_device_name(torch.device(args.device)))
            if str(args.device).startswith("cuda") else None
        ),
        "cuda_device_capability": (
            list(torch.cuda.get_device_capability(torch.device(args.device)))
            if str(args.device).startswith("cuda") else None
        ),
    }
    if (
        not resolved_model_devices
        or resolved_evaluator_device != resolved_requested_device
        or any(device != resolved_requested_device for device in resolved_model_devices)
    ) and not args.methodology_smoke:
        worker.close()
        raise RuntimeError("model/evaluator device readback is not the requested CUDA device")
    expected_evaluator = {
        "stack": expected_evaluator_stack,
        "direct_max_batch": int(engine_options.max_batch),
        "outer_max_batch": int(engine_options.max_batch),
        "n_slots": 2,
        "input_bf16": compact_bf16,
        "legal_bf16": compact_bf16,
        "compiled": compile_mode is not None,
        "model_wrapper_type": "OptimizedModule" if compile_mode is not None else type(model).__name__,
    }
    if realized_evaluator != expected_evaluator:
        worker.close()
        raise RuntimeError(
            "evaluator take-effect check failed: requested "
            f"{expected_evaluator!r}, realized {realized_evaluator!r}"
        )
    try:
        _require_search_take_effect(
            expected_mode=expected_mode,
            expected_workers=int(args.walkers),
            active_parameters=active_parameters,
            realized=realized_search,
        )
    except RuntimeError:
        worker.close()
        raise
    compile_contract = {
        "enabled": compile_mode is not None,
        "mode": compile_mode,
        "cache_dir": str(Path(args.compile_cache_dir).expanduser().resolve()),
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
    }
    preregistration_manifest = {
        **provenance,
        "requested_max_positions": int(args.max_positions),
        "requested_search": requested_search,
        "requested_model_search_contract": expected_model_search_contract,
        "requested_evaluator": expected_evaluator,
        "compile": compile_contract,
    }
    if args.write_preregistration is not None:
        if _params_candidate_inventory(checkpoint_path) != (
            initial_params_candidate_inventory
        ):
            worker.close()
            raise SystemExit(
                "params.json candidate inventory changed before preregistration write"
            )
        written = _write_preregistration_plan(
            args.write_preregistration,
            _preregistration_payload(preregistration_manifest),
            protected_files=[
                args.audit_set,
                matched_path,
                matched_report_path,
                checkpoint_path,
                *checkpoint_params_candidates,
                Path(__file__),
                Path(publication_module.__file__),
                Path(features_extension.__file__),
                Path(mcts_extension.__file__),
                Path(lc0_extension.__file__),
            ],
            protected_directories=origin_protected_directories,
        )
        if _params_candidate_inventory(checkpoint_path) != (
            initial_params_candidate_inventory
        ):
            worker.close()
            raise SystemExit(
                "params.json candidate inventory changed during preregistration write"
            )
        worker.close()
        print(f"[traj] wrote preregistration plan -> {written}")
        print("[traj] commit the plan, then rerun with --preregistration")
        return
    preregistration_failures = _preregistered_design_failures(
        preregistration_manifest
    )
    if preregistration_failures and not args.methodology_smoke:
        worker.close()
        raise SystemExit(
            "collection does not match its tracked preregistration: "
            + "; ".join(preregistration_failures)
        )
    warmup_nodes = max(256, int(args.chunk_sims))
    try:
        warmup_result = worker.run(
            chess.Board(),
            stop_event=threading.Event(),
            deadline=Deadline(deadline_ms=None),
            max_nodes=warmup_nodes,
            optimum_ms=None,
            allow_terminal_shortcuts=True,
        )
    except Exception:
        worker.close()
        raise
    finally:
        worker.reset_tree()
        if installed_tb_probe is not None:
            installed_tb_probe.reset_counts()
    search_warmup = {
        "completed": int(warmup_result.nodes) == warmup_nodes,
        "requested_nodes": warmup_nodes,
        "realized_nodes": int(warmup_result.nodes),
        "excluded_from_timing": True,
        "tree_reset_after": True,
        "tablebase_counters_reset_after": True,
    }
    if not search_warmup["completed"] and not args.methodology_smoke:
        worker.close()
        raise RuntimeError(
            "production search warmup did not complete its requested node budget"
        )
    print(f"[traj] {len(positions)} positions x {args.max_chunks} chunks of {args.chunk_sims} "
          f"(path={concurrency_mode} workers={concurrency_workers})",
          flush=True)
    def action_to_uci(act: int, board: chess.Board, ucis: list[str]) -> tuple[str | None, int]:
        # The worker's own action->move decode (handles policy encoding + orientation);
        # then locate it in the position's legal-move order for the regret lookup.
        try:
            if not 0 <= int(act) < POLICY_SIZE:
                raise ActionDecodeError(act, board, "outside the native action space")
            uci = index_to_move_strict(int(act), board).uci()
        except ActionDecodeError:
            return None, -1
        return (uci, ucis.index(uci)) if uci in ucis else (uci, -1)

    tmp_path = _pending_output_path(args.out)
    n_rows = 0
    completed_positions = 0
    completed_group_ids: dict[str, str] = {}
    reference_censoring_details: list[dict[str, Any]] = []
    excluded_positions: list[dict[str, Any]] = []
    started = time.perf_counter()
    collection_started = False
    active_position: Any | None = None
    active_snapshots: list[dict[str, Any]] = []
    active_search_result: dict[str, Any] | None = None
    completed_output_artifact: dict[str, Any]
    retained_output_fd = -1
    retained_output_parent_fd = -1
    _ACTIVE_PENDING_EVIDENCE = {
        "collection_complete": False,
        "pending_output": tmp_path,
        "output": args.out,
        "manifest": meta_path,
        "pending_manifest": pending_meta_path,
        "output_locks": output_locks,
        "provenance": provenance,
        "requested_position_count": len(positions),
        "requested_max_positions": int(args.max_positions),
        "excluded_positions": excluded_positions,
        "reference_censoring_details": reference_censoring_details,
    }
    try:
        with _open_staged_output_file(tmp_path) as (fh, pending_parent_fd):
            retained_output_fd = os.dup(fh.fileno())
            try:
                retained_output_parent_fd = os.dup(pending_parent_fd)
            except BaseException:
                os.close(retained_output_fd)
                retained_output_fd = -1
                raise
            collection_started = True
            for pi, pos in enumerate(positions):
                active_position = pos
                active_snapshots = []
                active_search_result = None
                board = chess.Board(pos.fen)
                ucis, legal_actions = legal_full_indices(board)
                regrets = move_regrets(pos, ucis)
                worst_listed_cp = min(pos.move_cp.values())
                regret_by_action = {
                    int(action): float(regret)
                    for action, regret in zip(legal_actions, regrets)
                }
                reference_by_action = {
                    int(action): (
                        float(pos.move_cp.get(uci, worst_listed_cp)),
                        uci in pos.move_cp,
                    )
                    for uci, action in zip(ucis, legal_actions, strict=True)
                }
                worker.reset_tree()
                snaps = active_snapshots
                position_started = time.perf_counter()

            # Default-arg binding captures this iteration's loop vars (avoids
            # cell-var-from-loop); the list defaults are intentional, not a bug.
                def on_chunk(
                    total_nodes: int, _b=board, _u=ucis, _s=snaps,
                    _t0=position_started, _best_cp=pos.best_cp,
                    _regret_by_action=regret_by_action,
                    _reference_by_action=reference_by_action,
                ) -> None:
                    actions, visits = worker._filtered_root_visits(None)
                    if actions.size == 0:
                        return
                    best = worker._emitted_action(actions, visits, None)
                    uci, li = action_to_uci(int(best), _b, _u)
                    if uci is None or li < 0:
                        raise RuntimeError(
                            "search emitted an action absent from the audit reference"
                        )
                    if any(int(action) not in _regret_by_action for action in actions):
                        raise RuntimeError(
                            "search exposed an action absent from the audit reference"
                        )
                    action_regret_cp = [
                        _regret_by_action[int(action)] for action in actions
                    ]
                    action_reference_cp = [
                        _reference_by_action[int(action)][0] for action in actions
                    ]
                    action_reference_listed = [
                        _reference_by_action[int(action)][1] for action in actions
                    ]
                    tot = float(visits.sum())
                    shares = (
                        visits.astype(np.float64) / tot
                        if tot > 0 else visits.astype(np.float64)
                    )
                    ngap = _visit_gap(actions, visits, int(best))
                    rq = 0.0
                    qg: float | None = None
                    child_q: list[float] = []
                    child_q_observed: list[bool] = []
                    pv_actions: list[int] = []
                    pv_uci: list[str] = []
                    tree, rid = worker._tree, worker._root_id
                    if tree is not None and rid is not None:
                        rq = float(tree.node_q(rid))
                        ca, _cv, cq = tree.get_children_q(rid, rq)
                        q_by_action = {
                            int(action): float(q)
                            for action, q in zip(ca.tolist(), cq.tolist())
                        }
                        if any(int(action) not in q_by_action for action in actions):
                            raise RuntimeError(
                                "search omitted a root action from child-Q readback"
                            )
                        child_q = [q_by_action[int(action)] for action in actions]
                        child_q_observed = [int(visit) > 0 for visit in visits]
                        if not math.isfinite(rq) or not all(map(math.isfinite, child_q)):
                            raise RuntimeError("search returned a non-finite root Q")
                        best_index = [int(action) for action in actions].index(int(best))
                        other_q = [
                            q for index, q in enumerate(child_q)
                            if index != best_index and child_q_observed[index]
                        ]
                        if child_q_observed[best_index] and other_q:
                            qg = child_q[best_index] - max(other_q)
                            if not math.isfinite(qg):
                                raise RuntimeError(
                                    "search returned a non-finite calculated q_gap"
                                )
                        pv_actions = _pv_from_root_action(tree, rid, int(best))
                        pv_uci = _strict_uci_pv(_b, pv_actions)
                    if (
                        len(child_q) != len(actions)
                        or not pv_actions
                        or len(pv_uci) != len(pv_actions)
                    ):
                        raise RuntimeError(
                            "search did not expose complete root Q/PV observations"
                        )
                    regret_cp = _regret_by_action[int(best)]
                    emitted_reference_cp = _reference_by_action[int(best)][0]
                    emitted_reference_listed = _reference_by_action[int(best)][1]
                    _s.append({
                        "nodes": total_nodes, "elapsed_ms": (time.perf_counter() - _t0) * 1000.0,
                        "emitted_action": int(best), "uci": uci,
                        "regret_cp": regret_cp,
                        "regret_score": _score(_best_cp) - _score(emitted_reference_cp),
                        "visit_gap": ngap, "visit_entropy": _entropy(shares),
                        "q_gap": qg, "root_q": rq,
                        "deep_reference_best_cp": float(_best_cp),
                        "actions": [int(a) for a in actions],
                        "visits": [int(v) for v in visits],
                        "shares": {int(a): float(s) for a, s in zip(actions, shares)},
                        "child_q": child_q,
                        "child_q_observed": child_q_observed,
                        "action_regret_cp": action_regret_cp,
                        "action_reference_cp": action_reference_cp,
                        "action_reference_listed": action_reference_listed,
                        "emitted_reference_listed": emitted_reference_listed,
                        "pv_actions": pv_actions,
                        "pv_uci": pv_uci,
                        "tb_probes": int(getattr(installed_tb_probe, "probes", 0)),
                        "tb_hits": int(getattr(installed_tb_probe, "hits", 0)),
                    })

                search_result = worker.run(
                    board, stop_event=threading.Event(), deadline=Deadline(deadline_ms=None),
                    max_nodes=int(args.max_chunks) * int(args.chunk_sims), optimum_ms=None,
                    allow_terminal_shortcuts=True, on_chunk=on_chunk,
                )
                active_search_result = {
                    "bestmove_uci": search_result.bestmove_uci,
                    "nodes": int(search_result.nodes),
                    "tbhits": int(search_result.tbhits),
                    "score_cp": int(search_result.score_cp),
                    "score_mate": search_result.score_mate,
                    "root_declined": search_result.root_declined,
                    "pv": list(search_result.pv),
                    "board_game_over": board.is_game_over(),
                }
                if len(snaps) != int(args.max_chunks):
                    terminal_shortcut = bool(
                        not snaps
                        and int(search_result.nodes) <= 1
                        and search_result.root_declined is None
                        and (
                            search_result.score_mate is not None
                            or int(search_result.tbhits) > 0
                            or board.is_game_over()
                        )
                    )
                    excluded_positions.append(
                        _excluded_position_evidence(
                            pos,
                            source_dir=source_dirs[pos.key],
                            source_shard=source_shards[pos.key],
                            game_id=game_ids[pos.key],
                            group_id=group_ids[pos.key],
                            chunks_required=int(args.max_chunks),
                            snapshots=snaps,
                            reason=(
                                "production_terminal_shortcut"
                                if terminal_shortcut
                                else "incomplete_search"
                            ),
                            search_result=active_search_result,
                        )
                    )
                    active_position = None
                    active_snapshots = []
                    active_search_result = None
                    continue
                final_uci = snaps[-1]["uci"]
                final_regret = float(snaps[-1]["regret_cp"])
                abort_last_best = -1
                stable_chunks = 0
                position_rows: list[dict[str, Any]] = []
                for k, s in enumerate(snaps):
                    prev = snaps[k - 1] if k > 0 else None
                    flip = bool(prev is not None and s["uci"] != prev["uci"])
                    abort_last_best, stable_chunks = _update_stability(
                        abort_last_best,
                        stable_chunks,
                        emitted_action=int(s["emitted_action"]),
                        visit_gap=float(s["visit_gap"]),
                        action_count=len(s["actions"]),
                    )
                    qdrift = abs(float(s["root_q"]) - float(prev["root_q"])) if prev else None
                    churn = None
                    if prev is not None:
                        keys = set(s["shares"]) | set(prev["shares"])
                        churn = 0.5 * sum(
                            abs(s["shares"].get(a, 0.0) - prev["shares"].get(a, 0.0))
                            for a in keys
                        )
                    row = {
                        "schema": _SCHEMA,
                        "key": pos.key,
                        "fen": pos.fen,
                        "source_dir": source_dirs[pos.key],
                        "shard": source_shards[pos.key],
                        "game_id": game_ids[pos.key],
                        "group_id": group_ids[pos.key],
                        "phase": pos.phase, "source": pos.source,
                        "piece_count": chess.popcount(board.occupied),
                        "legal_move_count": board.legal_moves.count(), "chunk": k + 1,
                        "nodes": s["nodes"], "elapsed_ms": s["elapsed_ms"],
                        "emitted_action": s["emitted_action"], "uci": s["uci"],
                        "regret_cp": s["regret_cp"], "regret_score": s["regret_score"],
                        **_deep_reference_position_fields(pos),
                        "visit_gap": s["visit_gap"], "visit_entropy": s["visit_entropy"],
                        "q_gap": s["q_gap"],
                        "root_q": s["root_q"], "root_actions": s["actions"],
                        "root_visits": s["visits"],
                        "root_visit_shares": [s["shares"][a] for a in s["actions"]],
                        "root_child_q": s["child_q"],
                        "root_child_q_observed": s["child_q_observed"],
                        "root_action_regret_cp": s["action_regret_cp"],
                        "root_action_reference_cp": s["action_reference_cp"],
                        "root_action_reference_listed": s["action_reference_listed"],
                        "emitted_reference_listed": s["emitted_reference_listed"],
                        "pv_actions": s["pv_actions"], "pv_uci": s["pv_uci"],
                        "tb_probes": s["tb_probes"], "tb_hits": s["tb_hits"],
                        "bestmove_flip": flip, "stable_chunks": stable_chunks,
                        "complexity_predicate_continue": _complexity_continue(
                            stable_chunks=stable_chunks,
                            visit_gap=float(s["visit_gap"]),
                            action_count=len(s["actions"]),
                        ),
                        "q_drift": qdrift, "visit_churn": churn,
                        "changes_to_final": bool(s["uci"] != final_uci),
                        "regret_vs_final_cp": float(s["regret_cp"]) - final_regret,
                    }
                    position_rows.append(row)
                censoring = _trajectory_reference_censoring(pos.key, position_rows)
                if censoring is not None:
                    reference_censoring_details.append(censoring)
                fh.write("".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in position_rows
                ))
                n_rows += len(position_rows)
                completed_positions += 1
                completed_group_id = group_ids[pos.key]
                if completed_group_id:
                    completed_group_ids[completed_group_id] = completed_group_id
                active_position = None
                active_snapshots = []
                active_search_result = None
                if (pi + 1) % 25 == 0:
                    print(f"[traj] {pi + 1}/{len(positions)}", flush=True)
                    if str(args.device).startswith("cuda"):
                        torch.cuda.empty_cache()
            completed_output_artifact = _durably_prepare_output_artifact(
                fh, tmp_path, args.out, parent_fd=pending_parent_fd,
            )
        _ACTIVE_PENDING_EVIDENCE.update({
            "collection_complete": True,
            "row_count": n_rows,
            "position_count": completed_positions,
            "output_artifact": completed_output_artifact,
            "retained_output_fd": retained_output_fd,
            "retained_output_parent_fd": retained_output_parent_fd,
        })
        retained_output_fd = -1
        retained_output_parent_fd = -1
    except BaseException as exc:
        if not collection_started:
            raise
        collection_error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if active_position is not None:
            excluded_positions.append(
                _excluded_position_evidence(
                    active_position,
                    source_dir=source_dirs.get(active_position.key),
                    source_shard=source_shards.get(active_position.key),
                    game_id=game_ids.get(active_position.key),
                    group_id=group_ids.get(active_position.key),
                    chunks_required=int(args.max_chunks),
                    snapshots=active_snapshots,
                    reason="collection_error",
                    search_result=active_search_result,
                    collection_error=collection_error,
                )
            )
        try:
            _write_invalid_recovery_diagnostic(
                tmp_path,
                args.out,
                meta_path,
                {
                    **provenance,
                    "decision_grade": False,
                    "complete": False,
                    "failure_stage": "trajectory_collection",
                    "collection_error": collection_error,
                    "raw_observations_preserved": True,
                    "reference_censoring": _reference_censoring_summary(
                        reference_censoring_details,
                    ),
                    "row_count": n_rows,
                    "position_count": completed_positions,
                    "excluded_positions": excluded_positions,
                },
                file_fd=retained_output_fd,
                parent_fd=retained_output_parent_fd,
            )
        finally:
            active_retained_output_fd = _ACTIVE_PENDING_EVIDENCE.get(
                "retained_output_fd"
            )
            active_retained_parent_fd = _ACTIVE_PENDING_EVIDENCE.get(
                "retained_output_parent_fd"
            )
            if (
                retained_output_fd >= 0
                and active_retained_output_fd != retained_output_fd
            ):
                os.close(retained_output_fd)
            if (
                retained_output_parent_fd >= 0
                and active_retained_parent_fd != retained_output_parent_fd
            ):
                os.close(retained_output_parent_fd)
            for output_lock in reversed(output_locks):
                output_lock.close()
        raise
    finally:
        worker.close()

    incomplete_exclusions = sum(
        entry["reason"] == "incomplete_search" for entry in excluded_positions
    )
    source_game_group_count = _source_game_group_count(completed_group_ids)
    source_group_resolution_passed = _source_group_resolution_passed(
        completed_group_ids,
    )
    reference_censoring = _reference_censoring_summary(reference_censoring_details)
    frozen_artifacts = {
        "producer_script": provenance["producer_script"],
        "publication_helper": provenance["publication_helper"],
        "checkpoint": provenance["checkpoint"],
        "checkpoint_params": provenance["checkpoint_params"],
        "audit_set": provenance["audit_set"],
        "matched_rows": provenance["matched_rows"],
        "matched_rows_report": provenance["matched_rows_report"],
        "preregistration": provenance["preregistration"],
        "features_extension": loaded_features_artifact,
        "mcts_extension": loaded_mcts_artifact,
        "lc0_extension": loaded_lc0_artifact,
    }
    try:
        final_params_candidate_inventory = _params_candidate_inventory(checkpoint_path)
    except (OSError, SystemExit):
        final_params_candidate_inventory = None
    current_artifacts = {
        "producer_script": _artifact_if_file(Path(__file__)),
        "publication_helper": _artifact_if_file(Path(publication_module.__file__)),
        "checkpoint": _authenticated_input_artifact_if_file(
            checkpoint_path,
            consumption="torch_load_from_same_open_file_description",
        ),
        "checkpoint_params": (
            _authenticated_input_artifact_if_file(
                checkpoint_params_path,
                consumption="json_decode_from_exact_authenticated_bytes",
            )
            if checkpoint_params_path is not None else None
        ),
        "audit_set": _artifact_if_file(args.audit_set),
        "matched_rows": _artifact_if_file(matched_path),
        "matched_rows_report": _artifact_if_file(matched_report_path),
        "preregistration": (
            _artifact_if_file(preregistration_path)
            if preregistration_path is not None else None
        ),
        "features_extension": _artifact_if_file(Path(features_extension.__file__)),
        "mcts_extension": _artifact_if_file(Path(mcts_extension.__file__)),
        "lc0_extension": _artifact_if_file(Path(lc0_extension.__file__)),
    }
    try:
        current_producer_sources = _producer_python_source_artifacts(
            producer_git_sha,
            require_tracked=False,
        )
    except (OSError, SystemExit):
        current_producer_sources = None
    changed_artifacts = sorted({
        *input_load_changes,
        *checkpoint_load_changes,
        *(f"producer_source_import:{name}" for name in producer_source_import_changes),
        *(f"native_import:{module}" for module in native_import_changes),
        *(
            ["params_candidate_inventory"]
            if final_params_candidate_inventory
            != provenance["params_candidate_inventory"] else []
        ),
        *(
            name for name, frozen in frozen_artifacts.items()
            if _artifact_identity(frozen) != _artifact_identity(current_artifacts[name])
        ),
    })
    if current_producer_sources != provenance["producer_sources"]:
        changed_artifacts.append("producer_sources")
    python_post_run_status = _preimport_python_surface_status(
        _PREIMPORT_PYTHON_SOURCES
    )
    provenance["python_preimport"]["post_run_check"] = python_post_run_status
    provenance["python_preimport"][
        "source_only_import"
    ] = _SOURCE_ONLY_IMPORT_GUARD.status()
    source_guard_status = provenance["python_preimport"]["source_only_import"]
    loaded_project_status = source_guard_status.get("loaded_project_modules")
    if not (
        source_guard_status.get("active") is True
        and source_guard_status.get("first_finder") is True
        and isinstance(loaded_project_status, dict)
        and loaded_project_status.get("passed") is True
        and loaded_project_status.get("unverified_modules") == []
    ):
        changed_artifacts.append("source_only_import_guard")
    if python_post_run_status.get("passed") is not True:
        changed_artifacts.append("python_preimport_surface")
    origin_snapshot = matched_origin_verification.get("snapshot_inventory")
    if isinstance(origin_snapshot, dict):
        try:
            final_origin_snapshot = _matched_snapshot_inventory(
                str(origin_snapshot.get("path", "")),
            )
        except (OSError, SystemExit):
            final_origin_snapshot = None
        if final_origin_snapshot != origin_snapshot:
            changed_artifacts.append("matched_rows_snapshot")
    try:
        current_syzygy_inventory = (
            _tablebase_inventory(
                str(args.syzygy_path),
                require_approved=not args.methodology_smoke,
                prior_content_verification=(
                    provenance["syzygy"].get("content_verification")
                    if not args.methodology_smoke
                    and isinstance(provenance.get("syzygy"), dict)
                    else None
                ),
            )
            if args.syzygy_path else None
        )
    except OSError:
        current_syzygy_inventory = None
    if current_syzygy_inventory != provenance["syzygy"]:
        changed_artifacts.append("syzygy")
    final_git_sha, final_git_dirty = _git_state()
    if final_git_sha != producer_git_sha or final_git_dirty:
        changed_artifacts.append("producer_checkout")
    changed_artifacts = sorted(set(changed_artifacts))
    artifact_stability = {
        "passed": not changed_artifacts,
        "changed": changed_artifacts,
        "final_git_sha": final_git_sha,
        "final_git_dirty": final_git_dirty,
    }
    manifest = {
        **provenance,
        "decision_grade": bool(
            not args.methodology_smoke
            and not preregistration_failures
            and panel_selection["decision_grade_passed"] is True
            and deep_reference_evidence["passed"] is True
            and not excluded_positions
            and incomplete_exclusions == 0
            and completed_positions > 0
            and source_group_resolution_passed
            and reference_censoring["passed"]
            and artifact_stability["passed"]
        ),
        "complete": True,
        "row_count": n_rows,
        "position_count": completed_positions,
        "requested_position_count": len(positions),
        "requested_max_positions": int(args.max_positions),
        "excluded_position_count": len(excluded_positions),
        "excluded_positions": excluded_positions,
        "incomplete_exclusion_count": incomplete_exclusions,
        "source_game_group_count": source_game_group_count,
        "minimum_decision_grade_source_games": _MIN_DECISION_GRADE_SOURCE_GAMES,
        "source_group_resolution_passed": source_group_resolution_passed,
        "reference_censoring": reference_censoring,
        "chunk_count": int(args.max_chunks),
        "runtime_seconds": time.perf_counter() - started,
        "elapsed_measurement": {
            "kind": "callback_instrumented_wall_time",
            "usable_for_controller_or_cost_analysis": False,
        },
        "output": completed_output_artifact,
        "requested_search": requested_search,
        "realized_search": realized_search,
        "requested_model_search_contract": expected_model_search_contract,
        "realized_model_search_contract": realized_model_search_contract,
        "requested_evaluator": expected_evaluator,
        "realized_evaluator": realized_evaluator,
        "runtime": runtime,
        "compile": compile_contract,
        "search_warmup": search_warmup,
        "artifact_stability": artifact_stability,
        "realized_tablebase": realized_tablebase,
        "features_extension": {
            **loaded_features_artifact,
            "build_attestation": native_build_attestations[features_module],
            "freshness_check": {
                "modules": native_modules,
                "minimum_gcc_major": 15,
                "production_recipe_required": True,
                "passed": not extension_issues,
                "issues": extension_issues,
            },
        },
        "mcts_extension": {
            **loaded_mcts_artifact,
            "build_attestation": native_build_attestations[mcts_module],
            "abi_version": int(getattr(mcts_extension, "ABI_VERSION", 0)),
            "required_abi_version": int(_REQUIRED_MCTS_ABI),
            "gss_halving_rev": loaded_halving_rev,
            "freshness_check": {
                "modules": native_modules,
                "minimum_gcc_major": 15,
                "production_recipe_required": True,
                "passed": not extension_issues,
                "issues": extension_issues,
            },
        },
        "lc0_extension": {
            **loaded_lc0_artifact,
            "build_attestation": native_build_attestations[lc0_module],
            "cboard_encode_full": bool(hasattr(CBoard.from_board(chess.Board()), "encode_full")),
            "freshness_check": {
                "modules": native_modules,
                "minimum_gcc_major": 15,
                "production_recipe_required": True,
                "passed": not extension_issues,
                "issues": extension_issues,
            },
        },
    }
    _publish_collected_evidence_pair(
        tmp_path,
        args.out,
        pending_meta_path,
        meta_path,
        manifest,
    )
    for output_lock in reversed(output_locks):
        output_lock.close()

    print(f"[traj] wrote {n_rows} rows -> {args.out}")
    print(f"[traj] provenance -> {meta_path}")


def _preserve_post_collection_failure(exc: BaseException) -> None:
    """Best-effort diagnostic sidecar for a fully collected unpublished bank."""
    state = _ACTIVE_PENDING_EVIDENCE
    if not state or state.get("collection_complete") is not True:
        return
    pending_output = state.get("pending_output")
    output = state.get("output")
    manifest_path = state.get("manifest")
    pending_manifest = state.get("pending_manifest")
    output_artifact = state.get("output_artifact")
    retained_output_fd = state.get("retained_output_fd")
    retained_output_parent_fd = state.get("retained_output_parent_fd")
    if (
        not isinstance(pending_output, Path)
        or not isinstance(output, Path)
        or not isinstance(manifest_path, Path)
        or not isinstance(pending_manifest, Path)
        or not isinstance(output_artifact, dict)
        or not isinstance(retained_output_fd, int)
        or retained_output_fd < 0
        or not isinstance(retained_output_parent_fd, int)
        or retained_output_parent_fd < 0
    ):
        return
    provenance = state.get("provenance")
    excluded_positions = state.get("excluded_positions")
    censoring_details = state.get("reference_censoring_details")
    if (
        not isinstance(provenance, dict)
        or not isinstance(excluded_positions, list)
        or not isinstance(censoring_details, list)
    ):
        return
    try:
        if _entry_name_exists(
            pending_manifest, parent_fd=retained_output_parent_fd,
        ) and isinstance(state.get("publication_manifest"), dict):
            return
        if not _entry_name_exists(
            pending_output, parent_fd=retained_output_parent_fd,
        ):
            return
        _write_invalid_recovery_diagnostic(
            pending_output,
            output,
            manifest_path,
            {
                **provenance,
                "decision_grade": False,
                "complete": False,
                "trajectory_collection_complete": True,
                "failure_stage": "post_collection_finalization",
                "finalization_error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "raw_observations_preserved": True,
                "row_count": state.get("row_count"),
                "position_count": state.get("position_count"),
                "requested_position_count": state.get("requested_position_count"),
                "requested_max_positions": state.get("requested_max_positions"),
                "excluded_position_count": len(excluded_positions),
                "excluded_positions": excluded_positions,
                "reference_censoring": _reference_censoring_summary(censoring_details),
            },
            file_fd=retained_output_fd,
            parent_fd=retained_output_parent_fd,
            expected_artifact=output_artifact,
        )
    except BaseException:
        publication_module._mark_manifest_recovery_invalid(
            manifest_path, parent_fd=retained_output_parent_fd,
        )


def main() -> None:
    """Run collection while retaining complete pending evidence on late failures."""
    global _ACTIVE_PENDING_EVIDENCE
    _ACTIVE_PENDING_EVIDENCE = None
    try:
        _main()
    except BaseException as exc:
        _preserve_post_collection_failure(exc)
        raise
    finally:
        state = _ACTIVE_PENDING_EVIDENCE
        if isinstance(state, dict):
            for name in ("retained_output_fd", "retained_output_parent_fd"):
                descriptor = state.get(name)
                if isinstance(descriptor, int) and descriptor >= 0:
                    os.close(descriptor)
                    state[name] = -1
        locks = state.get("output_locks") if isinstance(state, dict) else None
        if isinstance(locks, tuple):
            for output_lock in reversed(locks):
                output_lock.close()
        _ACTIVE_PENDING_EVIDENCE = None


if __name__ == "__main__":
    main()
