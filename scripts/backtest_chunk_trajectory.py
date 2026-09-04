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
import threading
import time
from collections import Counter, deque
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import IO, Any

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
from scripts.chunk_trajectory_publication import (
    CHUNK_TRAJECTORY_SCHEMA,
    _acquire_output_lock as _acquire_output_lock,
    _acquire_output_locks,
    _git_ignored_or_outside as _git_ignored_or_outside,
    _output_lock_path as _output_lock_path,
    _pending_manifest_path,
    _pending_output_path,
    _prepared_output_artifact,
    _publish_evidence_pair,
    _publish_output as _publish_output,
    _require_new_output_pair,
    _require_safe_output_paths,
    _write_json_atomic,
    _write_json_staged as _write_json_staged,
)
from scripts.repo_output_guard import repo_controlled_output, reserved_output_path

from chess_anti_engine.eval.audit import (
    AuditPosition,
    legal_full_indices,
    load_audit_set,
    move_regrets,
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
    _MIN_DECISION_GRADE_SOURCE_GAMES,
    _PRODUCTION_WALKERS,
    _canonical_cuda_device_string,
    _complexity_continue,
    _preregistration_payload,
    _preregistered_design_failures,
    _reference_censoring_summary,
    _score,
    _trajectory_reference_censoring,
    _update_stability,
)
from scripts.check_c_extensions_fresh import (
    check_extensions,
    extension_spec,
    native_build_attestation,
)

_SCHEMA = CHUNK_TRAJECTORY_SCHEMA
_PRODUCTION_WDL_FILES = 875
_PRODUCTION_DTZ_FILES = 510
_PRODUCTION_TB_COMPONENTS = ((510, 145), (365, 365))
_PRODUCTION_GSS_HALVING_REV = 3
_MIN_DECISION_GRADE_CHUNKS = 4
_PANEL_SELECTION_STRATEGY = "joint_audit_source_phase_piece_round_robin_v1"
_PANEL_REQUIRED_SOURCES = (0, 1)
_PANEL_SOURCE_BALANCE_MAX_DIFFERENCE = 1
_REQUIRED_PRODUCER_SOURCE_MODULES = {
    "producer_script",
    "scripts.chunk_trajectory_publication",
    "scripts.analyze_chunk_controller",
    "scripts.repo_output_guard",
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


def _producer_python_source_artifacts(
    producer_git_sha: str, *, require_tracked: bool,
) -> dict[str, dict[str, Any]]:
    """Bind every loaded project Python module to the producer revision."""
    repo_root = Path(__file__).resolve().parents[1]
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
        artifacts[name] = {
            **artifact,
            "repo_relative_path": relative_path,
            "matches_producer_git_revision": matches_commit,
        }
        if not matches_commit:
            unbound.append(name)
    if require_tracked and unbound:
        raise SystemExit(
            "decision-grade producer Python sources are not tracked by the producer "
            f"revision: {', '.join(unbound)}"
        )
    return artifacts


def _artifact_identity(artifact: Any) -> dict[str, Any] | None:
    if not isinstance(artifact, dict):
        return None
    return {
        name: artifact.get(name)
        for name in ("path", "size", "mtime_ns", "device", "inode", "sha256")
    }


def _loaded_native_build_attestation(
    module: Any, module_name: str, producer_git_sha: str,
) -> dict[str, Any]:
    """Bind one loaded binary's embedded stamp to exact producer-revision inputs."""
    dependency_bytes: dict[str, bytes] = {}
    current_inputs_match_revision = True
    for relative_path in extension_spec(module_name).dependencies:
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


def _tablebase_inventory(path_value: str) -> dict[str, Any]:
    """Cheap, durable identity for the large production tablebase directories."""
    directories: list[dict[str, Any]] = []
    for raw_directory in path_value.split(SEPARATOR):
        directory = Path(raw_directory.strip()).expanduser().resolve()
        digest = hashlib.sha256()
        counts = {"rtbw": 0, "rtbz": 0}
        total_bytes = 0
        files = sorted(
            entry for entry in directory.iterdir()
            if entry.is_file() and entry.suffix in (".rtbw", ".rtbz")
        )
        for entry in files:
            stat = entry.stat()
            counts[entry.suffix[1:]] += 1
            total_bytes += int(stat.st_size)
            digest.update(entry.name.encode())
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode())
            digest.update(b"\0")
            digest.update(str(stat.st_mtime_ns).encode())
            digest.update(b"\n")
        directories.append({
            "path": str(directory),
            "rtbw_count": counts["rtbw"],
            "rtbz_count": counts["rtbz"],
            "total_bytes": total_bytes,
            "inventory_sha256": digest.hexdigest(),
        })
    return {
        "path": path_value,
        "directories": directories,
        "rtbw_count": sum(int(row["rtbw_count"]) for row in directories),
        "rtbz_count": sum(int(row["rtbz_count"]) for row in directories),
    }


def _checkpoint_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    candidate = resolved / "trainer.pt" if resolved.is_dir() else resolved
    if not candidate.is_file():
        raise SystemExit(f"checkpoint has no trainer.pt: {resolved}")
    return candidate


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
        "deep_reference_best_cp": float(position.best_cp),
        "deep_reference_move_cp": {
            str(uci): float(cp) for uci, cp in position.move_cp.items()
        },
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
    repo_root = Path(__file__).resolve().parents[1]
    if reserved_output_path(output_path):
        raise SystemExit(
            "--write-preregistration must not use the output lock/staging namespace"
        )
    output = output_path.expanduser().resolve()
    try:
        output.relative_to(repo_root)
    except ValueError as exc:
        raise SystemExit("--write-preregistration must be inside the repository") from exc
    if output.exists():
        raise SystemExit(f"refusing to overwrite preregistration plan {output}")
    if repo_controlled_output(output, repo_root):
        raise SystemExit("--write-preregistration must target a new untracked file")
    if output in {path.expanduser().resolve() for path in protected_files}:
        raise SystemExit("--write-preregistration aliases a consumed input artifact")
    for directory in protected_directories:
        try:
            output.relative_to(directory.expanduser().resolve())
        except ValueError:
            continue
        raise SystemExit("--write-preregistration must not be inside a Syzygy directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output, payload)
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
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --recover-publication is used")
    matched_path = args.matched_rows or default_matched_rows_path(args.audit_set)
    checkpoint_path = _checkpoint_file(Path(args.checkpoint))
    producer_git_sha, producer_git_dirty = _git_state()
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
    initial_input_artifacts = {
        "producer_script": initial_producer_sources["producer_script"],
        "publication_helper": initial_producer_sources[
            "scripts.chunk_trajectory_publication"
        ],
        "checkpoint": _artifact(checkpoint_path, require_file=True),
        "audit_set": _artifact(args.audit_set, require_file=True),
        "matched_rows": (
            _artifact(matched_path, require_file=True) if matched_path.is_file() else None
        ),
        "preregistration": preregistration_artifact,
    }
    syzygy_directories = (
        [Path(part.strip()) for part in str(args.syzygy_path).split(SEPARATOR)]
        if args.syzygy_path else []
    )
    _require_safe_output_paths(
        args.out,
        meta_path,
        protected_files=[
            args.audit_set, matched_path, checkpoint_path, Path(__file__),
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
    if args.syzygy_path or not args.methodology_smoke:
        try:
            require_tablebases(args.syzygy_path, what="trajectory --syzygy-path")
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    syzygy_inventory = (
        _tablebase_inventory(str(args.syzygy_path)) if args.syzygy_path else None
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
        )
    ):
        raise SystemExit(
            "decision-grade trajectory banks require the complete production "
            f"Syzygy pair ({_PRODUCTION_WDL_FILES} WDL and "
            f"{_PRODUCTION_DTZ_FILES} DTZ files)"
        )
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
        load_audit_set(args.audit_set), int(args.max_positions),
    )
    _require_decision_grade_panel_selection(
        panel_selection, methodology_smoke=bool(args.methodology_smoke),
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
    post_load_artifacts = {
        "audit_set": _artifact_if_file(args.audit_set),
        "matched_rows": _artifact_if_file(matched_path),
    }
    input_load_changes = sorted(
        name for name in ("audit_set", "matched_rows")
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
    for pos in positions:
        gid = matched.game_id(pos.key) if matched is not None and pos.key in matched else None
        source_dir = matched.snapshot if matched is not None and pos.key in matched else None
        source_shard = (
            matched.source_shard(pos.key)
            if matched is not None and pos.key in matched else None
        )
        source_cluster_unique = bool(
            matched is not None
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

    provenance = {
        "schema": _SCHEMA,
        "decision_grade": not args.methodology_smoke,
        "analysis_scope": "fixed_node_horizons_only",
        "clock_conditioning_available": False,
        "root_position_history": "fen_only_from_audit_fen",
        "root_tree_state": "fresh_per_position_no_cross_move_reuse",
        "game_group_kind": "source_dir:game_id",
        "panel_selection": panel_selection,
        "complexity_predicate": {
            "kind": "clock_free_visit_gap_and_stability",
            "minimum_stable_chunks": int(_ABORT_MIN_STABLE_CHUNKS),
            "minimum_visit_gap": float(_ABORT_VISIT_GAP_MARGIN),
            "single_legal_move_is_decided": True,
        },
        "producer_git_sha": producer_git_sha,
        "producer_git_dirty": producer_git_dirty,
        "producer_script": initial_input_artifacts["producer_script"],
        "publication_helper": initial_input_artifacts["publication_helper"],
        "checkpoint": initial_input_artifacts["checkpoint"],
        "audit_set": initial_input_artifacts["audit_set"],
        "matched_rows": initial_input_artifacts["matched_rows"],
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
        _find_params_json,
        load_model_from_checkpoint,
    )
    from chess_anti_engine.uci.search import SearchWorker
    from chess_anti_engine.uci.time_manager import Deadline
    from chess_anti_engine.worker import _configure_shared_compile_cache

    producer_sources = _producer_python_source_artifacts(
        producer_git_sha,
        require_tracked=not args.methodology_smoke,
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
    checkpoint_params_path = _find_params_json(checkpoint_path)
    provenance["checkpoint_params"] = (
        _artifact(checkpoint_params_path, require_file=True)
        if checkpoint_params_path is not None else None
    )
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
            checkpoint_path,
            Path(__file__),
            Path(publication_module.__file__),
            Path(features_extension.__file__),
            Path(mcts_extension.__file__),
            Path(lc0_extension.__file__),
            *([checkpoint_params_path] if checkpoint_params_path is not None else []),
            *([preregistration_path] if preregistration_path is not None else []),
        ],
        protected_directories=syzygy_directories,
    )
    if not args.methodology_smoke and loaded_halving_rev != _PRODUCTION_GSS_HALVING_REV:
        raise SystemExit(
            "decision-grade trajectory banks require MCTS "
            f"GSS_HALVING_REV={_PRODUCTION_GSS_HALVING_REV}, loaded "
            f"{loaded_halving_rev} from {mcts_extension.__file__}"
        )

    model = load_model_from_checkpoint(
        checkpoint_path,
        device=args.device,
        require_complete=not args.methodology_smoke,
    )
    model.eval()
    post_checkpoint_params_path = _find_params_json(checkpoint_path)
    post_model_load_artifacts = {
        "checkpoint": _artifact_if_file(checkpoint_path),
        "checkpoint_params": (
            _artifact_if_file(post_checkpoint_params_path)
            if post_checkpoint_params_path is not None else None
        ),
    }
    checkpoint_load_changes = sorted(
        name for name, before in (
            ("checkpoint", provenance["checkpoint"]),
            ("checkpoint_params", provenance["checkpoint_params"]),
        )
        if _artifact_identity(before)
        != _artifact_identity(post_model_load_artifacts[name])
    )
    if checkpoint_load_changes and not args.methodology_smoke:
        raise SystemExit(
            "decision-grade checkpoint inputs changed while being loaded: "
            + ", ".join(checkpoint_load_changes)
        )
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
        written = _write_preregistration_plan(
            args.write_preregistration,
            _preregistration_payload(preregistration_manifest),
            protected_files=[
                args.audit_set,
                matched_path,
                checkpoint_path,
                Path(__file__),
                Path(publication_module.__file__),
                Path(features_extension.__file__),
                Path(mcts_extension.__file__),
                Path(lc0_extension.__file__),
                *([checkpoint_params_path] if checkpoint_params_path is not None else []),
            ],
            protected_directories=syzygy_directories,
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
    _ACTIVE_PENDING_EVIDENCE = {
        "collection_complete": False,
        "pending_output": tmp_path,
        "output": args.out,
        "pending_manifest": pending_meta_path,
        "output_locks": output_locks,
        "provenance": provenance,
        "requested_position_count": len(positions),
        "requested_max_positions": int(args.max_positions),
        "excluded_positions": excluded_positions,
        "reference_censoring_details": reference_censoring_details,
    }
    try:
        with tmp_path.open("x") as fh:
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
                        "deep_reference_best_cp": s["deep_reference_best_cp"],
                        "deep_reference_move_cp": {
                            str(uci): float(cp) for uci, cp in pos.move_cp.items()
                        },
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
        _ACTIVE_PENDING_EVIDENCE.update({
            "collection_complete": True,
            "row_count": n_rows,
            "position_count": completed_positions,
        })
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
            _write_json_staged(
                pending_meta_path,
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
                    "output": _prepared_output_artifact(tmp_path, args.out),
                },
            )
        finally:
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
        "preregistration": provenance["preregistration"],
        "features_extension": loaded_features_artifact,
        "mcts_extension": loaded_mcts_artifact,
        "lc0_extension": loaded_lc0_artifact,
    }
    current_params_path = _find_params_json(checkpoint_path)
    current_artifacts = {
        "producer_script": _artifact_if_file(Path(__file__)),
        "publication_helper": _artifact_if_file(Path(publication_module.__file__)),
        "checkpoint": _artifact_if_file(checkpoint_path),
        "checkpoint_params": (
            _artifact_if_file(current_params_path)
            if current_params_path is not None else None
        ),
        "audit_set": _artifact_if_file(args.audit_set),
        "matched_rows": _artifact_if_file(matched_path),
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
            name for name, frozen in frozen_artifacts.items()
            if _artifact_identity(frozen) != _artifact_identity(current_artifacts[name])
        ),
    })
    if current_producer_sources != provenance["producer_sources"]:
        changed_artifacts.append("producer_sources")
    try:
        current_syzygy_inventory = (
            _tablebase_inventory(str(args.syzygy_path)) if args.syzygy_path else None
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
        "output": _prepared_output_artifact(tmp_path, args.out),
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
    _publish_evidence_pair(
        tmp_path, args.out, pending_meta_path, meta_path, manifest,
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
    pending_manifest = state.get("pending_manifest")
    if (
        not isinstance(pending_output, Path)
        or not isinstance(output, Path)
        or not isinstance(pending_manifest, Path)
        or not pending_output.is_file()
        or pending_manifest.exists()
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
        _write_json_staged(
            pending_manifest,
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
                "output": _prepared_output_artifact(pending_output, output),
            },
        )
    except BaseException:
        # Diagnostics must never replace the producer's original nonzero failure.
        pass


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
        locks = state.get("output_locks") if isinstance(state, dict) else None
        if isinstance(locks, tuple):
            for output_lock in reversed(locks):
                output_lock.close()
        _ACTIVE_PENDING_EVIDENCE = None


if __name__ == "__main__":
    main()
