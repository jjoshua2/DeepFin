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

import numpy as np

_SCHEMA = "deepfin.chunk_trajectory.v2"
_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
_LEGACY_SMOKE_VISIT_GAP = 0.25
_LEGACY_SMOKE_STABLE_CHUNKS = 2
_ACTIVE_PARAMETER_KEYS = {
    "walker_puct": {
        "c_puct", "cpuct_factor", "cpuct_base", "fpu_reduction",
        "vloss_weight", "minibatch_size",
    },
    "gumbel": {
        "c_scale", "c_visit", "c_visit_root", "c_scale_root",
        "q_visit_exp_root", "topk", "vloss_weight", "minibatch_size",
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


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


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
    for name in ("producer_script", "checkpoint", "audit_set", "matched_rows"):
        artifact = manifest.get(name)
        artifact_sha = artifact.get("sha256") if isinstance(artifact, dict) else None
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("path"), str)
            or not isinstance(artifact.get("size"), int)
            or int(artifact.get("size", -1)) <= 0
            or not isinstance(artifact.get("mtime_ns"), int)
            or not isinstance(artifact_sha, str)
            or len(artifact_sha) != 64
            or any(char not in "0123456789abcdef" for char in artifact_sha.lower())
        ):
            failures.append(f"{name} artifact provenance is incomplete")
    if manifest.get("game_group_kind") != "source_dir:shard:game_id":
        failures.append(f"game_group_kind={manifest.get('game_group_kind')!r}")
    if manifest.get("root_position_history") != "fen_only_from_audit_fen":
        failures.append(f"root_position_history={manifest.get('root_position_history')!r}")
    requested = manifest.get("requested_search")
    realized = manifest.get("realized_search")
    if not isinstance(requested, dict) or not isinstance(realized, dict):
        failures.append("requested/realized search provenance is missing")
    else:
        active_path = requested.get("active_path")
        active = requested.get("active_parameters")
        expected_stack = (
            "BatchCoalescingDispatcher(ThreadSafeGPUDispatcher(DirectGPUEvaluator))"
            if active_path == "walker_puct"
            else "ThreadSafeGPUDispatcher(DirectGPUEvaluator)"
        )
        expected_keys = _ACTIVE_PARAMETER_KEYS.get(str(active_path))
        mismatch = (
            active_path != realized.get("concurrency_mode")
            or requested.get("walkers") != realized.get("concurrency_workers")
            or requested.get("chunk_sims") != realized.get("chunk_sims")
            or not isinstance(active, dict)
            or expected_keys is None
            or set(active) != expected_keys
            or requested.get("evaluator_stack") != expected_stack
            or not isinstance(requested.get("max_batch"), int)
            or int(requested.get("max_batch", 0)) <= 0
            or not isinstance(requested.get("leaf_gather"), int)
            or int(requested.get("leaf_gather", 0)) <= 0
            or not isinstance(requested.get("compact_bf16"), bool)
        )
        if isinstance(active, dict):
            mismatch = mismatch or any(realized.get(name) != value for name, value in active.items())
        if mismatch:
            failures.append("requested search does not match realized active parameters")
    row_count = manifest.get("row_count")
    chunk_count = manifest.get("chunk_count")
    position_count = manifest.get("position_count")
    requested_positions = manifest.get("requested_position_count")
    excluded_positions = manifest.get("excluded_position_count")
    counts_are_ints = all(isinstance(value, int) for value in (
        row_count, chunk_count, position_count,
        requested_positions, excluded_positions,
    ))
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
        ):
            failures.append("row/position/chunk accounting is inconsistent")
    decision_grade = manifest.get("decision_grade") is True and not failures
    if (failures or not decision_grade) and not methodology_smoke:
        detail = ", ".join(failures) if failures else "manifest is non-decision-grade"
        raise ValueError(f"trajectory provenance is not decision-grade: {detail}")
    return manifest, decision_grade


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
        rows.append(row)
    if not rows:
        raise ValueError("trajectory bank is empty")
    expected_rows = manifest.get("row_count")
    if not methodology_smoke and expected_rows != len(rows):
        raise ValueError(f"manifest says {expected_rows} rows, found {len(rows)}")

    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[str(row["key"])].append(row)
    transitions: list[Transition] = []
    has_score_regret = all("regret_score" in row for row in rows)
    if not has_score_regret and not methodology_smoke:
        raise ValueError("expected-score regret is missing")
    metric = "regret_score" if has_score_regret else "regret_cp"
    for key, trajectory in sorted(by_key.items()):
        trajectory.sort(key=lambda row: int(row["chunk"]))
        chunks = [int(row["chunk"]) for row in trajectory]
        if chunks != list(range(1, len(trajectory) + 1)) or len(trajectory) < 2:
            raise ValueError(f"{key}: chunks must be consecutive from 1 with at least two rows")
        last_best = -1
        stable = 0
        for index, (lower, upper) in enumerate(pairwise(trajectory)):
            observed_gap = _emitted_visit_gap(lower)
            if observed_gap is None:
                if not methodology_smoke:
                    raise ValueError(f"{key}: emitted-action visit provenance is missing")
                observed_gap = _finite(lower.get("visit_gap"))
                if index == 0 or bool(lower.get("bestmove_flip")):
                    stable = 0
                else:
                    stable += 1
            elif not math.isclose(
                observed_gap, _finite(lower.get("visit_gap")), rel_tol=1e-10, abs_tol=1e-12,
            ):
                raise ValueError(f"{key}: visit_gap is not the emitted action's gap")
            else:
                actions = lower["root_actions"]
                last_best, stable = _update_stability(
                    last_best,
                    stable,
                    emitted_action=int(lower["emitted_action"]),
                    visit_gap=observed_gap,
                    action_count=len(actions),
                )
            recorded_stable = lower.get("stable_chunks")
            if recorded_stable is not None and int(recorded_stable) != stable:
                raise ValueError(f"{key}: stable_chunks disagrees with emitted move history")
            game_id = lower.get("game_id")
            if game_id is None or int(game_id) < 0:
                if not methodology_smoke:
                    raise ValueError(f"{key}: source game_id is missing")
                game_id = int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)
            group_id = lower.get("group_id")
            source_dir = lower.get("source_dir")
            shard = lower.get("shard")
            expected_group = (
                "\0".join((str(source_dir), str(shard), str(game_id)))
                if source_dir and shard and game_id is not None else None
            )
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
            lo_regret = _finite(lower.get(metric))
            hi_regret = _finite(upper.get(metric))
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
                    complexity_continue = not (
                        stable >= _LEGACY_SMOKE_STABLE_CHUNKS
                        and observed_gap >= _LEGACY_SMOKE_VISIT_GAP
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


def cluster_bootstrap_delta(
    transitions: Sequence[Transition], m0: np.ndarray, m1: np.ndarray,
    *, allocation_fraction: float, samples: int, seed: int,
    min_oracle_headroom: float = 0.0,
) -> dict[str, float | int | None]:
    """Game-cluster bootstrap of M1-minus-M0 oracle capture."""
    groups = sorted({row.group_id for row in transitions})
    by_group = {
        group: np.flatnonzero([row.group_id == group for row in transitions])
        for group in groups
    }
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([by_group[str(group)] for group in drawn])
        rows = [transitions[index] for index in indices]
        delta = _weighted_capture_delta(
            rows, m0[indices], m1[indices], allocation_fraction,
            min_oracle_headroom,
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
        transitions, m0, m1, allocation_fraction=allocation_fraction,
        samples=bootstrap_samples, seed=seed,
        min_oracle_headroom=min_oracle_headroom,
    )
    advance = bool(
        capture_gain is not None
        and capture_gain >= min_capture_gain
        and every_horizon_eligible and every_horizon_positive
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
            "game_cluster_bootstrap_lower_95_above_zero": True,
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
        "bootstrap_M1_minus_M0": bootstrap,
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
    transitions, info = load_transitions(
        args.input_path, meta_path=args.meta, methodology_smoke=args.methodology_smoke,
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
