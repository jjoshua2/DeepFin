#!/usr/bin/env python3
"""Analyze paired varying-horizon search trajectories.

The primary comparison is M_budget versus the age-controlled M_age.  All
predictions are group-held-out.  Held-out decisions are applied independently:
continue iff predicted finite-horizon value-to-go is positive.  There is no
bank-wide quota, ranking, or stopped-position re-entry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

SCHEMA = "deepfin.varying_budget_trajectory.v1"
ANALYSIS_SCHEMA = "deepfin.varying_budget_analysis.v1"
HORIZONS = (4, 6, 8)
PRICES = (0.0, 0.000025, 0.00005, 0.0001)
PRIMARY_PRICE = 0.00005
CHUNK_SIMS = 2048
MAX_CHUNKS = 8
WALKERS = 2
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
ModelName = Literal["M_state", "M_age", "M_budget"]
MODELS: tuple[ModelName, ...] = ("M_state", "M_age", "M_budget")


@dataclass(frozen=True)
class Snapshot:
    chunk: int
    nodes: int
    regret_score: float
    visit_gap: float
    visit_entropy: float
    q_gap: float | None
    root_q: float
    bestmove_flip: bool
    stable_chunks: int
    q_drift: float | None
    visit_churn: float | None
    piece_count: int
    legal_move_count: int
    phase: int
    source: int
    complexity_continue: bool


@dataclass(frozen=True)
class Trajectory:
    key: str
    group_id: str
    group_kind: str
    snapshots: tuple[Snapshot, ...]


@dataclass(frozen=True)
class Example:
    key: str
    group_id: str
    horizon: int
    chunk: int
    snapshot: Snapshot
    target: float


@dataclass(frozen=True)
class Rollout:
    key: str
    group_id: str
    horizon: int
    phase: int
    source: int
    stop_chunk: int
    regret: float
    gain: float
    net_utility: float


@dataclass(frozen=True)
class Ridge:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        return np.column_stack([np.ones(len(z)), z]) @ self.coef


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _optional(value: Any, name: str) -> float | None:
    return None if value is None else _finite(value, name)


def _score(cp: float) -> float:
    exponent = float(cp) * math.log(10.0) / 300.0
    if exponent >= 0.0:
        return 1.0 / (1.0 + math.exp(-exponent))
    e = math.exp(exponent)
    return e / (1.0 + e)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stability(last: int, count: int, action: int, gap: float, n: int) -> tuple[int, int]:
    if gap <= 0.0 and n != 1:
        return last, 0
    return (last, count + 1) if action == last else (action, 0)


def _complexity_continue(stable: int, gap: float, n: int) -> bool:
    return not (stable >= 2 and (n == 1 or gap >= 0.25))


def load_trajectories(path: Path) -> list[Trajectory]:
    """Load complete fixed-node trajectories and recompute core derived fields."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        row = json.loads(text)
        if not isinstance(row, dict) or row.get("schema") != SCHEMA:
            raise ValueError(f"line {line_number}: wrong trajectory schema")
        key = row.get("key")
        if not isinstance(key, str) or not key:
            raise ValueError(f"line {line_number}: missing key")
        groups[key].append(row)
    if not groups:
        raise ValueError("trajectory bank is empty")

    trajectories: list[Trajectory] = []
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: int(row.get("chunk", -1)))
        if [row.get("chunk") for row in rows] != list(range(1, MAX_CHUNKS + 1)):
            raise ValueError(f"{key}: expected exactly {MAX_CHUNKS} consecutive chunks")
        first = rows[0]
        immutable = ("group_id", "group_kind", "fen", "phase", "source", "reference_best_cp")
        snapshots: list[Snapshot] = []
        last_action = -1
        stable = 0
        previous_q: float | None = None
        previous_shares: dict[int, float] | None = None
        for row in rows:
            if any(row.get(name) != first.get(name) for name in immutable):
                raise ValueError(f"{key}: trajectory identity/reference changes between chunks")
            chunk = _positive_int(row.get("chunk"), f"{key}: chunk")
            nodes = _positive_int(row.get("nodes"), f"{key}: nodes")
            if nodes != chunk * CHUNK_SIMS:
                raise ValueError(f"{key}: fixed-node chunk mismatch")
            phase = row.get("phase")
            source = row.get("source")
            if phase not in (0, 1, 2) or source not in (0, 1):
                raise ValueError(f"{key}: invalid phase/source")
            group_id = row.get("group_id")
            group_kind = row.get("group_kind")
            if not isinstance(group_id, str) or not group_id or group_kind not in ("source_game", "position"):
                raise ValueError(f"{key}: invalid group provenance")

            actions = row.get("root_actions")
            visits = row.get("root_visits")
            child_q = row.get("root_child_q")
            action = row.get("emitted_action")
            if (
                not isinstance(actions, list) or not actions
                or any(isinstance(a, bool) or not isinstance(a, int) for a in actions)
                or len(set(actions)) != len(actions)
                or not isinstance(visits, list) or len(visits) != len(actions)
                or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in visits)
                or not isinstance(child_q, list) or len(child_q) != len(actions)
                or isinstance(action, bool) or not isinstance(action, int) or action not in actions
            ):
                raise ValueError(f"{key}: malformed root observations")
            q_values = [_finite(value, f"{key}: child Q") for value in child_q]
            legal_count = _positive_int(row.get("legal_move_count"), f"{key}: legal count")
            if len(actions) != legal_count:
                raise ValueError(f"{key}: incomplete root action support")
            total = sum(visits)
            if total <= 0:
                if legal_count != 1:
                    raise ValueError(f"{key}: only a forced root may have zero visits")
                shares = [1.0]
            else:
                shares = [visit / total for visit in visits]
            action_index = actions.index(action)
            alternatives = shares[:action_index] + shares[action_index + 1:]
            gap = shares[action_index] - max(alternatives, default=0.0)
            entropy = -sum(value * math.log(value) for value in shares if value > 0.0)
            if not math.isclose(_finite(row.get("visit_gap"), f"{key}: visit gap"), gap, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"{key}: visit_gap disagrees with visits")
            if not math.isclose(_finite(row.get("visit_entropy"), f"{key}: entropy"), entropy, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"{key}: visit_entropy disagrees with visits")
            other_q = [q for index, (visit, q) in enumerate(zip(visits, q_values, strict=True)) if index != action_index and visit > 0]
            expected_q_gap = q_values[action_index] - max(other_q) if visits[action_index] > 0 and other_q else None
            observed_q_gap = _optional(row.get("q_gap"), f"{key}: q_gap")
            if expected_q_gap is None:
                if observed_q_gap is not None:
                    raise ValueError(f"{key}: q_gap must be null without two visited moves")
            elif observed_q_gap is None or not math.isclose(observed_q_gap, expected_q_gap, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"{key}: q_gap disagrees with child Q")

            best_cp = _finite(row.get("reference_best_cp"), f"{key}: best cp")
            chosen_cp = _finite(row.get("chosen_reference_cp"), f"{key}: chosen cp")
            regret = _finite(row.get("regret_score"), f"{key}: regret")
            expected_regret = _score(best_cp) - _score(chosen_cp)
            if not math.isclose(regret, expected_regret, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError(f"{key}: expected-score regret disagrees with reference CP")

            flip = bool(snapshots and action != last_action)
            if row.get("bestmove_flip") is not flip:
                raise ValueError(f"{key}: bestmove_flip disagrees with actions")
            last_action, stable = _stability(last_action, stable, action, gap, len(actions))
            if row.get("stable_chunks") != stable:
                raise ValueError(f"{key}: stable_chunks disagrees with history")
            if row.get("complexity_continue") is not _complexity_continue(stable, gap, len(actions)):
                raise ValueError(f"{key}: complexity predicate disagrees with state")
            root_q = _finite(row.get("root_q"), f"{key}: root Q")
            q_drift = _optional(row.get("q_drift"), f"{key}: q drift")
            expected_drift = None if previous_q is None else abs(root_q - previous_q)
            if (expected_drift is None) != (q_drift is None) or (expected_drift is not None and not math.isclose(float(q_drift), expected_drift, rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError(f"{key}: q_drift disagrees with history")
            share_map = dict(zip(actions, shares, strict=True))
            churn = _optional(row.get("visit_churn"), f"{key}: visit churn")
            expected_churn = None if previous_shares is None else 0.5 * sum(abs(share_map.get(a, 0.0) - previous_shares.get(a, 0.0)) for a in set(share_map) | set(previous_shares))
            if (expected_churn is None) != (churn is None) or (expected_churn is not None and not math.isclose(float(churn), expected_churn, rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError(f"{key}: visit_churn disagrees with history")

            snapshots.append(Snapshot(
                chunk=chunk, nodes=nodes, regret_score=regret, visit_gap=gap,
                visit_entropy=entropy, q_gap=expected_q_gap, root_q=root_q,
                bestmove_flip=flip, stable_chunks=stable, q_drift=q_drift,
                visit_churn=churn,
                piece_count=_positive_int(row.get("piece_count"), f"{key}: pieces"),
                legal_move_count=legal_count, phase=int(phase), source=int(source),
                complexity_continue=bool(row["complexity_continue"]),
            ))
            previous_q = root_q
            previous_shares = share_map
        trajectories.append(Trajectory(key, str(first["group_id"]), str(first["group_kind"]), tuple(snapshots)))
    return trajectories


def load_manifest(bank: Path, meta: Path | None = None) -> dict[str, Any] | None:
    path = meta or Path(str(bank) + ".meta.json")
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    output = manifest.get("output") if isinstance(manifest, dict) else None
    if not isinstance(output, dict) or output.get("size") != bank.stat().st_size or output.get("sha256") != _sha256(bank):
        raise ValueError("collector manifest does not match trajectory bank")
    return manifest


def collection_status(rows: Sequence[Trajectory], manifest: dict[str, Any] | None) -> dict[str, Any]:
    reasons: list[str] = []
    if manifest is None:
        return {"passed": False, "reasons": ["manifest_missing"]}
    config = manifest.get("config") if isinstance(manifest, dict) else None
    model = manifest.get("model") if isinstance(manifest, dict) else None
    if manifest.get("complete") is not True or manifest.get("completed_positions") != len(rows):
        reasons.append("incomplete_collection")
    expected = {
        "schema": SCHEMA, "chunk_sims": CHUNK_SIMS, "max_chunks": MAX_CHUNKS,
        "walkers": WALKERS, "compile_mode": "max-autotune", "production_shape": True,
    }
    if not isinstance(config, dict):
        reasons.append("config_missing")
    else:
        reasons.extend(f"{name}_mismatch" for name, value in expected.items() if config.get(name) != value)
        if not str(config.get("device", "")).startswith("cuda"):
            reasons.append("device_not_cuda")
        git_sha = config.get("git_sha")
        if not isinstance(git_sha, str) or len(git_sha) != 40:
            reasons.append("git_sha_invalid")
    if not isinstance(model, dict) or model.get("realized_search_path") != "walker":
        reasons.append("walker_not_realized")
    return {"passed": not reasons, "reasons": sorted(set(reasons))}


def value_to_go(snapshots: Sequence[Snapshot], chunk: int, horizon: int, price: float) -> float:
    current = snapshots[chunk - 1].regret_score
    return max(current - snapshots[depth - 1].regret_score - price * (depth - chunk) for depth in range(chunk + 1, horizon + 1))


def build_examples(rows: Sequence[Trajectory], horizons: Sequence[int], price: float) -> list[Example]:
    if price < 0.0 or not math.isfinite(price):
        raise ValueError("compute price must be finite and non-negative")
    result: list[Example] = []
    for trajectory in rows:
        for horizon in horizons:
            if horizon > len(trajectory.snapshots):
                raise ValueError("horizon exceeds collected trajectory")
            result.extend(
                Example(
                    trajectory.key,
                    trajectory.group_id,
                    horizon,
                    chunk,
                    trajectory.snapshots[chunk - 1],
                    value_to_go(trajectory.snapshots, chunk, horizon, price),
                )
                for chunk in range(1, horizon)
            )
    return result


def feature_vector(example: Example, model: ModelName) -> list[float]:
    snap = example.snapshot
    base = [
        snap.visit_gap, snap.visit_entropy, 0.0 if snap.q_gap is None else snap.q_gap,
        float(snap.q_gap is None), snap.root_q, float(snap.bestmove_flip),
        float(snap.stable_chunks), 0.0 if snap.q_drift is None else snap.q_drift,
        float(snap.q_drift is None), 0.0 if snap.visit_churn is None else snap.visit_churn,
        float(snap.visit_churn is None), float(snap.piece_count),
        float(snap.legal_move_count), float(snap.phase == 1), float(snap.phase == 2),
    ]
    if model == "M_state":
        return base
    aged = [*base, math.log1p(snap.nodes)]
    if model == "M_age":
        return aged
    if model != "M_budget":
        raise ValueError(f"unknown model {model}")
    remaining = example.horizon - example.chunk
    fraction = remaining / example.horizon
    hard_nodes = (snap.nodes // snap.chunk) * example.horizon
    return [*aged, math.log1p(hard_nodes), float(remaining), fraction,
            snap.visit_gap * fraction, snap.visit_entropy * fraction,
            float(snap.bestmove_flip) * fraction,
            (0.0 if snap.q_drift is None else snap.q_drift) * fraction,
            (0.0 if snap.visit_churn is None else snap.visit_churn) * fraction]


def _design(rows: Sequence[Example], model: ModelName) -> np.ndarray:
    return np.asarray([feature_vector(row, model) for row in rows], dtype=np.float64)


def grouped_folds(groups: Sequence[str], n_folds: int) -> np.ndarray:
    if n_folds < 2 or len(set(groups)) < 2:
        raise ValueError("grouped validation needs at least two folds/groups")
    counts: dict[str, int] = defaultdict(int)
    for group in groups:
        counts[str(group)] += 1
    actual = min(n_folds, len(counts))
    loads = [0] * actual
    mapping: dict[str, int] = {}
    for group, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        fold = min(range(actual), key=lambda index: (loads[index], index))
        mapping[group] = fold
        loads[fold] += count
    return np.asarray([mapping[str(group)] for group in groups], dtype=np.int32)


def _fit(x: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - mean) / scale
    aug = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(aug.shape[1]) * alpha
    penalty[0, 0] = 0.0
    lhs = aug.T @ aug + penalty
    rhs = aug.T @ y
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return Ridge(mean, scale, coef)


def _alpha(rows: Sequence[Example], model: ModelName, n_folds: int) -> float:
    if len({row.group_id for row in rows}) < 3:
        return 1.0
    folds = grouped_folds([row.group_id for row in rows], n_folds)
    x = _design(rows, model)
    y = np.asarray([row.target for row in rows])
    losses = {alpha: [] for alpha in ALPHAS}
    for fold in sorted(set(folds.tolist())):
        train, valid = folds != fold, folds == fold
        for alpha in ALPHAS:
            prediction = _fit(x[train], y[train], alpha).predict(x[valid])
            losses[alpha].append(float(np.mean((prediction - y[valid]) ** 2)))
    return min(ALPHAS, key=lambda value: (float(np.mean(losses[value])), value))


def cross_fitted_predictions(rows: Sequence[Example], model: ModelName, n_folds: int) -> tuple[np.ndarray, list[dict[str, int | float]]]:
    folds = grouped_folds([row.group_id for row in rows], n_folds)
    prediction = np.full(len(rows), np.nan)
    diagnostics: list[dict[str, int | float]] = []
    for fold in sorted(set(folds.tolist())):
        train_index = np.flatnonzero(folds != fold)
        test_index = np.flatnonzero(folds == fold)
        train = [rows[int(index)] for index in train_index]
        test = [rows[int(index)] for index in test_index]
        alpha = _alpha(train, model, n_folds)
        fitted = _fit(_design(train, model), np.asarray([row.target for row in train]), alpha)
        prediction[test_index] = fitted.predict(_design(test, model))
        diagnostics.append({"fold": int(fold), "alpha": alpha, "train_groups": len({row.group_id for row in train}), "test_groups": len({row.group_id for row in test})})
    if not np.isfinite(prediction).all():
        raise AssertionError("cross-fitting left examples unpredicted")
    return prediction, diagnostics


def rollout_policy(rows: Sequence[Trajectory], horizons: Sequence[int], price: float, predictions: dict[tuple[str, int, int], float] | None = None, *, complexity: bool = False, oracle: bool = False, fixed_full: bool = False) -> list[Rollout]:
    if sum((predictions is not None, complexity, oracle, fixed_full)) != 1:
        raise ValueError("select exactly one rollout mode")
    result: list[Rollout] = []
    for trajectory in rows:
        for horizon in horizons:
            if fixed_full:
                stop = horizon
            elif oracle:
                start = trajectory.snapshots[0].regret_score
                stop = max(range(1, horizon + 1), key=lambda depth: (start - trajectory.snapshots[depth - 1].regret_score - price * (depth - 1), -depth))
            else:
                stop = 1
                for chunk in range(1, horizon):
                    if complexity:
                        go = trajectory.snapshots[chunk - 1].complexity_continue
                    else:
                        assert predictions is not None
                        go = predictions[(trajectory.key, horizon, chunk)] > 0.0
                    if not go:
                        break
                    stop = chunk + 1
            first, final = trajectory.snapshots[0], trajectory.snapshots[stop - 1]
            gain = first.regret_score - final.regret_score
            result.append(Rollout(trajectory.key, trajectory.group_id, horizon, first.phase, first.source, stop, final.regret_score, gain, gain - price * (stop - 1)))
    return result


def _summary(rows: Sequence[Rollout]) -> dict[str, float | int]:
    regret = np.asarray([row.regret for row in rows])
    stop = np.asarray([row.stop_chunk for row in rows], dtype=np.float64)
    horizon = np.asarray([row.horizon for row in rows], dtype=np.float64)
    return {"n": len(rows), "mean_regret": float(regret.mean()), "p95_regret": float(np.quantile(regret, 0.95)), "p99_regret": float(np.quantile(regret, 0.99)), "mean_stop_chunk": float(stop.mean()), "mean_compute_fraction": float(np.mean(stop / horizon)), "mean_net_utility": float(np.mean([row.net_utility for row in rows]))}


def _pair(left: Sequence[Rollout], right: Sequence[Rollout]) -> list[tuple[Rollout, Rollout]]:
    a = {(row.key, row.horizon): row for row in left}
    b = {(row.key, row.horizon): row for row in right}
    if set(a) != set(b):
        raise ValueError("rollout coverage differs")
    return [(a[key], b[key]) for key in sorted(a)]


def policy_delta(budget: Sequence[Rollout], age: Sequence[Rollout]) -> dict[str, Any]:
    pairs = _pair(budget, age)
    return {"mean": float(np.mean([a.net_utility - b.net_utility for a, b in pairs])), "by_horizon": {str(horizon): float(np.mean([a.net_utility - b.net_utility for a, b in pairs if a.horizon == horizon])) for horizon in sorted({a.horizon for a, _ in pairs})}, "by_phase": {str(phase): [a.net_utility - b.net_utility for a, b in pairs if a.phase == phase] for phase in sorted({a.phase for a, _ in pairs})}, "by_source": {str(source): [a.net_utility - b.net_utility for a, b in pairs if a.source == source] for source in sorted({a.source for a, _ in pairs})}}


def cluster_bootstrap_delta(budget: Sequence[Rollout], age: Sequence[Rollout], samples: int, seed: int) -> dict[str, float | int | None]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    grouped: dict[str, list[float]] = defaultdict(list)
    for left, right in _pair(budget, age):
        grouped[left.group_id].append(left.net_utility - right.net_utility)
    groups = sorted(grouped)
    if len(groups) < 2:
        return {"samples": samples, "groups": len(groups), "mean": None, "lower_95": None, "upper_95": None}
    rng = np.random.default_rng(seed)
    draws = [float(np.mean([value for group in rng.choice(groups, len(groups), replace=True) for value in grouped[str(group)]])) for _ in range(samples)]
    values = [value for group in groups for value in grouped[group]]
    return {"samples": samples, "groups": len(groups), "mean": float(np.mean(values)), "lower_95": float(np.quantile(draws, 0.025)), "upper_95": float(np.quantile(draws, 0.975))}


def evaluate_price(rows: Sequence[Trajectory], horizons: Sequence[int], price: float, n_folds: int) -> tuple[dict[str, Any], dict[str, list[Rollout]], dict[str, Any]]:
    examples = build_examples(rows, horizons, price)
    policies: dict[str, list[Rollout]] = {}
    diagnostics: dict[str, Any] = {}
    for model in MODELS:
        prediction, diagnostics[model] = cross_fitted_predictions(examples, model, n_folds)
        lookup = {(row.key, row.horizon, row.chunk): float(value) for row, value in zip(examples, prediction, strict=True)}
        policies[model] = rollout_policy(rows, horizons, price, lookup)
    policies["complexity"] = rollout_policy(rows, horizons, price, complexity=True)
    policies["oracle"] = rollout_policy(rows, horizons, price, oracle=True)
    policies["fixed_full"] = rollout_policy(rows, horizons, price, fixed_full=True)
    return {name: _summary(values) for name, values in policies.items()}, policies, diagnostics


def analyze(rows: Sequence[Trajectory], *, horizons: Sequence[int] = HORIZONS, prices: Sequence[float] = PRICES, primary_price: float = PRIMARY_PRICE, n_folds: int = 5, bootstrap_samples: int = 1000, seed: int = 0, mode: Literal["pilot", "final"] = "pilot", manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    if not rows or n_folds < 2 or bootstrap_samples < 1:
        raise ValueError("analysis needs rows, >=2 folds, and positive bootstrap samples")
    if mode == "final" and bootstrap_samples < 1000:
        raise ValueError("final analysis requires at least 1000 bootstrap samples")
    horizons = tuple(sorted({int(value) for value in horizons}))
    prices = tuple(sorted({float(value) for value in prices}))
    if not horizons or max(horizons) > MAX_CHUNKS or any(price < 0.0 or not math.isfinite(price) for price in prices) or primary_price not in prices:
        raise ValueError("invalid horizons/prices")
    collection = collection_status(rows, manifest)
    if mode == "final" and not collection["passed"]:
        raise ValueError("final analysis requires a production-shaped collector manifest")

    curve: dict[str, Any] = {}
    primary_summaries: dict[str, Any] | None = None
    primary_policies: dict[str, list[Rollout]] | None = None
    primary_diagnostics: dict[str, Any] | None = None
    for price in prices:
        summaries, policies, diagnostics = evaluate_price(rows, horizons, price, n_folds)
        curve[f"{price:.8g}"] = {"summaries": summaries, "budget_minus_age": policy_delta(policies["M_budget"], policies["M_age"])}
        if math.isclose(price, primary_price, abs_tol=1e-15):
            primary_summaries, primary_policies, primary_diagnostics = summaries, policies, diagnostics
    assert primary_summaries is not None
    assert primary_policies is not None
    bootstrap = cluster_bootstrap_delta(primary_policies["M_budget"], primary_policies["M_age"], bootstrap_samples, seed)
    delta = policy_delta(primary_policies["M_budget"], primary_policies["M_age"])
    budget, age = primary_summaries["M_budget"], primary_summaries["M_age"]
    positive_horizons = sum(value > 0.0 for value in delta["by_horizon"].values())
    tail_ok = float(budget["p95_regret"]) <= float(age["p95_regret"]) + 0.0005 and float(budget["p99_regret"]) <= float(age["p99_regret"]) + 0.0005
    opportunities = sum(max((trajectory.snapshots[a].regret_score - trajectory.snapshots[b].regret_score for a in range(MAX_CHUNKS - 1) for b in range(a + 1, MAX_CHUNKS)), default=-math.inf) >= 0.001 for trajectory in rows)
    oracle, fixed = primary_summaries["oracle"], primary_summaries["fixed_full"]
    oracle_savings = 1.0 - float(oracle["mean_compute_fraction"]) / float(fixed["mean_compute_fraction"])
    oracle_change = float(oracle["mean_regret"]) - float(fixed["mean_regret"])
    oracle_headroom = (oracle_savings >= 0.10 and oracle_change <= 0.0001) or (oracle_change <= -0.00025 and float(oracle["mean_compute_fraction"]) <= float(fixed["mean_compute_fraction"]) + 1e-12)
    age_compute, budget_compute = float(age["mean_compute_fraction"]), float(budget["mean_compute_fraction"])
    savings = (age_compute - budget_compute) / age_compute if age_compute else 0.0
    regret_change = float(budget["mean_regret"]) - float(age["mean_regret"])
    practical = (savings >= 0.08 and regret_change <= 0.0001) or (regret_change <= -0.00025 and budget_compute <= age_compute + 1e-12)
    canonical = all(row.group_kind == "source_game" for row in rows)
    conformant = collection["passed"] and horizons == HORIZONS and prices == PRICES and math.isclose(primary_price, PRIMARY_PRICE, abs_tol=1e-15)
    common = {"protocol_conformant": conformant, "positions": len(rows), "groups": len({row.group_id for row in rows}), "source_game_grouping": canonical, "opportunities": int(opportunities), "positive_horizons": positive_horizons, "tail_rule_passed": tail_ok, "oracle_headroom": oracle_headroom, "practical_budget_vs_age": practical, "bootstrap": bootstrap, "budget_minus_age": delta}
    if mode == "pilot":
        gates = {"positions": len(rows) >= 512, "groups": len({row.group_id for row in rows}) >= 256, "protocol": conformant, "opportunities": opportunities >= 30, "oracle_headroom": oracle_headroom, "two_horizons": positive_horizons >= 2, "tails": tail_ok}
        signal = all(gates.values())
        verdict = "EXPAND_TO_1024" if signal and canonical else "POSITIVE_SMOKE_REQUIRES_SOURCE_GAME_REPLICATION" if signal else "INSUFFICIENT_PILOT_SAMPLE" if len(rows) < 512 else "STOP_NO_EXPANSION"
    else:
        subgroup_phase = sum(float(np.mean(values)) > 0.0 for values in delta["by_phase"].values() if len(values) >= 20) >= 2
        subgroup_source = sum(float(np.mean(values)) > 0.0 for values in delta["by_source"].values() if len(values) >= 20) >= 2
        lower = bootstrap["lower_95"]
        gates = {"positions": len(rows) >= 1024, "groups": len({row.group_id for row in rows}) >= 512, "protocol": conformant, "source_games": canonical, "bootstrap": lower is not None and float(lower) > 0.0, "practical": practical, "beats_complexity": float(budget["mean_net_utility"]) > float(primary_summaries["complexity"]["mean_net_utility"]), "two_horizons": positive_horizons >= 2, "tails": tail_ok, "subgroups": subgroup_phase and subgroup_source}
        verdict = "ADVANCE_TO_REAL_CLOCK_BANK" if all(gates.values()) else "INSUFFICIENT_FINAL_SAMPLE" if len(rows) < 1024 else "STOP_PERMANENTLY"
    return {"schema": ANALYSIS_SCHEMA, "experiment": {"horizons": list(horizons), "prices": list(prices), "primary_price": primary_price, "models": list(MODELS), "online_rule": "continue iff held-out prediction > 0", "quota_ranking": False, "reentry": False}, "collection_status": collection, "price_curve": curve, "primary_model_diagnostics": primary_diagnostics, "preregistered_verdict": {**common, "gates": gates, "verdict": verdict}}


def _csv(raw: str, cast: Any) -> list[Any]:
    values = [cast(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("empty comma-separated argument")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(prog="analyze_varying_budget_controller")
    parser.add_argument("--in", dest="input_path", type=Path, required=True)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--horizons", default=",".join(map(str, HORIZONS)))
    parser.add_argument("--prices", default=",".join(map(str, PRICES)))
    parser.add_argument("--primary-price", type=float, default=PRIMARY_PRICE)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("pilot", "final"), default="pilot")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        rows = load_trajectories(args.input_path)
        manifest = load_manifest(args.input_path, args.meta)
        result = analyze(rows, horizons=_csv(args.horizons, int), prices=_csv(args.prices, float), primary_price=args.primary_price, n_folds=args.folds, bootstrap_samples=args.bootstrap_samples, seed=args.seed, mode=args.mode, manifest=manifest)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        if args.out.resolve() == args.input_path.resolve():
            raise SystemExit("--out must not overwrite the input bank")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temp = args.out.with_name(f".{args.out.name}.tmp")
        temp.write_text(rendered + "\n", encoding="utf-8")
        temp.replace(args.out)
    print(rendered)


if __name__ == "__main__":
    main()
