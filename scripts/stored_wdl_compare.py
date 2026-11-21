"""Paired stored-input WDL measurement kernel, not a model loader or Elo test.

Callers supply independently qualified inference adapters. Checkpoint identities
and training memberships are declarations, not authenticated by this kernel.
The row digest authenticates supplied bytes only, not their historical origin.
Unlike audit_targets, both WDL forwards here consume identical stored planes.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np

LAYOUT = "stored-175-NCHW-float32-v1"
MAX_ROWS = 4096
MAX_BOOTSTRAPS = 10000


def rows_digest(x: np.ndarray, row_ids: Sequence[str], game_ids: Sequence[str]) -> str:
    """Canonical digest binds exact float32 rows and ordered composite identities."""
    if x.dtype != np.dtype("float32") or x.ndim != 4 or x.shape[1:] != (175, 8, 8):
        raise ValueError("expected float32 [N,175,8,8] stored rows")
    if not 1 <= len(x) <= MAX_ROWS or not np.isfinite(x).all():
        raise ValueError("invalid or excessive stored rows")
    if len(row_ids) != len(x) or len(game_ids) != len(x):
        raise ValueError("identity count mismatch")
    if any(not _nonempty(v) for v in [*row_ids, *game_ids]):
        raise ValueError("nonempty composite row and source-game identities required")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("duplicate row identity")
    h = hashlib.sha256()
    h.update(json.dumps([LAYOUT, list(x.shape), list(row_ids), list(game_ids)],
                        separators=(",", ":")).encode())
    h.update(np.ascontiguousarray(x, dtype="<f4").tobytes())
    return h.hexdigest()


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _membership(value: object) -> frozenset[str]:
    if not isinstance(value, frozenset) or any(not _nonempty(v) for v in value):
        raise ValueError("explicit immutable source-game membership sets required")
    return cast(frozenset[str], value)


def _identity(value: object) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("checkpoint and row identities must be lowercase SHA256")


def _simplex(value: np.ndarray, n: int, label: str) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (n, 3) or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any():
        raise ValueError(f"{label}: expected finite W,D,L probabilities")
    if not np.allclose(a.sum(axis=1), 1.0, rtol=0, atol=1e-6):
        raise ValueError(f"{label}: probabilities must sum to one")
    return a


def compare(
    x: np.ndarray, targets: np.ndarray, *,
    row_ids: Sequence[str], game_ids: Sequence[str], expected_rows_sha256: str,
    donor_sha256: str, candidate_sha256: str, layout: str,
    donor: Callable[[np.ndarray], np.ndarray],
    candidate: Callable[[np.ndarray], np.ndarray],
    donor_training_games: frozenset[str], candidate_training_games: frozenset[str],
    target_description: str, bootstraps: int = 2000, seed: int = 0,
) -> dict:
    """Membership-gated paired source-game comparison; existing contract preserved."""
    return _compare(x, targets, row_ids=row_ids, game_ids=game_ids,
                    expected_rows_sha256=expected_rows_sha256,
                    donor_sha256=donor_sha256, candidate_sha256=candidate_sha256,
                    layout=layout, donor=donor, candidate=candidate,
                    donor_training_games=donor_training_games,
                    candidate_training_games=candidate_training_games,
                    target_description=target_description, bootstraps=bootstraps, seed=seed)


def _compare(
    x: np.ndarray, targets: np.ndarray, *,
    row_ids: Sequence[str], game_ids: Sequence[str], expected_rows_sha256: str,
    donor_sha256: str, candidate_sha256: str, layout: str,
    donor: Callable[[np.ndarray], np.ndarray],
    candidate: Callable[[np.ndarray], np.ndarray],
    donor_training_games: frozenset[str] | None, candidate_training_games: frozenset[str] | None,
    target_description: str, bootstraps: int = 2000, seed: int = 0,
    descriptive: bool = False, group_kind: str = "source_game",
    expected_targets_sha256: str | None = None,
) -> dict:
    """Return equal-game-weight paired deltas (candidate minus donor).

    Targets and outputs are W,D,L probabilities in stored row side-to-move POV.
    Each adapter must use the declared checkpoint and this exact input layout;
    this interface cannot authenticate callback implementation or model identity.
    Training game sets must be COMPLETE, composite source/game IDs. Disjointness
    is checked against the supplied sets, not proven against inaccessible history.
    Bootstrap resamples source-game means, never positions. Negative delta is
    better target fit, not strength. Log loss uses an explicit 1e-12 floor.
    """
    row_ids, game_ids = tuple(row_ids), tuple(game_ids)
    if layout != LAYOUT:
        raise ValueError("unsupported stored layout; no FEN fallback")
    for identity in (expected_rows_sha256, donor_sha256, candidate_sha256):
        _identity(identity)
    if not _nonempty(target_description):
        raise ValueError("target semantics and POV description required")
    if type(bootstraps) is not int or not 1 <= bootstraps <= MAX_BOOTSTRAPS:
        raise ValueError("bounded positive bootstrap count required")
    if type(seed) is not int or seed < 0:
        raise ValueError("nonnegative integer seed required")
    if descriptive:
        if len(x) > 256 or group_kind not in {"source_game", "connected_component"}:
            raise ValueError("descriptive mode requires <=256 rows and explicit supported grouping")
        _identity(expected_targets_sha256)
    else:
        donor_training_games = _membership(donor_training_games)
        candidate_training_games = _membership(candidate_training_games)
    if rows_digest(x, row_ids, game_ids) != expected_rows_sha256:
        raise ValueError("stored-row digest mismatch")
    groups = sorted(set(game_ids))
    if len(groups) < 2:
        raise ValueError("at least two supplied groups required" if descriptive else
                         "at least two independent source-game groups required")
    if not descriptive and set(groups) & (cast(frozenset[str], donor_training_games) | cast(frozenset[str], candidate_training_games)):
        raise ValueError("declared training/holdout source-game overlap")
    target = _simplex(targets, len(x), "targets").copy()
    target.flags.writeable = False
    if descriptive and targets_digest(target, row_ids, game_ids, rows_sha256=expected_rows_sha256,
                                      target_description=target_description) != expected_targets_sha256:
        raise ValueError("ordered target identity mismatch")
    # Private copies prevent one adapter changing the other adapter's inputs.
    outputs = []
    for adapter in (donor, candidate):
        inputs = x.copy()
        inputs.flags.writeable = False
        output = adapter(inputs)
        if rows_digest(inputs, row_ids, game_ids) != expected_rows_sha256:
            raise ValueError("adapter mutated stored input")
        outputs.append(_simplex(output, len(x), "prediction").copy())
    losses = [np.column_stack((
            -(target * np.log(np.maximum(probabilities, 1e-12))).sum(axis=1),
            np.square(probabilities - target).sum(axis=1),
        )) for probabilities in outputs]
    indices = [np.flatnonzero(np.asarray(game_ids) == g) for g in groups]
    donor_groups, candidate_groups = [
        np.stack([loss[index].mean(axis=0) for index in indices]) for loss in losses
    ]
    delta = candidate_groups - donor_groups
    rng = np.random.default_rng(seed)
    sampled = np.empty((bootstraps, 2), dtype=np.float64)
    for i in range(bootstraps):
        sampled[i] = delta[rng.integers(0, len(groups), size=len(groups))].mean(axis=0)
    interval = np.quantile(sampled, [0.025, 0.975], axis=0)
    result: dict[str, Any] = {
        "schema": "stored-wdl-paired/v1", "layout": layout,
        "rows_sha256": expected_rows_sha256,
        "target_sha256": hashlib.sha256(np.ascontiguousarray(target, dtype="<f8").tobytes()).hexdigest(),
        "donor_sha256_declared": donor_sha256, "candidate_sha256_declared": candidate_sha256,
        "target_description": target_description, "rows": len(x), "source_games": len(groups),
        "weighting": "equal source-game means", "bootstrap_seed": seed,
        "bootstrap_replicates": bootstraps, "log_probability_floor": 1e-12,
        "metrics": {
            name: {"donor": float(donor_groups[:, j].mean()),
                   "candidate": float(candidate_groups[:, j].mean()),
                   "delta": float(delta[:, j].mean()),
                   "paired_percentile_95": interval[:, j].tolist()}
            for j, name in enumerate(("wdl_log_loss", "wdl_brier"))
        },
        "per_game": [{"source_game": g, "rows": len(index), "delta": delta[i].tolist()}
                     for i, (g, index) in enumerate(zip(groups, indices, strict=True))],
        "limits": "Supplied membership disjointness only; adapter/checkpoint and provenance declarations unverified; target fit is not Elo.",
    }

    if descriptive:
        result["schema"] = "stored-wdl-descriptive/v1"
        result["mode"] = "descriptive"
        result["group_kind_declared"] = group_kind
        result["groups"] = result.pop("source_games")
        result["weighting"] = "equal supplied-group means"
        result["per_group"] = [{"group": row["source_game"], "rows": row["rows"], "delta": row["delta"]}
                               for row in result.pop("per_game")]
        result["ordered_targets_sha256"] = expected_targets_sha256
        result["wdl_order"] = ["W", "D", "L"]
        result["pov_declared"] = "stored row side to move"
        result["training_exposure"] = "unknown"
        result["provenance"] = "unverified"
        result["controls_performed"] = []
        result["controls_required_but_not_run"] = ["donor-donor", "batch-invariance", "POV"]
        for metric in result["metrics"].values():
            metric["conditional_descriptive_percentile_95"] = metric.pop("paired_percentile_95")
        result["limits"] = ("Conditional descriptive variability across supplied groups only; "
                            "no heldout, independence, generalization or strength claim. "
                            "Training exposure, group provenance and callback/checkpoint identity unverified.")
    return result


def targets_digest(targets: np.ndarray, row_ids: Sequence[str], group_ids: Sequence[str], *,
                   rows_sha256: str, target_description: str) -> str:
    """Bind canonical WDL targets to ordered row/group identifiers, not origin."""
    row_ids, group_ids = tuple(row_ids), tuple(group_ids)
    target = _simplex(targets, len(row_ids), "targets")
    if len(group_ids) != len(row_ids):
        raise ValueError("target identity count mismatch")
    _identity(rows_sha256)
    if not _nonempty(target_description):
        raise ValueError("target semantics required")
    h = hashlib.sha256(json.dumps(["stored-STM-WDL/v1", rows_sha256, target_description,
                                 list(row_ids), list(group_ids)], separators=(",", ":")).encode())
    h.update(np.ascontiguousarray(target, dtype="<f8").tobytes())
    return h.hexdigest()


def compare_descriptive(
    x: np.ndarray, targets: np.ndarray, *, row_ids: Sequence[str], group_ids: Sequence[str],
    group_kind: str, expected_rows_sha256: str, expected_targets_sha256: str,
    donor_sha256: str, candidate_sha256: str, layout: str,
    donor: Callable[[np.ndarray], np.ndarray], candidate: Callable[[np.ndarray], np.ndarray],
    target_description: str, bootstraps: int = 2000, seed: int = 0,
) -> dict:
    """<=256-row descriptive drift; no invented training membership/holdout claim.

    Inputs/targets/predictions use W,D,L in stored-row side-to-move POV. Group
    identifiers are supplied units, not proof of original source games or
    independence. No control runs are implied by producing this report.
    """
    return _compare(x, targets, row_ids=row_ids, game_ids=group_ids,
                    expected_rows_sha256=expected_rows_sha256,
                    expected_targets_sha256=expected_targets_sha256,
                    donor_sha256=donor_sha256, candidate_sha256=candidate_sha256,
                    layout=layout, donor=donor, candidate=candidate,
                    donor_training_games=None, candidate_training_games=None,
                    target_description=target_description, bootstraps=bootstraps,
                    seed=seed, descriptive=True, group_kind=group_kind)
