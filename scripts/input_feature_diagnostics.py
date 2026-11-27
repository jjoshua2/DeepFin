"""Numerical kernels for frozen-network v2 input-feature diagnostics.

Perturbations measure current output sensitivity, not feature quality after
retraining or playing strength. No checkpoint, replay, training or GPU imports.
The caller supplies authentic row identities and source-qualified game keys.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

# Absolute channels in v2_threats: the first 112 history/metadata planes are
# always protected. Combined groups deliberately overlap their named families.
FEATURE_GROUPS: dict[str, tuple[int, int]] = {
    "king_safety": (112, 122),
    "pins": (122, 128),
    "pawn_structure": (128, 136),
    "mobility": (136, 142),
    "outposts_space": (142, 146),
    "attacks_all": (146, 148),
    "attacks_typed": (148, 160),
    "attacker_counts": (160, 162),
    "hanging": (162, 164),
    "cheaper_attacker": (164, 166),
    "safe_checks": (166, 170),
    "control": (170, 171),
    "pawn_tension": (171, 173),
    "pawn_storm": (173, 175),
    "v1_all": (112, 146),
    "threats_all": (146, 175),
    "extras_all": (112, 175),
}


def _nonblank_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_real_array(value: object) -> bool:
    return (
        isinstance(value, np.ndarray) and value.dtype.kind in "biuf"
        and bool(np.isfinite(value).all())
    )


def _identities(values: Sequence[str], n: int, name: str) -> list[str]:
    if isinstance(values, (str, bytes)) or len(values) != n:
        raise ValueError(f"{name} must contain one nonblank string per row")
    result = list(values)
    if not all(_nonblank_string(value) for value in result):
        raise ValueError(f"{name} must contain one nonblank string per row")
    return result


def _nonnegative_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def perturb_inputs(
    x: np.ndarray, *, family: str, mode: str, row_ids: Sequence[str], seed: int,
) -> np.ndarray:
    """Copy and perturb one v2 family, retaining every other input bit.

    ``spatial_shuffle`` uses one square permutation for every channel in the
    chosen family. Its stream depends only on seed and row identity, not row
    order, batch boundaries or family; overlapping family tests are comparable.
    It preserves per-plane histograms and within-family channel alignment while
    disrupting alignment with the unperturbed board. It is intentionally not a
    legal position or chess symmetry. ``identity`` still returns a fresh copy.
    """
    if (
        not _finite_real_array(x) or x.ndim != 4
        or x.shape[1:] != (175, 8, 8) or x.shape[0] == 0
    ):
        raise ValueError("x must be finite real nonempty [N, 175, 8, 8] v2 inputs")
    if family not in FEATURE_GROUPS:
        raise ValueError(f"unknown feature family {family!r}")
    if mode not in ("identity", "zero", "spatial_shuffle"):
        raise ValueError(f"unknown perturbation mode {mode!r}")
    identities = _identities(row_ids, len(x), "row_ids")
    _nonnegative_integer(seed, "seed")
    result = x.copy()
    start, stop = FEATURE_GROUPS[family]
    if mode == "zero":
        result[:, start:stop] = 0
    elif mode == "spatial_shuffle":
        for index, row_id in enumerate(identities):
            digest = hashlib.sha256(
                f"deepfin-feature-spatial-v1\0{seed}\0{row_id}".encode()
            ).digest()
            rng = np.random.Generator(np.random.PCG64(int.from_bytes(digest, "big")))
            permutation = rng.permutation(64)
            planes = x[index, start:stop].reshape(stop - start, 64)
            result[index, start:stop] = planes[:, permutation].reshape(stop - start, 8, 8)
    return result


def _logits(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    if (
        not _finite_real_array(value) or value.shape != shape or value.dtype.kind == "b"
    ):
        raise ValueError(f"{name} must be finite real logits with shape {shape}")
    return value.astype(np.float64, copy=False)


def _log_probabilities(logits: np.ndarray, legal: np.ndarray) -> np.ndarray:
    masked = np.where(legal, logits, -np.inf)
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = masked - np.max(masked, axis=1, keepdims=True)
        result = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    if not np.isfinite(result[legal]).all():
        raise ValueError("logit range exceeds finite float64 log-probabilities")
    return result


def compare_outputs(
    reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
    legal: np.ndarray,
) -> dict[str, np.ndarray]:
    """Per-row T=1 sensitivity of MAIN policy/WDL raw logits.

    Both dictionaries require ``policy`` [N,1858] and ``wdl`` [N,3] logits;
    WDL order is win/draw/loss. The caller maps the model's ``policy_own`` alias
    explicitly; auxiliary ``policy_sf`` is never a fallback. Policy is masked
    and renormalized over the same original legal set. Ties choose the lowest
    legal policy index. Signed Q change is candidate minus reference, Q = W-L.
    """
    if (
        not _finite_real_array(legal) or legal.ndim != 2
        or legal.shape[1] != 1858 or legal.shape[0] == 0
        or not np.isin(legal, (0, 1)).all()
    ):
        raise ValueError("legal must be a nonempty binary [N, 1858] array")
    mask = legal.astype(np.bool_, copy=False)
    if not mask.any(axis=1).all():
        raise ValueError("each row must have at least one legal move")
    n = len(mask)
    outputs: list[tuple[np.ndarray, np.ndarray]] = []
    for name, values in (("reference", reference), ("candidate", candidate)):
        if "policy" not in values or "wdl" not in values:
            raise ValueError(f"{name} requires main policy and wdl logits")
        policy = _logits(values["policy"], (n, 1858), f"{name}.policy")
        wdl = _logits(values["wdl"], (n, 3), f"{name}.wdl")
        outputs.append((
            _log_probabilities(policy, mask),
            np.exp(_log_probabilities(wdl, np.ones_like(wdl, dtype=np.bool_))),
        ))
    (ref_logp, ref_wdl), (cand_logp, cand_wdl) = outputs
    ref_p, cand_p = np.exp(ref_logp), np.exp(cand_logp)
    log_ratio = np.zeros_like(ref_logp)
    np.subtract(ref_logp, cand_logp, out=log_ratio, where=mask)
    q_change = (cand_wdl[:, 0] - cand_wdl[:, 2]) - (ref_wdl[:, 0] - ref_wdl[:, 2])
    return {
        "policy_kl": np.maximum((ref_p * log_ratio).sum(axis=1), 0.0),
        "top1_changed": np.argmax(ref_logp, axis=1) != np.argmax(cand_logp, axis=1),
        "policy_total_variation": np.abs(ref_p - cand_p).sum(axis=1) / 2.0,
        "signed_q_change": q_change,
        "abs_q_change": np.abs(q_change),
        "wdl_total_variation": np.abs(ref_wdl - cand_wdl).sum(axis=1) / 2.0,
    }


def summarize_metrics(
    metrics: dict[str, np.ndarray], *, cluster_keys: Sequence[str],
    bootstrap: int = 1000, seed: int = 0,
) -> dict[str, Any]:
    """Row and equal-game means; percentile bootstrap of equal-game means.

    Keys must identify the original game *qualified by source*, not row/shard
    IDs alone. This kernel validates nonblank keys, not their provenance or
    statistical independence. First average rows within a game, then give each
    game equal weight. Bootstrap games with replacement (all their rows stay
    together). This interval is NOT for the differently weighted row mean.
    One game cannot support a cluster interval; bootstrap=0 disables intervals.
    """
    if not metrics:
        raise ValueError("at least one per-row metric is required")
    _nonnegative_integer(bootstrap, "bootstrap")
    _nonnegative_integer(seed, "seed")
    n = len(cluster_keys)
    if n == 0:
        raise ValueError("at least one row is required")
    keys = _identities(cluster_keys, n, "source-qualified game cluster_keys")
    names = sorted(metrics)
    columns = []
    for name in names:
        value = metrics[name]
        if (
            not _finite_real_array(value) or value.shape != (n,)
        ):
            raise ValueError(f"metric {name!r} must be a finite real [{n}] vector")
        columns.append(value.astype(np.float64, copy=False))
    values = np.column_stack(columns)
    game_names, inverse = np.unique(np.asarray(keys), return_inverse=True)
    games = len(game_names)
    counts = np.bincount(inverse, minlength=games)
    game_means = np.zeros((games, len(names)), dtype=np.float64)
    np.add.at(game_means, inverse, values / counts[inverse, None])
    intervals: np.ndarray | None = None
    if bootstrap and games > 1:
        rng = np.random.default_rng(seed)
        draws = np.empty((bootstrap, len(names)), dtype=np.float64)
        # Bound intermediate [draw, game, metric] storage even for large banks.
        chunk = max(1, 65536 // (games * len(names)))
        for start in range(0, bootstrap, chunk):
            stop = min(start + chunk, bootstrap)
            sample = rng.integers(games, size=(stop - start, games))
            draws[start:stop] = (game_means[sample] / games).sum(axis=1)
        intervals = np.quantile(draws, (0.025, 0.975), axis=0)
    row_means = (values / n).sum(axis=0)
    equal_game_means = (game_means / games).sum(axis=0)
    return {
        "rows": n,
        "games": games,
        "bootstrap_replicates": bootstrap,
        "bootstrap_seed": seed,
        "estimand": "Equal weight per source-qualified game, mean over selected rows within game.",
        "interval_method": "95% percentile bootstrap resampling whole game means; not a row-mean interval.",
        "interval_status": (
            "disabled" if not bootstrap else "unavailable_one_game" if games == 1 else "available"
        ),
        "few_cluster_caution": games < 30,
        "caution": (
            "Fewer than 30 game clusters: interval coverage may be poor. " if games < 30 else ""
        ) + (
            "Game keys alone do not establish independent games or representative sampling. "
            "Frozen-input sensitivity does not establish retrained feature value or playing strength."
        ),
        "metrics": {
            name: {
                "row_mean": float(row_means[index]),
                "equal_game_mean": float(equal_game_means[index]),
                "equal_game_ci95": None if intervals is None else intervals[:, index].tolist(),
            }
            for index, name in enumerate(names)
        },
    }
