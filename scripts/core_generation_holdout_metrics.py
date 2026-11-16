"""Fixed all-row HOLD observations and equal-parent health inference.

Top-level stdlib only. Tensor scoring and local NumPy bootstrap are explicit
functions; this module does not load models, open data or issue authority.
"""
from __future__ import annotations

import math
from typing import Any

LABELS = ("A39", "A", "B", "C")
METRICS = ("wdl_ce", "brier", "score_mse", "policy_ce", "reference_forward_kl")
MIN_PARENTS = 48
MIN_ROWS = 2000
BOOTSTRAP_SEED = 2026111680
BOOTSTRAP_REPLICATIONS = 10000


def require(value: Any, message: str) -> None:
    if not value:
        raise ValueError(message)


def row_metrics(outputs: dict[str, Any], legal: Any, policy: Any, outcome: Any,
                reference_log_probs: Any | None = None) -> tuple[dict[str, Any], Any, Any]:
    """FP32 observations, win0/draw1/loss2, explicit legal support, no gradient."""
    import torch
    from chess_anti_engine.train.losses import apply_policy_mask_to_logits, soft_cross_entropy

    logits = outputs["policy"] if "policy" in outputs else outputs.get("policy_own")
    wdl = outputs.get("wdl")
    require(isinstance(logits, torch.Tensor) and isinstance(wdl, torch.Tensor), "missing policy/WDL outputs")
    n, width = logits.shape if logits.ndim == 2 else (0, 0)
    require(n > 0 and width == 1858 and wdl.shape == (n, 3) and legal.shape == policy.shape == logits.shape
            and outcome.shape == (n,), "HOLD metric shape mismatch")
    require(outcome.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            and bool(((outcome >= 0) & (outcome <= 2)).all()), "invalid genuine outcome; no censored draw fallback")
    require(all(bool(torch.isfinite(tensor).all()) for tensor in (logits, wdl, legal, policy)), "nonfinite metric inputs")
    require(bool(((legal == 0) | (legal == 1)).all()) and bool(legal.bool().any(dim=-1).all())
            and bool((policy >= 0).all()) and bool((policy.sum(-1) > 0).all())
            and not bool(policy[~legal.bool()].ne(0).any()), "invalid legal/target support")
    log_wdl = torch.log_softmax(wdl.float(), -1)
    probability = torch.softmax(wdl.float(), -1)
    truth = torch.nn.functional.one_hot(outcome.long(), 3).float()
    expected_score = probability[:, 0] + 0.5 * probability[:, 1]
    true_score = 1.0 - 0.5 * outcome.float()
    # Preserve actual CORE F32 legal-mask/target-renormalization convention.
    mask_batch = {"legal_mask": legal.float(), "has_legal_mask": torch.ones(n, device=logits.device)}
    masked = apply_policy_mask_to_logits(logits.float(), mask_batch, "legal_mask", "has_legal_mask")
    log_policy = torch.log_softmax(logits.float().masked_fill(~legal.bool(), -torch.inf), -1)
    safe_log_policy = torch.where(legal.bool(), log_policy, 0.0)
    if reference_log_probs is None:
        kl = torch.zeros(n, dtype=torch.float32, device=logits.device)
    else:
        require(reference_log_probs.shape == logits.shape and bool(torch.isfinite(reference_log_probs).all()), "invalid reference cache")
        q = torch.where(legal.bool(), reference_log_probs.float().exp(), 0.0)
        require(bool(torch.isfinite(q).all()) and bool((q.sum(-1) - 1).abs().le(2e-5).all()), "reference conditional policy not normalized")
        kl = (q * (reference_log_probs.float() - safe_log_policy)).sum(-1)
    values = {"wdl_ce": -log_wdl.gather(1, outcome.long().unsqueeze(1)).squeeze(1),
              "brier": ((probability - truth) ** 2).sum(-1),
              "score_mse": (expected_score - true_score) ** 2,
              "policy_ce": soft_cross_entropy(masked, policy.float()), "reference_forward_kl": kl}
    require(all(value.shape == (n,) and value.dtype == torch.float32 and bool(torch.isfinite(value).all())
                for value in values.values()), "nonfinite/unqualified row metrics")
    return values, safe_log_policy, log_wdl


def summarize(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute row and equal-game/equal-parent estimands from fixed sums."""
    require(type(games) is list and games, "empty HOLD observations")
    parents: dict[str, list[dict[str, Any]]] = {}
    physical: set[str] = set()
    for game in games:
        require(set(game) == {"physical_game_id", "parent_id", "rows", "row_identity_sha256", "models"}, "invalid game observation schema")
        require(type(game["physical_game_id"]) is str and game["physical_game_id"] not in physical
                and type(game["parent_id"]) is str and game["parent_id"], "duplicate/invalid observed game")
        physical.add(game["physical_game_id"])
        require(type(game["rows"]) is int and game["rows"] > 0 and set(game["models"]) == set(LABELS), "invalid observation denominator/model set")
        for model in game["models"].values():
            require(set(model) == set(METRICS) and all(type(v) in (int, float) and math.isfinite(v) for v in model.values()), "nonfinite/missing game numerator")
        parents.setdefault(game["parent_id"], []).append(game)
    require(all(len(members) == 1 for members in parents.values()), "selected one-game-per-parent rule violated")
    total_rows = sum(game["rows"] for game in games)
    result = {}
    for label in LABELS:
        row = {metric: math.fsum(game["models"][label][metric] for game in games) / total_rows for metric in METRICS}
        parent = {metric: math.fsum(math.fsum(game["models"][label][metric] / game["rows"] for game in members)
                   / len(members) for members in parents.values()) / len(parents) for metric in METRICS}
        result[label] = {"row_weighted": row, "equal_parent": parent}
    return {"games": len(games), "parents": len(parents), "rows": total_rows, "models": result}


def parent_health(games: list[dict[str, Any]]) -> dict[str, Any]:
    """Fixed CPU-only paired PCG64/linear percentile bootstrap, veto ONLY."""
    import numpy as np

    require(np.__version__ == "2.2.6", "bootstrap requires selected NumPy2.2.6")
    aggregate = summarize(games)
    require(aggregate["parents"] >= MIN_PARENTS and aggregate["rows"] >= MIN_ROWS, "UNREAD insufficient HOLD parents/rows")
    ordered = sorted(games, key=lambda game: game["parent_id"])
    differences = np.asarray([(game["models"]["C"]["wdl_ce"] - game["models"]["A39"]["wdl_ce"]) / game["rows"]
                              for game in ordered], dtype=np.float64)
    require(np.isfinite(differences).all(), "UNREAD nonfinite parent differences")
    before = np.random.get_state()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    require(type(rng.bit_generator).__name__ == "PCG64", "wrong bootstrap generator")
    draws = rng.integers(0, len(differences), size=(BOOTSTRAP_REPLICATIONS, len(differences)))
    distribution = differences[draws].mean(axis=1, dtype=np.float64)
    lower, upper = np.quantile(distribution, [0.025, 0.975], method="linear")
    after = np.random.get_state()
    require(before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:], "bootstrap changed global RNG")
    point = float(differences.mean(dtype=np.float64))
    require(all(math.isfinite(value) for value in (point, lower, upper)), "UNREAD invalid bootstrap interval")
    return {"aggregate": aggregate, "C_minus_A39_equal_parent_WDL_CE": {"point": point, "lower95": float(lower), "upper95": float(upper)},
            "bootstrap": {"seed": BOOTSTRAP_SEED, "replications": BOOTSTRAP_REPLICATIONS, "generator": "PCG64",
                          "quantile_method": "linear", "paired_unit": "complete_parent", "degenerate": bool(np.all(differences == differences[0]))},
            "severe_damage_veto": point > 0.20 and float(lower) > 0,
            "veto_scope": "VETO_ONLY; nontriggering does not establish safety, noninferiority, benefit or playing authority"}
