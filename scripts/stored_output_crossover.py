"""Synchronous CPU output crossover for a separately bounded search diagnostic.

No model loading, checkpoint authentication, history reconstruction or launcher.
The caller must qualify these first. Both root and leaf evaluations use the
same evaluator. Only the explicit Python search route is exposed; no cache,
precomputed root outputs, async/broker/C search or weight splicing is used.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import chess
import numpy as np
import torch

from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.mcts.gumbel import GumbelConfig, run_gumbel_root_many


class OutputCrossover:
    """Select policy logits from one frozen model and STM WDL logits from another.

    WDL order is the production W/D/L contract, unchanged at every node; search
    performs its usual backpropagation perspective changes. No root-only flip.
    This class deliberately exposes only the synchronous evaluate_encoded seam.
    """

    def __init__(
        self,
        policy_model: torch.nn.Module,
        value_model: torch.nn.Module,
        *,
        batch_size: int = 16,
        max_rows: int = 4096,
    ):
        for name, value, high in (
            ("batch_size", batch_size, 64),
            ("max_rows", max_rows, 8192),
        ):
            if type(value) is not int or not 1 <= value <= high:
                raise ValueError(name)
        for model in (policy_model, value_model):
            if any(m.training for m in model.modules()):
                raise ValueError("model must be eval")
            for tensor in list(model.parameters()) + list(model.buffers()):
                if tensor.device.type != "cpu":
                    raise ValueError("CPU only")
                if tensor.is_floating_point() and tensor.dtype != torch.float32:
                    raise ValueError("float32 model required")
            if any(p.requires_grad for p in model.parameters()):
                raise ValueError("frozen parameters required")
        self._policy = LocalModelEvaluator(policy_model, device="cpu", use_amp=False)
        self._value = (
            self._policy
            if policy_model is value_model
            else LocalModelEvaluator(value_model, device="cpu", use_amp=False)
        )
        self.batch_size = batch_size
        self.max_rows = max_rows
        self.evaluated_rows = 0
        self.calls: list[dict[str, object]] = []

    def evaluate_encoded(
        self, x: np.ndarray, relations: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if relations is not None:
            raise ValueError("relations are outside this diagnostic contract")
        if (
            x.dtype != np.float32
            or x.ndim != 4
            or x.shape[1:] != (175, 8, 8)
            or not 1 <= len(x) <= 8192
        ):
            raise ValueError("stored175 float32 input required")
        if self.evaluated_rows + len(x) > self.max_rows:
            raise ValueError("finite evaluator budget exceeded")
        if not np.isfinite(x).all():
            raise ValueError("nonfinite input")
        snapshot = x.copy(order="C")
        self.evaluated_rows += len(snapshot)
        policy, value = [], []
        for start in range(0, len(snapshot), self.batch_size):
            chunk = snapshot[start : start + self.batch_size]
            p, own_v = self._policy.evaluate_encoded(chunk.copy())
            _, v = (
                (p, own_v)
                if self._value is self._policy
                else self._value.evaluate_encoded(chunk.copy())
            )
            if p.shape != (len(chunk), 4672) or v.shape != (len(chunk), 3):
                raise ValueError("policy/WDL shape mismatch")
            if not np.isfinite(p).all() or not np.isfinite(v).all():
                raise ValueError("nonfinite model logits")
            policy.append(p.copy())
            value.append(v.copy())
        self.calls.append(
            {
                "rows": len(snapshot),
                "input_sha256": hashlib.sha256(snapshot.tobytes()).hexdigest(),
            }
        )
        return np.concatenate(policy), np.concatenate(value)


def search_one(
    board: chess.Board, evaluator: OutputCrossover, *, cfg: GumbelConfig, seed: int
):
    """One fresh synchronous Python search; no reused tree or root caches.

    Same explicit cfg/seed must be supplied to all four cells. This is not a
    production C-route equivalence claim. Full replay history is caller-gated.
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed")
    if type(cfg.simulations) is not int or not 1 <= cfg.simulations <= 64:
        raise ValueError("simulations")
    if (
        cfg.input_history_encoding != "lc0_root_legacy_meta"
        or cfg.input_extra_features != "v2_threats"
        or cfg.compute_relations
        or cfg.volatility_q_scale != 0
        or cfg.volatility_fpu != 0
    ):
        raise ValueError("unsupported search encoding or auxiliary evaluator path")
    return run_gumbel_root_many(
        None,
        [board.copy(stack=True)],
        device="cpu",
        rng=np.random.default_rng(seed),
        cfg=replace(cfg),
        evaluator=evaluator,
    )
