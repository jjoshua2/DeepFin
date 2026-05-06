from __future__ import annotations

import logging
from typing import cast

import torch

from chess_anti_engine.inference import (
    AOTEvaluator,
    DirectGPUEvaluator,
    ThreadedBatchEvaluator,
)
from chess_anti_engine.inference_threaded import ThreadedDispatcher


def maybe_apply_fp8_inference(model: torch.nn.Module) -> torch.nn.Module:
    """Quantize eligible Linear layers to FP8 (dynamic activation, FP8 weight).

    Skips layers where FP8 hurts accuracy or speed: Smolgen (small batch dim),
    non-16-aligned dims (tensor core requirement), tiny output heads.
    Requires torch.compile afterwards for actual speedup.
    """
    try:
        from torchao.quantization import (  # pyright: ignore[reportMissingImports]  # optional dep
            Float8DynamicActivationFloat8WeightConfig,
            PerRow,
            quantize_,
        )
    except ImportError:
        return model

    def _fp8_filter(mod: torch.nn.Module, fqn: str) -> bool:
        if not isinstance(mod, torch.nn.Linear):
            return False
        if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
            return False
        if "smolgen" in fqn:
            return False
        if fqn.endswith(".net.2") and mod.out_features <= 32:
            return False
        return True

    try:
        quantize_(
            model,
            Float8DynamicActivationFloat8WeightConfig(granularity=PerRow()),
            filter_fn=_fp8_filter,
        )
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "FP8 quantization failed, continuing with BF16: %s", exc,
        )
    return model


def maybe_compile_inference_model(
    model: torch.nn.Module,
    *,
    device: str,
    mode: str = "reduce-overhead",
    use_fp8: bool = False,
) -> torch.nn.Module:
    if not str(device).startswith("cuda"):
        return model
    if use_fp8:
        model = maybe_apply_fp8_inference(model)
    try:
        return cast("torch.nn.Module", torch.compile(model, mode=mode))
    except Exception:
        return model


def sync_evaluator_to_model(
    evaluator: DirectGPUEvaluator | ThreadedBatchEvaluator | ThreadedDispatcher | AOTEvaluator,
    model: torch.nn.Module,
) -> None:
    if isinstance(evaluator, (ThreadedBatchEvaluator, ThreadedDispatcher)):
        evaluator.update_model(model)
    elif isinstance(evaluator, AOTEvaluator):
        evaluator.load_weights(model.state_dict())
    else:
        evaluator.model = model
