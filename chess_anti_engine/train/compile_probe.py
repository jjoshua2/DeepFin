"""Trainer torch.compile diagnostics.

The trainer wraps the model with ``torch.compile`` when ``use_compile`` is
truthy, but compile failures (graph capture rejecting dynamic shapes,
cudagraph fallbacks) and recompile thrash are normally invisible — the
training loop still produces gradients, just without the speedup.

This module:

* ``apply_compile`` — wraps a model with ``torch.compile``, logs whether
  the wrapper actually attached, and supports ``"default"``,
  ``"reduce-overhead"``, ``"max-autotune"``, ``"off"``. Failures are
  logged loudly instead of silently swallowed.
* ``CompileProbe`` — a context-window probe. Snapshot the dynamo counters
  before training begins, then after N steps; the diff tells you how many
  graphs were captured, how many recompiles fired, and whether the
  trainer is in a recompile loop.
"""
from __future__ import annotations

import logging
from typing import cast

import torch

logger = logging.getLogger(__name__)

VALID_MODES = (
    "off",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",  # autotune kernels but skip CUDA graphs — useful when graphs reject dynamic shapes
)


def apply_compile(model: torch.nn.Module, *, mode: str, device: str) -> torch.nn.Module:
    """Wrap ``model`` with torch.compile per ``mode``; log outcome.

    Returns the (possibly wrapped) model. Never raises — a failed compile
    falls back to eager and the failure is logged at WARNING.
    """
    mode = (mode or "off").strip().lower()
    if mode not in VALID_MODES:
        logger.warning("compile mode %r unknown; expected one of %s — skipping compile", mode, VALID_MODES)
        return model
    if mode == "off":
        logger.info("torch.compile: disabled (mode=off)")
        return model
    if not device.startswith("cuda"):
        logger.info("torch.compile: skipped (device=%s, compile only useful on CUDA)", device)
        return model
    try:
        wrapped = cast("torch.nn.Module", torch.compile(model, mode=mode))
    except Exception as exc:  # noqa: BLE001 — last-resort fallback path
        logger.warning("torch.compile failed (mode=%s): %s — falling back to eager", mode, exc)
        return model
    if not hasattr(wrapped, "_orig_mod"):
        # torch.compile returned, but didn't install the OptimizedModule
        # wrapper. That means we silently got eager back.
        logger.warning("torch.compile(mode=%s) returned but no _orig_mod — eager fallback", mode)
        return wrapped
    logger.info("torch.compile: enabled (mode=%s)", mode)
    return wrapped


class CompileProbe:
    """Snapshot dynamo counters across a window of training steps.

    Usage::

        probe = CompileProbe()
        probe.snapshot_baseline()
        # ... run a few steps ...
        probe.report(step_count=10)

    Reports counts of graph captures, recompiles, fallbacks, and a
    "recompile loop" flag if recompiles ≥ step_count / 2 (suggests every
    step is recompiling, defeating the speedup).
    """

    def __init__(self) -> None:
        self._baseline: dict[str, dict[str, int]] | None = None

    def snapshot_baseline(self) -> None:
        try:
            from torch._dynamo.utils import counters
        except Exception:
            self._baseline = None
            return
        # Counters are nested {category: {key: count}} dicts of int.
        self._baseline = {
            cat: dict(d) for cat, d in counters.items()
        }

    def report(self, *, step_count: int) -> None:
        try:
            from torch._dynamo.utils import counters
        except Exception:
            logger.info("torch.compile probe: torch._dynamo not available")
            return
        baseline = self._baseline
        if baseline is None:
            logger.info("torch.compile probe: no baseline (counters unavailable at start)")
            return

        def _delta(cat: str, key: str) -> int:
            now = int(counters.get(cat, {}).get(key, 0))
            base = int(baseline.get(cat, {}).get(key, 0))
            return max(0, now - base)

        captures = _delta("frames", "ok")
        recompiles = _delta("recompiles", "recompile")
        all_breaks = sum(_delta("graph_break", k) for k in counters.get("graph_break", {}))
  # cudagraphify_called > 0 + cudagraph_skips == 0 → cudagraphs actually
  # firing. cudagraph_skips > 0 means inductor compiled but the cudagraph
  # layer rejected (dynamic shapes, cross-thread TLS miss, etc) and we
  # silently fell back to eager replay.
        cg_called = _delta("inductor", "cudagraphify_called")
        cg_skips = _delta("inductor", "cudagraph_skips")
        msg = (
            f"torch.compile probe ({step_count} steps): "
            f"frames_ok={captures} recompiles={recompiles} graph_breaks={all_breaks} "
            f"cudagraphify_called={cg_called} cudagraph_skips={cg_skips}"
        )
        if recompiles >= max(2, step_count // 2):
            logger.warning("%s — RECOMPILE LOOP (every other step or more)", msg)
        elif captures == 0 and step_count > 0:
            logger.warning("%s — no graphs captured; running eager", msg)
        else:
            logger.info(msg)
