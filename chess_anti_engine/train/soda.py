from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


SODA_REGULARIZE_KEY = "soda_regularize"
SODA_REPLACED_WEIGHT_DECAY_KEY = "soda_replaced_weight_decay"
SODA_STEP_KEY = "soda_k"
SODA_START_STEP_KEY = "soda_start_step"


def mark_soda_weight_decay_groups(
    param_groups: list[dict[str, Any]],
    *,
    start_step: int = 0,
    force: bool = False,
) -> bool:
    """Replace nonzero param-group weight decay with SODA regularization.

    The SODA wrapper is intended as a weight-decay replacement, so only groups
    that already requested decay are marked by default. ``force=True`` is for
    experiments that explicitly select a group for SODA even when ordinary
    weight decay is zero.
    """
    any_marked = False
    for group in param_groups:
        weight_decay = float(group.get("weight_decay", 0.0))
        if weight_decay <= 0.0 and not force:
            group.setdefault(SODA_REGULARIZE_KEY, False)
            continue
        if not group.get("params"):
            group.setdefault(SODA_REGULARIZE_KEY, False)
            continue
        group[SODA_REGULARIZE_KEY] = True
        group[SODA_REPLACED_WEIGHT_DECAY_KEY] = weight_decay
        group["weight_decay"] = 0.0
        group.setdefault(SODA_STEP_KEY, 0)
        group[SODA_START_STEP_KEY] = max(0, int(start_step))
        any_marked = True
    return any_marked


class SODAWeightDecayWrapper(torch.optim.Optimizer):
    """Apply SODAWrapper-style averaging on top of a base optimizer.

    This is intentionally a narrow wrapper for experiments: the base optimizer
    owns parameter groups and optimizer state, while this wrapper adds the
    anchor-to-initial-weights step for groups marked with ``soda_regularize``.
    """

    def __init__(self, base: torch.optim.Optimizer) -> None:
        self.base = base
        self._soda_anchors: dict[Tensor, Tensor] = {}
        self._initializing = True
        params = [
            param
            for group in base.param_groups
            for param in group["params"]
        ]
        super().__init__(params, defaults={})
        self._initializing = False
        self.param_groups = self.base.param_groups
        self.state = self.base.state
        self._ensure_group_defaults()
        self._capture_initial_anchors()

    def _ensure_group_defaults(self) -> None:
        for group in self.base.param_groups:
            group.setdefault(SODA_REGULARIZE_KEY, False)
            if bool(group.get(SODA_REGULARIZE_KEY, False)):
                group.setdefault(SODA_STEP_KEY, 0)
                group.setdefault(SODA_START_STEP_KEY, 0)

    @torch.no_grad()
    def _capture_initial_anchors(self) -> None:
        for group in self.base.param_groups:
            if not bool(group.get(SODA_REGULARIZE_KEY, False)):
                continue
            for param in group["params"]:
                self._soda_anchors.setdefault(
                    param,
                    torch.clone(param, memory_format=torch.preserve_format),
                )

    @torch.no_grad()
    def reset_anchors(self) -> None:
        """Re-capture SODA anchors from the model's current weights.

        Model-only checkpoint/bootstrap loads happen after optimizer
        construction. Without this hook, SODA would pull toward the random
        construction-time weights instead of the loaded checkpoint.
        """
        self._soda_anchors.clear()
        for group in self.base.param_groups:
            if not bool(group.get(SODA_REGULARIZE_KEY, False)):
                continue
            for param in group["params"]:
                self._soda_anchors[param] = torch.clone(
                    param,
                    memory_format=torch.preserve_format,
                )

    @property
    def last_uw_stats(self) -> dict[str, float]:
        return getattr(self.base, "last_uw_stats", {})

    def add_param_group(self, param_group: dict[str, Any]) -> None:  # type: ignore[override]
        if getattr(self, "_initializing", False):
            torch.optim.Optimizer.add_param_group(self, param_group)
            return
        if bool(param_group.get(SODA_REGULARIZE_KEY, False)):
            param_group.setdefault(SODA_STEP_KEY, 0)
            param_group.setdefault(SODA_START_STEP_KEY, 0)
        self.base.add_param_group(param_group)
        self.param_groups = self.base.param_groups
        self.state = self.base.state
        self._capture_initial_anchors()

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        self.base.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:  # type: ignore[override]
        out = self.base.state_dict()
        params = [
            param
            for group in self.base.param_groups
            for param in group["params"]
        ]
        param_to_idx = {param: idx for idx, param in enumerate(params)}
        out["soda_anchors"] = {
            param_to_idx[param]: anchor
            for param, anchor in self._soda_anchors.items()
            if param in param_to_idx
        }
        return out

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # type: ignore[override]
        base_state = dict(state_dict)
        soda_anchors = dict(base_state.pop("soda_anchors", {}))
        self.base.load_state_dict(base_state)
        self.param_groups = self.base.param_groups
        self.state = self.base.state
        params = [
            param
            for group in self.base.param_groups
            for param in group["params"]
        ]
        self._soda_anchors = {
            params[int(idx)]: anchor.to(device=params[int(idx)].device, dtype=params[int(idx)].dtype)
            for idx, anchor in soda_anchors.items()
            if int(idx) < len(params)
        }
        self._ensure_group_defaults()
        self._capture_initial_anchors()

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        prev_by_group: list[tuple[dict[str, Any], int, dict[Tensor, Tensor]]] = []
        marked_groups: list[tuple[dict[str, Any], int]] = []
        for group in self.base.param_groups:
            if not bool(group.get(SODA_REGULARIZE_KEY, False)):
                continue
            k = int(group.get(SODA_STEP_KEY, 0))
            marked_groups.append((group, k))
            if k < int(group.get(SODA_START_STEP_KEY, 0)):
                continue
            prev_map: dict[Tensor, Tensor] = {}
            for param in group["params"]:
                if param.grad is None:
                    continue
                prev_map[param] = torch.clone(param, memory_format=torch.preserve_format)
            if prev_map:
                prev_by_group.append((group, k, prev_map))

        self.base.step()

        for group, k, prev_map in prev_by_group:
            beta = 1.0 / float(k + 2)
            for param, prev in prev_map.items():
                anchor = self._soda_anchors.get(param)
                if anchor is None:
                    anchor = torch.clone(prev, memory_format=torch.preserve_format)
                    self._soda_anchors[param] = anchor
                param.add_(anchor - prev, alpha=beta)

        for group, k in marked_groups:
            group[SODA_STEP_KEY] = k + 1

        return loss
