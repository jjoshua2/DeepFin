"""A seat whose search value is not its own net's raw WDL.

Two arena options build this (``scripts/arena_standard.py``), each per seat:

* ``--candidate-value-from CKPT`` (``--reference-value-from``): the seat's
  POLICY comes from its own checkpoint and its WDL from ``CKPT``. Both nets run
  a full forward on the same input batch. The checkpoints have separate trunks
  (a fine-tuned net and the base it came from share no weights after training),
  so one trunk forward cannot serve both heads; the cost is a second forward
  per evaluation (the head-swap bank's hybrid arms ran 2.4-3.5x the wall of a
  plain arm, without CUDA graphs).
* ``--candidate-value-recal T,b`` (``--reference-value-recal``): the seat's WDL
  logits (W, D, L order, side-to-move POV) become ``logits / T + [0, b, 0]``
  at every forward (root and leaves). The search softmaxes them exactly as it
  softmaxes raw logits, so ``p = softmax(logits / T + [0, b, 0])``. Alone,
  this costs no second forward. With ``value-from`` it applies to the
  value-from net's logits.

Everything downstream (the evaluator, the 1858 -> 4672 mapping, legal masking,
the C search, pairing, adjudication) is the arena's own code: the wrapper only
replaces the ``wdl`` entry of the forward's output dict.

This is the in-repo replacement for the head-swap bank's
``headswap_arena.py``, which monkeypatched the loader.
"""
from __future__ import annotations

import math
from typing import Any, cast

import torch
from torch import nn

from chess_anti_engine.eval.seat_value import SeatValue

# Attributes the arena, ``pick_moves_for_boards`` and the evaluator read off the
# model. The policy net and the value net must agree on every one of them, or
# the two forwards would see different inputs and the hybrid is ill-defined.
SHARED_ATTRS = (
    "input_history_encoding",
    "input_extra_features",
    "policy_encoding",
    "use_dynamic_relations",
    "history_rep_fix",
)


class HybridValueNet(nn.Module):
    """``policy_net``'s outputs with ``wdl`` replaced by ``value_net``'s, recalibrated.

    ``value_net=None`` keeps the policy net's own WDL (one forward) and only
    recalibrates it. The recalibration bias is a registered buffer so the
    forward does no host-to-device copy, which keeps it capturable by the
    evaluator's CUDA graphs.
    """

    def __init__(self, policy_net: nn.Module, value_net: nn.Module | None, *,
                 recal: tuple[float, float] | None = None) -> None:
        super().__init__()
        if value_net is None and recal is None:
            raise ValueError("HybridValueNet needs a value net, a recalibration, or both")
        self.policy_net = policy_net
        self.value_net = value_net
        self.recal_t: float | None = None
        bias = 0.0
        if recal is not None:
            t, bias = float(recal[0]), float(recal[1])
            if not (math.isfinite(t) and t > 0 and math.isfinite(bias)):
                raise ValueError(f"bad recalibration T={t} b={bias}")
            self.recal_t = t
        self.register_buffer("recal_bias", torch.tensor([0.0, bias, 0.0]), persistent=False)
        for attr in SHARED_ATTRS:
            p_val = getattr(policy_net, attr, "<absent>")
            if value_net is not None:
                v_val = getattr(value_net, attr, "<absent>")
                if str(p_val) != str(v_val):
                    raise ValueError(
                        f"the policy net and the value net disagree on {attr}: {p_val!r} vs {v_val!r}; "
                        "their forwards would read different inputs")
            if p_val != "<absent>":
                object.__setattr__(self, attr, p_val)

    # The arena sets ``_inference_only`` (policy_own + wdl) on each model it
    # loads; forward it to both nets so each runs exactly the forward a plain
    # seat would run.
    @property
    def _inference_only(self) -> bool:
        p = bool(getattr(self.policy_net, "_inference_only", False))
        if self.value_net is not None and bool(getattr(self.value_net, "_inference_only", False)) != p:
            raise RuntimeError("HybridValueNet: _inference_only diverged between the two nets")
        return p

    @_inference_only.setter
    def _inference_only(self, value: bool) -> None:
        for net in (self.policy_net, self.value_net):
            if net is not None and hasattr(net, "_inference_only"):
                setattr(net, "_inference_only", bool(value))

    def forward(self, x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        out = dict(self.policy_net(x, **kwargs))
        wdl = out["wdl"] if self.value_net is None else self.value_net(x, **kwargs)["wdl"]
        if self.recal_t is not None:
            wdl = wdl.float() / self.recal_t + cast("torch.Tensor", self.recal_bias)
        out["wdl"] = wdl
        return out


def build_seat_model(policy_net: nn.Module, seat: SeatValue, *, device: str,
                     load: Any) -> nn.Module:
    """The model a seat plays: the plain net, or the hybrid around it.

    ``load(path, device=...)`` loads the value-from checkpoint (the arena passes
    its own loader, so both nets load through the same code).
    """
    if not seat.active:
        return policy_net
    value_net = None if seat.value_from is None else load(seat.value_from, device=device)
    return HybridValueNet(policy_net, value_net, recal=seat.recal).to(device).eval()
