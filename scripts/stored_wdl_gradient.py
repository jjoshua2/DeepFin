"""Synthetic-qualified local shared-value-body gradient compatibility primitive.

Not a training step, trajectory explanation, full-model gradient or safe loader.
Real use requires separately admitted receipts and a reviewed bounded child.
"""

from __future__ import annotations
import math
import torch
from chess_anti_engine.model.transformer import ChessNet
from chess_anti_engine.train.losses import soft_cross_entropy, _normalize_sf_wdl_probs

SUBSET = (
    "value_wdl.token_proj.weight",
    "value_wdl.token_proj.bias",
    "value_wdl.net.0.weight",
    "value_wdl.net.0.bias",
)


def gradient_geometry(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    if a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("finite matched gradients required")
    dot = float(torch.dot(a, b))
    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    return {
        "dot": dot,
        "wdl_norm": na,
        "categorical_norm": nb,
        "cosine": dot / (na * nb) if na and nb else None,
        "categorical_projection_on_wdl": dot / na if na else None,
        "interpretation": "Local shared-body Euclidean gradients only; no causal training or strength claim",
    }


def gradient_probe(
    model: torch.nn.Module,
    x: torch.Tensor,
    search_targets: torch.Tensor,
    categorical_targets: torch.Tensor,
    network_mask: torch.Tensor,
    categorical_mask: torch.Tensor,
    *,
    batch_size: int = 16,
    w_wdl: float = 1.0,
    w_categorical: float = 1.0,
) -> dict:
    """Uniform fixed-panel means; not unknown realized replay/sample weights.

    Reuse one detached production trunk evaluation per chunk. Only the four
    shared-body parameters participate in backward; private output projections
    are frozen. autograd.grad never accumulates parameter .grad or updates.
    Caller owns exact checkpoint/rows/target identity and process limits.
    """
    if not isinstance(model, ChessNet) or model.value_categorical_coupled is None:
        raise ValueError("actual coupled transformer required")
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("bounded exact batch size required")
    if not all(math.isfinite(float(w)) for w in (w_wdl, w_categorical)):
        raise ValueError("finite diagnostic weights required")
    tensors = [
        t.detach().clone()
        for t in (
            x,
            search_targets,
            categorical_targets,
            network_mask,
            categorical_mask,
        )
    ]
    x, search_targets, categorical_targets, network_mask, categorical_mask = tensors
    n = len(x)
    if not 1 <= n <= 256 or x.shape != (n, 175, 8, 8):
        raise ValueError("bounded stored175 rows required")
    if any(
        t.device.type != "cpu"
        or t.dtype != torch.float32
        or not torch.isfinite(t).all()
        for t in tensors
    ):
        raise ValueError("finite CPU float32 snapshots required")
    if search_targets.shape != (n, 3) or categorical_targets.shape != (n, 32):
        raise ValueError("exact target shapes required")
    if any(
        t.shape != (n,) or not torch.all((t == 0) | (t == 1))
        for t in (network_mask, categorical_mask)
    ):
        raise ValueError("exact binary presence masks required")
    for t, present in (
        (search_targets, torch.ones(n)),
        (categorical_targets, categorical_mask),
    ):
        sums = t.sum(1)
        valid = (torch.abs(sums - 1) <= 0.001) | ((present == 0) & (sums == 0))
        if (t < 0).any() or (t > 1).any() or not valid.all():
            raise ValueError(
                "admitted nonnegative near-simplex or absent-zero targets required"
            )
    named = dict(model.named_parameters())
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in named.values()):
        raise ValueError("CPU float32 model required")
    params = tuple(named[k] for k in SUBSET)
    flags = {k: p.requires_grad for k, p in named.items()}
    modes = {module: module.training for module in model.modules()}
    before = tuple(p.detach().clone() for p in params)
    g1 = [torch.zeros_like(p, dtype=torch.float64) for p in params]
    g2 = [torch.zeros_like(p, dtype=torch.float64) for p in params]
    netden = network_mask.sum().clamp_min(1)
    catmask = network_mask * categorical_mask
    catden = catmask.sum().clamp_min(1)
    losses = [0.0, 0.0]
    try:
        model.eval().requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        for start in range(0, n, batch_size):
            end = min(n, start + batch_size)
            with torch.no_grad():
                tokens, _ = model._encode_tokens(x[start:end])
            hidden = model.value_wdl.hidden(tokens.detach())
            logits_w = model.value_wdl.head_from_hidden(hidden)
            logits_c = model.value_categorical_coupled(hidden)
            target = _normalize_sf_wdl_probs(search_targets[start:end])
            assert target is not None
            lw = (
                float(w_wdl)
                * (soft_cross_entropy(logits_w, target) * network_mask[start:end]).sum()
                / netden
            )
            lc = (
                float(w_categorical)
                * (
                    soft_cross_entropy(logits_c, categorical_targets[start:end])
                    * catmask[start:end]
                ).sum()
                / catden
            )
            a = torch.autograd.grad(lw, params, retain_graph=True)
            b = torch.autograd.grad(lc, params)
            for total, grad in zip(g1, a, strict=True):
                total.add_(grad.detach().double())
            for total, grad in zip(g2, b, strict=True):
                total.add_(grad.detach().double())
            losses[0] += float(lw.detach())
            losses[1] += float(lc.detach())
        if any(
            not torch.equal(p.detach(), old)
            for p, old in zip(params, before, strict=True)
        ):
            raise ValueError("parameter changed during diagnostic")
    finally:
        for k, p in named.items():
            p.requires_grad_(flags[k])
        for module, mode in modes.items():
            module.training = mode
    a = torch.cat([g.flatten() for g in g1])
    b = torch.cat([g.flatten() for g in g2])
    return {
        "geometry": gradient_geometry(a, b),
        "wdl_gradient": a,
        "categorical_gradient": b,
        "weighted_wdl_loss": losses[0],
        "weighted_categorical_loss": losses[1],
        "parameter_subset": list(SUBSET),
        "rows": n,
        "network_rows": int(network_mask.sum()),
        "categorical_rows": int(catmask.sum()),
        "batch_size": batch_size,
        "weights": {"wdl": w_wdl, "categorical": w_categorical},
        "weighting": "uniform fixed-panel masked means; not realized training sampling",
        "trunk_backward": False,
        "optimizer_step": False,
    }
