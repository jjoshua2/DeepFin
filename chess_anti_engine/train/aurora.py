from __future__ import annotations

import math
from collections.abc import Callable
from itertools import repeat

import torch
from torch import Tensor

_POLAR_EXPRESS_COEFFS_RAW = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),
)


def _resolve_polar_dtype(dtype_name: str | torch.dtype | None) -> torch.dtype | None:
    if dtype_name is None:
        return None
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    name = str(dtype_name).lower()
    if name in ("auto", "", "none"):
        return None
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported Aurora polar dtype {dtype_name!r}")


def _polar_express_coeffs(steps: int, safety: float) -> tuple[tuple[float, float, float], ...]:
    if steps < 1:
        raise ValueError(f"polar steps must be >= 1, got {steps}")
    if safety <= 0.0:
        raise ValueError(f"polar safety must be positive, got {safety}")
    coeffs = [
        (a / safety, b / safety**3, c / safety**5)
        for a, b, c in _POLAR_EXPRESS_COEFFS_RAW[:-1]
    ] + [_POLAR_EXPRESS_COEFFS_RAW[-1]]
    if steps <= len(coeffs):
        return tuple(coeffs[:steps])
    return tuple(coeffs + list(repeat(coeffs[-1], steps - len(coeffs))))


def _polar_quintic(
    mat: Tensor,
    *,
    steps: int = 12,
    eps: float = 1e-7,
    work_dtype: torch.dtype | None = None,
) -> Tensor:
    """Approximate the polar factor with the simple-quintic Newton-Schulz iteration."""
    if mat.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(mat.shape)}")

    if work_dtype is None:
        work_dtype = torch.bfloat16 if mat.device.type == "cuda" else torch.float32

    x = mat.to(work_dtype)
    transposed = False
    if x.size(0) > x.size(1):
        x = x.transpose(0, 1)
        transposed = True

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    a, b, c = 2.0, -1.5, 0.5
    for _ in range(max(1, int(steps))):
        xx_t = x @ x.transpose(0, 1)
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.transpose(0, 1)
    return x.to(mat.dtype)


def _polar_express(
    mat: Tensor,
    *,
    steps: int = 8,
    eps: float = 1e-7,
    safety: float = 1.01,
    work_dtype: torch.dtype | None = torch.float16,
) -> Tensor:
    """Approximate the polar factor with Polar Express quintic coefficients."""
    if mat.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(mat.shape)}")

    if work_dtype is None:
        work_dtype = torch.float16 if mat.device.type == "cuda" else torch.float32
    elif mat.device.type != "cuda" and work_dtype in (torch.float16, torch.bfloat16):
        work_dtype = torch.float32

    x = mat.to(torch.float32)
    transposed = False
    if x.size(0) > x.size(1):
        x = x.transpose(0, 1)
        transposed = True

    x = x / (x.norm(dim=(-2, -1), keepdim=True) * float(safety) + eps)
    x = x.to(work_dtype)
    for a, b, c in _polar_express_coeffs(max(1, int(steps)), float(safety)):
        xx_t = x @ x.transpose(0, 1)
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.transpose(0, 1)
    return x.to(mat.dtype)


def _polar_factor(
    mat: Tensor,
    *,
    method: str,
    steps: int,
    eps: float,
    safety: float,
    work_dtype: torch.dtype | None,
) -> Tensor:
    method = str(method).lower()
    if method in ("simple", "simple_quintic", "quintic"):
        return _polar_quintic(mat, steps=steps, eps=eps, work_dtype=work_dtype)
    if method in ("polar_express", "pe", "pe8"):
        return _polar_express(mat, steps=steps, eps=eps, safety=safety, work_dtype=work_dtype)
    raise ValueError(f"Unknown Aurora polar method {method!r}. Supported: simple, polar_express")


def _aurora_update(
    update: Tensor,
    *,
    pp_iterations: int = 2,
    pp_beta: float = 0.5,
    polar_steps: int = 12,
    polar_method: str = "simple",
    polar_dtype: str | torch.dtype | None = None,
    polar_safety: float = 1.01,
    eps: float = 1e-7,
) -> Tensor:
    """Leverage-uniform polar update for a 2D matrix."""
    if update.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(update.shape)}")
    if pp_iterations < 1:
        raise ValueError(f"pp_iterations must be >= 1, got {pp_iterations}")
    if pp_beta <= 0.0:
        raise ValueError(f"pp_beta must be positive, got {pp_beta}")

    orig_rows, orig_cols = int(update.size(0)), int(update.size(1))
    work_dtype = _resolve_polar_dtype(polar_dtype)
    if orig_rows == orig_cols:
        out = _polar_factor(
            update,
            method=polar_method,
            steps=polar_steps,
            eps=eps,
            safety=polar_safety,
            work_dtype=work_dtype,
        )
    else:
        transposed = orig_rows < orig_cols
        tall = update.transpose(0, 1) if transposed else update
        rows, cols = int(tall.size(0)), int(tall.size(1))
        tall_f = tall.to(torch.float32)

        target_row_sq = float(cols) / float(rows)
        row_norm = tall_f.norm(dim=-1, keepdim=True).clamp_min(eps)
        scale = 1.0 / row_norm

        out_f: Tensor = tall_f
        for k in range(max(1, int(pp_iterations))):
            out_f = _polar_factor(
                scale * tall_f,
                method=polar_method,
                steps=polar_steps,
                eps=eps,
                safety=polar_safety,
                work_dtype=work_dtype,
            ).to(torch.float32)
            if k < int(pp_iterations) - 1:
                row_sq = out_f.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps * eps)
                scale = scale * (target_row_sq / row_sq).pow(float(pp_beta))

        out = out_f.transpose(0, 1) if transposed else out_f

    out = out * math.sqrt(max(1.0, float(orig_rows) / max(1.0, float(orig_cols))))
    if not torch.isfinite(out).all():
        raise RuntimeError(
            f"Aurora produced non-finite update for matrix shape {(orig_rows, orig_cols)}"
        )
    return out.to(update.dtype)


def _uw_stats(ratios: list[Tensor], scales: list[Tensor], *, lr: float, floor: float) -> dict[str, float]:
    """Summarize Aurora update/weight ratios from one optimizer step."""
    if not ratios:
        return {}
    ratio = torch.stack([r.float().reshape(()) for r in ratios])
    scale = torch.stack([s.float().reshape(()) for s in scales])
    effective = ratio * scale * float(lr)
    floored = scale > 1.000001
    return {
        "aurora_uw_floor": float(floor),
        "aurora_uw_count": float(ratio.numel()),
        "aurora_uw_ratio_min": float(ratio.min().item()),
        "aurora_uw_ratio_p10": float(torch.quantile(ratio, 0.10).item()),
        "aurora_uw_ratio_median": float(torch.quantile(ratio, 0.50).item()),
        "aurora_uw_ratio_p90": float(torch.quantile(ratio, 0.90).item()),
        "aurora_uw_scale_max": float(scale.max().item()),
        "aurora_uw_floored_frac": float(floored.float().mean().item()),
        "aurora_uw_effective_ratio_min": float(effective.min().item()),
        "aurora_uw_effective_ratio_median": float(torch.quantile(effective, 0.50).item()),
    }


class AuroraWithAuxAdam(torch.optim.Optimizer):
    """Aurora for 2D hidden weights, AdamW for auxiliary tensors.

    Aurora is a Muon-style matrix optimizer, so norms, biases, embeddings not
    selected by the caller, and output heads stay on AdamW fallback groups.
    """

    def __init__(
        self,
        params,
        *,
        aurora_momentum: float = 0.95,
        aurora_nesterov: bool = True,
        aurora_pp_iterations: int = 2,
        aurora_pp_beta: float = 0.5,
        aurora_polar_steps: int = 12,
        aurora_polar_method: str = "simple",
        aurora_polar_dtype: str | torch.dtype | None = None,
        aurora_polar_safety: float = 1.01,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
    ) -> None:
        defaults = {
            "lr": 0.0,
            "weight_decay": 0.0,
            "use_aurora": False,
            "aurora_uw_floor": 0.0,
            "betas": adam_betas,
            "eps": float(adam_eps),
            "aurora_momentum": float(aurora_momentum),
            "aurora_nesterov": bool(aurora_nesterov),
            "aurora_pp_iterations": int(aurora_pp_iterations),
            "aurora_pp_beta": float(aurora_pp_beta),
            "aurora_polar_steps": int(aurora_polar_steps),
            "aurora_polar_method": str(aurora_polar_method),
            "aurora_polar_dtype": "auto" if aurora_polar_dtype is None else aurora_polar_dtype,
            "aurora_polar_safety": float(aurora_polar_safety),
        }
        super().__init__(params, defaults)
        self.last_uw_stats: dict[str, float] = {}

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]  # pyright: ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", 0.0))
            use_aurora = bool(group.get("use_aurora", False))

            if use_aurora:
                uw_ratios: list[Tensor] = []
                uw_scales: list[Tensor] = []
                momentum = float(group.get("aurora_momentum", 0.95))
                nesterov = bool(group.get("aurora_nesterov", True))
                pp_iterations = int(group.get("aurora_pp_iterations", 2))
                pp_beta = float(group.get("aurora_pp_beta", 0.5))
                polar_steps = int(group.get("aurora_polar_steps", 12))
                polar_method = str(group.get("aurora_polar_method", "simple"))
                polar_dtype = group.get("aurora_polar_dtype", None)
                polar_safety = float(group.get("aurora_polar_safety", 1.01))
                uw_floor = float(group.get("aurora_uw_floor", 0.0))
                eps = float(group.get("eps", 1e-8))
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    grad = param.grad.detach()
                    if grad.is_sparse:
                        raise RuntimeError("Aurora does not support sparse gradients")
                    if param.ndim != 2:
                        raise RuntimeError("Aurora groups must contain only 2D tensors")
                    if weight_decay != 0.0:
                        param.mul_(1.0 - lr * weight_decay)

                    state = self.state[param]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(grad)
                        state["momentum_buffer"] = buf
                    buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)
                    update = torch.lerp(grad, buf, momentum) if nesterov else buf
                    update = _aurora_update(
                        update,
                        pp_iterations=pp_iterations,
                        pp_beta=pp_beta,
                        polar_steps=polar_steps,
                        polar_method=polar_method,
                        polar_dtype=polar_dtype,
                        polar_safety=polar_safety,
                        eps=eps,
                    )
                    weight_fro = param.float().norm().clamp_min(eps)
                    update_fro = update.float().norm().clamp_min(eps)
                    uw_ratio = update_fro / weight_fro
                    scale = torch.ones((), dtype=torch.float32, device=update.device)
                    if uw_floor > 0.0:
                        ratio_ok = torch.isfinite(uw_ratio) & (uw_ratio > 0.0)
                        floor = torch.as_tensor(float(uw_floor), dtype=torch.float32, device=update.device)
                        safe_ratio = torch.where(ratio_ok, uw_ratio, torch.ones_like(uw_ratio))
                        scale = torch.where(
                            ratio_ok,
                            torch.clamp_min(floor / safe_ratio, 1.0),
                            scale,
                        )
                        update = update * scale.to(update.dtype)
                    uw_ratios.append(uw_ratio.detach())
                    uw_scales.append(scale.detach())
                    param.add_(update, alpha=-lr)
                self.last_uw_stats = _uw_stats(uw_ratios, uw_scales, lr=lr, floor=uw_floor)
                continue

            beta1, beta2 = tuple(group.get("betas", (0.9, 0.95)))
            eps = float(group.get("eps", 1e-8))
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("Aurora AdamW fallback does not support sparse gradients")
                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                state = self.state[param]
                step = int(state.get("step", 0)) + 1
                state["step"] = step

                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg is None:
                    exp_avg = torch.zeros_like(grad)
                    exp_avg_sq = torch.zeros_like(grad)
                    state["exp_avg"] = exp_avg
                    state["exp_avg_sq"] = exp_avg_sq

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                denom = exp_avg_sq.sqrt() / math.sqrt(max(bias_correction2, 1e-12))
                denom.add_(eps)
                step_size = lr / max(bias_correction1, 1e-12)
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
