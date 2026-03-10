from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.optim import Optimizer


def _zeropower_via_newton_schulz5(
    grad: Tensor,
    *,
    steps: int = 5,
    eps: float = 1e-7,
    work_dtype: torch.dtype | None = None,
) -> Tensor:
    if grad.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(grad.shape)}")

    a, b, c = (3.4445, -4.7750, 2.0315)
    if work_dtype is None:
        if grad.device.type != "cuda":
            work_dtype = grad.dtype if grad.dtype in (torch.float32, torch.float64) else torch.float32
        else:
            work_dtype = torch.bfloat16 if grad.dtype not in (torch.float16, torch.bfloat16) else grad.dtype

    x = grad.to(work_dtype)
    x = x / (x.norm() + eps)

    transposed = False
    if x.size(0) > x.size(1):
        x = x.transpose(0, 1)
        transposed = True

    for _ in range(max(1, int(steps))):
        xx_t = x @ x.transpose(0, 1)
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

    if transposed:
        x = x.transpose(0, 1)
    return x.to(dtype=grad.dtype)


def _nesterov_momentum(
    grad: Tensor,
    exp_avg: Tensor,
    beta1: float,
    step: int,
    *,
    use_nesterov: bool,
) -> Tensor:
    bias_correction1 = 1.0 - beta1 ** max(1, int(step))
    m_hat = exp_avg / max(bias_correction1, 1e-12)
    if not use_nesterov:
        return m_hat
    return beta1 * m_hat + (1.0 - beta1) * grad


def _init_orthobasis_from_grad(
    grad_f: Tensor,
    *,
    rank: int,
    power_iters: int,
) -> Tensor:
    cols = grad_f.size(1)
    basis = torch.randn(cols, rank, device=grad_f.device, dtype=grad_f.dtype)
    basis, _ = torch.linalg.qr(basis, mode="reduced")

    for _ in range(max(0, int(power_iters))):
        basis = grad_f.transpose(0, 1) @ (grad_f @ basis)
        basis, _ = torch.linalg.qr(basis, mode="reduced")

    return basis


class COSMOSFast(Optimizer):
    """Faster engineering variant of COSMOS with AdamW fallback groups.

    Param groups marked with ``use_cosmos_fast=True`` receive the low-rank +
    Newton-Schulz update on 2D hidden matrices. All other groups fall back to
    AdamW-style updates inside the same optimizer so Trainer scheduling stays
    unchanged.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        lr_ratio: float = 0.1,
        rank: int = 64,
        weight_decay: float = 0.0,
        gamma: float = 0.2,
        nesterov: bool = True,
        ns_steps: int = 5,
        proj_update_every: int = 4,
        proj_init_power_iters: int = 2,
        proj_dtype: torch.dtype = torch.float32,
        stat_dtype: torch.dtype = torch.float32,
        residual_work_dtype: torch.dtype | None = None,
        *,
        maximize: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if rank <= 0:
            raise ValueError(f"Invalid rank: {rank}")
        if proj_update_every <= 0:
            raise ValueError(f"proj_update_every must be >= 1, got {proj_update_every}")
        if proj_init_power_iters < 0:
            raise ValueError(f"proj_init_power_iters must be >= 0, got {proj_init_power_iters}")
        if ns_steps <= 0:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            use_cosmos_fast=False,
        )
        super().__init__(params, defaults)
        self.lr_ratio = float(lr_ratio)
        self.rank = int(rank)
        self.gamma = float(gamma)
        self.nesterov = bool(nesterov)
        self.ns_steps = int(ns_steps)
        self.proj_update_every = int(proj_update_every)
        self.proj_init_power_iters = int(proj_init_power_iters)
        self.proj_dtype = proj_dtype
        self.stat_dtype = stat_dtype
        self.residual_work_dtype = residual_work_dtype

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("use_cosmos_fast", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            beta1, beta2 = tuple(group["betas"])
            eps = float(group["eps"])
            maximize = bool(group.get("maximize", False))
            use_cosmos_fast = bool(group.get("use_cosmos_fast", False))

            if use_cosmos_fast:
                for param in group["params"]:
                    grad = param.grad
                    if grad is None:
                        continue
                    if grad.is_sparse:
                        raise RuntimeError("COSMOSFast does not support sparse gradients")
                    if param.ndim != 2:
                        raise RuntimeError("COSMOSFast groups must contain only 2D tensors")

                    grad = -grad if maximize else grad
                    rows, cols = param.shape
                    if self.rank > min(rows, cols):
                        raise ValueError(
                            f"rank={self.rank} is larger than min(shape)={min(rows, cols)} for {tuple(param.shape)}"
                        )

                    state = self.state[param]
                    if len(state) == 0:
                        state["step"] = 0
                        state["initialized"] = False
                        state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        state["P"] = torch.zeros(cols, self.rank, device=param.device, dtype=self.proj_dtype)
                        state["GG"] = torch.zeros(self.rank, self.rank, device=param.device, dtype=self.proj_dtype)
                        state["exp_avg_sq"] = torch.zeros(rows, self.rank, device=param.device, dtype=self.stat_dtype)

                    state["step"] += 1
                    step = int(state["step"])

                    exp_avg: Tensor = state["exp_avg"]
                    proj_basis: Tensor = state["P"]
                    gram_ema: Tensor = state["GG"]
                    exp_avg_sq: Tensor = state["exp_avg_sq"]

                    if weight_decay != 0.0:
                        param.mul_(1.0 - lr * weight_decay)

                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    momentum = _nesterov_momentum(
                        grad,
                        exp_avg,
                        beta1,
                        step,
                        use_nesterov=self.nesterov,
                    )

                    grad_f = grad.to(self.proj_dtype)
                    momentum_f = momentum.to(self.proj_dtype)

                    if not state["initialized"]:
                        proj_basis.copy_(
                            _init_orthobasis_from_grad(
                                grad_f,
                                rank=self.rank,
                                power_iters=self.proj_init_power_iters,
                            )
                        )
                        state["initialized"] = True
                    elif (step % self.proj_update_every) == 0:
                        gp_old = grad_f @ proj_basis
                        candidate = beta2 * (proj_basis @ gram_ema) + (1.0 - beta2) * (grad_f.transpose(0, 1) @ gp_old)
                        new_basis, _ = torch.linalg.qr(candidate, mode="reduced")
                        rot = proj_basis.transpose(0, 1) @ new_basis
                        gram_ema.copy_(rot.transpose(0, 1) @ gram_ema @ rot)
                        proj_basis.copy_(new_basis)

                    gp = grad_f @ proj_basis
                    gram_ema.mul_(beta2).addmm_(gp.transpose(0, 1), gp, beta=1.0, alpha=1.0 - beta2)

                    gp_stat = gp.to(self.stat_dtype)
                    exp_avg_sq.mul_(beta2).addcmul_(gp_stat, gp_stat, value=1.0 - beta2)

                    bias_correction2 = 1.0 - beta2 ** step
                    momentum_proj = momentum_f @ proj_basis
                    denom = exp_avg_sq.sqrt().div_(math.sqrt(max(bias_correction2, 1e-12))).add_(eps)
                    low_rank_dir = (momentum_proj / denom.to(momentum_proj.dtype)) @ proj_basis.transpose(0, 1)

                    residual = momentum_f - (momentum_proj @ proj_basis.transpose(0, 1))
                    residual = _zeropower_via_newton_schulz5(
                        residual,
                        steps=self.ns_steps,
                        eps=eps,
                        work_dtype=self.residual_work_dtype,
                    )

                    scale = math.sqrt(float(rows * cols))
                    residual = residual * (scale / (residual.norm() + eps))

                    update = low_rank_dir + self.gamma * residual
                    update = update * (scale / (update.norm() + eps))
                    param.add_(update.to(param.dtype), alpha=-(self.lr_ratio * lr))
                continue

            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("COSMOSFast AdamW fallback does not support sparse gradients")

                grad = -grad if maximize else grad
                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                state = self.state[param]
                step = int(state.get("step", 0)) + 1
                state["step"] = step

                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg is None:
                    exp_avg = torch.zeros_like(param, memory_format=torch.preserve_format)
                    exp_avg_sq = torch.zeros_like(param, memory_format=torch.preserve_format)
                    state["exp_avg"] = exp_avg
                    state["exp_avg_sq"] = exp_avg_sq

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denom = exp_avg_sq.sqrt().div_(math.sqrt(max(bias_correction2, 1e-12))).add_(eps)
                step_size = lr / max(bias_correction1, 1e-12)
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
