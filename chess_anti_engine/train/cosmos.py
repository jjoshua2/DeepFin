"""COSMOS: A Hybrid Adaptive Optimizer for Memory-Efficient Training of LLMs.

Source: https://github.com/lliu606/COSMOS
Paper: "COSMOS: A Hybrid Adaptive Optimizer for Memory-Efficient Training of LLMs"

For 2D weight matrices with size(0) <= 10000, COSMOS uses a low-rank subspace
approach combining Adam-style updates in the subspace with Muon/Newton-Schulz
orthogonalization for the orthogonal complement. Falls back to AdamW for all
other tensors (embeddings, biases, large matrices).
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.optim import Optimizer


def _zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


# Defer torch.compile to first call to avoid triggering CUDA init at import time
# (segfaults in environments where CUDA driver is broken, e.g. WSL2).
_zeropower_compiled: None | object = None


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    global _zeropower_compiled
    if _zeropower_compiled is None:
        try:
            _zeropower_compiled = torch.compile(_zeropower_via_newtonschulz5)
        except Exception:
            _zeropower_compiled = _zeropower_via_newtonschulz5
    return _zeropower_compiled(G, steps=steps, eps=eps)


def cosmos(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    exp_avgs_GG: list[Tensor],
    exp_avgs_P: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[int],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    ratio: float,
    gamma: float,
    nesterov: bool,
) -> None:
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_gg = exp_avgs_GG[i]
        exp_avg_p = exp_avgs_P[i]

        step = state_steps[i]

        bias_correction1 = (1 - beta1 ** step) / (1 - beta1)
        bias_correction2 = 1 - beta2 ** step

        if len(param.size()) == 2 and param.size(0) <= 10000:
            exp_avg.mul_(beta1).add_(grad)

            if step == 1:
                W = torch.matmul(grad.T, grad)
                U, _, _ = torch.linalg.svd(W, full_matrices=False)
                exp_avg_p.data = U[:, :exp_avg_gg.size(0)]
                exp_avg_gg = torch.matmul(
                    torch.matmul(exp_avg_p.T, grad.T),
                    torch.matmul(grad, exp_avg_p),
                ) * (1 - beta2)
            else:
                t = exp_avg_p.T
                exp_avg_p = (
                    beta2 * torch.matmul(exp_avg_p, exp_avg_gg)
                    + (1 - beta2) * torch.matmul(grad.T, torch.matmul(grad, exp_avg_p))
                )
                exp_avg_p, _ = torch.linalg.qr(exp_avg_p, mode='reduced')
                t = torch.matmul(t, exp_avg_p)
                exp_avg_gg = beta2 * torch.matmul(t.T, torch.matmul(exp_avg_gg, t)) + (
                    1 - beta2
                ) * torch.matmul(
                    torch.matmul(grad, exp_avg_p).T, torch.matmul(grad, exp_avg_p)
                )

            scale = (grad.size(0) * grad.size(1)) ** 0.5
            low_rank_grad = torch.matmul(grad, exp_avg_p)
            exp_avg_sq.mul_(beta2).addcmul_(low_rank_grad, low_rank_grad.conj(), value=1 - beta2)

            if nesterov:
                grad.add_(exp_avg, alpha=beta1)
                grad.mul_(1 / (1 + beta1 * bias_correction1))
            else:
                grad.mul_(1 / bias_correction1)

            t = torch.matmul(grad, exp_avg_p)
            t1 = t / ((exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps))
            t1 = torch.matmul(t1, exp_avg_p.T)

            t = grad - torch.matmul(t, exp_avg_p.T)
            t = zeropower_via_newtonschulz5(t, steps=5)
            t = scale * t / (t.norm() + eps)

            t1.add_(t, alpha=gamma)

            if weight_decay > 0:
                param.data.mul_(1 - lr * weight_decay)

            param.add_(t1 / (t1.norm() + eps), alpha=-scale * ratio * lr)

            exp_avgs_P[i].copy_(exp_avg_p)
            exp_avgs_GG[i].copy_(exp_avg_gg)

        else:
            # AdamW fallback for 1D params, biases, embeddings, large matrices.
            bias_correction1_adam = 1 - beta1 ** step
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

            if amsgrad:
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1_adam

            if weight_decay > 0:
                param.data.mul_(1 - lr * weight_decay)

            param.addcdiv_(exp_avg, denom, value=-step_size)


class COSMOS(Optimizer):
    """COSMOS hybrid optimizer.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 1e-3).
        betas: Coefficients for computing running averages (default: (0.9, 0.999)).
        eps: Numerical stability term (default: 1e-8).
        lr_ratio: Scales the final update magnitude for 2D tensors (default: 0.1).
        rank: Low-rank subspace dimension (default: 64).
        weight_decay: Weight decay coefficient (default: 0).
        gamma: Weight of the Muon (Newton-Schulz) component (default: 0.2).
        nesterov: Use Nesterov momentum (default: True).
        amsgrad: Use AMSGrad variant (default: False).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        lr_ratio: float = 0.1,
        rank: int = 64,
        weight_decay: float = 0,
        gamma: float = 0.2,
        nesterov: bool = True,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
    ):
        if not lr >= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not eps >= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
        super().__init__(params, defaults)
        self.lr_ratio = lr_ratio
        self.rank = rank
        self.nesterov = nesterov
        self.gamma = gamma

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            exp_avgs_GG = []
            exp_avgs_P = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('COSMOS does not support sparse gradients')
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if len(p.size()) == 2 and p.size(0) <= 10000:
                        eff_rank = min(int(self.rank), int(p.size(0)), int(p.size(1)))
                        state['exp_avg_GG'] = torch.zeros(eff_rank, eff_rank, dtype=p.dtype, device=p.device)
                        state['exp_avg_P'] = torch.zeros(p.size(1), eff_rank, dtype=p.dtype, device=p.device)
                        state['exp_avg_sq'] = torch.zeros(p.size(0), eff_rank, dtype=p.dtype, device=p.device)
                    else:
                        state['exp_avg_GG'] = torch.zeros(0)
                        state['exp_avg_P'] = torch.zeros(0)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                exp_avgs_GG.append(state['exp_avg_GG'])
                exp_avgs_P.append(state['exp_avg_P'])
                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state['step'] += 1
                state_steps.append(state['step'])

            cosmos(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                exp_avgs_GG,
                exp_avgs_P,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                ratio=self.lr_ratio,
                gamma=self.gamma,
                nesterov=self.nesterov,
            )

        return loss
