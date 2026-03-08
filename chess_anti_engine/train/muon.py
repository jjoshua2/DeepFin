from __future__ import annotations

import math

import torch


class MuonWithAuxAdam(torch.optim.Optimizer):
    """Muon for hidden matrix weights, AdamW for everything else.

    This follows the common Muon setup: use Muon on 2D hidden/trunk weights and
    AdamW on auxiliary parameters such as heads, norms, and biases.
    """

    def __init__(
        self,
        params,
        *,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
    ) -> None:
        defaults = dict(
            lr=0.0,
            weight_decay=0.0,
            use_muon=False,
            betas=adam_betas,
            eps=float(adam_eps),
            muon_momentum=float(muon_momentum),
            muon_nesterov=bool(muon_nesterov),
            muon_ns_steps=int(muon_ns_steps),
        )
        super().__init__(params, defaults)

    @staticmethod
    def _as_matrix(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2:
            return t
        return t.reshape(t.shape[0], -1)

    @staticmethod
    def _zeropower_via_newton_schulz5(mat: torch.Tensor, *, steps: int) -> torch.Tensor:
        if mat.ndim != 2:
            raise ValueError(f"Muon expects a matrix update, got shape={tuple(mat.shape)}")

        transposed = mat.shape[0] > mat.shape[1]
        x = mat.transpose(0, 1) if transposed else mat
        x = x.to(dtype=torch.float32)
        x = x / (x.norm() + 1e-7)

        a = 3.4445
        b = -4.7750
        c = 2.0315
        for _ in range(max(1, int(steps))):
            xx_t = x @ x.transpose(0, 1)
            x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x

        if transposed:
            x = x.transpose(0, 1)
        return x.to(dtype=mat.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", 0.0))
            use_muon = bool(group.get("use_muon", False))

            if use_muon:
                momentum = float(group.get("muon_momentum", 0.95))
                nesterov = bool(group.get("muon_nesterov", True))
                ns_steps = int(group.get("muon_ns_steps", 5))
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    grad = param.grad.detach()
                    if grad.is_sparse:
                        raise RuntimeError("Muon does not support sparse gradients")
                    if weight_decay != 0.0:
                        param.mul_(1.0 - lr * weight_decay)

                    state = self.state[param]
                    buf = state.get("momentum_buffer")
                    if buf is None:
                        buf = torch.zeros_like(grad)
                        state["momentum_buffer"] = buf
                    buf.mul_(momentum).add_(grad)
                    update = grad.add(buf, alpha=momentum) if nesterov else buf

                    mat = self._as_matrix(update)
                    ortho = self._zeropower_via_newton_schulz5(mat, steps=ns_steps)
                    ortho = ortho * math.sqrt(max(1.0, mat.shape[0] / max(1, mat.shape[1])))
                    param.add_(ortho.reshape_as(param), alpha=-lr)
                continue

            beta1, beta2 = tuple(group.get("betas", (0.9, 0.95)))
            eps = float(group.get("eps", 1e-8))
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("AdamW fallback does not support sparse gradients")
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
                denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
                denom.add_(eps)
                step_size = lr / bias_correction1
                param.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
