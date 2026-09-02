from __future__ import annotations

import logging
import math
from itertools import pairwise
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
    polar_express_fn: Callable[..., Tensor] | None = None,
) -> Tensor:
    method = str(method).lower()
    if method in ("simple", "simple_quintic", "quintic"):
        return _polar_quintic(mat, steps=steps, eps=eps, work_dtype=work_dtype)
    if method in ("polar_express", "pe", "pe8"):
        function = _polar_express if polar_express_fn is None else polar_express_fn
        return function(mat, steps=steps, eps=eps, safety=safety, work_dtype=work_dtype)
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
    polar_express_fn: Callable[..., Tensor] | None = None,
    check_finite: bool = True,
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
            polar_express_fn=polar_express_fn,
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
                polar_express_fn=polar_express_fn,
            ).to(torch.float32)
            if k < int(pp_iterations) - 1:
                row_sq = out_f.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps * eps)
                scale = scale * (target_row_sq / row_sq).pow(float(pp_beta))

        out = out_f.transpose(0, 1) if transposed else out_f

    out = out * math.sqrt(max(1.0, float(orig_rows) / max(1.0, float(orig_cols))))
    if check_finite and not torch.isfinite(out).all():
        raise RuntimeError(
            f"Aurora produced non-finite update for matrix shape {(orig_rows, orig_cols)}"
        )
    return out.to(update.dtype)


class _CapturedAuroraUpdate:
    """Exact eager Aurora-update replay for one fixed CUDA parameter."""

    def __init__(
        self,
        sample: Tensor,
        *,
        pp_iterations: int,
        pp_beta: float,
        polar_steps: int,
        polar_method: str,
        polar_dtype: str | torch.dtype | None,
        polar_safety: float,
        eps: float,
    ) -> None:
        self.static_input = torch.empty_like(sample)
        self.static_input.copy_(sample)

        def run() -> Tensor:
            return _aurora_update(
                self.static_input,
                pp_iterations=pp_iterations,
                pp_beta=pp_beta,
                polar_steps=polar_steps,
                polar_method=polar_method,
                polar_dtype=polar_dtype,
                polar_safety=polar_safety,
                eps=eps,
                check_finite=False,
            )

        warmup_stream = torch.cuda.Stream(device=sample.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(sample.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream(sample.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(sample.device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = run()

    def __call__(self, update: Tensor) -> Tensor:
        self.static_input.copy_(update)
        self.graph.replay()
        return self.static_output


def polar_convergence(update: Tensor) -> tuple[float, float]:
    """Polar residual of one applied Aurora update: ``(sv_ratio, orth_err)``.

    ``sv_ratio`` is ``sigma_min / sigma_max`` of ``update`` and ``orth_err`` is
    ``||Q Q^T - I||_F / sqrt(n)`` of the same matrix rescaled to unit spectral
    norm and oriented wide (``n`` = the short side). Both are the quantities
    tabulated in `MODEL_OPT_AUDIT.md` Addendum II B3 ("full" and "orth"), and
    `tests/test_aurora_polar_convergence.py` pins this function against that
    table's real-momentum reference values at 8 and 12 polar steps.

    A perfectly converged polar factor reads ``(1.0, 0.0)``. Production's
    `aurora_polar_steps: 8` reads **0.0273 / 0.2114** on the DESIGNATED square
    tensor of `checkpoint_000478` (group index 0, the tensor `step()` actually
    samples): the iteration's amplification budget is short of what that
    momentum's conditioning needs (M4-1), so the update it applies is a
    partially-preconditioned step rather than an orthogonal one. Nothing
    measured this for the life of the run, which is why the number is an
    instrument here and not a fix.

    ⚑ Those are ONE TENSOR's readings, not the 16-tensor group means
    (0.0209 / 0.1082) that `MODEL_OPT_AUDIT.md` B3 tabulates. Across the 16
    square tensors the PE-8 spread is 0.0005-0.0455 on ``sv_ratio`` (94x) and
    0.0794-0.2114 on ``orth_err`` (2.7x), so a group mean is NOT a bar this
    column can be held to. The arm/control RATIO on the same designated tensor
    is the stable statistic -- 11.40-12.36 on ``sv_ratio`` across every
    possible designation, a 1.08x spread -- and is what the ledger gate uses.

    Both readings are invariant to a positive rescale of ``update``, so the
    trailing ``sqrt(rows/cols)`` factor and any ``aurora_uw_floor`` scaling
    that `step()` applies do not move them -- measuring the update Aurora
    actually applies and measuring the raw polar factor are the same
    measurement. The SVD runs in float64 because the interesting singular
    values sit 6+ orders of magnitude below ``sigma_max``; a power-iteration
    estimator cannot resolve them (Addendum II B2 measured one landing BELOW
    ``sigma_max`` on 11 of 20 real tensors) and is the reason this is not the
    "few power iterations, no SVD" sketch the audit's I3 proposed.
    """
    if update.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got shape={tuple(update.shape)}")
    mat = update.detach().to(torch.float64)
    svals = torch.linalg.svdvals(mat)
    sigma_max = float(svals[0].item())
    if not math.isfinite(sigma_max) or sigma_max <= 0.0:
        raise ValueError(f"degenerate update spectrum: sigma_max={sigma_max!r}")
    sv_ratio = float(svals[-1].item()) / sigma_max
    wide = mat / sigma_max
    if wide.size(0) > wide.size(1):
        wide = wide.transpose(0, 1)
    n = int(wide.size(0))
    eye = torch.eye(n, dtype=wide.dtype, device=wide.device)
    orth_err = float((wide @ wide.transpose(0, 1) - eye).norm().item()) / math.sqrt(n)
    return sv_ratio, orth_err


def _polar_stats(
    samples: dict[str, tuple[float, float]],
    *,
    polar_steps: int,
    errors: int,
) -> dict[str, float]:
    """Assemble the per-iteration polar-residual row from the sampled tensors.

    ``polar_steps`` travels with the readings on purpose: the numbers only mean
    anything against the step count that produced them, and the A8 A/B's whole
    question is whether a changed step count reached the optimizer.
    """
    out: dict[str, float] = {
        "aurora_polar_steps_configured": float(polar_steps),
        "aurora_polar_sv_samples": float(len(samples)),
        "aurora_polar_sv_errors": float(errors),
    }
    for shape_class, (sv_ratio, orth_err) in samples.items():
        out[f"aurora_polar_sv_ratio_{shape_class}"] = float(sv_ratio)
        out[f"aurora_polar_orth_err_{shape_class}"] = float(orth_err)
    return out


def _uw_stats(ratios: list[Tensor], scales: list[Tensor], *, lr: float, floor: float) -> dict[str, float]:
    """Summarize Aurora update/weight ratios from one optimizer step.

    ``aurora_uw_effective_ratio_*`` multiplies by ``lr``, and the caller
    (`Trainer.train_steps`) collects these on the LAST step of the training
    window. Under `lr_schedule: sqrt_release` with `lr_release_cycle_steps: 0`
    that window IS one whole WSD cycle, so the last step sits at the sawtooth
    FLOOR -- `lr_release_min_scale` (0.1 in production) times the peak. The
    effective-ratio pair is therefore ~10x below a typical step's, every
    iteration, by construction (M4-2). ``aurora_uw_lr`` is that exact ``lr``,
    reported alongside so the pair can be de-scaled against `opt_lr_mean` /
    `opt_lr_max` instead of being read as if it described a typical step.
    ``aurora_uw_ratio_*`` carries no ``lr`` factor and is unaffected.
    """
    if not ratios:
        return {}
    ratio = torch.stack([r.float().reshape(()) for r in ratios])
    scale = torch.stack([s.float().reshape(()) for s in scales])
    effective = ratio * scale * float(lr)
    floored = scale > 1.000001
    return {
        "aurora_uw_floor": float(floor),
        "aurora_uw_lr": float(lr),
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


  # Dtypes on which the batched AdamW fallback is BITWISE identical to the
  # per-parameter loop it replaces, and the only dtypes it is allowed to take.
  #
  # ⚑ This is an allowlist, not a "floating point" test, because ONE op breaks
  # the identity: `torch._foreach_mul_(tensors, <python float>)` disagrees with
  # `Tensor.mul_(<python float>)` in the last bit on reduced precision.
  # MEASURED on torch 2.11.0, CPU, shapes (512,512)/(512,)/(1858,512)/(3,):
  # every other op in the chain -- `add_(other, alpha=)`, `addcmul_`, `sqrt`,
  # `div_(scalar)`, `add_(scalar)`, `addcdiv_` -- is bitwise identical on
  # bfloat16 AND float16, while the scalar `mul_` differs on both; neither the
  # `ScalarList` nor the 0-d `TensorList` overload of `_foreach_mul_` closes it.
  # float32/float64 are exact for the whole chain. The MECHANISM (which side
  # keeps a wider accumulator for the scalar multiply) was not established --
  # only the disagreement was, which is all the allowlist needs, and
  # `test_bfloat16_would_not_survive_the_batched_path` re-measures it rather
  # than trusting this comment.
  #
  # Production trains fp32 parameters under a bf16 autocast (`Trainer` pins
  # `inference_autocast(..., dtype="bf16")` for the FORWARD only, so params and
  # grads stay fp32), which is why the live path is the batched one. Anything
  # else -- a reduced-precision master weight, a checkpoint whose moments were
  # saved at another dtype, a mixed param/grad dtype -- falls back to the exact
  # per-parameter sequence rather than being silently approximated.
_FOREACH_EXACT_DTYPES = frozenset({torch.float32, torch.float64})


def _adamw_group_aliases(params: list[Tensor]) -> bool:
    """True when two entries of the group share MEMORY, by object or by storage.

    The batched path is the loop's ordered sequential update only if no two
    entries of a bucket touch the same element: `_foreach_mul_` over a list
    holding one storage twice is not two sequential decays of it, and with
    weight decay on the two orders differ (P·(1-x)·(1-x) - u_a - u_b against
    (P·(1-x) - u_a)·(1-x) - u_b). Identity catches a tied Parameter reaching a
    group twice; it does NOT catch two distinct Parameter objects, or two
    views, over one storage, so the check is on the byte ranges the tensors
    actually cover: sort by first byte and look for a range that starts before
    the previous one ends. Disjoint views of one storage share no element and
    stay batchable; interleaved strided views overlap by span and are read
    CONSERVATIVELY as aliasing (the loop is exactly right there).

    Cost, MEASURED on the real 431-tensor production inventory (CPU, median of
    200 calls): the first cut ran a per-dimension strided-extent loop on every
    tensor and cost ~0.5 ms per step (reviewer's number: 527 us). This version
    takes the contiguous fast path -- `data_ptr()`, which already includes the
    storage offset, plus `numel() * element_size()` -- for every contiguous
    tensor (all 431 in production) and computes the strided extent only for a
    non-contiguous view: 61 us for the 142-tensor group + 126 us for the
    289-tensor group = ~0.19 ms per step, ~0.008% of a 2.26 s optimizer step.
    Runs once per group per step. On both `configs/lc0_positive_control.yaml`
    and `configs/pbt2_small.yaml`: 431 unique storages / 431 params, 0 overlaps.
    """
    if len({id(param) for param in params}) != len(params):
        return True
    spans: list[tuple[int, int]] = []
    for param in params:
        numel = param.numel()
        if numel == 0:
            continue
        element_size = param.element_size()
        start = int(param.data_ptr())
        if param.is_contiguous():
            extent = numel
        else:
  # Extent in elements of the strided view, not `numel()`: a non-contiguous
  # view covers more storage than it has elements.
            extent = 1 + sum(
                (int(size) - 1) * abs(int(stride)) for size, stride in zip(param.shape, param.stride())
            )
        spans.append((start, start + extent * element_size))
    spans.sort()
    return any(nxt_start < prev_end for (_, prev_end), (nxt_start, _) in pairwise(spans))


def _adamw_batchable(param: Tensor, grad: Tensor, exp_avg: Tensor, exp_avg_sq: Tensor) -> bool:
    """True when this parameter's four tensors may join a `_foreach_*` bucket."""
    if param.dtype not in _FOREACH_EXACT_DTYPES:
        return False
    return all(
        t.dtype == param.dtype and t.device == param.device
        for t in (grad, exp_avg, exp_avg_sq)
    )


def _adamw_update_one(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step: int,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> None:
    """One parameter's AdamW fallback update, in the original op order.

    This is the reference the batched path must reproduce bit for bit, and it
    is also the live path for any tensor `_adamw_batchable` rejects.
    """
    if weight_decay != 0.0:
        param.mul_(1.0 - lr * weight_decay)
    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    _adamw_finish_one(
        param, exp_avg, exp_avg_sq, step=step, lr=lr, beta1=beta1, beta2=beta2, eps=eps,
    )


def _adamw_finish_one(
    param: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    *,
    step: int,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> None:
    """The tail of `_adamw_update_one` from the denominator on: the ops AFTER
    the moments have been decayed and accumulated.

    Split out because it is also the recovery path for a bucket whose
    `_foreach_sqrt` failed to allocate (see `_DenominatorAllocationFailed`):
    the moments of every tensor in that bucket are already at their post-step
    values, so finishing each tensor here, one denominator at a time, lands the
    bucket exactly where the batched chain would have. Per-tensor `sqrt` /
    `div` / `add_` / `addcdiv_` are the reference ops the batched kernels are
    pinned to bit for bit, so this is not an approximation of the step -- it
    IS the step, in the pre-change order.
    """
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    denom = exp_avg_sq.sqrt() / math.sqrt(max(bias_correction2, 1e-12))
    denom.add_(eps)
    step_size = lr / max(bias_correction1, 1e-12)
    param.addcdiv_(exp_avg, denom, value=-step_size)


class _DenominatorAllocationFailed(RuntimeError):
    """`_foreach_sqrt` raised inside `_adamw_update_foreach`, AFTER the
    in-place moment kernels had run and BEFORE any parameter update.

    It is the one point in the batched chain where the state is both mutated
    and recoverable: the four in-place kernels before it allocate nothing and
    have completed for the whole bucket, `_foreach_sqrt` is out-of-place so a
    failure inside it leaves its inputs untouched, and everything after it
    only needs one denominator at a time. `step` catches this and finishes the
    bucket per tensor. The message is the original error's, so if the recovery
    itself fails and this propagates, `Trainer.train_steps`' "CUDA" retry test
    still sees what it would have seen.
    """


def _adamw_update_foreach(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    *,
    step: int,
    lr: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> None:
    """`_adamw_update_one` for a whole bucket, as batched `_foreach_*` kernels.

    Every op in `_adamw_update_one` is ELEMENTWISE -- no op mixes elements of
    one tensor, and none mixes tensors with each other -- so evaluating them
    parameter-major (the loop) or op-major (this) computes the same expression
    on the same inputs. The order below is the loop's order, unchanged.

    The caller guarantees the bucket shares `step`, dtype and device, because
    `bias_correction*` and `step_size` are step-dependent scalars and the
    identity above only holds within `_FOREACH_EXACT_DTYPES`. It also
    guarantees no tensor appears twice: `_foreach_mul_` over a list holding
    the same storage twice is not the loop's two sequential updates.
    """
    if weight_decay != 0.0:
        torch._foreach_mul_(params, 1.0 - lr * weight_decay)
    torch._foreach_mul_(exp_avgs, beta1)
    torch._foreach_add_(exp_avgs, grads, alpha=1.0 - beta1)
    torch._foreach_mul_(exp_avg_sqs, beta2)
    torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1.0 - beta2)

    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
  # ⚑ Scope of what a mid-chain failure leaves behind, stated exactly. The four
  # in-place kernels above are the ONLY mutation before the parameter update,
  # and `_foreach_sqrt` below is the ONLY allocation in the chain (one fp32
  # denominator per tensor -- 165.6 MiB for the production 142-tensor group),
  # so an allocator `RuntimeError` fires HERE, after every moment in the bucket
  # has been decayed and accumulated and before any parameter has moved (the
  # weight-decay `mul_` at the top excepted, when it is on). `Trainer.train_steps`
  # retries a step whose error mentions "CUDA" and records the retry as clean,
  # so left alone that retry would decay/accumulate the whole bucket's moments a
  # SECOND time with the replacement batch's gradients -- the old loop had the
  # same shape bounded to the one tensor it was on; batching widens it to the
  # bucket. Rather than reorder the kernels (a kernel change, which would need
  # its own phase-0b identity run), the failure is typed and `step` finishes the
  # bucket per tensor from the moments it already has: same ops, one
  # denominator at a time, bitwise the loop's result. Failures inside the
  # in-place kernels are NOT recoverable this way and propagate unchanged.
    try:
        denoms = torch._foreach_sqrt(exp_avg_sqs)
    except RuntimeError as exc:
        raise _DenominatorAllocationFailed(str(exc)) from exc
    torch._foreach_div_(denoms, math.sqrt(max(bias_correction2, 1e-12)))
    torch._foreach_add_(denoms, eps)
    step_size = lr / max(bias_correction1, 1e-12)
    torch._foreach_addcdiv_(params, exp_avgs, denoms, value=-step_size)


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
        aurora_cuda_graphs: bool = True,
        aurora_coalesce_finite_checks: bool = True,
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
        self.last_polar_stats: dict[str, float] = {}
  # Which AdamW-fallback path the LAST step ran, harvested onto the step row
  # by `Trainer.train_steps` like `last_uw_stats`. `adamw_foreach_params` is
  # the take-effect column for the batched path. It counts tensors that
  # CARRIED a gradient this step (a `grad is None` tensor is skipped by both
  # paths): the lc0 control reads 394 of its 431 fallback tensors in 2 buckets
  # with `adamw_loop_params` 0, the other 37 being dead heads with no grad.
  # A loop count that is not 0 means
  # a batchability predicate (dtype allowlist, duplicate guard, mixed
  # device/dtype) sent tensors down the per-parameter path, silently. Written
  # every step -- four Python ints, no device sync.
        self.last_adamw_stats: dict[str, float] = {}
  # Monotone count of buckets finished per tensor after a denominator
  # allocation failure, over the optimizer's lifetime. `last_adamw_stats`
  # is the LAST step only, so a recovery on any step but the last of a
  # `train_steps` window would publish 0 on the row; the trainer reports
  # the window's DIFFERENCE of this counter instead. Incremented only once
  # the per-tensor completion has succeeded -- a completion that dies is
  # not a recovery.
        self.adamw_foreach_recoveries_total = 0
        self._collect_uw_stats = True
        self._collect_polar_stats = False
        self._use_update_graphs = bool(aurora_cuda_graphs)
        self._coalesce_finite_checks = bool(aurora_coalesce_finite_checks)
        self._update_graphs: dict[tuple[object, ...], _CapturedAuroraUpdate] = {}

    def set_collect_uw_stats(self, collect: bool) -> None:
        """Collect zero-floor diagnostics on this step when requested."""
        self._collect_uw_stats = bool(collect)

    def set_collect_polar_stats(self, collect: bool) -> None:
        """Sample polar residual on this step when requested.

        Defaults OFF and is armed per step by the caller, because each sample
        costs a float64 SVD. `Trainer.train_steps` arms it on one step per
        iteration; at that cadence the cost is ~2 SVDs of a 512-wide matrix
        against a ~665 s iteration.
        """
        self._collect_polar_stats = bool(collect)

    def _aurora_update_for_param(
        self,
        param: Tensor,
        update: Tensor,
        *,
        pp_iterations: int,
        pp_beta: float,
        polar_steps: int,
        polar_method: str,
        polar_dtype: str | torch.dtype | None,
        polar_safety: float,
        eps: float,
    ) -> Tensor:
        use_graph = (
            update.device.type == "cuda"
            and self._use_update_graphs
            and self._coalesce_finite_checks
            and str(polar_method).lower() in ("polar_express", "pe", "pe8")
        )
        if not use_graph:
            return _aurora_update(
                update,
                pp_iterations=pp_iterations,
                pp_beta=pp_beta,
                polar_steps=polar_steps,
                polar_method=polar_method,
                polar_dtype=polar_dtype,
                polar_safety=polar_safety,
                eps=eps,
                check_finite=not self._coalesce_finite_checks,
            )
        key = (
            param,
            int(pp_iterations),
            float(pp_beta),
            int(polar_steps),
            str(polar_method),
            polar_dtype,
            float(polar_safety),
            float(eps),
        )
        captured = self._update_graphs.get(key)
        if captured is None:
            captured = _CapturedAuroraUpdate(
                update,
                pp_iterations=pp_iterations,
                pp_beta=pp_beta,
                polar_steps=polar_steps,
                polar_method=polar_method,
                polar_dtype=polar_dtype,
                polar_safety=polar_safety,
                eps=eps,
            )
            self._update_graphs[key] = captured
        return captured(update)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        adamw_foreach_buckets = 0
        adamw_foreach_params = 0
        adamw_loop_params = 0
        adamw_foreach_recoveries = 0
        for group in self.param_groups:
            lr = float(group["lr"])
            weight_decay = float(group.get("weight_decay", 0.0))
            use_aurora = bool(group.get("use_aurora", False))

            if use_aurora:
                uw_ratios: list[Tensor] = []
                uw_scales: list[Tensor] = []
                pending_updates: list[tuple[Tensor, Tensor]] = []
                finite_checks: list[Tensor] = []
                momentum = float(group.get("aurora_momentum", 0.95))
                nesterov = bool(group.get("aurora_nesterov", True))
                pp_iterations = int(group.get("aurora_pp_iterations", 2))
                pp_beta = float(group.get("aurora_pp_beta", 0.5))
                polar_steps = int(group.get("aurora_polar_steps", 12))
                polar_method = str(group.get("aurora_polar_method", "simple"))
                polar_dtype = group.get("aurora_polar_dtype", None)
                polar_safety = float(group.get("aurora_polar_safety", 1.01))
                uw_floor = float(group.get("aurora_uw_floor", 0.0))
                collect_uw_stats = self._collect_uw_stats or uw_floor > 0.0
                eps = float(group.get("eps", 1e-8))
  # One designated tensor per shape class per armed step: the FIRST square
  # and the FIRST non-square parameter of the group in group order. Not a
  # random draw -- there is no sampling rng to seed, and the same two
  # tensors are measured every iteration, so the series is a paired
  # comparison of one tensor against itself rather than a shape-mix
  # average that moves when the group's composition does.
                polar_samples: dict[str, tuple[float, float]] = {}
                polar_errors = 0
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
                    update = self._aurora_update_for_param(
                        param,
                        update,
                        pp_iterations=pp_iterations,
                        pp_beta=pp_beta,
                        polar_steps=polar_steps,
                        polar_method=polar_method,
                        polar_dtype=polar_dtype,
                        polar_safety=polar_safety,
                        eps=eps,
                    )
                    if collect_uw_stats:
                        weight_fro = param.float().norm().clamp_min(eps)
                        update_fro = update.float().norm().clamp_min(eps)
                        uw_ratio = update_fro / weight_fro
                        scale = torch.ones((), dtype=torch.float32, device=update.device)
                        if uw_floor > 0.0:
                            ratio_ok = torch.isfinite(uw_ratio) & (uw_ratio > 0.0)
                            floor = torch.as_tensor(
                                float(uw_floor), dtype=torch.float32, device=update.device,
                            )
                            safe_ratio = torch.where(
                                ratio_ok, uw_ratio, torch.ones_like(uw_ratio),
                            )
                            scale = torch.where(
                                ratio_ok,
                                torch.clamp_min(floor / safe_ratio, 1.0),
                                scale,
                            )
                            update = update * scale.to(update.dtype)
                        uw_ratios.append(uw_ratio.detach())
                        uw_scales.append(scale.detach())
  # Measured HERE, on the live `update` tensor this step is about to
  # apply, and never stashed for later: under CUDA graphs `update` is
  # the captured graph's static output buffer, which the next replay
  # overwrites. A copy taken now and measured after the loop would
  # report some other parameter's update under this one's name.
                    if self._collect_polar_stats:
                        shape_class = "square" if param.size(0) == param.size(1) else "rect"
                        if shape_class not in polar_samples:
                            try:
                                polar_samples[shape_class] = polar_convergence(update)
  # torch.linalg raises `_LinAlgError`, a RuntimeError subclass, when a
  # decomposition fails to converge. Counted rather than raised: this is a
  # diagnostic, and `aurora_polar_sv_errors` on the row is what says the
  # reading is missing instead of merely zero.
  #
  # OOM is NOT that. `RuntimeError` is also what an exhausted allocator
  # raises, and this is a diagnostic allocating a float64 copy of a
  # 512-wide matrix on a card the training step is already filling --
  # swallowing that would turn "the run is out of memory" into a silently
  # incremented counter and let the step proceed into whatever fails
  # next. Re-raise it, and name the class for everything else so a
  # non-zero counter is diagnosable from the log instead of being a
  # bare integer.
                            except torch.cuda.OutOfMemoryError:
                                raise
                            except (RuntimeError, ValueError) as exc:
                                polar_errors += 1
                                logging.getLogger(__name__).warning(
                                    "polar-convergence sample failed on a %s tensor "
                                    "(%s: %s); reporting aurora_polar_sv_errors",
                                    shape_class, type(exc).__name__, exc,
                                )
                    if self._coalesce_finite_checks:
                        finite_checks.append(torch.isfinite(update).all())
                        pending_updates.append((param, update))
                    else:
                        param.add_(update, alpha=-lr)
                if self._coalesce_finite_checks and finite_checks:
                    if not torch.stack(finite_checks).all():
                        raise RuntimeError("Aurora produced a non-finite matrix update")
                    for param, update in pending_updates:
                        param.add_(update, alpha=-lr)
                if collect_uw_stats:
                    self.last_uw_stats = _uw_stats(
                        uw_ratios, uw_scales, lr=lr, floor=uw_floor,
                    )
                if self._collect_polar_stats:
                    self.last_polar_stats = _polar_stats(
                        polar_samples, polar_steps=polar_steps, errors=polar_errors,
                    )
                continue

            beta1, beta2 = tuple(group.get("betas", (0.9, 0.95)))
            eps = float(group.get("eps", 1e-8))
            params = group["params"]
  # `step` is per-parameter (a tensor that had no grad on some step is behind
  # the rest), and `bias_correction*` / `step_size` are functions of it, so the
  # step count is part of the bucket key rather than a loop invariant.
  #
  # ⚑ Aliases disqualify the WHOLE group, checked before any state is
  # touched: a tied weight reaching one group twice, or two Parameters /
  # views over one storage (`_adamw_group_aliases`). `_foreach_mul_` applied
  # to a list holding one storage twice is not the loop's two sequential
  # updates of it. The loop stays exactly right in that case, so it is what
  # runs -- this repo has tied tensors (the 16 `layer_smolgens.N.gen_weight.weight`
  # keys are one storage), which is why the check is here and not assumed away.
            allow_batching = not _adamw_group_aliases(params)
            buckets: dict[
                tuple[torch.device, torch.dtype, int],
                tuple[
                    list[Tensor], list[Tensor], list[Tensor], list[Tensor],
                    list[dict[str, object]],
                ],
            ] = {}
            for param in params:
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("Aurora AdamW fallback does not support sparse gradients")

                state = self.state[param]
  # ⚑ `step` is COMPUTED here and COMMITTED only once the update that uses it
  # has run. `Trainer.train_steps` catches a `RuntimeError` whose message
  # contains "CUDA" and RETRIES the whole step up to three times
  # (`trainer.py`), so an optimizer step is not the unrecoverable event it
  # looks like -- and the batched path raises AFTER the scan has visited every
  # parameter, where the old loop raised part-way through. Writing the counter
  # during the scan would hand the retry a whole group of parameters whose
  # bias corrections had advanced without their moments, and the retry is
  # recorded as a successful step, so nothing downstream would ever say so.
  #
  # The moment BUFFERS are still committed during the scan, and that is
  # correct rather than an oversight: lazily created `exp_avg`/`exp_avg_sq`
  # are zeros, which is exactly the state a retry should find. This bounds
  # the damage of a failed step; it does not make one atomic (a bucket that
  # dies mid-chain has already mutated some tensors), and no cheap version of
  # it does.
                step = int(state.get("step", 0)) + 1

                exp_avg = state.get("exp_avg")
                exp_avg_sq = state.get("exp_avg_sq")
                if exp_avg is None:
                    exp_avg = torch.zeros_like(grad)
                    exp_avg_sq = torch.zeros_like(grad)
                    state["exp_avg"] = exp_avg
                    state["exp_avg_sq"] = exp_avg_sq

                if allow_batching and _adamw_batchable(param, grad, exp_avg, exp_avg_sq):
  # `get`, not `setdefault`: this runs once per parameter per step in a
  # change whose whole point is Python overhead, and `setdefault` would
  # build four throwaway lists and a tuple on every hit.
                    key = (param.device, param.dtype, step)
                    bucket = buckets.get(key)
                    if bucket is None:
                        bucket = ([], [], [], [], [])
                        buckets[key] = bucket
                    bucket[0].append(param)
                    bucket[1].append(grad)
                    bucket[2].append(exp_avg)
                    bucket[3].append(exp_avg_sq)
                    bucket[4].append(state)
                    continue

                _adamw_update_one(
                    param, grad, exp_avg, exp_avg_sq,
                    step=step, lr=lr, weight_decay=weight_decay,
                    beta1=beta1, beta2=beta2, eps=eps,
                )
                state["step"] = step
                adamw_loop_params += 1

  # Deferring the batched buckets past the scan reorders parameters against
  # each other, and that is safe for exactly the reason the batching itself is:
  # a parameter's update reads only its own four tensors, all of which the scan
  # leaves alone. It is NOT safe if two entries alias, which `allow_batching`
  # is what rules out.
            for (_device, _dtype, step), bucket in buckets.items():
                adamw_foreach_buckets += 1
                adamw_foreach_params += len(bucket[0])
                try:
                    _adamw_update_foreach(
                        bucket[0], bucket[1], bucket[2], bucket[3],
                        step=step, lr=lr, weight_decay=weight_decay,
                        beta1=beta1, beta2=beta2, eps=eps,
                    )
                except _DenominatorAllocationFailed as exc:
  # The bucket's moments are already at their post-step values and no
  # parameter has taken its update (see the class docstring). Finish each
  # tensor from those moments, one denominator at a time, and commit its
  # `step` the moment IT is complete -- so if this loop dies too, every
  # tensor's recorded step is true for that tensor: the ones it reached are
  # fully updated and committed, the ones it did not are moments-mutated
  # and uncommitted, and the error propagates to the trainer's retry, which
  # then double-applies exactly those. That residue is stated, not hidden:
  # `adamw_foreach_recoveries` counts the recoveries that completed, and the
  # warning below names the bucket.
                    logging.getLogger(__name__).warning(
                        "batched AdamW denominator allocation failed on a %d-tensor "
                        "bucket at step %d (%s); finishing the bucket per tensor "
                        "from its already-updated moments",
                        len(bucket[0]), step, exc,
                    )
                    for param, exp_avg, exp_avg_sq, state in zip(
                        bucket[0], bucket[2], bucket[3], bucket[4],
                    ):
                        _adamw_finish_one(
                            param, exp_avg, exp_avg_sq,
                            step=step, lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        )
                        state["step"] = step
                    adamw_foreach_recoveries += 1
                    self.adamw_foreach_recoveries_total += 1
                    continue
                for state in bucket[4]:
                    state["step"] = step

        self.last_adamw_stats = {
            "adamw_foreach_buckets": float(adamw_foreach_buckets),
            "adamw_foreach_params": float(adamw_foreach_params),
            "adamw_loop_params": float(adamw_loop_params),
            "adamw_foreach_recoveries": float(adamw_foreach_recoveries),
        }
        return loss
