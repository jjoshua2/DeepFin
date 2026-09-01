"""The batched AdamW fallback must be BITWISE identical to the loop it replaced.

`AuroraWithAuxAdam.step` used to walk its non-Aurora groups one parameter at a
time; it now buckets them by (device, dtype, step) and drives `torch._foreach_*`
kernels. That is admissible only if it changes nothing at all about the numbers,
because it ships mid-experiment into arms that are being compared across
branches -- so every assertion here is `torch.equal`, never `allclose`.

`_reference_adamw_loop` below is a verbatim copy of the pre-change loop. Keeping
it in the test rather than importing something is the point: it is the frozen
observation the new code is scored against, and it does not move when the
optimizer does.

⚑ Bitwise equality alone is a gate that cannot fail -- a batched path that
silently never engaged would pass every comparison in this file. So the tests
that assert equality also assert WHICH path ran, by counting calls into the two
update helpers.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from chess_anti_engine.train import aurora as aurora_module
from chess_anti_engine.train.aurora import (
    AuroraWithAuxAdam,
    _adamw_update_foreach,
    _adamw_update_one,
)

_LR = 3e-4
_BETA1 = 0.9
_BETA2 = 0.95
_EPS = 1e-8


@torch.no_grad()
def _reference_adamw_loop(
    params: Sequence[Tensor],
    state: dict[Tensor, dict[str, object]],
    *,
    lr: float,
    weight_decay: float,
    beta1: float = _BETA1,
    beta2: float = _BETA2,
    eps: float = _EPS,
) -> None:
    """Verbatim copy of the per-parameter AdamW fallback loop `step` used to run."""
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if grad.is_sparse:
            raise RuntimeError("Aurora AdamW fallback does not support sparse gradients")
        if weight_decay != 0.0:
            param.mul_(1.0 - lr * weight_decay)

        param_state = state.setdefault(param, {})
        step = int(param_state.get("step", 0)) + 1  # pyright: ignore[reportArgumentType]
        param_state["step"] = step

        exp_avg = param_state.get("exp_avg")
        exp_avg_sq = param_state.get("exp_avg_sq")
        if exp_avg is None:
            exp_avg = torch.zeros_like(grad)
            exp_avg_sq = torch.zeros_like(grad)
            param_state["exp_avg"] = exp_avg
            param_state["exp_avg_sq"] = exp_avg_sq
        assert isinstance(exp_avg, Tensor)
        assert isinstance(exp_avg_sq, Tensor)

        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        denom = exp_avg_sq.sqrt() / math.sqrt(max(bias_correction2, 1e-12))
        denom.add_(eps)
        step_size = lr / max(bias_correction1, 1e-12)
        param.addcdiv_(exp_avg, denom, value=-step_size)


class _PathCounter:
    """Counts which of the two update helpers `step` actually reached."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.foreach = 0
        self.single = 0
        real_foreach = aurora_module._adamw_update_foreach
        real_single = aurora_module._adamw_update_one

        def counted_foreach(*args: object, **kwargs: object) -> None:
            self.foreach += 1
            real_foreach(*args, **kwargs)  # pyright: ignore[reportArgumentType]

        def counted_single(*args: object, **kwargs: object) -> None:
            self.single += 1
            real_single(*args, **kwargs)  # pyright: ignore[reportArgumentType]

        monkeypatch.setattr(aurora_module, "_adamw_update_foreach", counted_foreach)
        monkeypatch.setattr(aurora_module, "_adamw_update_one", counted_single)

    def reset(self) -> None:
        self.foreach = 0
        self.single = 0


  # Index 5 always receives an ALL-ZERO gradient, which is what makes `eps`
  # observable: without it `denom` is exactly 0 and the `addcdiv_` divides
  # 0 by 0. At float32 an `eps` of 1e-8 is otherwise below the ulp of a
  # typical denominator, so dropping it is a silent no-op on ordinary rows.
_ZERO_GRAD_INDEX = 5


def _make_params(dtype: torch.dtype = torch.float32) -> list[torch.nn.Parameter]:
    """A deliberately mixed shape set: matrix, bias vector, square, 3-D, scalar."""
    generator = torch.Generator().manual_seed(1234)
    shapes: list[tuple[int, ...]] = [(6, 4), (5,), (3, 3), (2, 7, 4), (), (4, 2)]
    return [
        torch.nn.Parameter(torch.randn(shape, generator=generator).to(dtype))
        for shape in shapes
    ]


def _clone_params(params: list[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return [torch.nn.Parameter(p.detach().clone()) for p in params]


def _grads_for_step(
    params: list[torch.nn.Parameter], step_index: int, *, skip: set[int] | None = None,
) -> list[Tensor | None]:
    generator = torch.Generator().manual_seed(9000 + step_index)
    out: list[Tensor | None] = []
    for index, param in enumerate(params):
        if skip is not None and index in skip:
            out.append(None)
            continue
        if index == _ZERO_GRAD_INDEX:
            out.append(torch.zeros(param.shape, dtype=param.dtype))
            continue
        out.append(torch.randn(param.shape, generator=generator).to(param.dtype))
    return out


def _assert_states_bitwise_equal(
    opt: AuroraWithAuxAdam,
    opt_params: list[torch.nn.Parameter],
    ref_params: list[torch.nn.Parameter],
    ref_state: dict[Tensor, dict[str, object]],
) -> None:
    for index, (got, want) in enumerate(zip(opt_params, ref_params)):
        assert torch.equal(got.detach(), want.detach()), f"param {index} diverged"
        got_state = opt.state.get(got, {})
        want_state = ref_state.get(want, {})
        assert set(got_state) == set(want_state), f"state keys differ for param {index}"
        for key, want_value in want_state.items():
            got_value = got_state[key]
            if isinstance(want_value, Tensor):
                assert isinstance(got_value, Tensor)
                assert torch.equal(got_value, want_value), f"state[{key}] of param {index}"
            else:
                assert got_value == want_value, f"state[{key}] of param {index}"


def _run_paired(
    *,
    weight_decay: float,
    steps: int = 5,
    dtype: torch.dtype = torch.float32,
    skip_grads: dict[int, set[int]] | None = None,
    counter: _PathCounter | None = None,
) -> tuple[AuroraWithAuxAdam, list[torch.nn.Parameter], list[torch.nn.Parameter], dict[Tensor, dict[str, object]]]:
    base = _make_params(dtype)
    opt_params = _clone_params(base)
    ref_params = _clone_params(base)
    ref_state: dict[Tensor, dict[str, object]] = {}
    opt = AuroraWithAuxAdam(
        [{"params": opt_params, "lr": _LR, "weight_decay": weight_decay, "use_aurora": False}],
    )

    for step_index in range(steps):
        skip = None if skip_grads is None else skip_grads.get(step_index)
        grads = _grads_for_step(opt_params, step_index, skip=skip)
        for param, ref_param, grad in zip(opt_params, ref_params, grads):
            param.grad = None if grad is None else grad.clone()
            ref_param.grad = None if grad is None else grad.clone()
        if counter is not None:
            counter.reset()
        opt.step()
        _reference_adamw_loop(
            ref_params, ref_state, lr=_LR, weight_decay=weight_decay,
        )
    return opt, opt_params, ref_params, ref_state


@pytest.mark.parametrize("weight_decay", [0.0, 0.03])
def test_foreach_fallback_matches_the_reference_loop_bitwise(weight_decay: float) -> None:
    opt, opt_params, ref_params, ref_state = _run_paired(weight_decay=weight_decay)
    _assert_states_bitwise_equal(opt, opt_params, ref_params, ref_state)
  # A silent no-op would satisfy every assertion above, so prove the params
  # actually moved off their initial values.
    assert all(int(opt.state[p]["step"]) == 5 for p in opt_params)
    base = _make_params()
    moved = [
        i for i, (before, after) in enumerate(zip(base, opt_params))
        if not torch.equal(before.detach(), after.detach())
    ]
  # The zero-gradient parameter moves only through weight decay, which is
  # itself the check that the batched `mul_` reached it.
    expected_moved = [0, 1, 2, 3, 4] if weight_decay == 0.0 else [0, 1, 2, 3, 4, 5]
    assert moved == expected_moved
  # The zero-gradient parameter is the `eps` witness: it stays finite (and,
  # with weight decay off, exactly put) only because `denom` is floored.
    assert torch.isfinite(opt_params[_ZERO_GRAD_INDEX]).all()


@pytest.mark.parametrize("weight_decay", [0.0, 0.03])
def test_the_batched_path_is_the_one_that_runs(
    monkeypatch: pytest.MonkeyPatch, weight_decay: float,
) -> None:
    counter = _PathCounter(monkeypatch)
    opt, opt_params, ref_params, ref_state = _run_paired(
        weight_decay=weight_decay, counter=counter,
    )
  # One bucket: five parameters, one device, one dtype, one shared step count.
    assert counter.foreach == 1
    assert counter.single == 0
    _assert_states_bitwise_equal(opt, opt_params, ref_params, ref_state)


def test_parameters_at_different_step_counts_bucket_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = _PathCounter(monkeypatch)
  # Parameter 2 has no gradient for the first two steps, so its `step` lags the
  # rest by two -- and `bias_correction*` is a function of `step`, which is why
  # a single bucket for the whole group would be wrong rather than merely slow.
    opt, opt_params, ref_params, ref_state = _run_paired(
        weight_decay=0.03, skip_grads={0: {2}, 1: {2}}, counter=counter,
    )
    assert counter.foreach == 2, "lagging parameter must land in its own bucket"
    assert counter.single == 0
    assert int(opt.state[opt_params[2]]["step"]) == 3
    assert int(opt.state[opt_params[0]]["step"]) == 5
    _assert_states_bitwise_equal(opt, opt_params, ref_params, ref_state)


def test_mixed_dtypes_bucket_separately_and_stay_bitwise_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = _PathCounter(monkeypatch)
    single = torch.nn.Parameter(torch.randn(4, 3))
    double = torch.nn.Parameter(torch.randn(4, 3, dtype=torch.float64))
    ref_single = torch.nn.Parameter(single.detach().clone())
    ref_double = torch.nn.Parameter(double.detach().clone())
    ref_state: dict[Tensor, dict[str, object]] = {}
    opt = AuroraWithAuxAdam(
        [{"params": [single, double], "lr": _LR, "weight_decay": 0.02, "use_aurora": False}],
    )
    for step_index in range(3):
        generator = torch.Generator().manual_seed(step_index)
        grad_single = torch.randn(4, 3, generator=generator)
        grad_double = torch.randn(4, 3, generator=generator, dtype=torch.float64)
        single.grad, ref_single.grad = grad_single.clone(), grad_single.clone()
        double.grad, ref_double.grad = grad_double.clone(), grad_double.clone()
        counter.reset()
        opt.step()
        _reference_adamw_loop(
            [ref_single, ref_double], ref_state, lr=_LR, weight_decay=0.02,
        )
    assert counter.foreach == 2
    assert counter.single == 0
    _assert_states_bitwise_equal(
        opt, [single, double], [ref_single, ref_double], ref_state,
    )


def test_reduced_precision_falls_back_to_the_per_parameter_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  # ⚑ `torch._foreach_mul_(tensors, <python float>)` is NOT bitwise equal to
  # `Tensor.mul_(<python float>)` on bfloat16/float16, so those dtypes must
  # take the loop. This test is what keeps the allowlist honest: widen
  # `_FOREACH_EXACT_DTYPES` to bf16 and the equality below breaks.
    counter = _PathCounter(monkeypatch)
    opt, opt_params, ref_params, ref_state = _run_paired(
        weight_decay=0.03, steps=3, dtype=torch.bfloat16, counter=counter,
    )
    assert counter.foreach == 0
    assert counter.single == len(opt_params)
    _assert_states_bitwise_equal(opt, opt_params, ref_params, ref_state)


def test_bfloat16_would_not_survive_the_batched_path() -> None:
    """The measurement the dtype allowlist rests on, asserted rather than trusted.

    The discriminating op is the moment decay `mul_(beta1)`: `_foreach_mul_`
    with a python-float scalar rounds differently from `Tensor.mul_` on
    bfloat16, so the moments have to start NON-ZERO for the difference to
    exist at all (`0.0 * 0.9` is exact either way).

    If a torch upgrade ever makes this pass bitwise, this assertion is the
    tripwire that says `_FOREACH_EXACT_DTYPES` may widen -- it is deliberately
    a claim about the installed torch, not a permanent law.
    """
    generator = torch.Generator().manual_seed(7)

    def bf(*shape: int) -> Tensor:
        return torch.randn(shape, generator=generator).to(torch.bfloat16)

    grad, exp_avg, exp_avg_sq, param = bf(64, 32), bf(64, 32), bf(64, 32).abs(), bf(64, 32)
    loop = (param.clone(), exp_avg.clone(), exp_avg_sq.clone())
    batch = (param.clone(), exp_avg.clone(), exp_avg_sq.clone())
    kwargs = {
        "step": 4, "lr": _LR, "weight_decay": 0.03,
        "beta1": _BETA1, "beta2": _BETA2, "eps": _EPS,
    }
    _adamw_update_one(loop[0], grad, loop[1], loop[2], **kwargs)  # pyright: ignore[reportArgumentType]
    _adamw_update_foreach(
        [batch[0]], [grad], [batch[1]], [batch[2]], **kwargs,  # pyright: ignore[reportArgumentType]
    )
    assert not all(torch.equal(got, want) for got, want in zip(batch, loop))


def test_duplicate_parameters_in_a_group_fall_back_to_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  # A tied weight reaching one group twice aliases inside a bucket, and
  # `_foreach_*` over a list holding one storage twice is not two sequential
  # updates of it. The whole group must drop to the loop.
    counter = _PathCounter(monkeypatch)
    shared = torch.nn.Parameter(torch.randn(4, 3))
    other = torch.nn.Parameter(torch.randn(2))
    ref_shared = torch.nn.Parameter(shared.detach().clone())
    ref_other = torch.nn.Parameter(other.detach().clone())
    ref_state: dict[Tensor, dict[str, object]] = {}
    with pytest.warns(UserWarning, match="duplicate parameters"):
        opt = AuroraWithAuxAdam(
            [{
                "params": [shared, other, shared],
                "lr": _LR, "weight_decay": 0.02, "use_aurora": False,
            }],
        )
    for step_index in range(3):
        generator = torch.Generator().manual_seed(400 + step_index)
        grad_shared = torch.randn(4, 3, generator=generator)
        grad_other = torch.randn(2, generator=generator)
        shared.grad, ref_shared.grad = grad_shared.clone(), grad_shared.clone()
        other.grad, ref_other.grad = grad_other.clone(), grad_other.clone()
        counter.reset()
        opt.step()
        _reference_adamw_loop(
            [ref_shared, ref_other, ref_shared], ref_state, lr=_LR, weight_decay=0.02,
        )
    assert counter.foreach == 0
    assert counter.single == 3
    assert int(opt.state[shared]["step"]) == 6
    _assert_states_bitwise_equal(
        opt, [shared, other], [ref_shared, ref_other], ref_state,
    )


def test_a_failed_step_does_not_advance_the_step_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step that dies in the batched update must not leave `step` incremented.

    ⚑ `Trainer.train_steps` catches a `RuntimeError` whose message contains
    "CUDA" and RETRIES the whole step up to three times, so an optimizer step
    is NOT the unrecoverable event it looks like. The batched path raises after
    the scan has visited every parameter, where the loop raised part-way
    through -- so a counter committed during the scan would advance the bias
    correction of an ENTIRE group whose moments never moved, and the retry is
    recorded as a success, so nothing downstream would report it.

    The invariant asserted is the one that matters to the retry: a step that
    failed before any tensor was mutated must be indistinguishable from a step
    that never ran.
    """
    base = _make_params()
    opt_params = _clone_params(base)
    ref_params = _clone_params(base)
    ref_state: dict[Tensor, dict[str, object]] = {}
    opt = AuroraWithAuxAdam(
        [{"params": opt_params, "lr": _LR, "weight_decay": 0.03, "use_aurora": False}],
    )

    def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("CUDA error: out of memory")

    grads = _grads_for_step(opt_params, 0)
    for param, grad in zip(opt_params, grads):
        param.grad = None if grad is None else grad.clone()
    before = [p.detach().clone() for p in opt_params]

    monkeypatch.setattr(aurora_module, "_adamw_update_foreach", explode)
    with pytest.raises(RuntimeError, match="CUDA"):
        opt.step()
    monkeypatch.undo()

    assert all(int(opt.state[p].get("step", 0)) == 0 for p in opt_params)
    for index, (got, want) in enumerate(zip(opt_params, before)):
        assert torch.equal(got.detach(), want), f"param {index} mutated by a failed step"

  # Now the retry: one clean step must land exactly where a first clean step
  # would have, moments and all.
    for param, ref_param, grad in zip(opt_params, ref_params, grads):
        param.grad = None if grad is None else grad.clone()
        ref_param.grad = None if grad is None else grad.clone()
    opt.step()
    _reference_adamw_loop(ref_params, ref_state, lr=_LR, weight_decay=0.03)
    _assert_states_bitwise_equal(opt, opt_params, ref_params, ref_state)
    assert all(int(opt.state[p]["step"]) == 1 for p in opt_params)


def test_parameter_without_a_gradient_is_never_touched() -> None:
    used = torch.nn.Parameter(torch.randn(3, 2))
    unused = torch.nn.Parameter(torch.randn(3, 2))
    before = unused.detach().clone()
    opt = AuroraWithAuxAdam(
        [{"params": [used, unused], "lr": _LR, "weight_decay": 0.02, "use_aurora": False}],
    )
    used.grad = torch.randn(3, 2)
    opt.step()
    assert torch.equal(unused.detach(), before)
    assert opt.state.get(unused, {}) == {}


def test_sparse_gradients_are_still_refused() -> None:
    param = torch.nn.Parameter(torch.randn(4, 3))
    opt = AuroraWithAuxAdam(
        [{"params": [param], "lr": _LR, "weight_decay": 0.0, "use_aurora": False}],
    )
    indices = torch.tensor([[0, 1], [1, 2]])
    values = torch.tensor([1.0, 2.0])
    param.grad = torch.sparse_coo_tensor(indices, values, (4, 3))
    with pytest.raises(RuntimeError, match="does not support sparse gradients"):
        opt.step()


def test_foreach_and_loop_helpers_agree_on_a_single_tensor() -> None:
    """The two helpers are one expression written twice; pin them to each other."""
    generator = torch.Generator().manual_seed(11)
    for step in (1, 2, 17):
        for weight_decay in (0.0, 0.05):
            grad = torch.randn(9, 5, generator=generator)
            exp_avg = torch.randn(9, 5, generator=generator).abs()
            exp_avg_sq = torch.randn(9, 5, generator=generator).abs()
            param = torch.randn(9, 5, generator=generator)
            loop = (param.clone(), exp_avg.clone(), exp_avg_sq.clone())
            batch = (param.clone(), exp_avg.clone(), exp_avg_sq.clone())
            kwargs = {
                "step": step, "lr": _LR, "weight_decay": weight_decay,
                "beta1": _BETA1, "beta2": _BETA2, "eps": _EPS,
            }
            _adamw_update_one(loop[0], grad, loop[1], loop[2], **kwargs)  # pyright: ignore[reportArgumentType]
            _adamw_update_foreach(
                [batch[0]], [grad], [batch[1]], [batch[2]], **kwargs,  # pyright: ignore[reportArgumentType]
            )
            for got, want in zip(batch, loop):
                assert torch.equal(got, want)
