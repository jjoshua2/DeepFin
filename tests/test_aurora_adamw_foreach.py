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
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from chess_anti_engine.train import aurora as aurora_module
from chess_anti_engine.train.aurora import (
    AuroraWithAuxAdam,
    OptimizerStepFailed,
    _adamw_update_foreach,
    _adamw_update_one,
)
from chess_anti_engine.train.soda import SODAWeightDecayWrapper

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
    lr: float = _LR,
) -> tuple[AuroraWithAuxAdam, list[torch.nn.Parameter], list[torch.nn.Parameter], dict[Tensor, dict[str, object]]]:
    base = _make_params(dtype)
    opt_params = _clone_params(base)
    ref_params = _clone_params(base)
    ref_state: dict[Tensor, dict[str, object]] = {}
    opt = AuroraWithAuxAdam(
        [{"params": opt_params, "lr": lr, "weight_decay": weight_decay, "use_aurora": False}],
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
            ref_params, ref_state, lr=lr, weight_decay=weight_decay,
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


def test_two_parameters_over_one_storage_fall_back_to_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex P2 on the identity-only guard: two DISTINCT Parameter objects (a
    tensor and a view of it) share every element, so batching them is not the
    loop's ordered sequential update. Weight decay on is what makes the two
    orders differ -- (P(1-x) - u_a)(1-x) - u_b against P(1-x)^2 - u_a - u_b --
    so it is on here; the loop result is the frozen reference either way."""
    counter = _PathCounter(monkeypatch)
    storage = torch.randn(4, 3)
    first = torch.nn.Parameter(storage)
    second = torch.nn.Parameter(storage.view(-1))
    assert first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    ref_storage = storage.detach().clone()
    ref_first = torch.nn.Parameter(ref_storage)
    ref_second = torch.nn.Parameter(ref_storage.view(-1))
    ref_state: dict[Tensor, dict[str, object]] = {}
    opt = AuroraWithAuxAdam(
        [{"params": [first, second], "lr": _LR, "weight_decay": 0.03, "use_aurora": False}],
    )
    for step_index in range(3):
        generator = torch.Generator().manual_seed(500 + step_index)
        grad_first = torch.randn(4, 3, generator=generator)
        grad_second = torch.randn(12, generator=generator)
        first.grad, ref_first.grad = grad_first.clone(), grad_first.clone()
        second.grad, ref_second.grad = grad_second.clone(), grad_second.clone()
        counter.reset()
        opt.step()
        _reference_adamw_loop(
            [ref_first, ref_second], ref_state, lr=_LR, weight_decay=0.03,
        )
    assert counter.foreach == 0
    assert counter.single == 2
    assert opt.last_adamw_stats["adamw_loop_params"] == 2.0
    _assert_states_bitwise_equal(
        opt, [first, second], [ref_first, ref_second], ref_state,
    )


def test_disjoint_views_of_one_storage_stay_batched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Same storage, no shared element: nothing aliases, so the batched path is
  # still the loop bit for bit and the guard must not over-fire.
    counter = _PathCounter(monkeypatch)
    storage = torch.randn(8)
    head = torch.nn.Parameter(storage[:4])
    tail = torch.nn.Parameter(storage[4:])
    ref_storage = storage.detach().clone()
    ref_head = torch.nn.Parameter(ref_storage[:4])
    ref_tail = torch.nn.Parameter(ref_storage[4:])
    ref_state: dict[Tensor, dict[str, object]] = {}
    opt = AuroraWithAuxAdam(
        [{"params": [head, tail], "lr": _LR, "weight_decay": 0.03, "use_aurora": False}],
    )
    for step_index in range(3):
        generator = torch.Generator().manual_seed(600 + step_index)
        grads = torch.randn(8, generator=generator)
        head.grad, ref_head.grad = grads[:4].clone(), grads[:4].clone()
        tail.grad, ref_tail.grad = grads[4:].clone(), grads[4:].clone()
        counter.reset()
        opt.step()
        _reference_adamw_loop([ref_head, ref_tail], ref_state, lr=_LR, weight_decay=0.03)
    assert counter.foreach == 1
    assert counter.single == 0
    _assert_states_bitwise_equal(opt, [head, tail], [ref_head, ref_tail], ref_state)


def test_alias_guard_sees_a_strided_view_s_full_extent() -> None:
  # A non-contiguous view covers more storage than it has elements; the guard
  # measures the byte span it touches, not `numel()`. Two interleaved stride-2
  # views share no element but their spans overlap, and the guard is
  # deliberately CONSERVATIVE there: it reads them as aliasing and takes the
  # loop, which is exactly right, rather than proving element-disjointness.
    storage = torch.randn(8)
    even = torch.nn.Parameter(storage[::2])
    odd = torch.nn.Parameter(storage[1::2])
    assert aurora_module._adamw_group_aliases([even, odd]) is True
    tail = torch.nn.Parameter(storage[7:])
    assert aurora_module._adamw_group_aliases([even, tail]) is False, "even covers 0..6, tail is 7"
    assert aurora_module._adamw_group_aliases([even, torch.nn.Parameter(storage[6:])]) is True
    assert aurora_module._adamw_group_aliases([even, even]) is True


def test_a_failed_step_does_not_advance_the_step_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step that dies in the batched update must not leave `step` incremented.

    The batched path raises after the scan has visited every parameter, where
    the loop raised part-way through -- so a counter committed during the scan
    would advance the bias correction of an ENTIRE group whose moments never
    moved. `Trainer.train_steps` no longer retries an optimizer-phase failure
    (it ends the trial as `OptimizerStepFailed`, P2-3 below), so the invariant
    asserted is what makes the post-mortem state readable: a step that failed
    before any tensor was mutated is indistinguishable from a step that never
    ran, and a second `opt.step()` from there is one clean step.
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


@pytest.mark.parametrize("lr", [_LR, 1.0])
def test_foreach_and_loop_helpers_agree_on_a_single_tensor(lr: float) -> None:
    """The two helpers are one expression written twice; pin them to each other.

    Run at lr=1.0 as well as the production-sized lr: at 3e-4 a last-bit
    denominator difference vanishes below the parameter's ulp, so only the
    large-lr case has the power to see a finisher that rounds differently.
    """
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
                "step": step, "lr": lr, "weight_decay": weight_decay,
                "beta1": _BETA1, "beta2": _BETA2, "eps": _EPS,
            }
            _adamw_update_one(loop[0], grad, loop[1], loop[2], **kwargs)  # pyright: ignore[reportArgumentType]
            _adamw_update_foreach(
                [batch[0]], [grad], [batch[1]], [batch[2]], **kwargs,  # pyright: ignore[reportArgumentType]
            )
            for got, want in zip(batch, loop):
                assert torch.equal(got, want)


# --- P2-1: what a mid-chain failure leaves behind -----------------------------


def _raise_cuda_once(monkeypatch: pytest.MonkeyPatch, name: str) -> list[int]:
    """Make `torch.<name>` raise a retryable CUDA error on its FIRST call only.

    Returns the call log so a test can assert the injection actually fired --
    an injection that never ran would make the "recovered" assertions below
    hold vacuously, since a clean step also matches the reference.
    """
    real = getattr(torch, name)
    calls: list[int] = []

    def flaky(*args: object, **kwargs: object) -> object:
        calls.append(len(calls))
        if len(calls) == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 165.6 MiB")
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, name, flaky)
    return calls


  # ⚑ GATE POWER. At a production-sized lr (3e-4) a last-bit change to the
  # DENOMINATOR is absorbed below the parameter's ulp: the review's mutant
  # "add `eps` BEFORE the bias-correction divide in `_adamw_finish_one`"
  # survived every test at lr=3e-4 (5/135 denominators differed, 0/135
  # params). Helper-level tests own their hyperparameters, so the ones that
  # pin the finisher also run at lr=1.0, where the same mutant moves 5/135
  # parameters and is caught.
_GATE_LRS = [_LR, 1.0]


@pytest.mark.parametrize("lr", _GATE_LRS)
def test_a_denominator_allocation_failure_is_finished_per_tensor_bitwise(
    monkeypatch: pytest.MonkeyPatch, lr: float,
) -> None:
    """`_foreach_sqrt` is the only allocation in the batched chain and it fires
    AFTER the four in-place moment kernels. Left alone, the trainer's CUDA
    retry would decay/accumulate the whole bucket's moments a second time and
    record the step as clean. Instead the bucket is finished per tensor from
    the moments it already has, and the result is the loop's, bit for bit.
    """
    calls = _raise_cuda_once(monkeypatch, "_foreach_sqrt")
    counter = _PathCounter(monkeypatch)
    opt, opt_params, ref_params, ref_state = _run_paired(
        weight_decay=0.03, steps=3, counter=counter, lr=lr,
    )
    assert len(calls) == 3, "the injected _foreach_sqrt failure never fired"
  # No exception escaped `step`, the batched helper was the path every step,
  # and the recovery is counted exactly once -- on step 1, the failing one.
    assert counter.foreach == 1
    assert counter.single == 0
    _assert_states_bitwise_equal(opt, opt_params, ref_params, ref_state)
    assert all(int(opt.state[p]["step"]) == 3 for p in opt_params)
    assert opt.last_adamw_stats["adamw_foreach_recoveries"] == 0.0, (
        "the LAST step (a clean one) must not carry the earlier recovery"
    )
  # ...but the lifetime counter does, which is what the trainer differences
  # over a window so a step-1 recovery is not lost behind two clean steps.
    assert opt.adamw_foreach_recoveries_total == 1


def test_the_recovery_counter_reads_one_on_the_step_that_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _raise_cuda_once(monkeypatch, "_foreach_sqrt")
    finish_calls: list[int] = []
    real_finish = aurora_module._adamw_finish_one

    def counted_finish(*args: object, **kwargs: object) -> None:
        finish_calls.append(1)
        real_finish(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(aurora_module, "_adamw_finish_one", counted_finish)
    opt, opt_params, ref_params, ref_state = _run_paired(weight_decay=0.03, steps=1)
    assert opt.last_adamw_stats["adamw_foreach_recoveries"] == 1.0
    assert opt.last_adamw_stats["adamw_foreach_buckets"] == 1.0
    assert len(finish_calls) == len(opt_params), "every tensor of the bucket is finished"
    _assert_states_bitwise_equal(opt, opt_params, ref_params, ref_state)


def test_the_recovery_path_is_never_taken_on_a_healthy_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The successful path's arithmetic is untouched by the recovery hook: the
    per-tensor finisher is not called at all on the batched path, and the
    tensors still equal the frozen reference loop bit for bit."""
    finish_calls: list[int] = []
    real_finish = aurora_module._adamw_finish_one

    def counted_finish(*args: object, **kwargs: object) -> None:
        finish_calls.append(1)
        real_finish(*args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(aurora_module, "_adamw_finish_one", counted_finish)
    opt, opt_params, ref_params, ref_state = _run_paired(weight_decay=0.03, steps=4)
    assert finish_calls == []
    assert opt.last_adamw_stats["adamw_foreach_recoveries"] == 0.0
    _assert_states_bitwise_equal(opt, opt_params, ref_params, ref_state)


def test_a_failure_inside_an_in_place_moment_kernel_is_not_recovered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The honest residue, pinned rather than papered over.

    A CUDA error raised by an IN-PLACE kernel (here the `exp_avg_sq`
    accumulate, the last one before the allocation) is not the allocation
    case: the recovery hook does not fire, the error propagates (as
    `OptimizerStepFailed`) with its "CUDA" message intact, `step` is NOT
    committed, and the parameters and `exp_avg` HAVE been mutated. A second
    `opt.step()` from there -- what the trainer's CUDA retry USED to do, and
    the reason P2-3 below makes it not retry -- re-applies the weight-decay
    multiply and the `exp_avg` decay on top of that. This test asserts that
    exact double-applied state, so the day it changes the change is a
    deliberate one.
    """
    calls = _raise_cuda_once(monkeypatch, "_foreach_addcmul_")
    weight_decay = 0.03
    base = _make_params()
    opt_params = _clone_params(base)
    opt = AuroraWithAuxAdam(
        [{"params": opt_params, "lr": _LR, "weight_decay": weight_decay, "use_aurora": False}],
    )
    first_grads = _grads_for_step(opt_params, 0)
    for param, grad in zip(opt_params, first_grads):
        param.grad = None if grad is None else grad.clone()
    with pytest.raises(RuntimeError, match="CUDA") as excinfo:
        opt.step()
    assert not isinstance(excinfo.value, aurora_module._DenominatorAllocationFailed)
    assert calls == [0]
  # Nothing committed, but the bucket IS mutated: params decayed once and
  # `exp_avg` holds the first batch's accumulate; `exp_avg_sq` is still zero.
    assert all(int(opt.state[p].get("step", 0)) == 0 for p in opt_params)
    for param, grad in zip(opt_params, first_grads):
        assert grad is not None
        assert torch.equal(opt.state[param]["exp_avg"], grad * (1.0 - _BETA1))
        assert torch.equal(opt.state[param]["exp_avg_sq"], torch.zeros_like(grad))

  # The retry, as `Trainer.train_steps` would run it: a fresh batch.
    second_grads = _grads_for_step(opt_params, 1)
    for param, grad in zip(opt_params, second_grads):
        param.grad = None if grad is None else grad.clone()
    opt.step()
    assert all(int(opt.state[p]["step"]) == 1 for p in opt_params)

  # Expected: the reference loop run ONCE on the second batch, starting from
  # the mutated state the failure left -- i.e. the double application.
    ref_params = _clone_params(base)
    ref_state: dict[Tensor, dict[str, object]] = {}
    for ref_param, grad in zip(ref_params, first_grads):
        assert grad is not None
        with torch.no_grad():
            ref_param.mul_(1.0 - _LR * weight_decay)
        exp_avg = torch.zeros_like(grad)
        exp_avg.mul_(_BETA1).add_(grad, alpha=1.0 - _BETA1)
        ref_state[ref_param] = {
            "step": 0, "exp_avg": exp_avg, "exp_avg_sq": torch.zeros_like(grad),
        }
    for ref_param, grad in zip(ref_params, second_grads):
        ref_param.grad = None if grad is None else grad.clone()
    _reference_adamw_loop(ref_params, ref_state, lr=_LR, weight_decay=weight_decay)
    _assert_states_bitwise_equal(opt, opt_params, ref_params, ref_state)

  # ...and it is NOT what a clean step on the second batch would have given.
    clean_params = _clone_params(base)
    clean_state: dict[Tensor, dict[str, object]] = {}
    for clean_param, grad in zip(clean_params, second_grads):
        clean_param.grad = None if grad is None else grad.clone()
    _reference_adamw_loop(clean_params, clean_state, lr=_LR, weight_decay=weight_decay)
    assert not torch.equal(opt_params[0].detach(), clean_params[0].detach())


# --- P2-2: the production observation that the batched path ran --------------


def test_last_adamw_stats_round_trip_the_batched_path() -> None:
    opt, opt_params, _ref_params, _ref_state = _run_paired(weight_decay=0.03, steps=2)
    assert opt.last_adamw_stats == {
        "adamw_foreach_buckets": 1.0,
        "adamw_foreach_params": float(len(opt_params)),
        "adamw_loop_params": 0.0,
        "adamw_foreach_recoveries": 0.0,
    }


def test_last_adamw_stats_count_buckets_on_a_staggered_step() -> None:
    opt, opt_params, _ref_params, _ref_state = _run_paired(
        weight_decay=0.03, steps=3, skip_grads={0: {2}, 1: {2}},
    )
    assert opt.last_adamw_stats["adamw_foreach_buckets"] == 2.0
    assert opt.last_adamw_stats["adamw_foreach_params"] == float(len(opt_params))
    assert opt.last_adamw_stats["adamw_loop_params"] == 0.0


def test_last_adamw_stats_read_the_loop_when_the_batched_path_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  # The forced fallback: an empty allowlist routes every tensor to the loop,
  # and the row must SAY so -- foreach 0 / loop N -- rather than read like a
  # healthy step. This is the observation P2-2 exists for.
    monkeypatch.setattr(aurora_module, "_FOREACH_EXACT_DTYPES", frozenset())
    opt, opt_params, ref_params, ref_state = _run_paired(weight_decay=0.03, steps=2)
    assert opt.last_adamw_stats == {
        "adamw_foreach_buckets": 0.0,
        "adamw_foreach_params": 0.0,
        "adamw_loop_params": float(len(opt_params)),
        "adamw_foreach_recoveries": 0.0,
    }
    _assert_states_bitwise_equal(opt, opt_params, ref_params, ref_state)


def test_last_adamw_stats_exclude_parameters_without_a_gradient() -> None:
    used = torch.nn.Parameter(torch.randn(3, 2))
    unused = torch.nn.Parameter(torch.randn(3, 2))
    opt = AuroraWithAuxAdam(
        [{"params": [used, unused], "lr": _LR, "weight_decay": 0.02, "use_aurora": False}],
    )
    used.grad = torch.randn(3, 2)
    opt.step()
    assert opt.last_adamw_stats["adamw_foreach_params"] == 1.0
    assert opt.last_adamw_stats["adamw_loop_params"] == 0.0


def test_soda_wrapper_forwards_last_adamw_stats() -> None:
  # Production may wrap the optimizer; the trainer harvests off the wrapper,
  # so a missing passthrough is a row that silently reads all zeros.
    params = _make_params()
    base_opt = AuroraWithAuxAdam(
        [{"params": params, "lr": _LR, "weight_decay": 0.03, "use_aurora": False}],
    )
    wrapped = SODAWeightDecayWrapper(base_opt)
    for param, grad in zip(params, _grads_for_step(params, 0)):
        param.grad = None if grad is None else grad.clone()
    wrapped.step()
    assert wrapped.last_adamw_stats == base_opt.last_adamw_stats
    assert wrapped.last_adamw_stats["adamw_foreach_params"] == float(len(params))


def test_every_adamw_stat_key_is_a_train_metrics_field_and_a_ray_column() -> None:
  # `_build_metrics(**last_adamw_stats)` is a keyword splat: a key the
  # dataclass does not declare is a TypeError mid-iteration, and a field the
  # Ray row does not list reaches TensorBoard only.
    import dataclasses

    from chess_anti_engine.train import trainer as trainer_module
    from chess_anti_engine.tune import trainable_report

    opt, _p, _r, _s = _run_paired(weight_decay=0.0, steps=1)
    emitted = set(opt.last_adamw_stats)
    assert emitted == {
        "adamw_foreach_buckets", "adamw_foreach_params",
        "adamw_loop_params", "adamw_foreach_recoveries",
    }
    fields = {f.name for f in dataclasses.fields(trainer_module.TrainMetrics)}
    assert emitted <= fields
    assert emitted <= set(trainable_report._train_metrics_dict(None))


def test_ray_row_round_trips_the_adamw_path_values() -> None:
    from chess_anti_engine.train import trainer as trainer_module
    from chess_anti_engine.tune import trainable_report

    metrics = trainer_module.TrainMetrics(
        **dict.fromkeys(
            (
                "loss", "policy_loss", "soft_policy_loss", "future_policy_loss",
                "wdl_loss", "sf_move_loss", "sf_move_acc", "sf_eval_loss",
                "categorical_loss", "volatility_loss", "sf_volatility_loss",
                "moves_left_loss",
            ),
            0.0,
        ),
        adamw_foreach_buckets=2.0,
        adamw_foreach_params=431.0,
        adamw_loop_params=7.0,
        adamw_foreach_recoveries=1.0,
    )
    row = trainable_report._train_metrics_dict(metrics)
  # Values, not membership: a column wired to the wrong field would still be
  # present.
    assert row["adamw_foreach_buckets"] == 2.0
    assert row["adamw_foreach_params"] == 431.0
    assert row["adamw_loop_params"] == 7.0
    assert row["adamw_foreach_recoveries"] == 1.0
    assert set(trainable_report._train_metrics_dict(None)) == set(row)
    for key in ("adamw_foreach_buckets", "adamw_foreach_params",
                "adamw_loop_params", "adamw_foreach_recoveries"):
        assert trainable_report._TRAIN_METRIC_DEFAULTS[key] == 0.0


class _MatrixAndHeadModel(torch.nn.Module):
    """One Aurora-owned matrix plus AdamW tensors (a head weight and bias)."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(8, 8, bias=False)])
        self.head = torch.nn.Linear(8, 3)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32, device=self.head.weight.device),
        }


def _adamw_group_sizes(opt: torch.optim.Optimizer) -> int:
    return sum(
        len(group["params"]) for group in opt.param_groups if not group.get("use_aurora")
    )


@pytest.mark.parametrize("force_loop", [False, True])
def test_train_steps_carries_adamw_stats_onto_the_step_metrics(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch, force_loop: bool,
) -> None:
    """The whole production chain: `AuroraWithAuxAdam.step` writes
    `last_adamw_stats`, `train_steps` splats it into `_build_metrics`, and the
    fields exist on `TrainMetrics` to receive it. Read from the CONSUMER's
    metrics object, not the optimizer's. With the allowlist emptied the same
    row must flip to foreach 0 / loop N."""
    from chess_anti_engine.train import trainer as trainer_module
    from chess_anti_engine.train.trainer import Trainer

    if force_loop:
        monkeypatch.setattr(aurora_module, "_FOREACH_EXACT_DTYPES", frozenset())
    torch.manual_seed(4242)
    trainer = Trainer(
        _MatrixAndHeadModel(),
        device="cpu", lr=1e-3, optimizer="aurora", use_amp=False,
        log_dir=cast(Any, tmp_path), tb_log_interval=1000, prefetch_batches=False,
    )
    every_param = list(trainer.model.parameters())

    def fake_compute_loss(out: Any, batch: Any, **kwargs: Any) -> dict[str, Tensor]:
        del out, batch, kwargs
        total = cast(Tensor, sum((t * t).sum() for t in every_param))
        losses: dict[str, Tensor] = {"total": total}
        losses.update(dict.fromkeys(
            (
                "policy_ce", "soft_policy_ce", "future_policy_ce", "wdl_ce", "sf_move_ce",
                "sf_eval_ce", "categorical_ce", "volatility", "sf_volatility", "moves_left",
            ),
            total.detach(),
        ))
        return losses

    monkeypatch.setattr(trainer_module, "compute_loss", fake_compute_loss)
    monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda out, batch: {})

    def fake_batches(buf: Any, **kwargs: Any):
        del buf, kwargs
        while True:
            yield {"x": torch.zeros((1, 4, 8, 8))}

    monkeypatch.setattr(trainer, "_iter_prefetched_batches", fake_batches)

    expected = _adamw_group_sizes(trainer.opt)
    assert expected >= 2, "the fixture must put more than one tensor on AdamW"
    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=2)
    assert metrics.train_steps_done == 2
    if force_loop:
        assert metrics.adamw_foreach_params == 0.0
        assert metrics.adamw_foreach_buckets == 0.0
        assert metrics.adamw_loop_params == float(expected)
    else:
        assert metrics.adamw_foreach_params == float(expected)
        assert metrics.adamw_foreach_buckets >= 1.0
        assert metrics.adamw_loop_params == 0.0
    assert metrics.adamw_foreach_recoveries == 0.0


def test_the_finisher_denominator_matches_the_batched_chain_bitwise() -> None:
    """The denominators themselves, not their effect on the parameter.

    `_adamw_finish_one` and the `_foreach_sqrt` / `_foreach_div_` /
    `_foreach_add_` chain must agree on `denom` bit for bit -- the parameter
    comparison alone cannot see a last-bit denominator change at a small lr.
    Read off the update: with `exp_avg == 1` and `lr == 1`, `step == 1`,
    `param == 0`, the finisher writes `param = -1/denom` exactly, so two
    finishers with different denominators write different parameters.
    """
    generator = torch.Generator().manual_seed(21)
    for step in (1, 3, 40):
        exp_avg_sq = torch.randn(64, 32, generator=generator).abs() * 1e-6
        ones = torch.ones_like(exp_avg_sq)
        bias_correction2 = 1.0 - _BETA2**step
        denom_loop = exp_avg_sq.sqrt() / math.sqrt(max(bias_correction2, 1e-12))
        denom_loop.add_(_EPS)
        denoms = torch._foreach_sqrt([exp_avg_sq])
        torch._foreach_div_(denoms, math.sqrt(max(bias_correction2, 1e-12)))
        torch._foreach_add_(denoms, _EPS)
        assert torch.equal(denom_loop, denoms[0])
      # And the finisher reproduces exactly that denominator: param ends at
      # -(1 / (1 - beta1**step)) / denom, which `addcdiv_` computes from the
      # same `denom` only if the finisher built it the same way.
        param = torch.zeros_like(exp_avg_sq)
        aurora_module._adamw_finish_one(
            param, ones, exp_avg_sq.clone(),
            step=step, lr=1.0, beta1=_BETA1, beta2=_BETA2, eps=_EPS,
        )
        expected = torch.zeros_like(exp_avg_sq).addcdiv_(
            ones, denom_loop, value=-(1.0 / max(1.0 - _BETA1**step, 1e-12)),
        )
        assert torch.equal(param, expected)


def test_train_steps_sums_recoveries_over_the_window_not_the_last_step(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex P2: `last_adamw_stats` is the LAST step's, so a recovery on a
    non-final step of a multi-step window used to publish 0 on the row. The
    injected `_foreach_sqrt` failure fires on the FIRST of three steps; the
    row must still read it."""
    from chess_anti_engine.train import trainer as trainer_module
    from chess_anti_engine.train.trainer import Trainer

    calls = _raise_cuda_once(monkeypatch, "_foreach_sqrt")
    torch.manual_seed(4242)
    trainer = Trainer(
        _MatrixAndHeadModel(),
        device="cpu", lr=1e-3, optimizer="aurora", use_amp=False,
        log_dir=cast(Any, tmp_path), tb_log_interval=1000, prefetch_batches=False,
    )
    every_param = list(trainer.model.parameters())

    def fake_compute_loss(out: Any, batch: Any, **kwargs: Any) -> dict[str, Tensor]:
        del out, batch, kwargs
        total = cast(Tensor, sum((t * t).sum() for t in every_param))
        losses: dict[str, Tensor] = {"total": total}
        losses.update(dict.fromkeys(
            (
                "policy_ce", "soft_policy_ce", "future_policy_ce", "wdl_ce", "sf_move_ce",
                "sf_eval_ce", "categorical_ce", "volatility", "sf_volatility", "moves_left",
            ),
            total.detach(),
        ))
        return losses

    monkeypatch.setattr(trainer_module, "compute_loss", fake_compute_loss)
    monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda out, batch: {})

    def fake_batches(buf: Any, **kwargs: Any):
        del buf, kwargs
        while True:
            yield {"x": torch.zeros((1, 4, 8, 8))}

    monkeypatch.setattr(trainer, "_iter_prefetched_batches", fake_batches)

    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=3)
    assert metrics.train_steps_done == 3
    assert calls, "the injection never fired"
  # The failing step was step 1 of 3 and was recovered, not retried.
    assert metrics.transient_cuda_retry_batches == 0.0
    assert metrics.adamw_foreach_recoveries == 1.0
  # The path counts are still the last step's, and that step was clean.
    assert cast(Any, trainer.opt).last_adamw_stats["adamw_foreach_recoveries"] == 0.0
    assert metrics.adamw_foreach_params == float(_adamw_group_sizes(trainer.opt))

  # A second window starts from the counter where the first left it.
    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=2)
    assert metrics.adamw_foreach_recoveries == 0.0


def test_the_base_loop_had_the_same_retry_residue_param_major(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex P1 on the retry: the residue a mid-step CUDA error leaves for the
    trainer's retry is not new in this PR -- the frozen per-parameter loop at
    the PR base has the same CLASS of hazard, param-major instead of op-major.

    Inject a retryable error into the k-th tensor's `addcdiv_`. The base loop
    commits `step` BEFORE the tensor's ops, so it leaves: tensors < k fully
    stepped and committed; tensor k decayed, moments updated, `step` committed,
    parameter NOT updated; tensors > k untouched. The trainer's retry then
    double-steps everything at or below k and single-steps the rest, recorded
    as one clean step. The PR's batched chain changes the SHAPE of that residue
    (whole bucket at one op) and, unlike the base, does NOT commit `step`
    before the update; it does not introduce the class.
    """
    weight_decay = 0.03
    base = _make_params()
    params = _clone_params(base)
    state: dict[Tensor, dict[str, object]] = {}
    grads = _grads_for_step(params, 0)
    for param, grad in zip(params, grads):
        param.grad = None if grad is None else grad.clone()
    k = 2
    real_addcdiv = torch.Tensor.addcdiv_
    seen: list[int] = []

    def flaky_addcdiv(self: Tensor, *args: object, **kwargs: object) -> Tensor:
        seen.append(len(seen))
        if len(seen) == k + 1:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return real_addcdiv(self, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(torch.Tensor, "addcdiv_", flaky_addcdiv)
    with pytest.raises(RuntimeError, match="CUDA"):
        _reference_adamw_loop(params, state, lr=_LR, weight_decay=weight_decay)
    monkeypatch.undo()

    steps = [int(state.get(p, {}).get("step", 0)) for p in params]  # pyright: ignore[reportArgumentType]
    assert steps == [1, 1, 1, 0, 0, 0], "committed through k, untouched after"
    for index, (param, grad) in enumerate(zip(params, grads)):
        assert grad is not None
        if index < k:
            assert not torch.equal(param.detach(), base[index].detach() * (1.0 - _LR * weight_decay))
        elif index == k:
  # Decayed and moments updated, parameter not stepped: `step` says 1.
            assert torch.equal(param.detach(), base[index].detach() * (1.0 - _LR * weight_decay))
            exp_avg = state[param]["exp_avg"]
            assert isinstance(exp_avg, Tensor)
            assert torch.equal(exp_avg, grad * (1.0 - _BETA1))
        else:
            assert torch.equal(param.detach(), base[index].detach())
            assert param not in state

  # The retry, as the trainer runs it: a replacement batch, one more loop.
    second = _grads_for_step(params, 1)
    for param, grad in zip(params, second):
        param.grad = None if grad is None else grad.clone()
    _reference_adamw_loop(params, state, lr=_LR, weight_decay=weight_decay)
    steps = [int(state[p]["step"]) for p in params]  # pyright: ignore[reportArgumentType]
    assert steps == [2, 2, 2, 1, 1, 1], "double-stepped through k, single-stepped after"


# --- P2-3: optimizer-step failures are not retryable --------------------------
#
# The ledger prereg (2026-09-02 02:20Z, follow-up to #495): a failure INSIDE
# `opt.step()` must END `Trainer.train_steps` as `OptimizerStepFailed`, with
# `transient_cuda_retry_batches` never incremented and the step counter not
# advanced -- while a CUDA failure BEFORE the optimizer step (forward/backward)
# still retries. Every test here reads the retry from its consumer side:
# `_TrainBatchIterator.add_retry_batches` is the one call the retry branch makes
# besides the counter, so a spy on it that stays empty is "the retry did not
# fire" measured at the branch, not inferred from the exception type.


def _boundary_trainer(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, list[int]]:
    """A `Trainer` on the production optimizer with the forward/backward faked
    and the retry branch's `torch.cuda` calls neutralized (no device here, and
    a mutant that re-enables the retry must read as the SPY firing, not as a
    CUDA-less `synchronize()` raising something else). Returns the trainer and
    the list of `add_retry_batches` counts the retry branch handed the
    iterator."""
    from chess_anti_engine.train import trainer as trainer_module
    from chess_anti_engine.train.trainer import Trainer

    torch.manual_seed(4242)
    trainer = Trainer(
        _MatrixAndHeadModel(),
        device="cpu", lr=1e-3, optimizer="aurora", use_amp=False,
        log_dir=cast(Any, tmp_path), tb_log_interval=1000, prefetch_batches=False,
    )
    every_param = list(trainer.model.parameters())

    def fake_compute_loss(out: Any, batch: Any, **kwargs: Any) -> dict[str, Tensor]:
        del out, batch, kwargs
        total = cast(Tensor, sum((t * t).sum() for t in every_param))
        losses: dict[str, Tensor] = {"total": total}
        losses.update(dict.fromkeys(
            (
                "policy_ce", "soft_policy_ce", "future_policy_ce", "wdl_ce", "sf_move_ce",
                "sf_eval_ce", "categorical_ce", "volatility", "sf_volatility", "moves_left",
            ),
            total.detach(),
        ))
        return losses

    monkeypatch.setattr(trainer_module, "compute_loss", fake_compute_loss)
    monkeypatch.setattr(trainer, "_policy_accuracy_stats", lambda out, batch: {})

    def fake_batches(buf: Any, **kwargs: Any):
        del buf, kwargs
        while True:
            yield {"x": torch.zeros((1, 4, 8, 8))}

    monkeypatch.setattr(trainer, "_iter_prefetched_batches", fake_batches)
    monkeypatch.setattr(trainer_module.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(trainer_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(trainer_module.time, "sleep", lambda _seconds: None)

    retries: list[int] = []
    real_add = trainer_module._TrainBatchIterator.add_retry_batches

    def spy_add(self: Any, count: int) -> None:
        retries.append(int(count))
        real_add(self, count)

    monkeypatch.setattr(trainer_module._TrainBatchIterator, "add_retry_batches", spy_add)
    return trainer, retries


def _adamw_steps(opt: torch.optim.Optimizer) -> list[int]:
    return [
        int(opt.state[p].get("step", 0))
        for group in opt.param_groups if not group.get("use_aurora")
        for p in group["params"]
    ]


def test_an_in_place_kernel_failure_on_the_batched_path_does_not_retry(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The residue case of #495 (`..._is_not_recovered` above), driven through
    `train_steps`: the injected `_foreach_addcmul_` CUDA error ends the call
    as `OptimizerStepFailed` naming the bucket, carrying the cause; the retry
    never fires (spy empty, the injection fired exactly once -- a retry would
    have called the kernel again); nothing is committed."""
    trainer, retries = _boundary_trainer(tmp_path, monkeypatch)
    calls = _raise_cuda_once(monkeypatch, "_foreach_addcmul_")
    with pytest.raises(OptimizerStepFailed) as excinfo:
        trainer.train_steps(cast(Any, None), batch_size=1, steps=2)
    exc = excinfo.value
    assert calls == [0], "a retry would have re-run the kernel"
    assert retries == [], "the retry branch handed the iterator replacement batches"
    assert isinstance(exc.__cause__, RuntimeError)
    assert "CUDA out of memory" in str(exc.__cause__)
    assert "CUDA" in str(exc), "the message must keep the cause's text"
    assert exc.step_index == 1
    assert exc.location is not None
    assert exc.location.startswith("batched AdamW bucket")
  # The fixture's AdamW tensors sit in different groups, so the dying bucket
  # is the first one built: one float32 tensor at its first step.
    assert "(1 tensors, torch.float32, cpu, step 1)" in exc.location
    assert trainer.step == 0
    assert _adamw_steps(trainer.opt) == [0] * _adamw_group_sizes(trainer.opt)


def test_a_failure_in_the_per_parameter_loop_does_not_retry(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same boundary, the loop path (allowlist emptied so every tensor takes
    it): the failure is at the first tensor's `addcdiv_`, so the location
    names a per-parameter-loop tensor and no `step` is committed."""
    monkeypatch.setattr(aurora_module, "_FOREACH_EXACT_DTYPES", frozenset())
    trainer, retries = _boundary_trainer(tmp_path, monkeypatch)
    real_addcdiv = torch.Tensor.addcdiv_
    seen: list[int] = []

    def flaky_addcdiv(self: Tensor, *args: object, **kwargs: object) -> Tensor:
        seen.append(len(seen))
        if len(seen) == 1:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return real_addcdiv(self, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(torch.Tensor, "addcdiv_", flaky_addcdiv)
    with pytest.raises(OptimizerStepFailed) as excinfo:
        trainer.train_steps(cast(Any, None), batch_size=1, steps=2)
    monkeypatch.undo()
    exc = excinfo.value
    assert seen == [0], "a retry would have reached `addcdiv_` again"
    assert retries == []
    assert isinstance(exc.__cause__, RuntimeError)
    assert "illegal memory access" in str(exc.__cause__)
    assert exc.step_index == 1
    assert exc.location is not None
    assert "per-parameter loop" in exc.location
    assert trainer.step == 0
    assert _adamw_steps(trainer.opt) == [0] * _adamw_group_sizes(trainer.opt)


def test_a_non_cuda_failure_inside_the_optimizer_step_does_not_retry(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The boundary is keyed by CLASS, not by the "CUDA" substring: a plain
    `RuntimeError` from a moment kernel leaves as `OptimizerStepFailed` too."""
    trainer, retries = _boundary_trainer(tmp_path, monkeypatch)
    real = torch._foreach_addcmul_

    def broken(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("kernel refused the operands")

    monkeypatch.setattr(torch, "_foreach_addcmul_", broken)
    with pytest.raises(OptimizerStepFailed) as excinfo:
        trainer.train_steps(cast(Any, None), batch_size=1, steps=1)
    monkeypatch.setattr(torch, "_foreach_addcmul_", real)
    assert retries == []
    assert "CUDA" not in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert trainer.step == 0


def test_an_untracked_failure_is_wrapped_at_the_trainer_boundary_with_no_retry(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrapper or a foreign optimizer raising straight out of `opt.step()`
    -- nothing annotated it -- is still `OptimizerStepFailed` at the trainer
    boundary, with the step index filled in, the cause attached, no location,
    and no retry. Patched on the INSTANCE, so `AuroraWithAuxAdam.step`'s own
    wrap is bypassed and only the trainer's is under test."""
    trainer, retries = _boundary_trainer(tmp_path, monkeypatch)
    injected = RuntimeError("CUDA error: device-side assert triggered")

    def raw_step(closure: object = None) -> None:
        del closure
        raise injected

    monkeypatch.setattr(trainer.opt, "step", raw_step)
    with pytest.raises(OptimizerStepFailed) as excinfo:
        trainer.train_steps(cast(Any, None), batch_size=1, steps=2)
    exc = excinfo.value
    assert retries == []
    assert exc.__cause__ is injected
    assert exc.step_index == 1
    assert exc.location is None
    assert "CUDA error: device-side assert" in str(exc)
    assert trainer.step == 0


def test_a_cuda_failure_before_the_optimizer_step_still_gets_the_retry(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the prereg: a transient CUDA error raised BEFORE
    `opt.step()` (here in the matrix grad-norm, after backward) is still
    retried with a replacement batch, counted on the row, and the window
    completes."""
    trainer, retries = _boundary_trainer(tmp_path, monkeypatch)
    real_matrix_norm = trainer._matrix_grad_norm
    calls: list[int] = []

    def flaky_matrix_norm() -> float:
        calls.append(len(calls))
        if len(calls) == 1:
            raise RuntimeError("CUDA transient test failure")
        return real_matrix_norm()

    monkeypatch.setattr(trainer, "_matrix_grad_norm", flaky_matrix_norm)
    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=2)
    assert len(calls) == 3, "one discarded attempt plus two committed steps"
    assert retries == [1]
    assert metrics.transient_cuda_retry_batches == 1.0
    assert metrics.train_steps_done == 2
    assert trainer.step == 2


def test_a_denominator_allocation_failure_is_recovered_without_a_retry(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-optimizer recovery is untouched by the boundary: a `_foreach_sqrt`
    allocation failure is finished per tensor INSIDE `step`, so nothing reaches
    the trainer -- no exception, no retry, `adamw_foreach_recoveries == 1`,
    every step committed."""
    trainer, retries = _boundary_trainer(tmp_path, monkeypatch)
    calls = _raise_cuda_once(monkeypatch, "_foreach_sqrt")
    metrics = trainer.train_steps(cast(Any, None), batch_size=1, steps=2)
    assert calls, "the injection never fired"
    assert retries == []
    assert metrics.transient_cuda_retry_batches == 0.0
    assert metrics.adamw_foreach_recoveries == 1.0
    assert metrics.train_steps_done == 2
    assert _adamw_steps(trainer.opt) == [2] * _adamw_group_sizes(trainer.opt)
