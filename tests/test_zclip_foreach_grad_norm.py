"""`ForeachZClip._compute_grad_norm` must return the parent's float exactly.

Unlike the elementwise AdamW chain in `aurora.py`, this one swaps a REDUCTION:
N separate `Tensor.norm(2)` calls become one `torch._foreach_norm`. Reductions
have an accumulation order, so equality here is a MEASUREMENT, not an argument
from the ops -- which is why these tests compare against `ZClip`'s own method
rather than against a transcription of it.

⚑ What these tests prove is CPU-only. The production device is CUDA, where the
two kernels' reduction trees differ by construction, and this file cannot see
that (a live train owns the GPU). `scratchpad/optforeach_ab.sh` is the
instrument for the CUDA claim; a divergence there means dropping the zclip
commit, not weakening this file.
"""

from __future__ import annotations

from pathlib import Path

import torch
from zclip import ZClip

from chess_anti_engine.train.zclip_foreach import ForeachZClip


class _ParamList:
    """The surface `ZClip._compute_grad_norm` consumes, mirroring `_GradClipScope`."""

    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self._params = params

    def parameters(self):
        return iter(self._params)

    def modules(self):
        return iter(())


def _make_scope(
    *, dtype: torch.dtype = torch.float32, seed: int = 0, with_none_grad: bool = True,
) -> _ParamList:
  # Shapes chosen off the production trunk rather than at random: a 512x512
  # attention matrix, a 512-wide bias/norm vector, the 1858-wide policy head,
  # a smolgen-sized 1024x512, and a scalar (`log_temp` is one).
    generator = torch.Generator().manual_seed(seed)
    shapes: list[tuple[int, ...]] = [(512, 512), (512,), (1858, 512), (1024, 512), ()]
    params = [
        torch.nn.Parameter(torch.randn(shape, generator=generator).to(dtype))
        for shape in shapes
    ]
    for index, param in enumerate(params):
        if with_none_grad and index == 1:
            continue
        param.grad = torch.randn(param.shape, generator=generator).to(dtype)
    return _ParamList(params)


def _parent_norm(scope: _ParamList) -> float:
    """The installed `ZClip`'s own implementation, called unmodified."""
    return ZClip._compute_grad_norm(ZClip(), scope)


def test_foreach_grad_norm_is_bitwise_equal_to_the_parent() -> None:
    for seed in range(4):
        scope = _make_scope(seed=seed)
        want = _parent_norm(scope)
        got = ForeachZClip()._compute_grad_norm(scope)
        assert got == want, f"seed {seed}: {got!r} != {want!r}"
        assert got > 0.0


def test_foreach_grad_norm_is_bitwise_equal_in_float64() -> None:
    scope = _make_scope(dtype=torch.float64, seed=7)
    want = _parent_norm(scope)
    got = ForeachZClip()._compute_grad_norm(scope)
    assert got == want


def test_parameters_without_gradients_are_skipped_by_both() -> None:
  # The parent filters on `param.grad is not None`; so must the batched list,
  # or the norm silently gains a term. Same scope, one grad cleared.
    scope = _make_scope(seed=3, with_none_grad=False)
    params = list(scope.parameters())
    with_all = ForeachZClip()._compute_grad_norm(scope)
    params[2].grad = None
    assert ForeachZClip()._compute_grad_norm(scope) == _parent_norm(scope)
    assert ForeachZClip()._compute_grad_norm(scope) != with_all


def test_no_gradients_at_all_returns_zero_like_the_parent() -> None:
    scope = _make_scope(seed=5)
    for param in scope.parameters():
        param.grad = None
    assert ForeachZClip()._compute_grad_norm(scope) == 0.0
    assert _parent_norm(scope) == 0.0


def test_mixed_grad_dtype_is_cast_the_same_way() -> None:
  # The parent casts every grad to the FIRST parameter's dtype before
  # measuring. A grad stored at another precision must go through the same
  # `.to(dtype)`, not be measured where it lies.
    first = torch.nn.Parameter(torch.randn(64, 32))
    second = torch.nn.Parameter(torch.randn(64, 32, dtype=torch.float64))
    generator = torch.Generator().manual_seed(21)
    first.grad = torch.randn(64, 32, generator=generator)
    second.grad = torch.randn(64, 32, generator=generator, dtype=torch.float64)
    scope = _ParamList([first, second])
    assert ForeachZClip()._compute_grad_norm(scope) == _parent_norm(scope)


def test_subclass_keeps_the_parent_clipping_behaviour() -> None:
  # Only `_compute_grad_norm` is overridden; `step` and the EMA must still be
  # the parent's. Drive both through an identical warmup and compare the state
  # the rest of `Trainer` reads off `self.zclip`.
    kwargs = {"mode": "zscore", "alpha": 0.97, "z_thresh": 2.5, "warmup_steps": 3}
    baseline = ZClip(max_grad_norm=None, **kwargs)  # pyright: ignore[reportArgumentType]
    batched = ForeachZClip(max_grad_norm=None, **kwargs)  # pyright: ignore[reportArgumentType]
    for seed in range(6):
        scope_a = _make_scope(seed=seed)
        scope_b = _make_scope(seed=seed)
        assert baseline.step(scope_a) == batched.step(scope_b)
    assert batched.initialized is baseline.initialized
    assert batched.mean == baseline.mean
    assert batched.var == baseline.var
    assert batched.buffer == baseline.buffer


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32, device=self.head.weight.device),
        }


def test_trainer_actually_gets_the_batched_clipper(tmp_path: Path) -> None:
    """Take-effect check: an import that is never used is this repo's usual defect.

    Asserted on the constructed `Trainer`'s own attribute, not on the name the
    trainer module imported.
    """
    from chess_anti_engine.train import Trainer

    trainer = Trainer(
        _TinyModel(), device="cpu", lr=1e-3, optimizer="aurora", use_amp=False,
        log_dir=tmp_path, tb_log_interval=1000, prefetch_batches=False,
    )
    assert isinstance(trainer.zclip, ForeachZClip)
    assert issubclass(ForeachZClip, ZClip)
