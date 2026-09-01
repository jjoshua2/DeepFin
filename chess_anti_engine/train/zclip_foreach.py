"""`ZClip` with its per-parameter grad-norm listcomp batched into one kernel.

py-spy on a live `scripts/lc0_control_train.py` step (45 s, 3585 samples) put
9.7% of MainThread's active time inside the installed `zclip` package's
`_compute_grad_norm`, which walks every parameter and launches a separate
`.norm(2)` reduction for each. At 18% average GPU utilisation the step is
Python-launch-bound, so collapsing that walk into a single `_foreach_norm` is
worth the subclass.

The package is a third-party install and is NOT edited; this overrides the one
method and leaves everything else -- warmup, the EMA, `_compute_clip_val`,
`apply_in_place_clipping` -- to the parent.

⚑⚑ THIS IS THE HALF OF THE FOREACH WORK WHOSE BITWISE IDENTITY IS NOT PROVEN ON
THE PRODUCTION DEVICE. A vector 2-norm is a REDUCTION, so unlike the elementwise
AdamW chain in `aurora.py` it has an accumulation ORDER, and `_foreach_norm`'s
chunked multi-tensor kernel need not use the same one as `linalg_vector_norm`.
Measured equal bit for bit on CPU, float32 and float64, over the shapes this
model actually has (`tests/test_zclip_foreach_grad_norm.py`); NOT measured on
CUDA, where the two kernels' reduction trees differ by construction. A last-bit
difference here is not cosmetic -- it moves the clip threshold, which moves the
gradients, which moves the weights -- so the banked A/B in
`scratchpad/optforeach_ab.sh` is what decides whether this ships. If it
diverges, drop this module and keep the Aurora change; they are separate
commits for exactly that reason.
"""

from __future__ import annotations

from typing import Any

import torch
from zclip import ZClip, is_fsdp_model


class ForeachZClip(ZClip):
    """`ZClip` whose local grad-norm is one batched reduction instead of N."""

    def _compute_grad_norm(self, model: Any) -> float:
  # The sharded branch is left to the parent verbatim: it ends in an
  # `all_reduce`, and this repo never reaches it (training is
  # single-process, and `_GradClipScope.modules()` is empty by design so
  # the probe stays False). Batching a path we cannot exercise would be
  # an unreviewable change to distributed numerics.
        if is_fsdp_model(model):
            return super()._compute_grad_norm(model)

  # `first_param` reproduces the parent's dtype/device resolution exactly,
  # including its `StopIteration` on a parameter-less model -- the caller
  # already guarantees a non-empty scope (`Trainer.__init__` keeps
  # `_grad_clip_scope` None rather than empty), and inventing a different
  # answer here would hide that guarantee failing.
        first_param = next(model.parameters())
        dtype = first_param.dtype
        device = first_param.device

        grads = [
            param.grad.to(dtype) for param in model.parameters() if param.grad is not None
        ]
        if not grads:
            return 0.0
  # From here down this is the parent's tail, unchanged: stack the
  # per-tensor norms, sum their squares, take the root. Only the way the
  # per-tensor norms are produced differs.
        grad_norms_tensor = torch.stack(list(torch._foreach_norm(grads, 2))).to(device)
        total_norm = torch.sqrt(torch.sum(torch.pow(grad_norms_tensor, 2)))
        return total_norm.item()
