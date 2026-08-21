"""Target-blend wrapper serving ``policy_target = (1-a)*t0 + a*q`` on eligible rows.

``t0`` is the stored search-visit target exactly as live training consumes it;
``q`` is the stored same-position SF soft teacher (``sf_p0_policy_target``),
present only where ``has_sf_p0``. Non-eligible rows keep ``t0`` untouched, so
every row stays ONE row of weight 1 — the literal target blend from the
2026-08-19 dose-ladder prereg, chosen over a ``w_sf_own`` loss blend precisely
because the additive-CE route hands eligible rows ``1/(1-a)`` of the policy
gradient (3.33x at a=0.7), a confound with nothing to recommend it.

Both inputs are normalized distributions and the blend is convex, so no
renormalization happens here; renormalizing would silently repair a corrupt
stored target instead of surfacing it.

⚑ LIFETIME CONTRACT — the wrapper is a per-``train_steps``-call view, never a
long-lived replacement for the buffer. ``__getattr__`` delegates attribute
READS only: an attribute WRITE (``wrapper.capacity = n``,
``wrapper.sf_gap_priority_weight = w`` — both of which the live trial performs
on its buffer every iteration) would land on the wrapper and shadow the inner
buffer's value silently, which for ``capacity`` means the sliding window stops
being enforced. The live wiring therefore builds a fresh wrapper at the
``trainer.train_steps`` call boundary each iteration (from that iteration's
freshly reloaded config) and passes the RAW buffer everywhere else. That
rebuild is also what makes ``sf_p0_blend_alpha`` live-reloadable in both
directions, including 0 <-> positive, without any push machinery.

``a=0`` is OFF and must stay bitwise identical to no wrapper at all, which is
why activation requires a STRICTLY positive alpha — a zero goes through the
unwrapped path and cannot drift.
"""
from __future__ import annotations

from typing import Any


class SfP0BlendBuffer:
    """Blend the stored SF soft teacher into ``policy_target`` at sample time.

    Counters (``blended_rows`` / ``total_rows``) accumulate over the wrapper's
    lifetime — one training phase under the live wiring — so the realized
    eligible-row fraction ``f`` can be reported from the object the trainer
    actually consumed, never from the config that asked for it.
    """

    def __init__(self, inner: Any, alpha: float):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError(
                f"sf_p0_blend_alpha={alpha!r}: activation needs 0 < a <= 1 "
                "(a=0 is OFF and runs UNWRAPPED, bitwise)"
            )
        self._inner = inner
        self.alpha = float(alpha)
        self.blended_rows = 0
        self.total_rows = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __len__(self) -> int:
        return len(self._inner)

    def sample_batch_arrays(self, batch_size: int, **kw: Any) -> dict:
        arrs = self._inner.sample_batch_arrays(batch_size, **kw)
        t0 = arrs.get("policy_target")
        q = arrs.get("sf_p0_policy_target")
        has_q = arrs.get("has_sf_p0")
        if t0 is None or q is None or has_q is None:
            raise RuntimeError(
                f"sf_p0_blend_alpha={self.alpha} but the sampled batch lacks "
                "policy_target/sf_p0_policy_target/has_sf_p0 — this pool "
                "predates the sf_p0 teacher, so training would silently run "
                "pure t0 and read as a null"
            )
        mask = has_q.astype(bool)
        self.total_rows += int(t0.shape[0])
        self.blended_rows += int(mask.sum())
        blended = t0.copy()
        a = self.alpha
        blended[mask] = ((1.0 - a) * t0[mask] + a * q[mask]).astype(
            blended.dtype, copy=False,
        )
        out = dict(arrs)
        out["policy_target"] = blended
        return out
