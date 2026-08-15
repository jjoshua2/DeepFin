#!/usr/bin/env python3
"""Widen a checkpoint's per-layer FFN to tile-aligned widths, function-preserving.

WHY
---
The FFN hidden width is ``int(embed_dim * ffn_mult)``. Production's per-layer
``ffn_mult`` schedule produces widths like 796, 844, 907, 969 — none of them a
multiple of 128. Inductor's autotune picks ``BLOCK_N=128`` for the majority of
these GEMMs (59 of 101 winners in a full production build), so a width of 796
already executes ``ceil(796/128)*128 = 896`` columns of tiles and masks 100 of
them off. **The arithmetic is already being paid for and the result discarded.**

Rounding the architecture up to the tile boundary turns that discarded work into
real parameters at essentially unchanged cost: +709,632 params (+1.12%) for the
production schedule, with the dominant GEMM already measured at ~86% of BF16
peak, so there is no throughput headroom being claimed here — this buys
CAPACITY, not speed.

FUNCTION PRESERVATION
---------------------
For ``y = W2 @ mish(W1 x + b1) + b2``, a new hidden unit ``j`` contributes
``W2[:, j] * mish(W1[j]·x + b1[j])``. Setting **W2's new columns to exactly
zero** makes the contribution identically zero, so the widened network computes
the same function at init. Gradients still flow, in the right order:

* ``dL/dW2[:, j] = dL/dy * mish(W1[j]·x + b1[j])`` — NONZERO, because W1's new
  rows are randomly initialised. So W2's new columns move first.
* ``dL/dW1[j] = (dL/dy · W2[:, j]) * mish'(...) * x`` — zero at init, nonzero as
  soon as W2[:, j] has moved.

⚑ "Same function" is ALGEBRAIC, not bitwise. Widening changes the GEMM's N
dimension, which changes BLAS blocking and therefore float summation ORDER, so
even the UNTOUCHED units drift by ~2 ULP (measured: max relative difference
2.3e-7 in float32, epsilon ~1.2e-7). Padding with literal zero activations
reproduces the same drift, which is how the cause was identified. Do not write
a bitwise-equality check against a widened checkpoint; it will fail for a
correct implementation.

⚑ This is why W1's new rows must NOT also be zeroed: zero rows would leave
``mish(0) = 0``, killing the W2 gradient too, and both halves would stay dead
forever. A "symmetric, obviously safe" all-zero widening is the failure mode.

OPTIMIZER STATE
---------------
A topology change resets the optimizer unless the state is migrated:
``_remap_optimizer_state_for_new_params`` splices only when the donor is SHORTER
in tensor COUNT, and here the count is unchanged while shapes differ. Left
alone, ``load()`` reinitialises every moment plus the scheduler — and then the
readout measures the optimizer reset, not the widening.

Aurora persists exactly one buffer per parameter here (``momentum_buffer``,
same shape as the parameter; no matrix preconditioners are checkpointed), so
migration is a zero-pad in the new positions, addressed by name through
``opt_param_names`` rather than by position.

ARCH
----
``resume_model_config_from_arch`` takes topology from the CHECKPOINT's ``arch``,
and ``ffn_mult_by_layer`` is not in ``_RESUME_CONFIG_OWNED_ENCODING_KEYS``, so
the new schedule MUST be written into ``arch``. A widened ``state_dict`` with a
stale ``arch`` rebuilds the old widths and the load fails (or, worse, a tolerant
loader quietly skips the mismatched tensors).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

_FFN_IN_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.ffn\.0\.(weight|bias)$")
_FFN_OUT_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.ffn\.2\.weight$")


def align_up(value: int, align: int) -> int:
    """Smallest multiple of *align* that is >= *value*."""
    if int(align) <= 0:
        raise ValueError(f"align must be positive, got {align}")
    a = int(align)
    return -(-int(value) // a) * a


def aligned_ffn_mults(
    mults: Sequence[float], *, embed_dim: int, align: int
) -> tuple[float, ...]:
    """Round each layer's FFN width UP to *align*, expressed back as a mult.

    The returned mults are exact: ``int(embed_dim * m)`` reproduces the aligned
    width bit-for-bit, because the widths are multiples of *align* and the
    quotient is a dyadic rational whenever *align* and *embed_dim* are powers of
    two times a small integer. This is asserted rather than assumed — a mult
    that lands one ULP low would silently build a width one short.
    """
    if int(embed_dim) <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}")
    out: list[float] = []
    for m in mults:
        width = int(int(embed_dim) * float(m))
        target = align_up(width, align)
        mult = target / float(embed_dim)
        got = int(int(embed_dim) * mult)
        if got != target:
            raise ValueError(
                f"ffn_mult {mult!r} does not reproduce width {target} exactly "
                f"(int({embed_dim} * {mult!r}) == {got}); refusing to emit a "
                f"schedule whose realized width differs from the intended one"
            )
        out.append(mult)
    return tuple(out)


def _pad_rows(t: torch.Tensor, new_rows: int, *, fill: torch.Tensor | None) -> torch.Tensor:
    """Grow dim 0 to *new_rows*. New rows come from *fill* (zeros if None)."""
    old = t.shape[0]
    if new_rows == old:
        return t
    if new_rows < old:
        raise ValueError(f"refusing to SHRINK dim0 {old} -> {new_rows}")
    extra_shape = (new_rows - old, *t.shape[1:])
    extra = torch.zeros(extra_shape, dtype=t.dtype, device=t.device) if fill is None else fill
    return torch.cat([t, extra.to(dtype=t.dtype, device=t.device)], dim=0)


def _pad_cols(t: torch.Tensor, new_cols: int) -> torch.Tensor:
    """Grow dim 1 to *new_cols*, filling with EXACT ZEROS (function-preserving)."""
    old = t.shape[1]
    if new_cols == old:
        return t
    if new_cols < old:
        raise ValueError(f"refusing to SHRINK dim1 {old} -> {new_cols}")
    extra = torch.zeros(
        (t.shape[0], new_cols - old, *t.shape[2:]), dtype=t.dtype, device=t.device
    )
    return torch.cat([t, extra], dim=1)


def plan_widths(
    arch: dict[str, Any], *, align: int
) -> tuple[dict[int, tuple[int, int]], tuple[float, ...]]:
    """Return ``({layer: (old_width, new_width)}, new_mults)`` for *arch*.

    Only layers that actually grow appear in the mapping, so a re-run on an
    already-aligned checkpoint is a no-op rather than a rewrite.
    """
    embed_dim = int(arch["embed_dim"])
    mults = arch.get("ffn_mult_by_layer")
    if not mults:
        n = int(arch["num_layers"])
        mults = tuple(float(arch.get("ffn_mult", 2.0)) for _ in range(n))
    mults = tuple(float(m) for m in mults)
    new_mults = aligned_ffn_mults(mults, embed_dim=embed_dim, align=align)
    changes: dict[int, tuple[int, int]] = {}
    for i, (old_m, new_m) in enumerate(zip(mults, new_mults)):
        old_w, new_w = int(embed_dim * old_m), int(embed_dim * new_m)
        if new_w != old_w:
            changes[i] = (old_w, new_w)
    return changes, new_mults


def widen_checkpoint(
    ck: dict[str, Any], *, align: int, seed: int = 0
) -> tuple[dict[str, Any], dict[int, tuple[int, int]]]:
    """Widen *ck* in place to tile-aligned FFN widths. Returns ``(ck, changes)``.

    New ``ffn.0`` rows are drawn N(0, s) where s is the empirical std of the
    TRAINED rows of that same tensor — not the fresh-init scale — so a new unit
    is comparable in magnitude to the units it sits beside. New ``ffn.2``
    columns are exactly zero, which is what makes the widening function-
    preserving; see the module docstring.
    """
    arch = ck.get("arch")
    if not isinstance(arch, dict):
        raise ValueError("checkpoint has no 'arch' dict; cannot widen safely")
    changes, new_mults = plan_widths(arch, align=align)
    if not changes:
        return ck, changes

    gen = torch.Generator().manual_seed(int(seed))
    model = ck["model"]
    names: list[str] = list(ck.get("opt_param_names") or [])
    opt_state = (ck.get("opt") or {}).get("state", {})

    def _opt_entry(param_name: str) -> dict[str, Any] | None:
        if param_name not in names:
            return None
        i = names.index(param_name)
        return opt_state.get(i, opt_state.get(str(i)))

    for key in list(model.keys()):
        m_in, m_out = _FFN_IN_RE.search(key), _FFN_OUT_RE.search(key)
        matched = m_in or m_out
        if matched is None:
            continue
        layer = int(matched.group(1))
        if layer not in changes:
            continue
        _, new_w = changes[layer]
        t = model[key]

        if m_in is not None:
            if m_in.group(2) == "weight":
                std = float(t.float().std())
                extra = torch.randn(
                    (new_w - t.shape[0], t.shape[1]), generator=gen, dtype=torch.float32
                ) * std
                model[key] = _pad_rows(t, new_w, fill=extra)
            else:  # bias: new units start unbiased
                model[key] = _pad_rows(t, new_w, fill=None)
            ent = _opt_entry(key)
            if ent is not None and "momentum_buffer" in ent:
                ent["momentum_buffer"] = _pad_rows(ent["momentum_buffer"], new_w, fill=None)
        else:
            model[key] = _pad_cols(t, new_w)
            ent = _opt_entry(key)
            if ent is not None and "momentum_buffer" in ent:
                ent["momentum_buffer"] = _pad_cols(ent["momentum_buffer"], new_w)

    # ⚑ Without this the resume path rebuilds the OLD widths from the stale arch.
    arch["ffn_mult_by_layer"] = tuple(new_mults)
    return ck, changes


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", required=True, help="input trainer.pt")
    p.add_argument("--out", dest="dst", required=True, help="output trainer.pt (must not exist)")
    p.add_argument("--align", type=int, default=128, help="tile alignment (default: 128)")
    p.add_argument("--seed", type=int, default=0, help="seed for new-row init")
    p.add_argument(
        "--force", action="store_true", help="permit overwriting an existing --out"
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    dst = Path(args.dst)
    if dst.exists() and not args.force:
        print(f"error: {dst} exists (pass --force to overwrite)", file=sys.stderr)
        return 2
    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    before = sum(int(v.numel()) for v in ck["model"].values() if hasattr(v, "numel"))
    ck, changes = widen_checkpoint(ck, align=args.align, seed=args.seed)
    after = sum(int(v.numel()) for v in ck["model"].values() if hasattr(v, "numel"))
    if not changes:
        print("no layers need widening; checkpoint already tile-aligned")
    for layer in sorted(changes):
        old_w, new_w = changes[layer]
        print(f"  layer {layer:2d}: {old_w} -> {new_w}")
    print(f"state_dict params: {before:,} -> {after:,} ({after - before:+,})")
    print(f"arch ffn_mult_by_layer -> {list(ck['arch']['ffn_mult_by_layer'])}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
