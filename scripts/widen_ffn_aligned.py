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
real parameters at essentially unchanged cost: +710,325 params (+1.16%) for the
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

⚑ AN EARLIER REVISION OF THIS FILE CLAIMED "Aurora persists exactly one buffer
per parameter here (``momentum_buffer``)". **That was wrong, and it was a crash.**
Measured on a real production ``trainer.pt`` (bt4heads armB iter100, 479 params):
only **48** parameters carry ``momentum_buffer``; **431 carry ``exp_avg`` /
``exp_avg_sq`` / ``step``**, because ``AuroraWithAuxAdam`` routes everything
outside ``matrix_optimizer_scope`` through an AdamW fallback. Per-layer that
splits exactly on rank:

    blocks.N.ffn.0.weight  (768, 512)  -> momentum_buffer
    blocks.N.ffn.2.weight  (512, 768)  -> momentum_buffer
    blocks.N.ffn.0.bias    (768,)      -> exp_avg, exp_avg_sq, step   <-- WIDENED
    blocks.N.ffn.2.bias    (512,)      -> exp_avg, exp_avg_sq, step

``ffn.0.bias`` is widened by this tool and is on the AdamW side, so padding only
``momentum_buffer`` left its ``exp_avg``/``exp_avg_sq`` at the old width and the
first resumed step died in ``exp_avg.add_(grad)`` on a shape mismatch.

The migration is therefore **generic**: every tensor in a parameter's state entry
whose shape equals that parameter's OLD shape is padded the same way the
parameter was. Nothing is matched by buffer NAME, so a future optimizer that adds
another shape-coupled buffer is carried automatically instead of silently
dropped. Non-tensor and non-conforming entries (``step``) are left untouched.
Parameters are addressed by name through ``opt_param_names``, never by position.

⚑ WHICH BUILDS CAN EVEN PRODUCE AN INPUT FOR THIS TOOL
------------------------------------------------------
``opt_param_names`` is the key the optimizer migration is addressed by, and
``Trainer.save`` writes it only on ``ops/live-20260725`` and on PR #427. On
``main`` it is NOT written, so a checkpoint produced by ``main`` reaches the
refusal below and this tool is INOPERABLE against it. That is a merge-order
fact, not a bug: it fails loudly and refuses to emit a mis-migrated file. Do
not "fix" it by relaxing the refusal.

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

    swa = ck.get("swa_model")
    if swa:
        # Resuming would build a WIDENED AveragedModel, fail to load these
        # old-width tensors, and Trainer.load swallows that — leaving the SWA
        # object on its random construction weights, which then get averaged
        # into an exported model. Refusing beats a silent corruption. SWA is
        # off in production (ledger: reverted), so this is a guard, not a gap.
        raise ValueError(
            "checkpoint carries a 'swa_model' state dict; widening it is not "
            "implemented and resuming without it silently averages random "
            "weights into the export. Re-export without SWA, or extend this tool."
        )

    if arch.get("embed_dim_by_layer"):
        # TransformerBlock sizes its FFN from THAT LAYER's embed_dim
        # (transformer.py), while plan_widths uses the scalar. Production has
        # this None; refusing beats planning widths for a net we are not reading.
        raise ValueError(
            "arch carries 'embed_dim_by_layer'; this tool plans FFN widths from "
            "the scalar embed_dim and would compute the wrong widths for a "
            "variable-width net. Refusing."
        )

    gen = torch.Generator().manual_seed(int(seed))
    model = ck["model"]
    names: list[str] = list(ck.get("opt_param_names") or [])
    opt = ck.get("opt") or {}

    # ⚑ KEYED OFF `opt`, NOT `opt["state"]`. _ChainedOptimizer.state_dict()
    # returns {"optimizers": [...]} with NO "state" key, so a guard that read
    # opt["state"] saw {} , concluded "no optimizer state to migrate", and
    # silently skipped everything — accepted-then-ignored, this repo's
    # signature defect, inside the guard written to prevent it.
    for nested in ("optimizers", "soda_anchors"):
        if opt.get(nested) or ck.get(nested):
            raise ValueError(
                f"checkpoint carries '{nested}' (a chained/anchored optimizer "
                "state this tool does not traverse). Its buffers would stay at "
                "the old width and die at the first step, and "
                "reset_mismatched_optimizer_state does not cover them. Refusing."
            )
    opt_state = opt.get("state", {})
    if opt and not names:
        raise ValueError(
            "checkpoint has optimizer state but no 'opt_param_names'; the "
            "migration cannot be addressed by name. Refusing rather than "
            "emitting a checkpoint whose optimizer buffers are the wrong shape. "
            "(Trainer.save writes this key on the live branch and PR #427, but "
            "NOT on main — see the module docstring.)"
        )

    def _unwrap(name: str) -> str:
        """Strip compile/DDP wrappers, as Trainer._wrap_agnostic_name does.

        ⚑ The FFN regexes deliberately tolerate a `module.` / `_orig_mod.`
        prefix, but `opt_param_names` is stored wrap-AGNOSTIC. Looking the
        prefixed key up directly therefore missed every entry and skipped the
        whole migration with no error — the regex said a prefix was expected
        and the lookup said the opposite.
        """
        for pfx in ("module._orig_mod.", "_orig_mod.module.", "module.", "_orig_mod."):
            if name.startswith(pfx):
                return name[len(pfx):]
        return name

    def _opt_entry(param_name: str) -> dict[str, Any] | None:
        key = _unwrap(param_name)
        if key not in names:
            return None
        i = names.index(key)
        return opt_state.get(i, opt_state.get(str(i)))

    def _migrate(entry: dict[str, Any] | None, old_shape: tuple[int, ...],
                 grow: Any) -> None:
        """Pad EVERY shape-coupled tensor in *entry*, matched by shape not name.

        A name-matched migration is what broke: `momentum_buffer` was handled
        and AdamW's `exp_avg`/`exp_avg_sq` were not. Shape-matching carries any
        future buffer automatically and leaves scalars like `step` alone.
        """
        if entry is None:
            return
        for key, val in list(entry.items()):
            if isinstance(val, torch.Tensor) and tuple(val.shape) == old_shape:
                entry[key] = grow(val)

    widened: list[str] = []
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

        old_shape = tuple(t.shape)
        widened.append(key)
        if m_in is not None:
            if m_in.group(2) == "weight":
                std = float(t.float().std())
                extra = torch.randn(
                    (new_w - t.shape[0], t.shape[1]), generator=gen, dtype=torch.float32
                ) * std
                model[key] = _pad_rows(t, new_w, fill=extra)
            else:  # bias: new units start unbiased
                model[key] = _pad_rows(t, new_w, fill=None)
            # ⚑ Moments are ALWAYS zero-padded, including for ffn.0.weight whose
            # PARAMETER rows are random: a new unit has no gradient history.
            # w=new_w binds eagerly: _migrate calls this immediately, but a
            # late-bound closure over a loop variable is one refactor away from
            # padding every layer to the LAST layer's width (ruff B023).
            _migrate(_opt_entry(key), old_shape,
                     lambda v, w=new_w: _pad_rows(v, w, fill=None))
        else:
            model[key] = _pad_cols(t, new_w)
            _migrate(_opt_entry(key), old_shape,
                     lambda v, w=new_w: _pad_cols(v, w))

    # ⚑ F5: the refusal above only catches an ABSENT manifest. A truncated,
    # stale or mismatched one passes it and then resolves nothing, so the
    # migration is skipped with no error — the same silent skip one level in.
    # Every widened tensor must have found its optimizer entry, or we refuse.
    if opt_state:
        orphans = sorted(k for k in widened if _opt_entry(k) is None)
        if orphans:
            raise ValueError(
                f"widened {len(orphans)} tensor(s) with optimizer state present "
                f"but no matching 'opt_param_names' entry: {orphans[:4]}"
                f"{' ...' if len(orphans) > 4 else ''}. The manifest does not "
                "describe this checkpoint; refusing rather than emitting buffers "
                "at the wrong width."
            )

    # ⚑ Without this the resume path rebuilds the OLD widths from the stale arch.
    arch["ffn_mult_by_layer"] = tuple(new_mults)
    # F11: keep the scalar consistent with the schedule, as
    # scripts/shrink_ffn_checkpoint.py already does. Every in-tree consumer
    # reads ffn_mult_by_layer when present, so this is hygiene — but two tools
    # that disagree about the same field is how the next reader gets it wrong.
    arch["ffn_mult"] = float(new_mults[0])
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
    # ⚑ On the no-op path widen_checkpoint returns BEFORE materializing the
    # schedule, so arch may carry None or no key at all — list(None) raised and
    # no output file was written, on the very path the docstring advertises as
    # safe to re-run. Recompute for display instead of assuming.
    schedule = ck["arch"].get("ffn_mult_by_layer")
    if not schedule:
        _, schedule = plan_widths(ck["arch"], align=args.align)
    print(f"arch ffn_mult_by_layer -> {list(schedule)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
