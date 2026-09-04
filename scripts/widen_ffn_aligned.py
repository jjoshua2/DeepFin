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
real parameters at essentially unchanged cost. Measured by execution on the
production config (unique-storage count, NOT ``sum(numel())``):
**63,084,128 -> 63,794,453 = +710,325 params, +1.126%**. The dominant GEMM is
already at ~86% of BF16 peak, so there is no throughput headroom being claimed
here — this buys CAPACITY, not speed.

⚑ THAT WIDENING FIGURE IS PRE-bt4heads AND HAS NOT BEEN RE-MEASURED. Its
baseline 63,084,128 is ``main``'s ``configs/pbt2_small.yaml``; the bt4heads
promotion (``86492fa26``) moved the live config to 61,444,448 and nobody re-ran
the tool against it. The heads changed, not the FFN schedule, so the absolute
+710,325 is *probably* unmoved and the percentage *probably* is not — but
"probably" is not a measurement, so **do not requote +1.126% against the live
net**. Re-derive it by executing this tool on a live checkpoint before any
prereg quotes a capacity delta.

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

⚑⚑ FUNCTION-PRESERVING IS NOT DYNAMICS-PRESERVING — THIS IS A CONFOUND
----------------------------------------------------------------------
**The widening changes the effective step size of every layer it touches, for
the OLD rows as well as the new ones.** Aurora's final line
(``chess_anti_engine/train/aurora.py``, ``_aurora_update``) is::

    out = out * math.sqrt(max(1.0, orig_rows / orig_cols))

For ``blocks.N.ffn.0.weight`` of shape ``(h, embed_dim)`` that factor is
``sqrt(h / embed_dim)``, so raising ``h`` multiplies the ENTIRE update tensor —
not merely its new rows — by::

    sqrt(h_new / h_old)

Measured on a real production layer (796 -> 896 rows, 512 cols): total update
norm 28.213 -> 29.933, ratio **1.0610 == sqrt(896/796)**, and the updates to the
796 UNTOUCHED rows are 6.1% larger too. ``blocks.N.ffn.2.weight`` of shape
``(embed_dim, h)`` is UNAFFECTED, because ``max(1.0, 512/h) == 1.0`` for every
``h > 512``. Per production layer, the input-projection step-size increase is::

    +6.1, +3.0, +6.3, +2.8, +2.5, +3.3, +2.5, +5.9, +5.6, +0.2  (percent)

on the 10 of 16 blocks that grow; the other 6 are already tile-aligned and are
untouched.

⇒ **A widened arm differs from its control by TWO things, not one:** +1.126%
capacity AND a per-layer 0.2–6.3% effective step-size increase on 10 of 16 FFN
input projections. Any Elo / loss / regret readout that attributes a difference
to "more capacity" is CONFOUNDED with an LR change concentrated in exactly the
layers that grew. This MUST appear on the ledger prereg's **Confounds** line for
any experiment launched from a widened checkpoint; a readout without it is not a
capacity verdict.

Neutralising it would mean a per-layer LR compensation of ``sqrt(h_old/h_new)``
on the widened ``ffn.0.weight`` parameters. That is a SEPARATE intervention with
its own dynamics (it is not a no-op on the new rows either), so it is
deliberately NOT done here — this tool changes topology and nothing else.
``tests/test_widen_ffn_aligned.py::test_aurora_scale_factor_ratio_is_pinned``
pins the arithmetic, so if that line in ``aurora.py`` moves, the test fails and
tells the next reader this section is stale.

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

⚑ WHICH BUILDS CAN PRODUCE AN INPUT FOR THIS TOOL
--------------------------------------------------
``opt_param_names`` is the key the optimizer migration is addressed by, and
``Trainer.save`` writes it whenever ``_optimizer_param_names()`` can resolve the
mapping. **Current code writes it, so a freshly saved checkpoint operates
normally and the refusal below is inert against it.** Measured by execution on
this base: a checkpoint from this tree's own ``Trainer.save``
(``optimizer: aurora``, ``matrix_optimizer_scope: mlp_out``, SWA off) carries a
manifest for every optimizer slot and widens, and the SAME checkpoint with the
manifest deleted is refused. Both halves are pinned by
``tests/test_widen_ffn_aligned.py::test_a_real_trainer_save_carries_the_manifest_and_widens``
and ``::test_a_real_checkpoint_without_the_manifest_is_REFUSED``.

⚑ Checkpoints written before ``01b190f5d`` (2026-08-16) predate the key and are
refused BY DESIGN — the tool fails loudly rather than emitting a mis-migrated
file. The fix is to re-save through current code, NOT to relax the refusal.

⚑⚑ An earlier revision of this section asserted that ``main`` does not write the
key and that this tool is "INOPERABLE against it", and the operator-facing
refusal string repeated it. That was true when written (branch point
``f2677dd62``) and false about nine hours later: ``01b190f5d`` landed the key on
``main`` independently of PR #427, whose base was the live branch. The stale
claim is recorded rather than merely deleted because its failure mode is
specific — an operator who hit the refusal was told by the tool's own error
string to go solve a merge-order problem that no longer exists. A claim about
what ANOTHER branch does is dated the moment it is written; state the commit it
was measured at, or do not state it.

WHAT COUNTS AS "THE FFN" — AND HOW THAT ASSUMPTION IS CHECKED
--------------------------------------------------------------
This tool widens exactly three tensors per block —
``blocks.N.ffn.{0.weight, 0.bias, 2.weight}`` — because that is what a
``TransformerBlock`` FFN is today. That is an assumption about the model, made
inside a tool that only ever sees a checkpoint, so it is CHECKED rather than
trusted: after widening, no tensor under ``blocks.<N>.`` may still carry the OLD
hidden width as a dimension. A fourth width-coupled tensor (a per-unit scale, a
gate) would otherwise be left behind while ``arch`` claims the new width, and
``load_state_dict_tolerant`` drops it into fresh init with only a log line.

⚑ That guard is a runtime APPROXIMATION — see its raise site for the three
things it cannot see. The TOTAL check is model-derived and needs no list at all:
``tests/test_widen_ffn_aligned.py::test_real_model_round_trip_loads_strict``
rebuilds through the real ``resume_model_config_from_arch`` -> ``build_model``
and loads ``strict=True``, which is red for a missing key, an unexpected key AND
a size mismatch. "Exhaustive over the three keys I know about" is a hand-drawn
boundary; ``strict=True`` against the real module is not.

ARCH
----
``resume_model_config_from_arch`` takes topology from the CHECKPOINT's ``arch``,
and ``ffn_mult_by_layer`` is not in ``_RESUME_CONFIG_OWNED_ENCODING_KEYS``, so
the new schedule MUST be written into ``arch``. A widened ``state_dict`` with a
stale ``arch`` rebuilds the old widths and the load fails (or, worse, a tolerant
loader quietly skips the mismatched tensors).

⚑⚑ DEPLOYMENT — THE LIKELY ROUTE SILENTLY DROPS THE WIDENING
-------------------------------------------------------------
Writing the schedule into ``arch`` only helps on the paths that READ ``arch``.
``_maybe_load_bootstrap`` (``chess_anti_engine/tune/trainable_init.py``) does
NOT: it builds the model from the LAUNCH YAML, then loads the checkpoint's
weights through ``load_state_dict_tolerant`` and never calls
``peek_checkpoint_arch``. So pointing a fresh trial's ``bootstrap_checkpoint``
at a widened file while the launch yaml still carries the OLD
``ffn_mult_by_layer`` **shape-skips all 30 widened tensors into fresh init**.
The same hazard sits on the resume path: ``Trainer.load`` calls
``load_state_dict_tolerant`` WITHOUT ``require_complete=True``, so a mismatch
there is dropped rather than raised.

The run then boots healthy, reports no error, and trains randomly-initialised
FFNs in 10 of 16 blocks — well under the catastrophic-load threshold, so nothing
fires. **Update ``ffn_mult_by_layer`` in the launch yaml BEFORE starting the
arm.** ``main()`` prints the exact yaml line to paste; it is copy-pasteable on
purpose, because a Python list repr that has to be hand-translated is how the
wrong schedule gets typed.
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
_BLOCK_RE = re.compile(r"(?:^|\.)blocks\.(\d+)\.")


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


def _unwrap(name: str) -> str:
    """Strip compile/DDP wrappers from a model key. DEFENSIVE COVERAGE ONLY.

    ⚑ This is NOT parity with the trainer's own re-keying. That rule has
    exactly one definition in the project — the module-level
    `strip_compile_prefix_from_name` in
    `chess_anti_engine/train/trainer.py`, which is
    `name.replace("_orig_mod.", "", 1)`: the FIRST occurrence anywhere in
    the key, deliberately non-leading so it also reaches `AveragedModel`'s
    nested `module._orig_mod.*`, and it never strips a `module.` segment at
    all. Both `strip_compile_prefix` (a whole state_dict) and
    `Trainer._wrap_agnostic_name` (one `named_parameters()` name) DELEGATE
    to it, so the two cannot drift. This function strips a fixed set of
    LEADING prefixes instead, which is a different rule.

    ⚑⚑ An earlier revision of this docstring asserted that
    `Trainer._wrap_agnostic_name` "does not exist in this repo at all". It
    exists, in `chess_anti_engine/train/trainer.py`. The claim was measured
    at this branch's merge-base, where it was true, and this branch lands 52
    commits later. Same failure as the `opt_param_names` section above: a
    statement about code OUTSIDE this file is dated at the moment it is
    written. `test_the_trainers_real_rekeying_helper_is_what_the_unwrap_docstring_says`
    now pins BOTH helpers' existence and behaviour, so the next drift is a
    red test rather than a wrong comment.

    ⚑ It is also a no-op on every genuine input today: `Trainer.save` runs
    `strip_compile_prefix` over `state["model"]`, so real checkpoints carry
    bare `blocks.N.ffn.*` keys. This is therefore defensive coverage for
    keys `Trainer.save` does not currently emit — a hand-assembled or
    externally produced state dict — and NOT a live failure being repaired.
    The `opt_param_names` manifest is stored wrap-agnostic regardless, so if
    a prefixed key ever DID arrive, looking it up unstripped would miss its
    optimizer entry; the F5 backstop below would then refuse rather than
    emit a half-migrated file.
    """
    for pfx in ("module._orig_mod.", "_orig_mod.module.", "module.", "_orig_mod."):
        if name.startswith(pfx):
            return name[len(pfx):]
    return name


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

    # ⚑ F/R4: these two refusals answer "may this tool READ this checkpoint at
    # all", not "does it have work to do", so they MUST precede the `not changes`
    # early return below. They used to sit after it, and an already-aligned
    # SCALAR plan therefore walked straight past both: with embed_dim 8,
    # ffn_mult_by_layer (2.0,) and embed_dim_by_layer [10], plan_widths computes
    # 8*2.0 = 16 (aligned), reports "already tile-aligned", and says nothing —
    # while the width TransformerBlock actually builds is int(10 * 2.0) == 20,
    # which is not a multiple of 8. A clean bill of health for a net this tool
    # cannot read.
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

    changes, new_mults = plan_widths(arch, align=align)
    if not changes:
        return ck, changes

    gen = torch.Generator().manual_seed(int(seed))
    model = ck["model"]

    # ⚑ R3: `changes` is derived ENTIRELY from `arch`. Nothing had ever checked
    # it against the tensors it is about to pad, so an `arch` that disagrees
    # with the state_dict produced a file whose arch claims widths its weights
    # do not have — and `load_state_dict_tolerant` (called WITHOUT
    # require_complete=True on both the resume and bootstrap paths) drops those
    # tensors silently at load. Verify the plan against the stored rows first.
    in_weight_keys: dict[int, str] = {}
    for key in model:
        m = _FFN_IN_RE.search(key)
        if m is not None and m.group(2) == "weight":
            in_weight_keys[int(m.group(1))] = key
    for layer, (old_w, _) in sorted(changes.items()):
        stored_key = in_weight_keys.get(layer)
        if stored_key is None:
            continue  # caught by the post-loop "contributed no tensors" refusal
        stored_rows = int(model[stored_key].shape[0])
        if stored_rows != old_w:
            raise ValueError(
                f"arch width disagrees with the stored tensor: arch plans layer "
                f"{layer} from an FFN width of {old_w}, but {stored_key!r} has "
                f"{stored_rows} rows. Widening from a width the weights do not "
                "have would emit an arch that lies about its own tensors, which "
                "the tolerant loader then drops in silence. Refusing."
            )
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
            "(Trainer.save has written this key since 01b190f5d, 2026-08-16; a "
            "checkpoint from older code has none. Re-save it through current "
            "code — do not relax this refusal.)"
        )

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

    # ⚑ R3: `changes` says which layers the ARCH rewrite below will claim to
    # have widened. If a layer produced no tensors — because the model keys are
    # spelled in a way the FFN regexes do not match, e.g. `layers.N.ffn.*` —
    # then the emitted arch advertises widths that exist nowhere in the
    # state_dict, and every consumer that loads tolerantly (resume AND
    # bootstrap) drops those tensors into fresh init without a word.
    widened_layers = {
        int(m.group(1))
        for k in widened
        if (m := (_FFN_IN_RE.search(k) or _FFN_OUT_RE.search(k))) is not None
    }
    silent = sorted(set(changes) - widened_layers)
    if silent:
        raise ValueError(
            f"planned layer(s) {silent} contributed no tensors to the widening: "
            "their FFN parameters are absent from 'model' or are spelled in a "
            "way this tool's key patterns do not match. Rewriting arch anyway "
            "would emit a checkpoint whose declared widths no tensor has. "
            "Refusing."
        )

    # ⚑⚑ THE FFN IS THREE TENSORS ONLY BECAUSE IT IS THREE TENSORS TODAY.
    # Everything above keys off `ffn.0.weight`, `ffn.0.bias` and `ffn.2.weight`.
    # The day a block gains a FOURTH tensor sized by the hidden width — a
    # per-unit scale, a gate, a normalisation over the hidden axis — this tool
    # would widen the three it knows, rewrite `arch` to the new width, and leave
    # the fourth at the old one WITHOUT RAISING. Neither guard above sees it:
    # the arch/tensor check reads `ffn.0.weight`'s rows, and the
    # "contributed no tensors" check only needs the layer to contribute SOME
    # tensor. Downstream, `load_state_dict_tolerant` drops the stale tensor into
    # fresh init with a log line. Demonstrated by an independent reviewer, on a
    # real production state_dict, by injecting `blocks.5.ffn.scale` of shape
    # (796,): widened, arch rewritten, nothing raised.
    #
    # So: after widening, no tensor in a widened block may still carry the OLD
    # hidden width as one of its dimensions.
    #
    # ⚑ WHAT THIS CANNOT SEE, stated rather than implied — it is a runtime
    # approximation, not a total check:
    #   * a coupled tensor OUTSIDE the `blocks.<N>.` prefix;
    #   * a dimension that is a FUNCTION of the width rather than equal to it
    #     (a fused 2h gate, an h/2 split, a reshape to (h//k, k));
    #   * anything at all when the old and new widths are equal, which cannot
    #     happen here because only growing layers are in `changes`.
    # The TOTAL check is model-derived and needs no list: build the real model at
    # the emitted `arch` and `load_state_dict(..., strict=True)`, which is what
    # `test_real_model_round_trip_loads_strict` does. This guard exists because
    # that check cannot run inside a checkpoint-only tool.
    #
    # False positives are possible in principle — an unrelated tensor in the same
    # block whose dimension happens to equal the old hidden width. That is a loud
    # refusal, not a silent miswrite, and it is the right way round.
    widened_keys = set(widened)
    coupled_leftovers: list[str] = []
    for key, tensor in model.items():
        if key in widened_keys or not isinstance(tensor, torch.Tensor):
            continue
        block = _BLOCK_RE.search(key)
        if block is None:
            continue
        layer = int(block.group(1))
        if layer not in changes:
            continue
        old_w = changes[layer][0]
        if old_w in tuple(int(d) for d in tensor.shape):
            coupled_leftovers.append(f"{key} {tuple(int(d) for d in tensor.shape)}")
    if coupled_leftovers:
        raise ValueError(
            "tensor(s) still carrying the OLD FFN hidden width after widening: "
            f"{coupled_leftovers[:4]}{' ...' if len(coupled_leftovers) > 4 else ''}. "
            "This tool widens blocks.N.ffn.{0.weight,0.bias,2.weight} and nothing "
            "else, so a shape-coupled tensor it does not know about would be left "
            "at the old width while arch claims the new one — and the tolerant "
            "loader then drops it into fresh init with only a log line. Teach this "
            "tool to widen it, or reject this architecture. Refusing."
        )

    # ⚑ F5: the `opt_param_names` refusal near the top of this function only
    # catches an ABSENT manifest. A truncated, stale or mismatched one passes it
    # and then resolves nothing, so the migration is skipped with no error — the
    # same silent skip one level in. Every widened tensor must have found its
    # optimizer entry, or we refuse.
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
    # ⚑ R5: the scalar `ffn_mult` is DECORATIVE whenever a schedule is present.
    # All 12 in-tree read sites prefer `ffn_mult_by_layer`, and
    # `_resolve_ffn_mults` / `normalize_ffn_mult_by_layer` make the schedule
    # authoritative, so nothing downstream consumes the value written here.
    #
    # An earlier revision of this comment justified the line by saying
    # scripts/shrink_ffn_checkpoint.py "already does" the same thing. It does
    # NOT: that tool writes `ffn_mult = mean(normalized_schedule)` while this
    # one writes `new_mults[0]` — 1.734375 vs 1.5 on the production schedule.
    # The two tools genuinely DISAGREE about what belongs in this field, and
    # that disagreement is harmless for exactly the reason above: no consumer
    # reads it while a schedule exists. Do not "reconcile" them on the strength
    # of an agreement that was never there.
    arch["ffn_mult"] = float(new_mults[0])
    return ck, changes


def format_ffn_mult_yaml(mults: Sequence[float]) -> str:
    """Render *mults* as a copy-pasteable ``ffn_mult_by_layer:`` YAML line.

    A flow sequence of plain floats, so ``yaml.safe_load`` round-trips it back
    to the identical schedule. See the DEPLOYMENT section of the module
    docstring for why this is printed as yaml rather than as a Python repr.
    """
    body = ", ".join(repr(float(m)) for m in mults)
    return f"ffn_mult_by_layer: [{body}]"


def count_distinct_params(state: dict[str, Any]) -> int:
    """Parameter elements in *state*, deduped by STORAGE.

    ⚑ NOT ``sum(v.numel())``. Production ties one
    ``layer_smolgens.*.gen_weight.weight`` storage across 16 state_dict keys, so
    the naive sum reports 77,173,088 against a real 61,444,448 — over by exactly
    15 × 1,048,576 = 15,728,640. That specific wrong number is the one CLAUDE.md
    warns about and ``tests/test_param_count.py`` pins as
    ``"the wrong number CLAUDE.md warns about"``; a capacity tool printing it as
    its headline is how it gets quoted again.

    ⚑ SAY WHICH BRANCH. The pair above is the LIVE production config's
    (``ops/live-20260725``); ``main``'s copy of ``configs/pbt2_small.yaml``
    reads 78,812,768 against 63,084,128. The bt4heads promotion (``86492fa26``)
    moved it and ``0f5b9a6ae`` re-measured, but that commit swept only
    ``CLAUDE.md``, ``tcec.md`` and ``tests/test_param_count.py`` — this
    docstring kept quoting ``main``'s pair for two weeks. The OVERCOUNT is the
    branch-invariant half: 15 × 1,048,576 holds in both worlds, because the
    whole 1,639,680 delta landed on untied params.

    The dedup key matches ``tests/test_param_count.py::_count_distinct``
    (``data_ptr`` + ``storage_offset``) so the two cannot disagree about the
    production number. Zero-element tensors skip the dedup — several can share a
    null ``data_ptr`` without being tied — which costs nothing, since they
    contribute nothing either way.
    """
    seen: set[tuple[int, int]] = set()
    total = 0
    for tensor in state.values():
        if not isinstance(tensor, torch.Tensor):
            continue
        n = int(tensor.numel())
        if n:
            key = (tensor.untyped_storage().data_ptr(), int(tensor.storage_offset()))
            if key in seen:
                continue
            seen.add(key)
        total += n
    return total


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
    before = count_distinct_params(ck["model"])
    naive_before = sum(int(v.numel()) for v in ck["model"].values() if hasattr(v, "numel"))
    ck, changes = widen_checkpoint(ck, align=args.align, seed=args.seed)
    after = count_distinct_params(ck["model"])
    naive_after = sum(int(v.numel()) for v in ck["model"].values() if hasattr(v, "numel"))
    if not changes:
        print("no layers need widening; checkpoint already tile-aligned")
    for layer in sorted(changes):
        old_w, new_w = changes[layer]
        print(f"  layer {layer:2d}: {old_w} -> {new_w}")
    # ⚑ The unique-storage count is the headline; the naive sum is printed only
    # so a reader comparing against some other tool's output sees WHY the two
    # differ instead of assuming this one is wrong. On production the gap is
    # 15,728,640 — the tied Smolgen generator, counted 16 times.
    print(f"params (unique storage): {before:,} -> {after:,} ({after - before:+,})")
    print(
        f"  sum(numel) over state_dict: {naive_before:,} -> {naive_after:,} "
        f"({naive_after - naive_before:+,})  <-- NAIVE: double-counts tied storages"
    )
    # ⚑ On the no-op path widen_checkpoint returns BEFORE materializing the
    # schedule, so arch may carry None or no key at all — list(None) raised and
    # no output file was written, on the very path the docstring advertises as
    # safe to re-run. Recompute for display instead of assuming.
    schedule = ck["arch"].get("ffn_mult_by_layer")
    if not schedule:
        _, schedule = plan_widths(ck["arch"], align=args.align)
    # ⚑ Printed as YAML, not as a Python list repr, because the deployment path
    # that drops this widening silently is a launch yaml carrying the OLD
    # schedule (_maybe_load_bootstrap never reads the checkpoint's arch — see
    # the module docstring). Hand-translating a repr into yaml is where the
    # wrong schedule gets typed, so emit the line to paste.
    print("PASTE INTO THE LAUNCH YAML's model: section (see DEPLOYMENT above):")
    print(format_ffn_mult_yaml(schedule))
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, dst)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
