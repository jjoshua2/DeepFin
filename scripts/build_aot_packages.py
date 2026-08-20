#!/usr/bin/env python3
"""Build AOTInductor ``chess_b{N}.pt2`` packages for :class:`AOTEvaluator`.

Packages are weight-agnostic graphs: parameters are kept as external updatable
constants (``aot_inductor.package_constants_in_so=False``) so a single build
serves every future checkpoint of the same topology via
``AOTEvaluator.load_weights`` / ``load_constants``.

GPU-only for build and verify. Pure helpers (bucket selection, resume plan,
summary line, arg parsing) are unit-tested without CUDA.

Example::

    PYTHONPATH=. python3 scripts/build_aot_packages.py \\
        --config configs/pbt2_small.yaml \\
        --checkpoint path/to/trainer.pt \\
        --out-dir data/aot_models_512 \\
        --max-batch 4096 --resume --verify
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from chess_anti_engine.encoding import encode_positions_batch, input_plane_count
from chess_anti_engine.encoding.model_inputs import model_encoding_kwargs
from chess_anti_engine.inference import (
    _COMPILED_BATCH_BUCKETS,
    _aoti_load_package,
    _policy_output_full,
    build_aot_constants,
    model_constant_source,
)
from chess_anti_engine.model import (
    ModelConfig,
    build_model,
    load_state_dict_tolerant,
    model_config_from_flat_config,
)
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without CUDA)
# ---------------------------------------------------------------------------


def package_filename(bucket: int) -> str:
    """Filename for a single batch-bucket package."""
    return f"chess_b{int(bucket)}.pt2"


def package_path(out_dir: Path | str, bucket: int) -> Path:
    """Full path to ``chess_b{bucket}.pt2`` under ``out_dir``."""
    return Path(out_dir) / package_filename(bucket)


def parse_buckets_arg(text: str) -> tuple[int, ...]:
    """Parse a comma-separated ``--buckets`` string into positive ints."""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError("--buckets is empty; expected e.g. '1,6,24,128'")
    buckets: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"invalid bucket size {part!r} in --buckets") from exc
        if value <= 0:
            raise ValueError(f"bucket sizes must be positive, got {value}")
        buckets.append(value)
    return tuple(buckets)


def required_broker_buckets(max_batch: int) -> tuple[int, ...]:
    """The buckets the BROKER cannot run without, filtered by ``max_batch``.

    ``SlotBroker`` pads every forward to a ``_COMPILED_BATCH_BUCKETS`` value
    (``_compiled_padded_batch_size``) and loads only that ladder
    (``select_compiled_aot_buckets``). ``should_use_aot_forward`` then does an
    EXACT-KEY lookup on the padded total, so a package that is absent does NOT
    degrade to the next size up — the batch falls through to eager.
    """
    return tuple(int(b) for b in _COMPILED_BATCH_BUCKETS if int(b) <= int(max_batch))


def select_buckets(
    *,
    max_batch: int,
    buckets: Sequence[int] | None = None,
    allow_incomplete: bool = False,
) -> tuple[int, ...]:
    """Return bucket sizes to build, filtered by ``max_batch``.

    Default source is :data:`_COMPILED_BATCH_BUCKETS` — **the ladder the broker
    actually loads**, not :data:`_BATCH_BUCKETS`. The two are NOT nested: on
    2026-08-15 the compiled ladder carried six sizes (680, 1020, 1190, 1792,
    2336, 2720) that ``_BATCH_BUCKETS`` did not, and 1190 alone served ~49% of
    live batches. Defaulting to ``_BATCH_BUCKETS`` therefore produced a
    directory that looked complete, passed ``assert_uniform_constant_fqns``,
    logged nothing, and ran EAGER for ~71% of forwards.

    ``_BATCH_BUCKETS`` remains the ladder for :class:`AOTEvaluator`, which is
    the worker-side evaluator and is unreachable in production (the driver drops
    ``distributed_worker_aot_dir`` whenever ``distributed_worker_threaded`` is
    set, and it is pinned on). It is deliberately NOT the default here.

    Coverage is ENFORCED, not merely defaulted: an explicit ``--buckets`` that
    omits a broker-required size raises unless *allow_incomplete* is set. A
    partial ladder is legitimate for benchmarking, so the escape hatch exists —
    but it has to be asked for, because the failure it guards is silent.
    """
    if int(max_batch) <= 0:
        raise ValueError(f"--max-batch must be positive, got {max_batch}")
    source: Sequence[int] = (
        buckets if buckets is not None else _COMPILED_BATCH_BUCKETS
    )
    selected = tuple(int(b) for b in source if int(b) <= int(max_batch))
    if not selected:
        raise ValueError(
            f"no buckets <= max_batch={max_batch} "
            f"(source={list(source)!r})"
        )
    if not allow_incomplete:
        gap = sorted(set(required_broker_buckets(max_batch)) - set(selected))
        if gap:
            raise ValueError(
                f"requested bucket ladder omits {gap}, which the broker pads to "
                f"and loads (_COMPILED_BATCH_BUCKETS). Packages for those sizes "
                f"would be missing, load_aot_packages would skip them SILENTLY, "
                f"and should_use_aot_forward would fall through to eager for "
                f"every batch of that size. Add them, or pass "
                f"--allow-incomplete if this is a partial/bench build."
            )
    return selected


def plan_build_buckets(
    buckets: Sequence[int],
    out_dir: Path | str,
    *,
    resume: bool,
) -> tuple[list[int], list[int]]:
    """Split buckets into ``(to_build, skipped)`` under ``--resume``.

    A bucket is skipped only when ``resume`` is true and its ``.pt2`` already
    exists. Without resume, existing files are rebuilt.
    """
    out = Path(out_dir)
    to_build: list[int] = []
    skipped: list[int] = []
    for bucket in buckets:
        path = package_path(out, int(bucket))
        if resume and path.is_file():
            skipped.append(int(bucket))
        else:
            to_build.append(int(bucket))
    return to_build, skipped


def format_summary_line(
    *,
    built: int,
    skipped: int,
    verified: int,
    failed: int,
    out_dir: Path | str,
) -> str:
    """Single machine-readable summary line printed at the end of a run."""
    return (
        f"aot_build: built={int(built)} skipped={int(skipped)} "
        f"verified={int(verified)} failed={int(failed)} out={out_dir}"
    )


def build_inductor_configs() -> dict[str, Any]:
    """Inductor configs for max-autotune AOT packages with updatable constants.

    Raises if the installed torch cannot keep weights as external updatable
    constants (would silently bake one checkpoint into the graph).
    """
    # Confirm the compile entrypoint and the constants packaging knob exist.
    if not hasattr(torch._inductor, "aoti_compile_and_package"):
        raise RuntimeError(
            f"torch {torch.__version__} has no torch._inductor.aoti_compile_and_package; "
            "cannot build AOTInductor packages"
        )
    if not hasattr(torch._inductor, "aoti_load_package"):
        raise RuntimeError(
            f"torch {torch.__version__} has no torch._inductor.aoti_load_package; "
            "cannot load AOTInductor packages"
        )

    from torch._inductor import config as inductor_config

    aot_cfg = inductor_config.aot_inductor
    if not hasattr(aot_cfg, "package_constants_in_so"):
        raise RuntimeError(
            f"torch {torch.__version__} aot_inductor has no package_constants_in_so; "
            "cannot guarantee weights-as-updatable-constants. Refusing to build "
            "(would risk constant-folding / baking one checkpoint into the .pt2)."
        )

    # max-autotune quality, fixed-shape cudagraphs, weights NOT embedded in the .so
    # so get_constant_fqns()/load_constants() remain the weight-update path.
    configs: dict[str, Any] = {
        "max_autotune": True,
        "coordinate_descent_tuning": True,
        "triton.cudagraphs": True,
        "aot_inductor.package_constants_in_so": False,
        "aot_inductor.use_runtime_constant_folding": False,
    }
    return configs


def assert_package_has_updatable_constants(
    compiled: Any,
    state_dict: Mapping[str, torch.Tensor],
    *,
    bucket: int,
) -> list[str]:
    """Fail loud if a compiled package cannot rebind checkpoint weights.

    Returns the constant FQN list on success.
    """
    if not hasattr(compiled, "get_constant_fqns") or not hasattr(compiled, "load_constants"):
        raise RuntimeError(
            f"bucket {bucket}: compiled package lacks get_constant_fqns/load_constants "
            f"(type={type(compiled)!r}). Cannot serve as an AOTEvaluator package."
        )
    fqns = list(compiled.get_constant_fqns())
    if not fqns:
        raise RuntimeError(
            f"bucket {bucket}: get_constant_fqns() is empty — weights were constant-folded "
            "or not exported as updatable constants. Rebuild with "
            "aot_inductor.package_constants_in_so=False and "
            "aot_inductor.use_runtime_constant_folding=False. "
            "A constant-folded package would bake one checkpoint in and break load_weights()."
        )
    weight_fqns = [f for f in fqns if f.endswith(".weight")]
    if not weight_fqns:
        raise RuntimeError(
            f"bucket {bucket}: get_constant_fqns() has {len(fqns)} entries but no "
            "'.weight' FQNs — refusing a package that cannot rebind model weights."
        )
    overlap = sum(1 for f in fqns if f in state_dict)
    if overlap == 0:
        raise RuntimeError(
            f"bucket {bucket}: none of {len(fqns)} constant FQNs appear in the "
            "checkpoint state_dict; load_weights() would be a no-op. "
            f"Sample FQNs: {fqns[:5]}"
        )
    if overlap < max(1, len(fqns) // 2):
        raise RuntimeError(
            f"bucket {bucket}: only {overlap}/{len(fqns)} constant FQNs match the "
            "checkpoint state_dict; expected a near-complete weight-agnostic graph. "
            f"Sample unmatched: {[f for f in fqns if f not in state_dict][:5]}"
        )
    return fqns


# A relative scale + an absolute shift, so the probe moves a constant whatever
# its magnitude — including one that is exactly 0.0, where a pure multiply is a
# no-op and the probe would report an output-affecting constant as inert.
_REBIND_PROBE_SCALE = 1.5
_REBIND_PROBE_SHIFT = 0.125


def unrebindable_output_constants(
    model: torch.nn.Module,
    constant_fqns: Sequence[str],
    *,
    probe: torch.Tensor,
) -> list[str]:
    """Model constants that MOVE the compared outputs but the package cannot rebind.

    ⚑⚑ WHY THIS EXISTS — MEASURED 2026-08-16 ON THE REAL PACKAGES, and it is the
    mechanism behind the one reading the #432 verify run could not explain.

    ``load_constants`` can only write the FQNs the package DECLARES. Everything
    else AOTInductor folded into the compiled graph at build time, and it stays
    at the checkpoint the package was compiled from **forever** — every publish,
    every iteration. Nothing checked for that: :func:`build_aot_constants` fails
    loud when an FQN is missing from the checkpoint (``fqns`` not ⊆
    ``state_dict``) and :func:`assert_package_has_updatable_constants` requires
    half the FQNs to match, but **neither direction covers a model constant that
    is absent from ``fqns`` altogether** — the payload is built FROM ``fqns``, so
    it covers them by construction and even ``check_full_update=True`` cannot
    fire. A folded constant is not an unfilled constant; it is not a constant at
    all any more.

    Measured on ``data/aot_models_512/chess_b16.pt2`` (455 declared FQNs) and on
    ``data/aot_models_512_bt4heads/chess_b1190.pt2`` (457), against their own
    configs: **71-75 model constants are undeclared, and two of them are on the
    production inference path** —

        policy_own.log_temp      folded into the policy logit scale
        value_wdl.net.2.bias     folded into the WDL output bias

    (the rest are the tied ``layer_smolgens.*.gen_weight`` aliases, which share
    ONE storage with a declared FQN and are therefore rebound, and the auxiliary
    heads the inference graph never traces).

    Consequence, and it is the whole of the "1.0 -> 2.3" step: run the July
    packages on July weights and the folded values are the right ones, so the
    gate reads x1.0. Run the SAME files on August weights and the package applies
    a stale policy temperature and a stale WDL bias. ``policy_own.log_temp``
    moved -0.3282 -> -0.2772 between those two vintages and
    ``value_wdl.net.2.bias`` moved [0.0088, -0.0321, -0.0013] ->
    [-0.0023, -0.0141, 0.0023]; swapping ONLY those two values back on the
    August model reproduces mean row TV **0.0297 (policy) / 0.0031 (wdl)**
    against a **0.00006 / 0.00001** null on the July model — a step, then flat
    across aug10/aug11/aug12 exactly as the ratio arms read, with argmax
    agreement 1.0000 throughout (a temperature rescale cannot reorder a row).

    ⇒ **the ratio arm's weight-dependence is a TRUE POSITIVE about the packages,
    not the gate re-acquiring the defect it was written to remove.** This
    function is what makes that legible instead of "UNDIAGNOSED".

    Method. Reachability is MEASURED, not listed: perturb the undeclared
    constants and see whether the two arrays the gate actually compares
    (``_policy_output_full`` and ``wdl``) move. An allowlist of "auxiliary head"
    prefixes would be a source-grep guard that goes stale the first time a head
    is renamed.

    Two exclusions, both exact rather than heuristic:

    * **Tied storages.** An undeclared FQN whose ``untyped_storage`` is the SAME
      object as a declared one IS rebound, under the other name. The 16
      ``layer_smolgens.*.gen_weight.weight`` keys are one tensor and the package
      declares it as layer 15's; flagging the other 15 would be a guaranteed
      false failure on every package this repo builds.
    * **Non-floating-point constants.** Index/mask tables (``to_sq``,
      ``compact_to_full``, ``promo_from``) are TOPOLOGY, identical for every
      checkpoint of a given architecture, so a folded one cannot go stale — and
      perturbing an index buffer would index out of bounds rather than measure
      anything. They are skipped, which is a stated blind spot, not a claim.

    The probe restores every tensor it touched. Returns the offending FQNs
    sorted, or ``[]``.

    ⚑ WHY THIS FAILS THE BUCKET RATHER THAN WARNING, AND WHAT THE FIX IS.
    Rebuilding does NOT clear it — the same folding happens again, so a WARN
    would be a permanent one. The likely repair is one line of
    :func:`build_inductor_configs`: ``aot_inductor.use_runtime_constant_folding``
    is set ``False`` there, and torch documents that flag as *"whether to create
    a submodule for constant graph"* — i.e. ``True`` recomputes folded constants
    at load time from the updatable originals, which is exactly what
    ``exp(log_temp)`` and the final WDL bias-add need. ⚑ That is a HYPOTHESIS,
    not a measurement: flipping it needs a GPU rebuild, which nothing here has
    done. **The observation that confirms it** is one bucket rebuilt with the
    flag ``True``, then ``get_constant_fqns()`` containing ``policy_own.log_temp``
    and ``value_wdl.net.2.bias`` — at which point this function returns ``[]``
    and the bucket passes. Until that runs, a FAIL is the honest verdict: these
    packages cannot track the net, and the ratio arms are blind to it on exactly
    the vintage anyone would test them against.
    """
    src = model_constant_source(model)
    declared = {str(f) for f in constant_fqns}
    covered_storages = {
        src[f].untyped_storage().data_ptr() for f in declared if f in src
    }
    candidates = [
        k
        for k in sorted(src)
        if k not in declared
        and src[k].is_floating_point()
        and src[k].numel() > 0
        and src[k].untyped_storage().data_ptr() not in covered_storages
    ]
    if not candidates:
        return []

    def _compared(out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        return _policy_output_full(out).detach().clone(), out["wdl"].detach().clone()

    def _moves(names: Sequence[str], base: tuple[torch.Tensor, torch.Tensor]) -> bool:
        saved = {n: src[n].detach().clone() for n in names}
        try:
            with torch.no_grad():
                for n in names:
                    src[n].mul_(_REBIND_PROBE_SCALE).add_(_REBIND_PROBE_SHIFT)
                got = _compared(model(probe))
        finally:
            with torch.no_grad():
                for n in names:
                    src[n].copy_(saved[n])
        return not (
            torch.equal(got[0], base[0]) and torch.equal(got[1], base[1])
        )

    with torch.no_grad():
        baseline = _compared(model(probe))
    # Cheap path first: one probe over ALL candidates. A healthy package pays two
    # forwards; only a package that is actually folding pays the attribution pass.
    if not _moves(candidates, baseline):
        return []
    return sorted(n for n in candidates if _moves([n], baseline))


# ---------------------------------------------------------------------------
# Model / checkpoint loading
# ---------------------------------------------------------------------------


def load_model_config(config_path: Path | str) -> ModelConfig:
    """Build :class:`ModelConfig` from a run YAML's ``model:`` block."""
    raw = load_yaml_file(config_path)
    flat = flatten_run_config_defaults(raw)
    return model_config_from_flat_config(flat)


def load_checkpoint_state_dict(checkpoint: Path | str) -> dict[str, torch.Tensor]:
    """Load a trainer.pt or bare model state_dict."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        # Bare state_dict (tensor values) or unexpected layout.
        if not ckpt:
            raise ValueError(f"empty checkpoint: {path}")
        sample = next(iter(ckpt.values()))
        if not torch.is_tensor(sample):
            raise ValueError(
                f"checkpoint {path} has no 'model' key and values are not tensors "
                f"(sample type={type(sample)!r})"
            )
        state = ckpt
    else:
        raise ValueError(f"unsupported checkpoint type {type(ckpt)!r} at {path}")
    return {str(k): v for k, v in state.items()}


def build_reference_model(
    model_cfg: ModelConfig,
    state_dict: Mapping[str, torch.Tensor],
    *,
    device: str = "cuda",
) -> torch.nn.Module:
    """Build eval/bf16/cuda model matching AOTEvaluator's load path."""
    model = build_model(model_cfg)
    load_state_dict_tolerant(model, dict(state_dict), label="aot-build")
    if hasattr(model, "_inference_only"):
        # Match production broker/worker: only policy_own + wdl feed search.
        setattr(model, "_inference_only", True)
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Build / verify
# ---------------------------------------------------------------------------


def _require_cuda(action: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{action} requires CUDA (torch.cuda.is_available() is False). "
            "Run this script on a GPU host during a training-paused window."
        )


def compile_bucket_package(
    model: torch.nn.Module,
    *,
    bucket: int,
    input_planes: int,
    out_path: Path,
    inductor_configs: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Export + AOTInductor-compile one fixed batch-size package."""
    device = next(model.parameters()).device
    example = torch.randn(
        int(bucket),
        int(input_planes),
        8,
        8,
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        exported = torch.export.export(model, (example,))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Must end in .pt2 (aoti_compile_and_package requirement); write via temp
    # then rename so --resume never sees a half-written package.
    tmp_path = out_path.with_name(f".{out_path.stem}.partial.pt2")
    if tmp_path.exists():
        tmp_path.unlink()
    if out_path.exists():
        out_path.unlink()

    print(
        f"  compiling bucket={bucket} planes={input_planes} -> {out_path} ...",
        flush=True,
    )
    torch._inductor.aoti_compile_and_package(
        exported,
        package_path=str(tmp_path),
        inductor_configs=dict(inductor_configs),
    )

    # Fail loud before renaming if constants are not updatable.
    compiled = torch._inductor.aoti_load_package(str(tmp_path))
    fqns = assert_package_has_updatable_constants(
        compiled, state_dict, bucket=int(bucket),
    )
    del compiled
    tmp_path.replace(out_path)
    print(
        f"  bucket={bucket} OK ({len(fqns)} updatable constants) -> {out_path}",
        flush=True,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax (masked/illegal slots -> ~0)."""
    z = logits.astype(np.float64)
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(np.sum(e, axis=-1, keepdims=True), 1e-30, None)


def row_tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-row total-variation distance: ``0.5 * sum|p - q|``, bounded in [0, 1]."""
    return 0.5 * np.sum(np.abs(p - q), axis=-1)


def mean_row_tv(p: np.ndarray, q: np.ndarray) -> float:
    """Mean per-row total variation. 0 is identical, 1 is disjoint support.

    A whole-array ``max`` is deliberately NOT used — see :func:`_compare_bucket`.
    But a mean ALONE is not enough either: it dilutes a defect confined to a few
    rows, which is exactly the shape of a boundary/indexing bug at one bucket
    size. That is why the gate also carries a tail statistic
    (:func:`tail_row_tv`) and does not rely on this one on its own.
    """
    return float(np.mean(row_tv(p, q)))


def tail_row_tv(p: np.ndarray, q: np.ndarray, *, quantile: float = 0.99) -> float:
    """High quantile of the per-row TV — the arm with power against row-local damage.

    A kernel that is wrong on 2% of rows and perfect on the rest barely moves
    the mean but moves this a lot. It is a quantile rather than a max so that it
    is not an extreme-value statistic that grows with bucket size (the defect
    that broke the previous gate); at every bucket it estimates the same
    population quantity.
    """
    return float(np.quantile(row_tv(p, q), quantile))


def bf16_ulp_perturbation(ref_logits: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """``ref_logits`` nudged by ±1 bf16 ULP **of the raw value** — the FLOOR.

    ⚑⚑ WHY THIS EXISTS, and why the batch-shape control is not sufficient on its
    own. :func:`eager_batch_shape_control` measures the irreducible difference
    empirically, which is the right idea — but it is **degenerate on some
    backends**. Measured 2026-08-15 on CPU with the real ``ChessNet`` topology,
    the chunked control is **bitwise identical** to the full batch at every size
    from n=2 to n=128, so its TV is exactly 0. Dividing by that (even floored to
    1e-6) manufactures an arbitrarily large ratio and FAILS a healthy package:
    a measured case reads TV 4e-4 against control 0 as **x399, FAIL**, with
    argmax agreement a perfect 1.0.

    Note also that round-tripping ``ref_logits`` through bf16 is a NO-OP and
    cannot serve as the floor: the model already runs in bf16, so the reference
    logits are bf16 values widened to float32 and re-rounding changes nothing
    (verified: exact array equality, TV 0.0).

    So this is the smallest difference two bf16 pipelines could exhibit at all:
    each live logit moved by one unit in the last place of ITS OWN magnitude,

        delta_i = sign_i * 2^(floor(log2|z_i|) - 7)

    because bf16 carries 8 significand bits (1 implicit + 7 stored) and its
    spacing at ``|z|`` is therefore ``2^(floor(log2|z|) - 7)``. It is positive
    for any non-constant logits, it is backend-independent, and — the property
    the whole gate turns on — it **tracks sharpness**, because a fixed logit
    perturbation produces a larger probability movement as the softmax
    concentrates.

    ⚑⚑ THE PERTURBATION IS SHIFT-**COVARIANT**, NOT SHIFT-INVARIANT, AND THAT IS
    THE CORRECT PHYSICS. An earlier revision centred each row on its live mean
    first and then applied a *relative* ``2^-8``, making the nudge
    ``|z - rowmean| * 2^-8``. The argument for it was that the raw logits'
    "arbitrary common offset" would otherwise scale the floor, since adding a
    constant to already-computed logits leaves the softmax and the measured
    AOT-vs-eager TV unchanged. **That premise is mis-derived.** The offset is
    not arbitrary — it is what the net emits — and a net whose head bias
    actually moved emits larger-magnitude logits whose bf16 rounding error
    genuinely IS larger. Rounding acts on the raw value, so the floor must move
    with it: shift-COVARIANCE. Only a *post-hoc* constant added to a fixed
    logit array (which no bf16 pipeline ever performs) leaves the error alone.

    The centred form was not merely inelegant, it was **head-dependent by up to
    22x**, because the two production heads have very different offset/spread
    ratios. Measured 2026-08-15 on the real ``ChessNet`` (bt4heads iter100),
    per-head row mean / within-row spread:

        head    row mean  spread   centred floor (mean/tail)  raw-ULP floor      ratio
        policy   -5.98     3.20      1.27e-2 / 1.95e-2        3.66e-3 / 6.58e-3   0.3x
        wdl      -8.60     1.00      5.76e-4 / 1.27e-3        6.99e-3 / 2.86e-2   12.1x/22.6x

    Consequence, measured: a package whose WDL differs from eager by EXACTLY one
    bf16 ULP — the best a package can physically be — read ``wdl_mean=x12.25``,
    ``wdl_rows_over=26/64``, **FAIL**. Whole-network corroboration (jitter every
    weight by ±1 bf16 ULP, raw TV pol 2.91e-2 / wdl 6.14e-3) read ``pol_mean``
    2.44 but ``wdl_mean`` 13.21 — the SAME physical perturbation reading 5.4x
    differently on the two heads against one shared bound.

    ⚑ Sentinels stay UNPERTURBED. The production policy array is 4672 wide but
    the net emits 1858 real logits; ``_policy_output_full`` fills the other 2814
    slots (60% of the row) with a -1e9 sentinel that softmax sends to exactly 0.
    They must keep exactly zero mass, and an absolute ULP of -1e9 is ~3.9e6,
    which would move them by more than the entire live range. The ``live`` mask
    also protects the floor from the mask width: widening a row with sentinels
    cannot change it at all, which is the padding-invariance property the
    sentinel-mass defect (floor inflated 51x, every policy arm arithmetically
    unable to fire) violated.

    ⚑ WHAT IS AND IS NOT BANKED. The CPU-derivable half of every claim above is
    re-measured on each CI run by ``tests/test_aot_verify_gate.py`` — the
    round-trip no-op, non-degeneracy, sharpness tracking, padding invariance,
    shift covariance and the per-head magnitudes. The CUDA numbers (0.0176
    control and 0.01737 AOT-vs-eager at bucket 1190) are single-run readings on
    hardware CI does not have, and they were taken against the CENTRED floor;
    they are stated as provenance, not as a calibration, and **the gate has
    still never been run on CUDA against a real AOT/eager discrepancy.** A
    previous revision also quoted "2.31 (July packages on August weights,
    stale)" as support for 1.5. That number is withdrawn from the
    justification: it was never banked, and it argues against itself — 2.31 >
    1.5 means the new gate FAILS the very packages the rewrite was meant to
    stop condemning, which would relabel the verdict rather than correct it.
    """
    rng = np.random.default_rng(int(seed))
    z = ref_logits.astype(np.float32)
    # e^-80 ~ 1.8e-35 of the top entry's mass: below float32 resolution, so an
    # entry this far down is inert in the softmax by construction, not by
    # threshold-picking. Guards the all-sentinel row (live is then everything).
    live = z > (np.max(z, axis=-1, keepdims=True) - np.float32(80.0))
    absz = np.abs(z)
    # ``frexp`` gives |z| = m * 2**e with m in [0.5, 1), so floor(log2|z|) = e-1
    # and one bf16 ULP is 2^(e-1-7). Exact by construction — no log2 rounding
    # can land an entry on the wrong side of a binade boundary.
    _, binade = np.frexp(absz)
    ulp = np.ldexp(np.float32(1.0), binade - 8).astype(np.float32)
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=z.shape)
    # ⚑ z == 0 is left UNPERTURBED: log2(0) is undefined, and the honest bf16
    # step there is the smallest subnormal (~9.2e-41), whose effect on a softmax
    # is not representable — exp(9.2e-41) is exactly 1.0 in float32, so adding
    # it would be a no-op anyway. Leaving it alone says so explicitly instead of
    # relying on an underflow. It is also the only entry for which `frexp`
    # returns a meaningless exponent.
    return np.where(live & (absz > np.float32(0.0)), z + sign * ulp, z)


# ⚑⚑ NULL CALIBRATION — measured, not chosen.
#
# Each arm divides a statistic of (AOT vs eager) by the SAME statistic of the
# floor. That makes every arm scale-free, but it does NOT make them read the
# same number when healthy: numerator and denominator are different random
# draws, and an order statistic of a 3-column WDL row-TV is a far noisier
# estimator than a mean over 1858 columns.
#
# ⚑⚑ AN EARLIER REVISION ANSWERED THAT WITH A PER-ARM `_ARM_NULL` NORMALIZER,
# AND THAT WAS THE WRONG FIX. Re-measured 2026-08-15 on PRODUCTION-SHAPED
# arrays (policy 4672 wide with 1858 live, WDL 3 wide), 12 seeds x buckets
# {8, 32, 128, 256, 512, 1024, 2048, 4096}, n=96 healthy readings per arm:
#
#     arm        mean    p99    worst      verdict
#     pol_mean   1.003   1.18   1.181      keep
#     pol_tail   0.990   1.11   1.178      keep
#     wdl_mean   0.989   1.39   1.462      keep
#     pol_max    0.985   1.12   1.179      superseded (see below)
#     wdl_tail   1.040   2.09   2.495      DROPPED
#     wdl_max    1.065   2.16   2.552      DROPPED
#
# The two dropped arms are not mis-CALIBRATED, they are mis-SPECIFIED: their
# healthy spread is 2.4x their own mean, so no constant can separate healthy
# from corrupt. Normalizing them by a null big enough to stop false failures is
# the SAME defect this gate exists to prevent — an arm that cannot fire — just
# reached by arithmetic instead of by padding (cf. the sentinel-mass defect in
# `bf16_ulp_perturbation`, found the same day). A p99 or a max over three
# columns is a degenerate order statistic; it is deleted, not tuned.
#
# What the max arms were FOR — corruption confined to fewer than 1% of rows,
# which a mean or a p99 dilutes away — is now covered properly, and better, by
# `_ROW_EXCEED_K` below. No normalization table is needed and none is kept: the
# table only ever existed to prop up the estimators now removed.
#
# ⚑⚑ THE TABLE ABOVE WAS MEASURED WITH A CIRCULAR NULL, so read the one below
# instead. It built "healthy" as `bf16_ulp_perturbation(..., seed=other)` — the
# DENOMINATOR'S OWN FUNCTION — so every arm reads ~1.0 whatever that function
# computes, including when it computes something with no physical meaning. It
# also drew zero-mean logits only, the one regime where the centred and raw
# forms coincide. It could not have detected the F1 defect and did not.
#
# ⚑ RE-MEASURED 2026-08-15 with a healthy model written INDEPENDENTLY of the
# floor (a ±1 raw-bf16-ULP jitter derived from scratch in the test file), on
# PRODUCTION-SHAPED arrays at the PRODUCTION REGIME — policy 4672 wide with 1858
# live, row mean -6.08 spread 3.03; WDL 3 wide, row mean -8.46 spread 0.93:
#
#     bucket  trials  pol_mean med/worst  pol_tail med/worst  wdl_mean med/worst
#         16     200   1.00 / 1.15        1.01 / 1.52         0.99 / 2.24
#         32     200   1.00 / 1.06        1.00 / 1.29         1.03 / 1.53
#         64      96   1.00 / 1.06        1.00 / 1.22         0.99 / 1.33
#        128      96   1.00 / 1.04        1.01 / 1.13         1.01 / 1.21
#        256      48   1.00 / 1.02        1.00 / 1.09         0.99 / 1.14
#        512      48   1.00 / 1.01        0.99 / 1.08         1.00 / 1.07
#       1024      24   1.00 / 1.01        0.99 / 1.04         1.00 / 1.03
#       2048      12   1.00 / 1.00        1.00 / 1.03         0.99 / 1.03
#       4096      12   1.00 / 1.00        1.00 / 1.02         1.00 / 1.03
#
# `pol_rows_over` and `wdl_rows_over` were 0 in all 568 trials.
#
# ⚑ THE MEDIANS ARE 1.00 EVERYWHERE, THE TAILS ARE NOT, and the residual is
# entirely `wdl_mean` at the smallest buckets: a mean over THREE columns of 16
# rows is a noisy estimator, and at bucket 16 it breaches the 2.0 default in
# 2 of 200 healthy trials (worst 2.24). Bucket 16 is the smallest on the
# DEFAULT ladder, so a verify runs 2 trials there and false-fails ~2% of the
# time. That is stated, not tuned away: raising the bound to absorb it would
# weaken the arm on every other bucket, where it reads 1.00 +/- 0.03.
#
# tests/test_aot_verify_gate.py re-derives the null on each CI run, in both
# regimes and under two independent healthy models.

# A row whose TV exceeds this multiple of the floor's OWN p99 is not bf16 noise.
# Measured over 50 healthy runs per head at each of buckets {8, 32, 128, 512,
# 2048}: ZERO runs produced an exceeding row at k=4, k=6 or k=10, on either
# head. Sensitivity, same measurement: one corrupted row inside a 2048-row
# batch is caught exactly (1 row flagged) while `wdl_mean`/`pol_mean` read x0.02
# and x0.03 — i.e. FAR below 1.0, because the other 2047 rows are bit-exact and
# dilute the defect into invisibility. That is the single-bad-row blind spot,
# and it is the whole reason this criterion exists. 6.0 is the middle of the
# measured no-false-positive band.
_ROW_EXCEED_K = 6.0

# The analytic (ULP) and empirical (batch-shape control) floor estimates are
# independent derivations of the same quantity and agree to 15% on real
# hardware. 5x is far outside any disagreement either could legitimately
# produce, so a wider gap means one estimator is broken. See `_floor`.
_FLOOR_DIVERGENCE_MAX = 5.0

# Below this the SHAPE CONTROL is not a measurement. See `_floor`.
#
# ⚑ RE-MEASURED 2026-08-15 after the raw-ULP fix, on the real production net
# (bt4heads iter100, CPU bf16, 80 resamples per bucket). The honest ULP floor
# MEANS are now:
#
#     bucket    policy mean          wdl mean
#          8    3.81e-3 .. 8.02e-3   2.24e-3 .. 2.14e-2
#         16    4.32e-3 .. 6.98e-3   1.91e-3 .. 1.88e-2
#         64    5.03e-3 .. 6.34e-3   7.59e-3 .. 1.42e-2
#
# so 1e-4 sits 19x below the smallest reading either head produces at or above
# the smallest DEFAULT bucket (16), and ~1000x above the numerical case it has
# to catch (a control that only re-associates a few reductions reads ~1e-7).
#
# ⚑ THE OLD JUSTIFICATION WAS A POLICY-HEAD NUMBER APPLIED TO BOTH HEADS. It
# claimed "~200x below the real floor and ~5000x above the noise case" against
# a real floor of 0.018 — which was the CENTRED policy floor, inflated by the
# defect fixed in `bf16_ulp_perturbation`. On the WDL head the centred floor
# was 4.22e-4 .. 6.34e-4, i.e. only ~4x above this constant, and 7.8% of
# individual real WDL rows fell BELOW it. Fixing the physics moved the WDL
# floor up ~20x, so the constant is now clear on BOTH heads rather than on one.
#
# ⚑ AND THE ESCAPE IS ONE-SIDED. It applies to the SHAPE CONTROL only, because
# that is the estimator known to degenerate (bitwise identical on CPU). When it
# was two-sided it fired whenever EITHER estimate was small, so a small ULP
# estimate retired the cross-check and an inflated control then became the
# floor unchallenged — demonstrated at production shapes: ULP wdl floor 5.43e-5
# plus an inflated control made a COMPLETELY WRONG AOT wdl read
# `ok=True wdl_mean=x1.00 wdl_rows_over=0`, where a healthy control on the same
# inputs read `ok=False wdl_mean=x512.86`.
_FLOOR_DEGENERATE_TV = 1e-4

# ⚑ AN ABSOLUTE PLAUSIBILITY BOUND ON THE FLOOR ITSELF, which no ratio between
# the two estimates can supply: if BOTH are inflated the same way, the
# cross-check sees perfect agreement. A row TV of this size is not bf16 noise —
# it is most of the distance to "the two distributions are disjoint" (row TV is
# bounded by 1).
#
# Measured 2026-08-15 on the real net, worst floor over 80 resamples at each of
# buckets {1, 2, 8, 16, 64, 128}: policy tail 1.89e-2, wdl tail 6.17e-2. The
# original sentinel-mass defect read 0.9719. 0.25 is the geometric midpoint of
# those (sqrt(6.17e-2 * 0.9719) = 0.245): 4.1x above anything the healthy net
# produces, 3.9x below the defect it has to catch.
#
# It is also bounded analytically, not just empirically. One bf16 ULP moves a
# logit by at most |z| * 2^-7, so the largest logit GAP change a row can see is
# 2 * max|z| / 128, and a softmax's TV under a gap change d is at most d/4.
# With the net's measured max|z_live| of 20.1 that caps the floor at ~0.079;
# reaching 0.25 would need max|z_live| ~ 64, a 3x growth in logit magnitude
# which is itself a red flag. ⚑ That means this bound alone would have caught
# the sentinel defect (0.97) WITHOUT the cross-check — which matters because
# the cross-check is unavailable in two reachable regimes (see `_floor`).
_FLOOR_IMPLAUSIBLE_TV = 0.25


def _compare_bucket(
    *,
    aot_pol: np.ndarray,
    aot_wdl: np.ndarray,
    ref_pol: np.ndarray,
    ref_wdl: np.ndarray,
    ctl_pol: np.ndarray | None,
    ctl_wdl: np.ndarray | None,
    tv_ratio_max: float,
    floor_seed: int = 0,
) -> tuple[bool, str, int, int]:
    """Return (pass, detail) for one bucket, gated RELATIVE to an eager control.

    Compares in **probability space** (softmax), the quantities the broker
    actually consumes — MCTS priors and value WDL. Raw-logit max-abs-diff is a
    useless criterion here: ``_policy_output_full`` leaves masked illegal-move
    slots at a large sentinel whose bf16-vs-max-autotune representation differs
    by ~1e6, dwarfing every real difference.

    ⚑⚑ WHY THIS IS A RATIO AND NOT AN ABSOLUTE TOLERANCE. The previous gate was
    ``max |p_aot - p_ref| <= 0.02`` over ``N x 1858`` probabilities, and it broke
    on 2026-08-15. **An absolute probability tolerance is pinned to a sharpness
    regime and expires when the net sharpens**: the max always lands on a row's
    top-1 entry, and the net's top-1 probability grew 4.4x (max 0.221 -> 0.980)
    over the window.

    ⚑⚑ BUT THE HEADLINE EVIDENCE THIS DOCSTRING USED TO LEAD WITH IS CONFOUNDED,
    AND IT IS CORRECTED HERE (2026-08-16). It read: *"it broke on the WEIGHTS,
    not the packages — the identical month-deployed ``data/aot_models_512`` files
    read 0.0015-0.0052 (PASS) on July weights and up to 0.175 (FAIL) on August
    weights."* The files are identical; **their effective constants are not.**
    ``policy_own.log_temp`` and ``value_wdl.net.2.bias`` are constant-folded into
    every one of those packages and ``load_constants`` cannot write them (see
    :func:`unrebindable_output_constants`), so on August weights those files
    really do apply a July policy temperature and a July WDL bias. The
    two-vintage comparison therefore changed the weights AND the package's
    behaviour at the same time; it cannot separate "the gate read the weights"
    from "the packages were stale". It is not evidence for either.

    What DOES establish the old gate's defect with no confound is the **28/29
    failure on packages freshly built from, and verified against, THEIR OWN
    checkpoint** — no staleness is possible there — plus the two structural
    arguments below, which are arithmetic rather than measurement.

    Two further defects in that design, both structural:

    * A **global max over ``N x 1858``** is an extreme-value statistic. Its
      expectation grows with bucket size for reasons unrelated to correctness,
      so large buckets fail merely for being large. Measured exceedance
      ``P(rowmax > 0.02) = 0.01177`` gives ``P(pass at N) = (1-0.01177)^(2N)``:
      predicted 1.55 expected passes across 29 buckets, observed 1. It was a
      lottery, not a test.
    * It had no notion of the **irreducible** difference. Eager bf16 compared to
      ITSELF at a different batch shape differs by TV 0.0176 — same weights, no
      AOT anywhere — purely from BLAS reduction order. Any absolute threshold
      below that floor is unreachable by a perfect package.

    So the gate is now a RATIO against an irreducible floor, on four arms:
    mean row TV and tail (p99) row TV, for policy and for WDL. Each numerator is
    divided by the SAME functional computed on the floor, so a healthy package
    reads ~1 on every arm regardless of how sharp the net has become.

    The floor is ``max`` of two independent estimates of "irreducible bf16
    disagreement", because the true floor is at least each of them:

    * :func:`eager_batch_shape_control` — empirical, the same weights on the
      same rows at a different batch shape. ⚑ **This one can be exactly zero**
      (CPU: bitwise identical from n=2 to n=128), which is why it cannot be the
      sole denominator.
    * :func:`bf16_ulp_perturbation` — analytic, every live logit moved one bf16
      ULP **of its own raw magnitude**. Backend-independent, and tracks
      sharpness. ⚑ It is not "never degenerate": a WDL row whose three logits
      share a binade has three equal ULPs, so the 2-in-8 sign draw that moves
      them all the same way is a pure common shift the softmax cannot see. On
      the real net that is 12-17% of rows at any one floor seed, which is
      harmless at a batch statistic and NOT harmless at bucket 1 — see
      ``_ratio``.

    ⚑⚑ THE CROSS-CHECK BETWEEN THE TWO IS UNAVAILABLE IN TWO REACHABLE REGIMES
    — on CPU (the shape control is bitwise identical, so the degeneracy escape
    fires and the check never runs) and at ``ctl is None`` (there is no second
    estimate at all). Both read ``floor=...ulp-only...`` in the detail line.
    That is why the floor also carries an ABSOLUTE plausibility bound
    (``_FLOOR_IMPLAUSIBLE_TV``), which needs no second estimate.

    ⚑ There is NO arbitrary epsilon floor. If BOTH estimates are zero the
    reference is degenerate (constant logits), and then the only defensible
    verdict is exact: any nonzero numerator FAILS. A `1e-6` clamp was tried and
    removed — it silently converted "no information" into "x399, FAIL".

    ARGMAX IS NOT GATED HERE. It is returned as ``(matches, rows)`` and gated by
    the CALLER on rows POOLED across all ``verify_n`` trials. Per-trial it is a
    coin flip at small buckets: at the healthy per-row rate of 0.96, requiring
    ``>= 0.90`` of 8 rows means requiring 8/8, which spuriously fails **48%** of
    the time. Pooling is what makes the criterion mean what its name says.
    Healthy reads 0.958-0.970, the eager-only control 0.9597-0.9655, and the
    random floor is ~1/1858 = 0.0005.
    """
    aot_pp, ref_pp = _softmax(aot_pol), _softmax(ref_pol)
    aot_wp, ref_wp = _softmax(aot_wdl), _softmax(ref_wdl)

    def _floor(ref_logits: np.ndarray, ref_p: np.ndarray,
               ctl: np.ndarray | None) -> tuple[float, float, str, str]:
        """(mean, tail, note, source) of the irreducible floor.

        Each statistic is computed on the floor with the SAME functional and the
        SAME row count as its numerator, so every arm reads ~1 when healthy.

        ⚑ The two estimates CROSS-CHECK each other, and the check is reported.
        They are independent derivations of "irreducible bf16 disagreement" and
        measured on real hardware they agree to 15% (0.0202 analytic vs 0.0176
        empirical at bucket 1190, both pre-fix readings). A large divergence
        therefore means one of them is broken, and a BROKEN FLOOR IS THE ONE
        FAILURE THIS GATE CANNOT SELF-DETECT: it silently rescales every arm, in
        the passing direction if the floor is too big. This is not hypothetical
        — the sentinel-mass defect fixed in `bf16_ulp_perturbation` on
        2026-08-15 inflated the policy floor 51x and made all three policy arms
        arithmetically incapable of failing. It would have shown up here as a
        45x divergence.

        ⚑⚑ AND THE CROSS-CHECK IS UNAVAILABLE IN TWO REACHABLE REGIMES, so it
        cannot be the only guard against a broken floor:

        * **On CPU there is no cross-check.** `eager_batch_shape_control` is
          BITWISE identical to the full batch at every size from n=2 to n=128 on
          CPU, so its TV is exactly 0, the degeneracy escape fires, and the
          comparison never runs. Every CPU verify therefore runs `ulp-only`.
        * **At `ctl is None` there is no second estimate at all** — the batch is
          too small to re-chunk (bucket 1), or the control raised. Same outcome,
          different reason, and the detail line distinguishes them.

        That is why `_FLOOR_IMPLAUSIBLE_TV` exists: an ABSOLUTE bound on the
        floor, which needs no second estimate and would have caught the
        sentinel-mass defect (0.97) on its own.

        ⚑ THE DEGENERACY ESCAPE IS ONE-SIDED — it tests the SHAPE CONTROL, never
        the ULP estimate. See `_FLOOR_DEGENERATE_TV` for the measurement, and
        the F3 demonstration of what the two-sided version let through.

        ⚑ AND IT RETURNS THE ULP ESTIMATES ALONE, not `max(ulp, ctl)`. An
        earlier revision's comment said it fell back "to the ULP floor alone —
        the same thing the `ctl is None` path does" while the code still took
        the max, so a control declared a NON-MEASUREMENT was still allowed to
        WIDEN the floor whenever it was the larger of the two. That is exactly
        F3's mechanism, and it was invisible because the detail line still said
        `floor=shape+ulp`. The source label now says `ulp-only(ctl-degenerate)`.
        """
        ulp_p = _softmax(bf16_ulp_perturbation(ref_logits, seed=floor_seed))
        u_mean, u_tail = mean_row_tv(ulp_p, ref_p), tail_row_tv(ulp_p, ref_p)
        note = ""
        if ctl is None:
            f_mean, f_tail, src = u_mean, u_tail, "ulp-only"
        else:
            ctl_p = _softmax(ctl)
            c_mean, c_tail = mean_row_tv(ctl_p, ref_p), tail_row_tv(ctl_p, ref_p)
            # ⚑ DEGENERACY IS A RANGE, NOT AN EXACT ZERO — an earlier revision
            # tested `> 0.0`, the same "exact equality where a band was meant"
            # mistake the gate itself was built to stop. A control that merely
            # re-associates a couple of reductions gives a TV like 1.9e-8 —
            # mathematically the same softmax, numerically nonzero — and `hi/lo`
            # then reads ~60000 and FAILS A HEALTHY PACKAGE.
            if min(c_mean, c_tail) <= _FLOOR_DEGENERATE_TV:
                f_mean, f_tail, src = u_mean, u_tail, "ulp-only(ctl-degenerate)"
            else:
                # ⚑ BOTH STATISTICS ARE CROSS-CHECKED, not just the mean. The
                # TAIL sets the `_ROW_EXCEED_K` threshold, so a floor bug that
                # inflated only the tail would silently disable BOTH exceedance
                # arms — the gate's only power against single-row damage — while
                # a mean-only cross-check reported everything in order.
                for stat, u_stat, c_stat in (
                    ("mean", u_mean, c_mean), ("tail", u_tail, c_tail),
                ):
                    hi, lo = max(u_stat, c_stat), min(u_stat, c_stat)
                    if hi / lo > _FLOOR_DIVERGENCE_MAX:
                        note += (
                            f" ⚑FLOOR-DIVERGENCE[{stat}] ulp={u_stat:.5f} "
                            f"shape={c_stat:.5f} "
                            f"(x{hi / lo:.1f} > {_FLOOR_DIVERGENCE_MAX:g})"
                        )
                f_mean, f_tail = max(u_mean, c_mean), max(u_tail, c_tail)
                src = "shape+ulp"
        for stat, value in (("mean", f_mean), ("tail", f_tail)):
            if value > _FLOOR_IMPLAUSIBLE_TV:
                note += (
                    f" ⚑FLOOR-IMPLAUSIBLE[{stat}]={value:.4f} "
                    f"(> {_FLOOR_IMPLAUSIBLE_TV:g}; not bf16 noise)"
                )
        return f_mean, f_tail, note, src

    def _ratio(num: float, den: float) -> float:
        """No epsilon. A zero floor means the reference carries no scale, so the
        only honest verdict is exact equality — inf if it is not met.

        ⚑ ``inf``, NOT 0.0, and that is load-bearing: at a zero floor the
        exceedance arms also threshold at ``k * 0`` and would catch a nonzero
        numerator, so a mutant returning 0.0 here survives every verdict-only
        test. `tests/test_aot_verify_gate.py` asserts the arm VALUE, not just
        the verdict.

        ⚑ A ZERO FLOOR IS REACHABLE ON REAL WDL OUTPUT AT BUCKET 1. Measured on
        the production net: 12-17% of real WDL rows get a zero ULP floor at any
        one seed, because their three logits share a bf16 binade (equal ULPs)
        and the 2-in-8 draw that gives all three the same sign is a common shift
        the softmax cannot see. A real row of ``[-15.5625, -15.5625, -15.5625]``
        reads floor 0 at floor seeds 0 and 4 of 0..5 and ~2.8e-2 at the other
        four. At bucket >= 2 the batch mean/tail absorb it; at bucket 1 the
        floor IS that row, so ``wdl_mean`` reads ``inf`` and the bucket fails on
        a package that is one ULP off. Bucket 1 is not on the default ladder,
        but ``--buckets 1 --allow-incomplete`` reaches it and
        ``data/aot_models_512_bt4heads/chess_b1.pt2`` exists. Curing it means
        changing the floor from a single sign draw to something that cannot be
        a null draw, which re-banks every calibration in this file; it is filed
        rather than done here, and pinned by a test so it cannot drift silently.
        """
        if den > 0.0:
            return num / den
        return 0.0 if num == 0.0 else float("inf")

    ctl_p = ctl_pol if (ctl_pol is not None and ctl_wdl is not None) else None
    ctl_w = ctl_wdl if (ctl_pol is not None and ctl_wdl is not None) else None
    fp_mean, fp_tail, fp_note, fp_src = _floor(ref_pol, ref_pp, ctl_p)
    fw_mean, fw_tail, fw_note, fw_src = _floor(ref_wdl, ref_wp, ctl_w)

    # Three well-concentrated ratio arms. Each reads a MEDIAN of 1.00 on a
    # healthy package at every bucket; the worst reading is bucket-dependent and
    # is 2.24 at bucket 16 on `wdl_mean` — see the calibration block above for
    # the full table, for why the p99/max WDL arms are gone rather than
    # normalized, and for why no _ARM_NULL table is applied here any more.
    arms = {
        "pol_mean": _ratio(mean_row_tv(aot_pp, ref_pp), fp_mean),
        "pol_tail": _ratio(tail_row_tv(aot_pp, ref_pp), fp_tail),
        "wdl_mean": _ratio(mean_row_tv(aot_wp, ref_wp), fw_mean),
    }
    # ⚑ THE ROW-EXCEEDANCE ARM covers what a mean or a p99 structurally cannot:
    # corruption confined to a handful of rows — a kernel that damages only its
    # final boundary row at one exact bucket size. Measured, one bad row in a
    # 2048-row batch drives the mean arms DOWN to x0.02-0.03 (the other 2047
    # rows are bit-exact, so they dilute both the numerator and any tail
    # statistic), while leaving pooled argmax far above any sane threshold. So
    # the dilution is not merely unhelpful, it points the wrong way.
    #
    # This is a COUNT, not a ratio, and it is gated at zero. It is not scaled by
    # tv_ratio_max: `k` is already expressed in units of the package's own
    # floor, so the criterion is self-calibrating for the same reason the ratio
    # arms are, and loosening it with the same knob would double-count.
    exceed = {
        "pol_rows_over": int(np.sum(row_tv(aot_pp, ref_pp) > _ROW_EXCEED_K * fp_tail)),
        "wdl_rows_over": int(np.sum(row_tv(aot_wp, ref_wp) > _ROW_EXCEED_K * fw_tail)),
    }
    # A divergent floor FAILS rather than warns. The verdict is a ratio to the
    # floor, so an untrustworthy floor makes every arm meaningless — and a
    # verdict read off a failed instrument is not a verdict, in EITHER
    # direction. Reporting it and passing anyway would be the same "accepted
    # then silently ignored" defect the gate is built to catch.
    ok = (all(v <= tv_ratio_max for v in arms.values())
          and not any(exceed.values())
          and not (fp_note or fw_note))

    matches = int(np.sum(np.argmax(aot_pol, axis=-1) == np.argmax(ref_pol, axis=-1)))
    rows = int(ref_pol.shape[0])
    shown = " ".join(f"{k}=x{v:.2f}" for k, v in arms.items())
    shown += " " + " ".join(f"{k}={v}" for k, v in exceed.items())
    # ⚑ The floor SOURCE is per head, because degeneracy is: on CPU the WDL
    # control can be bitwise identical while the policy control is not. A single
    # label averaged over both heads is the "accepted then silently ignored"
    # shape — it would report `shape+ulp` for a bucket whose WDL arm was in fact
    # gated on the ULP estimate alone.
    detail = (
        f"{shown} floor=pol:{fp_src} wdl:{fw_src} argmax={matches}/{rows} "
        f"(tv_ratio_max={tv_ratio_max:g} k={_ROW_EXCEED_K:g}){fp_note}{fw_note}"
    )
    return ok, detail, matches, rows


def eager_batch_shape_control(
    model: torch.nn.Module, xt: torch.Tensor
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Same eager model, same rows, a DIFFERENT batch shape.

    This is the gate's denominator: the irreducible difference two runs of the
    SAME weights show purely because a different batch shape selects a different
    BLAS/Triton schedule and therefore a different float reduction order. AOT is
    not involved on either side.

    Measured 2026-08-15 at bucket 1190: eager-vs-eager TV **0.01760** against
    AOT-vs-eager **0.01737** — the packages add nothing beyond this floor. Same
    -shape reruns are bitwise identical (0.0), which is how the effect was
    attributed to shape rather than nondeterminism.

    ⚑ THIS ESTIMATE IS DEGENERATE ON SOME BACKENDS and must never be the sole
    floor. Measured on CPU with the real ``ChessNet`` topology it is BITWISE
    identical to the full batch at every size from n=2 to n=128, i.e. TV exactly
    0. That is why the gate takes the max of this and the analytic ULP floor,
    and why the two now cross-check each other in ``_floor`` — see
    ``_FLOOR_DIVERGENCE_MAX``.

    Returns ``(None, None)`` when the batch is too small to re-chunk into a
    different shape; the caller then runs on the ULP floor alone, says so in the
    detail line (``floor=ulp-only``), and loses the cross-check with it.
    """
    n = int(xt.shape[0])
    if n < 2:
        return None, None
    chunk = n // 2  # cheapest shape that is guaranteed != n
    pols: list[np.ndarray] = []
    wdls: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, n, chunk):
            o = model(xt[i : i + chunk])
            pols.append(_policy_output_full(o).detach().float().cpu().numpy())
            wdls.append(o["wdl"].detach().float().cpu().numpy())
    return np.concatenate(pols, axis=0), np.concatenate(wdls, axis=0)


def _real_position_batch(
    n: int,
    *,
    input_extra_features: str | None,
    input_history_encoding: str | None,
    seed: int,
) -> np.ndarray:
    """Encode ``n`` real board positions (random legal playouts from startpos).

    AOT/eager agreement must be judged on the deployment input distribution —
    sparse ~binary planes (~8% nonzero). ``standard_normal`` noise drives
    activations far off-distribution and amplifies benign bf16 kernel-ordering
    divergence into spurious verify FAILs.

    BOTH encoding keywords are required, and both are passed through. Only
    ``input_extra_features`` used to be: ``encode_positions_batch`` then fell
    back to the LEGACY history layout while the model is ``lc0_root_legacy_meta``
    (docs/rl_loop_audit.md M11 / encoding audit E2). The plane COUNT is identical
    either way (112 + n_extra), so the buffer shape, the package signature and
    the verify comparison all succeeded on the wrong content: measured on 200
    random-playout boards, 98 of 175 planes differ on at least one row, 92.0
    planes differ per row, 7.3 % of cells. The verdict — including
    ``argmax_min`` on the policy, the most distribution-sensitive metric here —
    was being read off a distribution the net never sees in deployment.
    """
    import chess

    rng = np.random.default_rng(int(seed))
    boards: list[chess.Board] = []
    for _ in range(int(n)):
        b = chess.Board()
        for _ in range(int(rng.integers(2, 50))):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(moves[int(rng.integers(0, len(moves)))])
        boards.append(b)
    x = encode_positions_batch(
        boards,
        input_extra_features=input_extra_features,
        input_history_encoding=input_history_encoding,
    )
    return np.ascontiguousarray(x, dtype=np.float32)


def verify_packages(
    *,
    out_dir: Path | str,
    model: torch.nn.Module,
    buckets: Sequence[int],
    max_batch: int,
    input_planes: int,
    tv_ratio_max: float,
    verify_n: int,
    seed: int,
    argmax_min: float = 0.90,
) -> tuple[int, int, list[tuple[int, str, str]]]:
    """Verify each built package against the eager reference.

    Loads each ``chess_b{N}.pt2`` **directly** (via ``_aoti_load_package`` +
    ``load_constants``) and runs it at its exact batch size ``N``. This is
    ladder-independent and mirrors the broker's real load path
    (``load_aot_packages``); it deliberately does NOT go through
    ``AOTEvaluator``, whose fixed ``_BATCH_BUCKETS`` ladder omits compiled-only
    sizes (e.g. 1190) and would fall back to a smaller package -> shape mismatch.

    The verification inputs are encoded in ``model``'s OWN declared encoding,
    read off the model rather than passed in: the encoding is not a free
    parameter of this gate, and a keyword that can be forgotten is exactly how
    the legacy-history bug (encoding audit E2) survived.

    What that buys, precisely (PR #321 review F5): the attributes are the values
    the model was CONSTRUCTED with, so they cannot drift from it the way a
    separately-threaded keyword can. It is NOT an extra guard against a model of
    unknown encoding — ``build_model`` normalizes ``None`` to ``legacy``/``v1``,
    so a real model always declares something and ``model_encoding_kwargs``'
    ``ValueError`` only fires for objects it did not build.

    Returns ``(verified_pass_count, failed_count, rows)`` where each row is
    ``(bucket, PASS|FAIL|SKIP, detail)``.
    """
    _ = int(max_batch), int(input_planes)  # kept for signature stability
    encoding = model_encoding_kwargs(model)
    out = Path(out_dir)
    device = str(next(model.parameters()).device)
    # Complete constant source: params + ALL buffers (incl. non-persistent),
    # since the packages externalize every constant. A bare state_dict() omits
    # non-persistent buffers -> unfilled package constant -> CUDA IMA.
    weight_source = model_constant_source(model)

    present = [b for b in buckets if package_path(out, int(b)).is_file()]
    missing = [b for b in buckets if int(b) not in {int(x) for x in present}]
    rows: list[tuple[int, str, str]] = [
        (int(b), "FAIL", "package missing") for b in missing
    ]

    n_pass = 0
    n_fail = len(missing)
    # Keyed on the declared FQN set, not on the bucket: the answer is a property
    # of the exported graph, and `assert_uniform_constant_fqns` already refuses a
    # package set that does not share one. A dir that somehow mixes two graphs
    # gets two probes rather than one silently reused.
    folded_cache: dict[frozenset[str], list[str]] = {}

    for bucket in present:
        b = int(bucket)
        # Load + rebind THIS exact package, then exercise it at batch == b.
        try:
            pkg = _aoti_load_package(str(package_path(out, b)))
            fqns = list(pkg.get_constant_fqns())
            if not fqns:
                raise RuntimeError(
                    f"bucket {b}: get_constant_fqns() empty — package is not "
                    "weight-updatable (rebuild with externalized constants)"
                )
            pkg.load_constants(
                build_aot_constants(weight_source, fqns, device=device),
                check_full_update=False,
            )
            all_ok = True
            details: list[str] = []
            argmax_matches = 0
            argmax_rows = 0
            for trial in range(int(verify_n)):
                x = _real_position_batch(
                    b,
                    input_extra_features=encoding["input_extra_features"],
                    input_history_encoding=encoding["input_history_encoding"],
                    seed=int(seed) + trial,
                )
                xt = torch.from_numpy(x).to(device=device, dtype=torch.bfloat16)
                # ⚑ The rebind above covers exactly the FQNs the package
                # DECLARES. Anything the graph folded is not in that list and is
                # not written by it — see `unrebindable_output_constants`. This
                # is the only check in the file that looks in that direction.
                # It reuses trial 0's rows rather than encoding its own, so it
                # costs no extra encode and probes the deployment distribution.
                key = frozenset(fqns)
                if trial == 0 and key not in folded_cache:
                    folded_cache[key] = unrebindable_output_constants(
                        model, fqns, probe=xt[:2]
                    )
                folded = folded_cache[key]
                if trial == 0 and folded:
                    # FAIL, not WARN. The ratio arms cannot see this at all on
                    # the vintage the package was built from — that is precisely
                    # when they read x1.0 — so a package that only fails HERE is
                    # the case the rest of the gate is blind to.
                    all_ok = False
                    details.append(
                        f"⚑UNREBINDABLE={len(folded)} {folded[:4]} "
                        "(constant-folded into the package; load_constants "
                        "cannot write them, so it drifts as the net trains)"
                    )
                with torch.no_grad():
                    out_aot = pkg(xt)
                    out_ref = model(xt)
                aot_pol = _policy_output_full(out_aot).detach().float().cpu().numpy()
                aot_wdl = out_aot["wdl"].detach().float().cpu().numpy()
                ref_pol = _policy_output_full(out_ref).detach().float().cpu().numpy()
                ref_wdl = out_ref["wdl"].detach().float().cpu().numpy()
                # The gate's denominator, recomputed per bucket from the CURRENT
                # weights — that is what stops it going stale as the net sharpens.
                ctl_pol, ctl_wdl = eager_batch_shape_control(model, xt)

                ok, detail, matches, rows_n = _compare_bucket(
                    aot_pol=aot_pol,
                    aot_wdl=aot_wdl,
                    ref_pol=ref_pol,
                    ref_wdl=ref_wdl,
                    ctl_pol=ctl_pol,
                    ctl_wdl=ctl_wdl,
                    tv_ratio_max=float(tv_ratio_max),
                    floor_seed=int(seed) + trial,
                )
                details.append(f"n{trial}:{detail}")
                argmax_matches += matches
                argmax_rows += rows_n
                if not ok:
                    all_ok = False

            # ⚑ The argmax criterion is applied ONCE on rows POOLED across every
            # trial, never per-trial. At the healthy per-row rate of 0.96, an
            # 8-row bucket needs 8/8 to clear 0.90 and spuriously FAILS 48% of
            # the time; pooling verify_n trials is what gives the threshold its
            # nominal meaning. This is also the only consumer of verify_n's
            # extra rows, so a verify_n that is silently ignored now changes a
            # verdict instead of changing nothing.
            argmax_rate = (argmax_matches / argmax_rows) if argmax_rows else 0.0
            if argmax_rate < float(argmax_min):
                all_ok = False
            details.append(
                f"pooled_argmax={argmax_matches}/{argmax_rows}"
                f"={argmax_rate:.4f} (argmax_min={float(argmax_min):g})"
            )
            status = "PASS" if all_ok else "FAIL"
            if all_ok:
                n_pass += 1
            else:
                n_fail += 1
            rows.append((b, status, "; ".join(details)))
        except Exception as exc:  # surface any verify fault as a bucket FAIL
            n_fail += 1
            rows.append((b, "FAIL", f"{type(exc).__name__}: {exc}"))

    return n_pass, n_fail, rows


def print_verify_table(rows: Sequence[tuple[int, str, str]]) -> None:
    """Print a compact per-bucket PASS/FAIL table."""
    print(f"{'bucket':>8} {'status':>6}  detail", flush=True)
    print("-" * 80, flush=True)
    for bucket, status, detail in rows:
        print(f"{bucket:>8} {status:>6}  {detail}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build AOTInductor chess_b{N}.pt2 packages for AOTEvaluator "
            "(weights-as-updatable-constants, max-autotune)."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pbt2_small.yaml"),
        help="Run YAML whose model: block defines topology (default: configs/pbt2_small.yaml)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="trainer.pt or model state_dict used to trace (any matching topology ckpt)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/aot_models_512"),
        help="Output directory for chess_b{N}.pt2 (default: data/aot_models_512)",
    )
    p.add_argument(
        "--max-batch",
        type=int,
        default=4096,
        help="Build/load only buckets <= this size (default: 4096)",
    )
    p.add_argument(
        "--buckets",
        type=str,
        default=None,
        help=(
            "Optional comma-separated bucket override (default: the broker's "
            "_COMPILED_BATCH_BUCKETS ladder). Must still cover every "
            "broker-required size unless --allow-incomplete is given."
        ),
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help=(
            "Permit a --buckets ladder that omits broker-required sizes. For "
            "bench/partial builds ONLY: the resulting directory runs eager for "
            "the omitted sizes, silently."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip buckets whose chess_b{N}.pt2 already exists",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="After build (or with --verify-only), compare AOT vs eager per bucket",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip building; only verify packages already in --out-dir",
    )
    p.add_argument(
        "--tv-ratio-max",
        type=float,
        default=2.0,
        help="Max TV(aot,eager) / TV(floor,eager) on each of the three ratio "
             "arms (pol_mean, pol_tail, wdl_mean). Unfilled constants read >10. "
             "Relative by construction, so it does not expire when the policy "
             "sharpens the way the old absolute --tol did. Does NOT loosen the "
             "row-exceedance arms, which are gated at zero in units of the "
             "package's own floor. "
             "MARGIN, re-measured 2026-08-15 at the production regime under an "
             "independently written +/-1 bf16-ULP healthy model (568 trials): "
             "the MEDIAN is 1.00 on every arm at every bucket, but the WORST "
             "reading is bucket-dependent -- 2.24 at bucket 16, 1.53 at 32, "
             "1.33 at 64, 1.21 at 128, and <=1.14 at 256 and above. Bucket 16 "
             "is the SMALLEST ON THE DEFAULT LADDER and it breaches 2.0 in "
             "2/200 healthy trials, all on wdl_mean (a mean over three columns "
             "of sixteen rows). A verify runs 2 trials there, so expect a ~2% "
             "false-fail rate per run at the default. That is reported rather "
             "than absorbed: raising the bound would weaken the arm at every "
             "other bucket, where it reads 1.00 +/- 0.03.",
    )
    p.add_argument(
        "--verify-n",
        type=int,
        default=2,
        help="Number of seeded random batches per bucket during --verify (default: 2)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for verify inputs (default: 0)",
    )
    p.add_argument(
        "--argmax-min",
        type=float,
        default=0.90,
        help="Minimum top-move match rate vs eager to PASS (default: 0.90; a "
             "correct package sits ~0.96 — near-tied moves flip under bf16 — "
             "while a broken one collapses toward random)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)

    if args.verify_only:
        args.verify = True

    if args.verify_n <= 0:
        print("error: --verify-n must be positive", file=sys.stderr)
        return 2
    # Same nan/inf hole as --tv-ratio-max, one arm over: `rate < nan` is False,
    # so `--argmax-min nan` silently retires the argmax check entirely.
    if not math.isfinite(args.argmax_min) or not 0.0 <= args.argmax_min <= 1.0:
        print("error: --argmax-min must be finite and in [0, 1]", file=sys.stderr)
        return 2
    if not math.isfinite(args.tv_ratio_max) or args.tv_ratio_max <= 0.0:
        # ⚑ isfinite, not just > 0: argparse's float() accepts "inf", and every
        # finite ratio then clears the bar — silently disabling all three ratio
        # arms while still reporting PASS. It also accepts "nan", and EVERY
        # comparison against nan is False, so `v <= nan` disables them just as
        # thoroughly while looking like an ordinary number in the log line.
        print("error: --tv-ratio-max must be positive and finite", file=sys.stderr)
        return 2
    # ⚑ NOT rejected at <=1.0. An earlier revision did, on the reasoning that
    # "the control IS the irreducible floor so <=1.0 can never pass" — which
    # this file's own measurement contradicts: a healthy package read 0.99
    # (AOT-vs-eager TV 0.01737 BELOW the control's 0.01760). The floor is an
    # estimate with its own variance, so ratios just under 1 are ordinary.

    try:
        buckets_override = (
            parse_buckets_arg(args.buckets) if args.buckets is not None else None
        )
        buckets = select_buckets(
            max_batch=int(args.max_batch),
            buckets=buckets_override,
            allow_incomplete=bool(args.allow_incomplete),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    built = 0
    skipped = 0
    verified = 0
    failed = 0
    out_dir = Path(args.out_dir)

    try:
        model_cfg = load_model_config(args.config)
        state_dict = load_checkpoint_state_dict(args.checkpoint)
        input_planes = int(input_plane_count(model_cfg.input_extra_features))

        if not args.verify_only:
            _require_cuda("building AOT packages")
            inductor_configs = build_inductor_configs()
            print(
                f"aot build: torch={torch.__version__} planes={input_planes} "
                f"buckets={list(buckets)} out={out_dir} "
                f"package_constants_in_so={inductor_configs['aot_inductor.package_constants_in_so']}",
                flush=True,
            )
            model = build_reference_model(model_cfg, state_dict, device="cuda")
            to_build, skipped_list = plan_build_buckets(
                buckets, out_dir, resume=bool(args.resume),
            )
            skipped = len(skipped_list)
            if skipped_list:
                print(
                    f"resume: skipping {len(skipped_list)} existing packages: "
                    f"{skipped_list}",
                    flush=True,
                )
            for bucket in to_build:
                compile_bucket_package(
                    model,
                    bucket=int(bucket),
                    input_planes=input_planes,
                    out_path=package_path(out_dir, int(bucket)),
                    inductor_configs=inductor_configs,
                    state_dict=model.state_dict(),
                )
                built += 1
            del model
            torch.cuda.empty_cache()
        else:
            print(
                f"aot verify-only: planes={input_planes} buckets={list(buckets)} "
                f"out={out_dir}",
                flush=True,
            )

        if args.verify:
            _require_cuda("verifying AOT packages")
            model = build_reference_model(model_cfg, state_dict, device="cuda")
            n_pass, n_fail, rows = verify_packages(
                out_dir=out_dir,
                model=model,
                buckets=buckets,
                max_batch=int(args.max_batch),
                input_planes=input_planes,
                tv_ratio_max=float(args.tv_ratio_max),
                verify_n=int(args.verify_n),
                seed=int(args.seed),
                argmax_min=float(args.argmax_min),
            )
            print_verify_table(rows)
            verified = n_pass
            failed = n_fail
            del model
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            format_summary_line(
                built=built, skipped=skipped, verified=verified, failed=failed,
                out_dir=out_dir,
            ),
            flush=True,
        )
        return 1

    summary = format_summary_line(
        built=built,
        skipped=skipped,
        verified=verified,
        failed=failed,
        out_dir=out_dir,
    )
    print(summary, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
