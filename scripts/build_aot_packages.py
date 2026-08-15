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
    _BATCH_BUCKETS,
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


def select_buckets(
    *,
    max_batch: int,
    buckets: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Return bucket sizes to build, filtered by ``max_batch``.

    Default source is :data:`_BATCH_BUCKETS` (authoritative AOT ladder).
    """
    if int(max_batch) <= 0:
        raise ValueError(f"--max-batch must be positive, got {max_batch}")
    source: Sequence[int] = buckets if buckets is not None else _BATCH_BUCKETS
    selected = tuple(int(b) for b in source if int(b) <= int(max_batch))
    if not selected:
        raise ValueError(
            f"no buckets <= max_batch={max_batch} "
            f"(source={list(source)!r})"
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
    """``ref_logits`` nudged by ±1 bf16 ULP — the FLOOR that never degenerates.

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
    each logit moved by one unit in the last place (bf16 keeps 8 mantissa bits,
    so the relative spacing is 2^-8). It is positive for any non-constant
    logits, it is backend-independent, and — the property the whole gate turns
    on — it **tracks sharpness**, because a fixed relative logit perturbation
    produces a larger probability movement as the softmax concentrates.

    It also corroborates the empirical control rather than replacing it. At the
    net's realized sharpness (top-1 ~0.56-0.59) this floor reads TV **0.0202**
    against the CUDA batch-shape control's **0.0176** at bucket 1190 — two
    independent derivations of "irreducible bf16 disagreement", agreeing to
    15%. The gate uses the LARGER of the two, since the true floor is at least
    each of them.

    ⚑ WHAT IS AND IS NOT BANKED. The CPU-derivable half of every claim above is
    re-measured on each CI run by ``tests/test_aot_verify_gate.py`` — the
    round-trip no-op, non-degeneracy, sharpness tracking, and the 0.0202 value
    itself. The CUDA numbers (0.0176 control and 0.01737 AOT-vs-eager at bucket
    1190) are single-run readings on hardware CI does not have; they are stated
    as provenance, not as a calibration, and ``--tv-ratio-max``'s 1.5 default is
    PROVISIONAL until a real calibration run banks a distribution of healthy
    ratios across the ladder. A previous revision also quoted "2.31 (July
    packages on August weights, stale)" as support for 1.5. That number is
    withdrawn from the justification: it was never banked, and it argues against
    itself — 2.31 > 1.5 means the new gate FAILS the very packages the rewrite
    was meant to stop condemning, which would relabel the verdict rather than
    correct it.
    """
    rng = np.random.default_rng(int(seed))
    step = np.float32(2.0) ** -8  # bf16: 1 implicit + 7 explicit mantissa bits
    # ⚑ CENTER FIRST. A relative perturbation of the RAW logits depends on their
    # arbitrary common offset: adding a constant to every logit leaves the
    # softmax — and the measured AOT-vs-eager TV — completely unchanged, but
    # scales this floor by ~offset/256. A head bias drifting in that common-mode
    # direction would then widen the denominator and let the same package
    # discrepancy start passing. That is the WEIGHT-DEPENDENT VERDICT this gate
    # exists to eliminate, reintroduced through the back door.
    # ⚑ Centered by the row MEAN, not the row max. Both are shift-invariant,
    # but max-centering pins the top logit at exactly 0, so the perturbation of
    # the single entry that dominates the softmax is exactly 0 too — measured,
    # that shrinks the floor 12x (0.0205 -> 0.00173) and would make the gate
    # spuriously strict. Mean-centering privileges no entry and preserves the
    # magnitude: 0.0205 centered vs 0.0194 raw at zero offset, and flat at
    # 0.0203/0.0204 for offsets of +10 and +100 where the raw form inflates to
    # 0.031 and 0.133.
    z = ref_logits.astype(np.float32)
    z = z - np.mean(z, axis=-1, keepdims=True)
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=z.shape)
    return z * (np.float32(1.0) + sign * step)


# ⚑⚑ NULL CALIBRATION — measured, not chosen.
#
# Each arm divides a statistic of (AOT vs eager) by the SAME statistic of the
# floor. That makes every arm scale-free, but it does NOT make them read the
# same number when healthy: numerator and denominator are different random
# draws, and a p99 or a max of a 3-column WDL row-TV is a far noisier estimator
# than a mean over 1858 columns. Measured on healthy synthetic packages
# perturbed at exactly the bf16 ULP scale, 40 seeds x buckets {64, 512, 4096}:
#
#     arm        median      worst observed
#     pol_mean     0.91          1.06
#     pol_tail     1.19-1.29     1.82
#     wdl_mean     1.15          1.70
#     wdl_tail     1.88-1.98     3.14
#     pol_max      1.25-1.71     2.37
#     wdl_max      2.01-2.34     3.64
#
# ⇒ a SINGLE threshold across all six arms is wrong, and an earlier revision of
# this file shipped exactly that: at 1.5 it would have spuriously FAILED a
# perfectly healthy package on wdl_tail (1.90) and wdl_max (2.07) every time.
# That is the original defect — a gate that fails good packages — wearing a
# different hat.
#
# So each arm is normalized by its own measured null before the threshold is
# applied. After normalization every arm reads ~1.0 when healthy and the worst
# observed reading across the whole sweep is 1.65, so the default bound of 2.0
# carries real margin while a package with unfilled constants reads >10.
#
# tests/test_aot_verify_gate.py re-derives these on every CI run.
_ARM_NULL = {
    "pol_mean": 0.91,
    "pol_tail": 1.30,
    "wdl_mean": 1.15,
    "wdl_tail": 1.90,
    "pol_max": 1.70,
    "wdl_max": 2.30,
}


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
    ``max |p_aot - p_ref| <= 0.02`` over ``N x 1858`` probabilities. It broke on
    2026-08-15, and it broke on the **WEIGHTS, not the packages**: the identical
    month-deployed ``data/aot_models_512`` files read 0.0015-0.0052 (PASS) on
    July weights and up to **0.175** (FAIL) on August weights. The max always
    landed on a row's top-1 entry, and the net's top-1 probability had grown
    4.4x (max 0.221 -> 0.980) as the policy concentrated. **An absolute
    probability tolerance is pinned to a sharpness regime and expires when the
    net sharpens.**

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
    * :func:`bf16_ulp_perturbation` — analytic, every logit moved one bf16 ULP.
      Never degenerate, backend-independent, and tracks sharpness.

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
               ctl: np.ndarray | None) -> tuple[float, float, float]:
        """(mean, tail, max) of the irreducible floor: max over both estimates.

        Each statistic is computed on the floor with the SAME functional and the
        SAME row count as its numerator, so every arm reads ~1 when healthy.
        """
        ulp_p = _softmax(bf16_ulp_perturbation(ref_logits, seed=floor_seed))
        f = [mean_row_tv(ulp_p, ref_p), tail_row_tv(ulp_p, ref_p),
             float(np.max(row_tv(ulp_p, ref_p)))]
        if ctl is not None:
            ctl_p = _softmax(ctl)
            f = [max(f[0], mean_row_tv(ctl_p, ref_p)),
                 max(f[1], tail_row_tv(ctl_p, ref_p)),
                 max(f[2], float(np.max(row_tv(ctl_p, ref_p))))]
        return f[0], f[1], f[2]

    def _ratio(num: float, den: float) -> float:
        """No epsilon. A zero floor means the reference carries no scale, so the
        only honest verdict is exact equality — inf if it is not met."""
        if den > 0.0:
            return num / den
        return 0.0 if num == 0.0 else float("inf")

    ctl_p = ctl_pol if (ctl_pol is not None and ctl_wdl is not None) else None
    ctl_w = ctl_wdl if (ctl_pol is not None and ctl_wdl is not None) else None
    fp_mean, fp_tail, fp_max = _floor(ref_pol, ref_pp, ctl_p)
    fw_mean, fw_tail, fw_max = _floor(ref_wdl, ref_wp, ctl_w)

    raw = {
        "pol_mean": _ratio(mean_row_tv(aot_pp, ref_pp), fp_mean),
        "pol_tail": _ratio(tail_row_tv(aot_pp, ref_pp), fp_tail),
        "wdl_mean": _ratio(mean_row_tv(aot_wp, ref_wp), fw_mean),
        "wdl_tail": _ratio(tail_row_tv(aot_wp, ref_wp), fw_tail),
        "pol_max": _ratio(float(np.max(row_tv(aot_pp, ref_pp))), fp_max),
        "wdl_max": _ratio(float(np.max(row_tv(aot_wp, ref_wp))), fw_max),
    }
    # Normalize by each arm's measured null so one threshold means the same
    # thing on all six. See _ARM_NULL.
    arms = {k: v / _ARM_NULL[k] for k, v in raw.items()}
    # ⚑ THE WORST-ROW ARMS (pol_max / wdl_max) exist because p99 by
    # construction ignores corruption confined to fewer than 1% of rows — a
    # kernel that damages only its final boundary row at one exact bucket size,
    # whose contribution to the mean is diluted by the whole batch and which
    # leaves the pooled argmax far above 0.90.
    #
    # A bare max was the ORIGINAL defect (an extreme-value statistic that grows
    # with N, so big buckets failed for being big). A RATIO OF TWO MAXIMA OVER
    # THE SAME N does not have that problem: both are the same order statistic
    # on the same row count, so the N-dependence largely cancels — and whatever
    # does not cancel is absorbed by _ARM_NULL, which was measured across three
    # bucket sizes for exactly that reason.
    ok = all(v <= tv_ratio_max for v in arms.values())

    matches = int(np.sum(np.argmax(aot_pol, axis=-1) == np.argmax(ref_pol, axis=-1)))
    rows = int(ref_pol.shape[0])
    shown = " ".join(f"{k}=x{v:.2f}" for k, v in arms.items())
    src = "shape+ulp" if ctl_p is not None else "ulp-only"
    detail = (
        f"{shown} floor={src} argmax={matches}/{rows} "
        f"(tv_ratio_max={tv_ratio_max:g})"
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

    Returns ``(None, None)`` when the batch is too small to re-chunk into a
    different shape; the caller degrades to the argmax floor and says so.
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
        help="Optional comma-separated bucket override (default: _BATCH_BUCKETS)",
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
        help="Max TV(aot,eager) / TV(eager-at-another-batch-shape,eager). "
             "Applied to arms NORMALIZED by their measured null (see _ARM_NULL), so "
             "1.0 is the healthy level on every arm and 2.0 is the default "
             "bound. Worst healthy reading observed across 40 seeds x buckets "
             "{64,512,4096} is 1.65; unfilled constants read >10. Relative by "
             "construction, so it does "
             "not expire when the policy sharpens the way the old absolute "
             "--tol did.",
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
    if not math.isfinite(args.tv_ratio_max) or args.tv_ratio_max <= 0.0:
        # ⚑ isfinite, not just > 0: argparse's float() accepts "inf", and every
        # finite ratio then clears the bar — silently disabling all six TV arms
        # while still reporting PASS. A package with completely wrong WDL and a
        # preserved policy argmax would verify clean.
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
        buckets = select_buckets(max_batch=int(args.max_batch), buckets=buckets_override)
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
