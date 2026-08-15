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
    as provenance, not as a calibration. ``--tv-ratio-max``'s default is now
    2.0, and it is no longer provisional in the way an earlier revision of this
    docstring said: the healthy distribution WAS banked on 2026-08-15, 96
    readings per arm across eight bucket sizes on production-shaped arrays (see
    the calibration block above ``_ROW_EXCEED_K``). What remains un-banked is
    the CUDA half — those 96 readings are synthetic-logit, so they calibrate the
    STATISTICS, not the hardware. A previous revision also quoted "2.31 (July
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
    # ⚑⚑ AND CENTER ON THE LIVE ENTRIES ONLY. The production policy array is
    # 4672 wide but the net emits 1858 real logits; `_policy_output_full` fills
    # the other 2814 slots (60% of the row!) with a -1e9 sentinel that softmax
    # sends to exactly 0. A plain row mean is dominated by that sentinel mass —
    # it lands near -6.0e8, so every REAL logit is centered to ~+6.0e8 and a
    # 2^-8 relative nudge moves it by ~2.3e6, obliterating the O(1) logit
    # spread. Measured 2026-08-15: the floor reads 0.0189 on the native 1858
    # array and 0.9719 on the padded 4672 one, 51x too large. Since a row TV is
    # bounded by 1, that CAPPED every policy arm below its own threshold and the
    # gate became arithmetically incapable of failing on the production shape —
    # a completely wrong policy passed with pol_mean x1.17, argmax 0/256.
    # Mean-centering is shift-invariant but NOT outlier-robust, and the padding
    # is 60% outliers. So: restrict both the center and the perturbation to the
    # entries that can affect the softmax at all, and leave the sentinels alone
    # (perturbing them is meaningless — exp(-1e9) is 0 either way).
    z = ref_logits.astype(np.float32)
    # e^-80 ~ 1.8e-35 of the top entry's mass: below float32 resolution, so an
    # entry this far down is inert in the softmax by construction, not by
    # threshold-picking. Guards the all-sentinel row (live is then everything).
    live = z > (np.max(z, axis=-1, keepdims=True) - np.float32(80.0))
    n_live = np.maximum(np.sum(live, axis=-1, keepdims=True), 1)
    mean_live = np.sum(np.where(live, z, np.float32(0.0)), axis=-1,
                       keepdims=True) / n_live
    z = z - mean_live  # shift-invariant: a common offset moves max and mean alike
    sign = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=z.shape)
    return np.where(live, z * (np.float32(1.0) + sign * step), z)


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
# `_ROW_EXCEED_K` below. So the surviving three arms are all well-concentrated
# estimators reading 1.00 +/- 0.02 when healthy, with worst observed 1.46
# against a default bound of 2.0. No normalization table is needed and none is
# kept: the table only ever existed to prop up the estimators now removed.
#
# tests/test_aot_verify_gate.py re-derives every number above on each CI run.

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

# Below this the empirical floor estimate is not a measurement. See `_floor`:
# a real bf16 floor here is ~0.018, an exactly-degenerate control is 0.0, and a
# control that only re-associates a few reductions lands at ~1e-8.
_FLOOR_DEGENERATE_TV = 1e-4


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
               ctl: np.ndarray | None) -> tuple[float, float, str]:
        """(mean, tail, note) of the irreducible floor: max over both estimates.

        Each statistic is computed on the floor with the SAME functional and the
        SAME row count as its numerator, so every arm reads ~1 when healthy.

        ⚑ The two estimates CROSS-CHECK each other, and the check is reported.
        They are independent derivations of "irreducible bf16 disagreement" and
        measured on real hardware they agree to 15% (0.0202 analytic vs 0.0176
        empirical at bucket 1190). A large divergence therefore means one of
        them is broken, and a BROKEN FLOOR IS THE ONE FAILURE THIS GATE CANNOT
        SELF-DETECT: it silently rescales every arm, in the passing direction if
        the floor is too big. This is not hypothetical — the sentinel-mass
        defect fixed in `bf16_ulp_perturbation` on 2026-08-15 inflated the
        policy floor 51x and made all three policy arms arithmetically incapable
        of failing. It would have shown up here as a 45x divergence.
        """
        ulp_p = _softmax(bf16_ulp_perturbation(ref_logits, seed=floor_seed))
        u_mean, u_tail = mean_row_tv(ulp_p, ref_p), tail_row_tv(ulp_p, ref_p)
        if ctl is None:
            return u_mean, u_tail, ""
        ctl_p = _softmax(ctl)
        c_mean, c_tail = mean_row_tv(ctl_p, ref_p), tail_row_tv(ctl_p, ref_p)
        note = ""
        hi, lo = max(u_mean, c_mean), min(u_mean, c_mean)
        # ⚑ DEGENERACY IS A RANGE, NOT AN EXACT ZERO — and an earlier revision
        # of this guard tested `lo > 0.0`, which is the same "exact equality
        # where a band was meant" mistake the gate itself was built to stop.
        # `eager_batch_shape_control` is BITWISE identical to the full batch on
        # some backends (TV exactly 0), and that case was handled. But a control
        # that merely re-associates a couple of reductions gives a TV like
        # 1.9e-8 — mathematically the same softmax, numerically nonzero — and
        # `hi/lo` then reads ~60000 and FAILS A HEALTHY PACKAGE. That is the
        # original defect (a gate that rejects good packages) reintroduced by
        # its own safety check.
        #
        # The ratio CANNOT separate the two dangerous-vs-harmless cases on its
        # own: both the near-degenerate control and the sentinel-inflated ULP
        # floor read `ulp >> shape`. What separates them is whether the SMALLER
        # estimate is a plausible bf16 floor at all. A real one is ~0.018 here;
        # 1e-4 sits ~200x below that and ~5000x above the numerical-noise case,
        # so it splits them with orders of magnitude to spare and is not a tuned
        # constant. Below it the control carries no information, so we fall back
        # to the ULP floor alone — the same thing the `ctl is None` path does —
        # rather than treating a non-measurement as a disagreement.
        if lo > _FLOOR_DEGENERATE_TV and hi / lo > _FLOOR_DIVERGENCE_MAX:
            note = (f" ⚑FLOOR-DIVERGENCE ulp={u_mean:.5f} shape={c_mean:.5f} "
                    f"(x{hi / lo:.1f} > {_FLOOR_DIVERGENCE_MAX:g})")
        return max(u_mean, c_mean), max(u_tail, c_tail), note

    def _ratio(num: float, den: float) -> float:
        """No epsilon. A zero floor means the reference carries no scale, so the
        only honest verdict is exact equality — inf if it is not met."""
        if den > 0.0:
            return num / den
        return 0.0 if num == 0.0 else float("inf")

    ctl_p = ctl_pol if (ctl_pol is not None and ctl_wdl is not None) else None
    ctl_w = ctl_wdl if (ctl_pol is not None and ctl_wdl is not None) else None
    fp_mean, fp_tail, fp_note = _floor(ref_pol, ref_pp, ctl_p)
    fw_mean, fw_tail, fw_note = _floor(ref_wdl, ref_wp, ctl_w)

    # Three well-concentrated ratio arms. Each reads 1.00 +/- 0.02 on a healthy
    # package and 1.46 at the worst of 96 measured healthy readings; see the
    # calibration block above for why the p99/max WDL arms are gone rather than
    # normalized, and why no _ARM_NULL table is applied here any more.
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
    src = "shape+ulp" if ctl_p is not None else "ulp-only"
    detail = (
        f"{shown} floor={src} argmax={matches}/{rows} "
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
             "arms (pol_mean, pol_tail, wdl_mean). All three read 1.00 +/- 0.02 "
             "when healthy; worst of 96 measured healthy readings across 8 "
             "bucket sizes is 1.46, and unfilled constants read >10. Relative "
             "by construction, so it does not expire when the policy sharpens "
             "the way the old absolute --tol did. Does NOT loosen the "
             "row-exceedance arms, which are gated at zero in units of the "
             "package's own floor.",
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
