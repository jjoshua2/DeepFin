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


def _compare_bucket(
    *,
    aot_pol: np.ndarray,
    aot_wdl: np.ndarray,
    ref_pol: np.ndarray,
    ref_wdl: np.ndarray,
    pol_tol: float,
    wdl_tol: float,
    argmax_min: float,
) -> tuple[bool, str]:
    """Return (pass, detail) for one bucket comparison.

    Compares in **probability space** (softmax), the quantities the broker
    actually consumes — MCTS priors and value WDL. Raw-logit max-abs-diff is a
    useless criterion here: ``_policy_output_full`` leaves masked illegal-move
    slots at a large sentinel whose bf16-vs-max-autotune representation differs
    by ~1e6, dwarfing every real difference. In prob space AOT-bf16 tracks
    eager-bf16 to ~2e-3 (policy) / ~3e-2 (wdl) on real positions. ``argmax_min``
    is a loose floor only — near-tied top moves flip under bf16 without any
    quality loss, so a correct package sits ~0.96 while a broken one (wrong/
    unfilled constants -> garbage) collapses toward random.
    """
    aot_pp, ref_pp = _softmax(aot_pol), _softmax(ref_pol)
    aot_wp, ref_wp = _softmax(aot_wdl), _softmax(ref_wdl)
    pol_pmad = float(np.max(np.abs(aot_pp - ref_pp)))
    wdl_pmad = float(np.max(np.abs(aot_wp - ref_wp)))
    argmax_rate = float(
        np.mean(np.argmax(aot_pol, axis=-1) == np.argmax(ref_pol, axis=-1))
    )
    ok = pol_pmad <= pol_tol and wdl_pmad <= wdl_tol and argmax_rate >= argmax_min
    detail = (
        f"pol_pmad={pol_pmad:.4g} wdl_pmad={wdl_pmad:.4g} "
        f"argmax_rate={argmax_rate:.4f} "
        f"(pol_tol={pol_tol:g} wdl_tol={wdl_tol:g} argmax_min={argmax_min:g})"
    )
    return ok, detail


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
    pol_tol: float,
    wdl_tol: float,
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

                ok, detail = _compare_bucket(
                    aot_pol=aot_pol,
                    aot_wdl=aot_wdl,
                    ref_pol=ref_pol,
                    ref_wdl=ref_wdl,
                    pol_tol=float(pol_tol),
                    wdl_tol=float(wdl_tol),
                    argmax_min=float(argmax_min),
                )
                details.append(f"n{trial}:{detail}")
                if not ok:
                    all_ok = False
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
        "--tol",
        type=float,
        default=2e-2,
        help="Max abs diff for policy softmax PROBABILITIES vs eager (default: "
             "2e-2; measured ~2e-3 AOT-vs-eager-bf16 on real positions)",
    )
    p.add_argument(
        "--wdl-tol",
        type=float,
        default=8e-2,
        help="Max abs diff for WDL softmax probabilities vs eager (default: "
             "8e-2; measured up to ~6e-2 AOT-vs-eager-bf16 on the largest "
             "buckets — divergence grows with batch; a broken package is >>1e-1)",
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
    if args.tol <= 0:
        print("error: --tol must be positive", file=sys.stderr)
        return 2

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
                pol_tol=float(args.tol),
                wdl_tol=float(args.wdl_tol),
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
