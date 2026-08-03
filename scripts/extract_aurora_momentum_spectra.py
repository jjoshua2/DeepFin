#!/usr/bin/env python3
"""Bank the singular-value spectra of a checkpoint's Aurora momentum buffers.

READ-ONLY against ``runs/``: it opens one ``trainer.pt``, copies the matrix
group's ``momentum_buffer`` tensors out, and writes only their spectra.

Why spectra and not the tensors. ``tests/test_aurora_polar_convergence.py``
needs the real conditioning of production momentum to pin
``polar_convergence`` against `MODEL_OPT_AUDIT.md` Addendum II B3, and the
tensors themselves are 23 MB. They do not have to be checked in, because the
Polar Express iterate is ORTHOGONALLY EQUIVARIANT: every step is
``x <- a x + (b xx^T + c (xx^T)^2) x``, so for ``M = U S V^T`` the result is
``U p(S) V^T`` and the output spectrum is a function of the INPUT SPECTRUM
ALONE. A matrix rebuilt as ``U diag(s) V^T`` from any orthogonal ``U``/``V``
therefore reproduces the reference readings exactly -- verified across three
reconstruction seeds and both float32 and float64, all four decimals stable.

That equivalence holds for the polar factor, which is what the reference table
measures. It does NOT hold through the rectangular branch's row-normalisation
loop (`_aurora_update` with ``rows != cols``), which is basis-dependent, so the
test pins the rectangular numbers on the polar path only.

Usage:
    PYTHONPATH=. python3 scripts/extract_aurora_momentum_spectra.py \\
        --checkpoint runs/.../checkpoint_000478/trainer.pt \\
        --out tests/data/aurora_polar_momentum_spectra.npz
"""
from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Any

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="trainer.pt to read")
    parser.add_argument("--out", required=True, type=Path, help="npz to write")
    parser.add_argument(
        "--rect-limit", type=int, default=4,
        help="how many rectangular tensors to keep (all square tensors are kept)",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint
    if not ckpt.exists():
        raise SystemExit(f"no such checkpoint: {ckpt}")
    print(f"reading (READ-ONLY) {ckpt}")
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)

    opt = state["opt"]
    aurora_groups = [g for g in opt["param_groups"] if g.get("use_aurora")]
    if len(aurora_groups) != 1:
        raise SystemExit(f"expected exactly one Aurora group, got {len(aurora_groups)}")
    group = aurora_groups[0]
    realized = {
        k: group[k]
        for k in sorted(group)
        if k.startswith("aurora_") or k in ("lr", "weight_decay")
    }
    print(f"realized group hyperparameters: {realized}")

    opt_state = opt["state"]
    shapes: collections.Counter[tuple[int, ...]] = collections.Counter()
    buffers: dict[str, torch.Tensor] = {}
    missing = 0
    for pid in group["params"]:
        entry = opt_state.get(pid)
        if not entry or "momentum_buffer" not in entry:
            missing += 1
            continue
        buf = entry["momentum_buffer"].detach().to(torch.float32)
        shapes[tuple(buf.shape)] += 1
        buffers[f"idx{int(pid):04d}"] = buf
    print(f"group: {len(group['params'])} params, {missing} without a momentum_buffer")
    print(f"shapes: {dict(shapes)}")

    square = {k: v for k, v in buffers.items() if v.shape[0] == v.shape[1]}
    rect = {k: v for k, v in buffers.items() if v.shape[0] != v.shape[1]}
    selected = dict(square)
    for name in sorted(rect)[: max(0, int(args.rect_limit))]:
        selected[name] = rect[name]
    print(f"selected {len(square)} square + {len(selected) - len(square)} rectangular")

  # `Any` values, not `np.ndarray`: `savez_compressed(**payload)` is typed
  # `(*args, allow_pickle: bool, **kwds)`, so a narrower value type makes the
  # splat look like it could bind `allow_pickle`.
    payload: dict[str, Any] = {}
    rows: list[str] = []
    for name in sorted(selected):
        mat = selected[name]
        svals = torch.linalg.svdvals(mat.double())
        payload[f"s_{name}"] = svals.numpy()
        rows.append(f"{name},{int(mat.shape[0])},{int(mat.shape[1])}")
        print(
            f"  {name} {tuple(mat.shape)} sigma_max={float(svals[0]):.6e} "
            f"sigma_min={float(svals[-1]):.6e} kappa={float(svals[0] / svals[-1]):.3e}",
        )
    payload["shapes"] = np.array(rows, dtype=np.str_)
    payload["provenance"] = np.array(
        [f"checkpoint={ckpt}", f"realized_group={realized}"], dtype=np.str_,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"wrote {len(rows)} spectra -> {args.out} ({args.out.stat().st_size / 1e3:.1f} kB)")


if __name__ == "__main__":
    main()
