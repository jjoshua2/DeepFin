#!/usr/bin/env python3
"""Fit the two value-recalibration parameters (T, b) for ``--<seat>-value-recal``.

The recalibration maps a net's WDL logits (W, D, L, side-to-move POV) to
``softmax(logits / T + [0, b, 0])``. This tool fits T and b so the
recalibrated net matches a REFERENCE net's WDL on a set of rows, by minimising
the mean cross-entropy ``-sum(p_ref * log p_net)`` over the rows. This is the
head-swap bank's ``ref=base|temp_drawbias|fitCE`` fit (``value-recalibration/
recalibrate.py``), with its exact search: a coarse grid over
``T = exp(linspace(log 0.25, log 4, 161))[::4]`` and ``b = linspace(-3, 3,
121)[::4]``, then 40 rounds of a 3 x 3 coordinate refine with halving steps
(T multiplicative, start 0.15 in log space; b additive, start 0.3). Given the
bank's banked logits it reproduces the bank's parameters.

Rows, one of:

* ``--collect-dir DIR``: a sealed simple-RL collect; only its HOLD-OUT games
  (``is_holdout_game(game_id, --holdout-frac)``, the split the learner never
  trains on) are used. A pilot collect plays the openings split's TRAIN lines
  and its match plays ``eval.pgn``; the split's orbit rule keeps every TRAIN
  game off the EVAL starts, so these rows are disjoint from the match openings
  by construction.
* ``--rows FILE.npz`` with an ``x`` array (N, planes, 8, 8).

Pre-register which rows before the match, and fit them before it is played.
The output JSON records both checkpoints' sha256, the rows' source and digest,
the objective before and after, and the exact ``--<seat>-value-recal`` value.

    PYTHONPATH=. python scripts/fit_value_recal.py --net Zq/model.pt \\
        --reference sl_base.pt --collect-dir RUN/gen_0001/collect \\
        --device cuda --out Zq_recal.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

TGRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 161))
BGRID = np.linspace(-3.0, 3.0, 121)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def recal_probs(logits: np.ndarray, t: float, b: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / t
    z = z.copy()
    z[..., 1] += b
    return _softmax(z)


def cross_entropy(logits: np.ndarray, ref_probs: np.ndarray, t: float, b: float) -> float:
    p = recal_probs(logits, t, b)
    return float(-(ref_probs * np.log(np.maximum(p, 1e-300))).sum(-1).mean())


def fit_temp_drawbias(logits: np.ndarray, ref_probs: np.ndarray) -> dict[str, float]:
    """(T, b) minimising the cross-entropy to ``ref_probs``; the bank's search, exactly."""
    logits = np.asarray(logits, dtype=np.float64)
    ref_probs = np.asarray(ref_probs, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 3 or ref_probs.shape != logits.shape:
        raise ValueError(f"need (N, 3) logits and reference probabilities, got {logits.shape} / {ref_probs.shape}")
    best = (np.inf, 1.0, 0.0)
    for t in TGRID[::4]:
        for b in BGRID[::4]:
            o = cross_entropy(logits, ref_probs, float(t), float(b))
            if o < best[0]:
                best = (o, float(t), float(b))
    _, t, b = best
    st, sb = 0.15, 0.3
    for _ in range(40):
        cand = [(t * float(np.exp(dt)), b + db) for dt in (-st, 0, st) for db in (-sb, 0, sb)]
        objs = [cross_entropy(logits, ref_probs, ct, cb) for ct, cb in cand]
        k = int(np.argmin(objs))
        if cand[k] == (t, b):
            st /= 2
            sb /= 2
        t, b = cand[k]
    return {"T": float(t), "b": float(b)}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if (args.rows is None) == (args.collect_dir is None):
        raise SystemExit("pass exactly one of --rows or --collect-dir")
    if args.rows is not None:
        with np.load(args.rows) as z:
            if "x" not in z.files:
                raise SystemExit(f"{args.rows} has no 'x' array")
            x = np.asarray(z["x"])
        return x, {"rows_file": str(args.rows), "rows_sha256": _sha256(args.rows)}
    from chess_anti_engine.simple_rl.collect import load_generation_rows
    from chess_anti_engine.simple_rl.learn import holdout_mask

    collect = Path(args.collect_dir)
    manifest = collect / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"{collect} is not a sealed collect (no manifest.json)")
    arrays, meta = load_generation_rows(collect, keys=("x",))
    held = holdout_mask(meta, float(args.holdout_frac))
    if not held.any():
        raise SystemExit(f"{collect} has no hold-out games at holdout_frac {args.holdout_frac}")
    return np.asarray(arrays["x"])[held], {
        "collect_dir": str(collect), "collect_manifest_sha256": _sha256(manifest),
        "holdout_frac": float(args.holdout_frac), "holdout_games": int(np.unique(meta["game_id"][held]).size),
    }


def wdl_logits(ckpt: str, x: np.ndarray, *, device: str, batch: int) -> np.ndarray:
    """The checkpoint's raw WDL logits, through the arena's own loader and forward."""
    import torch

    from chess_anti_engine.inference import _forward_no_grad
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    model = load_model_from_checkpoint(ckpt, device=device)
    if hasattr(model, "_inference_only"):
        setattr(model, "_inference_only", True)
    out = []
    for i in range(0, x.shape[0], batch):
        xt = torch.from_numpy(np.ascontiguousarray(x[i:i + batch], dtype=np.float32)).to(device)
        out.append(_forward_no_grad(model, xt, device=device)["wdl"].float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", required=True, help="the checkpoint to recalibrate (the seat's value net)")
    p.add_argument("--reference", required=True, help="the checkpoint whose WDL the fit matches (the SL base)")
    p.add_argument("--rows", type=Path, default=None, help="npz with an 'x' array")
    p.add_argument("--collect-dir", type=Path, default=None, help="a sealed simple-RL collect; hold-out games only")
    p.add_argument("--holdout-frac", type=float, default=0.05, help="the learner's holdout_frac (default 0.05)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    x, rows_record = load_rows(args)
    if x.shape[0] == 0:
        raise SystemExit("no rows to fit on")
    net_logits = wdl_logits(args.net, x, device=args.device, batch=int(args.batch))
    ref_probs = _softmax(wdl_logits(args.reference, x, device=args.device, batch=int(args.batch)))
    fit = fit_temp_drawbias(net_logits, ref_probs)
    record = {
        "method": "temp_drawbias, cross-entropy to the reference net's WDL (bank: ref=base|temp_drawbias|fitCE)",
        "apply": "p = softmax(logits / T + [0, b, 0]); logits W,D,L side-to-move",
        "T": fit["T"], "b": fit["b"],
        "cli": f"{fit['T']!r},{fit['b']!r}",
        "net": str(args.net), "net_sha256": _sha256(Path(args.net)),
        "reference": str(args.reference), "reference_sha256": _sha256(Path(args.reference)),
        "rows": int(x.shape[0]), **rows_record,
        "cross_entropy_identity": cross_entropy(net_logits, ref_probs, 1.0, 0.0),
        "cross_entropy_fitted": cross_entropy(net_logits, ref_probs, fit["T"], fit["b"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    print(f"[fit_value_recal] T={fit['T']:.6f} b={fit['b']:.6f} on {x.shape[0]} rows; "
          f"CE {record['cross_entropy_identity']:.6f} -> {record['cross_entropy_fitted']:.6f}; "
          f"pass --<seat>-value-recal {record['cli']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
