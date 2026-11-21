#!/usr/bin/env python3
"""Per-model gates of one checkpoint (any model.pt, including train snapshots), as JSON.

    PYTHONPATH=. python scripts/simple_rl_model_gates.py --model <model.pt> --audit-rows <positions.npz> \
        [--collect <run dir> --gen g [--train-target-temp T]] [--device cuda] [--thresholds t.json] [--out x.json]

- ``audit``: on the fixed audit rows (an ``.npz`` whose ``x`` holds the encoded
  inputs, e.g. the 3,080-row D3 bank ``rl-audit/positions.npz``), the value
  head's confidence: ``mean_abs_q`` = mean |W - L| and ``mean_p_draw`` = mean D of
  softmax(``wdl`` logits), side to move. The model is loaded by the arena's own
  loader (``load_model_from_checkpoint``); on CUDA the forward runs under bf16
  autocast, as the banked value-recalibration references were made.
- ``heldout`` (with ``--collect``/``--gen``): policy CE against the stored
  target re-tempered at ``--train-target-temp`` by the loss's own
  ``retemper_main_policy_target``, plus policy entropy and value NLL, over the
  SAME held-out rows the train stage scores (generation g's games with
  ``is_holdout_game``, at the run's pinned ``holdout_frac``), by the train stage's
  own ``heldout_metrics``. At T = 1 it is the train stage's held-out CE.

``--thresholds FILE``: as in simple_rl_collect_gates (min / max / equals); a
failed gate, or a missing / NaN metric, exits 3.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.simple_rl_collect_gates import EXIT_GATE, check_gates


def audit_confidence(model: Any, x: np.ndarray, device: str, batch: int = 512) -> dict[str, Any]:
    import torch

    qs, ds = [], []
    model.eval()
    cuda = str(device).startswith("cuda")
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch])).float().to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda):
                out = model(xb)
            p = torch.softmax(out["wdl"].float(), dim=-1).double().cpu().numpy()
            qs.append(p[:, 0] - p[:, 2])
            ds.append(p[:, 1])
    q, d = np.concatenate(qs), np.concatenate(ds)
    return {"rows": len(q), "mean_abs_q": float(np.abs(q).mean()), "mean_p_draw": float(d.mean()),
            "autocast_bf16": cuda}


def run_holdout_frac(run_dir: Path, generation: int) -> float:
    """The holdout share the run's train stage used (its sealed manifest), else the pinned one."""
    from chess_anti_engine.simple_rl.common import read_manifest

    train = read_manifest(Path(run_dir) / f"gen_{int(generation):04d}" / "train")
    if train is not None:
        return float(train["config"]["settings"]["holdout_frac"])
    config = Path(run_dir) / "run_config.json"
    if config.is_file():
        return float(json.loads(config.read_text())["settings"]["holdout_frac"])
    raise SystemExit(f"{run_dir}: no sealed gen {generation} train and no run_config.json to read holdout_frac from")


def heldout_rows(run_dir: Path, generation: int, holdout_frac: float) -> dict[str, np.ndarray]:
    """Generation g's held-out rows exactly as learn.load_window selects them."""
    from chess_anti_engine.simple_rl.collect import load_generation_rows_where
    from chess_anti_engine.simple_rl.learn import TRAIN_KEYS, holdout_mask

    collect = Path(run_dir) / f"gen_{int(generation):04d}" / "collect"
    if not (collect / "manifest.json").is_file():
        raise SystemExit(f"{collect} is not sealed")
    # Only the held-out rows are ever held (one shard at a time), as load_window does.
    arrays, _meta = load_generation_rows_where(collect, lambda m: holdout_mask(m, holdout_frac), keys=TRAIN_KEYS)
    return arrays


def retempered_heldout(model: Any, rows: dict[str, np.ndarray], temp: float, device: str) -> dict[str, Any]:
    import torch

    from chess_anti_engine.simple_rl.learn import heldout_metrics
    from chess_anti_engine.train.losses import retemper_main_policy_target

    target = torch.from_numpy(np.asarray(rows["policy_target"], dtype=np.float32))
    rows = {**rows, "policy_target": retemper_main_policy_target(target, temp=float(temp)).numpy()}
    return {"train_target_temp": float(temp), **heldout_metrics(model, rows, device)}


def model_gates(*, model_path: Path, audit_rows: Path | None, audit_key: str, collect: Path | None,
                generation: int | None, temp: float, device: str) -> dict[str, Any]:
    from chess_anti_engine.simple_rl.common import sha256_file
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    model = load_model_from_checkpoint(model_path, device=device)
    report: dict[str, Any] = {"model": str(model_path), "model_sha256": sha256_file(model_path), "device": device}
    if audit_rows is not None:
        with np.load(audit_rows) as z:
            x = z[audit_key]
        report["audit"] = {"path": str(audit_rows), "sha256": sha256_file(audit_rows), "key": audit_key,
                           **audit_confidence(model, x, device)}
    if collect is not None:
        if generation is None:
            raise SystemExit("--collect needs --gen")
        frac = run_holdout_frac(collect, generation)
        report["heldout"] = {"run_dir": str(collect), "generation": int(generation), "holdout_frac": frac,
                             **retempered_heldout(model, heldout_rows(collect, generation, frac), temp, device)}
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="any checkpoint with model + arch (a train model.pt or snapshot)")
    p.add_argument("--audit-rows", default=None, help=".npz of fixed audit rows (encoded inputs under --audit-key)")
    p.add_argument("--audit-key", default="x")
    p.add_argument("--collect", default=None, metavar="RUN_DIR", help="run dir whose generation --gen holdout to score")
    p.add_argument("--gen", type=int, default=None)
    p.add_argument("--train-target-temp", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--thresholds", default=None, help="JSON {metric.path: {min, max} | {equals}}; exit 3 when one fails")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    if a.audit_rows is None and a.collect is None:
        raise SystemExit("nothing to measure: pass --audit-rows and/or --collect/--gen")
    report = model_gates(model_path=Path(a.model), audit_rows=Path(a.audit_rows) if a.audit_rows else None,
                         audit_key=a.audit_key, collect=Path(a.collect) if a.collect else None, generation=a.gen,
                         temp=a.train_target_temp, device=a.device)
    code = 0
    if a.thresholds:
        gates = check_gates(report, json.loads(Path(a.thresholds).read_text()))
        report["gates"] = gates
        report["gates_passed"] = all(g["pass"] for g in gates)
        code = 0 if report["gates_passed"] else EXIT_GATE
    text = json.dumps(report, indent=1, sort_keys=True)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
