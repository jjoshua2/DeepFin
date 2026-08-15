"""EXECUTION proof of reachability: does sf_p0_regret_t move the gradient?

Runs the REAL `compute_loss` on CPU with REAL sf_p0_regret rows pulled from the
live replay shards, and asks the only question that matters: perturb
`sf_p0_regret_t` and see whether `total` and d(total)/d(logits) move.

Arms:
  A  w_sf_own_regret = 0.0  (the LIVE value in configs/pbt2_small.yaml)
  B  w_sf_own_regret = 0.7  (the value live from 2026-07-27 to 2026-08-06,
                             commit ed9de8ee9) -- the MUTATION control that
                             proves this harness can detect the effect at all.

PREDICTION: arm A total-delta and grad-delta are EXACTLY 0.0 (bitwise);
arm B is non-zero. A harness where BOTH are zero is a vacuous test.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import zarr

from chess_anti_engine.train.losses import compute_loss

BASE = (
    "/home/josh/projects/chess/runs/pbt2_small/replay/"
    "train_trial_5ce02_00000_0_lr=0.0000_2026-08-11_04-19-24/replay_shards"
)
B = 64


def load_rows() -> dict[str, np.ndarray]:
    nm = sorted(os.listdir(BASE))[-1]
    g = zarr.open(os.path.join(BASE, nm), mode="r")
    sel = np.nonzero(
        np.asarray(g["has_sf_p0_regret"][:]).astype(bool)
        & np.asarray(g["has_legal_mask"][:]).astype(bool),
    )[0][:B]
    return {
        "reg": np.asarray(g["sf_p0_regret"][:])[sel].astype(np.float32),
        "legal": np.asarray(g["legal_mask"][:])[sel].astype(np.float32),
        "pol": np.asarray(g["policy_target"][:])[sel].astype(np.float32),
        "wdl": np.asarray(g["wdl_target"][:])[sel].astype(np.int64),
        "x": np.asarray(g["x"][:])[sel].astype(np.float32),
    }


def run(w: float, reg: torch.Tensor, d: dict[str, torch.Tensor]) -> tuple[float, torch.Tensor]:
    logits = d["logits"].clone().requires_grad_(True)
    batch = {
        "policy_t": d["pol"], "wdl_t": d["wdl"], "x": d["x"],
        "legal_mask": d["legal"], "has_legal_mask": torch.ones(B),
        "has_policy": torch.ones(B),
        "sf_p0_regret_t": reg, "has_sf_p0_regret": torch.ones(B),
        "is_network_turn": torch.ones(B), "has_is_network_turn": torch.ones(B),
    }
    out = {"policy": logits, "wdl": d["wdl_logits"]}
    losses = compute_loss(out, batch, w_sf_own_regret=w)
    total = losses["total"]
    total.backward()
    assert logits.grad is not None
    return float(total.item()), logits.grad.detach().clone()


def main() -> None:
    torch.manual_seed(0)
    r = load_rows()
    n_actions = r["reg"].shape[1]
    d = {
        "pol": torch.from_numpy(r["pol"]),
        "wdl": torch.from_numpy(r["wdl"]),
        "x": torch.from_numpy(r["x"]),
        "legal": torch.from_numpy(r["legal"]),
        "logits": torch.randn(B, n_actions),
        "wdl_logits": torch.randn(B, 3),
    }
    reg0 = torch.from_numpy(r["reg"])
    # Perturbation: set every FABRICATED entry to 0.0 -- i.e. what the label
    # would be if the uncovered tail were simply not asserted.
    reg1 = reg0.clone()
    legal = d["legal"].bool()
    for i in range(B):
        v = reg1[i][legal[i]]
        if v.numel() == 0:
            continue
        dmax = v.max()
        row = reg1[i]
        row[legal[i] & (reg1[i] >= dmax - 1e-6)] = 0.0
    print(f"perturbation changed {(reg0 != reg1).sum().item()} entries "
          f"of {reg0.numel()}")
    for w in (0.0, 0.7):
        t0, g0 = run(w, reg0, d)
        t1, g1 = run(w, reg1, d)
        dt = abs(t1 - t0)
        dg = float((g1 - g0).abs().max().item())
        rel = float((g1 - g0).norm() / g0.norm())
        print(f"w_sf_own_regret={w}:  |dtotal|={dt:.10g}  "
              f"max|dgrad|={dg:.10g}  ||dgrad||/||grad||={rel:.6g}")


if __name__ == "__main__":
    main()
