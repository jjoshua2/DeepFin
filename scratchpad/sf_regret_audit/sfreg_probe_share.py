"""MEASURED: how much of the LIVE era-probe ruler is fabricated label?

Reproduces `eval/era_probe.py::score_probe`'s policy_eregret arithmetic exactly
(legal-masked softmax over policy_own, sum_m p(m)*sf_p0_regret(m)) on the real
frozen probe sets, with the live net, on CPU -- then splits the resulting scalar
into the COVERED (real SF evaluations) and UNCOVERED (fabricated default)
halves.

PREDICTION before the run: the fabricated share lands between the two proxies
already measured on live shards -- 0.55 (MCTS policy_target) and 0.74
(policy_soft_target) -- because the net's prior is trained toward a retempered
mixture of both.  Reproduced policy_eregret should also land within ~0.005 of
the live progress.csv columns (era 0.0696, inwindow 0.0672 at iter 218).
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

CKPT = (
    "/home/josh/projects/chess/runs/pbt2_small/tune/"
    "train_trial_5ce02_00000_0_lr=0.0000_2026-08-11_04-19-24/checkpoint_000218"
)
SETS = {
    "era": "/home/josh/projects/chess/data/era_probe/era_20260804.npz",
    "inwindow": "/home/josh/projects/chess/data/era_probe/inwindow_20260804.npz",
}


def main(n_rows: int = 512, bs: int = 64) -> None:
    torch.set_num_threads(8)
    model = load_model_from_checkpoint(CKPT, device="cpu")
    model.eval()
    for label, path in SETS.items():
        z = np.load(path)
        keys = list(z.keys())
        sel = (np.asarray(z["has_sf_p0_regret"]).astype(bool)
               & np.asarray(z["has_legal_mask"]).astype(bool))
        idx = np.nonzero(sel)[0][:n_rows]
        x = np.asarray(z["x"])[idx].astype(np.float32)
        reg = np.asarray(z["sf_p0_regret"])[idx].astype(np.float32)
        lm = np.asarray(z["legal_mask"])[idx].astype(np.float32)
        tot = 0.0
        fab = 0.0
        cov = 0.0
        p_unc = 0.0
        n = 0
        with torch.inference_mode():
            for s in range(0, len(idx), bs):
                xb = torch.from_numpy(x[s:s + bs])
                out = model(xb)
                logits = out["policy"] if "policy" in out else out["policy_own"]
                w = int(logits.shape[-1])
                m = torch.from_numpy(lm[s:s + bs])[:, :w]
                r = torch.from_numpy(reg[s:s + bs])[:, :w]
                masked = logits.float().masked_fill(m <= 0, float("-inf"))
                p = torch.softmax(masked, dim=-1)
                mb = m.bool()
                dmax = torch.where(mb, r, torch.full_like(r, -1.0)).max(dim=-1, keepdim=True).values
                is_fab = mb & (r >= dmax - 1e-6)
                per = (p * r).sum(-1)
                tot += float(per.sum())
                fab += float((p * r * is_fab).sum())
                cov += float((p * r * (mb & ~is_fab)).sum())
                p_unc += float((p * is_fab).sum())
                n += xb.shape[0]
        print(f"[{label}] n={n}  keys={len(keys)}")
        print(f"  policy_eregret (reproduced) = {tot / n:.6f}")
        print(f"    fabricated component      = {fab / n:.6f}  "
              f"({fab / tot:.4f} of the metric)")
        print(f"    real-SF component         = {cov / n:.6f}  "
              f"({cov / tot:.4f})")
        print(f"    net prob mass on uncovered moves = {p_unc / n:.4f}")


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
