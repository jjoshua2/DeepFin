"""Score both FROZEN era-probe sets with the live ruler AND covered-only masks.

CPU only. Reproduces ``eval/era_probe.py::score_probe_set``'s ``policy_eregret``
arithmetic exactly (gate: bit-exact against live progress.csv iter 218), then
recomputes it renormalised over the moves Stockfish ACTUALLY scored.

MEASURED FACT that shapes the masks (see probe_setstruct.py): both frozen sets
were cut 2026-08-04 from the 13a9f lineage, i.e. entirely inside the
``sf_multipv: 40`` era (40 from 2026-04-29 `02c64f700` until 2026-08-06
`ed9de8ee9`).  So a row can carry a fabricated ``(worst_covered+1)/2`` block
ONLY if it has more than 40 legal moves.  Verified: on rows with L > 40 the
number of legal moves tied at the row max equals **exactly L - 40** on 284/308
(era) and 265/295 (inwindow); the remainder tie higher because the default
collided with the 1000cp cap.  No row has fewer.

Three arms:
  A ``unmasked``  — the live ruler, verbatim.
  B ``mech``      — drop the tied-at-max block on rows with L > 40 only
                    (the mechanism-derived fabricated set), renormalise.
  C ``aggr``      — drop the tied-at-max block on EVERY row, renormalise.
                    This is the classifier ledger `c4f02bd5c` used; on
                    MultiPV-40 data it also discards REAL cap-saturated
                    evaluations, so it is an upper bound on the correction.

Usage:  python3 probe_masked.py <threads> <outdir> <ckpt> [<ckpt> ...]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

SETS = {
    "era": "/home/josh/projects/chess/data/era_probe/era_20260804.npz",
    "inwindow": "/home/josh/projects/chess/data/era_probe/inwindow_20260804.npz",
}
SF_MULTIPV_AT_CUT = 40


def _load_set(path: str) -> dict[str, np.ndarray]:
    z = np.load(path)
    return {
        "x": np.asarray(z["x"]),
        "reg": np.asarray(z["sf_p0_regret"]).astype(np.float32),
        "lm": np.asarray(z["legal_mask"]).astype(np.float32),
        "has_reg": np.asarray(z["has_sf_p0_regret"]).astype(np.float32),
        "has_lm": np.asarray(z["has_legal_mask"]).astype(np.float32),
    }


def score(model: torch.nn.Module, arrs: dict[str, np.ndarray], bs: int) -> dict[str, np.ndarray]:
    n = int(arrs["x"].shape[0])
    per = {k: np.zeros(n, dtype=np.float64) for k in
           ("unmasked", "mech", "aggr", "p_at_max", "p_at_max_big")}
    keep = np.zeros(n, dtype=bool)
    with torch.inference_mode():
        for s0 in range(0, n, bs):
            s1 = min(n, s0 + bs)
            xb = torch.from_numpy(arrs["x"][s0:s1].astype(np.float32))
            out = model(xb)
            logits = out["policy"] if "policy" in out else out["policy_own"]
            w = int(logits.shape[-1])
            lm = torch.from_numpy(arrs["lm"][s0:s1])[:, :w]
            r = torch.from_numpy(arrs["reg"][s0:s1])[:, :w]
            has_reg = torch.from_numpy(arrs["has_reg"][s0:s1])
            has_lm = torch.from_numpy(arrs["has_lm"][s0:s1])
            legal = (lm > 0) & (has_lm > 0).unsqueeze(-1)
            p = torch.softmax(logits.float().masked_fill(~legal, float("-inf")), dim=-1)

            L = legal.sum(-1, keepdim=True)
            rl = torch.where(legal, r, torch.full_like(r, -1.0))
            dmax = rl.max(dim=-1, keepdim=True).values
            at_max = legal & (r >= dmax - 1e-9)
            big = L > SF_MULTIPV_AT_CUT
            fab_mech = at_max & big
            fab_aggr = at_max

            e_un = (p * r * legal).sum(-1)
            for name, fab in (("mech", fab_mech), ("aggr", fab_aggr)):
                cov = legal & ~fab
                pc = (p * cov).sum(-1)
                e = torch.where(pc > 1e-9, (p * r * cov).sum(-1) / pc.clamp_min(1e-12), e_un)
                per[name][s0:s1] = e.double().numpy()
            per["unmasked"][s0:s1] = e_un.double().numpy()
            per["p_at_max"][s0:s1] = (p * fab_aggr).sum(-1).double().numpy()
            per["p_at_max_big"][s0:s1] = (p * fab_mech).sum(-1).double().numpy()
            keep[s0:s1] = (has_reg > 0).numpy()
    per["keep"] = keep
    return per


def main() -> None:
    torch.set_num_threads(int(sys.argv[1]))
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    paths = sys.argv[3:]
    sets = {k: _load_set(v) for k, v in SETS.items()}
    for path in paths:
        t0 = time.perf_counter()
        pt = path if path.endswith(".pt") else os.path.join(path, "trainer.pt")
        raw = torch.load(pt, map_location="cpu", weights_only=True)
        step = int(raw.get("step", -1))
        del raw
        model = load_model_from_checkpoint(path, device="cpu")
        model.eval()
        rec: dict[str, object] = {"path": path, "step": step}
        dump: dict[str, np.ndarray] = {}
        means: dict[str, dict[str, float]] = {}
        for label, arrs in sets.items():
            per = score(model, arrs, 128)
            k = per["keep"]
            means[label] = {a: float(per[a][k].mean())
                            for a in ("unmasked", "mech", "aggr", "p_at_max", "p_at_max_big")}
            means[label]["n"] = float(k.sum())
            for a in ("unmasked", "mech", "aggr"):
                dump[f"{label}_{a}"] = per[a]
            dump[f"{label}_keep"] = k
        rec["means"] = means
        for a in ("unmasked", "mech", "aggr"):
            rec[f"gap_{a}"] = means["era"][a] - means["inwindow"][a]
        rec["seconds"] = time.perf_counter() - t0
        tag = os.path.basename(path.rstrip("/")).replace(".pt", "")
        np.savez_compressed(os.path.join(outdir, f"probe_rows_{step:06d}_{tag}.npz"), **dump)
        print(json.dumps(rec), flush=True)
        del model


if __name__ == "__main__":
    main()
