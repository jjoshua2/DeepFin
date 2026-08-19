"""Same-population NET entropy for the SF-soft arm, on MATCHED SUPPORT.

Prereg: ledger `2f4d1ee2a`. The criterion frozen there is

    PASS:  mean[ H(t_alpha) - H(p_net) ] <= +0.05 nats

with BOTH entropies on `supp(t0) UNION supp(q_SF)`, which is exactly
`supp(t_alpha)`. The net is restricted onto that support and renormalised; the
mass the restriction discards is reported as `tail_mass_net` rather than
silently dropped. Own-support entropies are NOT comparable here -- the target
lives on the Gumbel candidate set (~22 moves) and the net's softmax is nonzero
on every legal move (~30-40), a difference of order 0.2 nats, which is the size
of the effect under test.

⚑ RUN IN A PAUSE WINDOW, or with `--device cpu` (which contends for no GPU at
all). Never side-by-side with training on the GPU.

⚑ BANK THE CHECKPOINT FIRST. Ray prunes live checkpoints; copy the one this is
pointed at out of the tune dir before relying on the number.

Two guards are copied from `scripts/audit_targets.py` because this feeds the
shards' STORED planes to the model, exactly as `--input-encoding stored` does:
a checkpoint declaring a different layout would score bytes that mean something
else, and a dynamic-relation checkpoint fed without its relation tensor would
silently be scored relation-less. Both raise rather than warn.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/josh/projects/chess")

from chess_anti_engine.eval.audit_history import (
    STORED_EXTRA_FEATURES,
    STORED_HISTORY_ENCODING,
)
from scripts.net_source import add_net_source_args, net_source_from_args

SHARDS = (
    "runs/pbt2_small/replay/"
    "train_trial_dea5e_00000_0_lr=0.0000_2026-08-16_12-38-11/replay_shards"
)
ALPHAS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
BUDGET, SENSITIVITY = 0.05, 0.10
# The layout the shards' `x` is written under. IMPORTED, never re-spelled: a
# hand-copied literal here would either reject the correct checkpoint or, worse,
# accept a wrong one. (Guessing it as "stack8" did the former.)


def row_entropy(p: np.ndarray) -> np.ndarray:
    """Row-wise entropy in nats. Rows are assumed already normalised."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, p * np.log(p), 0.0)
    return -t.sum(axis=1)


def load_rows(n_shards: int) -> dict[str, np.ndarray]:
    import zarr

    names = sorted(os.listdir(SHARDS))[-n_shards:]
    keep: dict[str, list[np.ndarray]] = {"x": [], "legal": [], "t0": [], "q": []}
    for s in names:
        g = zarr.open(os.path.join(SHARDS, s), mode="r")
        has = np.asarray(g["has_sf_p0"][:]).astype(bool)
        if not has.any():
            continue
        keep["x"].append(np.asarray(g["x"][:])[has])
        keep["legal"].append(np.asarray(g["legal_mask"][:])[has])
        keep["t0"].append(np.asarray(g["policy_target"][:])[has].astype(np.float64))
        keep["q"].append(np.asarray(g["sf_p0_policy_target"][:])[has].astype(np.float64))
    return {k: np.concatenate(v) for k, v in keep.items()}


def net_policy(model, x: np.ndarray, legal: np.ndarray, device: str,
               batch: int) -> np.ndarray:
    """`policy_own` softmax over LEGAL moves, from the shards' own planes."""
    out = np.empty((len(x), legal.shape[1]), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + batch])).to(device).float()
            lb = torch.from_numpy(np.ascontiguousarray(legal[i:i + batch])).to(device).bool()
            head = model(xb)["policy_own"].float()
            head = head.masked_fill(~lb, float("-inf"))
            out[i:i + batch] = torch.softmax(head, dim=-1).double().cpu().numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    add_net_source_args(ap, checkpoint_help="BANKED checkpoint, not a live tune path")
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.15)
    ap.add_argument("--shards", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    net = net_source_from_args(args)
    model = net.load(device=args.device, gpu_mem_fraction=args.gpu_mem_fraction,
                     tag="sfsoft-entropy")
    model.eval()

    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    if (hist, extra) != (STORED_HISTORY_ENCODING, STORED_EXTRA_FEATURES):
        raise SystemExit(
            f"this probe feeds the shards' STORED planes and so requires a "
            f"checkpoint encoded as {STORED_HISTORY_ENCODING}/"
            f"{STORED_EXTRA_FEATURES}; this one declares {hist}/{extra}"
        )
    if bool(getattr(model, "use_dynamic_relations", False)):
        raise SystemExit(
            "dynamic-relation checkpoint: the relation tensor is not stored in "
            "the shards, so scoring these planes without it would silently "
            "measure a relation-less model"
        )

    d = load_rows(args.shards)
    t0, q = d["t0"], d["q"]
    print(f"rows {len(t0)}  policy width {t0.shape[1]}  net {net}")

    p_net = net_policy(model, d["x"], d["legal"], args.device, args.batch)

    # MATCHED SUPPORT: supp(t0) U supp(q) == supp(t_alpha) for alpha in (0,1).
    support = (t0 > 0) | (q > 0)
    kept = (p_net * support).sum(axis=1)
    pn = np.where(support, p_net, 0.0) / np.clip(kept, 1e-12, None)[:, None]
    h_net = row_entropy(pn)

    print(f"\nsupport: t0 {(t0 > 0).sum(1).mean():.1f}  q {(q > 0).sum(1).mean():.1f}  "
          f"union {support.sum(1).mean():.1f} moves")
    print(f"tail_mass_net (net mass OUTSIDE the union, discarded by the restriction): "
          f"mean {1 - kept.mean():.4f}  p90 {np.percentile(1 - kept, 90):.4f}")
    print(f"\nH(p_net) on matched support: mean {h_net.mean():.4f}")
    print("(the ledger's 0.7819 is the AUDIT population on its OWN support -- not this)")

    print(f"\n{'alpha':>6} {'H(t_a)':>8} {'dH vs net':>10} {'p50':>8} {'p90':>8} "
          f"{'p99':>8}  verdict @ +{BUDGET}")
    ok = []
    for a in ALPHAS:
        ha = row_entropy((1.0 - a) * t0 + a * q)
        dd = ha - h_net
        m = dd.mean()
        if m <= BUDGET:
            ok.append(a)
        flag = "PASS" if m <= BUDGET else ("(sens)" if m <= SENSITIVITY else "fail")
        print(f"{a:6.2f} {ha.mean():8.4f} {m:+10.4f} {np.percentile(dd, 50):8.4f} "
              f"{np.percentile(dd, 90):8.4f} {np.percentile(dd, 99):8.4f}  {flag}")

    a_max = max(ok) if ok else None
    print(f"\nalpha_max at the frozen +{BUDGET} budget: "
          f"{a_max if a_max is not None else 'NONE — arm fails'}")
    if a_max:
        for f, lab in ((0.362, "today"), (0.63, "fresh-shard asymptote")):
            cp = f * a_max * 20.0
            print(f"  cp @ f={f} ({lab}), S_R=20.0 borrowed from the AUDIT population: "
                  f"{cp:.2f}  {'>= 3.0' if cp >= 3.0 else '< 3.0 — FAILS the bar'}")


if __name__ == "__main__":
    main()
