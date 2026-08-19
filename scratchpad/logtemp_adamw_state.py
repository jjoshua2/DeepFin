"""Extract the ACTUAL AdamW state for `policy_own.log_temp` from every checkpoint.

⚑ Replaces the withdrawn SGD-style ODE. Two errors compounded there:
  1. the update is `-lr * m_hat/(sqrt(v_hat)+eps)`, NOT `lr * G` (peer, correct);
  2. `lr` for THIS parameter is the group-3 value, and the Ray table's `lr` column
     is a different group -- measured 3e-06 vs the 3e-05 I used, a 10x error.
`policy_own.log_temp` is param 449, group 3, `use_aurora=False` -> the AdamW
fallback at aurora.py:624. betas (0.9, 0.95), eps 1e-8, weight_decay 0.0.

Reports the realized moment ratio m_hat/sqrt(v_hat), which is what actually sets
the step size, and compares `-lr * ratio` against the realized displacement.
"""
from __future__ import annotations

import json
import math
import os
import sys

import torch

NAME = "policy_own.log_temp"
PATHS: list[tuple[int, str]] = []
for it, p in [
    (190, "data/salvage/sfsl_baseline_iter190_20260817/checkpoint_000190"),
    (231, "data/salvage/sfsl_gate_iter231_20260818"),
    (297, "data/salvage/apf_endpoint_checkpoint_000297_20260818/checkpoint_000297"),
]:
    PATHS.append((it, p))
MID = "data/salvage/f_only_midpoint_20260818"
if os.path.isdir(MID):
    for root, _dirs, files in os.walk(MID):
        if "trainer.pt" in files:
            m = os.path.basename(root)
            it = int(m.split("_")[-1]) if m.startswith("checkpoint_") else 338
            PATHS.append((it, root))
            break
D = ("runs/pbt2_small/tune/"
     "train_trial_dea5e_00000_0_lr=0.0000_2026-08-16_12-38-11")
if os.path.isdir(D):
    for d in sorted(os.listdir(D)):
        if d.startswith("checkpoint_"):
            PATHS.append((int(d.split("_")[-1]), os.path.join(D, d)))

rows = []
for it, p in sorted(set(PATHS)):
    f = p if p.endswith("trainer.pt") else os.path.join(p, "trainer.pt")
    if not os.path.exists(f):
        continue
    try:
        ck = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  iter {it}: unreadable ({type(exc).__name__})", file=sys.stderr)
        continue
    names = ck.get("opt_param_names") or []
    if NAME not in names:
        continue
    i = names.index(NAME)
    opt = ck["opt"]
    st = opt["state"].get(i) or opt["state"].get(str(i))
    grp = next((g for g in opt["param_groups"] if i in g["params"]), None)
    if st is None or grp is None:
        continue
    b1, b2 = tuple(grp["betas"])
    step = int(st["step"])
    m = float(st["exp_avg"])
    v = float(st["exp_avg_sq"])
    mh = m / (1.0 - b1**step)
    vh = v / (1.0 - b2**step)
    ratio = mh / (math.sqrt(vh) + float(grp["eps"]))
    rows.append({
        "iter": it, "step": step, "value": float(ck["model"][NAME].reshape(-1)[0]),
        "lr": float(grp["lr"]), "wd": float(grp["weight_decay"]),
        "exp_avg": m, "exp_avg_sq": v, "ratio": ratio,
        "dl_per_step": -float(grp["lr"]) * ratio,
    })
    del ck

with open("scratchpad/logtemp_adamw_state.json", "w") as fh:
    json.dump(rows, fh, indent=1)

print(f"{'iter':>5s} {'step':>8s} {'log_temp':>10s} {'lr':>9s} "
      f"{'exp_avg':>11s} {'sqrt(v)':>10s} {'m/sqrt(v)':>10s} {'-lr*ratio':>11s}")
for r in rows:
    print(f"{r['iter']:5d} {r['step']:8d} {r['value']:+10.5f} {r['lr']:9.2e} "
          f"{r['exp_avg']:+11.3e} {math.sqrt(r['exp_avg_sq']):10.3e} "
          f"{r['ratio']:+10.4f} {r['dl_per_step']:+11.3e}")

# realized displacement vs the instantaneous prediction, between consecutive rows
print("\n  segment      d(step)   realized dl/step    predicted (mid-ratio)   ratio")
for a, b in zip(rows, rows[1:]):
    ds = b["step"] - a["step"]
    if ds <= 0:
        continue
    real = (b["value"] - a["value"]) / ds
    pred = 0.5 * (a["dl_per_step"] + b["dl_per_step"])
    rr = real / pred if pred else float("nan")
    print(f"  {a['iter']:4d}->{b['iter']:<4d} {ds:9d}   {real:+.4e}        "
          f"{pred:+.4e}       {rr:6.2f}x")
