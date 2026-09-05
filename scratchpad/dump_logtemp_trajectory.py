"""Append `policy_own.log_temp` from every SURVIVING checkpoint, deduped by iter.

⚑ Ray keeps only ~6 checkpoints (~29 min at ~290s/iter), so a scalar trajectory
is unrecoverable after the fact -- the F-only interim was already reduced to two
pre-flip anchors by pruning. This is the cheap half of the fix: the scalars cost
nothing, while a full salvage export is 637M and is reserved for the midpoint
and readout banks the audit actually needs.

Append-only and idempotent: re-running mid-window adds only new iterations.
"""
from __future__ import annotations

import glob
import json
import os
import re

import torch

D = ("runs/pbt2_small/tune/"
     "train_trial_dea5e_00000_0_lr=0.0000_2026-08-16_12-38-11")
OUT = "scratchpad/f_only_logtemp_trajectory.jsonl"

seen = set()
if os.path.exists(OUT):
    for line in open(OUT):
        if line.strip():
            seen.add(int(json.loads(line)["iter"]))

added = []
for path in sorted(glob.glob(os.path.join(D, "checkpoint_*"))):
    m = re.search(r"checkpoint_0*(\d+)$", path)
    if not m:
        continue
    it = int(m.group(1))
    if it in seen:
        continue
    f = os.path.join(path, "trainer.pt")
    if not os.path.exists(f):
        continue
    sd = torch.load(f, map_location="cpu", weights_only=False)
    for k in ("model", "model_state", "state_dict"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
            break
    val = step = None
    if isinstance(sd, dict):
        for k, v in sd.items():
            if "policy_own" in k and k.endswith("log_temp"):
                val = float(v)
                break
    if val is None:
        continue
    added.append({"iter": it, "policy_own_log_temp": val, "arm": "F_only" if it >= 301 else "A+F"})

with open(OUT, "a") as fh:
    for rec in sorted(added, key=lambda r: r["iter"]):
        fh.write(json.dumps(rec) + "\n")
print(json.dumps({"added": [r["iter"] for r in added], "already_had": sorted(seen)}, indent=2))
