"""Surgically reinitialize value heads in a salvage trainer.pt.

Xavier-uniform(gain=0.1) for 2-D params, zero for 1-D params, in:
  - value_wdl
  - value_sf_eval
  - value_categorical

Also zeros the AdamW exp_avg / exp_avg_sq entries for the reinited params
so stale optimizer momentum doesn't kick them on step 1.

Usage: python scripts/reinit_value_heads.py POOL_DIR [--dry-run]
Writes POOL_DIR/seeds/slot_000/trainer.pt in place (keeps trainer.pt.bak).
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn

from chess_anti_engine.model import ModelConfig, build_model

VALUE_HEADS = ("value_wdl", "value_sf_eval", "value_categorical")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ckpt_path = args.pool_dir / "seeds" / "slot_000" / "trainer.pt"
    if not ckpt_path.exists():
        sys.exit(f"missing: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model"]
    opt_state = ck["opt"]

    mcfg = ModelConfig(
        kind="transformer",
        embed_dim=384,
        num_layers=9,
        num_heads=12,
        ffn_mult=1.5,
        use_smolgen=True,
        use_nla=False,
    )
    model = build_model(mcfg)
    model.load_state_dict(state, strict=True)

    # Collect parameter objects we'll reinit, and their positions in the full
    # parameter list (to match optimizer state_dict indexing).
    all_params = list(model.parameters())
    all_param_ids = {id(p): i for i, p in enumerate(all_params)}

    reinit_param_indices: list[int] = []
    reinit_param_names: list[str] = []
    for head_name in VALUE_HEADS:
        head = getattr(model, head_name, None)
        if not isinstance(head, nn.Module):
            sys.exit(f"model missing head: {head_name}")
        for pname, p in head.named_parameters():
            full_name = f"{head_name}.{pname}"
            reinit_param_names.append(full_name)
            reinit_param_indices.append(all_param_ids[id(p)])
            with torch.no_grad():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.1)
                else:
                    nn.init.zeros_(p)

    # Write updated model weights back into state dict
    new_state = model.state_dict()
    for k in new_state:
        state[k] = new_state[k]

    # Zero optimizer state entries for reinited param indices.
    # opt_state is a dict: {'state': {idx: {'step','exp_avg','exp_avg_sq',...}}, 'param_groups': [...]}
    opt_inner = opt_state.get("state", {})
    zeroed = 0
    for idx in reinit_param_indices:
        entry = opt_inner.get(idx)
        if entry is None:
            continue
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            if key in entry and torch.is_tensor(entry[key]):
                entry[key].zero_()
                zeroed += 1

    print(f"reinit: {len(reinit_param_names)} params across {len(VALUE_HEADS)} heads")
    for n in reinit_param_names:
        print(f"  {n}")
    print(f"zeroed {zeroed} optimizer state tensors for reinited params")

    if args.dry_run:
        print("dry-run; not writing")
        return

    bak = ckpt_path.with_suffix(".pt.bak")
    if not bak.exists():
        shutil.copy2(ckpt_path, bak)
        print(f"backed up original to {bak}")

    torch.save(ck, ckpt_path)
    print(f"wrote {ckpt_path}")


if __name__ == "__main__":
    main()
