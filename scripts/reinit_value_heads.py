"""Surgically reinitialize value heads in a salvage trainer.pt.

Xavier-uniform(gain=0.1) for 2-D params, zero for 1-D params, in:
  - value_wdl
  - value_sf_eval
  - value_categorical

Drops the optimizer state entirely so AdamW rebuilds fresh momentum on
resume — the alternative (remapping per-param positional indices across
an architecture change) is more complexity than a recovery script
warrants. A few iterations of cold momentum is a fine cost.

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

from chess_anti_engine.model import ModelConfig, build_model, load_state_dict_tolerant

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
    load_state_dict_tolerant(model, state, label="reinit-value-heads")

    reinit_param_names: list[str] = []
    for head_name in VALUE_HEADS:
        head = getattr(model, head_name, None)
        if not isinstance(head, nn.Module):
            sys.exit(f"model missing head: {head_name}")
        for pname, p in head.named_parameters():
            reinit_param_names.append(f"{head_name}.{pname}")
            with torch.no_grad():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.1)
                else:
                    nn.init.zeros_(p)

    ck["model"] = model.state_dict()
    ck.pop("opt", None)  # force Adam to rebuild momentum; positional indices unsafe across arch changes

    print(f"reinit: {len(reinit_param_names)} params across {len(VALUE_HEADS)} heads")
    for n in reinit_param_names:
        print(f"  {n}")
    print("dropped optimizer state (trainer will rebuild on resume)")

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
