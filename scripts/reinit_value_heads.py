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
Only legacy fixed-slot pools are supported. Generation-selected pools must
not be edited in place, even when an older seeds/slot_000 still exists.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch import nn

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model, load_state_dict_tolerant
from chess_anti_engine.uci.model_loader import (
    _find_params_json,
    model_config_from_arch,
    _model_config_from_params,
)

VALUE_HEADS = ("value_wdl", "value_sf_eval", "value_categorical")

# The categorical head has two mutually exclusive shapes: the standalone
# `ValueHead` and the coupled 32-way Linear on `value_wdl`'s hidden. Exactly one
# is constructed and the other attribute is None, so a fixed name list hard-exits
# ("model missing head") on whichever topology it was not written for.
_ALTERNATIVE_HEAD_NAMES = {"value_categorical": ("value_categorical_coupled",)}


def _resolve_head(model: nn.Module, head_name: str) -> tuple[str, nn.Module]:
    """The module for `head_name`, accepting its known alternative spellings.

    Still exits when NONE of the candidates is a live module: silently skipping
    a head would leave it un-reinitialised while the script reported success.
    """
    for candidate in (head_name, *_ALTERNATIVE_HEAD_NAMES.get(head_name, ())):
        head = getattr(model, candidate, None)
        if isinstance(head, nn.Module):
            return candidate, head
    sys.exit(f"model missing head: {head_name}")


def _checkpoint_model_config(ckpt: dict, ckpt_path: Path) -> ModelConfig:
    """Load the checkpoint's actual architecture, failing loud if absent."""
    if isinstance(ckpt.get("arch"), dict):
        return model_config_from_arch(ckpt["arch"])

    params_path = _find_params_json(ckpt_path)
    if params_path is not None:
        with params_path.open() as fh:
            return _model_config_from_params(json.load(fh))

    raise SystemExit(
        f"{ckpt_path} has no embedded arch and no params.json in its ancestor "
        "tree; refusing to rewrite value heads with a guessed architecture"
    )


def _legacy_pool_checkpoint(pool_dir: Path) -> Path:
    """Refuse unsupported/ambiguous selection before loading or mutating weights."""
    manifest_path = pool_dir / "manifest.json"
    ckpt_path = pool_dir / "seeds" / "slot_000" / "trainer.pt"
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        if not manifest_path.is_symlink():
            return ckpt_path  # Pre-manifest legacy pools remain supported.
        raise SystemExit("unreadable pool manifest; refusing to rewrite value heads") from None
    except (OSError, UnicodeError):
        raise SystemExit("unreadable pool manifest; refusing to rewrite value heads") from None
    try:
        manifest = json.loads(text)
    except ValueError:
        raise SystemExit("invalid pool manifest; refusing to rewrite value heads") from None
    entries = manifest.get("entries") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise SystemExit("invalid pool manifest; refusing to rewrite value heads")
    slots: set[int] = set()
    for index, entry in enumerate(entries):
        slot = entry.get("slot", index) if isinstance(entry, dict) else None
        if type(slot) is not int or slot < 0 or slot in slots:
            raise SystemExit("ambiguous pool manifest; refusing to rewrite value heads")
        slots.add(slot)
        legacy_dir = pool_dir / "seeds" / f"slot_{slot:03d}"
        seed_dir = entry.get("seed_dir", str(Path("seeds") / f"slot_{slot:03d}"))
        if (not isinstance(seed_dir, str) or not seed_dir.strip()
                or (pool_dir / seed_dir).absolute() != legacy_dir.absolute()):
            raise SystemExit(
                "only legacy fixed-slot pools are supported; refusing to rewrite "
                "generation-selected or custom-path pool value heads in place"
            )
    if 0 not in slots:
        raise SystemExit("pool manifest has no slot zero; refusing to rewrite value heads")
    return ckpt_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool_dir", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ckpt_path = _legacy_pool_checkpoint(args.pool_dir)
    if not ckpt_path.exists():
        sys.exit(f"missing: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ck["model"]

    mcfg = _checkpoint_model_config(ck, ckpt_path)
    model = build_model(mcfg)
    load_state_dict_tolerant(model, state, label="reinit-value-heads")

    reinit_param_names: list[str] = []
    for head_name in VALUE_HEADS:
        resolved_name, head = _resolve_head(model, head_name)
        for pname, p in head.named_parameters():
            reinit_param_names.append(f"{resolved_name}.{pname}")
            with torch.no_grad():
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p, gain=0.1)
                else:
                    nn.init.zeros_(p)

    ck["model"] = model.state_dict()
    ck["arch"] = {
        "_schema_version": ARCH_SCHEMA_VERSION,
        **dataclasses.asdict(mcfg),
    }
    ck.pop("opt", None)  # force Adam to rebuild momentum; positional indices unsafe across arch changes
  # ⚑ The manifest MUST go with the state it describes. Popping `opt` alone left
  # `opt_param_names` behind, describing an optimizer state that is no longer in
  # the payload AND naming the OLD architecture while `ck["model"]` above was
  # just rewritten to the new one. That is a self-INCONSISTENT checkpoint: the
  # exact shape `_remap_optimizer_state_by_param_name` now raises
  # `UntrustedOptimizerStateError` on, because a manifest name that is not a key
  # of the model payload is proof the mapping is wrong. Harmless while `opt` is
  # absent -- nothing re-keys a state that is not there -- but this file is
  # precisely the tool that manufactures the hazard the guard exists to catch,
  # and a later edit that preserves `opt` would arm it. Found by independent
  # review of PR #427 (NIT 5). Fix the producer, not just the detector.
    ck.pop("opt_param_names", None)

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
