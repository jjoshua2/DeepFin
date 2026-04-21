"""Standalone checkpoint loader for the UCI engine.

Accepts either a ``trainer.pt`` file or a checkpoint directory containing
one. Reconstructs ``ModelConfig`` from the trial's ``params.json`` that
Ray Tune writes next to each trial's checkpoints, then calls
``build_model`` + ``load_state_dict_tolerant``.

No Trainer, no Ray — this path is deliberately minimal so it works in a
subprocess with nothing but the package importable.
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import torch

from chess_anti_engine.model import (
    ModelConfig,
    build_model,
    load_state_dict_tolerant,
)


def _resolve_trainer_pt(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidate = path / "trainer.pt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no trainer.pt at {path}")


def _find_params_json(trainer_pt: Path) -> Path | None:
    # Trial layout: <trial_dir>/{params.json, checkpoint_NNNN/trainer.pt}
    for parent in (trainer_pt.parent, trainer_pt.parent.parent):
        candidate = parent / "params.json"
        if candidate.is_file():
            return candidate
    return None


def _model_config_from_params(params: dict) -> ModelConfig:
    """Keep only the fields ``ModelConfig`` recognises.

    ``params.json`` is flat and contains many unrelated tune knobs. The
    overlap with ``ModelConfig`` is small and well-defined.
    """
    kind = str(params.get("model", "transformer"))
    valid = {f.name for f in fields(ModelConfig)}
    filtered = {k: v for k, v in params.items() if k in valid}
    filtered.setdefault("kind", kind)
    # 'use_smolgen' is stored as the negation 'no_smolgen' in some trials.
    if "no_smolgen" in params and "use_smolgen" not in filtered:
        filtered["use_smolgen"] = not bool(params["no_smolgen"])
    return ModelConfig(**filtered)  # type: ignore[arg-type]


def load_model_from_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    model_config: ModelConfig | None = None,
) -> torch.nn.Module:
    """Load a trained ChessNet in eval mode.

    ``path`` may be a trainer.pt file or a checkpoint directory. When
    ``model_config`` is omitted, it's inferred from the sibling/parent
    ``params.json``; raise if neither is available (we can't guess
    the architecture).
    """
    trainer_pt = _resolve_trainer_pt(Path(path))

    if model_config is None:
        params_path = _find_params_json(trainer_pt)
        if params_path is None:
            raise FileNotFoundError(
                f"no params.json near {trainer_pt}; pass model_config explicitly"
            )
        with params_path.open() as fh:
            params = json.load(fh)
        model_config = _model_config_from_params(params)

    model = build_model(model_config)
    # weights_only=True blocks arbitrary pickle execution — our trainer only
    # writes tensors + primitives, so this is strictly safer with no loss.
    ckpt = torch.load(trainer_pt, map_location=device, weights_only=True)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_state_dict_tolerant(model, state, label="uci-load")
    model.to(device).eval()
    return model
