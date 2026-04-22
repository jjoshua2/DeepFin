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
    ARCH_SCHEMA_VERSION,
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
    """Walk up from ``trainer.pt`` looking for a sibling ``params.json``.

    Only searches within the checkpoint's own ancestor chain — a params.json
    from a different training run would describe a different architecture,
    and ``load_state_dict_tolerant`` would then silently accept shape
    mismatches. New checkpoints should carry their architecture embedded
    under the ``arch`` key (see ``load_model_from_checkpoint``); this walk
    exists for the Ray trial layout
    ``<trial_dir>/{params.json, checkpoint_NNNN/trainer.pt}`` and a few
    nested salvage variants.
    """
    current = trainer_pt.parent
    for _ in range(6):
        candidate = current / "params.json"
        if candidate.is_file():
            return candidate
        if current.parent == current:  # filesystem root
            break
        current = current.parent
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


def _model_config_from_arch(arch: dict) -> ModelConfig:
    """Strict construction from the embedded ``arch`` payload.

    Unlike ``_model_config_from_params`` (which filters unknown keys
    because it's fed a noisy superset from Ray Tune), this path refuses
    to default-past anything it doesn't understand. A newer checkpoint
    with a new ModelConfig field would otherwise silently build a
    defaulted model and then hide the mismatch behind the tolerant
    state-dict loader — exactly the silent-corruption failure mode the
    schema version is meant to catch.
    """
    if not isinstance(arch.get("_schema_version"), int):
        raise ValueError("embedded arch is missing an integer _schema_version")
    version = int(arch["_schema_version"])
    if version > ARCH_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint arch schema v{version} is newer than this loader "
            f"(v{ARCH_SCHEMA_VERSION}); upgrade chess_anti_engine"
        )
    valid = {f.name for f in fields(ModelConfig)}
    payload = {k: v for k, v in arch.items() if k != "_schema_version"}
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(
            f"embedded arch has unknown keys {sorted(unknown)}; this loader "
            "does not recognise them and would silently default them away. "
            "Upgrade the package or re-save the checkpoint."
        )
    return ModelConfig(**payload)  # type: ignore[arg-type]


def load_model_from_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    model_config: ModelConfig | None = None,
) -> torch.nn.Module:
    """Load a trained ChessNet in eval mode.

    ``path`` may be a trainer.pt file or a checkpoint directory. Resolution
    order for the architecture (when ``model_config`` is not explicitly
    passed):
      1. ``ckpt["arch"]`` embedded by ``Trainer.save()``. Self-describing
         checkpoint; strict schema-version check.
      2. A ``params.json`` sibling / walked-up-from the checkpoint directory
         (Ray trial layout). Not scanned across unrelated training runs —
         that silently misloads on architecture drift.

    Raises if neither is available. ``load_state_dict_tolerant`` will
    otherwise accept shape-mismatched tensors silently, so we fail loud
    here rather than start a partly-random model.
    """
    trainer_pt = _resolve_trainer_pt(Path(path))
    # weights_only=True blocks arbitrary pickle execution — our trainer only
    # writes tensors + primitives (including the `arch` dict), so this stays
    # safe with no loss.
    ckpt = torch.load(trainer_pt, map_location=device, weights_only=True)

    if model_config is None:
        if isinstance(ckpt, dict) and isinstance(ckpt.get("arch"), dict):
            model_config = _model_config_from_arch(ckpt["arch"])
        else:
            params_path = _find_params_json(trainer_pt)
            if params_path is None:
                raise FileNotFoundError(
                    f"{trainer_pt} has no embedded arch and no params.json "
                    "in its own directory tree. Re-save with the current "
                    "Trainer to embed the arch key, or pass model_config "
                    "explicitly."
                )
            with params_path.open() as fh:
                params = json.load(fh)
            model_config = _model_config_from_params(params)

    model = build_model(model_config)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    load_state_dict_tolerant(model, state, label="uci-load")
    model.to(device).eval()
    return model
