from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


def _write_tiny_checkpoint(root: Path, *, params: dict | None, include_arch: bool) -> Path:
    ckpt_dir = root / "checkpoint_000001"
    ckpt_dir.mkdir()
    cfg = ModelConfig(kind="tiny")
    model = build_model(cfg)
    payload: dict = {"model": model.state_dict(), "step": 0}
    if include_arch:
        payload["arch"] = {
            "_schema_version": ARCH_SCHEMA_VERSION,
            **dataclasses.asdict(cfg),
        }
    torch.save(payload, ckpt_dir / "trainer.pt")
    if params is not None:
        (root / "params.json").write_text(json.dumps(params), encoding="utf-8")
    return ckpt_dir


def _write_tiny_checkpoint_with_cfg(root: Path, cfg: ModelConfig, *, params: dict | None = None) -> Path:
    ckpt_dir = root / "checkpoint_000001"
    ckpt_dir.mkdir()
    model = build_model(cfg)
    torch.save(
        {
            "model": model.state_dict(),
            "step": 0,
            "arch": {
                "_schema_version": ARCH_SCHEMA_VERSION,
                **dataclasses.asdict(cfg),
            },
        },
        ckpt_dir / "trainer.pt",
    )
    if params is not None:
        (root / "params.json").write_text(json.dumps(params), encoding="utf-8")
    return ckpt_dir


def test_uci_loader_defaults_legacy_history_for_arch_checkpoint(tmp_path: Path) -> None:
    ckpt = _write_tiny_checkpoint(tmp_path, params=None, include_arch=True)

    model = load_model_from_checkpoint(ckpt, device="cpu")

    assert getattr(model, "input_history_encoding") == "legacy"


def test_uci_loader_reads_history_encoding_from_params_json(tmp_path: Path) -> None:
    ckpt = _write_tiny_checkpoint(
        tmp_path,
        params={"model": "tiny", "input_history_encoding": "lc0_root_legacy_meta"},
        include_arch=True,
    )

    model = load_model_from_checkpoint(ckpt, device="cpu")

    assert getattr(model, "input_history_encoding") == "lc0_root_legacy_meta"


def test_uci_loader_reads_history_encoding_from_embedded_arch(tmp_path: Path) -> None:
    ckpt = _write_tiny_checkpoint_with_cfg(
        tmp_path,
        ModelConfig(kind="tiny", input_history_encoding="lc0_root"),
    )

    model = load_model_from_checkpoint(ckpt, device="cpu")

    assert getattr(model, "input_history_encoding") == "lc0_root"
