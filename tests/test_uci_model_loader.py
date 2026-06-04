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


def _write_tiny_checkpoint_with_legacy_arch(root: Path, *, params: dict | None = None) -> Path:
    ckpt_dir = root / "checkpoint_000001"
    ckpt_dir.mkdir()
    cfg = ModelConfig(kind="tiny")
    model = build_model(cfg)
    arch = {
        "_schema_version": ARCH_SCHEMA_VERSION,
        "kind": "tiny",
    }
    torch.save({"model": model.state_dict(), "step": 0, "arch": arch}, ckpt_dir / "trainer.pt")
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
        include_arch=False,
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


def test_uci_loader_preserves_per_layer_ffn_multipliers_from_embedded_arch(tmp_path: Path) -> None:
    ckpt = _write_tiny_checkpoint_with_cfg(
        tmp_path,
        ModelConfig(
            embed_dim=32,
            num_layers=3,
            num_heads=4,
            ffn_mult=1.0,
            ffn_mult_by_layer=(1.0, 1.25, 1.5),
            use_smolgen=False,
        ),
    )

    model = load_model_from_checkpoint(ckpt, device="cpu")

    assert getattr(model, "ffn_mult_by_layer") == (1.0, 1.25, 1.5)


def test_uci_loader_preserves_smolgen_pooling_from_embedded_arch(tmp_path: Path) -> None:
    ckpt = _write_tiny_checkpoint_with_cfg(
        tmp_path,
        ModelConfig(
            embed_dim=32,
            num_layers=1,
            num_heads=4,
            use_smolgen=True,
            smolgen_pooling="mean",
            smolgen_hidden_channels=8,
            smolgen_hidden_sz=16,
            smolgen_gen_sz=24,
        ),
    )

    model = load_model_from_checkpoint(ckpt, device="cpu")

    assert getattr(model, "smolgen_pooling") == "mean"
    assert getattr(model, "smolgen_hidden_channels") == 8
    assert getattr(model, "smolgen_hidden_sz") == 16
    assert getattr(model, "smolgen_gen_sz") == 24


def test_uci_loader_uses_params_history_when_embedded_arch_is_legacy_schema(tmp_path: Path) -> None:
    ckpt = _write_tiny_checkpoint_with_legacy_arch(
        tmp_path,
        params={"model": "tiny", "input_history_encoding": "lc0_root_legacy_meta"},
    )

    model = load_model_from_checkpoint(ckpt, device="cpu")

    assert getattr(model, "input_history_encoding") == "lc0_root_legacy_meta"


def test_uci_loader_embedded_arch_wins_over_stale_params_json(tmp_path: Path) -> None:
    ckpt = _write_tiny_checkpoint_with_cfg(
        tmp_path,
        ModelConfig(kind="tiny", input_history_encoding="lc0_root"),
        params={"model": "tiny", "input_history_encoding": "legacy"},
    )

    model = load_model_from_checkpoint(ckpt, device="cpu")

    assert getattr(model, "input_history_encoding") == "lc0_root"
