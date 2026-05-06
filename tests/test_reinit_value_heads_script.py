from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_reinit_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "reinit_value_heads.py"
    spec = importlib.util.spec_from_file_location("reinit_value_heads_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_model_config_uses_embedded_arch(tmp_path: Path) -> None:
    module = _load_reinit_module()
    ckpt_path = tmp_path / "trainer.pt"

    cfg = module._checkpoint_model_config(
        {
            "arch": {
                "_schema_version": 1,
                "kind": "transformer",
                "embed_dim": 64,
                "num_layers": 2,
                "num_heads": 4,
                "ffn_mult": 1.25,
                "use_smolgen": False,
                "use_nla": True,
                "use_qk_rmsnorm": True,
                "use_gradient_checkpointing": False,
            }
        },
        ckpt_path,
    )

    assert cfg.embed_dim == 64
    assert cfg.num_layers == 2
    assert cfg.num_heads == 4
    assert cfg.ffn_mult == 1.25
    assert cfg.use_smolgen is False
    assert cfg.use_nla is True
    assert cfg.use_qk_rmsnorm is True


def test_checkpoint_model_config_uses_params_fallback(tmp_path: Path) -> None:
    module = _load_reinit_module()
    trial_dir = tmp_path / "train_trial_abcd"
    ckpt_dir = trial_dir / "checkpoint_000001"
    ckpt_dir.mkdir(parents=True)
    (trial_dir / "params.json").write_text(
        json.dumps(
            {
                "model": "transformer",
                "embed_dim": 96,
                "num_layers": 3,
                "num_heads": 6,
                "ffn_mult": 1.5,
                "no_smolgen": True,
            }
        ),
        encoding="utf-8",
    )

    cfg = module._checkpoint_model_config({}, ckpt_dir / "trainer.pt")

    assert cfg.embed_dim == 96
    assert cfg.num_layers == 3
    assert cfg.num_heads == 6
    assert cfg.ffn_mult == 1.5
    assert cfg.use_smolgen is False


def test_checkpoint_model_config_refuses_to_guess(tmp_path: Path) -> None:
    module = _load_reinit_module()

    with pytest.raises(SystemExit, match="refusing to rewrite value heads"):
        module._checkpoint_model_config({}, tmp_path / "trainer.pt")
