from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
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


def test_uci_loader_preserves_per_layer_embed_dims_from_embedded_arch(tmp_path: Path) -> None:
    ckpt = _write_tiny_checkpoint_with_cfg(
        tmp_path,
        ModelConfig(
            embed_dim=32,
            num_layers=3,
            num_heads=4,
            embed_dim_by_layer=(32, 48, 40),
            use_smolgen=True,
            smolgen_mode="per_layer",
            smolgen_pooling="mean",
            smolgen_hidden_channels=8,
            smolgen_hidden_sz=16,
            smolgen_gen_sz=24,
        ),
    )

    model = load_model_from_checkpoint(ckpt, device="cpu")

    assert getattr(model, "embed_dim_by_layer") == (32, 48, 40)


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


# ---------------------------------------------------------------------------
# rl_loop_audit L12: the arena loads both sides COMPLETELY.
#
# Both sides of scripts/arena_standard.py come through this loader. When the
# model was rebuilt from the checkpoint's OWN `arch` payload there is no
# legitimate reason for a key to drop -- and up to 50% of them could vanish
# behind a single stdout line before the catastrophic-load guard fired, which
# is exactly how a lopsided Elo gets believed. Method rule 7 ("verify identity
# before believing a lopsided arena") was a human habit; these pin it as code.
# ---------------------------------------------------------------------------

def _drop_a_key(ckpt_dir: Path) -> str:
    """Remove one tensor from a saved checkpoint; return the key removed."""
    trainer_pt = ckpt_dir / "trainer.pt"
    payload = torch.load(trainer_pt, map_location="cpu", weights_only=True)
    dropped = sorted(payload["model"])[0]
    del payload["model"][dropped]
    torch.save(payload, trainer_pt)
    return dropped


def test_uci_loader_requires_complete_load_for_embedded_arch(tmp_path: Path) -> None:
    """An arch-bearing checkpoint must load exactly or raise."""
    ckpt = _write_tiny_checkpoint(tmp_path, params=None, include_arch=True)
    dropped = _drop_a_key(ckpt)

    with pytest.raises(RuntimeError) as exc:
        load_model_from_checkpoint(ckpt, device="cpu")

    message = str(exc.value)
    assert "require_complete=True" in message
    assert dropped in message


def test_uci_loader_complete_embedded_arch_load_still_succeeds(tmp_path: Path) -> None:
    """The new default strictness must not disturb an ordinary, exact load.

    This is the case every live tool is actually in, so a regression here would
    take the arena, value_regret and audit_targets down together.
    """
    ckpt = _write_tiny_checkpoint(tmp_path, params=None, include_arch=True)
    saved = torch.load(ckpt / "trainer.pt", map_location="cpu", weights_only=True)["model"]

    model = load_model_from_checkpoint(ckpt, device="cpu")

    loaded = model.state_dict()
    assert set(loaded) == set(saved)
    for key, value in loaded.items():
        assert torch.equal(value, saved[key])


def test_uci_loader_require_complete_false_restores_tolerance(tmp_path: Path) -> None:
    """The escape hatch for deliberately loading into a different architecture."""
    ckpt = _write_tiny_checkpoint(tmp_path, params=None, include_arch=True)
    _drop_a_key(ckpt)

    assert load_model_from_checkpoint(ckpt, device="cpu", require_complete=False) is not None


def test_uci_loader_params_json_path_stays_tolerant(tmp_path: Path) -> None:
    """No embedded arch means the arch is a guess, so drops stay survivable.

    ``params.json`` describes the trial, not the tensor file; demanding an
    exact load off it would break legitimate reads of pre-``arch`` checkpoints.
    """
    ckpt = _write_tiny_checkpoint(tmp_path, params={"model": "tiny"}, include_arch=False)
    _drop_a_key(ckpt)

    assert load_model_from_checkpoint(ckpt, device="cpu") is not None


def test_uci_loader_explicit_model_config_stays_tolerant(tmp_path: Path) -> None:
    """An explicitly passed config is the caller's claim, not the file's."""
    ckpt = _write_tiny_checkpoint(tmp_path, params=None, include_arch=True)
    _drop_a_key(ckpt)

    model = load_model_from_checkpoint(
        ckpt, device="cpu", model_config=ModelConfig(kind="tiny"),
    )

    assert model is not None


def test_uci_loader_require_complete_true_overrides_the_params_json_path(tmp_path: Path) -> None:
    """Strictness is still available where it is not the default."""
    ckpt = _write_tiny_checkpoint(tmp_path, params={"model": "tiny"}, include_arch=False)
    _drop_a_key(ckpt)

    with pytest.raises(RuntimeError, match="require_complete=True"):
        load_model_from_checkpoint(ckpt, device="cpu", require_complete=True)
