from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.script_loading import load_script_module


def _load_reinit_module():
    return load_script_module("reinit_value_heads.py", "reinit_value_heads_test_module")


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


def _write_coupled_pool(tmp_path: Path, *, coupled: bool) -> Path:
    """A salvage pool laid out the way the script expects, at either topology."""
    import dataclasses

    import torch

    from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model

    cfg = ModelConfig(
        kind="transformer", embed_dim=64, num_layers=2, num_heads=4,
        use_smolgen=False, categorical_head_coupled=coupled,
    )
    slot = tmp_path / "seeds" / "slot_000"
    slot.mkdir(parents=True)
    torch.save(
        {
            "model": build_model(cfg).state_dict(),
            "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **dataclasses.asdict(cfg)},
            "opt": {"fake": True},
        },
        slot / "trainer.pt",
    )
    return tmp_path


@pytest.mark.parametrize(
    ("coupled", "expected"),
    [(False, "value_categorical."), (True, "value_categorical_coupled.")],
)
def test_main_reinitialises_the_categorical_head_at_either_topology(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], coupled: bool, expected: str
) -> None:
    """⚑ Drives `main()`, NOT the resolver helper.

    Exactly one categorical head is constructed and the other attribute is None,
    so a fixed name list hard-exits `model missing head: value_categorical` on a
    coupled checkpoint — the script cannot run at all on the topology it would
    most plausibly be pointed at. Asserting on the helper alone would pass even
    if `main()` stopped calling it, which is the wiring this test exists for.
    """
    import sys

    module = _load_reinit_module()
    pool = _write_coupled_pool(tmp_path, coupled=coupled)
    argv = sys.argv
    try:
        sys.argv = ["reinit_value_heads.py", str(pool), "--dry-run"]
        module.main()
    finally:
        sys.argv = argv

    out = capsys.readouterr().out
    assert any(line.strip().startswith(expected) for line in out.splitlines()), out
    other = "value_categorical." if coupled else "value_categorical_coupled."
    assert not any(line.strip().startswith(other) for line in out.splitlines()), out


def test_resolve_head_still_exits_when_no_candidate_exists() -> None:
    """The tolerance must not become a silent skip: a genuinely missing head
    would otherwise be reported as reinitialised without being touched."""
    from torch import nn

    module = _load_reinit_module()
    with pytest.raises(SystemExit, match="model missing head: value_wdl"):
        module._resolve_head(nn.Linear(2, 2), "value_wdl")
