from __future__ import annotations

from pathlib import Path

import torch

from chess_anti_engine.train.trainer import Trainer


class _TinyMuonModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.head = torch.nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {
            "policy": self.head.weight[:1],
            "wdl": torch.zeros((1, 3), dtype=torch.float32, device=self.head.weight.device),
        }


def _make_trainer(tmp_path: Path) -> Trainer:
    return Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="muon",
        warmup_steps=10,
        warmup_lr_start=1e-5,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )


def test_muon_warmup_preserves_group_lr_ratio_from_step_zero(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    trunk_lr = float(trainer.opt.param_groups[0]["lr"])
    aux_lr = float(trainer.opt.param_groups[2]["lr"])
    assert trunk_lr == aux_lr * 20.0


def test_muon_warmup_handoff_reaches_group_base_lr_without_ratio_jump(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.step = trainer._warmup_steps - 1
    trainer._update_lr()

    base_lrs = trainer._base_lrs()
    assert float(trainer.opt.param_groups[0]["lr"]) == float(base_lrs[0])
    assert float(trainer.opt.param_groups[2]["lr"]) == float(base_lrs[2])
    assert float(trainer.opt.param_groups[0]["lr"]) == float(trainer.opt.param_groups[2]["lr"]) * 20.0


def test_muon_set_peak_lr_rebases_from_search_lr_not_trunk_lr(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path)
    trainer.step = trainer._warmup_steps
    old_base_lrs = trainer._base_lrs()

    trainer.set_peak_lr(2e-3, rescale_current=False)

    new_base_lrs = trainer._base_lrs()
    assert new_base_lrs[0] == old_base_lrs[0] * 2.0
    assert new_base_lrs[2] == old_base_lrs[2] * 2.0
    assert new_base_lrs[0] == new_base_lrs[2] * 20.0


def test_muon_load_restores_peak_lr_from_search_lr(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path / "src")
    ckpt = tmp_path / "trainer.pt"
    trainer.save(ckpt)

    loaded = _make_trainer(tmp_path / "dst")
    loaded.load(ckpt)

    assert loaded._peak_lr == 1e-3
    assert loaded._base_lrs()[0] == loaded._base_lrs()[2] * 20.0
