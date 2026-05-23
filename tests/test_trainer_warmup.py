from __future__ import annotations

from pathlib import Path

import torch

from chess_anti_engine.train.soda import SODAWeightDecayWrapper
from chess_anti_engine.train.trainer import Trainer, trainer_kwargs_from_config


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


class _TinyScopedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "ffn": torch.nn.Linear(4, 4),
                        "q_proj": torch.nn.Linear(4, 4),
                        "k_proj": torch.nn.Linear(4, 4),
                        "v_proj": torch.nn.Linear(4, 4),
                        "out_proj": torch.nn.Linear(4, 4),
                    }
                )
            ]
        )
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


def _make_aurora_trainer(tmp_path: Path) -> Trainer:
    return Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        warmup_steps=10,
        warmup_lr_start=1e-5,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )


def _make_custom_aurora_trainer(tmp_path: Path) -> Trainer:
    return Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        matrix_lr_multiplier=12.0,
        matrix_weight_decay=3e-5,
        aux_weight_decay=7e-5,
        warmup_steps=10,
        warmup_lr_start=1e-5,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )


def _make_muon_scope_trainer(tmp_path: Path, scope: str) -> Trainer:
    return Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="muon",
        matrix_optimizer_scope=scope,
        warmup_steps=10,
        warmup_lr_start=1e-5,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )


def _make_scoped_trainer(tmp_path: Path, optimizer: str, scope: str) -> Trainer:
    return Trainer(
        _TinyScopedModel(),
        device="cpu",
        lr=1e-3,
        optimizer=optimizer,
        matrix_optimizer_scope=scope,
        matrix_weight_decay=3e-5,
        aux_weight_decay=7e-5,
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


def test_aurora_uses_matrix_lr_and_adam_fallback_lr(tmp_path: Path) -> None:
    trainer = _make_aurora_trainer(tmp_path)
    aurora_groups = [pg for pg in trainer.opt.param_groups if pg.get("use_aurora", False)]
    fallback_groups = [pg for pg in trainer.opt.param_groups if not pg.get("use_aurora", False)]
    named_params = dict(trainer.model.named_parameters())
    aurora_param_ids = {id(param) for pg in aurora_groups for param in pg["params"]}
    fallback_param_ids = {id(param) for pg in fallback_groups for param in pg["params"]}

    assert len(aurora_groups) == 1
    assert all(param.ndim == 2 for param in aurora_groups[0]["params"])
    assert float(aurora_groups[0]["lr"]) == float(fallback_groups[0]["lr"]) * 20.0
    assert id(named_params["blocks.0.weight"]) in aurora_param_ids
    assert id(named_params["embed.weight"]) in fallback_param_ids


def test_aurora_accepts_matrix_lr_multiplier_and_weight_decay(tmp_path: Path) -> None:
    trainer = _make_custom_aurora_trainer(tmp_path)
    aurora_groups = [pg for pg in trainer.opt.param_groups if pg.get("use_aurora", False)]
    fallback_decay_groups = [
        pg for pg in trainer.opt.param_groups
        if not pg.get("use_aurora", False) and float(pg.get("weight_decay", 0.0)) > 0.0
    ]

    assert float(aurora_groups[0]["lr"]) == float(fallback_decay_groups[0]["lr"]) * 12.0
    assert float(aurora_groups[0]["weight_decay"]) == 3e-5
    assert float(fallback_decay_groups[0]["weight_decay"]) == 7e-5


def test_soda_weight_decay_mode_replaces_only_decay_groups(tmp_path: Path) -> None:
    trainer = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        optimizer="aurora",
        matrix_lr_multiplier=12.0,
        matrix_weight_decay=3e-5,
        aux_weight_decay=7e-5,
        weight_decay_mode="soda",
        warmup_steps=10,
        warmup_lr_start=1e-5,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )

    assert isinstance(trainer.opt, SODAWeightDecayWrapper)
    soda_groups = [pg for pg in trainer.opt.param_groups if pg.get("soda_regularize", False)]
    no_soda_groups = [pg for pg in trainer.opt.param_groups if not pg.get("soda_regularize", False)]

    assert len(soda_groups) == 2
    assert all(float(pg.get("weight_decay", 0.0)) == 0.0 for pg in soda_groups)
    assert sorted(float(pg["soda_replaced_weight_decay"]) for pg in soda_groups) == [3e-5, 7e-5]
    assert all(float(pg.get("weight_decay", 0.0)) == 0.0 for pg in no_soda_groups)


def test_soda_wrapper_applies_anchor_average_after_base_step() -> None:
    param = torch.nn.Parameter(torch.tensor([1.0]))
    base = torch.optim.SGD([
        {"params": [param], "lr": 0.1, "weight_decay": 0.0, "soda_regularize": True}
    ])
    opt = SODAWeightDecayWrapper(base)

    param.grad = torch.tensor([1.0])
    opt.step()
    assert torch.allclose(param, torch.tensor([0.9]))

    param.grad = torch.tensor([1.0])
    opt.step()
    assert torch.allclose(param, torch.tensor([0.8333333]), atol=1e-6)


def test_muon_matrix_scope_can_target_mlp_without_embed(tmp_path: Path) -> None:
    trainer = _make_muon_scope_trainer(tmp_path, "mlp_only")
    muon_groups = [pg for pg in trainer.opt.param_groups if pg.get("use_muon", False)]
    fallback_groups = [pg for pg in trainer.opt.param_groups if not pg.get("use_muon", False)]
    named_params = dict(trainer.model.named_parameters())
    muon_param_ids = {id(param) for pg in muon_groups for param in pg["params"]}
    fallback_param_ids = {id(param) for pg in fallback_groups for param in pg["params"]}

    assert id(named_params["blocks.0.weight"]) not in muon_param_ids
    assert id(named_params["embed.weight"]) in fallback_param_ids


def test_cosmos_fast_matrix_scope_can_target_mlp_with_adam_fallback(tmp_path: Path) -> None:
    trainer = _make_scoped_trainer(tmp_path, "cosmos_fast", "mlp_only")
    cosmos_groups = [pg for pg in trainer.opt.param_groups if pg.get("use_cosmos_fast", False)]
    fallback_groups = [pg for pg in trainer.opt.param_groups if not pg.get("use_cosmos_fast", False)]
    named_params = dict(trainer.model.named_parameters())
    cosmos_param_ids = {id(param) for pg in cosmos_groups for param in pg["params"]}
    fallback_param_ids = {id(param) for pg in fallback_groups for param in pg["params"]}

    assert id(named_params["blocks.0.ffn.weight"]) in cosmos_param_ids
    assert id(named_params["blocks.0.out_proj.weight"]) in fallback_param_ids
    assert id(named_params["head.weight"]) in fallback_param_ids
    assert float(cosmos_groups[0]["weight_decay"]) == 3e-5
    assert any(float(pg["weight_decay"]) == 7e-5 for pg in fallback_groups)


def test_aurora_matrix_scope_can_target_mlp_out_v_without_qk(tmp_path: Path) -> None:
    trainer = _make_scoped_trainer(tmp_path, "aurora", "mlp_out_v")
    aurora_groups = [pg for pg in trainer.opt.param_groups if pg.get("use_aurora", False)]
    fallback_groups = [pg for pg in trainer.opt.param_groups if not pg.get("use_aurora", False)]
    named_params = dict(trainer.model.named_parameters())
    aurora_param_ids = {id(param) for pg in aurora_groups for param in pg["params"]}
    fallback_param_ids = {id(param) for pg in fallback_groups for param in pg["params"]}

    assert id(named_params["blocks.0.ffn.weight"]) in aurora_param_ids
    assert id(named_params["blocks.0.out_proj.weight"]) in aurora_param_ids
    assert id(named_params["blocks.0.v_proj.weight"]) in aurora_param_ids
    assert id(named_params["blocks.0.q_proj.weight"]) in fallback_param_ids
    assert id(named_params["blocks.0.k_proj.weight"]) in fallback_param_ids


def test_cosmos_matrix_scope_can_target_mlp_with_adam_fallback(tmp_path: Path) -> None:
    trainer = _make_scoped_trainer(tmp_path, "cosmos", "mlp_only")
    cosmos_groups = [pg for pg in trainer.opt.param_groups if pg.get("use_cosmos", True)]
    fallback_groups = [pg for pg in trainer.opt.param_groups if not pg.get("use_cosmos", True)]
    named_params = dict(trainer.model.named_parameters())
    cosmos_param_ids = {id(param) for pg in cosmos_groups for param in pg["params"]}
    fallback_param_ids = {id(param) for pg in fallback_groups for param in pg["params"]}

    assert id(named_params["blocks.0.ffn.weight"]) in cosmos_param_ids
    assert id(named_params["blocks.0.out_proj.weight"]) in fallback_param_ids
    assert id(named_params["head.weight"]) in fallback_param_ids
    assert float(cosmos_groups[0]["weight_decay"]) == 3e-5
    assert any(float(pg["weight_decay"]) == 7e-5 for pg in fallback_groups)


def test_soap_matrix_scope_can_target_mlp_with_adam_fallback(tmp_path: Path) -> None:
    trainer = _make_scoped_trainer(tmp_path, "soap", "mlp_only")
    named_params = dict(trainer.model.named_parameters())
    soap_param_ids = {id(param) for param in trainer.opt.param_groups[0]["params"]}
    fallback_param_ids = {
        id(param)
        for pg in trainer.opt.param_groups[1:]
        for param in pg["params"]
    }

    assert id(named_params["blocks.0.ffn.weight"]) in soap_param_ids
    assert id(named_params["blocks.0.out_proj.weight"]) in fallback_param_ids
    assert id(named_params["head.weight"]) in fallback_param_ids
    assert float(trainer.opt.param_groups[0]["weight_decay"]) == 3e-5
    assert any(float(pg["weight_decay"]) == 7e-5 for pg in trainer.opt.param_groups[1:])


def test_zclip_max_norm_can_be_disabled_from_config(tmp_path: Path) -> None:
    kwargs = trainer_kwargs_from_config(
        {"zclip_max_norm": None, "zclip_clip_factor": 0.75},
        log_dir=tmp_path,
    )
    assert kwargs["zclip_max_norm"] is None
    assert kwargs["zclip_clip_factor"] == 0.75

    trainer = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        zclip_max_norm=kwargs["zclip_max_norm"],
        zclip_clip_factor=kwargs["zclip_clip_factor"],
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )

    assert trainer.zclip.max_grad_norm is None
    assert trainer.zclip.clip_factor == 0.75


def test_zclip_step_reports_hard_and_adaptive_clipping(tmp_path: Path) -> None:
    trainer = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        zclip_z_thresh=2.0,
        zclip_max_norm=1.0,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )
    for param in trainer.model.parameters():
        param.grad = torch.ones_like(param)

    _grad_norm, stats = trainer._zclip_step(collect_stats=True)

    assert stats is not None
    assert stats["hard_clip"] == 1.0
    assert stats["clipped"] == 1.0

    trainer = Trainer(
        _TinyMuonModel(),
        device="cpu",
        lr=1e-3,
        zclip_z_thresh=2.0,
        zclip_max_norm=None,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
    )
    trainer.zclip.initialized = True
    trainer.zclip.mean = 1.0
    trainer.zclip.var = 0.01
    for param in trainer.model.parameters():
        param.grad = torch.ones_like(param)

    _grad_norm, stats = trainer._zclip_step(collect_stats=True)

    assert stats is not None
    assert stats["adaptive_clip"] == 1.0
    assert stats["hard_clip"] == 0.0
    assert stats["clipped"] == 1.0
