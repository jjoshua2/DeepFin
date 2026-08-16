from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, ModelConfig, build_model
from chess_anti_engine.moves import POLICY_SIZE
from tests.script_loading import load_script_module


def _load_shrink_module() -> Any:
    return load_script_module("shrink_ffn_checkpoint.py", "shrink_ffn_checkpoint_test_module")


def _small_cfg(*, ffn_mult_by_layer: tuple[float, ...]) -> ModelConfig:
    return ModelConfig(
        kind="transformer",
        input_extra_features="v1",
        embed_dim=8,
        num_layers=2,
        num_heads=2,
        ffn_mult=sum(ffn_mult_by_layer) / len(ffn_mult_by_layer),
        ffn_mult_by_layer=ffn_mult_by_layer,
        use_smolgen=False,
        use_deepnorm=False,
    )


def test_shrink_model_state_dict_prunes_important_ffn_units() -> None:
    module = _load_shrink_module()
    source_cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    target_cfg = _small_cfg(ffn_mult_by_layer=(1.0, 1.0))
    source_state = build_model(source_cfg).state_dict()
    target_template = build_model(target_cfg).state_dict()

    with torch.no_grad():
        up_w = source_state["blocks.0.ffn.0.weight"]
        up_b = source_state["blocks.0.ffn.0.bias"]
        down_w = source_state["blocks.0.ffn.2.weight"]
        up_w.zero_()
        up_b.zero_()
        down_w.zero_()
        keep_units = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], dtype=torch.long)
        for rank, unit in enumerate(keep_units.tolist(), start=1):
            up_w[unit, 0] = float(rank)

    out, copied = module.shrink_model_state_dict(
        source_state,
        target_template,
        target_hidden_sizes=(8, 8),
    )

    assert copied
    assert out["embed.weight"].shape == source_state["embed.weight"].shape
    torch.testing.assert_close(out["embed.weight"], source_state["embed.weight"])
    torch.testing.assert_close(out["blocks.0.ffn.0.weight"], source_state["blocks.0.ffn.0.weight"][keep_units])
    torch.testing.assert_close(out["blocks.0.ffn.0.bias"], source_state["blocks.0.ffn.0.bias"][keep_units])
    torch.testing.assert_close(out["blocks.0.ffn.2.weight"], source_state["blocks.0.ffn.2.weight"][:, keep_units])
    torch.testing.assert_close(out["blocks.0.ffn.2.bias"], source_state["blocks.0.ffn.2.bias"])


def test_shrink_model_state_dict_can_use_external_unit_scores() -> None:
    module = _load_shrink_module()
    source_cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    target_cfg = _small_cfg(ffn_mult_by_layer=(1.0, 1.0))
    source_state = build_model(source_cfg).state_dict()
    target_template = build_model(target_cfg).state_dict()

    layer0_scores = torch.zeros(16)
    keep_units = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14], dtype=torch.long)
    layer0_scores[keep_units] = torch.arange(1, 9, dtype=torch.float32)

    out, _copied = module.shrink_model_state_dict(
        source_state,
        target_template,
        target_hidden_sizes=(8, 8),
        unit_scores_by_layer={0: layer0_scores},
    )

    torch.testing.assert_close(out["blocks.0.ffn.0.weight"], source_state["blocks.0.ffn.0.weight"][keep_units])
    torch.testing.assert_close(out["blocks.0.ffn.2.weight"], source_state["blocks.0.ffn.2.weight"][:, keep_units])


def test_activation_unit_scores_from_x_shapes() -> None:
    module = _load_shrink_module()
    source_cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    ckpt = {
        "model": build_model(source_cfg).state_dict(),
        "arch": {
            "_schema_version": ARCH_SCHEMA_VERSION,
            **dataclasses.asdict(source_cfg),
        },
    }
    x = torch.randn(4, 146, 8, 8)

    scores, stats = module.activation_unit_scores_from_x(
        ckpt,
        ckpt_path=Path("trainer.pt"),
        x=x,
        batch_size=2,
        device=torch.device("cpu"),
    )

    assert stats["activation_score_positions"] == 4
    assert stats["activation_score_layers"] == 2
    assert set(scores) == {0, 1}
    assert scores[0].shape == (16,)
    assert scores[1].shape == (12,)
    assert torch.isfinite(scores[0]).all()
    assert torch.isfinite(scores[1]).all()
    assert torch.all(scores[0] >= 0)
    assert torch.all(scores[1] >= 0)


def test_taylor_unit_scores_from_arrays_shapes() -> None:
    module = _load_shrink_module()
    source_cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    ckpt = {
        "model": build_model(source_cfg).state_dict(),
        "arch": {
            "_schema_version": ARCH_SCHEMA_VERSION,
            **dataclasses.asdict(source_cfg),
        },
    }
    n = 3
    arrs = {
        "x": torch.randn(n, 146, 8, 8).numpy(),
        "policy_target": torch.nn.functional.one_hot(
            torch.zeros(n, dtype=torch.long),
            num_classes=POLICY_SIZE,
        ).float().numpy(),
        "wdl_target": torch.zeros(n, dtype=torch.long).numpy(),
    }

    scores, stats = module.taylor_unit_scores_from_arrays(
        ckpt,
        ckpt_path=Path("trainer.pt"),
        arrs=arrs,
        batch_size=2,
        device=torch.device("cpu"),
        loss_kwargs={},
    )

    assert stats["taylor_score_positions"] == n
    assert stats["taylor_score_layers"] == 2
    assert set(scores) == {0, 1}
    assert scores[0].shape == (16,)
    assert scores[1].shape == (12,)
    assert torch.isfinite(scores[0]).all()
    assert torch.isfinite(scores[1]).all()
    assert torch.all(scores[0] >= 0)
    assert torch.all(scores[1] >= 0)


def test_optimize_target_schedule_by_scores_redistributes_budget() -> None:
    module = _load_shrink_module()
    layer0_scores = torch.arange(32, 0, -1, dtype=torch.float32)
    layer1_scores = torch.ones(32, dtype=torch.float32)

    schedule, stats = module.optimize_target_schedule_by_scores(
        source_hidden_sizes=(32, 32),
        source_embed_dim=16,
        budget_schedule=(1.0, 1.0),
        unit_scores_by_layer={0: layer0_scores, 1: layer1_scores},
        min_ffn_mult=0.5,
        hidden_multiple=8,
    )

    assert schedule == (1.5, 0.5)
    assert stats["optimized_target_hidden"] == (24, 8)
    assert sum(stats["optimized_target_hidden"]) == 32


def test_optimize_target_schedule_rounds_nonaligned_budget_without_error() -> None:
    module = _load_shrink_module()
    layer0_scores = torch.arange(32, 0, -1, dtype=torch.float32)
    layer1_scores = torch.ones(32, dtype=torch.float32)

    # Requested per-layer hidden = int(16*1.1)=17 and int(16*1.0)=16 -> total 33,
    # which is NOT on the (min_budget + k*multiple) grid. Previously this raised;
    # now it snaps to the nearest reachable grid point instead.
    schedule, stats = module.optimize_target_schedule_by_scores(
        source_hidden_sizes=(32, 32),
        source_embed_dim=16,
        budget_schedule=(1.1, 1.0),
        unit_scores_by_layer={0: layer0_scores, 1: layer1_scores},
        min_ffn_mult=0.5,
        hidden_multiple=8,
    )

    achieved = int(sum(stats["optimized_target_hidden"]))
    assert stats["optimized_requested_hidden_budget"] == 33
    assert stats["optimized_total_hidden_budget"] == achieved
    assert achieved % 8 == 0  # snapped onto the multiple grid
    assert abs(achieved - 33) <= 8  # within one block of the request
    assert schedule == (1.5, 0.5)


def test_shrink_checkpoint_updates_arch_and_drops_optimizer(tmp_path: Path) -> None:
    """⚑ Includes the manifest invariant: `opt_param_names` travels with `opt`.

    `new_ckpt = dict(ckpt)` copies the manifest across, and this script pops
    `opt` while rewriting `model` to a NARROWER architecture. Leaving
    `opt_param_names` behind therefore produces a checkpoint whose manifest
    describes an optimizer state it does not contain, at widths its payload no
    longer has — the self-inconsistent shape
    `Trainer._remap_optimizer_state_by_param_name` raises
    `UntrustedOptimizerStateError` on, and the one
    `scripts/reinit_value_heads.py` was fixed for in PR #427. This script is the
    sibling producer that fix missed; the source fixture below carries a real
    manifest so the assertion is not vacuous.
    """
    module = _load_shrink_module()
    source_cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    target_schedule = (1.0, 1.0)
    source_model = build_model(source_cfg)
    ckpt = {
        "model": source_model.state_dict(),
        "swa_model": build_model(source_cfg).state_dict(),
        "opt": {"state": {"stale": True}},
        "opt_param_names": [n for n, _ in source_model.named_parameters()],
        "scheduler": {"stale": True},
        "step": 123,
        "arch": {
            "_schema_version": ARCH_SCHEMA_VERSION,
            **dataclasses.asdict(source_cfg),
        },
    }
    assert ckpt["opt_param_names"], "fixture needs a real manifest to drop"

    new_ckpt, target_cfg, stats = module.shrink_checkpoint(
        ckpt,
        ckpt_path=tmp_path / "trainer.pt",
        target_schedule=target_schedule,
    )

    assert target_cfg.ffn_mult_by_layer == target_schedule
    assert tuple(new_ckpt["arch"]["ffn_mult_by_layer"]) == target_schedule
    assert new_ckpt["step"] == 123
    # THE INVARIANT. Survives any future edit that keeps `opt`, as long as the
    # manifest keeps travelling with it; fails only on a genuine desync.
    assert ("opt" in new_ckpt) == ("opt_param_names" in new_ckpt), (
        "manifest and optimizer state must travel together -- a checkpoint "
        "carrying exactly one of them describes a state it does not contain"
    )
    # TODAY'S BEHAVIOUR, pinned separately and deliberately: the script drops
    # both outright. Update these lines, not the one above, if that changes.
    assert "opt" not in new_ckpt
    assert "opt_param_names" not in new_ckpt, (
        "a manifest naming the pre-shrink parameters was left behind -- it "
        "describes an optimizer state this checkpoint does not contain, at "
        "widths its payload no longer has"
    )
    assert "scheduler" not in new_ckpt
    assert stats["dropped_optimizer"] is True
    assert stats["dropped_scheduler"] is True

    target_model = build_model(target_cfg)
    target_model.load_state_dict(new_ckpt["model"])
    target_model.load_state_dict(new_ckpt["swa_model"])


def test_shrink_checkpoint_handles_trainer_swa_state_dict(tmp_path: Path) -> None:
    module = _load_shrink_module()
    source_cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    target_schedule = (1.0, 1.0)
    source_model = build_model(source_cfg)
    swa_model = torch.optim.swa_utils.AveragedModel(source_model)
    swa_model.update_parameters(source_model)
    ckpt = {
        "model": source_model.state_dict(),
        "swa_model": swa_model.state_dict(),
        "arch": {
            "_schema_version": ARCH_SCHEMA_VERSION,
            **dataclasses.asdict(source_cfg),
        },
    }

    new_ckpt, target_cfg, _stats = module.shrink_checkpoint(
        ckpt,
        ckpt_path=tmp_path / "trainer.pt",
        target_schedule=target_schedule,
    )

    assert "n_averaged" in new_ckpt["swa_model"]
    assert "module.embed.weight" in new_ckpt["swa_model"]
    assert "embed.weight" not in new_ckpt["swa_model"]

    target_swa = torch.optim.swa_utils.AveragedModel(build_model(target_cfg))
    target_swa.load_state_dict(new_ckpt["swa_model"])


def test_shrink_checkpoint_preserves_compiled_swa_prefix(tmp_path: Path) -> None:
    module = _load_shrink_module()
    source_cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    target_schedule = (1.0, 1.0)
    source_model = build_model(source_cfg)
    swa_model = torch.optim.swa_utils.AveragedModel(source_model)
    swa_model.update_parameters(source_model)

    # Simulate a checkpoint saved with use_compile=True AND SWA enabled:
    # AveragedModel wraps the compiled module, so weight keys carry the
    # module._orig_mod. prefix (n_averaged stays unprefixed).
    compiled_swa = {
        (f"module._orig_mod.{key.removeprefix('module.')}" if key.startswith("module.") else key): value
        for key, value in swa_model.state_dict().items()
    }
    assert any(key.startswith("module._orig_mod.") for key in compiled_swa)

    ckpt = {
        "model": source_model.state_dict(),
        "swa_model": compiled_swa,
        "arch": {
            "_schema_version": ARCH_SCHEMA_VERSION,
            **dataclasses.asdict(source_cfg),
        },
    }

    new_ckpt, target_cfg, _stats = module.shrink_checkpoint(
        ckpt,
        ckpt_path=tmp_path / "trainer.pt",
        target_schedule=target_schedule,
    )

    out_swa = new_ckpt["swa_model"]
    weight_keys = [key for key in out_swa if key != "n_averaged"]
    # Compiled wrapper prefix is preserved (not collapsed to bare module.*).
    assert weight_keys
    assert all(key.startswith("module._orig_mod.") for key in weight_keys)
    assert "n_averaged" in out_swa
    # Pruned tensors still load into the freshly built target architecture.
    stripped = {
        key.removeprefix("module._orig_mod."): value
        for key, value in out_swa.items()
        if key != "n_averaged"
    }
    build_model(target_cfg).load_state_dict(stripped)


def test_shrink_checkpoint_reuses_main_keep_indices_for_swa(tmp_path: Path) -> None:
    module = _load_shrink_module()
    source_cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    target_schedule = (1.0, 1.0)
    source_model = build_model(source_cfg)
    swa_model = torch.optim.swa_utils.AveragedModel(source_model)
    source_state = source_model.state_dict()
    swa_state = swa_model.state_dict()
    main_keep = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14], dtype=torch.long)
    swa_preferred = torch.tensor([1, 3, 5, 7, 9, 11, 13, 15], dtype=torch.long)

    with torch.no_grad():
        main_up = source_state["blocks.0.ffn.0.weight"]
        main_bias = source_state["blocks.0.ffn.0.bias"]
        main_down = source_state["blocks.0.ffn.2.weight"]
        main_up.zero_()
        main_bias.zero_()
        main_down.zero_()
        for rank, unit in enumerate(main_keep.tolist(), start=1):
            main_up[unit, 0] = float(rank)

        swa_up = swa_state["module.blocks.0.ffn.0.weight"]
        swa_bias = swa_state["module.blocks.0.ffn.0.bias"]
        swa_down = swa_state["module.blocks.0.ffn.2.weight"]
        swa_up.zero_()
        swa_bias.zero_()
        swa_down.zero_()
        for rank, unit in enumerate(swa_preferred.tolist(), start=1):
            swa_up[unit, 0] = float(rank)

    ckpt = {
        "model": source_state,
        "swa_model": swa_state,
        "arch": {
            "_schema_version": ARCH_SCHEMA_VERSION,
            **dataclasses.asdict(source_cfg),
        },
    }

    new_ckpt, _target_cfg, _stats = module.shrink_checkpoint(
        ckpt,
        ckpt_path=tmp_path / "trainer.pt",
        target_schedule=target_schedule,
    )

    torch.testing.assert_close(
        new_ckpt["swa_model"]["module.blocks.0.ffn.0.weight"],
        swa_state["module.blocks.0.ffn.0.weight"].index_select(0, main_keep),
    )
    assert not torch.equal(
        new_ckpt["swa_model"]["module.blocks.0.ffn.0.weight"],
        swa_state["module.blocks.0.ffn.0.weight"].index_select(0, swa_preferred),
    )
