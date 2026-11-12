from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
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


def _calibration_shards(path: Path, widths: tuple[int, ...]) -> np.ndarray:
    from chess_anti_engine.replay import ReplaySample
    from chess_anti_engine.replay.shard import samples_to_arrays, save_local_shard_arrays

    chunks = []
    for number, width in enumerate(widths):
        samples = []
        for row in range(2):
            x = np.full((146, 8, 8), (number * 2 + row) / 8.0, dtype=np.float32)
            policy = np.full(width, 1.0 / width, dtype=np.float32)
            sample = ReplaySample(x=x, policy_target=policy, wdl_target=1)
            sample.input_history_encoding = "legacy"
            samples.append(sample)
            chunks.append(x)
        save_local_shard_arrays(
            path / f"shard_{number:06d}.zarr", arrs=samples_to_arrays(samples), meta={"positions": 2},
        )
    return np.stack(chunks)


def _calibration_kwargs(path: Path, cfg: ModelConfig, *, positions: int = 4) -> dict[str, Any]:
    return {
        "source_cfg": cfg, "replay_dir": path, "positions": positions,
        "prefer_recorded_lc0_root": False, "synthetic_lc0_root_history": False,
        "lc0_root_legacy_meta": False,
    }


@pytest.mark.parametrize("widths", [(1858, 4672), (1858, 1858)])
def test_activation_mixed_policy_widths_match_direct_scores(tmp_path: Path, widths: tuple[int, ...]) -> None:
    module = _load_shrink_module()
    cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    x = _calibration_shards(tmp_path, widths)
    ckpt = {"model": build_model(cfg).state_dict(), "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **dataclasses.asdict(cfg)}}
    common = {"ckpt_path": Path("trainer.pt"), "batch_size": 2, "device": torch.device("cpu")}
    expected, expected_stats = module.activation_unit_scores_from_x(ckpt, x=torch.as_tensor(x[:3]), **common)
    actual, actual_stats = module.activation_unit_scores_from_replay(
        ckpt, replay_dir=tmp_path, positions=3, prefer_recorded_lc0_root=False,
        synthetic_lc0_root_history=False, lc0_root_legacy_meta=False, **common,
    )
    assert actual_stats == expected_stats
    assert actual_stats["activation_score_positions"] == 3
    for layer, score in expected.items():
        torch.testing.assert_close(actual[layer], score, rtol=0, atol=0)


def test_activation_loader_releases_unused_targets_before_next_shard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import weakref
    from chess_anti_engine.replay import shard

    module = _load_shrink_module()
    expected = _calibration_shards(tmp_path, (1858, 4672, 1858))
    original = shard.load_shard_arrays
    original_validate = shard.validate_arrays
    refs = []

    def tracked_load(path: Path, *, lazy: bool = False) -> Any:
        # Content/history-before-yield is exercised by directory-prefix tests;
        # eager loading itself is not required for target lifetime isolation.
        assert all(ref() is None for ref in refs)
        arrays, meta = original(path, lazy=lazy)
        return arrays, meta

    def tracked_validate(arrays: dict[str, np.ndarray]) -> None:
        # Track decoded target storage, not lazy declaration proxies/cycles.
        assert isinstance(arrays["policy_target"], np.ndarray)
        refs.append(weakref.ref(arrays["policy_target"]))
        original_validate(arrays)

    monkeypatch.setattr(shard, "load_shard_arrays", tracked_load)
    monkeypatch.setattr(shard, "validate_arrays", tracked_validate)
    arrs = module._load_calibration_arrays_from_replay(
        **_calibration_kwargs(tmp_path, _small_cfg(ffn_mult_by_layer=(2.0, 1.5)), positions=6),
        inputs_only=True,
    )
    assert set(arrs) == {"x"}
    np.testing.assert_array_equal(arrs["x"], expected)
    assert len(refs) == 3
    assert all(ref() is None for ref in refs)


def test_taylor_default_loader_still_retains_targets(tmp_path: Path) -> None:
    module = _load_shrink_module()
    expected = _calibration_shards(tmp_path, (1858, 1858))
    kwargs = _calibration_kwargs(tmp_path, _small_cfg(ffn_mult_by_layer=(2.0, 1.5)))
    all_fields = module._load_calibration_arrays_from_replay(**kwargs)
    only_inputs = module._load_calibration_arrays_from_replay(**kwargs, inputs_only=True)
    assert all_fields["policy_target"].shape == (4, 1858)
    assert "wdl_target" in all_fields
    np.testing.assert_array_equal(all_fields["x"], expected)
    np.testing.assert_array_equal(only_inputs["x"], all_fields["x"])


def test_activation_loader_keeps_whole_shard_validation(tmp_path: Path) -> None:
    from chess_anti_engine.replay import ReplaySample
    from chess_anti_engine.replay.shard import samples_to_arrays, save_local_shard_arrays
    import zarr

    module = _load_shrink_module()
    policy = np.full(1858, 1.0 / 1858, dtype=np.float32)
    sample = ReplaySample(x=np.zeros((146, 8, 8), np.float32), policy_target=policy, wdl_target=1)
    arrs = samples_to_arrays([sample])
    path = save_local_shard_arrays(tmp_path / "shard_000001.zarr", arrs=arrs, meta={"positions": 1})
    group = zarr.open_group(str(path), mode="r+")
    group["policy_target"][0, 0] = np.nan
    cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    ckpt = {"model": build_model(cfg).state_dict(), "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **dataclasses.asdict(cfg)}}
    with pytest.raises(ValueError, match="policy_target"):
        module.activation_unit_scores_from_replay(
            ckpt, ckpt_path=Path("trainer.pt"), replay_dir=tmp_path, positions=1,
            batch_size=1, device=torch.device("cpu"), prefer_recorded_lc0_root=False,
            synthetic_lc0_root_history=False, lc0_root_legacy_meta=False,
        )


@pytest.mark.parametrize("recorded", [True, False])
def test_activation_input_only_preserves_history_selection(monkeypatch: pytest.MonkeyPatch, recorded: bool) -> None:
    from chess_anti_engine.replay import shard

    module = _load_shrink_module()
    cfg = dataclasses.replace(_small_cfg(ffn_mult_by_layer=(2.0, 1.5)), input_history_encoding="lc0_root")
    arrays = {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "_input_history_encoding": np.asarray("legacy"),
        "x_lc0_root": np.full((2, 146, 8, 8), 0.75, dtype=np.float32),
        "has_x_lc0_root": np.full(2, int(recorded), dtype=np.uint8),
    }
    monkeypatch.setattr(shard, "iter_shard_paths", lambda _: [Path("fixture.zarr")])
    monkeypatch.setattr(shard, "load_shard_arrays", lambda _path, *, lazy: (arrays, {}))
    kwargs = _calibration_kwargs(Path("fixture"), cfg, positions=1)
    if recorded:
        result = module._load_calibration_arrays_from_replay(**kwargs, inputs_only=True)
        np.testing.assert_array_equal(result["x"], arrays["x_lc0_root"][:1])
        assert set(result) == {"x"}
    else:
        with pytest.raises(ValueError, match="no recorded x_lc0_root"):
            module._load_calibration_arrays_from_replay(**kwargs, inputs_only=True)


@pytest.mark.parametrize(("batch_size", "positions"), [(1, 1), (3, 5), (5, 6), (8, 6)])
def test_streamed_activation_preserves_global_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int, positions: int,
) -> None:
    module = _load_shrink_module()
    cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    x = _calibration_shards(tmp_path, (1858, 4672, 1858))
    ckpt = {"model": build_model(cfg).state_dict(), "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **dataclasses.asdict(cfg)}}
    common = {"ckpt_path": Path("trainer.pt"), "batch_size": batch_size, "device": torch.device("cpu")}
    expected, stats = module.activation_unit_scores_from_x(ckpt, x=torch.as_tensor(x[:positions]), **common)

    def no_concat(_chunks: Any) -> Any:
        pytest.fail("activation replay must not concatenate selected inputs")

    monkeypatch.setattr(module, "_concat_calibration_chunks", no_concat)
    actual, actual_stats = module.activation_unit_scores_from_replay(
        ckpt, replay_dir=tmp_path, positions=positions, prefer_recorded_lc0_root=False,
        synthetic_lc0_root_history=False, lc0_root_legacy_meta=False, **common,
    )
    assert actual_stats == stats
    for layer, score in expected.items():
        torch.testing.assert_close(actual[layer], score, rtol=0, atol=0)


def test_stream_batches_release_previous_inputs_and_preserve_order() -> None:
    import weakref

    module = _load_shrink_module()
    refs = []

    def chunks():
        for number in range(7):
            assert all(ref() is None for ref in refs)
            x = np.full((2, 1), number, dtype=np.float32)
            refs.append(weakref.ref(x))
            yield {"x": x}
            del x

    batches = list(module._calibration_input_batches(chunks(), batch_size=3))
    assert [int(batch.shape[0]) for batch in batches] == [3, 3, 3, 3, 2]
    torch.testing.assert_close(torch.cat(batches).flatten(), torch.arange(7).repeat_interleave(2).float())
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize("failure", ["iterator", "forward", "empty"])
def test_activation_stream_removes_hooks_on_failure(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    module = _load_shrink_module()
    cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    model = build_model(cfg)
    ckpt = {"model": model.state_dict(), "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **dataclasses.asdict(cfg)}}
    monkeypatch.setattr(module, "build_model", lambda _cfg: model)

    def broken_forward(_x: torch.Tensor) -> Any:
        raise RuntimeError("fixture forward failure")

    def batches():
        if failure == "empty":
            return
        yield torch.zeros((1, 146, 8, 8))
        raise RuntimeError("fixture iterator failure")

    if failure == "forward":
        monkeypatch.setattr(model, "forward", broken_forward)
    error = ValueError if failure == "empty" else RuntimeError
    message = "calibration tensor is empty" if failure == "empty" else f"fixture {failure} failure"
    with pytest.raises(error, match=message):
        module._activation_unit_scores_from_batches(
            ckpt, ckpt_path=Path("trainer.pt"), batches=batches(),
            batch_size=1, device=torch.device("cpu"),
        )
    for component in model.modules():
        assert not component._forward_hooks


def test_stream_batches_reject_incompatible_input_shapes() -> None:
    module = _load_shrink_module()
    chunks = [{"x": np.zeros((1, 2), np.float32)}, {"x": np.zeros((1, 3), np.float32)}]
    with pytest.raises(ValueError, match="input shapes differ"):
        list(module._calibration_input_batches(chunks, batch_size=2))


@pytest.mark.parametrize("recorded", [True, False])
def test_activation_stream_preserves_recorded_history(monkeypatch: pytest.MonkeyPatch, recorded: bool) -> None:
    from chess_anti_engine.replay import shard

    module = _load_shrink_module()
    cfg = dataclasses.replace(_small_cfg(ffn_mult_by_layer=(2.0, 1.5)), input_history_encoding="lc0_root")
    arrays = {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "_input_history_encoding": np.asarray("legacy"),
        "x_lc0_root": np.full((2, 146, 8, 8), 0.75, dtype=np.float32),
        "has_x_lc0_root": np.full(2, int(recorded), dtype=np.uint8),
    }
    monkeypatch.setattr(shard, "iter_shard_paths", lambda _: [Path("fixture.zarr")])
    monkeypatch.setattr(shard, "load_shard_arrays", lambda _path, *, lazy: (arrays, {}))
    ckpt = {"model": build_model(cfg).state_dict(), "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **dataclasses.asdict(cfg)}}
    common = {"ckpt_path": Path("trainer.pt"), "batch_size": 3, "device": torch.device("cpu")}
    kwargs = {
        "replay_dir": Path("fixture"), "positions": 1, "prefer_recorded_lc0_root": False,
        "synthetic_lc0_root_history": False, "lc0_root_legacy_meta": False,
    }
    if not recorded:
        with pytest.raises(ValueError, match="no recorded x_lc0_root"):
            module.activation_unit_scores_from_replay(ckpt, **common, **kwargs)
        return
    actual, stats = module.activation_unit_scores_from_replay(ckpt, **common, **kwargs)
    expected, expected_stats = module.activation_unit_scores_from_x(
        ckpt, x=torch.as_tensor(arrays["x_lc0_root"][:1]), **common,
    )
    assert stats == expected_stats
    for layer, score in expected.items():
        torch.testing.assert_close(actual[layer], score, rtol=0, atol=0)


def test_activation_stream_rejects_late_malformed_target_and_removes_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import zarr

    module = _load_shrink_module()
    _calibration_shards(tmp_path, (1858, 4672))
    group = zarr.open_group(str(tmp_path / "shard_000001.zarr"), mode="r+")
    group["policy_target"][0, 0] = np.nan
    cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    model = build_model(cfg)
    ckpt = {"model": model.state_dict(), "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **dataclasses.asdict(cfg)}}
    monkeypatch.setattr(module, "build_model", lambda _cfg: model)
    with pytest.raises(ValueError, match="policy_target"):
        module.activation_unit_scores_from_replay(
            ckpt, ckpt_path=Path("trainer.pt"), replay_dir=tmp_path, positions=4,
            batch_size=1, device=torch.device("cpu"), prefer_recorded_lc0_root=False,
            synthetic_lc0_root_history=False, lc0_root_legacy_meta=False,
        )
    for component in model.modules():
        assert not component._forward_hooks


def test_stream_batches_do_not_allocate_unfilled_requested_capacity(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_shrink_module()
    arrays = [{"x": np.ones((1, 2), np.float32)}]
    original_cat = torch.cat
    allocated_rows = []

    def checked_cat(parts: Any, *, dim: int) -> torch.Tensor:
        rows = sum(int(part.shape[0]) for part in parts)
        allocated_rows.append(rows)
        assert rows == 1
        return original_cat(parts, dim=dim)

    def no_empty(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("do not allocate unused requested batch capacity")

    monkeypatch.setattr(module.torch, "empty", no_empty)
    monkeypatch.setattr(module.torch, "cat", checked_cat)
    batches = list(module._calibration_input_batches(arrays, batch_size=1_000_000))
    assert allocated_rows == [1]
    assert batches[0].shape == (1, 2)
