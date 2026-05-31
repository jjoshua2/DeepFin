from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves import COMPACT_TO_FULL_POLICY, FULL_TO_COMPACT_POLICY
from chess_anti_engine.replay.shard import (
    POLICY_ENCODING_ARRAY_KEY,
    local_shard_path,
    save_local_shard_arrays,
)
from chess_anti_engine.train.trainer import Trainer
from scripts.offline_replay_epoch import (
    _LiveShardSampler,
    _calibrate_global_board_adapter_init,
    _concat,
    _convert_policy_targets,
    _select_configured_input_history,
    _select_recorded_input_history,
)


def test_lc0_root_legacy_meta_copies_rule50_and_ep_planes() -> None:
    legacy = np.zeros((2, 146, 8, 8), dtype=np.float32)
    root = np.zeros((2, 146, 8, 8), dtype=np.float32)

    legacy[:, 100, :, 4] = 1.0
    legacy[0, 102, :, :] = 0.37
    legacy[1, 102, :, :] = 0.82
    root[:, 109, :, :] = 37.0
    root[:, 110, :, :] = 0.0
    root[:, 108, :, :] = 1.0

    out = _select_recorded_input_history(
        {
            "x": legacy,
            "x_lc0_root": root,
            "has_x_lc0_root": np.ones((2,), dtype=np.uint8),
        },
        input_history_encoding="lc0_root",
        prefer_recorded_lc0_root=True,
        lc0_root_legacy_meta=True,
    )

    np.testing.assert_allclose(out["x"][:, 109], legacy[:, 102])
    np.testing.assert_array_equal(out["x"][:, 110], legacy[:, 100])
    np.testing.assert_array_equal(out["x"][:, 108], root[:, 108])


def test_lc0_root_legacy_meta_is_opt_in() -> None:
    legacy = np.zeros((1, 146, 8, 8), dtype=np.float32)
    root = np.zeros((1, 146, 8, 8), dtype=np.float32)
    legacy[:, 102, :, :] = 0.5
    root[:, 109, :, :] = 50.0

    out = _select_recorded_input_history(
        {
            "x": legacy,
            "x_lc0_root": root,
            "has_x_lc0_root": np.ones((1,), dtype=np.uint8),
        },
        input_history_encoding="lc0_root",
        prefer_recorded_lc0_root=True,
        lc0_root_legacy_meta=False,
    )

    np.testing.assert_array_equal(out["x"], root)


def test_configured_lc0_root_uses_recorded_planes_without_extra_flags() -> None:
    legacy = np.zeros((1, 146, 8, 8), dtype=np.float32)
    root = np.ones((1, 146, 8, 8), dtype=np.float32)

    out = _select_configured_input_history(
        {
            "x": legacy,
            "x_lc0_root": root,
            "has_x_lc0_root": np.ones((1,), dtype=np.uint8),
        },
        input_history_encoding="lc0_root",
        prefer_recorded_lc0_root=False,
        synthetic_lc0_root_history=False,
        lc0_root_legacy_meta=False,
    )

    np.testing.assert_array_equal(out["x"], root)


def test_convert_policy_targets_preserves_compact_sf_move_indices() -> None:
    arrs = {
        "policy_target": np.zeros((2, 1858), dtype=np.float32),
        "_policy_size": np.array(1858, dtype=np.int32),
        "sf_move_index": np.array([10, 1000], dtype=np.int64),
        "has_sf_move": np.ones((2,), dtype=np.float32),
    }

    out = _convert_policy_targets(arrs, policy_encoding="lc0_1858")

    np.testing.assert_array_equal(out["sf_move_index"], arrs["sf_move_index"])
    np.testing.assert_array_equal(out["has_sf_move"], arrs["has_sf_move"])


def test_convert_policy_targets_keeps_compact_legal_masks_binary() -> None:
    mask = np.zeros((1, 4672), dtype=np.uint8)
    mask[0, COMPACT_TO_FULL_POLICY[[2, 5, 11]]] = 1
    arrs = {
        "policy_target": mask.astype(np.float32),
        "legal_mask": mask,
        "_policy_size": np.array(4672, dtype=np.int32),
    }

    out = _convert_policy_targets(arrs, policy_encoding="lc0_1858")

    assert out["legal_mask"].dtype == np.uint8
    assert set(np.unique(out["legal_mask"]).tolist()) == {0, 1}
    assert int(out["legal_mask"].sum()) == 3


def test_convert_policy_targets_expands_compact_targets_for_az() -> None:
    compact_policy = np.zeros((2, 1858), dtype=np.float32)
    compact_policy[0, 10] = 1.0
    compact_policy[1, 1000] = 1.0
    compact_mask = (compact_policy > 0).astype(np.uint8)
    arrs = {
        "policy_target": compact_policy,
        "legal_mask": compact_mask,
        "_policy_size": np.array(1858, dtype=np.int32),
        "sf_move_index": np.array([10, 1000], dtype=np.int64),
        "has_sf_move": np.ones((2,), dtype=np.float32),
    }

    out = _convert_policy_targets(arrs, policy_encoding="az_4672")

    assert out["policy_target"].shape == (2, 4672)
    assert out["legal_mask"].shape == (2, 4672)
    assert int(out["legal_mask"].sum()) == 2
    np.testing.assert_array_equal(
        out["sf_move_index"],
        COMPACT_TO_FULL_POLICY[np.asarray(arrs["sf_move_index"], dtype=np.int64)],
    )
    np.testing.assert_array_equal(
        FULL_TO_COMPACT_POLICY[out["sf_move_index"]],
        arrs["sf_move_index"],
    )


def test_convert_policy_targets_updates_policy_metadata_for_concat() -> None:
    def chunk(policy_size: int, encoding: str) -> dict[str, np.ndarray]:
        policy = np.zeros((1, policy_size), dtype=np.float32)
        policy[0, 0] = 1.0
        return {
            "x": np.zeros((1, 146, 8, 8), dtype=np.float32),
            "policy_target": policy,
            "wdl_target": np.zeros((1,), dtype=np.int8),
            "priority": np.ones((1,), dtype=np.float32),
            "has_policy": np.ones((1,), dtype=np.uint8),
            "_policy_size": np.array(policy_size, dtype=np.int32),
            POLICY_ENCODING_ARRAY_KEY: np.asarray(encoding),
        }

    converted_full = _convert_policy_targets(chunk(4672, "az_4672"), policy_encoding="lc0_1858")
    converted_compact = _convert_policy_targets(chunk(1858, "lc0_1858"), policy_encoding="lc0_1858")

    out = _concat([converted_full, converted_compact])

    assert int(out["_policy_size"].item()) == 1858
    assert str(out[POLICY_ENCODING_ARRAY_KEY].item()) == "lc0_1858"
    assert out["policy_target"].shape == (2, 1858)


def test_concat_preserves_optional_eval_targets_when_some_chunks_lack_them() -> None:
    policy_a = np.zeros((1, 1858), dtype=np.float32)
    policy_b = np.zeros((2, 1858), dtype=np.float32)
    full = {
        "x": np.zeros((1, 146, 8, 8), dtype=np.float32),
        "policy_target": policy_a,
        "wdl_target": np.zeros((1,), dtype=np.int8),
        "priority": np.ones((1,), dtype=np.float32),
        "has_policy": np.ones((1,), dtype=np.uint8),
        "sf_wdl": np.ones((1, 3), dtype=np.float16),
        "has_sf_wdl": np.ones((1,), dtype=np.uint8),
        "future_policy_target": policy_a.copy(),
        "has_future": np.ones((1,), dtype=np.uint8),
        "_policy_size": np.array(1858, dtype=np.int32),
    }
    minimal = {
        "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
        "policy_target": policy_b,
        "wdl_target": np.zeros((2,), dtype=np.int8),
        "priority": np.ones((2,), dtype=np.float32),
        "has_policy": np.ones((2,), dtype=np.uint8),
        "_policy_size": np.array(1858, dtype=np.int32),
    }

    out = _concat([full, minimal])

    assert out["sf_wdl"].shape == (3, 3)
    assert out["future_policy_target"].shape == (3, 1858)
    np.testing.assert_array_equal(out["has_sf_wdl"], np.array([1, 0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(out["has_future"], np.array([1, 0, 0], dtype=np.uint8))
    assert int(out["_policy_size"].item()) == 1858


def test_live_shard_sampler_rescans_and_samples_new_shards(tmp_path) -> None:
    def arrays(n: int) -> dict[str, np.ndarray]:
        policy = np.zeros((n, 4672), dtype=np.float32)
        policy[:, 0] = 1.0
        return {
            "x": np.zeros((n, 146, 8, 8), dtype=np.float32),
            "policy_target": policy,
            "wdl_target": np.zeros((n,), dtype=np.int8),
            "priority": np.ones((n,), dtype=np.float32),
            "has_policy": np.ones((n,), dtype=np.uint8),
        }

    save_local_shard_arrays(local_shard_path(tmp_path, 0), arrs=arrays(3), meta={"positions": 3})
    sampler = _LiveShardSampler(
        replay_dir=tmp_path,
        model_cfg=ModelConfig(kind="transformer"),
        rng=np.random.default_rng(1),
        prefer_recorded_lc0_root=False,
        synthetic_lc0_root_history=False,
        lc0_root_legacy_meta=False,
        cache_shards=1,
    )

    assert sampler.rescan() == 3
    assert sampler.total_positions == 3
    batch = sampler.sample_batch_arrays(2)
    assert batch["x"].shape == (2, 146, 8, 8)

    save_local_shard_arrays(local_shard_path(tmp_path, 1), arrs=arrays(4), meta={"positions": 4})
    assert sampler.rescan() == 4
    assert sampler.total_positions == 7


def test_calibrate_global_board_adapter_hits_requested_rms_ratio(tmp_path) -> None:
    model_cfg = ModelConfig(
        kind="transformer",
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_mult=1.0,
        input_global_embedding="bt4_board_adapter",
        input_global_embedding_channels=2,
        use_smolgen=False,
    )
    model = build_model(model_cfg)
    trainer = Trainer(
        model,
        device="cpu",
        lr=1e-4,
        use_amp=False,
        optimizer="adamw",
        prefetch_batches=False,
        model_config=model_cfg,
        log_dir=tmp_path,
    )
    args = Namespace(
        global_board_adapter_init_rms_ratio=0.2,
        init_checkpoint=None,
        global_board_adapter_init_batch_size=3,
        seed=123,
    )
    eval_arrs = {"x": np.random.default_rng(1).normal(size=(4, 146, 8, 8)).astype(np.float32)}
    trainer.model.train()

    stats = _calibrate_global_board_adapter_init(
        trainer=trainer,
        eval_arrs=eval_arrs,
        args=args,
    )

    assert trainer.model.training is True
    assert stats["global_board_adapter_init_batch_size"] == 3
    assert stats["global_board_adapter_init_weight_rms"] > 0.0
    assert stats["global_board_adapter_init_actual_ratio"] == pytest.approx(0.2, rel=1e-4)


def test_calibrate_global_board_adapter_rejects_checkpoint_init(tmp_path) -> None:
    model_cfg = ModelConfig(
        kind="transformer",
        embed_dim=32,
        num_layers=1,
        num_heads=4,
        ffn_mult=1.0,
        input_global_embedding="bt4_board_adapter",
        input_global_embedding_channels=2,
        use_smolgen=False,
    )
    trainer = Trainer(
        build_model(model_cfg),
        device="cpu",
        lr=1e-4,
        use_amp=False,
        optimizer="adamw",
        prefetch_batches=False,
        model_config=model_cfg,
        log_dir=tmp_path,
    )
    args = Namespace(
        global_board_adapter_init_rms_ratio=0.2,
        init_checkpoint="/tmp/model.pt",
        global_board_adapter_init_batch_size=1,
        seed=123,
    )

    with pytest.raises(ValueError, match="fresh-init"):
        _calibrate_global_board_adapter_init(
            trainer=trainer,
            eval_arrs={"x": np.zeros((1, 146, 8, 8), dtype=np.float32)},
            args=args,
        )
