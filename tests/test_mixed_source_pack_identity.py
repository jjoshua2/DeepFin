from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from chess_anti_engine.moves import COMPACT_POLICY_SIZE
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer
from chess_anti_engine.replay.mixed_source_pack_identity import (
    PackSourceIdentity,
    PreservedParentPack,
    SourceGameIdentity,
)
from chess_anti_engine.replay.shard import (
    ShardMeta,
    samples_to_arrays,
    save_local_shard_arrays,
)


def _source_row(game: int, row: int) -> ReplaySample:
    policy = np.zeros(COMPACT_POLICY_SIZE, dtype=np.float32)
    legal = np.zeros(COMPACT_POLICY_SIZE, dtype=np.uint8)
    policy[row] = 1
    legal[row] = 1
    return ReplaySample(
        x=np.full((146, 8, 8), row, dtype=np.float32),
        policy_target=policy, legal_mask=legal, wdl_target=row % 3,
        priority=1, game_id=game, ply_index=row, has_policy=True,
    )


def _write(root: Path, index: int, source: PackSourceIdentity,
           contract: PreservedParentPack) -> Path:
    rows = [_source_row(game, index * 16 + game * 2 + ply)
            for game in range(4) for ply in range(2)]
    identities = [SourceGameIdentity(
        source.run_manifest_sha256, source.source_namespace,
        source.opening_stratum, game,
    ) for game in range(4) for _ in range(2)]
    arrays = samples_to_arrays(rows)
    contract.validate_shard_rows(source, arrays, identities)
    path = contract.source_parent(root, source) / "shard_000000.zarr"
    save_local_shard_arrays(
        path, arrs=arrays,
        meta=ShardMeta(positions=len(rows), policy_encoding="lc0_1858",
                       policy_size=COMPACT_POLICY_SIZE,
                       input_history_encoding=None, history_rep_fix=False),
    )
    return path


def test_bt4_ceres_sf_local_ids_remain_distinct_through_actual_loader(
    tmp_path: Path,
) -> None:
    # Every source reuses game IDs 0..3. A flat physical parent would merge
    # these 12 semantic games into four loader games.
    sources = [PackSourceIdentity(code * 64, namespace, "fp16_root_a")
               for code, namespace in zip("abc", ("bt4", "ceres", "sf"), strict=True)]
    contract = PreservedParentPack(sources)
    assert contract.sha256 == PreservedParentPack(list(reversed(sources))).sha256
    root = tmp_path / "pack"
    shards = [(source, _write(root, index, source, contract))
              for index, source in enumerate(sources)]
    wrong_flat = tmp_path / "wrong_flat"
    wrong_flat.mkdir()
    for index, (_, path) in enumerate(shards):
        shutil.copytree(path, wrong_flat / f"shard_{index:06d}.zarr")
    collision = GameAwareEpochBuffer(
        shard_dir=wrong_flat, batch_size=4, seed=11, input_planes=146,
        input_history_encoding="legacy", history_rep_fix=False,
        mirror_augmentation=False, plan_workers=1, load_workers=1,
        max_working_set_bytes=64 * 1024 * 1024,
    )
    assert collision.plan.game_count == 4
    collision.close()
    staged = tmp_path / "staged"
    contract.stage_shards(root, staged, shards)
    assert len({path.resolve().parent for path in staged.iterdir()}) == 3

    buf = GameAwareEpochBuffer(
        shard_dir=staged, batch_size=4, seed=11, input_planes=146,
        input_history_encoding="legacy", history_rep_fix=False,
        mirror_augmentation=False, plan_workers=1, load_workers=1,
        max_working_set_bytes=64 * 1024 * 1024,
    )
    assert buf.plan.rows == 24
    assert buf.plan.game_count == 12
    assert buf.plan.source_count == 3
    seen: set[int] = set()
    for _ in range(buf.num_batches):
        batch = buf.sample_batch_arrays(4)
        rows = np.asarray(batch["ply_index"]).tolist()
        games = np.asarray(batch["game_id"]).tolist()
        assert len(games) == len(set(games))
        assert not seen.intersection(rows)
        seen.update(rows)
    assert len(seen) == 24
    assert buf.receipt()["complete"]
    buf.close()


def test_pack_contract_refuses_cross_source_rows_and_flattened_shard(
    tmp_path: Path,
) -> None:
    first = PackSourceIdentity("a" * 64, "bt4", "fp16_a")
    second = PackSourceIdentity("a" * 64, "bt4", "fp16_b")
    contract = PreservedParentPack([first, second])
    assert contract.source_parent(tmp_path, first) != contract.source_parent(tmp_path, second)
    row = SourceGameIdentity("a" * 64, "bt4", "fp16_a", 7)
    arrays = {"game_id": np.array([7], dtype=np.int64),
              "has_game_id": np.array([1], dtype=np.uint8)}
    with pytest.raises(ValueError, match="mismatch"):
        contract.validate_shard_rows(second, arrays, [row])
    with pytest.raises(ValueError, match="mismatch"):
        contract.validate_shard_rows(
            first, {**arrays, "game_id": np.array([8])}, [row]
        )
    with pytest.raises(ValueError, match="uint8 presence flag"):
        contract.validate_shard_rows(
            first, {**arrays, "has_game_id": np.array([True])}, [row]
        )
    with pytest.raises(ValueError, match="uint8 presence flag"):
        contract.validate_shard_rows(
            first, {**arrays, "has_game_id": np.array([[1]], dtype=np.uint8)}, [row]
        )
    with pytest.raises(ValueError, match="must equal 1"):
        contract.validate_shard_rows(
            first, {**arrays, "has_game_id": np.array([2], dtype=np.uint8)}, [row]
        )
    flat = tmp_path / "flat" / "shard_000000.zarr"
    flat.mkdir(parents=True)
    contract.source_parent(tmp_path, first).mkdir(parents=True)
    with pytest.raises(ValueError, match="outside its preserved source parent"):
        contract.stage_shards(tmp_path, tmp_path / "stage", [(first, flat)])
    alias = contract.source_parent(tmp_path, second)
    alias.symlink_to(contract.source_parent(tmp_path, first), target_is_directory=True)
    inside = contract.source_parent(tmp_path, first) / "shard_000000.zarr"
    inside.mkdir()
    with pytest.raises(ValueError, match="must not be symlinks"):
        contract.stage_shards(tmp_path, tmp_path / "stage", [(second, alias / inside.name)])


def test_zip_staging_is_explicit_and_rejects_symlinked_roots(tmp_path: Path) -> None:
    source = PackSourceIdentity("a" * 64, "bt4", "fp16_a")
    directory = PreservedParentPack([source])
    packed = PreservedParentPack([source], shard_format="zip_stored")
    assert packed.sha256 != directory.sha256
    root = tmp_path / "pack"
    physical = packed.source_parent(root, source) / "shard_000000.zarr.zip"
    physical.parent.mkdir(parents=True)
    physical.touch()
    stage = tmp_path / "stage"
    packed.stage_shards(root, stage, [(source, physical)])
    assert (stage / physical.name).resolve() == physical
    with pytest.raises(ValueError, match="outside its preserved source parent"):
        directory.stage_shards(root, tmp_path / "wrong_format", [(source, physical)])
    linked_root = tmp_path / "linked_pack"
    linked_root.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="canonical nonsymlinks"):
        packed.stage_shards(linked_root, tmp_path / "linked_stage", [(source, physical)])
