"""Dense validation reuse must preserve actual targets, schedules and mutation gates."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from chess_anti_engine.replay import target_overlay as storage
from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer
from tests.test_target_overlay_v2 import fixture


def _buffer(stage: Path, ref: dict[str, str]) -> GameAwareEpochBuffer:
    return GameAwareEpochBuffer(
        shard_dir=stage, overlay_storage_qualification=ref, batch_size=2, seed=12,
        input_planes=175, input_history_encoding='lc0_root_legacy_meta', history_rep_fix=False,
        mirror_augmentation=False, plan_workers=2, load_workers=2,
        objective_mask_counter=lambda arrays: {'policy': float(arrays['policy_target'].shape[0])},
    )


def test_reuses_dense_validation_once_per_shard_and_preserves_epoch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.lc0_control_train import stage_shards
    _, roots, _, ref = fixture(tmp_path)
    stage = tmp_path / 'stage'
    stage_shards(roots, stage)
    calls: Counter[Path] = Counter()
    original = storage._open_target_manifest

    def counted(path: Path, manifest: dict[str, Any], *, seal: storage.BaseSeal | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        calls[path] += 1
        return original(path, manifest, seal=seal)

    monkeypatch.setattr(storage, '_open_target_manifest', counted)
    cached = _buffer(stage, ref)
    assert sorted(calls.values()) == [1, 1]
    expected = [cached.sample_batch_arrays(2) for _ in range(2)]
    assert sorted(calls.values()) == [1, 1]
    plan = cached.plan
    cached.close()
    calls.clear()
    # Deliberately disable only reuse; all anchored checks still run. This
    # comparison uses the same source paths, objective census and RNG schedule.
    validate = storage._validated_overlay

    def uncached(path: Path, seal: storage.BaseSeal) -> storage._ValidatedOverlay:
        result = validate(path, seal)
        seal._validated_overlays.clear()
        return result

    monkeypatch.setattr(storage, '_validated_overlay', uncached)
    baseline = _buffer(stage, ref)
    assert sorted(calls.values()) == [9, 9]
    np.testing.assert_equal(asdict(baseline.plan), asdict(plan))
    actual = [baseline.sample_batch_arrays(2) for _ in range(2)]
    baseline.close()
    for left, right in zip(expected, actual, strict=True):
        assert left.keys() == right.keys()
        for key in left:
            np.testing.assert_array_equal(left[key], right[key], err_msg=key)


@pytest.mark.parametrize('change', [
    'target', 'base', 'base_receipt', 'qualification_receipt', 'target_inode',
    'overlay_membership', 'base_membership', 'local_membership', 'root_metadata',
])
def test_reuse_rejects_changed_anchored_inputs(tmp_path: Path, change: str) -> None:
    import json
    bases, roots, _, ref = fixture(tmp_path)
    paths = [p for root in roots for p in storage.shard_paths(root)]
    _, context = storage.qualified_paths(ref, paths)
    target = paths[0]
    manifest = json.loads((target / storage.MANIFEST).read_text())
    if change in {'target', 'base'}:
        path = target if change == 'target' else Path(manifest['base'])
        array: Any = zarr.open_group(str(path), mode='a')['search_wdl']
        array[0] = [1, 0, 0]
    elif change in {'base_receipt', 'qualification_receipt', 'target_inode'}:
        path = (Path(manifest['base_seal']['path']) if change == 'base_receipt'
                else Path(ref['path']) if change == 'qualification_receipt'
                else target / storage.MANIFEST)
        other = tmp_path / 'same-bytes-new-inode'
        other.write_bytes(path.read_bytes())
        other.replace(path)
    elif change == 'overlay_membership':
        (roots[0] / 'shard_999999.zarr').mkdir()
    elif change == 'base_membership':
        (bases[0] / 'shard_999999.zarr').mkdir()
    elif change == 'local_membership':
        (target / 'unexpected').write_text('new member')
    else:
        # Modify an already anchored metadata file, keeping root membership.
        metadata = next(p for p in bases[0].iterdir() if p.is_file())
        metadata.write_bytes(metadata.read_bytes() + b' ')
    with pytest.raises(ValueError, match='changed'):
        storage.overlay_content_sha256(target, seal=context)


def test_mutation_during_first_validation_is_not_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, roots, _, _ = fixture(tmp_path)
    path = storage.shard_paths(roots[0])[0]
    context = storage.seal_for_overlay(path)
    original = storage._validate_local

    def mutate(path: Path, base: Path, names: list[str]) -> dict[str, Any]:
        attrs = original(path, base, names)
        (path / 'late-member').write_text('mutation after dense validation')
        return attrs

    monkeypatch.setattr(storage, '_validate_local', mutate)
    with pytest.raises(ValueError, match='changed during identity read'):
        storage.overlay_content_sha256(path, seal=context)
    assert not context._validated_overlays


def test_returned_metadata_cannot_change_cached_identity(tmp_path: Path) -> None:
    _, roots, _, ref = fixture(tmp_path)
    paths = [p for root in roots for p in storage.shard_paths(root)]
    expected, context = storage.qualified_paths(ref, paths)
    manifest, attrs = storage._open_manifest(paths[0], seal=context)
    manifest['base_seal']['sha256'] = '0' * 64
    attrs['injected'] = True
    again, metadata = storage._open_manifest(paths[0], seal=context)
    assert again['base_seal']['sha256'] != '0' * 64
    assert 'injected' not in metadata
    assert storage.overlay_content_sha256(paths[0], seal=context) == expected[paths[0]]
