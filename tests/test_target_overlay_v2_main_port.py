"""Small schema-2 policy/value overlays exercise main's exact consumer."""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from numcodecs import Pickle

from chess_anti_engine.replay import target_overlay as storage
from chess_anti_engine.replay import target_overlay_v2 as api
from chess_anti_engine.replay.game_epoch import GameAwareEpochBuffer
from chess_anti_engine.replay.shard import (
    ShardMeta, iter_shard_paths, load_shard_arrays, samples_to_arrays,
    save_local_shard_arrays,
)
from chess_anti_engine.train.trainer import _guard_exact_host_overlap
from scripts.lc0_control_train import stage_shards
from scripts.target_overlay_storage import seal_base
from tests.test_game_aware_epoch_replay import _sample


def _base(tmp_path: Path, index: int) -> tuple[Path, Path, dict[str, str]]:
    work = tmp_path / f"base_{index}"
    base = work / "rows"
    base.mkdir(parents=True)
    samples = [_sample(game=2 * index + row, row=row, planes=175)
               for row in range(2)]
    for sample in samples:
        assert sample.legal_mask is not None
        sample.legal_mask[:] = 1
        sample.input_history_encoding = "lc0_root_legacy_meta"
        sample.history_rep_fix = True
        sample.search_wdl = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    shard = base / "shard_000000.zarr"
    save_local_shard_arrays(
        shard, arrs=samples_to_arrays(samples),
        meta=ShardMeta(
            positions=2, policy_encoding="lc0_1858", policy_size=1858,
            input_history_encoding="lc0_root_legacy_meta", history_rep_fix=True,
        ),
    )
    seal_path = work / "base-seal.json"
    seal_base(base, seal_path)
    return base, shard, {"path": str(seal_path), "sha256": storage.sha(seal_path)}


def _overlay(
    tmp_path: Path, index: int,
    replacements: tuple[str, ...] = ("policy_target", "search_wdl"),
    *, base: tuple[Path, Path, dict[str, str]] | None = None,
    roster_ref: dict[str, str] | None = None,
    recipe: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, str], dict[str, str]]:
    _base_root, shard, seal_ref = base or _base(tmp_path, index)
    root = tmp_path / f"overlay_{index}"
    root.mkdir(exist_ok=True)
    target = root / shard.name
    recipe = recipe if recipe is not None else _synthetic_recipe(index)
    if roster_ref is None:
        scopes = [_scope(shard, target, seal_ref, replacements, recipe=recipe)]
        roster_ref = storage.write_target_intent_roster(
            scopes,
            tmp_path / "intents" / f"overlay_{index}.json",
            registration_ref=_registration_ref(tmp_path, f"overlay_{index}", scopes),
        )
    storage.begin_target_shard(
        shard, target, seal_ref, replacements=replacements, roster_ref=roster_ref,
    )
    group: Any = zarr.open_group(str(target), mode="a")
    if "policy_target" in replacements:
        policy = np.zeros((2, 1858), dtype=np.float16)
        policy[:, 0] = 0.75 - 0.125 * index
        policy[:, 1] = 0.25 + 0.125 * index
        group["policy_target"][:] = policy
    if "search_wdl" in replacements:
        group["search_wdl"][:] = np.array(
            [[0.125 + 0.125 * index, 0.25, 0.625 - 0.125 * index]] * 2,
            dtype=np.float16,
        )
    storage.finish_target_shard(
        shard, target, seal_ref, recipe=recipe, roster_ref=roster_ref,
    )
    return root, target, seal_ref, roster_ref


def _synthetic_recipe(index: int) -> dict[str, str]:
    return {"kind": "synthetic-target-v1", "source_sha256": f"{index + 1:064x}"}


def _scope(
    shard: Path, target: Path, seal_ref: dict[str, str],
    replacements: tuple[str, ...] = ("policy_target", "search_wdl"),
    *, recipe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {"base": str(shard), "output": str(target),
            "base_seal": seal_ref, "replacements": list(replacements),
            "recipe": recipe if recipe is not None else _synthetic_recipe(0)}


def _registration_ref(
    tmp_path: Path, name: str, scopes: list[dict[str, Any]],
) -> dict[str, str]:
    path = tmp_path / f"{name}-registration.json"
    intents = [storage._target_intent(
        Path(scope["base"]), Path(scope["output"]), scope["base_seal"],
        sorted(scope["replacements"]),
    ) for scope in scopes]
    storage._atomic_new_json(path, {
        "schema": 2, "kind": "target-overlay-producer-registration",
        "intents": intents, "recipes": [scope["recipe"] for scope in scopes],
    })
    return {"path": str(path), "sha256": storage.sha(path)}


def _single_roster(
    tmp_path: Path, shard: Path, target: Path, seal_ref: dict[str, str],
    replacements: tuple[str, ...], name: str = "target",
    *, recipe: dict[str, Any] | None = None,
) -> dict[str, str]:
    scopes = [_scope(shard, target, seal_ref, replacements, recipe=recipe)]
    return storage.write_target_intent_roster(
        scopes,
        tmp_path / "intents" / f"{name}.json",
        registration_ref=_registration_ref(tmp_path, name, scopes),
    )


def _qualified(tmp_path: Path) -> tuple[list[Path], list[Path], dict[str, str]]:
    bases = [_base(tmp_path, index) for index in range(2)]
    roots = [tmp_path / f"overlay_{index}" for index in range(2)]
    for root in roots:
        root.mkdir()
    targets = [root / base[1].name for root, base in zip(roots, bases, strict=True)]
    scopes = [_scope(base[1], target, base[2], recipe=_synthetic_recipe(index))
              for index, (base, target) in enumerate(zip(bases, targets, strict=True))]
    roster_ref = storage.write_target_intent_roster(
        scopes,
        tmp_path / "intents" / "combined.json",
        registration_ref=_registration_ref(tmp_path, "combined", scopes),
    )
    for index in range(2):
        _overlay(tmp_path, index, base=bases[index], roster_ref=roster_ref)
    receipt = tmp_path / "qualified-targets.json"
    storage.qualify_target_roots(roots, receipt, roster_ref=roster_ref)
    return roots, targets, {"path": str(receipt), "sha256": storage.sha(receipt)}


def test_schema2_api_exports_shared_qualified_implementation() -> None:
    assert api.BaseSeals is storage.BaseSeals
    assert api.begin_target_shard is storage.begin_target_shard
    assert api.finish_target_shard is storage.finish_target_shard
    assert api.qualify_target_roots is storage.qualify_target_roots
    assert api.write_target_intent_roster is storage.write_target_intent_roster


@pytest.mark.parametrize("rewrite_request", [False, True])
def test_schema2_original_intent_refuses_missing_requested_value(
    tmp_path: Path, rewrite_request: bool,
) -> None:
    _base_root, shard, seal_ref = _base(tmp_path, 0)
    root = tmp_path / "overlay"
    root.mkdir()
    target = root / shard.name
    roster_ref = _single_roster(
        tmp_path, shard, target, seal_ref, ("policy_target", "search_wdl"),
    )
    storage.begin_target_shard(
        shard, target, seal_ref, replacements=("policy_target", "search_wdl"),
        roster_ref=roster_ref,
    )
    group: Any = zarr.open_group(str(target), mode="a")
    policy = np.zeros((2, 1858), dtype=np.float16)
    policy[:, 0] = 1
    group["policy_target"][:] = policy
    shutil.rmtree(target / "search_wdl")
    if rewrite_request:
        # A writer may rewrite its stage files and supply a new policy-only
        # ticket. Qualification still compares with the original pinned intent.
        (target / "target_overlay_request.json").write_text(
            json.dumps({"replacements": ["policy_target"]}),
        )
        forged_path = tmp_path / "intents" / "forged.json"
        forged = json.loads(Path(roster_ref["path"]).read_text())
        forged["intents"][0]["replacements"] = ["policy_target"]
        storage._atomic_new_json(forged_path, forged)
        forged_ref = {"path": str(forged_path), "sha256": storage.sha(forged_path)}
        with pytest.raises(ValueError, match="pinned producer registration"):
            storage.finish_target_shard(
                shard, target, seal_ref, recipe={"kind": "synthetic"},
                roster_ref=forged_ref,
            )
        assert not (target / storage.MANIFEST).exists()
        registration_path = tmp_path / "forged-registration.json"
        registration = json.loads(
            Path(forged["producer_registration"]["path"]).read_text(),
        )
        registration["intents"][0]["replacements"] = ["policy_target"]
        storage._atomic_new_json(registration_path, registration)
        forged["producer_registration"] = {
            "path": str(registration_path), "sha256": storage.sha(registration_path),
        }
        alternate_path = tmp_path / "intents" / "alternate.json"
        storage._atomic_new_json(alternate_path, forged)
        alternate_ref = {"path": str(alternate_path),
                         "sha256": storage.sha(alternate_path)}
        storage.finish_target_shard(
            shard, target, seal_ref, recipe=_synthetic_recipe(0),
            roster_ref=alternate_ref,
        )
        with pytest.raises(ValueError, match="authoritative qualification"):
            storage.qualify_target_roots(
                [root], tmp_path / "qualified.json", roster_ref=roster_ref,
            )
    else:
        with pytest.raises(ValueError, match="requested replacements"):
            storage.finish_target_shard(
                shard, target, seal_ref, recipe=_synthetic_recipe(0),
                roster_ref=roster_ref,
            )
        assert not (target / storage.MANIFEST).exists()
        with pytest.raises(FileNotFoundError):
            storage.qualify_target_roots(
                [root], tmp_path / "qualified.json", roster_ref=roster_ref,
            )
    assert not (tmp_path / "qualified.json").exists()


def test_schema2_pinned_intent_survives_restart_and_scopes_begin(tmp_path: Path) -> None:
    _base_root, shard, seal_ref = _base(tmp_path, 0)
    target = tmp_path / "overlay" / shard.name
    roster_ref = _single_roster(
        tmp_path, shard, target, seal_ref, ("policy_target", "search_wdl"),
    )
    # A new process needs only the caller's retained path/SHA pin.
    retained_ref = json.loads(json.dumps(roster_ref))
    target.parent.mkdir()
    with pytest.raises(ValueError, match="requested replacements"):
        storage.begin_target_shard(
            shard, target, seal_ref, replacements=("policy_target",),
            roster_ref=retained_ref,
        )
    assert not target.exists()
    storage.begin_target_shard(
        shard, target, seal_ref, replacements=("policy_target", "search_wdl"),
        roster_ref=retained_ref,
    )
    assert set(zarr.open_group(str(target), mode="r").array_keys()) == {
        "policy_target", "search_wdl",
    }


@pytest.mark.parametrize("drift", [pytest.param(True, id="bool"),
                                  pytest.param(1.0, id="float")])
def test_schema2_roster_refuses_numeric_recipe_type_drift(
    tmp_path: Path, drift: bool | float,
) -> None:
    _base_root, shard, seal_ref = _base(tmp_path, 0)
    root = tmp_path / "overlay"
    root.mkdir()
    recipe = {**_synthetic_recipe(0), "semantic_gain": 1}
    scopes = [_scope(shard, root / shard.name, seal_ref, recipe=recipe)]
    registration_ref = _registration_ref(tmp_path, "typed", scopes)
    scopes[0]["recipe"] = {**recipe, "semantic_gain": drift}
    roster_path = tmp_path / "intents" / "typed.json"
    with pytest.raises(ValueError, match="pinned producer registration"):
        storage.write_target_intent_roster(
            scopes, roster_path, registration_ref=registration_ref,
        )
    assert not roster_path.exists()


@pytest.mark.parametrize("mutation", ["wrong_kind", "extra_key", "bool", "float"])
def test_schema2_finish_refuses_recipe_different_from_registered_plan(
    tmp_path: Path, mutation: str,
) -> None:
    _base_root, shard, seal_ref = _base(tmp_path, 0)
    root = tmp_path / "overlay"
    root.mkdir()
    target = root / shard.name
    registered_recipe = {**_synthetic_recipe(0), "semantic_gain": 1}
    roster_ref = _single_roster(
        tmp_path, shard, target, seal_ref, ("policy_target", "search_wdl"),
        recipe=registered_recipe,
    )
    storage.begin_target_shard(
        shard, target, seal_ref, replacements=("policy_target", "search_wdl"),
        roster_ref=roster_ref,
    )
    base: Any = zarr.open_group(str(shard), mode="r")
    local: Any = zarr.open_group(str(target), mode="a")
    for name in ("policy_target", "search_wdl"):
        local[name][:] = base[name][:]
    recipe: dict[str, Any] = dict(registered_recipe)
    if mutation == "wrong_kind":
        recipe["kind"] = "different-recipe"
    elif mutation == "extra_key":
        recipe["unexpected_weight"] = 0.9
    elif mutation == "bool":
        recipe["semantic_gain"] = True
    else:
        recipe["semantic_gain"] = 1.0
    with pytest.raises(ValueError, match="recipe differs from pinned producer"):
        storage.finish_target_shard(
            shard, target, seal_ref, recipe=recipe,
            roster_ref=roster_ref,
        )
    assert not (target / storage.MANIFEST).exists()


@pytest.mark.parametrize("drift", [pytest.param(True, id="bool"),
                                  pytest.param(1.0, id="float")])
def test_schema2_qualification_refuses_numeric_recipe_type_drift(
    tmp_path: Path, drift: bool | float,
) -> None:
    registered_recipe = {**_synthetic_recipe(0), "semantic_gain": 1}
    root, target, _seal_ref, roster_ref = _overlay(
        tmp_path, 0, recipe=registered_recipe,
    )
    manifest_path = target / storage.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest["recipe"]["semantic_gain"] = drift
    manifest_path.write_text(json.dumps(manifest))
    receipt = tmp_path / "qualified.json"
    with pytest.raises(ValueError, match="recipe differs from pinned producer"):
        storage.qualify_target_roots([root], receipt, roster_ref=roster_ref)
    assert not receipt.exists()


@pytest.mark.parametrize("inside", ["overlay", "base"])
def test_schema2_begin_intent_stays_outside_mutable_roots(
    tmp_path: Path, inside: str,
) -> None:
    base_root, shard, seal_ref = _base(tmp_path, 0)
    root = tmp_path / "overlay"
    root.mkdir()
    roster_path = (root if inside == "overlay" else base_root) / "intent.json"
    scopes = [_scope(shard, root / shard.name, seal_ref)]
    with pytest.raises(ValueError, match="outside overlay and base roots"):
        storage.write_target_intent_roster(
            scopes, roster_path,
            registration_ref=_registration_ref(tmp_path, f"boundary-{inside}", scopes),
        )
    assert not roster_path.exists()


@pytest.mark.parametrize("phase", ["finish", "qualify"])
def test_schema2_rejects_pickle_filter_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    if phase == "finish":
        _base_root, shard, seal_ref = _base(tmp_path, 0)
        root = tmp_path / "overlay"
        root.mkdir()
        target = root / shard.name
        roster_ref = _single_roster(
            tmp_path, shard, target, seal_ref, ("policy_target",),
        )
        storage.begin_target_shard(
            shard, target, seal_ref, replacements=("policy_target",),
            roster_ref=roster_ref,
        )
    else:
        root, target, seal_ref, roster_ref = _overlay(tmp_path, 0, ("policy_target",))
        shard = Path(json.loads((target / storage.MANIFEST).read_text())["base"])

    shutil.rmtree(target / "policy_target")
    group: Any = zarr.open_group(str(target), mode="a")
    original: Any = zarr.open_group(str(shard), mode="r")["policy_target"]
    policy = np.zeros(original.shape, dtype=original.dtype)
    policy[:, 0] = 1
    replacement = group.create_dataset(
        "policy_target", shape=policy.shape, dtype=policy.dtype,
        chunks=original.chunks, filters=[Pickle()], compressor=None,
    )
    replacement[:] = policy
    if phase == "qualify":
        manifest_path = target / storage.MANIFEST
        manifest = json.loads(manifest_path.read_text())
        manifest["target_content_sha256"]["policy_target"] = storage._plain_content(
            target / "policy_target",
        )
        manifest_path.write_text(json.dumps(manifest))

    def refuse_decode(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Pickle decoded before the metadata-only codec guard")

    monkeypatch.setattr(Pickle, "decode", refuse_decode)
    if phase == "finish":
        with pytest.raises(ValueError, match="filters"):
            storage.finish_target_shard(
                shard, target, seal_ref, recipe=_synthetic_recipe(0),
                roster_ref=roster_ref,
            )
    else:
        with pytest.raises(ValueError, match="filters"):
            storage.qualify_target_roots(
                [root], tmp_path / "qualified.json", roster_ref=roster_ref,
            )
    if phase == "finish":
        assert not (target / storage.MANIFEST).exists()
    else:
        assert not (tmp_path / "qualified.json").exists()


@pytest.mark.parametrize("replacements", [
    ("policy_target",), ("search_wdl",), ("policy_target", "search_wdl"),
])
def test_schema2_exact_replacements_and_inherited_bytes(
    tmp_path: Path, replacements: tuple[str, ...],
) -> None:
    base_root, base_shard, base_ref = _base(tmp_path, 0)
    root, target, _, roster_ref = _overlay(
        tmp_path, 0, replacements, base=(base_root, base_shard, base_ref),
    )
    receipt = tmp_path / "qualified.json"
    storage.qualify_target_roots([root], receipt, roster_ref=roster_ref)
    ref = {"path": str(receipt), "sha256": storage.sha(receipt)}
    qualified, seal = storage.qualified_paths(ref, [target])
    assert list(qualified) == [target]
    actual, _meta = load_shard_arrays(
        target, lazy=False, allow_target_overlay=True, overlay_seal=seal,
    )
    original, _meta = load_shard_arrays(base_shard, lazy=False)
    for field in ("x", "legal_mask", "game_id", "ply_index"):
        assert np.array_equal(actual[field], original[field]), field
    for field in ("policy_target", "search_wdl"):
        if field in replacements:
            expected = np.asarray(zarr.open_group(str(target), mode="r")[field])
        else:
            expected = original[field]
        assert np.array_equal(actual[field], expected), field
    if "policy_target" in replacements:
        assert np.array_equal(actual["policy_target"][:, :2],
                              np.array([[0.75, 0.25]] * 2, dtype=np.float16))
        assert not np.asarray(actual["policy_target"][:, 2:]).any()
    if "search_wdl" in replacements:
        assert np.array_equal(actual["search_wdl"],
                              np.array([[0.125, 0.25, 0.625]] * 2,
                                       dtype=np.float16))


def test_schema1_policy_only_receipt_remains_accepted(tmp_path: Path) -> None:
    _base_root, base_shard, seal_ref = _base(tmp_path, 0)
    root = tmp_path / "schema1-policy"
    root.mkdir()
    target = root / base_shard.name
    storage.begin_policy_shard(base_shard, target, seal_ref)
    group: Any = zarr.open_group(str(target), mode="a")
    policy = np.zeros((2, 1858), dtype=np.float16)
    policy[:, 0] = 1
    group["policy_target"][:] = policy
    storage.finish_policy_shard(base_shard, target, seal_ref)
    receipt = tmp_path / "schema1-qualified.json"
    storage._atomic_new_json(receipt, {
        "schema": 1, "status": storage.OVERLAY_STATUS,
        "root": str(root), "base_seal": seal_ref,
        "root_files": storage._root_files(root),
        "shards": [{"name": target.name,
                    "content_sha256": storage.overlay_content_sha256(target)}],
    })
    ref = {"path": str(receipt), "sha256": storage.sha(receipt)}
    qualified, seal = storage.qualified_paths(ref, [target])
    assert list(qualified) == [target]
    assert type(seal) is storage.BaseSeal
    with pytest.raises(ValueError, match="paths/order"):
        storage.qualified_paths(ref, [target, target])


def _synthetic_legacy_e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
    mutation: str = "",
) -> tuple[Path, dict[str, str]]:
    replacements = (("policy_target",) if mutation == "missing_wdl"
                    else ("policy_target", "search_wdl"))
    root, target, seal_ref, _roster_ref = _overlay(
        tmp_path, 0, replacements,
    )
    manifest_path = target / storage.MANIFEST
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("intent_roster")
    recipe = {"kind": "factorial58-sffree-v1", "arm": "E",
              "source_sha256": "1" * 64}
    manifest["recipe"] = dict(recipe)
    if mutation == "recipe":
        manifest["recipe"]["source_sha256"] = "2" * 64
    manifest_path.write_text(json.dumps(manifest))
    context = storage.BaseSeals([seal_ref], legacy_recipes={
        str(root): {"recipe": recipe, "base": str(Path(manifest["base"]).parent)},
    })
    content = (storage.overlay_content_sha256(target, seal=context)
               if not mutation else "0" * 64)
    receipt_path = tmp_path / "legacy-qualified.json"
    storage._atomic_new_json(receipt_path, {
        "schema": 2, "status": storage.TARGET_OVERLAY_STATUS,
        "roots": [str(root)], "root_files": {str(root): storage._root_files(root)},
        "base_seals": [seal_ref],
        "shards": [{"path": str(target), "content_sha256": content, "rows": 2}],
        "rows": 2, "scope": "Synthetic legacy fixture",
    })
    qualified_ref = {"path": str(receipt_path), "sha256": storage.sha(receipt_path)}

    def pinned(name: str, value: dict[str, Any]) -> dict[str, str]:
        path = tmp_path / name
        storage._atomic_new_json(path, value)
        return {"path": str(path), "sha256": storage.sha(path)}

    manifests_ref = pinned("legacy-manifests.json", {
        "status": "PREPARED_NOT_EXECUTED", "rows": 2, "shards": 1,
        "cohorts": [{"rows": 2, "shards": 1}],
    })
    plan_ref = pinned("legacy-plan.json", {
        "status": "READY_REVIEWED_FROZEN", "manifests": manifests_ref,
    })
    cohort_ref = pinned("legacy-cohort.json", {
        "status": "COMPLETE_SFFREE_TARGET_COHORT", "root": str(root),
        "base": str(Path(manifest["base"]).parent), "recipe": recipe,
        "rows": 2, "shards": 1,
    })
    complete_ref = pinned("legacy-complete.json", {
        "status": "COMPLETE_SFFREE_35_COHORTS", "qualified": qualified_ref,
        "plan": plan_ref, "cohorts": [cohort_ref], "rows": 2, "shards": 1,
    })
    monkeypatch.setattr(storage, "LEGACY_E_QUALIFIED_SHA256", qualified_ref["sha256"])
    monkeypatch.setattr(storage, "LEGACY_E_COMPLETE_REF", complete_ref)
    monkeypatch.setattr(storage, "LEGACY_E_MANIFESTS_SHA256", manifests_ref["sha256"])
    monkeypatch.setattr(storage, "LEGACY_E_COHORT_COUNT", 1)
    return target, qualified_ref


def test_frozen_e_pre_roster_accepts_both_targets_only_under_pinned_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, ref = _synthetic_legacy_e(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="exact pinned receipt"):
        storage.verify_receipt(storage._read_pin(ref))
    qualified, context = storage.qualified_paths(ref, [target])
    assert list(qualified) == [target]
    actual, _meta = load_shard_arrays(
        target, lazy=False, allow_target_overlay=True, overlay_seal=context,
    )
    assert np.array_equal(actual["search_wdl"],
                          np.array([[0.125, 0.25, 0.625]] * 2, dtype=np.float16))
    assert np.array_equal(actual["policy_target"][:, :2],
                          np.array([[0.75, 0.25]] * 2, dtype=np.float16))
    with pytest.raises(ValueError, match="frozen E qualification"):
        load_shard_arrays(target, lazy=False, allow_target_overlay=True)


@pytest.mark.parametrize(("mutation", "message"), [
    ("missing_wdl", "replace policy and value"),
    ("recipe", "recipe differs"),
])
def test_frozen_e_pre_roster_refuses_missing_value_or_changed_recipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    mutation: str, message: str,
) -> None:
    target, ref = _synthetic_legacy_e(tmp_path, monkeypatch, mutation=mutation)
    with pytest.raises(ValueError, match=message):
        storage.qualified_paths(ref, [target])


def test_multi_root_order_and_base_seal_identity(tmp_path: Path) -> None:
    roots, targets, ref = _qualified(tmp_path)
    staged = tmp_path / "staged"
    assert stage_shards(roots, staged) == 2
    staged_paths = iter_shard_paths(staged)
    qualified, context = storage.qualified_paths(ref, staged_paths)
    assert list(qualified) == targets
    assert isinstance(context, storage.BaseSeals)
    assert len(context.contexts) == 2
    for index, target in enumerate(targets):
        arrays, _meta = load_shard_arrays(
            target, lazy=False, allow_target_overlay=True, overlay_seal=context,
        )
        assert np.array_equal(
            arrays["search_wdl"],
            np.array([[0.125 + 0.125 * index, 0.25,
                       0.625 - 0.125 * index]] * 2, dtype=np.float16),
        )
        assert np.array_equal(arrays["policy_target"][:, :2],
                              np.array([[0.75 - 0.125 * index,
                                         0.25 + 0.125 * index]] * 2,
                                       dtype=np.float16))
    with pytest.raises(ValueError, match="paths/order"):
        storage.qualified_paths(ref, staged_paths[::-1])
    first_manifest_path = targets[0] / storage.MANIFEST
    first_manifest = json.loads(first_manifest_path.read_text())
    first_manifest["base_seal"] = json.loads(
        (targets[1] / storage.MANIFEST).read_text())["base_seal"]
    first_manifest_path.write_text(json.dumps(first_manifest))
    with pytest.raises(ValueError, match=r"base|seal|changed|intent"):
        storage.qualified_paths(ref, staged_paths)


@pytest.mark.parametrize("mutation", ["policy", "value", "recipe", "extra_recipe",
                                      "seal", "intent"])
def test_qualified_provenance_and_storage_mutations_refuse(
    tmp_path: Path, mutation: str,
) -> None:
    _roots, targets, ref = _qualified(tmp_path)
    if mutation in ("policy", "value"):
        group: Any = zarr.open_group(str(targets[0]), mode="a")
        field = "policy_target" if mutation == "policy" else "search_wdl"
        group[field][0, 0] = 0.5
    elif mutation == "recipe":
        manifest_path = targets[0] / storage.MANIFEST
        manifest = json.loads(manifest_path.read_text())
        manifest["recipe"]["source_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "extra_recipe":
        manifest_path = targets[0] / storage.MANIFEST
        manifest = json.loads(manifest_path.read_text())
        manifest["recipe"]["semantic_gain"] = 0.9
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "seal":
        manifest = json.loads((targets[0] / storage.MANIFEST).read_text())
        seal_path = Path(manifest["base_seal"]["path"])
        seal_path.write_bytes(seal_path.read_bytes() + b" ")
    else:
        manifest = json.loads((targets[0] / storage.MANIFEST).read_text())
        roster_path = Path(manifest["intent_roster"]["path"])
        roster_path.write_bytes(roster_path.read_bytes() + b" ")
    with pytest.raises(ValueError, match=r"changed|differs|target|base|SHA"):
        storage.qualified_paths(ref, targets)


def test_duplicate_base_rows_across_roots_refuse(tmp_path: Path) -> None:
    base = _base(tmp_path, 0)
    roots = [tmp_path / "overlay_0", tmp_path / "overlay_1"]
    for root in roots:
        root.mkdir()
    scopes = [_scope(base[1], root / base[1].name, base[2],
                     recipe=_synthetic_recipe(index))
              for index, root in enumerate(roots)]
    roster_ref = storage.write_target_intent_roster(
        scopes,
        tmp_path / "intents" / "duplicate.json",
        registration_ref=_registration_ref(tmp_path, "duplicate", scopes),
    )
    first, _target, _, _ = _overlay(tmp_path, 0, base=base, roster_ref=roster_ref)
    second, _target, _, _ = _overlay(tmp_path, 1, base=base, roster_ref=roster_ref)
    with pytest.raises(ValueError, match="duplicate base rows"):
        storage.qualify_target_roots(
            [first, second], tmp_path / "duplicate.json",
            roster_ref=roster_ref,
        )


def test_schema2_host_overlap_delivery_guard(tmp_path: Path) -> None:
    roots, _targets, ref = _qualified(tmp_path)
    staged = tmp_path / "staged"
    stage_shards(roots, staged)
    buf = GameAwareEpochBuffer(
        shard_dir=staged, overlay_storage_qualification=ref,
        batch_size=2, seed=121, input_planes=175,
        input_history_encoding="lc0_root_legacy_meta", history_rep_fix=True,
        mirror_augmentation=False, plan_workers=2, load_workers=2,
        host_batch_overlap=True,
    )
    try:
        def source() -> Any:
            for _ in range(buf.num_batches):
                yield buf.sample_batch_arrays(2)

        batches = list(_guard_exact_host_overlap(source(), buf, buf.num_batches))
        assert len(batches) == buf.num_batches
        receipt = buf.receipt()
        assert receipt["complete"] is True
        assert receipt["delivered_batches"] == buf.num_batches
        assert receipt["overlap_aborted"] is False
    finally:
        buf.close()


def test_schema2_exact_epoch_delivers_target_bytes_by_row(tmp_path: Path) -> None:
    roots, _targets, ref = _qualified(tmp_path)
    staged = tmp_path / "staged"
    stage_shards(roots, staged)
    buf = GameAwareEpochBuffer(
        shard_dir=staged, overlay_storage_qualification=ref,
        batch_size=2, seed=121, input_planes=175,
        input_history_encoding="lc0_root_legacy_meta", history_rep_fix=True,
        mirror_augmentation=False, plan_workers=2, load_workers=2,
    )
    try:
        delivered: list[int] = []
        for _ in range(buf.num_batches):
            batch = buf.sample_batch_arrays(2)
            for row, game in enumerate(np.asarray(batch["game_id"], dtype=np.int64)):
                root_index = int(game) // 2
                delivered.append(int(game))
                expected_policy = np.zeros((1858,), dtype=np.float16)
                expected_policy[:2] = [0.75 - 0.125 * root_index,
                                       0.25 + 0.125 * root_index]
                expected_wdl = np.array(
                    [0.125 + 0.125 * root_index, 0.25,
                     0.625 - 0.125 * root_index], dtype=np.float16,
                )
                assert np.array_equal(batch["policy_target"][row], expected_policy)
                assert np.array_equal(batch["search_wdl"][row], expected_wdl)
        assert sorted(delivered) == [0, 1, 2, 3]
        assert buf.receipt()["complete"] is True
    finally:
        buf.close()


def test_schema2_host_overlap_early_close_poisoned(tmp_path: Path) -> None:
    roots, _targets, ref = _qualified(tmp_path)
    staged = tmp_path / "staged"
    stage_shards(roots, staged)
    buf = GameAwareEpochBuffer(
        shard_dir=staged, overlay_storage_qualification=ref,
        batch_size=2, seed=121, input_planes=175,
        input_history_encoding="lc0_root_legacy_meta", history_rep_fix=True,
        mirror_augmentation=False, plan_workers=2, load_workers=2,
        host_batch_overlap=True,
    )
    try:
        def source() -> Any:
            for _ in range(buf.num_batches):
                yield buf.sample_batch_arrays(2)

        guarded = _guard_exact_host_overlap(source(), buf, buf.num_batches)
        next(guarded)
        guarded.close()
        receipt = buf.receipt()
        assert receipt["complete"] is False
        assert receipt["overlap_aborted"] is True
        with pytest.raises(RuntimeError, match="aborted"):
            buf.begin_overlap()
    finally:
        buf.close()


@pytest.mark.parametrize("failure", ["first_dir_fsync", "after_link", "temp_unlink",
                                     "readback_corruption"])
def test_receipt_post_link_failure_rolls_back_named_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    receipt = tmp_path / "qualified-targets.json"
    original_fsync = os.fsync
    original_link = os.link
    original_unlink = os.unlink
    failed = False

    if failure == "first_dir_fsync":
        def fsync(fd: int) -> None:
            nonlocal failed
            if not failed and stat.S_ISDIR(os.fstat(fd).st_mode):
                failed = True
                raise OSError("injected directory fsync failure")
            original_fsync(fd)

        monkeypatch.setattr(storage.os, "fsync", fsync)
    elif failure == "after_link":
        def link(*args: Any, **kwargs: Any) -> None:
            nonlocal failed
            original_link(*args, **kwargs)
            failed = True
            raise OSError("injected error after link")

        monkeypatch.setattr(storage.os, "link", link)
    elif failure == "temp_unlink":
        def unlink(path: str | os.PathLike[str], **kwargs: Any) -> None:
            nonlocal failed
            if not failed and str(path).endswith(".writing"):
                failed = True
                raise OSError("injected temp unlink failure")
            original_unlink(path, **kwargs)

        monkeypatch.setattr(storage.os, "unlink", unlink)
    else:
        def fsync(fd: int) -> None:
            nonlocal failed
            # After the second directory fsync the final name is installed
            # and the temp name gone; corrupt the same inode before readback.
            if (stat.S_ISDIR(os.fstat(fd).st_mode) and not failed
                    and not any(tmp_path.glob("*.writing"))):
                receipt.write_bytes(b"corrupted")
                failed = True
            original_fsync(fd)

        monkeypatch.setattr(storage.os, "fsync", fsync)

    with pytest.raises((OSError, ValueError)):
        storage._atomic_new_json(receipt, {"schema": 2, "status": "synthetic"})
    assert failed
    assert not receipt.exists()
    assert not list(tmp_path.glob("*.writing"))


def test_receipt_rollback_failure_reports_ambiguous_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "qualified-targets.json"
    original_fsync = os.fsync
    original_unlink = os.unlink

    def fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("injected directory fsync failure")
        original_fsync(fd)

    def unlink(path: str | os.PathLike[str], **kwargs: Any) -> None:
        if str(path) == receipt.name:
            raise OSError("injected revocation failure")
        original_unlink(path, **kwargs)

    monkeypatch.setattr(storage.os, "fsync", fsync)
    monkeypatch.setattr(storage.os, "unlink", unlink)
    with pytest.raises(RuntimeError, match="AMBIGUOUS_STORAGE_RECEIPT_AUTHORITY"):
        storage._atomic_new_json(receipt, {"schema": 2, "status": "synthetic"})
    assert receipt.is_file()


def test_receipt_publisher_refuses_replacement_and_readbacks_exact_bytes(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "qualified-targets.json"
    value = {"schema": 2, "status": "synthetic"}
    storage._atomic_new_json(receipt, value)
    published = receipt.read_bytes()
    assert json.loads(published) == value
    assert storage.sha(receipt)
    with pytest.raises(ValueError, match="existing storage receipt"):
        storage._atomic_new_json(receipt, {"schema": 2, "status": "changed"})
    assert receipt.read_bytes() == published
