"""Actual seed export and warmstart readers must not mix donor generations."""
from __future__ import annotations

import argparse
import errno
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr

from chess_anti_engine.tune import salvage, trainable_init
from chess_anti_engine.tune import recovery
from chess_anti_engine.tune.trial_config import RestoreResult
from chess_anti_engine.utils import atomic
from tests.script_loading import load_script_module


def _donor(root: Path, name: str, *, shards: tuple[int, ...], sidecars: bool) -> Path:
    td = root / "tune" / "train_trial_abc12_00000_0_lr=0.1_2026-01-01_00-00-00"
    ckpt = td / "checkpoint_000001"
    ckpt.mkdir(parents=True)
    (ckpt / "trainer.pt").write_text(name)
    (ckpt / "trial_meta.json").write_text(json.dumps({"donor": name, "global_iter": 2}))
    if sidecars:
        for fn in ("pid_state.json", "rng_state.json"):
            (ckpt / fn).write_text(json.dumps({"donor": name}))
        for fn in ("gate_state.json", "best.json"):
            (td / fn).write_text(json.dumps({"donor": name}))
    (td / "result.json").write_text(json.dumps({
        "training_iteration": 2, "checkpoint_dir_name": ckpt.name,
    }) + "\n")
    if shards:
        replay = td / "replay_shards"
        replay.mkdir()
        for index in shards:
            group = zarr.open_group(str(replay / f"shard_{index:06d}.zarr"), mode="w")
            group.create_dataset("donor", data=np.array([name]))
    return td


def _export(root: Path, pool: Path, *, copy_replay: bool, top_n: int = 1) -> dict[str, Any]:
    salvage.export_seed_pool(argparse.Namespace(
        work_dir=str(root), salvage_out_dir=str(pool), salvage_source_run_id="abc12",
        salvage_metric="training_iteration", salvage_top_n=top_n, num_samples=top_n,
        salvage_copy_replay=copy_replay, salvage_dry_run=False,
        tune_replay_root_override="",
    ))
    return json.loads((pool / "manifest.json").read_text())


def _warmstart(
    pool: Path, dest: Path, monkeypatch: pytest.MonkeyPatch, *, donor_config: bool = False,
) -> dict[str, Any]:
    loaded: list[str] = []
    lr_calls: list[tuple[float, bool]] = []

    class InertTrainer:
        def load(self, path: Path) -> None:
            loaded.append(path.read_text())

        def set_peak_lr(self, value: float, *, rescale_current: bool) -> None:
            lr_calls.append((value, rescale_current))

    monkeypatch.setattr(trainable_init, "guard_warm_start_lr", lambda *_a, **_kw: None)
    restored = RestoreResult()
    config: dict[str, Any] = {
        "salvage_seed_pool_dir": str(pool), "salvage_restore_full_trainer_state": True,
        "salvage_restore_pid_state": True, "salvage_restore_donor_config": donor_config,
    }
    rng = trainable_init._restore_from_salvage_pool(
        config=config,
        trainer=InertTrainer(), device="cpu", trial_id="new", trial_dir=dest,
        rr=restored,
    )
    replay = dest / "replay_shards"
    trainable_init._seed_replay_from_warmstart(restore=restored, replay_shard_dir=replay)
    values = []
    for p in sorted(replay.glob("*.zarr")):
        values.extend(np.asarray(zarr.open_group(str(p), mode="r")["donor"]).tolist())
    return {"weights": loaded, "replay": values, "rng": rng,
            "pid": restored.restored_pid_state, "gate": restored.restored_gate_state,
            "config_lr": config.get("lr"), "trainer_lr": lr_calls}


def _snapshot(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize(("copy_replay", "b_shards", "expected"), [
    (False, (), []), (True, (2,), ["B"]), (True, (1,), ["B"]),
])
def test_reexport_uses_only_current_donor_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    copy_replay: bool, b_shards: tuple[int, ...], expected: list[str],
) -> None:
    _donor(tmp_path / "a", "A", shards=(1,), sidecars=True)
    _donor(tmp_path / "b", "B", shards=b_shards, sidecars=False)
    pool = tmp_path / "pool"
    _export(tmp_path / "a", pool, copy_replay=True)
    manifest = _export(tmp_path / "b", pool, copy_replay=copy_replay)
    observed = _warmstart(pool, tmp_path / "restored", monkeypatch)
    assert observed["weights"] == ["B"]
    assert manifest["entries"][0]["copied_replay_shards"] == len(expected)
    assert observed["replay"] == expected


@pytest.mark.parametrize("field", ["rng", "pid", "gate"])
def test_reexport_does_not_restore_missing_new_donor_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    _donor(tmp_path / "a", "A", shards=(), sidecars=True)
    _donor(tmp_path / "b", "B", shards=(), sidecars=False)
    pool = tmp_path / "pool"
    _export(tmp_path / "a", pool, copy_replay=False)
    _export(tmp_path / "b", pool, copy_replay=False)
    observed = _warmstart(pool, tmp_path / "restored", monkeypatch)
    assert observed["weights"] == ["B"]
    assert observed[field] is None


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt, SystemExit])
def test_failed_reexport_preserves_previous_pool_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException],
) -> None:
    _donor(tmp_path / "a", "A", shards=(1,), sidecars=True)
    b = _donor(tmp_path / "b", "B", shards=(2,), sidecars=True)
    pool = tmp_path / "pool"
    _export(tmp_path / "a", pool, copy_replay=True)
    before = _snapshot(pool)
    copy = salvage.shutil.copy2

    def fail_copy(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(src) == b / "checkpoint_000001" / "rng_state.json":
            raise error_type("injected sidecar copy failure")
        return copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(salvage.shutil, "copy2", fail_copy)
    if error_type is OSError:
        with pytest.raises(RuntimeError, match="existing manifest and seed artifacts"):
            _export(tmp_path / "b", pool, copy_replay=True)
    else:
        with pytest.raises(error_type, match="injected sidecar copy failure"):
            _export(tmp_path / "b", pool, copy_replay=True)
    assert {name: (pool / name).read_bytes() for name in before} == before
    assert _warmstart(pool, tmp_path / "restored", monkeypatch)["weights"] == ["A"]


def test_fresh_no_replay_export_has_no_replay_or_foreign_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _donor(tmp_path / "b", "B", shards=(), sidecars=False)
    pool = tmp_path / "pool"
    _export(tmp_path / "b", pool, copy_replay=False)
    assert _warmstart(pool, tmp_path / "restored", monkeypatch) == {
        "weights": ["B"], "replay": [], "rng": None, "pid": None, "gate": None,
        "config_lr": None, "trainer_lr": [],
    }


def test_restore_binds_selected_weights_and_pid_row_to_one_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _donor(tmp_path / "a", "A", shards=(), sidecars=True)
    b = _donor(tmp_path / "b", "B", shards=(), sidecars=True)
    for td, nodes in ((a, 111), (b, 222)):
        row = json.loads((td / "result.json").read_text())
        row["sf_nodes_next"] = nodes
        row["config"] = {"lr": nodes / 1000}
        (td / "result.json").write_text(json.dumps(row) + "\n")
    pool = tmp_path / "pool"
    _export(tmp_path / "a", pool, copy_replay=False)
    b_pool = pool / "generation_b"
    b_manifest = _export(tmp_path / "b", b_pool, copy_replay=False)
    b_manifest["entries"][0]["seed_dir"] = str(Path("generation_b") / b_manifest["entries"][0]["seed_dir"])
    select = trainable_init._select_salvage_seed_slot

    def publish_after_selection(**kwargs: Any) -> Any:
        selected = select(**kwargs)
        (pool / "manifest.json").write_text(json.dumps(b_manifest))
        return selected

    monkeypatch.setattr(trainable_init, "_select_salvage_seed_slot", publish_after_selection)
    observed = _warmstart(pool, tmp_path / "restored", monkeypatch, donor_config=True)
    assert observed["weights"] == ["A"]
    assert observed["pid"]["nodes"] == 111
    assert observed["config_lr"] == 0.111
    assert observed["trainer_lr"] == [(0.111, False)]


@pytest.mark.parametrize("replacement", [False, True])
def test_first_export_can_publish_partial_but_replacement_cannot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: bool,
) -> None:
    a = _donor(tmp_path / "work", "A", shards=(), sidecars=True)
    b = _donor(tmp_path / "other", "B", shards=(), sidecars=True)
    b = b.rename(a.parent / b.name.replace("00000", "00001", 1))
    pool = tmp_path / "pool"
    before = None
    if replacement:
        _export(tmp_path / "work", pool, copy_replay=False, top_n=2)
        before = _snapshot(pool)
    copy = salvage.shutil.copy2

    def fail_b(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(src) == b / "checkpoint_000001" / "rng_state.json":
            raise OSError("failed B copy")
        return copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(salvage.shutil, "copy2", fail_b)
    if replacement:
        with pytest.raises(RuntimeError, match="existing manifest and seed artifacts"):
            _export(tmp_path / "work", pool, copy_replay=False, top_n=2)
        assert before is not None
        assert {name: (pool / name).read_bytes() for name in before} == before
    else:
        manifest = _export(tmp_path / "work", pool, copy_replay=False, top_n=2)
        assert manifest["top_n"] == 1
        assert len(manifest["failed_slots"]) == 1
        assert _warmstart(pool, tmp_path / "restored", monkeypatch)["weights"] == ["A"]


@pytest.mark.parametrize("stage", ["write", "file-fsync", "replace", "dir-fsync"])
def test_manifest_publication_observes_real_commit_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], stage: str,
) -> None:
    _donor(tmp_path / "a", "A", shards=(1,), sidecars=True)
    _donor(tmp_path / "b", "B", shards=(2,), sidecars=True)
    pool = tmp_path / "pool"
    _export(tmp_path / "a", pool, copy_replay=True)
    before = _snapshot(pool)
    events: list[str] = []
    write = Path.write_text
    file_sync, dir_sync, replace = atomic._fsync_file, atomic._fsync_dir, atomic.os.replace

    def writing(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.parent == pool and path.name.startswith("manifest.json.tmp."):
            events.append("write")
            if stage == "write":
                raise OSError("injected publication write failure")
        return write(path, *args, **kwargs)

    def syncing_file(path: Path) -> None:
        if path.parent == pool:
            events.append("file-fsync")
            if stage == "file-fsync":
                raise OSError("injected publication file-fsync failure")
        file_sync(path)

    def replacing(src: str, dst: str) -> None:
        if Path(dst) == pool / "manifest.json":
            events.append("replace")
            if stage == "replace":
                raise OSError("injected publication replace failure")
            replace(src, dst)
            events.append("replaced")
        else:
            replace(src, dst)

    def syncing_dir(path: Path) -> None:
        # Inject at the root publication's post-replace barrier, not at a
        # preparatory ancestor barrier an atomic writer may also perform.
        if path == pool and "replaced" in events:
            events.append("dir-fsync")
            raise OSError("injected publication dir-fsync failure")
        dir_sync(path)

    monkeypatch.setattr(Path, "write_text", writing)
    monkeypatch.setattr(atomic, "_fsync_file", syncing_file)
    monkeypatch.setattr(atomic, "_fsync_dir", syncing_dir)
    monkeypatch.setattr(atomic.os, "replace", replacing)
    with pytest.raises(OSError, match="injected publication"):
        _export(tmp_path / "b", pool, copy_replay=True)
    assert "publication did not confirm success" in capsys.readouterr().out
    expected_events = {
        "write": ["write"], "file-fsync": ["write", "file-fsync"],
        "replace": ["write", "file-fsync", "replace"],
        "dir-fsync": ["write", "file-fsync", "replace", "replaced", "dir-fsync"],
    }
    assert events == expected_events[stage]
    for name, data in before.items():
        if name != "manifest.json":
            assert (pool / name).read_bytes() == data
    archives = list(pool.glob("generations/*/previous-manifest.json"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == before["manifest.json"]
    observed = _warmstart(pool, tmp_path / "restored", monkeypatch)
    if stage == "dir-fsync":
        assert observed["weights"] == ["B"]
        assert observed["replay"] == ["B"]
        current = (pool / "manifest.json").read_bytes()
        assert any(p.read_bytes() == current for p in pool.glob("generations/*/manifest.json"))
    else:
        assert (pool / "manifest.json").read_bytes() == before["manifest.json"]
        assert observed["weights"] == ["A"]
        assert observed["replay"] == ["A"]


def test_relative_output_and_explicit_old_manifest_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    _donor(tmp_path / "a", "A", shards=(1,), sidecars=True)
    _donor(tmp_path / "b", "B", shards=(2,), sidecars=False)
    pool = Path("pool")
    old = _export(tmp_path / "a", pool, copy_replay=True)
    _export(tmp_path / "b", pool, copy_replay=True)
    assert _warmstart(pool, tmp_path / "new", monkeypatch)["replay"] == ["B"]
    archived = next(pool.glob("generations/*/previous-manifest.json"))
    atomic.atomic_write_bytes(pool / "manifest.json", archived.read_bytes())
    assert json.loads((pool / "manifest.json").read_text()) == old
    restored = _warmstart(pool, tmp_path / "old", monkeypatch)
    assert restored["weights"] == ["A"]
    assert restored["replay"] == ["A"]


def test_legacy_manifest_without_seed_dir_still_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _donor(tmp_path / "a", "A", shards=(1,), sidecars=True)
    pool = tmp_path / "pool"
    manifest = _export(tmp_path / "a", pool, copy_replay=True)
    selected = pool / manifest["entries"][0].pop("seed_dir")
    shutil.copytree(selected, pool / "seeds" / "slot_000")
    (pool / "manifest.json").write_text(json.dumps(manifest))
    restored = _warmstart(pool, tmp_path / "restored", monkeypatch)
    assert restored["weights"] == ["A"]
    assert restored["replay"] == ["A"]


@pytest.mark.parametrize("symlink", [False, True])
def test_output_inside_replay_source_refuses_before_mutation(tmp_path: Path, symlink: bool) -> None:
    td = _donor(tmp_path / "a", "A", shards=(1,), sidecars=True)
    replay = td / "replay_shards"
    pool = replay / "pool"
    if symlink:
        link = tmp_path / "alias"
        link.symlink_to(replay, target_is_directory=True)
        pool = link / "pool"
    before = _snapshot(replay)
    with pytest.raises(ValueError, match="must not be inside a replay source"):
        _export(tmp_path / "a", pool, copy_replay=True)
    assert not pool.exists()
    assert _snapshot(replay) == before


def test_empty_snapshot_is_not_reloaded_from_a_new_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(_pool: Path) -> dict:
        pytest.fail("captured empty snapshot must not be re-read")

    monkeypatch.setattr(recovery, "_load_salvage_manifest", unexpected)
    assert recovery._select_salvage_seed_slot(
        seed_pool_dir=tmp_path, trial_dir=tmp_path, trial_id="new", manifest={},
    ) == (None, -1, 0)
    assert recovery._load_salvage_manifest_entry(seed_pool_dir=tmp_path, slot=0, manifest={}) is None


@pytest.mark.parametrize("contents", [None, "{bad", "[]", "null"])
def test_legacy_bad_or_missing_manifest_remains_unusable(tmp_path: Path, contents: str | None) -> None:
    if contents is not None:
        (tmp_path / "manifest.json").write_text(contents)
    assert recovery._select_salvage_seed_slot(
        seed_pool_dir=tmp_path, trial_dir=tmp_path, trial_id="new",
    ) == (None, -1, 0)
    assert recovery._load_salvage_manifest_entry(seed_pool_dir=tmp_path, slot=0) is None


def test_unpublished_legacy_slot_does_not_contaminate_first_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _donor(tmp_path / "a", "A", shards=(1,), sidecars=True)
    _donor(tmp_path / "b", "B", shards=(), sidecars=False)
    pool = tmp_path / "pool"
    old_manifest = _export(tmp_path / "a", pool, copy_replay=True)
    old_slot = pool / old_manifest["entries"][0]["seed_dir"]
    shutil.copytree(old_slot, pool / "seeds" / "slot_000")
    (pool / "manifest.json").unlink()
    before = _snapshot(pool)
    _export(tmp_path / "b", pool, copy_replay=False)
    assert {name: (pool / name).read_bytes() for name in before} == before
    restored = _warmstart(pool, tmp_path / "restored", monkeypatch)
    assert restored["weights"] == ["B"]
    assert restored["replay"] == []


def test_source_mutation_after_export_does_not_change_banked_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    td = _donor(tmp_path / "a", "A", shards=(1,), sidecars=True)
    pool = tmp_path / "pool"
    _export(tmp_path / "a", pool, copy_replay=True)
    before = _snapshot(pool)
    (td / "checkpoint_000001" / "trainer.pt").write_text("later-source")
    zarr.open_group(str(td / "replay_shards" / "shard_000001.zarr"), mode="a")["donor"][:] = ["Z"]
    assert _snapshot(pool) == before
    restored = _warmstart(pool, tmp_path / "restored", monkeypatch)
    assert restored["weights"] == ["A"]
    assert restored["replay"] == ["A"]


def test_reexport_keeps_already_restored_replay_links_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _donor(tmp_path / "a", "A", shards=(1,), sidecars=True)
    _donor(tmp_path / "b", "B", shards=(2,), sidecars=True)
    pool = tmp_path / "pool"
    _export(tmp_path / "a", pool, copy_replay=True)
    old_trial = tmp_path / "old_trial"
    assert _warmstart(pool, old_trial, monkeypatch)["replay"] == ["A"]
    _export(tmp_path / "b", pool, copy_replay=True)
    assert _warmstart(pool, tmp_path / "new_trial", monkeypatch)["replay"] == ["B"]
    old_shard = old_trial / "replay_shards" / "shard_000001.zarr"
    assert old_shard.is_symlink()
    assert np.asarray(zarr.open_group(str(old_shard), mode="r")["donor"]).tolist() == ["A"]


@pytest.mark.parametrize("first_directory_exists", [False, True])
def test_duplicate_slot_ids_are_rejected_before_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, first_directory_exists: bool,
) -> None:
    _donor(tmp_path / "a", "A", shards=(), sidecars=True)
    pool = tmp_path / "pool"
    manifest = _export(tmp_path / "a", pool, copy_replay=False)
    duplicate = dict(manifest["entries"][0])
    duplicate["seed_dir"] = "other_seed"
    shutil.copytree(pool / manifest["entries"][0]["seed_dir"], pool / "other_seed")
    (pool / "other_seed" / "trainer.pt").write_text("B")
    manifest["entries"].append(duplicate)
    if not first_directory_exists:
        manifest["entries"][0]["seed_dir"] = "missing_seed"
    (pool / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate salvage slot"):
        _warmstart(pool, tmp_path / "train_trial_abc12_00001_0", monkeypatch)


def test_failed_initial_export_does_not_replace_a_concurrent_complete_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _donor(tmp_path / "work", "A", shards=(), sidecars=True)
    b = _donor(tmp_path / "other", "B", shards=(), sidecars=True)
    b = b.rename(a.parent / b.name.replace("00000", "00001", 1))
    _donor(tmp_path / "successor", "C", shards=(), sidecars=True)
    pool = tmp_path / "pool"
    copy = salvage.shutil.copy2
    write_manifest = salvage.atomic_write_text
    published = False

    def fail_b(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(src) == b / "checkpoint_000001" / "rng_state.json":
            raise OSError("failed B copy")
        return copy(src, dst, *args, **kwargs)

    def publish_successor(path: Path, content: str, **kwargs: Any) -> None:
        nonlocal published
        write_manifest(path, content, **kwargs)
        if path.name == "manifest.json" and path.parent.parent == pool / "generations" and not published:
            published = True
            _export(tmp_path / "successor", pool, copy_replay=False)

    monkeypatch.setattr(salvage.shutil, "copy2", fail_b)
    monkeypatch.setattr(salvage, "atomic_write_text", publish_successor)
    with pytest.raises((FileExistsError, RuntimeError)):
        _export(tmp_path / "work", pool, copy_replay=False, top_n=2)
    assert published
    assert _warmstart(pool, tmp_path / "restored", monkeypatch)["weights"] == ["C"]


@pytest.mark.parametrize("fault", ["cross-device", "unsupported", "directory-fsync"])
def test_partial_publication_failure_obeys_link_commit_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    _donor(tmp_path / "a", "A", shards=(), sidecars=True)
    pool = tmp_path / "pool"
    original_copy = salvage.shutil.copy2
    link = salvage.os.link
    events: list[str] = []

    def fail_copy(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(src).name == "rng_state.json":
            raise OSError("failed slot copy")
        return original_copy(src, dst, *args, **kwargs)

    def publish(src: Any, dst: Any) -> None:
        events.append("link")
        if fault != "directory-fsync":
            code = errno.EXDEV if fault == "cross-device" else errno.EOPNOTSUPP
            raise OSError(code, "injected unsupported publication")
        link(src, dst)

    def fail_sync(_path: Path) -> None:
        events.append("directory-fsync")
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(salvage.shutil, "copy2", fail_copy)
    monkeypatch.setattr(salvage.os, "link", publish)
    monkeypatch.setattr(salvage, "_fsync_dir", fail_sync)
    with pytest.raises(OSError, match="injected"):
        _export(tmp_path / "a", pool, copy_replay=False)
    archived = next(pool.glob("generations/*/manifest.json"))
    assert json.loads(archived.read_text())["failed_slots"]
    assert not list(pool.glob("manifest.json.tmp.*"))
    if fault == "directory-fsync":
        assert events == ["link", "directory-fsync"]
        assert (pool / "manifest.json").read_bytes() == archived.read_bytes()
        assert not (pool / "manifest.json").samefile(archived)
        before = archived.read_bytes()
        (pool / "manifest.json").write_text("legacy in-place mutation")
        assert archived.read_bytes() == before
    else:
        assert events == ["link"]
        assert not (pool / "manifest.json").exists()


def test_distinct_slot_ids_keep_selected_entry_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = _donor(tmp_path / "work", "A", shards=(), sidecars=True)
    b = _donor(tmp_path / "other", "B", shards=(), sidecars=True)
    b = b.rename(a.parent / b.name.replace("00000", "00001", 1))
    pool = tmp_path / "pool"
    manifest = _export(tmp_path / "work", pool, copy_replay=False, top_n=2)
    for index, entry in enumerate(manifest["entries"]):
        entry["result_row"]["config"] = {"lr": (index + 1) / 1000}
        entry["result_row"]["sf_nodes_next"] = (index + 1) * 100
    (pool / "manifest.json").write_text(json.dumps(manifest))
    for index, donor in enumerate(("A", "B")):
        observed = _warmstart(
            pool, tmp_path / f"train_trial_abc12_{index:05d}_0", monkeypatch,
            donor_config=True,
        )
        assert observed["weights"] == [donor]
        assert observed["pid"]["nodes"] == (index + 1) * 100
        assert observed["config_lr"] == (index + 1) / 1000


@pytest.mark.parametrize("archive", ["previous-manifest.json", "manifest.json"])
def test_generation_manifest_write_failure_preserves_previous_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archive: str,
) -> None:
    _donor(tmp_path / "a", "A", shards=(1,), sidecars=True)
    _donor(tmp_path / "b", "B", shards=(2,), sidecars=True)
    pool = tmp_path / "pool"
    _export(tmp_path / "a", pool, copy_replay=True)
    before = _snapshot(pool)
    atomic_write = atomic.atomic_write

    def fail_archive(path: Path, *args: Any, **kwargs: Any) -> None:
        if path.name == archive and path.parent.parent == pool / "generations":
            raise OSError("injected archive failure")
        atomic_write(path, *args, **kwargs)

    monkeypatch.setattr(atomic, "atomic_write", fail_archive)
    with pytest.raises(OSError, match="injected archive failure"):
        _export(tmp_path / "b", pool, copy_replay=True)
    assert {name: (pool / name).read_bytes() for name in before} == before
    assert _warmstart(pool, tmp_path / "restored", monkeypatch)["weights"] == ["A"]


@pytest.mark.parametrize("legacy_slot", [False, True])
@pytest.mark.parametrize("dry_run", [False, True])
def test_reinit_refuses_generation_pool_before_model_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy_slot: bool, dry_run: bool,
) -> None:
    _donor(tmp_path / "a", "A", shards=(), sidecars=False)
    _donor(tmp_path / "b", "B", shards=(), sidecars=False)
    pool = tmp_path / "pool"
    if legacy_slot:
        manifest = _export(tmp_path / "a", pool, copy_replay=False)
        selected = pool / manifest["entries"][0]["seed_dir"]
        shutil.copytree(selected, pool / "seeds" / "slot_000")
        manifest["entries"][0]["seed_dir"] = "seeds/slot_000"
        (pool / "manifest.json").write_text(json.dumps(manifest))
    _export(tmp_path / "b", pool, copy_replay=False)
    before = _snapshot(pool)
    module = load_script_module("reinit_value_heads.py", "t1_reinit_generation_test")

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("unsupported generation pool must be refused before model load")

    monkeypatch.setattr(module.torch, "load", unexpected)
    monkeypatch.setattr("sys.argv", ["reinit_value_heads.py", str(pool)] + (["--dry-run"] if dry_run else []))
    with pytest.raises(SystemExit, match="only legacy fixed-slot pools"):
        module.main()
    assert _snapshot(pool) == before


@pytest.mark.parametrize("metadata", [None, {"entries": [{}]}, {"entries": [{"slot": 0, "seed_dir": "seeds/slot_000"}]}])
@pytest.mark.parametrize("dry_run", [False, True])
def test_reinit_legacy_pool_still_reaches_same_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: dict | None, dry_run: bool,
) -> None:
    checkpoint = tmp_path / "seeds" / "slot_000" / "trainer.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("legacy")
    if metadata is not None:
        (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    before = _snapshot(tmp_path)
    loaded: list[Path] = []
    module = load_script_module("reinit_value_heads.py", "t1_reinit_legacy_test")

    def stop_before_deserialization(path: Path, **_kwargs: Any) -> Any:
        loaded.append(path)
        raise RuntimeError("inert load boundary")

    monkeypatch.setattr(module.torch, "load", stop_before_deserialization)
    monkeypatch.setattr("sys.argv", ["reinit_value_heads.py", str(tmp_path)] + (["--dry-run"] if dry_run else []))
    with pytest.raises(RuntimeError, match="inert load boundary"):
        module.main()
    assert loaded == [checkpoint]
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize("metadata", [
    "{bad", "null", "[]", "{}", '{"entries":[]}', '{"entries":[null]}',
    '{"entries":[{"slot":0},{"slot":0}]}', '{"entries":[{"slot":true}]}',
    '{"entries":[{"slot":0.0}]}', '{"entries":[{"slot":"0"}]}',
    '{"entries":[{"slot":-1}]}', '{"entries":[{"slot":1}]}',
    '{"entries":[{"seed_dir":null}]}', '{"entries":[{"seed_dir":""}]}',
    '{"entries":[{"seed_dir":"other"}]}',
])
def test_reinit_bad_manifest_refuses_without_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: str,
) -> None:
    checkpoint = tmp_path / "seeds" / "slot_000" / "trainer.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("legacy")
    (tmp_path / "manifest.json").write_text(metadata)
    module = load_script_module("reinit_value_heads.py", "t1_reinit_bad_manifest_test")

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("ambiguous manifest must be refused before model load")

    monkeypatch.setattr(module.torch, "load", unexpected)
    monkeypatch.setattr("sys.argv", ["reinit_value_heads.py", str(tmp_path), "--dry-run"])
    with pytest.raises(SystemExit, match="refusing to rewrite"):
        module.main()
    assert checkpoint.read_text() == "legacy"


@pytest.mark.parametrize("fault", ["utf8", "unreadable", "broken-link"])
def test_reinit_existing_unreadable_manifest_is_not_legacy_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    checkpoint = tmp_path / "seeds" / "slot_000" / "trainer.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("legacy")
    manifest = tmp_path / "manifest.json"
    if fault == "broken-link":
        manifest.symlink_to(tmp_path / "missing.json")
    else:
        manifest.write_bytes(b"\xff" if fault == "utf8" else b"{}")
    if fault == "unreadable":
        read = Path.read_text

        def deny(path: Path, *args: Any, **kwargs: Any) -> str:
            if path == manifest:
                raise PermissionError("injected read refusal")
            return read(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", deny)
    module = load_script_module("reinit_value_heads.py", "t1_reinit_unreadable_test")

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("unreadable manifest must be refused before model load")

    monkeypatch.setattr(module.torch, "load", unexpected)
    monkeypatch.setattr("sys.argv", ["reinit_value_heads.py", str(tmp_path)])
    with pytest.raises(SystemExit, match="unreadable pool manifest"):
        module.main()
    assert checkpoint.read_text() == "legacy"
