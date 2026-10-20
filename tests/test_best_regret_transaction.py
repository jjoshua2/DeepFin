from __future__ import annotations

import contextlib
import json
import multiprocessing
import os
import sys
from collections.abc import Generator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from chess_anti_engine.tune import trainable_report as report


class _Trainer:
    step = 7

    def __init__(self, payload: str, *, fail: bool = False, entered: Any = None, release: Any = None) -> None:
        self.payload, self.fail = payload, fail
        self.entered, self.release = entered, release

    def save(self, path: Path) -> None:
        path.write_text(self.payload, encoding="utf-8")
        if self.entered is not None:
            self.entered.set()
            if not self.release.wait(10):
                raise RuntimeError("test release deadline")
        if self.fail:
            raise RuntimeError("fake save failure")


def _manifest_module() -> ModuleType:
    module = ModuleType("chess_anti_engine.tune.salvage")
    module.__dict__["build_pool_manifest_dict"] = lambda **kwargs: kwargs
    return module


def _install_manifest_fake() -> None:
    sys.modules["chess_anti_engine.tune.salvage"] = _manifest_module()


@pytest.fixture(autouse=True)
def _fake_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "chess_anti_engine.tune.salvage", _manifest_module())


def _save(root: Path, payload: str, regret: float = 0.1, **trainer_kwargs: Any) -> None:
    pid = SimpleNamespace(wdl_regret=regret, ema_winrate=0.5, state_dict=lambda: {"regret": regret})
    report._update_best_regret_checkpoints(
        trainer=_Trainer(payload, **trainer_kwargs), pid=pid, best_regret_dir=root,
        iteration_idx=2, opp_strength_ema=10.0, best_loss=1.0,
    )


def _index(root: Path) -> list[dict]:
    return json.loads((root / "index.json").read_text(encoding="utf-8"))


def test_same_metric_step_iteration_allocates_distinct_checkpoint_slots(tmp_path: Path) -> None:
    _save(tmp_path, "trial-a")
    _save(tmp_path, "trial-b")
    entries = _index(tmp_path)
    assert len(entries) == 2
    assert len({e["tag"] for e in entries}) == 2
    # train.sh salvage-export --iter selects this legacy suffix by glob.
    assert {p.name for p in tmp_path.glob("regret_*_iter2")} == {e["tag"] for e in entries}
    assert {(tmp_path / e["tag"] / "trainer.pt").read_text() for e in entries} == {"trial-a", "trial-b"}
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert {e["seed_dir"] for e in manifest["entries"]} == {e["tag"] for e in entries}


def test_failed_same_identity_save_cannot_delete_previous_checkpoint(tmp_path: Path) -> None:
    _save(tmp_path, "retained")
    before = _index(tmp_path)
    _save(tmp_path, "partial", fail=True)
    assert _index(tmp_path) == before
    assert (tmp_path / before[0]["tag"] / "trainer.pt").read_text() == "retained"
    assert {p.name for p in tmp_path.iterdir() if p.is_dir()} == {before[0]["tag"]}
    _save(tmp_path, "next")
    assert len(_index(tmp_path)) == 2


def test_top_n_retention_and_nonqualifying_save_remain_intact(tmp_path: Path) -> None:
    for regret in (0.1, 0.2, 0.3):
        _save(tmp_path, str(regret), regret)
    before = _index(tmp_path)
    _save(tmp_path, "not qualified", 0.4)
    assert _index(tmp_path) == before
    _save(tmp_path, "new best", 0.05)
    entries = _index(tmp_path)
    assert [e["regret"] for e in entries] == [0.05, 0.1, 0.2]
    assert all((tmp_path / e["tag"] / "trainer.pt").exists() for e in entries)
    assert not (tmp_path / before[-1]["tag"]).exists()


def test_index_publish_failure_does_not_prune_committed_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for regret in (0.1, 0.2, 0.3):
        _save(tmp_path, str(regret), regret)
    before = _index(tmp_path)
    write = report.atomic_write_text

    def fail_index(path: Path, text: str, **kwargs: Any) -> None:
        if path.name == "index.json":
            raise OSError("index publish failed")
        write(path, text, **kwargs)

    monkeypatch.setattr(report, "atomic_write_text", fail_index)
    with pytest.raises(OSError, match="index publish failed"):
        _save(tmp_path, "uncommitted", 0.05)
    assert _index(tmp_path) == before
    assert all((tmp_path / e["tag"] / "trainer.pt").exists() for e in before)
    monkeypatch.setattr(report, "atomic_write_text", write)
    _save(tmp_path, "retry", 0.04)
    assert _index(tmp_path)[0]["regret"] == 0.04


def test_writer_lock_covers_read_save_index_prune_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    for regret in (0.1, 0.2, 0.3):
        _save(tmp_path, str(regret), regret)
    held = False
    observed: set[str] = set()

    @contextlib.contextmanager
    def lock(_root: Path) -> Generator[None, None, None]:
        nonlocal held
        assert not held
        held = True
        try:
            yield
        finally:
            held = False

    read = Path.read_text
    save = _Trainer.save
    write = report.atomic_write_text
    remove = report.shutil.rmtree

    def read_checked(path: Path, *args: Any, **kwargs: Any) -> str:
        assert held
        observed.add("read")
        return read(path, *args, **kwargs)

    def save_checked(trainer: _Trainer, path: Path) -> None:
        assert held
        observed.add("save")
        save(trainer, path)

    def write_checked(path: Path, text: str, **kwargs: Any) -> None:
        assert held
        observed.add(path.name)
        write(path, text, **kwargs)

    def remove_checked(path: Path, **kwargs: Any) -> None:
        assert held
        observed.add("prune")
        remove(path, **kwargs)

    monkeypatch.setattr(report, "_best_regret_transaction", lock)
    monkeypatch.setattr(Path, "read_text", read_checked)
    monkeypatch.setattr(_Trainer, "save", save_checked)
    monkeypatch.setattr(report, "atomic_write_text", write_checked)
    monkeypatch.setattr(report.shutil, "rmtree", remove_checked)
    _save(tmp_path, "new best", 0.05)
    assert not held
    assert {"read", "save", "index.json", "prune", "manifest.json"} <= observed


def test_lock_sidecar_identity_is_stable_across_index_replacements(tmp_path: Path) -> None:
    _save(tmp_path, "first")
    lock = tmp_path / ".best_regret.lock"
    identity = lock.stat().st_ino
    _save(tmp_path, "second", 0.05)
    assert lock.stat().st_ino == identity


def _blocked_writer(root: str, entered: Any, release: Any) -> None:
    _install_manifest_fake()
    _save(Path(root), "child", entered=entered, release=release)


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock qualification; Windows needs its own lock backend check")
def test_another_process_cannot_lock_directory_during_checkpoint_save(tmp_path: Path) -> None:
    import fcntl

    ctx = multiprocessing.get_context("spawn")
    entered, release = ctx.Event(), ctx.Event()
    child = ctx.Process(target=_blocked_writer, args=(str(tmp_path), entered, release))
    child.start()
    try:
        assert entered.wait(10), "fake save did not reach its controlled hold"
        with (tmp_path / ".best_regret.lock").open("a+b") as contender:
            try:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            finally:
                fcntl.flock(contender.fileno(), fcntl.LOCK_UN)
    finally:
        release.set()
        child.join(timeout=10)
        if child.is_alive():
            child.terminate()
            child.join(timeout=5)
    assert child.exitcode == 0
    assert (tmp_path / _index(tmp_path)[0]["tag"] / "trainer.pt").read_text() == "child"
    _save(tmp_path, "parent", 0.05)
    assert len(_index(tmp_path)) == 2


def test_even_forced_unique_id_collision_never_reuses_an_existing_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report.uuid, "uuid4", lambda: SimpleNamespace(hex="forced-collision"))
    _save(tmp_path, "original")
    before = _index(tmp_path)
    with pytest.raises(FileExistsError):
        _save(tmp_path, "must not replace")
    assert _index(tmp_path) == before
    assert (tmp_path / before[0]["tag"] / "trainer.pt").read_text() == "original"
