"""Direct seed-pointer publication fixtures: no retire/feed CLI or scoring."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import stat

import pytest

from chess_anti_engine.utils import atomic, config_yaml
from scripts import blindspot_retire_step as retire

OLD = "/synthetic/old.txt"
NEW = "/synthetic/new.txt"


def config(seed: int = 1, pointer: str = OLD) -> bytes:
    return f"seed: {seed}\nselfplay:\n  opening_fen_list_path: {json.dumps(pointer)}\n".encode()


def parsed(path: Path) -> dict:
    return config_yaml.flatten_run_config_defaults(config_yaml.load_yaml_file(path))


@pytest.fixture
def target(tmp_path: Path) -> Path:
    path = tmp_path / "run.yaml"
    path.write_bytes(config())
    path.chmod(0o600)
    return path


def lock_path(target: Path) -> Path:
    return target.with_name(f".{target.name}.seed-repoint.lock")


@pytest.mark.parametrize("pointer", [NEW, "/synthetic/bad: path", "#hash", "a 'quote' \"double\"", "a\nline", "back\\slash", "雪.txt"])
def test_exact_path_round_trip(target: Path, pointer: str) -> None:
    assert retire._repoint_yaml(str(target), pointer)
    assert parsed(target) == {"seed": 1, "opening_fen_list_path": pointer}
    assert not list(target.parent.glob("*.tmp.*"))


def test_validation_and_write_never_expose_staging(target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_load = config_yaml.load_yaml_file
    observed = []

    def observe(path):
        assert Path(path) != target
        observed.append(target.read_bytes())
        return real_load(path)

    monkeypatch.setattr(config_yaml, "load_yaml_file", observe)
    assert retire._repoint_yaml(str(target), "/new: quoted")
    assert observed == [config(), config()]


@pytest.mark.parametrize("failure_at", [1, 2])
@pytest.mark.parametrize("concurrent", [False, True])
def test_validation_failure_never_rolls_back(target: Path, monkeypatch: pytest.MonkeyPatch, failure_at: int, concurrent: bool) -> None:
    real_load = config_yaml.load_yaml_file
    calls = 0
    edited = config(42, "/other.txt")

    def fail(path):
        nonlocal calls
        calls += 1
        if calls == failure_at:
            if concurrent:
                target.write_bytes(edited)
            raise ValueError("injected strict validation failure")
        return real_load(path)

    monkeypatch.setattr(config_yaml, "load_yaml_file", fail)
    assert not retire._repoint_yaml(str(target), NEW)
    assert target.read_bytes() == (edited if concurrent else config())
    assert not list(target.parent.glob("*.tmp.*"))


@pytest.mark.parametrize("original", [b"not: [valid", b"unknown_setting: true\nselfplay:\n  opening_fen_list_path: old\n", b"seed: 1\n", b"selfplay:\n  opening_fen_list_path:\nseed: 1\n", config() + b"  opening_fen_list_path: duplicate\n"])
def test_invalid_or_ambiguous_original_refuses(target: Path, original: bytes) -> None:
    target.write_bytes(original)
    assert not retire._repoint_yaml(str(target), NEW)
    assert target.read_bytes() == original


def test_unrelated_preexisting_settings_are_merged(target: Path) -> None:
    # List preparation used OLD, then another edit changed only an unrelated key.
    target.write_bytes(config(42))
    assert retire._repoint_yaml(str(target), NEW, expected_seed_path=OLD)
    assert parsed(target) == {"seed": 42, "opening_fen_list_path": NEW}


def test_stale_seed_pointer_refuses(target: Path) -> None:
    target.write_bytes(config(42, "/other.txt"))
    assert not retire._repoint_yaml(str(target), NEW, expected_seed_path=OLD)
    assert target.read_bytes() == config(42, "/other.txt")


def test_expected_pointer_uses_captured_bytes(target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target.write_bytes(config(42, "/other.txt"))
    real_load = config_yaml.load_yaml_file

    def edit_live_then_parse_snapshot(path):
        target.write_bytes(config())
        return real_load(path)

    monkeypatch.setattr(config_yaml, "load_yaml_file", edit_live_then_parse_snapshot)
    assert not retire._repoint_yaml(str(target), NEW, expected_seed_path=OLD)
    assert target.read_bytes() == config()


@pytest.mark.parametrize("replace_inode", [False, True])
@pytest.mark.parametrize("same_bytes", [False, True])
def test_observed_edit_during_validation_is_preserved(target: Path, monkeypatch: pytest.MonkeyPatch, replace_inode: bool, same_bytes: bool) -> None:
    real_load = config_yaml.load_yaml_file
    calls = 0
    edited = config() if same_bytes else config(42)

    def edit(path):
        nonlocal calls
        calls += 1
        result = real_load(path)
        if calls == 2:
            if replace_inode:
                replacement = target.with_name("editor.yaml")
                replacement.write_bytes(edited)
                os.replace(replacement, target)
            else:
                target.write_bytes(edited)
                # Pin the metadata change even on coarse timestamp filesystems.
                os.utime(target, ns=(1, 1))
        return result

    monkeypatch.setattr(config_yaml, "load_yaml_file", edit)
    assert not retire._repoint_yaml(str(target), NEW, expected_seed_path=OLD)
    assert target.read_bytes() == edited


def test_uncoordinated_last_check_to_replace_race_is_not_cas(target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Deliberate contract-limit witness, NOT a claim that arbitrary editors lock.
    real_sync = atomic._fsync_file

    def late_edit(path):
        target.write_bytes(config(42, "/late-editor.txt"))
        real_sync(path)

    monkeypatch.setattr(atomic, "_fsync_file", late_edit)
    assert retire._repoint_yaml(str(target), NEW, expected_seed_path=OLD)
    assert parsed(target) == {"seed": 1, "opening_fen_list_path": NEW}


@pytest.mark.parametrize("phase", ["stage_write", "file_sync", "replace"])
def test_precommit_io_failure_preserves_old(target: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    def fail(*_args, **_kwargs):
        raise OSError("injected precommit failure")

    if phase == "stage_write":
        monkeypatch.setattr(Path, "write_bytes", fail)
    elif phase == "file_sync":
        monkeypatch.setattr(atomic, "_fsync_file", fail)
    else:
        monkeypatch.setattr(atomic.os, "replace", fail)
    assert not retire._repoint_yaml(str(target), NEW, expected_seed_path=OLD)
    assert target.read_bytes() == config()
    assert not list(target.parent.glob("*.tmp.*"))


@pytest.mark.parametrize("later_editor", [False, True])
def test_postcommit_failure_is_unconfirmed_not_rollback(target: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], later_editor: bool) -> None:
    seen = []
    real_replace = atomic.os.replace
    real_sync_dir = atomic._fsync_dir
    committed = False

    def observe_replace(src, dst):
        nonlocal committed
        real_replace(src, dst)
        if Path(dst) == target:
            committed = True

    def fail_after_replace(path):
        # F2 may sync directories before commit too; bind injection to the
        # actual replace boundary, not a call count or current helper order.
        if not committed:
            return real_sync_dir(path)
        seen.append(parsed(target))
        if later_editor:
            target.write_bytes(config(42, "/later.txt"))
        raise OSError("injected directory durability failure")

    monkeypatch.setattr(atomic.os, "replace", observe_replace)
    monkeypatch.setattr(atomic, "_fsync_dir", fail_after_replace)
    assert not retire._repoint_yaml(str(target), NEW, expected_seed_path=OLD)
    assert seen == [{"seed": 1, "opening_fen_list_path": NEW}]
    assert parsed(target) == {"seed": 42 if later_editor else 1, "opening_fen_list_path": "/later.txt" if later_editor else NEW}
    assert "unconfirmed" in capsys.readouterr().err


@pytest.mark.parametrize(("mode", "mask"), [(0o600, 0o022), (0o640, 0o077), (0o664, 0o002)])
@pytest.mark.parametrize("fail", [False, True])
def test_mode_never_widens_during_stage(target: Path, monkeypatch: pytest.MonkeyPatch, mode: int, mask: int, fail: bool) -> None:
    target.chmod(mode)
    real_write = Path.write_bytes
    seen = []

    def inspect(path, data):
        before = stat.S_IMODE(path.stat().st_mode)
        assert before & ~mode == 0
        seen.append(before)
        result = real_write(path, data)
        if fail:
            raise OSError("injected staged failure")
        return result

    monkeypatch.setattr(Path, "write_bytes", inspect)
    old_mask = os.umask(mask)
    try:
        result = retire._repoint_yaml(str(target), NEW, expected_seed_path=OLD)
    finally:
        os.umask(old_mask)
    assert result is not fail
    assert seen
    assert stat.S_IMODE(target.stat().st_mode) == mode
    assert stat.S_IMODE(lock_path(target).stat().st_mode) == 0o600
    if fail:
        assert target.read_bytes() == config()


@pytest.mark.parametrize("kind", ["symlink", "dangling", "directory", "fifo"])
def test_terminal_target_refused(target: Path, kind: str) -> None:
    other = target.with_name("other.yaml")
    target.unlink()
    if kind == "symlink":
        other.write_bytes(config())
        target.symlink_to(other)
    elif kind == "dangling":
        target.symlink_to(other)
    elif kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)
    before = target.lstat()
    assert not retire._repoint_yaml(str(target), NEW)
    assert target.lstat().st_ino == before.st_ino
    if kind == "symlink":
        assert other.read_bytes() == config()
    if kind == "dangling":
        assert not other.exists()


@pytest.mark.parametrize("kind", ["symlink", "dangling", "directory", "fifo", "nonempty", "public", "hardlink"])
def test_foreign_lock_refused_without_mutation(target: Path, kind: str) -> None:
    lock = lock_path(target)
    other = target.with_name("foreign")
    if kind in {"symlink", "dangling"}:
        if kind == "symlink":
            other.write_bytes(b"foreign control")
        lock.symlink_to(other)
    elif kind == "directory":
        lock.mkdir()
    elif kind == "fifo":
        os.mkfifo(lock)
    else:
        lock.write_bytes(b"foreign control" if kind == "nonempty" else b"")
        lock.chmod(0o644 if kind == "public" else 0o600)
        if kind == "hardlink":
            os.link(lock, other)
    before = lock.lstat()
    assert not retire._repoint_yaml(str(target), NEW)
    assert target.read_bytes() == config()
    assert lock.lstat() == before
    if kind == "nonempty":
        assert lock.read_bytes() == b"foreign control"
    if kind == "symlink":
        assert other.read_bytes() == b"foreign control"


def test_foreign_lock_owner_refuses(target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock_path(target).write_bytes(b"")
    lock_path(target).chmod(0o600)
    owner = lock_path(target).stat().st_uid
    monkeypatch.setattr(retire.os, "geteuid", lambda: owner + 1)
    assert not retire._repoint_yaml(str(target), NEW)
    assert target.read_bytes() == config()


def test_busy_lock_refuses_then_reuses_same_inode(target: Path) -> None:
    with retire._yaml_repoint_lock(target):
        inode = lock_path(target).stat().st_ino
        assert not retire._repoint_yaml(str(target), NEW, expected_seed_path=OLD)
        assert target.read_bytes() == config()
    assert retire._repoint_yaml(str(target), NEW, expected_seed_path=OLD)
    assert lock_path(target).stat().st_ino == inode
    assert retire._repoint_yaml(str(target), OLD, expected_seed_path=NEW)
    assert lock_path(target).stat().st_ino == inode
    assert lock_path(target).read_bytes() == b""


def test_real_callers_bind_the_prepared_seed_pointer() -> None:
    # Source-only adjacent caller check: never import/run feed or either CLI.
    root = Path(__file__).resolve().parents[1]
    for name in ("blindspot_retire_step.py", "blindspot_feed_step.py"):
        tree = ast.parse((root / "scripts" / name).read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "_repoint_yaml"]
        assert len(calls) == 1
        expected = [kw.value for kw in calls[0].keywords if kw.arg == "expected_seed_path"]
        assert len(expected) == 1
        assert isinstance(expected[0], ast.Name)
        assert expected[0].id == "seed_path"


def test_observed_mutation_during_snapshot_refuses(target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_stat = retire.os.fstat
    calls = 0

    def edit_between_stats(fd):
        nonlocal calls
        result = real_stat(fd)
        if stat.S_ISREG(result.st_mode) and result.st_size:
            calls += 1
            if calls == 1:
                target.write_bytes(config(42))
                os.utime(target, ns=(1, 1))
        return result

    monkeypatch.setattr(retire.os, "fstat", edit_between_stats)
    assert not retire._repoint_yaml(str(target), NEW, expected_seed_path=OLD)
    assert target.read_bytes() == config(42)


def test_validation_uses_real_loader_on_exact_stage(target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_load = config_yaml.load_yaml_file
    calls = 0

    def corrupt_candidate_then_load(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            Path(path).write_bytes(b"seed: [\n")
        return real_load(path)

    monkeypatch.setattr(config_yaml, "load_yaml_file", corrupt_candidate_then_load)
    assert not retire._repoint_yaml(str(target), NEW, expected_seed_path=OLD)
    assert target.read_bytes() == config()


def test_lock_metadata_probe_does_not_follow_link(target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = lock_path(target)
    lock.symlink_to(target)
    real_stat = Path.stat
    followed = []

    def observe(path, **kwargs):
        if path == lock:
            followed.append(kwargs.get("follow_symlinks", True))
        return real_stat(path, **kwargs)

    monkeypatch.setattr(Path, "stat", observe)
    assert not retire._repoint_yaml(str(target), NEW)
    assert followed == [False]
    assert target.read_bytes() == config()
