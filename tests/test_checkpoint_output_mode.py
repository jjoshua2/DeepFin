"""Exercise the exact checkpoint publication tail using only owned inert bytes."""
from __future__ import annotations

import ast
import errno
import os
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar


class TestCheckpointOutputMode(unittest.TestCase):
    observations: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        self.atomic: dict[str, Any] = {}
        self.save_namespace: dict[str, Any] = {}
        self.temporary: tempfile.TemporaryDirectory[str] | None = None
        self.root = Path(__file__).parent
        self.barriers: list[tuple[str, bool]] = []
        self.tail = compile("pass", "inert-uninitialized-tail", "exec")
        self.events: list[dict[str, Any]] = []
        self.writer_calls = 0
        self.fail_writer = False
        self.precreated: list[dict[str, int]] = []

    def setUp(self) -> None:
        source = Path(__file__).resolve().parents[1]
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.temporary.name).resolve()
        atomic_path = source / "chess_anti_engine/utils/atomic.py"
        tree = ast.parse(atomic_path.read_text())
        allowed = {"__future__", "errno", "os", "re", "shutil", "uuid", "pathlib", "collections.abc"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name in allowed for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert node.module in allowed
                assert node.level == 0
        self.atomic = {"__name__": "inert_atomic"}
        exec(compile(tree, str(atomic_path), "exec"), self.atomic)
        self.barriers = []
        sync_dir = self.atomic["_fsync_dir"]

        def owned_sync(path: Path) -> None:
            resolved = path.resolve()
            owned = resolved == self.root or self.root in resolved.parents
            self.barriers.append((str(resolved), owned))
            # Even durable=True must not physically sync outside our fixture.
            if owned:
                sync_dir(path)

        self.atomic["_fsync_dir"] = owned_sync
        trainer_path = source / "chess_anti_engine/train/trainer.py"
        trainer_tree = ast.parse(trainer_path.read_text())
        trainer = next(n for n in trainer_tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
        save = next(n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == "save")
        tail = save.body[-1]
        assert isinstance(tail, ast.Expr)
        assert isinstance(tail.value, ast.Call)
        assert isinstance(tail.value.func, ast.Name)
        assert tail.value.func.id == "atomic_write"
        selected: list[ast.stmt] = [tail]
        if isinstance(tail.value.args[1], ast.Name):
            assert tail.value.args[1].id == "write_checkpoint"
            selected = save.body[-3:]
            assert isinstance(selected[0], ast.Assign)
            assert isinstance(selected[1], ast.FunctionDef)
            assert selected[1].name == "write_checkpoint"
            assert not selected[1].decorator_list
        assert {n.id for statement in selected for n in ast.walk(statement) if isinstance(n, ast.Name)} <= {
            "atomic_write", "path", "torch", "state", "str", "tmp", "_checkpoint_output_mode",
            "mode", "write_checkpoint", "Path",
        }
        helpers = [n for n in trainer_tree.body if isinstance(n, ast.FunctionDef) and n.name == "_checkpoint_output_mode"]
        self.save_namespace = {"Path": Path, "stat": stat, "atomic_write": self.atomic["atomic_write"]}
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *helpers], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(trainer_path), "exec"), self.save_namespace)
        self.tail = compile(ast.Module(body=selected, type_ignores=[]), str(trainer_path), "exec")
        self.events = []
        self.writer_calls = 0
        self.fail_writer = False
        self.atomic["os"] = SimpleNamespace(**{name: getattr(os, name) for name in dir(os)})
        replace = os.replace

        def observed_replace(src: str, dst: str) -> None:
            before = self.mode(Path(src))
            replace(src, dst)
            self.events.append({"publication_before": before, "publication_after": self.mode(Path(dst))})

        self.atomic["os"].replace = observed_replace
        self.precreated = []
        original_open = os.open

        def observed_open(path: str, flags: int, *args: Any) -> int:
            fd = original_open(path, flags, *args)
            try:
                if flags & os.O_CREAT:
                    assert flags & os.O_EXCL
                    temporary = Path(path)
                    assert self.root in temporary.parents
                    info = temporary.stat()
                    assert info.st_uid == os.geteuid()
                    assert info.st_nlink == 1
                    self.precreated.append({
                        "mode": self.mode(temporary), "size": info.st_size,
                        "flags": flags, "owner": info.st_uid, "links": info.st_nlink,
                    })
            except BaseException:
                os.close(fd)
                raise
            return fd

        self.atomic["os"].open = observed_open

    def tearDown(self) -> None:
        self.observations.append({"case": self.id(), "euid": os.geteuid(), "events": self.events, "barriers": self.barriers, "precreated": self.precreated})
        assert self.temporary is not None
        self.temporary.cleanup()

    @staticmethod
    def mode(path: Path) -> int:
        return stat.S_IMODE(path.stat().st_mode)

    def seed(self, name: str = "checkpoint.bin", mode: int = 0o600) -> Path:
        path = self.root / name
        path.write_bytes(b"old inert bytes")
        path.chmod(mode)
        return path

    def save_bytes(self, path: Path, mask: int = 0o022) -> None:
        def writer(_state: object, temporary: str) -> None:
            self.writer_calls += 1
            tmp = Path(temporary)
            self.events.append({"writer_entry": self.mode(tmp) if tmp.exists() else None})
            tmp.write_bytes(b"new inert bytes")
            self.events.append({"after_bytes": self.mode(tmp)})
            if self.fail_writer:
                raise RuntimeError("inert writer failure")

        self.save_namespace.update(path=path, state=None, torch=SimpleNamespace(save=writer))
        previous = os.umask(mask)
        try:
            exec(self.tail, self.save_namespace)
        finally:
            os.umask(previous)
            self.observations.append({
                "case": self.id(), "mask": mask, "euid": os.geteuid(),
                "events": list(self.events),
                "destination_lstat_mode": stat.S_IMODE(path.lstat().st_mode),
                "precreated": list(self.precreated),
            })

    def test_existing_private_and_shared_modes_cover_temp_and_publication(self) -> None:
        unrelated = self.seed("unrelated.bin", 0o600)
        for mode, mask in ((0o600, 0o022), (0o640, 0o022), (0o640, 0o077), (0o660, 0o077)):
            with self.subTest(mode=mode, mask=mask):
                path = self.seed(mode=mode)
                self.events = []
                self.save_bytes(path, mask)
                assert self.events == [
                    {"writer_entry": mode}, {"after_bytes": mode},
                    {"publication_before": mode, "publication_after": mode},
                ]
                assert self.mode(path) == mode
                assert path.read_bytes() == b"new inert bytes"
                assert not list(self.root.glob("*.tmp.*"))
        assert unrelated.read_bytes() == b"old inert bytes"
        assert self.mode(unrelated) == 0o600

    def test_restrictive_umasks_restore_only_requested_existing_mode_before_bytes(self) -> None:
        original_chmod = Path.chmod
        for mode, mask in ((0o600, 0o200), (0o600, 0o277), (0o640, 0o277), (0o624, 0o200)):
            with self.subTest(mode=mode, mask=mask):
                path = self.seed(mode=mode)
                self.events = []
                self.precreated = []
                changes: list[tuple[int, int]] = []

                def observed_chmod(
                    temporary: Path, requested: int, *, desired: int = mode,
                    recorded: list[tuple[int, int]] = changes,
                ) -> None:
                    assert self.root in temporary.parents
                    assert ".tmp." in temporary.name
                    before = self.mode(temporary)
                    assert before & ~desired == 0
                    assert requested == desired
                    recorded.append((before, requested))
                    original_chmod(temporary, requested)
                    assert self.mode(temporary) == desired

                setattr(Path, "chmod", observed_chmod)
                try:
                    self.save_bytes(path, mask)
                finally:
                    setattr(Path, "chmod", original_chmod)
                assert len(self.precreated) == 1
                assert self.precreated[0]["mode"] == mode & ~mask
                assert self.precreated[0]["size"] == 0
                assert self.precreated[0]["flags"] & os.O_EXCL
                assert changes == [(mode & ~mask, mode)]
                assert self.events == [
                    {"writer_entry": mode}, {"after_bytes": mode},
                    {"publication_before": mode, "publication_after": mode},
                ]
                assert self.mode(path) == mode
                assert path.read_bytes() == b"new inert bytes"
                assert not list(self.root.glob("*.tmp.*"))

    def test_new_destinations_keep_restrictive_umask_defaults_without_chmod(self) -> None:
        original_chmod = Path.chmod

        def refuse_chmod(_temporary: Path, _mode: int) -> None:
            raise AssertionError("new-file default must not be chmoded by checkpoint writer")

        setattr(Path, "chmod", refuse_chmod)
        try:
            for mask in (0o200, 0o277):
                with self.subTest(mask=mask):
                    path = self.root / f"new-restrictive-{mask}.bin"
                    self.events = []
                    self.save_bytes(path, mask)
                    mode = 0o666 & ~mask
                    assert self.events == [
                        {"writer_entry": None}, {"after_bytes": mode},
                        {"publication_before": mode, "publication_after": mode},
                    ]
                    assert self.mode(path) == mode
                    assert path.read_bytes() == b"new inert bytes"
                    assert not self.precreated
        finally:
            setattr(Path, "chmod", original_chmod)

    def test_precontent_chmod_failure_preserves_old_inode_mode_and_bytes(self) -> None:
        path = self.seed()
        old_inode = path.stat().st_ino
        original_chmod = Path.chmod
        attempted: list[tuple[int, int]] = []

        def fail_chmod(temporary: Path, requested: int) -> None:
            assert self.root in temporary.parents
            assert temporary.stat().st_size == 0
            attempted.append((self.mode(temporary), requested))
            raise PermissionError(errno.EPERM, "inert precontent chmod failure")

        setattr(Path, "chmod", fail_chmod)
        try:
            try:
                self.save_bytes(path, 0o200)
            except PermissionError as exc:
                error_number = exc.errno
            else:
                raise AssertionError("precontent chmod failure swallowed")
        finally:
            setattr(Path, "chmod", original_chmod)
        assert error_number == errno.EPERM
        assert attempted == [(0o400, 0o600)]
        assert self.writer_calls == 0
        assert not self.events
        assert path.read_bytes() == b"old inert bytes"
        assert self.mode(path) == 0o600
        assert path.stat().st_ino == old_inode
        assert list(self.root.iterdir()) == [path]

    def test_new_file_defaults_follow_umask(self) -> None:
        for mask in (0o000, 0o022, 0o077):
            with self.subTest(mask=mask):
                path = self.root / f"new-{mask}.bin"
                self.events = []
                self.save_bytes(path, mask)
                mode = 0o666 & ~mask
                assert self.events == [
                    {"writer_entry": None}, {"after_bytes": mode},
                    {"publication_before": mode, "publication_after": mode},
                ]
                assert path.read_bytes() == b"new inert bytes"
                assert self.mode(path) == mode

    def test_writer_failure_preserves_old_file_and_cleans_temporary(self) -> None:
        path = self.seed()
        self.fail_writer = True
        try:
            self.save_bytes(path)
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("writer failure swallowed")
        assert message == "inert writer failure"
        assert path.read_bytes() == b"old inert bytes"
        assert self.mode(path) == 0o600
        assert self.events == [{"writer_entry": 0o600}, {"after_bytes": 0o600}]
        assert list(self.root.iterdir()) == [path]

    def test_terminal_symlinks_and_nonregulars_refused_before_writer(self) -> None:
        target = self.seed("target.bin")
        link = self.root / "link.bin"
        link.symlink_to(target)
        dangling = self.root / "dangling.bin"
        dangling.symlink_to(self.root / "missing.bin")
        directory = self.root / "directory"
        directory.mkdir()
        fifo = self.root / "pipe"
        os.mkfifo(fifo)
        for path in (link, dangling, directory, fifo):
            with self.subTest(path=path):
                try:
                    self.save_bytes(path)
                except ValueError as exc:
                    message = str(exc)
                else:
                    raise AssertionError("unsafe destination accepted")
                assert "non-regular or symlink" in message
                assert self.writer_calls == 0
                assert not self.events
        assert link.is_symlink()
        assert dangling.is_symlink()
        assert directory.is_dir()
        assert stat.S_ISFIFO(fifo.lstat().st_mode)
        assert target.read_bytes() == b"old inert bytes"
        assert self.mode(target) == 0o600

    def test_special_mode_bits_are_refused_not_silently_masked(self) -> None:
        for mode in (0o4600, 0o2600, 0o1600):
            with self.subTest(mode=mode):
                path = self.seed(mode=mode)
                assert self.mode(path) == mode
                try:
                    self.save_bytes(path)
                except ValueError as exc:
                    message = str(exc)
                else:
                    raise AssertionError("special bits accepted")
                assert "special mode bits" in message
                assert self.writer_calls == 0
                assert path.read_bytes() == b"old inert bytes"
                assert self.mode(path) == mode

    def test_unsupported_owner_modes_are_refused_before_writer(self) -> None:
        for mode in (0o400, 0o200, 0o000, 0o060):
            with self.subTest(mode=mode, euid=os.geteuid()):
                path = self.seed(mode=mode)
                self.events = []
                try:
                    self.save_bytes(path)
                except ValueError as exc:
                    message = str(exc)
                else:
                    raise AssertionError("unsupported owner mode accepted")
                assert "requires owner read/write" in message
                assert self.writer_calls == 0
                assert not self.events
                assert not self.barriers
                assert self.mode(path) == mode
                assert not list(self.root.glob("*.tmp.*"))
                # Only our owned fixture is made readable to inspect old bytes.
                path.chmod(0o600)
                assert path.read_bytes() == b"old inert bytes"

    def test_hardlink_alias_keeps_old_inode_and_bytes(self) -> None:
        path = self.seed()
        alias = self.root / "alias.bin"
        os.link(path, alias)
        old_inode = alias.stat().st_ino
        self.save_bytes(path)
        assert alias.stat().st_ino == old_inode
        assert path.stat().st_ino != old_inode
        assert alias.read_bytes() == b"old inert bytes"
        assert path.read_bytes() == b"new inert bytes"
        assert self.mode(alias) == self.mode(path) == 0o600

    def test_parent_symlink_uses_trusted_stable_namespace(self) -> None:
        physical = self.root / "physical"
        physical.mkdir()
        parent = self.root / "alias-parent"
        parent.symlink_to(physical, target_is_directory=True)
        path = physical / "checkpoint.bin"
        path.write_bytes(b"old inert bytes")
        path.chmod(0o600)
        self.save_bytes(parent / path.name)
        assert path.read_bytes() == b"new inert bytes"
        assert self.mode(path) == 0o600
        assert parent.is_symlink()
        assert any(not owned for _, owned in self.barriers)

    def test_lstat_errors_propagate_before_writer(self) -> None:
        class BrokenPath:
            @staticmethod
            def lstat() -> object:
                raise PermissionError("inert lstat failure")

        helper = self.save_namespace["_checkpoint_output_mode"]
        try:
            helper(BrokenPath())
        except PermissionError as exc:
            message = str(exc)
        else:
            raise AssertionError("lstat error swallowed")
        assert message == "inert lstat failure"
        assert self.writer_calls == 0

    def test_atomic_failures_before_and_after_replace_keep_mode_contract(self) -> None:
        for stage in ("preflight", "chmod", "file_sync", "replace", "post_replace"):
            with self.subTest(stage=stage):
                path = self.seed()
                self.events = []
                self.writer_calls = 0
                original_dir = self.atomic["_fsync_dir"]
                original_file = self.atomic["_fsync_file"]
                original_chmod = self.atomic["os"].chmod
                original_replace = self.atomic["os"].replace
                calls = [0]

                def fail(*_args: object, failure_stage: str = stage) -> None:
                    raise OSError(errno.EIO, f"inert {failure_stage}")

                def directory_sync(
                    directory: Path, *, failure_stage: str = stage,
                    counter: list[int] = calls, prior_sync: Any = original_dir,
                    failure: Any = fail,
                ) -> None:
                    if directory.resolve() == self.root:
                        counter[0] += 1
                        if failure_stage == "preflight" or (failure_stage == "post_replace" and counter[0] == 2):
                            failure()
                    prior_sync(directory)

                self.atomic["_fsync_dir"] = directory_sync
                if stage == "chmod":
                    self.atomic["os"].chmod = fail
                if stage == "file_sync":
                    self.atomic["_fsync_file"] = fail
                if stage == "replace":
                    self.atomic["os"].replace = fail
                try:
                    try:
                        self.save_bytes(path)
                    except OSError as exc:
                        error_number = exc.errno
                    else:
                        raise AssertionError(f"{stage} failure swallowed")
                    assert error_number == errno.EIO
                finally:
                    self.atomic["_fsync_dir"] = original_dir
                    self.atomic["_fsync_file"] = original_file
                    self.atomic["os"].chmod = original_chmod
                    self.atomic["os"].replace = original_replace
                expected = b"new inert bytes" if stage == "post_replace" else b"old inert bytes"
                assert path.read_bytes() == expected
                assert self.mode(path) == 0o600
                assert not list(self.root.glob("*.tmp.*"))
                assert self.writer_calls == (0 if stage == "preflight" else 1)


if __name__ == "__main__":
    unittest.main()
