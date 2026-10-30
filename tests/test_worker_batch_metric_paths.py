"""Worker telemetry uses local launch artifacts, not server replay directories."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from chess_anti_engine.tune import distributed_runtime as runtime
from chess_anti_engine.tune import trainable


def _production_dirs(trial_dir: Path, server_root: Path) -> dict:
    # Execute only the real startup's pure path assignments, never train_trial.
    tree = ast.parse(inspect.getsource(trainable.train_trial))
    statements = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            name = target.value if isinstance(target, ast.Subscript) else target
            if isinstance(name, ast.Name) and name.id == "distributed_dirs":
                statements.append(node)
    namespace = dict(vars(trainable))
    namespace.update(trial_dir=trial_dir, distributed_server_root=server_root, trial_id="fixture")
    exec(compile(ast.Module(body=sorted(statements, key=lambda n: n.lineno), type_ignores=[]), "<startup-paths>", "exec"), namespace)
    return namespace["distributed_dirs"]


@pytest.fixture
def launched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured: list[Path] = []

    def command(**kwargs: Any) -> list[str]:
        captured.append(kwargs["worker_log"])
        return ["never-executed"]

    monkeypatch.setattr(runtime, "_build_distributed_worker_cmd", command)
    monkeypatch.setattr(runtime, "_spawn_with_reap", lambda **_kwargs: SimpleNamespace())
    trial = tmp_path / "trial"
    server = tmp_path / "server"
    for index in range(2):
        runtime._launch_distributed_worker(config={"distributed_server_root": str(server)}, trial_dir=trial, trial_id="fixture", worker_index=index)
    dirs = _production_dirs(trial, server)
    distracting = dirs["workers_root"] / "worker_00" / "worker.log"
    distracting.write_text("gumbel profile (avg=999)\n")
    return dirs, captured, distracting


def test_phase_scalar_reads_only_launcher_logs_once(launched) -> None:
    dirs, logs, distracting = launched
    logs[0].write_text("gumbel profile (avg=12)\n")
    logs[1].write_text("gumbel profile (avg=20)\n")
    values: list[tuple] = []
    trainer = SimpleNamespace(writer=SimpleNamespace(add_scalar=lambda *args: values.append(args)))
    offsets: dict[str, int] = {}
    kwargs: dict[str, Any] = {"trainer": trainer, "distributed_dirs": dirs, "worker_log_offsets": offsets, "selfplay_secs": 2, "train_secs": 3, "iter_so_far": 6}
    trainable._log_iter_phase_split(iteration_idx=1, **kwargs)
    assert ("selfplay/avg_batch_size", 16.0, 1) in values
    assert str(distracting) not in offsets
    assert set(offsets) == {str(p) for p in logs}
    values.clear()
    trainable._log_iter_phase_split(iteration_idx=2, **kwargs)
    assert all(name != "selfplay/avg_batch_size" for name, *_rest in values)
    with logs[0].open("a") as stream:
        stream.write("gumbel profile (avg=7)\n")
    assert trainable._avg_batch_from_worker_logs(dirs, offsets) == 7


@pytest.mark.parametrize("rotate", [False, True])
def test_truncation_and_rotation_keep_existing_offset_contract(launched, rotate: bool) -> None:
    dirs, logs, _ = launched
    log = logs[0]
    log.write_text("long prefix " * 10 + "gumbel profile (avg=12)\n")
    offsets: dict[str, int] = {}
    assert trainable._avg_batch_from_worker_logs(dirs, offsets) == 12
    if rotate:
        runtime._rotate_worker_logs(log)
    log.write_text("gumbel profile (avg=5)\n")
    # Existing contract skips the now-stale tail after shrink/rotation.
    assert trainable._avg_batch_from_worker_logs(dirs, offsets) == 0
    with log.open("a") as stream:
        stream.write("gumbel profile (avg=8)\n")
    assert trainable._avg_batch_from_worker_logs(dirs, offsets) == 8


def test_missing_artifact_root_does_not_scan_server_logs(launched) -> None:
    dirs, _, _ = launched
    dirs.pop("worker_artifacts_root", None)
    offsets: dict[str, int] = {}
    assert trainable._avg_batch_from_worker_logs(dirs, offsets) == 0
    assert offsets == {}


def test_unreadable_local_log_does_not_use_server_fallback(launched, monkeypatch: pytest.MonkeyPatch) -> None:
    dirs, logs, _ = launched
    logs[0].write_text("gumbel profile (avg=12)\n")
    original = Path.open

    def read(path: Path, *args: Any, **kwargs: Any):
        if path == logs[0]:
            raise PermissionError("fixture unreadable log")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", read)
    assert trainable._avg_batch_from_worker_logs(dirs, {}) == 0
