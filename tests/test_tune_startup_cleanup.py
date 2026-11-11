"""Driver ownership starts when a sidecar launcher returns its handle."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from chess_anti_engine.tune import harness


class StartupFailure(RuntimeError):
    pass


def _driver_stubs(
    monkeypatch: pytest.MonkeyPatch, *, fail_at: str | None = None,
    error: BaseException | None = None, server: bool = True, broker: bool = True,
    cleanup_error: str | None = None,
) -> tuple[list[str], object]:
    """Replace all resource acquisition, training and process termination."""
    import ray
    from ray import tune

    calls: list[str] = []
    answer = object()
    failure = error if error is not None else StartupFailure("injected startup failure")

    def operation(name: str, result: Any = None) -> Any:
        def invoke(*_args: object, **_kwargs: object) -> Any:
            calls.append(name)
            if fail_at == name:
                raise failure
            return result
        return invoke

    trainable = ModuleType("chess_anti_engine.tune.trainable")
    monkeypatch.setattr(trainable, "train_trial", object(), raising=False)
    monkeypatch.setitem(sys.modules, trainable.__name__, trainable)
    monkeypatch.setattr(harness, "_patch_ray_artifact_sync_excludes", operation("sync_patch"))
    monkeypatch.setattr(harness, "_launch_distributed_server", operation("server", "server" if server else None))
    monkeypatch.setattr(harness, "_maybe_launch_shared_broker", operation("broker", "broker" if broker else None))
    monkeypatch.setattr(harness, "_build_scheduler_and_param_space", operation("scheduler", ({}, None, None)))
    monkeypatch.setattr(harness, "_cleanup_old_tune_experiments", operation("retention"))
    monkeypatch.setattr(harness, "_find_valid_experiment_state", operation("find_state", Path("state.json")))
    monkeypatch.setattr(harness, "_resolve_resume_param_space", operation("resume_space", {}))
    monkeypatch.setattr(harness, "_hotpatch_scheduler_bounds", operation("scheduler_patch"))
    monkeypatch.setattr(harness, "_make_checkpoint_eviction_idempotent", operation("checkpoint_patch"))
    monkeypatch.setattr(harness, "_make_csv_logger_schema_safe", operation("csv_patch"))
    monkeypatch.setattr(ray, "init", operation("ray_init"))
    monkeypatch.setattr(tune, "with_resources", operation("resources", object()))
    monkeypatch.setattr(tune, "TuneConfig", operation("tune_config", object()))
    monkeypatch.setattr(tune, "RunConfig", operation("run_config", object()))
    monkeypatch.setattr(tune, "CheckpointConfig", operation("checkpoint_config", object()))

    class FakeTuner:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            operation("tuner")()

        @staticmethod
        def restore(*_args: object, **_kwargs: object) -> FakeTuner:
            operation("restore")()
            return object.__new__(FakeTuner)

        def fit(self) -> object:
            return operation("fit", answer)()

    monkeypatch.setattr(tune, "Tuner", FakeTuner)

    def terminate(handle: str | None) -> None:
        if handle is None:
            return
        calls.append(f"cleanup:{handle}")
        if cleanup_error in (handle, "both"):
            raise PermissionError(f"cannot terminate {handle}")

    monkeypatch.setattr(harness, "_terminate_process", terminate)
    return calls, answer


@pytest.mark.parametrize(("stage", "resume"), [
    ("broker", False), ("resources", False), ("scheduler", False),
    ("ray_init", False), ("retention", False), ("tune_config", False),
    ("checkpoint_config", False), ("run_config", False), ("tuner", False),
    ("checkpoint_patch", False), ("csv_patch", False), ("fit", False),
    ("find_state", True), ("resume_space", True), ("restore", True),
    ("scheduler_patch", True), ("checkpoint_patch", True), ("csv_patch", True),
    ("fit", True),
])
def test_every_startup_failure_releases_already_returned_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, resume: bool,
) -> None:
    failure = StartupFailure(stage)
    calls, _ = _driver_stubs(monkeypatch, fail_at=stage, error=failure)
    if resume:
        (tmp_path / "tune").mkdir()
    with pytest.raises(StartupFailure) as caught:
        harness.run_tune(base_config={}, work_dir=tmp_path, num_samples=1, resume=resume)
    assert caught.value is failure
    expected = ["cleanup:server"] if stage == "broker" else ["cleanup:broker", "cleanup:server"]
    assert [call for call in calls if call.startswith("cleanup:")] == expected
    assert calls[-len(expected):] == expected


@pytest.mark.parametrize("stage", ["sync_patch", "server"])
@pytest.mark.parametrize("error", [StartupFailure("early"), KeyboardInterrupt("early"), SystemExit(2)])
def test_failure_before_any_handle_is_returned_does_not_claim_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str, error: BaseException,
) -> None:
    calls, _ = _driver_stubs(monkeypatch, fail_at=stage, error=error)
    with pytest.raises(type(error)) as caught:
        harness.run_tune(base_config={}, work_dir=tmp_path, num_samples=1)
    assert caught.value is error
    assert not any(call.startswith("cleanup:") for call in calls)


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize(("server", "broker"), [(False, False), (False, True), (True, False), (True, True)])
def test_normal_fit_returns_unchanged_result_and_cleans_only_owned_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    resume: bool, server: bool, broker: bool,
) -> None:
    calls, answer = _driver_stubs(monkeypatch, server=server, broker=broker)
    if resume:
        (tmp_path / "tune").mkdir()
    assert harness.run_tune(base_config={}, work_dir=tmp_path, num_samples=1, resume=resume) is answer
    expected = (["cleanup:broker"] if broker else []) + (["cleanup:server"] if server else [])
    assert [call for call in calls if call.startswith("cleanup:")] == expected
    assert ("restore" in calls) is resume
    assert ("retention" in calls) is (not resume)


@pytest.mark.parametrize("error", [KeyboardInterrupt("interrupt"), SystemExit(2)])
@pytest.mark.parametrize("stage", ["broker", "scheduler"])
def test_non_exception_exit_during_startup_still_releases_owned_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException, stage: str,
) -> None:
    calls, _ = _driver_stubs(monkeypatch, fail_at=stage, error=error)
    with pytest.raises(type(error)) as caught:
        harness.run_tune(base_config={}, work_dir=tmp_path, num_samples=1)
    assert caught.value is error
    expected = ["cleanup:server"] if stage == "broker" else ["cleanup:broker", "cleanup:server"]
    assert [call for call in calls if call.startswith("cleanup:")] == expected


@pytest.mark.parametrize("fail_at", [None, "scheduler", "fit"])
@pytest.mark.parametrize("cleanup_error", ["broker", "server", "both"])
def test_cleanup_failure_does_not_prevent_second_cleanup_or_hide_original_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_at: str | None, cleanup_error: str,
) -> None:
    failure = StartupFailure(str(fail_at))
    calls, _ = _driver_stubs(monkeypatch, fail_at=fail_at, error=failure, cleanup_error=cleanup_error)
    with pytest.raises(PermissionError) as caught:
        harness.run_tune(base_config={}, work_dir=tmp_path, num_samples=1)
    assert calls[-2:] == ["cleanup:broker", "cleanup:server"]
    chain: list[BaseException] = []
    context: BaseException | None = caught.value
    while context is not None:
        chain.append(context)
        context = context.__context__
    if fail_at is not None:
        assert chain[-1] is failure
    else:
        assert all(isinstance(exc, PermissionError) for exc in chain)


@pytest.mark.parametrize("key", ["cpus_per_trial", "gpus_per_trial", "max_concurrent_trials"])
def test_real_numeric_configuration_error_releases_handles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, key: str,
) -> None:
    calls, _ = _driver_stubs(monkeypatch)
    with pytest.raises(ValueError, match="invalid"):
        harness.run_tune(base_config={key: "invalid"}, work_dir=tmp_path, num_samples=1)
    assert calls[-2:] == ["cleanup:broker", "cleanup:server"]


@pytest.mark.parametrize("keep_last", [-1, 0, 2, 3])
@pytest.mark.parametrize(("stage", "resume"), [
    ("scheduler", False), ("retention", False), ("tuner", False),
    ("fit", False), ("restore", True), ("fit", True),
])
def test_retention_and_sidecar_cleanup_keep_their_failure_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    keep_last: int, stage: str, resume: bool,
) -> None:
    """Cleanup releases handles, not a rollback of prior experiment pruning."""
    real_retention = harness._cleanup_old_tune_experiments
    failure = StartupFailure(stage)
    calls, _ = _driver_stubs(monkeypatch, fail_at=stage, error=failure)
    tune_dir = tmp_path / "run" / "tune"
    tune_dir.mkdir(parents=True)
    states, trials = [], []
    for stamp in ["2026-01-01", "2026-02-01"]:
        trial = tune_dir / f"train_trial_{stamp}_00000"
        trial.mkdir()
        (trial / "marker").write_text(stamp)
        state = tune_dir / f"experiment_state-{stamp}.json"
        state.write_text(json.dumps({"trial_data": [[json.dumps({
            "relative_logdir": trial.name,
        }), "{}"]]}))
        states.append(state)
        trials.append(trial)
    unrelated = tune_dir / "not-a-trial"
    unrelated.mkdir()
    (unrelated / "marker").write_text("unrelated")
    outside = tmp_path / "other-run" / "tune" / "train_trial_outside"
    outside.mkdir(parents=True)
    (outside / "marker").write_text("outside")

    def prune(**kwargs: Any) -> None:
        calls.append("retention")
        real_retention(**kwargs)
        if stage == "retention":
            raise failure

    monkeypatch.setattr(harness, "_cleanup_old_tune_experiments", prune)
    with pytest.raises(StartupFailure) as caught:
        harness.run_tune(
            base_config={"tune_keep_last_experiments": keep_last},
            work_dir=tune_dir.parent, num_samples=1, resume=resume,
        )
    assert caught.value is failure
    assert calls[-2:] == ["cleanup:broker", "cleanup:server"]
    pruned = not resume and stage != "scheduler" and keep_last > 0
    expected = [keep_last > 2, True] if pruned else [True, True]
    assert [state.exists() for state in states] == expected
    assert [trial.exists() for trial in trials] == expected
    assert ("retention" in calls) == (not resume and stage != "scheduler")
    assert (unrelated / "marker").read_text() == "unrelated"
    assert (outside / "marker").read_text() == "outside"


@pytest.mark.parametrize("artifact", [
    "experiment_state-old.json", "basic-variant-state-old.json",
    "train_trial_old", "pbt_policy_old.txt",
])
@pytest.mark.parametrize("keep_last", [1, "1"])
def test_fresh_retention_one_rejects_existing_artifacts_before_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str, keep_last: int | str,
) -> None:
    calls, _ = _driver_stubs(monkeypatch)
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    existing = tune_dir / artifact
    if artifact.startswith("train_trial_"):
        existing.mkdir()
        existing = existing / "checkpoint-marker"
    existing.write_text("preserve even malformed or orphan artifacts")
    with pytest.raises(ValueError, match="Use a new work directory"):
        harness.run_tune(
            base_config={"tune_keep_last_experiments": keep_last},
            work_dir=tmp_path, num_samples=1,
        )
    assert calls == ["sync_patch"]
    assert existing.read_text() == "preserve even malformed or orphan artifacts"


@pytest.mark.parametrize("layout", ["absent", "empty", "unrelated"])
def test_fresh_retention_one_without_existing_artifacts_can_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str,
) -> None:
    calls, answer = _driver_stubs(monkeypatch)
    tune_dir = tmp_path / "tune"
    if layout != "absent":
        tune_dir.mkdir()
    if layout == "unrelated":
        (tune_dir / "notes.txt").write_text("keep")
        (tune_dir / "unrelated").mkdir()
    assert harness.run_tune(
        base_config={"tune_keep_last_experiments": 1},
        work_dir=tmp_path, num_samples=1,
    ) is answer
    assert "retention" in calls
    assert calls[-2:] == ["cleanup:broker", "cleanup:server"]
    if layout == "unrelated":
        assert (tune_dir / "notes.txt").read_text() == "keep"
        assert (tune_dir / "unrelated").is_dir()


def test_resume_retention_one_keeps_existing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, answer = _driver_stubs(monkeypatch)
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    state = tune_dir / "experiment_state-old.json"
    state.write_text("preserve")
    assert harness.run_tune(
        base_config={"tune_keep_last_experiments": 1},
        work_dir=tmp_path, num_samples=1, resume=True,
    ) is answer
    assert "restore" in calls
    assert "retention" not in calls
    assert state.read_text() == "preserve"
