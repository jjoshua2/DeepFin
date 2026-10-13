from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from ray.train._internal.session import _TrainingResult
from ray.tune.experiment import Trial
from ray.tune.schedulers.pbt import _FutureTrainingResult

_spec = importlib.util.spec_from_file_location(
    "gpbt_pair_test", Path(__file__).parents[1] / "chess_anti_engine/tune/gpbt.py",
)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


class _Ready(_FutureTrainingResult):
    def __init__(self, result: Any) -> None:
        super().__init__(cast(Any, None))
        self.result = result

    def resolve(self, block: bool = True) -> Any:
        del block
        return self.result


@pytest.mark.parametrize("kind", [
    "running", "paused_future", "paused_committed", "fallback",
    "empty_future", "empty_checkpoint", "missing",
])
def test_exploit_preserves_checkpoint_report_pair(kind: str) -> None:
    scheduler = _module.GPBTPairwiseScheduler(
        metric="score", mode="max", perturbation_interval=2,
        hyperparam_mutations={"lr": [0.1, 0.9]}, log_config=False,
    )
    donor = Trial("stub", trial_id="donor", config={"lr": 0.8}, stub=True)
    recipient = Trial("stub", trial_id="recipient", config={"lr": 0.2}, stub=True)
    donor.set_status(Trial.RUNNING)
    recipient.set_status(Trial.RUNNING)
    controller: Any = SimpleNamespace(search_alg=None, get_live_trials=lambda: [donor, recipient])
    scheduler.on_trial_add(controller, donor)
    scheduler.on_trial_add(controller, recipient)
    ds = scheduler._trial_state[donor]
    rs = scheduler._trial_state[recipient]
    ds.last_score, rs.last_score = 1.0, 0.0
    old = _TrainingResult(checkpoint=cast(Any, "marker1"), metrics={"training_iteration": 1})
    current = _TrainingResult(checkpoint=cast(Any, "marker2"), metrics={"training_iteration": 2})
    manager = donor.run_metadata.checkpoint_manager
    assert manager is not None
    manager._latest_checkpoint_result = old
    ds.last_result = {"training_iteration": 2}
    donor.run_metadata.last_result = {"training_iteration": 3}
    scheduled: list[dict] = []

    def save(trial: Trial, result: dict) -> _Ready:
        assert trial is donor
        scheduled.append(dict(result))
        return _Ready(current)

    controller._schedule_trial_save = save
    if kind == "paused_future":
        donor.set_status(Trial.PAUSED)
        donor.temporary_state.saving_to = _Ready(current)
    elif kind == "paused_committed":
        donor.set_status(Trial.PAUSED)
    if kind in {"running", "paused_future", "paused_committed"}:
        scheduler._checkpoint_or_exploit(donor, controller, [donor], [recipient])
        if kind == "running":
            assert scheduled == [{"training_iteration": 2}]
        if kind != "paused_committed":
            ds.last_result = {"training_iteration": 3}
    elif kind == "empty_future":
        ds.last_checkpoint = _Ready(None)
    elif kind == "empty_checkpoint":
        ds.last_checkpoint = _Ready(_TrainingResult(checkpoint=None, metrics={"training_iteration": 2}))
    elif kind == "missing":
        manager._latest_checkpoint_result = None

    paired: list[tuple[Any, Any]] = []
    scheduler._exploit = lambda *_args: paired.append(
        (ds.last_checkpoint, ds.last_result["training_iteration"]),
    )
    scheduler._checkpoint_or_exploit(recipient, controller, [donor], [recipient])
    if kind in {"empty_future", "empty_checkpoint", "missing"}:
        assert paired == []
    elif kind in {"running", "paused_future"}:
        assert paired == [("marker2", 2)]
    else:
        assert paired == [("marker1", 1)]
