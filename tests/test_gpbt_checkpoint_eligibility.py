"""Metadata-only donor decisions: no Ray cluster, model or checkpoint restore."""
from __future__ import annotations

import copy
import importlib.util
import random
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from ray.train._internal.session import _TrainingResult
from ray.tune.experiment import Trial
from ray.tune.schedulers.pbt import _FutureTrainingResult

_spec = importlib.util.spec_from_file_location(
    "gpbt_eligibility_test", Path(__file__).parents[1] / "chess_anti_engine/tune/gpbt.py",
)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.fixture(autouse=True)
def _preserve_random_state() -> Iterator[None]:
    state = random.getstate()
    yield
    random.setstate(state)


class _Ready(_FutureTrainingResult):
    def __init__(self, result: Any) -> None:
        super().__init__(cast(Any, None))
        self.result = result
        self.resolutions = 0

    def resolve(self, block: bool = True) -> Any:
        del block
        self.resolutions += 1
        return self.result


def _fixture(mode: str, metrics: dict, kind: str = "fallback"):
    scheduler = _module.GPBTPairwiseScheduler(
        metric="score", mode=mode, time_attr="training_iteration", perturbation_interval=2,
        hyperparam_mutations={"lr": [0.1, 0.9]}, hyperparam_bounds={"lr": [0.0, 1.0]},
        resample_probability=0.0, log_config=False,
    )
    donor = Trial("stub", trial_id="donor", config={"lr": 0.8}, stub=True)
    recipient = Trial("stub", trial_id="recipient", config={"lr": 0.2}, stub=True)
    pauses = []
    controller: Any = SimpleNamespace(
        search_alg=None, get_live_trials=lambda: [donor, recipient],
        get_trials=lambda: [donor, recipient],
        pause_trial=lambda trial, should_checkpoint: pauses.append((trial, should_checkpoint)),
    )
    for trial in (donor, recipient):
        trial.set_status(Trial.RUNNING)
        scheduler.on_trial_add(controller, trial)
    ds, rs = scheduler._trial_state[donor], scheduler._trial_state[recipient]
    latest = 10.0 if mode == "max" else 1.0
    ds.last_score, rs.last_score = scheduler._metric_op * latest, scheduler._metric_op * 5.0
    ds.last_result = {"training_iteration": 1, "score": latest}
    rs.last_result = {"training_iteration": 2, "score": 5.0}
    checkpoint = _TrainingResult(checkpoint=cast(Any, "donor-marker"), metrics=metrics)
    donor_manager = donor.run_metadata.checkpoint_manager
    assert donor_manager is not None
    donor_manager._latest_checkpoint_result = checkpoint
    previous = _TrainingResult(
        checkpoint=cast(Any, "recipient-marker"),
        metrics={"training_iteration": 0, "score": 0.0 if mode == "max" else 100.0},
    )
    recipient_manager = recipient.run_metadata.checkpoint_manager
    assert recipient_manager is not None
    recipient_manager._latest_checkpoint_result = previous
    future = _Ready(checkpoint)
    controller._schedule_trial_save = lambda trial, result: future
    if kind in {"running_future", "paused_future", "paused_committed"}:
        if kind.startswith("paused"):
            donor.set_status(Trial.PAUSED)
        if kind == "paused_future":
            donor.temporary_state.saving_to = future
        scheduler._checkpoint_or_exploit(donor, controller, [donor], [recipient])
    elif kind == "cached":
        ds.last_checkpoint, ds.last_result = checkpoint.checkpoint, checkpoint.metrics
    return scheduler, controller, donor, recipient, pauses, previous, future


@pytest.mark.parametrize("kind", ["running_future", "paused_future", "paused_committed", "cached", "fallback"])
@pytest.mark.parametrize("mode", ["max", "min"])
@pytest.mark.parametrize("quality", ["worse", "equal", "older_better", "fresh"])
def test_transfer_requires_advantage_of_actual_checkpoint(kind: str, mode: str, quality: str) -> None:
    saved = {
        "max": {"worse": 1.0, "equal": 5.0, "older_better": 7.0, "fresh": 10.0},
        "min": {"worse": 9.0, "equal": 5.0, "older_better": 3.0, "fresh": 1.0},
    }[mode][quality]
    metrics = {"training_iteration": 1 if quality == "fresh" else 0, "score": saved}
    scheduler, controller, donor, recipient, pauses, previous, future = _fixture(mode, metrics, kind)
    ds = scheduler._trial_state[donor]
    latest_score = ds.last_score
    donor_checkpoint_before = ds.last_checkpoint
    donor_result_before = ds.last_result
    recipient_config = copy.deepcopy(recipient.config)
    donor_config = copy.deepcopy(donor.config)
    scheduler._pairwise_velocity[recipient.trial_id]["lr"] = 0.1
    velocity = copy.deepcopy(scheduler._pairwise_velocity)
    scheduler._checkpoint_or_exploit(recipient, controller, [donor], [recipient])
    assert ds.last_score == latest_score
    assert donor.config == donor_config
    assert future.resolutions == (1 if kind in {"running_future", "paused_future"} else 0)
    if quality in {"worse", "equal"}:
        assert scheduler._num_perturbations == 0
        assert pauses == []
        assert recipient.latest_checkpoint_result is previous
        assert recipient.config == recipient_config
        assert scheduler._pairwise_velocity == velocity
        assert ds.last_checkpoint is donor_checkpoint_before
        assert ds.last_result is donor_result_before
    else:
        assert scheduler._num_perturbations == 1
        assert pauses == [(recipient, False)]
        result = recipient.latest_checkpoint_result
        assert result is not None
        assert result.checkpoint == "donor-marker"
        assert result.metrics == metrics
        context = scheduler._pending_pairwise_log[recipient.trial_id]
        assert context["donor_score"] == latest_score
        assert context["checkpoint_score"] == scheduler._metric_op * saved
        assert context["checkpoint_score_gap"] == scheduler._metric_op * (saved - 5.0)


@pytest.mark.parametrize("metrics", [
    {}, {"score": None}, {"score": float("nan")}, {"score": float("inf")},
    {"score": -float("inf")}, {"score": "7"}, {"score": []},
])
def test_unusable_checkpoint_metric_does_not_borrow_latest_score(metrics: dict) -> None:
    scheduler, controller, donor, recipient, pauses, previous, _ = _fixture("max", metrics)
    scheduler._checkpoint_or_exploit(recipient, controller, [donor], [recipient])
    assert scheduler._num_perturbations == 0
    assert pauses == []
    assert recipient.latest_checkpoint_result is previous
    assert recipient.config == {"lr": 0.2}
    assert not scheduler._pairwise_velocity
    assert scheduler._trial_state[donor].last_score == 10.0


def test_real_report_callbacks_compare_against_current_recipient_not_old_checkpoint() -> None:
    scheduler, controller, donor, recipient, pauses, previous, _ = _fixture("max", {"score": 1.0})
    scheduler.on_trial_result(controller, donor, {"training_iteration": 1, "score": 10.0})
    scheduler.on_trial_result(controller, recipient, {"training_iteration": 2, "score": 5.0})
    assert scheduler._num_perturbations == 0
    assert pauses == []
    assert recipient.latest_checkpoint_result is previous
    assert scheduler._trial_state[recipient].last_score == 5.0


@pytest.mark.parametrize(("mode", "latest", "saved", "target", "accepted"), [
    ("max", -1.0, -9.0, -5.0, False), ("max", -1.0, -3.0, -5.0, True),
    ("min", -9.0, -1.0, -5.0, False), ("min", -9.0, -7.0, -5.0, True),
])
def test_signed_scores_follow_mode(mode: str, latest: float, saved: float, target: float, accepted: bool) -> None:
    scheduler, controller, donor, recipient, _, _, _ = _fixture(mode, {"score": saved})
    scheduler._trial_state[donor].last_score = scheduler._metric_op * latest
    scheduler._trial_state[recipient].last_score = scheduler._metric_op * target
    scheduler._checkpoint_or_exploit(recipient, controller, [donor], [recipient])
    assert scheduler._num_perturbations == int(accepted)


def test_rejected_checkpoint_consumes_one_selection_draw_and_no_exploration_randomness() -> None:
    scheduler, controller, donor, recipient, _, previous, _ = _fixture("max", {"score": 1.0})
    original_state = random.getstate()
    try:
        random.seed(4729)
        expected = random.Random(4729)
        expected.choices([donor], weights=[1.0], k=1)
        scheduler._checkpoint_or_exploit(recipient, controller, [donor], [recipient])
        assert random.getstate() == expected.getstate()
        assert recipient.latest_checkpoint_result is previous
    finally:
        random.setstate(original_state)


@pytest.mark.parametrize(("metrics", "accepted"), [
    ({"eval": {"score": 7.0}}, True), ({"eval/score": 7.0}, True),
    ({"eval": {}}, False), ({"eval": {"other": 7.0}}, False),
    ({"eval": {"score": None}}, False), ({"eval": {"score": float("nan")}}, False),
    # Match Ray's scheduler flattening, including nested-value collision order.
    ({"eval/score": 9.0, "eval": {"score": 1.0}}, False),
])
def test_checkpoint_metrics_follow_ray_flattened_report_schema(metrics: dict, accepted: bool) -> None:
    scheduler, controller, donor, recipient, _, _, _ = _fixture("max", metrics)
    scheduler._metric = "eval/score"
    original_metrics = copy.deepcopy(metrics)
    scheduler._checkpoint_or_exploit(recipient, controller, [donor], [recipient])
    assert scheduler._num_perturbations == int(accepted)
    assert metrics == original_metrics
    if accepted:
        result = recipient.latest_checkpoint_result
        assert result is not None
        assert result.metrics is metrics


@pytest.mark.parametrize("mode", ["max", "min"])
@pytest.mark.parametrize("saved", [-1.0, 0.0, 1.0])
def test_zero_recipient_uses_strict_mode_adjusted_advantage(mode: str, saved: float) -> None:
    scheduler, controller, donor, recipient, _, _, _ = _fixture(mode, {"score": saved})
    scheduler._trial_state[donor].last_score = 2.0
    scheduler._trial_state[recipient].last_score = 0.0
    scheduler._checkpoint_or_exploit(recipient, controller, [donor], [recipient])
    assert scheduler._num_perturbations == int(scheduler._metric_op * saved > 0.0)


@pytest.mark.parametrize("target", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_recipient_cannot_admit_transfer(target: float) -> None:
    scheduler, controller, donor, recipient, pauses, previous, _ = _fixture("max", {"score": 7.0})
    scheduler._trial_state[recipient].last_score = target
    scheduler._checkpoint_or_exploit(recipient, controller, [donor], [recipient])
    assert scheduler._num_perturbations == 0
    assert pauses == []
    assert recipient.latest_checkpoint_result is previous


def test_skipped_choice_does_not_try_another_eligible_donor(monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler, controller, donor, recipient, pauses, previous, future = _fixture(
        "max", {"score": 1.0}, "running_future",
    )
    other = Trial("stub", trial_id="other", config={"lr": 0.7}, stub=True)
    other.set_status(Trial.RUNNING)
    scheduler.on_trial_add(controller, other)
    state = scheduler._trial_state[other]
    state.last_score = 9.0
    state.last_result = {"score": 8.0}
    other_future = _Ready(_TrainingResult(checkpoint=cast(Any, "other-marker"), metrics=state.last_result))
    state.last_checkpoint = other_future
    choices = []

    def choose(candidates, *, weights, k):
        choices.append((candidates, weights, k))
        return [donor]

    monkeypatch.setattr(_module.random, "choices", choose)
    scheduler._checkpoint_or_exploit(recipient, controller, [donor, other], [recipient])
    assert len(choices) == 1
    assert choices[0][0] == [donor, other]
    assert future.resolutions == 1
    assert other_future.resolutions == 0
    assert scheduler._num_perturbations == 0
    assert pauses == []
    assert recipient.latest_checkpoint_result is previous


def test_rejection_preserves_resume_metadata_but_keeps_existing_nonupper_cache_invalidation() -> None:
    scheduler, controller, donor, recipient, pauses, previous, _ = _fixture("max", {"score": 1.0})
    rs = scheduler._trial_state[recipient]
    rs.last_checkpoint = "obsolete-scheduler-cache"
    current_report = rs.last_result
    scheduler._checkpoint_or_exploit(recipient, controller, [donor], [recipient])
    assert rs.last_checkpoint is None
    assert rs.last_result is current_report
    assert recipient.latest_checkpoint_result is previous
    assert recipient.config == {"lr": 0.2}
    assert pauses == []
    assert not scheduler._pairwise_velocity


@pytest.mark.parametrize("saved", [1.0, 7.0])
def test_future_resolution_precedes_any_transfer_side_effect(
    monkeypatch: pytest.MonkeyPatch, saved: float,
) -> None:
    scheduler, controller, donor, recipient, _, previous, future = _fixture(
        "max", {"score": saved}, "running_future",
    )
    events = []
    resolve = future.resolve
    get_config = scheduler._get_new_config

    def resolve_checked(block: bool = True):
        assert recipient.latest_checkpoint_result is previous
        assert recipient.config == {"lr": 0.2}
        assert not scheduler._pairwise_velocity
        events.append("resolve")
        return resolve(block)

    def get_config_checked(target, selected):
        assert events == ["resolve"]
        assert scheduler._trial_state[selected].last_result["score"] > 5.0
        events.append("explore")
        return get_config(target, selected)

    def pause_checked(target, should_checkpoint):
        assert target is recipient
        assert not should_checkpoint
        assert events == ["resolve", "explore"]
        events.append("pause")

    monkeypatch.setattr(future, "resolve", resolve_checked)
    monkeypatch.setattr(scheduler, "_get_new_config", get_config_checked)
    controller.pause_trial = pause_checked
    scheduler._checkpoint_or_exploit(recipient, controller, [donor], [recipient])
    assert events == (["resolve", "explore", "pause"] if saved > 5.0 else ["resolve"])


def test_cached_checkpoint_without_report_is_not_transferable() -> None:
    scheduler, controller, donor, recipient, pauses, previous, _ = _fixture("max", {}, "cached")
    scheduler._trial_state[donor].last_result = None
    scheduler._checkpoint_or_exploit(recipient, controller, [donor], [recipient])
    assert scheduler._num_perturbations == 0
    assert pauses == []
    assert recipient.latest_checkpoint_result is previous
    assert scheduler._trial_state[donor].last_result is None
