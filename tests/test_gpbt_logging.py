from __future__ import annotations

import copy
import importlib.util
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from ray.tune.experiment import Trial

_spec = importlib.util.spec_from_file_location(
    "gpbt_logging_test", Path(__file__).parents[1] / "chess_anti_engine/tune/gpbt.py",
)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def _fixture(tmp_path: Path, recipient_lr: float, donor_lr: float, log_config: bool):
    scheduler = _module.GPBTPairwiseScheduler(
        metric="score", mode="max", perturbation_interval=2,
        hyperparam_mutations={"lr": [0.1, 0.9]}, hyperparam_bounds={"lr": [0.0, 1.0]},
        resample_probability=0.0, log_config=log_config,
    )
    donor = Trial("stub", trial_id="donor", config={"lr": donor_lr}, stub=True)
    recipient = Trial("stub", trial_id="recipient", config={"lr": recipient_lr}, stub=True)
    pauses = []
    controller: Any = SimpleNamespace(
        search_alg=None, get_live_trials=lambda: [donor, recipient],
        pause_trial=lambda trial, should_checkpoint: pauses.append((trial, should_checkpoint)),
    )
    for trial in (donor, recipient):
        trial.set_status(Trial.RUNNING)
        trial.storage = cast(Any, SimpleNamespace(experiment_driver_staging_path=str(tmp_path)))
        scheduler.on_trial_add(controller, trial)
    ds, rs = scheduler._trial_state[donor], scheduler._trial_state[recipient]
    ds.last_score, rs.last_score = 1.0, 0.0
    ds.last_checkpoint = "checkpoint-marker"
    ds.last_result = {"training_iteration": 2}
    return scheduler, controller, recipient, donor, pauses


@pytest.mark.parametrize(("recipient_lr", "donor_lr"), [(0.2, 0.8), (0.8, 0.2)])
@pytest.mark.parametrize("log_config", [True, False])
def test_exploit_logs_recipient_transition_without_changing_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    recipient_lr: float, donor_lr: float, log_config: bool,
) -> None:
    scheduler, controller, recipient, donor, pauses = _fixture(
        tmp_path, recipient_lr, donor_lr, log_config,
    )
    recipient_before, donor_before = copy.deepcopy(recipient.config), copy.deepcopy(donor.config)
    monkeypatch.setattr(_module.random, "random", lambda: 0.5)
    # The existing numeric algorithm, not the logging, determines this midpoint.
    expected = (recipient_lr + donor_lr) / 2
    with caplog.at_level(logging.INFO, logger="ray.tune.schedulers.pbt"):
        scheduler._checkpoint_or_exploit(recipient, controller, [donor], [recipient])
    explore = next(record.message for record in caplog.records if "[Explore]" in record.message)
    assert f"lr : {recipient_lr} --- (pairwise(gap=1.000)) --> {expected}" in explore
    assert recipient.config == {"lr": expected}
    assert donor.config == donor_before
    assert pauses == [(recipient, False)]
    assert scheduler._pairwise_velocity[recipient.trial_id]["lr"] == pytest.approx(expected - recipient_lr)
    result = recipient.latest_checkpoint_result
    assert result is not None
    assert result.checkpoint == "checkpoint-marker"
    assert result.metrics == {"training_iteration": 2}
    assert scheduler._num_perturbations == 1
    assert scheduler._recipient_config_for_summary is None
    path = tmp_path / "gpbt_pairwise_global.txt"
    if log_config:
        payload = json.loads(path.read_text())
        assert payload["recipient_config"] == recipient_before
        assert payload["donor_config"] == donor_before
        assert payload["new_config"] == recipient.config
        assert payload["recipient_trial_id"] == "recipient"
        assert payload["donor_trial_id"] == "donor"
        assert json.loads((tmp_path / "gpbt_pairwise_recipient.txt").read_text()) == payload
        # Ray's compatibility policy keeps its original donor/checkpoint lineage.
        assert json.loads((tmp_path / "pbt_global.txt").read_text())[-2:] == [donor_before, recipient.config]
    else:
        assert not path.exists()


def test_exploit_summary_context_cleared_after_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scheduler, controller, recipient, donor, _ = _fixture(tmp_path, 0.2, 0.8, False)

    def fail(*_args):
        raise RuntimeError("explore failed")

    monkeypatch.setattr(scheduler, "_get_new_config", fail)
    with pytest.raises(RuntimeError, match="explore failed"):
        scheduler._exploit(controller, recipient, donor)
    assert scheduler._recipient_config_for_summary is None
    assert scheduler._summarize_hyperparam_changes({"lr": 0.8}, {"lr": 0.9}, {}) == "lr : 0.8 -----> 0.9\n"


def test_summary_preserves_missing_donor_parameter_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    scheduler, controller, recipient, donor, pauses = _fixture(tmp_path, 0.2, 0.8, True)
    recipient.config["recipient_only"] = 1
    scheduler._hyperparam_mutations["recipient_only"] = [1, 2]
    monkeypatch.setattr(_module.random, "random", lambda: 0.5)
    # Ray's later tag construction rejects missing mutation keys. Logging
    # must not introduce an earlier assertion or hide the existing error.
    with (
        caplog.at_level(logging.INFO, logger="ray.tune.schedulers.pbt"),
        pytest.raises(KeyError, match="recipient_only"),
    ):
        scheduler._exploit(controller, recipient, donor)
    assert pauses == []
    assert scheduler._recipient_config_for_summary is None
    payload = json.loads((tmp_path / "gpbt_pairwise_global.txt").read_text())
    assert payload["recipient_config"] == {"lr": 0.2, "recipient_only": 1}
    assert payload["new_config"] == {"lr": 0.5}


def test_clone_summary_preserves_donor_shape(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    scheduler, controller, recipient, donor, pauses = _fixture(tmp_path, 0.2, 0.8, True)
    recipient.config["lr"] = {"nested": 0.2}
    with caplog.at_level(logging.INFO, logger="ray.tune.schedulers.pbt"):
        scheduler._exploit(controller, recipient, donor)
    assert recipient.config == {"lr": 0.8}
    assert pauses == [(recipient, False)]
    payload = json.loads((tmp_path / "gpbt_pairwise_global.txt").read_text())
    assert payload["recipient_config"] == {"lr": {"nested": 0.2}}
    assert payload["donor_config"] == {"lr": 0.8}


def test_missing_recipient_parameter_keeps_donor_clone(tmp_path: Path) -> None:
    scheduler, controller, recipient, donor, pauses = _fixture(tmp_path, 0.2, 0.8, True)
    recipient.config.clear()
    scheduler._exploit(controller, recipient, donor)
    assert recipient.config == donor.config == {"lr": 0.8}
    assert pauses == [(recipient, False)]
    payload = json.loads((tmp_path / "gpbt_pairwise_global.txt").read_text())
    assert payload["recipient_config"] == {}


def test_numeric_resample_logs_recipient_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    scheduler, controller, recipient, donor, _ = _fixture(tmp_path, 0.2, 0.8, False)
    scheduler._resample_probability = 1.0
    monkeypatch.setattr(_module.random, "uniform", lambda *_args: 0.7)
    with caplog.at_level(logging.INFO, logger="ray.tune.schedulers.pbt"):
        scheduler._exploit(controller, recipient, donor)
    assert recipient.config == {"lr": 0.7}
    explore = next(record.message for record in caplog.records if "[Explore]" in record.message)
    assert "lr : 0.2 --- (resample) --> 0.7" in explore
