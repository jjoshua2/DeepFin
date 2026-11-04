from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from ray.tune.experiment import Trial

_spec = importlib.util.spec_from_file_location(
    "gpbt_gap_contract_test", Path(__file__).parents[1] / "chess_anti_engine/tune/gpbt.py",
)
assert _spec is not None
assert _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


@pytest.mark.parametrize(
    ("scores", "expected_lr", "expected_span", "velocity", "resample", "finished"),
    [
        pytest.param([0.0, 1.0], 0.5, 1.0, 0.0, False, (), id="two-wide-gap"),
        pytest.param([0.0, 1e-9], 0.5, 1e-9, 0.0, False, (), id="two-small-gap"),
        pytest.param([0.0, 1e-12], 0.2 + 3e-13, 1.0, 0.0, False, (), id="range-at-guard"),
        pytest.param([0.0, 1e-13], 0.2 + 3e-14, 1.0, 0.0, False, (), id="range-below-guard"),
        pytest.param([0.0, 0.0], 0.2, 1.0, 0.0, False, (), id="tie-no-eligible-donor"),
        pytest.param([0.0, 3.0, 1.0, 2.0], 0.5, 3.0, 0.0, False, (), id="four-extremes"),
        pytest.param([1.0, 2.0, 0.0, 3.0], 0.3, 3.0, 0.0, False, (), id="four-interior"),
        pytest.param([1e-8, 2e-8, 0.0, 3e-8], 0.3, 3e-8, 0.0, False, (), id="four-scaled"),
        pytest.param([1.0, 2.0, 0.0, 100.0], 0.203, 100.0, 0.0, False, (), id="four-outlier"),
        pytest.param([0.0, 1.0, 100.0], 0.5, 1.0, 0.0, False, (2,), id="finished-excluded"),
        pytest.param([0.0, 1.0, None], 0.5, 1.0, 0.0, False, (), id="unscored-excluded"),
        pytest.param([0.0, 1e-13], 0.3 + 3e-14, 1.0, 0.2, False, (), id="momentum-not-scaled"),
        pytest.param([0.0, 1e-13], 0.7, 1.0, 0.0, True, (), id="resample-not-scaled"),
    ],
)
def test_relative_range_scaling_contract(
    monkeypatch: pytest.MonkeyPatch, scores: list[float | None], expected_lr: float,
    expected_span: float, velocity: float, resample: bool, finished: tuple[int, ...],
) -> None:
    """Characterize existing finite-score policy, not statistical noise handling."""
    scheduler = _module.GPBTPairwiseScheduler(
        metric="score", mode="max", perturbation_interval=2,
        hyperparam_mutations={"lr": [0.1, 0.9]}, hyperparam_bounds={"lr": [0.0, 1.0]},
        quantile_fraction=0.5, resample_probability=float(resample), log_config=False,
    )
    controller: Any = SimpleNamespace(search_alg=None)
    trials = []
    for i, score in enumerate(scores):
        trial = Trial("stub", trial_id=f"t{i}", config={"lr": 0.2 if i == 0 else 0.8}, stub=True)
        trial.set_status(Trial.RUNNING)
        scheduler.on_trial_add(controller, trial)
        scheduler._trial_state[trial].last_score = score
        if i in finished:
            trial.set_status(Trial.TERMINATED)
        trials.append(trial)
    recipient, donor = trials[:2]
    lower, upper = scheduler._quantiles()
    assert recipient in lower
    assert donor in upper
    eligible = scheduler._select_pairwise_donor(trial=recipient, upper_quantile=[donor])
    assert (eligible is donor) == (scores[0] != scores[1])
    scheduler._pairwise_velocity[recipient.trial_id]["lr"] = velocity
    monkeypatch.setattr(_module.random, "random", lambda: 0.5)
    monkeypatch.setattr(_module.random, "uniform", lambda *_args: 0.7)
    # The tie case calls the config method directly to characterize its algebra;
    # selection above rejects a tied donor, so this is not an actual exploit.
    config, _ = scheduler._get_new_config(recipient, donor)
    assert scheduler._live_score_span() == pytest.approx(expected_span, rel=1e-14, abs=0.0)
    assert config["lr"] == pytest.approx(expected_lr, rel=0.0, abs=1e-15)
    assert recipient.config == {"lr": 0.2}
    assert donor.config == {"lr": 0.8}
    if resample:
        assert scheduler._pairwise_velocity[recipient.trial_id]["lr"] == velocity
    else:
        assert scheduler._pairwise_velocity[recipient.trial_id]["lr"] == pytest.approx(
            expected_lr - 0.2, rel=0.0, abs=1e-15,
        )
