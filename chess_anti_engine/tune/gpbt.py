from __future__ import annotations

import copy
import json
import logging
import os
import random
from collections import defaultdict
from ray.tune.experiment import Trial
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pbt import SafeFallbackEncoder, _FutureTrainingResult

logger = logging.getLogger(__name__)


class GPBTPairwiseScheduler(PopulationBasedTraining):
    """Ray-native GPBT-PL style scheduler.

    This keeps Ray's current PBT checkpoint/exploit lifecycle, but replaces:
    - donor selection: weighted pairwise choice among stronger trials
    - hyperparameter updates: PSO-inspired pairwise velocity with momentum

    Velocity update (per-param, per-update):
        v = r1 * inertia_weight * old_v + r2 * winner_weight * gap_scale * (donor - recipient)
    Position update:
        theta_new = theta_recipient + v

    where r1, r2 ~ Uniform[0,1] (re-sampled each update), and gap_scale is the
    normalised performance gap between donor and recipient (dampens exploitation
    when the gap is within noise).

    Based on the GPBT-PL paper (Pairwise Learning variant of Guided
    Population-Based Training).
    """

    def __init__(
        self,
        *,
        hyperparam_bounds: dict[str, list[float]] | None = None,
        trial_inertia_weight: float = 1.0,
        trial_winner_weight: float = 1.0,
        # Legacy aliases (ignored if the new names are also set)
        pairwise_lr: float | None = None,
        pairwise_momentum: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._hyperparam_bounds = dict(hyperparam_bounds or {})
        self._inertia_weight = float(trial_inertia_weight)
        self._winner_weight = float(trial_winner_weight)
        self._pairwise_velocity: dict[str, dict[str, float]] = defaultdict(dict)
        self._pending_pairwise_log: dict[str, dict] = {}

    def _live_score_span(self) -> float:
        scores = [
            float(state.last_score)
            for trial, state in self._trial_state.items()
            if state.last_score is not None and not trial.is_finished()
        ]
        if len(scores) < 2:
            return 1.0
        span = max(scores) - min(scores)
        return float(span) if span > 1e-12 else 1.0

    def _select_pairwise_donor(
        self,
        *,
        trial: Trial,
        upper_quantile: list[Trial],
    ) -> Trial | None:
        recipient_state = self._trial_state[trial]
        recipient_score = float(recipient_state.last_score)
        candidates: list[Trial] = []
        weights: list[float] = []
        score_span = self._live_score_span()

        for donor in upper_quantile:
            if donor is trial:
                continue
            donor_state = self._trial_state[donor]
            if donor_state.last_score is None:
                continue
            gap = float(donor_state.last_score) - recipient_score
            if gap <= 0.0:
                continue
            candidates.append(donor)
            weights.append(max(1e-6, gap / score_span))

        if not candidates:
            return None
        return random.choices(candidates, weights=weights, k=1)[0]

    def _pairwise_numeric_update(
        self,
        *,
        trial_id: str,
        param_name: str,
        recipient_value: float,
        donor_value: float,
        gap_scale: float,
    ) -> float:
        prev_vel = float(self._pairwise_velocity[trial_id].get(param_name, 0.0))
        r1 = random.random()
        r2 = random.random()
        velocity = (
            r1 * self._inertia_weight * prev_vel
            + r2 * self._winner_weight * gap_scale * (donor_value - recipient_value)
        )
        self._pairwise_velocity[trial_id][param_name] = float(velocity)
        return float(recipient_value + velocity)

    def _get_new_config(self, trial: Trial, trial_to_clone: Trial):
        new_config = copy.deepcopy(trial_to_clone.config)
        recipient_score = float(self._trial_state[trial].last_score)
        donor_score = float(self._trial_state[trial_to_clone].last_score)
        score_span = self._live_score_span()
        gap_scale = max(0.0, min(1.0, (donor_score - recipient_score) / score_span))
        operations: dict[str, str] = {}

        for param_name, mutation in self._hyperparam_mutations.items():
            if param_name not in new_config or param_name not in trial.config:
                continue

            donor_value = trial_to_clone.config[param_name]
            recipient_value = trial.config[param_name]

            if (
                param_name in self._hyperparam_bounds
                and isinstance(donor_value, (int, float))
                and isinstance(recipient_value, (int, float))
            ):
                lo, hi = self._hyperparam_bounds[param_name]
                if random.random() < self._resample_probability:
                    # Resample uniformly from the full range.  This is the
                    # only escape hatch when all trials converge to the same
                    # value (e.g. all pinned at the old ceiling): pairwise
                    # velocity is zero so the param would never move otherwise.
                    proposed = random.uniform(float(lo), float(hi))
                    operations[param_name] = "resample"
                else:
                    proposed = self._pairwise_numeric_update(
                        trial_id=trial.trial_id,
                        param_name=param_name,
                        recipient_value=float(recipient_value),
                        donor_value=float(donor_value),
                        gap_scale=float(gap_scale),
                    )
                proposed = max(float(lo), min(float(hi), float(proposed)))
                if isinstance(donor_value, int) and not isinstance(donor_value, bool):
                    proposed = int(round(proposed))
                new_config[param_name] = proposed
                if "resample" not in operations.get(param_name, ""):
                    operations[param_name] = f"pairwise(gap={gap_scale:.3f})"
                continue

            if isinstance(mutation, (list, tuple)):
                if random.random() < self._resample_probability:
                    new_config[param_name] = random.choice(list(mutation))
                    operations[param_name] = "pairwise_resample"
                else:
                    new_config[param_name] = donor_value
                    operations[param_name] = "pairwise_clone"
                continue

            new_config[param_name] = donor_value
            operations[param_name] = "pairwise_clone"

        if self._custom_explore_fn:
            new_config = self._custom_explore_fn(new_config)
            assert new_config is not None, "Custom explore fn failed to return new config"

        return new_config, operations

    def _checkpoint_or_exploit(
        self,
        trial: Trial,
        tune_controller,
        upper_quantile: list[Trial],
        lower_quantile: list[Trial],
    ):
        state = self._trial_state[trial]
        if trial in upper_quantile:
            if trial.status == Trial.PAUSED:
                if trial.temporary_state.saving_to and isinstance(
                    trial.temporary_state.saving_to, _FutureTrainingResult
                ):
                    state.last_checkpoint = trial.temporary_state.saving_to
                else:
                    state.last_checkpoint = trial.checkpoint
            else:
                state.last_checkpoint = tune_controller._schedule_trial_save(
                    trial, result=state.last_result
                )
            self._num_checkpoints += 1
        else:
            state.last_checkpoint = None

        if trial not in lower_quantile:
            return

        donor = self._select_pairwise_donor(trial=trial, upper_quantile=upper_quantile)
        if donor is None:
            return

        donor_state = self._trial_state[donor]
        last_checkpoint = donor_state.last_checkpoint

        logger.info(
            "[GPBT-PL] Pairwise exploit: recipient=%s donor=%s gap=%.6f",
            trial.trial_id,
            donor.trial_id,
            float(donor_state.last_score) - float(state.last_score),
        )

        if isinstance(last_checkpoint, _FutureTrainingResult):
            training_result = last_checkpoint.resolve()
            if training_result:
                donor_state.last_result = training_result.metrics
                donor_state.last_checkpoint = training_result.checkpoint
                last_checkpoint = donor_state.last_checkpoint
            else:
                last_checkpoint = None

        if not last_checkpoint:
            logger.info(
                "[gpbt_pl]: no checkpoint for donor trial %s. Skip exploit for %s",
                donor.trial_id,
                trial.trial_id,
            )
            return

        score_span = self._live_score_span()
        score_gap = float(donor_state.last_score) - float(state.last_score)
        gap_scale = max(0.0, min(1.0, score_gap / score_span))
        self._pending_pairwise_log[trial.trial_id] = {
            "recipient_trial_id": str(trial.trial_id),
            "donor_trial_id": str(donor.trial_id),
            "recipient_score": float(state.last_score),
            "donor_score": float(donor_state.last_score),
            "score_gap": float(score_gap),
            "gap_scale": float(gap_scale),
            "inertia_weight": float(self._inertia_weight),
            "winner_weight": float(self._winner_weight),
        }
        self._exploit(tune_controller, trial, donor)

    def _log_config_on_step(
        self,
        trial_state,
        new_state,
        trial: Trial,
        trial_to_clone: Trial,
        new_config: dict,
    ):
        context = self._pending_pairwise_log.pop(str(trial.trial_id), {})
        payload = {
            "scheduler": "gpbt_pl",
            "recipient_trial_name": str(trial_state.orig_tag),
            "donor_trial_name": str(new_state.orig_tag),
            "recipient_iter": int(trial.last_result.get("training_iteration", 0)),
            "donor_iter": int(trial_to_clone.last_result.get("training_iteration", 0)),
            "donor_config": trial_to_clone.config,
            "new_config": new_config,
            **context,
        }

        global_path = os.path.join(trial.local_experiment_path, "gpbt_pairwise_global.txt")
        trial_path = os.path.join(
            trial.local_experiment_path,
            "gpbt_pairwise_" + str(trial.trial_id) + ".txt",
        )
        with open(global_path, "a+", encoding="utf-8") as fh:
            print(json.dumps(payload, cls=SafeFallbackEncoder), file=fh)
        with open(trial_path, "a+", encoding="utf-8") as fh:
            print(json.dumps(payload, cls=SafeFallbackEncoder), file=fh)

        super()._log_config_on_step(trial_state, new_state, trial, trial_to_clone, new_config)

    def debug_string(self) -> str:
        return (
            "GPBTPairwiseScheduler: "
            f"{self._num_checkpoints} checkpoints, "
            f"{self._num_perturbations} perturbs"
        )
