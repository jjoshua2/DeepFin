"""Pin the lc0 positive control to the TRAINER production is running.

⚑⚑ THE SAME DEFECT AS ``lc0_control_arch``, ONE AXIS OVER, AND IT OUTLIVED THE
ARCHITECTURE FIX BY HALF A DAY.

The arm's premise is "our EXACT net and trainer, on someone else's data".
``lc0_control_arch`` moved the ARCHITECTURE reference off the in-tree
``configs/pbt2_small.yaml`` and onto the live file, because ``main`` has
committed nothing to that file since the live branch diverged. The TRAINER
reference was left pointing at the stale copy, and
``tests/test_lc0_control_config.py`` said so in its own docstring while
continuing to assert against it. Measured 2026-08-16, that gap was THIRTEEN
``trainer_kwargs_from_config`` entries (control -> live)::

    categorical_target_params    blend_frac 0.0  -> 0.69 (search 0.0 -> 0.31)
    lr_T0                        999999          -> 5000
    lr_T_mult                    1               -> 2
    rebuild_categorical_target   False           -> True
    search_wdl_frac              0.7             -> 0.31
    sf_target_params             sf_policy_score_mode wdl -> cp
    sf_wdl_frac                  0.0             -> 0.69
    w_categorical                0.3             -> 1.0
    w_sf_eval                    0.1             -> 0.0
    w_sf_move                    0.02            -> 0.05
    w_sf_own                     0.1             -> 0.0
    w_sf_own_regret              0.7             -> 0.0
    warmup_steps                 72              -> 1000

⚑ AND THE HEADLINE CONSEQUENCE WAS A VALUE NOBODY HAD RE-DERIVED. The control's
``search_wdl_frac: 0.70`` was justified by "it keeps ``game_frac`` at
production's 0.30". Live's blend is ``0.69 + 0.31 = 1.00``, i.e. **``game_frac``
0.00**. The 0.30 being protected was ``main``'s number; the whole choice was
downstream of a stale reference, and no test could see it because every test
compared the control to the same stale file.

So the trainer claim gets the same instrument the architecture claim got:

1. ``live_production_config_path()`` (shared with ``lc0_control_arch``) reads
   ``$CHESS_LIVE_PRODUCTION_CONFIG``. The repo is public; the live path is
   supplied by whoever launches the arm, never committed. The DRIVERS resolve
   it and pass it in, so this module's verdict never depends on the caller's
   shell.
2. When that names a readable yaml, the control is judged against **that
   file's** realized ``trainer_kwargs_from_config``.
3. Otherwise it is judged against ``LIVE_TRAINER_PIN`` below — a recorded
   measurement, dated, with the branch and commit it was taken on — and the
   provenance string says the live file was NOT read.
4. The allowed differences are ``LC0_TRAINER_DEVIATIONS``, a mapping from kwarg
   NAME to the REASON it is allowed to differ. A set would let an entry be
   added without one; the reason is the part that goes stale, so it is the part
   the type system carries.

⚑ ``lr_T0`` / ``lr_T_mult`` are in the pin at 5000 / 2 and NEITHER FILE SETS
THEM. Both run ``lr_schedule: sqrt_release``, and those two are read only by
the ``CosineAnnealingWarmRestarts`` branch, which is never constructed — the
live file deletes them with exactly that comment. They are pinned at the
``trainer_kwargs_from_config`` DEFAULTS both sides realize. Keeping a dead knob
in the pin is deliberate: this module compares what the trainer is
CONSTRUCTED with, and a value that is dead today is one ``lr_schedule`` edit
away from being live.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chess_anti_engine.eval.lc0_control_arch import LIVE_FILE_UNREAD
from chess_anti_engine.train.trainer import trainer_kwargs_from_config

# ── the recorded live trainer ────────────────────────────────────────────────
#
# Measured 2026-08-16 by running `trainer_kwargs_from_config` over the LIVE
# working tree's `configs/pbt2_small.yaml`.
# ⚑ Regenerate with `scripts/lc0_control_arch_pin.py --axis trainer`; do NOT
# hand-edit an entry to make a test pass.
LIVE_TRAINER_PIN: dict[str, Any] = {
    "recorded": "2026-08-16",
    "live_branch": "ops/live-20260725",
    "live_commit": "66ec9f74a",
    "provenance": (
        "bt4heads bundle promoted 2026-08-15 (w_categorical 0.0 -> 1.0, "
        "rebuild_categorical_target on, categorical blend 0.69/0.31); the "
        "SF-teacher heads zeroed; warmup 1000; sf_policy_score_mode cp"
    ),
    "kwargs": {
        "accum_steps": 1,
        "adjusted_wdl_regret_cap": 0.0,
        "adjusted_wdl_regret_scale": 1.0,
        "adjusted_wdl_regret_source": "sum",
        "aurora_cuda_graphs": True,
        "aurora_polar_dtype": "fp16",
        "aurora_polar_method": "polar_express",
        "aurora_polar_safety": 1.01,
        "aurora_polar_steps": 8,
        "aurora_pp_beta": 0.25,
        "aurora_pp_iterations": 3,
        "aurora_uw_floor": 0.0,
        "aux_weight_decay": 0.0001,
        "categorical_target_params": {
            "blend_frac": 0.69,
            "search_blend_frac": 0.31,
            "num_bins": 32,
            "sigma": 0.04,
        },
        "compile_mode": "max-autotune",
        "device": "cuda",
        "fdp_king_safety": None,
        "fdp_mobility": None,
        "fdp_outposts": None,
        "fdp_pawns": None,
        "fdp_pins": None,
        "feature_dropout_p": 0.0,
        "global_board_adapter_lr_multiplier": 1.0,
        "global_board_adapter_weight_decay": 0.0,
        "global_board_preprocess_lr_multiplier": 1.0,
        "global_board_preprocess_weight_decay": 0.0,
        "lr": 3e-05,
        "lr_T0": 5000,
        "lr_T_mult": 2,
        "lr_eta_min": 1e-05,
        "lr_release_cycle_steps": 0,
        "lr_release_min_scale": 0.1,
        "lr_release_shape": "sqrt",
        "lr_release_start_frac": 0.8,
        "lr_schedule": "sqrt_release",
        "matrix_lr_multiplier": 20.0,
        "matrix_optimizer_scope": "mlp_out",
        "matrix_weight_decay": 0.0001,
        "moves_left_max_plies": 450,
        "optimizer": "aurora",
        "policy_target_temp": 1.0,
        "rebuild_categorical_target": True,
        "rebuild_sf_targets": False,
        "resid_channel_balance_weight": 0.0,
        "resid_channel_dropout": 0.0,
        "search_wdl_frac": 0.31,
        "sf_policy_sparse_ce": False,
        "sf_search_dampen_sf_high": 0.0,
        "sf_search_dampen_sf_low": 0.0,
        "sf_target_params": {
            "sf_policy_temp": 0.012,
            "sf_policy_label_smooth": 0.01,
            "sf_wdl_use_cp_logistic": True,
            "sf_wdl_cp_slope": 0.006,
            "sf_wdl_cp_draw_width": 120.0,
            "sf_policy_score_mode": "cp",
            "sf_policy_cp_temp": 16.2,
        },
        "sf_wdl_conf_power": 1.0,
        "sf_wdl_draw_scale": 0.55,
        "sf_wdl_frac": 0.69,
        "sf_wdl_temperature": 1.0,
        "soda_scope": "decay",
        "soda_start_step": 0,
        "soft_policy_min_tv": 0.0,
        "swa_freq": 50,
        "swa_start": -1,
        "use_adjusted_wdl_target": False,
        "use_amp": True,
        "use_compile": True,
        "w_categorical": 1.0,
        "w_future": 0.01,
        "w_moves_left": 0.02,
        "w_policy": 1.0,
        "w_sf_eval": 0.0,
        "w_sf_move": 0.05,
        "w_sf_own": 0.0,
        "w_sf_own_regret": 0.0,
        "w_sf_volatility": 0.05,
        "w_soft": 1.0,
        "w_volatility": 0.1,
        "w_wdl": 1.0,
        "warmup_lr_start": None,
        "warmup_steps": 1000,
        "wdl_terminal_outcome_full_plies": 2,
        "wdl_terminal_outcome_plies": 0,
        "wdl_terminal_outcome_sf_frac": 0.0,
        "weight_decay_mode": "weight_decay",
        "zclip_alpha": 0.97,
        "zclip_clip_factor": 1.0,
        "zclip_max_norm": 6.5,
        "zclip_z_thresh": 2.0,
    },
}

# ── the ONLY trainer kwargs the control may differ on, and WHY ───────────────
#
# ⚑ A mapping, not a set: an allow-list entry without a stated reason is how a
# diff test decays into a rubber stamp. Every entry here has to survive being
# read out loud next to the live value.
#
# ⚑ Both entries are the SAME decision — lc0 rows carry no `sf_wdl`, so the
# live 0.69 SF share has to go somewhere — and they are listed separately
# because they fail separately: dropping only the first gives
# `sf 0.0 / search 0.31`, which trains 0.69 of the value target on the raw
# outcome and is the failure `value_blend_guard` exists to catch.
LC0_TRAINER_DEVIATIONS: dict[str, str] = {
    "sf_wdl_frac": (
        "LIVE 0.69 -> 0.0. lc0 rows carry no `sf_wdl`, and `losses.py` falls "
        "the SF component back to the RAW ONE-HOT GAME OUTCOME on an "
        "unlabelled row with no error and no warning — so live's value would "
        "land 0.69 of the target on the deep outcome. Measured through the "
        "real `compute_loss` on converted rows: outcome share 0.6900. "
        "`sf_wdl_frac_floor` is zeroed with it (not a trainer kwarg) so the "
        "PID ramp cannot reassert it; see `assert_pid_cannot_reassert_sf_wdl`."
    ),
    "search_wdl_frac": (
        "LIVE 0.31 -> 1.0. The freed 0.69 SF share, handed to lc0's own root "
        "best_q/best_d, which the converter stores as `search_wdl` and which "
        "every converted row carries. This holds `game_frac` at LIVE's 0.00 "
        "EXACTLY — live's sf 0.69 + search 0.31 sums to 1.00, so production "
        "puts NO weight on the raw game outcome and neither does this arm."
    ),
}


class ControlTrainerDrift(RuntimeError):
    """The control would train with a recipe production is not running."""


def _normalize(value: Any) -> Any:
    """Dataclass kwargs (``SfTargetParams``, ``CategoricalTargetParams``) -> dicts.

    ⚑ Recursive and by FIELD, not ``repr``. Comparing reprs would make the pin
    sensitive to a dataclass gaining a field with a default — which is a real
    trainer change and should fail — but ALSO to a rename that changes nothing,
    and would report both as one opaque string. Expanding to a dict names the
    field that moved.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _normalize(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Mapping):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def trainer_kwargs_signature(flat_config: Mapping[str, Any]) -> dict[str, Any]:
    """The realized ``Trainer`` kwargs of a flattened run config, JSON-able.

    ⚑ ``trainer_kwargs_from_config``, not a hand-listed subset. The question is
    "what would `Trainer(...)` be constructed with", and any key this function
    reads is part of the answer — including the ones no yaml sets, which reach
    the trainer as defaults and are exactly the ones a hand-written list would
    omit. ``lr_T0`` is such a key today.
    """
    return {k: _normalize(v) for k, v in trainer_kwargs_from_config(dict(flat_config)).items()}


def trainer_kwargs_drift(
    control: Mapping[str, Any], reference: Mapping[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Every kwarg where the two signatures disagree, as ``{key: (ctrl, live)}``."""
    return {
        key: (control.get(key, "<absent>"), reference.get(key, "<absent>"))
        for key in sorted(set(control) | set(reference))
        if control.get(key, "<absent>") != reference.get(key, "<absent>")
    }


def _reference_signature(
    live_config: Path | None, pinned: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """The trainer signature to judge against, and where it came from."""
    if live_config is None:
        origin = (
            "the recorded trainer pin"
            if pinned is LIVE_TRAINER_PIN else "a CALLER-SUPPLIED trainer pin"
        )
        return dict(pinned["kwargs"]), (
            f"{origin} ({pinned.get('recorded', '<undated>')}, "
            f"{pinned.get('live_branch', '<no branch>')} "
            f"{pinned.get('live_commit', '<no commit>')}) — no live config was "
            f"given, so {LIVE_FILE_UNREAD}"
        )
    from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

    flat = flatten_run_config_defaults(load_yaml_file(str(live_config)))
    return trainer_kwargs_signature(flat), f"the LIVE file {live_config}"


def assert_control_matches_live_trainer(
    control_flat_config: Mapping[str, Any],
    *,
    live_config: Path | None = None,
    pin: Mapping[str, Any] | None = None,
    allowed: Mapping[str, str] = LC0_TRAINER_DEVIATIONS,
    context: str = "",
) -> str:
    """Raise unless the control trains with PRODUCTION'S recipe, minus ``allowed``.

    ``control_flat_config`` is the FLATTENED config (unlike the architecture
    guard, which takes the raw mapping): the drift that matters here is in the
    realized kwargs, and flattening is what turns ``selfplay.max_plies`` into
    ``moves_left_max_plies``.

    Returns the provenance string of whatever it judged against, so the caller
    can print which instrument fired.

    ⚑ An ALLOWED kwarg that does NOT differ is also a failure. If production
    moves onto the control's value, the deviation has silently become a
    non-deviation and the reason recorded for it is describing nothing — the
    same "a gate that cannot fail" shape one level up. It is reported as a
    separate problem so the two are not confused.
    """
    pinned = dict(pin) if pin is not None else LIVE_TRAINER_PIN
    reference, provenance = _reference_signature(live_config, pinned)
    if not reference:
        raise ControlTrainerDrift(
            f"the reference trainer signature is EMPTY (judged against "
            f"{provenance}){f' [{context}]' if context else ''}. Every "
            "comparison against it is vacuous.",
        )
    control = trainer_kwargs_signature(control_flat_config)
    drift = trainer_kwargs_drift(control, reference)

    problems = [
        f"{key}: control={ctrl!r} live={live!r}"
        for key, (ctrl, live) in drift.items()
        if key not in allowed
    ]
    problems += [
        f"{key}: listed as a deliberate deviation ({allowed[key]}) but it does "
        f"NOT differ — both sides read {control.get(key, '<absent>')!r}"
        for key in sorted(allowed)
        if key not in drift
    ]
    if not problems:
        return provenance
    where = f" [{context}]" if context else ""
    raise ControlTrainerDrift(
        f"the lc0 control's TRAINER is NOT production's{where}. Judged against "
        f"{provenance}. Problems:\n  " + "\n  ".join(problems) + "\n"
        "⚑ The arm's premise is 'our EXACT net and trainer on someone else's "
        "data'. A loss weight, an LR schedule or a warmup that is not "
        "production's makes the held-out SLOPE — the deciding yardstick — a "
        "property of a training recipe we do not run. Deviations belong in "
        "`LC0_TRAINER_DEVIATIONS` WITH A REASON, and a new one is a "
        "training-affecting decision that needs a ledger entry, not a test "
        "edit. If production moved, regenerate the pin with "
        "`scripts/lc0_control_arch_pin.py --axis trainer` and decide whether "
        "the arm follows.",
    )
