from __future__ import annotations


# Runtime-mutable trainer attributes sourced from config each iteration.
# Shared by YAML allowlisting, live trainer sync, and salvage-donor overlay.
TRAINER_WEIGHT_KEYS: tuple[str, ...] = (
    "w_policy",
    "w_soft",
    "w_future",
    "w_sf_own",
    "w_sf_own_regret",
  # SF-approved-move probability floor. The WEIGHT only: the four
  # `sf_policy_floor_*` SHAPE keys are deliberately NOT here. This list is the
  # every-iteration live-push AND the salvage-donor overlay, and it pushes by
  # `setattr(trainer, key, float(...))` -- which the shape keys cannot ride,
  # because they are folded into one resolved, validated object at construction.
  # They are classified restart-required instead (`_STARTUP_ONLY_TRIAL_KEYS`),
  # so a live edit warns rather than no-ops.
    "w_sf_policy_floor",
  # SF-shape conditional KL. The WEIGHT only, same split as the floor above:
  # `sf_shape_temp_cp` is folded into a resolved, validated object at Trainer
  # construction and is classified restart-required instead. Being HERE is what
  # puts the term under the holdout ruler -- `eval_ruler.active_loss_terms`
  # hashes the set of non-zero weights read off this tuple, so flipping
  # `w_sf_shape` off 0.0 moves the ruler id and hands the best-model record over
  # instead of freezing it against an objective that gained a term.
    "w_sf_shape",
    "w_wdl",
    "w_sf_move",
    "w_sf_eval",
    "w_categorical",
    "w_volatility",
    "w_sf_volatility",
    "w_moves_left",
    "sf_wdl_frac",
    "search_wdl_frac",
    "sf_wdl_conf_power",
    "sf_wdl_draw_scale",
    "sf_wdl_temperature",
    "sf_search_dampen_sf_low",
    "sf_search_dampen_sf_high",
    "soft_policy_min_tv",
)


def is_inert_dead_config_value(value: object, inert: object) -> bool:
    """Whether *value* is the ONE harmless value for a dead config key.

    ⚑ THE SINGLE PREDICATE. ``trainable_config_ops.reject_dead_config_keys``
    (which raises at startup) and ``scripts/audit_realized_config.py`` (which
    reports) both go through this function rather than each re-deriving "is
    this value live". Two implementations that agree today drift the moment a
    dead key is added whose inert value is not falsey, and then the audit would
    call safe a value the trial refuses to start on -- a guard disagreeing with
    the criterion it guards.

    IT LIVES HERE, in the dependency-free key module, rather than beside the
    dead-key table: ``trainable_config_ops`` pulls in the selfplay package and
    its C extension, and ``classify_config_provenance`` states that it stays
    importable and testable without one. An import inside that function -- let
    alone inside its per-key loop -- would make that contract false.

    ``bool(...)`` FOR BOOLEAN KEYS ONLY: yaml spells the same intent ``true`` /
    ``True`` / ``1`` / ``yes``, and a ``1 != False`` comparison would flag a
    value the consumer treats as identical to the inert one. A NUMERIC dead key
    is compared numerically, because ``bool()`` collapses every non-zero number
    onto one class and would make ``gate_interval: 5`` indistinguishable from
    the inert ``1`` -- a predicate that cannot see the value it exists to
    refuse.
    """
    if isinstance(inert, bool):
        return bool(value) == bool(inert)
    if isinstance(inert, (int, float)) and isinstance(value, (int, float)):
        return float(value) == float(inert)
    return value == inert
