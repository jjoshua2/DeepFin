from __future__ import annotations

import logging
from pathlib import Path

import pytest

from chess_anti_engine.tune.trainable_config_ops import (
    _OPTIMIZER_CONSTRUCTION_KEYS,
    _apply_lr_gamma_weights,
    _play_batch_kwargs,
    _reload_yaml_into_config,
    reset_yaml_reload_key_tracking,
)
from chess_anti_engine.tune.trial_config import TrialConfig


class _FakeOpt:
    def __init__(self) -> None:
        self.param_groups = [
            {"use_aurora": True},
            {"use_aurora": False},
        ]


class _FakeModel:
    def __init__(self) -> None:
        self.resid_channel_dropout = 0.0
        self.resid_channel_balance_weight = 0.0


class _FakeTrainer:
    def __init__(self) -> None:
        self.opt = _FakeOpt()
        self.model = _FakeModel()
        self.peak_lr_calls: list[tuple[float, bool]] = []
        self.w_policy = 0.0
        self.w_soft = 0.0
        self.w_future = 0.0
        self.w_wdl = 0.0
        self.w_sf_move = 0.0
        self.w_sf_eval = 0.0
        self.w_categorical = 0.0
        self.w_volatility = 0.0
        self.w_sf_volatility = 0.0
        self.w_moves_left = 0.0
        self.sf_wdl_frac = 0.0
        self.search_wdl_frac = 0.0
        self.sf_wdl_conf_power = 0.0
        self.sf_wdl_draw_scale = 0.0
        self.sf_wdl_temperature = 0.0
        self.sf_search_dampen_sf_low = 0.0
        self.sf_search_dampen_sf_high = 0.0
        self.resid_channel_dropout = 0.0
        self.resid_channel_balance_weight = 0.0
        self.lr_release_calls: list[dict[str, object | None]] = []

    def set_peak_lr(self, value: float, *, rescale_current: bool) -> None:
        self.peak_lr_calls.append((value, rescale_current))

    def set_lr_release_config(
        self,
        *,
        cycle_steps: object | None = None,
        release_start_frac: object | None = None,
        min_scale: object | None = None,
        release_shape: object | None = None,
    ) -> None:
        self.lr_release_calls.append(
            {
                "cycle_steps": cycle_steps,
                "release_start_frac": release_start_frac,
                "min_scale": min_scale,
                "release_shape": release_shape,
            }
        )


def test_play_batch_kwargs_preserves_syzygy_adjudication_knobs() -> None:
    tc = TrialConfig.from_dict(
        {
            "syzygy_path": "/tmp/tb",
            "stockfish_syzygy_path": "/tmp/ssd-tb",
            "syzygy_rescore_policy": True,
            "syzygy_adjudicate": True,
            "syzygy_adjudicate_fraction": 0.5,
            "syzygy_in_search": True,
        }
    )

    game = _play_batch_kwargs(tc)["game"]

    assert game.syzygy_path == "/tmp/tb"
    assert game.stockfish_syzygy_path == "/tmp/ssd-tb"
    assert game.syzygy_rescore_policy is True
    assert game.syzygy_adjudicate is True
    assert game.syzygy_adjudicate_fraction == 0.5
    assert game.syzygy_in_search is True


def test_play_batch_kwargs_threads_temperature_and_gumbel_knobs() -> None:
    tc = TrialConfig.from_dict(
        {
            "temperature": 0.7,
            "temperature_after": 0.2,
            "temperature_decay_start_move": 15,
            "temperature_decay_moves": 20,
            "temperature_endgame": 0.1,
            "selfplay_temperature": 0.9,
            "selfplay_temperature_decay_start_move": 12,
            "selfplay_temperature_decay_moves": 18,
            "selfplay_temperature_endgame": 0.05,
            "gumbel_topk": 24,
            "gumbel_c_scale": 0.15,
            "gumbel_scale": 0.8,
            "gumbel_scale_after": 0.1,
            "gumbel_scale_decay_start_move": 10,
            "gumbel_scale_decay_moves": 16,
            "curriculum_gumbel_scale": 0.4,
            "curriculum_gumbel_scale_after": 0.05,
            "curriculum_gumbel_scale_decay_start_move": 8,
            "curriculum_gumbel_scale_decay_moves": 12,
        }
    )

    kwargs = _play_batch_kwargs(tc)
    temp = kwargs["temp"]
    search = kwargs["search"]

    assert temp.temperature == 0.7
    assert temp.after == 0.2
    assert temp.decay_start_move == 15
    assert temp.decay_moves == 20
    assert temp.endgame == 0.1
    assert temp.selfplay_temperature == 0.9
    assert temp.selfplay_decay_start_move == 12
    assert temp.selfplay_decay_moves == 18
    assert temp.selfplay_endgame == 0.05
    assert search.gumbel_topk == 24
    assert search.gumbel_c_scale == 0.15
    assert search.gumbel_scale == 0.8
    assert search.gumbel_scale_after == 0.1
    assert search.gumbel_scale_decay_start_move == 10
    assert search.gumbel_scale_decay_moves == 16
    assert search.curriculum_gumbel_scale == 0.4
    assert search.curriculum_gumbel_scale_after == 0.05
    assert search.curriculum_gumbel_scale_decay_start_move == 8
    assert search.curriculum_gumbel_scale_decay_moves == 12


def test_apply_lr_gamma_weights_syncs_all_trainer_loss_kwargs() -> None:
    trainer = _FakeTrainer()

    _apply_lr_gamma_weights(
        trainer,
        {
            "lr": 0.123,
            "w_policy": 1.1,
            "w_soft": 1.2,
            "w_future": 1.3,
            "w_wdl": 1.4,
            "w_sf_move": 1.5,
            "w_sf_eval": 1.6,
            "w_categorical": 1.7,
            "w_volatility": 1.8,
            "w_sf_volatility": 1.9,
            "w_moves_left": 2.0,
            "sf_wdl_frac": 2.1,
            "search_wdl_frac": 2.2,
            "sf_wdl_conf_power": 2.3,
            "sf_wdl_draw_scale": 2.4,
            "sf_wdl_temperature": 2.5,
            "sf_search_dampen_sf_low": 2.6,
            "sf_search_dampen_sf_high": 2.7,
        },
        rescale_current_lr=True,
    )

    assert trainer.peak_lr_calls == [(0.123, True)]
    assert trainer.w_policy == 1.1
    assert trainer.w_soft == 1.2
    assert trainer.w_future == 1.3
    assert trainer.w_wdl == 1.4
    assert trainer.w_sf_move == 1.5
    assert trainer.w_sf_eval == 1.6
    assert trainer.w_categorical == 1.7
    assert trainer.w_volatility == 1.8
    assert trainer.w_sf_volatility == 1.9
    assert trainer.w_moves_left == 2.0
    assert trainer.sf_wdl_frac == 2.1
    assert trainer.search_wdl_frac == 2.2
    assert trainer.sf_wdl_conf_power == 2.3
    assert trainer.sf_wdl_draw_scale == 2.4
    assert trainer.sf_wdl_temperature == 2.5
    assert trainer.sf_search_dampen_sf_low == 2.6
    assert trainer.sf_search_dampen_sf_high == 2.7


def test_apply_lr_gamma_weights_syncs_lr_release_knobs() -> None:
    trainer = _FakeTrainer()

    _apply_lr_gamma_weights(
        trainer,
        {
            "lr_schedule": "sqrt_release",
            "lr_release_cycle_steps": 0,
            "lr_release_start_frac": 0.8,
            "lr_release_min_scale": 0.2,
            "lr_release_shape": "cosine",
        },
        rescale_current_lr=False,
    )

    assert trainer.lr_release_calls == [
        {
            "cycle_steps": 0,
            "release_start_frac": 0.8,
            "min_scale": 0.2,
            "release_shape": "cosine",
        }
    ]


def test_apply_lr_gamma_weights_preserves_sf_volatility_fallback() -> None:
    trainer = _FakeTrainer()

    _apply_lr_gamma_weights(
        trainer,
        {"w_volatility": 0.33},
        rescale_current_lr=False,
    )

    assert trainer.w_volatility == 0.33
    assert trainer.w_sf_volatility == 0.33


def test_apply_lr_gamma_weights_syncs_aurora_and_channel_knobs() -> None:
    trainer = _FakeTrainer()

    _apply_lr_gamma_weights(
        trainer,
        {
            "aurora_pp_iterations": 3,
            "aurora_pp_beta": 0.25,
            "aurora_polar_steps": 8,
            "aurora_polar_method": "polar_express",
            "aurora_polar_dtype": "fp16",
            "aurora_polar_safety": 1.02,
            "resid_channel_dropout": 0.03,
            "resid_channel_balance_weight": 0.001,
        },
        rescale_current_lr=False,
    )

    aurora_group = trainer.opt.param_groups[0]
    adam_group = trainer.opt.param_groups[1]
    assert aurora_group["aurora_pp_iterations"] == 3
    assert aurora_group["aurora_pp_beta"] == 0.25
    assert aurora_group["aurora_polar_steps"] == 8
    assert aurora_group["aurora_polar_method"] == "polar_express"
    assert aurora_group["aurora_polar_dtype"] == "fp16"
    assert aurora_group["aurora_polar_safety"] == 1.02
    assert "aurora_pp_iterations" not in adam_group
    assert trainer.resid_channel_dropout == 0.03
    assert trainer.model.resid_channel_dropout == 0.03
    assert trainer.resid_channel_balance_weight == 0.001
    assert trainer.model.resid_channel_balance_weight == 0.001


def test_live_yaml_reload_does_not_add_missing_topology_keys(tmp_path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        """
model:
  input_history_encoding: lc0_root
  qkv_projection: split
  embed_dim_by_layer: [256, 384, 320]
  ffn_mult_by_layer: [1.0, 1.25, 1.5]
  smolgen_hidden_channels: 16
  smolgen_hidden_sz: 128
  smolgen_gen_sz: 128
train:
  lr: 0.0007
""",
        encoding="utf-8",
    )
    config = {"lr": 0.0003}

    # Live mid-run reload: a running broker/worker would be out of sync, so
    # topology keys absent from the config must NOT be applied.
    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)

    assert config["lr"] == 0.0007
    assert "input_history_encoding" not in config
    assert "qkv_projection" not in config
    assert "embed_dim_by_layer" not in config
    assert "ffn_mult_by_layer" not in config
    assert "smolgen_hidden_channels" not in config
    assert "smolgen_hidden_sz" not in config
    assert "smolgen_gen_sz" not in config


def test_startup_yaml_reload_propagates_new_worker_topology_flag(tmp_path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        """
selfplay:
  history_rep_fix: true
distributed_inference_slots_per_worker: 3
train:
  lr: 0.0007
""",
        encoding="utf-8",
    )
    # An older restored config that predates history_rep_fix (and the infra key)
    # and pins an existing topology key to a different value.
    config = {"lr": 0.0003, "input_history_encoding": "lc0"}

    # Startup/resume: the broker + workers are (re)built from this config after
    # the overlay, so a worker/infra/selfplay topology key absent from the
    # restored config must be applied from yaml — otherwise a flag introduced
    # after the checkpoint was saved silently defaults off.
    _reload_yaml_into_config(config, str(yaml_path))

    assert config["lr"] == 0.0007
    # The selfplay flag (shape-neutral) and the infra key both propagate.
    assert config["history_rep_fix"] is True
    assert config["distributed_inference_slots_per_worker"] == 3
    # An existing topology key keeps its restored value (the current != v skip
    # is unchanged): yaml does not have it, so it is untouched here.
    assert config["input_history_encoding"] == "lc0"


def test_startup_yaml_reload_does_not_inject_model_topology_keys(tmp_path) -> None:
    # A model-topology key (shape schedule OR encoding identity) absent from an
    # OLDER restored config must NOT be auto-filled from yaml on resume: arch
    # resume (or a no-arch salvage warm-start) would otherwise rebuild the model
    # at the yaml layout — different from the checkpoint tensors (policy
    # 4672<->1858, input 146<->175, a different layer/width schedule) — and the
    # tolerant loader zero-inits the mismatch. They require a deliberate
    # migration, not a silent resume-fill.
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        """
selfplay:
  history_rep_fix: true
model:
  policy_encoding: lc0_1858
  input_extra_features: v2_threats
  input_history_encoding: lc0_root
  use_dynamic_relations: true
  num_layers: 16
  embed_dim_by_layer: [512, 512]
  ffn_mult_by_layer: [4, 4]
  qkv_projection: split
  embed_dim: 512
  num_heads: 8
  use_smolgen: true
train:
  lr: 0.0007
""",
        encoding="utf-8",
    )
    # Restored config predates every model key above (e.g. a legacy az_4672 / v1
    # / legacy-history / narrower checkpoint).
    config = {"lr": 0.0003}

    _reload_yaml_into_config(config, str(yaml_path))

    assert config["lr"] == 0.0007
    # The safe worker/selfplay flag still propagates (no shape change).
    assert config["history_rep_fix"] is True
    # Neither encoding identity NOR shape model keys are injected — left absent
    # so the checkpoint arch / model defaults stay authoritative. Includes the
    # bare scalar shape fields (embed_dim/num_heads/use_smolgen) that are
    # ModelConfig fields but NOT in _TOPOLOGY_KEYS, so they would otherwise fall
    # through the gate to the unconditional overlay.
    for key in ("policy_encoding", "input_extra_features",
                "input_history_encoding", "use_dynamic_relations",
                "num_layers", "embed_dim_by_layer", "ffn_mult_by_layer",
                "qkv_projection", "embed_dim", "num_heads", "use_smolgen"):
        assert key not in config, f"{key} must not be auto-filled on resume"


def test_optimizer_construction_keys_cover_structural() -> None:
    # Lock the structural-vs-value split so a newly-added optimizer STRUCTURE
    # knob can't silently slip back into the resume-fillable set (drift trap).
    # Structural keys (change optimizer class / param grouping / moment shape)
    # MUST be blocked; per-group value knobs (lr/wd multipliers) must NOT be —
    # changing those on resume is as safe as changing lr.
    structural = {
        "optimizer", "matrix_optimizer_scope", "weight_decay_mode",
        "soda_scope", "soda_start_step",
    }
    value_only = {
        "matrix_lr_multiplier", "matrix_weight_decay", "aux_weight_decay",
        "global_board_preprocess_lr_multiplier", "global_board_preprocess_weight_decay",
        "global_board_adapter_lr_multiplier", "global_board_adapter_weight_decay",
    }
    assert structural <= _OPTIMIZER_CONSTRUCTION_KEYS
    assert not (value_only & _OPTIMIZER_CONSTRUCTION_KEYS)
    # soda_start_step in particular changes the per-group anchor schedule -> must block.
    assert "soda_start_step" in _OPTIMIZER_CONSTRUCTION_KEYS


def test_startup_yaml_reload_does_not_inject_optimizer_construction_keys(tmp_path) -> None:
    # The Trainer/optimizer is built from config BEFORE checkpoint restore, so
    # backfilling optimizer-construction keys absent from an older restored
    # config would build a different optimizer than the saved moments fit ->
    # Trainer.load reinitializes state. They must not be startup-auto-filled.
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        """
optimizer: aurora
matrix_optimizer_scope: mlp_out
weight_decay_mode: decoupled
soda_start_step: 128
num_samples: 4
train:
  lr: 0.0007
""",
        encoding="utf-8",
    )
    config = {"lr": 0.0003}

    _reload_yaml_into_config(config, str(yaml_path))

    assert config["lr"] == 0.0007
    # Safe infra key still propagates.
    assert config["num_samples"] == 4
    # Optimizer-construction keys are NOT injected (incl. soda_start_step, which
    # changes the per-group weight-decay anchor schedule).
    for key in ("optimizer", "matrix_optimizer_scope", "weight_decay_mode", "soda_start_step"):
        assert key not in config, f"{key} must not be auto-filled on resume"


def test_live_yaml_reload_does_not_propagate_new_worker_topology_flag(tmp_path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        """
selfplay:
  history_rep_fix: true
train:
  lr: 0.0007
""",
        encoding="utf-8",
    )
    config = {"lr": 0.0003}

    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)

    assert config["lr"] == 0.0007
    assert "history_rep_fix" not in config


def test_yaml_reload_does_not_overwrite_existing_topology_key_in_either_mode(tmp_path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        """
selfplay:
  history_rep_fix: true
""",
        encoding="utf-8",
    )

    # The existing value differs from yaml; the current != v skip must hold for
    # both startup and live reload.
    startup_config = {"history_rep_fix": False}
    _reload_yaml_into_config(startup_config, str(yaml_path))
    assert startup_config["history_rep_fix"] is False

    live_config = {"history_rep_fix": False}
    _reload_yaml_into_config(live_config, str(yaml_path), live_reload=True)
    assert live_config["history_rep_fix"] is False


def test_yaml_reload_does_not_change_existing_layer_schedules(tmp_path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        """
model:
  num_layers: 4
  embed_dim_by_layer: [256, 384, 384, 320]
  ffn_mult_by_layer: [1.0, 1.25, 1.5]
train:
  lr: 0.0007
""",
        encoding="utf-8",
    )
    config = {
        "lr": 0.0003,
        "num_layers": 3,
        "embed_dim_by_layer": (256, 256, 256),
        "ffn_mult_by_layer": (1.0, 1.0, 1.0),
    }

    _reload_yaml_into_config(config, str(yaml_path))

    assert config["lr"] == 0.0007
    assert config["num_layers"] == 3
    assert config["embed_dim_by_layer"] == (256, 256, 256)
    assert config["ffn_mult_by_layer"] == (1.0, 1.0, 1.0)


def test_live_yaml_reload_does_not_change_existing_lr_schedule(tmp_path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        """
train:
  lr_schedule: sqrt_release
  lr_release_cycle_steps: 0
""",
        encoding="utf-8",
    )
    config = {
        "lr_schedule": "cosine",
        "lr_release_cycle_steps": 1000,
    }

    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)

    assert config["lr_schedule"] == "cosine"
    assert config["lr_release_cycle_steps"] == 0


def test_startup_yaml_reload_can_change_existing_lr_schedule(tmp_path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        """
train:
  lr_schedule: sqrt_release
  lr_release_cycle_steps: 0
""",
        encoding="utf-8",
    )
    config = {
        "lr_schedule": "cosine",
        "lr_release_cycle_steps": 1000,
    }

    _reload_yaml_into_config(config, str(yaml_path))

    assert config["lr_schedule"] == "sqrt_release"
    assert config["lr_release_cycle_steps"] == 0


def test_yaml_reload_can_add_missing_lr_schedule_for_resume(tmp_path) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        """
train:
  lr_schedule: sqrt_release
""",
        encoding="utf-8",
    )
    config: dict[str, object] = {}

    _reload_yaml_into_config(config, str(yaml_path))

    assert config["lr_schedule"] == "sqrt_release"


# Positive move temperatures conflict with either armed target-only knob.
# Whether a one-key perturbation creates that conflict depends on the input
# configuration: main's template leaves both knobs off, while a live experiment
# may arm them. Neither case changes these keys' live-reloadability.
_TARGET_ONLY_MOVE_TEMPERATURES = frozenset({
    "selfplay_temperature",
    "selfplay_temperature_endgame",
    "temperature",
    "temperature_after",
    "temperature_endgame",
})


def test_restart_required_keys_match_the_reloader(
    tmp_path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Re-derive the provenance answer by RUNNING the reloader, not by restating it.

    ``restart_required_config_keys`` is composed from the same frozensets
    ``_reload_yaml_into_config`` branches on, so a test that compared it to
    those sets would agree by construction and prove nothing. This drives every
    key in the production yaml through an actual live reload with a changed
    value and asks which ones came back unchanged.

    That answer is what tells a reader whether ``params.json`` (the launch
    config) or the live yaml is authoritative for a given key -- rl_loop_audit
    J5. A key that silently moves between the two sides shows up here as a
    mismatch rather than as a wrong number in someone's audit six weeks later.

    The edited yaml keeps the production file's NESTED shape: the validator
    rejects unknown root-level keys, and a rejected reload leaves every value
    untouched -- which would make every key look restart-required and the
    assertion below pass for the wrong reason.

    ⚑ THAT HAZARD HAS A SECOND SOURCE, AND NESTING DOES NOT COVER IT. A
    CROSS-KEY validator can be tripped by a perturbation that is individually
    legal, so the document the loop writes is invalid for a reason the key
    itself is innocent of. The old value then survives and reads as
    "restart-required" -- a FALSE POSITIVE of exactly the kind this file
    exists to catch, pointed at the instrument instead of the code. So the
    loop now asks the reloader WHICH thing it did: a per-key decline is an
    observation, a whole-reload rejection is a void reading and is excluded and
    pinned by name for the tested configuration. Answering it
    the other way -- declaring the keys restart-required -- would make the
    suite green by writing down something false about production.
    """
    import copy

    import yaml as _yaml

    from chess_anti_engine.tune.trainable_config_ops import (
        dead_config_keys,
        restart_required_config_keys,
    )
    from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file

    production = Path(__file__).resolve().parents[1] / "configs" / "pbt2_small.yaml"
    raw = load_yaml_file(str(production))
    flat = flatten_run_config_defaults(raw)
    target_only_armed = (
        max(0, int(flat.get("gumbel_target_max_visit_cap", 0))) > 0
        or bool(flat.get("gumbel_target_untempered_prior", False))
    )
    expected_invalid = _TARGET_ONLY_MOVE_TEMPERATURES if target_only_armed else frozenset()
    # Where each key lives in the nested document, so the edit round-trips
    # through the validator instead of being rejected wholesale.
    section_of: dict[str, str | None] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            for sub in value:
                section_of[sub] = key
        else:
            section_of[key] = None
    # PB2-searched keys are preserved for a reason that belongs to the trial's
    # own bounds, not to the key, so they are the caller's job to exclude.
    searched = {k.removeprefix("pb2_bounds_") for k in flat if k.startswith("pb2_bounds_")}

    def _mutate(value: object) -> object:
        if isinstance(value, bool):
            return not value
        if isinstance(value, (int, float)):
            return type(value)(value + 1)
        if isinstance(value, str):
            return value + "_CHANGED"
        if isinstance(value, list) and value:
            return [*value, value[-1]]
        return value

    edited = tmp_path / "cfg.yaml"
    observed_restart_required: set[str] = set()
    exercised: set[str] = set()
    rejected_whole_reload: set[str] = set()
    for key in sorted(section_of):
        if key.startswith("pb2_bounds_") or key in searched or key not in flat:
            continue
        value = flat[key]
        changed = _mutate(value)
        if changed == value:  # None / dict / empty list: nothing to observe
            continue
        doc = copy.deepcopy(raw)
        section = section_of[key]
        if section is None:
            doc[key] = changed
        else:
            doc[section][key] = changed
        edited.write_text(_yaml.safe_dump(doc), encoding="utf-8")
        config = dict(flat)
        caplog.clear()
        with caplog.at_level(
            logging.WARNING, logger="chess_anti_engine.tune.trainable_config_ops",
        ):
            _reload_yaml_into_config(config, str(edited), live_reload=True)
        # The reloader reports a whole-document rejection separately from a
        # per-key decline. Only the latter is an observation about `key`.
        if any(r.getMessage().startswith("YAML reload failed")
               for r in caplog.records):
            assert config == flat, f"a rejected document changed config while testing {key}"
            rejected_whole_reload.add(key)
            continue
        exercised.add(key)
        if config[key] == value:
            observed_restart_required.add(key)

    # Only keys with a value that can be perturbed are observable; the yaml
    # carries a few keys declared with no value at all (`kind:`), and those say
    # nothing either way.
    # DEAD keys are the fourth class and are excluded here by the same rule
    # `restart_required_config_keys`'s docstring states: the reloader declines
    # them exactly like a restart-required key, but a restart REFUSES them
    # rather than applying them, so reporting them in this set would send an
    # operator into the one action that turns a silent no-op into a crash.
    observed_restart_required -= dead_config_keys()

    # Pin invalid perturbations for this input, independently of the observed
    # warnings. A different template can arm the guard without changing either
    # the validator or which keys need a restart.
    assert rejected_whole_reload == expected_invalid, (
        "the set of keys whose one-at-a-time perturbation invalidates the whole "
        f"document moved: now-void={sorted(rejected_whole_reload - expected_invalid)} "
        f"now-observable={sorted(expected_invalid - rejected_whole_reload)}"
    )
    # ...and they are excluded because the reading is void, NOT because they
    # need a restart. If one ever genuinely becomes restart-required, this
    # fails and the exclusion has to be re-argued rather than inherited.
    assert not (restart_required_config_keys() & rejected_whole_reload), sorted(
        restart_required_config_keys() & rejected_whole_reload
    )

    declared = restart_required_config_keys() & exercised
    assert observed_restart_required == declared, (
        "restart_required_config_keys() disagrees with what the reloader does; "
        f"reloader-only={sorted(observed_restart_required - declared)} "
        f"declared-only={sorted(declared - observed_restart_required)}"
    )
    # Guard against the degenerate pass where the validator rejected every
    # reload (which reads as "nothing is live-reloadable") or nothing ran.
    assert len(exercised) > 100, sorted(exercised)
    assert len(observed_restart_required) < len(exercised) // 2
    assert "opening_fen_dole_per_iter" not in declared, \
        "the key that made params.json look stale must be live-reloadable"


@pytest.mark.parametrize("temperature_key", sorted(_TARGET_ONLY_MOVE_TEMPERATURES))
@pytest.mark.parametrize(("target_key", "armed_value"), [
    ("gumbel_target_max_visit_cap", 5),
    ("gumbel_target_untempered_prior", True),
])
def test_target_only_temperature_conflict_rejects_the_whole_reload(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
    temperature_key: str, target_key: str, armed_value: int | bool,
) -> None:
    """Exercise the armed and unarmed worlds regardless of production defaults."""
    import yaml

    from chess_anti_engine.utils import flatten_run_config_defaults

    doc = {
        "selfplay": {**dict.fromkeys(_TARGET_ONLY_MOVE_TEMPERATURES, 0), target_key: armed_value},
        "train": {"w_policy": 1},
    }
    config = flatten_run_config_defaults(doc)
    original = dict(config)
    yaml_path = tmp_path / "target_only.yaml"
    doc["selfplay"][temperature_key] = 1
    doc["train"]["w_policy"] = 2
    yaml_path.write_text(yaml.safe_dump(doc), encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.tune.trainable_config_ops"):
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert config == original, "the invalid document must not even change unrelated weights"
    assert any(
        r.getMessage().startswith("YAML reload failed")
        and "positive move temperature" in r.getMessage()
        for r in caplog.records
    )

    # Fixing the conflicting temperature permits the entire document to apply.
    caplog.clear()
    doc["selfplay"][temperature_key] = 0
    yaml_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert config["w_policy"] == 2
    assert config[target_key] == armed_value
    assert not any(r.getMessage().startswith("YAML reload failed") for r in caplog.records)

    # A positive temperature is live-reloadable when the target-only knob is off.
    doc["selfplay"][target_key] = 0
    doc["selfplay"][temperature_key] = 1
    yaml_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert config[temperature_key] == 1
    assert config[target_key] == 0


# --- audit T4: a deletion from the live yaml reverts nothing, silently -----


def _reload_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if "no longer set by" in r.getMessage()
    ]


def _write_cfg(path: Path, *, body: str) -> None:
    path.write_text(body, encoding="utf-8")


def test_live_reload_names_every_key_the_yaml_stopped_setting(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The reloader is ADD/UPDATE-only, so removing a line cannot revert the
    value the trial is running. That is a behaviour this PR deliberately does
    NOT change -- but it now says so, naming the key and the value still in
    effect (audit T4)."""
    reset_yaml_reload_key_tracking()
    yaml_path = tmp_path / "cfg.yaml"
    _write_cfg(yaml_path, body="train:\n  lr: 0.0007\n  rebuild_sf_targets: true\n")
    config: dict = {}
    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert config["rebuild_sf_targets"] is True

    _write_cfg(yaml_path, body="train:\n  lr: 0.0007\n")
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.tune.trainable_config_ops"):
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)

    # Semantics unchanged: the deleted key keeps the value it was running.
    assert config["rebuild_sf_targets"] is True
    warnings = _reload_warnings(caplog)
    assert len(warnings) == 1, warnings
    assert "rebuild_sf_targets" in warnings[0]
    assert "delete does not revert" in warnings[0]
    assert "True" in warnings[0]


def test_live_reload_reports_a_nulled_key_as_a_deletion(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """`key: null` is the other spelling of "turn this off".
    ``flatten_run_config_defaults`` drops None before the reloader sees it, so
    it is indistinguishable from a delete -- and must be reported as one."""
    reset_yaml_reload_key_tracking()
    yaml_path = tmp_path / "cfg.yaml"
    _write_cfg(yaml_path, body="train:\n  lr: 0.0007\n  rebuild_sf_targets: true\n")
    config: dict = {}
    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)

    _write_cfg(yaml_path, body="train:\n  lr: 0.0007\n  rebuild_sf_targets: null\n")
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.tune.trainable_config_ops"):
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)

    assert config["rebuild_sf_targets"] is True
    assert any("rebuild_sf_targets" in m for m in _reload_warnings(caplog))


def test_live_reload_is_silent_when_no_key_was_removed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """NEGATIVE CONTROL. An unchanged reload, a value CHANGE and an ADD must
    all stay quiet: a warning that fires on ordinary operation is one an
    operator learns to scroll past."""
    reset_yaml_reload_key_tracking()
    yaml_path = tmp_path / "cfg.yaml"
    _write_cfg(yaml_path, body="train:\n  lr: 0.0007\n  rebuild_sf_targets: true\n")
    config: dict = {}
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.tune.trainable_config_ops"):
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
        _write_cfg(
            yaml_path,
            body="train:\n  lr: 0.0009\n  rebuild_sf_targets: true\n  batch_size: 512\n",
        )
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)

    assert config["lr"] == 0.0009
    assert config["batch_size"] == 512
    assert _reload_warnings(caplog) == []


def test_a_rejected_reload_does_not_move_the_deletion_baseline(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The validator is all-or-nothing: an unknown key rejects the WHOLE
    overlay. The key set that a later deletion is measured against must
    therefore come from the last SUCCESSFUL parse, or the deletion an operator
    makes while fixing the broken yaml is never reported."""
    reset_yaml_reload_key_tracking()
    yaml_path = tmp_path / "cfg.yaml"
    _write_cfg(yaml_path, body="train:\n  lr: 0.0007\n  rebuild_sf_targets: true\n")
    config: dict = {}
    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)

    _write_cfg(
        yaml_path,
        body="train:\n  lr: 0.0007\n  rebuild_sf_targets: true\n  totally_unknown_key: 1\n",
    )
    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)

    _write_cfg(yaml_path, body="train:\n  lr: 0.0007\n")
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.tune.trainable_config_ops"):
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)

    assert any("rebuild_sf_targets" in m for m in _reload_warnings(caplog))


def test_a_key_the_trial_never_carried_is_not_reported_as_kept(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Only a key with a value STILL IN EFFECT is worth a warning. A
    restart-required key the reloader declined never entered ``config``, so its
    removal from the yaml kept nothing."""
    reset_yaml_reload_key_tracking()
    yaml_path = tmp_path / "cfg.yaml"
    _write_cfg(yaml_path, body="train:\n  lr: 0.0007\nmodel:\n  num_layers: 8\n")
    config: dict = {}
    _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert "num_layers" not in config

    _write_cfg(yaml_path, body="train:\n  lr: 0.0007\n")
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.tune.trainable_config_ops"):
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)

    assert _reload_warnings(caplog) == []


def test_the_startup_reload_seeds_the_baseline_without_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """A startup reload has no previous reload to compare against, so it must
    only RECORD -- and the first live reload after it must still be able to
    report a key removed since launch."""
    reset_yaml_reload_key_tracking()
    yaml_path = tmp_path / "cfg.yaml"
    _write_cfg(yaml_path, body="train:\n  lr: 0.0007\n  rebuild_sf_targets: true\n")
    config: dict = {}
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.tune.trainable_config_ops"):
        _reload_yaml_into_config(config, str(yaml_path))
    assert _reload_warnings(caplog) == []

    _write_cfg(yaml_path, body="train:\n  lr: 0.0007\n")
    with caplog.at_level(logging.WARNING, logger="chess_anti_engine.tune.trainable_config_ops"):
        _reload_yaml_into_config(config, str(yaml_path), live_reload=True)
    assert any("rebuild_sf_targets" in m for m in _reload_warnings(caplog))
