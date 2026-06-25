from __future__ import annotations

from chess_anti_engine.tune.trainable_config_ops import (
    _OPTIMIZER_CONSTRUCTION_KEYS,
    _apply_lr_gamma_weights,
    _play_batch_kwargs,
    _reload_yaml_into_config,
)
from chess_anti_engine.tune.trial_config import TrialConfig


class _FakeOpt:
    def __init__(self) -> None:
        self.gamma = 0.0
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
            "cosmos_gamma": 0.456,
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
    assert trainer.opt.gamma == 0.456
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
        "soda_scope", "soda_start_step", "cosmos_rank",
    }
    value_only = {
        "matrix_lr_multiplier", "matrix_weight_decay", "aux_weight_decay",
        "global_board_preprocess_lr_multiplier", "global_board_preprocess_weight_decay",
        "global_board_adapter_lr_multiplier", "global_board_adapter_weight_decay",
    }
    assert structural <= _OPTIMIZER_CONSTRUCTION_KEYS
    assert not (value_only & _OPTIMIZER_CONSTRUCTION_KEYS)
    # cosmos_rank in particular changes low-rank moment shapes -> must block.
    assert "cosmos_rank" in _OPTIMIZER_CONSTRUCTION_KEYS


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
cosmos_rank: 128
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
    # Optimizer-construction keys are NOT injected (incl. cosmos_rank, which
    # changes low-rank moment shapes).
    for key in ("optimizer", "matrix_optimizer_scope", "weight_decay_mode", "cosmos_rank"):
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
