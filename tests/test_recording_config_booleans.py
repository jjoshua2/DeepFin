"""RS2 recording/encoding subset: YAML strings must not flip worker flags."""
from pathlib import Path

import pytest

from chess_anti_engine.model import ModelConfig
from chess_anti_engine.tune.distributed_runtime import build_recommended_worker
from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults, load_yaml_file

_FLAGS = (
    "record_lc0_root_input", "history_rep_fix", "record_relations",
    "record_prior_top1", "record_dense_sf_policy", "record_sf_p0_policy",
    "record_sf_p0_regret", "record_fast_ply_value",
)


def _load(tmp_path: Path, key: str, token: str, nested: bool):
    path = tmp_path / "config.yaml"
    setting = f"selfplay:\n  {key}: {token}\n" if nested else f"{key}: {token}\n"
    # The existing sparse-supervision guard still applies when dense recording
    # is genuinely false; this is a valid paired configuration, not a bypass.
    path.write_text(setting + "train:\n  sf_policy_sparse_ce: true\n")
    return flatten_run_config_defaults(load_yaml_file(path))


@pytest.mark.parametrize("key", _FLAGS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("token", ['"false"', '"true"', "0", "1"])
def test_recording_flags_reject_nonboolean_yaml(tmp_path, key, nested, token):
    with pytest.raises(ValueError, match="must be a boolean") as exc:
        _load(tmp_path, key, token, nested)
    assert key in str(exc.value)


@pytest.mark.parametrize("key", _FLAGS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("value", [False, True])
def test_real_booleans_reach_recommended_worker_unchanged(tmp_path, key, nested, value):
    config = _load(tmp_path, key, str(value).lower(), nested)
    reco = build_recommended_worker(
        config=config, model_cfg=ModelConfig(), sf_nodes=1, mcts_simulations=1,
    )
    assert config[key] is value
    assert reco[key] is value


@pytest.mark.parametrize("key", _FLAGS)
def test_null_recording_overrides_keep_existing_defaults(tmp_path, key):
    config = _load(tmp_path, key, "null", True)
    assert key not in config
    actual = build_recommended_worker(
        config=config, model_cfg=ModelConfig(), sf_nodes=1, mcts_simulations=1,
    )
    baseline = build_recommended_worker(
        config={}, model_cfg=ModelConfig(), sf_nodes=1, mcts_simulations=1,
    )
    assert actual[key] == baseline[key]
