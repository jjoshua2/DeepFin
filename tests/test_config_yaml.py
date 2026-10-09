import pytest

from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults


@pytest.mark.parametrize("section", ["stockfish", "selfplay", "train", "model", "tune"])
@pytest.mark.parametrize("bad_value", [None, False, [], "temperature: 0", 7])
def test_non_mapping_yaml_sections_fail_instead_of_being_ignored(section, bad_value):
    with pytest.raises(ValueError, match=rf"YAML '{section}:' section must be a mapping/dict"):
        flatten_run_config_defaults({section: bad_value})


@pytest.mark.parametrize("section", ["stockfish", "selfplay", "train", "model", "tune"])
def test_empty_yaml_sections_remain_valid(section):
    assert flatten_run_config_defaults({section: {}}) == {}


@pytest.mark.parametrize("bad_value", ["false", "true", 0, 1, None])
def test_model_use_smolgen_rejects_non_boolean_values(bad_value):
    with pytest.raises(ValueError, match="must be a boolean"):
        flatten_run_config_defaults({"model": {"use_smolgen": bad_value}})


@pytest.mark.parametrize(("use_smolgen", "no_smolgen"), [(True, False), (False, True)])
def test_model_use_smolgen_boolean_is_inverted_for_cli(use_smolgen, no_smolgen):
    assert flatten_run_config_defaults({"model": {"use_smolgen": use_smolgen}}) == {
        "no_smolgen": no_smolgen,
    }


@pytest.mark.parametrize(
    ("config", "key"),
    [
        ({"no_smolgen": "false"}, "no_smolgen"),
        ({"train": {"no_amp": "false"}}, "no_amp"),
        ({"tune": {"search_smolgen": 0}}, "search_smolgen"),
        ({"model": {"use_nla": "false"}}, "use_nla"),
        ({"salvage_restore_pid_state": "false"}, "salvage_restore_pid_state"),
        ({"salvage_restore_donor_config": "false"}, "salvage_restore_donor_config"),
        ({"salvage_restore_full_trainer_state": "false"}, "salvage_restore_full_trainer_state"),
        ({"salvage_reinit_volatility_heads": "false"}, "salvage_reinit_volatility_heads"),
    ],
)
def test_boolean_run_config_rejects_truthy_non_booleans(config, key):
    with pytest.raises(ValueError, match="must be a boolean") as exc:
        flatten_run_config_defaults(config)
    assert key in str(exc.value)


@pytest.mark.parametrize(
    "config",
    [
        {"no_smolgen": False},
        {"train": {"no_amp": False}},
        {"tune": {"search_smolgen": True}},
        {"model": {"use_nla": False}},
    ],
)
def test_boolean_run_config_accepts_yaml_booleans(config):
    flatten_run_config_defaults(config)
