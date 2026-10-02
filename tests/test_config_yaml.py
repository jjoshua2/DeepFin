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
