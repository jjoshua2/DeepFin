"""RS2 opening-selector subset: exercise only the real YAML loader and flatten boundary."""
from pathlib import Path

import pytest

from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults, load_yaml_file

_FLAGS = (
    "opening_fen_net_side_to_move", "opening_fen_selfplay_only",
)


def _load(tmp_path: Path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return flatten_run_config_defaults(load_yaml_file(path))


@pytest.mark.parametrize("key", _FLAGS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("token", ['"false"', '"true"', "0", "1", "0.5", "[]", "{}"])
def test_selfplay_flags_reject_nonboolean_yaml(tmp_path, key, nested, token):
    text = f"selfplay:\n  {key}: {token}\n" if nested else f"{key}: {token}\n"
    with pytest.raises(ValueError, match="must be a boolean") as exc:
        _load(tmp_path, text)
    assert key in str(exc.value)


@pytest.mark.parametrize("key", _FLAGS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("value", [False, True])
def test_selfplay_boolean_yaml_preserves_values(tmp_path, key, nested, value):
    token = str(value).lower()
    text = f"selfplay:\n  {key}: {token}\n" if nested else f"{key}: {token}\n"
    assert _load(tmp_path, text)[key] is value


@pytest.mark.parametrize("key", _FLAGS)
@pytest.mark.parametrize("nested", [False, True])
def test_selfplay_null_is_omitted(tmp_path, key, nested):
    text = f"selfplay:\n  {key}: null\n" if nested else f"{key}: null\n"
    assert key not in _load(tmp_path, text)


@pytest.mark.parametrize("key", _FLAGS)
@pytest.mark.parametrize(("token", "expected"), [("true", True), ("false", False), ("null", None)])
def test_nested_selfplay_override_precedes_flat_validation(tmp_path, key, token, expected):
    config = _load(tmp_path, f'{key}: "false"\nselfplay:\n  {key}: {token}\n')
    if expected is None:
        assert key not in config
    else:
        assert config[key] is expected


def test_omitted_opening_flags_keep_consumer_defaults(tmp_path):
    config = _load(tmp_path, "selfplay: {}\n")
    assert not any(key in config for key in _FLAGS)


@pytest.mark.parametrize("key", _FLAGS)
@pytest.mark.parametrize("value", [False, True])
def test_invalid_nested_opening_flag_overrides_valid_flat(tmp_path, key, value):
    text = f'{key}: {str(value).lower()}\nselfplay:\n  {key}: "false"\n'
    with pytest.raises(ValueError, match="must be a boolean") as exc:
        _load(tmp_path, text)
    assert key in str(exc.value)


