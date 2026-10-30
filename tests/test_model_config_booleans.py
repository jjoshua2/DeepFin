"""RS2 model subset: exercise only the real YAML loader and flatten boundary."""
from pathlib import Path

import pytest

from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults, load_yaml_file

_FLAGS = (
    "use_qk_rmsnorm", "use_deepnorm", "use_dynamic_relations",
    "policy_dynamic_relations", "categorical_head_coupled", "enable_policy_sf_head",
)


def _load(tmp_path: Path, text: str):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return flatten_run_config_defaults(load_yaml_file(path))


@pytest.mark.parametrize("key", _FLAGS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("token", ['"false"', '"true"', "0", "1", "0.5", "[]", "{}"])
def test_model_flags_reject_nonboolean_yaml(tmp_path, key, nested, token):
    text = f"model:\n  {key}: {token}\n" if nested else f"{key}: {token}\n"
    with pytest.raises(ValueError, match="must be a boolean") as exc:
        _load(tmp_path, text)
    assert key in str(exc.value)


@pytest.mark.parametrize("key", _FLAGS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("value", [False, True])
def test_model_boolean_yaml_preserves_values(tmp_path, key, nested, value):
    token = str(value).lower()
    text = f"model:\n  {key}: {token}\n" if nested else f"{key}: {token}\n"
    assert _load(tmp_path, text)[key] is value


@pytest.mark.parametrize("key", _FLAGS)
@pytest.mark.parametrize("nested", [False, True])
def test_model_null_is_omitted(tmp_path, key, nested):
    text = f"model:\n  {key}: null\n" if nested else f"{key}: null\n"
    assert key not in _load(tmp_path, text)


@pytest.mark.parametrize("key", _FLAGS)
@pytest.mark.parametrize(("token", "expected"), [("true", True), ("false", False), ("null", None)])
def test_nested_model_override_precedes_flat_validation(tmp_path, key, token, expected):
    config = _load(tmp_path, f'{key}: "false"\nmodel:\n  {key}: {token}\n')
    if expected is None:
        assert key not in config
    else:
        assert config[key] is expected


@pytest.mark.parametrize("value", [False, True])
def test_smolgen_alias_and_existing_booleans_unchanged(tmp_path, value):
    token = str(value).lower()
    config = _load(tmp_path, f"no_smolgen: {token}\nmodel:\n  use_smolgen: {token}\n  use_nla: {token}\n")
    assert config["no_smolgen"] is not value
    assert config["use_nla"] is value


@pytest.mark.parametrize("token", ['"false"', "0", "null"])
def test_smolgen_alias_keeps_existing_strict_contract(tmp_path, token):
    with pytest.raises(ValueError, match=r"model\.use_smolgen must be a boolean"):
        _load(tmp_path, f"model:\n  use_smolgen: {token}\n")
