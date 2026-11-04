"""Worker execution selectors must be native booleans at the YAML boundary."""
from pathlib import Path

import pytest

from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults, load_yaml_file

_WORKER_SELECTOR_KEYS = (
    "distributed_worker_threaded",
    "distributed_worker_inference_fp8",
)


def _load_worker_selector_yaml(tmp_path: Path, text: str):
    path = tmp_path / "worker-selectors.yaml"
    path.write_text(text)
    return flatten_run_config_defaults(load_yaml_file(path))


@pytest.mark.parametrize("key", _WORKER_SELECTOR_KEYS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "token", ['"false"', '"true"', '""', "0", "1", "-1", "0.5", "[]", "{}", "[false]"],
)
def test_worker_selector_rejects_nonboolean_yaml(tmp_path, key, nested, token):
    text = f"tune:\n  {key}: {token}\n" if nested else f"{key}: {token}\n"
    with pytest.raises(ValueError, match="must be a boolean") as exc:
        _load_worker_selector_yaml(tmp_path, text)
    assert key in str(exc.value)


@pytest.mark.parametrize("key", _WORKER_SELECTOR_KEYS)
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("value", [False, True])
def test_worker_selector_preserves_native_boolean(tmp_path, key, nested, value):
    token = str(value).lower()
    text = f"tune:\n  {key}: {token}\n" if nested else f"{key}: {token}\n"
    assert _load_worker_selector_yaml(tmp_path, text)[key] is value


@pytest.mark.parametrize("key", _WORKER_SELECTOR_KEYS)
@pytest.mark.parametrize("nested", [False, True])
def test_worker_selector_null_preserves_omission(tmp_path, key, nested):
    text = f"tune:\n  {key}: null\n" if nested else f"{key}: null\n"
    assert key not in _load_worker_selector_yaml(tmp_path, text)


@pytest.mark.parametrize("key", _WORKER_SELECTOR_KEYS)
@pytest.mark.parametrize(("token", "expected"), [("true", True), ("false", False), ("null", None)])
def test_worker_selector_nested_override_precedes_validation(tmp_path, key, token, expected):
    config = _load_worker_selector_yaml(tmp_path, f'{key}: "false"\ntune:\n  {key}: {token}\n')
    if expected is None:
        assert key not in config
    else:
        assert config[key] is expected


def test_omitted_worker_selectors_preserve_consumer_defaults(tmp_path):
    config = _load_worker_selector_yaml(tmp_path, "tune: {}\n")
    assert not any(key in config for key in _WORKER_SELECTOR_KEYS)


@pytest.mark.parametrize("key", _WORKER_SELECTOR_KEYS)
@pytest.mark.parametrize("value", [False, True])
def test_worker_selector_invalid_nested_overrides_valid_flat(tmp_path, key, value):
    text = f'{key}: {str(value).lower()}\ntune:\n  {key}: "false"\n'
    with pytest.raises(ValueError, match="must be a boolean") as exc:
        _load_worker_selector_yaml(tmp_path, text)
    assert key in str(exc.value)
