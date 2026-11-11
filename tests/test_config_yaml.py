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


@pytest.mark.parametrize("key", ["pb2_bounds_lr", "pb2_bounds_w_wdl", "pb2_bounds_custom"])
@pytest.mark.parametrize(
    "value",
    [
        0.1, False, "0.1,0.2", {}, [], [0.1], [0.1, 0.2, 0.3],
        [None, 1], [[], 1], ["bad", 1], [0, {}],
        [float("nan"), 1], [0, float("nan")],
        [float("-inf"), 1], [0, float("inf")],
        ["-Infinity", 1], [0, "NaN"], [0, "1e999"],
        [10**400, 1], [0, 10**400], [2, 1],
    ],
)
def test_pb2_bounds_reject_invalid_effective_values(key, value):
    config = {"tune": {key: value}}
    with pytest.raises(ValueError, match=rf"YAML {key}.*two-element.*finite.*lower"):
        flatten_run_config_defaults(config)


@pytest.mark.parametrize(
    "value",
    [
        [0, 0], [-2, -1], [-1, 0], [0.1, 1], ["0.1", "0.2"],
        [False, True], [True, True], (0, 1), ["-2", "0"],
        [5e-324, 1.7976931348623157e308],
    ],
)
def test_pb2_bounds_preserve_valid_representation(value):
    config = {"tune": {"pb2_bounds_lr": value}}
    flat = flatten_run_config_defaults(config)
    assert flat["pb2_bounds_lr"] is value


@pytest.mark.parametrize(
    "config",
    [
        {}, {"tune": {"pb2_bounds_lr": None}},
    ],
)
def test_pb2_bounds_null_omission(config):
    assert flatten_run_config_defaults(config) == {}


@pytest.mark.parametrize("value", [None, "bad", [0, 1]])
def test_pb2_bounds_root_syntax_remains_unsupported(value):
    with pytest.raises(ValueError, match="Unknown yaml root-level keys"):
        flatten_run_config_defaults(
            {"pb2_bounds_lr": value, "tune": {"pb2_bounds_lr": [0, 1]}}
        )


def test_pb2_bounds_validate_every_effective_bound():
    with pytest.raises(ValueError, match="pb2_bounds_w_wdl"):
        flatten_run_config_defaults(
            {"tune": {"pb2_bounds_lr": [0, 1], "pb2_bounds_w_wdl": "bad"}}
        )


def test_pb2_bounds_convert_endpoints_once():
    class Endpoint:
        def __init__(self, value):
            self.value = value
            self.calls = 0

        def __float__(self):
            self.calls += 1
            return self.value

    lo, hi = Endpoint(0.0), Endpoint(1.0)
    bounds = [lo, hi]
    assert flatten_run_config_defaults({"tune": {"pb2_bounds_lr": bounds}})["pb2_bounds_lr"] is bounds
    assert (lo.calls, hi.calls) == (1, 1)


def test_pb2_bounds_unrelated_conversion_failure_propagates():
    class Endpoint:
        def __float__(self):
            raise RuntimeError("endpoint failure")

    with pytest.raises(RuntimeError, match="endpoint failure"):
        flatten_run_config_defaults({"tune": {"pb2_bounds_lr": [Endpoint(), 1]}})


@pytest.mark.parametrize(
    "bounds_yaml", ["0.1", "[.nan, 1]", "[0, .inf]", "[-.inf, 1]", "[2, 1]"],
)
def test_pb2_bounds_yaml_parser_rejection(tmp_path, bounds_yaml):
    from chess_anti_engine.utils.config_yaml import load_yaml_file

    path = tmp_path / "bounds.yaml"
    path.write_text(f"tune:\n  pb2_bounds_lr: {bounds_yaml}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="pb2_bounds_lr"):
        flatten_run_config_defaults(load_yaml_file(path))


def test_pb2_bounds_yaml_parser_preserves_strings_and_omits_null(tmp_path):
    from chess_anti_engine.utils.config_yaml import load_yaml_file

    path = tmp_path / "bounds.yaml"
    path.write_text(
        "tune:\n  pb2_bounds_lr: ['0.1', '0.2']\n  pb2_bounds_w_wdl: null\n",
        encoding="utf-8",
    )
    assert flatten_run_config_defaults(load_yaml_file(path)) == {
        "pb2_bounds_lr": ["0.1", "0.2"],
    }
