import math
import sys

import pytest

from chess_anti_engine.utils.config_yaml import flatten_run_config_defaults, load_yaml_file


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


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "token",
    [".nan", ".inf", "-.inf", "1e999", "1.0e+999", '"nan"', '"inf"',
     '"-Infinity"', '"1e999"', '"bad"', "[]", "{}"],
)
def test_lr_yaml_rejects_unusable_effective_value(tmp_path, nested, token):
    path = tmp_path / "lr.yaml"
    path.write_text(("train:\n  lr: " if nested else "lr: ") + token + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'lr' must be a finite number"):
        flatten_run_config_defaults(load_yaml_file(path))


@pytest.mark.parametrize(
    "value",
    [0, -0.001, 0.002, "0.002", "-1e-3", sys.float_info.max,
     -sys.float_info.max, math.nextafter(0.0, 1.0)],
)
@pytest.mark.parametrize("nested", [False, True])
def test_lr_finite_values_preserve_representation(value, nested):
    config = {"train": {"lr": value}} if nested else {"lr": value}
    assert flatten_run_config_defaults(config)["lr"] is value


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("token", ["true", "false", "True", "yes", "off"])
def test_lr_yaml_rejects_boolean(tmp_path, nested, token):
    # Consumers cast with float(), so `lr: true` would silently mean LR 1.0.
    path = tmp_path / "lr.yaml"
    path.write_text(("train:\n  lr: " if nested else "lr: ") + token + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'lr' must be a finite number, not a boolean"):
        flatten_run_config_defaults(load_yaml_file(path))


def test_lr_integer_conversion_overflow_is_specific():
    with pytest.raises(ValueError, match="'lr' must be a finite number") as exc:
        flatten_run_config_defaults({"lr": 10**1000})
    assert isinstance(exc.value.__cause__, OverflowError)


@pytest.mark.parametrize("config", [{}, {"lr": None}, {"train": {"lr": None}}])
def test_lr_absent_or_null_is_omitted(config):
    assert "lr" not in flatten_run_config_defaults(config)


def test_lr_nested_precedence_validates_only_effective_value():
    assert flatten_run_config_defaults({"lr": float("nan"), "train": {"lr": "0.002"}})["lr"] == "0.002"
    with pytest.raises(ValueError, match="'lr' must be a finite number"):
        flatten_run_config_defaults({"lr": 0.002, "train": {"lr": float("nan")}})
    # Nested null overrides root first, then is omitted from the flat mapping.
    assert "lr" not in flatten_run_config_defaults({"lr": 0.002, "train": {"lr": None}})
    assert "lr" not in flatten_run_config_defaults({"lr": float("nan"), "train": {"lr": None}})


def test_lr_conversion_happens_once_without_replacing_value():
    class CountedFloat:
        calls = 0

        def __float__(self):
            self.calls += 1
            return 0.002

    value = CountedFloat()
    assert flatten_run_config_defaults({"lr": value})["lr"] is value
    assert value.calls == 1



def test_lr_unrelated_conversion_error_is_not_hidden():
    class BrokenFloat:
        def __float__(self):
            raise RuntimeError("unrelated conversion failure")

    with pytest.raises(RuntimeError, match="unrelated conversion failure"):
        flatten_run_config_defaults({"lr": BrokenFloat()})


@pytest.mark.parametrize("token", ["0.002", '"0.002"', "0", "-0.001", '"-1e-3"'])
@pytest.mark.parametrize("nested", [False, True])
def test_lr_finite_yaml_preserves_loaded_value(tmp_path, token, nested):
    path = tmp_path / "finite-lr.yaml"
    path.write_text(("train:\n  lr: " if nested else "lr: ") + token + "\n", encoding="utf-8")
    loaded = load_yaml_file(path)
    expected = loaded["train"]["lr"] if nested else loaded["lr"]
    assert flatten_run_config_defaults(loaded)["lr"] is expected


@pytest.mark.parametrize("token", ["true", "false"])
def test_lr_boolean_is_fatal_at_launch(tmp_path, monkeypatch, token):
    # The launch path reads the yaml through the same loader; a boolean `lr`
    # must stop the CLI before any server, broker or Tuner is built.
    from chess_anti_engine import run

    path = tmp_path / "launch.yaml"
    path.write_text("train:\n  lr: " + token + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "chess_anti_engine.run", "--config", str(path), "--mode", "train",
        "--work-dir", str(tmp_path), "--device", "cpu",
    ])
    with pytest.raises(ValueError, match="'lr' must be a finite number, not a boolean"):
        run.main()
