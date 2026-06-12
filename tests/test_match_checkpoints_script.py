from __future__ import annotations

from tests.script_loading import load_script_module


def _load_match_module():
    return load_script_module("match_checkpoints.py", "match_checkpoints_test_module")


def test_safe_log_label_removes_path_separators() -> None:
    module = _load_match_module()

    assert module._safe_log_label("../best model/a") == "best_model_a"
    assert module._safe_log_label("...") == "model"
