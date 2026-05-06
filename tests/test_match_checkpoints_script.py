from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_match_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "match_checkpoints.py"
    spec = importlib.util.spec_from_file_location("match_checkpoints_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_safe_log_label_removes_path_separators() -> None:
    module = _load_match_module()

    assert module._safe_log_label("../best model/a") == "best_model_a"
    assert module._safe_log_label("...") == "model"
