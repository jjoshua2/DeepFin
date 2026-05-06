from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_status_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "status.py"
    spec = importlib.util.spec_from_file_location("status_script_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trial_id_from_result_path_uses_ray_trial_suffix() -> None:
    module = _load_status_module()
    path = Path("runs/pbt2_small/tune/train_trial_ab12c_00003_3_lr=0.001/checkpoint/result.json")

    assert module._trial_id_from_result_path(path) == "ab12c_00003"


def test_trial_id_from_result_path_falls_back_to_parent_name() -> None:
    module = _load_status_module()

    assert module._trial_id_from_result_path(Path("somewhere/result.json")) == "somewhere"
