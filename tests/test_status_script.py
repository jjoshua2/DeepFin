from __future__ import annotations

from pathlib import Path

from tests.script_loading import load_script_module


def _load_status_module():
    return load_script_module("status.py", "status_script_test_module")


def test_trial_id_from_result_path_uses_ray_trial_suffix() -> None:
    module = _load_status_module()
    path = Path("runs/pbt2_small/tune/train_trial_ab12c_00003_3_lr=0.001/checkpoint/result.json")

    assert module._trial_id_from_result_path(path) == "ab12c_00003"


def test_trial_id_from_result_path_falls_back_to_parent_name() -> None:
    module = _load_status_module()

    assert module._trial_id_from_result_path(Path("somewhere/result.json")) == "somewhere"
