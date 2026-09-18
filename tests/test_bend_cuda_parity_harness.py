from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "native" / "bend_engine" / "cuda_parity" / "run_parity.py"

_spec = importlib.util.spec_from_file_location("deepfin_bend_cuda_parity", MODULE_PATH)
assert _spec is not None
assert _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)


def test_infer_bucket_from_deepfin_package_name() -> None:
    assert _mod.infer_bucket(Path("chess_b16.pt2"), None) == 16
    assert _mod.infer_bucket(Path("chess_b1190.pt2"), None) == 1190
    assert _mod.infer_bucket(Path("anything.pt2"), 7) == 7


def test_cuda_fixture_uses_real_cboard_encoding(tmp_path: Path) -> None:
    path = tmp_path / "fixture.bin"
    _mod.build_fixture(path, batch=3, input_planes=146, hist_mode=1)
    header = _mod.read_fixture_header(path)

    assert header == {
        "version": 1,
        "batch": 3,
        "planes": 146,
        "full_policy_width": 4672,
        "map_count": 1858,
    }

    header_bytes = len(_mod.MAGIC) + 5 * 4
    expected_bytes = (
        header_bytes
        + 1858 * 4
        + 3 * 146 * 8 * 8 * 2
    )
    assert path.stat().st_size == expected_bytes


def test_summary_is_stable_bit_level_contract() -> None:
    values = np.asarray([0.0, 1.0, -2.0, 3.5], dtype=np.float32)
    first = _mod.summarize_f32(values)
    second = _mod.summarize_f32(values.copy())
    assert first == second
    assert first[0] == 4
    assert first[1] != 0
    assert first[2] != 0
