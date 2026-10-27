from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.model import ARCH_SCHEMA_VERSION, build_model

from chess_anti_engine.replay import shard
from chess_anti_engine.train.trainer import select_input_history_arrays
from tests.test_shrink_ffn_checkpoint_script import _calibration_kwargs, _load_shrink_module, _small_cfg


class SliceOnly:
    def __init__(self, value: np.ndarray, reads: list[int]):
        self.value = value
        self.shape = value.shape
        self.dtype = value.dtype
        self.reads = reads

    def __array__(self, *args: Any, **_kwargs: Any) -> np.ndarray:
        raise AssertionError("full materialization")

    def __getitem__(self, index: slice) -> np.ndarray:
        out = self.value[index].copy()
        self.reads.append(len(out))
        assert len(out) <= 256
        return out


def fixture_arrays(rows: int = 257) -> dict[str, np.ndarray]:
    return {
        "x": np.arange(rows * 2 * 64, dtype=np.float32).reshape(rows, 2, 8, 8) / 1000,
        "policy_target": np.full((rows, 1858), 1 / 1858, np.float32),
        "wdl_target": np.ones(rows, np.int64),
        "_input_history_encoding": np.asarray("legacy"),
    }


def install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, arrays: dict[str, np.ndarray]) -> tuple[Path, list[int]]:
    path = tmp_path / "shard_000001.zarr"
    path.mkdir()
    reads: list[int] = []

    def load(_path: Path, *, lazy: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        proxies = {key: SliceOnly(value, reads) if value.ndim else value for key, value in arrays.items()}
        shard.validate_array_declarations(proxies)
        if not lazy:
            return {key: np.asarray(value) for key, value in proxies.items()}, {}
        return proxies, {}

    monkeypatch.setattr(shard, "load_shard_arrays", load)
    monkeypatch.setattr(shard, "iter_shard_paths", lambda _: [path])
    return path, reads


@pytest.mark.parametrize("positions", [1, 256, 257, 999])
def test_prefix_bounded_decode_and_exact_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, positions: int) -> None:
    arrays = fixture_arrays()
    path, reads = install(monkeypatch, tmp_path, arrays)
    cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    result = _load_shrink_module()._activation_directory_prefix(path, source_cfg=cfg, positions=positions)
    assert result is not None
    np.testing.assert_array_equal(np.concatenate([c["x"] for c in result]), arrays["x"][:positions])
    assert reads == [256, 256, 256, 1, 1, 1]
    assert sum(c["x"].nbytes for c in result) == arrays["x"][:positions].nbytes
    assert all(not np.shares_memory(c["x"], arrays["x"]) for c in result)


@pytest.mark.parametrize("bad", ["x", "policy_nan", "policy_zero", "wdl", "optional_nan", "orphan", "index", "shape"])
def test_late_invalid_content_refused_before_first_yield(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad: str) -> None:
    arrays = fixture_arrays()
    if bad == "x":
        arrays["x"][-1, 0, 0, 0] = np.nan
    elif bad == "policy_nan":
        arrays["policy_target"][-1, 0] = np.nan
    elif bad == "policy_zero":
        arrays["policy_target"][-1] = 0
    elif bad == "wdl":
        arrays["wdl_target"][-1] = 3
    elif bad == "optional_nan":
        arrays["x_lc0_root"] = arrays["x"].copy()
        arrays["x_lc0_root"][-1, 0, 0, 0] = np.nan
    elif bad == "orphan":
        arrays["has_x_lc0_root"] = np.zeros(257, np.uint8)
        arrays["has_x_lc0_root"][-1] = 1
    elif bad == "index":
        arrays["sf_move_index"] = np.zeros(257, np.int32)
        arrays["sf_move_index"][-1] = 1858
        arrays["has_sf_move"] = np.ones(257, np.uint8)
    else:
        arrays["wdl_target"] = np.ones(258, np.int64)
    message = {"x": "x contains", "policy_nan": "NaN/Inf", "policy_zero": "non-positive", "wdl": "out of range", "optional_nan": "NaN/Inf", "orphan": "missing", "index": "out of range", "shape": "matching x"}[bad]
    with pytest.raises(ValueError, match=message):
        shard.validate_arrays(arrays)
    install(monkeypatch, tmp_path, arrays)
    cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    chunks = _load_shrink_module()._iter_calibration_arrays_from_replay(
        **_calibration_kwargs(tmp_path, cfg, positions=1), inputs_only=True,
    )
    with pytest.raises(ValueError, match=message):
        next(chunks)


@pytest.mark.parametrize("recorded", [True, False])
def test_late_history_refusal_and_recorded_parity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recorded: bool) -> None:
    arrays = fixture_arrays()
    arrays["x_lc0_root"] = arrays["x"] + 3
    arrays["has_x_lc0_root"] = np.ones(257, np.uint8)
    arrays["has_x_lc0_root"][-1] = int(recorded)
    cfg = dataclasses.replace(_small_cfg(ffn_mult_by_layer=(2.0, 1.5)), input_history_encoding="lc0_root")
    path, _ = install(monkeypatch, tmp_path, arrays)
    if not recorded:
        with pytest.raises(ValueError, match="no recorded"):
            select_input_history_arrays(arrays, input_history_encoding="lc0_root")
        with pytest.raises(ValueError, match="no recorded"):
            _load_shrink_module()._activation_directory_prefix(path, source_cfg=cfg, positions=1)
    else:
        expected = select_input_history_arrays(arrays, input_history_encoding="lc0_root")["x"]
        result = _load_shrink_module()._activation_directory_prefix(path, source_cfg=cfg, positions=1)
        np.testing.assert_array_equal(result[0]["x"], expected[:1])


def test_non_scalar_identity_metadata_preserves_eager_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    arrays = fixture_arrays()
    arrays["_input_history_encoding"] = np.full(257, "legacy")
    path, reads = install(monkeypatch, tmp_path, arrays)
    cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    assert _load_shrink_module()._activation_directory_prefix(path, source_cfg=cfg, positions=1) is None
    assert reads == []


@pytest.mark.parametrize("rows", [0, 1, 257])
def test_global_shape_and_empty_parity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rows: int) -> None:
    arrays = fixture_arrays(rows)
    shard.validate_arrays(arrays)
    path, _ = install(monkeypatch, tmp_path, arrays)
    cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    result = _load_shrink_module()._activation_directory_prefix(path, source_cfg=cfg, positions=1)
    assert result is not None
    assert sum(len(c["x"]) for c in result) == min(rows, 1)


@pytest.mark.parametrize("encoding", ["lc0_root", "lc0_root_legacy_meta"])
def test_full_history_width_selected_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, encoding: str) -> None:
    arrays = fixture_arrays()
    arrays["x"] = np.zeros((257, 146, 8, 8), np.float16)
    arrays["x"][:, 104] = 0.75
    arrays["x_lc0_root"] = np.full_like(arrays["x"], 0.25)
    arrays["has_x_lc0_root"] = np.ones(257, np.uint8)
    cfg = dataclasses.replace(_small_cfg(ffn_mult_by_layer=(2.0, 1.5)), input_history_encoding=encoding)
    expected = select_input_history_arrays(arrays, input_history_encoding=encoding)["x"]
    path, _ = install(monkeypatch, tmp_path, arrays)
    result = _load_shrink_module()._activation_directory_prefix(path, source_cfg=cfg, positions=257)
    np.testing.assert_array_equal(np.concatenate([c["x"] for c in result]), expected)


def test_real_activation_caller_global_batch_parity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    arrays = fixture_arrays()
    arrays["x"] = np.random.default_rng(7).normal(size=(257, 146, 8, 8)).astype(np.float16)
    install(monkeypatch, tmp_path, arrays)
    cfg = _small_cfg(ffn_mult_by_layer=(2.0, 1.5))
    module = _load_shrink_module()
    ckpt = {"model": build_model(cfg).state_dict(), "arch": {"_schema_version": ARCH_SCHEMA_VERSION, **dataclasses.asdict(cfg)}}
    common = {"ckpt_path": Path("unused.pt"), "batch_size": 130, "device": torch.device("cpu")}
    expected, stats = module.activation_unit_scores_from_x(ckpt, x=torch.as_tensor(arrays["x"]), **common)
    actual, actual_stats = module.activation_unit_scores_from_replay(
        ckpt, replay_dir=tmp_path, positions=257, prefer_recorded_lc0_root=False,
        synthetic_lc0_root_history=False, lc0_root_legacy_meta=False, **common,
    )
    assert actual_stats == stats
    for layer in expected:
        torch.testing.assert_close(actual[layer], expected[layer], rtol=0, atol=0)
