"""shrink_ffn_checkpoint's history flags are refused loudly, never ignored.

Three flags used to be parsed and then ``del``-ed. The history selector decides
all three from the checkpoint (recorded ``x_lc0_root`` always preferred, legacy
meta iff the encoding asks, and NO lossy remap of legacy rows ever), so each now
exits 2 with a reason, and a legacy row with no recorded root tensor is refused
through this tool on every calibration route.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.replay import shard
from tests.test_shrink_directory_prefix import SliceOnly, install
from tests.test_shrink_ffn_checkpoint_script import _load_shrink_module, _small_cfg


def _legacy_arrays(rows: int = 5) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    return {
        "x": (rng.random((rows, 146, 8, 8)) < 0.1).astype(np.float32),
        "policy_target": np.full((rows, 1858), 1 / 1858, np.float32),
        "wdl_target": np.ones(rows, np.int64),
        "_input_history_encoding": np.asarray("legacy"),
    }


def _install_eager(monkeypatch: pytest.MonkeyPatch, arrays: dict[str, np.ndarray]) -> None:
    # A path that is not a directory: the prefix path declines, the eager loader runs.
    monkeypatch.setattr(shard, "iter_shard_paths", lambda _: [Path("not_a_dir.zarr")])
    monkeypatch.setattr(shard, "load_shard_arrays", lambda _p, *, lazy: (dict(arrays), {}))


@pytest.mark.parametrize("route", ["directory_prefix", "eager_inputs", "eager_taylor"])
@pytest.mark.parametrize("encoding", ["lc0_root", "lc0_root_legacy_meta"])
def test_legacy_rows_without_recorded_root_are_refused_on_every_route(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, route: str, encoding: str,
) -> None:
    """No route may silently synthesize LC0-root history (rl_loop_audit M12)."""
    arrays = _legacy_arrays()
    if route == "directory_prefix":
        install(monkeypatch, tmp_path, arrays)
    else:
        _install_eager(monkeypatch, arrays)
    cfg = dataclasses.replace(_small_cfg(ffn_mult_by_layer=(2.0, 1.5)), input_history_encoding=encoding)
    chunks = _load_shrink_module()._iter_calibration_arrays_from_replay(
        source_cfg=cfg, replay_dir=tmp_path, positions=5,
        inputs_only=route != "eager_taylor",
    )
    with pytest.raises(ValueError, match="no recorded x_lc0_root"):
        list(chunks)


def _run_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *extra: str) -> SystemExit:
    module = _load_shrink_module()
    argv = ["shrink_ffn_checkpoint.py", str(tmp_path / "missing_in.pt"), str(tmp_path / "out.pt"),
            "--target-ffn-mult-by-layer", "1.0,1.0", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as info:
        module.main()
    return info.value


@pytest.mark.parametrize("score_mode", ["weight", "activation", "taylor"])
@pytest.mark.parametrize(("flag", "reason"), [
    ("--prefer-recorded-lc0-root", "always uses a recorded x_lc0_root"),
    ("--synthetic-lc0-root-history", "never synthesizes LC0-root history"),
    ("--lc0-root-legacy-meta", "CHECKPOINT's input_history_encoding"),
])
def test_removed_flags_exit_with_a_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    flag: str, reason: str, score_mode: str,
) -> None:
    exc = _run_main(monkeypatch, tmp_path, flag, "--score-mode", score_mode)
    assert exc.code == 2
    err = capsys.readouterr().err
    assert "was removed" in err, err
    assert reason in err, err


def test_without_history_flags_main_reaches_the_checkpoint_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Positive control: the refusals above come from the flags, not from argv."""
    exc = _run_main(monkeypatch, tmp_path, "--score-mode", "activation")
    assert "missing input checkpoint" in str(exc.code)


# ── decode window aligned to the zarr row chunk ───────────────────────────────


class ChunkedSliceOnly(SliceOnly):
    def __init__(self, value: np.ndarray, reads: list[int], chunk_rows: int):
        super().__init__(value, reads)
        self.chunks = (chunk_rows, *value.shape[1:])

    def __getitem__(self, index: slice) -> np.ndarray:
        out = self.value[index].copy()
        self.reads.append(len(out))
        return out


@pytest.mark.parametrize(("chunk_rows", "window"), [
    (512, 512), (256, 256), (100, 300), (1, 256), (4096, 4096), (5000, 256),
])
def test_decode_window_rows(chunk_rows: int, window: int) -> None:
    module = _load_shrink_module()
    proxy = ChunkedSliceOnly(np.zeros((3, 1), np.float32), [], chunk_rows)
    assert module._decode_window_rows(proxy) == window
    assert module._decode_window_rows(np.zeros((3, 1))) == 256


@pytest.mark.parametrize("positions", [1, 600, 1100])
def test_prefix_steps_by_the_chunk_and_stays_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, positions: int,
) -> None:
    arrays = _legacy_arrays(1100)
    arrays["x"] = arrays["x"][:, :2]
    path = tmp_path / "shard_000001.zarr"
    path.mkdir()
    reads: list[int] = []

    def load(_path: Path, *, lazy: bool) -> tuple[dict[str, Any], dict[str, Any]]:
        assert lazy
        return {k: ChunkedSliceOnly(v, reads, 512) if v.ndim else v for k, v in arrays.items()}, {}

    monkeypatch.setattr(shard, "load_shard_arrays", load)
    result = _load_shrink_module()._activation_directory_prefix(
        path, source_cfg=_small_cfg(ffn_mult_by_layer=(2.0, 1.5)), positions=positions,
    )
    assert result is not None
    np.testing.assert_array_equal(np.concatenate([c["x"] for c in result]), arrays["x"][:positions])
    # Every row of every field is still decoded (validation on all rows), in
    # 512-row windows: x, policy_target, wdl_target per window.
    assert reads == [512] * 3 + [512] * 3 + [76] * 3


def test_real_zarr_shard_exposes_its_writer_chunk(tmp_path: Path) -> None:
    """The production writer's 512-row chunk reaches the window through the
    REAL lazy loader -- proxies alone cannot show ``.chunks`` exists there."""
    from chess_anti_engine.replay import ReplaySample
    from chess_anti_engine.replay.shard import load_shard_arrays, samples_to_arrays, save_local_shard_arrays

    x_all = _legacy_arrays(700)["x"]
    samples = []
    for x in x_all:
        sample = ReplaySample(x=x, policy_target=np.full(1858, 1 / 1858, np.float32), wdl_target=1)
        sample.input_history_encoding = "legacy"
        samples.append(sample)
    arrays = {"x": x_all}
    path = save_local_shard_arrays(
        tmp_path / "shard_000001.zarr", arrs=samples_to_arrays(samples), meta={"positions": 700},
    )
    lazy, _ = load_shard_arrays(path, lazy=True)
    module = _load_shrink_module()
    assert module._decode_window_rows(lazy["x"]) == 512
    result = module._activation_directory_prefix(
        path, source_cfg=_small_cfg(ffn_mult_by_layer=(2.0, 1.5)), positions=700,
    )
    assert result is not None
    np.testing.assert_array_equal(np.concatenate([c["x"] for c in result]), arrays["x"])
