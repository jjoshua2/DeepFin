from __future__ import annotations

import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.replay.shard import (
    INPUT_HISTORY_ENCODING_ARRAY_KEY,
    ShardMeta,
    save_local_shard_arrays,
    samples_to_arrays,
)
from scripts import diagnose, diagnose_arch
from scripts.diagnostic_replay_utils import MAX_SKIPPED_SHARD_DETAILS, record_skipped_shard
from scripts.diagnose_wdl_by_age import _concat_batches as _concat_wdl_age_batches
from scripts.diagnose_wdl_by_age import _take_rows as _take_wdl_age_rows
from scripts.diagnose_replay import sample_replay_arrays
from scripts.trace_sf_search_disagreement_regret import Sample as TraceSample
from scripts.trace_sf_search_disagreement_regret import _build_trace


ROOT = Path(__file__).resolve().parents[1]


def _touch_shard(path: Path) -> None:
    path.mkdir(parents=True)


def _sample(i: int = 0) -> ReplaySample:
    policy = np.zeros((4672,), dtype=np.float32)
    policy[i] = 1.0
    return ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=1,
    )


def _diagnostic_sample(*, game_id: int = 7, ply: int = 0) -> ReplaySample:
    policy = np.zeros((POLICY_SIZE,), dtype=np.float32)
    policy[0] = 1.0
    legal = np.ones((POLICY_SIZE,), dtype=np.uint8)
    sample = ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=0,
        priority=1.25,
        priority_q_delta=0.125,
        priority_sf_search_gap=0.25,
        game_id=game_id,
        ply_index=ply,
        sf_wdl=np.asarray([0.65, 0.20, 0.15], dtype=np.float32),
        search_wdl=np.asarray([0.20, 0.20, 0.60], dtype=np.float32),
        sf_played_rank=2,
        sf_played_regret=0.03125,
        sf_policy_target=policy.copy(),
        future_sf_regret_sum=0.03125,
        future_sf_regret_h4=0.03125,
        is_selfplay=True,
        policy_soft_target=policy.copy(),
        legal_mask=legal,
        sf_legal_mask=legal.copy(),
    )
    return sample


def _write_diagnostic_replay(tmp_path: Path) -> Path:
    replay_dir = tmp_path / "replay_shards"
    replay_dir.mkdir()
    samples = [_diagnostic_sample()]
    save_local_shard_arrays(
        replay_dir / "shard_000001.zarr",
        arrs=samples_to_arrays(samples),
        meta=ShardMeta(positions=len(samples)),
    )
    return replay_dir


def _run_diagnostic_script(script: str, replay_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath = str(ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script),
            "--replay-dir",
            str(replay_dir),
            "--max-shards",
            "1",
            *extra,
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )


def test_diagnose_replay_dir_prefers_trainer_replay_shards(tmp_path: Path) -> None:
    trial_dir = tmp_path / "train_trial_00000"
    replay_dir = trial_dir / "replay_shards"
    legacy_dir = trial_dir / "selfplay_shards"
    _touch_shard(replay_dir / "shard_000001.zarr")
    _touch_shard(legacy_dir / "shard_000001.zarr")

    resolved = diagnose._resolve_replay_dir(
        Namespace(replay_dir=None),
        cfg={},
        trial_dir=trial_dir,
    )

    assert resolved == replay_dir.resolve()


def test_diagnose_replay_dir_falls_back_to_selfplay_exports(tmp_path: Path) -> None:
    trial_dir = tmp_path / "train_trial_00000"
    legacy_dir = trial_dir / "selfplay_shards"
    _touch_shard(legacy_dir / "shard_000001.zarr")

    resolved = diagnose_arch._resolve_replay_dir(
        Namespace(replay_dir=None),
        cfg={},
        trial_dir=trial_dir,
    )

    assert resolved == legacy_dir.resolve()


def test_diagnose_replay_dir_errors_with_checked_paths(tmp_path: Path) -> None:
    trial_dir = tmp_path / "train_trial_00000"

    with pytest.raises(SystemExit) as exc:
        diagnose._resolve_replay_dir(
            Namespace(replay_dir=None),
            cfg={},
            trial_dir=trial_dir,
        )

    msg = str(exc.value)
    assert "No replay shards found" in msg
    assert "replay_shards" in msg
    assert "selfplay_shards" in msg


def test_diagnose_replay_dir_accepts_explicit_replay_dir(tmp_path: Path) -> None:
    explicit = tmp_path / "custom"
    explicit.mkdir()

    resolved = diagnose._resolve_replay_dir(
        Namespace(replay_dir=str(explicit)),
        cfg={},
        trial_dir=tmp_path / "ignored_trial",
    )

    assert resolved == explicit.resolve()


def test_sample_replay_arrays_broadcasts_scalar_history_metadata(tmp_path: Path) -> None:
    shard_dir = tmp_path / "replay"
    shard_dir.mkdir()
    save_local_shard_arrays(
        shard_dir / "shard_000001.zarr",
        arrs=samples_to_arrays([_sample(0), _sample(1)]),
        meta=ShardMeta(input_history_encoding="lc0_root", positions=2),
    )

    arrs, total, _shards = sample_replay_arrays(
        shard_dir,
        2,
        rng=np.random.default_rng(1),
        fields=("x", INPUT_HISTORY_ENCODING_ARRAY_KEY),
    )

    assert total == 2
    assert arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY].shape == (2,)
    assert set(arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY].astype(str).tolist()) == {"lc0_root"}


def test_sample_replay_arrays_supplies_blank_missing_history_metadata(tmp_path: Path) -> None:
    shard_dir = tmp_path / "replay"
    shard_dir.mkdir()
    save_local_shard_arrays(
        shard_dir / "shard_000001.zarr",
        arrs=samples_to_arrays([_sample(0)]),
        meta=ShardMeta(positions=1),
    )

    arrs, total, _shards = sample_replay_arrays(
        shard_dir,
        1,
        rng=np.random.default_rng(1),
        fields=("x", INPUT_HISTORY_ENCODING_ARRAY_KEY),
    )

    assert total == 1
    assert arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY].shape == (1,)
    assert arrs[INPUT_HISTORY_ENCODING_ARRAY_KEY].item() == ""


def test_wdl_age_sampler_broadcasts_history_metadata_per_shard() -> None:
    first = _take_wdl_age_rows(
        {
            "x": np.zeros((2, 146, 8, 8), dtype=np.float32),
            INPUT_HISTORY_ENCODING_ARRAY_KEY: np.asarray("lc0_root"),
        },
        np.asarray([0], dtype=np.int64),
    )
    second = _take_wdl_age_rows(
        {
            "x": np.ones((2, 146, 8, 8), dtype=np.float32),
            INPUT_HISTORY_ENCODING_ARRAY_KEY: np.asarray(""),
        },
        np.asarray([1], dtype=np.int64),
    )

    out = _concat_wdl_age_batches([first, second])

    assert out[INPUT_HISTORY_ENCODING_ARRAY_KEY].shape == (2,)
    assert out[INPUT_HISTORY_ENCODING_ARRAY_KEY].astype(str).tolist() == ["lc0_root", ""]


@pytest.mark.parametrize(
    ("script", "extra", "needle"),
    [
        ("diagnose_future_eval_bucketed.py", ("--horizons", "2"), "replay_dir="),
        ("diagnose_future_eval_weights.py", ("--horizons", "2"), "# Future Eval Blend Weights"),
        (
            "diagnose_sf_eval_head_blend.py",
            ("--checkpoint", "missing-checkpoint.pt", "--horizons", "2", "--device", "cpu"),
            "replay_dir=",
        ),
        (
            "diagnose_target_calibration.py",
            ("--window-positions", "1", "--buckets", "1", "--policy-sample-per-bucket", "1"),
            "# Target Calibration Diagnostics",
        ),
        ("trace_sf_search_disagreement_regret.py", ("--top", "1"), "# SF/search disagreement -> regret trace"),
    ],
)
def test_live_diagnostic_scripts_smoke(
    tmp_path: Path,
    script: str,
    extra: tuple[str, ...],
    needle: str,
) -> None:
    replay_dir = _write_diagnostic_replay(tmp_path)

    result = _run_diagnostic_script(script, replay_dir, *extra)

    assert result.returncode == 0, result.stderr
    assert needle in result.stdout


def test_record_skipped_shard_caps_details(tmp_path: Path) -> None:
    scan: dict[str, Any] = {"skipped_shards": [], "skipped_shards_omitted": 0}

    for i in range(MAX_SKIPPED_SHARD_DETAILS + 3):
        record_skipped_shard(scan, tmp_path / f"shard_{i:06d}.zarr", RuntimeError("boom"))

    assert len(scan["skipped_shards"]) == MAX_SKIPPED_SHARD_DETAILS
    assert scan["skipped_shards_omitted"] == 3


def test_disagreement_trace_normalizes_after_sample_to_trigger_pov() -> None:
    trigger = TraceSample(
        game_id=1,
        ply=10,
        sf_q=-0.2,
        search_q=0.4,
        regret=0.03,
        rank=2,
        priority=1.0,
        priority_q_delta=0.1,
        priority_sf_search_gap=0.6,
        shard="shard_000001.zarr",
        row=0,
    )
    after = TraceSample(
        game_id=1,
        ply=11,
        sf_q=0.3,
        search_q=-0.1,
        regret=float("nan"),
        rank=-1,
        priority=1.0,
        priority_q_delta=0.0,
        priority_sf_search_gap=0.0,
        shard="shard_000001.zarr",
        row=1,
    )

    trace = _build_trace(trigger, trigger, after, min_gap_reduction=0.02)

    assert trace.after_sf_q == pytest.approx(-0.3)
    assert trace.after_search_q == pytest.approx(0.1)
    assert trace.after_gap == pytest.approx(0.4)
    assert trace.search_toward_sf == pytest.approx(0.3)
    assert trace.verdict == "search_moved_toward_sf"


def test_wdl_age_concat_preserves_missing_history_metadata_rows() -> None:
    tagged = _take_wdl_age_rows(
        {
            "x": np.zeros((1, 146, 8, 8), dtype=np.float32),
            INPUT_HISTORY_ENCODING_ARRAY_KEY: np.asarray("lc0_root"),
        },
        np.asarray([0], dtype=np.int64),
    )
    untagged = _take_wdl_age_rows(
        {"x": np.ones((1, 146, 8, 8), dtype=np.float32)},
        np.asarray([0], dtype=np.int64),
    )

    out = _concat_wdl_age_batches([tagged, untagged])

    assert out[INPUT_HISTORY_ENCODING_ARRAY_KEY].shape == (2,)
    assert out[INPUT_HISTORY_ENCODING_ARRAY_KEY].astype(str).tolist() == ["lc0_root", ""]
