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
from scripts import (
    diagnose,
    diagnose_arch,
    diagnose_future_eval_bucketed,
    diagnose_future_eval_weights,
    diagnose_future_value_targets,
    diagnose_sf_eval_head_blend,
)
from scripts.diagnostic_replay_utils import (
    MAX_SKIPPED_SHARD_DETAILS,
    future_regret_field_names,
    record_skipped_shard,
)
from scripts.diagnose_target_calibration import _adjusted_wdl_game_target, _blend_wdl
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


def _wdl_from_q(q: float) -> np.ndarray:
    value = float(np.clip(q, -1.0, 1.0))
    if value >= 0.0:
        return np.asarray([value, 1.0 - value, 0.0], dtype=np.float32)
    return np.asarray([0.0, 1.0 + value, -value], dtype=np.float32)


def _diagnostic_sample(
    *,
    game_id: int = 7,
    ply: int = 0,
    sf_q: float = 0.5,
    search_q: float = -0.4,
    outcome: int = 0,
) -> ReplaySample:
    policy = np.zeros((POLICY_SIZE,), dtype=np.float32)
    policy[(game_id + ply) % POLICY_SIZE] = 1.0
    legal = np.ones((POLICY_SIZE,), dtype=np.uint8)
    sample = ReplaySample(
        x=np.zeros((146, 8, 8), dtype=np.float32),
        policy_target=policy,
        wdl_target=outcome,
        priority=1.25,
        priority_q_delta=0.125,
        priority_sf_search_gap=0.25,
        game_id=game_id,
        ply_index=ply,
        sf_wdl=_wdl_from_q(sf_q),
        search_wdl=_wdl_from_q(search_q),
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


def _write_diagnostic_replay(tmp_path: Path, samples: list[ReplaySample] | None = None) -> Path:
    replay_dir = tmp_path / "replay_shards"
    replay_dir.mkdir()
    if samples is None:
        samples = [_diagnostic_sample()]
    save_local_shard_arrays(
        replay_dir / "shard_000001.zarr",
        arrs=samples_to_arrays(samples),
        meta=ShardMeta(positions=len(samples)),
    )
    return replay_dir


def _write_multiply_diagnostic_replay(tmp_path: Path) -> Path:
    samples: list[ReplaySample] = []
    for game_id in range(100):
        outcome = game_id % 3
        base = ((game_id % 11) - 5) / 10.0
        for ply in (0, 2, 4):
            samples.append(
                _diagnostic_sample(
                    game_id=game_id,
                    ply=ply,
                    sf_q=float(np.clip(base + 0.05 * ply, -0.9, 0.9)),
                    search_q=float(np.clip(-base + 0.03 * ply, -0.9, 0.9)),
                    outcome=outcome,
                ),
            )
    return _write_diagnostic_replay(tmp_path, samples)


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
        timeout=60,
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
        ("diagnose_future_eval_bucketed.py", (), "row_all"),
        ("diagnose_future_eval_weights.py", (), "future_avg_q_regret_adjusted"),
        (
            "diagnose_sf_eval_head_blend.py",
            ("--checkpoint", "missing-checkpoint.pt", "--horizons", "2", "--device", "cpu"),
            "replay_dir=",
        ),
        (
            "diagnose_target_calibration.py",
            ("--window-positions", "200", "--buckets", "2", "--policy-sample-per-bucket", "2"),
            "# Target Calibration Diagnostics",
        ),
        (
            "diagnose_future_value_targets.py",
            (
                "--horizons",
                "2,4",
                "--taus",
                "2,4",
                "--sources",
                "avg",
                "--max-offset",
                "4",
            ),
            "# Future Value Target Diagnostics",
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
    if script == "diagnose_sf_eval_head_blend.py":
        replay_dir = _write_diagnostic_replay(tmp_path)
    else:
        replay_dir = _write_multiply_diagnostic_replay(tmp_path)

    result = _run_diagnostic_script(script, replay_dir, *extra)

    assert result.returncode == 0, result.stderr
    assert needle in result.stdout


def test_record_skipped_shard_caps_details(tmp_path: Path) -> None:
    scan: dict[str, Any] = {"skipped_shards": [], "skipped_shards_omitted": 0}

    for i in range(MAX_SKIPPED_SHARD_DETAILS + 3):
        record_skipped_shard(scan, tmp_path / f"shard_{i:06d}.zarr", RuntimeError("boom"))

    assert len(scan["skipped_shards"]) == MAX_SKIPPED_SHARD_DETAILS
    assert scan["skipped_shards_omitted"] == 3


def test_future_regret_field_names_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="adjusted_wdl_regret_source"):
        future_regret_field_names("d99")


def test_diagnostic_replay_utils_import_does_not_load_torch() -> None:
    env = os.environ.copy()
    pythonpath = str(ROOT)
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.diagnostic_replay_utils; raise SystemExit('torch' in sys.modules)",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_future_eval_weight_smoke_produces_fit_rows(tmp_path: Path) -> None:
    replay_dir = _write_multiply_diagnostic_replay(tmp_path)
    games, scan = diagnose_future_eval_weights._load_samples(replay_dir, max_shards=1)

    rows = diagnose_future_eval_weights._rows_for_horizon(
        games,
        horizon=2,
        gap_threshold=0.05,
        include_regret_adjusted=True,
    )

    assert scan["valid_samples"] == 300
    assert any(row.target == "future_avg_q" and row.n > 0 for row in rows)
    assert any(row.target == "future_avg_q_regret_adjusted" and row.n > 0 for row in rows)


def test_future_eval_bucketed_smoke_exercises_grouped_fit(tmp_path: Path) -> None:
    replay_dir = _write_multiply_diagnostic_replay(tmp_path)
    games, _scan, mean_regret = diagnose_future_eval_bucketed._load_samples(replay_dir, max_shards=1)
    game_ids, features, target, bucket, path_regret = diagnose_future_eval_bucketed._pairs_for_horizon(
        games,
        horizon=2,
        missing_regret_impute=mean_regret,
        eval_bins=[-0.5, 0.0, 0.5],
    )

    rows = diagnose_future_eval_bucketed._fit_modes(
        horizon=2,
        game_ids=game_ids,
        features=features,
        target=target,
        bucket=bucket,
        near_even=1.0,
    )

    assert features.shape == (200, 3)
    assert path_regret.shape == (200,)
    assert {"row_all", "game_mean", "game_eval_bucket_mean"} <= {row.mode for row in rows}


def test_future_eval_bucketed_does_not_impute_unknown_selfplay_state() -> None:
    def sample(game_id: int, ply: int, is_selfplay: bool | None) -> diagnose_future_eval_bucketed.Sample:
        return diagnose_future_eval_bucketed.Sample(
            game_id=game_id,
            ply=ply,
            sf_q=0.2,
            search_q=0.4,
            final_q=1.0,
            sf_played_regret=float("nan"),
            has_sf_played_regret=False,
            is_selfplay=is_selfplay,
        )

    games = {
        1: [sample(1, 0, None), sample(1, 2, None)],
        2: [sample(2, 0, False), sample(2, 2, False)],
    }

    game_ids, _features, _target, _bucket, path_regret = diagnose_future_eval_bucketed._pairs_for_horizon(
        games,
        horizon=2,
        missing_regret_impute=0.5,
        eval_bins=[],
    )

    by_game = {int(game_id): float(regret) for game_id, regret in zip(game_ids, path_regret, strict=True)}
    assert by_game[1] == 0.0
    assert by_game[2] == 0.5


def test_future_value_suffix_rows_use_all_future_same_pov_values() -> None:
    def sample(
        ply: int,
        sf_q: float,
        search_q: float,
        regret: float,
    ) -> diagnose_future_eval_bucketed.Sample:
        return diagnose_future_eval_bucketed.Sample(
            game_id=1,
            ply=ply,
            sf_q=sf_q,
            search_q=search_q,
            final_q=1.0,
            sf_played_regret=regret,
            has_sf_played_regret=True,
            is_selfplay=False,
        )

    games = {
        1: [
            sample(0, 0.0, 0.0, 0.10),
            sample(2, 0.2, 0.4, 0.05),
            sample(4, 0.6, 0.8, 0.00),
        ],
    }

    raw_spec = ("avg", None, False)
    adjusted_spec = ("avg", None, True)
    rows = diagnose_future_value_targets._suffix_rows_by_spec(
        games,
        specs=[raw_spec, adjusted_spec],
        min_offset=2,
        max_offset=4,
        missing_regret_impute=0.0,
    )
    game_ids, features, raw_target = rows[raw_spec]
    _game_ids, _features, adjusted_target = rows[adjusted_spec]

    assert game_ids.tolist() == [1, 1]
    np.testing.assert_allclose(features[0], np.asarray([0.0, 0.0, 1.0]))
    # Ply 0 uses both future same-POV values: avg_q(ply2)=0.3 and avg_q(ply4)=0.7.
    np.testing.assert_allclose(raw_target[0], 0.5)
    # Regret adjustment subtracts current regret from ply2, and current+ply2 regret from ply4.
    np.testing.assert_allclose(adjusted_target[0], ((0.3 - 2.0 * 0.10) + (0.7 - 2.0 * 0.15)) / 2.0)


def test_future_value_grid_candidates_select_on_train_split() -> None:
    train_ids = np.asarray([i for i in range(1, 26) if i % 5 != 0], dtype=np.int64)
    test_ids = np.asarray([5 * i for i in range(20)], dtype=np.int64)
    game_ids = np.concatenate([train_ids, test_ids])
    features = np.vstack(
        [
            np.tile(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64), (train_ids.size, 1)),
            np.tile(np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64), (test_ids.size, 1)),
        ],
    )
    target = np.ones((game_ids.size,), dtype=np.float64)

    candidates = diagnose_future_value_targets._grid_candidates({2: (game_ids, features, target)})
    best_avg = next(candidate for candidate in candidates if candidate.name == "grid_best_avg")

    assert best_avg.w_sf == 1.0
    assert best_avg.w_search == 0.0


def test_future_value_latest_trial_uses_result_json_mtime(tmp_path: Path) -> None:
    old_trial = tmp_path / "tune" / "train_trial_old"
    active_trial = tmp_path / "tune" / "train_trial_active"
    old_trial.mkdir(parents=True)
    active_trial.mkdir(parents=True)
    old_result = old_trial / "result.json"
    active_result = active_trial / "result.json"
    old_result.write_text('{"sf_wdl_frac": 0.10}\n')
    active_result.write_text('{"sf_wdl_frac": 0.35}\n')

    os.utime(old_trial, (3000.0, 3000.0))
    os.utime(active_trial, (1000.0, 1000.0))
    os.utime(old_result, (1000.0, 1000.0))
    os.utime(active_result, (3000.0, 3000.0))

    assert diagnose_future_value_targets._latest_trial_dir(tmp_path) == active_trial


def test_future_value_current_candidates_use_selected_replay_trial(tmp_path: Path) -> None:
    other_trial = tmp_path / "tune" / "train_trial_other"
    selected_trial = tmp_path / "tune" / "train_trial_selected"
    other_trial.mkdir(parents=True)
    selected_trial.mkdir(parents=True)
    other_result = other_trial / "result.json"
    selected_result = selected_trial / "result.json"
    other_result.write_text('{"sf_wdl_frac": 0.90, "search_wdl_frac": 0.05}\n')
    selected_result.write_text('{"sf_wdl_frac": 0.35, "search_wdl_frac": 0.35}\n')
    os.utime(other_result, (3000.0, 3000.0))
    os.utime(selected_result, (1000.0, 1000.0))
    replay_dir = tmp_path / "replay" / selected_trial.name / "replay_shards"
    replay_dir.mkdir(parents=True)

    trial_dir = diagnose_future_value_targets._trial_dir_for_replay(tmp_path, replay_dir)
    rows = diagnose_future_value_targets._current_candidates(
        tmp_path,
        tmp_path / "missing.yaml",
        trial_dir=trial_dir,
    )

    assert trial_dir == selected_trial
    assert rows[0].name == "reported_live"
    assert rows[0].w_sf == 0.35
    assert rows[0].w_search == 0.35


def test_sf_eval_head_blend_smoke_exercises_fit_rows(tmp_path: Path) -> None:
    replay_dir = _write_multiply_diagnostic_replay(tmp_path)
    games, _scan = diagnose_sf_eval_head_blend._load_samples(replay_dir, max_shards=1)
    pairs = diagnose_sf_eval_head_blend._build_pairs(games, horizons=[2])
    preds = {
        (str(pair.ref.shard), int(pair.ref.row)): (0.25 * pair.sf_now, 0.25 * pair.search_now)
        for pair in pairs
    }

    data, target, game_ids, meta = diagnose_sf_eval_head_blend._rows_for_horizon(
        pairs,
        preds,
        horizon=2,
        target_name="adjusted",
    )
    row = diagnose_sf_eval_head_blend._summarize_fit(
        name="all",
        feature_names=["sf_label", "search_label", "final", "model_wdl", "model_sf_eval"],
        features=np.stack(
            [
                data["sf_label"],
                data["search_label"],
                data["final"],
                data["model_wdl"],
                data["model_sf_eval"],
            ],
            axis=1,
        ),
        target=target,
        game_ids=game_ids,
        fold=0,
    )

    assert len(pairs) == 200
    assert meta["n"] == 200
    assert row["n_test"] == 40
    assert "w_model_sf_eval" in row


def test_target_calibration_adjusted_game_target_changes_blend() -> None:
    outcome = np.asarray([0, 1, 2], dtype=np.int64)
    future_regret = np.asarray([0.2, 0.2, 0.2], dtype=np.float64)
    has_future_regret = np.asarray([True, True, True])
    game_target = _adjusted_wdl_game_target(
        outcome,
        future_regret,
        has_future_regret,
        regret_scale=1.0,
        regret_cap=0.0,
    )
    sf = np.tile(np.asarray([[0.6, 0.3, 0.1]], dtype=np.float64), (3, 1))
    search = np.tile(np.asarray([[0.2, 0.6, 0.2]], dtype=np.float64), (3, 1))

    blend = _blend_wdl(
        game_target,
        sf,
        search,
        sf_frac=0.0,
        search_frac=0.0,
        dampen_sf_low=0.0,
        dampen_sf_high=0.0,
    )

    np.testing.assert_allclose(game_target[0], np.asarray([0.6, 0.4, 0.0]))
    np.testing.assert_allclose(game_target[1], np.asarray([0.0, 0.6, 0.4]))
    np.testing.assert_allclose(game_target[2], np.asarray([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(blend, game_target, atol=1e-8)


def test_target_calibration_interpolate_can_use_raw_game_fallback() -> None:
    game_target = np.asarray([[0.6, 0.4, 0.0]], dtype=np.float64)
    raw_game = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    sf = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
    search = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)

    blend = _blend_wdl(
        game_target,
        sf,
        search,
        raw_game_target=raw_game,
        fallback_target=raw_game,
        sf_frac=1.0,
        search_frac=0.0,
        dampen_sf_low=1.0,
        dampen_sf_high=0.0,
        blend_mode="interpolate",
    )

    np.testing.assert_allclose(blend, raw_game, atol=1e-8)


def test_target_calibration_renormalize_shifts_dampened_weight_to_search() -> None:
    raw_game = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float64)
    sf = np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
    search = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)

    blend = _blend_wdl(
        raw_game,
        sf,
        search,
        raw_game_target=raw_game,
        fallback_target=raw_game,
        sf_frac=0.5,
        search_frac=0.5,
        dampen_sf_low=1.0,
        dampen_sf_high=0.0,
        blend_mode="renormalize",
    )

    np.testing.assert_allclose(blend, search, atol=1e-8)


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
