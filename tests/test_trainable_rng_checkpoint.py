from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from chess_anti_engine.tune.trainable_phases import _finalize_iteration
from chess_anti_engine.tune.trainable_report import _write_rng_state_sidecar
from chess_anti_engine.tune.trial_config import (
    DifficultyState,
    DriftMetrics,
    PidResult,
    RestoreResult,
    SelfplayResult,
    TrainingResult,
    TrialConfig,
)


class _Writer:
    def add_scalar(self, *_args, **_kwargs) -> None:
        return None


class _Opt:
    param_groups = [{"lr": 1e-3}]


class _Trainer:
    model = object()
    writer = _Writer()
    opt = _Opt()
    _peak_lr = 1e-3
    w_wdl = 1.0
    w_soft = 1.0
    w_categorical = 0.0
    w_sf_move = 0.0
    sf_wdl_frac = 0.5
    sf_wdl_conf_power = 1.0
    sf_wdl_draw_scale = 1.0
    sf_wdl_temperature = 1.0
    _feature_group_dropout = [
        ("king", (), 0.0),
        ("pins", (), 0.0),
        ("pawns", (), 0.0),
        ("mobility", (), 0.0),
        ("outposts", (), 0.0),
    ]


def test_finalize_updates_checkpoint_rng_after_puzzle_eval(
    tmp_path: Path, monkeypatch,
) -> None:
    rng = np.random.default_rng(123)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    _write_rng_state_sidecar(ckpt_dir=ckpt_dir, rng=rng)

    def _consume_rng(*_args, **_kwargs):
        rng.integers(0, 2**31)
        return {"puzzle_accuracy": 0.0, "puzzle_correct": 0, "puzzle_total": 1}

    monkeypatch.setattr(
        "chess_anti_engine.tune.trainable_phases._run_puzzle_eval_if_due",
        _consume_rng,
    )

    reported_state = {}

    def _report(_report_dict, checkpoint=None):
        _ = checkpoint
        reported_state.update(
            json.loads((ckpt_dir / "rng_state.json").read_text(encoding="utf-8"))
        )

    _finalize_iteration(
        tc=TrialConfig.from_dict({"device": "cpu", "puzzle_interval": 1}),
        trainer=_Trainer(),
        pid=None,
        sp=SelfplayResult(),
        tr=TrainingResult(),
        drift=DriftMetrics(),
        pid_result=PidResult(),
        eval_dict={"eval_win": 0, "eval_draw": 0, "eval_loss": 0, "eval_winrate": 0.0},
        checkpoint=object(),
        best_loss=999.0,
        ckpt_dir=ckpt_dir,
        work_dir=tmp_path,
        trial_dir=tmp_path,
        status_csv_path=tmp_path / "status.csv",
        tune_report_fn=_report,
        puzzle_suite=object(),
        ds=DifficultyState(wdl_regret=-1.0, sf_nodes=500),
        distributed_pause_started_at=None,
        distributed_pause_active=False,
        restore=RestoreResult(),
        holdout_frozen=False,
        holdout_generation=0,
        buf_size=0,
        holdout_buf_size=0,
        iter_t0=0.0,
        iteration_idx=1,
        iteration_zero_based=0,
        global_iter=0,
        completed_iterations=0,
        device="cpu",
        rng=rng,
    )

    assert reported_state == rng.bit_generator.state
    status_cols = (tmp_path / "status.csv").read_text(encoding="utf-8").strip().split(",")
    assert status_cols[0] == "1"
    assert status_cols[1] == "1"
