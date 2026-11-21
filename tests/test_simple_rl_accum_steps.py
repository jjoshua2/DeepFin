"""simple_rl gradient accumulation (``--accum-steps``).

An optimizer step over ``accum_steps`` micro-batches of ``batch_size`` rows has
the gradient of ONE native batch of ``batch_size * accum_steps`` rows (every
row's loss coefficient is 1 / effective batch), presents the same rows in the
same order as the native plan, counts optimizer steps for the schedule and the
snapshots, and is recorded in the train manifest from the Trainer that ran it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from chess_anti_engine.replay.shard import samples_to_arrays
from chess_anti_engine.simple_rl.learn import (
    TRAIN_KEYS,
    ScheduledSampler,
    TrainSettings,
    build_trainer,
    plan_batches,
    planned_phase_updates,
    snapshot_updates_for_rows,
    step_row_counts,
    train_generation,
)
from tests.simple_rl_helpers import save_tiny_checkpoint, tiny_model_config
from tests.test_simple_rl_learn import _rows, _write_generation


def _arrays(n: int, seed: int = 0) -> dict[str, np.ndarray]:
    rows = samples_to_arrays(_rows(n, marker=0.0, seed=seed)[0])
    return {k: v for k, v in rows.items() if k in TRAIN_KEYS or k.startswith("_")}


def _step_grads(settings: TrainSettings, sampler: Any, steps: int) -> list[dict[str, torch.Tensor]]:
    """The gradient the optimizer sees at each step (read in a step pre-hook)."""
    torch.manual_seed(0)
    trainer = build_trainer(tiny_model_config(), settings, log_dir=None)
    assert trainer.accum_steps == settings.accum_steps
    got: list[dict[str, torch.Tensor]] = []

    def grab(_o: Any, _a: Any, _k: Any) -> None:
        got.append({n: p.grad.detach().clone() for n, p in trainer.model.named_parameters() if p.grad is not None})

    trainer.opt.register_step_pre_hook(grab)
    trainer.train_steps(sampler, batch_size=int(settings.batch_size), steps=steps)
    assert int(trainer.step) == steps
    return got


def _assert_same_grads(a: list[dict[str, torch.Tensor]], b: list[dict[str, torch.Tensor]]) -> None:
    assert a
    assert len(a) == len(b)
    for ga, gb in zip(a, b, strict=True):
        live = [n for n, g in gb.items() if float(g.abs().max()) > 1e-7]
        assert live
        assert set(ga) == set(gb)
        for n in live:
            np.testing.assert_allclose(ga[n].numpy(), gb[n].numpy(), rtol=2e-4, atol=1e-7, err_msg=n)


def test_accumulated_steps_have_the_native_big_batch_gradient_including_the_short_tail() -> None:
    n, b, k = 100, 16, 3  # steps of 48, 48 and a 4-row tail (split 2 / 1 / 1)
    rows = _arrays(n)
    native, _ = plan_batches(n, batch_size=b * k, sl_frac=0.0, sl_available=0, rng=np.random.default_rng(5))
    micro, _ = plan_batches(n, batch_size=b, sl_frac=0.0, sl_available=0, rng=np.random.default_rng(5),
                            accum_steps=k)
    assert [len(o) for o, _ in native] == [48, 48, 4]
    assert [len(o) for o, _ in micro] == [16, 16, 16, 16, 16, 16, 2, 1, 1]
    for s, (own, _) in enumerate(native):  # the same rows, in the same order, per step
        assert np.array_equal(np.concatenate([o for o, _ in micro[s * k:(s + 1) * k]]), own)
    big = _step_grads(TrainSettings(device="cpu", mirror_prob=0.0, batch_size=b * k),
                      ScheduledSampler(rows, native, rng=np.random.default_rng(0)), steps=3)
    acc_sampler = ScheduledSampler(rows, micro, rng=np.random.default_rng(0))
    acc = _step_grads(TrainSettings(device="cpu", mirror_prob=0.0, batch_size=b, accum_steps=k), acc_sampler, steps=3)
    _assert_same_grads(acc, big)
    assert acc_sampler.cursor == len(micro)


def test_one_micro_batch_is_not_the_big_batch_gradient() -> None:
    """The equality above is not vacuous: one 16-row micro-batch alone differs."""
    rows = _arrays(48)
    order = np.arange(48)
    big = _step_grads(TrainSettings(device="cpu", mirror_prob=0.0, batch_size=48),
                      ScheduledSampler(rows, [(order, np.zeros(0, np.int64))], rng=np.random.default_rng(0)), 1)
    one = _step_grads(TrainSettings(device="cpu", mirror_prob=0.0, batch_size=16),
                      ScheduledSampler(rows, [(order[:16], np.zeros(0, np.int64))], rng=np.random.default_rng(0)), 1)
    with pytest.raises(AssertionError):
        _assert_same_grads(one, big)


def test_a_tail_smaller_than_accum_steps_joins_the_previous_step_at_the_same_row_coefficient() -> None:
    assert step_row_counts(97, batch_size=16, accum_steps=3) == [48, 49]
    assert step_row_counts(97, batch_size=48, accum_steps=1) == [48, 48, 1]  # the historical plan is untouched
    rows = _arrays(97)
    micro, _ = plan_batches(97, batch_size=16, sl_frac=0.0, sl_available=0, rng=np.random.default_rng(3),
                            accum_steps=3)
    sizes = [len(o) for o, _ in micro]
    assert len(micro) == 6
    assert min(sizes) > 0
    assert sum(sizes) == 97
    assert np.array_equal(np.sort(np.concatenate([o for o, _ in micro])), np.arange(97))  # each row exactly once
    last = np.concatenate([o for o, _ in micro[3:]])
    # Native reference: those 49 rows as one batch at the effective batch's coefficient (1/48 per row).
    ref = _step_grads(TrainSettings(device="cpu", mirror_prob=0.0, batch_size=48),
                      ScheduledSampler(rows, [(last, np.zeros(0, np.int64))], rng=np.random.default_rng(0)), 1)
    acc = _step_grads(TrainSettings(device="cpu", mirror_prob=0.0, batch_size=16, accum_steps=3),
                      ScheduledSampler(rows, micro[3:], rng=np.random.default_rng(0)), 1)
    _assert_same_grads(acc, ref)


def test_snapshot_rows_resolve_to_the_first_step_that_reaches_them() -> None:
    steps = step_row_counts(100, batch_size=16, accum_steps=3)  # [48, 48, 4]
    assert steps == [48, 48, 4]
    assert snapshot_updates_for_rows((48, 49, 100), steps) == (1, 2, 3)
    with pytest.raises(ValueError, match="reach past"):
        snapshot_updates_for_rows((101,), steps)
    with pytest.raises(ValueError, match="repeated optimizer steps"):
        snapshot_updates_for_rows((10, 20), steps)


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"accum_steps": 0}, "accum_steps must be >= 1"),
    ({"accum_steps": 2, "kl_policy": 0.1, "kl_reference": "x"}, "per-micro-batch mean"),
    ({"accum_steps": 2, "kl_wdl": 0.1, "kl_reference": "x"}, "per-micro-batch mean"),
    ({"accum_steps": 2, "sl_mix_frac": 0.25, "sl_mix_dir": "x"}, "SL share is rounded per micro-batch"),
    ({"snapshot_rows": (10,), "snapshot_updates": (1,)}, "alternatives"),
    ({"snapshot_rows": (10, 10)}, "strictly increasing"),
    ({"snapshot_rows": (10,), "sample_with_replacement": True}, "exact-once schedule only"),
])
def test_settings_refuse_what_accumulation_cannot_honour(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        TrainSettings(**kwargs).validate()


def _run(tmp_path: Path, n: int) -> Path:
    run = tmp_path / "run"
    (run / "gen_0000").mkdir(parents=True)
    save_tiny_checkpoint(run / "gen_0000" / "model.pt")
    _write_generation(run, 1, n)
    return run


def test_train_generation_counts_optimizer_steps_and_records_the_realised_accumulation(tmp_path: Path) -> None:
    run = _run(tmp_path, 100)
    settings = TrainSettings(device="cpu", batch_size=8, accum_steps=3, holdout_frac=0.0, seed=7, lr=1e-3,
                             snapshot_rows=(24, 48))
    planned = planned_phase_updates(run, 1, settings)
    assert planned == len(step_row_counts(100, batch_size=8, accum_steps=3)) == 5  # 4 x 24 + a 4-row tail
    summary = train_generation(run_dir=run, generation=1, init_checkpoint=run / "gen_0000" / "model.pt",
                               prev_trainer_checkpoint=None, settings=settings)
    assert summary["updates"] == planned
    assert summary["end_step"] - summary["start_step"] == planned
    assert summary["accum_steps"] == 3
    assert summary["effective_batch"] == 24
    assert summary["rows_per_optimizer_step"] == {"steps": 5, "min": 4, "max": 24, "rows": 100}
    assert summary["snapshot_updates"] == [1, 2]
    assert summary["snapshot_rows"] == [24, 48]
    for k, rows in ((1, 24), (2, 48)):
        m = json.loads((run / "gen_0001" / "train" / "snapshots" / f"step_{k}" / "metrics.json").read_text())
        assert m["update"] == k
        assert m["rows_presented"] == rows
    manifest = json.loads((run / "gen_0001" / "train" / "manifest.json").read_text())
    assert manifest["config"]["realised"]["accum_steps"] == 3
    assert manifest["config"]["realised"]["effective_batch"] == 24
    assert manifest["config"]["settings"]["accum_steps"] == 3
    with np.load(run / "gen_0001" / "train" / "presented.npz") as z:
        assert np.array_equal(np.sort(z["row"]), np.arange(100))  # exact-once is unchanged


def test_driver_flag_reaches_the_trainer_is_pinned_and_a_change_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import simple_rl_loop
    from tests.test_simple_rl_resume import _setup
    from tests.test_simple_rl_snapshots_arms import _argv

    openings, model_config = _setup(tmp_path, monkeypatch)
    run = tmp_path / "acc"
    argv = _argv(run, openings, model_config, "--accum-steps", "2")
    assert simple_rl_loop.main(argv) == 0
    manifest = json.loads((run / "gen_0001" / "train" / "manifest.json").read_text())
    assert manifest["config"]["realised"]["accum_steps"] == 2  # read off the Trainer that ran
    assert manifest["config"]["realised"]["effective_batch"] == 16
    assert manifest["summary"]["rows_per_optimizer_step"]["max"] == 16
    assert json.loads((run / "run_config.json").read_text())["settings"]["accum_steps"] == 2
    more = list(argv)
    more[more.index("--generations") + 1] = "2"
    more[more.index("--accum-steps") + 1] = "4"
    with pytest.raises(SystemExit, match="accum_steps"):
        simple_rl_loop.main(more)


def test_a_run_pinned_before_the_flag_resumes_at_the_default_and_refuses_another_value(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import simple_rl_loop
    from tests.test_simple_rl_resume import _setup
    from tests.test_simple_rl_snapshots_arms import _argv, _stage

    openings, model_config = _setup(tmp_path, monkeypatch)
    run = tmp_path / "old"
    argv = _argv(run, openings, model_config)
    assert simple_rl_loop.main(argv) == 0
    cfg_path = run / "run_config.json"
    cfg = json.loads(cfg_path.read_text())
    for key in ("accum_steps", "snapshot_rows"):  # what a run pinned before these flags holds
        del cfg["settings"][key]
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit, match="accum_steps"):
        simple_rl_loop.main(_stage(argv, "train", "--redo", "--accum-steps", "2"))
    assert simple_rl_loop.main(argv) == 0  # the default reads as what the run implicitly had
