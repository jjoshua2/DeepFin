"""zclip's adaptive EMA must survive a restart, or every restart re-warms it.

`ZClip` keeps its whole adaptive state in four plain attributes --
`initialized`, `mean`, `var`, and the warmup `buffer` -- built up over the
first `warmup_steps` (25) optimizer steps and then updated as an EMA. None of
it was in the trainer checkpoint, so every restart reconstructed a `ZClip` in
the constructor and started from `initialized=False`.

That is not a cosmetic reset. While `initialized` is False, `ZClip.step`
returns BEFORE `_compute_clip_val` is ever called: the adaptive branch cannot
fire at all, only the fixed `max_grad_norm` cap applies, and the EMA then
re-converges from wherever those 25 warmup norms happened to sit rather than
from where the previous run's distribution actually was. A run restarted often
enough spends a fixed slice of every restart in a hard-cap-only regime, and the
adaptive threshold it eventually reaches is a function of the restart's first
25 steps instead of the run's history.

⚑ RESTART-GATED, and in a way worth stating precisely, because it is one
restart later than it looks. `save`/`load` are code, so a running trial keeps
its current behaviour. The FIRST restart onto this code still reads a
checkpoint written by the old code, finds no key, and warms fresh -- identical
to today. That restart's saves carry the key, so the restore first takes effect
at the SECOND restart. Anyone reading `zclip/restored == 0.0` on the first
restart is looking at correct behaviour, not a broken fix.

⚑ The tests here go through `Trainer.save` and `Trainer.load` rather than
calling `zclip_state_dict`/`load_zclip_state` directly, because the question is
not whether the pair round-trips -- it is whether the production resume path
reaches them. `trainable_init` resumes via `trainer.load(maybe)`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from chess_anti_engine.train.trainer import Trainer


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        del x
        return {"policy": self.lin.weight[:1]}


def _make_trainer(tmp_path: Path) -> Trainer:
    return Trainer(
        _Tiny(),
        device="cpu",
        lr=1e-3,
        optimizer="adamw",
        warmup_steps=10,
        warmup_lr_start=1e-5,
        use_amp=False,
        log_dir=tmp_path,
        tb_log_interval=1000,
        prefetch_batches=False,
        zclip_max_norm=6.5,
    )


def _warm(trainer: Trainer, *, mean: float, var: float) -> None:
    """Put the trainer's zclip in a converged state, the way a long run does."""
    trainer.zclip.initialized = True
    trainer.zclip.mean = mean
    trainer.zclip.var = var
    trainer.zclip.buffer = []


def _grads(trainer: Trainer, scale: float) -> None:
    """Give every parameter a gradient, so `ZClip.step` has a norm to read."""
    for p in trainer.model.parameters():
        p.grad = torch.full_like(p, scale)


def test_save_writes_the_adaptive_state_into_the_checkpoint(tmp_path: Path) -> None:
    """The payload itself, not just what `load` can be handed.

    A `load` that reads a key nothing writes is the same defect wearing the
    other hat, so assert on the file.
    """
    trainer = _make_trainer(tmp_path)
    _warm(trainer, mean=8.7, var=2.25)
    path = tmp_path / "trainer.pt"
    trainer.save(path)

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    assert "zclip" in ckpt, "the checkpoint must carry zclip's adaptive state"
    assert ckpt["zclip"]["initialized"] is True
    assert ckpt["zclip"]["mean"] == pytest.approx(8.7)
    assert ckpt["zclip"]["var"] == pytest.approx(2.25)


def test_a_round_trip_through_save_and_load_preserves_the_ema(tmp_path: Path) -> None:
    """The headline: restart, and the EMA is where the previous run left it."""
    src = _make_trainer(tmp_path / "src")
    _warm(src, mean=8.7, var=2.25)
    path = tmp_path / "trainer.pt"
    src.save(path)

    dst = _make_trainer(tmp_path / "dst")
    assert dst.zclip.initialized is False, "a fresh trainer starts in warmup"

    dst.load(path)

    assert dst.zclip.initialized is True, (
        "the restored trainer must skip the 25-step warmup; while initialized "
        "is False the adaptive branch cannot fire at all"
    )
    assert dst.zclip.mean == pytest.approx(8.7)
    assert dst.zclip.var == pytest.approx(2.25)


def test_a_partial_warmup_buffer_round_trips(tmp_path: Path) -> None:
    """A checkpoint written mid-warmup keeps the norms already collected.

    Otherwise a restart cadence shorter than 25 steps could never finish a
    warmup at all -- the EMA would be perpetually uninitialized.
    """
    src = _make_trainer(tmp_path / "src")
    src.zclip.initialized = False
    src.zclip.mean = None
    src.zclip.var = None
    src.zclip.buffer = [4.0, 5.5, 6.25]
    path = tmp_path / "trainer.pt"
    src.save(path)

    dst = _make_trainer(tmp_path / "dst")
    dst.load(path)

    assert dst.zclip.initialized is False
    assert dst.zclip.buffer == [4.0, 5.5, 6.25]


def test_a_checkpoint_without_the_key_loads_cleanly(tmp_path: Path) -> None:
    """GUARANTEED to happen at the first deploy, on every existing checkpoint.

    The old checkpoint is built by saving and then deleting the key, so it is
    the real payload shape minus the addition rather than a hand-written
    approximation of one.
    """
    src = _make_trainer(tmp_path / "src")
    _warm(src, mean=8.7, var=2.25)
    path = tmp_path / "trainer.pt"
    src.save(path)

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    del ckpt["zclip"]
    torch.save(ckpt, str(path))

    dst = _make_trainer(tmp_path / "dst")
    _warm(dst, mean=1.0, var=1.0)  # so a silent no-op is distinguishable
    dst.load(path)  # must not raise

    assert dst.zclip.initialized is True, (
        "an absent key must leave the constructor's state exactly as it was, "
        "not clear it"
    )
    assert dst.zclip.mean == pytest.approx(1.0)


def test_a_fresh_trainer_loading_an_old_checkpoint_stays_in_warmup(
    tmp_path: Path,
) -> None:
    """The realistic first-deploy shape: fresh trainer, keyless checkpoint.

    Paired with the case above: that one proves the restore does not CLOBBER,
    this one proves it does not fabricate an initialized EMA out of nothing.
    """
    src = _make_trainer(tmp_path / "src")
    path = tmp_path / "trainer.pt"
    src.save(path)
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    del ckpt["zclip"]
    torch.save(ckpt, str(path))

    dst = _make_trainer(tmp_path / "dst")
    dst.load(path)

    assert dst.zclip.initialized is False
    assert dst.zclip.mean is None
    assert dst.zclip.buffer == []


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_ema_is_REFUSED_rather_than_restored(
    tmp_path: Path, bad: float,
) -> None:
    """⚑ Persisting state means a poisoned EMA would outlive the restart.

    Per `_restore_zclip_stats`: one nan in `mean` is PERMANENT. Every later
    `z = (norm - nan) / std` is nan, `nan > z_thresh` is False, so
    `_compute_clip_val` returns None for the rest of the run -- the adaptive
    clipper is silently off while the fixed cap goes on reporting normally.
    Before this change a restart was the thing that cleared it. Making the
    state survive restarts is only safe together with this refusal, so the two
    ship in one commit.
    """
    src = _make_trainer(tmp_path / "src")
    _warm(src, mean=bad, var=2.25)
    path = tmp_path / "trainer.pt"
    src.save(path)

    dst = _make_trainer(tmp_path / "dst")
    dst.load(path)

    assert dst.zclip.initialized is False, (
        "a non-finite EMA must be refused; restoring it would carry a "
        "permanently disabled adaptive clipper across the restart that used "
        "to clear it"
    )
    assert dst.zclip.mean is None


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-dict",
        42,
        {"mean": 1.0, "var": 1.0},                       # no `initialized`
        {"initialized": True, "mean": None, "var": None},  # claims init, has none
        {"initialized": False, "buffer": ["x"]},           # unparseable buffer
    ],
)
def test_a_malformed_payload_falls_back_without_raising(
    tmp_path: Path, payload: Any,
) -> None:
    """A corrupt or future-shaped payload must degrade, never take the run down.

    Fresh warmup costs 25 steps; an exception here costs the restart.
    """
    trainer = _make_trainer(tmp_path)

    assert trainer.load_zclip_state(payload) is False
    assert trainer.zclip.initialized is False
    assert trainer.zclip.mean is None


def test_the_serialized_state_covers_every_field_the_snapshot_knows_about(
    tmp_path: Path,
) -> None:
    """One definition of "zclip's adaptive state", not two.

    `_zclip_stats_snapshot`/`_restore_zclip_stats` already existed for the
    discarded-step rollback. If `zclip_state_dict` grew its own field list, a
    field added to one and not the other would be dropped SILENTLY -- the
    symptom is just "it re-warms every restart", which is the original bug
    reappearing with the fix in place. Asserting the round trip reproduces the
    snapshot exactly is what makes the two share an instrument.
    """
    src = _make_trainer(tmp_path / "src")
    _warm(src, mean=8.7, var=2.25)
    src.zclip.buffer = []
    before = src._zclip_stats_snapshot()

    dst = _make_trainer(tmp_path / "dst")
    assert dst.load_zclip_state(src.zclip_state_dict()) is True

    assert dst._zclip_stats_snapshot() == before


def test_the_restore_is_skipped_when_the_optimizer_state_was_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A layout change means the EMA describes a DIFFERENT set of parameters.

    `load` already reinitialises the scheduler on that branch for the same
    reason. An EMA of gradient norms taken over other parameters is not a
    description of this run's gradients, so fresh warmup is the honest state.
    """
    src = _make_trainer(tmp_path / "src")
    _warm(src, mean=8.7, var=2.25)
    path = tmp_path / "trainer.pt"
    src.save(path)

    dst = _make_trainer(tmp_path / "dst")
    real_load = dst.opt.load_state_dict
    calls: list[int] = []

    def _boom_once(state: Any) -> None:
  # The FIRST call is the donor state, which is what a layout mismatch
  # rejects; the second is `load`'s own fresh-state fallback and must work.
        calls.append(1)
        if len(calls) == 1:
            raise ValueError("simulated parameter-layout mismatch")
        real_load(state)

    monkeypatch.setattr(dst.opt, "load_state_dict", _boom_once)

    dst.load(path)

    assert len(calls) >= 2, "the fallback branch must have been the one taken"
    assert dst.zclip.initialized is False, (
        "the EMA must not cross a parameter-layout change"
    )


def test_the_model_only_restore_does_NOT_carry_the_ema(tmp_path: Path) -> None:
    """The PB2 cross-optimizer donor path deliberately stays out of scope.

    `_restore_from_ray_checkpoint` takes `_load_model_only` when the donor
    trial ran a DIFFERENT optimizer family; it restores weights and nothing
    else, precisely because the donor's optimizer state does not describe this
    trial. An EMA of the donor's gradient norms is the same kind of object, so
    it stays out too. Pinned because the natural "improvement" here is to wire
    it in everywhere.
    """
    from chess_anti_engine.tune.trainable_init import _load_model_only

    src = _make_trainer(tmp_path / "src")
    _warm(src, mean=8.7, var=2.25)
    path = tmp_path / "trainer.pt"
    src.save(path)

    dst = _make_trainer(tmp_path / "dst")
    _load_model_only(path, dst, device="cpu", label="test")

    assert dst.zclip.initialized is False, (
        "a model-only restore must not inherit the donor's gradient statistics"
    )


def test_the_restore_changes_what_the_very_next_step_can_do(
    tmp_path: Path,
) -> None:
    """The OBSERVATION, not the attribute: behaviour of the first step differs.

    This is the test that distinguishes "the value was accepted" from "the
    value takes effect". A fresh zclip cannot adaptively clip anything on its
    first step no matter how large the gradient -- `ZClip.step` returns before
    `_compute_clip_val` while `initialized` is False. A restored one can, and
    that difference is visible in `grad_adaptive_clip_rate`, which is what
    `_zclip_step` reports into the iteration metrics.
    """
    src = _make_trainer(tmp_path / "src")
    _warm(src, mean=1.0, var=0.01)  # a big gradient is a huge outlier here
    path = tmp_path / "trainer.pt"
    src.save(path)

    fresh = _make_trainer(tmp_path / "fresh")
    _grads(fresh, 5.0)
    _, fresh_stats = fresh._zclip_step(collect_stats=True)
    assert fresh_stats is not None
    assert fresh_stats["adaptive_clip"] == 0.0, (
        "during warmup the adaptive branch is structurally unreachable"
    )

    restored = _make_trainer(tmp_path / "restored")
    restored.load(path)
    _grads(restored, 5.0)
    _, restored_stats = restored._zclip_step(collect_stats=True)
    assert restored_stats is not None
    assert restored_stats["adaptive_clip"] == 1.0, (
        "with the EMA restored the same gradient is recognised as the outlier "
        "it is, on the FIRST step of the restarted run"
    )


def test_the_load_emits_the_restored_scalar(tmp_path: Path) -> None:
    """`zclip/restored` is the on-restart observation, so pin that it is written.

    Both directions: 1.0 when the state came back, 0.0 when it did not. A
    scalar that is only emitted on success cannot distinguish "did not restore"
    from "did not run".
    """
    seen: list[tuple[str, float, int]] = []

    src = _make_trainer(tmp_path / "src")
    _warm(src, mean=8.7, var=2.25)
    src.step = 4242
    path = tmp_path / "trainer.pt"
    src.save(path)

    dst = _make_trainer(tmp_path / "dst")
    dst.writer.add_scalar = (
        lambda tag, value, step: seen.append((tag, float(value), int(step)))
    )
    dst.load(path)

    assert ("zclip/restored", 1.0, 4242) in seen, (
        "the scalar must land on the RESUMED step, not on 0, or it plots at "
        "the origin of every restart"
    )
    assert ("zclip/ema_mean", pytest.approx(8.7), 4242) in seen

    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    del ckpt["zclip"]
    torch.save(ckpt, str(path))

    seen.clear()
    other = _make_trainer(tmp_path / "other")
    other.writer.add_scalar = (
        lambda tag, value, step: seen.append((tag, float(value), int(step)))
    )
    other.load(path)

    assert ("zclip/restored", 0.0, 4242) in seen, (
        "the not-restored case must be emitted too, or the absence of a point "
        "is ambiguous between 'fresh warmup' and 'this code did not run'"
    )


def test_the_saved_mean_is_a_plain_float(tmp_path: Path) -> None:
    """Keep the payload free of tensors and device state.

    `ZClip` builds `mean`/`var` from `.item()` results today, but the checkpoint
    is loaded with `map_location` and read by tooling that peeks at single keys
    (`peek_checkpoint_peak_lr`). A tensor sneaking in here would tie the
    checkpoint to a device for no benefit.
    """
    trainer = _make_trainer(tmp_path)
    _warm(trainer, mean=8.7, var=2.25)
    state = trainer.zclip_state_dict()

    assert type(state["mean"]) is float
    assert type(state["var"]) is float
    assert all(type(v) is float for v in state["buffer"])
    assert math.isfinite(state["mean"])
