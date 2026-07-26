"""A warm start must not run a donor net far above its converged LR.

2026-07-11 the 512x16 swap adopted a net converged at ``peak_lr`` 3e-5 into a
trial configured ``lr: 0.0003``. ``Trainer.set_peak_lr`` rescales every base LR
by ``new/old``, so the matrix group started at 6e-3 -- double the 0.003 this
project had already recorded as model-destroying. Measured cost: -494 Elo over
74 iterations, then 272 iterations spent recovering as the schedule decayed
back toward the donor's regime. The donor's ``peak_lr`` was in the checkpoint
the entire time and nothing read it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from chess_anti_engine.tune.trainable_init import (
    guard_warm_start_lr,
    peek_checkpoint_peak_lr,
)


def _write_ckpt(tmp_path: Path, **extra: object) -> Path:
    p = tmp_path / "trainer.pt"
    torch.save({"model": {"w": torch.zeros(2)}, **extra}, str(p))
    return p


def test_it_reproduces_the_real_swap_and_refuses(tmp_path: Path) -> None:
    """The exact numbers from trial 4c17c."""
    ckpt = _write_ckpt(tmp_path, peak_lr=3e-05)

    with pytest.raises(ValueError, match=r"10\.0x"):
        guard_warm_start_lr(ckpt, {"lr": 0.0003}, source="salvage pool")


def test_the_message_names_the_lr_that_would_be_allowed(tmp_path: Path) -> None:
    ckpt = _write_ckpt(tmp_path, peak_lr=3e-05)

    with pytest.raises(ValueError, match="warm start") as exc:
        guard_warm_start_lr(ckpt, {"lr": 0.0003}, source="salvage pool")

    msg = str(exc.value)
    assert "6e-05" in msg
    assert "warm_start_lr_max_ratio: 0" in msg
    assert "salvage pool" in msg


def test_a_restart_at_the_donors_own_lr_is_allowed(tmp_path: Path) -> None:
    ckpt = _write_ckpt(tmp_path, peak_lr=3e-05)

    guard_warm_start_lr(ckpt, {"lr": 3e-05}, source="salvage pool")


def test_a_modest_increase_within_the_limit_is_allowed(tmp_path: Path) -> None:
    ckpt = _write_ckpt(tmp_path, peak_lr=3e-05)

    guard_warm_start_lr(ckpt, {"lr": 6e-05}, source="salvage pool")


def test_the_limit_is_configurable_and_zero_disables_it(tmp_path: Path) -> None:
    ckpt = _write_ckpt(tmp_path, peak_lr=3e-05)

  # Deliberate override -- the operator gets to make this call.
    guard_warm_start_lr(
        ckpt, {"lr": 0.0003, "warm_start_lr_max_ratio": 0}, source="salvage pool",
    )
  # ... and a wider limit admits what the default rejects.
    guard_warm_start_lr(
        ckpt, {"lr": 0.0003, "warm_start_lr_max_ratio": 20}, source="salvage pool",
    )


def test_an_older_checkpoint_without_peak_lr_does_not_block_startup(
    tmp_path: Path,
) -> None:
    """Absent metadata must not brick a restart -- fail open, not closed."""
    ckpt = _write_ckpt(tmp_path)

    assert peek_checkpoint_peak_lr(ckpt) is None
    guard_warm_start_lr(ckpt, {"lr": 0.0003}, source="checkpoint")


def test_a_nonsense_peak_lr_is_ignored_rather_than_dividing_by_zero(
    tmp_path: Path,
) -> None:
    ckpt = _write_ckpt(tmp_path, peak_lr=0.0)

    assert peek_checkpoint_peak_lr(ckpt) is None
    guard_warm_start_lr(ckpt, {"lr": 0.0003}, source="checkpoint")


def test_an_unreadable_checkpoint_does_not_raise_from_the_guard(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "trainer.pt"
    bad.write_bytes(b"not a torch file")

    assert peek_checkpoint_peak_lr(bad) is None
    guard_warm_start_lr(bad, {"lr": 0.0003}, source="checkpoint")


def test_a_config_with_no_lr_key_is_skipped(tmp_path: Path) -> None:
    ckpt = _write_ckpt(tmp_path, peak_lr=3e-05)

    guard_warm_start_lr(ckpt, {}, source="checkpoint")


def test_peek_reads_peak_lr_without_materializing_the_model(tmp_path: Path) -> None:
    p = tmp_path / "trainer.pt"
    torch.save({"model": {"w": torch.zeros(1024, 1024)}, "peak_lr": 1.5e-4}, str(p))

    assert peek_checkpoint_peak_lr(p) == pytest.approx(1.5e-4)


def test_both_restore_paths_call_the_guard() -> None:
    """Pins the wiring: a helper nothing calls would have saved nothing."""
    import inspect

    from chess_anti_engine.tune import trainable_init

    ray_path = inspect.getsource(trainable_init._restore_from_ray_checkpoint)
    salvage_path = inspect.getsource(trainable_init._restore_from_salvage_pool)

    assert 'guard_warm_start_lr(maybe, config, source="checkpoint")' in ray_path
    assert 'guard_warm_start_lr(maybe, config, source="salvage pool")' in salvage_path


def test_the_new_key_is_accepted_by_the_yaml_validator() -> None:
    """An unrecognised key rejects the WHOLE live reload, so it must be listed."""
    from chess_anti_engine.utils import config_yaml

    src = inspect_source(config_yaml)

    assert src.count('"warm_start_lr_max_ratio"') >= 2


def inspect_source(mod: object) -> str:
    import inspect

    return inspect.getsource(mod)  # pyright: ignore[reportArgumentType]
