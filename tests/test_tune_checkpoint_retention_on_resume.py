"""`tune_num_to_keep` must survive a resume, not just a fresh start.

`CheckpointConfig(num_to_keep=...)` is only passed to `tune.Tuner(...)` on the
fresh-start branch of `run_tune`. `Tuner.restore` takes no run config and
unpickles the one saved when the experiment was FIRST created, so a long-lived
run keeps whatever retention it was born with and later YAML edits are
silently ignored.

Observed live on 2026-07-25: `tune_num_to_keep: 6` in the YAML, exactly 2
checkpoints on disk — the code default the experiment was created with. The
YAML comment beside the key warns that PB2/PBT needs older checkpoints for
cloning/exploit and that keeping too few causes missing-checkpoint restore
errors, and the run had already hit one (Ray's restored checkpoint manager
raised evicting an already-deleted checkpoint, which aborted `on_checkpoint`
before the new checkpoint was registered).

This is the same failure `_hotpatch_scheduler_bounds` already exists for —
restored state shadowing current YAML — applied to a different setting.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from chess_anti_engine.tune.harness import _hotpatch_checkpoint_retention


def _fake_tuner(num_to_keep: int | None) -> SimpleNamespace:
    """Mimics `tuner._local_tuner._run_config.checkpoint_config.num_to_keep`."""
    checkpoint_config = SimpleNamespace(num_to_keep=num_to_keep)
    run_config = SimpleNamespace(checkpoint_config=checkpoint_config)
    return SimpleNamespace(_local_tuner=SimpleNamespace(_run_config=run_config))


def test_restored_retention_is_raised_to_the_yaml_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The live case: born with 2, YAML now says 6."""
    tuner = _fake_tuner(2)

    _hotpatch_checkpoint_retention(tuner=tuner, num_to_keep=6)

    assert tuner._local_tuner._run_config.checkpoint_config.num_to_keep == 6
    out = capsys.readouterr().out
    assert "Hotpatched checkpoint retention" in out
    assert "2 → 6" in out, "the line must show what it changed, not just that it did"


def test_a_matching_value_is_left_alone(capsys: pytest.CaptureFixture[str]) -> None:
    tuner = _fake_tuner(6)
    before = tuner._local_tuner._run_config.checkpoint_config

    _hotpatch_checkpoint_retention(tuner=tuner, num_to_keep=6)

    assert tuner._local_tuner._run_config.checkpoint_config is before, (
        "no need to replace an already-correct config"
    )
    assert "already" in capsys.readouterr().out


def test_a_ray_layout_change_cannot_stop_training_from_starting(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Best-effort, exactly like the scheduler hotpatch it mirrors.

    These are Ray private attributes. If an upgrade moves them, the correct
    outcome is a printed note and a run that still starts -- not a crash on a
    disk-retention nicety.
    """
    tuner = SimpleNamespace(_local_tuner=SimpleNamespace())  # no _run_config

    _hotpatch_checkpoint_retention(tuner=tuner, num_to_keep=6)

    assert "non-fatal" in capsys.readouterr().out


def test_zero_disables_the_hotpatch(capsys: pytest.CaptureFixture[str]) -> None:
    """num_to_keep=0 means "keep everything" in Ray; don't fabricate a limit."""
    tuner = _fake_tuner(None)

    _hotpatch_checkpoint_retention(tuner=tuner, num_to_keep=0)

    assert tuner._local_tuner._run_config.checkpoint_config.num_to_keep is None
    assert capsys.readouterr().out == ""


def test_the_patched_attribute_matches_the_real_ray_object() -> None:
    """Pins the private surface this reaches into, so a Ray bump fails loudly here.

    The fake tuner above would keep passing against any Ray version; this is
    what actually ties the test to the installed one. `_run_config` is the
    attribute `TunerInternal.__init__` assigns, and `CheckpointConfig` must
    still accept `num_to_keep`.
    """
    from ray.tune import CheckpointConfig
    from ray.tune.impl.tuner_internal import TunerInternal

    import inspect

    src = inspect.getsource(TunerInternal.__init__)
    assert "self._run_config" in src, (
        "TunerInternal no longer stores _run_config; the hotpatch target moved"
    )
    assert CheckpointConfig(num_to_keep=6).num_to_keep == 6
