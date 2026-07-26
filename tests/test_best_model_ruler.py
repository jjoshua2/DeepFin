"""Best-model tracking must not compare holdout loss against training loss.

They are different rulers. Live on 2026-07-25 they sat ~0.3 nats apart --
train 4.88-4.94, holdout 5.14-5.25 -- because training loss is measured on
batches the model has just fitted.

The holdout buffer is rebuilt empty on every process start, so for the first
several iterations after a restart `test_metrics` is None. The old code then
compared the LOWER training loss against a `best_loss` earned on the holdout,
won automatically, and pinned `best_loss` to the training scale -- after which
no holdout evaluation could ever beat it again.

That is the observed live state: `best.json` held
`{"best_loss": 4.893, "source": "train_loss"}` while every holdout evaluation
came in above 5.14, so best-model tracking had stopped responding to holdout
quality entirely. `best/` is an operator-facing artifact -- the publish-root
`best_model.pt` workers fetch is written by a different path -- so the damage
is a selection that silently means something other than what it says.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chess_anti_engine.tune.trainable import _load_best_state
from chess_anti_engine.tune.trainable_report import (
    _reset_best_on_cross_trial_restore,
    _update_best_model,
)


class _FakeTrainer:
    """Records that a best-model write happened, without touching torch."""

    step = 4242

    def __init__(self) -> None:
        self.saves: list[Path] = []
        self.exports: list[Path] = []

    def save(self, path: Path) -> None:
        self.saves.append(Path(path))

    def export_swa(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"weights")
        self.exports.append(Path(path))


def _metrics(loss: float) -> SimpleNamespace:
    return SimpleNamespace(loss=loss)


def _update(tmp_path: Path, **kw):
    trainer = _FakeTrainer()
    result = _update_best_model(
        trainer=trainer,
        best_dir=tmp_path / "best",
        best_state_path=tmp_path / "best.json",
        iteration_idx=kw.pop("iteration_idx", 1871),
        test_metrics_source_iter=kw.pop("test_metrics_source_iter", 1871),
        holdout_generation=kw.pop("holdout_generation", 0),
        opp_strength_ema=313.9,
        **kw,
    )
    return result, trainer


def test_training_loss_cannot_displace_a_holdout_record(tmp_path: Path) -> None:
    """The live bug, stated directly.

    Post-restart the holdout is empty, so only training loss exists -- and it
    is numerically lower for reasons that have nothing to do with quality.
    """
    (best_loss, source), trainer = _update(
        tmp_path,
        test_metrics=None,
        train_metrics=_metrics(4.8934),
        best_loss=5.1409,
        best_source="test_loss",
    )

    assert (best_loss, source) == (5.1409, "test_loss"), (
        "training loss overwrote a holdout record it cannot be compared to"
    )
    assert trainer.exports == [], "no model should have been written"
    assert not (tmp_path / "best.json").exists()


def test_a_holdout_result_takes_over_from_a_training_loss_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Recovery from the live state: 4.893/train_loss on record, holdout 5.14.

    5.14 is numerically WORSE, and must still be adopted -- otherwise the stale
    training-scale number locks holdout evaluation out permanently, which is
    exactly what had happened.
    """
    (best_loss, source), trainer = _update(
        tmp_path,
        test_metrics=_metrics(5.1409),
        train_metrics=_metrics(4.8825),
        best_loss=4.8934,
        best_source="train_loss",
    )

    assert (best_loss, source) == (5.1409, "test_loss")
    assert len(trainer.exports) == 1
    assert json.loads((tmp_path / "best.json").read_text())["source"] == "test_loss"
    assert "handover" in capsys.readouterr().out


def test_a_holdout_improvement_is_taken(tmp_path: Path) -> None:
    (best_loss, source), trainer = _update(
        tmp_path,
        test_metrics=_metrics(5.1000),
        train_metrics=_metrics(4.88),
        best_loss=5.1409,
        best_source="test_loss",
    )

    assert (best_loss, source) == (5.1000, "test_loss")
    assert len(trainer.exports) == 1


def test_a_holdout_regression_is_rejected(tmp_path: Path) -> None:
    (best_loss, source), trainer = _update(
        tmp_path,
        test_metrics=_metrics(5.2481),
        train_metrics=_metrics(4.88),
        best_loss=5.1409,
        best_source="test_loss",
    )

    assert (best_loss, source) == (5.1409, "test_loss")
    assert trainer.exports == []


def test_training_loss_still_improves_a_training_loss_record(tmp_path: Path) -> None:
    """Before the first holdout exists, training loss is all there is."""
    (best_loss, source), trainer = _update(
        tmp_path,
        test_metrics=None,
        train_metrics=_metrics(4.8777),
        best_loss=4.8934,
        best_source="train_loss",
    )

    assert (best_loss, source) == (4.8777, "train_loss")
    assert len(trainer.exports) == 1


def test_handover_happens_once_not_on_every_iteration(tmp_path: Path) -> None:
    """After the handover the record is on the holdout scale, so the next
    worse holdout result must be rejected normally rather than re-adopted."""
    (best_loss, source), _ = _update(
        tmp_path,
        test_metrics=_metrics(5.14),
        train_metrics=_metrics(4.88),
        best_loss=4.8934,
        best_source="train_loss",
    )
    (best_loss2, source2), trainer2 = _update(
        tmp_path,
        test_metrics=_metrics(5.24),
        train_metrics=_metrics(4.88),
        best_loss=best_loss,
        best_source=source,
    )

    assert (best_loss2, source2) == (5.14, "test_loss")
    assert trainer2.exports == []


def test_no_metrics_at_all_changes_nothing(tmp_path: Path) -> None:
    """First iteration, before any training step has produced a loss."""
    (best_loss, source), trainer = _update(
        tmp_path,
        test_metrics=None,
        train_metrics=None,
        best_loss=float("inf"),
        best_source="train_loss",
    )

    assert best_loss == float("inf")
    assert source == "train_loss"
    assert trainer.exports == []


def test_the_first_ever_result_is_accepted_from_either_ruler(tmp_path: Path) -> None:
    for metrics_kw, expected in (
        ({"test_metrics": _metrics(5.14), "train_metrics": None}, "test_loss"),
        ({"test_metrics": None, "train_metrics": _metrics(4.88)}, "train_loss"),
    ):
        (best_loss, source), trainer = _update(
            tmp_path, best_loss=float("inf"), best_source="train_loss", **metrics_kw
        )
        assert source == expected
        assert best_loss < float("inf")
        assert len(trainer.exports) == 1


# ---------------------------------------------------------------------------
# Reading the record back
# ---------------------------------------------------------------------------


def test_the_source_is_read_back_from_best_json(tmp_path: Path) -> None:
    path = tmp_path / "best.json"
    path.write_text(json.dumps({
        "best_loss": 4.893426540570381,
        "iter": 1860,
        "opp_strength_ema": 313.95355726270225,
        "source": "train_loss",
        "trainer_step": 75147,
    }))

    assert _load_best_state(path) == (4.893426540570381, 313.95355726270225, "train_loss")


def test_a_best_json_predating_the_source_field_defaults_to_train_loss(
    tmp_path: Path,
) -> None:
    """Conservative direction: assuming `train_loss` lets the next holdout
    evaluation take over the ruler. Assuming `test_loss` would lock it out --
    the exact failure being fixed."""
    path = tmp_path / "best.json"
    path.write_text(json.dumps({"best_loss": 4.9, "opp_strength_ema": 1.0}))

    assert _load_best_state(path)[2] == "train_loss"


def test_an_unrecognised_source_is_not_trusted(tmp_path: Path) -> None:
    path = tmp_path / "best.json"
    path.write_text(json.dumps({"best_loss": 4.9, "source": "eval_loss"}))

    assert _load_best_state(path)[2] == "train_loss"


def test_a_missing_best_json_starts_at_infinity(tmp_path: Path) -> None:
    assert _load_best_state(tmp_path / "nope.json") == (float("inf"), 0.0, "train_loss")


# --- A non-finite candidate must never become the record -----------------
#
# The same-ruler test rejects NaN on its own (`nan < x` is False), but the
# cross-ruler handover adopts unconditionally. One NaN written to best.json is
# permanent: every later `finite < nan - 1e-12` is False, across restarts,
# so the record becomes unbeatable and best-model tracking stops for good.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_holdout_loss_cannot_take_over_the_ruler(
    tmp_path: Path, bad: float
) -> None:
    (best_loss, source), trainer = _update(
        tmp_path,
        test_metrics=_metrics(bad),
        train_metrics=_metrics(4.88),
        best_loss=4.89,
        best_source="train_loss",
    )

    assert best_loss == 4.89
    assert source == "train_loss"
    assert trainer.exports == []
    assert not (tmp_path / "best.json").exists()


def test_a_non_finite_training_loss_cannot_take_over_the_ruler(
    tmp_path: Path,
) -> None:
    (best_loss, source), trainer = _update(
        tmp_path,
        test_metrics=None,
        train_metrics=_metrics(float("nan")),
        best_loss=float("inf"),
        best_source="train_loss",
    )

    assert best_loss == float("inf")
    assert source == "train_loss"
    assert trainer.exports == []


def test_a_finite_holdout_still_takes_over_after_a_rejected_nan(
    tmp_path: Path,
) -> None:
    """The guard must reject the NaN, not disable the handover."""
    (best_loss, source), _ = _update(
        tmp_path,
        test_metrics=_metrics(float("nan")),
        train_metrics=_metrics(4.88),
        best_loss=4.89,
        best_source="train_loss",
    )
    (best_loss, source), trainer = _update(
        tmp_path,
        test_metrics=_metrics(5.14),
        train_metrics=_metrics(4.88),
        best_loss=best_loss,
        best_source=source,
    )

    assert (best_loss, source) == (5.14, "test_loss")
    assert len(trainer.exports) == 1


# --- The holdout number and the exported weights are one iteration apart --
#
# With distributed_async_test_eval (production), the result collected during
# iteration N was computed on the snapshot from iteration N-1. Adoption is
# still right -- one iteration of drift is far smaller than the ~0.3-nat ruler
# gap -- but the artifact has to say so.


def test_best_json_records_which_iteration_the_holdout_number_describes(
    tmp_path: Path,
) -> None:
    _update(
        tmp_path,
        test_metrics=_metrics(5.14),
        train_metrics=_metrics(4.88),
        best_loss=float("inf"),
        best_source="test_loss",
        iteration_idx=1871,
        test_metrics_source_iter=1870,
    )

    written = json.loads((tmp_path / "best.json").read_text())
    assert written["iter"] == 1871
    assert written["eval_source_iter"] == 1870
    assert written["source"] == "test_loss"


def test_a_training_loss_record_describes_the_current_iteration(
    tmp_path: Path,
) -> None:
    """`train_metrics` is measured on the weights being exported, so there is
    no skew to record -- and the field must not leak a stale holdout iter."""
    _update(
        tmp_path,
        test_metrics=None,
        train_metrics=_metrics(4.88),
        best_loss=float("inf"),
        best_source="train_loss",
        iteration_idx=1871,
        test_metrics_source_iter=-1,
    )

    written = json.loads((tmp_path / "best.json").read_text())
    assert written["eval_source_iter"] == 1871


# --- A cross-trial restore invalidates this trial's best record ----------


def _reset(tmp_path: Path, restore, *, seed_disk: bool = True):
    best_dir = tmp_path / "best"
    best_dir.mkdir(exist_ok=True)
    state = tmp_path / "best.json"
    if seed_disk:
        state.write_text('{"best_loss": 5.09, "source": "test_loss"}')
        (best_dir / "best_model.pt").write_bytes(b"recipient weights")
        (best_dir / "trainer.pt").write_bytes(b"recipient trainer")
    out = _reset_best_on_cross_trial_restore(
        restore=restore, best_loss=5.09, best_source="test_loss",
        best_dir=best_dir, best_state_path=state,
    )
    return out, state, best_dir


def test_an_exploit_restore_clears_the_recipients_best_record(tmp_path: Path) -> None:
    """The donor's weights must not have to beat the recipient's number."""
    restore = SimpleNamespace(
        cross_trial_restore=True, startup_source="exploit_restore",
    )

    out, state, best_dir = _reset(tmp_path, restore)

    assert out == (float("inf"), "train_loss")
    assert not state.exists(), "the on-disk record still describes the dead lineage"
    assert list(best_dir.iterdir()) == [], "stale best/ weights survived the reset"
    assert best_dir.is_dir(), "best/ must stay usable for the next write"


def test_the_reset_is_not_left_to_a_later_iteration(tmp_path: Path) -> None:
    """If no metric is accepted this iteration, nothing else clears the record,
    and the NEXT ordinary resume reloads it -- cross_trial_restore is false by
    then. So the delete has to happen here, not as a side effect of an update."""
    restore = SimpleNamespace(cross_trial_restore=True, startup_source="exploit_restore")
    _, state, _ = _reset(tmp_path, restore)

    from chess_anti_engine.tune.trainable import _load_best_state

    assert _load_best_state(state) == (float("inf"), 0.0, "train_loss")


def test_an_ordinary_resume_keeps_the_best_record(tmp_path: Path) -> None:
    restore = SimpleNamespace(cross_trial_restore=False, startup_source="checkpoint")

    out, state, best_dir = _reset(tmp_path, restore)

    assert out == (5.09, "test_loss")
    assert state.exists(), "an ordinary resume must not delete the record"
    assert (best_dir / "best_model.pt").exists()


def test_a_restore_object_without_the_flag_is_treated_as_an_ordinary_resume(
    tmp_path: Path,
) -> None:
    """Fail towards keeping the record: clearing it on every start would make
    best-model tracking useless, which is worse than a stale lineage."""
    out, state, _ = _reset(tmp_path, SimpleNamespace())

    assert out == (5.09, "test_loss")
    assert state.exists()


def test_best_json_records_the_holdout_generation(tmp_path: Path) -> None:
    """Which holdout the recorded loss was measured on. Now that the counter
    is durable (it rides in the checkpoint's trial_meta.json), this field is
    the ruler identity the comparison below keys off."""
    _update(
        tmp_path,
        test_metrics=_metrics(5.14), train_metrics=_metrics(4.88),
        best_loss=float("inf"), best_source="test_loss", holdout_generation=3,
    )

    assert json.loads((tmp_path / "best.json").read_text())["holdout_generation"] == 3


def _record(tmp_path: Path, **fields) -> None:
    """An existing best.json, as a prior iteration would have left it."""
    (tmp_path / "best.json").write_text(
        json.dumps({"best_loss": 5.1409, "source": "test_loss", **fields}), encoding="utf-8",
    )


def test_a_new_holdout_generation_forces_a_handover(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A drift reset -- or a restart that could not restore the sidecar --
    rebuilds the holdout from a different sample. Two `test_loss` values from
    different generations are different rulers for the same reason train and
    holdout loss are, so the record is adopted, not beaten: 5.2481 is WORSE
    than the 5.1409 on record and must still take over."""
    _record(tmp_path, holdout_generation=3)

    (best_loss, source), trainer = _update(
        tmp_path,
        test_metrics=_metrics(5.2481), train_metrics=_metrics(4.88),
        best_loss=5.1409, best_source="test_loss", holdout_generation=4,
    )

    assert (best_loss, source) == (5.2481, "test_loss")
    assert len(trainer.exports) == 1
    assert "generation" in capsys.readouterr().out


def test_the_same_generation_is_an_ordinary_comparison(tmp_path: Path) -> None:
    """The common path -- an ordinary resume restores the holdout AND its
    generation, so a restart must not trigger a handover. This is what the
    counter could not be trusted for while it was in-memory only."""
    _record(tmp_path, holdout_generation=4)

    (best_loss, source), trainer = _update(
        tmp_path,
        test_metrics=_metrics(5.2481), train_metrics=_metrics(4.88),
        best_loss=5.1409, best_source="test_loss", holdout_generation=4,
    )

    assert (best_loss, source) == (5.1409, "test_loss")
    assert trainer.exports == []


def test_a_record_with_no_generation_is_left_to_the_ordinary_rules(tmp_path: Path) -> None:
    """A best.json written before the field existed reads as unknown, not as
    generation 0 -- zero is a real generation, and claiming a match there
    would suppress the handover exactly when the record is oldest. Unknown
    means: do not force a handover, just compare as before."""
    _record(tmp_path)

    (best_loss, source), trainer = _update(
        tmp_path,
        test_metrics=_metrics(5.2481), train_metrics=_metrics(4.88),
        best_loss=5.1409, best_source="test_loss", holdout_generation=4,
    )

    assert (best_loss, source) == (5.1409, "test_loss")
    assert trainer.exports == []


def test_a_generation_change_cannot_resurrect_a_training_loss_record(tmp_path: Path) -> None:
    """The generation rule is about two holdout numbers. It must not become a
    back door for training loss to overwrite a holdout record."""
    _record(tmp_path, holdout_generation=3)

    (best_loss, source), trainer = _update(
        tmp_path,
        test_metrics=None, train_metrics=_metrics(4.8934),
        best_loss=5.1409, best_source="test_loss", holdout_generation=4,
    )

    assert (best_loss, source) == (5.1409, "test_loss")
    assert trainer.exports == []
