"""What a restart carries, and what it used to drop on the floor.

Audit T1 (the opponent-strength EMA dies at every process start) and T10
(``salvage-export`` copies the checkpoint dir and leaves the trial's durable
state behind, so a salvage restart begins with an empty promotion-gate window
while an ordinary ``--resume`` keeps its own).

Both are the house defect: a value that is written, loaded, and then ignored.
So every test here asserts the value ARRIVES on the production path, not that a
function returns something.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from chess_anti_engine.tune._utils import (
    DURABLE_BEST_STATE,
    DURABLE_GATE_STATE,
    SIDECAR_TRIAL_META,
    load_optional_json,
)
from chess_anti_engine.tune.salvage import _export_one_seed, _ReplayPlan, _SeedExportPlan
from chess_anti_engine.tune.trainable import (
    _load_best_state,
    _startup_gate_state,
    _startup_opp_strength_ema,
)
from chess_anti_engine.tune.trainable_init import (
    _apply_restored_trial_meta,
    _restore_from_salvage_pool,
)
from chess_anti_engine.tune.trainable_report import _save_trial_checkpoint
from chess_anti_engine.tune.trial_config import RestoreResult


# --- audit T1: the opponent-strength EMA ----------------------------------


def _save_checkpoint(ckpt_dir: Path, *, opp_strength_ema: float) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _save_trial_checkpoint(
        trainer=SimpleNamespace(save=lambda p: Path(p).write_bytes(b"weights")),
        buf=SimpleNamespace(flush=lambda: None),
        ckpt_dir=ckpt_dir,
        rng=np.random.default_rng(0),
        trial_id="t1",
        trial_dir=ckpt_dir.parent,
        config={"optimizer": "aurora"},
        base_seed=7,
        restore=RestoreResult(),
        iteration_idx=1985,
        current_window=1_500_000,
        holdout_buf=None,
        holdout_frozen=True,
        holdout_generation=2,
        holdout_ruler="v1:full_pass:0123456789abcdef",
        opp_strength_ema=opp_strength_ema,
        Checkpoint=SimpleNamespace(from_directory=lambda d: d),
    )


def test_the_checkpoint_carries_the_ema_and_the_restore_reads_it(
    tmp_path: Path, monkeypatch,
) -> None:
    """Round trip through the two production functions a resume runs: the
    writer that builds trial_meta.json and the reader that fills RestoreResult.
    """
    monkeypatch.setattr(
        "chess_anti_engine.tune.trainable_report.save_holdout_rows",
        lambda **_kw: None,
    )
    ckpt_dir = tmp_path / "checkpoint_001985"
    _save_checkpoint(ckpt_dir, opp_strength_ema=327.1906)

    meta = load_optional_json(ckpt_dir / SIDECAR_TRIAL_META)
    assert meta is not None
    assert meta["opp_strength_ema"] == 327.1906

    rr = RestoreResult()
    _apply_restored_trial_meta(rr, meta)
    assert rr.opp_strength_ema == 327.1906


def test_a_resume_continues_the_ema_instead_of_re_seeding_it() -> None:
    """The line that discarded it. ``_run_pid_and_eval`` re-seeds the series
    from this iteration's raw opponent_strength when it is handed 0.0, so
    anything that zeroes this value IS the re-seed."""
    resumed = _startup_opp_strength_ema(
        restore=RestoreResult(startup_source="checkpoint", opp_strength_ema=327.1906),
        best_json_ema=1.0,
    )
    assert resumed == 327.1906
    assert resumed != 0.0


def test_best_json_is_the_fallback_when_the_checkpoint_predates_the_field(
    tmp_path: Path,
) -> None:
    """Checkpoints written before the EMA rode in trial_meta.json leave the
    restore with nothing, and best.json's copy -- until now read and thrown
    away -- is the next best record."""
    best_state_path = tmp_path / DURABLE_BEST_STATE
    best_state_path.write_text(
        json.dumps({"best_loss": 4.9, "opp_strength_ema": 313.95}), encoding="utf-8",
    )
    _best_loss, best_json_ema, _source = _load_best_state(best_state_path)

    resolved = _startup_opp_strength_ema(
        restore=RestoreResult(startup_source="checkpoint"), best_json_ema=best_json_ema,
    )
    assert resolved == 313.95


def test_a_fresh_start_still_re_seeds() -> None:
    """NEGATIVE CONTROL: with nothing to restore from, the old behaviour must
    survive -- 0.0 is what tells the loop to seed the series from iteration 1.
    A fix that invented a value here would put a number nobody measured into
    the scheduler's objective."""
    assert _startup_opp_strength_ema(
        restore=RestoreResult(startup_source="fresh"), best_json_ema=0.0,
    ) == 0.0


def test_an_exploit_donor_outranks_the_checkpoints_own_ema() -> None:
    """A PB2 exploit clone inherits the DONOR's series (the only writer of this
    field before this change), and that must keep winning over the recipient's
    own checkpoint value."""
    assert _startup_opp_strength_ema(
        restore=RestoreResult(startup_source="exploit_restore", opp_strength_ema=290.0),
        best_json_ema=327.19,
    ) == 290.0


def test_a_corrupt_ema_in_trial_meta_is_ignored_rather_than_propagated() -> None:
    rr = RestoreResult()
    _apply_restored_trial_meta(rr, {"opp_strength_ema": "not-a-number"})
    assert rr.opp_strength_ema is None
    rr2 = RestoreResult()
    _apply_restored_trial_meta(rr2, {"opp_strength_ema": float("nan")})
    assert rr2.opp_strength_ema is None


# --- audit T10: salvage-export and the durable state ----------------------


_GATE_STATE = {
    "matches": 41,
    "holds": 2,
    "window": [{"elo": 3.1}, {"elo": -1.2}],
    "hold": {"active": True, "anchor_refresh_failures": 3},
}


def _trial_dir_with_durable_state(tmp_path: Path) -> tuple[Path, Path]:
    """A trial dir shaped like a live one: a checkpoint dir plus the durable
    files that live one level above it."""
    td = tmp_path / "runs" / "13a9f_00000"
    ckpt_dir = td / "checkpoint_001985"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "trainer.pt").write_bytes(b"weights")
    (ckpt_dir / "pid_state.json").write_text(json.dumps({"wdl_regret": 40.0}), encoding="utf-8")
    (ckpt_dir / "trial_meta.json").write_text(
        json.dumps({"global_iter": 1985, "opp_strength_ema": 327.19}), encoding="utf-8",
    )
    (td / DURABLE_GATE_STATE).write_text(json.dumps(_GATE_STATE), encoding="utf-8")
    (td / DURABLE_BEST_STATE).write_text(
        json.dumps({"best_loss": 4.87, "source": "test_loss"}), encoding="utf-8",
    )
    return td, ckpt_dir


def _export(tmp_path: Path, td: Path, ckpt_dir: Path) -> tuple[Path, dict]:
    out_dir = tmp_path / "pool"
    seeds_dir = out_dir / "seeds"
    seeds_dir.mkdir(parents=True)
    entry = _export_one_seed(
        slot=0,
        picked_metric=323.6,
        picked_it=1985,
        td=td,
        plan=_SeedExportPlan(
            ckpt_dir=ckpt_dir,
            ckpt_source="result_row_checkpoint",
            row=None,
            training_iteration=1985,
            stale_result_rows=False,
            newest_ckpt_name=ckpt_dir.name,
            row_ckpt_name=ckpt_dir.name,
        ),
        metric_key="training_iteration",
        seeds_dir=seeds_dir,
        out_dir=out_dir,
        copy_replay=False,
        replay_plan=_ReplayPlan(source=None, tried=()),
    )
    return out_dir, entry


def test_salvage_export_banks_the_gate_and_best_state(tmp_path: Path) -> None:
    """The export copied the checkpoint's five files and nothing else, so the
    promotion gate's window, its hold budget and its anchor-failure counter
    were dropped by the one restart mode an operator chooses deliberately."""
    td, ckpt_dir = _trial_dir_with_durable_state(tmp_path)
    out_dir, entry = _export(tmp_path, td, ckpt_dir)

    slot_dir = out_dir / entry["seed_dir"]
    assert json.loads((slot_dir / DURABLE_GATE_STATE).read_text()) == _GATE_STATE
    assert json.loads((slot_dir / DURABLE_BEST_STATE).read_text())["best_loss"] == 4.87

    # The manifest says what came along AND what did not: a pool that quietly
    # lacks the anchor is how a restart discovers state loss after the fact.
    assert entry["durable_state_exported"] == [DURABLE_GATE_STATE, DURABLE_BEST_STATE]
    assert "gate_promoted_model.pt" in entry["durable_state_not_exported"]


def test_export_of_a_trial_with_no_gate_state_is_not_an_error(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: the gate is off in production today, so most trials
    have no gate_state.json at all. Absence must export cleanly and say so."""
    td, ckpt_dir = _trial_dir_with_durable_state(tmp_path)
    (td / DURABLE_GATE_STATE).unlink()
    out_dir, entry = _export(tmp_path, td, ckpt_dir)

    assert entry["durable_state_exported"] == [DURABLE_BEST_STATE]
    assert not (out_dir / entry["seed_dir"] / DURABLE_GATE_STATE).exists()


class _TinyTrainer:
    """Enough of a Trainer for the model-only salvage branch."""

    def __init__(self) -> None:
        self.model = nn.Linear(2, 2)
        self.step = 0

    def reset_optimizer_reference_weights(self) -> None:
        return None


def test_an_exported_pool_restores_the_gate_window_end_to_end(tmp_path: Path) -> None:
    """Export a pool, then drive the SALVAGE RESTORE over it and take the same
    decision ``train_trial`` takes at startup. This is the whole T10 loop: a
    salvage restart now finds the window a resume would have kept."""
    td, ckpt_dir = _trial_dir_with_durable_state(tmp_path)
    trainer = _TinyTrainer()
    torch.save(
        {"model": trainer.model.state_dict(), "peak_lr": 3e-5},
        ckpt_dir / "trainer.pt",
    )
    out_dir, entry = _export(tmp_path, td, ckpt_dir)
    (out_dir / "manifest.json").write_text(
        json.dumps({"metric": "training_iteration", "entries": [entry]}), encoding="utf-8",
    )

    trial_dir = tmp_path / "new_trial" / "abcde_00000"
    trial_dir.mkdir(parents=True)
    rr = RestoreResult()
    _restore_from_salvage_pool(
        config={
            "salvage_seed_pool_dir": str(out_dir),
            "lr": 3e-5,
            "salvage_restore_full_trainer_state": False,
        },
        trainer=_TinyTrainer(),
        device="cpu",
        trial_id="abcde_00000",
        trial_dir=trial_dir,
        rr=rr,
    )
    assert rr.restored_gate_state == _GATE_STATE
    # The EMA rides in the checkpoint's trial_meta.json, which the pool copies,
    # so a salvage restart continues the scheduler's metric too (T1).
    assert rr.opp_strength_ema == 327.19

    # And the startup decision uses it, because this trial has no state of its own.
    assert _startup_gate_state(
        gate_state_path=trial_dir / DURABLE_GATE_STATE, restore=rr,
    ) == _GATE_STATE


def test_a_resume_never_takes_a_window_from_a_pool(tmp_path: Path) -> None:
    """NEGATIVE CONTROL, and the one that matters: a trial's OWN gate state
    must win. Seeding over it would hand a running experiment a window earned
    by different weights."""
    own = {"matches": 7, "holds": 0}
    own_path = tmp_path / DURABLE_GATE_STATE
    own_path.write_text(json.dumps(own), encoding="utf-8")

    assert _startup_gate_state(
        gate_state_path=own_path,
        restore=RestoreResult(restored_gate_state=_GATE_STATE),
    ) == own
    # Neither source: an empty dict, which is what a fresh start has always had.
    assert _startup_gate_state(
        gate_state_path=tmp_path / "absent.json", restore=RestoreResult(),
    ) == {}
