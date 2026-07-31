"""salvage-export selection + staleness-guard tests.

Regression coverage for the 2026-07-08 incident: the tune dir's result.json
is a Ray sync COPY that can be observed as a truncated prefix (read racing
the sync rewrite, or a driver killed mid-sync), so `--metric
training_iteration` silently exported iteration 449 while checkpoint_000682
existed on disk. salvage-export must never silently export a snapshot older
than the newest on-disk checkpoint.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from pathlib import Path

from chess_anti_engine.tune.salvage import (
    _checkpoint_index,
    _latest_tune_run_id,
    _row_for_checkpoint,
    export_seed_pool,
)

RUN_ID = "abc12"
TRIAL_NAME = f"train_trial_{RUN_ID}_00000_0_lr=0.0003_2026-01-01_00-00-00"


def _row(it: int, ckpt_name: str | None, **extra: float) -> dict:
    row: dict = {"training_iteration": it, "checkpoint_dir_name": ckpt_name}
    row.update(extra)
    return row


def _write_rows(td: Path, rows: list[dict]) -> None:
    td.mkdir(parents=True, exist_ok=True)
    with (td / "result.json").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _mk_ckpt(td: Path, name: str, *, content: str, pid_nodes: int = 1000) -> Path:
    d = td / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "trainer.pt").write_text(content, encoding="utf-8")
    (d / "pid_state.json").write_text(
        json.dumps({"nodes": pid_nodes, "ema_winrate": 0.5}), encoding="utf-8",
    )
    return d


def _args(work_dir: Path, out_dir: Path, metric: str, *, dry_run: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        work_dir=str(work_dir),
        salvage_source_run_id=None,
        salvage_metric=metric,
        salvage_top_n=1,
        salvage_out_dir=str(out_dir),
        salvage_copy_replay=False,
        salvage_dry_run=bool(dry_run),
        tune_replay_root_override="",
        num_samples=1,
    )


def _export(work_dir: Path, out_dir: Path, metric: str) -> dict:
    export_seed_pool(_args(work_dir, out_dir, metric))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["top_n"] == 1
    return manifest["entries"][0]


def _dry_run(work_dir: Path, out_dir: Path, metric: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        export_seed_pool(_args(work_dir, out_dir, metric, dry_run=True))
    return buf.getvalue()


def _seed_pid(out_dir: Path) -> dict:
    return json.loads(
        (out_dir / "seeds" / "slot_000" / "pid_state.json").read_text(encoding="utf-8"),
    )


def test_happy_path_training_iteration(tmp_path: Path) -> None:
    """Fresh rows whose newest checkpoint exists on disk export exactly it."""
    td = tmp_path / "tune" / TRIAL_NAME
    rows = [
        _row(i, f"checkpoint_{i - 1:06d}", sf_nodes_next=100 * i, pid_ema_winrate=0.6)
        for i in range(1, 11)
    ]
    _write_rows(td, rows)
    for i in (7, 8, 9):
        _mk_ckpt(td, f"checkpoint_{i:06d}", content=f"weights-{i}")

    out = tmp_path / "pool"
    entry = _export(tmp_path, out, "training_iteration")

    assert entry["checkpoint_source"] == "result_row_checkpoint"
    assert entry["checkpoint_dir_name"] == "checkpoint_000009"
    assert entry["training_iteration"] == 10
    assert entry["picked_row_training_iteration"] == 10
    assert entry["stale_result_rows"] is False
    assert entry["newest_disk_checkpoint"] == "checkpoint_000009"
    exported = (out / "seeds" / "slot_000" / "trainer.pt").read_text(encoding="utf-8")
    assert exported == "weights-9"
  # pid_state aligned to the matching row's live values.
    assert _seed_pid(out)["nodes"] == 1000
    assert "nodes<-sf_nodes_next" in entry["pid_state_overrides"]


def test_truncated_rows_export_newest_disk_checkpoint(tmp_path: Path) -> None:
    """The 2026-07-08 incident: truncated result.json must not win over disk."""
    td = tmp_path / "tune" / TRIAL_NAME
  # result.json is a truncated prefix: rows 1..5, checkpoints long pruned.
    rows = [
        _row(i, f"checkpoint_{i - 1:06d}", sf_nodes_next=111, pid_ema_winrate=0.1)
        for i in range(1, 6)
    ]
    _write_rows(td, rows)
  # ... while the trainable has kept writing checkpoints directly to disk.
    _mk_ckpt(td, "checkpoint_000019", content="weights-19")
    _mk_ckpt(td, "checkpoint_000020", content="weights-20", pid_nodes=7777)

    out = tmp_path / "pool"
    entry = _export(tmp_path, out, "training_iteration")

    assert entry["checkpoint_source"] == "newest_disk_checkpoint"
    assert entry["checkpoint_dir_name"] == "checkpoint_000020"
    assert entry["stale_result_rows"] is True
  # True iteration unknown (no row matches the exported checkpoint) — must
  # NOT be mislabelled with the stale row's iteration.
    assert entry["training_iteration"] is None
    assert entry["result_row"] is None
    assert entry["picked_row_training_iteration"] == 5
    exported = (out / "seeds" / "slot_000" / "trainer.pt").read_text(encoding="utf-8")
    assert exported == "weights-20"
  # pid_state must keep the checkpoint's own values, not stale row values.
    assert _seed_pid(out)["nodes"] == 7777
    assert entry["pid_state_overrides"] == []


def test_dry_run_prints_plan_without_writing_files(tmp_path: Path) -> None:
    """Dry-run exposes stale-row checkpoint choice but does not create a pool."""
    td = tmp_path / "tune" / TRIAL_NAME
    rows = [_row(i, f"checkpoint_{i - 1:06d}") for i in range(1, 6)]
    _write_rows(td, rows)
    _mk_ckpt(td, "checkpoint_000020", content="weights-20")

    out = tmp_path / "pool"
    output = _dry_run(tmp_path, out, "training_iteration")

    assert "DRY-RUN: planning 1 seeds" in output
    assert "slot=00" in output
    assert "ckpt=checkpoint_000020" in output
    assert "checkpoint_source=newest_disk_checkpoint" in output
    assert "stale_result_rows=True" in output
    assert "DRY-RUN: no files written" in output
    assert not out.exists()


def test_training_iteration_top_n_uses_stale_trial_newest_disk_checkpoint(tmp_path: Path) -> None:
    """A stale current trial must not lose top-1 selection to older fresh rows."""
    stale_td = tmp_path / "tune" / TRIAL_NAME
    stale_rows = [_row(i, f"checkpoint_{i - 1:06d}") for i in range(1, 6)]
    _write_rows(stale_td, stale_rows)
    _mk_ckpt(stale_td, "checkpoint_000020", content="weights-20")

    fresh_td = (
        tmp_path / "tune"
        / f"train_trial_{RUN_ID}_00001_0_lr=0.0003_2026-01-01_00-00-00"
    )
    fresh_rows = [_row(i, f"checkpoint_{i - 1:06d}") for i in range(1, 11)]
    _write_rows(fresh_td, fresh_rows)
    _mk_ckpt(fresh_td, "checkpoint_000009", content="weights-9")

    out = tmp_path / "pool"
    entry = _export(tmp_path, out, "training_iteration")

    assert Path(entry["source_trial_dir"]).name == TRIAL_NAME
    assert entry["checkpoint_source"] == "newest_disk_checkpoint"
    assert entry["checkpoint_dir_name"] == "checkpoint_000020"
    assert entry["stale_result_rows"] is True
    exported = (out / "seeds" / "slot_000" / "trainer.pt").read_text(encoding="utf-8")
    assert exported == "weights-20"


def test_within_tolerance_keeps_row_checkpoint(tmp_path: Path) -> None:
    """A small row/disk checkpoint-index gap (<= tolerance) is not stale."""
    td = tmp_path / "tune" / TRIAL_NAME
    rows = [_row(i, f"checkpoint_{i - 1:06d}") for i in range(1, 20)]
    _write_rows(td, rows)
    for i in (18, 19, 20):
        _mk_ckpt(td, f"checkpoint_{i:06d}", content=f"weights-{i}")

    out = tmp_path / "pool"
    entry = _export(tmp_path, out, "training_iteration")

    assert entry["checkpoint_source"] == "result_row_checkpoint"
    assert entry["checkpoint_dir_name"] == "checkpoint_000018"
    assert entry["training_iteration"] == 19
    assert entry["stale_result_rows"] is False


def test_best_metric_fallback_records_true_iteration(tmp_path: Path) -> None:
    """Fresh-rows ckpt/ fallback: manifest carries ckpt/'s TRUE (near-live)
    iteration and aligns pid to the matching newest row, not the picked one."""
    td = tmp_path / "tune" / TRIAL_NAME
    rows = [
        _row(
            i, f"checkpoint_{i - 1:06d}",
            opponent_strength=99.0 if i == 3 else 1.0,
            sf_nodes_next=100 * i,
        )
        for i in range(1, 11)
    ]
    _write_rows(td, rows)
  # No checkpoint_* dirs survive pruning; only the mutable ckpt/ remains.
    _mk_ckpt(td, "ckpt", content="weights-live", pid_nodes=42)

    out = tmp_path / "pool"
    entry = _export(tmp_path, out, "opponent_strength")

    assert entry["checkpoint_source"] == "mutable_ckpt_fallback"
    assert entry["checkpoint_dir_name"] == "ckpt"
    assert entry["picked_row_training_iteration"] == 3
    assert entry["picked_row_metric"] == 99.0
  # The exported state is near-live: label it with the newest row's truth.
    assert entry["training_iteration"] == 10
    assert entry["metric"] == 1.0
    assert entry["result_row"]["training_iteration"] == 10
    assert entry["stale_result_rows"] is False
  # pid aligned to the newest row (matches the exported weights).
    assert _seed_pid(out)["nodes"] == 1000


def test_best_metric_stale_rows_fallback_unknown_iteration(tmp_path: Path) -> None:
    """Stale rows + missing row checkpoint: fallback iteration is unknown and
    the manifest must say so instead of trusting the truncated rows."""
    td = tmp_path / "tune" / TRIAL_NAME
    rows = [
        _row(i, f"checkpoint_{i - 1:06d}", opponent_strength=float(i), sf_nodes_next=111)
        for i in range(1, 6)
    ]
    _write_rows(td, rows)
    _mk_ckpt(td, "checkpoint_000020", content="weights-20")
    _mk_ckpt(td, "ckpt", content="weights-live", pid_nodes=42)

    out = tmp_path / "pool"
    entry = _export(tmp_path, out, "opponent_strength")

  # Best-metric export keeps the row pick (older rows are legitimate) but
  # its checkpoint is gone -> ckpt/ fallback with unknown true iteration.
    assert entry["checkpoint_source"] == "mutable_ckpt_fallback"
    assert entry["stale_result_rows"] is True
    assert entry["training_iteration"] is None
    assert entry["metric"] is None
    assert entry["result_row"] is None
    assert entry["picked_row_training_iteration"] == 5
    assert _seed_pid(out)["nodes"] == 42
    assert entry["pid_state_overrides"] == []


def test_latest_run_id_mtime_fallback(tmp_path: Path) -> None:
    """Without pbt_policy files, the newest trial dir's run id wins."""
    tune_dir = tmp_path / "tune"
    old = tune_dir / "train_trial_old01_00000_0_lr=0.0003_2026-01-01_00-00-00"
    new = tune_dir / f"train_trial_{RUN_ID}_00000_0_lr=0.0003_2026-02-01_00-00-00"
    old.mkdir(parents=True)
    new.mkdir(parents=True)

    os.utime(old, (1_000_000, 1_000_000))
    os.utime(new, (2_000_000, 2_000_000))
    assert _latest_tune_run_id(tune_dir) == RUN_ID


def test_checkpoint_index_parsing() -> None:
    assert _checkpoint_index("checkpoint_000686") == 686
    assert _checkpoint_index("checkpoint_0") == 0
    assert _checkpoint_index("ckpt") is None
    assert _checkpoint_index("") is None


def test_row_for_checkpoint_picks_max_iter() -> None:
  # Post-resume, Ray can reuse a checkpoint dir for consecutive rows
  # (live trial 2026-07-08: rows 690 AND 691 -> checkpoint_000686).
    rows = [
        _row(690, "checkpoint_000686"),
        _row(691, "checkpoint_000686"),
        _row(689, "checkpoint_000685"),
    ]
    match = _row_for_checkpoint(rows, "checkpoint_000686")
    assert match is not None
    assert match["training_iteration"] == 691
    assert _row_for_checkpoint(rows, "checkpoint_000000") is None


# ── replay-shard source selection (2026-07-30) ───────────────────────────────
#
# The distributed layout keeps shards under
# `tune_replay_root_override/<trial>/replay_shards`, while the TRIAL dir holds
# an EMPTY `replay_shards/`. Selecting the source with `is_dir()` therefore
# succeeded vacuously on the empty primary and never tried the override, so
# every "revert point" banked was weights-only while reporting the fact only as
# a 0 in the manifest. Verified against production 2026-07-30: the pool banked
# that day recorded copied_replay_shards=0 with 839 shards (3.4G) unread.


def _replay_args(
    work_dir: Path, out_dir: Path, override: str, *, copy_replay: bool = True,
) -> argparse.Namespace:
    a = _args(work_dir, out_dir, "training_iteration")
    a.salvage_copy_replay = copy_replay
    a.tune_replay_root_override = override
    return a


def _mk_run(tmp_path: Path) -> tuple[Path, Path]:
    work = tmp_path / "work"
    td = work / "tune" / TRIAL_NAME
    _write_rows(td, [_row(7, "checkpoint_000007")])
    _mk_ckpt(td, "checkpoint_000007", content="w7")
    return work, td


def _mk_shards(root: Path, n: int) -> Path:
    """`n` zarr-style shard dirs, the layout iter_shard_paths recognises."""
    root.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        d = root / f"shard_{i:06d}.zarr"
        d.mkdir(parents=True, exist_ok=True)
        (d / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")
    return root


def test_empty_trial_replay_dir_falls_through_to_override(tmp_path: Path) -> None:
    """THE BUG. An EMPTY `replay_shards/` in the trial dir must not shadow the
    override that actually holds the shards."""
    work, td = _mk_run(tmp_path)
    (td / "replay_shards").mkdir(parents=True, exist_ok=True)   # empty decoy
    override = tmp_path / "replayroot"
    _mk_shards(override / TRIAL_NAME / "replay_shards", 3)

    out = tmp_path / "pool"
    export_seed_pool(_replay_args(work, out, str(override)))
    entry = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["entries"][0]

    assert entry["copied_replay_shards"] == 3
    assert entry["replay_shard_source"].startswith(str(override))
    # Both paths recorded, so a future zero is diagnosable from the pool alone.
    assert len(entry["replay_shard_paths_tried"]) == 2
    assert len(list((out / entry["seed_dir"] / "replay_shards").iterdir())) == 3


def test_production_answer_wins_over_a_stale_trial_dir(tmp_path: Path) -> None:
    """When the override is set, PRODUCTION writes there — so it is the live
    window and must win even if the trial dir also holds shards.

    Selecting merely "whichever is non-empty, trial dir first" would prefer a
    stale decoy and report a plausible non-zero count. That the trial dir's
    `replay_shards/` exists at all is evidence something other than the
    trainable writes there.
    """
    work, td = _mk_run(tmp_path)
    _mk_shards(td / "replay_shards", 2)          # stale decoy
    override = tmp_path / "replayroot"
    _mk_shards(override / TRIAL_NAME / "replay_shards", 9)   # live window

    out = tmp_path / "pool"
    export_seed_pool(_replay_args(work, out, str(override)))
    entry = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["entries"][0]

    assert entry["copied_replay_shards"] == 9
    assert entry["replay_shard_source"].startswith(str(override))


def test_trial_dir_is_used_when_no_override_is_configured(tmp_path: Path) -> None:
    """Control: with no override, production's answer IS the trial dir, so the
    fix must not redirect anywhere else."""
    work, td = _mk_run(tmp_path)
    _mk_shards(td / "replay_shards", 2)

    out = tmp_path / "pool"
    export_seed_pool(_replay_args(work, out, ""))
    entry = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["entries"][0]

    assert entry["copied_replay_shards"] == 2
    assert entry["replay_shard_source"] == str(td / "replay_shards")


def test_zero_shards_fails_loudly_instead_of_banking_a_dud(tmp_path: Path) -> None:
    """A rollback point with no replay window is not a rollback point. When
    copy-replay is requested and nothing is found anywhere, the export must
    FAIL rather than hand back a pool that looks complete."""
    import pytest

    work, td = _mk_run(tmp_path)
    (td / "replay_shards").mkdir(parents=True, exist_ok=True)   # empty
    override = tmp_path / "replayroot"                          # nothing there

    out = tmp_path / "pool"
    with pytest.raises(SystemExit) as ei:
        export_seed_pool(_replay_args(work, out, str(override)))
    msg = str(ei.value)
    assert "ZERO replay shards" in msg
    assert "--no-copy-replay" in msg          # names the deliberate opt-out
    assert str(td / "replay_shards") in msg   # names every path tried
    assert str(override) in msg


def test_no_copy_replay_is_still_allowed_to_export_nothing(tmp_path: Path) -> None:
    """The opt-out must remain silent: --no-copy-replay is the supported way to
    bank weights+optimizer only, and must not trip the new failure."""
    work, td = _mk_run(tmp_path)
    (td / "replay_shards").mkdir(parents=True, exist_ok=True)

    out = tmp_path / "pool"
    export_seed_pool(_replay_args(work, out, "", copy_replay=False))
    entry = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["entries"][0]
    assert entry["copied_replay_shards"] == 0
    assert entry["replay_shard_source"] == ""
    assert entry["replay_shard_paths_tried"] == []


def test_preflight_aborts_before_writing_anything(tmp_path: Path) -> None:
    """F1: a bad slot must not abort AFTER earlier slots paid multi-GB copies.

    SystemExit is a BaseException, so raising it inside the per-slot
    `except Exception` isolation would slip past it and leave orphan seed dirs
    with NO manifest.json. Resolving every slot up front means the failure
    happens before out_dir exists.
    """
    import pytest

    work = tmp_path / "work"
    good = f"train_trial_{RUN_ID}_00000_0_lr=0.0003_2026-01-01_00-00-00"
    bad = f"train_trial_{RUN_ID}_00001_0_lr=0.0003_2026-01-01_00-00-00"
    for name, it in ((good, 9), (bad, 7)):
        td = work / "tune" / name
        _write_rows(td, [_row(it, f"checkpoint_{it:06d}")])
        _mk_ckpt(td, f"checkpoint_{it:06d}", content=f"w{it}")
        (td / "replay_shards").mkdir(parents=True, exist_ok=True)
    override = tmp_path / "replayroot"
    _mk_shards(override / good / "replay_shards", 5)   # only the GOOD slot has any

    out = tmp_path / "pool"
    a = _replay_args(work, out, str(override))
    a.salvage_top_n = 2
    with pytest.raises(SystemExit) as ei:
        export_seed_pool(a)

    msg = str(ei.value)
    assert "ZERO replay shards" in msg
    assert bad in msg
    assert "1 of 2 slot(s)" in msg
    assert "Nothing has been written" in msg
    # THE POINT: no partial pool, no orphan seed dirs, no missing manifest.
    assert not out.exists(), f"pre-flight wrote {sorted(p.name for p in out.iterdir())}"


def test_rerun_into_same_out_dir_is_idempotent(tmp_path: Path) -> None:
    """F3: the ledger protocol pins a fixed --out <label>, so re-running into
    the same directory is normal. copytree without pre-clean raises
    FileExistsError, and per-slot isolation then silently DROPS the slot from
    the manifest."""
    work, _td = _mk_run(tmp_path)
    override = tmp_path / "replayroot"
    _mk_shards(override / TRIAL_NAME / "replay_shards", 4)

    out = tmp_path / "pool"
    for _ in range(2):
        export_seed_pool(_replay_args(work, out, str(override)))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["entries"], "slot was dropped from the manifest on re-run"
    assert manifest["entries"][0]["copied_replay_shards"] == 4
    assert len(list((out / "seeds" / "slot_000" / "replay_shards").iterdir())) == 4


def test_dry_run_previews_the_replay_source(tmp_path: Path) -> None:
    """F4: --dry-run is the one way to check an export before paying for it, so
    it must show the replay source and must surface the abort condition."""
    work, td = _mk_run(tmp_path)
    (td / "replay_shards").mkdir(parents=True, exist_ok=True)
    override = tmp_path / "replayroot"
    _mk_shards(override / TRIAL_NAME / "replay_shards", 3)

    a = _replay_args(work, tmp_path / "pool", str(override))
    a.salvage_dry_run = True
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        export_seed_pool(a)
    text = buf.getvalue()
    assert "replay:" in text
    assert str(override) in text
    assert not (tmp_path / "pool").exists()
