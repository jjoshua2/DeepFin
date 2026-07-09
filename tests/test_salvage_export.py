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
import json
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


def _args(work_dir: Path, out_dir: Path, metric: str) -> argparse.Namespace:
    return argparse.Namespace(
        work_dir=str(work_dir),
        salvage_source_run_id=None,
        salvage_metric=metric,
        salvage_top_n=1,
        salvage_out_dir=str(out_dir),
        salvage_copy_replay=False,
        tune_replay_root_override="",
        num_samples=1,
    )


def _export(work_dir: Path, out_dir: Path, metric: str) -> dict:
    export_seed_pool(_args(work_dir, out_dir, metric))
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["top_n"] == 1
    return manifest["entries"][0]


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
    import os

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
