"""Fresh-start pruning must never delete the newest experiment with a checkpoint.

A fresh launch that fails after `Tuner.fit` wrote its experiment_state leaves a
NEWER state file pointing at an errored trial with no checkpoint. When such a
file used up a `tune_keep_last_experiments` slot, the next fresh launch kept
only it: with `keep_last=2, reserve_new=True` the real experiment's trial dir
and every checkpoint in it were deleted. Two salvage-restart attempts in a row
(always fresh) are enough.

The fixtures follow the production layout under `<work_dir>/tune`: trial dirs
`train_trial_<id>_00000_0_lr=..._<ts>`, Ray checkpoints as
`checkpoint_NNNNNN/trainer.pt` plus sidecars, the trainable's `ckpt/` staging
dir, `error.txt`/`error.pkl` in an errored trial, one
`basic-variant-state-<ts>.json` and `pbt_policy_<trial_id>.txt` beside each
`experiment_state-<ts>.json`.
"""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from chess_anti_engine.tune import harness
from chess_anti_engine.tune.harness import (
    _cleanup_old_tune_experiments,
    _delete_obsolete_state_files,
    _prune_orphan_pb2_policy_logs,
    _trial_dirs_referenced_by,
)


def _main_cleanup(*, tune_dir: Path, keep_last: int, reserve_new: bool = False) -> None:
    """Verbatim body of `_cleanup_old_tune_experiments` on main before this fix.

    The reference for "ordinary rotation is unchanged": it keeps the newest
    `keep_last` state files whatever they reference.
    """
    keep_last = int(keep_last)
    if keep_last <= 0 or not tune_dir.exists():
        return
    if reserve_new:
        keep_last -= 1

    state_files = sorted(tune_dir.glob("experiment_state-*.json"))
    if len(state_files) <= keep_last:
        return

    keep = state_files[-keep_last:] if keep_last else []

    protected: set[str] = set()
    protection_is_complete = True
    for p in keep:
        names = _trial_dirs_referenced_by(p)
        if names is None:
            protection_is_complete = False
        else:
            protected |= names

    ts_re = re.compile(r"^experiment_state-(?P<ts>.+)\.json$")
    keep_set = set(keep)
    for p in state_files:
        if p in keep_set:
            continue
        m = ts_re.match(p.name)
        if m:
            _delete_obsolete_state_files(tune_dir=tune_dir, ts=m.group("ts"))

    if not protection_is_complete:
        return

    for d in sorted(tune_dir.glob("train_trial_*")):
        if not d.is_dir() or d.name in protected:
            continue
        shutil.rmtree(d, ignore_errors=True)

    _prune_orphan_pb2_policy_logs(tune_dir=tune_dir)


def _trial_name(trial_id: str, ts: str) -> str:
    return f"train_trial_{trial_id}_00000_0_lr=0.0003_{ts}"


def _write_state(tune_dir: Path, ts: str, *, trials: list[str], status: str) -> None:
    """Ray-shaped experiment state plus its basic-variant-state sibling."""
    errored = status == "ERROR"
    payload = {
        "trial_data": [
            [
                json.dumps({
                    "stub": False,
                    "trainable_name": "train_trial",
                    "trial_id": name.split("_")[2] + "_00000",
                    "status": status,
                    "relative_logdir": name,
                    "storage": {"_type": "CLOUDPICKLE_FALLBACK", "value": "8005"},
                }),
                json.dumps({
                    "start_time": 0.0,
                    "num_failures": int(errored),
                    "error_filename": "error.txt" if errored else None,
                    "pickled_error_filename": "error.pkl" if errored else None,
                }),
            ]
            for name in trials
        ],
        "runner_data": {},
        "stats": {},
    }
    (tune_dir / f"experiment_state-{ts}.json").write_text(json.dumps(payload))
    (tune_dir / f"basic-variant-state-{ts}.json").write_text("{}")


def _healthy_trial(root: Path, name: str) -> Path:
    d = root / name
    for idx in (249, 250):
        ck = d / f"checkpoint_{idx:06d}"
        ck.mkdir(parents=True)
        (ck / "trainer.pt").write_bytes(f"weights {name} {idx}".encode())
        for side in ("pid_state.json", "rng_state.json", "trial_meta.json"):
            (ck / side).write_text("{}")
    (d / "ckpt").mkdir()
    (d / "ckpt" / "trainer.pt").write_bytes(b"staging")
    (d / "events.out.tfevents.1").write_bytes(b"tb")
    return d


def _failed_trial(root: Path, name: str, *, partial: bool = False) -> Path:
    """Errored before Ray persisted any checkpoint.

    `partial` adds what a crash during the first save can leave: the staging
    `ckpt/trainer.pt` and a `checkpoint_000000` without `trainer.pt`. Neither is
    a checkpoint a restart could use.
    """
    d = root / name
    d.mkdir(parents=True)
    (d / "error.txt").write_text("RuntimeError: salvage seed missing trainer.pt")
    (d / "error.pkl").write_bytes(b"pickle")
    (d / "events.out.tfevents.2").write_bytes(b"tb")
    if partial:
        (d / "ckpt").mkdir()
        (d / "ckpt" / "trainer.pt").write_bytes(b"staging only")
        (d / "checkpoint_000000").mkdir()
        (d / "checkpoint_000000" / "rng_state.json").write_text("{}")
    return d


def _experiment(
    tune_dir: Path, trial_id: str, ts: str, *, healthy: bool,
    resumes: tuple[str, ...] = (), partial: bool = False,
) -> str:
    name = _trial_name(trial_id, ts)
    if healthy:
        _healthy_trial(tune_dir, name)
    else:
        _failed_trial(tune_dir, name, partial=partial)
    for stamp in (ts, *resumes):
        _write_state(tune_dir, stamp, trials=[name], status="RUNNING" if healthy else "ERROR")
    (tune_dir / f"pbt_policy_{trial_id}_00000.txt").write_text("policy")
    return name


def _survivors(tune_dir: Path) -> set[str]:
    return {
        str(p.relative_to(tune_dir)) for p in tune_dir.rglob("*")
        if p.is_file() or p.is_symlink()
    }


def _owned_by(survivors: set[str], trial_id: str, ts: str) -> set[str]:
    """Every surviving file that belongs to one experiment."""
    return {
        s for s in survivors
        if s.startswith(_trial_name(trial_id, ts))
        or s in (
            f"experiment_state-{ts}.json", f"basic-variant-state-{ts}.json",
            f"pbt_policy_{trial_id}_00000.txt",
        )
    }


def _assert_experiment_intact(tune_dir: Path, trial_id: str, ts: str) -> None:
    name = _trial_name(trial_id, ts)
    assert (tune_dir / f"experiment_state-{ts}.json").is_file()
    assert (tune_dir / f"basic-variant-state-{ts}.json").is_file()
    assert (tune_dir / f"pbt_policy_{trial_id}_00000.txt").is_file()
    assert (tune_dir / name).is_dir()


def _assert_experiment_gone(tune_dir: Path, trial_id: str, ts: str) -> None:
    assert _owned_by(_survivors(tune_dir), trial_id, ts) == set()


A_TS = "2026-08-16_12-38-11"


def _reviewer_layout(tune_dir: Path, *, failed: int) -> list[tuple[str, str]]:
    """Real experiment A (resumed twice) then `failed` fresh launches that died."""
    _experiment(
        tune_dir, "dea5e", A_TS, healthy=True,
        resumes=("2026-08-17_19-02-16", "2026-08-20_16-13-10"),
    )
    out = []
    for i in range(failed):
        trial_id, ts = f"f{i:04x}", f"2026-11-1{i}_09-00-00"
        _experiment(tune_dir, trial_id, ts, healthy=False, partial=i == 1)
        out.append((trial_id, ts))
    return out


def test_reviewers_a_plus_b_keeps_the_real_experiment(tmp_path: Path) -> None:
    """The reproduced window: A with checkpoints, a newer failed B, keep 2."""
    reference = tmp_path / "main"
    reference.mkdir()
    _reviewer_layout(reference, failed=1)
    _main_cleanup(tune_dir=reference, keep_last=2, reserve_new=True)
    assert not (reference / _trial_name("dea5e", A_TS)).exists(), (
        "fixture no longer reproduces the defect on main's rule"
    )

    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    ((b_id, b_ts),) = _reviewer_layout(tune_dir, failed=1)
    a_trial = tune_dir / _trial_name("dea5e", A_TS)
    weights = (a_trial / "checkpoint_000250" / "trainer.pt").read_bytes()

    _cleanup_old_tune_experiments(tune_dir=tune_dir, keep_last=2, reserve_new=True)

    assert (a_trial / "checkpoint_000250" / "trainer.pt").read_bytes() == weights
    assert (a_trial / "checkpoint_000249" / "trainer.pt").is_file()
    assert (tune_dir / "experiment_state-2026-08-20_16-13-10.json").is_file()
    assert (tune_dir / "basic-variant-state-2026-08-20_16-13-10.json").is_file()
    assert (tune_dir / "pbt_policy_dea5e_00000.txt").is_file()
    # A's older resume states are surplus copies; its newest one is retained.
    assert not (tune_dir / f"experiment_state-{A_TS}.json").exists()
    assert not (tune_dir / "experiment_state-2026-08-17_19-02-16.json").exists()
    _assert_experiment_intact(tune_dir, b_id, b_ts)


def test_three_failed_launches_in_a_row_keep_the_real_experiment(tmp_path: Path) -> None:
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    failed = _reviewer_layout(tune_dir, failed=3)

    _cleanup_old_tune_experiments(tune_dir=tune_dir, keep_last=2, reserve_new=True)

    a_trial = tune_dir / _trial_name("dea5e", A_TS)
    assert (a_trial / "checkpoint_000250" / "trainer.pt").is_file()
    assert (tune_dir / "experiment_state-2026-08-20_16-13-10.json").is_file()
    for trial_id, ts in failed:
        _assert_experiment_intact(tune_dir, trial_id, ts)

    # And a fourth fresh attempt over that is a no-op for A as well.
    before = _survivors(tune_dir)
    _cleanup_old_tune_experiments(tune_dir=tune_dir, keep_last=2, reserve_new=True)
    assert _survivors(tune_dir) == before


MIXED = [
    ("a0001", "2026-09-01_00-00-00", True),
    ("a0002", "2026-09-02_00-00-00", True),
    ("b0001", "2026-09-03_00-00-00", False),
    ("a0003", "2026-09-04_00-00-00", True),
    ("b0002", "2026-09-05_00-00-00", False),
]


@pytest.mark.parametrize(("keep_last", "retained"), [
    (2, {"a0003", "b0002"}),
    (3, {"a0002", "b0001", "a0003", "b0002"}),
])
def test_mixed_healthy_and_failed(
    tmp_path: Path, keep_last: int, retained: set[str],
) -> None:
    """Retained = the newest keep_last-1 checkpointed experiments plus every
    failed one newer than them; older failed ones are pruned as garbage."""
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    for trial_id, ts, healthy in MIXED:
        _experiment(tune_dir, trial_id, ts, healthy=healthy)

    _cleanup_old_tune_experiments(tune_dir=tune_dir, keep_last=keep_last, reserve_new=True)

    for trial_id, ts, _healthy in MIXED:
        if trial_id in retained:
            _assert_experiment_intact(tune_dir, trial_id, ts)
        else:
            _assert_experiment_gone(tune_dir, trial_id, ts)


def test_symlinked_trial_dirs(tmp_path: Path) -> None:
    """A trial dir symlinked elsewhere counts by its target's checkpoints, and
    pruning an old symlinked trial never deletes the target's contents."""
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    external = tmp_path / "archive"
    external.mkdir()

    old_name = _trial_name("01d00", "2026-08-01_00-00-00")
    _healthy_trial(external, old_name)
    (tune_dir / old_name).symlink_to(external / old_name, target_is_directory=True)
    _write_state(tune_dir, "2026-08-01_00-00-00", trials=[old_name], status="TERMINATED")

    a_name = _trial_name("dea5e", A_TS)
    _healthy_trial(external, a_name)
    (tune_dir / a_name).symlink_to(external / a_name, target_is_directory=True)
    _write_state(tune_dir, A_TS, trials=[a_name], status="RUNNING")
    (tune_dir / "pbt_policy_dea5e_00000.txt").write_text("policy")

    _experiment(tune_dir, "f0000", "2026-11-10_09-00-00", healthy=False)

    _cleanup_old_tune_experiments(tune_dir=tune_dir, keep_last=2, reserve_new=True)

    assert (tune_dir / a_name).is_symlink()
    assert (tune_dir / a_name / "checkpoint_000250" / "trainer.pt").is_file()
    assert (tune_dir / f"experiment_state-{A_TS}.json").is_file()
    assert (tune_dir / "pbt_policy_dea5e_00000.txt").is_file()
    _assert_experiment_intact(tune_dir, "f0000", "2026-11-10_09-00-00")
    assert not (tune_dir / "experiment_state-2026-08-01_00-00-00.json").exists()
    assert (external / old_name / "checkpoint_000250" / "trainer.pt").is_file()


def _healthy_rotation(tune_dir: Path) -> None:
    """Four healthy experiments, two of them resumed (one state file each)."""
    _experiment(tune_dir, "c0001", "2026-07-01_00-00-00", healthy=True)
    _experiment(
        tune_dir, "c0002", "2026-07-05_00-00-00", healthy=True,
        resumes=("2026-07-06_00-00-00", "2026-07-07_00-00-00"),
    )
    _experiment(tune_dir, "c0003", "2026-07-10_00-00-00", healthy=True)
    _experiment(
        tune_dir, "c0004", "2026-07-12_00-00-00", healthy=True,
        resumes=("2026-07-13_00-00-00",),
    )
    (tune_dir / "train_trial_orphan_00000_0_lr=0.1_2026-06-01_00-00-00").mkdir()
    (tune_dir / "pbt_policy_gone0_00000.txt").write_text("orphan log")
    (tune_dir / "tuner.pkl").write_bytes(b"tuner")


@pytest.mark.parametrize("reserve_new", [False, True])
@pytest.mark.parametrize("keep_last", [-1, 0, 1, 2, 3, 4, 7, 10])
def test_ordinary_healthy_rotation_matches_main(
    tmp_path: Path, keep_last: int, reserve_new: bool,
) -> None:
    survivors: dict[str, set[str]] = {}
    rules: dict[str, Callable[..., None]] = {
        "main": _main_cleanup, "new": _cleanup_old_tune_experiments,
    }
    for label, rule in rules.items():
        tune_dir = tmp_path / label
        tune_dir.mkdir()
        _healthy_rotation(tune_dir)
        rule(tune_dir=tune_dir, keep_last=keep_last, reserve_new=reserve_new)
        survivors[label] = _survivors(tune_dir) | {
            str(d.relative_to(tune_dir)) + "/" for d in tune_dir.glob("train_trial_*")
        }
    assert survivors["new"] == survivors["main"]


def test_only_checkpointless_experiments_prunes_nothing(tmp_path: Path) -> None:
    """With nothing durable anywhere, no experiment is provably garbage."""
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    for i in range(4):
        _experiment(tune_dir, f"f{i:04x}", f"2026-11-0{i + 1}_00-00-00", healthy=False, partial=True)
    before = _survivors(tune_dir)

    _cleanup_old_tune_experiments(tune_dir=tune_dir, keep_last=2, reserve_new=True)

    assert _survivors(tune_dir) == before


def test_checkpoint_definition_matches_salvage(tmp_path: Path) -> None:
    """Pins the on-disk meaning of "holds a checkpoint" to salvage's."""
    from chess_anti_engine.tune.salvage import _newest_disk_checkpoint

    healthy = _healthy_trial(tmp_path, _trial_name("a0001", "2026-09-01_00-00-00"))
    partial = _failed_trial(tmp_path, _trial_name("b0001", "2026-09-02_00-00-00"), partial=True)
    for d in (healthy, partial, tmp_path / "missing"):
        expected = d.is_dir() and _newest_disk_checkpoint(d) is not None
        assert harness._trial_dir_holds_checkpoint(d) is expected
    assert harness._trial_dir_holds_checkpoint(healthy)
    assert not harness._trial_dir_holds_checkpoint(partial)

