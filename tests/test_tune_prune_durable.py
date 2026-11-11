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

import contextlib
import io
import json
import os
import random
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


def _pr1217_cleanup(*, tune_dir: Path, keep_last: int, reserve_new: bool = False) -> None:
    """Verbatim body of the #1217 rule (checkpoint-less state files use no
    slot), before unreferenced checkpointed trial dirs were protected."""
    keep_last = int(keep_last)
    if keep_last <= 0 or not tune_dir.exists():
        return
    if reserve_new:
        keep_last -= 1

    state_files = sorted(tune_dir.glob("experiment_state-*.json"))
    if len(state_files) <= keep_last:
        return

    keep: list[Path] = []
    protected: set[str] = set()
    protection_is_complete = True
    slots_used = 0
    for p in reversed(state_files):
        if slots_used >= keep_last:
            break
        keep.append(p)
        names = _trial_dirs_referenced_by(p)
        if names is None:
            protection_is_complete = False
            continue
        protected |= names
        if any(harness._trial_dir_holds_checkpoint(tune_dir / n) for n in names):
            slots_used += 1

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



def _write_raw_state(tune_dir: Path, ts: str, trial_data: list) -> None:
    (tune_dir / f"experiment_state-{ts}.json").write_text(json.dumps({
        "trial_data": trial_data, "runner_data": {}, "stats": {},
    }))
    (tune_dir / f"basic-variant-state-{ts}.json").write_text("{}")


@pytest.mark.parametrize("trial_data", [
    # Ray leaves relative_logdir null until the trial is scheduled, so a launch
    # that dies before scheduling writes exactly this.
    [[json.dumps({"trainable_name": "train_trial", "relative_logdir": None}), "{}"]],
    [],
], ids=["relative_logdir_null", "empty_trial_data"])
def test_a_state_file_referencing_no_trial_dir_uses_no_slot(
    tmp_path: Path, trial_data: list,
) -> None:
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    _reviewer_layout(tune_dir, failed=0)
    _write_raw_state(tune_dir, "2026-11-11_09-00-00", trial_data)

    _cleanup_old_tune_experiments(tune_dir=tune_dir, keep_last=2, reserve_new=True)

    a_trial = tune_dir / _trial_name("dea5e", A_TS)
    assert (a_trial / "checkpoint_000250" / "trainer.pt").is_file()
    assert (tune_dir / "experiment_state-2026-08-20_16-13-10.json").is_file()
    assert (tune_dir / "pbt_policy_dea5e_00000.txt").is_file()
    assert (tune_dir / "experiment_state-2026-11-11_09-00-00.json").is_file()


@pytest.mark.parametrize("form", ["absolute", "dotdot", "nested"])
def test_a_state_file_naming_a_dir_outside_the_tune_dir_uses_no_slot(
    tmp_path: Path, form: str,
) -> None:
    """Only direct children of the tune dir are trial dirs. A name that points
    elsewhere must not let a checkpoint there take the real run's slot."""
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    _reviewer_layout(tune_dir, failed=0)
    outside = tmp_path / "outside" / "train_trial_zzzzz"
    (outside / "checkpoint_000001").mkdir(parents=True)
    (outside / "checkpoint_000001" / "trainer.pt").write_bytes(b"x")
    nested = tune_dir / "sub" / "train_trial_yyyyy"
    (nested / "checkpoint_000001").mkdir(parents=True)
    (nested / "checkpoint_000001" / "trainer.pt").write_bytes(b"x")
    name = {"absolute": str(outside), "dotdot": "../outside/train_trial_zzzzz",
            "nested": "sub/train_trial_yyyyy"}[form]
    _write_raw_state(tune_dir, "2026-11-11_09-00-00", [
        [json.dumps({"trainable_name": "train_trial", "relative_logdir": name}), "{}"],
    ])

    _cleanup_old_tune_experiments(tune_dir=tune_dir, keep_last=2, reserve_new=True)

    a_trial = tune_dir / _trial_name("dea5e", A_TS)
    assert (a_trial / "checkpoint_000250" / "trainer.pt").is_file()
    assert (tune_dir / "experiment_state-2026-08-20_16-13-10.json").is_file()
    assert (outside / "checkpoint_000001" / "trainer.pt").is_file()


def test_a_population_with_one_checkpointed_trial_uses_a_slot(tmp_path: Path) -> None:
    """PBT: one trial errored, its sibling holds a checkpoint -> the state file
    is checkpoint-bearing, so it fills the slot and the older run goes."""
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    _experiment(tune_dir, "c0001", "2026-07-01_00-00-00", healthy=True)
    ts = "2026-07-05_00-00-00"
    errored = _failed_trial(tune_dir, _trial_name("p0001", ts))
    healthy = _healthy_trial(tune_dir, _trial_name("p0002", ts))
    _write_state(tune_dir, ts, trials=[errored.name, healthy.name], status="RUNNING")

    _cleanup_old_tune_experiments(tune_dir=tune_dir, keep_last=2, reserve_new=True)

    assert errored.is_dir()
    assert (healthy / "checkpoint_000250" / "trainer.pt").is_file()
    _assert_experiment_gone(tune_dir, "c0001", "2026-07-01_00-00-00")


# -- property test: random experiment sequences, new rule vs main's ---------

_KINDS = ("healthy", "healthy", "failed", "staging_only", "no_trainer",
          "corrupt", "nullref", "population")


def _random_layout(root: Path, outside: Path, rng: random.Random) -> dict:
    """Random launches/resumes in the production layout. Returns, per state
    file name, the trial dirs it references (None if unreadable)."""
    refs_by_state: dict[str, list[str] | None] = {}
    t = counter = 0
    prev: list[str] = []
    kinds: list[str] = []
    for _ in range(rng.randint(1, 6)):
        kind = rng.choice(_KINDS)
        kinds.append(kind)
        pop = rng.randint(2, 3) if kind == "population" or rng.random() < 0.2 else 1
        trials = []
        for _ in range(pop):
            counter += 1
            name = _trial_name(f"{counter:05x}", f"T{t:04d}")
            trials.append(name)
            target = root / name
            if rng.random() < 0.1:
                (outside / name).mkdir(parents=True)
                target.symlink_to(outside / name, target_is_directory=True)
                target = outside / name
            else:
                target.mkdir()
            sub = "failed" if kind == "population" and rng.random() < 0.3 else kind
            if sub in ("healthy", "population", "corrupt", "nullref"):
                (target / "checkpoint_000003").mkdir()
                (target / "checkpoint_000003" / "trainer.pt").write_bytes(b"w")
            elif sub == "failed":
                (target / "error.txt").write_text("boom")
            elif sub == "staging_only":
                (target / "ckpt").mkdir()
                (target / "ckpt" / "trainer.pt").write_bytes(b"w")
            elif sub == "no_trainer":
                (target / "checkpoint_000001").mkdir()
                (target / "checkpoint_000001" / "pid_state.json").write_text("{}")
            (root / f"pbt_policy_{counter:05x}_00000.txt").write_text("x")
        n_states = rng.randint(1, 3)
        for s in range(n_states):
            t += rng.randint(1, 3)
            ts = f"2026-08-{10 + t // 1440:02d}_{(t // 60) % 24:02d}-{t % 60:02d}-00"
            refs = trials + (prev if prev and rng.random() < 0.15 else [])
            if kind == "corrupt" and s == n_states - 1:
                (root / f"experiment_state-{ts}.json").write_text("{tru ncated")
                (root / f"basic-variant-state-{ts}.json").write_text("{}")
                refs_by_state[f"experiment_state-{ts}.json"] = None
                continue
            _write_raw_state(root, ts, [
                [json.dumps({"relative_logdir": None if kind == "nullref" else r}), "{}"]
                for r in refs
            ])
            refs_by_state[f"experiment_state-{ts}.json"] = [] if kind == "nullref" else refs
        prev = trials
    if rng.random() < 0.2:
        counter += 1
        orphan = root / f"train_trial_{counter:05x}_00000_0_orphan"
        (orphan / "checkpoint_000009").mkdir(parents=True)
        (orphan / "checkpoint_000009" / "trainer.pt").write_bytes(b"w")
    return {"refs": refs_by_state, "healthy_only": set(kinds) == {"healthy"}}


def _tree(root: Path) -> set[str]:
    out = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for n in dirnames + filenames:
            out.add(os.path.relpath(os.path.join(dirpath, n), root))
    return out


def _has_checkpoint(d: Path) -> bool:
    return any(d.glob("checkpoint_*/trainer.pt"))


@pytest.mark.parametrize("block", range(4))
def test_property_random_sequences_against_main(tmp_path: Path, block: int) -> None:
    """4 x 75 seeded cases, the current rule against main's and #1217's:
    (a) the newest prior checkpoint-bearing state file's checkpointed trial
    dirs always survive; (b) nothing main or #1217 keeps is deleted; (c) on
    healthy-only layouts survivors are identical to both, apart from an
    unreferenced checkpointed (orphan) dir; (d) a trial dir holding a
    checkpoint that no readable state file names always survives."""
    rules: dict[str, Callable[..., None]] = {
        "main": _main_cleanup, "pr1217": _pr1217_cleanup,
        "new": _cleanup_old_tune_experiments,
    }
    for seed in range(block * 75, (block + 1) * 75):
        rng = random.Random(seed)
        keep = rng.choice([-1, 0, 1, 2, 2, 3, 4])
        reserve = rng.random() < 0.7
        link_root = rng.random() < 0.15
        trees: dict[str, set[str]] = {}
        healthy_only = False
        for label, rule in rules.items():
            base = tmp_path / f"{seed}-{label}"
            outside, root = base / "outside", base / "tune"
            outside.mkdir(parents=True)
            root.mkdir()
            meta = _random_layout(root, outside, random.Random(seed * 7919 + 1))
            healthy_only = meta["healthy_only"]
            newest = None
            named: set[str] = set()
            for name in sorted(meta["refs"]):
                refs = meta["refs"][name]
                named |= set(refs or [])
                ck = [r for r in refs or [] if _has_checkpoint(root / r)]
                if ck:
                    newest = ck
            unnamed_ckpt = [
                d.name for d in root.glob("train_trial_*")
                if d.name not in named and _has_checkpoint(d)
            ]
            call_root = root
            if link_root:
                call_root = base / "tune_link"
                call_root.symlink_to(root, target_is_directory=True)
            with contextlib.redirect_stdout(io.StringIO()):
                rule(tune_dir=call_root, keep_last=keep, reserve_new=reserve)
            trees[label] = _tree(root) | {"OUT:" + x for x in _tree(outside)}
            if label == "new":
                if keep - int(reserve) >= 1 and newest is not None:
                    assert all(_has_checkpoint(root / r) for r in newest), (seed, newest)
                assert all(_has_checkpoint(root / n) for n in unnamed_ckpt), (seed, unnamed_ckpt)
        assert trees["main"] <= trees["pr1217"] <= trees["new"], seed
        if healthy_only:
            def strip(tree: set[str]) -> set[str]:
                return {x for x in tree if "_orphan" not in x}
            assert strip(trees["main"]) == strip(trees["pr1217"]) == strip(trees["new"]), seed


# -- unreferenced trial dirs holding checkpoints ------------------------------

LIVE_TS = ("2026-08-16_12-38-11", "2026-08-17_19-02-16", "2026-08-20_16-13-10")


def _live_behind_old(tune_dir: Path) -> tuple[Path, Path]:
    """An old finished experiment, then the live one resumed twice."""
    _experiment(tune_dir, "9827e", "2026-07-11_12-50-16", healthy=True,
                resumes=("2026-07-12_00-00-00",))
    _experiment(tune_dir, "dea5e", LIVE_TS[0], healthy=True, resumes=LIVE_TS[1:])
    return (tune_dir / _trial_name("9827e", "2026-07-11_12-50-16"),
            tune_dir / _trial_name("dea5e", LIVE_TS[0]))


@pytest.mark.parametrize("suffix", [".corrupt", ".bak_pre_lr3e5", ".failed_missing_ckpt103"])
def test_live_state_files_renamed_keeps_the_live_trial(
    tmp_path: Path, suffix: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """Reviewer case B: operators rename the live run's state files. The live
    trial dir is then referenced by no experiment_state-*.json at all."""
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    old, live = _live_behind_old(tune_dir)
    for ts in LIVE_TS:
        f = tune_dir / f"experiment_state-{ts}.json"
        f.rename(f.with_name(f.name + suffix))

    _cleanup_old_tune_experiments(tune_dir=tune_dir, keep_last=2, reserve_new=True)

    assert (live / "checkpoint_000250" / "trainer.pt").is_file()
    assert (tune_dir / "pbt_policy_dea5e_00000.txt").is_file()
    for ts in LIVE_TS:
        assert (tune_dir / f"experiment_state-{ts}.json{suffix}").is_file()
    assert (old / "checkpoint_000250" / "trainer.pt").is_file()
    assert f"WARNING: keeping unreferenced trial dir {live}" in capsys.readouterr().out


def test_state_files_without_relative_logdir_keep_every_checkpointed_trial(
    tmp_path: Path,
) -> None:
    """Reviewer case C: the files parse but name no trial (e.g. a Ray upgrade
    renames the field). Nothing accounts for any deletion; checkpointed dirs
    stay, a checkpoint-less orphan still goes."""
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    old, live = _live_behind_old(tune_dir)
    for f in tune_dir.glob("experiment_state-*.json"):
        payload = json.loads(f.read_text())
        for entry in payload["trial_data"]:
            trial = json.loads(entry[0])
            trial["log_dir_name"] = trial.pop("relative_logdir")
            entry[0] = json.dumps(trial)
        f.write_text(json.dumps(payload))
    garbage = _failed_trial(tune_dir, _trial_name("f0000", "2026-08-01_00-00-00"))

    _cleanup_old_tune_experiments(tune_dir=tune_dir, keep_last=2, reserve_new=True)

    assert (live / "checkpoint_000250" / "trainer.pt").is_file()
    assert (old / "checkpoint_000250" / "trainer.pt").is_file()
    assert not garbage.exists()


def test_rotation_still_deletes_a_checkpointed_trial_named_by_a_pruned_state(
    tmp_path: Path,
) -> None:
    """The exception is narrow: an old experiment whose readable state file is
    pruned this pass is ordinary rotation and still goes, checkpoints and all."""
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    old, live = _live_behind_old(tune_dir)

    _cleanup_old_tune_experiments(tune_dir=tune_dir, keep_last=2, reserve_new=True)

    assert not old.exists()
    assert not (tune_dir / "pbt_policy_9827e_00000.txt").exists()
    assert (live / "checkpoint_000250" / "trainer.pt").is_file()


def test_symlinked_unreferenced_trial_dirs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Symlinked orphan with a checkpoint in its target: kept and reported (the
    check follows the link). Symlinked orphan without one: its target's
    contents are never deleted."""
    tune_dir = tmp_path / "tune"
    tune_dir.mkdir()
    external = tmp_path / "archive"
    _live_behind_old(tune_dir)
    ck_name = _trial_name("aaaa1", "2026-06-01_00-00-00")
    _healthy_trial(external, ck_name)
    (tune_dir / ck_name).symlink_to(external / ck_name, target_is_directory=True)
    bare_name = _trial_name("aaaa2", "2026-06-02_00-00-00")
    _failed_trial(external, bare_name)
    (tune_dir / bare_name).symlink_to(external / bare_name, target_is_directory=True)

    _cleanup_old_tune_experiments(tune_dir=tune_dir, keep_last=2, reserve_new=True)

    assert (tune_dir / ck_name).is_symlink()
    assert (tune_dir / ck_name / "checkpoint_000250" / "trainer.pt").is_file()
    assert f"keeping unreferenced trial dir {tune_dir / ck_name}" in capsys.readouterr().out
    assert (external / bare_name / "error.txt").is_file()
