"""Fresh-start cleanup must keep the trial dirs the retained experiments use.

`_cleanup_old_tune_experiments` keeps the newest N `experiment_state-*.json`
files and deletes trial directories belonging to the rest. It used to pair the
two by filename timestamp: `glob(f"train_trial_*{ts}")`.

That pairing is wrong. A trial directory is named with the timestamp of the
experiment that CREATED it, but every resume writes a new state file under a
NEW timestamp. After a few restarts the kept state files share no timestamp
with any trial directory on disk, so the glob protected nothing -- `keep_last`
realized as zero however the YAML was set.

Live on 2026-07-25: 66 state files for 2 experiments, and the newest kept
file's `relative_logdir` named the very directory the glob would have deleted
-- the live trial, holding every checkpoint and all TensorBoard history since
07-11. It never fired only because the cleanup is guarded by `if not resume`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_anti_engine.tune.harness import (
    _cleanup_old_tune_experiments,
    _trial_dirs_referenced_by,
)


def _write_state(tune_dir: Path, ts: str, *, trial_dirs: list[str]) -> Path:
    """An experiment-state file shaped like Ray's, with the fields we read."""
    payload = {
        "trial_data": [
            [
                json.dumps(
                    {
                        "stub": False,
                        "trainable_name": "train_trial",
                        "trial_id": name.split("_")[2],
                        "relative_logdir": name,
                        "storage": {"_type": "CLOUDPICKLE_FALLBACK", "value": "8005"},
                    }
                ),
                json.dumps({"start_time": 0.0}),
            ]
            for name in trial_dirs
        ],
        "runner_data": {},
        "stats": {},
    }
    path = tune_dir / f"experiment_state-{ts}.json"
    path.write_text(json.dumps(payload))
    (tune_dir / f"basic-variant-state-{ts}.json").write_text("{}")
    return path


def _make_trial_dir(tune_dir: Path, name: str) -> Path:
    d = tune_dir / name
    (d / "checkpoint_000250").mkdir(parents=True)
    (d / "checkpoint_000250" / "state.pt").write_bytes(b"irreplaceable")
    return d


LIVE_TRIAL = "train_trial_4c17c_00000_0_lr=0.0003_2026-07-11_13-16-47"
OLD_TRIAL = "train_trial_9827e_00000_0_lr=0.0003_2026-07-11_12-50-16"


def _live_layout(tune_dir: Path) -> tuple[Path, Path]:
    """The 2026-07-25 shape: two experiments, many resumes each."""
    _write_state(tune_dir, "2026-07-11_12-50-16", trial_dirs=[OLD_TRIAL])
    _write_state(tune_dir, "2026-07-11_13-16-47", trial_dirs=[LIVE_TRIAL])
    for ts in ("2026-07-12_03-38-20", "2026-07-19_13-38-40", "2026-07-24_22-52-27"):
        _write_state(tune_dir, ts, trial_dirs=[LIVE_TRIAL])
    _write_state(tune_dir, "2026-07-25_12-12-34", trial_dirs=[LIVE_TRIAL])
    return _make_trial_dir(tune_dir, OLD_TRIAL), _make_trial_dir(tune_dir, LIVE_TRIAL)


def test_the_kept_experiments_trial_dir_survives_a_timestamp_it_does_not_share(
    tmp_path: Path,
) -> None:
    """The regression. Kept states are dated 07-24/07-25; the dir they point at
    is dated 07-11, and it must not be deleted for that."""
    old_dir, live_dir = _live_layout(tmp_path)

    _cleanup_old_tune_experiments(tune_dir=tmp_path, keep_last=2)

    assert (live_dir / "checkpoint_000250" / "state.pt").exists(), (
        "deleted the trial dir that the newest retained state file names"
    )
    assert not old_dir.exists(), "the genuinely unreferenced experiment should go"


def test_old_state_files_are_pruned_and_the_kept_ones_are_not(tmp_path: Path) -> None:
    _live_layout(tmp_path)

    _cleanup_old_tune_experiments(tune_dir=tmp_path, keep_last=2)

    remaining = sorted(p.name for p in tmp_path.glob("experiment_state-*.json"))
    assert remaining == [
        "experiment_state-2026-07-24_22-52-27.json",
        "experiment_state-2026-07-25_12-12-34.json",
    ]
    assert sorted(p.name for p in tmp_path.glob("basic-variant-state-*.json")) == [
        "basic-variant-state-2026-07-24_22-52-27.json",
        "basic-variant-state-2026-07-25_12-12-34.json",
    ]


def test_an_orphan_trial_dir_with_no_referencing_state_is_deleted(
    tmp_path: Path,
) -> None:
    """Cleanup still has to reclaim disk, not just protect things."""
    _live_layout(tmp_path)
    orphan = _make_trial_dir(tmp_path, "train_trial_dead0_00000_0_lr=0.1_2026-06-01_00-00-00")

    _cleanup_old_tune_experiments(tune_dir=tmp_path, keep_last=2)

    assert not orphan.exists()


def test_an_unreadable_kept_state_file_protects_every_trial_dir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Fail safe. Not knowing what a retained experiment uses must mean "keep
    everything", never "keep nothing" -- the downside is disk, the other way is
    months of training."""
    old_dir, live_dir = _live_layout(tmp_path)
    (tmp_path / "experiment_state-2026-07-25_12-12-34.json").write_text("{tru ncated")

    _cleanup_old_tune_experiments(tune_dir=tmp_path, keep_last=2)

    assert live_dir.exists()
    assert old_dir.exists()
    assert "Keeping every trial dir" in capsys.readouterr().out


def test_a_corrupt_kept_state_file_still_lets_the_old_state_files_go(
    tmp_path: Path,
) -> None:
    """The fail-safe covers trial dirs only; state files are small and stale."""
    _live_layout(tmp_path)
    (tmp_path / "experiment_state-2026-07-25_12-12-34.json").write_text("{tru ncated")

    _cleanup_old_tune_experiments(tune_dir=tmp_path, keep_last=2)

    assert len(list(tmp_path.glob("experiment_state-*.json"))) == 2


def test_nothing_happens_when_there_is_nothing_to_prune(tmp_path: Path) -> None:
    _write_state(tmp_path, "2026-07-25_12-12-34", trial_dirs=[LIVE_TRIAL])
    live_dir = _make_trial_dir(tmp_path, LIVE_TRIAL)

    _cleanup_old_tune_experiments(tune_dir=tmp_path, keep_last=2)

    assert live_dir.exists()
    assert len(list(tmp_path.glob("experiment_state-*.json"))) == 1


def test_keep_last_zero_disables_cleanup_entirely(tmp_path: Path) -> None:
    """0 means "no pruning", not "keep nothing"."""
    old_dir, live_dir = _live_layout(tmp_path)

    _cleanup_old_tune_experiments(tune_dir=tmp_path, keep_last=0)

    assert old_dir.exists()
    assert live_dir.exists()
    assert len(list(tmp_path.glob("experiment_state-*.json"))) == 6


def test_a_missing_tune_dir_is_not_an_error(tmp_path: Path) -> None:
    _cleanup_old_tune_experiments(tune_dir=tmp_path / "nope", keep_last=2)


def test_a_state_file_naming_several_trials_protects_all_of_them(
    tmp_path: Path,
) -> None:
    """PB2 runs many trials per experiment; one reference must not shadow the rest."""
    names = [f"train_trial_p{i}_00000_0_lr=0.1_2026-07-01_00-00-00" for i in range(3)]
    _write_state(tmp_path, "2026-07-01_00-00-00", trial_dirs=names)
    _write_state(tmp_path, "2026-07-02_00-00-00", trial_dirs=names)
    _write_state(tmp_path, "2026-07-03_00-00-00", trial_dirs=names)
    dirs = [_make_trial_dir(tmp_path, n) for n in names]

    _cleanup_old_tune_experiments(tune_dir=tmp_path, keep_last=2)

    assert all(d.exists() for d in dirs)


def test_a_live_trials_pb2_policy_log_is_not_deleted(tmp_path: Path) -> None:
    """`split("_", 2)[1]` on `train_trial_4c17c_00000_...` is the literal
    "trial", so the old live-set could never match and every policy log was
    deleted on each fresh start -- the exact inverse of the intent."""
    _live_layout(tmp_path)
    live_log = tmp_path / "pbt_policy_4c17c_00000.txt"
    live_log.write_text("exploit history worth keeping")
    dead_log = tmp_path / "pbt_policy_9827e_00000.txt"
    dead_log.write_text("belongs to the deleted trial")

    _cleanup_old_tune_experiments(tune_dir=tmp_path, keep_last=2)

    assert live_log.exists(), "deleted the surviving trial's own policy log"
    assert not dead_log.exists()


def test_a_policy_log_is_matched_on_the_whole_trial_id(tmp_path: Path) -> None:
    """`4c17c_00000` and `4c17c_00001` share a prefix but are different trials."""
    _live_layout(tmp_path)
    sibling = tmp_path / "pbt_policy_4c17c_00001.txt"
    sibling.write_text("a different trial that happens to share a prefix")

    _cleanup_old_tune_experiments(tune_dir=tmp_path, keep_last=2)

    assert not sibling.exists(), "prefix-matching would have spared this"


def test_relative_logdir_is_what_ray_actually_serialises() -> None:
    """Pins the field the fix reads.

    The fakes above would keep passing against a Ray that named the field
    something else, at which point the parse returns None, the fail-safe fires,
    and cleanup would quietly stop reclaiming disk forever.
    """
    import inspect

    from ray.tune.experiment.trial import Trial

    assert "relative_logdir" in inspect.getsource(Trial)
    src = inspect.getsource(Trial.get_json_state)
    assert "self.__getstate__()" in src, (
        "trial state is no longer a plain __getstate__ dump; re-check the field"
    )


def test_unparseable_state_returns_none_rather_than_an_empty_set(
    tmp_path: Path,
) -> None:
    """The distinction the fail-safe rests on: None ("unknown") is not set()
    ("references nothing")."""
    bad = tmp_path / "experiment_state-2026-07-25_12-12-34.json"
    bad.write_text("not json at all")
    assert _trial_dirs_referenced_by(bad) is None

    empty = tmp_path / "experiment_state-2026-07-25_12-12-35.json"
    empty.write_text(json.dumps({"trial_data": []}))
    assert _trial_dirs_referenced_by(empty) == set()
