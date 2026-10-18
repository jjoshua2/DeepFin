"""Synthetic shard directories only; no Zarr, training, or live pool access."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

from scripts import feed_bootstrap_shards as feed


def _shard(path: Path, content: str) -> None:
    path.mkdir(parents=True)
    (path / "fixture").write_text(content, encoding="utf-8")


def _feed(monkeypatch: pytest.MonkeyPatch, live: Path, boot: Path, holdout: Path, cadence: int) -> int:
    monkeypatch.setattr(sys, "argv", [
        "feed_bootstrap_shards.py", "--live-dir", str(live), "--boot-dir", str(boot),
        "--holdout-dir", str(holdout), "--holdout-every-n", str(cadence),
        "--settle-seconds", "0", "--copy",
    ])
    return feed.main()


@pytest.mark.parametrize("cadence", [0, 40], ids=["disabled", "enabled"])
@pytest.mark.parametrize("trial,membership", [
    pytest.param(feed._LEGACY_PRETAG_TRIAL, "bare", id="legacy-bare"),
    pytest.param(feed._LEGACY_PRETAG_TRIAL, "tagged", id="legacy-tagged"),
    pytest.param("train_trial_other", "tagged", id="other-tagged"),
])
def test_trained_shard_never_enters_holdout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trial: str, membership: str, cadence: int,
) -> None:
    live = tmp_path / trial / "replay_shards"
    src = live / "shard_000040.zarr"
    _shard(src, "source")
    boot, holdout = tmp_path / "boot", tmp_path / "holdout"
    name = src.name if membership == "bare" else f"{src.stem}.{feed._trial_tag(live)}.zarr"
    _shard(boot / name, "already trained")

    for _ in range(2):
        assert _feed(monkeypatch, live, boot, holdout, cadence) == 0
        assert {path.name for path in boot.iterdir()} == {name}
        assert (boot / name / "fixture").read_text(encoding="utf-8") == "already trained"
        assert list(holdout.iterdir()) == []


@pytest.mark.parametrize("cadence", [0, 40], ids=["disabled", "enabled"])
@pytest.mark.parametrize("existing", ["bare", "foreign-tagged"])
def test_other_trial_same_basename_is_still_fed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str, cadence: int,
) -> None:
    live = tmp_path / "train_trial_other" / "replay_shards"
    src = live / "shard_000040.zarr"
    _shard(src, "new trial")
    boot, holdout = tmp_path / "boot", tmp_path / "holdout"
    legacy_live = tmp_path / feed._LEGACY_PRETAG_TRIAL / "replay_shards"
    old_name = src.name if existing == "bare" else f"{src.stem}.{feed._trial_tag(legacy_live)}.zarr"
    _shard(boot / old_name, "other trial already trained")
    new_name = f"{src.stem}.{feed._trial_tag(live)}.zarr"
    destination = holdout if cadence else boot

    for _ in range(2):
        assert _feed(monkeypatch, live, boot, holdout, cadence) == 0
        assert (destination / new_name / "fixture").read_text(encoding="utf-8") == "new trial"
        assert (boot / old_name / "fixture").read_text(encoding="utf-8") == "other trial already trained"
        assert {path.name for path in boot.iterdir()} == ({old_name} if cadence else {old_name, new_name})
        assert {path.name for path in holdout.iterdir()} == ({new_name} if cadence else set())
