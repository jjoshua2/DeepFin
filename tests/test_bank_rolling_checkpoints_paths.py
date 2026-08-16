"""`bank_rolling_checkpoints.sh` must survive a checkout path with a space.

The path scrub replaced a hardcoded absolute root with one derived from the
script's own location. That is correct -- and it ARMS a latent defect: the tune
glob was held in a variable and expanded UNQUOTED (`for trial in $TUNE_GLOB`),
which word-splits on IFS and re-globs the whole path. While the root was a fixed
literal with no space in it the split was invisible; derived from the checkout,
it is whatever directory the repo happens to sit in.

The test runs the real script against a synthetic checkout under a directory
named with a space, and asserts the checkpoint is actually banked.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "bank_rolling_checkpoints.sh"


def _make_checkout(root: Path) -> Path:
    """A minimal tree with one bankable checkpoint (index 5, a multiple of 5)."""
    trial = root / "runs" / "pbt2_small" / "tune" / "train_trial_abc123"
    ckpt = trial / "checkpoint_000005"
    ckpt.mkdir(parents=True)
    (ckpt / "trainer.pt").write_text("weights", encoding="utf-8")
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SCRIPT, scripts / SCRIPT.name)
    return scripts / SCRIPT.name


def _run_one_pass(script: Path) -> subprocess.CompletedProcess[str]:
    """One loop iteration. The script loops forever, so `timeout` bounds it.

    `timeout` signals the process it spawned, by pid -- never a name pattern.
    """
    return subprocess.run(
        ["timeout", "20", "bash", str(script)],
        capture_output=True, text=True, check=False,
        env={"PATH": "/usr/bin:/bin", "SLEEP": "600", "EVERY": "5", "KEEP": "24"},
    )


@pytest.mark.parametrize(
    "dirname",
    [
        pytest.param("plain_checkout", id="plain"),
        pytest.param("check out with spaces", id="space"),
        pytest.param("checkout[weird]", id="glob-metachar"),
    ],
)
def test_the_checkpoint_is_banked_whatever_the_checkout_is_called(
    tmp_path: Path, dirname: str,
) -> None:
    root = tmp_path / dirname
    root.mkdir()
    script = _make_checkout(root)

    result = _run_one_pass(script)

    banked = root / "data" / "salvage" / "rolling" / "checkpoint_000005"
    assert banked.is_dir(), (
        f"nothing banked from {dirname!r}. stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert (banked / "trainer.pt").read_text(encoding="utf-8") == "weights"
    assert "banked checkpoint_000005" in result.stdout, result.stdout


def test_no_half_written_temp_dir_is_left_behind(tmp_path: Path) -> None:
    root = tmp_path / "check out with spaces"
    root.mkdir()
    script = _make_checkout(root)

    _run_one_pass(script)

    rolling = root / "data" / "salvage" / "rolling"
    assert [p.name for p in rolling.iterdir()] == ["checkpoint_000005"]


def test_the_glob_is_expanded_at_the_point_of_use() -> None:
    """A regression pin on the SHAPE, not just the outcome.

    Reintroducing `TUNE_GLOB="...train_trial_*"` + `for trial in $TUNE_GLOB`
    passes the plain-directory case above, so the outcome test alone would not
    say why it broke. An unquoted bare `$VAR` in a `for ... in` is the defect.
    """
    text = SCRIPT.read_text(encoding="utf-8")
    offenders = [
        line.strip() for line in text.splitlines()
        if line.strip().startswith("for ") and " in $" in line
    ]
    assert not offenders, (
        f"unquoted variable expansion in a for-loop: {offenders}. Quote the "
        'prefix and let only the trailing `*` glob: `for x in "$DIR"/pat_*`.'
    )
