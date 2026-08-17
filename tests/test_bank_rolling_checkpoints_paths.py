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
    """One loop iteration, ENDED by the script's own `ONCE` seam.

    ⚑ Not by waiting for `timeout` to kill it. The banking work finishes in the
    first milliseconds and the script then sleeps; without the seam every case
    paid the full timeout, ~80s per serial suite run for four cases that were
    already done. `timeout` stays as a backstop only, and a return code of 124
    (killed) is itself a failure — see
    `test_the_once_seam_actually_ends_the_loop`.

    `timeout` signals the process it spawned, by pid -- never a name pattern.
    """
    return subprocess.run(
        ["timeout", "60", "bash", str(script)],
        capture_output=True, text=True, check=False,
        env={"PATH": "/usr/bin:/bin", "SLEEP": "600", "EVERY": "5", "KEEP": "24",
             "ONCE": "1"},
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


def test_the_once_seam_actually_ends_the_loop(tmp_path: Path) -> None:
    """⚑ The seam must TERMINATE the script, not merely be accepted by it.

    A knob that is read and then ignored is this codebase's signature defect;
    here it would show up only as a slow suite, which nobody reads as a bug. A
    124 means `timeout` did the killing and `ONCE` did nothing.
    """
    root = tmp_path / "plain_checkout"
    root.mkdir()
    script = _make_checkout(root)

    result = _run_one_pass(script)

    # ⚑ EXIT CODE, NOT WALL CLOCK. An `elapsed < N` assertion here was flaky on
    # a loaded box (this repo runs its tests beside live training), and a flaky
    # guard is a guard people learn to re-run rather than read. `timeout`
    # returns 124 when it had to do the killing, so a 0 IS the proof the script
    # ended itself — deterministic at any load, with the negative control below
    # supplying the contrast.
    assert result.returncode == 0, (
        f"exit {result.returncode} (124 = killed by timeout, so ONCE was read "
        f"and then ignored). stderr={result.stderr!r}"
    )


def test_without_the_seam_the_script_still_loops(tmp_path: Path) -> None:
    """The negative control: `ONCE` unset must NOT exit, or it is not a seam.

    Without this, deleting the `while :;` loop entirely would pass every other
    test in this file — the daemon would bank once and quit, and the rolling
    bank would silently stop rolling.
    """
    root = tmp_path / "plain_checkout"
    root.mkdir()
    script = _make_checkout(root)

    result = subprocess.run(
        ["timeout", "3", "bash", str(script)],
        capture_output=True, text=True, check=False,
        env={"PATH": "/usr/bin:/bin", "SLEEP": "600", "EVERY": "5", "KEEP": "24"},
    )
    assert result.returncode == 124, (
        f"the script exited on its own (code {result.returncode}) with ONCE "
        "unset — it is a daemon and must keep looping"
    )


def test_the_wrong_tree_is_a_LOUD_failure_not_a_silent_no_op(tmp_path: Path) -> None:
    """⚑⚑ #441 review N4: the banking daemon silently banked nothing.

    Measured before the guard, from a `git worktree`::

        $ ONCE=1 bash scripts/bank_rolling_checkpoints.sh
        exit=0          # no output, nothing banked, empty data/salvage/rolling

    Deriving the root from the checkout is right; it is also what turns "the
    trial is not here" from impossible into ordinary, because CLAUDE.md mandates
    doing branch work in a worktree. This is the salvage-banking daemon the
    "snapshot before big changes" revert-point rule depends on, and with `ONCE`
    unset it would have looped on the empty tree forever.

    The assertion is on the EXIT CODE and stderr, not on stdout: a silent exit 0
    is exactly the failure, so a test that only checked "did it bank anything"
    would have passed on the broken version too.
    """
    root = tmp_path / "worktree_with_no_runs_dir"
    root.mkdir()
    script = _make_checkout(root)
    # A checkout that has the script but no trial: precisely a fresh worktree.
    shutil.rmtree(root / "runs")

    result = _run_one_pass(script)

    assert result.returncode == 2, (
        f"exit {result.returncode} — a missing tune dir must be a loud exit 2 "
        f"(the `feed_bootstrap_shards.py` shape), not a silent no-op. "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "ERROR" in result.stderr, result.stderr
    assert "no tune dir" in result.stderr, result.stderr
    assert not (root / "data" / "salvage" / "rolling").exists(), (
        "the destination directory was created for a run that can never bank "
        "anything — an empty bank that LOOKS provisioned is what hid this"
    )


def test_TUNE_DIR_overrides_the_derived_location(tmp_path: Path) -> None:
    """The guard must not become a wall: the override has to actually work.

    Without this, `TUNE_DIR=${TUNE_DIR:-...}` could be reverted to a plain
    assignment and every other test here would still pass — the knob would be
    accepted and ignored, which is the defect class the guard exists for.
    """
    root = tmp_path / "empty_checkout"
    root.mkdir()
    script = _make_checkout(root)
    elsewhere = root / "runs"          # the real trial, moved out of the way
    (root / "somewhere_else").mkdir()
    shutil.move(str(elsewhere / "pbt2_small" / "tune"), str(root / "somewhere_else" / "tune"))
    shutil.rmtree(elsewhere)

    result = subprocess.run(
        ["timeout", "60", "bash", str(script)],
        capture_output=True, text=True, check=False,
        env={"PATH": "/usr/bin:/bin", "SLEEP": "600", "EVERY": "5", "KEEP": "24",
             "ONCE": "1", "TUNE_DIR": str(root / "somewhere_else" / "tune")},
    )
    assert result.returncode == 0, f"stderr={result.stderr!r}"
    banked = root / "data" / "salvage" / "rolling" / "checkpoint_000005"
    assert banked.is_dir(), (
        f"TUNE_DIR was accepted and ignored. stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
