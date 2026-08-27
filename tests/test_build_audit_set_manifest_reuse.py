"""`build_audit_set.py --replay-dir` must be required only where it is READ.

Stage 1 (sample a manifest) reads it; stage 2 (label) does not. Building a
referee PAIR -- the same positions labelled by a second Stockfish -- is
exactly the reuse path, and `required=True` forced that caller to pass a
directory the run then ignores. A flag that is accepted and silently ignored
is this codebase's signature defect, so the requirement is now conditional on
the manifest actually being absent.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_ROW = {"key": "k0", "fen": "8/8/8/8/8/8/8/K6k w - - 0 1", "phase": 0, "source": "selfplay"}


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "build_audit_set.py"), *args],
        cwd=cwd, capture_output=True, text=True, check=False,
        env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
    )


def test_reuse_path_runs_with_no_replay_dir(tmp_path: Path) -> None:
    """Manifest present + every row already labelled: no SF, no --replay-dir."""
    out = tmp_path / "audit_set_ref_b.jsonl"
    (tmp_path / "audit_set_ref_b.jsonl.manifest.jsonl").write_text(
        json.dumps(_ROW) + "\n", encoding="utf-8",
    )
    out.write_text(json.dumps({**_ROW, "move_cp": [["a1b1", 0]]}) + "\n", encoding="utf-8")

    r = _run(["--out", str(out), "--stockfish", "/nonexistent/sf", "--resume"], tmp_path)

    assert r.returncode == 0, f"stdout={r.stdout!r} stderr={r.stderr!r}"
    assert "reusing" in r.stdout, r.stdout
    readme = (tmp_path / "audit_set_ref_b_README.md").read_text(encoding="utf-8")
    assert "manifest reused; not re-sampled" in readme, readme


def test_sampling_path_still_demands_replay_dir(tmp_path: Path) -> None:
    """No manifest to reuse => the flag IS read => it stays mandatory."""
    out = tmp_path / "audit_set_new.jsonl"
    r = _run(["--out", str(out), "--stockfish", "/nonexistent/sf"], tmp_path)

    assert r.returncode != 0
    combined = r.stdout + r.stderr
    assert "--replay-dir is required to SAMPLE a new manifest" in combined, combined
