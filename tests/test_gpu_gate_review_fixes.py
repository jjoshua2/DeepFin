"""Regressions for the #1518 review findings (no torch import).

1. A listed script run as a module (``python -m scripts.<name>``) is GPU work for the argv clause.
2. ``match_vs_uci.engine_may_use_gpu`` resolves the EFFECTIVE device: ``--devices`` overrides
   ``--device``, so ``--device cpu --devices cuda:0,cuda:1`` is GPU.
3. A trailing ``--allow-busy-gpu`` with no reason is a usage error (exit 2) in the shell launchers,
   not an endless argument loop.
"""

from __future__ import annotations

import functools
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

from chess_anti_engine.utils import gpu_gate as g

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("argv", "hit"), [
    (["python3", "-m", "scripts.core_generation_bank_learner", "--x"], "-m scripts.core_generation_bank_learner"),
    (["python3", "-mscripts.arena_standard"], "-m scripts.arena_standard"),
    (["python3", "-m", "scripts.status"], None),
])
def test_listed_script_run_as_a_module_is_an_entry_point(argv, hit):
    assert g.argv_entry_point(argv) == hit


def test_process_scan_sees_a_module_launched_listed_script(tmp_path):
    from tests.test_gpu_gate import GPU_RDEV, SELF, UID, FakeSmi, Proc

    proc = Proc(tmp_path / "proc")
    proc.add(610, 400, ["python3", "-m", "scripts.core_generation_bank_learner"])
    seams = g.Seams(proc_root=proc.root, home=tmp_path, uid=UID, pid=SELF, nvidia_smi=FakeSmi(),
                    sleep=lambda s: None, gpu_rdevs=lambda: {GPU_RDEV}, fd_stat=proc.fd_stat)
    d = g.check("arena_standard", env={}, seams=seams)
    assert d.code == 8
    assert "-m scripts.core_generation_bank_learner" in d.reasons[0]


@functools.cache
def _match_module():
    name = "match_vs_uci_under_test"
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts/match_vs_uci.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # its dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize(("command", "gpu"), [
    ("stockfish", False),
    ("/usr/local/bin/stockfish-17 --threads 2", False),
    ("python -m chess_anti_engine.uci --checkpoint x.pt --device cpu", False),
    ("python -m chess_anti_engine.uci --checkpoint x.pt --device=cpu", False),
    ("python -m chess_anti_engine.uci --checkpoint x.pt", True),
    ("python -m chess_anti_engine.uci --checkpoint x.pt --device cuda", True),
    ("python -m chess_anti_engine.uci --device cpu --devices cuda:0,cuda:1", True),
    ("python -m chess_anti_engine.uci --devices=cuda:0 --device cpu", True),
    ("python -m chess_anti_engine.uci --device cuda --devices cpu", False),
    ("/opt/lc0/lc0 --backend=cuda", True),
])
def test_engine_device_is_resolved_with_devices_precedence(command, gpu):
    assert _match_module().engine_may_use_gpu(command) is gpu


@pytest.mark.parametrize(("tool", "argv"), [
    ("tcec_autotune.sh", ["--checkpoint", "x.pt", "--allow-busy-gpu"]),
    ("bench_batch_wait.sh", ["--allow-busy-gpu"]),
])
def test_trailing_override_without_reason_is_a_usage_error(tool, argv):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAE_")}
    env["CUDA_VISIBLE_DEVICES"] = ""
    r = subprocess.run(["bash", f"scripts/{tool}", *argv], cwd=REPO, env=env, capture_output=True,
                       text=True, check=False, timeout=30)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert "needs a REASON" in r.stderr
