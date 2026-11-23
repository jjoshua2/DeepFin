"""Take-effect proof per gated tool: each entry point, run as a real subprocess, stops at its gate.

With a fake BUSY GPU (a fake ``nvidia-smi`` on PATH reporting 30,000 MiB), each tool must exit 8
at the gate, before any GPU work. With ``--allow-busy-gpu`` (or the env twin) it must proceed past
the gate. The test seam ``CAE_GPU_GATE_TEST=1`` makes the gate exit 97 right after a pass, so no
tool ever runs past its gate here. Every subprocess runs with ``CUDA_VISIBLE_DEVICES=`` (empty) and
``PYTORCH_NVML_BASED_CUDA_CHECK=1``, so the real device is never reached. The seam makes the gate
probe anyway.

``torch_imported`` at the refusal is pinned per tool. ``False`` means the gate precedes the torch
import. ``True`` means torch is imported at module level (directly or through a ``chess_anti_engine`` import)
before ``main()`` can run (see
``TORCH_BEFORE_GATE``). There the refusal still precedes any CUDA allocation and any child process.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# tool -> (argv after the script, torch already imported when the gate refuses)
TOOLS: dict[str, tuple[list[str], bool]] = {
    "arena_standard.py": (["--candidate", "x.pt", "--reference", "y.pt"], True),
    "match_vs_uci.py": (["--engine-a", "python -m chess_anti_engine.uci --checkpoint x.pt", "--engine-b", "stockfish"], False),
    "match_checkpoints.py": (["--a", "x.pt", "--b", "y.pt"], True),
    "match_vs_handicapped_sf.py": (["--checkpoint", "x.pt", "--regret", "0.1", "--out", "o"], True),
    "lc0_control_train.py": (["--shards", "s", "--out-dir", "o", "--steps", "1"], True),
    "bench_uci_engine.py": (["--checkpoint", "x.pt"], False),
    "uci_prewarm.py": (["--checkpoint", "x.pt"], False),
    "train_bootstrap.py": (["--config", "configs/pbt2_small.yaml"], True),
    "e2e_strength_test.py": ([], True),
    "bench_arena_identity.py": (["--model", "x", "--syzygy", "s", "--starts", "st", "--out", "o"], True),
    "bench_multi_worker.py": ([], False),
    "bench_production_sf_workers.py": ([], False),
    "bench_selfplay_identity.py": (["--model", "x", "--sf-path", "s", "--syzygy", "s", "--configs", "c",
                                    "--starts", "st", "--out", "o"], True),
    "profile_selfplay.py": ([], True),
    "profile_training.py": ([], True),
    "profile_worker_inference.py": (["--checkpoint", "x.pt"], True),
    "profile_distributed.py": ([], False),
    "bench_pipeline.py": ([], True),
    "factorial_prepare_and_train.py": (["--config", "c.json", "--sha256", "0" * 64], False),
}
# Tools whose module imports torch before main() runs (the gate cannot precede that import):
TORCH_BEFORE_GATE = sorted(t for t, (_, imported) in TOOLS.items() if imported)
NO_FLAG = {"bench_pipeline.py"}  # no argparse: the env twin carries the override


def _env(tmp_path: Path) -> dict[str, str]:
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    smi = b / "nvidia-smi"
    smi.write_text("#!/bin/sh\ncase \"$1\" in --query-gpu=memory.used) echo 30000;; *) echo 97;; esac\n")
    smi.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("CAE_")}
    env.update(PATH=f"{b}:{env.get('PATH', '')}", HOME=str(tmp_path), CUDA_VISIBLE_DEVICES="",
               PYTORCH_NVML_BASED_CUDA_CHECK="1", PYTHONPATH=str(REPO), CAE_GPU_GATE_TEST="1",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    return env


def _run(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, check=False, timeout=600)


@pytest.mark.parametrize("tool", sorted(TOOLS))
def test_python_tool_stops_at_its_gate_on_a_busy_gpu_and_passes_with_the_flag(tool, tmp_path):
    argv, torch_first = TOOLS[tool]
    env = _env(tmp_path)
    cmd = [sys.executable, f"scripts/{tool}", *argv]
    r = _run(cmd, env)
    assert r.returncode == 8, (r.returncode, r.stderr[-3000:])
    assert f"gpu_gate: {tool[:-3]}: REFUSED: GPU memory.used 30000 MiB" in r.stderr, r.stderr[-3000:]
    assert f"gpu_gate-test: torch_imported={torch_first}" in r.stderr, r.stderr[-3000:]
    if tool in NO_FLAG:
        r = _run(cmd, {**env, "CAE_ALLOW_BUSY_GPU": "take-effect test"})
    else:
        r = _run([*cmd, "--allow-busy-gpu", "take-effect test"], env)
    assert r.returncode == 97, (r.returncode, r.stderr[-3000:])
    assert "BUSY-GPU OVERRIDE" in r.stderr
    assert "take-effect test" in r.stderr


@pytest.mark.parametrize(("tool", "extra"), [("tcec_autotune.sh", ["--checkpoint", "CKPT"]), ("bench_batch_wait.sh", [])])
def test_shell_tool_stops_at_its_gate_on_a_busy_gpu_and_passes_with_the_flag(tool, extra, tmp_path):
    ckpt = tmp_path / "trainer.pt"
    ckpt.write_bytes(b"not a checkpoint")
    argv = [str(ckpt) if a == "CKPT" else a for a in extra]
    env = _env(tmp_path)
    r = _run(["bash", f"scripts/{tool}", *argv], env)
    assert r.returncode == 8, (r.returncode, r.stdout[-2000:], r.stderr[-2000:])
    assert f"gpu_gate: {tool[:-3]}: REFUSED: GPU memory.used 30000 MiB" in r.stderr
    r = _run(["bash", f"scripts/{tool}", *argv, "--allow-busy-gpu", "take-effect test"], env)
    assert r.returncode == 97, (r.returncode, r.stderr[-2000:])
    assert "BUSY-GPU OVERRIDE" in r.stderr
