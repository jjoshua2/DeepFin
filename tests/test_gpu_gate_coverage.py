"""Drift guard for the GPU launch gate: static (AST / text), no imports of the gated scripts.

* Every HIGH-risk agent tool gated in PR 1 still calls the gate, under its own tool name.
* Every gated tool is a listed GPU entry point (so a running copy is seen by other gates).
* Live-run infrastructure never calls the gate (the gate counts it as busy; it must not gate itself).
* Every script under scripts/ with a GPU signal is a listed entry point. A new GPU script that is not
  listed would be invisible to the argv clause while it loads.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from chess_anti_engine.utils.gpu_entry_points import GPU_ENTRY_POINT_NAMES

REPO = Path(__file__).resolve().parents[1]

# HIGH-risk agent tools gated in PR 1: path -> True if it also exposes --allow-busy-gpu as a flag.
GATED_PY = {
    "scripts/arena_standard.py": True,
    "scripts/match_vs_uci.py": True,
    "scripts/match_checkpoints.py": True,
    "scripts/match_vs_handicapped_sf.py": True,
    "scripts/lc0_control_train.py": True,
    "scripts/bench_uci_engine.py": True,
    "scripts/uci_prewarm.py": True,
    "scripts/train_bootstrap.py": True,
    "scripts/e2e_strength_test.py": True,
    "scripts/bench_arena_identity.py": True,
    "scripts/bench_multi_worker.py": True,
    "scripts/bench_production_sf_workers.py": True,
    "scripts/bench_selfplay_identity.py": True,
    "scripts/profile_selfplay.py": True,
    "scripts/profile_training.py": True,
    "scripts/profile_worker_inference.py": True,
    "scripts/profile_distributed.py": True,
    "scripts/factorial_prepare_and_train.py": True,  # research launcher: startup + before training A
    "scripts/bench_pipeline.py": False,  # no argparse: the override is CAE_ALLOW_BUSY_GPU only
}
GATED_SH = ["scripts/tcec_autotune.sh", "scripts/bench_batch_wait.sh"]

# Never gated: the live run and what it starts, and the match engine itself (GUIs / the owner launch it).
NEVER_GATED = [
    "scripts/train.sh", "scripts/watchdog_loop.sh", "scripts/watchdog_pbt.sh", "scripts/monitor_fen.sh",
    "scripts/recover_stall.sh", "scripts/graceful_restart.py", "scripts/ratchet_loop.sh",
    "scripts/daily_gate_ratchet.sh", "scripts/blindspot_panel.py", "scripts/blindspot_feed_step.py",
    "scripts/blindspot_retire_step.py", "chess_anti_engine/run.py", "chess_anti_engine/worker.py",
    "chess_anti_engine/worker_pool.py", "chess_anti_engine/inference.py",
    "chess_anti_engine/uci/__main__.py", "chess_anti_engine/uci/cli.py", "scripts/deepfin",
]

# Scripts that match the GPU signal below but cannot reach the GPU (reviewed by hand). Empty today.
REVIEWED_NOT_GPU: frozenset[str] = frozenset()
GPU_SIGNAL = re.compile(
    r"""["']cuda(:\d+)?["']|torch\.cuda\.(is_available|set_device|synchronize)|CUDAExecutionProvider"""
    r"""|TensorrtExecutionProvider|--device[ =]cuda|-m chess_anti_engine\.(run|worker|uci|arena)\b"""
)


def _gate_calls(tree: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and getattr(n.func, "id", getattr(n.func, "attr", None)) == "require_gpu_idle"]


@pytest.mark.parametrize(("path", "flag"), sorted(GATED_PY.items()))
def test_gated_python_tool_calls_the_gate_under_its_own_name(path, flag):
    tree = ast.parse((REPO / path).read_text())
    calls = _gate_calls(tree)
    tool = Path(path).stem
    assert any(c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == tool for c in calls), (
        f"{path} no longer calls require_gpu_idle({tool!r})")
    if flag:
        names = {getattr(n.func, "id", None) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        assert "add_allow_busy_gpu_argument" in names, f"{path} lost its --allow-busy-gpu flag"
        assert any(any(k.arg == "allow_reason" for k in c.keywords) for c in calls), (
            f"{path}: the flag is parsed but not passed to the gate (accepted and ignored)")


@pytest.mark.parametrize("path", GATED_SH)
def test_gated_shell_tool_calls_the_wrapper(path):
    text = (REPO / path).read_text()
    tool = Path(path).stem
    assert re.search(rf"gpu_gate\.sh\"? --tool {tool} --for-pid \$\$", text), path
    assert re.search(r"\|\| exit \$\?", text), f"{path} must stop on a refusal"
    assert "--allow-busy-gpu)" in text, f"{path} lost its --allow-busy-gpu flag"


@pytest.mark.parametrize("path", sorted(GATED_PY) + GATED_SH)
def test_gated_tools_are_listed_entry_points(path):
    assert Path(path).name in GPU_ENTRY_POINT_NAMES


@pytest.mark.parametrize("path", NEVER_GATED)
def test_live_run_and_the_engine_never_call_the_gate(path):
    text = (REPO / path).read_text()
    assert "gpu_gate" not in text, path
    assert "require_gpu_idle" not in text, path


def test_every_script_with_a_gpu_signal_is_a_listed_entry_point():
    missing = []
    for p in sorted((REPO / "scripts").rglob("*")):
        if p.suffix not in (".py", ".sh") or "__pycache__" in p.parts or p.name == "gpu_gate.sh":
            continue
        if GPU_SIGNAL.search(p.read_text(errors="replace")) and p.name not in GPU_ENTRY_POINT_NAMES | REVIEWED_NOT_GPU:
            missing.append(str(p.relative_to(REPO)))
    assert not missing, (
        "GPU-signal scripts missing from chess_anti_engine/utils/gpu_entry_points.py "
        f"(add them, or to REVIEWED_NOT_GPU with a reason): {missing}")
