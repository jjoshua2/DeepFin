"""scripts/uci_prewarm.py must warm the engine configuration the operator registers.

It used to pass its own defaults (`--chunk-sims 512 --vl-gather 256`) where the
engine defaults to 2048 / 512, so a bare prewarm compiled and timed a search
shape the tournament engine never runs. Engine flags are now passed only when
given, and the engine's own defaults apply otherwise.
"""
from __future__ import annotations

import importlib.util
from itertools import pairwise
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]

_ENGINE_FLAGS = ("--chunk-sims", "--max-batch", "--walkers", "--vl-gather",
                 "--search-parallel", "--compile-mode", "--device", "--devices",
                 "--compile-cache-dir")


def _prewarm() -> ModuleType:
    spec = importlib.util.spec_from_file_location("uci_prewarm", ROOT / "scripts" / "uci_prewarm.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cmd(argv: list[str]) -> list[str]:
    mod = _prewarm()
    a = mod.build_parser().parse_args(argv)
    return mod._engine_cmd(
        checkpoint=a.checkpoint, device=a.device, devices=a.devices,
        multi_gpu_pucv=bool(a.multi_gpu_pucv), search_parallel=a.search_parallel,
        compile_mode=a.compile_mode, no_compile=bool(a.no_compile), vl_gather=a.vl_gather,
        max_batch=a.max_batch, chunk_sims=a.chunk_sims, walkers=a.walkers,
        compile_cache_dir=a.compile_cache_dir,
    )


def test_a_bare_prewarm_passes_no_engine_search_flag() -> None:
    cmd = _cmd(["--checkpoint", "net.pt"])
    assert not set(cmd) & set(_ENGINE_FLAGS), cmd
    assert cmd[cmd.index("--checkpoint") + 1] == "net.pt"


def test_given_engine_flags_are_passed_through_unchanged() -> None:
    cmd = _cmd(["--checkpoint", "net.pt", "--chunk-sims", "2048", "--vl-gather", "512",
                "--max-batch", "4096", "--walkers", "1", "--search-parallel", "gumbel",
                "--compile-mode", "default", "--devices", "cuda:0,cuda:1"])
    pairs = dict(pairwise(cmd))
    assert pairs["--chunk-sims"] == "2048"
    assert pairs["--vl-gather"] == "512"
    assert pairs["--max-batch"] == "4096"
    assert pairs["--walkers"] == "1"
    assert pairs["--search-parallel"] == "gumbel"
    assert pairs["--compile-mode"] == "default"
    assert pairs["--devices"] == "cuda:0,cuda:1"


def test_device_and_compile_cache_dir_pass_through_only_when_given() -> None:
    pairs = dict(pairwise(_cmd(["--checkpoint", "net.pt", "--device", "cuda:1",
                                "--compile-cache-dir", "/srv/cache"])))
    assert pairs["--device"] == "cuda:1"
    assert pairs["--compile-cache-dir"] == "/srv/cache"


def test_the_engine_inherits_the_environment() -> None:
    """DEEPFIN_COMPILE_CACHE reaches the engine: it is started without env=."""
    import ast

    tree = ast.parse((ROOT / "scripts" / "uci_prewarm.py").read_text(encoding="utf-8"))
    popens = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute) and n.func.attr == "Popen"]
    assert popens
    assert all(kw.arg != "env" for call in popens for kw in call.keywords)
