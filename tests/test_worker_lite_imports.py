"""Pin the lite-client packaging contract.

``pyproject.toml`` keeps the core dependency list to exactly what the
selfplay client needs: the transitive import graph of ``worker.py`` must
stay within core deps + the ``[worker]`` extra. This test walks that graph
the same way the pyproject comment was derived, so the contract fails
loudly when worker code grows a new third-party import (instead of
surfacing as a missing-dep crash on a lite install).
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE = "chess_anti_engine"
REPO_ROOT = Path(__file__).resolve().parent.parent

# Distributions a lite client install provides: core deps from
# pyproject.toml (python-chess, numpy, torch, pyyaml, zarr — numcodecs is
# a zarr dependency) plus requests from the [worker] extra.
ALLOWED_THIRD_PARTY = {"chess", "numpy", "torch", "yaml", "zarr", "numcodecs", "requests"}


def _worker_external_imports() -> set[str]:
    seen: set[Path] = set()
    queue = [REPO_ROOT / PACKAGE / "worker.py"]
    external: set[str] = set()
    while queue:
        path = queue.pop()
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules = [node.module]
            else:
                continue
            for module in modules:
                if module.startswith(PACKAGE):
                    rel = REPO_ROOT / module.replace(".", "/")
                    queue += [rel.with_suffix(".py"), rel / "__init__.py"]
                else:
                    external.add(module.split(".")[0])
    return external


def test_worker_import_graph_stays_within_lite_install() -> None:
    third_party = {
        mod
        for mod in _worker_external_imports()
        if mod not in sys.stdlib_module_names
    }
    leaked = third_party - ALLOWED_THIRD_PARTY
    assert not leaked, (
        f"worker.py import graph now reaches {sorted(leaked)} — either add the "
        "distribution to core deps/[worker] in pyproject.toml (keeping the lite "
        "client as small as possible) or make the import lazy/optional."
    )
