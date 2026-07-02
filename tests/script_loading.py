"""Shared loader for scripts/ modules under test (scripts/ is not a package)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_script_module(filename: str, module_name: str | None = None) -> ModuleType:
    """Exec ``scripts/<filename>`` and return it as a module.

    Registers the module in ``sys.modules`` under ``module_name`` (default:
    the file stem) so dataclasses and pickling inside the script resolve
    normally.
    """
    name = module_name if module_name is not None else Path(filename).stem
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
