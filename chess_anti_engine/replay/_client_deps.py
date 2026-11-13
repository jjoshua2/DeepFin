"""Fail clearly when the replay stack's dependencies are not installed.

Imported first by ``chess_anti_engine/replay/__init__.py``. The replay package
(and so the selfplay worker, ``chess_anti_engine.run`` and the trainer) needs
zarr/numcodecs, which the ``match`` (engine-only) install does not ship.
Without this, ``python -m chess_anti_engine.worker`` on a match install died
with a bare ``ModuleNotFoundError: No module named 'zarr'``. The UCI engine
never imports this package (tests/test_packaging_extras.py).
"""
from __future__ import annotations

import importlib

try:
    for _name in ("zarr", "numcodecs"):
        importlib.import_module(_name)
except ImportError as exc:
    raise ModuleNotFoundError(
        f"{exc}. The replay / selfplay / training code needs the client or server "
        "install, not the engine-only `match` install: selfplay worker "
        "`pip install -e \".[client]\"`, training server `pip install -e \".[server]\"`.",
        name=exc.name,
    ) from exc
