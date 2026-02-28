from __future__ import annotations

from typing import Any

__all__ = ["Trainer"]


def __getattr__(name: str) -> Any:
    """Lazily import Trainer to avoid hard runtime dependency on zclip.

    Modules that only need lightweight utilities (e.g. target builders) should
    not fail import just because training-only extras are unavailable.
    """
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
