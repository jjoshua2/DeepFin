from __future__ import annotations

from typing import Any

__all__ = ["Trainer", "trainer_kwargs_from_config"]


def __getattr__(name: str) -> Any:
    """Lazily import Trainer to avoid hard runtime dependency on zclip.

    Modules that only need lightweight utilities (e.g. target builders) should
    not fail import just because training-only extras are unavailable.
    """
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    if name == "trainer_kwargs_from_config":
        from .trainer import trainer_kwargs_from_config

        return trainer_kwargs_from_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
