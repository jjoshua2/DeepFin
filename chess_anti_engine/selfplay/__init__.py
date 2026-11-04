"""Selfplay APIs without loading model execution for pure game helpers."""
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .manager import play_batch as play_batch
    from .match import MatchStats as MatchStats
    from .match import play_match_batch as play_match_batch

__all__ = ["MatchStats", "play_batch", "play_match_batch"]

_EXPORTS = {"play_batch": ".manager", "MatchStats": ".match", "play_match_batch": ".match"}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
