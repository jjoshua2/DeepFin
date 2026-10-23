"""Search APIs, loaded on first access rather than for pure sampling imports."""
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gumbel import GumbelConfig as GumbelConfig
    from .gumbel import run_gumbel_root as run_gumbel_root
    from .gumbel import run_gumbel_root_many as run_gumbel_root_many
    from .puct import MCTSConfig as MCTSConfig
    from .puct import run_mcts as run_mcts
    from .puct import run_mcts_many as run_mcts_many

__all__ = [
    "GumbelConfig",
    "MCTSConfig",
    "run_gumbel_root",
    "run_gumbel_root_many",
    "run_mcts",
    "run_mcts_many",
]

_EXPORTS = {
    "GumbelConfig": ".gumbel", "run_gumbel_root": ".gumbel",
    "run_gumbel_root_many": ".gumbel", "MCTSConfig": ".puct",
    "run_mcts": ".puct", "run_mcts_many": ".puct",
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    # import_module retains Python's normal per-module initialization lock.
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
