"""Evaluation APIs; pure plane fingerprints do not initialize Torch puzzles."""
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .puzzles import PuzzleResult as PuzzleResult
    from .puzzles import PuzzleSuite as PuzzleSuite
    from .puzzles import load_epd as load_epd
    from .puzzles import load_lichess_csv as load_lichess_csv
    from .puzzles import run_policy_sequence_eval as run_policy_sequence_eval
    from .puzzles import run_puzzle_eval as run_puzzle_eval
    from .puzzles import run_value_head_puzzle_eval as run_value_head_puzzle_eval

__all__ = [
    "PuzzleResult",
    "PuzzleSuite",
    "load_epd",
    "load_lichess_csv",
    "run_policy_sequence_eval",
    "run_puzzle_eval",
    "run_value_head_puzzle_eval",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".puzzles", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
