"""Where an arena seat's search value comes from (torch-free; see ``hybrid_value``).

Imported at the top of ``scripts/arena_standard.py``, which must not import
torch before it has pointed the compile cache somewhere, so this module holds
only the parsing and the records.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def parse_recal(text: str) -> tuple[float, float]:
    """``"T,b"`` -> ``(T, b)``; T finite and > 0, b finite."""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"value recalibration must be 'T,b', got {text!r}")
    try:
        t, b = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"value recalibration must be two numbers 'T,b', got {text!r}") from exc
    if not (math.isfinite(t) and t > 0):
        raise ValueError(f"value recalibration T must be finite and > 0, got {t}")
    if not math.isfinite(b):
        raise ValueError(f"value recalibration b must be finite, got {b}")
    return t, b


@dataclass(frozen=True)
class SeatValue:
    """Where one seat's search value comes from. The default is the plain net."""

    value_from: str | None = None
    recal: tuple[float, float] | None = None

    @property
    def active(self) -> bool:
        return self.value_from is not None or self.recal is not None

    def settings_record(self) -> dict[str, Any]:
        """What the game-log fingerprint binds (paths, like the seats themselves)."""
        return {
            "value_from": self.value_from,
            "recal": None if self.recal is None else [float(self.recal[0]), float(self.recal[1])],
        }

    def describe(self) -> str:
        parts = []
        if self.value_from is not None:
            parts.append(f"value_from={self.value_from}")
        if self.recal is not None:
            parts.append(f"recal(T={self.recal[0]:g},b={self.recal[1]:g})")
        return " ".join(parts) or "own value"
