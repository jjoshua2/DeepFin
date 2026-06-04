from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def normalize_ffn_mult_by_layer(
    value: Any,
    *,
    num_layers: int | None = None,
) -> tuple[float, ...] | None:
    """Normalize an optional comma/list FFN multiplier schedule.

    ``None`` means "use the scalar ffn_mult for every layer". Non-empty
    schedules must provide exactly one positive finite multiplier per layer
    when ``num_layers`` is known.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if raw.lower() in ("", "none", "null", "off", "false"):
            return None
        parts: Iterable[Any] = raw.replace(";", ",").split(",")
    elif isinstance(value, (int, float, bool)):
        raise ValueError("ffn_mult_by_layer must be a sequence, not a scalar")
    else:
        try:
            parts = iter(value)
        except TypeError as exc:
            raise ValueError("ffn_mult_by_layer must be a sequence") from exc

    vals = tuple(float(part) for part in parts if str(part).strip())
    if not vals:
        return None
    for val in vals:
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError(f"ffn_mult_by_layer values must be positive finite floats, got {val!r}")
    if num_layers is not None and len(vals) != int(num_layers):
        raise ValueError(
            f"ffn_mult_by_layer length {len(vals)} must match num_layers {int(num_layers)}"
        )
    return vals


def normalize_phase_piece_thresholds(value: Any = None) -> tuple[int, int]:
    """Normalize phase buckets as ``(end_max_pieces, mid_max_pieces)``."""
    if value is None:
        return (13, 22)
    if isinstance(value, str):
        raw = value.strip()
        if raw.lower() in ("", "none", "null"):
            return (13, 22)
        parts: Iterable[Any] = raw.replace(";", ",").split(",")
    elif isinstance(value, (int, float, bool)):
        raise ValueError("phase_piece_thresholds must be a sequence, not a scalar")
    else:
        try:
            parts = iter(value)
        except TypeError as exc:
            raise ValueError("phase_piece_thresholds must be a sequence") from exc

    vals = tuple(int(part) for part in parts if str(part).strip())
    if len(vals) != 2:
        raise ValueError("phase_piece_thresholds must contain exactly two integers")
    low, high = vals
    if low < 2 or high > 32 or low >= high:
        raise ValueError(
            "phase_piece_thresholds must be ordered chess piece counts, "
            f"got {(low, high)!r}"
        )
    return (low, high)
