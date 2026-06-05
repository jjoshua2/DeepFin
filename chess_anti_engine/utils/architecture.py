from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any


def _coerce_positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} values must be positive integers, got {value!r}")
    if isinstance(value, int):
        val = value
    elif isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"{name} values must be positive integers, got {value!r}")
        val = int(value)
    else:
        val = int(value)
    if val <= 0:
        raise ValueError(f"{name} values must be positive integers, got {value!r}")
    return val


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


def normalize_embed_dim_by_layer(
    value: Any,
    *,
    num_layers: int | None = None,
) -> tuple[int, ...] | None:
    """Normalize an optional comma/list embedding-width schedule.

    ``None`` means "use embed_dim for every layer". Non-empty schedules must
    provide exactly one positive integer width per layer when ``num_layers`` is
    known. The fixed head-count divisibility constraint is checked in the model
    builder because it needs ``num_heads`` too.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if raw.lower() in ("", "none", "null", "off", "false"):
            return None
        parts: Iterable[Any] = raw.replace(";", ",").split(",")
    elif isinstance(value, (int, float, bool)):
        raise ValueError("embed_dim_by_layer must be a sequence, not a scalar")
    else:
        try:
            parts = iter(value)
        except TypeError as exc:
            raise ValueError("embed_dim_by_layer must be a sequence") from exc

    vals = tuple(_coerce_positive_int(part, name="embed_dim_by_layer") for part in parts if str(part).strip())
    if not vals:
        return None
    if num_layers is not None and len(vals) != int(num_layers):
        raise ValueError(
            f"embed_dim_by_layer length {len(vals)} must match num_layers {int(num_layers)}"
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
    # A full board has 32 pieces, and the open bucket is ``count > high``; high
    # must stay <= 31 so that bucket is reachable. low >= 2 keeps both kings.
    if low < 2 or high > 31 or low >= high:
        raise ValueError(
            "phase_piece_thresholds must be ordered chess piece counts "
            "(2 <= end_max < mid_max <= 31), "
            f"got {(low, high)!r}"
        )
    return (low, high)
