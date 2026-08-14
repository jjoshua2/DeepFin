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


def normalize_aux_policy_head_dim(value: Any = None) -> int | None:
    """Normalize the AUXILIARY policy heads' projection width.

    ``None`` (and yaml's ``null`` / an absent key) means "trunk width" — today's
    exact behaviour, bit-identical. Anything else must be a positive integer.

    ⚑ Kept None-preserving on purpose. Collapsing ``None`` to ``0`` or to a
    hardcoded default would build zero-width or wrongly-sized ``q``/``k``
    projections that ``load_state_dict_tolerant`` then DROPS with only a
    ``print()`` — the silent-wrongness shape this repo keeps getting bitten by.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if raw.lower() in ("", "none", "null"):
            return None
        value = raw
    try:
        return _coerce_positive_int(value, name="aux_policy_head_dim")
    except (TypeError, ValueError) as exc:
        # ⚑ Every rejection path is normalized to ONE ValueError that NAMES the
        # key. Raw `int()` failures do not: a yaml list/dict surfaces as
        # `TypeError`, and `aux_policy_head_dim: wide` surfaces as
        # "invalid literal for int() with base 10: 'wide'" — which tells an
        # operator staring at a boot failure nothing about which of the ~200
        # yaml keys they mistyped.
        raise ValueError(
            f"aux_policy_head_dim must be a positive int or None, got {value!r}"
        ) from exc


# The project's ONE definition of a game phase, as piece counts:
# ``end`` is ``count <= 13``, ``open`` is ``count > 22``, ``mid`` is the rest.
#
# Lives here — in the dependency-free architecture helpers — rather than in
# either consumer, because both `eval/audit.py` (per-phase deep-SF regret) and
# `train/losses.py` (per-phase loss split) bucket against it and the two are
# only comparable while they agree. It was previously written out twice in this
# function and a third time in `eval/audit.py`; a rule stated three times is a
# rule stated inconsistently.
DEFAULT_PHASE_PIECE_THRESHOLDS: tuple[int, int] = (13, 22)


def normalize_phase_piece_thresholds(value: Any = None) -> tuple[int, int]:
    """Normalize phase buckets as ``(end_max_pieces, mid_max_pieces)``."""
    if value is None:
        return DEFAULT_PHASE_PIECE_THRESHOLDS
    if isinstance(value, str):
        raw = value.strip()
        if raw.lower() in ("", "none", "null"):
            return DEFAULT_PHASE_PIECE_THRESHOLDS
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
