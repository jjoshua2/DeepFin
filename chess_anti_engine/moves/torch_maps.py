"""Device-cached torch lookup tensors for the compact policy encoding.

Single home for the ``COMPACT_TO_FULL_POLICY`` / ``FULL_TO_COMPACT_POLICY``
device tensors that inference, the loss module, and the sparse SF-CE path
all need: one ``lru_cache`` keyed on (device type, index) instead of one
private copy per module. Kept out of ``moves/__init__`` so numpy-only
consumers of the move encoding never import torch.
"""
from __future__ import annotations

from functools import lru_cache
from typing import cast

import torch

from chess_anti_engine.moves.encode import (
    COMPACT_POLICY_SIZE,
    COMPACT_TO_FULL_POLICY,
    FULL_TO_COMPACT_POLICY,
    POLICY_SIZE,
)


@lru_cache(maxsize=16)
def _cached_table(name: str, device_type: str, device_index: int | None) -> torch.Tensor:
    device = (
        torch.device(device_type, device_index)
        if device_index is not None else torch.device(device_type)
    )
    src = COMPACT_TO_FULL_POLICY if name == "c2f" else FULL_TO_COMPACT_POLICY
    return torch.as_tensor(src, dtype=torch.long, device=device)


def _device_key(device: torch.device) -> tuple[str, int | None]:
    """Cache key for a device. torch.device("cuda") and ("cuda", 0) are the
    same physical device — normalize so they share one cached tensor."""
    # torch's stubs type device.index as int, but it is None at runtime for
    # unindexed devices like torch.device("cuda").
    index = cast("int | None", device.index)
    if index is None and device.type == "cuda":
        index = torch.cuda.current_device() if torch.cuda.is_available() else 0
    return device.type, index


def compact_to_full_index(device: torch.device) -> torch.Tensor:
    """(1858,) long tensor mapping compact indices to full-4672 indices."""
    return _cached_table("c2f", *_device_key(device))


def compact_to_full_index_for(tensor: torch.Tensor) -> torch.Tensor:
    """Same as :func:`compact_to_full_index`, keyed on a tensor's device."""
    return _cached_table("c2f", *_device_key(tensor.device))


def full_to_compact_index(device: torch.device) -> torch.Tensor:
    """(4672,) long tensor mapping full indices to compact (-1 = no move)."""
    return _cached_table("f2c", *_device_key(device))


def policy_index_remap_table(
    src_width: int, dst_width: int, device: torch.device,
) -> torch.Tensor | None:
    """Lookup table mapping ``src_width``-space policy indices to
    ``dst_width``-space, or None when the widths already agree. Raises on
    unknown width pairs."""
    if src_width == dst_width:
        return None
    if src_width == POLICY_SIZE and dst_width == COMPACT_POLICY_SIZE:
        return full_to_compact_index(device)
    if src_width == COMPACT_POLICY_SIZE and dst_width == POLICY_SIZE:
        return compact_to_full_index(device)
    raise ValueError(
        f"policy width {src_width} is incompatible with target width {dst_width}"
    )
