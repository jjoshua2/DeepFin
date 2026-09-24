"""Shared immutable input schema for numeric-map operation drivers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Workload:
    name: str
    bits: int
    initial: tuple[tuple[int, int], ...]
    ops: tuple[tuple[int, int, int], ...]  # get=0, put=1, remove=2
