"""Stable completion sweeps shared by game producers.

Domain adapters decide what acceptance means: a selfplay callback returning is
not a durable raw-file or companion acknowledgment. This module only releases
independently accepted slots, in their existing order. It owns no boards, RNG,
inference, outcome policy, publication, or admission budget.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

Slot = TypeVar("Slot")
Receipt = TypeVar("Receipt")


@dataclass(frozen=True)
class AcceptedCompletion(Generic[Receipt]):
    """An adapter's accepted result, including an intentionally empty receipt."""

    receipt: Receipt


def drain_completed_slots(
    slots: Iterable[Slot], *,
    poll: Callable[[Slot], AcceptedCompletion[Receipt] | None],
    release: Callable[[Slot, Receipt], None],
) -> int:
    """Poll each owned slot once; release only accepted completions.

    ``None`` means still owned, not a barrier for later slots. Snapshotting the
    iteration permits release/refill to mutate the owner's slot collection
    without admitting a replacement to this same sweep. The adapter must mark
    or remove accepted slots, and decides whether release permits refill.

    The sweep adds no waits of its own. An adapter owns its existing blocking
    policy (including selfplay's finite/teardown label flush) as well as its
    acceptance condition. Poll errors propagate without calling release for
    that slot. Release errors propagate unchanged; the adapter owns atomicity
    and recovery. Earlier accepted completions are not rolled back.
    """
    completed = 0
    for slot in tuple(slots):
        result = poll(slot)
        if result is not None:
            release(slot, result.receipt)
            completed += 1
    return completed
