"""Explicit incomplete search outcomes shared by native and caller envelopes."""
from __future__ import annotations

from typing import Any


class SolvedRootWitnessUnavailable(RuntimeError):
    """Known root outcomes without corresponding action proofs at the limit.

    Partial results retain completed batch members. Unresolved members have
    action -1 and zero policy, and must never be appended as training decisions.
    fallback_actions are legal UNKNOWN children, not proven preserving moves.
    """

    def __init__(self, outcomes: dict[int, int],
                 partial_result: tuple[Any, ...],
                 fallback_actions: dict[int, int], child_checks: dict[int, int]) -> None:
        super().__init__("no outcome-preserving child witness within simulation budget "
                         "and bounded tablebase allowance")
        self.outcomes = outcomes
        self.partial_result = partial_result
        self.fallback_actions = fallback_actions
        self.child_checks = child_checks
