"""Bounded stdlib-only, caller-trusted HOLDOUT metadata acceptance.

Returns a readonly evaluation view without changing candidate bytes or issuing
learning authority. Issuer trust/hash pins are the caller's responsibility.
"""
from __future__ import annotations

from typing import cast

from scripts import core_generation_bank_acceptance as shared

ROOT_STATUS = shared.HOLDOUT_ROOT_STATUS
REVIEW_STATUS = shared.HOLDOUT_REVIEW_STATUS
ADMISSION_STATUS = shared.HOLDOUT_ADMISSION_STATUS
AcceptedHoldout = shared.AcceptedHoldout


def validate_accepted_holdout(root_ref: dict[str, str], independent_ref: dict[str, str], *,
                              max_file_bytes: int, max_total_bytes: int, max_games: int) -> AcceptedHoldout:
    return cast(AcceptedHoldout, shared._validate_generation(root_ref, independent_ref,
        max_file_bytes=max_file_bytes, max_total_bytes=max_total_bytes, max_games=max_games, holdout=True))
