from __future__ import annotations

import re


def parse_version(v: str) -> tuple[int, int, int]:
    """Parse a version string into a (major, minor, patch) tuple.

    Accepts semver-ish strings like '0.0.1', '1.2.3rc1', 'v2.0'.
    Missing parts default to 0.
    """
    s = str(v).strip()
    if s.startswith("v"):
        s = s[1:]

    # Split on any non-digit.
    parts = [p for p in re.split(r"[^0-9]+", s) if p]
    nums = [int(p) for p in parts[:3]]
    while len(nums) < 3:
        nums.append(0)
    return int(nums[0]), int(nums[1]), int(nums[2])


def version_lt(a: str, b: str) -> bool:
    return parse_version(a) < parse_version(b)


def version_gt(a: str, b: str) -> bool:
    return parse_version(a) > parse_version(b)


def version_eq(a: str, b: str) -> bool:
    return parse_version(a) == parse_version(b)
