from __future__ import annotations

import re
import unicodedata

# PEP 440's release-stage order, as sort ranks: a dev build precedes every
# pre-release, pre-releases precede the release, and a post-release follows it.
#   1.0.dev1 < 1.0a1 < 1.0b1 < 1.0rc1 < 1.0 < 1.0.post1
_STAGE_DEV = -4
_STAGE_RELEASE = 0
_PRE_STAGE_RANK = {
    "a": -3, "alpha": -3,
    "b": -2, "beta": -2,
    "c": -1, "rc": -1, "pre": -1, "preview": -1,
}

_VERSION_RE = re.compile(
    r"""^\s*v?
    (?P<release>\d+(?:\.\d+)*)
    (?:[-_.]?(?P<pre_l>alpha|beta|preview|pre|rc|a|b|c)[-_.]?(?P<pre_n>\d+)?)?
    (?:[-_.]?(?P<post_l>post|rev|r)[-_.]?(?P<post_n>\d+)?)?
    (?:[-_.]?(?P<dev_l>dev)[-_.]?(?P<dev_n>\d+)?)?
    (?:\+(?P<local>[a-z0-9.]+))?
    \s*$""",
    re.VERBOSE | re.IGNORECASE,
)


def parse_version(v: str) -> tuple[int, int, int]:
    """Parse a version string into its (major, minor, patch) RELEASE triple.

    Accepts semver-ish and PEP 440 strings: ``'0.0.1'``, ``'1.2.3rc1'``,
    ``'v2.0'``, ``'0.1.dev5+g1234567'``. Missing parts default to 0.

    ⚑ This is the release triple ONLY, so ``'1.2.3rc1'`` and ``'1.2.3'`` both
    return ``(1, 2, 3)``. That is the honest answer to "which release is this",
    and it is deliberately NOT an ordering. Compare with :func:`version_lt`,
    which adds the pre/post/dev rank that separates them.

    Unparseable input returns ``(0, 0, 0)`` rather than raising. Both consumers
    are compatibility GATES, and only one of the three call sites has an
    exception handler around it (``server/app.py``); ``worker.py`` calls this
    on a value taken straight out of the server manifest with nothing catching,
    so raising would turn a malformed version string into a crashed worker.
    ``(0, 0, 0)`` sorts oldest, which makes an unparseable version read as "too
    old" -- the gate rejects rather than admits.
    """
    m = _VERSION_RE.match(str(v))
    if m is None:
  # Last resort for a shape we do not model: pull the leading digit runs
  # out of it. Never raises, for the reason in the docstring.
        parts = [p for p in re.split(r"[^0-9]+", str(v)) if p][:3]
    else:
        parts = m.group("release").split(".")[:3]
    nums = [int(p) for p in parts]
    while len(nums) < 3:
        nums.append(0)
    return int(nums[0]), int(nums[1]), int(nums[2])


_StageNumber = tuple[int, str]
_VersionKey = tuple[int, int, int, int, _StageNumber, _StageNumber, int, _StageNumber]


def _stage_number(value: str | None) -> _StageNumber:
    """Numeric ordering for accepted decimal digits without an int-size limit."""
    digits = value or ""
    # The unchanged regex accepts Unicode decimal digits (\\d), just as int()
    # did. Normalize them before lexical comparison; ASCII is the common path.
    if not digits.isascii():
        digits = "".join(str(unicodedata.decimal(c)) for c in digits)
    digits = digits.lstrip("0")
    return len(digits), digits


def _version_sort_key(v: str) -> _VersionKey:
    """Order pre/post/dev stages separately within the existing release triple.

    A developmental suffix belongs to its pre/post release: ``1.0b1.dev1``
    follows ``1.0a1``, and ``1.0.post1.dev1`` follows ``1.0``. Flattening all
    developmental versions into one rank loses that parent release identity.

    This retains the existing accepted syntax, release-triple comparison,
    ignored local labels and malformed-input fallback. It does not validate
    worker headers, minimum versions or wheel metadata.
    """
    major, minor, patch = parse_version(v)
    m = _VERSION_RE.match(str(v))
    if m is None:
        return (major, minor, patch, _STAGE_RELEASE, (0, ""), (-1, ""), 1, (0, ""))
    pre_label = m.group("pre_l")
    has_post = m.group("post_l") is not None
    has_dev = m.group("dev_l") is not None
    if pre_label is not None:
        pre_rank = _PRE_STAGE_RANK[pre_label.lower()]
        pre_number = _stage_number(m.group("pre_n"))
    elif has_dev and not has_post:
        pre_rank, pre_number = _STAGE_DEV, (0, "")
    else:
        pre_rank, pre_number = _STAGE_RELEASE, (0, "")
    post_number = _stage_number(m.group("post_n")) if has_post else (-1, "")
    # A missing dev suffix follows every numbered dev build of this stage.
    # Encode presence before number so no finite version number is a sentinel.
    dev_rank = 0 if has_dev else 1
    dev_number = _stage_number(m.group("dev_n"))
    return (major, minor, patch, pre_rank, pre_number, post_number, dev_rank, dev_number)


def version_lt(a: str, b: str) -> bool:
    """True when *a* orders strictly before *b*, PEP 440 stages included."""
    return _version_sort_key(a) < _version_sort_key(b)
