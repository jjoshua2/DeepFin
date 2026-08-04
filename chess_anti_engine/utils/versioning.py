from __future__ import annotations

import re

# PEP 440's release-stage order, as sort ranks: a dev build precedes every
# pre-release, pre-releases precede the release, and a post-release follows it.
#   1.0.dev1 < 1.0a1 < 1.0b1 < 1.0rc1 < 1.0 < 1.0.post1
_STAGE_DEV = -4
_STAGE_RELEASE = 0
_STAGE_POST = 1
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


def _version_sort_key(v: str) -> tuple[int, int, int, int, int]:
    """``(major, minor, patch, stage_rank, stage_number)`` -- a total order.

    The stage rank is what the previous implementation threw away. It split on
    any non-digit run and took the first three numbers, so every PEP 440 suffix
    either vanished or was PROMOTED INTO A VERSION COMPONENT:

    * ``'1.2.3rc1'``          -> ``(1, 2, 3)``, i.e. EQUAL to ``'1.2.3'``;
    * ``'1.2.3.post1'``       -> ``(1, 2, 3)``, likewise equal;
    * ``'0.1.dev5+g1234567'`` -> ``(0, 1, 5)``, so a dev build of 0.1
      outranked the real ``0.1.4``.

    A worker on ``X.Y.Zrc1`` therefore passed a ``min_worker_version: X.Y.Z``
    gate that exists precisely to keep pre-release workers out, and a dev build
    passed ANY min-version gate in its own minor series.

    ⚑ Latent rather than live today: ``pyproject.toml`` pins a static
    ``version = "0.0.2"`` with no setuptools-scm, so no suffixed string is
    produced right now. The trap is that the version is hand-bumped, and
    ``0.0.3rc1`` during a staged rollout is exactly the situation a
    min-version gate exists for.
    """
    major, minor, patch = parse_version(v)
    m = _VERSION_RE.match(str(v))
    if m is None:
        return (major, minor, patch, _STAGE_RELEASE, 0)
  # Order matters: `1.0rc1.dev2` is a dev build OF a pre-release, and dev is
  # the lower rank, so it must be tested first.
    if m.group("dev_l") is not None:
        return (major, minor, patch, _STAGE_DEV, int(m.group("dev_n") or 0))
    if m.group("pre_l") is not None:
        return (major, minor, patch, _PRE_STAGE_RANK[m.group("pre_l").lower()],
                int(m.group("pre_n") or 0))
    if m.group("post_l") is not None:
        return (major, minor, patch, _STAGE_POST, int(m.group("post_n") or 0))
    return (major, minor, patch, _STAGE_RELEASE, 0)


def version_lt(a: str, b: str) -> bool:
    """True when *a* orders strictly before *b*, PEP 440 stages included."""
    return _version_sort_key(a) < _version_sort_key(b)
