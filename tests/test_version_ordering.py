"""A pre-release must not pass a min-version gate set to its own release (A21).

`parse_version` split on any non-digit run and took the first three numbers, so
every PEP 440 suffix either vanished or was promoted into a version component:

    '1.2.3rc1'          -> (1, 2, 3)   equal to '1.2.3'
    '1.2.3.post1'       -> (1, 2, 3)   equal to '1.2.3'
    '0.1.dev5+g1234567' -> (0, 1, 5)   a dev build of 0.1 outranking 0.1.4

Both compatibility gates consume it: `worker.py:2666` (`version_too_old`),
`worker.py:2703` (wheel self-update) and `server/app.py:1262` (server-side
reject). A worker on `X.Y.Zrc1` passed a `min_worker_version: X.Y.Z` gate that
exists precisely to keep pre-release workers out, and a dev build passed ANY
min-version gate in its own minor series.

⚑ WHY ORDERING RATHER THAN REJECTION. The alternative was to reject suffixed
strings loudly. Only ONE of the three call sites has an exception handler:
`server/app.py:1262` wraps `version_lt` in `try/except Exception` and turns it
into "bad worker version header". The two `worker.py` sites call it bare, on a
value read straight out of the server manifest, so raising there converts a
malformed version string into a crashed worker — replacing a gate that admits
the wrong worker with one that kills the right one. Ordering keeps every
caller's control flow identical for the release-only strings that exist today
and makes the suffixed cases sort correctly.

⚑ LATENT, not live: `pyproject.toml` pins a static `version = "0.0.2"` with no
setuptools-scm, so nothing produces a suffixed string right now. The version is
hand-bumped, and `0.0.3rc1` during a staged rollout is exactly the case the
gate exists for.
"""

from __future__ import annotations

import pytest

from chess_anti_engine.utils.versioning import parse_version, version_lt


@pytest.mark.parametrize(
    ("older", "newer"),
    [
  # The reported cases.
        ("1.2.3rc1", "1.2.3"),
        ("0.1.dev5+g1234567", "0.1.0"),
        ("0.1.dev5+g1234567", "0.1.4"),
        ("1.2.3", "1.2.3.post1"),
  # The full PEP 440 stage ladder within one release.
        ("1.0.dev1", "1.0a1"),
        ("1.0a1", "1.0b1"),
        ("1.0b1", "1.0rc1"),
        ("1.0rc1", "1.0"),
        ("1.0", "1.0.post1"),
  # Numbering within a stage.
        ("1.0rc1", "1.0rc2"),
        ("1.0.dev1", "1.0.dev2"),
        ("1.0.post1", "1.0.post2"),
  # Ordinary release ordering must be unchanged.
        ("0.0.1", "0.0.2"),
        ("0.9.9", "1.0.0"),
        ("1.2.3", "1.3.0"),
    ],
)
def test_the_older_version_sorts_first(older: str, newer: str) -> None:
    assert version_lt(older, newer) is True, f"{older!r} must sort below {newer!r}"
    assert version_lt(newer, older) is False, f"{newer!r} must not sort below {older!r}"


def test_a_pre_release_does_not_pass_a_gate_set_to_its_own_release() -> None:
    """The failure scenario, stated as the gates state it.

    `worker.py:2666` is `version_too_old = version_lt(PACKAGE_VERSION, min_v)`.
    An rc worker against `min_worker_version: 1.2.3` must read as too old.
    """
    assert version_lt("1.2.3rc1", "1.2.3") is True


def test_a_dev_build_does_not_outrank_a_real_release() -> None:
    """`0.1.dev5+g1234567` used to parse as (0, 1, 5) and beat 0.1.4."""
    assert version_lt("0.1.dev5+g1234567", "0.1.4") is True
    assert parse_version("0.1.dev5+g1234567") == (0, 1, 0), (
        "the local/dev suffix must not be promoted into the patch component"
    )


@pytest.mark.parametrize(
    ("a", "b"),
    [("1.2.3", "1.2.3"), ("v2.0", "2.0.0"), ("2.0", "2.0.0"), ("1.0rc1", "1.0.rc1")],
)
def test_equal_versions_order_neither_way(a: str, b: str) -> None:
    """Ties must be ties in BOTH directions.

    A one-sided check passes for a comparator that reports everything as less
    than everything else.
    """
    assert version_lt(a, b) is False
    assert version_lt(b, a) is False


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.0.1", (0, 0, 1)),
        ("1.2.3rc1", (1, 2, 3)),
        ("1.2.3.post1", (1, 2, 3)),
        ("v2.0", (2, 0, 0)),
        ("0.1.dev5+g1234567", (0, 1, 0)),
        ("1.2.3.4", (1, 2, 3)),
        ("  1.2.3  ", (1, 2, 3)),
    ],
)
def test_parse_version_returns_the_release_triple(
    text: str, expected: tuple[int, int, int],
) -> None:
    """`parse_version` is the RELEASE triple, deliberately not an ordering.

    `'1.2.3rc1'` and `'1.2.3'` both answer (1, 2, 3) to "which release is
    this". The stage rank that separates them lives in `version_lt`.
    """
    assert parse_version(text) == expected


@pytest.mark.parametrize("text", ["", "garbage", "not-a-version", "...", "v"])
def test_unparseable_input_sorts_oldest_and_never_raises(text: str) -> None:
    """⚑ The gate must reject, not crash.

    Two of the three call sites (`worker.py:2666`, `worker.py:2703`) call
    `version_lt` with nothing catching, on a value out of the server manifest.
    Raising there turns a malformed string into a dead worker. `(0, 0, 0)`
    sorts oldest, so garbage reads as "too old" and the gate rejects.
    """
    assert parse_version(text) == (0, 0, 0)
    assert version_lt(text, "0.0.1") is True
    assert version_lt("0.0.1", text) is False


def test_the_shipped_package_version_still_parses() -> None:
    """The one string that is actually in play today.

    `pyproject.toml` pins it statically, so if this ever stops being a plain
    release triple the gates' behaviour changes and this test says so.
    """
    from chess_anti_engine.version import PACKAGE_VERSION

    major, minor, patch = parse_version(PACKAGE_VERSION)
    assert (major, minor, patch) >= (0, 0, 1)
    assert version_lt(PACKAGE_VERSION, PACKAGE_VERSION) is False


def test_the_order_is_transitive_across_the_whole_ladder() -> None:
    """Sorting the ladder must reproduce it.

    Pairwise checks can pass for a comparator that is not a total order; this
    fails if any stage rank collides with another.
    """
    import functools

    ladder = [
        "0.9.9", "1.0.dev1", "1.0.dev2", "1.0a1", "1.0b1", "1.0rc1", "1.0rc2",
        "1.0", "1.0.post1", "1.0.1", "1.1.0",
    ]
    shuffled = list(reversed(ladder))

    ordered = sorted(
        shuffled, key=functools.cmp_to_key(lambda a, b: -1 if version_lt(a, b) else 1),
    )

    assert ordered == ladder
