"""Contract tests for the bounded CPU-only Ceres feed-store candidate."""
from __future__ import annotations

from hashlib import sha256
import random
import zlib

import pytest

from scripts.ceres_v8_feed_store_screen import CompressedFeedMap, FEED_BYTES


def test_mapping_matches_physical_consumer_operations() -> None:
    feeds = [bytes([index]) + b"\0" * (FEED_BYTES - 1) for index in range(3)]
    mapping = CompressedFeedMap(max_payload_bytes=1000)
    for index, feed in enumerate(feeds):
        mapping.add((7, index), feed, sha256(feed).hexdigest())
    assert len(mapping) == 3
    assert set(mapping) == {(7, 0), (7, 1), (7, 2)}
    assert [mapping.get((7, index)) for index in range(3)] == feeds
    assert mapping.get((8, 0)) is None
    assert sha256(b"".join(mapping.get((7, index), b"") for index in range(3))).digest() == sha256(
        b"".join(feeds)).digest()


def test_duplicate_changed_and_capacity_refusals() -> None:
    feed = b"\0" * FEED_BYTES
    digest = sha256(feed).hexdigest()
    mapping = CompressedFeedMap(max_payload_bytes=100)
    mapping.add((0, 0), feed, digest)
    with pytest.raises(ValueError, match="duplicate"):
        mapping.add((0, 0), feed, digest)
    with pytest.raises(ValueError, match="SHA"):
        mapping.add((0, 1), feed, "0" * 64)
    with pytest.raises(ValueError, match="budget"):
        mapping.add((0, 1), feed, digest)
    assert len(mapping) == 1


def test_corrupt_compressed_value_is_rejected() -> None:
    feed = b"\0" * FEED_BYTES
    mapping = CompressedFeedMap(max_payload_bytes=1000)
    mapping.add((0, 0), feed, sha256(feed).hexdigest())
    mapping._entries[(0, 0)] = (zlib.compress(b"x" * FEED_BYTES, 1), sha256(feed).digest(), True)
    with pytest.raises(ValueError, match="changed"):
        _ = mapping[(0, 0)]


def test_incompressible_feed_uses_exact_raw_fallback() -> None:
    feed = random.Random(20260927).randbytes(FEED_BYTES)
    mapping = CompressedFeedMap(max_payload_bytes=FEED_BYTES + 40)
    mapping.add((0, 0), feed, sha256(feed).hexdigest())
    assert mapping._entries[(0, 0)][2] is False
    assert mapping[(0, 0)] == feed
    assert mapping.payload_bytes == FEED_BYTES + 33
