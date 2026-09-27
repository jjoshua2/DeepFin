#!/usr/bin/env python3
"""CPU-only memory screen for the frozen Ceres v8 physical-call feed reader.

This is a candidate data structure and synthetic benchmark, not a v8 patch or
source admission. The frozen ledger consumer uses Mapping.get, key iteration,
and len; no model, tablebase, archive, or CUDA provider is opened here.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from hashlib import sha256
import json
import os
import random
import resource
import sys
import time
import zlib

import chess

from chess_anti_engine.encoding.ceres_tpg import encode_ceres_tpg_bytes


FEED_BYTES = 64 * 137
_KEY = tuple[int, int]


class CompressedFeedMap(Mapping[_KEY, bytes]):
    """Bounded immutable-byte store with the frozen ledger's Mapping surface."""

    def __init__(self, *, max_payload_bytes: int) -> None:
        if type(max_payload_bytes) is not int or max_payload_bytes < 1:
            raise ValueError("positive feed payload cap required")
        self._entries: dict[_KEY, tuple[bytes, bytes, bool]] = {}
        self._max_payload_bytes = max_payload_bytes
        self.payload_bytes = 0

    def add(self, key: _KEY, raw: bytes, expected_sha256: str) -> None:
        if (not isinstance(key, tuple) or len(key) != 2
                or any(type(part) is not int or part < 0 for part in key)
                or not isinstance(raw, bytes) or len(raw) != FEED_BYTES
                or key in self._entries):
            raise ValueError("duplicate or malformed physical feed")
        digest = sha256(raw).digest()
        if digest.hex() != expected_sha256:
            raise ValueError("physical feed SHA differs")
        # TPG feeds are sparse. A cheap zero census avoids wasting compression
        # CPU and expanding payload on arbitrary incompressible bytes.
        should_try = raw.count(0) * 4 >= len(raw) * 3
        encoded = zlib.compress(raw, level=1) if should_try else raw
        is_compressed = len(encoded) < len(raw)
        if not is_compressed:
            encoded = raw
        charge = len(encoded) + len(digest) + 1
        if self.payload_bytes + charge > self._max_payload_bytes:
            raise ValueError("physical feed payload budget exceeded")
        self._entries[key] = encoded, digest, is_compressed
        self.payload_bytes += charge

    def __getitem__(self, key: _KEY) -> bytes:
        encoded, digest, is_compressed = self._entries[key]
        if is_compressed:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(encoded, FEED_BYTES + 1)
            if (not decoder.eof or decoder.unconsumed_tail
                    or decoder.unused_data):
                raise ValueError("stored physical feed changed")
        else:
            raw = encoded
        if len(raw) != FEED_BYTES or sha256(raw).digest() != digest:
            raise ValueError("stored physical feed changed")
        return raw

    def __iter__(self) -> Iterator[_KEY]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def estimated_python_bytes(self) -> int:
        """Shallow entry accounting; RSS is measured separately in the screen."""
        return (sys.getsizeof(self._entries)
                + sum(sys.getsizeof(key) + sys.getsizeof(key[0])
                      + sys.getsizeof(key[1]) + sys.getsizeof(value)
                      + sys.getsizeof(value[0]) + sys.getsizeof(value[1])
                      for key, value in self._entries.items()))


def _rss_bytes() -> int:
    with open("/proc/self/statm", encoding="ascii") as stream:
        resident_pages = int(stream.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _feed_templates(scenario: str) -> list[bytes]:
    if scenario == "incompressible":
        rng = random.Random(20260927)
        return [rng.randbytes(FEED_BYTES) for _ in range(128)]
    if scenario != "sparse":
        raise ValueError("unknown screen scenario")
    rng = random.Random(20260927)
    board = chess.Board()
    feeds: list[bytes] = []
    for _ in range(128):
        feed = encode_ceres_tpg_bytes(board).tobytes(order="C")
        if len(feed) != FEED_BYTES:
            raise ValueError("Ceres feed shape differs")
        feeds.append(feed)
        if board.is_game_over():
            board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))
    return feeds


def screen(*, rows: int, scenario: str, variant: str) -> dict[str, object]:
    if type(rows) is not int or not 1 <= rows <= 8192 or rows % 32:
        raise ValueError("screen rows must be a positive multiple of 32 <=8192")
    if variant not in ("raw", "compressed"):
        raise ValueError("unknown feed store variant")
    templates = _feed_templates(scenario)
    baseline_rss = _rss_bytes()
    begun = time.perf_counter()
    if variant == "compressed":
        store: Mapping[_KEY, bytes] = CompressedFeedMap(
            max_payload_bytes=rows * (FEED_BYTES + 64))
    else:
        store = {}
    for index in range(rows):
        # A real reader materializes one distinct row buffer per key. Copy a
        # template so repeated synthetic positions cannot make raw RSS look
        # artificially cheap through shared Python bytes objects.
        raw = memoryview(templates[index % len(templates)]).tobytes()
        key = (index // 400, index % 400)
        digest = sha256(raw).hexdigest()
        if isinstance(store, CompressedFeedMap):
            store.add(key, raw, digest)
        else:
            if key in store or sha256(raw).hexdigest() != digest:
                raise ValueError("duplicate or changed physical feed")
            store[key] = raw
    build_seconds = time.perf_counter() - begun
    rss_after_build = _rss_bytes()
    begun = time.perf_counter()
    physical_digest = sha256()
    seen: set[_KEY] = set()
    for start in range(0, rows, 32):
        physical: list[bytes] = []
        for index in range(start, start + 32):
            key = (index // 400, index % 400)
            raw = store.get(key)
            expected = sha256(templates[index % len(templates)]).hexdigest()
            if (raw is None or len(raw) != FEED_BYTES or key in seen
                    or sha256(raw).hexdigest() != expected):
                raise ValueError("physical-call roster differs")
            seen.add(key)
            physical.append(raw)
        physical_digest.update(b"".join(physical))
    if seen != set(store) or len(store) != rows:
        raise ValueError("physical-call census differs")
    lookup_seconds = time.perf_counter() - begun
    if isinstance(store, CompressedFeedMap):
        payload_bytes = store.payload_bytes
        estimated_python_bytes = store.estimated_python_bytes()
    else:
        payload_bytes = sum(len(value) for value in store.values())
        estimated_python_bytes = (sys.getsizeof(store)
                                  + sum(sys.getsizeof(key) + sys.getsizeof(key[0])
                                        + sys.getsizeof(key[1]) + sys.getsizeof(value)
                                        for key, value in store.items()))
    return {
        "status": "SYNTHETIC_CPU_ONLY_NO_ADOPTION", "rows": rows,
        "scenario": scenario, "variant": variant,
        "raw_feed_bytes": rows * FEED_BYTES,
        "payload_bytes": payload_bytes,
        "estimated_python_bytes": estimated_python_bytes,
        "rss_baseline_bytes": baseline_rss,
        "rss_after_build_bytes": rss_after_build,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "build_seconds": build_seconds,
        "lookup_seconds": lookup_seconds,
        "physical_feed_sha256": physical_digest.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--scenario", choices=("sparse", "incompressible"), required=True)
    parser.add_argument("--variant", choices=("raw", "compressed"), required=True)
    args = parser.parse_args()
    print(json.dumps(screen(rows=args.rows, scenario=args.scenario,
                            variant=args.variant), sort_keys=True))


if __name__ == "__main__":
    main()
