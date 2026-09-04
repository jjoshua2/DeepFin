"""Immutable production Syzygy corpus layout approved for decision-grade banks.

The layout digest is SHA-256 over compact ASCII JSON with no whitespace.  Rows
are sorted by filename and encoded as ``[[filename,size],...]`` where ``size``
is the decimal byte length.  Host paths, inode numbers, and timestamps are
deliberately absent: those belong to each run's mutable filesystem-identity
inventory, while this module pins the portable corpus layout.

The production pair was inventoried on 2026-09-04.  Its official
``3-4-5-6.md5`` catalog contains 1,020 unique filename/checksum entries: all
510 WDL names and all 510 DTZ names.  The split production directories contain
overlapping 6-man WDL names, so their combined 1,385 files reduce to exactly
that 1,020-name logical catalog.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass


APPROVED_SYZYGY_LAYOUT_SCHEMA = "deepfin.approved_syzygy_layout.v1"
APPROVED_SYZYGY_FILENAME_SIZE_ENCODING = (
    "sorted_compact_ascii_json_array_of_filename_and_decimal_size"
)
APPROVED_SYZYGY_CHECKSUM_CATALOG_NAME = "3-4-5-6.md5"
APPROVED_SYZYGY_CHECKSUM_CATALOG_SIZE = 47_570
APPROVED_SYZYGY_CHECKSUM_CATALOG_RAW_SHA256 = (
    "e5039f7d0a63bb8607cc2342357353f162f40ae601853f116e763809003683ab"
)
APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRY_COUNT = 1020
APPROVED_SYZYGY_CHECKSUM_CATALOG_WDL_COUNT = 510
APPROVED_SYZYGY_CHECKSUM_CATALOG_DTZ_COUNT = 510
APPROVED_SYZYGY_CHECKSUM_CATALOG_ENTRIES_SHA256 = (
    "4e2a2577cc0bdae8025f71dc64b69bf0caec422ee220f2a2995f0ca122ad0d62"
)


@dataclass(frozen=True)
class ApprovedSyzygyComponent:
    """Portable identity of one component in the production path order."""

    directory_name: str
    rtbw_count: int
    rtbz_count: int
    file_count: int
    total_bytes: int
    filename_size_sha256: str


APPROVED_SYZYGY_COMPONENTS = (
    ApprovedSyzygyComponent(
        directory_name="syzygy_3-4-5",
        rtbw_count=510,
        rtbz_count=145,
        file_count=655,
        total_bytes=73_818_025_392,
        filename_size_sha256=(
            "796607668b96e5128e5493cb6fcbb8c0b1155ffd1f686a6e7e12b4a4fea61b78"
        ),
    ),
    ApprovedSyzygyComponent(
        directory_name="syzygy_6",
        rtbw_count=365,
        rtbz_count=365,
        file_count=730,
        total_bytes=160_225_616_032,
        filename_size_sha256=(
            "32b67df63f6005216c29778f3176d242a1f531aa1f225e4916bbc5a1ccfa6618"
        ),
    ),
)


def filename_size_document(rows: Iterable[tuple[str, int]]) -> bytes:
    """Return the canonical portable filename/size encoding."""
    normalized = sorted((str(name), int(size)) for name, size in rows)
    return json.dumps(
        normalized, ensure_ascii=True, separators=(",", ":"),
    ).encode("ascii")


def filename_size_sha256(rows: Iterable[tuple[str, int]]) -> str:
    """Hash the canonical portable filename/size encoding."""
    return hashlib.sha256(filename_size_document(rows)).hexdigest()


def checksum_catalog_entries_sha256(rows: Iterable[tuple[str, str]]) -> str:
    """Hash sorted ``[filename,lowercase-md5]`` catalog rows as compact JSON."""
    normalized = sorted((str(name), str(digest).lower()) for name, digest in rows)
    document = json.dumps(
        normalized, ensure_ascii=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(document).hexdigest()
