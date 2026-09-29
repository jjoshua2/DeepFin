"""Preserved-parent game identity contract for mixed-source training packs.

The exact-epoch loader groups by (resolved shard parent, local game_id), while
schema-2 provenance requires the stored game_id to remain source-local. Give
each immutable (run manifest, namespace, opening stratum) one physical shard
parent. Stage flat symlinks to those shards for the trainer; the resolved
parents remain distinct. This module does not admit source or target bytes.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MAX_INT64 = (1 << 63) - 1
MIN_INT64 = -(1 << 63)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _validate_source_fields(manifest: object, namespace: object, stratum: object) -> None:
    """Validate untyped runtime inputs without trusting constructor annotations."""
    if not isinstance(manifest, str) or _SHA256.fullmatch(manifest) is None:
        raise ValueError("run_manifest_sha256 must be lowercase SHA-256")
    if (not isinstance(namespace, str) or not isinstance(stratum, str)
            or not namespace or not stratum):
        raise ValueError("source namespace and opening stratum must be nonempty strings")


@dataclass(frozen=True, order=True)
class PackSourceIdentity:
    run_manifest_sha256: str
    source_namespace: str
    opening_stratum: str

    def __post_init__(self) -> None:
        _validate_source_fields(
            self.run_manifest_sha256, self.source_namespace, self.opening_stratum,
        )

    def as_list(self) -> list[str]:
        return [self.run_manifest_sha256, self.source_namespace,
                self.opening_stratum]


@dataclass(frozen=True, order=True)
class SourceGameIdentity:
    run_manifest_sha256: str
    source_namespace: str
    opening_stratum: str
    game_id: int

    @property
    def source(self) -> PackSourceIdentity:
        return PackSourceIdentity(
            self.run_manifest_sha256, self.source_namespace,
            self.opening_stratum,
        )

    def __post_init__(self) -> None:
        self.source  # validate the three source fields
        raw_id: object = self.game_id
        if isinstance(raw_id, (bool, np.bool_)) or (
            type(raw_id) is not int and not isinstance(raw_id, np.integer)
        ) or not MIN_INT64 <= int(raw_id) <= MAX_INT64:
            raise ValueError("source game_id must be signed int64")


class PreservedParentPack:
    """Sorted source ordinals give distinct parents without hash truncation."""

    def __init__(
        self,
        sources: Sequence[PackSourceIdentity],
        *,
        shard_format: str = "directory",
    ) -> None:
        if shard_format not in ("directory", "zip_stored"):
            raise ValueError("shard format must be directory or zip_stored")
        self.shard_format = shard_format
        self.sources = tuple(sorted(set(sources)))
        self._ordinals = {source: index for index, source in enumerate(self.sources)}

    def source_parent(self, pack_root: Path, source: PackSourceIdentity) -> Path:
        try:
            index = self._ordinals[source]
        except KeyError as exc:
            raise ValueError("source absent from sealed pack roster") from exc
        return pack_root / "sources" / f"source_{index:08d}"

    def canonical_bytes(self) -> bytes:
        return (json.dumps(
            ["mixed_source_preserved_parent_pack_v1", self.shard_format,
             [source.as_list() for source in self.sources]],
            ensure_ascii=True, separators=(",", ":"),
        ) + "\n").encode("ascii")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def manifest(self) -> dict[str, str | int]:
        return {
            "schema": "mixed_source_preserved_parent_pack_v1",
            "source_count": len(self.sources),
            "roster_sha256": self.sha256,
            "shard_format": self.shard_format,
            "source_parent_rule": "sources/source_{sorted_source_ordinal:08d}",
            "stored_game_id_rule": "unchanged_source_local_signed_int64",
        }

    def validate_shard_rows(
        self,
        source: PackSourceIdentity,
        arrays: Mapping[str, Any],
        row_identities: Sequence[SourceGameIdentity],
    ) -> None:
        """Check a packer's row provenance before it writes a physical shard."""
        self.source_parent(Path("."), source)  # source must be in the roster
        ids = np.asarray(arrays["game_id"])
        present = np.asarray(arrays["has_game_id"])
        if ids.ndim != 1 or ids.dtype != np.dtype("int64") or (
            present.shape != ids.shape or present.dtype != np.dtype("uint8")
            or len(row_identities) != ids.size
        ):
            raise ValueError("one int64 game ID and uint8 presence flag are required per row")
        if not np.all(present == 1):
            raise ValueError("all packed game-ID presence flags must equal 1")
        for index, identity in enumerate(row_identities):
            if identity.source != source or int(ids[index]) != int(identity.game_id):
                raise ValueError(f"source game identity mismatch at row {index}")

    def stage_shards(
        self,
        pack_root: Path,
        staging: Path,
        shards: Sequence[tuple[PackSourceIdentity, Path]],
    ) -> None:
        """Stage a flat exact-epoch view while preserving resolved parents."""
        ordered = sorted(shards, key=lambda item: (item[0], item[1].name))
        if not ordered or len({path for _, path in ordered}) != len(ordered):
            raise ValueError("stage requires distinct physical shards")
        try:
            resolved_root = pack_root.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError("pack root is missing") from exc
        if (resolved_root != pack_root.absolute()
                or (pack_root / "sources").is_symlink()):
            raise ValueError("pack root and parent chain must be canonical nonsymlinks")
        suffix = ".zarr.zip" if self.shard_format == "zip_stored" else ".zarr"
        owner_for_parent: dict[Path, PackSourceIdentity] = {}
        resolved_shards: set[Path] = set()
        physical_shards: set[tuple[int, int]] = set()
        for source, path in ordered:
            source_dir = self.source_parent(pack_root, source)
            try:
                expected = source_dir.resolve(strict=True)
                actual = path.resolve(strict=True)
            except FileNotFoundError as exc:
                raise ValueError("preserved source parent or shard is missing") from exc
            if source_dir.is_symlink() or path.is_symlink():
                raise ValueError("physical source parent and shard must not be symlinks")
            if actual.parent != expected or not path.name.endswith(suffix):
                raise ValueError("shard is outside its preserved source parent")
            if self.shard_format == "zip_stored" and not actual.is_file():
                raise ValueError("packed Zarr shard must be a regular file")
            if self.shard_format == "directory" and not actual.is_dir():
                raise ValueError("ordinary Zarr shard must be a directory")
            stamp = actual.stat()
            physical_identity = (stamp.st_dev, stamp.st_ino)
            previous = owner_for_parent.setdefault(expected, source)
            if (previous != source or actual in resolved_shards
                    or physical_identity in physical_shards):
                raise ValueError("source parents and physical shards must be distinct")
            resolved_shards.add(actual)
            physical_shards.add(physical_identity)
        resolved_staging = staging.resolve()
        if any(resolved_staging == shard or shard in resolved_staging.parents
               for shard in resolved_shards):
            raise ValueError("staging must be outside physical source shards")
        staging.mkdir(parents=True, exist_ok=False)
        try:
            for index, (_, path) in enumerate(ordered):
                (staging / f"shard_{index:06d}{suffix}").symlink_to(
                    path.resolve(strict=True)
                )
        except BaseException:
            shutil.rmtree(staging)
            raise
