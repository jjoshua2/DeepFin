"""Metadata-only source-selection contract shared by preflight and consumers."""

from pathlib import Path
import re
from typing import Any


def validate_selection_metadata(
    selection: Any,
    *,
    source_dir: Path,
    source_config_sha256: Any,
    source_manifest_sha256: str,
) -> None:
    """Check the exact header before any selected raw payload is opened.

    Inventory membership, row claims and raw payload hashes remain the consuming
    deriver's responsibility. This does not qualify a selected corpus by itself.
    """
    required = {
        "schema", "source_dir", "source_config_sha256",
        "source_manifest_sha256", "shards",
    }
    if (not isinstance(selection, dict) or set(selection) != required
            or selection["schema"] != 1):
        raise ValueError("invalid source-shards schema")
    if (selection["source_dir"] != str(source_dir.resolve())
            or selection["source_config_sha256"] != source_config_sha256
            or selection["source_manifest_sha256"] != source_manifest_sha256):
        raise ValueError("source-shards source binding mismatch")
    entries = selection["shards"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("source-shards must contain a nonempty shard list")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"source_shard", "rows", "source_sha256"}:
            raise ValueError("invalid source-shards entry")
        name = entry["source_shard"]
        if (not isinstance(name, str) or Path(name).name != name
                or name in (".", "..") or name in seen):
            raise ValueError("duplicate or invalid selected shard name")
        seen.add(name)
        if type(entry["rows"]) is not int or entry["rows"] <= 0:
            raise ValueError(f"selected shard row claim mismatch: {name}")
        if (not isinstance(entry["source_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", entry["source_sha256"]) is None):
            raise ValueError("invalid selected raw SHA256")
