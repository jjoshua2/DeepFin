"""Bounded, non-pickle construction artifacts for fresh exact-epoch samplers.

The caller pins the manifest digest. Mutable source bytes are still verified by
the constructor; this artifact grants no authority to omit content validation.
"""

from __future__ import annotations

import dataclasses
import ctypes
import hashlib
import inspect
import json
import math
import os
import platform
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

SCHEMA = "game_epoch_construction_v1"
MAX_BYTES = 256 * 1024**2


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def encode(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or value.nbytes > MAX_BYTES:
            raise ValueError("unsupported construction array")
        array = np.ascontiguousarray(value)
        return {
            "tag": "array",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "hex": array.tobytes().hex(),
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "tag": "record",
            "name": type(value).__name__,
            "fields": {
                f.name: encode(getattr(value, f.name))
                for f in dataclasses.fields(value)
            },
        }
    if isinstance(value, Path):
        return {"tag": "path", "value": str(value)}
    if isinstance(value, (tuple, list)):
        return {
            "tag": "tuple" if isinstance(value, tuple) else "list",
            "items": [encode(item) for item in value],
        }
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {
            "tag": "dict",
            "items": {key: encode(item) for key, item in value.items()},
        }
    if value is None or type(value) in (bool, int, str):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return float(value)
    raise ValueError("unsupported construction value")


def decode(value: Any, classes: dict[str, type]) -> Any:
    if not isinstance(value, dict):
        if (
            value is None
            or type(value) in (bool, int, str)
            or (type(value) is float and math.isfinite(value))
        ):
            return value
        raise ValueError("untagged construction value")
    tag = value.get("tag")
    if tag == "array" and set(value) == {"tag", "dtype", "shape", "hex"}:
        shape = value["shape"]
        if (
            type(shape) is not list
            or len(shape) > 4
            or any(type(n) is not int or n < 0 for n in shape)
        ):
            raise ValueError("construction array shape")
        dtype = np.dtype(value["dtype"])
        size = math.prod(shape) * dtype.itemsize
        if (
            dtype.kind not in "biuf"
            or size > MAX_BYTES
            or type(value["hex"]) is not str
            or len(value["hex"]) != 2 * size
        ):
            raise ValueError("construction array bound/dtype")
        return (
            np.frombuffer(bytes.fromhex(value["hex"]), dtype=dtype)
            .reshape(shape)
            .copy()
        )
    if tag == "record" and set(value) == {"tag", "name", "fields"}:
        if value["name"] not in classes:
            raise ValueError("construction record type")
        cls = classes[value["name"]]
        if type(value["fields"]) is not dict or set(value["fields"]) != {
            f.name for f in dataclasses.fields(cls)
        }:
            raise ValueError("incomplete construction record")
        return cls(
            **{key: decode(item, classes) for key, item in value["fields"].items()}
        )
    if tag == "path" and set(value) == {"tag", "value"} and type(value["value"]) is str:
        return Path(value["value"])
    if (
        tag in ("tuple", "list")
        and set(value) == {"tag", "items"}
        and type(value["items"]) is list
    ):
        items = [decode(item, classes) for item in value["items"]]
        return tuple(items) if tag == "tuple" else items
    if (
        tag == "dict"
        and set(value) == {"tag", "items"}
        and type(value["items"]) is dict
    ):
        return {key: decode(item, classes) for key, item in value["items"].items()}
    raise ValueError("unknown construction schema")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate construction JSON key")
        result[key] = value
    return result


def _read(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("construction artifact must be a regular file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("construction artifact bound")
    return raw


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(temporary: Path, destination: Path) -> None:
    # Linux RENAME_NOREPLACE gives the no-overwrite guarantee atomically,
    # including a competing empty directory. Unsupported hosts fail closed.
    library = ctypes.CDLL(None, use_errno=True)
    rename = getattr(library, "renameat2", None)
    if rename is None:
        raise OSError("construction publication requires renameat2")
    result = rename(-100, os.fsencode(temporary), -100, os.fsencode(destination), 1)
    if result:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))


def publish(path: Path, state: dict[str, Any], bindings: dict[str, Any]) -> str:
    """Publish a new directory; failed temporary directories remain recoverable."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise ValueError("construction cache destination already exists")
    raw = canonical(encode(state))
    if len(raw) > MAX_BYTES:
        raise ValueError("construction artifact bound")
    manifest = canonical(
        {
            "schema": SCHEMA,
            "bindings": bindings,
            "state_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )
    if len(manifest) > MAX_BYTES:
        raise ValueError("construction manifest bound")
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.pending-", dir=path.parent))
    # Manifest is the final commit marker inside the privately built directory.
    for name, content in (("state.json", raw), ("manifest.json", manifest)):
        with (temporary / name).open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    _fsync_directory(temporary)
    _publish_directory(temporary, path)
    _fsync_directory(path.parent)
    return hashlib.sha256(manifest).hexdigest()


def load(
    path: Path, digest: str, bindings: dict[str, Any], classes: dict[str, type]
) -> dict[str, Any]:
    path = Path(path)
    if (
        path.is_symlink()
        or not path.is_dir()
        or {p.name for p in path.iterdir()} != {"state.json", "manifest.json"}
    ):
        raise ValueError("incomplete construction publication")
    manifest = _read(path / "manifest.json")
    if hashlib.sha256(manifest).hexdigest() != digest:
        raise ValueError("construction manifest digest")
    document = json.loads(manifest, object_pairs_hook=_unique_object)
    if (
        type(document) is not dict
        or set(document) != {"schema", "bindings", "state_sha256"}
        or document["schema"] != SCHEMA
        or document["bindings"] != bindings
    ):
        raise ValueError("construction binding mismatch")
    raw = _read(path / "state.json")
    if hashlib.sha256(raw).hexdigest() != document["state_sha256"]:
        raise ValueError("construction state digest")
    result = decode(json.loads(raw, object_pairs_hook=_unique_object), classes)
    if type(result) is not dict or set(result) != {
        "census",
        "plan",
        "shuffled_records",
        "rng",
    }:
        raise ValueError("construction state fields")
    return result


def source_binding(counter: Any) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    names = (
        "replay/startup_cache.py",
        "replay/game_epoch.py",
        "replay/shard.py",
        "replay/target_overlay.py",
        "replay/packed_zarr.py",
        "replay/disk_buffer.py",
        "encoding/lc0.py",
    )
    pins = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }
    counter_identity = None
    if counter is not None:
        source = inspect.getsourcefile(counter)
        if source is None:
            raise ValueError("construction counter requires inspectable source")
        counter_identity = {
            "name": counter.__qualname__,
            "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        }
    return {
        "files": pins,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "rng": "PCG64/SeedSequence",
        "counter": counter_identity,
    }


def validate(
    state: dict[str, Any], paths: list[Path], config: dict[str, Any], ge: Any
) -> None:
    records, shuffled, plan = state["census"], state["shuffled_records"], state["plan"]
    if (
        type(records) is not list
        or not records
        or type(shuffled) is not list
        or type(plan) is not ge.GameEpochPlan
    ):
        raise ValueError("construction record schema")
    roster = {path.resolve(strict=True) for path in paths}
    if len(roster) != len(paths) or len({r.path for r in records}) != len(records):
        raise ValueError("construction duplicate source")
    key_for_game: dict[tuple[str, int], int] = {}
    for record in records:
        if (
            type(record) is not ge._ShardGames
            or record.path not in roster
            or record.path != record.path.resolve(strict=True)
        ):
            raise ValueError("construction source namespace")
        if (
            type(record.rows) is not int
            or record.rows <= 0
            or record.row_bytes <= 0
            or record.scalar_bytes < 0
        ):
            raise ValueError("construction record counts")
        for array in (record.game_ids, record.game_keys, record.game_counts):
            if (
                not isinstance(array, np.ndarray)
                or array.dtype.str != "<i8"
                or array.ndim != 1
            ):
                raise ValueError("construction game array")
        if (
            not (
                record.game_ids.shape
                == record.game_keys.shape
                == record.game_counts.shape
            )
            or np.any(record.game_counts <= 0)
            or sum(map(int, record.game_counts)) != record.rows
            or np.any(np.diff(record.game_ids) <= 0)
        ):
            raise ValueError("construction game coverage")
        expected = []
        for raw in record.game_ids.tolist():
            identity = (str(record.path.parent), int(raw))
            expected.append(key_for_game.setdefault(identity, len(key_for_game)))
        if not np.array_equal(record.game_keys, np.asarray(expected, dtype=np.int64)):
            raise ValueError("construction game namespace keys")
        if sum(width for _, width in record.row_field_bytes) != record.row_bytes or any(
            type(width) is not int or width <= 0 for _, width in record.row_field_bytes
        ):
            raise ValueError("construction byte accounting")
        if any(
            type(weight) is not float or not math.isfinite(weight) or weight < 0
            for _, weight in record.objective_mask_weights
        ):
            raise ValueError("construction objective weights")

    # The planner can move a frontier shard after its initial seeded shuffle.
    # Preserve that complete returned order, pinned by the manifest digest.
    def by_path(record: Any) -> str:
        return str(record.path)

    if canonical(encode(sorted(shuffled, key=by_path))) != canonical(
        encode(sorted(records, key=by_path))
    ):
        raise ValueError("construction shuffled record membership")
    for name in (
        "seed",
        "batch_size",
        "load_workers",
        "max_working_set_bytes",
        "mirror_augmentation",
        "input_history_encoding",
        "history_rep_fix",
    ):
        if getattr(plan, name) != config[name]:
            raise ValueError("construction plan config")
    if (
        plan.rows != sum(r.rows for r in records)
        or plan.shard_count != len(records)
        or plan.source_count != len({r.path.parent for r in records})
        or plan.game_count != len(key_for_game)
        or plan.corpus_sha256 != ge._corpus_sha256(records)
    ):
        raise ValueError("construction aggregate census")
    if {record.policy_size for record in records} != {plan.policy_size}:
        raise ValueError("construction policy width")
    balanced = ge._balanced_batch_rows(
        rows=plan.rows,
        batch_size=plan.batch_size,
        max_game_rows=max(ge._game_totals(records).values()),
    )
    if not np.array_equal(plan.batch_rows, balanced):
        raise ValueError("construction balanced batch schedule")
    names = tuple(name for name, _ in records[0].objective_mask_weights)
    if any(
        tuple(name for name, _ in r.objective_mask_weights) != names for r in records
    ):
        raise ValueError("construction objective keys")
    weights = tuple(
        (name, math.fsum(dict(r.objective_mask_weights)[name] for r in shuffled))
        for name in names
    )
    if weights != plan.objective_mask_weights:
        raise ValueError("construction objective census")
    for name, dtype in (
        ("load_counts", "<i4"),
        ("batch_rows", "<i4"),
        ("resident_bytes_after_batch", "<i8"),
    ):
        array = getattr(plan, name)
        if (
            not isinstance(array, np.ndarray)
            or array.dtype.str != dtype
            or array.shape != (plan.batches,)
            or np.any(array < 0)
        ):
            raise ValueError("construction schedule array")
    if (
        plan.batches <= 0
        or np.any(plan.batch_rows <= 0)
        or np.any(plan.batch_rows > plan.batch_size)
        or sum(map(int, plan.batch_rows)) != plan.rows
        or sum(map(int, plan.load_counts)) != len(records)
        or plan.min_batch_rows != int(plan.batch_rows.min())
        or plan.full_batches
        != int(np.count_nonzero(plan.batch_rows == plan.batch_size))
        or plan.ragged_batches != plan.batches - plan.full_batches
        or plan.resident_bytes_after_batch[-1] != 0
        or plan.peak_working_set_bytes > plan.max_working_set_bytes
    ):
        raise ValueError("construction schedule coverage")
    reserve = (
        ge.HOST_OVERLAP_BATCH_COPIES
        * ge._batch_bytes_for_records(take=plan.batch_size, records=records)
        if config["host_batch_overlap"]
        else 0
    )
    if plan.host_overlap_reserve_bytes != reserve:
        raise ValueError("construction overlap reserve")
    expected_rng = {
        name: ge._seeded_rng(config["seed"], stream).bit_generator.state
        for name, stream in (("_choice_rng", 1), ("_row_rng", 2), ("rng", 3))
    }
    if state["rng"] != expected_rng:
        raise ValueError("construction fresh RNG")
