"""Typed raw-to-derived WDL reuse; never impersonates direct teacher inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from scripts import bt4_raw_corpus_sidecar as raw
from chess_anti_engine.encoding import lc0
from scripts.sf_policy_rewrite import require
from scripts.bt4_policy_dump import file_sha256

COLUMNS = ("x", "game_id", "ply_index", "has_game_id", "has_ply_index")
HISTORY = "lc0_root_legacy_meta"
SUMMARY = "derive_targets_summary.json"

PROFILE = "verified-raw-wdl-derived-adaptation-v1"


def pin(item: dict[str, str]) -> Path:
    require(set(item) == {"path", "sha256"}, "expected explicit path/SHA256 pin")
    path = Path(item["path"])
    require(
        path.is_absolute() and path == path.resolve(), "pinned paths must be canonical"
    )
    require(file_sha256(path) == item["sha256"], "changed adaptation pin")
    return path


def contract(manifest: dict[str, Any]) -> dict[str, str]:
    value = manifest.get("wdl")
    require(
        isinstance(value, dict) and set(value) == {"output", "kind", "dtype"},
        "explicit native WDL contract required",
    )
    assert isinstance(value, dict)
    require(
        value["kind"] == "probabilities"
        and value["dtype"] in ("float16", "float32", "float64")
        and isinstance(value["output"], str)
        and bool(value["output"].strip())
        and value["output"] != manifest["teacher"]["policy_output"],
        "requires a distinct native WDL probability output",
    )
    return value


def canonical_hashes(x: np.ndarray) -> np.ndarray:
    # Canonical float32 LC0 tensors, not a claim about historical ORT feed dtype.
    require(
        x.dtype == np.dtype("float16") and x.ndim == 4 and x.shape[1:] == (175, 8, 8),
        "expected stored float16 history",
    )
    require(bool(np.isfinite(x).all()), "nonfinite stored history")
    feed = lc0.x_to_lc0_planes(x, input_history_encoding=HISTORY)
    return np.asarray(
        [list(hashlib.sha256(row.tobytes(order="C")).digest()) for row in feed],
        dtype=np.uint8,
    )


def binding(
    manifest: dict[str, Any],
    manifest_pin: dict[str, str],
    source: Path,
    rows: int,
    state: str,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "profile": PROFILE,
        "source_dir": str(source.parent),
        "source_shard": source.name,
        "source_summary_sha256": manifest["derived_summary"]["sha256"],
        "source_storage_identity": state,
        "rows": rows,
        "onnx": manifest["teacher"]["onnx"]["path"],
        "onnx_sha256": manifest["teacher"]["onnx"]["sha256"],
        "requested_wdl": {k: contract(manifest)[k] for k in ("output", "kind")},
        "adapter_manifest": manifest_pin,
        "history_lineage": "Verified original raw history joined to derived physical rows; exact canonical LC0 feed equality; no new inference.",
        "producer": {
            str(
                Path(p).resolve().relative_to(Path(__file__).resolve().parents[1])
            ): file_sha256(p)
            for p in (
                __file__,
                Path(__file__).with_name("adapt_raw_bt4_sidecars.py"),
                raw.__file__,
                lc0.__file__,
            )
        },
    }


def write_shard(
    path: Path,
    *,
    source: Path,
    group: Any,
    manifest: dict[str, Any],
    manifest_pin: dict[str, str],
    state: str,
    values: np.ndarray,
    feeds: np.ndarray,
    row_provenance_sha256: str,
) -> None:
    n = len(values)
    wanted = contract(manifest)
    raw.validate_wdl_values(values, n, wanted)
    arrays = {k: np.asarray(group[k][:]) for k in COLUMNS}
    require(
        all(np.all(arrays[k] == 1) for k in ("has_game_id", "has_ply_index")),
        "adapted WDL requires complete row identities",
    )
    require(
        all(
            arrays[k].dtype.kind in "iu" and np.all(arrays[k] >= 0)
            for k in ("game_id", "ply_index")
        ),
        "invalid adapted WDL row identities",
    )
    require(
        np.array_equal(feeds, canonical_hashes(arrays["x"])),
        "derived feed changed during adaptation",
    )
    payload = {
        "bt4_wdl_raw": values,
        "row_index": np.arange(n, dtype=np.uint64),
        "game_id": arrays["game_id"],
        "ply_index": arrays["ply_index"],
        "lc0_feed_sha256": feeds,
    }
    path.mkdir(parents=True)
    out: Any = zarr.open_group(str(path), mode="w")
    for name, data in payload.items():
        out.create_dataset(
            name,
            data=data,
            chunks=(min(n, 512), *data.shape[1:]),
            compressor=raw._COMPRESSOR,
        )
        require(np.array_equal(out[name][:], data), "adapted WDL readback differs")
    hashes = {k: raw.sha_array(v) for k, v in payload.items()}
    out.attrs.update(
        {
            "complete": True,
            "binding": binding(manifest, manifest_pin, source, n, state),
            "source_array_sha256": {k: raw.sha_array(v) for k, v in arrays.items()},
            "array_sha256": hashes,
            "row_provenance_sha256": row_provenance_sha256,
            "input": {
                "name": "canonical_lc0_planes",
                "dtype": "float32",
                "shape": [112, 8, 8],
                "encoding": HISTORY,
                "kind": "canonical_reconstructed_feed",
            },
            "providers": manifest["teacher"]["providers"],
            "wdl": {
                "schema": 1,
                **wanted,
                "rows": n,
                "order": ["win", "draw", "loss"],
                "pov": "side_to_move",
                "semantic_basis": "explicit_named_output_contract",
                "sha256": hashes["bt4_wdl_raw"],
            },
        }
    )


def expected_binding(
    path: Path, expected: dict[str, Any], manifest_pin: dict[str, str]
) -> dict[str, Any]:
    """Validate explicit adaptation metadata before the normal saved-array verifier."""
    manifest = json.loads(pin(manifest_pin).read_text())
    wanted = contract(manifest)
    source = Path(expected["source_dir"]) / expected["source_shard"]
    require(
        manifest["derived_summary"]
        == {
            "path": str(source.parent / SUMMARY),
            "sha256": expected["source_summary_sha256"],
        },
        "adapter source summary differs",
    )
    require(
        manifest["teacher"]["onnx"]
        == {"path": expected["onnx"], "sha256": expected["onnx_sha256"]}
        and {k: wanted[k] for k in ("output", "kind")} == expected["requested_wdl"],
        "adapter teacher/head differs",
    )
    # The parent policy+value publication is atomic; incomplete .writing roots have no summary.
    summary = json.loads(
        (path.parent.parent / "bt4_policy_sidecar_summary.json").read_text()
    )
    require(
        path.parent.name == "wdl"
        and not path.parent.parent.name.endswith(".writing")
        and summary["adapter"]["manifest"] == manifest_pin
        and summary["adapted_wdl"]
        == {
            "profile": PROFILE,
            "path": "wdl",
            "rows": summary["rows"],
            "shards": summary["sidecar_shards"],
            "contract": wanted,
            "new_teacher_evaluations": 0,
        },
        "incomplete adapted WDL publication",
    )
    bound = binding(
        manifest,
        manifest_pin,
        source,
        expected["rows"],
        expected["source_storage_identity"],
    )
    attrs = dict(zarr.open_group(str(path), mode="r").attrs)
    raw.validate_wdl_metadata(
        zarr.open_group(str(path), mode="r"),
        attrs,
        expected["rows"],
        wanted,
        required=True,
    )
    require(
        attrs["input"]
        == {
            "name": "canonical_lc0_planes",
            "dtype": "float32",
            "shape": [112, 8, 8],
            "encoding": HISTORY,
            "kind": "canonical_reconstructed_feed",
        }
        and attrs["providers"] == manifest["teacher"]["providers"],
        "adapted input/provider provenance differs",
    )
    entries = {s["path"]: s for s in summary["adapter"]["written_shards"]}
    require(
        attrs["row_provenance_sha256"] == entries[source.name]["row_provenance_sha256"],
        "adapted row provenance differs",
    )
    for item in [
        manifest["derived_summary"],
        *[i[k] for i in manifest["sources"] for k in ("manifest", "receipts")],
    ]:
        pin(item)
    return bound
