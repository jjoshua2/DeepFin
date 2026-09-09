#!/usr/bin/env python3
"""Copy B100 policy corpus and mix SF/BT4 search_wdl (default 90% SF + 10% BT4).

Consumes completed derived-row WDL sidecars; performs no inference. All other
compressed arrays are copied and verified byte for byte. No resume or overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import bt4_derived_wdl_sidecar as wdl
from scripts import sf_policy_rewrite as sf_rewrite
from scripts import raw_wdl_adaptation as reused
from scripts.sf_policy_rewrite import ARRAYS, require

SUMMARY = "bt4_value_rewrite_summary.json"
DERIVE_SUMMARY = "derive_targets_summary.json"
POLICY_SUMMARY = "bt4_policy_mix_summary.json"
VALUE_SCHEME = "sf90-bt4-native10"
VALUE_SOURCE = "stored-sf-search-and-derived-bt4-wdl"
ALGORITHM = "normalized-wdl-arithmetic-90-10-float16-v1"


def checked_alpha(alpha: float) -> float:
    require(
        type(alpha) in (float, int) and math.isfinite(alpha) and 0 <= alpha <= 1,
        "alpha must be finite and in [0, 1]",
    )
    return float(alpha)


def value_scheme(alpha: float = 0.1) -> str:
    alpha = checked_alpha(alpha)
    return VALUE_SCHEME if alpha == 0.1 else f"sf-bt4-native-alpha={alpha!r}"


def algorithm(alpha: float = 0.1) -> str:
    return (
        ALGORITHM
        if checked_alpha(alpha) == 0.1
        else "normalized-wdl-arithmetic-float16-v1"
    )


def value_source(model_sha256: str, output: str, alpha: float = 0.1) -> str:
    """Identity consumed by the historical trainer, including the named teacher."""
    alpha = checked_alpha(alpha)
    original = f"{VALUE_SOURCE};onnx={model_sha256};output={output}"
    return original if alpha == 0.1 else f"{original};bt4_weight={alpha!r}"


def equal_json(a: Any, b: Any) -> bool:
    # Original summaries contain historical NaN diagnostics.
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 2 and values.shape[1] == 3, "WDL shape differs")
    require(
        bool(np.isfinite(values).all() and (values >= 0).all()), "invalid WDL values"
    )
    mass = values.sum(axis=1, keepdims=True)
    require(bool((np.abs(mass - 1) <= 2**-10).all()), "WDL is not probability mass")
    return values / mass


def target(sf: np.ndarray, bt4: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    require(sf.shape == bt4.shape, "WDL row counts differ")
    alpha = checked_alpha(alpha)
    return ((1 - alpha) * normalized(sf) + alpha * normalized(bt4)).astype(np.float16)


def file_map(path: Path) -> dict[str, str]:
    wdl.storage_identity(path)  # refuses symlinks and special descendants
    return {
        str(p.relative_to(path)): wdl.file_sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
    }


def inventory(root: Path, specs: list[dict[str, Any]]) -> None:
    require(
        [p.name for p in sorted(root.glob("shard_*.zarr"))]
        == [s["path"] for s in specs],
        "shard coverage differs",
    )
    require(
        not list(root.glob("*.writing")) and not (root / "failed.json").exists(),
        "partial or failed source",
    )


def rewrite(args: argparse.Namespace) -> dict[str, Any]:
    alpha = checked_alpha(args.alpha)
    adapter_path = getattr(args, 'wdl_adapter_manifest', None)
    adapter_sha = getattr(args, 'expected_wdl_adapter_manifest_sha256', None)
    require(bool(adapter_path) == bool(adapter_sha), 'adapted WDL requires manifest and SHA256')
    adapted_pin: dict[str, str] | None = None
    if adapter_path:
        if not isinstance(adapter_sha, str):
            raise ValueError('adapter manifest SHA256 must be a string')
        adapted_pin = {'path': str(Path(adapter_path).resolve()), 'sha256': adapter_sha}
        reused.pin(adapted_pin)
    wdl.set_nthreads(2)
    require(
        type(args.batch_size) is int and 0 < args.batch_size <= 4096,
        "invalid batch size",
    )
    require(
        math.isfinite(args.minimum_free_gib) and args.minimum_free_gib >= 0,
        "invalid reserve",
    )
    source, sf_root, side_root, out = [
        Path(getattr(args, k)).resolve() for k in ("source", "sf_source", "wdl", "out")
    ]
    writing = out.with_name(out.name + ".writing")
    roots = (source, sf_root, side_root)
    require(len(set(roots)) == 3, "input roots must be distinct")
    require(
        all(out != p and p not in out.parents and out not in p.parents for p in roots),
        "output overlaps input",
    )
    require(
        not os.path.lexists(out) and not os.path.lexists(writing),
        "output or partial exists",
    )
    pins = {
        source / DERIVE_SUMMARY: args.expected_source_summary_sha256,
        source / POLICY_SUMMARY: args.expected_policy_summary_sha256,
        sf_root / DERIVE_SUMMARY: args.expected_sf_summary_sha256,
    }
    for path, digest in pins.items():
        require(wdl.file_sha256(path) == digest, "source summary pin differs")
    base = json.loads((source / DERIVE_SUMMARY).read_text())
    policy = json.loads((source / POLICY_SUMMARY).read_text())
    sf, specs = wdl.source_inventory(
        argparse.Namespace(
            source=str(sf_root),
            expected_source_summary_sha256=args.expected_sf_summary_sha256,
            start_shard=0,
            max_shards=2**31,
        )
    )
    require(
        equal_json(
            {k: v for k, v in base.items() if k != "policy_target_postprocess"}, sf
        )
        and equal_json(base.get("policy_target_postprocess"), policy),
        "B100 source changed original value/history lineage",
    )
    rows = sum(s["rows"] for s in specs)
    expected_policy = {
        "kind": "global",
        "algorithm": "legal-normalized-global-arithmetic-v1",
        "alpha": 1.0,
        "bt4_temperature": 0.5,
        "rows": rows,
        "expected_shards": len(specs),
        "source_dir": str(sf_root),
        "source_derive_summary_sha256": args.expected_sf_summary_sha256,
        "mutated_arrays": ["policy_target"],
    }
    require(
        all(policy.get(k) == v for k, v in expected_policy.items()),
        "requires unchanged B100 policy recipe",
    )
    for root in roots:
        inventory(root, specs)
    states = {
        root / s["path"]: wdl.storage_identity(root / s["path"])
        for root in roots
        for s in specs
    }
    producer = {
        str(Path(p).resolve()): wdl.file_sha256(Path(p))
        for p in (
            __file__,
            wdl.__file__,
            wdl.raw.__file__,
            wdl.lc0.__file__,
            sf_rewrite.__file__,
        )
    }
    if adapted_pin is not None:
        producer[str(Path(reused.__file__).resolve())] = wdl.file_sha256(reused.__file__)
        pins[Path(adapted_pin['path'])] = adapted_pin['sha256']
    writing.mkdir(parents=True)

    def guard() -> None:
        require(
            not (writing / "STOP").exists() and not (out.parent / "STOP").exists(),
            "STOP requested",
        )
        require(
            shutil.disk_usage(writing).free >= args.minimum_free_gib * 1024**3,
            "disk reserve breached",
        )

    outputs = []
    output_states: dict[Path, str] = {}
    changed = 0
    max_error = 0.0
    try:
        for spec in specs:
            guard()
            name, n = spec["path"], spec["rows"]
            src, original, side = source / name, sf_root / name, side_root / name
            original_group = wdl.source_arrays(original, sf, n)
            group: Any = zarr.open_group(str(src), mode="r")
            require(
                set(group.array_keys())
                == set(original_group.array_keys())
                == set(ARRAYS),
                "exact 17-column corpus required",
            )
            attrs = dict(group.attrs)
            require(
                {
                    k: v
                    for k, v in attrs.items()
                    if not k.startswith("policy_target_mix_")
                }
                == dict(original_group.attrs),
                "B100 shard changed original metadata",
            )
            require(
                attrs.get("policy_target_mix_kind") == "global"
                and attrs.get("policy_target_mix_alpha") == 1.0
                and attrs.get("policy_target_mix_bt4_temperature") == 0.5,
                "B100 shard policy stamp differs",
            )
            for column in ARRAYS:
                wdl.complete_chunks(group[column])
                require(group[column].shape[0] == n, "source array row count differs")
            require(
                group["search_wdl"].shape == (n, 3)
                and group["search_wdl"].dtype == np.dtype("float16"),
                "source WDL layout differs",
            )
            side_group: Any = zarr.open_group(str(side), mode="r")
            binding = dict(side_group.attrs)["binding"]
            expected = wdl.binding(
                argparse.Namespace(
                    source=str(sf_root),
                    onnx=binding["onnx"],
                    expected_source_summary_sha256=args.expected_sf_summary_sha256,
                    expected_onnx_sha256=args.expected_onnx_sha256,
                    wdl_output=args.wdl_output,
                    wdl_output_kind="probabilities",
                ),
                spec,
                states[original],
            )
            if adapted_pin is not None:
                expected = reused.expected_binding(side, expected, adapted_pin)
            side_attrs = wdl.verify_cached(side, expected, args.batch_size)
            # Verify original SF to B100 nonpolicy compressed bytes, then immutable copy.
            src_files, sf_files = file_map(src), file_map(original)

            def nonpolicy(files):
                return {
                    k: v
                    for k, v in files.items()
                    if k != ".zattrs" and k.split("/")[0] != "policy_target"
                }

            require(
                nonpolicy(src_files) == nonpolicy(sf_files),
                "B100 nonpolicy source differs",
            )
            destination = writing / name
            shutil.copytree(src, destination)
            require(file_map(destination) == src_files, "copied bytes differ")
            dest: Any = zarr.open_group(str(destination), mode="a")
            hashes = {k: hashlib.sha256() for k in wdl.COLUMNS}
            value_hash = hashlib.sha256()
            shard_changed = 0
            for start in range(0, n, args.batch_size):
                guard()
                end = min(n, start + args.batch_size)
                batch = {
                    k: np.asarray(original_group[k][start:end]) for k in wdl.COLUMNS
                }
                for k, digest in hashes.items():
                    digest.update(batch[k].tobytes(order="C"))
                require(
                    bool(
                        (batch["has_game_id"] == 1).all()
                        and (batch["has_ply_index"] == 1).all()
                    ),
                    "missing source identity",
                )
                require(
                    np.array_equal(side_group["game_id"][start:end], batch["game_id"])
                    and np.array_equal(
                        side_group["ply_index"][start:end], batch["ply_index"]
                    ),
                    "WDL identity differs",
                )
                feed = wdl.stored_feed(batch["x"]).astype(side_attrs["input"]["dtype"])
                require(
                    np.array_equal(
                        wdl.row_digests(feed), side_group["lc0_feed_sha256"][start:end]
                    ),
                    "WDL feed identity differs",
                )
                require(
                    bool((group["has_search_wdl"][start:end] == 1).all()),
                    "missing SF value coverage",
                )
                old = np.asarray(group["search_wdl"][start:end])
                stored = target(
                    old, np.asarray(side_group["bt4_wdl_raw"][start:end]), alpha
                )
                dest["search_wdl"][start:end] = stored
                readback = np.asarray(dest["search_wdl"][start:end])
                require(np.array_equal(readback, stored), "value readback differs")
                error = float(np.abs(readback.astype(np.float64).sum(axis=1) - 1).max())
                require(error <= 2**-10, "stored value mass differs")
                max_error = max(max_error, error)
                shard_changed += int(np.any(stored != old, axis=1).sum())
                value_hash.update(readback.tobytes(order="C"))
            require(
                {k: h.hexdigest() for k, h in hashes.items()}
                == side_attrs["source_array_sha256"],
                "WDL source array proof differs",
            )
            stamp = {
                "schema": 1,
                "algorithm": algorithm(alpha),
                "sf_weight": 1 - alpha,
                "bt4_weight": alpha,
                "sidecar_attrs_sha256": wdl.file_sha256(side / ".zattrs"),
                "source_derive_summary_sha256": args.expected_source_summary_sha256,
                "search_wdl_sha256": value_hash.hexdigest(),
            }
            dest.attrs.update(
                derive_schema=2,
                derive_value_scheme=value_scheme(alpha),
                derive_value_source=value_source(
                    args.expected_onnx_sha256, args.wdl_output, alpha
                ),
                value_target_postprocess=stamp,
            )
            final_files = file_map(destination)

            def unchanged(files):
                return {
                    k: v
                    for k, v in files.items()
                    if k != ".zattrs" and k.split("/")[0] != "search_wdl"
                }

            require(
                unchanged(src_files) == unchanged(final_files),
                "nonvalue arrays changed",
            )
            outputs.append(
                {
                    "path": name,
                    "rows": n,
                    "changed_rows": shard_changed,
                    "stamp": stamp,
                    "files_manifest_sha256": hashlib.sha256(
                        json.dumps(final_files, sort_keys=True).encode()
                    ).hexdigest(),
                    "attrs_sha256": wdl.file_sha256(destination / ".zattrs"),
                    "source_storage_identity": states[src],
                    "sidecar_storage_identity": states[side],
                }
            )
            output_states[destination] = wdl.storage_identity(destination)
            changed += shard_changed
        guard()
        for path, state in states.items():
            require(wdl.storage_identity(path) == state, "source or sidecar changed")
        for root in roots:
            inventory(root, specs)
        for path, digest in pins.items():
            require(wdl.file_sha256(path) == digest, "source summary changed")
        recipe = {
            "schema": 1,
            "status": "COMPLETE",
            "kind": "bt4_value_rewrite",
            "algorithm": algorithm(alpha),
            "sf_weight": 1 - alpha,
            "bt4_weight": alpha,
            "wdl_order": "WDL",
            "wdl_pov": "side_to_move",
            "wdl_kind": "probabilities",
            "wdl_output": args.wdl_output,
            "onnx_sha256": args.expected_onnx_sha256,
            "rows": rows,
            "shards": len(specs),
            "source_dir": str(source),
            "sf_source_dir": str(sf_root),
            "wdl_dir": str(side_root),
            "source_derive_summary_sha256": args.expected_source_summary_sha256,
            "source_policy_summary_sha256": args.expected_policy_summary_sha256,
            "sf_derive_summary_sha256": args.expected_sf_summary_sha256,
            "mutated_arrays": ["search_wdl"],
            "unchanged_arrays": sorted(ARRAYS - {"search_wdl"}),
            "value_scheme": value_scheme(alpha),
            "value_source": value_source(
                args.expected_onnx_sha256, args.wdl_output, alpha
            ),
            "changed_rows": changed,
            "stored_mass_error_max": max_error,
            "producer_sha256": producer,
            "outputs": outputs,
        }
        if adapted_pin is not None:
            recipe['wdl_adaptation'] = {'profile': reused.PROFILE, 'manifest': adapted_pin,
                                        'new_teacher_evaluations': 0}
        derived = dict(base)
        derived["value_target_postprocess"] = {
            k: v for k, v in recipe.items() if k != "outputs"
        }
        # Top-level original scheme remains source history; actual value metadata is explicit.
        derived["value_scheme"] = {
            "name": value_scheme(alpha),
            "source": value_source(args.expected_onnx_sha256, args.wdl_output, alpha),
        }
        (writing / POLICY_SUMMARY).write_bytes((source / POLICY_SUMMARY).read_bytes())
        (writing / SUMMARY).write_text(
            json.dumps(recipe, indent=2, sort_keys=True) + "\n"
        )
        (writing / DERIVE_SUMMARY).write_text(
            json.dumps(derived, indent=2, sort_keys=True) + "\n"
        )
        guard()
        for path, state in states.items():
            require(
                wdl.storage_identity(path) == state,
                "source or sidecar changed before publication",
            )
        for path, digest in pins.items():
            require(
                wdl.file_sha256(path) == digest,
                "source summary changed before publication",
            )
        for path, state in output_states.items():
            require(
                wdl.storage_identity(path) == state, "output changed before publication"
            )
        writing.rename(out)
        return recipe
    except BaseException as error:
        (writing / "failed.json").write_text(
            json.dumps({"complete": False, "error": str(error)})
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "sf-source", "wdl", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-summary", "policy-summary", "sf-summary", "onnx"):
        parser.add_argument("--expected-" + name + "-sha256", required=True)
    parser.add_argument("--wdl-output", default="/output/wdl")
    parser.add_argument("--wdl-adapter-manifest", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--expected-wdl-adapter-manifest-sha256", default=argparse.SUPPRESS)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="BT4 value weight in [0, 1]; SF weight is 1-alpha",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--minimum-free-gib", type=float, default=150)
    return parser


if __name__ == "__main__":
    rewrite(build_parser().parse_args())
