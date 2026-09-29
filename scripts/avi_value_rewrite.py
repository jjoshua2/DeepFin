#!/usr/bin/env python3
"""Rewrite only the main search-WDL target from a complete AVI successor bank.

Two treatments share one frozen collection:

* ``root``   — SF-anchored ordinary root self-distillation control.
* ``backup`` — SF-anchored all-legal one-ply successor backup.

The source corpus is B100 (policy changed, original SF values retained). The AVI bank
is collected against the corresponding original SF corpus. This producer verifies that
B100 differs from that original corpus only in policy storage, then mutates only
``search_wdl`` in a fresh output namespace.
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

from chess_anti_engine.mcts.one_ply import blend_wdl_targets
from scripts import avi_successor_sidecar as avi
from scripts import bt4_derived_wdl_sidecar as derived
from scripts import bt4_value_rewrite as bt4_value
from scripts.sf_policy_rewrite import ARRAYS, require

SUMMARY = "avi_value_rewrite_summary.json"
DERIVE_SUMMARY = "derive_targets_summary.json"
POLICY_SUMMARY = "bt4_policy_mix_summary.json"
SCHEMA = 1


def checked_alpha(alpha: float) -> float:
    require(
        math.isfinite(float(alpha)) and 0.0 <= float(alpha) <= 1.0,
        "alpha must be finite and in [0, 1]",
    )
    return float(alpha)


def _checked_sha256(value: str, *, label: str) -> str:
    require(
        len(value) == 64
        and all(char in "0123456789abcdef" for char in value.lower()),
        f"{label} must be a 64-character SHA256",
    )
    return value.lower()


def value_scheme(mode: str, alpha: float) -> str:
    require(mode in {"root", "backup"}, "AVI value mode must be root or backup")
    return f"sf-avi-{mode}-alpha={checked_alpha(alpha)!r}"


def value_source(
    mode: str,
    alpha: float,
    checkpoint_sha256: str,
    avi_summary_sha256: str,
) -> str:
    checkpoint_sha256 = _checked_sha256(
        checkpoint_sha256, label="teacher checkpoint SHA256"
    )
    avi_summary_sha256 = _checked_sha256(
        avi_summary_sha256, label="AVI summary SHA256"
    )
    return (
        f"stored-sf-search-plus-frozen-deepfin-{mode};"
        f"checkpoint={checkpoint_sha256};avi_summary={avi_summary_sha256};"
        f"neural_weight={checked_alpha(alpha)!r}"
    )


def _nonpolicy(files: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in files.items()
        if key != ".zattrs" and key.split("/")[0] != "policy_target"
    }


def _unchanged_nonvalue(files: dict[str, str]) -> dict[str, str]:
    return {
        key: value
        for key, value in files.items()
        if key != ".zattrs" and key.split("/")[0] != "search_wdl"
    }


def _sidecar_manifest(
    path: Path,
    expected_sha256: str,
    sf_root: Path,
    sf_summary_sha256: str,
    specs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require(
        derived.file_sha256(path) == expected_sha256,
        "AVI sidecar summary SHA256 differs",
    )
    raw_payload: Any = json.loads(path.read_text())
    if not isinstance(raw_payload, dict):
        raise ValueError("AVI sidecar summary must be a JSON object")
    payload: dict[str, Any] = raw_payload
    rows = sum(int(spec["rows"]) for spec in specs)
    require(
        payload.get("schema") == avi.SCHEMA
        and payload.get("kind") == avi.KIND
        and payload.get("status") == "COMPLETE_SELECTION"
        and payload.get("full_source_coverage") is True,
        "AVI sidecar is not a complete source collection",
    )
    source_dir = payload.get("source_dir")
    if not isinstance(source_dir, str):
        raise ValueError("AVI sidecar source directory identity missing")
    require(
        Path(source_dir).resolve() == sf_root
        and payload.get("source_summary_sha256") == sf_summary_sha256
        and int(payload.get("selected_rows", -1)) == rows
        and int(payload.get("source_rows", -1)) == rows
        and int(payload.get("selected_shards", -1)) == len(specs),
        "AVI sidecar source coverage differs",
    )
    checkpoint = payload.get("checkpoint")
    checkpoint_sha256 = payload.get("checkpoint_sha256")
    if not isinstance(checkpoint, str):
        raise ValueError("AVI checkpoint path identity missing")
    if not isinstance(checkpoint_sha256, str):
        raise ValueError("AVI checkpoint SHA256 identity missing")
    _checked_sha256(checkpoint_sha256, label="AVI checkpoint SHA256")

    outputs = payload.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(specs):
        raise ValueError("AVI output inventory differs")
    by_shard: dict[str, dict[str, Any]] = {}
    for item in outputs:
        if not isinstance(item, dict):
            raise ValueError("AVI output inventory contains a non-object entry")
        source_shard = item.get("source_shard")
        if not isinstance(source_shard, str):
            raise ValueError("AVI output is missing its source shard identity")
        if source_shard in by_shard:
            raise ValueError(f"duplicate AVI output for source shard {source_shard}")
        by_shard[source_shard] = item
    require(
        set(by_shard) == {str(spec["path"]) for spec in specs},
        "AVI output shard membership differs",
    )
    return payload, by_shard


def rewrite(args: argparse.Namespace) -> dict[str, Any]:
    alpha = checked_alpha(args.alpha)
    require(args.mode in {"root", "backup"}, "invalid AVI mode")
    require(
        type(args.batch_size) is int and args.batch_size > 0,
        "batch size must be positive",
    )
    require(
        math.isfinite(float(args.minimum_free_gib))
        and float(args.minimum_free_gib) >= 0,
        "invalid disk reserve",
    )

    source = Path(args.source).resolve()
    sf_root = Path(args.sf_source).resolve()
    avi_root = Path(args.avi).resolve()
    out = Path(args.out).resolve()
    writing = out.with_name(out.name + ".writing")
    roots = (source, sf_root, avi_root)
    require(len(set(roots)) == 3, "input roots must be distinct")
    require(
        all(
            out != root and root not in out.parents and out not in root.parents
            for root in roots
        ),
        "output overlaps input",
    )
    require(
        not os.path.lexists(out) and not os.path.lexists(writing),
        "output or partial exists",
    )

    for path, expected in (
        (source / DERIVE_SUMMARY, args.expected_source_summary_sha256),
        (source / POLICY_SUMMARY, args.expected_policy_summary_sha256),
        (sf_root / DERIVE_SUMMARY, args.expected_sf_summary_sha256),
        (avi_root / avi.SUMMARY, args.expected_avi_summary_sha256),
    ):
        require(
            derived.file_sha256(path) == expected,
            f"input pin differs: {path.name}",
        )

    base = json.loads((source / DERIVE_SUMMARY).read_text())
    policy = json.loads((source / POLICY_SUMMARY).read_text())
    sf_args = argparse.Namespace(
        source=str(sf_root),
        expected_source_summary_sha256=args.expected_sf_summary_sha256,
        start_shard=0,
        max_shards=2**31,
    )
    sf_summary, specs = derived.source_inventory(sf_args)
    require(
        bt4_value.equal_json(
            {
                key: value
                for key, value in base.items()
                if key != "policy_target_postprocess"
            },
            sf_summary,
        )
        and bt4_value.equal_json(base.get("policy_target_postprocess"), policy),
        "B100 source changed original value/history lineage",
    )
    rows = sum(int(spec["rows"]) for spec in specs)
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
        all(policy.get(key) == value for key, value in expected_policy.items()),
        "requires unchanged B100 policy recipe",
    )
    bt4_value.inventory(source, specs)
    bt4_value.inventory(sf_root, specs)
    require(
        avi_root.is_dir() and not avi_root.name.endswith(".writing"),
        "invalid AVI sidecar root",
    )
    avi_summary, avi_outputs = _sidecar_manifest(
        avi_root / avi.SUMMARY,
        args.expected_avi_summary_sha256,
        sf_root,
        args.expected_sf_summary_sha256,
        specs,
    )
    teacher_checkpoint_sha = _checked_sha256(
        str(avi_summary["checkpoint_sha256"]), label="AVI checkpoint SHA256"
    )
    avi_summary_sha = _checked_sha256(
        str(args.expected_avi_summary_sha256), label="AVI summary SHA256"
    )
    value_identity = value_source(
        args.mode,
        alpha,
        teacher_checkpoint_sha,
        avi_summary_sha,
    )

    source_states = {
        root / spec["path"]: derived.storage_identity(root / spec["path"])
        for root in (source, sf_root)
        for spec in specs
    }
    writing.mkdir(parents=True)
    outputs: list[dict[str, Any]] = []
    output_states: dict[Path, str] = {}
    changed = 0
    max_mass_error = 0.0

    def guard() -> None:
        require(
            not (writing / "STOP").exists()
            and not (out.parent / "STOP").exists(),
            "STOP requested",
        )
        require(
            shutil.disk_usage(writing).free
            >= float(args.minimum_free_gib) * 1024**3,
            "disk reserve breached",
        )

    try:
        for spec in specs:
            guard()
            name, n = str(spec["path"]), int(spec["rows"])
            src, original = source / name, sf_root / name
            original_group = derived.source_arrays(original, sf_summary, n)
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
                    key: value
                    for key, value in attrs.items()
                    if not key.startswith("policy_target_mix_")
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
                derived.complete_chunks(group[column])
                require(
                    group[column].shape[0] == n,
                    "source array row count differs",
                )
            require(
                group["search_wdl"].shape == (n, 3)
                and group["search_wdl"].dtype == np.dtype("float16"),
                "source WDL layout differs",
            )

            source_files = bt4_value.file_map(src)
            original_files = bt4_value.file_map(original)
            require(
                _nonpolicy(source_files) == _nonpolicy(original_files),
                "B100 nonpolicy source differs",
            )

            side_record = avi_outputs[name]
            require(side_record.get("rows") == n, "AVI sidecar row count differs")
            require(
                side_record.get("source_storage_identity") == source_states[original],
                "AVI sidecar source storage identity differs",
            )
            side_name = side_record.get("path")
            if not isinstance(side_name, str) or Path(side_name).name != side_name:
                raise ValueError("AVI sidecar path must be one canonical basename")
            require(
                side_name == avi.sidecar_name(name),
                "AVI sidecar name differs from source shard",
            )
            side_path = avi_root / side_name
            require(
                side_path.parent == avi_root and side_path.is_file(),
                "AVI sidecar path differs",
            )
            side_sha = side_record.get("sha256")
            if not isinstance(side_sha, str):
                raise ValueError("AVI sidecar payload SHA256 missing")
            _checked_sha256(side_sha, label="AVI sidecar payload SHA256")
            require(
                derived.file_sha256(side_path) == side_sha,
                "AVI sidecar payload SHA256 differs",
            )
            with np.load(side_path, allow_pickle=False) as archive:
                root_wdl = np.asarray(archive["root_wdl"], dtype=np.float32)
                backup_wdl = np.asarray(archive["backup_wdl"], dtype=np.float32)
            require(
                root_wdl.shape == backup_wdl.shape == (n, 3),
                "AVI WDL sidecar shape differs",
            )
            require(
                hashlib.sha256(np.ascontiguousarray(root_wdl).tobytes()).hexdigest()
                == side_record.get("root_wdl_sha256")
                and hashlib.sha256(
                    np.ascontiguousarray(backup_wdl).tobytes()
                ).hexdigest()
                == side_record.get("backup_wdl_sha256"),
                "AVI WDL payload digest differs",
            )
            neural = root_wdl if args.mode == "root" else backup_wdl

            destination = writing / name
            shutil.copytree(src, destination)
            require(
                bt4_value.file_map(destination) == source_files,
                "copied source bytes differ",
            )
            dest: Any = zarr.open_group(str(destination), mode="a")
            value_hash = hashlib.sha256()
            shard_changed = 0
            for start in range(0, n, args.batch_size):
                guard()
                end = min(n, start + args.batch_size)
                old = np.asarray(group["search_wdl"][start:end])
                if alpha == 0.0:
                    stored = old.copy()
                else:
                    stored = blend_wdl_targets(
                        old, neural[start:end], alpha=alpha
                    ).astype(np.float16)
                dest["search_wdl"][start:end] = stored
                readback = np.asarray(dest["search_wdl"][start:end])
                require(
                    bool(np.array_equal(readback, stored)),
                    "AVI value readback differs",
                )
                error = float(
                    np.abs(readback.astype(np.float64).sum(axis=1) - 1).max()
                )
                require(error <= 2**-10, "stored AVI value mass differs")
                max_mass_error = max(max_mass_error, error)
                shard_changed += int(np.any(readback != old, axis=1).sum())
                value_hash.update(readback.tobytes(order="C"))
            stamp = {
                "schema": SCHEMA,
                "algorithm": "normalized-wdl-sf-anchored-frozen-deepfin-v1",
                "mode": args.mode,
                "sf_weight": 1.0 - alpha,
                "neural_weight": alpha,
                "teacher_checkpoint_sha256": teacher_checkpoint_sha,
                "avi_summary_sha256": avi_summary_sha,
                "avi_sidecar_sha256": side_sha,
                "search_wdl_sha256": value_hash.hexdigest(),
            }
            dest.attrs.update(
                derive_schema=2,
                derive_value_scheme=value_scheme(args.mode, alpha),
                derive_value_source=value_identity,
                value_target_postprocess=stamp,
            )
            final_files = bt4_value.file_map(destination)
            require(
                _unchanged_nonvalue(source_files)
                == _unchanged_nonvalue(final_files),
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
                    "attrs_sha256": derived.file_sha256(destination / ".zattrs"),
                    "source_storage_identity": source_states[src],
                    "sf_source_storage_identity": source_states[original],
                    "avi_sidecar_sha256": side_sha,
                }
            )
            output_states[destination] = derived.storage_identity(destination)
            changed += shard_changed

        guard()
        for path, state in source_states.items():
            require(
                derived.storage_identity(path) == state,
                "source changed during AVI rewrite",
            )
        for path, expected in (
            (source / DERIVE_SUMMARY, args.expected_source_summary_sha256),
            (source / POLICY_SUMMARY, args.expected_policy_summary_sha256),
            (sf_root / DERIVE_SUMMARY, args.expected_sf_summary_sha256),
            (avi_root / avi.SUMMARY, args.expected_avi_summary_sha256),
        ):
            require(
                derived.file_sha256(path) == expected,
                "pinned input changed during AVI rewrite",
            )
        recipe = {
            "schema": SCHEMA,
            "status": "COMPLETE",
            "kind": "avi_value_rewrite",
            "algorithm": "normalized-wdl-sf-anchored-frozen-deepfin-v1",
            "mode": args.mode,
            "sf_weight": 1.0 - alpha,
            "neural_weight": alpha,
            "wdl_order": "WDL",
            "wdl_pov": "side_to_move",
            "teacher_checkpoint": avi_summary["checkpoint"],
            "teacher_checkpoint_sha256": teacher_checkpoint_sha,
            "avi_summary_sha256": avi_summary_sha,
            "rows": rows,
            "shards": len(specs),
            "source_dir": str(source),
            "sf_source_dir": str(sf_root),
            "avi_dir": str(avi_root),
            "source_derive_summary_sha256": args.expected_source_summary_sha256,
            "source_policy_summary_sha256": args.expected_policy_summary_sha256,
            "sf_derive_summary_sha256": args.expected_sf_summary_sha256,
            "mutated_arrays": ["search_wdl"],
            "unchanged_arrays": sorted(ARRAYS - {"search_wdl"}),
            "value_scheme": value_scheme(args.mode, alpha),
            "value_source": value_identity,
            "changed_rows": changed,
            "stored_mass_error_max": max_mass_error,
            "outputs": outputs,
        }
        derived_summary = dict(base)
        derived_summary["value_target_postprocess"] = {
            key: value for key, value in recipe.items() if key != "outputs"
        }
        derived_summary["value_scheme"] = {
            "name": recipe["value_scheme"],
            "source": recipe["value_source"],
        }
        (writing / POLICY_SUMMARY).write_bytes((source / POLICY_SUMMARY).read_bytes())
        (writing / SUMMARY).write_text(
            json.dumps(recipe, indent=2, sort_keys=True) + "\n"
        )
        (writing / DERIVE_SUMMARY).write_text(
            json.dumps(derived_summary, indent=2, sort_keys=True) + "\n"
        )
        for path, state in output_states.items():
            require(
                derived.storage_identity(path) == state,
                "AVI output changed before publication",
            )
        os.replace(writing, out)
        return recipe
    except BaseException as error:
        (writing / "failed.json").write_text(
            json.dumps({"complete": False, "error": str(error)}) + "\n"
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="B100 policy corpus")
    parser.add_argument(
        "--sf-source",
        type=Path,
        required=True,
        help="Corresponding original SF corpus",
    )
    parser.add_argument(
        "--avi",
        type=Path,
        required=True,
        help="Complete AVI successor sidecar directory",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-source-summary-sha256", required=True)
    parser.add_argument("--expected-policy-summary-sha256", required=True)
    parser.add_argument("--expected-sf-summary-sha256", required=True)
    parser.add_argument("--expected-avi-summary-sha256", required=True)
    parser.add_argument("--mode", choices=("root", "backup"), required=True)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--minimum-free-gib", type=float, default=150)
    return parser


if __name__ == "__main__":
    rewrite(build_parser().parse_args())
