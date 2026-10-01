"""Source-only identity and coverage guards for the Tactical300 diagnostic."""
from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path


# Existing B100 identity in bt4_one_epoch_screen.verify_ceres_recipe, pinned at
# 8bd1f1021011c7c096c0d4e4e6021f663f88ad70. Do not import that operational
# module merely to check a manifest: this leaf has no launch or model access.
BT4_MODEL_SHA256 = "1d3c0bd28ebfb42b015d18f67831cb1d6d15ad5d358b25b8a8cf500786262fc0"
BT4_POLICY_OUTPUT = "/output/policy"


def validate_teacher(teacher: object) -> None:
    """An authenticated arbitrary teacher is not necessarily registered B100."""
    if not isinstance(teacher, Mapping):
        raise ValueError("registered B100 teacher mapping required")
    onnx = teacher.get("onnx")
    if not isinstance(onnx, Mapping) or onnx.get("sha256") != BT4_MODEL_SHA256:
        raise ValueError("adapter teacher is not the registered B100 ONNX")
    if teacher.get("policy_output") != BT4_POLICY_OUTPUT:
        raise ValueError("adapter teacher is not the registered B100 policy output")


def derived_inventory(source: Path, names: Collection[str]) -> list[Path]:
    """Require the complete manifest roster each time coverage is asserted."""
    paths = sorted(source.glob("shard_*.zarr"))
    if not paths or len(paths) != len(names) or set(names) != {p.name for p in paths}:
        raise ValueError("derived shard inventory differs from summary")
    return paths


def claim_source_identity(
    seen: set[tuple[str, str, int]], identity: tuple[str, str, int],
) -> None:
    """The caller retains this set for the entire selected cohort."""
    if identity in seen:
        raise ValueError("duplicate source-qualified derived row")
    seen.add(identity)


def top_set_is_final_mate(top: set[str], final_mates: set[str]) -> bool:
    """Every tied top move, not merely one of them, must be a winning mate."""
    return bool(top) and top <= final_mates
