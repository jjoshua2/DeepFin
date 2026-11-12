"""Explicit E2 one-game paired batch adapter; no optimizer or trainer-loop wiring.

Uses production Zarr loading, collate_arrays and compute_loss. Both arms keep
raw search labels plus natural outcomes; search fractions1/.8 blend ONCE in
compute_loss. Already blended WDL overlay arrays are verification references,
never fed as search labels. No augmentation: identical ordered rows and all
other tensors/weights. Caller must still qualify whole-bank/trace/model/LR/TB.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import torch

from chess_anti_engine.replay.dataset import collate_arrays
from chess_anti_engine.replay.shard import load_shard_arrays
from chess_anti_engine.train.losses import compute_loss
from chess_anti_engine.train.targets import categorical_target_value, hlgauss_target
from scripts.e2_bank_gate import digest, inventory


@dataclass(frozen=True)
class PairedBatch:
    search_only: dict[str, torch.Tensor]
    outcome20: dict[str, torch.Tensor]
    receipt: dict[str, Any]
    tensor_bindings: tuple[tuple[str, str, str], ...]


def tensor_hash(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(
        str((array.shape, array.dtype.str)).encode() + array.tobytes()
    ).hexdigest()


def load_paired_batch(
    *,
    shard: Path,
    candidate: Path,
    qualified: Path,
    expected_manifest_sha256: str,
    indices: tuple[int, ...],
    expected_indices_sha256: str,
    categorical_bins: int,
    categorical_sigma: float,
) -> PairedBatch:
    if (
        type(indices) is not tuple
        or not 1 <= len(indices) <= 256
        or any(type(i) is not int for i in indices)
    ):
        raise ValueError("bounded explicit frozen indices required")
    ordered_hash = hashlib.sha256(
        json.dumps(indices, separators=(",", ":")).encode()
    ).hexdigest()
    if ordered_hash != expected_indices_sha256:
        raise ValueError("batch order identity mismatch")
    if (
        type(categorical_bins) is not int
        or categorical_bins != 32
        or categorical_sigma != 0.04
    ):
        raise ValueError("model/config categorical contract must be32/.04")
    manifest = qualified / "GAME-READBACK-COMMITTED.json"
    if manifest.is_symlink() or manifest.stat().st_size > 2 * 1024**2:
        raise ValueError("manifest bound")
    raw = manifest.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("external manifest identity mismatch")
    m = json.loads(raw)
    if (
        m["status"] != "GAME_READBACK_QUALIFIED_NOT_EXPERIMENT_READY"
        or m["all175_replayed"] is not True
    ):
        raise ValueError("unqualified game readback")
    if (
        digest(candidate) != m["candidate_sha256"]
        or inventory(shard) != m["source_inventory"]
    ):
        raise ValueError("load-time source identity mismatch")
    if any(not 0 <= i < m["rows"] for i in indices):
        raise ValueError("batch index range")
    target_path = qualified / "arm-targets.npz"
    if target_path.is_symlink() or target_path.stat().st_size > 16 * 1024**2:
        raise ValueError("target archive bound")
    target_bytes = target_path.read_bytes()
    if hashlib.sha256(target_bytes).hexdigest() != m["arm_targets_sha256"]:
        raise ValueError("load-time target identity mismatch")
    with zipfile.ZipFile(io.BytesIO(target_bytes)) as z:
        if (
            len(z.infolist()) != 4
            or sum(f.file_size for f in z.infolist()) > 16 * 1024**2
        ):
            raise ValueError("decoded target archive bound")
    with np.load(io.BytesIO(target_bytes), allow_pickle=False) as z:
        targets = {key: np.array(z[key], copy=True) for key in z.files}
    arrays, meta = load_shard_arrays(shard, validate=True)
    if (
        meta.get("policy_encoding") != "lc0_1858"
        or meta.get("history_rep_fix") is not True
        or meta.get("input_history_encoding") != "lc0_root_legacy_meta"
    ):
        raise ValueError("encoding metadata mismatch")
    n = m["rows"]
    for flag in (
        "has_policy",
        "has_search_wdl",
        "has_game_id",
        "has_ply_index",
        "has_is_network_turn",
        "is_network_turn",
        "has_is_selfplay",
        "is_selfplay",
    ):
        if (
            flag + "/.zarray" not in m["source_inventory"]
            or flag not in arrays
            or arrays[flag].shape != (n,)
            or not np.all(arrays[flag] == 1)
        ):
            raise ValueError("required physical mask")
    search = arrays["search_wdl"].astype(np.float64)
    search /= search.sum(axis=1, keepdims=True)
    labels = arrays["wdl_target"].astype(np.int64)
    natural = np.eye(3)[labels]
    expected = {
        "wdl_a": search.astype(np.float32),
        "wdl_b": (0.8 * search + 0.2 * natural).astype(np.float32),
    }
    for suffix, frac in (("a", 1.0), ("b", 0.8)):
        expected["categorical_" + suffix] = np.array(
            [
                hlgauss_target(
                    categorical_target_value(
                        (1.0, 0.0, -1.0)[v],
                        None,
                        blend_frac=0.0,
                        search_wdl=s,
                        search_blend_frac=frac,
                    ),
                    num_bins=32,
                    sigma=0.04,
                )
                for v, s in zip(labels, search)
            ]
        )
    if set(targets) != set(expected) or any(
        not np.array_equal(targets[k], v) for k, v in expected.items()
    ):
        raise ValueError("arm target semantics mismatch")
    # Copy every source tensor before collate; no transformation/augmentation.
    batch_arrays = {
        k: np.array(v[list(indices)], copy=True)
        for k, v in arrays.items()
        if isinstance(v, np.ndarray)
        and v.ndim > 0
        and v.shape[0] == n
        and not k.startswith("_")
    }
    # Overlay ONLY categorical. Raw search/outcome/policy/sample weights stay same.
    batch_arrays["categorical_target"] = targets["categorical_a"][list(indices)].copy()
    batch_arrays["has_categorical"] = np.ones(len(indices), dtype=np.uint8)
    a = collate_arrays(batch_arrays, device="cpu")
    batch_arrays["categorical_target"] = targets["categorical_b"][list(indices)].copy()
    b = collate_arrays(batch_arrays, device="cpu")
    a = {k: v.clone() for k, v in a.items()}
    b = {k: v.clone() for k, v in b.items()}
    if any(not torch.equal(a[k], b[k]) for k in a if k != "categorical_t"):
        raise ValueError("non-target arm drift")
    if (
        inventory(shard) != m["source_inventory"]
        or digest(candidate) != m["candidate_sha256"]
        or digest(manifest) != expected_manifest_sha256
    ):
        raise ValueError("source changed during batch load")
    return PairedBatch(
        a,
        b,
        {
            "manifest_sha256": expected_manifest_sha256,
            "indices_sha256": ordered_hash,
            "indices": indices,
            "augmentation": "none",
            "wdl_input": "raw search plus natural class; blend once in loss",
            "categorical_bins": 32,
            "categorical_sigma": 0.04,
            "no_training_launched": True,
        },
        tuple((k, tensor_hash(a[k]), tensor_hash(b[k])) for k in sorted(a)),
    )


def compute_pair_losses(pair: PairedBatch, outputs_a, outputs_b, *, shared_kwargs):
    """Production-loss seam for paired validation, no backward/optimizer step.

    A real trainer must retain these controls and the same immutable batch trace.
    Both output dictionaries must come from the caller's separately pinned models.
    """
    if (
        tuple(
            (k, tensor_hash(pair.search_only[k]), tensor_hash(pair.outcome20[k]))
            for k in sorted(pair.search_only)
        )
        != pair.tensor_bindings
    ):
        raise ValueError("batch tensor identity changed")
    fixed = {
        "sf_wdl_frac": 0.0,
        "use_adjusted_wdl_target": False,
        "wdl_terminal_outcome_plies": 0,
        "sf_search_dampen_sf_low": 0.0,
        "sf_search_dampen_sf_high": 0.0,
    }
    forbidden = set(fixed) | {"search_wdl_frac"}
    if forbidden.intersection(shared_kwargs):
        raise ValueError("cannot override E2 mixture controls")
    if (
        outputs_a["categorical"].shape[-1] != 32
        or outputs_b["categorical"].shape[-1] != 32
    ):
        raise ValueError("actual categorical head mismatch")
    if set(pair.search_only) != set(pair.outcome20) or any(
        not torch.equal(pair.search_only[k], pair.outcome20[k])
        for k in pair.search_only
        if k != "categorical_t"
    ):
        raise ValueError("paired batch mutated")
    return (
        compute_loss(
            outputs_a, pair.search_only, search_wdl_frac=1.0, **fixed, **shared_kwargs
        ),
        compute_loss(
            outputs_b, pair.outcome20, search_wdl_frac=0.8, **fixed, **shared_kwargs
        ),
    )
