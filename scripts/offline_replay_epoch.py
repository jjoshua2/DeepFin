#!/usr/bin/env python3
"""Train candidate optimizer scopes for one fixed replay epoch.

This bypasses selfplay/Ray entirely: it streams existing replay shards once,
optionally converts policy targets to the requested output encoding, and writes
one JSONL row per candidate. It is intended for architecture/optimizer probes
where generating fresh games would confound the comparison.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from chess_anti_engine.model import (
    ModelConfig,
    build_model,
    model_config_from_flat_config,
    normalize_embed_dim_by_layer,
    normalize_ffn_mult_by_layer,
)
from chess_anti_engine.utils.architecture import normalize_phase_piece_thresholds
from chess_anti_engine.encoding import input_plane_count, version_for_input_planes
from chess_anti_engine.encoding.lc0 import uses_lc0_root_history, uses_lc0_root_legacy_meta
from chess_anti_engine.replay.threat_upgrade import (
    V1_INPUT_PLANES,
    upgrade_arrays_to_planes,
)
from chess_anti_engine.moves import (
    COMPACT_TO_FULL_POLICY,
    COMPACT_POLICY_SIZE,
    FULL_TO_COMPACT_POLICY,
    POLICY_ENCODING_LC0_1858,
    POLICY_SIZE,
    normalize_policy_encoding,
    policy_batch_to_encoding,
    policy_batch_to_full,
)
from chess_anti_engine.replay.buffer import ReplayBuffer
from chess_anti_engine.replay.dataset import LEGAL_MASK_FIELDS
from chess_anti_engine.replay.shard import (
    DEFAULT_CATEGORICAL_BINS,
    INPUT_HISTORY_ENCODING_ARRAY_KEY,
    POLICY_ENCODING_ARRAY_KEY,
    _REQUIRED_STORAGE_FIELDS,
    _SHARD_FIELDS,
    iter_shard_paths,
    load_shard_arrays,
    shard_positions,
    zeros_for_storage_field,
)
from chess_anti_engine.train.trainer import (
    Trainer,
    TrainMetrics,
    _apply_lc0_root_legacy_meta,
    _legacy_x_to_synthetic_lc0_root,
    select_input_history_arrays,
    trainer_kwargs_from_config,
)
from chess_anti_engine.utils import flatten_run_config_defaults, load_yaml_file


def _limit_shards(shard_paths: list[Path], args: argparse.Namespace) -> list[Path]:
    if int(args.max_shards) <= 0:
        return shard_paths
    count = int(args.max_shards)
    return shard_paths[-count:] if bool(args.newest_shards) else shard_paths[:count]


POLICY_FIELDS = (
    "policy_target",
    "sf_policy_target",
    "policy_soft_target",
    "future_policy_target",
)

def _as_replay_buffer(sampler: Any) -> ReplayBuffer:
    """Duck-typed stand-in: Trainer.train_steps/eval_steps only require
    sample_batch_arrays + rng (the hasattr fast path in _sample_batch_host)."""
    return cast(ReplayBuffer, sampler)


class _FixedBatch:
    def __init__(self, arrs: dict[str, np.ndarray], rng: np.random.Generator):
        self._arrs = arrs
        self.rng = rng

    def sample_batch_arrays(self, batch_size: int, *, wdl_balance: bool = True) -> dict[str, np.ndarray]:
        del batch_size, wdl_balance
        return self._arrs


class _ArraySampler:
    def __init__(self, arrs: dict[str, np.ndarray], rng: np.random.Generator):
        self._arrs = arrs
        self.rng = rng
        self._n = int(np.asarray(arrs["x"]).shape[0])

    def sample_batch_arrays(self, batch_size: int, *, wdl_balance: bool = True) -> dict[str, np.ndarray]:
        del wdl_balance
        idx = self.rng.integers(0, self._n, size=int(batch_size), dtype=np.int64)
        return _slice_arrays(self._arrs, idx)


class _LiveShardSampler:
    """Random batch sampler over a replay shard directory that is changing live."""

    def __init__(
        self,
        *,
        replay_dir: str | Path,
        model_cfg: ModelConfig,
        rng: np.random.Generator,
        prefer_recorded_lc0_root: bool,
        synthetic_lc0_root_history: bool,
        lc0_root_legacy_meta: bool,
        upgrade_v1_planes: bool = False,
        cache_shards: int = 4,
    ) -> None:
        self.replay_dir = Path(replay_dir)
        self.model_cfg = model_cfg
        self.rng = rng
        self.prefer_recorded_lc0_root = bool(prefer_recorded_lc0_root)
        self.synthetic_lc0_root_history = bool(synthetic_lc0_root_history)
        self.lc0_root_legacy_meta = (
            bool(lc0_root_legacy_meta)
            or uses_lc0_root_legacy_meta(model_cfg.input_history_encoding)
        )
        self.upgrade_v1_planes = bool(upgrade_v1_planes)
        self.target_input_planes = input_plane_count(model_cfg.input_extra_features)
        # Floor 1: batches MIX across the cached shards, so an empty cache
        # (old --live-cache-shards 0 = "no caching") would make sampling
        # impossible (final-review note on d27285f).
        self.cache_shards = max(1, int(cache_shards))
        self._paths: list[Path] = []
        self._positions: np.ndarray = np.zeros((0,), dtype=np.int64)
        self._total_positions = 0
        self._cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self._known_positions: dict[Path, int] = {}

    @property
    def total_positions(self) -> int:
        return int(self._total_positions)

    @property
    def shard_count(self) -> int:
        return len(self._paths)

    def rescan(self) -> int:
        """Refresh the shard list and return positions in newly-seen shards."""
        paths = iter_shard_paths(self.replay_dir)
        positions: list[int] = []
        new_positions = 0
        seen = set(paths)
        for path in paths:
            n = self._known_positions.get(path)
            if n is None:
                n = int(shard_positions(path))
                if n > 0:
                    new_positions += n
                self._known_positions[path] = n
            positions.append(max(0, int(n)))

        for old in list(self._known_positions):
            if old not in seen:
                self._known_positions.pop(old, None)
                self._cache.pop(old, None)

        self._paths = paths
        self._positions = np.asarray(positions, dtype=np.int64)
        self._total_positions = int(self._positions.sum()) if self._positions.size else 0
        return int(new_positions)

    def _load_prepared(self, path: Path) -> dict[str, np.ndarray]:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached

        arrs, _meta = load_shard_arrays(path, lazy=False)
        arrs = _select_configured_input_history(
            arrs,
            input_history_encoding=self.model_cfg.input_history_encoding,
            prefer_recorded_lc0_root=self.prefer_recorded_lc0_root,
            synthetic_lc0_root_history=self.synthetic_lc0_root_history,
            lc0_root_legacy_meta=self.lc0_root_legacy_meta,
            target_input_planes=self.target_input_planes,
            upgrade_v1_planes=self.upgrade_v1_planes,
        )
        prepared = _convert_policy_targets(arrs, policy_encoding=self.model_cfg.policy_encoding)
        if self.cache_shards > 0:
            self._cache[path] = prepared
            self._cache.move_to_end(path)
            while len(self._cache) > self.cache_shards:
                self._cache.popitem(last=False)
        return prepared

    def sample_batch_arrays(self, batch_size: int, *, wdl_balance: bool = True) -> dict[str, np.ndarray]:
        del wdl_balance
        if not self._paths or self._total_positions <= 0:
            raise ValueError("live replay sampler has no shards")
        for _attempt in range(8):
            weights = self._positions.astype(np.float64, copy=True)
            if weights.sum() <= 0:
                raise ValueError("live replay sampler has no non-empty shards")
            idx = int(self.rng.choice(len(self._paths), p=weights / weights.sum()))
            path = self._paths[idx]
            try:
                # Refresh the LRU with the newly drawn shard, then MIX the
                # batch across every cached shard (rows weighted by shard
                # size). Single-shard batches (256 rows from ONE iteration's
                # games, no class balance) let the value head fit per-batch
                # outcome bias and drift into memorization — the 07-09
                # frozen-ruler regression (83.6→101.4cp while pool-eval
                # improved, tail-driven, both phases) had exactly that
                # signature; the live trainer never trains this way. Mixing
                # over the cache adds no IO: still one shard load per step.
                self._load_prepared(path)
                cached = [(pp, aa) for pp, aa in self._cache.items()]
                sizes = np.array(
                    [int(np.asarray(a["x"]).shape[0]) for _, a in cached], dtype=np.float64,
                )
                if sizes.sum() <= 0:
                    raise ValueError(f"empty shard {path}")
                picks = self.rng.choice(
                    len(cached), size=int(batch_size), p=sizes / sizes.sum(),
                )
                parts = []
                for ci in np.unique(picks):
                    k = int((picks == ci).sum())
                    n = int(sizes[ci])
                    rows = self.rng.integers(0, n, size=k, dtype=np.int64)
                    parts.append(_slice_arrays(cached[ci][1], rows))
                if len(parts) == 1:
                    return parts[0]
                return {
                    key: np.concatenate([pt[key] for pt in parts], axis=0)
                    for key in parts[0]
                }
            except Exception as exc:
                print(json.dumps({
                    "event": "live_shard_skip",
                    "path": str(path),
                    "error": str(exc),
                }), flush=True)
                self._known_positions.pop(path, None)
                self._cache.pop(path, None)
                self.rescan()
        raise RuntimeError("failed to sample a live replay batch after repeated shard reloads")


def _slice_arrays(arrs: dict[str, Any], idx: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in arrs.items():
        arr = np.asarray(value)
        if arr.ndim == 0:
            out[key] = np.array(arr, copy=True)
        else:
            out[key] = np.array(arr[idx], copy=True, order="C")
    return out


def _renorm_rows(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    sums = out.sum(axis=1, keepdims=True)
    good = sums[:, 0] > 0.0
    if np.any(good):
        out[good] = out[good] / sums[good]
    return out


def _convert_policy_targets(arrs: dict[str, Any], *, policy_encoding: str) -> dict[str, np.ndarray]:
    enc = normalize_policy_encoding(policy_encoding)
    out = {k: np.asarray(v) for k, v in arrs.items()}

    source_policy_size = int(np.asarray(out.get("_policy_size", 0)).item()) if "_policy_size" in out else 0
    if source_policy_size <= 0 and "policy_target" in out:
        policy = np.asarray(out["policy_target"])
        if policy.ndim == 2:
            source_policy_size = int(policy.shape[1])

    if enc == POLICY_ENCODING_LC0_1858:
        for key in POLICY_FIELDS:
            if key not in out:
                continue
            arr = np.asarray(out[key])
            if arr.ndim == 2 and int(arr.shape[1]) == POLICY_SIZE:
                out[key] = _renorm_rows(policy_batch_to_encoding(arr, policy_encoding=enc))
        for key in LEGAL_MASK_FIELDS:
            if key not in out:
                continue
            arr = np.asarray(out[key])
            if arr.ndim == 2 and int(arr.shape[1]) == POLICY_SIZE:
                out[key] = arr[:, COMPACT_TO_FULL_POLICY].astype(arr.dtype, copy=False)

        if "sf_move_index" in out and source_policy_size == POLICY_SIZE:
            idx = np.asarray(out["sf_move_index"], dtype=np.int64)
            valid = (idx >= 0) & (idx < POLICY_SIZE)
            compact = np.full(idx.shape, -1, dtype=np.int64)
            compact[valid] = FULL_TO_COMPACT_POLICY[idx[valid]]
            has = compact >= 0
            out["sf_move_index"] = np.where(has, compact, 0).astype(np.int64)
            if "has_sf_move" in out:
                out["has_sf_move"] = (
                    np.asarray(out["has_sf_move"], dtype=np.float32) * has.astype(np.float32)
                )
        out[POLICY_ENCODING_ARRAY_KEY] = np.asarray(enc)
        out["_policy_size"] = np.array(COMPACT_POLICY_SIZE, dtype=np.int32)
        return out

    for key in POLICY_FIELDS:
        if key not in out:
            continue
        arr = np.asarray(out[key])
        if arr.ndim == 2 and int(arr.shape[1]) == COMPACT_POLICY_SIZE:
            out[key] = _renorm_rows(
                policy_batch_to_full(
                    arr,
                    policy_encoding=POLICY_ENCODING_LC0_1858,
                    fill_value=0.0,
                ),
            )
    for key in LEGAL_MASK_FIELDS:
        if key not in out:
            continue
        arr = np.asarray(out[key])
        if arr.ndim == 2 and int(arr.shape[1]) == COMPACT_POLICY_SIZE:
            out[key] = policy_batch_to_full(
                arr,
                policy_encoding=POLICY_ENCODING_LC0_1858,
                fill_value=0.0,
            ).astype(arr.dtype, copy=False)

    if "sf_move_index" in out and source_policy_size == COMPACT_POLICY_SIZE:
        idx = np.asarray(out["sf_move_index"], dtype=np.int64)
        valid = (idx >= 0) & (idx < COMPACT_POLICY_SIZE)
        full = np.full(idx.shape, -1, dtype=np.int64)
        full[valid] = COMPACT_TO_FULL_POLICY[idx[valid]]
        out["sf_move_index"] = np.where(valid, full, 0).astype(np.int64)
        if "has_sf_move" in out:
            out["has_sf_move"] = (
                np.asarray(out["has_sf_move"], dtype=np.float32) * valid.astype(np.float32)
            )
    out[POLICY_ENCODING_ARRAY_KEY] = np.asarray(enc)
    out["_policy_size"] = np.array(POLICY_SIZE, dtype=np.int32)
    return out


def _maybe_synthetic_history(
    arrs: dict[str, Any],
    *,
    synthetic_lc0_root_history: bool,
    lc0_root_legacy_meta: bool,
) -> dict[str, np.ndarray]:
    if not synthetic_lc0_root_history:
        return {k: np.asarray(v) for k, v in arrs.items()}
    out = {k: np.asarray(v) for k, v in arrs.items()}
    legacy_x = np.asarray(out["x"])
    out["x"] = _legacy_x_to_synthetic_lc0_root(legacy_x)
    if lc0_root_legacy_meta:
        out["x"] = _apply_lc0_root_legacy_meta(np.asarray(out["x"]), legacy_x)
    return out


def _select_recorded_input_history(
    arrs: dict[str, Any],
    *,
    input_history_encoding: str,
    prefer_recorded_lc0_root: bool,
    lc0_root_legacy_meta: bool,
) -> dict[str, np.ndarray]:
    out = {k: np.asarray(v) for k, v in arrs.items()}
    if (
        prefer_recorded_lc0_root
        and uses_lc0_root_history(input_history_encoding)
        and "x_lc0_root" in out
    ):
        has = np.asarray(out.get("has_x_lc0_root", np.zeros((out["x"].shape[0],), dtype=np.uint8)))
        if bool(np.all(has != 0)):
            legacy_x = np.asarray(out["x"])
            root_x = np.asarray(out["x_lc0_root"])
            if lc0_root_legacy_meta or uses_lc0_root_legacy_meta(input_history_encoding):
                root_x = _apply_lc0_root_legacy_meta(root_x, legacy_x)
            out["x"] = root_x
    return out


def _is_known_version_width(planes: int) -> bool:
    """True if ``planes`` is the input-plane count of a registered version."""
    try:
        version_for_input_planes(int(planes))
    except ValueError:
        return False
    return True


def _normalize_x_planes_to_target(
    arrs: dict[str, np.ndarray],
    *,
    target_planes: int,
    upgrade_v1_planes: bool,
    history_encoding: str | None = None,
) -> dict[str, np.ndarray]:
    """Bring stored ``x``/``x_lc0_root`` up to the model's input-plane count.

    Mirrors ``DiskReplayBuffer._upgrade_x_planes`` + ``_pad_x_planes``: when the
    model expects a multi-version width (>146), ``upgrade_v1_planes`` is set, and
    the stored width is <= the target, the extra block is recomputed from the
    stored 112 LC0 base planes (``upgrade_arrays_to_planes``) — for ANY
    cross-version mismatch, not just v1/v2. Equal width is NOT a passthrough:
    rows the pre-upgrade zero-pad path persisted at the target width with an
    all-zero extra block are repaired in place (native chunks pass through
    cheaply via the helper's all-zero gate). Zero-padding a stored block of a
    DIFFERENT version into the target's extra slots is silent input corruption,
    so the only routine zero-pad is up from a width with NO extra block; the
    recompute's validation-failure fallback also zero-pads, by design. Shards
    storing MORE planes than the model expects are a hard error (never
    truncated).

    ``history_encoding`` is the configured ``input_history_encoding`` resolved by
    the caller; it is threaded into ``upgrade_arrays_to_planes`` so the EP-plane
    decode reads from the slot the run actually uses. Without it, older shards
    that lack a stored ``_input_history_encoding`` fall back to legacy EP decoding
    inside the recompute, fail the v1-prefix validation, and get the whole chunk
    zero-padded (silent signal loss for v3 training).
    """
    target = int(target_planes)
    if target <= V1_INPUT_PLANES:
        # v1 model: nothing to recompute; pad below handles narrower shards.
        upgrade_v1_planes = False
    out = arrs
    stored = int(np.asarray(arrs["x"]).shape[1])
    # stored > target is a config error handled (raised) by the pad loop below.
    # Equal width is NOT a passthrough: upgrade_arrays_to_planes repairs rows
    # the pre-upgrade zero-pad path persisted at the target width with an
    # all-zero extra block (native chunks pass through cheaply via its all-zero
    # gate), so route stored <= target through it.
    if upgrade_v1_planes and stored <= target and _is_known_version_width(target):
        try:
            out, _stats = upgrade_arrays_to_planes(
                arrs, target, history_encoding=history_encoding,
            )
        except ValueError as exc:
            print(json.dumps({
                "event": "plane_upgrade_failed",
                "target_planes": target,
                "stored_planes": stored,
                "error": str(exc),
            }), flush=True)
            out = arrs
    result = dict(out)
    for key in ("x", "x_lc0_root"):
        v = result.get(key)
        if v is None:
            continue
        v = np.asarray(v)
        cur = int(v.shape[1])
        if cur == target:
            continue
        if cur > target:
            raise ValueError(
                f"shard stores {cur} input planes but the model expects {target}; "
                f"refusing to truncate encoded inputs"
            )
        pad = np.zeros((v.shape[0], target - cur, *v.shape[2:]), dtype=v.dtype)
        result[key] = np.concatenate([v, pad], axis=1)
    return result


def _select_configured_input_history(
    arrs: dict[str, Any],
    *,
    input_history_encoding: str,
    prefer_recorded_lc0_root: bool,
    synthetic_lc0_root_history: bool,
    lc0_root_legacy_meta: bool,
    target_input_planes: int | None = None,
    upgrade_v1_planes: bool = False,
) -> dict[str, np.ndarray]:
    if uses_lc0_root_history(input_history_encoding):
        out = select_input_history_arrays(arrs, input_history_encoding=input_history_encoding)
    else:
        recorded = _select_recorded_input_history(
            arrs,
            input_history_encoding=input_history_encoding,
            prefer_recorded_lc0_root=prefer_recorded_lc0_root,
            lc0_root_legacy_meta=lc0_root_legacy_meta,
        )
        out = _maybe_synthetic_history(
            recorded,
            synthetic_lc0_root_history=synthetic_lc0_root_history,
            lc0_root_legacy_meta=lc0_root_legacy_meta,
        )
    if target_input_planes is not None:
        out = _normalize_x_planes_to_target(
            out,
            target_planes=int(target_input_planes),
            upgrade_v1_planes=bool(upgrade_v1_planes),
            history_encoding=input_history_encoding,
        )
    return out


def _concat(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ValueError("cannot concatenate empty eval chunk list")

    present_keys = set().union(*(set(c.keys()) for c in chunks))
    keys = [
        name for name in _SHARD_FIELDS
        if name in _REQUIRED_STORAGE_FIELDS or name in present_keys
    ]
    categorical_bins = next(
        (int(np.asarray(c["categorical_target"]).shape[1]) for c in chunks if "categorical_target" in c),
        DEFAULT_CATEGORICAL_BINS,
    )
    out: dict[str, np.ndarray] = {}
    for key in keys:
        parts: list[np.ndarray] = []
        for chunk in chunks:
            if key in chunk:
                parts.append(np.asarray(chunk[key]))
                continue
            parts.append(
                zeros_for_storage_field(
                    key,
                    n=int(np.asarray(chunk["x"]).shape[0]),
                    policy_size=int(np.asarray(chunk["policy_target"]).shape[1]),
                    x_planes=int(np.asarray(chunk["x"]).shape[1]),
                    categorical_bins=categorical_bins,
                )
            )
        out[key] = np.concatenate(parts, axis=0)

    for key in sorted(present_keys - set(keys)):
        values = [np.asarray(chunk[key]) for chunk in chunks if key in chunk]
        if not values:
            continue
        if all(value.ndim == 0 for value in values):
            if len(values) != len(chunks):
                raise ValueError(f"metadata field {key!r} missing from some eval chunks")
            first = values[0].item()
            if any(value.item() != first for value in values[1:]):
                raise ValueError(f"mixed eval metadata for {key!r}")
            out[key] = np.array(values[0], copy=True)
            continue
        if key == INPUT_HISTORY_ENCODING_ARRAY_KEY:
            parts = []
            for chunk in chunks:
                n = int(np.asarray(chunk["x"]).shape[0])
                if key not in chunk:
                    parts.append(np.full((n,), "", dtype=object))
                    continue
                arr = np.asarray(chunk[key])
                if arr.ndim == 0:
                    parts.append(np.full((n,), arr.item(), dtype=object))
                else:
                    parts.append(arr)
            out[key] = np.concatenate(parts, axis=0)
    return out


def _candidate_spec(name: str) -> tuple[str, str]:
    raw = str(name).strip().lower()
    aliases = {
        "adamw": ("adamw", "default"),
        "aurora": ("aurora", "block_all"),
        "aurora_blocks": ("aurora", "block_all"),
        "aurora_block_all": ("aurora", "block_all"),
        "aurora_all_block": ("aurora", "mlp_attn_all"),
        "aurora_mlp_attn_all": ("aurora", "mlp_attn_all"),
        "aurora_mlp": ("aurora", "mlp_only"),
        "aurora_mlp_only": ("aurora", "mlp_only"),
        "aurora_mlp_out": ("aurora", "mlp_out"),
        "aurora_mlp_o": ("aurora", "mlp_out"),
        "aurora_mlp_out_v": ("aurora", "mlp_out_v"),
        "aurora_mlp_v_out": ("aurora", "mlp_out_v"),
        "aurora_mlp_attn_ov": ("aurora", "mlp_out_v"),
        "aurora_all": ("aurora", "mlp_attn_all"),
        "muon_mlp": ("muon", "mlp_only"),
        "muon_mlp_only": ("muon", "mlp_only"),
        "muon_mlp_out": ("muon", "mlp_out"),
        "muon_mlp_o": ("muon", "mlp_out"),
        "muon_mlp_out_v": ("muon", "mlp_out_v"),
        "muon_mlp_attn_ov": ("muon", "mlp_out_v"),
        "muon_all": ("muon", "mlp_attn_all"),
        "muon_all_block": ("muon", "mlp_attn_all"),
        "cosmos": ("cosmos", "default"),
        "cosmos_mlp": ("cosmos", "mlp_only"),
        "cosmos_mlp_only": ("cosmos", "mlp_only"),
        "cosmos_mlp_out": ("cosmos", "mlp_out"),
        "cosmos_mlp_out_v": ("cosmos", "mlp_out_v"),
        "cosmos_mlp_attn_ov": ("cosmos", "mlp_out_v"),
        "cosmos_all": ("cosmos", "mlp_attn_all"),
        "cosmos_fast": ("cosmos_fast", "default"),
        "cosmos_fast_mlp": ("cosmos_fast", "mlp_only"),
        "cosmos_fast_mlp_only": ("cosmos_fast", "mlp_only"),
        "cosmos_fast_mlp_out": ("cosmos_fast", "mlp_out"),
        "cosmos_fast_mlp_out_v": ("cosmos_fast", "mlp_out_v"),
        "cosmos_fast_mlp_attn_ov": ("cosmos_fast", "mlp_out_v"),
        "cosmos_fast_all": ("cosmos_fast", "mlp_attn_all"),
        "soap": ("soap", "default"),
        "soap_mlp": ("soap", "mlp_only"),
        "soap_mlp_only": ("soap", "mlp_only"),
        "soap_mlp_out": ("soap", "mlp_out"),
        "soap_mlp_out_v": ("soap", "mlp_out_v"),
        "soap_mlp_attn_ov": ("soap", "mlp_out_v"),
        "soap_all": ("soap", "mlp_attn_all"),
    }
    if raw not in aliases:
        raise ValueError(f"unknown candidate {name!r}; expected one of {sorted(aliases)}")
    return aliases[raw]


def _build_trainer_for_candidate(
    *,
    candidate: str,
    cfg: dict[str, Any],
    model_cfg: ModelConfig,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[Trainer, str, str]:
    optimizer, scope = _candidate_spec(candidate)
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    model = build_model(model_cfg)
    trainer_kwargs = trainer_kwargs_from_config(cfg, log_dir=run_dir / "tb")
    trainer_kwargs.update(
        optimizer=optimizer,
        matrix_optimizer_scope=scope,
        matrix_lr_multiplier=float(args.matrix_lr_multiplier),
        matrix_weight_decay=float(args.matrix_weight_decay),
        aux_weight_decay=float(args.aux_weight_decay),
        global_board_preprocess_lr_multiplier=float(args.global_board_preprocess_lr_multiplier),
        global_board_preprocess_weight_decay=float(args.global_board_preprocess_weight_decay),
        global_board_adapter_lr_multiplier=float(args.global_board_adapter_lr_multiplier),
        global_board_adapter_weight_decay=float(args.global_board_adapter_weight_decay),
        weight_decay_mode=str(args.weight_decay_mode),
        soda_scope=str(args.soda_scope),
        soda_start_step=int(args.soda_start_step),
        aurora_uw_floor=float(args.aurora_uw_floor),
        aurora_pp_iterations=int(args.aurora_pp_iterations),
        aurora_pp_beta=float(args.aurora_pp_beta),
        aurora_polar_steps=int(args.aurora_polar_steps),
        aurora_polar_method=str(args.aurora_polar_method),
        aurora_polar_dtype=str(args.aurora_polar_dtype),
        aurora_polar_safety=float(args.aurora_polar_safety),
        use_compile=bool(args.compile),
        use_amp=not bool(args.no_amp),
        prefetch_batches=False,
        model_config=model_cfg,
    )
    if args.zclip_max_norm is not None:
        trainer_kwargs["zclip_max_norm"] = _parse_optional_float_arg(args.zclip_max_norm)
    if args.zclip_z_thresh is not None:
        trainer_kwargs["zclip_z_thresh"] = float(args.zclip_z_thresh)
    if args.zclip_clip_factor is not None:
        trainer_kwargs["zclip_clip_factor"] = float(args.zclip_clip_factor)
    trainer = Trainer(model, **trainer_kwargs)
    # Capture the per-group LRs the caller asked for (cfg / --lr, incl. matrix
    # multipliers). NOT pg["lr"]: Trainer.__init__ ends with _set_initial_lrs(),
    # which (warmup_steps > 0) sets every param group to the WARMUP-START rate
    # (~lr_eta_min, e.g. 1e-5) — capturing that would rebase a phase-2 "--lr
    # 1e-4" drop to 1e-5. The scheduler is constructed BEFORE _set_initial_lrs,
    # so its base_lrs still hold the true requested per-group rates.
    _sched0 = getattr(trainer, "_scheduler", None)
    desired_lrs = [float(v) for v in getattr(_sched0, "base_lrs", None) or []]
    if not desired_lrs:
        desired_lrs = [float(pg.get("lr", 0.0)) for pg in trainer.opt.param_groups]
    if args.init_checkpoint:
        trainer.load(Path(args.init_checkpoint))
        if args.lr is not None:
            for pg, lr in zip(trainer.opt.param_groups, desired_lrs, strict=True):
                pg["lr"] = float(lr)
            sched = getattr(trainer, "_scheduler", None)
            if sched is not None:
                if hasattr(sched, "base_lrs"):
                    sched.base_lrs = list(desired_lrs)
                if hasattr(sched, "_last_lr"):
                    sched._last_lr = list(desired_lrs)
            if hasattr(trainer, "_peak_lr"):
                trainer._peak_lr = float(args.lr)
            print(json.dumps({
                "event": "lr_override_after_load",
                "lr": float(args.lr),
                "param_group_lrs": list(desired_lrs),
                "init_checkpoint": str(args.init_checkpoint),
            }), flush=True)
    return trainer, optimizer, scope


def _calibrate_global_board_adapter_init(
    *,
    trainer: Trainer,
    eval_arrs: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, float | int]:
    target_ratio = float(args.global_board_adapter_init_rms_ratio)
    if target_ratio <= 0.0:
        return {}
    if args.init_checkpoint:
        raise ValueError("--global-board-adapter-init-rms-ratio is only valid for fresh-init runs")

    model = trainer.model
    preprocess = getattr(model, "global_board_preprocess", None)
    adapter = getattr(model, "global_board_adapter", None)
    if preprocess is None or adapter is None:
        raise ValueError("--global-board-adapter-init-rms-ratio requires input_global_embedding=bt4_board_adapter")

    x_np = np.asarray(eval_arrs["x"])
    batch_size = min(int(args.global_board_adapter_init_batch_size), int(x_np.shape[0]))
    if batch_size <= 0:
        raise ValueError("cannot calibrate global-board adapter with an empty eval batch")
    device = torch.device(trainer.device)
    x = torch.as_tensor(x_np[:batch_size], dtype=torch.float32, device=device)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        # torch stubs type .index as int, but a bare "cuda" device has index None.
        device_index = cast("int | None", x.device.index)
        devices = [device_index] if x.is_cuda and device_index is not None else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(args.seed) + 104729)
            if x.is_cuda:
                torch.cuda.manual_seed_all(int(args.seed) + 104729)
            torch.nn.init.xavier_uniform_(adapter.weight)

        b, c, h, w = x.shape
        if (h, w) != (8, 8):
            raise ValueError(f"expected x shape (B,C,8,8), got {tuple(x.shape)}")
        tokens = x.reshape(b, c, 64).permute(0, 2, 1).contiguous()
        if getattr(model, "input_pos_encoding", "none") in ("arc", "arc_adapter"):
            arc_pos = model._buffers["arc_pos_encoding"]
            if arc_pos is None:
                raise ValueError("arc_pos_encoding buffer missing for input_pos_encoding=arc*")
            pos = arc_pos.to(dtype=tokens.dtype, device=device)
            if getattr(model, "input_pos_encoding", "none") == "arc":
                tokens = torch.cat((tokens, pos.expand(b, -1, -1)), dim=-1)

        board_summary = x[:, :12].reshape(b, 12 * 64)
        global_tokens = preprocess(board_summary).view(
            b,
            64,
            int(getattr(model, "input_global_embedding_channels")),
        )
        # nn.Module.__getattr__ types dynamic attributes as Tensor | Module;
        # pin the real types for this experimental-arch access.
        embed = cast(torch.nn.Module, model.embed)
        embed_gate_mul_raw = cast(torch.Tensor, model._embed_gate_mul_raw)
        embed_gate_add = cast(torch.Tensor, model.embed_gate_add)
        token_residual = F.mish(embed(tokens))
        token_residual = (
            token_residual * F.softplus(embed_gate_mul_raw) + embed_gate_add
        )
        square_gate_mul = cast("torch.Tensor | None", getattr(model, "square_embed_gate_mul_raw", None))
        if square_gate_mul is not None:
            token_residual = token_residual * F.softplus(square_gate_mul).unsqueeze(0)
        square_gate_add = cast("torch.Tensor | None", getattr(model, "square_embed_gate_add", None))
        if square_gate_add is not None:
            token_residual = token_residual + square_gate_add.unsqueeze(0)

        branch = adapter(global_tokens)
        token_rms = token_residual.float().square().mean().sqrt().clamp_min(1e-12)
        branch_rms_before = branch.float().square().mean().sqrt().clamp_min(1e-12)
        scale = torch.as_tensor(target_ratio, device=device, dtype=torch.float32) * token_rms / branch_rms_before
        adapter.weight.mul_(scale.to(dtype=adapter.weight.dtype))
        branch_after = adapter(global_tokens)
        branch_rms_after = branch_after.float().square().mean().sqrt()

    if was_training:
        model.train()

    stats: dict[str, float | int] = {
        "global_board_adapter_init_batch_size": int(batch_size),
        "global_board_adapter_init_target_ratio": float(target_ratio),
        "global_board_adapter_init_token_rms": float(token_rms.item()),
        "global_board_adapter_init_branch_rms_before": float(branch_rms_before.item()),
        "global_board_adapter_init_branch_rms_after": float(branch_rms_after.item()),
        "global_board_adapter_init_actual_ratio": float((branch_rms_after / token_rms).item()),
        "global_board_adapter_init_weight_rms": float(adapter.weight.float().square().mean().sqrt().item()),
    }
    print(json.dumps({"event": "global_board_adapter_calibrated", **stats}), flush=True)
    return stats


def _load_eval_arrs(
    *,
    shard_paths: list[Path],
    model_cfg: ModelConfig,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    eval_chunks: list[dict[str, np.ndarray]] = []
    eval_left = int(args.eval_positions)
    for shard in shard_paths:
        if eval_left <= 0:
            break
        try:
            arrs, _meta = load_shard_arrays(shard, lazy=False)
        except Exception as exc:
            print(json.dumps({
                "event": "eval_shard_skip",
                "path": str(shard),
                "error": str(exc),
            }), flush=True)
            continue
        arrs = _select_configured_input_history(
            arrs,
            input_history_encoding=model_cfg.input_history_encoding,
            prefer_recorded_lc0_root=bool(args.prefer_recorded_lc0_root),
            synthetic_lc0_root_history=bool(args.synthetic_lc0_root_history),
            lc0_root_legacy_meta=(
                bool(args.lc0_root_legacy_meta)
                or uses_lc0_root_legacy_meta(model_cfg.input_history_encoding)
            ),
            target_input_planes=input_plane_count(model_cfg.input_extra_features),
            upgrade_v1_planes=bool(args.replay_upgrade_v1_planes),
        )
        arrs = _convert_policy_targets(arrs, policy_encoding=model_cfg.policy_encoding)
        n = min(eval_left, int(np.asarray(arrs["x"]).shape[0]))
        eval_chunks.append(_slice_arrays(arrs, np.arange(n, dtype=np.int64)))
        eval_left -= n
    if not eval_chunks:
        raise ValueError("could not build eval set from replay shards")
    return _concat(eval_chunks)


def _parse_optional_float_arg(value: str) -> float | None:
    raw = str(value).strip().lower()
    if raw in ("none", "null", "off", "false"):
        return None
    return float(raw)


def _parse_bool_choice(value: str) -> bool:
    raw = str(value).strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected one of on/off/true/false, got {value!r}")


def _parse_ffn_mult_by_layer(value: str) -> tuple[float, ...] | None:
    try:
        return normalize_ffn_mult_by_layer(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_embed_dim_by_layer(value: str) -> tuple[int, ...] | None:
    try:
        return normalize_embed_dim_by_layer(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_phase_piece_thresholds(value: str) -> tuple[int, int]:
    try:
        return normalize_phase_piece_thresholds(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _model_config_from_offline_config(cfg: dict[str, Any]) -> ModelConfig:
    return model_config_from_flat_config(
        cfg,
        embed_dim_default=256,
        num_layers_default=6,
        num_heads_default=8,
    )


def _train_candidate(
    *,
    candidate: str,
    cfg: dict[str, Any],
    shard_paths: list[Path],
    eval_arrs: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = Path(args.out_dir) / candidate
    run_dir.mkdir(parents=True, exist_ok=True)

    model_cfg = _model_config_from_offline_config(cfg)
    trainer, optimizer, scope = _build_trainer_for_candidate(
        candidate=candidate,
        cfg=cfg,
        model_cfg=model_cfg,
        args=args,
        run_dir=run_dir,
    )
    adapter_init_stats = _calibrate_global_board_adapter_init(
        trainer=trainer,
        eval_arrs=eval_arrs,
        args=args,
    )
    rng = np.random.default_rng(int(args.seed))

    steps = 0
    positions = 0
    skipped_shards = 0
    last_metrics: TrainMetrics | None = None
    t0 = time.time()
    for shard_i, shard in enumerate(shard_paths, start=1):
        try:
            arrs, _meta = load_shard_arrays(shard, lazy=False)
        except Exception as exc:
            skipped_shards += 1
            print(json.dumps({
                "event": "train_shard_skip",
                "candidate": candidate,
                "shard": shard_i,
                "path": str(shard),
                "error": f"{type(exc).__name__}: {exc}",
            }), flush=True)
            continue
        arrs = _select_configured_input_history(
            arrs,
            input_history_encoding=model_cfg.input_history_encoding,
            prefer_recorded_lc0_root=bool(args.prefer_recorded_lc0_root),
            synthetic_lc0_root_history=bool(args.synthetic_lc0_root_history),
            lc0_root_legacy_meta=(
                bool(args.lc0_root_legacy_meta)
                or uses_lc0_root_legacy_meta(model_cfg.input_history_encoding)
            ),
            target_input_planes=input_plane_count(model_cfg.input_extra_features),
            upgrade_v1_planes=bool(args.replay_upgrade_v1_planes),
        )
        arrs = _convert_policy_targets(arrs, policy_encoding=model_cfg.policy_encoding)
        n = int(np.asarray(arrs["x"]).shape[0])
        order = rng.permutation(n)
        for start in range(0, n, int(args.batch_size)):
            idx = order[start:start + int(args.batch_size)]
            batch = _slice_arrays(arrs, idx)
            last_metrics = trainer.train_steps(_as_replay_buffer(_FixedBatch(batch, rng)), batch_size=int(idx.shape[0]), steps=1)
            steps += 1
            positions += int(idx.shape[0])
        if shard_i % int(args.report_every_shards) == 0:
            elapsed = time.time() - t0
            row: dict[str, Any] = {
                "event": "progress",
                "candidate": candidate,
                "shards": shard_i,
                "skipped_shards": skipped_shards,
                "positions": positions,
                "steps": steps,
                "positions_per_s": positions / max(elapsed, 1e-9),
            }
            if last_metrics is not None:
                metric_values = dataclasses.asdict(last_metrics)
                for key in (
                    "loss",
                    "policy_loss",
                    "soft_policy_loss",
                    "future_policy_loss",
                    "wdl_loss",
                    "blended_wdl_loss",
                    "sf_move_loss",
                    "sf_move_acc",
                    "policy_own_acc_top1",
                    "policy_future_acc_top1",
                    "channel_balance",
                    "aurora_uw_ratio_min",
                    "aurora_uw_ratio_p10",
                    "aurora_uw_ratio_median",
                    "aurora_uw_ratio_p90",
                    "aurora_uw_scale_max",
                    "aurora_uw_floored_frac",
                    "aurora_uw_effective_ratio_min",
                    "aurora_uw_effective_ratio_median",
                ):
                    row[f"last_{key}"] = metric_values[key]
            print(json.dumps(row), flush=True)

    eval_steps = max(1, int(args.eval_steps))
    eval_metrics = trainer.eval_steps(
        _as_replay_buffer(_ArraySampler(eval_arrs, np.random.default_rng(int(args.seed) + 999))),
        batch_size=int(args.batch_size),
        steps=eval_steps,
    )
    out = {
        "candidate": candidate,
        "optimizer": optimizer,
        "matrix_optimizer_scope": scope,
        "matrix_lr_multiplier": float(args.matrix_lr_multiplier),
        "matrix_weight_decay": float(args.matrix_weight_decay),
        "aux_weight_decay": float(args.aux_weight_decay),
        "global_board_preprocess_lr_multiplier": float(args.global_board_preprocess_lr_multiplier),
        "global_board_preprocess_weight_decay": float(args.global_board_preprocess_weight_decay),
        "global_board_adapter_lr_multiplier": float(args.global_board_adapter_lr_multiplier),
        "global_board_adapter_weight_decay": float(args.global_board_adapter_weight_decay),
        "global_board_adapter_init_rms_ratio": float(args.global_board_adapter_init_rms_ratio),
        "global_board_adapter_init_batch_size": int(args.global_board_adapter_init_batch_size),
        "weight_decay_mode": str(args.weight_decay_mode),
        "soda_scope": str(args.soda_scope),
        "soda_start_step": int(args.soda_start_step),
        "aurora_uw_floor": float(args.aurora_uw_floor),
        "aurora_pp_iterations": int(args.aurora_pp_iterations),
        "aurora_pp_beta": float(args.aurora_pp_beta),
        "aurora_polar_steps": int(args.aurora_polar_steps),
        "aurora_polar_method": str(args.aurora_polar_method),
        "aurora_polar_dtype": str(args.aurora_polar_dtype),
        "aurora_polar_safety": float(args.aurora_polar_safety),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else "",
        **adapter_init_stats,
        "w_future": float(cfg.get("w_future", 0.15)),
        "positions": positions,
        "steps": steps,
        "skipped_shards": skipped_shards,
        "elapsed_s": time.time() - t0,
        "model_config": dataclasses.asdict(model_cfg),
        **{f"eval_{k}": v for k, v in dataclasses.asdict(eval_metrics).items()},
    }
    trainer.save(run_dir / "trainer.pt")
    return out


def _train_candidate_live_follow(
    *,
    candidate: str,
    cfg: dict[str, Any],
    model_cfg: ModelConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_dir = Path(args.out_dir) / candidate
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer, optimizer, scope = _build_trainer_for_candidate(
        candidate=candidate,
        cfg=cfg,
        model_cfg=model_cfg,
        args=args,
        run_dir=run_dir,
    )
    rng = np.random.default_rng(int(args.seed))
    sampler = _LiveShardSampler(
        replay_dir=args.replay_dir,
        model_cfg=model_cfg,
        rng=rng,
        prefer_recorded_lc0_root=bool(args.prefer_recorded_lc0_root),
        synthetic_lc0_root_history=bool(args.synthetic_lc0_root_history),
        lc0_root_legacy_meta=(
            bool(args.lc0_root_legacy_meta)
            or uses_lc0_root_legacy_meta(model_cfg.input_history_encoding)
        ),
        upgrade_v1_planes=bool(args.replay_upgrade_v1_planes),
        cache_shards=int(args.live_cache_shards),
    )
    sampler.rescan()
    min_positions = max(int(args.batch_size), int(args.live_min_positions))
    if sampler.total_positions < min_positions:
        raise SystemExit(
            f"live replay has only {sampler.total_positions} positions; "
            f"need at least {min_positions}"
        )

    target_reuse = float(args.live_target_reuse)
    credit_samples = float(sampler.total_positions) * max(0.0, float(args.live_initial_credit_frac)) * target_reuse
    credit_cap = (
        float("inf") if int(args.live_credit_cap_steps) <= 0
        else float(int(args.live_credit_cap_steps) * int(args.batch_size))
    )
    credit_samples = min(credit_samples, credit_cap)
    max_steps = int(args.live_max_steps)
    rescan_steps = max(1, int(args.live_rescan_steps))
    save_every = max(0, int(args.live_save_every_steps))
    eval_every = max(0, int(args.live_eval_every_steps))
    report_every = max(1, int(args.live_report_every_steps))
    idle_sleep_s = max(0.0, float(args.live_idle_sleep))
    # Static pools (frozen salvage dirs) never grow: without an idle-exit the
    # loop sleeps forever after credit is spent. 0 = legacy wait-forever (true
    # live shard streams); bootstrap uses a positive value so the phase ends
    # and the driver can drop LR / start the next phase.
    idle_exit_after_s = max(0.0, float(args.live_idle_exit_after))
    plateau_evals = max(0, int(args.live_plateau_evals))
    plateau_min_delta = float(args.live_plateau_min_delta)

    steps = 0
    samples = 0
    t0 = time.time()
    stop_reason = "max_steps" if max_steps > 0 else "running"
    credit_idle_since: float | None = None
    best_eval_loss: float | None = None
    plateau_streak = 0
    # Warm continuations must not clobber the prior run's trainer_best_eval.pt.
    # Preferred seed for the plateau bar: the loaded best's PERSISTED eval loss
    # (trainer_best_eval.json) — comparing against our own first eval would let
    # a later eval that improves only relative to a bad restart point overwrite
    # the genuinely-better loaded best. Fallback (no sidecar, e.g. a pre-fix
    # run dir): seed from the first eval. Caveat noted: the eval sample drifts
    # as the pool grows, so a stale baseline errs conservative (protects the
    # best file), which is the right direction.
    seed_best_pending = False
    if args.init_checkpoint and plateau_evals > 0:
        sidecar = run_dir / "trainer_best_eval.json"
        try:
            best_eval_loss = float(json.loads(sidecar.read_text(encoding="utf-8"))["eval_loss"])
            print(json.dumps({
                "event": "live_best_eval_baseline_loaded",
                "candidate": candidate,
                "eval_loss": best_eval_loss,
                "path": str(sidecar),
            }), flush=True)
        except (OSError, ValueError, KeyError):
            seed_best_pending = True

    print(json.dumps({
        "event": "live_follow_ready",
        "candidate": candidate,
        "shards": sampler.shard_count,
        "positions": sampler.total_positions,
        "target_reuse": target_reuse,
        "initial_credit_samples": int(credit_samples) if math.isfinite(credit_samples) else "inf",
        "credit_cap_steps": int(args.live_credit_cap_steps),
        "idle_exit_after_s": idle_exit_after_s,
        "plateau_evals": plateau_evals,
    }), flush=True)

    while max_steps <= 0 or steps < max_steps:
        if steps == 0 or steps % rescan_steps == 0:
            new_positions = sampler.rescan()
            if steps > 0 and new_positions > 0:
                credit_samples = min(credit_samples + float(new_positions) * target_reuse, credit_cap)
                credit_idle_since = None
            if sampler.total_positions < min_positions:
                print(json.dumps({
                    "event": "live_follow_wait",
                    "candidate": candidate,
                    "reason": "not_enough_positions",
                    "positions": sampler.total_positions,
                    "min_positions": min_positions,
                    "sleep_s": idle_sleep_s,
                }), flush=True)
                time.sleep(idle_sleep_s)
                continue

        if credit_samples < int(args.batch_size):
            new_positions = sampler.rescan()
            if new_positions > 0:
                credit_samples = min(credit_samples + float(new_positions) * target_reuse, credit_cap)
                credit_idle_since = None
                continue
            now = time.time()
            if credit_idle_since is None:
                credit_idle_since = now
            idle_for = now - credit_idle_since
            print(json.dumps({
                "event": "live_follow_wait",
                "candidate": candidate,
                "reason": "credit_exhausted",
                "credit_samples": int(credit_samples),
                "positions": sampler.total_positions,
                "shards": sampler.shard_count,
                "sleep_s": idle_sleep_s,
                "idle_for_s": idle_for,
                "idle_exit_after_s": idle_exit_after_s,
            }), flush=True)
            if idle_exit_after_s > 0.0 and idle_for >= idle_exit_after_s:
                stop_reason = "credit_exhausted_idle"
                print(json.dumps({
                    "event": "live_follow_stop",
                    "candidate": candidate,
                    "reason": stop_reason,
                    "steps": steps,
                    "samples": samples,
                    "positions": sampler.total_positions,
                    "idle_for_s": idle_for,
                }), flush=True)
                break
            time.sleep(idle_sleep_s)
            continue

        last_metrics = trainer.train_steps(_as_replay_buffer(sampler), batch_size=int(args.batch_size), steps=1)
        steps += 1
        samples += int(args.batch_size)
        credit_samples -= int(args.batch_size)
        credit_idle_since = None

        if steps % report_every == 0:
            elapsed = time.time() - t0
            row: dict[str, Any] = {
                "event": "live_progress",
                "candidate": candidate,
                "steps": steps,
                "samples": samples,
                "credit_samples": int(credit_samples),
                "shards": sampler.shard_count,
                "positions": sampler.total_positions,
                "samples_per_s": samples / max(elapsed, 1e-9),
            }
            metric_values = dataclasses.asdict(last_metrics)
            for key in (
                "loss",
                "policy_loss",
                "soft_policy_loss",
                "future_policy_loss",
                "wdl_loss",
                "blended_wdl_loss",
                "sf_move_loss",
                "sf_move_acc",
                "policy_own_acc_top1",
                "policy_future_acc_top1",
                "channel_balance",
            ):
                row[f"last_{key}"] = metric_values[key]
            print(json.dumps(row), flush=True)

        if save_every > 0 and steps % save_every == 0:
            trainer.save(run_dir / f"trainer_step_{steps:08d}.pt")
            trainer.save(run_dir / "trainer.pt")
            print(json.dumps({
                "event": "live_checkpoint",
                "candidate": candidate,
                "steps": steps,
                "path": str(run_dir / "trainer.pt"),
            }), flush=True)

        if eval_every > 0 and steps % eval_every == 0:
            eval_paths = _limit_shards(iter_shard_paths(args.replay_dir), args)
            eval_arrs = _load_eval_arrs(shard_paths=eval_paths, model_cfg=model_cfg, args=args)
            eval_metrics = trainer.eval_steps(
                _as_replay_buffer(_ArraySampler(eval_arrs, np.random.default_rng(int(args.seed) + 999 + steps))),
                batch_size=int(args.batch_size),
                steps=max(1, int(args.eval_steps)),
            )
            eval_row = {
                "event": "live_eval",
                "candidate": candidate,
                "steps": steps,
                **{f"eval_{k}": v for k, v in dataclasses.asdict(eval_metrics).items()},
            }
            print(json.dumps(eval_row), flush=True)
            # Plateau stop: static windows overfit past the eval minimum; the
            # bootstrap playbook says stop the phase there (then LR-drop / refresh).
            if plateau_evals > 0:
                cur = float(eval_metrics.loss)
                if seed_best_pending and best_eval_loss is None:
                    seed_best_pending = False
                    best_eval_loss = cur
                    print(json.dumps({
                        "event": "live_best_eval_seeded",
                        "candidate": candidate,
                        "steps": steps,
                        "eval_loss": cur,
                        "init_checkpoint": str(args.init_checkpoint),
                    }), flush=True)
                elif best_eval_loss is None or cur < best_eval_loss - plateau_min_delta:
                    best_eval_loss = cur
                    plateau_streak = 0
                    trainer.save(run_dir / "trainer_best_eval.pt")
                    trainer.save(run_dir / "trainer.pt")
                    # Sidecar so a later `continue` seeds its plateau bar from
                    # the loaded best's KNOWN loss instead of its own first
                    # eval (which, when worse, would let a merely-relatively-
                    # improving later eval overwrite the loaded best).
                    (run_dir / "trainer_best_eval.json").write_text(
                        json.dumps({"steps": steps, "eval_loss": cur}) + "\n",
                        encoding="utf-8")
                    print(json.dumps({
                        "event": "live_best_eval",
                        "candidate": candidate,
                        "steps": steps,
                        "eval_loss": cur,
                        "path": str(run_dir / "trainer_best_eval.pt"),
                    }), flush=True)
                else:
                    plateau_streak += 1
                    print(json.dumps({
                        "event": "live_plateau_tick",
                        "candidate": candidate,
                        "steps": steps,
                        "eval_loss": cur,
                        "best_eval_loss": best_eval_loss,
                        "plateau_streak": plateau_streak,
                        "plateau_evals": plateau_evals,
                    }), flush=True)
                    if plateau_streak >= plateau_evals:
                        stop_reason = "eval_plateau"
                        print(json.dumps({
                            "event": "live_follow_stop",
                            "candidate": candidate,
                            "reason": stop_reason,
                            "steps": steps,
                            "samples": samples,
                            "eval_loss": cur,
                            "best_eval_loss": best_eval_loss,
                        }), flush=True)
                        break

    if stop_reason == "running":
        if max_steps > 0 and steps >= max_steps:
            stop_reason = "max_steps"
        else:
            stop_reason = "completed"

    # On an eval-plateau stop the in-memory weights are from the NON-improving
    # tail: park them (forensics), then RESTORE the best checkpoint BEFORE the
    # final eval so results.jsonl / live_follow_done describe the same weights
    # the trainer.pt artifact holds (downstream `continue` loads trainer.pt).
    best_path = run_dir / "trainer_best_eval.pt"
    restored_best = False
    if stop_reason == "eval_plateau" and best_path.exists():
        trainer.save(run_dir / "trainer_final.pt")
        trainer.load(best_path)
        restored_best = True
    eval_paths = _limit_shards(iter_shard_paths(args.replay_dir), args)
    eval_arrs = _load_eval_arrs(shard_paths=eval_paths, model_cfg=model_cfg, args=args)
    eval_metrics = trainer.eval_steps(
        _as_replay_buffer(_ArraySampler(eval_arrs, np.random.default_rng(int(args.seed) + 999))),
        batch_size=int(args.batch_size),
        steps=max(1, int(args.eval_steps)),
    )
    trainer.save(run_dir / "trainer.pt")
    out = {
        "candidate": candidate,
        "optimizer": optimizer,
        "matrix_optimizer_scope": scope,
        "mode": "live_follow",
        "stop_reason": stop_reason,
        "restored_best_eval": restored_best,
        "matrix_lr_multiplier": float(args.matrix_lr_multiplier),
        "matrix_weight_decay": float(args.matrix_weight_decay),
        "aux_weight_decay": float(args.aux_weight_decay),
        "global_board_preprocess_lr_multiplier": float(args.global_board_preprocess_lr_multiplier),
        "global_board_preprocess_weight_decay": float(args.global_board_preprocess_weight_decay),
        "global_board_adapter_lr_multiplier": float(args.global_board_adapter_lr_multiplier),
        "global_board_adapter_weight_decay": float(args.global_board_adapter_weight_decay),
        "weight_decay_mode": str(args.weight_decay_mode),
        "soda_scope": str(args.soda_scope),
        "soda_start_step": int(args.soda_start_step),
        "aurora_uw_floor": float(args.aurora_uw_floor),
        "aurora_pp_iterations": int(args.aurora_pp_iterations),
        "aurora_pp_beta": float(args.aurora_pp_beta),
        "aurora_polar_steps": int(args.aurora_polar_steps),
        "aurora_polar_method": str(args.aurora_polar_method),
        "aurora_polar_dtype": str(args.aurora_polar_dtype),
        "aurora_polar_safety": float(args.aurora_polar_safety),
        "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else "",
        "steps": steps,
        "samples": samples,
        "elapsed_s": time.time() - t0,
        "model_config": dataclasses.asdict(model_cfg),
        **{f"eval_{k}": v for k, v in dataclasses.asdict(eval_metrics).items()},
    }
    print(json.dumps({
        "event": "live_follow_done",
        "candidate": candidate,
        "stop_reason": stop_reason,
        "steps": steps,
        "samples": samples,
        "eval_loss": float(eval_metrics.loss),
    }), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/bt4_aurora_asha.yaml")
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--out-dir", default="runs/offline_replay_epoch")
    ap.add_argument(
        "--candidates",
        nargs="+",
        default=["adamw", "aurora_mlp_only", "aurora_mlp_out", "aurora_mlp_attn_all", "aurora_blocks"],
    )
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-positions", type=int, default=2048)
    ap.add_argument("--eval-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--init-checkpoint",
        default="",
        help="Optional Trainer.save() checkpoint to load before streaming the replay epoch.",
    )
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument(
        "--w-future",
        type=float,
        default=None,
        help="Override the future-policy loss weight for this offline run.",
    )
    ap.add_argument(
        "--newest-shards",
        action="store_true",
        help="When --max-shards is set, use the newest shards instead of the oldest.",
    )
    ap.add_argument("--max-shards", type=int, default=0)
    ap.add_argument("--report-every-shards", type=int, default=50)
    ap.add_argument("--embed-dim", type=int, default=None)
    ap.add_argument(
        "--embed-dim-by-layer",
        type=str,
        default=None,
        help="Comma-separated embedding widths, one per transformer layer.",
    )
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--ffn-mult", type=float, default=None)
    ap.add_argument(
        "--ffn-mult-by-layer",
        type=str,
        default=None,
        help="Comma-separated FFN multipliers, one per layer; overrides uniform --ffn-mult per block.",
    )
    ap.add_argument("--smolgen", type=_parse_bool_choice, default=None, help="Override model.use_smolgen.")
    ap.add_argument("--smolgen-mode", choices=["shared", "per_layer"], default=None)
    ap.add_argument("--smolgen-pooling", choices=["flatten", "mean"], default=None)
    ap.add_argument("--smolgen-hidden-channels", type=int, default=None)
    ap.add_argument("--smolgen-hidden-sz", type=int, default=None)
    ap.add_argument("--smolgen-gen-sz", type=int, default=None)
    ap.add_argument("--smolgen-bias-scale", choices=["none", "layer", "layer_head"], default=None)
    ap.add_argument("--smolgen-bias-norm", choices=["none", "center", "center_rms"], default=None)
    ap.add_argument("--arc-attention-bias", choices=["none", "basic"], default=None)
    ap.add_argument("--smolgen-relation-basis", type=_parse_bool_choice, default=None)
    ap.add_argument(
        "--smolgen-relation-norm",
        choices=["none", "branch_center", "branch_center_rms", "basis_center", "basis_center_rms"],
        default=None,
    )
    ap.add_argument("--smolgen-relation-coeff-norm", choices=["none", "rms"], default=None)
    ap.add_argument("--smolgen-relation-scale", choices=["none", "layer", "layer_head"], default=None)
    ap.add_argument("--phase-output-adapter", type=_parse_bool_choice, default=None)
    ap.add_argument("--phase-output-adapter-dim", type=int, default=None)
    ap.add_argument("--phase-smolgen", type=_parse_bool_choice, default=None)
    ap.add_argument(
        "--phase-piece-thresholds",
        type=_parse_phase_piece_thresholds,
        default=None,
        help="Comma-separated end/mid root piece thresholds, e.g. 13,22.",
    )
    ap.add_argument("--input-pos-encoding", choices=["none", "arc", "arc_adapter"], default=None)
    ap.add_argument("--qkv-projection", choices=["fused", "split"], default=None)
    ap.add_argument("--use-qk-rmsnorm", type=_parse_bool_choice, default=None)
    ap.add_argument("--input-global-embedding", choices=["none", "bt4_board", "bt4_board_adapter"], default=None)
    ap.add_argument("--input-global-embedding-channels", type=int, default=None)
    ap.add_argument("--input-square-embedding", choices=["none", "add", "ma_gate"], default=None)
    ap.add_argument("--deepnorm", type=_parse_bool_choice, default=None, help="Override model.use_deepnorm.")
    ap.add_argument("--policy-encoding", default=None)
    ap.add_argument("--input-history-encoding", default=None)
    ap.add_argument(
        "--input-extra-features",
        default=None,
        help="Override model.input_extra_features (v1 / v2_threats / v3_checks). "
        "Sets the model's input-plane count and the on-load upgrade target.",
    )
    ap.add_argument(
        "--replay-upgrade-v1-planes",
        type=_parse_bool_choice,
        default=None,
        help="Recompute the extra-feature block on shard load instead of zero-padding "
        "narrower shards up to the model's plane count. Defaults to the config value.",
    )
    ap.add_argument(
        "--synthetic-lc0-root-history",
        action="store_true",
        help="Approximate LC0 root-history planes by remapping stored legacy replay x tensors.",
    )
    ap.add_argument(
        "--prefer-recorded-lc0-root",
        action="store_true",
        help="Use shard x_lc0_root as x when training an lc0_root input model and the field is present.",
    )
    ap.add_argument(
        "--lc0-root-legacy-meta",
        action="store_true",
        help=(
            "For lc0_root replay inputs, copy legacy normalized rule50 plane 102 "
            "to root plane 109 and legacy EP plane 100 to root plane 110."
        ),
    )
    ap.add_argument("--matrix-lr-multiplier", type=float, default=20.0)
    ap.add_argument("--matrix-weight-decay", type=float, default=1e-4)
    ap.add_argument("--aux-weight-decay", type=float, default=1e-4)
    ap.add_argument("--global-board-preprocess-lr-multiplier", type=float, default=1.0)
    ap.add_argument("--global-board-preprocess-weight-decay", type=float, default=0.0)
    ap.add_argument("--global-board-adapter-lr-multiplier", type=float, default=1.0)
    ap.add_argument("--global-board-adapter-weight-decay", type=float, default=0.0)
    ap.add_argument(
        "--global-board-adapter-init-rms-ratio",
        type=float,
        default=0.0,
        help=(
            "Fresh-init only: initialize bt4_board_adapter's final projection and rescale its "
            "output RMS to this fraction of pre-branch token RMS on the eval batch. 0 keeps "
            "the default exact zero-init no-op."
        ),
    )
    ap.add_argument(
        "--global-board-adapter-init-batch-size",
        type=int,
        default=512,
        help="Eval-batch positions used to calibrate --global-board-adapter-init-rms-ratio.",
    )
    ap.add_argument(
        "--weight-decay-mode",
        choices=["weight_decay", "soda"],
        default="weight_decay",
        help="Use ordinary optimizer weight decay, or replace nonzero decay groups with SODA averaging.",
    )
    ap.add_argument(
        "--soda-scope",
        choices=["decay", "hidden_matrix_only"],
        default="decay",
        help="SODA scope: preserve old nonzero-decay behavior, or force hidden matrix groups only.",
    )
    ap.add_argument(
        "--soda-start-step",
        type=int,
        default=0,
        help="Optimizer step at which SODA averaging starts; beta still uses the global step count.",
    )
    ap.add_argument("--aurora-uw-floor", type=float, default=0.0)
    ap.add_argument("--aurora-pp-iterations", type=int, default=2)
    ap.add_argument("--aurora-pp-beta", type=float, default=0.5)
    ap.add_argument("--aurora-polar-steps", type=int, default=12)
    ap.add_argument("--aurora-polar-method", choices=["simple", "polar_express"], default="simple")
    ap.add_argument("--aurora-polar-dtype", choices=["auto", "fp16", "bf16", "fp32"], default="auto")
    ap.add_argument("--aurora-polar-safety", type=float, default=1.01)
    ap.add_argument("--zclip-max-norm", default=None)
    ap.add_argument("--zclip-z-thresh", type=float, default=None)
    ap.add_argument("--zclip-clip-factor", type=float, default=None)
    ap.add_argument("--resid-channel-dropout", type=float, default=None)
    ap.add_argument("--resid-channel-balance-weight", type=float, default=None)
    ap.add_argument(
        "--live-follow",
        action="store_true",
        help="Continuously train by random sampling from a live replay shard directory instead of one fixed epoch.",
    )
    ap.add_argument(
        "--live-target-reuse",
        type=float,
        default=3.0,
        help="Training samples to spend per newly observed live replay position.",
    )
    ap.add_argument(
        "--live-initial-credit-frac",
        type=float,
        default=1.0,
        help="Initial live-window fraction credited at startup; use 0 to train only on future shard churn.",
    )
    ap.add_argument("--live-rescan-steps", type=int, default=100)
    ap.add_argument("--live-report-every-steps", type=int, default=50)
    ap.add_argument("--live-save-every-steps", type=int, default=1000)
    ap.add_argument("--live-eval-every-steps", type=int, default=1000)
    ap.add_argument("--live-max-steps", type=int, default=0, help="0 means run until stopped.")
    ap.add_argument("--live-min-positions", type=int, default=0)
    ap.add_argument("--live-idle-sleep", type=float, default=30.0)
    ap.add_argument(
        "--live-idle-exit-after",
        type=float,
        default=0.0,
        help=(
            "If >0, exit live-follow when credit is exhausted and no new "
            "positions appear for this many seconds (static/frozen pools). "
            "0 keeps the legacy wait-forever behavior for true live streams."
        ),
    )
    ap.add_argument(
        "--live-plateau-evals",
        type=int,
        default=0,
        help=(
            "If >0, stop the phase after this many consecutive live_eval "
            "checks without eval_loss improvement (static-window overfit stop)."
        ),
    )
    ap.add_argument(
        "--live-plateau-min-delta",
        type=float,
        default=1e-4,
        help="Minimum eval_loss improvement to reset the plateau counter.",
    )
    ap.add_argument("--live-cache-shards", type=int, default=16,
                    help="LRU shard cache; batches MIX across all cached shards, so this "
                         "is also the batch-decorrelation width (was 4 pre-mixing)")
    ap.add_argument(
        "--live-credit-cap-steps",
        type=int,
        default=2000,
        help="Maximum stored training credit in steps; <=0 disables the cap.",
    )
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()
    if float(args.global_board_adapter_init_rms_ratio) > 0.0:
        if args.init_checkpoint:
            raise SystemExit("--global-board-adapter-init-rms-ratio is only valid for fresh-init runs")
        if args.live_follow:
            raise SystemExit("--global-board-adapter-init-rms-ratio is not supported with --live-follow")

    cfg = flatten_run_config_defaults(load_yaml_file(args.config))
    cfg["policy_encoding"] = normalize_policy_encoding(args.policy_encoding or cfg.get("policy_encoding", POLICY_ENCODING_LC0_1858))
    if args.embed_dim is not None:
        cfg["embed_dim"] = int(args.embed_dim)
    if args.num_layers is not None:
        cfg["num_layers"] = int(args.num_layers)
    if args.num_heads is not None:
        cfg["num_heads"] = int(args.num_heads)
    if args.embed_dim_by_layer is not None:
        try:
            schedule = _parse_embed_dim_by_layer(str(args.embed_dim_by_layer))
        except argparse.ArgumentTypeError as exc:
            raise SystemExit(f"--embed-dim-by-layer: {exc}") from exc
        if schedule is None:
            cfg.pop("embed_dim_by_layer", None)
        else:
            cfg["embed_dim_by_layer"] = schedule
    if args.ffn_mult is not None:
        cfg["ffn_mult"] = float(args.ffn_mult)
    if args.ffn_mult_by_layer is not None:
        try:
            schedule = _parse_ffn_mult_by_layer(str(args.ffn_mult_by_layer))
        except argparse.ArgumentTypeError as exc:
            raise SystemExit(f"--ffn-mult-by-layer: {exc}") from exc
        if schedule is None:
            cfg.pop("ffn_mult_by_layer", None)
        else:
            cfg["ffn_mult_by_layer"] = schedule
    if args.input_history_encoding is not None:
        cfg["input_history_encoding"] = str(args.input_history_encoding)
    if args.input_extra_features is not None:
        cfg["input_extra_features"] = str(args.input_extra_features)
    if args.replay_upgrade_v1_planes is None:
        args.replay_upgrade_v1_planes = bool(cfg.get("replay_upgrade_v1_planes", False))
    if args.smolgen is not None:
        cfg["use_smolgen"] = bool(args.smolgen)
    if args.smolgen_mode is not None:
        cfg["smolgen_mode"] = str(args.smolgen_mode)
    if args.smolgen_pooling is not None:
        cfg["smolgen_pooling"] = str(args.smolgen_pooling)
    if args.smolgen_hidden_channels is not None:
        cfg["smolgen_hidden_channels"] = int(args.smolgen_hidden_channels)
    if args.smolgen_hidden_sz is not None:
        cfg["smolgen_hidden_sz"] = int(args.smolgen_hidden_sz)
    if args.smolgen_gen_sz is not None:
        cfg["smolgen_gen_sz"] = int(args.smolgen_gen_sz)
    if args.smolgen_bias_scale is not None:
        cfg["smolgen_bias_scale"] = str(args.smolgen_bias_scale)
    if args.smolgen_bias_norm is not None:
        cfg["smolgen_bias_norm"] = str(args.smolgen_bias_norm)
    if args.arc_attention_bias is not None:
        cfg["arc_attention_bias"] = str(args.arc_attention_bias)
    if args.smolgen_relation_basis is not None:
        cfg["smolgen_relation_basis"] = bool(args.smolgen_relation_basis)
    if args.smolgen_relation_norm is not None:
        cfg["smolgen_relation_norm"] = str(args.smolgen_relation_norm)
    if args.smolgen_relation_coeff_norm is not None:
        cfg["smolgen_relation_coeff_norm"] = str(args.smolgen_relation_coeff_norm)
    if args.smolgen_relation_scale is not None:
        cfg["smolgen_relation_scale"] = str(args.smolgen_relation_scale)
    if args.phase_output_adapter is not None:
        cfg["phase_output_adapter"] = bool(args.phase_output_adapter)
    if args.phase_output_adapter_dim is not None:
        cfg["phase_output_adapter_dim"] = int(args.phase_output_adapter_dim)
    if args.phase_smolgen is not None:
        cfg["phase_smolgen"] = bool(args.phase_smolgen)
    if args.phase_piece_thresholds is not None:
        cfg["phase_piece_thresholds"] = args.phase_piece_thresholds
    if args.input_pos_encoding is not None:
        cfg["input_pos_encoding"] = str(args.input_pos_encoding)
    if args.qkv_projection is not None:
        cfg["qkv_projection"] = str(args.qkv_projection)
    if args.use_qk_rmsnorm is not None:
        cfg["use_qk_rmsnorm"] = bool(args.use_qk_rmsnorm)
    if args.input_global_embedding is not None:
        cfg["input_global_embedding"] = str(args.input_global_embedding)
    if args.input_global_embedding_channels is not None:
        cfg["input_global_embedding_channels"] = int(args.input_global_embedding_channels)
    if args.input_square_embedding is not None:
        cfg["input_square_embedding"] = str(args.input_square_embedding)
    if args.deepnorm is not None:
        cfg["use_deepnorm"] = bool(args.deepnorm)
    if args.lr is not None:
        cfg["lr"] = float(args.lr)
    if args.w_future is not None:
        cfg["w_future"] = float(args.w_future)
    if args.resid_channel_dropout is not None:
        cfg["resid_channel_dropout"] = float(args.resid_channel_dropout)
    if args.resid_channel_balance_weight is not None:
        cfg["resid_channel_balance_weight"] = float(args.resid_channel_balance_weight)
    cfg["global_board_preprocess_lr_multiplier"] = float(args.global_board_preprocess_lr_multiplier)
    cfg["global_board_preprocess_weight_decay"] = float(args.global_board_preprocess_weight_decay)
    cfg["global_board_adapter_lr_multiplier"] = float(args.global_board_adapter_lr_multiplier)
    cfg["global_board_adapter_weight_decay"] = float(args.global_board_adapter_weight_decay)
    model_cfg = _model_config_from_offline_config(cfg)

    shard_paths = _limit_shards(iter_shard_paths(args.replay_dir), args)
    if not shard_paths:
        raise SystemExit(f"no replay shards found in {args.replay_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    print(
        json.dumps({
            "event": "start",
            "shards": len(shard_paths),
            "candidates": args.candidates,
            "policy_encoding": model_cfg.policy_encoding,
            "input_history_encoding": model_cfg.input_history_encoding,
            "embed_dim": model_cfg.embed_dim,
            "embed_dim_by_layer": (
                list(model_cfg.embed_dim_by_layer)
                if model_cfg.embed_dim_by_layer is not None
                else None
            ),
            "ffn_mult": model_cfg.ffn_mult,
            "ffn_mult_by_layer": (
                list(model_cfg.ffn_mult_by_layer)
                if model_cfg.ffn_mult_by_layer is not None
                else None
            ),
            "input_pos_encoding": model_cfg.input_pos_encoding,
            "qkv_projection": model_cfg.qkv_projection,
            "use_qk_rmsnorm": model_cfg.use_qk_rmsnorm,
            "input_global_embedding": model_cfg.input_global_embedding,
            "input_global_embedding_channels": model_cfg.input_global_embedding_channels,
            "input_square_embedding": model_cfg.input_square_embedding,
            "use_smolgen": model_cfg.use_smolgen,
            "smolgen_mode": model_cfg.smolgen_mode,
            "smolgen_pooling": model_cfg.smolgen_pooling,
            "smolgen_hidden_channels": model_cfg.smolgen_hidden_channels,
            "smolgen_hidden_sz": model_cfg.smolgen_hidden_sz,
            "smolgen_gen_sz": model_cfg.smolgen_gen_sz,
            "smolgen_bias_scale": model_cfg.smolgen_bias_scale,
            "smolgen_bias_norm": model_cfg.smolgen_bias_norm,
            "arc_attention_bias": model_cfg.arc_attention_bias,
            "smolgen_relation_basis": model_cfg.smolgen_relation_basis,
            "smolgen_relation_norm": model_cfg.smolgen_relation_norm,
            "smolgen_relation_coeff_norm": model_cfg.smolgen_relation_coeff_norm,
            "smolgen_relation_scale": model_cfg.smolgen_relation_scale,
            "phase_output_adapter": model_cfg.phase_output_adapter,
            "phase_output_adapter_dim": model_cfg.phase_output_adapter_dim,
            "phase_smolgen": model_cfg.phase_smolgen,
            "phase_piece_thresholds": list(model_cfg.phase_piece_thresholds),
            "use_deepnorm": model_cfg.use_deepnorm,
            "synthetic_lc0_root_history": bool(args.synthetic_lc0_root_history),
            "prefer_recorded_lc0_root": bool(args.prefer_recorded_lc0_root),
            "lc0_root_legacy_meta": bool(args.lc0_root_legacy_meta),
            "input_extra_features": model_cfg.input_extra_features,
            "input_planes": input_plane_count(model_cfg.input_extra_features),
            "replay_upgrade_v1_planes": bool(args.replay_upgrade_v1_planes),
            "weight_decay_mode": str(args.weight_decay_mode),
            "soda_scope": str(args.soda_scope),
            "soda_start_step": int(args.soda_start_step),
            "aurora_pp_iterations": int(args.aurora_pp_iterations),
            "aurora_pp_beta": float(args.aurora_pp_beta),
            "aurora_polar_steps": int(args.aurora_polar_steps),
            "aurora_polar_method": str(args.aurora_polar_method),
            "aurora_polar_dtype": str(args.aurora_polar_dtype),
            "aurora_polar_safety": float(args.aurora_polar_safety),
            "global_board_preprocess_lr_multiplier": float(args.global_board_preprocess_lr_multiplier),
            "global_board_preprocess_weight_decay": float(args.global_board_preprocess_weight_decay),
            "global_board_adapter_lr_multiplier": float(args.global_board_adapter_lr_multiplier),
            "global_board_adapter_weight_decay": float(args.global_board_adapter_weight_decay),
            "global_board_adapter_init_rms_ratio": float(args.global_board_adapter_init_rms_ratio),
            "global_board_adapter_init_batch_size": int(args.global_board_adapter_init_batch_size),
            "resid_channel_dropout": float(cfg.get("resid_channel_dropout", 0.0) or 0.0),
            "resid_channel_balance_weight": float(cfg.get("resid_channel_balance_weight", 0.0) or 0.0),
            "w_future": float(cfg.get("w_future", 0.15)),
            "live_follow": bool(args.live_follow),
            "live_target_reuse": float(args.live_target_reuse),
            "live_initial_credit_frac": float(args.live_initial_credit_frac),
            "results": str(results_path),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else "",
        }),
        flush=True,
    )
    with results_path.open("a", encoding="utf-8") as fh:
        for candidate in args.candidates:
            if args.live_follow:
                row = _train_candidate_live_follow(
                    candidate=candidate,
                    cfg=cfg,
                    model_cfg=model_cfg,
                    args=args,
                )
            else:
                eval_arrs = _load_eval_arrs(shard_paths=shard_paths, model_cfg=model_cfg, args=args)
                row = _train_candidate(
                    candidate=candidate,
                    cfg=cfg,
                    shard_paths=shard_paths,
                    eval_arrs=eval_arrs,
                    args=args,
                )
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            print(json.dumps({"event": "candidate_done", **row}), flush=True)


if __name__ == "__main__":
    main()
