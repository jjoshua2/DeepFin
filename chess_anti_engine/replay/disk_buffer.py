"""Disk-backed replay buffer with array-backed hot shuffle storage.

Stores training data as NPZ shards on disk and keeps a small hot subset in
memory for efficient sampling. The hot path stays columnar/array-backed so the
trainer can batch directly from NumPy arrays instead of inflating tens of
thousands of ``ReplaySample`` Python objects.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from pathlib import Path

import numpy as np

from chess_anti_engine.encoding import version_for_input_planes
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS

from .buffer import ReplaySample
from .shard import (
    HISTORY_REP_FIX_ARRAY_KEY,
    INPUT_HISTORY_ENCODING_ARRAY_KEY,
    POLICY_ENCODING_ARRAY_KEY,
    _OPTIONAL_STORAGE_PAIRS,
    _SHARD_FIELDS,
    arrays_to_samples,
    delete_shard_path,
    densify_chunk,
    iter_shard_paths,
    load_shard_arrays,
    local_shard_path,
    prune_storage_arrays,
    samples_to_arrays,
    save_local_shard_arrays,
    shard_index,
    shard_positions,
    sparsify_chunk,
    zeros_for_storage_field,
)
from .threat_upgrade import (
    V1_INPUT_PLANES,
    upgrade_arrays_to_planes,
)
import contextlib

log = logging.getLogger(__name__)

_ARRAY_FIELD_ORDER = _SHARD_FIELDS
_SCALAR_METADATA_FIELDS = (
    INPUT_HISTORY_ENCODING_ARRAY_KEY,
    POLICY_ENCODING_ARRAY_KEY,
    # Same encoding name can carry different repetition planes; refuse to
    # concatenate legacy and rep-fix chunks into one training window.
    HISTORY_REP_FIX_ARRAY_KEY,
)

# Lookup: value field name → has-flag field name (derived from shard.py's canonical pairs).
_VALUE_TO_FLAG = dict(_OPTIONAL_STORAGE_PAIRS)


def _batch_dims(arrs: dict[str, np.ndarray]) -> tuple[int, int, int]:
    x = arrs["x"]
    policy_target = arrs["policy_target"]
    policy_size = int(np.asarray(arrs.get("_policy_size", policy_target.shape[1])).item())
    return int(x.shape[0]), policy_size, int(x.shape[1])


def _slice_array_batch(arrs: dict[str, np.ndarray], idxs: np.ndarray) -> dict[str, np.ndarray]:
    ii = np.asarray(idxs, dtype=np.int64).reshape(-1)
    return {
        k: (np.array(v, copy=True) if np.asarray(v).ndim == 0 else np.array(v[ii], copy=True, order="C"))
        for k, v in arrs.items()
    }


def _concat_sparse_batches(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ValueError("no chunks to concatenate")
    if len(chunks) == 1:
        return chunks[0]
    policy_sizes = {int(_batch_dims(chunk)[1]) for chunk in chunks}
    if len(policy_sizes) > 1:
        raise ValueError(f"mixed replay policy sizes cannot be concatenated: {sorted(policy_sizes)}")

    categorical_bins = next(
        (int(c["categorical_target"].shape[1]) for c in chunks if "categorical_target" in c),
        DEFAULT_CATEGORICAL_BINS,
    )

    out: dict[str, np.ndarray] = {}
    for name in _ARRAY_FIELD_ORDER:
        parts: list[np.ndarray] = []
        for chunk in chunks:
            n, chunk_policy_size, chunk_x_planes = _batch_dims(chunk)
            if name in chunk:
                parts.append(np.asarray(chunk[name]))
            else:
                parts.append(
                    zeros_for_storage_field(
                        name,
                        n=n,
                        policy_size=chunk_policy_size,
                        x_planes=chunk_x_planes,
                        categorical_bins=categorical_bins,
                    )
                )
        merged = np.concatenate(parts, axis=0)
  # Keep required fields explicit; drop optional fields that are uniformly absent.
        if any(name in chunk for chunk in chunks) or name in ("x", "policy_target", "wdl_target", "priority", "has_policy"):
            out[name] = merged
            continue
  # Value arrays are only needed when the corresponding has_* flag is present.
        flag_name = _VALUE_TO_FLAG.get(name)
        if flag_name is not None and flag_name in out:
            out[name] = merged
    for name in _SCALAR_METADATA_FIELDS:
        values: list[str | None] = []
        for chunk in chunks:
            if name not in chunk:
                values.append(None)
                continue
            arr = np.asarray(chunk[name])
            values.append(str(arr.reshape(-1)[0]) if arr.size else None)
        concrete = [value for value in values if value is not None]
        if not concrete:
            continue
        some_missing = len(concrete) != len(values)
        disagree = any(value != concrete[0] for value in concrete)
        if not (some_missing or disagree):
            out[name] = np.asarray(concrete[0])
            continue
        # Mixed across shards. rep-fix is benign to mix (the corrected lc0-root
        # repetition planes differ from legacy on only ~0.2% of positions, and
        # the change is net-neutral), unlike the shape/semantic encodings. A
        # window straddling a rep-fix rollout (old shards reload "false", new
        # ones "true") would otherwise crash the buffer here. Tolerate it,
        # resolving toward "true" since the run is migrating to fixed planes,
        # and log. NOTE this DOES durably relabel the few legacy rows as "true"
        # in the re-written window shard — acceptable because the mislabel is
        # confined to the transition window (which ages out) and rep-fix is
        # net-neutral; a hard-fail here would instead crash a live restart.
        if name == HISTORY_REP_FIX_ARRAY_KEY:
            label_set = sorted({"<missing>" if v is None else v for v in values})
            log.warning("replay window mixes %s across shards (%s) — tolerating (rep-fix is benign)",
                        name, label_set)
            out[name] = np.asarray("true" if "true" in concrete else concrete[0])
            continue
        labels = sorted({"<missing>" if value is None else str(value) for value in values})
        raise ValueError(f"mixed replay metadata {name}: {labels}")
    return out


class DiskReplayBuffer:
    """Disk-backed replay buffer with array-backed hot shuffle storage."""

    def __init__(
        self,
        capacity: int,
        *,
        shard_dir: Path,
        rng: np.random.Generator,
        shuffle_cap: int = 20_000,
        shard_size: int = 1000,
        refresh_interval: int = 5,
        refresh_shards: int = 3,
        draw_cap_frac: float = 0.90,
        wl_max_ratio: float = 1.5,
        input_planes: int | None = None,
        upgrade_v1_planes: bool = False,
        sf_gap_priority_weight: float = 0.0,
        sf_gap_priority_signed: bool = False,
        fast_low_surprise_priority: float = 1.0,
        diff_focus_pol_scale: float = 0.0,
        diff_focus_q_weight: float = 0.0,
    ):
        self.capacity = int(capacity)
        self.rng = rng
  # When set, shards stored with FEWER x planes (e.g. v1 146-plane shards
  # in a v2_threats 175-plane run) are normalized on load — never
  # re-encoded. With upgrade_v1_planes the 29 threat planes are
  # recomputed from the stored planes (replay/threat_upgrade.py);
  # otherwise (or if a chunk fails the recompute's validation) the
  # missing planes are zero-padded, matching the feature_dropout_p
  # convention of zeroing the whole extra block.
        self._input_planes = None if input_planes is None else int(input_planes)
        self._upgrade_v1_planes = bool(upgrade_v1_planes)
        self._upgrade_failures = 0
        self._shard_dir = Path(shard_dir)
        self._shard_dir.mkdir(parents=True, exist_ok=True)

        self._shuffle_cap = int(shuffle_cap)
        self._shard_size = int(shard_size)
        self._refresh_interval = int(refresh_interval)
        self._refresh_shards = int(refresh_shards)
        self._prefetch_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
        self._prefetch_lock = threading.Lock()
        self._prefetch_request = threading.Event()
        self._prefetch_stop = threading.Event()
        self._prefetch_inflight = False
        self._prefetched_refresh: list[dict[str, np.ndarray]] | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_generation = 0

  # In-memory hot replay as chunked sparse arrays with logical front offsets.
  # Sampling stays random over the hot pool while trims avoid front-copy churn.
        self._shuffle_buf: deque[dict[str, np.ndarray]] = deque()
        self._shuffle_sizes: deque[int] = deque()
        self._shuffle_offsets: deque[int] = deque()
        self._shuffle_size_total = 0
        self._shuffle_priority_store = np.zeros((0,), dtype=np.float32)
        self._shuffle_wdl_store = np.zeros((0,), dtype=np.int8)
        self._shuffle_head = 0

  # Write buffer: chunked arrays that accumulate until shard_size, then flush.
        self._write_buf: list[dict[str, np.ndarray]] = []
        self._write_buf_sizes: list[int] = []
        self._write_buf_rows = 0

  # Disk shard tracking (ordered oldest-first).
        self._shard_paths: deque[Path] = deque()
        self._shard_sizes: deque[int] = deque()
        self._total_positions = 0
        self._shard_index = 0

  # Counter for scheduling shuffle buffer refreshes.
        self._sample_count = 0

  # KataGo-style surprise weighting: 50% uniform, 50% priority.
        self.surprise_mix = 0.5
  # Surprise-priority shaping applied when rows enter the shuffle buffer
  # (sampling frequency only — the stored shard arrays are untouched).
  # Defaults are no-ops. Must be set BEFORE the resume shard scan below so
  # the seeded hot pool honors configured shaping (Codex P2 on PR #104);
  # the tune trainable additionally pushes live values each iteration.
        self.sf_gap_priority_weight = float(sf_gap_priority_weight)
  # Downside-only (signed) gap: when True, boost by max(search_sig - sf_sig, 0)
  # recomputed from the stored WDL arrays, i.e. ONLY rows where the net's own
  # search over-values the position vs SF (the over-optimism / collapse
  # direction). The default abs() gap is symmetric and wastes half its mass on
  # the net-pessimistic direction, which trains the net MORE optimistic — the
  # opposite of what the value blind spot needs. Offline-experiment knob.
        self.sf_gap_priority_signed = bool(sf_gap_priority_signed)
        self.fast_low_surprise_priority = float(fast_low_surprise_priority)
  # Selfplay diff-focus weights, mirrored here ONLY for priority-mass
  # telemetry (decomposing stored priorities back into kl/qd term mass);
  # they do not alter sampling. Pushed live alongside the shaping knobs.
        self.diff_focus_pol_scale = float(diff_focus_pol_scale)
        self.diff_focus_q_weight = float(diff_focus_q_weight)
        self._pmass = self._pmass_zero()
        self.draw_cap_frac = float(draw_cap_frac)
        self.wl_max_ratio = float(wl_max_ratio)

  # Scan existing shards on disk (for resume).
        self._scan_existing_shards()
  # The seed scan appends to the shuffle pool through the same path as live
  # ingest; drop that mass so the first pop reports only the first
  # iteration's appends (Codex P2 on PR #106).
        self._pmass = self._pmass_zero()
        self._ensure_prefetch_thread()
        self._schedule_refresh_prefetch()

    def _snapshot_shards(self) -> list[Path]:
        with self._prefetch_lock:
            return list(self._shard_paths)

    def _append_shard_record(self, path: Path, n: int) -> None:
        with self._prefetch_lock:
            self._shard_paths.append(path)
            self._shard_sizes.append(int(n))
            self._total_positions += int(n)

    def _pop_oldest_shard_record(self) -> tuple[Path, int] | None:
        with self._prefetch_lock:
            if not self._shard_paths:
                return None
            oldest = self._shard_paths.popleft()
            n = int(self._shard_sizes.popleft())
            self._total_positions -= n
            return oldest, n

    def _clear_shard_records(self) -> list[Path]:
        with self._prefetch_lock:
            out = list(self._shard_paths)
            self._shard_paths = deque()
            self._shard_sizes = deque()
            self._total_positions = 0
            return out

    def _tracked_shard_positions(self) -> int:
        with self._prefetch_lock:
            return int(self._total_positions)

    def _effective_shuffle_cap(self) -> int:
        return max(1, min(int(self._shuffle_cap), int(self.capacity)))

    def _shuffle_len(self) -> int:
        return int(self._shuffle_size_total)

    def _upgrade_target_version(self) -> str | None:
        """Extra-features version to upgrade stored chunks to, or None.

        Returns the registered version whose total input-plane count equals
        the configured ``_input_planes`` (v2_threats=175, v3_checks=179, …)
        when it exceeds the v1 base — i.e. the configured width carries an
        extra-feature block worth recomputing. A v1 (146) or unrecognized
        width disables the recompute path (the plane-count guard in
        ``_pad_x_planes`` still handles those).
        """
        target = self._input_planes
        if target is None or int(target) <= V1_INPUT_PLANES:
            return None
        try:
            return version_for_input_planes(int(target))
        except ValueError:
            return None

    def _upgrade_x_planes(self, arrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Recompute the extra-feature planes for a stored chunk (opt-in).

        Generalizes the v1→v2_threats upgrade to ANY cross-version width
        mismatch (e.g. v3_checks 179 → v3_xray 181): the full extra block is
        recomputed from the stored 112 LC0 base planes whenever the stored
        width differs from the target and the target is a recognized version.
        This is the only correct cross-version path — zero-padding a stored
        block of a DIFFERENT version into the target's extra slots is silent
        input corruption (the stored extra planes are never a prefix of the
        target's beyond the 34 v1 planes). Stored-width == target normally
        passes through untouched, but a chunk the pre-upgrade zero-pad path
        persisted at the target width with an all-zero extra block is repaired
        in place (recomputed only for those rows) — covering both 175-plane v2
        chunks and any wider recognized version.
        Validation failure (a chunk the decoder can't faithfully reconstruct)
        falls back to the zero-pad path in ``_pad_x_planes`` instead of killing
        the prefetch thread — old data trains as before, just without
        recomputed extra-plane signal for that chunk.
        """
        if not self._upgrade_v1_planes or self._input_planes is None:
            return arrs
        target_planes = int(self._input_planes)
        version = self._upgrade_target_version()
        if version is None:
            return arrs
        stored = int(np.asarray(arrs["x"]).shape[1])
        if stored > target_planes:
            # Wider-than-target is a config error handled (raised) by
            # _pad_x_planes after this returns.
            return arrs
        # Equal width is NOT an unconditional passthrough: a chunk persisted at
        # the target width by the pre-upgrade zero-pad path carries an all-zero
        # extra block that must be repaired. upgrade_arrays_to_planes recomputes
        # only those rows (native chunks pass through cheaply via its all-zero
        # gate), so route equal-width through it too. stored < target_planes
        # gets the full cross-version recompute.
        try:
            upgraded, _stats = upgrade_arrays_to_planes(arrs, target_planes)
        except ValueError as exc:
            self._upgrade_failures += 1
            # Log the first few in full, then a periodic cumulative count so a
            # systematic decode failure stays visible instead of going quiet.
            if self._upgrade_failures <= 3 or self._upgrade_failures % 50 == 0:
                print(
                    f"[replay] {version} plane upgrade failed "
                    f"({self._upgrade_failures} chunk(s) so far), zero-padding: {exc}"
                )
            return arrs
        return upgraded

    def _pad_x_planes(self, arrs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Normalize stored x (and x_lc0_root) up to the configured plane count.

        Gated by the stored channel count: older shards encoded with fewer
        extra-feature planes are upgraded (when enabled) or gain zero planes
        at the end of the extra block; shards with MORE planes than the
        model expects are a config error.
        """
        target = self._input_planes
        if target is None:
            return arrs
        out = self._upgrade_x_planes(arrs)
        for key in ("x", "x_lc0_root"):
            v = out.get(key)
            if v is None:
                continue
            stored = int(np.asarray(v).shape[1])
            if stored == target:
                continue
            if stored > target:
                raise ValueError(
                    f"shard stores {stored} input planes but the model expects "
                    f"{target}; refusing to truncate encoded inputs"
                )
            pad = np.zeros(
                (v.shape[0], target - stored, *v.shape[2:]), dtype=v.dtype,
            )
            if out is arrs:
                out = dict(arrs)
            out[key] = np.concatenate([np.asarray(v), pad], axis=1)
        return out

    def _append_shuffle_arrays(self, arrs: dict[str, np.ndarray]) -> None:
        arrs = self._pad_x_planes(arrs)
        n, _, _ = _batch_dims(arrs)
        if n <= 0:
            return
        self._shuffle_buf.append(sparsify_chunk(arrs))
        self._shuffle_sizes.append(n)
        self._shuffle_offsets.append(0)
        self._shuffle_size_total += n
        raw_pri = np.asarray(arrs["priority"], dtype=np.float32)
        bad = ~np.isfinite(raw_pri)
        if bad.any():
            raw_pri = raw_pri.copy()
            raw_pri[bad] = 1.0
        raw_pri = self._shape_shuffle_priority(arrs, raw_pri)
        self._accumulate_priority_mass(arrs, raw_pri)
        self._shuffle_priority_store = np.concatenate(
            [self._shuffle_priority_store, raw_pri], axis=0,
        )
        self._shuffle_wdl_store = np.concatenate(
            [self._shuffle_wdl_store, np.asarray(arrs["wdl_target"], dtype=np.int8)],
            axis=0,
        )

    def _gap_boost_and_mask(
        self, arrs: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Per-row SF-gap boost value and its validity mask, or None if the
        columns this mode needs are absent (skip — mirroring the abs path,
        which also skips a chunk without its column).

        Signed mode boosts ONLY the over-optimism direction —
        ``max(search_sig - sf_sig, 0)`` recomputed from the stored WDL columns —
        so it must gate on sf_wdl/search_wdl, NOT the stored abs-gap column (the
        two co-occur on finalize-produced shards, but a WDL-only chunk must
        still be shaped). Abs mode boosts the stored symmetric
        ``priority_sf_search_gap``. Shared by ``_shape_shuffle_priority`` and the
        priority-mass telemetry so the reported ``gap_term`` always matches the
        mass actually injected.
        """
        if self.sf_gap_priority_signed:
            if not (
                "sf_wdl" in arrs and "has_sf_wdl" in arrs
                and "search_wdl" in arrs and "has_search_wdl" in arrs
            ):
                return None
            sf = np.asarray(arrs["sf_wdl"], dtype=np.float32)
            sw = np.asarray(arrs["search_wdl"], dtype=np.float32)
            has = (np.asarray(arrs["has_sf_wdl"], dtype=bool)
                   & np.asarray(arrs["has_search_wdl"], dtype=bool))
            boost = (sw[:, 0] - sw[:, 2]) - (sf[:, 0] - sf[:, 2])
        else:
            if not (
                "priority_sf_search_gap" in arrs
                and "has_priority_sf_search_gap" in arrs
            ):
                return None
            has = np.asarray(arrs["has_priority_sf_search_gap"], dtype=bool)
            boost = np.asarray(arrs["priority_sf_search_gap"], dtype=np.float32)
        # copy=True: never mutate the caller's arrs in place.
        boost = np.nan_to_num(boost, nan=0.0, posinf=0.0, neginf=0.0)
        return boost, has

    def _shape_shuffle_priority(self, arrs: dict[str, np.ndarray], pri: np.ndarray) -> np.ndarray:
        """Surprise-priority shaping for the sampling store.

        Full rows: boost by the stored ``priority_sf_search_gap`` (|SF value
        label − own search value|). The diff-focus priority compares the net
        to its own search, so it reads ~0 exactly where both are blind
        together; the SF gap is the external signal that still sees those
        rows.
        Fast (value-only) rows arrive with a neutral priority of 1.0; demote
        them to ``fast_low_surprise_priority`` unless the recorded 32-sim
        search read missed the game outcome by a full point (sign-flip
        surprise) — keeps low-information rows from crowding the surprise
        half of sampling while the uniform half preserves coverage.
        """
        w_gap = float(self.sf_gap_priority_weight)
        fast_low = float(self.fast_low_surprise_priority)
        gap = self._gap_boost_and_mask(arrs) if w_gap > 0.0 else None
        shape_fast = (
            fast_low < 1.0
            and "search_wdl" in arrs
            and "has_search_wdl" in arrs
            and "has_policy" in arrs
        )
        if gap is None and not shape_fast:
            return pri
        pri = pri.copy()
        if gap is not None:
            # Over-optimism-only boost: signed (downside) gap when
            # sf_gap_priority_signed, else the stored symmetric abs gap.
            gap_boost, gap_has = gap
            pri[gap_has] += w_gap * np.maximum(gap_boost[gap_has], 0.0)
        if shape_fast:
            hp = np.asarray(arrs["has_policy"], dtype=bool)
            hs = np.asarray(arrs["has_search_wdl"], dtype=bool)
            sw = np.asarray(arrs["search_wdl"], dtype=np.float32)
            z = np.asarray(arrs["wdl_target"])
            z_q = np.where(z == 0, 1.0, np.where(z == 2, -1.0, 0.0)).astype(np.float32)
            zgap = np.abs(z_q - (sw[:, 0] - sw[:, 2]))
            low = (~hp) & hs & (zgap < 1.0)
            pri[low] = np.minimum(pri[low], fast_low)
        return pri

    @staticmethod
    def _pmass_zero() -> dict[str, float]:
        return {
            "rows_full": 0.0, "rows_fast": 0.0, "rows_fast_demoted": 0.0,
            "kl_term": 0.0, "qd_term": 0.0, "gap_term": 0.0,
            "eff_full": 0.0, "eff_fast": 0.0,
        }

    def _accumulate_priority_mass(self, arrs: dict[str, np.ndarray], pri: np.ndarray) -> None:
        """Telemetry: decompose appended sampling-priority mass by source term.

        Uses the mirrored diff-focus weights to reconstruct the kl/qd term
        mass baked into stored priorities at selfplay time, plus the gap
        boost and fast-row demotion applied here at append time. Popped once
        per iteration by the tune trainable (pop_priority_mass_stats).
        """
        pm = self._pmass
        if "has_policy" in arrs:
            hp = np.asarray(arrs["has_policy"], dtype=bool)
        else:
            hp = np.ones(pri.shape[0], dtype=bool)
        pm["rows_full"] += float(hp.sum())
        pm["rows_fast"] += float((~hp).sum())
        pm["eff_full"] += float(pri[hp].sum())
        pm["eff_fast"] += float(pri[~hp].sum())
  # Fast rows can carry kl/qd fields, but finalize neutralizes their stored
  # sampling priority — mask by has_policy so term mass matches what the
  # effective priorities actually contain.
        if self.diff_focus_pol_scale > 0 and "priority_policy_kl" in arrs and "has_priority_policy_kl" in arrs:
            has = np.asarray(arrs["has_priority_policy_kl"], dtype=bool) & hp
            kl = np.nan_to_num(np.asarray(arrs["priority_policy_kl"], dtype=np.float64), nan=0.0)
            pm["kl_term"] += float(np.maximum(kl[has], 0.0).sum()) * self.diff_focus_pol_scale
        if self.diff_focus_q_weight > 0 and "priority_q_delta" in arrs and "has_priority_q_delta" in arrs:
            has = np.asarray(arrs["has_priority_q_delta"], dtype=bool) & hp
            qd = np.nan_to_num(np.asarray(arrs["priority_q_delta"], dtype=np.float64), nan=0.0)
            pm["qd_term"] += float(np.abs(qd[has]).sum()) * self.diff_focus_q_weight
        if self.sf_gap_priority_weight > 0:
            # Same (signed or abs) boost the shaper applies, so gap_term reflects
            # the mass actually injected — not the symmetric abs sum in signed mode.
            gap = self._gap_boost_and_mask(arrs)
            if gap is not None:
                gap_boost, gap_has = gap
                pm["gap_term"] += (
                    float(np.maximum(gap_boost[gap_has], 0.0).sum())
                    * self.sf_gap_priority_weight
                )
        if (
            self.fast_low_surprise_priority < 1.0
            and "search_wdl" in arrs
            and "has_search_wdl" in arrs
            and "has_policy" in arrs
        ):
            hs = np.asarray(arrs["has_search_wdl"], dtype=bool)
            sw = np.asarray(arrs["search_wdl"], dtype=np.float64)
            z = np.asarray(arrs["wdl_target"])
            z_q = np.where(z == 0, 1.0, np.where(z == 2, -1.0, 0.0))
            zgap = np.abs(z_q - (sw[:, 0] - sw[:, 2]))
            pm["rows_fast_demoted"] += float(((~hp) & hs & (zgap < 1.0)).sum())

    def pop_priority_mass_stats(self) -> dict[str, float]:
        """Return and reset per-iteration priority-mass telemetry.

        Shares are of the TOTAL effective (post-shaping) priority mass
        appended to the shuffle buffer since the last pop — i.e. what the
        surprise half of sampling actually sees.
        """
        pm, self._pmass = self._pmass, self._pmass_zero()
        total = pm["eff_full"] + pm["eff_fast"]
        n_fast = pm["rows_fast"]
        return {
            "replay_pmass_rows_full": pm["rows_full"],
            "replay_pmass_rows_fast": pm["rows_fast"],
            "replay_pmass_kl_share": pm["kl_term"] / total if total > 0 else 0.0,
            "replay_pmass_qd_share": pm["qd_term"] / total if total > 0 else 0.0,
            "replay_pmass_gap_share": pm["gap_term"] / total if total > 0 else 0.0,
            "replay_pmass_fast_share": pm["eff_fast"] / total if total > 0 else 0.0,
            "replay_fast_demoted_frac": pm["rows_fast_demoted"] / n_fast if n_fast > 0 else 0.0,
        }

    def _drop_oldest_from_shuffle(self, count: int) -> None:
        drop = int(max(0, count))
        if drop <= 0 or self._shuffle_size_total <= 0:
            return
        drop = min(drop, self._shuffle_size_total)
        remaining = drop
        while remaining > 0 and self._shuffle_buf:
            first_n = self._shuffle_sizes[0]
            if remaining >= first_n:
                self._shuffle_buf.popleft()
                self._shuffle_sizes.popleft()
                self._shuffle_offsets.popleft()
                self._shuffle_size_total -= first_n
                remaining -= first_n
                continue
            self._shuffle_offsets[0] += remaining
            self._shuffle_sizes[0] = first_n - remaining
            self._shuffle_size_total -= remaining
            remaining = 0
        self._shuffle_head += drop
        self._maybe_compact_shuffle_metadata()

    def _maybe_compact_shuffle_metadata(self, *, force: bool = False) -> None:
        if self._shuffle_head <= 0 and not force:
            return
        active_n = int(self._shuffle_size_total)
        if not force:
            threshold = max(4096, self._shuffle_priority_store.shape[0] // 2)
            if self._shuffle_head < threshold:
                return
        if active_n > 0:
            start = int(self._shuffle_head)
            end = start + active_n
            self._shuffle_priority_store = np.array(
                self._shuffle_priority_store[start:end],
                copy=True,
                order="C",
            )
            self._shuffle_wdl_store = np.array(
                self._shuffle_wdl_store[start:end],
                copy=True,
                order="C",
            )
        else:
            self._shuffle_priority_store = np.zeros((0,), dtype=np.float32)
            self._shuffle_wdl_store = np.zeros((0,), dtype=np.int8)
        self._shuffle_head = 0

    def _active_shuffle_priority(self) -> np.ndarray:
        start = int(self._shuffle_head)
        end = start + int(self._shuffle_size_total)
        return self._shuffle_priority_store[start:end]

    def _active_shuffle_wdl(self) -> np.ndarray:
        start = int(self._shuffle_head)
        end = start + int(self._shuffle_size_total)
        return self._shuffle_wdl_store[start:end]

    def _trim_shuffle_buf(self) -> None:
        cap = self._effective_shuffle_cap()
        overflow = self._shuffle_len() - cap
        if overflow > 0:
            self._drop_oldest_from_shuffle(overflow)

    def _append_write_arrays(self, arrs: dict[str, np.ndarray]) -> None:
        n, _, _ = _batch_dims(arrs)
        if n <= 0:
            return
        self._write_buf.append(dict(arrs))
        self._write_buf_sizes.append(n)
        self._write_buf_rows += n

    def _take_write_prefix(self, count: int) -> dict[str, np.ndarray]:
        take = int(max(0, count))
        if take <= 0 or self._write_buf_rows <= 0:
            raise ValueError("write buffer is empty")
        take = min(take, self._write_buf_rows)
        gathered: list[dict[str, np.ndarray]] = []
        remaining = take
        while remaining > 0 and self._write_buf:
            first_n = self._write_buf_sizes[0]
            chunk = self._write_buf[0]
            if remaining >= first_n:
                gathered.append(chunk)
                self._write_buf.pop(0)
                self._write_buf_sizes.pop(0)
                self._write_buf_rows -= first_n
                remaining -= first_n
                continue
            gathered.append(_slice_array_batch(chunk, np.arange(remaining, dtype=np.int64)))
            for name, value in list(chunk.items()):
                arr = np.asarray(value)
                chunk[name] = arr if arr.ndim == 0 else arr[remaining:]
            self._write_buf_sizes[0] = first_n - remaining
            self._write_buf_rows -= remaining
            remaining = 0
        if len(gathered) == 1:
            return gathered[0]
        return _concat_sparse_batches(gathered)

    def _scan_existing_shards(self) -> None:
        """Discover shards already on disk (e.g. after trial restart)."""
        existing = iter_shard_paths(self._shard_dir)
        if not existing:
            return
        sizes: list[int] = []
        total_positions = 0
        for p in existing:
            idx = shard_index(p)
            if idx >= 0:
                self._shard_index = max(self._shard_index, idx + 1)
            n = shard_positions(p)
            sizes.append(n)
            total_positions += n
        with self._prefetch_lock:
            self._shard_paths.extend(existing)
            self._shard_sizes.extend(sizes)
            self._total_positions += total_positions

        self._enforce_window()

        seed_paths = self._snapshot_shards()
        if seed_paths:
            n_seed = min(len(seed_paths), self._refresh_shards * 2)
            for sp in seed_paths[-n_seed:]:
                try:
                    arrs, _ = load_shard_arrays(sp, lazy=False)
                    self._append_shuffle_arrays(arrs)
                except Exception:
                    pass
            self._trim_shuffle_buf()

    def _ensure_prefetch_thread(self) -> None:
        if self._refresh_interval <= 0 or self._refresh_shards <= 0:
            return
        if self._prefetch_thread is not None:
            return
        t = threading.Thread(
            target=self._prefetch_loop,
            args=(self._prefetch_generation, self._prefetch_stop, self._prefetch_request),
            name=f"replay-prefetch-{id(self):x}",
            daemon=True,
        )
        self._prefetch_thread = t
        t.start()

    def _prefetch_loop(
        self,
        generation: int,
        stop_event: threading.Event,
        request_event: threading.Event,
    ) -> None:
  # NOTE: the loop deliberately refills whenever the prefetched slot is
  # empty — including on plain timeout ticks with no request — so a chunk
  # is always warm for the next refresh. The request event only shortens
  # the wakeup latency after a consume; it is not the load trigger. The
  # loop body is sequential (single thread), so loads can never overlap.
        while not stop_event.is_set():
            request_event.wait(timeout=0.1)
            if stop_event.is_set():
                return
            request_event.clear()
            with self._prefetch_lock:
                if generation != self._prefetch_generation:
                    return
                if self._prefetched_refresh is not None:
                    self._prefetch_inflight = False
                    continue
                shard_paths = list(self._shard_paths)
                refresh_shards = int(self._refresh_shards)
            loaded = self._load_refresh_chunks(
                shard_paths=shard_paths,
                refresh_shards=refresh_shards,
                rng=self._prefetch_rng,
            )
            with self._prefetch_lock:
                if generation != self._prefetch_generation:
                    return
                self._prefetch_inflight = False
                if loaded and not stop_event.is_set():
                    self._prefetched_refresh = loaded

    def _schedule_refresh_prefetch(self) -> None:
        self._ensure_prefetch_thread()
        if self._prefetch_thread is None:
            return
        with self._prefetch_lock:
            if self._prefetch_inflight or self._prefetched_refresh is not None:
                return
            if not self._shard_paths or self._refresh_shards <= 0:
                return
            self._prefetch_inflight = True
        self._prefetch_request.set()

    def _take_prefetched_refresh(self) -> list[dict[str, np.ndarray]] | None:
        with self._prefetch_lock:
            loaded = self._prefetched_refresh
            self._prefetched_refresh = None
            return loaded

    def _load_refresh_chunks(
        self,
        *,
        shard_paths: list[Path],
        refresh_shards: int,
        rng: np.random.Generator,
    ) -> list[dict[str, np.ndarray]]:
        if not shard_paths:
            return []

        n_shards = len(shard_paths)
        n_pick = min(int(refresh_shards), n_shards)
        if n_pick <= 0:
            return []

        weights = np.arange(1, n_shards + 1, dtype=np.float64)
        weights /= weights.sum()
        chosen_idxs = rng.choice(n_shards, size=n_pick, replace=False, p=weights)

        loaded: list[dict[str, np.ndarray]] = []
        for idx in np.asarray(chosen_idxs, dtype=np.int64):
            try:
                arrs, _ = load_shard_arrays(shard_paths[int(idx)], lazy=False)
                loaded.append(arrs)
            except Exception:
                pass
        return loaded

    def _apply_refresh_chunks(self, loaded: list[dict[str, np.ndarray]]) -> None:
        loaded_n = sum(int(arrs["x"].shape[0]) for arrs in loaded if "x" in arrs)
        if loaded_n <= 0:
            return

        replace_n = min(self._shuffle_len(), loaded_n)
        if replace_n > 0:
            self._drop_oldest_from_shuffle(replace_n)
        for arrs in loaded:
            self._append_shuffle_arrays(arrs)
        self._trim_shuffle_buf()

    def __len__(self) -> int:
        """Total positions on disk + in write buffer."""
        return self._tracked_shard_positions() + self._write_buf_rows

    def add(self, sample: ReplaySample) -> None:
        self.add_many([sample])

    def add_many(self, samples: list[ReplaySample]) -> None:
        """Add samples: into shuffle buffer immediately, flush to disk when full."""
        if not samples:
            return
        arrs = prune_storage_arrays(samples_to_arrays(samples))

  # Keep newest data available for training immediately in the hot buffer.
        self._append_shuffle_arrays(arrs)
        self._trim_shuffle_buf()

        self._append_write_arrays(arrs)
        while self._write_buf_rows >= self._shard_size:
            self._flush_shard_arrays(self._take_write_prefix(self._shard_size))

        self._enforce_window()

    def add_many_arrays(self, arrs: dict[str, np.ndarray]) -> None:
        """Add array-backed samples without materializing ReplaySample objects."""
  # Pad before the write buffer too — a stale narrower chunk landing in
  # the same write accumulator as current-width chunks would otherwise
  # fail at shard-flush concatenation.
        sparse = prune_storage_arrays(self._pad_x_planes(arrs))
        n = int(sparse["x"].shape[0])
        if n <= 0:
            return

        self._append_shuffle_arrays(sparse)
        self._trim_shuffle_buf()

        self._append_write_arrays(sparse)
        while self._write_buf_rows >= self._shard_size:
            self._flush_shard_arrays(self._take_write_prefix(self._shard_size))

        self._enforce_window()

    def flush(self) -> None:
        """Force-write any remaining samples in write buffer to disk."""
        if self._write_buf_rows > 0:
            self._flush_shard_arrays(self._take_write_prefix(self._write_buf_rows))
            self._enforce_window()

    def enforce_window(self) -> None:
        """Apply the current capacity limit immediately."""
        self._enforce_window()

    def _flush_shard_arrays(self, arrs: dict[str, np.ndarray]) -> None:
        """Write a shard to disk."""
        path = local_shard_path(self._shard_dir, self._shard_index)
        saved_path = save_local_shard_arrays(path, arrs=arrs)
        n = int(arrs["x"].shape[0])
        self._append_shard_record(saved_path, n)
        self._shard_index += 1

    def _enforce_window(self) -> None:
        """Delete oldest shards when total exceeds capacity."""
        current_total = self._tracked_shard_positions()
        if current_total > self.capacity and not self._snapshot_shards():
            print(
                f"[disk_buf] BUG: total_pos={current_total} > cap={self.capacity} "
                f"but no tracked shards to delete!"
            )
        deleted = 0
        while current_total > self.capacity:
            popped = self._pop_oldest_shard_record()
            if popped is None:
                break
            oldest, n = popped
            current_total -= n
            deleted += 1
            try:
                delete_shard_path(oldest)
            except Exception as e:
                print(f"[disk_buf] WARNING: failed to delete {oldest}: {e}")
        if deleted:
            print(
                f"[disk_buf] enforce_window: deleted {deleted} shards, "
                f"total_pos={self._tracked_shard_positions()}, cap={self.capacity}, "
                f"tracked={len(self._snapshot_shards())}"
            )

    def _sample_indices(self, pool: np.ndarray, k: int) -> np.ndarray:
        if k <= 0 or pool.size == 0:
            return np.zeros((0,), dtype=np.int64)
        pool = np.asarray(pool, dtype=np.int64)
        k_uni = round(k * (1.0 - self.surprise_mix))
        k_pri = k - k_uni
        picks: list[np.ndarray] = []
        if k_uni > 0:
            chosen = self.rng.choice(pool.shape[0], size=k_uni, replace=True)
            picks.append(pool[np.asarray(chosen, dtype=np.int64)])
        if k_pri > 0:
            pri = self._active_shuffle_priority()[pool].astype(np.float64, copy=True)
            np.nan_to_num(pri, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.maximum(pri, 0.0, out=pri)
            ps = float(pri.sum())
            if ps <= 0.0 or not np.isfinite(ps):
                chosen = self.rng.choice(pool.shape[0], size=k_pri, replace=True)
                picks.append(pool[np.asarray(chosen, dtype=np.int64)])
            else:
                p = pri / ps
                chosen = self.rng.choice(pool.shape[0], size=k_pri, replace=True, p=p)
                picks.append(pool[np.asarray(chosen, dtype=np.int64)])
        if not picks:
            return np.zeros((0,), dtype=np.int64)
        if len(picks) == 1:
            return picks[0]
        return np.concatenate(picks, axis=0)

    def _sample_all_indices(self, batch_size: int) -> np.ndarray:
        n = self._shuffle_len()
        bs = int(batch_size)
        if bs <= 0 or n <= 0:
            return np.zeros((0,), dtype=np.int64)

        k_uni = round(bs * (1.0 - self.surprise_mix))
        k_pri = bs - k_uni
        picks: list[np.ndarray] = []
        if k_uni > 0:
            picks.append(np.asarray(self.rng.integers(0, n, size=k_uni), dtype=np.int64))
        if k_pri > 0:
            pri = self._active_shuffle_priority().astype(np.float64, copy=True)
            np.nan_to_num(pri, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.maximum(pri, 0.0, out=pri)
            ps = float(pri.sum())
            if ps <= 0.0 or not np.isfinite(ps):
                picks.append(np.asarray(self.rng.integers(0, n, size=k_pri), dtype=np.int64))
            else:
                picks.append(
                    np.asarray(self.rng.choice(n, size=k_pri, replace=True, p=(pri / ps)), dtype=np.int64)
                )
        if not picks:
            return np.zeros((0,), dtype=np.int64)
        if len(picks) == 1:
            return picks[0]
        return np.concatenate(picks, axis=0)

    def _gather_rows(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        if not self._shuffle_buf:
            raise ValueError("DiskReplayBuffer shuffle buffer is empty")
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1:
            idx = idx.reshape(-1)
        x_planes = int(next(iter(self._shuffle_buf))["x"].shape[1])
        policy_size = int(np.asarray(next(iter(self._shuffle_buf)).get("_policy_size", POLICY_SIZE)).item())
        if idx.size == 0:
            return {
                name: zeros_for_storage_field(
                    name, n=0, policy_size=policy_size, x_planes=x_planes,
                )
                for name in ("x", "policy_target", "wdl_target", "priority", "has_policy")
            }

        if np.any(idx < 0) or np.any(idx >= self._shuffle_len()):
            raise IndexError("sample index out of shuffle-buffer range")

  # Densify each chunk's selected rows, then merge into output.
  # This avoids shape mismatches when chunks have different sparse K values.
        selected_sparse: list[tuple[dict[str, np.ndarray], np.ndarray]] = []
        selected_policy_sizes: set[int] = set()
        selected_x_planes: set[int] = set()
        all_keys: set[str] = set()
        start = 0
        for chunk, chunk_n, chunk_off in zip(self._shuffle_buf, self._shuffle_sizes, self._shuffle_offsets, strict=True):
            end = start + chunk_n
            mask = (idx >= start) & (idx < end)
            if np.any(mask):
                local = idx[mask] - start + chunk_off
                rows = {k: (v if np.asarray(v).ndim == 0 else v[local]) for k, v in chunk.items()}
                _, row_policy_size, row_x_planes = _batch_dims(rows)
                selected_policy_sizes.add(row_policy_size)
                selected_x_planes.add(row_x_planes)
                selected_sparse.append((rows, mask))
            start = end
        if len(selected_policy_sizes) > 1:
            raise ValueError(f"mixed replay policy sizes in shuffle buffer: {sorted(selected_policy_sizes)}")
        if len(selected_x_planes) > 1:
            raise ValueError(f"mixed replay input plane counts in shuffle buffer: {sorted(selected_x_planes)}")
        policy_size = next(iter(selected_policy_sizes))

        selected: list[tuple[dict[str, np.ndarray], np.ndarray]] = []
        for rows, mask in selected_sparse:
            dense_rows = densify_chunk(rows, policy_size=policy_size)
            selected.append((dense_rows, mask))
            all_keys.update(dense_rows.keys())

  # Build prototype from ALL selected chunks so optional fields present
  # in any chunk are allocated (not just those in the first chunk).
        proto: dict[str, np.ndarray] = {}
        for dense_rows, _ in selected:
            for k, v in dense_rows.items():
                if k not in proto:
                    proto[k] = v
        n_out = int(idx.shape[0])
        out: dict[str, np.ndarray] = {}
        for k in sorted(all_keys):
            if k in proto:
                out[k] = np.zeros((n_out, *proto[k].shape[1:]), dtype=proto[k].dtype)
        for dense_rows, mask in selected:
            for name, value in dense_rows.items():
                if name in out:
                    out[name][mask] = value
        return out

    def sample_batch_arrays(self, batch_size: int, *, wdl_balance: bool = True) -> dict[str, np.ndarray]:
        """Sample a batch as compact arrays without materializing ReplaySample objects."""
        self._trim_shuffle_buf()
        n = self._shuffle_len()
        if n == 0:
            raise ValueError("DiskReplayBuffer shuffle buffer is empty")

        self._sample_count += 1
        if self._refresh_interval > 0 and (self._sample_count % self._refresh_interval == 0) and self._snapshot_shards():
            loaded = self._take_prefetched_refresh()
            if loaded:
                self._apply_refresh_chunks(loaded)
            else:
                self._refresh_shuffle_buf()
            self._schedule_refresh_prefetch()
            n = self._shuffle_len()

        bs = int(batch_size)
        if not wdl_balance:
            return self._sample_raw_arrays(bs)

        active_wdl = self._active_shuffle_wdl()
        win_idx = np.flatnonzero(active_wdl == 0)
        draw_idx = np.flatnonzero(active_wdl == 1)
        loss_idx = np.flatnonzero(active_wdl == 2)

        if win_idx.size == 0 or loss_idx.size == 0:
            return self._sample_raw_arrays(bs)

        p_draw = float(draw_idx.size) / float(max(1, n))
        n_draw = round(bs * p_draw)
        n_draw_cap = int(np.floor(self.draw_cap_frac * bs))
        n_draw = min(n_draw, n_draw_cap)
        n_draw = max(0, min(bs, n_draw))

        bs_decisive = bs - n_draw
        n_win = 0
        n_loss = 0
        if bs_decisive > 0:
            p_win = float(win_idx.size) / float(win_idx.size + loss_idx.size)
            n_win = round(bs_decisive * p_win)
            n_win = max(0, min(bs_decisive, n_win))
            n_loss = bs_decisive - n_win

            r = self.wl_max_ratio
            if n_win > int(np.floor(r * n_loss)):
                n_loss = int(np.ceil(bs_decisive / (1.0 + r)))
                n_win = bs_decisive - n_loss
            elif n_loss > int(np.floor(r * n_win)):
                n_win = int(np.ceil(bs_decisive / (1.0 + r)))
                n_loss = bs_decisive - n_win

        picks = [
            self._sample_indices(draw_idx, n_draw),
            self._sample_indices(win_idx, n_win),
            self._sample_indices(loss_idx, n_loss),
        ]
        chosen = np.concatenate([p for p in picks if p.size > 0], axis=0) if any(p.size > 0 for p in picks) else np.zeros((0,), dtype=np.int64)

        if chosen.shape[0] < bs:
            extra = self._sample_raw_indices(bs - chosen.shape[0])
            chosen = np.concatenate([chosen, extra], axis=0) if chosen.size > 0 else extra
        elif chosen.shape[0] > bs:
            chosen = chosen[:bs]

        self.rng.shuffle(chosen)
        return self._gather_rows(chosen)

    def sample_batch(self, batch_size: int, *, wdl_balance: bool = True) -> list[ReplaySample]:
        """Compatibility path returning ReplaySample objects."""
        return arrays_to_samples(self.sample_batch_arrays(batch_size, wdl_balance=wdl_balance))

    def _sample_raw_indices(self, batch_size: int) -> np.ndarray:
        return self._sample_all_indices(batch_size)

    def _sample_raw_arrays(self, batch_size: int) -> dict[str, np.ndarray]:
        return self._gather_rows(self._sample_raw_indices(batch_size))

    def _refresh_shuffle_buf(self) -> None:
        """Replace oldest shuffle samples with rows from randomly chosen disk shards."""
        loaded = self._load_refresh_chunks(
            shard_paths=self._snapshot_shards(),
            refresh_shards=self._refresh_shards,
            rng=self.rng,
        )
        self._apply_refresh_chunks(loaded)

    def clear(self) -> None:
        """Remove all data (disk and memory)."""
        self.close()
        self._shuffle_buf = deque()
        self._shuffle_sizes = deque()
        self._shuffle_offsets = deque()
        self._shuffle_size_total = 0
        self._shuffle_priority_store = np.zeros((0,), dtype=np.float32)
        self._shuffle_wdl_store = np.zeros((0,), dtype=np.int8)
        self._shuffle_head = 0
        self._write_buf = []
        self._write_buf_sizes = []
        self._write_buf_rows = 0
        for p in self._clear_shard_records():
            with contextlib.suppress(Exception):
                delete_shard_path(p)

    def close(self) -> None:
        t = self._prefetch_thread
        stop_event = self._prefetch_stop
        request_event = self._prefetch_request
        with self._prefetch_lock:
            self._prefetch_generation += 1
            self._prefetch_inflight = False
            self._prefetched_refresh = None
        stop_event.set()
        request_event.set()
        if t is not None and t.is_alive():
            t.join(timeout=1.0)
        self._prefetch_thread = None
        self._prefetch_stop = threading.Event()
        self._prefetch_request = threading.Event()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()
