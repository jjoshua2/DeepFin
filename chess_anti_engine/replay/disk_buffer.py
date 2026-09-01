"""Disk-backed replay buffer with array-backed hot shuffle storage.

Stores training data as NPZ shards on disk and keeps a small hot subset in
memory for efficient sampling. The hot path stays columnar/array-backed so the
trainer can batch directly from NumPy arrays instead of inflating tens of
thousands of ``ReplaySample`` Python objects.
"""
from __future__ import annotations

import fcntl
import logging
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from chess_anti_engine.encoding import version_for_input_planes
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS

from .buffer import ReplaySample
from .shard import (
    HISTORY_REP_FIX_ARRAY_KEY,
    INPUT_HISTORY_ENCODING_ARRAY_KEY,
    NONZERO_DEFAULT_STORAGE_FIELDS,
    POLICY_ENCODING_ARRAY_KEY,
    _REQUIRED_STORAGE_FIELDS,
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
    validate_arrays,
    zeros_for_storage_field,
)
from .threat_upgrade import (
    V1_INPUT_PLANES,
    upgrade_arrays_to_planes,
)
import contextlib

log = logging.getLogger(__name__)

# Consecutive fully-empty shuffle refreshes before we say so out loud. The
# prefetch thread retries roughly every 0.1s, so this is a few seconds of
# genuinely stalled refresh -- long enough to ride out a burst of window
# trimming, short enough to precede any training damage.
_REFRESH_EMPTY_STREAK_ALARM = 50
# Throttle for the tracked-shard load-failure line. Generous because the
# failure it reports is per-shard and can arrive in bursts.
_SHARD_LOAD_WARN_INTERVAL_S = 60.0

# Threads used to decode ONE shuffle refresh's shards. Deliberately a module
# constant rather than a constructor kwarg: every `DiskReplayBuffer` ctor kwarg
# has to be classified by `assert_buffer_kwargs_are_classified`, and this is a
# decode-scheduling detail that changes no sampled row, so it has no business
# in the lc0-control replay pin.
#
# 4 is measured, not guessed. On the production path -- the refresh runs on the
# trainer's prefetch WORKER thread -- decoding 3 shards with the validation memo
# already in place: serial 5.409s, 2 workers 4.486s (1.21x), 4 workers 2.315s
# (2.34x).
#
# ⚑ WHY THERE IS ANYTHING TO WIN AT ALL, since blosc already decompresses with 8
# internal threads: numcodecs turns that OFF on non-main threads
# (`_get_use_threads()` is True on the main thread, False on a worker -- measured).
# The refresh has ALWAYS run on a worker thread, so its decode was already
# single-threaded; this pool is what puts the parallelism back. The same fact is
# why the outer pool is worth nothing when the caller IS the main thread (there,
# serial gets blosc's 8 threads for free and a 4-way pool measured SLOWER), and
# why raising this number is not free on a box that is also running selfplay.
_REFRESH_LOAD_WORKERS = 4

# How often the shard-validation memo reports its hit rate. Throttled because
# production runs `refresh_interval` as low as 1.
_VALIDATION_MEMO_LOG_INTERVAL_S = 300.0

# Advisory lock naming the one process allowed to write shards into a window
# dir. Two writers is not hypothetical: on 2026-07-11 two of them interleaved
# index streams into one dir for 87 minutes and left 20.5k byte-identical
# duplicate rows (1.37% of the window), which then rode into the salvage pool
# `swap_512x16_20260711` and is still in the live window 15 days later. Nothing
# noticed at the time; audit G7 found it. The lock is a DETECTOR, not a guard --
# it logs and continues, because refusing to write would turn a data-quality
# defect into a production outage.
_WRITER_LOCK_NAME = ".writer.lock"

_ARRAY_FIELD_ORDER = _SHARD_FIELDS
_SCALAR_METADATA_FIELDS = (
    INPUT_HISTORY_ENCODING_ARRAY_KEY,
    POLICY_ENCODING_ARRAY_KEY,
    # Same encoding name can carry different repetition planes; refuse to
    # concatenate legacy and rep-fix chunks into one training window.
    HISTORY_REP_FIX_ARRAY_KEY,
)

# The shard-recency draw exponent that reproduces the pre-2026-08-02 code
# exactly: weight ∝ rank, oldest rank 1. Named so the one value that must not
# drift is written once and referenced everywhere that claims to preserve it.
DEFAULT_SHARD_RECENCY_EXPONENT = 1.0
_DRAW_DECILES = 10

# Smallest positive NORMAL float64 (2.225e-308). Post-max-subtraction weights
# below this are raised to it before normalization, so every shard keeps a
# strictly positive draw probability. See shard_draw_weights.
_SHARD_DRAW_WEIGHT_FLOOR = float(np.finfo(np.float64).tiny)

# Shard count used ONLY to derive MAX_SHARD_RECENCY_EXPONENT below. Absurdly
# generous on purpose: the live window tracks ~834 shards at shard_size 2000
# against a 1.5M-row cap, so 1e6 shards would be 2e9 rows -- ~1300x a window
# this project can hold.
_BOUND_REFERENCE_N_SHARDS = 1_000_000

# Upper bound on the recency exponent, DERIVED rather than picked: the
# log-space weights span `alpha * ln(n)` nats, so nothing underflows while
# `alpha <= ln(1/tiny) / ln(n)` = 708.396 / ln(n). At the reference window
# above that is 708.396 / 13.8155 = 51.28, rounded down to 50.
#
# ⚑ The bound is NOT what stops the underflow failures -- the floor is, and it
# holds for every finite alpha. The bound exists because past the underflow
# point the knob stops MEANING anything: every larger alpha produces the same
# "newest shard, probability 1 - 1e-305" draw, so the yaml value would read
# back correctly and select a regime rather than a value. That is this repo's
# signature defect, so it is refused loudly instead. Even at the bound the draw
# is already far past any proposed experiment -- alpha = 50 puts the mean drawn
# shard at the 98.1st recency percentile, while arm G is about alpha in [0, 1].
MAX_SHARD_RECENCY_EXPONENT = 50.0


def validate_shard_recency_exponent(recency_exponent: float) -> float:
    """Return the exponent as a float, or raise naming the offending value.

    Separate from the weight computation so the CONSTRUCTOR can reject a bad
    yaml value at startup. A bad exponent otherwise first bites inside the
    PREFETCH THREAD, on the first refresh, as a probability vector
    ``rng.choice`` refuses — where the exception is swallowed and the only
    symptom is a shuffle pool that quietly stops being refreshed. That is the
    failure this function exists to move to startup, so the bounds it enforces
    have to be the ones that actually reach it.

    Three rejections, each a decision rather than an oversight:

    * **non-finite** — a NaN or inf weight vector.
    * **negative** — refused DELIBERATELY. An oldest-heavy draw (alpha < 0
      inverts the ranking and trains preferentially on the stalest end of the
      window) is not a hypothesis anyone has proposed; the open question, rig
      arm G, is whether to FLATTEN recency, i.e. alpha in [0, 1]. The
      log-space form below is correct for negative alpha, so enabling it is a
      one-line relaxation here — behind a ledger entry, not a yaml edit.
    * **above MAX_SHARD_RECENCY_EXPONENT** — see that constant. Not a numeric
      safety bound (the floor is), a meaningfulness bound.
    """
    alpha = float(recency_exponent)
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > MAX_SHARD_RECENCY_EXPONENT:
        raise ValueError(
            f"shard recency exponent must be finite and within "
            f"[0.0, {MAX_SHARD_RECENCY_EXPONENT}], got {recency_exponent!r}; "
            "1.0 = the production linear-recency draw, 0.0 = uniform over the "
            "window. Negative (oldest-heavy) and above-bound (saturated onto "
            "the newest shard, where every larger value draws identically) are "
            "both refused on purpose -- see validate_shard_recency_exponent"
        )
    return alpha


def _shard_draw_log_weights(n_shards: int, alpha: float) -> np.ndarray:
    """Unnormalized LOG draw weights, oldest first, shifted so the max is 0.

    ``alpha * log(rank)`` rather than ``rank ** alpha``: the log form's dynamic
    range is linear in alpha instead of exponential in it, so the intermediate
    cannot overflow to ``inf`` for any bounded alpha. Subtracting the max makes
    the largest weight ``exp(0) == 1.0`` EXACTLY, which is what lets the caller
    reason about the floor and the sum in absolute terms rather than relative
    ones: every weight is in ``(0, 1]``, so the normalizing sum is always in
    ``[1, n]`` and can neither vanish nor overflow.

    ``log(rank)`` is exactly 0.0 at rank 1 and finite and increasing after, so
    this is well defined for negative alpha too (the max then sits at rank 1);
    only ``validate_shard_recency_exponent`` restricts the sign.
    """
    ranks = np.arange(1, int(n_shards) + 1, dtype=np.float64)
    if ranks.size == 0:
        return ranks
    log_weights = alpha * np.log(ranks)
    return log_weights - log_weights.max()


def shard_draw_weights(n_shards: int, recency_exponent: float) -> np.ndarray:
    """Normalized shard-draw probabilities, OLDEST first: ``w_i ∝ rank_i**α``.

    ``rank`` is ``1..n_shards`` oldest→newest, so α = 1.0 is the linear recency
    weighting the refresh draw has always used (at production scale, 834
    tracked shards, the newest shard is drawn 834× as often as the oldest and
    the mean sampled shard sits at the 2/3 recency percentile). α = 0.0 is
    uniform over the window; α > 1 sharpens onto the newest shards.

    ⚑ α = 1.0 takes its own branch and is BYTE-IDENTICAL to the code this
    replaced, which is the whole point of the default. It is not a
    micro-optimisation and must not be folded into the general expression: it
    is what lets the knob ship into a live run whose sampling distribution is
    an open experimental question (rig arm G) without perturbing it. Folding it
    away would make the default depend on whether ``np.power(x, 1.0)`` returns
    ``x`` bit-for-bit, which is a libm/numpy-version property, not a promise.
    ``tests/test_replay_shard_recency_exponent.py`` pins the equality against
    an inline copy of the original two lines.

    The general branch computes in LOG space (see
    :func:`_shard_draw_log_weights`) and then FLOORS at the smallest positive
    normal float64 before normalizing. That closes two DISTINCT failures of the
    ``(rank / n) ** alpha`` form this branch used first, measured rather than
    reasoned about (``tests/test_replay_shard_recency_exponent.py``
    reconstructs both):

    * the OLDEST weight reaches exactly 0.0 at alpha ≈ 110.5 for the live
      window of 834 shards. No exception — the draw keeps working and that
      shard is simply PERMANENTLY INELIGIBLE, a shard sitting in the window
      that can never be sampled. This is the silent half and the worse one.
    * ``count_nonzero(p) < refresh_shards``, which is what actually makes
      ``rng.choice(..., replace=False)`` raise ``ValueError: Fewer non-zero
      entries in p than size``, needs alpha ≈ 154,987 at n = 834. On the
      production path that raise happens inside the PREFETCH THREAD, where
      nothing reads it and the only symptom is a shuffle pool that silently
      stops refreshing.

    Overflow is not among them: ``(rank / n) ** alpha`` is bounded by 1 for
    every alpha ≥ 0, and the log form is bounded by 1 by construction. Only a
    raw ``rank ** alpha`` could overflow, which this code has never used.

    The floor makes both unreachable for any finite alpha: the post-shift
    maximum is exactly 1.0, so the sum is in ``[1, n]`` and every floored entry
    lands at ``>= tiny / n``, which stays representable (and strictly positive)
    for any n below ~4.5e15. ``MAX_SHARD_RECENCY_EXPONENT`` then keeps
    production well clear of the floor entirely; the floor is the belt that
    survives anyone relaxing that bound.

    The floored mass is negligible by construction, not by hope: at most
    ``n * tiny`` ≈ 2e-305 of the total for any plausible n, i.e. ~300 orders of
    magnitude below the least significant digit of the printed decile table.
    """
    alpha = validate_shard_recency_exponent(recency_exponent)
    if alpha == DEFAULT_SHARD_RECENCY_EXPONENT:
  # ⚑ UNTOUCHED, and deliberately not routed through the log-space form: this
  # is the byte-identity the default rests on. See the docstring above.
        weights = np.arange(1, int(n_shards) + 1, dtype=np.float64)
        weights /= weights.sum()
        return weights

    weights = np.exp(_shard_draw_log_weights(int(n_shards), alpha))
    if weights.size == 0:
        return weights
    np.maximum(weights, _SHARD_DRAW_WEIGHT_FLOOR, out=weights)
    weights /= weights.sum()
    return weights


def shard_draw_floored_count(n_shards: int, recency_exponent: float) -> int:
    """How many shards :func:`shard_draw_weights` had to raise off the floor.

    The one number that says whether the floor engaged, for the construction
    line. Computed from the same expression the weights use — ``raw < floor``
    on the pre-floor vector — rather than re-deriving the threshold in log
    space, so the count cannot disagree with the flooring by a boundary ulp.

    Always 0 on the default path, which never floors: its weights are the
    integers 1..n.
    """
    alpha = validate_shard_recency_exponent(recency_exponent)
    if alpha == DEFAULT_SHARD_RECENCY_EXPONENT:
        return 0
    raw = np.exp(_shard_draw_log_weights(int(n_shards), alpha))
    return int(np.count_nonzero(raw < _SHARD_DRAW_WEIGHT_FLOOR))


def shard_draw_decile_mass_realized(weights: np.ndarray) -> np.ndarray:
    """Sum a REALIZED draw-weight vector into 10 equal-rank bins, oldest first.

    The counterpart to :func:`shard_draw_decile_mass`, which is the continuum
    limit. This one is what the sampler will actually do at the window size the
    buffer currently holds: it carries the discreteness (at n = 3 each shard is
    a third of the window and no continuum decile describes it) and any
    flooring, because it is a sum over the very vector handed to
    ``rng.choice``.
    """
    n_shards = int(weights.shape[0])
    if n_shards == 0:
        return np.zeros((_DRAW_DECILES,), dtype=np.float64)
    bins = (np.arange(n_shards) * _DRAW_DECILES) // n_shards
    return np.bincount(bins, weights=weights, minlength=_DRAW_DECILES).astype(np.float64)


def shard_draw_decile_mass(recency_exponent: float) -> np.ndarray:
    """Analytic draw mass per window decile, OLDEST decile first.

    The continuum limit of :func:`shard_draw_weights`: with density ∝ f**α over
    the window fraction f (0 = oldest, 1 = newest), the mass below f is
    ``f**(α+1)``, so decile j spans ``(j/10)**(α+1) - ((j-1)/10)**(α+1)``. For
    the production α = 1.0 the cumulative mass of the NEWEST fraction f is
    ``1 - (1-f)**2 = 2f - f²`` — the closed form the realized-draw test checks
    the sampler against.

    Analytic rather than measured because this is printed at construction, when
    a fresh trial has no shards yet and a resumed one has a window whose size
    changes every iteration; the shape of the draw does not depend on either,
    and a decile table that silently reported "no shards" would be exactly the
    unreadable observability this line exists to avoid.
    """
    alpha = validate_shard_recency_exponent(recency_exponent)
    edges = np.arange(0, _DRAW_DECILES + 1, dtype=np.float64) / float(_DRAW_DECILES)
    return np.diff(edges ** (alpha + 1.0))


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
  # (There used to be a second branch here retaining a uniformly-absent VALUE
  # array when its `has_*` flag had already been written to `out`. It could
  # never fire: `_ARRAY_FIELD_ORDER` is `_SHARD_FIELDS`, which emits `spec.arr`
  # BEFORE `spec.flag` for every pair, so a value name is always decided while
  # its own flag is still unwritten. It read as a safety net holding nothing;
  # the ordering it depended on is now pinned by
  # tests/test_replay_field_defaults.py.)
        if any(name in chunk for chunk in chunks) or name in _REQUIRED_STORAGE_FIELDS:
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
    """Disk-backed replay buffer with array-backed hot shuffle storage.

    **Constructing this against a live shard directory DELETES SHARDS.**
    ``__init__`` scans the directory and immediately calls ``_enforce_window``,
    so a tool that opens the live window "just to read it" with a capacity that
    differs from the live one evicts the excess off disk before the constructor
    even returns (audit hazard G12). Pass ``read_only=True`` for any probe,
    audit or offline read: it refuses every mutating call and turns window
    enforcement into a no-op, making the constructor safe to point at
    production data. Reading zarr groups directly, without this class, remains
    the lightest option.

    ``read_only`` is a REQUIRED keyword with no default, deliberately. A
    default of False documents the hazard instead of closing it: the next probe
    someone writes gets the shard-deleting behaviour by forgetting a kwarg,
    which is exactly the shape G12 records. With no default, every construction
    site has to state its intent and a new one that does not is a type error
    before it is a deleted production window. The cost is a kwarg on the few
    writers, which is the cheaper side of that trade.
    """

    def __init__(
        self,
        capacity: int,
        *,
        shard_dir: Path,
        rng: np.random.Generator,
        read_only: bool,
        shuffle_cap: int = 20_000,
        shard_size: int = 1000,
        refresh_interval: int = 5,
        refresh_shards: int = 3,
        shard_recency_exponent: float = DEFAULT_SHARD_RECENCY_EXPONENT,
        draw_cap_frac: float = 0.90,
        wl_max_ratio: float = 1.5,
        input_planes: int | None = None,
        upgrade_v1_planes: bool = False,
        sf_gap_priority_weight: float = 0.0,
        sf_gap_priority_signed: bool = False,
        fast_low_surprise_priority: float = 1.0,
        diff_focus_pol_scale: float = 0.0,
        diff_focus_q_weight: float = 0.0,
        deterministic_refresh: bool = False,
    ):
        self.capacity = int(capacity)
        self.rng = rng
  # Set BEFORE the resume scan at the end of __init__: that scan enforces the
  # window, i.e. deletes shards, which is exactly what a read-only opener must
  # not do (audit G12).
        self._read_only = bool(read_only)
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
        if not self._read_only:
  # A read-only open must leave the filesystem exactly as it found it, and a
  # mistyped --replay-dir should read as "empty window", not quietly appear
  # to exist.
            self._shard_dir.mkdir(parents=True, exist_ok=True)
  # Taken lazily on the first shard write, not here: a read-only or
  # short-lived buffer should not claim the dir, and `clear()` routes
  # through `close()` and then keeps writing.
        self._writer_lock_fd: int | None = None
        self._writer_lock_reported = False

        self._shuffle_cap = int(shuffle_cap)
        self._shard_size = int(shard_size)
        self._refresh_interval = int(refresh_interval)
        self._refresh_shards = int(refresh_shards)
  # Validated HERE, not on the first refresh: see validate_shard_recency_exponent.
  # It has to be BEFORE the resume scan at the end of __init__, because that
  # scan calls _enforce_window() -- a bad yaml exponent must raise before it can
  # delete shards, not after. The matching REPORT runs after the scan instead,
  # where there is a real shard count to describe.
        self._shard_recency_exponent = validate_shard_recency_exponent(
            shard_recency_exponent,
        )
        self._prefetch_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
  # OFF in production; ON makes the draw sequence a pure function of the seed.
  #
  # The shuffle-pool refresh has TWO implementations and picks between them by
  # WHO WON A RACE: `sample_batch_arrays` takes the background thread's chunk
  # if one is ready and otherwise loads synchronously. They do not consume the
  # same rng -- the prefetch path draws shards from `_prefetch_rng` and leaves
  # `self.rng` untouched, the synchronous path draws from `self.rng` and
  # ADVANCES it. So one lost race changes both which shards enter the pool and
  # the state of the generator that picks every subsequent row, permanently.
  # Whether the thread wins depends on machine load, which is why the same
  # offline probe returned two different numbers on the same seed (measured
  # 2026-07-31: 0.0056 nats excess / 0.0075 agreement between the two states).
  #
  # Setting this suppresses the thread entirely: `_schedule_refresh_prefetch`
  # then finds no thread and returns, `_take_prefetched_refresh` is always
  # None, and every refresh goes down the synchronous path. Both paths draw the
  # same recency-weighted `refresh_shards` without replacement, so this changes
  # WHICH stream the draw comes from, never the distribution it comes from.
  #
  # Not the production default: production trades the refresh's disk read off
  # the sampling thread for throughput, and its own reproducibility argument is
  # a checkpoint, not a seed. Offline rulers have the opposite need, so they
  # opt in.
        self._deterministic_refresh = bool(deterministic_refresh)
        self._prefetch_lock = threading.Lock()
        self._prefetch_request = threading.Event()
        self._prefetch_stop = threading.Event()
        self._prefetch_inflight = False
        self._prefetched_refresh: list[dict[str, np.ndarray]] | None = None
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_generation = 0

  # Shard-load accounting for the shuffle refresh. Split into two counters
  # because the two causes need opposite responses: a shard the window
  # trimmer deleted out from under us is EXPECTED (see
  # _note_shard_load_failure), while any other failure is a real problem
  # worth a line. Without the split there is only "except Exception: pass",
  # and the number of shards actually loaded is never compared against the
  # number requested.
        self._refresh_vanished_total = 0
        self._refresh_failed_total = 0
        self._refresh_empty_streak = 0
        self._shard_load_warned_at = 0.0

  # ⚑ VALIDATION MEMO. `validate_arrays` is a pure function of the arrays it is
  # handed, and the shuffle refresh re-reads the SAME unchanged local shards for
  # the whole run -- at refresh_interval=5 / refresh_shards=3 over a 9.7k-step
  # run each of ~650 shards is decoded and re-validated ~19 times. Measured on
  # the production path (the refresh runs on the trainer's prefetch WORKER
  # thread, warm cache, 3 shards): 7.467s with validation, 1.022s without --
  # 7.31x. It is that lopsided off the main thread because numcodecs turns
  # blosc's internal threading OFF on non-main threads, so the decode half is
  # already single-threaded while `validate_arrays` is uncompensated numpy.
  #
  # Keyed on the shard's CONTENT FOOTPRINT, not its directory mtime: a `.zarr`
  # shard is a DIRECTORY of ~291 chunk files, and rewriting a chunk in place
  # leaves the directory's own mtime untouched -- so a path+dir-mtime key would
  # be a gate that cannot fire. `_shard_validation_fingerprint` walks the tree
  # instead (file count, total size, newest mtime_ns), measured at 3.4 ms
  # against a 878 ms load: 0.39%.
  #
  # ⚑ WHAT THIS GIVES UP, STATED: a shard mutated in place with its size AND
  # every file's mtime preserved is validated once and trusted thereafter. That
  # is bit-rot or a deliberate forgery, not the trim/rewrite races this buffer
  # actually sees, and both of those move the fingerprint. The memo is per
  # INSTANCE and never persisted, so a fresh buffer re-validates from scratch.
        self._validated_shards: dict[str, tuple[int, int, int]] = {}
        self._validation_memo_lock = threading.Lock()
        self._shard_validations_run = 0
        self._shard_validations_skipped = 0
  # ⚑ SEEDED WITH now(), NOT 0.0. With 0.0 the first line lands on the FIRST
  # refresh -- where a hit is arithmetically impossible, because nothing has
  # been read twice yet -- so it always reported "memo hit rate 0.0%" and a
  # reader would conclude the memo was inert. Measured: a 100-batch run that
  # finished at a 38% hit rate printed 0.0%. Starting the clock here means
  # every line that IS printed describes a warmed-up buffer.
        self._validation_memo_logged_at = time.time()

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
  # After the scan, so a resume describes the draw over its ACTUAL window
  # rather than the continuum limit. On a resume this lands just below the
  # `[disk_buf] shuffle seed` line the scan emits, and the two are meant to be
  # read together -- see _report_shard_recency_draw on why their two recency
  # numbers are not the same quantity.
        self._report_shard_recency_draw()
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
            "kl_raw": 0.0, "kl_rows": 0.0, "qd_raw": 0.0, "qd_rows": 0.0,
        }

    def _accumulate_priority_mass(self, arrs: dict[str, np.ndarray], pri: np.ndarray) -> None:
        """Telemetry: decompose appended sampling-priority mass by source term.

        Uses the mirrored diff-focus weights to reconstruct the kl/qd term
        mass baked into stored priorities at selfplay time, plus the gap
        boost and fast-row demotion applied here at append time. Popped once
        per iteration by the tune trainable (pop_priority_mass_stats).

        ⚑ THE ``*_share`` OUTPUTS ARE ONLY VALID WHILE THE WEIGHTS ARE CONSTANT.
        Sampling uses the STORED ``priority`` column, which the worker computed
        with the ``diff_focus_pol_scale``/``q_weight`` that were live when the
        ply was played. This decomposition multiplies the stored ``kl``/
        ``q_delta`` columns by TODAY's weights. Recalibrate ``pol_scale`` and
        every row already in the window is decomposed with a factor its own
        priority never saw, so ``replay_pmass_kl_share`` misreports the actual
        sampling until the ~1.5M-position window turns over (~133 iterations).
        The ``*_raw_mean`` outputs below carry no weight at all -- they are the
        stored quantities themselves -- so they stay comparable across a
        recalibration and across a search change. Guard on those.
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
            kl_pos = np.maximum(kl[has], 0.0)
            pm["kl_term"] += float(kl_pos.sum()) * self.diff_focus_pol_scale
            # Unweighted, so a pol_scale recalibration cannot move it: this is
            # the search-vs-prior divergence itself. Its own row count, because
            # the kl and q_delta presence masks are independent.
            pm["kl_raw"] += float(kl_pos.sum())
            pm["kl_rows"] += float(has.sum())
        if self.diff_focus_q_weight > 0 and "priority_q_delta" in arrs and "has_priority_q_delta" in arrs:
            has = np.asarray(arrs["has_priority_q_delta"], dtype=bool) & hp
            qd = np.nan_to_num(np.asarray(arrs["priority_q_delta"], dtype=np.float64), nan=0.0)
            qd_abs = np.abs(qd[has])
            pm["qd_term"] += float(qd_abs.sum()) * self.diff_focus_q_weight
            pm["qd_raw"] += float(qd_abs.sum())
            pm["qd_rows"] += float(has.sum())
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
            # Config-free companions to the shares above. A `pol_scale` change
            # moves every `*_share` on rows it never applied to (see
            # `_accumulate_priority_mass`); these two move only when the DATA
            # moves, which is what makes them readable across a recalibration.
            "replay_pmass_kl_raw_mean": (
                pm["kl_raw"] / pm["kl_rows"] if pm["kl_rows"] > 0 else 0.0
            ),
            "replay_pmass_kl_raw_rows": pm["kl_rows"],
            "replay_pmass_qd_raw_mean": (
                pm["qd_raw"] / pm["qd_rows"] if pm["qd_rows"] > 0 else 0.0
            ),
            # Its own denominator, emitted for the same reason the kl one is: a
            # mean of 0.0 must be distinguishable from "no rows carried the
            # column". Accumulating a count and then not reporting it would be
            # this repo's signature defect inside the telemetry meant to catch it.
            "replay_pmass_qd_raw_rows": pm["qd_rows"],
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
            self._seed_shuffle_pool(seed_paths)

    def _seed_shuffle_pool(self, seed_paths: list[Path]) -> None:
        """Fill the hot pool at open through the STEADY-STATE refresh draw.

        The pool is memory-only and never checkpointed, so a restart starts
        from whatever this puts in it. It used to take the newest
        ``2 * refresh_shards`` shards outright — in production 10 of 834, i.e.
        1.2% of the window and a mean recency percentile of ~1.0. Measured
        against a warm arm (6 seeds, paired), the first three optimizer steps
        after a restart drew 100% of their rows from that slice and ran
        +0.45 policy_ce above warm, decaying to noise by step ~8 as live
        ingest diluted it.

        The draw is ``_load_refresh_chunks`` itself, asked for the same
        ``2 * refresh_shards`` shards the old slice took: same shard COUNT,
        same wall time, zero extra I/O, but recency-weighted, so the seeded
        rows land at whatever mean percentile the steady state already samples
        at instead of at 1.0. That percentile is 2/3 at the default recency
        exponent and ``(α+1)/(α+2)`` in general; the point of routing the seed
        through the same function is that it tracks the knob automatically.

        ONE call of ``2 * refresh_shards``, not two of ``refresh_shards``. Two
        calls are independent draws, so they can pick the same shard twice --
        certain, not merely possible, whenever the window holds fewer shards
        than ``2 * refresh_shards``, which double-weights those rows in the
        pool and loads a shard for nothing. One call draws without replacement
        and degrades to "every shard once" on a small window, which is what
        the old slice did there too.

        Filling the pool to its cap instead was considered and rejected:
        expected draws per loaded row is ``batch_size / injection_rate``
        regardless of pool size, so it fixes nothing distributional, and it
        would hand every read-only probe a multi-GB pool at construction.
        """
  # The comment below is a claim about CONSTRUCTION ORDER, and it is load
  # bearing: `_prefetch_rng` belongs to the prefetch thread, and
  # `np.random.Generator` has no internal lock. `__init__` calls
  # `_scan_existing_shards()` (the only caller of this method) strictly before
  # `_ensure_prefetch_thread()`, so today there is genuinely no concurrent
  # user. Nothing enforced that, and the refactor that breaks it is an
  # innocuous-looking one -- "start the prefetch thread earlier so the pool is
  # warm sooner" -- after which this draw and `_load_refresh_chunks` on the
  # prefetch thread advance the same bit-generator state non-atomically. The
  # result is torn or duplicated shard indices, i.e. a silently different
  # training-data composition, presenting as an unreproducible sampling
  # distribution rather than as a crash. Assert rather than document.
        assert self._prefetch_thread is None, (
            "_seed_shuffle_pool draws from _prefetch_rng on the MAIN thread, "
            "which is only safe because __init__ runs _scan_existing_shards() "
            "before _ensure_prefetch_thread(). That ordering has been broken: "
            "two threads now share one np.random.Generator, which is not "
            "thread-safe, and the corruption lands in which shards enter the "
            "shuffle pool. Restore the ordering, or give this call its own rng."
        )
        picked: list[int] = []
        rows_per_pick: list[int] = []
        loaded = self._load_refresh_chunks(
            shard_paths=seed_paths,
            refresh_shards=self._refresh_shards * 2,
  # Not self.rng: the sampling stream stays exactly where it was, and this
  # prefill is the prefetch thread's work done early. The thread does not
  # start until after this scan, so there is no concurrent user yet (asserted
  # above).
            rng=self._prefetch_rng,
            context="shuffle seed",
            chosen_out=picked,
        )
        for arrs in loaded:
  # Counted here, not from _shuffle_sizes: chunks can be dropped or
  # front-trimmed on the way in, and this must stay 1:1 with `picked`.
            rows_per_pick.append(int(_batch_dims(arrs)[0]))
            self._append_shuffle_arrays(arrs)
        self._trim_shuffle_buf()
        self._report_seed_recency(
            n_shards=len(seed_paths), picked=picked, rows_per_pick=rows_per_pick,
        )

    def _report_seed_recency(
        self, *, n_shards: int, picked: list[int], rows_per_pick: list[int],
    ) -> None:
        """One line proving which part of the window the restart seed came from.

        This is the observation that tells the two seeding regimes apart on the
        production path, and the only one: the pool is memory-only, so nothing
        downstream records where its rows came from. ``mean_recency`` is the
        row-weighted mean of ``(rank + 1) / n_shards`` over the seeded shards,
        oldest-first. Read it as ~1.0 = newest-only (the pre-fix defect, back),
        ~0.667 = the steady-state recency-weighted draw AT THE DEFAULT recency
        exponent; under another exponent the reference is the
        ``mean_rank_pct`` the ``[disk_buf] shard draw`` line printed at
        construction, not 0.667.
        """
        ranks = np.asarray(picked, dtype=np.float64)
        rows = np.asarray(rows_per_pick, dtype=np.float64)
        seeded_rows = int(rows.sum()) if rows.size else 0
        if n_shards <= 0 or ranks.size != rows.size or seeded_rows <= 0:
            mean_recency = float("nan")
        else:
            mean_recency = float(((ranks + 1.0) / n_shards * rows).sum() / seeded_rows)
        print(
            f"[disk_buf] shuffle seed: rows={seeded_rows} pool={self._shuffle_len()} "
            f"shards={len(picked)}/{n_shards} mean_recency={mean_recency:.3f} "
            f"(1.0=newest-only, {2 / 3:.3f}=steady-state refresh draw)",
            flush=True,
        )

    def _report_shard_recency_draw(self) -> None:
        """One line at construction proving WHICH recency draw this buffer runs.

        ``print``, not ``log.info``, and that is not a style choice: the Ray
        trial actor installs no logging handler, so INFO from this module
        reaches no file and no console on the production path — a proof-of-
        effect line that logs at INFO proves nothing, because its absence and
        its presence look identical. Every other buffer-side observation on
        this path (``[disk_buf] shuffle seed``, the empty-refresh alarm) is a
        ``print`` for the same reason.

        ``source=realized`` whenever the buffer tracks shards: the table is
        then summed over the very probability vector ``rng.choice`` will be
        handed, at this window size, AFTER any flooring — so it reports what
        the sampler does rather than the formula it was derived from, and
        ``floored=K/N`` says whether the floor engaged at all (0 everywhere in
        production; see ``MAX_SHARD_RECENCY_EXPONENT``). A fresh trial tracks
        no shards yet, and there the continuum form is the only honest answer;
        it is printed with ``source=analytic`` so the two are never confused.

        ⚠ Both forms are MARGINAL, single-shard: the probability that ONE draw
        lands in each decile. A real refresh takes ``refresh_shards`` WITHOUT
        replacement, and once the picks are a large fraction of the window that
        draw is pulled toward the window's UNWEIGHTED mean — downward, not
        upward, because taking most of the shards leaves recency nothing to
        select on. Measured against this line's ``mean_rank_pct``, α = 1.0,
        4000 draws: n = 834 / 5 picks (production) **+0.002**; n = 40 / 6
        **−0.009**; n = 10 / 6 **−0.048**; n = 6 / 6 **−0.139**. So the line is
        exact where it matters and optimistic on a nearly-exhausted window.
        Compare it against another run's line, not against the
        ``[disk_buf] shuffle seed`` line's ``mean_recency``, which is a
        row-weighted multi-pick number and answers a different question.
        ``test_the_printed_marginal_is_the_without_replacement_mean_at_scale``
        pins both ends of that.
        """
        alpha = self._shard_recency_exponent
        n_shards = len(self._shard_paths)
        if n_shards > 0:
            weights = shard_draw_weights(n_shards, alpha)
            deciles = shard_draw_decile_mass_realized(weights)
            ranks = np.arange(1, n_shards + 1, dtype=np.float64) / float(n_shards)
            mean_rank_pct = float((ranks * weights).sum())
            source = (
                f"realized n_shards={n_shards} "
                f"floored={shard_draw_floored_count(n_shards, alpha)}"
            )
        else:
            deciles = shard_draw_decile_mass(alpha)
            mean_rank_pct = (alpha + 1.0) / (alpha + 2.0)
            source = "analytic n_shards=0"
        table = " ".join(f"{m:.4f}" for m in deciles)
        print(
            f"[disk_buf] shard draw: recency_exponent={alpha:.4f} "
            f"(1.0=linear/production, 0.0=uniform) source={source} "
            f"decile_mass_oldest_to_newest=[{table}] "
            f"newest_decile={deciles[-1]:.4f} oldest_decile={deciles[0]:.4f} "
            f"mean_rank_pct={mean_rank_pct:.3f} (marginal, single-draw)",
            flush=True,
        )

    def _ensure_prefetch_thread(self) -> None:
        if self._deterministic_refresh:
  # Single gate for the whole feature: with no thread the prefetched slot
  # is never filled, so `sample_batch_arrays` always falls through to the
  # synchronous `_refresh_shuffle_buf`. See `_deterministic_refresh`.
            return
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
        context: str = "shuffle refresh",
        chosen_out: list[int] | None = None,
    ) -> list[dict[str, np.ndarray]]:
  # `context` and `chosen_out` exist for the open-time pool seed, which runs
  # this same draw (see _seed_shuffle_pool) and needs to label its log lines
  # and report which shards it drew. Both default to the steady-state
  # behaviour, so the refresh path is byte-identical to what it was.
        if not shard_paths:
            return []

        n_shards = len(shard_paths)
        n_pick = min(int(refresh_shards), n_shards)
        if n_pick <= 0:
            return []

  # `shard_paths` is oldest-first (iter_shard_paths sorts by shard index), so
  # rank 1 is the oldest shard in the window. At the default exponent this is
  # the same `np.arange(1, n+1)` vector it always was, drawn from the same rng
  # in the same one call -- see shard_draw_weights.
        weights = shard_draw_weights(n_shards, self._shard_recency_exponent)
        chosen_idxs = rng.choice(n_shards, size=n_pick, replace=False, p=weights)

  # ⚑ DECODED CONCURRENTLY, ASSEMBLED IN SUBMISSION ORDER. The shard DRAW above
  # is a single `rng.choice` that has already happened, so the loads consume no
  # randomness and their completion order cannot reach the sampled rows. What
  # would reach them is the order they are appended to `loaded` -- that is the
  # order they enter the shuffle pool -- so results are zipped back onto
  # `picks` positionally rather than appended as futures complete.
  # `[f.result() for f in futures]` iterates the SUBMISSION list, so a shard
  # that finishes first still lands in its own slot. `_load_refresh_chunks` is
  # therefore byte-identical to the serial version it replaces, which
  # `test_parallel_refresh_decode_is_byte_identical_to_serial` pins.
        picks = [int(idx) for idx in np.asarray(chosen_idxs, dtype=np.int64)]
        workers = min(_REFRESH_LOAD_WORKERS, len(picks))
  # ⚑ THE POOL IS FOR WORKER-THREAD CALLERS ONLY, and the branch is the same
  # numcodecs fact the pool exists for. blosc's internal 8-thread decompression
  # is enabled only on the MAIN thread, so a main-thread refresh already
  # decodes 8-way and handing it to a 4-way outer pool measured 0.82x -- a
  # REGRESSION. Production's refresh always runs on a worker (the trainer's
  # prefetch pool, or this buffer's own `_prefetch_loop`), so gating costs the
  # production path nothing and protects the offline/main-thread callers
  # (`_seed_shuffle_pool` at construction, scripts that sample inline).
        if threading.current_thread() is threading.main_thread():
            workers = 1
        if workers <= 1:
            results = [
                self._try_load_shard(shard_paths[idx], context=context)
                for idx in picks
            ]
        else:
  # A pool PER REFRESH, not a persistent one on the instance. Spawning
  # `workers` threads costs well under a millisecond against a multi-second
  # decode, and the `with` block joins them on every exit path including an
  # exception -- whereas a long-lived pool would need its own shutdown wired
  # into buffer teardown, which is precisely the abandoned-thread failure this
  # module already carries scar tissue for (see `_prefetch_loop`).
            try:
                with ThreadPoolExecutor(
                    max_workers=workers, thread_name_prefix="replay-refresh-load",
                ) as pool:
                    futures = [
                        pool.submit(
                            self._try_load_shard, shard_paths[idx], context=context,
                        )
                        for idx in picks
                    ]
                    results = [f.result() for f in futures]
            except RuntimeError:
  # ⚑ INTERPRETER TEARDOWN. `_prefetch_loop` is a DAEMON thread, so it can
  # still be inside a refresh when `threading._shutdown` runs, after which
  # `submit` raises "cannot schedule new futures after interpreter
  # shutdown". Nothing is wrong with the run at that point -- the process
  # is exiting -- so the only visible effect would be a spurious traceback
  # on a clean shutdown. Fall back to the serial path, which needs no new
  # threads; `_try_load_shard` swallows and accounts for its own failures,
  # so a torn-down interpreter yields Nones rather than raising again.
                results = [
                    self._try_load_shard(shard_paths[idx], context=context)
                    for idx in picks
                ]

        loaded: list[dict[str, np.ndarray]] = []
        for idx, arrs in zip(picks, results):
            if arrs is not None:
                loaded.append(arrs)
                if chosen_out is not None:
  # Appended only for shards that actually loaded, so it stays 1:1
  # with `loaded` and a vanished shard cannot shift the ranks.
                    chosen_out.append(idx)

  # The realized-vs-configured check this function never had: n_pick shards
  # were asked for, len(loaded) arrived. Only a fully empty result is worth
  # escalating -- a partial load still refreshes the buffer, and the trim
  # race makes small shortfalls routine.
        alarm = self._note_refresh_outcome(loaded_any=bool(loaded))
        if alarm is not None:
            streak, vanished, failed = alarm
            print(
                f"[disk_buf] WARNING: shuffle refresh has returned 0 of "
                f"{n_pick} requested shards {streak} times in a row "
                f"(vanished={vanished} failed={failed}, tracked_shards="
                f"{len(shard_paths)}); the shuffle buffer is no longer being "
                f"refreshed and training is sampling stale data",
                flush=True,
            )
        return loaded

    def _note_refresh_outcome(self, *, loaded_any: bool) -> tuple[int, int, int] | None:
        """Advance or reset the empty-refresh streak; report when to alarm.

        A refresh that returns nothing is INVISIBLE upstream: _prefetch_worker
        only stores a non-empty result, so the slot stays empty, the next tick
        retries, and training keeps sampling a shuffle buffer that has silently
        stopped being refreshed -- at roughly 10 attempts/s, forever, with no
        log line. We alarm on the STREAK rather than on any single empty
        result, which is normal while the window is being trimmed hard.

        The whole read-modify-decide runs under ``_prefetch_lock`` because
        ``_refresh_shuffle_buf`` (sampling thread) and ``_prefetch_worker``
        (its own thread) both reach here. Unsynchronised, the failure is not
        just a delayed alarm: a successful refresh writing 0 can interleave
        with a stale thread writing 50 and fire the alarm right after a
        refresh that actually worked. A false "training is sampling stale
        data" is worse than a late one -- it sends the reader chasing a
        non-problem and discredits the counter. Holding the lock across the
        decision also stops two threads reaching the threshold together and
        printing it twice.

        Returns ``(streak, vanished, failed)`` when the caller should print --
        totals snapshotted under the same lock so they agree with the streak
        -- or ``None``. The print itself stays with the caller, outside the
        lock, because it is I/O on a path the prefetch thread hits in bursts.
        """
        with self._prefetch_lock:
            if loaded_any:
                self._refresh_empty_streak = 0
                return None
            self._refresh_empty_streak += 1
            streak = self._refresh_empty_streak
            if streak == _REFRESH_EMPTY_STREAK_ALARM or (
                streak > _REFRESH_EMPTY_STREAK_ALARM
                and streak % (_REFRESH_EMPTY_STREAK_ALARM * 20) == 0
            ):
                return streak, self._refresh_vanished_total, self._refresh_failed_total
            return None

    @staticmethod
    def _shard_validation_fingerprint(sp: Path) -> tuple[int, int, int] | None:
        """(file count, total bytes, newest mtime_ns) over a shard's whole tree.

        A `.zarr` shard is a directory, so `stat()` on the shard path itself
        reports the DIRECTORY's mtime -- which does not move when a chunk file
        inside it is rewritten in place. Walking the tree is what makes the
        memo key actually track content; it costs 3.4 ms against a 878 ms load.

        Returns None if the shard cannot be stat'd at all (the window trimmer
        deleting it mid-walk is routine), which callers treat as "cannot vouch
        for this one" and fall back to validating.
        """
        try:
            n_files = 0
            total_size = 0
            newest_ns = 0
            for root, _dirs, fnames in os.walk(sp):
                for fn in fnames:
                    st = os.stat(os.path.join(root, fn))
                    n_files += 1
                    total_size += st.st_size
                    newest_ns = max(newest_ns, st.st_mtime_ns)
            if n_files == 0:
  # A legacy `.npz` shard is a FILE, not a tree: os.walk yields nothing.
                st = os.stat(sp)
                return (1, st.st_size, st.st_mtime_ns)
            return (n_files, total_size, newest_ns)
        except OSError:
            return None

    def _try_load_shard(
        self, sp: Path, *, context: str,
    ) -> dict[str, np.ndarray] | None:
        """Load one shard, accounting for -- rather than swallowing -- failures."""
  # Always RESOLVED, so the staging symlink farm (`stage_shards` points N
  # names at one converted shard) shares one entry per real shard rather than
  # re-validating the same bytes under every alias.
        key = str(sp.resolve())
        before = self._shard_validation_fingerprint(sp)
        already_validated = False
        if before is not None:
            with self._validation_memo_lock:
                already_validated = self._validated_shards.get(key) == before
        try:
            arrs, _ = load_shard_arrays(sp, lazy=False, validate=not already_validated)
        except Exception as exc:
            self._note_shard_load_failure(sp, exc, context=context)
            return None

  # ⚑⚑ RE-STAT AFTER THE DECODE, AND THE PR THAT ADDED THIS MEMO GOT IT WRONG.
  # The first cut fingerprinted only BEFORE the read and argued that could
  # "only be conservative". It cannot: a shard rewritten in the window between
  # the fingerprint and the decode is read as NEW bytes while
  # `already_validated` still reflects the OLD footprint, so validation is
  # skipped on content nothing ever checked. Review reproduced it and got 256
  # corrupt rows into the shuffle pool. A second walk costs ~2.6 ms against an
  # 878 ms load.
  #
  # The invariant is now: ONLY A FOOTPRINT THAT HELD STILL ACROSS THE WHOLE
  # READ MAY BE MEMOIZED, because it is the only one the bytes we actually
  # decoded can be attributed to. Anything else drops the entry, so the next
  # read re-validates -- the safe direction.
        after = self._shard_validation_fingerprint(sp)
  # Carried as the FOOTPRINT ITSELF rather than a bool, so the only value that
  # can ever be memoized below is one this line proved stable -- there is no
  # second place where "which fingerprint do we record" could get it wrong.
        stable = after if (after is not None and after == before) else None

        revalidated = False
        if already_validated and stable is None:
  # We skipped `validate_arrays` on the strength of a footprint that no
  # longer describes the file, so check what we actually decoded. This is
  # the branch that keeps the corrupt rows out.
            try:
                validate_arrays(arrs)
            except Exception as exc:
                self._note_shard_load_failure(sp, exc, context=context)
                with self._validation_memo_lock:
                    self._validated_shards.pop(key, None)
                return None
            revalidated = True

        with self._validation_memo_lock:
            if already_validated and not revalidated:
                self._shard_validations_skipped += 1
            else:
                self._shard_validations_run += 1
            if stable is not None:
  # Recorded only once `validate_arrays` has returned without raising for
  # THIS footprint, so a shard that fails validation is never memoized as
  # good.
                self._validated_shards[key] = stable
            else:
                self._validated_shards.pop(key, None)
        self._maybe_log_validation_memo()
        return arrs

    def _maybe_log_validation_memo(self) -> None:
        """Put the memo's hit/miss counts somewhere production can see them.

        A counter only tests ever read is this repo's accepted-then-ignored
        smell wearing a performance hat: if the memo silently stopped hitting
        -- a corpus being rewritten under the run, a fingerprint that never
        matches -- the whole 7.31x would be handed back with nothing in the log
        saying so. Throttled rather than per-refresh because production runs
        `refresh_interval` as low as 1.
        """
        now = time.time()
        with self._validation_memo_lock:
            if now - self._validation_memo_logged_at < _VALIDATION_MEMO_LOG_INTERVAL_S:
                return
            self._validation_memo_logged_at = now
            run = self._shard_validations_run
            skipped = self._shard_validations_skipped
            memoized = len(self._validated_shards)
        total = run + skipped
        rate = (skipped / total) if total else 0.0
        print(
            f"[disk_buf] refresh: shard validations run={run} skipped={skipped} "
            f"(memo hit rate {rate:.1%}, {memoized} shard(s) memoized)",
            flush=True,
        )

    def _note_shard_load_failure(
        self, sp: Path, exc: BaseException, *, context: str,
    ) -> None:
        """Classify a shard-load failure as the benign trim race or a real fault.

        Classification is by TRACKING, not by exception type, and deliberately
        so. ``_enforce_window`` pops a shard out of ``_shard_paths`` under the
        lock and only then calls ``delete_shard_path``, so a path that is no
        longer tracked was deliberately deleted and losing it is expected.
        Going by exception type instead would be both wrong and fragile:
        ``zarr.open_group`` on a missing group raises ``GroupNotFoundError``
        (a ``ValueError``, NOT a ``FileNotFoundError``), and because
        ``delete_shard_path`` uses ``shutil.rmtree(..., ignore_errors=True)``
        a reader can also catch a half-deleted group and surface a third kind
        of error entirely. The tracking check is immune to all of that and to
        the zarr version.
        """
        now = time.monotonic()
  # Counters and throttle live under the lock we already need for the
  # membership test: _refresh_shuffle_buf runs on the sampling thread while
  # _prefetch_worker runs on its own, so both can land here at once and
  # bare `+= 1` would drop counts and let the throttle emit twice.
        with self._prefetch_lock:
            if sp not in self._shard_paths:
                self._refresh_vanished_total += 1
                return
            self._refresh_failed_total += 1
            failed = self._refresh_failed_total
            vanished = self._refresh_vanished_total
            if now - self._shard_load_warned_at < _SHARD_LOAD_WARN_INTERVAL_S:
                return
            self._shard_load_warned_at = now

  # Printed outside the lock -- this is I/O on a path the prefetch thread
  # hits in bursts, and the sampling thread contends for the same lock.
  # print, not log: this runs inside the Ray trial actor, whose stdout
  # train.sh captures into the training log, while logging traffic can
  # reach only the Ray session logs.
        print(
            f"[disk_buf] WARNING: {context} failed to load a TRACKED shard "
            f"{sp.name}: {type(exc).__name__}: {exc} "
            f"(failed={failed} vanished={vanished} since start)",
            flush=True,
        )

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

    def _reject_if_read_only(self, op: str) -> None:
        if self._read_only:
            raise RuntimeError(
                f"DiskReplayBuffer({self._shard_dir}) was opened read_only=True; "
                f"{op} would mutate the shard window. Open it writable, or work "
                f"on a copy of the directory."
            )

    def add(self, sample: ReplaySample) -> None:
        self.add_many([sample])

    def add_many(self, samples: list[ReplaySample]) -> None:
        """Add samples: into shuffle buffer immediately, flush to disk when full."""
        self._reject_if_read_only("add_many")
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
        self._reject_if_read_only("add_many_arrays")
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
        self._reject_if_read_only("flush")
        if self._write_buf_rows > 0:
            self._flush_shard_arrays(self._take_write_prefix(self._write_buf_rows))
            self._enforce_window()

    def enforce_window(self) -> None:
        """Apply the current capacity limit immediately."""
        self._reject_if_read_only("enforce_window")
        self._enforce_window()

    def _claim_writer(self) -> None:
        """Take the shard dir's advisory writer lock, or say who already has it.

        Detection only: a second writer keeps writing. What it produces is
        silent duplication rather than a crash -- two buffers allocate shard
        indices from their own counters, so the streams interleave instead of
        colliding, and every row written while both are live lands in the
        window twice. The lock is released when the process exits (or in
        ``close()``), so a crash cannot leave a stale one behind.
        """
        if self._writer_lock_fd is not None:
            return
        lock_path = self._shard_dir / _WRITER_LOCK_NAME
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        except OSError as exc:
            log.warning("[disk_buf] cannot open writer lock %s: %s", lock_path, exc)
            self._writer_lock_reported = True
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if not self._writer_lock_reported:
                holder = ""
                with contextlib.suppress(Exception):
                    holder = os.pread(fd, 64, 0).decode("utf-8", "replace").strip()
                log.error(
                    "[disk_buf] CONCURRENT WRITER: %s is already being written by pid %s; "
                    "this process (pid %d) is writing into it too. Every row ingested from "
                    "here on lands in the window TWICE, and each writer draws its own "
                    "holdout split, so a duplicated row can sit in one holdout and the "
                    "other's training set (audit G7).",
                    self._shard_dir, holder or "unknown", os.getpid(),
                )
                self._writer_lock_reported = True
            os.close(fd)
            return
        with contextlib.suppress(Exception):
            os.ftruncate(fd, 0)
            os.pwrite(fd, f"{os.getpid()}\n".encode(), 0)
        self._writer_lock_fd = fd

    def _release_writer(self) -> None:
        fd = self._writer_lock_fd
        self._writer_lock_fd = None
        if fd is None:
            return
        with contextlib.suppress(Exception):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(Exception):
            os.close(fd)

    def _next_free_shard_path(self) -> Path:
        """Shard path for the next index, skipping any that already exists.

        A path that exists at the index this buffer thinks is free means
        something else wrote it. Overwriting would silently destroy the other
        writer's rows on top of duplicating our own, so step over it instead.
        """
        path = local_shard_path(self._shard_dir, self._shard_index)
        skipped = 0
  # `is_symlink` too: a seeded salvage link at this index is occupied even
  # when its target has since been removed, and writing through a dangling
  # link would land the shard in the salvage pool.
        while path.exists() or path.is_symlink():
            skipped += 1
            self._shard_index += 1
            path = local_shard_path(self._shard_dir, self._shard_index)
        if skipped:
            log.error(
                "[disk_buf] shard index collision in %s: skipped %d existing shard(s) to "
                "reach %s. Another writer owns those indices (audit G7).",
                self._shard_dir, skipped, path.name,
            )
        return path

    def _flush_shard_arrays(self, arrs: dict[str, np.ndarray]) -> None:
        """Write a shard to disk."""
  # BEFORE `_claim_writer`, not after: the writer lock is a real file created
  # inside the shard dir, so claiming it is itself a mutation of the directory
  # a read-only open promised not to touch.
        self._reject_if_read_only("_flush_shard_arrays")
        self._claim_writer()
        path = self._next_free_shard_path()
        saved_path = save_local_shard_arrays(path, arrs=arrs)
        n = int(arrs["x"].shape[0])
        self._append_shard_record(saved_path, n)
        self._shard_index += 1

    def _enforce_window(self) -> None:
        """Delete oldest shards when total exceeds capacity.

        A no-op under ``read_only=True``. This is the one deleter, and
        ``__init__`` reaches it through ``_scan_existing_shards`` before the
        caller ever gets the object back, so the flag has to be honoured here
        for a read-only open to be safe at all (audit G12).
        """
        if self._read_only:
            return
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
  # The allocation below is `np.zeros` over the UNION of the selected chunks'
  # keys, so a field one chunk carries and another lacks arrives as zeros for
  # the second chunk's rows. For every optional field that is the schema's own
  # default (absent flag == 0, absent target == zeros) and harmless. For the two
  # fields whose schema default is NOT zeros it is silent corruption in the
  # direction that cannot be seen: `has_policy = 0` deletes those rows from the
  # main policy loss and its accuracy stats, and `policy_loss` then reports the
  # survivors' mean with nothing counting the drop. Refuse instead of guessing.
  # Unreachable today (both names are in `_REQUIRED_STORAGE_FIELDS`, every
  # producer path emits them) -- this exists so that if it ever becomes
  # reachable the run stops instead of quietly training on fewer rows than it
  # reports. Cost: two `in` scans over the selected chunks, not the ~60 keys.
        if len(selected) > 1:
            for name in sorted(NONZERO_DEFAULT_STORAGE_FIELDS):
                present = sum(1 for dense_rows, _ in selected if name in dense_rows)
                if 0 < present < len(selected):
                    raise ValueError(
                        f"shuffle-buffer chunks disagree about {name!r} "
                        f"({present}/{len(selected)} carry it); zero-filling it would "
                        "silently change training rows -- refusing to sample",
                    )
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
        self._reject_if_read_only("clear")
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
        self._release_writer()
        self._writer_lock_reported = False
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
  # ⚑ THIS CANNOT STOP A RUNNING PREFETCH THREAD, so do not read it as one.
  # `_ensure_prefetch_thread` passes the BOUND METHOD `self._prefetch_loop` as
  # the thread target, and a running thread is an external GC root via
  # `threading._active` -- so a buffer whose prefetch thread is alive is
  # strongly reachable and is never collected, and this finaliser never runs.
  # Measured 2026-08-24: drop the last reference, `gc.collect()` three times,
  # thread still alive. It fires only for a buffer that never started a thread
  # (`deterministic_refresh`, `refresh_interval <= 0`, or never sampled), where
  # it still does useful work releasing the writer lock. Anything that needs
  # the thread STOPPED must call `close()` explicitly; the test suite does it
  # from an autouse fixture in `tests/conftest.py`.
        with contextlib.suppress(Exception):
            self.close()
