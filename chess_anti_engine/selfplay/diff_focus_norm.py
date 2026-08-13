"""Scale-free robust normalizer for the difficulty-focus curriculum.

WHY THIS EXISTS. ``difficulty = |q_delta| * q_weight + kl * pol_scale`` feeds
two decisions: ``keep_prob = clamp(difficulty * slope, min_keep, 1.0)`` (which
plies are RECORDED AT ALL) and the stored ``priority`` (which recorded rows the
replay sampler draws). Both thresholds are FIXED ABSOLUTE numbers, so they only
mean anything relative to the absolute scale of ``kl`` -- and ``kl`` is
KL(policy prior || search visit target), whose scale is a property of the SEARCH
CONFIGURATION, not of the curriculum. Nothing in the diff-focus group knows that.

On 2026-08-09 a search bundle (``gumbel_policy_temp`` -> 1.5, ``gumbel_c_scale``
-> 0.1) moved the KL scale by ~11.7x on fresh games. No diff-focus key changed
and every value stayed in range, but the clamp saturated: measured
``diff_focus_keep_rate`` 0.803 -> 0.956 and ``diff_focus_keep_limited_frac``
0.368 -> 0.114. The ply-selection curriculum went inert and the sampler
concentrated on the highest-KL rows.

⚑ This is a distinct species of the codebase's signature defect. The usual form
is a value accepted and then silently ignored. This is worse and harder to see:
a knob whose CALIBRATION silently depends on the absolute scale of a quantity
that a DIFFERENT, unrelated change is free to move. Every value here was read;
every value was applied; the numbers were simply no longer comparable to the
constants they were being compared against.

THE FIX. Divide ``difficulty`` by a running robust scale before the fixed
thresholds see it, so the thresholds live in units of "typical difficulty right
now" instead of nats. The normalized statistic is then invariant to ANY positive
rescaling of the difficulty distribution -- which is exactly the property whose
absence caused the incident, and which
``tests/test_diff_focus_norm.py`` asserts as a negative control.

WHY THE MEDIAN, AND NOT A ROBUST Z-SCORE. ``difficulty`` is non-negative and
right-skewed, and ``difficulty == 0`` carries meaning ("search agreed with the
prior and the value did not move") that must keep mapping to ``min_keep``.
Subtracting a location estimate would destroy that anchor and make some
difficulties negative, breaking the monotone map into ``keep_prob``. Scale-only
normalization by a quantile is the operation that fits the quantity: it keeps
zero at zero, is equivariant under ``d -> c*d`` for any ``c > 0``, and at the
median has the maximal 50% breakdown point. ``quantile`` is exposed so the
reference point can be moved without a code change; the measured cross-era
drift is flat in it between p25 and p60 (see the ledger entry for the sweep).

WHAT THE ESTIMATOR'S POPULATION ACTUALLY IS -- READ THIS BEFORE TRUSTING IT.
It is the last ``window`` POLICY-BEARING plies observed by ONE INSTANCE. One
instance is built per ``SelfplayState``, i.e. per ``play_batch`` call. It is NOT
a global quantile over the fleet, and calling it one would be the
``same name != same measurement`` error:

* Only ``has_policy`` rows are observed. ``keep_prob`` is consumed for exactly
  those rows (``selfplay/finalize.py`` gates the drop on ``row_has_policy``) and
  the ``diff_focus_*`` telemetry counts exactly those rows. An estimator fed the
  fast-ply rows too would normalize one population by another's median.
* Local, not global, because a global quantile would need a server round-trip
  per ply. That is affordable ONLY because the quantity being estimated is a
  property of the (published net, search config) pair, which is identical across
  workers at any instant -- so local medians estimate the same population
  parameter and differ only by sampling noise. Measured on 744,715 real
  post-bundle rows, the median of an 8192-sample draw has relative sd 1.81%
  (95% spread of the ratio between two independent estimators: 1.072). That is
  negligible against the 2.83x scale shift this exists to absorb, and it is a
  real, bounded cost to state rather than a property to assume.

⚑⚑ "PER WORKER" IS NOT WHAT PRODUCTION REALIZES, AND THAT IS THE EXPENSIVE PART.
This docstring said "per worker process" until 2026-08-12. It is per
``SelfplayState``, and the worker runs ``--threaded-selfplay --selfplay-threads
32``: ``_run_selfplay_threaded`` submits one ``play_batch`` per thread, so a
worker holds **32 independent estimators**, and the live 4-worker run holds
**128**. Each one pays ``warmup`` separately AND sees only 1/32 of its worker's
ply rate, so the warm-up costs 32x the rows and takes 32x the wall clock that
reading ``diff_focus_norm_warmup: 1024`` suggests: 128 x 1024 = **131,072
policy-bearing plies per restart**, ~8.7% of the 1.5M-row replay window.

Measured 2026-08-12 on the LIVE replay window (798 shards, 1,498,168 policy
rows) -- the window is a rolling buffer seeded across trials, so attribute rows
by when the shard was written, not by which trial directory holds it. It held
two restart transients at that moment. In each, the over-clip fraction (see the
armed-check below) runs 1.3-2.0% for ~35 shards and does not reach zero for ~65,
about 37 minutes; priorities reach **17.59** against ``diff_focus_norm_clip:
8.0``, and **~160k of the 1,498,168 rows (10.7%)** were written on the unarmed
branch. ``progress.csv`` shows the same two events independently:
``diff_focus_priority_max`` above the clip for iterations 1-8 after one restart,
1-23 after the previous one, and again at iteration ~140 -- **a mid-run session
restart re-pays it, so this is not a boot-only cost.**
``diff_focus_norm_shared`` (default OFF) lets the worker hand ONE instance to
every selfplay thread, which removes the 32x multiplier without changing any
arithmetic on the armed path. When it is on, ``observe`` is called from all 32
threads, which is why the ring update below is under a lock.

WARM-UP, AND HOW TO TELL IT APART FROM WORKING. Until ``warmup`` samples have
been seen, ``scale`` is 0.0 and the caller passes 0.0 to the C path, which then
takes the ORIGINAL unnormalized branch. The warm-up regime is therefore not a
half-configured third behaviour; it is exactly the pre-fix behaviour.
⚑ Do not read "pre-fix" as "calibrated": since the 2026-08-09 search bundle the
pre-fix branch is the DEcalibrated one (that is the whole reason this module
exists), so warm-up rows get ``diff_focus_slope: 3.0`` against an 11.7x-moved KL
scale and an unclipped ``priority``. Keeping warm-up on one documented code path
is a design choice about SIMPLICITY, not a claim that the rows are fine. It is
observable without new plumbing: once armed, the stored ``priority`` is
``difficulty / q50`` hard-capped at ``clip``, so
``diff_focus_priority_max <= diff_focus_norm_clip`` holds in ``progress.csv`` by
construction and cannot hold on the unarmed path (measured unarmed max: 96.0
against a clip of 8.0). That is a value read on an existing column, not a
presence check.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["DiffFocusNormalizer"]


class DiffFocusNormalizer:
    """Running robust scale for ``difficulty``, over ONE INSTANCE's recent plies.

    Not "one worker's" — that is what this line said until 2026-08-12, and the
    module docstring above explains why it costs 32x. One instance is built per
    ``SelfplayState``, i.e. per selfplay thread, unless ``norm_shared`` is on.

    ``observe`` takes the RAW (unnormalized) difficulties of the policy-bearing
    plies in one ply batch; ``scale`` returns the reference quantile of the
    retained window, or ``0.0`` while the estimator is not yet armed. ``0.0`` is
    the caller's "normalization off" sentinel, so an unarmed estimator and a
    disabled feature take the identical code path.

    ``observe`` is thread-safe. One instance may be shared by every selfplay
    thread of a worker (``diff_focus_norm_shared``); without the lock two
    concurrent calls read the same ``_pos``, write the SAME ring slots and then
    both advance ``_count``, leaving the tail of the ring at its 0.0 fill while
    the estimator believes it is full -- a silently depressed reference quantile,
    i.e. inflated priorities, which is the failure this whole module exists to
    prevent. The lock is uncontended in the unshared configuration and is taken
    once per ply batch, not once per row.
    """

    __slots__ = (
        "_armed_logged", "_buf", "_count", "_lock", "_pos", "_quantile", "_scale",
        "_warmup",
    )

    def __init__(self, *, window: int, warmup: int, quantile: float) -> None:
        if window <= 0:
            raise ValueError(f"diff_focus_norm_window must be > 0, got {window}")
        if not 0.0 < quantile < 1.0:
            raise ValueError(
                f"diff_focus_norm_quantile must be in (0, 1), got {quantile}",
            )
        # A warmup longer than the window could never be reached from a ring
        # that only ever holds `window` samples' worth of signal; clamp rather
        # than let a misconfiguration silently pin the feature off forever.
        self._warmup = max(1, min(int(warmup), int(window)))
        self._quantile = float(quantile)
        self._buf = np.zeros(int(window), dtype=np.float64)
        self._pos = 0
        self._count = 0
        self._scale = 0.0
        self._armed_logged = False
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        """Samples observed since construction (saturating at the window size)."""
        return int(self._count)

    @property
    def armed(self) -> bool:
        """True once ``scale`` is a usable positive reference quantile."""
        return self._scale > 0.0

    @property
    def scale(self) -> float:
        """Reference quantile of the retained window, or 0.0 while unarmed.

        Read without the lock: it is a single float attribute, and under
        CPython's GIL an attribute load is one bytecode, so a reader gets either
        the previous or the next armed value and never a torn one. (That is a
        CPython guarantee, not a language one — on a free-threaded build this
        read would need the lock, or the attribute would need to be an atomic
        box.) Taking the lock here would serialise every ply batch of every
        selfplay thread against the one writer for no added guarantee today.
        """
        return float(self._scale)

    def observe(self, difficulties: np.ndarray) -> None:
        """Fold one ply batch's RAW difficulties into the window.

        Non-finite values are dropped rather than clamped: the C path already
        substitutes 1.0 for a non-finite ``difficulty`` when computing
        ``keep_prob``, and folding that substituted constant into the scale
        estimate would let a numerical accident drag the reference quantile.
        """
        d = np.asarray(difficulties, dtype=np.float64).ravel()
        d = d[np.isfinite(d)]
        if d.size == 0:
            return

        with self._lock:
            self._observe_locked(d)

    def _observe_locked(self, d: np.ndarray) -> None:
        """Ring update + re-quantile. Caller holds ``self._lock``."""
        n = self._buf.size
        # More arrivals than the ring holds: keep only the newest `n`, which is
        # what the ring would contain after replaying them in order.
        if d.size >= n:
            d = d[-n:]
            self._buf[:] = d
            self._pos = 0
            self._count = n
        else:
            end = self._pos + d.size
            if end <= n:
                self._buf[self._pos:end] = d
            else:
                split = n - self._pos
                self._buf[self._pos:] = d[:split]
                self._buf[: end - n] = d[split:]
            self._pos = end % n
            self._count = min(n, self._count + d.size)

        if self._count < self._warmup:
            return

        q = float(np.quantile(self._buf[: self._count], self._quantile))
        # A non-positive reference quantile means the window is degenerate (an
        # all-zero difficulty stream). Dividing by it would produce inf/nan
        # priorities on the production path, so stay unarmed and keep the
        # original behaviour instead.
        if not np.isfinite(q) or q <= 0.0:
            self._scale = 0.0
            return

        self._scale = q
        if not self._armed_logged:
            self._armed_logged = True
            log.warning(
                "diff-focus normalization ARMED after %d plies: reference q%.2f = "
                "%.6g. From here `priority` is difficulty/q%.2f (capped) and "
                "`keep_prob` uses diff_focus_norm_slope, NOT diff_focus_slope. "
                "Rows recorded before this line are in RAW units.",
                self._count, self._quantile, q, self._quantile,
            )
