"""Value-head optimism stratified by an OBJECTIVE (Stockfish) evaluation.

The recorded finding this exists to make re-measurable: the value head reads
roughly even where deep Stockfish reads lost, and the proposed cause is that
the WDL target's game-outcome half comes from a PID-HANDICAPPED opponent, so
losing positions really were survivable in the data. That claim was first
measured on a tail-selected sample (the worst value mismatch per lost game),
which can only give an upper bound.

This module is the level instrument for that claim. It is deliberately NOT a
second copy of ``scripts/value_regret.py``: that scores value RANKING (the
1-ply deep-SF regret of the move the value head would pick) and is the
project's VALUE yardstick. This scores value LEVEL (a signed expected-score
error against an SF ruler), which is a different quantity, so mixing the two
into one number would quietly change what "value_regret said X" means.

Three rules the design follows:

1. **The ruler picks the sample, never the net.** Rows are bucketed by an SF
   evaluation of the row's own position. The net's opinion is the thing under
   test and must not select the stratum.
2. **Stratify, do not filter.** Every bucket across the range is reported, so
   the losing-position number is read against the same instrument's behaviour
   everywhere else. A single-bucket error rate is uninterpretable.
3. **Score the TARGET next to the HEAD.** A head that matches an optimistic
   target is a data defect; a head that misses a sound target is a fitting
   defect. Reporting ``net``, ``target`` and its three components in the same
   bucket separates them.

**``net_minus_target`` is the PRIMARY axis, not ``net_minus_sf``.** Bucketing on
a ruler can bias a comparison against that ruler, but ``net`` and ``target`` are
both measured on the same row and neither is computed from the bucket, so a head
that fit its target perfectly would read zero in every bucket under ANY
bucketing. That axis is artifact-free by construction; the ruler-relative one is
not, and must be read next to the check below.

**Whether bucketing biases the ruler-relative number is an empirical question,
and this module answers it rather than assuming.** Regression toward the mean
would require the TRUE value of positions in an extreme bucket to be less
extreme than the ruler says. The realized game outcome is an unbiased draw of
exactly that quantity, so ``outcome_score`` versus ``sf_ruler_score`` measures
the bias directly, and ``perfect_head_tail_asymmetry`` turns it into the NULL
that ``tail_asymmetry`` must be judged against. On the live 2026-07-31 sample
the outcome was MORE extreme than the ruler in both tails, i.e. the bias runs
the other way and the head's compression is real, not manufactured. Never quote
``tail_asymmetry`` against a null of zero: it is not zero under the shuffle
control either (-0.0051 measured).

Expected score (``W + 0.5*D``) is the primary unit because it never saturates.
The centipawn equivalent is reported alongside it for comparability with the
original claim, via the exact inverse of the production cp-logistic
(``chess_anti_engine.stockfish.wdl.cp_to_wdl``) with an explicit clamp and a
reported clamp rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Bucket edges in centipawns from the scored side's point of view. -300 is the
# level the original claim reported for Stockfish (-300.9 cp); +-100 separates
# "one side is meaningfully better" from balanced. Shared so two scorers can
# never put the same position in different buckets (same reason
# CRITICALITY_GAP_EDGES lives in eval/audit.py).
SF_EVAL_BUCKET_EDGES: tuple[float, ...] = (-300.0, -100.0, 100.0, 300.0)
SF_EVAL_BUCKET_NAMES: tuple[str, ...] = (
    "lost(<=-300)", "losing(-300,-100]", "balanced(-100,+100]",
    "winning(+100,+300]", "won(>+300)",
)

# The cp axis is clamped here before any comparison. cp_to_wdl saturates: at
# the production slope a +-1500 cp score is already 0.9999 / 0.0001, so the
# inverse is numerically meaningless beyond it and a single decisive score
# would otherwise dominate a bucket mean.
#
# The clamp is set by that saturation point, NOT by the mate band: it used to
# be justified as "1500 is _MATE_BASE_CP so nothing real lands above it", which
# was never quite true (SF emits |cp| ~ 20000 in decisive endgames) and is now
# plainly false — `mate_to_effective_cp` maps mates to ~+/-100000 so that they
# outrank every raw cp score. Mates and large cp labels both clamp to the rail
# here, which is the intended behaviour for a bucket mean.
CP_CLAMP: float = 1500.0


# Piece values in plane order (pawn, knight, bishop, rook, queen, king), used
# only by the shard-integrity guard below.
_PIECE_VALUES: tuple[float, ...] = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)

# Minimum rank correlation between a shard's own material balance and its own
# SF label. Anything sane clears this by a wide margin (live shards measure
# +0.58..+0.71); a detached label block measures ~0.00. See
# `sf_label_attachment_corr`.
SF_LABEL_ATTACHMENT_MIN: float = 0.25

# Maximum share of labelled rows carrying NO MultiPV block. The gate's sharp
# axis for SF-desync corruption.
#
# THE PRIMARY JUSTIFICATION IS THE SHAPE OF THE SOUND DISTRIBUTION, NOT THE GAP.
# Over all 834 shards of trial 13a9f, **89.9% of accepted shards read EXACTLY
# 0.000000**; median 0.000000, p99 0.004603, max 0.008032. A sound shard has a
# HARD FLOOR AT ZERO — every labelled row is supposed to carry its candidate
# list — so any materially nonzero rate is anomalous wherever the cut is placed.
# That argument does not depend on the threshold.
#
# The separation gap is real but SECONDARY, and partly circular: accepted max
# 0.008032 vs first rejected 0.010511 is a 0.002478 gap with 22.2x headroom over
# the accepted p90 (0.000450) — but that gap only exists AFTER removing the 122
# shards this threshold rejects. Sensitivity, which is the honest way to read it:
# 0.008 -> 123 rejects, 0.009 -> 122, 0.01 -> 122, 0.02 -> 114, 0.005 -> 128. The
# 0.008-0.01 plateau is why the exact value does not matter much.
# See `sf_multipv_missing_rate`.
SF_MULTIPV_MISS_MAX: float = 0.01

# The bestmove-is-first-legal rate is kept as a DIAGNOSTIC and is OFF as a gate
# by default (1.0 = never reject). It has no defensible threshold: over the same
# 834 shards the highest sound value is 0.1496 and the lowest corrupt value is
# 0.1505, so any cut sits in a 0.0009 gap between two adjacent order statistics
# of the same quantity — 1.7x headroom, versus 22.2x for the MultiPV axis. Setting
# it at 0.15 leaked seven shards sitting *inside* runs of rejects. That is
# precisely the "a gate tuned on the episode that produced it detects that
# episode" failure, so the number is reported and not enforced. It also catches
# nothing the other two axes miss on this data (union is 120 shards either way).
SF_DESYNC_MAX: float = 1.0

# Minimum evaluable rows before an axis will return a value at all. Below this a
# rate is noise, and a noisy rate that happens to fall on the pass side is worse
# than no reading — hence `AxisStatus.TOO_FEW_ROWS`, which callers must treat as
# a reject rather than a pass.
SF_AXIS_MIN_ROWS: int = 30


def sf_eval_bucket(cp: float, edges: tuple[float, ...] = SF_EVAL_BUCKET_EDGES) -> int:
    """Index into the bucket list for one side-to-move cp evaluation."""
    for i, edge in enumerate(edges):
        if cp <= edge:
            return i
    return len(edges)


def sf_eval_bucket_array(
    cp: np.ndarray, edges: tuple[float, ...] = SF_EVAL_BUCKET_EDGES,
) -> np.ndarray:
    """Vectorised `sf_eval_bucket` (int64 indices), equal per element."""
    return np.searchsorted(
        np.asarray(edges, dtype=np.float64),
        np.asarray(cp, dtype=np.float64),
        side="left",
    ).astype(np.int64)


def bucket_names_for(edges: tuple[float, ...]) -> tuple[str, ...]:
    """Half-open interval labels for arbitrary edges: (lo, hi], lowest closed.

    The DEFAULT edges are the comparable ones — two runs with different edges
    are two different instruments and their bucket numbers must not be put in
    the same table.
    """
    if edges == SF_EVAL_BUCKET_EDGES:
        return SF_EVAL_BUCKET_NAMES
    if not edges:
        return ("all",)
    names = [f"<={edges[0]:+.0f}"]
    names += [f"({edges[i]:+.0f},{edges[i + 1]:+.0f}]" for i in range(len(edges) - 1)]
    names.append(f">{edges[-1]:+.0f}")
    return tuple(names)


def expected_score(wdl: np.ndarray) -> np.ndarray:
    """(..., 3) WDL -> expected score W + 0.5*D, renormalised, in [0, 1]."""
    p = np.clip(np.asarray(wdl, dtype=np.float64), 0.0, None)
    total = np.clip(p.sum(axis=-1), 1e-12, None)
    return (p[..., 0] + 0.5 * p[..., 1]) / total


def cp_to_expected_score(cp: np.ndarray, *, slope: float, draw_width_cp: float) -> np.ndarray:
    """Closed form of ``expected_score(cp_to_wdl(cp))``.

    With ``draw_width_cp >= 0`` the logistic's draw mass is never clipped
    (p_win + p_loss <= 1 always holds), so the normaliser is exactly 1 and
    score = 0.5 * (sigmoid(s*(cp - d)) + sigmoid(s*(cp + d))).
    ``tests/test_value_optimism.py`` pins this against ``cp_to_wdl`` itself.
    """
    if slope <= 0.0 or draw_width_cp < 0.0:
        raise ValueError(f"needs slope>0 and draw_width_cp>=0, got {slope=} {draw_width_cp=}")
    c = np.asarray(cp, dtype=np.float64)
    lo = 1.0 / (1.0 + np.exp(-slope * (c - draw_width_cp)))
    hi = 1.0 / (1.0 + np.exp(-slope * (c + draw_width_cp)))
    return 0.5 * (lo + hi)


def expected_score_to_cp(
    score: np.ndarray, *, slope: float, draw_width_cp: float,
    cp_clamp: float = CP_CLAMP,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert `cp_to_expected_score` by bisection on [-cp_clamp, +cp_clamp].

    Returns ``(cp, clamped)`` where ``clamped`` marks entries whose score fell
    outside the representable band and was pinned to the clamp. Callers MUST
    report the clamp rate: a silently pinned tail is exactly how an absolute
    cp bar stops meaning what its name says.
    """
    s = np.asarray(score, dtype=np.float64)
    lo_cp = -abs(float(cp_clamp))
    hi_cp = abs(float(cp_clamp))
    s_lo = cp_to_expected_score(np.array(lo_cp), slope=slope, draw_width_cp=draw_width_cp)
    s_hi = cp_to_expected_score(np.array(hi_cp), slope=slope, draw_width_cp=draw_width_cp)
    clamped = (s <= s_lo) | (s >= s_hi)
    lo = np.full(s.shape, lo_cp, dtype=np.float64)
    hi = np.full(s.shape, hi_cp, dtype=np.float64)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        too_low = cp_to_expected_score(mid, slope=slope, draw_width_cp=draw_width_cp) < s
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return np.clip(0.5 * (lo + hi), lo_cp, hi_cp), clamped


@dataclass(frozen=True)
class AxisReading:
    """One integrity-axis measurement, with WHY it is unusable when it is.

    A bare NaN collapses "the field is missing", "too few rows to estimate" and
    "genuinely corrupt" into one indistinguishable value. A shard written before
    the field existed would then be rejected under the same string as a poisoned
    one — or, if the sign were ever flipped, admitted under it. The status is
    what keeps those three apart in the reject log.
    """

    value: float
    status: str = "ok"

    @property
    def usable(self) -> bool:
        return self.status == "ok" and bool(np.isfinite(self.value))

    def describe(self, name: str) -> str:
        if self.status == "ok":
            return f"{name} {self.value:+.4f}"
        return f"{name} UNEVALUABLE ({self.status})"


def material_balance_from_planes(x: np.ndarray) -> np.ndarray:
    """Side-to-move material balance from the stored input planes.

    Planes 0..5 are the mover's pawn..king and 6..11 the opponent's, in both
    the legacy and lc0-root layouts (see eval/audit.decode_board_from_planes),
    so this needs no board reconstruction and no chess library.
    """
    arr = np.asarray(x[:, 0:12])
    counts = (arr > 0.5).sum(axis=(2, 3)).astype(np.float64)
    values = np.asarray(_PIECE_VALUES, dtype=np.float64)
    return (counts[:, 0:6] * values).sum(axis=1) - (counts[:, 6:12] * values).sum(axis=1)


def rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, ties averaged. NaN for degenerate input."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.size < 3 or x.size != y.size:
        return float("nan")

    def _ranks(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="stable")
        s = v[order]
        r = np.empty(v.size, dtype=np.float64)
        i = 0
        while i < v.size:
            j = i
            while j + 1 < v.size and s[j + 1] == s[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r

    rx, ry = _ranks(x), _ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0.0 else float("nan")


def sf_label_attachment_corr(
    x: np.ndarray, sf_wdl: np.ndarray, mask: np.ndarray | None = None,
) -> AxisReading:
    """Is a shard's SF label actually attached to the position it sits on?

    Rank correlation of the mover's material balance (read from ``x``) against
    the SF label's signal ``W - L`` (record POV, so material and label point the
    same way). It is a RULER-FREE integrity check: no model, no Stockfish, no
    arena, and it needs nothing but the shard itself.

    **Why it is kept even though it now rejects no shard the MultiPV axis
    misses.** On the live trial `ATT-only` is EMPTY, so the count-based
    criterion that demoted the bestmove-is-first-legal rate to a diagnostic
    would, applied mechanically, condemn this axis too. It is kept on two
    grounds that a reject count cannot express. (a) MECHANISM: it detects a
    genuinely different failure — the label block landing on the wrong ROWS,
    which leaves every per-row field internally consistent and can therefore
    coexist with a perfect MultiPV rate. Redundancy against the corruption modes
    that happen to be in this trial is not redundancy against the next one.
    (b) Its own separation is honest and threshold-independent: the lowest
    ACCEPTED shard reads +0.4189 (p10 +0.6092) while detached shards read
    ~0.00 (median +0.1477, max +0.2497), so the 0.25 line sits inside a 0.169
    gap that was not created by the line itself.

    It exists because on 2026-07-31 the SF label block silently detached from
    the rows it was written on — the labels stayed internally self-consistent
    and looked like real Stockfish output, so every check that read the label
    alone passed, while 45% of the WDL target had become noise. This one
    number went +0.63 -> -0.02 across the boundary. A value scorer that reads
    those shards reports a real-looking table about nothing, which is exactly
    the failure this repo keeps hitting: a value accepted and then silently
    ignored.
    """
    p = np.clip(np.asarray(sf_wdl, dtype=np.float64), 0.0, None)
    total = np.clip(p.sum(axis=-1), 1e-12, None)
    signal = (p[..., 0] - p[..., 2]) / total
    material = material_balance_from_planes(x)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        signal, material = signal[m], material[m]
    if signal.size < SF_AXIS_MIN_ROWS:
        return AxisReading(float("nan"), "too_few_rows")
    value = rank_corr(material, signal)
    # A degenerate shard (all material equal, or a constant label) yields NaN.
    # It is NOT evidence of soundness, so it carries a status of its own rather
    # than sharing "too_few_rows" or leaking through as a pass.
    return AxisReading(value) if np.isfinite(value) else AxisReading(value, "degenerate")


def sf_multipv_missing_rate(
    has_sf_multipv_raw: np.ndarray, labelled: np.ndarray | None = None,
) -> AxisReading:
    """Share of SF-LABELLED rows that carry no MultiPV candidate block.

    The sharp fingerprint of a Stockfish UCI desync. When a timed-out ``search``
    raises while the engine is still calculating, the stale ``bestmove`` is left
    in the pty buffer and nothing resynchronises; the recorded reply then belongs
    to a different position and its candidate list fails to survive the
    legality/parse path, so the row lands with an SF eval but no MultiPV block.
    (Cause and fix belong to PR #297; this is only the detector.)

    Preferred over the bestmove-is-first-legal rate because of the SHAPE of its
    sound distribution: **89.9% of accepted shards read exactly 0.000000**
    (median 0.000000, p99 0.004603, max 0.008032). Every labelled row is meant
    to carry a candidate list, so the sound rate has a hard floor at zero and
    any material excess is anomalous no matter where the cut sits. The
    bestmove-is-first-legal rate has no such floor — it sits at ~0.08 on sound
    shards for ordinary reasons — and its sound maximum and corrupt minimum are
    0.0009 apart, so no threshold on it can be honest.
    """
    has = np.asarray(has_sf_multipv_raw).astype(bool).reshape(-1)
    if has.size == 0:
        return AxisReading(float("nan"), "unusable_input")
    # Default to ALL rows. Defaulting to `has` would restrict the denominator to
    # rows that have a MultiPV block and force the rate to 0.0 by construction —
    # a detector that can only ever report "clean".
    sel = np.ones_like(has) if labelled is None else np.asarray(labelled, dtype=bool)
    if sel.shape != has.shape:
        return AxisReading(float("nan"), "unusable_input")
    n = int(sel.sum())
    if n < SF_AXIS_MIN_ROWS:
        return AxisReading(float("nan"), "too_few_rows")
    return AxisReading(float((~has[sel]).mean()))


def sf_bestmove_is_first_legal_rate(
    sf_move_index: np.ndarray, sf_legal_mask: np.ndarray, mask: np.ndarray | None = None,
) -> AxisReading:
    """Share of labelled rows whose SF bestmove is just the first legal move.

    The fingerprint of a Stockfish UCI DESYNC. When a timed-out ``search``
    raises while the engine is still calculating, the stale ``bestmove`` is left
    in the pty buffer and nothing resynchronises, so later searches return the
    PREVIOUS query's answer. The recorded bestmove then often fails to match any
    legal move at the current position and the builder's silent
    ``legal_indices[0]`` fallback fires, which is what this counts. (Cause and
    fix are PR #297's; this is only the detector.)

    Kept as a DIAGNOSTIC, not a default gate — see `SF_DESYNC_MAX`. It is a real
    signal (0.080 baseline, 0.16-0.97 inside an episode) but it admits no honest
    threshold, and `sf_multipv_missing_rate` detects the same failure against a
    sound distribution that is pinned at exactly zero.

    Returns a non-"ok" status when the inputs cannot support the check; callers
    must treat that as a REJECT rather than a pass — a gate that cannot evaluate
    a shard has not cleared it.
    """
    smi = np.asarray(sf_move_index).reshape(-1)
    legal = np.asarray(sf_legal_mask)
    if smi.size == 0 or legal.ndim != 2 or legal.shape[0] != smi.size:
        return AxisReading(float("nan"), "unusable_input")
    ok = smi >= 0
    ok &= legal.sum(axis=1) > 0
    if mask is not None:
        ok &= np.asarray(mask, dtype=bool)
    if int(ok.sum()) < SF_AXIS_MIN_ROWS:
        return AxisReading(float("nan"), "too_few_rows")
    first_legal = np.argmax(legal[ok] > 0, axis=1)
    return AxisReading(float((smi[ok] == first_legal).mean()))


def desync_reject_reason(
    *,
    attachment: AxisReading,
    multipv: AxisReading,
    attachment_min: float = SF_LABEL_ATTACHMENT_MIN,
    multipv_miss_max: float = SF_MULTIPV_MISS_MAX,
) -> str:
    """The shipped two-axis SF-desync verdict. Empty string == accept.

    ⚑ THIS IS THE ONE COPY. It used to be three: an inline branch in
    ``scripts/value_optimism.py``, a re-implementation in
    ``scripts/quarantine_desync_shards.py::judge``, and a test that pinned only
    the second against the axis functions here. ``quarantine_desync_shards``'s
    own docstring recorded the consequence: an independent review changed the
    scorer's ``multipv.value > multipv_miss_max`` to ``> 999.0``, disabling its
    multipv axis and flipping 118 of 834 live shards, and the whole suite still
    passed. Every new consumer must call THIS, never restate it — a guard that
    does not share the criterion's instrument is not a guard.

    An UNUSABLE reading on either axis is a REJECT, not a pass: a gate that
    could not evaluate a shard has not cleared it. ``usable`` rather than
    ``status == "ok"`` because a finite check has to happen somewhere, and the
    only two places it could live are here and in every caller.

    The bestmove-is-first-legal axis is deliberately absent. It is a
    DIAGNOSTIC (``SF_DESYNC_MAX`` defaults to 1.0 = never reject) because its
    sound maximum and corrupt minimum are 0.0009 apart; folding it in here
    would make it look enforced.
    """
    if not attachment.usable:
        return attachment.describe("attachment")
    if attachment.value < float(attachment_min):
        return f"attachment {attachment.value:+.4f} < {attachment_min}"
    if not multipv.usable:
        return multipv.describe("multipv-miss")
    if multipv.value > float(multipv_miss_max):
        return f"multipv-miss {multipv.value:.6f} > {multipv_miss_max}"
    return ""


@dataclass(frozen=True)
class OptimismRows:
    """Per-row inputs to the scorer, all aligned and already POV-consistent.

    Every array is from the SCORED SIDE's point of view (the side to move at
    the row's own position). ``sf_cp`` is the objective ruler and is the only
    thing allowed to define the stratum.
    """

    sf_cp: np.ndarray        # (N,) SF evaluation of the row's OWN position, cp
    sf_ruler_score: np.ndarray  # (N,) that same evaluation as an expected score
    net_score: np.ndarray    # (N,) expected score from the net's wdl head
    target_score: np.ndarray  # (N,) expected score of the blended training target
    outcome_score: np.ndarray  # (N,) realized game result as a score (1/0.5/0)
    target_sf_score: np.ndarray  # (N,) the SF component the blend actually uses
    search_score: np.ndarray     # (N,) expected score of the MCTS root WDL
    game_id: np.ndarray      # (N,) cluster id for the bootstrap
    piece_count: np.ndarray  # (N,) total pieces, for the tablebase-range split

    def __post_init__(self) -> None:
        n = int(self.sf_cp.shape[0])
        for name in (
            "sf_ruler_score", "net_score", "target_score", "outcome_score",
            "target_sf_score", "search_score", "game_id", "piece_count",
        ):
            arr = getattr(self, name)
            if int(np.asarray(arr).shape[0]) != n:
                raise ValueError(f"OptimismRows.{name} has {arr.shape[0]} rows, expected {n}")

    def select(self, mask: np.ndarray) -> OptimismRows:
        """Row subset (used for the >=8-man robustness split)."""
        m = np.asarray(mask, dtype=bool)
        return OptimismRows(
            sf_cp=self.sf_cp[m], sf_ruler_score=self.sf_ruler_score[m],
            net_score=self.net_score[m],
            target_score=self.target_score[m], outcome_score=self.outcome_score[m],
            target_sf_score=self.target_sf_score[m], search_score=self.search_score[m],
            game_id=self.game_id[m], piece_count=self.piece_count[m],
        )

    def with_shuffled_net(self, rng: np.random.Generator) -> OptimismRows:
        """NEGATIVE CONTROL: break the position <-> net-evaluation association.

        Permutes ONLY the net's evaluations across rows. Everything else — the
        ruler, the buckets, the target, the outcomes — is untouched, so any
        bucket structure that survives is structure the scorer invented rather
        than measured. A scorer that still reports a losing-bucket effect here
        cannot fail, and a scorer that cannot fail is not a scorer.
        """
        perm = rng.permutation(int(self.sf_cp.shape[0]))
        return OptimismRows(
            sf_cp=self.sf_cp, sf_ruler_score=self.sf_ruler_score,
            net_score=self.net_score[perm],
            target_score=self.target_score, outcome_score=self.outcome_score,
            target_sf_score=self.target_sf_score, search_score=self.search_score,
            game_id=self.game_id, piece_count=self.piece_count,
        )


@dataclass(frozen=True)
class BucketStat:
    """One SF-eval stratum. Differences are means of per-row differences."""

    name: str
    n: int
    n_games: int
    sf_cp_mean: float
    net_score: float
    target_score: float
    sf_ruler_score: float
    outcome_score: float
    target_sf_score: float
    search_score: float
    net_minus_sf: float
    net_minus_sf_ci: tuple[float, float]
    target_minus_sf: float
    target_minus_sf_ci: tuple[float, float]
    net_minus_target: float
    net_minus_target_ci: tuple[float, float]
    net_cp_mean: float
    net_minus_sf_cp: float
    net_minus_sf_cp_ci: tuple[float, float]
    target_minus_sf_cp: float
    optimistic_frac: float
    cp_clamped_frac: float
    tb_range_frac: float


def cluster_bootstrap_ci(
    values: np.ndarray, game_id: np.ndarray, *, n_boot: int, rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for a mean, resampling GAMES rather than rows.

    Rows inside one game are strongly correlated (consecutive plies of the same
    position sequence), so a row-level bootstrap reports a CI several times too
    tight. The cluster is the game.

    Public because ``scripts/value_loss_scorer.py`` scores a different sample
    with the same clustering: two copies of this would be two chances to get
    the cluster wrong, and a row-level bootstrap in one of them would report a
    CI several times too tight while looking identical in the output.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return (float("nan"), float("nan"))
    games, inv = np.unique(np.asarray(game_id), return_inverse=True)
    n_games = int(games.size)
    if n_games < 2 or n_boot <= 0:
        return (float("nan"), float("nan"))
    order = np.argsort(inv, kind="stable")
    v_sorted = v[order]
    starts = np.searchsorted(inv[order], np.arange(n_games), side="left")
    ends = np.searchsorted(inv[order], np.arange(n_games), side="right")
    sums = np.add.reduceat(v_sorted, starts) if v_sorted.size else np.zeros(n_games)
    counts = (ends - starts).astype(np.float64)
    draws = rng.integers(0, n_games, size=(n_boot, n_games))
    boot = sums[draws].sum(axis=1) / np.clip(counts[draws].sum(axis=1), 1e-12, None)
    return (
        float(np.quantile(boot, alpha / 2.0)),
        float(np.quantile(boot, 1.0 - alpha / 2.0)),
    )


def score_buckets(
    rows: OptimismRows, *, slope: float, draw_width_cp: float,
    n_boot: int = 2000, seed: int = 0, cp_clamp: float = CP_CLAMP,
    edges: tuple[float, ...] = SF_EVAL_BUCKET_EDGES,
) -> list[BucketStat]:
    """Stratified value-optimism table, one entry per SF-eval bucket.

    Buckets with no rows are omitted. The cp columns are per-row differences of
    per-row cp values (never a cp of a mean), matching how the original
    "+287 cp" claim was formed.
    """
    rng = np.random.default_rng(seed)
    names_all = bucket_names_for(edges)
    buckets = sf_eval_bucket_array(np.clip(rows.sf_cp, -cp_clamp, cp_clamp), edges)
    net_cp, net_clamped = expected_score_to_cp(
        rows.net_score, slope=slope, draw_width_cp=draw_width_cp, cp_clamp=cp_clamp,
    )
    target_cp, _ = expected_score_to_cp(
        rows.target_score, slope=slope, draw_width_cp=draw_width_cp, cp_clamp=cp_clamp,
    )
    sf_cp_clamped = np.clip(rows.sf_cp, -cp_clamp, cp_clamp)

    out: list[BucketStat] = []
    for b, name in enumerate(names_all):
        sel = buckets == b
        n = int(sel.sum())
        if n == 0:
            continue
        d_net_sf = rows.net_score[sel] - rows.sf_ruler_score[sel]
        d_tgt_sf = rows.target_score[sel] - rows.sf_ruler_score[sel]
        d_net_tgt = rows.net_score[sel] - rows.target_score[sel]
        d_net_sf_cp = net_cp[sel] - sf_cp_clamped[sel]
        gid = rows.game_id[sel]
        out.append(BucketStat(
            name=name,
            n=n,
            n_games=int(np.unique(gid).size),
            sf_cp_mean=float(sf_cp_clamped[sel].mean()),
            net_score=float(rows.net_score[sel].mean()),
            target_score=float(rows.target_score[sel].mean()),
            sf_ruler_score=float(rows.sf_ruler_score[sel].mean()),
            outcome_score=float(rows.outcome_score[sel].mean()),
            target_sf_score=float(rows.target_sf_score[sel].mean()),
            search_score=float(rows.search_score[sel].mean()),
            net_minus_sf=float(d_net_sf.mean()),
            net_minus_sf_ci=cluster_bootstrap_ci(d_net_sf, gid, n_boot=n_boot, rng=rng),
            target_minus_sf=float(d_tgt_sf.mean()),
            target_minus_sf_ci=cluster_bootstrap_ci(d_tgt_sf, gid, n_boot=n_boot, rng=rng),
            net_minus_target=float(d_net_tgt.mean()),
            net_minus_target_ci=cluster_bootstrap_ci(d_net_tgt, gid, n_boot=n_boot, rng=rng),
            net_cp_mean=float(net_cp[sel].mean()),
            net_minus_sf_cp=float(d_net_sf_cp.mean()),
            net_minus_sf_cp_ci=cluster_bootstrap_ci(d_net_sf_cp, gid, n_boot=n_boot, rng=rng),
            target_minus_sf_cp=float((target_cp[sel] - sf_cp_clamped[sel]).mean()),
            optimistic_frac=float((d_net_sf > 0.0).mean()),
            cp_clamped_frac=float(net_clamped[sel].mean()),
            tb_range_frac=float((rows.piece_count[sel] <= 7).mean()),
        ))
    return out


def bucket_net_score_spread(stats: list[BucketStat]) -> float:
    """Control statistic: max-min of the bucket mean net score.

    Under the shuffled-association control the net's evaluations are
    independent of the bucket, so every bucket mean estimates the same global
    mean and this collapses to sampling noise. It is the single number the
    negative control has to kill; it is deliberately independent of the SF
    ruler's own level, which the shuffle does NOT disturb.
    """
    if not stats:
        return 0.0
    scores = [s.net_score for s in stats]
    return float(max(scores) - min(scores))


def tail_asymmetry(
    stats: list[BucketStat], edges: tuple[float, ...] = SF_EVAL_BUCKET_EDGES,
) -> float | None:
    """``net_minus_sf`` in the FIRST bucket PLUS that in the LAST bucket.

    Any error the bucketing itself induces enters the two tails with opposite
    signs and cancels here, so this isolates the directional part. Returns None
    unless both tail buckets are populated.

    **Three ways to misread this number, all of which have happened.**

    (a) Its null is NOT zero. Under the shuffled-net negative control it
    measured -0.0051 on live rows — the statistic has an offset that comes from
    the ruler's own asymmetry about the net's mean, not from the head.
    (b) A perfect head's null is also not zero; get it from
    `perfect_head_tail_asymmetry`, which reads it off the realized outcomes.
    Judge `tail_asymmetry` as an EXCESS over that null, with a CI from
    `tail_asymmetry_ci` — never as a raw level.
    (c) The tails are whatever `edges[0]` / `edges[-1]` select. With fine edges
    those are the SATURATED extremes, where the cp axis is flat and the score
    axis compressed, and the number is NOT the same quantity as the +-300 mirror
    pair a reader may be looking at. Say which pair produced it.
    """
    names = bucket_names_for(edges)
    by_name = {s.name: s for s in stats}
    lo = by_name.get(names[0])
    hi = by_name.get(names[-1])
    if lo is None or hi is None:
        return None
    return float(lo.net_minus_sf + hi.net_minus_sf)


def perfect_head_tail_asymmetry(
    stats: list[BucketStat], edges: tuple[float, ...] = SF_EVAL_BUCKET_EDGES,
) -> float | None:
    """The `tail_asymmetry` an OUTCOME-PERFECT head would score. The null.

    Substitutes the realized game outcome for the net. Because the outcome is an
    unbiased draw of the position's true value under the actual continuation, a
    head that predicted it exactly would produce this number — so it is the
    reference `tail_asymmetry` must be judged against, and the direct empirical
    test of whether bucketing biases the ruler-relative comparison at all.
    """
    names = bucket_names_for(edges)
    by_name = {s.name: s for s in stats}
    lo = by_name.get(names[0])
    hi = by_name.get(names[-1])
    if lo is None or hi is None:
        return None
    return float(
        (lo.outcome_score - lo.sf_ruler_score) + (hi.outcome_score - hi.sf_ruler_score)
    )


def _game_tail_sums(
    values: np.ndarray, inv: np.ndarray, n_games: int, mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(n_games, dtype=np.float64)
    counts = np.zeros(n_games, dtype=np.float64)
    np.add.at(sums, inv[mask], values[mask])
    np.add.at(counts, inv[mask], 1.0)
    return sums, counts


def tail_asymmetry_ci(
    rows: OptimismRows, *, n_boot: int = 2000, seed: int = 0,
    cp_clamp: float = CP_CLAMP, edges: tuple[float, ...] = SF_EVAL_BUCKET_EDGES,
    alpha: float = 0.05,
) -> tuple[float, float] | None:
    """95% game-clustered bootstrap CI for `tail_asymmetry`.

    Both tails are recomputed inside every resample from the SAME draw of games,
    so their correlation is preserved — resampling them independently would
    understate the width of a difference of two means over overlapping games.
    """
    buckets = sf_eval_bucket_array(np.clip(rows.sf_cp, -cp_clamp, cp_clamp), edges)
    lo_mask = buckets == 0
    hi_mask = buckets == len(edges)
    if not lo_mask.any() or not hi_mask.any() or n_boot <= 0:
        return None
    diff = rows.net_score - rows.sf_ruler_score
    games, inv = np.unique(rows.game_id, return_inverse=True)
    n_games = int(games.size)
    if n_games < 2:
        return None
    lo_s, lo_c = _game_tail_sums(diff, inv, n_games, lo_mask)
    hi_s, hi_c = _game_tail_sums(diff, inv, n_games, hi_mask)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_games, size=(n_boot, n_games))
    boot = (
        lo_s[draws].sum(axis=1) / np.clip(lo_c[draws].sum(axis=1), 1e-12, None)
        + hi_s[draws].sum(axis=1) / np.clip(hi_c[draws].sum(axis=1), 1e-12, None)
    )
    return (
        float(np.quantile(boot, alpha / 2.0)),
        float(np.quantile(boot, 1.0 - alpha / 2.0)),
    )


@dataclass(frozen=True)
class OutcomeCalibration:
    """One bucket of the outcome-vs-ruler arm, for one population."""

    name: str
    n: int
    ruler_score: float
    outcome_score: float
    delta: float
    ci: tuple[float, float]


def outcome_calibration(
    *, ruler_cp: np.ndarray, outcome_score: np.ndarray, game_id: np.ndarray,
    slope: float, draw_width_cp: float, edges: tuple[float, ...] = SF_EVAL_BUCKET_EDGES,
    n_boot: int = 2000, seed: int = 0, cp_clamp: float = CP_CLAMP,
) -> list[OutcomeCalibration]:
    """Did games from an SF-evaluated position score better than the eval says?

    This is the arm that can see the PID handicap, and it exists because the
    head/target arm structurally cannot. The head/target arm needs two rows that
    are consecutive plies of one game, and curriculum games — the ONLY ones the
    handicapped Stockfish plays in — never produce such a pair, so that arm is
    100% selfplay by construction. This arm needs no pairing and no model: it
    uses the row's own SF label as the ruler, so it covers every labelled row in
    the window and can be split by ``is_selfplay``.

    ``ruler_cp`` must already be in the SCORED SIDE's point of view.
    """
    rng = np.random.default_rng(seed)
    cp = np.clip(np.asarray(ruler_cp, dtype=np.float64), -cp_clamp, cp_clamp)
    ruler = cp_to_expected_score(cp, slope=slope, draw_width_cp=draw_width_cp)
    out = np.asarray(outcome_score, dtype=np.float64)
    buckets = sf_eval_bucket_array(cp, edges)
    result: list[OutcomeCalibration] = []
    for b, name in enumerate(bucket_names_for(edges)):
        sel = buckets == b
        if not sel.any():
            continue
        delta = out[sel] - ruler[sel]
        result.append(OutcomeCalibration(
            name=name,
            n=int(sel.sum()),
            ruler_score=float(ruler[sel].mean()),
            outcome_score=float(out[sel].mean()),
            delta=float(delta.mean()),
            ci=cluster_bootstrap_ci(delta, game_id[sel], n_boot=n_boot, rng=rng),
        ))
    return result
