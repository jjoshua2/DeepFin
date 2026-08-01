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
# inverse is numerically meaningless beyond it and a single mate score would
# otherwise dominate a bucket mean. 1500 is _MATE_BASE_CP, the value
# mate_to_effective_cp assigns the LONGEST mates, so no real label sits inside
# the clamp and outside the representable band.
CP_CLAMP: float = 1500.0


# Piece values in plane order (pawn, knight, bishop, rook, queen, king), used
# only by the shard-integrity guard below.
_PIECE_VALUES: tuple[float, ...] = (1.0, 3.0, 3.0, 5.0, 9.0, 0.0)

# Minimum rank correlation between a shard's own material balance and its own
# SF label. Anything sane clears this by a wide margin (live shards measure
# +0.58..+0.71); a detached label block measures ~0.00. See
# `sf_label_attachment_corr`.
SF_LABEL_ATTACHMENT_MIN: float = 0.25


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
) -> float:
    """Is a shard's SF label actually attached to the position it sits on?

    Rank correlation of the mover's material balance (read from ``x``) against
    the SF label's signal ``W - L`` (record POV, so material and label point the
    same way). It is a RULER-FREE integrity check: no model, no Stockfish, no
    arena, and it needs nothing but the shard itself.

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
    return rank_corr(material, signal)


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


def _cluster_bootstrap_ci(
    values: np.ndarray, game_id: np.ndarray, *, n_boot: int, rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI for a mean, resampling GAMES rather than rows.

    Rows inside one game are strongly correlated (consecutive plies of the same
    position sequence), so a row-level bootstrap reports a CI several times too
    tight. The cluster is the game.
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
            net_minus_sf_ci=_cluster_bootstrap_ci(d_net_sf, gid, n_boot=n_boot, rng=rng),
            target_minus_sf=float(d_tgt_sf.mean()),
            target_minus_sf_ci=_cluster_bootstrap_ci(d_tgt_sf, gid, n_boot=n_boot, rng=rng),
            net_minus_target=float(d_net_tgt.mean()),
            net_minus_target_ci=_cluster_bootstrap_ci(d_net_tgt, gid, n_boot=n_boot, rng=rng),
            net_cp_mean=float(net_cp[sel].mean()),
            net_minus_sf_cp=float(d_net_sf_cp.mean()),
            net_minus_sf_cp_ci=_cluster_bootstrap_ci(d_net_sf_cp, gid, n_boot=n_boot, rng=rng),
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
            ci=_cluster_bootstrap_ci(delta, game_id[sel], n_boot=n_boot, rng=rng),
        ))
    return result
