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
   everywhere else. A single-bucket error rate is uninterpretable, and worse:
   bucketing on a NOISY ruler regresses every extreme bucket toward the mean,
   which manufactures apparent optimism when losing and apparent pessimism
   when winning *even for a perfect head*. That artifact is SYMMETRIC, so only
   the ASYMMETRY between the two tails (``tail_asymmetry`` below) is evidence
   of a directional defect.
3. **Score the TARGET next to the HEAD.** A head that matches an optimistic
   target is a data defect; a head that misses a sound target is a fitting
   defect. Reporting ``net``, ``target`` and its three components in the same
   bucket separates them.

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
    """``net_minus_sf`` in the lost bucket PLUS that in the won bucket.

    Regression toward the mean from a noisy ruler pushes the lost bucket
    optimistic and the won bucket pessimistic by the SAME amount, so it cancels
    here. What survives is the directional part of the effect. Returns None
    unless BOTH tail buckets are populated — a one-sided read of a symmetric
    artifact is exactly the mistake this number exists to prevent.
    """
    names = bucket_names_for(edges)
    by_name = {s.name: s for s in stats}
    lo = by_name.get(names[0])
    hi = by_name.get(names[-1])
    if lo is None or hi is None:
        return None
    return float(lo.net_minus_sf + hi.net_minus_sf)
