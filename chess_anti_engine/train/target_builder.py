"""Rebuild SF-derived training targets from sparse MultiPV labels.

Shards (schema v2+) store the raw MultiPV candidate rows the live pipeline
built its targets from (``sf_multipv_raw``/``sf_label_meta``; layout in
replay/shard.py). These pure functions replay the exact live construction
(selfplay/stockfish_turn.py) with arbitrary parameters, so target questions —
``sf_policy_temp``, ``sf_policy_label_smooth``, cp→logistic ``slope`` /
``draw_width``, logistic-vs-native WDL — can be re-answered against data that
already exists, instead of waiting ~18 h for a 1.5M-row replay window to turn
over. `train.rebuild_sf_targets` (default OFF) applies the rebuild to every
sampled batch of LIVE training; scripts/retarget_retrain.py drives the offline
variant sweep.

With params equal to the capture-time config, the rebuilt targets match the
stored ones to float precision (parity-tested in
tests/test_sparse_multipv_labels.py; measured TV 7.3e-5 mean over 16.8k live
shard rows on 2026-07-27).

WHAT THIS CANNOT REBUILD, and what happens instead, is in
docs/target_rebuildability.md — read it before assuming a target moved.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chess_anti_engine.replay.shard import SF_CP_SENTINEL
from chess_anti_engine.stockfish.wdl import (
    cp_to_wdl,
    cp_to_wdl_array,
    mate_to_effective_cp_array,
)
from chess_anti_engine.train.targets import (
    DEFAULT_CATEGORICAL_BINS,
    categorical_target_value,
    hlgauss_target,
)


@dataclass(frozen=True)
class SfTargetParams:
    """Knobs of the SF target construction (defaults match compute paths'
    function defaults; pass live config values for parity)."""

    sf_policy_temp: float = 0.25
    sf_policy_label_smooth: float = 0.05
    sf_wdl_use_cp_logistic: bool = False
    sf_wdl_cp_slope: float = 0.010
    sf_wdl_cp_draw_width: float = 60.0


@dataclass(frozen=True)
class CategoricalTargetParams:
    """Knobs of the categorical (HL-Gauss) value target (mirror the selfplay
    ``categorical_*`` / ``hlgauss_sigma`` config; defaults match GameConfig)."""

    blend_frac: float = 0.0
    search_blend_frac: float = 0.0
    num_bins: int = DEFAULT_CATEGORICAL_BINS
    sigma: float = 0.04


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def _row_score(
    cp: int, mate: int, wdl_w: int, wdl_d: int, params: SfTargetParams,
) -> float | None:
    """w + 0.5*d for one MultiPV row — mirrors stockfish_turn._pv_wdl_score.

    Logistic path: cp/mate → normalized (w, d, l). Native path: the stored
    permille wdl rescaled to fractions — the live path scores fractions
    (_parse_wdl normalizes), and sf_policy_temp's softmax is scale-sensitive,
    so reproducing the live fraction scale exactly is the point.
    """
    has_cp = cp != SF_CP_SENTINEL
    has_mate = mate != 0
    if params.sf_wdl_use_cp_logistic and (has_cp or has_mate):
        wdl = cp_to_wdl(
            cp if has_cp else None,
            mate if has_mate else None,
            slope=params.sf_wdl_cp_slope,
            draw_width_cp=params.sf_wdl_cp_draw_width,
        )
        return float(wdl[0]) + 0.5 * float(wdl[1])
    if wdl_w < 0 or wdl_d < 0:
        return None
    return (float(wdl_w) + 0.5 * float(wdl_d)) / 1000.0


def rebuild_sf_policy_target(
    multipv_raw: np.ndarray,
    *,
    legal_indices: np.ndarray,
    policy_size: int,
    params: SfTargetParams,
) -> np.ndarray | None:
    """(K, 5) raw rows → dense SF policy target in the rows' move space.

    Mirrors stockfish_turn._build_sf_policy_target: softmax over candidate
    scores / temp, scatter-add, optional legal-set smoothing, renormalize.
    Returns None when no scoreable rows remain (caller keeps the stored
    target — matches the live fallback which never produced sparse rows in
    that case).
    """
    rows = np.asarray(multipv_raw)
    rows = rows[rows[:, 0] >= 0]
    idxs: list[int] = []
    scores: list[float] = []
    for move_idx, cp, mate, w, d in rows.tolist():
        score = _row_score(int(cp), int(mate), int(w), int(d), params)
        if score is None:
            continue
        idxs.append(int(move_idx))
        scores.append(score)
    if not idxs:
        return None

    p_top = _softmax(
        np.array(scores, dtype=np.float64) / max(1e-6, float(params.sf_policy_temp))
    ).astype(np.float32, copy=False)
    p_sf = np.zeros((int(policy_size),), dtype=np.float32)
    for a, p in zip(idxs, p_top, strict=True):
        p_sf[a] += float(p)

    smooth = float(params.sf_policy_label_smooth)
    legal = np.asarray(legal_indices, dtype=np.int64)
    # Smooth only when candidates don't cover every legal move — mirrors the
    # live builder (stockfish_turn._build_sf_policy_target) so rebuilt targets
    # match the dense ones recorded by selfplay.
    n_covered = int(np.isin(legal, idxs).sum())  # legal moves with a scored PV
    has_uncovered = n_covered < int(legal.size)
    if smooth > 0.0 and legal.size > 0 and has_uncovered:
        p_sf *= 1.0 - smooth
        p_sf[legal] += smooth / float(legal.size)

    total = float(p_sf.sum())
    if total > 0:
        p_sf /= total
    return p_sf


def rebuild_sf_wdl(label_meta: np.ndarray, params: SfTargetParams) -> np.ndarray | None:
    """(6,) label meta → record-POV (W, D, L) — mirrors
    stockfish_turn._sf_result_wdl_for_record including the POV flip."""
    meta = np.asarray(label_meta).reshape(-1)
    cp, mate = int(meta[2]), int(meta[3])
    wdl_w, wdl_d = int(meta[4]), int(meta[5])
    has_cp = cp != SF_CP_SENTINEL
    has_mate = mate != 0
    if params.sf_wdl_use_cp_logistic and (has_cp or has_mate):
        wdl_stm = np.asarray(cp_to_wdl(
            cp if has_cp else None,
            mate if has_mate else None,
            slope=params.sf_wdl_cp_slope,
            draw_width_cp=params.sf_wdl_cp_draw_width,
        ), dtype=np.float32)
        return wdl_stm[::-1].copy()  # flip_wdl_pov
    if wdl_w < 0 or wdl_d < 0:
        return None
    # Fractions, matching the live rec.sf_wdl scale (consumers like
    # finalize's sf_search_gap read the vector unrenormalized).
    wdl_stm = np.array(
        [wdl_w / 1000.0, wdl_d / 1000.0, (1000 - wdl_w - wdl_d) / 1000.0],
        dtype=np.float32,
    )
    return wdl_stm[::-1].copy()


def _batch_row_scores(
    raw: np.ndarray, params: SfTargetParams,
) -> tuple[np.ndarray, np.ndarray]:
    """(B, K, 5) raw rows → (scores float64 (B, K), scoreable bool (B, K)).

    Batched `_row_score` + the `move_idx >= 0` padding filter, in one pass.
    ``scores`` is meaningless where ``scoreable`` is False.
    """
    move = raw[..., 0]
    cp = raw[..., 1].astype(np.int64, copy=False)
    mate = raw[..., 2].astype(np.int64, copy=False)
    wdl_w = raw[..., 3]
    wdl_d = raw[..., 4]

    valid = move >= 0
    native_ok = (wdl_w >= 0) & (wdl_d >= 0)
    scores = (wdl_w.astype(np.float64) + 0.5 * wdl_d.astype(np.float64)) / 1000.0

    if not params.sf_wdl_use_cp_logistic:
        return scores, np.asarray(valid & native_ok)

    has_cp = cp != SF_CP_SENTINEL
    has_mate = mate != 0
    use_log = has_cp | has_mate
    # Evaluate the logistic only on the entries that take it (padding rows are
    # ~half the (B, K) grid at production multipv), so the two exp() calls —
    # the dominant cost here — run over the compressed set.
    sel = valid & use_log
    if bool(sel.any()):
        m_sel = mate[sel]
        # cp_to_wdl gives mate precedence over cp; mirror that with a select.
        eff_cp = np.where(
            m_sel != 0, mate_to_effective_cp_array(m_sel), cp[sel].astype(np.float64),
        )
        logistic = cp_to_wdl_array(
            eff_cp,
            slope=params.sf_wdl_cp_slope,
            draw_width_cp=params.sf_wdl_cp_draw_width,
        )
        scores[sel] = (
            logistic[:, 0].astype(np.float64) + 0.5 * logistic[:, 1].astype(np.float64)
        )
    return scores, np.asarray(valid & (use_log | native_ok))


def rebuild_sf_policy_targets_batch(
    multipv_raw: np.ndarray,
    *,
    legal_dense: np.ndarray | None,
    policy_size: int,
    params: SfTargetParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized `rebuild_sf_policy_target` over a whole batch.

    ``multipv_raw`` is (B, K, 5); ``legal_dense`` the (B, policy_size)
    ``sf_legal_mask`` (None ⇒ no smoothing, matching the scalar function's
    empty-``legal_indices`` path). Returns ``(targets (B, policy_size)
    float32, ok (B,) bool)``; ``ok`` is False exactly where the scalar
    function returns None (no scoreable rows) and those target rows are zero
    — the caller keeps the stored target there.

    Bitwise-equal to the scalar path, including on repeated move indices
    (tests/test_sparse_multipv_labels.py::test_batch_rebuild_*). The arithmetic
    order is kept identical on purpose: softmax in float64, cast to float32,
    then every subsequent step in float32, with the label-smoothing scale
    applied AFTER the scatter because the scatter accumulates.
    """
    raw = np.asarray(multipv_raw)
    n = int(raw.shape[0])
    width = int(policy_size)
    out = np.zeros((n, width), dtype=np.float32)
    if n == 0:
        return out, np.zeros((0,), dtype=bool)

    scores, scoreable = _batch_row_scores(raw, params)
    ok = np.asarray(scoreable.any(axis=1))
    if not bool(ok.any()):
        return out, ok

    # Masked row-wise softmax over the (B, K) grid. Non-scoreable slots carry
    # z = -inf, and zmax is forced finite even on all-masked rows, so
    # exp(z - zmax) = exp(-inf) is an EXACT +0.0 there — additively neutral in
    # the row sum, the same value the scalar path gets by compacting first.
    # That -inf → +0.0 identity is what keeps masked slots out of the result;
    # no re-zeroing after the exp is needed (scores are bounded, so no
    # scoreable slot can produce a nan/inf that would escape the mask).
    z = scores / max(1e-6, float(params.sf_policy_temp))
    z = np.where(scoreable, z, -np.inf)
    zmax = np.max(z, axis=1, keepdims=True)
    zmax = np.where(np.isfinite(zmax), zmax, 0.0)  # all-masked rows: keep exp finite
    e = np.exp(z - zmax)
    row_sum = e.sum(axis=1, keepdims=True)
    p_top = (e / np.where(row_sum > 0.0, row_sum, 1.0)).astype(np.float32, copy=False)

    # Candidate (row, col) pairs. Scattered on the 2-D array, NOT on a
    # flattened `rows * width + cols`: a move index >= policy_size would make
    # the flat form silently write into the NEXT row's target, whereas the 2-D
    # index raises IndexError exactly like the scalar path's `p_sf[a] += p`.
    # No live row does this today, but a policy-encoding mismatch (legacy
    # 4672-space raw against an 1858-wide target, or a future width change)
    # would otherwise turn a crash into wrong training targets that nothing
    # reports. Pinned by test_batch_rebuild_raises_on_out_of_range_move_index.
    keep = scoreable.reshape(-1)
    rows = np.repeat(np.arange(n, dtype=np.int64), raw.shape[1])[keep]
    cols = raw[..., 0].astype(np.int64, copy=False).reshape(-1)[keep]
    # Distinct (row, col) pairs — needed twice below, for the scatter's
    # duplicate guard and for the smoothing coverage count. Flattening is safe
    # HERE because `flat` is only compared and decomposed, never used as a
    # write index: an out-of-range move index can alias another row's slot in
    # it, but the only consequence is taking the `np.add.at` branch, which
    # raises the same IndexError the 2-D writes do.
    flat = rows * np.int64(width) + cols
    uniq = np.unique(flat)

    smooth = float(params.sf_policy_label_smooth)
    legal = None
    apply = np.zeros((n,), dtype=bool)
    share = np.zeros((n,), dtype=np.float32)

    vals = p_top.reshape(-1)[keep]
    if uniq.size == flat.size:
        # No repeated (row, col) pair: a plain scatter-assign into the
        # all-zero output IS the scatter-add (nothing accumulates), ~5x
        # cheaper than `np.add.at`'s assume-nothing ufunc loop.
        out[rows, cols] = vals
    else:
        # Repeated move indices must ACCUMULATE to match the scalar path's
        # `p_sf[a] += p` (test_batch_rebuild_matches_scalar_with_duplicate_
        # move_indices). Real MultiPV rows never repeat a move, but the
        # storage format does not forbid it.
        np.add.at(out, (rows, cols), vals)

    if smooth > 0.0 and legal_dense is not None:
        legal = np.asarray(legal_dense) != 0
        legal_n = legal.sum(axis=1)
        # Per-row count of DISTINCT legal moves holding a scored PV, from the
        # unique candidate pairs. Replaces a zeros/scatter/and/sum over the
        # full (B, policy_size) grid (~1.5 ms/batch at production shape) with
        # work proportional to the candidate count; exact, because both forms
        # count the same set of distinct (row, col) pairs. Any out-of-range
        # column already raised in the scatter above, so the `legal` gather
        # cannot go out of bounds.
        rows_u = uniq // width
        cols_u = uniq % width
        covered_n = np.bincount(rows_u[legal[rows_u, cols_u]], minlength=n)
        apply = ok & (legal_n > 0) & (covered_n < legal_n)
        # float64 divide then one cast to float32, mirroring the scalar path's
        # `p_sf[legal] += smooth / float(legal.size)` (NumPy's weak-scalar
        # rule rounds the float64 scalar exactly once, at the add).
        share64 = np.zeros((n,), dtype=np.float64)
        np.divide(smooth, legal_n.astype(np.float64), out=share64, where=apply)
        share = share64.astype(np.float32)

    if legal is not None and bool(apply.any()):
        # Only ~25% of rows smooth (SF's 40 PVs cover every legal move in the
        # rest), so gather those rows once and do both smoothing steps on the
        # gather rather than paying two full-width passes.
        #
        # The `1 - smooth` scale runs AFTER the scatter, as in the scalar path.
        # Folding it into the (B, K) candidate vector beforehand looks free but
        # is not order-neutral: `np.add.at` is an ACCUMULATION, so a repeated
        # move index would compute p1*s + p2*s where the scalar path computes
        # (p1 + p2)*s. Real MultiPV rows never repeat a move, but "never in
        # practice" is not the same as bitwise-equal, and doing it here costs
        # nothing extra — the gather/scatter of these rows was already needed
        # for the smoothing add.
        #
        # legal is 0/1, so the multiply reproduces the scalar path's "add
        # `share` at legal indices, leave the rest" exactly (1 * s == s,
        # 0 * s == +0.0, and every entry of `out` is non-negative).
        sel = np.flatnonzero(apply)
        block = out[sel]
        block *= np.float32(1.0 - smooth)
        block += legal[sel] * share[sel, None]
        out[sel] = block

    # Divide by 1.0 instead of masking off empty rows: exact, and it keeps the
    # divide on numpy's fast unmasked loop.
    total = out.sum(axis=1)
    out /= np.where(total > 0.0, total, np.float32(1.0))[:, None]
    return out, ok


def rebuild_sf_wdl_batch(
    label_meta: np.ndarray, params: SfTargetParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized `rebuild_sf_wdl`. (B, 6) meta → ((B, 3) float32, ok (B,))."""
    meta = np.asarray(label_meta)
    n = int(meta.shape[0])
    out = np.zeros((n, 3), dtype=np.float32)
    if n == 0:
        return out, np.zeros((0,), dtype=bool)
    cp = meta[:, 2].astype(np.int64, copy=False)
    mate = meta[:, 3].astype(np.int64, copy=False)
    wdl_w = meta[:, 4].astype(np.int64, copy=False)
    wdl_d = meta[:, 5].astype(np.int64, copy=False)

    use_log = (
        (cp != SF_CP_SENTINEL) | (mate != 0)
        if params.sf_wdl_use_cp_logistic
        else np.zeros((n,), dtype=bool)
    )
    native_ok = (wdl_w >= 0) & (wdl_d >= 0)

    if bool(use_log.any()):
        # Evaluate the logistic on the use_log rows ONLY, matching the policy
        # twin's convention (_batch_row_scores). Running it full-width fed
        # exp() the rows whose cp is the -32768 sentinel — mathematically
        # discarded by the select, but overflowing float64 for
        # sf_wdl_cp_slope >= ~0.022, i.e. exactly on the knob this rebuild
        # exists to sweep. All the ops are elementwise, so compressing first
        # is bit-identical (test_batch_sf_wdl_logistic_has_no_sentinel_overflow).
        m_sel = mate[use_log]
        eff_cp = np.where(
            m_sel != 0, mate_to_effective_cp_array(m_sel), cp[use_log].astype(np.float64),
        )
        logistic = cp_to_wdl_array(
            eff_cp,
            slope=params.sf_wdl_cp_slope,
            draw_width_cp=params.sf_wdl_cp_draw_width,
        )
        out[use_log] = logistic[:, ::-1]  # flip_wdl_pov

    native_rows = native_ok & ~use_log
    if bool(native_rows.any()):
        # float64 divides then one cast, as in the scalar path's
        # `np.array([w/1000.0, ...], dtype=np.float32)`.
        stm = np.stack(
            [
                wdl_w / 1000.0,
                wdl_d / 1000.0,
                (1000 - wdl_w - wdl_d) / 1000.0,
            ],
            axis=1,
        ).astype(np.float32)
        out[native_rows] = stm[native_rows][:, ::-1]

    return out, np.asarray(use_log | native_ok)


@dataclass(frozen=True)
class SfRebuildCoverage:
    """What one call to `rebuild_sf_targets_in_arrays` actually touched.

    Returned rather than discarded because a rebuild whose coverage cannot be
    observed is unfalsifiable: the transition log proves the config PUSH, not
    the effect, and `has_sf_p0_frac -> 0` only proves it on a window that has
    p0 rows at all. `sf_rebuild_policy_frac` is the number that shows the flip
    took effect, and — see `metric_kwargs` — the number that detects poisoned
    SF labels in the window it ran on.
    """

    rows: int = 0
    policy_rebuilt: int = 0
    wdl_rebuilt: int = 0
    cross_ply_masked: int = 0   # ROWS that lost >=1 cross-ply target, not flags
  # Per-flag PRE-mask presence counts (rows whose flag was set before the
  # mask cleared it). The mask zeroes `has_sf_p0`/`has_sf_volatility`
  # indistinguishably from "never recorded", which pins `has_sf_p0_frac` —
  # the sf_p0 outage detector (trainable_report.py) — at 0.0 for the whole
  # rebuild experiment. These carry the pre-mask signal so the detector
  # keeps working while the flag is on.
    p0_masked: int = 0
    volatility_masked: int = 0

    def __add__(self, other: SfRebuildCoverage) -> SfRebuildCoverage:
        return SfRebuildCoverage(
            rows=self.rows + other.rows,
            policy_rebuilt=self.policy_rebuilt + other.policy_rebuilt,
            wdl_rebuilt=self.wdl_rebuilt + other.wdl_rebuilt,
            cross_ply_masked=self.cross_ply_masked + other.cross_ply_masked,
            p0_masked=self.p0_masked + other.p0_masked,
            volatility_masked=self.volatility_masked + other.volatility_masked,
        )

    def metric_kwargs(self) -> dict[str, float]:
        """The reported columns. All zero (not absent) when nothing ran, so a
        rebuild that silently stopped happening reads as 0.0 rather than as a
        missing column that a consumer would skip.

        DENOMINATOR: ``rows`` counts only batches that actually went through
        ``rebuild_sf_targets_in_arrays`` — the accumulator receives coverage
        from nowhere else — so every ``_frac`` here is a fraction of the
        REBUILT batches, not of all batches trained on. With the flag on for
        a whole iteration the two are the same thing; on the iteration where
        a live flip lands mid-way the fracs are over the rebuilt subset only,
        a one-row transient (docs/target_rebuildability.md, "Before flipping
        the flag live").

        ⚑ ``sf_rebuild_policy_frac`` BELOW ``sf_rebuild_wdl_frac`` IS A DESYNC
        ALARM, NOT A COVERAGE COST. The two share a denominator, and the
        selfplay writer stamps ``sf_multipv_raw`` and ``sf_label_meta`` on a
        labelled row TOGETHER (`selfplay/stockfish_turn.py::
        _stamp_sparse_sf_labels`). The only way a row gets meta but no raw is
        ``_collect_sparse_pv_rows`` returning None — not ONE of Stockfish's
        MultiPV moves was legal at the position queried, which is the
        fingerprint of a desynced UCI engine answering a DIFFERENT position
        (`_SF_NO_LEGAL_PV_WARN_RATE`, and `eval/value_optimism.py::
        sf_multipv_missing_rate`, the offline twin of the same measurement).
        So ``wdl_frac - policy_frac`` IS the poisoned-label share of the batch.
        Measured over 6,535 shards / 11.05M labelled rows: exactly 0.000000 on
        every clean stretch, 0.192 over the 122 shards quarantined 2026-08-01.
        This column was previously documented as reporting a ~5.4% structural
        gap; that figure came from a 10-shard sample drawn inside a 2026-07-27
        desync episode. There is no structural floor — it is zero.

        ``sf_rebuild_masked_p0_frac`` / ``_volatility_frac`` decompose
        ``sf_rebuild_masked_frac`` per flag, and are PRE-mask presence
        fractions: while a rebuild experiment runs they are the replacement
        for ``has_sf_p0_frac``, which the mask pins at 0.0. A masked_p0 frac
        of 0.0 with the flag on means the selfplay workers stopped recording
        sf_p0 rows — the outage the original column existed to catch.
        """
        denom = float(max(1, self.rows))
        return {
            "sf_rebuild_policy_frac": float(self.policy_rebuilt) / denom,
            "sf_rebuild_wdl_frac": float(self.wdl_rebuilt) / denom,
            "sf_rebuild_masked_frac": float(self.cross_ply_masked) / denom,
            "sf_rebuild_masked_p0_frac": float(self.p0_masked) / denom,
            "sf_rebuild_masked_volatility_frac": float(self.volatility_masked) / denom,
        }


def rebuild_sf_targets_in_arrays(
    arrs: dict[str, np.ndarray], *, params: SfTargetParams,
) -> tuple[dict[str, np.ndarray], SfRebuildCoverage]:
    """Recompute ``sf_policy_target`` / ``sf_wdl`` in a sampled batch dict.

    Only rows carrying sparse labels are rebuilt; rows without them keep their
    stored targets. Returns ``(arrs, coverage)`` — ``arrs`` mutated in place on
    fresh copies of the touched fields.

    Coverage over SF-LABELLED rows is TOTAL on healthy data — every labelled
    row is written with ``sf_multipv_raw`` — so a params change is a clean swap
    over the labelled window, not a mixture of two target regimes. The caller
    still reports coverage, because a shortfall is the alarm: see
    ``SfRebuildCoverage.metric_kwargs``. (Coverage over ALL rows is ~97 %, the
    SF-labelled fraction; the un-labelled rows have no SF target to rebuild.)

    Fully vectorized: ~13 ms per 512-row batch at policy width 1858 on the
    host prefetch thread, against a ~90 ms/step training budget (was ~275 ms
    as a per-row loop, i.e. 3x the whole step). That is what makes
    ``rebuild_sf_targets`` usable live and not only in the offline retarget
    driver.

    Two stored targets are DERIVED from what this rebuilds but live on a
    DIFFERENT shard row, so a sampled batch cannot rebuild them —
    ``sf_p0_policy_target`` (ply t-1's ``sf_policy_target``) and
    ``sf_volatility_target`` (|sf_wdl[t+6] - sf_wdl[t]|). Their presence flags
    are cleared here rather than left pointing at capture-time values; see
    ``docs/target_rebuildability.md`` for the full table and the storage that
    would make them rebuildable. Masking is UNCONDITIONAL — not "only when the
    params actually moved" — so that a control run (flag on, capture-identical
    params) and a treatment run mask exactly the same rows and the comparison
    stays paired.
    """
    n_rows = 0
    for probe in (
        "has_sf_multipv_raw", "sf_policy_target", "has_sf_label_meta", "sf_wdl",
        # Fall back to the cross-ply flags: a batch carrying ONLY those still
        # gets rows counted by the mask below, and rows=0 would turn
        # sf_rebuild_masked_frac from a fraction into a raw count (> 1.0).
        # Unreachable from the live schema today (every producer ships the
        # four keys above); the guard is cheaper than that failure mode.
        *CROSS_PLY_SF_FLAGS,
    ):
        cand = arrs.get(probe)
        if cand is not None and np.asarray(cand).ndim >= 1:
            n_rows = int(np.asarray(cand).shape[0])
            break
    n_policy = 0
    n_wdl = 0

    has_raw = np.asarray(arrs.get("has_sf_multipv_raw", ()), dtype=bool)
    if has_raw.size and has_raw.any() and "sf_multipv_raw" in arrs and "sf_policy_target" in arrs:
        pol = np.array(arrs["sf_policy_target"], copy=True)
        policy_size = int(pol.shape[1])
        legal_dense = arrs.get("sf_legal_mask")
        legal_rows = None if legal_dense is None else np.asarray(legal_dense)[has_raw]
        rebuilt, ok = rebuild_sf_policy_targets_batch(
            np.asarray(arrs["sf_multipv_raw"])[has_raw],
            legal_dense=legal_rows,
            policy_size=policy_size,
            params=params,
        )
        rows_idx = np.flatnonzero(has_raw)
        # No astype before the write: fancy-index assignment casts f32 to the
        # stored dtype (fp16 in shards) element-by-element with the same
        # rounding astype uses, so a pre-cast only materialises an extra
        # (B, width) temporary (~0.9 ms + 1.9 MB/batch measured) for a
        # bit-identical result (test_rebuild_in_arrays_writeback_matches_astype).
        if bool(ok.all()):
            # Common case: every labelled row rebuilt. Skip the `[ok]` gather,
            # which at policy width 1858 is a full copy of the output.
            pol[rows_idx] = rebuilt
        else:
            pol[rows_idx[ok]] = rebuilt[ok]
        arrs["sf_policy_target"] = pol
        n_policy = int(np.count_nonzero(ok))

    has_meta = np.asarray(arrs.get("has_sf_label_meta", ()), dtype=bool)
    if has_meta.size and has_meta.any() and "sf_label_meta" in arrs and "sf_wdl" in arrs:
        wdl = np.array(arrs["sf_wdl"], copy=True)
        rebuilt_wdl, ok_wdl = rebuild_sf_wdl_batch(
            np.asarray(arrs["sf_label_meta"])[has_meta], params,
        )
        write = np.flatnonzero(has_meta)[ok_wdl]
        wdl[write] = rebuilt_wdl[ok_wdl]  # assignment casts; see the policy write
        arrs["sf_wdl"] = wdl
        n_wdl = int(write.size)

    masked = mask_cross_ply_sf_targets(arrs)
    return arrs, SfRebuildCoverage(
        rows=n_rows,
        policy_rebuilt=n_policy,
        wdl_rebuilt=n_wdl,
        cross_ply_masked=masked.rows,
        p0_masked=masked.p0,
        volatility_masked=masked.volatility,
    )


# Presence flags of targets that are a function of `sf_policy_target` /
# `sf_wdl` on a DIFFERENT row than the one that carries them, so no in-batch
# rebuild can move them with their source.
CROSS_PLY_SF_FLAGS: tuple[str, ...] = ("has_sf_p0", "has_sf_volatility")


@dataclass(frozen=True)
class CrossPlyMaskCounts:
    """PRE-mask row counts from one `mask_cross_ply_sf_targets` call.

    ``rows`` is the number of ROWS that lost at least one cross-ply target —
    not the number of flags cleared, which would exceed the row count
    whenever a row carried both and make a column named ``_frac`` report
    > 1.0. ``p0`` / ``volatility`` are the per-flag presence counts BEFORE
    the mask cleared them; after it, a cleared flag is indistinguishable from
    one the worker never recorded, so these counts are the only place the
    outage signal survives (see ``SfRebuildCoverage.metric_kwargs``).
    """

    rows: int = 0
    p0: int = 0
    volatility: int = 0


def mask_cross_ply_sf_targets(arrs: dict[str, np.ndarray]) -> CrossPlyMaskCounts:
    """Zero the presence flags of the cross-ply SF targets.

    ``sf_p0_policy_target[t]`` IS ``sf_policy_target[t-1]`` (verified exactly
    on live shards) and ``sf_volatility_target[t]`` IS
    ``abs(sf_wdl[t+6] - sf_wdl[t])``; neither source row is in a randomly sampled
    batch. Training the sf_p0 own-move teacher on capture-time targets while
    ``sf_policy_target`` moves underneath it is the failure this prevents.

    ``sf_p0_regret`` is deliberately NOT masked: it is a normalized cp-regret
    over the same raw rows and carries no `SfTargetParams` dependence at all,
    so it stays valid under any rebuild.
    """
    touched: np.ndarray | None = None
    per_flag: dict[str, int] = {}
    for flag in CROSS_PLY_SF_FLAGS:
        cur = arrs.get(flag)
        if cur is None:
            continue
        arr = np.asarray(cur)
        if arr.size == 0:
            continue
        nonzero = arr != 0
        per_flag[flag] = int(np.count_nonzero(nonzero))
        touched = nonzero if touched is None else (touched | nonzero)
        arrs[flag] = np.zeros_like(arr)
    return CrossPlyMaskCounts(
        rows=0 if touched is None else int(np.count_nonzero(touched)),
        p0=per_flag.get("has_sf_p0", 0),
        volatility=per_flag.get("has_sf_volatility", 0),
    )


def rebuild_categorical_target_in_arrays(
    arrs: dict[str, np.ndarray], *, params: CategoricalTargetParams,
) -> dict[str, np.ndarray]:
    """Recompute ``categorical_target`` in a sampled batch from the stored hard
    outcome (``wdl_target``) blended with SF's eval (``sf_wdl``).

    Mirrors selfplay/finalize (``categorical_target_value`` + ``hlgauss_target``)
    so the offline sidecar can screen ``categorical_blend_frac`` on stored shards
    — the live finalize path bakes this target at capture time, so without a
    rebuild an offline replay of old shards would always measure the control.
    Rows without an SF eval (``has_sf_wdl == 0`` / missing) fall back to the
    ternary outcome. No-op when ``blend_frac <= 0`` (the stored target already
    equals the ternary HL-Gauss) or when the required fields are absent.

    Cost boundary: per-row numpy HL-Gauss, same as the live finalize path.
    Acceptable for the flag-gated offline use (overlapped by the host prefetch
    thread); vectorize before it could become a live-training default.
    """
    if float(params.blend_frac) <= 0.0 and float(params.search_blend_frac) <= 0.0:
        return arrs
    if "categorical_target" not in arrs or "wdl_target" not in arrs:
        return arrs
    wdl = np.asarray(arrs["wdl_target"]).reshape(-1)
    n = int(wdl.shape[0])
    cat = np.array(arrs["categorical_target"], copy=True)
    if n == 0 or cat.ndim != 2:
        return arrs
    num_bins = int(cat.shape[1])  # match stored width so the head's shape holds
    sf_wdl = arrs.get("sf_wdl")
    has_sf = np.asarray(
        arrs.get("has_sf_wdl", np.zeros((n,), dtype=np.float32))
    ).reshape(-1)
    search_wdl = arrs.get("search_wdl")
    has_search = np.asarray(
        arrs.get("has_search_wdl", np.zeros((n,), dtype=np.float32))
    ).reshape(-1)
    for i in range(n):
        scalar_v = 1.0 if int(wdl[i]) == 0 else (0.0 if int(wdl[i]) == 1 else -1.0)
        row = (
            np.asarray(sf_wdl[i])
            if sf_wdl is not None and i < has_sf.shape[0] and bool(has_sf[i])
            else None
        )
        srow = (
            np.asarray(search_wdl[i])
            if search_wdl is not None and i < has_search.shape[0] and bool(has_search[i])
            else None
        )
        value = categorical_target_value(
            scalar_v, row,
            blend_frac=params.blend_frac,
            search_wdl=srow,
            search_blend_frac=params.search_blend_frac,
        )
        cat[i] = hlgauss_target(
            value, num_bins=num_bins, sigma=params.sigma,
        ).astype(cat.dtype, copy=False)
    arrs["categorical_target"] = cat
    return arrs
