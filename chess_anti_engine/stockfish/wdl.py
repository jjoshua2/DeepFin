from __future__ import annotations

import math

import numpy as np


# Largest magnitude a raw cp score can reach: the shard clamps cp to +/-32000
# and Stockfish really does emit |cp| ~ 20000 in decisive endgames.
SF_CP_CLAMP_CP = 32000.0

# The mate band. Every consumer of `mate_to_effective_cp` compares its output
# DIRECTLY against a raw cp score drawn from the same MultiPV list, so the two
# bands must not overlap: a base inside the cp range ranks a large non-mating
# cp line ABOVE a forced mate. The historical base of 1500 did exactly that on
# 1.34% of live scored rows (a mate-in-3 scoring 2440 against cp-19998
# decliners), which is what split this mapping in two. Keep
# `_MATE_BASE_CP - _MATE_MAX_PLIES * _MATE_DEPTH_STEP_CP > SF_CP_CLAMP_CP`;
# `tests/test_mate_score_single_home.py` pins the inequality.
_MATE_BASE_CP = 100000.0
_MATE_DEPTH_STEP_CP = 100.0
# Distance-to-mate floor. Without it a long enough mate walks down through the
# cp band and eventually flips sign; SF mates are far shorter than 500 plies,
# so this is unreachable in practice and exists to keep the ordering total.
_MATE_MAX_PLIES = 500.0


def mate_to_effective_cp(mate_in: int) -> float:
    """Map a mate-in-N score to an effective centipawn value that DOMINATES cp.

    THE single mate -> score mapping: `selfplay.finalize._sf_move_score` and its
    C twin in `encoding/_lc0_ext.c` are mirrors of this function, not
    independent formulas. Sign(mate_in) carries the side, magnitude shrinks
    with distance to mate so a quicker mate outranks a slower one, and the
    whole band sits above `SF_CP_CLAMP_CP` so every mate outranks every raw cp
    score. `cp_to_wdl` saturates the band as decisive.

    ``mate_in=0`` means "no mate" in every stored row and each production
    caller guards on it before calling; it is mapped to a positive mate here
    only so the vectorised twin can compute unconditionally under a mask.
    """
    sign = 1.0 if mate_in >= 0 else -1.0
    plies = min(float(abs(int(mate_in))), _MATE_MAX_PLIES)
    return sign * (_MATE_BASE_CP - plies * _MATE_DEPTH_STEP_CP)


# `np.exp` overflows past ~709.78, and the mate band puts the logistic argument
# at slope * 100000 = 600 for production slope 0.006 — one slope bump away from
# a RuntimeWarning on every mate row. Clipping the EXPONENT is bitwise-neutral:
# below the clip nothing changes, and above it exp is ~1e304 so 1/(1+exp) is
# 0.0, the same value the overflowed inf produced. Do NOT rewrite this as a
# branchy stable sigmoid — `cp_to_wdl` and `cp_to_wdl_array` are pinned to
# bitwise agreement in tests/test_wdl_vectorised.py.
_EXP_CLIP = 700.0


def cp_to_wdl(
    cp: float | None,
    mate: int | None,
    *,
    slope: float,
    draw_width_cp: float,
) -> np.ndarray:
    """Convert a Stockfish cp/mate score to (W, D, L) probabilities.

    Logistic with explicit draw zone:
        p_win  = sigmoid( slope * (eff_cp - draw_width_cp) )
        p_loss = sigmoid( slope * (-eff_cp - draw_width_cp) )
        p_draw = 1 - p_win - p_loss   (clamped >= 0, then renormalised)

    ``slope`` (per-cp) controls steepness; ``draw_width_cp`` is the half
    width of the draw zone — at |cp| ~= draw_width_cp the side ahead has
    p_win ≈ 0.5. Both must be > 0; callers wanting the no-op should keep
    SF's UCI_ShowWDL output instead.

    ``mate`` takes precedence over ``cp`` when present (matches the UCI
    convention — SF emits at most one of the two per info line).
    """
    if slope <= 0.0 or draw_width_cp < 0.0:
        raise ValueError(f"cp_to_wdl requires slope>0 and draw_width_cp>=0, got {slope=} {draw_width_cp=}")
    if mate is not None:
        eff_cp = mate_to_effective_cp(int(mate))
    elif cp is not None:
        eff_cp = float(cp)
    else:
        raise ValueError("cp_to_wdl needs either cp or mate")
    p_win = 1.0 / (1.0 + np.exp(np.clip(-slope * (eff_cp - draw_width_cp), -_EXP_CLIP, _EXP_CLIP)))
    p_loss = 1.0 / (1.0 + np.exp(np.clip(-slope * (-eff_cp - draw_width_cp), -_EXP_CLIP, _EXP_CLIP)))
    p_draw = max(0.0, 1.0 - p_win - p_loss)
    total = p_win + p_loss + p_draw
    return np.array([p_win, p_draw, p_loss], dtype=np.float32) / float(total)


# The two modes `selfplay.search_wdl_draw_mode` selects between. Strings, not a
# bool, because a third D source is already sketched in the ledger (tree-backed-up
# D) and `search_wdl_draw_parametric: false` would have to be renamed to add it.
SEARCH_WDL_DRAW_NET_RAW = "net_raw"
SEARCH_WDL_DRAW_PARAMETRIC_Q = "parametric_q"
SEARCH_WDL_DRAW_MODES = (SEARCH_WDL_DRAW_NET_RAW, SEARCH_WDL_DRAW_PARAMETRIC_Q)


def parametric_draw_from_q(q: float, *, slope: float, draw_width_cp: float) -> float:
    """Draw mass implied by ``cp_to_wdl`` for a position whose W-L score is ``q``.

    THE definition of the ``parametric_q`` search-WDL draw channel; its C twin
    lives in ``mcts/_mcts_tree.c`` (``py_batch_process_ply``) and
    ``tests/test_search_wdl_draw_mode.py`` pins the two to agreement on a grid.

    Derivation (exact, not a fit). With ``a = sigmoid(slope*(cp - width))``,
    ``b = sigmoid(-slope*(cp + width))``, ``cp_to_wdl`` returns ``(a, 1-a-b, b)``
    -- the draw mass is never clamped and the triple always sums to 1, because
    ``a + b`` rises monotonically from ``2*sigmoid(-w) < 1`` at ``cp = 0`` to 1
    as ``|cp| -> inf`` (its derivative is ``s'(u-w) - s'(u+w) > 0`` for u > 0).
    So with ``x = exp(slope*cp)``, ``W = exp(w)`` and ``w = slope*draw_width_cp``:

        q = W(x^2 - 1) / [W x^2 + (1+W^2) x + W]
        D = (W^2 - 1) x / [W x^2 + (1+W^2) x + W]

    Dividing through by ``x`` and eliminating ``x + 1/x`` via
    ``(x + 1/x)^2 - (x - 1/x)^2 = 4`` gives a relation with no ``x`` in it,

        D^2 - 2*coth(w)*D + (1 - q^2) = 0   =>   D = coth(w) - sqrt(csch(w)^2 + q^2)

    (the other root exceeds 1). ``cp`` has vanished: D is a function of the
    searched q ALONE, which is the whole point -- the production construction
    reads the net's own raw root draw output instead
    (``_mcts_tree.c``, ``float d_raw = wdl_net[1]``).

    Properties, all exact rather than approximate:
      * ``D(0) = tanh(w/2)`` -- the cp-logistic's own draw mass at cp = 0
        (0.34521 at production slope 0.006 / width 120).
      * ``D(+-1) = 0`` exactly, since ``csch^2 + 1 = coth^2``.
      * strictly decreasing in ``|q|`` (derivative ``-|q|/sqrt(csch^2+q^2)``).
      * ``0.5*(1 - D +- q) >= 0`` on ``q in [-1, 1]`` IN EXACT ARITHMETIC, so
        the triple is a valid simplex point by construction rather than by a
        clamp -- which is the second defect this mode removes: the production
        form clamps the searched q to ``+-(1 - d_raw)``, capping the target's
        confidence on decisive rows (measured 12.3% of the lowest-d_raw
        quartile). ⚑ "By construction" is a statement about the ALGEBRA, not
        about the float32 the caller stores: rounding D up to float32 can make
        ``1 - D < |q|`` and drive W negative at wide draw zones (measured at
        w = 10). ``q_to_wdl_parametric`` and its C twin therefore clamp W in
        float32; that is a rounding guard, not a re-clamp of q.

    ``slope`` and ``draw_width_cp`` MUST be the same knobs the SF component
    uses (``sf_wdl_cp_slope`` / ``sf_wdl_cp_draw_width``); passing anything else
    silently desynchronises the two halves of the blended value target.
    """
    if slope <= 0.0 or draw_width_cp <= 0.0:
        # width 0 is legal for `cp_to_wdl` (a zero-width draw zone) but sends
        # coth/csch to infinity here. Refuse rather than special-case a
        # degenerate curve nobody asked for.
        raise ValueError(
            "parametric_draw_from_q requires slope>0 and draw_width_cp>0, got "
            f"{slope=} {draw_width_cp=}",
        )
    w = float(slope) * float(draw_width_cp)
    sh = math.sinh(w)
    coth = math.cosh(w) / sh
    csch = 1.0 / sh
    qc = max(-1.0, min(1.0, float(q)))
    d = coth - math.sqrt(csch * csch + qc * qc)
    # Range guard on floating-point dust ONLY: in exact arithmetic
    # 0 <= D <= 1 - |q| holds everywhere on [-1, 1] (both bounds proved above),
    # and at |q| = 1 the two coincide at 0 -- where double rounding lands
    # -2.2e-16. ⚑ This is NOT the production `d_raw` clamp reintroduced: that
    # one bounds the searched q by a bound the NET chose (and binds on 12.3% of
    # the lowest-d_raw quartile); this one bounds D by a function of the SAME q,
    # so it cannot bind on any input the algebra admits.
    return min(max(0.0, d), 1.0 - abs(qc))


def q_to_wdl_parametric(q: float, *, slope: float, draw_width_cp: float) -> np.ndarray:
    """``(W, D, L)`` built from the searched q alone -- no net-WDL input at all.

    ``D = parametric_draw_from_q(q)``, ``W = 0.5*(1 - D + q)``,
    ``L = 0.5*(1 - D - q)``. Sums to 1 identically and is non-negative on
    ``q in [-1, 1]``; see ``parametric_draw_from_q`` for the derivation and for
    why the q clamp there is a range guard, not the production d_raw clamp.

    ⚑ The triple is rebuilt in **float32, one operation at a time**, because
    that is exactly what the C twin does with the same rounded ``D`` --- so the
    two paths agree BITWISE rather than to a tolerance, and
    ``tests/test_search_wdl_draw_mode.py`` asserts that. Computing the tail in
    float64 and casting at the end would differ from C by an ulp and, worse,
    would hide the failure this shape exists to prevent: rounding ``D`` UP to
    float32 can leave ``1 - D < |q|``, so the float32 ``W`` goes negative even
    though the float64 one does not. Clamp where the value is stored.
    """
    qc = max(-1.0, min(1.0, float(q)))
    d = np.float32(parametric_draw_from_q(qc, slope=slope, draw_width_cp=draw_width_cp))
    # `rem - w` rather than `0.5*(rem - q)` so the triple sums to `d + rem`
    # exactly, matching the C twin and the shape of the net_raw branch it
    # replaces.
    rem = np.float32(1.0) - d
    w = np.float32(0.5) * (rem + np.float32(qc))
    if w < np.float32(0.0):
        w = np.float32(0.0)
    if w > rem:
        w = rem
    return np.array([w, d, rem - w], dtype=np.float32)


def mate_to_effective_cp_array(mate_in: np.ndarray) -> np.ndarray:
    """Vectorised twin of `mate_to_effective_cp` (elementwise, float64 out).

    Same arithmetic, same constants; `tests/test_wdl_vectorised.py` pins the
    two to bitwise agreement so the scalar version stays the definition.
    """
    m = np.asarray(mate_in)
    sign = np.where(m >= 0, 1.0, -1.0)
    plies = np.minimum(np.abs(m).astype(np.float64, copy=False), _MATE_MAX_PLIES)
    return sign * (_MATE_BASE_CP - plies * _MATE_DEPTH_STEP_CP)


def cp_to_wdl_array(
    eff_cp: np.ndarray,
    *,
    slope: float,
    draw_width_cp: float,
) -> np.ndarray:
    """Vectorised twin of `cp_to_wdl`, over an array of EFFECTIVE cp.

    Takes effective cp (mate already folded in by `mate_to_effective_cp_array`)
    because batch callers resolve the mate/cp precedence themselves, with a
    mask, rather than per element. Returns ``(..., 3)`` float32 (W, D, L),
    reproducing the scalar function's exact dtype sequence: the triple is cast
    to float32 and only then divided by the float64 total, matching NumPy's
    weak-scalar rule in `cp_to_wdl`. `tests/test_wdl_vectorised.py` pins the
    two to bitwise agreement.
    """
    if slope <= 0.0 or draw_width_cp < 0.0:
        raise ValueError(f"cp_to_wdl requires slope>0 and draw_width_cp>=0, got {slope=} {draw_width_cp=}")
    eff = np.asarray(eff_cp, dtype=np.float64)
    p_win = 1.0 / (1.0 + np.exp(np.clip(-slope * (eff - draw_width_cp), -_EXP_CLIP, _EXP_CLIP)))
    p_loss = 1.0 / (1.0 + np.exp(np.clip(-slope * (-eff - draw_width_cp), -_EXP_CLIP, _EXP_CLIP)))
    p_draw = np.maximum(0.0, 1.0 - p_win - p_loss)
    total = p_win + p_loss + p_draw
    stacked = np.stack([p_win, p_draw, p_loss], axis=-1).astype(np.float32, copy=False)
    return stacked / total[..., None].astype(np.float32, copy=False)
