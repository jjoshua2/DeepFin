"""Target-surgery ladder (arms R / V / G): the join key and the three edits.

The 2026-08-22 prereg ("target-surgery ladder R/V/G") uses SF as an EDITOR of
the stored search target rather than as a CE-teacher — the form dose-ladder2
killed. This module is the piece the label pass (``scripts/rvg_label_pass.py``)
and the offline retrain rig (``scripts/retarget_retrain.py``) share, so the two
sides cannot drift:

* the JOIN KEY, derived from the stored input planes on BOTH sides by the SAME
  function (never from a board on one side and planes on the other — see
  ``position_fingerprints``);
* MultiPV folding to per-move cp regret, reusing
  ``chess_anti_engine.stockfish.wdl.mate_to_effective_cp`` and the audit's
  ``AUDIT_REGRET_CAP_CP`` rather than a second copy of those rules;
* the three edits, as pure array functions with hand-checkable arithmetic.

⚑ THE LABEL IS A WIDE-MultiPV / 150k-node / ``fresh=True`` / Syzygy-on SHALLOW
pass. It is NOT the stored MPV6 ``sf_p0_*`` fields: the coverage curve
(2026-08-22) put MPV6's reach at 60.9% of the target's bad mass against MPV20's
95.3%, and the warm-TT finding the same day showed the banked MPV6 rows do not
even reproduce themselves (32.3% set agreement). Nothing here reads
``sf_p0_regret``/``sf_p0_policy_target`` as an INPUT.

⚑ THE LABEL CANNOT SEE REPETITION HISTORY, AND NEITHER CAN THE JOIN KEY. The
label pass rebuilds a board from the stored planes and hands SF a FEN, so the
repetition stack is gone: SF scores the position as a first occurrence while the
training row may have arisen after one, and two rows differing only in
repetition count share ONE key and ONE label. Accepted for v1, and neither
consequence is invisible: a fortress or perpetual whose true value is a draw by
repetition can be labeled as a live evaluation. The direction is the safe one —
an over-optimistic ``best_cp`` makes the veto fire LESS, not more — and it
applies uniformly to every arm and to the control, so it cannot manufacture a
difference between them.

TB-LINE SENTINELS (``cp ~ +/-20000`` with Syzygy on) ARE TREATED AS REAL
EVALUATIONS. They are folded through the ordinary ``best_cp - cp`` regret and
then clipped at ``AUDIT_REGRET_CAP_CP`` (1000 cp), exactly like the mate band
``mate_to_effective_cp`` puts near +/-100000. A tablebase loss really is a
losing move, so a hard veto on it is the intended behaviour, not a sentinel
leak; the cap is what stops a decided position from dominating a mean. The one
consequence worth stating: any V ``veto_cp <= 1000`` vetoes every TB-loss and
every mated line, because both saturate the cap.
"""
from __future__ import annotations

import array
import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from chess_anti_engine.encoding.lc0 import (
    LC0_FULL,
    normalize_lc0_history_encoding,
    uses_lc0_root_history,
)
from chess_anti_engine.eval.audit import AUDIT_REGRET_CAP_CP
from chess_anti_engine.stockfish.wdl import mate_to_effective_cp

#: Bumped whenever a label file's meaning changes. The rig refuses a file whose
#: version it does not know rather than joining against a different convention.
#: v2 stores per-move EFFECTIVE CP instead of regret: regret is relative to a
#: pass's own ``best_cp``, so two passes' regrets cannot be overlaid — only
#: their cp can, with ``best_cp`` recomputed over the overlaid set.
RVG_LABEL_SCHEMA_VERSION = 2

#: Arm B's external-policy file format. Separate from the label schema on
#: purpose: the two files are produced by different programs on different
#: branches, and a shared version number would make a bump in one of them read
#: as a bump in the other.
RVG_EXTERNAL_POLICY_SCHEMA_VERSION = 1


def file_provenance(path: str | Path) -> dict[str, object]:
    """``{path, bytes, mtime_utc}`` for an input file, for the run report.

    ⚑ A PATH ALONE DOES NOT IDENTIFY A FILE. ``--rvg-labels labels.jsonl`` names
    a file that a resumed label pass appends to while a sweep reads it, and two
    runs quoting the same path can have read different bytes. Size and mtime are
    what distinguish them; they are not a content hash, and are not claimed to
    be — hashing a multi-gigabyte label file at the head of every sweep would
    cost minutes to defend against a case (a same-size same-mtime edit) that
    does not occur here.
    """
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return {"path": str(p), "bytes": None, "mtime_utc": None}
    return {
        "path": str(p),
        "bytes": int(st.st_size),
        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
    }

#: Digest width of the join key, in bytes. 16 bytes over ~1.5M corpus rows puts
#: the birthday collision probability at ~1e-10; the key is also EXACT in the
#: sense that both sides hash the identical canonical bytes, so a collision is
#: the only failure mode and it is bounded here rather than argued away.
FINGERPRINT_BYTES = 16

#: Sentinel produced when a row's planes cannot be canonicalized (wrong plane
#: count / corrupt row). Never joins: the label pass skips these rows and the
#: rig counts them as uncovered.
UNFINGERPRINTABLE = b""


# ---------------------------------------------------------------------------
# Join key
# ---------------------------------------------------------------------------
#
# ⚑ BOTH SIDES HASH THE PLANES, NOT A BOARD. `decode_board_from_planes` drops an
# en-passant square python-chess deems invalid (no capturer) and rejects rows
# whose status is not VALID; a key taken off the BOARD on one side and off the
# PLANES on the other would therefore disagree on exactly those rows, silently,
# and the disagreement would read as "this row has no label" — the repo's
# signature defect. `board_fingerprint` exists to PROVE the plane derivation is
# a faithful position identity (tests), never to produce a key for the join.


def _plane_layout(input_history_encoding: str | None) -> tuple[tuple[int, ...], int, int | None]:
    """``(castle_planes, rule50_plane, ep_plane)`` for one history encoding.

    Castling order is ``(us-K, us-Q, them-K, them-Q)`` — the same order
    ``chess_anti_engine.eval.audit.decode_board_from_planes`` uses, read off
    ``encoding/lc0.py``'s writers rather than re-derived. ``ep_plane`` is
    ``None`` for plain ``lc0_root``, which carries no EP metadata plane.
    """
    hist = normalize_lc0_history_encoding(input_history_encoding)
    if uses_lc0_root_history(hist):
        base = LC0_FULL.root_metadata_base
        ep = base + 6 if hist == "lc0_root_legacy_meta" else None
        return (base + 1, base, base + 3, base + 2), base + 5, ep
    base = LC0_FULL.metadata_base
    return (base, base + 1, base + 2, base + 3), base + 6, base + 4


def _rule50_scale(input_history_encoding: str | None) -> float:
    """Multiplier turning the stored rule50 plane back into a half-move count.

    ``lc0_root`` stores the RAW counter; ``legacy`` and ``lc0_root_legacy_meta``
    store ``min(clock, 100) / 100``. Getting this backwards would not raise —
    it would quantize every row's clock to 0 and hand two genuinely different
    positions the same key.
    """
    hist = normalize_lc0_history_encoding(input_history_encoding)
    return 1.0 if hist == "lc0_root" else 100.0


def position_fingerprints(
    x: np.ndarray, *, input_history_encoding: str | None,
) -> list[bytes]:
    """Join keys for a batch of stored input planes. Vectorized.

    ``x`` is ``(N, C, 8, 8)``. Returns one ``FINGERPRINT_BYTES``-long digest per
    row (``UNFINGERPRINTABLE`` for rows whose plane block is too narrow to carry
    the metadata).

    The hashed canonical form is the position AS THE NETWORK SEES IT: the 12
    side-to-move-POV piece bitboards (bit ``rank*8 + file``, matching
    ``encoding/plane_decode.decode_step0_bitboards`` and the square arithmetic
    in ``audit.decode_board_from_planes``), the four castling bits, the EP file
    (+1, 0 = none) and the half-move clock. Colour is deliberately absent —
    the encoding is POV-normalized, so a position and its colour-mirror are ONE
    position here, share one SF label, and legitimately share one key.

    The half-move clock is IN the key. ``audit.position_key`` drops it (it is a
    dedup key for a set whose evaluations are clock-insensitive); a shallow
    label with Syzygy on is not clock-insensitive — DTZ conversion and the
    50-move rule both read it — so two rows differing only in rule50 must not
    share a label.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[None, ...]
    n = int(arr.shape[0])
    castle_planes, rule50_plane, ep_plane = _plane_layout(input_history_encoding)
    need = max((*castle_planes, rule50_plane, ep_plane or 0)) + 1
    if arr.ndim != 4 or arr.shape[-2:] != (8, 8) or arr.shape[1] < need:
        return [UNFINGERPRINTABLE] * n

    bits = (arr[:, :12] > 0.5).astype(np.uint64)
    weights = (np.uint64(1) << np.arange(64, dtype=np.uint64)).reshape(8, 8)
    boards = (bits * weights).sum(axis=(-2, -1), dtype=np.uint64)   # (N, 12)

    castle = np.zeros((n,), dtype=np.uint64)
    for shift, plane in enumerate(castle_planes):
        castle |= (arr[:, plane].max(axis=(-2, -1)) > 0.5).astype(np.uint64) << np.uint64(shift)

    if ep_plane is None:
        ep = np.zeros((n,), dtype=np.uint64)
    else:
        per_file = arr[:, ep_plane].max(axis=-2) > 0.5              # (N, 8)
        # +1 so "no EP" (0) is distinct from "EP on the a-file" (1). A plane
        # with more than one file set is corrupt; it reads as 0 here and the
        # row simply fails to match a label rather than matching a wrong one.
        single = per_file.sum(axis=-1) == 1
        ep = np.where(single, np.argmax(per_file, axis=-1) + 1, 0).astype(np.uint64)

    scale = _rule50_scale(input_history_encoding)
    clock = np.clip(np.rint(arr[:, rule50_plane, 0, 0] * scale), 0, 150).astype(np.uint64)

    canon = np.concatenate(
        [boards, castle[:, None], ep[:, None], clock[:, None]], axis=1,
    ).astype("<u8", copy=False)
    return [
        hashlib.blake2b(row.tobytes(), digest_size=FINGERPRINT_BYTES).digest()
        for row in canon
    ]


def board_fingerprint(board: chess.Board) -> bytes:
    """The same key, derived from a side-to-move-canonical ``chess.Board``.

    VERIFICATION ONLY — the join never calls this (see the module note above).
    Its job is to let a test assert
    ``board_fingerprint(decode_board_from_planes(x)) == position_fingerprints(x)``
    on real corpus rows, which is what makes "the digest identifies the
    position" a measured claim rather than a comment.

    ``board`` must be the canonical form ``audit.decode_board_from_planes``
    returns: WHITE to move, white = the original side to move.
    """
    if board.turn != chess.WHITE:
        raise ValueError("board_fingerprint expects the side-to-move-canonical form (white to move)")
    boards = np.array(
        [int(board.pieces_mask(pt, color))
         for color in (chess.WHITE, chess.BLACK)
         for pt in chess.PIECE_TYPES],
        dtype=np.uint64,
    )
    castle = np.uint64(
        (1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
        | (2 if board.has_queenside_castling_rights(chess.WHITE) else 0)
        | (4 if board.has_kingside_castling_rights(chess.BLACK) else 0)
        | (8 if board.has_queenside_castling_rights(chess.BLACK) else 0)
    )
    ep = np.uint64(
        0 if board.ep_square is None else chess.square_file(board.ep_square) + 1
    )
    clock = np.uint64(max(0, min(150, int(board.halfmove_clock))))
    canon = np.concatenate([boards, [castle], [ep], [clock]]).astype("<u8", copy=False)
    return hashlib.blake2b(canon.tobytes(), digest_size=FINGERPRINT_BYTES).digest()


# ---------------------------------------------------------------------------
# MultiPV -> per-move cp regret
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiPvLine:
    """One raw MultiPV line, as the engine reported it.

    A typed record rather than a loose dict because ``cp``/``mate`` are the two
    fields whose ``None`` handling decides whether a position is scoreable at
    all — a dict of ``object`` makes every one of those reads a cast, and a cast
    is exactly where an unchecked ``None`` slips through.
    """

    move: str
    cp: int | None
    mate: int | None


@dataclass(frozen=True)
class FoldedMultiPv:
    """One position's shallow label, folded to per-move cp regret.

    ``moves`` are UCI strings in the SAME (side-to-move-canonical) frame the
    planes are stored in, so they encode straight into the shard's policy space.
    ``regret_cp`` is aligned with ``moves`` and already clipped to
    ``AUDIT_REGRET_CAP_CP``.
    """

    moves: tuple[str, ...]
    cp_eff: tuple[float, ...]
    regret_cp: tuple[float, ...]
    best_cp: float


def fold_multipv(pvs: Sequence[MultiPvLine]) -> FoldedMultiPv | None:
    """Fold raw MultiPV lines into per-move cp regret. ``None`` if unscoreable.

    Rules, all shared with the audit path rather than re-invented:

    * a mate line folds through ``mate_to_effective_cp`` (dominates every cp);
    * FIRST LISTING WINS on a duplicated move — the engine's own MultiPV rank
      order is authoritative, and re-ranking by cp would build a silently
      BETTER ruler than the one production's label sees;
    * a line with neither cp nor mate is skipped, not crashed on;
    * ``regret = clip(best_cp - cp, 0, AUDIT_REGRET_CAP_CP)``.

    ⚑ ``best_cp`` is the MAX over LISTED lines, so it is the shallow pass's own
    best — not a deeper ruler's. Regret is therefore relative to what THIS label
    knows, which is the quantity every edit here is entitled to use.
    """
    move_cp: dict[str, float] = {}
    for row in pvs:
        if row.mate is not None and int(row.mate) != 0:
            eff = float(mate_to_effective_cp(int(row.mate)))
        elif row.cp is not None:
            eff = float(row.cp)
        else:
            continue
        move_cp.setdefault(row.move, eff)
    if not move_cp:
        return None
    best = max(move_cp.values())
    moves = tuple(move_cp)
    cp_eff = tuple(float(move_cp[m]) for m in moves)
    return FoldedMultiPv(
        moves=moves,
        cp_eff=cp_eff,
        regret_cp=regrets_from_cp(cp_eff, best),
        best_cp=float(best),
    )


def regrets_from_cp(
    cp_eff: Sequence[float], best_cp: float | None = None,
) -> tuple[float, ...]:
    """``clip(best - cp, 0, AUDIT_REGRET_CAP_CP)`` — the ONE regret derivation.

    ⚑ REGRET IS RELATIVE TO ``best_cp``, WHICH IS A PROPERTY OF THE SET. That is
    why label files store effective CP and this function is called once the set
    is final: overlaying two passes' REGRETS would silently mix two different
    baselines (a deep-narrow pass's best and a wide-shallow pass's best are not
    the same number), and the result would be finite, in range, and wrong.
    ``best_cp=None`` recomputes it over ``cp_eff`` — what the layered join wants.
    """
    if not len(cp_eff):
        return ()
    best = max(cp_eff) if best_cp is None else float(best_cp)
    return tuple(
        float(min(AUDIT_REGRET_CAP_CP, max(0.0, best - float(cp)))) for cp in cp_eff
    )


# ---------------------------------------------------------------------------
# The three edits
# ---------------------------------------------------------------------------
#
# All three take ONE row: the stored target `t` (P,), the listed moves' policy
# indices `idx` (K,) and their cp regrets `reg` (K,). All three are pure, take
# no config object, and return a NEW array — the wrappers in the rig do the
# masking and the counting, so the arithmetic below is what the unit tests can
# hand-check against a 3-move example.


def _renormalize_or_fall_back(
    edited: np.ndarray, original: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """``(t', fell_back)`` — renormalize, or return ``original`` untouched.

    An edit that removes ALL of a row's mass leaves nothing to normalize. The
    row is then served UNEDITED and counted, rather than served as a uniform or
    a zero vector: a zero target silently drops the row from the policy CE
    (``has_policy`` still says it is present), and a uniform one is a fabricated
    label. Falling back keeps the row honest and makes the event countable.
    """
    total = float(edited.sum())
    if not np.isfinite(total) or total <= 0.0:
        return original, True
    return (edited / total).astype(original.dtype, copy=False), False


def apply_veto_edit(
    t: np.ndarray,
    idx: np.ndarray,
    reg_cp: np.ndarray,
    *,
    lam: float,
    tau_cp: float,
    veto_cp: float,
) -> tuple[np.ndarray, bool]:
    """Arm V. ``t'(m) ~ t(m) * exp(-lam * relu(regret(m) - tau))`` on LISTED m.

    Plus a hard zero for listed moves whose regret is ``>= veto_cp``. Returns
    ``(t', fell_back)``.

    ⚑ UNLISTED MOVES ARE UNTOUCHED — no fabricated regret, no default weight.
    Their multiplier is exactly 1.0, which is the whole difference between this
    and June's PR #78 form (unlisted defaulted to regret 1.0, i.e. a MEMBERSHIP
    prior wearing a magnitude arm's clothes). The final renormalization does
    rescale them, because ``t'`` has to remain a distribution — that is the
    ``~`` in the formula, and it is a global constant, not a per-move edit.

    ``lam = 0`` with ``veto_cp`` above the cap is the identity up to
    renormalization; the rig refuses to activate an arm that cannot move
    anything rather than letting it report as a dose.
    """
    t = np.asarray(t, dtype=np.float32)
    weights = np.ones_like(t, dtype=np.float64)
    if idx.size:
        r = np.asarray(reg_cp, dtype=np.float64)
        w = np.exp(-float(lam) * np.maximum(0.0, r - float(tau_cp)))
        w = np.where(r >= float(veto_cp), 0.0, w)
        # `np.minimum.at`, not plain fancy assignment: a duplicated index (two
        # MultiPV lines folding to the same policy slot — under-promotions and
        # the castling spellings both can) must take the HARSHEST weight rather
        # than whichever line happened to be written last. Silent order
        # dependence is exactly what `fold_multipv`'s first-listing-wins rule
        # exists to remove upstream; this closes it downstream too.
        np.minimum.at(weights, np.asarray(idx, dtype=np.int64), w)
    return _renormalize_or_fall_back((t.astype(np.float64) * weights), t)


def apply_geometric_blend(
    t: np.ndarray,
    idx: np.ndarray,
    reg_cp: np.ndarray,
    *,
    alpha: float,
    temp_cp: float,
    q_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Arm G. Log-linear pool of ``t`` with the sharpened SF distribution ``q``.

    ``q(m) ~ exp(-regret(m) / T)`` over LISTED moves; on those moves
    ``t'(m) ~ t(m)^(1-alpha) * q(m)^alpha``. Returns ``(t', fell_back)``.

    ⚑⚑ THE UNLISTED CONVENTION, DECIDED AND DOCUMENTED. The prereg writes
    ``q = 0`` for unlisted moves and then says they "keep reduced t-mass with
    alpha < 1". Those two statements are inconsistent: ``t^(1-a) * 0^a == 0``
    for every ``a > 0``, so a literal ``q = 0`` ANNIHILATES every unlisted move
    rather than reducing it. This implementation follows the stated BEHAVIOUR,
    not the literal formula:

        listed   m: t'(m) ~ t(m)^(1-alpha) * q(m)^alpha
        unlisted m: t'(m) ~ t(m)                      (multiplier exactly 1)

    i.e. the geometric pool is RESTRICTED to the listed set and unlisted moves
    keep their t-mass up to the global renormalization — the same
    "never fabricate a value for a move SF did not list" rule arm V follows, and
    the same reason arm R must not default unlisted regret. The alternative
    reading (annihilate the unlisted set) is a strictly harder edit than V's
    hard veto and is NOT what this arm tests.

    ``q_weights`` (arm B) replaces the regret-derived ``q`` with an EXTERNAL
    per-move distribution — an lc0/BT4 net's policy, read from a file — aligned
    with ``idx``. Everything else is identical, deliberately: the support rule,
    the unlisted convention, the renormalization and the fall-back are the same
    code, so arm B differs from arm G in its INPUT and in nothing else.
    ``temp_cp`` is ignored when ``q_weights`` is given (see below).

    ``0 ** 0``: cannot arise from ``alpha`` (the rig refuses ``alpha <= 0``) and
    cannot arise from ``q`` (``exp`` of a finite number is strictly positive on
    every listed move). It CAN arise from ``t(m) == 0`` on a listed move, where
    ``t ** (1-alpha)`` with ``alpha < 1`` is ``0`` — the move stays at zero.
    That is deliberate: G edits the SUPPORT OF ``t`` and never adds mass to a
    move the search target never visited. ``alpha == 1`` would make
    ``t ** 0 == 1`` and grow the support, which is why the rig's band is
    ``0 < alpha < 1``.
    """
    t = np.asarray(t, dtype=np.float32)
    if not idx.size:
        return t, False
    a = float(alpha)
    if q_weights is None:
        r = np.asarray(reg_cp, dtype=np.float64)
        q = np.exp(-r / float(temp_cp))
    else:
        # EXTERNAL q (arm B): an already-shaped per-move distribution supplied by
        # another net, aligned with `idx`. `temp_cp` is NOT applied to it — the
        # file's probabilities are the distribution, and re-tempering a
        # distribution someone else calibrated would silently change the arm.
        # Negative entries are a corrupt file, not a sharpening: clamp at 0 and
        # let the all-zero case take the fall-back path below.
        q = np.clip(np.asarray(q_weights, dtype=np.float64), 0.0, None)
    q_total = float(q.sum())
    if not np.isfinite(q_total) or q_total <= 0.0:
        return t, True
    q = q / q_total

    edited = t.astype(np.float64).copy()
    j = np.asarray(idx, dtype=np.int64)
    # Duplicate policy slots: sum the q-mass onto the slot (it is a
    # distribution, so mass adds) before the power, so the pooled value does not
    # depend on which duplicate was written last.
    q_row = np.zeros_like(edited)
    np.add.at(q_row, j, q)
    listed = np.zeros(edited.shape, dtype=bool)
    listed[j] = True
    edited[listed] = (
        np.power(edited[listed], 1.0 - a) * np.power(q_row[listed], a)
    )
    return _renormalize_or_fall_back(edited, t)


def distribution_entropy_nats(p: np.ndarray) -> float:
    """Shannon entropy of one target row, in NATS. Zero-mass entries contribute 0.

    ⚑⚑ CALIBRATION INSTRUMENT ONLY — NEVER A SUCCESS METRIC AND NEVER A GATE.
    Arm G's ``(alpha, T)`` are chosen so the post-edit target entropy lands near
    a reference anchor (~0.970 nats, BT4), and that choice has to be checkable
    from the run report instead of asserted. It is reported for exactly that
    reason and for no other:
    [[an_arm_that_is_the_gradient_of_the_metric_always_wins]] — G's temperature
    IS the gradient of entropy, so scoring G on entropy would make the highest-T
    arm win by construction and tell us nothing. The deciding yardstick stays
    row-(a) top-1 deep-SF regret with the E[regret] co-criterion.

    The row is renormalized first: the stored targets are float16 and sum to
    1.0 only to within that precision, and an unnormalized vector's "entropy" is
    not an entropy.
    """
    q = np.asarray(p, dtype=np.float64)
    total = float(q.sum())
    if not np.isfinite(total) or total <= 0.0:
        return 0.0
    q = q / total
    nz = q[q > 0.0]
    return float(-(nz * np.log(nz)).sum())


def listed_only_regret_vector(
    width: int,
    idx: np.ndarray,
    reg_cp: np.ndarray,
    *,
    cap_cp: float,
) -> np.ndarray:
    """Arm R's per-move regret vector: LISTED moves only, unlisted exactly 0.

    Returns a ``(width,)`` float32 vector in ``[0, 1]`` (``regret_cp / cap_cp``)
    — the units ``train/losses.py`` expects for ``sf_p0_regret``, normalized by
    ``train/constants.SF_OWN_REGRET_CAP_CP``, which the caller passes so this
    module does not import the training package.

    ⚑ UNLISTED MOVES GET 0.0, WHICH IS *NOT* WHAT PRODUCTION'S VECTOR CARRIES.
    ``selfplay/finalize._build_sf_p0_regret_vector`` fills unlisted AND illegal
    moves with ``(worst_surfaced + 1) / 2`` (>= 0.5 always), and
    ``losses.sf_policy_floor_deficit`` RELIES on that: its membership set is
    ``regret <= delta_cp / cap`` (0.02 at the production 20 cp), so a vector
    with unlisted moves at 0.0 admits every unlisted legal move to the floor and
    floors it at ``tau``. The rig therefore REFUSES to activate arm R unless
    ``w_sf_policy_floor == 0`` for that arm — see
    ``scripts/retarget_retrain.py``. This docstring is the spec for that guard;
    do not relax one without the other.

    Duplicate policy slots take the MINIMUM regret (the best line wins), which
    is the harshest-for-the-net reading and matches ``fold_multipv``'s
    first-listing-wins rank authority.
    """
    out = np.zeros((int(width),), dtype=np.float64)
    if idx.size:
        r = np.clip(np.asarray(reg_cp, dtype=np.float64), 0.0, float(cap_cp)) / float(cap_cp)
        big = np.full(out.shape, np.inf)
        np.minimum.at(big, np.asarray(idx, dtype=np.int64), r)
        out = np.where(np.isfinite(big), big, 0.0)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Label file -> in-memory join index
# ---------------------------------------------------------------------------


def _header_number(header: dict[str, object], key: str) -> float:
    """A numeric provenance field, or 0.0. Headers are JSON, so every value
    arrives as ``object``; a bare ``float(header[key])`` is a cast, and a cast
    is where a string budget silently becomes an ordering."""
    value = header.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


class RvgLabelIndex:
    """Fingerprint -> (listed policy indices, cp regrets) for a label file set.

    Stored as flat arrays plus a per-key offset rather than ~1.5M small arrays:
    the banked corpus is ~1.5M rows at up to 40 lines each, and the flat form
    slices in constant time on the training hot path.

    ⚑ THE FILE IS VALIDATED, NOT TRUSTED. A schema version this code does not
    know, or a policy encoding that differs from the corpus being edited, is a
    refusal — both would otherwise produce an index that joins fine and edits
    the WRONG SLOTS, which no counter downstream can distinguish from a correct
    run.

    LAYERED MODE (default OFF, ``load`` takes one path unless asked otherwise).
    Passing several files overlays them per MOVE: the pass with more
    NODES-PER-LINE supplies a move's cp, and a move only the wider pass listed
    keeps the wider pass's cp. Regret is then recomputed over the OVERLAID set —
    which is exactly why records store effective CP and not regret
    (``regrets_from_cp``): a deep-narrow pass's ``best_cp`` and a wide-shallow
    pass's ``best_cp`` are different baselines, so overlaying regrets would mix
    them silently. ``mix`` reports how many moves each pass supplied.

    ⚑ THE OVERLAY ORDER IS DERIVED, NOT THE CLI ORDER. Passes are sorted
    ASCENDING by ``(nodes_per_line, nodes, name)`` and applied weakest-first so
    the strongest overwrites; the result therefore does not depend on the order
    the operator happened to type the files in.
    """

    def __init__(
        self,
        offsets: np.ndarray,
        move_index: np.ndarray,
        cp_eff: np.ndarray,
        regret_cp: np.ndarray,
        pass_id: np.ndarray,
        lookup: dict[bytes, int],
        header: dict[str, object],
        passes: list[dict[str, object]],
    ) -> None:
        self._offsets = offsets
        self._move_index = move_index
        self._cp_eff = cp_eff
        self._regret_cp = regret_cp
        self._pass_id = pass_id
        self._lookup = lookup
        self.header = header
        self.passes = passes

    def __len__(self) -> int:
        return len(self._lookup)

    @property
    def policy_encoding(self) -> str:
        return str(self.header.get("policy_encoding", ""))

    @property
    def mix(self) -> list[dict[str, object]]:
        """Per-pass count of the MOVES that pass finally supplied.

        The layered join's wiring proof: a second pass that supplied zero moves
        did not participate, however confidently the command line named it.
        """
        counts = np.bincount(self._pass_id, minlength=len(self.passes))
        return [
            {**meta, "moves_supplied": int(counts[i])}
            for i, meta in enumerate(self.passes)
        ]

    def get(self, key: bytes) -> tuple[np.ndarray, np.ndarray] | None:
        """``(move_index, regret_cp)`` for one position, or ``None``."""
        slot = self._lookup.get(key)
        if slot is None:
            return None
        lo, hi = int(self._offsets[slot]), int(self._offsets[slot + 1])
        return self._move_index[lo:hi], self._regret_cp[lo:hi]

    def cp_for(self, key: bytes) -> np.ndarray | None:
        """Effective cp for one position — the overlaid values, for diagnostics."""
        slot = self._lookup.get(key)
        if slot is None:
            return None
        lo, hi = int(self._offsets[slot]), int(self._offsets[slot + 1])
        return self._cp_eff[lo:hi]

    def pass_for(self, key: bytes) -> np.ndarray | None:
        """Which pass supplied each of this position's moves (index into
        ``passes``)."""
        slot = self._lookup.get(key)
        if slot is None:
            return None
        lo, hi = int(self._offsets[slot]), int(self._offsets[slot + 1])
        return self._pass_id[lo:hi]

    @staticmethod
    def _pass_strength(header: dict[str, object], name: str) -> tuple[float, float, str]:
        """Sort key: nodes per line, then total nodes, then the file name.

        Nodes-per-line is the depth a single line actually got, which is the
        quantity that decides whose cp is better resolved. The two tie-breaks
        make the order total, so two passes with the same budget still overlay
        deterministically instead of by dict iteration order.
        """
        nodes = _header_number(header, "nodes")
        multipv = _header_number(header, "multipv")
        return (nodes / multipv if multipv > 0 else 0.0, nodes, name)

    @classmethod
    @classmethod
    def _load_one_streaming(
        cls, file_path: Path, *, policy_encoding: str | None,
    ) -> RvgLabelIndex:
        """One label file, parsed straight into the flat arrays.

        Every refusal the overlay path makes is made here: schema version,
        move/cp length agreement, duplicate keys, and the corpus policy-space
        check. What is NOT here is the two intermediate dicts — the per-record
        move dict is transient, and the columns accumulate in ``array.array``
        (compact C storage, amortized append) rather than in lists of ndarrays.

        Move order and duplicate-slot handling match the overlay path exactly:
        moves ascending, and a slot named twice inside one record keeps the LAST
        cp — the same thing ``slot[move] = ...`` does there. That equality is
        what makes this a memory optimization rather than a second convention.
        """
        header: dict[str, object] = {}
        encodings: set[str] = set()
        lookup: dict[bytes, int] = {}
        offsets = array.array("q", [0])
        move_index = array.array("i")
        cp_eff = array.array("f")
        regret_cp = array.array("f")
        total = 0
        with open(file_path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                rec = json.loads(stripped)
                if rec.get("record") == "provenance":
                    header = dict(rec)
                    continue
                version = int(rec.get("v", -1))
                if version != RVG_LABEL_SCHEMA_VERSION:
                    raise SystemExit(
                        f"{file_path}: label schema version {version}, this "
                        f"build reads {RVG_LABEL_SCHEMA_VERSION}"
                    )
                encodings.add(str(rec.get("policy_encoding", "")))
                key = bytes.fromhex(str(rec["key"]))
                mi = [int(v) for v in rec["move_index"]]
                cp = [float(v) for v in rec["cp_eff"]]
                if len(mi) != len(cp):
                    raise SystemExit(
                        f"{file_path}: record {rec['key']} has {len(mi)} moves "
                        f"and {len(cp)} cp values"
                    )
                if key in lookup:
                    raise SystemExit(
                        f"{file_path}: duplicate key {rec['key']} — the join "
                        "would be ambiguous"
                    )
                # A slot named twice keeps the LAST cp — the same thing
                # `slot[move] = ...` does in the overlay path, and the equality
                # the two paths are compared on.
                slot: dict[int, float] = dict(zip(mi, cp, strict=True))
                moves = sorted(slot)
                cps = [slot[m] for m in moves]
                lookup[key] = len(lookup)
                move_index.extend(moves)
                cp_eff.extend(cps)
                # `best_cp=None`: over THIS record's set, the same rule the
                # overlay path applies over the overlaid set.
                regret_cp.extend(regrets_from_cp(cps))
                total += len(moves)
                offsets.append(total)

        file_encoding = next(iter(encodings)) if encodings else ""
        if policy_encoding is not None and file_encoding and file_encoding != policy_encoding:
            raise SystemExit(
                f"labels are in {file_encoding!r} policy space but the corpus "
                f"stores {policy_encoding!r} — the edits would address different "
                "moves than the ones they were computed for"
            )
        header.setdefault("nodes", 0)
        header.setdefault("multipv", 0)
        header["policy_encoding"] = file_encoding
        header["label_files"] = [str(file_path)]
        header["label_file_provenance"] = [file_provenance(file_path)]
        strength = cls._pass_strength(header, file_path.name)
        return cls(
            offsets=np.frombuffer(offsets, dtype=np.int64).copy(),
            move_index=np.frombuffer(move_index, dtype=np.int32).copy(),
            cp_eff=np.frombuffer(cp_eff, dtype=np.float32).copy(),
            regret_cp=np.frombuffer(regret_cp, dtype=np.float32).copy(),
            pass_id=np.zeros((total,), dtype=np.int8),
            lookup=lookup,
            header=header,
            passes=[{
                "name": file_path.name,
                "nodes": int(_header_number(header, "nodes")),
                "multipv": int(_header_number(header, "multipv")),
                "nodes_per_line": strength[0],
            }],
        )

    @classmethod
    def load(
        cls,
        path: str | Path | Sequence[str | Path],
        *,
        policy_encoding: str | None = None,
    ) -> RvgLabelIndex:
        paths = [Path(path)] if isinstance(path, (str, Path)) else [Path(p) for p in path]
        if not paths:
            raise SystemExit("RvgLabelIndex.load: no label files given")
        if len(paths) == 1:
            # ⚑ THE ONLY SHAPE v1 RUNS, AND THE OVERLAY PATH COSTS ~9.3 GB TO
            # BUILD IT. That path materializes the whole corpus TWICE in Python
            # objects (`rows`, then `merged`) before flattening — a standing cost
            # held for the entire sweep, on the same box as training. With one
            # file there is nothing to overlay, so stream straight into the flat
            # arrays. Same refusals, same output; proved equal to the overlay
            # path on the same file by
            # `test_the_streaming_single_file_load_equals_the_overlay_path`.
            return cls._load_one_streaming(paths[0], policy_encoding=policy_encoding)
        return cls._load_overlay(paths, policy_encoding=policy_encoding)

    @classmethod
    def _load_overlay(
        cls, paths: list[Path], *, policy_encoding: str | None = None,
    ) -> RvgLabelIndex:
        """Several passes, overlaid per move, weakest pass first.

        Kept reachable on ONE file (called directly) so
        ``test_the_streaming_single_file_load_equals_the_overlay_path`` can
        compare the two on identical input — a memory optimization that is not
        differentially tested is a second convention.
        """

        parsed: list[tuple[tuple[float, float, str], Path, dict[str, object],
                           dict[bytes, tuple[list[int], list[float]]]]] = []
        encodings: set[str] = set()
        for file_path in paths:
            header: dict[str, object] = {}
            rows: dict[bytes, tuple[list[int], list[float]]] = {}
            with open(file_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if rec.get("record") == "provenance":
                        header = dict(rec)
                        continue
                    version = int(rec.get("v", -1))
                    if version != RVG_LABEL_SCHEMA_VERSION:
                        raise SystemExit(
                            f"{file_path}: label schema version {version}, this "
                            f"build reads {RVG_LABEL_SCHEMA_VERSION}"
                        )
                    encodings.add(str(rec.get("policy_encoding", "")))
                    key = bytes.fromhex(str(rec["key"]))
                    mi = [int(v) for v in rec["move_index"]]
                    cp = [float(v) for v in rec["cp_eff"]]
                    if len(mi) != len(cp):
                        raise SystemExit(
                            f"{file_path}: record {rec['key']} has {len(mi)} moves "
                            f"and {len(cp)} cp values"
                        )
                    if key in rows:
                        raise SystemExit(
                            f"{file_path}: duplicate key {rec['key']} — the join "
                            "would be ambiguous"
                        )
                    rows[key] = (mi, cp)
            header.setdefault("nodes", 0)
            header.setdefault("multipv", 0)
            parsed.append(
                (cls._pass_strength(header, file_path.name), file_path, header, rows),
            )

        if len(encodings) > 1:
            raise SystemExit(f"mixed policy encodings across label files {sorted(encodings)}")
        file_encoding = next(iter(encodings)) if encodings else ""
        if policy_encoding is not None and file_encoding and file_encoding != policy_encoding:
            raise SystemExit(
                f"labels are in {file_encoding!r} policy space but the corpus "
                f"stores {policy_encoding!r} — the edits would address different "
                "moves than the ones they were computed for"
            )

        # WEAKEST FIRST, so the strongest pass overwrites. Reversing this is the
        # mutant `test_the_overlay_is_not_applied_strongest_first` kills.
        parsed.sort(key=lambda item: item[0])
        passes: list[dict[str, object]] = [
            {
                "name": file_path.name,
                "nodes": int(_header_number(header, "nodes")),
                "multipv": int(_header_number(header, "multipv")),
                "nodes_per_line": strength[0],
            }
            for strength, file_path, header, _rows in parsed
        ]

        merged: dict[bytes, dict[int, tuple[float, int]]] = {}
        for pass_id, (_strength, _file_path, _header, rows) in enumerate(parsed):
            for key, (mi, cp) in rows.items():
                slot = merged.setdefault(key, {})
                for move, value in zip(mi, cp, strict=True):
                    slot[move] = (value, pass_id)

        keys = list(merged)
        counts: list[int] = []
        idx_chunks: list[np.ndarray] = []
        cp_chunks: list[np.ndarray] = []
        reg_chunks: list[np.ndarray] = []
        pid_chunks: list[np.ndarray] = []
        for key in keys:
            slot = merged[key]
            moves = sorted(slot)
            cps = [slot[m][0] for m in moves]
            counts.append(len(moves))
            idx_chunks.append(np.asarray(moves, dtype=np.int32))
            cp_chunks.append(np.asarray(cps, dtype=np.float32))
            # `best_cp=None`: recomputed over the OVERLAID set, never inherited
            # from a single pass's header.
            reg_chunks.append(np.asarray(regrets_from_cp(cps), dtype=np.float32))
            pid_chunks.append(np.asarray([slot[m][1] for m in moves], dtype=np.int8))

        offsets = np.zeros((len(keys) + 1,), dtype=np.int64)
        if counts:
            np.cumsum(np.asarray(counts, dtype=np.int64), out=offsets[1:])
        header_out = dict(parsed[-1][2])
        header_out["policy_encoding"] = file_encoding
        header_out["label_files"] = [str(p) for _s, p, _h, _r in parsed]
        # Not a path list: SIZE AND MTIME PER FILE. A label file grows while a
        # resumed pass appends to it, so "which file" and "which bytes" are
        # different questions and the report has to answer the second one.
        header_out["label_file_provenance"] = [
            file_provenance(p) for _s, p, _h, _r in parsed
        ]
        return cls(
            offsets=offsets,
            move_index=(np.concatenate(idx_chunks) if idx_chunks
                        else np.zeros((0,), dtype=np.int32)),
            cp_eff=(np.concatenate(cp_chunks) if cp_chunks
                    else np.zeros((0,), dtype=np.float32)),
            regret_cp=(np.concatenate(reg_chunks) if reg_chunks
                       else np.zeros((0,), dtype=np.float32)),
            pass_id=(np.concatenate(pid_chunks) if pid_chunks
                     else np.zeros((0,), dtype=np.int8)),
            lookup={k: i for i, k in enumerate(keys)},
            header=header_out,
            passes=passes,
        )


class RvgExternalPolicyIndex:
    """Fingerprint -> an external per-move policy (arm B's ``q`` source).

    Records are ``{"key": <hex>, "policy": {"<uci>": prob, ...}}`` in the SAME
    key space the label pass writes, so one enumeration and one join key serve
    both. The UCI strings are in the side-to-move-canonical frame the label pass
    uses (that is what "keyed the same way" has to mean for the moves to line
    up), and they are resolved to policy indices HERE, once, against the label
    file's own move list rather than by re-deriving a board.

    ⚑ ALIGNMENT IS BY POLICY INDEX, NOT BY POSITION IN THE LIST. A row's
    external policy and its SF label list the same moves in different orders far
    more often than not; zipping them would produce a finite, normalized,
    completely wrong ``q``.

    Moves the file does not mention get ``q = 0`` on that row. With the geometric
    blend that removes them from the target (``t**(1-a) * 0**a == 0``), which is
    the honest reading for an external teacher: a move it assigns no probability
    is a move it is voting against, not a move it failed to see — unlike SF's
    MultiPV truncation, where absence really is absence. Stated here because the
    two sources genuinely differ on this point.
    """

    def __init__(
        self, rows: dict[bytes, dict[int, float]], header: dict[str, object],
    ) -> None:
        self._rows = rows
        self.header = header

    def __len__(self) -> int:
        return len(self._rows)

    def weights_for(self, key: bytes, move_index: np.ndarray) -> np.ndarray | None:
        """``q`` aligned with ``move_index``, or ``None`` if the row is absent."""
        row = self._rows.get(key)
        if row is None:
            return None
        return np.asarray(
            [row.get(int(m), 0.0) for m in np.asarray(move_index).tolist()],
            dtype=np.float64,
        )

    @classmethod
    def load(
        cls, path: str | Path, *, policy_encoding: str | None = None,
    ) -> RvgExternalPolicyIndex:
        from chess_anti_engine.moves.encode import (
            POLICY_ENCODING_LC0_1858,
            uci_to_policy_index,
        )
        from chess_anti_engine.moves.encode import (
            compact_policy_index,
        )

        encoding = policy_encoding or POLICY_ENCODING_LC0_1858
        if encoding != POLICY_ENCODING_LC0_1858:
            raise SystemExit(
                f"external policy source: only {POLICY_ENCODING_LC0_1858!r} is "
                f"supported, got {encoding!r}"
            )
        header: dict[str, object] = {}
        rows: dict[bytes, dict[int, float]] = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("record") == "provenance":
                    header = dict(rec)
                    continue
                key = bytes.fromhex(str(rec["key"]))
                if key in rows:
                    raise SystemExit(
                        f"{path}: duplicate key {rec['key']} — the join would be "
                        "ambiguous"
                    )
                per_move: dict[int, float] = {}
                for uci, prob in dict(rec["policy"]).items():
                    # `True`: the canonical frame is always white-to-move, the
                    # same assumption `audit.legal_full_indices` documents.
                    full = uci_to_policy_index(str(uci), True)
                    if full < 0:
                        continue
                    slot = int(compact_policy_index(full))
                    if slot < 0:
                        continue
                    # A move appearing twice (two spellings of one slot) sums:
                    # it is probability mass, and mass adds.
                    per_move[slot] = per_move.get(slot, 0.0) + float(prob)
                rows[key] = per_move
        # ⚑ THE VERSION IS REQUIRED, AND ITS ABSENCE IS A REFUSAL. This file is
        # produced by a DIFFERENT program on a DIFFERENT branch; "no version
        # declared" means the two sides have never agreed on what a record means,
        # which is exactly when silently consuming it is worst. Declared on the
        # provenance line (a per-row version on a policy file is pure bulk):
        #   {"record": "provenance", "v": 1, "name": "...", "policy_encoding": ...}
        declared_v = header.get("v")
        if declared_v is None:
            raise SystemExit(
                f"{path}: external policy file declares no schema version. Add a "
                '\'{"record": "provenance", "v": '
                f'{RVG_EXTERNAL_POLICY_SCHEMA_VERSION}, ...}}\' line — an '
                "undeclared format is one the rig cannot promise it reads the "
                "way the producer wrote it."
            )
        version = _header_number({"v": declared_v}, "v")
        if int(version) != RVG_EXTERNAL_POLICY_SCHEMA_VERSION:
            raise SystemExit(
                f"{path}: external policy schema version {int(version)}, this "
                f"build reads {RVG_EXTERNAL_POLICY_SCHEMA_VERSION}"
            )
        # And the file's OWN declared policy space, checked against the corpus's
        # — the `encoding` check above only refuses a corpus this loader cannot
        # serve, which is a different question and would pass a file written in
        # `az_4672`.
        declared_encoding = str(header.get("policy_encoding", "") or "")
        if declared_encoding and declared_encoding != encoding:
            raise SystemExit(
                f"{path}: external policy is in {declared_encoding!r} policy "
                f"space but the corpus stores {encoding!r} — its probabilities "
                "would be applied to different moves than they were computed for"
            )
        header.setdefault("policy_encoding", encoding)
        header["file_provenance"] = file_provenance(path)
        return cls(rows, header)
