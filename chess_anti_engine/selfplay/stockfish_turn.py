"""Stockfish annotation + curriculum-opponent move selection.

* ``_eff_sf_nodes`` — adaptive per-slot SF node budget (scales down when
  the previous net decision used fast sims).
* ``submit_sf_queries`` — dispatch async queries to ``StockfishPool``
  (no-op for synchronous stockfish).
* ``finish_sf_annotation_and_moves`` — resolve futures (or run sync),
  then ``_process_sf_results`` per slot: build the softmax SF policy
  target, attach (with per-head legal mask) to the last ``_NetRecord``
  for that slot, and play the curriculum opponent's move for non-
  selfplay games.

All entry points expect slot-index iterables disjoint from the network-
turn indices — the driver in ``manager.play_batch`` enforces this
partitioning via ``classify_active_slots``.
"""

from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass
from typing import Any

import numpy as np

import chess

from chess_anti_engine.encoding.cboard_encode import encode_cboard
from chess_anti_engine.moves import POLICY_SIZE
from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.selfplay.state import SelfplayState, _NetRecord
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.stockfish.wdl import cp_to_wdl as _cp_to_wdl
from chess_anti_engine.replay.shard import SF_CP_SENTINEL, SF_MULTIPV_RAW_MAX


from chess_anti_engine.utils.numpy_helpers import softmax_1d as _softmax_np


_LOG = logging.getLogger("chess_anti_engine.selfplay")

_multipv_truncation_warned = False

# One health line per this many REPORTER CALLS. Only label events reach the
# reporter: a curriculum MOVE query produces no training row, so including it
# would measure something the line is not named after. ~17 labels/s live, so
# roughly one line every four minutes.
#
# ⚑ THE WINDOW COUNTS CALLS; THE RATES DIVIDE BY `labelled`, AND THE TWO ARE
# NOT THE SAME NUMBER. Until 2026-08-03 they were: every call incremented
# `labelled`, including the two failure sites that label no row at all
# (audit P9), so `no_legal_pv/labelled` and `bestmove_illegal/labelled` were
# diluted by the failure count — the third term of the very rule they feed.
# The window still advances on failures, deliberately: a window that is ALL
# failures must still emit a line, and pinning it to `labelled` would make a
# total label outage silent.
_SF_LABEL_REPORT_EVERY = 4096

# THE detector. Fraction of labelled rows on which not ONE of Stockfish's
# MultiPV moves was legal at the position we queried, i.e. the stored
# `sf_multipv_raw` came out empty. Calibrated against the four episodes visible
# in the then-live window (2026-07-30/31), 721 shards / 1,280,663 labelled rows:
#
#     no-PV rate   "clean" 0.0008 | ep1 0.1241  ep2 0.3275  ep3 0.2541  ep4 0.5684
#
# One desynced engine out of `distributed_worker_sf_workers` (8 live) lands near
# 0.074, i.e. a **7.4x** margin over the 0.01 threshold, and the episodes above
# clear it by 12-57x.
#
# ⚑ The "~100x separation / 90x margin" this comment used to claim were both
# computed against the 0.0008 "clean" figure that correction 2 below repudiates
# (0.074 / 0.0008 ~= 92). Deleting the baseline while keeping the margins derived
# from it would leave a calibration comment stating two mutually inconsistent
# numbers, with the wrong one read first. Against the TRUE floor of exactly
# 0.000000 no ratio is defined at all, which is why the margin is now quoted
# against the threshold rather than against the baseline.
#
# ⚑ TWO CORRECTIONS to the calibration above (2026-08-01). Neither changes the
# threshold; both change what a reading MEANS.
#
# 1. "the four known episodes" was the visible window, not the history. The bug
#    (`uci.py` whole-search deadline, fixed in PR #297) dates to at least 07-05:
#    across 6,535 unique historical shards / 11,052,418 labelled rows, 07-05
#    reads 0.084, 07-13 0.153, 07-14 0.120. 07-30/31 was at least the THIRD
#    occurrence. Aggregate contamination over retained history is 1.76%.
#
# 2. "clean 0.0008" is NOT the structural floor — that baseline was itself
#    measured on shards carrying residual contamination. The structural rate is
#    EXACTLY 0.000000: 90.48% of all historical shards, and 641 of the 713
#    shards in the post-quarantine window, read exactly zero. Never treat a
#    small non-zero reading as "the normal background".
#
# ⚑⚑ WHAT THIS THRESHOLD CANNOT SEE, BY CONSTRUCTION. 0.01 is a *desynced
# engine* detector, not a contamination detector. The 2026-08-01 quarantine used
# the same 0.01 bar and, having removed 122 shards, left 72 shards holding 252
# no-PV rows — every one of them below the bar. Those 72 are burst EDGES, not a
# floor: median distance to the nearest quarantined shard is 14 ids against 43.5
# +/- 6.6 for a uniform draw of the same 72 over the same range (p < 1e-3, 2000
# draws), and 55 of them sit in two runs butting against quarantined blocks.
#
# DECISION (not deferred): the threshold STAYS at 0.01. Tightening it to ~0.002
# would catch those edges, but they are 252 rows in 1,264,058 — 0.02% of the
# window, orders of magnitude below anything that moves a training target — while
# the failure this detector exists for (k=1 desynced engine, 0.074) clears either
# bar by 7-37x. Buying a 0.02% cleanup with a 5x cut in alarm margin is a bad
# trade. The blind spot is documented instead: a PASS here means "no engine is
# detached", and must never be read as "this data is uncontaminated".
_SF_NO_LEGAL_PV_WARN_RATE = 0.01

# NOT the trigger, reported for context only. `bestmove_illegal` is floored by
# structure: illegal ~= 0.079 + (k/8)*0.82 for k desynced engines, so k=1 reads
# 0.182 and k=2 reads 0.284. Any threshold that clears the 0.079 baseline is a
# "two or more engines" detector, and episode 1's window mean was 0.241 — it
# would have read HEALTHY through the largest episode. It also undercounts
# detachment ~10%, because a stale bestmove is sometimes coincidentally legal
# in the position it was misfiled against.
_SF_BESTMOVE_ILLEGAL_CONTEXT_RATE = 0.25

# THE OTHER HALF OF THE LABEL. `no_legal_pv` above reads the POLICY block; this
# reads the record-level SF eval that becomes `rec.sf_wdl` — realized
# `sf_wdl_frac` 0.45 of the trained value target, and until 2026-08-03 carrying
# no detector in either direction (SELFPLAY_AUDIT P2). It fires when that eval's
# own PV was dropped as illegal, leaving the score attached to a move that is
# not playable in the queried position. Definition and floor:
# `replay/shard.py::sf_eval_pv_orphan_flags`.
#
# ⚑ THE TWO POPULATIONS ARE DISJOINT. `no_legal_pv` counts rows with NO MultiPV
# block; this is only computable on rows that HAVE one. So this rate inspects
# exactly the set the policy detector passes, and the two must never be summed
# onto a shared denominator.
#
# THE BAR IS 0.01, THE SAME AS THE POLICY HALF, AND THAT IS A CHOICE. One bar
# keeps the two halves of one label directly comparable on one instrument.
# Calibration on the live 4096-label window (2026-08-03, numbers and provenance
# in docs/experiment_ledger.md):
#
#   policy-clean post-quarantine, 275 windows : max 0.001465, 257 read EXACTLY 0
#                                               -> 6.8x headroom under the bar
#   122 quarantined shards, 51 windows        : median 0.102205, mean 0.143583
#                                               -> 49 of 51 windows fire
#
# ⚑ 2 of those 51 windows read BELOW the bar. This is a SECOND view, not a
# superset of the policy one: a desynced search whose rank-1 move is
# coincidentally legal here, or whose (cp, mate) ties the top surviving line,
# passes.
# Combined with `no_legal_pv` it flags 62,881 of the 209,259 labelled rows on
# the quarantined set — 0.300494 against 0.207461 for the policy half alone,
# i.e. +44.8% more rows detected, and ~85% of the ~0.352 true poisoned share
# the module's own ~0.59 pass-through calibration implies, against ~59% before.
# The two counts are DISJOINT by construction and observed disjoint (0 overlap),
# which is what licenses adding them.
_SF_EVAL_PV_ORPHAN_WARN_RATE = 0.01

_sf_label_lock = threading.Lock()
_sf_label_counts = {
    "calls": 0, "labelled": 0, "failed": 0, "bestmove_illegal": 0,
    "no_legal_pv": 0, "eval_pv_orphan": 0, "eval_pv_checked": 0,
    "wdl_degenerate": 0, "wdl_orphaned": 0,
}


def _sf_wdl_is_wellformed(wdl: np.ndarray | None) -> bool:
    """Is an attached ``sf_wdl`` a usable (W, D, L) distribution?

    Deliberately the same predicate as ``train/losses.py::sf_wdl_wellformed``,
    which is the shard-side twin. It is stated twice because the two run on
    different objects (a live record's array vs a collated batch tensor) and a
    shared helper would have to launder one into the other; the tolerances are
    pinned equal by test, not by hope.
    """
    if wdl is None:
        return False
    v = np.asarray(wdl, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        return False
    if (v < -1e-6).any() or (v > 1.0 + 1e-6).any():
        return False
    if abs(float(v.sum()) - 1.0) > 1e-3:
        return False
    return not bool((np.abs(v - 1.0 / 3.0) < 1e-4).all())


def _sf_eval_pv_orphaned(rec) -> bool:
    """Does the record's stored SF eval disagree with its stored top PV?

    Reads the STORED ``sf_label_meta`` / ``sf_multipv_raw``, never a
    recomputation — the same rule ``no_legal_pv`` follows, so this live counter
    and ``TrainMetrics.sf_eval_pv_orphan_frac`` are one measurement rather than
    two descriptions that can drift. False when either block is absent: an
    unmeasurable row is not a clean one, and the count of measurable rows is
    reported next to the rate for exactly that reason.
    """
    raw = getattr(rec, "sf_multipv_raw", None)
    meta = getattr(rec, "sf_label_meta", None)
    if raw is None or meta is None:
        return False
    raw_arr = np.asarray(raw)
    meta_arr = np.asarray(meta)
    if raw_arr.ndim != 2 or raw_arr.shape[0] < 1 or raw_arr.shape[1] < 3:
        return False
    if meta_arr.ndim != 1 or meta_arr.shape[0] < 4 or int(raw_arr[0, 0]) < 0:
        return False
    return (
        int(raw_arr[0, 1]) != int(meta_arr[2])
        or int(raw_arr[0, 2]) != int(meta_arr[3])
    )


def _sf_value_half_counts(rec) -> dict[str, int]:
    """The four VALUE-half arguments to ``_report_sf_label_health`` for *rec*.

    One place, so the two label call sites cannot instrument the value half
    differently — which is how the POLICY half ended up with a live counter and
    the value half with nothing (SELFPLAY_AUDIT P2).

    ``eval_pv_checked`` is the count of rows the orphan comparison is DEFINED
    on, and it is 1 only when both stored blocks are present. It is reported
    rather than assumed for the reason the module keeps re-learning: a rate
    over an unmeasured population reads exactly like a clean one.
    """
    wdl = getattr(rec, "sf_wdl", None)
    wellformed = _sf_wdl_is_wellformed(wdl)
    checked = (
        getattr(rec, "sf_multipv_raw", None) is not None
        and getattr(rec, "sf_label_meta", None) is not None
    )
    return {
        "eval_pv_orphan": int(_sf_eval_pv_orphaned(rec)),
        "eval_pv_checked": int(checked),
        "wdl_degenerate": int(wdl is not None and not wellformed),
  # The blind spot itself: the POLICY block is gone and a well-formed value
  # label rode through anyway, at 0.45 of the value target with nothing
  # marking it.
        "wdl_orphaned": int(
            getattr(rec, "sf_multipv_raw", None) is None and wellformed,
        ),
    }


def _report_sf_label_health(
    *, failed: int = 0, bestmove_illegal: int = 0, no_legal_pv: int = 0,
    eval_pv_orphan: int = 0, eval_pv_checked: int = 0,
    wdl_degenerate: int = 0, wdl_orphaned: int = 0,
) -> None:
    """Accumulate label-path health and emit one line per report window.

    Exists because the 2026-07-31 corruption had NO observable: a Stockfish
    process one search behind returns a real search of another position, and
    every identity the caller controls (``sf_legal_mask``, the record, the turn)
    stays correct, so the row binding looks perfect. Three separate silent
    substitutions then absorb the damage — ``_process_sf_label_result_for_record``
    swaps in ``legal_indices[0]`` for the illegal bestmove, ``_collect_sparse_pv_rows``
    and ``_collect_sf_pv_candidates`` drop every illegal PV move, and the caller
    fabricates a degenerate one-hot when nothing survives. A poisoned row
    therefore carries a fake bestmove AND a fake policy distribution.

    ``no_legal_pv`` is read from the STORED ``sf_multipv_raw``, so this live
    counter and the offline shard check are the same measurement by
    construction rather than by agreement between two descriptions of one.
    ``eval_pv_orphan`` follows the same rule against the STORED
    ``sf_label_meta`` / ``sf_multipv_raw`` pair.

    ``failed`` participates in the escalation deliberately: post-fix an
    abandoned search is the incident itself, and the whole point of the repair
    is that its firing is visible rather than inferred.

    ⚑ ``failed`` IS NOT A LABEL, AND SINCE 2026-08-03 IT IS NOT COUNTED AS ONE
    (audit P9). Both failure sites call this with ``failed=1`` on a path where
    no row was labelled; incrementing ``labelled`` there put the failure count
    into the denominator of ``no_legal_pv/labelled`` and
    ``bestmove_illegal/labelled``, damping both by ``F/(L+F)``. The report
    WINDOW still advances on every call (``calls``), so a window of pure
    failures still emits a line instead of going silent.

    The escalation VERDICT is unchanged by that fix and cannot have changed:
    the only windows where the two denominators differ are windows with
    ``failed > 0``, and ``failed > 0`` is itself an unconditional term of
    ``unhealthy``. So every window whose rates the fix moves was already
    WARNING under both denominators. What changes is the number a human reads,
    which is now the rate over rows that actually got a label. No threshold was
    retuned; the effective stringency is identical.

    ``eval_pv_orphan`` / ``wdl_degenerate`` / ``wdl_orphaned`` are the VALUE
    half (P2). Only the first escalates — see ``_SF_EVAL_PV_ORPHAN_WARN_RATE``
    for its bar and for the two things it cannot see. ``wdl_degenerate`` reads
    exactly 0 on known-poisoned data and is a producer tripwire, not a desync
    one; ``wdl_orphaned`` is ``no_legal_pv``'s twin by design and counts the
    blind spot rather than detecting anything new. Neither is worth an alarm on
    its own and neither is in ``unhealthy``.
    """
    with _sf_label_lock:
        _sf_label_counts["calls"] += 1
  # A failure labels no row, so it is not in the label denominator. Guarded on
  # `failed` rather than on the caller's intent because the two failure sites
  # pass nothing else -- if a future site passes both, the row is a failure.
        if not failed:
            _sf_label_counts["labelled"] += 1
        _sf_label_counts["failed"] += int(failed)
        _sf_label_counts["bestmove_illegal"] += int(bestmove_illegal)
        _sf_label_counts["no_legal_pv"] += int(no_legal_pv)
        _sf_label_counts["eval_pv_orphan"] += int(eval_pv_orphan)
        _sf_label_counts["eval_pv_checked"] += int(eval_pv_checked)
        _sf_label_counts["wdl_degenerate"] += int(wdl_degenerate)
        _sf_label_counts["wdl_orphaned"] += int(wdl_orphaned)
        if _sf_label_counts["calls"] % _SF_LABEL_REPORT_EVERY:
            return
        counts = dict(_sf_label_counts)
        _sf_label_counts.update(
            calls=0, labelled=0, failed=0, bestmove_illegal=0, no_legal_pv=0,
            eval_pv_orphan=0, eval_pv_checked=0, wdl_degenerate=0,
            wdl_orphaned=0,
        )
    labelled = max(1, counts["labelled"])
    no_pv_rate = counts["no_legal_pv"] / labelled
    illegal_rate = counts["bestmove_illegal"] / labelled
  # Its OWN denominator: rows where BOTH blocks were stored, which is the only
  # population the comparison is defined on. Dividing by `labelled` would let
  # the no-PV rows -- which by construction can never be orphans -- dilute the
  # rate by exactly the amount the policy detector already reported.
    eval_pv_checked = max(1, counts["eval_pv_checked"])
    orphan_rate = counts["eval_pv_orphan"] / eval_pv_checked
    unhealthy = (
        no_pv_rate > _SF_NO_LEGAL_PV_WARN_RATE
        or orphan_rate > _SF_EVAL_PV_ORPHAN_WARN_RATE
        or counts["failed"] > 0
        or illegal_rate > _SF_BESTMOVE_ILLEGAL_CONTEXT_RATE
    )
    _LOG.log(
        logging.WARNING if unhealthy else logging.INFO,
        "sf label health: calls=%d labelled=%d no_legal_pv=%d (%.4f) "
        "bestmove_illegal=%d (%.3f) failed=%d eval_pv_orphan=%d (%.4f of %d "
        "checked) wdl_degenerate=%d wdl_orphaned=%d — no_legal_pv above %.3f "
        "means Stockfish is answering a DIFFERENT position than the one "
        "queried (a desynced engine), so the whole SF label block on those "
        "rows is detached from its row; eval_pv_orphan above %.3f says the "
        "same thing about the VALUE half (sf_wdl, 0.45 of the value target) "
        "on rows the no_legal_pv check PASSES",
        counts["calls"], counts["labelled"], counts["no_legal_pv"], no_pv_rate,
        counts["bestmove_illegal"], illegal_rate, counts["failed"],
        counts["eval_pv_orphan"], orphan_rate, counts["eval_pv_checked"],
        counts["wdl_degenerate"], counts["wdl_orphaned"],
        _SF_NO_LEGAL_PV_WARN_RATE, _SF_EVAL_PV_ORPHAN_WARN_RATE,
    )


def _warn_multipv_truncated() -> None:
    """Warn once: sf_multipv > SF_MULTIPV_RAW_MAX breaks the rebuild parity contract.

    The live target uses ALL legal MultiPV candidates while the stored sparse
    rows are capped, so train-time rebuilds (train/target_builder.py) would
    diverge from the stored dense targets for these records.
    """
    global _multipv_truncation_warned
    if not _multipv_truncation_warned:
        _multipv_truncation_warned = True
        _LOG.warning(
            "sparse MultiPV capture truncated at %d rows; live targets use all "
            "candidates, so rebuilt SF targets will diverge from stored ones. "
            "Raise SF_MULTIPV_RAW_MAX (replay/shard.py) or lower sf_multipv.",
            SF_MULTIPV_RAW_MAX,
        )


@dataclass
class _PendingSfLabel:
    future: Any
    record: _NetRecord
    turn: bool
    legal_indices: np.ndarray
    # Escalation context (sf_label_escalate_*, see
    # _maybe_submit_label_escalation): the query position, slot, and syzygy
    # path captured at SUBMIT time — by poll time the board has advanced, so
    # they cannot be recovered from state. Defaults keep legacy constructions
    # valid; an empty query_fen disables escalation for the entry.
    query_fen: str = ""
    slot: int = -1
    syzygy_path: str | None = None
    # Set once the escalated re-query replaced ``future``: the ORIGINAL
    # (shallow) StockfishResult, attached later as ``rec.sf_wdl_original``
    # (and the airbag fallback if the deep re-query itself fails).
    escalated_from_res: Any = None


def flip_wdl_pov(wdl: np.ndarray) -> np.ndarray:
    """Flip a ``[W, D, L]`` vector to the opposite POV (swaps W and L)."""
    wdl = np.asarray(wdl, dtype=np.float32)
    if wdl.shape != (3,):
        return wdl.astype(np.float32, copy=False)
    return np.array(
        [float(wdl[2]), float(wdl[1]), float(wdl[0])], dtype=np.float32,
    )


def _sf_result_wdl_for_record(
    res,
    *,
    sf_wdl_use_cp_logistic: bool,
    sf_wdl_cp_slope: float,
    sf_wdl_cp_draw_width: float,
) -> np.ndarray | None:
    """Return SF WDL in the sampled net-record POV.

    SF label searches are run after the network move has been pushed, both for
    curriculum opponent replies and for async selfplay annotation. Therefore
    Stockfish's WDL is from the opponent side-to-move and must be flipped back
    before attaching it to the previous network-turn record.
    """
    if sf_wdl_use_cp_logistic and (res.cp is not None or res.mate is not None):
        wdl_stm = _cp_to_wdl(
            res.cp,
            res.mate,
            slope=sf_wdl_cp_slope,
            draw_width_cp=sf_wdl_cp_draw_width,
        )
        return flip_wdl_pov(wdl_stm)
    if res.wdl is not None:
        return flip_wdl_pov(res.wdl)
    return None


def _choose_curriculum_opponent_move(
    *,
    rng: np.random.Generator,
    legal_indices: np.ndarray,
    cand_indices: list[int],
    cand_scores: list[float],
    regret_limit: float,
) -> int:
    """Pick a curriculum-opponent move index from SF's candidate list.

    * Empty candidate list -> uniform random legal move.
    * ``regret_limit == inf`` -> take SF's top choice verbatim (used by
      eval/gate matches where we want full-strength SF).
    * Otherwise -> uniform random among moves within ``regret_limit``
      score of the best candidate.
    """
    if not cand_indices:
        return int(legal_indices[int(rng.integers(len(legal_indices)))])

    if not math.isfinite(float(regret_limit)):
        # MultiPV lists PVs in rank order so cand_indices[0] is SF's best.
        return cand_indices[0]

    best_score = max(float(s) for s in cand_scores)
    acceptable = [
        idx
        for idx, score in zip(cand_indices, cand_scores, strict=False)
        if (best_score - float(score)) <= float(regret_limit) + 1e-12
    ]
    if not acceptable:
        acceptable = [cand_indices[0]]
    return acceptable[int(rng.integers(len(acceptable)))]


def _sf_played_move_diagnostics(
    played_idx: int,
    cand_idxs: list[int],
    cand_scores: list[float],
) -> tuple[int | None, float | None]:
    if not cand_idxs or not cand_scores:
        return None, None
    played_score = None
    for idx, score in zip(cand_idxs, cand_scores, strict=False):
        if int(idx) == int(played_idx):
            played_score = float(score)
            break
    if played_score is None:
        # Regret is measured only against the sampled MultiPV set; outside-set moves are missing diagnostics.
        return None, None
    best_score = max(float(score) for score in cand_scores)
    rank = 1 + sum(1 for score in cand_scores if float(score) > played_score + 1e-12)
    return int(rank), max(0.0, best_score - played_score)


def _slot_in_sf_refute(state: SelfplayState, idx: int) -> bool:
    """True while slot ``idx`` is still in its SF-refute opponent phase.

    ``sf_refute_opp_plies_left[idx] > 0`` marks the plies where SF (not the net)
    plays the opponent seat of a selfplay-tagged refute game. getattr: unit-test
    SimpleNamespace fixtures may omit the array (≡ zeros).
    """
    refute_left = getattr(state, "sf_refute_opp_plies_left", None)
    return (
        refute_left is not None
        and idx < len(refute_left)
        and int(refute_left[idx]) > 0
    )


def _eff_sf_nodes(
    state: SelfplayState, idx: int, *, for_move: bool = False, for_label: bool = False,
) -> int | None:
    """Return the per-slot SF node budget, scaled down on fast-sim plies.

    On fast (playout-capped) plies the SF budget is scaled by
    ``game.sf_fast_ply_node_scale`` (full-sim plies always use 1.0). This is an
    intentional compute optimization, NOT a weakened training target: SF labels
    only attach to full plies (``has_policy`` <=> ``is_full`` <=>
    ``last_net_full``; see network_turn.py and the label-reuse guard in
    ``_slot_latest_record_needs_sf_label``), so every label already runs at full
    nodes. The scale only makes the opponent play cheaply on the ~75% of fast
    plies that are not training targets. Default 0.25 (see GameConfig); raise
    toward 1.0 only for a more consistently strong opponent at extra SF cost.

    ``for_label`` marks label-only queries (selfplay P1 analysis — never a move
    the opponent plays). When ``game.sf_label_nodes_cap`` > 0 those are capped
    at that budget, decoupling label cost from the PID-ramped opponent budget.
    When ``game.sf_label_nodes_floor`` > 0 they are also raised to at least
    that budget — the floor decouples label QUALITY upward the same way the cap
    decouples cost downward. Without it the teacher silently rides the PID
    difficulty knob: the 2026-08-04 fresh restart began at sf_nodes 50k, an 11x
    teacher cut versus the old lineage's realized ~698k, and nothing said so.
    The floor is applied after the cap, so on a conflicting config the floor
    wins (GameConfig.__post_init__ warns). Curriculum move queries
    (``for_move=True``) get neither.
    """
    if for_move and for_label:
        raise ValueError("_eff_sf_nodes: a query cannot be both a move and a label")
    base_nodes = int(state.base_nodes)
    if for_move:
        move_nodes = int(getattr(state.game, "sf_move_nodes", 0) or 0)
        if move_nodes > 0:
            base_nodes = move_nodes
    elif for_label:
        label_cap = int(getattr(state.game, "sf_label_nodes_cap", 0) or 0)
        if label_cap > 0:
            base_nodes = min(base_nodes, label_cap)
        label_floor = int(getattr(state.game, "sf_label_nodes_floor", 0) or 0)
        if label_floor > 0:
            base_nodes = max(base_nodes, label_floor)
    if base_nodes <= 0:
        return None
    # SF-refute MOVE queries can opt out of the fast-ply scale so the punishing
    # move is searched at full strength even on fast plies (sf_refute_full_node_
    # moves). Only for_move queries in an active refute phase; labels and
    # ordinary curriculum/selfplay behaviour are untouched (flag default off).
    refute_full_move = (
        for_move
        and _slot_in_sf_refute(state, idx)
        and bool(getattr(
            getattr(state, "opening", None), "sf_refute_full_node_moves", False,
        ))
    )
    if bool(state.last_net_full[idx]) or refute_full_move:
        fast_scale = 1.0
    else:
        fast_scale = float(getattr(state.game, "sf_fast_ply_node_scale", 0.25))
    return max(1, round(float(base_nodes) * fast_scale))


def _sf_syzygy_path_for_slot(state: SelfplayState, idx: int) -> str | None:
    """Pick the low-IO or DTZ-capable Stockfish tablebase path for one root.

    Most SF calls are high-volume curriculum labeling where SSD-only WDL/DTZ is
    enough. For the explicit non-adjudicated play-through tail, switch that SF
    process to the full tablebase path once the root itself is <=6 pieces so
    Stockfish can choose conversion-correct DTZ moves.
    """
    normal_path = state.game.stockfish_syzygy_path or state.game.syzygy_path
    full_path = state.game.syzygy_path
    if not full_path or str(normal_path or "") == str(full_path):
        return normal_path
    cb = state.cboards[idx]
    occ = int(cb.occ_white) | int(cb.occ_black)
    if occ.bit_count() > 6 or int(cb.castling) != 0:
        return normal_path
    playthrough = (not state.game.syzygy_adjudicate) or (not bool(state.tb_adj_roll_arr[idx]))
    return full_path if playthrough else normal_path


def _search_stockfish_sync(
    stockfish: Any,
    fen: str,
    *,
    nodes: int | None,
    syzygy_path: str | None,
) -> Any:
    if syzygy_path is None:
        return stockfish.search(fen, nodes=nodes)
    try:
        return stockfish.search(fen, nodes=nodes, syzygy_path=syzygy_path)
    except TypeError as exc:
        if "syzygy_path" not in str(exc):
            raise
        return stockfish.search(fen, nodes=nodes)


def _slot_latest_record_needs_sf_label(state: SelfplayState, idx: int) -> bool:
    if not state.samples_per_game[idx]:
        return False
    rec = state.samples_per_game[idx][-1]
    if not bool(rec.has_policy):
        return False
    # SF-refute opp rows carry has_policy=True (a MAIN policy target) but are the
    # SF-to-move position, not a net turn — they must never receive a P1 reply
    # label. Skip them so a stray label query can't mis-attach.
    if bool(getattr(rec, "is_sf_refute_opp", False)):
        return False
    return rec.sf_policy_target is None and rec.sf_move_index is None


def submit_sf_queries(
    state: SelfplayState, idxs: list[int], *, for_move: bool = False,
    for_label: bool = False,
) -> dict[int, Any]:
    """Submit SF queries to the pool without blocking; return futures dict.

    Only valid when ``state.stockfish`` is a ``StockfishPool``.  The
    caller guards with ``isinstance`` (same pattern as the original
    nested closure). Pass ``for_label=True`` for label-only queries so
    ``sf_label_nodes_cap`` applies — omitting it silently runs the full
    (PID-ramped) opponent budget.
    """
    assert isinstance(state.stockfish, StockfishPool)
    return {
        idx: state.stockfish.submit(
            state.cboards[idx].fen(),
            nodes=_eff_sf_nodes(state, idx, for_move=for_move, for_label=for_label),
            syzygy_path=_sf_syzygy_path_for_slot(state, idx),
        )
        for idx in idxs
    }


def submit_async_curriculum_move_queries(state: SelfplayState, idxs: list[int]) -> int:
    """Submit curriculum SF move queries and leave them pending per slot."""
    if not idxs:
        return 0
    assert isinstance(state.stockfish, StockfishPool)
    submitted = 0
    for idx in idxs:
        if idx in state.pending_sf_moves:
            continue
        state.pending_sf_moves[idx] = state.stockfish.submit(
            state.cboards[idx].fen(),
            nodes=_eff_sf_nodes(state, idx, for_move=True),
            syzygy_path=_sf_syzygy_path_for_slot(state, idx),
        )
        submitted += 1
    return submitted


def submit_async_sf_labels_from_curriculum_moves(state: SelfplayState, idxs: list[int]) -> int:
    """Reuse full-strength curriculum move futures as labels when possible."""
    if int(getattr(state.game, "sf_move_nodes", 0) or 0) > 0:
        return submit_async_sf_label_queries(state, idxs)
    submitted = 0
    max_pending = max(1, int(state.batch_size) * 8)
    for idx in idxs:
        if len(state.pending_sf_labels) >= max_pending:
            break
        fut = state.pending_sf_moves.get(idx)
        if fut is None or not _slot_latest_record_needs_sf_label(state, idx):
            continue
        legal_indices = state.cboards[idx].legal_move_indices()
        if legal_indices.size == 0:
            continue
        # The reused move future was submitted for the CURRENT position (the
        # board doesn't advance until finish_pending_curriculum_moves pushes
        # the reply), so capturing the escalation context from the board here
        # matches the query — the same assumption the turn/legal_indices
        # snapshot above already makes.
        state.pending_sf_labels.append(
            _PendingSfLabel(
                future=fut,
                record=state.samples_per_game[idx][-1],
                turn=bool(state.cboards[idx].turn),
                legal_indices=np.asarray(legal_indices, dtype=np.int64).copy(),
                query_fen=state.cboards[idx].fen(),
                slot=int(idx),
                syzygy_path=_sf_syzygy_path_for_slot(state, idx),
            ),
        )
        submitted += 1
    return submitted


def finish_pending_curriculum_moves(
    state: SelfplayState, *, block: bool = False,
    block_timeout_s: float | None = None,
) -> int:
    """Apply completed pending curriculum moves.

    If ``block`` is true and no move is ready, wait only until the first
    Stockfish move completes, or for ``block_timeout_s`` when supplied. This
    avoids per-step head-of-line blocking while still making progress when all
    runnable slots are waiting, while a bounded wait lets stop/pause/model
    checks regain control promptly.
    """
    if not state.pending_sf_moves:
        return 0
    if block and not any(fut.done() for fut in state.pending_sf_moves.values()):
        wait(
            tuple(state.pending_sf_moves.values()),
            timeout=block_timeout_s,
            return_when=FIRST_COMPLETED,
        )

    ready = [(idx, fut) for idx, fut in state.pending_sf_moves.items() if fut.done()]
    completed: list[tuple[int, Any]] = []
    for idx, fut in ready:
        completed.append((idx, fut.result()))
        del state.pending_sf_moves[idx]

    for idx, res in completed:
        if state.finalized_arr[idx] or state.done_arr[idx]:
            continue
        _process_sf_results(
            state,
            [idx],
            results={idx: res},
            play_curriculum_moves=True,
            attach_labels=False,
        )
    return len(completed)


def finish_sf_annotation_and_moves(
    state: SelfplayState,
    idxs: list[int],
    *,
    play_curriculum_moves: bool,
    attach_labels: bool = True,
    for_move: bool = False,
    futures: dict[int, Any] | None = None,
) -> None:
    """Collect SF results (from futures or synchronously), then process."""
    if not idxs:
        return
    label_pass = attach_labels and not play_curriculum_moves
    label_only = label_pass and not for_move
    if label_pass:
        idxs = [idx for idx in idxs if _slot_latest_record_needs_sf_label(state, idx)]
        if not idxs:
            return
    if futures is not None:
        results = {idx: futures[idx].result() for idx in idxs if idx in futures}
    elif isinstance(state.stockfish, StockfishPool):
        futs = {
            idx: state.stockfish.submit(
                state.cboards[idx].fen(),
                nodes=_eff_sf_nodes(state, idx, for_move=for_move, for_label=label_only),
                syzygy_path=_sf_syzygy_path_for_slot(state, idx),
            )
            for idx in idxs
        }
        results = {idx: fut.result() for idx, fut in futs.items()}
    else:
        results = {
            idx: _search_stockfish_sync(
                state.stockfish,
                state.cboards[idx].fen(),
                nodes=_eff_sf_nodes(state, idx, for_move=for_move, for_label=label_only),
                syzygy_path=_sf_syzygy_path_for_slot(state, idx),
            )
            for idx in idxs
        }
    _process_sf_results(
        state, idxs, results=results, play_curriculum_moves=play_curriculum_moves,
        attach_labels=attach_labels,
    )


def _pv_wdl_score(
    pv,
    *,
    sf_wdl_use_cp_logistic: bool,
    sf_wdl_cp_slope: float,
    sf_wdl_cp_draw_width: float,
) -> float | None:
    if sf_wdl_use_cp_logistic and (pv.cp is not None or pv.mate is not None):
        wdl = _cp_to_wdl(
            pv.cp,
            pv.mate,
            slope=sf_wdl_cp_slope,
            draw_width_cp=sf_wdl_cp_draw_width,
        )
    else:
        wdl = pv.wdl
    if wdl is None:
        return None
    w_sf, d_sf = float(wdl[0]), float(wdl[1])
    return w_sf + 0.5 * d_sf


def _collect_sparse_pv_rows(res, *, turn: bool, legal_set: set[int]) -> np.ndarray | None:
    """Raw MultiPV rows (K, 5) int16 in FULL policy space, rank order.

    Same legality filter + ordering as ``_collect_sf_pv_candidates`` so
    train-time target rebuilds (train/target_builder.py) see exactly the
    candidate set the live targets were built from. Columns documented at
    replay/shard.py::SF_MULTIPV_RAW_MAX. Indices are converted to the shard's
    policy encoding in finalize, like ``sf_move_index``.
    """
    rows: list[tuple[int, int, int, int, int]] = []
    truncated = False
    for pv in getattr(res, "pvs", None) or []:
        a = uci_to_policy_index(pv.move_uci, turn)
        if a < 0 or a not in legal_set:
            continue
        if len(rows) >= SF_MULTIPV_RAW_MAX:
            truncated = True
            break
        cp = SF_CP_SENTINEL if pv.cp is None else int(np.clip(int(pv.cp), -32000, 32000))
        mate = 0 if pv.mate is None else int(np.clip(int(pv.mate), -127, 127))
        if pv.wdl is None:
            w = d = -1
        else:
            # _parse_wdl normalizes UCI permille to fractions; store permille
            # per the shard schema (replay/shard.py cols 3-4).
            w, d = round(float(pv.wdl[0]) * 1000), round(float(pv.wdl[1]) * 1000)
        rows.append((a, cp, mate, w, d))
    if truncated:
        _warn_multipv_truncated()
    if not rows:
        return None
    return np.array(rows, dtype=np.int16)


def _collect_sf_label_meta(res) -> np.ndarray:
    """Record-level SF eval metadata (6,) int32; layout in replay/shard.py."""
    nodes = getattr(res, "nodes", None)
    depth = getattr(res, "depth", None)
    cp = SF_CP_SENTINEL if res.cp is None else int(np.clip(int(res.cp), -32000, 32000))
    mate = 0 if res.mate is None else int(np.clip(int(res.mate), -127, 127))
    if res.wdl is None:
        w = d = -1
    else:
        # _parse_wdl normalizes UCI permille to fractions; store permille
        # per the shard schema (replay/shard.py meta layout).
        w, d = round(float(res.wdl[0]) * 1000), round(float(res.wdl[1]) * 1000)
    # nodes/depth are config-driven; clamp so a huge budget can't overflow the
    # int32 field and raise out of _process_sf_results mid-selfplay.
    return np.array(
        [min(int(nodes), 2**31 - 1) if nodes is not None else -1,
         min(int(depth), 2**31 - 1) if depth is not None else -1,
         cp, mate, w, d],
        dtype=np.int32,
    )


def _stamp_sparse_sf_labels(rec, res, *, turn: bool, legal_set: set[int]) -> None:
    rec.sf_multipv_raw = _collect_sparse_pv_rows(res, turn=turn, legal_set=legal_set)
    rec.sf_label_meta = _collect_sf_label_meta(res)


def _collect_sf_pv_candidates(
    res,
    *,
    _turn: bool,
    legal_set: set[int],
    sf_wdl_use_cp_logistic: bool = False,
    sf_wdl_cp_slope: float = 0.010,
    sf_wdl_cp_draw_width: float = 60.0,
) -> tuple[list[int], list[float]]:
    """Extract (action_idx, w + 0.5*d) per legal SF MultiPV candidate."""
    cand_idxs: list[int] = []
    cand_scores: list[float] = []
    for pv in getattr(res, "pvs", None) or []:
        a = uci_to_policy_index(pv.move_uci, _turn)
        if a < 0 or a not in legal_set:
            continue
        score = _pv_wdl_score(
            pv,
            sf_wdl_use_cp_logistic=sf_wdl_use_cp_logistic,
            sf_wdl_cp_slope=sf_wdl_cp_slope,
            sf_wdl_cp_draw_width=sf_wdl_cp_draw_width,
        )
        if score is None:
            continue
        cand_idxs.append(a)
        cand_scores.append(score)
    return cand_idxs, cand_scores


def _build_sf_policy_target(
    cand_idxs: list[int], cand_scores: list[float],
    *, legal_indices: np.ndarray,
    sf_policy_temp: float, sf_policy_label_smooth: float,
) -> np.ndarray:
    """Softmax over MultiPV candidates → POLICY_SIZE vector with optional
    legal-set label smoothing. Final vector is renormalized."""
    scores = np.array(cand_scores, dtype=np.float64) / max(1e-6, sf_policy_temp)
    p_top = _softmax_np(scores).astype(np.float32, copy=False)
    p_sf = np.zeros((POLICY_SIZE,), dtype=np.float32)
    for a, p in zip(cand_idxs, p_top, strict=False):
        p_sf[int(a)] += float(p)

    # Only smooth when SF's candidates don't already cover every legal move. When
    # fully covered (the common case — multipv=40 ≥ legal count for ~83% of
    # positions) the softmax is already a complete distribution and the uniform
    # floor would just flatten it. When uncovered legal moves exist, the floor
    # gives them mass strictly below every covered move (covered = floor + share).
    n_covered = int(np.isin(legal_indices, cand_idxs).sum())  # legal moves SF scored
    has_uncovered = n_covered < int(legal_indices.size)
    if sf_policy_label_smooth > 0.0 and legal_indices.size > 0 and has_uncovered:
        p_sf *= 1.0 - sf_policy_label_smooth
        p_sf[legal_indices] += sf_policy_label_smooth / float(legal_indices.size)

    ps = float(p_sf.sum())
    if ps > 0:
        p_sf /= ps
    return p_sf


def _attach_sf_target_to_last_record(
    state: SelfplayState, idx: int,
    *, p_sf: np.ndarray, a_idx: int, res, legal_indices: np.ndarray,
    turn: bool,
    sf_wdl_use_cp_logistic: bool = False,
    sf_wdl_cp_slope: float = 0.010,
    sf_wdl_cp_draw_width: float = 60.0,
) -> None:
    """`_attach_sf_target_to_record` against the game's latest _NetRecord.

    No-op when the game has no record yet; otherwise identical, including the
    idempotency skip.
    """
    if not state.samples_per_game[idx]:
        return
    _attach_sf_target_to_record(
        state.samples_per_game[idx][-1],
        p_sf=p_sf, a_idx=a_idx, res=res, legal_indices=legal_indices, turn=turn,
        sf_wdl_use_cp_logistic=sf_wdl_use_cp_logistic,
        sf_wdl_cp_slope=sf_wdl_cp_slope,
        sf_wdl_cp_draw_width=sf_wdl_cp_draw_width,
    )


def _attach_sf_target_to_record(
    rec: _NetRecord,
    *,
    p_sf: np.ndarray,
    a_idx: int,
    res,
    legal_indices: np.ndarray,
    # REQUIRED, and not `bool | None`. It used to default to None, with the
    # sparse stamp below skipped in that case -- a caller that forgot it would
    # emit a fully-labelled row carrying `has_sf_multipv_raw = 0`, which is the
    # exact fingerprint `sf_multipv_presence_counts` reports as SF DESYNC
    # CONTAMINATION. Both call sites always passed it, so the skip only ever
    # existed to manufacture a false desync reading. Keep it un-defaulted.
    turn: bool,
    sf_wdl_use_cp_logistic: bool = False,
    sf_wdl_cp_slope: float = 0.010,
    sf_wdl_cp_draw_width: float = 60.0,
) -> None:
    """Stamp SF policy target / move idx / wdl / legal_mask onto *rec*.

    Idempotent: a record that already carries an SF label is left alone.
    """
    if rec.sf_policy_target is not None or rec.sf_move_index is not None:
        return
    rec.sf_policy_target = p_sf
    rec.sf_move_index = a_idx
    rec.sf_wdl = _sf_result_wdl_for_record(
        res,
        sf_wdl_use_cp_logistic=sf_wdl_use_cp_logistic,
        sf_wdl_cp_slope=sf_wdl_cp_slope,
        sf_wdl_cp_draw_width=sf_wdl_cp_draw_width,
    )
    _stamp_sparse_sf_labels(
        rec, res, turn=bool(turn), legal_set={int(x) for x in legal_indices},
    )
    _sf_mask = np.zeros((POLICY_SIZE,), dtype=np.uint8)
    _sf_mask[legal_indices] = 1
    rec.sf_legal_mask = _sf_mask


def _process_sf_label_result_for_record(
    state: SelfplayState,
    *,
    rec: _NetRecord,
    res,
    turn: bool,
    legal_indices: np.ndarray,
    original_res: Any = None,
) -> None:
    """Build + attach the SF label from ``res``.

    ``original_res`` is the shallow pre-escalation result when ``res`` is an
    escalated deep re-query (sf_label_escalate_*); its label WDL is preserved
    as ``rec.sf_wdl_original`` for the blind-spot harvester (see
    blindspot_harvest._harvest_sf_wdl for the ordering rationale).
    """
    if legal_indices.size == 0:
        return
    sf_policy_temp = float(state.game.sf_policy_temp)
    sf_policy_label_smooth = float(state.game.sf_policy_label_smooth)
    legal_set = {int(x) for x in legal_indices}

    a_idx = uci_to_policy_index(res.bestmove_uci, bool(turn))
    bestmove_illegal = a_idx < 0 or a_idx not in legal_set
    if bestmove_illegal:
        a_idx = int(legal_indices[0])

    cand_idxs, cand_scores = _collect_sf_pv_candidates(
        res, _turn=bool(turn), legal_set=legal_set,
        sf_wdl_use_cp_logistic=bool(state.game.sf_wdl_use_cp_logistic),
        sf_wdl_cp_slope=float(state.game.sf_wdl_cp_slope),
        sf_wdl_cp_draw_width=float(state.game.sf_wdl_cp_draw_width),
    )
    if not cand_idxs:
        # Every MultiPV move was illegal here. On a healthy engine that is a
        # rounding event (0.0008 of labelled rows); on a desynced one it is the
        # norm, and the fabricated one-hot below lands on legal_indices[0] — so
        # the row gets a fake bestmove AND a fake policy distribution, both
        # looking well-formed. `no_legal_pv` in the health line counts exactly
        # this; do not make it silent again.
        cand_idxs = [a_idx]
        cand_scores = [0.0]

    p_sf = _build_sf_policy_target(
        cand_idxs, cand_scores,
        legal_indices=legal_indices,
        sf_policy_temp=sf_policy_temp,
        sf_policy_label_smooth=sf_policy_label_smooth,
    )
    # The attach below is idempotent (skips already-labeled records); only tag
    # sf_wdl_original when THIS call actually stamps the label, so a duplicate
    # pending entry can't retroactively mark a record as escalated.
    will_attach = rec.sf_policy_target is None and rec.sf_move_index is None
    _attach_sf_target_to_record(
        rec, p_sf=p_sf, a_idx=a_idx, res=res, legal_indices=legal_indices,
        turn=bool(turn),
        sf_wdl_use_cp_logistic=bool(state.game.sf_wdl_use_cp_logistic),
        sf_wdl_cp_slope=float(state.game.sf_wdl_cp_slope),
        sf_wdl_cp_draw_width=float(state.game.sf_wdl_cp_draw_width),
    )
    if original_res is not None and will_attach:
        rec.sf_wdl_original = _sf_result_wdl_for_record(
            original_res,
            sf_wdl_use_cp_logistic=bool(state.game.sf_wdl_use_cp_logistic),
            sf_wdl_cp_slope=float(state.game.sf_wdl_cp_slope),
            sf_wdl_cp_draw_width=float(state.game.sf_wdl_cp_draw_width),
        )
    if will_attach:
        # Read the STORED value, not a recomputation of it — the shard-side
        # check reads the same field, so the two cannot drift apart. Same rule
        # for the three VALUE-half counts: `rec.sf_wdl` and the stored
        # meta/raw pair, never `res`.
        _report_sf_label_health(
            bestmove_illegal=int(bestmove_illegal),
            no_legal_pv=int(rec.sf_multipv_raw is None),
            **_sf_value_half_counts(rec),
        )


def _label_q_gap(rec: _NetRecord, label_wdl: np.ndarray | None) -> float | None:
    """|net search root Q − SF label Q| in the harvester's nq/sq units.

    Both sides are record-POV ``[W, D, L]``; q = W − L in [-1, 1] (the same
    convention ``blindspot_harvest._q`` uses). ``None`` disables the
    comparison (either eval missing or malformed).
    """
    if label_wdl is None:
        return None
    search_wdl = getattr(rec, "search_wdl_est", None)
    if search_wdl is None:
        return None
    sw = np.asarray(search_wdl, dtype=np.float32)
    lw = np.asarray(label_wdl, dtype=np.float32)
    if sw.shape != (3,) or lw.shape != (3,):
        return None
    return abs(float(sw[0] - sw[2]) - float(lw[0] - lw[2]))


def _maybe_submit_label_escalation(
    state: SelfplayState, pending: _PendingSfLabel, res,
) -> bool:
    """Escalate a completed label query to a deep cold-TT re-search on
    net-vs-label disagreement; returns True when the escalated query is now
    ``pending.future`` (caller re-queues instead of attaching).

    Research bet (``sf_label_escalate_q_gap``, default 0.0 = OFF — the gate
    returns before any engine interaction): deep-SF audits of harvested
    net-vs-label disagreements show the ~700k-node label is wrong in 70-81%
    of high-gap cases (deep SF sides with the net), so exactly those labels
    are re-queried at ``sf_label_escalate_nodes`` with ``fresh=True``
    (ucinewgame — a warm TT from the shallow pass would bias the re-search)
    and the deep result replaces the recorded label. The escalated submit is
    independent of the PID opponent budget and of ``sf_label_nodes_cap``.

    Per-game cap: first-N-over-threshold (``sf_label_escalate_max_per_game``).
    Highest-gap-first ordering is impractical here — labels attach in a
    STREAMING pipeline (results consumed as they arrive, mid-game), so
    ranking a game's gaps would mean buffering every label until game end and
    re-plumbing finalize's flush; not worth it for a ~2-per-game budget.
    """
    q_gap = float(getattr(state.game, "sf_label_escalate_q_gap", 0.0) or 0.0)
    if q_gap <= 0.0:
        return False  # flag off: provable no-op — nothing below runs
    if pending.escalated_from_res is not None or not pending.query_fen:
        return False
    rec = pending.record
    if rec.sf_policy_target is not None or rec.sf_move_index is not None:
        return False  # duplicate pending entry for an already-labeled record
    if not isinstance(state.stockfish, StockfishPool):
        return False  # production (pool) label path only
    label_wdl = _sf_result_wdl_for_record(
        res,
        sf_wdl_use_cp_logistic=bool(state.game.sf_wdl_use_cp_logistic),
        sf_wdl_cp_slope=float(state.game.sf_wdl_cp_slope),
        sf_wdl_cp_draw_width=float(state.game.sf_wdl_cp_draw_width),
    )
    gap = _label_q_gap(rec, label_wdl)
    if gap is None or gap < q_gap:
        return False
    counts = getattr(state, "sf_label_escalations", None)
    slot = int(pending.slot)
    if counts is None or slot < 0 or slot >= len(counts):
        return False
    if counts[slot] >= int(getattr(state.game, "sf_label_escalate_max_per_game", 0)):
        return False
    counts[slot] += 1
    pending.escalated_from_res = res
    pending.future = state.stockfish.submit(
        pending.query_fen,
        nodes=max(1, int(getattr(state.game, "sf_label_escalate_nodes", 0))),
        syzygy_path=pending.syzygy_path,
        fresh=True,
    )
    return True


def _resolve_pending_label_result(pending: _PendingSfLabel) -> Any:
    """``pending.future.result()`` with an escalation airbag: if the deep
    re-query itself failed, fall back to the ORIGINAL label result — an
    escalation hiccup must not cost the row the valid shallow label it
    already had. Clearing ``escalated_from_res`` lets the caller either
    retry-escalate (bounded by the per-game budget) or attach the original
    un-marked."""
    try:
        return pending.future.result()
    except Exception:
        if pending.escalated_from_res is None:
            raise
        _LOG.debug(
            "sf label escalation query failed; keeping original label",
            exc_info=True,
        )
        res = pending.escalated_from_res
        pending.escalated_from_res = None
        return res


def submit_async_sf_label_queries(state: SelfplayState, idxs: list[int]) -> int:
    """Submit selfplay SF label queries without blocking.

    The pending task captures the exact record and legal move set from query
    time, so later game advancement or slot recycling cannot stamp the result
    onto the wrong sample.
    """
    if not idxs:
        return 0
    assert isinstance(state.stockfish, StockfishPool)
    max_pending = max(1, int(state.batch_size) * 8)
    submitted = 0
    for idx in idxs:
        if len(state.pending_sf_labels) >= max_pending:
            break
        if not _slot_latest_record_needs_sf_label(state, idx):
            continue
        legal_indices = state.cboards[idx].legal_move_indices()
        if legal_indices.size == 0:
            continue
        fen = state.cboards[idx].fen()
        syzygy_path = _sf_syzygy_path_for_slot(state, idx)
        fut = state.stockfish.submit(
            fen,
            nodes=_eff_sf_nodes(state, idx, for_move=False, for_label=True),
            syzygy_path=syzygy_path,
        )
        state.pending_sf_labels.append(
            _PendingSfLabel(
                future=fut,
                record=state.samples_per_game[idx][-1],
                turn=bool(state.cboards[idx].turn),
                legal_indices=np.asarray(legal_indices, dtype=np.int64).copy(),
                query_fen=fen,
                slot=int(idx),
                syzygy_path=syzygy_path,
            ),
        )
        submitted += 1
    return submitted


def poll_async_sf_labels(state: SelfplayState) -> tuple[int, int]:
    """Attach completed async SF labels. Returns ``(attached, failed)``.

    With label escalation active (``sf_label_escalate_q_gap`` > 0), a
    completed label whose net-vs-label gap trips the threshold is re-submitted
    at the deep budget and RE-QUEUED instead of attached, keeping this poll
    non-blocking; the escalated result attaches on a later poll (or at the
    finalize flush).
    """
    if not state.pending_sf_labels:
        return 0, 0
    still_pending: list[_PendingSfLabel] = []
    attached = 0
    failed = 0
    for pending in state.pending_sf_labels:
        if not pending.future.done():
            still_pending.append(pending)
            continue
        try:
            res = _resolve_pending_label_result(pending)
            if _maybe_submit_label_escalation(state, pending, res):
                still_pending.append(pending)
                continue
            _process_sf_label_result_for_record(
                state,
                rec=pending.record,
                res=res,
                turn=pending.turn,
                legal_indices=pending.legal_indices,
                original_res=pending.escalated_from_res,
            )
            attached += 1
        except Exception as exc:  # pragma: no cover - defensive drop on SF failure.
            failed += 1
            _report_sf_label_health(failed=1)
            _LOG.debug("async SF label failed: %s", exc, exc_info=True)
    state.pending_sf_labels = still_pending
    return attached, failed


def flush_async_sf_labels_for_records(
    state: SelfplayState, records: list[_NetRecord],
) -> tuple[int, int]:
    """Wait for pending labels attached to these records before replay emit."""
    if not state.pending_sf_labels or not records:
        return 0, 0
    target_records = set(records)
    still_pending: list[_PendingSfLabel] = []
    attached = 0
    failed = 0
    for pending in state.pending_sf_labels:
        if pending.record not in target_records:
            still_pending.append(pending)
            continue
        try:
            res = _resolve_pending_label_result(pending)
            if _maybe_submit_label_escalation(state, pending, res):
                # Finalize path: the record is about to be emitted to replay,
                # so block on the escalated result now (bounded by the
                # per-game escalation cap — at most a couple of deep searches).
                res = _resolve_pending_label_result(pending)
            _process_sf_label_result_for_record(
                state,
                rec=pending.record,
                res=res,
                turn=pending.turn,
                legal_indices=pending.legal_indices,
                original_res=pending.escalated_from_res,
            )
            attached += 1
        except Exception as exc:  # pragma: no cover - defensive drop on SF failure.
            failed += 1
            _report_sf_label_health(failed=1)
            _LOG.debug("async SF label failed during finalize: %s", exc, exc_info=True)
    state.pending_sf_labels = still_pending
    return attached, failed


def has_pending_sf_labels_for_records(
    state: SelfplayState, records: list[_NetRecord],
) -> bool:
    """Whether unfinished async labels still belong to ``records``.

    The manager polls completed futures before finalization, so any matching
    entry left in ``pending_sf_labels`` would make finalize block. This cheap
    preflight lets other slots continue until that result is ready.
    """
    if not state.pending_sf_labels or not records:
        return False
    target_records = set(records)
    return any(pending.record in target_records for pending in state.pending_sf_labels)


def _sf_refute_net_visit_provider(
    _state: SelfplayState, _idx: int,
) -> np.ndarray | None:
    """SEAM: the net's own MCTS visit distribution at the SF-to-move position.

    Running a net search at an opponent turn is architecturally invasive — the
    network-turn pipeline is built around net-color slots — so this provider is
    intentionally unimplemented and returns None. ``sf_refute_opp_policy_net_
    blend`` > 0 is REJECTED at config validation (utils/config_yaml.py), so the
    blend branch that consumes this is never reached in practice; the blend math
    is present so only the provider remains to be filled in.
    """
    return None


def _sf_refute_opp_policy_target(
    state: SelfplayState, idx: int,
    *, cand_idxs: list[int], cand_scores: list[float], legal_indices: np.ndarray,
) -> np.ndarray:
    """MAIN-policy target for an SF-refute opp row.

    Reuses the existing ``_build_sf_policy_target`` soft-label helper (WDL
    softmax over MultiPV by ``sf_policy_temp`` + label smoothing) — the same
    transform the ``policy_sf`` label path uses — then optionally blends the
    net's own visit distribution (``sf_refute_opp_policy_net_blend``).
    """
    p_sf = _build_sf_policy_target(
        cand_idxs, cand_scores,
        legal_indices=legal_indices,
        sf_policy_temp=float(state.game.sf_policy_temp),
        sf_policy_label_smooth=float(state.game.sf_policy_label_smooth),
    )
    blend = float(getattr(state.opening, "sf_refute_opp_policy_net_blend", 0.0) or 0.0)
    if blend > 0.0:
        net_visits = _sf_refute_net_visit_provider(state, idx)
        if net_visits is not None:
            nv = np.asarray(net_visits, dtype=np.float32)
            if nv.shape == p_sf.shape:
                blended = (1.0 - blend) * p_sf + blend * nv
                s = float(blended.sum())
                if s > 0:
                    blended /= s
                return blended.astype(np.float32, copy=False)
    return p_sf


def _emit_sf_refute_opp_record(
    state: SelfplayState, idx: int,
    *, res, legal_indices: np.ndarray, turn: bool,
) -> None:
    """Append a training row at the SF-to-move refute position (opt-in).

    Mirrors the net-turn ``_NetRecord`` shape but marks ``is_sf_refute_opp`` so
    finalize builds it with MAIN policy + wdl/sf_wdl targets only (all aux heads
    masked). The record POV is the SF seat (side to move here), so finalize fills
    the game-outcome WDL for that seat and the sf_wdl label needs NO POV flip
    (contrast a P1 reply label attached to the prior net record).
    """
    if legal_indices.size == 0:
        return
    legal_set = {int(x) for x in legal_indices}
    a_idx = uci_to_policy_index(res.bestmove_uci, bool(turn))
    bestmove_illegal = a_idx < 0 or a_idx not in legal_set
    if bestmove_illegal:
        a_idx = int(legal_indices[0])
    cand_idxs, cand_scores = _collect_sf_pv_candidates(
        res, _turn=bool(turn), legal_set=legal_set,
        sf_wdl_use_cp_logistic=bool(state.game.sf_wdl_use_cp_logistic),
        sf_wdl_cp_slope=float(state.game.sf_wdl_cp_slope),
        sf_wdl_cp_draw_width=float(state.game.sf_wdl_cp_draw_width),
    )
    # This row type never stamps a sparse MultiPV block (no _stamp_sparse_sf_labels
    # below), so unlike the other two sites the stored field cannot be read back —
    # `not cand_idxs` is the same event at its only available site. It is also
    # inert in production: sf_refute_record_opp_rows is false.
    #
    # For the same reason the VALUE-half counts are all left at 0 here rather
    # than faked from `res`: with no stored meta/raw pair the orphan comparison
    # is UNDEFINED, and `eval_pv_checked = 0` is the honest report of that. The
    # `sf_wdl` on this row is also attached AFTER this call, so there is
    # nothing well-formed to judge yet. If this row type ever ships, it needs
    # its own instrumentation, not a borrowed one.
    _report_sf_label_health(
        bestmove_illegal=int(bestmove_illegal),
        no_legal_pv=int(not cand_idxs),
    )
    if not cand_idxs:
        cand_idxs = [a_idx]
        cand_scores = [0.0]
    p_target = _sf_refute_opp_policy_target(
        state, idx,
        cand_idxs=cand_idxs, cand_scores=cand_scores, legal_indices=legal_indices,
    )
    cb = state.cboards[idx]
    x = encode_cboard(
        cb,
        input_history_encoding=state.game.input_history_encoding,
        input_extra_features=state.game.input_extra_features,
    )
    # STM (SF-seat) cp-logistic eval; record POV IS this position's STM, so NO
    # flip (contrast _sf_result_wdl_for_record, which flips a P1 reply label).
    sf_wdl_stm: np.ndarray | None = None
    if bool(state.game.sf_wdl_use_cp_logistic) and (res.cp is not None or res.mate is not None):
        sf_wdl_stm = _cp_to_wdl(
            res.cp, res.mate,
            slope=float(state.game.sf_wdl_cp_slope),
            draw_width_cp=float(state.game.sf_wdl_cp_draw_width),
        )
    elif res.wdl is not None:
        sf_wdl_stm = np.asarray(res.wdl, dtype=np.float32)
    neutral = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    legal_mask = np.zeros((POLICY_SIZE,), dtype=np.uint8)
    legal_mask[legal_indices] = 1
    rec = _NetRecord(
        x=x,
        policy_probs=p_target,
        net_wdl_est=neutral.copy(),
        search_wdl_est=neutral.copy(),
        pov_color=chess.WHITE if turn else chess.BLACK,
        ply_index=int(cb.ply),
        has_policy=True,
        priority=1.0,
        sample_weight=1.0,
        keep_prob=1.0,
        legal_mask=legal_mask,
        sf_wdl=sf_wdl_stm,
        # SF's move is pushed AFTER this record is appended, so every move
        # played so far precedes the recorded position (contrast the net turn,
        # which records after its own push).
        move_offset=len(state.move_idx_history[idx]),
        pos_hash=int(cb.zobrist_hash),
    )
    rec.is_sf_refute_opp = True
    state.samples_per_game[idx].append(rec)


def _push_curriculum_opponent_move(
    state: SelfplayState, idx: int,
    *, legal_indices: np.ndarray,
    cand_idxs: list[int], cand_scores: list[float], regret_limit: float,
) -> None:
    """Pick a curriculum-strength opponent move + push it on the board + advance
    the tree root for next-ply reuse. Marks the slot done if the push terminates."""
    opp_move_idx = _choose_curriculum_opponent_move(
        rng=state.rng,
        legal_indices=legal_indices,
        cand_indices=cand_idxs,
        cand_scores=cand_scores,
        regret_limit=regret_limit,
    )
    if state.samples_per_game[idx]:
        rank, regret = _sf_played_move_diagnostics(opp_move_idx, cand_idxs, cand_scores)
        rec = state.samples_per_game[idx][-1]
        rec.sf_played_move_index = int(opp_move_idx)
        rec.sf_played_rank = rank
        rec.sf_played_regret = regret
    state.cboards[idx].push_index(opp_move_idx)
    state.move_idx_history[idx].append(opp_move_idx)
    if state.mcts_tree is not None and state.root_ids[idx] >= 0:
        state.root_ids[idx] = state.mcts_tree.find_child(
            state.root_ids[idx], opp_move_idx,
        )
    # SF-refute phase countdown: after M opponent SF plies, opponent becomes
    # the net (selfplay_arr is already 1 for these games — no type flip).
    # getattr: unit-test SimpleNamespace fixtures may omit the array (≡ zeros).
    refute_left = getattr(state, "sf_refute_opp_plies_left", None)
    if (
        refute_left is not None
        and idx < len(refute_left)
        and int(refute_left[idx]) > 0
    ):
        refute_left[idx] = int(refute_left[idx]) - 1
    if state.cboards[idx].is_game_over():
        state.done_arr[idx] = 1


def _process_sf_results(
    state: SelfplayState,
    idxs: list[int],
    *,
    results: dict,
    play_curriculum_moves: bool,
    attach_labels: bool = True,
) -> None:
    """Attach SF policy target (+legal mask) to the last _NetRecord per slot
    and, for curriculum games, push the SF-chosen move onto the board."""
    if not idxs:
        return

    sf_policy_temp = float(state.game.sf_policy_temp)
    sf_policy_label_smooth = float(state.game.sf_policy_label_smooth)
    sf_wdl_use_cp_logistic = bool(state.game.sf_wdl_use_cp_logistic)
    sf_wdl_cp_slope = float(state.game.sf_wdl_cp_slope)
    sf_wdl_cp_draw_width = float(state.game.sf_wdl_cp_draw_width)
    pid_regret_limit = (
        float(state.opponent.wdl_regret_limit)
        if state.opponent.wdl_regret_limit is not None else float("inf")
    )

    for idx in idxs:
        res = results[idx]
        legal_indices = state.cboards[idx].legal_move_indices()
        if legal_indices.size == 0:
            state.done_arr[idx] = 1
            continue

        _turn = bool(state.cboards[idx].turn)
        legal_set = {int(x) for x in legal_indices}

        a_idx = uci_to_policy_index(res.bestmove_uci, _turn)
        bestmove_illegal = a_idx < 0 or a_idx not in legal_set
        if bestmove_illegal:
            a_idx = int(legal_indices[0])

        cand_idxs, cand_scores = _collect_sf_pv_candidates(
            res, _turn=_turn, legal_set=legal_set,
        )
        if not cand_idxs:
            cand_idxs = [a_idx]
            cand_scores = [0.0]

        if attach_labels and _slot_latest_record_needs_sf_label(state, idx):
            label_cand_idxs, label_cand_scores = _collect_sf_pv_candidates(
                res, _turn=_turn, legal_set=legal_set,
                sf_wdl_use_cp_logistic=sf_wdl_use_cp_logistic,
                sf_wdl_cp_slope=sf_wdl_cp_slope,
                sf_wdl_cp_draw_width=sf_wdl_cp_draw_width,
            )
            if not label_cand_idxs:
                label_cand_idxs = [a_idx]
                label_cand_scores = [0.0]
            p_sf = _build_sf_policy_target(
                label_cand_idxs, label_cand_scores,
                legal_indices=legal_indices,
                sf_policy_temp=sf_policy_temp,
                sf_policy_label_smooth=sf_policy_label_smooth,
            )
            _attach_sf_target_to_last_record(
                state, idx, p_sf=p_sf, a_idx=a_idx, res=res, legal_indices=legal_indices,
                turn=_turn,
                sf_wdl_use_cp_logistic=sf_wdl_use_cp_logistic,
                sf_wdl_cp_slope=sf_wdl_cp_slope,
                sf_wdl_cp_draw_width=sf_wdl_cp_draw_width,
            )
            # Inside the gate: a curriculum MOVE query produces no training row,
            # so counting it would put a denominator in "sf label health" that
            # is not labels. Reads the stored field, like the async path.
            if state.samples_per_game[idx]:
                _labelled_rec = state.samples_per_game[idx][-1]
                _report_sf_label_health(
                    bestmove_illegal=int(bestmove_illegal),
                    no_legal_pv=int(_labelled_rec.sf_multipv_raw is None),
                    **_sf_value_half_counts(_labelled_rec),
                )

        # SF opponent when curriculum OR mid SF-refute phase (selfplay-tagged).
        in_refute = _slot_in_sf_refute(state, idx)
        uses_sf_opp = in_refute or (not bool(state.selfplay_arr[idx]))
        if play_curriculum_moves and uses_sf_opp:
            # Opt-in: emit a training row at THIS (SF-to-move) refute position
            # BEFORE the move is pushed, so the net trains its MAIN policy head
            # on the punishing move. Default off → byte-identical to today.
            if in_refute and bool(getattr(
                getattr(state, "opening", None), "sf_refute_record_opp_rows", False,
            )):
                _emit_sf_refute_opp_record(
                    state, idx, res=res, legal_indices=legal_indices, turn=_turn,
                )
            # SF-refute plies force full-strength best move (inf regret), not
            # PID-handicapped airbag play — that's the whole point of the channel.
            move_regret = float("inf") if in_refute else pid_regret_limit
            _push_curriculum_opponent_move(
                state, idx, legal_indices=legal_indices,
                cand_idxs=cand_idxs, cand_scores=cand_scores,
                regret_limit=move_regret,
            )


__all__ = [
    "finish_pending_curriculum_moves",
    "finish_sf_annotation_and_moves",
    "flush_async_sf_labels_for_records",
    "has_pending_sf_labels_for_records",
    "poll_async_sf_labels",
    "submit_async_curriculum_move_queries",
    "submit_async_sf_label_queries",
    "submit_async_sf_labels_from_curriculum_moves",
    "submit_sf_queries",
]
