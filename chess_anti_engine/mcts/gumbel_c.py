"""Gumbel MCTS with C-accelerated tree + CBoard operations.

Uses MCTSTree (array-based C tree) for select/expand/backprop and CBoard
for board state management.  The entire tree traversal loop runs in C,
eliminating Python interpreter overhead that was the dominant CPU bottleneck.

Architecture:
  - MCTSTree: array-based C tree holding N/W/prior/children per node
  - CBoard: C chess board for encoding, legal moves, terminal detection
  - Gumbel simulation: C start_gumbel_sims / continue_gumbel_sims state machine
    (tree traversal + sequential-halving scoring all in C).
  - Expand: C expand_from_logits (softmax + tree insert)
  - Backprop: C backprop_many (batched value propagation)
"""
from __future__ import annotations

import logging as _logging
import os
import sys as _sys
import time as _time
from collections.abc import Sequence
from typing import Literal, cast, overload

import chess
import numpy as np
import torch
from numpy.typing import NDArray

from chess_anti_engine.encoding._lc0_ext import CBoard
from chess_anti_engine.encoding import check_encode_buffer_planes, input_plane_count
from chess_anti_engine.inference_dispatcher import supports_inplace_api
from chess_anti_engine.mcts._mcts_tree import batch_compute_relations
from chess_anti_engine.encoding.lc0 import (
    LC0_HISTORY_ROOT_LEGACY_META,
    c_input_history_mode,
    normalize_lc0_history_encoding,
    uses_lc0_root_history,
)
from chess_anti_engine.inference import (  # skylos: ignore (AsyncBatchEvaluator used via stringified cast)
    AsyncBatchEvaluator,
    BatchEvaluator,
    LocalModelEvaluator,
    _COMPILED_BATCH_BUCKETS,
)
import chess_anti_engine.mcts._mcts_tree as _mcts_tree_ext
from chess_anti_engine.mcts._mcts_tree import (
    MCTSTree,
    batch_encode_146,
    batch_encode_146_lc0_root,
    batch_encode_146_lc0_root_legacy_meta,
)

try:
    from chess_anti_engine.mcts._mcts_tree import (
        batch_encode_146_bf16,
        batch_encode_146_lc0_root_bf16,
        batch_encode_146_lc0_root_legacy_meta_bf16,
    )
except ImportError:  # pragma: no cover - older local extension fallback
    batch_encode_146_bf16 = None
    batch_encode_146_lc0_root_bf16 = None
    batch_encode_146_lc0_root_legacy_meta_bf16 = None
from chess_anti_engine.mcts.gumbel import (
    GumbelConfig,
    _completed_q_transform,
    _gumbel,
    _policy_logits_to_full,
    _softmax,
    policy_temp_active,
    _wdl_to_q,
    assert_c_path_can_run,
    gumbel_policy_diagnostics,
    target_log_prior,
)
from chess_anti_engine.mcts.root_tactics import (
    immediate_terminal_cboard_policy_or_draws,
)
from chess_anti_engine.mcts.sampling import sample_action_with_temperature
from chess_anti_engine.moves import POLICY_SIZE


# --- opt-in duplicate telemetry (audit C17) -------------------------------
# A GPU batch under `target_batch=0` accumulates the visits of one
# sequential-halving round, and within a round the tree cannot update -- so
# every visit allocated to a candidate descends to the SAME leaf and produces a
# byte-identical row. Measured on 8 boards / 256 sims: the final round sends 496
# rows for 16 distinct positions (31x). This counts that directly, at the only
# place it is unambiguous: the buffer actually handed to the evaluator.
#
# OFF unless CAE_DUP_STATS is set, because it hashes rows on the hot path.
# CAE_DUP_STRIDE subsamples each row. DEFAULT 1 = EXACT, and it must stay that
# way: the encoded planes are mostly zero, so a strided fingerprint collides
# constantly. Measured at stride 128 it read 52.3% duplication on an arm whose
# true rate is 0.0% -- collisions under-count distinct rows and so OVER-state
# duplication. Validated at stride 1 against an exact-hashing evaluator on four
# arms: 77.8/0.0/0.0/0.0 against 77.5/0.0/0.0/0.0 with bucketing disabled.
#
# With real bucketing ON the two disagree by design, and the gap is the whole
# reason pad rows are counted separately. The exact evaluator hashes the buffer
# it is HANDED, padding included; pad rows hold the PREVIOUS batch's content, so
# they are mostly distinct from the current one and DILUTE the ratio. Measured:
# 0.7783 here against 0.7684 there, and 0.0000 against 0.0139 on target_batch=1
# -- that 0.0139 is pure padding, not surviving search duplication. Bucket
# padding was UNDER-stating duplication, the reverse of the guess that prompted
# the split.
_DUP_STATS = os.environ.get("CAE_DUP_STATS", "") not in ("", "0")
_DUP_STRIDE = max(1, int(os.environ.get("CAE_DUP_STRIDE", "1")))
# Hash every Nth batch. Exact hashing costs ~49% on a CPU-only benchmark where
# search dominates; sampling buys the rate back at 1/N the cost, and a rate
# estimated over hundreds of batches needs no more. 1 = every batch.
_DUP_SAMPLE = max(1, int(os.environ.get("CAE_DUP_SAMPLE", "1")))
_dup_seen = 0
_dup_rows = 0
_dup_dupes = 0
_dup_calls = 0
_dup_pad = 0


def _record_batch_dup(x: np.ndarray, n_real: int, padded: int) -> None:
    """Accumulate duplicate-row counts for one GPU batch. No-op when disabled.

    Hashes only the ``n_real`` rows the search actually produced. The GPU is
    handed ``padded`` rows because `_pad_for_bucket` rounds the batch up to a
    bucket size, and the pad rows are STALE BUFFER CONTENT -- they hash as
    duplicates of each other. Counting them would fold bucket padding into a
    number named "duplicate rate", which is this project's signature defect: a
    quantity that does not mean what its name says. They are real GPU work, so
    they are counted, but separately, as `pad_rows`.
    """
    global _dup_rows, _dup_dupes, _dup_calls, _dup_seen, _dup_pad
    if not _DUP_STATS:
        return
    _dup_seen += 1
    if _dup_seen % _DUP_SAMPLE:
        return
    arr = np.asarray(x)
    n = max(0, min(int(n_real), int(arr.shape[0])))
    _dup_calls += 1
    _dup_rows += n
    _dup_pad += max(0, int(padded) - n)
    if n <= 1:
        return
    flat = np.ascontiguousarray(arr[:n].reshape(n, -1)[:, ::_DUP_STRIDE])
    raw = flat.view(np.uint8).reshape(n, -1)
    _dup_dupes += n - len({raw[i].tobytes() for i in range(n)})


def duplicate_stats() -> tuple[int, int, int, int]:
    """``(rows, duplicate_rows, batches_hashed, pad_rows)`` since the last reset.

    ``rows`` counts only real search leaves; ``pad_rows`` is the bucket padding
    the GPU also evaluated. Redundant GPU work is ``duplicate_rows + pad_rows``,
    but only ``duplicate_rows`` is C17.

    ``batches_hashed`` is NOT decoration: under sampling it can be 0 while the
    search ran thousands of batches, and then ``duplicate_rows / rows`` is 0/0.
    A consumer that divides without checking it reports "no duplication" when
    it means "no observations" -- the same ambiguity that made a masked-mean
    loss unreadable until `has_sf_p0_frac` was added beside it. Use
    :func:`duplicate_rate`, which returns None instead of lying.
    """
    return (_dup_rows, _dup_dupes, _dup_calls, _dup_pad)


def duplicate_rate() -> float | None:
    """Duplicate share of hashed REAL rows, or None when nothing was sampled."""
    if _dup_calls == 0 or _dup_rows == 0:
        return None
    return _dup_dupes / _dup_rows


def pad_rate() -> float | None:
    """Bucket padding as a share of all rows sent, or None when unsampled."""
    if _dup_calls == 0 or (_dup_rows + _dup_pad) == 0:
        return None
    return _dup_pad / (_dup_rows + _dup_pad)


def reset_duplicate_stats() -> None:
    global _dup_rows, _dup_dupes, _dup_calls, _dup_seen, _dup_pad
    _dup_rows = _dup_dupes = _dup_calls = _dup_seen = _dup_pad = 0


# The former GumbelConfig.q_visit_exp / q_global_scale / q_visit_floor defaults.
#
# Those three DESCENT value-transform knobs were deleted (never promoted, absent
# from every config and from PLAY_SEARCH_DEFAULTS). The .c was deliberately NOT
# touched -- editing it would make the next `build_production_extensions.py` a
# silent deploy -- so `start_gumbel_sims` still takes all three as optional
# positional arguments, and the arguments AFTER them (halving_div, c_visit_root,
# c_scale_root, q_visit_exp_root, vloss_mode) are live. They therefore have to be
# passed positionally, and these are the values that make the C search identical
# to the pre-deletion default search:
#
#   q_visit_exp   1.0   linear max_visit term  (`_mcts_tree.c` mv_term arm)
#   q_global_scale  0   descent scales by the LOCAL node's max_visit
#   q_visit_floor -1.0  legacy coupled floor c_scale*(c_visit + mv_term)
#
# They are also exactly the C's OWN declared defaults for those parameters
# (`_mcts_tree.c` MCTSTree_start_gumbel_sims), so the two sides cannot drift
# apart silently: if the C defaults ever move, the literals here still pin the
# search this repo measured.
_DELETED_Q_VISIT_EXP = 1.0
_DELETED_Q_GLOBAL_SCALE = 0
_DELETED_Q_VISIT_FLOOR = -1.0

# The trailing positional block of `start_gumbel_sims`, after `enc_buf`:
#   vloss_weight, target_batch, input_history_mode, rel_buf,
#   q_visit_exp, q_global_scale, q_visit_floor,
#   halving_div, c_visit_root, c_scale_root, q_visit_exp_root, vloss_mode
_StartGumbelTrailingArgs = tuple[
    int, int, int, NDArray[np.uint8] | None,
    float, int, float,
    int, float, float, float, int,
]


def _start_gumbel_trailing_args(
    *,
    cfg: GumbelConfig,
    vloss_weight: int,
    target_batch: int,
    vloss_mode: int,
    rel_buf: NDArray[np.uint8] | None,
) -> _StartGumbelTrailingArgs:
    """The twelve trailing positional args both `start_gumbel_sims` sites pass.

    ⚑ ONE definition on purpose, and the duplication it replaces is why.
    `run_gumbel_root_many_c` starts the C state machine from TWO places -- the
    pipelined 2-group path (`_trees[g].start_gumbel_sims`) and the sequential
    path (`tree.start_gumbel_sims`) -- and the block is twelve positional
    arguments, three of which are the deleted descent q-knobs pinned at their
    former defaults (see the `_DELETED_Q_*` note above) while the rest are live.
    A dropped or reordered argument silently shifts every one that follows.

    The two sites cannot be covered by one spy when they are written out twice:
    a spy is installed by passing `tree=`, and passing `tree=` is exactly what
    turns the pipeline OFF (see the F5 tree-carry guard above), so a spy tree
    can only ever reach the sequential site. The group site is NOT dead code in
    that regime -- `selfplay/match.py` and `scripts/search_gain_probe.py` both
    call with no `tree=`, so an async evaluator plus >= 64 boards takes the
    pipeline. Building the tuple in one place makes a single observation of the
    sequential site cover both by construction.
    """
    return (
        int(vloss_weight),
        int(target_batch),
        c_input_history_mode(cfg.input_history_encoding),
        rel_buf,
      # The three deleted descent q-knobs, pinned at the exact values their
      # GumbelConfig defaults carried (q_visit_exp 1.0 = linear, q_global_scale
      # 0 = local max_visit, q_visit_floor -1.0 = the legacy coupled floor).
      # The C still accepts them as optional positional args, and the args
      # AFTER them are live, so they must be passed rather than omitted.
        _DELETED_Q_VISIT_EXP,
        _DELETED_Q_GLOBAL_SCALE,
        _DELETED_Q_VISIT_FLOOR,
        int(cfg.halving_div),
        float(cfg.c_visit_root),
        float(cfg.c_scale_root),
        float(cfg.q_visit_exp_root),
        int(vloss_mode),
    )


# Minimum compiled-extension ABI the C search path requires. ABI 2 added the
# start_gumbel_sims c_scale_root/q_visit_exp_root args; calling an older compiled
# start_gumbel_sims with them raises a cryptic mid-search TypeError. CANONICAL
# definition (uci/search.py imports it) so the guard covers EVERY C-path consumer —
# selfplay/training and eval call run_gumbel_root_many_c directly, not just UCI.
# See _mcts_tree.c PyInit (ABI_VERSION). Raised to 3 for the audit-W1 fix: an
# un-rebuilt .so still keys the transposition table on the ep-blind repetition
# hash and copies donor child action lists unverified, which injects illegal
# moves into the tree. That failure is SILENT, so it gets the loud guard.
# Raised to 4 for batch_process_ply's search_wdl draw-mode args: selfplay passes
# them on EVERY ply, so a stale .so dies on the first ply of the first game with
# a TypeError about argument counts. Loud, but not actionable — the marker makes
# it say "rebuild" instead, at import, before any game starts.
_REQUIRED_MCTS_ABI = 4

# Mirrors the VLOSS_MODE_* defines in _mcts_tree.c (205-206). LEGACY scores a
# pending leaf as a loss (parallel-PUCT pessimism); VIRTUAL_MEAN scores it at
# the child's existing mean. See the guard in run_gumbel_root_many_c: the
# Gumbel descent only implements LEGACY correctly.
VLOSS_MODE_LEGACY = 0
VLOSS_MODE_VIRTUAL_MEAN = 1

GumbelManyCResult = tuple[list[np.ndarray], list[int], list[float], list[np.ndarray], MCTSTree, list[int]]
GumbelManyCDiagnosticsResult = tuple[
    list[np.ndarray],
    list[int],
    list[float],
    list[np.ndarray],
    MCTSTree,
    list[int],
    list[dict[str, float] | None],
]


def _batch_encoders(input_history_encoding: str | None):
    hist_enc = normalize_lc0_history_encoding(input_history_encoding)
    if hist_enc == LC0_HISTORY_ROOT_LEGACY_META:
        return batch_encode_146_lc0_root_legacy_meta, batch_encode_146_lc0_root_legacy_meta_bf16
    if uses_lc0_root_history(hist_enc):
        return batch_encode_146_lc0_root, batch_encode_146_lc0_root_bf16
    return batch_encode_146, batch_encode_146_bf16


_log = _logging.getLogger(__name__)
_EncodeBuffer = NDArray[np.float32] | NDArray[np.uint16]

# One-shot flags for the two "this silently costs you something" warnings below
# (audit F5 tree-carry vs pipeline, F7 policy_temp vs compact-legal bf16).
# Module-level so a per-ply search does not re-log every move.
_PIPELINE_TREE_WARNED = False
_LEGAL_BF16_TEMP_WARNED = False


def _mark_pipeline_tree_warned() -> None:
    global _PIPELINE_TREE_WARNED
    _PIPELINE_TREE_WARNED = True


def _mark_legal_bf16_temp_warned() -> None:
    global _LEGAL_BF16_TEMP_WARNED
    _LEGAL_BF16_TEMP_WARNED = True


# The reuse gate rejects for two reasons that must NOT share a counter, because
# one is an alarm and the other is routine bookkeeping:
#
#   * MISSING ACTION (audit W2, the original event) — the carried root's child
#     set does not contain something this search is about to look at. That means
#     the tree disagrees with the rules about the position it is sitting on;
#     something upstream changed and it needs a human. Counted in
#     `_ROOT_COVERAGE_MISSES`, read with `root_coverage_miss_count()`.
#   * NARROWED SUPPORT — every action IS present, but the carried root ALSO
#     holds children this search has deliberately excluded (winning-root
#     terminal-draw prune, or `allowed_root_indices_batch`/`searchmoves`). The
#     tree is fine; the search simply asked a narrower question than the ply that
#     built the root. This fires as a matter of course in normal selfplay, so
#     folding it into the alarm would bury a corruption signal under routine
#     traffic. Counted in `_ROOT_NARROWED_REBUILDS`, read with
#     `root_support_narrowed_count()`.
#
# Both are process-cumulative so a run can be asked "did this ever fire?" — a
# guard nobody can observe is a guard nobody knows is dead.
#
# `_ROOT_COVERAGE_MISSES`, its accessor and the `root_coverage_miss=` token in
# the operator line below keep their names AND their original meaning: they are
# the greppable identity of the W2 alarm in shipped selfplay/UCI logs and in the
# ledger's W2 entry. The narrowed case is what is NEW, so it gets the new name
# rather than diluting the old one.
_ROOT_COVERAGE_MISSES = 0
_ROOT_COVERAGE_WARNED = False
_ROOT_NARROWED_REBUILDS = 0
_ROOT_NARROWED_WARNED = False


def _warn_root_coverage_miss() -> None:
    global _ROOT_COVERAGE_MISSES, _ROOT_COVERAGE_WARNED
    _ROOT_COVERAGE_MISSES += 1
    if not _ROOT_COVERAGE_WARNED:
        _ROOT_COVERAGE_WARNED = True
        _log.warning(
            "gumbel: discarded a reused root whose child set was MISSING an "
            "action this search needs (audit W2); rebuilding the root. The tree "
            "disagrees with the position's legal moves — something upstream "
            "changed. Further occurrences are counted, not logged."
        )


def _warn_root_support_narrowed() -> None:
    global _ROOT_NARROWED_REBUILDS, _ROOT_NARROWED_WARNED
    _ROOT_NARROWED_REBUILDS += 1
    if not _ROOT_NARROWED_WARNED:
        _ROOT_NARROWED_WARNED = True
        _log.warning(
            "gumbel: rebuilt a carried root because this search NARROWED its "
            "root support below the carried expansion (winning-root draw prune, "
            "or searchmoves). Routine, not a fault: reusing the wider root would "
            "let an excluded child skew the halving transform of the included "
            "ones. Cost is one ply of tree carry. Further occurrences are "
            "counted, not logged."
        )


def root_coverage_miss_count() -> int:
    """Reused roots rejected for MISSING an action, since process start (W2 alarm)."""
    return _ROOT_COVERAGE_MISSES


def root_support_narrowed_count() -> int:
    """Carried roots rebuilt because this search narrowed its support (routine)."""
    return _ROOT_NARROWED_REBUILDS


# Operator surface for the audit-W1/W2 guards. A counter no production path
# reads is the same defect the guards exist to fix — a value accepted and then
# silently ignored — so the shared C-path entry point emits a line when a guard
# has fired since the last report.
#
# ⚑ Only the two ALARMS re-trigger the line: `tt_donor_reject` and
# `root_coverage_miss`. `root_support_narrowed` is routine and, on a winning-root
# selfplay position, monotonically increasing — re-printing on it would emit a
# line every 60 s forever and train every operator to ignore this message, which
# is how the alarms would get lost. It is announced ONCE (the first time it
# leaves zero) and thereafter only rides along in whatever an alarm prints, plus
# the one-shot WARNING from `_warn_root_support_narrowed` and
# `root_support_narrowed_count()` for anyone who asks.
#
# stderr, NOT stdout: this path is also the UCI engine's search, whose stdout is
# the protocol channel; an unsolicited line there desynchronises the GUI. Both
# production consumers capture stderr into their logs — the selfplay workers via
# `stderr=subprocess.STDOUT` (`tune/distributed_runtime.py`) and the trial itself
# via `> "$LOG" 2>&1` (`scripts/train.sh`) — and the trial actor has no logging
# handler, so a `_log.info` would reach nothing.
#
# Zero cost when the guards are silent: the interval check is one monotonic()
# compare, and the counters are not even read until it elapses.
_TT_HEALTH_INTERVAL_S = 60.0
_tt_health_next_check = 0.0
_tt_health_reported = (0, 0, 0)


def _report_guard_health() -> None:
    global _tt_health_next_check, _tt_health_reported
    now = _time.monotonic()
    if now < _tt_health_next_check:
        return
    _tt_health_next_check = now + _TT_HEALTH_INTERVAL_S

    try:
        reject = int(_mcts_tree_ext.tt_stats()["reject"])
    except (AttributeError, KeyError, TypeError):
        return  # pre-fix .so; the ABI guard above already covers that case
    current = (reject, _ROOT_COVERAGE_MISSES, _ROOT_NARROWED_REBUILDS)
    alarms_moved = current[:2] != (0, 0) and current[:2] != _tt_health_reported[:2]
    narrowed_first_seen = current[2] > 0 and _tt_health_reported[2] == 0
    if not (alarms_moved or narrowed_first_seen):
        return
    _tt_health_reported = current
    print(
        f"[mcts] search guards FIRED since process start: "
        f"tt_donor_reject={current[0]} root_coverage_miss={current[1]} "
        f"root_support_narrowed={current[2]}. "
        "ALARMS — tt_donor_reject>0 means the transposition key no longer "
        "implies the legal move set (audit W1); root_coverage_miss>0 means a "
        "carried tree root was MISSING an action the search needed (audit W2). "
        "Neither corrupts the search, but both mean something upstream changed. "
        "ROUTINE — root_support_narrowed>0 only means searches narrowed their "
        "own root support (winning-root draw prune, or searchmoves) and the "
        "carried root was rebuilt over it; that is correct behaviour and its "
        "only cost is a ply of tree carry. It is reported once, not per "
        "occurrence.",
        file=_sys.stderr,
        flush=True,
    )


# The root sequential-halving SEMANTIC is a compiled constant, and nothing else
# makes it observable. Merging the repo does not deploy it; rebuilding the .so
# for ANY unrelated .c change does. `ABI_VERSION` deliberately does not move
# (see the GSS_HALVING_REV comment in _mcts_tree.c), so without this line a
# regret-series step or an arena delta caused by the elimination rule changing
# under a routine rebuild has nothing to attribute it to — the exact
# "MERGED != DEPLOYED" trap, one level below the source.
#
# Announced from the CONSUMER's own parameter: the value comes off the loaded
# extension module, so the line reports which .so this process is running, not
# what the checkout says it should be. An .so predating the constant IS the old
# semantic, hence the default of 1 rather than "unknown".
#
# stderr and once per process, for the same reasons as _report_guard_health.
_GSS_HALVING_REV_LEGACY = 1
_halving_rev_reported = False


def _report_halving_rev() -> None:
    global _halving_rev_reported
    if _halving_rev_reported:
        return
    _halving_rev_reported = True
    rev = int(getattr(_mcts_tree_ext, "GSS_HALVING_REV", _GSS_HALVING_REV_LEGACY))
    rule = (
        "fresh root_qs (mctx / gumbel.py reference)" if rev >= 2
        else "running W[root]/N[root] (pre-fix; rebuild the C extension)"
    )
    print(
        f"[mcts] gss_halving_rev={rev} loaded from {_mcts_tree_ext.__file__}: "
        f"root sequential-halving eliminates against the {rule}. "
        "Not gating — a stale .so runs, it just searches by the old rule.",
        file=_sys.stderr,
        flush=True,
    )


def _zero_root_output(value: float) -> tuple[np.ndarray, int, float, np.ndarray]:
    return (
        np.zeros((POLICY_SIZE,), dtype=np.float32),
        0,
        float(value),
        np.zeros(POLICY_SIZE, dtype=np.float64),
    )


# Verdicts from _classify_expanded_root_support. Ints rather than an Enum: this
# runs once per board per ply on the search's hot path.
_ROOT_SUPPORT_EQUAL = 0
_ROOT_SUPPORT_MISSING_ACTION = 1
_ROOT_SUPPORT_NARROWED = 2


def _classify_expanded_root_support(
    tree: MCTSTree, root_id: int, actions: np.ndarray,
) -> int:
    """How ``root_id``'s child ACTION SET relates to ``actions``, the support this
    search is about to look at. Reuse requires ``_ROOT_SUPPORT_EQUAL``.

    EQUALITY, not coverage. The predicate this replaced accepted any SUPERSET
    (``np.isin(actions, child_actions).all()``), on the reading that a carried
    root with spare children can only help. It cannot: the C halving scorer
    (``gss_score_and_halve``) derives ``max_visit``, ``total_visits``, the
    prior-weighted ``weighted_q``/``mixed_value`` and the ``min_q``/``max_q``
    normalisation from ALL of the root's children, so a child the CURRENT search
    has deliberately excluded still moves the Q transform every INCLUDED
    candidate is scored through, and can flip which of them survives sequential
    halving. Two production paths narrow the root support below the previous
    ply's expansion — winning-root terminal-draw pruning, and
    ``allowed_root_indices_batch`` (UCI ``searchmoves``) — and a carried root is
    exactly the wide expansion those narrowings just walked away from. The
    improved policy on top is built over the current support only, so accepting
    the superset leaves the eliminator and the returned target disagreeing about
    which moves exist.

    The two rejection verdicts are kept apart because they mean opposite things
    to an operator: ``MISSING_ACTION`` is the W2 alarm (the tree disagrees with
    the rules), ``NARROWED`` is routine (the search asked a narrower question).
    A root that is BOTH missing an action and carrying extras reports
    ``MISSING_ACTION`` — the alarm outranks the bookkeeping.

    ⚑ Scope, so this is not read as more than it is: equality makes the C path
    agree with the Python reference (``gumbel.py``, which never reuses a tree)
    about WHICH MOVES EXIST at the root. It does NOT deliver full parity with a
    fresh root. A reused root still carries the PREVIOUS ply's ``t->prior``
    values, which ``gss_score_and_halve`` reads for ``weighted_q``, while the
    Python side rebuilds ``root_priors`` from this ply's evaluation. That
    divergence is pre-existing, unchanged here, and out of scope.

    Order and visits are irrelevant — only membership. Sorted comparison rather
    than a two-way ``isin`` so a duplicated child action cannot read as a match
    on a set basis; both sides are duplicate-free by construction, and n <= 218.
    """
    child_actions, _visits = tree.get_children_visits(root_id)
    if child_actions.size == actions.size and (
        actions.size == 0
        or bool(np.array_equal(np.sort(child_actions), np.sort(actions)))
    ):
        return _ROOT_SUPPORT_EQUAL
    if actions.size and not bool(np.isin(actions, child_actions).all()):
        return _ROOT_SUPPORT_MISSING_ACTION
    # Every requested action is present and the sets are not equal, so the root
    # carries extras: this search narrowed its support. (An empty `actions`
    # against a root with children lands here too, and is narrowing taken to its
    # limit.)
    return _ROOT_SUPPORT_NARROWED


def _tb_override(tree: MCTSTree | None, probe, wdl: np.ndarray) -> None:
    if probe is None or tree is None:
        return
    indices, leaves = tree.get_pending_tb_leaves(probe.max_pieces)
    if not leaves:
        return
    # solved_out feeds mark_tb_solved so subtrees with proven WDL short-circuit
    # MCTS selection (and propagate up). 0 = no TB hit / skip.
    solved_out = np.zeros(len(leaves), dtype=np.int8)
    probe.apply(leaves, wdl, indices=indices, solved_out=solved_out)
    if (solved_out != 0).any():
        tree.mark_tb_solved(indices.astype(np.int32, copy=False), solved_out)


def leaf_buffer_rows(n_boards: int, *, topk: int, pipelined: bool) -> int:
    """Rows the C leaf encode buffer is sized to, BEFORE any evaluator cap.

    ⚑ THIS IS A SEARCH-SHAPE QUANTITY, not a memory knob. When the buffer
    fills, ``_mcts_tree.c`` does NOT flush and retry: it appends the leaf as a
    ``SOLVED_UNKNOWN`` pseudo-terminal carrying the ROOT's Q
    (``stored_append_terminal(..., g->root_qs[bi], SOLVED_UNKNOWN)``), so leaves
    beyond the buffer are ABSORBED rather than evaluated. Shrinking this below
    what the search asked for silently changes which move is played.

    An evaluator's ``_max_batch`` is min'd against the value returned here, so a
    caller that sets ``max_batch`` below it is choosing a smaller search. This
    function exists so that a caller can compute, BEFORE playing anything, the
    value its cap will be compared against -- read from the search's own code
    rather than from a formula re-derived at the call site, which is exactly the
    kind of duplicate that drifts.

    ``pipelined`` selects the 2-group overlap path (used when the evaluator is
    async, ``n_boards >= 64``, and relations are off) or the single-buffer path.
    The two are not ordered: at topk 32 the single path at 63 boards wants 4032
    rows while the pipelined path at 64 boards wants 2048, so a caller bounding
    a RANGE of board counts must take the max over both regimes rather than
    evaluating either one at its largest n.
    """
    if pipelined:
        mid = n_boards // 2
        max_grp = max(mid, n_boards - mid)  # ceil half for odd splits
        return max(512, max_grp * max(2, int(topk)) * 2)
    return max(256, n_boards * max(2, int(topk))) * 2


@torch.no_grad()
@overload
def run_gumbel_root_many_c(
    model: torch.nn.Module | None,
    boards: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    cfg: GumbelConfig,
    evaluator: BatchEvaluator | None = None,
    pre_pol_logits: np.ndarray | None = None,
    pre_wdl_logits: np.ndarray | None = None,
    per_game_simulations: list[int] | None = None,
    per_game_add_noise: list[bool] | None = None,
    per_game_gumbel_scale: list[float] | None = None,
    cboards: list | None = None,
    tree: MCTSTree | None = None,
    root_node_ids: list[int] | None = None,
    allowed_root_indices_batch: Sequence[set[int] | None] | None = None,
    allow_terminal_root_shortcuts: bool = True,
    tb_probe=None,
    pre_wdl_logits_tb_probed: bool = False,
    target_batch: int = 0,
    vloss_weight: int = 0,
    vloss_mode: int = 0,
    return_diagnostics: Literal[False] = False,
) -> GumbelManyCResult: ...


@overload
def run_gumbel_root_many_c(
    model: torch.nn.Module | None,
    boards: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    cfg: GumbelConfig,
    evaluator: BatchEvaluator | None = None,
    pre_pol_logits: np.ndarray | None = None,
    pre_wdl_logits: np.ndarray | None = None,
    per_game_simulations: list[int] | None = None,
    per_game_add_noise: list[bool] | None = None,
    per_game_gumbel_scale: list[float] | None = None,
    cboards: list | None = None,
    tree: MCTSTree | None = None,
    root_node_ids: list[int] | None = None,
    allowed_root_indices_batch: Sequence[set[int] | None] | None = None,
    allow_terminal_root_shortcuts: bool = True,
    tb_probe=None,
    pre_wdl_logits_tb_probed: bool = False,
    target_batch: int = 0,
    vloss_weight: int = 0,
    vloss_mode: int = 0,
    return_diagnostics: Literal[True],
) -> GumbelManyCDiagnosticsResult: ...


def run_gumbel_root_many_c(
    model: torch.nn.Module | None,
    boards: list[chess.Board],
    *,
    device: str,
    rng: np.random.Generator,
    cfg: GumbelConfig,
    evaluator: BatchEvaluator | None = None,
    pre_pol_logits: np.ndarray | None = None,
    pre_wdl_logits: np.ndarray | None = None,
    per_game_simulations: list[int] | None = None,
    per_game_add_noise: list[bool] | None = None,
    per_game_gumbel_scale: list[float] | None = None,
    cboards: list | None = None,
    tree: MCTSTree | None = None,
    root_node_ids: list[int] | None = None,
    allowed_root_indices_batch: Sequence[set[int] | None] | None = None,
    allow_terminal_root_shortcuts: bool = True,
    tb_probe=None,
    pre_wdl_logits_tb_probed: bool = False,
    target_batch: int = 0,
    vloss_weight: int = 0,
    vloss_mode: int = 0,
    return_diagnostics: bool = False,
) -> GumbelManyCResult | GumbelManyCDiagnosticsResult:
    """Gumbel root search with MCTSTree C tree + CBoard.

    Same API as ``run_gumbel_root_many`` -- drop-in replacement, plus two
    batching controls the Python reference has no equivalent for:

    ``target_batch``
        Leaves to accumulate before handing a batch to the evaluator. 0 (the
        production default) means ``GSS_GPU_BATCH`` = 1024, which spans several
        sequential-halving reps. 1 flushes once per rep, exactly what the
        Python reference does.

    ``vloss_weight``
        Virtual loss applied to a leaf's path while it awaits evaluation. 0
        (the production default) is bit-identical to the pre-virtual-loss code
        path -- and is the C17 defect: with no penalty a later rep re-walks an
        unchanged tree straight back to a leaf already in the batch and
        back-propagates the same value again. At the production shape that is
        38% of evaluated rows and a 37% deficit in distinct nodes. >0 removes
        the duplication while KEEPING the large cross-rep batches, which is
        what ``target_batch=1`` has to give up (~6x the GPU round trips).
        Flipping the default is data-affecting -- it changes every stored
        policy target -- so it needs its own ledger entry; see
        ``docs/rl_loop_audit.md`` C17.
    """
  # Fail fast on a stale compiled extension at the shared C-path entry, so EVERY
  # consumer (selfplay/training + eval call this directly, not just UCI) gets a clear
  # rebuild message instead of a cryptic mid-search TypeError from start_gumbel_sims'
  # new c_scale_root/q_visit_exp_root args. Cheap int compare per call.
    _abi = getattr(_mcts_tree_ext, "ABI_VERSION", 0)
    if _abi < _REQUIRED_MCTS_ABI:
        raise RuntimeError(
            f"compiled _mcts_tree ABI_VERSION={_abi} < required {_REQUIRED_MCTS_ABI} "
            "(missing the start_gumbel_sims root-scale args, the audit-W1 "
            "transposition-key fix and/or batch_process_ply's search_wdl "
            "draw-mode args); rebuild the C extension: "
            "python3 scripts/build_production_extensions.py"
        )
  # Operator surface for the audit-W1/W2 guards. Here rather than in a caller
  # because this is the choke point EVERY C-path consumer goes through —
  # selfplay, training-time eval and UCI all land on this function.
    _report_guard_health()
  # Same choke point, for the semantic that has no guard at all: which root
  # halving rule the LOADED .so implements (see _report_halving_rev).
    _report_halving_rev()
  # Same reasoning, for the OTHER silent-null shape: a GumbelConfig field this
  # path does not implement. Guarding the dispatch boundary rather than each
  # caller's CLI is the point — the CLI is not what chooses the C path, and a
  # caller that grew a python-only knob would otherwise get a clean, wrong,
  # perfectly reproducible measurement.
  #
  # This REPLACED a hand-written `if volatility_search_enabled(cfg): raise`
  # that sat a few lines below. Two guards for one rule is worse than one: the
  # hand-written one named the two fields it knew about, so a THIRD python-only
  # field would have been added to `GumbelConfig` and silently dropped here.
  # `PY_ONLY_GUMBEL_KNOBS` is the named set both this and the dispatchers read.
    assert_c_path_can_run(cfg, where="run_gumbel_root_many_c")
    if int(vloss_mode) == VLOSS_MODE_VIRTUAL_MEAN:
        # `tree_gumbel_select_child` (_mcts_tree.c:2941-2944) mirrors
        # `tree_select_child`'s VIRTUAL_MEAN accounting for the CHILD term and
        # NOT for the PARENT term: there is no VIRTUAL_MEAN branch on
        # parent_N/parent_W, so parent_Q -- which is the FPU for every
        # unvisited child and the weighted_q fallback -- still carries exactly
        # the parallel-PUCT pessimism VIRTUAL_MEAN exists to remove. No caller
        # passes vloss_mode=1 to the Gumbel path today (play-path audit
        # 2026-08-03, F4), so this refuses the trap instead of silently
        # running a half-mirrored descent for whoever wires the knob through.
        # Lift it in the same commit that adds the C parent branch.
        raise ValueError(
            "vloss_mode=VLOSS_MODE_VIRTUAL_MEAN (1) is not implemented for the "
            "Gumbel descent: tree_gumbel_select_child mirrors tree_select_child's "
            "VIRTUAL_MEAN accounting for the child term only, leaving parent_Q "
            "(the FPU for unvisited children) with legacy virtual-loss pessimism "
            "(play-path audit 2026-08-03, F4). Use vloss_mode=0 until the C "
            "parent branch is mirrored."
        )
    _t_init = 0.0
    _t_prepare = 0.0
    _t_gpu = 0.0
    _t_finish = 0.0
    _t_score = 0.0
    _t_policy = 0.0
    _t_python_glue = 0.0
    _n_gpu_calls = 0
    _n_gpu_positions = 0
    _t_func_start = _time.perf_counter()

    n_boards = len(boards)
    if n_boards == 0:
        out_empty = ([], [], [], [], (tree if tree is not None else MCTSTree()), [])
        if return_diagnostics:
            return (*out_empty, [])
        return out_empty

    sim_budget = max(1, int(cfg.simulations))
    _, batch_encode_bf16 = _batch_encoders(cfg.input_history_encoding)

    eval_impl = evaluator
    if eval_impl is None:
        if model is None:
            raise ValueError("run_gumbel_root_many_c requires model or evaluator")
        eval_impl = LocalModelEvaluator(model, device=device)

  # -- 1. Batch root evaluation ------------------------------------------
    root_cboards = cboards if cboards is not None else [CBoard.from_board(b) for b in boards]

    _has_async = hasattr(eval_impl, 'evaluate_encoded_async')
    _has_legal_bf16 = (
        hasattr(eval_impl, "evaluate_legal_bf16")
        and bool(getattr(eval_impl, "supports_legal_bf16", True))
    )
    _has_input_bf16 = (
        batch_encode_bf16 is not None
        and bool(getattr(eval_impl, "supports_input_bf16_bits", False))
    )
  # All async-capable evaluators conform to the protocol; _has_async is the runtime check.
    _async_eval = cast("AsyncBatchEvaluator", eval_impl)
    _use_pipeline = _has_async and n_boards >= 64 and not cfg.compute_relations  # relations ride the single-loop fallback path
  # The pipeline builds its OWN ephemeral sub-trees, ignores the caller's
  # `tree`/`root_node_ids` entirely and returns root ids [-1]*n_boards, so a
  # caller that asked for a persistent tree silently lost every root for the
  # ply -- and lost it as a function of BATCH SIZE, so crossing 64 boards
  # turned tree reuse off with no signal (play-path audit 2026-08-03, F5).
  # A caller passing `tree` is asking for tree carry, which is a search-shape
  # property; the pipeline is a throughput optimisation. Honour the contract
  # and drop the optimisation, loudly, rather than discarding the argument.
  # Production distributed selfplay never reaches this: SlotInferenceClient has
  # no `evaluate_encoded_async`, so `_has_async` is False there.
    if _use_pipeline and tree is not None:
        if not _PIPELINE_TREE_WARNED:
            _log.warning(
                "gumbel_c: disabling the 2-group eval pipeline for this call because a "
                "persistent tree was supplied (n_boards=%d >= 64). The pipeline builds "
                "ephemeral sub-trees and would discard the caller's tree/root_node_ids "
                "(play-path audit 2026-08-03, F5). Pass tree=None to opt back into the "
                "pipeline and accept cold roots every ply.",
                n_boards,
            )
            _mark_pipeline_tree_warned()
        _use_pipeline = False

  # Zero-copy path: when the evaluator exposes get_input_buffer + evaluate_inplace_async
  # (DirectGPUEvaluator with n_slots>=needed), we route the C tree walks to write
  # encodes directly into pinned host memory. This eliminates one numpy memcpy per
  # rep AND lets H2D DMA start immediately when the C walk returns. Pipelined mode
  # also requires 2 slots so submit(g=0)+submit(g=1) don't share output buffers
  # (otherwise the next async submit overwrites pol/wdl before C reads them, forcing
  # a defensive .numpy().copy()).
    _slots_needed = 2 if _use_pipeline else 1
  # supports_inplace_api (not hasattr): dispatcher wrappers forward the
  # slot methods unconditionally, so hasattr lies when the wrapped inner
  # (e.g. MultiGPUDispatcher) has no slot API.
    _inplace = (
        supports_inplace_api(eval_impl)
        and getattr(eval_impl, "n_slots", 1) >= _slots_needed
    )

    def _root_legal_indices_for_eval(i: int) -> np.ndarray:
        legal_idx = root_cboards[i].legal_move_indices()
        if allowed_root_indices_batch is not None:
            allowed_root_indices = allowed_root_indices_batch[i]
            if allowed_root_indices is not None:
                if allowed_root_indices:
                    allowed_arr = np.fromiter(allowed_root_indices, dtype=np.int32)
                    legal_idx = legal_idx[np.isin(legal_idx, allowed_arr)]
                else:
                    legal_idx = legal_idx[:0]
        return legal_idx

    # Broker clients already support compact BF16 legal-policy transport for
    # leaves. Roots know their legal indices before the model call too, so an
    # evaluator can explicitly opt in and avoid returning all POLICY_SIZE
    # float32 logits. Direct/local evaluators keep their zero-copy dense path.
    _use_compact_root = (
        bool(getattr(eval_impl, "supports_compact_root_policy", False))
        and pre_pol_logits is None
        and pre_wdl_logits is None
        and _has_legal_bf16
        and _has_input_bf16
        and not _inplace
        and not cfg.compute_relations
        and not policy_temp_active(float(getattr(cfg, "policy_temp", 1.0)))
    )
    _root_eval_legal = (
        [_root_legal_indices_for_eval(i) for i in range(n_boards)]
        if _use_compact_root else None
    )
    _root_compact_logits: list[np.ndarray] | None = None
    pol_logits_batch: np.ndarray | None = None

    if pre_pol_logits is not None and pre_wdl_logits is not None:
        # Raw assignment only — the unconditional _policy_logits_to_full below
        # converts to full + applies policy_temp once for every path. Applying it
        # here too would divide cached root logits by policy_temp TWICE (T^2).
        pol_logits_batch = np.asarray(pre_pol_logits, dtype=np.float32)
        wdl_logits_batch = np.asarray(pre_wdl_logits, dtype=np.float32)
    elif _inplace:
        batch_enc, batch_enc_bf16 = _batch_encoders(cfg.input_history_encoding)
        if _has_input_bf16 and hasattr(eval_impl, "get_input_buffer_bf16_bits"):
            assert batch_enc_bf16 is not None
            root_buf = eval_impl.get_input_buffer_bf16_bits(n_boards, slot=0)  # pyright: ignore[reportAttributeAccessIssue]
            check_encode_buffer_planes(
                root_buf, cfg.input_extra_features,
                where="run_gumbel_many_c root inplace bf16",
            )
            batch_enc_bf16(root_cboards, root_buf)
        else:
            root_buf = eval_impl.get_input_buffer(n_boards, slot=0)  # pyright: ignore[reportAttributeAccessIssue]
            check_encode_buffer_planes(
                root_buf, cfg.input_extra_features,
                where="run_gumbel_many_c root inplace",
            )
            batch_enc(root_cboards, root_buf)
        if cfg.compute_relations:
            _root_rel = np.empty((n_boards, 5, 64, 64), dtype=np.uint8)
            batch_compute_relations(root_cboards, _root_rel)
            pol_t, wdl_t, event = eval_impl.evaluate_inplace_async(n_boards, slot=0, relations=_root_rel)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            pol_t, wdl_t, event = eval_impl.evaluate_inplace_async(n_boards, slot=0)  # pyright: ignore[reportAttributeAccessIssue]
        if event is not None:
            event.synchronize()
        pol_logits_batch = pol_t.numpy()
        wdl_logits_batch = wdl_t.numpy()
    else:
        batch_enc, batch_enc_bf16 = _batch_encoders(cfg.input_history_encoding)
        _n_planes = input_plane_count(cfg.input_extra_features)
        if _has_input_bf16 and hasattr(eval_impl, "evaluate_encoded"):
            assert batch_enc_bf16 is not None
            xs = np.empty((n_boards, _n_planes, 8, 8), dtype=np.uint16)
            batch_enc_bf16(root_cboards, xs)
        else:
            xs = np.empty((n_boards, _n_planes, 8, 8), dtype=np.float32)
            batch_enc(root_cboards, xs)
        root_rel = None
        if cfg.compute_relations:
            root_rel = np.empty((n_boards, 5, 64, 64), dtype=np.uint8)
            batch_compute_relations(root_cboards, root_rel)
        if _use_compact_root:
            assert _root_eval_legal is not None
            legal_counts = np.fromiter(
                (len(legal) for legal in _root_eval_legal),
                dtype=np.int32,
                count=n_boards,
            )
            legal_flat = np.concatenate(_root_eval_legal).astype(np.int32, copy=False)
            compact_bits, wdl_logits_batch = eval_impl.evaluate_legal_bf16(  # pyright: ignore[reportAttributeAccessIssue]
                xs, legal_flat, legal_counts,
            )
            compact_logits = (
                torch.from_numpy(np.asarray(compact_bits, dtype=np.uint16))
                .view(torch.bfloat16)
                .float()
                .numpy()
            )
            split_at = np.cumsum(legal_counts, dtype=np.int64)[:-1]
            _root_compact_logits = list(np.split(compact_logits, split_at))
        elif _has_async:
            pol_t, wdl_t, event = (
                _async_eval.evaluate_encoded_async(xs, relations=root_rel)
                if root_rel is not None else _async_eval.evaluate_encoded_async(xs)
            )
            if event is not None:
                event.synchronize()
            pol_logits_batch = pol_t.numpy()
            wdl_logits_batch = wdl_t.numpy()
        else:
            pol_logits_batch, wdl_logits_batch = (
                eval_impl.evaluate_encoded(xs, relations=root_rel)
                if root_rel is not None else eval_impl.evaluate_encoded(xs)
            )
    if _root_compact_logits is None:
        assert pol_logits_batch is not None
        pol_logits_batch = _policy_logits_to_full(pol_logits_batch, cfg=cfg)

  # Override root wdl_logits before root_qs is derived (root_qs seeds FPU
  # and the values_out initial pass). UCI may pass cached logits that already
  # include this override; selfplay passes raw batched model logits.
    if tb_probe is not None and not pre_wdl_logits_tb_probed:
        tb_probe.apply(root_cboards, wdl_logits_batch)

    root_qs = [_wdl_to_q(wdl_logits_batch[i]) for i in range(n_boards)]

  # -- 2. Init C tree + roots --------------------------------------------
    _own_tree = tree is None
    if _own_tree:
        tree = MCTSTree()

    probs_out: list[np.ndarray | None] = [None] * n_boards
    actions_out: list[int | None] = [None] * n_boards
    values_out: list[float] = list(root_qs)

    root_ids: list[int] = [-1] * n_boards  # node IDs in C tree
    root_legal: list[np.ndarray | None] = [None] * n_boards
    root_search_legal: list[np.ndarray | None] = [None] * n_boards
    root_pri: list[np.ndarray | None] = [None] * n_boards
    candidates_per_board: list[list[int] | None] = [None] * n_boards
    remaining_per_board: list[list[int] | None] = [None] * n_boards
    budget_remaining: list[int]
    if per_game_simulations is not None:
        budget_remaining = [max(1, int(s)) for s in per_game_simulations]
    else:
        budget_remaining = [sim_budget] * n_boards
    gumbels_per_board: list[np.ndarray | None] = [None] * n_boards

    _full_tree = bool(cfg.full_tree)
    _c_puct = float(cfg.c_puct)
    _fpu_reduction = float(cfg.fpu_reduction)
    _c_visit = float(cfg.c_visit)
    _c_scale = float(cfg.c_scale)

    _t0 = _time.perf_counter()
    for i in range(n_boards):
        root_cb = root_cboards[i]
        legal_idx = (
            _root_eval_legal[i]
            if _root_eval_legal is not None else _root_legal_indices_for_eval(i)
        )
        compact_ll = (
            _root_compact_logits[i]
            if _root_compact_logits is not None else None
        )

        if root_cb.is_game_over():
            probs_out[i], actions_out[i], values_out[i], root_pri[i] = (
                _zero_root_output(float(root_cb.terminal_value()))
            )
            root_qs[i] = values_out[i]
            continue

        if legal_idx.size == 0:
            probs_out[i], actions_out[i], values_out[i], root_pri[i] = (
                _zero_root_output(float(root_qs[i]))
            )
            continue

        root_legal[i] = legal_idx

        terminal_mate = None
        terminal_draws: set[int] = set()
        if allow_terminal_root_shortcuts:
  # Draw terminals are only consumed to prune them from a *winning* root
  # (the block below). Skip the draw scan otherwise so non-winning roots
  # don't pay for the reply-claim work whose result we'd discard. Mate is
  # always detected.
            want_draws = float(root_qs[i]) > 0.0 and legal_idx.size > 1
            terminal_mate, terminal_draws = immediate_terminal_cboard_policy_or_draws(
                root_cb, legal_idx, detect_draws=want_draws,
            )

        if float(root_qs[i]) > 0.0 and legal_idx.size > 1 and terminal_draws:
            draw_arr = np.fromiter(terminal_draws, dtype=np.int32)
            keep = ~np.isin(legal_idx, draw_arr)
            if keep.any():
                legal_idx = legal_idx[keep]
                if compact_ll is not None:
                    compact_ll = compact_ll[keep]
        root_search_legal[i] = legal_idx

  # Softmax priors
        ll = (
            compact_ll.astype(np.float64)
            if compact_ll is not None
            else cast(np.ndarray, pol_logits_batch)[i][legal_idx].astype(np.float64)
        )
        ll -= ll.max()
        e = np.exp(ll)
        s = float(e.sum())
        priors = (e / s) if s > 0 else np.full_like(e, 1.0 / e.size)

        pri = np.zeros(POLICY_SIZE, dtype=np.float64)
        pri[legal_idx] = priors
        root_pri[i] = pri

  # Reuse existing root from persistent tree, or create new one.
  # Skip when pipelining — pipeline creates its own sub-trees.
        if not _use_pipeline:
            _reused = False
            if root_node_ids is not None and root_node_ids[i] >= 0:
                rid = root_node_ids[i]
                # The support check runs on EVERY reuse, including selfplay
                # (audit W2). It used to be short-circuited by
                # `allowed_root_indices_batch is None`, which selfplay always
                # is — so the only path that carries a tree across plies was
                # the only path the check never ran on. A reused root whose
                # child set is missing an action we are about to search makes
                # tree_gumbel_collect_leaf bail at depth 1 (`child_id < 0`),
                # silently spending the whole simulation on the root.
                #
                # It demands EQUALITY, not coverage: `legal_idx` above may have
                # been NARROWED below the previous ply's expansion, by the
                # winning-root terminal-draw prune a few lines up or by
                # `allowed_root_indices_batch`. A carried root still holding the
                # excluded children feeds their visits and Q into the C halving
                # scorer's max_visit / mixed_value / min-max normalisation, so a
                # move this search says does not exist still decides which of the
                # moves that DO exist survives. See
                # _classify_expanded_root_support. Either rejection falls through
                # to the fresh-root build below, which expands exactly
                # `legal_idx` — the Python reference's semantic for a narrowed
                # root — but they are counted apart: MISSING is the alarm, and
                # NARROWED is routine and fires on ordinary winning-root plies.
                if tree.is_expanded(rid):
                    _support = _classify_expanded_root_support(tree, rid, legal_idx)
                    if _support == _ROOT_SUPPORT_EQUAL:
                        root_ids[i] = rid
                        _reused = True
                    elif _support == _ROOT_SUPPORT_MISSING_ACTION:
                        _warn_root_coverage_miss()
                    else:
                        _warn_root_support_narrowed()

            if not _reused:
                rid = tree.add_root(1, float(root_qs[i]))
                root_ids[i] = rid
                tree.expand(rid, legal_idx.astype(np.int32), priors)

        if terminal_mate is not None:
            probs_out[i], actions_out[i], values_out[i] = terminal_mate
            root_qs[i] = values_out[i]
            continue

        if legal_idx.size == 1:
            a0 = int(legal_idx[0])
            p = np.zeros((POLICY_SIZE,), dtype=np.float32)
            p[a0] = 1.0
            probs_out[i] = p
            actions_out[i] = a0
            continue

  # Gumbel noise -> select top-m
        log_pri = np.log(np.maximum(pri[legal_idx], 1e-12))
        _noise_this = per_game_add_noise[i] if per_game_add_noise is not None else cfg.add_noise
        _gumbel_scale = (
            float(per_game_gumbel_scale[i])
            if per_game_gumbel_scale is not None
            else float(cfg.gumbel_scale)
        ) if _noise_this else 0.0
        g = (
            _gumbel_scale * _gumbel(rng, legal_idx.size)
            if _gumbel_scale > 0.0
            else np.zeros(legal_idx.size, dtype=np.float64)
        )
        score: np.ndarray = g + log_pri

        _game_budget = budget_remaining[i]
        if _game_budget <= 1:
            m = 1
        else:
            m_cap = max(2, (_game_budget + 1) // 2)
            m = int(min(int(cfg.topk), int(legal_idx.size), int(m_cap)))
            m = max(2, m)

        kth = min(m - 1, int(score.size) - 1)
        top_idx = np.argpartition(-score, kth)[:m]
        cands = legal_idx[top_idx].astype(int).tolist()

        candidates_per_board[i] = list(cands)
        remaining_per_board[i] = list(cands)
  # Store gumbel values indexed by legal_idx for scoring
        g_full = np.zeros(POLICY_SIZE, dtype=np.float64)
        g_full[legal_idx] = g
        gumbels_per_board[i] = g_full

    _t_init = _time.perf_counter() - _t0
  # -- 3. Sequential halving with C tree ---------------------------------

  # Floor at 256 so single-game UCI (n_boards=1) gets a usefully-sized GPU
  # batch. _enc_buf is this doubled, so it gives a 512-slot buffer minimum
  # (~19 MB). Without the floor, 1 board × topk=32 caps at 64 slots and
  # gss_step flushes the halving round across 4-5 tiny GPU calls.
    _single_leaf_rows = leaf_buffer_rows(n_boards, topk=cfg.topk, pipelined=False)
    _BUCKETS = _COMPILED_BATCH_BUCKETS

  # ---- Pipelined simulation: split games into 2 groups ----------------
  # While GPU evaluates group A's leaves, C does tree walks for group B,
  # and vice versa.  CPU (C tree walks) and GPU overlap on separate hardware.

    def _pad_for_bucket(nl, buf_len):
        for _b in _BUCKETS:
            if _b >= nl:
                return min(_b, buf_len)
        return min(nl, buf_len)

    if _use_pipeline:
        mid = n_boards // 2
        _grp = [list(range(mid)), list(range(mid, n_boards))]
        _trees = [MCTSTree(), MCTSTree()]
        _leaf_cap = leaf_buffer_rows(n_boards, topk=cfg.topk, pipelined=True)
        if _inplace:
  # Pinned-host views: C writes encodes directly here, eval reads from the
  # same memory (no memcpy on submit). Two slots so g=0 / g=1 outputs don't
  # collide.
            _max_batch = getattr(eval_impl, "_max_batch", _leaf_cap)
            _leaf_cap = min(_leaf_cap, _max_batch)
            _enc_bufs = [
                eval_impl.get_input_buffer(_leaf_cap, slot=g)  # pyright: ignore[reportAttributeAccessIssue]
                for g in range(2)
            ]
            for _buf in _enc_bufs:
                check_encode_buffer_planes(
                    _buf, cfg.input_extra_features,
                    where="run_gumbel_many_c split leaf inplace",
                )
        else:
            _enc_dtype = np.uint16 if _has_input_bf16 else np.float32
            _enc_bufs = [
                np.empty(
                    (_leaf_cap, input_plane_count(cfg.input_extra_features), 8, 8),
                    dtype=_enc_dtype,
                )
                for _ in range(2)
            ]

  # Create fresh root nodes in each sub-tree and build local root_ids
        _sub_root_ids: list[list[int]] = [[], []]
        for g in range(2):
            for i in _grp[g]:
                _pri_i = root_pri[i]
                _legal_i = root_search_legal[i]
                if _pri_i is None or _legal_i is None:
                    _sub_root_ids[g].append(-1)
                    continue
                priors = _pri_i[_legal_i].astype(np.float64)
                rid = _trees[g].add_root(1, float(root_qs[i]))
                _trees[g].expand(rid, _legal_i.astype(np.int32), priors)
                _sub_root_ids[g].append(rid)

  # Start both groups
        _n_leaves: list[int | None] = [None, None]
        _tp0 = _time.perf_counter()
        for g in range(2):
            idx = _grp[g]
            ng = len(idx)
            if ng == 0:
                continue
            _cb_g = [root_cboards[i] for i in idx]
            _rid_g = np.array(_sub_root_ids[g], dtype=np.int32)
  # idx slots were populated in the init loop above, so items are non-None.
            _rem_g = cast("list[list[int]]", [remaining_per_board[i] for i in idx])
            _gum_g = cast("list[np.ndarray]", [gumbels_per_board[i] for i in idx])
            _pri_g = cast("list[np.ndarray]", [root_pri[i] for i in idx])
            _bud_g = np.array([budget_remaining[i] for i in idx], dtype=np.int32)
            _rqs_g = np.array([root_qs[i] for i in idx], dtype=np.float64)

            _n_leaves[g] = _trees[g].start_gumbel_sims(
                _cb_g, _rid_g, _rem_g, _gum_g, _pri_g, _bud_g, _rqs_g,
                _c_scale, _c_visit, _c_puct, _fpu_reduction, _full_tree,
                cast(_EncodeBuffer, _enc_bufs[g]),
              # Shared with the sequential site below; see the helper's note --
              # a spy tree cannot reach THIS call (passing `tree` disables the
              # pipeline), so the twelve trailing args are built once.
              # `rel_buf=None`: the pipeline is gated off when
              # `cfg.compute_relations` is set.
                *_start_gumbel_trailing_args(
                    cfg=cfg, vloss_weight=vloss_weight, target_batch=target_batch,
                    vloss_mode=vloss_mode, rel_buf=None,
                ),
            )
        _t_prepare += _time.perf_counter() - _tp0

  # Pipeline loop --------------------------------------------------
  # Each group independently cycles: GPU eval → C tree walks → GPU eval → ...
  # We overlap GPU(A) with C(B) by launching async GPU for one group,
  # then doing continue_gumbel_sims for the other.

        def _drain_sequential(g):
            """Drain remaining simulation for group g without pipelining."""
            nonlocal _t_gpu, _t_prepare, _n_gpu_calls, _n_gpu_positions
            while _n_leaves[g] is not None:
                nl = int(_n_leaves[g])
                padded = _pad_for_bucket(nl, len(_enc_bufs[g]))
                _record_batch_dup(_enc_bufs[g], nl, padded)
                _tg0 = _time.perf_counter()
                if _inplace:
                    pol_t, wdl_t, ev = eval_impl.evaluate_inplace_async(padded, slot=g)  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    pol_t, wdl_t, ev = _async_eval.evaluate_encoded_async(_enc_bufs[g][:padded])
                if ev is not None:
                    ev.synchronize()
                _t_gpu += _time.perf_counter() - _tg0
                _n_gpu_calls += 1
                _n_gpu_positions += nl
                _tp0 = _time.perf_counter()
                _wdl_slice = wdl_t[:nl].numpy()
                _tb_override(_trees[g], tb_probe, _wdl_slice)
                _n_leaves[g] = _trees[g].continue_gumbel_sims(
                    _policy_logits_to_full(pol_t[:nl].numpy(), cfg=cfg),
                    _wdl_slice,
                )
                _t_prepare += _time.perf_counter() - _tp0

  # Main pipelined loop: GPU(g) overlaps with C tree walks(other).
  # The first iteration runs asymmetrically (no pending group 1 results
  # yet), then settles into a steady-state pattern:
  #   1. Submit GPU(0) async
  #   2. C tree walks for group 1 using previous GPU(1) results
  #   3. Sync GPU(0), copy results out of pinned buffers
  #   4. Submit GPU(1) async
  #   5. C tree walks for group 0 using copied GPU(0) results
  #   6. Sync GPU(1), copy results → _pending_g1 for next iteration
  #
  # We copy results from pinned buffers immediately after sync because
  # the next evaluate_encoded_async reuses the same _pinned_pol /
  # _pinned_wdl buffers, invalidating any views.
        _pending_g1 = None  # (pol_np, wdl_np) — synced + copied numpy

        _max_iters = n_boards * max(*budget_remaining, 1) + 100
        for _ in range(_max_iters):
            if _n_leaves[0] is None and _n_leaves[1] is None:
                break

  # Only one group active → flush pending, then drain
            if _n_leaves[0] is None:
                if _pending_g1 is not None:
                    _tp0 = _time.perf_counter()
                    _tb_override(_trees[1], tb_probe, _pending_g1[1])
                    _n_leaves[1] = _trees[1].continue_gumbel_sims(*_pending_g1)
                    _t_prepare += _time.perf_counter() - _tp0
                    _pending_g1 = None
                _drain_sequential(1)
                break
            if _n_leaves[1] is None and _pending_g1 is None:
                _drain_sequential(0)
                break

  # Both active — one pipelined iteration:

  # 1) Submit GPU for group 0 (async — GPU starts working)
            nl0 = int(_n_leaves[0])
            padded0 = _pad_for_bucket(nl0, len(_enc_bufs[0]))
            _record_batch_dup(_enc_bufs[0], nl0, padded0)
            _tg0 = _time.perf_counter()
            if _inplace:
                pol_t0, wdl_t0, ev0 = eval_impl.evaluate_inplace_async(padded0, slot=0)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                pol_t0, wdl_t0, ev0 = _async_eval.evaluate_encoded_async(_enc_bufs[0][:padded0])

  # 2) While GPU processes group 0, do C tree walks for group 1
  #    (continue_gumbel_sims releases GIL; CPU and GPU run in parallel)
            if _pending_g1 is not None:
                _tp0 = _time.perf_counter()
                _tb_override(_trees[1], tb_probe, _pending_g1[1])
                _n_leaves[1] = _trees[1].continue_gumbel_sims(*_pending_g1)
                _t_prepare += _time.perf_counter() - _tp0
                _pending_g1 = None

                if _n_leaves[1] is None:
  # Group 1 finished — wait for GPU(0) and drain group 0
                    if ev0 is not None:
                        ev0.synchronize()
                    _t_gpu += _time.perf_counter() - _tg0
                    _n_gpu_calls += 1
                    _n_gpu_positions += nl0
                    _tp0 = _time.perf_counter()
                    _wdl0 = wdl_t0[:nl0].numpy()
                    _tb_override(_trees[0], tb_probe, _wdl0)
                    _n_leaves[0] = _trees[0].continue_gumbel_sims(
                        _policy_logits_to_full(pol_t0[:nl0].numpy(), cfg=cfg),
                        _wdl0,
                    )
                    _t_prepare += _time.perf_counter() - _tp0
                    _drain_sequential(0)
                    break

  # 3) Wait for GPU(0). Slot-aware path: pinned slot 0 outputs stay live
  # until the next submit(slot=0), which happens at the *top* of next loop
  # iteration — well after step (5) reads them. Legacy path must copy
  # because both submits share one output buffer.
            if ev0 is not None:
                ev0.synchronize()
            _t_gpu += _time.perf_counter() - _tg0
            _n_gpu_calls += 1
            _n_gpu_positions += nl0
            if _inplace:
                pol_np0 = pol_t0[:nl0].numpy()
                wdl_np0 = wdl_t0[:nl0].numpy()
            else:
                pol_np0 = pol_t0[:nl0].numpy().copy()
                wdl_np0 = wdl_t0[:nl0].numpy().copy()
            pol_np0 = _policy_logits_to_full(pol_np0, cfg=cfg)

  # 4) Submit GPU for group 1 (async — safe: group 0 results consumed below)
            if _n_leaves[1] is not None:
                nl1 = int(_n_leaves[1])
                padded1 = _pad_for_bucket(nl1, len(_enc_bufs[1]))
                _record_batch_dup(_enc_bufs[1], nl1, padded1)
                _tg1 = _time.perf_counter()
                if _inplace:
                    pol_t1, wdl_t1, ev1 = eval_impl.evaluate_inplace_async(padded1, slot=1)  # pyright: ignore[reportAttributeAccessIssue]
                else:
                    pol_t1, wdl_t1, ev1 = _async_eval.evaluate_encoded_async(_enc_bufs[1][:padded1])

  # 5) While GPU processes group 1, do C tree walks for group 0
  #    (uses copied numpy arrays — safe from pinned buffer reuse)
            _tp0 = _time.perf_counter()
            _tb_override(_trees[0], tb_probe, wdl_np0)
            _n_leaves[0] = _trees[0].continue_gumbel_sims(pol_np0, wdl_np0)
            _t_prepare += _time.perf_counter() - _tp0

  # 6) Sync GPU(1), copy results for next iteration's step 2.
  # Same-condition re-check — ev1/_tg1/nl1/pol_t1/wdl_t1 are all bound
  # from step (4). Pyright can't narrow across the blocks.
            if _n_leaves[1] is not None:
                _ev1 = ev1  # pyright: ignore[reportPossiblyUnboundVariable]
                _tg1_l = _tg1  # pyright: ignore[reportPossiblyUnboundVariable]
                _nl1 = nl1  # pyright: ignore[reportPossiblyUnboundVariable]
                _pol1 = pol_t1  # pyright: ignore[reportPossiblyUnboundVariable]
                _wdl1 = wdl_t1  # pyright: ignore[reportPossiblyUnboundVariable]
                if _ev1 is not None:
                    _ev1.synchronize()
                _t_gpu += _time.perf_counter() - _tg1_l
                _n_gpu_calls += 1
                _n_gpu_positions += _nl1
  # Inplace path: slot 1's pinned outputs persist until next submit(slot=1)
  # in step (4) of next iter — safe to alias. Legacy path shares one output
  # buffer across both submits, so we must clone before submit(slot=0).
                if _inplace:
                    _pending_g1 = (
                        _policy_logits_to_full(_pol1[:_nl1].numpy(), cfg=cfg),
                        _wdl1[:_nl1].numpy(),
                    )
                else:
                    _pending_g1 = (
                        _policy_logits_to_full(_pol1[:_nl1].numpy().copy(), cfg=cfg),
                        _wdl1[:_nl1].numpy().copy(),
                    )
        else:
            raise RuntimeError(f"pipeline loop did not converge in {_max_iters} iterations")

  # Retrieve remaining candidates from both trees, merge back
        _rem_a = _trees[0].get_gumbel_remaining()
        _rem_b = _trees[1].get_gumbel_remaining()
        remaining_per_board = [None] * n_boards
        for gi, i in enumerate(_grp[0]):
            remaining_per_board[i] = _rem_a[gi] if gi < len(_rem_a) else None
        for gi, i in enumerate(_grp[1]):
            remaining_per_board[i] = _rem_b[gi] if gi < len(_rem_b) else None

  # Store tree refs + root IDs for policy extraction. `None` on the outer
  # type is reserved for the non-pipelined else-branch below (signals
  # "use single tree"); pipelined path always populates every slot.
        _tree_for_board: list[MCTSTree | None] | None = list[MCTSTree | None]([None] * n_boards)
        _rid_for_board: list[int] | None = [0] * n_boards
        for g in range(2):
            for gi, i in enumerate(_grp[g]):
                _tree_for_board[i] = _trees[g]
                _rid_for_board[i] = _sub_root_ids[g][gi]

    else:
  # Non-pipelined fallback (small batches or no async)
        _use_legal_bf16 = (
            _has_legal_bf16
            and not cfg.compute_relations  # compact-legal eval path has no relations input
            and hasattr(tree, "get_pending_legal_indices")
            and hasattr(tree, "continue_gumbel_sims_legal_bf16")
            # The legal-BF16 leaf path softmaxes raw BF16 logits in C with no
            # temperature hook; policy_temp!=1 would leave those leaf priors
            # untempered while root/dense priors are tempered. policy_temp is a
            # rare experiment knob (production=1.0), so fall back to the
            # tempering-aware path when it's set rather than re-pack BF16.
            and not policy_temp_active(float(getattr(cfg, "policy_temp", 1.0)))
        )
  # The gate above is correct, but its PRICE was invisible at the config
  # surface: setting policy_temp to anything but 1.0 costs ~1.9x end-to-end
  # search time (1.63 s -> 3.12 s over 40 searches x 8 boards x 256 sims on
  # CPU; play-path audit 2026-08-03 F7, scratchpad/code_audit_20260803/
  # profile_search.py), because the leaf transport falls back from compact
  # legal bf16 to dense float32 4672. Say so once per process so a
  # policy_temp sweep prices itself.
        if (
            not _use_legal_bf16
            and _has_legal_bf16
            and not cfg.compute_relations
            and hasattr(tree, "get_pending_legal_indices")
            and hasattr(tree, "continue_gumbel_sims_legal_bf16")
            and policy_temp_active(float(getattr(cfg, "policy_temp", 1.0)))
            and not _LEGAL_BF16_TEMP_WARNED
        ):
            _log.warning(
                "gumbel_c: policy_temp=%.6g != 1.0 disables the compact-legal bf16 leaf "
                "transport; leaves fall back to dense float32 %d-wide. The gate is "
                "deliberate (the C bf16 leaf softmax has no temperature hook) -- this "
                "is the price, not a bug. COST: the ~1.9x from the play-path audit "
                "(2026-08-03, F7) was measured on the DIRECT evaluator, which has no "
                "bf16 leaf transport to lose; on the broker path distributed selfplay "
                "actually runs it did NOT reproduce (0.87-1.01x, non-monotone in T, "
                "inside the instrument's own +/-13%% noise). See docs/"
                "experiment_ledger.md \"selfplay search policy temperature\" (e) -- "
                "re-measure on your own transport before budgeting for a slowdown.",
                float(getattr(cfg, "policy_temp", 1.0)), POLICY_SIZE,
            )
            _mark_legal_bf16_temp_warned()
        _use_input_bf16 = _has_input_bf16 and _use_legal_bf16
        if _inplace:
            _max_batch = getattr(eval_impl, "_max_batch", _single_leaf_rows)
            _cap = min(_single_leaf_rows, _max_batch)
            if _use_input_bf16 and hasattr(eval_impl, "get_input_buffer_bf16_bits"):
                _enc_buf = eval_impl.get_input_buffer_bf16_bits(_cap, slot=0)  # pyright: ignore[reportAttributeAccessIssue]
            else:
                _enc_buf = eval_impl.get_input_buffer(_cap, slot=0)  # pyright: ignore[reportAttributeAccessIssue]
            # Buffer-sourced plane count vs cfg's: the C encoder trusts the
            # buffer and never sees cfg.input_extra_features (audit E4).
            check_encode_buffer_planes(
                _enc_buf, cfg.input_extra_features,
                where="run_gumbel_many_c leaf inplace",
            )
        else:
            _enc_dtype = np.uint16 if _use_input_bf16 else np.float32
            _enc_buf = np.empty(
                (_single_leaf_rows, input_plane_count(cfg.input_extra_features), 8, 8),
                dtype=_enc_dtype,
            )
        _root_ids_arr = np.array(root_ids, dtype=np.int32)
        _budget_arr = np.array(budget_remaining, dtype=np.int32)
        _root_qs_arr = np.array(root_qs, dtype=np.float64)

        _rel_buf = (
  # zeros (not empty): padding rows go through the forward like the
  # zeroed enc-buffer rows; uninitialized bytes would be UB-ish noise.
            np.zeros((len(_enc_buf), 5, 64, 64), dtype=np.uint8)
            if cfg.compute_relations else None
        )
        _tp0 = _time.perf_counter()
        n_leaves = tree.start_gumbel_sims(
            root_cboards, _root_ids_arr,
            cast("list[list[int]]", remaining_per_board),
            cast("list[np.ndarray]", gumbels_per_board),
            cast("list[np.ndarray]", root_pri),
            _budget_arr, _root_qs_arr,
            _c_scale, _c_visit, _c_puct, _fpu_reduction, _full_tree,
            cast(np.ndarray, _enc_buf),
          # Same twelve trailing args as the group site above, from the same
          # helper: this is the only one of the two an argument spy can observe.
            *_start_gumbel_trailing_args(
                cfg=cfg, vloss_weight=vloss_weight, target_batch=target_batch,
                vloss_mode=vloss_mode, rel_buf=_rel_buf,
            ),
        )
        _t_prepare += _time.perf_counter() - _tp0

        while n_leaves is not None:
            n_leaves = int(n_leaves)
            padded = _pad_for_bucket(n_leaves, len(_enc_buf))
            # Above the branch on purpose: every arm below sends the same
            # `_enc_buf` rows, and the zero-copy (`_inplace`) arm is the one
            # production actually takes -- wiring this only into the copying
            # arms would have made the instrument read nothing live.
            _record_batch_dup(_enc_buf, n_leaves, n_leaves if _use_legal_bf16 else padded)
            _tg0 = _time.perf_counter()
            if _use_legal_bf16:
                legal_flat, legal_counts = tree.get_pending_legal_indices()
                pol_all, wdl_all = eval_impl.evaluate_legal_bf16(  # pyright: ignore[reportAttributeAccessIssue]
                    _enc_buf[:n_leaves], legal_flat, legal_counts,
                )
            elif _inplace:
                if _rel_buf is not None:
                    pol_t, wdl_t, event = eval_impl.evaluate_inplace_async(  # pyright: ignore[reportAttributeAccessIssue]
                        padded, slot=0, relations=_rel_buf[:padded],
                    )
                else:
                    pol_t, wdl_t, event = eval_impl.evaluate_inplace_async(padded, slot=0)  # pyright: ignore[reportAttributeAccessIssue]
                if event is not None:
                    event.synchronize()
                pol_all = pol_t[:n_leaves].numpy()
                wdl_all = wdl_t[:n_leaves].numpy()
            elif _has_async:
                pol_t, wdl_t, event = (
                    _async_eval.evaluate_encoded_async(_enc_buf[:padded], relations=_rel_buf[:padded])
                    if _rel_buf is not None
                    else _async_eval.evaluate_encoded_async(_enc_buf[:padded])
                )
                if event is not None:
                    event.synchronize()
                pol_all = pol_t[:n_leaves].numpy()
                wdl_all = wdl_t[:n_leaves].numpy()
            else:
                pol_all, wdl_all = (
                    eval_impl.evaluate_encoded(_enc_buf[:padded], relations=_rel_buf[:padded])
                    if _rel_buf is not None
                    else eval_impl.evaluate_encoded(_enc_buf[:padded])
                )
                pol_all = pol_all[:n_leaves]
                wdl_all = wdl_all[:n_leaves]
            _t_gpu += _time.perf_counter() - _tg0
            _n_gpu_calls += 1
            _n_gpu_positions += n_leaves

            _tp0 = _time.perf_counter()
            _tb_override(tree, tb_probe, wdl_all)
            if _use_legal_bf16:
                n_leaves = tree.continue_gumbel_sims_legal_bf16(pol_all, wdl_all)
            else:
                pol_all = _policy_logits_to_full(pol_all, cfg=cfg)
                n_leaves = tree.continue_gumbel_sims(pol_all, wdl_all)
            _t_prepare += _time.perf_counter() - _tp0

        remaining_per_board = cast("list[list[int] | None]", tree.get_gumbel_remaining())
        _tree_for_board = None  # signal to use single tree
        _rid_for_board = None

  # -- 4. Build improved policies from C tree ----------------------------
    _tp0 = _time.perf_counter()
    for i in range(n_boards):
        if probs_out[i] is not None:
            continue

        pri = root_pri[i]
        remaining = remaining_per_board[i]
        if _tree_for_board is not None and _rid_for_board is not None:
            _qtree = _tree_for_board[i]
            rid = _rid_for_board[i]
        else:
            assert tree is not None
            _qtree = tree
            rid = root_ids[i]
        if pri is None or remaining is None or rid < 0:
            probs_out[i] = np.zeros((POLICY_SIZE,), dtype=np.float32)
            actions_out[i] = 0
            continue

        legal = np.nonzero(pri > 0)[0].astype(int)
        root_q_i = float(root_qs[i])

  # Get children stats from C tree (completed_q already negated)
        assert _qtree is not None
        child_actions, child_visits, child_q = _qtree.get_children_q(rid, root_q_i)
        action_to_slot = {}
        for j in range(child_actions.size):
            action_to_slot[int(child_actions[j])] = j

        completed_q = np.empty(legal.size, dtype=np.float64)
        visits = np.empty(legal.size, dtype=np.float64)
        for j, a in enumerate(legal):
            slot = action_to_slot.get(int(a))
            if slot is not None:
                completed_q[j] = float(child_q[slot])
                visits[j] = float(child_visits[slot])
            else:
                completed_q[j] = root_q_i
                visits[j] = 0.0

  # The C search returns the TREE; the improved policy on top of it is built
  # here, in Python. That is why the decoupled target sigma needs no .c change
  # and no extension rebuild -- both search paths converge on this arithmetic.
  # Mirrors gumbel.py::_build_improved_policy_for_board exactly.
        log_prior = np.log(np.maximum(pri[legal], 1e-12))
        q_play = _completed_q_transform(
            actions=legal,
            priors=pri[legal],
            visits=visits,
            qvalues=completed_q,
            raw_value=root_q_i,
            cfg=cfg,
            root=True,
        )
        imp_all = _softmax(log_prior + q_play)
  # The STORED row may use a smaller sigma than the PLAYED move did
  # (`target_max_visit_cap`, the Q term) and/or an UNTEMPERED prior
  # (`target_untempered_prior`, the log_prior term). Written out long-hand
  # rather than via a helper closure: this is inside the per-board loop, so a
  # closure captures loop variables (ruff B023) and widens their narrowed types
  # back to `| None`.
        target_cap = int(cfg.target_max_visit_cap)
        log_prior_store = target_log_prior(log_prior, cfg=cfg)
        if target_cap <= 0 and log_prior_store is log_prior:
            imp_store = imp_all
        else:
            q_store = q_play if target_cap <= 0 else _completed_q_transform(
                actions=legal,
                priors=pri[legal],
                visits=visits,
                qvalues=completed_q,
                raw_value=root_q_i,
                cfg=cfg,
                root=True,
                max_visit_cap=target_cap,
            )
            imp_store = _softmax(log_prior_store + q_store)
        probs = np.zeros((POLICY_SIZE,), dtype=np.float32)
        probs[legal] = imp_store.astype(np.float32)

        best_a = int(remaining[0])
  # Gumbel sequential halving leaves the survivor at remaining[0]; map that
  # back to its position in the full ``legal`` array (= imp_all), as the
  # Python reference does (gumbel.py). This used to be an inlined
  # re-implementation of the shared primitive whose degenerate fallback was
  # `best_a` and which exponentiated out of log space with no isfinite guard
  # (play-path audit 2026-08-03, F10).
  #
  # ONE searchsorted, IN-RANGE-CHECKED, reused by the value lookup below --
  # which already carried that check, so the codebase has never treated
  # `best_a in legal` as guaranteed. Unchecked, an out-of-range hit would
  # silently select a DIFFERENT legal action, or IndexError when `best_a`
  # exceeds every entry. It is unreachable by construction (the candidate set
  # is drawn from the same pruned `legal_idx` the priors were written on, and
  # a prior only leaves `legal` by underflowing to exactly 0.0, i.e. a >745
  # float64 logit gap), so this is belt-and-braces, not a live path.
        j_best = int(np.searchsorted(legal, best_a)) if legal.size > 0 else 0
        best_in_legal = j_best < legal.size and int(legal[j_best]) == best_a
        if best_in_legal:
            action = sample_action_with_temperature(
                rng, legal, imp_all, float(cfg.temperature), argmax_idx=j_best,
            )
        else:
  # Pre-F10 behaviour at temperature <= 0. It also differs from pre-F10 at
  # temperature > 0, which sampled from `legal` (and consumed an rng draw):
  # a survivor outside the returned policy's support means the played-move
  # and returned-policy criteria have already diverged (audit F3), so the
  # draw carries no information and the survivor is the honest answer.
            action = best_a

        probs_out[i] = probs
        actions_out[i] = action

  # Value from child
        slot = action_to_slot.get(best_a)
        if slot is not None and int(child_visits[slot]) > 0 and best_in_legal:
            values_out[i] = float(completed_q[j_best])
        else:
            values_out[i] = root_q_i

  # Build legal masks
    legal_masks_out: list[np.ndarray] = []
    for i in range(n_boards):
        mask = np.zeros((POLICY_SIZE,), dtype=np.bool_)
        rl = root_legal[i]
        if rl is not None:
            mask[rl] = True
        legal_masks_out.append(mask)

    _t_policy = _time.perf_counter() - _tp0
    _t_total = _time.perf_counter() - _t_func_start
    _t_python_glue = _t_total - _t_init - _t_prepare - _t_gpu - _t_finish - _t_score - _t_policy
    if _log.isEnabledFor(_logging.DEBUG):
        _avg_batch = _n_gpu_positions / max(1, _n_gpu_calls)
        _log.debug(
            "gumbel profile (n_boards=%d): total=%.3fs init=%.3f prep=%.3f "
            "gpu=%.3f(%dcalls,%dpos,avg=%.1f) "
            "finish=%.3f score=%.3f policy=%.3f glue=%.3f%s",
            n_boards, _t_total, _t_init, _t_prepare,
            _t_gpu, _n_gpu_calls, _n_gpu_positions, _avg_batch,
            _t_finish, _t_score, _t_policy, _t_python_glue,
            " PIPELINE" if _use_pipeline else "",
        )
  # When pipelining, sub-trees are ephemeral — invalidate root IDs so
  # the caller doesn't try to reuse nodes that don't exist in the main tree.
    _ret_root_ids = root_ids if not _use_pipeline else [-1] * n_boards
    assert tree is not None
    if return_diagnostics:
        diagnostics_out: list[dict[str, float] | None] = [None] * n_boards
        for i in range(n_boards):
            probs = probs_out[i]
            action = actions_out[i]
            legal = root_legal[i]
            if probs is None or action is None or legal is None:
                continue
            diagnostics_out[i] = gumbel_policy_diagnostics(
                probs=probs,
                action=int(action),
                legal=legal.astype(int, copy=False),
                candidates=candidates_per_board[i],
            )
        return (
            cast("list[np.ndarray]", probs_out),
            cast("list[int]", actions_out),
            values_out,
            legal_masks_out,
            tree,
            _ret_root_ids,
            diagnostics_out,
        )
    return (
        cast("list[np.ndarray]", probs_out),
        cast("list[int]", actions_out),
        values_out,
        legal_masks_out,
        tree,
        _ret_root_ids,
    )
