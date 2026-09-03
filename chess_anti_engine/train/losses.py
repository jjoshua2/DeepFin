from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F

from chess_anti_engine.train.sparse_sf_ce import sparse_sf_policy_ce

if TYPE_CHECKING:
    from chess_anti_engine.train.target_builder import SfTargetParams

from chess_anti_engine.moves import COMPACT_POLICY_SIZE, POLICY_SIZE
from chess_anti_engine.moves.torch_maps import compact_to_full_index_for as _compact_to_full_index_for
from chess_anti_engine.train.constants import (
    DEFAULT_GUMBEL_TOPK,
    REGRET_TO_Q_SCALE,
    SF_OWN_REGRET_CAP_CP,
    future_regret_field_names,
    normalize_gumbel_topk,
)

# Game-phase buckets for per-phase loss reporting, by PIECE COUNT — the same
# definition and the same constant as `eval/audit.py`'s per-phase deep-SF
# regret, so a training column and an audit column now name the same set of
# positions.
#
# ⚑⚑ THIS USED TO BUCKET ON `moves_left`, AND THAT WAS AN INSTRUMENT BUG, NOT A
# DISTRIBUTION. `selfplay/finalize.py:931` writes `moves_left` as
# ``(total_plies_played - ply_index) / max_plies`` — the divisor is the
# CONFIGURED PLY CAP (`max_plies: 450` in production), not the game's own
# length — so the quantity is "plies REMAINING as a share of the cap" and says
# nothing about the board. A row at ply 2 of a 60-ply adjudicated game scored
# 0.129 and was labelled `end`. The old cuts (0.45 / 0.31, calibrated
# 2026-04-25 in commit 0fcf899e4 and correct then) had drifted to P99.4 / P96.4
# of the realized distribution, which put 96.4 % of rows in `end`: measured
# 2026-08-02 over the whole live window (713 shards, 1,273,501 rows,
# `runs/pbt2_small/replay/train_trial_13a9f_.../replay_shards`) at
# 0.61 % / 3.03 % / 96.37 %, median `moves_left` 0.113, implied median game
# length 123 plies. Re-deriving the two thresholds would have restored three
# equal buckets of a quantity that still is not game phase — the failure mode
# this repo keeps paying for, a metric that does not mean what its name says,
# with healthy-looking numbers on top. Under the piece-count definition below
# the same window reads 30.7 / 32.4 / 36.9.
#
# ⚑ THE COLUMNS WERE RENAMED IN THE SAME COMMIT — `wdl_loss_open` ->
# `wdl_loss_phase_open`, and likewise for `policy_loss_*` and the `test_`
# twins. A ruler change must invalidate its records: the old and new columns
# measure different sets of rows, so they must not share a name that lets them
# be plotted as one series.
#
# `moves_left` is untouched as a TARGET (the `moves_left` head still regresses
# it); only the reporting split moved off it. NOTE for anyone who returns to
# that field: it is now internally consistent, and this comment used to say the
# opposite. `finalize.py:931` (`_build_replay_samples`) divides by an ABSOLUTE
# game total — `total_plies_played=int(cb.ply)` at `finalize.py:1489`/`:1502` —
# and BOTH play paths now supply an ABSOLUTE `ply_index`: the production C path
# in `mcts/_mcts_tree.c:4870` (`ply_out[i] = (int32_t)cb->ply` inside
# `py_batch_process_ply`), and the Python fallback in
# `selfplay/network_turn.py:932` (`_append_records_via_python`, which reads
# `state.cboards[idx].ply`). Until the resume-format v3 change the fallback set
# `ply_index` RELATIVE to its own `chess.Board.move_stack`, which mixed a
# relative numerator with an absolute total. PAYOFF: a FEN seed entering at
# absolute ply 136 and playing 40 plies ends at `total_plies_played` 176, so
# its FIRST record read `(176 - 0) / 450 = 0.391` — "87% of the cap still to
# play" — where the truth is `(176 - 136) / 450 = 0.089`. Every fallback row of
# such a game was shifted by the seed's entry ply. The C path was always
# correct, so production data was never affected; the fallback and its
# consumers were.
from chess_anti_engine.utils.architecture import DEFAULT_PHASE_PIECE_THRESHOLDS

# Deliberately the module constant and NOT `model_cfg.phase_piece_thresholds`:
# that knob shapes the model's own phase adapter, and letting the reporting
# split follow it would silently break comparability with `eval/audit.py`,
# which is the entire reason for bucketing this way.
_PHASE_END_MAX_PIECES, _PHASE_MID_MAX_PIECES = DEFAULT_PHASE_PIECE_THRESHOLDS

# Planes 0:12 are the current position's 12 piece-occupancy planes in every
# shipping LC0 encoding (legacy and root-history), so their sum is the exact
# root piece count. Same slice and same assumption as
# `model/transformer.py::_phase_indices_from_input`.
_PIECE_PLANE_COUNT = 12

# The suffixes from `_phase_split_masks` that partition by game phase. Only
# these get row-count columns: `selfplay`/`curriculum` split on a boolean flag
# whose balance is already reported by `frac_is_selfplay` / `frac_tagged`.
_PHASE_BUCKET_SUFFIXES = ("phase_open", "phase_mid", "phase_end")

# Stable order shared with the exact-epoch census.  These names identify the
# optimizer terms, not reporting columns: floor and shape deliberately share
# an eligibility mask but remain separate configured heads.
EXACT_OBJECTIVE_NAMES: tuple[str, ...] = (
    "policy",
    "soft_policy",
    "future_policy",
    "sf_own",
    "sf_own_regret",
    "wdl",
    "sf_move",
    "sf_eval",
    "categorical",
    "volatility",
    "sf_volatility",
    "moves_left",
    "sf_policy_floor",
    "sf_shape",
)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of x over mask==1 entries. mask is broadcastable to x."""
    mask = mask.to(x.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def masked_sum_and_count(
    x: torch.Tensor, mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The unreduced ingredients of ``masked_mean``: (sum over mask, mask count).

    Reported instead of the ratio when the caller aggregates ACROSS batches:
    summing these and dividing once gives the true mask-weighted mean over the
    whole iteration, whereas averaging per-batch ``masked_mean`` values weights
    every batch equally and is a different estimator as soon as the mask count
    varies between batches. Both accumulate in float32 regardless of the
    autocast dtype ``x`` arrives in.
    """
    m = mask.to(torch.float32)
    return (x.to(torch.float32) * m).sum(), m.sum()


def sf_multipv_presence_counts(
    batch: dict[str, torch.Tensor], *, has_sf_wdl: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(no-MultiPV labelled rows, labelled rows CHECKED) for this batch, as 0-d sums.

    **DENOMINATOR, stated once: rows of this batch with ``has_sf_wdl`` set.**
    Not ``net_mask * has_sf_wdl``, and not all batch rows. That is the exact
    population ``eval/value_optimism.py::sf_multipv_missing_rate`` divides by —
    the offline gate that selected the 122 shards quarantined 2026-08-01 — so
    the live column and the gate's per-shard reading are the same quantity and
    compare without rescaling. (The sibling ``sf_rebuild_*_frac`` pair divides
    by ALL rows of the rebuilt batch; quoting one against the other's
    denominator is how 0.191973 and 0.207461 ended up describing the same 122
    shards in one docstring. Do not repeat it here.)

    The numerator is the desync fingerprint. The selfplay writer stamps
    ``sf_multipv_raw`` on a labelled row together with its SF eval
    (``selfplay/stockfish_turn.py::_stamp_sparse_sf_labels``); a labelled row
    that arrives without the block came through
    ``_collect_sparse_pv_rows``'s ``if not rows: return None``, i.e. not ONE of
    Stockfish's MultiPV moves was legal at the position queried — a UCI engine
    answering a DIFFERENT position. On healthy data this is **exactly
    0.000000**: 11.05 M labelled rows over every clean stretch on disk (PR #302),
    2 878 620 across four separated quiet stretches. It is a LOWER bound on
    contamination, not the poisoned share: a desynced engine strips the block on
    only ~59 % of the labels it poisons, so divide by ~0.59 for the true share.

    ⚑ The second return value is not decoration. ``has_sf_multipv_raw`` is an
    OPTIONAL shard field, so a batch that never carried it must not report a
    perfect zero rate. When the field is absent BOTH counts are zero, which
    drives the reported ``sf_multipv_checked_frac`` to 0.0 and marks the rate
    unmeasured — as opposed to measured-and-clean. A consumer that reads the
    rate without the checked-frac cannot tell a healthy window from a blind
    instrument, which is the exact defect this column exists to catch.

    ``checked_frac == 0.0`` covers TWO cases, and they are operationally
    identical — nothing was inspected, so the rate above it means nothing:
    (a) the batch carried no ``has_sf_multipv_raw`` field at all (the early
    return here), and (b) the batch carried the field but had no ``has_sf_wdl``
    rows, so the mask summed to zero. Do not read (b) as "clean"; on the
    production window ``checked_frac`` sits near 0.99 and any collapse toward
    zero is a blind instrument whichever cause produced it.

    ⚑ SHARD HETEROGENEITY READS AS CONTAMINATION, NOT AS HEALTH.
    ``DiskReplayBuffer._gather_rows`` builds its ``proto`` as the UNION of the
    fields across the sampled chunks, so a chunk written without
    ``has_sf_multipv_raw`` is zero-filled to match a chunk that has it — and a
    zero there is indistinguishable from "labelled but no MultiPV block", i.e.
    contamination. That is a false ALARM rather than a silent pass, the right
    direction for a tripwire, but a future "any non-zero is an incident" rule
    would then fire on a mixed pool instead of on a desync.

    Currently moot, and by a slightly stronger margin than "every shard has the
    field": over the 713 shards of the live window, 712 carry ``has_sf_wdl``
    and ``has_sf_multipv_raw`` TOGETHER and exactly 0 carry ``has_sf_wdl``
    without the raw flag — which is the only combination that could produce the
    false alarm. The single exception (``shard_033951.zarr``) carries NEITHER,
    so the union zero-fills its ``has_sf_wdl`` as well and its rows fall out of
    the denominator instead of into the numerator.

    Unconditional by construction: it reads two ``has_`` vectors that every
    collated batch either carries or does not, and consults no flag. The
    Earlier dense-only data treated the rebuild policy/WDL coverage gap as the
    same signal, but supported sparse-policy rows legitimately open that gap.
    This unconditional presence check is now the sole in-loop label-health
    contract and does not depend on ``rebuild_sf_targets``.
    """
    has_raw = batch.get("has_sf_multipv_raw")
    if has_raw is None:
        zero = has_sf_wdl.new_zeros(())
        return zero, zero.clone()
    missing = 1.0 - has_raw.to(torch.float32)
    return masked_sum_and_count(missing, has_sf_wdl)


def sf_wdl_wellformed(sf_wdl: torch.Tensor | None) -> torch.Tensor | None:
    """Per-row 1.0/0.0: is this ``sf_wdl`` a usable (W, D, L) distribution?

    Degenerate means non-finite, outside [0, 1], not summing to 1, or exactly
    uniform. Kept as its own function because it is the ONE test the P2 audit
    named that the desync does not move, and a reader needs to be able to see
    that the thing being counted is a well-formedness test and nothing cleverer.
    """
    if sf_wdl is None or sf_wdl.ndim != 2 or sf_wdl.shape[-1] != 3:
        return None
    v = sf_wdl.to(torch.float32)
    finite = torch.isfinite(v).all(dim=-1)
    in_range = ((v >= -1e-6) & (v <= 1.0 + 1e-6)).all(dim=-1)
    sums_to_one = (v.sum(dim=-1) - 1.0).abs() <= 1e-3
    uniform = ((v - 1.0 / 3.0).abs() < 1e-4).all(dim=-1)
    return (finite & in_range & sums_to_one & ~uniform).to(torch.float32)


def sf_wdl_health_counts(
    batch: dict[str, torch.Tensor], *, has_sf_wdl: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(degenerate rows, orphaned rows, sf_wdl rows) for this batch, as 0-d sums.

    The VALUE half of the SF label had no detector in either direction until
    2026-08-03 (SELFPLAY_AUDIT P2), while carrying realized ``sf_wdl_frac``
    0.45 of the trained value target. These two counts close the two directions
    the audit named; the third value is their shared denominator.

    **DENOMINATOR: rows with ``has_sf_wdl`` set** — the same population
    ``sf_multipv_presence_counts`` divides by, and published in its own right
    (``sf_wdl_rows``) rather than borrowed from ``sf_multipv_checked_rows``.
    Borrowing would make both value-side rates go blind the moment the POLICY
    field went missing from the batch, which is the one circumstance under
    which they matter most.

    ``degenerate`` — ``sf_wdl`` present but not a usable distribution. State
    plainly what this is worth as a desync detector: **nothing.** Measured
    2026-08-03 it is exactly 0 rows on the 122 quarantined shards (209,259
    labelled) AND exactly 0 on the 1,264,058-row post-quarantine window. A
    desynced engine returns a real search of another position, so its cp is an
    ordinary number and the logistic maps it to an ordinary distribution: the
    label is WELL-FORMED AND WRONG. The count is kept because its floor is a
    proven exact zero over 1.47 M rows, which makes it a cheap tripwire for a
    different class — a cp→WDL parameter, POV or dtype change reaching the
    writer — and because "we looked and it is not this" is the finding.

    ``orphaned`` — the P2 blind spot itself, counted: rows the POLICY detector
    flagged (no MultiPV block) that nonetheless carry a well-formed ``sf_wdl``,
    which trains at 0.45 weight with nothing marking it. On the quarantined
    shards this was 43,413 of 43,417 flagged rows (0.9999). It is deliberately
    a near-twin of ``sf_no_multipv_rows``: publishing both makes the audit's
    claim machine-checked rather than documented. Equality means every flagged
    row carried an unmarked value label; a divergence would mean some flagged
    rows at least carry a marker, and that is worth knowing the moment it
    changes. It is NOT an independent detector and must not be summed with the
    policy rate.
    """
    wdl_rows = has_sf_wdl.to(torch.float32).sum()
    wellformed = sf_wdl_wellformed(batch.get("sf_wdl"))
    if wellformed is None:
  # No sf_wdl column: nothing to judge. Report zero rows so the rates read
  # UNMEASURED via `sf_wdl_rows` rather than reading a clean 0.0 over a
  # denominator that still counts rows.
        zero = has_sf_wdl.new_zeros(())
        return zero, zero.clone(), zero.clone()
    degenerate = ((1.0 - wellformed) * has_sf_wdl.to(torch.float32)).sum()
    has_raw = batch.get("has_sf_multipv_raw")
    if has_raw is None:
        return degenerate, has_sf_wdl.new_zeros(()), wdl_rows
    missing = 1.0 - has_raw.to(torch.float32)
    orphaned = (missing * wellformed * has_sf_wdl.to(torch.float32)).sum()
    return degenerate, orphaned, wdl_rows


def sf_eval_pv_orphan_counts(
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """(orphaned rows, checked rows) for the value-half desync check, as 0-d sums.

    Both are computed on the HOST by ``Trainer._prepare_host_arrays``, because
    the two shard fields the predicate needs — ``sf_multipv_raw`` and
    ``sf_label_meta`` — do not reach the GPU in production. The definition, the
    calibration and the honest list of what it cannot see all live in ONE
    place, ``replay/shard.py::sf_eval_pv_orphan_flags``; this only reduces.

    ⚑ ``checked`` is the denominator AND the blind-instrument column, same
    contract as ``sf_multipv_checked_rows``: it is zero when the host never
    derived the flags (a caller that skipped ``_prepare_host_arrays``) and when
    no row carried all three blocks. A rate above a zero ``checked`` means
    nothing was inspected — never "clean".

    ⚑ This population is DISJOINT from ``sf_multipv_presence_counts``'
    numerator by construction: that one counts rows with no MultiPV block, this
    one is only computable on rows that have one. It is the first instrument
    that looks INSIDE the set the policy detector passes.
    """
    orphaned = batch.get("sf_eval_pv_orphan")
    checked = batch.get("sf_eval_pv_checked")
    if orphaned is None or checked is None:
        ref = batch["wdl_t"] if "wdl_t" in batch else next(iter(batch.values()))
        zero = ref.new_zeros((), dtype=torch.float32)
        return zero, zero.clone()
    return orphaned.to(torch.float32).sum(), checked.to(torch.float32).sum()


def normalize_distribution(probs: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Renormalize a distribution along the last axis so each row sums to 1.

    ``clamp_min(eps)`` keeps all-zero rows finite (they stay ~0 rather than
    producing NaN), so callers can pass missing/masked rows through safely.
    """
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with a soft target distribution.

    Soft targets are *meant* to be normalized along the last axis, but the
    replay shards persist them as float16 (``policy_t``, ``policy_soft_t``,
    ``future_policy_t``, ``categorical_t``, ``sf_policy_t``). Round-tripping a
    distribution through f16 leaves each row summing to ``1 ± O(width * 2**-11)``
    instead of exactly 1 (smaller on peaked visit distributions, up to ~1e-3 on
    flatter targets like ``categorical_t`` / smoothed ``sf_policy_t``). Because
    soft CE is *linear* in the target, that residual row-sum scales the whole
    sample's loss and gradient, most impactfully on the main policy head
    (``w_policy = 1.0``). Renormalizing here removes that f16 bias and makes the
    documented invariant true for every soft-CE head.

    It is a no-op (within fp error) for already-normalized targets such as the
    post-clamp ``sf_wdl_probs`` and the convex WDL blend, and all-zero rows
    (missing/masked targets) stay zero — ``clamp_min`` keeps the division
    finite so they contribute 0 loss rather than NaN.
    """
    target_probs = normalize_distribution(target_probs)
    return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)


# Accepted `policy_target_temp` range. NEITHER endpoint is a numerical limit
# (see the guard in the function) -- both are typo catchers, set clear of any
# value we would deliberately use: the offline screen's arm is 1.5,
# `scripts/retarget_retrain.py`'s reachability probe uses 0.5/2.0, and lc0's
# `--policy-softmax-temp` has run at 1.36-2.20, so [0.5, 4.0] cannot refuse a
# value anyone would set on purpose while still catching the decimal-point class
# of mistake (`15` for `1.5`).
#
# ⚑ THAT lc0 RANGE IS NOT A TRAINING-TARGET TEMPERATURE, and it is cited here
# only as evidence that numbers of this MAGNITUDE are ones a person sets on
# purpose -- never as a recommended value for THIS knob. `--policy-softmax-temp`
# is a SEARCH-TIME PRIOR temperature: it softens the net's policy output before
# the tree uses it as a prior, leaving the training target untouched. Our direct
# analogue of it is `gumbel_policy_temp` (production 1.5, against lc0's current
# default 1.45), which is a different knob in a different file. THIS constant
# bounds a power transform applied to the STORED target at training time. Both
# get written down as "T=1.5" and they are not the same T.
_POLICY_TARGET_TEMP_MIN = 0.5
_POLICY_TARGET_TEMP_MAX = 4.0


def policy_target_temp_active(temp: float) -> bool:
    """True when ``policy_target_temp`` will actually reshape the target.

    THE single definition of "the reshape is on". ``retemper_main_policy_target``
    below gates its early return on this, and ``Trainer.__init__``'s realized-value
    log line reports it, so the arithmetic and the operator-facing claim cannot
    drift apart: a guard has to share the criterion's instrument or it is guarding
    a different question.

    No validation here -- an out-of-range temperature is rejected by
    ``retemper_main_policy_target`` (and, for the live path, by ``Trainer.__init__``
    calling it at construction). This predicate answers only "does it bite".
    """
    return float(temp) != 1.0


def retemper_main_policy_target(pol_target: torch.Tensor, *, temp: float) -> torch.Tensor:
    """Flatten the MAIN policy target by a temperature. IDENTITY at ``temp == 1.0``.

    ⚑ THE MECHANISM IS UNPROVEN AND THIS IS PLUMBING ONLY. An earlier revision of this
    docstring motivated the knob with the fixed point: the target was measured
    (2026-08-08) to carry far more CONFIDENCE than INFORMATION — 256 sims of search
    improve ranking concordance over the net's own prior by only +0.0077
    [+0.0065, +0.0088] while dropping entropy 1.185 -> 1.104 nats — and every term in
    the target is a function of the net, so CE against it converges to a fixed point.
    **That argument does not survive, and it does not survive because of this
    transform.** ``p ** (1/T)`` is a deterministic monotone function of the same
    target, so the retempered target is ALSO entirely a function of the net and adds
    exactly zero information. A knob cannot break a self-reference by reshaping one
    side of it. See the ledger entry and
    memory `kl_target_prior_is_the_training_signal`.

    ⚑ AN EARLIER REVISION OF THIS PARAGRAPH ALSO CLAIMED THE TRANSFORM "leaves
    ``KL(target‖prior)`` unchanged". **THAT IS FALSE** (review, codex P2), and it is
    false in the direction that matters: a power transform followed by a renormalise
    moves the target toward uniform for ``T > 1``, and a broad prior is nearer uniform
    than a peaked target is, so the KL FALLS. Measured on a sparse fixture
    (target ``[.90 .05 .03 .01 .005 .005]``, prior ``[.40 .25 .15 .10 .06 .04]``):

        T      1.0      1.3      1.5      2.2      4.0      0.7
        KL   0.5552   0.3380   0.2322   0.0609   0.0563   0.7891

    Nothing in the argument above depends on the retracted claim — the fixed-point
    framing dies on "deterministic function of the same target", not on KL invariance
    — so the withdrawal stands and gets no weaker. But the standing finding's tracked
    QUANTITY does move under this knob, which is a thing anyone reading
    `kl_target_prior_is_the_training_signal` beside this docstring has to know: do not
    use ``KL(target‖prior)`` as an arm-invariant number here.

    What is left is a DIFFERENT and narrower mechanism, which is what this actually is:
    label smoothing. It does not add information; it reduces the overconfidence
    pressure on the fitted solution, which is a claim about optimisation and
    calibration rather than about the information content of the target. That claim is
    plausible and UNTESTED here. Nothing in this PR measures it, so nothing downstream
    should cite this docstring as evidence for it — the entry in
    docs/experiment_ledger.md carries the hypothesis, the single deciding yardstick and
    the kill threshold, and no arm may run at ``temp != 1.0`` without it.

    Accepted range is ``[0.5, 4.0]`` (see the guard); both endpoints are typo
    catchers rather than numerical limits, and the ceiling in particular is NOT
    optional — over-flattening is invisible on the pinned eval ruler, so nothing
    downstream would notice a dropped decimal point.

    ``temp`` > 1 flattens (``p ** (1/temp)``, renormalised); < 1 sharpens. It
    deliberately does NOT resurrect zeros: a zero stays zero under a power, and the
    moves this target zeroes were measured to lose a median 538cp, so raising them is
    not the goal. The goal is to stop asserting near-certainty AMONG THE MOVES THAT
    CARRY MASS.

    ⚑ THIS RESHAPES WHAT THE MODEL IS TRAINED ON, so it moves the FLOOR of any CE
    computed against it: ``CE = H(target) + KL(target‖model)``, and flattening raises
    ``H``. The DIRECTION of the total is NOT fixed — flattening can lower ``KL`` by
    more than it raises ``H``, and on a randomly-initialised fixture `policy_ce` was
    measured to FALL (4.17504 -> 4.17148 at temp 1.3, review #2). Do not quote a
    magnitude without naming the model and fixture it was measured on. The argument for
    the eval pin needs only that the ruler MOVES, which it does in either direction:
    `Trainer._eval_loss_kwargs` pins temp to 1.0 for holdout/EMA eval so the eval CE
    stays a fixed ruler across arms. Anything else scoring this target must do the same
    or say plainly that its number is arm-relative.

    ⚑ SECOND EFFECT, and it is not on the ``policy_own`` head. ``compute_loss``
    reassigns ``pol_target`` to the retempered target BEFORE the ``soft_policy_min_tv``
    gate is computed from it, so this temperature also decides which rows the
    ``policy_soft`` head trains on: a flatter hard target sits closer to the soft one,
    the TV falls below the threshold, and the row is dropped (measured kept_frac
    1.000 -> 0.000 across temp 1.0 -> 1.3 on a fixture straddling the threshold; see
    tests/test_policy_target_reshape.py). Latent while ``soft_policy_min_tv`` is 0.0 --
    which in ``configs/pbt2_small.yaml`` means ABSENT rather than set: the key is not in
    the production yaml at all (grep exits 1 on the repo copy and on the live file), and
    the 0.0 is this function's own default below. Only
    ``configs/exp_soft_policy_divergent_only.yaml`` sets it. Note the asymmetry if both
    are ever on: the eval pin
    fixes the temperature but NOT ``soft_policy_min_tv``, so training and eval would
    mask different row sets for that head.

    ⚑ NOT expressible as a loss weight. Soft CE is LINEAR in the target, so mixing a
    second distribution into the target is identical to adding a weighted CE term
    (which is what ``w_sf_own`` already does for the same-position SF label). A power
    transform is not, which is why this one needs code and an SF blend does not.
    """
    t = float(temp)
  # ⚑ BOTH ENDPOINTS ARE TYPO CATCHERS, not numerical necessities — the
  # max-scaling below already makes any positive temperature finite. They exist
  # because the two directions FAIL DIFFERENTLY, and NEITHER of them fails
  # loudly on the channel an operator watches:
  #   * over-SHARPENING drives the target toward one-hot, which LOWERS the CE
  #     floor toward zero, so the mistake reads as the loss IMPROVING. Review
  #     found `policy_target_temp: 0.001` doing exactly that.
  #   * over-FLATTENING drives it toward uniform over its support, which raises
  #     the CE floor. An earlier revision of this comment claimed that "gets
  #     noticed" and so needed no ceiling. ⚑ THAT WAS WRONG, and it was wrong
  #     for the reason this whole knob exists: `_eval_loss_kwargs` PINS eval to
  #     1.0, so the holdout `policy_ce` an operator actually watches does NOT
  #     move. Only the train-side `policy_loss` does, on an arm that was
  #     launched expecting it to move. `policy_target_temp: 15` (a dropped
  #     decimal point) is therefore silently wrong, not loudly wrong — a
  #     ceiling is the only thing that catches it.
  # `not (t >= MIN)` also catches NaN, which compares False against everything;
  # `+inf` fails the `<= MAX` arm, so the range check subsumes the old explicit
  # infinity test.
    if not (_POLICY_TARGET_TEMP_MIN <= t <= _POLICY_TARGET_TEMP_MAX):
        raise ValueError(
            f"policy_target_temp must be finite and in "
            f"[{_POLICY_TARGET_TEMP_MIN}, {_POLICY_TARGET_TEMP_MAX}], got {temp!r}. "
            "It is an exponent denominator: 0 divides by zero, a negative value "
            "inverts the target's ordering while still summing to 1, a small "
            "positive value collapses the target toward one-hot -- which lowers "
            "`policy_ce` and so reads as the loss improving -- and a large value "
            "drives it to uniform over its support while the PINNED eval ruler "
            "shows nothing. None of them fail loudly."
        )
    if not policy_target_temp_active(t):
        return pol_target
  # ⚑ DIVIDE BY THE ROW MAX BEFORE THE EXPONENT. The obvious `p ** (1/t)`
  # underflows: review found that `policy_target_temp: 0.001` -- a plausible
  # typo -- drove every entry of a broad target to 0 in fp32. The renormalise
  # then divided 0 by a clamped denominator instead of raising, so the target
  # was all-zero, `policy_ce` was EXACTLY 0.0, and the policy head trained on
  # nothing while the loss read as perfect. Eval is pinned to temp 1.0, so the
  # holdout could not see it either.
  #
  # Scaling by the max first makes that impossible rather than merely illegal:
  # the largest entry is exactly 1.0 after the power, so the pre-normalise sum
  # is >= 1.0 at every temperature and no epsilon clamp can ever bind. The
  # result is unchanged where the naive form was already safe (max abs diff
  # 1.2e-7 at t=0.5/1.3/2.2 -- float32 rounding), and extreme sharpening now
  # degrades to a ONE-HOT target instead of an empty one. A power transform is
  # scale-free, so dividing by a positive constant cannot change the output.
    scaled = pol_target.clamp_min(0.0)
    peak = scaled.amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return normalize_distribution((scaled / peak) ** (1.0 / t))


def align_policy_target(target: torch.Tensor, width: int) -> torch.Tensor:
    """Return a policy target in the same action encoding width as logits.

    Replay shards are allowed to remain in the full 4672 action space while a
    model emits compact LC0-1858 logits. Compact projection gathers the valid
    full actions and renormalizes because invalid 4672 padding can carry tiny
    mass from smoothing or legacy producers.
    """
    src_width = int(target.shape[-1])
    dst_width = int(width)
    if src_width == dst_width:
        return target
    if src_width == POLICY_SIZE and dst_width == COMPACT_POLICY_SIZE:
        out = target.index_select(-1, _compact_to_full_index_for(target))
        return normalize_distribution(out)
    if src_width == COMPACT_POLICY_SIZE and dst_width == POLICY_SIZE:
        out = target.new_zeros((*target.shape[:-1], POLICY_SIZE))
        out.index_copy_(-1, _compact_to_full_index_for(target), target)
        return out
    raise ValueError(f"policy target width {src_width} is incompatible with logits width {dst_width}")


def align_policy_mask(mask: torch.Tensor, width: int) -> torch.Tensor:
    """Return a legal-move mask in the same action encoding width as logits."""
    src_width = int(mask.shape[-1])
    dst_width = int(width)
    if src_width == dst_width:
        return mask
    if src_width == POLICY_SIZE and dst_width == COMPACT_POLICY_SIZE:
        return mask.index_select(-1, _compact_to_full_index_for(mask))
    if src_width == COMPACT_POLICY_SIZE and dst_width == POLICY_SIZE:
        out = mask.new_zeros((*mask.shape[:-1], POLICY_SIZE))
        out.index_copy_(-1, _compact_to_full_index_for(mask), mask)
        return out
    raise ValueError(f"policy mask width {src_width} is incompatible with logits width {dst_width}")


def align_action_values(values: torch.Tensor, width: int) -> torch.Tensor:
    """Reindex a per-action VALUE vector (e.g. cp-regret) to the logits' width.

    Unlike :func:`align_policy_target`, this does NOT renormalize: the vector is
    not a probability distribution, so a 4672->1858 compact projection must
    gather the valid actions and leave their magnitudes untouched. The reindex
    is identical to a legal-mask reindex, so we delegate to that path.
    """
    return align_policy_mask(values, width)


def apply_policy_mask_to_logits(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    mask_key: str,
    has_key: str,
) -> torch.Tensor:
    """Policy-specific mask application that supports full/compact encodings."""
    mask = batch.get(mask_key)
    if mask is None:
        return logits
    mask = align_policy_mask(mask, int(logits.shape[-1]))
    has = batch.get(has_key)
    active = has.unsqueeze(-1) if has is not None else 1.0
    return logits + (1.0 - mask) * -1e9 * active


def _huber_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(pred, target, delta=0.1, reduction="none").mean(dim=-1)


def _get_mask(batch: dict[str, torch.Tensor], key: str, *, default: float = 0.0) -> torch.Tensor:
    """Get a per-sample mask from batch, defaulting to a constant tensor."""
    v = batch.get(key)
    if v is not None:
        return v
    return torch.full((batch["x"].shape[0],), default, device=batch["x"].device)


def policy_legal_bool(batch: dict[str, torch.Tensor], *, width: int) -> torch.Tensor | None:
    """The boolean legal-move support of ``apply_policy_mask_to_logits``.

    Returns ``None`` when the batch carries no ``legal_mask`` at all, which is
    the same case in which that function returns the logits untouched — every
    action is then in the softmax's support and the caller must treat every
    action as legal.

    ⚑ THE `has_legal_mask` CONVENTION IS NOT `_get_mask`'s. A batch WITHOUT the
    ``has_`` vector is fully masked (``active = 1.0``), while ``_get_mask``
    would default it to 0.0 and call every action legal. Reading the flag the
    convenient way would hand this term a support the softmax does not have --
    illegal moves with a real regret entry could then be floored, which is the
    one thing the ``regret <= delta_cp/CAP`` window is relied on to prevent.
    ``tests/test_sf_policy_floor.py`` pins this function against the logit
    masker's own output rather than against a second reading of the rule.
    """
    mask = batch.get("legal_mask")
    if mask is None:
        return None
    aligned = align_policy_mask(mask, width) > 0.5
    has = batch.get("has_legal_mask")
    if has is None:
        return aligned
    return aligned | (has.unsqueeze(-1) <= 0.5)


# Default floor for the SF-approved set. Calibrated, not a round number: see
# `SfPolicyFloorParams.resolve` for the two separable roles of tau and why this
# sits ABOVE the deterministic prior-rank threshold rather than at it.
SF_POLICY_FLOOR_TAU_DEFAULT = 0.15

# ⚑⚑ THE FLOOR THRESHOLDS HAVE THEIR OWN DTYPE, PINNED, AND IT IS NOT `probs`'.
# The feasibility rule is a statement about the numbers the loss ACTUALLY
# materializes, so config validity must not depend on what dtype some future
# policy softmax happens to produce. Pinning it here means the value
# `SfPolicyFloorParams` validates and the value `sf_policy_floor_deficit`
# subtracts from `probs` are the same representation, by construction rather
# than by coincidence. (`po_probs` is fp32 in production today -- softmax is on
# autocast's fp32 list -- so this is currently a no-op that stays a no-op.)
_FLOOR_THRESHOLD_DTYPE = torch.float32


def _as_floor_threshold(value: float) -> float:
    """`value` as the loss will actually represent it, widened back to a double.

    Configuration arrives as Python doubles and is validated as doubles, but the
    loss materializes it in `_FLOOR_THRESHOLD_DTYPE`. Those disagree, and near
    the feasibility boundary the disagreement is the whole question:
    `0.6 + 0.4 == 1.0` exactly as doubles, while the float32 pair sums to
    1.0000000298 -- an infeasible mandatory set that a double-precision
    validator waves through. Round-tripping through the consumer's own dtype is
    what makes the validator answer the question the loss will ask.
    """
    return float(torch.tensor(float(value), dtype=_FLOOR_THRESHOLD_DTYPE))


def search_inclusion_guarantee_tau(gumbel_topk: object) -> float:
    """The smallest tau that buys a top-k slot BY PRIOR RANK: ``1 / topk``.

    The root sampler (`mcts/gumbel._select_top_m_with_gumbel`) keeps the top
    ``m`` of ``gumbel_scale * Gumbel + log(prior)``, where

        m = 1                                            if sims <= 1
        m = max(2, min(topk, n_legal, max(2, (sims+1)//2)))  otherwise

    ⚑⚑ ``1/topk`` IS AN EXACT DETERMINISTIC PRIOR-RANK THRESHOLD, NOT A
    STOCHASTIC ADMISSION GUARANTEE. If ``p_i >= tau`` then at most
    ``floor(1/tau) - 1`` other moves can exceed it, so ``tau >= 1/topk`` puts
    move ``i`` inside the top-``topk`` **by prior**. The sampler instead ranks
    ``gumbel_scale * Gumbel + log(prior)``, and Gumbel noise has UNBOUNDED
    support, so no prior threshold can make noisy admission certain.

    ⚑ AND THE KNOB THIS DOCSTRING USED TO CITE WAS THE WRONG ONE. It justified
    near-determinism with "``gumbel_c_scale = 0.1``, noise ~0.13 in log-prob
    units". ``gumbel_c_scale`` is the SEARCH's exploration constant; the root
    sampler above is scaled by ``gumbel_scale``. The two relevant trees differ,
    which is why measured numbers below name the scale rather than saying
    "production".

    MEASURED (`_select_top_m_with_gumbel`, ``add_noise=True``, a move at exactly
    ``p = 1/16``, ``n_legal = 40``), as a function of the ROOT noise scale:

    ==============  ==================  ==================
    gumbel_scale    peaked prior tail   broad/uniform tail
    ==============  ==================  ==================
    1.00            1.0000              **0.7250**
    0.75            1.0000              **0.8290**
    0.50            1.0000              0.9607
    0.00            1.0000              1.0000
    ==============  ==================  ==================

    ⚑ QUOTE THE SCALE, NEVER "production". ``configs/pbt2_small.yaml`` on this
    branch runs ``gumbel_scale: 0.75`` decaying to ``gumbel_scale_after: 0.0``,
    while the live branch runs ``1.0 -> 0.5``. The broad-tail exclusion is about
    17% and 27% respectively at the pre-decay scale; neither number is a theorem
    about real peaked priors.

    The earlier "MIN (not mean) P(searched) is exactly 1.0000 at prior >= 1/16"
    measured on 3000 real rows is not contradicted: those priors were peaked.
    It is an empirical property of that prior distribution and search config,
    NOT a bound the sampler provides.

    `tests/test_sf_policy_floor.py::test_inclusion_under_production_noise_is_not_a_guarantee`
    pins the measured noisy-admission effect, because an argument this docstring
    got wrong once should not be the thing defending it.

    ⚑ WHEN THE SIM BUDGET CAN BITE. ``1/topk`` is a statement about top-``topk``
    by prior; the search keeps top-``m``. If ``m < topk`` and ``n_legal > m``,
    the corresponding deterministic prior-rank threshold is ``1/m`` instead.
    If ``n_legal <= m``, every legal move is a candidate regardless of prior or
    Gumbel noise.

    At ``gumbel_topk: 16``, ``m < topk`` needs ``sims < 31``. The branch config's
    full and fast budgets stay at or above that, so its deterministic threshold
    is still ``1/16`` at those budgets. **That says nothing about stochastic
    admission under positive Gumbel noise.**

    ⚑ THE FLOOR IS ON THE RAW PROBABILITY; THE SEARCH SELECTS ON THE TEMPERED
    ONE (``logits / gumbel_policy_temp``). The two bars happened to coincide at
    one measured operating point; that is not an identity. Re-derive it if
    ``gumbel_policy_temp``, ``gumbel_topk`` or the policy encoding changes.
    """
    return 1.0 / float(normalize_gumbel_topk(gumbel_topk))


def warn_if_below_search_inclusion(
    *, tau: float, tau_top1: float, tau_played: float,
    gumbel_topk: object, context: str,
) -> float:
    """Warn when a floor drops below the deterministic prior-rank threshold.

    This guard is deliberately NOT a noisy-search admission guarantee. It checks
    whether each floor is at least ``1/gumbel_topk``, which is enough to put a
    move inside the deterministic top-k **by prior rank** when the sim budget
    allows ``m == topk``. Under positive Gumbel noise admission remains
    probabilistic even above the threshold.

    All three thresholds are checked because each can be the only floor applied
    to a move on some row. ``tau_played == 0.0`` stays silent because that is the
    documented collar-ablation arm; zero is not a corresponding ablation for
    ``tau`` or ``tau_top1``.

    Returns the deterministic rank threshold so callers can report it without
    re-deriving it. The internal helper name is historical; operator-facing
    messages must call this a threshold, never an inclusion guarantee.
    """
    guarantee = search_inclusion_guarantee_tau(gumbel_topk)
    topk = normalize_gumbel_topk(gumbel_topk)
    for name, value, silent_at_zero in (
        ("sf_policy_floor_tau", float(tau), False),
        ("sf_policy_floor_tau_top1", float(tau_top1), False),
        ("sf_policy_floor_tau_played", float(tau_played), True),
    ):
        if value >= guarantee or (silent_at_zero and value == 0.0):
            continue
        warnings.warn(
            f"{name}={value!r} is BELOW the deterministic prior-rank threshold "
            f"1/gumbel_topk={guarantee!r} (gumbel_topk={topk}, {context}). "
            "Prior rank alone therefore no longer puts the floored move in the "
            "top-k candidate band. No tau guarantees admission under Gumbel "
            "noise; this warning is only about deterministic prior-rank coverage. "
            "Raise the floor, or widen gumbel_topk, if falling below that "
            "threshold was not intended.",
            RuntimeWarning,
            stacklevel=3,
        )
    return guarantee


@dataclass(frozen=True)
class SfPolicyFloorParams:
    """Resolved, validated parameters of the SF-approved-move probability floor.

    ONE object, resolved once, and the same object the loss consumes -- so the
    value that gets logged cannot be a second derivation of the value that gets
    used. (`docs/rl_loop_audit.md`: "announce from the consumer's own
    parameter". A resolve-and-print that re-derives the default at the call site
    passes every wiring test we have and is still wrong.)

    ``w`` 0.0 (the default) means the term contributes NOTHING to ``total`` --
    not "multiplied by zero", see ``compute_loss``. The diagnostic columns are
    still computed, which is the point: the binding rate is readable before the
    weight is ever raised.
    """

    w: float = 0.0
    delta_cp: float = 20.0
    tau: float = SF_POLICY_FLOOR_TAU_DEFAULT
    tau_top1: float = SF_POLICY_FLOOR_TAU_DEFAULT
    tau_played: float = 1.0 / DEFAULT_GUMBEL_TOPK

    def __post_init__(self) -> None:
        for name, value, hi in (
            ("w_sf_policy_floor", self.w, None),
            ("sf_policy_floor_delta_cp", self.delta_cp, None),
            ("sf_policy_floor_tau", self.tau, 1.0),
            ("sf_policy_floor_tau_top1", self.tau_top1, 1.0),
            ("sf_policy_floor_tau_played", self.tau_played, 1.0),
        ):
            val = float(value)
            bad = not math.isfinite(val) or val < 0.0 or (hi is not None and val > hi)
            if bad:
                band = "[0, 1]" if hi is not None else ">= 0"
                raise ValueError(f"{name} must be finite and {band}, got {value!r}")
  # ⚑ THE MANDATORY ROLES MUST FIT IN A PROBABILITY BUDGET OF 1, AS REPRESENTED.
  # The rule is not raw `max(tau, tau_top1) + tau_played <= 1` on the configured
  # Python doubles -- each OPERAND is first rounded to `_FLOOR_THRESHOLD_DTYPE`,
  # because that is what the loss materializes and therefore what feasibility is
  # a statement about. ⚑ THE ADDITION THEN STAYS IN DOUBLE, and that is not a
  # detail: it is what makes this bit-exact against the loss's own float64
  # `mandatory.sum()` over float32 entries. Adding in float32 instead would round
  # 0.6+0.4, 0.9+0.1 and 1/3+2/3 all back to a clean 1.0 and REOPEN the hole. `0.6 + 0.4` is exactly 1.0 as doubles and 1.0000000298 as
  # float32; only the second is the question being asked. Every other
  # member of F is optional -- `sf_policy_floor_deficit` admits them in
  # ascending SF regret and stops before the budget is exceeded -- but SF's
  # top-1 and the played-move collar are structural and are ALWAYS applied, so
  # a configuration whose two mandatory floors alone cannot coexist is
  # unsatisfiable on some row no matter what the net does. The worst case is
  # top1 and played being DISTINCT moves with top1 also inside the adaptive
  # window, where the max rule floors it at `max(tau, tau_top1)`. Rejected at
  # resolve time rather than clamped at loss time, because a floor the net can
  # never clear is a permanent gradient the operator did not ask for.
  # ⚑ ROUNDED TO THE CONSUMER'S DTYPE FIRST. Validating the Python doubles
  # answers a different question than the one the loss asks: `0.6 + 0.4` is
  # exactly 1.0 as doubles and 1.0000000298 as float32, so a double-precision
  # check accepts a mandatory pair that is INFEASIBLE once materialized. And
  # this failure is invisible downstream -- neither role is ever truncated (the
  # cap only drops OPTIONAL members), and `applied_mass` narrows 1.0000000298
  # back to a clean `1.0`, so every counter reads healthy over a false
  # invariant. Reviewer finding, PR #448.
        tau_r = _as_floor_threshold(self.tau)
        top1_r = _as_floor_threshold(self.tau_top1)
        played_r = _as_floor_threshold(self.tau_played)
        mandatory = max(tau_r, top1_r) + played_r
        if mandatory > 1.0:
            msg = (
                "max(sf_policy_floor_tau, sf_policy_floor_tau_top1) + "
                f"sf_policy_floor_tau_played = {mandatory!r} exceeds 1.0 as "
                f"{_FLOOR_THRESHOLD_DTYPE} (tau={self.tau!r}, "
                f"tau_top1={self.tau_top1!r}, tau_played={self.tau_played!r}); "
                "the mandatory top-1 and collar floors cannot both be satisfied "
                "on a row where they land on different moves. Neither role is "
                "ever truncated, so this cannot be repaired at loss time. "
                "⚑ If no sf_policy_floor_tau_played was configured it is DERIVED "
                "as 1/gumbel_topk, so the key to change is probably gumbel_topk."
            )
  # ⚑⚑ AN INERT TERM MUST NOT BE ABLE TO KILL THE RUN. At `w == 0.0` this term
  # contributes nothing to `total` (`compute_loss` adds it under an `if`), so
  # raising here would take down a trial over a configuration that changes no
  # objective. And it is REACHABLE WITHOUT ANY sf_policy_floor KEY BEING SET:
  # `tau_played` defaults to `1/gumbel_topk`, and `gumbel_topk: 1` makes it 1.0,
  # so the shipped 0.15/0.15 defaults sum to 1.15. `gumbel_topk` is a LIVE
  # selfplay key, so that lands as a CLAUDE.md category (b) death --
  # `_reload_yaml_into_config` succeeds, `from_dict` raises inside the iteration
  # loop, and the loop has a `finally:` and zero `except:`. `sync_search_width`
  # reaches the same constructor through `dataclasses.replace`.
  #
  # So the raise is gated on the term actually being in the objective. The
  # warning still fires at `w == 0.0`, because the DIAGNOSTIC columns are
  # computed either way and would be reporting an infeasible set.
  # Reviewer finding P2-1, PR #448.
            if float(self.w) != 0.0:
                raise ValueError(msg)
  # ⚑ `stacklevel=3` is right for `resolve()` and WRONG via `dataclasses.replace`,
  # where it blames `dataclasses.py`. Left as is deliberately: the two entry paths
  # have different depths so NO single value is correct for both, and the message
  # already names every key an operator needs -- the attribution line would not
  # change what they do. Decided, not deferred (reviewer P3-E, PR #448).
            warnings.warn(
                f"{msg} The term is OFF (w_sf_policy_floor=0.0), so this is a "
                "warning rather than a fatal error -- but the sf_policy_floor_* "
                "diagnostic columns will describe an infeasible set, and raising "
                "the weight will make it fatal.",
                RuntimeWarning,
                stacklevel=3,
            )

    @classmethod
    def resolve(
        cls,
        *,
        w: float | None = None,
        delta_cp: float | None = None,
        tau: float | None = None,
        tau_top1: float | None = None,
        tau_played: float | None = None,
        gumbel_topk: int = DEFAULT_GUMBEL_TOPK,
    ) -> SfPolicyFloorParams:
        """Fill defaults, validate, and check the deterministic rank threshold.

        ``tau`` has two separable roles. ``1/gumbel_topk`` is the deterministic
        prior-rank threshold: above it, prior rank alone places the move inside
        top-k when ``m == topk``. It is NOT a noisy-admission guarantee. Tau above
        that threshold additionally changes ranking pressure; the calibrated
        default remains 0.15.

        A resolved floor below ``1/topk`` warns rather than raises because a
        deliberate sub-threshold floor is a legitimate experiment. The warning
        says exactly what is lost: deterministic prior-rank coverage, not
        stochastic search inclusion.
        * ``tau = 1/gumbel_topk`` (0.0625 at the production topk of 16) is the
          SEARCH-INCLUSION GUARANTEE -- see ``search_inclusion_guarantee_tau``.
          BELOW it the guarantee is simply lost.

          ⚑ OBSERVATION, RECORDED AND NOT ACTED ON (2026-08-18): if the term's
          purpose is SEARCH ADMISSION -- keeping SF's move above the Gumbel
          candidate threshold so it can be refuted -- then 0.0625 is what that
          purpose requires and the shipped ``sf_policy_floor_tau: 0.15`` is
          **2.40x past it**. The default is deliberately NOT changed here: the
          ranking role below is the reason it was set above the guarantee, and
          re-deciding between the two roles is a ledger entry of its own, not a
          drive-by edit inside an unrelated change.
        * tau ABOVE that buys RANKING on the rows where inclusion is already
          satisfied: +0.278 [+0.237, +0.322] against the random-floor control at
          tau 0.15, +0.548 at 0.35, and 0.0% harmful rows at both -- it costs
          nothing where we are already right.

        ``tau_played`` -- the collar -- defaults to ``1/topk`` because its job is
        to keep the played move above that deterministic prior-rank bar while
        the SF floor redistributes mass. Under Gumbel noise actual admission is
        still empirical. ``tau_played: 0.0`` disables the collar as a clean
        ablation arm.

        ``gumbel_topk`` is the width the trial ACTUALLY runs with, so callers
        must pass their own rather than let it default.

        ⚑⚑ AND IT CAN RAISE, WHICH THE REST OF THIS DOCSTRING USED NOT TO SAY.
        ``__post_init__`` refuses a config whose two MANDATORY floors cannot
        coexist -- ``max(tau, tau_top1) + tau_played > 1`` once each operand is
        rounded to the dtype the loss materializes. That is the one behaviour here
        that can take down a trial, and it was documented nowhere, while four
        sentences above describe the OTHER, harmless warn-not-raise decision.

        The raise is gated on ``w != 0.0``: an inert term warns instead, because a
        term contributing nothing to ``total`` must not be able to kill a run. ⚑ It
        is reachable with NO ``sf_policy_floor_*`` key set at all -- ``tau_played``
        derives from ``1/gumbel_topk``, and ``gumbel_topk: 1`` makes it 1.0, so the
        shipped defaults sum to 1.15. Re-validation on ACTIVATION happens at
        ``Trainer._loss_kwargs``, which rebuilds this object with
        ``dataclasses.replace`` on every read -- NOT at the live-push site, which
        does a bare ``setattr`` and validates nothing.
        """
        resolved_tau = SF_POLICY_FLOOR_TAU_DEFAULT if tau is None else float(tau)
        guarantee = search_inclusion_guarantee_tau(gumbel_topk)
        params = cls(
            w=0.0 if w is None else float(w),
            delta_cp=20.0 if delta_cp is None else float(delta_cp),
            tau=resolved_tau,
            # Falls back to the RESOLVED `tau`, not to the dataclass default, so
            # `tau_top1: null` tracks an explicit `tau` instead of silently
            # flooring SF's own best move at a different bar than the rest of F.
            tau_top1=resolved_tau if tau_top1 is None else float(tau_top1),
            # The collar defaults to the deterministic prior-rank threshold.
            # It does not make noisy admission certain; 0.0 disables the collar.
            tau_played=guarantee if tau_played is None else float(tau_played),
        )
  # AFTER the range check, deliberately: an out-of-range value must RAISE before
  # the threshold warning, so the log names the actual configuration error first.
        warn_if_below_search_inclusion(
            tau=params.tau, tau_top1=params.tau_top1, tau_played=params.tau_played,
            gumbel_topk=gumbel_topk, context="at config load",
        )
        return params


class SfPolicyFloorOutputs(NamedTuple):
    """Per-row outputs of ``sf_policy_floor_deficit``. All shaped ``(B,)``.

    ``deficit`` and ``binds`` are the loss and the binding indicator. The five
    that follow are DIAGNOSTIC ONLY -- nothing multiplies them into ``total`` --
    and they exist to make the feasibility cap readable rather than silent:

    * ``member_count_raw`` / ``requested_mass`` -- the set and the probability
      mass the UNCAPPED rule would have demanded, i.e. the size of the demand
      the cap had to cut down. ⚑ NOT an infeasibility test -- see below.
    * ``truncated`` -- 0/1, did the cap actually drop a member on this row.
    * ``member_count_applied`` / ``applied_mass`` -- the set and mass after the
      cap. ``applied_mass <= 1`` EXACTLY, not to within a slack -- ⚑ CONDITIONAL
      on ``w != 0.0``; at weight 0 an infeasible MANDATORY pair is permitted with
      a warning and these columns then correctly describe an impossible floor,
      where the resolve-time WARNING rather than ``truncated`` is the signal. The
      admission
      budget is compared in float64 with no positive epsilon.

    ⚑ TWO THINGS THESE COLUMNS DO NOT MEAN.
    (1) `requested_mass > 1` is SUFFICIENT to expect truncation, not NECESSARY:
        the admission test is exact (float64) but the mass columns are narrowed to
        float32, so ten floors of 0.1 (true sum 1.0000000149) truncate while the
        column reads 1.0.
    (2) ⚑⚑ AND THESE ARE ROW MEANS over `sf_own_regret_rows`, not per-row values,
        so `requested_mass > 1.0` is nearly UNREACHABLE at the column level even
        when a large minority of rows are infeasible -- measured 0.552 on a batch
        whose `truncated_frac` was 0.333.
    ⇒ `truncated_frac` is the ONLY column that answers "did the cap fire".

    ⚑ READ ``requested_mass`` AGAINST ``applied_mass``, NOT ALONE. The whole
    point of the pair is to answer, after the fact, whether F's flattening was
    driven by the hidden ``|F| * tau`` strength rather than by the tau the
    operator set -- a question a single column cannot answer.
    """

    deficit: torch.Tensor
    binds: torch.Tensor
    member_count_raw: torch.Tensor
    requested_mass: torch.Tensor
    truncated: torch.Tensor
    member_count_applied: torch.Tensor
    applied_mass: torch.Tensor


def sf_policy_floor_deficit(
    probs: torch.Tensor,
    regret: torch.Tensor,
    legal: torch.Tensor | None,
    played_target: torch.Tensor | None = None,
    *,
    params: SfPolicyFloorParams,
) -> SfPolicyFloorOutputs:
    """One-sided probability floor on SF-approved moves. See ``SfPolicyFloorOutputs``.

    ``deficit`` is per-row ``sum_{m in F} relu(tau_m - p_m)``; ``binds`` is the
    per-row 0/1 indicator that at least one move in ``F`` was actually below its
    floor. NEITHER is masked to the covered rows -- the caller applies its own
    row mask, exactly as the other per-sample losses here do.

    The set, recovered from ``sf_p0_regret`` alone (no new shard field):

        top1 = argmin over LEGAL moves of regret          # SF's best
        F    = {top1} u {m : regret_m <= delta_cp/CAP AND regret_m < regret_ours}

    with ``ours = argmax p``. Three properties, each pinned by a test:

    1. ``top1`` IS IN F UNCONDITIONALLY. That is the whole mechanism: MCTS can
       never learn about a move it never expands, so the floor raises SF's move
       toward search visibility. Crossing ``1/topk`` guarantees only its
       deterministic prior rank; Gumbel-noisy admission remains probabilistic.
    2. THE `regret_m < regret_ours` CLAUSE, STRICT. When our argmax already IS
       SF's best the second set is EMPTY, so ``F = {our own move}`` and the term
       is either silent (p >= tau) or pushes mass ONTO the move we got right --
       it can never drag mass off a correct pick. Strict ``<`` because cp scores
       are integer-quantised and ties are common; ``<=`` would floor a move
       merely EQUAL to ours.
    3. ``regret <= delta_cp/CAP`` ALONE EXCLUDES UNSURFACED AND ILLEGAL MOVES.
       `_build_sf_p0_regret_vector` fills both with ``(worst_surfaced + 1) / 2``,
       which is ``>= 0.5`` always, against ``delta_cp/CAP = 0.02`` at 20cp. The
       legality term in ``adaptive`` below is therefore belt-and-braces at the
       production delta, and load-bearing only if someone sets
       ``delta_cp >= 500`` -- which is why the test that isolates it does.

    A FOURTH member, the COLLAR, is added when ``played_target`` is given: the
    argmax of the SEARCH's own policy target, floored at ``tau_played``. It is
    not part of F -- it is the counterweight. The mass the floor adds comes out
    of every non-member proportionally and the biggest absolute loser is our own
    top move, so an uncollared floor can push the move search actually PLAYED
    below the deterministic rank bar. See ``SfPolicyFloorParams.resolve``.

    ⚑ IT IS THE `policy_target` ARGMAX, NOT THE NET'S. The net's argmax is by
    construction the highest-probability move and essentially cannot be squeezed
    out; the PLAYED move can, and precisely when search chose something the raw
    net ranked lower. The two differ often -- the played move matches SF's label
    best on only 43.0% of rows.

    ⚑ A MOVE IN TWO ROLES GETS ONE FLOOR, AT THE MAX OF ITS THRESHOLDS. The
    played move can also be SF's top-1, or sit inside the window and beat our
    pick; summing the thresholds would floor it at their sum, and letting the
    first one win would silently drop the higher bar. So the per-move threshold
    is a running MAX and there is exactly one ``relu`` per move.

    ⚑ THE MAX RULE MAKES `tau_top1 < tau` ASYMMETRIC, AND THE ASYMMETRY RUNS THE
    UNCOMFORTABLE WAY. SF's top-1 enters ``adaptive`` only when our argmax is
    something else (the strict ``regret < our_r`` clause excludes it from its own
    comparison), so with ``tau_top1 = 0.10`` and ``tau = 0.50``:

    * our argmax IS SF's best -> ``adaptive`` is empty -> its threshold is
      ``tau_top1`` alone, **0.10**;
    * our argmax is WRONG -> it joins ``adaptive`` -> threshold
      ``max(tau, tau_top1)`` = **0.50**.

    So a ``tau_top1`` below ``tau`` floors SF's best move LOWER on exactly the
    rows invariant 1 calls the whole mechanism -- the ones where we already
    picked it and it is merely under-weighted. That is a real property of the
    knob, not a paradox: it is inert at the shipped default (``tau_top1: null``
    resolves to ``tau``), and it is stated rather than denied because an earlier
    version of this docstring argued the reverse from the circular premise that
    the floor top-1 "already earns" is ``tau_top1``. Set ``tau_top1 >= tau`` if
    you want a strictly stronger floor on SF's best move.

    ⚑⚑ F IS CAPPED SO THE CONSTRAINT SET IS NEVER EMPTY -- WHENEVER THE TERM IS IN
    THE OBJECTIVE. (At ``w == 0.0`` an infeasible MANDATORY pair is admitted with
    a warning instead of refused, so an inert term cannot kill a live trial; the
    cap cannot repair that case and is not claimed to, because it only ever drops
    OPTIONAL members. See ``SfPolicyFloorParams``.) The floors are a set of
    simultaneous lower bounds on a distribution; if their sum exceeds 1 NO
    distribution satisfies them, so ``relu(tau - p)`` leaves a residual on that
    row at every step forever and the gradient can never resolve. The uncapped
    rule had no such bound: ``|F|`` is data-dependent and the requested mass is
    ``~|F| * tau``.

    The cap does NOT rescale the thresholds -- that would move the calibrated
    bar on rows that were already satisfiable. It shrinks the SET instead:

    1. the two MANDATORY roles are paid first -- SF's top-1 and the played-move
       collar, NEITHER of which is ever truncated -- and ``SfPolicyFloorParams``
       refuses (only when ``w != 0.0``; otherwise warns) at resolve time any config
       where those two alone can exceed 1;
    2. the remaining adaptive members are admitted in ASCENDING SF REGRET, so
       the moves SF likes best are the ones that survive;
    3. admission stops before the running total would exceed 1.

    A move already carrying a mandatory floor costs only the DIFFERENCE
    ``max(tau, mandatory) - mandatory`` to admit, which is the same max rule
    stated as an increment; the collar is therefore never dropped, only its
    upgrade to ``tau`` can be.

    ``member_count_raw`` / ``requested_mass`` report the uncapped set and
    ``truncated`` says whether the cap bit, so the cap cannot be silent.

    ⚑ ``truncated`` IS THE EXACT ANSWER TO "DID IT FIRE"; ``requested_mass`` IS
    NOT, WITHIN ONE FLOAT32 ULP. The admission test is exact (float64 over the
    float32 floors), but the mass columns are narrowed to the probs dtype so
    they can join the float32 metric accumulators. Ten floors of ``0.15``... or,
    concretely, ten of ``0.1``: the true sum is ``1.0000000149``, the cap
    correctly drops a member, and the reported ``requested_mass`` rounds to
    ``1.0``. So "``requested_mass > 1``" is SUFFICIENT to expect truncation and
    not NECESSARY. Read ``truncated_frac`` for the event and the mass pair for
    the magnitude; a disagreement between them at the fourth decimal is this
    rounding and nothing else.

    MEASURED on 5,881 live rows (2026-08-19): ``|F|`` max 6, mean 2.487,
    infeasible fraction 0.000000. ⚑ That scan measured the ADAPTIVE set only --
    its max mass of 0.900 is ``6 * tau``; the ``requested_mass`` column here
    also carries the collar, so the same row reports up to ``0.9625``. Do not
    quote 0.900 as this diagnostic's maximum.

    ⚑⚑ AND THE MPV40 CLAIM IS A BOUND IN THE OTHER DIRECTION FROM THE ONE THIS
    COMMENT FIRST STATED. 13.1% of those rows sat AT ``|F| = 6`` and are
    RIGHT-CENSORED by the live ``sf_multipv: 6``, so they are the CANDIDATE
    population in which a seventh qualifying move could exist -- not a set that
    is known to acquire one. MultiPV is enumerated best-to-worst, so a row whose
    sixth move already fails the ``<= delta_cp`` / ``< our_r`` test gains
    nothing from moves 7..40. From this measurement alone

        0% <= P_MPV40(|F| >= 7) <= 13.1%

    and the actual rate at ``sf_multipv: 40`` is UNMEASURED. The cap is worth
    having because the upper end of that interval is not small; calling it
    "load-bearing there" would be asserting the upper bound as the value.

    ⚑ AND THAT INTERVAL IS AN APPROXIMATION, NOT A CROSS-WIDTH THEOREM. The
    prefix argument holds only if the top-six cp scores are INVARIANT to MultiPV
    width. They are not exactly: at a shared node budget MultiPV 40 searches each
    line shallower than MultiPV 6, so a borderline sixth move can cross
    ``delta_cp`` in EITHER direction and the MPV40 population is not a superset
    of a re-scored MPV6 one. So read 13.1% as the MPV6 right-censored CANDIDATE
    SHARE under a score-stability approximation. Only a measurement on
    MPV40-labelled rows settles it.

    ``legal=None`` means the batch had no legal mask and every action is in the
    softmax's support (see ``policy_legal_bool``).

    ⚑⚑ THE GRADIENT ANALYSIS, CORRECTED 2026-08-18, AND IT UNDERCUTS THE CLAIM
    THIS TERM WAS BUILT ON. ``relu(tau - p)`` has a constant derivative WITH
    RESPECT TO ``p`` -- which is true and is not where training acts. Through the
    softmax, ``dp/dz = p(1-p)``, so::

        dL/dz = -p(1 - p)      ->  VANISHES linearly as p -> 0

    the SAME vanishing that kills ``sum_m p_m * r_m``. Measured logit gradient on
    SF's best move:

    ====== ======== ============ ========== =============
    p      arm A    prob-hinge   log-hinge  log^2-hinge
    ====== ======== ============ ========== =============
    0.05   0.00515  0.04750      0.950      0.424
    0.01   0.00103  0.00990      0.990      3.629
    0.003  0.00031  0.00299      0.997      6.055
    1e-4   0.00001  0.00010      0.9999     12.874
    ====== ======== ============ ========== =============

    ⇒ this floor beats arm A by a CONSTANT ~9.7x and shares its asymptotic
    behaviour. **It does not solve the buried-move problem it exists for.** A
    LOG-SPACE hinge does, because ``dlog(p)/dz = 1 - p ~ 1``::

        L = max(0, log(tau) - log(p))        # ~constant logit pressure at any depth
        L = max(0, log(tau) - log(p)) ** 2   # pressure proportional to log-units below

    That variant is the INTENDED SUCCESSOR and is deliberately not implemented
    here: the A+F ablation is running against this exact arithmetic and changing
    it mid-window would void the readout.

    ⚑ AND ON ``tau``: if the term's remaining purpose is SEARCH ADMISSION, the
    relevant threshold is ``1/gumbel_topk = 0.0625``, against the shipped 0.15 --
    2.40x past it. A modest MARGIN (~0.07-0.08) is probably right rather than the
    bare threshold, because "1/16 is the pigeonhole threshold for a top-16" and
    "THIS implementation deterministically admits a move at p >= 1/16" are
    different claims, and only the first has been established here. The
    simulation quoted in ``search_inclusion_guarantee_tau`` measured min
    P(searched) = 1.0000 at ``tau = 1/topk``, but it simulated the SELECTION
    RULE, not the production path. Verifying the second claim against the real
    root sampler is worth doing once. The default is NOT changed here.
    """
    if legal is None:
        legal = torch.ones_like(regret, dtype=torch.bool)
    thr = params.delta_cp / SF_OWN_REGRET_CAP_CP
    # Illegal moves are pushed above every real regret (which is in [0, 1]) so
    # the argmin cannot land on one.
    #
    # ⚑ THE SAME SUBSTITUTION MAKES `our_r` PERMISSIVE, NOT RESTRICTIVE, and an
    # earlier version of this comment claimed the opposite. On a row whose argmax
    # is somehow ILLEGAL, `our_r` is 2.0, so `regret < our_r` is true of every
    # real regret and the WHOLE cp window is admitted (measured: deficit 0.98 on
    # a two-member probe row, where "admits nothing" predicts 0.49). That branch
    # is unreachable under a legal-masked softmax -- an illegal move carries
    # probability 0 and cannot be the argmax unless every legal move is also 0 --
    # and admitting the window is the better fallback if it ever were reachable.
    # It is stated correctly here because the comment is the only spec: nothing
    # tests the branch, so nothing else can contradict it.
    ranked = torch.where(legal, regret, torch.full_like(regret, 2.0))
    top1 = ranked.argmin(dim=-1, keepdim=True)
    our_r = ranked.gather(-1, probs.argmax(dim=-1, keepdim=True))

  # The per-move THRESHOLD carries the membership: it is 0.0 for a non-member,
  # and `relu(0 - p) == 0` for every probability, so no separate selection mask
  # is needed and a threshold of 0.0 (`tau_played: 0.0`, the collar ablation) is
  # a clean no-op rather than a branch.
    adaptive = (regret <= thr) & legal & (regret < our_r)

  # MANDATORY floors first, and SEPARATELY from the adaptive ones, because the
  # feasibility cap below may drop an adaptive member and may never drop these.
  # `scatter_reduce(amax)` rather than a sum: a move in two roles gets ONE
  # floor, at the max of its thresholds (see the docstring).
  # ⚑ THE THRESHOLDS ARE MATERIALIZED IN `_FLOOR_THRESHOLD_DTYPE`, NOT
  # `probs.dtype`. That is the representation `SfPolicyFloorParams` validated
  # against, so the feasibility guarantee holds by construction instead of
  # depending on what dtype the policy softmax happens to return. `relu(floors -
  # probs)` promotes, so a narrower `probs` costs nothing here.
    mandatory = torch.zeros(
        probs.shape, dtype=_FLOOR_THRESHOLD_DTYPE, device=probs.device,
    )
    mandatory = mandatory.scatter_reduce(
        -1, top1,
        torch.full_like(top1, float(params.tau_top1), dtype=_FLOOR_THRESHOLD_DTYPE),
        reduce="amax", include_self=True,
    )
    if played_target is not None:
        played = played_target.argmax(dim=-1, keepdim=True)
  # A row whose policy target carries no mass has no played move to protect
  # (an absent or masked-out target argmaxes to index 0), so its collar
  # threshold is 0.0 -- the same no-op the ablation uses.
        has_target = (played_target.sum(-1, keepdim=True) > 0).to(_FLOOR_THRESHOLD_DTYPE)
        mandatory = mandatory.scatter_reduce(
            -1, played, has_target * float(params.tau_played),
            reduce="amax", include_self=True,
        )

    tau_t = torch.full_like(mandatory, float(params.tau))
  # What the UNCAPPED rule asked for. Kept as its own tensor because it is the
  # diagnostic pair's numerator and because the cap is defined as the
  # difference between the two.
    raw_floors = torch.maximum(mandatory, torch.where(adaptive, tau_t, mandatory.new_zeros(())))

  # ⚑ FEASIBILITY. `sum_m floor_m > 1` is not merely a large penalty -- it is a
  # constraint set with no distribution in it, so `relu(floor - p)` keeps a
  # residual on EVERY row of that shape forever and the gradient never resolves.
  # Measured on 5,881 live production rows the infeasible fraction was
  # 0.000000 with |F| <= 6, and 13.1% of those rows sat AT |F| = 6, right-
  # censored by the live `sf_multipv: 6`. ⚑ DO NOT READ THAT AS "load-bearing at
  # sf_multipv: 40" -- an earlier version of this comment did, and it
  # contradicted the docstring above. See the docstring for the interval and its
  # caveat; the MPV40 rate is UNMEASURED.
  #
  # The cap is NOT a rescaling of the floors: shrinking every tau would change
  # the calibrated bar on the rows that were already fine. Instead the SET is
  # made structurally feasible -- the two mandatory roles are paid first (the
  # resolve-time validator guarantees they fit WHEN THE TERM IS ON; at w == 0.0 it
  # only warns, and that is the one case this cap cannot repair), then adaptive
  # members are
  # admitted in ASCENDING SF REGRET, best move first, until admitting another
  # would exceed the budget.
  # Non-members sort last (their regret key is 2.0, above every real regret),
  # and carry a zero increment, so where they land cannot change the outcome.
    order = torch.argsort(
        torch.where(adaptive, regret, torch.full_like(regret, 2.0)), dim=-1, stable=True,
    )
  # ⚑⚑ THE BUDGET ARITHMETIC IS EXACT, AND A POSITIVE SLACK IS NOT AN OPTION.
  # An earlier version added `+1e-6` here so a set that fits EXACTLY would not
  # be rejected by float32 rounding. That defeated the invariant the cap exists
  # to enforce: it admitted genuinely infeasible sets whose mass lands in
  # `(1, 1+1e-6]`. MEASURED -- `tau = tau_top1 = 0.3333334`, collar off, three
  # members: float32 mass 1.000000238 > 1, and every member was admitted, so
  # the reported `applied_mass` exceeded 1 while the docstring promised it
  # could not. Reviewer finding, PR #448.
  #
  # The fix is not a smaller epsilon -- any positive slack is the same bug with
  # a smaller radius. The sums are done in float64, which represents each
  # float32 floor EXACTLY, so the comparison is the real one and the rounding
  # bias runs the safe way: a set that overflows by one ULP loses its last
  # optional member instead of being kept.
  #
  # ⚑ THE `autocast(enabled=False)` GUARD BELOW IS BELT-AND-BRACES, AND AN
  # EARLIER VERSION OF THIS COMMENT JUSTIFIED IT WITH A FALSE CLAIM: that cumsum
  # sits on autocast's fp32 cast list and would narrow float64 back to float32.
  # It does not. Autocast's cast policies apply only to ELIGIBLE dtypes, and
  # `float64` is not one, so float64 is never touched. (⚑ The broader claim "the
  # fp32 policy promotes and never demotes" is only half right and is NOT what
  # this rests on: the cast lists differ BY DEVICE -- measured, `bf16.cumsum`
  # stays bf16 on cpu and becomes float32 on cuda. Float64 ineligibility is the
  # load-bearing fact, and it holds on both.) MEASURED (torch 2.11.0+cu128, bf16
  # autocast, cpu AND cuda):
  # `float64.cumsum -> float64`, `float64.sum -> float64`, while
  # `float32.cumsum -> float32`. So the guard is currently INERT and is kept
  # only so this block's exactness does not silently depend on that staying
  # true. Reviewer finding P3-4, PR #448: do not re-derive a reason for it from
  # the old comment, because the old comment was wrong.
  # ⚑ THE SUBTRACTION IS PART OF THE ARITHMETIC, NOT A PRE-STEP. Widening an
  # increment that was already rounded in float32 does not recover it: at
  # `tau = 0.5, tau_top1 = 0.1` the float32 `tau - mandatory` rounds UP by
  # 7.5e-9, and the exactly-feasible pair `0.5 + 0.5 = 1.0` then reads as
  # infeasible and loses a member. Every term below is the float32 floor value
  # WIDENED, with the subtraction done in float64, so `mandatory + increment`
  # reproduces the applied floor bit for bit and the comparison is the real one.
    with torch.amp.autocast(device_type=probs.device.type, enabled=False):
        mandatory64 = mandatory.to(torch.float64)
        increment = torch.where(
            adaptive,
            (tau_t.to(torch.float64) - mandatory64).clamp(min=0.0),
            mandatory64.new_zeros(()),
        )
        cumulative = increment.gather(-1, order).cumsum(-1)
        budget = 1.0 - mandatory64.sum(-1, keepdim=True)
  # ⚑ `~(cum > budget)`, NOT `cum <= budget`. The two agree on every real number
  # and DISAGREE on NaN: `NaN <= budget` is False, so a NaN tau would admit
  # nothing and the cap would silently swallow a poisoned parameter into a
  # finite, plausible-looking loss -- this repo's signature defect, and exactly
  # what `test_a_nan_in_the_term_cannot_reach_total_at_weight_zero` exists to
  # catch. The negated form admits the NaN member and lets it propagate to the
  # loss, where the trainer's own NaN guards can see it. The cap is a
  # feasibility rule, not a validator.
    keep = torch.zeros_like(adaptive).scatter(-1, order, ~(cumulative > budget))
    admitted = adaptive & keep

    floors = torch.maximum(mandatory, torch.where(admitted, tau_t, mandatory.new_zeros(())))
    deficit = torch.relu(floors - probs)
  # The two MASS columns are summed in float64 for the same reason the budget
  # is: a float32 reduction over a different order than the cumulative sum can
  # disagree with it by an ULP, and `applied_mass <= 1` would then be a
  # contract the diagnostic itself could break. Narrowed back to the probs
  # dtype on the way out -- rounding to nearest cannot carry a value at or
  # below 1.0 above it, because 1.0 is exactly representable.
    with torch.amp.autocast(device_type=probs.device.type, enabled=False):
        requested_mass = raw_floors.to(torch.float64).sum(-1).to(probs.dtype)
        applied_mass = floors.to(torch.float64).sum(-1).to(probs.dtype)
    return SfPolicyFloorOutputs(
  # Narrowed back so every field of this tuple is in the caller's dtype, as it
  # was before the thresholds were pinned. A no-op in production, where
  # `po_probs` is already fp32.
        deficit=deficit.sum(-1).to(probs.dtype),
        binds=(deficit > 0).any(-1).to(probs.dtype),
        member_count_raw=(raw_floors > 0).sum(-1).to(probs.dtype),
        requested_mass=requested_mass,
        truncated=(admitted != adaptive).any(-1).to(probs.dtype),
        member_count_applied=(floors > 0).sum(-1).to(probs.dtype),
        applied_mass=applied_mass,
    )


# Default teacher temperature of the SF-shape term, in CENTIPAWNS -- the unit
# `sf_p0_regret` is quoted in before normalization, and the same unit
# `sf_policy_floor_delta_cp` uses. NOT a dimensionless "temperature": the softmax
# runs over `regret * SF_OWN_REGRET_CAP_CP`, so 100.0 means "a move 100 cp worse
# than SF's best gets e^-1 of its weight". The units are in the NAME because the
# repo already carries two differently-scaled quantities spelled "T"
# (`policy_target_temp`, `gumbel_policy_temp`) and has been bitten by the
# collision (see `_POLICY_TARGET_TEMP_MIN`).
#
# ⚑ IT IS A PLACEHOLDER, NOT A CALIBRATION, AND NOTHING HERE ESTABLISHES IT.
# The knob this term turns is exactly the ENTROPY of the teacher, so it must be
# CHOSEN so that `sf_shape_h_sf_given_s` lands on a reference conditional
# entropy measured on real rows -- not hand-tuned toward whatever makes the loss
# curve look good, which is the "arm that is the gradient of the metric" trap.
# 100.0 is a round number picked to be readable, the term ships at `w = 0.0`, and
# the calibration is deliberately out of scope for the change that introduced it.
# `sf_shape_h_sf_given_s` is reported at zero weight precisely so the
# calibration can be done from production rows before the term is ever switched on.
SF_SHAPE_TEMP_CP_DEFAULT = 100.0

# Band for the above, in the same centipawn units. The LOWER bound is a typo guard
# (an in-sign absurdity like `1e-9` -- the `zclip_max_norm` shape -- would be accepted
# in silence and quietly empty the entropy column).
#
# ⚑⚑ THE UPPER BOUND IS NOT A TYPO GUARD. IT IS A SOUNDNESS LIMIT OF THE SURFACED-SET
# RECOVERY, AND IT IS DERIVED, NOT CHOSEN. `sf_surfaced_move_mask` recovers S from the
# regret vector alone, and a REAL SF move sitting at the `SF_OWN_REGRET_CAP_CP` cap ties
# the writer's fill and is DROPPED from S -- the one-directional lossiness that function's
# docstring admits. That docstring then argues the loss is harmless "because the teacher
# gives such a move ~0 weight anyway", and THAT argument is temperature-dependent:
#
#     relative teacher weight of a capped move = exp(-SF_OWN_REGRET_CAP_CP / temp_cp)
#       temp_cp =  100  ->  4.5e-5   negligible, the argument holds
#       temp_cp = 1000  ->  0.368    A THIRD OF THE BEST MOVE'S MASS, silently missing
#
# An earlier revision of this file set the ceiling to 10000 as a round "obviously wide
# enough" number, which made that unsoundness REACHABLE -- and the deferred calibration is
# explicitly a TEMPERATURE SWEEP, so it walks straight into it. Found by an independent
# grok review of PR #479. [[wiring_a_dead_knob_can_arm_a_crash]]
#
# So the ceiling is the largest temperature at which a capped move stays below
# SF_SHAPE_CAPPED_MOVE_MAX_WEIGHT of the best move's mass:
#     temp_cp <= SF_OWN_REGRET_CAP_CP / ln(1 / eps)
# ⚑ Raising this ceiling REQUIRES first fixing the recovery (a fill strictly above the
# cap, which is a WRITER change and reprocesses shards). Do not widen it to unblock a
# high-temperature arm; the arm would be training on a teacher missing its heaviest tail.
SF_SHAPE_TEMP_CP_MIN = 1.0
# eps is a JUDGEMENT CALL and is stated as one. 1e-2 admits the ~200 cp region the
# existing temperature-sweep test already exercises (evidence about what this repo
# considers a plausible teacher temperature) while still excluding the regime where the
# omission is gross: a capped move carries 1% of the best move's mass at the ceiling and
# 37% at temp_cp=1000. It was 1e-3 for one revision, chosen for roundness rather than for
# a reason, and that silently outlawed a temperature the suite already used -- picking the
# constant to make a test pass would be the trap, but so is keeping an arbitrary one that
# blocks legitimate work.
SF_SHAPE_CAPPED_MOVE_MAX_WEIGHT = 1.0e-2
SF_SHAPE_TEMP_CP_MAX = float(
    int(SF_OWN_REGRET_CAP_CP / math.log(1.0 / SF_SHAPE_CAPPED_MOVE_MAX_WEIGHT))
)


@dataclass(frozen=True)
class SfShapeParams:
    """Resolved, validated parameters of the SF-shape conditional-KL term.

    ONE object, resolved once and consumed as-is, for the same reason
    ``SfPolicyFloorParams`` is: the value that gets logged must not be a second
    derivation of the value that gets used.

    ``w`` 0.0 (the default) means the term contributes NOTHING to ``total`` --
    not "multiplied by zero", see ``compute_loss``. The entropy instrument is
    still computed, which is the entire point of the diagnostic half: "are we
    sharper than our own teacher" has to be readable BEFORE anyone raises the
    weight, because that drift ran for months with no column carrying it.
    """

    w: float = 0.0
    temp_cp: float = SF_SHAPE_TEMP_CP_DEFAULT

    def __post_init__(self) -> None:
        w = float(self.w)
        if not math.isfinite(w) or w < 0.0:
            raise ValueError(f"w_sf_shape must be finite and >= 0, got {self.w!r}")
        temp = float(self.temp_cp)
  # STRICTLY positive: it is a divisor. 0.0 is not "off" here -- the off switch
  # is `w` -- it is an inf/NaN generator wearing the word "disabled".
        if not math.isfinite(temp) or temp <= 0.0:
            raise ValueError(
                f"sf_shape_temp_cp must be finite and > 0 (it is a divisor, in "
                f"centipawns), got {self.temp_cp!r}"
            )
  # ⚑ AND A BAND, because "> 0" leaves the `zclip_max_norm: 1e-9` shape open:
  # in-sign, absurd, silently accepted, and slow to notice -- CLAUDE.md's
  # category (c). This key is a DIVISOR in centipawns whose own docstring says
  # it MUST be swept against a reference conditional entropy, so values will be
  # typed into it by hand. Outside this band `q_S` is a delta (tiny) or uniform
  # (huge), and `sf_shape_h_sf_given_s` -- the column the calibration is read
  # off -- becomes meaningless rather than wrong-looking.
  # The band is wide on purpose: it is a typo guard, not a calibration claim.
  # 1 cp cannot be a sensible teacher temperature and 10000 cp is 100 pawns.
  # This makes a bad value a LOUD launch failure instead of a quiet instrument
  # corruption; the key is startup-only, so it cannot kill a running trial from
  # a mid-run edit the way a live-read validated key can.
        if not (SF_SHAPE_TEMP_CP_MIN <= temp <= SF_SHAPE_TEMP_CP_MAX):
            raise ValueError(
                f"sf_shape_temp_cp must be within "
                f"[{SF_SHAPE_TEMP_CP_MIN}, {SF_SHAPE_TEMP_CP_MAX}] centipawns "
                f"(a typo guard, not a calibration claim), got {self.temp_cp!r}"
            )

    @classmethod
    def resolve(
        cls, *, w: float | None = None, temp_cp: float | None = None,
    ) -> SfShapeParams:
        """Fill the ``None`` defaults and validate. The single entry point."""
        return cls(
            w=0.0 if w is None else float(w),
            temp_cp=SF_SHAPE_TEMP_CP_DEFAULT if temp_cp is None else float(temp_cp),
        )


def sf_surfaced_move_mask(
    regret: torch.Tensor, legal: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The moves Stockfish ACTUALLY scored, recovered from ``sf_p0_regret`` alone.

    Returns ``(surfaced, count)`` -- a bool tensor shaped like ``regret``, and its
    per-row sum as float32.

    ⚑⚑ THERE IS NO SURFACED-MASK FIELD IN THE TRAINING BATCH, AND THE ONE
    RAW-MultiPV FIELD THAT IS THERE IS THE WRONG PLY. ``sf_multipv_raw``
    describes the position AFTER the row's move; SF's read of THIS position is
    the PREVIOUS row's block, which is exactly the join
    ``_build_sf_p0_regret_vector`` is fed from. It is also gated behind
    ``train.sf_policy_sparse_ce`` and is dropped from the H2D payload in
    production (`replay/dataset.py`). So the mask has to be recovered from the
    regret vector itself, and this function is that recovery.

    THE RULE, and why it cannot admit a fabricated entry::

        fill     = max over the FULL row of regret
        surfaced = legal AND (regret < fill)

    ``_build_sf_p0_regret_vector`` fills EVERY index of the full policy vector
    with one scalar ``d = float32((worst_surfaced + 1) / 2)`` and then overwrites
    only the covered ones. So every fabricated entry holds a bit-identical ``d``,
    and ``d >= worst_surfaced >= r`` for every covered ``r``. Two consequences,
    and only the second is a limitation:

    1. **Nothing invented can enter the set.** A fabricated entry equals the row
       max exactly and the comparison is strict. This holds WITHOUT assuming
       ``d`` is a value no real regret can take: it is a comparison against the
       row's own maximum, not against a magic constant.
    2. **A surfaced move at the 1000 cp CAP is dropped.** ``d == worst`` iff
       ``worst == 1.0``, i.e. SF surfaced a move at or beyond
       ``SF_OWN_REGRET_CAP_CP``; those moves then tie ``d`` and fall out. The
       error is ONE-SIDED -- the recovered set is always a SUBSET of the true one
       -- and the moves it drops are the worst SF scored, which the teacher
       softmax gives ~0 weight anyway. The term simply asserts nothing about
       them, which is the contract it already has for the unsurfaced ~79%.

    ``fill`` is the FULL-row max rather than the max over LEGAL entries
    deliberately: in an endgame every legal move can be surfaced, and a
    legal-only max would then be ``worst_surfaced`` and would drop SF's worst
    legal move on exactly the rows where coverage is perfect. The fill covers
    illegal indices too, so the full-row max is ``d`` on every real row.

    NaN-tolerant by construction: a NaN anywhere makes ``fill`` NaN, every
    comparison False and the set empty -- which the caller treats as "no SF
    opinion on this row", never as a value to propagate into ``total``.
    """
    fill = regret.amax(dim=-1, keepdim=True)
    surfaced = regret < fill
    if legal is not None:
        surfaced = surfaced & legal
    return surfaced, surfaced.sum(-1).to(torch.float32)


def row_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Per-row Shannon entropy in nats, safe at exact zeros.

    ``0 * log(tiny) == 0`` for the zeros a legal mask or an empty set leaves
    behind, where ``0 * log(0)`` would be NaN. ``clamp_min`` applies to the LOG's
    argument only -- the probability itself is untouched -- so this is the limit
    value rather than a clamped distribution.
    """
    p = probs.to(torch.float32)
    return -(p * p.clamp_min(torch.finfo(torch.float32).tiny).log()).sum(-1)


@dataclass(frozen=True)
class MatchedSupportStats:
    """A policy/target entropy pair over ONE EXPLICIT support, plus what it drops.

    All members are per-row, shape ``(B,)``, and all are detached: this is an
    instrument, not a loss.
    """

    h_ours: torch.Tensor
    h_target: torch.Tensor
    support_size: torch.Tensor
    tail_mass_ours: torch.Tensor


def matched_support_entropy_stats(
    probs: torch.Tensor, target: torch.Tensor, legal: torch.Tensor | None,
) -> MatchedSupportStats:
    """Entropy of ``probs`` and of ``target`` over the TARGET'S OWN support.

    The support is ``T = (target > 0) AND legal``. Both distributions are
    restricted to ``T`` and renormalized there, so the two entropies are
    comparable and their difference means what it looks like it means.

    ⚑⚑ THIS EXISTS BECAUSE AN UNMATCHED-SUPPORT COMPARISON PRODUCES A LARGE,
    PLAUSIBLE, WRONG NUMBER. `scripts/audit_targets.py` reports the net's raw
    policy at 0.8827 nats over ~27 legal moves and the production training target
    at 0.6255 over the ~16-move Gumbel candidate set, ZERO elsewhere. Read as a
    gap that says "cross-entropy is actively pushing the net to concentrate",
    which would be a major finding. But the target is zero outside the candidate
    set while the policy spreads mass over every legal move, and a ~5% tail over
    ~20 moves is worth ``0.05 * ln(20 / 0.05) ~ 0.30`` nats ON ITS OWN -- MORE
    than the 0.26 gap. So the whole effect may be an artifact of support size.
    THAT COMPARISON IS UNVERIFIED AND MUST NOT BE CITED.

    ``tail_mass_ours`` is the honest home for what the restriction throws away:
    the probability our policy places OUTSIDE the target's support. It is
    reported as its own column instead of being allowed to leak into an entropy
    difference, and it is independently interesting -- it is our mass on moves
    the search never made a candidate.

    ⚑ WHAT THE PAIR WOULD ANSWER, stated so the measurement has a purpose and
    NOT answered here: if the target really is sharper than the net's own output
    ON MATCHED SUPPORT, then the ordinary policy CE is itself a SHARPENING
    teacher (``dL/dz = p - t``), which would explain why ``policy_own``'s
    ``log_temp`` learned NEGATIVE -- the head trying to flatten -- while the net
    stayed over-sharp. If the gap vanishes on matched support, the sharpening
    story is an artifact and the search target is not the culprit. Both readings
    are available from these columns and neither is asserted anywhere in this
    file.
    """
  # ⚑ `no_grad`, not a per-field `.detach()`. `probs` is the LIVE `base_probs`
  # tensor, so without this the class's own "all are detached" contract was
  # false: `h_ours` and `tail_mass_ours` came back as autograd products, and the
  # only thing keeping them out of the objective was that no caller happened to
  # sum them. That is a promise held up by luck rather than by the code, and it
  # is the shape this repo's review doctrine names -- an asserted safety
  # property that nothing enforces. Reported by an independent review of
  # PR #479. `no_grad` also drops the graph for the whole block rather than
  # building it and discarding it four times.
    with torch.no_grad():
        p = probs.to(torch.float32)
        t = target.to(torch.float32)
        support = t > 0.0
        if legal is not None:
            support = support & legal
        sup = support.to(torch.float32)
        tiny = torch.finfo(torch.float32).tiny
        p_mass = (p * sup).sum(-1, keepdim=True)
        t_mass = (t * sup).sum(-1, keepdim=True)
      # `clamp_min` on the DIVISOR only. An empty support (a row with no policy
      # target) then gives an all-zero conditional, whose `row_entropy` is 0.0
      # and whose tail mass is 1.0 -- the honest reading, and finite.
        return MatchedSupportStats(
            h_ours=row_entropy(p * sup / p_mass.clamp_min(tiny)),
            h_target=row_entropy(t * sup / t_mass.clamp_min(tiny)),
            support_size=sup.sum(-1),
            tail_mass_ours=1.0 - p_mass.squeeze(-1),
        )


@dataclass(frozen=True)
class SfShapeReadout:
    """Per-row output of :func:`sf_shape_conditional_kl`. All shape ``(B,)``.

    ``kl`` is the only member that carries a gradient; every other member is
    detached, because they are instrument columns and nothing should be able to
    train on them by accident.
    """

    kl: torch.Tensor
    h_sf_given_s: torch.Tensor
    h_ours_given_s: torch.Tensor
    h_ours_full_legal: torch.Tensor
    surfaced_count: torch.Tensor
    surfaced_mass: torch.Tensor
    p_sf_best: torch.Tensor
    regret_cp_given_s: torch.Tensor


def sf_shape_conditional_kl(
    masked_logits: torch.Tensor,
    probs: torch.Tensor,
    regret: torch.Tensor,
    legal: torch.Tensor | None,
    *,
    params: SfShapeParams,
) -> SfShapeReadout:
    """``KL(q_S || p_S)``: SF's shape over the surfaced set, and ours, conditioned.

    Both sides are CONDITIONAL distributions over ``S``, the moves SF actually
    scored (:func:`sf_surfaced_move_mask`)::

        q_S = softmax_{i in S}( -regret_cp_i / temp_cp )   # the teacher
        p_S = softmax_{i in S}( z_i )                      # ours, renormalized
        L   = sum_{i in S} q_i * (log q_i - log p_i)

    ⚑⚑ THE CONDITIONING IS THE DESIGN, NOT A NORMALIZATION DETAIL. Because both
    softmaxes run over ``S`` alone, ``dL/dz_i == 0`` exactly for every ``i`` not
    in ``S``, and ``sum_{i in S} dL/dz_i == p_S - q`` summed over ``S`` ``== 0``.
    So the term provably CANNOT move probability into or out of ``S`` -- it only
    ever redistributes within it. That guarantee is the whole point: at
    ``sf_multipv: 6`` roughly 79% of the move list carries a FABRICATED
    ``default_regret``, so any term that lets SF's opinion leak onto the
    unsurfaced tail is training on invented data. A full-length cross-entropy
    against a mass-preserving target looks equivalent and is not: its gradient
    still couples to the moves outside ``S``, and the guarantee would then rest
    on bookkeeping instead of on the arithmetic.

    ⚑ WHY THIS REACHES THE TAIL WHERE ``sum_m p_m * r_m`` CANNOT. That term's
    gradient carries a factor ``p_i``, so its pull on a move fades to nothing as
    that move's probability goes to zero -- SF's best move at prior 0.008 gets
    ~38x less gradient than our own top pick. A conditional KL sets RELATIVE
    proportions, so it pulls equally hard whether SF's move sits at 0.30 or at
    0.003.

    DEGENERATE ROWS ARE EXACTLY ZERO, BY ARITHMETIC AND NOT BY A BRANCH.
    ``|S| == 1`` gives ``q`` and ``p_S`` both one-hot on the same move, so
    ``L == 0``; ``|S| == 0`` leaves both softmaxes over an all-``finfo.min`` row,
    which is uniform on BOTH sides, so ``L == 0`` again. Neither can contribute
    to ``total`` at any weight, and neither needs a special case.

    Everything is computed in float32 regardless of the autocast dtype the logits
    arrive in. The masked softmaxes are filled with ``finfo.min`` rather than
    ``-inf``: an all-masked ``-inf`` row softmaxes to NaN, and NaN in an unused
    branch of ``torch.where`` still poisons the gradient of the branch that IS
    used.
    """
    reg = regret.to(torch.float32)
    surfaced, count = sf_surfaced_move_mask(reg, legal)
    neg = torch.full_like(reg, torch.finfo(torch.float32).min)
  # cp units: `temp_cp` reads as "a move this many cp worse than SF's best gets
  # e^-1 of its weight". Higher score = better for the mover.
    teacher = -(reg * SF_OWN_REGRET_CAP_CP) / float(params.temp_cp)
    log_q = torch.log_softmax(torch.where(surfaced, teacher, neg), dim=-1)
  # ⚑ `torch.where` IS THE GRADIENT CUT. Its derivative wrt the unselected branch
  # is exactly zero, so no logit outside `S` receives any gradient from this term
  # -- the property the docstring above claims, expressed in the one operation
  # that makes it true rather than asserted around it.
    log_p = torch.log_softmax(
        torch.where(surfaced, masked_logits.to(torch.float32), neg), dim=-1,
    )
    q = log_q.exp()
    kl = (q * (log_q - log_p)).sum(-1)
    with torch.no_grad():
        surfaced_f = surfaced.to(torch.float32)
        p32 = probs.to(torch.float32)
        surfaced_mass = (p32 * surfaced_f).sum(-1)
  # SF's single best move: the argmin of regret over LEGAL moves, with illegal
  # entries pushed above every real regret so the argmin cannot land on one --
  # the SAME rule `sf_policy_floor_deficit` uses for its unconditional `top1`
  # member, so the two families name the same move. Zeroed where the set is
  # empty, because there the argmin lands on a fabricated entry and the number
  # would be about nothing.
        ranked = reg if legal is None else torch.where(legal, reg, torch.full_like(reg, 2.0))
        top1 = ranked.argmin(dim=-1, keepdim=True)
        p_sf_best = p32.gather(-1, top1).squeeze(-1) * (count > 0).to(torch.float32)
        return SfShapeReadout(
            kl=kl,
  # ⚑⚑ SAME SUPPORT, AND THE NAMES SAY SO. `h_sf_given_s` and `h_ours_given_s`
  # are BOTH conditioned on S -- neither is a full-width entropy -- so their
  # difference is a statement about shape and not about how many moves each
  # distribution happens to spread over. A bare "policy entropy" compared against
  # a teacher entropy over a different support is how three separate readings
  # were confounded before this instrument existed; the third field below carries
  # `full_legal` in its NAME for exactly that reason and is never differenced
  # against either of the first two.
            h_sf_given_s=row_entropy(q * surfaced_f),
            h_ours_given_s=row_entropy(log_p.exp() * surfaced_f),
            h_ours_full_legal=row_entropy(probs),
  # Mean |S|: the mask-health column. If the recovery above ever silently
  # collapses -- a changed fill rule, a realignment that zero-pads the row --
  # this is what says so, and it is the first number to read before any other
  # column in this family.
            surfaced_count=count,
  # ⚑⚑ M_S -- THE QUESTION THIS TERM CANNOT ANSWER, AND THE ONE THAT DECIDES
  # WHETHER IT SHOULD EVER BE SWITCHED ON. There are two independent pathologies
  # and the conditional KL only reaches the first:
  #   (A) WRONG SHAPE INSIDE S -- our conditional is sharper than SF's. This term
  #       fixes it.
  #   (B) WRONG MASS ON S -- most of our probability sits on moves SF never
  #       scored. The conditional KL is INVARIANT to M_S BY CONSTRUCTION and
  #       cannot touch it; that needs a WIDER LABELLING SET (`searchmoves` over
  #       the net's own top moves), not a loss.
  # So a low M_S with a matched conditional shape means the loss addresses the
  # wrong thing and must NOT be given weight. The share of our mass on moves SF
  # never scored is exactly `1 - M_S` -- published as one quantity rather than
  # two, so the pair cannot drift.
  #
  # It has to be measured on live MultiPV-6 rows: an offline attempt over banked
  # wide-era shards read it trivially, because those labels cover 26.63 of 26.82
  # legal moves and the set was therefore not restricted at all.
            surfaced_mass=surfaced_mass,
  # p_own on SF's SINGLE best move, in ABSOLUTE terms -- not conditioned on S, so
  # unlike everything above it moves with M_S. It is the number
  # `sf_policy_floor` is a floor on, which makes the two families readable
  # against each other.
            p_sf_best=p_sf_best,
  # ⚑ THE MOST INTERPRETABLE OF THE SIX: "how bad are the moves our conditional
  # policy prefers, in centipawns, according to SF". Entropy alone cannot see
  # this -- two distributions with IDENTICAL entropy can rank the surfaced moves
  # in opposite orders, and one of them is right. Reported in CP rather than in
  # the vector's normalized units so it needs no mental multiplication by the
  # 1000 cp cap.
          # ⚑⚑ `torch.where`, NOT `* surfaced_f`. Masking a NaN by MULTIPLYING by
          # 0.0 does not mask it: `0.0 * NaN == NaN`. A single non-finite entry
          # in `sf_p0_regret` -- on exactly the entries the surfaced rule then
          # excludes -- used to poison this whole column, and because the
          # reported value is an iteration-wide SUM, one bad row took out the
          # readout for the entire iteration. That defeats the point of the
          # change: this instrument has to be readable at `w_sf_shape: 0.0`
          # while the temperature is being calibrated, and
          # `sf_shape_regret_cp_given_s` is the most interpretable of the six.
          # `total` was never at risk (the term is excluded at w=0 and the KL
          # itself is computed on the masked set) -- but "NaN-tolerant by
          # construction" was a claim about the READOUT too, and it was false
          # for this one column. Found by an independent review of PR #479;
          # the guard is pinned by a test that asserts every `sf_shape*` column
          # is finite on a NaN regret row.
          # [[a_clamp_is_not_a_validator_nan_propagates]]
            regret_cp_given_s=(
                (
                    log_p.exp()
                    * torch.where(surfaced, reg, torch.zeros_like(reg))
                ).sum(-1)
                * SF_OWN_REGRET_CAP_CP
            ),
        )


def terminal_outcome_transfer_taper(
    batch: dict[str, torch.Tensor],
    *,
    plies: int,
    full_plies: int,
    max_plies: float,
) -> torch.Tensor | None:
    """Per-row taper for the terminal-proximal outcome transfer.

    Returns a ``(B, 1)`` float32 tensor in ``[0, 1]`` — a FRACTION of whatever
    the realized blend fracs are at this step, never an absolute weight, so the
    caller cannot accidentally pin the transfer to a stale 0.31. Returns
    ``None`` when the feature is off (``plies <= 0``) or the batch carries no
    ``moves_left`` field at all, which is the caller's signal to take the
    untouched blend path bit-for-bit.

    Why the outcome near the terminal: measured offline over 149k rows
    (2026-08-07), the game outcome's noise as a value label ramps with
    plies-to-terminal — sd ~0.10 at d=1-2, 0.22 at d=5-6, crossing the search
    component's own noise at d~6-7, 0.77 deep. Within a few plies of the end
    the outcome is the CLEANEST of the three estimators, crisper even than the
    deliberately-soft cp-logistic SF label; far from the end it is the
    catastrophic noise that carried a value collapse, which is why the GLOBAL
    ``game_frac`` stays 0 and this transfer is strictly local to small ``d``.

    ``d`` (plies to terminal) is recovered from the stored ``moves_left``
    field, which ``selfplay/finalize.py`` writes as
    ``(total_plies_played - ply_index) / max_plies`` — the divisor is the
    CONFIGURED PLY CAP, not the game's own length, so ``max_plies`` must be
    the cap the rows were WRITTEN under (production: 450). It is stored
    float16, whose quantization puts the reconstruction just off an integer,
    hence ``round``.

    The taper is ``clamp((plies - d) / (plies - full_plies), 0, 1)``: ``d <=
    full_plies`` transfers the whole eligible share, ``d >= plies`` transfers
    nothing. ``full_plies >= plies`` degenerates to the step function that
    formula tends to.

    ⚑ WHICH share the caller applies this to is the caller's decision, and by
    default it is the SEARCH share only. The SF share is load-bearing
    supervision (zeroing it crashed winrate 0.64 -> 0.40) and moves only when
    ``wdl_terminal_outcome_sf_frac`` is deliberately raised off 0.0.
    """
    if int(plies) <= 0:
        return None
    moves_left = batch.get("moves_left")
    if moves_left is None:
        return None
    divisor = float(max_plies)
    if divisor <= 0.0:
        raise ValueError(
            "terminal-proximal outcome transfer is enabled "
            f"(wdl_terminal_outcome_plies={int(plies)}) but moves_left_max_plies "
            f"is {divisor!r}. `moves_left` is normalized by the selfplay ply cap, "
            "so without it the plies-to-terminal distance cannot be recovered — "
            "pass the run's `max_plies` rather than guessing a divisor."
        )
    plies_f = float(int(plies))
    full = float(min(max(int(full_plies), 0), int(plies)))
    span = plies_f - full
    d = torch.round(moves_left.to(torch.float32) * divisor)
    if span <= 0.0:
        taper = (d < plies_f).to(torch.float32)
    else:
        taper = ((plies_f - d) / span).clamp(0.0, 1.0)
  # Rows whose shard never carried `moves_left` have no `d` — they keep the
  # unchanged blend rather than being treated as d=0 (the field defaults to
  # 0.0, which would otherwise read as "terminal" and hand them the FULL
  # transfer, the worst possible failure direction).
    has_moves_left = _get_mask(batch, "has_moves_left").to(torch.float32)
    return (taper * has_moves_left).unsqueeze(1)


def normalize_value_blend_fracs(
    sf_wdl_frac: float, search_wdl_frac: float,
) -> tuple[float, float, float]:
    """``(sf, search, game)`` shares of the WDL value target, as applied.

    Extracted from ``compute_loss`` so the value-blend GUARD can read the same
    arithmetic the trained objective uses instead of restating it. A guard that
    reimplements its criterion measures its own copy: if the clamp, the
    renormalisation or the ``game_frac`` complement ever moves here, a
    duplicated version in the guard would keep passing while the two drifted
    apart. One implementation, two callers.

    ⚑ ``game`` is the share that is DELIBERATELY the raw one-hot game outcome.
    It is not the whole outcome-borne share: when a row carries no usable
    ``sf_wdl``, ``compute_loss`` falls the SF component back to that same
    one-hot, so the realized outcome mass is ``game + sf`` on those rows. That
    gap is what ``train/value_blend_guard.py`` exists to make loud — see
    ``value_blend_readout``.
    """
    sf = max(0.0, float(sf_wdl_frac))
    search = max(0.0, float(search_wdl_frac))
    blend_sum = sf + search
    if blend_sum > 1.0:
        return sf / blend_sum, search / blend_sum, 0.0
    return sf, search, 1.0 - blend_sum


def _compute_sf_wdl_mask(
    *,
    net_mask: torch.Tensor,
    has_sf_wdl: torch.Tensor,
    sf_wdl_probs: torch.Tensor | None,
    wdl_target: torch.Tensor,
    conf_power: float,
    draw_scale: float,
) -> torch.Tensor:
    """Row mask for the AUXILIARY ``sf_eval`` head. NOT the WDL value target.

    ⚑ Read this before tuning ``sf_wdl_conf_power`` / ``sf_wdl_draw_scale``.
    The mask returned here has exactly ONE consumer, ``m_sf_eval``. The WDL
    value target -- the load-bearing blend in ``compute_loss`` -- weights its
    SF component by ``sf_effective = sf_available * keep``, and ``keep`` carries
    only the ``sf_search_dampen_sf_*`` terms. Neither knob appears in that
    expression, so neither can damp the value target: change either one and
    ``wdl_ce`` is bit-identical while ``sf_eval_ce`` moves (a 0.1-weighted head,
    ~0.006 % of total loss). Pinned by
    ``tests/test_sf_wdl_conf_knobs_are_aux_only.py``; the name is kept because
    it is a live yaml key and the live-yaml validator rejects unknown keys.

    Damping: ``(1 - draw_prob)^power`` zeros out high-draw rows where SF's
    label barely disagrees with a fresh-init model. Draw rescale boosts/cuts
    the contribution of game-decided-as-draw outcomes.
    """
    mask = net_mask * has_sf_wdl
    if sf_wdl_probs is None:
        return mask
    if conf_power > 0.0:
        sf_conf = (1.0 - sf_wdl_probs[:, 1]).clamp(0.0, 1.0).pow(conf_power)
        mask = mask * sf_conf
    if draw_scale != 1.0:
        draw_mask = (wdl_target == 1).to(torch.float32)
        mask = mask * (1.0 - draw_mask + draw_mask * draw_scale)
    return mask


def _normalize_sf_wdl_probs(
    sf_wdl_raw: torch.Tensor | None, *, temperature: float = 1.0
) -> torch.Tensor | None:
    """Clamp negatives to 0, optionally soften via ``p^(1/T)``, renormalize.

    ⚑⚑ THIS FUNCTION LAUNDERS ``-inf`` INTO A VALID-LOOKING DISTRIBUTION.
    ``clamp_min(0.0)`` maps ``-inf`` to ``0.0``, so a row like
    ``[-inf, 0.5, 0.5]`` leaves here as ``[0.0, 0.5, 0.5]`` -- finite,
    normalized, and indistinguishable from a real label. ``+inf`` does NOT
    survive (``inf / inf`` is NaN), so the two infinities behave OPPOSITELY
    across this call. Any finiteness test on a label must therefore be made on
    the RAW batch tensor, BEFORE this function -- see `_finite_blend_component`,
    which takes both and requires both.
    """
    if sf_wdl_raw is None:
        return None
    p = sf_wdl_raw.clamp_min(0.0)
    if temperature != 1.0 and temperature > 0.0:
        p = p.clamp_min(1e-6).pow(1.0 / float(temperature))
    return normalize_distribution(p)


def _finite_blend_component(
    probs: torch.Tensor | None,
    *,
    raw: torch.Tensor | None,
    weight: torch.Tensor,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """One side of the value-target blend, with masked-off NaN label rows removed.

    Returns ``(component, claimed_nonfinite_rows, unclaimed_nonfinite_rows)``.
    Both counts are ``None`` when the field is absent, and device scalars
    otherwise; the caller must pass the claimed count to
    `_assert_blend_labels_finite` BEFORE the component reaches ``target``.

    ⚑⚑ ``raw`` IS THE BATCH TENSOR AND ``probs`` IS ITS NORMALIZED FORM, AND
    BOTH ARE TESTED. Testing only the normalized tensor is not a weaker check,
    it is a check with a HOLE: `_normalize_sf_wdl_probs` opens with
    ``clamp_min(0.0)``, so a CLAIMED row of ``[-inf, 0.5, 0.5]`` arrives here as
    the perfectly ordinary ``[0.0, 0.5, 0.5]`` and TRAINS SILENTLY, while the
    same row with ``+inf`` raises (``inf / inf`` is NaN). Same class of corrupt
    label, opposite outcomes, decided by a sign. Testing ``raw`` first makes
    ``-inf``, ``+inf`` and ``NaN`` one case. ``probs`` is still tested as well,
    because the normalise is where a finite-but-degenerate row can go
    non-finite.

    ⚑ THE PRE-EXISTING PROTECTION IS KEYED ON FIELD-ABSENCE, NOT ON THE ROW MASK,
    and that is the hazard this function closes. With the field ABSENT the caller
    substitutes ``fallback`` wholesale and the blend is finite at any frac. With
    the field PRESENT and a row NaN, the per-row arithmetic
    ``weight * probs + (1 - weight) * fallback`` is ``0.0 * nan`` for a row whose
    own mask is 0 — so a row the blend does not want at all still poisons the
    target, at ANY frac including 0.0, and from there the CE, ``total`` and every
    gradient. ``_normalize_sf_wdl_probs`` is not a defence either: `clamp_min`
    and the renormalise both PROPAGATE NaN rather than sanitize it (a clamp is
    not a validator), so this test is deliberately made on the NORMALIZED tensor
    the blend actually consumes rather than on the raw batch field.

    ⚑ TWO REGIMES, AND ONLY ONE OF THEM IS A REGIME.
      - mask 0 (the row does not claim the label): its content is by definition
        irrelevant, so the row takes ``fallback`` EXACTLY. `torch.where` selects,
        it does not average, so nothing about a finite row's value changes.
      - mask non-zero (the row CLAIMS the label and the label is NaN): dirty
        label data, not a training regime. Counted here and raised on by
        `_assert_blend_labels_finite` — silently substituting the fallback there
        would train the value head on the game outcome while the shard says it
        has an SF opinion, which is this repo's signature defect exactly.

    ⚑ BIT-IDENTICAL ON EVERY CURRENTLY-FINITE PATH. ``blended`` is the parent
    expression, character for character, and `torch.where` returns its value
    unmodified wherever the row is finite — which is every row of every batch
    that works today. Pinned at FULL RESOLUTION (``torch.equal`` on the
    component, not ``==`` on a reduced scalar -- a sub-ULP target change does not
    survive the CE) by
    `tests/test_value_blend_nan_labels.py::test_the_component_is_bit_identical_to_the_unguarded_expression`.
    """
    if probs is None:
        return fallback, None, None
    if raw is None:
        raise ValueError(
            "_finite_blend_component: `raw` is required whenever `probs` is "
            "present. The finiteness test has to be made on the batch tensor, "
            "BEFORE `_normalize_sf_wdl_probs`'s clamp_min turns -inf into 0.0."
        )
    row_finite = (
        torch.isfinite(raw).all(dim=-1, keepdim=True)
        & torch.isfinite(probs).all(dim=-1, keepdim=True)
    )
    claimed = weight != 0.0
    claimed_nonfinite = (claimed & ~row_finite).to(torch.float32).sum()
    unclaimed_nonfinite = (~claimed & ~row_finite).to(torch.float32).sum()
    blended = weight * probs + (1.0 - weight) * fallback
    return (
        torch.where(row_finite, blended, fallback),
        claimed_nonfinite,
        unclaimed_nonfinite,
    )


def _assert_blend_labels_finite(
    counts: tuple[tuple[str, torch.Tensor | None], ...],
) -> None:
    """Raise when a value-label row claims to be valid and is non-finite.

    ⚑⚑ DELIBERATELY NOT GATED ON ``w_wdl``, AND THE SCOPE IS THE POINT. Asked
    whether this should stay silent on an arm that has the value head weighted
    off, the answer is no, as stated policy: a CLAIMED value label that is
    non-finite is a DATA DEFECT, never a training regime, and the shard is just
    as broken when nothing happens to be reading it. Gating on the weight would
    make the same corrupt shard raise or not raise depending on a live-reloadable
    number, which is the worst of both -- it would go undetected on exactly the
    arm that is not looking, and then fire later on an arm that is, with the bad
    rows already banked in the replay window. Pinned by
    `tests/test_value_blend_nan_labels.py::test_a_claimed_nan_label_raises_even_with_the_value_head_off`
    so the scope is a test, not an implication.

    ⚑ ONE host transfer for every field, and it is UNAVOIDABLE: a Python
    exception is a host decision, so the counts have to be read. Batched into a
    single `tolist()` rather than one `item()` per field because the training
    loop already pays exactly one sync per microbatch (`_extract_loss_scalars`),
    and this must not turn that into three.

    Absent fields contribute no count and are skipped; with every field absent
    there is no transfer at all.
    """
    present = [(name, c) for name, c in counts if c is not None]
    if not present:
        return
    bad = torch.stack([c for _, c in present]).tolist()
    offenders = [
        f"{name}: {int(n)} row(s)" for (name, _), n in zip(present, bad, strict=True) if n > 0
    ]
    if offenders:
        raise ValueError(
            "non-finite value-target label rows that CLAIM to be valid "
            f"({'; '.join(offenders)}). A row with a zero blend mask may be "
            "non-finite -- it falls back to the game outcome exactly -- but a "
            "row with a non-zero mask asserts the label is real, so this is "
            "dirty label data rather than a training regime. Fix the shard; do "
            "not lower the mask to hide it."
        )


def _q_to_wdl_probs(q: torch.Tensor) -> torch.Tensor:
    q_clamped = q.clamp(-1.0, 1.0)
    win = q_clamped.clamp_min(0.0)
    loss = (-q_clamped).clamp_min(0.0)
    draw = (1.0 - win - loss).clamp_min(0.0)
    return torch.stack((win, draw, loss), dim=1)


def _future_regret_tensor(batch: dict[str, torch.Tensor], source: str) -> tuple[torch.Tensor, torch.Tensor]:
    key, has_key = future_regret_field_names(source)
    return (
        _get_mask(batch, key).to(torch.float32),
        _get_mask(batch, has_key).to(torch.float32),
    )


def piece_counts_from_input(x: torch.Tensor) -> torch.Tensor:
    """Root piece count per row, from the 12 piece-occupancy planes.

    Rounded before comparison because the planes travel through fp16/bf16
    autocast on the eval path, where a 32-piece sum can land at 31.9995 and a
    bare ``> 22`` on the raw sum would still be right but a ``<= 13`` boundary
    row would not be.

    ``flatten(1)`` rather than ``sum(dim=(1, 2, 3))``: production ``x`` is
    (B, planes, 8, 8), but compute_loss is also called on reduced synthetic
    batches, and a hard-coded rank makes this raise ``IndexError`` there — a
    reporting-only split must never be able to take down the loss.
    """
    return torch.round(x[:, :_PIECE_PLANE_COUNT].float().flatten(1).sum(dim=1))


def _phase_split_masks(
    *,
    has_is_selfplay: torch.Tensor,
    is_selfplay: torch.Tensor,
    piece_counts: torch.Tensor,
) -> tuple[tuple[str, torch.Tensor], ...]:
    """selfplay/curriculum + opening/midgame/endgame masks for split loss reporting.

    The phase masks PARTITION every row — there is no ``has_`` gate, because
    every row carries its own board planes. The previous `moves_left` version
    needed one and silently dropped any row whose shard lacked that optional
    field.
    """
    sp_mask = has_is_selfplay * is_selfplay
    cur_mask = has_is_selfplay - sp_mask
  # Identical predicate to `eval.audit.phase_bucket`: end is `<= low`, open is
  # `> high`, mid is the remainder. Written as the same two comparisons rather
  # than as a call so the tensor path stays free of host round-trips.
    end_mask = (piece_counts <= _PHASE_END_MAX_PIECES).to(torch.float32)
    open_mask = (piece_counts > _PHASE_MID_MAX_PIECES).to(torch.float32)
    mid_mask = 1.0 - end_mask - open_mask
    return (
        ("selfplay", sp_mask),
        ("curriculum", cur_mask),
        ("phase_open", open_mask),
        ("phase_mid", mid_mask),
        ("phase_end", end_mask),
    )


# The smallest value the constant tail can take, and it is ARITHMETIC rather than
# empirical: `selfplay/finalize.py` sets the fill to `(worst_surfaced + 1.0) / 2.0` with
# `worst_surfaced` in [0, 1], so the fill lies in [0.5, 1.0] for every row ever built.
# ⇒ a row whose max is below this PROVABLY carries no fill. Exactly representable in
# float16/32/64 alike, so comparing against it needs no tolerance — which is the point:
# `sf_p0_regret` is stored **float16** and any tolerance-based test here would carry a
# constant that silently rots if that dtype ever changes.
_SF_REGRET_MIN_FILL = 0.5


def resolve_sf_regret_gate_keys(
    listed_mass_min: float, unlisted_scale: float,
) -> tuple[float, float]:
    """The two fabricated-tail gate keys, sanitized, WITH A WARNING IF THEY MOVED.

    Non-finite -> the OFF value (``0.0`` / ``1.0``); finite out-of-range -> the
    nearest endpoint of ``[0, 1]``. See ``sf_regret_gate_scale`` for why the two
    cases are deliberately treated differently.

    ⚑⚑ THE WARNING IS THE POINT, and it is what makes this a function rather than
    four inline lines. Neither key is range-validated by ``TrialConfig`` (CLAUDE.md
    category (c)), so before this the two typo classes were SILENT and, worse,
    silently OPPOSITE: ``listed_mass_min: 10`` realized as ``1.0`` and gated every
    tail row, while ``listed_mass_min: 1e400`` realized as ``0.0`` and disabled the
    gate outright. Both echoed the operator's number straight back in
    ``params.json``, and ``sf_own_regret_gated_frac`` reads ``0.0`` for "off" and
    for "disabled by fallback" alike -- so no instrument could tell an operator
    which of the two they had. That is this repo's signature defect exactly: a
    value accepted and then quietly replaced.

    ⚑ It WARNS rather than RAISES, unlike ``SfPolicyFloorParams.resolve`` 70 lines
    up, and the asymmetry is deliberate rather than an oversight. Both keys are
    startup-only, so a raise here is a FAILURE TO BOOT -- and per CLAUDE.md a
    launch-time ``ValueError`` means the process never starts and there is no old
    config to fall back to. The floor buys that risk with a term whose shape is
    irrecoverable if wrong; this gate degrades to a documented, bit-exact identity
    in every bad case, so killing a trial over it would cost more than it saves.
    ⇒ warn loudly, name both numbers, and let the run continue at the safe value.
    The caller that stores the RESULT (``Trainer.__init__``) is what closes the
    loop: from then on the attribute, ``_loss_kwargs`` and the Ray RESULT ROW all
    carry the realized value instead of the typed one.

    ⚑ ``params.json`` DOES NOT. It persists the Ray ``config``, which is what the
    operator typed, and nothing here rewrites it -- so a run configured
    ``listed_mass_min: 10`` trains at ``1.0`` with its saved config still saying
    ``10``. That is why ``tune/trainable_report.py`` emits both realized values as
    result-row columns: read the ROW against the yaml, never the yaml alone. An
    earlier revision of this docstring said "the reported config" and meant the
    row; the ambiguity mattered enough that a reviewer read it as a claim about
    ``params.json``, which would have been false.
    """
    mass_min = float(listed_mass_min)
    scale = float(unlisted_scale)
    eff_min = 0.0 if not math.isfinite(mass_min) else min(max(mass_min, 0.0), 1.0)
    eff_scale = 1.0 if not math.isfinite(scale) else min(max(scale, 0.0), 1.0)
  # ⚑ `!=` and not `math.isclose`: this fires only when the sanitizer actually
  # substituted a value, and NaN != NaN is True, which is the reading we want --
  # a NaN input DID move.
    moved = [
        (name, typed, realized)
        for name, typed, realized in (
            ("sf_own_regret_listed_mass_min", mass_min, eff_min),
            ("sf_own_regret_unlisted_scale", scale, eff_scale),
        )
        if typed != realized
    ]
    if moved:
        detail = "; ".join(
            f"{name}={typed!r} realized as {realized!r}" for name, typed, realized in moved
        )
        warnings.warn(
            f"fabricated-tail gate key out of range: {detail}. Both keys are "
            "probabilities and are clamped to [0, 1]; a non-finite value falls back "
            "to the OFF value (listed_mass_min 0.0 / unlisted_scale 1.0). The trial "
            "continues at the realized value -- `sf_own_regret_gated_frac` reads 0.0 "
            "for a disabled gate and cannot tell you this happened, so fix the yaml.",
            RuntimeWarning,
            stacklevel=2,
        )
    return eff_min, eff_scale


def sf_regret_surfaced_mask(
    reg_vec: torch.Tensor, legal_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-move mask of the moves SF ACTUALLY SURFACED, from the regret vector alone.

    ⚑ NOT EXACT ON A CAP ROW, and the name oversells it there. When a surfaced line is
    >= 1000cp worse than best, `alpha == 1.0` and a REAL capped regret is numerically
    identical to the fill, so `reg < row_max` drops genuinely-surfaced moves: measured
    on live shards this returns 1 surfaced move on rows where 4 were really read, and
    **7.32% of eligible rows have `row_max >= 1.0`**. No current caller is harmed
    because `sf_regret_gate_scale` refuses those rows outright, but this function is
    module-public and a future caller that trusts the name would be wrong. Stated here
    rather than only at the call site.

    ``sf_p0_regret`` is a **constant-tail** construction, not a per-move
    measurement: up to ``sf_multipv`` real normalized cp-regrets, then ONE fitted
    value repeated over every other legal move. ``selfplay/finalize.py``'s comment
    says absent moves "default to 1.0", but that is the CAP, not what production
    stores — measured on live shards the fill is a fitted constant (e.g. 0.5259 on
    a 28-legal-move row) and only 3.75% of legal entries are exactly 1.0. So
    ``reg < 1.0`` is NOT the surfaced set; it selects ~every legal move.

    The fill IS the row MAXIMUM, and that is measured rather than assumed: over
    2350 live rows with >= 8 legal moves, the max is a plateau (multiplicity >= 2)
    in **2350/2350**, median multiplicity 26, and **2350/2350** carry <= 6 values
    strictly below it — exactly ``sf_multipv: 6``. Hence ``reg < row_max`` is the
    surfaced set.

    ⚑ WHY NOT ``sf_multipv_raw``, which stores the move indices directly: that
    field is the PREVIOUS ply's read (``finalize.py`` builds this row's regret from
    ``prepare_multipv(prev_idx)``), and the replay buffer shuffles rows
    independently, so the aligned partner row is not in the batch. Reading it here
    would silently mask the wrong position — the P0-alignment defect that was
    caught once already by an impossible coverage of 1.04.
    """
    return _sf_regret_surfaced_and_row_max(reg_vec, legal_mask.to(torch.bool))[0]


def _sf_regret_surfaced_and_row_max(
    reg_vec: torch.Tensor, legal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(surfaced_mask, row_max)`` from one pass.

    ⚑ BOTH OUTPUTS FROM ONE COMPUTATION, because both callers need both and the
    row max is a full-width ``torch.where`` + ``amax``. ``sf_regret_gate_scale``
    used to call ``sf_regret_surfaced_mask`` (which derived the max internally,
    then discarded it) and then rebuild the very same tensor for its plateau test
    — two 1858-wide temporaries per eligible batch for one quantity.

    ⚑ The alternative — letting the gate compute ``row_max`` itself and inline
    ``legal & (reg < row_max)`` — was rejected: it would put the definition of
    "surfaced" in two places, and that rule is the single most load-bearing line
    in this feature (the whole gate is a no-op under the ``reg < 1.0`` reading of
    it). One private helper keeps one definition AND one computation.
    """
  # Rows with no legal move would make `amax` read the -inf sentinel; clamp the
  # comparison to legal entries only so an empty row yields an all-False mask
  # rather than a NaN that propagates into the loss.
    neg_inf = torch.finfo(reg_vec.dtype).min
    row_max = torch.where(legal, reg_vec, torch.full_like(reg_vec, neg_inf)).amax(
        dim=-1, keepdim=True,
    )
    return legal & (reg_vec < row_max), row_max


def sf_regret_gate_scale(
    reg_vec: torch.Tensor,
    target_probs: torch.Tensor,
    legal_mask: torch.Tensor,
    *,
    listed_mass_min: float,
    unlisted_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row scale for the ``sf_own_regret`` term, plus the gated indicator.

    WHY THIS GATE EXISTS. ``sf_own_regret`` is ``sum_m p_own(m) * regret(m)``, so
    it puts gradient proportional to ``regret(m)``. Every UNSURFACED move shares
    ONE fitted value, so the term carries **no information about their relative
    merit** — it uniformly shoves mass into SF's six weighted by a number SF never
    produced. On rows whose target already lives inside the surfaced set that is
    harmless; on rows whose target lives in the tail it is the dominant signal.
    This scales the term down on exactly the latter rows.

    ⚑ The gate reads the STORED target, not the net's current policy. Gating on
    the net's own mass would make the weight a function of the thing the term is
    trying to move, i.e. a feedback loop that rewards the intervention for having
    happened. Row exposure is a property of the DATA and is fixed at ingest.

    ⚑ Returns a BIT-EXACT identity at the defaults (``listed_mass_min`` 0.0,
    ``unlisted_scale`` 1.0): policy mass is non-negative, so ``mass < 0.0`` is
    False on every row and the scale is all-ones. ``tests/`` asserts that with
    ``torch.equal`` against an ungated run rather than a tolerance.

    ⚑⚑ A ROW WITH NO FABRICATED TAIL IS **ALMOST** NEVER GATED — and the weasel word
    is load-bearing, because an earlier revision of this line said "NEVER" and that
    was FALSE on **22.2% of the gate's own firings**. The gate's justification is
    that the tail's magnitudes are invented, so on a row where SF surfaced EVERY
    legal move there is nothing to distrust and scaling the term down discards real
    supervision. Three such shapes exist and an independent review found all three
    mis-scored by the first version: a row with <= ``sf_multipv`` legal moves (fully
    covered), a forced move (one legal), and a row where a *surfaced* move sits at
    the cp cap so the row max is a real regret rather than the fill.

    ⚑ The 2350-row plateau validation could not see the first two: it was restricted
    to rows with >= 8 legal moves, so it excluded them BY CONSTRUCTION. A second
    review then showed the PLATEAU test alone does not exclude them either — two real
    regrets tying at the max of a fully-covered row read as a tail. What excludes
    them is the arithmetic range of the fill (see ``_SF_REGRET_MIN_FILL``).

    ⚑ MEASURED RESIDUAL, not a guarantee: over 2,931 plateau rows on 8 live shards,
    ~150 carried no fill and the range test rejects **149 of them**. The survivor is a
    fully-covered row whose tied real max is itself >= 0.5. So the honest claim is
    "provably no fill below 0.5", NOT "no false positive" — roughly 1 row in 2,931.

    ⚑ ``gated`` is the rows actually SCALED, not the rows MATCHING the predicate.
    At ``unlisted_scale`` 1.0 the predicate can match while the scale stays 1.0 and
    nothing is downweighted; reporting those as gated would make the metric read
    non-zero on a run where the gate provably did nothing -- a counter that is not
    the mechanism behind it.
    """
    legal = legal_mask.to(torch.bool)
    surfaced, row_max = _sf_regret_surfaced_and_row_max(reg_vec, legal)
  # ⚑ A FABRICATED TAIL IS A PLATEAU AT THE ROW MAX, not merely "some move was not
  # surfaced". `reg < row_max` excludes the argmax by construction, so
  # `surfaced_count < legal_count` is True on EVERY row with distinct values and
  # would classify a fully-covered row as having a tail -- measured: it gated all
  # three no-tail shapes. The fill covers MANY moves with ONE value, so multiplicity
  # >= 2 at the max is the discriminator, and it is the property actually validated
  # on live data (2350/2350 rows plateaued, median multiplicity 26).
  #
  # ⚑ TWO KNOWN LIMITS, both in the SAFE direction (the gate under-fires):
  #   * a tail of exactly ONE move has multiplicity 1 and is indistinguishable from
  #     a real argmax, so such rows are never gated;
  #   * two REAL regrets that are bit-equal at the max would read as a plateau. The
  #     cost is gating one row that had real supervision -- the same cost the
  #     unguarded version paid on every distinct-valued row, now rare instead of
  #     universal.
    at_max = legal & (reg_vec == row_max)
  # ⚑⚑ A ROW WHOSE MAX SITS AT THE CAP IS NEVER GATED. Normalized regret is capped
  # at 1.0, so a REAL regret that hit the cap is numerically INDISTINGUISHABLE from
  # the fill -- and then `reg < row_max` classifies those capped REAL moves as tail,
  # understating the surfaced set. An independent review measured the cost: a
  # cap-plateau row discarded 0.1771 of gradient, **2x the gate's intended-target
  # row (0.0880)**, so this is not a rounding concern. Ties at the max are
  # STRUCTURAL here rather than rare, which is why the "two bit-equal real regrets"
  # caveat an earlier revision called rare was wrong.
  #
  # ⚑ THE COST OF THE CAP GUARD, AT ROW LEVEL — an earlier revision sized it with an
  # ENTRY count (~3.75% of legal entries are exactly 1.0), which is the wrong
  # population for a row-level, gradient-weighted cost. Measured on 3,703 eligible
  # live rows / 98,507 legal entries: **7.32% of ROWS** have `row_max >= 1.0` and are
  # refused, and the plateau-carried term on them averages **0.41366 vs 0.08308 on
  # non-cap rows — 5.0x** — so **28.2% of the total plateau-borne term sits on rows
  # the gate structurally cannot touch**, and its ceiling is ~48.6% of that term at
  # `listed_mass_min 0.5`. The mechanism: `alpha = (worst_surfaced + 1) / 2`, so
  # `alpha == 1.0` exactly when a surfaced line is >= 1000cp worse — a mate or a lost
  # line in the top 6, precisely the rows whose fill is 1.0 on EVERY unsurfaced move.
  # ⇒ **the gate's coverage is anti-correlated with the harm it removes.** Stated
  # because the arm's expected effect is sized off this number.
  # ⚑ On a cap row the surfaced/tail split is UNIDENTIFIABLE, so 0.41366 is an upper
  # bound that includes real capped regrets — which is exactly why refusing is right.
    has_tail = (at_max.to(torch.int64).sum(-1) >= 2) & (row_max.squeeze(-1) < 1.0)
  # ⚑⚑ A FULLY-COVERED ROW CANNOT CARRY THE FILL, AND THE PLATEAU TEST ALONE DOES NOT
  # KNOW THAT. When every legal move was really scored, the row max is a REAL regret;
  # two real regrets tying there (routine, since regrets are integer cp / 1000) make
  # the plateau test fire on a row with no tail at all. An independent review measured
  # it at **22.2% of the gate's firings** at `listed_mass_min 0.5` — and gating
  # ENRICHES for these rows, because excluding the tied max drives `listed_mass` down.
  #
  # The fix is arithmetic, not a heuristic. The builder
  # (`selfplay/finalize.py::_build_sf_p0_regret_vector`) sets
  # `default_regret = (worst_surfaced + 1.0) / 2.0` with `worst_surfaced` in [0, 1],
  # so **the fill always lands in [0.5, 1.0]** ⇒ a row whose max is below 0.5
  # PROVABLY contains no fill. 0.5 is exactly representable in every float dtype, so
  # this needs no tolerance.
  #
  # ⚑ WHY NOT the two alternatives, both of which were measured rather than argued:
  #   * `legal_count > sf_multipv` is exact, but `sf_multipv` is a LABEL-BUILDER
  #     property and the replay window holds ~a day of rows built under whatever
  #     width was live THEN. Reading today's 6 from config would silently mis-judge
  #     older rows. Both tests here read the ROW, so they cannot go stale.
  #   * the algebraic fingerprint `2*row_max - 1 == worst_surfaced` is exact in BOTH
  #     directions, but needs a dtype-dependent tolerance: `sf_p0_regret` is stored
  #     **float16**, whose eps at 0.5 is 4.88e-4, so the residual on genuinely filled
  #     rows runs to ~5e-4 while non-filled rows sit at >= 0.80 — a clean 1600x
  #     separation, but one whose tolerance constant silently rots if the stored dtype
  #     ever changes. Measured on 2,931 plateau rows across 8 live shards, the
  #     fingerprint and the `>= 0.5` test agree to **149 of 150** false positives, so
  #     the exact-but-fragile version buys ~1 row in 2,931.
  # ⇒ take the provable, dtype-exact one.
  #
  # ⚑ RESIDUAL, stated as a measurement and NOT as "never": a fully-covered row whose
  # tied real max is itself >= 0.5 (SF's 6th-best >= 500cp worse than best, and tied)
  # still reads as a tail. **~1 row in 2,931 plateau rows** on live data. The
  # guarantee is "provably no fill below 0.5", not "no false positive".
    has_tail = has_tail & (row_max.squeeze(-1) >= _SF_REGRET_MIN_FILL)
  # ⚑ Normalise over the LEGAL support. `policy_t` is stored fp16 and is not
  # guaranteed to sum to 1.0 after alignment, so an unnormalised sum makes the
  # threshold mean subtly different things on different rows. Rows with no mass
  # at all (`has_policy == 0`) get mass 0.0 and are excluded via `has_tail` only
  # if they genuinely have a tail -- so they are handled explicitly below.
    probs = (target_probs.to(torch.float32) * legal.to(torch.float32)).clamp_min(0.0)
    total = probs.sum(-1)
    listed_mass = torch.where(
        total > 0.0,
        (probs * surfaced.to(torch.float32)).sum(-1) / total.clamp_min(1e-12),
      # No stored target mass on this row => no evidence of tail exposure, so do
      # not gate it. ⚑ `1.0` is not-gated only because `mass_min` is CLAMPED to
      # <= 1.0 below and the comparison is strict `<`. An earlier revision claimed
      # "above every reachable `listed_mass_min`" with NO clamp in place, which was
      # simply false -- `listed_mass_min: 10` reached the comparison and gated every
      # row including these. The guarantee lives in the clamp, not in this constant.
        torch.ones_like(total),
    )
  # ⚑⚑ BOTH keys are clamped, and NON-FINITE values are handled EXPLICITLY rather
  # than by the clamp. Neither key is range-validated by `TrialConfig` (CLAUDE.md category
  # (c)): the schema accepts it, nothing range-checks it, and the consumer gets it
  # raw, so a decimal typo lands silently. Three concrete escapes, all measured:
  #   * `unlisted_scale: -5` gave `scale = -5.0`, making the optimizer MAXIMISE SF
  #     regret on exactly the rows the gate exists to protect;
  #   * `listed_mass_min: 10` gates 100% of rows (mass is a fraction, so 1.0 is the
  #     ceiling), and `-1` silently never fires;
  #   * ⚑ NaN SURVIVES a `min`/`max` clamp for BOTH keys -- Python's `min`/`max`
  #     PROPAGATE NaN rather than rejecting it (every comparison with NaN is False, so
  #     the first argument wins), so a clamp is a RANGE guard and never a finiteness
  #     guard. ⚑ But the CONSEQUENCE differs between the two keys and only one is a
  #     poisoning hazard, so they must not be quoted together:
  #       - `unlisted_scale: .nan` takes `total` to NaN **even at
  #         `w_sf_own_regret: 0.0`** (because `0.0 * nan == nan`), while
  #         `sf_own_regret_gated_frac` still reads 0.0. Worst shape available:
  #         production dies and the instrument says nothing happened.
  #       - `listed_mass_min: .nan` CANNOT poison anything: `mass < nan` is False on
  #         every row, so the gate simply never fires. On NaN ALONE its guard is
  #         DEFENSIVE-ONLY -- no observation distinguishes it from the unguarded
  #         path, and the `.nan` half of it was correctly labelled an EQUIVALENT
  #         mutant when this guard was `math.isnan`.
  #         ⚑ THAT LABEL DIED WITH THE `isfinite` WIDENING BELOW, and it is worth
  #         saying why rather than just deleting it: an equivalent mutant is
  #         equivalent with respect to the REACHABLE INPUT SET, not in itself, so
  #         widening the guard's domain can make one killable without the guard's
  #         own line changing. `listed_mass_min: .inf` now falls back to 0.0 (gate
  #         off) where the unguarded path would clamp it to 1.0 and gate every
  #         tail row -- a difference any batch shows. Both halves of the guard are
  #         behaviourally covered now
  #         (`test_a_non_finite_gate_value_falls_back_to_off_not_to_maximum_on`).
  #   * ⚑⚑ `.inf` IS NOT NaN, AND AN `isnan` GUARD LETS IT THROUGH INTO THE CLAMP
  #     -- where it lands on the RANGE ENDPOINT, which for both keys is the
  #     MAXIMUM-ON value, not the off value. `listed_mass_min: .inf` clamps to
  #     1.0 and gates every row carrying a tail; `unlisted_scale: -.inf` clamps
  #     to 0.0 and suppresses those rows completely. An earlier revision of this
  #     block guarded with `math.isnan` while the line below it promised
  #     "non-finite falls back to the OFF value" -- a claim the code did not keep
  #     for either infinity, which is this repo's signature defect stated in a
  #     comment. ⚑ It is REACHABLE from yaml without anyone typing `.inf`, and the
  #     route is worth spelling correctly because an earlier revision got it
  #     wrong: `yaml.safe_load("1e400")` does NOT return `inf`, it returns the
  #     STRING `'1e400'` (PyYAML's 1.1 float resolver needs a `.`, so `1.0e400`
  #     is a string too). The infinity is minted one layer down, by the
  #     `float(config.get(...))` in `trainer_kwargs_from_config`: `float('1e400')`
  #     IS `inf`. So the escape is real end to end -- an over-long exponent typo
  #     reaches the consumer as an infinity -- but it arrives through the float
  #     conversion, not through the yaml parser. MEASURED, not read off the spec.
  # ⇒ `math.isfinite` first, then clamp. Every non-finite input -- NaN and both
  # infinities -- falls back to the OFF value, so a typo degrades to "gate
  # disabled" rather than to a poisoned run OR to a silently maximal one. The
  # clamp still handles finite out-of-range values (`10` -> 1.0, `-5` -> 0.0),
  # because those ARE plausible decimal typos for an in-range value and the
  # nearest endpoint is the honest reading of them.
  # ⚑ The rule lives in `resolve_sf_regret_gate_keys` so the Trainer can apply it
  # ONCE at construction and store the realized value, rather than every caller
  # re-deriving it. Re-resolving an already-resolved pair here is a no-op and
  # stays silent, which is what keeps the warning to one line per trial.
    mass_min, eff_scale = resolve_sf_regret_gate_keys(listed_mass_min, unlisted_scale)
    matches = (listed_mass < mass_min) & has_tail
    scale = torch.where(matches, torch.full_like(listed_mass, eff_scale),
                        torch.ones_like(listed_mass))
  # Rows actually SCALED -- empty whenever `eff_scale` is 1.0, by construction.
    gated = (matches & (scale < 1.0)).to(torch.float32)
    return scale, gated


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    w_policy: float = 1.0,
    w_soft: float = 0.5,
    w_future: float = 0.15,
    w_sf_own: float = 0.0,
    w_sf_own_regret: float = 0.0,
  # Fabricated-tail gate on the sf_own_regret term. Defaults are a BIT-EXACT
  # identity — see `sf_regret_gate_scale`. Both are read here, not just accepted:
  # ⚑ this function returns the COUNT `sf_own_regret_gated_rows`; the ratio an
  # operator actually reads, `sf_own_regret_gated_frac`, is derived from it
  # against `sf_own_regret_rows` by `_RATIO_METRIC_FIELDS` in `train/trainer.py`.
  # Naming the derived column as though it came back from here sends anyone
  # grepping for it to the wrong module.
    sf_own_regret_listed_mass_min: float = 0.0,
    sf_own_regret_unlisted_scale: float = 1.0,
    w_wdl: float = 1.0,
    w_sf_move: float = 0.15,
    w_sf_eval: float = 0.15,
    w_categorical: float = 0.10,
    w_volatility: float = 0.05,
    w_sf_volatility: float | None = None,
    w_moves_left: float = 0.02,
    sf_wdl_frac: float = 0.0,
    search_wdl_frac: float = 0.0,
    sf_wdl_conf_power: float = 0.0,
    sf_wdl_draw_scale: float = 1.0,
    sf_wdl_temperature: float = 1.0,
    sf_search_dampen_sf_low: float = 0.0,
    sf_search_dampen_sf_high: float = 0.0,
    use_adjusted_wdl_target: bool = False,
    adjusted_wdl_regret_source: str = "sum",
    adjusted_wdl_regret_scale: float = 1.0,
    adjusted_wdl_regret_cap: float = 0.0,
    wdl_terminal_outcome_plies: int = 0,
    wdl_terminal_outcome_full_plies: int = 2,
    wdl_terminal_outcome_sf_frac: float = 0.0,
    moves_left_max_plies: float = 0.0,
    soft_policy_min_tv: float = 0.0,
    policy_target_temp: float = 1.0,
    sf_sparse_params: SfTargetParams | None = None,
    sf_policy_floor: SfPolicyFloorParams | None = None,
    sf_shape: SfShapeParams | None = None,
    report_exact_masked_sums: bool = False,
    exact_corpus_rows: int | None = None,
    exact_objective_mask_weights: Mapping[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute multi-head training loss.

    ``soft_policy_min_tv`` masks the soft-policy loss to zero on samples
    whose soft target is within that total-variation distance of the hard
    target (they're a deterministic retempering of the same distribution —
    see scripts/probe_policy_targets.py). 0.0 keeps current behavior exactly.

    ``wdl_terminal_outcome_plies`` (0 = OFF, and bit-identical to the blend
    without this feature) moves part of the SEARCH share of the value target
    onto the recorded game outcome for rows within that many plies of the
    game's end; ``wdl_terminal_outcome_full_plies`` is the distance at or below
    which the whole search share moves. ``moves_left_max_plies`` is the
    selfplay ply cap the ``moves_left`` field was normalized by and is
    REQUIRED once the feature is on — see ``terminal_outcome_transfer_taper``.

    ``wdl_terminal_outcome_sf_frac`` (default 0.0 — the SF share is then never
    touched at any ``d``) is the fraction of the SF share the outcome may
    ADDITIONALLY take, on the same taper. It exists for the offline screen's
    aggressive arm; the SF component is load-bearing supervision, so raising it
    is a training-target experiment in its own right.

    ``sf_policy_floor`` is the SF-approved-move probability floor (see
    ``SfPolicyFloorParams`` / ``sf_policy_floor_deficit``). ``None`` is the same
    as the all-default object: weight 0.0, so ``total`` is bit-identical to a
    build without the term, while ``sf_policy_floor`` / ``sf_policy_floor_binds``
    are still reported.

    ``sf_shape`` is the SF-shape conditional-KL term (see ``SfShapeParams`` /
    ``sf_shape_conditional_kl``): it matches ``policy_own``'s shape INSIDE the
    set of moves Stockfish actually scored to SF's own, and provably asserts
    nothing about the rest of the move list. ``None`` is the same as the
    all-default object: weight 0.0, so ``total`` is bit-identical to a build
    without the term, while the entropy instrument
    (``sf_shape_h_sf_given_s_sum`` and friends) is reported either way -- it is
    a MONITOR first and a loss second, and the drift it watches for ran for
    months because no column carried it.

    ``sf_sparse_params`` switches the ``policy_sf`` loss to sparse CE over
    gathered log-probs (train/sparse_sf_ce.py) for rows carrying sparse
    MultiPV labels; rows without them keep the dense soft CE. None (the
    default) leaves the dense path untouched.

    Value-loss reporting: ``wdl_ce`` and ``blended_wdl_ce`` are the same
    tensor — the blended soft CE that ``total`` (and therefore every
    gradient) is built from. The hard one-hot CE against the recorded game
    result is a diagnostic and is reported separately as ``wdl_onehot_ce``.
    """
    net_mask = _get_mask(batch, "is_network_turn", default=1.0).to(torch.float32)

    def _apply_legal_mask(logits: torch.Tensor) -> torch.Tensor:
        return apply_policy_mask_to_logits(logits, batch, "legal_mask", "has_legal_mask")

    base_policy_logits = outputs["policy"] if "policy" in outputs else outputs.get("policy_own")
    if base_policy_logits is None:
        raise KeyError("Model outputs must include either 'policy' or 'policy_own'.")

    # Legal-masked base logits, computed once and reused by the main policy CE,
    # the sf_p0 CE, and the sf_own_regret softmax below (all train policy_own /
    # policy in the same legal space) — the mask is a full-width align+fill, so
    # recomputing it per term wastes work every training step.
    masked_base = _apply_legal_mask(base_policy_logits)

    aligned_pol_target = align_policy_target(
        batch["policy_t"], int(base_policy_logits.shape[-1]),
    )
  # Kept UNRETEMPERED for the floor's collar, which needs only the argmax -- the
  # move search actually played. Retempering is monotone so the argmax is the
  # same today, but reading the pre-reshape target means the collar cannot start
  # protecting a different move because someone moved `policy_target_temp`, and
  # `policy_target_temp` is pinned OFF in eval (see `_eval_loss_kwargs`), which
  # would otherwise make the collar a different member in train and in eval.
    pol_target = retemper_main_policy_target(
        aligned_pol_target, temp=float(policy_target_temp),
    )
    pol_ce = soft_cross_entropy(masked_base, pol_target)
    zero_loss = torch.zeros_like(pol_ce)
    has_policy = _get_mask(batch, "has_policy", default=1.0)
  # The legal-masked `policy_own` distribution, computed ONCE: the SF-shape
  # family, the floor and the matched-support instrument all read the same
  # tensor, so no two of them can end up describing slightly different
  # distributions.
    base_probs = torch.softmax(masked_base, dim=-1)
    base_legal = policy_legal_bool(batch, width=int(base_policy_logits.shape[-1]))
  # ⚑ MATCHED-SUPPORT instrument for the ORDINARY policy target -- the one the
  # main CE trains against, which has NOTHING to do with the SF term below. Both
  # entropies are taken over the TARGET'S support and renormalized there, and
  # what the restriction drops is published separately as
  # `policy_tail_mass_ours`. See `matched_support_entropy_stats` for the
  # confound this shape exists to make impossible.
    pol_support = matched_support_entropy_stats(base_probs, pol_target, base_legal)

    has_soft = _get_mask(batch, "has_policy_soft")
    has_soft_before_keep = has_soft
    soft_logits = outputs.get("policy_soft", base_policy_logits)
    soft_target = batch.get("policy_soft_t")
    soft_ce = (
        soft_cross_entropy(
            _apply_legal_mask(soft_logits),
            align_policy_target(soft_target, int(soft_logits.shape[-1])),
        )
        if soft_target is not None else zero_loss
    )
    soft_mask_kept_frac = torch.ones((), device=pol_ce.device, dtype=torch.float32)
    if float(soft_policy_min_tv) > 0.0 and soft_target is not None:
        # Drop the soft loss where TV(hard, soft) is below the threshold: there
        # the soft target is a retempering of (nearly) the same distribution, so
        # for the trunk the gradient largely duplicates the main policy CE (the
        # policy_soft head itself still loses its only signal on masked rows).
        # Strictly off at the default 0.0.
        _soft_aligned = align_policy_target(soft_target, int(base_policy_logits.shape[-1]))
        _tv = 0.5 * (pol_target.float() - _soft_aligned.float()).abs().sum(dim=-1)
        _keep = (_tv >= float(soft_policy_min_tv)).to(has_soft.dtype)
        soft_mask_kept_frac = masked_mean(_keep, net_mask * has_soft)
        has_soft = has_soft * _keep

  # Future policy (t+2): target and legal mask are in the t+2 move space.
    has_future = _get_mask(batch, "has_future")
    future_logits = outputs.get("policy_future", base_policy_logits)
    future_target = batch.get("future_policy_t")
    if future_target is not None:
        future_ce = soft_cross_entropy(
            apply_policy_mask_to_logits(future_logits, batch, "future_legal_mask", "has_future_legal_mask"),
            align_policy_target(future_target, int(future_logits.shape[-1])),
        )
    else:
        future_ce = zero_loss

    # P0 own-move SF teacher on policy_own (the head MCTS reads as the search
    # prior). Unlike policy_sf (a separate, search-invisible head trained on the
    # opponent reply), this blends SF's recommended move for THIS position into
    # the prior itself. Same legal space as the main policy. Masked to the ~15%
    # eligible selfplay rows (has_sf_p0).
    has_sf_p0 = _get_mask(batch, "has_sf_p0")
    sf_p0_target = batch.get("sf_p0_policy_t")
    if sf_p0_target is not None:
        sf_p0_ce = soft_cross_entropy(
            masked_base,
            align_policy_target(sf_p0_target, int(base_policy_logits.shape[-1])),
        )
    else:
        sf_p0_ce = zero_loss

    # Regret-weighted SF teacher on policy_own: minimize the net's EXPECTED SF
    # cp-regret = sum_m p_own(m) * regret(m), where regret(m) in [0,1] is the
    # normalized cp loss vs SF's best move at THIS (P0) position. Pushes mass
    # toward low-regret moves; auto-weighted by how much each position matters
    # (flat positions have tiny regrets -> tiny gradient). Same legal-masked
    # policy_own head as sf_p0_ce, masked to eligible selfplay rows.
    #
    # ⚑⚑ THIS IS THE MAGNITUDE ARM, AND IT IS DEAD ON ALL THREE POPULATIONS
    # TESTED: `sum_m p_m * r_m` scored AT CHANCE, and significantly worse than a
    # RANDOM floor. `w_sf_own_regret` stays 0.0; do not wire it to anything, do
    # not change that default, and do not model a new term on its shape. The
    # floor below is structurally different on purpose -- a one-sided floor on a
    # MEMBERSHIP set, not a regret-weighted expectation over the whole
    # distribution, which is what let a confident wrong argmax pay for this
    # term out of its tail.
    has_sf_p0_regret = _get_mask(batch, "has_sf_p0_regret")
    sf_p0_regret_t = batch.get("sf_p0_regret_t")
  # One-sided probability FLOOR on the SF-approved moves, on the same head and
  # the same rows. Complementary to the term above rather than a variant of it:
  # `sf_own_regret` is a whole-distribution pull that a confident wrong argmax
  # can pay for out of the tail, while this one only ever ADDS mass, and only to
  # moves SF ranks at or above the one we picked. Deliberately modest -- it
  # raises SF's move toward search visibility; noisy admission remains empirical,
  # so this is not itself a policy correction or a search-inclusion theorem.
    floor_params = sf_policy_floor if sf_policy_floor is not None else SfPolicyFloorParams()
  # SF-SHAPED policy target on the same head and the same rows: a DISTRIBUTION
  # over the surfaced set, where the two terms above are a scalar penalty and a
  # one-sided floor. `sum_m p_m * r_m` teaches ORDERING and its gradient carries
  # a factor `p_i`, so it cannot reach a move it has already starved; the floor
  # only ever adds mass, and only to a membership set. Neither says anything
  # about the SHAPE of the surfaced tail, which is the quantity we are measurably
  # wrong about (our policy is sharper than its own SF teacher on 63.8% of rows).
    shape_params = sf_shape if sf_shape is not None else SfShapeParams()
    if sf_p0_regret_t is not None:
        po_probs = base_probs
        reg_vec = align_action_values(sf_p0_regret_t, int(base_policy_logits.shape[-1]))
        sf_legal = base_legal
        sf_own_regret = (po_probs * reg_vec).sum(-1)
      # ⚑ Gated by the row's SURFACED-set exposure, using the STORED target
      # (`batch["policy_t"]`) — NOT `pol_target`, which has already been through
      # `retemper_main_policy_target`. The gate must describe the DATA, so a
      # training knob must not be able to move which rows it selects.
      #
      # ⚑⚑ `legal_mask` IS FETCHED WITH `.get`, NOT BY SUBSCRIPT. It is OPTIONAL
      # everywhere else in this function — every other consumer goes through
      # `_get_mask`/`apply_policy_mask_to_logits`/`.get` — and subscripting it
      # here raises `KeyError` on 6 tests in `test_sf_p0_teacher_metrics.py`
      # (re-measured on this tree: 49 -> 43 passed; the `test_phase_loss_buckets.py`
      # half of the older count belonged to the `policy_t` guard that is gone).
      # Without legality the surfaced set is not identifiable, so the gate DOES NOT
      # FIRE rather than guessing.
      # ⚑ Do not "fix" this with `ones_like`, but NOT for the reason an earlier
      # revision of this comment gave. It said an all-True mask "makes the row max
      # the padding fill and the gate silently never fire on any row". THAT IS
      # FALSE, and measured to be false: `finalize.py::_build_sf_p0_regret_vector`
      # writes `np.full((POLICY_SIZE,), default_regret)` -- the fill covers ILLEGAL
      # indices too -- so an all-True mask leaves `row_max` and the surfaced set
      # unchanged, and the gate fires IDENTICALLY. It is also identical when the
      # padding densifies to 0.0 instead, because the stored target puts no mass
      # there, so `listed_mass` cannot move. The real objection is the row below.
      #
      # ⚑ THE TARGET IS `aligned_pol_target`, NOT A SECOND READ OF `policy_t`.
      # Earlier waves of this branch did `align_policy_target(batch.get("policy_t"),
      # ...)` here, when `policy_t` was still optional in this function. It is not
      # any more: PR #448 hoisted `aligned_pol_target = align_policy_target(
      # batch["policy_t"], ...)` to the top of `compute_loss` as a HARD SUBSCRIPT,
      # so a batch without it now raises long before this line and the second
      # `.get` guarded a branch that can no longer be reached. Reusing the hoisted
      # tensor drops a redundant full-width align per training step and, more to
      # the point, keeps the two readers provably identical -- and it is the SAME
      # decision this gate already made for its own reason: `aligned_pol_target`
      # is the STORED target, kept UNRETEMPERED (see its comment above), which is
      # exactly what the gate needs. `pol_target` is the retempered one and must
      # never be used here; `test_the_gate_reads_the_STORED_target_not_the_
      # retempered_one` fails if it is.
      # ⚑ `has_legal_mask` is deliberately NOT conjoined here, unlike `masked_base`
      # above, which routes through `apply_policy_mask_to_logits(..., "legal_mask",
      # "has_legal_mask")` and multiplies the mask by the row's flag. Reasons, in order:
      # the failure direction is SAFE (a row whose flag is clear but whose mask is
      # all-zero yields `legal` all-False => no plateau => `has_tail` False => never
      # gated, so it degrades to the identity, never to a wrong scale); the live
      # incidence is **0 of 3,426 eligible rows** across 8 shards of the running trial;
      # and `eval/era_probe.py` documents that `has_sf_p0_regret` set with
      # `has_legal_mask` clear did occur HISTORICALLY, so the rows can exist in an old
      # window. If a future change makes an unflagged mask non-zero rather than absent,
      # conjoin the flag -- the guard here is the all-zero shape, not the flag.
      # ⚑ AND THAT IS THE WHOLE REASON THIS DOES NOT CALL `policy_legal_bool`
      # (added to this module by PR #448, after this branch was cut). MEASURED, on
      # a production-shaped 1858-wide row under both padding conventions:
      #
      #   legality source                          scale   gated
      #   true legal_mask                          0.000    1.0
      #   policy_legal_bool / ones_like            0.000    1.0
      #   unflagged row, all-zero legal_mask       1.000    0.0   <- the difference
      #
      # ⇒ the two agree everywhere EXCEPT on `has_legal_mask == 0` rows, and that
      # single column is the argument. `policy_legal_bool` returns
      # `aligned | (has_legal_mask <= 0.5)`, so on such a row it hands back a
      # FABRICATED 1858-move legal set and the gate would scale the term down off
      # it. This path refuses instead. Correct for the floor -- its contract is
      # "the support the softmax actually has", and the softmax is unmasked there
      # -- and wrong for a gate whose whole subject is which moves SF really
      # listed. Two callers, one key, different questions; sharing the helper
      # would silently give the gate the floor's answer on exactly the rows
      # `eval/era_probe.py` says exist in older windows.
        legal_for_gate = batch.get("legal_mask")
        if legal_for_gate is not None:
            sf_own_regret_scale, sf_own_regret_gated = sf_regret_gate_scale(
                reg_vec,
                aligned_pol_target,
                align_policy_mask(legal_for_gate, int(base_policy_logits.shape[-1])),
                listed_mass_min=sf_own_regret_listed_mass_min,
                unlisted_scale=sf_own_regret_unlisted_scale,
            )
            sf_own_regret = sf_own_regret * sf_own_regret_scale
        else:
            sf_own_regret_gated = torch.zeros_like(sf_own_regret)
      # ⚑ The floor is computed from the UNGATED `po_probs`/`reg_vec` and is not
      # scaled by the gate. Deliberate: the gate's subject is the fabricated
      # CONSTANT TAIL that `sf_own_regret`'s expectation integrates over, and the
      # floor never touches that tail -- it only ever ADDS mass to the surfaced,
      # SF-approved set, so there is nothing fabricated in its support to gate.
        floor_out = sf_policy_floor_deficit(
            po_probs, reg_vec,
            policy_legal_bool(batch, width=int(base_policy_logits.shape[-1])),
            aligned_pol_target * has_policy.unsqueeze(-1),
            params=floor_params,
        )
        sf_floor = floor_out.deficit
        sf_floor_binds = floor_out.binds
      # ⚑ THE GRAPH IS BUILT ONLY WHEN THE TERM IS IN THE OBJECTIVE, and the decision
      # lives HERE rather than inside the kernel. `sf_shape_conditional_kl` is a maths
      # function and stays unconditionally differentiable -- an earlier revision put this
      # guard inside it and broke five kernel tests that legitimately check gradient
      # properties at default params, which is the signal that the policy belonged at the
      # CALLER. At the shipped `w = 0.0` the KL VALUE is still wanted (`m_sf_shape` is the
      # instrument the whole change exists to expose) but its GRAPH is not: the term is
      # skipped below under `if w == 0.0: continue`, so `total.backward()` never traverses
      # it, while every microbatch carrying `sf_p0_regret_t` still allocated a full-width
      # fp32 log-softmax graph that stayed alive until the losses dict dropped. A
      # default-ON memory tax for a default-OFF term. Found by an independent grok review
      # of PR #479. Values are identical either way; only graph recording changes.
        with torch.set_grad_enabled(
            torch.is_grad_enabled() and float(shape_params.w) != 0.0
        ):
            shape_out = sf_shape_conditional_kl(
                masked_base, po_probs, reg_vec, sf_legal, params=shape_params,
            )
  # "Did the term SELECT anything", the column that separates a weight that
  # reaches the loss and does nothing from a dead knob. A row with fewer than two
  # surfaced moves is EXACTLY zero at every weight (see
  # `sf_shape_conditional_kl`), so it is a structural zero in the mean and only
  # this rate can say so.
        sf_shape_active = (shape_out.surfaced_count >= 2.0).to(torch.float32)
    else:
        sf_own_regret = zero_loss
        sf_own_regret_gated = torch.zeros_like(zero_loss)
        sf_floor = zero_loss
        sf_floor_binds = zero_loss
        floor_out = SfPolicyFloorOutputs(*([zero_loss] * 7))
        shape_out = SfShapeReadout(
            kl=zero_loss, h_sf_given_s=zero_loss, h_ours_given_s=zero_loss,
            h_ours_full_legal=zero_loss, surfaced_count=zero_loss,
            surfaced_mass=zero_loss, p_sf_best=zero_loss,
            regret_cp_given_s=zero_loss,
        )
        sf_shape_active = zero_loss

  # DIAGNOSTIC ONLY — hard one-hot CE against the recorded game result. The
  # optimizer never sees this term (see ``blended_wdl_ce`` below, which is the
  # value loss in ``total``). Reported as ``wdl_onehot_ce`` so nothing can
  # mistake it for the trained loss again; see docs/rl_loop_audit.md I7.
    wdl_onehot_ce = F.cross_entropy(outputs["wdl"], batch["wdl_t"], reduction="none")

  # SF move prediction: target and legal mask are in the t+1 move space (opp POV).
    has_sf_move = _get_mask(batch, "has_sf_move")
    sf_pol_logits = outputs.get("policy_sf")
    sf_policy_target = batch.get("sf_policy_t")
    has_sf_policy = _get_mask(batch, "has_sf_policy").to(torch.float32)
    if sf_policy_target is not None:
        # ``has_sf_move`` predates the dense SF-policy mask, so it remains a
        # fallback only when this row carries an actual dense distribution.
        # Mixed-schema concatenation zero-fills a missing target; treating the
        # best-move flag alone as eligibility would add that targetless row to
        # the denominator while its soft CE contributes exactly zero.
        dense_target_present = (
            sf_policy_target.to(torch.float32).sum(dim=-1) > 0.0
        ).to(torch.float32)
        has_sf_policy = torch.maximum(
            has_sf_policy,
            has_sf_move.to(torch.float32) * dense_target_present,
        )
    if sf_pol_logits is None:
        # Deliberately tolerant: partial models (offline rigs, exported subsets)
        # rely on "absent optional head -> zero loss". The `enable_policy_sf_head:
        # false` + `w_sf_move > 0` combination that this tolerance would hide is
        # caught ONCE at Trainer construction instead -- see
        # `Trainer._assert_gated_heads_exist`.
        sf_move_ce = zero_loss
    else:
        masked_sf_logits = apply_policy_mask_to_logits(
            sf_pol_logits, batch, "sf_legal_mask", "has_sf_legal_mask",
        )
        sf_move_ce = (
            soft_cross_entropy(
                masked_sf_logits,
                align_policy_target(sf_policy_target, int(sf_pol_logits.shape[-1])),
            )
            if sf_policy_target is not None else zero_loss
        )
        sf_legal = batch.get("sf_legal_mask")
        if sf_sparse_params is not None and "sf_multipv_raw" in batch and sf_legal is not None:
            sparse_ce, sparse_ok = sparse_sf_policy_ce(
                masked_sf_logits, batch, params=sf_sparse_params,
                legal_aligned=align_policy_mask(sf_legal, int(sf_pol_logits.shape[-1])),
            )
            keep_sparse = sparse_ok > 0
            sf_move_ce = torch.where(keep_sparse, sparse_ce, sf_move_ce)
            # Rows whose shards no longer carry a dense target (the
            # record_dense_sf_policy=false transition) still train via the
            # sparse path; widen the head mask to include them.
            has_sf_policy = torch.maximum(
                has_sf_policy.to(torch.float32), sparse_ok.to(torch.float32),
            )

    has_sf_wdl = _get_mask(batch, "has_sf_wdl")
    sf_wdl_probs = _normalize_sf_wdl_probs(batch.get("sf_wdl"), temperature=sf_wdl_temperature)
    sf_eval_logits = outputs.get("sf_eval")
    sf_eval_ce = (
        soft_cross_entropy(sf_eval_logits, sf_wdl_probs)
        if sf_eval_logits is not None and sf_wdl_probs is not None else zero_loss
    )

    has_search_wdl = _get_mask(batch, "has_search_wdl")
    search_wdl_probs = _normalize_sf_wdl_probs(batch.get("search_wdl"))

    wdl_t = batch["wdl_t"].to(torch.int64)
    game_oh = F.one_hot(wdl_t, 3).float()
    if bool(use_adjusted_wdl_target):
        future_regret, has_future_regret = _future_regret_tensor(batch, adjusted_wdl_regret_source)
        # Individual regrets are stored in winrate units [0, 1], while Q spans [-1, 1].
        # The 2x scale maps a single regret onto Q units; cumulative sources still need
        # explicit source/scale/cap choices. The one-sided shift intentionally discounts
        # W->D and D->L; a drawn game with future SF mistakes is treated as
        # under-converted anti-engine value.
        # Clamp the scalar config and per-sample tensor separately; they guard different inputs.
        correction = REGRET_TO_Q_SCALE * max(0.0, float(adjusted_wdl_regret_scale)) * future_regret.clamp_min(0.0)
        if float(adjusted_wdl_regret_cap) > 0.0:
            correction = correction.clamp_max(float(adjusted_wdl_regret_cap))
        game_q = game_oh[:, 0] - game_oh[:, 2]
        adjusted_wdl_probs = _q_to_wdl_probs(game_q - correction)
        has_adjusted_wdl = has_future_regret.unsqueeze(1)
        game_target = has_adjusted_wdl * adjusted_wdl_probs + (1.0 - has_adjusted_wdl) * game_oh
    else:
        game_target = game_oh
    # Regret adjustment is a heuristic on the game-outcome component only.
    # Missing/dampened SF/search WDL labels fall back to the raw result so the
    # adjusted target does not silently expand beyond ``game_frac``.
    blend_fallback_target = game_oh
    sf_wdl_frac_f, search_wdl_frac_f, game_frac = normalize_value_blend_fracs(
        sf_wdl_frac, search_wdl_frac,
    )
    target = game_frac * game_target
    sf_available = has_sf_wdl.float()
    search_available = has_search_wdl.float()

  # Per-sample SF/search disagreement is split by direction so each side of
  # the lever can be dampened independently:
  #   sf_low : SF says STM losing but search says STM winning. Real-data
  #            calibration (handicapped-opponent regime) shows SF is often
  #            wrong about *outcomes* here (the handicapped opponent fails
  #            to convert), but SF is still right about *objective* eval —
  #            partial dampening keeps that signal alive.
  #   sf_high: SF says STM winning but search says STM losing. Less
  #            common; SF over-confident, search hedging. Usually best to
  #            dampen lightly or not at all (SF is the search-horizon
  #            reference).
    if sf_wdl_probs is not None and search_wdl_probs is not None:
        sf_sig = sf_wdl_probs[:, 0] - sf_wdl_probs[:, 2]
        sr_sig = search_wdl_probs[:, 0] - search_wdl_probs[:, 2]
        joint = sf_available * search_available
        agree = ((sf_sig * sr_sig) > 0).float() * joint
        dis_sf_low = ((sf_sig < 0) & (sr_sig > 0)).float() * joint
        dis_sf_high = ((sf_sig > 0) & (sr_sig < 0)).float() * joint
        sf_search_joint_rows = joint.sum()
        joint_count = sf_search_joint_rows.clamp_min(1.0)
        sf_search_agree_frac = (agree.sum() / joint_count).detach()
        sf_search_disagree_sf_low_frac = (dis_sf_low.sum() / joint_count).detach()
        sf_search_disagree_sf_high_frac = (dis_sf_high.sum() / joint_count).detach()
    else:
        dis_sf_low = torch.zeros_like(sf_available)
        dis_sf_high = torch.zeros_like(sf_available)
        zero_scalar = torch.zeros((), device=sf_available.device)
        joint = torch.zeros_like(sf_available)
        sf_search_agree_frac = zero_scalar
        sf_search_disagree_sf_low_frac = zero_scalar
        sf_search_disagree_sf_high_frac = zero_scalar

    keep = 1.0 - (
        float(sf_search_dampen_sf_low) * dis_sf_low
        + float(sf_search_dampen_sf_high) * dis_sf_high
    )
    sf_effective = sf_available * keep
    sf_effective_b = sf_effective.unsqueeze(1)
  # Terminal-proximal outcome: within `wdl_terminal_outcome_plies` of the end,
  # part of the SEARCH share (and, only when `wdl_terminal_outcome_sf_frac` is
  # raised off its 0.0 default, part of the SF share) is handed to the recorded
  # game outcome, which is the lower-noise estimator there — see
  # `terminal_outcome_transfer_taper`. The global `game_frac` is untouched and
  # every component still sums to 1 per row, so the blend does too.
  #
  # ⚑ The SF share is LOAD-BEARING supervision. `wdl_terminal_outcome_sf_frac`
  # exists for an OFFLINE screen arm and defaults to 0.0, which leaves the SF
  # weight exactly at the realized `sf_wdl_frac` at every `d`.
    terminal_taper = terminal_outcome_transfer_taper(
        batch,
        plies=int(wdl_terminal_outcome_plies),
        full_plies=int(wdl_terminal_outcome_full_plies),
        max_plies=float(moves_left_max_plies),
    )
    search_available_b = search_available.unsqueeze(1)
  # ⚑ A PRESENT-BUT-NaN LABEL ROW IS NOT DISARMED BY ITS OWN MASK. `0.0 * nan`
  # is `nan`, so before this call a row with `has_sf_wdl == 0` and a NaN
  # `sf_wdl` still poisoned `target` -> `blended_wdl_ce` -> `total`, at ANY
  # frac including 0.0 and through `w_wdl = 1.0`, which no zero weight could
  # ever have disarmed. Field ABSENCE was handled (the `is None` fallback);
  # field PRESENCE with a NaN row was not. See `_finite_blend_component`.
    sf_component, sf_bad_rows, sf_unclaimed_bad = _finite_blend_component(
        sf_wdl_probs, raw=batch.get("sf_wdl"),
        weight=sf_effective_b, fallback=blend_fallback_target,
    )
    search_component, search_bad_rows, search_unclaimed_bad = _finite_blend_component(
        search_wdl_probs, raw=batch.get("search_wdl"),
        weight=search_available_b, fallback=blend_fallback_target,
    )
  # ⚑ THE SUBSTITUTION MUST NOT BE SILENT. An unclaimed non-finite row is
  # tolerated -- it takes the fallback -- but "tolerated" and "invisible" are
  # different things, and a guard that quietly repairs corrupt input is the
  # accepted-then-ignored shape this repo keeps re-growing. Counted for BOTH
  # fields, on-device, and announced once per iteration by `Trainer.train_steps`.
    blend_unclaimed_nonfinite_rows = torch.zeros(
        (), device=blend_fallback_target.device, dtype=torch.float32,
    )
    for _unclaimed in (sf_unclaimed_bad, search_unclaimed_bad):
        if _unclaimed is not None:
            blend_unclaimed_nonfinite_rows = blend_unclaimed_nonfinite_rows + _unclaimed
  # ⚑ MUST PRECEDE THE FIRST USE OF EITHER COMPONENT. A row that CLAIMS the
  # label and is NaN gets `fallback` from the sanitiser too, so without this
  # check the value head would train on the game outcome while the shard
  # asserts an SF opinion -- silently, which is worse than the NaN was.
    _assert_blend_labels_finite(
        (("sf_wdl", sf_bad_rows), ("search_wdl", search_bad_rows)),
    )
  # ⚑ THE TWO FALLBACKS, COUNTED — the denominators of the outcome-borne share.
  # BOTH components fall back to `blend_fallback_target` (the raw one-hot), and
  # until 2026-08-16 only the SF side had a count, so `search_wdl_frac` — 0.70
  # of the lc0 control's value target — could land entirely on the outcome with
  # every guard reading clean (PR #438 review F1).
  #
  # These are the EFFECTIVE mass, not the label count, and the difference is
  # load-bearing in two ways `has_sf_wdl.sum()` cannot express:
  #   * `sf_effective = sf_available * keep`, so the `1 - keep` shortfall the
  #     sf_search_dampen_* knobs remove also lands on the one-hot (F10). Both
  #     knobs are 0.0 in production today, which is exactly the kind of "agrees
  #     by coincidence" that stops being true after one live edit.
  #   * when the `sf_wdl` / `search_wdl` COLUMN is absent the component is the
  #     one-hot for every row regardless of the mask, so the count must be 0
  #     rather than the mask's sum.
    zero_rows = torch.zeros((), device=search_available.device)
    sf_wdl_effective_rows = (
        sf_effective.sum() if sf_wdl_probs is not None else zero_rows
    )
    search_wdl_effective_rows = (
        search_available.sum() if search_wdl_probs is not None else zero_rows
    )
    if terminal_taper is None:
  # Bit-identical to the pre-feature blend: same expressions, same order.
        target += sf_wdl_frac_f * sf_component
        target += search_wdl_frac_f * search_component
        terminal_outcome_weight_sum = torch.zeros((), device=search_available.device)
        terminal_outcome_rows = torch.zeros((), device=search_available.device)
    else:
  # Both transfers ride the SAME taper and are taken off the REALIZED fracs at
  # this step (the SF one is PID-recomputed every iteration), never off a
  # hard-coded share.
        sf_transfer_frac = min(1.0, max(0.0, float(wdl_terminal_outcome_sf_frac)))
        sf_outcome_w = (sf_wdl_frac_f * sf_transfer_frac) * terminal_taper
        search_outcome_w = search_wdl_frac_f * terminal_taper
        terminal_outcome_w = sf_outcome_w + search_outcome_w
        target += (sf_wdl_frac_f - sf_outcome_w) * sf_component
        target += (search_wdl_frac_f - search_outcome_w) * search_component
        target += terminal_outcome_w * game_oh
  # Proof-of-effect columns. `wdl_terminal_outcome_frac` is the mean weight
  # this actually moved onto the outcome across ALL batch rows — it is exactly
  # 0.0 while the knob is off, so a non-zero value is the observation that the
  # config reached the trained target. Read it with its row count: the frac
  # alone cannot tell "knob off" from "no near-terminal rows".
        terminal_outcome_flat = terminal_outcome_w.squeeze(1)
        terminal_outcome_weight_sum = terminal_outcome_flat.sum()
        terminal_outcome_rows = (terminal_outcome_flat > 0.0).to(torch.float32).sum()
    blended_wdl_ce = soft_cross_entropy(outputs["wdl"], target.detach())

    has_moves_left = _get_mask(batch, "has_moves_left")
    ml_pred = outputs.get("moves_left")
    moves_left_t = batch.get("moves_left")
    ml_loss = (
        F.smooth_l1_loss(ml_pred.squeeze(-1), moves_left_t, reduction="none")
        if ml_pred is not None and moves_left_t is not None else zero_loss
    )

    has_cat = _get_mask(batch, "has_categorical")
    cat_logits = outputs.get("categorical")
    categorical_t = batch.get("categorical_t")
    cat_ce = (
        soft_cross_entropy(cat_logits, categorical_t)
        if cat_logits is not None and categorical_t is not None else zero_loss
    )

    has_vol = _get_mask(batch, "has_volatility")
    vol_pred = outputs.get("volatility")
    volatility_t = batch.get("volatility_t")
    vol_loss = (
        _huber_per_sample(vol_pred, volatility_t)
        if vol_pred is not None and volatility_t is not None else zero_loss
    )

    has_sf_vol = _get_mask(batch, "has_sf_volatility")
    sf_vol_pred = outputs.get("sf_volatility")
    sf_volatility_t = batch.get("sf_volatility_t")
    sf_vol_loss = (
        _huber_per_sample(sf_vol_pred, sf_volatility_t)
        if sf_vol_pred is not None and sf_volatility_t is not None else zero_loss
    )

  # Loss weights — float() casts defend against numpy scalars from Ray Tune config mutation
    w_sf_volatility = float(w_sf_volatility) if w_sf_volatility is not None else float(w_volatility)
  # Named for what it masks (the aux `sf_eval` head), not for the knobs that
  # shape it: `sf_wdl_conf_power` / `sf_wdl_draw_scale` read as if they scale
  # the SF component of the VALUE blend and they do not. See
  # `_compute_sf_wdl_mask`'s docstring; the single consumer is `m_sf_eval`.
    m_sf_eval_mask = _compute_sf_wdl_mask(
        net_mask=net_mask, has_sf_wdl=has_sf_wdl, sf_wdl_probs=sf_wdl_probs,
        wdl_target=batch["wdl_t"],
        conf_power=max(0.0, float(sf_wdl_conf_power)),
        draw_scale=max(0.0, float(sf_wdl_draw_scale)),
    )

  # Precompute the per-sample base mask for each head so the downstream
    # split reductions don't recompute `net_mask * has_X` once per bucket.
    pol_base = net_mask * has_policy
    soft_base = net_mask * has_soft
    future_base = net_mask * has_future
    sf_move_base = net_mask * has_sf_policy
    categorical_base = net_mask * has_cat
    volatility_base = net_mask * has_vol
    sf_volatility_base = net_mask * has_sf_vol
    moves_left_base = net_mask * has_moves_left
    m_policy = masked_mean(pol_ce, pol_base)
    m_soft = masked_mean(soft_ce, soft_base)
    m_future = masked_mean(future_ce, future_base)
  # A row only counts as eligible if the TARGET is actually in the batch, not
  # merely because the shard set the `has_` flag: with the target absent the
  # loss tensor is `zero_loss` and the term trains nothing, so counting those
  # rows would make `has_sf_p0_frac` report a live teacher over a dead one —
  # the exact false negative these columns exist to rule out. `masked_mean` is
  # unaffected either way (its numerator is all zeros in that case).
    no_rows = torch.zeros_like(net_mask)
    sf_p0_base = net_mask * has_sf_p0 if sf_p0_target is not None else no_rows
    sf_p0_regret_base = (
        net_mask * has_sf_p0_regret if sf_p0_regret_t is not None else no_rows
    )
    m_sf_own = masked_mean(sf_p0_ce, sf_p0_base)
    m_sf_own_regret = masked_mean(sf_own_regret, sf_p0_regret_base)
    sf_own_ce_sum, sf_own_rows = masked_sum_and_count(sf_p0_ce, sf_p0_base)
    sf_own_regret_sum, sf_own_regret_rows = masked_sum_and_count(
        sf_own_regret, sf_p0_regret_base,
    )
  # ⚑ THE OBSERVATION THAT PROVES THE GATE REACHED PRODUCTION. Without it an
  # operator cannot distinguish "gate configured" from "gate applied": at the
  # identity defaults it reads exactly 0.0, and any non-zero value is the share of
  # eligible rows whose sf_own_regret term was scaled.
  # ⚑ EMITTED AS A COUNT, NOT A PER-BATCH RATE, and registered in
  # `_RATIO_METRIC_FIELDS` against `sf_own_regret_rows`. A `masked_mean` here would
  # be aggregated as an unweighted mean of per-batch rates, which that table's own
  # comment says is the wrong estimator for exactly the sf_p0 terms — their
  # eligible count swings batch to batch. Numerator and denominator share the SAME
  # mask, so the pair cannot disagree about how many rows were eligible.
    sf_own_regret_gated_rows = (
        sf_own_regret_gated.to(torch.float32) * sf_p0_regret_base.to(torch.float32)
    ).sum()
    m_sf_policy_floor = masked_mean(sf_floor, sf_p0_regret_base)
  # BINDING RATE, the observation that separates "the weight reached the loss"
  # from "the term did something". A floor that never binds is a weight
  # multiplied into a structural zero -- accepted, threaded, and silently
  # inert, which is this repo's signature defect. Both columns divide by
  # `sf_own_regret_rows`, the SAME count, because they are masked by the SAME
  # tensor; a second count derived here could disagree with it.
    sf_policy_floor_sum, _ = masked_sum_and_count(sf_floor, sf_p0_regret_base)
    sf_policy_floor_binds_sum, _ = masked_sum_and_count(sf_floor_binds, sf_p0_regret_base)
  # FEASIBILITY-CAP diagnostics, over the SAME `sf_own_regret_rows` denominator
  # as the two columns above -- one population for the whole term.
    sf_policy_floor_raw_members_sum, _ = masked_sum_and_count(
        floor_out.member_count_raw, sf_p0_regret_base,
    )
    sf_policy_floor_requested_mass_sum, _ = masked_sum_and_count(
        floor_out.requested_mass, sf_p0_regret_base,
    )
    sf_policy_floor_truncated_sum, _ = masked_sum_and_count(
        floor_out.truncated, sf_p0_regret_base,
    )
    sf_policy_floor_applied_members_sum, _ = masked_sum_and_count(
        floor_out.member_count_applied, sf_p0_regret_base,
    )
    sf_policy_floor_applied_mass_sum, _ = masked_sum_and_count(
        floor_out.applied_mass, sf_p0_regret_base,
    )
  # SF-shape term + its permanent entropy instrument. All nine share
  # `sf_p0_regret_base`, so they divide by the SAME `sf_own_regret_rows` count as
  # the floor's columns and none of them can disagree with the others about the
  # population. Emitted as sums for the same reason the rest of this family is:
  # the eligible count swings batch to batch, so a mean of per-batch means is the
  # wrong estimator.
  #
  # ⚑⚑ THE ENTROPY PAIR IS THE POINT OF THE CHANGE, IT IS LIVE AT
  # `w_sf_shape: 0.0`, AND IT IS SAME-SUPPORT ON PURPOSE.
  # `sf_shape_entropy_gap_sum` is `H(q_S) - H(p_S)`, both conditioned on the SAME
  # surfaced set, so a positive value means WE are the sharp one.
  #
  # ⚑ DO NOT COMPARE ACROSS SUPPORTS. The reading that motivated this change --
  # our 0.6784 nats over ~27 legal moves against "SF's 1.0572" -- is INVALID as
  # stated and is not repeated as a target anywhere here: those are two
  # distributions over different supports, and the ~6.6% of the full-width SF
  # target's mass that sits outside its top 6 was allocated by OUR fabricated
  # `default_regret`, not by Stockfish. The genuine teacher object is the
  # CONDITIONAL top-K distribution and nothing else; the full-distribution
  # numbers are a historical diagnostic.
  #
  # `sf_shape_sharper_sum` is the gap as a ROW RATE, which the mean cannot give:
  # a big gap on few rows and a small gap on all of them read alike.
    sf_shape_ce_sum, _ = masked_sum_and_count(shape_out.kl, sf_p0_regret_base)
    sf_shape_active_sum, _ = masked_sum_and_count(sf_shape_active, sf_p0_regret_base)
    sf_shape_h_sf_given_s_sum, _ = masked_sum_and_count(
        shape_out.h_sf_given_s, sf_p0_regret_base,
    )
    sf_shape_h_ours_given_s_sum, _ = masked_sum_and_count(
        shape_out.h_ours_given_s, sf_p0_regret_base,
    )
    sf_shape_entropy_gap_sum, _ = masked_sum_and_count(
        shape_out.h_sf_given_s - shape_out.h_ours_given_s, sf_p0_regret_base,
    )
    sf_shape_sharper_sum, _ = masked_sum_and_count(
        (shape_out.h_ours_given_s < shape_out.h_sf_given_s).to(torch.float32),
        sf_p0_regret_base,
    )
    sf_shape_regret_cp_sum, _ = masked_sum_and_count(
        shape_out.regret_cp_given_s, sf_p0_regret_base,
    )
  # The FULL legal-support entropy of `policy_own`, kept as the continuity column
  # with the ledger's historical 0.6784. ⚑ It is NOT comparable to
  # `sf_shape_h_sf_given_s` (support ~27 legal moves against ~5.6 surfaced);
  # it is published so a number can be traced to its support instead of guessed.
    sf_shape_h_ours_full_legal_sum, _ = masked_sum_and_count(
        shape_out.h_ours_full_legal, sf_p0_regret_base,
    )
  # Matched-support instrument for the MAIN policy target. Its own population --
  # `pol_base`, every row that HAS a policy target -- and therefore its own row
  # count, because it is not an SF quantity and borrowing `sf_own_regret_rows`
  # would silently restrict it to the ~21% of rows that carry SF regret.
    policy_support_h_ours_sum, policy_target_rows = masked_sum_and_count(
        pol_support.h_ours, pol_base,
    )
    policy_support_h_target_sum, _ = masked_sum_and_count(pol_support.h_target, pol_base)
    policy_support_gap_sum, _ = masked_sum_and_count(
        pol_support.h_target - pol_support.h_ours, pol_base,
    )
    policy_support_size_sum, _ = masked_sum_and_count(pol_support.support_size, pol_base)
    policy_tail_mass_sum, _ = masked_sum_and_count(pol_support.tail_mass_ours, pol_base)
    sf_shape_surfaced_sum, _ = masked_sum_and_count(
        shape_out.surfaced_count, sf_p0_regret_base,
    )
    sf_shape_surfaced_mass_sum, _ = masked_sum_and_count(
        shape_out.surfaced_mass, sf_p0_regret_base,
    )
    sf_shape_p_sf_best_sum, _ = masked_sum_and_count(
        shape_out.p_sf_best, sf_p0_regret_base,
    )
    m_sf_shape = masked_mean(shape_out.kl, sf_p0_regret_base)
    net_rows = net_mask.to(torch.float32).sum()
    sf_no_multipv_rows, sf_multipv_checked_rows = sf_multipv_presence_counts(
        batch, has_sf_wdl=has_sf_wdl,
    )
    sf_wdl_degenerate_rows, sf_wdl_orphaned_rows, sf_wdl_rows = sf_wdl_health_counts(
        batch, has_sf_wdl=has_sf_wdl,
    )
    sf_eval_pv_orphan_rows, sf_eval_pv_checked_rows = sf_eval_pv_orphan_counts(batch)
    batch_rows = net_mask.new_full((), float(net_mask.shape[0]))
    m_wdl_onehot = masked_mean(wdl_onehot_ce, net_mask)
    m_blended_wdl = masked_mean(blended_wdl_ce, net_mask)
    m_sf_move = masked_mean(sf_move_ce, sf_move_base)
    m_sf_eval = masked_mean(sf_eval_ce, m_sf_eval_mask)
    m_cat = masked_mean(cat_ce, categorical_base)
    m_vol = masked_mean(vol_loss, volatility_base)
    m_sf_vol = masked_mean(sf_vol_loss, sf_volatility_base)
    m_ml = masked_mean(ml_loss, moves_left_base)

  # Gated on `has_is_selfplay` so legacy shards without the tag are excluded
  # from the split (they won't contribute to either selfplay_ or curriculum_ keys).
    has_is_sp = _get_mask(batch, "has_is_selfplay").to(torch.float32)
    is_sp_bool = _get_mask(batch, "is_selfplay", default=0.0).to(torch.float32)
    split_masks = _phase_split_masks(
        has_is_selfplay=has_is_sp, is_selfplay=is_sp_bool,
        piece_counts=piece_counts_from_input(batch["x"]),
    )
  # Split reductions use the TRAINED per-sample value loss (blended soft CE),
  # not the one-hot diagnostic — before 2026-07-26 these were the diagnostic,
  # which made `wdl_loss_selfplay` / `_open` / ... track a term no gradient
  # ever came from.
    split_losses: dict[str, torch.Tensor] = {}
    split_weight_masks: dict[str, torch.Tensor] = {}
    for suffix, m in split_masks:
        policy_bucket_mask = pol_base * m
        split_losses[f"policy_loss_{suffix}"] = masked_mean(pol_ce, policy_bucket_mask)
        wdl_bucket_mask = net_mask * m
        split_losses[f"wdl_loss_{suffix}"] = masked_mean(blended_wdl_ce, wdl_bucket_mask)
        split_weight_masks[f"policy_loss_{suffix}"] = policy_bucket_mask
        split_weight_masks[f"wdl_loss_{suffix}"] = wdl_bucket_mask
  # The DENOMINATORS of the two lines above, as raw row counts. Without them a
  # bucket cannot be told apart from a good one: `masked_mean` clamps its
  # denominator to 1.0, so a bucket holding zero rows reports 0.0, which reads
  # as the best possible value. Summed (not averaged) across the iteration's
  # microbatches — see `_RAW_COUNT_METRIC_FIELDS` in train/trainer.py.
  #
  # BOTH heads, because their denominators are DIFFERENT: the policy mask
  # carries `has_policy` on top of `net_mask`, so a phase can be well populated
  # for the value head and empty for the policy head in the same batch. One
  # head's count cannot stand in for the other's.
        if suffix in _PHASE_BUCKET_SUFFIXES:
            split_losses[f"wdl_rows_{suffix}"] = wdl_bucket_mask.to(torch.float32).sum()
            split_losses[f"policy_rows_{suffix}"] = policy_bucket_mask.to(torch.float32).sum()

  # ⚑ EVERY TERM IS ADDED UNDER AN `if`, NOT MULTIPLIED BY ITS WEIGHT, and the
  # TABLE is the point: a head added to `total` inherits the guard instead of
  # depending on someone remembering to repeat it. Two things the `if` buys:
  #
  # (1) `0.0 * x` IS NOT ZERO FOR EVERY `x`. `0.0 * float("nan")` is NaN, so ONE
  #     NaN component poisons `total` -- and every gradient built from it --
  #     through a weight that is supposed to mean "off". `masked_mean` is NOT a
  #     defence: its DENOMINATOR is `clamp_min(1.0)`, but its NUMERATOR is
  #     `(x * mask).sum()`, and `0.0 * nan` is NaN there too, so an EMPTY mask
  #     over a NaN term returns NaN rather than 0.0. That regime is the expected
  #     one and not an edge case: the AZ-purity arm zeroes several loss weights
  #     and gen-0 shards carry NO SF fields at all, so a zero-weighted SF term
  #     over an empty denominator is what a normal iteration of that arm looks
  #     like. Same shape as "a clamp is not a validator", where min/max quietly
  #     propagate NaN while the guard's own counter reads healthy.
  # (2) INERTNESS IS EXACT: at weight 0.0 the objective is the one that existed
  #     before the term, bit-identically rather than identical-up-to-a-`+ 0.0`.
  #
  # ⚑ THE PREDICATE IS EXACT EQUALITY WITH ZERO, NOT `<= 0.0`. A NEGATIVE weight
  # is still a term in `total` (sign-flipped, but present), which is exactly what
  # `eval_ruler.active_loss_terms` reports for a plain multiplier. The two rules
  # have to agree or the holdout ruler hashes a term set the objective does not
  # have. `-0.0` compares equal to `0.0` and is therefore off in both.
  #
  # The DIAGNOSTIC columns are computed either way, above and in the returned
  # dict below, so switching a weight on is never the first time anyone sees
  # what its term does.
    weighted_terms: tuple[
        tuple[str, float, torch.Tensor, torch.Tensor], ...
    ] = (
        ("policy", float(w_policy), m_policy, pol_base),
        ("soft_policy", float(w_soft), m_soft, soft_base),
        ("future_policy", float(w_future), m_future, future_base),
        ("sf_own", float(w_sf_own), m_sf_own, sf_p0_base),
        ("sf_own_regret", float(w_sf_own_regret), m_sf_own_regret, sf_p0_regret_base),
        ("wdl", float(w_wdl), m_blended_wdl, net_mask),
        ("sf_move", float(w_sf_move), m_sf_move, sf_move_base),
        ("sf_eval", float(w_sf_eval), m_sf_eval, m_sf_eval_mask),
        ("categorical", float(w_categorical), m_cat, categorical_base),
        ("volatility", float(w_volatility), m_vol, volatility_base),
        ("sf_volatility", float(w_sf_volatility), m_sf_vol, sf_volatility_base),
        ("moves_left", float(w_moves_left), m_ml, moves_left_base),
        ("sf_policy_floor", float(floor_params.w), m_sf_policy_floor, sf_p0_regret_base),
        ("sf_shape", float(shape_params.w), m_sf_shape, sf_p0_regret_base),
    )
    if tuple(name for name, _, _, _ in weighted_terms) != EXACT_OBJECTIVE_NAMES:
        raise AssertionError("exact objective names drifted from weighted terms")
  # Folded LEFT in declaration order, so with every weight non-zero this performs
  # the same sequence of float32 additions as the flat `a + b + c + ...`
  # expression it replaces: `total` is bit-identical there, not merely close.
    total: torch.Tensor | None = None
    for _name, w, term, _mask in weighted_terms:
        if w == 0.0:
            continue
        total = w * term if total is None else total + w * term
    if total is None:
  # Every weight is zero, so the objective is empty and `total` is a constant.
  # It carries no gradient path -- there is no term left to carry one -- which
  # is the honest reading of "nothing is being trained" and is not something a
  # config asking for no objective at all should be able to hide.
        total = torch.zeros_like(m_policy)

  # ⚑ WHAT THE `if` ABOVE COSTS: a NaN in a zero-weighted term no longer reaches
  # `total`, so it no longer trips `_run_optimizer_step`'s non-finite-GRADIENT
  # guard either. Nothing is left to announce it -- the NaN survives as a silent
  # TB column that reads like any other diagnostic. This counts exactly the terms
  # the guard disarmed AND found non-finite, so a zero weight cannot quietly
  # become a place NaNs go to die. `Trainer.train_steps` is the consumer and logs
  # it once per iteration off its OWN accumulated value.
  #
  # Counted on-device (no `.item()`, no sync): `w == 0.0` is a host-side float
  # comparison, so only the isfinite reduction touches the GPU, and only for the
  # terms that are actually off.
    disarmed_nonfinite_terms = torch.zeros((), device=m_policy.device, dtype=torch.float32)
    for _name, w, term, _mask in weighted_terms:
        if w == 0.0:
            disarmed_nonfinite_terms = disarmed_nonfinite_terms + (
                ~torch.isfinite(term.detach())
            ).to(torch.float32)

    frac_is_selfplay = masked_mean(is_sp_bool, has_is_sp)
    frac_tagged = masked_mean(has_is_sp, net_mask)

    exact_masked_sums: dict[str, torch.Tensor] = {}
    if bool(report_exact_masked_sums):
        # In an exact epoch, a physical row must retain the same optimizer
        # coefficient when the final batches are ragged. Each objective is a
        # masked mean over a different population, so scaling their already-
        # combined mean by physical batch rows is wrong: a head with one
        # eligible row would change weight merely because unrelated rows share
        # its batch. Recover every term's masked numerator, restore its
        # corpus-wide masked-mean normalization below, and normalize the
        # combined objective by physical rows before Trainer applies n/B.
        if exact_corpus_rows is None or int(exact_corpus_rows) <= 0:
            raise ValueError(
                "exact loss normalization requires a positive exact_corpus_rows",
            )
        if exact_objective_mask_weights is None:
            raise ValueError(
                "exact loss normalization requires preflighted objective-mask "
                "weights",
            )
        supplied_names = tuple(exact_objective_mask_weights)
        if supplied_names != EXACT_OBJECTIVE_NAMES:
            raise ValueError(
                "exact objective-mask weights have keys/order "
                f"{supplied_names}, expected {EXACT_OBJECTIVE_NAMES}",
            )
        global_weights_list = [
            float(exact_objective_mask_weights[name])
            for name in EXACT_OBJECTIVE_NAMES
        ]
        if any(
            not math.isfinite(value) or value < 0.0
            for value in global_weights_list
        ):
            raise ValueError(
                "exact objective-mask weights must be finite and non-negative",
            )
        objective_weights = torch.stack([
            mask.to(torch.float32) for _, _, _, mask in weighted_terms
        ]).sum(dim=1)
        objective_numerators = torch.stack([
            term.to(torch.float32) for _, _, term, _ in weighted_terms
        ]) * objective_weights.clamp_min(1.0)
        # Preserve the configured meaning of every masked head: globally its
        # term is still w * (sum eligible loss / eligible weight).  N/K turns
        # that global masked mean into per-corpus-row units; division by this
        # physical batch's n and Trainer's existing n/B ragged factor then give
        # each eligible contribution coefficient N/(K*B), independent of which
        # other rows happened to share its batch.  K=0 remains an exact zero
        # term because no batch can then carry a non-zero numerator.
        # Form the ratio in host float64 before creating the fp32 device
        # constants. Production corpora exceed 2**24 rows, so casting N and K
        # separately to fp32 before division needlessly loses integer bits.
        global_normalizers = objective_weights.new_tensor(
            [
                float(int(exact_corpus_rows)) / weight if weight > 0.0 else 0.0
                for weight in global_weights_list
            ],
        )
        exact_total: torch.Tensor | None = None
        for idx, (_name, w, _, _) in enumerate(weighted_terms):
            if w == 0.0:
                continue
            contribution = (
                w * objective_numerators[idx] * global_normalizers[idx]
            )
            exact_total = (
                contribution
                if exact_total is None else exact_total + contribution
            )
        if exact_total is None:
            exact_total = torch.zeros_like(m_policy)
        total = exact_total / float(max(1, int(net_mask.shape[0])))

        # Exact epochs pool every masked diagnostic by the observations behind
        # it.  Keep this channel internal: legacy training/eval deliberately
        # retains its restart-gated historical estimator, while exact mode has
        # no historical rows whose meaning could be silently changed.
        soft_pre_mask = net_mask * has_soft_before_keep
        soft_kept_weight_mask = (
            soft_pre_mask
            if float(soft_policy_min_tv) > 0.0 and soft_target is not None
            else torch.ones_like(net_mask)
        )
        specs: list[tuple[str, torch.Tensor, torch.Tensor]] = [
            ("policy_loss", m_policy, pol_base),
            ("soft_policy_loss", m_soft, soft_base),
            ("future_policy_loss", m_future, future_base),
            ("wdl_loss", m_blended_wdl, net_mask),
            ("blended_wdl_loss", m_blended_wdl, net_mask),
            ("wdl_onehot_loss", m_wdl_onehot, net_mask),
            ("sf_move_loss", m_sf_move, sf_move_base),
            ("sf_eval_loss", m_sf_eval, m_sf_eval_mask),
            ("categorical_loss", m_cat, categorical_base),
            ("volatility_loss", m_vol, volatility_base),
            ("sf_volatility_loss", m_sf_vol, sf_volatility_base),
            ("moves_left_loss", m_ml, moves_left_base),
            ("frac_is_selfplay", frac_is_selfplay, has_is_sp),
            ("frac_tagged", frac_tagged, net_mask),
            ("soft_mask_kept_frac", soft_mask_kept_frac, soft_kept_weight_mask),
            ("sf_search_agree_frac", sf_search_agree_frac, joint),
            (
                "sf_search_disagree_sf_low_frac",
                sf_search_disagree_sf_low_frac,
                joint,
            ),
            (
                "sf_search_disagree_sf_high_frac",
                sf_search_disagree_sf_high_frac,
                joint,
            ),
            *(
                (field, split_losses[field], mask)
                for field, mask in split_weight_masks.items()
            ),
        ]
        # One batched reduction, not one GPU launch per diagnostic.  Every mask
        # is row-shaped; stacking them makes the exact-only instrumentation a
        # fixed three vector kernels instead of ~30 tiny reductions.
        weights = torch.stack([
            mask.to(torch.float32) for _, _, mask in specs
        ]).sum(dim=1)
        # ``masked_mean`` clamps its divisor to 1.0.  Binary masks normally
        # have integer weight, but the confidence-weighted sf_eval mask can sum
        # to (0, 1); undo the exact divisor it used, not the unclamped weight.
        weighted_means = torch.stack([
            mean.to(torch.float32) for _, mean, _ in specs
        ]) * weights.clamp_min(1.0)
        for idx, (field, _, _) in enumerate(specs):
            exact_masked_sums[f"_exact_{field}_sum"] = weighted_means[idx]
            exact_masked_sums[f"_exact_{field}_weight"] = weights[idx]

  # Reported value-loss names (docs/rl_loop_audit.md I7):
  #   wdl_ce / blended_wdl_ce -> the SAME tensor, the loss the optimizer sees.
  #     `wdl_ce` is the name people reach for (it becomes the `wdl_loss`
  #     column), `blended_wdl_ce` is kept because existing readers use it.
  #   wdl_onehot_ce -> the hard one-hot diagnostic, never in `total`.
    result = {
        "total": total,
  # How many loss components were BOTH weighted off and non-finite. Zero on
  # every healthy batch; > 0 means a NaN exists that the zero-weight guard is
  # keeping out of `total` and that therefore no gradient check can see.
        "disarmed_nonfinite_terms": disarmed_nonfinite_terms,
  # How many value-label ROWS were non-finite with a zero blend mask, summed
  # over `sf_wdl` and `search_wdl`. These are the rows `_finite_blend_component`
  # silently replaced with the game outcome -- correct, and still a shard
  # defect. Counts +-inf and NaN identically, which is the whole reason the
  # finiteness test is taken on the RAW tensor.
        "blend_unclaimed_nonfinite_rows": blend_unclaimed_nonfinite_rows,
        "policy_ce": m_policy,
        "wdl_ce": m_blended_wdl,
        "blended_wdl_ce": m_blended_wdl,
        "wdl_onehot_ce": m_wdl_onehot,
        "soft_policy_ce": m_soft,
        "soft_mask_kept_frac": soft_mask_kept_frac,
        "future_policy_ce": m_future,
        "sf_own_ce": m_sf_own,
        "sf_own_regret": m_sf_own_regret,
        "sf_own_regret_gated_rows": sf_own_regret_gated_rows,
  # sf_p0 policy-teacher observability. Emitted as SUMS + eligible-row COUNTS
  # rather than as per-batch means because the trainer accumulates these over
  # every microbatch of the iteration and divides once: eligibility is a
  # property of consecutive full-sim plies, so the eligible count varies per
  # batch and a mean of per-batch means would be the wrong estimator. The
  # counts are also the outage detector — `sf_own_rows` goes to exactly 0
  # when recording stops, whatever `w_sf_own` happens to be, which a masked
  # mean alone cannot distinguish from eligible rows sitting at zero loss.
  # Mapped to the m_sf_own / has_sf_p0_frac columns in train/trainer.py.
        "sf_own_ce_sum": sf_own_ce_sum,
        "sf_own_rows": sf_own_rows,
  # SF-approved-move floor. `sf_policy_floor` is the per-batch masked mean (the
  # column that is comparable to `sf_own_regret`); the two SUMS below feed the
  # row-weighted `m_sf_policy_floor` / `sf_policy_floor_binds_frac` columns, over
  # the SAME `sf_own_regret_rows` denominator. Read the BINDING RATE first: the
  # mean can fall either because the net learned to clear the floor or because
  # the term stopped selecting anything, and only the rate tells those apart.
        "sf_policy_floor": m_sf_policy_floor,
        "sf_policy_floor_sum": sf_policy_floor_sum,
        "sf_policy_floor_binds_sum": sf_policy_floor_binds_sum,
  # ⚑ THE CAP MUST NOT BE SILENT. `requested_mass` is what the uncapped rule
  # asked for and `applied_mass` is what a distribution can actually carry;
  # `truncated_frac` is the share of eligible rows where they differ. Read the
  # raw/applied member pair to answer whether F's flattening came from the
  # hidden `|F| * tau` strength rather than from the configured tau.
        "sf_policy_floor_raw_members_sum": sf_policy_floor_raw_members_sum,
        "sf_policy_floor_requested_mass_sum": sf_policy_floor_requested_mass_sum,
        "sf_policy_floor_truncated_sum": sf_policy_floor_truncated_sum,
        "sf_policy_floor_applied_members_sum": sf_policy_floor_applied_members_sum,
        "sf_policy_floor_applied_mass_sum": sf_policy_floor_applied_mass_sum,
        "sf_own_regret_sum": sf_own_regret_sum,
        "sf_own_regret_rows": sf_own_regret_rows,
  # SF-shape conditional KL + the entropy instrument, over the SAME
  # `sf_own_regret_rows` denominator as the floor's columns. `sf_shape` is the
  # per-batch masked mean (comparable to `sf_policy_floor`); everything ending
  # in `_sum` feeds a row-weighted column in train/trainer.py. Read
  # `sf_shape_active_sum` first -- below 2 surfaced moves the term is a
  # structural zero at every weight -- and read `sf_shape_surfaced_sum` before
  # any of it, because it is the health of the mask the whole family stands on.
        "sf_shape": m_sf_shape,
        "sf_shape_ce_sum": sf_shape_ce_sum,
        "sf_shape_active_sum": sf_shape_active_sum,
        "sf_shape_h_sf_given_s_sum": sf_shape_h_sf_given_s_sum,
        "sf_shape_h_ours_given_s_sum": sf_shape_h_ours_given_s_sum,
        "sf_shape_entropy_gap_sum": sf_shape_entropy_gap_sum,
        "sf_shape_sharper_sum": sf_shape_sharper_sum,
        "sf_shape_regret_cp_sum": sf_shape_regret_cp_sum,
        "sf_shape_h_ours_full_legal_sum": sf_shape_h_ours_full_legal_sum,
        "sf_shape_surfaced_sum": sf_shape_surfaced_sum,
        "sf_shape_surfaced_mass_sum": sf_shape_surfaced_mass_sum,
        "sf_shape_p_sf_best_sum": sf_shape_p_sf_best_sum,
  # MATCHED-SUPPORT instrument for the ordinary policy target. Separate family,
  # separate population, separate row count -- it answers "is the search target
  # itself a sharpening teacher", which is a different question from anything the
  # SF columns above measure. `policy_tail_mass_sum` is the mass the restriction
  # drops, published rather than allowed to leak into the entropy difference.
        "policy_support_h_ours_sum": policy_support_h_ours_sum,
        "policy_support_h_target_sum": policy_support_h_target_sum,
        "policy_support_gap_sum": policy_support_gap_sum,
        "policy_support_size_sum": policy_support_size_sum,
        "policy_tail_mass_sum": policy_tail_mass_sum,
        "policy_target_rows": policy_target_rows,
        "net_rows": net_rows,
  # SF-label contamination detector, ALWAYS ON. Sums + row counts for the same
  # reason as the sf_p0 pair above: the trainer accumulates them over every
  # microbatch and divides once, so the ratio is row-weighted rather than a
  # mean of per-batch means. See `sf_multipv_presence_counts` for the
  # denominator and the zero floor; mapped to `sf_labelled_no_multipv_frac` /
  # `sf_multipv_checked_frac` in train/trainer.py.
        "sf_no_multipv_rows": sf_no_multipv_rows,
        "sf_multipv_checked_rows": sf_multipv_checked_rows,
  # SF-label contamination detector, VALUE half — the same channel and the same
  # sums-then-divide-once treatment. `sf_wdl_rows` is their denominator and is
  # deliberately NOT `sf_multipv_checked_rows`, so a missing POLICY field
  # cannot blind the value-side rates. See `sf_wdl_health_counts` and
  # `sf_eval_pv_orphan_counts`; mapped to `sf_wdl_degenerate_frac` /
  # `sf_wdl_orphaned_frac` / `sf_eval_pv_orphan_frac` / `sf_eval_pv_checked_frac`
  # in train/trainer.py.
        "sf_wdl_degenerate_rows": sf_wdl_degenerate_rows,
        "sf_wdl_orphaned_rows": sf_wdl_orphaned_rows,
        "sf_wdl_rows": sf_wdl_rows,
  # Value-blend fallback denominators — see the blend site. `sf_wdl_rows` above
  # is the LABEL count and is NOT interchangeable with these: it ignores `keep`
  # and reads non-zero on a batch whose `sf_wdl` column is missing entirely.
  # Mapped to `sf_wdl_effective_frac` / `search_wdl_effective_frac`.
        "sf_wdl_effective_rows": sf_wdl_effective_rows,
        "search_wdl_effective_rows": search_wdl_effective_rows,
        "sf_eval_pv_orphan_rows": sf_eval_pv_orphan_rows,
        "sf_eval_pv_checked_rows": sf_eval_pv_checked_rows,
        "batch_rows": batch_rows,
        "sf_move_ce": m_sf_move,
        "sf_eval_ce": m_sf_eval,
        "categorical_ce": m_cat,
        "volatility": m_vol,
        "sf_volatility": m_sf_vol,
        "moves_left": m_ml,
        **split_losses,
        "frac_is_selfplay": frac_is_selfplay,
        "frac_tagged": frac_tagged,
  # Terminal-proximal outcome transfer (see the blend site above). Emitted as
  # a SUM + a row COUNT, not as per-batch means, for the same reason as the
  # sf_p0 pair: the trainer accumulates them over every microbatch and divides
  # once, so the published fraction is row-weighted. Both are exactly 0.0 while
  # `wdl_terminal_outcome_plies` is 0.
        "wdl_terminal_outcome_weight_sum": terminal_outcome_weight_sum,
        "wdl_terminal_outcome_rows": terminal_outcome_rows,
        "sf_search_agree_frac": sf_search_agree_frac,
        "sf_search_disagree_sf_low_frac": sf_search_disagree_sf_low_frac,
        "sf_search_disagree_sf_high_frac": sf_search_disagree_sf_high_frac,
        **exact_masked_sums,
    }
    return result


def wdl_calibration_stats(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    n_bins: int = 10,
) -> dict[str, torch.Tensor]:
    """Per-sample Brier + per-bin calibration aggregates (all on-device, no syncs).

    Returns accumulator-friendly sums: callers sum these across eval batches and
    derive global Brier/ECE at the end. Doing per-batch ECE and then averaging
    weights small batches the same as large ones, which is wrong.

    Keys in the returned dict:
      - ``brier_sum``: scalar, sum of per-sample Brier over the batch
      - ``n``: scalar, number of samples in the batch
      - ``bin_conf_sum``: (n_bins,) sum of max-prob confidence per bin
      - ``bin_correct_sum``: (n_bins,) sum of correctness (0/1) per bin
      - ``bin_n``: (n_bins,) count per bin
    """
    probs = F.softmax(logits, dim=-1)
    one_hot = F.one_hot(target.to(torch.int64), num_classes=3).to(probs.dtype)
    brier_per_sample = ((probs - one_hot) ** 2).sum(dim=-1)

    conf, pred = probs.max(dim=-1)
    correct = (pred == target).to(probs.dtype)
  # bucketize boundaries: n_bins-1 inner edges so bins span [0,1/n), ... ,[(n-1)/n, 1].
  # Clamp to [0, n_bins-1] so the topmost conf==1.0 doesn't land in an n_bins slot.
    inner_edges = torch.linspace(
        1.0 / n_bins, 1.0 - 1.0 / n_bins, n_bins - 1, device=logits.device, dtype=probs.dtype
    )
    bin_idx = torch.bucketize(conf.detach(), inner_edges).clamp_max(n_bins - 1)
    bin_n = torch.zeros(n_bins, device=logits.device, dtype=probs.dtype).scatter_add_(
        0, bin_idx, torch.ones_like(conf)
    )
    bin_conf_sum = torch.zeros(n_bins, device=logits.device, dtype=probs.dtype).scatter_add_(
        0, bin_idx, conf
    )
    bin_correct_sum = torch.zeros(n_bins, device=logits.device, dtype=probs.dtype).scatter_add_(
        0, bin_idx, correct
    )
    return {
        "brier_sum": brier_per_sample.sum(),
        "n": torch.tensor(float(target.numel()), device=logits.device, dtype=probs.dtype),
        "bin_conf_sum": bin_conf_sum,
        "bin_correct_sum": bin_correct_sum,
        "bin_n": bin_n,
    }


def wdl_brier_ece_from_stats(stats: dict[str, torch.Tensor]) -> tuple[float, float]:
    """Combine accumulated calibration stats into (mean_brier, global_ece).

    ECE = sum_b |correct_sum[b] - conf_sum[b]| / n_total — algebraically identical
    to the standard definition, since bin_acc[b]*bin_n[b] = correct_sum[b] and
    bin_conf[b]*bin_n[b] = conf_sum[b], and the per-bin weight is bin_n[b]/n_total.
    """
    n = float(stats["n"].item())
    if n <= 0:
        return 0.0, 0.0
    brier = float(stats["brier_sum"].item()) / n
    diff = (stats["bin_correct_sum"] - stats["bin_conf_sum"]).abs().sum()
    ece = float(diff.item()) / n
    return brier, ece
