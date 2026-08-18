from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

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
# DISTRIBUTION. `selfplay/finalize.py:924` writes `moves_left` as
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
# that field: `finalize.py:924` divides by an ABSOLUTE game total, while the
# Python fallback in `selfplay/network_turn.py:558` sets `ply_index` RELATIVE
# to the search root where the production C path (`mcts/_mcts_tree.c:4734`)
# sets it ABSOLUTE. That mismatch no longer touches this split.
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
    tripwire it replaces (``sf_rebuild_policy_frac`` minus
    ``sf_rebuild_wdl_frac``) is definitionally the same signal but is gated
    behind ``rebuild_sf_targets``, which defaults False and is in no config
    file — so it has read 0.0 through three separate desync episodes.
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
# sits ABOVE the search-inclusion guarantee rather than at it.
SF_POLICY_FLOOR_TAU_DEFAULT = 0.15


def search_inclusion_guarantee_tau(gumbel_topk: object) -> float:
    """The smallest tau that buys a root-candidate slot BY PRIOR RANK: ``1 / topk``.

    The production root sampler (`mcts/gumbel._select_top_m_with_gumbel`) keeps
    the top ``m`` of ``gumbel_scale * Gumbel + log(prior)``, where

        m = 1                                            if sims <= 1
        m = max(2, min(topk, n_legal, max(2, (sims+1)//2)))  otherwise

    ⚑⚑ "GUARANTEE" IS EXACT FOR THE DETERMINISTIC RANKING AND ONLY EMPIRICAL
    UNDER PRODUCTION NOISE. DO NOT QUOTE IT AS A PROBABILITY OF 1.
    If ``p_i >= tau`` then at most ``floor(1/tau) - 1`` other moves can exceed
    it (they are disjoint and sum to <= 1), so ``tau >= 1/topk`` puts move ``i``
    inside the top-``topk`` **by prior**. That argument is about the ranking of
    ``log(prior)``; the sampler ranks ``gumbel_scale * Gumbel + log(prior)``,
    and Gumbel noise has UNBOUNDED support, so no prior threshold can make
    inclusion certain.

    ⚑ AND THE KNOB THIS DOCSTRING USED TO CITE WAS THE WRONG ONE. It justified
    near-determinism with "``gumbel_c_scale = 0.1``, noise ~0.13 in log-prob
    units". ``gumbel_c_scale`` is the SEARCH's exploration constant; the root
    sampler above is scaled by ``gumbel_scale``, which production runs at
    **1.0**, decaying to ``gumbel_scale_after: 0.5`` from move 12. The real
    noise is ~10x the figure that was used to call it "very nearly
    deterministic".

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

    ⚑ QUOTE THE SCALE, NEVER "production". The two trees differ:
    ``configs/pbt2_small.yaml`` on this branch runs ``gumbel_scale: 0.75`` decaying
    to ``gumbel_scale_after: 0.0``, while the live branch runs ``1.0 -> 0.5``. The
    claim holds on both -- 17% and 27% exclusion respectively at the pre-decay
    scale -- but a number attached to the word "production" without naming the
    tree is how a measurement of one config gets quoted about another.

    So the move the threshold is named for is dropped up to **27%** of the time when
    the other legal moves share the remaining mass evenly. The earlier "MIN (not
    mean) P(searched) is exactly 1.0000 at prior >= 1/16, and 0.1042 at prior >=
    0.02" was measured on 3000 REAL production rows and is not contradicted: real
    priors are peaked, which is the left column. It is a property of the empirical
    prior distribution, NOT a bound the sampler provides -- so it holds until the
    policy flattens, and nothing warns when it stops holding.

    `tests/test_sf_policy_floor.py::test_inclusion_under_production_noise_is_not_a_guarantee`
    pins the measured numbers, because an argument this docstring got wrong
    once should not be the thing defending it.

    ⚑ WHEN THE SIM BUDGET CAN BITE, AND WHY IT DOES NOT HERE. ``1/topk`` buys a
    slot in the top-``topk``; what the search keeps is the top-``m``. The two
    differ only when ``m < topk`` AND ``n_legal > m``, and each half is worth
    stating because a review of this branch got both backwards:

    * ``n_legal <= m``: EVERY legal move is a candidate, so inclusion is
      trivially guaranteed and ``tau`` is irrelevant. The narrow case is the
      SAFE one, not the broken one.
    * ``m < topk`` needs ``(sims+1)//2 < topk``, i.e. ``sims < 2*topk - 1 = 31``
      at ``gumbel_topk: 16``. PRODUCTION NEVER GETS THERE. `configs/pbt2_small.yaml`
      runs ``mcts_simulations: 256`` (ramped down to 100 by `progressive_mcts`,
      still >= 31) and ``fast_simulations: 32``, so a FAST ply gives
      ``m = min(16, n_legal, 16) = 16`` -- the guarantee holds on fast plies
      exactly as on full ones, which is the case the review claimed was broken.

    So the guarantee is exact at every sim budget production runs. It would
    weaken only under a deliberate sub-31-sim configuration, and then the honest
    floor is ``1/m``, not ``1/topk``.

    ⚑ THE FLOOR IS ON THE RAW PROBABILITY; THE SEARCH SELECTS ON THE TEMPERED
    ONE (``logits / gumbel_policy_temp``, production 1.5). The two bars coincide
    at the realized legal-move count (mean 27.3) -- raw ``1/16`` is the smallest
    raw floor whose whole measured population clears the tempered ``1/16`` bar --
    but that is a COINCIDENCE OF THIS OPERATING POINT, not an identity.
    Re-derive it if ``gumbel_policy_temp``, ``gumbel_topk`` or the policy
    encoding changes.
    """
    return 1.0 / float(normalize_gumbel_topk(gumbel_topk))


def warn_if_below_search_inclusion(
    *, tau: float, tau_top1: float, tau_played: float,
    gumbel_topk: object, context: str,
) -> float:
    """Warn for each floor that has quietly stopped guaranteeing root inclusion.

    ⚑ IT CHECKS ALL THREE, AND `tau_played` IS THE ONE THAT MATTERS. The first
    version of this guard checked `tau` only -- the RANKING knob, default 0.15,
    which is 2.4x the production guarantee and therefore essentially never trips
    -- while `tau_played`, the one parameter whose documented sole job IS the
    inclusion guarantee, was never looked at. `resolve(tau=0.15,
    tau_played=0.001, gumbel_topk=16)` returned silently. A guard aimed at the
    parameter that cannot lose the property, and blind on the parameter that
    can, is the "gate that cannot fail" shape: present, green, and inert.

    ⚑ `tau_top1` JOINED IT for the same reason one level down. It is the floor
    SF's top-1 gets on the rows where our argmax already IS that move -- the
    strict `regret < our_r` clause keeps a move out of its own comparison, so
    `adaptive` is empty there and `max(tau, tau_top1)` reduces to `tau_top1`
    alone. Those are the rows invariant 1 calls the whole mechanism, and a
    sub-guarantee `tau_top1` revokes the root slot on exactly them. Nothing
    makes our own argmax safe: at the realized mean of 27.3 legal moves a
    flat-ish policy can put its top move under `1/16`. The reporting line in
    `Trainer` used to print `guarantees_inclusion=True` beside
    `tau_top1=0.001`; the mechanism was right and the BOOLEAN was lying,
    because it took a min over the two thresholds this guard happened to check.
    Guard and printed claim now read the same three.

    ``tau_played == 0.0`` stays SILENT: that is the documented collar-ablation
    arm, a deliberate off switch rather than a floor that has stopped working.
    Hence ``0 < tau_played < guarantee``, not ``tau_played < guarantee``. There
    is no equivalent for `tau_top1`: zero there is not an ablation, it is SF's
    best move losing its unconditional floor on the rows above.

    Returns the guarantee so callers can report it without re-deriving it.
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
            f"{name}={value!r} is BELOW the root-search inclusion guarantee "
            f"1/gumbel_topk={guarantee!r} (gumbel_topk={topk}, {context}): the "
            "floored move is no longer guaranteed a root-candidate slot, so the "
            "term stops buying the search routing it exists for and becomes a "
            "ranking nudge only. Raise it, or widen gumbel_topk, if that was not "
            "the intent.",
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
        """Fill the ``None`` defaults, validate, and check the search guarantee.

        ⚑ TAU HAS TWO SEPARABLE ROLES, AND THE DEFAULT SERVES THE SECOND ONE.
        Partitioning the arm contrast by whether the floor is already inert for
        SEARCH purposes (rows where our prior on deep-SF's best move already
        clears ``1/topk``) splits them cleanly:

        * ``tau = 1/gumbel_topk`` (0.0625 at the production topk of 16) is the
          SEARCH-INCLUSION GUARANTEE -- see ``search_inclusion_guarantee_tau``.
          BELOW it the guarantee is simply lost.
        * tau ABOVE that buys RANKING on the rows where inclusion is already
          satisfied: +0.278 [+0.237, +0.322] against the random-floor control at
          tau 0.15, +0.548 at 0.35, and 0.0% harmful rows at both -- it costs
          nothing where we are already right.

        So the default is **0.15**, comfortably above the 0.0625 guarantee, and
        the derivation survives as a GUARD rather than as the default: a
        resolved tau BELOW ``1/topk`` warns loudly, naming both numbers. It
        warns rather than raises because a deliberate sub-guarantee tau is a
        legitimate experiment -- but a tau that has quietly stopped guaranteeing
        anything is the "accepted, then silently meaningless" defect this
        codebase is full of, and it does not get to be silent.

        ⚑ `tau_played` -- THE COLLAR -- DEFAULTS TO THAT GUARANTEE, `1/topk`,
        because it is doing the opposite job. The floor's mass comes out of every
        non-member proportionally, and the biggest absolute loser is our own top
        move; if the move SEARCH ACTUALLY PLAYED lands under `1/topk` it drops out
        of the root candidate set entirely. Measured on 3072 production rows, the
        share of rows whose played move is squeezed below the bar by an
        UNCOLLARED floor: 0.26% at tau 0.10, **2.67% at tau 0.15**, 12.17% at
        0.35 -- against the 4.03% of rows the floor exists to fix. It would hand
        back most of its own win. With the collar the squeeze is 0.00% at every
        tau, and the collar itself fires on 0.81%-12.73% of rows.

        ``gumbel_topk`` is the width the trial ACTUALLY runs with, so callers
        must pass their own rather than let it default.
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
            # The collar's job is exactly "stay a root candidate", so its default
            # IS the inclusion guarantee -- not `tau`, which is calibrated for a
            # different job (ranking). 0.0 disables the collar; that is the
            # ablation arm and it is a clean no-op, not a special case.
            tau_played=guarantee if tau_played is None else float(tau_played),
        )
  # AFTER the range check, deliberately: an out-of-range value must RAISE, and a
  # warning emitted first would put "your tau is below the guarantee" on the log
  # ahead of the error that actually explains the failure.
        warn_if_below_search_inclusion(
            tau=params.tau, tau_top1=params.tau_top1, tau_played=params.tau_played,
            gumbel_topk=gumbel_topk, context="at config load",
        )
        return params


def sf_policy_floor_deficit(
    probs: torch.Tensor,
    regret: torch.Tensor,
    legal: torch.Tensor | None,
    played_target: torch.Tensor | None = None,
    *,
    params: SfPolicyFloorParams,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-sided probability floor on SF-approved moves. Returns (deficit, binds).

    ``deficit`` is per-row ``sum_{m in F} relu(tau_m - p_m)``; ``binds`` is the
    per-row 0/1 indicator that at least one move in ``F`` was actually below its
    floor. NEITHER is masked to the covered rows -- the caller applies its own
    row mask, exactly as the other per-sample losses here do.

    The set, recovered from ``sf_p0_regret`` alone (no new shard field):

        top1 = argmin over LEGAL moves of regret          # SF's best
        F    = {top1} u {m : regret_m <= delta_cp/CAP AND regret_m < regret_ours}

    with ``ours = argmax p``. Three properties, each pinned by a test:

    1. ``top1`` IS IN F UNCONDITIONALLY. That is the whole mechanism: MCTS never
       expands a move at ~0 prior, so the net never learns why SF's move beats
       its own. Guaranteeing a small prior gets the move SEARCHED and refutable.
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
    top move, so an uncollared floor squeezes the move search actually PLAYED out
    of the root candidate set on 2.67% of production rows at ``tau = 0.15``,
    against the 4.03% it exists to fix. See ``SfPolicyFloorParams.resolve``.

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

    ``legal=None`` means the batch had no legal mask and every action is in the
    softmax's support (see ``policy_legal_bool``).
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
    floors = torch.where(adaptive, torch.full_like(probs, float(params.tau)), 0.0)
    floors = floors.scatter_reduce(
        -1, top1, torch.full_like(top1, float(params.tau_top1), dtype=probs.dtype),
        reduce="amax", include_self=True,
    )
    if played_target is not None:
        played = played_target.argmax(dim=-1, keepdim=True)
  # A row whose policy target carries no mass has no played move to protect
  # (an absent or masked-out target argmaxes to index 0), so its collar
  # threshold is 0.0 -- the same no-op the ablation uses.
        has_target = (played_target.sum(-1, keepdim=True) > 0).to(probs.dtype)
        floors = floors.scatter_reduce(
            -1, played, has_target * float(params.tau_played),
            reduce="amax", include_self=True,
        )
    deficit = torch.relu(floors - probs)
    return deficit.sum(-1), (deficit > 0).any(-1).to(probs.dtype)


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
    """Clamp negatives to 0, optionally soften via ``p^(1/T)``, renormalize."""
    if sf_wdl_raw is None:
        return None
    p = sf_wdl_raw.clamp_min(0.0)
    if temperature != 1.0 and temperature > 0.0:
        p = p.clamp_min(1e-6).pow(1.0 / float(temperature))
    return normalize_distribution(p)


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


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    w_policy: float = 1.0,
    w_soft: float = 0.5,
    w_future: float = 0.15,
    w_sf_own: float = 0.0,
    w_sf_own_regret: float = 0.0,
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

    has_soft = _get_mask(batch, "has_policy_soft")
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
  # moves SF ranks at or above the one we picked. Deliberately modest -- it buys
  # SF's move a SEARCH SLOT so it can be refuted, it is not a policy correction.
    floor_params = sf_policy_floor if sf_policy_floor is not None else SfPolicyFloorParams()
    if sf_p0_regret_t is not None:
        po_probs = torch.softmax(masked_base, dim=-1)
        reg_vec = align_action_values(sf_p0_regret_t, int(base_policy_logits.shape[-1]))
        sf_own_regret = (po_probs * reg_vec).sum(-1)
        sf_floor, sf_floor_binds = sf_policy_floor_deficit(
            po_probs, reg_vec,
            policy_legal_bool(batch, width=int(base_policy_logits.shape[-1])),
            aligned_pol_target * has_policy.unsqueeze(-1),
            params=floor_params,
        )
    else:
        sf_own_regret = zero_loss
        sf_floor = zero_loss
        sf_floor_binds = zero_loss

  # DIAGNOSTIC ONLY — hard one-hot CE against the recorded game result. The
  # optimizer never sees this term (see ``blended_wdl_ce`` below, which is the
  # value loss in ``total``). Reported as ``wdl_onehot_ce`` so nothing can
  # mistake it for the trained loss again; see docs/rl_loop_audit.md I7.
    wdl_onehot_ce = F.cross_entropy(outputs["wdl"], batch["wdl_t"], reduction="none")

  # SF move prediction: target and legal mask are in the t+1 move space (opp POV).
    has_sf_move = _get_mask(batch, "has_sf_move")
    has_sf_policy = _get_mask(batch, "has_sf_policy") if "has_sf_policy" in batch else has_sf_move

    sf_pol_logits = outputs.get("policy_sf")
    sf_policy_target = batch.get("sf_policy_t")
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
    sf_wdl_frac_f = max(0.0, float(sf_wdl_frac))
    search_wdl_frac_f = max(0.0, float(search_wdl_frac))
    blend_sum = sf_wdl_frac_f + search_wdl_frac_f
    if blend_sum > 1.0:
        sf_wdl_frac_f /= blend_sum
        search_wdl_frac_f /= blend_sum
        game_frac = 0.0
    else:
        game_frac = 1.0 - blend_sum
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
        joint_count = joint.sum().clamp_min(1.0)
        sf_search_agree_frac = (agree.sum() / joint_count).detach()
        sf_search_disagree_sf_low_frac = (dis_sf_low.sum() / joint_count).detach()
        sf_search_disagree_sf_high_frac = (dis_sf_high.sum() / joint_count).detach()
    else:
        dis_sf_low = torch.zeros_like(sf_available)
        dis_sf_high = torch.zeros_like(sf_available)
        zero_scalar = torch.zeros((), device=sf_available.device)
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
    if sf_wdl_probs is not None:
        sf_component = (
            sf_effective_b * sf_wdl_probs + (1.0 - sf_effective_b) * blend_fallback_target
        )
    else:
        sf_component = blend_fallback_target
    search_available_b = search_available.unsqueeze(1)
    if search_wdl_probs is not None:
        search_component = (
            search_available_b * search_wdl_probs + (1.0 - search_available_b) * blend_fallback_target
        )
    else:
        search_component = blend_fallback_target
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
    m_policy = masked_mean(pol_ce, pol_base)
    m_soft = masked_mean(soft_ce, net_mask * has_soft)
    m_future = masked_mean(future_ce, net_mask * has_future)
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
    m_sf_policy_floor = masked_mean(sf_floor, sf_p0_regret_base)
  # BINDING RATE, the observation that separates "the weight reached the loss"
  # from "the term did something". A floor that never binds is a weight
  # multiplied into a structural zero -- accepted, threaded, and silently
  # inert, which is this repo's signature defect. Both columns divide by
  # `sf_own_regret_rows`, the SAME count, because they are masked by the SAME
  # tensor; a second count derived here could disagree with it.
    sf_policy_floor_sum, _ = masked_sum_and_count(sf_floor, sf_p0_regret_base)
    sf_policy_floor_binds_sum, _ = masked_sum_and_count(sf_floor_binds, sf_p0_regret_base)
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
    m_sf_move = masked_mean(sf_move_ce, net_mask * has_sf_policy)
    m_sf_eval = masked_mean(sf_eval_ce, m_sf_eval_mask)
    m_cat = masked_mean(cat_ce, net_mask * has_cat)
    m_vol = masked_mean(vol_loss, net_mask * has_vol)
    m_sf_vol = masked_mean(sf_vol_loss, net_mask * has_sf_vol)
    m_ml = masked_mean(ml_loss, net_mask * has_moves_left)

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
    for suffix, m in split_masks:
        policy_bucket_mask = pol_base * m
        split_losses[f"policy_loss_{suffix}"] = masked_mean(pol_ce, policy_bucket_mask)
        wdl_bucket_mask = net_mask * m
        split_losses[f"wdl_loss_{suffix}"] = masked_mean(blended_wdl_ce, wdl_bucket_mask)
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

    total = (
        float(w_policy) * m_policy
        + float(w_soft) * m_soft
        + float(w_future) * m_future
        + float(w_sf_own) * m_sf_own
        + float(w_sf_own_regret) * m_sf_own_regret
        + float(w_wdl) * m_blended_wdl
        + float(w_sf_move) * m_sf_move
        + float(w_sf_eval) * m_sf_eval
        + float(w_categorical) * m_cat
        + float(w_volatility) * m_vol
        + float(w_sf_volatility) * m_sf_vol
        + float(w_moves_left) * m_ml
    )

  # ⚑ ADDED ONLY WHEN THE WEIGHT IS NON-ZERO, unlike every term above, and that
  # is deliberate on two counts. (1) INERTNESS IS EXACT: at the default 0.0 the
  # expression for `total` is the one that existed before this term, so the
  # objective is bit-identical rather than identical-up-to-a-`+ 0.0`. (2) `0.0 *
  # x` IS NOT ZERO FOR EVERY `x`: a NaN or inf leaking out of the floor would
  # poison `total` through a weight that is supposed to mean "off" -- the same
  # shape as "a clamp is not a validator", where min/max quietly propagate NaN
  # while the guard's own counter reads healthy. The diagnostic columns below
  # are computed either way, so switching the weight on cannot be the first time
  # anyone sees what the term does.
    if float(floor_params.w) != 0.0:
        total = total + float(floor_params.w) * m_sf_policy_floor

  # Reported value-loss names (docs/rl_loop_audit.md I7):
  #   wdl_ce / blended_wdl_ce -> the SAME tensor, the loss the optimizer sees.
  #     `wdl_ce` is the name people reach for (it becomes the `wdl_loss`
  #     column), `blended_wdl_ce` is kept because existing readers use it.
  #   wdl_onehot_ce -> the hard one-hot diagnostic, never in `total`.
    return {
        "total": total,
        "policy_ce": m_policy,
        "wdl_ce": m_blended_wdl,
        "blended_wdl_ce": m_blended_wdl,
        "wdl_onehot_ce": m_wdl_onehot,
        "soft_policy_ce": m_soft,
        "soft_mask_kept_frac": soft_mask_kept_frac,
        "future_policy_ce": m_future,
        "sf_own_ce": m_sf_own,
        "sf_own_regret": m_sf_own_regret,
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
        "sf_own_regret_sum": sf_own_regret_sum,
        "sf_own_regret_rows": sf_own_regret_rows,
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
        "frac_is_selfplay": masked_mean(is_sp_bool, has_is_sp),
        "frac_tagged": masked_mean(has_is_sp, net_mask),
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
    }


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
