from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Literal, overload

import chess
import numpy as np
import torch

from chess_anti_engine.encoding.lc0 import LC0_HISTORY_LEGACY
from chess_anti_engine.encoding import encode_positions_batch
from chess_anti_engine.encoding.features import EXTRA_FEATURES_V2_THREATS, relation_matrices
from chess_anti_engine.inference import BatchEvaluator, LocalModelEvaluator
from chess_anti_engine.mcts.puct import (
    Node,
    _backprop,
    _expand_sparse,
    _terminal_value,
)
from chess_anti_engine.mcts.puct import (
    _value_scalar_from_wdl_logits as _wdl_to_q,
)
from chess_anti_engine.mcts.root_tactics import (
    immediate_mate_root_policy,
    immediate_terminal_draw_indices,
)
from chess_anti_engine.moves import (
    MODEL_POLICY_ENCODING,
    POLICY_SIZE,
    policy_batch_to_full_if_needed,
)
from chess_anti_engine.moves.encode import legal_move_indices

GumbelManyResult = tuple[list[np.ndarray], list[int], list[float], list[np.ndarray]]
GumbelManyDiagnosticsResult = tuple[
    list[np.ndarray], list[int], list[float], list[np.ndarray], list[dict[str, float] | None]
]


def _gumbel(rng: np.random.Generator, size: int) -> np.ndarray:
    u = rng.random(size=size)
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return -np.log(-np.log(u))


from chess_anti_engine.mcts.sampling import sample_action_with_temperature  # noqa: E402
from chess_anti_engine.utils.numpy_helpers import softmax_1d as _softmax  # noqa: E402


# Dataset-mean default for the volatility anchor; re-derive per experiment
# (see configs/exp_volatility_search.yaml) — this is the single source the
# SearchConfig/TrialConfig/worker plumbing defaults mirror.
DEFAULT_VOLATILITY_ANCHOR = 0.05

# Single source of truth for the SELFPLAY/RL Gumbel value-transform scale — the
# `gumbel_c_scale` the yaml pins and every plumbing default mirrors (SearchConfig,
# TrialConfig, the worker reco fallback, the published recommended_worker).
#
# DELIBERATELY different from PLAY_SEARCH_DEFAULTS["c_scale"] = 0.025 below. Do NOT
# "clean up" the two to agree: the optimal q_scale keys off the TOTAL sim budget, so
# no single value is sim-invariant and each deployment gets its own measured optimum.
#   selfplay @256 sims (2026-06-16 puzzle screen, n=1000): 0.1 -> 0.688 accuracy,
#     0.05 -> 0.652, 0.025 -> 0.598. Lowering it HURTS the RL regime.
#   UCI/match @8000 sims: 0.025 measured +301 Elo, adopted for the high-sim regime.
# tests/test_selfplay_gumbel_c_scale.py fails if anyone unifies them.
SELFPLAY_GUMBEL_C_SCALE = 0.1

# Single source of truth for the production PLAY/EVAL Gumbel search settings (UCI,
# puzzle eval, the standardized arena, the training-gate match). Reference this
# from every such entry point so the tuned optimum never drifts across call sites.
# DISTINCT from SELFPLAY/training search, which is YAML/reco-driven and intentionally
# keeps c_scale=0.1, no root split, linear root (the c_scale=0.025 win + root-log are
# high-sim PLAY tunings; for low-sim selfplay they are a no-op-to-sub-significant lever
# that needs a live A/B, not an offline flip). Search SHAPE only — NOT the sim count
# (that varies: UCI chunk_sims, puzzle --sims, arena matched_sims). Keys are
# GumbelConfig fields. Provenance (frozen deep-SF audit, paired regret-vs-sims):
# c_scale=0.025 +301 Elo @8k; c_visit_root=900 (inert under root-log, kept);
# c_scale_root=7/q_visit_exp_root=-1 LOG root keeps q_scale ~100 from 256 to millions
# of nodes instead of exploding (~25000 @1M) and reversing under more search.
#
# EVERY key here must be a knob a Gumbel search can actually act on. The four
# PUCT knobs that used to live here (c_puct / cpuct_factor / cpuct_base /
# fpu_reduction) were removed 2026-08-03: see INERT_GUMBEL_KNOBS below and
# PLAY_PUCT_DEFAULTS for their new home.
PLAY_SEARCH_DEFAULTS: dict[str, float | int] = {
    "c_scale": 0.025,
    "c_visit": 50.0,
    "c_visit_root": 900.0,
    "c_scale_root": 7.0,
    "q_visit_exp_root": -1.0,
    "topk": 32,
  # 1.0 is a numeric no-op (`apply_policy_temp` returns the logits unchanged),
  # so adding it here changed no search. It is here because ABSENCE was the
  # defect: `policy_temp` is a live positive control in the very audit that
  # named the four inert knobs, and it was the one shape knob no play-path
  # config surface could set -- not this dict, not `--*-gumbel`'s realized
  # dump, not the UCI CLI. A knob that cannot be set is not "at its default",
  # it is unreachable, and an unreachable knob reads as a settled question.
    "policy_temp": 1.0,
}

# GumbelConfig fields that PROVABLY cannot change a Gumbel search's output, and
# so must never appear in a "realized Gumbel search config" (play-path audit
# 2026-08-03, F2, repro `scratchpad/code_audit_20260803/repro_inert_knobs.py`:
# perturbing each gives L1 0.000000 over the returned policy and 0/4 moves
# changed, against positive controls topk/halving_div/policy_temp that move it).
#
# Why they are inert: the PUCT descent (`tree_select_child`, the ONLY consumer
# of c_puct/fpu_reduction in the tree) is reached only when `sel->full_tree` is
# false (`_mcts_tree.c:3160-3169`), and `GumbelConfig.full_tree` defaults True
# with nothing anywhere setting it false. cpuct_factor/cpuct_base are not even
# arguments of `start_gumbel_sims`; the C reads them off the tree via
# `set_cpuct_scaling`, which only the PUCT entry points call.
#
# The fields stay on GumbelConfig because `uci/search.py` and `uci/walker_pool.py`
# read them OFF the shared config object to drive their genuine PUCT walkers --
# they are live there, and only there.
INERT_GUMBEL_KNOBS: frozenset[str] = frozenset(
    {"c_puct", "cpuct_factor", "cpuct_base", "fpu_reduction"}
)

# GumbelConfig fields the C fast path (`run_gumbel_root_many_c`) does not
# implement. They are NOT inert -- `run_gumbel_root_many` (Python) really acts
# on them -- but they are dropped WITHOUT A WORD by whichever dispatcher picks
# the C path, which is the other way a knob returns a flawless null.
#
# The guard therefore has to live on the DISPATCH, not on a CLI parser: the CLI
# is not what chooses the path. `selfplay.match.pick_moves_for_boards` already
# routes volatility-enabled configs to the Python path and warns
# (`warn_volatility_python_path`); `assert_c_path_can_run` is the same contract
# stated once, so a NEW python-only field cannot be added without a dispatcher
# either honouring it or refusing loudly. See
# `tests/test_uci_search_options.py::test_the_c_path_refuses_a_python_only_knob`.
PY_ONLY_GUMBEL_KNOBS: frozenset[str] = frozenset(
    {"volatility_q_scale", "volatility_fpu"}
)


def python_only_knobs_set(cfg: GumbelConfig) -> tuple[str, ...]:
    """Python-path-only fields on ``cfg`` that are at a non-default value."""
    base = GumbelConfig()
    return tuple(
        sorted(
            k for k in PY_ONLY_GUMBEL_KNOBS
            if float(getattr(cfg, k)) != float(getattr(base, k))
        )
    )


def assert_c_path_can_run(cfg: GumbelConfig, *, where: str) -> None:
    """Refuse to enter the C Gumbel path with a Python-only knob set.

    Raising beats warning here. A warning on a knob that silently does nothing
    still produces a complete, reproducible, wrong measurement; the caller's
    only correct moves are to drop to the Python path or to zero the knob, and
    both are things the caller must decide.
    """
    offenders = python_only_knobs_set(cfg)
    if not offenders:
        return
    raise ValueError(
        f"{where}: {', '.join(offenders)} are Python-path only; call "
        "run_gumbel_root_many (mcts/gumbel.py) instead. The C path has no code "
        "for them, so it would drop them silently and return a search that "
        "looks like a clean null. Zero them, or route to the Python path "
        "(see PY_ONLY_GUMBEL_KNOBS)."
    )

# Lc0 classic-search defaults (Oct 2025 engine flags page) for the UCI engine's
# PUCT descent -- the walker pool and the multi-GPU PUCV pool, which really do
# consume these. NOT a Gumbel surface: do not merge back into
# PLAY_SEARCH_DEFAULTS (tests/test_inert_gumbel_knobs.py fails if you do).
PLAY_PUCT_DEFAULTS: dict[str, float] = {
    "c_puct": 1.75,
    "cpuct_factor": 3.89,
    "cpuct_base": 38739.0,
    "fpu_reduction": 0.33,
}

# The two C-path search controls that are NOT GumbelConfig fields, so no
# `dataclasses.replace`-based override surface can reach them --- not
# PLAY_SEARCH_DEFAULTS above, not `gumbel_overrides`, not arena_standard's
# `--cand-gumbel`. That is precisely how they went missing: every arena Elo in
# docs/experiment_ledger.md was measured at `vloss_weight=0`, the pre-C17
# duplicate-leaf search, while production selfplay ran `gumbel_vloss_weight: 1`
# (ledger 2026-07-28, "SECOND, INDEPENDENT arena/production divergence"). They
# are function arguments of `run_gumbel_root_many_c`, so they have to be passed
# explicitly at every call site; these constants are the PLAY/UCI side of that
# pair, mirrored by `uci.search.SearchWorker` and the `--vloss-weight` /
# `MinibatchSize` defaults so the play shape cannot drift from the engine.
# The TRAINING side is not here: it is yaml-driven (`gumbel_vloss_weight` /
# `gumbel_target_batch` in the selfplay SearchConfig) and must be read from the
# production config, never hard-coded.
PLAY_SEARCH_VLOSS_WEIGHT = 3
PLAY_SEARCH_TARGET_BATCH = 0


@dataclass
class GumbelConfig:
    simulations: int = 50
    topk: int = 16
    temperature: float = 1.0
    # Prior temperature on the policy-head logits before they seed the tree
    # (logits/policy_temp). >1 softens the prior, <1 sharpens it; 1.0 = no-op.
    policy_temp: float = 1.0
    c_visit: float = 50.0
    # Library default == the SELFPLAY/RL optimum; PLAY/EVAL entry points override
    # it from PLAY_SEARCH_DEFAULTS. `GumbelConfig()` is used as "the RL shape by
    # construction" (scripts/audit_targets.py), so the two must stay bound.
    c_scale: float = SELFPLAY_GUMBEL_C_SCALE
    c_puct: float = 2.5
    # Lc0 classic visit-dependent Cpuct (PUCT descent only). factor<=0 = fixed.
    # Library default factor=0 preserves training bit-identity; UCI play uses
    # PLAY_SEARCH_DEFAULTS (1.75 / 3.89 / 38739).
    cpuct_factor: float = 0.0
    cpuct_base: float = 38739.0
    fpu_reduction: float = 1.2
    # ⚑ The three DESCENT value-transform knobs that used to sit here --
    # `q_visit_exp` (exponent on max_visit), `q_global_scale` (scale the descent
    # by the ROOT's max child-visit) and `q_visit_floor` (additive decoupled
    # floor) -- were DELETED, never promoted. They were sweep leftovers from the
    # root/descent split work: no config in this repo ever set one, they are
    # absent from PLAY_SEARCH_DEFAULTS, and the axis that DID pay off is the
    # separate ROOT family (`c_scale_root` / `q_visit_exp_root` / `c_visit_root`)
    # below, which production PLAY runs as the log root.
    #
    # The C still accepts all three as optional `start_gumbel_sims` arguments, so
    # `gumbel_c` passes their former defaults (1.0 / 0 / -1.0) as literals -- see
    # the call sites there. That keeps the search bit-identical while removing the
    # knob: deleting a knob pinned at its default is behaviour-identical by
    # construction, and `tests/test_descent_qknob_deletion_parity.py` measures it.
    # ── sigma does two jobs with one number; this splits them ──────────────
    # The improved policy softmax(log_prior + sigma*Qbar) is used for TWO
    # different purposes that want DIFFERENT sigmas:
    #   PLAY   -- the move actually made. Wants the strength-optimal sigma; the
    #             c_scale sweep put that near 0.2 for this net.
    #   TARGET -- the row stored for training. Wants a SMALLER sigma, because a
    #             move the target never puts mass on is a move the next
    #             generation's prior stops proposing, so search never revisits
    #             it. Sharpening the target is not free the way sharpening play
    #             is: play errors cost one game, target errors are absorbed.
    # ⇒ the strength-optimal sigma is an UPPER BOUND on the training-optimal
    # sigma, and running one number for both pins the target to that bound.
    #
    # This knob expresses the split the way the operator thinks about it --
    # "search deep, but only TRUST the result as much as a shallow search" --
    # by clamping the max_visit that feeds sigma for the STORED target only:
    #     sigma_target = c_scale * (c_visit + min(max_visit, cap))
    # Sims-invariant by construction: raise `mcts_simulations` later and the
    # target's trust stays put, which a bare c_scale override cannot do.
    #
    # 0 = OFF and bit-identical (one sigma, current behaviour). The played move
    # is NEVER affected: sequential halving keeps the uncapped sigma, and so
    # does the temperature sample. That also means an arena is structurally
    # blind to this knob -- the only yardsticks are target quality against the
    # deep-SF ruler and, downstream, a training run.
    target_max_visit_cap: int = 0
    # ── the OTHER term of the same softmax: log_prior ─────────────────────
    # `policy_temp` divides the policy-head logits before they seed the tree,
    # so the prior that reaches `softmax(log_prior + sigma*Qbar)` is the
    # TEMPERED one. That is right for CANDIDATE SELECTION -- softening the
    # prior is how temperature puts more moves INTO the search -- and wrong for
    # the row we store, because the stored row is the next generation's
    # training target and the head is tempered AGAIN on the next pass:
    #     L -> L/T -> target = L/T + sigma*Qbar -> head learns it -> L/T again
    # whose fixed point is L* = sigma*Qbar * T/(T-1), i.e. 3*sigma*Qbar at
    # T=1.5. The policy head's independently learned information is discounted
    # geometrically toward a target determined ENTIRELY by Q.
    #
    # lc0 runs PolicyTemperature 1.45 safely because it trains on VISIT COUNTS,
    # which carry no additive log_prior term, so tempering never re-enters the
    # target algebraically. Gumbel-MuZero's completed-Q target does. Same knob,
    # different target algebra, opposite safety.
    #
    # Measured on live shards 2026-08-10 (ledger commit 46c1489e6): stored
    # H(policy_target) 0.9849 -> 1.2136 against a pre-registered 1.14 plateau
    # bar, and median KL(prior||target) up 11.8x. In a loop that is converging
    # that KL SHRINKS; it grew an order of magnitude because the target kept
    # running away from the head chasing it.
    #
    # True = keep the tempered prior for candidate selection and undo the
    # temperature in the STORED target's log_prior term only. False = OFF and
    # bit-identical. Like `target_max_visit_cap` this leaves the played move on
    # the tempered arm, so an arena cannot see it -- the yardstick is target
    # quality against the deep-SF audit set.
    target_untempered_prior: bool = False
    # Sequential-halving divisor: each round keeps ceil(n_cands/halving_div).
    # 2 = standard halving (keep top half; default). 3/4 = more aggressive
    # elimination (fewer rounds, visits concentrate on survivors sooner).
    # C path only (run_gumbel_root_many_c).
    halving_div: int = 2
    # Root-halving c_visit override (value-transform floor at the root sequential-
    # halving site only; descent keeps c_visit). Root and descent want different
    # floors: a large root floor (~670) makes one c_scale fit every sim count's
    # root q_scale, while a small descent floor (c_visit~50) keeps deep subtrees
    # from over-trusting Q. >= 0 sets the root floor; < 0 (default) = use c_visit
    # at both sites (legacy). C path only (run_gumbel_root_many_c).
    c_visit_root: float = -1.0
    # Root-halving-ONLY value-transform overrides (root and descent want
    # DIFFERENT transforms: the high-sim plateau is a root over-trust problem,
    # but the good mid-sim regret needs a tiny LINEAR descent q_scale). The root
    # reads c_scale_root / q_visit_exp_root; the descent keeps c_scale and is
    # always LINEAR in max_visit. Set q_visit_exp_root<0 for a LOG root
    # (slow-growth, sim-invariant 256->millions) while the descent stays linear.
    # Sentinels: c_scale_root<0 -> use c_scale; q_visit_exp_root>=90 -> exponent
    # 1.0, i.e. a LINEAR root (what selfplay runs). That sentinel used to read
    # the deleted `q_visit_exp`, whose default was 1.0, so pinning it to the
    # literal is the same search. C path only (run_gumbel_root_many_c).
    c_scale_root: float = -1.0
    q_visit_exp_root: float = 99.0
    full_tree: bool = True
    add_noise: bool = True  # Backward-compatible gate; use gumbel_scale for partial noise.
    gumbel_scale: float = 1.0
    input_history_encoding: str = LC0_HISTORY_LEGACY
    input_extra_features: str = EXTRA_FEATURES_V2_THREATS
    policy_encoding: str = MODEL_POLICY_ENCODING
  # Compute dynamic board-relation matrices per eval and pass them to the
  # evaluator as attention-bias input (model.use_dynamic_relations).
    compute_relations: bool = False
  # ── Volatility-aware search (Python path only; both default OFF) ────────
  # volatility_q_scale: exponent scaling the sigma(q) value-transform
  # constant per node by predicted volatility. The effective scale is
  #   c_scale * (volatility_anchor / vol)^volatility_q_scale
  # clipped to [1/volatility_factor_clip, volatility_factor_clip] x c_scale,
  # so at vol == anchor the behavior is IDENTICAL to today. High predicted
  # volatility -> smaller sigma -> flatter value transform -> candidates
  # survive halving longer; low volatility -> trust the value sooner.
    volatility_q_scale: float = 0.0
  # volatility_fpu: pessimistic first-play urgency — unvisited children's
  # completed value becomes mixed_value - volatility_fpu * vol (Q units).
    volatility_fpu: float = 0.0
  # Dataset-mean anchor for the volatility head's scalar summary (mean of
  # the 3 head components). Derive from a recent shard window before an
  # experiment (see configs/exp_volatility_search.yaml) and pin it here —
  # the normalization must stay frozen within an arena sweep.
    volatility_anchor: float = DEFAULT_VOLATILITY_ANCHOR
    volatility_factor_clip: float = 4.0


# Sanity band for the policy prior temperature, in the ONE place both the
# loader (`trial_config._policy_temperature`) and the hot-path predicate
# (`policy_temp_active`) read it from.
#
# ⚑ These are semantic bounds, not float32 bounds, and deliberately far tighter.
# The float32 cliffs are ~1e-38 (a logit of magnitude 1 divides to +inf) and
# ~1e39 (every logit divides to zero / denormal, i.e. an exactly uniform prior).
# Anything past ~20 is already uniform IN EFFECT long before it is numerically
# invalid --- at T=1e300 `apply_policy_temp` returns all-zero logits, which is
# search with the policy head switched off, and `math.isfinite` waves it
# through. That is the failure the band exists to close, so the band has to sit
# where the knob stops being a temperature rather than where IEEE754 stops.
#
# 0.05 / 20.0 are 8x beyond the widest values anything here has used: the
# sharpest tested is 0.4, the softest 2.5, and production runs 1.5 with a
# decided floor of 1.2. A run wanting more than a 13x departure from production
# is not tuning a prior temperature and should say so in the ledger first.
POLICY_TEMP_MIN = 0.05
POLICY_TEMP_MAX = 20.0


def policy_temp_active(policy_temp: float) -> bool:
    """True when ``policy_temp`` will actually change the priors.

    THE single definition of "tempering is on". ``apply_policy_temp`` below,
    both bf16 fast-path gates in ``gumbel_c`` and the worker's realized-shape
    log line all call this, so the transport gate, the arithmetic and the
    operator-facing claim can never drift apart: a guard has to share the
    criterion's instrument or it is guarding a different question.

    Out-of-band values read as *off* rather than raising because this is the hot
    path; config loading rejects the same set (``trial_config._policy_temperature``)
    so they cannot reach here from the yaml in the first place.

    ⚑ The band, not just ``isfinite``, is what makes this safe. ``nan`` falls out
    via the comparison and ``inf`` used to read as ACTIVE --- ``apply_policy_temp``
    then returns all-zero logits, a uniform prior over every legal move,
    announced in the realized-shape log line as tempering working normally. But
    ``isfinite`` closes only the literal: ``1e300`` produces the identical
    all-zero prior and ``1e-300`` produces all-inf, both finite and both
    ACTIVE under a bare ``pt > 0``. The yaml cannot deliver any of them, but
    ``arena_standard.py --cand-gumbel policy_temp=1e300`` reaches
    ``dataclasses.replace`` without passing the loader's validator. This
    predicate is THE definition of "tempering is on", so it agrees with the
    loader on the whole rejection set rather than leaving one entry point a hole.
    """
    pt = float(policy_temp)
    return math.isfinite(pt) and POLICY_TEMP_MIN <= pt <= POLICY_TEMP_MAX and pt != 1.0


# The divisor the C sequential halving silently raises anything smaller to:
# `g->halving_div = (halving_div >= 2) ? halving_div : 2` (_mcts_tree.c).
MIN_HALVING_DIV = 2

# The candidate count BOTH Python candidate-selection sites silently raise
# anything smaller to: `m = max(2, ...)` in `gumbel_c._select_root_candidates`
# and in `gumbel._init_board_search_state`. ⚑ `topk` never reaches the C at all
# (zero occurrences in `_mcts_tree.c`): the C is handed the already-chosen
# candidate LIST, so "the C signature rejects it" -- which an earlier revision
# of this file claimed as the reason topk needed no check -- was false, and
# `topk` 0 / 1 / -5 all run m=2 while every record names the operator's number.
# `topk` is in PLAY_SEARCH_DEFAULTS, so that record is written on every row.
MIN_TOPK = 2


def validate_gumbel_config(cfg: GumbelConfig, *, where: str) -> None:
    """Refuse a search config carrying a value the search will not run.

    THE band's second home is not allowed to exist, so this is where every
    externally-built ``GumbelConfig`` is checked -- CLI overrides, ``setoption``
    seeds, anything that reaches ``dataclasses.replace`` without passing the
    yaml loader's validator (``trial_config._policy_temperature``).

    Call it at CONSTRUCTION/OVERRIDE boundaries, not per search call: the hot
    path deliberately cannot raise (``policy_temp_active`` reads an out-of-band
    temperature as *off* because it runs per leaf), and that is exactly why the
    refusal has to happen where the config is built instead.

    ⚑ It refuses rather than repairs. Clamping ``policy_temp=1e300`` to 20.0
    would run a search the operator did not ask for, and defaulting it to 1.0 is
    what happens today -- silently, under a record naming 1e300. A sweep over
    out-of-band temperatures produces identical arms banked as different
    settings, which is the c_puct Swiss (play-path audit 2026-08-03 F2) with a
    live knob instead of a dead one.

    What it checks, and why each one is a silent no-op rather than a crash:

    * **every numeric field finite.** Each sentinel in this file is a
      COMPARISON, and every comparison against ``nan`` is False, so a non-finite
      knob does not fail loudly -- it takes the fallback arm. ``nan`` is neither
      ``< 90`` nor ``>= 90``, so ``q_visit_exp_root=nan`` silently reverts the
      root to the linear exponent 1.0; ``c_scale_root=nan`` silently reverts to
      ``c_scale``; ``policy_temp=inf`` reads as off.
    * **``policy_temp`` inside the band.** ⚑ 1.0 is INSIDE the band and valid --
      it is the identity, and ``policy_temp_active`` reports it as "off" for
      that reason alone, not because it is out of range. So the check cannot be
      ``not policy_temp_active(...)``: that would refuse the shipped default and
      the whole PLAY shape with it. Everything genuinely outside the band
      reaches ``apply_policy_temp`` and returns the priors untouched.
    * **``halving_div >= MIN_HALVING_DIV``** and **``topk >= MIN_TOPK``**, both
      integral. Below the minimum each is silently raised (see the constants);
      a fractional value is silently TRUNCATED by the ``int()`` every consumer
      applies, so ``halving_div=2.9`` and ``topk=2.9`` are 2.

    What it deliberately does NOT check, so this does not grow into a second
    opinion about knobs that already have one:

    * the sentinel RANGES (``c_scale_root < 0``, ``c_visit_root < 0``,
      ``q_visit_exp_root >= 90``, ``target_max_visit_cap <= 0``). Every value
      inside one of those is a legitimate spelling of "off" rather than a
      request that got dropped, and ``mcts.search_options.branch_note`` is the
      shipped answer for telling an operator so.
    * ``INERT_GUMBEL_KNOBS`` and ``PY_ONLY_GUMBEL_KNOBS`` -- refused by the
      ``--*-gumbel`` parsers and by ``assert_c_path_can_run`` respectively. A
      knob with a guard does not need a second one.
    * ``simulations`` and the rest: no measured silent-clamp site. The rule for
      adding one is the rule these three met -- point at the line that swallows
      the value, not at an intuition about what "should" be out of range.

    Raises ``ValueError``; CLI surfaces wrap it in ``SystemExit`` so an operator
    gets the message and not a traceback.
    """
    problems: list[str] = []
    for f in fields(cfg):
        value = getattr(cfg, f.name)
      # ⚑ Coerce, do not isinstance-whitelist. `np.float32(nan)` is not an
      # instance of `float` (only float64 subclasses it), so a whitelist SKIPS
      # numpy scalars -- an accept-then-ignore inside the guard against
      # accept-then-ignore. `str` is excluded first because the encoding fields
      # are strings and `float("nan")` would parse one spelled that way.
        if isinstance(value, (bool, str)):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            problems.append(
                f"{f.name}={value!r} is not finite, so every gate that reads it "
                "takes its fallback arm instead of failing: the search runs a "
                "shape nobody asked for and the record names yours"
            )
  # Deliberately NOT skipped for a non-finite value already reported above:
  # `nan`/`inf` are out of band too, and the operator-facing phrase "outside
  # the band" is what tells them the knob -- not the float -- is the problem.
    pt = float(cfg.policy_temp)
    if pt != 1.0 and not policy_temp_active(pt):
        problems.append(
            f"policy_temp={pt!r} is outside the band where it does anything "
            f"([{POLICY_TEMP_MIN}, {POLICY_TEMP_MAX}] inclusive; 1.0 is inside "
            "it and valid -- the identity), so apply_policy_temp would return "
            "the priors untouched and the search would run UNTEMPERED (T=1.0) "
            "while every record of it names your value"
        )
    problems.extend(
        _clamped_int_problems(
            "halving_div", cfg.halving_div, minimum=MIN_HALVING_DIV,
            clamped_by=(
                "the C search (`g->halving_div = (halving_div >= 2) ? "
                "halving_div : 2`, _mcts_tree.c)"
            ),
            runs_instead="STANDARD halving",
        )
    )
    problems.extend(
        _clamped_int_problems(
            "topk", cfg.topk, minimum=MIN_TOPK,
            clamped_by=(
                "both Python candidate-selection sites (`m = max(2, ...)` in "
                "gumbel_c._select_root_candidates and in "
                "gumbel._init_board_search_state; topk never reaches the C)"
            ),
            runs_instead=f"a {MIN_TOPK}-candidate root",
        )
    )
    if problems:
        raise ValueError(f"{where}: " + "; ".join(problems))


def _clamped_int_problems(
    name: str, raw: float | int, *, minimum: int, clamped_by: str,
    runs_instead: str,
) -> list[str]:
    """Refusals for an integer knob that is silently clamped and truncated.

    Two ways to lose a value in one knob, so two messages: below ``minimum`` it
    is raised by the consumer, and a fractional value is truncated by the
    ``int()`` every consumer applies. Neither says anything at runtime.
    """
  # A non-finite value is reported by the finiteness pass; comparing it here
  # would take the fallback arm and report nothing.
    value = float(raw)
    if not math.isfinite(value):
        return []
    out: list[str] = []
    if value < minimum:
        out.append(
            f"{name}={raw!r} is below {minimum}, which {clamped_by} silently "
            f"raises to {minimum}, so the search would run {runs_instead} "
            "under a record naming your value"
        )
    elif not value.is_integer():
        out.append(
            f"{name}={raw!r} is not an integer and every consumer applies "
            f"int(), so the search would run {name}={int(value)} under a "
            "record naming your value"
        )
    return out


def apply_policy_temp(pol: np.ndarray, *, cfg: GumbelConfig) -> np.ndarray:
    """Scale policy logits by the prior temperature before they seed the tree.
    T>1 softens the prior, T<1 sharpens it, 1.0 (or <=0) = no-op. Shared by the
    Python and C (gumbel_c) search paths so both honor policy_temp identically."""
    pt = float(getattr(cfg, "policy_temp", 1.0))
    if policy_temp_active(pt):
        return pol / pt
    return pol


def _policy_logits_to_full(pol_logits: np.ndarray, *, cfg: GumbelConfig) -> np.ndarray:
    pol = apply_policy_temp(np.asarray(pol_logits, dtype=np.float32), cfg=cfg)
    return policy_batch_to_full_if_needed(pol, fill_value=-1e9)


def target_log_prior(log_prior: np.ndarray, *, cfg: GumbelConfig) -> np.ndarray:
    """The STORED improved policy's ``log_prior`` term, with ``policy_temp`` undone.

    Returns ``log_prior`` ITSELF (same object, so callers can test identity and
    skip the second softmax entirely) whenever the two terms coincide: the knob
    is off, or tempering was not applied in the first place.

    ── why multiplying by T *is* the raw-logit derivation, not an approximation

    Both call sites build the prior as a softmax over the search's legal set of
    the ALREADY-TEMPERED logits (``gumbel._masked_priors``; the inline root
    softmax in ``gumbel_c``), and ``apply_policy_temp`` is literally
    ``logits / T``. So for every legal action ``a``::

        log pri[a] = raw[a]/T - logsumexp(raw[legal]/T)
        T * log pri[a] = raw[a] - T*logsumexp(raw[legal]/T)

    The subtracted term does not depend on ``a``, and the improved policy is a
    softmax, which is invariant to an additive constant. ``T * log_prior`` is
    therefore EXACTLY the untempered log-prior the raw logits would have given,
    not a reconstruction of it -- no second softmax, no extra array.

    ── why not plumb the raw logits down instead

    They are not uniformly available. ``_build_improved_policy_for_board`` sees
    only ``_BoardSearchState.priors``, and ``gumbel_c``'s compact-root arm never
    materialises full-width logits at all (it receives bf16 logits for the legal
    subset). Carrying them to both sites would add two new hot-path shapes to
    recompute a value that is already exact here.

    ── the two places the identity has an edge, both harmless

    * ``log_prior`` is floored at ``log(1e-12)``. A prior AT that floor scales to
      ``T*log(1e-12)`` rather than its true (smaller) untempered value, i.e. this
      over-states a move that already holds < 1e-12 of the prior mass. After the
      softmax it is ~1e-18 either way; both arms round it to nothing.
    * The identity assumes ``pri`` is a PURE tempered softmax over the search's
      legal set. It is -- Gumbel adds its noise to the SCORE, not to the prior,
      and ``gumbel_c`` prunes terminal-draw moves BEFORE the softmax, not after.
      That assumption is not left as a comment:
      ``tests/test_untempered_target_prior.py`` runs both search paths and
      compares against a prior recomputed from the model's raw logits.

    ⚑ The gate is ``policy_temp_active``, THE definition of "tempering is on",
    shared with ``apply_policy_temp``. It has to be the same predicate: a
    non-active-but-finite ``policy_temp`` (1e300, 1e-300, nan) leaves the priors
    UNTEMPERED, and undoing a temperature that was never applied would multiply
    the log-prior by 1e300. A guard that does not share the criterion's
    instrument is guarding a different question.
    """
    if not bool(cfg.target_untempered_prior):
        return log_prior
    pt = float(getattr(cfg, "policy_temp", 1.0))
    if not policy_temp_active(pt):
        return log_prior
    return log_prior * pt


def gumbel_policy_diagnostics(
    *,
    probs: np.ndarray,
    action: int,
    legal: np.ndarray,
    candidates: list[int] | np.ndarray | None,
) -> dict[str, float]:
    """Cheap diagnostics for the final Gumbel training policy."""
    legal = np.asarray(legal, dtype=np.int64)
    if legal.size == 0:
        return {}

    p = np.asarray(probs, dtype=np.float64)
    legal_p = np.maximum(p[legal], 0.0)
    total = float(legal_p.sum(dtype=np.float64))
    if total <= 0.0 or not np.isfinite(total):
        return {}
    legal_p = legal_p / total

    top_local = int(np.argmax(legal_p))
    top_action = int(legal[top_local])
    top_prob = float(legal_p[top_local])
    action_prob = float(p[int(action)]) if 0 <= int(action) < p.shape[0] else 0.0
    positive = legal_p[legal_p > 0.0]
    entropy = float(-(positive * np.log(positive)).sum(dtype=np.float64))

    cand_mask = np.zeros_like(legal, dtype=np.bool_)
    cand_count = 0
    if candidates is not None:
        cand_set = {int(a) for a in candidates}
        cand_count = len(cand_set)
        if cand_count > 0:
            cand_mask = np.array([int(a) in cand_set for a in legal], dtype=np.bool_)
    cand_mass = float(legal_p[cand_mask].sum(dtype=np.float64)) if cand_count > 0 else 0.0
    non_cand_top = (
        float(legal_p[~cand_mask].max(initial=0.0)) if cand_count > 0 else 0.0
    )

    return {
        "top_prob": top_prob,
        "action_prob": max(0.0, float(action_prob)),
        "entropy": entropy,
        "eff_moves": float(np.exp(entropy)),
        "candidate_mass": cand_mass,
        "non_candidate_top_prob": non_cand_top,
        "argmax_is_candidate": 1.0 if (cand_count > 0 and bool(cand_mask[top_local])) else 0.0,
        "argmax_is_action": 1.0 if top_action == int(action) else 0.0,
        "legal_count": float(legal.size),
        "candidate_count": float(cand_count),
    }


def _masked_priors(pol_logits: np.ndarray, board: chess.Board) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (full_priors, mask, legal_indices)."""
    legal_idx = legal_move_indices(board)
    if legal_idx.size == 0:
        return (np.zeros(POLICY_SIZE, dtype=np.float64),
                np.zeros(POLICY_SIZE, dtype=np.bool_),
                legal_idx)
  # Compute softmax only over legal moves
    legal_logits = pol_logits[legal_idx].astype(np.float64)
    legal_logits -= legal_logits.max()
    e = np.exp(legal_logits)
    s = float(e.sum())
    legal_priors = (e / s) if s > 0 else np.full_like(e, 1.0 / e.size)
  # Scatter into full-size arrays
    pri = np.zeros(POLICY_SIZE, dtype=np.float64)
    pri[legal_idx] = legal_priors
    mask = np.zeros(POLICY_SIZE, dtype=np.bool_)
    mask[legal_idx] = True
    return pri, mask, legal_idx


def _sigma_scale(*, max_visit: int, cfg: GumbelConfig) -> float:
    return float(cfg.c_scale) * (float(cfg.c_visit) + float(max_visit))


def _root_sigma_scale(*, max_visit: int, cfg: GumbelConfig) -> float:
    """Root-halving sigma(q) scale — mirrors the C root site exactly.

    Must match ``_mcts_tree.c`` ``gss_score_and_halve`` so the final improved
    policy reconstructed at the searched root uses the SAME q_scale the C
    halving loop used to eliminate candidates. Otherwise ``probs_out`` and any
    ``temperature > 0`` sampling (e.g. ``arena_standard`` at 0.1) are scored
    with the legacy linear transform while the played move came from the
    root-log transform — two different search shapes for one PLAY config.

    Falls back to the descent knobs when the root-only ones are unset
    (``c_visit_root``/``c_scale_root`` < 0), matching the C construction-time
    resolution.

    ⚑ ``q_visit_exp_root >= 90`` resolves to the literal exponent ``1.0``. It
    used to resolve to ``cfg.q_visit_exp``, one of the three deleted descent
    knobs, whose default was exactly 1.0 -- so this is the same arithmetic with
    the dead branch removed, not a behaviour change. The C keeps the same
    resolution (``q_visit_exp_root < 90 ? q_visit_exp_root : q_visit_exp``)
    because ``gumbel_c`` passes it the literal 1.0 for ``q_visit_exp``.
    The deleted ``q_visit_floor`` branch (``floor + csr*mv_term`` when >= 0)
    was likewise unreachable at its -1.0 default.
    """
    cvr = float(cfg.c_visit_root) if float(cfg.c_visit_root) >= 0.0 else float(cfg.c_visit)
    csr = float(cfg.c_scale_root) if float(cfg.c_scale_root) >= 0.0 else float(cfg.c_scale)
    qer = float(cfg.q_visit_exp_root) if float(cfg.q_visit_exp_root) < 90.0 else 1.0
    mv = float(max_visit)
    if qer < 0.0:
        return csr * math.log1p(cvr + mv)
    mv_term = mv if qer == 1.0 else mv**qer
    return csr * (cvr + mv_term)


def volatility_search_enabled(cfg: GumbelConfig) -> bool:
    """True when any volatility-aware search mechanism is switched on."""
    return float(cfg.volatility_q_scale) != 0.0 or float(cfg.volatility_fpu) != 0.0


_volatility_python_path_warned = False


def warn_volatility_python_path() -> None:
    """Warn once per process: volatility flags force the Python search path.

    The C fast path (mcts/gumbel_c.py) does not implement
    volatility_q_scale/volatility_fpu; callers that would normally take it
    must drop to run_gumbel_root_many and say so in the log. Porting to C is
    a follow-up gated on the arena clearing the Elo bar.
    """
    global _volatility_python_path_warned
    if not _volatility_python_path_warned:
        _volatility_python_path_warned = True
        import logging

        logging.getLogger("chess_anti_engine.mcts").warning(
            "volatility-aware Gumbel search enabled: forcing the (slower) "
            "Python search path - the C fast path does not implement "
            "volatility_q_scale/volatility_fpu. Use matched_sims, not "
            "matched_time, when comparing against the C path."
        )


def _volatility_sigma_factor(vol: float, cfg: GumbelConfig) -> float:
    """Multiplier on the sigma(q) scale for a node with predicted ``vol``.

    For positive ``volatility_q_scale``: 1.0 when the mechanism is off or
    vol equals the anchor; <1 (flatter) above the anchor, >1 (sharper)
    below it. A NEGATIVE exponent deliberately inverts that mapping (high
    volatility -> sharper) — only use it to sweep the opposite hypothesis.
    Clipped so a wild head output cannot collapse or explode the transform.
    """
    k = float(cfg.volatility_q_scale)
    if k == 0.0:
        return 1.0
    anchor = max(1e-9, float(cfg.volatility_anchor))
    ratio = anchor / max(1e-9, float(vol))
    clip = max(1.0, float(cfg.volatility_factor_clip))
    return float(np.clip(ratio ** k, 1.0 / clip, clip))


def _volatility_fpu_penalty(vol: float, cfg: GumbelConfig) -> float:
    """Pessimistic unvisited-child value offset (Q units, subtracted)."""
    k = float(cfg.volatility_fpu)
    return k * float(vol) if k != 0.0 else 0.0


def _completed_q_transform(
    *,
    actions: list[int] | np.ndarray,
    priors: np.ndarray,
    visits: np.ndarray,
    qvalues: np.ndarray,
    raw_value: float,
    cfg: GumbelConfig,
    epsilon: float = 1e-8,
    sigma_factor: float = 1.0,
    fpu_penalty: float = 0.0,
    root: bool = False,
    max_visit_cap: int = 0,
) -> np.ndarray:
    """DeepMind mctx completed-by-mix-value Q transform for Gumbel scores.

    ``sigma_factor`` scales the sigma(q) constant (volatility_q_scale) and
    ``fpu_penalty`` is subtracted from the unvisited-children mix value
    (volatility_fpu). Both default to the exact legacy behavior.

    ``max_visit_cap`` > 0 clamps the ``max_visit`` that feeds sigma(q). Used
    only by the stored-target arm of the improved policy (see
    ``GumbelConfig.target_max_visit_cap``); 0 = no clamp = legacy.

    ⚑ It shrinks SIGMA ONLY -- it is not "the transform a smaller search would
    have produced", and must not be described that way. The completed-Q vector
    still comes from the full search: the visit counts, the visit-weighted
    ``mixed_value`` and therefore the value imputed to every UNVISITED child are
    all untouched. The two coincide exactly only when every child is visited, so
    the mix-value branch is untaken -- which at production's sims/legal-move
    ratio is the exception, not the rule (with two unvisited children out of
    five the capped and genuinely-shallow transforms differ by ~0.03 in Q-scale
    units). The honest statement is the one in
    ``GumbelConfig.target_max_visit_cap``: search deep, TRUST the result like a
    shallow search.
    """
    actions_arr = np.asarray(actions, dtype=np.int64)
    visits_f = np.asarray(visits, dtype=np.float64)
    q = np.asarray(qvalues, dtype=np.float64)
    prior = np.maximum(np.asarray(priors, dtype=np.float64), np.finfo(np.float64).tiny)

    if actions_arr.size == 0:
        return np.zeros((0,), dtype=np.float64)

    visited = visits_f > 0.0
    sum_visits = float(visits_f.sum(dtype=np.float64))
    sum_probs = float(prior[visited].sum(dtype=np.float64)) if visited.any() else 0.0
    if sum_probs > 0.0 and np.isfinite(sum_probs):
        weighted_q = float((prior[visited] * q[visited] / sum_probs).sum(dtype=np.float64))
    else:
        weighted_q = float(raw_value)
    mixed_value = (float(raw_value) + sum_visits * weighted_q) / (sum_visits + 1.0)

    completed = np.where(visited, q, mixed_value - float(fpu_penalty))
    min_q = float(completed.min())
    max_q = float(completed.max())
    completed = (completed - min_q) / max(max_q - min_q, float(epsilon))
    max_visit = int(visits_f.max(initial=0.0))
    if max_visit_cap > 0 and max_visit > max_visit_cap:
        max_visit = int(max_visit_cap)
    scale = (
        _root_sigma_scale(max_visit=max_visit, cfg=cfg)
        if root
        else _sigma_scale(max_visit=max_visit, cfg=cfg)
    )
    return float(sigma_factor) * scale * completed


def _completed_q(*, root_q: float, root: Node, action: int) -> float:
    child = root.children.get(int(action))
    if child is None or child.N <= 0:
        return float(root_q)
    return float(-child.Q)


def _improved_policy_probs(
    *,
    node: Node,
    cfg: GumbelConfig,
) -> tuple[list[int], np.ndarray]:
    children = node.children
    actions = list(children.keys())
    if not actions:
        return [], np.zeros((0,), dtype=np.float64)

    n_act = len(actions)
    logits = np.empty(n_act, dtype=np.float64)
    visits = np.empty(n_act, dtype=np.float64)
    qvalues = np.empty(n_act, dtype=np.float64)
    priors = np.empty(n_act, dtype=np.float64)
    v_pi = node.Q

    for i, a in enumerate(actions):
        ch = children[a]
        n = ch.N
        priors[i] = max(ch.prior, 1e-12)
        logits[i] = math.log(priors[i])
        visits[i] = float(n)
        qvalues[i] = (-ch.W / n) if n > 0 else v_pi

    q_logits = _completed_q_transform(
        actions=actions,
        priors=priors,
        visits=visits,
        qvalues=qvalues,
        raw_value=float(v_pi),
        cfg=cfg,
        sigma_factor=_volatility_sigma_factor(node.vol, cfg),
        fpu_penalty=_volatility_fpu_penalty(node.vol, cfg),
    )
    probs = _softmax(logits + q_logits)
    return actions, probs


def _select_full_gumbel_child(node: Node, *, cfg: GumbelConfig) -> tuple[int, Node]:
    children = node.children
    actions, probs = _improved_policy_probs(node=node, cfg=cfg)
    if not actions:
        raise ValueError("Cannot select from an unexpanded node with no children")

    total_visits = 0
    for a in actions:
        total_visits += children[a].N
    inv_total = 1.0 / float(1 + total_visits)

    best_idx = 0
    best_score = -1e30
    for i, a in enumerate(actions):
        score = float(probs[i]) - float(children[a].N) * inv_total
        if score > best_score:
            best_score = score
            best_idx = i
    a = int(actions[best_idx])
    return a, children[a]


def _init_root_from_logits(
    board: chess.Board,
    *,
    pol_logits: np.ndarray,
    root_q: float,
) -> tuple[Node, np.ndarray, np.ndarray]:
    root = Node(board.copy(stack=True), parent=None, prior=1.0)
    if root.board.is_game_over():
        root.N = 1
        root.W = _terminal_value(root.board)
        zeros = np.zeros((POLICY_SIZE,), dtype=np.float64)
        return root, zeros, zeros.astype(np.bool_)
    pri, mask, legal_idx = _masked_priors(pol_logits, root.board)
    if legal_idx.size > 0:
        _expand_sparse(root, legal_idx, pri[legal_idx])
    root.N = 1
    root.W = float(root_q)
    return root, pri, mask


def _collect_forced_leaf(
    *,
    root: Node,
    forced_action: int,
    cfg: GumbelConfig,
) -> tuple[Node | None, list[Node], float | None]:
    child = root.children.get(int(forced_action))
    if child is None:
        return None, [root], float(root.Q)

    node = child
    path = [root, child]
    while node.expanded and node.children:
  # Unconditionally the Gumbel descent: `cfg.full_tree` defaults True and
  # nothing in the tree ever sets it false, so the PUCT arm that used to
  # live here was dead code advertising c_puct/fpu_reduction as live
  # Gumbel knobs (play-path audit 2026-08-03, F12). The PUCT descent is
  # reached through mcts/puct*.py and uci/, not from here.
        _, node = _select_full_gumbel_child(node, cfg=cfg)
        path.append(node)
  # Expanded nodes with children are never terminal — skip is_game_over()
  # here. Terminal detection happens after the loop exits.

    if node.board.is_game_over():
        return None, path, _terminal_value(node.board)
    return node, path, None


def _eval_with_optional_volatility(
    eval_impl: BatchEvaluator,
    xs: np.ndarray,
    *,
    relations: np.ndarray | None,
    cfg: GumbelConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Evaluate a batch; include the volatility head when the search needs it.

    Volatility-aware search is Python-path-only and requires an evaluator
    exposing ``evaluate_encoded_with_volatility`` (LocalModelEvaluator) —
    fail loud rather than silently searching with vol=0 everywhere.
    """
    if not volatility_search_enabled(cfg):
        if relations is not None:
            pol, wdl = eval_impl.evaluate_encoded(xs, relations=relations)
        else:
            pol, wdl = eval_impl.evaluate_encoded(xs)
        return pol, wdl, None
    fn = getattr(eval_impl, "evaluate_encoded_with_volatility", None)
    if fn is None:
        raise ValueError(
            "volatility-aware Gumbel search needs an evaluator with "
            "evaluate_encoded_with_volatility (LocalModelEvaluator); got "
            f"{type(eval_impl).__name__}"
        )
    if relations is not None:
        return fn(xs, relations=relations)
    return fn(xs)


def _resolve_root_logits(
    boards: list[chess.Board],
    *,
    model: torch.nn.Module | None,
    evaluator: BatchEvaluator | None,
    device: str,
    cfg: GumbelConfig,
    pre_pol_logits: np.ndarray | None,
    pre_wdl_logits: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, BatchEvaluator | None, np.ndarray | None]:
    """Phase 1: root pol/wdl logits (+ volatility) + a leaf evaluator.

    Reuses caller-provided ``pre_*_logits`` (one forward pass saved per ply
    when called from selfplay). Always resolves a ``leaf_eval`` for use by
    sequential halving — even when pre-logits short-circuit phase 1.
    With volatility-aware search enabled the pre-logits shortcut is skipped
    (they don't carry the volatility head) and the roots are re-evaluated.
    """
    vol_on = volatility_search_enabled(cfg)
    if pre_pol_logits is not None and pre_wdl_logits is not None and not vol_on:
        pol = _policy_logits_to_full(pre_pol_logits, cfg=cfg)
        wdl = np.asarray(pre_wdl_logits, dtype=np.float32)
        leaf_eval = evaluator if evaluator is not None else (
            LocalModelEvaluator(model, device=device) if model is not None else None
        )
        return pol, wdl, leaf_eval, None

    eval_impl = evaluator
    if eval_impl is None:
        if model is None:
            raise ValueError("run_gumbel_root_many requires model or evaluator")
        eval_impl = LocalModelEvaluator(model, device=device)
    xs = encode_positions_batch(
        boards,
        add_features=True,
        input_history_encoding=cfg.input_history_encoding,
        input_extra_features=cfg.input_extra_features,
    )
    rel = (
        np.stack([relation_matrices(b) for b in boards], axis=0)
        if cfg.compute_relations else None
    )
    pol, wdl, vol = _eval_with_optional_volatility(eval_impl, xs, relations=rel, cfg=cfg)
    pol = _policy_logits_to_full(pol, cfg=cfg)
    return pol, wdl, eval_impl, vol


def _evaluate_and_backprop_leaves(
    leaf_nodes: list[Node],
    leaf_paths: list[list[Node]],
    leaf_eval: BatchEvaluator | None,
    cfg: GumbelConfig,
) -> None:
    """Batched leaf NN eval + expand + backprop. No-op when ``leaf_nodes`` is empty."""
    if not leaf_nodes:
        return
    if leaf_eval is None:
        raise ValueError("run_gumbel_root_many requires model or evaluator")
    leaf_xs = encode_positions_batch(
        [node.board for node in leaf_nodes],
        add_features=True,
        input_history_encoding=cfg.input_history_encoding,
        input_extra_features=cfg.input_extra_features,
    )
    leaf_rel = (
        np.stack([relation_matrices(node.board) for node in leaf_nodes], axis=0)
        if cfg.compute_relations else None
    )
    pol_logits_leaf, wdl_logits_leaf, vol_leaf = _eval_with_optional_volatility(
        leaf_eval, leaf_xs, relations=leaf_rel, cfg=cfg,
    )
    pol_logits_leaf = _policy_logits_to_full(pol_logits_leaf, cfg=cfg)
    for li, (node, path, pol_logits, wdl_logits) in enumerate(zip(
        leaf_nodes, leaf_paths, pol_logits_leaf, wdl_logits_leaf, strict=True,
    )):
        if vol_leaf is not None:
            node.vol = float(vol_leaf[li])
        pri, _, legal_idx = _masked_priors(pol_logits, node.board)
        if legal_idx.size > 0:
            _expand_sparse(node, legal_idx, pri[legal_idx])
        _backprop(path, _wdl_to_q(wdl_logits.reshape(-1)))


def _select_top_m_with_gumbel(
    *,
    legal: np.ndarray,
    pri: np.ndarray,
    sim_budget: int,
    topk: int,
    add_noise: bool,
    gumbel_scale: float,
    rng: np.random.Generator,
) -> tuple[list[int], dict[int, float]]:
    """Sample top-m root actions via Gumbel(logit + noise). Caller filters trivial cases.

    Returns (cands, gumbels_for_all_legal). ``m`` is bounded so sequential halving
    can still allocate ≥1 visit per action per round.
    """
    log_pri = np.log(np.maximum(pri[legal], 1e-12))
    scale = float(gumbel_scale) if add_noise else 0.0
    g = scale * _gumbel(rng, legal.size) if scale > 0.0 else np.zeros(legal.size, dtype=np.float64)
    score: np.ndarray = g + log_pri

    if sim_budget <= 1:
        m = 1
    else:
        m_cap = max(2, (sim_budget + 1) // 2)
        m = max(2, int(min(int(topk), int(legal.size), int(m_cap))))

    kth = min(m - 1, int(score.size) - 1)
    top_idx = np.argpartition(-score, kth)[:m]
    cands = legal[top_idx].astype(int).tolist()
    gumbels = {int(a): float(gg) for a, gg in zip(legal.tolist(), g.tolist(), strict=True)}
    return cands, gumbels


@dataclass
class _BoardSearchState:
    """Per-board state for sequential halving. ``finished`` short-circuits halving."""
    root: Node
    priors: np.ndarray
    candidates: list[int] | None
    remaining: list[int] | None
    gumbels: dict[int, float] | None
    finished_probs: np.ndarray | None
    finished_action: int | None
    finished_value: float | None


def _init_board_search_state(
    board: chess.Board,
    *,
    pol_logits: np.ndarray,
    root_q: float,
    sim_budget: int,
    cfg: GumbelConfig,
    rng: np.random.Generator,
) -> _BoardSearchState:
    """Phase 2 per-board: init root, early-exit trivial cases, else select top-m."""
    root, pri, mask = _init_root_from_logits(board, pol_logits=pol_logits, root_q=root_q)

    def _finish(probs: np.ndarray, action: int, value: float) -> _BoardSearchState:
        return _BoardSearchState(
            root=root, priors=pri, candidates=None, remaining=None, gumbels=None,
            finished_probs=probs, finished_action=action, finished_value=value,
        )

    if root.board.is_game_over():
        return _finish(np.zeros((POLICY_SIZE,), dtype=np.float32), 0, float(root.Q))

    legal = np.nonzero(mask)[0]
    if legal.size == 0:
        return _finish(np.zeros((POLICY_SIZE,), dtype=np.float32), 0, root_q)

    mate = immediate_mate_root_policy(board)
    if mate is not None:
        probs, action, value = mate
        return _finish(probs, action, value)

    if root_q > 0.0 and legal.size > 1:
        terminal_draws = immediate_terminal_draw_indices(board)
        if terminal_draws:
            keep = np.array([int(a) not in terminal_draws for a in legal], dtype=np.bool_)
            if keep.any():
                pri[np.fromiter(terminal_draws, dtype=np.int64)] = 0.0
                legal = legal[keep]

    if legal.size == 1:
        a0 = int(legal[0])
        p = np.zeros((POLICY_SIZE,), dtype=np.float32)
        p[a0] = 1.0
        return _finish(p, a0, root_q)

    cands, gumbels = _select_top_m_with_gumbel(
        legal=legal, pri=pri, sim_budget=sim_budget,
        topk=int(cfg.topk), add_noise=cfg.add_noise,
        gumbel_scale=float(cfg.gumbel_scale), rng=rng,
    )
    return _BoardSearchState(
        root=root, priors=pri,
        candidates=cands, remaining=list(cands), gumbels=gumbels,
        finished_probs=None, finished_action=None, finished_value=None,
    )


def _collect_forced_leaves_round(
    *,
    active: list[int],
    states: list[_BoardSearchState],
    visits_per_action: dict[int, int],
    rep: int,
    cfg: GumbelConfig,
) -> tuple[list[Node], list[list[Node]]]:
    """One sequential-halving round: walk forced lines from root, collect non-terminal leaves.

    Backprops terminal-value paths immediately; returns leaves that need NN eval.
    """
    leaf_nodes: list[Node] = []
    leaf_paths: list[list[Node]] = []
    for bi in active:
        st = states[bi]
        rem = st.remaining
        if rem is None or rep >= visits_per_action[bi]:
            continue
        for action in rem:
            leaf, path, terminal_value = _collect_forced_leaf(
                root=st.root, forced_action=int(action), cfg=cfg,
            )
            if terminal_value is not None:
                _backprop(path, float(terminal_value))
            elif leaf is not None:
                leaf_nodes.append(leaf)
                leaf_paths.append(path)
    return leaf_nodes, leaf_paths


# --- pure halving-schedule arithmetic (mirrors _mcts_tree.c) ------------------
#
# ⚑ These lived in `uci/root_parallel_gumbel.py`, which meant the Python
# reference search (`run_gumbel_root_many`, below) had its own hardcoded copy of
# the div-2 schedule and never read `cfg.halving_div` at all. That path is not
# hypothetical: `selfplay.match.pick_moves_for_boards` routes to it whenever
# volatility search is on or the C extension is missing, so
# `--cand-gumbel halving_div=4 --volatility-q-scale 0.5` validated, banked 4 and
# searched at 2 -- this PR's own defect, inside the fix for it. One
# implementation now, imported by both.


def halving_rounds_left(n_candidates: int, halving_div: int = 2) -> int:
    """Ceil-divisions of ``n_candidates`` by ``halving_div`` down to 1.

    Mirrors the ``while (tmp > 1) { rounds_left++; tmp = (tmp+div-1)/div; }``
    loop in ``gss_begin_round``.
    """
    div = max(MIN_HALVING_DIV, int(halving_div))
    rounds = 0
    tmp = int(n_candidates)
    while tmp > 1:
        rounds += 1
        tmp = (tmp + div - 1) // div
    return rounds


def halving_visits_per_action(
    n_candidates: int, budget_remaining: int, halving_div: int = 2,
) -> int:
    """Per-candidate sims for the current round (``gss_begin_round``)."""
    if n_candidates <= 1:
        return int(budget_remaining)
    rounds_left = halving_rounds_left(n_candidates, halving_div)
    return max(1, int(budget_remaining) // (int(n_candidates) * rounds_left))


def halving_keep_count(n_candidates: int, halving_div: int = 2) -> int:
    """Survivors after one halving round (``gss_score_and_halve``)."""
    div = max(MIN_HALVING_DIV, int(halving_div))
    keep = (int(n_candidates) + div - 1) // div
    return max(1, min(keep, int(n_candidates) - 1))


def _halve_remaining_for_board(
    st: _BoardSearchState,
    *,
    root_q: float,
    cfg: GumbelConfig,
) -> None:
    """Re-rank ``st.remaining`` by completed-Q and halve it. No-op when ≤1 candidate left."""
    rem = st.remaining
    if rem is None or st.gumbels is None or len(rem) <= 1:
        return
    pri = st.priors
    gmap = st.gumbels
    root = st.root
    legal = np.nonzero(pri > 0)[0].astype(int)
    visits = np.empty(legal.size, dtype=np.float64)
    qvalues = np.empty(legal.size, dtype=np.float64)
    for i, a in enumerate(legal):
        ch = root.children.get(int(a))
        n = 0 if ch is None else int(ch.N)
        visits[i] = float(n)
        qvalues[i] = (-ch.W / n) if (ch is not None and n > 0) else float(root_q)
    q_logits = _completed_q_transform(
        actions=legal,
        priors=pri[legal],
        visits=visits,
        qvalues=qvalues,
        raw_value=float(root_q),
        cfg=cfg,
        sigma_factor=_volatility_sigma_factor(root.vol, cfg),
        fpu_penalty=_volatility_fpu_penalty(root.vol, cfg),
        root=True,
    )
    q_by_action = {int(a): float(q_logits[i]) for i, a in enumerate(legal.tolist())}
    rem.sort(
        key=lambda a: (
            float(gmap.get(int(a), 0.0))
            + float(np.log(max(float(pri[int(a)]), 1e-12)))
            + q_by_action.get(int(a), 0.0)
        ),
        reverse=True,
    )
    st.remaining = rem[: halving_keep_count(len(rem), int(cfg.halving_div))]


def _build_improved_policy_for_board(
    st: _BoardSearchState,
    *,
    root_q: float,
    cfg: GumbelConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, float]:
    """Phase 4: completed-Q policy improvement at the searched root + temperature sample."""
    root = st.root
    pri = st.priors
    remaining = st.remaining
    if remaining is None or st.candidates is None:
        return np.zeros((POLICY_SIZE,), dtype=np.float32), 0, root_q

    legal = np.nonzero(pri > 0)[0].astype(int)
    visits = np.empty(legal.size, dtype=np.float64)
    qvalues = np.empty(legal.size, dtype=np.float64)
    for i, a in enumerate(legal):
        ch = root.children.get(int(a))
        n = 0 if ch is None else int(ch.N)
        visits[i] = float(n)
        qvalues[i] = (-ch.W / n) if (ch is not None and n > 0) else float(root_q)
    log_prior = np.log(np.maximum(pri[legal], 1e-12))
    sigma_factor = _volatility_sigma_factor(root.vol, cfg)
    fpu_penalty = _volatility_fpu_penalty(root.vol, cfg)

  # imp_all drives the PLAYED move (the temperature sample); imp_store is what
  # is written to the shard. They are the SAME object unless one of the two
  # target-only knobs is set, so the off path computes exactly once, as before.
  # Kept parallel to the C path's copy in gumbel_c.py -- change both together.
    q_play = _completed_q_transform(
        actions=legal,
        priors=pri[legal],
        visits=visits,
        qvalues=qvalues,
        raw_value=float(root_q),
        cfg=cfg,
        sigma_factor=sigma_factor,
        fpu_penalty=fpu_penalty,
        root=True,
    )
    imp_all = _softmax(log_prior + q_play)
  # Two independent target-only adjustments, one to each term of the same
  # softmax: `target_max_visit_cap` shrinks sigma on the Q term,
  # `target_untempered_prior` undoes policy_temp on the prior term. Either off
  # arm reuses the play-side value rather than recomputing an identical one, so
  # turning on only one of them costs only that one.
    target_cap = int(cfg.target_max_visit_cap)
    log_prior_store = target_log_prior(log_prior, cfg=cfg)
    if target_cap <= 0 and log_prior_store is log_prior:
        imp_store = imp_all
    else:
        q_store = q_play if target_cap <= 0 else _completed_q_transform(
            actions=legal,
            priors=pri[legal],
            visits=visits,
            qvalues=qvalues,
            raw_value=float(root_q),
            cfg=cfg,
            sigma_factor=sigma_factor,
            fpu_penalty=fpu_penalty,
            root=True,
            max_visit_cap=target_cap,
        )
        imp_store = _softmax(log_prior_store + q_store)
    probs = np.zeros((POLICY_SIZE,), dtype=np.float32)
    probs[legal] = imp_store.astype(np.float32)

    best_a = int(remaining[0])
  # Gumbel sequential halving leaves the survivor at remaining[0]; map
  # that back to its position in the full ``legal`` array (= imp_all).
    argmax_idx = int(np.searchsorted(legal, best_a)) if legal.size > 0 else 0
    action = sample_action_with_temperature(
        rng, legal, imp_all, float(cfg.temperature), argmax_idx=argmax_idx,
    )
    value = _completed_q(root_q=root_q, root=root, action=best_a)
    return probs, action, value


@torch.no_grad()
@overload
def run_gumbel_root_many(
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
    return_diagnostics: Literal[False] = False,
) -> GumbelManyResult: ...


@overload
def run_gumbel_root_many(
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
    return_diagnostics: Literal[True],
) -> GumbelManyDiagnosticsResult: ...


def run_gumbel_root_many(
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
    return_diagnostics: bool = False,
) -> GumbelManyResult | GumbelManyDiagnosticsResult:
    """Root Gumbel search with sequential halving.

    This follows the paper's root-search structure much more closely than the
    previous shallow approximation:
      1. Evaluate the root once for priors + root value.
      2. Sample top-m root actions using gumbel(log π(a)).
      3. Allocate actual subtree simulations to those candidates via
         sequential halving, forcing each simulation through a chosen root
         action and then using ordinary tree search below the root.
      4. Build the returned policy from completed-Q policy improvement on the
         searched root.

    Returns (probs_list, actions, root_values), where root_values are the best
    searched child values from the root perspective.
    """
    n_boards = len(boards)
    if n_boards == 0:
        if return_diagnostics:
            return [], [], [], [], []
        return [], [], [], []

    sim_budget = max(1, int(cfg.simulations))
    if per_game_simulations is not None:
        if len(per_game_simulations) != n_boards:
            raise ValueError(
                f"per_game_simulations length {len(per_game_simulations)} does not match "
                f"boards length {n_boards}"
            )
        budget_remaining = [max(1, int(v)) for v in per_game_simulations]
    else:
        budget_remaining = [sim_budget] * n_boards

  # ── 1. Batch root evaluation + resolve leaf evaluator for phase 3 ────────
    pol_logits_batch, wdl_logits_batch, leaf_eval, root_vols = _resolve_root_logits(
        boards,
        model=model, evaluator=evaluator, device=device,
        cfg=cfg,
        pre_pol_logits=pre_pol_logits, pre_wdl_logits=pre_wdl_logits,
    )
    root_qs = [_wdl_to_q(wdl_logits_batch[i]) for i in range(n_boards)]

  # ── 2. Per-board root init + Gumbel candidate selection ──────────────────
    states: list[_BoardSearchState] = []
    for i, b in enumerate(boards):
        board_cfg = replace(
            cfg,
            add_noise=bool(per_game_add_noise[i]) if per_game_add_noise is not None else bool(cfg.add_noise),
            gumbel_scale=(
                float(per_game_gumbel_scale[i])
                if per_game_gumbel_scale is not None
                else float(cfg.gumbel_scale)
            ),
        )
        st = _init_board_search_state(
            b,
            pol_logits=pol_logits_batch[i],
            root_q=float(root_qs[i]),
            sim_budget=budget_remaining[i],
            cfg=board_cfg,
            rng=rng,
        )
        if root_vols is not None:
            st.root.vol = float(root_vols[i])
        states.append(st)

  # ── 3. Sequential halving with real subtree simulations ──────────────────
    while True:
        active = [
            i for i, st in enumerate(states)
            if st.finished_probs is None
            and st.remaining is not None
            and len(st.remaining) >= 1
            and budget_remaining[i] > 0
        ]
        if not active:
            break

        visits_per_action: dict[int, int] = {}
        for bi in active:
            rem = states[bi].remaining
            assert rem is not None
            visits_per_action[bi] = halving_visits_per_action(
                len(rem), int(budget_remaining[bi]), int(cfg.halving_div),
            )

        max_reps = max(visits_per_action.values(), default=0)
        for rep in range(max_reps):
            leaf_nodes, leaf_paths = _collect_forced_leaves_round(
                active=active, states=states,
                visits_per_action=visits_per_action, rep=rep, cfg=cfg,
            )
            _evaluate_and_backprop_leaves(leaf_nodes, leaf_paths, leaf_eval, cfg)

        for bi in active:
            st = states[bi]
            rem = st.remaining
            if rem is None:
                continue
            budget_remaining[bi] = max(
                0, int(budget_remaining[bi] - visits_per_action[bi] * len(rem)),
            )
            _halve_remaining_for_board(st, root_q=float(root_qs[bi]), cfg=cfg)

  # ── 4. Build improved policies + legal masks ─────────────────────────────
    probs_out: list[np.ndarray] = []
    actions_out: list[int] = []
    values_out: list[float] = []
    legal_masks_out: list[np.ndarray] = []
    diagnostics_out: list[dict[str, float] | None] = []
    for i, st in enumerate(states):
        if st.finished_probs is not None:
            probs_out.append(st.finished_probs)
            actions_out.append(int(st.finished_action or 0))
            values_out.append(float(st.finished_value if st.finished_value is not None else root_qs[i]))
            diagnostics_out.append(None)
        else:
            probs, action, value = _build_improved_policy_for_board(
                st, root_q=float(root_qs[i]), cfg=cfg, rng=rng,
            )
            probs_out.append(probs)
            actions_out.append(action)
            values_out.append(value)
            legal = np.nonzero(st.priors > 0)[0].astype(int)
            diagnostics_out.append(gumbel_policy_diagnostics(
                probs=probs, action=int(action), legal=legal, candidates=st.candidates,
            ))

        mask = np.zeros((POLICY_SIZE,), dtype=np.bool_)
        for a in st.root.children:
            mask[a] = True
        legal_masks_out.append(mask)

    if return_diagnostics:
        return probs_out, actions_out, values_out, legal_masks_out, diagnostics_out
    return probs_out, actions_out, values_out, legal_masks_out


@torch.no_grad()
def run_gumbel_root(
    model: torch.nn.Module,
    board: chess.Board,
    *,
    device: str,
    rng: np.random.Generator,
    cfg: GumbelConfig,
) -> tuple[np.ndarray, int, float]:
    probs, acts, vals, _masks = run_gumbel_root_many(model, [board], device=device, rng=rng, cfg=cfg)
    return probs[0], acts[0], float(vals[0])
