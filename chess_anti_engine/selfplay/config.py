from __future__ import annotations

import logging
from dataclasses import dataclass

from chess_anti_engine.encoding.features import EXTRA_FEATURES_V2_THREATS
from chess_anti_engine.mcts.gumbel import DEFAULT_VOLATILITY_ANCHOR, SELFPLAY_GUMBEL_C_SCALE
from chess_anti_engine.encoding.lc0 import LC0_HISTORY_LEGACY
from chess_anti_engine.moves import MODEL_POLICY_ENCODING
from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpponentConfig:
    wdl_regret_limit: float | None = None


@dataclass(frozen=True)
class TemperatureConfig:
    temperature: float = 1.0
    drop_plies: int = 0
    after: float = 0.0
    decay_start_move: int = 20
    decay_moves: int = 60
    endgame: float = 0.6
    selfplay_temperature: float | None = None
    selfplay_decay_start_move: int | None = None
    selfplay_decay_moves: int | None = None
    selfplay_endgame: float | None = None


@dataclass(frozen=True)
class SearchConfig:
    simulations: int = 50
    mcts_type: str = "puct"
    playout_cap_fraction: float = 0.25
    # Research bet (default 0.0 = off, byte-identical iid schedule): fraction
    # of base full plies that FORCE the slot's next net turn full too. The
    # base rate is rescaled to playout_cap_fraction/(1+pair) so the TOTAL
    # full-ply fraction — and therefore search cost and SF label volume — is
    # unchanged; fulls just arrive in consecutive pairs. Why: the sf_p0
    # policy-teacher (one-ply label shift, selfplay rows) only exists where
    # ply t-1 is ALSO a full labeled ply — iid scheduling makes that
    # playout_cap_fraction (~25%) of full rows; pairing doubles teacher
    # coverage for free. 1.0 = every base full is paired (~2x coverage).
    full_ply_pair_fraction: float = 0.0
    fast_simulations: int = 8
    fpu_reduction: float = 1.2
    fpu_at_root: float = 1.0
    gumbel_topk: int = 16
    # Clamps max_visit when sizing sigma(q) for the STORED training target
    # only -- 'search deep, trust it like a shallow search'. The played move
    # keeps the uncapped sigma, so play strength is untouched and an arena
    # cannot see this knob. 0 = off and bit-identical.
    # Rationale: mcts.gumbel.GumbelConfig.target_max_visit_cap.
    gumbel_target_max_visit_cap: int = 0
    # lc0's PolicyTemperature for SELFPLAY search: the policy-head logits are
    # divided by this before they seed the tree, at the root AND at every leaf
    # (mcts.gumbel.apply_policy_temp, shared by the Python and C paths).
    # >1 softens the prior, <1 sharpens it, 1.0 = exact no-op.
    #
    # 1.0 is the production default and keeps the search bit-identical. It is
    # NOT free to move: policy_temp != 1.0 disables the compact-legal bf16 leaf
    # transport in gumbel_c (the C bf16 leaf softmax has no temperature hook,
    # so the alternative would be UNTEMPERED leaf priors), and with it the bf16
    # input transport. Price it on the broker path before raising it --- see
    # docs/experiment_ledger.md, "selfplay search policy temperature".
    gumbel_policy_temp: float = 1.0
    gumbel_c_scale: float = SELFPLAY_GUMBEL_C_SCALE
    gumbel_scale: float = 1.0
    gumbel_scale_after: float = 0.0
    gumbel_scale_decay_start_move: int = 0
    gumbel_scale_decay_moves: int = 0
    curriculum_gumbel_scale: float = 0.0
    curriculum_gumbel_scale_after: float = 0.0
    curriculum_gumbel_scale_decay_start_move: int = 0
    curriculum_gumbel_scale_decay_moves: int = 0
  # Volatility-aware Gumbel search (Python path only; see mcts/gumbel.py).
  # Any non-zero value forces the Python search path with a logged warning.
    volatility_q_scale: float = 0.0
    volatility_fpu: float = 0.0
    volatility_anchor: float = DEFAULT_VOLATILITY_ANCHOR
  # C-search batching (audit C17). Both default 0 = byte-identical to the
  # behaviour before they existed, which is also what production runs today:
  # selfplay passed NEITHER of these, so the C runner fell back to
  # GSS_GPU_BATCH=1024 leaves per batch with no virtual loss.
  #
  # Within one sequential-halving round the tree cannot update, so every visit
  # allocated to a candidate descends to the SAME leaf. Measured at the realized
  # per-thread shape (~24 concurrent games, production 25/75 sim mix): 56.4% of
  # rows sent to the GPU are byte-identical duplicates. The rate is a function of
  # leaves-per-board = 1024/boards-per-call, so it is only meaningful quoted with
  # a board count -- 0.565 @12, 0.564 @24, 0.608 @48.
  #
  # gumbel_vloss_weight > 0 makes an in-flight leaf unattractive to the next
  # walker, so a batch fills with DISTINCT leaves instead: 2.2-2.4x more distinct
  # positions for ~4% fewer rows at the same batch size. LEGACY (the only mode
  # selfplay uses) values the pending visit as a loss; that pessimism IS the
  # spreading mechanism, which is why VLOSS_MODE_VIRTUAL_MEAN -- theoretically
  # cleaner, no value bias -- cannot fill a batch and is not offered here.
  #
  # gumbel_target_batch = 1 also removes the duplicates, by flushing every rep,
  # but costs ~8x the GPU round trips. Kept only as a control arm.
    gumbel_target_batch: int = 0
    gumbel_vloss_weight: int = 0


@dataclass(frozen=True)
class DiffFocusConfig:
    enabled: bool = True
    q_weight: float = 6.0
    pol_scale: float = 3.5
    slope: float = 3.0
    min_keep: float = 0.025


@dataclass(frozen=True)
class GameConfig:
    max_plies: int = 240
    selfplay_fraction: float = 0.0
    # Optional node budget for blocking curriculum opponent moves. 0 means
    # use the main SF label/eval node budget.
    sf_move_nodes: int = 0
    # On fast-sim (playout-capped) plies, SF's per-slot node budget is scaled by
    # this factor (full-sim plies always use 1.0). INTENDED optimization, not a
    # weakened training target: SF labels (sf_policy/sf_wdl) only attach to full
    # plies (has_policy <=> is_full <=> last_net_full), which always get full
    # nodes -- so this never touches a training target. It only lets the
    # opponent play cheaply on the ~75% throwaway fast plies, saving SF compute.
    # Default 0.25 preserves the long-standing equilibrated behavior; raise
    # toward 1.0 only for a more consistently strong opponent (costs SF compute
    # on fast plies + a PID re-equilibration).
    sf_fast_ply_node_scale: float = 0.25
    # Research bet (default 0 = off): cap the node budget of LABEL-ONLY SF
    # queries (the selfplay P1 analysis queries) at this value, decoupling
    # label cost from the PID-controlled OPPONENT budget. The PID ramps
    # sf_nodes for opponent strength (winrate servo), but labels ride the same
    # number: at ~700k nodes nearly every recorded selfplay ply pays a ~700k
    # SF analysis, which dominates iteration wall time. The value-anchor
    # rationale assumes sf_wdl_use_cp_logistic=true: the cp-logistic label
    # (slope ~0.006) can barely distinguish 100k-node cp from 700k-node cp, so
    # the anchor loses ~nothing. With cp_logistic OFF, capped labels are SF's
    # native (sharper) WDL at the reduced depth — __post_init__ warns on that
    # combination because the load-bearing anchor silently degrades. The
    # MultiPV POLICY teachers (sf_policy/sf_p0/regret) do get a shallower
    # teacher — that's the tradeoff to watch (audit teacher regret). Applies
    # ONLY to label-only queries: curriculum opponent moves keep the full
    # budget, and curriculum label reuse (free full-strength move futures) is
    # untouched. See configs/exp_throughput_views.yaml.
    sf_label_nodes_cap: int = 0
  # Floor for LABEL-ONLY SF queries (default 0 = off), the cap's mirror: raise
  # label queries to at least this budget, decoupling label QUALITY upward
  # from the PID-controlled opponent budget. Why it exists: with both knobs at
  # 0 the teacher rides the PID difficulty knob — the 2026-08-04 fresh restart
  # started sf_nodes at 50k, silently cutting the label teacher 11x versus the
  # old lineage's realized ~698k (median depth 8 at MultiPV 40, mate-blind in
  # sharp positions). The floor pins teacher depth while the PID stays free to
  # ramp opponent difficulty from anywhere. Cost: label queries no longer get
  # cheaper when the PID lowers sf_nodes; at ~700k the labels dominate
  # iteration wall time (that is the old lineage's long-standing cost shape).
  # Applied after the cap, so on a conflicting config (floor > cap > 0) the
  # floor wins — __post_init__ warns. Label-escalation re-queries are
  # unaffected (they have their own budget). Curriculum move queries are never
  # floored.
    sf_label_nodes_floor: int = 0
  # Research bet (default 0.0 = off; provable no-op: the gate returns before
  # any engine call and no record field is ever written): adaptive escalation
  # of the SF label on net-vs-label disagreement. When
  # |net search root Q - label Q| >= this gap (harvester nq/sq units:
  # q = wdl[W] - wdl[L] in [-1, 1], record POV), the label query is re-run
  # cold-TT (ucinewgame) at sf_label_escalate_nodes and the deep result
  # REPLACES the recorded label (sf_wdl + sf_policy target + sparse rows +
  # meta). Why: deep-SF audits of harvested net-vs-label disagreements show
  # the ~700k-node label is WRONG in 70-81% of high-gap cases (deep SF sides
  # with the net) — escalation fixes that label noise at the source. The
  # ORIGINAL label is preserved on the record (sf_wdl_original) so the
  # blind-spot harvester still sees the original gap — otherwise escalation
  # would hide exactly the positions it mines. Pool (async) label path only;
  # the sync StockfishUCI fallback path ignores the flag. See
  # configs/exp_label_escalation.yaml.
    sf_label_escalate_q_gap: float = 0.0
  # Node budget for the escalated re-query. Never capped by
  # sf_label_nodes_cap and never touches the PID opponent budget.
    sf_label_escalate_nodes: int = 3_000_000
  # Hard per-game escalation cap — an escalated query costs ~10-30s of SF
  # CPU; uncapped it would crater selfplay throughput. First-N-over-threshold
  # (the streaming label path attaches results as they arrive, so a game's
  # gaps cannot be ranked up front; see _maybe_submit_label_escalation).
    sf_label_escalate_max_per_game: int = 2
    sf_policy_temp: float = 0.25
    sf_policy_label_smooth: float = 0.05
    sf_wdl_use_cp_logistic: bool = False
    sf_wdl_cp_slope: float = 0.010
    sf_wdl_cp_draw_width: float = 60.0
  # Candidate scoring for the SF POLICY target softmax (labels only — the
  # curriculum move-selection scores stay w+0.5d, the PID wdl_regret unit).
  # "wdl": w + 0.5*d, saturates in decisive positions (all moves ~1.0 when
  # clearly won, so the target cannot rank them). "cp": raw effective
  # centipawns (mate folded via mate_to_effective_cp), screened 07-31 +
  # conjugacy-audited 08-04 — decisive-position blunder mass roughly halved.
  # cp mode softmaxes at sf_policy_cp_temp (CENTIPAWN units; 16.2 matches the
  # screen's production-sharpness calibration, ~1.18 nats mean entropy) —
  # sf_policy_temp is win-fraction units and is not used in cp mode.
    sf_policy_score_mode: str = "wdl"
    sf_policy_cp_temp: float = 16.2
    soft_policy_temp: float = 2.0
    timeout_adjudication_threshold: float = 0.90
    volatility_source: str = "raw"
    syzygy_path: str | None = None
    # Optional lower-IO Syzygy path for ordinary high-volume Stockfish
    # searches. When set, selfplay can still route non-adjudicated <=6-piece
    # roots to syzygy_path for DTZ-correct play-throughs.
    stockfish_syzygy_path: str | None = None
    syzygy_rescore_policy: bool = False
  # If true, end the game as soon as the position becomes TB-eligible and
  # use the TB-proven WDL as the outcome. Saves the rest of the MCTS work
  # that would have been spent playing out a known-result endgame.
    syzygy_adjudicate: bool = False
  # Fraction of games that actually adjudicate when syzygy_adjudicate is
  # on. 1.0 = always (max compute savings). < 1.0 lets some games play
  # through the endgame so the NN keeps training on endgame samples with
  # its own labels — prevents it from silently losing endgame skill while
  # the TB does the work.
    syzygy_adjudicate_fraction: float = 1.0
  # If true, override leaf WDL logits during MCTS with TB truth (UCI-style
  # in-tree probing). Pinned Q values collapse noisy endgame search paths.
    syzygy_in_search: bool = False
    categorical_bins: int = DEFAULT_CATEGORICAL_BINS
    hlgauss_sigma: float = 0.04
    # Research bet (default 0.0 = off, byte-identical to the ternary target):
    # blend the categorical (HL-Gauss) value target toward SF's objective eval
    # expected score so the 32-bin distributional head trains on a continuous
    # value instead of three {-1,0,+1} spikes. See
    # configs/exp_categorical_continuous.yaml.
    categorical_blend_frac: float = 0.0
    # Optional 3rd blend source for the categorical target: the net's own search
    # WDL (mirrors the main value head's search_wdl_frac). 0.0 = SF-only blend.
    categorical_search_blend_frac: float = 0.0
    policy_encoding: str = MODEL_POLICY_ENCODING
    input_history_encoding: str = LC0_HISTORY_LEGACY
    input_extra_features: str = EXTRA_FEATURES_V2_THREATS
    record_lc0_root_input: bool = False
  # Gated candidate: correct the lc0-root per-slot repetition planes. The C
  # encoder otherwise under-reports repetitions older than the cleared
  # hash_stack window (see encoding/rep_fix.py). Changes net inputs, so it is
  # warm-start + arena gated — default off keeps the encoding byte-identical.
    history_rep_fix: bool = False
  # Compute + store dynamic board-relation matrices: search evals get them as
  # attention-bias input, and samples carry them into shards for training.
    record_relations: bool = False
  # Write the dense sf_policy_target alongside the sparse MultiPV rows.
  # Flip to False only once (a) the full replay window carries sparse labels
  # AND (b) training runs with train.sf_policy_sparse_ce: true — the sparse
  # CE is then the sole policy_sf supervision and the dense vector is dead
  # weight in the shard.
    record_dense_sf_policy: bool = True
  # Research bet: record the P0 own-move SF teacher (the prior full ply's
  # sf_policy) on eligible selfplay rows, so policy_own can be taught toward
  # SF's recommended move for its own position (not just its own MCTS visits).
  # Default off (adds a policy-sized shard field). Pairs with train.w_sf_own.
    record_sf_p0_policy: bool = False

  # Research bet: record the P0 own-move SF cp-regret vector (per-move
  # normalized regret at the net's own position, from the prior full ply's raw
  # MultiPV) on eligible selfplay rows, so policy_own can be trained to minimize
  # expected SF regret directly. Default off (policy-sized shard field). Pairs
  # with train.w_sf_own_regret.
    record_sf_p0_regret: bool = False

  # Research bet (default off): keep fast-ply (playout-capped, has_policy=False)
  # records as VALUE-ONLY training rows instead of dropping them in finalize.
  # This is the missing half of KataGo's playout-cap randomization: cheap plies
  # exist to generate more games per compute, and their positions still carry
  # valid outcome/search-WDL/moves_left targets (~4x value rows per game, zero
  # extra SF cost — fast plies never get SF label queries). Policy heads stay
  # masked out via has_policy=0 (losses.py already handles it). Mind the batch
  # policy-density dilution (~1:4) and the positions-denominated replay window
  # spanning ~4x fewer games. See configs/exp_throughput_views.yaml.
    record_fast_ply_value: bool = False

  # Inline blind-spot harvesting: if set, finished games append value-blind
  # positions (net's search says fine, in-loop SF says lost) as seed lines with
  # real history to this file (selfplay/blindspot_harvest.py). "" = off. Purely
  # a side output itself; the cadenced monitor later vets and promotes candidates
  # into opening_fen_list_path through the validated feed step.
    blindspot_harvest_out_path: str = ""

    def __post_init__(self) -> None:
        if int(self.sf_label_nodes_cap or 0) > 0 and not self.sf_wdl_use_cp_logistic:
            _LOG.warning(
                "sf_label_nodes_cap=%d with sf_wdl_use_cp_logistic=false: capped "
                "label queries produce SF's NATIVE WDL at the reduced depth, "
                "degrading the load-bearing sf_wdl anchor. The cap's cost-free "
                "rationale only holds for cp-logistic labels — enable "
                "sf_wdl_use_cp_logistic or drop the cap.",
                int(self.sf_label_nodes_cap),
            )
        cap = int(self.sf_label_nodes_cap or 0)
        floor = int(self.sf_label_nodes_floor or 0)
        if 0 < cap < floor:
            _LOG.warning(
                "sf_label_nodes_floor=%d exceeds sf_label_nodes_cap=%d: the "
                "floor is applied after the cap and wins, so the cap is "
                "effectively dead. Drop one of the two.",
                floor, cap,
            )
        # Hard error, not a warning: a typo'd mode would silently fall back to
        # "wdl" at every consumer (they all compare == "cp"), reverting the
        # target change while the config claims it is live.
        if self.sf_policy_score_mode not in ("wdl", "cp"):
            raise ValueError(
                f"sf_policy_score_mode must be 'wdl' or 'cp', got "
                f"{self.sf_policy_score_mode!r}"
            )
        if float(self.sf_policy_cp_temp) <= 0.0:
            raise ValueError(
                f"sf_policy_cp_temp must be > 0 (centipawns), got "
                f"{self.sf_policy_cp_temp!r}"
            )


__all__ = [
    "DiffFocusConfig",
    "GameConfig",
    "OpponentConfig",
    "SearchConfig",
    "TemperatureConfig",
]
