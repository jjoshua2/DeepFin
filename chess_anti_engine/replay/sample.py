"""ReplaySample — the single-position training record.

Leaf module: imported by both replay.buffer and replay.shard (keeping it here
avoids a buffer <-> shard import cycle).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ReplaySample:
    x: np.ndarray  # (C,8,8) float32
    policy_target: np.ndarray  # (POLICY_SIZE,) float32 distribution
    wdl_target: int  # 0/1/2

  # Sampling priority (KataGo-style surprise weighting)
    priority: float = 1.0
    priority_policy_kl: float | None = None
    priority_q_delta: float | None = None
    priority_sf_search_gap: float | None = None
    game_id: int | None = None
    ply_index: int | None = None
    seed_id: int | None = None
    seed_family_id: int | None = None
    opening_source_code: int | None = None
    has_policy: bool = True
    x_lc0_root: np.ndarray | None = None  # Optional alternate LC0-root input planes.
    relations: np.ndarray | None = None  # Optional (5,64,64) uint8 dynamic relation matrices.
    input_history_encoding: str | None = None
  # Whether the gated repetition-plane fix was active when x was encoded.
  # Part of replay identity alongside input_history_encoding: same encoding
  # name, different planes. Default False matches every pre-fix sample.
    history_rep_fix: bool = False

  # Optional auxiliary targets (for spec completeness; not all are trained yet)
  #
  # NOTE: With the "train on network turns only" scheme, SF targets (policy + eval)
  # are attached to the *network-turn* sample, representing Stockfish's reply to the
  # network's move and the evaluation after that reply.
    sf_wdl: np.ndarray | None = None  # (3,) float32
    sf_move_index: int | None = None  # action index for SF chosen move
    sf_played_move_index: int | None = None  # regret-sampled curriculum reply, if different from best
    sf_played_rank: int | None = None  # 1=best among MultiPV candidates
    sf_played_regret: float | None = None  # best_score - played_score in WDL winrate units
  # The generating net's RAW root policy prior, captured at selfplay time
  # BEFORE search re-ranks it. Together with policy_target (the search-improved
  # distribution) these give a SAME-MODEL (prior top-1 vs MCTS choice) pair on
  # every row: both come from the one theta that played the ply, by
  # construction. Nothing else in the schema records the generating prior --
  # _NetRecord.policy_probs never leaves selfplay memory -- so a checkpoint's
  # prior can otherwise only be paired against a DIFFERENT (historical) net's
  # played move. Written only when selfplay.record_prior_top1 is on.
  # index is in the SHARD's policy encoding (compact when policy_encoding is
  # lc0_1858), like sf_move_index -- so it MUST be mirror-remapped, not copied.
    prior_top1_index: int | None = None
  # ⚑ Softmax at T = 1.0 over the legal moves: the NET's mass on that move, NOT
  # the mass the search seeded its tree with (search divides root logits by
  # gumbel_policy_temp, production 1.5, so the search's prior is flatter and
  # this number is the higher of the two). The INDEX is unaffected -- argmax is
  # temperature-invariant. Rationale for storing the untempered one:
  # selfplay/network_turn._prior_top1.
    prior_top1_prob: float | None = None  # in [0, 1]

    sf_policy_target: np.ndarray | None = None  # (POLICY_SIZE,) float32 SF reply distribution
    sf_multipv_raw: np.ndarray | None = None  # (SF_MULTIPV_RAW_MAX, 5) int16 raw MultiPV rows
    sf_label_meta: np.ndarray | None = None   # (6,) int32 record-level SF eval metadata
    search_wdl: np.ndarray | None = None  # (3,) float32 — MCTS-improved value head prediction
    future_sf_regret_sum: float | None = None  # cumulative future SF reply regret in expected-score units
    future_sf_regret_d95: float | None = None
    future_sf_regret_d98: float | None = None
    future_sf_regret_max: float | None = None
    future_sf_regret_h4: float | None = None
    future_sf_regret_h6: float | None = None
    future_sf_regret_h12: float | None = None
    future_sf_regret_h24: float | None = None
    future_sf_regret_h50: float | None = None
    future_sf_regret_count: int | None = None
    moves_left: float | None = None
    is_network_turn: bool | None = None
    is_selfplay: bool | None = None

    categorical_target: np.ndarray | None = None  # (num_bins,) float32

    policy_soft_target: np.ndarray | None = None  # (POLICY_SIZE,) float32
    future_policy_target: np.ndarray | None = None  # (POLICY_SIZE,) float32
    has_future: bool | None = None

  # SF's recommended move for THIS position (P0 teacher for policy_own).
  # Only the prior ply's sf_policy_target is SF's analysis of *this* position
  # (SF labels run at P1 = after a move). In selfplay the net plays every ply,
  # so two consecutive full plies let the earlier record's sf_policy serve as
  # this record's own-move teacher. None except on those eligible rows.
    sf_p0_policy_target: np.ndarray | None = None  # (POLICY_SIZE,) float32
    has_sf_p0: bool | None = None

  # Per-move normalized SF cp-regret at THIS position (P0), same one-ply shift
  # as sf_p0_policy_target. Value vector in [0,1] (best move 0.0), NOT a
  # distribution. Drives the regret-weighted SF teacher (train.w_sf_own_regret).
    sf_p0_regret: np.ndarray | None = None  # (POLICY_SIZE,) float32
    has_sf_p0_regret: bool | None = None

    volatility_target: np.ndarray | None = None  # (3,) float32
    has_volatility: bool | None = None

    sf_volatility_target: np.ndarray | None = None  # (3,) float32
    has_sf_volatility: bool | None = None

  # LC0-style illegal move masking: 1=legal, 0=illegal, shape (POLICY_SIZE,).
  # Applied to policy logits before softmax during training to avoid wasting
  # probability mass on illegal moves. None for old shards (masking skipped).
    legal_mask: np.ndarray | None = None  # (POLICY_SIZE,) bool/uint8 — legal at t, net POV
  # Legal mask at t+1 (opponent POV) for policy_sf head.
    sf_legal_mask: np.ndarray | None = None
  # Legal mask at t+2 (net POV, next own move) for policy_future head.
    future_legal_mask: np.ndarray | None = None
