from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from chess_anti_engine.encoding._lc0_ext import CBoard

class MCTSTree:
    def find_child(self, node_id: int, action: int) -> int: ...
    def add_root(self, N: int, W: float) -> int: ...
    def expand(self, node_id: int, actions: NDArray[np.int32], priors: NDArray[np.float64]) -> None: ...
    def expand_from_logits(self, node_id: int, legal: NDArray[np.int32], logits: NDArray[np.float32]) -> None: ...
    def batch_wdl_to_q(self, wdl: NDArray[np.float32]) -> NDArray[np.float64]: ...
    def select_leaves(
        self, root_ids: NDArray[np.int32], c_puct: float, fpu_at_root: float, fpu_reduction: float,
    ) -> list[tuple[int, NDArray[np.int32], NDArray[np.int32], bool]]: ...
    def backprop(self, node_path: NDArray[np.int32], value: float) -> None: ...
    def backprop_many(self, paths: list[NDArray[np.int32]], values: list[float]) -> None: ...
    def start_gumbel_sims(
        self,
        root_cbs: list[CBoard],
        root_ids: NDArray[np.int32],
        remaining_per_board: list[Any],
        gumbels_per_board: list[NDArray[np.float64]],
        root_priors: list[NDArray[np.float64]],
        budget_remaining: NDArray[np.int32],
        root_qs: NDArray[np.float64],
        c_scale: float,
        c_visit: float,
        c_puct: float,
        fpu_reduction: float,
        full_tree: bool | int,
        enc_buf: NDArray[np.float32],
        vloss_weight: int = ...,
        target_batch: int = ...,
    ) -> int | None: ...
    def continue_gumbel_sims(self, pol: NDArray[np.float32], wdl: NDArray[np.float32]) -> int | None: ...
    def get_pending_tb_leaves(self, max_pieces: int) -> tuple[NDArray[np.int32], list[CBoard]]: ...
    def mark_tb_solved(self, indices: NDArray[np.int32], statuses: NDArray[np.int8]) -> int: ...
    def get_solved_status(self, node_id: int) -> int: ...
    def mark_solved_path(self, node_path: NDArray[np.int32], status: int) -> None: ...
    def get_gumbel_remaining(self) -> list[list[int]]: ...
    def get_children_visits(self, node_id: int) -> tuple[NDArray[np.int32], NDArray[np.int32]]: ...
    def get_children_q(self, node_id: int, default_q: float) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64]]: ...
    def node_q(self, node_id: int) -> float: ...
    def is_expanded(self, node_id: int) -> bool: ...
    def node_count(self) -> int: ...
    def memory_bytes(self) -> int: ...
    def reset(self) -> None: ...
    def reserve(self, node_cap: int, child_cap: int = ...) -> None: ...
    def get_virtual_loss(self, node_id: int) -> int: ...
    def apply_vloss_path(self, path: NDArray[np.int32]) -> None: ...
    def remove_vloss_path(self, path: NDArray[np.int32]) -> None: ...
    def walker_descend_puct(
        self,
        root_id: int,
        root_cboard: CBoard,
        c_puct: float,
        fpu_root: float,
        fpu_reduction: float,
        vloss_weight: int,
        enc_out: NDArray[np.float32],
    ) -> tuple[int, NDArray[np.int32], NDArray[np.int32], float | None]: ...
    def walker_integrate_leaf(
        self,
        node_path: NDArray[np.int32],
        legal: NDArray[np.int32],
        pol_logits: NDArray[np.float32],
        wdl_logits: NDArray[np.float32],
        vloss_weight: int,
    ) -> None: ...
    def batch_descend_puct(
        self,
        root_id: int,
        root_cboard: CBoard,
        n_leaves: int,
        c_puct: float,
        fpu_root: float,
        fpu_reduction: float,
        vloss_weight: int,
        enc_buf: NDArray[np.float32],
        leaf_ids: NDArray[np.int32],
        path_buf: NDArray[np.int32],
        path_lens: NDArray[np.int32],
        legal_buf: NDArray[np.int32],
        legal_lens: NDArray[np.int32],
        term_qs: NDArray[np.float64],
        is_term: NDArray[np.int8],
    ) -> int: ...
    def batch_integrate_leaves(
        self,
        n_leaves: int,
        path_buf: NDArray[np.int32],
        path_lens: NDArray[np.int32],
        legal_buf: NDArray[np.int32],
        legal_lens: NDArray[np.int32],
        is_term: NDArray[np.int8],
        pol_logits: NDArray[np.float32],
        wdl_logits: NDArray[np.float32],
        vloss_weight: int,
    ) -> None: ...

def batch_process_ply(
    cboards: list[CBoard],
    pol: NDArray[np.float32],
    wdl: NDArray[np.float32],
    actions: NDArray[np.int32],
    values: NDArray[np.float64],
    mcts_probs: NDArray[np.float32],
    df_enabled: int,
    df_q_weight: float,
    df_pol_scale: float,
    df_min: float,
    df_slope: float,
) -> tuple[
    NDArray[np.float32],  # x
    NDArray[np.float32],  # probs
    NDArray[np.float32],  # wdl_net
    NDArray[np.float32],  # wdl_search
    NDArray[np.float64],  # priority
    NDArray[np.float64],  # keep_prob
    NDArray[np.int32],    # legal_mask
    NDArray[np.int32],    # ply
    NDArray[np.int32],    # pov
    NDArray[np.int32],    # game_over
]: ...

def batch_encode_146(
    cboards: list[CBoard],
    out: NDArray[np.float32],
) -> None: ...

def classify_games(
    cboards: list[CBoard],
    net_color: NDArray[np.int8],
    done: NDArray[np.int8],
    finalized: NDArray[np.int8],
    selfplay_game: NDArray[np.int8],
    max_plies: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]: ...

def temperature_resample(
    probs: NDArray[np.float32],
    temps: NDArray[np.float64],
    actions: NDArray[np.int32],
    rand_vals: NDArray[np.float64],
) -> None: ...
