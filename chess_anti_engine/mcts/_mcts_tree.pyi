from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from chess_anti_engine.encoding._lc0_ext import CBoard

# C ABI capability marker (see PyInit__mcts_tree); bumped on detect-worthy ABI changes.
ABI_VERSION: int

# Which root sequential-halving semantic this .so was built with (see the
# GSS_HALVING_REV comment in _mcts_tree.c). NOT an ABI gate: absent on an .so
# built before the constant existed, which IS revision 1.
GSS_HALVING_REV: int

# `batch_process_ply`'s swdl_draw_mode encoding (SWDL_DRAW_* in _mcts_tree.c).
# Exported by the extension so the Python side never keeps a second copy of the
# int mapping — see selfplay/network_turn.py::_SWDL_DRAW_MODE_TO_C.
SWDL_DRAW_NET_RAW: int
SWDL_DRAW_PARAMETRIC_Q: int

def set_history_rep_fix(enabled: bool, /) -> None: ...
def tt_stats(reset: bool = False) -> dict[str, int]: ...

class MCTSTree:
    # --- eval-plugin seam -------------------------------------------------
    # The tree holds a POINTER to a value provider; NNUE is the first one. See
    # chess_anti_engine/mcts/_value_provider.h for the status contract and the
    # in-check rule (callers resolve check nodes recursively; the provider's
    # refusal is the enforcement backstop, not a substitute).
    # `provider` is a known name ("nnue") or the value_provider_capsule the
    # implementing extension publishes — the provider is imported, never
    # compiled into this extension, so its kernel selection and weight cache are
    # the ones its own module exposes.
    def set_value_provider(self, provider: str | object, weights_path: str) -> None: ...
    def clear_value_provider(self) -> None: ...
    def value_provider_name(self) -> str | None: ...
    def value_provider_kernel(self) -> str | None: ...
    def value_provider_eval(self, board: CBoard) -> int: ...
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
        enc_buf: NDArray[np.float32] | NDArray[np.uint16],
        vloss_weight: int = ...,
        target_batch: int = ...,
        input_history_lc0_root: int = ...,
        rel_buf: NDArray[np.uint8] | None = ...,
        q_visit_exp: float = ...,
        q_global_scale: int = ...,
        q_visit_floor: float = ...,
        halving_div: int = ...,
        c_visit_root: float = ...,
        c_scale_root: float = ...,
        q_visit_exp_root: float = ...,
        vloss_mode: int = ...,
    ) -> int | None: ...
    def continue_gumbel_sims(self, pol: NDArray[np.float32], wdl: NDArray[np.float32]) -> int | None: ...
    def continue_gumbel_sims_legal_bf16(self, pol_bf16_bits: NDArray[np.uint16], wdl: NDArray[np.float32]) -> int | None: ...
    def get_pending_legal_indices(self) -> tuple[NDArray[np.int32], NDArray[np.int32]]: ...
    # The WHOLE pending batch in encoded-planes row order. `get_pending_tb_leaves`
    # returns only the Syzygy-eligible subset; this one drops nothing, which is
    # what a CPU value function that reads the POSITION (not the planes) needs.
    def pending_leaf_cboards(self) -> list[CBoard]: ...
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
    def node_capacity(self) -> int: ...
    def memory_bytes(self) -> int: ...
    def reset(self) -> None: ...
    def reset_compact(self) -> None: ...
    def reserve(self, node_cap: int, child_cap: int = ...) -> None: ...
    def set_cpuct_scaling(self, factor: float, base: float = ...) -> None: ...
    def get_cpuct_scaling(self) -> tuple[float, float]: ...
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
        rel_out: NDArray[np.uint8] | None = ...,
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
        vloss_mode: int = ...,
        cache_keys: NDArray[np.uint64] | None = ...,
        rel_buf: NDArray[np.uint8] | None = ...,
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
    input_history_lc0_root: int = ...,
    n_extra: int = ...,
    with_relations: int = ...,
    df_norm_scale: float = ...,
    df_norm_slope: float = ...,
    df_norm_clip: float = ...,
    swdl_draw_mode: int = ...,
    swdl_cp_slope: float = ...,
    swdl_cp_draw_width: float = ...,
) -> tuple[NDArray[Any], ...]: ...

def batch_compute_relations(
    cboards: list[CBoard],
    out: NDArray[np.uint8],
) -> None: ...

def batch_encode_146(
    cboards: list[CBoard],
    out: NDArray[np.float32],
) -> None: ...

def batch_encode_146_lc0_root(
    cboards: list[CBoard],
    out: NDArray[np.float32],
) -> None: ...

def batch_encode_146_lc0_root_legacy_meta(
    cboards: list[CBoard],
    out: NDArray[np.float32],
) -> None: ...

def batch_encode_146_bf16(
    cboards: list[CBoard],
    out: NDArray[np.uint16],
) -> None: ...

def batch_encode_146_lc0_root_bf16(
    cboards: list[CBoard],
    out: NDArray[np.uint16],
) -> None: ...

def batch_encode_146_lc0_root_legacy_meta_bf16(
    cboards: list[CBoard],
    out: NDArray[np.uint16],
) -> None: ...

def classify_games(
    cboards: list[CBoard],
    net_color: NDArray[np.int8],
    done: NDArray[np.int8],
    finalized: NDArray[np.int8],
    selfplay_game: NDArray[np.int8],
    starting_ply: NDArray[np.int32],
    max_plies: int,
    check_terminal: bool = True,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]: ...

def temperature_resample(
    probs: NDArray[np.float32],
    temps: NDArray[np.float64],
    actions: NDArray[np.int32],
    rand_vals: NDArray[np.float64],
) -> None: ...
