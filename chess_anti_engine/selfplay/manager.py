from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from chess_anti_engine.inference import BatchEvaluator, LocalModelEvaluator
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.stockfish.uci import StockfishUCI, StockfishResult
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.mcts import MCTSConfig, GumbelConfig
from chess_anti_engine.mcts.puct import run_mcts_many
from chess_anti_engine.mcts.gumbel import run_gumbel_root_many

try:
    from chess_anti_engine.mcts.puct_c import run_mcts_many_c as _run_mcts_many_c
    _HAS_C_TREE = True
except ImportError:
    _HAS_C_TREE = False

try:
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c as _run_gumbel_root_many_c
    _HAS_GUMBEL_C = True
except ImportError:
    _HAS_GUMBEL_C = False

if TYPE_CHECKING:
    from chess_anti_engine.mcts.puct_c import run_mcts_many_c as _run_mcts_many_c  # noqa: F401,F811
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c as _run_gumbel_root_many_c  # noqa: F401,F811
from chess_anti_engine.moves import POLICY_SIZE, move_to_index, index_to_move, legal_move_mask
from chess_anti_engine.moves.encode import uci_to_policy_index
from chess_anti_engine.train.targets import hlgauss_target
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.opening import OpeningConfig, make_starting_board
from chess_anti_engine.selfplay.tablebase import rescore_game_samples, probe_best_move, _eligible
from chess_anti_engine.selfplay.temperature import temperature_for_ply
import chess



def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature to a probability distribution.

    Equivalent to softmax(log(p)/T) for p>0.
    """
    t = float(temperature)
    if t == 1.0:
        return probs.astype(np.float32, copy=False)
    p = probs.astype(np.float64, copy=True)
    p = np.maximum(p, 0.0)
    if float(p.sum()) <= 0:
        return probs.astype(np.float32, copy=False)
    p = p / float(p.sum())
    p = np.power(p, 1.0 / t)
    s = float(p.sum())
    if s > 0:
        p /= s
    return p.astype(np.float32, copy=False)


def _effective_curriculum_topk(
    *,
    random_move_prob: float,
    stage_end: float,
    topk_max: int,
    topk_min: int = 1,
) -> int:
    """Map the easy-stage random-move curriculum onto a top-k SF sampler.

    While random_move_prob is above `stage_end`, keep the full top-k sampler.
    As random_move_prob falls from `stage_end` to 0, shrink the sampler from
    `topk_max` to `topk_min`.
    """
    k_max = max(int(topk_min), int(topk_max))
    k_min = max(1, min(int(topk_min), k_max))
    stage = max(1e-6, float(stage_end))
    rp = max(0.0, float(random_move_prob))
    if rp >= stage:
        return int(k_max)
    frac = max(0.0, min(1.0, rp / stage))
    # Log-scale interpolation: k = k_max^frac (with k_min=1, log(1)=0).
    # This spends more time at low topk values where each step matters
    # (top-1 vs top-3 is huge; top-8 vs top-12 is nearly indistinguishable).
    log_k = math.log(max(1, k_min)) + frac * (math.log(max(1, k_max)) - math.log(max(1, k_min)))
    k = int(round(math.exp(log_k)))
    return max(k_min, min(k_max, k))


def _effective_curriculum_wdl_regret(
    *,
    random_move_prob: float,
    random_move_prob_start: float,
    random_move_prob_floor: float,
    regret_max: float,
    regret_min: float,
) -> float:
    """Map the current PID difficulty onto a max allowed WDL regret.

    Returns the maximum tolerated drop in Stockfish WDL win-equivalent score
    relative to the best PV move. Larger values allow weaker suboptimal moves.
    Negative inputs disable the regret filter and return infinity.
    """
    if float(regret_max) < 0.0 or float(regret_min) < 0.0:
        return float("inf")
    r_min = max(0.0, min(float(regret_min), float(regret_max)))
    r_max = max(r_min, float(regret_max))
    floor = max(0.0, float(random_move_prob_floor))
    start = max(floor + 1e-6, float(random_move_prob_start))
    rp = max(floor, min(start, float(random_move_prob)))
    frac = (rp - floor) / max(1e-6, start - floor)
    return float(r_min + frac * (r_max - r_min))


def _choose_curriculum_opponent_move(
    *,
    rng: np.random.Generator,
    legal_indices: np.ndarray,
    cand_indices: list[int],
    cand_scores: list[float],
    curriculum_topk: int,
    random_move_prob: float,
    regret_limit: float,
) -> int:
    """Choose the curriculum opponent move index from Stockfish candidates.

    Returns a policy index (int).  No python-chess objects needed.
    """
    if not cand_indices:
        return int(legal_indices[int(rng.integers(len(legal_indices)))])

    topk_n = max(1, min(int(curriculum_topk), len(cand_indices)))
    topk_indices = cand_indices[:topk_n]
    topk_scores = cand_scores[:topk_n]
    rand_p = max(0.0, float(random_move_prob))

    if not math.isfinite(float(regret_limit)):
        if rand_p > 0.0 and rng.random() < rand_p:
            return int(legal_indices[int(rng.integers(len(legal_indices)))])
        return topk_indices[int(rng.integers(len(topk_indices)))]

    best_score = max(float(s) for s in topk_scores)
    acceptable = [
        idx
        for idx, score in zip(topk_indices, topk_scores, strict=False)
        if (best_score - float(score)) <= float(regret_limit) + 1e-12
    ]
    if not acceptable:
        acceptable = [topk_indices[0]]
    if rand_p > 0.0 and rng.random() < rand_p:
        return int(legal_indices[int(rng.integers(len(legal_indices)))])
    return acceptable[int(rng.integers(len(acceptable)))]


def _is_network_turn(*, board_turn: chess.Color, network_color: chess.Color) -> bool:
    """Return True when the side to move is the network-assigned color."""
    return board_turn == network_color


def _net_color(arr: np.ndarray, i: int) -> chess.Color:
    """Convert _net_color_arr[i] (1=WHITE, 0=BLACK) to chess.Color."""
    return chess.WHITE if arr[i] else chess.BLACK


@dataclass
class BatchStats:
    games: int
    positions: int
    w: int
    d: int
    l: int
    total_game_plies: int = 0
    adjudicated_games: int = 0
    total_draw_games: int = 0
    selfplay_games: int = 0
    selfplay_adjudicated_games: int = 0
    selfplay_draw_games: int = 0
    curriculum_games: int = 0
    curriculum_adjudicated_games: int = 0
    curriculum_draw_games: int = 0
    checkmate_games: int = 0
    stalemate_games: int = 0
    plies_win: int = 0
    plies_draw: int = 0
    plies_loss: int = 0

    # Adaptive difficulty PID diagnostics (optional)
    sf_nodes: int | None = None
    sf_nodes_next: int | None = None
    pid_ema_winrate: float | None = None
    random_move_prob: float | None = None
    skill_level: int | None = None

    # Log-only: mean abs delta of SF's winrate-like eval over 6 plies.
    # When training only on network turns, this is computed between SF reply evals
    # attached to samples t and t+3 (i.e. 6 plies apart).
    sf_eval_delta6: float = 0.0
    sf_eval_delta6_n: int = 0


@dataclass
class CompletedGameBatch:
    samples: list[ReplaySample]
    games: int = 1
    positions: int = 0
    w: int = 0
    d: int = 0
    l: int = 0
    total_game_plies: int = 0
    adjudicated_games: int = 0
    total_draw_games: int = 0
    selfplay_games: int = 0
    selfplay_adjudicated_games: int = 0
    selfplay_draw_games: int = 0
    curriculum_games: int = 0
    curriculum_adjudicated_games: int = 0
    curriculum_draw_games: int = 0
    checkmate_games: int = 0
    stalemate_games: int = 0
    plies_win: int = 0
    plies_draw: int = 0
    plies_loss: int = 0


class _NetRecord:
    """Per-ply sample record.  Uses __slots__ to avoid dict overhead."""
    __slots__ = (
        "x", "policy_probs", "net_wdl_est", "search_wdl_est",
        "pov_color", "ply_index", "has_policy", "priority",
        "sample_weight", "keep_prob", "legal_mask",
        "sf_policy_target", "sf_move_index", "sf_wdl",
    )

    x: np.ndarray
    policy_probs: np.ndarray
    net_wdl_est: np.ndarray
    search_wdl_est: np.ndarray
    pov_color: bool
    ply_index: int
    has_policy: bool
    priority: float
    sample_weight: float
    keep_prob: float
    legal_mask: np.ndarray | None
    sf_policy_target: np.ndarray | None
    sf_move_index: int | None
    sf_wdl: np.ndarray | None

    def __init__(
        self, x, policy_probs, net_wdl_est, search_wdl_est,
        pov_color, ply_index, has_policy, priority,
        sample_weight, keep_prob, legal_mask=None,
        sf_policy_target=None, sf_move_index=None, sf_wdl=None,
    ):
        self.x = x
        self.policy_probs = policy_probs
        self.net_wdl_est = net_wdl_est
        self.search_wdl_est = search_wdl_est
        self.pov_color = pov_color
        self.ply_index = ply_index
        self.has_policy = has_policy
        self.priority = priority
        self.sample_weight = sample_weight
        self.keep_prob = keep_prob
        self.legal_mask = legal_mask
        self.sf_policy_target = sf_policy_target
        self.sf_move_index = sf_move_index
        self.sf_wdl = sf_wdl


def play_batch(
    model: torch.nn.Module | None,
    *,
    device: str,
    rng: np.random.Generator,
    stockfish: StockfishUCI | StockfishPool,
    evaluator: BatchEvaluator | None = None,
    games: int,
    target_games: int = 0,
    on_game_complete: Callable[[CompletedGameBatch], None] | None = None,
    on_step: Callable[[], None] | None = None,
    stop_fn: Callable[[], bool] | None = None,
    # Config groups (frozen dataclasses with sensible defaults).
    opponent: OpponentConfig = OpponentConfig(),
    temp: TemperatureConfig = TemperatureConfig(),
    search: SearchConfig = SearchConfig(),
    opening: OpeningConfig = OpeningConfig(),
    diff_focus: DiffFocusConfig = DiffFocusConfig(),
    game: GameConfig = GameConfig(),
) -> tuple[list[ReplaySample], BatchStats]:
    """Play a batch of games.

    Design goals:
    - keep GPU busy via batched inference
    - keep SF queries minimal (one per SF ply)
    - compute volatility targets from a consistent network-side WDL series without per-ply overhead

    Continuous mode (stop_fn provided, target_games=0): runs forever with all
    finished slots recycled immediately, until stop_fn() returns True.  Samples
    are delivered incrementally via on_game_complete; the returned sample list
    is empty to avoid unbounded memory growth.

    Finite mode (target_games > 0 or stop_fn is None): plays exactly
    target_games (or `games` if target_games=0) then returns all samples.
    """

    requested_batch = int(games)
    continuous = stop_fn is not None and int(target_games) <= 0
    target = int(target_games) if int(target_games) > 0 else requested_batch
    batch_size = min(requested_batch, target)
    if batch_size <= 0:
        raise ValueError("play_batch requires at least one game")

    eval_impl = evaluator
    if eval_impl is None:
        if model is None:
            raise ValueError("play_batch requires model or evaluator")
        eval_impl = LocalModelEvaluator(model, device=device)

    boards = [make_starting_board(rng=rng, cfg=opening) for _ in range(batch_size)]
    # Parallel CBoard state — primary board for hot-path ops (fen, game_over, ply, encoding).
    # Use from_board (not from_raw) to preserve ply count and history from openings.
    from chess_anti_engine.encoding._lc0_ext import CBoard as _CBoard
    cboards = [_CBoard.from_board(b) for b in boards]
    # Keep a copy of the starting position for tablebase replay after game ends.
    starting_boards = [b.copy() for b in boards] if game.syzygy_path else None
    # Move index history per game — used to reconstruct python-chess boards at
    # finalization (cold path only). Hot-path uses CBoard exclusively.
    move_idx_history: list[list[int]] = [[] for _ in range(batch_size)]
    # Per-game flags as numpy int8 arrays (single source of truth).
    # Used directly by C classify_games and by Python logic (truthy checks work for 0/1).
    _done_arr = np.zeros(batch_size, dtype=np.int8)
    _finalized_arr = np.zeros(batch_size, dtype=np.int8)
    # Alternate which color the network plays so it sees both perspectives.
    # 1 = WHITE, 0 = BLACK.
    _net_color_arr = np.array([1 if (i % 2 == 0) else 0 for i in range(batch_size)], dtype=np.int8)
    _selfplay_arr = np.array([
        1 if rng.random() < max(0.0, min(1.0, float(game.selfplay_fraction))) else 0
        for _ in range(batch_size)
    ], dtype=np.int8)
    _has_classify_c = False
    try:
        from chess_anti_engine.mcts._mcts_tree import classify_games as _c_classify, temperature_resample as _c_temp_resample
        _has_classify_c = True
    except ImportError:
        pass

    samples_per_game: list[list[_NetRecord]] = [[] for _ in range(batch_size)]

    # NN eval cache — avoids redundant GPU evals for positions seen in MCTS
    _nn_cache = None
    _mcts_tree = None  # Persistent tree for cross-ply reuse
    _root_ids: list[int] = [-1] * batch_size  # Per-game root node IDs
    if _HAS_GUMBEL_C:
        from chess_anti_engine.mcts._mcts_tree import NNCache as _NNCache, MCTSTree as _MCTSTree
        _nn_cache = _NNCache(1 << 17)  # 131072 slots
        _mcts_tree = _MCTSTree()

    # Check C per-ply availability once. `batch_process_ply` and `batch_encode_146`
    # live in the same C module; importing together keeps the two `None`-fallback
    # paths coupled (prevents drift where only one is available, which would
    # break the xs_batch invariant in _run_network_turn).
    try:
        from chess_anti_engine.mcts._mcts_tree import (
            batch_process_ply as _c_process_ply,
            batch_encode_146 as _batch_enc_146,
        )
        _has_c_ply = True
    except ImportError:
        _c_process_ply = None
        _batch_enc_146 = None
        _has_c_ply = False
        logging.getLogger("chess_anti_engine.selfplay").warning(
            "C per-ply fast path unavailable (batch_process_ply/batch_encode_146 not importable "
            "from _mcts_tree); falling back to Python. Rebuild the C extension for production."
        )

    SOFT_RESIGN_THRESHOLD = 0.05
    SOFT_RESIGN_CONSECUTIVE = 5
    consecutive_low_winrate: list[int] = [0] * batch_size
    sf_resign_scale: list[float] = [1.0] * batch_size
    last_net_full: list[bool] = [True] * batch_size

    # ── Stats accumulators (updated by _finalize_game) ────────────────────────
    all_samples: list[ReplaySample] = []
    _st_w = _st_d = _st_l = 0
    _st_game_plies = 0
    _st_adjudicated = 0
    _st_draw = 0
    _st_sp_games = _st_sp_adj = _st_sp_draw = 0
    _st_cur_games = _st_cur_adj = _st_cur_draw = 0
    _st_plies_win = _st_plies_draw = _st_plies_loss = 0
    _st_sf_d6_sum = 0.0
    _st_sf_d6_n = 0
    _st_checkmate = 0
    _st_stalemate = 0

    base_nodes = int(getattr(stockfish, "nodes", 0) or 0)
    terminal_eval_nodes = (5 * base_nodes) if base_nodes > 0 else 1000

    vs = str(game.volatility_source).lower().strip()
    if vs not in ("raw", "search"):
        vs = "raw"

    def _sf_terminal_result(turn_is_white: bool, sf_res: StockfishResult | None) -> str:
        if sf_res is None or sf_res.wdl is None:
            return "1/2-1/2"
        wdl_stm = sf_res.wdl
        if not turn_is_white:
            wdl_white = np.array([float(wdl_stm[2]), float(wdl_stm[1]), float(wdl_stm[0])], dtype=np.float32)
        else:
            wdl_white = np.asarray(wdl_stm, dtype=np.float32)
        if float(wdl_white[0]) > float(game.timeout_adjudication_threshold):
            return "1-0"
        if float(wdl_white[2]) > float(game.timeout_adjudication_threshold):
            return "0-1"
        return "1/2-1/2"

    def _finalize_game(i: int) -> None:
        """Finalize a completed game: compute labels, build samples, update stats."""
        nonlocal _st_w, _st_d, _st_l, _st_game_plies, _st_adjudicated, _st_draw
        nonlocal _st_sp_games, _st_sp_adj, _st_sp_draw
        nonlocal _st_cur_games, _st_cur_adj, _st_cur_draw
        nonlocal _st_sf_d6_sum, _st_sf_d6_n, _st_checkmate, _st_stalemate
        nonlocal _st_plies_win, _st_plies_draw, _st_plies_loss

        cb = cboards[i]
        # When C per-ply is active, boards[i] is the starting position —
        # reconstruct by replaying move_idx_history.  When Python fallback
        # is active, boards[i] is already the final position (pushed each ply).
        if _has_c_ply:
            b = boards[i].copy(stack=False)
            for _mi in move_idx_history[i]:
                _mv = index_to_move(_mi, b)
                b.push(_mv)
        else:
            b = boards[i]
        result = cb.result()
        _game_plies = int(cb.ply)
        _st_game_plies += _game_plies
        _is_cm = cb.is_checkmate()
        _is_sm = cb.is_stalemate()
        if _is_cm:
            _st_checkmate += 1
        elif _is_sm:
            _st_stalemate += 1

        was_adjudicated = False
        if result == "*":
            _st_adjudicated += 1
            was_adjudicated = True
            try:
                if isinstance(stockfish, StockfishPool):
                    sf_res = stockfish.submit(cb.fen(), nodes=int(terminal_eval_nodes)).result()
                else:
                    sf_res = stockfish.search(cb.fen(), nodes=int(terminal_eval_nodes))
            except Exception:
                sf_res = None
            result = _sf_terminal_result(bool(cb.turn), sf_res)

        # Debug: log unexpected short wins
        if not _selfplay_arr[i] and result in ("1-0", "0-1") and _game_plies < int(game.max_plies) - 10:
            outcome = b.outcome(claim_draw=True)
            _term = outcome.termination.name if outcome else "adjudicated"
            _log = logging.getLogger("chess_anti_engine.selfplay")
            _log.warning(
                "Short win: %s at %d plies (max=%d), term=%s, adj=%s, is_over=%s",
                result, _game_plies, int(game.max_plies), _term,
                was_adjudicated, cb.is_game_over(),
            )

        records = samples_per_game[i]

        # Syzygy tablebase rescoring
        tb_policy_overrides: dict[int, np.ndarray] = {}
        if game.syzygy_path and starting_boards is not None:
            replay_board = starting_boards[i].copy()
            replay_boards: list[chess.Board] = []
            move_stack = list(b.move_stack)
            opening_len = len(starting_boards[i].move_stack)

            for mv in move_stack[opening_len:]:
                replay_board.push(mv)
                replay_boards.append(replay_board.copy())

            tb_result = rescore_game_samples(replay_boards, game.syzygy_path)
            if tb_result is not None:
                result = tb_result

            if game.syzygy_policy:
                replay_board = starting_boards[i].copy()
                sample_idx = 0
                _is_sp = bool(_selfplay_arr[i])
                for mv in move_stack[opening_len:]:
                    board_before = replay_board.copy()
                    is_net = _is_network_turn(board_turn=replay_board.turn, network_color=_net_color(_net_color_arr, i))
                    # In selfplay games the network plays both sides, so every
                    # ply produces a sample.  In curriculum games only the
                    # network-color turns produce samples.
                    is_sample_turn = is_net or _is_sp
                    if is_sample_turn:
                        if sample_idx >= len(records):
                            break
                        if _eligible(board_before):
                            best = probe_best_move(board_before, game.syzygy_path)
                            if best is not None:
                                try:
                                    a = int(move_to_index(best, board_before))
                                except Exception:
                                    a = -1
                                if a >= 0:
                                    p = np.zeros((POLICY_SIZE,), dtype=np.float32)
                                    p[a] = 1.0
                                    tb_policy_overrides[sample_idx] = p
                        sample_idx += 1
                    replay_board.push(mv)

        game_w = 0
        game_d = 0
        game_l = 0
        game_total_draws = 0
        game_selfplay_games = 0
        game_selfplay_adj = 0
        game_selfplay_draws = 0
        game_curriculum_games = 0
        game_curriculum_adj = 0
        game_curriculum_draws = 0

        # Stats
        if _selfplay_arr[i]:
            _st_sp_games += 1
            game_selfplay_games = 1
            if was_adjudicated:
                _st_sp_adj += 1
                game_selfplay_adj = 1
        else:
            _st_cur_games += 1
            game_curriculum_games = 1
            if was_adjudicated:
                _st_cur_adj += 1
                game_curriculum_adj = 1

        if result == "1/2-1/2":
            _st_draw += 1
            game_total_draws = 1
            if _selfplay_arr[i]:
                _st_sp_draw += 1
                game_selfplay_draws = 1
            else:
                _st_cur_draw += 1
                game_curriculum_draws = 1

        if not _selfplay_arr[i]:
            net_col = _net_color(_net_color_arr, i)
            if result == "1/2-1/2":
                _st_d += 1
                game_d = 1
            elif (result == "1-0" and net_col == chess.WHITE) or (result == "0-1" and net_col == chess.BLACK):
                _st_w += 1
                game_w = 1
            else:
                _st_l += 1
                game_l = 1

        if not _selfplay_arr[i]:
            if game_w:
                _st_plies_win += _game_plies
            elif game_d:
                _st_plies_draw += _game_plies
            elif game_l:
                _st_plies_loss += _game_plies

        n = len(records)
        ply_to_index = {int(rec.ply_index): idx for idx, rec in enumerate(records)}

        # Volatility targets + SF eval delta6 metric (single pass over records).
        vol_targets: list[np.ndarray | None] = [None] * n
        sf_vol_targets: list[np.ndarray | None] = [None] * n
        for t in range(n):
            th = ply_to_index.get(int(records[t].ply_index) + 6)
            if th is not None:
                if vs == "search":
                    w0 = records[t].search_wdl_est
                    w6 = records[th].search_wdl_est
                else:
                    w0 = records[t].net_wdl_est
                    w6 = records[th].net_wdl_est
                vol_targets[t] = np.abs(w6 - w0).astype(np.float32, copy=False)

                sf0 = records[t].sf_wdl
                sf6 = records[th].sf_wdl
                if (sf0 is not None) and (sf6 is not None):
                    sf_vol_targets[t] = np.abs(sf6 - sf0).astype(np.float32, copy=False)
                    wr0 = float(sf0[0]) + 0.5 * float(sf0[1])
                    wr6 = float(sf6[0]) + 0.5 * float(sf6[1])
                    _st_sf_d6_sum += abs(wr6 - wr0)
                    _st_sf_d6_n += 1

        # Build ReplaySample objects
        sample_start = len(all_samples)
        for t, rec in enumerate(records):
            if float(rec.sample_weight) < 1.0 and rng.random() > float(rec.sample_weight):
                continue
            if float(rec.keep_prob) < 1.0 and rng.random() > float(rec.keep_prob):
                continue
            if not bool(rec.has_policy):
                continue

            if result == "1/2-1/2":
                wdl = 1
            elif (result == "1-0" and rec.pov_color == chess.WHITE) or \
                 (result == "0-1" and rec.pov_color == chess.BLACK):
                wdl = 0
            else:
                wdl = 2

            total_plies_played = max(1, int(cb.ply))
            moves_left = float(max(0, total_plies_played - int(rec.ply_index))) / max(1.0, float(game.max_plies))

            scalar_v = 1.0 if wdl == 0 else (0.0 if wdl == 1 else -1.0)
            cat = hlgauss_target(scalar_v, num_bins=game.categorical_bins, sigma=game.hlgauss_sigma)

            eff_probs = tb_policy_overrides.get(t, rec.policy_probs)
            soft = _apply_temperature(eff_probs, game.soft_policy_temp)

            future = None
            future_idx = ply_to_index.get(int(rec.ply_index) + 2)
            if future_idx is not None and bool(records[future_idx].has_policy):
                future = records[future_idx].policy_probs

            vol = vol_targets[t]
            sf_vol = sf_vol_targets[t]

            all_samples.append(
                ReplaySample(
                    x=rec.x,
                    policy_target=eff_probs,
                    wdl_target=int(wdl),
                    priority=float(rec.priority),
                    has_policy=bool(rec.has_policy),
                    sf_wdl=rec.sf_wdl,
                    sf_move_index=rec.sf_move_index,
                    sf_policy_target=rec.sf_policy_target,
                    moves_left=moves_left,
                    is_network_turn=True,
                    categorical_target=cat,
                    policy_soft_target=soft,
                    future_policy_target=future,
                    has_future=(future is not None),
                    volatility_target=vol,
                    has_volatility=(vol is not None),
                    sf_volatility_target=sf_vol,
                    has_sf_volatility=(sf_vol is not None),
                    legal_mask=rec.legal_mask,
                )
            )

        if on_game_complete is not None:
            game_samples = list(all_samples[sample_start:])
            if game_samples:
                on_game_complete(
                    CompletedGameBatch(
                        samples=game_samples,
                        positions=len(game_samples),
                        w=game_w,
                        d=game_d,
                        l=game_l,
                        total_game_plies=_game_plies,
                        adjudicated_games=1 if was_adjudicated else 0,
                        total_draw_games=game_total_draws,
                        selfplay_games=game_selfplay_games,
                        selfplay_adjudicated_games=game_selfplay_adj,
                        selfplay_draw_games=game_selfplay_draws,
                        curriculum_games=game_curriculum_games,
                        curriculum_adjudicated_games=game_curriculum_adj,
                        curriculum_draw_games=game_curriculum_draws,
                        checkmate_games=1 if _is_cm else 0,
                        stalemate_games=1 if _is_sm else 0,
                        plies_win=_game_plies if game_w else 0,
                        plies_draw=_game_plies if game_d else 0,
                        plies_loss=_game_plies if game_l else 0,
                    )
                )
        # In continuous mode, samples are delivered via on_game_complete;
        # clear the list to prevent unbounded memory growth.
        if continuous:
            all_samples.clear()

    # ── Slot recycling ────────────────────────────────────────────────────────
    games_started = batch_size
    games_completed = 0
    def _recycle_slot(i: int) -> None:
        nonlocal games_started
        boards[i] = make_starting_board(rng=rng, cfg=opening)
        cboards[i] = _CBoard.from_board(boards[i])
        if starting_boards is not None:
            starting_boards[i] = boards[i].copy()
        move_idx_history[i] = []
        _done_arr[i] = 0
        _finalized_arr[i] = 0
        _net_color_arr[i] = 1 if (games_started % 2 == 0) else 0
        _selfplay_arr[i] = 1 if rng.random() < max(0.0, min(1.0, float(game.selfplay_fraction))) else 0
        samples_per_game[i] = []
        consecutive_low_winrate[i] = 0
        sf_resign_scale[i] = 1.0
        last_net_full[i] = True
        _root_ids[i] = -1  # Reset tree reuse for new game
        games_started += 1

    # ── Network turn ──────────────────────────────────────────────────────────

    def _run_network_turn(net_idxs: list[int]) -> None:
        if not net_idxs:
            return

        _bsz = len(net_idxs)
        # Bucket size for torch.compile shape stability
        _ROOT_BUCKETS = (32, 64, 128, 256, 512)
        _padded_bsz = _bsz
        for _b in _ROOT_BUCKETS:
            if _b >= _bsz:
                _padded_bsz = _b
                break

        _cb_encode_list = [cboards[_idx] for _idx in net_idxs]

        # Fast path: encode directly into evaluator's pinned buffer (zero-copy H2D)
        _use_inplace = _batch_enc_146 is not None and hasattr(eval_impl, "get_input_buffer")
        if _use_inplace:
            # get_input_buffer + evaluate_inplace exist only on DirectGPU-style evaluators;
            # _use_inplace is the runtime guard.
            _buf = eval_impl.get_input_buffer(_padded_bsz)  # type: ignore[attr-defined]
            # batch_encode_146 memsets + encodes into first _bsz slots;
            # remaining slots are zero-padded by get_input_buffer's pinned memory.
            assert _batch_enc_146 is not None
            _batch_enc_146(_cb_encode_list, _buf)
            pol_logits_padded, wdl_logits_raw_padded = eval_impl.evaluate_inplace(  # type: ignore[attr-defined]
                _padded_bsz, copy_out=True,
            )
        else:
            xs_batch = np.empty((_bsz, 146, 8, 8), dtype=np.float32)
            if _batch_enc_146 is not None:
                _batch_enc_146(_cb_encode_list, xs_batch)
            else:
                for _j, _idx in enumerate(net_idxs):
                    xs_batch[_j] = cboards[_idx].encode_146()
            if _padded_bsz > _bsz:
                _pad = np.zeros((_padded_bsz - _bsz, *xs_batch.shape[1:]), dtype=xs_batch.dtype)
                xs_padded = np.concatenate([xs_batch, _pad], axis=0)
            else:
                xs_padded = xs_batch
            pol_logits_padded, wdl_logits_raw_padded = eval_impl.evaluate_encoded(xs_padded)

        pol_logits = pol_logits_padded[:_bsz]
        wdl_logits_raw = wdl_logits_raw_padded[:_bsz]
        # Pure numpy softmax (avoids torch tensor creation roundtrip for small arrays)
        wdl_f = wdl_logits_raw.astype(np.float64, copy=True)
        wdl_f -= wdl_f.max(axis=-1, keepdims=True)
        np.exp(wdl_f, out=wdl_f)
        wdl_f /= wdl_f.sum(axis=-1, keepdims=True)
        wdl_est = wdl_f.astype(np.float32, copy=False)

        is_full = rng.random(size=len(net_idxs)) < float(search.playout_cap_fraction)

        eff_full_sims = [int(search.simulations)] * len(net_idxs)
        eff_fast_sims = [int(search.fast_simulations)] * len(net_idxs)
        sample_weights = [1.0] * len(net_idxs)
        for j, idx in enumerate(net_idxs):
            win_p = float(wdl_est[j][0])
            if win_p < SOFT_RESIGN_THRESHOLD:
                consecutive_low_winrate[idx] += 1
            else:
                consecutive_low_winrate[idx] = 0

            if consecutive_low_winrate[idx] >= SOFT_RESIGN_CONSECUTIVE:
                ratio = win_p / SOFT_RESIGN_THRESHOLD
                sample_weights[j] = 0.1 + 0.9 * ratio

            sf_resign_scale[idx] = 1.0

        probs_list: list[np.ndarray | None] = [None] * len(net_idxs)
        actions: list[int | None] = [None] * len(net_idxs)
        values_list: list[float | None] = [None] * len(net_idxs)
        masks_list: list[np.ndarray | None] = [None] * len(net_idxs)

        # Per-game temperature based on each game's own ply count.
        temps = [
            temperature_for_ply(
                ply=cboards[i].ply // 2 + 1,
                temperature=float(temp.temperature),
                drop_plies=int(temp.drop_plies),
                after=float(temp.after),
                decay_start_move=int(temp.decay_start_move),
                decay_moves=int(temp.decay_moves),
                endgame=float(temp.endgame),
            )
            for i in net_idxs
        ]

        def _run_mcts_group(idxs: list[int], sims_per: list[int], *, per_game_noise: list[bool] | None = None) -> None:
            if not idxs:
                return

            # All games share one MCTS call with per-game sim budgets and
            # temperature applied after search.  Maximizes GPU batch size.
            group = idxs
            sub_cboards = [cboards[net_idxs[j]] for j in group]
            sub_temps = [temps[j] for j in group]
            sub_sims = [int(sims_per[j]) for j in group]
            sim_count = max(sub_sims)

            gumbel_low_sims = max(64, int(search.fast_simulations))
            use_gumbel = (str(search.mcts_type) == "gumbel") or (int(sim_count) <= int(gumbel_low_sims))
            sub_pol = pol_logits[group, :]
            sub_wdl = wdl_logits_raw[group, :]

            if use_gumbel:
                _gumbel_fn = _run_gumbel_root_many_c if _HAS_GUMBEL_C else run_gumbel_root_many
                # C gumbel only uses cboards; Python fallback needs python-chess boards
                sub_boards = sub_cboards if _HAS_GUMBEL_C else [boards[net_idxs[j]] for j in group]
                sub_noise = [per_game_noise[j] for j in group] if per_game_noise is not None else None
                # Map group indices to game-level root IDs for tree reuse
                sub_root_ids = [_root_ids[net_idxs[j]] for j in group] if _mcts_tree is not None else None
                _gumbel_result = _gumbel_fn(
                    model,
                    sub_boards,  # type: ignore[arg-type]
                    device=device,
                    rng=rng,
                    cfg=GumbelConfig(simulations=int(sim_count), temperature=1.0, add_noise=True),
                    evaluator=eval_impl,
                    pre_pol_logits=sub_pol,
                    pre_wdl_logits=sub_wdl,
                    per_game_simulations=sub_sims,
                    per_game_add_noise=sub_noise,
                    cboards=sub_cboards,  # type: ignore[call-arg]
                    nn_cache=_nn_cache,  # type: ignore[call-arg]
                    tree=_mcts_tree,  # type: ignore[call-arg]
                    root_node_ids=sub_root_ids,  # type: ignore[call-arg]
                )
                # C version returns 6-tuple (with tree, root_ids), Python returns 4-tuple
                p_sub, a_sub, v_sub, m_sub = _gumbel_result[:4]
                # Store returned root IDs for tree reuse
                if _mcts_tree is not None and len(_gumbel_result) >= 6:
                    _ret_root_ids = _gumbel_result[5]
                    for gi, jj in enumerate(group):
                        _root_ids[net_idxs[jj]] = _ret_root_ids[gi]
            else:
                # PUCT needs python-chess boards.  When the C fast path is
                # active, boards[idx] stays at the starting position — rebuild
                # current state from move_idx_history (same as _finalize_game).
                if _has_c_ply:
                    sub_boards = []
                    for j in group:
                        idx = net_idxs[j]
                        b = boards[idx].copy(stack=False)
                        for _mi in move_idx_history[idx]:
                            b.push(index_to_move(_mi, b))
                        sub_boards.append(b)
                else:
                    sub_boards = [boards[net_idxs[j]] for j in group]
                _puct_fn = _run_mcts_many_c if _HAS_C_TREE else run_mcts_many
                p_sub, a_sub, v_sub, m_sub = _puct_fn(
                    model,
                    sub_boards,
                    device=device,
                    rng=rng,
                    cfg=MCTSConfig(
                        simulations=int(sim_count),
                        temperature=1.0,
                        fpu_reduction=float(search.fpu_reduction),
                        fpu_at_root=float(search.fpu_at_root),
                    ),
                    evaluator=eval_impl,
                    pre_pol_logits=sub_pol,
                    pre_wdl_logits=sub_wdl,
                    cboards=sub_cboards,  # type: ignore[call-arg]
                )

            # Re-select actions with per-game temperature from the
            # improved policy (probs are temperature-independent).
            _temps_arr = np.array(sub_temps, dtype=np.float64)
            _need_resample = _temps_arr != 1.0
            if _need_resample.any():
                _p_stack = np.stack(p_sub)  # (G, 4672)
                if _has_classify_c:
                    # C path: GIL released during pow/sample
                    _a_arr = np.array(a_sub, dtype=np.int32)
                    _rand_arr = rng.random(len(sub_temps))
                    _c_temp_resample(_p_stack, _temps_arr, _a_arr, _rand_arr)  # pyright: ignore[reportPossiblyUnboundVariable]
                    for gi in range(len(a_sub)):
                        a_sub[gi] = int(_a_arr[gi])
                else:
                    _nonzero = _need_resample.nonzero()[0]
                    for gi in _nonzero:
                        p = _p_stack[gi]
                        t = _temps_arr[gi]
                        legal = np.flatnonzero(p > 0)
                        if len(legal) == 0:
                            continue
                        if t <= 0:
                            a_sub[gi] = int(legal[np.argmax(p[legal])])
                        else:
                            pw = np.power(p[legal], 1.0 / t)
                            ps = float(pw.sum())
                            if ps > 0:
                                pw /= ps
                                a_sub[gi] = int(rng.choice(legal, p=pw))

            for jj, p, a, v, m in zip(group, p_sub, a_sub, v_sub, m_sub, strict=True):
                probs_list[jj] = p
                actions[jj] = a
                values_list[jj] = float(v)
                masks_list[jj] = m
                # Advance tree root to chosen move's child for next-ply reuse
                if _mcts_tree is not None:
                    game_idx = net_idxs[jj]
                    rid = _root_ids[game_idx]
                    if rid >= 0:
                        child = _mcts_tree.find_child(rid, int(a))
                        _root_ids[game_idx] = child  # -1 if not found

        # Run all games in one MCTS call with per-game sim budgets and noise.
        # Full-sim games get Gumbel noise for exploration; fast games don't
        # (KataGo playout cap convention).
        all_idxs = list(range(len(net_idxs)))
        combined_sims = [
            int(eff_full_sims[j]) if bool(is_full[j]) else int(eff_fast_sims[j])
            for j in all_idxs
        ]
        noise_flags = [bool(is_full[j]) for j in all_idxs]
        _run_mcts_group(all_idxs, combined_sims, per_game_noise=noise_flags)

        # Pre-allocate reusable buffers for per-sample computation
        _lg_buf = np.empty(POLICY_SIZE, dtype=np.float64)
        _swdl_buf = np.empty(3, dtype=np.float32)
        _df_enabled = bool(diff_focus.enabled)
        _df_q_w = float(diff_focus.q_weight)
        _df_p_s = float(diff_focus.pol_scale)
        _df_slope = float(diff_focus.slope)
        _df_min = float(diff_focus.min_keep)

        # ── C-accelerated per-ply processing (GIL released) ──────────
        if _has_c_ply and len(net_idxs) > 0:
            _n = len(net_idxs)
            _cb_list = [cboards[net_idxs[j]] for j in range(_n)]
            _actions_arr = np.array(actions, dtype=np.int32)
            _values_arr = np.array(values_list, dtype=np.float64)
            # All slots filled by _run_mcts_group above; cast for np.stack's strict ArrayLike protocol.
            _probs_arr = np.stack(cast("list[np.ndarray]", probs_list)).astype(np.float32, copy=False)

            assert _c_process_ply is not None
            (c_x, c_probs, c_wdl_net, c_wdl_search, c_priority,
             c_keep, c_mask, c_ply, c_pov, c_over) = _c_process_ply(
                _cb_list, pol_logits[:_n], wdl_logits_raw[:_n],
                _actions_arr, _values_arr, _probs_arr,
                int(_df_enabled), float(_df_q_w), float(_df_p_s), float(_df_min), float(_df_slope),
            )

            # Pre-extract Python scalars from numpy arrays (batch conversion
            # is cheaper than per-element conversion in the loop).
            _c_ply_list = c_ply.tolist()
            _c_pov_list = c_pov.tolist()
            _c_priority_list = c_priority.tolist()
            _c_keep_list = c_keep.tolist()
            _c_over_list = c_over.tolist()
            _is_full_list = is_full.tolist()
            _sw_list = sample_weights if isinstance(sample_weights, list) else list(sample_weights)
            _act_list = _actions_arr.tolist()

            for j in range(_n):
                idx = net_idxs[j]
                move_idx_history[idx].append(_act_list[j])
                last_net_full[idx] = _is_full_list[j]

                samples_per_game[idx].append(
                    _NetRecord(
                        c_x[j], c_probs[j], c_wdl_net[j], c_wdl_search[j],
                        chess.WHITE if _c_pov_list[j] else chess.BLACK,
                        _c_ply_list[j], _is_full_list[j],
                        _c_priority_list[j], _sw_list[j], _c_keep_list[j],
                        c_mask[j],
                    )
                )

                if _c_over_list[j]:
                    _done_arr[idx] = 1

        else:
            # Python fallback (original per-ply loop)
            for j, (idx, probs, a, v) in enumerate(zip(net_idxs, probs_list, actions, values_list, strict=True)):
                assert probs is not None and a is not None and v is not None

                board_before = boards[idx]
                ply_index = int(len(board_before.move_stack))
                pov_color = board_before.turn

                mask = masks_list[j]
                if mask is None:
                    mask = legal_move_mask(board_before)

                np.copyto(_lg_buf, pol_logits[j])
                _lg_buf[~mask] = -1e9
                _lg_buf -= float(np.max(_lg_buf))
                np.exp(_lg_buf, out=_lg_buf)
                _lg_buf[~mask] = 0.0
                s = float(_lg_buf.sum())
                if s > 0:
                    raw = (_lg_buf / s).astype(np.float32, copy=False)
                else:
                    raw = mask.astype(np.float32) / float(mask.sum())

                imp = np.maximum(probs.astype(np.float32, copy=False), 1e-12)
                raw_c = np.maximum(raw, 1e-12)
                kl = float(np.sum(raw_c * (np.log(raw_c) - np.log(imp))))

                orig_q = float(wdl_est[j][0] - wdl_est[j][2])
                best_q = float(v)
                q_surprise = abs(best_q - orig_q)

                difficulty = q_surprise * _df_q_w + kl * _df_p_s
                if not math.isfinite(difficulty):
                    difficulty = 1.0
                if not _df_enabled:
                    keep_prob = 1.0
                else:
                    keep_prob = max(_df_min, min(1.0, difficulty * _df_slope))

                move = index_to_move(int(a), board_before)
                board_before.push(move)
                cboards[idx].push_index(int(a))
                move_idx_history[idx].append(int(a))

                d_raw = float(wdl_est[j][1])
                rem = max(0.0, 1.0 - d_raw)
                q = float(max(-rem, min(rem, best_q)))
                w_search = 0.5 * (rem + q)
                _swdl_buf[0] = w_search
                _swdl_buf[1] = d_raw
                _swdl_buf[2] = rem - w_search
                search_wdl_est = _swdl_buf.copy()
                if not np.all(np.isfinite(search_wdl_est)):
                    search_wdl_est = np.array([0.0, 1.0, 0.0], dtype=np.float32)

                last_net_full[idx] = bool(is_full[j])

                # `not _has_c_ply` ⇒ `_batch_enc_146 is None` (coupled import above)
                # ⇒ `_use_inplace is False` ⇒ `xs_batch` was set in the else-branch.
                samples_per_game[idx].append(
                    _NetRecord(
                        x=xs_batch[j],  # pyright: ignore[reportPossiblyUnboundVariable]
                        policy_probs=probs,
                        net_wdl_est=wdl_est[j] if np.all(np.isfinite(wdl_est[j])) else np.array([0.0, 1.0, 0.0], dtype=np.float32),
                        search_wdl_est=search_wdl_est,
                        pov_color=pov_color,
                        ply_index=ply_index,
                        has_policy=bool(is_full[j]),
                        priority=float(difficulty),
                        sample_weight=float(sample_weights[j]),
                        keep_prob=float(keep_prob),
                        legal_mask=mask.view(np.uint8),
                    )
                )

                if cboards[idx].is_game_over():
                    _done_arr[idx] = 1

    # ── Stockfish annotation + opponent moves ─────────────────────────────────

    def _eff_nodes(idx: int) -> int | None:
        if base_nodes <= 0:
            return None
        fast_scale = 1.0 if bool(last_net_full[idx]) else 0.25
        return max(1, int(round(float(base_nodes) * float(fast_scale))))

    def _submit_sf_queries(idxs: list[int]) -> dict[int, Any]:
        """Submit SF queries to pool without blocking. Returns futures dict.

        Only valid when ``stockfish`` is a ``StockfishPool`` — callers guard with isinstance.
        """
        assert isinstance(stockfish, StockfishPool)
        return {idx: stockfish.submit(cboards[idx].fen(), nodes=_eff_nodes(idx)) for idx in idxs}

    def _finish_sf_annotation_and_moves(
        idxs: list[int], *, play_curriculum_moves: bool,
        futures: dict[int, Any] | None = None,
    ) -> None:
        """Collect SF results (from futures or synchronous) and process."""
        if not idxs:
            return
        if futures is not None:
            results = {idx: futures[idx].result() for idx in idxs if idx in futures}
        elif isinstance(stockfish, StockfishPool):
            futs = {idx: stockfish.submit(cboards[idx].fen(), nodes=_eff_nodes(idx)) for idx in idxs}
            results = {idx: fut.result() for idx, fut in futs.items()}
        else:
            results = {idx: stockfish.search(cboards[idx].fen(), nodes=_eff_nodes(idx)) for idx in idxs}
        _process_sf_results(idxs, results=results, play_curriculum_moves=play_curriculum_moves)

    def _process_sf_results(
        idxs: list[int], *, results: dict, play_curriculum_moves: bool,
    ) -> None:
        if not idxs:
            return

        sf_policy_temp_local = float(game.sf_policy_temp)
        sf_policy_label_smooth_local = float(game.sf_policy_label_smooth)

        def _softmax_np(x: np.ndarray) -> np.ndarray:
            z = x.astype(np.float64, copy=False)
            z = z - float(np.max(z))
            e = np.exp(z)
            s = float(e.sum())
            if s <= 0:
                return np.full_like(z, 1.0 / float(z.size))
            return e / s

        def _flip_wdl_pov(wdl: np.ndarray) -> np.ndarray:
            wdl = np.asarray(wdl, dtype=np.float32)
            if wdl.shape != (3,):
                return wdl.astype(np.float32, copy=False)
            return np.array([float(wdl[2]), float(wdl[1]), float(wdl[0])], dtype=np.float32)

        rand_p = float(opponent.random_move_prob)
        curriculum_topk_max = max(1, int(getattr(stockfish, "multipv", 1) or 1))
        curriculum_topk = _effective_curriculum_topk(
            random_move_prob=rand_p,
            stage_end=float(opponent.topk_stage_end),
            topk_max=curriculum_topk_max,
            topk_min=int(opponent.topk_min),
        )
        if opponent.wdl_regret_limit is not None:
            regret_limit = float(opponent.wdl_regret_limit)
        else:
            regret_limit = _effective_curriculum_wdl_regret(
                random_move_prob=rand_p,
                random_move_prob_start=float(opponent.random_move_prob_start),
                random_move_prob_floor=float(opponent.random_move_prob_min),
                regret_max=float(opponent.suboptimal_wdl_regret_max),
                regret_min=float(opponent.suboptimal_wdl_regret_min),
            )

        for idx in idxs:
            res = results[idx]
            legal_indices = cboards[idx].legal_move_indices()
            if legal_indices.size == 0:
                _done_arr[idx] = 1
                continue

            _turn = bool(cboards[idx].turn)
            legal_set = set(int(x) for x in legal_indices)

            a_idx = uci_to_policy_index(res.bestmove_uci, _turn)
            if a_idx < 0 or a_idx not in legal_set:
                a_idx = int(legal_indices[0])

            cand_idxs: list[int] = []
            cand_scores: list[float] = []
            if getattr(res, "pvs", None):
                for pv in res.pvs:
                    if pv.wdl is None:
                        continue
                    a = uci_to_policy_index(pv.move_uci, _turn)
                    if a < 0 or a not in legal_set:
                        continue
                    w_sf, d_sf = float(pv.wdl[0]), float(pv.wdl[1])
                    cand_idxs.append(a)
                    cand_scores.append(w_sf + 0.5 * d_sf)

            if not cand_idxs:
                cand_idxs = [a_idx]
                cand_scores = [0.0]

            scores = np.array(cand_scores, dtype=np.float64) / max(1e-6, sf_policy_temp_local)
            p_top = _softmax_np(scores).astype(np.float32, copy=False)

            p_sf = np.zeros((POLICY_SIZE,), dtype=np.float32)
            for a, p in zip(cand_idxs, p_top, strict=False):
                p_sf[int(a)] += float(p)

            if sf_policy_label_smooth_local > 0.0:
                n_legal = legal_indices.size
                if n_legal > 0:
                    p_sf *= (1.0 - sf_policy_label_smooth_local)
                    p_sf[legal_indices] += sf_policy_label_smooth_local / float(n_legal)

            ps = float(p_sf.sum())
            if ps > 0:
                p_sf /= ps

            if samples_per_game[idx]:
                rec = samples_per_game[idx][-1]
                if rec.sf_policy_target is None and rec.sf_move_index is None:
                    rec.sf_policy_target = p_sf
                    rec.sf_move_index = a_idx
                    if res.wdl is not None:
                        rec.sf_wdl = _flip_wdl_pov(res.wdl)

            if not play_curriculum_moves or _selfplay_arr[idx]:
                continue

            _opp_move_idx = _choose_curriculum_opponent_move(
                rng=rng,
                legal_indices=legal_indices,
                cand_indices=cand_idxs,
                cand_scores=cand_scores,
                curriculum_topk=int(curriculum_topk),
                random_move_prob=rand_p,
                regret_limit=regret_limit,
            )

            cboards[idx].push_index(_opp_move_idx)
            move_idx_history[idx].append(_opp_move_idx)
            # Advance tree root through opponent's move
            if _mcts_tree is not None and _root_ids[idx] >= 0:
                _root_ids[idx] = _mcts_tree.find_child(_root_ids[idx], _opp_move_idx)
            if cboards[idx].is_game_over():
                _done_arr[idx] = 1

    # ── Main game loop (rolling batch) ────────────────────────────────────────
    if continuous:
        max_steps = 2**62  # effectively infinite; stop_fn controls exit
    else:
        max_steps = int(target) * (int(game.max_plies) // 2 + 2)  # safety bound
    _t_net = 0.0
    _t_sf = 0.0

    for _step in range(max_steps):  # skylos: ignore (_step loop var unused by convention)
        # Allow caller to update model/evaluator between moves.
        if on_step is not None:
            on_step()
        if stop_fn is not None and stop_fn():
            break

        # C path: classify games + check game-over with GIL released.
        if _has_classify_c:
            _c_net_arr, _c_sp_arr, _c_cur_arr = _c_classify(  # pyright: ignore[reportPossiblyUnboundVariable]
                cboards, _net_color_arr, _done_arr, _finalized_arr,
                _selfplay_arr, int(game.max_plies),
            )
            net_idxs = _c_net_arr.tolist()
            selfplay_opp_idxs = _c_sp_arr.tolist()
            curriculum_opp_idxs = _c_cur_arr.tolist()
            if not net_idxs and not selfplay_opp_idxs and not curriculum_opp_idxs:
                if not np.any(_finalized_arr == 0):
                    break
        else:
            active_idxs = [i for i in range(batch_size) if not _finalized_arr[i]]
            if not active_idxs:
                break
            for i in active_idxs:
                if not _done_arr[i] and (cboards[i].is_game_over() or cboards[i].ply >= int(game.max_plies)):
                    _done_arr[i] = 1
            active_idxs = [i for i in range(batch_size) if not _finalized_arr[i] and not _done_arr[i]]
            net_idxs = [i for i in active_idxs if cboards[i].turn == _net_color(_net_color_arr, i)]
            selfplay_opp_idxs = [i for i in active_idxs if cboards[i].turn != _net_color(_net_color_arr, i) and _selfplay_arr[i]]
            curriculum_opp_idxs = [i for i in active_idxs if cboards[i].turn != _net_color(_net_color_arr, i) and not _selfplay_arr[i]]

        # Submit SF queries for curriculum games FIRST — curriculum boards
        # are disjoint from net/selfplay boards, so we can overlap SF I/O
        # with the full combined network turn below.
        _sf_futures: dict[int, Any] | None = None
        if curriculum_opp_idxs and isinstance(stockfish, StockfishPool):
            _t0 = time.time()
            _sf_futures = _submit_sf_queries(curriculum_opp_idxs)
            _t_sf += time.time() - _t0

        # Combined network turn: merge net_idxs + selfplay_opp_idxs into a
        # single MCTS call.  This doubles the GPU batch size (256 vs 128),
        # giving much better GPU utilization and pipeline overlap.  Both
        # sets are disjoint and independent — same MCTS treatment, same
        # per-ply processing.
        _combined_net_idxs = net_idxs + selfplay_opp_idxs
        if _combined_net_idxs:
            _t0 = time.time()
            _run_network_turn(_combined_net_idxs)
            _t_net += time.time() - _t0

        # Submit selfplay SF queries immediately after network turn
        # (boards now have the move pushed).  These run in the SF pool
        # while we collect curriculum results below.
        _sf_sp_futures: dict[int, Any] | None = None
        if selfplay_opp_idxs and isinstance(stockfish, StockfishPool):
            _t0 = time.time()
            _sf_sp_futures = _submit_sf_queries(selfplay_opp_idxs)
            _t_sf += time.time() - _t0

        # Collect curriculum SF results (overlapped with combined net turn)
        if curriculum_opp_idxs:
            _t0 = time.time()
            _finish_sf_annotation_and_moves(curriculum_opp_idxs, play_curriculum_moves=True, futures=_sf_futures)
            _t_sf += time.time() - _t0

        # Collect selfplay SF move results (submitted above, overlapped with curriculum)
        if selfplay_opp_idxs:
            _t0 = time.time()
            _finish_sf_annotation_and_moves(selfplay_opp_idxs, play_curriculum_moves=True, futures=_sf_sp_futures)
            _t_sf += time.time() - _t0
            # Submit label queries immediately; collect after (overlaps with
            # finalization below for negligible extra latency).
            selfplay_label_idxs = [i for i in selfplay_opp_idxs if not _done_arr[i]]
            _sf_label_futures: dict[int, Any] | None = None
            if selfplay_label_idxs and isinstance(stockfish, StockfishPool):
                _t0 = time.time()
                _sf_label_futures = _submit_sf_queries(selfplay_label_idxs)
                _t_sf += time.time() - _t0
            if selfplay_label_idxs:
                _t0 = time.time()
                _finish_sf_annotation_and_moves(selfplay_label_idxs, play_curriculum_moves=False, futures=_sf_label_futures)
                _t_sf += time.time() - _t0

        # Finalize completed games and optionally recycle slots
        for i in range(batch_size):
            if _done_arr[i] and not _finalized_arr[i]:
                _finalize_game(i)
                _finalized_arr[i] = 1
                games_completed += 1
                if continuous or games_started < target:
                    _recycle_slot(i)

        # Reset tree when it gets too large and no roots reference old nodes
        if _mcts_tree is not None and _mcts_tree.node_count() > 500_000:
            if all(rid < 0 for rid in _root_ids):
                _mcts_tree.reset()

    # ── Timing summary ─────────────────────────────────────────────────────────
    logging.getLogger("chess_anti_engine.worker").info(
        "play_batch timing: net=%.1fs sf=%.1fs (net %.0f%%, sf %.0f%%)",
        _t_net, _t_sf,
        _t_net / max(0.001, _t_net + _t_sf) * 100,
        _t_sf / max(0.001, _t_net + _t_sf) * 100,
    )

    # ── Return results ────────────────────────────────────────────────────────
    sf_nodes = int(getattr(stockfish, "nodes", 0) or 0)
    skill_lvl = getattr(stockfish, "skill_level", None)
    skill_lvl_i = None if skill_lvl is None else int(skill_lvl)

    mean_sf_d6 = float(_st_sf_d6_sum / max(1, _st_sf_d6_n)) if _st_sf_d6_n > 0 else 0.0
    return all_samples, BatchStats(
        games=int(games_completed),
        positions=len(all_samples),
        w=_st_w,
        d=_st_d,
        l=_st_l,
        total_game_plies=int(_st_game_plies),
        adjudicated_games=int(_st_adjudicated),
        total_draw_games=int(_st_draw),
        selfplay_games=int(_st_sp_games),
        selfplay_adjudicated_games=int(_st_sp_adj),
        selfplay_draw_games=int(_st_sp_draw),
        curriculum_games=int(_st_cur_games),
        curriculum_adjudicated_games=int(_st_cur_adj),
        curriculum_draw_games=int(_st_cur_draw),
        checkmate_games=int(_st_checkmate),
        stalemate_games=int(_st_stalemate),
        plies_win=int(_st_plies_win),
        plies_draw=int(_st_plies_draw),
        plies_loss=int(_st_plies_loss),
        sf_nodes=sf_nodes if sf_nodes > 0 else None,
        sf_nodes_next=None,
        pid_ema_winrate=None,
        random_move_prob=float(opponent.random_move_prob),
        skill_level=skill_lvl_i,
        sf_eval_delta6=mean_sf_d6,
        sf_eval_delta6_n=int(_st_sf_d6_n),
    )
