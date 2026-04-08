from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

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
from chess_anti_engine.encoding import encode_position, encode_positions_batch
from chess_anti_engine.moves import POLICY_SIZE, move_to_index, index_to_move, legal_move_mask
from chess_anti_engine.moves.encode import legal_move_indices
from chess_anti_engine.train.targets import DEFAULT_CATEGORICAL_BINS, hlgauss_target
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
    legal_moves: list[chess.Move],
    cand_moves: list[chess.Move],
    cand_scores: list[float],
    curriculum_topk: int,
    random_move_prob: float,
    regret_limit: float,
) -> chess.Move:
    """Choose the curriculum opponent move from Stockfish candidates.

    Both finite and infinite regret paths share the same structure:
    - `random_move_prob` chance of a truly random legal move (blunder corruption)
    - Otherwise, uniform pick among acceptable moves (top-k for infinite regret,
      regret-filtered top-k for finite regret)
    """
    if not cand_moves:
        return legal_moves[int(rng.integers(len(legal_moves)))]

    topk_n = max(1, min(int(curriculum_topk), len(cand_moves)))
    topk_moves = cand_moves[:topk_n]
    topk_scores = cand_scores[:topk_n]
    rand_p = max(0.0, float(random_move_prob))

    if not math.isfinite(float(regret_limit)):
        if rand_p > 0.0 and rng.random() < rand_p:
            return legal_moves[int(rng.integers(len(legal_moves)))]
        return topk_moves[int(rng.integers(len(topk_moves)))]

    # Candidates are in Stockfish PV order (by centipawn eval), NOT by WDL
    # score.  Use the max WDL score in the top-k as the reference for regret.
    best_score = max(float(s) for s in topk_scores)
    acceptable_moves = [
        mv
        for mv, score in zip(topk_moves, topk_scores, strict=False)
        if (best_score - float(score)) <= float(regret_limit) + 1e-12
    ]
    if not acceptable_moves:
        acceptable_moves = [topk_moves[0]]
    # Random corruption: truly random legal move (like infinite-regret path)
    if rand_p > 0.0 and rng.random() < rand_p:
        return legal_moves[int(rng.integers(len(legal_moves)))]
    # Non-random: uniform among acceptable moves (not always best)
    return acceptable_moves[int(rng.integers(len(acceptable_moves)))]


def _is_network_turn(*, board_turn: chess.Color, network_color: chess.Color) -> bool:
    """Return True when the side to move is the network-assigned color."""
    return board_turn == network_color


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


@dataclass
class GameBatchCollector:
    games: list[CompletedGameBatch] = field(default_factory=list)

    def on_game_complete(self, batch: CompletedGameBatch) -> None:
        self.games.append(batch)


@dataclass
class _NetRecord:
    x: np.ndarray
    policy_probs: np.ndarray
    net_wdl_est: np.ndarray
    search_wdl_est: np.ndarray
    pov_color: chess.Color
    ply_index: int

    # sampling / curriculum
    has_policy: bool
    priority: float
    sample_weight: float
    keep_prob: float

    # LC0-style: legal move mask at the position, for masked-softmax during training.
    legal_mask: np.ndarray | None = None  # (POLICY_SIZE,) uint8

    # SF reply targets (optional; absent when game ends after network move)
    sf_policy_target: np.ndarray | None = None
    sf_move_index: int | None = None
    sf_wdl: np.ndarray | None = None


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

    When target_games > 0, finished game slots are recycled with fresh games
    until target_games have been completed. This keeps the inference batch at
    full capacity throughout.  When 0 (default), runs exactly `games` games
    with no replenishment (backward-compatible behavior).
    """

    requested_batch = int(games)
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
    # Keep a copy of the starting position for tablebase replay after game ends.
    starting_boards = [b.copy() for b in boards] if game.syzygy_path else None
    done = [False] * batch_size

    # Alternate which color the network plays so it sees both perspectives.
    network_color: list[chess.Color] = [
        chess.WHITE if (i % 2 == 0) else chess.BLACK for i in range(batch_size)
    ]
    selfplay_game: list[bool] = [
        bool(rng.random() < max(0.0, min(1.0, float(game.selfplay_fraction))))
        for _ in range(batch_size)
    ]

    samples_per_game: list[list[_NetRecord]] = [[] for _ in range(batch_size)]

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
    _st_sf_d6_sum = 0.0
    _st_sf_d6_n = 0

    base_nodes = int(getattr(stockfish, "nodes", 0) or 0)
    terminal_eval_nodes = (5 * base_nodes) if base_nodes > 0 else 1000

    vs = str(game.volatility_source).lower().strip()
    if vs not in ("raw", "search"):
        vs = "raw"

    def _sf_terminal_result(board: chess.Board, sf_res: StockfishResult | None) -> str:
        if sf_res is None or sf_res.wdl is None:
            return "1/2-1/2"
        wdl_stm = sf_res.wdl
        if board.turn == chess.BLACK:
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
        nonlocal _st_sf_d6_sum, _st_sf_d6_n

        b = boards[i]
        result = b.result(claim_draw=True)
        _st_game_plies += int(len(b.move_stack))

        was_adjudicated = False
        if result == "*":
            _st_adjudicated += 1
            was_adjudicated = True
            try:
                if isinstance(stockfish, StockfishPool):
                    sf_res = stockfish.submit(b.fen(), nodes=int(terminal_eval_nodes)).result()
                else:
                    sf_res = stockfish.search(b.fen(), nodes=int(terminal_eval_nodes))
            except Exception:
                sf_res = None
            result = _sf_terminal_result(b, sf_res)

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
                _is_sp = bool(selfplay_game[i])
                for mv in move_stack[opening_len:]:
                    board_before = replay_board.copy()
                    is_net = _is_network_turn(board_turn=replay_board.turn, network_color=network_color[i])
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
        if bool(selfplay_game[i]):
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
            if bool(selfplay_game[i]):
                _st_sp_draw += 1
                game_selfplay_draws = 1
            else:
                _st_cur_draw += 1
                game_curriculum_draws = 1

        if not selfplay_game[i]:
            net_col = network_color[i]
            if result == "1/2-1/2":
                _st_d += 1
                game_d = 1
            elif (result == "1-0" and net_col == chess.WHITE) or (result == "0-1" and net_col == chess.BLACK):
                _st_w += 1
                game_w = 1
            else:
                _st_l += 1
                game_l = 1

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

            total_plies_played = max(1, len(b.move_stack))
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
                        total_game_plies=int(len(b.move_stack)),
                        adjudicated_games=1 if was_adjudicated else 0,
                        total_draw_games=game_total_draws,
                        selfplay_games=game_selfplay_games,
                        selfplay_adjudicated_games=game_selfplay_adj,
                        selfplay_draw_games=game_selfplay_draws,
                        curriculum_games=game_curriculum_games,
                        curriculum_adjudicated_games=game_curriculum_adj,
                        curriculum_draw_games=game_curriculum_draws,
                    )
                )

    # ── Slot recycling ────────────────────────────────────────────────────────
    games_started = batch_size
    games_completed = 0
    finalized = [False] * batch_size

    def _recycle_slot(i: int) -> None:
        nonlocal games_started
        boards[i] = make_starting_board(rng=rng, cfg=opening)
        if starting_boards is not None:
            starting_boards[i] = boards[i].copy()
        done[i] = False
        finalized[i] = False
        network_color[i] = chess.WHITE if (games_started % 2 == 0) else chess.BLACK
        selfplay_game[i] = bool(rng.random() < max(0.0, min(1.0, float(game.selfplay_fraction))))
        samples_per_game[i] = []
        consecutive_low_winrate[i] = 0
        sf_resign_scale[i] = 1.0
        last_net_full[i] = True
        games_started += 1

    # ── Network turn ──────────────────────────────────────────────────────────

    def _run_network_turn(net_idxs: list[int]) -> None:
        if not net_idxs:
            return

        xs_batch = encode_positions_batch([boards[i] for i in net_idxs], add_features=True)

        pol_logits, wdl_logits_raw = eval_impl.evaluate_encoded(xs_batch)
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

        probs_list = [None] * len(net_idxs)
        actions = [None] * len(net_idxs)
        values_list = [None] * len(net_idxs)
        masks_list: list[np.ndarray | None] = [None] * len(net_idxs)

        # Per-game temperature based on each game's own ply count.
        temps = [
            temperature_for_ply(
                ply=len(boards[i].move_stack) // 2 + 1,
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
            sub_boards = [boards[net_idxs[j]] for j in group]
            sub_temps = [temps[j] for j in group]
            sub_sims = [int(sims_per[j]) for j in group]
            sim_count = max(sub_sims)

            gumbel_low_sims = max(64, int(search.fast_simulations))
            use_gumbel = (str(search.mcts_type) == "gumbel") or (int(sim_count) <= int(gumbel_low_sims))
            sub_pol = pol_logits[group, :]
            sub_wdl = wdl_logits_raw[group, :]

            if use_gumbel:
                _gumbel_fn = _run_gumbel_root_many_c if _HAS_GUMBEL_C else run_gumbel_root_many
                sub_noise = [per_game_noise[j] for j in group] if per_game_noise is not None else None
                p_sub, a_sub, v_sub, m_sub = _gumbel_fn(
                    model,
                    sub_boards,
                    device=device,
                    rng=rng,
                    cfg=GumbelConfig(simulations=int(sim_count), temperature=1.0, add_noise=True),
                    evaluator=eval_impl,
                    pre_pol_logits=sub_pol,
                    pre_wdl_logits=sub_wdl,
                    per_game_simulations=sub_sims,
                    per_game_add_noise=sub_noise,
                )
            else:
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
                )

            # Re-select actions with per-game temperature from the
            # improved policy (probs are temperature-independent).
            for gi, (p, t) in enumerate(zip(p_sub, sub_temps)):
                if t == 1.0:
                    continue
                legal = np.flatnonzero(p > 0)
                if len(legal) == 0:
                    continue
                if t <= 0:
                    a_sub[gi] = int(legal[np.argmax(p[legal])])
                else:
                    pw = np.power(p[legal], 1.0 / float(t))
                    ps = float(pw.sum())
                    if ps > 0:
                        pw /= ps
                        a_sub[gi] = int(rng.choice(legal, p=pw))

            for jj, p, a, v, m in zip(group, p_sub, a_sub, v_sub, m_sub, strict=True):
                probs_list[jj] = p
                actions[jj] = a
                values_list[jj] = float(v)
                masks_list[jj] = m

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

        for j, (idx, probs, a, v) in enumerate(zip(net_idxs, probs_list, actions, values_list, strict=True)):
            assert probs is not None and a is not None and v is not None

            board_before = boards[idx]
            ply_index = int(len(board_before.move_stack))
            pov_color = board_before.turn

            mask = masks_list[j]
            if mask is None:
                mask = legal_move_mask(board_before)

            # Raw network policy (masked softmax) — reuse buffer
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

            # KL divergence (diff focus)
            imp = np.maximum(probs.astype(np.float32, copy=False), 1e-12)
            raw_c = np.maximum(raw, 1e-12)
            kl = float(np.sum(raw_c * (np.log(raw_c) - np.log(imp))))

            orig_q = float(wdl_est[j][0] - wdl_est[j][2])
            best_q = float(v)
            q_surprise = abs(best_q - orig_q)

            difficulty = q_surprise * _df_q_w + kl * _df_p_s
            if not math.isfinite(difficulty):
                difficulty = 1.0  # safe default — NaN from model output or KL overflow
            if not _df_enabled:
                keep_prob = 1.0
            else:
                keep_prob = max(_df_min, min(1.0, difficulty * _df_slope))

            move = index_to_move(int(a), board_before)
            board_before.push(move)

            # Search WDL estimate — reuse buffer, copy to new array for storage
            d_raw = float(wdl_est[j][1])
            rem = max(0.0, 1.0 - d_raw)
            q = float(max(-rem, min(rem, best_q)))
            w_search = 0.5 * (rem + q)
            _swdl_buf[0] = w_search
            _swdl_buf[1] = d_raw
            _swdl_buf[2] = rem - w_search
            search_wdl_est = _swdl_buf.copy()
            if not np.all(np.isfinite(search_wdl_est)):
                search_wdl_est = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # draw prior

            last_net_full[idx] = bool(is_full[j])

            samples_per_game[idx].append(
                _NetRecord(
                    x=xs_batch[j],
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

            if boards[idx].is_game_over():
                done[idx] = True

    # ── Stockfish annotation + opponent moves ─────────────────────────────────

    def _eff_nodes(idx: int) -> int | None:
        if base_nodes <= 0:
            return None
        fast_scale = 1.0 if bool(last_net_full[idx]) else 0.25
        return max(1, int(round(float(base_nodes) * float(fast_scale))))

    def _submit_sf_queries(idxs: list[int]) -> dict[int, object]:
        """Submit SF queries to pool without blocking. Returns futures dict."""
        return {idx: stockfish.submit(boards[idx].fen(), nodes=_eff_nodes(idx)) for idx in idxs}

    def _finish_sf_annotation_and_moves(
        idxs: list[int], *, play_curriculum_moves: bool,
        futures: dict[int, object] | None = None,
    ) -> None:
        """Collect SF results (from futures or synchronous) and process."""
        if not idxs:
            return
        if futures is not None:
            results = {idx: futures[idx].result() for idx in idxs if idx in futures}
        elif isinstance(stockfish, StockfishPool):
            futs = {idx: stockfish.submit(boards[idx].fen(), nodes=_eff_nodes(idx)) for idx in idxs}
            results = {idx: fut.result() for idx, fut in futs.items()}
        else:
            results = {idx: stockfish.search(boards[idx].fen(), nodes=_eff_nodes(idx)) for idx in idxs}
        _process_sf_results(idxs, results=results, play_curriculum_moves=play_curriculum_moves)

    def _run_sf_annotation_and_moves(idxs: list[int], *, play_curriculum_moves: bool) -> None:
        """Submit + collect SF results in one blocking call."""
        _finish_sf_annotation_and_moves(idxs, play_curriculum_moves=play_curriculum_moves)

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
            legal_moves = list(boards[idx].legal_moves)
            if not legal_moves:
                done[idx] = True
                continue

            sf_move = chess.Move.from_uci(res.bestmove_uci)
            if sf_move not in legal_moves:
                sf_move = legal_moves[0]
            a_idx = int(move_to_index(sf_move, boards[idx]))

            cand_idxs: list[int] = []
            cand_scores: list[float] = []
            cand_moves: list[chess.Move] = []
            if getattr(res, "pvs", None):
                for pv in res.pvs:
                    if pv.wdl is None:
                        continue
                    try:
                        mv = chess.Move.from_uci(pv.move_uci)
                    except Exception:
                        continue
                    if mv not in boards[idx].legal_moves:
                        continue
                    try:
                        a = int(move_to_index(mv, boards[idx]))
                    except Exception:
                        continue
                    w_sf, d_sf = float(pv.wdl[0]), float(pv.wdl[1])
                    cand_idxs.append(a)
                    cand_scores.append(w_sf + 0.5 * d_sf)
                    cand_moves.append(mv)

            if not cand_idxs:
                cand_idxs = [a_idx]
                cand_scores = [0.0]
                cand_moves = [sf_move]

            scores = np.array(cand_scores, dtype=np.float64) / max(1e-6, sf_policy_temp_local)
            p_top = _softmax_np(scores).astype(np.float32, copy=False)

            p_sf = np.zeros((POLICY_SIZE,), dtype=np.float32)
            for a, p in zip(cand_idxs, p_top, strict=False):
                p_sf[int(a)] += float(p)

            if sf_policy_label_smooth_local > 0.0:
                legal_idx = legal_move_indices(boards[idx])
                n_legal = legal_idx.size
                if n_legal > 0:
                    p_sf *= (1.0 - sf_policy_label_smooth_local)
                    p_sf[legal_idx] += sf_policy_label_smooth_local / float(n_legal)

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

            if not play_curriculum_moves or selfplay_game[idx]:
                continue

            move_to_play = _choose_curriculum_opponent_move(
                rng=rng,
                legal_moves=legal_moves,
                cand_moves=cand_moves,
                cand_scores=cand_scores,
                curriculum_topk=int(curriculum_topk),
                random_move_prob=rand_p,
                regret_limit=regret_limit,
            )

            boards[idx].push(move_to_play)
            if boards[idx].is_game_over():
                done[idx] = True

    # ── Main game loop (rolling batch) ────────────────────────────────────────
    max_steps = int(target) * (int(game.max_plies) // 2 + 1)  # safety bound
    _t_net = 0.0
    _t_sf = 0.0

    for _step in range(max_steps):
        active_idxs = [i for i in range(batch_size) if not finalized[i]]
        if not active_idxs:
            break

        # Mark games that hit max_plies or are over
        for i in active_idxs:
            if not done[i] and (boards[i].is_game_over() or len(boards[i].move_stack) >= int(game.max_plies)):
                done[i] = True

        active_idxs = [i for i in range(batch_size) if not finalized[i] and not done[i]]
        if not active_idxs:
            # All active games are done — finalize them below, then check if we need more
            pass

        net_idxs = [i for i in active_idxs if _is_network_turn(board_turn=boards[i].turn, network_color=network_color[i])]
        if net_idxs:
            _t0 = time.time()
            _run_network_turn(net_idxs)
            _t_net += time.time() - _t0

        all_opp_idxs = [i for i in active_idxs if (not done[i]) and boards[i].turn != network_color[i]]

        # Submit SF queries asynchronously (CPU), then overlap with
        # selfplay network turns while SF processes run.
        _sf_futures: dict[int, object] | None = None
        if all_opp_idxs and isinstance(stockfish, StockfishPool):
            _t0 = time.time()
            _sf_futures = _submit_sf_queries(all_opp_idxs)
            _t_sf += time.time() - _t0

        # While SF is running, do the selfplay opponent network turn
        selfplay_opp_idxs = [
            i for i in all_opp_idxs
            if selfplay_game[i] and (not done[i]) and boards[i].turn != network_color[i]
        ]
        if selfplay_opp_idxs:
            _t0 = time.time()
            _run_network_turn(selfplay_opp_idxs)
            _t_net += time.time() - _t0

        # Collect SF results and process them
        if all_opp_idxs:
            _t0 = time.time()
            _finish_sf_annotation_and_moves(all_opp_idxs, play_curriculum_moves=True, futures=_sf_futures)
            _t_sf += time.time() - _t0

        # SF annotation for selfplay label (no move playing)
        if selfplay_opp_idxs:
            selfplay_label_idxs = [i for i in selfplay_opp_idxs if not done[i]]
            if selfplay_label_idxs:
                _t0 = time.time()
                _run_sf_annotation_and_moves(selfplay_label_idxs, play_curriculum_moves=False)
                _t_sf += time.time() - _t0

        # Finalize completed games and optionally recycle slots
        for i in range(batch_size):
            if done[i] and not finalized[i]:
                _finalize_game(i)
                finalized[i] = True
                games_completed += 1
                if games_started < target:
                    _recycle_slot(i)

    # ── Timing summary ─────────────────────────────────────────────────────────
    import logging as _logging
    _logging.getLogger("chess_anti_engine.worker").info(
        "play_batch timing: net=%.1fs sf=%.1fs other=%.1fs (net %.0f%%, sf %.0f%%)",
        _t_net, _t_sf, max(0, batch_elapsed - _t_net - _t_sf) if (batch_elapsed := _t_net + _t_sf) else 0,
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
        sf_nodes=sf_nodes if sf_nodes > 0 else None,
        sf_nodes_next=None,
        pid_ema_winrate=None,
        random_move_prob=float(opponent.random_move_prob),
        skill_level=skill_lvl_i,
        sf_eval_delta6=mean_sf_d6,
        sf_eval_delta6_n=int(_st_sf_d6_n),
    )
