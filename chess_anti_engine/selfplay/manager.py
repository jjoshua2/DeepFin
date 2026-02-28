from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.utils.amp import inference_autocast
from chess_anti_engine.stockfish.uci import StockfishUCI, StockfishResult
from chess_anti_engine.stockfish.pool import StockfishPool
from chess_anti_engine.mcts import MCTSConfig, GumbelConfig
from chess_anti_engine.mcts.puct import run_mcts_many
from chess_anti_engine.mcts.gumbel import run_gumbel_root_many
from chess_anti_engine.encoding import encode_position
from chess_anti_engine.moves import POLICY_SIZE, move_to_index, index_to_move, legal_move_mask
from chess_anti_engine.train.targets import hlgauss_target
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
class _NetRecord:
    x: np.ndarray
    policy_probs: np.ndarray
    net_wdl_est: np.ndarray
    search_wdl_est: np.ndarray

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
    model: torch.nn.Module,
    *,
    device: str,
    rng: np.random.Generator,
    stockfish: StockfishUCI | StockfishPool,
    games: int,
    temperature: float,
    opponent_random_move_prob: float = 0.0,
    temperature_drop_plies: int = 0,
    temperature_after: float = 0.0,
    temperature_decay_start_move: int = 20,
    temperature_decay_moves: int = 60,
    temperature_endgame: float = 0.6,
    max_plies: int,
    mcts_simulations: int = 50,
    mcts_type: str = "puct",
    playout_cap_fraction: float = 0.25,
    fast_simulations: int = 8,
    sf_policy_temp: float = 0.25,
    sf_policy_label_smooth: float = 0.05,
    opening_book_path: str | None = None,
    opening_book_max_plies: int = 4,
    opening_book_max_games: int = 200_000,
    opening_book_prob: float = 1.0,
    random_start_plies: int = 0,
    syzygy_path: str | None = None,
    syzygy_policy: bool = False,
    # Volatility target definition (ablation):
    # - "raw": use the network's raw WDL head output at time t
    # - "search": use a search-adjusted WDL distribution derived from the root search value
    volatility_source: str = "raw",
    # LC0-style diff focus (probabilistic skip of easy positions)
    diff_focus_enabled: bool = True,
    diff_focus_q_weight: float = 6.0,
    diff_focus_pol_scale: float = 3.5,
    diff_focus_slope: float = 3.0,
    diff_focus_min: float = 0.025,
    # Categorical value head settings (tunable for Ray Tune ablations)
    categorical_bins: int = 32,
    hlgauss_sigma: float = 0.04,
    # FPU (First Play Urgency) reduction for PUCT MCTS
    fpu_reduction: float = 1.2,
    fpu_at_root: float = 1.0,
) -> tuple[list[ReplaySample], BatchStats]:
    """Play a batch of games.

    Design goals:
    - keep GPU busy via batched inference
    - keep SF queries minimal (one per SF ply)
    - compute volatility targets from a consistent network-side WDL series without per-ply overhead
    """

    op_cfg = OpeningConfig(
        opening_book_path=opening_book_path,
        opening_book_max_plies=int(opening_book_max_plies),
        opening_book_max_games=int(opening_book_max_games),
        opening_book_prob=float(opening_book_prob),
        random_start_plies=int(random_start_plies),
    )
    boards = [make_starting_board(rng=rng, cfg=op_cfg) for _ in range(int(games))]
    # Keep a copy of the starting position for tablebase replay after game ends.
    starting_boards = [b.copy() for b in boards] if syzygy_path else None
    done = [False] * int(games)

    # Alternate which color the network plays so it sees both perspectives.
    # The encoding is always side-to-move relative, so the model is symmetric;
    # this just ensures balanced position diversity.
    network_color: list[chess.Color] = [
        chess.WHITE if (i % 2 == 0) else chess.BLACK for i in range(int(games))
    ]

    # Per-game list of network-turn samples.
    #
    # Each record corresponds to ONE "move pair":
    #   (network move, then Stockfish reply move)
    # and is trained from the network-turn position.
    samples_per_game: list[list[_NetRecord]] = [[] for _ in range(int(games))]

    # Soft resignation state per game: count of consecutive moves where
    # network's MCTS win probability < 5%.
    SOFT_RESIGN_THRESHOLD = 0.05
    SOFT_RESIGN_CONSECUTIVE = 5
    consecutive_low_winrate: list[int] = [0] * int(games)
    # Per-game scale factor for SF node budget during soft resignation playthrough.
    sf_resign_scale: list[float] = [1.0] * int(games)

    for _ply in range(int(max_plies)):
        active_idxs = [i for i in range(int(games)) if not done[i]]
        if not active_idxs:
            break

        for i in active_idxs:
            if boards[i].is_game_over():
                done[i] = True

        active_idxs = [i for i in range(int(games)) if not done[i]]
        if not active_idxs:
            break

        # Network turns: whichever color the network was assigned for this game
        net_idxs = [i for i in active_idxs if _is_network_turn(board_turn=boards[i].turn, network_color=network_color[i])]
        if net_idxs:
            xs = [encode_position(boards[i], add_features=True) for i in net_idxs]

            # Batch-evaluate raw policy + WDL at the root.
            # We reuse the WDL output to compute volatility targets without extra inference passes.
            with torch.no_grad():
                xt = torch.from_numpy(np.stack(xs, axis=0)).to(device)
                with inference_autocast(device=device, enabled=True, dtype="auto"):
                    out = model(xt)
                policy_out = out["policy"] if "policy" in out else out["policy_own"]
                pol_logits = policy_out.detach().float().cpu().numpy()
                wdl_logits_raw = out["wdl"].detach().float().cpu().numpy()  # (B, 3) pre-softmax
                wdl_est = (
                    torch.softmax(out["wdl"].detach().float(), dim=-1)
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

            is_full = rng.random(size=len(net_idxs)) < float(playout_cap_fraction)

            # Soft resignation: compute effective simulation counts per game.
            # When winrate < 5% for SOFT_RESIGN_CONSECUTIVE moves, reduce playouts proportionally.
            eff_full_sims = [int(mcts_simulations)] * len(net_idxs)
            eff_fast_sims = [int(fast_simulations)] * len(net_idxs)
            sample_weights = [1.0] * len(net_idxs)
            for j, idx in enumerate(net_idxs):
                win_p = float(wdl_est[j][0])  # W component from network WDL
                if win_p < SOFT_RESIGN_THRESHOLD:
                    consecutive_low_winrate[idx] += 1
                else:
                    consecutive_low_winrate[idx] = 0

                if consecutive_low_winrate[idx] >= SOFT_RESIGN_CONSECUTIVE:
                    ratio = win_p / SOFT_RESIGN_THRESHOLD  # 0..1
                    eff_full_sims[j] = max(2, int(fast_simulations + (mcts_simulations - fast_simulations) * ratio))
                    eff_fast_sims[j] = max(2, int(2 + (fast_simulations - 2) * ratio))
                    sample_weights[j] = 0.1 + 0.9 * ratio

                    # Keep a roughly constant SF-vs-network effort ratio by scaling SF nodes
                    # similarly to how we scale the network's full-search budget.
                    if int(mcts_simulations) > 0:
                        min_scale = float(fast_simulations) / float(mcts_simulations)
                        sf_resign_scale[idx] = float(min_scale + (1.0 - min_scale) * ratio)

                else:
                    sf_resign_scale[idx] = 1.0

            probs_list = [None] * len(net_idxs)
            actions = [None] * len(net_idxs)
            values_list = [None] * len(net_idxs)

            # Note: _ply counts full moves in this selfplay loop (we do white then black each iteration).
            move_number = int(_ply) + 1  # 1-based, matches LC0-style flag semantics
            turn_temp = temperature_for_ply(
                ply=move_number,
                temperature=float(temperature),
                drop_plies=int(temperature_drop_plies),
                after=float(temperature_after),
                decay_start_move=int(temperature_decay_start_move),
                decay_moves=int(temperature_decay_moves),
                endgame=float(temperature_endgame),
            )

            def _run_mcts_group(idxs: list[int], sims_per: list[int]) -> None:
                """Run MCTS for a group of games, batching those with identical sim counts."""
                if not idxs:
                    return

                # Group by sim count for batching efficiency.
                from collections import defaultdict

                by_sims: dict[int, list[int]] = defaultdict(list)
                for j in idxs:
                    by_sims[int(sims_per[j])].append(j)

                # If sims are small, prefer Gumbel at the root.
                # (Empirically more stable than PUCT at low budgets.)
                gumbel_low_sims = max(64, int(fast_simulations))

                for sim_count, group in by_sims.items():
                    sub_boards = [boards[net_idxs[j]] for j in group]

                    use_gumbel = (str(mcts_type) == "gumbel") or (int(sim_count) <= int(gumbel_low_sims))
                    if use_gumbel:
                        # Pass pre-computed root logits so gumbel skips a duplicate forward pass.
                        # `group` contains j-values = positions in net_idxs; pol_logits row j == net_idxs[j].
                        sub_pol = pol_logits[group, :]       # (|group|, POLICY_SIZE)
                        sub_wdl = wdl_logits_raw[group, :]   # (|group|, 3)
                        p_sub, a_sub, v_sub = run_gumbel_root_many(
                            model,
                            sub_boards,
                            device=device,
                            rng=rng,
                            cfg=GumbelConfig(simulations=int(sim_count), temperature=float(turn_temp)),
                            pre_pol_logits=sub_pol,
                            pre_wdl_logits=sub_wdl,
                        )
                    else:
                        p_sub, a_sub, v_sub = run_mcts_many(
                            model,
                            sub_boards,
                            device=device,
                            rng=rng,
                            cfg=MCTSConfig(simulations=int(sim_count), temperature=float(turn_temp),
                                          fpu_reduction=float(fpu_reduction), fpu_at_root=float(fpu_at_root)),
                        )

                    for jj, p, a, v in zip(group, p_sub, a_sub, v_sub, strict=True):
                        probs_list[jj] = p
                        actions[jj] = a
                        values_list[jj] = float(v)

            # Full search subset
            full_idxs = [j for j, v in enumerate(is_full) if bool(v)]
            _run_mcts_group(full_idxs, eff_full_sims)

            # Fast search subset (used for playing but not for policy targets)
            fast_idxs = [j for j, v in enumerate(is_full) if not bool(v)]
            _run_mcts_group(fast_idxs, eff_fast_sims)

            for j, (idx, probs, a, v) in enumerate(zip(net_idxs, probs_list, actions, values_list, strict=True)):
                assert probs is not None and a is not None and v is not None

                # Raw policy distribution for surprise priority (masked softmax)
                mask = legal_move_mask(boards[idx])
                lg = pol_logits[j].astype(np.float64, copy=True)
                lg[~mask] = -1e9
                lg = lg - float(np.max(lg))
                rp = np.exp(lg)
                rp[~mask] = 0.0
                s = float(rp.sum())
                raw = (rp / s).astype(np.float32) if s > 0 else (mask.astype(np.float32) / float(mask.sum()))

                imp = np.maximum(probs.astype(np.float32, copy=False), 1e-12)
                raw_c = np.maximum(raw, 1e-12)
                kl = float(np.sum(raw_c * (np.log(raw_c) - np.log(imp))))

                # LC0-style diff focus difficulty signal
                orig_q = float(wdl_est[j][0] - wdl_est[j][2])
                best_q = float(v)
                q_surprise = abs(best_q - orig_q)
                pol_surprise = float(kl)

                difficulty = q_surprise * float(diff_focus_q_weight) + pol_surprise * float(diff_focus_pol_scale)
                if not bool(diff_focus_enabled):
                    keep_prob = 1.0
                else:
                    keep_prob = float(difficulty) * float(diff_focus_slope)
                    keep_prob = max(float(diff_focus_min), min(1.0, keep_prob))

                move = index_to_move(int(a), boards[idx])
                boards[idx].push(move)

                # Store keep_prob for post-hoc probabilistic skip (skip removes BOTH value and policy signal).
                # Store difficulty as sampling priority (surprise weighting).
                # For volatility-target ablations, also store a search-adjusted WDL estimate.
                # We use the raw draw probability but replace (W-L) with the root search value estimate.
                d_raw = float(wdl_est[j][1])
                rem = max(0.0, 1.0 - d_raw)
                q = float(max(-rem, min(rem, best_q)))
                w_search = 0.5 * (rem + q)
                l_search = rem - w_search
                search_wdl_est = np.array([w_search, d_raw, l_search], dtype=np.float32)

                samples_per_game[idx].append(
                    _NetRecord(
                        x=xs[j],
                        policy_probs=probs,
                        net_wdl_est=wdl_est[j],
                        search_wdl_est=search_wdl_est,
                        has_policy=bool(is_full[j]),
                        priority=float(difficulty),
                        sample_weight=float(sample_weights[j]),
                        keep_prob=float(keep_prob),
                        legal_mask=mask.astype(np.uint8),
                    )
                )

                # If the network move ended the game, mark done so we do not query SF.
                if boards[idx].is_game_over():
                    done[idx] = True

        # Stockfish turns: whichever color is NOT the network's.
        # Resigned games still query SF for analysis labels (to keep training the
        # sf_wdl / sf_policy heads) but use the NETWORK to select the actual move
        # so the game is network-vs-network → balanced W/L labels regardless of
        # how bad the model is.
        all_opp_idxs = [i for i in active_idxs if (not done[i]) and boards[i].turn != network_color[i]]
        sf_idxs = all_opp_idxs
        if sf_idxs:
            base_nodes = int(getattr(stockfish, "nodes", 0) or 0)

            def _eff_nodes(idx: int) -> int | None:
                if base_nodes <= 0:
                    return None
                scale = float(sf_resign_scale[idx])
                return max(1, int(round(float(base_nodes) * scale)))

            if isinstance(stockfish, StockfishPool):
                futures = {idx: stockfish.submit(boards[idx].fen(), nodes=_eff_nodes(idx)) for idx in sf_idxs}
                results = {idx: fut.result() for idx, fut in futures.items()}
            else:
                results = {idx: stockfish.search(boards[idx].fen(), nodes=_eff_nodes(idx)) for idx in sf_idxs}

            sf_policy_temp = float(sf_policy_temp)
            sf_policy_label_smooth = float(sf_policy_label_smooth)

            def _softmax_np(x: np.ndarray) -> np.ndarray:
                z = x.astype(np.float64, copy=False)
                z = z - float(np.max(z))
                e = np.exp(z)
                s = float(e.sum())
                if s <= 0:
                    return np.full_like(z, 1.0 / float(z.size))
                return e / s

            def _flip_wdl_to_white(wdl: np.ndarray) -> np.ndarray:
                """Convert Stockfish WDL (side-to-move POV) to white POV.

                Stockfish UCI_ShowWDL is interpreted as probabilities for the side to move.
                When it's black to move, [W,D,L] is for black, so we swap W/L to get white.
                """
                wdl = np.asarray(wdl, dtype=np.float32)
                if wdl.shape != (3,):
                    return wdl.astype(np.float32, copy=False)
                return np.array([float(wdl[2]), float(wdl[1]), float(wdl[0])], dtype=np.float32)

            rand_p = float(opponent_random_move_prob)

            for idx in sf_idxs:
                res = results[idx]

                legal_moves = list(boards[idx].legal_moves)
                if not legal_moves:
                    done[idx] = True
                    continue

                # Stockfish best-move index (teacher signal / bookkeeping)
                sf_move = chess.Move.from_uci(res.bestmove_uci)
                if sf_move not in legal_moves:
                    sf_move = legal_moves[0]
                a_idx = int(move_to_index(sf_move, boards[idx]))

                # Optionally *play* a random legal move instead of Stockfish's.
                # We still record Stockfish's targets (policy + WDL) for the previous network-turn sample.
                if rand_p > 0.0 and rng.random() < rand_p:
                    move_to_play = legal_moves[int(rng.integers(len(legal_moves)))]
                else:
                    move_to_play = sf_move

                # Soft SF-policy target from MultiPV, scored by winrate-like scalar: W + 0.5*D.
                cand_idxs: list[int] = []
                cand_scores: list[float] = []
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
                        w, d = float(pv.wdl[0]), float(pv.wdl[1])
                        cand_idxs.append(a)
                        cand_scores.append(w + 0.5 * d)

                if not cand_idxs:
                    cand_idxs = [a_idx]
                    cand_scores = [0.0]

                scores = np.array(cand_scores, dtype=np.float64) / max(1e-6, sf_policy_temp)
                p_top = _softmax_np(scores).astype(np.float32, copy=False)

                p_sf = np.zeros((POLICY_SIZE,), dtype=np.float32)
                for a, p in zip(cand_idxs, p_top, strict=False):
                    p_sf[int(a)] += float(p)

                # Label smoothing over legal moves (ensures non-top moves not trained to exact 0).
                mask = legal_move_mask(boards[idx]).astype(np.float32)
                ms = float(mask.sum())
                uniform = (mask / ms) if ms > 0 else (np.ones((POLICY_SIZE,), dtype=np.float32) / float(POLICY_SIZE))
                p_sf = (1.0 - sf_policy_label_smooth) * p_sf + sf_policy_label_smooth * uniform

                ps = float(p_sf.sum())
                if ps > 0:
                    p_sf /= ps

                # Attach SF targets to the *previous* network-turn sample (if present).
                if samples_per_game[idx]:
                    rec = samples_per_game[idx][-1]
                    if rec.sf_policy_target is None and rec.sf_move_index is None:
                        rec.sf_policy_target = p_sf
                        rec.sf_move_index = a_idx
                        if res.wdl is not None:
                            # SF evaluated position t+1 (opponent's turn, side-to-move = opponent).
                            # The network's perspective at t = flip(opponent's perspective at t+1).
                            # This single flip is always correct regardless of which color the
                            # network plays: opponent wins at t+1 ↔ network loses at t.
                            rec.sf_wdl = _flip_wdl_to_white(res.wdl)

                boards[idx].push(move_to_play)
                if boards[idx].is_game_over():
                    done[idx] = True

    all_samples: list[ReplaySample] = []
    w = d = l = 0
    sf_d6_sum = 0.0
    sf_d6_n = 0

    # Volatility is defined over a 6-ply horizon. With net-only records (one per full move),
    # that is a +3 record offset.
    VOL_HORIZON_RECORDS = 3

    # Terminal SF eval for max_plies timeout games.
    # Instead of labeling all timeouts as draws, query SF on the final position
    # to get a realistic WDL estimate. This breaks the draw-trap circular dependency
    # where value head learns "everything is a draw" and Gumbel Q≈0.
    # Use a higher node budget for more accurate position evaluation.
    # Skill level 0 can weaken SF's WDL estimates too, so use more nodes to compensate.
    _TERMINAL_EVAL_NODES = 1000
    timeout_idxs = [i for i, b in enumerate(boards) if b.result(claim_draw=True) == "*"]
    terminal_results: dict[int, StockfishResult | None] = {}
    if timeout_idxs:
        try:
            if isinstance(stockfish, StockfishPool):
                futures = {i: stockfish.submit(boards[i].fen(), nodes=_TERMINAL_EVAL_NODES) for i in timeout_idxs}
                for i, fut in futures.items():
                    try:
                        terminal_results[i] = fut.result()
                    except Exception:
                        terminal_results[i] = None
            else:
                for i in timeout_idxs:
                    try:
                        terminal_results[i] = stockfish.search(boards[i].fen(), nodes=_TERMINAL_EVAL_NODES)
                    except Exception:
                        terminal_results[i] = None
        except Exception:
            pass  # Fall back to draw labeling if SF eval fails

    def _sf_terminal_result(board: chess.Board, sf_res: StockfishResult | None) -> str:
        """Convert a terminal SF eval to a game result string, from white's POV."""
        if sf_res is None or sf_res.wdl is None:
            return "1/2-1/2"
        wdl_stm = sf_res.wdl  # side-to-move POV, normalized 0..1
        # Convert to white POV: if black to move, swap W/L
        if board.turn == chess.BLACK:
            wdl_white = np.array([float(wdl_stm[2]), float(wdl_stm[1]), float(wdl_stm[0])], dtype=np.float32)
        else:
            wdl_white = np.asarray(wdl_stm, dtype=np.float32)
        # Require high confidence (85%) to label a timeout as decisive.
        # Random 200-ply positions often have slight material imbalances that SF
        # reads as a weak advantage at low node counts — using a high threshold
        # prevents mislabeling those as decisive and biasing the W/D/L distribution.
        if float(wdl_white[0]) > 0.85:
            return "1-0"
        if float(wdl_white[2]) > 0.85:
            return "0-1"
        return "1/2-1/2"

    for i, b in enumerate(boards):
        result = b.result(claim_draw=True)
        # Games that reach max_plies without a decisive result return "*" (still in progress).
        # Use SF terminal eval to label these correctly rather than always calling them draws.
        if result == "*":
            result = _sf_terminal_result(b, terminal_results.get(i))

        records = samples_per_game[i]

        # Syzygy tablebase rescoring: replay the game, find the first position
        # with ≤7 pieces and no castling, probe the tablebase, and use that
        # proven result for ALL positions in the game.
        # Optionally also rescore policy targets for TB-eligible *network-turn* positions
        # with 100% weight on the DTZ-optimal move.
        tb_policy_overrides: dict[int, np.ndarray] = {}
        if syzygy_path and starting_boards is not None:
            replay_board = starting_boards[i].copy()
            replay_boards: list[chess.Board] = []
            move_stack = list(b.move_stack)
            opening_len = len(starting_boards[i].move_stack)

            for mv in move_stack[opening_len:]:
                replay_board.push(mv)
                replay_boards.append(replay_board.copy())

            tb_result = rescore_game_samples(replay_boards, [True] * len(replay_boards), syzygy_path)
            if tb_result is not None:
                result = tb_result

            if syzygy_policy:
                replay_board = starting_boards[i].copy()
                sample_idx = 0
                for mv in move_stack[opening_len:]:
                    board_before = replay_board.copy()
                    if _is_network_turn(board_turn=replay_board.turn, network_color=network_color[i]):
                        if sample_idx >= len(records):
                            break
                        if _eligible(board_before):
                            best = probe_best_move(board_before, syzygy_path)
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

        # Track wins/draws/losses from the network's perspective (not always white).
        net_col = network_color[i]
        if result == "1/2-1/2":
            d += 1
        elif (result == "1-0" and net_col == chess.WHITE) or (result == "0-1" and net_col == chess.BLACK):
            w += 1
        else:
            l += 1

        n = len(records)

        vs = str(volatility_source).lower().strip()
        if vs not in ("raw", "search"):
            vs = "raw"

        # Precompute volatility targets:
        # - network volatility: |WDL[t+6ply] - WDL[t]| where WDL is chosen by volatility_source
        # - sf volatility:      |sf_wdl[t+6ply] - sf_wdl[t]| (only when both SF evals exist)
        vol_targets: list[np.ndarray | None] = [None] * n
        sf_vol_targets: list[np.ndarray | None] = [None] * n
        for t in range(n):
            th = t + int(VOL_HORIZON_RECORDS)
            if th < n:
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

        # Log-only SF delta metric: abs change in (W + 0.5*D) over 6 plies.
        for t in range(n):
            th = t + int(VOL_HORIZON_RECORDS)
            if th < n:
                sf0 = records[t].sf_wdl
                sf6 = records[th].sf_wdl
                if (sf0 is not None) and (sf6 is not None):
                    wr0 = float(sf0[0]) + 0.5 * float(sf0[1])
                    wr6 = float(sf6[0]) + 0.5 * float(sf6[1])
                    sf_d6_sum += abs(wr6 - wr0)
                    sf_d6_n += 1

        for t, rec in enumerate(records):
            # Soft resignation: probabilistically skip positions with reduced weight.
            if float(rec.sample_weight) < 1.0 and rng.random() > float(rec.sample_weight):
                continue

            # Diff-focus skip gate (LC0-style): probabilistically drop easy positions.
            # This removes both value and policy signal (position is not added to replay).
            if float(rec.keep_prob) < 1.0 and rng.random() > float(rec.keep_prob):
                continue

            # WDL from the network's perspective (encoding is side-to-move relative,
            # so the network always sees itself as the moving side).
            if result == "1/2-1/2":
                wdl = 1
            elif (result == "1-0" and network_color[i] == chess.WHITE) or \
                 (result == "0-1" and network_color[i] == chess.BLACK):
                wdl = 0  # network won
            else:
                wdl = 2  # network lost

            moves_left = float(n - t) / max(1.0, float(max_plies))

            scalar_v = 1.0 if wdl == 0 else (0.0 if wdl == 1 else -1.0)
            cat = hlgauss_target(scalar_v, num_bins=categorical_bins, sigma=hlgauss_sigma)

            # Tablebase policy override: replace policy target with DTZ-optimal move.
            eff_probs = tb_policy_overrides.get(t, rec.policy_probs)

            soft = _apply_temperature(eff_probs, 2.0)

            future = None
            if t + 1 < n and bool(records[t + 1].has_policy):
                future = records[t + 1].policy_probs

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

    # NOTE: PID updates are performed by the outer loop (after a new net is trained),
    # not inside play_batch. This keeps difficulty changes aligned to model updates,
    # and prevents multiple small PID steps per iteration when selfplay is chunked.
    sf_nodes = int(getattr(stockfish, "nodes", 0) or 0)
    skill_lvl = getattr(stockfish, "skill_level", None)
    skill_lvl_i = None if skill_lvl is None else int(skill_lvl)

    mean_sf_d6 = float(sf_d6_sum / max(1, sf_d6_n)) if sf_d6_n > 0 else 0.0
    return all_samples, BatchStats(
        games=int(games),
        positions=len(all_samples),
        w=w,
        d=d,
        l=l,
        sf_nodes=sf_nodes if sf_nodes > 0 else None,
        sf_nodes_next=None,
        pid_ema_winrate=None,
        random_move_prob=float(opponent_random_move_prob),
        skill_level=skill_lvl_i,
        sf_eval_delta6=mean_sf_d6,
        sf_eval_delta6_n=int(sf_d6_n),
    )
