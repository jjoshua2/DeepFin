from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.selfplay.manager import _apply_temperature, BatchStats
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.stockfish import StockfishPool, StockfishUCI
from chess_anti_engine.mcts import MCTSConfig, GumbelConfig
from chess_anti_engine.mcts.puct import run_mcts_many
from chess_anti_engine.mcts.gumbel import run_gumbel_root_many
from chess_anti_engine.encoding import encode_position
from chess_anti_engine.moves import POLICY_SIZE, move_to_index, index_to_move, legal_move_mask
from chess_anti_engine.train.targets import hlgauss_target

import chess


def play_batch_timed(
    model: torch.nn.Module,
    *,
    device: str,
    rng: np.random.Generator,
    stockfish: StockfishUCI | StockfishPool,
    games: int,
    temperature: float,
    max_plies: int,
    mcts_simulations: int = 50,
    mcts_type: str = "puct",
    playout_cap_fraction: float = 0.25,
    fast_simulations: int = 8,
) -> tuple[list[ReplaySample], BatchStats, dict[str, float]]:
    """Timed variant of `selfplay.manager.play_batch`.

    Splits timing into:
    - play_sec: game advancement (Stockfish + MCTS + move selection)
    - posthoc_sec: post-hoc target construction, including the batched WDL pass

    Note: This intentionally mirrors the current implementation in
    `chess_anti_engine.selfplay.manager.play_batch`.
    """

    t0 = time.perf_counter()

    boards = [chess.Board() for _ in range(int(games))]
    done = [False] * int(games)

    # Each entry: (x, policy_probs, pov_white, sf_wdl, sf_move_index, has_policy, priority)
    samples_per_game: list[list[tuple[np.ndarray, np.ndarray, bool, np.ndarray | None, int | None, bool, float]]] = [
        [] for _ in range(int(games))
    ]

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

        # Network turns (white to move)
        net_idxs = [i for i in active_idxs if boards[i].turn == chess.WHITE]
        if net_idxs:
            xs = [encode_position(boards[i], add_features=True) for i in net_idxs]

            # policy logits for surprise priority only
            with torch.no_grad():
                xt = torch.from_numpy(np.stack(xs, axis=0)).to(device)
                out = model(xt)
                policy_out = out["policy"] if "policy" in out else out["policy_own"]
                pol_logits = policy_out.detach().float().cpu().numpy()

            is_full = rng.random(size=len(net_idxs)) < float(playout_cap_fraction)

            probs_list = [None] * len(net_idxs)
            actions = [None] * len(net_idxs)

            full_idxs = [j for j, v in enumerate(is_full) if bool(v)]
            if full_idxs:
                sub_boards = [boards[net_idxs[j]] for j in full_idxs]
                if mcts_type == "gumbel":
                    p_sub, a_sub, _v_sub = run_gumbel_root_many(
                        model,
                        sub_boards,
                        device=device,
                        rng=rng,
                        cfg=GumbelConfig(simulations=int(mcts_simulations), temperature=float(temperature)),
                    )
                else:
                    p_sub, a_sub, _v_sub = run_mcts_many(
                        model,
                        sub_boards,
                        device=device,
                        rng=rng,
                        cfg=MCTSConfig(simulations=int(mcts_simulations), temperature=float(temperature)),
                    )
                for jj, p, a in zip(full_idxs, p_sub, a_sub, strict=True):
                    probs_list[jj] = p
                    actions[jj] = a

            fast_idxs = [j for j, v in enumerate(is_full) if not bool(v)]
            if fast_idxs:
                sub_boards = [boards[net_idxs[j]] for j in fast_idxs]
                if mcts_type == "gumbel":
                    p_sub, a_sub, _v_sub = run_gumbel_root_many(
                        model,
                        sub_boards,
                        device=device,
                        rng=rng,
                        cfg=GumbelConfig(simulations=int(fast_simulations), temperature=float(temperature)),
                    )
                else:
                    p_sub, a_sub, _v_sub = run_mcts_many(
                        model,
                        sub_boards,
                        device=device,
                        rng=rng,
                        cfg=MCTSConfig(simulations=int(fast_simulations), temperature=float(temperature)),
                    )
                for jj, p, a in zip(fast_idxs, p_sub, a_sub, strict=True):
                    probs_list[jj] = p
                    actions[jj] = a

            for j, (idx, probs, a) in enumerate(zip(net_idxs, probs_list, actions, strict=True)):
                assert probs is not None and a is not None

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

                move = index_to_move(int(a), boards[idx])
                boards[idx].push(move)

                samples_per_game[idx].append((xs[j], probs, True, None, None, bool(is_full[j]), kl))

        # Stockfish turns (black to move)
        sf_idxs = [i for i in active_idxs if boards[i].turn == chess.BLACK]
        if sf_idxs:
            if isinstance(stockfish, StockfishPool):
                futures = {idx: stockfish.submit(boards[idx].fen()) for idx in sf_idxs}
                results = {idx: fut.result() for idx, fut in futures.items()}
            else:
                results = {idx: stockfish.search(boards[idx].fen()) for idx in sf_idxs}

            for idx in sf_idxs:
                res = results[idx]
                move = chess.Move.from_uci(res.bestmove_uci)
                if move not in boards[idx].legal_moves:
                    move = next(iter(boards[idx].legal_moves))

                x = encode_position(boards[idx], add_features=True)
                a_idx = int(move_to_index(move, boards[idx]))
                onehot = np.zeros((POLICY_SIZE,), dtype=np.float32)
                onehot[a_idx] = 1.0

                boards[idx].push(move)
                samples_per_game[idx].append((x, onehot, False, res.wdl, a_idx, False, 0.0))

    t_play_done = time.perf_counter()

    # Post-hoc batched WDL estimation for all recorded positions.
    infer_bs = 512
    all_x: list[np.ndarray] = []
    offsets: list[int] = [0]
    for g in range(int(games)):
        all_x.extend([rec[0] for rec in samples_per_game[g]])
        offsets.append(len(all_x))

    wdl_est_all = np.zeros((len(all_x), 3), dtype=np.float32)
    if all_x:
        with torch.no_grad():
            for s in range(0, len(all_x), infer_bs):
                e = min(len(all_x), s + infer_bs)
                xt = torch.from_numpy(np.stack(all_x[s:e], axis=0)).to(device)
                out = model(xt)
                wdl = torch.softmax(out["wdl"].detach().float(), dim=-1).cpu().numpy().astype(np.float32, copy=False)
                wdl_est_all[s:e] = wdl

    all_samples: list[ReplaySample] = []
    w = d = l = 0

    for i, b in enumerate(boards):
        result = b.result(claim_draw=True)
        if result == "1-0":
            w += 1
        elif result == "0-1":
            l += 1
        else:
            d += 1

        total_plies = len(samples_per_game[i])
        wdl_series = wdl_est_all[offsets[i] : offsets[i + 1]]

        vol_targets: list[np.ndarray | None] = [None] * total_plies
        for t in range(total_plies):
            t6 = t + 6
            if t6 < total_plies:
                vol_targets[t] = np.abs(wdl_series[t6] - wdl_series[t]).astype(np.float32, copy=False)

        for ply_idx, (x, probs, pov_white, sf_wdl, sf_move_idx, has_policy, priority) in enumerate(samples_per_game[i]):
            if result == "1/2-1/2":
                wdl = 1
            else:
                white_won = result == "1-0"
                wdl = 0 if (white_won == pov_white) else 2

            moves_left = float(total_plies - ply_idx) / max(1.0, float(max_plies))

            scalar_v = 1.0 if wdl == 0 else (0.0 if wdl == 1 else -1.0)
            cat = hlgauss_target(scalar_v, num_bins=32, sigma=0.04)

            soft = _apply_temperature(probs, 2.0)

            future = None
            if ply_idx + 2 < total_plies:
                future = samples_per_game[i][ply_idx + 2][1]

            vol = vol_targets[ply_idx]

            all_samples.append(
                ReplaySample(
                    x=x,
                    policy_target=probs,
                    wdl_target=int(wdl),
                    priority=priority,
                    has_policy=bool(has_policy),
                    sf_wdl=sf_wdl,
                    sf_move_index=sf_move_idx,
                    moves_left=moves_left,
                    is_network_turn=bool(pov_white),
                    categorical_target=cat,
                    policy_soft_target=soft,
                    future_policy_target=future,
                    has_future=(future is not None),
                    volatility_target=vol,
                    has_volatility=(vol is not None),
                )
            )

    t_done = time.perf_counter()

    timings = {
        "play_sec": float(t_play_done - t0),
        "posthoc_sec": float(t_done - t_play_done),
        "total_sec": float(t_done - t0),
    }

    stats = BatchStats(games=int(games), positions=len(all_samples), w=w, d=d, l=l)
    return all_samples, stats, timings


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark play_batch time vs post-hoc pass")
    ap.add_argument("--stockfish-path", type=str, required=True)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--model", type=str, default="transformer", choices=["tiny", "transformer"])
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--ffn-mult", type=int, default=2)
    ap.add_argument("--no-smolgen", action="store_true")
    ap.add_argument("--use-nla", action="store_true")

    ap.add_argument("--sf-nodes", type=int, default=50)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=1.25)
    ap.add_argument("--mcts", type=str, default="puct", choices=["puct", "gumbel"])
    ap.add_argument("--mcts-simulations", type=int, default=16)
    ap.add_argument("--playout-cap-fraction", type=float, default=0.25)
    ap.add_argument("--fast-simulations", type=int, default=8)
    ap.add_argument("--sf-workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3)

    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(int(args.seed))

    model = build_model(
        ModelConfig(
            kind=str(args.model),
            embed_dim=int(args.embed_dim),
            num_layers=int(args.num_layers),
            num_heads=int(args.num_heads),
            ffn_mult=int(args.ffn_mult),
            use_smolgen=not bool(args.no_smolgen),
            use_nla=bool(args.use_nla),
        )
    ).to(device)
    model.eval()

    if int(args.sf_workers) > 1:
        sf = StockfishPool(path=str(args.stockfish_path), nodes=int(args.sf_nodes), num_workers=int(args.sf_workers))
    else:
        sf = StockfishUCI(str(args.stockfish_path), nodes=int(args.sf_nodes))

    try:
        times = []
        for _ in range(int(args.repeat)):
            _samples, stats, t = play_batch_timed(
                model,
                device=device,
                rng=rng,
                stockfish=sf,
                games=int(args.games),
                temperature=float(args.temperature),
                max_plies=int(args.max_plies),
                mcts_simulations=int(args.mcts_simulations),
                mcts_type=str(args.mcts),
                playout_cap_fraction=float(args.playout_cap_fraction),
                fast_simulations=int(args.fast_simulations),
            )
            times.append(t)

        play = float(np.mean([t["play_sec"] for t in times]))
        post = float(np.mean([t["posthoc_sec"] for t in times]))
        total = float(np.mean([t["total_sec"] for t in times]))
        overhead = 100.0 * (post / max(1e-9, total))

        print(
            "play_batch benchmark (avg over repeat):\n"
            f"  games={stats.games} positions={stats.positions} W/D/L={stats.w}/{stats.d}/{stats.l}\n"
            f"  play_sec={play:.4f} posthoc_sec={post:.4f} total_sec={total:.4f}\n"
            f"  posthoc_overhead_pct={overhead:.2f}%"
        )

    finally:
        sf.close()


if __name__ == "__main__":
    main()
