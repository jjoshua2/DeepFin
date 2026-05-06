from __future__ import annotations

import argparse
import time

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import encode_position, encode_positions_batch
from chess_anti_engine.inference import _policy_output
from chess_anti_engine.mcts import GumbelConfig, MCTSConfig
from chess_anti_engine.mcts.gumbel import run_gumbel_root_many
from chess_anti_engine.mcts.puct import run_mcts_many
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves import (
    POLICY_SIZE,
    index_to_move,
    legal_move_mask,
    move_to_index,
)
from chess_anti_engine.replay.buffer import ReplaySample
from chess_anti_engine.selfplay.game import _result_to_wdl
from chess_anti_engine.selfplay.manager import BatchStats
from chess_anti_engine.selfplay.temperature import apply_policy_temperature
from chess_anti_engine.stockfish import StockfishPool, StockfishUCI
from chess_anti_engine.train.targets import hlgauss_target


def _run_mcts_for_subset(
    model: torch.nn.Module,
    sub_boards: list[chess.Board],
    *,
    mcts_type: str,
    simulations: int,
    temperature: float,
    device: str,
    rng: np.random.Generator,
):
    if mcts_type == "gumbel":
        cfg = GumbelConfig(simulations=int(simulations), temperature=float(temperature))
        return run_gumbel_root_many(model, sub_boards, device=device, rng=rng, cfg=cfg)
    return run_mcts_many(
        model,
        sub_boards,
        device=device,
        rng=rng,
        cfg=MCTSConfig(simulations=int(simulations), temperature=float(temperature)),
    )


def _build_replay_sample(
    *,
    rec: tuple[np.ndarray, np.ndarray, bool, np.ndarray | None, int | None, bool, float],
    wdl: int,
    moves_left: float,
    cat: np.ndarray,
    soft: np.ndarray,
    future: np.ndarray | None,
    vol: np.ndarray | None,
) -> ReplaySample:
    x, probs, pov_white, sf_wdl, sf_move_idx, has_policy, priority = rec
    return ReplaySample(
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


def _assemble_samples_for_game(
    *,
    samples: list[tuple[np.ndarray, np.ndarray, bool, np.ndarray | None, int | None, bool, float]],
    wdl_series: np.ndarray,
    result: str,
    max_plies: int,
) -> list[ReplaySample]:
    total_plies = len(samples)
    vol_targets: list[np.ndarray | None] = [None] * total_plies
    for t in range(total_plies):
        t6 = t + 6
        if t6 < total_plies:
            vol_targets[t] = np.abs(wdl_series[t6] - wdl_series[t]).astype(np.float32, copy=False)

    out: list[ReplaySample] = []
    for ply_idx, rec in enumerate(samples):
        pov_white = rec[2]
        wdl = _result_to_wdl(result, pov_white=bool(pov_white))
        scalar_v = 1.0 if wdl == 0 else (0.0 if wdl == 1 else -1.0)
        future = samples[ply_idx + 2][1] if ply_idx + 2 < total_plies else None
        out.append(_build_replay_sample(
            rec=rec,
            wdl=wdl,
            moves_left=float(total_plies - ply_idx) / max(1.0, float(max_plies)),
            cat=hlgauss_target(scalar_v, num_bins=32, sigma=0.04),
            soft=apply_policy_temperature(rec[1], 2.0),
            future=future,
            vol=vol_targets[ply_idx],
        ))
    return out


_GameSampleRec = tuple[np.ndarray, np.ndarray, bool, "np.ndarray | None", "int | None", bool, float]


def _do_network_turn(
    model: torch.nn.Module,
    boards: list[chess.Board],
    samples_per_game: list[list[_GameSampleRec]],
    net_idxs: list[int],
    *,
    device: str,
    rng: np.random.Generator,
    mcts_type: str,
    simulations: int,
    fast_simulations: int,
    playout_cap_fraction: float,
    temperature: float,
) -> None:
    """Network plies for ``net_idxs``: encode → MCTS → push move → record sample.

    Splits net_idxs into full/fast sim buckets via ``playout_cap_fraction`` and runs
    MCTS once per bucket. Mutates ``boards`` and ``samples_per_game`` in place.
    """
    xs_batch = encode_positions_batch([boards[i] for i in net_idxs], add_features=True)
    with torch.no_grad():
        xt = torch.from_numpy(xs_batch).to(device)
        out = model(xt)
        pol_logits = _policy_output(out).detach().float().cpu().numpy()

    is_full = rng.random(size=len(net_idxs)) < float(playout_cap_fraction)

    probs_list: list[np.ndarray | None] = [None] * len(net_idxs)
    actions: list[int | None] = [None] * len(net_idxs)
    for sub_idxs, sims in (
        ([j for j, v in enumerate(is_full) if bool(v)], simulations),
        ([j for j, v in enumerate(is_full) if not bool(v)], fast_simulations),
    ):
        if not sub_idxs:
            continue
        sub_boards = [boards[net_idxs[j]] for j in sub_idxs]
        p_sub, a_sub, _v_sub, _m_sub = _run_mcts_for_subset(
            model, sub_boards,
            mcts_type=mcts_type, simulations=sims, temperature=temperature,
            device=device, rng=rng,
        )
        for jj, p, a in zip(sub_idxs, p_sub, a_sub, strict=True):
            probs_list[jj] = p
            actions[jj] = a

    for j, (idx, probs, a) in enumerate(zip(net_idxs, probs_list, actions, strict=True)):
        assert probs is not None and a is not None
        priority = _surprise_priority(pol_logits[j], probs, boards[idx])
        boards[idx].push(index_to_move(int(a), boards[idx]))
        samples_per_game[idx].append(
            (xs_batch[j], probs, True, None, None, bool(is_full[j]), priority),
        )


def _surprise_priority(pol_logits: np.ndarray, probs: np.ndarray, board: chess.Board) -> float:
    """KL(masked-renormalized raw policy || MCTS-improved policy). Used as a
    sampling priority; higher KL means the search disagreed more with the prior."""
    mask = legal_move_mask(board)
    lg = pol_logits.astype(np.float64, copy=True)
    lg[~mask] = -1e9
    lg = lg - float(np.max(lg))
    rp = np.exp(lg)
    rp[~mask] = 0.0
    s = float(rp.sum())
    raw = (rp / s).astype(np.float32) if s > 0 else (mask.astype(np.float32) / float(mask.sum()))

    imp = np.maximum(probs.astype(np.float32, copy=False), 1e-12)
    raw_c = np.maximum(raw, 1e-12)
    return float(np.sum(raw_c * (np.log(raw_c) - np.log(imp))))


def _do_sf_turn(
    stockfish: StockfishUCI | StockfishPool,
    boards: list[chess.Board],
    samples_per_game: list[list[_GameSampleRec]],
    sf_idxs: list[int],
) -> None:
    """SF plies: query engine (in parallel for pool), push bestmove, record sample.

    Mutates ``boards`` and ``samples_per_game`` in place.
    """
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


def _estimate_wdl_for_all_samples(
    model: torch.nn.Module,
    samples_per_game: list[list[_GameSampleRec]],
    *,
    device: str,
    infer_bs: int = 512,
) -> tuple[np.ndarray, list[int]]:
    """Post-hoc batched WDL pass over every recorded position. Returns
    (wdl_est_all[N,3], per-game offsets into wdl_est_all)."""
    all_x: list[np.ndarray] = []
    offsets: list[int] = [0]
    for game_samples in samples_per_game:
        all_x.extend(rec[0] for rec in game_samples)
        offsets.append(len(all_x))

    wdl_est_all = np.zeros((len(all_x), 3), dtype=np.float32)
    if all_x:
        with torch.no_grad():
            for s in range(0, len(all_x), infer_bs):
                e = min(len(all_x), s + infer_bs)
                xt = torch.from_numpy(np.stack(all_x[s:e], axis=0)).to(device)
                out = model(xt)
                wdl_est_all[s:e] = (
                    torch.softmax(out["wdl"].detach().float(), dim=-1)
                    .cpu().numpy().astype(np.float32, copy=False)
                )
    return wdl_est_all, offsets


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
    samples_per_game: list[list[_GameSampleRec]] = [[] for _ in range(int(games))]

    for _ply in range(int(max_plies)):
        for i in range(int(games)):
            if not done[i] and boards[i].is_game_over():
                done[i] = True
        active_idxs = [i for i in range(int(games)) if not done[i]]
        if not active_idxs:
            break

        net_idxs = [i for i in active_idxs if boards[i].turn == chess.WHITE]
        if net_idxs:
            _do_network_turn(
                model, boards, samples_per_game, net_idxs,
                device=device, rng=rng, mcts_type=mcts_type,
                simulations=mcts_simulations, fast_simulations=fast_simulations,
                playout_cap_fraction=playout_cap_fraction, temperature=temperature,
            )

        sf_idxs = [i for i in active_idxs if boards[i].turn == chess.BLACK]
        if sf_idxs:
            _do_sf_turn(stockfish, boards, samples_per_game, sf_idxs)

    t_play_done = time.perf_counter()

    wdl_est_all, offsets = _estimate_wdl_for_all_samples(
        model, samples_per_game, device=device,
    )

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
        all_samples.extend(_assemble_samples_for_game(
            samples=samples_per_game[i],
            wdl_series=wdl_est_all[offsets[i] : offsets[i + 1]],
            result=result,
            max_plies=int(max_plies),
        ))

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
    ap.add_argument("--ffn-mult", type=float, default=2)
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

    if args.games <= 0:
        raise SystemExit("--games must be > 0")
    if args.max_plies <= 0:
        raise SystemExit("--max-plies must be > 0")
    if args.mcts_simulations <= 0:
        raise SystemExit("--mcts-simulations must be > 0")
    if args.fast_simulations <= 0:
        raise SystemExit("--fast-simulations must be > 0")
    if not 0.0 <= args.playout_cap_fraction <= 1.0:
        raise SystemExit("--playout-cap-fraction must be between 0 and 1")
    if args.sf_workers <= 0:
        raise SystemExit("--sf-workers must be > 0")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be > 0")
    if args.temperature < 0:
        raise SystemExit("--temperature must be >= 0")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(int(args.seed))

    model = build_model(
        ModelConfig(
            kind=str(args.model),
            embed_dim=int(args.embed_dim),
            num_layers=int(args.num_layers),
            num_heads=int(args.num_heads),
            ffn_mult=float(args.ffn_mult),
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
