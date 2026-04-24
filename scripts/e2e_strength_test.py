#!/usr/bin/env python3
"""E2E strength validation: train vs SF, measure semantic correctness.

Measures: policy accuracy vs SF, game length, value-SF correlation, W/D/L.
Usage: python scripts/e2e_strength_test.py [--hours 1.0]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.model.transformer import ChessNet, TransformerConfig
from chess_anti_engine.moves import legal_move_mask, move_to_index
from chess_anti_engine.replay.disk_buffer import DiskReplayBuffer
from chess_anti_engine.replay.shard import samples_to_arrays
from chess_anti_engine.selfplay import play_batch
from chess_anti_engine.selfplay.config import (
    DiffFocusConfig,
    GameConfig,
    OpponentConfig,
    SearchConfig,
    TemperatureConfig,
)
from chess_anti_engine.selfplay.opening import OpeningConfig
from chess_anti_engine.stockfish.uci import StockfishUCI
from chess_anti_engine.train import Trainer

EVAL_FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    "r1bqk2r/2ppbppp/p1n2n2/1p2p3/4P3/1B3N2/PPPP1PPP/RNBQR1K1 b kq - 3 7",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 6 8",
    "r1bq1rk1/pp3ppp/2n1pn2/3p4/3P4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 9",
    "8/5pk1/6p1/8/3R4/6PP/5PK1/1r6 w - - 0 40",
    "8/8/4k3/8/2P5/8/4K3/8 w - - 0 50",
    "r1bqk2r/ppp2ppp/2n1pn2/3p4/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 4 5",
    "rnbqkb1r/pp3ppp/2p1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5",
    "r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "r2q1rk1/ppp2ppp/3bbn2/3p4/3P4/2NBBN2/PPP2PPP/R2Q1RK1 w - - 0 10",
    "2r2rk1/pp3ppp/8/3p4/3P4/8/PP3PPP/2R2RK1 w - - 0 25",
    "r1bq1rk1/pp2nppp/2n1p3/3pP3/3P4/P1N2N2/1PP2PPP/R1BQ1RK1 w - - 0 10",
    "r2qr1k1/pppb1ppp/2np1n2/4p3/2B1P1b1/2NP1N2/PPP1QPPP/R1B2RK1 w - - 0 10",
    "r1bqkb1r/pppp1ppp/5n2/4N3/4n3/8/PPPPQPPP/RNB1KB1R w KQkq - 0 5",
    "8/8/8/1k6/8/1PK5/8/8 w - - 0 50",
    "r1bq1rk1/pppnnppp/4p3/3pP3/3P4/2PB1N2/PP3PPP/R1BQ1RK1 w - - 0 9",
]

def _build_eval_set(sf):
    out = []
    for fen in EVAL_FENS:
        board = chess.Board(fen)
        if board.is_game_over():
            continue
        try:
            result = sf.search(fen)
            if result is None or result.bestmove_uci == "0000":
                continue
            move = chess.Move.from_uci(result.bestmove_uci)
            if move not in board.legal_moves:
                continue
            # Extract WDL (may be None)
            wdl = result.wdl  # numpy (3,) or None
            # Derive centipawn-like score from WDL: score = 400*log10(W/L)
            score_cp = None
            if wdl is not None:
                w, _d, l = float(wdl[0]), float(wdl[1]), float(wdl[2])
                if w > 0.001 and l > 0.001:
                    score_cp = 400.0 * np.log10(w / l)
                elif w > l:
                    score_cp = 1000.0
                else:
                    score_cp = -1000.0
            out.append({
                "fen": fen, "board": board,
                "sf_move_idx": move_to_index(move, board),
                "sf_wdl": wdl,
                "sf_score_cp": score_cp,
            })
        except Exception as e:
            print(f"  Warning: skipping {fen[:30]}... ({e})")
            continue
    return out

@torch.no_grad()
def _eval_model(model, eval_set, device):
    model.eval()
    top1 = top3 = 0
    val_pairs = []
    for item in eval_set:
        board = item["board"].copy()
        x = torch.from_numpy(encode_position(board, add_features=True)).unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            out = model(x)
        logits = out["policy_own"][0].float().cpu()
        mask = torch.from_numpy(legal_move_mask(board)).float()
        logits[mask == 0] = -1e9
        topk = torch.topk(logits, k=min(3, int(mask.sum().item()))).indices.tolist()
        if topk[0] == item["sf_move_idx"]:
            top1 += 1
        if item["sf_move_idx"] in topk:
            top3 += 1
        if "wdl" in out:
            wdl_p = torch.softmax(out["wdl"][0].float().cpu(), dim=-1)
            mev = float(wdl_p[0] - wdl_p[2])
            if item["sf_score_cp"] is not None:
                cp = item["sf_score_cp"]
                sev = 2.0 / (1.0 + 10.0 ** (-cp / 400.0)) - 1.0
                val_pairs.append((mev, sev))
    model.train()
    n = max(1, len(eval_set))
    corr = 0.0
    if len(val_pairs) >= 3:
        c = float(np.corrcoef([v[0] for v in val_pairs], [v[1] for v in val_pairs])[0, 1])
        corr = 0.0 if np.isnan(c) else c
    return {"top1": top1 / n, "top3": top3 / n, "corr": corr,
            "top1_n": top1, "top3_n": top3, "total": len(eval_set)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stockfish-path", default="/home/josh/projects/chess/e2e_server/publish/stockfish")
    ap.add_argument("--sf-nodes", type=int, default=200)
    ap.add_argument("--sf-eval-nodes", type=int, default=5000)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--games-per-iter", type=int, default=20)
    ap.add_argument("--selfplay-batch", type=int, default=20)
    ap.add_argument("--mcts-simulations", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--iterations", type=int, default=500)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=5)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--work-dir", default="/tmp/e2e_strength_validation")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Model: embed={args.embed_dim}, L={args.num_layers}, H={args.num_heads}")
    print(f"Games/iter={args.games_per_iter}, sims={args.mcts_simulations}, hours={args.hours}")

    model = ChessNet(TransformerConfig(
        in_planes=146, embed_dim=args.embed_dim,
        num_layers=args.num_layers, num_heads=args.num_heads,
        use_smolgen=True, use_nla=False))
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    sf = StockfishUCI(args.stockfish_path, nodes=args.sf_nodes, multipv=3)

    print("Building eval set...")
    sf_eval = StockfishUCI(args.stockfish_path, nodes=args.sf_eval_nodes, multipv=1)
    eval_set = _build_eval_set(sf_eval)
    sf_eval.close()
    print(f"Eval positions: {len(eval_set)}")

    trainer = Trainer(model, device=device, lr=args.lr, log_dir=work_dir / "tb",
                      use_amp=(device == "cuda"), feature_dropout_p=0.15,
                      warmup_steps=args.warmup_steps, warmup_lr_start=1e-5,
                      swa_start=0, swa_freq=50)

    buf = DiskReplayBuffer(100_000, shard_dir=str(work_dir / "replay"),
                           rng=rng, shuffle_cap=20_000, shard_size=500)

    # Baseline
    bl = _eval_model(model, eval_set, device)
    print(f"\nBaseline: top1={bl['top1']:.1%} top3={bl['top3']:.1%} corr={bl['corr']:.3f}")

    hdr = (f"{'It':>4} {'Step':>5} {'W':>2} {'D':>2} {'L':>2} {'AvgL':>5} "
           f"{'Loss':>8} {'Pol':>7} {'WDL':>6} {'SFev':>6} "
           f"{'Top1%':>6} {'Top3%':>6} {'Corr':>6} {'Min':>5}")
    print(f"\n{hdr}\n{'-'*len(hdr)}")

    t_start = time.time()
    deadline = t_start + args.hours * 3600
    cum_w = cum_d = cum_l = cum_games = cum_plies = 0

    sp_kw = dict(
        device=device, rng=rng, stockfish=sf,
        opponent=OpponentConfig(random_move_prob=0.0),
        temp=TemperatureConfig(
            temperature=1.0, drop_plies=0, after=0.0,
            decay_start_move=20, decay_moves=60, endgame=0.6,
        ),
        search=SearchConfig(
            simulations=args.mcts_simulations, mcts_type="gumbel",
            playout_cap_fraction=0.25, fast_simulations=8,
            fpu_reduction=1.2, fpu_at_root=1.0,
        ),
        opening=OpeningConfig(random_start_plies=4),
        diff_focus=DiffFocusConfig(),
        game=GameConfig(
            max_plies=args.max_plies, selfplay_fraction=0.0,
            sf_policy_temp=0.25, sf_policy_label_smooth=0.05,
            timeout_adjudication_threshold=0.995, volatility_source="raw",
            categorical_bins=32, hlgauss_sigma=0.04,
        ),
    )

    log = []
    try:
        for it in range(args.iterations):
            if time.time() > deadline:
                print(f"\nTime limit ({args.hours}h) reached.")
                break

            samples, stats = play_batch(trainer.model, games=args.selfplay_batch,
                                        target_games=args.games_per_iter, **sp_kw)
            cum_w += stats.w
            cum_d += stats.d
            cum_l += stats.l
            cum_games += stats.games
            cum_plies += stats.positions

            if samples:
                buf.add_many_arrays(samples_to_arrays(samples))
            del samples
            buf.flush()
            if len(buf) == 0:
                continue

            steps = max(1, stats.positions // args.batch_size)
            m = trainer.train_steps(buf, batch_size=args.batch_size, steps=steps)
            avg_len = stats.positions / max(1, stats.games)

            ev = None
            t1_s = t3_s = co_s = ""
            if it % args.eval_interval == 0:
                ev = _eval_model(model, eval_set, device)
                t1_s = f"{ev['top1']:5.1%}"
                t3_s = f"{ev['top3']:5.1%}"
                co_s = f"{ev['corr']:6.3f}"

            mins = (time.time() - t_start) / 60.0
            print(f"{it:4d} {trainer.step:5d} {stats.w:2d} {stats.d:2d} {stats.l:2d} "
                  f"{avg_len:5.1f} {m.loss:8.4f} {m.policy_loss:7.4f} "
                  f"{m.wdl_loss:6.4f} {m.sf_eval_loss:6.4f} "
                  f"{t1_s:>6} {t3_s:>6} {co_s:>6} {mins:5.1f}", flush=True)

            entry = {"iter": it, "step": trainer.step, "w": stats.w, "d": stats.d,
                     "l": stats.l, "avg_len": avg_len, "loss": m.loss,
                     "pol": m.policy_loss, "wdl": m.wdl_loss, "sfev": m.sf_eval_loss}
            if ev:
                entry.update(top1=ev["top1"], top3=ev["top3"], corr=ev["corr"])
            log.append(entry)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        sf.close()

    # Final
    fin = _eval_model(model, eval_set, device)
    print(f"\n{'='*70}")
    print(f"FINAL RESULTS  ({(time.time()-t_start)/60:.1f} min, {cum_games} games, "
          f"{cum_plies} positions, {trainer.step} steps)")
    print(f"{'='*70}")
    print(f"  W/D/L: {cum_w}/{cum_d}/{cum_l}")
    print(f"  Avg game length: {cum_plies/max(1,cum_games):.1f}")
    print(f"  Baseline top-1: {bl['top1']:.1%}  →  Final: {fin['top1']:.1%}")
    print(f"  Baseline top-3: {bl['top3']:.1%}  →  Final: {fin['top3']:.1%}")
    print(f"  Baseline corr:  {bl['corr']:.3f}  →  Final: {fin['corr']:.3f}")

    # Show progression of eval metrics
    evals = [e for e in log if "top1" in e]
    if len(evals) >= 2:
        print(f"\n  Eval progression (every {args.eval_interval} iters):")
        for e in evals:
            print(f"    iter={e['iter']:4d} top1={e['top1']:.1%} top3={e['top3']:.1%} "
                  f"corr={e['corr']:.3f} loss={e['loss']:.4f}")

    # Save log
    log_path = work_dir / "results.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=float)
    print(f"\nResults saved to {log_path}")

if __name__ == "__main__":
    main()
