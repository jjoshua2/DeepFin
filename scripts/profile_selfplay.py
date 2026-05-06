"""Profile selfplay bottlenecks with GPU inference.

Measures wall-clock time for each phase of the self-play loop at 64 boards:
  1. Position encoding (python-chess vs CBoard, single vs batch)
  2. GPU forward pass (various batch sizes, with/without compile)
  3. legal_move_mask / legal_move_indices
  4. Board replay (python-chess copy+push vs CBoard copy+push_index)
  5. End-to-end MCTS (puct Python, puct C-tree, gumbel)
  6. cProfile of a single MCTS run to pinpoint hotspots

Run:
    python3 scripts/profile_selfplay.py --boards 64 --compile
    python3 scripts/profile_selfplay.py --boards 64 --compile --mcts gumbel
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import encode_position, encode_positions_batch
from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.mcts import GumbelConfig, MCTSConfig
from chess_anti_engine.mcts.gumbel import run_gumbel_root_many
from chess_anti_engine.mcts.puct import run_mcts_many
from chess_anti_engine.model import ModelConfig, build_model
from chess_anti_engine.moves.encode import legal_move_indices, legal_move_mask
from chess_anti_engine.utils.amp import inference_autocast

try:
    from chess_anti_engine.mcts.puct_c import run_mcts_many_c
    _HAS_C_TREE = True
except ImportError:
    _HAS_C_TREE = False

try:
    from chess_anti_engine.mcts.gumbel_c import run_gumbel_root_many_c
    _HAS_GUMBEL_C = True
except ImportError:
    _HAS_GUMBEL_C = False

try:
    from chess_anti_engine.encoding._lc0_ext import CBoard
    from chess_anti_engine.encoding.cboard_encode import encode_cboard
    _HAS_CBOARD = True
except ImportError:
    _HAS_CBOARD = False


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_boards(n: int, rng: np.random.Generator) -> list[chess.Board]:
    """Make n boards at various game stages for realistic profiling."""
    boards: list[chess.Board] = []
    for _ in range(n):
        b = chess.Board()
        plies = int(rng.integers(0, 40))
        for _ in range(plies):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(moves[int(rng.integers(0, len(moves)))])
            if b.is_game_over():
                b = chess.Board()
        boards.append(b)
    return boards


def _timer(fn, repeats: int = 5, warmup: int = 2) -> float:
    """Return average seconds per call."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _timer_cuda(fn, device: str, repeats: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def _hdr(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ── Benchmarks ───────────────────────────────────────────────────────────────

def bench_encoding(boards: list[chess.Board], repeats: int) -> None:
    _hdr("ENCODING (CPU)")
    n = len(boards)

    # 1a. encode_position (per-board, original)
    t = _timer(lambda: [encode_position(b, add_features=True) for b in boards], repeats)
    print(f"  encode_position (per-board):   {t*1000:8.2f} ms  ({t/n*1000:.3f} ms/pos)")

    # 1b. encode_positions_batch
    t = _timer(lambda: encode_positions_batch(boards, add_features=True), repeats)
    print(f"  encode_positions_batch:        {t*1000:8.2f} ms  ({t/n*1000:.3f} ms/pos)")

    # 1c. CBoard
    if _HAS_CBOARD:
        cboards = [CBoard.from_board(b) for b in boards]
        def _cb_batch():
            out = np.empty((len(cboards), 146, 8, 8), dtype=np.float32)
            for i, cb in enumerate(cboards):
                out[i] = encode_cboard(cb)
            return out
        t = _timer(_cb_batch, repeats)
        print(f"  CBoard encode_146 (batch):     {t*1000:8.2f} ms  ({t/n*1000:.3f} ms/pos)")


def bench_legal_moves(boards: list[chess.Board], repeats: int) -> None:
    _hdr("LEGAL MOVE GENERATION (CPU)")
    n = len(boards)

    t = _timer(lambda: [legal_move_indices(b) for b in boards], repeats)
    print(f"  legal_move_indices (per-board): {t*1000:8.2f} ms  ({t/n*1e6:.1f} µs/pos)")

    t = _timer(lambda: [legal_move_mask(b) for b in boards], repeats)
    print(f"  legal_move_mask (per-board):    {t*1000:8.2f} ms  ({t/n*1e6:.1f} µs/pos)")


def bench_board_ops(boards: list[chess.Board], repeats: int) -> None:
    _hdr("BOARD OPERATIONS (CPU)")
    n = len(boards)

    # board.copy(stack=False)
    t = _timer(lambda: [b.copy(stack=False) for b in boards], repeats)
    print(f"  board.copy(stack=False):        {t*1000:8.2f} ms  ({t/n*1e6:.1f} µs/pos)")

    # board.copy() with stack
    t = _timer(lambda: [b.copy() for b in boards], repeats)
    print(f"  board.copy() (with stack):      {t*1000:8.2f} ms  ({t/n*1e6:.1f} µs/pos)")

    # board.is_game_over()
    t = _timer(lambda: [b.is_game_over() for b in boards], repeats)
    print(f"  board.is_game_over():           {t*1000:8.2f} ms  ({t/n*1e6:.1f} µs/pos)")

    # CBoard ops
    if _HAS_CBOARD:
        cboards = [CBoard.from_board(b) for b in boards]
        t = _timer(lambda: [cb.copy() for cb in cboards], repeats)
        print(f"  CBoard.copy():                  {t*1000:8.2f} ms  ({t/n*1e6:.1f} µs/pos)")

        t = _timer(lambda: [cb.is_game_over() for cb in cboards], repeats)
        print(f"  CBoard.is_game_over():          {t*1000:8.2f} ms  ({t/n*1e6:.1f} µs/pos)")

        t = _timer(lambda: [cb.legal_move_indices() for cb in cboards], repeats)
        print(f"  CBoard.legal_move_indices():    {t*1000:8.2f} ms  ({t/n*1e6:.1f} µs/pos)")

        t = _timer(lambda: [CBoard.from_board(b) for b in boards], repeats)
        print(f"  CBoard.from_board():            {t*1000:8.2f} ms  ({t/n*1e6:.1f} µs/pos)")


def bench_gpu(model: torch.nn.Module, device: str, repeats: int) -> None:
    _hdr("GPU FORWARD PASS")

    for bs in [1, 4, 8, 16, 32, 64, 128]:
        x = torch.randn(bs, 146, 8, 8, device=device)
        def _fwd(x=x):
            with torch.no_grad():
                with inference_autocast(device=device, enabled=True, dtype="auto"):
                    return model(x)
        t = _timer_cuda(_fwd, device, repeats, warmup=3)
        print(f"  batch={bs:4d}: {t*1000:8.2f} ms total  ({t/bs*1000:.3f} ms/pos, {bs/t:.0f} pos/sec)")

    # H2D transfer
    _hdr("HOST-TO-DEVICE TRANSFER")
    for bs in [1, 8, 64]:
        x_np = np.random.randn(bs, 146, 8, 8).astype(np.float32)
        def _h2d(x=x_np):
            return torch.from_numpy(x).to(device)
        t = _timer_cuda(_h2d, device, repeats)
        print(f"  H2D batch={bs:4d}: {t*1e6:.1f} µs")

        def _h2d_pin(x=x_np):
            return torch.from_numpy(x).pin_memory().to(device, non_blocking=True)
        t = _timer_cuda(_h2d_pin, device, repeats)
        print(f"  H2D+pin batch={bs:4d}: {t*1e6:.1f} µs")

    # evaluate_encoded
    _hdr("LocalModelEvaluator.evaluate_encoded")
    evaluator = LocalModelEvaluator(model, device=device)
    for bs in [1, 8, 64]:
        x_np = np.random.randn(bs, 146, 8, 8).astype(np.float32)
        t = _timer_cuda(lambda x=x_np: evaluator.evaluate_encoded(x), device, repeats, warmup=3)
        print(f"  batch={bs:4d}: {t*1000:8.2f} ms total  ({t/bs*1000:.3f} ms/pos)")


def bench_mcts_e2e(
    model: torch.nn.Module,
    device: str,
    boards: list[chess.Board],
    sims: int,
    repeats: int,
) -> None:
    _hdr(f"END-TO-END MCTS ({len(boards)} boards × {sims} sims)")
    n = len(boards)
    evaluator = LocalModelEvaluator(model, device=device)

    # Gumbel Python
    def _gumbel():
        return run_gumbel_root_many(
            model, boards, device=device, rng=np.random.default_rng(42),
            cfg=GumbelConfig(simulations=sims), evaluator=evaluator,
        )
    t_gumbel_py = _timer_cuda(_gumbel, device, repeats, warmup=2)
    print(f"  Gumbel (Python):        {t_gumbel_py*1000:8.1f} ms  ({t_gumbel_py/n*1000:.1f} ms/board, {n/t_gumbel_py:.1f} boards/sec)")

    # Gumbel C (CBoard)
    if _HAS_GUMBEL_C:
        def _gumbel_c():
            return run_gumbel_root_many_c(
                model, boards, device=device, rng=np.random.default_rng(42),
                cfg=GumbelConfig(simulations=sims), evaluator=evaluator,
            )
        t_gumbel_c = _timer_cuda(_gumbel_c, device, repeats, warmup=2)
        print(f"  Gumbel (CBoard):        {t_gumbel_c*1000:8.1f} ms  ({t_gumbel_c/n*1000:.1f} ms/board, {n/t_gumbel_c:.1f} boards/sec)")
        speedup = t_gumbel_py / t_gumbel_c if t_gumbel_c > 0 else 0
        print(f"  Gumbel speedup:         {speedup:.2f}x")

    # PUCT C tree (for reference)
    if _HAS_C_TREE:
        def _puct_c():
            return run_mcts_many_c(
                model, boards, device=device, rng=np.random.default_rng(42),
                cfg=MCTSConfig(simulations=sims), evaluator=evaluator,
            )
        t = _timer_cuda(_puct_c, device, repeats, warmup=2)
        print(f"  PUCT (C tree+CBoard):   {t*1000:8.1f} ms  ({t/n*1000:.1f} ms/board, {n/t:.1f} boards/sec)")


def profile_mcts(
    model: torch.nn.Module,
    device: str,
    boards: list[chess.Board],
    sims: int,
    mcts_type: str,
) -> None:
    _hdr(f"cProfile: {mcts_type.upper()} ({len(boards)} boards × {sims} sims)")
    evaluator = LocalModelEvaluator(model, device=device)
    rng = np.random.default_rng(42)

    # Warmup
    if mcts_type == "gumbel_c" and _HAS_GUMBEL_C:
        run_gumbel_root_many_c(model, boards[:2], device=device, rng=rng,
                               cfg=GumbelConfig(simulations=4), evaluator=evaluator)
    elif mcts_type == "gumbel":
        run_gumbel_root_many(model, boards[:2], device=device, rng=rng,
                             cfg=GumbelConfig(simulations=4), evaluator=evaluator)
    elif mcts_type == "puct_c" and _HAS_C_TREE:
        run_mcts_many_c(model, boards[:2], device=device, rng=rng,
                        cfg=MCTSConfig(simulations=4), evaluator=evaluator)
    else:
        run_mcts_many(model, boards[:2], device=device, rng=rng,
                      cfg=MCTSConfig(simulations=4), evaluator=evaluator)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    pr = cProfile.Profile()
    pr.enable()

    rng2 = np.random.default_rng(42)
    if mcts_type == "gumbel_c" and _HAS_GUMBEL_C:
        run_gumbel_root_many_c(model, boards, device=device, rng=rng2,
                               cfg=GumbelConfig(simulations=sims), evaluator=evaluator)
    elif mcts_type == "gumbel":
        run_gumbel_root_many(model, boards, device=device, rng=rng2,
                             cfg=GumbelConfig(simulations=sims), evaluator=evaluator)
    elif mcts_type == "puct_c" and _HAS_C_TREE:
        run_mcts_many_c(model, boards, device=device, rng=rng2,
                        cfg=MCTSConfig(simulations=sims), evaluator=evaluator)
    else:
        run_mcts_many(model, boards, device=device, rng=rng2,
                      cfg=MCTSConfig(simulations=sims), evaluator=evaluator)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(40)
    print(s.getvalue())


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Profile selfplay bottlenecks")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--ffn-mult", type=float, default=2)
    ap.add_argument("--simulations", type=int, default=64)
    ap.add_argument("--boards", type=int, default=64)
    ap.add_argument("--mcts", type=str, default="gumbel_c", choices=["puct", "puct_c", "gumbel", "gumbel_c"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--profile-only", action="store_true", help="Skip micro-benchmarks, only run cProfile")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    n_boards = int(args.boards)
    sims = int(args.simulations)
    repeats = int(args.repeats)
    if n_boards <= 0:
        raise SystemExit("--boards must be > 0")
    if sims <= 0:
        raise SystemExit("--simulations must be > 0")
    if repeats <= 0:
        raise SystemExit("--repeats must be > 0")

    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Model: embed={args.embed_dim} layers={args.num_layers} heads={args.num_heads} ffn_mult={args.ffn_mult}")
    print(f"Batch: {n_boards} boards, {sims} simulations, MCTS={args.mcts}")
    print(f"Compile: {args.compile}")

    model = build_model(
        ModelConfig(
            kind="transformer",
            embed_dim=int(args.embed_dim),
            num_layers=int(args.num_layers),
            num_heads=int(args.num_heads),
            ffn_mult=float(args.ffn_mult),
            use_smolgen=True,
        )
    ).to(device)
    model.eval()

    if args.compile and device.startswith("cuda"):
        print("Compiling model (this may take a minute)...")
        model = torch.compile(model, mode="reduce-overhead")
        # Warmup compile with various batch sizes
        for bs in [1, 4, 16, 64]:
            dummy = torch.randn(bs, 146, 8, 8, device=device)
            with torch.no_grad():
                with inference_autocast(device=device, enabled=True, dtype="auto"):
                    model(dummy)
        torch.cuda.synchronize()
        print("Compile warmup done.")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M")

    boards = _make_boards(n_boards, rng)

    if not args.profile_only:
        bench_encoding(boards, repeats)
        bench_legal_moves(boards, repeats)
        bench_board_ops(boards, repeats)
        bench_gpu(model, device, repeats)
        bench_mcts_e2e(model, device, boards, sims, repeats)

    profile_mcts(model, device, boards, sims, str(args.mcts))


if __name__ == "__main__":
    main()
