#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from chess_anti_engine.encoding import encode_cboard_for_model, model_input_plane_count
from chess_anti_engine.encoding.cboard_encode import CBoard
from chess_anti_engine.moves import (
    index_to_move_for_encoding,
    legal_move_indices_for_encoding,
    normalize_policy_encoding,
)
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


@dataclass(frozen=True)
class BeamItem:
    board: chess.Board
    seq: tuple[str, ...]
    score: float


def _iter_games(path: Path):
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.lower().endswith(".pgn"):
                continue
            pgn = io.StringIO(z.read(name).decode("utf-8", errors="replace"))
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                yield game


def _seq_board(game: chess.pgn.Game, plies: int) -> tuple[tuple[str, ...], chess.Board] | None:
    board = game.board()
    node = game
    seq: list[str] = []
    for _ in range(plies):
        nxt = node.variation(0) if node.variations else None
        if nxt is None:
            return None
        move = nxt.move
        if move not in board.legal_moves:
            return None
        board.push(move)
        seq.append(move.uci())
        node = nxt
    return tuple(seq), board


def _final_key(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def _game_from_seq(seq: tuple[str, ...], *, source: str) -> chess.pgn.Game | None:
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = "PolicyBeam"
    game.headers["Site"] = "local"
    game.headers["Date"] = "????.??.??"
    game.headers["Round"] = "1"
    game.headers["White"] = "PolicyBeam"
    game.headers["Black"] = "PolicyBeam"
    game.headers["Result"] = "1/2-1/2"
    game.headers["Source"] = source
    node = game
    for uci in seq:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            return None
        board.push(move)
        node = node.add_variation(move)
    return game


def _write_game(out: io.StringIO, game: chess.pgn.Game) -> None:
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    out.write(game.accept(exporter))
    out.write("\n\n")


def _stockfish_abs_cp(engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int) -> int | None:
    info = engine.analyse(board, chess.engine.Limit(nodes=int(nodes)))
    score = info.get("score")
    if score is None:
        return None
    pov = score.pov(chess.WHITE)
    cp = pov.score(mate_score=100000)
    if abs(int(cp)) >= 100000:
        return None
    return abs(int(cp))


@torch.no_grad()
def _policy_top_moves(
    model: torch.nn.Module,
    boards: list[chess.Board],
    *,
    device: str,
    policy_encoding: str,
    top_k: int,
) -> list[list[tuple[chess.Move, float]]]:
    if not boards:
        return []
    # Planes and history layout both come from the model, never from a
    # hardcoded v1 default (docs/rl_loop_audit.md M11).
    xs = np.empty((len(boards), model_input_plane_count(model), 8, 8), dtype=np.float32)
    for i, board in enumerate(boards):
        xs[i] = encode_cboard_for_model(model, CBoard.from_board(board))
    x = torch.from_numpy(xs).to(device)
    autocast_enabled = str(device).startswith("cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        out = model(x)
    logits = out["policy_own"].float().cpu().numpy()
    results: list[list[tuple[chess.Move, float]]] = []
    for row, board in zip(logits, boards, strict=True):
        legal = legal_move_indices_for_encoding(board, policy_encoding=policy_encoding)
        if legal.size == 0:
            results.append([])
            continue
        legal_logits = row[legal].astype(np.float64, copy=False)
        order = np.argsort(legal_logits)[::-1][: max(1, min(int(top_k), legal.size))]
        selected = legal[order]
        max_logit = float(np.max(legal_logits))
        denom = float(np.exp(legal_logits - max_logit).sum())
        moves: list[tuple[chess.Move, float]] = []
        for idx in selected:
            try:
                move = index_to_move_for_encoding(int(idx), board, policy_encoding=policy_encoding)
            except ValueError:
                continue
            if move not in board.legal_moves:
                continue
            prob = float(math.exp(float(row[int(idx)]) - max_logit) / max(denom, 1e-30))
            moves.append((move, prob))
        results.append(moves)
    return results


def _expand_prefix(
    model: torch.nn.Module,
    start: chess.Board,
    seq: tuple[str, ...],
    *,
    device: str,
    policy_encoding: str,
    top_k: int,
    beam_width: int,
    target_plies: int,
    batch_size: int,
) -> list[BeamItem]:
    beam = [BeamItem(start.copy(stack=True), seq, 0.0)]
    while beam and len(beam[0].seq) < target_plies:
        next_items: list[BeamItem] = []
        for offset in range(0, len(beam), batch_size):
            chunk = beam[offset: offset + batch_size]
            top = _policy_top_moves(
                model,
                [item.board for item in chunk],
                device=device,
                policy_encoding=policy_encoding,
                top_k=top_k,
            )
            for item, moves in zip(chunk, top, strict=True):
                for move, prob in moves:
                    b = item.board.copy(stack=True)
                    b.push(move)
                    next_items.append(
                        BeamItem(
                            b,
                            (*item.seq, move.uci()),
                            item.score + math.log(max(prob, 1e-30)),
                        )
                    )
        next_items.sort(key=lambda item: item.score, reverse=True)
        beam = next_items[: max(1, int(beam_width))]
    return beam


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand thin 2-move opening prefixes with NN policy beam search.")
    parser.add_argument("--two-move-book", type=Path, required=True)
    parser.add_argument("--eight-move-book", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stockfish-path", type=str, default="")
    parser.add_argument("--sf-nodes", type=int, default=5000)
    parser.add_argument("--max-abs-cp", type=int, default=145)
    parser.add_argument("--continuation-threshold", type=int, default=5)
    parser.add_argument("--target-plies", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--max-prefixes", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    print(f"[policybeam] loading model {args.checkpoint}", flush=True)
    model = load_model_from_checkpoint(args.checkpoint, device=args.device).eval()
    policy_encoding = normalize_policy_encoding(getattr(model, "policy_encoding", "lc0_1858"))

    existing_final_keys: set[str] = set()
    continuations: dict[tuple[str, ...], set[tuple[str, ...]]] = {}
    for game in _iter_games(args.eight_move_book):
        s4 = _seq_board(game, 4)
        s16 = _seq_board(game, args.target_plies)
        if s4 is None or s16 is None:
            continue
        continuations.setdefault(s4[0], set()).add(s16[0])
        existing_final_keys.add(_final_key(s16[1]))

    prefixes: list[tuple[tuple[str, ...], chess.Board, int]] = []
    seen_prefixes: set[tuple[str, ...]] = set()
    for game in _iter_games(args.two_move_book):
        parsed = _seq_board(game, 4)
        if parsed is None:
            continue
        seq, board = parsed
        if seq in seen_prefixes:
            continue
        seen_prefixes.add(seq)
        existing_count = len(continuations.get(seq, set()))
        if 0 < existing_count <= int(args.continuation_threshold):
            prefixes.append((seq, board, existing_count))
    prefixes.sort(key=lambda x: (x[2], x[0]))
    if int(args.max_prefixes) > 0:
        prefixes = prefixes[: int(args.max_prefixes)]

    print(
        f"[policybeam] selected_prefixes={len(prefixes)} "
        f"threshold={args.continuation_threshold} existing_final_keys={len(existing_final_keys)}",
        flush=True,
    )

    generated = 0
    dup_existing = 0
    dup_generated = 0
    sf_reject = 0
    accepted = 0
    by_existing_count: dict[str, int] = {}
    buffer = io.StringIO()
    generated_keys: set[str] = set()
    sf_enabled = bool(args.stockfish_path)
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path) if sf_enabled else None
    try:
        for i, (seq, board, existing_count) in enumerate(prefixes, start=1):
            endings = _expand_prefix(
                model,
                board,
                seq,
                device=args.device,
                policy_encoding=policy_encoding,
                top_k=args.top_k,
                beam_width=args.beam_width,
                target_plies=args.target_plies,
                batch_size=args.batch_size,
            )
            for item in endings:
                generated += 1
                key = _final_key(item.board)
                if key in existing_final_keys:
                    dup_existing += 1
                    continue
                if key in generated_keys:
                    dup_generated += 1
                    continue
                abs_cp = None
                if engine is not None:
                    abs_cp = _stockfish_abs_cp(engine, item.board, args.sf_nodes)
                    if abs_cp is None or abs_cp > int(args.max_abs_cp):
                        sf_reject += 1
                        continue
                game = _game_from_seq(
                    item.seq,
                    source=(
                        f"policybeam threshold<={args.continuation_threshold} "
                        f"existing={existing_count} abs_cp={abs_cp}"
                    ),
                )
                if game is None:
                    continue
                generated_keys.add(key)
                existing_final_keys.add(key)
                _write_game(buffer, game)
                accepted += 1
                by_existing_count[str(existing_count)] = by_existing_count.get(str(existing_count), 0) + 1
            if i % 50 == 0:
                print(
                    f"[policybeam] prefixes={i}/{len(prefixes)} generated={generated} "
                    f"accepted={accepted} dup_existing={dup_existing} sf_reject={sf_reject}",
                    flush=True,
                )
    finally:
        if engine is not None:
            engine.quit()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("book.pgn", buffer.getvalue().encode("utf-8"))

    summary = {
        "two_move_book": str(args.two_move_book),
        "eight_move_book": str(args.eight_move_book),
        "checkpoint": str(args.checkpoint),
        "output": str(args.output),
        "output_size_bytes": args.output.stat().st_size,
        "selected_prefixes": len(prefixes),
        "continuation_threshold": int(args.continuation_threshold),
        "target_plies": int(args.target_plies),
        "top_k": int(args.top_k),
        "beam_width": int(args.beam_width),
        "generated": generated,
        "accepted": accepted,
        "dup_existing_final_fen": dup_existing,
        "dup_generated_final_fen": dup_generated,
        "sf_enabled": sf_enabled,
        "sf_nodes": int(args.sf_nodes),
        "max_abs_cp": int(args.max_abs_cp),
        "sf_reject": sf_reject,
        "accepted_by_existing_count": dict(sorted(by_existing_count.items(), key=lambda kv: int(kv[0]))),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
