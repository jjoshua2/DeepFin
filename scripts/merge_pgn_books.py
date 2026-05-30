#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn


def _iter_games(path: Path):
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".pgn"):
        pgn = io.StringIO(path.read_text(encoding="utf-8", errors="replace"))
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            yield game
        return
    if suffixes.endswith(".zip"):
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
        return
    raise ValueError(f"Unsupported PGN input: {path}")


def _game_after_max_plies(game: chess.pgn.Game, max_plies: int) -> tuple[chess.Board, chess.pgn.Game] | None:
    board = game.board()
    out = chess.pgn.Game()
    out.headers.update(game.headers)
    node = out
    src = game
    for _ in range(max_plies):
        nxt = src.variation(0) if src.variations else None
        if nxt is None:
            return None
        move = nxt.move
        if move not in board.legal_moves:
            return None
        board.push(move)
        node = node.add_variation(move)
        src = nxt
    return board, out


def _final_key(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def _write_game(out: TextIO, game: chess.pgn.Game) -> None:
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    out.write(game.accept(exporter))
    out.write("\n\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge PGN opening books with final-position dedupe.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--add", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-plies", type=int, default=16)
    args = parser.parse_args()

    seen: set[str] = set()
    buffer = io.StringIO()
    base_read = 0
    base_written = 0
    base_duplicate_final_fen = 0
    add_summaries: list[dict[str, int | str]] = []

    for game in _iter_games(args.base):
        base_read += 1
        materialized = _game_after_max_plies(game, int(args.max_plies))
        if materialized is None:
            continue
        board, trimmed = materialized
        key = _final_key(board)
        if key in seen:
            base_duplicate_final_fen += 1
            continue
        seen.add(key)
        _write_game(buffer, trimmed)
        base_written += 1

    total_added_written = 0
    for add_path in args.add:
        add_read = 0
        add_written = 0
        add_duplicate_final_fen = 0
        add_short_or_bad = 0
        for game in _iter_games(add_path):
            add_read += 1
            materialized = _game_after_max_plies(game, int(args.max_plies))
            if materialized is None:
                add_short_or_bad += 1
                continue
            board, trimmed = materialized
            key = _final_key(board)
            if key in seen:
                add_duplicate_final_fen += 1
                continue
            seen.add(key)
            _write_game(buffer, trimmed)
            add_written += 1
        total_added_written += add_written
        add_summaries.append(
            {
                "path": str(add_path),
                "read": add_read,
                "written": add_written,
                "duplicate_final_fen": add_duplicate_final_fen,
                "short_or_bad": add_short_or_bad,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("book.pgn", buffer.getvalue().encode("utf-8"))

    summary = {
        "base": str(args.base),
        "output": str(args.output),
        "output_size_bytes": args.output.stat().st_size,
        "max_plies": int(args.max_plies),
        "base_read": base_read,
        "base_written": base_written,
        "base_duplicate_final_fen": base_duplicate_final_fen,
        "adds": add_summaries,
        "added_written": total_added_written,
        "combined_games_written": base_written + total_added_written,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
