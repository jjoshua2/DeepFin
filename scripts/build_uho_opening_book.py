#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn


_EVAL_RE = re.compile(r"\beval=([+-]\d+)\b")


def _iter_pgn_text(path: Path) -> list[tuple[str, str]]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".pgn"):
        return [(path.name, path.read_text(encoding="utf-8", errors="replace"))]
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            out: list[tuple[str, str]] = [
                (Path(name).name, z.read(name).decode("utf-8", errors="replace"))
                for name in z.namelist()
                if name.lower().endswith(".pgn")
            ]
        if not out:
            raise ValueError(f"No PGN members found in {path}")
        return out
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


def _parse_eval_cp(game: chess.pgn.Game) -> int | None:
    annotator = game.headers.get("Annotator", "")
    match = _EVAL_RE.search(annotator)
    if match is None:
        return None
    return int(match.group(1))


def build_book(
    *,
    existing: Path,
    raw_zip: Path,
    raw_member: str,
    output: Path,
    report: Path,
    min_cp: int,
    max_cp_exclusive: int,
    max_plies: int,
) -> None:
    seen: set[str] = set()
    existing_read = 0
    existing_written = 0
    existing_dups = 0
    raw_read = 0
    raw_missing_eval = 0
    raw_out_of_range = 0
    raw_short_or_bad = 0
    raw_dup_existing = 0
    raw_dup_raw = 0
    raw_written = 0
    raw_buckets: dict[str, int] = {}

    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    buffer = io.StringIO()
    for _, text in _iter_pgn_text(existing):
        pgn = io.StringIO(text)
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            existing_read += 1
            materialized = _game_after_max_plies(game, max_plies)
            if materialized is None:
                continue
            board, trimmed = materialized
            key = _final_key(board)
            if key in seen:
                existing_dups += 1
                continue
            seen.add(key)
            _write_game(buffer, trimmed)
            existing_written += 1

    raw_seen: set[str] = set()
    with zipfile.ZipFile(raw_zip) as z, z.open(raw_member) as f:
        pgn = io.StringIO(f.read().decode("utf-8", errors="replace"))
    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        raw_read += 1
        eval_cp = _parse_eval_cp(game)
        if eval_cp is None:
            raw_missing_eval += 1
            continue
        if eval_cp < min_cp or eval_cp >= max_cp_exclusive:
            raw_out_of_range += 1
            continue
        materialized = _game_after_max_plies(game, max_plies)
        if materialized is None:
            raw_short_or_bad += 1
            continue
        board, trimmed = materialized
        key = _final_key(board)
        bucket = str((eval_cp // 5) * 5)
        raw_buckets[bucket] = raw_buckets.get(bucket, 0) + 1
        if key in seen:
            raw_dup_existing += 1
            continue
        if key in raw_seen:
            raw_dup_raw += 1
            continue
        raw_seen.add(key)
        seen.add(key)
        trimmed.headers["Source"] = "SPCC UHO 2024 RawData"
        trimmed.headers["Annotator"] = f"eval={eval_cp:+04d}"
        _write_game(buffer, trimmed)
        raw_written += 1

    data = buffer.getvalue().encode("utf-8")
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("book.pgn", data)

    summary = {
        "source_existing": str(existing),
        "raw_zip": str(raw_zip),
        "raw_member": raw_member,
        "output": str(output),
        "output_size_bytes": output.stat().st_size,
        "max_plies": max_plies,
        "eval_min_cp": min_cp,
        "eval_max_cp_exclusive": max_cp_exclusive,
        "existing_games_read": existing_read,
        "existing_unique_final_fens_written": existing_written,
        "existing_duplicate_final_fens": existing_dups,
        "raw_read": raw_read,
        "raw_missing_eval": raw_missing_eval,
        "raw_out_of_range": raw_out_of_range,
        "raw_short_or_bad": raw_short_or_bad,
        "raw_dup_existing_fen": raw_dup_existing,
        "raw_dup_raw_fen": raw_dup_raw,
        "raw_accepted_unique_final_fens_written": raw_written,
        "combined_games_written": existing_written + raw_written,
        "combined_expansion_pct_vs_existing_unique": (
            (100.0 * raw_written / existing_written) if existing_written else 0.0
        ),
        "raw_eval_5cp_buckets_before_dedupe": dict(sorted(raw_buckets.items(), key=lambda kv: int(kv[0]))),
    }
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge UHO raw-data openings into an existing PGN zip.")
    parser.add_argument("--existing", type=Path, required=True)
    parser.add_argument("--raw-zip", type=Path, required=True)
    parser.add_argument("--raw-member", default="RawData_8mvs_-199_+199.pgn")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-cp", type=int, required=True)
    parser.add_argument("--max-cp-exclusive", type=int, required=True)
    parser.add_argument("--max-plies", type=int, default=16)
    args = parser.parse_args()
    build_book(
        existing=args.existing,
        raw_zip=args.raw_zip,
        raw_member=args.raw_member,
        output=args.output,
        report=args.report,
        min_cp=args.min_cp,
        max_cp_exclusive=args.max_cp_exclusive,
        max_plies=args.max_plies,
    )


if __name__ == "__main__":
    main()
