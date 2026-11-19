"""Deterministic TRAIN / held-out EVAL split of opening books.

A book line is the mainline of one PGN game, truncated to the book's ply cap
(production: 16 plies for the 8-move book, 4 for the 2-move book). Lines are
deduplicated, and lines whose final position is terminal, forced or illegal are
dropped (the self-play opening consumer would skip them).

EVAL = the ``eval_lines`` lines with the smallest ``sha256(salt, line)``, so the
choice depends only on the line text and the salt, never on file order. Every
remaining line whose FINAL POSITION equals an eval line's final position, its
colour-mirror (``Board.mirror``) or its left-right reflection (the trainer's
mirror augmentation) is excluded from TRAIN, so neither a transposition nor a
mirror of an eval start can be trained from.

Outputs (in ``out_dir``):
* ``train.jsonl`` / ``eval.jsonl`` -- one ``{"id","book","line"}`` per line,
  ``line`` in the ``<root_fen> | <uci ...>`` grammar self-play doles;
* ``eval.pgn`` -- the eval lines as PGN games, for ``arena_standard --openings``
  (use ``--opening-plies`` >= the longest line);
* ``manifest.json``.
"""
from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from chess_anti_engine.simple_rl.common import input_ref, write_json_atomic, write_manifest

STANDARD_FEN = chess.STARTING_FEN


@dataclass(frozen=True)
class BookLine:
    book: int
    root_fen: str
    moves: tuple[str, ...]

    @property
    def line(self) -> str:
        return f"{self.root_fen} | {' '.join(self.moves)}".rstrip()

    def final_board(self) -> chess.Board:
        board = chess.Board(self.root_fen)
        for uci in self.moves:
            board.push_uci(uci)
        return board


def line_id(line: str) -> str:
    return hashlib.sha256(line.encode()).hexdigest()[:16]


def position_keys(board: chess.Board) -> set[str]:
    """The final position and the two mirrors that must not cross the split."""
    keys = {board.epd(), board.mirror().epd()}
    flipped = board.transform(chess.flip_horizontal)
    keys.add(flipped.epd())
    return keys


def read_book(path: Path, *, book: int, max_plies: int) -> list[BookLine]:
    from chess_anti_engine.selfplay.opening import _iter_pgn_bytes_from_path

    lines: list[BookLine] = []
    for blob in _iter_pgn_bytes_from_path(Path(path)):
        stream = io.StringIO(blob.decode("utf-8", errors="ignore"))
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            board = game.board()
            moves: list[str] = []
            for move in game.mainline_moves():
                if len(moves) >= int(max_plies):
                    break
                moves.append(move.uci())
            if moves:
                lines.append(BookLine(book=book, root_fen=board.fen(), moves=tuple(moves)))
    return lines


def usable(line: BookLine) -> bool:
    from chess_anti_engine.selfplay.opening import _fen_reject_reason

    return _fen_reject_reason(line.line) is None


def split_lines(lines: list[BookLine], *, eval_lines: int, salt: str) -> tuple[list[BookLine], list[BookLine], dict[str, int]]:
    unique: dict[str, BookLine] = {}
    for line in lines:
        unique.setdefault(line.line, line)  # first book wins for a shared line
    candidates = [line for line in unique.values() if usable(line)]
    if len(candidates) <= eval_lines:
        raise ValueError(f"only {len(candidates)} usable lines; cannot hold out {eval_lines}")

    def score(line: BookLine) -> str:
        return hashlib.sha256(f"{salt}\n{line.line}".encode()).hexdigest()

    ranked = sorted(candidates, key=lambda line: (score(line), line.line))
    eval_set = ranked[:eval_lines]
    banned: set[str] = set()
    for line in eval_set:
        banned |= position_keys(line.final_board())
    train: list[BookLine] = []
    excluded = 0
    for line in ranked[eval_lines:]:
        if line.final_board().epd() in banned:
            excluded += 1
            continue
        train.append(line)
    train.sort(key=lambda line: line.line)
    eval_set.sort(key=lambda line: line.line)
    counts = {"parsed": len(lines), "unique": len(unique), "usable": len(candidates),
              "eval": len(eval_set), "train": len(train), "train_excluded_by_eval_position": excluded}
    return train, eval_set, counts


def write_pgn(path: Path, lines: list[BookLine]) -> None:
    with Path(path).open("w") as fh:
        for line in lines:
            if line.root_fen != STANDARD_FEN:
                raise ValueError("eval PGN lines must start from the standard position "
                                 "(arena_standard replays book moves from startpos)")
            game = chess.pgn.Game()
            game.headers["Event"] = f"simple_rl eval {line_id(line.line)}"
            node: chess.pgn.GameNode = game
            for uci in line.moves:
                node = node.add_variation(chess.Move.from_uci(uci))
            print(game, file=fh, end="\n\n")


def write_jsonl(path: Path, lines: list[BookLine]) -> None:
    with Path(path).open("w") as fh:
        for line in lines:
            fh.write(json.dumps({"id": line_id(line.line), "book": line.book, "line": line.line}) + "\n")


def load_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(row) for row in Path(path).read_text().splitlines() if row.strip()]


def split_openings(*, books: list[tuple[Path, int]], out_dir: Path, eval_lines: int, salt: str) -> dict[str, Any]:
    """``books`` = [(path, max_plies), ...]; book numbers are 1-based list order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: list[BookLine] = []
    per_book: dict[str, int] = {}
    for number, (path, max_plies) in enumerate(books, start=1):
        parsed = read_book(path, book=number, max_plies=max_plies)
        per_book[str(number)] = len(parsed)
        lines.extend(parsed)
    train, eval_set, counts = split_lines(lines, eval_lines=eval_lines, salt=salt)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "eval.jsonl", eval_set)
    write_pgn(out_dir / "eval.pgn", eval_set)
    max_eval_plies = max(len(line.moves) for line in eval_set)
    summary = {"eval_max_plies": max_eval_plies,
               "train_by_book": {str(b): sum(1 for line in train if line.book == b) for b in range(1, len(books) + 1)},
               "eval_by_book": {str(b): sum(1 for line in eval_set if line.book == b) for b in range(1, len(books) + 1)}}
    write_json_atomic(out_dir / "summary.json", summary)
    write_manifest(
        out_dir, stage="split-openings",
        config={"eval_lines": eval_lines, "salt": salt,
                "books": [{"path": str(p), "max_plies": m, "book": i + 1} for i, (p, m) in enumerate(books)]},
        inputs={f"book{i + 1}": input_ref(p) for i, (p, _m) in enumerate(books)},
        outputs={name: input_ref(out_dir / name) for name in ("train.jsonl", "eval.jsonl", "eval.pgn")},
        counts={**counts, "parsed_by_book": per_book, **summary}, seeds={"salt": salt},
    )
    return {**counts, **summary}
