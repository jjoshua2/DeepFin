"""Deterministic TRAIN / held-out EVAL split of opening books.

A book line is the mainline of one PGN game, truncated to the book's ply cap
(production: 16 plies for the 8-move book, 4 for the 2-move book). Lines are
deduplicated, and lines whose final position is terminal, forced or illegal are
dropped (the self-play opening consumer would skip them).

A position's ORBIT is the position, its colour mirror (``Board.mirror``), its
left-right reflection (the trainer's mirror augmentation) and the reflection
of the colour mirror -- the group the two symmetries generate.

EVAL = the ``eval_lines`` lines with the smallest ``sha256(salt, line)``, so the
choice depends only on the line text and the salt, never on file order; a
candidate whose final-position orbit repeats an already-chosen EVAL line's is
skipped (a duplicate or mirror wastes the held-out budget and correlates
paired games). A remaining line is excluded from TRAIN when any position on
its path (root to final) is in the orbit of an EVAL final position, or when
its final position is in the orbit of any position on an EVAL line's path --
so no TRAIN game starts at, passes through, or is a prefix of an EVAL start,
mirrors included. TRAIN is NOT de-duplicated by position (naturally recurring
positions stay); the duplicate counts are reported in the summary. Lines with
PGN parse errors are rejected and counted.

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
from collections.abc import Iterator
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
    """The position's orbit: itself, its colour mirror, its left-right
    reflection and the reflection of the colour mirror."""
    mirrored = board.mirror()
    return {board.epd(), mirrored.epd(), board.transform(chess.flip_horizontal).epd(),
            mirrored.transform(chess.flip_horizontal).epd()}


def path_boards(line: BookLine) -> Iterator[chess.Board]:
    board = chess.Board(line.root_fen)
    yield board.copy(stack=False)
    for uci in line.moves:
        board.push_uci(uci)
        yield board.copy(stack=False)


def read_book(path: Path, *, book: int, max_plies: int, stats: dict[str, int] | None = None) -> list[BookLine]:
    """Mainlines cut to ``max_plies``. A game with a PGN parse error (the
    parser keeps only the legal prefix) is rejected, never truncated silently;
    a blob that is not valid UTF-8 is decoded with replacement characters and
    counted, so its damaged games surface as parse errors."""
    from chess_anti_engine.selfplay.opening import _iter_pgn_bytes_from_path

    stats = stats if stats is not None else {}
    stats.setdefault("rejected_parse_errors", 0)
    stats.setdefault("non_utf8_blobs", 0)
    lines: list[BookLine] = []
    for blob in _iter_pgn_bytes_from_path(Path(path)):
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            stats["non_utf8_blobs"] += 1
            text = blob.decode("utf-8", errors="replace")
        stream = io.StringIO(text)
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            if game.errors:
                stats["rejected_parse_errors"] += 1
                continue
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


def split_lines(lines: list[BookLine], *, eval_lines: int, salt: str) -> tuple[list[BookLine], list[BookLine], dict[str, Any]]:
    unique: dict[str, BookLine] = {}
    for line in lines:
        unique.setdefault(line.line, line)  # first book wins for a shared line
    candidates = [line for line in unique.values() if usable(line)]
    if len(candidates) <= eval_lines:
        raise ValueError(f"only {len(candidates)} usable lines; cannot hold out {eval_lines}")

    def score(line: BookLine) -> str:
        return hashlib.sha256(f"{salt}\n{line.line}".encode()).hexdigest()

    ranked = sorted(candidates, key=lambda line: (score(line), line.line))
    eval_set: list[BookLine] = []
    eval_final: set[str] = set()  # orbits of EVAL final positions
    rest: list[BookLine] = []
    eval_duplicates = 0
    for line in ranked:
        if len(eval_set) >= eval_lines:
            rest.append(line)
            continue
        keys = position_keys(line.final_board())
        if keys & eval_final:
            eval_duplicates += 1
            rest.append(line)
            continue
        eval_set.append(line)
        eval_final |= keys
    if len(eval_set) < eval_lines:
        raise ValueError(f"only {len(eval_set)} position-distinct lines; cannot hold out {eval_lines}")
    eval_path: set[str] = set()  # orbits of every position on every EVAL path
    for line in eval_set:
        for board in path_boards(line):
            eval_path |= position_keys(board)
    train: list[BookLine] = []
    excluded = {"final_matches_eval_start": 0, "passes_through_eval_start": 0, "ends_on_eval_path": 0}
    for line in rest:
        path = [board.epd() for board in path_boards(line)]
        # Both sets are orbit-closed, so a raw epd test is the orbit test.
        if path[-1] in eval_final:
            excluded["final_matches_eval_start"] += 1
        elif any(p in eval_final for p in path[:-1]):
            excluded["passes_through_eval_start"] += 1
        elif path[-1] in eval_path:
            excluded["ends_on_eval_path"] += 1
        else:
            train.append(line)
    train.sort(key=lambda line: line.line)
    eval_set.sort(key=lambda line: line.line)
    finals = [line.final_board() for line in train]
    raw = [b.epd() for b in finals]
    orbit_rep = [min(position_keys(b)) for b in finals]
    counts = {"parsed": len(lines), "unique": len(unique), "usable": len(candidates),
              "eval": len(eval_set), "eval_skipped_duplicate_or_mirror": eval_duplicates,
              "train": len(train), "train_excluded_by_eval_position": sum(excluded.values()),
              **{f"train_excluded_{k}": v for k, v in excluded.items()},
              # Reported, not removed: naturally recurring positions stay in TRAIN.
              "train_lines_sharing_a_final_position": len(raw) - len(set(raw)),
              "train_lines_sharing_a_final_orbit": len(orbit_rep) - len(set(orbit_rep))}
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
    read_stats: dict[str, dict[str, int]] = {}
    for number, (path, max_plies) in enumerate(books, start=1):
        stats: dict[str, int] = {}
        parsed = read_book(path, book=number, max_plies=max_plies, stats=stats)
        per_book[str(number)] = len(parsed)
        read_stats[str(number)] = stats
        lines.extend(parsed)
    train, eval_set, counts = split_lines(lines, eval_lines=eval_lines, salt=salt)
    counts["read_by_book"] = read_stats
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
