from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import chess
import chess.pgn


@dataclass(frozen=True)
class OpeningConfig:
    # If set, sample openings from this file.
    # Supported:
    # - .bin (Polyglot)
    # - .pgn
    # - .pgn.zip (a zip containing one or more .pgn files)
    opening_book_path: str | None = None

    # How many plies to apply from the book (max). For the stockfish 2-move book,
    # this would typically be 4 (2 full moves).
    opening_book_max_plies: int = 4

    # For PGN-based books: maximum number of games to parse (0 means all; can be slow).
    opening_book_max_games: int = 200_000

    # Probability of using the opening book (vs random-start plies).
    opening_book_prob: float = 1.0

    # Optional second opening book (e.g. a deeper 8-move book).
    opening_book_path_2: str | None = None
    opening_book_max_plies_2: int = 16
    opening_book_max_games_2: int = 200_000
    # Fraction of book games that use book 2 (the remainder use book 1).
    opening_book_mix_prob_2: float = 0.0

    # If >0, play this many random legal plies from the start position.
    random_start_plies: int = 0


def _random_playout_from_start(*, rng, plies: int) -> chess.Board:
    b = chess.Board()
    for _ in range(int(plies)):
        if b.is_game_over():
            break
        moves = list(b.legal_moves)
        if not moves:
            break
        b.push(moves[int(rng.integers(0, len(moves)))])
    return b


def _iter_pgn_bytes_from_path(path: Path) -> Iterable[bytes]:
    """Yield raw PGN bytes from a file that may be a .pgn or .pgn.zip."""
    suffixes = "".join(path.suffixes).lower()

    if suffixes.endswith(".pgn"):
        yield path.read_bytes()
        return

    if suffixes.endswith(".pgn.zip") or suffixes.endswith(".zip"):
        z = zipfile.ZipFile(path)
        # Heuristic: read all members ending in .pgn
        pgn_members = [n for n in z.namelist() if n.lower().endswith(".pgn")]
        if not pgn_members:
            raise ValueError(f"No .pgn files found in zip: {path}")
        for name in pgn_members:
            with z.open(name, "r") as f:
                yield f.read()
        return

    raise ValueError(f"Unsupported opening book format: {path}")


@lru_cache(maxsize=8)
def _load_pgn_opening_sequences(
    path_str: str, *, max_plies: int, max_games: int
) -> tuple[list[tuple[str, ...]], list[int]]:
    """Load PGN games into a weighted list of UCI move sequences.

    We aggregate identical prefixes into counts so large PGN books remain manageable.
    Cache keys are path-based, so replacing a book in place requires a process
    restart. In practice we version book filenames (`..._v2.pgn.zip`) when
    changing assets, which keeps cache invalidation explicit.

    Returns (seqs, weights).
    """
    path = Path(path_str)
    counts: dict[tuple[str, ...], int] = {}

    games_read = 0
    for blob in _iter_pgn_bytes_from_path(path):
        pgn_io = io.StringIO(blob.decode("utf-8", errors="ignore"))
        while True:
            game = chess.pgn.read_game(pgn_io)
            if game is None:
                break
            node = game
            seq: list[str] = []
            for _ in range(int(max_plies)):
                nxt = node.variation(0) if node.variations else None
                if nxt is None:
                    break
                mv = nxt.move
                seq.append(mv.uci())
                node = nxt

            if seq:
                key = tuple(seq)
                counts[key] = counts.get(key, 0) + 1

            games_read += 1
            if int(max_games) > 0 and games_read >= int(max_games):
                break

        if int(max_games) > 0 and games_read >= int(max_games):
            break

    if not counts:
        return ([], [])

    seqs = list(counts.keys())
    weights = [counts[s] for s in seqs]
    return (seqs, weights)


def _sample_from_pgn(*, rng, path: str, max_plies: int, max_games: int) -> chess.Board:
    seqs, weights = _load_pgn_opening_sequences(path, max_plies=int(max_plies), max_games=int(max_games))
    if not seqs:
        return chess.Board()

    import numpy as np

    p = np.array(weights, dtype=np.float64)
    p /= float(p.sum())
    idx = int(rng.choice(np.arange(len(seqs)), p=p))

    b = chess.Board()
    for u in seqs[idx]:
        mv = chess.Move.from_uci(u)
        if mv not in b.legal_moves:
            break
        b.push(mv)
    return b


def _sample_from_polyglot(*, rng, path: str, max_plies: int) -> chess.Board:
    import chess.polyglot

    b = chess.Board()

    with chess.polyglot.open_reader(path) as reader:
        for _ in range(int(max_plies)):
            if b.is_game_over():
                break
            entries = list(reader.find_all(b))
            if not entries:
                break

            # Polyglot entries have a weight; sample proportional to that weight.
            import numpy as np

            ws = np.array([max(0, int(e.weight)) for e in entries], dtype=np.float64)
            s = float(ws.sum())
            if s <= 0:
                mv = entries[int(rng.integers(0, len(entries)))].move
            else:
                ws /= s
                mv = entries[int(rng.choice(np.arange(len(entries)), p=ws))].move

            if mv not in b.legal_moves:
                break
            b.push(mv)

    return b


def _sample_book(*, rng, path: str, max_plies: int, max_games: int) -> chess.Board:
    p = Path(path)
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".bin"):
        board = _sample_from_polyglot(rng=rng, path=str(p), max_plies=max_plies)
    elif suffixes.endswith(".pgn") or suffixes.endswith(".pgn.zip") or suffixes.endswith(".zip"):
        board = _sample_from_pgn(rng=rng, path=str(p), max_plies=max_plies, max_games=max_games)
    else:
        raise ValueError(f"Unknown opening book format: {p}")

    if not board.move_stack:
        raise ValueError(f"Opening book produced no usable opening moves: {p}")
    return board


def make_starting_board(*, rng, cfg: OpeningConfig) -> chess.Board:
    """Create a starting position according to config.

    Priority:
    - with probability opening_book_prob, use opening book if provided
      - among book games, use book 2 with opening_book_mix_prob_2, else book 1
    - otherwise use random_start_plies if >0
    - otherwise startpos
    """
    if cfg.opening_book_path and float(rng.random()) < float(cfg.opening_book_prob):
        use_book2 = (
            cfg.opening_book_path_2
            and float(cfg.opening_book_mix_prob_2) > 0.0
            and float(rng.random()) < float(cfg.opening_book_mix_prob_2)
        )
        if use_book2:
            return _sample_book(
                rng=rng,
                path=str(cfg.opening_book_path_2),
                max_plies=int(cfg.opening_book_max_plies_2),
                max_games=int(cfg.opening_book_max_games_2),
            )
        return _sample_book(
            rng=rng,
            path=str(cfg.opening_book_path),
            max_plies=int(cfg.opening_book_max_plies),
            max_games=int(cfg.opening_book_max_games),
        )

    if int(cfg.random_start_plies) > 0:
        return _random_playout_from_start(rng=rng, plies=int(cfg.random_start_plies))

    return chess.Board()
