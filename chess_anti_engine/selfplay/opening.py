from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chess
import chess.pgn

_log = logging.getLogger(__name__)


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

  # Blind-spot FEN seeding: if set, with probability opening_fen_prob the game
  # starts from a seed sampled uniformly from this file (one seed per line;
  # blanks, '#' comment lines and inline '# ...' comments are skipped) instead
  # of the book/random/startpos flow. Each line is either a plain FEN or
  # '<start_fen> | <uci moves>' (see seed_board_from_line): the latter replays
  # the moves so the seed carries real LC0 history rather than repeat-filled
  # planes. Used to replay positions the net has historically misplayed (see
  # data/blindspot_fens_v1.txt) so refutations enter the training distribution
  # — search cannot rescue value-blind positions, only data can
  # (docs/experiment_ledger.md, 2026-07-03 probe).
    opening_fen_list_path: str | None = None
    opening_fen_prob: float = 0.0

  # When a FEN start is used, force the net to play the FEN's side to move
  # (the seat that blundered there). Without this, color alternation puts the
  # net on the punisher side half the time — useful data, but the point of
  # seeding is making the net face the decision it historically got wrong.
  # Net-vs-net selfplay slots are unaffected (the net plays both seats).
    opening_fen_net_side_to_move: bool = True


@dataclass(frozen=True)
class OpeningStart:
    board: chess.Board
    source: str


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

    if suffixes.endswith((".pgn.zip", ".zip")):
        with zipfile.ZipFile(path) as z:
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


def seed_board_from_line(line: str) -> chess.Board:
    """Build the seed (terminal) board from one FEN-list line.

    Line grammar (backward-compatible):
      ``<fen>``                    plain seed — no move history (repeat-fill).
      ``<start_fen> | <uci> ...``  start from ``start_fen`` and replay the UCI
                                   moves; the TERMINAL position is the seed and
                                   its move_stack carries real LC0 history (the
                                   ~8 preceding plies), instead of the empty
                                   stack a bare FEN gets (which the encoder
                                   repeat-fills — 84/175 planes wrong, a ~14%
                                   top-1 flip; see docs/experiment_ledger.md).

    Raises ValueError on an unparseable FEN or an illegal move in the line.
    """
    fen_part, _, moves_part = line.partition("|")
    board = chess.Board(fen_part.strip())  # raises ValueError on a bad FEN
    for tok in moves_part.split():
        move = chess.Move.from_uci(tok)
        if move not in board.legal_moves:
            raise ValueError(f"illegal move {tok!r} in seed line")
        board.push(move)
    return board


def _fen_reject_reason(line: str) -> str | None:
    """Why this seed line is unusable, or None if it's fine.

    The seed is the TERMINAL position after replaying any history moves; it must
    be parseable (FEN + legal moves), legal, non-terminal (under claim-draw,
    matching CBoard's cboard_is_game_over which treats halfmove_clock>=100 as
    over), and have >=2 legal moves (a forced position is skipped by
    _apply_forced_moves, so the net never faces the seeded decision).
    """
    try:
        board = seed_board_from_line(line)
    except ValueError as exc:
        return str(exc) if "illegal move" in str(exc) else "unparseable FEN"
    if not board.is_valid():
        return "illegal position"  # missing kings, impossible pawns, etc.
    if board.is_game_over(claim_draw=True):
        return "terminal position"
    if board.legal_moves.count() < 2:
        return "forced (single legal move)"
    return None


@lru_cache(maxsize=8)
def _load_fen_list(path_str: str) -> tuple[str, ...]:
    """Load a seed list (one seed per line, '#' comments, blanks skipped).

    Each line is a seed in the ``seed_board_from_line`` grammar — a plain FEN
    or ``<start_fen> | <uci moves>`` for real history — with an optional inline
    ``# ...`` provenance comment (src/round/ply from mining) stripped here.
    Malformed/unusable lines are SKIPPED with a logged warning rather than
    raised — one bad line in a hand-edited or future seed file must not crash
    the whole worker fleet (Codex review). We raise only when the file yields
    ZERO usable seeds, so an empty/all-comment/all-bad file fails fast at
    startup (warm_opening_book_cache) instead of mid-run. Cache keys are
    path-based like the PGN loader: version the filename to replace a list.
    """
    fens: list[str] = []
    skipped: list[str] = []
    # utf-8-sig: strip a leading BOM (editors/exporters on Windows add one) so
    # the first FEN — or a first-line '# comment' — is not corrupted.
    for lineno, raw in enumerate(Path(path_str).read_text(encoding="utf-8-sig").splitlines(), 1):
        # Strip an inline '# provenance' comment (neither FEN nor UCI contains
        # '#'), then whitespace; blank / whole-line-comment lines drop out.
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        reason = _fen_reject_reason(line)
        if reason is not None:
            skipped.append(f"{path_str}:{lineno} ({reason}): {line!r}")
            continue
        fens.append(line)
    if skipped:
        _log.warning(
            "opening FEN list %s: skipped %d unusable seed(s):\n  %s",
            path_str, len(skipped), "\n  ".join(skipped),
        )
    if not fens:
        raise ValueError(
            f"opening FEN list {path_str} has no usable seeds "
            f"(skipped {len(skipped)}); fix the file and restart",
        )
    return tuple(fens)


def _sample_fen_list(*, rng, path: str) -> chess.Board:
    fens = _load_fen_list(path)
    return seed_board_from_line(fens[int(rng.integers(0, len(fens)))])


def _sample_book(*, rng, path: str, max_plies: int, max_games: int) -> chess.Board:
    p = Path(path)
    suffixes = "".join(p.suffixes).lower()
    if suffixes.endswith(".bin"):
        board = _sample_from_polyglot(rng=rng, path=str(p), max_plies=max_plies)
    elif suffixes.endswith((".pgn", ".pgn.zip", ".zip")):
        board = _sample_from_pgn(rng=rng, path=str(p), max_plies=max_plies, max_games=max_games)
    else:
        raise ValueError(f"Unknown opening book format: {p}")

    if not board.move_stack:
        raise ValueError(f"Opening book produced no usable opening moves: {p}")
    return board


def warm_opening_book_cache(cfg: OpeningConfig) -> None:
    """Preload PGN opening books before launching many selfplay threads."""
    if cfg.opening_fen_list_path:
        _load_fen_list(str(cfg.opening_fen_list_path))
    for path, max_plies, max_games in (
        (cfg.opening_book_path, cfg.opening_book_max_plies, cfg.opening_book_max_games),
        (cfg.opening_book_path_2, cfg.opening_book_max_plies_2, cfg.opening_book_max_games_2),
    ):
        if not path:
            continue
        suffixes = "".join(Path(str(path)).suffixes).lower()
        if suffixes.endswith((".pgn", ".pgn.zip", ".zip")):
            _load_pgn_opening_sequences(
                str(path),
                max_plies=int(max_plies),
                max_games=int(max_games),
            )


def sample_starting_board(*, rng, cfg: OpeningConfig) -> OpeningStart:
    """Create a starting position and label its low-cardinality source.

    Priority:
    - with probability opening_fen_prob, use the blind-spot FEN list if provided
    - with probability opening_book_prob, use opening book if provided
      - among book games, use book 2 with opening_book_mix_prob_2, else book 1
    - otherwise use random_start_plies if >0
    - otherwise startpos
    """
    if (
        cfg.opening_fen_list_path
        and float(cfg.opening_fen_prob) > 0.0
        and float(rng.random()) < float(cfg.opening_fen_prob)
    ):
        return OpeningStart(
            board=_sample_fen_list(rng=rng, path=str(cfg.opening_fen_list_path)),
            source="fenlist",
        )

    if cfg.opening_book_path and float(rng.random()) < float(cfg.opening_book_prob):
        use_book2 = (
            cfg.opening_book_path_2
            and float(cfg.opening_book_mix_prob_2) > 0.0
            and float(rng.random()) < float(cfg.opening_book_mix_prob_2)
        )
        if use_book2:
            return OpeningStart(
                board=_sample_book(
                    rng=rng,
                    path=str(cfg.opening_book_path_2),
                    max_plies=int(cfg.opening_book_max_plies_2),
                    max_games=int(cfg.opening_book_max_games_2),
                ),
                source="book2",
            )
        return OpeningStart(
            board=_sample_book(
                rng=rng,
                path=str(cfg.opening_book_path),
                max_plies=int(cfg.opening_book_max_plies),
                max_games=int(cfg.opening_book_max_games),
            ),
            source="book1",
        )

    if int(cfg.random_start_plies) > 0:
        return OpeningStart(
            board=_random_playout_from_start(rng=rng, plies=int(cfg.random_start_plies)),
            source="random",
        )

    return OpeningStart(board=chess.Board(), source="start")


def make_starting_board(*, rng, cfg: OpeningConfig) -> chess.Board:
    return sample_starting_board(rng=rng, cfg=cfg).board
