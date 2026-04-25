"""Puzzle evaluation harness for detecting general chess ability regression.

Loads EPD puzzle files (FEN + best move), runs the network via MCTS on each
position, and reports top-1 accuracy.  Useful as a canary: if puzzle accuracy
drops while anti-engine win rate rises, the network is overspecializing.

EPD format expected (one per line):
    <FEN> bm <SAN_move>;  [optional id "...";]

Also supports a simpler format:
    <FEN> bm <UCI_move>;

Lines starting with '#' or blank lines are skipped.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import torch

from chess_anti_engine.mcts import MCTSConfig, run_mcts_many
from chess_anti_engine.moves import index_to_move

# Default rating buckets for Lichess-style evaluation, matching the LC0 blog
# (https://lczero.org/blog/2024/02/...) coarse buckets.
DEFAULT_RATING_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 1000),
    (1000, 1500),
    (1500, 2000),
    (2000, 2500),
    (2500, 3000),
    (3000, 9999),
)

# ---------------------------------------------------------------------------
# Puzzle data structures
# ---------------------------------------------------------------------------

@dataclass
class Puzzle:
    board: chess.Board
    best_moves: list[chess.Move]  # any of these is accepted as correct
    puzzle_id: str = ""
    rating: int | None = None  # Lichess Elo, optional (None for non-Lichess sources)
    themes: tuple[str, ...] = ()


@dataclass
class PuzzleResult:
    total: int
    correct: int
    accuracy: float  # correct / total
    # Per-rating-bucket accuracy: ordered list of (low, high, total, correct, accuracy).
    # Empty when the suite has no rating annotations.
    by_rating: list[tuple[int, int, int, int, float]] = field(default_factory=list)


@dataclass
class PuzzleSuite:
    """A loaded set of puzzles ready for evaluation."""

    puzzles: list[Puzzle] = field(default_factory=list)
    name: str = ""

    def __len__(self) -> int:
        return len(self.puzzles)


# ---------------------------------------------------------------------------
# EPD parsing
# ---------------------------------------------------------------------------

def _parse_san_or_uci(board: chess.Board, token: str) -> chess.Move | None:
    """Try to parse a move token as SAN first, then UCI."""
  # Strip trailing semicolons / whitespace.
    token = token.strip().rstrip(";").strip()
    if not token:
        return None
  # Try SAN.
    try:
        return board.parse_san(token)
    except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError):
        pass
  # Try UCI.
    try:
        m = chess.Move.from_uci(token)
        if m in board.legal_moves:
            return m
    except (ValueError, chess.InvalidMoveError):
        pass
    return None


def load_epd(path: str | Path) -> PuzzleSuite:
    """Load puzzles from an EPD file.

    Supports standard EPD with ``bm`` (best move) opcode.
    Multiple best moves separated by spaces are all accepted.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Puzzle file not found: {p}")

    puzzles: list[Puzzle] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

  # EPD: everything before "bm" is the FEN (possibly 4-field or 6-field).
        bm_idx = line.find(" bm ")
        if bm_idx < 0:
            continue  # skip lines without bm opcode

        fen_part = line[:bm_idx].strip()
        rest = line[bm_idx + 4:]  # after " bm "

  # FEN may have 4 or 6 fields.  python-chess needs at least 4.
        fen_fields = fen_part.split()
        if len(fen_fields) < 4:
            continue
  # Pad to 6 fields if needed (halfmove=0, fullmove=1).
        while len(fen_fields) < 6:
            fen_fields.append("0" if len(fen_fields) == 4 else "1")
        fen = " ".join(fen_fields)

        try:
            board = chess.Board(fen)
        except ValueError:
            continue

  # Parse best moves: everything up to the first ";" that's followed by
  # another opcode (e.g. "id") or end of line.
  # Moves may be separated by spaces.
        bm_section = re.split(r";", rest)[0].strip()
        tokens = bm_section.split()
        best_moves: list[chess.Move] = []
        for tok in tokens:
            m = _parse_san_or_uci(board, tok)
            if m is not None:
                best_moves.append(m)

        if not best_moves:
            continue

  # Try to extract puzzle id.
        pid = ""
        id_match = re.search(r'id\s+"([^"]+)"', rest)
        if id_match:
            pid = id_match.group(1)

        puzzles.append(Puzzle(board=board, best_moves=best_moves, puzzle_id=pid))

    return PuzzleSuite(puzzles=puzzles, name=p.stem)


def load_lichess_csv(
    path: str | Path,
    *,
    max_puzzles: int | None = None,
    min_rating: int | None = None,
    max_rating: int | None = None,
    themes_filter: tuple[str, ...] = (),
) -> PuzzleSuite:
    """Load puzzles from a Lichess puzzle CSV dump.

    Format (https://database.lichess.org/#puzzles):
        PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

    The FEN is the position *before* the opponent's setup move; ``Moves`` is a
    space-separated UCI sequence whose first move is that setup. We apply the
    setup, then take the second move as the puzzle's expected best move (the
    user's first reply). Multi-move puzzle continuations are ignored — top-1
    accuracy on the post-setup position matches LC0's blog methodology.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Puzzle CSV not found: {p}")

    puzzles: list[Puzzle] = []
    with p.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                rating = int(row.get("Rating", "") or 0)
            except ValueError:
                rating = 0
            if min_rating is not None and rating < min_rating:
                continue
            if max_rating is not None and rating > max_rating:
                continue

            themes = tuple((row.get("Themes") or "").split())
            if themes_filter and not any(t in themes for t in themes_filter):
                continue

            fen = (row.get("FEN") or "").strip()
            moves_str = (row.get("Moves") or "").strip()
            if not fen or not moves_str:
                continue

            try:
                board = chess.Board(fen)
            except ValueError:
                continue

            move_tokens = moves_str.split()
            if len(move_tokens) < 2:
                continue

            # Apply opponent's setup move; the next move is the user's expected reply.
            try:
                setup = chess.Move.from_uci(move_tokens[0])
            except (ValueError, chess.InvalidMoveError):
                continue
            if setup not in board.legal_moves:
                continue
            board.push(setup)

            try:
                best = chess.Move.from_uci(move_tokens[1])
            except (ValueError, chess.InvalidMoveError):
                continue
            if best not in board.legal_moves:
                continue

            puzzles.append(Puzzle(
                board=board,
                best_moves=[best],
                puzzle_id=(row.get("PuzzleId") or "").strip(),
                rating=rating if rating > 0 else None,
                themes=themes,
            ))

            if max_puzzles is not None and len(puzzles) >= max_puzzles:
                break

    return PuzzleSuite(puzzles=puzzles, name=p.stem)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_puzzle_eval(
    model: torch.nn.Module,
    suite: PuzzleSuite,
    *,
    device: str,
    mcts_simulations: int = 200,
    batch_size: int = 32,
    rng: np.random.Generator | None = None,
    rating_buckets: tuple[tuple[int, int], ...] = DEFAULT_RATING_BUCKETS,
) -> PuzzleResult:
    """Evaluate the model on a puzzle suite.

    For each puzzle, run MCTS with the given simulation budget and check
    whether the top move matches any of the accepted best moves. When the
    suite has rating annotations (Lichess CSV), also report per-bucket
    accuracy in ``PuzzleResult.by_rating``.

    Args:
        model: The network to evaluate (already on device or will be moved).
        suite: Loaded puzzle suite.
        device: Torch device string.
        mcts_simulations: Simulation budget per puzzle position.
        batch_size: Number of positions to evaluate in one MCTS batch.
        rng: Numpy RNG; a default is created if None.
        rating_buckets: Half-open ``[low, high)`` bands. Ignored when no
            puzzle has a rating set.

    Returns:
        PuzzleResult with total, correct, accuracy, and (when applicable)
        bucketed accuracy.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    model.eval()
    cfg = MCTSConfig(simulations=mcts_simulations, temperature=0.0)

    total = len(suite)
    correct_flags = [False] * total

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_puzzles = suite.puzzles[start:end]
        boards = [p.board.copy() for p in batch_puzzles]

        _, actions, *_ = run_mcts_many(
            model, boards, device=device, rng=rng, cfg=cfg,
        )

        for offset, (puzzle, action_idx) in enumerate(zip(batch_puzzles, actions)):
            chosen = index_to_move(int(action_idx), puzzle.board)
            if chosen in puzzle.best_moves:
                correct_flags[start + offset] = True

    correct = sum(correct_flags)
    accuracy = float(correct) / max(1, total)

    by_rating: list[tuple[int, int, int, int, float]] = []
    if any(p.rating is not None for p in suite.puzzles):
        for low, high in rating_buckets:
            b_total = 0
            b_correct = 0
            for puzzle, ok in zip(suite.puzzles, correct_flags):
                r = puzzle.rating
                if r is None or r < low or r >= high:
                    continue
                b_total += 1
                if ok:
                    b_correct += 1
            if b_total > 0:
                by_rating.append((low, high, b_total, b_correct, b_correct / b_total))

    return PuzzleResult(
        total=total, correct=correct, accuracy=accuracy, by_rating=by_rating,
    )


@torch.no_grad()
def run_value_head_puzzle_eval(
    model: torch.nn.Module,
    suite: PuzzleSuite,
    *,
    device: str,
    batch_size: int = 256,
    rating_buckets: tuple[tuple[int, int], ...] = DEFAULT_RATING_BUCKETS,
) -> PuzzleResult:
    """Score the value head by argmax over legal-move push-evals (LC0-blog).

    For each puzzle, push every legal move and query the value head on the
    resulting position. After a push the side-to-move flips, so the
    *opponent's* P(W) on the resulting board corresponds to the mover's
    P(L). The candidate that minimises that — equivalently, that maximises
    ``P(L_opp) - P(W_opp)`` — is the value head's pick. Score 1 if it
    matches any of the puzzle's accepted best moves.

    This is independent of the policy head; results test value-head
    correctness only.
    """
    from chess_anti_engine.encoding import encode_position  # local import; slow

    model.eval()
    total = len(suite)
    correct_flags = [False] * total

    # Flatten (puzzle_idx, candidate_move) pairs across the suite, then batch.
    flat_idx: list[int] = []
    flat_moves: list[chess.Move] = []
    flat_x: list[np.ndarray] = []
    flat_groups: list[tuple[int, int]] = []  # (puzzle_idx, # legal moves) — used to slice scores

    legal_per_puzzle: list[list[chess.Move]] = []
    for puzzle in suite.puzzles:
        legal = list(puzzle.board.legal_moves)
        legal_per_puzzle.append(legal)
        for mv in legal:
            puzzle.board.push(mv)
            flat_x.append(encode_position(puzzle.board))
            puzzle.board.pop()
            flat_moves.append(mv)
        flat_groups.append((len(flat_idx), len(legal)))
        flat_idx.extend([len(legal_per_puzzle) - 1] * len(legal))

    # Batched value-head forward.
    pos_count = len(flat_x)
    scores = np.empty(pos_count, dtype=np.float32)
    for start in range(0, pos_count, batch_size):
        end = min(start + batch_size, pos_count)
        x = torch.from_numpy(np.stack(flat_x[start:end])).to(device, non_blocking=True)
        out = model(x)
        wdl_logits = out["wdl"] if isinstance(out, dict) else out[1]
        wdl_p = torch.softmax(wdl_logits, dim=-1).float().cpu().numpy()
        # After-push board is opponent's POV: P_opp(L) - P_opp(W) = mover's signed value.
        scores[start:end] = wdl_p[:, 2] - wdl_p[:, 0]

    for puzzle_idx, (slot_start, n_legal) in enumerate(flat_groups):
        if n_legal == 0:
            continue
        slot = scores[slot_start:slot_start + n_legal]
        best = int(np.argmax(slot))
        chosen = flat_moves[slot_start + best]
        if chosen in suite.puzzles[puzzle_idx].best_moves:
            correct_flags[puzzle_idx] = True

    correct = sum(correct_flags)
    accuracy = float(correct) / max(1, total)

    by_rating: list[tuple[int, int, int, int, float]] = []
    if any(p.rating is not None for p in suite.puzzles):
        for low, high in rating_buckets:
            b_total = 0
            b_correct = 0
            for puzzle, ok in zip(suite.puzzles, correct_flags):
                r = puzzle.rating
                if r is None or r < low or r >= high:
                    continue
                b_total += 1
                if ok:
                    b_correct += 1
            if b_total > 0:
                by_rating.append((low, high, b_total, b_correct, b_correct / b_total))

    return PuzzleResult(
        total=total, correct=correct, accuracy=accuracy, by_rating=by_rating,
    )
