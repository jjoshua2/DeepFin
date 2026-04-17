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

import re
from dataclasses import dataclass, field
from pathlib import Path

import chess
import numpy as np
import torch

from chess_anti_engine.mcts import MCTSConfig, run_mcts_many
from chess_anti_engine.moves import index_to_move


# ---------------------------------------------------------------------------
# Puzzle data structures
# ---------------------------------------------------------------------------

@dataclass
class Puzzle:
    board: chess.Board
    best_moves: list[chess.Move]  # any of these is accepted as correct
    puzzle_id: str = ""


@dataclass
class PuzzleResult:
    total: int
    correct: int
    accuracy: float  # correct / total


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
    mcts_type: str = "puct",
    batch_size: int = 32,
    rng: np.random.Generator | None = None,
) -> PuzzleResult:
    """Evaluate the model on a puzzle suite.

    For each puzzle, run MCTS with the given simulation budget and check
    whether the top move matches any of the accepted best moves.

    Args:
        model: The network to evaluate (already on device or will be moved).
        suite: Loaded puzzle suite.
        device: Torch device string.
        mcts_simulations: Simulation budget per puzzle position.
        mcts_type: MCTS variant ("puct" only for now; gumbel could be added).
        batch_size: Number of positions to evaluate in one MCTS batch.
        rng: Numpy RNG; a default is created if None.

    Returns:
        PuzzleResult with total, correct, accuracy.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    model.eval()
    cfg = MCTSConfig(simulations=mcts_simulations, temperature=0.0)

    correct = 0
    total = len(suite)

    # Process in batches for GPU efficiency.
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_puzzles = suite.puzzles[start:end]
        boards = [p.board.copy() for p in batch_puzzles]

        _probs, actions, _vals, _masks = run_mcts_many(
            model, boards, device=device, rng=rng, cfg=cfg,
        )

        for j, (puzzle, action_idx) in enumerate(zip(batch_puzzles, actions)):
            chosen = index_to_move(int(action_idx), puzzle.board)
            if chosen in puzzle.best_moves:
                correct += 1

    accuracy = float(correct) / max(1, total)
    return PuzzleResult(total=total, correct=correct, accuracy=accuracy)
