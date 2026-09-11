#!/usr/bin/env python3
"""Bank a frozen network's root WDL and one-ply AVI-style value backup.

This is a bounded mechanism/collection instrument, not a training launcher. Input
positions use DeepFin's replayable seed format (plain FEN or ``<start_fen> | <moves>``),
so history-bearing rows keep the same move stack the model encoder consumes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.mcts.one_ply import (
    evaluate_wdl_probabilities,
    one_ply_value_backups,
)
from chess_anti_engine.selfplay.opening import _load_fen_list, seed_board_from_line
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint


SCHEMA = 1
KIND = "avi_one_ply_value_diagnostic"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_unique_boards(path: Path, *, max_positions: int) -> tuple[list[chess.Board], int]:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    lines = _load_fen_list(str(path))
    boards: list[chess.Board] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for line in lines:
        board = seed_board_from_line(line)
        key = (board.fen(), tuple(move.uci() for move in board.move_stack))
        if key in seen:
            continue
        seen.add(key)
        boards.append(board)
        if len(boards) >= max_positions:
            break
    if not boards:
        raise ValueError("no usable non-terminal positions")
    return boards, len(lines)


def q_from_wdl(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(f"expected [N,3] WDL array, got {values.shape!r}")
    return values[:, 0] - values[:, 2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--positions", required=True)
    parser.add_argument("--out", required=True, help="New .npz output path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-positions", type=int, default=4096)
    parser.add_argument(
        "--claim-draws",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Treat claimable threefold/50-move draws as terminal; twofold is never terminal",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    positions = Path(args.positions).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    summary_path = out.with_suffix(".json")
    if out.suffix != ".npz":
        raise SystemExit("--out must end in .npz")
    if out.exists() or summary_path.exists():
        raise SystemExit("refusing to overwrite existing output")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    boards, source_lines = load_unique_boards(
        positions, max_positions=args.max_positions
    )
    started = time.monotonic()
    model = load_model_from_checkpoint(str(checkpoint), device=str(args.device))
    model.eval()
    evaluator = LocalModelEvaluator(model, device=str(args.device), use_amp=True)
    history_encoding = getattr(model, "input_history_encoding", None)
    extra_features = getattr(model, "input_extra_features", None)
    compute_relations = bool(getattr(model, "use_dynamic_relations", False))

    root_wdl = evaluate_wdl_probabilities(
        boards,
        evaluator,
        batch_size=args.batch_size,
        input_history_encoding=history_encoding,
        input_extra_features=extra_features,
        compute_relations=compute_relations,
    )
    backups = one_ply_value_backups(
        boards,
        evaluator,
        batch_size=args.batch_size,
        input_history_encoding=history_encoding,
        input_extra_features=extra_features,
        compute_relations=compute_relations,
        claim_draws=bool(args.claim_draws),
    )
    backup_wdl = np.stack([backup.wdl for backup in backups], axis=0)
    root_q = q_from_wdl(root_wdl)
    backup_q = q_from_wdl(backup_wdl)
    abs_q_delta = np.abs(backup_q - root_q)
    total_variation = 0.5 * np.abs(backup_wdl - root_wdl).sum(axis=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        fen=np.asarray([board.fen() for board in boards]),
        history_uci=np.asarray(
            [" ".join(move.uci() for move in board.move_stack) for board in boards]
        ),
        root_wdl=np.asarray(root_wdl, dtype=np.float32),
        backup_wdl=np.asarray(backup_wdl, dtype=np.float32),
        chosen_move=np.asarray([backup.move.uci() for backup in backups]),
        chosen_q=np.asarray([backup.q for backup in backups], dtype=np.float32),
        legal_moves=np.asarray([backup.legal_moves for backup in backups], dtype=np.int16),
        terminal_children=np.asarray(
            [backup.terminal_children for backup in backups], dtype=np.int16
        ),
        evaluated_children=np.asarray(
            [backup.evaluated_children for backup in backups], dtype=np.int16
        ),
        selected_terminal=np.asarray(
            [backup.selected_terminal for backup in backups], dtype=np.bool_
        ),
    )
    summary = {
        "schema": SCHEMA,
        "kind": KIND,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "positions": str(positions),
        "positions_sha256": file_sha256(positions),
        "source_lines": int(source_lines),
        "rows": len(boards),
        "max_positions": int(args.max_positions),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
        "claim_draws": bool(args.claim_draws),
        "input_history_encoding": history_encoding,
        "input_extra_features": extra_features,
        "compute_relations": compute_relations,
        "legal_children": int(sum(backup.legal_moves for backup in backups)),
        "network_evaluated_children": int(
            sum(backup.evaluated_children for backup in backups)
        ),
        "terminal_children": int(sum(backup.terminal_children for backup in backups)),
        "selected_terminal_rows": int(
            sum(backup.selected_terminal for backup in backups)
        ),
        "mean_total_variation_root_vs_backup": float(total_variation.mean()),
        "mean_abs_q_delta_root_vs_backup": float(abs_q_delta.mean()),
        "p95_abs_q_delta_root_vs_backup": float(np.quantile(abs_q_delta, 0.95)),
        "max_abs_q_delta_root_vs_backup": float(abs_q_delta.max()),
        "elapsed_seconds": float(time.monotonic() - started),
        "npz": str(out),
        "npz_sha256": file_sha256(out),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
