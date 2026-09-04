#!/usr/bin/env python3
"""Score one checkpoint on the frozen audit set under BOTH input encodings.

`scripts/value_regret.py --input-encoding X` and `scripts/audit_targets.py
--input-encoding X` each score one encoding with one head. This rig scores the
POLICY and VALUE heads across several encoding ARMS in a single model load and
dumps per-position regrets, which is what a paired game-clustered bootstrap
needs. It is the instrument behind the 2026-08-02 audit-v2 ledger entries and
the lineage/SWA tables that cite "arm `v2`".

Arms (only the INPUT ENCODING changes; positions, deep-SF labels, legal-move
indexing and metrics are identical across arms):

  v1      ≡ `--input-encoding fen_only`. `chess.Board(fen)` -> empty move
          stack: 93 of 175 planes structurally zero, colour flag wrong on
          ~51% of rows. The historical ruler.
  v1_stm  attribution arm, no CLI equivalent: v1 plus the TRUE colour flag
          (plane 108). Separates the colour flag from the history frames.
  v2      ≡ `--input-encoding stored`. The stored production row for the root,
          and children derived from it by the bit-exact `pov_flip_slot` shift.

The arm names are the ledger's published ones and are deliberately NOT renamed
to `fen_only`/`stored`; every printed line and every dumped record carries both
the arm and its encoding so a number cannot be lifted out of context.

⚑ A RULER CHANGE INVALIDATES ITS RECORDS — v1 and v2 numbers must never share
a table, a trend line or a threshold.

Metrics, pinned to the canonical tools:

  POLICY  raw-prior EXPECTED and top-1 deep-SF regret (cp) — `cand.raw` of
          scripts/audit_targets.py.
          ⚑ For any within-position paired contrast use EXPECTED, not top-1:
          top-1 is an argmax statistic that moves on only ~19% of positions
          when the encoding is swapped, which inflates the CI past the effect
          (ledger, 2026-08-02 METHOD RULE). Both are dumped; pick before you
          look.
  VALUE   1-ply value-match top-1 deep-SF regret (cp) — scripts/value_regret.py,
          TB-excluded (--min-pieces 8). This head has NO smooth analogue of
          expected regret, so a value-vs-policy comparison is not powered the
          way the policy leg is; that is a known gap, not something this rig
          hides.
          ⚑ BATCH-SIZE DEPENDENT: 0.66 cp between 128 and 256 on the same
          checkpoint. `--value-batch` defaults to the PINNED 256 and is echoed
          on every line.

Usage (~4 min/checkpoint at --gpu-mem-fraction 0.15):

    PYTHONPATH=. python3 scripts/score_audit_v2.py \\
        --checkpoint <ckpt.pt> --label <name> \\
        --out runs/audit_v2/score_<name>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import model_encoding_kwargs
from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.eval.audit import (
    legal_full_indices,
    load_audit_set,
    move_regrets,
)
from chess_anti_engine.eval.audit_history import (
    STM_PLANE,
    MatchedAuditRows,
    child_planes_for_encoding,
    default_matched_rows_path,
    root_input_planes,
)
from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.moves import POLICY_SIZE, policy_batch_to_full_if_needed
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

# arm -> the `--input-encoding` value it corresponds to (None = no CLI twin).
ARM_ENCODING: dict[str, str | None] = {
    "v1": "fen_only",
    "v1_stm": None,
    "v2": "stored",
}


def arm_tag(arm: str, value_batch: int, policy_batch: int) -> str:
    """The provenance stamp every report line and dumped record carries."""
    enc = ARM_ENCODING.get(arm) or "fen_only+true_stm"
    return f"[arm={arm} enc={enc} pb={policy_batch} vb={value_batch}]"


def _softmax_rows(a: np.ndarray) -> np.ndarray:
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def _piece_count(fen: str) -> int:
    return sum(1 for c in fen.split(" ", 1)[0] if c.isalpha())


def root_planes(arm: str, fen_planes: np.ndarray, stored: np.ndarray) -> np.ndarray:
    """Root input for `arm`.

    `v1` and `v2` DELEGATE to the shared encoding switch rather than
    reimplementing it: a second copy of those two branches is a second place a
    silent regression can live, and the `v1` BASELINE going wrong would make
    every v1-vs-v2 contrast read 0.00 while the run looked healthy.
    `v1_stm` is the only branch that is genuinely local — it is an attribution
    arm with no `--input-encoding` twin.
    """
    if arm == "v1_stm":
        out = np.array(fen_planes, copy=True)
        out[STM_PLANE] = stored[STM_PLANE]
        return out
    return root_input_planes(
        encoding=_require_encoding(arm), fen_planes=fen_planes, stored_planes=stored,
    )


def child_planes(
    arm: str, child_fen_planes: np.ndarray, stored_parent: np.ndarray,
) -> np.ndarray:
    """1-ply child input for `arm`. Same delegation as `root_planes`."""
    if arm == "v1_stm":
        out = np.array(child_fen_planes, copy=True)
        out[STM_PLANE] = 1.0 - float(stored_parent[STM_PLANE, 0, 0])
        return out
    return child_planes_for_encoding(
        encoding=_require_encoding(arm),
        child_fen_planes=child_fen_planes,
        stored_parent=stored_parent,
    )


def _require_encoding(arm: str) -> str:
    enc = ARM_ENCODING.get(arm)
    if enc is None:
        raise ValueError(f"arm {arm!r} has no --input-encoding equivalent")
    return enc


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    ap.add_argument("--matched-rows", type=Path, default=None,
                    help="matched-rows index (default: <audit-set>.matched_rows.npz); "
                         "built by scripts/match_audit_rows.py, not checked in")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", default=",".join(ARM_ENCODING))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--policy-batch", type=int, default=256)
    ap.add_argument("--value-batch", type=int, default=256,
                    help="⚑ the VALUE ruler is BATCH-SIZE DEPENDENT (0.66 cp between "
                         "128 and 256). 256 is the value PINNED by every banked "
                         "audit-v2 readout; changing it makes this run incomparable "
                         "to all of them.")
    ap.add_argument("--pos-chunk", type=int, default=128)
    ap.add_argument("--min-pieces", type=int, default=8)
    ap.add_argument("--gpu-mem-fraction", type=float, default=0.15)
    ap.add_argument("--max-positions", type=int, default=0)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARM_ENCODING]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; expected {list(ARM_ENCODING)}")

    if args.gpu_mem_fraction and str(args.device).startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(float(args.gpu_mem_fraction), 0)

    matched = MatchedAuditRows(
        args.matched_rows or default_matched_rows_path(args.audit_set)
    )
    positions = load_audit_set(args.audit_set)
    positions = [p for p in positions if _piece_count(p.fen) >= args.min_pieces]
    # Every arm scores the SAME rows, including the v1 arm, so the arms stay
    # paired: restricting only the v2 arm to matched rows would make the v1-vs-v2
    # contrast a comparison of two different position sets.
    positions = [p for p in positions if p.key in matched]
    if args.max_positions:
        positions = positions[: args.max_positions]
    n = len(positions)
    if not n:
        raise SystemExit("no positions left after min-pieces and matched-row filters")
    # Resolve the conservative cluster identity before loading a checkpoint.
    # The accessor is fail-closed on legacy/masked artifacts, so a bad index
    # cannot spend the GPU pass and then emit first-match or zero-filled game
    # clusters.
    game_cluster_ids = [matched.game_cluster_id(p.key) for p in positions]
    print(f"[set] n={n} (min_pieces>={args.min_pieces}, matched-only; "
          f"{matched.n_matched}/{matched.n_audit_rows} rows matched)")

    stored = np.stack([matched.stored_row(p.key) for p in positions])

    model = load_model_from_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    enc_kwargs = model_encoding_kwargs(model)
    matched.require_model_compatible(enc_kwargs)
    if bool(getattr(model, "use_dynamic_relations", False)):
        raise SystemExit(
            "dynamic-relation checkpoints are not supported: the relation tensor "
            "is rebuilt from the bare board and would contradict spliced history"
        )
    ev = LocalModelEvaluator(model, device=args.device)
    pol_enc = str(getattr(model, "policy_encoding", "lc0_1858"))
    print(f"[model] {args.label} enc={enc_kwargs}")

    # FEN-only root encodings and per-position legal data, computed once.
    boards = [chess.Board(p.fen) for p in positions]
    root_fen = np.stack([
        encode_cboard(CBoard.from_board(b), **enc_kwargs) for b in boards
    ]).astype(np.float32)
    legal = [legal_full_indices(b) for b in boards]
    regrets = [move_regrets(p, legal[i][0]) for i, p in enumerate(positions)]

    out: dict[str, object] = {
        "label": args.label,
        "checkpoint": args.checkpoint,
        "audit_set": str(args.audit_set),
        "matched_rows": str(matched.path),
        "n": n,
        "min_pieces": args.min_pieces,
        "policy_batch": args.policy_batch,
        "value_batch": args.value_batch,
        "arm_encoding": {a: ARM_ENCODING[a] for a in arms},
        "key": [p.key for p in positions],
        "game_cluster_id": game_cluster_ids,
        "phase": [int(p.phase) for p in positions],
    }
    arms_out: dict[str, dict[str, object]] = {}
    out["arms"] = arms_out

    for arm in arms:
        tag = arm_tag(arm, int(args.value_batch), int(args.policy_batch))

        # ---------------- POLICY (root only) ----------------
        expected = np.full(n, np.nan)
        policy_top1 = np.full(n, np.nan)
        roots = np.stack([root_planes(arm, root_fen[i], stored[i]) for i in range(n)])
        for s in range(0, n, args.policy_batch):
            xb = roots[s:s + args.policy_batch]
            with torch.no_grad():
                logits, _ = ev.evaluate_encoded(xb)
            logits = np.asarray(logits, dtype=np.float32)
            if logits.shape[1] != POLICY_SIZE:
                logits = policy_batch_to_full_if_needed(
                    logits, policy_encoding=pol_enc, fill_value=-1e9,
                )
            for j in range(xb.shape[0]):
                i = s + j
                lg = logits[j, legal[i][1]].astype(np.float64)
                lg -= lg.max()
                probs = np.exp(lg)
                probs /= probs.sum()
                expected[i] = float((probs * regrets[i]).sum())
                policy_top1[i] = float(regrets[i][int(np.argmax(probs))])
        del roots

        # ---------------- VALUE (1-ply children) ----------------
        value_top1 = np.full(n, np.nan)
        for cs in range(0, n, args.pos_chunk):
            hi = min(n, cs + args.pos_chunk)
            encs: list[np.ndarray] = []
            owner: list[tuple[int, int]] = []
            recs: list[tuple[np.ndarray, np.ndarray]] = []
            for i in range(cs, hi):
                board = boards[i]
                moves = list(board.legal_moves)
                move_reg = move_regrets(positions[i], [m.uci() for m in moves])
                parent_value = np.full(len(moves), -np.inf, dtype=np.float64)
                for mi, m in enumerate(moves):
                    board.push(m)
                    if board.is_checkmate():
                        parent_value[mi] = 2.0
                    elif board.is_game_over():
                        parent_value[mi] = 0.0
                    else:
                        child_fen = np.asarray(
                            encode_cboard(CBoard.from_board(board), **enc_kwargs),
                            dtype=np.float32,
                        )
                        encs.append(child_planes(arm, child_fen, stored[i]))
                        owner.append((i - cs, mi))
                    board.pop()
                recs.append((move_reg, parent_value))
            for s in range(0, len(encs), args.value_batch):
                xb = np.stack(encs[s:s + args.value_batch])
                with torch.no_grad():
                    _, wdl = ev.evaluate_encoded(xb)
                wdl = np.asarray(wdl, dtype=np.float64)
                if not np.allclose(wdl.sum(axis=1), 1.0, atol=1e-3):
                    wdl = _softmax_rows(wdl)
                for j in range(xb.shape[0]):
                    lpi, mi = owner[s + j]
                    recs[lpi][1][mi] = wdl[j, 2] - wdl[j, 0]
            for lpi, (move_reg, parent_value) in enumerate(recs):
                if parent_value.size:
                    value_top1[cs + lpi] = float(
                        move_reg[int(np.argmax(parent_value))]
                    )
            if (cs // args.pos_chunk) % 5 == 0:
                print(f"[{args.label}] {tag} value {hi}/{n}", flush=True)
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()

        arms_out[arm] = {
            "input_encoding": ARM_ENCODING[arm],
            "policy_expected": expected.tolist(),
            "policy_top1": policy_top1.tolist(),
            "value_top1": value_top1.tolist(),
            "mean_policy_expected": float(np.nanmean(expected)),
            "mean_policy_top1": float(np.nanmean(policy_top1)),
            "mean_value_top1": float(np.nanmean(value_top1)),
        }
        summary = arms_out[arm]
        print(
            f"[{args.label}] {tag} POLICY exp={summary['mean_policy_expected']:.2f} "
            f"top1={summary['mean_policy_top1']:.2f}  "
            f"VALUE top1={summary['mean_value_top1']:.2f}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out), encoding="utf-8")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
