#!/usr/bin/env python3
"""Value-head STRENGTH via 1-ply deep-SF regret (lc0-style value match, offline).

Brier scores value-head CALIBRATION and is fooled by chronic under-confidence —
a head can be badly hedged yet still RANK moves correctly, and ranking is what
matters for play. This scores the value head as a 1-ply player: for every frozen
audit position, evaluate the value head on EVERY legal child one ply deep, pick
the move that maximises the parent-mover's value (the move lc0's value-match
would play), and score THAT move's deep-SF regret in cp. The number is directly
comparable to the policy top-1 regret in ``scripts/audit_targets.py``.

Parent value of a move = child P(loss) - child P(win) from the CHILD's
side-to-move POV (the child is the opponent to move, so flip). Terminal children:
checkmate-by-the-move = best (parent wins); any other game-over = draw.

Forward-only, GPU-light — safe to run concurrent with training. Re-run on later
checkpoints to track whether value RANKING improves with training (which Brier
cannot show). See docs/eval_protocol.md and scripts/audit_targets.py.

⚑ TWO RULER PROPERTIES, BOTH OF WHICH MAKE NUMBERS NON-COMPARABLE ACROSS RUNS.
Both are echoed on every report line for exactly that reason.

1. `--input-encoding` (default `fen_only`, the historical behaviour) selects
   the INPUT the value head is scored on. `fen_only` rebuilds each child from
   `chess.Board(fen)` with an empty move stack, which under the production
   175-plane encoding leaves 93 planes structurally zero and pins the colour
   flag to the wrong value on ~51% of rows.
   ⚑ TWO DIFFERENT PLANE COUNTS APPEAR IN THIS FILE AND BOTH ARE RIGHT.
   **93** is the count of planes that are zero on EVERY row by construction of
   the FEN-only encoding (the 91 history planes 13..103, the current-frame
   repetition plane 12, and the colour flag 108) — measured on the 4000-row
   audit set, where the same count under `stored` is 0. The **117** quoted in
   the note at the bottom of this file, and in docs/rl_loop_audit.md M35, is a
   different quantity: the mean number of all-zero planes in a SINGLE row,
   which also counts planes that are empty because of the position rather than
   the encoding. On this audit set that per-row mean is 127.8 for `fen_only`
   against 62.4 for `stored`. `stored` feeds the real production
   history (audit-v2, see chess_anti_engine/eval/audit_history.py). These are
   DIFFERENT RULERS: a ruler change invalidates its records, so a `stored`
   number must never be put in a table, trend or threshold with a `fen_only`
   one.

2. **This ruler is BATCH-SIZE DEPENDENT.** Measured 2026-08-02 on the frozen
   set: boot512 reads **61.19 cp at --batch-size 128 vs 60.53 at 256 (0.66 cp)**
   and iter477 74.35 vs 74.64 (0.29 cp) — same checkpoint, same positions, same
   encoding. The mechanism is the same one already flagged for raw policy
   regret in scripts/audit_targets.py (~0.8 cp between 64 and 256): batch
   composition changes the GPU reduction order, and a 1-ply ARGMAX over child
   values turns a float-level wobble into a different chosen move. 0.66 cp is
   larger than several effects this ruler has been used to judge, so PIN the
   batch size across every arm of a comparison and STATE it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import chess
import numpy as np
import torch

from chess_anti_engine.encoding import model_encoding_kwargs
from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.eval.audit import PHASE_NAMES, load_audit_set, move_regrets
from chess_anti_engine.eval.audit_cache import (
    audit_set_provenance,
    stamp_summary,
    write_audit_cache,
)
from chess_anti_engine.eval.audit_history import (
    INPUT_ENCODINGS,
    INPUT_ENCODING_DEFAULT,
    MatchedAuditRows,
    child_planes_for_encoding,
    default_matched_rows_path,
    normalize_input_encoding,
)
from chess_anti_engine.eval.value_optimism import (
    SF_EVAL_BUCKET_NAMES,
    sf_eval_bucket,
)
from chess_anti_engine.inference import LocalModelEvaluator
from scripts.net_source import (
    NetSource,
    add_net_source_args,
    apply_gpu_mem_cap,
    net_source_from_args,
    reject_stored_encoding_for_onnx,
)


def _softmax_rows(a: np.ndarray) -> np.ndarray:
    a = a - a.max(axis=1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=1, keepdims=True)


def _piece_count(fen: str) -> int:
    """Total pieces on the board (both kings included) from a FEN.

    Used to exclude Syzygy-range positions from the ruler: with <=7 pieces the
    engine plays via tablebase (selfplay TB-optimal moves, UCI root probe), so
    the net's value there never decides a real move — scoring it measures value
    quality we do not use. See chess_anti_engine/tablebase.py (<=7, no castling).
    """
    return sum(1 for c in fen.split(" ", 1)[0] if c.isalpha())


def value_1ply_regret(
    *, net: NetSource, positions, device: str, batch_size: int, pos_chunk: int,
    input_encoding: str = INPUT_ENCODING_DEFAULT,
    matched_rows: MatchedAuditRows | None = None,
    gpu_mem_fraction: float | None = None,
) -> tuple[float, dict[int, float], np.ndarray]:
    """Return (overall mean regret cp, {phase: mean cp}, per-position regrets).

    The third element is the per-position top-1 regret array (NaN for
    terminal roots), aligned with ``positions`` — dump it for paired
    checkpoint comparisons (scripts/paired_compare.py).

    ``net`` is one of ours (``--checkpoint``) or a foreign ONNX net
    (``--onnx``); it carries exactly one and raises otherwise, so this ruler
    cannot silently score a different net from the one requested. The child
    planes are built from whatever encoding the loaded model DECLARES
    (``model_encoding_kwargs``), which for an LC0/Ceres net is the lc0_root
    history layout its adapter needs — not our production layout.

    ``input_encoding`` selects what the value head is fed. ``fen_only`` is the
    historical path and is byte-for-byte unchanged: the child encoding handed
    to the evaluator is exactly ``encode_cboard`` of the pushed board.
    ``stored`` splices the parent's real production history onto each child
    (audit-v2) and requires ``matched_rows`` covering every scored position.

    ``gpu_mem_fraction`` is plumbed all the way to ``net.load`` because on the
    ``--onnx`` path the cap is a SESSION-CONSTRUCTION argument (ORT's
    ``gpu_mem_limit``); there is no later point at which it can be applied.
    """
    encoding = normalize_input_encoding(input_encoding)
    if encoding == "stored" and matched_rows is None:
        raise ValueError("input_encoding='stored' requires matched_rows")
    model = net.load(
        device=device, gpu_mem_fraction=gpu_mem_fraction, tag="value-regret",
    )
    enc_kwargs = model_encoding_kwargs(model)
    if matched_rows is not None and encoding == "stored":
        matched_rows.require_model_compatible(enc_kwargs)
    # Dynamic-relation checkpoints apply their attention bias only when the
    # relation tensor is passed; without it we'd silently score a relation-less
    # model. Carry relations exactly like scripts/audit_targets.py does.
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    if use_rel and encoding == "stored":
        # Relations are recomputed from the bare pushed board, so they would
        # describe a position without the history the planes now carry.
        raise SystemExit(
            "--input-encoding stored does not support dynamic-relation "
            "checkpoints: the relation tensor is rebuilt from the bare child "
            "board and would contradict the spliced history planes"
        )
    ev = LocalModelEvaluator(model, device=device)

    top1 = np.full(len(positions), np.nan, dtype=np.float64)
    phases = np.array([p.phase for p in positions], dtype=np.int64)

    for cs in range(0, len(positions), pos_chunk):
        chunk = positions[cs:cs + pos_chunk]
        encs: list[np.ndarray] = []
        rels: list[np.ndarray] = []
        owner: list[tuple[int, int]] = []          # (local_pos_idx, move_idx)
        recs: list[tuple[np.ndarray, np.ndarray]] = []  # (regrets, parent_val)
        for lpi, pos in enumerate(chunk):
            board = chess.Board(pos.fen)
            stored_parent = (
                matched_rows.stored_row(pos.key)
                if matched_rows is not None and encoding == "stored" else None
            )
            legal = list(board.legal_moves)
            regrets = move_regrets(pos, [m.uci() for m in legal])
            pv = np.full(len(legal), -np.inf, dtype=np.float64)
            for mi, m in enumerate(legal):
                board.push(m)
                if board.is_checkmate():
                    pv[mi] = 2.0                    # parent mates -> best
                elif board.is_game_over():
                    pv[mi] = 0.0                    # stalemate / draw
                else:
                    cb = CBoard.from_board(board)
                    # `fen_only` is the identity branch: the array appended
                    # here is the same object encode_cboard returned.
                    encs.append(child_planes_for_encoding(
                        encoding=encoding,
                        child_fen_planes=encode_cboard(cb, **enc_kwargs),
                        stored_parent=stored_parent,
                    ))
                    if use_rel:
                        rels.append(cb.compute_relations())
                    owner.append((lpi, mi))
                board.pop()
            recs.append((regrets, pv))
        for s in range(0, len(encs), batch_size):
            xs = np.stack(encs[s:s + batch_size])
            rel_batch = np.stack(rels[s:s + batch_size]) if use_rel else None
            with torch.no_grad():
                _, wdl = ev.evaluate_encoded(xs, relations=rel_batch)
            wdl = np.asarray(wdl, dtype=np.float64)
            if not np.allclose(wdl.sum(axis=1), 1.0, atol=1e-3):
                wdl = _softmax_rows(wdl)
            for j in range(len(xs)):
                lpi, mi = owner[s + j]
                recs[lpi][1][mi] = wdl[j, 2] - wdl[j, 0]   # P(loss)-P(win), child POV
        for lpi, (regrets, pv) in enumerate(recs):
            if pv.size == 0:
                continue  # terminal root (no legal moves) — leave top1 as NaN
            top1[cs + lpi] = float(regrets[int(np.argmax(pv))])

    overall = float(np.nanmean(top1))
    per_phase = {
        ph: float(np.nanmean(top1[phases == ph]))
        for ph in range(3) if (phases == ph).any()
    }
    return overall, per_phase, top1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_net_source_args(
        ap,
        checkpoint_help="one of ours: trainer.pt or checkpoint dir. Mutually "
        "exclusive with --onnx; exactly one is required.",
    )
    ap.add_argument("--audit-set", default="data/audit_set_v1.jsonl")
    ap.add_argument("--max-positions", type=int, default=0,
                    help="0 (default) = score ALL rows. The audit set is written in "
                         "sorted phase/source strata, so a prefix slice is biased "
                         "(over-weights early strata, can omit openings) and not "
                         "comparable to audit_targets.py — only limit for a quick check.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=256,
                    help="child forward batch. ⚑ THIS RULER IS BATCH-SIZE DEPENDENT: "
                         "0.66 cp between 128 and 256 on the same checkpoint "
                         "(boot512 61.19 -> 60.53, measured 2026-08-02). The 1-ply "
                         "argmax turns a reduction-order wobble into a different "
                         "chosen move. Pin it across every arm of a comparison; it "
                         "is echoed on every report line so a number cannot be "
                         "lifted out of context.")
    ap.add_argument("--input-encoding", choices=INPUT_ENCODINGS,
                    default=INPUT_ENCODING_DEFAULT,
                    help="what the value head is fed. 'fen_only' (DEFAULT, and "
                         "bit-identical to every historical run) rebuilds each child "
                         "from chess.Board(fen), leaving 93 of 175 planes zero and "
                         "the colour flag wrong on ~51%% of rows. 'stored' is "
                         "audit-v2: the real production history, spliced from the "
                         "matched stored parent row. ⚑ DIFFERENT RULERS — never mix "
                         "the two in one table, trend or threshold.")
    ap.add_argument("--matched-rows", type=Path, default=None,
                    help="matched-rows index for --input-encoding stored "
                         "(default: <audit-set>.matched_rows.npz). Built by "
                         "scripts/match_audit_rows.py; not checked in.")
    ap.add_argument("--pos-chunk", type=int, default=128,
                    help="positions buffered per forward pass group (bounds RAM)")
    ap.add_argument("--gpu-mem-fraction", type=float, default=None,
                    help="cap this process's CUDA memory fraction (coexist with "
                         "training). Applied to BOTH allocators a run can use: the "
                         "torch caching allocator (set_per_process_memory_fraction) "
                         "and, on --onnx, onnxruntime's CUDA arena (gpu_mem_limit, "
                         "computed against the card's total memory). The two are "
                         "separate: torch's cap does not bound an ORT session.")
    ap.add_argument("--dump-per-position", default=None, metavar="PATH",
                    help="write per-position JSONL (fen, phase, top1 regret cp) for "
                         "paired checkpoint comparison via scripts/paired_compare.py")
    ap.add_argument("--sf-strata", action="store_true",
                    help="additionally break the SAME regret down by deep-SF evaluation "
                         "of the position (lost/losing/balanced/winning/won), using the "
                         "shared buckets in eval/value_optimism.py. Answers whether value "
                         "RANKING degrades where the side to move is already lost. The "
                         "headline number is unchanged — this only adds rows, so a "
                         "historical value_regret figure stays comparable.")
    ap.add_argument("--min-pieces", type=int, default=8,
                    help="Minimum total pieces to score. DEFAULT 8 EXCLUDES "
                         "Syzygy-range (<=7-man) positions: the engine plays those via "
                         "tablebase (see tablebase.py), so the net's value there never "
                         "decides a real move and scoring it measures value quality we "
                         "never use — the default ruler now tracks PLAY-RELEVANT value. "
                         "Pass --min-pieces 0 to score ALL positions (pre-2026-07-20 "
                         "behavior), required to reproduce historical full-set numbers "
                         "(e.g. the 70-76cp band / BT4=43cp ref were full-set).")
    args = ap.parse_args()
    # Resolve the net BEFORE any audit-set or GPU work: a typo'd path or a
    # graph with no 1858 policy head must fail here, not after the scoring
    # loop has produced a plausible-looking number for the wrong weights.
    net = net_source_from_args(args)
    # ...and the flag combination that cannot mean anything, before the
    # matched-rows index is even opened.
    reject_stored_encoding_for_onnx(net, args.input_encoding)

    # Caps the TORCH allocator and says so. The --onnx session is capped at
    # load time, through ORT's own gpu_mem_limit -- torch's fraction cannot
    # reach it, which is the bug this split exists to stop reprinting.
    apply_gpu_mem_cap(
        net=net,
        device=str(args.device),
        gpu_mem_fraction=args.gpu_mem_fraction,
        tag="value-regret",
    )

    encoding = normalize_input_encoding(args.input_encoding)
    # Every metric line below carries this tag. The two knobs in it each move
    # the number by more than several effects this ruler has been asked to
    # judge, so a figure that does not name them cannot be compared to anything.
    tag = f"[enc={encoding} b={args.batch_size}]"

    positions = load_audit_set(args.audit_set)
    # Slice the canonical max_positions subset (e.g. v1-2k = first 2000) FIRST,
    # then drop TB-range — so --min-pieces filters WITHIN the standard subset
    # (2000 -> 1723) rather than pulling later strata forward from the full set.
    if args.max_positions > 0:
        positions = positions[: args.max_positions]
    if args.min_pieces > 0:
        n_before = len(positions)
        positions = [p for p in positions if _piece_count(p.fen) >= args.min_pieces]
        print(f"[value-regret] min-pieces>={args.min_pieces}: dropped "
              f"{n_before - len(positions)} TB-range positions ({len(positions)} kept)")

    matched: MatchedAuditRows | None = None
    if encoding == "stored":
        matched = MatchedAuditRows(
            args.matched_rows or default_matched_rows_path(args.audit_set)
        )
        n_before = len(positions)
        positions = [p for p in positions if p.key in matched]
        print(f"[value-regret] input-encoding=stored: {matched.path} covers "
              f"{matched.n_matched}/{matched.n_audit_rows} audit rows; dropped "
              f"{n_before - len(positions)} unmatched ({len(positions)} kept). "
              f"⚑ RULER CHANGE — these numbers are NOT comparable to fen_only ones.")
        if not positions:
            raise SystemExit(
                "no scored position has a stored row; the matched-rows index does "
                "not cover this audit set"
            )
    print(f"[value-regret] {tag} {len(positions)} positions from {args.audit_set}")
    overall, per_phase, per_position = value_1ply_regret(
        net=net, positions=positions, device=args.device,
        batch_size=args.batch_size, pos_chunk=args.pos_chunk,
        input_encoding=encoding, matched_rows=matched,
        gpu_mem_fraction=args.gpu_mem_fraction,
    )
    if args.dump_per_position:
        # ⚑ STAMPED, via `write_audit_cache` rather than a bare `json.dumps`
        # loop. `paired_compare.require_same_stamp` short-circuits on
        # `if not a.stamp or not b.stamp`, so an UNSTAMPED dump takes the warn
        # path and the gate never fires — and `scripts/monitor_fen.sh` is the
        # only automated caller of `paired_compare`, feeding it exactly this
        # dump. The gate was therefore inert on the one production path it has.
        # A guard that cannot fire where it is actually invoked is this
        # codebase's signature defect, so the writer stamps.
        #
        # force=True for the same reason `audit_targets` uses it: this option
        # has no default path, so the operator always names the target and
        # there is no silent-default clobber to guard against.
        stamp = write_audit_cache(
            Path(args.dump_per_position),
            (
                {
                    "fen": pos.fen, "phase": int(pos.phase),
                    "value": None if np.isnan(r) else float(r),
                    # A dump is a report: it must carry the ruler it was made
                    # with, or a downstream paired compare can silently join
                    # two different rulers.
                    "input_encoding": encoding, "batch_size": int(args.batch_size),
                    # Which NET produced the row, not just which ruler read it:
                    # a paired compare that joins two dumps must be able to see
                    # that one of them is a foreign ONNX net.
                    "net": net.label,
                }
                for pos, r in zip(positions, per_position, strict=True)
            ),
            force=True,
            extra={"producer": "value_regret.py --dump-per-position",
                   "input_encoding": encoding,
                   **audit_set_provenance(Path(args.audit_set))},
        )
        print(f"[value-regret] {tag} per-position dump -> "
              f"{args.dump_per_position} [{stamp_summary(stamp)}]")

    ruler = "TB-excluded" if args.min_pieces > 0 else "FULL-SET (incl. TB)"
    print(f"\n=== value-head 1-ply deep-SF regret @ {net.label} "
          f"| input-encoding={encoding} | batch-size={args.batch_size} ===")
    print(f"  {tag} ruler: {ruler}"
          + (f" (>={args.min_pieces}-man)" if args.min_pieces > 0 else "")
          + f"  |  OVERALL {overall:.1f} cp (n={len(positions)})")
    for ph in sorted(per_phase):
        print(f"  {tag} {PHASE_NAMES[ph]:11s} {per_phase[ph]:.1f} cp")
    # Tail view: the audit mean is tail-dominated (median ~10cp vs mean ~75) and
    # the >300cp blowups are the Cheese single-collapse failure mode, so the
    # tail is reported alongside the mean rather than hidden inside it.
    finite = per_position[~np.isnan(per_position)]
    if finite.size:
        print(f"  {tag} TAIL {'all':11s} med={float(np.median(finite)):6.1f} "
              f"P90={float(np.percentile(finite, 90)):7.1f} "
              f">100cp={100 * float((finite > 100).mean()):5.1f}% "
              f">300cp={100 * float((finite > 300).mean()):5.1f}%")
        pos_phases = np.array([int(p.phase) for p in positions])
        for ph in sorted(per_phase):
            v = per_position[pos_phases == ph]
            v = v[~np.isnan(v)]
            if v.size:
                print(f"  {tag} TAIL {PHASE_NAMES[ph]:11s} med={float(np.median(v)):6.1f} "
                      f"P90={float(np.percentile(v, 90)):7.1f} "
                      f">100cp={100 * float((v > 100).mean()):5.1f}% "
                      f">300cp={100 * float((v > 300).mean()):5.1f}%")

    if args.sf_strata:
        # The stratum comes from the deep-SF label, never from the net: the
        # net's own opinion is the quantity under test and must not select the
        # sample it is scored on.
        buckets = np.array([sf_eval_bucket(float(p.best_cp)) for p in positions])
        print("\n  --- by deep-SF evaluation of the position (RANKING axis) ---")
        for b, name in enumerate(SF_EVAL_BUCKET_NAMES):
            v = per_position[buckets == b]
            v = v[~np.isnan(v)]
            if not v.size:
                continue
            print(f"  {tag} SF {name:20s} n={v.size:5d} mean={float(v.mean()):6.1f} cp  "
                  f"med={float(np.median(v)):6.1f} P90={float(np.percentile(v, 90)):7.1f} "
                  f">300cp={100 * float((v > 300).mean()):5.1f}%")
        print("  This is RANKING (which move the value head would play), not LEVEL. For the "
              "level — is the head optimistic about losing positions — use "
              "scripts/value_optimism.py, which scores production input planes.")
        if encoding == "fen_only":
            print("  ⚑ At --input-encoding fen_only these inputs have 93 planes zero on "
                  "EVERY row by construction (127.8 all-zero per row on this set vs 62.4 "
                  "under stored; docs/rl_loop_audit.md M35 quotes 117 for the per-row "
                  "figure on its own position set). That compromises ABSOLUTE cp levels. "
                  "--input-encoding stored removes that specific defect; it does not make "
                  "the two sets of numbers comparable to each other.")


if __name__ == "__main__":
    main()
