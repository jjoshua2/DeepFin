#!/usr/bin/env python3
"""Score a foreign net (or our own checkpoint) on the frozen deep-SF audit set.

Renamed from ``bt4_audit.py`` on 2026-08-13 when it grew a second input format
and a checkpoint mode. The cache schema and the report layout are unchanged and
``--cache-out`` still defaults to the BT4 cache that ``audit_compare_buckets.py``
reads; only ``--out`` moved off the old ``runs/bt4_audit.md`` name, so pass it
explicitly to keep writing there.

Caches are written through :mod:`chess_anti_engine.eval.audit_cache`, so they
carry a policy-map + audit-ruler provenance stamp and an EXISTING ``--cache-out``
is refused before any forward pass unless ``--force-cache-out`` is given.

⚑ This script gathers the 1858 policy DIRECTLY (``_leela_idxs``), it does NOT go
through :class:`chess_anti_engine.onnx.load.OnnxChessNet`. That is deliberate —
it keeps the BT4 numbers on the code path that produced the banked cache — but it
means a number from here vouches for the ENCODER and the 1858 table, not for the
wrapper. ``tests/test_onnx_chessnet_remap.py`` covers the wrapper, and
``test_wrapper_matches_the_direct_1858_gather`` pins the two together.

Three net kinds share ONE ruler — expected / top-1 deep-SF regret from
``chess_anti_engine.eval.audit`` — so their scores are directly comparable:

  * ``--onnx --input-format lc0_planes`` : LC0/BT4, 112 planes;
  * ``--onnx --input-format ceres_tpg``  : Ceres C1/C2/C3, 64x137 square
    records (:mod:`chess_anti_engine.encoding.ceres_tpg`);
  * ``--checkpoint``                     : one of our own torch checkpoints,
    raw policy forward, no search.

BT4 is a ~191M-param Leela net (far stronger than our ~46M model). Running it
on the same audit positions gives two things:

  1. A strong-reference regret curve vs the frozen deep-SF labels — in
     particular, how much regret BT4 *itself* bleeds in quiet positions. If a
     3500-Elo net also bleeds ~70cp where SF's "best" is near-arbitrary, that
     regret is a label-noise floor, not our model's weakness.
  2. A persistent per-position cache (bestmove / top-k policy / WDL) so we can
     compare our net to BT4 DIRECTLY (move agreement, policy divergence),
     not only through SF regret. Reusable for later analyses.

Input encoding is canonical-LC0 (`--history lc0_root`, validated: startpos →
sensible openings + correct WDL). Audit FENs are side-to-move canonical
(white to move), so no mirroring is needed. BT4's 1858 policy is gathered
directly in 1858 space at the legal moves via the canonical Leela index map
(``LC0_1858_UCI_TO_IDX`` / ``_leela_idxs``) — NOT through the project's compact
1858 ordering, which differs from Leela's on nearly every index. Ceres uses
Leela's table VERBATIM (``EncodedMove.NEURAL_NET_MOVE_STR`` agrees with
``LC0_1858_UCI_TO_IDX`` on all 1858 slots), so the same gather serves it; only
`--history`/`--history-fill` are inert for it, because a TPG record carries its
own eight history slots.

Usage:
  PYTHONPATH=. python3 scripts/foreign_net_audit.py \
    --onnx data/lc0/onnx/BT4-it332-vanilla-winner.onnx \
    --cache-out data/lc0/bt4_audit_cache.jsonl --out runs/bt4_audit.md

  PYTHONPATH=. python3 scripts/foreign_net_audit.py \
    --onnx data/ceres/C1-512-15/C1-512-15.onnx --input-format ceres_tpg \
    --gpu-mem-gb 0 --ort-threads 4 \
    --cache-out data/ceres/C1-512-15_audit_cache.jsonl --out runs/ceres_512_15.md

  PYTHONPATH=. python3 scripts/foreign_net_audit.py \
    --checkpoint <ckpt> --device cpu \
    --cache-out data/ceres/ours_audit_cache.jsonl --out runs/ours_audit.md
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import chess
import numpy as np

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.ceres_tpg import encode_ceres_tpg_batch
from chess_anti_engine.encoding.lc0 import fill_lc0_history_repeat
from chess_anti_engine.eval.audit import (
    CRITICALITY_BUCKET_NAMES,
    PHASE_NAMES,
    SOURCE_NAMES,
    criticality_bucket,
    criticality_gap,
    expected_and_top1_regret,
    legal_full_indices,
    move_regrets,
    parse_audit_record,
)
from chess_anti_engine.eval.audit_cache import (
    ensure_cache_writable,
    stamp_summary,
    write_audit_cache,
)
from chess_anti_engine.moves.encode import COMPACT_POLICY_SIZE, POLICY_SIZE, move_to_index
from chess_anti_engine.moves.leela_index import leela_index_for_move
from chess_anti_engine.onnx.load import (
    INPUT_FORMAT_CERES_TPG,
    INPUT_FORMAT_LC0_PLANES,
    ONNX_INPUT_FORMATS,
)


def _leela_idxs(board: chess.Board, ucis: list[str]) -> np.ndarray:
    """Map each legal UCI to its canonical LC0 1858 policy slot (-1 if absent).

    Delegates to :func:`~chess_anti_engine.moves.leela_index.leela_index_for_move`,
    which orients the move for a black-to-move board AND applies Leela's own
    spelling for the two families a UCI-string lookup gets wrong.

    ⚑ It used to do the lookup itself: mirror the UCI string, then index
    ``LC0_1858_UCI_TO_IDX`` with it. That is correct for ordinary moves and for
    promotions, and WRONG for castling — Leela spells O-O king-takes-rook
    (``e1h1``) while python-chess emits the king's two-square move (``e1g1``),
    and Leela's table also CONTAINS an ordinary ``e1g1`` slide entry, so the
    lookup returned a plausible logit for an unrelated move instead of failing.
    Measured on ``audit_set_v1``: 296 castling moves across 256 of the 4000
    positions (6.4%), every one of them mis-mapped, and in 41 of those the
    deep-SF best move IS the castle. This is the same defect
    ``moves/leela_index.py`` was written to kill, surviving in this script
    because the script never went through it. Any cache produced before
    2026-08-13 — including ``data/lc0/bt4_audit_cache.jsonl`` — carries it.
    """
    return np.array(
        [leela_index_for_move(board, chess.Move.from_uci(u)) for u in ucis],
        dtype=np.int64,
    )


def _session(
    onnx: str, gpu_mem_gb: float, *, input_format: str, ort_threads: int,
) -> tuple[Any, str, np.dtype]:
    import onnxruntime as ort

    providers: list = []
    if gpu_mem_gb > 0:
        providers.append(
            ("CUDAExecutionProvider",
             {"device_id": 0, "gpu_mem_limit": int(gpu_mem_gb * 1024 ** 3)}),
        )
    providers.append("CPUExecutionProvider")
    options = None
    if input_format == INPUT_FORMAT_CERES_TPG or ort_threads > 0:
        options = ort.SessionOptions()
        if input_format == INPUT_FORMAT_CERES_TPG:
            # ORT 1.23's ALL-level SimplifiedLayerNormFusion throws on the fp16
            # Ceres graphs and the session never opens; see onnx/load.py.
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        if ort_threads > 0:
            options.intra_op_num_threads = int(ort_threads)
    sess = ort.InferenceSession(onnx, options, providers=providers)
    name = sess.get_inputs()[0].name
    dtype = np.dtype(
        np.float16 if next(i.type for i in sess.get_inputs() if i.name == name)
        == "tensor(float16)" else np.float32,
    )
    return sess, name, dtype


def _to_wdl_probs(raw: np.ndarray) -> np.ndarray:
    """Cache WDL as a probability distribution so the downstream Brier/ECE are
    valid. Probabilities are non-negative AND sum to ~1; a logits net (negatives,
    or rows that merely sum near 1 in a tiny smoke run) must still be softmaxed.
    Use both signals — mirrors OnnxChessNet — so the audit numbers aren't
    computed against invalid distributions. Ceres's `value` is LOGITS
    (`NNEvaluatorONNX` is constructed with ``valueHeadLogistic: true``)."""
    w = raw.astype(np.float64)
    is_probs = bool((w >= -1e-4).all()) and np.allclose(w.sum(axis=1), 1.0, atol=0.1)
    if is_probs:
        return w
    w = np.exp(w - w.max(axis=1, keepdims=True))
    return w / w.sum(axis=1, keepdims=True)


def _score_onnx(
    boards: list[chess.Board], args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    """(policy in LEELA 1858 order, WDL probabilities) for an ONNX net."""
    sess, in_name, in_dtype = _session(
        args.onnx, args.gpu_mem_gb,
        input_format=args.input_format, ort_threads=int(args.ort_threads),
    )
    print(f"[audit] providers: {sess.get_providers()}; input {in_name} {in_dtype}")

    def _enc_lc0(b: chess.Board) -> np.ndarray:
        # `lc0_root` emits LC0-canonical metadata in planes 104..111 — verified
        # directly against LC0's layout: side-relative castling (us-Q, us-K,
        # them-Q, them-K, flipped for black-to-move), side-to-move flag (108),
        # raw rule50 count (109), zeros (110), ones (111). It writes NO
        # en-passant plane, which is correct: BT4's T-format has no EP plane and
        # conveys en passant through the move history. The single caveat is that
        # `_fill_history_repeat` fakes the 7 empty history frames, so for the
        # ~1% of audit FENs whose key move is an en-passant capture, BT4 can't
        # see the double-push that would make it available (a single-FEN
        # limitation, not a metadata bug). The `*_legacy_meta` variant DOES pack
        # an EP file into plane 110 and must NOT be used here.
        e = encode_position(b, add_features=False, input_history_encoding=args.history)
        return fill_lc0_history_repeat(e) if args.history_fill == "repeat" else e

    if args.input_format == INPUT_FORMAT_CERES_TPG:
        # `fill_in_history=True` is Ceres's own FEN-only behaviour, and it is the
        # same convention as `--history-fill repeat` on the LC0 side, so the two
        # nets are being asked the same question about a rootless position.
        feats = encode_ceres_tpg_batch(boards)
    else:
        feats = np.stack([_enc_lc0(b) for b in boards]).astype(np.float32)
    feats = feats.astype(in_dtype, copy=False)

    out_names = [o.name for o in sess.get_outputs()]
    pol = np.empty((len(boards), COMPACT_POLICY_SIZE), dtype=np.float32)
    wdl = np.empty((len(boards), 3), dtype=np.float32)
    bs = int(args.batch_size)
    pol_idx = wdl_idx = -1
    for s in range(0, len(boards), bs):
        out = [np.asarray(o) for o in sess.run(None, {in_name: feats[s:s + bs]})]
        if pol_idx < 0:
            # Prefer the caller's names; otherwise pick by width, not position:
            # some LC0/BT4 ONNX graphs emit WDL (3-wide) before policy
            # (1858-wide), so out[0] is not always the policy tensor. Among
            # several 3-wide outputs the FIRST is the primary value head —
            # Ceres emits `value` then `value2`, and `value` is what
            # NNEvaluatorONNX feeds to W1/L1.
            widths = [a.shape[-1] for a in out]
            pol_idx = (out_names.index(args.policy_output) if args.policy_output
                       else int(np.argmax(widths)))
            wdl_idx = (out_names.index(args.wdl_output) if args.wdl_output
                       else next((i for i, w in enumerate(widths) if w == 3), -1))
            if wdl_idx < 0:
                raise SystemExit(
                    "no 3-wide WDL output found; ONNX outputs have widths "
                    f"{widths}. Pass an LC0/Ceres net with a WDL value head."
                )
            print(f"[audit] policy={out_names[pol_idx]} wdl={out_names[wdl_idx]}")
        pol[s:s + bs] = out[pol_idx][:, :COMPACT_POLICY_SIZE]
        wdl[s:s + bs] = _to_wdl_probs(out[wdl_idx])
        if s % (bs * 8) == 0:
            print(f"[audit] {s + min(bs, len(boards) - s)}/{len(boards)}", flush=True)
    return pol, wdl


def _score_checkpoint(
    boards: list[chess.Board], checkpoint: str, *, device: str, batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(policy widened to OUR 4672 order, WDL probabilities) for our own net.

    Deliberately the same raw-policy forward `audit_targets.py` candidate (a)
    runs — encoding read off the model, no search — so "our net" on this report
    means the same object it means there."""
    import torch

    from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
    from chess_anti_engine.inference import LocalModelEvaluator
    from chess_anti_engine.moves import policy_batch_to_full_if_needed
    from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

    model = load_model_from_checkpoint(checkpoint, device=device)
    model.eval()
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    pol_enc = str(getattr(model, "policy_encoding", "lc0_1858"))
    use_rel = bool(getattr(model, "use_dynamic_relations", False))
    print(f"[audit] checkpoint encoding: {hist}/{extra}, policy {pol_enc}, "
          f"relations {use_rel}, device {device}")
    evaluator = LocalModelEvaluator(model, device=device)

    pol_chunks: list[np.ndarray] = []
    wdl_chunks: list[np.ndarray] = []
    for s in range(0, len(boards), batch_size):
        chunk = boards[s:s + batch_size]
        cbs = [CBoard.from_board(b) for b in chunk]
        xs = np.stack([
            encode_cboard(cb, input_history_encoding=hist, input_extra_features=extra)
            for cb in cbs
        ])
        rels = np.stack([cb.compute_relations() for cb in cbs]) if use_rel else None
        with torch.no_grad():
            if rels is None:
                pol_logits, net_wdl = evaluator.evaluate_encoded(xs)
            else:
                pol_logits, net_wdl = evaluator.evaluate_encoded(xs, relations=rels)
        pol_logits = np.asarray(pol_logits, dtype=np.float32)
        if pol_logits.shape[1] != POLICY_SIZE:
            pol_logits = policy_batch_to_full_if_needed(
                pol_logits, policy_encoding=pol_enc, fill_value=-1e9,
            )
        pol_chunks.append(pol_logits)
        wdl_chunks.append(_to_wdl_probs(np.asarray(net_wdl, dtype=np.float32)))
        print(f"[audit] {min(s + batch_size, len(boards))}/{len(boards)}", flush=True)
    return np.concatenate(pol_chunks), np.concatenate(wdl_chunks).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-set", type=Path, default=Path("data/audit_set_v1.jsonl"))
    ap.add_argument("--onnx", type=str,
                    default="data/lc0/onnx/BT4-it332-vanilla-winner.onnx")
    ap.add_argument("--history", default="lc0_root",
                    help="LC0 input history layout (lc0_root = canonical, validated)")
    ap.add_argument("--history-fill", choices=("repeat", "zero"), default="repeat",
                    help="empty-history fill: 'repeat' the current position into "
                         "all 8 slots (lc0's actual behavior, required for a sane "
                         "value head) or 'zero' (legacy, breaks BT4's WDL)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--gpu-mem-gb", type=float, default=4.0,
                    help="hard cap on the CUDA arena so a concurrent trainer "
                         "is not faulted; 0 forces CPU")
    ap.add_argument("--topk", type=int, default=5, help="policy moves cached per position")
    ap.add_argument("--cache-out", type=Path, default=Path("data/lc0/bt4_audit_cache.jsonl"))
    ap.add_argument("--force-cache-out", action="store_true",
                    help="allow --cache-out to overwrite an EXISTING cache. "
                         "Without it an existing file is refused before any "
                         "forward pass: every banked cache is a measurement "
                         "something else joins against.")
    ap.add_argument("--out", type=Path, default=Path("runs/foreign_net_audit.md"))
    ap.add_argument("--max-positions", type=int, default=0)
    ap.add_argument("--input-format", choices=ONNX_INPUT_FORMATS,
                    default=INPUT_FORMAT_LC0_PLANES,
                    help="ONNX input contract: 112 LC0 planes, or Ceres's 64x137 "
                         "TPG square records (--history/--history-fill are inert "
                         "for the latter, which carries its own history slots)")
    ap.add_argument("--policy-output", default=None,
                    help="ONNX policy output name; default picks the widest output")
    ap.add_argument("--wdl-output", default=None,
                    help="ONNX WDL output name. Default picks the FIRST 3-wide "
                         "output, which for Ceres is `value` (the primary head, "
                         "NNEvaluatorONNX's ValuesRaw). `value2` is the secondary "
                         "head Ceres blends in at 0.4 weight, not a WDL twin.")
    ap.add_argument("--ort-threads", type=int, default=0,
                    help="cap ORT's intra-op pool (0 = ORT default). Use a small "
                         "value when a training run owns the machine.")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="score one of OUR torch checkpoints instead of an ONNX "
                         "net: raw policy forward, no search, same ruler")
    ap.add_argument("--device", default="cpu", help="torch device for --checkpoint")
    args = ap.parse_args()

    # BEFORE the audit set is parsed, before the ONNX session exists, and long
    # before the forward pass: an unwanted clobber must cost zero compute.
    ensure_cache_writable(args.cache_out, force=args.force_cache_out)

    positions = [parse_audit_record(ln) for ln in
                 args.audit_set.read_text().splitlines() if ln.strip()]
    if args.max_positions > 0:
        positions = positions[: args.max_positions]
    print(f"[audit] {len(positions)} positions from {args.audit_set}")

    boards = [chess.Board(p.fen) for p in positions]
    if args.checkpoint:
        net_label = args.checkpoint
        pol_full, wdl = _score_checkpoint(
            boards, args.checkpoint, device=args.device, batch_size=int(args.batch_size),
        )
        pol = None
    else:
        net_label = f"{args.onnx} [{args.input_format}]"
        pol, wdl = _score_onnx(boards, args)
        pol_full = None

    # Aggregate by group; also keep per-criticality-bucket regret (shared edges).
    bucket_names = list(CRITICALITY_BUCKET_NAMES)

    sums: dict[str, list[float]] = {}
    cnts: dict[str, int] = {}

    def add(group: str, exp_r: float, top1_r: float) -> None:
        s = sums.setdefault(group, [0.0, 0.0])
        s[0] += exp_r
        s[1] += top1_r
        cnts[group] = cnts.get(group, 0) + 1

    cache_rows: list[dict] = []
    for i, (pos, board) in enumerate(zip(positions, boards, strict=True)):
        legal_ucis, _ = legal_full_indices(board)
        if not legal_ucis:
            continue
        regrets = move_regrets(pos, legal_ucis)
        if pol_full is not None:
            # Our own net already speaks OUR 4672 action ids.
            ours = np.array([move_to_index(chess.Move.from_uci(u), board)
                             for u in legal_ucis], dtype=np.int64)
            lg = pol_full[i][ours].astype(np.float64)
        else:
            assert pol is not None
            leela = _leela_idxs(board, legal_ucis)
            lg = np.where(leela >= 0, pol[i][leela.clip(0)], -1e9).astype(np.float64)
        lg = lg - lg.max()
        p = np.exp(lg)
        p = p / p.sum()
        exp_r, top1_r = expected_and_top1_regret(p, regrets)

        gap = criticality_gap(pos.move_cp)
        bname = bucket_names[criticality_bucket(gap)]
        for g in ("overall", PHASE_NAMES[pos.phase], SOURCE_NAMES[pos.source], bname):
            add(g, exp_r, top1_r)

        order = np.argsort(-p)[: int(args.topk)]
        cache_rows.append({
            "key": pos.key, "phase": pos.phase, "source": pos.source,
            # null (not inf -> non-standard JSON "Infinity") for <2-move positions
            "gap_cp": float(gap) if np.isfinite(gap) else None,
            "best_move": legal_ucis[int(np.argmax(p))],
            "wdl": [round(float(v), 4) for v in wdl[i]],
            "exp_regret": exp_r, "top1_regret": top1_r,
            "topk": [[legal_ucis[int(o)], round(float(p[int(o)]), 4)] for o in order],
        })

    stamp = write_audit_cache(
        args.cache_out, cache_rows, force=args.force_cache_out,
        # No "rows" here on purpose: `write_audit_cache` records the true
        # count itself, and a caller-supplied one would read as though the
        # caller were the authority on it — the exact thing the row binding
        # exists to deny.
        extra={"net": net_label, "audit_set": str(args.audit_set),
               "topk": int(args.topk)},
    )
    print(f"[audit] cache → {args.cache_out} ({len(cache_rows)} rows) "
          f"[{stamp_summary(stamp)}]")

    groups = (["overall", *PHASE_NAMES, *SOURCE_NAMES, *bucket_names])
    encoding_note = (
        "our own encoding, policy in OUR 4672 ids" if args.checkpoint
        else f"input format {args.input_format}"
        + (f"; lc0 history {args.history}/{args.history_fill}"
           if args.input_format == INPUT_FORMAT_LC0_PLANES
           else "; TPG carries its own 8 history slots (fill_in_history=True)")
        + "; policy mapped 1858→legal"
    )
    lines = [f"# Frozen-audit-set policy regret @ {net_label}", "",
             f"- audit set: {args.audit_set} ({len(cache_rows)} scored)",
             f"- {encoding_note}",
             f"- provenance: {stamp_summary(stamp)}",
             "", "| group | E[regret] cp | top-1 cp | n |", "|---|---|---|---|"]
    for g in groups:
        if g not in cnts:
            continue
        n = cnts[g]
        lines.append(f"| {g} | {sums[g][0] / n:.1f} | {sums[g][1] / n:.1f} | {n} |")
    report = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
