"""tb4_bt4.py -- score the selected rows with BT4 (SF-agnostic ruler). CPU only.

For each row we ask BT4 two independent questions about four candidate moves
(target argmax / SF best / foreign-target argmax [shuffle control] / random legal
[positive control]):

  VALUE  Q(m) = L_child - W_child from BT4's WDL at parent.push(m). The child is
         opponent-to-move, so negating gives the mover's score. Terminal children
         are scored exactly (+1 checkmate, 0 stalemate/insufficient) and never
         handed to the net.
  POLICY BT4's softmax prior over the parent's legal moves, gathered from the
         1858 head via the board-aware Leela remap.

Providers are pinned to CPU only, so this structurally cannot touch the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import chess
import numpy as np

from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import fill_lc0_history_repeat
from chess_anti_engine.moves import index_to_move_for_encoding
from chess_anti_engine.moves.leela_index import leela_index_for_move

OUT = "/home/josh/projects/chess/scratchpad/target_vs_bt4"
CAND = ("tgt", "sf", "foreign", "rand")


def enc(board: chess.Board) -> np.ndarray:
    return fill_lc0_history_repeat(
        encode_position(board, add_features=False, input_history_encoding="lc0_root"),
    ).astype(np.float32)


def session(net: str, threads: int):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = int(threads)
    s = ort.InferenceSession(net, so, providers=["CPUExecutionProvider"])
    got = s.get_providers()
    if got != ["CPUExecutionProvider"]:
        raise SystemExit(f"refusing to run: providers {got} include a GPU provider")
    return s, s.get_inputs()[0].name


def run_batches(sess, in_name, feats, bs, tag):
    pol = np.empty((feats.shape[0], 1858), dtype=np.float32)
    wdl = np.empty((feats.shape[0], 3), dtype=np.float32)
    t0 = time.time()
    for s in range(0, feats.shape[0], bs):
        out = [np.asarray(o) for o in sess.run(None, {in_name: feats[s:s + bs]})]
        widths = [a.shape[-1] for a in out]
        pi = int(np.argmax(widths))
        wi = next(i for i, w in enumerate(widths) if w == 3)
        pol[s:s + bs] = out[pi][:, :1858]
        w = out[wi].astype(np.float64)
        if not (bool((w >= -1e-4).all()) and np.allclose(w.sum(axis=1), 1.0, atol=0.1)):
            w = np.exp(w - w.max(axis=1, keepdims=True))
            w = w / w.sum(axis=1, keepdims=True)
        wdl[s:s + bs] = w
        if (s // bs) % 20 == 0:
            done = min(s + bs, feats.shape[0])
            el = time.time() - t0
            print(f"[{tag}] {done}/{feats.shape[0]} {el:.0f}s "
                  f"eta {el / max(done, 1) * (feats.shape[0] - done):.0f}s", flush=True)
    return pol, wdl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="data/lc0/onnx/BT4-it332-vanilla-winner.onnx")
    ap.add_argument("--tag", default="winner")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--policy", type=int, default=1, help="also run the parent policy pass")
    ap.add_argument("--real-history", type=int, default=0,
                    help="extra parent pass fed the row's OWN stored 112 planes "
                         "(plane 110 zeroed) instead of the repeat fill (control A4)")
    args = ap.parse_args()

    d = np.load(os.path.join(OUT, "tb4_rows.npz"), allow_pickle=True)
    fens = list(d["fens"])
    n = len(fens) if args.limit <= 0 else min(args.limit, len(fens))
    idxs = {c: d[f"{c}_idx"][:n] for c in CAND}

    boards = [chess.Board(f) for f in fens[:n]]
    moves: dict[str, list[chess.Move]] = {c: [] for c in CAND}
    for i, b in enumerate(boards):
        for c in CAND:
            m = index_to_move_for_encoding(int(idxs[c][i]), b)
            assert m in b.legal_moves, (i, c, b.fen(), m)
            moves[c].append(m)

    # ---- child positions, deduplicated by (row, move) -----------------------
    child_key: dict[tuple[int, str], int] = {}
    child_feats: list[np.ndarray] = []
    terminal: dict[int, float] = {}
    ref = np.full((n, len(CAND)), -1, dtype=np.int64)
    for i, b in enumerate(boards):
        for ci, c in enumerate(CAND):
            k = (i, moves[c][i].uci())
            if k not in child_key:
                child = b.copy(stack=True)
                child.push(moves[c][i])
                j = len(child_feats)
                child_key[k] = j
                if child.is_checkmate():
                    terminal[j] = 1.0
                    child_feats.append(np.zeros((112, 8, 8), dtype=np.float32))
                elif child.is_stalemate() or child.is_insufficient_material():
                    terminal[j] = 0.0
                    child_feats.append(np.zeros((112, 8, 8), dtype=np.float32))
                else:
                    child_feats.append(enc(child))
            ref[i, ci] = child_key[k]
    child_arr = np.stack(child_feats)
    print(f"[setup] rows={n} unique children={child_arr.shape[0]} "
          f"terminal={len(terminal)}", flush=True)

    sess, in_name = session(args.net, args.threads)
    print(f"[setup] providers={sess.get_providers()} net={args.net}", flush=True)

    _, cwdl = run_batches(sess, in_name, child_arr, args.batch_size, f"child-{args.tag}")
    q_child = cwdl[:, 2].astype(np.float64) - cwdl[:, 0].astype(np.float64)  # L - W
    for j, v in terminal.items():
        q_child[j] = v  # mover's score, already from the mover's POV
    Q = q_child[ref]  # (n, 4) mover-POV score of each candidate move

    res: dict[str, object] = {
        "net": args.net, "tag": args.tag, "n_rows": int(n),
        "n_unique_children": int(child_arr.shape[0]),
        "n_terminal_children": int(len(terminal)),
    }
    np.savez_compressed(os.path.join(OUT, f"tb4_q_{args.tag}.npz"),
                        Q=Q, cand=np.array(CAND), terminal_ref=ref)

    # ---- parent policy pass -------------------------------------------------
    if args.policy:
        par = np.stack([enc(b) for b in boards])
        ppol, pwdl = run_batches(sess, in_name, par, args.batch_size, f"parent-{args.tag}")
        top1_uci: list[str] = []
        prob = np.zeros((n, len(CAND)))
        rank = np.zeros((n, len(CAND)), dtype=np.int64)
        for i, b in enumerate(boards):
            legal = list(b.legal_moves)
            li = np.array([leela_index_for_move(b, m) for m in legal], dtype=np.int64)
            lg = np.where(li >= 0, ppol[i][li.clip(0)], -1e9).astype(np.float64)
            p = np.exp(lg - lg.max())
            p /= p.sum()
            order = np.argsort(-p)
            rk = np.empty(len(legal), dtype=np.int64)
            rk[order] = np.arange(1, len(legal) + 1)
            top1_uci.append(legal[int(order[0])].uci())
            for ci, c in enumerate(CAND):
                w = legal.index(moves[c][i])
                prob[i, ci] = p[w]
                rank[i, ci] = rk[w]
        np.savez_compressed(os.path.join(OUT, f"tb4_policy_{args.tag}.npz"),
                            prob=prob, rank=rank,
                            top1_uci=np.array(top1_uci, dtype=object),
                            cand_uci=np.array([[moves[c][i].uci() for c in CAND]
                                               for i in range(n)], dtype=object),
                            wdl=pwdl, allow_pickle=True)
        res["parent_wdl_mean"] = [float(v) for v in pwdl.mean(axis=0)]

    with open(os.path.join(OUT, f"tb4_bt4_{args.tag}_meta.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
