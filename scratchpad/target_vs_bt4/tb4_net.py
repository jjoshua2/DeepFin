"""tb4_net.py -- run the production net on the phase-1 rows. Phase 2, stage 1.

⚑ TWO INDEX SPACES. `LocalModelEvaluator.evaluate_encoded` returns policy logits
in OUR FULL 4672 action-id space (it widens the compact head internally --
measured: shape (B, 4672) off this checkpoint), while every stored shard array
(`policy_target`, `legal_mask`, and phase 1's `*_idx` columns) is COMPACT
lc0_1858. Indexing a 4672-wide vector with a compact id returns a plausible
logit for an unrelated move. So each row carries BOTH index vectors for the same
ordered list of legal moves, and the compact one is asserted equal to the row's
stored `legal_mask` -- the phase-1 alignment control, carried forward and
re-run rather than assumed.

Device defaults to CPU so this structurally cannot contend with the other
agent's GPU job.
"""
from __future__ import annotations

import argparse
import json
import os

import chess
import numpy as np
import torch
import zarr

from chess_anti_engine.encoding.cboard_encode import CBoard, encode_cboard
from chess_anti_engine.inference import LocalModelEvaluator
from chess_anti_engine.moves import (
    legal_move_mask_for_encoding,
    move_to_index,
    move_to_index_for_encoding,
)
from chess_anti_engine.uci.model_loader import load_model_from_checkpoint

OUT = "/home/josh/projects/chess/scratchpad/target_vs_bt4"
CKPT = ("/home/josh/projects/chess/data/salvage/bt4heads_iter100_20260815"
        "/seeds/slot_000/trainer.pt")
REPLAY = "/home/josh/projects/chess/runs/pbt2_small/replay"
ERA_E = "train_trial_1d175_00000_0_lr=0.0000_2026-08-14_13-53-53"

CAND = ("tgt_idx", "sf_idx", "foreign_idx", "rand_idx")


def stored_legal_masks(row_index: np.ndarray, shard_slice: slice) -> np.ndarray:
    sd = os.path.join(REPLAY, ERA_E, "replay_shards")
    names = sorted(n for n in os.listdir(sd) if n.endswith(".zarr"))[shard_slice]
    lm = np.concatenate([np.asarray(zarr.open(os.path.join(sd, n), mode="r")["legal_mask"][:])
                         for n in names], axis=0)
    return lm[row_index].astype(bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=os.path.join(OUT, "tb4_rows.npz"))
    ap.add_argument("--tag", default="new")
    ap.add_argument("--shard-start", type=int, default=-16)
    ap.add_argument("--shard-stop", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(int(args.threads))
    else:
        torch.cuda.set_per_process_memory_fraction(0.10, 0)

    d = np.load(args.rows, allow_pickle=True)
    fens = [str(x) for x in d["fens"]]
    n = len(fens)
    sl = slice(args.shard_start, args.shard_stop if args.shard_stop else None)
    stored_lm = stored_legal_masks(d["row_index"], sl)

    model = load_model_from_checkpoint(CKPT, device=args.device)
    model.eval()
    hist = str(getattr(model, "input_history_encoding", "legacy"))
    extra = str(getattr(model, "input_extra_features", "v1"))
    penc = str(getattr(model, "policy_encoding", "lc0_1858"))
    meta = {"ckpt": CKPT, "n": n, "device": args.device, "history": hist,
            "extra": extra, "policy_encoding": penc,
            "use_dynamic_relations": bool(getattr(model, "use_dynamic_relations", False))}
    if (hist, extra, penc) != ("lc0_root_legacy_meta", "v2_threats", "lc0_1858"):
        raise SystemExit(f"checkpoint encoding does not match the shards: {meta}")

    ev = LocalModelEvaluator(model, device=args.device)
    boards = [chess.Board(f) for f in fens]

    moves: list[list[chess.Move]] = []
    compact: list[np.ndarray] = []
    full: list[np.ndarray] = []
    a1_exact = 0
    for i, b in enumerate(boards):
        ml = list(b.legal_moves)
        c = np.array([move_to_index_for_encoding(m, b, policy_encoding="lc0_1858")
                      for m in ml], dtype=np.int64)
        f = np.array([move_to_index(m, b) for m in ml], dtype=np.int64)
        mask = np.asarray(legal_move_mask_for_encoding(
            b, policy_encoding="lc0_1858")).astype(bool).reshape(-1)
        recon = np.zeros(1858, dtype=bool)
        recon[c] = True
        # A1 carried forward: the compact ids I gather with must BE the row's mask
        a1_exact += int(np.array_equal(recon, stored_lm[i]) and np.array_equal(mask, stored_lm[i]))
        moves.append(ml)
        compact.append(c)
        full.append(f)
    meta["A1_n_exact"] = a1_exact
    meta["A1_frac_exact"] = a1_exact / n
    print(json.dumps(meta), flush=True)
    if a1_exact != n:
        raise SystemExit(f"alignment control FAILED: {a1_exact}/{n}")

    net_argmax_c = np.empty(n, dtype=np.int64)
    p_netmax = np.zeros(n)
    probs = {k: np.zeros(n) for k in CAND}
    rank_tgt = np.zeros(n, dtype=np.int64)
    entropy = np.zeros(n)
    bs = int(args.batch_size)
    for s in range(0, n, bs):
        chunk = boards[s:s + bs]
        xs = np.stack([encode_cboard(CBoard.from_board(b), input_history_encoding=hist,
                                     input_extra_features=extra) for b in chunk])
        with torch.no_grad():
            logits, _ = ev.evaluate_encoded(xs)
        logits = np.asarray(logits, dtype=np.float64)
        assert logits.shape[1] == 4672, logits.shape
        for j in range(len(chunk)):
            i = s + j
            lg = logits[j][full[i]]           # gather in FULL space
            lg = lg - lg.max()
            p = np.exp(lg)
            p /= p.sum()
            k = int(np.argmax(p))
            net_argmax_c[i] = int(compact[i][k])   # report in COMPACT space
            p_netmax[i] = p[k]
            entropy[i] = float(-(p * np.log(np.maximum(p, 1e-30))).sum())
            order = np.argsort(-p)
            rk = np.empty(p.size, dtype=np.int64)
            rk[order] = np.arange(1, p.size + 1)
            pos = {int(v): q for q, v in enumerate(compact[i])}
            for key in CAND:
                w = pos.get(int(d[key][i]))
                if w is not None:
                    probs[key][i] = p[w]
            wt = pos.get(int(d["tgt_idx"][i]))
            rank_tgt[i] = rk[wt] if wt is not None else -1
        if (s // bs) % 10 == 0:
            print(f"[net-{args.tag}] {min(s + bs, n)}/{n}", flush=True)

    net_uci = np.array([moves[i][int(np.flatnonzero(compact[i] == net_argmax_c[i])[0])].uci()
                        for i in range(n)], dtype=object)
    np.savez_compressed(
        os.path.join(OUT, f"tb4_net_{args.tag}.npz"),
        net_argmax=net_argmax_c, net_uci=net_uci, p_netmax=p_netmax,
        p_tgt=probs["tgt_idx"], p_sf=probs["sf_idx"],
        p_foreign=probs["foreign_idx"], p_rand=probs["rand_idx"],
        rank_tgt=rank_tgt, entropy=entropy, allow_pickle=True)
    with open(os.path.join(OUT, f"tb4_net_{args.tag}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
