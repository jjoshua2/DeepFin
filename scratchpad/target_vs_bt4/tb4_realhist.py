"""tb4_realhist.py -- control A4: feed BT4 the row's OWN stored 112 planes.

The main arm hands BT4 a board rebuilt from the row and lc0's own empty-history
fill, so BT4 never sees the real previous 7 plies. This arm instead ships the
STORED planes, which carry the true 8-slot history, converted from
`lc0_root_legacy_meta` to canonical `lc0_root`:

    plane 109  rule50   stored as min(hm,100)/100  ->  raw count (x100)
    plane 110  EP file  (legacy_meta extra)        ->  zeros (lc0's movecount)

Only the PARENT policy reading is available this way (there are no stored planes
for a child position), so this controls the policy ruler, not the value ruler.
"""
from __future__ import annotations

import json
import os

import chess
import numpy as np
import zarr

from chess_anti_engine.moves.leela_index import leela_index_for_move

OUT = "/home/josh/projects/chess/scratchpad/target_vs_bt4"
REPLAY = "/home/josh/projects/chess/runs/pbt2_small/replay"
ERA_E = "train_trial_1d175_00000_0_lr=0.0000_2026-08-14_13-53-53"
N_SHARDS = 16


def main() -> None:
    import onnxruntime as ort

    d = np.load(os.path.join(OUT, "tb4_rows.npz"), allow_pickle=True)
    fens = list(d["fens"])
    rid = d["row_index"]

    sd = os.path.join(REPLAY, ERA_E, "replay_shards")
    names = sorted(n for n in os.listdir(sd) if n.endswith(".zarr"))[-N_SHARDS:]
    xs = np.concatenate([np.asarray(zarr.open(os.path.join(sd, n), mode="r")["x"][:, :112])
                         for n in names], axis=0)
    feats = xs[rid].astype(np.float32)
    feats[:, 109] *= 100.0
    feats[:, 110] = 0.0

    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    s = ort.InferenceSession("data/lc0/onnx/BT4-it332-vanilla-winner.onnx", so,
                             providers=["CPUExecutionProvider"])
    assert s.get_providers() == ["CPUExecutionProvider"]
    inn = s.get_inputs()[0].name

    pol = np.empty((feats.shape[0], 1858), dtype=np.float32)
    wdl = np.empty((feats.shape[0], 3), dtype=np.float32)
    for a in range(0, feats.shape[0], 16):
        out = [np.asarray(o) for o in s.run(None, {inn: feats[a:a + 16]})]
        w = [q.shape[-1] for q in out]
        pol[a:a + 16] = out[int(np.argmax(w))][:, :1858]
        wdl[a:a + 16] = out[next(i for i, q in enumerate(w) if q == 3)]
        if (a // 16) % 25 == 0:
            print(f"[realhist] {a}/{feats.shape[0]}", flush=True)

    top1: list[str] = []
    for i, f in enumerate(fens):
        b = chess.Board(f)
        legal = list(b.legal_moves)
        li = np.array([leela_index_for_move(b, m) for m in legal], dtype=np.int64)
        lg = np.where(li >= 0, pol[i][li.clip(0)], -1e9).astype(np.float64)
        top1.append(legal[int(np.argmax(lg))].uci())
    np.savez_compressed(os.path.join(OUT, "tb4_realhist.npz"),
                        top1_uci=np.array(top1, dtype=object), wdl=wdl, allow_pickle=True)
    print(json.dumps({"n": len(top1), "wdl_mean": [float(v) for v in wdl.mean(axis=0)]}))


if __name__ == "__main__":
    main()
