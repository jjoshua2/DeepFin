"""BT4 1-ply value of the NET's argmax, on the phase-1 rows (metric 3)."""
from __future__ import annotations
import json, os, time
import chess, numpy as np, onnxruntime as ort
from chess_anti_engine.encoding import encode_position
from chess_anti_engine.encoding.lc0 import fill_lc0_history_repeat
OUT="/home/josh/projects/chess/scratchpad/target_vs_bt4"
rows=np.load(f"{OUT}/tb4_rows.npz",allow_pickle=True)
net=np.load(f"{OUT}/tb4_net_new.npz",allow_pickle=True)
fens=[str(x) for x in rows["fens"]]; uci=[str(x) for x in net["net_uci"]]
n=len(fens)
feats=[];terminal={};ref=np.full(n,-1,dtype=np.int64);key={}
for i,f in enumerate(fens):
    b=chess.Board(f); mv=chess.Move.from_uci(uci[i]); assert mv in b.legal_moves,(i,f,uci[i])
    kk=(i,uci[i])
    c=b.copy(stack=True); c.push(mv); j=len(feats); key[kk]=j
    if c.is_checkmate(): terminal[j]=1.0; feats.append(np.zeros((112,8,8),np.float32))
    elif c.is_stalemate() or c.is_insufficient_material(): terminal[j]=0.0; feats.append(np.zeros((112,8,8),np.float32))
    else: feats.append(fill_lc0_history_repeat(encode_position(c,add_features=False,
            input_history_encoding="lc0_root")).astype(np.float32))
    ref[i]=j
F=np.stack(feats); print("children",F.shape[0],"terminal",len(terminal),flush=True)
so=ort.SessionOptions(); so.intra_op_num_threads=8
s=ort.InferenceSession("data/lc0/onnx/BT4-it332-vanilla-winner.onnx",so,providers=["CPUExecutionProvider"])
assert s.get_providers()==["CPUExecutionProvider"]
inn=s.get_inputs()[0].name; wdl=np.empty((F.shape[0],3),np.float32); t0=time.time()
for a in range(0,F.shape[0],16):
    out=[np.asarray(o) for o in s.run(None,{inn:F[a:a+16]})]
    w=[q.shape[-1] for q in out]; wdl[a:a+16]=out[next(i for i,q in enumerate(w) if q==3)]
    if (a//16)%40==0: print(f"[netmove] {a}/{F.shape[0]} {time.time()-t0:.0f}s",flush=True)
q=wdl[:,2].astype(np.float64)-wdl[:,0].astype(np.float64)
for j,v in terminal.items(): q[j]=v
np.savez_compressed(f"{OUT}/tb4_q_netmove.npz",Q_net=q[ref])
print(json.dumps({"n":n,"mean_Q_net":float(q[ref].mean())}))
