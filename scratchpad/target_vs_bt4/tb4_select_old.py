"""OLD-SATURATED exposure arm: same selection, shards[300:316] (~8.8h before ckpt)."""
from __future__ import annotations
import importlib.util, json, os, sys
import numpy as np, zarr, chess
spec=importlib.util.spec_from_file_location("tb4_select",
    "/home/josh/projects/chess/scratchpad/target_vs_bt4/tb4_select.py")
S=importlib.util.module_from_spec(spec); sys.modules["tb4_select"]=S; spec.loader.exec_module(S)
from chess_anti_engine.encoding.lc0 import LC0_HISTORY_ROOT_LEGACY_META
from chess_anti_engine.moves import legal_move_mask_for_encoding, move_to_index_for_encoding
OUT=S.OUT; CAP=S.CAP
sd=os.path.join(S.REPLAY,S.ERA_E,"replay_shards")
names=sorted(n for n in os.listdir(sd) if n.endswith(".zarr"))[300:316]
acc={}
for nm in names:
    z=zarr.open(os.path.join(sd,nm),mode="r")
    for k in ("x","policy_target","legal_mask","has_policy","sf_p0_regret","has_sf_p0_regret"):
        acc.setdefault(k,[]).append(np.asarray(z[k][:]))
D={k:np.concatenate(v,axis=0) for k,v in acc.items()}
hs=D["has_sf_p0_regret"].astype(bool)&D["has_policy"].astype(bool)
idx=np.flatnonzero(hs); rng=np.random.default_rng(20260816)
n=min(2000,idx.size); sel=np.sort(rng.choice(idx,size=n,replace=False))
pol=D["policy_target"][sel].astype(np.float64); lm=D["legal_mask"][sel].astype(bool)
R=D["sf_p0_regret"][sel].astype(np.float64); X=D["x"][sel]
P=pol*lm; P/=np.maximum(P.sum(axis=1,keepdims=True),1e-12)
fill=np.array([R[i][~lm[i]][0] if (~lm[i]).any() else np.nan for i in range(R.shape[0])])
covered=lm&(R!=fill[:,None])
sf_i=np.argmin(np.where(lm,R,np.inf),axis=1); tg_i=np.argmax(P,axis=1)
top1=P[np.arange(n),tg_i]; nleg=lm.sum(axis=1)
perm=rng.permutation(n); foreign_i=np.argmax(np.where(lm,pol[perm],-1.0),axis=1)
fens=[];keep=[];rand_i=np.empty(n,dtype=np.int64)
for i in range(n):
    b=S.decode_board(X[i])
    m=np.asarray(legal_move_mask_for_encoding(b,policy_encoding="lc0_1858")).astype(bool).reshape(-1)
    if not np.array_equal(m,lm[i]): continue
    legal=list(b.legal_moves)
    rand_i[i]=move_to_index_for_encoding(legal[int(rng.integers(len(legal)))],b,policy_encoding="lc0_1858")
    fens.append(b.fen()); keep.append(i)
k=np.array(keep,dtype=np.int64)
tb_cov=covered[np.arange(n),tg_i]
np.savez_compressed(os.path.join(OUT,"tb4_rows_old.npz"),
    row_index=sel[k],fens=np.array(fens,dtype=object),tgt_idx=tg_i[k],sf_idx=sf_i[k],
    foreign_idx=foreign_i[k],rand_idx=rand_i[k],top1_mass=top1[k],n_legal=nleg[k],
    tgt_listed=tb_cov[k],sf_cp_regret_tgt=(R[np.arange(n),tg_i]*CAP)[k],
    agree=(sf_i==tg_i)[k],allow_pickle=True)
rep={"shards":[names[0],names[-1]],"n_rows_read":int(D["policy_target"].shape[0]),
 "n_sf_labelled":int(idx.size),"n_sampled":int(n),"A1_n_exact":int(k.size),
 "A1_frac_exact":float(k.size/n),"agree":float((sf_i==tg_i)[k].mean()),
 "mean_top1":float(top1[k].mean()),"frac_not_listed":float((~tb_cov[k]).mean()),
 "mean_nlegal":float(nleg[k].mean())}
json.dump(rep,open(os.path.join(OUT,"tb4_select_old_report.json"),"w"),indent=2,default=float)
print(json.dumps(rep,indent=2,default=float))
