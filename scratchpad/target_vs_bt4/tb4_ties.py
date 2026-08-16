from __future__ import annotations
import json, os, numpy as np, zarr
OUT="/home/josh/projects/chess/scratchpad/target_vs_bt4"
REPLAY="/home/josh/projects/chess/runs/pbt2_small/replay"
ERA="train_trial_1d175_00000_0_lr=0.0000_2026-08-14_13-53-53"
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return [round(float(c-h),4),round(float(c+h),4)]
def cell(qt,qs):
    n=int(qt.size)
    if n==0: return {"n":0}
    k=int((qs>qt).sum())
    return {"n":n,"C":round(k/n,4),"ci":wil(k,n),"mean_dQ":round(float((qt-qs).mean()),4)}
rows=np.load(f"{OUT}/tb4_rows.npz",allow_pickle=True)
sd=os.path.join(REPLAY,ERA,"replay_shards")
names=sorted(n for n in os.listdir(sd) if n.endswith(".zarr"))[-16:]
R=np.concatenate([np.asarray(zarr.open(os.path.join(sd,n),mode="r")["sf_p0_regret"][:]) for n in names])
rid=rows["row_index"]; tgt=rows["tgt_idx"]
r_tgt=np.array([float(R[rid[i],tgt[i]]) for i in range(len(tgt))])
dis=~rows["agree"].astype(bool)
strict = r_tgt>0.0          # SF says the target's move is STRICTLY worse
tie    = dis & (r_tgt==0.0) # different move, but SF rates it EXACTLY equal
res={"frac_rows_index_disagree":float(dis.mean()),
     "frac_rows_SF_strictly_worse":float(strict.mean()),
     "frac_rows_tied_at_SF_best":float(tie.mean())}
for nm in ("winner","q"):
    Q=np.load(f"{OUT}/tb4_q_{nm}.npz")["Q"]; qt,qs=Q[:,0],Q[:,1]
    res[nm]={"index_disagreement":cell(qt[dis],qs[dis]),
             "SF_strictly_worse_only":cell(qt[strict],qs[strict]),
             "SF_tied_at_best_only":cell(qt[tie],qs[tie])}
print(json.dumps(res,indent=1)); json.dump(res,open(f"{OUT}/tb4_ties.json","w"),indent=2)
