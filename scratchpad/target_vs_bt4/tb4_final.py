from __future__ import annotations
import json, os, numpy as np, zarr
OUT="/home/josh/projects/chess/scratchpad/target_vs_bt4"
REPLAY="/home/josh/projects/chess/runs/pbt2_small/replay"
ERA="train_trial_1d175_00000_0_lr=0.0000_2026-08-14_13-53-53"
RNG=np.random.default_rng(777)
def wil(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return (float(c-h),float(c+h))
def cell(qt,qs):
    n=qt.size
    if n==0: return {"n":0}
    k=int((qs>qt).sum()); return {"n":n,"C":round(k/n,4),"ci":[round(v,4) for v in wil(k,n)],
        "mean_dQ":round(float((qt-qs).mean()),4)}
rows=np.load(f"{OUT}/tb4_rows.npz",allow_pickle=True)
Qw=np.load(f"{OUT}/tb4_q_winner.npz")["Q"]; Qq=np.load(f"{OUT}/tb4_q_q.npz")["Q"]
dis=~rows["agree"].astype(bool); listed=rows["tgt_listed"].astype(bool); top1=rows["top1_mass"]
out={}
for nm,Q in (("winner",Qw),("q",Qq)):
    qt,qs=Q[:,0],Q[:,1]
    out[nm]={"decider":cell(qt[dis],qs[dis]),
             "listed":cell(qt[dis&listed],qs[dis&listed]),
             "not_listed":cell(qt[dis&~listed],qs[dis&~listed]),
             "top1_bins":{f"[{a},{b})":cell(qt[m],qs[m]) for a,b in
                 ((0,.5),(.5,.9),(.9,.99),(.99,1.01)) for m in [dis&(top1>=a)&(top1<b)]}}
# how unique is "SF's best move"?
sd=os.path.join(REPLAY,ERA,"replay_shards")
names=sorted(n for n in os.listdir(sd) if n.endswith(".zarr"))[-16:]
R=np.concatenate([np.asarray(zarr.open(os.path.join(sd,n),mode="r")["sf_p0_regret"][:]) for n in names])
L=np.concatenate([np.asarray(zarr.open(os.path.join(sd,n),mode="r")["legal_mask"][:]) for n in names]).astype(bool)
rid=rows["row_index"]; Rs=R[rid].astype(np.float64); Ls=L[rid]
nzero=(np.where(Ls,Rs,np.inf)==0.0).sum(axis=1)
tgt=rows["tgt_idx"]
tie_hit=np.array([Rs[i,tgt[i]]==0.0 for i in range(len(tgt))])
out["sf_best_uniqueness"]={"mean_n_moves_at_regret_0":float(nzero.mean()),
  "frac_rows_with_ties_at_best":float((nzero>1).mean()),
  "frac_rows_target_argmax_at_regret_0":float(tie_hit.mean()),
  "frac_agree_by_index":float((~dis).mean())}
print(json.dumps(out,indent=1))
json.dump(out,open(f"{OUT}/tb4_final.json","w"),indent=2,default=float)
