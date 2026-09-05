"""The tail is defined by |dQ|>=0.10, which phase 1 showed is 30% rows where the
TARGET is BETTER. Split it by sign -- 'does the net learn the BAD half' is the
question, and the unsigned tail cannot answer it."""
from __future__ import annotations
import json, numpy as np
O="/home/josh/projects/chess/scratchpad/target_vs_bt4/"
RNG=np.random.default_rng(4242)
def wil(k,n,z=1.96):
    if n==0: return [float('nan')]*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return [round(float(c-h),4),round(float(c+h),4)]
def bm(a,reps=10000):
    a=np.asarray(a,float)
    if a.size==0: return [float('nan')]*3
    m=a[RNG.integers(0,a.size,size=(reps,a.size))].mean(axis=1)
    return [round(float(a.mean()),4),round(float(np.percentile(m,2.5)),4),round(float(np.percentile(m,97.5)),4)]
r=np.load(O+"tb4_rows.npz",allow_pickle=True); nt=np.load(O+"tb4_net_new.npz",allow_pickle=True)
Q=np.load(O+"tb4_q_winner.npz")["Q"]; Qn=np.load(O+"tb4_q_netmove.npz")["Q_net"]
post=np.load(O+"tb4_postckpt_mask.npy"); tr=~post
qt,qs=Q[:,0],Q[:,1]; dQ=qt-qs; dis=~r["agree"].astype(bool)
at=(nt["net_argmax"]==r["tgt_idx"]); af=(nt["net_argmax"]==r["foreign_idx"])
pt,pf=nt["p_tgt"],nt["p_foreign"]; nl=r["n_legal"].astype(float)
def cell(m,name):
    n=int(m.sum())
    if n==0: return {"name":name,"n":0}
    return {"name":name,"n":n,"chance":round(float(np.mean(1/nl[m])),4),
        "P_net_eq_target":round(float(at[m].mean()),4),"ci":wil(int(at[m].sum()),n),
        "P_net_eq_PERM":round(float(af[m].mean()),4),
        "E_excess":round(float(at[m].mean()-af[m].mean()),4),
        "M_excess_mass":round(float(pt[m].mean()-pf[m].mean()),4),
        "mean_dQ_target_vs_sf":round(float(dQ[m].mean()),4),
        "mean_Q_net_minus_target":bm(Qn[m]-qt[m]),
        "mean_Q_net_minus_sfbest":bm(Qn[m]-qs[m])}
bad=tr&dis&(dQ<=-0.10); good=tr&dis&(dQ>=0.10); mid=tr&dis&(np.abs(dQ)<0.10)
R={"tail_BAD_target_worse":cell(bad,"dQ<=-0.10 target WORSE"),
   "tail_GOOD_target_better":cell(good,"dQ>=+0.10 target BETTER"),
   "non_tail":cell(mid,"|dQ|<0.10")}
for k in ("tail_BAD_target_worse","tail_GOOD_target_better"):
    R[k]["ratio_argmax_vs_nontail"]=round(R[k]["E_excess"]/R["non_tail"]["E_excess"],3)
    R[k]["ratio_mass_vs_nontail"]=round(R[k]["M_excess_mass"]/R["non_tail"]["M_excess_mass"],3)
# how much of the tail deficit does the net inherit?
R["inheritance_on_BAD_tail"]={
  "mean_dQ_target_vs_sf":round(float(dQ[bad].mean()),4),
  "mean_dQ_NET_vs_sf":round(float((Qn-qs)[bad].mean()),4),
  "frac_of_target_deficit_inherited":round(float((Qn-qs)[bad].mean()/dQ[bad].mean()),4)}
json.dump(R,open(O+"tb4_tailsign.json","w"),indent=2,default=float)
print(json.dumps(R,indent=1,default=float))
