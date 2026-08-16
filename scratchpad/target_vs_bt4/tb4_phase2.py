"""Phase 2 verdict pass. Every cross-stratum statistic is an EXCESS over the
permuted-target control, because the strata have different chance levels."""
from __future__ import annotations
import json, os
import numpy as np
OUT="/home/josh/projects/chess/scratchpad/target_vs_bt4"
RNG=np.random.default_rng(20260816)
def wil(k,n,z=1.96):
    if n==0: return [float('nan')]*2
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d; return [round(float(c-h),4),round(float(c+h),4)]
def bmean(a,reps=10000):
    a=np.asarray(a,float)
    if a.size==0: return [float('nan')]*3
    m=a[RNG.integers(0,a.size,size=(reps,a.size))].mean(axis=1)
    return [round(float(a.mean()),4),round(float(np.percentile(m,2.5)),4),round(float(np.percentile(m,97.5)),4)]
def track(m, agree_t, agree_f, p_t, p_f, nl):
    n=int(m.sum())
    if n==0: return {"n":0}
    at=float(agree_t[m].mean()); af=float(agree_f[m].mean())
    return {"n":n,"chance_E_inv_nlegal":round(float(np.mean(1/nl[m])),4),
            "P_net_eq_target":round(at,4),"ci":wil(int(agree_t[m].sum()),n),
            "P_net_eq_PERMUTED_target":round(af,4),"ci_perm":wil(int(agree_f[m].sum()),n),
            "E_excess_argmax":round(at-af,4),
            "mass_on_target":round(float(p_t[m].mean()),4),
            "mass_on_PERMUTED_target":round(float(p_f[m].mean()),4),
            "M_excess_mass":round(float(p_t[m].mean()-p_f[m].mean()),4)}
def ratio_boot(mA,mB,agree_t,agree_f,p_t,p_f,reps=10000):
    iA=np.flatnonzero(mA); iB=np.flatnonzero(mB); out_a=[];out_m=[]
    for _ in range(reps):
        a=iA[RNG.integers(0,iA.size,iA.size)]; b=iB[RNG.integers(0,iB.size,iB.size)]
        ea=agree_t[a].mean()-agree_f[a].mean(); eb=agree_t[b].mean()-agree_f[b].mean()
        ma=p_t[a].mean()-p_f[a].mean();        mb=p_t[b].mean()-p_f[b].mean()
        if eb>1e-9: out_a.append(ea/eb)
        if mb>1e-9: out_m.append(ma/mb)
    f=lambda v:[round(float(np.percentile(v,2.5)),3),round(float(np.percentile(v,97.5)),3)] if v else None
    return f(out_a), f(out_m)

def load(tag,rowfile):
    r=np.load(f"{OUT}/{rowfile}",allow_pickle=True); nt=np.load(f"{OUT}/tb4_net_{tag}.npz",allow_pickle=True)
    return r,nt
R={}
# ---------------- primary population -----------------------------------
r,nt=load("new","tb4_rows.npz")
post=np.load(f"{OUT}/tb4_postckpt_mask.npy"); trained=~post
Q=np.load(f"{OUT}/tb4_q_winner.npz")["Q"]; Qnet=np.load(f"{OUT}/tb4_q_netmove.npz")["Q_net"]
qt,qs=Q[:,0],Q[:,1]; dQ=qt-qs
agree_t=(nt["net_argmax"]==r["tgt_idx"]); agree_f=(nt["net_argmax"]==r["foreign_idx"])
agree_s=(nt["net_argmax"]==r["sf_idx"])
p_t,p_f=nt["p_tgt"],nt["p_foreign"]; nl=r["n_legal"].astype(float)
dis=~r["agree"].astype(bool); listed=r["tgt_listed"].astype(bool); tail=dis&(np.abs(dQ)>=0.10)
R["populations"]={"NEW_TRAINED":int(trained.sum()),"NEVER_SEEN":int(post.sum())}
R["metric1_2_NEW_TRAINED"]={
 "all":track(trained,agree_t,agree_f,p_t,p_f,nl),
 "disagreement":track(trained&dis,agree_t,agree_f,p_t,p_f,nl),
 "BT4_TAIL":track(trained&tail,agree_t,agree_f,p_t,p_f,nl),
 "BT4_NONTAIL":track(trained&dis&~tail,agree_t,agree_f,p_t,p_f,nl),
 "argmax_NOT_listed":track(trained&~listed,agree_t,agree_f,p_t,p_f,nl),
 "argmax_listed":track(trained&listed,agree_t,agree_f,p_t,p_f,nl)}
def ratios(a,b,name):
    A=R["metric1_2_NEW_TRAINED"][a]; B=R["metric1_2_NEW_TRAINED"][b]
    return {"pair":name,"ratio_argmax":round(A["E_excess_argmax"]/B["E_excess_argmax"],3),
            "ratio_mass":round(A["M_excess_mass"]/B["M_excess_mass"],3)}
R["DECIDER_ratios"]={"tail_vs_nontail":ratios("BT4_TAIL","BT4_NONTAIL","tail/nontail"),
                     "notlisted_vs_listed":ratios("argmax_NOT_listed","argmax_listed","notlisted/listed")}
ca,cm=ratio_boot(trained&tail,trained&dis&~tail,agree_t,agree_f,p_t,p_f)
R["DECIDER_ratios"]["tail_vs_nontail"]["ratio_argmax_ci95"]=ca
R["DECIDER_ratios"]["tail_vs_nontail"]["ratio_mass_ci95"]=cm
ca,cm=ratio_boot(trained&~listed,trained&listed,agree_t,agree_f,p_t,p_f)
R["DECIDER_ratios"]["notlisted_vs_listed"]["ratio_argmax_ci95"]=ca
R["DECIDER_ratios"]["notlisted_vs_listed"]["ratio_mass_ci95"]=cm
# ---------------- metric 3 ---------------------------------------------
def m3(m):
    n=int(m.sum())
    if n==0: return {"n":0}
    return {"n":n,"mean_Q_net_minus_target":bmean(Qnet[m]-qt[m]),
            "mean_Q_net_minus_sfbest":bmean(Qnet[m]-qs[m]),
            "mean_Q_target_minus_sfbest":bmean(qt[m]-qs[m]),
            "P_net_better_than_target":round(float((Qnet[m]>qt[m]).mean()),4)}
R["metric3_NEW_TRAINED"]={"BT4_TAIL":m3(trained&tail),"BT4_NONTAIL":m3(trained&dis&~tail),
  "argmax_NOT_listed":m3(trained&~listed),"disagreement":m3(trained&dis),"all":m3(trained)}
# ---------------- exposure arms ----------------------------------------
R["exposure"]={"NEVER_SEEN":track(post,agree_t,agree_f,p_t,p_f,nl),
               "NEVER_SEEN_tail_INDICATIVE":track(post&tail,agree_t,agree_f,p_t,p_f,nl),
               "NEW_TRAINED":R["metric1_2_NEW_TRAINED"]["all"]}
ro,nto=load("old","tb4_rows_old.npz")
ao_t=(nto["net_argmax"]==ro["tgt_idx"]); ao_f=(nto["net_argmax"]==ro["foreign_idx"])
po_t,po_f=nto["p_tgt"],nto["p_foreign"]; nlo=ro["n_legal"].astype(float)
diso=~ro["agree"].astype(bool); listo=ro["tgt_listed"].astype(bool)
R["exposure"]["OLD_SATURATED"]=track(np.ones(len(nlo),bool),ao_t,ao_f,po_t,po_f,nlo)
R["exposure"]["OLD_notlisted"]=track(~listo,ao_t,ao_f,po_t,po_f,nlo)
R["exposure"]["OLD_listed"]=track(listo,ao_t,ao_f,po_t,po_f,nlo)
A=R["exposure"]["OLD_notlisted"];B=R["exposure"]["OLD_listed"]
R["exposure"]["OLD_ratio_notlisted_vs_listed"]={
  "ratio_argmax":round(A["E_excess_argmax"]/B["E_excess_argmax"],3),
  "ratio_mass":round(A["M_excess_mass"]/B["M_excess_mass"],3)}
R["net_vs_sf"]={"P_net_eq_sfbest_NEW":round(float(agree_s[trained].mean()),4)}
json.dump(R,open(f"{OUT}/tb4_phase2_results.json","w"),indent=2,default=float)
print(json.dumps(R,indent=1,default=float))
