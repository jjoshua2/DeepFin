"""One bounded descriptive pass over existing qualified banks; no fitting/inference."""
from pathlib import Path
import collections,hashlib,json,time
import numpy as np
ROOT=Path('/home/josh/projects/chess'); HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def normalized(x):
 a=np.asarray(x,dtype=np.float64);assert np.isfinite(a).all() and (a>=0).all()
 return a/a.sum(-1,keepdims=True)
def avg(x,w):return float(np.asarray(w)@np.asarray(x)/np.sum(w))
def quantile(x,w,q):
 order=np.argsort(x,kind='stable');return float(np.asarray(x)[order][np.searchsorted(np.cumsum(np.asarray(w)[order]),q*np.sum(w))])
def main():
 started=time.monotonic();plan=json.loads((HERE/'analysis_plan.json').read_text());assert sha(HERE/'analysis_plan.json')=='0496cecf9cfc13546efc5367f813aaa932340ea7642d1b7e1e761abde057e8c6'
 valuebase=ROOT/'scratchpad/bt4_joint20/value_bootstrap_readiness_v1/sample_v1';vp=valuebase/'value_rows.json';assert sha(vp)=='740d421764cfe1f4b58f809b16bf6606cdaf523ea0fdb59fe3d20244ea8ed1f5'
 vr=json.loads(vp.read_text());assert len(vr)==128
 sf=normalized([r['native_outputs']['SF'] for r in vr]);bt=normalized([r['native_outputs']['BT4'] for r in vr]);weights=np.array([r['weight'] for r in vr]);qsf=sf[:,0]-sf[:,2]
 assert np.max(np.abs(sf-np.array([r['probabilities_WDL']['SF'] for r in vr])))<1e-12
 masks={'overall':np.ones(128,bool),'SF_abs_q_le_0.1':abs(qsf)<=.1,'SF_abs_q_between_0.1_and_0.8':(abs(qsf)>.1)&(abs(qsf)<.8),'SF_abs_q_ge_0.8':abs(qsf)>=.8,'white_to_move':np.array([r['stm']=='w' for r in vr]),'black_to_move':np.array([r['stm']=='b' for r in vr])}
 values={};value_rows=[]
 for alpha in plan['value_doses']:
  ideal=(1-alpha)*sf+alpha*bt;stored=ideal.astype(np.float16);p=normalized(stored);q=p[:,0]-p[:,2];dq=q-qsf;tv=np.abs(p-sf).sum(1)/2;draw=p[:,1]-sf[:,1]
  assert np.max(np.abs(np.abs(ideal-sf).sum(1)/2-alpha*np.abs(bt-sf).sum(1)/2))<1e-12
  groups={}
  for name,mask in masks.items():
   w=weights[mask];groups[name]={'rows':int(mask.sum()),'weight_fraction':float(w.sum()/weights.sum()),'mean_TV':avg(tv[mask],w),'mean_abs_q_shift':avg(abs(dq[mask]),w),'mean_signed_q_shift':avg(dq[mask],w),'median_abs_q_shift':quantile(abs(dq[mask]),w,.5),'p90_abs_q_shift':quantile(abs(dq[mask]),w,.9),'max_abs_q_shift':float(abs(dq[mask]).max()),'mean_abs_draw_shift':avg(abs(draw[mask]),w),'fraction_abs_q_ge_0.01':avg(abs(dq[mask])>=.01,w),'fraction_abs_q_ge_0.05':avg(abs(dq[mask])>=.05,w),'fraction_abs_q_ge_0.1':avg(abs(dq[mask])>=.1,w),'fraction_stored_row_changed':avg(np.any(stored[mask]!=np.asarray([r['native_outputs']['SF'] for r in vr],dtype=np.float16)[mask],axis=1),w)}
  values[str(alpha)]=groups
  for i,r in enumerate(vr):value_rows.append({'dose':alpha,**{k:r[k] for k in ['source','raw_shard','physical_row','derived_shard','derived_row','game_id','ply','input_key','stratum','weight']},'stored_WDL':stored[i].astype(float).tolist(),'normalized_WDL':p[i].tolist(),'TV':float(tv[i]),'q_shift':float(dq[i]),'draw_shift':float(draw[i])})
 bank=ROOT/'scratchpad/bt4_joint20/soft_sf_qualified_sample_v1/bank';complete=json.loads((bank/'complete.json').read_text());op=bank/'observations.json';assert sha(op)==complete['outputs']['observations.json']['sha256'];observations=json.loads(op.read_text());assert len(observations)==4096
 grouped=collections.defaultdict(list)
 for r in observations:grouped[r['derived_shard']].append(r)
 policies=[];pins={'value_rows':sha(vp),'value_review':sha(valuebase/'independent_value_readout_review.json'),'policy_complete':sha(bank/'complete.json'),'policy_observations':sha(op),'analysis_plan':sha(HERE/'analysis_plan.json')};arraypins={}
 for shard,rows in grouped.items():
  f=bank/(shard+'.npz');proof=complete['outputs'][f.name];assert f.stat().st_size==proof['bytes'] and sha(f)==proof['sha256'];arraypins[f.name]=proof['sha256']
  with np.load(f,allow_pickle=False) as z:
   legal=z['legal_mask'].astype(bool);b=z['BT4_policy'].astype(float);cp=z['effective_cp'];selected=z['selected_rows'];gids=z['game_id'];plies=z['ply_index']
  index={int(row):i for i,row in enumerate(selected)}
  assert len(index)==len(rows)
  for meta in rows:
   i=index[meta['derived_row']];assert int(gids[i])==meta['game_id'] and int(plies[i])==meta['ply'];idx=np.flatnonzero(legal[i]);raw=b[i,idx];scores=cp[i,idx];assert np.isfinite(scores).all() and np.isfinite(raw).all() and (raw>=0).all() and raw.sum()>0
   assert np.all(b[i,~legal[i]]==0);native=raw/raw.sum();positive=native>0;logits=np.full(len(idx),-np.inf);logits[positive]=np.log(native[positive])/.5;p=np.exp(logits-logits.max());p/=p.sum();order=np.lexsort((idx,-p));top=int(order[0]);bestcp=float(scores.max());assert bestcp==meta['best_effective_cp']
   for rule in plan['reverse_policy']['candidate_rules']:
    selected_local=np.array([j for j in order[:3] if p[j]>=rule['ratio_to_max']*p[top]],int);assert top in selected_local
    winner=max(selected_local,key=lambda j:(scores[j],p[j],-int(idx[j])));changed=winner!=top
    policies.append({**{k:meta[k] for k in ['source_dir','raw_shard','physical_row','derived_shard','derived_row','game_id','ply','input_key','stratum','weight','mate_present']},'rule':rule['name'],'candidate_indices':idx[selected_local].tolist(),'candidate_size':len(selected_local),'candidate_mass':float(p[selected_local].sum()),'global_SF_best_covered':bool(np.any(scores[selected_local]==bestcp)),'BT4_top_is_global_SF_best':bool(scores[top]==bestcp),'SF_changes_BT4_top':bool(changed),'BT4_top_index':int(idx[top]),'SF_selected_index':int(idx[winner]),'BT4_top_effective_cp':float(scores[top]),'SF_selected_effective_cp':float(scores[winner]),'SF_best_effective_cp':bestcp,'SF_selected_relative_BT4_probability':float(p[winner]/p[top]),'raw_cp_gain':float(scores[winner]-scores[top])})
 reverse={}
 for rule in plan['reverse_policy']['candidate_rules']:
  rs=[r for r in policies if r['rule']==rule['name']];out={}
  for group in ['overall','no_mate','mate_present']:
   rr=[r for r in rs if group=='overall' or r['mate_present']==(group=='mate_present')];w=np.array([r['weight'] for r in rr]);multi=np.array([r['candidate_size']>=2 for r in rr]);change=np.array([r['SF_changes_BT4_top'] for r in rr]);cp_gain=np.array([r['raw_cp_gain'] for r in rr]);out[group]={'rows':len(rr),'mean_candidate_size':avg([r['candidate_size'] for r in rr],w),'multi_choice_fraction':avg(multi,w),'mean_candidate_mass':avg([r['candidate_mass'] for r in rr],w),'global_SF_best_covered_fraction':avg([r['global_SF_best_covered'] for r in rr],w),'BT4_top_global_SF_best_fraction':avg([r['BT4_top_is_global_SF_best'] for r in rr],w),'SF_changes_BT4_top_fraction':avg(change,w),'SF_changes_BT4_top_rows':int(change.sum()),'disagreement_given_multi_choice':avg(change[multi],w[multi]) if multi.any() else None,'mean_nonmate_cp_gain':avg(cp_gain,w) if group=='no_mate' else None,'mean_nonmate_cp_gain_when_changed':avg(cp_gain[change],w[change]) if group=='no_mate' and change.any() else None,'mean_changed_move_relative_BT4_probability':avg([rr[i]['SF_selected_relative_BT4_probability'] for i in np.flatnonzero(change)],w[change]) if change.any() else None}
  reverse[rule['name']]=out
 report={'status':'COMPLETED_DESCRIPTIVE_VALUE_DOSE_AND_REVERSE_POLICY_ANALYSIS','elapsed_seconds':time.monotonic()-started,'value_rows':128,'policy_rows':4096,'value_dose':values,'reverse_policy':reverse,'input_pins':pins,'policy_array_pins':arraypins,'analysis_sha256':sha(Path(__file__)),'scope':['Existing training samples, inherited source-qualified joins and conditional sampling weights; no iid intervals, calibration/strength verdict or optimal dose/threshold selection.','Value mixture computes only search_wdl; B100 remains100% sharpened BT4 policy with SF value. Changing policy SF role is a separate axis.','Reverse candidate gates use reconstructed ideal B100T.5 legal probabilities before float32/float16 storage; exact stored B100 corpus admission is not performed.','SF reranking uses original float64 all-legal d9 effective cp, not quantized stored SF maxima. Candidate SF scores only rank moves here; no policy target, loss, cp-temperature or game is produced.','No original corpus/model/engine/GPU or raw-history replay; only saved value JSON and64 existing sample NPZ parts.']}
 for name,data in [('readout.json',report),('value_rows.json',value_rows),('reverse_policy_rows.json',policies)]:
  with (HERE/name).open('x') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'value':{k:v['overall'] for k,v in values.items()},'reverse':{k:v['overall'] for k,v in reverse.items()},'elapsed':report['elapsed_seconds']},indent=2))
if __name__=='__main__':main()
