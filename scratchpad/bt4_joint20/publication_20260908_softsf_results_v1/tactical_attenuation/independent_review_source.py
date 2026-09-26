from pathlib import Path
import json,hashlib,collections,math,time
import numpy as np
r=Path('/home/josh/projects/chess/scratchpad/bt4_joint20');s=r/'sf_tactical_filter_readiness_v1';b=r/'soft_sf_qualified_sample_v1/bank'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((s/'readout.json').read_text());c=json.loads((b/'complete.json').read_text());obs=json.loads((b/'observations.json').read_text())
assert a['script_sha256']==sha(s/'analyze.py') and a['plan_sha256']==sha(s/'PLAN.md')
assert a['observations_sha256']==sha(b/'observations.json')==c['outputs']['observations.json']['sha256']
lookup={}
for x in a['rows']:
 key=(x['source_dir'],x['raw_shard'],x['physical_row'],x['gap_cp']);assert key not in lookup;lookup[key]=x
assert len(lookup)==8192
groups=collections.defaultdict(list)
for x in obs:groups[x['derived_shard']].append(x)
checked=0;maxtv=0.;maxerr=0.
for name,rows in groups.items():
 p=b/(name+'.npz');assert sha(p)==a['input_pins'][p.name]==c['outputs'][p.name]['sha256']
 with np.load(p,allow_pickle=False) as z:
  idx={int(x):j for j,x in enumerate(z['selected_rows'])}
  for row in rows:
   i=idx[row['derived_row']];assert z['game_id'][i]==row['game_id'] and z['ply_index'][i]==row['ply']
   legal=z['legal_mask'][i].astype(bool);prob=z['BT4_policy'][i,legal].astype(np.float64);cp=z['effective_cp'][i,legal].astype(np.float64)
   prob=prob**2;prob/=prob.sum();loss=max(cp)-cp
   for gap in (100,200):
    weights=np.ones(len(cp)) if row['mate_present'] else np.exp(np.maximum(np.log(.1),-np.maximum(0,loss-gap)/100))
    q=prob*weights;q/=q.sum();o=lookup[(row['source_dir'],row['raw_shard'],row['physical_row'],gap)]
    assert o['game_id']==row['game_id'] and o['weight']==row['weight'] and o['mate_excluded']==bool(row['mate_present'])
    ent=lambda p:-sum(float(t)*math.log(float(t)) for t in p if t>0)
    metrics={'exposed_mass':sum(prob[weights<1]),'tv':sum(abs(q-prob))/2,'top_changed':np.argmax(q)!=np.argmax(prob),'entropy_change':ent(q)-ent(prob)}
    for k,v in metrics.items():maxerr=max(maxerr,abs(float(v)-o[k]));assert math.isclose(float(v),o[k],abs_tol=1e-12,rel_tol=1e-11),(k,v,o[k])
    keep=(weights==1)&(prob>0)
    assert np.array_equal(np.argsort(prob[keep],kind='stable'),np.argsort(q[keep],kind='stable'))
    if row['mate_present']:maxtv=max(maxtv,metrics['tv']);assert metrics['tv']<1e-14
    checked+=1
for gap in ('100','200'):
 rows=[x for x in a['rows'] if x['gap_cp']==int(gap)];w=sum(x['weight'] for x in rows)
 for k,v in a['summary'][gap].items():assert math.isclose(sum(x[k]*x['weight'] for x in rows)/w,v,abs_tol=1e-12)
report={'schema':1,'status':'PASS_INDEPENDENT_DESCRIPTIVE_ARITHMETIC','source_readout_sha256':sha(s/'readout.json'),'source_plan_sha256':sha(s/'PLAN.md'),'source_analysis_sha256':sha(s/'analyze.py'),'original_bank_complete_sha256':sha(b/'complete.json'),'review_source_sha256':sha(Path(__file__)),'rows':4096,'saved_npz_parts':64,'row_gap_cases':checked,'maximum_metric_error':maxerr,'mate_maximum_TV_roundoff':maxtv,'summary':a['summary'],'input_pins':a['input_pins'],'observations_sha256':a['observations_sha256'],'findings':[],'limits':['Single saved training sample; no new inference/engine/corpus scan.','Numerical renormalization causes up to recorded machine roundoff on intended unchanged mate rows; no stored target produced.','SF d9 disagreement is not verified tactical error/depth-stable ground truth;27.56percent weighted mate-containing rows are unattenuated.','Ideal sharpened BT4 probability arithmetic checked; actual float32/float16 target rewrite/admission and strength not qualified.']}
(s/'independent_review.json').write_text(json.dumps(report,indent=2)+'\n')
print(sha(s/'independent_review.json'),checked,maxerr)
