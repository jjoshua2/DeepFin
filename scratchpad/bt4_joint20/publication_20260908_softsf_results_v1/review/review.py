from pathlib import Path
import sys,json,math,statistics,hashlib,time
import numpy as np
TOOLS=Path('/tmp/deepfin-softsf10-screen-tools')
sys.path.insert(0,str(TOOLS))
from scripts import bt4_recipe_readout as reader
from chess_anti_engine.eval.sprt import SprtMonitor
ROOT=Path('/home/josh/projects/chess')
BASE=ROOT/'scratchpad/bt4_joint20/SoftSF10_B100_recipe_screen_run01'
OUT=BASE/'independent_result_review_v1'
bindings={}
def read(path):
 p=Path(path);raw=p.read_bytes();bindings[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
def pin(item):
 a=read(item['path']);assert bindings[item['path']]==item['sha256'];return a
def close(a,b): assert math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-10),(a,b)
package=read(BASE/'complete.json');assert package['complete'] is True
for side in ('low','high'):
 c=pin(package[side]);pin(c['readout']);pin(c['reader_manifest'])
# Existing production admission, one high read (which includes low), no model inference.
high_manifest=read(BASE/'high.reader_manifest.json')
replayed=reader.read_cell(high_manifest)
assert replayed['status']=='VALID_CELL'
all_rows={};calc={};pairs={};manifests={}
for side in ('low','high'):
 manifest=read(BASE/(side+'.reader_manifest.json'));manifests[side]=manifest
 for key in ('result','process','launch','opening_panel'): pin(manifest[key])
 proc=read(BASE/side/'process.json');assert proc['exit_code']==0 and proc['process_complete']
 assert proc['stage_seconds']<=5400
 assert proc['cwd']=='/tmp/deepfin-ordered-arena-lookahead-runtime'
 launch=read(BASE/(side+'.launch.json'))
 assert launch['candidate_role']=='SoftSF10' and launch['reference_role']=='B100'
 assert launch['identities']['candidate']['sha256']=='3cf979d0a2f2b4d0f8fd5a9398c36d6d7c4f39158144b475e08f68a2ac44b1bd'
 assert launch['identities']['reference']['sha256']=='b30ab345d0cf3acfb51bea6c90a91aef3c1dd5edb78da3c92d3a504fb2735d62'
 p=BASE/side/'arena.games.jsonl';raw=p.read_bytes();bindings[str(p)]=hashlib.sha256(raw).hexdigest();assert bindings[str(p)]==manifest['bank']['sha256']
 records=[json.loads(x) for x in raw.splitlines()];assert records[0]['kind']=='header'
 rows=records[1:];all_rows[side]=rows;by={};scores={};counts={'W':0,'D':0,'L':0}
 for row in rows:
  assert row['kind']=='game'
  key=(row['pair_id'],row['half']);assert key not in by
  white={'1-0':1.,'1/2-1/2':.5,'0-1':0.}[row['result']]
  score=white if row['half']==0 else 1-white
  assert score==row['score_candidate'];by[key]=row
  if row['pair_id']<128: counts[{1.:'W',.5:'D',0.:'L'}[score]]+=1
 for i in range(128): scores[i]=(by[i,0]['score_candidate']+by[i,1]['score_candidate'])/2
 pairs[side]=scores
 mu=statistics.mean(scores.values());se=statistics.stdev(scores.values())/math.sqrt(128)
 elo=lambda x:400*math.log10(x/(1-x))
 result={'WDL':counts,'score':mu,'score_se':se,'elo':elo(mu),'elo_ci95':[elo(mu-1.96*se),elo(mu+1.96*se)],'pairs':128,'finished_games':len(rows)}
 saved=read(BASE/side/'readout.stdout.json');assert saved['status']=='VALID_CELL'
 for k in ('score','score_se','elo'):close(result[k],saved['result'][k])
 for a,b in zip(result['elo_ci95'],saved['result']['elo_ci95']):close(a,b)
 assert saved['fixed_core_pair_scores']==list(scores.values())
 result['stage_seconds']=proc['stage_seconds'];calc[side]=result
 if side=='high':assert len(rows)==256 and set(by)=={(i,h) for i in range(128) for h in (0,1)}
# Replay pair arrivals in bank order, independently of final reader's bulk replay.
monitor=SprtMonitor(reader.SPEC,pairs_cap=500,granularity='pair')
seen={};first=None
for n,row in enumerate(all_rows['low'],1):
 key=row['pair_id'],row['half'];seen[key]=row['score_candidate']
 complete={i:seen[i,0]+seen[i,1] for i in range(500) if (i,0) in seen and (i,1) in seen}
 monitor.update(list(complete.values()),pair_ids=list(complete))
 if monitor.crossed() and first is None:first={'finished_record_ordinal':n,'pairs':monitor.pairs,'verdict':monitor.verdict,'llr':monitor.llr}
assert first and first['pairs']==128 and first['verdict']=='H0'
assert len(monitor.trajectory)==1 and monitor.trajectory[0][0]==128
observed=read(BASE/'low/readout.stdout.json')['sprt']
close(monitor.llr,observed['llr']);assert monitor.pair_scores==[pairs['low'][i]*2 for i in range(128)]
assert observed['scored_pair_ids']==list(range(128))
assert len(all_rows['low'])==371 and len(observed['inflight_games'])==13 and observed['not_started_games']==616
assert {(r['pair_id'],r['half']) for r in all_rows['low']}|{tuple(x) for x in observed['inflight_games']}=={(i,h) for i in range(192) for h in (0,1)}
calc['low'].update(first_crossing=first,finished_outside_deciding_prefix=115,inflight=13,not_started=616,admitted=384,complete_pairs=len(complete),consultations_not_reconstructible=True)
# Same exact opening IDs/FENs at each depth, covariance-preserving independent bootstrap.
low={(r['pair_id'],r['half']):r for r in all_rows['low']}
for row in all_rows['high']:assert row['start_fen']==low[row['pair_id'],row['half']]['start_fen']
delta=np.array([pairs['high'][i]-pairs['low'][i] for i in range(128)])
rng=np.random.Generator(np.random.PCG64(20260903))
means=delta[rng.integers(0,128,size=(10000,128))].mean(axis=1)
contrast={'mean_score400_minus100':float(delta.mean()),'CI95':np.percentile(means,[2.5,97.5]).tolist(),'pairs':128,'samples':10000,'seed':20260903,'generator':'PCG64'}
saved=read(BASE/'high/readout.stdout.json')['fixed_core_cross_budget']
close(contrast['mean_score400_minus100'],saved['score_advantage_400_minus_100']);assert contrast['CI95']==saved['paired_bootstrap_ci95']
close(sum(calc[s]['stage_seconds'] for s in ('low','high')),package['gpu_seconds'])
close(package['gpu_seconds']+package['training_gpu_seconds'],package['package_gpu_seconds'])
assert package['package_gpu_seconds']<7.5*3600
prior=read(ROOT/'scratchpad/bt4_joint20/SoftSF10_preparation_v1/training_arena_registration_v1/independent_arena_launch_review_v1/receipt.json')
assert prior['status']=='PASS_FINAL_SOFTSF10_B100_LAUNCH_READINESS_NO_LAUNCH'
for p in (Path(__file__),TOOLS/'scripts/bt4_recipe_readout.py',TOOLS/'chess_anti_engine/eval/sprt.py'):
 bindings[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
for p,h in bindings.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
receipt={'schema':1,'status':'PASS_COMPLETED_SOFTSF10_B100_INDEPENDENT_REVIEW','reviewed_unix':time.time(),'bindings':bindings,'calculations':calc,'aligned_interaction':contrast,'arena_seconds':package['gpu_seconds'],'package_seconds':package['package_gpu_seconds'],'findings':[],'conclusions':['The registered low ordered GSPRT reaches H0 at its first128-pair look; stopped Elo and its ordinary interval are descriptive, not sequentially calibrated.','Protected high400 probe completed all128 pairs independently of low verdict and strongly favors B100 at these matched checkpoints.','Aligned score interaction interval crosses zero: no supported relative search-scaling recovery from100 to400, and two budgets do not determine a scaling curve.','No promotion; this is a same-seed development comparison. SoftSF10 keeps SF value supervision and changes the policy recipe; this does not decide every softer SF temperature or neural-value intervention.'],'verification':['One actual production high reader admission including low, plus independent raw result/POV/pair arithmetic and arrival-order canonical monitor replay.','Both banks/settings/process argv/terminal runtime/checkpoint hashes and same opening panel bind to the previously independently qualified final launch.','First crossing stays128 despite later completed suffix; admitted384=371finished+13inflight, with616unstarted. Consultation count/inflight reality remain producer telemetry, not inferred from bank ordering.','Independent PCG64 paired bootstrap recomputes registered aligned interaction; metadata/source hashes stable through review.'],'limits':['No new matches, inference, model loading, original corpus scan, native build or test suite.','Full played move lists and consumed initial history are not stored in these game rows; legal play/history lineage relies on frozen producer and qualified opening panel.','Inherited training/purity/runtime caveats from the pinned launch review remain.','No controlled lookahead speedup claim: checkpoints, game trajectories and stopping samples differ from historical matches.']}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'status':receipt['status'],'sha256':hashlib.sha256((OUT/'receipt.json').read_bytes()).hexdigest(),'calculations':calc,'contrast':contrast}))
