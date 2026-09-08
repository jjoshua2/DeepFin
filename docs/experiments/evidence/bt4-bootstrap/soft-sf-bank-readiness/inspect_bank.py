"""Bounded cached-bank qualification only; no target or corpus processing."""
from pathlib import Path
import hashlib,json,os,time,resource,zipfile
ROOT=Path('/home/josh/projects/chess');OUT=Path(__file__).resolve().parent
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and os.sched_getaffinity(0)=={4,5}
start=time.time()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
scan=ROOT/'scratchpad/bt4_joint20/rank_gap_marginal_scan.json';s=load(scan)
paths={k:Path(s['inputs'][k]) for k in ['audit_set','d9_labels','bt4_cache']}
pins={str(p):s['inputs'][k+'_sha256'] for k,p in paths.items()}
assert all(sha(Path(p))==h for p,h in pins.items())
rows={k:[json.loads(x) for x in p.read_text().splitlines() if x.strip()] for k,p in paths.items()}
assert all(len(v)==4000 for v in rows.values())
assert all(len({r['key'] for r in v})==4000 for v in rows.values())
assert len({frozenset(r['key'] for r in v) for v in rows.values()})==1
fields={k:sorted(set().union(*(r.keys() for r in v))) for k,v in rows.items()}
counters={'single_d9_block':0,'timed_out':0,'d9_lines':0,'explicit_mate_lines':0,'cp_lines_without_mate':0,'missing_cp_and_mate':0}
for r in rows['d9_labels']:
 blocks=[b for b in r['depths'] if b['depth']==9]
 counters['single_d9_block']+=len(blocks)==1;counters['timed_out']+=bool(r['timed_out'])
 for b in blocks:
  for line in b['lines']:
   counters['d9_lines']+=1;counters['explicit_mate_lines']+=line[3] is not None
   counters['cp_lines_without_mate']+=line[2] is not None and line[3] is None
   counters['missing_cp_and_mate']+=line[2] is None and line[3] is None
npz=Path(s['raw_npz'])
with zipfile.ZipFile(npz) as z:npz_members=z.namelist()
geometry=ROOT/'scratchpad/bt4_joint20/target_geometry_v1/readout.json';g=load(geometry)
prior={k:g['overall']['targets'][k] for k in ['C20T05','cp10_stored','cp20_stored','cp40_stored','cp80_stored']}
more=[scan,geometry,ROOT/'scratchpad/bt4_joint20/target_geometry_v1/independent_review.json',ROOT/'scripts/build_audit_set.py',ROOT/'scratchpad/bt4_joint20/B100_preparation_v1/completed_readout.json',Path(__file__),OUT/'plan.json']
pins.update({str(p):sha(p) for p in more})
assert all(sha(Path(p))==h for p,h in pins.items())
result={'schema':1,'status':'INSUFFICIENT_4K_TRAINING_JOIN_NO_NEW_TEMPERATURE_SELECTED','input_pins':pins,'rows_per_bank':{k:len(v) for k,v in rows.items()},'observed_fields':fields,'d9_observation_counts':counters,'rank_npz_members':npz_members,'join':'Exact matching4000unique normalized FEN keys; source is selfplay/curriculum category, not corpus identity.','missing':['Source corpus/shard/physical row and game identity linking these4000positions to original18.91M training rows','Original full-history input identity and actual stored C20T05 target on those same rows','Evidence that audit sample is a qualified training-only selection rather than historical replay-derived audit positions'], 'effective_cp_semantics':'Raw cp/mate d9 observations retained. Existing mixer _effective_cp prioritizes explicit mate via mate_to_effective_cp, then raw cp, and refuses both missing. Do not recover cp from quantized targets. Raw cp may itself contain encoded mate-band values.','training_sample_already_available':{'rows':g['rows'],'games':g['games'],'readout_sha256':sha(geometry),'previous_descriptive_results_reused_not_recomputed':prior,'previous_descriptive_closest_cp_temperature':g['descriptive_closest_mean_stored_entropy_temperature_cp']['C20T05'],'limits':'128 selected seen-training rows,43games,one shard; not representative or an independent confirmation; previous10cp descriptive closest is not adopted as a new4K choice.'},'next_smallest_gap':'Bank a separately declared source-qualified training slice retaining raw complete legal d9 effective-cp observations and exact stored C targets/history joins. Existing128-row bank demonstrates format; no full corpus decoding or new slice collected here.','wall_seconds':time.time()-start,'cpu_seconds':time.process_time(),'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'limits':['No new softmax computation/temperature selection because required4Ktrainingjoin is absent.','No payload corpus scan, model/inference/GPU, training, materialization or PR.','Field/identity probe does not independently certify every d9 legal support set.']}
with (OUT/'readiness.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'status':result['status'],'counts':counters,'wall_seconds':result['wall_seconds'],'sha256':sha(OUT/'readiness.json')}))
