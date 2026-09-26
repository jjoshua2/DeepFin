"""Metadata-only target-blind sample registration; no array/payload reads."""
import datetime, hashlib, json, subprocess
from pathlib import Path
ROOT=Path('/home/josh/projects/chess')
OUT=Path(__file__).resolve().parent
RT=Path('/tmp/deepfin-soft-sf-qualified-sample')
SF=ROOT/'data/nnue_derived/armB/qtemp_0.0005_hist_20m'
C=SF.parent/'qtemp_0.0005_hist_20m_bt4_sfclose_C20T05'
BT4=ROOT/'data/lc0/bt4_policy_sidecars/armB_qtemp0005_hist20m'
RAW=ROOT/'data/nnue_bootstrap/run03_s3'
sha=lambda b:hashlib.sha256(b).hexdigest()
pins={}
def pin(p):
 b=p.read_bytes();pins[str(p)]={'sha256':sha(b),'bytes':len(b)};return json.loads(b)
summary=pin(SF/'derive_targets_summary.json'); raw=pin(RAW/'summary.json'); mix=pin(C/'bt4_policy_mix_summary.json')
pin(C/'derive_targets_summary.json');pin(BT4/'bt4_policy_sidecar_summary.json')
assert pins[str(SF/'derive_targets_summary.json')]['sha256']=='391837e49773465edced77bfd13f4084edc60feeff0484078280873d942e50ef'
assert pins[str(C/'bt4_policy_mix_summary.json')]['sha256']=='5bf8502a12af0b9ce938a39ddfd9d95df4bfb2b1a0f80292d80d04109cce7100'
assert mix['source_dir']==str(SF) and mix['sidecar_dir']==str(BT4)
shards=summary['shards']; assert len(shards)==2309 and sum(s['rows'] for s in shards)==18910484
selected=[]
for stratum in range(64):
 lo=stratum*len(shards)//64;hi=(stratum+1)*len(shards)//64
 def draw(label): return sha(f'SoftSF-training-20260908-v1|{label}'.encode())
 index=min(range(lo,hi),key=lambda j:draw(f'shard:{j}'))
 shard=shards[index];n=shard['rows']
 rows=sorted(sorted(range(n),key=lambda r:draw(f'row:{index}:{r}'))[:64])
 selected.append({'stratum':stratum,'stratum_shard_start':lo,'stratum_shard_stop':hi,'shard':shard['path'],'shard_rows':n,'rows':rows,'row_inclusion_probability':64/(n*(hi-lo)),'inverse_probability_weight':n*(hi-lo)/64})
 for folder in [SF,C,BT4]:
  p=folder/shard['path'];pin(p/'.zattrs');pin(p/'.zgroup')
  for child in sorted(p.iterdir()):
   if child.is_dir() and (child/'.zarray').exists():pin(child/'.zarray')
for rel in ['scripts/derive_corpus_targets.py','scripts/sf_d9_rank_sidecar.py','scripts/bt4_policy_mix.py','scripts/audit_label_candidates.py','scripts/gen_sf_rooted_corpus.py','chess_anti_engine/stockfish/wdl.py','chess_anti_engine/moves/encode.py']:
 p=RT/rel;b=p.read_bytes();pins[str(p)]={'bytes':len(b),'sha256':sha(b)}
plan={'schema':1,'status':'PREPARED_NOT_EXECUTED','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runtime_checkout':str(RT),'runtime_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=RT,text=True).strip(),'source_dir':str(RAW),'sf_dir':str(SF),'c_dir':str(C),'bt4_dir':str(BT4),'population_rows':18910484,'population_shards':2309,'seed_label':'SoftSF-training-20260908-v1','sampling':'64 contiguous equal-shard-count strata; lowest SHA256 seed-label/shard-index in each; lowest64 seed-label/shard-index/row-index hashes, before target reads. Stratified two-stage cluster probability sample, not iid. Retain inverse inclusion weights; selected games remain correlated.','selection':selected,'rows':4096,'temperatures_cp':[10,20,40,80],'temperature_rule':'After complete qualified bank only: minimum absolute weighted mean entropy difference from actual stored C, cooler tie break; report ideal and float16-stored+renormalized candidates. Exploratory training-distribution control parameter, not GPU selection or strength.','limits':{'wall_seconds_including_kill':600,'term_seconds':570,'kill_after_seconds':30,'affinity':[6,7],'numeric_threads':2,'max_raw_shards':192,'max_raw_compressed_bytes':2147483648,'max_output_bytes':536870912,'gpu_visible_devices':'','no_resampling_after_failure':True},'pins':pins,'output_dir':str(OUT/'bank'),'command':['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','NUMEXPR_NUM_THREADS=2','BLOSC_NTHREADS=2','PYTHONPATH='+str(RT),'nice','-n','19','ionice','-c','3','taskset','-c','6,7','timeout','--kill-after=30s','570s','/tmp/deepfin-bt4-sf-close-followup/.venv/bin/python',str(OUT/'collect.py'),'--plan',str(OUT/'plan.json')]}
with (OUT/'plan.json').open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
print(json.dumps({'rows':4096,'selected_shards':len(selected),'metadata_pins':len(pins),'plan_sha256':sha((OUT/'plan.json').read_bytes()),'selected_first_last':[selected[0]['shard'],selected[-1]['shard']]}))
