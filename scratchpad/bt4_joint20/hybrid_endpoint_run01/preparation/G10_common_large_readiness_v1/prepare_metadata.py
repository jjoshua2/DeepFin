"""Fixed metadata-only next batch; never opens raw or Zarr payloads."""
import copy, datetime, hashlib, importlib.util, json, math, os, pathlib, shutil, stat
P=pathlib.Path
ROOT=P('/home/josh/projects/chess')
PREP=ROOT/'scratchpad/bt4_joint20/hybrid_endpoint_run01/preparation'
OLD=PREP/'G10_common_parallel_v1'
OUT=PREP/'G10_common_large_readiness_v1'
assert not OUT.exists(), 'exclusive preparation'
OUT.mkdir()
def sha(p): return hashlib.sha256(P(p).read_bytes()).hexdigest()
def pin(p): return {'path':str(p),'sha256':sha(p)}
def write(p,v):
 with p.open('x') as f: json.dump(v,f,indent=2);f.write('\n')
def read(p): return json.loads(P(p).read_text())
def stable_bytes(p):
 a=p.stat();b=p.read_bytes();z=p.stat();assert all(getattr(a,k)==getattr(z,k) for k in ["st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns"]);return b
def records(p,key,wanted):
 found={}
 with p.open() as f:
  for line in f:
   if not line.endswith('\n'): break # growing final record is not closed evidence
   o=json.loads(line);n=key(o)
   if n not in wanted:continue
   assert n not in found, (p,n,'duplicate closed receipt')
   found[n]=(o,line)
 return found
oldplan=read(OLD/'runner_manifest_v2.json')
prior=[f'w00-{i:05d}.jsonl.zst' for i in range(192)]+[f'w01-{i:05d}.jsonl.zst' for i in range(128)]
reviews=[read(PREP/state/'independent_completed_review_v1/review.json') for state in ['G10_common_parallel_v1','G10_common_parallel_next_v2']]
review_pins=[pin(PREP/state/'independent_completed_review_v1/review.json') for state in ['G10_common_parallel_v1','G10_common_parallel_next_v2']]
output_per_row=max(r['totals']['peak_sampled_aggregate_output_bytes']/r['totals']['physical_rows'] for r in reviews)
seconds_per_row={sid:max((r['sources'][sid]['end_unix']-r['sources'][sid]['start_unix'])/r['sources'][sid]['physical_rows'] for r in reviews) for sid in [x['source_id'] for x in oldplan['sources']]}
coverage={};all_raw={};all_teachers={}
for source in oldplan['sources']:
 sid=source['source_id'];src=P(source['source_dir']);teacher=P(source['sidecar_dir']);by={};receipt_bank={}
 with (teacher/'bt4_raw_sidecar.progress.jsonl').open() as f:
  for line in f:
   if not line.endswith('\n'):break
   d=json.loads(line);n=d.get('source_shard','')
   if not n.endswith('.jsonl.zst'):continue
   w=n.split('-')[0];by.setdefault(w,[]).append(int(n.split('-')[1].split('.')[0]))
   if w in ['w02','w03']:
    assert n not in receipt_bank;receipt_bank[n]=(d,line)
 coverage[sid]={w:{'count':len(ns),'min':min(ns),'max':max(ns)} for w,ns in sorted(by.items())}
 raw={}
 for w in ['w00','w01','w02','w03']:
  raw.update(records(src/(w+'.progress.jsonl'),lambda o:P(o.get('path','')).name,set(prior)|set(receipt_bank)))
 all_raw[sid]=raw;all_teachers[sid]=receipt_bank
candidates=sorted(set.intersection(*(set(bank)&set(all_raw[sid]) for sid,bank in all_teachers.items())))
names=[];totals={sid:0 for sid in all_raw};next_refused=None
for n in candidates:
 proposed={sid:totals[sid]+all_raw[sid][n][0]['rows'] for sid in totals}
 time_bound=max(proposed[sid]*seconds_per_row[sid] for sid in proposed)*1.25+30
 output_bound=sum(proposed.values())*output_per_row*2
 if time_bound>14400 or output_bound>8*1024**3:
  next_refused={'source_shard':n,'rows_by_source':proposed,'time_with25pct_and_grace_seconds':time_bound,'output_with2x_bytes':output_bound};break
 names.append(n);totals=proposed
assert names and next_refused
prospective=names
write(OUT/'coverage_and_sizing.json',{'status':'PROPOSED_METADATA_ONLY','coverage':coverage,'common_unused_w02_w03_candidates':len(candidates),'selected_shards_per_source':len(names),'selected_names':names,'row_counts':totals,'resource_basis':review_pins,'seconds_per_source_row':seconds_per_row,'maximum_output_bytes_per_raw_row':output_per_row,'next_canonical_shard_refused':next_refused,'rule':'Largest canonical shared w02/w03 prefix fitting25% slower-than-worst observed per-lane timing plus30s grace and2x maximum observed output/row. No credit for unmeasured overlap speedup. This sizing margin is a prospective assumption, not a proven performance bound.'})
sources=[];availability={};game_proofs={}
for lane,oldsource in enumerate(oldplan['sources']):
 sid=oldsource['source_id'];src=P(oldsource['source_dir']);bt=P(oldsource['sidecar_dir']);dest=OUT/sid;dest.mkdir();(dest/'sidecar_attrs').mkdir()
 manifest_bytes=stable_bytes(src/'manifest.json');manifest=json.loads(manifest_bytes)
 assert hashlib.sha256(manifest_bytes).hexdigest()==oldsource['source_manifest']['sha256']
 (dest/'source_manifest.json').write_bytes(manifest_bytes)
 raw=all_raw[sid]
 teachers={n:all_teachers[sid][n] for n in names}
 assert set(prior+names)<=set(raw) and set(names)<=set(teachers)
 availability[sid]={'selected_labelled':len(teachers),'missing_teacher_receipts':sorted(set(prospective)-set(teachers))}
 oldgames=set();newgames=set();owners={}
 for n in prior:
  games=raw[n][0]['games'];assert len(games)==len(set(games));oldgames.update(games)
 for n in names:
  games=raw[n][0]['games'];assert len(games)==len(set(games));assert not newgames.intersection(games);newgames.update(games)
 assert not oldgames.intersection(newgames)
 game_proofs[sid]={'source_dir':str(src),'source_config_sha256':manifest['config_sha256'],'prior_shards':prior,'selected_shards':names,'prior_games':len(oldgames),'selected_games':len(newgames),'prior_overlap':0,'within_selection_overlap':0,'selected_game_ids':sorted(newgames),'prior_game_ids_sha256':hashlib.sha256(json.dumps(sorted(oldgames),separators=(',',':')).encode()).hexdigest(),'basis':'closed selected w02/w03 and prior w00 plus active w01 progress; game identity qualified by original source/config; not cross-source bare game IDs'}
 entries=[];metadata=[]
 for n in names:
  r=raw[n][0];t=teachers[n][0];path=src/n;st=path.lstat();assert stat.S_ISREG(st.st_mode) and st.st_nlink==1
  side=bt/t['sidecar'];assert side.name==n.replace('.jsonl.zst','.bt4.zarr') and not side.is_symlink()
  attrsbytes=stable_bytes(side/'.zattrs');a=json.loads(attrsbytes)
  assert r['path']==str(path) and r['rows']==t['positions']==a['positions']==a['source_rows_claimed']
  assert a['source_dir']==str(src) and a['source_config_sha256']==manifest['config_sha256'] and a['source_manifest_sha256']==sha(src/'manifest.json')
  assert a['source_file_bytes']==st.st_size and a['source_file_mtime_ns']==st.st_mtime_ns
  for k,v in t.items():
   if k=='sidecar':continue
   assert a[k]==v,(sid,n,k)
  assert a['history_rep_fix'] is True and a['search_nodes']==0 and a['stored_dtype']=='float32' and a['policy_size']==1858
  assert a['policy_encoding']=='lc0_1858' and a['input_history_encoding']=='lc0_root_legacy_meta' and a['input_extra_features']=='v2_threats'
  # Same teacher and functional input/remapping provenance as the previous qualified batch.
  oldattrs=read(P(oldsource['source_metadata']['path']))[0]['sidecar_attrs_snapshot']['path'];oldattrs=read(oldattrs)
  for k in ['onnx_sha256','policy_output','providers','teacher_evaluations_per_position']:
   assert a[k]==oldattrs[k]
  assert a['remap_provenance']['blobs']==oldattrs['remap_provenance']['blobs']
  ap=dest/'sidecar_attrs'/(side.name+'.json');ap.write_bytes(attrsbytes)
  metadata.append({'source_path':str(path),'device':st.st_dev,'inode':st.st_ino,'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns,'sidecar_path':str(side),'sidecar_attrs_snapshot':pin(ap)})
  entries.append({'source_shard':n,'rows':r['rows'],'source_sha256':t['source_sha256']})
  assert path.lstat()==st
 for filename,bank in [('closed_source_progress.jsonl',raw),('closed_bt4_receipts.jsonl',teachers)]:
  (dest/filename).write_text(''.join(bank[n][1] for n in names))
 # Preserve the prior game ledger too so overlap can be independently recomputed without live progress.
 (dest/'prior_closed_source_progress.jsonl').write_text(''.join(raw[n][1] for n in prior))
 write(dest/'source_shards.json',{'schema':1,'source_dir':str(src),'source_config_sha256':manifest['config_sha256'],'source_manifest_sha256':sha(src/'manifest.json'),'shards':entries})
 write(dest/'source_metadata.json',metadata)
 s=copy.deepcopy(oldsource)
 for k in ['prior_w00_00000_00063_game_overlap']:s.pop(k,None)
 s.update(physical_rows=sum(x['rows'] for x in entries),shards=len(names),games=len(newgames),prior_and_active_game_overlap=0,compressed_raw_bytes=sum(x['bytes'] for x in metadata),cpu_affinity=[lane*2+2,lane*2+3])
 s['missing_result_count_ceiling']=math.floor(s['physical_rows']*.02)
 for k,filename in [('manifest_snapshot','source_manifest.json'),('selection','source_shards.json'),('closed_source_progress','closed_source_progress.jsonl'),('closed_bt4_receipts','closed_bt4_receipts.jsonl'),('source_metadata','source_metadata.json')]:s[k]=pin(dest/filename)
 for k,d in [('derived_output','derived'),('adapted_output','bt4'),('rank_output','rank')]:s[k]=str(dest/d);assert not P(s[k]).exists()
 s.pop('missing_result_count_ceiling',None)
 s.pop('max_policy_support_misses',None)
 s.pop('support_drop_ceiling',None)
 s.pop('missing_result_fraction_ceiling',None)
 sources.append(s)
write(OUT/'game_disjointness.json',game_proofs)
rows=sum(s['physical_rows'] for s in sources)
estimate=output_per_row*rows
wall=max(s['physical_rows']*seconds_per_row[s['source_id']] for s in sources)
write(OUT/'eligibility_and_resources.json',{'status':'PROPOSAL_NOT_REGISTERED_NOT_LAUNCHED','availability':availability,'prior_reviews':review_pins,'selected_rows':rows,'row_linear_output_estimate_bytes':estimate,'twofold_estimate_bytes':estimate*2,'cap_bytes':8*1024**3,'row_linear_elapsed_estimate_seconds':wall,'elapsed_with25pct_and30s_grace':wall*1.25+30,'hard_seconds_proposed':14400,'minimum_free_bytes':150*1024**3,'cpu_affinities_proposed':[[2,3],[4,5]],'limitations':['No throughput credit for newly adopted overlap; use slower prior sequential stages per lane.','Output sampling is not a quota; extrapolation is not a guarantee.','Larger manifest/index and shifted shard compositions can alter CPU/memory costs.','No source payload reads, exclusion census, inference or derivation. Raw SHA values are verified closed teacher receipt claims; actual consumers hash later.','Final runtime and omission budgets not bound; root reviews actual active Worker01 completion and host resources before registration.']})
# Confirm actual registered previous selections, not assumed naming conventions.
prior_proofs=[]
for state,file in [('G10_pipeline_pilot_v1','preregistration.json'),('G10_common_batch_v3','preregistration.json')]:
 p=PREP/state/file;d=read(p); selections={}
 for src in d['sources']:
  ns=[src['source_shard']] if 'source_shard' in src else [x['source_shard'] for x in src['selection']]
  assert all(n.startswith('w00-') for n in ns); selections[src['source_id']]=ns
 prior_proofs.append({'registration':pin(p),'selections':selections})
for state in ['G10_common_increment_v1','G10_common_parallel_v1','G10_common_parallel_next_v1','G10_common_parallel_next_v2']:
 for sid in [s['source_id'] for s in sources]:
  p=PREP/state/sid/'source_shards.json'; d=read(p); ns=[x['source_shard'] for x in d['shards']]
  assert all(n in prior for n in ns)
  prior_proofs.append({'selection':pin(p),'source_id':sid,'shards':ns})
active=read(PREP/'G10_common_worker01_v1/runner_manifest_v2.json')
for src in active['sources']:
 p=P(src['selection']['path']);assert sha(p)==src['selection']['sha256'];ns=[x['source_shard'] for x in read(p)['shards']];assert all(n in prior for n in ns)
 prior_proofs.append({'active_worker01_selection':pin(p),'source_id':src['source_id'],'shards':ns})
write(OUT/'prior_selection_review.json',{'status':'PASS_METADATA','all_prior_or_active_shards_workers00_01':True,'proofs':prior_proofs,'scope':'Pilot, original common batch, increment, parallel batch and both next-block attempts. Includes current selected batch independent of eventual completion.'})
helper=P('/tmp/deepfin-selection-admission/scripts/corpus_selection_schema.py')
spec=importlib.util.spec_from_file_location('selection_schema',helper);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
for src in sources:
 h.validate_selection_metadata(read(src['selection']['path']),source_dir=P(src['source_dir']),source_config_sha256=read(src['manifest_snapshot']['path'])['config_sha256'],source_manifest_sha256=src['source_manifest']['sha256'])
write(OUT/'metadata_admission.json',{'status':'PASS_SHARED_SCHEMA_METADATA_ONLY','helper':pin(helper),'checks':['Exact five-key source/header binding and three-key entry grammar','All selected closed receipt/attrs/row/storage metadata joins','Same teacher and functional mapping blobs','Source-qualified zero prior or within-selection game overlap'],'not_done':['No raw or sidecar chunk payload reads or hashes','No actual inventory/content-hashing consumer call','No exclusion census or survivor qualification','No runtime/runner adoption, registration, launch or derived data']})
shutil.copyfile(__file__,OUT/'prepare_metadata.py')
files=[pin(p) for p in sorted(OUT.rglob('*')) if p.is_file()]
write(OUT/'metadata_snapshot.json',{'schema':1,'status':'PREPARED_NOT_LAUNCHED','captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':sources,'physical_rows':rows,'shards':2*len(names),'selected_window':{'first':names[0],'last':names[-1],'per_source':len(names)},'scope':'exact closed metadata only; payload pins are existing verified teacher claims; no raw or policy payload read','files':files})

write(OUT/'prepared.json',{'status':'READINESS_ONLY_NOT_REGISTERED_NOT_LAUNCHED','metadata':pin(OUT/'metadata_snapshot.json'),'admission':pin(OUT/'metadata_admission.json'),'prior_selection_review':pin(OUT/'prior_selection_review.json'),'games':pin(OUT/'game_disjointness.json'),'resources':pin(OUT/'eligibility_and_resources.json'),'rows':rows,'shards':2*len(names),'next_steps':'Root selects and registers current Worker01 runtime, omission budgets and launch only after active Worker01 completion and review. Fresh paths are proposals; no output or launch manifest exists.'})
print(json.dumps(read(OUT/'prepared.json'),indent=2))
