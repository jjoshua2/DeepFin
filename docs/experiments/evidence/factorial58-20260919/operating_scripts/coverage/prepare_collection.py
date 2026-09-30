"""Prepare only: exact missing audited Ceres rows, no GPU/queue mutation."""
import copy, hashlib,json,math,shutil,subprocess
from pathlib import Path
B=Path('/home/josh/projects/chess/scratchpad/bt4_joint20'); HOME=B/'factorial58_20260919'; OUT=HOME/'collection'; OUT.mkdir(exist_ok=True)
R=Path('/tmp/deepfin-ceres-audited-source'); OLD=B/'takeover_20260916/ceres_fast_remaining_v2/whole12'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pin(p):return {'path':str(p),'sha256':sha(p)}
def put(p,j): p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(j,indent=2)+'\n')
template=read(OLD/'shard000000/plan.json');head=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
wrapper=(OLD/'run_chunk.py').read_text();a=wrapper.index("    require(type(start) is int");b=wrapper.index("    require(operation['selection']",a)
wrapper=wrapper[:a]+'''    require(type(start) is int and type(width) is int and 0 <= start and 1 <= width <= 4,
            'bounded grouped shard range required')
    summary = Path(p['selection']['source']) / 'derive_targets_summary.json'
    require(sha(summary) == p['selection']['summary_sha256'], 'source summary pin differs')
    source_specs = json.loads(summary.read_text())['shards']
    require(len(source_specs) == p['selection']['source_shards'], 'source shard count differs')
    expected_selection = source_specs[start:start + width]
    require(len(expected_selection) == width, 'incomplete selected shard range')
'''+wrapper[b:]
wrapper=wrapper.replace("'--g10-common-qualification': p['g10_qualification']['path']", "'--audited-source-manifest': p['audited_qualification']['path']").replace("'--expected-g10-common-qualification-sha256': p['g10_qualification']['sha256']", "'--expected-audited-source-manifest-sha256': p['audited_qualification']['sha256']")
(OUT/'run_chunk.py').write_text(wrapper);shutil.copy2(OLD/'wsl_mapped_libraries.py',OUT/'wsl_mapped_libraries.py')
blocks=[]
for index,row in enumerate(read(HOME/'coverage/coverage.json')['roots'][21:]):
 name=Path(row['v50_root']).parent.name; d=OUT/name;d.mkdir(exist_ok=True);(d/'bank').mkdir(exist_ok=True)
 source=Path(row['sf_source']); summary=source/'derive_targets_summary.json';s=read(summary); specs=s['shards']
 admit=read(Path(row['v50_root'])/'bt4_value_rewrite_summary.json')['audited_source_admission']['qualification']
 chunks=[]
 for start in range(0,len(specs),4):
  chosen=specs[start:start+4]; state=d/f'shard{start:06d}';state.mkdir(exist_ok=True);p=copy.deepcopy(template)
  p.update(runtime=str(R),runtime_head=head,output=str(d/'bank'/state.name),start_shard=start,max_shards=len(chosen),audited_qualification=admit,
   purpose='Complete exact missing Ceres labels for authorized58M factorial',selection={'source':str(source),'summary_sha256':sha(summary),'source_rows':row['rows'],'source_shards':len(specs),'shards':chosen,'rule':'Pinned audited retained row sequence'})
  p.pop('g10_qualification',None); p['environment']['CUDA_CACHE_PATH']=str(d/'cuda_cache')
  p['stop_paths']=[str(x/'STOP') for x in [OUT,d,state,d/'driver',Path(p['output'])]]
  real=sum(x['rows'] for x in chosen);padding=sum((-x['rows'])%512 for x in chosen)
  p['counts']={'real_rows':real,'padding_rows':padding,'calls':(real+padding)//512,'input_rows':real+padding};p['calls']=p['counts']['calls']
  argv=['--source',str(source),'--expected-source-summary-sha256',sha(summary),'--onnx',p['model']['path'],'--expected-onnx-sha256',p['model']['sha256'],'--out',p['output'],'--start-shard',str(start),'--max-shards',str(len(chosen)),'--wdl-output','value','--wdl-output-kind','logits','--batch-size','512','--threads','2','--gpu-mem-gb','16','--gpu-lock',p['gpu_lock'],'--minimum-free-gib','150','--max-output-gib','0.125','--max-seconds','300','--stop',str(state/'STOP'),'--pad-final-batch','--retain-value2','--audited-source-manifest',admit['path'],'--expected-audited-source-manifest-sha256',admit['sha256']]
  p['operations']=[{'collector_argv':argv,'counts':p['counts'],'selection':chosen}]
  p['small_pins']=[pin(x) for x in [summary,Path(admit['path']),OUT/'run_chunk.py',OUT/'wsl_mapped_libraries.py',Path(__file__),HOME/'coverage/coverage.json']]
  for file in R.glob('scripts/*.py'):p['small_pins'].append(pin(file))
  p['reviewed_collector_sha256']=sha(R/'scripts/ceres_derived_sidecar.py')
  put(state/'plan.json',p)
  cmd=['/usr/bin/env',*[k+'='+v for k,v in p['environment'].items()],'/usr/bin/nice','-n','19','/usr/bin/ionice','-c','3','/usr/bin/taskset','-c','2,3',p['python'],str(OUT/'run_chunk.py'),'--state',str(state),'--expected-plan-sha256',sha(state/'plan.json'),'--execute']
  chunks.append({'argv':cmd,'completion_mode':'ceres_invocations','expected_padding_rows':padding,'expected_rows':real,'id':state.name,'max_shards':len(chosen),'output_directory':p['output'],'start_shard':start,'timeout_seconds':300,'working_directory':str(R)})
 seconds=math.ceil(row['rows']/700)+1200
 plan={'schema':1,'pinned_files':{str(x):sha(x) for x in [OUT/'run_chunk.py',OUT/'wsl_mapped_libraries.py',summary,Path(admit['path'])]},'chunks':chunks,'minimum_free_gib':150,'overall_seconds':seconds,'pause_between_chunks_seconds':0,'state_directory':str(d/'driver')};put(d/'driver.plan.json',plan)
 command=['/usr/bin/python3',str(R/'scripts/ceres_collection_batches.py'),'--plan',str(d/'driver.plan.json'),'--expected-plan-sha256',sha(d/'driver.plan.json'),'--execute']
 # Driver success proves collected invocations; separate consumer qualification must follow.
 reg={'argv':command,'cwd':str(R),'env':{'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},'pins':[pin(d/'driver.plan.json'),pin(OUT/'run_chunk.py'),pin(R/'scripts/ceres_collection_batches.py')], 'completion':{'path':str(d/'driver/manifest.json'),'status_key':'status','expected':'COMPLETE'}}
 put(d/'registered_command.json',reg)
 blocks.append({'id':'factorial58_ceres_'+name,'rows':row['rows'],'shards':row['shards'],'source':str(source),'source_manifest':admit,'command_file':str(d/'registered_command.json'),'command_sha256':sha(d/'registered_command.json'),'max_seconds':seconds+60,'output_root':str(d/'bank'),'driver_plan':str(d/'driver.plan.json')})
put(OUT/'blocks.prepared.json',{'status':'PREPARED_PENDING_REVIEW_NOT_QUEUED','rows':sum(x['rows'] for x in blocks),'shards':sum(x['shards'] for x in blocks),'blocks':blocks,'estimated_seconds_at_1006rps':sum(x['rows'] for x in blocks)/1006,'consumer_qualification':'Required after collection; collector driver COMPLETE is not training admission.'})
print('prepared',len(blocks),'cohorts',sum(len(read(Path(x['driver_plan']))['chunks']) for x in blocks),'grouped invocations')
