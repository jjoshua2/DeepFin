"""Freeze preparation descriptor after the parent runtime and collection plans settle."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
RUNTIME=Path('/tmp/deepfin-factorial58-runtime')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return {'path':str(Path(p).resolve()),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text())
def write(p,m):Path(p).write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
coverage_path=ROOT/'coverage/coverage.json'; index_path=ROOT/'collection/blocks.prepared.json'
coverage,index=read(coverage_path),read(index_path)
blocks={b['source']:b for b in index['blocks']}
cohorts=[];pins=[ref(coverage_path),ref(index_path),ref(HERE/'run.py'),ref(ROOT/'coverage/assemble_collected.py')]
for i,c in enumerate(coverage['roots']):
 if c['ceres_candidates']:
  candidate=c['ceres_candidates'][0]; teacher={'path':candidate['manifest'],'sha256':candidate['manifest_sha256']}; pins.append(teacher)
 else:
  block=blocks[c['sf_source']]; teacher={'path':str(Path(block['driver_plan']).parent/'ceres_policy_manifest.json')}
 summary=ref(Path(c['v50_root'])/'derive_targets_summary.json'); pins.append(summary)
 cohorts.append({'index':i,'base':c['v50_root'],'base_summary':summary,'sf_source':c['sf_source'],
  'rows':c['rows'],'shards':c['shards'],'ceres_manifest':teacher,'output':str(ROOT/'outputs'/f'cohort{i:02d}')})
for b in index['blocks']:
 pins.extend([ref(b['driver_plan']),b['source_manifest']])
# Pin the offline producer's transitive target/collector implementations plus storage/consumer.
sys.path.insert(0,str(RUNTIME))
from scripts import ceres_value_mix
paths=set(ceres_value_mix.producer_pins())
paths.update(str(RUNTIME/p) for p in ['scripts/bootstrap_factorial_targets.py','scripts/target_overlay_storage.py',
 'chess_anti_engine/replay/target_overlay.py','chess_anti_engine/replay/target_overlay_v2.py',
 'chess_anti_engine/replay/shard.py','chess_anti_engine/replay/game_epoch.py'])
pins.extend(ref(p) for p in sorted(paths))
pins.append({'path':coverage['training_plan'],'sha256':coverage['training_plan_sha256']})
pins=list({p['path']:p for p in pins}.values())
plan={'schema':1,'runtime':str(RUNTIME),'python':'/usr/bin/python3','rows':coverage['rows'],'shards':coverage['shards'],
 'base_plan':{'path':coverage['training_plan'],'sha256':coverage['training_plan_sha256']},
 'cohorts':cohorts,'pins':pins,'assembler':str(ROOT/'coverage/assemble_collected.py'),
 'collected_complete':str(ROOT/'collection/manifests.complete.json'),
 'driver_completions':[str(Path(b['driver_plan']).parent/'driver/manifest.json') for b in index['blocks']],
 'disk_floor_gib':80,'available_memory_floor_gib':32,'process_memory_cap_gib':32,'max_seconds':57600}
write(HERE/'plan.json',plan)
descriptor={'argv':['/usr/bin/python3',str(HERE/'run.py'),'--plan',str(HERE/'plan.json'),
 '--sha256',sha(HERE/'plan.json'),'--execute'],'cwd':str(RUNTIME),'env':{},
 'pins':[ref(HERE/'plan.json'),*[p for p in pins if Path(p['path']).stat().st_size <= 8*1024**2]],
 'completion':{'path':str(HERE/'complete.json'),'status_key':'status','expected':'COMPLETE_FACTORIAL58_TARGETS'}}
write(HERE/'registered_command.json',descriptor)
write(HERE/'proposed_queue_item.json',{'id':'factorial58_target_preparation','kind':'registered_command',
 'command_file':str(HERE/'registered_command.json'),'command_sha256':sha(HERE/'registered_command.json'),
 'max_seconds':57720,'status':'queued','out':str(HERE/'scheduler_run')})
print(json.dumps({'plan':ref(HERE/'plan.json'),'descriptor':ref(HERE/'registered_command.json'),
 'cohorts':len(cohorts),'rows':plan['rows'],'shards':plan['shards']}))
