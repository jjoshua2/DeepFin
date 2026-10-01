"""Bind successful producer receipts to immutable continuation inputs."""
import hashlib,importlib.util,json,sys
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def ref(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}
def pinned(item):
 if sha(item['path'])!=item['sha256']:raise ValueError('input pin changed: '+item['path'])
 return read(item['path'])
def successful_donor(plan):
 terminal=read(plan['donor_terminal'])
 if terminal['returncode']!=0:raise ValueError('donor process failed')
 receipt=read(plan['donor_complete'])
 if receipt['status']!='PASS_V50_CONTINUED_EPOCHS2_4' or receipt['plan_sha256']!=plan['donor_plan_sha256']:raise ValueError('donor completion contract differs')
 summary_path=Path(plan['donor_checkpoint']).parent/'summary.json'
 if sha(summary_path)!=receipt['summary_sha256']:raise ValueError('donor summary changed')
 summary=read(summary_path);c=receipt['continuation']
 if c!=summary['continuation'] or not summary['sampling']['complete'] or c['step_end']!=plan['expected_donor_step']:raise ValueError('donor not final complete horizon')
 checkpoints=[x for x in receipt['checkpoints'] if x['path']==plan['donor_checkpoint']]
 if len(checkpoints)!=1:raise ValueError('final donor checkpoint missing')
 ck=checkpoints[0]
 if sha(ck['path'])!=ck['sha256']:raise ValueError('donor checkpoint hash differs')
 return {'checkpoint':ck,'step':c['step_end'],'receipt':ref(plan['donor_complete']),'terminal':ref(plan['donor_terminal']),'summary':ref(summary_path)}
def bind(plan):
 donor=successful_donor(plan);d=read(plan['dataset_complete'])
 if d['status']!='COMPLETE_V50_CORPUS_AND_SCHEDULE_NOT_TRAINING' or (d['rows'],d['shards'])!=(50548069,6185):raise ValueError('dataset incomplete or wrong size')
 if d['union']['path']!=plan['union_path'] or d['schedule']['path']!=plan['schedule_path']:raise ValueError('wrong dataset generation')
 union=pinned(d['union']);report=pinned(d['schedule']);prior=pinned(plan['prior_union'])
 if union['cohorts'][:21]!=prior['cohorts'] or len(union['cohorts'])!=31:raise ValueError('original corpus changed or union incomplete')
 if report['status']!='PASS_CORPUS_SET_PROSPECTIVE_NOT_TRAINING' or report['manifest_sha256']!=d['union']['sha256']:raise ValueError('prospective admission differs')
 if report['rows']!=50548069 or union['expected_rows']!=50548069 or union['expected_shards']!=6185:raise ValueError('union dimensions differ')
 spec=importlib.util.spec_from_file_location('schedule',plan['schedule_helper']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 sys.path.insert(0,str(Path(plan['schedule_helper']).parent.parent))
 from scripts import audited_source_admission,baseline_row_exclusions
 for dependency in (audited_source_admission,baseline_row_exclusions):
  if Path(dependency.__file__).resolve().parent!=Path(plan['schedule_helper']).parent:raise ValueError('admission import context differs')
 mapping=module.admit(union)
 if mapping!=report['mapping']:raise ValueError('recipe/source admission changed')
 roots=[str(module.root_path(c,'V50')) for c in union['cohorts']]
 if roots!=report['trainer_shards']['V50'] or roots!=plan['expected_roots'] or len(set(roots))!=31:raise ValueError('full ordered V50 roots differ')
 source_plan=report['arms']['V50']['physical_plan']
 if source_plan['rows_planned']!=50548069 or source_plan['batch_size']!=512:raise ValueError('prospective rows/batch differ')
 return {'donor':donor,'dataset_receipt':ref(plan['dataset_complete']),'union':d['union'],'schedule':d['schedule'],'roots':roots,'rows':50548069,'shards':6185,'sampling_seed':105,'prospective_seed':report['seed'],'prospective_batches':source_plan['batches_planned'],'scope':'Prospective101 proves corpus/recipe; actual105 sampler defines new order and realized optimizer budget. Full trainer history/value gates remain active.'}
