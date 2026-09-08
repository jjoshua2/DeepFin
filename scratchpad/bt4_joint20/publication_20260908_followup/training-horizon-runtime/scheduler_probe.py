"""One disposable existing20-row fixture with short warmup to observe release cycles."""
import hashlib,json,os,pathlib,sys,time
from numcodecs import blosc
blosc.set_nthreads(2)
import torch,numpy as np,yaml
import pytest,zarr
from scripts import lc0_control_train as driver
from tests.test_offline_game_epochs import arguments
OUT=pathlib.Path(__file__).parent
FIX=OUT/'scheduler_fixture';FIX.mkdir(exist_ok=False)
argv=arguments(FIX,FIX/'run')
config=pathlib.Path(argv[argv.index('--config')+1]);raw=yaml.safe_load(config.read_text());raw['train']['warmup_steps']=2;config.write_text(yaml.safe_dump(raw))
torch.set_num_threads(2)
def digest(b):return hashlib.sha256(b).hexdigest()
def state(trainer,buf):
 return {'step':trainer.step,'torch_rng':digest(torch.get_rng_state().numpy().tobytes()),'augmentation_rng':digest(json.dumps(buf.rng.bit_generator.state,sort_keys=True).encode()),'scheduler':trainer._scheduler.state_dict(),'lrs':[p['lr'] for p in trainer.opt.param_groups]}
records=[];original=driver.Trainer.train_steps
identities=[]
def observed(self,buf,**kwargs):
 identities.append((id(self),id(self.opt),id(self._scheduler),id(buf.rng)))
 before=state(self,buf);result=original(self,buf,**kwargs);records.append({'seed':buf.plan.seed,'requested':kwargs['steps'],'before':before,'after':state(self,buf)})
 assert self._warmup_steps==2 and self._lr_schedule=='sqrt_release' and self._lr_release_cycle_steps==0
 return result
driver.Trainer.train_steps=observed
start=time.time()
try:assert driver.main(argv)==0
finally:driver.Trainer.train_steps=original
assert len(set(identities))==1
assert [x['before']['step'] for x in records]==[0,2,4,5,7,9]
assert [x['after']['step'] for x in records]==[2,4,5,7,9,10]
for left,right in zip(records,records[1:]):
 for field in ['step','torch_rng','augmentation_rng','scheduler','lrs']:assert left['after'][field]==right['before'][field],field
assert any(a['before']['torch_rng']!=a['after']['torch_rng'] for a in records)
assert records[2]['seed']==0 and records[3]['seed']==1
first=torch.load(FIX/'run/checkpoint_epoch1.pt',map_location='cpu',weights_only=False);last=torch.load(FIX/'run/checkpoint.pt',map_location='cpu',weights_only=False)
assert first['step']==5 and last['step']==10
summary=json.loads((FIX/'run/summary.json').read_text());assert summary['sampling']['complete'] and len(summary['sampling']['epochs'])==2
assert all(x['sampling']['complete'] and x['sampling']['plan_sha256']==x['sampling']['realized_sha256'] for x in summary['sampling']['epochs'])
assert not torch.cuda.is_initialized()
loaded={n:{'path':str(getattr(m,'__file__',''))} for n,m in sys.modules.items() if n.startswith('chess_anti_engine.') and str(getattr(m,'__file__','')).endswith('.so')}
for entry in loaded.values():entry['sha256']=digest(pathlib.Path(entry['path']).read_bytes())
result={'status':'PASS_CPU_ONLY','wall_seconds':time.time()-start,'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'numpy':np.__version__,'zarr':zarr.__version__,'torch_threads':torch.get_num_threads(),'blosc_threads':blosc.get_nthreads(),'affinity':sorted(os.sched_getaffinity(0)),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'cuda_initialized':torch.cuda.is_initialized(),'argv':argv,'config_sha256':digest(config.read_bytes()),'loaded_native':loaded,'records':records,'checkpoint_keys':list(last),'epochs':summary['sampling']['epochs'],'summary_sha256':digest((FIX/'run/summary.json').read_bytes()),'scope':'Actual20-row two-epoch CPU trajectory with2-step warmup to exercise sqrt-release; toy model/no compile. Not CUDA, full-corpus, checkpoint-resume or production-budget qualification.'}
(OUT/'scheduler_probe.json').write_text(json.dumps(result,indent=2)+'\n')
print('CPU_SCHEDULER_QUALIFICATION_PASS',result['wall_seconds'])
