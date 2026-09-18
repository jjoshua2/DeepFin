"""Exclusive bounded checkpoint continuation; no inference on import or preflight."""
import argparse,fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time
import bind
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,'/tmp/deepfin-ceres-output-scan/scripts')
from bootstrap_experiment_operator import terminate_owned_group

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def dump(path,obj):
 with Path(path).open('x') as f:json.dump(obj,f,indent=2);f.write('\n')
def main():
 if sys.flags.optimize:raise RuntimeError("optimization disables runtime guards")
 a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--sha256',required=True);a.add_argument('--execute',action='store_true');args=a.parse_args();assert sha(args.plan)==args.sha256;p=json.loads(args.plan.read_text())
 for item in p['pins']:assert sha(item['path'])==item['sha256'],item['path']
 assert subprocess.check_output(['git','-C',p['runtime'],'rev-parse','HEAD'],text=True).strip()==p['runtime_head']
 subprocess.run(['git','-C',p['runtime'],'diff','--exit-code','HEAD','--'],check=True)
 assert not Path(p['out']).exists()
 if not args.execute:print('PASS_CPU_PREFLIGHT_NOT_EXECUTED');return 0
 def stop(sig,frame):raise InterruptedError(f'signal {sig}')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 start=time.monotonic();train_start=None;child=None;lease=None;receipt={'status':'INCOMPLETE','started_unix':time.time(),'plan_sha256':args.sha256}
 def guard():
  assert time.monotonic()-start<p['internal_seconds'],'whole time budget'
  if train_start is not None:assert time.monotonic()-train_start<p['training_seconds'],'training time budget'
  assert not (HERE/'STOP').exists(),'STOP'
  available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
  assert available>=32*2**30,'RAM reserve'
  assert shutil.disk_usage(HERE).free>=150*2**30,'disk reserve'
 try:
  while True:
   if (HERE/'STOP').exists():raise InterruptedError('STOP while waiting')
   if Path(p['dataset_failed']).exists():raise RuntimeError('dataset preparation failed')
   if Path(p['donor_terminal']).exists() and json.loads(Path(p['donor_terminal']).read_text())['returncode']!=0:raise RuntimeError('donor process failed')
   if time.monotonic()-start>=p['dependency_seconds']:raise TimeoutError('dependency budget exhausted')
   if Path(p['donor_terminal']).exists() and Path(p['donor_complete']).exists() and Path(p['dataset_complete']).exists():
    try:json.loads(Path(p['dataset_complete']).read_text());break
    except json.JSONDecodeError:pass
   time.sleep(30)
  guard();bound=bind.bind(p);dump(HERE/'bound_inputs.json',bound)
  preserved=HERE/'pre_expansion_checkpoint.pt';shutil.copy2(bound['donor']['checkpoint']['path'],preserved)
  assert sha(preserved)==bound['donor']['checkpoint']['sha256'],'preserved donor differs'
  command=p['command_prefix']+['--shards',*bound['roots'],'--resume-checkpoint',str(preserved),'--resume-checkpoint-sha256',bound['donor']['checkpoint']['sha256'],'--resume-step',str(bound['donor']['step'])]
  dump(HERE/'actual_command.json',{'command':command,'bound':bound,'preserved_checkpoint':str(preserved)})
  guard();assert shutil.disk_usage(HERE).free>=170*2**30,'startup disk reserve'
  train_start=time.monotonic()
  lease=open(p['gpu_lock'],'a')
  fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  dump(HERE/'started.json',receipt)
  with open(HERE/'training.log','xb') as log:
   child=subprocess.Popen(command,cwd=p['runtime'],env={**os.environ,**p['env']},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   receipt['pid']=child.pid
   while child.poll() is None:
    guard()
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:pass
   assert child.returncode==0,f'training returncode {child.returncode}'
  s=json.loads((Path(p['out'])/'summary.json').read_text());r=s['continuation'];assert r['sha256']==bound['donor']['checkpoint']['sha256'] and r['step_start']==bound['donor']['step'] and r['additional_epochs']==3
  assert r['step_end']==r['step_start']+s['steps_realized'] and s['sampling']['complete']
  assert s['sampling']['rows_realized']==151644207 and sum(e['sampling']['rows_planned'] for e in s['sampling']['epochs'])==151644207
  assert s['sampling']['epochs_completed']==3 and [e['sampling']['seed'] for e in s['sampling']['epochs']]==[106,107,108]
  assert s['seed']==106 and s['sampling']['batches_realized']==s['steps_realized']
  for pin in [bound['dataset_receipt'],bound['union'],bound['schedule'],bound['donor']['receipt'],bound['donor']['summary']]:assert sha(pin['path'])==pin['sha256'],'bound input changed'
  checkpoints=[]
  for name in ['checkpoint_epoch1.pt','checkpoint_epoch2.pt','checkpoint.pt']:
   f=Path(p['out'])/name;assert f.is_file();checkpoints.append({'path':str(f),'sha256':sha(f)})
  guard();receipt.update(status='PASS_EXPANDED50M_EPOCHS2_4',summary_sha256=sha(Path(p['out'])/'summary.json'),continuation=r,checkpoints=checkpoints,bound_inputs=bound,preserved_donor={'path':str(preserved),'sha256':sha(preserved)},actual_added_steps=s['steps_realized'])
  return 0
 except BaseException as e:
  receipt['error']=repr(e);raise
 finally:
  try:
   if child is not None:terminate_owned_group(child,grace=20)
  finally:
   if lease is not None:lease.close()
  receipt.update(ended_unix=time.time(),elapsed_seconds=time.monotonic()-start)
  dump(HERE/'complete.json',receipt)
if __name__=='__main__':raise SystemExit(main())
