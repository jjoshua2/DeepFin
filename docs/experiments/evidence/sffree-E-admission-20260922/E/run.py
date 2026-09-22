"""One fresh, bounded factorial arm; target preparation must already be complete."""
import argparse,fcntl,hashlib,json,math,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from disk_pause import DiskPauseGuard
from admission import admit
HERE=Path(__file__).resolve().parent

def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def dump(path,value):
 with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
def ref(path):return {'path':str(path),'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text())
def initial(path):
 value=read(path);require(value['seed']==121,'initial seed differs')
 require(isinstance(value['tensor_sha256'],str) and len(value['tensor_sha256'])==64,'invalid initial tensor digest')
 return value

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--sha256',required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 require(not sys.flags.optimize,'optimized interpreter disables inherited guards')
 require(sha(args.plan)==args.sha256,'plan digest differs');p=read(args.plan)
 require(p['status']=='FROZEN_READY','runtime and code pins must be frozen before use')
 require(p['arm']=='E','E-only runner')
 for item in p['pins']:require(sha(item['path'])==item['sha256'],'pin changed: '+item['path'])
 require(subprocess.check_output(['git','-C',p['runtime'],'rev-parse','HEAD'],text=True).strip()==p['runtime_head'],'runtime HEAD changed')
 subprocess.run(['git','-C',p['runtime'],'diff','--exit-code','HEAD','--'],check=True)
 require(not Path(p['out']).exists(),'fresh arm output required')
 # Draft/future admission binds full E preparation and successful D,D_C,D_B.
 roots,qualification,pins,anchor_value=admit(p)
 anchor=Path(p['initial_anchor'])
 command=p['command_prefix']+['--shards',*roots]
 if qualification is not None:command+=['--overlay-storage-qualification',qualification['path'],'--expected-overlay-storage-qualification-sha256',qualification['sha256']]
 require('--resume-checkpoint' not in command,'factorial arms must start fresh')
 if not args.execute:print('PASS_BOUND_CPU_PREFLIGHT_NOT_EXECUTED');return 0
 sys.path.insert(0,p['operator_runtime']);from bootstrap_experiment_operator import terminate_owned_group
 def stop(sig,frame):raise InterruptedError(f'signal {sig}')
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 wall=time.monotonic();start=wall;train_start=None;child=None;lease=None;initial_ref=None
 receipt={'status':'INCOMPLETE','arm':p['arm'],'started_unix':time.time(),'plan_sha256':args.sha256,'bound_inputs':pins}
 def interruption():
  require(time.monotonic()-wall<p['internal_seconds']+p['pause_seconds'],'whole wall-time budget')
  require(not (HERE/'STOP').exists() and not (HERE.parent/'STOP').exists() and not (HERE.parent.parent/'STOP').exists(),'STOP')
  available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
  require(available>=32*2**30,'RAM reserve')
 disk_pause=DiskPauseGuard(HERE,budget=p['pause_seconds'],check_interrupt=interruption)
 def check_initial(required=False):
  nonlocal initial_ref
  path=Path(p['out'])/'initial_state.json'
  if initial_ref is not None:return
  if not path.exists():require(not required,'initial model receipt missing');return
  try:value=initial(path)
  except json.JSONDecodeError:
   require(not required,'initial model receipt incomplete');return
  if anchor_value is not None:require(value['tensor_sha256']==anchor_value['tensor_sha256'],'initial model tensors differ from A')
  initial_ref=ref(path);dump(HERE/'initial_state_verified.json',{'arm':p['arm'],'initial':initial_ref,'tensor_sha256':value['tensor_sha256'],'anchor':str(anchor),'verified_unix':time.time()})
 def guard():
  nonlocal start,train_start
  interruption();require(time.monotonic()-start<p['internal_seconds'],'active whole-job budget')
  if train_start is not None:require(time.monotonic()-train_start<p['training_seconds'],'active training budget')
  paused=disk_pause.check(child);start+=paused
  if train_start is not None:train_start+=paused
  check_initial()
 try:
  guard();require(shutil.disk_usage(HERE).free>=20*2**30,'startup disk reserve')
  dump(HERE/'actual_command.json',{'command':command,'bound_inputs':pins,'fresh_seed':121})
  lease=open(p['gpu_lock'],'a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
  train_start=time.monotonic();dump(HERE/'started.json',receipt)
  with open(HERE/'training.log','xb') as log:
   child=subprocess.Popen(command,cwd=p['runtime'],env={**os.environ,**p['env']},stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   receipt['pid']=child.pid
   while child.poll() is None:
    guard()
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:pass
   require(child.returncode==0,f'training exited {child.returncode}')
  check_initial(required=True);summary=read(Path(p['out'])/'summary.json');sampling=summary['sampling']
  require(summary['seed']==121 and summary['batch_size']==512 and summary['steps_realized']==113459,'realized training settings differ')
  require(summary.get('continuation') is None,'unexpected checkpoint continuation')
  require(sampling['complete'] and sampling['rows_planned']==58090688 and sampling['rows_realized']==58090688,'incomplete exact58M epoch')
  require(sampling['batches_realized']==summary['steps_realized'],'realized optimizer/batch count differs')
  require(sampling['same_game_repeats_max']==0 and sampling['plan_sha256']==sampling['realized_sha256'],'realized schedule differs')
  require(summary['train_window_metrics'] and summary['train_window_metrics'][-1]['steps_cumulative']==113459,'training windows incomplete')
  require(all(isinstance(w.get(k),(int,float)) and math.isfinite(w[k]) for w in summary['train_window_metrics']+[summary['metrics']] for k in ['loss','policy_loss','wdl_loss']),'nonfinite training loss')
  if qualification is not None:require(summary['overlay_storage_qualification']==qualification,'realized overlay qualification differs')
  for item in pins+[initial_ref]:require(sha(item['path'])==item['sha256'],'bound input changed')
  checkpoint=Path(p['out'])/'checkpoint.pt';require(checkpoint.is_file(),'final checkpoint missing');guard()
  receipt.update(status='PASS_FACTORIAL58_ARM',initial_state=initial_ref,tensor_sha256=initial(Path(initial_ref['path']))['tensor_sha256'],summary=ref(Path(p['out'])/'summary.json'),checkpoint=ref(checkpoint),rows=58090688,steps_realized=summary['steps_realized'],sampling=sampling)
  return 0
 except BaseException as exc:receipt['error']=repr(exc);raise
 finally:
  cleanup_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
  try:
   try:
    if child is not None:DiskPauseGuard.resume_owned_group(child);terminate_owned_group(child,grace=20)
   finally:
    if lease is not None:lease.close()
   receipt.update(ended_unix=time.time(),active_elapsed_seconds=time.monotonic()-start,wall_elapsed_seconds=time.monotonic()-wall,disk_pause_seconds=disk_pause.used)
   dump(HERE/'complete.json',receipt)
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,cleanup_mask)
if __name__=='__main__':raise SystemExit(main())
