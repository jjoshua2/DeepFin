"""Symmetric receipt-bound paired arena between two fresh factorial arms."""
import argparse,fcntl,hashlib,json,math,os,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,'/tmp/deepfin-bootstrap-operator/scripts')
import bootstrap_experiment_operator as op

def require(ok,message):
 if not ok:raise RuntimeError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(2**20),b''):h.update(block)
 return h.hexdigest()
def reference(path):return {'path':str(path),'sha256':sha(path)}
def bind_one(donor):
 require(donor['plan_sha256'] is not None,'training plan must be frozen')
 require(sha(donor['plan'])==donor['plan_sha256'],'training plan pin differs')
 plan=op.load(Path(donor['plan']));require(plan['status']=='FROZEN_READY' and plan['arm']==donor['arm'],'training plan status/arm differs')
 terminal=op.load(Path(donor['terminal']));require(terminal['returncode']==0,'training supervisor did not succeed')
 receipt=op.load(Path(donor['complete']));require(receipt['status']=='PASS_FACTORIAL58_ARM' and receipt['arm']==donor['arm'] and receipt['plan_sha256']==donor['plan_sha256'],'training receipt lineage differs')
 checkpoint=receipt['checkpoint'];summary=receipt['summary'];initial=receipt['initial_state']
 require(checkpoint['path']==donor['checkpoint']==str(Path(plan['out'])/'checkpoint.pt'),'checkpoint namespace differs')
 require(summary['path']==str(Path(plan['out'])/'summary.json') and initial['path']==str(Path(plan['out'])/'initial_state.json'),'training artifacts namespace differs')
 for pin in [checkpoint,summary,initial]:require(sha(pin['path'])==pin['sha256'],'training artifact digest differs')
 facts=op.load(Path(summary['path']));sampling=facts['sampling']
 require(facts['seed']==121 and facts['batch_size']==512 and facts['steps_realized']==113459 and facts.get('continuation') is None,'requires fresh matched training')
 require(sampling['complete'] and sampling['rows_planned']==58090688 and sampling['rows_realized']==58090688 and sampling['batches_realized']==facts['steps_realized'],'exact epoch is incomplete')
 require(sampling['same_game_repeats_max']==0 and sampling['plan_sha256']==sampling['realized_sha256'],'donor schedule differs')
 require(facts['train_window_metrics'] and facts['train_window_metrics'][-1]['steps_cumulative']==113459,'donor training windows incomplete')
 require(all(isinstance(w.get(k),(int,float)) and math.isfinite(w[k]) for w in facts['train_window_metrics']+[facts['metrics']] for k in ['loss','policy_loss','wdl_loss']),'donor nonfinite loss')
 initial_facts=op.load(Path(initial['path']));require(initial_facts['seed']==121 and initial_facts['tensor_sha256']==receipt['tensor_sha256'],'initial tensor identity differs')
 return {'arm':donor['arm'],'checkpoint':checkpoint,'summary':summary,'initial_state':initial,'tensor_sha256':receipt['tensor_sha256'],'receipt':reference(donor['complete']),'terminal':reference(donor['terminal']),'plan':reference(donor['plan'])}
def bind(p):
 result={role:bind_one(p['donors'][role]) for role in ['candidate','reference']}
 require(result['candidate']['tensor_sha256']==result['reference']['tensor_sha256'],'donor initialization differs')
 require(result['candidate']['checkpoint']['path']!=result['reference']['checkpoint']['path'],'self comparison forbidden')
 return result

def main():
 require(not sys.flags.optimize,'optimized interpreter disallowed')
 parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--sha256',required=True);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
 require(sha(args.plan)==args.sha256,'arena plan pin differs');p=op.load(Path(args.plan));require(p['status']=='FROZEN_READY','freeze training dependencies before use')
 op.RUNTIME=Path(p['runtime'])
 require(subprocess.check_output(['git','-C',str(op.RUNTIME),'rev-parse','HEAD'],text=True).strip()==p['runtime_head'],'arena runtime HEAD differs')
 subprocess.run(['git','-C',str(op.RUNTIME),'diff','--exit-code','HEAD','--'],check=True)
 for pin in p['pins']:require(sha(pin['path'])==pin['sha256'],'arena code/input pin differs')
 bound=bind(p);out=Path(p['out']);require(not out.exists(),'fresh arena output required')
 cmd=p['command']
 for flag,role in [('--candidate','candidate'),('--reference','reference')]:
  require(cmd.count(flag)==1 and cmd[cmd.index(flag)+1]==bound[role]['checkpoint']['path'],'command donor differs')
 if not args.execute:print('PASS_BOUND_FACTORIAL_ARENA_NOT_EXECUTED');return 0
 out.mkdir();child=None;lease=None;started=time.monotonic();result={'status':'INCOMPLETE','started_unix':time.time(),'plan_sha256':args.sha256}
 def stop(sig,frame):raise InterruptedError(sig)
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 def guard():
  require(time.monotonic()-started<p['internal_seconds'],'arena deadline')
  require(op.mem_avail_gib()>=32 and op.disk_free_gib()>=150,'arena resource reserve')
  require(not (Path(args.plan).parent/'STOP').exists() and not (Path(args.plan).parent.parent/'STOP').exists() and not (Path(args.plan).parent.parent.parent/'STOP').exists(),'STOP')
 try:
  guard();lease=open(p['gpu_lock'],'a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);require(not op.gpu_apps(),'GPU already owned')
  op.dump(out/'bound_inputs.json',bound);op.dump(out/'command.json',{'command':cmd,'cwd':str(op.RUNTIME)})
  env=os.environ.copy();env.update(p['env'])
  for key in ['PYTHONOPTIMIZE','PYTHONHOME','LD_PRELOAD']:env.pop(key,None)
  with (out/'arena.log').open('x') as log:
   child=subprocess.Popen(cmd,cwd=op.RUNTIME,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   while child.poll() is None:
    guard()
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:pass
  require(child.returncode==0,'arena failed');score=op.harvest_results_jsonl(out/'arena.results.jsonl');require(score is not None,'missing arena result')
  op.validate_arena_bank({'games':256,'out':str(out)},score)
  for donor in bound.values():
   for key in ['checkpoint','summary','initial_state','receipt','terminal','plan']:
    pin=donor[key];require(sha(pin['path'])==pin['sha256'],'bound training artifact changed')
  result.update(status='PASS_FACTORIAL58_256_GAME_ARENA',result=score,bound=bound,bank_sha256=sha(out/'arena.games.jsonl'))
 except BaseException as exc:result['error']=repr(exc);raise
 finally:
  cleanup_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
  try:
   try:
    if child is not None:op.terminate_owned_group(child)
   finally:
    if lease is not None:lease.close()
   result.update(ended_unix=time.time(),elapsed_seconds=time.monotonic()-started);op.dump(Path(args.plan).parent/'complete.json',result)
  finally:signal.pthread_sigmask(signal.SIG_SETMASK,cleanup_mask)
 return 0
if __name__=='__main__':raise SystemExit(main())
