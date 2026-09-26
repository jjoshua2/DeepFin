import importlib.util,json,hashlib,subprocess,tempfile,sys,time,fcntl,os
from pathlib import Path
source=Path('/home/josh/projects/chess/scratchpad/bt4_joint20/B100_preparation_v1/materialization/run_materialization.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=[]
with tempfile.TemporaryDirectory(prefix='b100-supervisor-fixture-') as td:
 root=Path(td)
 for case in ['success','failure','stop','coordinator_death']:
  state=root/case;state.mkdir();base=state/'base';base.mkdir();out=state/'out'
  code="from pathlib import Path;import time;Path("+repr(str(state/'started'))+").touch();time.sleep(2)"
  if case=='success':code += ';Path('+repr(str(out))+').mkdir()'
  if case=='failure':code += ';raise SystemExit(7)'
  plan={'argv':['/usr/bin/timeout','--kill-after=1s','3s',sys.executable,'-c',code], 'pins':{},'cwd':'/tmp/deepfin-h20-recovery-tools','commit':'44aafa625f7143008625b5ed26087279ef296577','supervisor_sha256':sha(source),'output':str(out),'partial':str(out)+'.writing'}
  (state/'launch.json').write_text(json.dumps(plan))
  wrapper="import importlib.util,sys,types;from pathlib import Path;s=importlib.util.spec_from_file_location('fixture',"+repr(str(source))+");m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.STATE=Path("+repr(str(state))+");m.BASE=Path("+repr(str(base))+");m.shutil.disk_usage=lambda p:types.SimpleNamespace(free=999*1024**3);m.verify_publication=lambda p:None;m.sha=lambda p:'fake' if Path(p).name in ('bt4_policy_mix_summary.json','derive_targets_summary.json') else __import__('hashlib').sha256(Path(p).read_bytes()).hexdigest();sys.argv=['fixture',m.sha(m.STATE/'launch.json')];raise SystemExit(m.main())"
  with (state/'fixture.log').open('w') as log:
   p=subprocess.Popen([sys.executable,'-c',wrapper],stdout=log,stderr=log)
   try:
    deadline=time.monotonic()+5
    while not (state/'started').exists() and p.poll() is None and time.monotonic()<deadline:time.sleep(.03)
    assert (state/'started').exists(),(case,(state/'fixture.log').read_text())
    if case=='stop':(state/'STOP').touch()
    if case=='coordinator_death':
     p.kill();p.wait(timeout=3)
     with (base/'preparation.lock').open('a') as lock:
      try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
      except BlockingIOError: pass
      else:raise AssertionError('child failed to retain lock')
      time.sleep(4)
      fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
     checks.append({'case':case,'result':'PASS','evidence':'real timeout child retains lock after coordinator SIGKILL, then releases it'})
     continue
    rc=p.wait(timeout=8);status=json.loads((state/'status.json').read_text())
    assert (rc==0)==(case=='success'),(case,rc,status)
    assert status['status']==('COMPLETE' if case=='success' else 'FAILED_OR_STOPPED'),status
    with (base/'preparation.lock').open('a') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    checks.append({'case':case,'result':'PASS','status':status['status'],'returncode':status['returncode']})
   finally:
    if p.poll() is None:p.kill();p.wait()
 spec=importlib.util.spec_from_file_location('guard',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 q=root/'size';q.mkdir();(q/'file').write_bytes(b'x'*100)
 assert m.output_bytes({'output':str(q),'partial':str(q)+'.writing'})>0
 link=root/'alias';link.symlink_to(q,target_is_directory=True)
 try:m.output_bytes({'output':str(link),'partial':str(link)+'.writing'})
 except ValueError:pass
 else:raise AssertionError('symlink accepted')
 checks.append({'case':'output_sample_and_symlink_refusal','result':'PASS'})
receipt={'status':'PASS','script_sha256':sha(source),'cases':checks,'limits':'Tiny real owned subprocess/filesystem fixtures only; publication metadata and free-space observations substituted; no operational data or bulk launch.'}
out=source.parent/'supervisor_qualification.json'
with out.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
