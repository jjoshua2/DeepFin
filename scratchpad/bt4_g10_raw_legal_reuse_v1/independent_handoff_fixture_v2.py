from pathlib import Path
import tempfile,subprocess,fcntl,time,hashlib,json,re,importlib.util
base=Path('/home/josh/projects/chess/scratchpad/bt4_g10_raw_legal_reuse_v1');src=base/'driver.sh';helper=base/'consume_handoff.py';template=base/'handoff.request.template.json'
spec=importlib.util.spec_from_file_location('handoff',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
results=[];expected=hashlib.sha256(template.read_bytes()).hexdigest()
with tempfile.TemporaryDirectory(prefix='label-handoff-v2-fixture-') as td:
 root=Path(td)
 for case in ['matching','changed','replace_during_rename','newer_operator_pause','initial_symlink','racing_symlink','existing_archive']:
  d=root/case;d.mkdir();source=d/'pause.request';archive=d/'consumed';target=d/'external';target.write_text('external preserved')
  source.write_bytes(template.read_bytes() if case!='changed' else b'operator pause')
  if case=='initial_symlink':source.unlink();source.symlink_to(target)
  if case=='existing_archive':archive.mkdir();(archive/'request.json').write_text('prior archive')
  rename=Path.rename
  def intercepted(self,to):
   if self==source:
    if case in ['replace_during_rename','newer_operator_pause']:self.write_text('changed operator pause')
    if case=='racing_symlink':self.unlink();self.symlink_to(target)
    result=rename(self,to)
    if case=='newer_operator_pause':source.write_text('even newer pause')
    return result
   return rename(self,to)
  Path.rename=intercepted
  try:
   try:accepted=m.consume(source,archive,expected)
   except FileExistsError:
    assert case=='existing_archive';accepted=False
   assert accepted==(case=='matching'),case
   if case=='matching':assert not source.exists() and (archive/'request.json').read_bytes()==template.read_bytes()
   else:
    assert source.exists(),case
    if case=='newer_operator_pause':assert source.read_text()=='even newer pause' and (archive/'request.json').read_text()=='changed operator pause'
    if case in ['initial_symlink','racing_symlink']:assert source.is_symlink() and target.read_text()=='external preserved'
    if case=='existing_archive':assert (archive/'request.json').read_text()=='prior archive'
   results.append({'case':'helper_'+case,'PASS':True,'accepted':accepted})
  finally:Path.rename=rename
 for case in ['matching_boundary','changed_request','new_operator_pause','old_fail_proof']:
  d=root/case;d.mkdir();old=d/'old';ops=d/'new';old.mkdir();ops.mkdir();(d/'model').touch();(ops/'consume_handoff.py').write_bytes(helper.read_bytes())
  text=src.read_text();text=re.sub(r'^LIVE=.*$','LIVE='+str(d),text,flags=re.M)
  for key,val in [('WT',d),('OLD_OPS',old),('OPS',ops),('ONNX',d/'model')]:text=re.sub(r'^'+key+r'=.*$',key+'='+str(val),text,flags=re.M)
  text=re.sub(r'verify_code\(\) \{.*?\n\}','verify_code() {\n  true\n}',text,count=1,flags=re.S)
  text=re.sub(r'run_group\(\) \{.*?\n\}','run_group() {\n  touch "$OPS/work"\n  failed=0\n  exit 0\n}',text,count=1,flags=re.S)
  script=d/'driver.sh';script.write_text(text)
  with (old/'driver.lock').open('w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   p=subprocess.Popen(['/bin/bash',str(script)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   try:
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
     if (ops/'driver.log').exists() and 'waiting for previous' in (ops/'driver.log').read_text():break
     time.sleep(.02)
    assert p.poll() is None and not (ops/'work').exists(),case
    (old/'driver.paused').touch();(old/'pause.request').write_bytes(template.read_bytes() if case!='changed_request' else b'operator pause')
    if case=='new_operator_pause':(ops/'pause.request').write_text('operator pause')
    if case=='old_fail_proof':(old/'driver.fail').touch()
    fcntl.flock(lock,fcntl.LOCK_UN);rc=p.wait(timeout=4);work=(ops/'work').exists()
    if case=='matching_boundary':assert rc==0 and work and not (old/'pause.request').exists()
    else:assert not work and (old/'pause.request').exists(),(case,rc,work,(ops/'driver.log').read_text())
    results.append({'case':'shell_'+case,'PASS':True,'returncode':rc,'work_started':work,'lock_prevented_work_before_boundary':True})
   finally:
    if p.poll() is None:p.kill();p.wait()
print(json.dumps({'status':'PASS','driver_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),'checks':results,'limits':'Actual helper/filesystem/rename and shell/flock in disposable paths. Runtime verifier and labeling replaced with no-op/work marker only in shell fixture; no operational marker or GPU touched.'},indent=2))
