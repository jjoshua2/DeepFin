import json,pathlib,hashlib,sys,subprocess,os,time
p=pathlib.Path('/home/josh/chess-artifacts/operations/factorial58-sffree-training-20260922')
sys.path.insert(0,'/tmp/deepfin-bootstrap-operator/scripts')
import bootstrap_experiment_operator as op
result={'status':'PASS_CPU_DESCRIPTOR_AND_REFUSAL_CHECKS','jobs':{},'checked_unix':time.time(),'no_gpu_launch':True,'queue_mutated':False}
for n in ['E','E_D','D122','E122','E122_D122']:
 item=json.loads((p/n/'queue_item.json').read_text());spec=op.registered_spec(item)
 proc=subprocess.run(spec['argv'][:-1],capture_output=True,text=True,env={**os.environ,**spec['env']})
 expected=0 if n=='E' else 1
 assert proc.returncode==expected,(n,proc.returncode,proc.stderr)
 if n!='E':assert 'FileNotFoundError' in proc.stderr and ('complete.json' in proc.stderr or 'parent_outer_terminal.json' in proc.stderr),(n,proc.stderr)
 for forbidden in ['started.json','actual_command.json','complete.json']:
  assert not (p/n/forbidden).exists()
 result['jobs'][n]={'queue_item':item,'plan_sha256':hashlib.sha256((p/n/'plan.json').read_bytes()).hexdigest(),'cpu_preflight_returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr,'expected_pending_donor_refusal':n!='E','registered_spec_pass':True}
pathlib.Path('/tmp/E-final-descriptors-preflight-20260922.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'])
