import copy, hashlib, importlib.util,json,pathlib,sys,tempfile
root=pathlib.Path('/home/josh/chess-artifacts/operations/factorial58-sffree-training-20260922')
sys.path[:0]=[str(root/'D122'),'/tmp/deepfin-bootstrap-operator/scripts']
import admission as a
P=pathlib.Path
count=0
with tempfile.TemporaryDirectory() as tmp:
 t=P(tmp)
 def put(name,d):
  p=t/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d));return {'path':str(p),'sha256':a.sha(p)}
 q=put('Dqualified.json',{'roots':['Droot']})
 source={'command_prefix':['train','--out-dir','old','--seed','121'],'D_qualification':q}
 for k in ['runtime','runtime_head','env','gpu_lock','training_seconds','pause_seconds','internal_seconds']:source[k]=k
 sourcepin=put('source.json',source)
 dplan={'status':'FROZEN_READY','arm':'D122','out':str(t/'Dout')};dplanpin=put('Dplan.json',dplan)
 p={'status':'FROZEN_READY','arm':'D122','source_E121_plan':sourcepin,'command_prefix':['train','--out-dir',str(t/'Dout'),'--seed','122'],'out':str(t/'Dout'),'initial_anchor':str(t/'Dout/initial_state.json'),'expected_steps':113459,**{k:source[k] for k in ['runtime','runtime_head','env','gpu_lock','training_seconds','pause_seconds','internal_seconds']}}
 a.admit121=lambda source:(['Eroot'],{'path':'Equalification'},[],{})
 arenaout=t/'arena';arenaout.mkdir();bank=arenaout/'arena.games.jsonl'
 bank.write_text(''.join(json.dumps({'kind':'game','pair_id':i,'half':h,'a_is_white':not h,'opening_fen':str(i),'score_candidate':0.5})+'\n' for i in range(128) for h in range(2)))
 dummy=put('dummy.json',{})
 def donor(arm):return {'arm':arm,**{k:dummy for k in ['checkpoint','summary','initial_state','receipt','terminal','plan']}}
 b={'candidate':donor('E'),'reference':donor('D')};b['candidate']['plan']=sourcepin
 arena={'status':'PASS_FACTORIAL58_256_GAME_ARENA','bound':b,'bank_sha256':a.sha(bank),'result':{'games':256,'pairs':128,'truncated':False,'score':0.5}}
 ap=put('arena_plan.json',{'out':str(arenaout)});arena['plan_sha256']=ap['sha256'];cp=put('arena_complete.json',arena);tp=put('arena_terminal.json',{'returncode':0})
 p['prerequisite_arena']={'plan':ap,'complete':cp['path'],'terminal':tp['path']}
 assert a.admit(p)[0]==['Droot'];count+=1
 for change in [('seed',lambda p:p['command_prefix'].__setitem__(-1,'121')),('runtime',lambda p:p.__setitem__('runtime','other')),('steps',lambda p:p.__setitem__('expected_steps',1)),('status',lambda p:p.__setitem__('status','DRAFT'))]:
  bad=copy.deepcopy(p);change[1](bad)
  try:a.admit(bad)
  except RuntimeError:count+=1
  else:raise AssertionError(change[0])
 summary={'seed':122,'batch_size':512,'steps_realized':113459,'sampling':{'complete':True,'rows_planned':58090688,'rows_realized':58090688,'batches_realized':113459,'same_game_repeats_max':0,'plan_sha256':'a','realized_sha256':'a'}}
 sp=put('Dout/summary.json',summary);ip=put('Dout/initial_state.json',{'seed':122,'tensor_sha256':'c'*64});ck=put('Dout/checkpoint.pt',{})
 dc={'status':'PASS_FACTORIAL58_ARM','arm':'D122','plan_sha256':dplanpin['sha256'],'summary':sp,'initial_state':ip,'checkpoint':ck,'tensor_sha256':'c'*64}
 dcp=put('Dcomplete.json',dc);dtp=put('Dterminal.json',{'returncode':0})
 ep=copy.deepcopy(p);ep['arm']='E122';ep['out']=str(t/'Eout');ep['command_prefix'][2]=ep['out'];ep['paired_D122']={'arm':'D122','plan':dplanpin,'complete':dcp['path'],'terminal':dtp['path']}
 assert a.admit(ep)[0]==['Eroot'] and a.admit(ep)[3]['tensor_sha256']=='c'*64;count+=1
 for key,value in [('seed',121),('steps_realized',5)]:
  bad=copy.deepcopy(summary);bad[key]=value;dc['summary']=put('Dout/summary.json',bad);put('Dcomplete.json',dc)
  try:a.admit(ep)
  except RuntimeError:count+=1
  else:raise AssertionError(key)
 dc['summary']=put('Dout/summary.json',summary);put('Dcomplete.json',dc)
 for key,value in [('same_game_repeats_max',1),('realized_sha256','b'),('rows_realized',1)]:
  bad=copy.deepcopy(summary);bad['sampling'][key]=value;dc['summary']=put('Dout/summary.json',bad);put('Dcomplete.json',dc)
  try:a.admit(ep)
  except RuntimeError:count+=1
  else:raise AssertionError(key)
 for key,value in [('status','INCOMPLETE'),('plan_sha256','bad')]:
  bad=copy.deepcopy(arena);bad[key]=value;put('arena_complete.json',bad)
  try:a.admit(p)
  except RuntimeError:count+=1
  else:raise AssertionError(key)
 put('arena_complete.json',arena)
 put('arena_terminal.json',{'returncode':1})
 try:a.admit(p)
 except RuntimeError:count+=1
 else:raise AssertionError('terminal')
print(json.dumps({'status':'PASS','checks':count,'scope':'synthetic replica admission; existing E121 admission mocked, actual paired-bank validator and hash guards used'}))
